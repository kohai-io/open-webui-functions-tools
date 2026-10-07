"""Notes authorization and deterministic, saved-source chat briefs."""

import base64
import copy
import json
import sys
import unittest
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

import test_document_reader as base
import test_document_reader_inputs as inputs


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    file_record = base.PipeExecutionTests.file_record
    capture_render = base.PipeExecutionTests.capture_render
    metadata = base.PipeExecutionTests.metadata
    run_pipe = base.PipeExecutionTests.run_pipe
    embed_events = base.PipeExecutionTests.embed_events
    prepare = inputs.InputTests.prepare
    followup = inputs.InputTests.followup

    def setUp(self):
        base.PipeExecutionTests.setUp(self)
        self.note = NS(
            id="note-1",
            user_id=self.user.id,
            title="Pilot note",
            data={"content": {"md": inputs.TEXT}},
        )
        self.notes = AsyncMock(return_value=self.note)
        self.note_access = AsyncMock(return_value=False)
        self.note_feature = AsyncMock(return_value=True)
        modules = {}
        for name, key, value in (
            ("notes", "Notes", NS(get_note_by_id=self.notes)),
            ("access_grants", "AccessGrants", NS(has_access=self.note_access)),
            ("config", "Config", NS(get=AsyncMock(return_value={}))),
        ):
            module = ModuleType("open_webui.models." + name)
            setattr(module, key, value)
            modules[module.__name__] = module
        context = patch.dict(sys.modules, modules)
        context.start()
        self.addCleanup(context.stop)
        self.modules["open_webui.utils.access_control"].has_permission = (
            self.note_feature
        )

    def input_metadata(self, kind="paste", text=inputs.TEXT):
        metadata = inputs.InputTests.input_metadata(self, kind, text)
        if kind == "note":
            metadata["user_message"].update(
                content="Prepare this document",
                files=[
                    {
                        "type": "note",
                        "id": "note-1",
                        "name": "Untrusted title",
                        "content": "untrusted body",
                    }
                ],
            )
        return metadata

    def brief_text(self, changes=None, chat="saved-reader-chat"):
        ids = [p["id"] for p in self.snapshot["passages"] if not p["source_only"]]
        ref = {
            "v": 1,
            "message": "assistant-message",
            "fingerprint": self.snapshot["fingerprint"],
            "passages": ids,
            "explanations": False,
        }
        ref.update(changes or {})
        encoded = (
            base64.urlsafe_b64encode(json.dumps(ref).encode()).decode().rstrip("=")
        )
        return f"# Reader brief\n\nSend selected points.\n\n[Source: saved Reader](/c/{chat}#document-reader-brief-v1={encoded})"

    async def publish(self, changes=None, chat="saved-reader-chat"):
        metadata = self.metadata()
        metadata["user_message"] = {
            "role": "user",
            "content": self.brief_text(changes, chat),
        }
        return await self.run_pipe(__metadata__=metadata, __files__=[])

    async def test_note_uses_authorized_server_markdown_not_attachment_body(self):
        snapshot = await self.prepare("note")
        self.assertEqual(base.source_text(snapshot), inputs.TEXT)
        self.assertEqual(snapshot["filename"], "Pilot note.md")
        self.assertEqual(
            snapshot["source_origin"], {"kind": "note", "note_id": "note-1"}
        )
        self.files.assert_not_awaited()
        self.note_feature.assert_awaited_with(self.user.id, "features.notes", {})

    async def test_note_owner_still_needs_notes_feature(self):
        self.note_feature.return_value = False
        output = await self.run_pipe(__metadata__=self.input_metadata("note"))
        self.assertIn("access to OWUI Notes", output)
        self.notes.assert_not_awaited()
        self.complete.assert_not_awaited()

    async def test_shared_note_requires_existing_read_grant(self):
        self.note.user_id = "someone-else"
        output = await self.run_pipe(__metadata__=self.input_metadata("note"))
        self.assertIn("no longer have access", output)
        self.complete.assert_not_awaited()
        self.note_access.return_value = True
        await self.prepare("note")
        self.note_access.assert_awaited_with(
            user_id=self.user.id,
            resource_type="note",
            resource_id="note-1",
            permission="read",
        )

    async def test_admin_note_access_matches_owui_route(self):
        self.user.role = "admin"
        self.note.user_id = "someone-else"
        self.note_feature.return_value = False
        await self.prepare("note")
        self.note_feature.assert_not_awaited()
        self.note_access.assert_not_awaited()

    async def test_missing_empty_and_mixed_notes_fail_before_generation(self):
        self.notes.return_value = None
        self.assertIn(
            "unavailable", await self.run_pipe(__metadata__=self.input_metadata("note"))
        )
        self.notes.return_value = self.note
        self.note.data = {"content": {"md": ""}}
        self.assertIn(
            "no readable Markdown",
            await self.run_pipe(__metadata__=self.input_metadata("note")),
        )
        metadata = self.input_metadata("note")
        metadata["user_message"]["files"].append({"type": "file", "id": base.FILE_ID})
        self.assertIn("exactly one", await self.run_pipe(__metadata__=metadata))
        self.complete.assert_not_awaited()

    async def test_note_retry_reuses_then_rejects_changed_source(self):
        await self.prepare("note")
        self.assertIn("Reader ready", await self.followup())
        self.complete.assert_not_awaited()
        self.note.data["content"]["md"] += "An amendment."
        self.assertIn("changed", await self.followup())
        self.complete.assert_not_awaited()

    async def test_revoked_note_access_stops_questions_and_briefs(self):
        await self.prepare("note")
        self.note.user_id = "someone-else"
        self.assertIn("no longer have access", await self.followup(question=True))
        self.assertIn("no longer have access", await self.publish())
        self.complete.assert_not_awaited()
        self.assertFalse(
            any(c.args[0]["type"] == "citation" for c in self.emitter.await_args_list)
        )

    async def test_note_question_uses_saved_edition_and_note_citation(self):
        await self.prepare("note")
        passage = next(p for p in self.snapshot["passages"] if not p["source_only"])
        unit = next(u for u in passage["units"] if u["text"].strip())
        self.note.data["content"]["md"] += "Changed after preparation."
        self.complete.return_value = base.completion_response(
            {
                "status": "answered",
                "points": [{"text": "Approval required.", "evidence": [unit["id"]]}],
            }
        )
        self.assertIn("saved edition", await self.followup(question=True))
        event = next(
            c.args[0]
            for c in self.emitter.await_args_list
            if c.args[0]["type"] == "citation"
        )
        self.assertEqual(event["data"]["metadata"][0]["note_id"], "note-1")
        self.assertEqual(event["data"]["document"], [unit["text"]])

    async def test_brief_is_deterministic_and_needs_no_model(self):
        await self.prepare()
        self.model_access.side_effect = RuntimeError("model removed")
        result = await self.publish()
        self.assertIn("## Reading brief:", result)
        self.assertIn("[1]", result)
        self.complete.assert_not_awaited()
        self.assertEqual(self.embed_events(), [])
        event = next(
            c.args[0]
            for c in self.emitter.await_args_list
            if c.args[0]["type"] == "citation"
        )
        self.assertEqual(
            event["data"]["metadata"][0]["source_message_id"], "input-message"
        )
        all_source = {u["text"] for p in self.snapshot["passages"] for u in p["units"]}
        self.assertIn(event["data"]["document"][0], all_source)

    async def test_brief_rejects_tampered_cross_chat_and_unknown_selections(self):
        await self.prepare()
        for changes in (
            {"passages": []},
            {"passages": ["p0001"] * 51},
            {"passages": ["p0001", "p0001"]},
            {"passages": ["p999999"]},
            {"explanations": "true"},
            {"content": "invented brief"},
            {"fingerprint": "a" * 64},
        ):
            self.assertNotIn("## Reading brief:", await self.publish(changes))
        self.assertNotIn("## Reading brief:", await self.publish(chat="other-chat"))
        self.assertFalse(
            any(c.args[0]["type"] == "citation" for c in self.emitter.await_args_list)
        )
        self.complete.assert_not_awaited()

    async def test_brief_requires_owned_chat_and_original_input(self):
        await self.prepare()
        self.chat_owner.return_value = False
        self.assertIn("own", await self.publish())
        self.chat_owner.return_value = True
        self.store.pop("input-message")
        self.assertIn("unavailable", await self.publish())
        self.complete.assert_not_awaited()

    async def test_brief_warns_about_changed_note_and_can_include_explanations(self):
        await self.prepare("note")
        self.note.data["content"]["md"] += "Changed."
        result = await self.publish({"explanations": True})
        self.assertIn("Explanation:", result)
        self.assertIn("source has changed", result)
        self.complete.assert_not_awaited()

    async def test_brief_validates_saved_evidence_before_emitting_citations(self):
        await self.prepare()
        passage = next(p for p in self.snapshot["passages"] if not p["source_only"])
        passage["generated"]["takeaways"][0]["evidence"] = ["u999999"]
        self.store["assistant-message"]["embeds"] = [
            self.reader.render_reader(self.snapshot)
        ]
        self.assertIn("invalid evidence", await self.publish())
        self.assertFalse(
            any(c.args[0]["type"] == "citation" for c in self.emitter.await_args_list)
        )

    async def test_outer_owui_sources_do_not_shift_reader_citations(self):
        await self.prepare("note")
        metadata = self.metadata()
        # OWUI groups repeated chunks by metadata.source, not source object count.
        metadata["sources"] = [
            {
                "document": ["chunk one", "chunk two"],
                "metadata": [{"source": "note-1"}, {"source": "note-1"}],
                "source": {"id": "note-1"},
            },
            {"document": ["another source"], "source": {"id": "another"}},
        ]
        metadata["user_message"] = {"role": "user", "content": self.brief_text()}
        result = await self.run_pipe(__metadata__=metadata)
        self.assertIn("[3]", result)
        self.assertNotIn("[1]", result)
        event = next(
            c.args[0]
            for c in self.emitter.await_args_list
            if c.args[0]["type"] == "citation"
        )
        self.assertTrue(
            event["data"]["metadata"][0]["source"].startswith(
                "document-reader:assistant-message:"
            )
        )
        self.complete.assert_not_awaited()
        original_metadata = self.metadata
        self.metadata = lambda: {**original_metadata(), "sources": metadata["sources"]}
        passage = next(p for p in self.snapshot["passages"] if not p["source_only"])
        uid = next(u["id"] for u in passage["units"] if u["text"].strip())
        self.complete.return_value = base.completion_response(
            {
                "status": "answered",
                "points": [{"text": "A saved point.", "evidence": [uid]}],
            }
        )
        answer = await self.followup(question=True)
        self.assertIn("[3]", answer)
        self.assertNotIn("[1]", answer)

    async def test_source_only_brief_is_inert_and_size_is_bounded(self):
        await self.prepare()
        passage = next(p for p in self.snapshot["passages"] if not p["source_only"])
        passage.pop("generated")
        ref = {"passages": [passage["id"]], "explanations": False}
        result, citations = self.reader.chat_brief(self.snapshot, ref)
        self.assertIn("**Source only**", result)
        self.assertIn("```text", result)
        passage["units"][0]["text"] = "x" * 60001
        with self.assertRaises(self.reader.ReaderError):
            self.reader.chat_brief(self.snapshot, ref)
