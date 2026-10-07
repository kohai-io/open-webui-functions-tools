"""Pasted input and OWUI Attach Webpage reuse chat-owned source snapshots."""

import base64
import copy
import hashlib
import json
import unittest

import test_document_reader as base


TEXT = "# Synthetic access pilot\r\n\r\nThe pilot lasts four weeks. Access requires approval from the security lead. Do not start until approval is recorded.\r\n"


class InputTests(unittest.IsolatedAsyncioTestCase):
    setUp = base.PipeExecutionTests.setUp
    file_record = base.PipeExecutionTests.file_record
    capture_render = base.PipeExecutionTests.capture_render
    metadata = base.PipeExecutionTests.metadata
    run_pipe = base.PipeExecutionTests.run_pipe
    embed_events = base.PipeExecutionTests.embed_events

    def input_metadata(self, kind="paste", text=TEXT):
        metadata = self.metadata()
        message = {"id": "input-message", "role": "user", "content": text, "files": []}
        metadata.update(user_message=message, user_message_id=message["id"])
        if kind in ("web", "text"):
            message["content"] = "Prepare this document"
            message["files"] = [
                {
                    "type": "text",
                    "status": "uploaded",
                    "name": "Attached page",
                    "collection_name": "untrusted-collection",
                    "file": {
                        "data": {"content": text},
                        "meta": {"name": "Attached page"},
                    },
                }
            ]
            if kind == "web":
                message["files"][0]["url"] = "https://example.test/pilot"
                message["files"][0]["file"]["meta"][
                    "source"
                ] = "https://example.test/pilot"
        return metadata

    async def prepare(self, kind="paste", text=TEXT, partial=False):
        metadata = self.input_metadata(kind, text)
        name, source, fid, _, _ = await self.pipe._load_input(
            metadata, self.user, self.pipe.valves, self.emitter
        )
        expected = self.reader.build_snapshot(
            source, name, fid, base.MODEL_ID, self.pipe.valves
        )
        batches = self.reader.make_batches(expected, self.pipe.valves)
        self.complete.side_effect = [
            (
                RuntimeError("failed")
                if partial
                else base.completion_response(base.batch_response(expected, b))
            )
            for b in batches
        ]
        output = await self.run_pipe(__metadata__=metadata, __files__=[])
        self.assertIn("Reader ready", output)
        self.snapshot = copy.deepcopy(self.rendered[-1])
        self.original = metadata["user_message"]
        self.batches = batches
        self.render_patch.stop()
        self.store = {
            "assistant-message": {
                "role": "assistant",
                "embeds": [self.reader.render_reader(self.snapshot)],
            },
            "input-message": self.original,
        }
        self.chat_message.side_effect = lambda chat, mid: self.store.get(mid)
        self.complete.reset_mock(side_effect=True)
        self.emitter.reset_mock()
        return self.snapshot

    async def followup(self, question=False):
        passage = next(p for p in self.snapshot["passages"] if not p["source_only"])
        ref = {
            "v": 1,
            "message": "assistant-message",
            "fingerprint": self.snapshot["fingerprint"],
            "passage": passage["id"],
        }
        encoded = (
            base64.urlsafe_b64encode(json.dumps(ref).encode()).decode().rstrip("=")
        )
        operation = "question" if question else "retry"
        text = f"# Reader {operation}\n\nWhat approval is required?\n\n[Source: saved Reader](/c/saved-reader-chat#document-reader-{operation}-v1={encoded})"
        metadata = self.metadata()
        metadata.update(
            message_id="new-reader",
            user_message_id="followup-message",
            user_message={"id": "followup-message", "role": "user", "content": text},
        )
        return await self.run_pipe(__metadata__=metadata, __files__=[])

    async def test_pasted_markdown_preserves_exact_source_and_origin(self):
        snapshot = await self.prepare()
        self.assertEqual(base.source_text(snapshot), TEXT)
        self.assertEqual(snapshot["filename"], "Synthetic access pilot.md")
        self.assertEqual(
            snapshot["source_origin"], {"kind": "paste", "message_id": "input-message"}
        )
        self.assertEqual(snapshot["file_id"], "")
        self.files.assert_not_awaited()

    async def test_short_plain_text_is_a_valid_document(self):
        snapshot = await self.prepare(text="Visitors must sign in.")
        self.assertEqual(base.source_text(snapshot), "Visitors must sign in.")

    async def test_web_attachment_uses_complete_embedded_text_not_collection_or_prompt(
        self,
    ):
        snapshot = await self.prepare("web")
        self.assertEqual(base.source_text(snapshot), TEXT)
        self.assertEqual(snapshot["source_origin"]["kind"], "web")
        self.assertEqual(snapshot["source_origin"]["url"], "https://example.test/pilot")
        self.assertNotIn("untrusted-collection", json.dumps(snapshot))
        self.files.assert_not_awaited()

    async def test_inline_text_attachment_needs_no_file_record(self):
        snapshot = await self.prepare("text")
        self.assertEqual(snapshot["source_origin"]["kind"], "text")
        self.files.assert_not_awaited()

    async def test_txt_file_uses_server_extraction_and_file_permissions(self):
        self.files.return_value.filename = "pasted-content.txt"
        name, source, fid, origin, _ = await self.pipe._load_input(
            self.metadata(), self.user, self.pipe.valves, self.emitter
        )
        self.assertEqual(fid, base.FILE_ID)
        self.assertEqual(origin, {"kind": "file"})
        self.assertEqual(source, base.CORPUS["brief"]["text"])
        self.files.return_value.user_id = "other-user"
        with self.assertRaises(self.reader.ReaderError):
            await self.pipe._load_input(
                self.metadata(), self.user, self.pipe.valves, self.emitter
            )

    async def test_regeneration_uses_stored_input_when_current_metadata_is_absent(self):
        metadata = self.input_metadata("web")
        self.chat_message.return_value = metadata.pop("user_message")
        name, source, fid, origin, _ = await self.pipe._load_input(
            metadata, self.user, self.pipe.valves, self.emitter
        )
        self.assertEqual(source, TEXT)
        self.assertEqual(origin["kind"], "web")
        self.assertEqual(fid, "")

    async def test_complete_paste_retry_reuses_results_without_file_lookup(self):
        await self.prepare()
        self.assertIn("Reader ready", await self.followup())
        self.complete.assert_not_awaited()
        self.files.assert_not_awaited()
        self.assertEqual(len(self.embed_events()), 1)

    async def test_missing_web_batches_can_be_retried_from_saved_attachment(self):
        await self.prepare("web", partial=True)
        self.complete.side_effect = [
            base.completion_response(base.batch_response(self.snapshot, b))
            for b in self.batches
        ]
        self.assertIn("Document Reader ready", await self.followup())
        self.assertEqual(self.complete.await_count, len(self.batches))
        self.files.assert_not_awaited()

    async def test_changed_pasted_source_rejects_retry(self):
        await self.prepare()
        self.original["content"] += "Updated policy."
        self.assertIn("changed", await self.followup())
        self.complete.assert_not_awaited()

    async def test_removed_original_web_message_refuses_retry(self):
        await self.prepare("web")
        self.store.pop("input-message")
        self.assertIn("unavailable", await self.followup())
        self.complete.assert_not_awaited()

    async def test_paste_question_cites_saved_units_with_message_provenance(self):
        await self.prepare()
        passage = next(p for p in self.snapshot["passages"] if not p["source_only"])
        unit = next(u for u in passage["units"] if u["text"].strip())
        self.complete.return_value = base.completion_response(
            {
                "status": "answered",
                "points": [{"text": "Approval is required.", "evidence": [unit["id"]]}],
            }
        )
        self.assertIn("Answer about this passage", await self.followup(question=True))
        citations = [
            c.args[0]
            for c in self.emitter.await_args_list
            if c.args[0]["type"] == "citation"
        ]
        self.assertEqual(citations[0]["data"]["document"], [unit["text"]])
        self.assertEqual(
            citations[0]["data"]["metadata"][0]["source_message_id"], "input-message"
        )
        self.assertNotIn("file_id", citations[0]["data"]["metadata"][0])
        self.assertEqual(self.embed_events(), [])
        self.files.assert_not_awaited()

    async def test_changed_web_text_answers_saved_edition_and_missing_message_refuses(
        self,
    ):
        await self.prepare("web")
        self.original["files"][0]["file"]["data"]["content"] = TEXT + "A new condition."
        self.complete.return_value = base.completion_response(
            {"status": "insufficient_context", "points": []}
        )
        self.assertIn("saved edition", await self.followup(question=True))
        self.store.pop("input-message")
        self.complete.reset_mock()
        self.assertIn("unavailable", await self.followup(question=True))
        self.complete.assert_not_awaited()

    async def test_failed_pending_or_empty_web_attachments_never_fall_back_to_prompt(
        self,
    ):
        for status in ("uploading", "error", "uploaded"):
            metadata = self.input_metadata(
                "web", text="" if status == "uploaded" else TEXT
            )
            metadata["user_message"]["files"][0]["status"] = status
            self.assertIn(
                "not supplied usable", await self.run_pipe(__metadata__=metadata)
            )
        self.complete.assert_not_awaited()

    async def test_bare_links_commands_and_oversized_pastes_make_no_calls(self):
        for text in (
            "https://example.test/a",
            "Prepare this document",
            "",
            "x" * 100001,
        ):
            await self.run_pipe(__metadata__=self.input_metadata(text=text))
        self.complete.assert_not_awaited()
        self.files.assert_not_awaited()

    async def test_multiple_and_unsupported_attachments_are_not_combined(self):
        metadata = self.input_metadata("web")
        metadata["user_message"]["files"].append({"type": "file", "id": base.FILE_ID})
        self.assertIn("exactly one", await self.run_pipe(__metadata__=metadata))
        metadata["user_message"]["files"] = [
            {
                "type": "web",
                "url": "https://example.test/a",
                "collection_name": "private",
            }
        ]
        self.assertIn("not supplied usable", await self.run_pipe(__metadata__=metadata))
        self.complete.assert_not_awaited()

    async def test_inline_inputs_still_require_chat_and_model_access(self):
        metadata = self.input_metadata("web")
        self.chat_owner.return_value = False
        await self.run_pipe(__metadata__=metadata)
        self.chat_owner.return_value = True
        self.model_access.side_effect = RuntimeError("denied")
        await self.run_pipe(__metadata__=metadata)
        self.complete.assert_not_awaited()
