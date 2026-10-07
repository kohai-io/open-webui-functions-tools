"""Passage questions use saved, authorised source; never regenerate a Reader."""

import asyncio
import base64
import copy
import hashlib
import json
import unittest

import test_document_reader as base


class QuestionTests(unittest.IsolatedAsyncioTestCase):
    setUp = base.PipeExecutionTests.setUp
    file_record = base.PipeExecutionTests.file_record
    capture_render = base.PipeExecutionTests.capture_render
    metadata = base.PipeExecutionTests.metadata
    expected = base.PipeExecutionTests.expected
    run_pipe = base.PipeExecutionTests.run_pipe
    embed_events = base.PipeExecutionTests.embed_events

    def question_setup(self):
        self.render_patch.stop()
        snapshot, _ = self.expected()
        snapshot.update(
            reader_chat_id=self.metadata()["chat_id"],
            reader_message_id="saved-reader",
            source_sha256=hashlib.sha256(
                base.source_text(snapshot).encode()
            ).hexdigest(),
        )
        self.snapshot = snapshot
        self.passage = next(p for p in snapshot["passages"] if not p["source_only"])
        self.ref = dict(
            v=1,
            message="saved-reader",
            fingerprint=snapshot["fingerprint"],
            passage=self.passage["id"],
        )
        self.stored = dict(
            role="assistant", embeds=[self.reader.render_reader(snapshot)]
        )
        self.chat_message.return_value = self.stored
        self.unit = next(u for u in self.passage["units"] if not u.get("excluded"))
        self.complete.return_value = base.completion_response(
            {
                "status": "answered",
                "points": [
                    {
                        "text": "Approval is required before acting.",
                        "evidence": [self.unit["id"]],
                    }
                ],
            }
        )

    def draft(self, ref=None, question="What conditions apply?"):
        encoded = (
            base64.urlsafe_b64encode(json.dumps(ref or self.ref).encode())
            .decode()
            .rstrip("=")
        )
        return f"# Reader question\n\n{question}\n\n[Source: synthetic brief](/c/{self.metadata()['chat_id']}#document-reader-question-v1={encoded})"

    async def ask(self, text=None):
        metadata = self.metadata()
        metadata["user_message"] = dict(
            id="question-message", role="user", content=text or self.draft()
        )
        return await self.run_pipe(
            __metadata__=metadata,
            body={
                "model": "document_reader",
                "messages": [{"role": "user", "content": text or self.draft()}],
            },
        )

    async def test_question_and_regeneration_use_saved_source_not_client_text(self):
        self.question_setup()
        for _ in range(2):
            output = await self.ask()
            self.assertIn("Approval is required", output)
            self.assertIn("[1]", output)
        self.assertEqual(self.complete.await_count, 2)
        self.assertEqual(self.embed_events(), [])
        events = [
            c.args[0]
            for c in self.emitter.await_args_list
            if c.args[0]["type"] == "citation"
        ]
        self.assertEqual(events[0]["data"]["document"], [self.unit["text"]])
        request, payload, _ = self.complete.await_args.args
        self.assertEqual(set(request.scope["state"]), {"token"})
        self.assertNotIn(self.reader.QUESTION_MARKER, payload["messages"][1]["content"])
        context_units = json.loads(payload["messages"][1]["content"])["source_context"][
            "units"
        ]
        self.assertIn(self.unit["text"], [unit["text"] for unit in context_units])

    async def test_foreign_missing_and_mismatched_snapshot_stop_before_model(self):
        self.question_setup()
        for mutation in (
            lambda: self.stored.update(role="user"),
            lambda: self.stored.update(embeds=[]),
            lambda: self.stored.update(embeds=["<script>untrusted</script>"]),
        ):
            mutation()
            await self.ask()
        self.assertEqual(self.complete.await_count, 0)
        self.assertEqual(self.embed_events(), [])

    async def test_revoked_file_and_model_access_stop_question(self):
        self.question_setup()
        self.files.return_value = self.file_record(user_id="someone-else")
        self.assertIn("no longer have access", await self.ask())
        self.assertEqual(self.complete.await_count, 0)
        self.files.return_value = self.file_record()
        self.model_access.side_effect = RuntimeError("secret")
        self.assertIn("do not have access", await self.ask())
        self.assertEqual(self.complete.await_count, 0)

    async def test_changed_extraction_still_answers_frozen_edition(self):
        self.question_setup()
        self.files.return_value = self.file_record(
            data={"content": "Replacement extraction", "status": "completed"}
        )
        output = await self.ask()
        self.assertIn("saved edition", output)
        self.assertNotIn(
            "Replacement extraction",
            self.complete.await_args.args[1]["messages"][1]["content"],
        )

    async def test_bad_citations_get_one_repair_and_no_unsafe_answer(self):
        self.question_setup()
        self.complete.return_value = base.completion_response(
            {
                "status": "answered",
                "points": [{"text": "An unsupported claim", "evidence": ["u999999"]}],
            }
        )
        output = await self.ask()
        self.assertEqual(self.complete.await_count, 2)
        self.assertIn("valid passage evidence", output)
        self.assertNotIn("unsupported claim", output)
        self.assertFalse(
            any(c.args[0]["type"] == "citation" for c in self.emitter.await_args_list)
        )

    async def test_question_respects_single_call_valve(self):
        self.question_setup()
        self.pipe.valves.MAX_MODEL_CALLS = 1
        self.complete.return_value = base.completion_response(
            {"status": "answered", "points": []}
        )
        self.assertIn("valid passage evidence", await self.ask())
        self.assertEqual(self.complete.await_count, 1)

    async def test_insufficient_context_makes_no_claim_or_citation(self):
        self.question_setup()
        self.complete.return_value = base.completion_response(
            {"status": "insufficient_context", "points": []}
        )
        self.assertIn("insufficient", await self.ask())
        self.assertEqual(self.complete.await_count, 1)
        self.assertEqual(self.embed_events(), [])

    async def test_transport_failure_and_cancellation_are_not_retried(self):
        self.question_setup()
        self.complete.side_effect = RuntimeError("private provider body")
        self.assertNotIn("private provider body", await self.ask())
        self.assertEqual(self.complete.await_count, 1)
        self.complete.reset_mock()
        self.complete.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.ask()
        self.assertEqual(self.complete.await_count, 1)

    async def test_malformed_and_foreign_links_never_fall_back_to_preparation(self):
        self.question_setup()
        for text in (
            self.draft().replace("/c/saved-reader-chat", "/c/foreign"),
            self.draft().replace("(/c/", "(https://evil.test/c/"),
            self.draft() + self.draft(),
            "# Reader question\n\nMissing reference",
            self.draft(question="x" * 2001),
        ):
            await self.ask(text)
        self.assertEqual(self.complete.await_count, 0)
        self.assertEqual(self.embed_events(), [])

    def test_snapshot_offsets_hash_and_duplicate_fields_are_validated(self):
        self.question_setup()
        changed = copy.deepcopy(self.snapshot)
        changed["passages"][0]["units"][0]["text"] += "forged"
        with self.assertRaises(self.reader.ReaderError):
            self.reader.question_snapshot(
                self.reader.render_reader(changed), self.ref, self.metadata()["chat_id"]
            )
        with self.assertRaises(self.reader.ReaderError):
            self.reader.parse_question(
                self.draft().replace(
                    "document-reader-question-v1=",
                    "document-reader-question-v1=notbase64",
                ),
                self.metadata()["chat_id"],
            )

    def test_context_excludes_furniture_and_never_silently_truncates_target(self):
        self.question_setup()
        context = self.reader.passage_context(self.snapshot, self.passage["id"])
        self.assertTrue(
            all(
                u["passage"]
                in {
                    p["id"]
                    for p in self.snapshot["passages"]
                    if p["section_id"] == self.passage["section_id"]
                }
                for u in context["units"]
            )
        )
        self.passage["units"][0]["text"] = "x" * 12001
        with self.assertRaises(self.reader.ReaderError):
            self.reader.passage_context(self.snapshot, self.passage["id"])
