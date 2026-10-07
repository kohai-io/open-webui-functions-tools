"""Bounded preparation, saved-result reuse and optional provider controls."""

import asyncio
import base64
import copy
import json
import unittest

import test_document_reader as base
import test_document_reader_questions as questions


class PreparationTests(unittest.IsolatedAsyncioTestCase):
    setUp = base.PipeExecutionTests.setUp
    file_record = base.PipeExecutionTests.file_record
    capture_render = base.PipeExecutionTests.capture_render
    metadata = base.PipeExecutionTests.metadata
    expected = base.PipeExecutionTests.expected
    run_pipe = base.PipeExecutionTests.run_pipe
    embed_events = base.PipeExecutionTests.embed_events

    async def prepare_partial(self):
        self.pipe.valves.MAX_BATCH_PASSAGES = 1
        snapshot, batches = self.expected()
        self.assertGreater(len(batches), 1)
        self.complete.side_effect = [
            (
                base.completion_response(base.batch_response(snapshot, b))
                if i
                else RuntimeError("provider unavailable")
            )
            for i, b in enumerate(batches)
        ]
        await self.run_pipe()
        self.saved = copy.deepcopy(self.rendered[-1])
        self.assertEqual(self.saved["status"], "partial")
        self.render_patch.stop()
        self.store_saved()
        self.complete.reset_mock(side_effect=True)
        self.emitter.reset_mock()
        self.complete.return_value = base.completion_response(
            base.batch_response(snapshot, batches[0])
        )
        return snapshot, batches

    def store_saved(self):
        self.chat_message.return_value = {
            "role": "assistant",
            "embeds": [self.reader.render_reader(self.saved)],
        }

    def retry_draft(self):
        ref = dict(
            v=1,
            message=self.saved["reader_message_id"],
            fingerprint=self.saved["fingerprint"],
            passage=self.saved["passages"][0]["id"],
        )
        encoded = (
            base64.urlsafe_b64encode(json.dumps(ref).encode()).decode().rstrip("=")
        )
        return f"# Reader retry\n\nRetry missing sections.\n\n[Source: saved Reader](/c/{self.saved['reader_chat_id']}#document-reader-retry-v1={encoded})"

    async def retry(self, text=None):
        metadata = self.metadata()
        metadata["message_id"] = "new-reader"
        metadata["user_message"] = {
            "role": "user",
            "content": text or self.retry_draft(),
        }
        return await self.run_pipe(__metadata__=metadata, __files__=[])

    def emitted_snapshot(self):
        html = self.embed_events()[-1]["data"]["embeds"][0]
        ref = dict(
            message="new-reader",
            fingerprint=self.saved["fingerprint"],
            passage=self.saved["passages"][0]["id"],
        )
        return self.reader.question_snapshot(html, ref, self.saved["reader_chat_id"])

    async def test_retry_reuses_successes_and_revalidates_source_with_no_attachment(
        self,
    ):
        snapshot, batches = await self.prepare_partial()
        before = copy.deepcopy(self.saved)
        self.assertIn("Document Reader ready", await self.retry())
        final = self.emitted_snapshot()
        self.assertEqual(final["generation"]["calls"], 1)
        self.assertEqual(final["generation"]["reused_batches"], len(batches) - 1)
        self.assertEqual(
            [b["id"] for b in final["overviews"]], [b["id"] for b in batches]
        )
        self.assertEqual(base.source_text(final), base.source_text(snapshot))
        self.assertEqual(self.saved, before)
        self.complete.assert_awaited_once()
        self.files.assert_awaited()

    async def test_retry_rejects_changed_source_model_and_generation_settings(self):
        await self.prepare_partial()
        for change in ("source", "settings", "model"):
            with self.subTest(change=change):
                file = copy.deepcopy(self.files.return_value)
                models = copy.deepcopy(self.models)
                budget = self.pipe.valves.MAX_OUTPUT_TOKENS
                if change == "source":
                    self.files.return_value.data[
                        "content"
                    ] += "\nAn updated requirement."
                elif change == "settings":
                    self.pipe.valves.MAX_OUTPUT_TOKENS += 1
                else:
                    self.models[base.MODEL_ID]["info"]["params"]["temperature"] = 0.2
                self.assertIn("changed", await self.retry())
                self.complete.assert_not_awaited()
                self.assertEqual(self.embed_events(), [])
                self.files.return_value = file
                self.models.clear()
                self.models.update(models)
                self.pipe.valves.MAX_OUTPUT_TOKENS = budget

    async def test_retry_checks_current_file_permission_and_chat_ownership(self):
        await self.prepare_partial()
        self.files.return_value.user_id = "another-user"
        await self.retry()
        self.complete.assert_not_awaited()
        self.assertEqual(self.embed_events(), [])
        self.files.return_value.user_id = base.USER_ID
        self.chat_owner.return_value = False
        self.assertIn("own", await self.retry())
        self.complete.assert_not_awaited()

    async def test_foreign_chat_malformed_and_missing_references_never_prepare(self):
        await self.prepare_partial()
        for text in (
            self.retry_draft().replace("/c/saved-reader-chat", "/c/other-chat"),
            "# Reader retry\nMissing link",
            self.retry_draft().replace(
                "#document-reader-retry-v1=", "#document-reader-retry-v1=@@"
            ),
        ):
            await self.retry(text)
        self.chat_message.return_value = None
        await self.retry()
        self.complete.assert_not_awaited()
        self.assertEqual(self.embed_events(), [])

    async def test_invalid_cached_evidence_is_regenerated_not_reused(self):
        snapshot, batches = await self.prepare_partial()
        cached = self.saved["batch_results"][batches[1]["id"]]
        cached["passages"][0]["takeaways"][0]["evidence"] = ["u999999"]
        self.store_saved()
        self.complete.side_effect = [
            base.completion_response(base.batch_response(snapshot, b))
            for b in batches[:2]
        ]
        await self.retry()
        self.assertEqual(self.complete.await_count, 2)
        self.assertEqual(
            self.emitted_snapshot()["generation"]["reused_batches"], len(batches) - 2
        )

    async def test_retries_can_use_a_smaller_call_budget_and_two_workers(self):
        _, batches = await self.prepare_partial()
        self.pipe.valves.MAX_MODEL_CALLS = 1
        self.pipe.valves.CONCURRENT_BATCHES = 2
        await self.retry()
        final = self.emitted_snapshot()
        self.assertEqual(final["status"], "complete")
        self.assertEqual(final["generation"]["reused_batches"], len(batches) - 1)
        self.complete.assert_awaited_once()

    async def test_complete_saved_reader_requires_zero_new_calls(self):
        await self.prepare_partial()
        await self.retry()
        self.saved = self.emitted_snapshot()
        self.store_saved()
        self.emitter.reset_mock()
        self.complete.reset_mock()
        await self.retry()
        self.complete.assert_not_awaited()
        self.assertEqual(self.emitted_snapshot()["status"], "complete")

    async def test_parallel_results_keep_source_order_and_share_repair_budget(self):
        self.pipe.valves.MAX_BATCH_PASSAGES = 1
        self.pipe.valves.CONCURRENT_BATCHES = 2
        snapshot, batches = self.expected()
        active, peak, finished = 0, 0, []
        responses = iter(enumerate(batches))

        async def complete(*args):
            nonlocal active, peak
            index, batch = next(responses)
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.03 if index == 0 else 0.001)
                finished.append(index)
                return base.completion_response(base.batch_response(snapshot, batch))
            finally:
                active -= 1

        self.complete.side_effect = complete
        await self.run_pipe()
        final = self.rendered[-1]
        self.assertEqual(peak, 2)
        self.assertNotEqual(finished[0], 0)
        self.assertEqual(
            [b["id"] for b in final["overviews"]], [b["id"] for b in batches]
        )
        self.complete.reset_mock(side_effect=True)
        self.complete.return_value = base.completion_response("invalid JSON")
        self.pipe.valves.MAX_MODEL_CALLS = len(batches)
        await self.run_pipe()
        self.assertEqual(self.complete.await_count, len(batches))
        self.assertEqual(self.rendered[-1]["generation"]["calls"], len(batches))

    async def test_parallel_cancellation_awaits_all_requests(self):
        self.pipe.valves.MAX_BATCH_PASSAGES = 1
        self.pipe.valves.CONCURRENT_BATCHES = 2
        started, stopped = [], []
        ready = asyncio.Event()

        async def complete(*args):
            started.append(True)
            if len(started) == 2:
                ready.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.append(True)

        self.complete.side_effect = complete
        task = asyncio.create_task(self.run_pipe())
        await asyncio.wait_for(ready.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(len(stopped), 2)
        self.assertEqual(self.complete.await_count, 2)
        self.assertEqual(self.embed_events(), [])

    async def test_schema_and_effort_are_optional_and_operation_specific(self):
        self.complete.return_value = base.completion_response("{}")
        await self.pipe._complete(
            self.request, self.user, base.MODEL_ID, [], self.pipe.valves
        )
        payload = self.complete.await_args.args[1]
        self.assertNotIn("reasoning_effort", payload)
        self.assertNotIn("response_format", payload)
        self.pipe.valves.USE_JSON_MODE = True
        self.pipe.valves.USE_JSON_SCHEMA = True
        self.pipe.valves.PREPARATION_REASONING_EFFORT = "minimal"
        self.pipe.valves.QUESTION_REASONING_EFFORT = "low"
        for question in (False, True):
            await self.pipe._complete(
                self.request,
                self.user,
                base.MODEL_ID,
                [],
                self.pipe.valves,
                question=question,
            )
            payload = self.complete.await_args.args[1]
            self.assertEqual(
                payload["reasoning_effort"], "low" if question else "minimal"
            )
            fmt = payload["response_format"]
            self.assertEqual(fmt["type"], "json_schema")
            schema = fmt["json_schema"]["schema"]
            self.assertEqual(set(schema["required"]), set(schema["properties"]))
            for node in schema["$defs"].values():
                self.assertFalse(node["additionalProperties"])
                self.assertEqual(set(node["required"]), set(node["properties"]))


class SavedPdfQuestionTests(unittest.IsolatedAsyncioTestCase):
    setUp = questions.QuestionTests.setUp
    file_record = questions.QuestionTests.file_record
    capture_render = questions.QuestionTests.capture_render
    metadata = questions.QuestionTests.metadata
    expected = questions.QuestionTests.expected
    run_pipe = questions.QuestionTests.run_pipe
    embed_events = questions.QuestionTests.embed_events
    question_setup = questions.QuestionTests.question_setup
    draft = questions.QuestionTests.draft
    ask = questions.QuestionTests.ask

    async def test_saved_footer_labels_are_accepted_but_not_sent_as_evidence(self):
        self.question_setup()
        unit = self.snapshot["passages"][-1]["units"][-1]
        unit["excluded"] = "Repeated page footer"
        self.stored["embeds"] = [self.reader.render_reader(self.snapshot)]
        self.assertIn("Answer about this passage", await self.ask())
        payload = self.complete.await_args.args[1]
        self.assertNotIn(unit["id"], payload["messages"][1]["content"])

    async def test_unknown_exclusion_labels_are_rejected(self):
        self.question_setup()
        self.snapshot["passages"][-1]["units"][-1]["excluded"] = "Forged label"
        self.stored["embeds"] = [self.reader.render_reader(self.snapshot)]
        await self.ask()
        self.complete.assert_not_awaited()
