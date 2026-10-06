"""Offline generation diagnostics: safe failure reasons and no implicit resubmission."""

import asyncio
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch

from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse


PRIVATE_DETAIL = "private-source-token-and-provider-detail"


def completion(content, finish_reason="stop", **message_fields):
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"content": content, **message_fields},
            }
        ]
    }


class GenerationDiagnosticTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "document_reader_diagnostics_under_test",
            Path(__file__).parents[1] / "document_reader.py",
        )
        self.reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.reader)
        self.pipe = self.reader.Pipe()
        self.valves = self.reader.Pipe.Valves(BASE_MODEL_ID="fixture-model")
        self.complete = AsyncMock()
        modules = {}
        for name in ("open_webui", "open_webui.utils", "open_webui.utils.chat"):
            module = ModuleType(name)
            module.__path__ = []
            modules[name] = module
        modules["open_webui.utils.chat"].generate_chat_completion = self.complete
        module_patch = patch.dict(sys.modules, modules)
        module_patch.start()
        self.addCleanup(module_patch.stop)
        self.request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/chat/completions",
                "headers": [],
                "state": {},
                "app": SimpleNamespace(),
            }
        )
        self.user = SimpleNamespace(id="fixture-user", role="user")
        self.emitter = AsyncMock()

    async def invoke(self):
        return await self.pipe._complete(
            self.request, self.user, "fixture-model", [], self.valves
        )

    def prepared_source(self):
        snapshot = self.reader.build_snapshot(
            "# First\n\nApproval remains conditional.\n\n"
            "# Second\n\nDelivery is planned for next week.\n",
            "fixture.md",
            "fixture-file",
            "fixture-model",
            self.valves,
        )
        batches = self.reader.make_batches(snapshot, self.valves)
        self.assertEqual(len(batches), 2)
        return snapshot, batches

    def valid_result(self, snapshot, batch):
        passages = {p["id"]: p for p in snapshot["passages"]}
        rows = []
        refs = []
        for pid in batch["passage_ids"]:
            uid = next(u["id"] for u in passages[pid]["units"] if u["text"].strip())
            refs.append(uid)
            rows.append(
                {
                    "id": pid,
                    "extract_ids": [uid],
                    "explanation": [
                        {"text": "A conditional proposal.", "evidence": [uid]}
                    ],
                    "takeaways": [{"text": "Check the condition.", "evidence": [uid]}],
                }
            )
        return {
            "passages": rows,
            "overview": {"text": "A qualified proposal.", "evidence": refs[:1]},
        }

    async def generate(self, snapshot, batches):
        await self.pipe._generate(
            snapshot, batches, self.request, self.user, self.valves, self.emitter
        )

    async def test_http_response_status_survives_without_provider_body(self):
        for status in (400, 401, 403, 404, 408, 413, 422, 429, 500, 503, 504):
            with self.subTest(status=status):
                self.complete.return_value = JSONResponse(
                    {"error": {"message": PRIVATE_DETAIL}}, status_code=status
                )
                with self.assertRaises(self.reader.ReaderError) as raised:
                    await self.invoke()
                message = str(raised.exception)
                self.assertIn(f"HTTP {status}", message)
                self.assertIn("not retried", message)
                self.assertNotIn(PRIVATE_DETAIL, message)
                self.assertLess(len(message), 400)

    async def test_http_exception_maps_status_without_its_detail(self):
        self.complete.side_effect = HTTPException(429, detail=PRIVATE_DETAIL)
        with self.assertRaises(self.reader.ReaderError) as raised:
            await self.invoke()
        self.assertIn("HTTP 429", str(raised.exception))
        self.assertNotIn(PRIVATE_DETAIL, str(raised.exception))
        self.assertEqual(self.complete.await_count, 1)

    async def test_transport_exception_never_exposes_provider_text(self):
        self.complete.side_effect = RuntimeError(PRIVATE_DETAIL)
        with self.assertRaises(self.reader.ReaderError) as raised:
            await self.invoke()
        self.assertIn("result is unknown", str(raised.exception))
        self.assertNotIn(PRIVATE_DETAIL, str(raised.exception))
        self.assertEqual(self.complete.await_count, 1)

    async def test_incomplete_filtered_and_empty_responses_are_distinguished(self):
        cases = (
            (completion('{"valid":"json"}', "length"), "finish_reason=length"),
            (
                completion('{"valid":"json"}', "content_filter"),
                "finish_reason=content_filter",
            ),
            (completion("some text", refusal=PRIVATE_DETAIL), "refused this batch"),
            (completion(" \n"), "empty text content"),
            (completion(None), "no text content"),
            ({"choices": []}, "no completion choices"),
            ({"choices": [None]}, "unexpected completion format"),
            ({"error": {"message": PRIVATE_DETAIL}}, "provider error"),
            (
                Response(PRIVATE_DETAIL, media_type="text/plain"),
                "unreadable response body",
            ),
            (StreamingResponse(iter([])), "unexpected stream"),
        )
        for response, expected in cases:
            with self.subTest(expected=expected):
                self.complete.return_value = response
                with self.assertRaises(self.reader.ReaderError) as raised:
                    await self.invoke()
                self.assertIn(expected, str(raised.exception))
                self.assertNotIn(PRIVATE_DETAIL, str(raised.exception))

    async def test_failed_first_batch_keeps_reason_then_prepares_second_without_retry(
        self,
    ):
        snapshot, batches = self.prepared_source()
        self.complete.side_effect = [
            JSONResponse({"error": PRIVATE_DETAIL}, status_code=503),
            completion(json.dumps(self.valid_result(snapshot, batches[1]))),
        ]
        await self.generate(snapshot, batches)
        lookup = {p["id"]: p for p in snapshot["passages"]}
        self.assertEqual(self.complete.await_count, 2)
        self.assertEqual(snapshot["generation"]["calls"], 2)
        self.assertEqual(snapshot["generation"]["completed_batches"], 1)
        self.assertEqual(snapshot["status"], "partial")
        self.assertIn("HTTP 503", lookup[batches[0]["passage_ids"][0]]["error"])
        self.assertIsNotNone(lookup[batches[1]["passage_ids"][0]]["generated"])
        statuses = [
            call.args[0]["data"]["description"] for call in self.emitter.await_args_list
        ]
        self.assertTrue(
            any(
                "Batch 1/2 unavailable" in value and "HTTP 503" in value
                for value in statuses
            )
        )
        self.assertNotIn(PRIVATE_DETAIL, json.dumps(snapshot) + json.dumps(statuses))

    async def test_invalid_evidence_has_one_repair_and_a_specific_safe_reason(self):
        snapshot, batches = self.prepared_source()
        invalid = self.valid_result(snapshot, batches[0])
        invalid["passages"][0]["extract_ids"] = [PRIVATE_DETAIL]
        self.complete.side_effect = [
            completion(json.dumps(invalid)),
            completion(json.dumps(invalid)),
            completion(json.dumps(self.valid_result(snapshot, batches[1]))),
        ]
        await self.generate(snapshot, batches)
        self.assertEqual(self.complete.await_count, 3)
        error = next(p["error"] for p in snapshot["passages"] if "error" in p)
        self.assertIn("invalid source references", error)
        self.assertNotIn(PRIVATE_DETAIL, error)
        self.assertEqual(snapshot["generation"]["completed_batches"], 1)

    async def test_cancellation_and_timeout_remain_control_flow(self):
        for error in (asyncio.CancelledError(), asyncio.TimeoutError()):
            with self.subTest(error=type(error).__name__):
                self.complete.side_effect = error
                with self.assertRaises(type(error)):
                    await self.invoke()


if __name__ == "__main__":
    unittest.main()
