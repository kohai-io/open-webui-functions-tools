"""Offline SSE collection and resource-lifecycle tests; no model or network calls."""

import asyncio
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch

from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import StreamingResponse


PRIVATE_DETAIL = "private-provider-details-and-source"


def event(content=None, *, finish=None, delta=None, newline="\n"):
    value = {
        "choices": [
            {
                "index": 0,
                "delta": delta if delta is not None else {"content": content},
                "finish_reason": finish,
            }
        ]
    }
    return "data: " + json.dumps(value, ensure_ascii=False) + newline + newline


class TrackedIterator:
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.closed = 0
        self.waiting = asyncio.Event()

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            value = next(self.chunks)
        except StopIteration:
            raise StopAsyncIteration from None
        if value is None:
            self.waiting.set()
            await asyncio.Future()
        if isinstance(value, BaseException):
            raise value
        return value

    async def aclose(self):
        self.closed += 1


class StreamingCompletionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "document_reader_streaming_under_test",
            Path(__file__).parents[1] / "document_reader.py",
        )
        self.reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.reader)
        self.pipe = self.reader.Pipe()
        self.valves = self.reader.Pipe.Valves(
            BASE_MODEL_ID="fixture-model", STREAM_COMPLETIONS=True
        )
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
        self.token = SimpleNamespace(credentials="fixture-token")
        self.request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/chat/completions",
                "headers": [],
                "app": SimpleNamespace(),
                "state": {
                    "token": self.token,
                    "metadata": {"chat_id": "outer-chat"},
                    "direct": True,
                },
            }
        )
        self.user = SimpleNamespace(id="fixture-user", role="user")

    def stream(self, chunks, status_code=200):
        self.iterator = TrackedIterator(chunks)
        self.background = AsyncMock()
        self.complete.return_value = StreamingResponse(
            self.iterator,
            status_code=status_code,
            media_type="text/event-stream",
            background=BackgroundTask(self.background),
        )

    async def invoke(self):
        return await self.pipe._complete(
            self.request, self.user, "fixture-model", [], self.valves
        )

    def assert_released(self):
        self.assertEqual(self.iterator.closed, 1)
        self.background.assert_awaited_once()

    async def assert_failure(self, expected):
        with self.assertRaises(self.reader.ReaderError) as raised:
            await self.invoke()
        self.assertIn(expected, str(raised.exception))
        self.assertNotIn(PRIVATE_DETAIL, str(raised.exception))
        self.assert_released()

    async def test_fragmented_crlf_and_utf8_collect_content_only_with_isolated_request(
        self,
    ):
        expected = '{"text":"Café 👩🏽‍💻"}'
        wire = (
            ": heartbeat\r\n\r\n"
            + event(
                delta={"role": "assistant", "reasoning_content": PRIVATE_DETAIL},
                newline="\r\n",
            )
            + event(expected, newline="\r\n")
            + "data: [DONE]\r\n\r\n"
        ).encode("utf-8")
        # Byte-sized chunks split multibyte characters and CRLF boundaries.
        self.stream([wire[i : i + 1] for i in range(len(wire))])
        self.assertEqual(await self.invoke(), expected)
        self.complete.assert_awaited_once()
        inner, payload, user = self.complete.await_args.args
        self.assertIs(user, self.user)
        self.assertTrue(payload["stream"])
        self.assertNotIn("metadata", payload)
        self.assertIs(inner.state.token, self.token)
        self.assertFalse(hasattr(inner.state, "metadata"))
        self.assertFalse(hasattr(inner.state, "direct"))
        self.assertEqual(self.request.state.metadata, {"chat_id": "outer-chat"})
        self.assert_released()

    async def test_terminal_stop_without_done_and_multiline_data_are_supported(self):
        self.stream(
            [
                'data: {\ndata: "choices": [{"delta":{"content":"complete"}}]\ndata: }\n\n',
                event(delta={}, finish="stop"),
            ]
        )
        self.assertEqual(await self.invoke(), "complete")
        self.assert_released()

    async def test_clean_eof_without_terminal_marker_rejects_even_valid_json(self):
        self.stream([event('{"valid":"json"}')])
        await self.assert_failure("without a terminal")

    async def test_unterminated_done_frame_is_not_accepted_as_a_complete_stream(self):
        self.stream([event("text"), "data: [DONE]"])
        await self.assert_failure("without a terminal")

    async def test_provider_error_event_drops_partial_content_and_hides_details(self):
        for error_frame in (
            "event: error\ndata: " + PRIVATE_DETAIL + "\n\n",
            "data: " + json.dumps({"error": {"message": PRIVATE_DETAIL}}) + "\n\n",
        ):
            with self.subTest(error_frame=error_frame[:12]):
                self.stream([event("partial"), error_frame])
                await self.assert_failure("provider error")

    async def test_truncation_filter_and_refusal_never_accept_partial_or_valid_json(
        self,
    ):
        for frame, reason in (
            (event('{"valid":true}', finish="length"), "finish_reason=length"),
            (
                event('{"valid":true}', finish="content_filter"),
                "finish_reason=content_filter",
            ),
            (event(delta={"refusal": PRIVATE_DETAIL}), "refused this batch"),
        ):
            with self.subTest(reason=reason):
                self.stream([frame])
                await self.assert_failure(reason)

    async def test_content_and_wire_limits_are_enforced(self):
        self.stream([event("x" * 150001), "data: [DONE]\n\n"])
        await self.assert_failure("150,000-character")
        self.stream([b":" + b"x" * (2 * 1024 * 1024)])
        await self.assert_failure("2 MiB")

    async def test_responses_api_events_fail_with_a_clear_protocol_diagnostic(self):
        self.stream(
            [
                "event: response.output_text.delta\ndata: "
                + json.dumps(
                    {"type": "response.output_text.delta", "delta": PRIVATE_DETAIL}
                )
                + "\n\n"
            ]
        )
        await self.assert_failure("Responses API streaming events")

    async def test_malformed_json_and_incomplete_utf8_are_bounded_failures(self):
        self.stream(['data: {"source":' + PRIVATE_DETAIL + "\n\n"])
        await self.assert_failure("malformed completion JSON")
        self.stream([b"data: \xf0"])
        await self.assert_failure("within a UTF-8")

    async def test_network_failure_releases_resources_and_does_not_retry(self):
        self.stream([event("partial"), RuntimeError(PRIVATE_DETAIL)])
        await self.assert_failure("result is unknown")
        self.complete.assert_awaited_once()

    async def test_cancellation_closes_iterator_and_runs_background_cleanup(self):
        self.stream([event("partial"), None])
        task = asyncio.create_task(self.invoke())
        await asyncio.wait_for(self.iterator.waiting.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assert_released()
        self.complete.assert_awaited_once()

    async def test_timeout_closes_iterator_and_runs_background_cleanup(self):
        self.stream([event("partial"), None])
        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(self.invoke(), timeout=0.02)
        self.assert_released()
        self.complete.assert_awaited_once()

    async def test_http_stream_error_and_disabled_mode_release_resources(self):
        self.stream([event(PRIVATE_DETAIL)], status_code=503)
        await self.assert_failure("HTTP 503")
        self.valves.STREAM_COMPLETIONS = False
        self.stream([event(PRIVATE_DETAIL)])
        await self.assert_failure("unexpected stream")
        self.assertFalse(self.complete.await_args.args[1]["stream"])

    async def test_snapshot_records_selected_transport_even_without_generation_batches(
        self,
    ):
        snapshot = {"passages": [], "warnings": [], "status": "complete"}
        await self.pipe._generate(
            snapshot, [], self.request, self.user, self.valves, None
        )
        self.assertTrue(snapshot["generation"]["stream_completions"])
        self.complete.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
