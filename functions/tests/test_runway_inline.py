"""Offline Runway contract, authentication, events and Files API regression tests."""
import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import httpx
import jsonschema
import pytest


TASK = "17f20503-6c24-4c16-946b-35dbbce2af2f"
MP4 = b"\x00\x00\x00\x18ftypisom" + b"fixture"


@pytest.fixture
def runway(monkeypatch):
    for name in (
        "open_webui", "open_webui.models", "open_webui.models.files", "open_webui.models.users",
        "open_webui.models.chats", "open_webui.storage", "open_webui.storage.provider",
        "open_webui.routers", "open_webui.routers.files", "open_webui.utils",
        "open_webui.utils.access_control", "open_webui.utils.access_control.files",
    ):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["open_webui.models.files"].Files = NS(get_file_by_id=AsyncMock())
    sys.modules["open_webui.models.users"].Users = NS(get_user_by_id=AsyncMock(return_value=owner()))
    sys.modules["open_webui.models.chats"].Chats = NS(insert_chat_files=AsyncMock())
    sys.modules["open_webui.storage.provider"].Storage = NS(get_file=Mock())
    sys.modules["open_webui.routers.files"].upload_file_handler = AsyncMock(return_value=NS(id="abcd", filename="clip.mp4"))
    sys.modules["open_webui.utils.access_control.files"].has_access_to_file = AsyncMock(return_value=False)
    spec = importlib.util.spec_from_file_location("runway_under_test", Path(__file__).parents[1] / "runway_inline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def owner():
    return NS(id="user-a", role="user")


def request():
    return NS(scope={"root_path": "/webui"}, app=NS(url_path_for=lambda route, id: f"/api/v1/files/{id}/content"))


def body(prompt="A cat walking on the moon"):
    return {"messages": [{"role": "user", "content": prompt}]}


def options(runway, **overrides):
    opts, _ = runway.Pipe()._options(None)
    return {**opts, **overrides}


def transport(monkeypatch, runway, handler):
    original = httpx.AsyncClient
    monkeypatch.setattr(runway.httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))


@pytest.mark.parametrize("model,has_image", [
    ("seedance2_5", False), ("seedance2_5", True), ("gen4.5", False),
    ("gen4.5", True), ("gen4_turbo", True),
])
def test_payloads_match_published_runway_schema(runway, model, has_image):
    schemas = json.loads((Path(__file__).parent / "fixtures/runway_video_requests.json").read_text())["schemas"]
    pipe = runway.Pipe()
    for resolution, ratios in pipe._formats(model, has_image).items():
        for shape, pixels in ratios.items():
            for duration in ((4, 30) if model == "seedance2_5" else (2, 10)):
                endpoint, payload = pipe._payload("Camera pans across a lake", "https://example.test/frame.png" if has_image else None,
                    options(runway, model=model, duration=duration, aspect_ratio=shape, resolution=resolution))
                jsonschema.Draft202012Validator(schemas[endpoint]).validate(payload)
                assert payload["ratio"] == pixels
                assert ("audio" in payload) == (model == "seedance2_5")
                if has_image:
                    assert payload["promptImage"][0]["position"] == "first"


@pytest.mark.parametrize("settings,image", [
    ({"model": "gen4_turbo"}, None), ({"model": "gen4.5", "resolution": "1080p"}, "image"),
    ({"model": "gen4.5", "aspect_ratio": "1:1"}, None), ({"duration": 3}, None),
    ({"duration": 31}, None), ({"duration": True}, None), ({"duration": 4.5}, None),
])
def test_invalid_model_combinations_rejected(runway, settings, image):
    with pytest.raises(ValueError):
        runway.Pipe()._payload("scene", image, options(runway, **settings))


def test_injected_user_settings_and_inline_override_precedence(runway):
    pipe = runway.Pipe()
    prefs = pipe.UserValves(MODEL="seedance2_5", RESOLUTION="1080p", ASPECT_RATIO="9:16", DURATION=8)
    settings, _ = pipe._options({"valves": prefs})
    prompt, effective = pipe._inline_options('scene {"duration": 12, "ratio": "1280:720", "audio": false}', settings, False)
    assert prompt == "scene" and effective["duration"] == 12 and effective["resolution"] == "720p"
    assert effective["audio"] is False
    assert pipe._options(None)[0]["duration"] == 5
    prompt, effective = pipe._inline_options('scene <runway duration="8" model="gen4.5" ratio="9:16" />', options(runway), False)
    assert effective["duration"] == 8 and effective["aspect_ratio"] == "9:16"
    assert effective["model"] == "gen4.5" and effective["audio"] is False


def test_input_uses_latest_message_and_deduplicates_file_representations(runway):
    data = {"messages": [
        {"role": "user", "content": "![old](https://example.test/old.png)"},
        {"role": "user", "content": "new scene"},
    ]}
    assert runway.Pipe._input(data, []) == ("new scene", None)
    data["messages"][-1]["content"] = [{"type": "text", "text": "Animate"}, {"type": "image_url", "image_url": {"url": "/webui/api/v1/files/abcd/content"}}]
    prompt, ref = runway.Pipe._input(data, [{"type": "file", "content_type": "image/png", "id": "abcd"}])
    assert prompt == "Animate" and ref.endswith("/abcd/content")
    with pytest.raises(ValueError, match="one first-frame"):
        runway.Pipe._input(body("![one](https://example.test/one.png) ![two](https://example.test/two.png)"), [])


def test_private_image_requires_access_before_storage_read(runway, tmp_path):
    file = tmp_path / "image.png"
    file.write_bytes(b"png-bytes")
    lookup = sys.modules["open_webui.models.files"].Files.get_file_by_id
    lookup.return_value = NS(id="abcd", user_id="other", path=str(file), filename="image.png", meta={"content_type": "image/png"})
    storage = sys.modules["open_webui.storage.provider"].Storage.get_file
    storage.return_value = str(file)
    with pytest.raises(ValueError, match="access"):
        asyncio.run(runway.Pipe()._read_image("/api/v1/files/abcd/content", owner()))
    storage.assert_not_called()
    sys.modules["open_webui.utils.access_control.files"].has_access_to_file.return_value = True
    assert asyncio.run(runway.Pipe()._read_image("/api/v1/files/abcd/content", owner())) == (b"png-bytes", "image/png")


@pytest.mark.parametrize("ref", ["data:image/gif;base64,YQ==", "data:image/png;base64,!!!", "http://example.test/a.png", "https://127.0.0.1/a.png"])
def test_bad_image_references_stop_early(runway, ref):
    with pytest.raises(ValueError):
        asyncio.run(runway.Pipe()._read_image(ref, owner()))


@pytest.mark.parametrize("answer", [False, None, {"error": "Client session disconnected."}, {"error": "timeout"}, "true", 1])
def test_confirmation_requires_literal_true_and_no_provider_call(runway, answer):
    pipe = runway.Pipe()
    pipe.valves.RUNWAY_API_KEY = "secret"
    pipe._api = AsyncMock()
    pipe._prepare_image = AsyncMock()
    asyncio.run(pipe.pipe(body(), __user__={"id": "user-a"}, __request__=request(), __event_call__=AsyncMock(return_value=answer)))
    pipe._api.assert_not_awaited()
    pipe._prepare_image.assert_not_awaited()


def test_missing_or_timed_out_dialog_never_proceeds(runway):
    pipe = runway.Pipe()
    for caller in (None, AsyncMock(side_effect=asyncio.TimeoutError)):
        with pytest.raises(runway.UserCancelled):
            asyncio.run(pipe._ask(caller, {"type": "confirmation"}))


def test_options_dialog_uses_native_selects_and_validates_answers(runway):
    pipe = runway.Pipe()
    caller = AsyncMock(side_effect=["seedance2_5", "1080p|9:16", "12", "no"])
    selected = asyncio.run(pipe._choose_options(caller, options(runway), False))
    assert selected == options(runway, resolution="1080p", aspect_ratio="9:16", duration=12, audio=False)
    events = [c.args[0] for c in caller.call_args_list]
    assert all(e["type"] == "input" and e["data"]["input"]["type"] == "select" for e in events)
    assert "gen4_turbo" not in [o["value"] for o in events[0]["data"]["input"]["options"]]
    with pytest.raises(runway.UserCancelled):
        asyncio.run(pipe._choose_options(AsyncMock(return_value="unlisted"), options(runway), False))


@pytest.mark.parametrize("task", ["title_generation", "follow_up_generation", "tags_generation"])
def test_auxiliary_tasks_do_not_create_video_or_dialogs(runway, task):
    pipe = runway.Pipe()
    pipe._api = AsyncMock()
    caller = AsyncMock()
    assert asyncio.run(pipe.pipe(body(), __task__=task, __event_call__=caller)) == ""
    caller.assert_not_awaited()
    pipe._api.assert_not_awaited()


@pytest.mark.parametrize("has_image", [False, True])
def test_full_flow_uses_correct_endpoint_auth_storage_and_embed(runway, monkeypatch, has_image):
    seen = []
    def handler(req):
        seen.append(req)
        if req.url.host == "api.dev.runwayml.com":
            assert req.headers["Authorization"] == "Bearer secret"
            assert req.headers["X-Runway-Version"] == "2024-11-06"
            if req.method == "POST":
                assert req.url.path == ("/v1/image_to_video" if has_image else "/v1/text_to_video")
                payload = json.loads(req.content)
                assert payload["model"] == "seedance2_5" and payload["duration"] == 5
                if has_image:
                    assert payload["promptImage"] == [{"uri": "data:image/png;base64,YQ==", "position": "first"}]
                return httpx.Response(200, json={"id": TASK})
            return httpx.Response(200, json={"id": TASK, "status": "SUCCEEDED", "output": ["https://cdn.example.test/video.mp4"]})
        assert "Authorization" not in req.headers and "X-Runway-Version" not in req.headers
        return httpx.Response(200, content=MP4, headers={"content-type": "video/mp4"})
    transport(monkeypatch, runway, handler)
    pipe = runway.Pipe()
    pipe.valves.RUNWAY_API_KEY = "secret"
    emitter = AsyncMock()
    data = body("Animate ![frame](data:image/png;base64,YQ==)" if has_image else "A cat")
    response = asyncio.run(pipe.pipe(data, __user__={"id": "user-a"}, __request__=request(),
        __event_call__=AsyncMock(return_value=True), __event_emitter__=emitter,
        __metadata__={"chat_id": "chat-a", "message_id": "message-a"}))
    assert "[Download video 1](/webui/api/v1/files/abcd/content?attachment=true)" in response
    upload = sys.modules["open_webui.routers.files"].upload_file_handler.call_args
    assert upload.kwargs["user"].id == "user-a" and upload.kwargs["process"] is False
    assert upload.kwargs["file"].file.closed
    assert upload.kwargs["metadata"]["generation_options"]["model"] == "seedance2_5"
    sys.modules["open_webui.models.chats"].Chats.insert_chat_files.assert_awaited_once()
    embeds = [c.args[0] for c in emitter.call_args_list if c.args[0]["type"] == "embeds"]
    assert len(embeds) == 1
    assert "secret" not in str(embeds) and "cdn.example" not in str(embeds)
    assert len(seen) == 3


def test_large_image_upload_has_no_api_key_on_storage_request(runway, monkeypatch):
    seen = []
    def handler(req):
        seen.append(req)
        assert "authorization" not in req.headers
        assert b'name="policy"' in req.content and b'name="file"' in req.content
        return httpx.Response(204)
    transport(monkeypatch, runway, handler)
    pipe = runway.Pipe()
    pipe._api = AsyncMock(return_value={"uploadUrl": "https://storage.example.test/upload", "fields": {"policy": "signed-policy"}, "runwayUri": "runway://uploaded-image"})
    result = asyncio.run(pipe._prepare_image((b"a" * 4 * 1024 * 1024, "image/png"), None))
    assert result == "runway://uploaded-image" and len(seen) == 1
    pipe._api.assert_awaited_once_with(None, "POST", "uploads", {"filename": "frame.png", "type": "ephemeral"})


def test_generation_network_failure_does_not_retry_or_expose_secret(runway):
    pipe = runway.Pipe()
    client = NS(request=AsyncMock(side_effect=httpx.ReadTimeout("secret signed-url")))
    with pytest.raises(runway.RunwayError, match="may have been created") as error:
        asyncio.run(pipe._api(client, "POST", "text_to_video", {}))
    assert "secret" not in str(error.value)
    assert client.request.await_count == 1


def test_poll_retries_transient_gets_and_keeps_all_outputs(runway, monkeypatch):
    pipe = runway.Pipe()
    sleep = AsyncMock()
    monkeypatch.setattr(runway.asyncio, "sleep", sleep)
    pipe._api = AsyncMock(side_effect=[runway.RunwayError("throttled", 429, 12),
        {"status": "PENDING"}, {"status": "RUNNING", "progress": 0.5},
        {"status": "SUCCEEDED", "output": ["one", "two"]}])
    assert asyncio.run(pipe._poll(None, TASK, None))["output"] == ["one", "two"]
    assert sleep.call_args_list[0].args[0] == 12
    assert all(c.args[1] == "GET" for c in pipe._api.call_args_list)


@pytest.mark.parametrize("result", [{"status": "FAILED", "failureCode": "SAFETY.INPUT"}, {"status": "CANCELLED"}, {"status": "SUCCEEDED", "output": []}])
def test_terminal_failures_are_not_retried(runway, result):
    pipe = runway.Pipe()
    pipe._api = AsyncMock(return_value=result)
    with pytest.raises(runway.RunwayError):
        asyncio.run(pipe._poll(None, TASK, None))
    assert pipe._api.await_count == 1


def test_partial_storage_failure_keeps_other_outputs(runway):
    pipe = runway.Pipe()
    pipe._download = AsyncMock(return_value=MP4)
    pipe._save = AsyncMock(side_effect=[RuntimeError("storage"), ("/api/v1/files/good/content", "good.mp4")])
    emitter = AsyncMock()
    response = asyncio.run(pipe._deliver({"id": TASK, "output": ["one", "two"]}, request(), owner(), {}, options(runway), emitter))
    assert "Video 1 could not be saved" in response and "Download video 2" in response
    assert len([c for c in emitter.call_args_list if c.args[0]["type"] == "embeds"][0].args[0]["data"]["embeds"]) == 1


def test_cancel_poll_attempts_to_cancel_only_created_task(runway, monkeypatch):
    transport(monkeypatch, runway, lambda req: httpx.Response(200, json={}))
    pipe = runway.Pipe()
    pipe.valves.RUNWAY_API_KEY = "secret"
    pipe._api = AsyncMock(return_value={"id": TASK})
    pipe._poll = AsyncMock(side_effect=asyncio.CancelledError)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(pipe.pipe(body(), __user__={"id": "user-a"}, __request__=request(), __event_call__=AsyncMock(return_value=True)))
    assert pipe._api.call_args.args[1:] == ("DELETE", f"tasks/{TASK}")


def test_encryption_remains_compatible_and_fails_when_key_lost(runway, monkeypatch):
    monkeypatch.setenv("WEBUI_SECRET_KEY", "test-secret")
    valves = runway.Pipe.Valves(RUNWAY_API_KEY="test-api-key")
    assert str(valves.RUNWAY_API_KEY).startswith("encrypted:")
    assert runway.EncryptedStr.decrypt(str(valves.RUNWAY_API_KEY)) == "test-api-key"
    monkeypatch.delenv("WEBUI_SECRET_KEY")
    with pytest.raises(ValueError, match="decrypt"):
        runway.EncryptedStr.decrypt(str(valves.RUNWAY_API_KEY))
