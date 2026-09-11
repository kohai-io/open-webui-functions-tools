"""Opt-in player test against a local authenticated Files API stand-in.

RUN_RUNWAY_BROWSER_TESTS=1 enables this test; requires Playwright, Chrome and ffmpeg.
"""
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from test_runway_inline import runway


@pytest.mark.skipif(os.getenv("RUN_RUNWAY_BROWSER_TESTS") != "1", reason="opt-in local browser test")
@pytest.mark.parametrize("player_count", [1, 3])
def test_authenticated_player_sandbox_reload_seeking_and_failures(runway, tmp_path, player_count):
    from playwright.sync_api import sync_playwright

    clip = tmp_path / "fixture.mp4"
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=blue:s=160x90:r=10",
        "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(clip),
    ], check=True, capture_output=True)
    video = clip.read_bytes()
    requests = []
    state = {"code": 200}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            if self.path == "/webui/chat":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Set-Cookie", "token=test-session; Path=/; SameSite=Lax; HttpOnly")
                content = b"<!doctype html><html><body></body></html>"
            else:
                requests.append((self.path, self.headers.get("Cookie", "")))
                code = state["code"] if "token=test-session" in self.headers.get("Cookie", "") else 401
                self.send_response(code)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Disposition", 'attachment; filename="fixture.mp4"')
                content = video if code == 200 else b"denied"
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    html = [runway.Pipe._video_embed_html(f"/webui/api/v1/files/abcd-{i}/content", f"fixture-{i}.mp4")
            for i in range(player_count)]
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True)
            try:
                page = browser.new_page()
                def mount(same_origin):
                    page.goto(f"http://127.0.0.1:{server.server_port}/webui/chat")
                    page.evaluate("""({html, sameOrigin}) => {
                      for (const template of html) {
                        const frame = document.createElement('iframe');
                        frame.width = '100%';
                        frame.sandbox = 'allow-scripts allow-downloads' + (sameOrigin ? ' allow-same-origin' : '');
                        frame.srcdoc = template;
                        document.body.append(frame);
                        window.addEventListener('message', e => {
                            if (e.source === frame.contentWindow && e.data.type === 'iframe:height')
                                frame.style.height = e.data.height + 'px';
                        });
                      }
                    }""", {"html": html, "sameOrigin": same_origin})
                    return [page.locator("iframe").nth(i).content_frame for i in range(player_count)]

                for frame in mount(False):
                    frame.get_by_text("To play this saved video", exact=False).wait_for()
                assert requests == []  # Opaque sandbox never attempts the protected fetch.

                for _ in range(2):  # Reopening uses the durable API URL, not a saved blob URL.
                    for i, frame in enumerate(mount(True)):
                        player = frame.locator("video")
                        player.wait_for(state="visible")
                        player.evaluate("el => new Promise(resolve => el.readyState >= 1 ? resolve() : el.addEventListener('loadedmetadata', resolve, {once:true}))")
                        assert player.evaluate("el => el.duration") == pytest.approx(2, abs=0.2)
                        player.evaluate("el => el.play()")
                        player.evaluate("el => new Promise(resolve => { el.addEventListener('timeupdate', resolve, {once:true}); })")
                        assert player.evaluate("el => el.currentTime") > 0
                        player.evaluate("el => { el.pause(); el.currentTime = 1; }")
                        assert player.evaluate("el => el.currentTime") == pytest.approx(1, abs=0.2)
                        assert frame.locator("#download").get_attribute("href").startswith("blob:")
                        assert page.locator("iframe").nth(i).evaluate("el => el.clientHeight") > 150

                assert len(requests) == 2 * player_count
                for i in range(player_count):
                    assert sum(path == f"/webui/api/v1/files/abcd-{i}/content" for path, _ in requests) == 2
                assert all("token=test-session" in cookie for _, cookie in requests)
                for code, message in [(403, "Sign in to Open WebUI"), (404, "The saved video is unavailable")]:
                    state["code"] = code
                    for frame in mount(True):
                        frame.get_by_text(message, exact=False).wait_for()
                        assert not frame.locator("video").is_visible()
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
