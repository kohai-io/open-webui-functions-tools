# Runway inline Pipe 3.0.0

A standalone Open WebUI Function for text-to-video and single-image-to-video.
Reviewed against this checkout's Open WebUI 0.11.1 and Runway's API schema on
11 September 2026. This replaces the older image-only `runway_inline.py`.

## Deploy and configure

1. Import or replace the Function with `runway_inline.py`, then enable it.
2. Set `RUNWAY_API_KEY` to a **Runway developer API key**, or provide it through
   the server environment variable `RUNWAYML_API_SECRET`.
3. Set the administrator `MODEL` valve to `seedance2_5`. New installations default
   to this; existing installations retain their saved valve values, including the
   old `gen4_turbo` default. Saved credentials remain compatible with this version.
4. Keep the default API base `https://api.dev.runwayml.com/v1`. Both generation
   endpoints use the required `X-Runway-Version: 2024-11-06` header.
5. For inline playback, each viewer must enable **Settings → Interface → iframe
   Sandbox Allow Same Origin**, then reopen the chat. A saved Files API download
   link is available even when iframe playback is blocked.

Runtime Python dependencies are `httpx`, `cryptography` and Pydantic 2 (including
its matching `pydantic-core`). They are already in this checkout's full and minimal
backend requirements. FastAPI and the Files/storage/database modules come from
Open WebUI. There is no dependency on the `runwayml` SDK, Google SDK, Veo Pipe,
`c2pa-python`, FFmpeg, Playwright or pytest at runtime.

The encrypted key valve uses `WEBUI_SECRET_KEY`. Preserve that secret across
deployments. When no secret is configured, the valve is stored as plaintext;
use the server environment variable if you prefer not to store it in Function
settings. A key encrypted with a missing or different secret fails with guidance
to restore the secret or re-enter the key.

## Models and options

| Model | Modes | Duration | Formats exposed by this Pipe |
| --- | --- | --- | --- |
| `seedance2_5` (default) | Text or first-frame image | 4–30 whole seconds | 480p, 720p, 1080p; six aspect ratios; audio on/off |
| `gen4.5` | Text or first-frame image | 2–10 whole seconds | 720p; landscape/portrait for text, six aspect ratios for images; silent |
| `gen4_turbo` | First-frame image required | 2–10 whole seconds | 720p; six aspect ratios; silent |

The six shapes are 21:9, 16:9, 4:3, 1:1, 3:4 and 9:16. The Pipe maps them to
Runway's model-specific pixel ratios; for example, Seedance 1080p landscape becomes
`1920:1080`. Changing a model never silently substitutes another model or lowers
resolution. Invalid combinations stop before a provider request.

Administrator defaults are five seconds and `RATIO=1280:720`. In **User Valves**,
set `MODEL`, `DURATION`, `ASPECT_RATIO`, `RESOLUTION` and `AUDIO`. Model/shape/
resolution values of `default`, and duration `0`, inherit administrator settings.
Audio defaults to on for Seedance and is omitted from Gen-4 requests. Preferences
are read from the current caller's injected valves for every request.

Older per-message overrides still work when placed at the end of the prompt:

```text
A slow tracking shot through a forest.
{"model": "seedance2_5", "duration": 8, "ratio": "1920:1080", "audio": true}
```

The old `<runway duration="8" ratio="16:9" model="seedance2_5" />` form also works.
Recognized options are removed from the provider prompt and validated. They
override User Valves for that request; interactive choices, when enabled, come last.

## Dialogs and events

`REQUIRE_CONFIRMATION=True` is the default. Before sending any prompt or image to
Runway, the Pipe shows an OWUI confirmation containing the model, duration, shape,
resolution, audio setting and prompt preview. It states that generation uses
Runway credits. Only a literal positive confirmation starts a task.

For interactive settings, enable the **User Valve `ASK_OPTIONS`**. This adds native
OWUI `input` select dialogs for model, format, duration and (for Seedance) audio.
The choices reflect the selected model and whether an image is present. The
confirmation follows those choices. These use `__event_call__`; no browser
JavaScript execution event or chat UI change is required.

Cancelling, closing, disconnecting or exceeding the 180-second dialog timeout
stops before upload/generation. `EVENT_TIMEOUT` configures the Pipe's bound;
OWUI's `WEBSOCKET_EVENT_CALLER_TIMEOUT` may impose a shorter bound. An API-only
client cannot answer dialogs. Administrators can set `REQUIRE_CONFIRMATION=False`
for those clients, and their User Valves must also have `ASK_OPTIONS=False`.

Status events show submission, Runway task state and saving. The task ID appears
in the status and final response. Auxiliary title/tag/follow-up tasks return
without generation or dialogs.

## Images and saved videos

Attach one PNG, JPEG or WebP, or include an explicit Markdown image in the latest
user message. Without an image, the Pipe uses text-to-video. It does not search
older messages for an image. An unreadable or unsupported attached image stops
the request rather than falling back to text-to-video.

OWUI Files references are checked for ownership or read access before reading
storage. Small private images are submitted as base64 data URIs; when the encoded
URI exceeds 5 MiB, the Pipe uses a Runway ephemeral upload. The default local
image limit is 20 MiB (`MAX_INPUT_MB`). Public HTTPS image URLs can be passed to
Runway directly: they must use a domain name, support HEAD, serve a supported
image MIME type, and not redirect. Existing `runway://` upload URIs are accepted.

One generation task is created per invocation. Every returned MP4 is downloaded
and saved with OWUI's async `upload_file_handler(process=False)`, owned by the
authenticated caller and linked to the saved chat/message. A failure to save one
output does not discard the others. Temporary chats have no persistent chat-file
association. Both the Pipe's `MAX_VIDEO_MB` (256 MiB default) and OWUI's upload
limit apply. The Pipe adds no C2PA signing or video transcoding.

API requests use a separate HTTP client from signed uploads and output downloads,
so the Runway API key is never forwarded to storage/CDN hosts. Persistent embeds
contain only the protected OWUI file path and a fixed HTML template. Playback
fetches the video with the current viewer's session cookie and creates a temporary
blob URL. No credentials, video bytes or expiring Runway output URLs enter embeds.

The iframe same-origin setting applies to all of a viewer's rich embeds; enable
it only with trusted embeds. If `IFRAME_CSP` is configured, allow this template's
inline script/style, same-origin `connect-src` and `blob:` in `media-src`. Playback
fetches the full video before starting and depends on the browser's codec support.
Anonymous viewers cannot access protected files.

## Failures and limits

Generation POSTs are not automatically retried. A network failure after submission
can leave a running task even if no ID was received; check Runway before retrying.
Polling waits at least five seconds between GETs and retries rate-limit/transient
server failures within `MAX_POLL_TIME` (15 minutes by default).

A polling timeout leaves the provider task running and reports its ID. If OWUI
cancels the Pipe while it is polling, it attempts to cancel only that request's
task. Cancellation is best effort; a stopped browser or server cannot guarantee
that Runway stopped or refunded a task. A server restart interrupts this Pipe's
polling; there is no background recovery worker. Failed downloads should be
retrieved through Runway before the provider's output URLs expire.

This version covers text and a single first-frame image. It does not expose
video-to-video, multiple reference images, last frames, audio references, automatic
duration, 4K/HDR, batches of generation requests or Runway's Model Router.

## Validation

Offline tests validate generated payloads against a snapshot of Runway's published
request schemas, plus authentication, image ownership, upload/download credential
separation, native dialog payloads, cancellation, polling and partial storage
failures. The browser tests exercise one and three players, opaque/default versus
same-origin sandbox behaviour, authenticated playback, seeking, reload and errors.

Run from the functions/tools repository root:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest functions/tests/test_runway_inline.py -q

# Optional: requires Playwright, Chrome and ffmpeg.
$env:RUN_RUNWAY_BROWSER_TESTS = '1'
python -m pytest functions/tests/test_runway_inline.py functions/tests/test_runway_inline_browser.py -q
```

Tests require pytest and jsonschema, with no live Runway key or OWUI database.
No paid generation, live Function installation or live storage write was run.

Sources: [Runway API](https://docs.dev.runwayml.com/api/),
[Runway input requirements](https://docs.dev.runwayml.com/assets/inputs/),
[Runway models](https://docs.dev.runwayml.com/guides/models/),
[OWUI events](https://docs.openwebui.com/features/extensibility/plugin/development/events/).
