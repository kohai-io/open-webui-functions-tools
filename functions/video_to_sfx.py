"""
title: Video → SFX (ElevenLabs) Pipeline
author: open-webui
date: 2025-09-27
version: 1.0
license: MIT
description: Generate sound effects for a video by extracting client-provided frames, crafting an SFX prompt via GPT-4o, generating audio via ElevenLabs SFX API, and returning a downloadable file. Merging with the video is intended to happen on the client with ffmpeg.wasm.
requirements: aiohttp, cryptography, pydantic, imageio-ffmpeg

USAGE
- Client extracts 4 frames at ~1s intervals from a video (on the client) and sends them as images in the last user message along with text instructions.
- This pipeline will:
  1) Parse the last user message for up to 4 images and the accompanying text.
  2) Ask the OpenAI-compatible vision model (e.g., GPT-4o) to craft an ElevenLabs SFX prompt.
  3) Call ElevenLabs Sound Effects API to generate an audio file with the prompt.
  4) Save the audio to Open WebUI Files DB and return a link.

OUTPUT FORMAT
- Returns a message with a clickable download link to the generated audio file, e.g., /api/v1/files/{id}/content
- Also returns a compact JSON hint block that frontends can parse to run ffmpeg.wasm merge:
  { "type": "video_sfx_result", "file_id": "...", "mime": "audio/mpeg", "suggested_action": "merge_with_video_client" }

OPEN WEBUI PROMPT-VARIABLES (Turbo-safe)
User text instruction (pairs with frames included as images in the same message):
{{instruction | textarea:placeholder="Describe the desired sound mood/effects matching the video scene." :required}}

Optional output format override (leave empty to use valve default):
{{output_format | text:placeholder="e.g., mp3_22050_32"}}

{"output_format": "{{output_format}}"}
"""

from typing import Optional, Callable, Awaitable, Any, List, Tuple
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from cryptography.fernet import Fernet, InvalidToken
import aiohttp
import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import sys
import time
import tempfile
import uuid
from pydantic_core import core_schema
import subprocess
from pathlib import Path
import imageio_ffmpeg
import shutil

# Open WebUI files + storage
from open_webui.models.files import Files as FilesDB
from open_webui.storage.provider import Storage


# Encrypted string helper (same pattern used in other pipelines)
class EncryptedStr(str):
    """A string type that automatically handles encryption/decryption"""

    @classmethod
    def _get_encryption_key(cls) -> Optional[bytes]:
        secret = os.getenv("WEBUI_SECRET_KEY")
        if not secret:
            return None
        hashed_key = hashlib.sha256(secret.encode()).digest()
        return base64.urlsafe_b64encode(hashed_key)

    @classmethod
    def encrypt(cls, value: str) -> str:
        if not value or value.startswith("encrypted:"):
            return value
        key = cls._get_encryption_key()
        if not key:
            return value
        f = Fernet(key)
        encrypted = f.encrypt(value.encode())
        return f"encrypted:{encrypted.decode()}"

    @classmethod
    def decrypt(cls, value: str) -> str:
        if not value or not value.startswith("encrypted:"):
            return value
        key = cls._get_encryption_key()
        if not key:
            return value
        try:
            encrypted_part = value[len("encrypted:") :]
            f = Fernet(key)
            decrypted = f.decrypt(encrypted_part.encode())
            return decrypted.decode()
        except (InvalidToken, Exception):
            return value

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.union_schema(
            [
                core_schema.is_instance_schema(cls),
                core_schema.chain_schema(
                    [
                        core_schema.str_schema(),
                        core_schema.no_info_plain_validator_function(
                            lambda value: cls(cls.encrypt(value) if value else value)
                        ),
                    ]
                ),
            ],
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: str(instance)
            ),
        )

    def get_decrypted(self) -> str:
        return self.decrypt(self)


def _subprocess_run_wrapper(cmd: list[str]):
    """Small wrapper so we can call subprocess.run in a background thread consistently."""
    import subprocess as _sub

    return _sub.run(cmd, capture_output=True)


class Pipe:
    class Valves(BaseModel):
        # ElevenLabs
        ELEVEN_API_KEY: EncryptedStr = Field(
            default="",
            description="ElevenLabs API key (xi-api-key). Get it from https://elevenlabs.io/",
        )
        ELEVEN_API_BASE_URL: str = Field(
            default="https://api.elevenlabs.io/v1",
            description="Base URL for ElevenLabs API",
        )
        OUTPUT_FORMAT: str = Field(
            default="mp3_22050_32",
            description="Output audio format (e.g., mp3_22050_32, mp3_44100_128, wav).",
        )
        LOOP: bool = Field(
            default=False,
            description="Enable seamless looping for sounds longer than 30 seconds.",
        )
        PROMPT_INFLUENCE: Optional[float] = Field(
            default=None,
            description="Control prompt interpretation (0.0=creative, 1.0=literal). None = default.",
        )
        # OpenAI-compatible (for GPT-4o vision)
        OPENAI_API_KEY: EncryptedStr = Field(
            default="",
            description="OpenAI-compatible API key for vision model (e.g., GPT-4o).",
        )
        OPENAI_API_BASE_URL: str = Field(
            default="https://api.openai.com/v1",
            description="OpenAI-compatible base URL (must support image input).",
        )
        OPENAI_MODEL: str = Field(
            default="gpt-4o-mini",
            description="Vision-capable model to craft the SFX prompt (e.g., gpt-4o or gpt-4o-mini).",
        )
        TIMEOUT: int = Field(
            default=180,
            description="Max seconds to wait for remote API calls.",
        )
        DEBUG: bool = Field(
            default=False,
            description="Enable verbose debug logging for troubleshooting.",
        )
        MAX_SFX_PROMPT_CHARS: int = Field(
            default=450,
            description="Max characters allowed for ElevenLabs SFX text. Requests will be trimmed to this length.",
        )
        AUTO_DETECT_DURATION: bool = Field(
            default=True,
            description="Automatically detect video duration and use it for SFX generation (max 30s per ElevenLabs API).",
        )
        FRAME_PROMPT_MAX_CHARS: int = Field(
            default=120,
            description="Max characters for each per-frame SFX prompt returned in the message output.",
        )
        EMIT_INTERVAL: float = Field(
            default=0.5, description="Interval in seconds between status emissions"
        )
        ENABLE_STATUS_INDICATOR: bool = Field(
            default=True, description="Enable or disable status indicator emissions"
        )

    def __init__(self):
        self.name = "Video → SFX (ElevenLabs)"
        self.valves = self.Valves()
        self.last_emit_time = 0
        self.log = logging.getLogger("video_to_sfx_pipeline")
        self.log.setLevel(logging.DEBUG if self.valves.DEBUG else logging.INFO)

    async def emit_status(
        self,
        event_emitter: Optional[Callable[[dict], Awaitable[None]]],
        level: str,
        message: str,
        done: bool = False,
    ) -> None:
        """Emit status updates to Open WebUI"""
        if not event_emitter:
            return
        
        if not self.valves.ENABLE_STATUS_INDICATOR:
            return

        current_time = time.time()
        if current_time - self.last_emit_time >= self.valves.EMIT_INTERVAL or done:
            try:
                await event_emitter(
                    {
                        "type": "status",
                        "data": {
                            "description": message,
                            "done": done,
                        },
                    }
                )
                self.last_emit_time = current_time
            except Exception as e:
                self.log.warning(f"Failed to emit status: {e}")

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __event_call__: Optional[Callable[[dict], Awaitable[dict]]] = None,
    ) -> Optional[str]:
        await self.emit_status(
            __event_emitter__, "info", "Initializing Video → SFX pipeline..."
        )
        
        try:
            self.log.debug(f"video_to_sfx: incoming body keys={list(body.keys())}")
            self.log.debug(
                f"video_to_sfx: messages_count={len(body.get('messages', []))}"
            )
        except Exception:
            pass
        # Decrypt keys
        eleven_key = self.valves.ELEVEN_API_KEY.get_decrypted()
        openai_key = self.valves.OPENAI_API_KEY.get_decrypted()
        if not eleven_key:
            err = "ElevenLabs API key not configured. Please set ELEVEN_API_KEY in the pipeline settings."
            self.log.error(err)
            await self.emit_status(__event_emitter__, "error", err, True)
            return f"❌ Error: {err}"
        if not openai_key:
            err = "OpenAI API key not configured. Please set OPENAI_API_KEY in the pipeline settings."
            self.log.error(err)
            await self.emit_status(__event_emitter__, "error", err, True)
            return f"❌ Error: {err}"

        # Extract last user message: text + up to 4 images (frames)
        messages = body.get("messages", [])
        # Require the latest message to be authored by the user (avoid assistant follow-ups triggering work)
        try:
            if not messages or (messages[-1] or {}).get("role") != "user":
                return "❌ Error: Please send your instruction as a user message, together with frames or a video link."
        except Exception:
            return "❌ Error: Please send your instruction as a user message."

        # Guarded extraction (latest user-only)
        user_text, image_urls = self._extract_user_text_and_images(
            messages, max_images=4
        )
        self.log.debug(
            f"video_to_sfx: extracted user_text_len={len(user_text or '')}, images_found={len(image_urls)}"
        )
        if not user_text:
            return "❌ Error: No user instruction provided. Please include text describing the desired sound."
        original_video_url: Optional[str] = None
        if not image_urls:
            # Attempt server-side extraction if a video URL is present
            video_url = self._extract_video_url(messages)
            self.log.debug(f"video_to_sfx: _extract_video_url => {video_url}")
            if video_url:
                original_video_url = video_url
                await self.emit_status(
                    __event_emitter__, "info", "Extracting frames from video..."
                )
                try:
                    tmp_video = await self._download_temp_file(video_url)
                    self.log.debug(
                        f"video_to_sfx: downloaded/resolved video to temp path: {tmp_video}"
                    )
                    frame_paths = await self._extract_frames_ffmpeg(
                        tmp_video, [0, 1, 2, 3]
                    )
                    image_urls = [self._file_to_data_url(p) for p in frame_paths]
                    self.log.debug(
                        f"video_to_sfx: converted frames to data URLs count={len(image_urls)}"
                    )
                finally:
                    try:
                        if (
                            "tmp_video" in locals()
                            and tmp_video
                            and os.path.exists(tmp_video)
                        ):
                            os.unlink(tmp_video)
                            self.log.debug("video_to_sfx: cleaned up temp video file")
                    except Exception:
                        pass
            if not image_urls:
                return "❌ Error: No frames provided. Please include up to 4 images (frames) or attach a video URL/file so the server can extract frames."

        # Parse parameters from natural language first, then check for JSON overrides
        user_text, nl_params = self._parse_natural_language_params(user_text)
        user_text, json_overrides = self._parse_inline_overrides(user_text)
        
        # Merge: JSON overrides take precedence over natural language, then valves
        overrides = {**nl_params, **json_overrides}
        output_format = (overrides.get("output_format") or self.valves.OUTPUT_FORMAT).strip()
        loop = overrides.get("loop") if "loop" in overrides else self.valves.LOOP
        prompt_influence = overrides.get("prompt_influence") or self.valves.PROMPT_INFLUENCE

        openai_base = (self.valves.OPENAI_API_BASE_URL or "").rstrip("/")
        openai_model = self.valves.OPENAI_MODEL
        eleven_base = (self.valves.ELEVEN_API_BASE_URL or "").rstrip("/")
        self.log.debug(
            f"video_to_sfx: using openai_base={openai_base}, model={openai_model}, eleven_base={eleven_base}"
        )

        try:
            # 1) Detect video duration if enabled (do this first so prompts can reference it)
            video_duration: Optional[float] = None
            if self.valves.AUTO_DETECT_DURATION and original_video_url:
                await self.emit_status(
                    __event_emitter__, "info", "Detecting video duration..."
                )
                try:
                    tmp_video_for_probe = await self._download_temp_file(original_video_url)
                    video_duration = await self._get_video_duration(tmp_video_for_probe)
                    try:
                        if tmp_video_for_probe and os.path.exists(tmp_video_for_probe):
                            os.unlink(tmp_video_for_probe)
                    except Exception:
                        pass
                    # Cap at 30 seconds per ElevenLabs API limit
                    if video_duration and video_duration > 30:
                        self.log.info(f"Video duration {video_duration}s exceeds 30s limit, capping at 30s")
                        video_duration = 30.0
                    self.log.debug(f"video_to_sfx: detected video duration={video_duration}s")
                except Exception as e:
                    self.log.warning(f"Failed to detect video duration: {e}")

            # 2) Ask OpenAI to craft an ElevenLabs SFX prompt given frames + user text
            await self.emit_status(
                __event_emitter__, "info", "Analyzing video frames with vision model..."
            )
            sfx_prompt = await self._craft_sfx_prompt(
                openai_base, openai_key, openai_model, user_text, image_urls,
                duration_seconds=video_duration, loop=loop
            )
            if not sfx_prompt:
                return "❌ Error: Failed to create SFX prompt from frames."
            # Enforce ElevenLabs maximum text length (e.g., 450 chars)
            sfx_prompt = self._enforce_prompt_limit(sfx_prompt)

            # Additionally, create short prompts per keyframe for user transparency
            await self.emit_status(
                __event_emitter__, "info", "Creating per-frame sound variations..."
            )
            frame_prompts: List[str] = await self._craft_per_frame_prompts(
                openai_base,
                openai_key,
                openai_model,
                user_text,
                image_urls,
                self.valves.FRAME_PROMPT_MAX_CHARS,
                duration_seconds=video_duration,
                loop=loop,
            )

            # 3) Generate SFX via ElevenLabs
            await self.emit_status(
                __event_emitter__, "info", "Generating sound effects with ElevenLabs..."
            )
            audio_bytes, ext, content_type = await self._generate_sfx(
                eleven_base, eleven_key, sfx_prompt, output_format, 
                duration_seconds=video_duration, loop=loop, prompt_influence=prompt_influence
            )
            if not audio_bytes:
                return "❌ Error: Sound generation failed. Please try again."

            # 4) Save to Files DB
            user_id = __user__.get("id") if __user__ else None
            file_id = await self._save_audio_file(
                audio_bytes,
                sfx_prompt,
                output_format,
                ext,
                content_type,
                user_id,
            )

            size_mb = len(audio_bytes) / (1024 * 1024)
            dl = f"/api/v1/files/{file_id}/content" if file_id else None
            
            await self.emit_status(
                __event_emitter__, "info", "Saving generated audio..."
            )
            # Try server-side merge if we have an original video reference
            merged_file_id: Optional[str] = None
            # Also prepare per-frame merged outputs
            per_frame_merged: List[tuple[int, str]] = []  # (index, file_id)
            try:
                if not original_video_url:
                    # Re-scan user messages only
                    original_video_url = self._extract_video_url(messages)
                if original_video_url:
                    await self.emit_status(
                        __event_emitter__, "info", "Merging audio with video..."
                    )
                    merged_file_id = await self._merge_video_audio_server(
                        original_video_url,
                        audio_bytes,
                        ext,
                        content_type,
                        __user__.get("id") if __user__ else None,
                    )
                    if merged_file_id:
                        self.log.debug(
                            f"video_to_sfx: merged video saved file_id={merged_file_id}"
                        )

                    # Multi-merge: for each per-frame prompt, generate audio and merge with full original video
                    if frame_prompts:
                        await self.emit_status(
                            __event_emitter__, "info", f"Creating {len(frame_prompts)} frame-specific audio variations..."
                        )
                        for i, fp_txt in enumerate(frame_prompts, start=1):
                            await self.emit_status(
                                __event_emitter__, "info", f"Generating & merging frame {i}/{len(frame_prompts)}..."
                            )
                            try:
                                short_fp = (
                                    self._enforce_prompt_limit(fp_txt) if fp_txt else ""
                                )
                                if not short_fp:
                                    continue
                                a_bytes_i, ext_i, ct_i = await self._generate_sfx(
                                    eleven_base, eleven_key, short_fp, output_format,
                                    duration_seconds=video_duration, loop=loop, prompt_influence=prompt_influence
                                )
                                if not a_bytes_i:
                                    continue
                                merged_id_i = (
                                    await self._merge_video_audio_server_custom(
                                        original_video_url,
                                        a_bytes_i,
                                        ext_i,
                                        ct_i,
                                        __user__.get("id") if __user__ else None,
                                        label=f"frame_{i}",
                                    )
                                )
                                if merged_id_i:
                                    per_frame_merged.append((i, merged_id_i))
                            except Exception as me_i:
                                self.log.error(
                                    f"video_to_sfx: per-frame merge failed (index={i}): {me_i}"
                                )
            except Exception as me:
                self.log.error(f"video_to_sfx: merge failed: {me}")

            # JSON block output removed per request
            await self.emit_status(
                __event_emitter__, "info", "Video SFX generation complete!", True
            )
            
            if file_id and dl:
                if merged_file_id:
                    merged_url = f"/api/v1/files/{merged_file_id}/content"
                    details = ""
                    if frame_prompts:
                        details = "\n\nPer-frame prompts:\n" + "\n".join(
                            [f"- Frame {i+1}: {p}" for i, p in enumerate(frame_prompts)]
                        )
                    if per_frame_merged:
                        merged_list = "\n".join(
                            [
                                f"- [Merged video (Frame {idx})](/api/v1/files/{fid}/content)"
                                for idx, fid in per_frame_merged
                            ]
                        )
                        details += "\n\nPer-frame merged outputs:\n" + merged_list
                    return (
                        "✅ Video SFX generated and merged server-side!\n"
                        f"- [Download merged video]({merged_url})\n"
                        f"- [Download audio only]({dl}) ({size_mb:.1f}MB)\n\n"
                        f'<audio controls src="{dl}" />' + details
                    )
                else:
                    details = ""
                    if frame_prompts:
                        details = "\n\nPer-frame prompts:\n" + "\n".join(
                            [f"- Frame {i+1}: {p}" for i, p in enumerate(frame_prompts)]
                        )
                    if per_frame_merged:
                        merged_list = "\n".join(
                            [
                                f"- [Merged video (Frame {idx})](/api/v1/files/{fid}/content)"
                                for idx, fid in per_frame_merged
                            ]
                        )
                        details += "\n\nPer-frame merged outputs:\n" + merged_list
                    return (
                        f"✅ Video SFX generated! ({size_mb:.1f}MB)\n\n"
                        f'<audio controls src="{dl}" />\n\n'
                        f"[Download audio]({dl})" + details
                    )
            else:
                details = ""
                if frame_prompts:
                    details = "\n\nPer-frame prompts:\n" + "\n".join(
                        [f"- Frame {i+1}: {p}" for i, p in enumerate(frame_prompts)]
                    )
                if per_frame_merged:
                    merged_list = "\n".join(
                        [
                            f"- [Merged video (Frame {idx})](/api/v1/files/{fid}/content)"
                            for idx, fid in per_frame_merged
                        ]
                    )
                    details += "\n\nPer-frame merged outputs:\n" + merged_list
                return (
                    f"✅ Video SFX generated but failed to save. Size: {size_mb:.1f}MB"
                    + details
                )
        except Exception as e:
            self.log.exception(f"video_to_sfx pipeline error: {e}")
            error_msg = f"Pipeline error: {str(e)}"
            await self.emit_status(__event_emitter__, "error", error_msg, True)
            return f"❌ Error: {str(e)}"

    def _enforce_prompt_limit(self, text: str) -> str:
        """Trim whitespace and enforce MAX_SFX_PROMPT_CHARS limit safely."""
        try:
            max_len = int(getattr(self.valves, "MAX_SFX_PROMPT_CHARS", 450) or 450)
        except Exception:
            max_len = 450
        if not text:
            return ""
        # Normalize whitespace a bit
        cleaned = re.sub(r"\s+", " ", str(text)).strip()
        if len(cleaned) <= max_len:
            return cleaned
        trimmed = cleaned[:max_len].rstrip()
        # Prefer cutting at a sentence boundary if within the last 40 chars
        tail = trimmed[-40:]
        p = max(tail.rfind("."), tail.rfind("!"), tail.rfind("?"))
        if p > 10:
            trimmed = trimmed[: len(trimmed) - (40 - p)].rstrip()
        return trimmed

    async def _craft_per_frame_prompts(
        self,
        openai_base: str,
        openai_key: str,
        model: str,
        user_text: str,
        image_urls: List[str],
        max_len: int,
        duration_seconds: Optional[float] = None,
        loop: bool = False,
    ) -> List[str]:
        """Ask the OpenAI-compatible model to propose a short SFX micro-prompt per keyframe image.
        Returns a list aligned to image_urls length. If any error occurs, returns an empty list.
        """
        try:
            if not image_urls:
                return []
            url = f"{openai_base}/chat/completions"
            headers = {
                "Authorization": f"Bearer {openai_key}",
                "Content-Type": "application/json",
            }
            # Build a single message containing numbered frames; ask for concise outputs.
            duration_hint = ""
            if duration_seconds:
                duration_hint = f" The audio will be {duration_seconds:.1f} seconds long."
            loop_hint = ""
            if loop:
                loop_hint = " The audio should be suitable for seamless looping."
            
            content_parts: List[dict] = [
                {
                    "type": "text",
                    "text": (
                        "Create a SOUND EFFECT prompt for EACH numbered keyframe (max {max_len} chars each).\n\n"
                        "SOUND-ONLY: Describe audio, not visuals.\n\n"
                        "Use audio terms: Impact (collisions), Whoosh (movement), Ambience (atmosphere), "
                        "Braam (cinematic hit), Glitch (sci-fi), Drone (continuous texture).\n\n"
                        "Structure: Simple & clear OR sequence of events.\n\n"
                        f"{duration_hint}{loop_hint}\n\n"
                        "Return as JSON array of strings: [\"sound prompt 1\", \"sound prompt 2\", ...]"
                    ).format(max_len=max_len),
                }
            ]
            for idx, u in enumerate(image_urls, start=1):
                content_parts.append({"type": "text", "text": f"Frame {idx}:"})
                content_parts.append({"type": "image_url", "image_url": {"url": u}})
            content_parts.append(
                {"type": "text", "text": f"User description: {user_text}"}
            )

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a sound designer creating audio/sound effect prompts. Describe SOUNDS, not visuals."},
                    {"role": "user", "content": content_parts},
                ],
                "temperature": 0.4,
                "max_tokens": 300,
            }

            timeout = aiohttp.ClientTimeout(total=self.valves.TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        txt = (await resp.text())[:500]
                        self.log.error(f"per-frame prompts error {resp.status}: {txt}")
                        return []
                    data = await resp.json()
                    content = None
                    try:
                        content = data["choices"][0]["message"]["content"].strip()
                    except Exception:
                        return []
                    # Normalize possible markdown code-fence output
                    try:
                        # If returned as ```json ... ``` extract inner
                        m = re.search(r"```[a-zA-Z0-9_-]*\s*([\s\S]*?)```", content)
                        if m:
                            content = m.group(1).strip()
                    except Exception:
                        pass

                    # Try to extract a JSON array substring first
                    prompts: List[str] = []
                    parsed = False
                    try:
                        i0 = content.find("[")
                        i1 = content.rfind("]")
                        if i0 != -1 and i1 != -1 and i1 > i0:
                            arr_text = content[i0 : i1 + 1]
                            arr = json.loads(arr_text)
                            if isinstance(arr, list):
                                prompts = [str(x)[:max_len].strip() for x in arr]
                                parsed = True
                    except Exception:
                        parsed = False

                    if not parsed:
                        # Fallback: split into lines and clean bullets/quotes/commas
                        raw_lines = [
                            l.strip() for l in content.splitlines() if l.strip()
                        ]
                        cleaned_lines = []
                        for l in raw_lines:
                            # Remove leading list markers and JSON syntax hints
                            l = re.sub(r"^[\-\*\d\.)\s]+", "", l)  # bullets/numbers
                            l = l.lstrip(",")
                            l = l.strip('"')
                            if l.lower() in {"json", "[", "]"}:
                                continue
                            if l in {"[", "]"}:
                                continue
                            cleaned_lines.append(l)
                        prompts = [l[:max_len].strip() for l in cleaned_lines if l]
                    # Normalize length to images count
                    if len(prompts) < len(image_urls):
                        prompts += [""] * (len(image_urls) - len(prompts))
                    return prompts[: len(image_urls)]
        except Exception as e:
            self.log.error(f"_craft_per_frame_prompts error: {e}")
            return []

    def _extract_video_url(self, messages: List[dict]) -> Optional[str]:
        """Search latest then preceding chat messages for a usable video reference.
        Accepts:
        - Items of type 'video' or 'video_url' with url fields
        - Generic file-like items with a 'url' property
        - Nested file objects: item['file'] = { url, mime/mime_type }
        - Bare Open WebUI file API paths (/api/v1/files/{id}/content), relative or absolute
        - Direct http(s) URLs ending in common video extensions
        """
        video_exts = (".mp4", ".mov", ".webm", ".mkv", ".avi")

        def scan_message(msg: dict) -> Optional[str]:
            content = msg.get("content", "")
            candidates: List[str] = []

            # Strings: pull URLs and file API paths
            if isinstance(content, str):
                candidates.extend(re.findall(r"https?://[^\s]+", content))
                # Capture Open WebUI file API paths, allow absolute or relative
                api_matches = re.findall(
                    r"(?:https?://[^\s]+)?(/api/v1/files/[a-f0-9\-]+/content)",
                    content,
                    flags=re.IGNORECASE,
                )
                for m in api_matches:
                    candidates.append(m if m.startswith("/") else f'/{m.lstrip("/")}')

            # Arrays: inspect multimodal parts
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    t = item.get("type")
                    # Explicit video containers
                    if t in ("video", "video_url"):
                        if "video_url" in item and isinstance(item["video_url"], dict):
                            u = item["video_url"].get("url")
                            if u:
                                candidates.append(u)
                        u = item.get("url") or item.get("content")
                        if u:
                            candidates.append(u)
                    # Nested file object with potential video mime
                    if isinstance(item.get("file"), dict):
                        fobj = item["file"]
                        u = fobj.get("url")
                        mime = fobj.get("mime") or fobj.get("mime_type")
                        # Prefer explicit file id if present
                        fid = fobj.get("id") or fobj.get("file_id")
                        if isinstance(fid, str) and fid:
                            candidates.append(f"/api/v1/files/{fid}/content")
                        if isinstance(u, str) and u:
                            # Prefer if mime says it's video/*
                            if isinstance(mime, str) and mime.lower().startswith(
                                "video/"
                            ):
                                candidates.append(u)
                            else:
                                candidates.append(u)
                    # Generic file-like items with URL
                    u = item.get("url")
                    if isinstance(u, str) and u:
                        candidates.append(u)
                    # Strings inside items may contain URLs
                    for key in ("content", "text"):
                        if isinstance(item.get(key), str):
                            candidates.extend(re.findall(r"https?://[^\s]+", item[key]))
                            api_matches = re.findall(
                                r"(?:https?://[^\s]+)?(/api/v1/files/[a-f0-9\-]+/content)",
                                item[key],
                                flags=re.IGNORECASE,
                            )
                            for m in api_matches:
                                candidates.append(
                                    m if m.startswith("/") else f'/{m.lstrip("/")}'
                                )

            # Dict content: tolerate shapes like { text, file, url }
            elif isinstance(content, dict):
                # Direct url
                u = content.get("url")
                if isinstance(u, str) and u:
                    candidates.append(u)
                # File object
                if isinstance(content.get("file"), dict):
                    fobj = content["file"]
                    fid = fobj.get("id") or fobj.get("file_id")
                    if isinstance(fid, str) and fid:
                        candidates.append(f"/api/v1/files/{fid}/content")
                    u2 = fobj.get("url")
                    if isinstance(u2, str) and u2:
                        candidates.append(u2)
                # Top-level file id
                fid2 = content.get("id") or content.get("file_id")
                if isinstance(fid2, str) and fid2:
                    candidates.append(f"/api/v1/files/{fid2}/content")
                # Strings within dict
                for key in ("content", "text"):
                    if isinstance(content.get(key), str):
                        candidates.extend(re.findall(r"https?://[^\s]+", content[key]))
                        api_matches = re.findall(
                            r"(?:https?://[^\s]+)?(/api/v1/files/[a-f0-9\-]+/content)",
                            content[key],
                            flags=re.IGNORECASE,
                        )
                        for m in api_matches:
                            candidates.append(
                                m if m.startswith("/") else f'/{m.lstrip("/")}'
                            )

            # Normalize and filter
            for u in candidates:
                if not isinstance(u, str):
                    continue
                low = u.lower()
                # Open WebUI file API link (with or without /content)
                m_api = re.search(
                    r"(/api/v1/files/[a-f0-9\-]+)(/content)?", low, flags=re.IGNORECASE
                )
                if m_api:
                    base = m_api.group(1)
                    # Always normalize to /content for downstream handling
                    return base + "/content"
                # Direct video URLs by extension
                if any(low.endswith(ext) for ext in video_exts):
                    return u
            # Log candidates for diagnostics
            if candidates:
                try:
                    self.log.debug(
                        f"video_to_sfx: scanned message candidates (role={msg.get('role')}): {candidates}"
                    )
                except Exception:
                    pass
            return None

        # Search last USER messages only (avoid assistant follow-ups), preferring most recent
        for msg in reversed(messages or []):
            if msg.get("role") != "user":
                continue
            found = scan_message(msg)
            if found:
                self.log.debug(
                    f"video_to_sfx: found video reference in message role={msg.get('role')}: {found}"
                )
                return found

        # Fallback: allow assistant messages only for Open WebUI file links, to reference
        # a previously returned file attachment. This does not trigger processing by itself,
        # it only provides the file reference when the latest message is a user one.
        for msg in reversed(messages or []):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                m = re.search(
                    r"(?:https?://[^\s]+)?(/api/v1/files/[a-f0-9\-]+/content)",
                    content,
                    flags=re.IGNORECASE,
                )
                if m:
                    link = m.group(1)
                    link = link if link.startswith("/") else f'/{link.lstrip("/")}'
                    self.log.debug(
                        "video_to_sfx: fallback assistant file link used: %s", link
                    )
                    return link
        try:
            self.log.info("video_to_sfx: no video file reference found in chat history")
        except Exception:
            pass
        return None

    async def _download_temp_file(self, url: str) -> str:
        """Download or resolve a URL to a temporary local file path.
        - For internal Open WebUI file URLs (/api/v1/files/{id}/content or absolute variants), resolve via Files + Storage.
        - Otherwise, HTTP GET the URL into a temp file.
        """
        # Try resolving internal file URL first
        # Normalize a bare /api/v1/files/{id} to /content form
        bare = re.search(r"^(/api/v1/files/[a-f0-9\-]+)$", url, flags=re.IGNORECASE)
        if bare:
            url = bare.group(1) + "/content"
        self.log.debug(f"video_to_sfx: attempting internal resolution for url={url}")
        local_path = self._try_resolve_internal_file(url)
        if local_path:
            # Copy to temp path to ensure lifecycle management
            suffix = Path(local_path).suffix or ""
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp_path = tmp.name
            shutil.copyfile(local_path, tmp_path)
            self.log.debug(
                f"video_to_sfx: resolved internal file to local={local_path}, temp_copy={tmp_path}"
            )
            return tmp_path

        # Fallback: HTTP download
        suffix = Path(url).suffix or ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
        timeout = aiohttp.ClientTimeout(total=self.valves.TIMEOUT)
        headers = {}
        async with aiohttp.ClientSession(timeout=timeout) as session:
            self.log.debug(
                f"video_to_sfx: downloading external URL via HTTP GET: {url}"
            )
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    txt = (await resp.text())[:500]
                    raise RuntimeError(
                        f"Failed to download video ({resp.status}): {txt}"
                    )
                data = await resp.read()
                with open(tmp_path, "wb") as f:
                    f.write(data)
        self.log.debug(f"video_to_sfx: downloaded external file to temp={tmp_path}")
        return tmp_path

    def _try_resolve_internal_file(self, url: str) -> Optional[str]:
        """If URL looks like /api/v1/files/{id}/content (optionally with host), resolve to local path via Files + Storage."""
        try:
            # Accept absolute or relative
            m = re.search(r"/api/v1/files/([\w-]+)/content", url)
            if not m:
                return None
            file_id = m.group(1)
            file_rec = FilesDB.get_file_by_id(file_id)
            if not file_rec or not file_rec.path:
                return None
            # Ensure local path via storage provider
            return Storage.get_file(file_rec.path)
        except Exception:
            return None

    async def _extract_frames_ffmpeg(
        self, video_path: str, seconds: List[int]
    ) -> List[str]:
        """Use ffmpeg (via imageio-ffmpeg) to extract frames to PNG files and return their paths."""
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        self.log.debug(f"video_to_sfx: using ffmpeg_exe={ffmpeg_exe}")
        out_paths: List[str] = []
        # Create a temp dir for frames
        frames_dir = Path(tempfile.mkdtemp(prefix="frames_"))
        try:
            # Use fps-based extraction to get up to 4 frames regardless of duration
            pattern = frames_dir / "frame_%02d.png"
            cmd = [
                ffmpeg_exe,
                "-y",
                "-i",
                video_path,
                "-vf",
                "fps=1",
                "-vframes",
                "4",
                str(pattern),
            ]
            self.log.debug(
                f"video_to_sfx: running ffmpeg fps-extract cmd: {' '.join(cmd)}"
            )
            import subprocess as _sub

            completed = await asyncio.to_thread(_subprocess_run_wrapper, cmd)
            err_txt = (
                completed.stderr.decode(errors="ignore") if completed.stderr else ""
            )
            # Collect generated files
            generated = sorted([str(p) for p in frames_dir.glob("frame_*.png")])[:4]
            if completed.returncode != 0 or not generated:
                self.log.error(f"video_to_sfx: ffmpeg stderr: {err_txt[:500]}")
                raise RuntimeError(f"ffmpeg failed extracting frames: {err_txt[:300]}")
            out_paths.extend(generated)
            self.log.debug(f"video_to_sfx: extracted frames => {out_paths}")
            return out_paths
        except Exception:
            # Cleanup any partials
            for p in out_paths:
                try:
                    if os.path.exists(p):
                        os.unlink(p)
                except Exception:
                    pass
            raise

    async def _merge_video_audio_server(
        self,
        video_url: str,
        audio_bytes: bytes,
        audio_ext: str,
        audio_mime: str,
        user_id: Optional[str],
    ) -> Optional[str]:
        """Resolve/download the original video, mux with generated audio, save merged MP4 to Files DB, and return file_id."""
        tmp_video = None
        tmp_audio = None
        tmp_out = None
        try:
            # Prepare temp files
            tmp_video = await self._download_temp_file(video_url)
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=audio_ext or ".mp3"
            ) as ta:
                ta.write(audio_bytes)
                tmp_audio = ta.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as to:
                tmp_out = to.name

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            self.log.debug(f"video_to_sfx: merge using ffmpeg_exe={ffmpeg_exe}")
            cmd = [
                ffmpeg_exe,
                "-y",
                "-i",
                tmp_video,
                "-i",
                tmp_audio,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                tmp_out,
            ]
            self.log.debug(f"video_to_sfx: running merge cmd: {' '.join(cmd)}")
            completed = await asyncio.to_thread(_subprocess_run_wrapper, cmd)
            err_txt = (
                completed.stderr.decode(errors="ignore") if completed.stderr else ""
            )
            if (
                completed.returncode != 0
                or not os.path.exists(tmp_out)
                or os.path.getsize(tmp_out) == 0
            ):
                self.log.error(f"video_to_sfx: merge ffmpeg stderr: {err_txt[:500]}")
                return None

            # Upload merged file
            from open_webui.models.files import Files, FileForm

            timestamp = int(time.time())
            filename = f"video_with_sfx_{timestamp}.mp4"
            with open(tmp_out, "rb") as f:
                file_data, file_path = Storage.upload_file(
                    f,
                    filename,
                    {"content_type": "video/mp4", "source": "video_to_sfx_merge"},
                )
            merged_id = str(uuid.uuid4())
            record = Files.insert_new_file(
                user_id or "system",
                FileForm(
                    id=merged_id,
                    filename=filename,
                    path=file_path,
                    meta={
                        "name": filename,
                        "content_type": "video/mp4",
                        "size": os.path.getsize(tmp_out),
                        "source": "video_to_sfx_merge",
                    },
                ),
            )
            return record.id if record else None
        except Exception as e:
            self.log.error(f"video_to_sfx: _merge_video_audio_server error: {e}")
            return None
        finally:
            for p in (tmp_video, tmp_audio, tmp_out):
                try:
                    if p and os.path.exists(p):
                        os.unlink(p)
                except Exception:
                    pass

    async def _merge_video_audio_server_custom(
        self,
        video_url: str,
        audio_bytes: bytes,
        audio_ext: str,
        audio_mime: str,
        user_id: Optional[str],
        label: str = "",
    ) -> Optional[str]:
        """Same as _merge_video_audio_server but allows a custom label in the filename/meta."""
        tmp_video = None
        tmp_audio = None
        tmp_out = None
        try:
            tmp_video = await self._download_temp_file(video_url)
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=audio_ext or ".mp3"
            ) as ta:
                ta.write(audio_bytes)
                tmp_audio = ta.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as to:
                tmp_out = to.name

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe,
                "-y",
                "-i",
                tmp_video,
                "-i",
                tmp_audio,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                tmp_out,
            ]
            completed = await asyncio.to_thread(_subprocess_run_wrapper, cmd)
            if (
                completed.returncode != 0
                or not os.path.exists(tmp_out)
                or os.path.getsize(tmp_out) == 0
            ):
                return None

            from open_webui.models.files import Files, FileForm

            timestamp = int(time.time())
            safe_label = re.sub(r"[^a-zA-Z0-9_\-]", "_", label or "custom")
            filename = f"video_with_sfx_{safe_label}_{timestamp}.mp4"
            with open(tmp_out, "rb") as f:
                file_data, file_path = Storage.upload_file(
                    f,
                    filename,
                    {"content_type": "video/mp4", "source": "video_to_sfx_merge"},
                )
            merged_id = str(uuid.uuid4())
            record = Files.insert_new_file(
                user_id or "system",
                FileForm(
                    id=merged_id,
                    filename=filename,
                    path=file_path,
                    meta={
                        "name": filename,
                        "content_type": "video/mp4",
                        "size": os.path.getsize(tmp_out),
                        "source": "video_to_sfx_merge",
                        "label": label,
                    },
                ),
            )
            return record.id if record else None
        except Exception:
            return None
        finally:
            for p in (tmp_video, tmp_audio, tmp_out):
                try:
                    if p and os.path.exists(p):
                        os.unlink(p)
                except Exception:
                    pass

    def _file_to_data_url(self, path: str) -> str:
        with open(path, "rb") as f:
            b = f.read()
        b64 = base64.b64encode(b).decode()
        return f"data:image/png;base64,{b64}"

    def _extract_user_text_and_images(
        self, messages: List[dict], max_images: int = 4
    ) -> Tuple[str, List[str]]:
        """Return (text, image_urls) from the last user message. Supports text-only and content array messages."""
        text = ""
        urls: List[str] = []
        for msg in reversed(messages or []):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts_text = []
                for item in content:
                    t = item.get("type")
                    if t == "text":
                        parts_text.append(item.get("text", ""))
                    elif t in ("image_url", "image"):
                        # image_url style: { type: 'image_url', image_url: { url: '...' } }
                        u = None
                        if "image_url" in item and isinstance(item["image_url"], dict):
                            u = item["image_url"].get("url")
                        if not u:
                            u = item.get("url")
                        if u:
                            urls.append(u)
                text = "\n".join([p for p in parts_text if p]).strip()
            break
        # Trim to max_images
        urls = urls[: max_images or 4]
        return text.strip(), urls

    def _parse_natural_language_params(self, text: str) -> tuple[str, dict]:
        """Extract looping and prompt influence from natural language."""
        params: dict = {}
        cleaned = text or ""
        
        # Extract looping: "with looping", "loop enabled", "seamless loop", "looping", "that loops"
        loop_patterns = [
            r'\b(?:with\s+)?(?:seamless\s+)?loop(?:ing)?(?:\s+enabled)?\b',
            r'\b(?:enable|turn\s+on)\s+loop(?:ing)?\b',
            r'\b(?:make\s+it\s+)?loop(?:able)?\b',
            r'\bthat\s+loops\b',
        ]
        for pattern in loop_patterns:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                params["loop"] = True
                # Remove the matched text
                cleaned = cleaned[:match.start()] + cleaned[match.end():]
                break
        
        # Extract prompt influence: "very literal", "creative", "strict", "prompt influence 0.8"
        influence_patterns = [
            (r'prompt\s+influence\s+(\d+(?:\.\d+)?)', lambda m: float(m.group(1))),
            (r'\b(?:very\s+)?literal(?:\s+interpretation)?\b', lambda m: 0.9),
            (r'\bstrict(?:ly)?\b', lambda m: 0.9),
            (r'\bcreative(?:\s+interpretation)?\b', lambda m: 0.3),
            (r'\bloose(?:ly)?\b', lambda m: 0.3),
        ]
        for pattern, value_fn in influence_patterns:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                try:
                    influence = value_fn(match)
                    if 0.0 <= influence <= 1.0:
                        params["prompt_influence"] = influence
                        # Remove the matched text
                        cleaned = cleaned[:match.start()] + cleaned[match.end():]
                        break
                except (ValueError, AttributeError):
                    pass
        
        # Clean up extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned, params
    
    def _parse_inline_overrides(self, text: str) -> tuple[str, dict]:
        overrides: dict = {}
        cleaned = text or ""
        try:
            json_match = re.findall(r"\{[^{}]*\}", cleaned, flags=re.DOTALL)
            if json_match:
                candidate = json_match[-1]
                try:
                    data = json.loads(candidate)
                    if isinstance(data, dict):
                        for k in ("output_format", "loop", "prompt_influence"):
                            if k in data and data[k] is not None:
                                overrides[k] = data[k]
                        if overrides:
                            cleaned = cleaned.replace(candidate, "", 1).strip()
                            return cleaned, overrides
                except Exception:
                    pass
        except Exception:
            pass
        return cleaned, overrides

    async def _craft_sfx_prompt(
        self,
        openai_base: str,
        openai_key: str,
        model: str,
        user_text: str,
        image_urls: List[str],
        duration_seconds: Optional[float] = None,
        loop: bool = False,
    ) -> Optional[str]:
        """
        Use an OpenAI-compatible chat endpoint with vision input to craft a concise, high-quality ElevenLabs SFX prompt.
        """
        url = f"{openai_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {openai_key}",
            "Content-Type": "application/json",
        }
        duration_hint = ""
        if duration_seconds:
            duration_hint = f" The audio will be {duration_seconds:.1f} seconds long, so consider appropriate pacing and development."
        loop_hint = ""
        if loop:
            loop_hint = " The audio should work for seamless looping, with a continuous feel rather than a definitive ending."
        
        content_parts = [
            {
                "type": "text",
                "text": (
                    "You are an expert sound designer creating prompts for ElevenLabs Sound Effects API. "
                    "Based on the video frames and user description, write a single concise sound effect prompt (max 400 characters).\n\n"
                    "SOUND-ONLY FOCUS: Describe ONLY audio/sounds, NOT visuals.\n\n"
                    "AUDIO TERMINOLOGY (use when appropriate):\n"
                    "• Impact: Collision/contact sounds\n"
                    "• Whoosh: Movement through air\n"
                    "• Ambience: Background environmental atmosphere\n"
                    "• Braam: Cinematic trailer hit\n"
                    "• Glitch: Malfunction/sci-fi sounds\n"
                    "• Drone: Continuous textured atmosphere\n\n"
                    "STRUCTURE:\n"
                    "• Simple: Clear, concise (e.g., 'Thunder rumbling in the distance')\n"
                    "• Complex: Sequence of events (e.g., 'Wind whistling, then leaves rustling')\n"
                    "• Musical: Include BPM, key if relevant\n\n"
                    f"{duration_hint}{loop_hint}\n"
                    "Output only the sound prompt, no commentary."
                ),
            }
        ]
        for u in image_urls:
            content_parts.append({"type": "image_url", "image_url": {"url": u}})
        content_parts.append({"type": "text", "text": f"User description: {user_text}"})

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional sound designer. Create audio/sound effect prompts that describe SOUNDS, not visuals. Think like a Foley artist.",
                },
                {"role": "user", "content": content_parts},
            ],
            "temperature": 0.6,
            "max_tokens": 250,
        }

        timeout = aiohttp.ClientTimeout(total=self.valves.TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    txt = (await resp.text())[:500]
                    self.log.error(f"OpenAI vision error {resp.status}: {txt}")
                    return None
                data = await resp.json()
                try:
                    # OpenAI format
                    return (
                        data["choices"][0]["message"]["content"].strip()
                        if data.get("choices")
                        else None
                    )
                except Exception:
                    return None

    def _guess_ext_and_mime(self, output_format: str) -> tuple[str, str]:
        fmt = (output_format or "").lower()
        if fmt.startswith("mp3"):
            return ".mp3", "audio/mpeg"
        if fmt.startswith("wav"):
            return ".wav", "audio/wav"
        if fmt.startswith("pcm"):
            return ".pcm", "audio/L16"
        return ".mp3", "audio/mpeg"

    async def _get_video_duration(self, video_path: str) -> Optional[float]:
        """Use ffmpeg to detect video duration in seconds (works with imageio-ffmpeg)."""
        try:
            # Use ffmpeg directly since imageio-ffmpeg provides it (but not ffprobe)
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            self.log.debug(f"Using ffmpeg for duration detection: {ffmpeg_exe}")
            
            # Use ffmpeg -i to get file info, parse duration from stderr
            cmd = [
                ffmpeg_exe,
                "-i", video_path,
                "-f", "null",
                "-"
            ]
            
            completed = await asyncio.to_thread(_subprocess_run_wrapper, cmd)
            
            # ffmpeg outputs file info to stderr
            stderr = completed.stderr.decode(errors='ignore') if completed.stderr else ''
            
            # Parse duration from stderr
            # Format: "Duration: 00:00:10.50, start: 0.000000, bitrate: ..."
            duration_match = re.search(r'Duration:\s*(\d+):(\d+):(\d+\.\d+)', stderr)
            if duration_match:
                hours = int(duration_match.group(1))
                minutes = int(duration_match.group(2))
                seconds = float(duration_match.group(3))
                duration = hours * 3600 + minutes * 60 + seconds
                self.log.debug(f"Detected video duration: {duration}s")
                return duration if duration > 0 else None
            else:
                self.log.warning(f"Could not parse duration from ffmpeg output: {stderr[:300]}")
            return None
        except Exception as e:
            self.log.warning(f"Failed to detect video duration: {e}")
            return None

    async def _generate_sfx(
        self,
        api_base: str,
        api_key: str,
        text: str,
        output_format: str,
        duration_seconds: Optional[float] = None,
        loop: bool = False,
        prompt_influence: Optional[float] = None,
    ) -> tuple[Optional[bytes], str, str]:
        """
        Call ElevenLabs Sound Effects API and return (audio_bytes, file_ext, content_type).
        Endpoint: POST {api_base}/sound-generation
        Headers: { "xi-api-key": api_key, "Content-Type": "application/json" }
        Query: output_format=<format>
        Body: { "text": <prompt>, "duration_seconds": <optional>, "loop": <bool>, "prompt_influence": <optional> }
        """
        url = f"{api_base}/sound-generation"
        params = {}
        if output_format:
            params["output_format"] = output_format

        payload = {"text": text}
        if duration_seconds is not None:
            payload["duration_seconds"] = duration_seconds
        if loop:
            payload["loop"] = loop
        if prompt_influence is not None:
            payload["prompt_influence"] = prompt_influence
        
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        }

        timeout = aiohttp.ClientTimeout(total=self.valves.TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                url, headers=headers, json=payload, params=params
            ) as resp:
                if resp.status != 200:
                    txt = (await resp.text())[:500]
                    self.log.error(f"ElevenLabs SFX error {resp.status}: {txt}")
                    return None, "", ""

                ctype = (
                    resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
                )
                try:
                    if ctype == "application/json":
                        data = await resp.json()
                        b64 = data.get("audio_base64") or data.get("audio") or None
                        if isinstance(b64, str):
                            audio_bytes = base64.b64decode(b64)
                            ext, mime = self._guess_ext_and_mime(output_format)
                            return audio_bytes, ext, mime
                        file_url = data.get("url") or data.get("audio_url")
                        if file_url:
                            async with session.get(file_url) as r2:
                                if r2.status == 200:
                                    blob = await r2.read()
                                    ext, mime = self._guess_ext_and_mime(output_format)
                                    return blob, ext, mime
                                else:
                                    self.log.error(
                                        f"Failed to fetch audio URL: {r2.status}"
                                    )
                                    return None, "", ""
                    else:
                        blob = await resp.read()
                        ext, mime = self._guess_ext_and_mime(output_format)
                        return blob, ext, mime
                except Exception as e:
                    self.log.error(f"Failed parsing SFX response: {e}")
                    return None, "", ""

    async def _save_audio_file(
        self,
        audio_data: bytes,
        prompt: str,
        output_format: str,
        ext: str,
        content_type: str,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        try:
            from open_webui.models.files import Files, FileForm
            from open_webui.storage.provider import Storage

            safe_prompt = (
                "".join(c for c in prompt[:40] if c.isalnum() or c in (" ", "-", "_"))
                .rstrip()
                .replace(" ", "_")
            )
            timestamp = int(time.time())
            filename = f"video_sfx_{safe_prompt}_{timestamp}{ext}"

            # Temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name
            try:
                # Upload via Storage
                with open(tmp_path, "rb") as f:
                    file_data, file_path = Storage.upload_file(
                        f,
                        filename,
                        {"content_type": content_type, "source": "video_to_sfx"},
                    )

                # DB record
                file_id = str(uuid.uuid4())
                record = Files.insert_new_file(
                    user_id or "system",
                    FileForm(
                        id=file_id,
                        filename=filename,
                        path=file_path,
                        meta={
                            "name": filename,
                            "content_type": content_type,
                            "size": len(audio_data),
                            "source": "video_to_sfx",
                            "output_format": output_format,
                            "prompt": prompt[:200],
                        },
                    ),
                )
                if record:
                    return record.id
                return None
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        except Exception as e:
            self.log.error(f"Failed to save audio file: {e}")
            return None
