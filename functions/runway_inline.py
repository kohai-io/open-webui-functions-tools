"""
title: Runway Video Generation (Inline)
author: open-webui
version: 3.0.0
license: MIT
description: Standalone Runway text/image-to-video Pipe with Seedance 2.5, Gen-4.5 and Gen-4 Turbo, optional option dialogs, confirmation, and user-owned Files API players. Enable iframe Sandbox Allow Same Origin for playback.
requirements: httpx, cryptography, pydantic
"""

import asyncio
import base64
import hashlib
import io
import ipaddress
import json
import logging
import mimetypes
import os
import re
import time
from html import escape
from typing import Any, Literal, Optional
from urllib.parse import urlsplit
from uuid import UUID

import httpx
from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import core_schema


class EncryptedStr(str):
    """Compatible with the previous Pipe's encrypted credential valve."""
    @staticmethod
    def _key():
        secret = os.getenv("WEBUI_SECRET_KEY")
        return base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest()) if secret else None

    @classmethod
    def encrypt(cls, value):
        if not value or value.startswith("encrypted:") or not cls._key():
            return value
        return "encrypted:" + Fernet(cls._key()).encrypt(value.encode()).decode()

    @classmethod
    def decrypt(cls, value):
        if not value.startswith("encrypted:"):
            return value
        try:
            if not cls._key():
                raise ValueError("WEBUI_SECRET_KEY is unavailable")
            return Fernet(cls._key()).decrypt(value[10:].encode()).decode()
        except (InvalidToken, ValueError):
            raise ValueError("Cannot decrypt RUNWAY_API_KEY. Restore WEBUI_SECRET_KEY or re-enter the key.") from None

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler):
        return core_schema.no_info_after_validator_function(
            lambda v: cls(cls.encrypt(v)), core_schema.str_schema(),
            serialization=core_schema.to_string_ser_schema(),
        )


class UserCancelled(Exception):
    pass


class RunwayError(Exception):
    def __init__(self, message, status=None, retry_after=5):
        super().__init__(message)
        self.status, self.retry_after = status, retry_after


# Model pixel ratios from Runway's published OpenAPI schema, checked 2026-09-11.
SHAPES = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
SEEDANCE_RATIOS = {
    "480p": dict(zip(SHAPES, ("992:432", "854:480", "752:560", "640:640", "560:752", "480:854"))),
    "720p": dict(zip(SHAPES, ("1470:630", "1280:720", "1112:834", "960:960", "834:1112", "720:1280"))),
    "1080p": dict(zip(SHAPES, ("2206:946", "1920:1080", "1664:1248", "1440:1440", "1248:1664", "1080:1920"))),
}
GEN4_RATIOS = dict(zip(SHAPES, ("1584:672", "1280:720", "1104:832", "960:960", "832:1104", "720:1280")))
MODELS = {"seedance2_5": "Seedance 2.5", "gen4.5": "Runway Gen-4.5", "gen4_turbo": "Runway Gen-4 Turbo"}
IMAGE_MIMES = {"image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/webp": ".webp"}


class Pipe:
    class Valves(BaseModel):
        RUNWAY_API_KEY: EncryptedStr = Field(default="", description="Runway developer API key. Falls back to RUNWAYML_API_SECRET. Encrypted when WEBUI_SECRET_KEY is set.")
        API_BASE_URL: str = Field(default="https://api.dev.runwayml.com/v1", description="Runway developer API base. Must use HTTPS.")
        MODEL: Literal["seedance2_5", "gen4.5", "gen4_turbo"] = "seedance2_5"
        DURATION: int = Field(default=5, ge=2, le=30, description="Default seconds. Seedance: 4–30; Gen-4.5/Turbo: 2–10.")
        RATIO: str = Field(default="1280:720", description="Default model pixel ratio, e.g. 1280:720 or 1920:1080. User Valves override shape and resolution.")
        REQUIRE_CONFIRMATION: bool = Field(default=True, description="Require a positive OWUI confirmation before sending the prompt or image to Runway. API-only calls cannot confirm.")
        EVENT_TIMEOUT: int = Field(default=180, ge=10, le=1800, description="Maximum seconds per dialog. Timeout cancels generation.")
        POLL_INTERVAL: int = Field(default=5, ge=5, le=60)
        MAX_POLL_TIME: int = Field(default=900, ge=30, le=3600)
        DOWNLOAD_TIMEOUT: int = Field(default=180, ge=10, le=600)
        MAX_INPUT_MB: int = Field(default=20, ge=1, le=200, description="Maximum image bytes read from OWUI or data URIs.")
        MAX_VIDEO_MB: int = Field(default=256, ge=1, le=1024, description="Maximum bytes downloaded per video. OWUI's upload limit also applies.")
        ENABLE_STATUS_INDICATOR: bool = True

    class UserValves(BaseModel):
        MODEL: Literal["default", "seedance2_5", "gen4.5", "gen4_turbo"] = "default"
        DURATION: int = Field(default=0, ge=0, le=30, description="Seconds, or 0 to use the administrator default.")
        ASPECT_RATIO: Literal["default", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"] = "default"
        RESOLUTION: Literal["default", "480p", "720p", "1080p"] = "default"
        AUDIO: bool = Field(default=True, description="Generate audio with Seedance 2.5. Gen-4 models do not accept this option.")
        ASK_OPTIONS: bool = Field(default=False, description="Show model, format, duration and audio selection dialogs before confirmation.")

    def __init__(self):
        self.name = "Runway Video (Inline)"
        self.valves = self.Valves()
        self.log = logging.getLogger("runway_inline")

    async def _status(self, emitter, message, done=False):
        if emitter and self.valves.ENABLE_STATUS_INDICATOR:
            try:
                await emitter({"type": "status", "data": {"description": message, "done": done}})
            except Exception:
                self.log.warning("Status delivery failed")

    def _options(self, context):
        raw = (context or {}).get("valves")
        if isinstance(raw, BaseModel):
            raw = raw.model_dump()
        prefs = self.UserValves.model_validate(raw or {})
        model = self.valves.MODEL if prefs.MODEL == "default" else prefs.MODEL
        choices = [(res, shape) for res, ratios in SEEDANCE_RATIOS.items() for shape, px in ratios.items() if px == self.valves.RATIO]
        choices += [("720p", shape) for shape, px in GEN4_RATIOS.items() if px == self.valves.RATIO]
        if not choices:
            raise ValueError("Set the administrator RATIO to a supported pixel ratio such as 1280:720.")
        resolution, shape = choices[0]
        return {
            "model": model, "duration": prefs.DURATION or self.valves.DURATION,
            "aspect_ratio": shape if prefs.ASPECT_RATIO == "default" else prefs.ASPECT_RATIO,
            "resolution": resolution if prefs.RESOLUTION == "default" else prefs.RESOLUTION,
            "audio": prefs.AUDIO if model == "seedance2_5" else False,
        }, prefs.ASK_OPTIONS

    @staticmethod
    def _formats(model, has_image):
        if model == "seedance2_5":
            return SEEDANCE_RATIOS
        ratios = GEN4_RATIOS if has_image else {s: GEN4_RATIOS[s] for s in ("16:9", "9:16")}
        return {"720p": ratios}

    def _payload(self, prompt, image, options):
        model = options["model"]
        if model not in MODELS:
            raise ValueError("Select Seedance 2.5, Gen-4.5 or Gen-4 Turbo.")
        if model == "gen4_turbo" and not image:
            raise ValueError("Gen-4 Turbo requires an image. Attach one or select Seedance 2.5 / Gen-4.5.")
        minimum, maximum = (4, 30) if model == "seedance2_5" else (2, 10)
        duration = options["duration"]
        if isinstance(duration, bool) or not isinstance(duration, int) or not minimum <= duration <= maximum:
            raise ValueError(f"{MODELS[model]} requires {minimum}–{maximum} whole seconds.")
        try:
            ratio = self._formats(model, bool(image))[options["resolution"]][options["aspect_ratio"]]
        except KeyError:
            raise ValueError("This model/mode does not support the selected format. Enable ASK_OPTIONS or change the User Valves.") from None
        limit = 15000 if model == "seedance2_5" else 1000
        if len(prompt.encode("utf-16-le")) // 2 > limit:
            raise ValueError(f"This model accepts a prompt of at most {limit} UTF-16 characters. Shorten the prompt.")
        if not prompt and (not image or model == "gen4.5"):
            raise ValueError("Describe the video you want to generate.")
        payload = {"model": model, "ratio": ratio, "duration": duration}
        if prompt:
            payload["promptText"] = prompt
        if model == "seedance2_5":
            payload["audio"] = options["audio"]
        if image:
            payload["promptImage"] = [{"uri": image, "position": "first"}]
        return ("image_to_video" if image else "text_to_video"), payload

    async def _ask(self, caller, event):
        if caller is None:
            raise UserCancelled("Interactive options/confirmation need a connected Open WebUI chat. No generation was started.")
        try:
            result = await asyncio.wait_for(caller(event), timeout=self.valves.EVENT_TIMEOUT)
        except asyncio.TimeoutError:
            raise UserCancelled("The dialog timed out. No generation was started.") from None
        except Exception:
            raise UserCancelled("The dialog could not reach your browser. No generation was started.") from None
        if result is False or result is None or isinstance(result, dict):
            raise UserCancelled("The dialog was cancelled or disconnected. No generation was started.")
        return result

    async def _choose(self, caller, title, choices, current):
        values = [v for v, label in choices]
        result = await self._ask(caller, {"type": "input", "data": {
            "title": title, "message": "Choose a setting for this generation.",
            "input": {"type": "select", "options": [{"value": v, "label": label} for v, label in choices]},
            "value": current if current in values else values[0],
        }})
        if not isinstance(result, str) or result not in values:
            raise UserCancelled("No valid option was selected. No generation was started.")
        return result

    async def _choose_options(self, caller, options, has_image):
        options = dict(options)
        options["model"] = await self._choose(caller, "Video model", [
            (m, label) for m, label in MODELS.items() if has_image or m != "gen4_turbo"
        ], options["model"])
        selected = await self._choose(caller, "Video format", [
            (f"{res}|{shape}", f"{shape} · {res} ({px})")
            for res, ratios in self._formats(options["model"], has_image).items() for shape, px in ratios.items()
        ], f"{options['resolution']}|{options['aspect_ratio']}")
        options["resolution"], options["aspect_ratio"] = selected.split("|")
        lo, hi = (4, 30) if options["model"] == "seedance2_5" else (2, 10)
        options["duration"] = int(await self._choose(caller, "Video duration", [
            (str(i), f"{i} seconds") for i in range(lo, hi + 1)
        ], str(options["duration"])))
        if options["model"] == "seedance2_5":
            options["audio"] = await self._choose(caller, "Generated audio", [
                ("yes", "Include audio"), ("no", "Silent video")
            ], "yes" if options["audio"] else "no") == "yes"
        else:
            options["audio"] = False
        return options

    @staticmethod
    def _input(body, files):
        message = next((m for m in reversed(body.get("messages", [])) if m.get("role") == "user"), {})
        content = message.get("content", "")
        text, refs = [], []
        if isinstance(content, str):
            text.append(content)
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    text.append(part.get("text", ""))
                elif part.get("type") in ("image_url", "input_image"):
                    ref = part.get("image_url") or part.get("url")
                    refs.append(ref.get("url") if isinstance(ref, dict) else ref)
        for item in files or []:
            if not isinstance(item, dict):
                continue
            nested = item.get("file") or {}
            meta = nested.get("meta") or item.get("meta") or {}
            mime = item.get("content_type") or item.get("mime_type") or item.get("mime") or meta.get("content_type") or ""
            if item.get("type") == "image" or mime.startswith("image/"):
                fid = item.get("id") or nested.get("id")
                refs.append(f"/api/v1/files/{fid}/content" if fid else item.get("url") or nested.get("url"))
            elif item.get("type") == "file" or mime:
                raise ValueError("Attach a PNG, JPEG or WebP image for image-to-video. Other attachment types are not supported.")
        refs.extend(message.get("images") or [])
        prompt = "\n".join(text).strip()
        def extract(match):
            refs.append(match.group(1))
            return ""
        prompt = re.sub(r"!\[[^\]]*\]\(([^\s)]+)\)", extract, prompt)
        prompt = re.sub(r"(?:https?://[^\s/]+)?(/(?:[^\s/]+/)*api/v1/files/[a-fA-F0-9-]+/content)(?:\?[^\s]*)?", extract, prompt)
        unique = {}
        for ref in refs:
            if not isinstance(ref, str) or not ref:
                raise ValueError("An attached image could not be read. Attach it again.")
            match = re.search(r"/api/v1/files/([a-fA-F0-9-]+)/content", ref)
            unique[match.group(1) if match else ref] = ref
        if len(unique) > 1:
            raise ValueError("Attach one first-frame image per generation. Multiple-image references are not supported by this Pipe.")
        return prompt.strip(), next(iter(unique.values()), None)

    def _inline_options(self, prompt, options, has_image):
        """Retain the previous Pipe's explicit trailing JSON / runway-tag overrides."""
        match = re.search(r"\s*(\{[^{}]*\}|<runway\s+[^>]+/?>)\s*$", prompt, re.IGNORECASE)
        if not match:
            return prompt, options
        block = match.group(1)
        if block.startswith("{"):
            try:
                overrides = json.loads(block)
            except ValueError:
                return prompt, options
            if not set(overrides).intersection({"model", "duration", "ratio", "audio"}):
                return prompt, options
            if set(overrides) - {"model", "duration", "ratio", "audio"}:
                raise ValueError("Inline options support only model, duration, ratio and audio.")
        else:
            overrides = dict(re.findall(r'(model|duration|ratio|audio)\s*=\s*"([^"]*)"', block))
            if "duration" in overrides:
                if not overrides["duration"].isdigit():
                    raise ValueError("Inline duration must be a whole number.")
                overrides["duration"] = int(overrides["duration"])
            if "audio" in overrides:
                if overrides["audio"] not in ("true", "false"):
                    raise ValueError("Inline audio must be true or false.")
                overrides["audio"] = overrides["audio"] == "true"
        result = {**options, **{k: v for k, v in overrides.items() if k != "ratio"}}
        if result["model"] not in MODELS or not isinstance(result["audio"], bool):
            raise ValueError("Invalid inline model or audio option.")
        if "ratio" in overrides:
            ratio = overrides["ratio"]
            if ratio in SHAPES:
                result["aspect_ratio"] = ratio
            else:
                found = [(res, shape) for res, shapes in self._formats(result["model"], has_image).items() for shape, px in shapes.items() if px == ratio]
                if not found:
                    raise ValueError("Unsupported inline pixel ratio for this model/mode.")
                result["resolution"], result["aspect_ratio"] = found[0]
        if result["model"] != "seedance2_5":
            result["audio"] = False
        return prompt[:match.start()].strip(), result

    def _check_image(self, data, mime):
        if mime not in IMAGE_MIMES:
            raise ValueError("Use PNG, JPEG or WebP; GIF and SVG are not supported.")
        if not data or len(data) > self.valves.MAX_INPUT_MB * 1024 * 1024:
            raise ValueError(f"The image must be non-empty and at most {self.valves.MAX_INPUT_MB} MB.")
        return data, mime

    async def _read_image(self, ref, user):
        if not ref:
            return None
        match = re.search(r"/api/v1/files/([a-fA-F0-9-]+)/content(?:[?#]|$)", ref)
        if match:
            from open_webui.models.files import Files
            from open_webui.storage.provider import Storage
            from open_webui.utils.access_control.files import has_access_to_file
            item = await Files.get_file_by_id(match.group(1))
            if not item or not (item.user_id == user.id or user.role == "admin" or await has_access_to_file(item.id, "read", user)):
                raise ValueError("The attached image is unavailable or you do not have access to it.")
            mime = (item.meta or {}).get("content_type") or mimetypes.guess_type(item.filename)[0]
            path = await asyncio.to_thread(Storage.get_file, item.path)
            def read():
                with open(path, "rb") as f:
                    return f.read(self.valves.MAX_INPUT_MB * 1024 * 1024 + 1)
            return self._check_image(await asyncio.to_thread(read), mime)
        if ref.startswith("data:"):
            if len(ref) > (self.valves.MAX_INPUT_MB * 1024 * 1024 * 4 // 3) + 100:
                raise ValueError("The encoded image exceeds the configured size limit.")
            match = re.fullmatch(r"data:(image/[\w.+-]+);base64,(.+)", ref, re.DOTALL)
            if not match:
                raise ValueError("The image data URI is invalid.")
            try:
                data = base64.b64decode(match.group(2), validate=True)
            except ValueError:
                raise ValueError("The image contains invalid base64 data.") from None
            return self._check_image(data, match.group(1))
        if ref.startswith("runway://") and 13 <= len(ref) <= 5000:
            return ref
        parts = urlsplit(ref)
        if parts.scheme != "https" or not parts.hostname or parts.username or parts.password or len(ref) > 2048:
            raise ValueError("Image references must be OWUI files, data URIs, Runway uploads or public HTTPS URLs.")
        try:
            ipaddress.ip_address(parts.hostname)
        except ValueError:
            pass
        else:
            raise ValueError("Runway image URLs must use a domain name, not an IP address.")
        return ref  # Runway fetches this; do not download arbitrary URLs on the OWUI server.

    async def _api(self, client, method, path, payload=None):
        # Never retry generation POSTs: a lost response may still have created a paid task.
        try:
            response = await client.request(method, path, json=payload)
        except httpx.HTTPError:
            if method == "POST" and path in ("text_to_video", "image_to_video"):
                raise RunwayError("The submission response was lost. A task may have been created; check Runway before retrying.") from None
            raise RunwayError("Runway could not be reached.", status=503) from None
        if response.is_error or response.is_redirect:
            guidance = {400: "Runway rejected these inputs or settings.", 401: "Check RUNWAY_API_KEY.",
                        403: "Check your Runway API access and model permissions.", 404: "The Runway resource was not found.",
                        429: "Runway's rate or credit limit was reached."}.get(response.status_code, "Runway could not complete the request.")
            try:
                delay = max(5, min(60, float(response.headers.get("Retry-After", "5"))))
            except ValueError:
                delay = 5
            raise RunwayError(f"{guidance} (HTTP {response.status_code})", response.status_code, delay)
        if method == "DELETE":
            return {}
        try:
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError()
            return data
        except ValueError:
            raise RunwayError("Runway returned an invalid response. Check the task in Runway before retrying.") from None

    async def _prepare_image(self, image, api):
        if image is None or isinstance(image, str):
            return image
        data, mime = image
        uri = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        if len(uri) <= 5 * 1024 * 1024:
            return uri
        upload = await self._api(api, "POST", "uploads", {"filename": "frame" + IMAGE_MIMES[mime], "type": "ephemeral"})
        if urlsplit(upload.get("uploadUrl", "")).scheme != "https" or not str(upload.get("runwayUri", "")).startswith("runway://"):
            raise RunwayError("Runway returned an invalid upload location.")
        # The signed storage upload must not receive the Runway bearer token.
        async with httpx.AsyncClient(timeout=self.valves.DOWNLOAD_TIMEOUT) as media:
            response = await media.post(upload["uploadUrl"], data=upload["fields"], files={"file": ("frame" + IMAGE_MIMES[mime], data, mime)})
            if not response.is_success:
                raise RunwayError("The temporary image upload failed. No generation was started.")
        return upload["runwayUri"]

    async def _poll(self, api, task_id, emitter):
        deadline = time.monotonic() + self.valves.MAX_POLL_TIME
        last_status = None
        while time.monotonic() < deadline:
            delay = self.valves.POLL_INTERVAL
            try:
                task = await self._api(api, "GET", f"tasks/{task_id}")
                status = task.get("status")
                if status == "SUCCEEDED":
                    if not isinstance(task.get("output"), list) or not task["output"]:
                        raise RunwayError("Runway finished without any video outputs.")
                    return task
                if status == "FAILED":
                    code = re.sub(r"[^A-Za-z0-9_.-]", "", str(task.get("failureCode", "unknown")))[:100]
                    raise RunwayError(f"Runway generation failed ({code}). Check this task in Runway for details.")
                if status == "CANCELLED":
                    raise RunwayError("Runway cancelled the generation.")
                if status not in ("PENDING", "THROTTLED", "RUNNING"):
                    raise RunwayError("Runway returned an unexpected task status.")
                if status != last_status:
                    await self._status(emitter, f"Runway: {status.lower()} · task {task_id}")
                    last_status = status
            except RunwayError as error:
                if error.status != 429 and not (error.status and error.status >= 500):
                    raise
                delay = max(delay, error.retry_after)
            await asyncio.sleep(min(delay, max(0, deadline - time.monotonic())))
        raise RunwayError("The wait limit was reached. The task may still finish in Runway; check it before generating again.")

    async def _download(self, uri):
        if not isinstance(uri, str) or urlsplit(uri).scheme != "https":
            raise RunwayError("Runway returned an invalid video URL.")
        limit = self.valves.MAX_VIDEO_MB * 1024 * 1024
        # A separate unauthenticated client keeps API credentials away from the CDN.
        async with httpx.AsyncClient(timeout=self.valves.DOWNLOAD_TIMEOUT, follow_redirects=True) as media:
            async with media.stream("GET", uri) as response:
                response.raise_for_status()
                mime = response.headers.get("Content-Type", "").split(";", 1)[0]
                if mime not in ("video/mp4", "application/octet-stream"):
                    raise RunwayError("The output is not an MP4 video.")
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(data) + len(chunk) > limit:
                        raise RunwayError("The video exceeds MAX_VIDEO_MB.")
                    data.extend(chunk)
                if len(data) < 12 or data[4:8] != b"ftyp":
                    raise RunwayError("The output does not contain an MP4 file header.")
                return bytes(data)

    async def _save(self, data, request, user, metadata, options, task_id, index):
        from fastapi import UploadFile
        from starlette.datastructures import Headers
        from open_webui.routers.files import upload_file_handler
        from open_webui.models.chats import Chats
        filename = f"runway_{options['model']}_{task_id}_{index}.mp4"
        upload = UploadFile(file=io.BytesIO(data), filename=filename, headers=Headers({"content-type": "video/mp4"}))
        try:
            record = await upload_file_handler(request, file=upload, user=user, process=False, metadata={
                "source": "runway_inline", "task_id": task_id, "generation_options": options,
                "result_index": index, "chat_id": metadata.get("chat_id"), "message_id": metadata.get("message_id"),
            })
        finally:
            await upload.close()
        if not record or not record.id:
            raise RunwayError("The Files API did not return a saved file.")
        chat, message = metadata.get("chat_id"), metadata.get("message_id")
        if chat and message and not str(chat).startswith(("local:", "temporary:", "channel:")):
            try:
                await Chats.insert_chat_files(chat_id=chat, message_id=message, file_ids=[record.id], user_id=user.id)
            except Exception:
                self.log.warning("Saved video could not be linked to the chat")
        root = request.scope.get("root_path", "").rstrip("/")
        url = root + str(request.app.url_path_for("get_file_content_by_id", id=record.id))
        return url, record.filename

    async def _deliver(self, task, request, user, metadata, options, emitter):
        saved, links, failures = [], [], []
        for index, uri in enumerate(task["output"], 1):
            await self._status(emitter, f"Saving video {index} of {len(task['output'])}...")
            try:
                data = await self._download(uri)
                url, filename = await self._save(data, request, user, metadata, options, task["id"], index)
                saved.append(self._video_embed_html(url, filename))
                links.append(f"[Download video {index}]({url}?attachment=true)")
            except Exception:
                self.log.warning("Could not save Runway task %s output %s", task["id"], index)
                failures.append(f"Video {index} could not be saved; check storage limits and retrieve it from Runway before its output URL expires.")
        if saved and emitter:
            try:
                await emitter({"type": "embeds", "data": {"embeds": saved}})
            except Exception:
                failures.append("Inline players could not be delivered. Use the saved download links.")
        await self._status(emitter, f"{len(saved)} video(s) saved" if saved else "No videos could be saved", True)
        return "\n\n".join([f"{len(saved)} video(s) saved.", *links, *failures, f"Runway task: `{task['id']}`"])

    async def pipe(self, body: dict, __user__: Optional[dict] = None, __request__: Any = None,
                   __event_emitter__: Any = None, __event_call__: Any = None,
                   __files__: Optional[list] = None, __metadata__: Optional[dict] = None,
                   __task__: Optional[str] = None) -> str:
        metadata = {**(body.get("metadata") or {}), **(__metadata__ or {})}
        if __task__ or metadata.get("task"):
            return ""
        task_id = None
        try:
            from open_webui.models.users import Users
            uid = (__user__ or {}).get("id")
            user = await Users.get_user_by_id(uid) if uid else None
            if user is None or __request__ is None:
                raise ValueError("An authenticated Open WebUI user and request are required.")
            options, ask_options = self._options(__user__)
            prompt, ref = self._input(body, __files__ if __files__ is not None else metadata.get("files", body.get("files", [])))
            image = await self._read_image(ref, user)
            prompt, options = self._inline_options(prompt, options, image is not None)
            if ask_options:
                options = await self._choose_options(__event_call__, options, image is not None)
            endpoint, _ = self._payload(prompt, "pending-image" if image is not None else None, options)
            key = EncryptedStr.decrypt(str(self.valves.RUNWAY_API_KEY)) or os.getenv("RUNWAYML_API_SECRET", "")
            if not key:
                raise ValueError("Set RUNWAY_API_KEY or RUNWAYML_API_SECRET to a Runway developer API key.")
            base = self.valves.API_BASE_URL.rstrip("/") + "/"
            parts = urlsplit(base)
            if parts.scheme != "https" or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
                raise ValueError("API_BASE_URL must be a valid HTTPS API base URL.")
            if self.valves.REQUIRE_CONFIRMATION:
                audio = ("audio on" if options["audio"] else "silent") if options["model"] == "seedance2_5" else "silent"
                answer = await self._ask(__event_call__, {"type": "confirmation", "data": {
                    "title": "Generate this video with Runway?",
                    "message": f"{MODELS[options['model']]} · {options['duration']} seconds · {options['aspect_ratio']} · {options['resolution']} · {audio}. "
                    + ("Use the attached image as the first frame. " if image is not None else "Text-to-video. ")
                    + "Your prompt and selected image will be sent to Runway. This generation uses Runway credits.\n\n"
                    + (prompt[:600] + ("…" if len(prompt) > 600 else "") or "Animate the attached image."),
                }})
                if answer is not True:
                    raise UserCancelled("Generation was not confirmed. No task was started.")
            async with httpx.AsyncClient(base_url=base, timeout=60, headers={
                "Authorization": f"Bearer {key}", "X-Runway-Version": "2024-11-06",
            }) as api:
                await self._status(__event_emitter__, "Preparing the Runway request...")
                prepared = await self._prepare_image(image, api)
                endpoint, payload = self._payload(prompt, prepared, options)
                submitted = await self._api(api, "POST", endpoint, payload)
                try:
                    task_id = str(UUID(submitted["id"]))
                except (KeyError, ValueError, TypeError, AttributeError):
                    raise RunwayError("Runway did not return a valid task ID. Check Runway before retrying.") from None
                await self._status(__event_emitter__, f"Runway task submitted: {task_id}")
                try:
                    task = await self._poll(api, task_id, __event_emitter__)
                except asyncio.CancelledError:
                    try:
                        await asyncio.wait_for(self._api(api, "DELETE", f"tasks/{task_id}"), timeout=10)
                    except Exception:
                        self.log.warning("Could not cancel Runway task %s", task_id)
                    raise
                task["id"] = task_id
                return await self._deliver(task, __request__, user, metadata, options, __event_emitter__)
        except (UserCancelled, ValueError, RunwayError) as error:
            await self._status(__event_emitter__, "Runway request stopped", True)
            return str(error) + (f"\n\nRunway task: `{task_id}`" if task_id else "")
        except Exception:
            self.log.warning("Runway request failed%s", f" for task {task_id}" if task_id else "")
            await self._status(__event_emitter__, "Runway request failed", True)
            return "The Runway request could not be completed. Check the server configuration and Runway task status before retrying." + (f"\n\nRunway task: `{task_id}`" if task_id else "")

    @staticmethod
    def _video_embed_html(content_url: str, filename: str) -> str:
        # Keep only a durable file URL in chat history, never video bytes or credentials.
        # Attribute escaping also prevents a filename/URL from injecting executable HTML.
        return '''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body { margin: 0; font: 14px system-ui, sans-serif; color: #666; }
video { display: block; width: 100%; max-height: 560px; background: #000; border-radius: 12px; }
video[hidden] { display: none; }
#status { margin: 8px 0; }
</style></head><body>
<video id="video" controls playsinline preload="metadata" hidden></video>
<p id="status" role="status">Loading video…</p>
<a id="download" href="__CONTENT_URL__?attachment=true" data-content-url="__CONTENT_URL__"
   download="__FILENAME__" target="_blank" rel="noopener">Download video</a>
<noscript>Enable iframe scripts to play the video, or use the download link below the player.</noscript>
<script>
(() => {
  const video = document.getElementById('video');
  const status = document.getElementById('status');
  const download = document.getElementById('download');
  let objectUrl;
  const controller = new AbortController();
  const reportHeight = () => parent.postMessage({
    type: 'iframe:height', height: document.documentElement.scrollHeight
  }, '*');
  new ResizeObserver(reportHeight).observe(document.body);
  window.addEventListener('load', reportHeight);
  window.addEventListener('pagehide', () => {
    controller.abort();
    if (objectUrl) URL.revokeObjectURL(objectUrl);
  }, { once: true });
  video.addEventListener('error', () => {
    status.textContent = 'This browser could not play the video. Use the download link.';
    reportHeight();
  });
  async function loadVideo() {
    try {
      // A srcdoc iframe with the default sandbox has an opaque origin.
      // Do not try reading the parent's token or changing its settings.
      if (window.origin === 'null') {
        status.textContent = 'To play this saved video, enable Settings → Interface → iframe Sandbox Allow Same Origin, then reopen the chat. You can also use the download link below this player.';
        return;
      }
      const url = new URL(download.dataset.contentUrl, document.baseURI);
      if (url.origin !== window.location.origin && url.origin !== window.origin) {
        throw new Error('The video must be served by this Open WebUI instance.');
      }
      const response = await fetch(url.href, {
        credentials: 'same-origin', signal: controller.signal
      });
      if (response.status === 401 || response.status === 403) {
        throw new Error('Sign in to Open WebUI with an account that can access this video.');
      }
      if (!response.ok) throw new Error('The saved video is unavailable (HTTP ' + response.status + ').');
      const blob = await response.blob();
      if (!blob.type.startsWith('video/')) throw new Error('The file response is not a video.');
      objectUrl = URL.createObjectURL(blob);
      video.src = objectUrl;
      video.hidden = false;
      download.href = objectUrl;
      status.textContent = '';
    } catch (error) {
      if (error.name !== 'AbortError') status.textContent = error.message + ' Use the download link below the player.';
    } finally { reportHeight(); }
  }
  loadVideo();
})();
</script></body></html>'''.replace('__CONTENT_URL__', escape(content_url, quote=True)).replace('__FILENAME__', escape(filename, quote=True))
