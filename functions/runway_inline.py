"""
title: RunwayML Image-to-Video Pipeline (Inline Display)
author: open-webui
date: 2025-09-27
version: 2.1
license: MIT
description: A pipeline for generating videos from images using RunwayML's image-to-video API with inline video display
requirements: aiohttp, cryptography, pydantic

USAGE
- Attach an image (drag/drop) or reference an Open WebUI file URL like /api/v1/files/{id}/content.
- Provide a text instruction describing camera motion, look, and style.
- The generated video will be displayed inline in the chat message.

PER-MESSAGE OVERRIDES
You can override key parameters directly in your prompt (the block is stripped before sending to Runway):
- JSON block at the end:
  {"duration": 10, "ratio": "16:9", "model": "gen4_turbo"}
- Tag form:
  <runway duration="10" ratio="16:9" model="gen4_turbo" />
Keys supported: duration, ratio, model.
"""

from typing import Optional, Callable, Awaitable, Any
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from cryptography.fernet import Fernet, InvalidToken
import time
import aiohttp
import asyncio
import json
import os
import base64
import hashlib
import logging
import re
import mimetypes
import tempfile
import uuid
from pydantic_core import core_schema
from open_webui.routers.images import upload_image, load_b64_image_data
from open_webui.routers.files import upload_file_handler
from open_webui.models.users import Users
from open_webui.models.files import Files as FilesDB
from open_webui.storage.provider import Storage
from fastapi import UploadFile
import io

# Simplified encryption implementation with automatic handling
class EncryptedStr(str):
    """A string type that automatically handles encryption/decryption"""
    @classmethod
    def _get_encryption_key(cls) -> Optional[bytes]:
        """Generate encryption key from WEBUI_SECRET_KEY if available"""
        secret = os.getenv("WEBUI_SECRET_KEY")
        if not secret:
            return None
        hashed_key = hashlib.sha256(secret.encode()).digest()
        return base64.urlsafe_b64encode(hashed_key)

    @classmethod
    def encrypt(cls, value: str) -> str:
        """Encrypt a string value if a key is available"""
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
        """Decrypt an encrypted string value if a key is available"""
        if not value or not value.startswith("encrypted:"):
            return value
        key = cls._get_encryption_key()
        if not key:
            return value[len("encrypted:"):]
        try:
            encrypted_part = value[len("encrypted:"):]
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
        """Get the decrypted value"""
        return self.decrypt(self)


class Pipe:
    class Valves(BaseModel):
        RUNWAY_API_KEY: EncryptedStr = Field(
            default="",
            description="RunwayML API key for authentication. Get it from https://app.runwayml.com/",
        )
        API_BASE_URL: str = Field(
            default="https://api.dev.runwayml.com/v1",
            description="RunwayML API base URL.",
        )
        MODEL: str = Field(
            default="gen4_turbo",
            description="The RunwayML model to use (gen4_turbo, gen4).",
        )
        DURATION: int = Field(
            default=5,
            description="Video duration in seconds (1-10).",
        )
        RATIO: str = Field(
            default="1280:720",
            description="Video aspect ratio (1280:720, 768:768, 512:768, etc.).",
        )
        POLL_INTERVAL: int = Field(
            default=5,
            description="Seconds between status checks for video generation (default: 5).",
        )
        MAX_POLL_TIME: int = Field(
            default=300,
            description="Maximum time in seconds to wait for video generation (default: 300 = 5 minutes).",
        )
        EMIT_INTERVAL: float = Field(
            default=0.5, description="Interval in seconds between status emissions"
        )
        ENABLE_STATUS_INDICATOR: bool = Field(
            default=True, description="Enable or disable status indicator emissions"
        )
        PROGRESS_UPDATE_INTERVALS: str = Field(
            default="15,30,60,120",
            description="Comma-separated seconds for progressive status updates (e.g., '15,30,60,120')"
        )

    def __init__(self):
        self.name = "RunwayML Image-to-Video (Inline)"
        self.valves = self.Valves()
        self.last_emit_time = 0
        self.log = logging.getLogger("runway_inline_pipeline")
        self.log.setLevel(logging.INFO)

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
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
        __event_call__: Callable[[dict], Awaitable[dict]] = None,
    ) -> Optional[dict]:
        """Main pipeline method for RunwayML image-to-video generation with inline display.

        Returns the full body dict with an appended assistant message so the
        frontend can render attached files (video) properly, consistent with
        other pipelines like nano_banana_chat.
        """
        await self.emit_status(
            __event_emitter__, "info", "Initializing RunwayML video generation..."
        )
        
        # Decrypt API key
        api_key = self.valves.RUNWAY_API_KEY.get_decrypted()
        if not api_key:
            error_msg = "RunwayML API key not configured. Please set RUNWAY_API_KEY in the pipeline settings."
            await self.emit_status(__event_emitter__, "error", f"❌ Error: {error_msg}", True)
            self.log.error(error_msg)
            body["messages"].append({"role": "assistant", "content": f"❌ Error: {error_msg}"})
            return body

        # Extract prompt and image from the latest USER message (guarded)
        messages = body.get("messages", [])
        prompt = self._extract_latest_user_prompt(messages)
        image_ref = None

        if messages:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        # Handle multimodal content (text + image)
                        for item in content:
                            if item.get("type") == "image_url":
                                image_ref = item.get("image_url", {}).get("url", "")
                            elif item.get("type") in {"file", "input_image", "input_file"}:
                                # Accept generic file items if they look like images
                                meta = item.get("file", {}) or item
                                url = meta.get("url") or meta.get("image_url", {}).get("url", "")
                                mime = meta.get("mime") or meta.get("mime_type")
                                if url and (not mime or str(mime).startswith("image/")):
                                    image_ref = url
                    break

        if not prompt:
            error_msg = "No prompt provided for video generation."
            await self.emit_status(__event_emitter__, "error", f"❌ Error: {error_msg}", True)
            body["messages"].append(
                {"role": "assistant", "content": f"❌ Error: {error_msg}"}
            )
            return f"❌ Error: {error_msg}"

        if not image_ref:
            # Fallback: search previous messages for image references
            for prev in reversed(messages[:-1]):
                c = prev.get("content", "")
                if isinstance(c, str):
                    urls = self._extract_images_from_content(c)
                    if urls:
                        image_ref = urls[0]
                        break
                elif isinstance(c, list):
                    # Support multimodal older messages
                    for item in c:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            u = item.get("image_url", {}).get("url", "")
                            if u:
                                image_ref = u
                                break
                    if image_ref:
                        break

        if not image_ref:
            error_msg = "No image provided for video generation. Please upload an image."
            await self.emit_status(__event_emitter__, "error", f"❌ Error: {error_msg}", True)
            body["messages"].append(
                {"role": "assistant", "content": f"❌ Error: {error_msg}"}
            )
            return f"❌ Error: {error_msg}"

        # Resolve image reference
        await self.emit_status(
            __event_emitter__, "info", "Processing reference image..."
        )
        prompt_image = self._resolve_prompt_image(image_ref)
        if not prompt_image:
            error_msg = "The provided image cannot be accessed by RunwayML. Please provide a public http(s) URL or use a file saved in Open WebUI."
            await self.emit_status(__event_emitter__, "error", f"❌ Error: {error_msg}", True)
            self.log.error(error_msg)
            body["messages"].append({"role": "assistant", "content": f"❌ Error: {error_msg}"})
            return body

        # Parse inline parameter overrides and normalize params
        cleaned_prompt, inline_overrides = self._parse_inline_overrides(prompt)
        prompt = cleaned_prompt
        params = self._normalize_params(inline_overrides)

        try:
            self.log.info(f"Starting RunwayML video generation with prompt: {prompt[:100]}...")

            await self.emit_status(
                __event_emitter__, "info", "Starting video generation with RunwayML..."
            )
            # Start video generation
            task_id = await self._start_video_generation(
                prompt, prompt_image, api_key, params["model"], 
                params["duration"], params["ratio"], params["api_base_url"]
            )

            if not task_id:
                raise Exception("Failed to start video generation. Please check your API key and image URL.")

            await self.emit_status(
                __event_emitter__, "info", "Video generation in progress (this may take several minutes)..."
            )
            # Poll for completion
            video_url = await self._poll_video_status(task_id, api_key, params["api_base_url"], __event_emitter__)

            if video_url:
                await self.emit_status(
                    __event_emitter__, "info", "Downloading generated video..."
                )
                # Download video bytes
                video_data = await self._download_video(video_url, api_key)

                if video_data:
                    await self.emit_status(
                        __event_emitter__, "info", "Saving video file..."
                    )
                    # Save video once via OWUI storage
                    user_id = __user__.get("id") if __user__ else None
                    saved_video_id = await self._save_video_file(video_data, prompt, task_id, user_id)

                    video_size_mb = len(video_data) / (1024 * 1024)

                    if saved_video_id:
                        await self.emit_status(
                            __event_emitter__, "info", "RunwayML video generation complete!", True
                        )
                        # Build full content URL and return a simple markdown link (known working pattern)
                        webui_base = os.getenv("WEBUI_URL", "http://localhost:8080").rstrip("/")
                        content_url = f"{webui_base}/api/v1/files/{saved_video_id}/content"

                        response_content = (
                            f"✅ RunwayML video generated! [Click here to download]({content_url}) "
                            f"({video_size_mb:.1f}MB)"
                        )

                        # Also append to messages for history, but primary return is the string content
                        body["messages"].append({
                            "role": "assistant",
                            "content": response_content,
                        })

                        self.log.info(f"Video processed: {video_size_mb:.1f}MB, saved with ID: {saved_video_id}")
                        return response_content
                    else:
                        response_content = (
                            f"✅ RunwayML video generated but failed to save. Size: {video_size_mb:.1f}MB"
                        )
                        body["messages"].append({"role": "assistant", "content": response_content})
                        return response_content
                else:
                    response_content = "❌ Error: Failed to download generated video."
                    body["messages"].append({"role": "assistant", "content": response_content})
                    return response_content
            else:
                response_content = "❌ Error: Video generation failed or timed out. Please try again."
                body["messages"].append({"role": "assistant", "content": response_content})
                self.log.error("Video generation failed - no URL returned")
                return response_content

        except Exception as e:
            error_msg = f"❌ Error: Error during video generation: {str(e)}"
            self.log.exception(error_msg)
            await self.emit_status(__event_emitter__, "error", error_msg, True)
            body["messages"].append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
            return f"❌ Error: {str(e)}"

    def _extract_latest_user_prompt(self, messages: list[dict]) -> str:
        """Return the latest user-authored text prompt that passes guard checks."""
        for msg in reversed(messages or []):
            if (msg or {}).get("role") != "user":
                continue
            content = (msg or {}).get("content", "")
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text" and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                    elif isinstance(item.get("content"), str):
                        parts.append(item["content"])  # tolerate other shapes
                text = "\n".join(p for p in parts if p)
            elif isinstance(content, dict):
                text = content.get("text") or content.get("content") or json.dumps(content)

            text = (text or "").strip()
            if not text:
                continue
            if self._is_guarded_prompt(text):
                self.log.info("Guard skipped a suspicious user prompt candidate.")
                continue
            return text
        return ""

    def _is_guarded_prompt(self, text: str) -> bool:
        """Heuristics to avoid meta/suggestion/system-like text as a generation prompt."""
        lower = (text or "").strip().lower()
        if lower.startswith("task:"):
            return True
        if lower.startswith("### task"):
            return True
        if "suggest" in lower and "follow-up" in lower:
            return True
        if re.search(r"\bsuggest\s+\d+\s*-?\s*\d*\s*", lower) and ("question" in lower or "prompt" in lower):
            return True
        if len(lower.split()) <= 1 and len(lower) < 4:
            return True
        return False

    # Include all the helper methods from the original pipeline
    async def _start_video_generation(self, prompt: str, image_url: str, api_key: str, 
                                    model: str, duration: int, ratio: str, api_base_url: str) -> Optional[str]:
        """Start the video generation process and return the task ID."""
        url = f"{api_base_url}/image_to_video"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Runway-Version": "2024-11-06",
        }
        data = {
            "promptImage": image_url,
            "promptText": prompt,
            "model": model,
            "duration": duration,
            "ratio": ratio,
            "seed": 4294967295,
            "contentModeration": {"publicFigureThreshold": "auto"},
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        task_id = result.get("id")
                        self.log.info(f"Video generation started with task ID: {task_id}")
                        return task_id
                    else:
                        error_text = await response.text()
                        self.log.error(f"Failed to start generation: {response.status} - {error_text}")
                        return None
        except Exception as e:
            self.log.error(f"Error starting video generation: {str(e)}")
            return None

    async def _poll_video_status(self, task_id: str, api_key: str, api_base_url: str, event_emitter: Optional[Callable[[dict], Awaitable[None]]] = None) -> Optional[str]:
        """Poll for video generation completion."""
        url = f"{api_base_url}/tasks/{task_id}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "X-Runway-Version": "2024-11-06",
        }

        start_time = time.time()
        
        # Progressive status update intervals from valve configuration
        try:
            status_intervals = [int(x.strip()) for x in self.valves.PROGRESS_UPDATE_INTERVALS.split(',')]
        except Exception:
            status_intervals = [15, 30, 60, 120]  # Fallback to default
        next_status_index = 0
        
        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < self.valves.MAX_POLL_TIME:
                elapsed = int(time.time() - start_time)
                
                # Show progress at progressive intervals
                if next_status_index < len(status_intervals):
                    if elapsed >= status_intervals[next_status_index]:
                        await self.emit_status(
                            event_emitter, "info", f"Still generating video... ({elapsed}s elapsed)"
                        )
                        next_status_index += 1
                elif elapsed % 120 == 0 and elapsed > 0:
                    # After 120s, update every 2 minutes
                    await self.emit_status(
                        event_emitter, "info", f"Still generating video... ({elapsed}s elapsed)"
                    )
                
                try:
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            result = await response.json()
                            status = result.get("status", "")
                            
                            if status == "SUCCEEDED":
                                video_url = result.get("output", [])
                                if video_url and len(video_url) > 0:
                                    video_url = video_url[0]
                                    self.log.info(f"Video generation completed: {video_url}")
                                    return video_url
                                else:
                                    self.log.error("No video URL in successful response")
                                    return None
                            elif status == "FAILED":
                                error_msg = result.get("failure", {}).get("reason", "Unknown error")
                                self.log.error(f"Generation failed: {error_msg}")
                                return None
                            elif status in ["PENDING", "RUNNING"]:
                                self.log.debug(f"Still processing... Status: {status}")
                        else:
                            self.log.error(f"Status check failed: {response.status}")
                except Exception as e:
                    self.log.error(f"Poll error: {str(e)}")

                await asyncio.sleep(self.valves.POLL_INTERVAL)

        self.log.error(f"Video generation timed out after {self.valves.MAX_POLL_TIME} seconds")
        return None

    async def _download_video(self, video_url: str, api_key: str) -> Optional[bytes]:
        """Download the video from RunwayML."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "X-Runway-Version": "2024-11-06",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(video_url, headers=headers) as response:
                    if response.status == 200:
                        video_data = await response.read()
                        self.log.info(f"Video downloaded successfully, size: {len(video_data)} bytes")
                        return video_data
                    else:
                        self.log.error(f"Failed to download video: {response.status}")
                        return None
        except Exception as e:
            self.log.error(f"Error downloading video: {str(e)}")
            return None

    async def _save_video_file(self, video_data: bytes, prompt: str, task_id: str, user_id: str = None) -> Optional[str]:
        """Save video data using Open WebUI's file system."""
        try:
            from open_webui.models.files import Files, FileForm
            from open_webui.storage.provider import Storage
            
            # Create safe filename
            safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (" ", "-", "_")).rstrip()
            safe_prompt = safe_prompt.replace(" ", "_")
            timestamp = int(time.time())
            filename = f"runway_inline_{safe_prompt}_{timestamp}.mp4"
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(video_data)
                temp_file_path = temp_file.name
            
            try:
                # Upload using Open WebUI's storage system
                with open(temp_file_path, "rb") as f:
                    file_data, file_path = Storage.upload_file(
                        f, filename, {"content_type": "video/mp4", "source": "runway_inline_pipeline"}
                    )
                
                # Create database record
                file_id = str(uuid.uuid4())
                file_record = Files.insert_new_file(
                    user_id or "system",
                    FileForm(
                        id=file_id,
                        filename=filename,
                        path=file_path,
                        meta={
                            "name": filename,
                            "content_type": "video/mp4",
                            "size": len(video_data),
                            "source": "runway_inline_pipeline",
                            "prompt": prompt[:100],
                            "task_id": task_id
                        }
                    )
                )
                
                if file_record:
                    self.log.info(f"Video saved with file ID: {file_record.id}")
                    return file_record.id
                else:
                    self.log.error("Failed to create file database record")
                    return None
                    
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        except ImportError as e:
            self.log.error(f"Failed to import Open WebUI components: {str(e)}")
            return None

    def _parse_inline_overrides(self, prompt_text: str) -> tuple[str, dict]:
        """Extract inline parameter overrides from prompt."""
        overrides: dict = {}
        cleaned = prompt_text or ""
        
        try:
            # Try JSON object
            json_match = re.findall(r"\{[^{}]*\}", cleaned, flags=re.DOTALL)
            if json_match:
                candidate = json_match[-1]
                try:
                    data = json.loads(candidate)
                    if isinstance(data, dict):
                        for k in ("duration", "ratio", "model"):
                            if k in data:
                                overrides[k] = data[k]
                        if overrides:
                            cleaned = cleaned.replace(candidate, "", 1).strip()
                            return cleaned, overrides
                except Exception:
                    pass

            # Try XML-ish tag
            tag = re.search(r"<runway\s+([^>]*)/?>", cleaned, flags=re.IGNORECASE)
            if tag:
                attrs = tag.group(1)
                for key in ("duration", "ratio", "model"):
                    m = re.search(fr"{key}\s*=\s*\"([^\"]+)\"", attrs, flags=re.IGNORECASE)
                    if m:
                        overrides[key] = m.group(1)
                if overrides:
                    cleaned = cleaned.replace(tag.group(0), "").strip()
        except Exception:
            pass
            
        return cleaned, overrides

    def _normalize_params(self, inline_overrides: dict) -> dict:
        """Normalize parameters from valves and inline overrides."""
        # Model
        model = (inline_overrides.get("model") or self.valves.MODEL).strip() if isinstance(inline_overrides.get("model"), str) else self.valves.MODEL
        allowed_models = {"gen4_turbo", "gen4"}
        if model not in allowed_models:
            model = self.valves.MODEL

        # Duration
        try:
            dur_val = inline_overrides.get("duration", self.valves.DURATION)
            duration = int(dur_val)
        except Exception:
            duration = int(self.valves.DURATION)
        duration = max(1, min(10, duration))
        if model == "gen4_turbo":
            duration = 5 if duration <= 7 else 10

        # Ratio
        ratio_input = inline_overrides.get("ratio", self.valves.RATIO)
        ratio_str = str(ratio_input or "").strip().lower()
        
        alias_map = {"square": "1:1", "portrait": "9:16", "landscape": "16:9"}
        ratio = alias_map.get(ratio_str, ratio_str)
        ratio = re.sub(r"\s*:\s*", ":", ratio) if ratio else ratio

        # Validate ratio for model
        allowed_ratios_turbo = {"1280:720", "720:1280", "1104:832", "832:1104", "960:960", "1584:672"}
        allowed_ratios_all = allowed_ratios_turbo | {"16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "768:1280", "1280:768"}
        allowed_ratios = allowed_ratios_turbo if model == "gen4_turbo" else allowed_ratios_all

        if ratio not in allowed_ratios:
            if model == "gen4_turbo" and ratio in {"16:9", "9:16", "1:1"}:
                ratio = {"16:9": "1280:720", "9:16": "720:1280", "1:1": "960:960"}[ratio]
            else:
                ratio = self.valves.RATIO

        api_base_url = (self.valves.API_BASE_URL or "").rstrip("/")

        return {"model": model, "duration": duration, "ratio": ratio, "api_base_url": api_base_url}

    def _resolve_prompt_image(self, image_ref: str) -> Optional[str]:
        """Resolve image reference to form acceptable by Runway."""
        try:
            if not image_ref or not isinstance(image_ref, str):
                return None

            # Already a data URI
            if image_ref.startswith("data:"):
                return image_ref

            # Try to resolve Open WebUI file ID
            m = re.search(r"/api/v1/files/([a-f0-9\-]+)/content", image_ref, re.IGNORECASE)
            if m:
                file_id = m.group(1)
                file_model = FilesDB.get_file_by_id(file_id)
                if file_model and file_model.path:
                    local_path = Storage.get_file(file_model.path)
                    with open(local_path, "rb") as f:
                        data_bytes = f.read()
                    mime_type = None
                    if file_model.meta and isinstance(file_model.meta, dict):
                        mime_type = file_model.meta.get("content_type") or file_model.meta.get("mime_type")
                    if not mime_type:
                        mime_type = mimetypes.guess_type(file_model.filename or local_path)[0] or "image/png"
                    b64 = base64.b64encode(data_bytes).decode("utf-8")
                    return f"data:{mime_type};base64,{b64}"

            # Accept direct http(s) URL
            if image_ref.startswith("http://") or image_ref.startswith("https://"):
                return image_ref

            return None
        except Exception as e:
            self.log.error(f"Failed to resolve prompt image: {e}")
            return None

    def _extract_images_from_content(self, content: str) -> list[str]:
        """Extract image URLs from markdown or HTML content."""
        urls: list[str] = []
        try:
            # Markdown images
            md = re.findall(r'!\[.*?\]\((data:image/[^)]+|https?://[^)]+|/[^)]+)\)', content)
            urls.extend(md)

            # HTML <img src>
            html = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', content, flags=re.IGNORECASE)
            urls.extend(html)

            # Bare Open WebUI file API paths
            file_api = re.findall(r'(?:https?://[^\s)]+)?(/api/v1/files/[a-f0-9\-]+/content)', content, flags=re.IGNORECASE)
            for m in file_api:
                urls.append(m if m.startswith('/') else f'/{m.lstrip("/")}')

            # Bare absolute image URLs
            abs_img = re.findall(r'https?://[^\s)]+\.(?:png|jpg|jpeg|webp|gif)(?:\?[^\s)]*)?', content, flags=re.IGNORECASE)
            urls.extend(abs_img)

            # Deduplicate
            seen = set()
            deduped = []
            for u in urls:
                if u not in seen:
                    seen.add(u)
                    deduped.append(u)
            return deduped
        except Exception:
            return urls
