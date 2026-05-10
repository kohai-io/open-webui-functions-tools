"""
title: OpenAI GPT Image 2 (Chat Pipe)
author: open-webui
date: 2026-04-21
version: 1.2
license: MIT
description: A chat pipe for image generation and editing with OpenAI GPT Image 2 using conversation-aware reference images, sticky output settings, and Open WebUI image persistence.
requirements: openai, cryptography, requests
"""

import asyncio
import base64
import hashlib
import json
import mimetypes
import os
import re
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import requests
from cryptography.fernet import Fernet, InvalidToken
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import core_schema

from open_webui.models.files import Files as FilesDB
from open_webui.models.users import Users
from open_webui.routers.images import get_image_data, upload_image
from open_webui.storage.provider import Storage


class EncryptedStr(str):
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
            return value[len("encrypted:") :]

        try:
            encrypted_part = value[len("encrypted:") :]
            f = Fernet(key)
            decrypted = f.decrypt(encrypted_part.encode())
            return decrypted.decode()
        except (InvalidToken, Exception):
            return value

    def get_decrypted(self) -> str:
        return self.decrypt(self)

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


class Pipe:
    class Valves(BaseModel):
        api_key: EncryptedStr = Field(
            default="",
            description="OpenAI API key for GPT Image 2. Stored encrypted when WEBUI_SECRET_KEY is configured.",
        )
        api_base_url: str = Field(
            default="https://api.openai.com/v1",
            description="OpenAI-compatible API base URL. Defaults to the official OpenAI v1 endpoint.",
        )
        api_version: str = Field(
            default="",
            description="Optional API version query parameter for OpenAI-compatible endpoints that require it.",
        )
        model_name: str = Field(
            default="gpt-image-2",
            description="Image model name to use for generation and edits.",
        )
        max_history_images: int = Field(
            default=4,
            description="Maximum number of reference images to include from current uploads and chat history.",
        )
        enable_iterative: bool = Field(
            default=True,
            description="Enable reuse of prior uploaded/generated images as references for follow-up image requests.",
        )
        edit_mode: bool = Field(
            default=False,
            description="When true, use only the most recent reference image for edits. When false, include up to max_history_images references.",
        )
        edit_guidance: str = Field(
            default=(
                "You are an image editor. Modify the provided image to satisfy the instruction. "
                "Preserve subject identity, composition, camera angle, style, and lighting unless explicitly asked to change them."
            ),
            description="Guidance used when the user uploads an image without a textual edit prompt.",
        )
        size: str = Field(
            default="auto",
            description="Output size. GPT Image 2 supports auto or valid WxH values that meet model constraints.",
        )
        quality: str = Field(
            default="auto",
            description="Output quality: auto, low, medium, or high.",
        )
        background: str = Field(
            default="auto",
            description="Background mode: auto, transparent, or opaque.",
        )
        output_format: str = Field(
            default="png",
            description="Output format: png, jpeg, or webp.",
        )
        output_compression: int = Field(
            default=100,
            description="Compression percentage for jpeg/webp output. Ignored for png.",
        )
        moderation: str = Field(
            default="auto",
            description="Moderation level for GPT Image models: auto or low.",
        )
        mask_filename_hint: str = Field(
            default="mask",
            description="Uploaded image files whose filename contains this substring will be treated as the edit mask.",
        )
        debug: bool = Field(
            default=False,
            description="Enable verbose debug logging.",
        )
        download_timeout: int = Field(
            default=20,
            description="Timeout in seconds when downloading images from URLs.",
        )
        image_request_timeout: int = Field(
            default=180,
            description="Read timeout in seconds for OpenAI image generation and edit requests.",
        )
        api_connect_timeout: int = Field(
            default=20,
            description="Connection timeout in seconds for OpenAI image API requests.",
        )
        retry_attempts: int = Field(
            default=3,
            description="Maximum retry attempts for transient image downloads.",
        )
        retry_backoff_base: float = Field(
            default=1.5,
            description="Exponential backoff multiplier between image download retries.",
        )
        ENABLE_STATUS_INDICATOR: bool = Field(
            default=True,
            description="Enable status indicator emissions.",
        )
        EMIT_INTERVAL: float = Field(
            default=0.5,
            description="Minimum seconds between status emissions.",
        )
        ENABLE_CUSTOM_FOLLOW_UPS: bool = Field(
            default=True,
            description="Enable custom follow-up suggestion generation for image workflows.",
        )
        FOLLOW_UP_MODEL: str = Field(
            default="gpt-4o-mini",
            description="Chat model used to generate follow-up image prompts.",
        )
        FOLLOW_UP_TIMEOUT: int = Field(
            default=45,
            description="Timeout in seconds for follow-up generation.",
        )
        FOLLOW_UP_PROMPT_TEMPLATE: str = Field(
            default="",
            description="Optional custom prompt template for follow-up generation. Leave empty to use the built-in template.",
        )

    def __init__(self):
        self.id = "gpt_image_2_chat"
        self.name = "GPT Image 2"
        self.valves = self.Valves()
        self.last_emit_time = 0.0

    async def on_startup(self):
        print("[gpt_image_2_chat] Pipe loaded successfully - version 1.0")
        print("[gpt_image_2_chat] Pipe type: Default (inline display)")
        print("[gpt_image_2_chat] Features: generation, multi-image references, mask edits, sticky settings")
        print(f"[gpt_image_2_chat] Debug mode: {self.valves.debug}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    def _debug(self, msg: str) -> None:
        if self.valves.debug:
            print(f"[gpt_image_2_chat][DEBUG] {msg}")

    async def emit_status(
        self,
        event_emitter: Optional[Callable[[dict], Awaitable[None]]],
        level: str,
        message: str,
        done: bool = False,
    ) -> None:
        if not event_emitter or not self.valves.ENABLE_STATUS_INDICATOR:
            return

        current_time = time.time()
        if current_time - self.last_emit_time < self.valves.EMIT_INTERVAL and not done:
            return

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
            self._debug(f"Failed to emit status: {e}")

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Any] = None,
        __event_call__: Optional[Any] = None,
        __request__: Optional[Any] = None,
        __files__: Optional[list] = None,
    ) -> Optional[Any]:
        metadata = body.get("metadata", {})
        is_follow_up_task = metadata.get("task") == "FOLLOW_UP_GENERATION"

        if is_follow_up_task and self.valves.ENABLE_CUSTOM_FOLLOW_UPS:
            return await self._generate_image_follow_ups(body)

        await self.emit_status(
            __event_emitter__, "info", "Initializing GPT Image 2..."
        )

        api_key = EncryptedStr.decrypt(self.valves.api_key)
        if not api_key:
            error_msg = "❌ Error: OpenAI API key not configured. Please set your API key in the pipeline settings."
            await self.emit_status(__event_emitter__, "error", error_msg, True)
            body.setdefault("messages", []).append(
                {"role": "assistant", "content": error_msg}
            )
            return error_msg

        try:
            messages = body.get("messages", [])
            if not messages:
                error_msg = "❌ Error: No messages provided."
                body.setdefault("messages", []).append(
                    {"role": "assistant", "content": error_msg}
                )
                return error_msg

            await self.emit_status(
                __event_emitter__, "info", "Processing your request..."
            )

            prompt, user_has_image = self._extract_latest_user_prompt_and_image_flag(
                messages
            )
            file_items = self._get_file_items(body, __files__)
            uploaded_images, uploaded_mask = self._extract_uploaded_images(file_items)

            if not prompt and (user_has_image or uploaded_images) and self.valves.edit_guidance:
                prompt = self.valves.edit_guidance

            if not prompt:
                if self._is_system_followup(messages):
                    self._debug("Ignoring system follow-up instruction")
                    return None

                error_msg = "❌ Error: No prompt provided for image generation."
                body.setdefault("messages", []).append(
                    {"role": "assistant", "content": error_msg}
                )
                return error_msg

            await self.emit_status(
                __event_emitter__,
                "info",
                "Building context from conversation history...",
            )

            reference_images = self._collect_reference_images(
                messages=messages,
                request=__request__,
                current_images=uploaded_images,
            )

            size, quality, background = self._resolve_sticky_settings(messages, prompt)
            output_format = self._normalize_output_format(self.valves.output_format)
            output_compression = self._normalize_output_compression(
                self.valves.output_compression, output_format
            )
            moderation = self._normalize_moderation(self.valves.moderation)

            is_edit_request = bool(reference_images or uploaded_mask)
            self._debug(
                f"prompt='{prompt[:120]}', images={len(reference_images)}, mask={bool(uploaded_mask)}, edit={is_edit_request}, size={size}, quality={quality}, background={background}, output_format={output_format}"
            )

            await self.emit_status(
                __event_emitter__,
                "info",
                "Generating image with GPT Image 2...",
            )

            response_json = await self._call_image_api(
                api_key=api_key,
                prompt=prompt,
                reference_images=reference_images,
                mask=uploaded_mask,
                is_edit_request=is_edit_request,
                size=size,
                quality=quality,
                background=background,
                output_format=output_format,
                output_compression=output_compression,
                moderation=moderation,
            )

            generated_image_markdowns: List[str] = []
            revised_prompts: List[str] = []

            for item in response_json.get("data", []):
                b64_json = item.get("b64_json")
                if not b64_json:
                    continue

                revised_prompt = item.get("revised_prompt")
                if revised_prompt:
                    revised_prompts.append(revised_prompt)

                mime_type = self._mime_type_for_output_format(output_format)
                data_uri = f"data:{mime_type};base64,{b64_json}"

                try:
                    if __request__ is None or __user__ is None or not __user__.get("id"):
                        raise ValueError("Missing request or user context for image upload")

                    image_data, content_type = get_image_data(data_uri)
                    url = upload_image(
                        __request__,
                        image_data,
                        content_type,
                        {
                            "model": self.valves.model_name,
                            "prompt": prompt,
                            "size": size,
                            "quality": quality,
                            "background": background,
                            "output_format": output_format,
                            "moderation": moderation,
                            "source": "gpt_image_2_chat",
                        },
                        Users.get_user_by_id(__user__["id"]),
                    )
                    generated_image_markdowns.append(f"![Generated Image]({url})")
                except Exception as e:
                    self._debug(f"Upload failed, falling back to data URI: {e}")
                    generated_image_markdowns.append(f"![Generated Image]({data_uri})")

            if not generated_image_markdowns:
                error_msg = "❌ Error: GPT Image 2 returned no image data."
                body.setdefault("messages", []).append(
                    {"role": "assistant", "content": error_msg}
                )
                await self.emit_status(__event_emitter__, "error", error_msg, True)
                return error_msg

            response_parts = ["Here's your generated image:"]
            if revised_prompts and self.valves.debug:
                response_parts.append(
                    "\n".join(
                        [
                            f"Revised prompt {idx + 1}: {text}"
                            for idx, text in enumerate(revised_prompts)
                        ]
                    )
                )
            response_parts.append("\n".join(generated_image_markdowns))
            response_content = "\n\n".join([part for part in response_parts if part])

            await self.emit_status(
                __event_emitter__, "info", "Image generation complete!", True
            )

            body.setdefault("messages", []).append(
                {"role": "assistant", "content": response_content}
            )
            return response_content
        except Exception as e:
            error_msg = f"❌ Error: {self._format_exception(e)}"
            self._debug(error_msg)
            await self.emit_status(__event_emitter__, "error", error_msg, True)
            body.setdefault("messages", []).append(
                {"role": "assistant", "content": error_msg}
            )
            return error_msg

    async def _generate_image_follow_ups(self, body: dict) -> dict:
        try:
            messages = body.get("messages", [])
            template = self.valves.FOLLOW_UP_PROMPT_TEMPLATE or (
                """### Task:
Suggest 3-5 relevant follow-up image generation prompts that the user might want to create next, based on the image they just generated and the conversation history.

### Guidelines:
- Focus on image-specific concepts: variations, styles, zoom levels, perspectives, lighting, color palettes, compositions, moods, and alternate contexts.
- Write each suggestion from the user's perspective as a request for a new image.
- Keep prompts concise but specific.
- Use the conversation's primary language.
- Respond with JSON only.

### Output Format:
JSON: {"follow_ups": ["Prompt 1", "Prompt 2", "Prompt 3"]}

### Chat History (last 6 messages):
{{MESSAGES:END:6}}"""
            )

            recent_messages = messages[-6:] if len(messages) > 6 else messages
            messages_text = "\n".join(
                [
                    f"{msg.get('role', 'unknown')}: {self._stringify_content(msg.get('content', ''))}"
                    for msg in recent_messages
                ]
            )
            prompt = template.replace("{{MESSAGES:END:6}}", messages_text)

            api_key = EncryptedStr.decrypt(self.valves.api_key)
            if not api_key:
                return {"choices": [{"message": {"content": json.dumps({"follow_ups": []})}}]}

            client = AsyncOpenAI(
                api_key=api_key,
                base_url=self._normalized_api_base_url(),
                timeout=self.valves.FOLLOW_UP_TIMEOUT,
            )
            response = await client.chat.completions.create(
                model=self.valves.FOLLOW_UP_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You generate concise follow-up prompts for image generation workflows. Return valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )
            content = (
                response.choices[0].message.content
                if response and response.choices
                else ""
            )
            follow_ups = self._extract_follow_ups_from_text(content)
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"follow_ups": follow_ups})
                        }
                    }
                ]
            }
        except Exception as e:
            self._debug(f"Follow-up generation failed: {e}")
            return {
                "choices": [
                    {"message": {"content": json.dumps({"follow_ups": []})}}
                ]
            }

    def _extract_follow_ups_from_text(self, text: str) -> List[str]:
        if not text:
            return []

        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                parsed = json.loads(text[start:end])
                items = parsed.get("follow_ups", [])
                return [item for item in items if isinstance(item, str) and item.strip()]
        except Exception:
            pass

        lines = [line.strip("- \t") for line in text.splitlines() if line.strip()]
        return lines[:5]

    def _get_file_items(self, body: dict, __files__: Optional[list]) -> List[dict]:
        if __files__:
            return __files__
        if body.get("files"):
            return body["files"]
        metadata = body.get("metadata", {})
        return metadata.get("files", []) or []

    def _extract_uploaded_images(
        self, file_items: List[dict]
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        images: List[Dict[str, Any]] = []
        mask: Optional[Dict[str, Any]] = None
        mask_hint = (self.valves.mask_filename_hint or "mask").lower()

        for file_item in file_items or []:
            try:
                file_id = file_item.get("id") or file_item.get("file", {}).get("id")
                file_model = FilesDB.get_file_by_id(file_id) if file_id else None
                if not file_model or not file_model.path:
                    continue

                local_path = Storage.get_file(file_model.path)
                with open(local_path, "rb") as f:
                    data = f.read()

                filename = (
                    file_model.filename
                    or file_item.get("name")
                    or file_item.get("file", {}).get("meta", {}).get("name")
                    or os.path.basename(local_path)
                )
                mime_type = None
                if file_model.meta and isinstance(file_model.meta, dict):
                    mime_type = file_model.meta.get("content_type") or file_model.meta.get(
                        "mime_type"
                    )
                if not mime_type:
                    mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

                if not mime_type.startswith("image/"):
                    continue

                item = {
                    "data": data,
                    "mime_type": mime_type,
                    "filename": filename,
                }
                if mask_hint and mask_hint in filename.lower() and mask is None:
                    mask = item
                else:
                    images.append(item)
            except Exception as e:
                self._debug(f"Failed to process uploaded file item: {e}")

        return images, mask

    def _collect_reference_images(
        self,
        messages: List[Dict[str, Any]],
        request: Optional[Any] = None,
        current_images: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.valves.enable_iterative:
            return (current_images or [])[:1 if self.valves.edit_mode else self.valves.max_history_images]

        max_images = 1 if self.valves.edit_mode else max(0, int(self.valves.max_history_images))
        collected: List[Dict[str, Any]] = []
        seen_hashes = set()

        def add_image(item: Optional[Dict[str, Any]]) -> None:
            if not item or len(collected) >= max_images:
                return
            try:
                fingerprint = hashlib.sha256(item["data"]).hexdigest()
            except Exception:
                fingerprint = None
            if fingerprint and fingerprint in seen_hashes:
                return
            if fingerprint:
                seen_hashes.add(fingerprint)
            collected.append(item)

        for item in current_images or []:
            add_image(item)
            if len(collected) >= max_images:
                return collected

        for msg in reversed(messages or []):
            if len(collected) >= max_images:
                break

            role = msg.get("role")
            content = msg.get("content")
            if role == "assistant" and isinstance(content, str):
                for url in self._extract_images_from_content(content):
                    add_image(self._download_image(url, request))
                    if len(collected) >= max_images:
                        break
            elif role == "user" and isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url:
                            add_image(self._download_image(url, request))
                    if len(collected) >= max_images:
                        break

        return collected

    async def _call_image_api(
        self,
        api_key: str,
        prompt: str,
        reference_images: List[Dict[str, Any]],
        mask: Optional[Dict[str, Any]],
        is_edit_request: bool,
        size: str,
        quality: str,
        background: str,
        output_format: str,
        output_compression: Optional[int],
        moderation: str,
    ) -> Dict[str, Any]:
        payload = self._build_payload(
            prompt=prompt,
            size=size,
            quality=quality,
            background=background,
            output_format=output_format,
            output_compression=output_compression,
            moderation=moderation,
        )

        headers = {"Authorization": f"Bearer {api_key}"}
        if not is_edit_request:
            headers["Content-Type"] = "application/json"

        url = self._build_api_url(
            "/images/edits" if is_edit_request else "/images/generations"
        )
        request_timeout = (
            max(5, int(self.valves.api_connect_timeout)),
            max(60, int(self.valves.image_request_timeout)),
        )

        def do_request() -> requests.Response:
            if is_edit_request:
                files = []
                image_field_name = "image[]" if len(reference_images) > 1 else "image"
                for image in reference_images:
                    files.append(
                        (
                            image_field_name,
                            (
                                image.get("filename") or "image.png",
                                image["data"],
                                image.get("mime_type") or "image/png",
                            ),
                        )
                    )
                if mask:
                    files.append(
                        (
                            "mask",
                            (
                                mask.get("filename") or "mask.png",
                                mask["data"],
                                mask.get("mime_type") or "image/png",
                            ),
                        )
                    )
                response = requests.post(
                    url,
                    headers=headers,
                    data=payload,
                    files=files,
                    timeout=request_timeout,
                )
            else:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=request_timeout,
                )
            return response

        try:
            response = await asyncio.to_thread(do_request)
        except requests.exceptions.ReadTimeout:
            raise RuntimeError(
                f"OpenAI image request timed out after {request_timeout[1]}s. Increase image_request_timeout in the pipe settings for slower or larger GPT Image 2 generations."
            )
        except requests.exceptions.ConnectTimeout:
            raise RuntimeError(
                f"OpenAI image API connection timed out after {request_timeout[0]}s. Check network connectivity or increase api_connect_timeout in the pipe settings."
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"OpenAI image request timed out after {request_timeout[1]}s. Increase image_request_timeout in the pipe settings."
            )

        if not response.ok:
            raise RuntimeError(self._format_api_error(response))

        try:
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Image API returned invalid JSON: {e}")

    def _build_payload(
        self,
        prompt: str,
        size: str,
        quality: str,
        background: str,
        output_format: str,
        output_compression: Optional[int],
        moderation: str,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.valves.model_name,
            "prompt": prompt,
        }

        if size:
            payload["size"] = size
        if quality:
            payload["quality"] = quality
        if background:
            payload["background"] = background
        if output_format:
            payload["output_format"] = output_format
        if output_compression is not None:
            payload["output_compression"] = output_compression
        if moderation:
            payload["moderation"] = moderation

        return payload

    def _build_api_url(self, path: str) -> str:
        base = self._normalized_api_base_url()
        url = f"{base}{path}"
        if self.valves.api_version:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}api-version={self.valves.api_version}"
        return url

    def _normalized_api_base_url(self) -> str:
        base = (self.valves.api_base_url or "https://api.openai.com/v1").strip()
        return base[:-1] if base.endswith("/") else base

    def _format_api_error(self, response: requests.Response) -> str:
        try:
            payload = response.json()
            if isinstance(payload, dict):
                error = payload.get("error", payload)
                if isinstance(error, dict):
                    message = error.get("message") or json.dumps(error)
                else:
                    message = str(error)
            else:
                message = str(payload)
        except Exception:
            message = response.text[:1000]
        return f"API request failed ({response.status_code}): {message}"

    def _format_exception(self, exc: Exception) -> str:
        if isinstance(exc, RuntimeError):
            return str(exc)
        return str(exc)

    def _is_system_followup(self, messages: List[Dict[str, Any]]) -> bool:
        last_user_msg = None
        for msg in reversed(messages or []):
            if msg.get("role") == "user":
                last_user_msg = msg
                break

        if not last_user_msg:
            return False

        content = last_user_msg.get("content", "")
        if not isinstance(content, str):
            return False

        text = content.lower().strip()
        if re.search(r"suggest.*follow[- ]up.*(question|prompt|topic)", text):
            return True
        if re.match(r"^(please\s+)?(suggest|provide|generate)\s+\d*\s*follow[- ]up", text):
            return True
        return False

    def _is_guarded_prompt(self, text: str) -> bool:
        lower = (text or "").strip().lower()
        if lower.startswith("task:") or lower.startswith("### task"):
            return True
        if "suggest" in lower and "follow-up" in lower:
            if re.search(r"suggest.*follow[- ]up.*(question|prompt|topic)", lower):
                return True
            if re.match(r"^(please\s+)?(suggest|provide|generate)\s+.*follow[- ]up", lower):
                return True
        if re.search(r"\bsuggest\s+\d+\s*-?\s*\d*\s*", lower) and (
            "question" in lower or "prompt" in lower
        ):
            return True
        if len(lower.split()) <= 1 and len(lower) < 4:
            return True
        return False

    def _extract_latest_user_prompt_and_image_flag(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[str, bool]:
        for msg in reversed(messages or []):
            if (msg or {}).get("role") != "user":
                continue

            content = (msg or {}).get("content", "")
            prompt_parts: List[str] = []
            user_has_image = False

            if isinstance(content, str):
                prompt_parts.append(content.strip())
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text":
                        text = (item.get("text", "") or "").strip()
                        if text:
                            prompt_parts.append(text)
                    elif item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url:
                            user_has_image = True
            elif isinstance(content, dict):
                text = content.get("text") or content.get("content")
                if isinstance(text, str) and text.strip():
                    prompt_parts.append(text.strip())

            prompt = "\n".join([part for part in prompt_parts if part]).strip()
            if prompt and self._is_guarded_prompt(prompt):
                continue
            return prompt, user_has_image

        return "", False

    def _resolve_sticky_settings(
        self, messages: List[Dict[str, Any]], prompt: str
    ) -> Tuple[str, str, str]:
        size = self._normalize_size(self.valves.size)
        quality = self._normalize_quality(self.valves.quality)
        background = self._normalize_background(self.valves.background)

        hist_size, hist_quality, hist_background = self._extract_sticky_settings_from_history(
            messages
        )

        if not size and hist_size:
            size = hist_size
        if not quality and hist_quality:
            quality = hist_quality
        if not background and hist_background:
            background = hist_background

        if not size:
            size = self._extract_size_from_prompt(prompt)
        if not quality:
            quality = self._extract_quality_from_prompt(prompt)
        if not background:
            background = self._extract_background_from_prompt(prompt)

        if not size:
            size = "auto"
        if not quality:
            quality = "auto"
        if not background:
            background = "auto"

        if background == "transparent" and self._normalize_output_format(self.valves.output_format) == "jpeg":
            background = "auto"

        return size, quality, background

    def _extract_sticky_settings_from_history(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[str, str, str]:
        found_size = ""
        found_quality = ""
        found_background = ""

        for msg in reversed(messages or []):
            if msg.get("role") != "user":
                continue

            content = msg.get("content", "")
            text = self._stringify_content(content)
            if not text:
                continue

            if not found_size:
                found_size = self._extract_size_from_prompt(text)
            if not found_quality:
                found_quality = self._extract_quality_from_prompt(text)
            if not found_background:
                found_background = self._extract_background_from_prompt(text)

            if found_size and found_quality and found_background:
                break

        return found_size, found_quality, found_background

    def _extract_size_from_prompt(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        normalized = text.lower()
        match = re.search(r"(\d{3,4})\s*[x×]\s*(\d{3,4})", normalized)
        if match:
            size = f"{match.group(1)}x{match.group(2)}"
            if self._is_valid_size(size):
                return size

        keyword_map = [
            (r"\b4k\s+(landscape|wide|cinematic)\b", "3840x2160"),
            (r"\b4k\s+(portrait|vertical|tall)\b", "2160x3840"),
            (r"\b2k\s+square\b", "2048x2048"),
            (r"\b2k\s+(landscape|wide|cinematic)\b", "2048x1152"),
            (r"\b(portrait|vertical|tall)\b", "1024x1536"),
            (r"\b(landscape|wide|cinematic)\b", "1536x1024"),
            (r"\bsquare\b", "1024x1024"),
        ]
        for pattern, value in keyword_map:
            if re.search(pattern, normalized):
                return value

        if re.search(r"\bauto\b", normalized):
            return "auto"

        return ""

    def _extract_quality_from_prompt(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        normalized = text.lower()
        direct = re.search(r"\bquality\s*[:=]?\s*(auto|low|medium|high)\b", normalized)
        if direct:
            return direct.group(1)
        if re.search(r"\b(draft|quick|fast)\b", normalized):
            return "low"
        if re.search(r"\b(final|polished|premium|high[- ]quality)\b", normalized):
            return "high"
        return ""

    def _extract_background_from_prompt(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        normalized = text.lower()
        direct = re.search(
            r"\bbackground\s*[:=]?\s*(auto|transparent|opaque)\b", normalized
        )
        if direct:
            return direct.group(1)
        if re.search(r"\b(transparent|alpha channel|cutout|no background)\b", normalized):
            return "transparent"
        if re.search(r"\b(opaque background|solid background)\b", normalized):
            return "opaque"
        return ""

    def _normalize_size(self, value: str) -> str:
        value = (value or "").strip().lower()
        if not value:
            return ""
        if value == "auto":
            return "auto"
        if self._is_valid_size(value):
            return value
        return ""

    def _is_valid_size(self, value: str) -> bool:
        match = re.fullmatch(r"(\d{3,4})x(\d{3,4})", value or "")
        if not match:
            return False

        width = int(match.group(1))
        height = int(match.group(2))
        if width > 3840 or height > 3840:
            return False
        if width % 16 != 0 or height % 16 != 0:
            return False
        long_edge = max(width, height)
        short_edge = min(width, height)
        if short_edge == 0 or long_edge / short_edge > 3:
            return False
        total_pixels = width * height
        return 655360 <= total_pixels <= 8294400

    def _normalize_quality(self, value: str) -> str:
        value = (value or "").strip().lower()
        return value if value in {"auto", "low", "medium", "high"} else ""

    def _normalize_background(self, value: str) -> str:
        value = (value or "").strip().lower()
        return value if value in {"auto", "transparent", "opaque"} else ""

    def _normalize_output_format(self, value: str) -> str:
        value = (value or "png").strip().lower()
        if value == "jpg":
            value = "jpeg"
        return value if value in {"png", "jpeg", "webp"} else "png"

    def _normalize_output_compression(
        self, value: int, output_format: str
    ) -> Optional[int]:
        if output_format not in {"jpeg", "webp"}:
            return None
        try:
            value_int = int(value)
        except Exception:
            return None
        return max(0, min(100, value_int))

    def _normalize_moderation(self, value: str) -> str:
        value = (value or "auto").strip().lower()
        return value if value in {"auto", "low"} else "auto"

    def _mime_type_for_output_format(self, output_format: str) -> str:
        if output_format == "jpeg":
            return "image/jpeg"
        if output_format == "webp":
            return "image/webp"
        return "image/png"

    def _extract_images_from_content(self, content: str) -> List[str]:
        image_urls: List[str] = []
        markdown_pattern = r"!\[.*?\]\((data:image/[^)]+|https?://[^)]+|/[^)]+)\)"
        image_urls.extend(re.findall(markdown_pattern, content or ""))

        html_pattern = r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>'
        image_urls.extend(re.findall(html_pattern, content or ""))
        return image_urls

    def _download_image(
        self, image_url: str, request: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        try:
            if image_url.startswith("data:"):
                header, data = image_url.split(",", 1)
                mime_type = header.split(";")[0].split(":", 1)[1]
                image_data = base64.b64decode(data)
                ext = (mimetypes.guess_extension(mime_type) or ".png").lstrip(".")
                return {
                    "data": image_data,
                    "mime_type": mime_type,
                    "filename": f"inline.{ext}",
                }

            file_match = re.search(
                r"/api/v1/files/([a-f0-9\-]+)/content", image_url, re.IGNORECASE
            )
            if file_match:
                file_id = file_match.group(1)
                file_model = FilesDB.get_file_by_id(file_id)
                if file_model and file_model.path:
                    local_path = Storage.get_file(file_model.path)
                    with open(local_path, "rb") as f:
                        data_bytes = f.read()
                    mime_type = None
                    if file_model.meta and isinstance(file_model.meta, dict):
                        mime_type = file_model.meta.get("content_type") or file_model.meta.get(
                            "mime_type"
                        )
                    if not mime_type:
                        mime_type = (
                            mimetypes.guess_type(file_model.filename or local_path)[0]
                            or "image/png"
                        )
                    return {
                        "data": data_bytes,
                        "mime_type": mime_type,
                        "filename": file_model.filename or os.path.basename(local_path),
                    }

            if image_url.startswith("/"):
                base = os.getenv("WEBUI_URL", "").rstrip("/")
                if not base and request is not None:
                    try:
                        proto = request.headers.get("x-forwarded-proto")
                        host = request.headers.get("x-forwarded-host") or request.headers.get(
                            "host"
                        )
                        if proto and host:
                            base = f"{proto}://{host}"
                        else:
                            base = str(getattr(request, "base_url", "")).rstrip("/")
                    except Exception:
                        base = ""
                if not base:
                    return None
                image_url = f"{base}{image_url}"

            headers = {}
            try:
                if request is not None and getattr(request, "headers", None):
                    auth = request.headers.get("authorization")
                    cookie = request.headers.get("cookie")
                    if auth:
                        headers["Authorization"] = auth
                    if cookie:
                        headers["Cookie"] = cookie
            except Exception:
                pass

            attempts = max(1, int(self.valves.retry_attempts))
            backoff = 1.0
            response = None
            for attempt in range(1, attempts + 1):
                try:
                    response = requests.get(
                        image_url,
                        headers=headers or None,
                        timeout=max(1, int(self.valves.download_timeout)),
                    )
                    if response.status_code == 429 or 500 <= response.status_code <= 599:
                        if attempt < attempts:
                            time.sleep(backoff)
                            backoff *= float(self.valves.retry_backoff_base)
                            continue
                    response.raise_for_status()
                    break
                except Exception as ex:
                    self._debug(f"Download attempt {attempt} failed: {ex}")
                    if attempt < attempts:
                        time.sleep(backoff)
                        backoff *= float(self.valves.retry_backoff_base)
                        continue
                    raise

            if response is None:
                return None

            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                mime_type = mimetypes.guess_type(image_url)[0] or "image/png"
            else:
                mime_type = content_type.split(";", 1)[0]

            return {
                "data": response.content,
                "mime_type": mime_type,
                "filename": os.path.basename(image_url.split("?", 1)[0]) or "image.png",
            }
        except Exception as e:
            self._debug(f"Failed to download image {image_url}: {e}")
            return None

    def _stringify_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            return str(content.get("text") or content.get("content") or "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        parts.append(text)
            return "\n".join(parts)
        return str(content or "")
