"""
title: ElevenLabs Sound Effects (Tool)
requirements: aiohttp, cryptography
"""

from typing import Optional, Any
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import core_schema
from cryptography.fernet import Fernet, InvalidToken
import aiohttp
import base64
import hashlib
import json
import logging
import os
import re
import tempfile
import time
import uuid

# Open WebUI imports
from fastapi import Request
from open_webui.models.files import Files, FileForm
from open_webui.storage.provider import Storage
from open_webui.models.users import Users


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
            return value[len("encrypted:") :]
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


class Tools:
    """ElevenLabs SFX Tool: generate sound effect from text and return a downloadable link."""

    class Valves(BaseModel):
        api_key: EncryptedStr = Field(
            default="",
            description="ElevenLabs API key (xi-api-key)",
        )
        api_base_url: str = Field(
            default="https://api.elevenlabs.io/v1",
            description="Base URL for ElevenLabs API",
        )
        output_format: str = Field(
            default="mp3_22050_32",
            description="Default output audio format (e.g., mp3_44100_128, wav)",
        )
        timeout: int = Field(
            default=120,
            description="Max seconds to wait for generation/response.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.log = logging.getLogger("elevenlabs_sfx_tool")
        self.log.setLevel(logging.INFO)

    async def generate_sound_effect(
        self,
        text: str = Field(..., description="Describe the sound to generate"),
        output_format: Optional[str] = Field(
            default=None, description="Override output format for this run"
        ),
        __request__: Request = None,
        __user__: dict = None,
        __event_emitter__=None,
    ) -> str:
        if not isinstance(text, str) or not text.strip():
            return "Error: 'text' is required."

        api_key = EncryptedStr.decrypt(self.valves.api_key)
        if not api_key:
            return "Error: ElevenLabs API key is missing. Please configure it in the tool settings."

        fmt = (output_format or self.valves.output_format or "mp3_22050_32").strip()
        api_base = (self.valves.api_base_url or "").rstrip("/")

        if __event_emitter__:
            await __event_emitter__(
                {"type": "status", "data": {"description": "Generating sound…", "done": False}}
            )

        try:
            audio_bytes, ext, content_type = await self._call_elevenlabs(api_base, api_key, text, fmt)
            if not audio_bytes:
                if __event_emitter__:
                    await __event_emitter__(
                        {"type": "status", "data": {"description": "Generation failed", "done": True}}
                    )
                return "Tell the user that sound generation failed"

            file_id = await self._save_file(audio_bytes, text, fmt, ext, content_type, __user__)
            size_mb = len(audio_bytes) / (1024 * 1024)

            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": "Done", "done": True}}
                )

            if file_id:
                return (
                    f"✅ Sound effect generated! [Download here](/api/v1/files/{file_id}/content) "
                    f"({size_mb:.1f}MB)"
                )
            else:
                return f"✅ Sound effect generated but failed to save. Size: {size_mb:.1f}MB"
        except Exception as e:
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": f"Error: {e}", "done": True}}
                )
            return f"Tell the user there was an error: {e}"

    async def _call_elevenlabs(
        self, api_base: str, api_key: str, text: str, output_format: str
    ) -> tuple[Optional[bytes], str, str]:
        url = f"{api_base}/sound-generation"
        params = {"output_format": output_format} if output_format else {}
        payload = {"text": text}
        headers = {"xi-api-key": api_key, "Content-Type": "application/json"}

        timeout = aiohttp.ClientTimeout(total=self.valves.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload, params=params) as resp:
                if resp.status != 200:
                    self.log.error(f"ElevenLabs SFX error {resp.status}: {(await resp.text())[:300]}")
                    return None, "", ""
                ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
                try:
                    if ctype == "application/json":
                        data = await resp.json()
                        b64 = data.get("audio_base64") or data.get("audio")
                        if isinstance(b64, str):
                            blob = base64.b64decode(b64)
                            ext, mime = self._guess_ext_and_mime(output_format)
                            return blob, ext, mime
                        file_url = data.get("url") or data.get("audio_url")
                        if file_url:
                            async with session.get(file_url) as r2:
                                if r2.status == 200:
                                    blob = await r2.read()
                                    ext, mime = self._guess_ext_and_mime(output_format)
                                    return blob, ext, mime
                                else:
                                    self.log.error(f"Failed fetching audio URL: {r2.status}")
                                    return None, "", ""
                    else:
                        blob = await resp.read()
                        ext, mime = self._guess_ext_and_mime(output_format)
                        return blob, ext, mime
                except Exception as e:
                    self.log.error(f"Parse SFX response failed: {e}")
                    return None, "", ""

    def _guess_ext_and_mime(self, output_format: str) -> tuple[str, str]:
        fmt = (output_format or "").lower()
        if fmt.startswith("mp3"):
            return ".mp3", "audio/mpeg"
        if fmt.startswith("wav"):
            return ".wav", "audio/wav"
        if fmt.startswith("pcm"):
            return ".pcm", "audio/L16"
        return ".mp3", "audio/mpeg"

    async def _save_file(
        self,
        audio_data: bytes,
        prompt: str,
        output_format: str,
        ext: str,
        content_type: str,
        __user__: Optional[dict],
    ) -> Optional[str]:
        safe_prompt = "".join(c for c in prompt[:40] if c.isalnum() or c in (" ", "-", "_")).rstrip().replace(" ", "_")
        timestamp = int(time.time())
        filename = f"sfx_{safe_prompt}_{timestamp}{ext}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name
        try:
            with open(tmp_path, "rb") as f:
                file_data, file_path = Storage.upload_file(
                    f, filename, {"content_type": content_type, "source": "elevenlabs_sfx_tool"}
                )
            file_id = str(uuid.uuid4())
            record = Files.insert_new_file(
                (__user__ or {}).get("id") or "system",
                FileForm(
                    id=file_id,
                    filename=filename,
                    path=file_path,
                    meta={
                        "name": filename,
                        "content_type": content_type,
                        "size": len(audio_data),
                        "source": "elevenlabs_sfx_tool",
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
