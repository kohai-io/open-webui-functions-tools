"""
title: n8n Pipeline
author: owndev
author_url: https://github.com/owndev/
project_url: https://github.com/owndev/Open-WebUI-Functions
funding_url: https://github.com/sponsors/owndev
n8n_template: https://github.com/owndev/Open-WebUI-Functions/blob/master/pipelines/n8n/Open_WebUI_Test_Agent.json
version: 2.0.1
license: Apache License 2.0
description: A pipeline for interacting with N8N workflows, enabling seamless communication with various N8N workflows via configurable headers and robust error handling. This includes support for dynamic message handling and real-time interaction with N8N workflows.
features:
  - Integrates with N8N for seamless communication.
  - Supports dynamic message handling.
  - Enables real-time interaction with N8N workflows.
  - Provides configurable status emissions.
  - Cloudflare Access support for secure communication.
  - Encrypted storage of sensitive API keys
"""

from typing import Optional, Callable, Awaitable, Any, Dict, AsyncGenerator, List
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from cryptography.fernet import Fernet, InvalidToken
import time
import aiohttp
import os
import base64
import hashlib
import logging
import json
from open_webui.env import AIOHTTP_CLIENT_TIMEOUT, SRC_LOG_LEVELS
from pydantic_core import core_schema


# Simplified encryption implementation with automatic handling
class EncryptedStr(str):
    """A string type that automatically handles encryption/decryption"""

    @classmethod
    def _get_encryption_key(cls) -> Optional[bytes]:
        """
        Generate encryption key from WEBUI_SECRET_KEY if available
        Returns None if no key is configured
        """
        secret = os.getenv("WEBUI_SECRET_KEY")
        if not secret:
            return None
        hashed_key = hashlib.sha256(secret.encode()).digest()
        return base64.urlsafe_b64encode(hashed_key)

    @classmethod
    def encrypt(cls, value: str) -> str:
        """
        Encrypt a string value if a key is available
        Returns the original value if no key is available
        """
        if not value or value.startswith("encrypted:"):
            return value
        key = cls._get_encryption_key()
        if not key:  # No encryption if no key
            return value
        f = Fernet(key)
        encrypted = f.encrypt(value.encode())
        return f"encrypted:{encrypted.decode()}"

    @classmethod
    def decrypt(cls, value: str) -> str:
        """
        Decrypt an encrypted string value if a key is available
        Returns the original value if no key is available or decryption fails
        """
        if not value or not value.startswith("encrypted:"):
            return value
        key = cls._get_encryption_key()
        if not key:  # No decryption if no key
            return value[len("encrypted:") :]  # Return without prefix
        try:
            encrypted_part = value[len("encrypted:") :]
            f = Fernet(key)
            decrypted = f.decrypt(encrypted_part.encode())
            return decrypted.decode()
        except (InvalidToken, Exception):
            return value

    # Pydantic integration
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


# Helper function for cleaning up aiohttp resources
async def cleanup_session(session: Optional[aiohttp.ClientSession]) -> None:
    """
    Clean up the aiohttp session.
    Args:
        session: The ClientSession object to close
    """
    if session:
        await session.close()


class Pipe:
    class Valves(BaseModel):
        N8N_URL: str = Field(
            default="https://<your-endpoint>/webhook/<your-webhook>",
            description="URL for the N8N webhook",
        )
        N8N_BEARER_TOKEN: EncryptedStr = Field(
            default="",
            description="Bearer token for authenticating with the N8N webhook",
        )
        INPUT_FIELD: str = Field(
            default="chatInput",
            description="Field name for the input message in the N8N payload",
        )
        RESPONSE_FIELD: str = Field(
            default="output",
            description="Field name for the response message in the N8N payload",
        )
        EMIT_INTERVAL: float = Field(
            default=2.0, description="Interval in seconds between status emissions"
        )
        ENABLE_STATUS_INDICATOR: bool = Field(
            default=True, description="Enable or disable status indicator emissions"
        )
        CF_ACCESS_CLIENT_ID: EncryptedStr = Field(
            default="",
            description="Only if behind Cloudflare: https://developers.cloudflare.com/cloudflare-one/identity/service-tokens/",
        )
        CF_ACCESS_CLIENT_SECRET: EncryptedStr = Field(
            default="",
            description="Only if behind Cloudflare: https://developers.cloudflare.com/cloudflare-one/identity/service-tokens/",
        )

    def __init__(self):
        self.name = "N8N Conversational Agent"
        self.valves = self.Valves()
        self.last_emit_time = 0
        self.log = logging.getLogger("n8n_pipeline")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))

    async def emit_status(
        self,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]],
        level: str,
        message: str,
        done: bool,
    ):
        current_time = time.time()
        if (
            __event_emitter__
            and self.valves.ENABLE_STATUS_INDICATOR
            and (
                current_time - self.last_emit_time >= self.valves.EMIT_INTERVAL or done
            )
        ):
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "status": "complete" if done else "in_progress",
                        "level": level,
                        "description": message,
                        "done": done,
                    },
                }
            )
            self.last_emit_time = current_time

    def extract_event_info(self, event_emitter):
        if not event_emitter or not event_emitter.__closure__:
            return None, None
        for cell in event_emitter.__closure__:
            if isinstance((request_info := cell.cell_contents), dict):
                chat_id = request_info.get("chat_id")
                message_id = request_info.get("message_id")
                return chat_id, message_id
        return None, None

    def get_headers(self) -> Dict[str, str]:
        """
        Constructs the headers for the API request.
        Returns:
            Dictionary containing the required headers for the API request.
        """
        headers = {"Content-Type": "application/json"}
        # Add bearer token if available
        bearer_token = self.valves.N8N_BEARER_TOKEN
        if bearer_token:
            # Handle both EncryptedStr objects and plain strings
            if hasattr(bearer_token, 'get_decrypted'):
                bearer_token = bearer_token.get_decrypted()
            else:
                bearer_token = EncryptedStr.decrypt(str(bearer_token))
            if bearer_token:
                headers["Authorization"] = f"Bearer {bearer_token}"
        # Add Cloudflare Access headers if available
        cf_client_id = self.valves.CF_ACCESS_CLIENT_ID
        if cf_client_id:
            # Handle both EncryptedStr objects and plain strings
            if hasattr(cf_client_id, 'get_decrypted'):
                cf_client_id = cf_client_id.get_decrypted()
            else:
                cf_client_id = EncryptedStr.decrypt(str(cf_client_id))
            if cf_client_id:
                headers["CF-Access-Client-Id"] = cf_client_id
        cf_client_secret = self.valves.CF_ACCESS_CLIENT_SECRET
        if cf_client_secret:
            # Handle both EncryptedStr objects and plain strings
            if hasattr(cf_client_secret, 'get_decrypted'):
                cf_client_secret = cf_client_secret.get_decrypted()
            else:
                cf_client_secret = EncryptedStr.decrypt(str(cf_client_secret))
            if cf_client_secret:
                headers["CF-Access-Client-Secret"] = cf_client_secret
        # Request streaming if available, but still accept JSON
        headers.setdefault(
            "Accept", "text/event-stream, application/x-ndjson, application/json"
        )
        return headers

    async def pipe(
        self,
        body: Dict[str, Any],
        user: Optional[Dict[str, Any]] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __event_call__: Optional[Callable[[dict], Awaitable[dict]]] = None,
        **_: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming pipeline for Open WebUI.

        Yields only the token content from n8n 'item' events.
        Falls back to a single reply if the webhook does not stream.
        """
        # Normalise inputs
        body = body or {}
        user = user or {}

        # Extract the last user message
        messages: List[Dict[str, Any]] = body.get("messages") or []
        user_message: str = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "") or ""
                break
        if not user_message:
            yield "Error: No user message found"
            return

        # Optional status start
        await self.emit_status(
            __event_emitter__, "info", f"Calling {self.name} ...", False
        )

        # Prepare payload
        system_prompt = ""
        if messages:
            system_prompt = messages[0].get("content", "") or ""
            if "Prompt: " in system_prompt:
                system_prompt = system_prompt.split("Prompt: ")[-1]
        question = user_message
        if "Prompt: " in question:
            question = question.split("Prompt: ")[-1]

        chat_id, message_id = self.extract_event_info(__event_emitter__)
        payload = {
            "systemPrompt": system_prompt,
            "user_id": user.get("id"),
            "user_email": user.get("email"),
            "user_name": user.get("name"),
            "user_role": user.get("role"),
            "chat_id": chat_id,
            "message_id": message_id,
        }
        payload[self.valves.INPUT_FIELD] = question

        headers = self.get_headers()

        session: Optional[aiohttp.ClientSession] = None
        try:
            session = aiohttp.ClientSession(
                trust_env=True,
                timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT),
            )
            async with session.post(
                self.valves.N8N_URL, json=payload, headers=headers
            ) as response:
                content_type = response.headers.get("Content-Type", "")
                self.log.debug(
                    f"N8N response status {response.status}, content-type {content_type}"
                )

                if response.status != 200:
                    error_text = await response.text()
                    await self.emit_status(
                        __event_emitter__,
                        "error",
                        f"Error: {response.status} - {error_text}",
                        True,
                    )
                    yield f"Error: {response.status} - {error_text}"
                    return

                # Streaming reader that handles NDJSON or SSE
                partial = ""
                emitted_any = False
                nonstream_buffer: List[str] = []
                end_seen = False

                async for raw in response.content:
                    if not raw:
                        continue
                    buffer = partial + raw.decode("utf-8", errors="ignore")

                    # Split into lines, keep last partial (no trailing newline) for next chunk
                    segments = buffer.split("\n")
                    if buffer.endswith("\n"):
                        partial = ""
                    else:
                        partial = segments.pop()  # last segment incomplete

                    for line in segments:
                        if not line:
                            continue
                        s = line.rstrip("\r")
                        # Handle SSE "data:" prefix
                        data = s[5:].strip() if s.startswith("data:") else s.strip()

                        # Skip keep-alives/comments
                        if not data or data.startswith(":"):
                            continue
                        if data in ("[DONE]", "done"):
                            end_seen = True
                            continue

                        payload_obj = None
                        try:
                            payload_obj = json.loads(data)
                        except json.JSONDecodeError:
                            payload_obj = None

                        if isinstance(payload_obj, dict):
                            evt_type = payload_obj.get("type")
                            if evt_type == "item":
                                token = payload_obj.get("content") or ""
                                if token:
                                    emitted_any = True
                                    yield token
                                continue
                            if evt_type == "begin":
                                # Ignore begin events
                                continue
                            if evt_type == "end":
                                end_seen = True
                                continue
                            # Fallbacks for other JSON shapes
                            token = (
                                payload_obj.get(self.valves.RESPONSE_FIELD)
                                or payload_obj.get("delta")
                                or payload_obj.get("token")
                                or payload_obj.get("text")
                            )
                            if token:
                                emitted_any = True
                                yield token
                                continue
                            # Single-key dict with a string
                            if len(payload_obj) == 1:
                                only_val = next(iter(payload_obj.values()))
                                if isinstance(only_val, str):
                                    emitted_any = True
                                    yield only_val
                                    continue
                            # Unknown JSON event: ignore in stream
                            continue
                        else:
                            # Not JSON. If we have not detected streaming yet, buffer for fallback.
                            if not emitted_any:
                                nonstream_buffer.append(data)

                    if end_seen:
                        break

                # If we never emitted streaming tokens, treat the body as a single JSON or text
                if not emitted_any:
                    if partial:
                        nonstream_buffer.append(partial)
                    buffered = "\n".join(nonstream_buffer).strip()
                    if buffered:
                        try:
                            obj = json.loads(buffered)
                            if (
                                isinstance(obj, dict)
                                and self.valves.RESPONSE_FIELD in obj
                            ):
                                yield obj[self.valves.RESPONSE_FIELD] or ""
                            elif isinstance(obj, str):
                                yield obj
                            else:
                                # Last resort: yield raw buffered text
                                yield buffered
                        except json.JSONDecodeError:
                            yield buffered

            # Optional explicit stream done event for UIs that expect it
            if __event_emitter__:
                await __event_emitter__({"type": "stream", "data": {"done": True}})

            # Optional status complete
            await self.emit_status(__event_emitter__, "info", "Complete", True)

        except Exception as e:
            err = f"Error during sequence execution: {e}"
            self.log.exception(err)
            await self.emit_status(__event_emitter__, "error", err, True)
            yield err
        finally:
            if session:
                await cleanup_session(session)
