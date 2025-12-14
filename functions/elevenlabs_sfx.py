"""
title: ElevenLabs Sound Effects Pipeline
author: open-webui
date: 2025-09-26
version: 1.2
license: MIT
description: Generate sound effects from text using ElevenLabs Sound Effects API with inline audio display. Supports duration, looping, and prompt influence controls. Enhanced with comprehensive error handling and logging.
requirements: aiohttp, cryptography, pydantic

USAGE
- Type a description of the sound effect you want, e.g., "Whoosh with airy tail and subtle reverb".
- The last user message is parsed for the text prompt. No images required.

PARAMETERS
- Duration: Set specific length (0.1-30 seconds). Costs 40 credits/second when specified.
- Looping: Enable seamless looping for sounds longer than 30 seconds.
- Prompt influence: Control interpretation (0.0=creative, 1.0=literal).

OUTPUT FORMAT
- The pipeline requests audio in the format configured in valves (default mp3_22050_32).
- Supported formats: mp3_22050_32, mp3_44100_64, mp3_44100_128, wav, pcm_16000, etc.

OPEN WEBUI PROMPT-VARIABLES (Turbo-safe)
Describe the sound effect you want (text only):
{{instruction | textarea:placeholder="e.g., Cinematic braam suitable for trailer impacts; dark, wide, sub-heavy." :required}}

Optional overrides (leave empty to use valve defaults):
{{output_format | text:placeholder="e.g., mp3_22050_32"}}
{{duration_seconds | text:placeholder="e.g., 5.0"}}
{{loop | text:placeholder="true or false"}}
{{prompt_influence | text:placeholder="0.0 to 1.0"}}

If provided, the pipeline will use the overrides for this request.

{"output_format": "{{output_format}}", "duration_seconds": {{duration_seconds}}, "loop": {{loop}}, "prompt_influence": {{prompt_influence}}}
"""

from typing import Optional, Callable, Awaitable, Any
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
import time
import tempfile
import uuid
from pydantic_core import core_schema

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


class Pipe:
    class Valves(BaseModel):
        ELEVEN_API_KEY: EncryptedStr = Field(
            default="",
            description="ElevenLabs API key (xi-api-key). Get it from https://elevenlabs.io/",
        )
        API_BASE_URL: str = Field(
            default="https://api.elevenlabs.io/v1",
            description="Base URL for ElevenLabs API",
        )
        OUTPUT_FORMAT: str = Field(
            default="mp3_22050_32",
            description="Output audio format (e.g., mp3_22050_32, mp3_44100_128, wav).",
        )
        DURATION_SECONDS: Optional[float] = Field(
            default=None,
            description="Duration in seconds (0.1-30). None = auto-determined. Note: 40 credits/second.",
        )
        LOOP: bool = Field(
            default=False,
            description="Enable seamless looping for sounds longer than 30 seconds.",
        )
        PROMPT_INFLUENCE: Optional[float] = Field(
            default=None,
            description="Control prompt interpretation (0.0=creative, 1.0=literal). None = default.",
        )
        TIMEOUT: int = Field(
            default=120,
            description="Max seconds to wait for generation/response.",
        )
        EMIT_INTERVAL: float = Field(
            default=0.5, description="Interval in seconds between status emissions"
        )
        ENABLE_STATUS_INDICATOR: bool = Field(
            default=True, description="Enable or disable status indicator emissions"
        )

    def __init__(self):
        self.name = "ElevenLabs Sound Effects"
        self.valves = self.Valves()
        self.log = logging.getLogger("elevenlabs_sfx_pipeline")
        self.log.setLevel(logging.INFO)
        self.last_emit_time = 0

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
    
    async def handle_error(
        self,
        body: dict,
        event_emitter: Optional[Callable[[dict], Awaitable[None]]],
        short_status: str,
        detailed_message: str,
    ) -> str:
        """Handle errors with short status updates and detailed chat messages.
        
        Args:
            body: The request body to append the message to
            event_emitter: Event emitter for status updates
            short_status: Brief error indicator for status (e.g., "Configuration error")
            detailed_message: Full error details for chat response
        
        Returns:
            The detailed error message
        """
        # Emit short status update (limited length)
        await self.emit_status(event_emitter, "error", f"❌ {short_status}", True)
        
        # Log the full error
        self.log.error(detailed_message)
        
        # Add detailed message to chat
        full_response = f"❌ **{short_status}**\n\n{detailed_message}"
        body["messages"].append({"role": "assistant", "content": full_response})
        
        return full_response

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
        __event_call__: Callable[[dict], Awaitable[dict]] = None,
    ) -> Optional[str]:
        # Log the incoming request structure
        self.log.info(f"=== ElevenLabs SFX Pipeline Called ===")
        self.log.info(f"Body keys: {list(body.keys())}")
        self.log.info(f"Messages count: {len(body.get('messages', []))}")
        
        await self.emit_status(
            __event_emitter__, "info", "Initializing ElevenLabs Sound Effects..."
        )
        
        # Validate API key configuration
        api_key = self.valves.ELEVEN_API_KEY.get_decrypted()
        if not api_key:
            return await self.handle_error(
                body, __event_emitter__,
                "Configuration error",
                "ElevenLabs API key not configured.\n\n"
                "**Solution:** Set your `ELEVEN_API_KEY` in the pipeline settings.\n\n"
                "Get your API key from: https://elevenlabs.io/"
            )

        # Extract prompt text and optional override block
        messages = body.get("messages", [])
        prompt_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    prompt_text = content
                elif isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            prompt_text = item.get("text", "")
                break
        if not prompt_text:
            return await self.handle_error(
                body, __event_emitter__,
                "Invalid request",
                "No text provided for sound effect generation.\n\n"
                "**Solution:** Provide a text description of the sound effect you want to create.\n\n"
                "**Example:** 'Cinematic whoosh with airy tail and subtle reverb'"
            )
        
        self.log.info(f"Extracted prompt text: {prompt_text[:100]}...")

        # Parse parameters from natural language first, then check for JSON overrides
        prompt_text, nl_params = self._parse_natural_language_params(prompt_text)
        prompt_text, json_overrides = self._parse_inline_overrides(prompt_text)
        
        # Merge: JSON overrides take precedence over natural language, then valves
        overrides = {**nl_params, **json_overrides}
        output_format = (overrides.get("output_format") or self.valves.OUTPUT_FORMAT).strip()
        duration_seconds = overrides.get("duration_seconds") or self.valves.DURATION_SECONDS
        loop = overrides.get("loop") if "loop" in overrides else self.valves.LOOP
        prompt_influence = overrides.get("prompt_influence") or self.valves.PROMPT_INFLUENCE
        api_base = (self.valves.API_BASE_URL or "").rstrip("/")

        await self.emit_status(
            __event_emitter__, "info", "Generating sound effect..."
        )
        
        try:
            self.log.info(f"Generating SFX with params: format={output_format}, duration={duration_seconds}, loop={loop}, influence={prompt_influence}")
            audio_bytes, fmt_ext, content_type = await self._generate_sfx(
                api_base, api_key, prompt_text, output_format, duration_seconds, loop, prompt_influence
            )
            if not audio_bytes:
                return await self.handle_error(
                    body, __event_emitter__,
                    "Generation failed",
                    "Sound effect generation failed.\n\n"
                    "**Possible causes:**\n"
                    "- Invalid API key or insufficient credits\n"
                    "- API service temporarily unavailable\n"
                    "- Invalid parameters or prompt\n\n"
                    "**Solution:** Check the logs for more details and verify your API key and credits."
                )

            await self.emit_status(
                __event_emitter__, "info", "Saving audio file..."
            )
            
            # Save to Files DB
            user_id = __user__.get("id") if __user__ else None
            file_id = await self._save_audio_file(
                audio_bytes,
                prompt_text,
                output_format,
                fmt_ext,
                content_type,
                user_id,
            )

            size_mb = len(audio_bytes) / (1024 * 1024)
            if file_id:
                audio_url = f"/api/v1/files/{file_id}/content"
                response_content = (
                    f"✅ Sound effect generated! ({size_mb:.1f}MB)\n\n"
                    f'<audio controls src="{audio_url}" />\n\n'
                    f"[Download]({audio_url})"
                )
            else:
                response_content = f"✅ Sound effect generated but failed to save. Size: {size_mb:.1f}MB"
            
            await self.emit_status(
                __event_emitter__, "info", "Sound effect generation complete!", True
            )
            
            body["messages"].append({"role": "assistant", "content": response_content})
            return response_content
        except aiohttp.ClientError as e:
            self.log.exception(f"Network error during SFX generation: {e}")
            return await self.handle_error(
                body, __event_emitter__,
                "Network error",
                f"Network error occurred during sound generation.\n\n"
                f"**Error details:** {str(e)}\n\n"
                f"**Possible causes:**\n"
                f"- Network connectivity issues\n"
                f"- ElevenLabs API service unavailable\n"
                f"- Timeout (current: {self.valves.TIMEOUT}s)\n\n"
                f"**Solutions:**\n"
                f"- Check your internet connection\n"
                f"- Increase the `TIMEOUT` setting if needed\n"
                f"- Try again in a few moments"
            )
        except json.JSONDecodeError as e:
            self.log.exception(f"JSON parsing error: {e}")
            return await self.handle_error(
                body, __event_emitter__,
                "API response error",
                f"Failed to parse API response.\n\n"
                f"**Error details:** {str(e)}\n\n"
                f"This may indicate an API format change or server error.\n\n"
                f"**Solution:** Check the ElevenLabs API status and report this issue if it persists."
            )
        except Exception as e:
            self.log.exception("Unexpected error during sound effect generation")
            return await self.handle_error(
                body, __event_emitter__,
                "Unexpected error",
                f"An unexpected error occurred during sound effect generation.\n\n"
                f"**Error details:** {str(e)}\n\n"
                f"**Error type:** {type(e).__name__}\n\n"
                f"Please check the logs for more information."
            )

    def _parse_natural_language_params(self, text: str) -> tuple[str, dict]:
        """Extract duration, looping, and prompt influence from natural language."""
        params: dict = {}
        cleaned = text or ""
        
        # Extract duration: "5 seconds", "10 second", "3.5s", "for 8 seconds", etc.
        duration_patterns = [
            r'(?:for\s+)?(\d+(?:\.\d+)?)\s*(?:second|sec|s)(?:s)?(?:\s+long)?',
            r'(?:duration|length)(?:\s+of)?\s+(\d+(?:\.\d+)?)\s*(?:second|sec|s)?',
        ]
        for pattern in duration_patterns:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                try:
                    duration = float(match.group(1))
                    if 0.1 <= duration <= 30:
                        params["duration_seconds"] = duration
                        # Remove the matched text
                        cleaned = cleaned[:match.start()] + cleaned[match.end():]
                        break
                except ValueError:
                    pass
        
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
        """Parse JSON overrides from text prompt.
        
        Extracts JSON objects containing parameter overrides like:
        {"output_format": "mp3_44100_128", "duration_seconds": 5}
        """
        overrides: dict = {}
        cleaned = text or ""
        try:
            json_match = re.findall(r"\{[^{}]*\}", cleaned, flags=re.DOTALL)
            if json_match:
                candidate = json_match[-1]
                try:
                    data = json.loads(candidate)
                    if isinstance(data, dict):
                        for k in ("output_format", "duration_seconds", "loop", "prompt_influence"):
                            if k in data and data[k] is not None:
                                overrides[k] = data[k]
                        if overrides:
                            self.log.info(f"Parsed JSON overrides: {overrides}")
                            cleaned = cleaned.replace(candidate, "", 1).strip()
                            return cleaned, overrides
                except json.JSONDecodeError as e:
                    self.log.debug(f"Failed to parse JSON override: {e}")
                    pass
        except Exception as e:
            self.log.debug(f"Error parsing inline overrides: {e}")
            pass
        return cleaned, overrides

    def _guess_ext_and_mime(self, output_format: str) -> tuple[str, str]:
        """Guess file extension and MIME type from output format string."""
        fmt = (output_format or "").lower()
        if fmt.startswith("mp3"):
            return ".mp3", "audio/mpeg"
        if fmt.startswith("wav"):
            return ".wav", "audio/wav"
        if fmt.startswith("pcm"):
            return ".pcm", "audio/L16"
        # default
        return ".mp3", "audio/mpeg"
    
    def _format_api_error(self, status_code: int, error_text: str) -> str:
        """Format ElevenLabs API errors into user-friendly messages.
        
        Args:
            status_code: HTTP status code from API response
            error_text: Error response text from API
        
        Returns:
            Formatted error message with helpful context
        """
        try:
            # Try to parse as JSON for structured errors
            error_data = json.loads(error_text)
            error_message = error_data.get('detail') or error_data.get('message') or error_data.get('error') or str(error_data)
        except (json.JSONDecodeError, Exception):
            error_message = error_text[:500]  # Truncate long error text
        
        # Common ElevenLabs API error codes
        if status_code == 401:
            return (
                "Invalid API key or authentication failed.\n\n"
                "**Solution:** Verify your `ELEVEN_API_KEY` is correct in pipeline settings.\n\n"
                f"**API response:** {error_message}"
            )
        elif status_code == 403:
            return (
                "Access forbidden - possible causes:\n"
                "- Insufficient API credits\n"
                "- Feature not available on your plan\n"
                "- Account suspended\n\n"
                "**Solution:** Check your ElevenLabs account status and credits at https://elevenlabs.io/\n\n"
                f"**API response:** {error_message}"
            )
        elif status_code == 422:
            return (
                "Invalid request parameters.\n\n"
                "**Possible issues:**\n"
                "- Invalid output format\n"
                "- Duration out of range (must be 0.1-30 seconds)\n"
                "- Prompt influence out of range (must be 0.0-1.0)\n"
                "- Invalid prompt text\n\n"
                f"**API response:** {error_message}"
            )
        elif status_code == 429:
            return (
                "Rate limit exceeded.\n\n"
                "**Solution:** Wait a moment and try again, or upgrade your ElevenLabs plan for higher limits.\n\n"
                f"**API response:** {error_message}"
            )
        elif status_code == 500:
            return (
                "ElevenLabs API server error.\n\n"
                "**Solution:** This is a temporary issue on ElevenLabs' side. Please try again in a few moments.\n\n"
                f"**API response:** {error_message}"
            )
        elif status_code == 503:
            return (
                "ElevenLabs API service unavailable.\n\n"
                "**Solution:** The service may be under maintenance. Check https://status.elevenlabs.io/ and try again later.\n\n"
                f"**API response:** {error_message}"
            )
        else:
            return (
                f"API request failed with HTTP {status_code}.\n\n"
                f"**Error details:** {error_message}\n\n"
                "Check the ElevenLabs API documentation or contact support if this persists."
            )

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
        Response: JSON likely including audio_base64; we handle both base64 or direct bytes.
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
        
        self.log.info(f"Making API request to: {url}")
        self.log.info(f"Request params: {params}")
        self.log.info(f"Request payload keys: {list(payload.keys())}")

        timeout = aiohttp.ClientTimeout(total=self.valves.TIMEOUT)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload, params=params) as resp:
                    self.log.info(f"Response status: {resp.status}")
                    self.log.info(f"Response content-type: {resp.headers.get('Content-Type', 'unknown')}")
                    
                    if resp.status != 200:
                        error_text = await resp.text()
                        self.log.error(f"ElevenLabs SFX API error {resp.status}: {error_text[:500]}")
                        
                        # Parse and format API error for user
                        error_msg = self._format_api_error(resp.status, error_text)
                        raise Exception(error_msg)

                    # Parse response based on content type
                    ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
                    self.log.info(f"Processing response with content-type: {ctype}")
                    
                    try:
                        if ctype == "application/json":
                            data = await resp.json()
                            self.log.info(f"JSON response keys: {list(data.keys())}")
                            
                            # Common field name used by other endpoints
                            b64 = data.get("audio_base64") or data.get("audio") or None
                            if isinstance(b64, str):
                                self.log.info(f"Found base64 audio data: {len(b64)} characters")
                                audio_bytes = base64.b64decode(b64)
                                ext, mime = self._guess_ext_and_mime(output_format)
                                self.log.info(f"Decoded audio: {len(audio_bytes)} bytes, format: {ext}, mime: {mime}")
                                return audio_bytes, ext, mime
                            
                            # Some responses may embed a URL (future-proof)
                            file_url = data.get("url") or data.get("audio_url")
                            if file_url:
                                self.log.info(f"Found audio URL in response: {file_url[:100]}...")
                                async with session.get(file_url) as r2:
                                    if r2.status == 200:
                                        blob = await r2.read()
                                        ext, mime = self._guess_ext_and_mime(output_format)
                                        self.log.info(f"Downloaded audio from URL: {len(blob)} bytes")
                                        return blob, ext, mime
                                    else:
                                        error_text = await r2.text()
                                        self.log.error(f"Failed to fetch audio URL: {r2.status} - {error_text[:200]}")
                                        raise Exception(f"Failed to download audio from URL (HTTP {r2.status})")
                            
                            # No recognizable audio data found
                            self.log.error(f"JSON response has no audio_base64, audio, url, or audio_url field")
                            raise Exception("API returned JSON but no audio data was found in the response")
                        else:
                            # If API returns raw audio bytes directly
                            blob = await resp.read()
                            if not blob:
                                self.log.error("Received empty audio response")
                                raise Exception("API returned empty audio data")
                            ext, mime = self._guess_ext_and_mime(output_format)
                            self.log.info(f"Received raw audio bytes: {len(blob)} bytes, format: {ext}, mime: {mime}")
                            return blob, ext, mime
                    except json.JSONDecodeError as e:
                        self.log.error(f"Failed to parse JSON response: {e}")
                        raise Exception(f"Invalid JSON response from API: {str(e)}")
                    except Exception as e:
                        self.log.error(f"Failed processing SFX response: {type(e).__name__}: {e}")
                        raise
        except aiohttp.ClientTimeout:
            self.log.error(f"Request timed out after {self.valves.TIMEOUT} seconds")
            raise Exception(f"Request timed out after {self.valves.TIMEOUT} seconds. Try increasing the TIMEOUT setting.")
        except aiohttp.ClientError as e:
            self.log.error(f"Network error during API request: {type(e).__name__}: {e}")
            raise

    async def _save_audio_file(
        self,
        audio_data: bytes,
        prompt: str,
        output_format: str,
        ext: str,
        content_type: str,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """Save audio file to Open WebUI storage.
        
        Args:
            audio_data: Raw audio bytes
            prompt: Original prompt text
            output_format: Audio format string
            ext: File extension
            content_type: MIME type
            user_id: User ID for file ownership
        
        Returns:
            File ID if successful, None otherwise
        """
        try:
            from open_webui.models.files import Files, FileForm
            from open_webui.storage.provider import Storage

            # Create safe filename from prompt
            safe_prompt = "".join(c for c in prompt[:40] if c.isalnum() or c in (" ", "-", "_")).rstrip().replace(" ", "_")
            if not safe_prompt:
                safe_prompt = "sound_effect"
            timestamp = int(time.time())
            filename = f"sfx_{safe_prompt}_{timestamp}{ext}"
            
            self.log.info(f"Saving audio file: {filename} ({len(audio_data)} bytes)")

            # Create temp file
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(audio_data)
                    tmp_path = tmp.name
                    self.log.info(f"Created temp file: {tmp_path}")
                
                # Upload via Storage
                with open(tmp_path, "rb") as f:
                    file_data, file_path = Storage.upload_file(
                        f,
                        filename,
                        {"content_type": content_type, "source": "elevenlabs_sfx"},
                    )
                    self.log.info(f"Uploaded to storage: {file_path}")

                # Create DB record
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
                            "source": "elevenlabs_sfx",
                            "output_format": output_format,
                            "prompt": prompt[:200],
                        },
                    ),
                )
                if record:
                    self.log.info(f"Successfully saved audio file with ID: {record.id}")
                    return record.id
                else:
                    self.log.error("Failed to create database record for audio file")
                    return None
            finally:
                # Clean up temp file
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                        self.log.info(f"Cleaned up temp file: {tmp_path}")
                    except Exception as cleanup_error:
                        self.log.warning(f"Failed to clean up temp file {tmp_path}: {cleanup_error}")
        except ImportError as e:
            self.log.error(f"Failed to import required modules for file storage: {e}")
            return None
        except Exception as e:
            self.log.exception(f"Unexpected error saving audio file: {type(e).__name__}: {e}")
            return None
