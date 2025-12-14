"""title: ElevenLabs Podcast Generator
author: open-webui
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 1.3
license: MIT
description: Convert text into a NotebookLM-style 2-person conversational podcast using ElevenLabs voices. Uses LLM to generate natural dialogue between two hosts.
requirements: aiohttp, cryptography, pydantic, imageio-ffmpeg
icon_url: data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIiIGhlaWdodD0iMzIiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGNpcmNsZSBjeD0iMTAiIGN5PSIxNiIgcj0iNiIgZmlsbD0iIzRDNEM0QyIvPjxjaXJjbGUgY3g9IjIyIiBjeT0iMTYiIHI9IjYiIGZpbGw9IiM0QzRDNEMiLz48L3N2Zz4=
required_open_webui_version: 0.3.10
"""

import os
import re
import time
import json
import base64
import hashlib
import logging
import tempfile
import uuid
import asyncio
import subprocess
import aiohttp
import imageio_ffmpeg
from pathlib import Path
from typing import Optional, Callable, Awaitable, Any, Dict, List, Union, Tuple
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from cryptography.fernet import Fernet, InvalidToken
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
        # ElevenLabs API Configuration
        ELEVEN_API_KEY: EncryptedStr = Field(
            default="",
            description="ElevenLabs API key (xi-api-key). Get it from https://elevenlabs.io/"
        )
        API_BASE_URL: str = Field(
            default="https://api.elevenlabs.io/v1",
            description="Base URL for ElevenLabs API",
        )
        
        # Voice Configuration - Host A (Default: Lily)
        VOICE_A_ID: str = Field(
            default="pFZP5JQG7iQjIQuC4Bku",
            description="Voice ID for Host A (default: Lily). Find voices at https://elevenlabs.io/voice-library",
        )
        VOICE_A_NAME: str = Field(
            default="Lily",
            description="Name for Host A in dialogue (e.g., Lily, Sarah, Emma)",
        )
        
        # Voice Configuration - Host B (Default: Daniel)
        VOICE_B_ID: str = Field(
            default="onwK4e9ZLuTAKqWW03F9",
            description="Voice ID for Host B (default: Daniel). Common: Adam=pNInz6obpgDQGcFmaJgB, Josh=TxGEqnHWrfWFTfGW9XjX",
        )
        VOICE_B_NAME: str = Field(
            default="Daniel",
            description="Name for Host B in dialogue (e.g., Daniel, Mike, Alex)",
        )
        
        # TTS Settings
        MODEL_ID: str = Field(
            default="eleven_multilingual_v2",
            description="TTS model to use (e.g., eleven_multilingual_v2, eleven_turbo_v2_5)",
        )
        STABILITY: float = Field(
            default=0.6,
            description="Voice stability (0.0-1.0). Higher = more consistent.",
        )
        SIMILARITY_BOOST: float = Field(
            default=0.75,
            description="Similarity boost (0.0-1.0). Higher = closer to original voice.",
        )
        
        # LLM Configuration for Dialogue Generation
        OPENAI_API_KEY: EncryptedStr = Field(
            default="",
            description="(Optional) OpenAI API key for direct API calls. If set, bypasses socket.io 60s timeout. Get from https://platform.openai.com/api-keys",
        )
        LLM_MODEL: str = Field(
            default="gpt-4o-mini",
            description="LLM model for dialogue generation. If OPENAI_API_KEY is set: use OpenAI model name (gpt-4o-mini, gpt-3.5-turbo). Otherwise: use Open WebUI format (openai/gpt-4o-mini) with 60s limit.",
        )
        LLM_TIMEOUT: int = Field(
            default=120,
            description="Timeout in seconds for LLM dialogue generation (only applies with direct API key, not event_call)",
        )
        MAX_DIALOGUE_TURNS: int = Field(
            default=12,
            description="Maximum number of dialogue turns (back-and-forth exchanges)",
        )
        
        # Podcast Style
        PODCAST_STYLE: str = Field(
            default="engaging",
            description="Podcast style: engaging, educational, casual, or professional",
        )
        
        # Processing Settings
        TIMEOUT: int = Field(
            default=180,
            description="Max seconds to wait for podcast generation (needs to be high for multiple TTS calls)",
        )
        EMIT_INTERVAL: float = Field(
            default=1.0,
            description="Interval in seconds between status emissions",
        )
        ENABLE_STATUS_INDICATOR: bool = Field(
            default=True,
            description="Enable or disable status indicator emissions",
        )
        CONCATENATE_AUDIO: bool = Field(
            default=True,
            description="Merge all podcast segments into one audio file using ffmpeg (requires imageio-ffmpeg)",
        )

    def __init__(self):
        self.name = "ElevenLabs Podcast"
        self.valves = self.Valves()
        self.log = logging.getLogger("elevenlabs_podcast_pipeline")
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
        """Handle errors with short status updates and detailed chat messages."""
        await self.emit_status(event_emitter, "error", f"❌ {short_status}", True)
        self.log.error(detailed_message)
        full_response = f"❌ **{short_status}**\n\n{detailed_message}"
        body["messages"].append({"role": "assistant", "content": full_response})
        return full_response

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
                "Get your API key from: https://elevenlabs.io/"
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
                "- Invalid voice ID\n"
                "- Text too long or empty\n"
                "- Invalid voice settings (stability/similarity_boost must be 0.0-1.0)\n"
                "- Invalid model ID\n\n"
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

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
        __event_call__: Callable[[dict], Awaitable[dict]] = None,
    ) -> Optional[str]:
        """Main pipeline: Convert text to podcast dialogue with 2 voices"""
        self.log.info(f"=== ElevenLabs Podcast Pipeline Called ===")
        self.log.info(f"Body keys: {list(body.keys())}")
        
        await self.emit_status(
            __event_emitter__, "info", "🎙️ Initializing podcast generator..."
        )
        
        # Validate API key
        api_key = self.valves.ELEVEN_API_KEY.get_decrypted()
        if not api_key:
            return await self.handle_error(
                body, __event_emitter__,
                "Configuration error",
                "ElevenLabs API key not configured.\n\n"
                "**Solution:** Set your `ELEVEN_API_KEY` in the pipeline settings.\n\n"
                "Get your API key from: https://elevenlabs.io/"
            )
        
        # Extract input text - check user message first (for explicit requests), then assistant
        messages = body.get("messages", [])
        self.log.info(f"Total messages in body: {len(messages)}")
        
        # Debug: log last few messages
        for i, msg in enumerate(messages[-3:] if len(messages) > 3 else messages):
            role = msg.get("role", "unknown")
            content_preview = str(msg.get("content", ""))[:50]
            self.log.info(f"Message {i}: role={role}, content={content_preview}...")
        
        input_text = None
        
        # Try user message first (for "convert this article" requests)
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) > 100:  # Meaningful content
                    input_text = content
                    self.log.info("Using user message as input")
                    break
                elif isinstance(content, list):
                    # Handle list-type content (for robustness)
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_content = item.get("text", "")
                            if len(text_content) > 100:
                                input_text = text_content
                                self.log.info("Using user message (list format) as input")
                                break
                    if input_text:
                        break
        
        # Fallback to assistant message (for auto-converting responses)
        if not input_text:
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        input_text = content
                        self.log.info("Using assistant message as input")
                        break
                    elif isinstance(content, list):
                        # Handle list-type content (for robustness)
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                input_text = item.get("text", "")
                                self.log.info("Using assistant message (list format) as input")
                                break
                        if input_text:
                            break
        
        if not input_text or len(input_text) < 50:
            return await self.handle_error(
                body, __event_emitter__,
                "Invalid request",
                "No suitable text found to convert into a podcast.\n\n"
                "**Solution:** Provide text content (at least 50 characters) such as:\n"
                "- An article or blog post\n"
                "- Documentation or notes\n"
                "- An assistant response you'd like as a podcast\n\n"
                "**Example:** 'Convert this article about AI into a podcast: [article text]'"
            )
        
        self.log.info(f"Input text length: {len(input_text)} characters")
        
        # Step 1: Generate dialogue using LLM
        await self.emit_status(
            __event_emitter__, "info", f"🤖 Generating dialogue with {self.valves.LLM_MODEL}..."
        )
        
        try:
            dialogue_text = await self._generate_dialogue(
                input_text,
                self.valves.PODCAST_STYLE,
                self.valves.LLM_MODEL,
                self.valves.MAX_DIALOGUE_TURNS,
                __event_call__,
            )
            
            if not dialogue_text:
                return await self.handle_error(
                    body, __event_emitter__,
                    "Dialogue generation failed",
                    "Failed to generate podcast dialogue.\n\n"
                    "**Possible causes:**\n"
                    "- LLM call timed out (takes >120 seconds)\n"
                    "- LLM model not available or configured incorrectly\n"
                    "- Input text too long or complex\n"
                    "- LLM refused to generate content\n\n"
                    f"**Current LLM model:** {self.valves.LLM_MODEL}\n\n"
                    "**Solutions:**\n"
                    "- Try a faster model (gpt-4o-mini, claude-3-5-haiku)\n"
                    "- Reduce input text length (currently limited to 3000 chars)\n"
                    "- Check your LLM configuration and model availability\n\n"
                    "**Tip:** Check the logs for specific timeout or model errors."
                )
            
            # Step 2: Parse dialogue into segments
            segments = self._parse_dialogue(dialogue_text)
            
            if not segments:
                return await self.handle_error(
                    body, __event_emitter__,
                    "Dialogue parsing failed",
                    "Failed to parse dialogue into speaker segments.\n\n"
                    "**LLM output format was incorrect.** Expected format:\n"
                    f"{self.valves.VOICE_A_NAME.upper()}: text\n"
                    f"{self.valves.VOICE_B_NAME.upper()}: text\n\n"
                    "**Solution:** Try again or adjust your LLM model."
                )
            
            self.log.info(f"Parsed {len(segments)} dialogue segments")
            
            # Step 3: Generate TTS for each segment
            await self.emit_status(
                __event_emitter__, "info", f"🎤 Generating audio for {len(segments)} segments..."
            )
            
            audio_segments = []
            for i, segment in enumerate(segments):
                speaker = segment["speaker"]
                text = segment["text"]
                voice_id = self.valves.VOICE_A_ID if speaker == self.valves.VOICE_A_NAME.upper() else self.valves.VOICE_B_ID
                
                self.log.info(f"Generating segment {i+1}/{len(segments)}: {speaker} ({len(text)} chars)")
                
                await self.emit_status(
                    __event_emitter__, "info", 
                    f"🎤 Generating audio {i+1}/{len(segments)}: {speaker}..."
                )
                
                audio_bytes = await self._generate_tts(
                    api_key,
                    text,
                    voice_id,
                    self.valves.MODEL_ID,
                    self.valves.STABILITY,
                    self.valves.SIMILARITY_BOOST,
                )
                
                if audio_bytes:
                    audio_segments.append({
                        "speaker": speaker,
                        "text": text,
                        "audio": audio_bytes,
                        "duration_estimate": len(text) / 15.0  # Rough estimate: ~15 chars/second
                    })
                else:
                    self.log.warning(f"Failed to generate audio for segment {i+1}")
            
            if not audio_segments:
                return await self.handle_error(
                    body, __event_emitter__,
                    "Audio generation failed",
                    "Failed to generate audio for any segments.\n\n"
                    "**Possible causes:**\n"
                    "- Invalid API key or insufficient credits\n"
                    "- API service temporarily unavailable\n"
                    "- Invalid voice IDs\n\n"
                    "**Solution:** Check the logs for details and verify your ElevenLabs account."
                )
            
            # Step 4: Save audio file (concatenated)
            if self.valves.CONCATENATE_AUDIO:
                await self.emit_status(
                    __event_emitter__, "info", f"🔗 Merging {len(audio_segments)} segments into one file..."
                )
                
                merged_file_id = await self._concatenate_audio_segments(
                    audio_segments,
                    __user__,
                )
                
                if not merged_file_id:
                    self.log.error("Audio concatenation failed")
                    return await self.handle_error(
                        body, __event_emitter__,
                        "Audio concatenation failed",
                        "Failed to merge podcast segments into one file.\n\n"
                        "**Possible causes:**\n"
                        "- ffmpeg not available or failed (requires imageio-ffmpeg)\n"
                        "- Corrupted audio segment\n"
                        "- Disk space or permission issues on temp directory\n\n"
                        "**Solutions:**\n"
                        "1. Install ffmpeg: `pip install imageio-ffmpeg`\n"
                        "2. Check logs for specific ffmpeg errors\n"
                        "3. Verify disk space and temp directory permissions\n"
                        "4. Disable CONCATENATE_AUDIO valve if you want individual files"
                    )
                
                file_ids = [merged_file_id]  # Single merged file only
            else:
                # Save individual segments (only if user explicitly disabled concatenation)
                file_ids = await self._save_individual_segments(audio_segments, __user__, __event_emitter__)
            
            # Step 5: Build response with audio players
            total_duration = sum(seg["duration_estimate"] for seg in audio_segments)
            total_size_mb = sum(len(seg["audio"]) for seg in audio_segments) / (1024 * 1024)
            
            if self.valves.CONCATENATE_AUDIO and len(file_ids) == 1 and file_ids[0]:
                # Single merged file
                audio_url = f"/api/v1/files/{file_ids[0]}/content"
                response_lines = [
                    f"🎙️ **Podcast Generated!** ({len(audio_segments)} segments merged, ~{total_duration/60:.1f} minutes, {total_size_mb:.1f}MB)\n\n",
                    f"🎧 **Full Podcast Audio:**\n",
                    f'<audio controls src="{audio_url}" />\n\n',
                    f"[Download Full Podcast]({audio_url})\n\n",
                    f"**Dialogue Breakdown:**\n"
                ]
                
                for i, segment in enumerate(audio_segments):
                    response_lines.append(
                        f"{i+1}. **{segment['speaker']}:** \"{segment['text'][:80]}{'...' if len(segment['text']) > 80 else ''}\"\n"
                    )
            else:
                # Individual segment files
                response_lines = [
                    f"🎙️ **Podcast Generated!** ({len(audio_segments)} segments, ~{total_duration/60:.1f} minutes, {total_size_mb:.1f}MB)\n"
                ]
                
                for i, (segment, file_id) in enumerate(zip(audio_segments, file_ids)):
                    if file_id:
                        audio_url = f"/api/v1/files/{file_id}/content"
                        response_lines.append(
                            f"\n**{segment['speaker']}:** \"{segment['text'][:100]}{'...' if len(segment['text']) > 100 else ''}\"\n"
                            f'<audio controls src="{audio_url}" />\n'
                            f"[Download]({audio_url})\n"
                        )
                    else:
                        response_lines.append(
                            f"\n**{segment['speaker']}:** \"{segment['text'][:100]}\" ❌ (failed to save)\n"
                        )
            
            await self.emit_status(
                __event_emitter__, "info", "✅ Podcast generation complete!", True
            )
            
            response_content = "\n".join(response_lines)
            body["messages"].append({"role": "assistant", "content": response_content})
            return response_content
            
        except aiohttp.ClientError as e:
            self.log.exception(f"Network error during podcast generation: {e}")
            return await self.handle_error(
                body, __event_emitter__,
                "Network error",
                f"Network error occurred during podcast generation.\n\n"
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
        except Exception as e:
            self.log.exception("Unexpected error during podcast generation")
            return await self.handle_error(
                body, __event_emitter__,
                "Unexpected error",
                f"An unexpected error occurred during podcast generation.\n\n"
                f"**Error details:** {str(e)}\n\n"
                f"**Error type:** {type(e).__name__}\n\n"
                f"Please check the logs for more information."
            )

    async def _generate_dialogue(
        self,
        text: str,
        style: str,
        model: str,
        max_turns: int,
        event_call: Optional[Callable[[dict], Awaitable[dict]]] = None,
    ) -> Optional[str]:
        """Use LLM to convert input text into conversational dialogue."""
        voice_a_name = self.valves.VOICE_A_NAME
        voice_b_name = self.valves.VOICE_B_NAME
        
        system_prompt = f"""You are a podcast script writer. Convert the provided text into a natural, {style} conversation between two hosts: {voice_a_name} and {voice_b_name}.

Guidelines:
- Make it conversational and {style}
- {voice_a_name} typically introduces topics and asks questions
- {voice_b_name} provides insights and follow-up questions
- Include natural transitions, reactions, and enthusiasm
- Break down complex topics into digestible chunks
- Keep segments relatively short (2-4 sentences per speaker turn)
- Maximum {max_turns} total turns (back-and-forth exchanges)

Format your response EXACTLY as:
{voice_a_name.upper()}: [dialogue text]
{voice_b_name.upper()}: [dialogue text]
{voice_a_name.upper()}: [dialogue text]
...

CRITICAL: Use ONLY the speaker name in CAPS followed by colon. No stage directions, sound effects, or other text."""

        user_prompt = f"""Convert the following text into a {style} podcast conversation between {voice_a_name} and {voice_b_name}:

{text[:3000]}

Remember: Format as "{voice_a_name.upper()}: dialogue" ONLY. Make it natural and {style}!"""

        # Check if we should use direct API or event_call
        openai_key = self.valves.OPENAI_API_KEY.get_decrypted() if hasattr(self.valves, 'OPENAI_API_KEY') else None
        
        if not openai_key and not event_call:
            self.log.error("No event_call provided and no OPENAI_API_KEY set - cannot call LLM")
            return None
        
        try:
            self.log.info(f"Calling LLM for dialogue generation: model={model}")
            
            import asyncio
            import time
            start_time = time.time()
            
            # Option 1: Direct OpenAI API call (no timeout limit)
            if openai_key:
                self.log.info(f"Using direct OpenAI API with model={model}, timeout={self.valves.LLM_TIMEOUT}s")
                try:
                    timeout_config = aiohttp.ClientTimeout(total=self.valves.LLM_TIMEOUT)
                    async with aiohttp.ClientSession(timeout=timeout_config) as session:
                        async with session.post(
                            "https://api.openai.com/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {openai_key}",
                                "Content-Type": "application/json",
                            },
                            json={
                                "model": model,
                                "messages": [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}
                                ],
                                "temperature": 0.7,
                            },
                        ) as resp:
                            if resp.status != 200:
                                error_text = await resp.text()
                                self.log.error(f"OpenAI API error {resp.status}: {error_text[:500]}")
                                return None
                            
                            result = await resp.json()
                            elapsed = time.time() - start_time
                            self.log.info(f"OpenAI API response received in {elapsed:.1f}s")
                            
                            if "choices" in result and len(result["choices"]) > 0:
                                response = result["choices"][0]["message"]["content"]
                            else:
                                self.log.error(f"Unexpected OpenAI response format: {result}")
                                return None
                except aiohttp.ClientTimeout:
                    elapsed = time.time() - start_time
                    self.log.error(f"OpenAI API timed out after {elapsed:.1f}s (timeout: {self.valves.LLM_TIMEOUT}s)")
                    self.log.info("SOLUTION: Increase LLM_TIMEOUT or reduce MAX_DIALOGUE_TURNS")
                    return None
                except Exception as e:
                    self.log.exception(f"Error calling OpenAI API directly: {e}")
                    return None
            
            # Option 2: Use Open WebUI's event_call (60s socket.io limit)
            else:
                self.log.info(f"Using Open WebUI event_call (60s socket.io limit) - model={model}")
                self.log.warning("TIP: Set OPENAI_API_KEY in Valves to bypass 60s timeout limit")
                try:
                    self.log.info(f"Sending event_call with model={model}, text_len={len(text)}, max_turns={max_turns}")
                    result = await event_call({
                        "type": "call",
                        "data": {
                            "model": model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            "stream": False,
                        },
                    })
                    elapsed = time.time() - start_time
                    self.log.info(f"Event_call response received in {elapsed:.1f}s")
                    
                    # Parse event_call response
                    if isinstance(result, dict):
                        if "content" in result:
                            response = result["content"]
                        elif "choices" in result and len(result["choices"]) > 0:
                            response = result["choices"][0]["message"]["content"]
                        elif "message" in result and "content" in result["message"]:
                            response = result["message"]["content"]
                        else:
                            self.log.error(f"Unexpected response format: {result}")
                            return None
                    elif isinstance(result, str):
                        response = result
                    else:
                        self.log.error(f"Unexpected response type: {type(result)}")
                        return None
                        
                except asyncio.TimeoutError:
                    elapsed = time.time() - start_time
                    self.log.error(f"LLM call timed out after {elapsed:.1f}s (socket.io 60s limit)")
                    self.log.error(f"Model: {model} | Text length: {len(text)} chars | Max turns: {max_turns}")
                    self.log.info("SOLUTIONS: 1) Set OPENAI_API_KEY in Valves to bypass timeout, 2) Reduce MAX_DIALOGUE_TURNS to 6")
                    return None
                except Exception as e:
                    elapsed = time.time() - start_time
                    if "TimeoutError" in str(type(e)):
                        self.log.error(f"Socket.io timeout after {elapsed:.1f}s - model {model} too slow")
                        self.log.error(f"Text: {len(text)} chars | Max turns: {max_turns}")
                        self.log.info("SOLUTIONS: 1) Set OPENAI_API_KEY in Valves to bypass timeout, 2) Reduce MAX_DIALOGUE_TURNS to 6")
                        return None
                    raise
            
            # Response is already extracted as string
            if isinstance(response, str):
                self.log.info(f"LLM generated dialogue: {len(response)} characters")
                return response
            else:
                self.log.error(f"Unexpected final response type: {type(response)}")
                return None
                
        except Exception as e:
            self.log.exception(f"Error calling LLM for dialogue generation: {e}")
            return None

    def _parse_dialogue(self, dialogue_text: str) -> List[Dict[str, str]]:
        """Parse dialogue text into segments with speaker and text."""
        segments = []
        voice_a_name = self.valves.VOICE_A_NAME.upper()
        voice_b_name = self.valves.VOICE_B_NAME.upper()
        
        # Split by lines and parse
        lines = dialogue_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to match "SPEAKER: text" pattern
            match = re.match(r'^([A-Z]+):\s*(.+)$', line)
            if match:
                speaker = match.group(1)
                text = match.group(2).strip()
                
                # Validate speaker name
                if speaker in [voice_a_name, voice_b_name]:
                    segments.append({
                        "speaker": speaker,
                        "text": text
                    })
                    self.log.debug(f"Parsed segment: {speaker}: {text[:50]}...")
                else:
                    self.log.warning(f"Unknown speaker '{speaker}' in line: {line[:50]}...")
            else:
                self.log.warning(f"Failed to parse line: {line[:50]}...")
        
        self.log.info(f"Parsed {len(segments)} segments from dialogue")
        return segments

    async def _generate_tts(
        self,
        api_key: str,
        text: str,
        voice_id: str,
        model_id: str,
        stability: float,
        similarity_boost: float,
    ) -> Optional[bytes]:
        """Generate TTS audio using ElevenLabs API."""
        url = f"{self.valves.API_BASE_URL}/text-to-speech/{voice_id}"
        
        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
            },
        }
        
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        }
        
        timeout = aiohttp.ClientTimeout(total=self.valves.TIMEOUT)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        formatted_error = self._format_api_error(resp.status, error_text)
                        self.log.error(f"ElevenLabs TTS API error {resp.status}: {formatted_error}")
                        return None
                    
                    audio_bytes = await resp.read()
                    if not audio_bytes:
                        self.log.error("Received empty audio response")
                        return None
                    
                    self.log.info(f"Generated audio: {len(audio_bytes)} bytes")
                    return audio_bytes
        except aiohttp.ClientTimeout:
            self.log.error(f"Request timed out after {self.valves.TIMEOUT} seconds")
            return None
        except aiohttp.ClientError as e:
            self.log.error(f"Network error during API request: {type(e).__name__}: {e}")
            return None

    async def _save_podcast_segments(
        self,
        segments: List[Dict],
        user_id: Optional[str] = None,
    ) -> List[Optional[str]]:
        """Save all podcast audio segments and return list of file IDs."""
        file_ids = []
        
        for i, segment in enumerate(segments):
            try:
                from open_webui.models.files import Files, FileForm
                from open_webui.storage.provider import Storage
                
                speaker = segment["speaker"]
                text = segment["text"]
                audio_data = segment["audio"]
                
                # Create filename
                safe_text = "".join(c for c in text[:30] if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")
                if not safe_text:
                    safe_text = "segment"
                timestamp = int(time.time())
                filename = f"podcast_{speaker.lower()}_{i+1}_{safe_text}_{timestamp}.mp3"
                
                # Create temp file
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                        tmp.write(audio_data)
                        tmp_path = tmp.name
                    
                    # Upload via Storage
                    with open(tmp_path, "rb") as f:
                        file_data, file_path = Storage.upload_file(
                            f,
                            filename,
                            {"content_type": "audio/mpeg", "source": "elevenlabs_podcast"},
                        )
                    
                    # Create DB record
                    file_id = str(uuid.uuid4())
                    record = FilesDB.insert_new_file(
                        user_id or "system",
                        FileForm(
                            id=file_id,
                            filename=filename,
                            path=file_path,
                            meta={
                                "name": filename,
                                "content_type": "audio/mpeg",
                                "size": len(audio_data),
                                "source": "elevenlabs_podcast",
                                "speaker": speaker,
                                "segment_number": i + 1,
                                "text": text[:200],
                            },
                        ),
                    )
                    if record:
                        file_ids.append(record.id)
                    else:
                        file_ids.append(None)
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass
            except Exception as e:
                self.log.exception(f"Error saving segment {i+1}: {e}")
                file_ids.append(None)
        
        return file_ids

    async def _save_individual_segments(
        self,
        audio_segments: List[Dict[str, Any]],
        __user__: dict,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> List[Optional[str]]:
        """Save individual audio segments with status updates."""
        await self.emit_status(
            __event_emitter__, "info", f"💾 Saving {len(audio_segments)} audio files..."
        )
        
        user_id = __user__.get("id") if isinstance(__user__, dict) else None
        file_ids = await self._save_podcast_segments(audio_segments, user_id)
        
        return file_ids

    async def _concatenate_audio_segments(
        self,
        audio_segments: List[Dict[str, Any]],
        __user__: dict,
    ) -> Optional[str]:
        """Concatenate all audio segments into one MP3 file using ffmpeg."""
        try:
            self.log.info(f"Concatenating {len(audio_segments)} audio segments")
            
            # Get ffmpeg binary from imageio-ffmpeg
            try:
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                self.log.info(f"Using ffmpeg: {ffmpeg_exe}")
            except Exception as e:
                self.log.error(f"Failed to get ffmpeg binary: {e}")
                self.log.info("Install imageio-ffmpeg: pip install imageio-ffmpeg")
                return None
            
            # Create temp directory for intermediate files
            temp_dir = Path(tempfile.mkdtemp(prefix="podcast_concat_"))
            temp_files = []
            concat_list_path = None
            output_path = None
            
            try:
                # Save each segment as a temporary MP3 file
                for i, segment in enumerate(audio_segments):
                    if not segment.get("audio"):
                        self.log.warning(f"Segment {i+1} has no audio, skipping")
                        continue
                    
                    temp_file = temp_dir / f"segment_{i:03d}.mp3"
                    with open(temp_file, "wb") as f:
                        f.write(segment["audio"])
                    temp_files.append(temp_file)
                    self.log.debug(f"Saved segment {i+1} to {temp_file}")
                
                if not temp_files:
                    self.log.error("No audio segments to concatenate")
                    return None
                
                # Create concat list file for ffmpeg
                concat_list_path = temp_dir / "concat_list.txt"
                with open(concat_list_path, "w", encoding="utf-8") as f:
                    for temp_file in temp_files:
                        # Use absolute paths and escape quotes for ffmpeg
                        abs_path = str(temp_file.absolute()).replace("\\", "/")
                        f.write(f"file '{abs_path}'\n")
                
                self.log.debug(f"Created concat list: {concat_list_path}")
                
                # Output file
                output_path = temp_dir / "podcast_merged.mp3"
                
                # Run ffmpeg concat
                cmd = [
                    ffmpeg_exe,
                    "-y",  # Overwrite output
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(concat_list_path),
                    "-c", "copy",  # Copy codec (fast, no re-encoding)
                    str(output_path),
                ]
                
                self.log.info(f"Running ffmpeg concat: {' '.join(cmd)}")
                
                # Run ffmpeg in background thread
                def run_ffmpeg():
                    return subprocess.run(cmd, capture_output=True)
                
                completed = await asyncio.to_thread(run_ffmpeg)
                
                stderr = completed.stderr.decode(errors="ignore") if completed.stderr else ""
                
                if completed.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
                    self.log.error(f"ffmpeg concat failed (return code: {completed.returncode})")
                    self.log.error(f"ffmpeg stderr: {stderr[:500]}")
                    return None
                
                self.log.info(f"ffmpeg concat successful: {output_path.stat().st_size} bytes")
                
                # Upload merged file to Open WebUI
                user_id = __user__.get("id") if isinstance(__user__, dict) else None
                timestamp = int(time.time())
                filename = f"podcast_merged_{timestamp}.mp3"
                
                with open(output_path, "rb") as f:
                    file_data, file_path = Storage.upload_file(
                        f,
                        filename,
                        {
                            "content_type": "audio/mpeg",
                            "source": "elevenlabs_podcast_merged",
                            "segments_count": len(audio_segments),
                        },
                    )
                
                self.log.info(f"Uploaded merged podcast: {file_path}")
                
                # Create DB record
                from open_webui.models.files import FileForm
                
                file_id = str(uuid.uuid4())
                record = FilesDB.insert_new_file(
                    user_id or "system",
                    FileForm(
                        id=file_id,
                        filename=filename,
                        path=file_path,
                        meta={
                            "name": filename,
                            "content_type": "audio/mpeg",
                            "size": output_path.stat().st_size,
                            "source": "elevenlabs_podcast_merged",
                            "segments_count": len(audio_segments),
                            "speakers": f"{self.valves.VOICE_A_NAME}, {self.valves.VOICE_B_NAME}",
                        },
                    ),
                )
                
                if record:
                    self.log.info(f"Created DB record for merged podcast: {record.id}")
                    return record.id
                else:
                    self.log.error("Failed to create DB record for merged podcast")
                    return None
                    
            finally:
                # Cleanup temp files
                try:
                    for temp_file in temp_files:
                        if temp_file.exists():
                            temp_file.unlink()
                    if concat_list_path and concat_list_path.exists():
                        concat_list_path.unlink()
                    if output_path and output_path.exists():
                        output_path.unlink()
                    temp_dir.rmdir()
                    self.log.debug("Cleaned up temporary files")
                except Exception as cleanup_error:
                    self.log.warning(f"Failed to cleanup temp files: {cleanup_error}")
                    
        except Exception as e:
            self.log.exception(f"Error concatenating audio segments: {e}")
            return None
