"""
title: Google Gemini Video Understanding (Chat Pipeline)
author: open-webui
date: 2025-09-20
version: 1.1
license: MIT
description: A pipeline for video understanding using Google Gemini with conversation history and Open WebUI file handling. Supports YouTube videos, local uploads, and HTTP video URLs with robust error handling.
requirements: google-genai, cryptography, requests
references:
  - https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/video-analysis/youtube_video_analysis.ipynb
  - https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/video-understanding
"""

import base64
import json
import mimetypes
import os
import hashlib
import re
import requests
import tempfile
import time
import logging
from typing import Optional, Any, List, Dict, Callable, Awaitable
from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import core_schema

# Google Gemini imports
from google import genai
from google.genai import types

# Open WebUI imports
from open_webui.models.files import Files as FilesDB
from open_webui.storage.provider import Storage


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

    def get_decrypted(self) -> str:
        """Instance method to decrypt the stored value"""
        return self.decrypt(self)

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


class Pipe:
    class Valves(BaseModel):
        # Authentication options
        use_vertex_ai: bool = Field(
            default=False,
            description="Use Vertex AI instead of Google AI API. Requires service account or ADC.",
        )
        api_key: EncryptedStr = Field(
            default="",
            description="Your Google AI API key (only used if use_vertex_ai is False)",
        )
        project_id: str = Field(
            default="",
            description="Google Cloud project ID (for Vertex AI). Defaults to GOOGLE_CLOUD_PROJECT env var.",
        )
        location: str = Field(
            default="us-central1",
            description="Google Cloud location/region (for Vertex AI). Defaults to GOOGLE_CLOUD_LOCATION env var.",
        )
        SERVICE_ACCOUNT_JSON: EncryptedStr = Field(
            default="",
            description="Service account JSON key content (for Vertex AI authentication). Will be encrypted for security.",
        )
        SERVICE_ACCOUNT_PATH: str = Field(
            default="",
            description="Path to service account JSON file (alternative to SERVICE_ACCOUNT_JSON)",
        )

        # Model configuration
        model_name: str = Field(
            default="gemini-1.5-flash-002",
            description="The Google AI model name for video understanding. Recommended: gemini-2.0-flash-001 for summarization, gemini-1.5-pro for long videos (up to 2 hours)",
        )
        max_history_videos: int = Field(
            default=1,
            description="Maximum number of previous videos to include in context",
        )
        download_timeout: int = Field(
            default=20,
            description="Timeout (seconds) for downloading videos via HTTP",
        )
        retry_attempts: int = Field(
            default=3,
            description="Max retry attempts for transient HTTP errors when downloading media.",
        )
        retry_backoff_base: float = Field(
            default=1.5,
            description="Exponential backoff multiplier between retries when downloading media.",
        )
        use_uri_for_youtube: bool = Field(
            default=True,
            description="When a YouTube URL is detected, pass it to Gemini via Part.from_uri instead of downloading bytes. This is the recommended approach per Google's documentation.",
        )
        youtube_video_length_warning_minutes: int = Field(
            default=60,
            description="Warn if YouTube video appears to be longer than this many minutes. Gemini 1.5 supports up to 1 hour, Gemini 2.0 supports up to 2 hours.",
        )
        use_uri_for_http_media: bool = Field(
            default=False,
            description="If true, send generic http(s) media URLs via Part.from_uri; otherwise download bytes.",
        )
        debug: bool = Field(default=False, description="Enable verbose debug logging")
        EMIT_INTERVAL: float = Field(
            default=0.5, description="Interval in seconds between status emissions"
        )
        ENABLE_STATUS_INDICATOR: bool = Field(
            default=True, description="Enable or disable status indicator emissions"
        )

    def __init__(self):
        self.type = "manifold"
        self.id = "gemini_video_understanding"
        self.name = "Gemini Video Understanding"
        self.valves = self.Valves()
        self.last_emit_time = 0
        self.log = logging.getLogger(self.name.replace(" ", "_").lower())
        self.log.setLevel(logging.INFO)

    async def on_startup(self):
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    def _debug(self, msg: str) -> None:
        if getattr(self.valves, "debug", False):
            print(f"[DEBUG] {msg}")

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
                self._debug(f"Failed to emit status: {e}")

    async def handle_error(
        self,
        body: dict,
        event_emitter: Optional[Callable[[dict], Awaitable[None]]],
        short_status: str,
        detailed_message: str,
    ) -> dict:
        """Handle errors with short status updates and detailed chat messages.
        
        Args:
            body: The request body to append the message to
            event_emitter: Event emitter for status updates
            short_status: Brief error indicator for status (e.g., "Configuration error")
            detailed_message: Full error details for chat response
        
        Returns:
            The body dict with error message appended
        """
        # Emit short status update (limited length)
        await self.emit_status(event_emitter, "error", f"❌ {short_status}", True)
        
        # Log the full error
        self.log.error(detailed_message)
        
        # Add detailed message to chat
        full_response = f"❌ **{short_status}**\n\n{detailed_message}"
        body.setdefault("messages", []).append({"role": "assistant", "content": full_response})
        
        return body

    def _format_error_message(self, error_data: Any, is_youtube: bool = False) -> str:
        """Format error data into a user-friendly message with YouTube-specific detection.
        
        Args:
            error_data: Error information from API
            is_youtube: Whether this error is related to YouTube video processing
        
        Returns:
            Formatted user-friendly error message
        """
        try:
            # Handle dict errors (REST API format)
            if isinstance(error_data, dict):
                code = error_data.get('code', '')
                message = error_data.get('message', str(error_data))
            else:
                # Handle string errors
                message = str(error_data)
                code = ''
            
            message_lower = message.lower()
            
            # Check for safety filter errors
            safety_info = self._detect_safety_filter(message)
            if safety_info:
                return safety_info
            
            # YouTube-specific error handling
            if is_youtube:
                if 'not found' in message_lower or '404' in message_lower:
                    return (
                        "**YouTube Video Not Found**\n\n"
                        "The YouTube video may have been deleted, made private, or the URL is incorrect.\n\n"
                        "**Solutions:**\n"
                        "- Verify the YouTube URL is correct\n"
                        "- Check if the video is public and accessible\n"
                        "- Try a different video"
                    )
                elif 'permission' in message_lower or '403' in message_lower or 'forbidden' in message_lower:
                    return (
                        "**YouTube Video Access Denied**\n\n"
                        "The YouTube video has regional restrictions, requires sign-in, or is age-restricted.\n\n"
                        "**Solutions:**\n"
                        "- Try a different publicly accessible video\n"
                        "- Verify the video doesn't have viewing restrictions\n"
                        "- Check if the video is available in your region"
                    )
                elif 'invalid' in message_lower or 'malformed' in message_lower:
                    return (
                        "**Invalid YouTube URL**\n\n"
                        "The YouTube URL format appears to be invalid.\n\n"
                        "**Supported formats:**\n"
                        "- https://www.youtube.com/watch?v=VIDEO_ID\n"
                        "- https://youtu.be/VIDEO_ID\n"
                        "- https://www.youtube.com/shorts/VIDEO_ID"
                    )
                elif 'quota' in message_lower or 'rate limit' in message_lower:
                    return (
                        "**API Quota Exceeded**\n\n"
                        "The Gemini API quota has been exceeded or rate limit reached for YouTube video processing.\n\n"
                        "**Solutions:**\n"
                        "- Wait a few minutes and try again\n"
                        "- Check your Google Cloud quota limits\n"
                        "- Consider upgrading your API quota"
                    )
            
            # General error patterns
            if 'quota' in message_lower or 'rate limit' in message_lower:
                return f"API quota exceeded or rate limit reached. Please try again later. ({message})"
            
            if 'permission' in message_lower or 'forbidden' in message_lower:
                return f"Permission denied. Please check your API credentials and project access. ({message})"
            
            if 'authentication' in message_lower or 'api key' in message_lower:
                return f"Authentication failed. Please verify your API key or service account credentials. ({message})"
            
            # Default format
            return message or f"Error code {code}"
        except Exception as e:
            self.log.error(f"Error formatting error message: {e}")
            return str(error_data)
    
    def _detect_safety_filter(self, message: str) -> Optional[str]:
        """Detect Gemini safety filter errors and return formatted guidance.
        
        Args:
            message: Error message to check for safety filter indicators
            
        Returns:
            Formatted error message with safety guidance, or None if not a safety filter error
        """
        if not message or not isinstance(message, str):
            return None
        
        # Check for common safety filter error patterns
        lower_msg = message.lower()
        is_safety_error = (
            'violate' in lower_msg and ('usage guidelines' in lower_msg or 'policies' in lower_msg) or
            'safety' in lower_msg and ('filter' in lower_msg or 'block' in lower_msg) or
            'harmful' in lower_msg or
            'inappropriate' in lower_msg or
            'policy violation' in lower_msg or
            'content policy' in lower_msg
        )
        
        if not is_safety_error:
            return None
        
        # Generic safety filter message for video understanding
        return (
            "🛡️ **Content Safety Filter Triggered**\n\n"
            "Your video or prompt may contain content that violates safety guidelines.\n\n"
            "**Common issues:**\n"
            "- Violence, weapons, or dangerous content in the video\n"
            "- Inappropriate or explicit content\n"
            "- Hateful or harmful content\n"
            "- Sensitive personal information\n\n"
            "**What to try:**\n"
            "- Try a different video\n"
            "- Rephrase your prompt to avoid sensitive topics\n"
            "- Ensure the video content is appropriate for all audiences\n\n"
            f"**Details:** {message[:300]}..."
        )

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Any] = None,
        __event_call__: Optional[Any] = None,
        __request__: Optional[Any] = None,
    ) -> Optional[dict]:
        await self.emit_status(
            __event_emitter__, "info", "Initializing Gemini video analyzer..."
        )

        # Validate authentication (API key or Vertex AI)
        if self.valves.use_vertex_ai:
            # Check Vertex AI configuration
            project_id = self.valves.project_id or os.getenv("GOOGLE_CLOUD_PROJECT", "")
            if not project_id:
                return await self.handle_error(
                    body, __event_emitter__,
                    "Configuration error",
                    "Google Cloud Project ID not configured for Vertex AI.\n\n"
                    "**Solutions:**\n"
                    "- Set `project_id` in pipeline settings, or\n"
                    "- Set the `GOOGLE_CLOUD_PROJECT` environment variable"
                )
            
            # Check if service account credentials are provided
            if not self.valves.SERVICE_ACCOUNT_JSON and not self.valves.SERVICE_ACCOUNT_PATH:
                return await self.handle_error(
                    body, __event_emitter__,
                    "Configuration error",
                    "Service account credentials not configured for Vertex AI.\n\n"
                    "**Solutions:**\n"
                    "- Provide `SERVICE_ACCOUNT_JSON` in pipeline settings, or\n"
                    "- Provide `SERVICE_ACCOUNT_PATH` in pipeline settings, or\n"
                    "- Set `use_vertex_ai=False` to use API key authentication instead"
                )
        else:
            # Check API key configuration
            decrypted_key_for_check = EncryptedStr.decrypt(self.valves.api_key)
            if not decrypted_key_for_check:
                return await self.handle_error(
                    body, __event_emitter__,
                    "Configuration error",
                    "Google AI API key not configured.\n\n"
                    "**Solution:** Set your `api_key` in the pipeline settings."
                )

        # Initialize credential cleanup variables before try block
        temp_creds_path = None
        old_creds = None

        try:
            await self.emit_status(
                __event_emitter__, "info", "Processing your request..."
            )

            messages = body.get("messages", [])
            if not messages:
                error_msg = "❌ Error: No messages provided."
                await self.emit_status(__event_emitter__, "error", error_msg, True)
                body.setdefault("messages", []).append(
                    {"role": "assistant", "content": error_msg}
                )
                return body

            last_message = messages[-1]
            if last_message.get("role") != "user":
                error_msg = "❌ Error: Last message must be from user."
                await self.emit_status(__event_emitter__, "error", error_msg, True)
                body["messages"].append({"role": "assistant", "content": error_msg})
                return body

            raw_content = last_message.get("content", "")
            prompt = ""
            video_urls: List[str] = []

            # Parse content
            if isinstance(raw_content, str):
                prompt = raw_content.strip()
                # Extract any inline video URLs
                video_urls.extend(self._extract_video_urls_from_content(prompt))
            elif isinstance(raw_content, list):
                for item in raw_content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text":
                        t = (item.get("text", "") or "").strip()
                        if t:
                            prompt = f"{prompt}\n{t}".strip() if prompt else t
                    elif item.get("type") == "video_url":
                        url = item.get("video_url", {}).get("url", "")
                        if url:
                            video_urls.append(url)
                    elif item.get("type") in {"file", "input_video"}:
                        # Generic file structure with URL + mime
                        meta = item.get("file", {}) or item
                        url = meta.get("url") or meta.get("video_url", {}).get(
                            "url", ""
                        )
                        mime = meta.get("mime") or meta.get("mime_type")
                        if url and (not mime or mime.startswith("video/")):
                            video_urls.append(url)

            # Fallback: search previous assistant/user messages for video URLs (e.g., when user says "Analyze the video above")
            if not video_urls:
                for msg in reversed(messages[:-1]):
                    if len(video_urls) >= self.valves.max_history_videos:
                        break
                    c = msg.get("content", "")
                    if isinstance(c, str):
                        urls = self._extract_video_urls_from_content(c)
                        for u in urls:
                            if len(video_urls) >= self.valves.max_history_videos:
                                break
                            video_urls.append(u)

            if not prompt:
                # Allow video-only requests by providing a default instruction
                prompt = "Analyze the provided video and describe key events, objects, and actions."

            if not video_urls:
                error_msg = "❌ Error: No video provided. Please attach a video or include a video URL."
                await self.emit_status(__event_emitter__, "error", error_msg, True)
                body["messages"].append({"role": "assistant", "content": error_msg})
                return body

            # Build Gemini content
            await self.emit_status(__event_emitter__, "info", "Loading video(s)...")

            parts: List[types.Part] = []
            skipped_videos: List[str] = []  # Track failed video loads
            is_youtube_analysis = False  # Track if we're analyzing YouTube videos
            
            # Limit to configured number of videos
            selected_urls = video_urls[: max(1, int(self.valves.max_history_videos))]
            self.log.info(f"Selected {len(selected_urls)} video(s) for analysis")
            self._debug(f"Selected {len(selected_urls)} video(s) for analysis")
            
            for idx, url in enumerate(selected_urls, 1):
                if len(selected_urls) > 1:
                    await self.emit_status(
                        __event_emitter__,
                        "info",
                        f"Loading video {idx} of {len(selected_urls)}...",
                    )
                
                is_youtube = self._is_youtube_url(url)
                if is_youtube:
                    is_youtube_analysis = True
                    # Validate YouTube URL format
                    is_valid, video_id = self._validate_youtube_url(url)
                    if not is_valid:
                        self.log.warning(f"Invalid YouTube URL format, will attempt anyway: {url}")
                
                if is_youtube and self.valves.use_uri_for_youtube:
                    self.log.info(f"Adding YouTube URL via Part.from_uri: {url}")
                    self._debug(f"Adding YouTube URL via Part.from_uri: {url}")
                    
                    # Use Part.from_uri with mime_type="video/webm" for YouTube videos
                    # This is the recommended approach per Google's documentation:
                    # https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/video-analysis/youtube_video_analysis.ipynb
                    # 
                    # Important notes:
                    # - YouTube videos must be publicly accessible
                    # - Gemini 1.5 models support up to 1 hour of video
                    # - Gemini 2.0 models support up to 2 hours of video
                    # - The video/webm MIME type is correct for all YouTube videos
                    parts.append(
                        types.Part.from_uri(file_uri=url, mime_type="video/webm")
                    )
                    continue
                elif is_youtube and not self.valves.use_uri_for_youtube:
                    # Warn about YouTube download limitation
                    self.log.warning(
                        f"Attempting to download YouTube video {url} directly. "
                        "This typically fails. Consider enabling 'use_uri_for_youtube' setting."
                    )

                if url.startswith("http") and self.valves.use_uri_for_http_media:
                    self.log.info(f"Adding HTTP media URL via Part.from_uri: {url}")
                    self._debug(f"Adding HTTP media URL via Part.from_uri: {url}")
                    # Use a generic video mime; Vertex may infer actual type
                    parts.append(types.Part.from_uri(file_uri=url, mime_type="video/*"))
                    continue

                media = self._download_media(url, __request__)
                if not media:
                    self.log.warning(f"Skipping video due to download failure: {url}")
                    self._debug(f"Skipping video due to download failure: {url}")
                    skipped_videos.append(url)
                    continue
                parts.append(
                    types.Part.from_bytes(
                        data=media["data"], mime_type=media["mime_type"]
                    )
                )
            
            # Warn user about skipped videos
            if skipped_videos:
                skip_msg = f"⚠️ Could not load {len(skipped_videos)} video(s)"
                await self.emit_status(__event_emitter__, "warning", skip_msg)
                self.log.warning(f"Skipped {len(skipped_videos)} video(s): {skipped_videos}")
            
            parts.append(types.Part.from_text(text=prompt))

            if len(parts) <= 1:
                error_msg = "❌ Error: Unable to load the provided video(s)."
                await self.emit_status(__event_emitter__, "error", error_msg, True)
                body["messages"].append({"role": "assistant", "content": error_msg})
                return body

            contents = [types.Content(role="user", parts=parts)]

            # Initialize client with appropriate authentication
            if self.valves.use_vertex_ai:
                project_id = self.valves.project_id or os.getenv(
                    "GOOGLE_CLOUD_PROJECT", ""
                )
                location = self.valves.location or os.getenv(
                    "GOOGLE_CLOUD_LOCATION", "us-central1"
                )

                self.log.info(
                    f"Using Vertex AI with service account authentication, project={project_id}, location={location}"
                )
                self._debug(
                    f"Using Vertex AI with service account authentication, project={project_id}, location={location}"
                )

                # For Vertex AI with service account, set credentials via environment
                if self.valves.SERVICE_ACCOUNT_JSON:
                    service_account_json = (
                        self.valves.SERVICE_ACCOUNT_JSON.get_decrypted()
                    )
                    service_account_info = json.loads(service_account_json)

                    # Write to temp file and set environment variable
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".json", delete=False
                    ) as f:
                        json.dump(service_account_info, f)
                        temp_creds_path = f.name

                    # Set environment variable - will be cleaned up in finally block
                    old_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_creds_path
                else:
                    # Use SERVICE_ACCOUNT_PATH directly
                    old_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
                        self.valves.SERVICE_ACCOUNT_PATH
                    )

                client = genai.Client(
                    vertexai=True,
                    project=project_id,
                    location=location,
                )
            else:
                decrypted_key = EncryptedStr.decrypt(self.valves.api_key)
                self.log.info("Using API key authentication")
                self._debug("Using API key authentication")
                client = genai.Client(api_key=decrypted_key)

            generate_content_config = types.GenerateContentConfig(
                response_modalities=["TEXT"]
            )

            await self.emit_status(
                __event_emitter__, "info", "Analyzing video with Gemini..."
            )

            # Stream response (handle mid-stream errors and preserve partial output)
            streamed_text: List[str] = []
            try:
                for chunk in client.models.generate_content_stream(
                    model=self.valves.model_name,
                    contents=contents,
                    config=generate_content_config,
                ):
                    if not chunk:
                        continue
                    try:
                        # Safely extract text content
                        text_content = getattr(chunk, "text", None)
                        if text_content:
                            streamed_text.append(text_content)
                            if __event_emitter__:
                                await __event_emitter__(
                                    {
                                        "type": "message",
                                        "data": {"content": text_content},
                                    }
                                )
                    except AttributeError as ae:
                        self.log.warning(f"Unexpected chunk structure: {ae}")
                        continue
            except Exception as e:
                # Format error message with context
                error_msg = self._format_error_message(e, is_youtube=is_youtube_analysis)
                
                # Log full error details
                self.log.error(f"Streaming error: {str(e)}")
                import traceback
                self.log.error(traceback.format_exc())
                
                # If streaming fails mid-way, produce a helpful assistant message
                partial = "".join(streamed_text).strip()
                
                if partial:
                    # We got some content before the error
                    await self.emit_status(__event_emitter__, "warning", "Partial response received", True)
                    body.setdefault("messages", []).append(
                        {
                            "role": "assistant",
                            "content": f"⚠️ **Partial Response**\n\n{partial}\n\n---\n\n❌ **Error occurred:**\n{error_msg}",
                        }
                    )
                    return body
                else:
                    # Complete failure, no content
                    await self.emit_status(__event_emitter__, "error", "Analysis failed", True)
                    body.setdefault("messages", []).append(
                        {
                            "role": "assistant",
                            "content": f"❌ **Video Analysis Failed**\n\n{error_msg}",
                        }
                    )
                    return body

            if not streamed_text:
                streamed_text = [
                    "I couldn't extract any information from the provided video."
                ]

            await self.emit_status(
                __event_emitter__, "info", "Video analysis complete!", True
            )

            final_text = "".join(streamed_text).strip()
            body["messages"].append({"role": "assistant", "content": final_text})
            return body

        except Exception as e:
            import traceback
            
            # Log full error details
            self.log.error(f"Error during video understanding: {str(e)}")
            self.log.error(traceback.format_exc())
            
            # Format error for user
            error_msg = self._format_error_message(e)
            
            return await self.handle_error(
                body, __event_emitter__,
                "Unexpected error",
                f"An unexpected error occurred during video analysis.\n\n"
                f"**Error details:** {error_msg}\n\n"
                f"**Error type:** {type(e).__name__}\n\n"
                "Please check the logs for more information."
            )
        finally:
            # Clean up temp credentials file and restore original environment
            if temp_creds_path and os.path.exists(temp_creds_path):
                try:
                    os.unlink(temp_creds_path)
                    self.log.info(f"Cleaned up temp credentials file: {temp_creds_path}")
                except Exception as e:
                    self.log.warning(f"Failed to clean up temp file: {e}")

            # Restore original GOOGLE_APPLICATION_CREDENTIALS
            if old_creds is not None:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = old_creds
            elif "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
                del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

    def _extract_video_urls_from_content(self, content: str) -> List[str]:
        """Extract video URLs from markdown, HTML, or bare text.

        Supports:
        - Markdown links/media
        - HTML video/source tags
        - Bare Open WebUI file URLs (/api/v1/files/{id}/content)
        - Bare http(s) video file URLs (.mp4, .webm, .mov, .m4v)
        - YouTube links (youtube.com, youtu.be)
        """
        urls: List[str] = []

        # 1) Markdown links/media: [text](url) or ![alt](url)
        md_pattern = r"(?:!\[.*?\]|\[.*?\])\((data:video/[^)]+|https?://[^)]+|/[^)]+)\)"
        urls.extend(re.findall(md_pattern, content, flags=re.IGNORECASE))

        # 2) HTML <video src> or <source src>
        html_pattern = r"<(?:video|source)[^>]+src=\"([^\"]+)\"[^>]*>"
        urls.extend(re.findall(html_pattern, content, flags=re.IGNORECASE))

        # 3) Bare file API URLs (absolute or site-relative): /api/v1/files/{id}/content
        file_api_pattern = r"(?:https?://[^\s)]+)?(/api/v1/files/[a-f0-9\-]+/content)"
        file_api_matches = re.findall(file_api_pattern, content, flags=re.IGNORECASE)
        for m in file_api_matches:
            # If the match is site-relative, keep as-is; if absolute was included, the group returns only the path.
            urls.append(m if m.startswith("/") else f"/{m.lstrip('/')}")

        # 4) Bare absolute video URLs with common extensions
        abs_video_pattern = r"https?://[^\s)]+\.(?:mp4|webm|mov|m4v)(?:\?[^\s)]*)?"
        urls.extend(re.findall(abs_video_pattern, content, flags=re.IGNORECASE))

        # 5) Bare site-relative video URLs with common extensions
        rel_video_pattern = r"/[^\s)]+\.(?:mp4|webm|mov|m4v)(?:\?[^\s)]*)?"
        urls.extend(re.findall(rel_video_pattern, content, flags=re.IGNORECASE))

        # 6) YouTube links (absolute) — watch, short youtu.be, shorts
        yt_pattern = r"https?://(?:www\.)?(?:youtube\.com/(?:watch\?v=[^\s&#]+|shorts/[^\s/?#&]+)|youtu\.be/[^\s/?#&]+)"
        urls.extend(re.findall(yt_pattern, content, flags=re.IGNORECASE))

        # Deduplicate while preserving order
        seen = set()
        deduped: List[str] = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                deduped.append(u)

        self.log.info(f"Extractor found {len(deduped)} URL(s): {deduped}")
        self._debug(f"Extractor found {len(deduped)} URL(s): {deduped}")
        return deduped

    def _is_youtube_url(self, url: str) -> bool:
        """Check if URL is a YouTube video URL."""
        try:
            return bool(
                re.match(
                    r"https?://(?:www\.)?(?:youtube\.com|youtu\.be)/",
                    url,
                    flags=re.IGNORECASE,
                )
            )
        except Exception:
            return False
    
    def _validate_youtube_url(self, url: str) -> tuple[bool, Optional[str]]:
        """Validate YouTube URL and extract video ID.
        
        Args:
            url: YouTube URL to validate
            
        Returns:
            Tuple of (is_valid, video_id). video_id is None if invalid.
        """
        try:
            # Extract video ID from different YouTube URL formats
            if "youtube.com/watch" in url:
                match = re.search(r"[?&]v=([^&#]+)", url)
                if match:
                    video_id = match.group(1)
                    self.log.info(f"Extracted YouTube video ID from watch URL: {video_id}")
                    return True, video_id
            elif "youtu.be/" in url:
                match = re.search(r"youtu\.be/([^?&#]+)", url)
                if match:
                    video_id = match.group(1)
                    self.log.info(f"Extracted YouTube video ID from short URL: {video_id}")
                    return True, video_id
            elif "youtube.com/shorts/" in url:
                match = re.search(r"shorts/([^?&#]+)", url)
                if match:
                    video_id = match.group(1)
                    self.log.info(f"Extracted YouTube video ID from shorts URL: {video_id}")
                    return True, video_id
            
            self.log.warning(f"Could not extract video ID from YouTube URL: {url}")
            return False, None
        except Exception as e:
            self.log.error(f"Error validating YouTube URL: {e}")
            return False, None

    def _download_media(
        self, media_url: str, request: Optional[Any] = None
    ) -> Optional[Dict]:
        """Download media (video) from URL, data URI, or Open WebUI file endpoint."""
        is_youtube = self._is_youtube_url(media_url)
        
        try:
            # data URI
            if media_url.startswith("data:"):
                self.log.info("Processing data URI video")
                header, data = media_url.split(",", 1)
                mime_type = header.split(";")[0].split(":")[1]
                media_data = base64.b64decode(data)
                self.log.info(f"Data URI video loaded: {len(media_data)} bytes, type: {mime_type}")
                return {"data": media_data, "mime_type": mime_type}

            # Direct Open WebUI file by ID: /api/v1/files/{id}/content
            try:
                m = re.search(
                    r"/api/v1/files/([a-f0-9\-]+)/content", media_url, re.IGNORECASE
                )
                if m:
                    file_id = m.group(1)
                    self.log.info(f"Attempting to load file by ID: {file_id}")
                    file_model = FilesDB.get_file_by_id(file_id)
                    if file_model and file_model.path:
                        local_path = Storage.get_file(file_model.path)
                        with open(local_path, "rb") as f:
                            data_bytes = f.read()
                        mime_type = None
                        if file_model.meta and isinstance(file_model.meta, dict):
                            mime_type = file_model.meta.get(
                                "content_type"
                            ) or file_model.meta.get("mime_type")
                        if not mime_type:
                            mime_type = (
                                mimetypes.guess_type(file_model.filename or local_path)[
                                    0
                                ]
                                or "video/mp4"
                            )
                        self.log.info(f"File loaded: {len(data_bytes)} bytes, type: {mime_type}")
                        return {"data": data_bytes, "mime_type": mime_type}
            except Exception as e:
                self.log.warning(
                    f"Direct file read by ID failed, will try URL resolution: {e}"
                )

            # Resolve site-relative URLs
            if media_url.startswith("/"):
                base = os.getenv("WEBUI_URL", "").rstrip("/")
                if not base and request is not None:
                    try:
                        proto = request.headers.get("x-forwarded-proto")
                        host = request.headers.get(
                            "x-forwarded-host"
                        ) or request.headers.get("host")
                        if proto and host:
                            base = f"{proto}://{host}"
                        else:
                            base = str(getattr(request, "base_url", "")).rstrip("/")
                    except Exception:
                        base = ""
                if base:
                    media_url = f"{base}{media_url}"
                    self.log.info(f"Resolved relative URL to: {media_url}")
                else:
                    self.log.warning(
                        f"Cannot resolve relative URL (no WEBUI_URL or request base). Skipping: {media_url}"
                    )
                    return None

            # Prepare headers from incoming request for authenticated endpoints
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

            # HTTP/HTTPS fetch with simple retry/backoff for transient errors
            attempts = max(1, int(self.valves.retry_attempts))
            backoff = 1.0
            last_exc: Optional[Exception] = None
            for attempt in range(1, attempts + 1):
                try:
                    response = requests.get(
                        media_url,
                        headers=headers or None,
                        timeout=max(1, int(self.valves.download_timeout)),
                    )
                    # Handle rate limit / server errors as transient
                    if (
                        response.status_code in (429,)
                        or 500 <= response.status_code <= 599
                    ):
                        last_exc = Exception(
                            f"HTTP {response.status_code}: {response.text[:200]}"
                        )
                        if attempt < attempts:
                            # sleep and retry
                            import time as _t

                            _t.sleep(backoff)
                            backoff *= float(self.valves.retry_backoff_base)
                            continue
                        response.raise_for_status()
                    response.raise_for_status()
                    break
                except Exception as ex:
                    last_exc = ex
                    if attempt < attempts:
                        import time as _t

                        _t.sleep(backoff)
                        backoff *= float(self.valves.retry_backoff_base)
                        continue
                    raise

            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("video/"):
                mime_type = mimetypes.guess_type(media_url)[0]
                if not mime_type or not mime_type.startswith("video/"):
                    mime_type = "video/mp4"
            else:
                mime_type = content_type

            self.log.info(f"Video downloaded: {len(response.content)} bytes, type: {mime_type}")
            return {"data": response.content, "mime_type": mime_type}
        except Exception as e:
            if is_youtube:
                self.log.error(
                    f"Failed to download YouTube video {media_url}: {e}. "
                    "Note: Direct YouTube downloads typically require 'use_uri_for_youtube=True' setting."
                )
            else:
                self.log.error(f"Failed to download media {media_url}: {e}")
            self._debug(f"Failed to download media {media_url}: {e}")
            return None