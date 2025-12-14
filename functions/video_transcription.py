"""
title: Video Transcription Pipeline
author: open-webui
date: 2024-12-07
version: 3.9
license: MIT
description: Extract audio from video files using ffmpeg and transcribe to text using Whisper. Generates SRT subtitle files and extracts embedded subtitles if available. Supports audio splitting for large files.
requirements: aiohttp, cryptography, pydantic, imageio-ffmpeg, openai-whisper

USAGE
- Upload a video file or provide a video URL in the chat
- The pipeline will:
  1) Extract the video file from __files__ parameter
  2) Extract audio track using ffmpeg
  3) Transcribe audio using Whisper (OpenAI API or local model)
  4) Return formatted transcription with optional timestamps

OUTPUT FORMAT
- Returns transcription text with optional timestamps
- Can export as plain text, SRT subtitles, or VTT format
- Saves transcription to Files DB for download
"""

from typing import Optional, Callable, Awaitable, Any, List, Dict
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
from pathlib import Path
from pydantic_core import core_schema
import subprocess
import shutil
import imageio_ffmpeg

# Open WebUI files + storage
from open_webui.models.files import Files as FilesDB, FileForm
from open_webui.storage.provider import Storage
from open_webui.routers.retrieval import ProcessFileForm, process_file


# Encrypted string helper
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
    """Wrapper for subprocess.run to use in thread pool"""
    return subprocess.run(cmd, capture_output=True)


class Pipe:
    class Valves(BaseModel):
        # Whisper Configuration
        WHISPER_MODE: str = Field(
            default="openai",
            description="Transcription mode: 'openai' (API), 'local' (whisper package), or 'openai-compatible' (custom endpoint)",
        )
        OPENAI_API_KEY: EncryptedStr = Field(
            default="",
            description="OpenAI API key for Whisper API (required for 'openai' mode)",
        )
        OPENAI_API_BASE_URL: str = Field(
            default="https://api.openai.com/v1",
            description="OpenAI API base URL (for 'openai' or 'openai-compatible' modes)",
        )
        WHISPER_MODEL: str = Field(
            default="whisper-1",
            description="Whisper model: 'whisper-1', 'gpt-4o-transcribe', 'gpt-4o-mini-transcribe', 'gpt-4o-transcribe-diarize' (OpenAI), or 'tiny/base/small/medium/large' (local)",
        )
        ENABLE_DIARIZATION: bool = Field(
            default=False,
            description="Enable speaker diarization (identifies who is speaking). Auto-uses gpt-4o-transcribe-diarize model. Note: Disables word-level timestamps.",
        )
        DIARIZATION_KNOWN_SPEAKERS: str = Field(
            default="",
            description="Comma-separated list of known speaker names for diarization (e.g., 'Alice,Bob,Charlie'). Leave empty for auto-detection.",
        )
        WHISPER_LANGUAGE: Optional[str] = Field(
            default=None,
            description="Language code (e.g., 'en', 'es', 'fr'). None = auto-detect",
        )
        WHISPER_TEMPERATURE: float = Field(
            default=0.0,
            description="Sampling temperature (0.0-1.0). Higher = more random",
        )
        INCLUDE_TIMESTAMPS: bool = Field(
            default=True,
            description="Include timestamps in transcription output",
        )
        OUTPUT_FORMAT: str = Field(
            default="text",
            description="Output format: 'text', 'srt', 'vtt', or 'json'",
        )
        MAX_AUDIO_SIZE_MB: int = Field(
            default=25,
            description="Maximum audio file size in MB (OpenAI API limit is 25MB)",
        )
        AUDIO_SAMPLE_RATE: int = Field(
            default=16000,
            description="Audio sample rate for extraction (Hz). 16000 recommended for Whisper",
        )
        TIMEOUT: int = Field(
            default=300,
            description="Maximum seconds to wait for transcription",
        )
        DEBUG: bool = Field(
            default=True,
            description="Enable verbose debug logging",
        )
        EMIT_INTERVAL: float = Field(
            default=0.5,
            description="Interval in seconds between status emissions",
        )
        ENABLE_STATUS_INDICATOR: bool = Field(
            default=True,
            description="Enable or disable status indicator emissions",
        )
        AUDIO_FORMAT: str = Field(
            default="mp3",
            description="Audio extraction format: 'mp3' (compressed) or 'wav' (uncompressed)",
        )
        MP3_BITRATE: str = Field(
            default="64k",
            description="MP3 bitrate for compression (e.g., '64k', '128k'). Lower = smaller file",
        )
        SPLIT_LARGE_FILES: bool = Field(
            default=True,
            description="Automatically split audio >25MB into chunks for processing",
        )
        GENERATE_SRT: bool = Field(
            default=True,
            description="Generate SRT subtitle file from transcription",
        )
        EXTRACT_EMBEDDED_SUBTITLES: bool = Field(
            default=True,
            description="Extract embedded subtitles from video if available",
        )
        SAVE_SUBTITLE_FILES: bool = Field(
            default=True,
            description="Save generated/extracted subtitles as files in Open WebUI",
        )
        MAX_SUBTITLE_DURATION: float = Field(
            default=3.0,
            description="Maximum duration in seconds for each subtitle segment (2-3s recommended for readability)",
        )
        MIN_SUBTITLE_WORDS: int = Field(
            default=2,
            description="Minimum words per subtitle (prevents single-word orphans like 'to')",
        )
        PAUSE_THRESHOLD: float = Field(
            default=0.3,
            description="Gap between words (seconds) to consider a natural pause/break point",
        )
        MAX_SUBTITLE_CHARS: int = Field(
            default=42,
            description="Maximum characters per subtitle line (standard is 42 for readability, max 84)",
        )
        SAVE_TO_FILE_METADATA: bool = Field(
            default=True,
            description="Save transcription to video file metadata for RAG searchability",
        )

    def __init__(self):
        self.name = "Video Transcription"
        self.valves = self.Valves()
        self.last_emit_time = 0
        self.log = logging.getLogger("video_transcription_pipeline")
        self.log.setLevel(logging.DEBUG if self.valves.DEBUG else logging.INFO)

    async def emit_status(
        self,
        event_emitter: Optional[Callable[[dict], Awaitable[None]]],
        level: str,
        message: str,
        done: bool = False,
    ) -> None:
        """Emit status updates to Open WebUI"""
        if not event_emitter or not self.valves.ENABLE_STATUS_INDICATOR:
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
        __files__: Optional[list] = None,
        __request__: Optional[Any] = None,
    ) -> str:
        self.log.debug(f"Video Transcription Pipeline - Processing request")

        await self.emit_status(
            __event_emitter__, "info", "Initializing Video Transcription pipeline..."
        )

        try:
            # Validate configuration
            if self.valves.WHISPER_MODE == "openai":
                openai_key = self.valves.OPENAI_API_KEY.get_decrypted()
                if not openai_key:
                    err = "OpenAI API key not configured. Please set OPENAI_API_KEY in pipeline settings."
                    self.log.error(err)
                    await self.emit_status(__event_emitter__, "error", err, True)
                    return f"❌ Error: {err}"

            # Extract video URL from messages
            messages = body.get("messages", [])
            if not messages or messages[-1].get("role") != "user":
                return "❌ Error: Please send a message with a video file attached."
            
            # Skip system-generated prompts (RAG, title, tags)
            last_msg = messages[-1]
            content = last_msg.get("content", "")
            if isinstance(content, str):
                # Detect system prompts by their characteristic phrases
                system_indicators = [
                    "### Task:",
                    "Generate a concise",
                    "Generate search queries",
                    "Analyze the chat history",
                    "### Guidelines:",
                    "### Output:",
                ]
                if any(indicator in content for indicator in system_indicators):
                    self.log.debug("Skipping system-generated prompt (RAG/title/tags)")
                    return ""  # Return empty to skip processing

            # Find video file from __files__ parameter
            video_url = None
            video_file_id = None  # Track original video file ID for metadata
            if __files__:
                self.log.info(f"Checking {len(__files__)} file(s) for video")
                for file_item in __files__:
                    file_id = file_item.get("id")
                    if file_id:
                        file_record = FilesDB.get_file_by_id(file_id)
                        if file_record and file_record.meta.get("content_type", "").startswith("video/"):
                            video_url = f"/api/v1/files/{file_id}/content"
                            video_file_id = file_id  # Save for metadata update
                            self.log.info(f"Found video file: {file_id}")
                            break
            
            # Fallback: Check if video URL is in messages (e.g., user pasted a URL)
            if not video_url:
                video_url = self._extract_video_url(messages)
            
            if not video_url:
                self.log.error("No video file found")
                return "❌ Error: No video file found. Please attach a video file or provide a video URL."

            self.log.debug(f"Found video URL: {video_url}")

            # Download video to temp file
            await self.emit_status(__event_emitter__, "info", "Downloading video...")
            tmp_video = await self._download_temp_file(video_url)
            
            try:
                # Get video info
                video_info = await self._get_video_info(tmp_video)
                duration = video_info.get("duration", 0)
                self.log.debug(f"Video duration: {duration}s")
                
                # Extract embedded subtitles if enabled
                embedded_subtitles = None
                if self.valves.EXTRACT_EMBEDDED_SUBTITLES:
                    await self.emit_status(__event_emitter__, "info", "Checking for embedded subtitles...")
                    embedded_subtitles = await self._extract_embedded_subtitles(tmp_video)
                
                # Extract audio from video
                await self.emit_status(__event_emitter__, "info", "Extracting audio from video...")
                tmp_audio = await self._extract_audio(tmp_video)
                chunk_paths = []
                
                try:
                    # Check audio file size and split if needed
                    audio_size_mb = os.path.getsize(tmp_audio) / (1024 * 1024)
                    self.log.debug(f"Audio file size: {audio_size_mb:.2f}MB")
                    
                    if audio_size_mb > self.valves.MAX_AUDIO_SIZE_MB:
                        if not self.valves.SPLIT_LARGE_FILES:
                            return f"❌ Error: Audio file too large ({audio_size_mb:.1f}MB). Maximum: {self.valves.MAX_AUDIO_SIZE_MB}MB. Enable SPLIT_LARGE_FILES to process."
                        
                        self.log.info(f"Audio exceeds {self.valves.MAX_AUDIO_SIZE_MB}MB, splitting into chunks...")
                        await self.emit_status(__event_emitter__, "info", f"Splitting large audio file ({audio_size_mb:.1f}MB)...")
                        chunk_paths = await self._split_audio(tmp_audio, max_size_mb=24)
                    else:
                        chunk_paths = [tmp_audio]
                    
                    # Transcribe each chunk
                    await self.emit_status(__event_emitter__, "info", f"Transcribing audio with Whisper ({len(chunk_paths)} chunk{'s' if len(chunk_paths) > 1 else ''})...")
                    
                    all_segments = []
                    time_offset = 0.0
                    
                    for i, chunk_path in enumerate(chunk_paths):
                        if len(chunk_paths) > 1:
                            self.log.info(f"Transcribing chunk {i+1}/{len(chunk_paths)}")
                            await self.emit_status(__event_emitter__, "info", f"Transcribing chunk {i+1}/{len(chunk_paths)}...")
                        
                        chunk_result = await self._transcribe_audio(chunk_path)
                        
                        if not chunk_result:
                            return f"❌ Error: Transcription failed for chunk {i+1}"
                        
                        # Map top-level words to segments for OpenAI API responses
                        if "words" in chunk_result and "segments" in chunk_result:
                            chunk_result = self._map_words_to_segments(chunk_result)
                        
                        # Adjust timestamps for chunks after the first
                        if "segments" in chunk_result and chunk_result["segments"]:
                            for seg in chunk_result["segments"]:
                                seg["start"] += time_offset
                                seg["end"] += time_offset
                                # Also adjust word timestamps if present
                                if "words" in seg:
                                    for word in seg["words"]:
                                        word["start"] += time_offset
                                        word["end"] += time_offset
                                all_segments.append(seg)
                            
                            # Update offset for next chunk
                            time_offset = chunk_result["segments"][-1]["end"]
                    
                    # Combine all segments into single transcription
                    if all_segments:
                        transcription = {
                            "text": " ".join(seg.get("text", "") for seg in all_segments),
                            "segments": all_segments
                        }
                    else:
                        # Fallback for non-verbose format
                        transcription = chunk_result
                    
                    if not transcription:
                        return "❌ Error: Transcription failed. Please try again."
                    
                    # Generate SRT file from transcription if enabled
                    srt_content = None
                    srt_file_id = None
                    if self.valves.GENERATE_SRT and "segments" in transcription:
                        await self.emit_status(__event_emitter__, "info", "Generating SRT subtitle file...")
                        # Check if diarization was used (segments have speaker labels)
                        has_speakers = self.valves.ENABLE_DIARIZATION and any("speaker" in seg for seg in transcription["segments"])
                        srt_content = self._generate_srt(transcription["segments"], include_speakers=has_speakers)
                        
                        # Save SRT file if enabled
                        if self.valves.SAVE_SUBTITLE_FILES and __user__:
                            srt_file_id = await self._save_subtitle_file(
                                srt_content,
                                "video",  # Will be timestamped
                                "srt",
                                __user__.get("id"),
                                None  # No parent file ID for pipe
                            )
                    
                    # Save embedded subtitles as files if found
                    embedded_file_ids = {}
                    if embedded_subtitles and self.valves.SAVE_SUBTITLE_FILES and __user__:
                        for lang, content in embedded_subtitles.items():
                            sub_file_id = await self._save_subtitle_file(
                                content,
                                "video",
                                lang,
                                __user__.get("id"),
                                None
                            )
                            if sub_file_id:
                                embedded_file_ids[lang] = sub_file_id
                    
                    # Format output
                    await self.emit_status(__event_emitter__, "info", "Formatting transcription...")
                    formatted_output = self._format_transcription(transcription, duration)
                    
                    # Save transcription to video file metadata for RAG searchability
                    if self.valves.SAVE_TO_FILE_METADATA and video_file_id:
                        try:
                            # Get plain text (no timestamps) for RAG
                            plain_text = transcription.get("text", "")
                            if not plain_text and "segments" in transcription:
                                plain_text = " ".join(seg.get("text", "") for seg in transcription["segments"])
                            
                            # Update video file metadata
                            file_record = FilesDB.get_file_by_id(video_file_id)
                            if file_record:
                                current_data = file_record.data or {}
                                current_data["transcription"] = plain_text
                                FilesDB.update_file_data_by_id(video_file_id, {"data": current_data})
                                self.log.info(f"Saved transcription to video file metadata")
                                
                                # Process for RAG if we have request object
                                if __request__ and __user__:
                                    try:
                                        # Create user object for process_file
                                        from open_webui.models.users import UserModel
                                        user = UserModel(**__user__)
                                        
                                        # Process the transcription text for RAG indexing
                                        process_file(
                                            __request__,
                                            ProcessFileForm(file_id=video_file_id, content=plain_text),
                                            user=user
                                        )
                                        self.log.info(f"Processed video transcription for RAG indexing")
                                    except Exception as rag_error:
                                        self.log.warning(f"Failed to process for RAG: {rag_error}")
                        except Exception as e:
                            self.log.warning(f"Failed to save transcription to metadata: {e}")
                    
                    # Save transcription to file
                    user_id = __user__.get("id") if __user__ else None
                    file_id = await self._save_transcription_file(
                        formatted_output,
                        transcription,
                        user_id,
                    )
                    
                    # Emit file attachments for subtitle files as direct messages
                    if __event_emitter__:
                        # Emit generated SRT file
                        if srt_file_id:
                            await __event_emitter__({
                                "type": "message",
                                "data": {
                                    "content": f"[📥 Download Generated SRT Subtitles](/api/v1/files/{srt_file_id}/content)\n\n"
                                              f"*Generated from Whisper transcription with {len(srt_content.strip().split(chr(10)+chr(10)))} subtitle entries*"
                                }
                            })
                        
                        # Emit embedded subtitle files
                        if embedded_file_ids:
                            for lang, sub_file_id in embedded_file_ids.items():
                                await __event_emitter__({
                                    "type": "message",
                                    "data": {
                                        "content": f"[📥 Download Embedded Subtitles ({lang.upper()})](/api/v1/files/{sub_file_id}/content)"
                                    }
                                })
                        
                        # Emit legacy transcription file download
                        if file_id:
                            await __event_emitter__({
                                "type": "message",
                                "data": {
                                    "content": f"[📥 Download Transcription File ({self.valves.OUTPUT_FORMAT.upper()})](/api/v1/files/{file_id}/content)"
                                }
                            })
                    
                    await self.emit_status(
                        __event_emitter__, "info", "Transcription complete!", True
                    )
                    
                    # Build comprehensive response
                    response_parts = []
                    response_parts.append("# Video Transcription Complete\n")
                    response_parts.append(f"**Duration:** {self._format_duration(duration)}\n" if duration > 0 else "")
                    
                    # Embedded subtitles section
                    if embedded_subtitles:
                        response_parts.append("\n## 📝 Embedded Subtitles Found")
                        response_parts.append(f"Found **{len(embedded_subtitles)}** embedded subtitle track(s): {', '.join(embedded_subtitles.keys())}\n")
                        for lang, content in embedded_subtitles.items():
                            line_count = len(content.strip().split('\n\n'))
                            response_parts.append(f"- **{lang}**: {line_count} subtitle entries")
                            if lang in embedded_file_ids:
                                response_parts.append(f" ([Download {lang}.srt](/api/v1/files/{embedded_file_ids[lang]}/content))")
                        response_parts.append("")
                    else:
                        response_parts.append("\n## 📝 No Embedded Subtitles")
                        response_parts.append("No embedded subtitles found in video.\n")
                    
                    # Generated SRT section
                    if srt_content:
                        subtitle_count = len(srt_content.strip().split('\n\n'))
                        response_parts.append(f"\n## 🎬 Generated SRT Subtitles")
                        response_parts.append(f"Created SRT file with **{subtitle_count}** subtitle entries from Whisper transcription.")
                        if srt_file_id:
                            response_parts.append(f" [Download SRT file](/api/v1/files/{srt_file_id}/content)")
                        response_parts.append("")
                    
                    # Legacy transcription file download
                    if file_id:
                        response_parts.append(f"\n[Download Transcription File ({self.valves.OUTPUT_FORMAT.upper()})](/api/v1/files/{file_id}/content)\n")
                    
                    # Transcription text
                    response_parts.append("\n## 📄 Transcription Text\n")
                    response_parts.append(formatted_output)
                    
                    return "\n".join(response_parts)
                    
                finally:
                    # Cleanup audio temp files (main file + any chunks)
                    if os.path.exists(tmp_audio):
                        os.unlink(tmp_audio)
                    
                    for chunk_path in chunk_paths:
                        if chunk_path != tmp_audio and os.path.exists(chunk_path):
                            os.unlink(chunk_path)
                        
            finally:
                # Cleanup video temp file
                if os.path.exists(tmp_video):
                    os.unlink(tmp_video)

        except Exception as e:
            self.log.exception(f"Video transcription pipeline error: {e}")
            error_msg = f"Pipeline error: {str(e)}"
            await self.emit_status(__event_emitter__, "error", error_msg, True)
            return f"❌ Error: {str(e)}"

    def _extract_video_url(self, messages: List[dict]) -> Optional[str]:
        """Extract video URL from message history using multiple strategies.
        
        Extraction order (prioritized for reliability):
        1. 'images' array - File IDs or URLs (Open WebUI pattern)
        2. 'files' array - Uploaded files with metadata
        3. 'content' string - Regex extraction:
           - Markdown links: [text](url) or ![alt](url)
           - HTML tags: <video src> or <source src>
           - File API URLs: /api/v1/files/{id}/content
           - Absolute video URLs: https://example.com/video.mp4
           - Relative video URLs: /videos/clip.mp4
        4. 'content' array - Multimodal message items:
           - video_url type
           - file/input_video types with URL + mime
           - Generic URLs
        
        Returns the first video URL found, or None if no video detected.
        """
        video_exts = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".flv", ".wmv")

        def scan_message(msg: dict) -> Optional[str]:
            # Check for 'images' array first (used by some Open WebUI components)
            # Even though we want videos, files might be in 'images' array
            if "images" in msg:
                images = msg.get("images", [])
                self.log.debug(f"Found 'images' array with {len(images)} items")
                if images and len(images) > 0:
                    first_item = images[0]
                    if isinstance(first_item, str):
                        # Could be a file ID or URL
                        if first_item.startswith('/api/v1/files/'):
                            self.log.debug(f"Found file URL in images array: {first_item}")
                            return first_item
                        elif '/' not in first_item:
                            # Looks like a file ID, construct the path
                            url = f"/api/v1/files/{first_item}/content"
                            self.log.debug(f"Constructed URL from file ID in images array: {url}")
                            return url
            
            # Check for 'files' array (Open WebUI common pattern for file uploads)
            files = msg.get("files", [])
            if files:
                self.log.debug(f"Found 'files' array with {len(files)} items")
                for f in files:
                    if isinstance(f, dict):
                        self.log.debug(f"File item keys: {list(f.keys())}")
                        # Check file type
                        file_type = f.get("type") or f.get("file", {}).get("type") if isinstance(f.get("file"), dict) else None
                        content_type = f.get("content_type") or f.get("file", {}).get("content_type") if isinstance(f.get("file"), dict) else None
                        
                        self.log.debug(f"File type: {file_type}, content_type: {content_type}")
                        
                        # Check if it's a video
                        if (file_type and file_type == "file") or (content_type and content_type.startswith("video/")):
                            # Get file ID
                            file_id = f.get("id") or (f.get("file", {}).get("id") if isinstance(f.get("file"), dict) else None)
                            if file_id:
                                url = f"/api/v1/files/{file_id}/content"
                                self.log.debug(f"Found video file ID: {file_id} -> {url}")
                                return url
            
            content = msg.get("content", "")
            candidates: List[str] = []

            # String content - use comprehensive regex extraction
            if isinstance(content, str):
                # 1) Markdown links/media: [text](url) or ![alt](url)
                md_pattern = r"(?:!\[.*?\]|\[.*?\])\((data:video/[^)]+|https?://[^)]+|/[^)]+)\)"
                candidates.extend(re.findall(md_pattern, content, flags=re.IGNORECASE))

                # 2) HTML <video src> or <source src>
                html_pattern = r"<(?:video|source)[^>]+src=\"([^\"]+)\"[^>]*>"
                candidates.extend(re.findall(html_pattern, content, flags=re.IGNORECASE))

                # 3) Bare file API URLs: /api/v1/files/{id}/content (with or without /content)
                file_api_pattern = r"(?:https?://[^\s)]+)?(/api/v1/files/[a-f0-9\-]+)(?:/content)?"
                file_api_matches = re.findall(file_api_pattern, content, flags=re.IGNORECASE)
                for m in file_api_matches:
                    base = m if m.startswith("/") else f'/{m.lstrip("/")}'
                    # Ensure /content suffix
                    if not base.endswith("/content"):
                        base = base + "/content"
                    candidates.append(base)

                # 4) Bare absolute video URLs with common extensions
                abs_video_pattern = r"https?://[^\s)]+\.(?:mp4|webm|mov|m4v|avi|mkv|flv|wmv)(?:\?[^\s)]*)?"
                candidates.extend(re.findall(abs_video_pattern, content, flags=re.IGNORECASE))

                # 5) Bare site-relative video URLs with common extensions
                rel_video_pattern = r"/[^\s)]+\.(?:mp4|webm|mov|m4v|avi|mkv|flv|wmv)(?:\?[^\s)]*)?"
                candidates.extend(re.findall(rel_video_pattern, content, flags=re.IGNORECASE))

                # 6) Generic http(s) URLs (fallback)
                candidates.extend(re.findall(r"https?://[^\s]+", content))

            # Array content (multimodal)
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    
                    # Video type items
                    t = item.get("type")
                    if t in ("video", "video_url"):
                        if "video_url" in item and isinstance(item["video_url"], dict):
                            u = item["video_url"].get("url")
                            if u:
                                candidates.append(u)
                        u = item.get("url") or item.get("content")
                        if u:
                            candidates.append(u)
                    
                    # File objects
                    if isinstance(item.get("file"), dict):
                        fobj = item["file"]
                        u = fobj.get("url")
                        mime = fobj.get("mime") or fobj.get("mime_type")
                        fid = fobj.get("id") or fobj.get("file_id")
                        
                        if isinstance(fid, str) and fid:
                            candidates.append(f"/api/v1/files/{fid}/content")
                        if isinstance(u, str) and u:
                            candidates.append(u)
                    
                    # Generic URL
                    u = item.get("url")
                    if isinstance(u, str) and u:
                        candidates.append(u)

            # Dict content
            elif isinstance(content, dict):
                u = content.get("url")
                if isinstance(u, str) and u:
                    candidates.append(u)
                
                if isinstance(content.get("file"), dict):
                    fobj = content["file"]
                    fid = fobj.get("id") or fobj.get("file_id")
                    if isinstance(fid, str) and fid:
                        candidates.append(f"/api/v1/files/{fid}/content")
                    u2 = fobj.get("url")
                    if isinstance(u2, str) and u2:
                        candidates.append(u2)

            # Deduplicate candidates while preserving order
            seen = set()
            deduped: List[str] = []
            for u in candidates:
                if u and u not in seen:
                    seen.add(u)
                    deduped.append(u)
            
            self.log.debug(f"Found {len(deduped)} candidate URL(s) in content: {deduped[:3]}...")
            
            # Check candidates (prioritize file API URLs, then video extensions)
            for u in deduped:
                if not isinstance(u, str):
                    continue
                
                # Skip data URLs (too large)
                if u.startswith("data:"):
                    continue
                    
                low = u.lower()
                
                # Priority 1: Open WebUI file API (most reliable for uploaded files)
                m_api = re.search(
                    r"(/api/v1/files/[a-f0-9\-]+)(/content)?", low, flags=re.IGNORECASE
                )
                if m_api:
                    base = m_api.group(1)
                    url = base + "/content"
                    self.log.debug(f"Selected file API URL: {url}")
                    return url
                
                # Priority 2: Direct video URLs by extension
                if any(low.endswith(ext) for ext in video_exts):
                    self.log.debug(f"Selected video URL by extension: {u}")
                    return u
            
            return None

        # Search user messages first
        for msg in reversed(messages or []):
            if msg.get("role") != "user":
                continue
            found = scan_message(msg)
            if found:
                return found

        # Fallback to assistant messages for file references
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
                    return link if link.startswith("/") else f'/{link.lstrip("/")}'
        
        return None

    async def _download_temp_file(self, url: str) -> str:
        """Download or resolve URL to temporary local file"""
        # Try internal file resolution first
        bare = re.search(r"^(/api/v1/files/[a-f0-9\-]+)$", url, flags=re.IGNORECASE)
        if bare:
            url = bare.group(1) + "/content"
        
        local_path = self._try_resolve_internal_file(url)
        if local_path:
            suffix = Path(local_path).suffix or ""
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp_path = tmp.name
            shutil.copyfile(local_path, tmp_path)
            self.log.debug(f"Resolved internal file: {local_path}")
            return tmp_path

        # HTTP download
        suffix = Path(url).suffix or ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
        
        timeout = aiohttp.ClientTimeout(total=self.valves.TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            self.log.debug(f"Downloading: {url}")
            async with session.get(url) as resp:
                if resp.status != 200:
                    txt = (await resp.text())[:500]
                    raise RuntimeError(f"Failed to download video ({resp.status}): {txt}")
                data = await resp.read()
                with open(tmp_path, "wb") as f:
                    f.write(data)
        
        self.log.debug(f"Downloaded to: {tmp_path}")
        return tmp_path

    def _try_resolve_internal_file(self, url: str) -> Optional[str]:
        """Resolve Open WebUI file URL to local path"""
        try:
            m = re.search(r"/api/v1/files/([\w-]+)/content", url)
            if not m:
                return None
            file_id = m.group(1)
            file_rec = FilesDB.get_file_by_id(file_id)
            if not file_rec or not file_rec.path:
                return None
            return Storage.get_file(file_rec.path)
        except Exception:
            return None

    async def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video information using ffmpeg"""
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [ffmpeg_exe, "-i", video_path, "-f", "null", "-"]
            
            completed = await asyncio.to_thread(_subprocess_run_wrapper, cmd)
            stderr = completed.stderr.decode(errors='ignore') if completed.stderr else ''
            
            info = {}
            
            # Parse duration
            duration_match = re.search(r'Duration:\s*(\d+):(\d+):(\d+\.\d+)', stderr)
            if duration_match:
                hours = int(duration_match.group(1))
                minutes = int(duration_match.group(2))
                seconds = float(duration_match.group(3))
                info["duration"] = hours * 3600 + minutes * 60 + seconds
            
            return info
        except Exception as e:
            self.log.warning(f"Failed to get video info: {e}")
            return {}

    async def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video using ffmpeg"""
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            
            # Choose format based on valve setting
            if self.valves.AUDIO_FORMAT == "mp3":
                suffix = ".mp3"
                codec_params = [
                    "-acodec", "libmp3lame",
                    "-b:a", self.valves.MP3_BITRATE,
                ]
            else:
                suffix = ".wav"
                codec_params = [
                    "-acodec", "pcm_s16le",
                ]
            
            tmp_audio = tempfile.mktemp(suffix=suffix)
            
            cmd = [
                ffmpeg_exe,
                "-i", video_path,
                "-vn",  # No video
                *codec_params,
                "-ar", str(self.valves.AUDIO_SAMPLE_RATE),
                "-ac", "1",  # Mono
                tmp_audio
            ]
            
            self.log.debug(f"Extracting audio: {' '.join(cmd)}")
            completed = await asyncio.to_thread(_subprocess_run_wrapper, cmd)
            
            if completed.returncode != 0:
                stderr = completed.stderr.decode(errors='ignore') if completed.stderr else ''
                raise Exception(f"FFmpeg failed: {stderr}")
            
            if not os.path.exists(tmp_audio):
                raise Exception("Audio extraction failed")
            
            # Log audio file info
            audio_size_mb = os.path.getsize(tmp_audio) / (1024 * 1024)
            self.log.info(f"Audio extracted successfully: {audio_size_mb:.2f}MB ({self.valves.AUDIO_FORMAT})")
            
            return tmp_audio
            
        except Exception as e:
            self.log.error(f"Audio extraction failed: {e}")
            raise

    async def _transcribe_audio(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """Transcribe audio using Whisper"""
        if self.valves.WHISPER_MODE == "openai":
            return await self._transcribe_openai(audio_path)
        elif self.valves.WHISPER_MODE == "local":
            return await self._transcribe_local(audio_path)
        elif self.valves.WHISPER_MODE == "openai-compatible":
            return await self._transcribe_openai_compatible(audio_path)
        else:
            raise ValueError(f"Invalid WHISPER_MODE: {self.valves.WHISPER_MODE}")

    async def _transcribe_openai(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """Transcribe using OpenAI Whisper API"""
        try:
            api_key = self.valves.OPENAI_API_KEY.get_decrypted()
            base_url = self.valves.OPENAI_API_BASE_URL.rstrip("/")
            url = f"{base_url}/audio/transcriptions"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
            }
            
            # Prepare form data
            with open(audio_path, "rb") as f:
                audio_data = f.read()
            
            # Build form data
            data = aiohttp.FormData()
            data.add_field("file", audio_data, filename="audio.wav", content_type="audio/wav")
            
            # Determine model based on diarization setting
            if self.valves.ENABLE_DIARIZATION:
                model = "gpt-4o-transcribe-diarize"
                self.log.info("Diarization enabled - using gpt-4o-transcribe-diarize model")
            else:
                model = self.valves.WHISPER_MODEL
            
            data.add_field("model", model)
            
            # Handle diarization vs regular transcription
            if self.valves.ENABLE_DIARIZATION:
                # Diarization requires specific format and chunking
                data.add_field("response_format", "diarized_json")
                data.add_field("chunking_strategy", "auto")
                
                # Add known speakers if provided
                if self.valves.DIARIZATION_KNOWN_SPEAKERS.strip():
                    speakers = [s.strip() for s in self.valves.DIARIZATION_KNOWN_SPEAKERS.split(",") if s.strip()]
                    for speaker in speakers:
                        data.add_field("known_speaker_names[]", speaker)
                    self.log.info(f"Using known speakers: {speakers}")
            elif self.valves.INCLUDE_TIMESTAMPS:
                data.add_field("response_format", "verbose_json")
                # Request word-level timestamps for better subtitle granularity (not supported with diarization)
                data.add_field("timestamp_granularities[]", "word")
                data.add_field("timestamp_granularities[]", "segment")
            else:
                data.add_field("response_format", "json")
            
            if self.valves.WHISPER_LANGUAGE:
                data.add_field("language", self.valves.WHISPER_LANGUAGE)
            
            if not self.valves.ENABLE_DIARIZATION:
                data.add_field("temperature", str(self.valves.WHISPER_TEMPERATURE))
            
            timeout = aiohttp.ClientTimeout(total=self.valves.TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, data=data) as resp:
                    if resp.status != 200:
                        txt = (await resp.text())[:500]
                        self.log.error(f"OpenAI Whisper API error {resp.status}: {txt}")
                        return None
                    
                    result = await resp.json()
                    self.log.debug(f"Transcription result: {result}")
                    return result
        
        except Exception as e:
            self.log.error(f"OpenAI transcription failed: {e}")
            return None

    async def _transcribe_local(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """Transcribe using local Whisper model"""
        try:
            import whisper
            
            model_name = self.valves.WHISPER_MODEL
            self.log.debug(f"Loading Whisper model: {model_name}")
            
            # Load model in thread pool to avoid blocking
            model = await asyncio.to_thread(whisper.load_model, model_name)
            
            # Transcribe with word-level timestamps
            options = {
                "language": self.valves.WHISPER_LANGUAGE,
                "temperature": self.valves.WHISPER_TEMPERATURE,
                "word_timestamps": True,  # Enable word-level timestamps for better subtitles
            }
            
            result = await asyncio.to_thread(model.transcribe, audio_path, **options)
            
            return result
        
        except ImportError:
            self.log.error("Whisper package not installed. Run: pip install openai-whisper")
            return None
        except Exception as e:
            self.log.error(f"Local transcription failed: {e}")
            return None

    async def _transcribe_openai_compatible(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """Transcribe using OpenAI-compatible API endpoint"""
        # Same as OpenAI but with custom base URL
        return await self._transcribe_openai(audio_path)

    def _format_transcription(self, transcription: Dict[str, Any], duration: float = 0) -> str:
        """Format transcription based on output format"""
        if self.valves.OUTPUT_FORMAT == "json":
            return json.dumps(transcription, indent=2)
        
        elif self.valves.OUTPUT_FORMAT == "srt":
            return self._format_srt(transcription)
        
        elif self.valves.OUTPUT_FORMAT == "vtt":
            return self._format_vtt(transcription)
        
        else:  # text
            return self._format_text(transcription)

    def _format_text(self, transcription: Dict[str, Any]) -> str:
        """Format as plain text with optional timestamps"""
        if not transcription:
            return ""
        
        # Handle different response formats
        if "text" in transcription:
            text = transcription["text"]
        else:
            text = transcription.get("result", "")
        
        if not self.valves.INCLUDE_TIMESTAMPS or "segments" not in transcription:
            return text.strip()
        
        # Format with timestamps
        lines = []
        for segment in transcription.get("segments", []):
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            segment_text = segment.get("text", "").strip()
            
            timestamp = f"[{self._format_timestamp(start)} → {self._format_timestamp(end)}]"
            lines.append(f"{timestamp} {segment_text}")
        
        return "\n".join(lines)

    def _format_srt(self, transcription: Dict[str, Any]) -> str:
        """Format as SRT subtitles"""
        if not transcription or "segments" not in transcription:
            return transcription.get("text", "")
        
        lines = []
        for i, segment in enumerate(transcription.get("segments", []), 1):
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            text = segment.get("text", "").strip()
            
            start_time = self._format_srt_timestamp(start)
            end_time = self._format_srt_timestamp(end)
            
            lines.append(f"{i}")
            lines.append(f"{start_time} --> {end_time}")
            lines.append(text)
            lines.append("")
        
        return "\n".join(lines)

    def _format_vtt(self, transcription: Dict[str, Any]) -> str:
        """Format as WebVTT subtitles"""
        if not transcription or "segments" not in transcription:
            return f"WEBVTT\n\n{transcription.get('text', '')}"
        
        lines = ["WEBVTT", ""]
        
        for segment in transcription.get("segments", []):
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            text = segment.get("text", "").strip()
            
            start_time = self._format_vtt_timestamp(start)
            end_time = self._format_vtt_timestamp(end)
            
            lines.append(f"{start_time} --> {end_time}")
            lines.append(text)
            lines.append("")
        
        return "\n".join(lines)

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as MM:SS"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    def _format_duration(self, seconds: float) -> str:
        """Format duration as human-readable string"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    def _format_srt_timestamp(self, seconds: float) -> str:
        """Format timestamp for SRT format: HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _map_words_to_segments(self, transcription: Dict[str, Any]) -> Dict[str, Any]:
        """Map top-level words array from OpenAI API into segments
        
        OpenAI's timestamp_granularities response format has words at the top level,
        not nested inside segments. This method distributes them correctly.
        
        Args:
            transcription: Response with top-level 'words' and 'segments' arrays
            
        Returns:
            Modified transcription with words nested inside their segments
        """
        words = transcription.get("words", [])
        segments = transcription.get("segments", [])
        
        if not words or not segments:
            return transcription
        
        self.log.debug(f"Mapping {len(words)} words to {len(segments)} segments")
        
        # Create a new segments list with words embedded
        for segment in segments:
            segment_start = segment.get("start", 0)
            segment_end = segment.get("end", 0)
            
            # Find all words that belong to this segment
            segment_words = []
            for word in words:
                word_start = word.get("start", 0)
                # Word belongs to segment if it starts within the segment's time range
                if segment_start <= word_start < segment_end:
                    segment_words.append(word)
            
            # Add words to segment
            segment["words"] = segment_words
            self.log.debug(f"Segment {segment_start:.2f}-{segment_end:.2f}: {len(segment_words)} words")
        
        return transcription
    
    def _split_into_subtitle_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split long Whisper segments into shorter subtitle-appropriate segments
        
        Args:
            segments: Original Whisper segments with word-level timestamps
            
        Returns:
            List of shorter segments suitable for subtitles
        """
        subtitle_segments = []
        
        for segment in segments:
            words = segment.get("words", [])
            
            # If no word-level timestamps, use original segment
            if not words:
                subtitle_segments.append(segment)
                continue
            
            # Build shorter segments
            current_segment = {
                "start": words[0].get("start", segment.get("start", 0)),
                "text": "",
                "words": []
            }
            
            for i, word_info in enumerate(words):
                word = word_info.get("word", "")
                word_start = word_info.get("start", 0)
                word_end = word_info.get("end", 0)
                
                # Add space before word if not first word in segment
                word_with_space = word if not current_segment["text"] else " " + word
                
                # Check if adding this word would exceed limits
                potential_text = current_segment["text"] + word_with_space
                duration = word_end - current_segment["start"]
                word_count = len(current_segment["words"])
                
                # Detect natural pause: check time gap to NEXT word
                has_pause_after = False
                if i < len(words) - 1:
                    next_word_start = words[i + 1].get("start", 0)
                    gap = next_word_start - word_end
                    has_pause_after = gap > self.valves.PAUSE_THRESHOLD
                
                # Check if current word is a conjunction/preposition (shouldn't end on these)
                word_lower = word.strip().lower()
                is_connector = word_lower in ['to', 'and', 'or', 'but', 'so', 'the', 'a', 'an', 'of', 'in', 'on', 'at', 'for']
                
                # Hard limits that force a split
                exceeds_duration = duration > self.valves.MAX_SUBTITLE_DURATION
                exceeds_chars = len(potential_text.strip()) > self.valves.MAX_SUBTITLE_CHARS
                
                # Should we split after adding this word?
                should_split = False
                
                # Don't split if we don't have minimum words yet (unless hard limit exceeded)
                if word_count + 1 < self.valves.MIN_SUBTITLE_WORDS:
                    should_split = exceeds_duration or exceeds_chars
                else:
                    # We have enough words, look for good break points
                    if exceeds_duration or exceeds_chars:
                        # Hard limit exceeded, must split (but not on a connector word if possible)
                        should_split = not is_connector or word_count + 1 >= self.valves.MIN_SUBTITLE_WORDS + 2
                    elif has_pause_after and not is_connector:
                        # Natural pause detected and not ending on a connector word
                        should_split = True
                
                # Add word to current segment
                current_segment["text"] += word_with_space
                current_segment["words"].append(word_info)
                
                # Split if needed
                if should_split and current_segment["text"].strip():
                    # Save current segment
                    current_segment["end"] = word_end
                    current_segment["text"] = current_segment["text"].strip()
                    subtitle_segments.append(current_segment)
                    
                    # Start new segment for next iteration
                    if i < len(words) - 1:  # Only if not last word
                        next_word_start = words[i + 1].get("start", 0)
                        current_segment = {
                            "start": next_word_start,
                            "text": "",
                            "words": []
                        }
            
            # Add final segment
            if current_segment["text"].strip():
                if current_segment["words"]:
                    current_segment["end"] = current_segment["words"][-1].get("end", segment.get("end", 0))
                else:
                    current_segment["end"] = segment.get("end", 0)
                current_segment["text"] = current_segment["text"].strip()
                subtitle_segments.append(current_segment)
        
        return subtitle_segments
    
    def _generate_srt(self, segments: List[Dict[str, Any]], include_speakers: bool = False) -> str:
        """Generate SRT subtitle file content from Whisper segments
        
        Args:
            segments: Transcription segments (may include speaker labels for diarization)
            include_speakers: If True, prefix each subtitle with [Speaker Name]
        """
        # Split into subtitle-appropriate segments (only if not diarization)
        if not include_speakers:
            subtitle_segments = self._split_into_subtitle_segments(segments)
        else:
            # Diarization segments are already properly split
            subtitle_segments = segments
        
        srt_lines = []
        
        for i, seg in enumerate(subtitle_segments, 1):
            start_time = self._format_srt_timestamp(seg.get("start", 0))
            end_time = self._format_srt_timestamp(seg.get("end", 0))
            text = seg.get("text", "").strip()
            
            # Add speaker label if available
            if include_speakers and "speaker" in seg:
                speaker = seg["speaker"]
                text = f"[{speaker}]: {text}"
            
            # SRT format:
            # 1
            # 00:00:00,000 --> 00:00:02,500
            # Subtitle text here
            # (blank line)
            srt_lines.append(str(i))
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(text)
            srt_lines.append("")  # Blank line between entries
        
        return "\n".join(srt_lines)

    def _format_vtt_timestamp(self, seconds: float) -> str:
        """Format timestamp for VTT format: HH:MM:SS.mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    async def _save_transcription_file(
        self,
        formatted_output: str,
        transcription: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """Save transcription to Files DB"""
        try:
            # Determine file extension
            ext_map = {
                "text": ".txt",
                "srt": ".srt",
                "vtt": ".vtt",
                "json": ".json",
            }
            ext = ext_map.get(self.valves.OUTPUT_FORMAT, ".txt")
            
            timestamp = int(time.time())
            filename = f"transcription_{timestamp}{ext}"
            
            # Write to temp file
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=ext, encoding="utf-8") as tmp:
                tmp.write(formatted_output)
                tmp_path = tmp.name
            
            try:
                # Upload via Storage
                with open(tmp_path, "rb") as f:
                    file_data, file_path = Storage.upload_file(
                        f,
                        filename,
                        {"content_type": "text/plain", "source": "video_transcription"},
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
                            "content_type": "text/plain",
                            "size": len(formatted_output.encode("utf-8")),
                            "source": "video_transcription",
                            "output_format": self.valves.OUTPUT_FORMAT,
                            "language": transcription.get("language", "unknown"),
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
            self.log.error(f"Failed to save transcription file: {e}")
            return None
    
    async def _get_audio_duration(self, audio_path: str) -> Optional[float]:
        """Get audio duration in seconds using ffmpeg"""
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [ffmpeg_exe, "-i", audio_path, "-f", "null", "-"]
            
            completed = await asyncio.to_thread(_subprocess_run_wrapper, cmd)
            stderr = completed.stderr.decode(errors='ignore') if completed.stderr else ''
            
            # Parse duration from stderr: "Duration: 00:05:30.50"
            match = re.search(r'Duration:\s*(\d+):(\d+):(\d+\.\d+)', stderr)
            if match:
                hours, minutes, seconds = float(match.group(1)), float(match.group(2)), float(match.group(3))
                duration = hours * 3600 + minutes * 60 + seconds
                return duration
            return None
        except Exception as e:
            self.log.warning(f"Failed to get audio duration: {e}")
            return None
    
    async def _split_audio(self, audio_path: str, max_size_mb: float = 24) -> List[str]:
        """Split audio file into chunks under max_size_mb"""
        try:
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            
            if file_size_mb <= max_size_mb:
                return [audio_path]  # No split needed
            
            # Get total duration
            duration = await self._get_audio_duration(audio_path)
            if not duration:
                raise Exception("Cannot split audio: duration unknown")
            
            # Calculate chunk duration to stay under size limit
            num_chunks = int(file_size_mb / max_size_mb) + 1
            chunk_duration = duration / num_chunks
            
            self.log.info(f"Splitting {file_size_mb:.1f}MB audio into {num_chunks} chunks of ~{chunk_duration:.1f}s each")
            
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            suffix = os.path.splitext(audio_path)[1]
            chunk_paths = []
            
            for i in range(num_chunks):
                start_time = i * chunk_duration
                chunk_path = tempfile.mktemp(suffix=f"_chunk{i}{suffix}")
                
                cmd = [
                    ffmpeg_exe,
                    "-i", audio_path,
                    "-ss", str(start_time),
                    "-t", str(chunk_duration),
                    "-c", "copy",  # Copy codec (no re-encode)
                    chunk_path
                ]
                
                completed = await asyncio.to_thread(_subprocess_run_wrapper, cmd)
                if completed.returncode == 0 and os.path.exists(chunk_path):
                    chunk_size = os.path.getsize(chunk_path) / (1024 * 1024)
                    self.log.info(f"Created chunk {i+1}/{num_chunks}: {chunk_size:.1f}MB")
                    chunk_paths.append(chunk_path)
                else:
                    raise Exception(f"Failed to create chunk {i}")
            
            return chunk_paths
            
        except Exception as e:
            self.log.error(f"Audio splitting failed: {e}")
            raise
    
    async def _extract_embedded_subtitles(self, video_path: str) -> Optional[Dict[str, str]]:
        """Extract embedded subtitles from video file using ffmpeg
        
        Returns:
            Dictionary mapping language codes to SRT content, or None if no subtitles found
        """
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            
            # First, probe video to find subtitle streams
            cmd = [ffmpeg_exe, "-i", video_path]
            completed = await asyncio.to_thread(_subprocess_run_wrapper, cmd)
            stderr = completed.stderr.decode(errors='ignore') if completed.stderr else ''
            
            # Parse subtitle streams from ffmpeg output
            # Format: "Stream #0:2(eng): Subtitle: subrip"
            subtitle_streams = []
            for line in stderr.split('\n'):
                if 'Subtitle:' in line and 'Stream' in line:
                    # Extract stream index and language
                    match = re.search(r'Stream #(\d+):(\d+)(?:\((\w+)\))?.*Subtitle:\s*(\w+)', line)
                    if match:
                        stream_idx = f"{match.group(1)}:{match.group(2)}"
                        lang = match.group(3) or "unknown"
                        codec = match.group(4)
                        subtitle_streams.append((stream_idx, lang, codec))
                        self.log.info(f"Found subtitle stream: {stream_idx} ({lang}, {codec})")
            
            if not subtitle_streams:
                self.log.info("No embedded subtitle streams found")
                return None
            
            # Extract each subtitle stream
            extracted_subs = {}
            
            for stream_idx, lang, codec in subtitle_streams:
                try:
                    tmp_subtitle = tempfile.mktemp(suffix=f"_{lang}.srt")
                    
                    # Extract subtitle to SRT format
                    cmd = [
                        ffmpeg_exe,
                        "-i", video_path,
                        "-map", stream_idx,
                        "-c:s", "srt",  # Convert to SRT format
                        tmp_subtitle
                    ]
                    
                    completed = await asyncio.to_thread(_subprocess_run_wrapper, cmd)
                    
                    if completed.returncode == 0 and os.path.exists(tmp_subtitle):
                        with open(tmp_subtitle, 'r', encoding='utf-8', errors='ignore') as f:
                            subtitle_content = f.read()
                        
                        if subtitle_content.strip():
                            extracted_subs[lang] = subtitle_content
                            self.log.info(f"Extracted {lang} subtitles ({len(subtitle_content)} bytes)")
                        
                        os.unlink(tmp_subtitle)
                    else:
                        self.log.warning(f"Failed to extract subtitle stream {stream_idx}")
                        
                except Exception as e:
                    self.log.warning(f"Error extracting subtitle {stream_idx}: {e}")
            
            return extracted_subs if extracted_subs else None
            
        except Exception as e:
            self.log.warning(f"Embedded subtitle extraction failed: {e}")
            return None
    
    async def _save_subtitle_file(
        self,
        content: str,
        base_filename: str,
        subtitle_type: str,
        user_id: str,
        parent_file_id: Optional[str]
    ) -> Optional[str]:
        """Save subtitle content as a file in Open WebUI
        
        Args:
            content: Subtitle file content (SRT format)
            base_filename: Base filename (without extension)
            subtitle_type: Type/language of subtitle (e.g., 'srt', 'eng', 'spa')
            user_id: User ID who owns the file
            parent_file_id: ID of the parent video file (optional)
            
        Returns:
            File ID of saved subtitle file, or None if save failed
        """
        try:
            # Generate filename with timestamp
            timestamp = int(time.time())
            subtitle_filename = f"{base_filename}_{subtitle_type}_{timestamp}.srt"
            
            # Generate unique file ID
            file_id = str(uuid.uuid4())
            
            # Write to temp file
            tmp_file = tempfile.mktemp(suffix='.srt')
            with open(tmp_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            try:
                # Upload to storage using the correct signature
                with open(tmp_file, 'rb') as f:
                    file_data, storage_path = Storage.upload_file(
                        f,
                        file_id,
                        {"subtitle_type": subtitle_type}
                    )
            finally:
                # Clean up temp file
                if os.path.exists(tmp_file):
                    os.unlink(tmp_file)
            
            # Create file record in database
            file_meta = {
                "content_type": "application/x-subrip",
                "size": len(content),
                "subtitle_type": subtitle_type,
            }
            if parent_file_id:
                file_meta["parent_video_id"] = parent_file_id
            
            file_form = FileForm(
                id=file_id,
                filename=subtitle_filename,
                path=storage_path,  # Add the storage path
                meta=file_meta,
                data={
                    "content": content[:1000],  # Store preview
                }
            )
            
            FilesDB.insert_new_file(user_id, file_form)
            
            self.log.info(f"Saved subtitle file: {subtitle_filename} ({file_id})")
            return file_id
            
        except Exception as e:
            self.log.error(f"Failed to save subtitle file: {e}")
            return None
