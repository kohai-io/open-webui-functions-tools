"""
title: Video Transcription Tool
author: open-webui
date: 2024-12-06
version: 3.1
license: MIT
description: Extract audio from video files using ffmpeg and transcribe to text using Whisper. Generates SRT subtitle files and extracts embedded subtitles if available.
requirements: aiohttp, cryptography, pydantic, imageio-ffmpeg, openai-whisper
required_open_webui_version: 0.3.9
"""

from typing import Optional, Callable, Awaitable, Any, Dict, List
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
import tempfile
import uuid
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


class Tools:
    class Valves(BaseModel):
        # Whisper Configuration
        WHISPER_MODE: str = Field(
            default="openai",
            description="Transcription mode: 'openai' (API) or 'local' (whisper package)",
        )
        OPENAI_API_KEY: EncryptedStr = Field(
            default="",
            description="OpenAI API key for Whisper API (required for 'openai' mode)",
        )
        OPENAI_API_BASE_URL: str = Field(
            default="https://api.openai.com/v1",
            description="OpenAI API base URL",
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
        INCLUDE_TIMESTAMPS: bool = Field(
            default=True,
            description="Include timestamps in transcription output",
        )
        MAX_AUDIO_SIZE_MB: int = Field(
            default=25,
            description="Maximum audio file size in MB",
        )
        AUDIO_SAMPLE_RATE: int = Field(
            default=16000,
            description="Audio sample rate for extraction (Hz)",
        )
        TIMEOUT: int = Field(
            default=300,
            description="Maximum seconds to wait for transcription",
        )
        SAVE_TO_FILE_METADATA: bool = Field(
            default=True,
            description="Save transcription to file metadata for RAG searchability",
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

    def __init__(self):
        self.valves = self.Valves()
        self.log = logging.getLogger("video_transcription_tool")
        self.log.setLevel(logging.INFO)

    def _find_video_file_in_messages(self, messages: Optional[List[Dict]]) -> Optional[str]:
        """Find the most recent video file ID in the conversation messages"""
        if not messages:
            self.log.info("DEBUG: _find_video_file_in_messages - messages is None/empty")
            return None
        
        self.log.info(f"DEBUG: Searching {len(messages)} messages for video files...")
        
        # Search messages in reverse order (most recent first)
        for idx, msg in enumerate(reversed(messages)):
            self.log.info(f"DEBUG: Checking message {idx}, role: {msg.get('role', 'unknown')}")
            
            # Check for files array in message
            files = msg.get("files", [])
            self.log.info(f"DEBUG: Message has 'files' array with {len(files)} items")
            
            for f in files:
                if isinstance(f, dict):
                    file_id = f.get("id")
                    file_type = f.get("type", "")
                    if file_id and file_type.startswith("video/"):
                        self.log.info(f"Found video file in message: {file_id} ({file_type})")
                        return file_id
            
            # Also check content for file API URLs
            content = msg.get("content", "")
            self.log.info(f"DEBUG: Message content type: {type(content)}, length: {len(content) if isinstance(content, str) else 'N/A'}")
            
            if isinstance(content, str):
                # Match /api/v1/files/{id}/content pattern
                match = re.search(r'/api/v1/files/([a-f0-9-]{36})/content', content)
                self.log.info(f"DEBUG: URL regex match found: {match is not None}")
                if match:
                    file_id = match.group(1)
                    # Verify it's a video
                    file_record = FilesDB.get_file_by_id(file_id)
                    if file_record:
                        content_type = file_record.meta.get("content_type", "")
                        if content_type.startswith("video/"):
                            self.log.info(f"Found video file in URL: {file_id}")
                            return file_id
        
        self.log.info("DEBUG: No video file found in messages, returning None")
        return None

    def _find_most_recent_video_file(self, user_id: Optional[str] = None) -> Optional[str]:
        """Find the most recently uploaded video file in the database, optionally filtered by user"""
        try:
            # Get all files
            from open_webui.models.files import Files as FilesModel
            
            all_files = FilesModel.get_files()
            
            # Filter for video files, optionally by user
            video_files = [
                f for f in all_files
                if f.meta and f.meta.get("content_type", "").startswith("video/")
                and (user_id is None or f.user_id == user_id)  # User scoping
            ]
            
            if video_files:
                # Sort by created_at descending (most recent first)
                video_files.sort(key=lambda f: f.created_at, reverse=True)
                most_recent = video_files[0]
                self.log.info(f"Found most recent video: {most_recent.id} (user: {most_recent.user_id})")
                return most_recent.id
            
            self.log.warning(f"No video files found for user: {user_id}")
            return None
        except Exception as e:
            self.log.error(f"Error finding recent video: {e}")
            return None

    async def transcribe_video(
        self,
        file_id: str = "",
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __messages__: Optional[List[Dict]] = None,
        __files__: Optional[List[Dict]] = None,
        __user__: Optional[Dict] = None,
        __request__: Optional[Any] = None,
        **kwargs  # Capture any other parameters
    ) -> str:
        """
        Transcribe a video file to text.
        
        Args:
            file_id: The ID of the uploaded video file. If not provided or placeholder, will auto-detect from recent messages.
            
        Returns:
            Transcription text with optional timestamps
        """
        
        self.log.info(f"Transcribing video - file_id: {file_id or 'auto-detect'}")
        
        try:
            # Auto-detect video file if file_id is missing or placeholder
            if not file_id or file_id == "video_file_id" or file_id == "file_id":
                self.log.info("Auto-detecting video file...")
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {"description": "Looking for video file..."}
                    })
                
                # Priority 1: Check __files__ parameter (uploaded files in this request)
                if __files__ and len(__files__) > 0:
                    self.log.info(f"Checking __files__ parameter ({len(__files__)} files)...")
                    for f in __files__:
                        if isinstance(f, dict):
                            # __files__ structure: [{'type': 'file', 'file': {'id': '...', 'meta': {'content_type': 'video/mp4'}}}]
                            file_obj = f.get("file", {})
                            f_id = file_obj.get("id") or f.get("id")
                            meta = file_obj.get("meta", {})
                            content_type = meta.get("content_type", "")
                            
                            self.log.info(f"File: {f_id}, content_type: {content_type}")
                            
                            if f_id and content_type.startswith("video/"):
                                file_id = f_id
                                self.log.info(f"Found video in __files__: {file_id}")
                                break
                
                # Priority 2: Check messages for file references
                if not file_id:
                    self.log.info("Checking messages for video files...")
                    file_id = self._find_video_file_in_messages(__messages__)
                
                # Priority 3: Fallback to most recent video in database
                if not file_id:
                    self.log.info("Falling back to database search for recent video...")
                    user_id = __user__.get("id") if __user__ else None
                    file_id = self._find_most_recent_video_file(user_id)
                
                if not file_id:
                    return "Error: No video file found. Please upload a video file first."
                
                self.log.info(f"Auto-detected video file: {file_id}")
            
            # Emit status
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Loading video file..."}
                })
            
            # Get file from database
            file_record = FilesDB.get_file_by_id(file_id)
            if not file_record:
                return f"Error: File not found: {file_id}"
            
            # Check if it's a video file
            content_type = file_record.meta.get("content_type", "")
            if not content_type.startswith("video/"):
                return f"Error: File is not a video. Content type: {content_type}"
            
            # Get video file path
            video_path = Storage.get_file(file_record.path)
            if not video_path or not os.path.exists(video_path):
                return f"Error: Video file not found in storage"
            
            self.log.info(f"Video file path: {video_path}")
            
            # Extract audio
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Extracting audio from video..."}
                })
            
            tmp_audio = await self._extract_audio(video_path)
            chunk_paths = []
            
            try:
                # Check audio size and split if needed
                audio_size_mb = os.path.getsize(tmp_audio) / (1024 * 1024)
                
                if audio_size_mb > self.valves.MAX_AUDIO_SIZE_MB:
                    if not self.valves.SPLIT_LARGE_FILES:
                        return f"Error: Audio file too large ({audio_size_mb:.1f}MB). Maximum: {self.valves.MAX_AUDIO_SIZE_MB}MB. Enable SPLIT_LARGE_FILES to process."
                    
                    self.log.info(f"Audio exceeds {self.valves.MAX_AUDIO_SIZE_MB}MB, splitting into chunks...")
                    if __event_emitter__:
                        await __event_emitter__({
                            "type": "status",
                            "data": {"description": f"Splitting large audio file ({audio_size_mb:.1f}MB)..."}
                        })
                    
                    chunk_paths = await self._split_audio(tmp_audio, max_size_mb=24)
                else:
                    chunk_paths = [tmp_audio]
                
                # Transcribe each chunk
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {"description": f"Transcribing audio with Whisper ({len(chunk_paths)} chunk{'s' if len(chunk_paths) > 1 else ''})..."}
                    })
                
                all_segments = []
                time_offset = 0.0
                
                for i, chunk_path in enumerate(chunk_paths):
                    if len(chunk_paths) > 1:
                        self.log.info(f"Transcribing chunk {i+1}/{len(chunk_paths)}")
                        if __event_emitter__:
                            await __event_emitter__({
                                "type": "status",
                                "data": {"description": f"Transcribing chunk {i+1}/{len(chunk_paths)}..."}
                            })
                    
                    chunk_result = await self._transcribe_audio(chunk_path)
                    
                    if not chunk_result:
                        return f"Error: Transcription failed for chunk {i+1}"
                    
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
                    return "Error: Transcription failed"
                
                # Extract embedded subtitles if enabled
                embedded_subtitles = None
                if self.valves.EXTRACT_EMBEDDED_SUBTITLES:
                    if __event_emitter__:
                        await __event_emitter__({
                            "type": "status",
                            "data": {"description": "Checking for embedded subtitles..."}
                        })
                    embedded_subtitles = await self._extract_embedded_subtitles(video_path)
                
                # Generate SRT file from transcription if enabled
                srt_content = None
                srt_file_id = None
                if self.valves.GENERATE_SRT and "segments" in transcription:
                    if __event_emitter__:
                        await __event_emitter__({
                            "type": "status",
                            "data": {"description": "Generating SRT subtitle file..."}
                        })
                    # Check if diarization was used (segments have speaker labels)
                    has_speakers = self.valves.ENABLE_DIARIZATION and any("speaker" in seg for seg in transcription["segments"])
                    srt_content = self._generate_srt(transcription["segments"], include_speakers=has_speakers)
                    
                    # Save SRT file if enabled
                    if self.valves.SAVE_SUBTITLE_FILES and __user__:
                        srt_file_id = await self._save_subtitle_file(
                            srt_content,
                            file_record.filename,
                            "srt",
                            __user__.get("id"),
                            file_id
                        )
                
                # Save embedded subtitles as files if found
                embedded_file_ids = {}
                if embedded_subtitles and self.valves.SAVE_SUBTITLE_FILES and __user__:
                    for lang, content in embedded_subtitles.items():
                        sub_file_id = await self._save_subtitle_file(
                            content,
                            file_record.filename,
                            lang,
                            __user__.get("id"),
                            file_id
                        )
                        if sub_file_id:
                            embedded_file_ids[lang] = sub_file_id
                
                # Format output
                formatted_output = self._format_transcription(transcription)
                
                # Build comprehensive response
                response_parts = []
                response_parts.append("# Video Transcription Complete\n")
                response_parts.append(f"**Video:** {file_record.filename}\n")
                
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
                
                # Transcription text
                response_parts.append("\n## 📄 Transcription Text\n")
                response_parts.append(formatted_output)
                
                formatted_output = "\n".join(response_parts)
                
                # Save transcription to file metadata and process for RAG searchability
                if self.valves.SAVE_TO_FILE_METADATA:
                    try:
                        # Get plain text (no timestamps) for RAG
                        plain_text = transcription.get("text", "")
                        if not plain_text and "segments" in transcription:
                            plain_text = " ".join(seg.get("text", "") for seg in transcription["segments"])
                        
                        # Update file metadata
                        file_record = FilesDB.get_file_by_id(file_id)
                        if file_record:
                            current_data = file_record.data or {}
                            current_data["transcription"] = plain_text
                            FilesDB.update_file_data_by_id(file_id, {"data": current_data})
                            self.log.info(f"Saved transcription to file metadata")
                            
                            # Process for RAG if we have request object
                            if __request__ and __user__:
                                try:
                                    # Create user object for process_file
                                    from open_webui.models.users import UserModel
                                    user = UserModel(**__user__)
                                    
                                    # Process the transcription text for RAG indexing
                                    process_file(
                                        __request__,
                                        ProcessFileForm(file_id=file_id, content=plain_text),
                                        user=user
                                    )
                                    self.log.info(f"Processed transcription for RAG indexing")
                                except Exception as rag_error:
                                    self.log.warning(f"Failed to process for RAG: {rag_error}")
                    except Exception as e:
                        self.log.warning(f"Failed to save transcription to metadata: {e}")
                
                # Emit file attachments for subtitle files
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
                    
                    # Final status
                    await __event_emitter__({
                        "type": "status",
                        "data": {"description": "Transcription complete!", "done": True}
                    })
                
                return formatted_output
                
            finally:
                # Cleanup temp audio files (main file + any chunks)
                if os.path.exists(tmp_audio):
                    os.unlink(tmp_audio)
                
                for chunk_path in chunk_paths:
                    if chunk_path != tmp_audio and os.path.exists(chunk_path):
                        os.unlink(chunk_path)
                    
        except Exception as e:
            self.log.exception(f"Transcription error: {e}")
            return f"Error: {str(e)}"

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
                content_type = "audio/mpeg"
            else:
                suffix = ".wav"
                codec_params = [
                    "-acodec", "pcm_s16le",
                ]
                content_type = "audio/wav"
            
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
    
    async def _transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio using Whisper"""
        if self.valves.WHISPER_MODE == "openai":
            return await self._transcribe_openai(audio_path)
        elif self.valves.WHISPER_MODE == "local":
            return await self._transcribe_local(audio_path)
        else:
            raise Exception(f"Unknown WHISPER_MODE: {self.valves.WHISPER_MODE}")

    async def _transcribe_openai(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe using OpenAI Whisper API"""
        try:
            api_key = self.valves.OPENAI_API_KEY.get_decrypted()
            if not api_key:
                raise Exception("OpenAI API key not configured")
            
            url = f"{self.valves.OPENAI_API_BASE_URL}/audio/transcriptions"
            
            # Read audio file and check size
            with open(audio_path, "rb") as f:
                audio_data = f.read()
            
            audio_size_mb = len(audio_data) / (1024 * 1024)
            self.log.info(f"Sending {audio_size_mb:.2f}MB audio to OpenAI Whisper API")
            
            if audio_size_mb > 25:
                raise Exception(f"Audio file too large for OpenAI API: {audio_size_mb:.1f}MB (max 25MB)")
            
            # Determine content type and filename from file extension
            ext = os.path.splitext(audio_path)[1].lower()
            if ext == ".mp3":
                content_type = "audio/mpeg"
                filename = "audio.mp3"
            else:
                content_type = "audio/wav"
                filename = "audio.wav"
            
            form_data = aiohttp.FormData()
            form_data.add_field("file", audio_data, filename=filename, content_type=content_type)
            
            # Determine model based on diarization setting
            if self.valves.ENABLE_DIARIZATION:
                model = "gpt-4o-transcribe-diarize"
                self.log.info("Diarization enabled - using gpt-4o-transcribe-diarize model")
            else:
                model = self.valves.WHISPER_MODEL
            
            form_data.add_field("model", model)
            
            if self.valves.WHISPER_LANGUAGE:
                form_data.add_field("language", self.valves.WHISPER_LANGUAGE)
            
            # Handle diarization vs regular transcription
            if self.valves.ENABLE_DIARIZATION:
                # Diarization requires specific format and chunking
                form_data.add_field("response_format", "diarized_json")
                form_data.add_field("chunking_strategy", "auto")
                
                # Add known speakers if provided
                if self.valves.DIARIZATION_KNOWN_SPEAKERS.strip():
                    speakers = [s.strip() for s in self.valves.DIARIZATION_KNOWN_SPEAKERS.split(",") if s.strip()]
                    for speaker in speakers:
                        form_data.add_field("known_speaker_names[]", speaker)
                    self.log.info(f"Using known speakers: {speakers}")
            elif self.valves.INCLUDE_TIMESTAMPS:
                form_data.add_field("response_format", "verbose_json")
                # Request word-level timestamps for better subtitle granularity (not supported with diarization)
                form_data.add_field("timestamp_granularities[]", "word")
                form_data.add_field("timestamp_granularities[]", "segment")
            else:
                form_data.add_field("response_format", "json")
            
            headers = {"Authorization": f"Bearer {api_key}"}
            
            self.log.info(f"Calling OpenAI Whisper API (timeout: {self.valves.TIMEOUT}s)...")
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        data=form_data,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.valves.TIMEOUT),
                    ) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            self.log.error(f"OpenAI API response: status={resp.status}, body={error_text[:500]}")
                            raise Exception(f"OpenAI API error {resp.status}: {error_text}")
                        
                        result = await resp.json()
                        self.log.info("OpenAI transcription successful")
                        return result
                        
            except asyncio.TimeoutError:
                raise Exception(f"OpenAI API timeout after {self.valves.TIMEOUT}s - try increasing TIMEOUT setting")
            except aiohttp.ClientError as e:
                raise Exception(f"Network error calling OpenAI API: {e}")
                    
        except Exception as e:
            self.log.error(f"OpenAI transcription failed: {e}")
            raise

    async def _transcribe_local(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe using local Whisper model"""
        try:
            import whisper
            
            self.log.info(f"Loading Whisper model: {self.valves.WHISPER_MODEL}")
            model = whisper.load_model(self.valves.WHISPER_MODEL)
            
            # Transcribe with word-level timestamps for better subtitle segmentation
            result = await asyncio.to_thread(
                model.transcribe,
                audio_path,
                language=self.valves.WHISPER_LANGUAGE,
                word_timestamps=True,  # Enable word-level timestamps
            )
            
            return result
            
        except Exception as e:
            self.log.error(f"Local transcription failed: {e}")
            raise

    def _format_transcription(self, transcription: Dict[str, Any]) -> str:
        """Format transcription output"""
        if "text" in transcription:
            text = transcription["text"]
        else:
            text = "\n".join([seg.get("text", "") for seg in transcription.get("segments", [])])
        
        if self.valves.INCLUDE_TIMESTAMPS and "segments" in transcription:
            lines = []
            for seg in transcription["segments"]:
                start = self._format_timestamp(seg.get("start", 0))
                end = self._format_timestamp(seg.get("end", 0))
                seg_text = seg.get("text", "").strip()
                lines.append(f"[{start} - {end}] {seg_text}")
            return "\n".join(lines)
        else:
            return text

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as MM:SS"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
    
    def _format_srt_timestamp(self, seconds: float) -> str:
        """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
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
        original_filename: str,
        subtitle_type: str,
        user_id: str,
        parent_file_id: str
    ) -> Optional[str]:
        """Save subtitle content as a file in Open WebUI
        
        Args:
            content: Subtitle file content (SRT format)
            original_filename: Original video filename
            subtitle_type: Type/language of subtitle (e.g., 'srt', 'eng', 'spa')
            user_id: User ID who owns the file
            parent_file_id: ID of the parent video file
            
        Returns:
            File ID of saved subtitle file, or None if save failed
        """
        try:
            # Generate filename
            base_name = os.path.splitext(original_filename)[0]
            subtitle_filename = f"{base_name}_{subtitle_type}.srt"
            
            # Generate unique file ID
            file_id = str(uuid.uuid4())
            
            # Write to temp file
            tmp_file = tempfile.mktemp(suffix='.srt')
            with open(tmp_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            try:
                # Upload to storage using the file_id as the path
                with open(tmp_file, 'rb') as f:
                    file_data, storage_path = Storage.upload_file(
                        f, 
                        file_id,
                        {"subtitle_type": subtitle_type}
                    )
                
                # Create file record in database
                file_form = FileForm(
                    id=file_id,
                    filename=subtitle_filename,
                    path=storage_path,  # Add the storage path
                    meta={
                        "name": subtitle_filename,
                        "content_type": "application/x-subrip",
                        "size": len(content),
                        "subtitle_type": subtitle_type,
                        "parent_video_id": parent_file_id,
                    },
                    data={
                        "content": content[:1000],  # Store preview
                    }
                )
                
                FilesDB.insert_new_file(user_id, file_form)
                
            finally:
                # Clean up temp file
                if os.path.exists(tmp_file):
                    os.unlink(tmp_file)
            
            self.log.info(f"Saved subtitle file: {subtitle_filename} ({file_id})")
            return file_id
            
        except Exception as e:
            self.log.error(f"Failed to save subtitle file: {e}")
            return None
