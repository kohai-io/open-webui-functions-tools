"""
title: Video Transcription + Subtitle Tool
author: open-webui
date: 2024-12-04
version: 2.4
license: MIT
description: Enhanced video transcription tool that extracts audio transcription, built-in subtitles, detects scene changes, captures keyframes, performs OCR, and compares transcription vs subtitles.
requirements: aiohttp, cryptography, pydantic, imageio-ffmpeg, openai-whisper, pillow, opencv-python, easyocr
required_open_webui_version: 0.3.9
"""

from typing import Optional, Callable, Awaitable, Any, Dict, List, Tuple
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
from difflib import SequenceMatcher
from PIL import Image, ImageEnhance

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
            description="Whisper model: 'whisper-1' for API, or 'tiny/base/small/medium/large' for local",
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
        
        # NEW: Subtitle & Scene Detection Configuration
        EXTRACT_SUBTITLES: bool = Field(
            default=True,
            description="Extract built-in subtitles from video if available",
        )
        SCENE_DETECTION_THRESHOLD: float = Field(
            default=0.4,
            description="Scene change detection threshold (0.0-1.0, lower = more sensitive). Set to 0 to disable scene detection.",
        )
        FRAME_EXTRACTION_MODE: str = Field(
            default="interval",
            description="Frame extraction mode: 'scene' (scene changes only), 'interval' (regular intervals), 'both' (combine both methods)",
        )
        FRAME_INTERVAL_SECONDS: float = Field(
            default=0.5,
            description="Extract a frame every N seconds (used in 'interval' or 'both' modes). Use 0.5-1.0 for burnt-in subtitles.",
        )
        MAX_KEYFRAMES: int = Field(
            default=100,
            description="Maximum number of keyframes to extract (increase for longer videos or more frequent subtitle changes)",
        )
        OCR_ENGINE: str = Field(
            default="easyocr",
            description="OCR engine: 'pytesseract', 'easyocr', or 'mistral'",
        )
        OCR_LANGUAGE: str = Field(
            default="pt",
            description="OCR language code: pytesseract/easyocr use 'eng'/'pt'/'es', Mistral is multilingual (use 'en', 'pt', 'es', 'fr', etc.)",
        )
        ENHANCE_FRAMES_FOR_OCR: bool = Field(
            default=True,
            description="Pre-process frames (contrast, sharpness) before OCR to improve text detection accuracy",
        )
        SUBTITLE_ROI_ENABLED: bool = Field(
            default=True,
            description="Extract only subtitle region (center 70% height, full width) to exclude headers/footers",
        )
        SUBTITLE_ROI_TOP_PERCENT: float = Field(
            default=15.0,
            description="Top boundary of subtitle region as % of frame height (0-100). Excludes top 15% by default.",
        )
        SUBTITLE_ROI_BOTTOM_PERCENT: float = Field(
            default=15.0,
            description="Bottom boundary of subtitle region as % of frame height (0-100). Excludes bottom 15% by default.",
        )
        DEDUPLICATE_OCR_TEXT: bool = Field(
            default=True,
            description="Remove duplicate OCR results when consecutive frames show the same subtitle text (recommended for burnt-in subtitles)",
        )
        COMPARE_TRANSCRIPTIONS: bool = Field(
            default=True,
            description="Compare audio transcription with subtitle text and generate similarity report",
        )
        SAVE_KEYFRAMES: bool = Field(
            default=False,
            description="Save extracted keyframe images to storage",
        )
        
        # Mistral OCR Configuration
        MISTRAL_API_KEY: EncryptedStr = Field(
            default="",
            description="Mistral API key for Pixtral vision model OCR",
        )
        MISTRAL_API_BASE_URL: str = Field(
            default="https://api.mistral.ai/v1",
            description="Mistral API base URL",
        )
        MISTRAL_MODEL: str = Field(
            default="mistral-ocr-latest",
            description="Mistral OCR model: 'mistral-ocr-latest' (recommended), 'pixtral-12b-2409', or 'pixtral-large-latest'",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.log = logging.getLogger("video_transcription_subtitle_tool")
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

    async def transcribe_video_with_subtitles(
        self,
        file_id: str = "",
        ocr_language: str = "",
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __messages__: Optional[List[Dict]] = None,
        __files__: Optional[List[Dict]] = None,
        __user__: Optional[Dict] = None,
        **kwargs  # Capture any other parameters
    ) -> str:
        """
        Transcribe a video file to text with enhanced subtitle extraction, scene detection, and comparison.
        
        Args:
            file_id: The ID of the uploaded video file. If not provided or placeholder, will auto-detect from recent messages.
            ocr_language: OCR language code for subtitle extraction (e.g., 'en', 'pt', 'es', 'fr'). If not specified, uses OCR_LANGUAGE valve setting.
            language: (Deprecated, use ocr_language instead) Falls back to this if ocr_language not specified.
            
        Returns:
            Comprehensive transcription report with audio transcription, subtitles, OCR text, and comparison analysis
        """
        
        # Handle both ocr_language and legacy language parameter
        language = kwargs.get("language", "")
        effective_ocr_lang = ocr_language or language
        
        self.log.info(f"Transcribing video with subtitle analysis - file_id: {file_id or 'auto-detect'}, ocr_language: {effective_ocr_lang or 'default'}")
        
        # Store original OCR language and override if user specified
        original_ocr_language = self.valves.OCR_LANGUAGE
        if effective_ocr_lang:
            self.valves.OCR_LANGUAGE = effective_ocr_lang
            self.log.info(f"[OCR_LANGUAGE] User specified: {effective_ocr_lang} (overriding default: {original_ocr_language})")
        
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
            
            # Initialize results dictionary
            results = {
                "audio_transcription": None,
                "subtitles": None,
                "scene_keyframes": [],
                "ocr_text": [],
                "comparison": None
            }
            
            # Extract audio and transcribe
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
                    
                    # Adjust timestamps for chunks after the first
                    if "segments" in chunk_result and chunk_result["segments"]:
                        for seg in chunk_result["segments"]:
                            seg["start"] += time_offset
                            seg["end"] += time_offset
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
                
                results["audio_transcription"] = transcription
                
            finally:
                # Cleanup temp audio files (main file + any chunks)
                if os.path.exists(tmp_audio):
                    os.unlink(tmp_audio)
                
                for chunk_path in chunk_paths:
                    if chunk_path != tmp_audio and os.path.exists(chunk_path):
                        os.unlink(chunk_path)
            
            # Extract subtitles if enabled
            if self.valves.EXTRACT_SUBTITLES:
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {"description": "Extracting built-in subtitles..."}
                    })
                
                subtitles = await self._extract_subtitles(video_path)
                results["subtitles"] = subtitles
            
            # Detect scene changes and extract keyframes
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": f"Extracting keyframes ({self.valves.FRAME_EXTRACTION_MODE} mode)..."}
                })
            
            keyframes = await self._extract_keyframes(video_path)
            results["scene_keyframes"] = keyframes
            
            if not keyframes:
                self.log.warning(f"No keyframes extracted in '{self.valves.FRAME_EXTRACTION_MODE}' mode. Try 'interval' mode or lower threshold.")
            
            # Perform OCR on keyframes
            if keyframes:
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {"description": f"Performing OCR on {len(keyframes)} keyframes..."}
                    })
                
                ocr_results = await self._perform_ocr_on_keyframes(keyframes)
                results["ocr_text"] = ocr_results
            
            # Compare transcriptions if enabled
            if self.valves.COMPARE_TRANSCRIPTIONS:
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {"description": "Comparing audio transcription with subtitles..."}
                    })
                
                comparison = self._compare_transcriptions(
                    results["audio_transcription"],
                    results["subtitles"],
                    results["ocr_text"]
                )
                results["comparison"] = comparison
            
            # Format comprehensive output
            formatted_output = self._format_comprehensive_output(results)
            
            # Save transcription to file metadata
            if self.valves.SAVE_TO_FILE_METADATA:
                try:
                    # Get plain text (no timestamps) for RAG
                    plain_text = transcription.get("text", "")
                    if not plain_text and "segments" in transcription:
                        plain_text = " ".join(seg.get("text", "") for seg in transcription["segments"])
                    
                    # Add subtitle text if available
                    if results["subtitles"] and results["subtitles"].get("text"):
                        plain_text += f"\n\n[SUBTITLES]\n{results['subtitles']['text']}"
                    
                    # Add OCR text if available
                    if results["ocr_text"]:
                        ocr_combined = "\n".join([ocr.get("text", "") for ocr in results["ocr_text"] if ocr.get("text")])
                        if ocr_combined:
                            plain_text += f"\n\n[OCR TEXT FROM KEYFRAMES]\n{ocr_combined}"
                    
                    # Update file metadata
                    file_record = FilesDB.get_file_by_id(file_id)
                    if file_record:
                        current_data = file_record.data or {}
                        current_data["transcription"] = plain_text
                        current_data["subtitle_analysis"] = results["comparison"]
                        FilesDB.update_file_data_by_id(file_id, {"data": current_data})
                        self.log.info(f"Saved comprehensive transcription to file metadata")
                        
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
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Analysis complete!", "done": True}
                })
            
            return formatted_output
                    
        except Exception as e:
            self.log.exception(f"Transcription error: {e}")
            return f"Error: {str(e)}"
        finally:
            # Restore original OCR language setting
            if effective_ocr_lang:
                self.valves.OCR_LANGUAGE = original_ocr_language

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

    async def _extract_subtitles(self, video_path: str) -> Optional[Dict[str, Any]]:
        """Extract built-in subtitles from video using ffmpeg"""
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            
            # First, probe for subtitle streams
            probe_cmd = [
                ffmpeg_exe,
                "-i", video_path,
                "-c", "copy",
                "-f", "null",
                "-"
            ]
            
            probe_result = await asyncio.to_thread(_subprocess_run_wrapper, probe_cmd)
            stderr = probe_result.stderr.decode(errors='ignore') if probe_result.stderr else ''
            
            # Check if subtitles exist
            subtitle_match = re.search(r'Stream #\d+:\d+.*: Subtitle:', stderr)
            if not subtitle_match:
                self.log.info("No built-in subtitles found in video")
                return None
            
            # Extract subtitles to SRT format
            tmp_subtitle = tempfile.mktemp(suffix=".srt")
            
            extract_cmd = [
                ffmpeg_exe,
                "-i", video_path,
                "-map", "0:s:0",  # First subtitle stream
                tmp_subtitle
            ]
            
            extract_result = await asyncio.to_thread(_subprocess_run_wrapper, extract_cmd)
            
            if extract_result.returncode != 0 or not os.path.exists(tmp_subtitle):
                self.log.warning("Subtitle extraction failed")
                return None
            
            # Parse SRT file
            with open(tmp_subtitle, 'r', encoding='utf-8', errors='ignore') as f:
                srt_content = f.read()
            
            os.unlink(tmp_subtitle)
            
            # Parse SRT into structured format
            subtitle_data = self._parse_srt(srt_content)
            
            self.log.info(f"Extracted {len(subtitle_data.get('segments', []))} subtitle segments")
            return subtitle_data
            
        except Exception as e:
            self.log.warning(f"Subtitle extraction failed: {e}")
            return None

    def _parse_srt(self, srt_content: str) -> Dict[str, Any]:
        """Parse SRT subtitle format into structured data"""
        segments = []
        lines = srt_content.strip().split('\n\n')
        
        for block in lines:
            lines_in_block = block.strip().split('\n')
            if len(lines_in_block) < 3:
                continue
            
            try:
                # Parse timestamp line (format: 00:00:20,000 --> 00:00:24,400)
                timestamp_line = lines_in_block[1]
                times = timestamp_line.split(' --> ')
                if len(times) != 2:
                    continue
                
                start = self._srt_timestamp_to_seconds(times[0])
                end = self._srt_timestamp_to_seconds(times[1])
                
                # Get subtitle text (remaining lines)
                text = ' '.join(lines_in_block[2:])
                
                segments.append({
                    "start": start,
                    "end": end,
                    "text": text
                })
            except Exception as e:
                self.log.debug(f"Failed to parse SRT block: {e}")
                continue
        
        # Combine all text
        full_text = ' '.join(seg['text'] for seg in segments)
        
        return {
            "text": full_text,
            "segments": segments
        }

    def _srt_timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert SRT timestamp (HH:MM:SS,mmm) to seconds"""
        # Remove milliseconds
        time_part, ms_part = timestamp.replace(',', '.').split('.')
        h, m, s = map(int, time_part.split(':'))
        ms = int(ms_part)
        return h * 3600 + m * 60 + s + ms / 1000.0

    async def _extract_keyframes(self, video_path: str) -> List[Dict[str, Any]]:
        """Extract keyframes using configured mode (scene detection, interval, or both)"""
        try:
            mode = self.valves.FRAME_EXTRACTION_MODE.lower()
            
            if mode == "scene":
                return await self._extract_keyframes_scene(video_path)
            elif mode == "interval":
                return await self._extract_keyframes_interval(video_path)
            elif mode == "both":
                # Combine both methods
                scene_frames = await self._extract_keyframes_scene(video_path)
                interval_frames = await self._extract_keyframes_interval(video_path)
                
                # Merge and deduplicate by frame number
                all_frames = scene_frames + interval_frames
                # Sort by index and limit
                all_frames.sort(key=lambda x: x.get("index", 0))
                return all_frames[:self.valves.MAX_KEYFRAMES]
            else:
                self.log.error(f"Unknown FRAME_EXTRACTION_MODE: {mode}")
                return await self._extract_keyframes_interval(video_path)  # Fallback
                
        except Exception as e:
            self.log.error(f"Keyframe extraction failed: {e}")
            return []

    async def _extract_keyframes_scene(self, video_path: str) -> List[Dict[str, Any]]:
        """Extract keyframes at scene changes using ffmpeg"""
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            
            # Create temp directory for keyframes
            tmp_dir = tempfile.mkdtemp(prefix="keyframes_scene_")
            self.log.info(f"[KEYFRAMES:scene] Created temp directory: {tmp_dir}")
            
            # Use ffmpeg scene detection filter to extract keyframes
            output_pattern = os.path.join(tmp_dir, "frame_%04d.jpg")
            
            cmd = [
                ffmpeg_exe,
                "-i", video_path,
                "-vf", f"select='gt(scene,{self.valves.SCENE_DETECTION_THRESHOLD})',setpts=N/FRAME_RATE/TB",
                "-vsync", "vfr",
                "-frames:v", str(self.valves.MAX_KEYFRAMES),
                "-q:v", "2",  # High quality
                output_pattern
            ]
            
            self.log.info(f"[KEYFRAMES:scene] Extracting with threshold={self.valves.SCENE_DETECTION_THRESHOLD}, max={self.valves.MAX_KEYFRAMES}")
            self.log.debug(f"[KEYFRAMES:scene] Command: {' '.join(cmd)}")
            completed = await asyncio.to_thread(_subprocess_run_wrapper, cmd)
            
            if completed.returncode != 0:
                stderr = completed.stderr.decode(errors='ignore') if completed.stderr else ''
                self.log.warning(f"[KEYFRAMES:scene] FFmpeg warning (code {completed.returncode}): {stderr[:500]}")
            else:
                stderr = completed.stderr.decode(errors='ignore') if completed.stderr else ''
                self.log.debug(f"[KEYFRAMES:scene] FFmpeg output: {stderr[:500]}")
            
            return await self._collect_keyframes_from_dir(tmp_dir, "scene")
            
        except Exception as e:
            self.log.error(f"Scene-based keyframe extraction failed: {e}")
            return []

    async def _extract_keyframes_interval(self, video_path: str) -> List[Dict[str, Any]]:
        """Extract keyframes at regular time intervals"""
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            
            # Get video duration first
            duration = await self._get_video_duration(video_path)
            if not duration:
                self.log.error("[KEYFRAMES:interval] Could not determine video duration")
                return []
            
            # Calculate number of frames to extract
            num_frames = min(
                int(duration / self.valves.FRAME_INTERVAL_SECONDS),
                self.valves.MAX_KEYFRAMES
            )
            
            self.log.info(f"[KEYFRAMES:interval] Video duration: {duration:.1f}s, extracting {num_frames} frames at {self.valves.FRAME_INTERVAL_SECONDS}s intervals")
            
            # Create temp directory
            tmp_dir = tempfile.mkdtemp(prefix="keyframes_interval_")
            self.log.info(f"[KEYFRAMES:interval] Created temp directory: {tmp_dir}")
            
            output_pattern = os.path.join(tmp_dir, "frame_%04d.jpg")
            
            # Extract frames at regular intervals using fps filter
            # fps=1/N means one frame every N seconds
            cmd = [
                ffmpeg_exe,
                "-i", video_path,
                "-vf", f"fps=1/{self.valves.FRAME_INTERVAL_SECONDS}",
                "-frames:v", str(num_frames),
                "-q:v", "2",  # High quality
                output_pattern
            ]
            
            self.log.debug(f"[KEYFRAMES:interval] Command: {' '.join(cmd)}")
            completed = await asyncio.to_thread(_subprocess_run_wrapper, cmd)
            
            if completed.returncode != 0:
                stderr = completed.stderr.decode(errors='ignore') if completed.stderr else ''
                self.log.warning(f"[KEYFRAMES:interval] FFmpeg warning (code {completed.returncode}): {stderr[:500]}")
            else:
                stderr = completed.stderr.decode(errors='ignore') if completed.stderr else ''
                self.log.debug(f"[KEYFRAMES:interval] FFmpeg output: {stderr[:500]}")
            
            return await self._collect_keyframes_from_dir(tmp_dir, "interval", duration)
            
        except Exception as e:
            self.log.error(f"Interval-based keyframe extraction failed: {e}")
            return []

    async def _get_video_duration(self, video_path: str) -> Optional[float]:
        """Get video duration in seconds using ffmpeg"""
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [ffmpeg_exe, "-i", video_path, "-f", "null", "-"]
            
            completed = await asyncio.to_thread(_subprocess_run_wrapper, cmd)
            stderr = completed.stderr.decode(errors='ignore') if completed.stderr else ''
            
            # Parse duration from stderr: "Duration: 00:00:30.50"
            match = re.search(r'Duration:\s*(\d+):(\d+):(\d+\.\d+)', stderr)
            if match:
                hours, minutes, seconds = float(match.group(1)), float(match.group(2)), float(match.group(3))
                duration = hours * 3600 + minutes * 60 + seconds
                return duration
            return None
        except Exception as e:
            self.log.warning(f"Failed to get video duration: {e}")
            return None

    async def _collect_keyframes_from_dir(self, tmp_dir: str, mode: str, duration: Optional[float] = None) -> List[Dict[str, Any]]:
        """Collect extracted keyframes from directory"""
        keyframes = []
        
        # Check if directory exists and list files
        if not os.path.exists(tmp_dir):
            self.log.error(f"[KEYFRAMES:{mode}] Temp directory does not exist: {tmp_dir}")
            return []
        
        all_files = os.listdir(tmp_dir)
        self.log.info(f"[KEYFRAMES:{mode}] Files in temp dir: {len(all_files)} total")
        
        frame_files = sorted([f for f in all_files if f.endswith('.jpg')])
        self.log.info(f"[KEYFRAMES:{mode}] Found {len(frame_files)} JPG files: {frame_files[:5]}..." if len(frame_files) > 5 else f"[KEYFRAMES:{mode}] Found {len(frame_files)} JPG files: {frame_files}")
        
        for idx, frame_file in enumerate(frame_files):
            frame_path = os.path.join(tmp_dir, frame_file)
            file_size = os.path.getsize(frame_path) if os.path.exists(frame_path) else 0
            
            self.log.debug(f"[KEYFRAMES:{mode}] Processing frame {idx}: {frame_file} ({file_size} bytes)")
            
            # Calculate approximate timestamp for interval mode
            timestamp = None
            if mode == "interval" and duration is not None:
                # Frame index * interval = timestamp
                timestamp = idx * self.valves.FRAME_INTERVAL_SECONDS
            
            keyframe_data = {
                "index": idx,
                "path": frame_path,
                "filename": frame_file,
                "size_bytes": file_size,
                "temp_dir": tmp_dir,
                "extraction_mode": mode,
                "timestamp": timestamp  # Add timestamp for better matching
            }
            
            # Save to storage if enabled
            if self.valves.SAVE_KEYFRAMES:
                try:
                    with open(frame_path, 'rb') as f:
                        image_data = f.read()
                    
                    # Upload to storage
                    saved_path = Storage.upload_file(
                        file=image_data,
                        filename=f"keyframe_{mode}_{uuid.uuid4().hex[:8]}_{frame_file}",
                        content_type="image/jpeg"
                    )
                    keyframe_data["storage_path"] = saved_path
                    self.log.info(f"[KEYFRAMES:{mode}] Saved to storage: {saved_path}")
                except Exception as e:
                    self.log.warning(f"[KEYFRAMES:{mode}] Failed to save keyframe to storage: {e}")
            
            keyframes.append(keyframe_data)
        
        self.log.info(f"[KEYFRAMES:{mode}] ✓ Extracted {len(keyframes)} keyframes")
        
        return keyframes

    def _is_valid_ocr_text(self, text: str) -> bool:
        """
        Filter out gibberish OCR results using heuristics.
        Returns True if text appears to be valid language content.
        """
        if not text or len(text.strip()) < 3:
            return False
        
        # Count alphabetic characters vs total characters
        alpha_chars = sum(c.isalpha() for c in text)
        total_chars = len(text.replace(' ', ''))
        
        if total_chars == 0:
            return False
        
        alpha_ratio = alpha_chars / total_chars
        
        # Valid text should be at least 50% alphabetic characters
        # This filters out: "${ }^{\text{m}}", random symbols, etc.
        if alpha_ratio < 0.5:
            self.log.debug(f"[OCR:filter] Rejected: low alpha ratio ({alpha_ratio:.1%}): '{text[:50]}...'")
            return False
        
        # Check for excessive character repetition (gibberish indicator)
        # Count unique chars vs total
        unique_chars = len(set(text.lower().replace(' ', '')))
        if unique_chars > 0 and len(text) / unique_chars > 8:
            self.log.debug(f"[OCR:filter] Rejected: excessive repetition: '{text[:50]}...'")
            return False
        
        return True

    def _deduplicate_ocr_results(self, ocr_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove consecutive duplicate OCR results to keep only unique subtitle changes.
        Uses similarity comparison to detect duplicates (allows minor OCR variations).
        """
        if not self.valves.DEDUPLICATE_OCR_TEXT or not ocr_results:
            return ocr_results
        
        deduplicated = []
        prev_text = None
        
        for result in ocr_results:
            current_text = result.get("text", "").strip().lower()
            
            # Skip if text is very similar to previous (>85% similar)
            if prev_text:
                similarity = SequenceMatcher(None, prev_text, current_text).ratio()
                if similarity > 0.85:
                    self.log.debug(
                        f"[OCR:dedup] Skipping frame {result['frame_index']}: "
                        f"{similarity:.0%} similar to previous"
                    )
                    continue
            
            deduplicated.append(result)
            prev_text = current_text
        
        if len(deduplicated) < len(ocr_results):
            self.log.info(
                f"[OCR:dedup] Removed {len(ocr_results) - len(deduplicated)} duplicate frames, "
                f"kept {len(deduplicated)} unique subtitle changes"
            )
        
        return deduplicated

    async def _perform_ocr_on_keyframes(self, keyframes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform OCR on extracted keyframes"""
        ocr_results = []
        
        if not keyframes:
            self.log.warning("[OCR] No keyframes provided for OCR processing")
            return []
        
        self.log.info(f"[OCR] Starting OCR on {len(keyframes)} keyframes using '{self.valves.OCR_ENGINE}' engine")
        
        try:
            if self.valves.OCR_ENGINE == "pytesseract":
                ocr_results = await self._ocr_with_pytesseract(keyframes)
            elif self.valves.OCR_ENGINE == "easyocr":
                ocr_results = await self._ocr_with_easyocr(keyframes)
            elif self.valves.OCR_ENGINE == "mistral":
                ocr_results = await self._ocr_with_mistral(keyframes)
            else:
                self.log.error(f"[OCR] Unknown OCR engine: {self.valves.OCR_ENGINE}")
                return []
            
            self.log.info(f"[OCR] Raw results: {len(ocr_results)} frames processed")
            
            # Filter out frames with no text or invalid/gibberish text
            filtered_results = [
                r for r in ocr_results 
                if r.get("text", "").strip() and self._is_valid_ocr_text(r.get("text", ""))
            ]
            
            self.log.info(f"[OCR] After filtering: {len(filtered_results)}/{len(keyframes)} frames contained valid text")
            
            # Deduplicate consecutive frames showing the same subtitle
            deduplicated_results = self._deduplicate_ocr_results(filtered_results)
            
            self.log.info(f"[OCR] ✓ Completed: {len(deduplicated_results)} unique subtitle frames from {len(keyframes)} total frames")
            
            return deduplicated_results
            
        except Exception as e:
            self.log.error(f"OCR processing failed: {e}")
        
        return ocr_results

    async def _ocr_with_pytesseract(self, keyframes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform OCR using pytesseract"""
        try:
            import pytesseract
            
            self.log.info(f"[OCR:pytesseract] Processing {len(keyframes)} frames with language={self.valves.OCR_LANGUAGE}")
            results = []
            
            for keyframe in keyframes:
                try:
                    # Crop to subtitle ROI first (if enabled)
                    cropped_path = self._crop_subtitle_roi(keyframe["path"])
                    
                    # Enhance frame for better OCR (if enabled)
                    enhanced_path = self._enhance_frame_for_ocr(cropped_path)
                    
                    # Read image with PIL
                    image = Image.open(enhanced_path)
                    self.log.debug(f"[OCR:pytesseract] Frame {keyframe['index']}: size={image.size}, mode={image.mode}")
                    
                    # Perform OCR
                    text = await asyncio.to_thread(
                        pytesseract.image_to_string,
                        image,
                        lang=self.valves.OCR_LANGUAGE
                    )
                    
                    self.log.debug(f"[OCR:pytesseract] Frame {keyframe['index']}: extracted {len(text)} chars")
                    
                    results.append({
                        "frame_index": keyframe["index"],
                        "frame_file": keyframe["filename"],
                        "text": text.strip(),
                        "ocr_engine": "pytesseract"
                    })
                    
                except Exception as e:
                    self.log.warning(f"[OCR:pytesseract] Failed for frame {keyframe['index']}: {e}")
                    continue
            
            return results
            
        except ImportError:
            self.log.error("pytesseract not installed. Install with: pip install pytesseract")
            return []

    async def _ocr_with_easyocr(self, keyframes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform OCR using easyocr"""
        try:
            import easyocr
            
            # Map common language codes for easyocr
            lang_map = {
                "en": "en", "eng": "en",
                "pt": "pt", "por": "pt",
                "es": "es", "spa": "es",
                "fr": "fr", "fra": "fr"
            }
            easyocr_lang = lang_map.get(self.valves.OCR_LANGUAGE.lower(), self.valves.OCR_LANGUAGE)
            
            self.log.info(f"[OCR:easyocr] Initializing reader with language={easyocr_lang}")
            # Initialize reader (cached for multiple frames)
            reader = easyocr.Reader([easyocr_lang])
            
            results = []
            
            for keyframe in keyframes:
                try:
                    # Crop to subtitle ROI first (if enabled)
                    cropped_path = self._crop_subtitle_roi(keyframe["path"])
                    
                    # Enhance frame for better OCR (if enabled)
                    enhanced_path = self._enhance_frame_for_ocr(cropped_path)
                    
                    self.log.debug(f"[OCR:easyocr] Processing frame {keyframe['index']}: {enhanced_path}")
                    
                    # Perform OCR on processed image
                    ocr_result = await asyncio.to_thread(
                        reader.readtext,
                        enhanced_path
                    )
                    
                    self.log.debug(f"[OCR:easyocr] Frame {keyframe['index']}: detected {len(ocr_result)} text regions")
                    
                    # Combine all detected text
                    text = ' '.join([detection[1] for detection in ocr_result])
                    
                    self.log.debug(f"[OCR:easyocr] Frame {keyframe['index']}: extracted {len(text)} chars: '{text[:100]}...'")
                    
                    result_data = {
                        "frame_index": keyframe["index"],
                        "frame_file": keyframe["filename"],
                        "text": text.strip(),
                        "ocr_engine": "easyocr",
                        "detections_count": len(ocr_result)
                    }
                    # Add timestamp if available
                    if keyframe.get("timestamp") is not None:
                        result_data["timestamp"] = keyframe["timestamp"]
                    results.append(result_data)
                    
                except Exception as e:
                    self.log.warning(f"[OCR:easyocr] Failed for frame {keyframe['index']}: {e}")
                    continue
            
            return results
            
        except ImportError:
            self.log.error("easyocr not installed. Install with: pip install easyocr")
            return []

    def _crop_subtitle_roi(self, image_path: str) -> str:
        """
        Crop frame to subtitle region of interest (ROI) to exclude headers/footers.
        Returns path to cropped image (overwrites original if ROI enabled).
        """
        if not self.valves.SUBTITLE_ROI_ENABLED:
            return image_path
        
        try:
            img = Image.open(image_path)
            width, height = img.size
            
            # Calculate crop boundaries
            top_crop = int(height * (self.valves.SUBTITLE_ROI_TOP_PERCENT / 100))
            bottom_crop = int(height * (self.valves.SUBTITLE_ROI_BOTTOM_PERCENT / 100))
            
            # Crop: (left, top, right, bottom)
            # Keep full width, remove top and bottom percentages
            crop_box = (
                0,                      # left (full width)
                top_crop,               # top (remove header)
                width,                  # right (full width)
                height - bottom_crop    # bottom (remove footer)
            )
            
            cropped_img = img.crop(crop_box)
            
            # Save cropped image (overwrite original)
            cropped_img.save(image_path, 'JPEG', quality=95)
            
            self.log.debug(
                f"[OCR:crop] Cropped to subtitle ROI: {os.path.basename(image_path)} "
                f"({width}x{height} → {width}x{height - top_crop - bottom_crop})"
            )
            return image_path
            
        except Exception as e:
            self.log.warning(f"[OCR:crop] Failed to crop frame {image_path}: {e}")
            return image_path  # Return original path if cropping fails

    def _enhance_frame_for_ocr(self, image_path: str) -> str:
        """
        Enhance frame image to improve OCR accuracy.
        Returns path to enhanced image (overwrites original if enhancement enabled).
        """
        if not self.valves.ENHANCE_FRAMES_FOR_OCR:
            return image_path
        
        try:
            # Open image
            img = Image.open(image_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Enhance contrast (helps with faded text) - moderate boost
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.3)  # 30% more contrast (reduced from 50%)
            
            # Enhance sharpness (helps with blurry text) - moderate boost
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.5)  # 50% more sharpness (reduced from 2x)
            
            # Save enhanced image (overwrite original)
            img.save(image_path, 'JPEG', quality=95)
            
            self.log.debug(f"[OCR:enhance] Enhanced frame: {os.path.basename(image_path)}")
            return image_path
            
        except Exception as e:
            self.log.warning(f"[OCR:enhance] Failed to enhance frame {image_path}: {e}")
            return image_path  # Return original path if enhancement fails

    async def _ocr_with_mistral(self, keyframes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform OCR using Mistral's dedicated OCR endpoint with file upload workflow"""
        try:
            api_key = self.valves.MISTRAL_API_KEY.get_decrypted()
            if not api_key:
                raise Exception("Mistral API key not configured")
            
            headers = {
                "Authorization": f"Bearer {api_key}"
            }
            
            self.log.info(f"[OCR:mistral] Processing {len(keyframes)} frames with model={self.valves.MISTRAL_MODEL} (auto-detect language)")
            results = []
            
            async with aiohttp.ClientSession() as session:
                for keyframe in keyframes:
                    file_id = None
                    try:
                        # Step 0: Crop to subtitle ROI first (if enabled)
                        cropped_path = self._crop_subtitle_roi(keyframe["path"])
                        
                        # Step 1: Enhance frame for better OCR (if enabled)
                        enhanced_path = self._enhance_frame_for_ocr(cropped_path)
                        
                        # Step 2: Upload image to Mistral
                        upload_url = f"{self.valves.MISTRAL_API_BASE_URL}/files"
                        
                        with open(enhanced_path, "rb") as img_file:
                            form_data = aiohttp.FormData()
                            form_data.add_field("purpose", "ocr")
                            form_data.add_field(
                                "file",
                                img_file,
                                filename=keyframe["filename"],
                                content_type="image/jpeg"
                            )
                            
                            self.log.debug(f"[OCR:mistral] Frame {keyframe['index']}: Uploading to Mistral")
                            
                            async with session.post(
                                upload_url,
                                data=form_data,
                                headers=headers,
                                timeout=aiohttp.ClientTimeout(total=30)
                            ) as resp:
                                if resp.status != 200:
                                    error_text = await resp.text()
                                    self.log.warning(f"[OCR:mistral] Upload failed for frame {keyframe['index']}: {error_text[:200]}")
                                    continue
                                
                                upload_result = await resp.json()
                                file_id = upload_result.get("id")
                                if not file_id:
                                    self.log.warning(f"[OCR:mistral] No file ID in upload response for frame {keyframe['index']}")
                                    continue
                        
                        # Step 3: Get signed URL
                        url_endpoint = f"{self.valves.MISTRAL_API_BASE_URL}/files/{file_id}/url"
                        
                        async with session.get(
                            url_endpoint,
                            headers=headers,
                            params={"expiry": 1},
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as resp:
                            if resp.status != 200:
                                error_text = await resp.text()
                                self.log.warning(f"[OCR:mistral] Failed to get URL for frame {keyframe['index']}: {error_text[:200]}")
                                continue
                            
                            url_result = await resp.json()
                            signed_url = url_result.get("url")
                            if not signed_url:
                                self.log.warning(f"[OCR:mistral] No signed URL for frame {keyframe['index']}")
                                continue
                        
                        # Step 4: Process OCR with signed URL
                        ocr_url = f"{self.valves.MISTRAL_API_BASE_URL}/ocr"
                        ocr_headers = {
                            **headers,
                            "Content-Type": "application/json"
                        }
                        
                        payload = {
                            "model": self.valves.MISTRAL_MODEL,
                            "document": {
                                "type": "document_url",
                                "document_url": signed_url
                            },
                            "include_image_base64": False
                        }
                        
                        # Note: Mistral OCR endpoint doesn't support 'languages' parameter
                        # It auto-detects language from the image
                        
                        self.log.debug(f"[OCR:mistral] Frame {keyframe['index']}: Processing OCR (auto-detect language)")
                        
                        async with session.post(
                            ocr_url,
                            json=payload,
                            headers=ocr_headers,
                            timeout=aiohttp.ClientTimeout(total=60)
                        ) as resp:
                            if resp.status != 200:
                                error_text = await resp.text()
                                self.log.warning(f"[OCR:mistral] OCR failed for frame {keyframe['index']}: HTTP {resp.status} - {error_text[:200]}")
                                continue
                            
                            result = await resp.json()
                            
                            # Extract text from response
                            pages = result.get("pages", [])
                            if pages and len(pages) > 0:
                                extracted_text = pages[0].get("markdown", "")
                            else:
                                extracted_text = ""
                            
                            if extracted_text and extracted_text.strip():
                                self.log.info(f"[OCR:mistral] Frame {keyframe['index']}: ✓ Text detected ({len(extracted_text)} chars)")
                                result_data = {
                                    "frame_index": keyframe["index"],
                                    "frame_file": keyframe["filename"],
                                    "text": extracted_text.strip(),
                                    "ocr_engine": "mistral"
                                }
                                # Add timestamp if available
                                if keyframe.get("timestamp") is not None:
                                    result_data["timestamp"] = keyframe["timestamp"]
                                results.append(result_data)
                            else:
                                self.log.debug(f"[OCR:mistral] Frame {keyframe['index']}: No text detected")
                        
                    except Exception as e:
                        self.log.warning(f"[OCR:mistral] Failed for frame {keyframe['index']}: {e}")
                    finally:
                        # Step 5: Cleanup - delete uploaded file
                        if file_id:
                            try:
                                delete_url = f"{self.valves.MISTRAL_API_BASE_URL}/files/{file_id}"
                                async with session.delete(
                                    delete_url,
                                    headers=headers,
                                    timeout=aiohttp.ClientTimeout(total=10)
                                ) as resp:
                                    if resp.status == 200:
                                        self.log.debug(f"[OCR:mistral] Deleted file {file_id}")
                            except Exception as cleanup_error:
                                self.log.debug(f"[OCR:mistral] Cleanup failed for {file_id}: {cleanup_error}")
            
            return results
            
        except Exception as e:
            self.log.error(f"[OCR:mistral] Initialization failed: {e}")
            return []

    def _compare_transcriptions(
        self,
        audio_transcription: Optional[Dict[str, Any]],
        subtitles: Optional[Dict[str, Any]],
        ocr_text: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare audio transcription with subtitles and OCR text"""
        
        comparison = {
            "audio_vs_subtitles": None,
            "audio_vs_ocr": None,
            "subtitles_vs_ocr": None,
            "summary": ""
        }
        
        # Get text strings
        audio_text = audio_transcription.get("text", "") if audio_transcription else ""
        subtitle_text = subtitles.get("text", "") if subtitles else ""
        ocr_combined = " ".join([ocr.get("text", "") for ocr in ocr_text if ocr.get("text")])
        
        # Compare audio vs subtitles
        if audio_text and subtitle_text:
            similarity = self._calculate_similarity(audio_text, subtitle_text)
            comparison["audio_vs_subtitles"] = {
                "similarity": similarity,
                "differences": self._find_differences(audio_text, subtitle_text)
            }
        
        # Compare audio vs OCR
        if audio_text and ocr_combined:
            similarity = self._calculate_similarity(audio_text, ocr_combined)
            comparison["audio_vs_ocr"] = {
                "similarity": similarity,
                "differences": self._find_differences(audio_text, ocr_combined)
            }
        
        # Compare subtitles vs OCR
        if subtitle_text and ocr_combined:
            similarity = self._calculate_similarity(subtitle_text, ocr_combined)
            comparison["subtitles_vs_ocr"] = {
                "similarity": similarity,
                "differences": self._find_differences(subtitle_text, ocr_combined)
            }
        
        # Generate summary
        summary_parts = []
        
        if comparison["audio_vs_subtitles"]:
            sim = comparison["audio_vs_subtitles"]["similarity"]
            summary_parts.append(f"Audio vs Subtitles: {sim:.1%} similar")
        
        if comparison["audio_vs_ocr"]:
            sim = comparison["audio_vs_ocr"]["similarity"]
            summary_parts.append(f"Audio vs OCR: {sim:.1%} similar")
        
        if comparison["subtitles_vs_ocr"]:
            sim = comparison["subtitles_vs_ocr"]["similarity"]
            summary_parts.append(f"Subtitles vs OCR: {sim:.1%} similar")
        
        comparison["summary"] = " | ".join(summary_parts) if summary_parts else "No comparisons available"
        
        return comparison

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two text strings"""
        # Normalize texts
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        if not text1 or not text2:
            return 0.0
        
        # Use SequenceMatcher for similarity
        matcher = SequenceMatcher(None, text1, text2)
        return matcher.ratio()

    def _find_differences(self, text1: str, text2: str, max_diff_items: int = 10) -> List[str]:
        """Find major differences between two texts"""
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        if not text1 or not text2:
            return []
        
        # Use difflib to find differences
        from difflib import ndiff
        
        words1 = text1.split()
        words2 = text2.split()
        
        diff = list(ndiff(words1, words2))
        
        # Extract unique additions and deletions
        differences = []
        for item in diff:
            if item.startswith('- ') or item.startswith('+ '):
                differences.append(item)
                if len(differences) >= max_diff_items:
                    break
        
        return differences

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
            form_data.add_field("model", self.valves.WHISPER_MODEL)
            
            if self.valves.WHISPER_LANGUAGE:
                form_data.add_field("language", self.valves.WHISPER_LANGUAGE)
            
            if self.valves.INCLUDE_TIMESTAMPS:
                form_data.add_field("response_format", "verbose_json")
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
            
            result = await asyncio.to_thread(
                model.transcribe,
                audio_path,
                language=self.valves.WHISPER_LANGUAGE,
            )
            
            return result
            
        except Exception as e:
            self.log.error(f"Local transcription failed: {e}")
            raise

    def _format_comprehensive_output(self, results: Dict[str, Any]) -> str:
        """Format comprehensive output including all analysis results"""
        output = ["# 📹 Video Transcription + Subtitle Analysis Report\n"]
        
        # Audio Transcription Section
        output.append("## 🎤 Audio Transcription\n")
        if results["audio_transcription"]:
            formatted = self._format_transcription(results["audio_transcription"])
            output.append(formatted)
        else:
            output.append("*No audio transcription available*")
        
        output.append("\n---\n")
        
        # Subtitles Section
        output.append("## 📝 Built-in Subtitles\n")
        if results["subtitles"]:
            subtitle_formatted = self._format_transcription(results["subtitles"])
            output.append(subtitle_formatted)
        else:
            output.append("*No built-in subtitles found in video*")
        
        output.append("\n---\n")
        
        # Keyframes Section
        output.append("## 🎬 Keyframe Extraction\n")
        if results["scene_keyframes"]:
            output.append(f"**Extracted {len(results['scene_keyframes'])} keyframes** from scene changes\n")
            # Show first few keyframe details
            for kf in results["scene_keyframes"][:5]:
                size_kb = kf.get("size_bytes", 0) / 1024
                output.append(f"- Frame {kf['index']}: `{kf['filename']}` ({size_kb:.1f} KB)\n")
            if len(results["scene_keyframes"]) > 5:
                output.append(f"- *...and {len(results['scene_keyframes']) - 5} more*\n")
        else:
            output.append("*No keyframes extracted (try lowering SCENE_DETECTION_THRESHOLD)*\n")
        
        output.append("\n")
        
        # OCR Section
        output.append("## 🔍 OCR Text from Keyframes\n")
        if results["ocr_text"]:
            output.append(f"*Extracted text from {len(results['ocr_text'])} keyframes with visible text:*\n\n")
            for idx, ocr_result in enumerate(results["ocr_text"][:10], 1):  # Limit to first 10
                frame_idx = ocr_result.get("frame_index", "?")
                text = ocr_result.get("text", "")[:200]  # Limit text length
                engine = ocr_result.get("ocr_engine", "unknown")
                output.append(f"**Frame {frame_idx}** ({engine}): {text}...\n\n")
            if len(results["ocr_text"]) > 10:
                output.append(f"*...and {len(results['ocr_text']) - 10} more frames*\n")
        else:
            if results["scene_keyframes"]:
                output.append(f"*No text detected in {len(results['scene_keyframes'])} keyframes (OCR engine: {self.valves.OCR_ENGINE})*\n")
            else:
                output.append("*No keyframes available for OCR*\n")
        
        output.append("\n---\n")
        
        # Comparison Section
        output.append("## 📊 Comparison Analysis\n")
        if results["comparison"]:
            comp = results["comparison"]
            output.append(f"**Summary:** {comp.get('summary', 'N/A')}\n")
            
            if comp.get("audio_vs_subtitles"):
                avs = comp["audio_vs_subtitles"]
                output.append(f"\n### Audio vs Subtitles\n")
                output.append(f"- **Similarity:** {avs['similarity']:.1%}\n")
                if avs.get("differences"):
                    output.append(f"- **Sample Differences:** {len(avs['differences'])} found\n")
            
            if comp.get("audio_vs_ocr"):
                avo = comp["audio_vs_ocr"]
                output.append(f"\n### Audio vs OCR Text\n")
                output.append(f"- **Similarity:** {avo['similarity']:.1%}\n")
            
            if comp.get("subtitles_vs_ocr"):
                svo = comp["subtitles_vs_ocr"]
                output.append(f"\n### Subtitles vs OCR Text\n")
                output.append(f"- **Similarity:** {svo['similarity']:.1%}\n")
        else:
            output.append("*No comparison data available*")
        
        return "\n".join(output)

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
