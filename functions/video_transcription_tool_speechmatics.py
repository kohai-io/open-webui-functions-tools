"""
title: Video Transcription Tool (Speechmatics)
author: open-webui
date: 2024-12-06
version: 3.2
license: MIT
description: Extract audio from video files using ffmpeg and transcribe to text using Speechmatics Batch API. Generates SRT subtitle files with speaker diarization support.
requirements: aiohttp, cryptography, pydantic, imageio-ffmpeg, speechmatics-batch
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
        # Speechmatics Configuration
        SPEECHMATICS_API_KEY: EncryptedStr = Field(
            default="",
            description="Speechmatics API key (required)",
        )
        SPEECHMATICS_API_URL: str = Field(
            default="https://asr.api.speechmatics.com/v2",
            description="Speechmatics API base URL",
        )
        LANGUAGE: str = Field(
            default="en",
            description="Language code (e.g., 'en', 'es', 'fr', 'de'). See Speechmatics docs for full list",
        )
        OPERATING_POINT: str = Field(
            default="enhanced",
            description="Operating point: 'standard' or 'enhanced' (enhanced has better accuracy)",
        )
        ENABLE_DIARIZATION: bool = Field(
            default=False,
            description="Enable speaker diarization (identifies who is speaking)",
        )
        MAX_SPEAKERS: Optional[int] = Field(
            default=None,
            description="Maximum number of speakers (leave None for auto-detect)",
        )
        ENABLE_ENTITIES: bool = Field(
            default=False,
            description="Enable entity detection (names, places, etc.)",
        )
        ENABLE_SENTIMENT: bool = Field(
            default=False,
            description="Enable sentiment analysis",
        )
        PUNCTUATION_PERMITTED_MARKS: str = Field(
            default="all",
            description="Punctuation marks to include: 'all', 'none', or comma-separated list like '.,-?!'",
        )
        MAX_AUDIO_SIZE_MB: int = Field(
            default=500,
            description="Maximum audio file size in MB (Speechmatics supports up to 500MB)",
        )
        AUDIO_SAMPLE_RATE: int = Field(
            default=16000,
            description="Audio sample rate for extraction (Hz)",
        )
        TIMEOUT: int = Field(
            default=600,
            description="Maximum seconds to wait for transcription (batch jobs can take longer)",
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
            description="Maximum duration in seconds for each subtitle segment",
        )
        MIN_SUBTITLE_WORDS: int = Field(
            default=2,
            description="Minimum words per subtitle (prevents single-word orphans)",
        )
        PAUSE_THRESHOLD: float = Field(
            default=0.3,
            description="Gap between words (seconds) to consider a natural pause/break point",
        )
        MAX_SUBTITLE_CHARS: int = Field(
            default=42,
            description="Maximum characters per subtitle line (standard is 42 for readability)",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.log = logging.getLogger("video_transcription_speechmatics")
        self.log.setLevel(logging.INFO)

    def _find_video_file_in_messages(self, messages: Optional[List[Dict]]) -> Optional[str]:
        """Find the most recent video file ID in the conversation messages"""
        if not messages:
            return None
        
        # Search messages in reverse order (most recent first)
        for msg in reversed(messages):
            # Check for files array in message
            files = msg.get("files", [])
            for f in files:
                if isinstance(f, dict):
                    file_id = f.get("id")
                    file_type = f.get("type", "")
                    if file_id and file_type.startswith("video/"):
                        self.log.info(f"Found video file in message: {file_id} ({file_type})")
                        return file_id
            
            # Also check content for file API URLs
            content = msg.get("content", "")
            if isinstance(content, str):
                # Match /api/v1/files/{id}/content pattern
                match = re.search(r'/api/v1/files/([a-f0-9-]{36})/content', content)
                if match:
                    file_id = match.group(1)
                    self.log.info(f"Found video file ID in content: {file_id}")
                    return file_id
        
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
        Transcribe a video file using Speechmatics Batch API.
        
        Args:
            file_id: The ID of the uploaded video file. If not provided, will auto-detect from recent uploads.
            
        Returns:
            Transcription text with optional timestamps
        """
        
        self.log.info(f"Transcribing video - file_id: {file_id or 'auto-detect'}")
        
        try:
            # Auto-detect video file if file_id is missing or placeholder
            if not file_id or file_id == "video_file_id" or file_id == "file_id" or file_id == "auto-detect":
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
                
                # Priority 2: Check messages for video files
                if not file_id:
                    self.log.info("Checking messages for video files...")
                    detected_id = self._find_video_file_in_messages(__messages__)
                    if detected_id:
                        file_id = detected_id
                        self.log.info(f"Found video in messages: {file_id}")
                
                # If still not found, error
                if not file_id:
                    error_msg = "❌ No video file found. Please upload a video file first."
                    self.log.warning(error_msg)
                    if __event_emitter__:
                        await __event_emitter__({
                            "type": "status",
                            "data": {"description": error_msg, "done": True}
                        })
                    return error_msg
            
            # Get file record
            file_record = FilesDB.get_file_by_id(file_id)
            if not file_record:
                return f"❌ Error: File not found: {file_id}"
            
            self.log.info(f"Video file path: {file_record.path}")
            
            # Status update
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": f"Extracting audio from {file_record.filename}..."}
                })
            
            # Extract audio
            video_path = file_record.path
            audio_path = await self._extract_audio(video_path)
            
            # Status update
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Transcribing with Speechmatics..."}
                })
            
            # Transcribe with Speechmatics
            transcription = await self._transcribe_speechmatics(audio_path)
            
            # Clean up audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
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
            if self.valves.GENERATE_SRT and transcription.get("results"):
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {"description": "Generating SRT subtitle file..."}
                    })
                has_speakers = self.valves.ENABLE_DIARIZATION
                srt_content = self._generate_srt(transcription["results"], include_speakers=has_speakers)
                
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
            response_parts.append("# Video Transcription Complete (Speechmatics)\n")
            response_parts.append(f"**Video:** {file_record.filename}\n")
            
            # Embedded subtitles section
            if embedded_subtitles:
                response_parts.append("\n## 📝 Embedded Subtitles Found")
                response_parts.append(f"Found **{len(embedded_subtitles)}** embedded subtitle track(s): {', '.join(embedded_subtitles.keys())}\n")
                for lang, content in embedded_subtitles.items():
                    line_count = len(content.strip().split('\n\n'))
                    response_parts.append(f"- **{lang}**: {line_count} subtitle entries")
                    # Note: Download links are emitted as separate messages below
            else:
                response_parts.append("\n## 📝 No Embedded Subtitles")
                response_parts.append("No embedded subtitles found in video.")
            
            # Generated SRT section
            if srt_content:
                subtitle_count = len(srt_content.strip().split('\n\n'))
                response_parts.append(f"\n## 🎬 Generated SRT Subtitles")
                speaker_note = " (with speaker labels)" if self.valves.ENABLE_DIARIZATION else ""
                response_parts.append(f"Created SRT file with **{subtitle_count}** subtitle entries{speaker_note} from Speechmatics transcription.")
                if srt_file_id:
                    response_parts.append(f"\n\n**File ID:** `{srt_file_id}`")
                    response_parts.append(f"\n[Download SRT file](/api/v1/files/{srt_file_id}/content)")
                response_parts.append("")
            
            # Transcription text section
            response_parts.append("\n## 📄 Transcription Text")
            response_parts.append(formatted_output)
            
            # Save transcription to file metadata for RAG
            if self.valves.SAVE_TO_FILE_METADATA:
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {"description": "Saving transcription metadata..."}
                    })
                
                # Extract plain text
                full_text = self._extract_text_from_transcription(transcription)
                
                # Update file data with transcription
                current_data = file_record.data or {}
                current_data["transcription"] = full_text
                FilesDB.update_file_data_by_id(file_id, {"data": current_data})
                self.log.info("Saved transcription to file metadata")
                
                # Process for RAG indexing (if we have request object)
                if __request__ and __user__:
                    try:
                        from open_webui.models.users import UserModel
                        user = UserModel(**__user__)
                        
                        process_file(
                            __request__,
                            ProcessFileForm(file_id=file_id, content=full_text),
                            user=user
                        )
                        self.log.info("Processed transcription for RAG indexing")
                    except Exception as e:
                        self.log.warning(f"Failed to process transcription for RAG: {e}")
            
            # Emit file attachments for subtitle files
            if __event_emitter__:
                self.log.info(f"About to emit messages - srt_file_id: {srt_file_id}, embedded_ids: {list(embedded_file_ids.keys())}")
                # Emit generated SRT file
                if srt_file_id:
                    self.log.info(f"Emitting SRT download link for file: {srt_file_id}")
                    await __event_emitter__({
                        "type": "message",
                        "data": {
                            "content": f"[📥 Download Generated SRT Subtitles](/api/v1/files/{srt_file_id}/content)\n\n"
                                      f"*Generated from Speechmatics transcription with {len(srt_content.strip().split(chr(10)+chr(10)))} subtitle entries*"
                        }
                    })
                    self.log.info("SRT download link message emitted")
                
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
                self.log.info("Emitting final 'Transcription complete!' status")
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Transcription complete!", "done": True}
                })
                self.log.info("Final status emitted")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            error_msg = f"❌ Error during transcription: {str(e)}"
            self.log.error(error_msg)
            self.log.error("Full traceback:", exc_info=True)
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": error_msg, "done": True}
                })
            return error_msg

    async def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video using ffmpeg"""
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            
            # Create temporary file for audio
            audio_ext = self.valves.AUDIO_FORMAT
            temp_audio = tempfile.NamedTemporaryFile(suffix=f".{audio_ext}", delete=False)
            audio_path = temp_audio.name
            temp_audio.close()
            
            # Build ffmpeg command
            cmd = [
                ffmpeg_exe,
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "libmp3lame" if audio_ext == "mp3" else "pcm_s16le",
                "-ar", str(self.valves.AUDIO_SAMPLE_RATE),
            ]
            
            if audio_ext == "mp3":
                cmd.extend(["-b:a", self.valves.MP3_BITRATE])
            
            cmd.extend(["-y", audio_path])
            
            # Run ffmpeg in thread pool
            completed = await asyncio.to_thread(_subprocess_run_wrapper, cmd)
            
            if completed.returncode != 0:
                stderr = completed.stderr.decode(errors='ignore') if completed.stderr else ''
                raise Exception(f"FFmpeg failed: {stderr}")
            
            # Check file size
            audio_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            self.log.info(f"Audio extracted successfully: {audio_size_mb:.2f}MB ({audio_ext})")
            
            return audio_path
            
        except Exception as e:
            self.log.error(f"Audio extraction failed: {e}")
            raise

    async def _transcribe_speechmatics(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe using Speechmatics Batch API"""
        try:
            # Import Speechmatics SDK (correct module path for v0.4.4)
            try:
                from speechmatics.batch._async_client import AsyncClient
                from speechmatics.batch._models import (
                    ConnectionConfig, TranscriptionConfig, JobConfig, FormatType,
                    JobType, OperatingPoint
                )
                from speechmatics.batch._auth import StaticKeyAuth
                from speechmatics.batch._exceptions import AuthenticationError, JobError
                self.log.info("Speechmatics SDK imported successfully")
            except ImportError as e:
                raise Exception(
                    f"Speechmatics SDK not installed correctly. Error: {e}\n"
                    "Please ensure speechmatics-batch is installed:\n"
                    "  pip install speechmatics-batch"
                )
            
            api_key = self.valves.SPEECHMATICS_API_KEY.get_decrypted()
            if not api_key:
                raise Exception("Speechmatics API key not configured")
            
            # Connection configuration (timeouts only)
            connection_config = ConnectionConfig(
                connect_timeout=30.0,
                operation_timeout=float(self.valves.TIMEOUT),
            )
            
            # Build transcription configuration (use enums)
            operating_point = OperatingPoint.ENHANCED if self.valves.OPERATING_POINT.lower() == "enhanced" else OperatingPoint.STANDARD
            transcription_config = TranscriptionConfig(
                language=self.valves.LANGUAGE,
                operating_point=operating_point,
            )
            
            # Add optional features
            if self.valves.ENABLE_DIARIZATION:
                transcription_config.diarization = "speaker"
                if self.valves.MAX_SPEAKERS:
                    transcription_config.speaker_diarization_config = {
                        "max_speakers": self.valves.MAX_SPEAKERS
                    }
            
            if self.valves.ENABLE_ENTITIES:
                transcription_config.enable_entities = True
            
            if self.valves.ENABLE_SENTIMENT:
                transcription_config.sentiment_analysis = {"enable": True}
            
            # Configure punctuation
            if self.valves.PUNCTUATION_PERMITTED_MARKS != "all":
                transcription_config.punctuation_overrides = {
                    "permitted_marks": [] if self.valves.PUNCTUATION_PERMITTED_MARKS == "none" 
                        else list(self.valves.PUNCTUATION_PERMITTED_MARKS.replace(" ", ""))
                }
            
            # Job configuration (use enum for type)
            job_config = JobConfig(
                type=JobType.TRANSCRIPTION,
                transcription_config=transcription_config,
            )
            
            audio_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            self.log.info(f"Sending {audio_size_mb:.2f}MB audio to Speechmatics API")
            
            # Use AsyncClient (url and auth passed directly, not via ConnectionConfig)
            async with AsyncClient(
                auth=StaticKeyAuth(api_key),
                url=self.valves.SPEECHMATICS_API_URL,
                conn_config=connection_config
            ) as client:
                try:
                    # Submit job (audio_file parameter, config not job_config)
                    job_details = await client.submit_job(
                        audio_file=audio_path,
                        config=job_config,
                    )
                    job_id = job_details.id
                    self.log.info(f"Job {job_id} submitted successfully, waiting for transcript")
                    
                    # Wait for completion
                    await client.wait_for_completion(job_id)
                    
                    # Get transcript (use FormatType.JSON, not JSON_V2)
                    transcript_obj = await client.get_transcript(job_id, format_type=FormatType.JSON)
                    self.log.info("Speechmatics transcription successful")
                    
                    # Convert Transcript dataclass to dict
                    from dataclasses import asdict
                    transcript = asdict(transcript_obj)
                    
                    # Log transcript structure for debugging
                    self.log.info(f"Transcript keys: {list(transcript.keys())}")
                    if "results" in transcript:
                        self.log.info(f"Results count: {len(transcript['results'])}")
                    
                    return transcript
                    
                except AuthenticationError:
                    raise Exception("Invalid Speechmatics API key - Check your API_KEY configuration")
                except JobError as e:
                    raise Exception(f"Speechmatics job error: {e}")
                except Exception as e:
                    raise Exception(f"Speechmatics API error: {e}")
                        
        except Exception as e:
            self.log.error(f"Speechmatics transcription failed: {e}")
            raise

    def _format_transcription(self, transcription: Dict[str, Any]) -> str:
        """Format Speechmatics transcription output with timestamps
        
        Speechmatics structure: results[i] = {"type": "word", "start_time": ..., "alternatives": [{"content": "word"}]}
        """
        if not transcription.get("results"):
            return "No transcription results"
        
        lines = []
        current_line = []
        current_start = None
        last_end = 0
        
        for result in transcription["results"]:
            # Skip non-word results
            if not isinstance(result, dict) or result.get("type") != "word":
                continue
                
            alternatives = result.get("alternatives", [])
            if not alternatives or not isinstance(alternatives[0], dict):
                continue
            
            word = alternatives[0].get("content", "")
            start_time = result.get("start_time", 0)
            end_time = result.get("end_time", 0)
            
            # Start new line if gap is significant or line is long
            if current_start is None:
                current_start = start_time
            
            if start_time - last_end > 2.0 and current_line:
                # Finish current line
                text = " ".join(current_line)
                timestamp = self._format_timestamp(current_start)
                lines.append(f"[{timestamp}] {text}")
                current_line = []
                current_start = start_time
            
            current_line.append(word)
            last_end = end_time
        
        # Add final line
        if current_line:
            text = " ".join(current_line)
            timestamp = self._format_timestamp(current_start)
            lines.append(f"[{timestamp}] {text}")
        
        return "\n".join(lines) if lines else "No speech detected"

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
    
    def _extract_text_from_transcription(self, transcription: Dict[str, Any]) -> str:
        """Extract plain text from Speechmatics transcription
        
        Speechmatics structure: results[i] = {"type": "word", "alternatives": [{"content": "word"}]}
        """
        if not transcription.get("results"):
            return ""
        
        words = []
        for result in transcription["results"]:
            # Skip non-word results
            if not isinstance(result, dict) or result.get("type") != "word":
                continue
                
            alternatives = result.get("alternatives", [])
            if not alternatives or not isinstance(alternatives[0], dict):
                continue
            
            words.append(alternatives[0].get("content", ""))
        
        return " ".join(words)
    
    def _generate_srt(self, results: List[Dict[str, Any]], include_speakers: bool = False) -> str:
        """Generate SRT subtitle file content from Speechmatics results
        
        Args:
            results: Speechmatics results array
            include_speakers: If True, prefix each subtitle with [Speaker Name]
        """
        # Build segments from word-level data
        segments = self._build_segments_from_results(results)
        
        # Split into subtitle-appropriate segments if not using speakers
        if not include_speakers:
            subtitle_segments = self._split_into_subtitle_segments(segments)
        else:
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
            
            # SRT format
            srt_lines.append(str(i))
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(text)
            srt_lines.append("")  # Blank line between entries
        
        return "\n".join(srt_lines)
    
    def _build_segments_from_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build segments with word-level timestamps from Speechmatics results
        
        Speechmatics JSON structure:
        results[i] = {
            "type": "word",
            "start_time": 0.5,
            "end_time": 0.8,
            "alternatives": [{"content": "word", "confidence": 0.99}],
            "attaches_to": "previous",
            "is_eos": false,
            ...
        }
        """
        segments = []
        current_segment = {"words": [], "text": "", "speaker": None}
        
        for result in results:
            # Handle if results is not the expected format
            if not isinstance(result, dict):
                continue
            
            # Speechmatics puts word info at top level, not nested  
            result_type = result.get("type")
            if result_type == "word":
                alternatives = result.get("alternatives", [])
                if not alternatives or not isinstance(alternatives[0], dict):
                    continue
                    
                word_data = {
                    "word": alternatives[0].get("content", ""),
                    "start": result.get("start_time", 0),
                    "end": result.get("end_time", 0),
                }
                
                # Check for speaker change (if diarization enabled)
                speaker = result.get("speaker")
                if speaker and speaker != current_segment["speaker"] and current_segment["words"]:
                    # Save current segment
                    current_segment["start"] = current_segment["words"][0]["start"]
                    current_segment["end"] = current_segment["words"][-1]["end"]
                    current_segment["text"] = " ".join([w["word"] for w in current_segment["words"]])
                    segments.append(current_segment)
                    
                    # Start new segment
                    current_segment = {"words": [word_data], "text": "", "speaker": speaker}
                else:
                    current_segment["words"].append(word_data)
                    if speaker:
                        current_segment["speaker"] = speaker
        
        # Add final segment
        if current_segment["words"]:
            current_segment["start"] = current_segment["words"][0]["start"]
            current_segment["end"] = current_segment["words"][-1]["end"]
            current_segment["text"] = " ".join([w["word"] for w in current_segment["words"]])
            segments.append(current_segment)
        
        return segments
    
    def _split_into_subtitle_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split long segments into shorter subtitle-appropriate segments"""
        subtitle_segments = []
        
        for segment in segments:
            words = segment.get("words", [])
            
            if not words:
                subtitle_segments.append(segment)
                continue
            
            # Build shorter segments
            current_segment = {
                "start": words[0]["start"],
                "text": "",
                "words": []
            }
            
            for i, word_info in enumerate(words):
                word = word_info["word"]
                word_start = word_info["start"]
                word_end = word_info["end"]
                
                # Add space before word if not first word in segment
                word_with_space = word if not current_segment["text"] else " " + word
                
                # Check if adding this word would exceed limits
                potential_text = current_segment["text"] + word_with_space
                duration = word_end - current_segment["start"]
                word_count = len(current_segment["words"])
                
                # Detect natural pause: check time gap to NEXT word
                has_pause_after = False
                if i < len(words) - 1:
                    next_word_start = words[i + 1]["start"]
                    gap = next_word_start - word_end
                    has_pause_after = gap > self.valves.PAUSE_THRESHOLD
                
                # Check if current word is a connector word
                word_lower = word.strip().lower()
                is_connector = word_lower in ['to', 'and', 'or', 'but', 'so', 'the', 'a', 'an', 'of', 'in', 'on', 'at', 'for']
                
                # Hard limits
                exceeds_duration = duration > self.valves.MAX_SUBTITLE_DURATION
                exceeds_chars = len(potential_text.strip()) > self.valves.MAX_SUBTITLE_CHARS
                
                # Should we split?
                should_split = False
                
                if word_count + 1 < self.valves.MIN_SUBTITLE_WORDS:
                    should_split = exceeds_duration or exceeds_chars
                else:
                    if exceeds_duration or exceeds_chars:
                        should_split = not is_connector or word_count + 1 >= self.valves.MIN_SUBTITLE_WORDS + 2
                    elif has_pause_after and not is_connector:
                        should_split = True
                
                # Add word to current segment
                current_segment["text"] += word_with_space
                current_segment["words"].append(word_info)
                
                # Split if needed
                if should_split and current_segment["text"].strip():
                    current_segment["end"] = word_end
                    current_segment["text"] = current_segment["text"].strip()
                    subtitle_segments.append(current_segment)
                    
                    # Start new segment
                    if i < len(words) - 1:
                        next_word_start = words[i + 1]["start"]
                        current_segment = {
                            "start": next_word_start,
                            "text": "",
                            "words": []
                        }
            
            # Add final segment
            if current_segment["text"].strip():
                if current_segment["words"]:
                    current_segment["end"] = current_segment["words"][-1]["end"]
                else:
                    current_segment["end"] = segment.get("end", 0)
                current_segment["text"] = current_segment["text"].strip()
                subtitle_segments.append(current_segment)
        
        return subtitle_segments
    
    async def _extract_embedded_subtitles(self, video_path: str) -> Optional[Dict[str, str]]:
        """Extract embedded subtitles from video file using ffmpeg"""
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            
            # Probe video for subtitle streams
            cmd = [ffmpeg_exe, "-i", video_path]
            completed = await asyncio.to_thread(_subprocess_run_wrapper, cmd)
            stderr = completed.stderr.decode(errors='ignore') if completed.stderr else ''
            
            # Parse subtitle streams
            subtitle_streams = []
            for line in stderr.split('\n'):
                if 'Subtitle:' in line and 'Stream #' in line:
                    # Extract stream index and language
                    match = re.search(r'Stream #0:(\d+)(\((\w+)\))?.*Subtitle', line)
                    if match:
                        stream_idx = match.group(1)
                        lang = match.group(3) or f"sub{stream_idx}"
                        subtitle_streams.append((stream_idx, lang))
            
            if not subtitle_streams:
                self.log.info("No embedded subtitle streams found")
                return None
            
            self.log.info(f"Found {len(subtitle_streams)} subtitle stream(s): {[lang for _, lang in subtitle_streams]}")
            
            # Extract each subtitle stream
            subtitles = {}
            for stream_idx, lang in subtitle_streams:
                temp_sub = tempfile.NamedTemporaryFile(suffix=".srt", delete=False)
                sub_path = temp_sub.name
                temp_sub.close()
                
                cmd = [
                    ffmpeg_exe,
                    "-i", video_path,
                    "-map", f"0:{stream_idx}",
                    "-y", sub_path
                ]
                
                completed = await asyncio.to_thread(_subprocess_run_wrapper, cmd)
                
                if completed.returncode == 0 and os.path.exists(sub_path):
                    with open(sub_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if content.strip():
                            subtitles[lang] = content
                            self.log.info(f"Extracted {lang} subtitles")
                
                if os.path.exists(sub_path):
                    os.remove(sub_path)
            
            return subtitles if subtitles else None
            
        except Exception as e:
            self.log.warning(f"Failed to extract embedded subtitles: {e}")
            return None
    
    async def _save_subtitle_file(
        self,
        content: str,
        video_filename: str,
        subtitle_type: str,
        user_id: str,
        parent_file_id: Optional[str] = None
    ) -> Optional[str]:
        """Save subtitle file to Open WebUI storage"""
        try:
            # Generate filename
            base_name = os.path.splitext(video_filename)[0]
            filename = f"{base_name}_{subtitle_type}.srt"
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as f:
                f.write(content)
                temp_path = f.name
            
            # Upload to storage
            file_id = str(uuid.uuid4())
            with open(temp_path, 'rb') as f:
                file_data, storage_path = Storage.upload_file(
                    f,
                    file_id,
                    {"subtitle_type": subtitle_type}
                )
            
            # Create file record
            file_form = FileForm(
                id=file_id,
                filename=filename,
                path=storage_path,
                meta={
                    "subtitle_type": subtitle_type,
                    "parent_file_id": parent_file_id,
                    "source": "speechmatics"
                }
            )
            
            file_record = FilesDB.insert_new_file(user_id, file_form)
            
            # Clean up temp file
            os.remove(temp_path)
            
            self.log.info(f"Saved subtitle file: {filename} ({file_id})")
            return file_id
            
        except Exception as e:
            self.log.error(f"Failed to save subtitle file: {e}")
            return None
