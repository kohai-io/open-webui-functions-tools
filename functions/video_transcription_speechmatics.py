"""
title: Video Transcription Pipeline (Speechmatics)
author: open-webui
date: 2024-12-07
version: 3.4
license: MIT
description: Extract audio from video files using ffmpeg and transcribe to text using Speechmatics Batch API. Generates SRT subtitle files with speaker diarization support.
requirements: aiohttp, cryptography, pydantic, imageio-ffmpeg, speechmatics-batch

USAGE
- Upload a video file in the chat
- The pipeline will:
  1) Extract the video file from __files__ parameter
  2) Extract audio track using ffmpeg
  3) Transcribe audio using Speechmatics Batch API
  4) Return formatted transcription with timestamps
  5) Generate SRT subtitle files with word-level timing

OUTPUT FORMAT
- Returns transcription text with timestamps
- Generates SRT subtitles with granular timing
- Supports speaker diarization
- Saves transcription to Files DB for RAG search
"""

from typing import Optional, Callable, Awaitable, Any, List, Dict, Generator
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


class Pipe:
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
        SAVE_TO_FILE_METADATA: bool = Field(
            default=True,
            description="Save transcription to video file metadata for RAG searchability",
        )

    def __init__(self):
        self.type = "pipe"
        self.id = "video_transcription_speechmatics"
        self.name = "Video Transcription (Speechmatics)"
        self.valves = self.Valves()
        self.log = logging.getLogger("video_transcription_speechmatics_pipe")
        self.log.setLevel(logging.INFO)

    def pipes(self) -> list[dict[str, str]]:
        return [{"id": "video_transcription_speechmatics", "name": "Video Transcription (Speechmatics)"}]

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __event_call__: Optional[Callable[[dict], Awaitable[dict]]] = None,
        __files__: Optional[list] = None,
        __request__: Optional[Any] = None,
    ) -> Generator[str, None, None] | str:
        """
        Process messages and transcribe any video files using Speechmatics
        """
        self.log.info("Video Transcription Pipeline (Speechmatics) - Processing request")
        
        # Check for uploaded files
        if not __files__:
            self.log.debug("No files in __files__ parameter")
        else:
            self.log.info(f"Received {len(__files__)} file(s)")
        
        # Extract messages
        messages = body.get("messages", [])
        if not messages:
            return "No messages provided"
        
        # Find video file from __files__ parameter
        video_file_id = None
        if __files__:
            for file_item in __files__:
                file_id = file_item.get("id")
                if file_id:
                    file_record = FilesDB.get_file_by_id(file_id)
                    if file_record and file_record.meta.get("content_type", "").startswith("video/"):
                        video_file_id = file_id
                        self.log.info(f"Found video file: {file_id}")
                        break
        
        if not video_file_id:
            self.log.debug("No video file uploaded, skipping transcription")
            return body
        
        self.log.info(f"Video file detected: {video_file_id}")
        
        # Transcribe the video
        try:
            transcription_text = await self._transcribe_video_file(
                video_file_id,
                __user__,
                __event_emitter__,
                __request__
            )
            
            # Add transcription as assistant message
            return transcription_text
            
        except Exception as e:
            error_msg = f"❌ Transcription error: {str(e)}"
            self.log.error(error_msg)
            self.log.error("Full traceback:", exc_info=True)
            return error_msg

    async def _transcribe_video_file(
        self,
        file_id: str,
        __user__: Optional[dict],
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]],
        __request__: Optional[Any] = None
    ) -> str:
        """Transcribe a video file using Speechmatics"""
        
        # Get file record
        file_record = FilesDB.get_file_by_id(file_id)
        if not file_record:
            return f"❌ Error: File not found: {file_id}"
        
        self.log.info(f"Processing video: {file_record.filename}")
        
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
        
        # Generate SRT if enabled
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
            
            # Save SRT file if we have user info
            if __user__:
                srt_file_id = await self._save_subtitle_file(
                    srt_content,
                    file_record.filename,
                    "srt",
                    __user__.get("id"),
                    file_id
                )
        
        # Format output
        formatted_output = self._format_transcription(transcription)
        
        # Save transcription to video file metadata for RAG searchability
        if self.valves.SAVE_TO_FILE_METADATA and file_id:
            try:
                # Get plain text (no timestamps) for RAG
                plain_text = ""
                if "results" in transcription:
                    words = []
                    for result in transcription["results"]:
                        if "alternatives" in result:
                            for alt in result["alternatives"]:
                                if "content" in alt:
                                    words.append(alt["content"])
                    plain_text = " ".join(words)
                
                # Update video file metadata
                file_record = FilesDB.get_file_by_id(file_id)
                if file_record:
                    current_data = file_record.data or {}
                    current_data["transcription"] = plain_text
                    FilesDB.update_file_data_by_id(file_id, {"data": current_data})
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
                                ProcessFileForm(file_id=file_id, content=plain_text),
                                user=user
                            )
                            self.log.info(f"Processed video transcription for RAG indexing")
                        except Exception as rag_error:
                            self.log.warning(f"Failed to process for RAG: {rag_error}")
            except Exception as e:
                self.log.warning(f"Failed to save transcription to metadata: {e}")
        
        # Build response
        response_parts = []
        response_parts.append("# Video Transcription Complete (Speechmatics)\n")
        response_parts.append(f"**Video:** {file_record.filename}\n")
        
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
        
        # Final status
        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {"description": "Transcription complete!", "done": True}
            })
        
        return "\n".join(response_parts)

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
            # Import Speechmatics SDK
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
            
            # Connection configuration
            connection_config = ConnectionConfig(
                connect_timeout=30.0,
                operation_timeout=float(self.valves.TIMEOUT),
            )
            
            # Build transcription configuration
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
            
            # Job configuration
            job_config = JobConfig(
                type=JobType.TRANSCRIPTION,
                transcription_config=transcription_config,
            )
            
            audio_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            self.log.info(f"Sending {audio_size_mb:.2f}MB audio to Speechmatics API")
            
            # Use AsyncClient
            async with AsyncClient(
                auth=StaticKeyAuth(api_key),
                url=self.valves.SPEECHMATICS_API_URL,
                conn_config=connection_config
            ) as client:
                try:
                    # Submit job
                    job_details = await client.submit_job(
                        audio_file=audio_path,
                        config=job_config,
                    )
                    job_id = job_details.id
                    self.log.info(f"Job {job_id} submitted successfully, waiting for transcript")
                    
                    # Wait for completion
                    await client.wait_for_completion(job_id)
                    
                    # Get transcript
                    transcript_obj = await client.get_transcript(job_id, format_type=FormatType.JSON)
                    self.log.info("Speechmatics transcription successful")
                    
                    # Convert Transcript dataclass to dict
                    from dataclasses import asdict
                    transcript = asdict(transcript_obj)
                    
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
        """Format Speechmatics transcription output with timestamps"""
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
            
            # Start new line if gap is significant
            if current_start is None:
                current_start = start_time
            
            if start_time - last_end > 2.0 and current_line:
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
    
    def _generate_srt(self, results: List[Dict[str, Any]], include_speakers: bool = False) -> str:
        """Generate SRT subtitle file content from Speechmatics results"""
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
        """Build segments with word-level timestamps from Speechmatics results"""
        segments = []
        current_segment = {"words": [], "text": "", "speaker": None}
        
        for result in results:
            if not isinstance(result, dict):
                continue
            
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
                
                # Check for speaker change
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
                
                # Add space before word if not first word
                word_with_space = word if not current_segment["text"] else " " + word
                
                # Check if adding this word would exceed limits
                potential_text = current_segment["text"] + word_with_space
                duration = word_end - current_segment["start"]
                word_count = len(current_segment["words"])
                
                # Detect natural pause
                has_pause_after = False
                if i < len(words) - 1:
                    next_word_start = words[i + 1]["start"]
                    gap = next_word_start - word_end
                    has_pause_after = gap > self.valves.PAUSE_THRESHOLD
                
                # Check if current word is a connector
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
