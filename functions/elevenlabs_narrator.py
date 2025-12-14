"""title: ElevenLabs Script Narrator
author: open-webui
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 3.1
license: MIT
description: Convert SRT subtitles or drama scripts to synchronized speech using ElevenLabs TTS. Supports SRT format with timestamps and screenplay format with scene headers, character dialogue, and stage directions. Features multi-voice narration, speaker detection, and precise duration tracking.
requirements: aiohttp, cryptography, pydantic, imageio-ffmpeg, pydub, PyPDF2
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
from typing import Optional, Callable, Awaitable, Any, Dict, List, Tuple
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from cryptography.fernet import Fernet, InvalidToken
from pydantic_core import core_schema

# Open WebUI files + storage
from open_webui.models.files import Files as FilesDB, FileForm
from open_webui.storage.provider import Storage


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
        
        # Default Voice Configuration
        DEFAULT_VOICE_ID: str = Field(
            default="21m00Tcm4TlvDq8ikWAM",
            description="Default voice ID (Rachel). Find voices at https://elevenlabs.io/voice-library",
        )
        
        # Speaker-specific voices (format: "Speaker Name:voice_id")
        SPEAKER_VOICE_MAP: str = Field(
            default="",
            description="Map speaker names to voice IDs (comma-separated). Example: 'Alice:pFZP5JQG7iQjIQuC4Bku,Bob:onwK4e9ZLuTAKqWW03F9'",
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
        
        # Timing Settings
        TIMING_MODE: str = Field(
            default="natural",
            description="How to handle timing: 'natural' (allow overlap/gaps, best quality), 'trim' (cut audio that exceeds subtitle duration)",
        )
        
        # Processing Settings
        TIMEOUT: int = Field(
            default=180,
            description="Max seconds to wait for generation (needs to be high for multiple TTS calls)",
        )
        EMIT_INTERVAL: float = Field(
            default=1.0,
            description="Interval in seconds between status emissions",
        )
        
        # Drama Script Settings
        INCLUDE_STAGE_DIRECTIONS: bool = Field(
            default=False,
            description="Include stage directions in narration (drama scripts only)",
        )
        SCENE_ANNOUNCEMENTS: bool = Field(
            default=True,
            description="Announce scene changes with scene headers (drama scripts only)",
        )
        SCENE_VOICE_ID: str = Field(
            default="",
            description="Voice ID for scene announcements (leave empty to use default voice)",
        )
        DEFAULT_DIALOGUE_DURATION: int = Field(
            default=3000,
            description="Default duration per dialogue line in milliseconds (drama scripts only)",
        )
        PAUSE_BETWEEN_LINES: int = Field(
            default=500,
            description="Pause duration between dialogue lines in milliseconds (drama scripts only)",
        )

    def __init__(self):
        self.name = "ElevenLabs Script Narrator"
        self.valves = self.Valves()
        self.log = logging.getLogger("elevenlabs_script_narrator")
        self.log.setLevel(logging.INFO)
        self.last_emit_time = 0

    async def emit_status(
        self,
        event_emitter: Optional[Callable[[dict], Awaitable[None]]],
        level: str,
        message: str,
        done: bool = False,
    ):
        """Emit status with rate limiting."""
        if not event_emitter:
            return
        
        current_time = time.time()
        if current_time - self.last_emit_time < self.valves.EMIT_INTERVAL and not done:
            return
        
        try:
            await event_emitter({
                "type": "status",
                "data": {
                    "description": message,
                    "done": done,
                }
            })
            self.last_emit_time = current_time
        except Exception as e:
            self.log.warning(f"Failed to emit status: {e}")

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __files__: Optional[list] = None,
    ) -> Optional[str]:
        """Main pipeline: Convert SRT subtitles or drama scripts to synchronized speech"""
        self.log.info("=== ElevenLabs Script Narrator Pipeline Called ===")
        
        await self.emit_status(
            __event_emitter__, "info", "🎙️ Initializing script narrator..."
        )
        
        # Validate API key
        api_key = self.valves.ELEVEN_API_KEY.get_decrypted()
        if not api_key:
            return "❌ **Error:** ElevenLabs API key not configured.\n\n**Solution:** Set your `ELEVEN_API_KEY` in the pipeline settings.\n\nGet your API key from: https://elevenlabs.io/"
        
        # Find SRT or drama script file from __files__ parameter
        script_file_id = None
        script_filename = None
        if __files__:
            self.log.info(f"Checking {len(__files__)} file(s) for SRT or drama script")
            for file_item in __files__:
                file_id = file_item.get("id")
                if file_id:
                    file_record = FilesDB.get_file_by_id(file_id)
                    if file_record:
                        filename = file_record.meta.get("name", "").lower()
                        content_type = file_record.meta.get("content_type", "")
                        
                        # Check for SRT, TXT, or PDF files
                        # Also check for "script" keyword in filename
                        is_srt = filename.endswith(".srt") or "srt" in content_type.lower()
                        is_text = filename.endswith(".txt") or content_type == "text/plain"
                        is_pdf = filename.endswith(".pdf") or "pdf" in content_type.lower()
                        has_script_keyword = "script" in filename
                        
                        if is_srt or is_text or (is_pdf and has_script_keyword):
                            script_file_id = file_id
                            script_filename = file_record.meta.get("name", "unknown")
                            self.log.info(f"Found script file: {script_filename} (ID: {file_id})")
                            break
        
        if not script_file_id:
            return "❌ **Error:** No script file found.\n\n**Solution:** Please upload one of the following:\n- SRT subtitle file (.srt)\n- Drama script file (.txt or .pdf with 'script' in filename)\n\n**Example:** Upload `10242-PSS (Script).pdf` or `dialogue.srt`"
        
        # Read script file content
        await self.emit_status(
            __event_emitter__, "info", "📄 Reading script file..."
        )
        
        file_record = FilesDB.get_file_by_id(script_file_id)
        script_content = None
        
        try:
            # Handle PDF files
            if script_filename.lower().endswith('.pdf'):
                try:
                    import PyPDF2
                    with open(file_record.path, "rb") as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        pages_text = []
                        for page in pdf_reader.pages:
                            pages_text.append(page.extract_text())
                        script_content = "\n".join(pages_text)
                    self.log.info(f"Extracted text from {len(pdf_reader.pages)} PDF pages")
                except ImportError:
                    return "❌ **Error:** PyPDF2 not installed.\n\n**Solution:** Install PyPDF2 to process PDF files:\n```\npip install PyPDF2\n```"
            else:
                # Handle text files (SRT, TXT)
                with open(file_record.path, "r", encoding="utf-8") as f:
                    script_content = f.read()
        except Exception as e:
            self.log.error(f"Failed to read script file: {e}")
            return f"❌ **Error:** Failed to read script file: {str(e)}"
        
        if not script_content or not script_content.strip():
            return "❌ **Error:** Script file is empty or unreadable."
        
        # Detect format
        await self.emit_status(
            __event_emitter__, "info", "🔍 Detecting format..."
        )
        
        format_type = self._detect_format(script_content)
        self.log.info(f"Detected format: {format_type}")
        
        # Parse based on detected format
        await self.emit_status(
            __event_emitter__, "info", f"📊 Parsing {format_type.replace('_', ' ')}..."
        )
        
        if format_type == 'srt':
            subtitles = self._parse_srt(script_content)
            if not subtitles:
                return "❌ **Error:** Failed to parse SRT file. Please ensure it's a valid SRT format."
        elif format_type == 'drama_script':
            subtitles = self._parse_drama_script(script_content)
            if not subtitles:
                return "❌ **Error:** Failed to parse drama script. Please ensure it contains dialogue in the format:\n\n```\nINT. LOCATION\nCHARACTER NAME: dialogue text\n```"
        else:
            return f"❌ **Error:** Unrecognized format.\n\n**Detected:** {format_type}\n\n**Supported formats:**\n- SRT subtitles (with timestamps `-->` )\n- Drama scripts (with `INT.`/`EXT.` scenes and `CHARACTER:` dialogue)"
        
        self.log.info(f"Parsed {len(subtitles)} entries from {format_type}")
        
        # Parse speaker voice map
        speaker_voice_map = self._parse_speaker_voice_map()
        
        # Generate TTS for each subtitle
        await self.emit_status(
            __event_emitter__, "info", f"🎤 Generating speech for {len(subtitles)} subtitle entries..."
        )
        
        audio_segments = []
        for i, subtitle in enumerate(subtitles):
            await self.emit_status(
                __event_emitter__, "info", 
                f"🎤 Generating audio {i+1}/{len(subtitles)}: {subtitle['text'][:50]}..."
            )
            
            # Determine voice ID based on speaker
            speaker = subtitle['speaker']
            
            # Handle special speakers for drama scripts
            if speaker == '__SCENE__':
                # Use scene voice or default
                voice_id = self.valves.SCENE_VOICE_ID if self.valves.SCENE_VOICE_ID else self.valves.DEFAULT_VOICE_ID
            elif speaker == '__DIRECTION__':
                # Use default voice for stage directions
                voice_id = self.valves.DEFAULT_VOICE_ID
            else:
                # Regular character dialogue - check voice map
                voice_id = speaker_voice_map.get(speaker, self.valves.DEFAULT_VOICE_ID)
            
            # Generate TTS with timing data
            tts_result = await self._generate_tts(
                api_key,
                subtitle['text'],
                voice_id,
            )
            
            if not tts_result:
                self.log.warning(f"Failed to generate audio for subtitle {i+1}, skipping")
                continue
            
            audio_bytes, actual_duration_seconds = tts_result
            actual_duration_ms = int(actual_duration_seconds * 1000)
            
            audio_segments.append({
                'index': subtitle['index'],
                'start_ms': subtitle['start_ms'],
                'end_ms': subtitle['end_ms'],
                'duration_ms': subtitle['duration_ms'],
                'actual_duration_ms': actual_duration_ms,  # Actual TTS duration from API
                'text': subtitle['text'],
                'speaker': subtitle['speaker'],
                'audio': audio_bytes,
            })
        
        if not audio_segments:
            return "❌ **Error:** Failed to generate any audio segments."
        
        # Synchronize timing and concatenate
        await self.emit_status(
            __event_emitter__, "info", f"🔗 Synchronizing {len(audio_segments)} segments with timing..."
        )
        
        final_audio_id = await self._create_synchronized_audio(
            audio_segments,
            __user__,
            file_record.meta.get("name", "subtitles.srt"),
        )
        
        if not final_audio_id:
            return "❌ **Error:** Failed to create synchronized audio file."
        
        await self.emit_status(
            __event_emitter__, "info", "✅ Audio generation complete!", True
        )
        
        # Build response
        total_duration_sec = subtitles[-1]['end_ms'] / 1000 if subtitles else 0
        
        # Format-specific metadata
        format_emoji = "🎬" if format_type == 'drama_script' else "📝"
        format_label = "Drama Script" if format_type == 'drama_script' else "SRT Subtitles"
        
        # Count unique speakers/characters
        unique_speakers = set(s['speaker'] for s in subtitles if s['speaker'] not in ['__SCENE__', '__DIRECTION__'])
        
        response = f"""# {format_emoji} {format_label} Narration Complete

**Source:** {script_filename}
**Format:** {format_label}
**Entries:** {len(subtitles)}
**Audio Segments:** {len(audio_segments)}
**Characters/Speakers:** {len(unique_speakers)}
**Total Duration:** {int(total_duration_sec // 60)}m {int(total_duration_sec % 60)}s
**Timing Mode:** {self.valves.TIMING_MODE}

## 📥 Download Audio

[**Download Narrated Audio**](/api/v1/files/{final_audio_id}/content)

**File ID:** `{final_audio_id}`

✨ The audio uses natural speech without pitch/speed changes."""
        
        # Add format-specific notes
        if format_type == 'drama_script':
            response += f"""

## 🎭 Drama Script Settings

- **Scene Announcements:** {'✅ Enabled' if self.valves.SCENE_ANNOUNCEMENTS else '❌ Disabled'}
- **Stage Directions:** {'✅ Included' if self.valves.INCLUDE_STAGE_DIRECTIONS else '❌ Excluded'}
- **Default Dialogue Duration:** {self.valves.DEFAULT_DIALOGUE_DURATION}ms
- **Pause Between Lines:** {self.valves.PAUSE_BETWEEN_LINES}ms

**Character Voices:** {len(speaker_voice_map)} mapped, {len(unique_speakers) - len(speaker_voice_map)} using default voice
"""
        else:
            response += "\n\nEach segment is positioned at its subtitle timestamp, creating authentic pauses and flow."
        
        return response

    def _parse_srt(self, srt_content: str) -> List[Dict]:
        """Parse SRT content into structured subtitle entries."""
        subtitles = []
        
        # Split into blocks (separated by blank lines)
        blocks = re.split(r'\n\s*\n', srt_content.strip())
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue
            
            try:
                # Line 1: Index
                index = int(lines[0].strip())
                
                # Line 2: Timestamp (00:00:00,000 --> 00:00:05,000)
                timestamp_match = re.match(
                    r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})',
                    lines[1].strip()
                )
                
                if not timestamp_match:
                    self.log.warning(f"Failed to parse timestamp: {lines[1]}")
                    continue
                
                # Parse start time
                start_h, start_m, start_s, start_ms = map(int, timestamp_match.groups()[:4])
                start_total_ms = (start_h * 3600000) + (start_m * 60000) + (start_s * 1000) + start_ms
                
                # Parse end time
                end_h, end_m, end_s, end_ms = map(int, timestamp_match.groups()[4:])
                end_total_ms = (end_h * 3600000) + (end_m * 60000) + (end_s * 1000) + end_ms
                
                # Line 3+: Text (may span multiple lines)
                text_lines = lines[2:]
                text = ' '.join(line.strip() for line in text_lines if line.strip())
                
                # Detect speaker from text (format: "Speaker: text" or "[Speaker] text")
                speaker = "Narrator"
                speaker_match = re.match(r'^(?:\[([^\]]+)\]|([^:]+):)\s*(.+)$', text)
                if speaker_match:
                    speaker = (speaker_match.group(1) or speaker_match.group(2)).strip()
                    text = speaker_match.group(3).strip()
                
                subtitles.append({
                    'index': index,
                    'start_ms': start_total_ms,
                    'end_ms': end_total_ms,
                    'duration_ms': end_total_ms - start_total_ms,
                    'text': text,
                    'speaker': speaker,
                })
                
            except Exception as e:
                self.log.warning(f"Failed to parse subtitle block: {e}")
                continue
        
        return subtitles

    def _detect_format(self, content: str) -> str:
        """Detect if content is SRT or drama script format.
        
        Returns:
            'srt', 'drama_script', or 'unknown'
        """
        # SRT has timestamp arrows in early content
        if '-->' in content[:1000]:
            return 'srt'
        
        # Drama script indicators:
        # - Scene headers (INT./EXT.)
        # - Numbered dialogue lines
        # - Character names in caps before dialogue
        has_scene_headers = bool(re.search(r'\b(INT\.|EXT\.)', content[:2000]))
        has_numbered_dialogue = bool(re.search(r'^\d+\s+[A-Z\s]{2,}:', content, re.MULTILINE))
        has_character_dialogue = bool(re.search(r'^[A-Z\s]{2,}:\s*[A-Z]', content, re.MULTILINE))
        
        if has_scene_headers and (has_numbered_dialogue or has_character_dialogue):
            return 'drama_script'
        
        return 'unknown'

    def _parse_drama_script(self, script_content: str) -> List[Dict]:
        """Parse drama script into structured dialogue entries.
        
        Extracts dialogue from screenplay format:
        - Detects scene headers (INT./EXT.)
        - Parses numbered dialogue: "1 CHARACTER_NAME: text"
        - Parses un-numbered dialogue: "CHARACTER_NAME: text"
        - Optionally includes stage directions
        - Generates sequential timing
        
        Returns:
            List of dialogue dictionaries with timing and speaker info
        """
        dialogues = []
        current_scene = ""
        current_timestamp = 0  # Sequential timing in ms
        dialogue_duration = self.valves.DEFAULT_DIALOGUE_DURATION
        pause_duration = self.valves.PAUSE_BETWEEN_LINES
        
        lines = script_content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            # Detect scene headers (INT. or EXT.)
            if re.match(r'^(INT\.|EXT\.)', line):
                current_scene = line
                
                # Add scene announcement if enabled
                if self.valves.SCENE_ANNOUNCEMENTS:
                    # Clean up scene header for narration
                    scene_text = re.sub(r'\(.*?\)', '', current_scene)  # Remove parentheses
                    scene_text = re.sub(r'\s+', ' ', scene_text).strip()  # Normalize whitespace
                    
                    dialogues.append({
                        'index': len(dialogues) + 1,
                        'start_ms': current_timestamp,
                        'end_ms': current_timestamp + dialogue_duration,
                        'duration_ms': dialogue_duration,
                        'text': f"Scene: {scene_text}",
                        'speaker': '__SCENE__',  # Special marker for scene announcements
                        'scene': current_scene,
                    })
                    
                    current_timestamp += dialogue_duration + pause_duration
                
                continue
            
            # Parse numbered dialogue: "1 CHARACTER_NAME: text"
            dialogue_match = re.match(r'^\d+\s+([A-Z\s]+):\s*(.+)$', line)
            
            # Also try un-numbered format: "CHARACTER_NAME: text"
            if not dialogue_match:
                dialogue_match = re.match(r'^([A-Z\s]{2,}):\s*(.+)$', line)
            
            if dialogue_match:
                character = dialogue_match.group(1).strip()
                text = dialogue_match.group(2).strip()
                
                # Skip empty dialogue
                if not text:
                    continue
                
                # Handle stage directions in parentheses
                if text.startswith('(') and text.endswith(')'):
                    if self.valves.INCLUDE_STAGE_DIRECTIONS:
                        # Include as stage direction with special speaker
                        dialogues.append({
                            'index': len(dialogues) + 1,
                            'start_ms': current_timestamp,
                            'end_ms': current_timestamp + dialogue_duration,
                            'duration_ms': dialogue_duration,
                            'text': text.strip('()'),
                            'speaker': '__DIRECTION__',  # Special marker for stage directions
                            'scene': current_scene,
                        })
                        current_timestamp += dialogue_duration + pause_duration
                    # Skip if not including stage directions
                    continue
                
                # Estimate duration based on text length
                # Rough estimate: ~150 words per minute = 2.5 words/second = 400ms per word
                word_count = len(text.split())
                estimated_duration = max(dialogue_duration, word_count * 400)
                
                dialogues.append({
                    'index': len(dialogues) + 1,
                    'start_ms': current_timestamp,
                    'end_ms': current_timestamp + estimated_duration,
                    'duration_ms': estimated_duration,
                    'text': text,
                    'speaker': character,
                    'scene': current_scene,
                })
                
                current_timestamp += estimated_duration + pause_duration
                continue
            
            # Handle continuation lines (dialogue that spans multiple lines)
            # These typically don't have a character name prefix
            if dialogues and not line.isupper() and not re.match(r'^\d+\s', line):
                # Check if it looks like dialogue continuation (not a stage direction)
                if not (line.startswith('(') or line.endswith(')')):
                    # Append to previous dialogue
                    prev_dialogue = dialogues[-1]
                    
                    # Only append to actual dialogue, not scene announcements or stage directions
                    if prev_dialogue['speaker'] not in ['__SCENE__', '__DIRECTION__']:
                        prev_dialogue['text'] += ' ' + line
                        
                        # Increase duration for additional text
                        word_count = len(line.split())
                        additional_duration = word_count * 400
                        prev_dialogue['duration_ms'] += additional_duration
                        prev_dialogue['end_ms'] += additional_duration
                        current_timestamp += additional_duration
        
        self.log.info(f"Parsed {len(dialogues)} dialogue entries from drama script")
        if current_scene:
            self.log.info(f"Last scene detected: {current_scene}")
        
        return dialogues

    def _parse_speaker_voice_map(self) -> Dict[str, str]:
        """Parse speaker-to-voice mapping from valve configuration."""
        voice_map = {}
        
        if not self.valves.SPEAKER_VOICE_MAP:
            return voice_map
        
        # Parse format: "Alice:voice_id_1,Bob:voice_id_2"
        for mapping in self.valves.SPEAKER_VOICE_MAP.split(','):
            mapping = mapping.strip()
            if ':' not in mapping:
                continue
            
            speaker, voice_id = mapping.split(':', 1)
            voice_map[speaker.strip()] = voice_id.strip()
            self.log.info(f"Mapped speaker '{speaker.strip()}' to voice '{voice_id.strip()}'")
        
        return voice_map

    async def _generate_tts(
        self,
        api_key: str,
        text: str,
        voice_id: str,
    ) -> Optional[Tuple[bytes, float]]:
        """Generate TTS audio using ElevenLabs API with timing data.
        
        Returns:
            Tuple of (audio_bytes, actual_duration_seconds) or None on error
        """
        url = f"{self.valves.API_BASE_URL}/text-to-speech/{voice_id}/with-timestamps"
        
        payload = {
            "text": text,
            "model_id": self.valves.MODEL_ID,
            "voice_settings": {
                "stability": self.valves.STABILITY,
                "similarity_boost": self.valves.SIMILARITY_BOOST,
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
                        self.log.error(f"ElevenLabs TTS API error {resp.status}: {error_text}")
                        return None
                    
                    response_data = await resp.json()
                    
                    # Decode base64 audio
                    audio_base64 = response_data.get("audio_base64")
                    if not audio_base64:
                        self.log.error("No audio_base64 in response")
                        return None
                    
                    audio_bytes = base64.b64decode(audio_base64)
                    
                    # Get actual duration from timing data
                    alignment = response_data.get("alignment", {})
                    char_end_times = alignment.get("character_end_times_seconds", [])
                    
                    if char_end_times:
                        actual_duration_seconds = char_end_times[-1]  # Last character end time
                    else:
                        # Fallback: estimate from audio size (rough approximation)
                        actual_duration_seconds = len(audio_bytes) / (128 * 1024 / 8)  # 128kbps MP3
                    
                    self.log.info(f"Generated audio: {len(audio_bytes)} bytes, duration: {actual_duration_seconds:.2f}s")
                    return (audio_bytes, actual_duration_seconds)
                    
        except aiohttp.ClientTimeout:
            self.log.error(f"Request timed out after {self.valves.TIMEOUT} seconds")
            return None
        except aiohttp.ClientError as e:
            self.log.error(f"Network error during API request: {type(e).__name__}: {e}")
            return None
        except Exception as e:
            self.log.error(f"Error processing TTS response: {e}")
            return None

    async def _create_synchronized_audio(
        self,
        audio_segments: List[Dict],
        __user__: dict,
        source_filename: str,
    ) -> Optional[str]:
        """Create synchronized audio file with proper timing using pydub/ffmpeg."""
        try:
            # Import pydub for audio manipulation
            from pydub import AudioSegment
            
            # Set ffmpeg path for pydub (from imageio-ffmpeg)
            AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
            
            self.log.info(f"Creating synchronized audio with {len(audio_segments)} segments")
            
            # Create silent audio track for the full duration
            if audio_segments:
                total_duration_ms = max(seg['end_ms'] for seg in audio_segments)
            else:
                return None
            
            # Start with silence
            final_audio = AudioSegment.silent(duration=total_duration_ms)
            
            # Process each segment
            for segment in audio_segments:
                # Load audio from bytes
                temp_file = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                        tmp.write(segment['audio'])
                        temp_file = tmp.name
                    
                    audio_clip = AudioSegment.from_mp3(temp_file)
                    
                    # Use actual duration from ElevenLabs API timing data
                    actual_duration_ms = segment['actual_duration_ms']
                    target_duration_ms = segment['duration_ms']
                    
                    self.log.info(f"Segment {segment['index']}: actual={actual_duration_ms}ms (API), target={target_duration_ms}ms (SRT), diff={actual_duration_ms - target_duration_ms:+d}ms")
                    
                    # Always use natural speed, position at start time
                    # If audio is shorter than target, it will leave silence until next segment
                    # If audio is longer, it will overlap with next segment (natural flow)
                    
                    # Option: Trim if too long (prevents overlap)
                    if self.valves.TIMING_MODE == "trim" and actual_duration_ms > target_duration_ms:
                        audio_clip = audio_clip[:target_duration_ms]
                        self.log.debug(f"Trimmed segment {segment['index']} from {actual_duration_ms}ms to {target_duration_ms}ms")
                    
                    # Overlay at the exact start timestamp
                    # Natural gaps will occur if audio is shorter than allocated time
                    final_audio = final_audio.overlay(audio_clip, position=segment['start_ms'])
                    
                finally:
                    if temp_file and os.path.exists(temp_file):
                        os.unlink(temp_file)
            
            # Export final audio
            output_file = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    output_file = tmp.name
                
                final_audio.export(output_file, format="mp3", bitrate="128k")
                
                # Create filename
                base_name = source_filename.replace(".srt", "")
                timestamp = int(time.time())
                filename = f"{base_name}_narrated_{timestamp}.mp3"
                
                # Upload to Open WebUI storage
                user_id = __user__.get("id") if __user__ else None
                file_id = str(uuid.uuid4())
                
                with open(output_file, "rb") as f:
                    file_data, file_path = Storage.upload_file(
                        f,
                        file_id,
                        {"content_type": "audio/mpeg", "source": "elevenlabs_srt_narrator"}
                    )
                
                # Create file record
                FilesDB.insert_new_file(
                    user_id or "system",
                    FileForm(
                        id=file_id,
                        filename=filename,
                        path=file_path,
                        meta={
                            "name": filename,
                            "content_type": "audio/mpeg",
                            "size": len(file_data),
                            "source": "elevenlabs_srt_narrator",
                            "segments": len(audio_segments),
                            "duration_ms": total_duration_ms,
                        },
                    ),
                )
                
                self.log.info(f"Created synchronized audio file: {file_id}")
                return file_id
                
            finally:
                if output_file and os.path.exists(output_file):
                    os.unlink(output_file)
                    
        except ImportError as ie:
            self.log.error(f"pydub not installed: {ie}")
            self.log.error("If auto-install failed, manually run: pip install pydub")
            return None
        except Exception as e:
            self.log.error(f"Failed to create synchronized audio: {e}", exc_info=True)
            return None
