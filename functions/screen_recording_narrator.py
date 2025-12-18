"""
title: Screen Recording Narrator
author: open-webui
date: 2025-12-18
version: 2.87
license: MIT
description: Generate voiceover narration for silent screen recordings using Gemini video understanding and ElevenLabs TTS. Analyzes video content, creates time-stamped script, and synthesizes synchronized audio voiceover.
requirements: google-genai, aiohttp, cryptography, pydantic, imageio-ffmpeg, pydub, requests
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
import shutil
import aiohttp
import imageio_ffmpeg
import requests
from pathlib import Path
from typing import Optional, Callable, Awaitable, Any, Dict, List, Tuple
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from cryptography.fernet import Fernet, InvalidToken
from pydantic_core import core_schema

# Google Gemini imports
from google import genai
from google.genai import types

# Open WebUI imports
from open_webui.models.files import Files as FilesDB, FileForm
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
        # Gemini Authentication Options
        USE_VERTEX_AI: bool = Field(
            default=False,
            description="Use Vertex AI instead of Google AI API. Requires service account or ADC.",
        )
        GEMINI_API_KEY: EncryptedStr = Field(
            default="",
            description="Google AI API key for Gemini video analysis (only used if USE_VERTEX_AI is False)",
        )
        PROJECT_ID: str = Field(
            default="",
            description="Google Cloud project ID (for Vertex AI). Defaults to GOOGLE_CLOUD_PROJECT env var.",
        )
        LOCATION: str = Field(
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
        
        # Gemini Model Configuration
        GEMINI_MODEL: str = Field(
            default="gemini-2.0-flash-exp",
            description="Gemini model for video analysis (gemini-2.0-flash-exp recommended for video)",
        )
        
        # ElevenLabs API Configuration
        ELEVEN_API_KEY: EncryptedStr = Field(
            default="",
            description="ElevenLabs API key (xi-api-key). Get it from https://elevenlabs.io/",
        )
        ELEVEN_API_BASE_URL: str = Field(
            default="https://api.elevenlabs.io/v1",
            description="Base URL for ElevenLabs API",
        )
        VOICE_ID: str = Field(
            default="21m00Tcm4TlvDq8ikWAM",
            description="Default ElevenLabs voice ID (Rachel). This will be overridden if user specifies a voice in their message.",
        )
        AVAILABLE_VOICES: str = Field(
            default="Rachel:21m00Tcm4TlvDq8ikWAM,Adam:pNInz6obpgDQGcFmaJgB,Antoni:ErXwobaYiN019PkySvjV,Arnold:VR6AewLTigWG4xSOukaG,Callum:N2lVS1w4EtoT3dr4eOWO,Charlie:IKne3meq5aSn9XLyUdCD,Charlotte:XB0fDUnXU5powFXDhCwa,Clyde:2EiwWnXFnvU5JabPnv8n,Daniel:onwK4e9ZLuTAKqWW03F9,Dave:CYw3kZ02Hs0563khs1Fj,Domi:AZnzlk1XvdvUeBnXmlld,Dorothy:ThT5KcBeYPX3keUQqHPh,Drew:29vD33N1CtxCmqQRPOHJ,Emily:LcfcDJNUP1GQjkzn1xUU,Ethan:g5CIjZEefAph4nQFvHAz,Fin:D38z5RcWu1voky8WS1ja,Freya:jsCqWAovK2LkecY7zXl4,George:JBFqnCBsd6RMkjVDRZzb,Gigi:jBpfuIE2acCO8z3wKNLl,Giovanni:zcAOhNBS3c14rBihAFp1,Glinda:z9fAnlkpzviPz146aGWa,Grace:oWAxZDx7w5VEj9dCyTzz,Harry:SOYHLrjzK2X1ezoPC6cr,James:ZQe5CZNOzWyzPSCn5a3c,Jeremy:bVMeCyTHy58xNoL34h3p,Jessie:t0jbNlBVZ17f02VDIeMI,Joseph:Zlb1dXrM653N07WRdFW3,Josh:TxGEqnHWrfWFTfGW9XjX,Liam:TX3LPaxmHKxFdv7VOQHJ,Matilda:XrExE9yKIg1WjnnlVkGX,Michael:flq6f7yk4E4fJM5XTYuZ,Lily:pFZP5JQG7iQjIQuC4Bku,Bill:pqHfZKP75CvOlQylNhV4,Brian:nPczCjzI2devNBz1zQrb,Sarah:EXAVITQu4vr4xnSDxMaL,Serena:pMsXgVXv3BLzUgSXRplE",
            description="🎤 Default ElevenLabs voices (see https://elevenlabs.io/docs/voices#default). Format: 'Name:voice_id'. Users can request by name (e.g., 'use British accent', 'female voice', 'Adam voice'). Popular: Rachel (F-US), Adam (M-US), Antoni (M-US), Charlotte (F-UK), Sarah (F-AUS).",
        )
        TTS_MODEL_ID: str = Field(
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
        
        # Script Generation Settings
        SCRIPT_INTERVAL_SECONDS: int = Field(
            default=10,
            description="Generate narration segments every N seconds of video",
        )
        NARRATION_STYLE: str = Field(
            default="professional",
            description="Narration style: 'professional', 'casual', 'tutorial', 'educational'",
        )
        INCLUDE_TECHNICAL_DETAILS: bool = Field(
            default=True,
            description="Include technical details in narration (UI elements, actions, etc.)",
        )
        GENERATE_VOICE_BY_DEFAULT: bool = Field(
            default=False,
            description="Generate voice synthesis by default. If False, only generates script unless user explicitly requests voice.",
        )
        VOICE_KEYWORDS: str = Field(
            default="with voice,synthesize,generate audio,add voice,voiceover,narrate audio,speak it",
            description="Comma-separated keywords that trigger voice synthesis when detected in user prompt",
        )
        
        # Output Settings
        GENERATE_MERGED_VIDEO: bool = Field(
            default=True,
            description="Generate final video with voiceover merged (requires voice synthesis). If False, only outputs audio file.",
        )
        GENERATE_SUBTITLES: bool = Field(
            default=True,
            description="Generate SRT subtitle file and embed in video. Subtitles show the narration text with timestamps.",
        )
        ENABLE_SCRIPT_PREVIEW: bool = Field(
            default=False,
            description="Show generated script for review/editing before TTS synthesis. User can approve or provide edited version.",
        )
        
        # Processing Settings
        TIMEOUT: int = Field(
            default=180,
            description="Max seconds to wait for API calls",
        )
        EMIT_INTERVAL: float = Field(
            default=1.0,
            description="Interval in seconds between status emissions",
        )

    def __init__(self):
        self.name = "Screen Recording Narrator"
        self.valves = self.Valves()
        self.log = logging.getLogger("screen_recording_narrator")
        self.log.setLevel(logging.INFO)
        self.last_emit_time = 0
        self.cached_voices = None
        self.cached_voice_details = None
        self.voices_cache_time = 0
        self.VOICE_CACHE_TTL = 3600

    async def _fetch_elevenlabs_voices(self, include_details: bool = False) -> Dict[str, Tuple[str, str]]:
        """Fetch available voices from ElevenLabs API.
        
        Args:
            include_details: If True, also cache full voice metadata
        
        Returns:
            Dict mapping lowercase voice name to (display_name, voice_id)
        """
        try:
            # Check cache first (1 hour TTL)
            current_time = time.time()
            if self.cached_voices and (current_time - self.voices_cache_time) < self.VOICE_CACHE_TTL:
                # If details requested but not cached, need to re-fetch
                if include_details and not self.cached_voice_details:
                    self.log.info("Cache hit for voices, but details not cached - refetching with details")
                else:
                    return self.cached_voices
            
            eleven_key = self.valves.ELEVEN_API_KEY.get_decrypted()
            if not eleven_key:
                self.log.warning("No ElevenLabs API key available for voice fetching")
                return self._parse_static_voices()
            
            url = f"{self.valves.ELEVEN_API_BASE_URL.replace('/v1', '')}/v2/voices"
            headers = {"xi-api-key": eleven_key}
            params = {
                "page_size": 100,
                "voice_type": "default"
            }
            
            self.log.info(f"Fetching voices from ElevenLabs API: {url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        voices = {}
                        voice_details = {}
                        
                        for voice in data.get("voices", []):
                            name = voice.get("name", "")
                            voice_id = voice.get("voice_id", "")
                            labels = voice.get("labels", {})
                            description = voice.get("description", "")
                            is_legacy = voice.get("is_legacy", False)
                            
                            # Skip legacy voices
                            if is_legacy:
                                continue
                            
                            if name and voice_id:
                                voices[name.lower()] = (name, voice_id)
                                
                                # Store full voice metadata if requested
                                if include_details:
                                    voice_details[name] = {
                                        "voice_id": voice_id,
                                        "gender": labels.get("gender", ""),
                                        "accent": labels.get("accent", ""),
                                        "age": labels.get("age", ""),
                                        "description": description,
                                        "use_case": labels.get("use_case", "")
                                    }
                                
                                # Add gender/accent variants for better matching
                                gender = labels.get("gender", "").lower()
                                accent = labels.get("accent", "").lower()
                                
                                if gender and accent:
                                    key = f"{accent} {gender}"
                                    if key.lower() not in voices:
                                        voices[key.lower()] = (name, voice_id)
                        
                        self.log.info(f"Fetched {len(voices)} voices from ElevenLabs API")
                        self.cached_voices = voices
                        if include_details:
                            self.cached_voice_details = voice_details
                        self.voices_cache_time = current_time
                        return voices
                    else:
                        self.log.warning(f"Failed to fetch voices from API: {response.status}")
                        return self._parse_static_voices()
        
        except Exception as e:
            self.log.error(f"Error fetching voices from API: {e}")
            return self._parse_static_voices()
    
    def _parse_static_voices(self) -> Dict[str, Tuple[str, str]]:
        """Parse voices from static configuration as fallback.
        
        Returns:
            Dict mapping lowercase voice name to (display_name, voice_id)
        """
        available_voices = {}
        
        for voice_mapping in self.valves.AVAILABLE_VOICES.split(','):
            voice_mapping = voice_mapping.strip()
            if ':' in voice_mapping:
                name, voice_id = voice_mapping.split(':', 1)
                available_voices[name.strip().lower()] = (name.strip(), voice_id.strip())
        
        return available_voices
    
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
        """Main pipeline: Generate voiceover narration for screen recording"""
        self.log.info("=== Screen Recording Narrator Pipeline Called ===")
        
        await self.emit_status(
            __event_emitter__, "info", "🎬 Initializing screen recording narrator..."
        )
        
        # Determine if voice synthesis is requested and which voice to use
        messages = body.get("messages", [])
        user_message = messages[-1].get("content", "") if messages else ""
        should_generate_voice = self._should_generate_voice(user_message)
        selected_voice_name, selected_voice_id = await self._detect_voice_preference(user_message)
        
        # Detect narrative tense and perspective from user prompt
        narrative_tense, narrative_perspective = self._detect_narrative_style(user_message)
        
        self.log.info(f"Voice synthesis: {'ENABLED' if should_generate_voice else 'DISABLED (script only)'}")
        if should_generate_voice and selected_voice_name:
            self.log.info(f"Selected voice: {selected_voice_name} ({selected_voice_id})")
        self.log.info(f"Narrative style: {narrative_perspective}, {narrative_tense} tense")
        
        # Validate authentication (API key or Vertex AI)
        if self.valves.USE_VERTEX_AI:
            # Check Vertex AI configuration
            project_id = self.valves.PROJECT_ID or os.getenv("GOOGLE_CLOUD_PROJECT", "")
            if not project_id:
                return "❌ **Error:** Google Cloud Project ID not configured for Vertex AI.\n\n**Solutions:**\n- Set `PROJECT_ID` in pipeline settings, or\n- Set the `GOOGLE_CLOUD_PROJECT` environment variable"
            
            # Check if service account credentials are provided
            if not self.valves.SERVICE_ACCOUNT_JSON and not self.valves.SERVICE_ACCOUNT_PATH:
                return "❌ **Error:** Service account credentials not configured for Vertex AI.\n\n**Solutions:**\n- Provide `SERVICE_ACCOUNT_JSON` in pipeline settings, or\n- Provide `SERVICE_ACCOUNT_PATH` in pipeline settings, or\n- Set `USE_VERTEX_AI=False` to use API key authentication instead"
        else:
            # Check API key configuration
            gemini_key = self.valves.GEMINI_API_KEY.get_decrypted()
            if not gemini_key:
                return "❌ **Error:** Gemini API key not configured.\n\n**Solution:** Set your `GEMINI_API_KEY` in the pipeline settings.\n\nGet your API key from: https://aistudio.google.com/apikey"
        
        # Only validate ElevenLabs key if voice synthesis is requested
        eleven_key = None
        if should_generate_voice:
            eleven_key = self.valves.ELEVEN_API_KEY.get_decrypted()
            if not eleven_key:
                return "❌ **Error:** ElevenLabs API key not configured.\n\n**Solution:** Set your `ELEVEN_API_KEY` in the pipeline settings.\n\nGet your API key from: https://elevenlabs.io/\n\n💡 **Tip:** If you only want the script without voice synthesis, just ask for the script without voice-related keywords."
        
        # Find video file from __files__ parameter
        video_file_id = None
        video_filename = None
        video_path = None
        
        if __files__:
            self.log.info(f"Checking {len(__files__)} file(s) for video")
            for file_item in __files__:
                file_id = file_item.get("id")
                if file_id:
                    file_record = FilesDB.get_file_by_id(file_id)
                    if file_record:
                        content_type = file_record.meta.get("content_type", "")
                        
                        if content_type.startswith("video/"):
                            video_file_id = file_id
                            video_filename = file_record.meta.get("name", "unknown.mp4")
                            video_path = file_record.path
                            self.log.info(f"Found video file: {video_filename} (ID: {file_id})")
                            break
        
        if not video_file_id or not video_path:
            return "❌ **Error:** No video file found.\n\n**Solution:** Please upload a screen recording video (MP4, WebM, MOV, etc.)\n\n**Example:** Upload `screen-recording.mp4`"
        
        # Check if user provided edited script (skip Gemini if so)
        script_segments = None
        edited_script_content = None
        
        if self.valves.ENABLE_SCRIPT_PREVIEW:
            # First check current message for edited script (multiple formats)
            msg_lower = user_message.lower()
            
            # Check for [MM:SS] timestamp format
            if re.search(r'\[(\d{1,2}):(\d{2})\]', user_message):
                edited_script_content = user_message
                self.log.info("Detecting edited script in current message (timestamp format)")
            # Check for segment+timing format
            else:
                has_segments = "segment 1" in msg_lower or "segment 2" in msg_lower
                has_timing = "timing:" in msg_lower
                if has_segments and has_timing:
                    edited_script_content = user_message
                    self.log.info("Detecting edited script in current message (segment format)")
            
            # If not in current message but voice synthesis requested, check message history
            if not edited_script_content and should_generate_voice:
                edited_script_content = self._find_edited_script_in_history(messages)
                if edited_script_content:
                    self.log.info("Found edited script in previous message")
            
            # Parse the edited script if found
            if edited_script_content:
                await self.emit_status(
                    __event_emitter__, "info", "📝 Using your edited script..."
                )
                script_segments = self._parse_edited_script_standalone(edited_script_content)
                if script_segments:
                    self.log.info(f"Parsed {len(script_segments)} edited segments")
        
        # Step 1: Analyze video with Gemini to generate time-stamped script (if not already provided)
        if not script_segments:
            await self.emit_status(
                __event_emitter__, "info", "🔍 Analyzing video content with Gemini..."
            )
            
            # Pass API key (will be None if using Vertex AI)
            gemini_key = None if self.valves.USE_VERTEX_AI else self.valves.GEMINI_API_KEY.get_decrypted()
            
            script_segments = await self._generate_script_from_video(
                gemini_key,
                video_path,
                __event_emitter__,
                narrative_tense,
                narrative_perspective,
            )
            
            if not script_segments:
                return "❌ **Error:** Failed to generate script from video analysis.\n\nPlease check the video file and try again."
        
        self.log.info(f"Using {len(script_segments)} script segments")
        
        # If voice synthesis not requested, return script only
        if not should_generate_voice:
            await self.emit_status(
                __event_emitter__, "info", "✅ Script generation complete!", True
            )
            return await self._format_script_only_response(script_segments, video_filename, narrative_tense, narrative_perspective)
        
        # If script preview is enabled, show script for review/editing
        if self.valves.ENABLE_SCRIPT_PREVIEW:
            # Check if this is the initial generation or a follow-up with approval/edits
            if not self._is_script_approval_or_edit(user_message):
                await self.emit_status(
                    __event_emitter__, "info", "📋 Script ready for review", True
                )
                return self._format_script_preview(script_segments, video_filename, selected_voice_name)
            
            # User has provided approval or edits, parse edited script if present
            edited_segments = self._parse_edited_script(user_message, script_segments)
            if edited_segments:
                script_segments = edited_segments
                self.log.info(f"Using edited script with {len(script_segments)} segments")
        
        # Step 2: Generate TTS for each script segment
        await self.emit_status(
            __event_emitter__, "info", f"🎤 Generating voiceover for {len(script_segments)} segments..."
        )
        
        audio_segments = []
        for i, segment in enumerate(script_segments):
            await self.emit_status(
                __event_emitter__, "info",
                f"🎤 Generating audio {i+1}/{len(script_segments)}: {segment['text'][:50]}..."
            )
            
            tts_result = await self._generate_tts(
                eleven_key,
                segment['text'],
                selected_voice_id,  # Use selected voice
            )
            
            if not tts_result:
                self.log.warning(f"Failed to generate audio for segment {i+1}, skipping")
                continue
            
            audio_bytes, actual_duration_seconds = tts_result
            actual_duration_ms = int(actual_duration_seconds * 1000)
            
            audio_segments.append({
                'index': segment['index'],
                'start_ms': segment['start_ms'],
                'end_ms': segment['end_ms'],
                'duration_ms': segment['duration_ms'],
                'actual_duration_ms': actual_duration_ms,
                'text': segment['text'],
                'audio': audio_bytes,
            })
        
        if not audio_segments:
            return "❌ **Error:** Failed to generate any audio segments."
        
        # Step 3: Synchronize audio segments with video timeline
        await self.emit_status(
            __event_emitter__, "info", f"🔗 Synchronizing {len(audio_segments)} segments with video timeline..."
        )
        
        final_audio_id = await self._create_synchronized_audio(
            audio_segments,
            __user__,
            video_filename,
        )
        
        if not final_audio_id:
            return "❌ **Error:** Failed to create synchronized audio file."
        
        # Step 4: Optionally merge audio with original video
        merged_video_id = None
        if self.valves.GENERATE_MERGED_VIDEO:
            await self.emit_status(
                __event_emitter__, "info", "🎬 Merging voiceover with original video..."
            )
            
            # Generate subtitle file if enabled
            subtitle_file_id = None
            subtitle_path = None
            if self.valves.GENERATE_SUBTITLES:
                await self.emit_status(
                    __event_emitter__, "info", "📝 Generating subtitle file..."
                )
                subtitle_file_id, subtitle_path = await self._generate_subtitle_file(
                    script_segments,
                    video_filename,
                    user_id=body.get("user", {}).get("id") if body.get("user") else None
                )
            
            # Get audio file for muxing
            audio_file = FilesDB.get_file_by_id(final_audio_id)
            if audio_file:
                with open(audio_file.path, "rb") as f:
                    audio_bytes = f.read()
                
                # Use video_path as source
                merged_video_id = await self._merge_video_audio(
                    video_path,
                    audio_bytes,
                    subtitle_path,
                    __user__.get("id") if __user__ else None,
                    __event_emitter__,
                )
                
                if merged_video_id:
                    self.log.info(f"Successfully merged video with voiceover: {merged_video_id}")
                else:
                    self.log.warning("Failed to merge video with audio, will only return audio file")
        
        await self.emit_status(
            __event_emitter__, "info", "✅ Voiceover generation complete!", True
        )
        
        # Build response
        total_duration_sec = script_segments[-1]['end_ms'] / 1000 if script_segments else 0
        
        response = f"""# 🎬 Screen Recording Voiceover Complete

**Source Video:** {video_filename}
**Script Segments:** {len(script_segments)}
**Audio Segments:** {len(audio_segments)}
**Total Duration:** {int(total_duration_sec // 60)}m {int(total_duration_sec % 60)}s
**Narration Style:** {self.valves.NARRATION_STYLE}
**Narrative Style:** {narrative_perspective.title()}, {narrative_tense} tense
**Voice:** {selected_voice_name}

"""
        
        # Add merged video section if available
        if merged_video_id:
            response += f"""## 🎥 Download Final Video (with Voiceover)

[**Download Video with Narration**](/api/v1/files/{merged_video_id}/content)"""
            if subtitle_file_id:
                response += " *(includes embedded subtitles)*"
            response += "\n\n"
        
        # Add subtitle download if available
        if subtitle_file_id:
            response += f"""## � Download Subtitle File

[**Download SRT Subtitles**](/api/v1/files/{subtitle_file_id}/content)

"""
        
        # Add audio download
        response += f"""## 📥 Download Voiceover Audio

[**Download Voiceover Track**](/api/v1/files/{final_audio_id}/content)

**File ID:** `{final_audio_id}`

## 🎯 How to Use

1. Download the voiceover audio file above
2. Import your screen recording and the voiceover audio into your video editor
3. The audio is already synchronized to the video timeline
4. Adjust volume levels as needed
5. Export your final video with narration

## 📝 Generated Script

"""
        
        # Add script preview
        for segment in script_segments[:10]:  # Show first 10 segments
            timestamp = self._format_timestamp(segment['start_ms'])
            response += f"\n**[{timestamp}]** {segment['text']}"
        
        if len(script_segments) > 10:
            response += f"\n\n*...and {len(script_segments) - 10} more segments*"
        
        response += "\n\n💡 **Tip:** This includes voice synthesis. For script-only output, omit voice-related keywords from your request."
        
        return response
    
    def _format_script_preview(
        self,
        script_segments: List[Dict],
        video_filename: str,
        voice_name: str,
    ) -> str:
        """Format script preview for user review before TTS generation."""
        response = f"""# 📋 Script Preview for {video_filename}

**Voice:** {voice_name}  
**Segments:** {len(script_segments)}

---

## ✅ Next Steps

**To proceed with TTS generation:**

1. **Approve as-is:** Reply with `approve`, `looks good`, or `continue`
2. **Edit script:** Copy the script below, make your changes, and paste it back with your voice request
3. **Cancel:** Say `cancel` to stop

---

## 📝 Editable Script

Copy this entire code block to edit:

```
"""
        
        # Display all script segments with timestamps in code block
        for i, segment in enumerate(script_segments, 1):
            start_time = self._format_timestamp(segment['start_ms'])
            end_time = self._format_timestamp(segment['end_ms'])
            response += f"### Segment {i}\n"
            response += f"**Timing:** {start_time} → {end_time}\n\n"
            response += f"{segment['text']}\n\n"
            response += "---\n\n"
        
        response += """```

### 💡 Editing Instructions

You can now edit the script seamlessly in **two ways**:

**Option 1: Edit and request voice in same message**
```
### Segment 1
**Timing:** 00:00 → 00:10

Your edited text here

---

Generate with British voice
```

**Option 2: Edit first, then request voice separately** ✨
1. Copy and paste the edited script above
2. Send it
3. In your **next message**, say `"Generate with British voice"`
4. System will automatically use your edited script from the previous message!

### Step-by-Step Workflow

1. **Copy** the script above (click the copy button)
2. **Edit** any segment text you want to change
3. **Paste** the edited script and send it
4. **Reply** with your voice request: `"Make a voiceover with British accent"`
5. System automatically finds your edits and uses them for TTS

💡 **The system checks your last 5 messages for edited scripts** - so you can edit now and request voice later!
"""
        
        return response
    
    def _find_edited_script_in_history(self, messages: List[Dict]) -> Optional[str]:
        """Search recent user messages for edited script content."""
        # Look at last 5 user messages in reverse order
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        
        for msg in reversed(user_messages[-5:]):
            content = msg.get("content", "")
            content_lower = content.lower()
            
            # Check for script-only format: [00:00] text
            if re.search(r'\[(\d{1,2}):(\d{2})\]', content):
                self.log.info("Found edited script in previous message (timestamp format)")
                return content
            
            # Check for segment+timing format
            has_segments = "segment 1" in content_lower or "segment 2" in content_lower
            has_timing = "timing:" in content_lower
            if has_segments and has_timing:
                self.log.info("Found edited script in previous message (segment format)")
                return content
        
        return None
    
    def _is_script_approval_or_edit(self, user_message: str) -> bool:
        """Check if user message indicates approval or contains edited script."""
        message_lower = user_message.lower().strip()
        
        # Check for approval keywords
        approval_keywords = [
            "approve", "approved", "looks good", "looks great", "perfect",
            "continue", "proceed", "go ahead", "generate", "yes", "ok", "okay"
        ]
        
        for keyword in approval_keywords:
            if keyword in message_lower:
                return True
        
        # Check for voice generation requests (indicates approval to proceed)
        voice_keywords = ["voiceover", "voice over", "tts", "synthesize", "narration", "narrate"]
        for keyword in voice_keywords:
            if keyword in message_lower:
                return True
        
        # Check if message contains script structure (edited script) - both formats
        if "### segment" in message_lower or "**timing:**" in message_lower:
            return True
        if "segment 1" in message_lower and "timing:" in message_lower:
            return True
        
        return False
    
    def _parse_edited_script_standalone(self, user_message: str) -> Optional[List[Dict]]:
        """Parse edited script from user message, extracting both timing and text."""
        try:
            # Check for [MM:SS] timestamp format (script-only output)
            timestamp_lines = re.findall(r'\[(\d{1,2}):(\d{2})\]\s*(.+)', user_message)
            if timestamp_lines:
                edited_segments = []
                for i, (minutes, seconds, text) in enumerate(timestamp_lines):
                    start_ms = (int(minutes) * 60 + int(seconds)) * 1000
                    # Calculate end time (next segment's start or +10 seconds for last)
                    if i < len(timestamp_lines) - 1:
                        next_min, next_sec = timestamp_lines[i + 1][:2]
                        end_ms = (int(next_min) * 60 + int(next_sec)) * 1000
                    else:
                        end_ms = start_ms + 10000  # Default 10 seconds for last segment
                    
                    duration_ms = end_ms - start_ms
                    edited_segments.append({
                        'index': i + 1,
                        'text': text.strip(),
                        'start_ms': start_ms,
                        'end_ms': end_ms,
                        'duration_ms': duration_ms,
                    })
                
                if edited_segments:
                    self.log.info(f"Parsed {len(edited_segments)} edited segments from timestamp format")
                    return edited_segments
            
            # Check for segment markers (both markdown and plain text)
            if "segment" not in user_message.lower():
                return None
            
            edited_segments = []
            lines = user_message.split('\n')
            
            current_segment_num = None
            current_timing = None
            current_text_lines = []
            
            for line in lines:
                line_stripped = line.strip()
                line_lower = line_stripped.lower()
                
                # Check for segment header (both "### Segment 1" and "Segment 1")
                if line_lower.startswith("### segment") or (line_lower.startswith("segment") and len(line_stripped.split()) == 2):
                    # Save previous segment if exists
                    if current_segment_num and current_timing and current_text_lines:
                        start_ms, end_ms = current_timing
                        duration_ms = end_ms - start_ms
                        edited_segments.append({
                            'index': current_segment_num,
                            'text': ' '.join(current_text_lines).strip(),
                            'start_ms': start_ms,
                            'end_ms': end_ms,
                            'duration_ms': duration_ms,
                        })
                        current_text_lines = []
                    
                    # Extract segment number
                    try:
                        current_segment_num = int(line_stripped.split()[-1])
                        current_timing = None
                    except (ValueError, IndexError):
                        current_segment_num = None
                
                # Parse timing line (both "**Timing:**" and "Timing:")
                elif line_lower.startswith("**timing:**") or line_lower.startswith("timing:"):
                    # Extract timestamps: "00:00 → 00:10" or "00:00:00 → 00:00:10"
                    timing_match = re.search(r'(\d{1,2}):(\d{2})(?::(\d{2}))?\s*[→-]\s*(\d{1,2}):(\d{2})(?::(\d{2}))?', line_stripped)
                    if timing_match:
                        groups = timing_match.groups()
                        # Start time
                        start_min = int(groups[0])
                        start_sec = int(groups[1])
                        start_ms = (start_min * 60 + start_sec) * 1000
                        
                        # End time
                        end_min = int(groups[3])
                        end_sec = int(groups[4])
                        end_ms = (end_min * 60 + end_sec) * 1000
                        
                        current_timing = (start_ms, end_ms)
                
                # Skip separator lines
                elif line_stripped == "---":
                    continue
                
                # Collect text lines for current segment
                elif current_segment_num and current_timing and line_stripped:
                    current_text_lines.append(line_stripped)
            
            # Save last segment
            if current_segment_num and current_timing and current_text_lines:
                start_ms, end_ms = current_timing
                duration_ms = end_ms - start_ms
                edited_segments.append({
                    'index': current_segment_num,
                    'text': ' '.join(current_text_lines).strip(),
                    'start_ms': start_ms,
                    'end_ms': end_ms,
                    'duration_ms': duration_ms,
                })
            
            # Validate we got segments
            if not edited_segments:
                self.log.warning("No valid segments found in edited script")
                return None
            
            self.log.info(f"Parsed {len(edited_segments)} edited segments from segment format")
            return edited_segments
            
        except Exception as e:
            self.log.error(f"Failed to parse edited script: {e}", exc_info=True)
            return None
    
    def _parse_edited_script(
        self,
        user_message: str,
        original_segments: List[Dict],
    ) -> Optional[List[Dict]]:
        """Parse edited script from user message and return updated segments."""
        try:
            # Look for segment markers in the message
            if "### segment" not in user_message.lower():
                # No edits detected, return None to use original
                return None
            
            edited_segments = []
            lines = user_message.split('\n')
            
            current_segment = None
            current_text_lines = []
            
            for line in lines:
                line_stripped = line.strip()
                
                # Check for segment header
                if line_stripped.lower().startswith("### segment"):
                    # Save previous segment if exists
                    if current_segment and current_text_lines:
                        current_segment['text'] = ' '.join(current_text_lines).strip()
                        edited_segments.append(current_segment)
                        current_text_lines = []
                    
                    # Extract segment number
                    try:
                        segment_num = int(line_stripped.split()[-1])
                        # Use original timing from corresponding segment
                        if 1 <= segment_num <= len(original_segments):
                            current_segment = original_segments[segment_num - 1].copy()
                        else:
                            current_segment = None
                    except (ValueError, IndexError):
                        current_segment = None
                
                # Skip timing lines
                elif line_stripped.lower().startswith("**timing:**"):
                    continue
                
                # Skip separator lines
                elif line_stripped == "---":
                    continue
                
                # Collect text lines for current segment
                elif current_segment and line_stripped:
                    current_text_lines.append(line_stripped)
            
            # Save last segment
            if current_segment and current_text_lines:
                current_segment['text'] = ' '.join(current_text_lines).strip()
                edited_segments.append(current_segment)
            
            # Validate we got segments
            if not edited_segments:
                self.log.warning("No valid segments found in edited script")
                return None
            
            self.log.info(f"Parsed {len(edited_segments)} edited segments from user message")
            return edited_segments
            
        except Exception as e:
            self.log.error(f"Failed to parse edited script: {e}")
            return None
    
    def _should_generate_voice(self, user_message: str) -> bool:
        """Determine if voice synthesis should be generated based on user prompt."""
        if self.valves.GENERATE_VOICE_BY_DEFAULT:
            return True
        
        # Check for voice-related keywords in user message
        message_lower = user_message.lower()
        keywords = [kw.strip().lower() for kw in self.valves.VOICE_KEYWORDS.split(',')]
        
        for keyword in keywords:
            if keyword and keyword in message_lower:
                self.log.info(f"Voice synthesis triggered by keyword: '{keyword}'")
                return True
        
        return False
    
    async def _detect_voice_preference(self, user_message: str) -> Tuple[str, str]:
        """Detect voice preference from user message and return (voice_name, voice_id).
        
        Returns:
            Tuple of (voice_name, voice_id). Defaults to configured VOICE_ID if no preference detected.
        """
        # Fetch available voices from API (with caching and fallback to static config)
        available_voices = await self._fetch_elevenlabs_voices()
        default_voice_name = "Default Voice"
        
        self.log.info(f"Voice detection: {len(available_voices)} voices available")
        self.log.debug(f"Available voice keys: {list(available_voices.keys())}")
        
        # Track if default voice is in the list
        for voice_key, (voice_name, voice_id) in available_voices.items():
            if voice_id == self.valves.VOICE_ID:
                default_voice_name = voice_name
                break
        
        # Check user message for voice preferences
        # Strip markdown formatting (**, *, _, etc.) that might wrap voice names
        import re
        cleaned_message = re.sub(r'[*_`]', '', user_message)
        message_lower = cleaned_message.lower()
        self.log.info(f"Checking message for voice preference: '{user_message}' (cleaned: '{cleaned_message}')")
        
        # Direct voice name matches - check if voice name appears as a word in message
        for voice_key, (voice_name, voice_id) in available_voices.items():
            # Use word boundary matching to avoid partial matches
            if re.search(r'\b' + re.escape(voice_key) + r'\b', message_lower):
                self.log.info(f"Detected voice preference: {voice_name} (matched '{voice_key}')")
                return (voice_name, voice_id)
        
        # Check if user explicitly specified a voice name that doesn't exist
        # Patterns: "using X", "with X voice", "X voice"
        explicit_voice_patterns = [
            r'using\s+(\w+)',
            r'with\s+(\w+)\s+voice',
            r'(\w+)\s+voice'
        ]
        
        for pattern in explicit_voice_patterns:
            match = re.search(pattern, message_lower)
            if match:
                requested_voice = match.group(1)
                # Skip common words that aren't voice names
                skip_words = {'voice', 'with', 'using', 'british', 'american', 'female', 'male', 'accent'}
                if requested_voice not in skip_words:
                    self.log.warning(f"Voice '{requested_voice}' not found in available voices. Using default voice: {default_voice_name}")
                    return (default_voice_name, self.valves.VOICE_ID)
        
        # Combined gender + accent detection (prioritize specific combinations)
        has_female = 'female' in message_lower
        has_male = 'male' in message_lower and not has_female
        
        # British + gender
        if any(word in message_lower for word in ['british', 'uk', 'british accent', 'english accent']):
            if has_female:
                for voice_key, (voice_name, voice_id) in available_voices.items():
                    if 'british' in voice_key and 'female' in voice_key:
                        self.log.info(f"Detected female British accent request: {voice_name}")
                        return (voice_name, voice_id)
            elif has_male:
                for voice_key, (voice_name, voice_id) in available_voices.items():
                    if 'british' in voice_key and 'male' in voice_key and 'female' not in voice_key:
                        self.log.info(f"Detected male British accent request: {voice_name}")
                        return (voice_name, voice_id)
            else:
                # No gender specified, return first British voice
                for voice_key, (voice_name, voice_id) in available_voices.items():
                    if 'british' in voice_key:
                        self.log.info(f"Detected British accent request: {voice_name}")
                        return (voice_name, voice_id)
        
        # American + gender
        if any(word in message_lower for word in ['american', 'us', 'american accent']):
            if has_female:
                for voice_key, (voice_name, voice_id) in available_voices.items():
                    if 'american' in voice_key and 'female' in voice_key:
                        self.log.info(f"Detected female American accent request: {voice_name}")
                        return (voice_name, voice_id)
            elif has_male:
                for voice_key, (voice_name, voice_id) in available_voices.items():
                    if 'american' in voice_key and 'male' in voice_key and 'female' not in voice_key:
                        self.log.info(f"Detected male American accent request: {voice_name}")
                        return (voice_name, voice_id)
            else:
                # No gender specified, return first American voice
                for voice_key, (voice_name, voice_id) in available_voices.items():
                    if 'american' in voice_key:
                        self.log.info(f"Detected American accent request: {voice_name}")
                        return (voice_name, voice_id)
        
        # Australian + gender
        if any(word in message_lower for word in ['australian', 'aussie', 'australian accent']):
            if has_female:
                for voice_key, (voice_name, voice_id) in available_voices.items():
                    if 'australian' in voice_key and 'female' in voice_key:
                        self.log.info(f"Detected female Australian accent request: {voice_name}")
                        return (voice_name, voice_id)
            elif has_male:
                for voice_key, (voice_name, voice_id) in available_voices.items():
                    if 'australian' in voice_key and 'male' in voice_key and 'female' not in voice_key:
                        self.log.info(f"Detected male Australian accent request: {voice_name}")
                        return (voice_name, voice_id)
            else:
                # No gender specified, return first Australian voice
                for voice_key, (voice_name, voice_id) in available_voices.items():
                    if 'australian' in voice_key:
                        self.log.info(f"Detected Australian accent request: {voice_name}")
                        return (voice_name, voice_id)
        
        # Gender-only detection (fallback when no accent specified)
        if has_male:
            for voice_key, (voice_name, voice_id) in available_voices.items():
                if 'male' in voice_key and 'female' not in voice_key:
                    self.log.info(f"Detected male voice request: {voice_name}")
                    return (voice_name, voice_id)
        
        if has_female:
            for voice_key, (voice_name, voice_id) in available_voices.items():
                if 'female' in voice_key:
                    self.log.info(f"Detected female voice request: {voice_name}")
                    return (voice_name, voice_id)
        
        # Default to configured voice
        return (default_voice_name, self.valves.VOICE_ID)
    
    def _detect_narrative_style(self, user_message: str) -> Tuple[str, str]:
        """Detect narrative tense and perspective from user message.
        
        Returns:
            Tuple of (tense, perspective):
            - tense: 'past' or 'present'
            - perspective: 'first-person' or 'third-person'
        """
        message_lower = user_message.lower()
        
        # Detect tense
        tense = "present"  # default
        past_indicators = [
            "i did", "i clicked", "i selected", "i opened", "i went",
            "past tense", "what i did", "how i",
            "walked through", "showed how i"
        ]
        for indicator in past_indicators:
            if indicator in message_lower:
                tense = "past"
                self.log.info(f"Detected past tense from: '{indicator}'")
                break
        
        # Detect perspective
        perspective = "first-person"  # default
        third_person_indicators = [
            "the user", "they click", "we see", "the system",
            "third person", "third-person", "user actions",
            "narrator", "observer"
        ]
        for indicator in third_person_indicators:
            if indicator in message_lower:
                perspective = "third-person"
                self.log.info(f"Detected third-person perspective from: '{indicator}'")
                break
        
        return (tense, perspective)
    
    async def _format_available_voices(self) -> str:
        """Format available voices for display to user with details from API."""
        # Fetch voices with full details
        await self._fetch_elevenlabs_voices(include_details=True)
        
        # Check if we have detailed voice metadata from API
        if self.cached_voice_details:
            # Build markdown table
            table = "| Voice | Gender | Accent | Age | Description |\n"
            table += "|-------|--------|--------|-----|-------------|\n"
            
            for name in sorted(self.cached_voice_details.keys()):
                details = self.cached_voice_details[name]
                gender = details.get("gender", "").title() or "-"
                accent = details.get("accent", "") or "-"
                age = details.get("age", "") or "-"
                desc = details.get("description", "") or "-"
                
                # Escape pipe characters in description
                desc = desc.replace("|", "\\|")
                
                table += f"| **{name}** | {gender} | {accent} | {age} | {desc} |\n"
            
            return table
        
        # Fallback to simple table from static config
        table = "| Voice |\n"
        table += "|-------|\n"
        
        for voice_mapping in self.valves.AVAILABLE_VOICES.split(','):
            voice_mapping = voice_mapping.strip()
            if ':' in voice_mapping:
                name, _ = voice_mapping.split(':', 1)
                table += f"| **{name.strip()}** |\n"
        
        return table
    
    async def _format_script_only_response(self, script_segments: List[Dict], video_filename: str, narrative_tense: str, narrative_perspective: str) -> str:
        """Format response for script-only mode (no voice synthesis)."""
        total_duration_sec = script_segments[-1]['end_ms'] / 1000 if script_segments else 0
        
        response = f"""# 📝 Screen Recording Script Generated

**Source Video:** {video_filename}
**Script Segments:** {len(script_segments)}
**Total Duration:** {int(total_duration_sec // 60)}m {int(total_duration_sec % 60)}s
**Narration Style:** {self.valves.NARRATION_STYLE}
**Narrative Style:** {narrative_perspective.title()}, {narrative_tense} tense

## 📋 Generated Script

"""
        
        # Add all script segments
        for segment in script_segments:
            timestamp = self._format_timestamp(segment['start_ms'])
            response += f"\n**[{timestamp}]** {segment['text']}"
        
        response += f"""\n\n---

## 🎤 Want Voice Synthesis?

To generate voiceover audio from this script, ask again with one of these phrases:
- "Generate the script **with voice**"
- "Create narration and **synthesize audio**"
- "Make a **voiceover** for this video"

### Available Voices:
{await self._format_available_voices()}

You can specify a voice like:
- "Generate with voice using **British accent**"
- "Create voiceover with **American Male** voice"

You can also specify narrative style:
- "Generate script in **first person past tense** (I did this, I clicked that)"
- "Use **first person present tense** (I click here, I select this)"
- "Use **third person** (the user clicks, we see)"

💡 **Current mode:** Script-only (voice synthesis disabled by default)
💡 **To enable by default:** Set `GENERATE_VOICE_BY_DEFAULT = True` in pipeline settings
"""
        
        return response

    async def _generate_script_from_video(
        self,
        api_key: Optional[str],
        video_path: str,
        event_emitter: Optional[Callable[[dict], Awaitable[None]]],
        narrative_tense: str = "present",
        narrative_perspective: str = "third-person",
    ) -> List[Dict]:
        """Analyze video with Gemini and generate time-stamped narration script."""
        # Initialize credential cleanup variables
        temp_creds_path = None
        old_creds = None
        
        try:
            # Initialize Gemini client with appropriate authentication
            if self.valves.USE_VERTEX_AI:
                project_id = self.valves.PROJECT_ID or os.getenv(
                    "GOOGLE_CLOUD_PROJECT", ""
                )
                location = self.valves.LOCATION or os.getenv(
                    "GOOGLE_CLOUD_LOCATION", "us-central1"
                )

                self.log.info(
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
                self.log.info("Using API key authentication")
                client = genai.Client(api_key=api_key)
            
            # Upload video file to Gemini
            await self.emit_status(
                event_emitter, "info", "📤 Uploading video to Gemini..."
            )
            
            with open(video_path, "rb") as f:
                video_data = f.read()
            
            # Detect mime type
            import mimetypes
            mime_type, _ = mimetypes.guess_type(video_path)
            if not mime_type or not mime_type.startswith("video/"):
                mime_type = "video/mp4"
            
            # Create prompt for time-stamped narration
            interval = self.valves.SCRIPT_INTERVAL_SECONDS
            style_instructions = {
                "professional": "Use clear, professional language suitable for business presentations.",
                "casual": "Use friendly, conversational language as if explaining to a friend.",
                "tutorial": "Use instructional language with step-by-step guidance.",
                "educational": "Use clear explanations with emphasis on learning and understanding.",
            }
            
            style = style_instructions.get(
                self.valves.NARRATION_STYLE,
                style_instructions["professional"]
            )
            
            technical_detail = "Include specific UI elements, button names, menu items, and actions." if self.valves.INCLUDE_TECHNICAL_DETAILS else "Focus on high-level actions and outcomes."
            
            # Build narrative style instructions
            if narrative_perspective == "first-person":
                if narrative_tense == "past":
                    perspective_instruction = "Use first-person past tense (e.g., 'I clicked on the menu', 'I opened the settings')"
                    example = "'I clicked on the File menu and selected New Document'"
                else:
                    perspective_instruction = "Use first-person present tense (e.g., 'I click on the menu', 'I open the settings')"
                    example = "'I click on the File menu and select New Document'"
            else:  # third-person
                if narrative_tense == "past":
                    perspective_instruction = "Use third-person past tense (e.g., 'The user clicked on the menu', 'We opened the settings')"
                    example = "'The user clicked on the File menu and selected New Document'"
                else:
                    perspective_instruction = "Use third-person present tense (e.g., 'The user clicks on the menu', 'We see the settings open')"
                    example = "'The user clicks on the File menu and selects New Document'"
            
            # Create prompt for script generation
            prompt = f"""Analyse this screen recording video and generate a narration script with ACCURATE TIMING.

CRITICAL: You must analyze the ACTUAL video content to determine when actions happen and how long they take. Do NOT use fixed intervals.

Create a time-stamped voiceover script with the following requirements:

1. **TIMING ACCURACY (MOST IMPORTANT):**
   - Watch the video carefully and identify when each distinct action or scene change occurs
   - Set timestamps based on when things ACTUALLY HAPPEN in the video, not at fixed intervals
   - Each segment's timestamp should mark when that action/scene begins
   - Consider scene transitions, UI changes, window switches, loading states, etc.
   - Aim for segments approximately every {self.valves.SCRIPT_INTERVAL_SECONDS} seconds, but ADJUST based on actual content
   - If an action takes 5 seconds, the next timestamp should be ~5 seconds later, not exactly 10

2. **Narration style:** {self.valves.NARRATION_STYLE}

3. **Narrative perspective:** {perspective_instruction}
   - Example: {example}

4. **Content guidelines:**
   - Each segment describes what's happening at that specific moment
   - Keep segments concise (1-2 sentences)
   - Focus on user actions and important visual elements
   - Use clear, easy-to-understand language
   - {style}
   - {technical_detail}

5. **Timing verification:**
   - Ensure no overlapping segments
   - Each segment should have enough time for narration (minimum ~5 seconds)
   - Final timestamp should not exceed video duration

OUTPUT FORMAT (JSON):
{{
  "segments": [
    {{"timestamp": "00:00", "text": "The screen shows..."}},
    {{"timestamp": "00:08", "text": "Next, we see..."}},
    {{"timestamp": "00:15", "text": "The user navigates to..."}}
  ]
}}

IMPORTANT: Watch the video frame-by-frame and set timestamps based on ACTUAL timing, not theoretical intervals."""
            
            # Create Gemini content
            parts = [
                types.Part.from_bytes(data=video_data, mime_type=mime_type),
                types.Part.from_text(text=prompt)
            ]
            
            contents = [types.Content(role="user", parts=parts)]
            
            await self.emit_status(
                event_emitter, "info", "🤖 Generating script with Gemini AI..."
            )
            
            # Generate script
            response = client.models.generate_content(
                model=self.valves.GEMINI_MODEL,
                contents=contents,
            )
            
            # Extract and parse JSON response
            response_text = response.text.strip()
            
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            
            # Parse JSON
            try:
                script_data = json.loads(response_text)
                segments_raw = script_data.get("segments", [])
            except json.JSONDecodeError:
                # Fallback: try to parse as plain text with timestamps
                self.log.warning("Failed to parse JSON, falling back to text parsing")
                segments_raw = self._parse_script_from_text(response_text)
            
            # Convert to structured format with millisecond timestamps
            script_segments = []
            for i, seg in enumerate(segments_raw):
                timestamp_str = seg.get("timestamp", f"{i * interval:02d}:00")
                text = seg.get("text", "").strip()
                
                if not text:
                    continue
                
                # Parse timestamp to milliseconds
                start_ms = self._parse_timestamp_to_ms(timestamp_str)
                
                # Estimate duration (until next segment or default interval)
                if i + 1 < len(segments_raw):
                    next_timestamp = segments_raw[i + 1].get("timestamp", "")
                    end_ms = self._parse_timestamp_to_ms(next_timestamp)
                else:
                    end_ms = start_ms + (interval * 1000)
                
                script_segments.append({
                    'index': i + 1,
                    'start_ms': start_ms,
                    'end_ms': end_ms,
                    'duration_ms': end_ms - start_ms,
                    'text': text,
                })
            
            self.log.info(f"Generated {len(script_segments)} script segments from Gemini")
            return script_segments
            
        except Exception as e:
            self.log.error(f"Failed to generate script from video: {e}", exc_info=True)
            return []
        finally:
            # Clean up temp credentials file and restore original environment
            if temp_creds_path and os.path.exists(temp_creds_path):
                try:
                    os.unlink(temp_creds_path)
                except Exception as e:
                    self.log.warning(f"Failed to delete temp credentials file: {e}")
            
            # Restore original credentials environment variable
            if old_creds is not None:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = old_creds
            elif temp_creds_path:
                # If there was no original value, remove the env var we set
                os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

    def _parse_script_from_text(self, text: str) -> List[Dict]:
        """Parse script from plain text format (fallback)."""
        segments = []
        
        # Look for timestamp patterns like [00:10], (00:10), 00:10 -, etc.
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to extract timestamp and text
            match = re.match(r'[\[\(]?(\d{1,2}:\d{2})[\]\)]?[\s\-:]*(.+)', line)
            if match:
                timestamp = match.group(1)
                narration = match.group(2).strip()
                segments.append({"timestamp": timestamp, "text": narration})
        
        return segments

    def _parse_timestamp_to_ms(self, timestamp_str: str) -> int:
        """Parse timestamp string (MM:SS or HH:MM:SS) to milliseconds."""
        try:
            parts = timestamp_str.strip().split(':')
            if len(parts) == 2:
                # MM:SS format
                minutes, seconds = map(int, parts)
                return (minutes * 60 + seconds) * 1000
            elif len(parts) == 3:
                # HH:MM:SS format
                hours, minutes, seconds = map(int, parts)
                return (hours * 3600 + minutes * 60 + seconds) * 1000
            else:
                return 0
        except:
            return 0

    def _format_timestamp(self, milliseconds: int) -> str:
        """Format milliseconds to MM:SS timestamp."""
        total_seconds = milliseconds // 1000
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    async def _generate_tts(
        self,
        api_key: str,
        text: str,
        voice_id: str,
    ) -> Optional[Tuple[bytes, float]]:
        """Generate TTS audio using ElevenLabs API with timing data."""
        url = f"{self.valves.ELEVEN_API_BASE_URL}/text-to-speech/{voice_id}/with-timestamps"
        
        payload = {
            "text": text,
            "model_id": self.valves.TTS_MODEL_ID,
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
                        actual_duration_seconds = char_end_times[-1]
                    else:
                        # Fallback: estimate from audio size
                        actual_duration_seconds = len(audio_bytes) / (128 * 1024 / 8)
                    
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
    
    async def _generate_subtitle_file(
        self,
        script_segments: List[Dict],
        video_filename: str,
        user_id: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Generate SRT subtitle file and upload to storage.
        
        Returns:
            Tuple of (file_id, temp_file_path) or (None, None) on error
        """
        try:
            # Generate SRT content
            srt_content = self._format_srt_subtitles(script_segments)
            
            # Create temp file
            temp_srt = tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                suffix='.srt',
                delete=False
            )
            temp_srt.write(srt_content)
            temp_srt.close()
            
            # Upload to storage
            timestamp = int(time.time())
            base_name = Path(video_filename).stem
            filename = f"{base_name}_subtitles_{timestamp}.srt"
            
            with open(temp_srt.name, 'rb') as f:
                file_data, file_path = Storage.upload_file(
                    f,
                    filename,
                    {"content_type": "application/x-subrip", "source": "screen_recording_narrator"}
                )
            
            # Create database record
            file_id = str(uuid.uuid4())
            file_size = os.path.getsize(temp_srt.name)
            
            record = FilesDB.insert_new_file(
                user_id or "system",
                FileForm(
                    id=file_id,
                    filename=filename,
                    path=file_path,
                    meta={
                        "name": filename,
                        "content_type": "application/x-subrip",
                        "size": file_size,
                        "source": "screen_recording_narrator",
                    },
                ),
            )
            
            self.log.info(f"Generated subtitle file: {filename} ({file_size} bytes)")
            return (record.id if record else None, temp_srt.name)
            
        except Exception as e:
            self.log.error(f"Failed to generate subtitle file: {e}", exc_info=True)
            return (None, None)
    
    def _format_srt_subtitles(self, script_segments: List[Dict]) -> str:
        """Format script segments as SRT subtitle content."""
        srt_lines = []
        
        for i, segment in enumerate(script_segments, 1):
            # SRT format:
            # 1
            # 00:00:00,000 --> 00:00:10,000
            # Subtitle text
            # (blank line)
            
            start_time = self._format_srt_timestamp(segment['start_ms'])
            end_time = self._format_srt_timestamp(segment['end_ms'])
            
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(segment['text'])
            srt_lines.append("")  # Blank line separator
        
        return "\n".join(srt_lines)
    
    def _format_srt_timestamp(self, milliseconds: int) -> str:
        """Format milliseconds to SRT timestamp format (HH:MM:SS,mmm)."""
        total_seconds = milliseconds // 1000
        ms = milliseconds % 1000
        
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"
    
    async def _get_media_duration(self, file_path: str) -> Optional[float]:
        """Get duration of media file in seconds using ffprobe."""
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            # ffprobe is usually in the same directory as ffmpeg
            ffprobe_exe = ffmpeg_exe.replace('ffmpeg', 'ffprobe')
            
            # If ffprobe doesn't exist in the same path, try using ffmpeg with -i
            if not os.path.exists(ffprobe_exe):
                # Use ffmpeg to get duration from stderr
                cmd = [ffmpeg_exe, "-i", file_path]
                result = await asyncio.to_thread(
                    subprocess.run,
                    cmd,
                    capture_output=True,
                )
                stderr = result.stderr.decode(errors="ignore")
                
                # Parse duration from ffmpeg output
                # Format: Duration: 00:01:23.45
                match = re.search(r'Duration: (\d{2}):(\d{2}):(\d{2})\.(\d{2})', stderr)
                if match:
                    hours, minutes, seconds, centiseconds = map(int, match.groups())
                    duration = hours * 3600 + minutes * 60 + seconds + centiseconds / 100.0
                    return duration
            else:
                # Use ffprobe for more accurate duration
                cmd = [
                    ffprobe_exe,
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    file_path
                ]
                result = await asyncio.to_thread(
                    subprocess.run,
                    cmd,
                    capture_output=True,
                )
                if result.returncode == 0:
                    duration_str = result.stdout.decode().strip()
                    return float(duration_str)
            
            return None
        except Exception as e:
            self.log.warning(f"Failed to get media duration: {e}")
            return None
    
    async def _extend_video_with_freeze_frame(
        self,
        video_path: str,
        target_duration: float,
    ) -> Optional[str]:
        """Extend video by freezing the last frame to reach target duration."""
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            
            # Get current video duration
            current_duration = await self._get_media_duration(video_path)
            if not current_duration or target_duration <= current_duration:
                return None
            
            freeze_duration = target_duration - current_duration
            
            # Create temp output file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tf:
                tmp_output = tf.name
            
            # FFmpeg command to extend video with frozen last frame
            # Strategy: Use tpad filter to pad the end with the last frame
            cmd = [
                ffmpeg_exe,
                "-y",
                "-i", video_path,
                "-vf", f"tpad=stop_mode=clone:stop_duration={freeze_duration}",
                "-c:a", "copy",  # Copy audio if exists
                tmp_output,
            ]
            
            self.log.debug(f"Extending video with command: {' '.join(cmd)}")
            
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
            )
            
            if result.returncode != 0:
                stderr = result.stderr.decode(errors="ignore") if result.stderr else ""
                self.log.error(f"Failed to extend video: {stderr[:500]}")
                if os.path.exists(tmp_output):
                    os.unlink(tmp_output)
                return None
            
            if not os.path.exists(tmp_output) or os.path.getsize(tmp_output) == 0:
                self.log.error("Extended video file is empty or missing")
                if os.path.exists(tmp_output):
                    os.unlink(tmp_output)
                return None
            
            return tmp_output
            
        except Exception as e:
            self.log.error(f"Error extending video: {e}", exc_info=True)
            return None

    async def _merge_video_audio(
        self,
        video_path: str,
        audio_bytes: bytes,
        subtitle_path: Optional[str] = None,
        user_id: Optional[str] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> Optional[str]:
        """Merge audio with original video using ffmpeg."""
        tmp_audio = None
        tmp_out = None
        tmp_extended_video = None
        
        try:
            # Write audio to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as ta:
                ta.write(audio_bytes)
                tmp_audio = ta.name
            
            # Get video and audio durations
            video_duration = await self._get_media_duration(video_path)
            audio_duration = await self._get_media_duration(tmp_audio)
            
            # Check if we need to extend the video
            video_to_use = video_path
            if audio_duration and video_duration and audio_duration > video_duration:
                self.log.info(f"Audio ({audio_duration:.2f}s) is longer than video ({video_duration:.2f}s), extending video...")
                await self.emit_status(
                    __event_emitter__, 
                    "info", 
                    f"🎬 Audio ({audio_duration:.2f}s) is longer than video ({video_duration:.2f}s), extending video..."
                )
                target_duration = audio_duration + 1.0  # Add 1 second buffer
                tmp_extended_video = await self._extend_video_with_freeze_frame(
                    video_path,
                    target_duration
                )
                if tmp_extended_video:
                    video_to_use = tmp_extended_video
                    self.log.info(f"Extended video to {target_duration:.2f}s")
                    await self.emit_status(
                        __event_emitter__, 
                        "info", 
                        f"✅ Video extended to {target_duration:.2f}s with freeze frame"
                    )
                else:
                    self.log.warning("Failed to extend video, using original")
            
            # Create temp output file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as to:
                tmp_out = to.name
            
            # Build ffmpeg command
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            self.log.info(f"Merging video with audio using ffmpeg: {ffmpeg_exe}")
            
            cmd = [
                ffmpeg_exe,
                "-y",  # Overwrite output file
                "-i", video_to_use,  # Input video (original or extended)
                "-i", tmp_audio,   # Input audio
            ]
            
            # Add subtitle input if available
            if subtitle_path and os.path.exists(subtitle_path):
                cmd.extend(["-i", subtitle_path])
                # Map video, audio, and subtitle streams
                cmd.extend([
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-map", "2:s:0",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-c:s", "mov_text",  # Subtitle codec for MP4
                    "-b:a", "192k",
                    "-metadata:s:s:0", "language=eng",
                    "-metadata:s:s:0", "title=Narration",
                    "-shortest",
                    tmp_out,
                ])
            else:
                # No subtitles, just video and audio
                cmd.extend([
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-shortest",
                    tmp_out,
                ])
            
            self.log.debug(f"FFmpeg command: {' '.join(cmd)}")
            
            # Run ffmpeg
            completed = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
            )
            
            # Check for errors
            if completed.returncode != 0:
                stderr = completed.stderr.decode(errors="ignore") if completed.stderr else ""
                self.log.error(f"FFmpeg merge failed: {stderr[:500]}")
                return None
            
            if not os.path.exists(tmp_out) or os.path.getsize(tmp_out) == 0:
                self.log.error("FFmpeg produced empty output file")
                return None
            
            # Upload merged video to storage
            video_basename = Path(video_path).stem
            timestamp = int(time.time())
            filename = f"{video_basename}_with_voiceover_{timestamp}.mp4"
            
            file_id = str(uuid.uuid4())
            
            with open(tmp_out, "rb") as f:
                file_data, file_path = Storage.upload_file(
                    f,
                    filename,
                    {"content_type": "video/mp4", "source": "screen_recording_narrator_merge"}
                )
            
            # Create file record
            record = FilesDB.insert_new_file(
                user_id or "system",
                FileForm(
                    id=file_id,
                    filename=filename,
                    path=file_path,
                    meta={
                        "name": filename,
                        "content_type": "video/mp4",
                        "size": os.path.getsize(tmp_out),
                        "source": "screen_recording_narrator_merge",
                        "has_subtitles": subtitle_path is not None,
                    },
                ),
            )
            
            self.log.info(f"Created merged video file: {file_id}")
            return file_id
            
        except Exception as e:
            self.log.error(f"Failed to merge video with audio: {e}", exc_info=True)
            return None
        finally:
            # Cleanup temp files
            for tmp_file in [tmp_audio, tmp_out, tmp_extended_video]:
                if tmp_file and os.path.exists(tmp_file):
                    try:
                        os.unlink(tmp_file)
                    except Exception as e:
                        self.log.warning(f"Failed to delete temp file {tmp_file}: {e}")
    
    async def _create_synchronized_audio(
        self,
        audio_segments: List[Dict],
        __user__: dict,
        source_filename: str,
    ) -> Optional[str]:
        """Create synchronized audio file with proper timing using pydub/ffmpeg."""
        try:
            from pydub import AudioSegment
            
            # Set ffmpeg path for pydub
            AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
            
            self.log.info(f"Creating synchronized audio with {len(audio_segments)} segments")
            
            # Create silent audio track for the full duration
            if audio_segments:
                total_duration_ms = max(seg['end_ms'] for seg in audio_segments)
            else:
                return None
            
            # Start with silence
            final_audio = AudioSegment.silent(duration=total_duration_ms)
            
            # Track the end position of the last audio to prevent overlaps
            last_audio_end_ms = 0
            
            # Process each segment
            for segment in audio_segments:
                temp_file = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                        tmp.write(segment['audio'])
                        temp_file = tmp.name
                    
                    audio_clip = AudioSegment.from_mp3(temp_file)
                    actual_audio_length_ms = len(audio_clip)
                    
                    # Calculate placement position to prevent overlaps
                    intended_start_ms = segment['start_ms']
                    actual_start_ms = max(intended_start_ms, last_audio_end_ms)
                    
                    # Log if we're delaying to prevent overlap
                    if actual_start_ms > intended_start_ms:
                        delay_ms = actual_start_ms - intended_start_ms
                        self.log.info(
                            f"Segment {segment['index']}: Delaying by {delay_ms}ms to prevent overlap "
                            f"(intended: {intended_start_ms}ms, actual: {actual_start_ms}ms)"
                        )
                    
                    # Overlay at the calculated position
                    final_audio = final_audio.overlay(audio_clip, position=actual_start_ms)
                    
                    # Update the end position tracker
                    last_audio_end_ms = actual_start_ms + actual_audio_length_ms
                    
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
                base_name = source_filename.rsplit('.', 1)[0]
                timestamp = int(time.time())
                filename = f"{base_name}_voiceover_{timestamp}.mp3"
                
                # Upload to Open WebUI storage
                user_id = __user__.get("id") if __user__ else None
                file_id = str(uuid.uuid4())
                
                with open(output_file, "rb") as f:
                    file_data, file_path = Storage.upload_file(
                        f,
                        file_id,
                        {"content_type": "audio/mpeg", "source": "screen_recording_narrator"}
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
                            "source": "screen_recording_narrator",
                            "segments": len(audio_segments),
                            "duration_ms": total_duration_ms,
                        },
                    ),
                )
                
                self.log.info(f"Created synchronized voiceover audio file: {file_id}")
                return file_id
                
            finally:
                if output_file and os.path.exists(output_file):
                    os.unlink(output_file)
                    
        except ImportError as ie:
            self.log.error(f"pydub not installed: {ie}")
            return None
        except Exception as e:
            self.log.error(f"Failed to create synchronized audio: {e}", exc_info=True)
            return None
