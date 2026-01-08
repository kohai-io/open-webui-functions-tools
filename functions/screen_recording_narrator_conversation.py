"""
title: Screen Recording Narrator (Conversational)
author: open-webui
date: 2025-12-19
version: 1.3.2
license: MIT
description: Agentic conversational interface for screen recording narration using Gemini function calling. Chat naturally to generate scripts, select voices, and create voiceovers. Features Whisper-powered accurate subtitle generation with word-level timestamps.
requirements: google-genai, aiohttp, cryptography, pydantic, imageio-ffmpeg, pydub, requests, openai-whisper
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
            description="Google AI API key for Gemini (only used if USE_VERTEX_AI is False)",
        )
        PROJECT_ID: str = Field(
            default="",
            description="Google Cloud project ID (for Vertex AI).",
        )
        LOCATION: str = Field(
            default="us-central1",
            description="Google Cloud location/region (for Vertex AI).",
        )
        SERVICE_ACCOUNT_JSON: EncryptedStr = Field(
            default="",
            description="Service account JSON key content (for Vertex AI authentication).",
        )
        SERVICE_ACCOUNT_PATH: str = Field(
            default="",
            description="Path to service account JSON file (alternative to SERVICE_ACCOUNT_JSON)",
        )
        
        # Gemini Model Configuration
        GEMINI_MODEL: str = Field(
            default="gemini-2.0-flash-exp",
            description="Gemini model for orchestration and video analysis",
        )
        
        # ElevenLabs API Configuration
        ELEVEN_API_KEY: EncryptedStr = Field(
            default="",
            description="ElevenLabs API key (xi-api-key)",
        )
        ELEVEN_API_BASE_URL: str = Field(
            default="https://api.elevenlabs.io/v1",
            description="Base URL for ElevenLabs API",
        )
        DEFAULT_VOICE_ID: str = Field(
            default="CwhRBWXzGAHq8TQ4Fs17",
            description="Default voice ID (Roger)",
        )
        TTS_MODEL_ID: str = Field(
            default="eleven_multilingual_v2",
            description="TTS model to use",
        )
        STABILITY: float = Field(default=0.6, description="Voice stability (0.0-1.0)")
        SIMILARITY_BOOST: float = Field(default=0.75, description="Similarity boost (0.0-1.0)")
        
        # Script Generation Settings
        SCRIPT_INTERVAL_SECONDS: int = Field(
            default=10,
            description="Generate narration segments every N seconds",
        )
        NARRATION_STYLE: str = Field(
            default="professional",
            description="Default narration style",
        )
        INCLUDE_TECHNICAL_DETAILS: bool = Field(
            default=True,
            description="Include technical details in narration (UI elements, actions, etc.)",
        )
        TIMEOUT: int = Field(
            default=180,
            description="Max seconds to wait for API calls",
        )
        GENERATE_SUBTITLES: bool = Field(
            default=True,
            description="Generate SRT subtitle file and embed in video",
        )
        
        # Whisper Transcription for Accurate Subtitles
        WHISPER_MODE: str = Field(
            default="openai",
            description="Transcription mode for subtitles: 'openai' (API), 'local' (whisper package), or 'openai-compatible' (custom endpoint)",
        )
        OPENAI_API_KEY_WHISPER: EncryptedStr = Field(
            default="",
            description="OpenAI API key for Whisper API (can be same as OPENAI_API_KEY if empty, uses OPENAI_API_KEY from valves)",
        )
        OPENAI_API_BASE_URL_WHISPER: str = Field(
            default="https://api.openai.com/v1",
            description="OpenAI API base URL for Whisper (for 'openai' or 'openai-compatible' modes)",
        )
        WHISPER_MODEL: str = Field(
            default="whisper-1",
            description="Whisper model: 'whisper-1' (OpenAI) or 'tiny/base/small/medium/large' (local)",
        )
        WHISPER_LANGUAGE: Optional[str] = Field(
            default=None,
            description="Language code (e.g., 'en', 'es', 'fr'). None = auto-detect",
        )
        
        # Subtitle Quality Settings
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
        
        # Video Output Settings
        VIDEO_END_BUFFER: float = Field(
            default=1.0,
            description="Buffer duration (seconds) to add after audio/narration ends. Prevents abrupt cutoff.",
        )

    def __init__(self):
        self.name = "Screen Recording Narrator (Conversational)"
        self.valves = self.Valves()
        self.log = logging.getLogger("screen_recording_narrator_conv")
        self.log.setLevel(logging.INFO)
        
        # Session state
        self.script_cache = None
        self.selected_voice_id = None
        self.selected_voice_name = None
        self.video_file_id = None
        self.cached_voices = None
        self.cached_voice_details = None
        self.voices_cache_time = 0
        self.VOICE_CACHE_TTL = 3600

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __event_call__: Optional[Callable[[dict], Awaitable[dict]]] = None,
        __files__: Optional[list] = None,
    ) -> str:
        """Agentic pipeline with LLM-driven orchestration."""
        
        self.log.info("=== Conversational Narrator Pipeline Called ===")
        
        try:
            # Get user message and files
            messages = body.get("messages", [])
            user_message = messages[-1].get("content", "") if messages else ""
            
            # Check for video file
            if __files__:
                for file_item in __files__:
                    file_id = file_item.get("id")
                    if file_id:
                        file_record = FilesDB.get_file_by_id(file_id)
                        if file_record and "video" in file_record.meta.get("content_type", ""):
                            self.video_file_id = file_id
                            self.log.info(f"Video file detected: {file_record.filename}")
                            break
            
            # Fetch available voices
            await self._fetch_elevenlabs_voices(include_details=True)
            
            # Build conversation history
            conversation_history = self._build_conversation_history(messages[:-1])
            
            # Create Gemini client with function calling
            gemini_client = await self._create_gemini_client()
            function_declarations = await self._get_function_declarations()
            
            # System instruction for the agent
            system_instruction = await self._get_system_instruction()
            
            # Call Gemini with function calling enabled
            self.log.info("Calling Gemini agent for intent understanding")
            
            response = await gemini_client.aio.models.generate_content(
                model=self.valves.GEMINI_MODEL,
                contents=conversation_history + [types.Content(role="user", parts=[types.Part.from_text(text=user_message)])],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=[types.Tool(function_declarations=function_declarations)],
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(
                            mode="AUTO"
                        )
                    ),
                    temperature=0.3
                )
            )
            
            # Process response
            if not response.candidates:
                return "I encountered an error processing your request. Please try again."
            
            candidate = response.candidates[0]
            
            # Check for function calls
            function_calls = []
            text_response = ""
            
            for part in candidate.content.parts:
                if part.function_call:
                    function_calls.append(part.function_call)
                elif part.text:
                    text_response += part.text
            
            # Execute function calls if any
            if function_calls:
                self.log.info(f"Gemini requested {len(function_calls)} function calls")
                
                function_responses = []
                last_function_name = None
                last_function_result = None
                
                for func_call in function_calls:
                    result = await self._execute_function_call(
                        func_call,
                        __event_emitter__,
                        __user__
                    )
                    last_function_name = func_call.name
                    last_function_result = result
                    
                    function_responses.append(
                        types.FunctionResponse(
                            name=func_call.name,
                            response={"result": result}
                        )
                    )
                
                # Get final response from Gemini after function execution
                self.log.info("Requesting final conversational response from Gemini...")
                final_response = await gemini_client.aio.models.generate_content(
                    model=self.valves.GEMINI_MODEL,
                    contents=conversation_history + [
                        types.Content(role="user", parts=[types.Part.from_text(text=user_message)]),
                        candidate.content,
                        types.Content(role="user", parts=function_responses)
                    ],
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.7
                    )
                )
                
                self.log.info("Final response received from Gemini")
                
                if final_response.candidates:
                    self.log.info(f"Response has {len(final_response.candidates)} candidates")
                    candidate = final_response.candidates[0]
                    self.log.info(f"Candidate has {len(candidate.content.parts)} parts")
                    
                    for i, part in enumerate(candidate.content.parts):
                        self.log.info(f"Part {i}: has text={hasattr(part, 'text')}, has function_call={hasattr(part, 'function_call')}")
                        if hasattr(part, 'text') and part.text:
                            self.log.info(f"Part {i} text: {part.text[:200]}...")
                    
                    if candidate.content.parts and hasattr(candidate.content.parts[0], 'text') and candidate.content.parts[0].text:
                        response_text = candidate.content.parts[0].text
                        self.log.info(f"Returning response: {response_text[:100]}...")
                        return response_text
                
                self.log.warning(f"No text in final response, formatting {last_function_name} result directly")
                
                # If generate_voiceover was called, emit the video file for inline rendering
                if last_function_name == "generate_voiceover" and last_function_result.get("success"):
                    video_id = last_function_result.get("video_file_id")
                    if video_id and __event_emitter__:
                        try:
                            await __event_emitter__({
                                "type": "message",
                                "data": {
                                    "content": f"📹 **Narrated Video Ready!**\n\n[Download Video with Narration](/api/v1/files/{video_id}/content)"
                                }
                            })
                            # Also emit the file for inline rendering
                            await __event_emitter__({
                                "type": "file",
                                "data": {
                                    "file_id": video_id
                                }
                            })
                            self.log.info(f"Emitted video file {video_id} for inline rendering")
                        except Exception as e:
                            self.log.warning(f"Failed to emit video file: {e}")
                
                # Format response based on which function was called
                return self._format_function_result(last_function_name, last_function_result)
            
            # No function calls, return text response
            return text_response if text_response else "I'm ready to help you create narration for your screen recording. What would you like me to do?"
            
        except Exception as e:
            self.log.error(f"Pipeline error: {e}", exc_info=True)
            return f"❌ **Error:** {str(e)}\n\nPlease check your configuration and try again."

    async def _get_function_declarations(self) -> List[types.FunctionDeclaration]:
        """Define available functions for Gemini to call."""
        
        # Get available voices for enum
        voice_names = []
        if self.cached_voice_details:
            voice_names = sorted(self.cached_voice_details.keys())[:20]  # Limit to 20 for performance
        
        return [
            types.FunctionDeclaration(
                name="generate_script",
                description="Generate a time-stamped narration script from the uploaded video. Call this when user wants to create a script or narration.",
                parameters={
                    "type": "object",
                    "properties": {
                        "narrative_style": {
                            "type": "string",
                            "enum": ["first_person_present", "first_person_past", "third_person"],
                            "description": "Narrative perspective: first_person_present='I click', first_person_past='I clicked', third_person='The user clicks'"
                        },
                        "tone": {
                            "type": "string",
                            "enum": ["professional", "casual", "tutorial", "educational"],
                            "description": "Tone of the narration"
                        }
                    },
                    "required": []
                }
            ),
            types.FunctionDeclaration(
                name="generate_voiceover",
                description="Generate voiceover audio using the script. Call this when user wants audio/voice synthesis. Requires a script to exist.",
                parameters={
                    "type": "object",
                    "properties": {
                        "voice_name": {
                            "type": "string",
                            "description": f"Voice to use. Available voices: {', '.join(voice_names[:10]) if voice_names else 'loading...'}. Leave empty to use default."
                        },
                        "regenerate_script": {
                            "type": "boolean",
                            "description": "If true, regenerate the script before creating audio"
                        }
                    },
                    "required": []
                }
            ),
            types.FunctionDeclaration(
                name="list_available_voices",
                description="Show all available voices with their characteristics (gender, accent, description). Call when user asks about voice options.",
                parameters={"type": "object", "properties": {}}
            ),
            types.FunctionDeclaration(
                name="change_voice",
                description="Change the voice for the next voiceover generation. Call when user requests a specific voice or wants to change voice.",
                parameters={
                    "type": "object",
                    "properties": {
                        "voice_name": {
                            "type": "string",
                            "description": "Name of the voice to use"
                        }
                    },
                    "required": ["voice_name"]
                }
            ),
            types.FunctionDeclaration(
                name="get_current_status",
                description="Get the current state of the narration project (what has been generated, what voice is selected, etc.). Call when user asks about status or 'what's been done'.",
                parameters={"type": "object", "properties": {}}
            )
        ]

    async def _get_system_instruction(self) -> str:
        """Build system instruction for the Gemini agent."""
        
        status = []
        if self.video_file_id:
            status.append("✅ Video uploaded")
        else:
            status.append("⏳ No video uploaded yet")
        
        if self.script_cache:
            status.append(f"✅ Script generated ({len(self.script_cache)} segments)")
        else:
            status.append("⏳ No script yet")
        
        if self.selected_voice_name:
            status.append(f"🎤 Voice: {self.selected_voice_name}")
        else:
            status.append("🎤 Voice: Default")
        
        return f"""You are an intelligent assistant for a screen recording narration pipeline.

**Current State:**
{chr(10).join(status)}

**Your Capabilities:**
1. **generate_script**: Create time-stamped narration scripts from videos
2. **generate_voiceover**: Synthesize voiceover audio with selected voice
3. **list_available_voices**: Show available voice options
4. **change_voice**: Select a different voice
5. **get_current_status**: Check project status

**Guidelines:**
- Be conversational and helpful
- When user uploads a video, ask if they want to generate a script
- When script is ready, **SHOW THE FULL SCRIPT** to the user for review/editing
- After showing script, ask if they want to make changes or proceed to voiceover
- If voice request is ambiguous, call list_available_voices first
- If voice name doesn't exist, call list_available_voices and suggest alternatives
- Always confirm actions before executing
- Be proactive but not pushy

**Conversation Flow:**
1. User uploads video → Suggest generating script
2. Script generated → **Display the COMPLETE script with all timestamps and text**
3. User reviews → Can ask for edits or proceed to voiceover
4. User requests voiceover → Validate voice name, generate audio
5. Audio ready → Provide download links

**CRITICAL: When generate_script returns, you MUST display the full_script field to the user in a readable format. Do not summarize or truncate it.**

**Tone:** Friendly, professional, clear"""

    def _build_conversation_history(self, messages: List[dict]) -> List[types.Content]:
        """Build conversation history for context."""
        history = []
        for msg in messages[-5:]:  # Last 5 messages for context
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                # Map roles to Gemini's expected roles: "user" or "model"
                gemini_role = "user" if role in ["user", "system"] else "model"
                history.append(types.Content(role=gemini_role, parts=[types.Part.from_text(text=content)]))
        return history

    async def _execute_function_call(
        self,
        function_call: types.FunctionCall,
        event_emitter: Optional[Callable[[dict], Awaitable[None]]],
        user: dict
    ) -> dict:
        """Execute the function Gemini requested."""
        
        func_name = function_call.name
        args = dict(function_call.args) if function_call.args else {}
        
        self.log.info(f"Executing function: {func_name} with args: {args}")
        
        try:
            if func_name == "generate_script":
                return await self._func_generate_script(args, event_emitter)
            
            elif func_name == "generate_voiceover":
                return await self._func_generate_voiceover(args, event_emitter, user)
            
            elif func_name == "list_available_voices":
                return await self._func_list_voices()
            
            elif func_name == "change_voice":
                return await self._func_change_voice(args)
            
            elif func_name == "get_current_status":
                return self._func_get_status()
            
            else:
                return {"success": False, "error": f"Unknown function: {func_name}"}
                
        except Exception as e:
            self.log.error(f"Function execution error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def _func_generate_script(self, args: dict, event_emitter) -> dict:
        """Generate script from video."""
        if not self.video_file_id:
            return {"success": False, "error": "No video file uploaded"}
        
        # Get video file
        file_record = FilesDB.get_file_by_id(self.video_file_id)
        if not file_record:
            return {"success": False, "error": "Video file not found"}
        
        # Determine narrative style
        narrative_style = args.get("narrative_style", "first_person_present")
        tone = args.get("tone", self.valves.NARRATION_STYLE)
        
        # Parse narrative style into tense and perspective
        if narrative_style == "first_person_present":
            narrative_tense = "present"
            narrative_perspective = "first-person"
        elif narrative_style == "first_person_past":
            narrative_tense = "past"
            narrative_perspective = "first-person"
        else:  # third_person
            narrative_tense = "present"
            narrative_perspective = "third-person"
        
        # Get API key
        api_key = None
        if not self.valves.USE_VERTEX_AI:
            api_key = self.valves.GEMINI_API_KEY.get_decrypted()
        
        # Generate script
        self.script_cache = await self._generate_script_from_video(
            api_key=api_key,
            video_path=file_record.path,
            event_emitter=event_emitter,
            narrative_tense=narrative_tense,
            narrative_perspective=narrative_perspective,
        )
        
        if not self.script_cache:
            return {"success": False, "error": "Failed to generate script"}
        
        await self.emit_status(event_emitter, "info", f"✅ Generated {len(self.script_cache)} script segments", True)
        
        # Format FULL script for user review
        full_script = "\n".join([
            f"[{self._format_timestamp(seg['start_ms'])}] {seg['text']}" 
            for seg in self.script_cache
        ])
        
        return {
            "success": True,
            "segments": len(self.script_cache),
            "full_script": full_script,
            "message": "Script generated successfully. Please review the full script below. You can ask me to modify it or proceed to generate voiceover."
        }

    async def _func_generate_voiceover(self, args: dict, event_emitter, user: dict) -> dict:
        """Generate voiceover audio."""
        if not self.script_cache:
            return {"success": False, "error": "No script available. Generate script first."}
        
        if not self.video_file_id:
            return {"success": False, "error": "No video file found"}
        
        # Get video file
        file_record = FilesDB.get_file_by_id(self.video_file_id)
        if not file_record:
            return {"success": False, "error": "Video file not found"}
        
        voice_name = args.get("voice_name")
        voice_id = self.selected_voice_id or self.valves.DEFAULT_VOICE_ID
        
        if voice_name:
            # Find voice ID
            found_id = None
            if self.cached_voices:
                for key, (name, vid) in self.cached_voices.items():
                    if name.lower() == voice_name.lower():
                        found_id = vid
                        self.selected_voice_id = vid
                        self.selected_voice_name = name
                        voice_id = vid
                        break
            
            if not found_id:
                return {"success": False, "error": f"Voice '{voice_name}' not found"}
        
        await self.emit_status(event_emitter, "info", f"🎤 Generating voiceover with {self.selected_voice_name or 'default voice'}...")
        
        # Generate TTS for each segment
        eleven_key = self.valves.ELEVEN_API_KEY.get_decrypted()
        if not eleven_key:
            return {"success": False, "error": "ElevenLabs API key not configured"}
        
        audio_segments = []
        for i, segment in enumerate(self.script_cache, 1):
            await self.emit_status(
                event_emitter,
                "info",
                f"🎤 Generating audio {i}/{len(self.script_cache)}: {segment['text'][:50]}..."
            )
            
            result = await self._generate_tts(eleven_key, segment['text'], voice_id)
            if result:
                audio_bytes, duration_sec = result
                audio_segments.append({
                    'index': segment['index'],
                    'start_ms': segment['start_ms'],
                    'end_ms': segment['end_ms'],
                    'text': segment['text'],
                    'audio': audio_bytes,
                    'duration_sec': duration_sec
                })
        
        if not audio_segments:
            return {"success": False, "error": "Failed to generate audio segments"}
        
        # Create synchronized audio file
        await self.emit_status(event_emitter, "info", "🔗 Synchronizing audio segments...")
        audio_file_id = await self._create_synchronized_audio(audio_segments, user, file_record.filename)
        
        if not audio_file_id:
            return {"success": False, "error": "Failed to create synchronized audio"}
        
        # Generate subtitles if enabled - transcribe the actual audio for accurate timing
        subtitle_file_id = None
        subtitle_path = None
        if self.valves.GENERATE_SUBTITLES:
            await self.emit_status(event_emitter, "info", "📝 Transcribing audio for accurate subtitles...")
            subtitle_file_id, subtitle_path = await self._generate_subtitle_file_from_audio(
                audio_file_id,
                file_record.filename,
                user.get("id") if user else None,
                event_emitter
            )
        
        # Merge with video
        await self.emit_status(event_emitter, "info", "🎬 Merging audio with video...")
        
        # Get audio file to merge
        audio_record = FilesDB.get_file_by_id(audio_file_id)
        if not audio_record:
            return {"success": False, "error": "Audio file not found"}
        
        with open(audio_record.path, 'rb') as f:
            audio_bytes = f.read()
        
        merged_video_id = await self._merge_video_audio(
            file_record.path,
            audio_bytes,
            subtitle_path,
            user.get("id") if user else None,
            event_emitter
        )
        
        if not merged_video_id:
            return {"success": False, "error": "Failed to merge video with audio"}
        
        await self.emit_status(event_emitter, "info", "✅ Voiceover complete!", True)
        
        return {
            "success": True,
            "audio_file_id": audio_file_id,
            "video_file_id": merged_video_id,
            "subtitle_file_id": subtitle_file_id,
            "voice_used": self.selected_voice_name or "Default",
            "segments": len(self.script_cache)
        }

    async def _func_list_voices(self) -> dict:
        """List all available voices."""
        if not self.cached_voice_details:
            return {"success": False, "error": "Voices not loaded"}
        
        voices = []
        for name in sorted(self.cached_voice_details.keys())[:20]:
            details = self.cached_voice_details[name]
            voices.append({
                "name": name,
                "gender": details.get("gender", ""),
                "accent": details.get("accent", ""),
                "description": details.get("description", "")
            })
        
        return {
            "success": True,
            "voices": voices,
            "count": len(voices)
        }

    async def _func_change_voice(self, args: dict) -> dict:
        """Change selected voice."""
        voice_name = args.get("voice_name")
        
        if not voice_name:
            return {"success": False, "error": "Voice name required"}
        
        self.log.info(f"Looking for voice: '{voice_name}'")
        
        # Try cached_voices first
        if self.cached_voices:
            for key, (name, vid) in self.cached_voices.items():
                if name.lower() == voice_name.lower():
                    self.log.info(f"Found voice: {name} -> {vid}")
                    self.selected_voice_id = vid
                    self.selected_voice_name = name
                    return {
                        "success": True,
                        "voice_name": name,
                        "message": f"Voice changed to {name}"
                    }
        
        # Fallback: try cached_voice_details
        if self.cached_voice_details:
            for name, details in self.cached_voice_details.items():
                if name.lower() == voice_name.lower():
                    vid = details.get("voice_id")
                    if vid:
                        self.log.info(f"Found voice: {name} -> {vid}")
                        self.selected_voice_id = vid
                        self.selected_voice_name = name
                        return {
                            "success": True,
                            "voice_name": name,
                            "message": f"Voice changed to {name}"
                        }
        
        self.log.warning(f"Voice '{voice_name}' not found")
        return {"success": False, "error": f"Voice '{voice_name}' not found"}

    def _func_get_status(self) -> dict:
        """Get current status."""
        return {
            "success": True,
            "video_uploaded": self.video_file_id is not None,
            "script_generated": self.script_cache is not None,
            "script_segments": len(self.script_cache) if self.script_cache else 0,
            "selected_voice": self.selected_voice_name or "Default"
        }

    def _format_function_result(self, function_name: str, result: dict) -> str:
        """Format function result for display when Gemini doesn't generate text."""
        
        if not result or not isinstance(result, dict):
            return "✅ Task completed successfully!"
        
        if not result.get("success"):
            error = result.get("error", "Unknown error")
            return f"❌ **Error:** {error}"
        
        # Format based on function type
        if function_name == "generate_script":
            if self.script_cache:
                script_text = "\n".join([
                    f"**[{self._format_timestamp(seg['start_ms'])}]** {seg['text']}" 
                    for seg in self.script_cache
                ])
                return f"""✅ **Script Generated Successfully!**

I've analyzed your video and created a {len(self.script_cache)}-segment narration script:

{script_text}

**What's next?**
- Review the script above
- If you want to make changes, let me know which segment to edit
- When ready, say "generate voiceover" or "add voice" to create the audio and final video!"""
            else:
                return "✅ Script generated successfully!"
        
        elif function_name == "list_available_voices":
            voices = result.get("voices", [])
            if voices:
                voice_list = "\n".join([
                    f"- **{v['name']}** ({v['gender']}, {v['accent']}) - {v['description'][:60]}..." 
                    if len(v['description']) > 60 else f"- **{v['name']}** ({v['gender']}, {v['accent']}) - {v['description']}"
                    for v in voices[:15]
                ])
                more = f"\n\n*...and {len(voices) - 15} more voices available*" if len(voices) > 15 else ""
                
                return f"""🎤 **Available Voices** ({len(voices)} total)

{voice_list}{more}

**To use a voice:**
- Say "use [voice name]" or "change voice to [name]"
- Or say "generate voiceover with [name]"

**Example:** "generate voiceover with Alice" or "use British male voice"
"""
            else:
                return "No voices available. Please check your ElevenLabs API configuration."
        
        elif function_name == "generate_voiceover":
            video_id = result.get("video_file_id")
            audio_id = result.get("audio_file_id")
            subtitle_id = result.get("subtitle_file_id")
            voice = result.get("voice_used", "default voice")
            segments = result.get("segments", 0)
            
            # Build file links
            files_section = ""
            if video_id:
                files_section += f"\n\n**📹 Narrated Video:** [Download](/api/v1/files/{video_id}/content)"
            if audio_id:
                files_section += f"\n**🎵 Audio Track:** [Download](/api/v1/files/{audio_id}/content)"
            if subtitle_id:
                files_section += f"\n**📝 Subtitles:** [Download](/api/v1/files/{subtitle_id}/content)"
            
            return f"""🎉 **Voiceover Complete!**

✅ Generated audio for {segments} segments using **{voice}**
✅ Merged audio with video
✅ Created subtitles
{files_section}

**Want to make changes?**
- Try a different voice: "regenerate with [voice name]"
- Edit the script and regenerate
"""
        
        elif function_name == "change_voice":
            voice = result.get("voice_name", "Unknown")
            return f"✅ Voice changed to **{voice}**. Ready to generate voiceover!"
        
        elif function_name == "get_current_status":
            video = "✅ Uploaded" if result.get("video_uploaded") else "❌ Not uploaded"
            script = f"✅ {result.get('script_segments', 0)} segments" if result.get("script_generated") else "❌ Not generated"
            voice = result.get("selected_voice", "Default")
            
            return f"""📊 **Project Status**

**Video:** {video}
**Script:** {script}
**Voice:** {voice}

**Next steps:**
- Upload video if needed
- Generate script: "create script"
- Generate voiceover: "add voiceover"
"""
        
        else:
            return f"✅ Task completed: {function_name}"

    async def _create_gemini_client(self):
        """Create Gemini client with proper authentication."""
        if self.valves.USE_VERTEX_AI:
            # Vertex AI authentication
            project_id = self.valves.PROJECT_ID or os.getenv("GOOGLE_CLOUD_PROJECT", "")
            location = self.valves.LOCATION or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            
            if not project_id:
                raise ValueError("PROJECT_ID required for Vertex AI")
            
            # Handle service account auth
            if self.valves.SERVICE_ACCOUNT_JSON.get_decrypted():
                sa_json = self.valves.SERVICE_ACCOUNT_JSON.get_decrypted()
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    f.write(sa_json)
                    temp_sa_path = f.name
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_sa_path
            elif self.valves.SERVICE_ACCOUNT_PATH:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.valves.SERVICE_ACCOUNT_PATH
            
            return genai.Client(vertexai=True, project=project_id, location=location)
        else:
            # Google AI API
            api_key = self.valves.GEMINI_API_KEY.get_decrypted()
            if not api_key:
                raise ValueError("GEMINI_API_KEY required for Google AI API")
            return genai.Client(api_key=api_key)

    async def _fetch_elevenlabs_voices(self, include_details: bool = False) -> Dict[str, Tuple[str, str]]:
        """Fetch voices from ElevenLabs API."""
        try:
            current_time = time.time()
            if self.cached_voices and (current_time - self.voices_cache_time) < self.VOICE_CACHE_TTL:
                if include_details and not self.cached_voice_details:
                    pass  # Re-fetch with details
                else:
                    return self.cached_voices
            
            eleven_key = self.valves.ELEVEN_API_KEY.get_decrypted()
            if not eleven_key:
                self.log.warning("No ElevenLabs API key")
                return {}
            
            url = f"{self.valves.ELEVEN_API_BASE_URL.replace('/v1', '')}/v2/voices"
            headers = {"xi-api-key": eleven_key}
            params = {"page_size": 100, "voice_type": "default"}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        voices = {}
                        voice_details = {}
                        
                        for voice in data.get("voices", []):
                            full_name = voice.get("name", "")
                            voice_id = voice.get("voice_id", "")
                            is_legacy = voice.get("is_legacy", False)
                            
                            if is_legacy or not full_name or not voice_id:
                                continue
                            
                            # Extract clean name (API returns "Name - Description")
                            clean_name = full_name.split(" - ")[0].strip()
                            
                            voices[clean_name.lower()] = (clean_name, voice_id)
                            
                            if include_details:
                                labels = voice.get("labels", {})
                                voice_details[clean_name] = {
                                    "voice_id": voice_id,
                                    "gender": labels.get("gender", ""),
                                    "accent": labels.get("accent", ""),
                                    "age": labels.get("age", ""),
                                    "description": voice.get("description", ""),
                                    "full_name": full_name  # Keep original for reference
                                }
                        
                        self.cached_voices = voices
                        if include_details:
                            self.cached_voice_details = voice_details
                        self.voices_cache_time = current_time
                        
                        self.log.info(f"Fetched {len(voices)} voices from ElevenLabs")
                        return voices
                    else:
                        self.log.warning(f"Failed to fetch voices: {response.status}")
                        return {}
        except Exception as e:
            self.log.error(f"Error fetching voices: {e}")
            return {}

    async def emit_status(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        level: str,
        message: str,
        done: bool = False,
    ):
        """Emit status message to UI."""
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "status": "complete" if done else "in_progress",
                        "level": level,
                        "description": message,
                        "done": done,
                    },
                }
            )

    # Core implementation methods ported from original screen_recording_narrator.py
    
    async def _generate_script_from_video(
        self,
        api_key: Optional[str],
        video_path: str,
        event_emitter: Optional[Callable[[dict], Awaitable[None]]],
        narrative_tense: str = "present",
        narrative_perspective: str = "first-person",
    ) -> List[Dict]:
        """Analyze video with Gemini and generate time-stamped narration script."""
        temp_creds_path = None
        old_creds = None
        
        try:
            if self.valves.USE_VERTEX_AI:
                project_id = self.valves.PROJECT_ID or os.getenv("GOOGLE_CLOUD_PROJECT", "")
                location = self.valves.LOCATION or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

                self.log.info(f"Using Vertex AI, project={project_id}, location={location}")

                if self.valves.SERVICE_ACCOUNT_JSON:
                    service_account_json = self.valves.SERVICE_ACCOUNT_JSON.get_decrypted()
                    service_account_info = json.loads(service_account_json)

                    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                        json.dump(service_account_info, f)
                        temp_creds_path = f.name

                    old_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_creds_path
                else:
                    old_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.valves.SERVICE_ACCOUNT_PATH

                client = genai.Client(vertexai=True, project=project_id, location=location)
            else:
                self.log.info("Using API key authentication")
                client = genai.Client(api_key=api_key)
            
            await self.emit_status(event_emitter, "info", "📤 Uploading video to Gemini...")
            
            with open(video_path, "rb") as f:
                video_data = f.read()
            
            import mimetypes
            mime_type, _ = mimetypes.guess_type(video_path)
            if not mime_type or not mime_type.startswith("video/"):
                mime_type = "video/mp4"
            
            interval = self.valves.SCRIPT_INTERVAL_SECONDS
            style_instructions = {
                "professional": "Use clear, professional language suitable for business presentations.",
                "casual": "Use friendly, conversational language as if explaining to a friend.",
                "tutorial": "Use instructional language with step-by-step guidance.",
                "educational": "Use clear explanations with emphasis on learning and understanding.",
            }
            
            style = style_instructions.get(self.valves.NARRATION_STYLE, style_instructions["professional"])
            technical_detail = "Include specific UI elements, button names, menu items, and actions." if self.valves.INCLUDE_TECHNICAL_DETAILS else "Focus on high-level actions and outcomes."
            
            if narrative_perspective == "first-person":
                if narrative_tense == "past":
                    perspective_instruction = "Use first-person past tense (e.g., 'I clicked on the menu', 'I opened the settings')"
                    example = "'I clicked on the File menu and selected New Document'"
                else:
                    perspective_instruction = "Use first-person present tense (e.g., 'I click on the menu', 'I open the settings')"
                    example = "'I click on the File menu and select New Document'"
            else:
                if narrative_tense == "past":
                    perspective_instruction = "Use third-person past tense (e.g., 'The user clicked on the menu', 'We opened the settings')"
                    example = "'The user clicked on the File menu and selected New Document'"
                else:
                    perspective_instruction = "Use third-person present tense (e.g., 'The user clicks on the menu', 'We see the settings open')"
                    example = "'The user clicks on the File menu and selects New Document'"
            
            prompt = f"""Analyse this screen recording video and generate a narration script with ACCURATE TIMING.

CRITICAL: You must analyze the ACTUAL video content to determine when actions happen and how long they take. Do NOT use fixed intervals.

Create a time-stamped voiceover script with the following requirements:

1. **TIMING ACCURACY (MOST IMPORTANT):**
   - Watch the video carefully and identify when each distinct action or scene change occurs
   - Set timestamps based on when things ACTUALLY HAPPEN in the video, not at fixed intervals
   - Each segment's timestamp should mark when that action/scene begins
   - Consider scene transitions, UI changes, window switches, loading states, etc.
   - Aim for segments approximately every {interval} seconds, but ADJUST based on actual content

2. **Narration style:** {self.valves.NARRATION_STYLE}
3. **Narrative perspective:** {perspective_instruction}
   - Example: {example}

4. **Content guidelines:**
   - Each segment describes what's happening at that specific moment
   - Keep segments concise (1-2 sentences)
   - Focus on user actions and important visual elements
   - {style}
   - {technical_detail}

OUTPUT FORMAT (JSON):
{{
  "segments": [
    {{"timestamp": "00:00", "text": "The screen shows..."}},
    {{"timestamp": "00:08", "text": "Next, we see..."}}
  ]
}}"""
            
            parts = [
                types.Part.from_bytes(data=video_data, mime_type=mime_type),
                types.Part.from_text(text=prompt)
            ]
            
            await self.emit_status(event_emitter, "info", "🤖 Generating script with Gemini AI...")
            
            response = client.models.generate_content(
                model=self.valves.GEMINI_MODEL,
                contents=[types.Content(role="user", parts=parts)],
            )
            
            response_text = response.text.strip()
            
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            
            try:
                script_data = json.loads(response_text)
                segments_raw = script_data.get("segments", [])
            except json.JSONDecodeError:
                self.log.warning("Failed to parse JSON, falling back to text parsing")
                segments_raw = self._parse_script_from_text(response_text)
            
            script_segments = []
            for i, seg in enumerate(segments_raw):
                timestamp_str = seg.get("timestamp", f"{i * interval:02d}:00")
                text = seg.get("text", "").strip()
                
                if not text:
                    continue
                
                start_ms = self._parse_timestamp_to_ms(timestamp_str)
                
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
            if temp_creds_path and os.path.exists(temp_creds_path):
                try:
                    os.unlink(temp_creds_path)
                except Exception as e:
                    self.log.warning(f"Failed to delete temp credentials file: {e}")
            
            if old_creds is not None:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = old_creds
            elif temp_creds_path:
                os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

    def _parse_script_from_text(self, text: str) -> List[Dict]:
        """Parse script from plain text format (fallback)."""
        segments = []
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
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
                minutes, seconds = map(int, parts)
                return (minutes * 60 + seconds) * 1000
            elif len(parts) == 3:
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
                    
                    audio_base64 = response_data.get("audio_base64")
                    if not audio_base64:
                        self.log.error("No audio_base64 in response")
                        return None
                    
                    audio_bytes = base64.b64decode(audio_base64)
                    
                    alignment = response_data.get("alignment", {})
                    char_end_times = alignment.get("character_end_times_seconds", [])
                    
                    if char_end_times:
                        actual_duration_seconds = char_end_times[-1]
                    else:
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
    
    async def _transcribe_audio_whisper(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """Transcribe audio using Whisper API or local model for subtitle generation."""
        try:
            if self.valves.WHISPER_MODE == "openai" or self.valves.WHISPER_MODE == "openai-compatible":
                return await self._transcribe_openai_whisper(audio_path)
            elif self.valves.WHISPER_MODE == "local":
                return await self._transcribe_local_whisper(audio_path)
            else:
                self.log.error(f"Invalid WHISPER_MODE: {self.valves.WHISPER_MODE}")
                return None
        except Exception as e:
            self.log.error(f"Audio transcription failed: {e}", exc_info=True)
            return None
    
    async def _transcribe_openai_whisper(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """Transcribe using OpenAI Whisper API with word-level timestamps."""
        try:
            # Get API key - prefer OPENAI_API_KEY_WHISPER, fallback to regular API key
            api_key = self.valves.OPENAI_API_KEY_WHISPER.get_decrypted()
            if not api_key:
                self.log.warning("OPENAI_API_KEY_WHISPER not set, attempting to use GEMINI_API_KEY (may not work)")
                return None
            
            base_url = self.valves.OPENAI_API_BASE_URL_WHISPER.rstrip("/")
            url = f"{base_url}/audio/transcriptions"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
            }
            
            # Read audio file
            with open(audio_path, "rb") as f:
                audio_data = f.read()
            
            # Build form data
            data = aiohttp.FormData()
            data.add_field("file", audio_data, filename="audio.mp3", content_type="audio/mpeg")
            data.add_field("model", self.valves.WHISPER_MODEL)
            data.add_field("response_format", "verbose_json")
            
            # Request word-level timestamps for accurate subtitle segmentation
            data.add_field("timestamp_granularities[]", "word")
            data.add_field("timestamp_granularities[]", "segment")
            
            if self.valves.WHISPER_LANGUAGE:
                data.add_field("language", self.valves.WHISPER_LANGUAGE)
            
            data.add_field("temperature", "0.0")
            
            timeout = aiohttp.ClientTimeout(total=self.valves.TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, data=data) as resp:
                    if resp.status != 200:
                        txt = (await resp.text())[:500]
                        self.log.error(f"OpenAI Whisper API error {resp.status}: {txt}")
                        return None
                    
                    result = await resp.json()
                    self.log.debug(f"Transcription successful, got {len(result.get('segments', []))} segments")
                    return result
        
        except Exception as e:
            self.log.error(f"OpenAI Whisper transcription failed: {e}", exc_info=True)
            return None
    
    async def _transcribe_local_whisper(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """Transcribe using local Whisper model with word-level timestamps."""
        try:
            import whisper
            
            model_name = self.valves.WHISPER_MODEL
            self.log.debug(f"Loading Whisper model: {model_name}")
            
            # Load model in thread pool to avoid blocking
            model = await asyncio.to_thread(whisper.load_model, model_name)
            
            # Transcribe with word-level timestamps
            options = {
                "language": self.valves.WHISPER_LANGUAGE,
                "temperature": 0.0,
                "word_timestamps": True,  # Critical for accurate subtitles
            }
            
            result = await asyncio.to_thread(model.transcribe, audio_path, **options)
            self.log.debug(f"Local transcription successful, got {len(result.get('segments', []))} segments")
            
            return result
        
        except ImportError:
            self.log.error("Whisper package not installed. Run: pip install openai-whisper")
            return None
        except Exception as e:
            self.log.error(f"Local Whisper transcription failed: {e}", exc_info=True)
            return None
    
    def _map_words_to_segments(self, transcription: Dict[str, Any]) -> Dict[str, Any]:
        """Map top-level words array from OpenAI API into segments.
        
        OpenAI's timestamp_granularities response format has words at the top level,
        not nested inside segments. This method distributes them correctly.
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
        """Split long Whisper segments into shorter subtitle-appropriate segments.
        
        Uses word-level timestamps to create well-timed, readable subtitles.
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
    
    def _generate_srt_from_segments(self, segments: List[Dict[str, Any]]) -> str:
        """Generate SRT subtitle file content from Whisper segments."""
        # Split into subtitle-appropriate segments using word-level timestamps
        subtitle_segments = self._split_into_subtitle_segments(segments)
        
        srt_lines = []
        for i, segment in enumerate(subtitle_segments, 1):
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            text = segment.get("text", "").strip()
            
            if not text:
                continue
            
            start_time = self._format_srt_timestamp_seconds(start)
            end_time = self._format_srt_timestamp_seconds(end)
            
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(text)
            srt_lines.append("")
        
        return "\n".join(srt_lines)
    
    def _format_srt_timestamp_seconds(self, seconds: float) -> str:
        """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    async def _generate_subtitle_file_from_audio(
        self,
        audio_file_id: str,
        video_filename: str,
        user_id: Optional[str] = None,
        event_emitter: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Generate SRT subtitle file by transcribing the actual voiceover audio.
        
        This provides accurate word-level timestamps for properly synchronized subtitles.
        """
        try:
            # Get audio file from database
            audio_record = FilesDB.get_file_by_id(audio_file_id)
            if not audio_record:
                self.log.error(f"Audio file not found: {audio_file_id}")
                return (None, None)
            
            audio_path = audio_record.path
            self.log.info(f"Transcribing audio for subtitles: {audio_path}")
            
            # Transcribe audio with Whisper
            await self.emit_status(event_emitter, "info", "🎤 Analyzing voiceover audio with Whisper...")
            transcription = await self._transcribe_audio_whisper(audio_path)
            
            if not transcription or "segments" not in transcription:
                self.log.error("Failed to transcribe audio for subtitles")
                return (None, None)
            
            # Map top-level words to segments if needed (OpenAI API format)
            if "words" in transcription and transcription["words"]:
                transcription = self._map_words_to_segments(transcription)
            
            segments = transcription["segments"]
            self.log.info(f"Transcription complete: {len(segments)} segments")
            
            # Generate SRT content with intelligent segmentation
            await self.emit_status(event_emitter, "info", "✂️ Creating optimized subtitle segments...")
            srt_content = self._generate_srt_from_segments(segments)
            
            if not srt_content:
                self.log.error("Failed to generate SRT content")
                return (None, None)
            
            # Save SRT file
            temp_srt = tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                suffix='.srt',
                delete=False
            )
            temp_srt.write(srt_content)
            temp_srt.close()
            
            timestamp = int(time.time())
            base_name = Path(video_filename).stem
            filename = f"{base_name}_subtitles_{timestamp}.srt"
            
            with open(temp_srt.name, 'rb') as f:
                file_data, file_path = Storage.upload_file(
                    f,
                    filename,
                    {"content_type": "application/x-subrip", "source": "screen_recording_narrator_conv_whisper"}
                )
            
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
                        "source": "screen_recording_narrator_conv_whisper",
                        "subtitle_count": len(srt_content.strip().split('\n\n')),
                    },
                ),
            )
            
            subtitle_count = len(srt_content.strip().split('\n\n'))
            self.log.info(f"Generated subtitle file with {subtitle_count} entries: {filename} ({file_size} bytes)")
            await self.emit_status(event_emitter, "info", f"✅ Created {subtitle_count} subtitle segments")
            
            return (record.id if record else None, temp_srt.name)
            
        except Exception as e:
            self.log.error(f"Failed to generate subtitle file from audio: {e}", exc_info=True)
            return (None, None)
    
    async def _generate_subtitle_file(
        self,
        script_segments: List[Dict],
        video_filename: str,
        user_id: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Generate SRT subtitle file from script segments (fallback method).
        
        Note: This uses script timestamps which may not match actual audio timing.
        Prefer _generate_subtitle_file_from_audio() for accurate subtitles.
        """
        try:
            srt_content = self._format_srt_subtitles(script_segments)
            
            temp_srt = tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                suffix='.srt',
                delete=False
            )
            temp_srt.write(srt_content)
            temp_srt.close()
            
            timestamp = int(time.time())
            base_name = Path(video_filename).stem
            filename = f"{base_name}_subtitles_{timestamp}.srt"
            
            with open(temp_srt.name, 'rb') as f:
                file_data, file_path = Storage.upload_file(
                    f,
                    filename,
                    {"content_type": "application/x-subrip", "source": "screen_recording_narrator_conv"}
                )
            
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
                        "source": "screen_recording_narrator_conv",
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
            start_time = self._format_srt_timestamp(segment['start_ms'])
            end_time = self._format_srt_timestamp(segment['end_ms'])
            
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(segment['text'])
            srt_lines.append("")
        
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
            ffprobe_exe = ffmpeg_exe.replace('ffmpeg', 'ffprobe')
            
            if not os.path.exists(ffprobe_exe):
                cmd = [ffmpeg_exe, "-i", file_path]
                result = await asyncio.to_thread(subprocess.run, cmd, capture_output=True)
                stderr = result.stderr.decode(errors="ignore")
                
                match = re.search(r'Duration: (\d{2}):(\d{2}):(\d{2})\.(\d{2})', stderr)
                if match:
                    hours, minutes, seconds, centiseconds = map(int, match.groups())
                    duration = hours * 3600 + minutes * 60 + seconds + centiseconds / 100.0
                    return duration
            else:
                cmd = [
                    ffprobe_exe,
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    file_path
                ]
                result = await asyncio.to_thread(subprocess.run, cmd, capture_output=True)
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
        """Extend video by freezing the last frame to reach target duration.
        
        Optimized approach: Extract last frame, create extension video, concat without re-encoding.
        This is 10-100x faster than tpad filter which re-encodes the entire video.
        """
        tmp_last_frame = None
        tmp_extension = None
        
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            
            current_duration = await self._get_media_duration(video_path)
            if not current_duration or target_duration <= current_duration:
                return None
            
            freeze_duration = target_duration - current_duration
            self.log.info(f"Extending video by {freeze_duration:.2f}s using optimized concat method")
            
            # Step 1: Extract the last frame as an image
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tf:
                tmp_last_frame = tf.name
            
            extract_cmd = [
                ffmpeg_exe,
                "-y",
                "-sseof", "-1",  # Seek to 1 second before end
                "-i", video_path,
                "-update", "1",
                "-frames:v", "1",
                tmp_last_frame,
            ]
            
            result = await asyncio.to_thread(subprocess.run, extract_cmd, capture_output=True)
            if result.returncode != 0 or not os.path.exists(tmp_last_frame):
                self.log.error("Failed to extract last frame")
                return None
            
            # Step 2: Create extension video from the static frame (much faster than re-encoding original)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tf:
                tmp_extension = tf.name
            
            # Use loop to create video from single image for the freeze duration
            # With -c:v libx264 -preset ultrafast for speed
            extension_cmd = [
                ffmpeg_exe,
                "-y",
                "-loop", "1",
                "-i", tmp_last_frame,
                "-t", str(freeze_duration),
                "-c:v", "libx264",
                "-preset", "ultrafast",  # Fastest encoding for the extension
                "-pix_fmt", "yuv420p",
                "-r", "30",  # 30 fps is sufficient for static frames
                "-an",  # No audio in extension
                tmp_extension,
            ]
            
            result = await asyncio.to_thread(subprocess.run, extension_cmd, capture_output=True)
            if result.returncode != 0 or not os.path.exists(tmp_extension):
                stderr = result.stderr.decode(errors="ignore") if result.stderr else ""
                self.log.error(f"Failed to create extension video: {stderr[:500]}")
                return None
            
            # Step 3: Concatenate original + extension using concat demuxer (no re-encoding!)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tf:
                tmp_output = tf.name
            
            # Create concat file list
            concat_list = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".txt")
            concat_list.write(f"file '{video_path}'\n")
            concat_list.write(f"file '{tmp_extension}'\n")
            concat_list.close()
            
            try:
                # Use concat demuxer with copy codec (no re-encoding)
                concat_cmd = [
                    ffmpeg_exe,
                    "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_list.name,
                    "-c", "copy",  # Critical: copy codec = no re-encoding
                    tmp_output,
                ]
                
                result = await asyncio.to_thread(subprocess.run, concat_cmd, capture_output=True)
                
                if result.returncode != 0:
                    stderr = result.stderr.decode(errors="ignore") if result.stderr else ""
                    self.log.error(f"Failed to concat videos: {stderr[:500]}")
                    if os.path.exists(tmp_output):
                        os.unlink(tmp_output)
                    return None
                
                if not os.path.exists(tmp_output) or os.path.getsize(tmp_output) == 0:
                    self.log.error("Extended video file is empty or missing")
                    if os.path.exists(tmp_output):
                        os.unlink(tmp_output)
                    return None
                
                self.log.info(f"Video extended successfully using fast concat method")
                return tmp_output
                
            finally:
                # Cleanup concat list
                if os.path.exists(concat_list.name):
                    os.unlink(concat_list.name)
            
        except Exception as e:
            self.log.error(f"Error extending video: {e}", exc_info=True)
            return None
        finally:
            # Cleanup temporary files
            for tmp_file in [tmp_last_frame, tmp_extension]:
                if tmp_file and os.path.exists(tmp_file):
                    try:
                        os.unlink(tmp_file)
                    except:
                        pass

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
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as ta:
                ta.write(audio_bytes)
                tmp_audio = ta.name
            
            video_duration = await self._get_media_duration(video_path)
            audio_duration = await self._get_media_duration(tmp_audio)
            
            video_to_use = video_path
            
            # Always ensure video is long enough for audio + buffer
            if audio_duration and video_duration:
                target_duration = audio_duration + self.valves.VIDEO_END_BUFFER
                
                if target_duration > video_duration:
                    # Video needs to be extended
                    extension_needed = target_duration - video_duration
                    self.log.info(f"Extending video by {extension_needed:.2f}s (audio: {audio_duration:.2f}s + buffer: {self.valves.VIDEO_END_BUFFER:.2f}s = {target_duration:.2f}s)")
                    await self.emit_status(__event_emitter__, "info", f"🎬 Extending video by {extension_needed:.1f}s to add end buffer...")
                    
                    tmp_extended_video = await self._extend_video_with_freeze_frame(video_path, target_duration)
                    if tmp_extended_video:
                        video_to_use = tmp_extended_video
                        await self.emit_status(__event_emitter__, "info", f"✅ Video extended to {target_duration:.2f}s")
                else:
                    self.log.info(f"Video duration ({video_duration:.2f}s) sufficient for audio ({audio_duration:.2f}s) + buffer ({self.valves.VIDEO_END_BUFFER:.2f}s)")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as to:
                tmp_out = to.name
            
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            
            cmd = [ffmpeg_exe, "-y", "-i", video_to_use, "-i", tmp_audio]
            
            # Note: Using -shortest ensures output matches the shorter of video/audio
            # Since we've extended the video to audio + buffer, the video will be longer
            # and -shortest will keep the full audio + show the frozen frame buffer
            if subtitle_path and os.path.exists(subtitle_path):
                cmd.extend(["-i", subtitle_path, "-map", "0:v:0", "-map", "1:a:0", "-map", "2:s:0",
                           "-c:v", "copy", "-c:a", "aac", "-c:s", "mov_text", "-b:a", "192k",
                           "-metadata:s:s:0", "language=eng", "-shortest", tmp_out])
            else:
                cmd.extend(["-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                           "-b:a", "192k", "-shortest", tmp_out])
            
            completed = await asyncio.to_thread(subprocess.run, cmd, capture_output=True)
            
            if completed.returncode != 0 or not os.path.exists(tmp_out) or os.path.getsize(tmp_out) == 0:
                self.log.error("FFmpeg merge failed")
                return None
            
            video_basename = Path(video_path).stem
            timestamp = int(time.time())
            filename = f"{video_basename}_with_voiceover_{timestamp}.mp4"
            file_id = str(uuid.uuid4())
            
            with open(tmp_out, "rb") as f:
                file_data, file_path = Storage.upload_file(
                    f, filename, {"content_type": "video/mp4", "source": "screen_recording_narrator_conv"}
                )
            
            FilesDB.insert_new_file(
                user_id or "system",
                FileForm(
                    id=file_id, filename=filename, path=file_path,
                    meta={"name": filename, "content_type": "video/mp4",
                          "size": os.path.getsize(tmp_out), "source": "screen_recording_narrator_conv"}
                ),
            )
            
            self.log.info(f"Created merged video file: {file_id}")
            return file_id
            
        except Exception as e:
            self.log.error(f"Failed to merge video with audio: {e}", exc_info=True)
            return None
        finally:
            for tmp_file in [tmp_audio, tmp_out, tmp_extended_video]:
                if tmp_file and os.path.exists(tmp_file):
                    try:
                        os.unlink(tmp_file)
                    except:
                        pass
    
    async def _create_synchronized_audio(
        self, audio_segments: List[Dict], __user__: dict, source_filename: str,
    ) -> Optional[str]:
        """Create synchronized audio file with proper timing using pydub/ffmpeg."""
        try:
            from pydub import AudioSegment
            AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
            
            if not audio_segments:
                return None
                
            total_duration_ms = max(seg['end_ms'] for seg in audio_segments)
            final_audio = AudioSegment.silent(duration=total_duration_ms)
            last_audio_end_ms = 0
            
            for segment in audio_segments:
                temp_file = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                        tmp.write(segment['audio'])
                        temp_file = tmp.name
                    
                    audio_clip = AudioSegment.from_mp3(temp_file)
                    actual_audio_length_ms = len(audio_clip)
                    
                    intended_start_ms = segment['start_ms']
                    actual_start_ms = max(intended_start_ms, last_audio_end_ms)
                    
                    if actual_start_ms > intended_start_ms:
                        self.log.info(f"Segment {segment['index']}: Delaying by {actual_start_ms - intended_start_ms}ms")
                    
                    final_audio = final_audio.overlay(audio_clip, position=actual_start_ms)
                    last_audio_end_ms = actual_start_ms + actual_audio_length_ms
                    
                finally:
                    if temp_file and os.path.exists(temp_file):
                        os.unlink(temp_file)
            
            output_file = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    output_file = tmp.name
                
                final_audio.export(output_file, format="mp3", bitrate="128k")
                
                base_name = source_filename.rsplit('.', 1)[0]
                timestamp = int(time.time())
                filename = f"{base_name}_voiceover_{timestamp}.mp3"
                
                user_id = __user__.get("id") if __user__ else None
                file_id = str(uuid.uuid4())
                
                with open(output_file, "rb") as f:
                    file_data, file_path = Storage.upload_file(
                        f, file_id, {"content_type": "audio/mpeg", "source": "screen_recording_narrator_conv"}
                    )
                
                FilesDB.insert_new_file(
                    user_id or "system",
                    FileForm(
                        id=file_id, filename=filename, path=file_path,
                        meta={"name": filename, "content_type": "audio/mpeg", "size": len(file_data),
                              "source": "screen_recording_narrator_conv", "segments": len(audio_segments)}
                    ),
                )
                
                self.log.info(f"Created synchronized voiceover audio file: {file_id}")
                return file_id
                
            finally:
                if output_file and os.path.exists(output_file):
                    os.unlink(output_file)
                    
        except ImportError:
            self.log.error("pydub not installed")
            return None
        except Exception as e:
            self.log.error(f"Failed to create synchronized audio: {e}", exc_info=True)
            return None
