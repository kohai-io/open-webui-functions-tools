"""
title: ElevenLabs Text-to-Speech Tool
funding_url: https://github.com/open-webui
version: 0.2.4
license: MIT
"""

import os
import uuid
import requests
import base64
from typing import Callable, Union, Any, Optional
from pydantic import BaseModel, Field

from open_webui.config import DATA_DIR
from open_webui.models.files import Files

# from open_webui.models.knowledge import KnowledgeTable  # (unused)

DEBUG = False

class Tools:
    class Valves(BaseModel):
        ELEVENLABS_API_KEY: Optional[str] = Field(
            default=None, description="Your ElevenLabs API key."
        )
        ELEVENLABS_MODEL_ID: str = Field(
            default="eleven_multilingual_v2",
            description="ID of the ElevenLabs TTS model to use.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.voice_id_cache = {}

    def fetch_available_voices(self) -> str:
        """
        Fetch the list of available voices from the ElevenLabs API.
        """
        if DEBUG:
            print("Debug: Fetching available voices")

        base_url = "https://api.elevenlabs.io/v1"
        headers = {
            "xi-api-key": self.valves.ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
        }

        voices_url = f"{base_url}/voices"
        try:
            response = requests.get(voices_url, headers=headers)
            response.raise_for_status()
            voices_data = response.json()

            message = "Here are the available voices from ElevenLabs:\n\n"
            for voice in voices_data.get("voices", []):
                message += f"- {voice['name']}: {voice.get('description', 'No description available.')}\n"
                self.voice_id_cache[voice["name"].lower()] = voice["voice_id"]

            if DEBUG:
                print(f"Debug: Found {len(voices_data.get('voices', []))} voices")

            return message
        except requests.RequestException as e:
            if DEBUG:
                print(f"Debug: Error fetching voices: {str(e)}")
            return "Sorry, I couldn't fetch the list of available voices at the moment."

    def get_voice_list(self) -> str:
        """
        Retrieve and return a list of available voices as a formatted string.
        """
        return self.fetch_available_voices()

    async def elevenlabs_text_to_speech(
        self,
        text: str,
        voice_name: str = "Bradford",
        __user__: dict = {},
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Convert text to speech using the ElevenLabs API. This function handles the complete text-to-speech conversion process and provides the user with a downloadable audio file.

        :param text: The text to convert to speech or "list voices" to retrieve available voices.
        :param voice_name: The name of the voice to use for speech generation. Defaults to "Bradford".
        :param __user__: A dictionary containing user information (must include 'id').
        :param __event_emitter__: Optional callback to emit status events.
        :return: Completion status for the LLM context.
        """

        if DEBUG:
            print(
                f"Debug: Starting TTS for voice '{voice_name}' with text '{text[:20]}...'"
            )

        def status_object(
            description="Unknown State", status="in_progress", done=False
        ):
            return {
                "type": "status",
                "data": {"status": status, "description": description, "done": done},
            }

        if __event_emitter__:
            await __event_emitter__(
                status_object("Initializing ElevenLabs Text-to-Speech")
            )

        if not self.valves.ELEVENLABS_API_KEY:
            if __event_emitter__:
                await __event_emitter__(
                    status_object("Error: API key not set", status="error", done=True)
                )
            return "ElevenLabs API key is not set. Please set it in your environment variables."

        if "id" not in __user__:
            if __event_emitter__:
                await __event_emitter__(
                    status_object(
                        "Error: User not authenticated", status="error", done=True
                    )
                )
            return "Error: User ID is not available. Please ensure you're logged in."

        if text.lower().strip() in {
            "list voices",
            "show voices",
            "available voices",
            "what voices are available",
        }:
            voices = self.get_voice_list()
            if __event_emitter__:
                await __event_emitter__(
                    status_object(
                        "Available voices fetched", status="complete", done=True
                    )
                )
            return voices

        # Resolve voice_id (cache -> fetch -> cache)
        voice_id = self.voice_id_cache.get(voice_name.lower())
        if not voice_id:
            voices_message = self.fetch_available_voices()
            if voices_message.startswith("Sorry, I couldn't fetch"):
                if __event_emitter__:
                    await __event_emitter__(
                        status_object(
                            "Error: Could not fetch voices", status="error", done=True
                        )
                    )
                return voices_message

            voice_id = self.voice_id_cache.get(voice_name.lower())
            if not voice_id:
                if __event_emitter__:
                    await __event_emitter__(
                        status_object(
                            f"Error: Voice '{voice_name}' not found",
                            status="error",
                            done=True,
                        )
                    )
                return f"Error: Voice '{voice_name}' not found. Use 'list voices' to see available options."

        if __event_emitter__:
            await __event_emitter__(status_object("Generating speech"))

        base_url = "https://api.elevenlabs.io/v1"
        headers = {
            "xi-api-key": self.valves.ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
        }

        tts_url = f"{base_url}/text-to-speech/{voice_id}"
        payload = {
            "text": text,
            "model_id": self.valves.ELEVENLABS_MODEL_ID,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
        }

        try:
            response = requests.post(tts_url, json=payload, headers=headers)
            response.raise_for_status()

            audio_data = response.content
            file_name = f"tts_{uuid.uuid4()}.mp3"

            file_id = self._create_file(
                file_name, "Generated Audio", audio_data, "audio/mpeg", __user__
            )
            if file_id:
                file_url = self._get_file_url(file_id)
                
                if __event_emitter__:
                    await __event_emitter__(
                        status_object(
                            "Generated successfully", status="complete", done=True
                        )
                    )
                    
                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {
                                "content": f'🎵 **Audio Generated Successfully!**\n\n**Text:** "{text}"\n**Voice:** {voice_name}\n**Format:** MP3\n**File:** `{file_name}`\n\n🎧 **[Click to Play/Download Audio]({file_url})**\n\n*Click the link above to play the audio in your browser or right-click and select "Save Link As..." to download the file.*\n\n✅ **Task completed successfully.**'
                            },
                        }
                    )
            return ""
        except requests.RequestException as e:
            if __event_emitter__:
                await __event_emitter__(
                    status_object(
                        "Error: API request failed", status="error", done=True
                    )
                )
            return f"Error generating speech: {str(e)}"

    def _create_file(
        self,
        file_name: str,
        title: str,
        content: Union[str, bytes],
        content_type: str,
        __user__: dict = {},
    ) -> Optional[str]:
        """
        Create and save a file in DATA_DIR and register it with the Files API.
        """
        if DEBUG:
            print("Debug: Entering _create_file")
            print("Debug: DATA_DIR:", DATA_DIR)
            print("Debug: File name:", file_name)
            print("Debug: Content type:", content_type)
            print("Debug: User:", __user__)

        if "id" not in __user__:
            if DEBUG:
                print("Debug: User ID is not available")
            return None

        try:
            base_path = os.path.abspath(DATA_DIR)  # ensure absolute path
            os.makedirs(base_path, exist_ok=True)

            file_id = str(uuid.uuid4())
            file_path = os.path.join(base_path, f"{file_id}_{file_name}")
            mode = "w" if isinstance(content, str) else "wb"

            with open(file_path, mode) as f:
                f.write(content)

            if not os.path.exists(file_path):
                if DEBUG:
                    print("Debug: File missing after write:", file_path)
                return None

            meta = {
                "source": file_path,
                "title": title,
                "content_type": content_type,
                "size": os.path.getsize(file_path),
                "path": file_path,
                "storage": "local",
            }

            class FileForm(BaseModel):
                id: str
                filename: str
                path: Optional[str] = None  # top-level path used by /content
                meta: dict = {}

            formData = FileForm(
                id=file_id, filename=file_name, path=file_path, meta=meta
            )
            file = Files.insert_new_file(__user__["id"], formData)

            if DEBUG:
                print("Debug: File saved:", file_path, "-> id:", file.id)

            return file.id
        except Exception as e:
            if DEBUG:
                print("Debug: Error saving file:", repr(e))
            return None

    def _get_file_url(self, file_id: str) -> str:
        """
        Construct the URL to access the file content by its ID.
        """
        return f"/api/v1/files/{file_id}/content"
