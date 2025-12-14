"""
title: Google Gemini 2.x Flash Image Generation (Streaming)
requirements: google-genai, cryptography
"""

import base64
import mimetypes
import os
import hashlib
from typing import Optional, Any
from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import core_schema
from open_webui.routers.images import upload_image, load_b64_image_data
from open_webui.models.users import Users

# Open WebUI imports
from fastapi import Request

# Google Gemini (banana-style) imports
from google import genai
from google.genai import types


# Encryption implementation from Veo3 pipeline
class EncryptedStr(str):
    """A string type that automatically handles encryption/decryption"""

    @classmethod
    def _get_encryption_key(cls) -> Optional[bytes]:
        """
        Generate encryption key from WEBUI_SECRET_KEY if available
        Returns None if no key is configured
        """
        secret = os.getenv("WEBUI_SECRET_KEY")
        if not secret:
            return None

        hashed_key = hashlib.sha256(secret.encode()).digest()
        return base64.urlsafe_b64encode(hashed_key)

    @classmethod
    def encrypt(cls, value: str) -> str:
        """
        Encrypt a string value if a key is available
        Returns the original value if no key is available
        """
        if not value or value.startswith("encrypted:"):
            return value

        key = cls._get_encryption_key()
        if not key:  # No encryption if no key
            return value

        f = Fernet(key)
        encrypted = f.encrypt(value.encode())
        return f"encrypted:{encrypted.decode()}"

    @classmethod
    def decrypt(cls, value: str) -> str:
        """
        Decrypt an encrypted string value if a key is available
        Returns the original value if no key is available or decryption fails
        """
        if not value or not value.startswith("encrypted:"):
            return value

        key = cls._get_encryption_key()
        if not key:  # No decryption if no key
            return value[len("encrypted:") :]  # Return without prefix

        try:
            encrypted_part = value[len("encrypted:") :]
            f = Fernet(key)
            decrypted = f.decrypt(encrypted_part.encode())
            return decrypted.decode()
        except (InvalidToken, Exception):
            return value

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


class Tools:
    """Container class for Open WebUI tools."""

    class Valves(BaseModel):
        """User-configurable settings for the tool."""

        api_key: EncryptedStr = Field(default="", description="Your Google AI API key here")
        # According to the official banana example, the default uses the 2.5 flash image preview.
        model_name: str = Field(
            default="gemini-2.5-flash-image-preview",
            description="The Google AI model name for image+text generation (streaming)",
        )

    def __init__(self):
        """Initialize the Tool."""
        self.valves = self.Valves()

    async def gemini_generate_image(
        self,
        prompt: str,
        __request__: Request,
        __user__: dict,
        __event_emitter__=None,
    ) -> str:
        """
        Generates image(s) and/or text from Gemini using streaming API.
        Streams TEXT chunks to UI and uploads IMAGE parts to Open WebUI storage.
        Returns a short status message for the LLM.
        """
        if not self.valves.api_key:
            return (
                "Error: API key is missing. Please configure it in the tool settings."
            )

        if not isinstance(prompt, str):
            return "Error: The prompt must be a string."

        # Start status
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Generating (streaming) with Gemini…",
                        "done": False,
                    },
                }
            )

        try:
            client = genai.Client(api_key=EncryptedStr.decrypt(self.valves.api_key))

            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                )
            ]
            generate_content_config = types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"]
            )

            # Track results
            image_count = 0
            text_seen = False

            # Stream chunks
            for chunk in client.models.generate_content_stream(
                model=self.valves.model_name,
                contents=contents,
                config=generate_content_config,
            ):
                # Some chunks may be heartbeats/empty; guard checks
                if (
                    not chunk
                    or chunk.candidates is None
                    or not chunk.candidates
                    or chunk.candidates[0].content is None
                    or chunk.candidates[0].content.parts is None
                ):
                    continue

                parts = chunk.candidates[0].content.parts

                # 1) In the official example, chunk.text is printed; we directly push the visible text to the UI.
                if getattr(chunk, "text", None):
                    text_seen = True
                    if __event_emitter__ and chunk.text.strip():
                        await __event_emitter__(
                            {
                                "type": "message",
                                "data": {"content": chunk.text},
                            }
                        )

                # 2) IMAGE: Scan the inline_data within the parts and upload the bytes to Open WebUI.
                for part in parts:
                    inline = getattr(part, "inline_data", None)
                    if inline and inline.data:
                        mime_type: str = inline.mime_type or "image/png"
                        # Directly return the bytes
                        b64 = base64.b64encode(inline.data).decode("utf-8")
                        data_uri = f"data:{mime_type};base64,{b64}"

                        # Handled by Open WebUI for parsing and archiving
                        image_data, content_type = load_b64_image_data(data_uri)
                        url = upload_image(
                            __request__,
                            metadata={
                                "instances": {"prompt": prompt},
                                "parameters": {
                                    "sampleCount": 1,
                                    "outputOptions": {"mimeType": mime_type},
                                },
                            },
                            image_data=image_data,
                            content_type=content_type,
                            user=Users.get_user_by_id(__user__["id"]),
                        )
                        image_count += 1

                        # Send image message
                        if __event_emitter__:
                            await __event_emitter__(
                                {
                                    "type": "message",
                                    "data": {"content": f"![Generated Image]({url})"},
                                }
                            )

            # Done status
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Done. Images: {image_count} | Text: {'yes' if text_seen else 'no'}",
                            "done": True,
                        },
                    }
                )

            if image_count > 0:
                return "Notify the user that the image has been successfully generated"
            elif text_seen:
                return "Notify the user that only text was generated"
            else:
                return "Notify the user that no output was generated"

        except Exception as err:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"An error occurred: {err}",
                            "done": True,
                        },
                    }
                )
            return f"Tell the user: {err}"