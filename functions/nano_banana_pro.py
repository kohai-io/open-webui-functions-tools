"""
title: Google Gemini 3 Pro Image Generation (Chat Pipe)
author: open-webui
date: 2025-12-11
version: 3.5
license: MIT
description: A pipe for professional image generation using Google Gemini 3 Pro Image with high-resolution output (1K/2K/4K) and up to 14 reference images. Aspect ratio and resolution settings are sticky across conversation.
requirements: google-genai>=1.50.0, cryptography, requests, google-auth, c2pa-python
"""

import base64
import mimetypes
import os
import hashlib
import re
import requests
import time
import json
import io
from datetime import datetime
from typing import Optional, Any, List, Dict, Union, Callable, Awaitable
from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import core_schema

# C2PA imports (optional - will gracefully degrade if not installed)
try:
    from c2pa import Builder as C2paBuilder, C2paSignerInfo, C2paSigningAlg, Signer as C2paSigner
    C2PA_AVAILABLE = True
except ImportError:
    C2PA_AVAILABLE = False

# Google Gemini imports
from google import genai
from google.genai import types
from open_webui.main import (
    generate_chat_completions as main_generate_chat_completions,
)
from open_webui.routers.images import upload_image, get_image_data
from open_webui.models.users import Users
from open_webui.models.files import Files as FilesDB
from open_webui.storage.provider import Storage


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

    def get_decrypted(self) -> str:
        """Instance method to decrypt the string value"""
        return self.decrypt(self)

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


class Pipe:
    class Valves(BaseModel):
        """Pipeline configuration"""

        use_vertex_ai: bool = Field(
            default=True,
            description="Use Vertex AI authentication (True) or API key authentication (False)",
        )
        project_id: str = Field(
            default="",
            description="Google Cloud Project ID (for Vertex AI). Defaults to GOOGLE_CLOUD_PROJECT env var.",
        )
        location: str = Field(
            default="us-central1",
            description="Google Cloud location/region (for Vertex AI). Defaults to GOOGLE_CLOUD_LOCATION env var.",
        )
        SERVICE_ACCOUNT_JSON: EncryptedStr = Field(
            default="",
            description="Service account JSON key content (for Vertex AI authentication). Will be encrypted for security.",
        )
        SERVICE_ACCOUNT_PATH: str = Field(
            default="",
            description="Path to service account JSON file (for Vertex AI authentication). Alternative to SERVICE_ACCOUNT_JSON.",
        )
        api_key: EncryptedStr = Field(
            default="",
            description="Your Google AI API key (only used if use_vertex_ai=False)",
        )
        model_name: str = Field(
            default="gemini-3-pro-image-preview",
            description="The Google AI model name for image+text generation. Use 'gemini-3-pro-image-preview' for professional-grade 1K/2K/4K output with advanced reasoning, or 'gemini-2.5-flash-image' for faster generation.",
        )
        max_history_images: int = Field(
            default=14,
            description="Maximum number of previous images to include in context. Gemini 3 Pro supports up to 14 reference images (up to 6 objects + up to 5 humans). Use lower values for faster generation.",
        )
        enable_iterative: bool = Field(
            default=True,
            description="Enable iterative image generation using conversation history",
        )
        edit_mode: bool = Field(
            default=False,
            description="When true, uses only the most recent image (max 1) for consistent editing. When false, uses up to max_history_images for multi-reference generation. Set to False for Gemini 3 Pro's multi-image capability (up to 14 references).",
        )
        edit_guidance: str = Field(
            default=(
                "You are an image editor. Modify the provided image to satisfy the instruction. "
                "Preserve subject identity, composition, camera angle, style, and lighting unless explicitly asked to change them."
            ),
            description="Guidance text prepended to help the model perform consistent edits.",
        )
        debug: bool = Field(
            default=False, description="Enable verbose debug logging for this pipeline."
        )
        download_timeout: int = Field(
            default=20,
            description="Timeout (seconds) for downloading images via HTTP",
        )
        retry_attempts: int = Field(
            default=3,
            description="Max retry attempts for transient HTTP errors when downloading images.",
        )
        retry_backoff_base: float = Field(
            default=1.5,
            description="Exponential backoff multiplier between retries when downloading images.",
        )
        aspect_ratio: str = Field(
            default="",
            description="Optional output aspect ratio. Gemini 3 Pro supports: '1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '21:9'. If empty, model default is used.",
        )
        resolution: str = Field(
            default="",
            description="Optional output resolution for Gemini 3 Pro Image. Supported: '1K', '2K', '4K'. If empty, model default is used. Higher resolutions require more processing time.",
        )
        EMIT_INTERVAL: float = Field(
            default=0.5, description="Interval in seconds between status emissions"
        )
        ENABLE_STATUS_INDICATOR: bool = Field(
            default=True, description="Enable or disable status indicator emissions"
        )
        ENABLE_CUSTOM_FOLLOW_UPS: bool = Field(
            default=True,
            description="Enable custom follow-up generation optimized for image workflows",
        )
        FOLLOW_UP_MODEL: str = Field(
            default="gemini-2.5-flash",
            description="Model to use for generating follow-up prompts (e.g., gemini-2.5-flash)",
        )
        FOLLOW_UP_PROMPT_TEMPLATE: str = Field(
            default="",
            description="Custom prompt template for follow-up generation. Leave empty to use default image-optimized template.",
        )
        ENABLE_C2PA_SIGNING: bool = Field(
            default=False,
            description="Enable C2PA content provenance signing for generated images. Requires c2pa-python library and certificate configuration.",
        )
        C2PA_CERT_CONTENT: EncryptedStr = Field(
            default="",
            description="C2PA signing certificate content (.pem format). Will be encrypted for security. Alternative to C2PA_CERT_PATH.",
        )
        C2PA_CERT_PATH: str = Field(
            default="",
            description="Path to C2PA signing certificate (.pem file). Alternative to C2PA_CERT_CONTENT.",
        )
        C2PA_KEY_CONTENT: EncryptedStr = Field(
            default="",
            description="C2PA private key content (.pem format). Will be encrypted for security. Alternative to C2PA_KEY_PATH.",
        )
        C2PA_KEY_PATH: str = Field(
            default="",
            description="Path to C2PA private key (.pem file). Alternative to C2PA_KEY_CONTENT.",
        )
        C2PA_TSA_URL: str = Field(
            default="",
            description="Timestamp Authority URL for C2PA signing (optional but recommended for production).",
        )
        C2PA_TRAINING_POLICY: str = Field(
            default="notAllowed",
            description="AI training policy for generated images: 'allowed', 'notAllowed', or 'constrained'. Controls c2pa.training-mining assertion.",
        )

    def __init__(self):
        # Don't set self.type - use default pipe behavior (like veo_inline.py)
        self.id = "nano_banana_chat"
        self.name = "Gemini 3 Pro Image"
        self.valves = self.Valves()
        self.last_emit_time = 0

        # Auto-migrate old configurations to Gemini 3 Pro defaults
        migrations = []
        if self.valves.max_history_images == 5:
            self.valves.max_history_images = 14
            migrations.append("max_history_images: 5 → 14")
        if self.valves.edit_mode == True:
            self.valves.edit_mode = False
            migrations.append("edit_mode: True → False")
        if self.valves.model_name == "gemini-2.5-flash-image":
            self.valves.model_name = "gemini-3-pro-image-preview"
            migrations.append(
                "model_name: gemini-2.5-flash-image → gemini-3-pro-image-preview"
            )

        if migrations:
            print(f"[nano_banana_chat] Auto-migrated config: {', '.join(migrations)}")

    async def on_startup(self):
        print(
            f"[nano_banana_chat] Pipe loaded successfully - version 3.3 (Gemini 3 Pro Image + C2PA support)"
        )
        print(f"[nano_banana_chat] Pipe type: Default (inline display)")
        print(
            f"[nano_banana_chat] Features: 1K/2K/4K resolution, 10 aspect ratios, up to 14 reference images"
        )
        print(
            f"[nano_banana_chat] Sticky settings: History checked BEFORE prompt to avoid false positives"
        )
        print(f"[nano_banana_chat] Debug mode: {self.valves.debug}")
        print(
            f"[nano_banana_chat] Auth mode: {'Vertex AI' if self.valves.use_vertex_ai else 'API Key'}"
        )
        print(
            f"[nano_banana_chat] C2PA library: {'Available' if C2PA_AVAILABLE else 'Not installed'}"
        )
        if self.valves.ENABLE_C2PA_SIGNING:
            has_cert = bool(self.valves.C2PA_CERT_CONTENT or self.valves.C2PA_CERT_PATH)
            has_key = bool(self.valves.C2PA_KEY_CONTENT or self.valves.C2PA_KEY_PATH)
            cert_source = "content" if self.valves.C2PA_CERT_CONTENT else ("path" if self.valves.C2PA_CERT_PATH else "missing")
            key_source = "content" if self.valves.C2PA_KEY_CONTENT else ("path" if self.valves.C2PA_KEY_PATH else "missing")
            
            if has_cert and has_key:
                print(f"[nano_banana_chat] C2PA signing: Enabled (cert: {cert_source}, key: {key_source})")
            else:
                print(f"[nano_banana_chat] C2PA signing: Enabled but missing credentials (cert: {cert_source}, key: {key_source})")
        else:
            print(f"[nano_banana_chat] C2PA signing: Disabled")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    def _debug(self, msg: str) -> None:
        if getattr(self.valves, "debug", False):
            print(f"[DEBUG] {msg}")

    async def emit_status(
        self,
        event_emitter: Optional[Callable[[dict], Awaitable[None]]],
        level: str,
        message: str,
        done: bool = False,
    ) -> None:
        """Emit status updates to Open WebUI"""
        if not event_emitter:
            return

        if not self.valves.ENABLE_STATUS_INDICATOR:
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
                self._debug(f"Failed to emit status: {e}")

    IMAGE_SAFETY_FILTER_CODES = {
        "58061214": "child",
        "17301594": "child",
        "29310472": "celebrity",
        "15236754": "celebrity",
        "62263041": "dangerous",
        "57734940": "hate",
        "22137204": "hate",
        "74803281": "other",
        "29578790": "other",
        "42876398": "other",
        "39322892": "peopleFace",
        "92201652": "personalInfo",
        "89371032": "prohibited",
        "49114662": "prohibited",
        "72817394": "prohibited",
        "90789179": "sexual",
        "63429089": "sexual",
        "43188360": "sexual",
        "78610348": "toxic",
        "61493863": "violence",
        "56562880": "violence",
        "32635315": "vulgar",
        "64151117": "celebrityOrChild",
    }

    IMAGE_SAFETY_CATEGORY_GUIDANCE = {
        "child": {
            "title": "Child Safety Restriction",
            "description": "Your request appears to involve child-related content that isn't allowed with the current safety settings.",
            "suggestions": [
                "Remove references to minors/children, age cues, school settings, or child-like descriptions",
                "Use a non-human character or an abstract/illustrated style",
                "Focus on environments or objects rather than people",
            ],
        },
        "celebrity": {
            "title": "Celebrity Depiction Restriction",
            "description": "Your request may involve a photorealistic depiction of a celebrity, which can be restricted.",
            "suggestions": [
                "Avoid naming real celebrities or public figures",
                "Use a fictional character instead",
                "Use non-photorealistic styles (illustration, cartoon, painterly)",
            ],
        },
        "dangerous": {
            "title": "Dangerous Content Detected",
            "description": "Your prompt may include dangerous content.",
            "suggestions": [
                "Remove references to weapons, explosives, self-harm, or instructions",
                "Focus on safe, non-harmful scenarios",
            ],
        },
        "hate": {
            "title": "Hate Content Detected",
            "description": "Your prompt may include hateful or discriminatory content.",
            "suggestions": [
                "Remove slurs, hate symbols, or targeting of protected groups",
                "Use neutral, respectful language",
            ],
        },
        "personalInfo": {
            "title": "Personal Information Detected",
            "description": "Your prompt may include personally identifying information.",
            "suggestions": [
                "Remove addresses, phone numbers, emails, ID numbers, payment details",
                "Replace specifics with generic placeholders",
            ],
        },
        "prohibited": {
            "title": "Prohibited Content Detected",
            "description": "Your request may involve prohibited content.",
            "suggestions": [
                "Remove requests for illegal or disallowed content",
                "Try rephrasing with a safer, non-sensitive scenario",
            ],
        },
        "sexual": {
            "title": "Sexual Content Detected",
            "description": "Your prompt may include sexual or explicit content.",
            "suggestions": [
                "Remove sexually explicit descriptions",
                "Keep the request suitable for all audiences",
            ],
        },
        "toxic": {
            "title": "Toxic Content Detected",
            "description": "Your prompt may include toxic or harassing content.",
            "suggestions": [
                "Remove insults, threats, or aggressive language",
                "Use constructive and non-hostile phrasing",
            ],
        },
        "violence": {
            "title": "Violence Content Detected",
            "description": "Your prompt may include violence-related content.",
            "suggestions": [
                "Avoid depictions of injury, gore, fighting, or weapons",
                "Focus on peaceful alternatives",
            ],
        },
        "vulgar": {
            "title": "Vulgar Content Detected",
            "description": "Your prompt may include vulgar content.",
            "suggestions": [
                "Remove explicit profanity or crude descriptions",
                "Use neutral, non-explicit wording",
            ],
        },
        "peopleFace": {
            "title": "People/Face Restriction",
            "description": "Your request may involve generating a person or face when not allowed under the current request settings.",
            "suggestions": [
                "Use an artistic or animated style instead of photorealistic",
                "Focus on environments/objects or use silhouettes",
            ],
        },
        "celebrityOrChild": {
            "title": "Celebrity or Child Restriction",
            "description": "Your request may involve restricted depictions of a celebrity or a child.",
            "suggestions": [
                "Avoid naming real people and avoid child-related descriptors",
                "Use a fictional subject or non-photorealistic style",
            ],
        },
        "other": {
            "title": "Safety Filter Triggered",
            "description": "Your request was blocked by a safety filter.",
            "suggestions": [
                "Remove potentially sensitive or disallowed details",
                "Try a more general, non-sensitive prompt",
            ],
        },
    }

    def _extract_support_codes(self, message: str) -> list[str]:
        if not message or not isinstance(message, str):
            return []
        return re.findall(r"(?:support code|code)s?\s*:?\s*(\d{8})", message, re.IGNORECASE)

    def _format_policy_guardrails_message(
        self,
        *,
        title: str,
        description: str,
        support_code: Optional[str] = None,
        category: Optional[str] = None,
        phase: Optional[str] = None,
    ) -> str:
        msg = f"🛡️ **{title}**\n\n{description}\n\n"
        if category:
            msg += f"**Safety Category:** `{category}`\n"
        if support_code:
            msg += f"**Support Code:** `{support_code}`\n"
        if phase:
            msg += f"**Filtered:** `{phase}`\n"
        if support_code:
            msg += "\nYou can report the support code to Google Cloud support if you believe this is a mistake."
        return msg

    def _detect_image_safety_filter(
        self,
        message: str,
        *,
        phase: Optional[str] = None,
    ) -> Optional[str]:
        if not message or not isinstance(message, str):
            return None

        codes = self._extract_support_codes(message)
        if not codes:
            return None

        for code in codes:
            category = self.IMAGE_SAFETY_FILTER_CODES.get(code) or "other"
            guidance = self.IMAGE_SAFETY_CATEGORY_GUIDANCE.get(category) or self.IMAGE_SAFETY_CATEGORY_GUIDANCE.get("other")
            title = guidance.get("title", "Safety Filter Triggered")
            description = guidance.get("description", "Your request was blocked by a safety filter.")
            suggestions = guidance.get("suggestions") or []

            formatted = self._format_policy_guardrails_message(
                title=title,
                description=description,
                support_code=code,
                category=category,
                phase=phase,
            )
            if suggestions:
                formatted = (
                    formatted
                    + "\n\n**Suggestions to resolve:**\n"
                    + "\n".join(f"- {s}" for s in suggestions)
                )
            return formatted

        return None

    def _get_finish_reason(self, candidate: Any) -> str:
        fr = getattr(candidate, "finish_reason", None)
        if fr is None:
            fr = getattr(candidate, "finishReason", None)
        return str(fr) if fr is not None else ""

    def _is_model_refusal(self, *, finish_reason: str, text: str, image_count: int) -> bool:
        if image_count > 0:
            return False
        fr = (finish_reason or "").upper()
        if fr != "STOP":
            return False
        t = (text or "").lower()
        if not t:
            return False
        refusal_markers = [
            "can't help",
            "cannot help",
            "can't comply",
            "cannot comply",
            "can't generate",
            "cannot generate",
            "can't create",
            "cannot create",
            "unable to",
            "i'm not able",
            "i am not able",
        ]
        safety_markers = ["policy", "safety", "unsafe", "guideline", "not allowed", "prohibited"]
        return any(m in t for m in refusal_markers) and any(m in t for m in safety_markers)

    async def _emit_policy_block(
        self,
        body: dict,
        event_emitter: Optional[Any],
        message: str,
    ) -> str:
        await self.emit_status(event_emitter, "error", "❌ Blocked by safety policy", True)
        body["messages"].append({"role": "assistant", "content": message})
        return message

    def _create_c2pa_manifest(self, prompt: str, user: dict) -> dict:
        """
        Create C2PA manifest for AI-generated images.
        
        Includes:
        - AI generation assertion
        - Training/mining policy
        - Creator information
        - Generation parameters
        """
        manifest = {
            "claim_generator": "OpenWebUI/NanoBananaPro",
            "claim_generator_info": [
                {
                    "name": "ITV Agent Hub - Gemini 3 Pro Image Pipeline",
                    "version": "3.1"
                }
            ],
            "title": f"AI Generated Image - {datetime.utcnow().isoformat()}",
            "format": "image/jpeg",
            "assertions": [
                {
                    "label": "c2pa.ai-generated",
                    "data": {
                        "generator": self.valves.model_name,
                        "prompt": prompt if prompt else "Not provided",
                        "date": datetime.utcnow().isoformat(),
                        "pipeline": "nano_banana_pro"
                    }
                },
                {
                    "label": "c2pa.training-mining",
                    "data": {
                        "entries": {
                            "c2pa.ai_training": {
                                "use": self.valves.C2PA_TRAINING_POLICY
                            },
                            "c2pa.ai_inference": {
                                "use": self.valves.C2PA_TRAINING_POLICY
                            },
                            "c2pa.data_mining": {
                                "use": self.valves.C2PA_TRAINING_POLICY
                            }
                        }
                    }
                }
            ]
        }
        
        # Add creator attribution
        if user:
            # Extract user name with fallback chain
            user_name = user.get("name") or user.get("email") or user.get("username") or "Unknown"
            user_id = user.get("id", "unknown")
            
            manifest["assertions"].append({
                "label": "stds.schema-org.CreativeWork",
                "data": {
                    "@context": "https://schema.org",
                    "@type": "ImageObject",
                    "creator": {
                        "@type": "Person",
                        "name": user_name,
                        "identifier": user_id
                    }
                }
            })
        
        return manifest

    def _sign_image_with_c2pa(
        self, 
        image_data: bytes, 
        mime_type: str,
        prompt: str,
        user: dict
    ) -> bytes:
        """
        Sign image data with C2PA manifest.
        
        Args:
            image_data: Raw image bytes
            mime_type: MIME type (e.g., 'image/jpeg', 'image/png')
            prompt: Generation prompt
            user: User information
        
        Returns:
            Signed image bytes (or original if signing fails/disabled)
        """
        
        # Check if C2PA is available and enabled
        if not C2PA_AVAILABLE:
            self._debug("C2PA library not installed - skipping signing")
            return image_data
        
        if not self.valves.ENABLE_C2PA_SIGNING:
            self._debug("C2PA signing disabled in settings")
            return image_data
        
        # Check for required certificate/key configuration (content OR path)
        has_cert = bool(self.valves.C2PA_CERT_CONTENT or self.valves.C2PA_CERT_PATH)
        has_key = bool(self.valves.C2PA_KEY_CONTENT or self.valves.C2PA_KEY_PATH)
        
        if not has_cert or not has_key:
            self._debug("C2PA certificate/key not configured - skipping signing")
            return image_data
        
        try:
            self._debug(f"Signing image with C2PA (size: {len(image_data)} bytes)")
            
            # Get certificate and key as bytes (content takes precedence)
            if self.valves.C2PA_CERT_CONTENT:
                cert_data = self.valves.C2PA_CERT_CONTENT.get_decrypted().encode('utf-8')
                self._debug("Using C2PA certificate from content (encrypted)")
            else:
                cert_path = self.valves.C2PA_CERT_PATH
                if not os.path.exists(cert_path):
                    self._debug(f"C2PA certificate not found: {cert_path}")
                    return image_data
                with open(cert_path, "rb") as f:
                    cert_data = f.read()
                self._debug(f"Using C2PA certificate from path: {cert_path}")
            
            if self.valves.C2PA_KEY_CONTENT:
                key_data = self.valves.C2PA_KEY_CONTENT.get_decrypted().encode('utf-8')
                self._debug("Using C2PA private key from content (encrypted)")
            else:
                key_path = self.valves.C2PA_KEY_PATH
                if not os.path.exists(key_path):
                    self._debug(f"C2PA private key not found: {key_path}")
                    return image_data
                with open(key_path, "rb") as f:
                    key_data = f.read()
                self._debug(f"Using C2PA private key from path: {key_path}")
            
            # Create manifest JSON
            manifest = self._create_c2pa_manifest(prompt, user)
            manifest_json = json.dumps(manifest)
            
            # Create signer info using the current API (v0.27.1)
            # Build kwargs conditionally - only include ta_url if configured
            signer_kwargs = {
                "alg": C2paSigningAlg.ES256,  # Elliptic Curve (most common for ES256 keys)
                "sign_cert": cert_data,
                "private_key": key_data,
            }
            
            # Add timestamp authority only if configured (optional parameter)
            if self.valves.C2PA_TSA_URL:
                signer_kwargs["ta_url"] = self.valves.C2PA_TSA_URL
                self._debug(f"Using TSA: {self.valves.C2PA_TSA_URL}")
            else:
                self._debug("No TSA configured - signing without timestamp")
            
            signer_info = C2paSignerInfo(**signer_kwargs)
            signer = C2paSigner.from_info(signer_info)
            
            # Sign the image using Builder context manager
            source_stream = io.BytesIO(image_data)
            dest_stream = io.BytesIO()
            
            with C2paBuilder(manifest_json) as builder:
                manifest_bytes = builder.sign(
                    signer=signer,
                    format=mime_type,
                    source=source_stream,
                    dest=dest_stream
                )
            
            # Extract signed data
            signed_data = dest_stream.getvalue()
            
            self._debug(f"C2PA signing successful! Manifest size: {len(manifest_bytes)} bytes")
            self._debug(f"Signed image size: {len(signed_data)} bytes (original: {len(image_data)} bytes)")
            
            return signed_data
            
        except Exception as e:
            self._debug(f"C2PA signing failed: {e}")
            import traceback
            self._debug(f"C2PA traceback: {traceback.format_exc()}")
            # Return original unsigned image on error
            return image_data

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Any] = None,
        __event_call__: Optional[Any] = None,
        __request__: Optional[Any] = None,
    ) -> Optional[str]:
        """Main entrypoint for Gemini 3 Pro Image generation.

        Returns the response content as a string for inline UI rendering (following veo_inline.py pattern).
        Also appends the message to body["messages"] for chat history persistence.

        Error-handling strategy:
        - Validate decrypted API key up front and return a clear user-facing error.
        - During streaming, capture partial output and return it with an explicit error if an exception occurs mid-stream.
        - When downloading images, use retry/backoff and return None on failure to allow graceful degradation.
        """

        # Check if this is a follow-up generation request
        metadata = body.get("metadata", {})
        is_follow_up_task = metadata.get("task") == "FOLLOW_UP_GENERATION"

        if is_follow_up_task and self.valves.ENABLE_CUSTOM_FOLLOW_UPS:
            self._debug(
                "Detected follow-up generation request - using custom image follow-ups"
            )
            return await self._generate_image_follow_ups(body, __event_emitter__)

        await self.emit_status(
            __event_emitter__, "info", "Initializing Gemini image generator..."
        )

        # Validate authentication configuration
        if self.valves.use_vertex_ai:
            # Check Vertex AI configuration
            project_id = self.valves.project_id or os.getenv("GOOGLE_CLOUD_PROJECT", "")
            if not project_id:
                error_msg = "❌ Error: Google Cloud Project ID not configured. Please set project_id in pipeline settings or GOOGLE_CLOUD_PROJECT environment variable."
                await self.emit_status(__event_emitter__, "error", error_msg, True)
                body["messages"].append({"role": "assistant", "content": error_msg})
                return error_msg

            # Check if service account credentials are provided
            if (
                not self.valves.SERVICE_ACCOUNT_JSON
                and not self.valves.SERVICE_ACCOUNT_PATH
            ):
                error_msg = "❌ Error: Service account credentials not configured for Vertex AI. Please provide SERVICE_ACCOUNT_JSON or SERVICE_ACCOUNT_PATH in pipeline settings, or set use_vertex_ai=False to use API key authentication."
                await self.emit_status(__event_emitter__, "error", error_msg, True)
                body["messages"].append({"role": "assistant", "content": error_msg})
                return error_msg
        else:
            # Check API key configuration
            decrypted_key_check = EncryptedStr.decrypt(self.valves.api_key)
            if not decrypted_key_check:
                error_msg = "❌ Error: Google AI API key not configured. Please set your API key in the pipeline settings."
                await self.emit_status(__event_emitter__, "error", error_msg, True)
                body["messages"].append({"role": "assistant", "content": error_msg})
                return error_msg

        # Setup for credential cleanup
        temp_creds_path = None
        old_creds = None

        try:
            # Extract messages and a guarded prompt from the latest USER message
            messages = body.get("messages", [])

            if not messages:
                error_msg = "❌ Error: No messages provided."
                body["messages"].append({"role": "assistant", "content": error_msg})
                return error_msg

            await self.emit_status(
                __event_emitter__, "info", "Processing your request..."
            )

            prompt, user_has_image = self._extract_latest_user_prompt_and_image_flag(
                messages
            )
            self._debug(
                f"Extracted prompt: '{prompt[:100]}...' (has_image={user_has_image})"
            )

            # Allow image-only user message by falling back to edit_guidance
            if (
                not prompt
                and user_has_image
                and getattr(self.valves, "edit_guidance", "")
            ):
                prompt = self.valves.edit_guidance
                self._debug(f"Using edit_guidance as prompt")

            if not prompt:
                # Check if this is a system follow-up instruction (should be silently ignored)
                last_user_msg = None
                for msg in reversed(messages or []):
                    if msg.get("role") == "user":
                        last_user_msg = msg
                        break

                is_system_followup = False
                if last_user_msg:
                    content = last_user_msg.get("content", "")
                    if isinstance(content, str):
                        text = content.lower().strip()
                        # Detect system follow-up patterns
                        if re.search(
                            r"suggest.*follow[- ]up.*(question|prompt|topic)", text
                        ):
                            is_system_followup = True
                        elif re.match(
                            r"^(please\s+)?(suggest|provide|generate)\s+\d*\s*follow[- ]up",
                            text,
                        ):
                            is_system_followup = True

                if is_system_followup:
                    # Silently ignore system follow-up instructions - don't generate or show error
                    self._debug(
                        "Ignoring system follow-up instruction (not an image generation request)"
                    )
                    return body  # Return without adding error message
                else:
                    # Real error - no valid prompt found
                    error_msg = "❌ Error: No prompt provided for image generation."
                    body["messages"].append({"role": "assistant", "content": error_msg})
                    return body

            # Build Gemini content with conversation history
            await self.emit_status(
                __event_emitter__,
                "info",
                "Building context from conversation history...",
            )
            contents = self._build_gemini_context(messages, prompt, __request__)

            # Generate with Gemini - create client based on authentication mode
            if self.valves.use_vertex_ai:
                project_id = self.valves.project_id or os.getenv(
                    "GOOGLE_CLOUD_PROJECT", ""
                )
                location = self.valves.location or os.getenv(
                    "GOOGLE_CLOUD_LOCATION", "us-central1"
                )

                self._debug(
                    f"Using Vertex AI with service account authentication, project={project_id}, location={location}"
                )

                # For Vertex AI with service account, set credentials via environment
                # Keep them set for the entire operation duration
                import tempfile

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
                decrypted_key = EncryptedStr.decrypt(self.valves.api_key)
                self._debug("Using API key authentication")
                client = genai.Client(api_key=decrypted_key)

            # Build config with optional aspect ratio and resolution per Gemini 3 Pro docs
            # Use uppercase casing per Gemini 3 Pro docs: ['IMAGE', 'TEXT']
            _cfg_kwargs: dict = {"response_modalities": ["IMAGE", "TEXT"]}
            _ar = getattr(self.valves, "aspect_ratio", "") or ""
            _res = getattr(self.valves, "resolution", "") or ""
            self._debug(f"Initial valve aspect_ratio: '{_ar}', resolution: '{_res}'")
            self._debug(f"User prompt: {prompt[:200]}...")

            # Check sticky settings from history FIRST to avoid false positives from keywords
            if not _ar or not _res:
                try:
                    sticky_ar, sticky_res = self._extract_sticky_settings_from_history(
                        messages
                    )
                    if not _ar and sticky_ar:
                        _ar = sticky_ar
                        self._debug(f"✓ Using sticky aspect ratio from history: {_ar}")
                    if not _res and sticky_res:
                        _res = sticky_res
                        self._debug(f"✓ Using sticky resolution from history: {_res}")
                except Exception as e:
                    self._debug(f"Failed to extract sticky settings from history: {e}")

            # Only extract from current prompt if no valve AND no sticky setting found
            # This prevents false positives like "tall hat" triggering portrait mode
            if not _ar:
                try:
                    _from_prompt = self._extract_aspect_ratio_from_prompt(prompt)
                    if _from_prompt:
                        _ar = _from_prompt
                        self._debug(f"✓ Extracted aspect ratio from prompt: {_ar}")
                    else:
                        self._debug(f"No aspect ratio detected in prompt")
                except Exception as e:
                    self._debug(f"Failed to extract aspect ratio from prompt: {e}")
                    import traceback

                    self._debug(traceback.format_exc())

            if not _res:
                try:
                    _res_from_prompt = self._extract_resolution_from_prompt(prompt)
                    if _res_from_prompt:
                        _res = _res_from_prompt
                        self._debug(f"✓ Extracted resolution from prompt: {_res}")
                    else:
                        self._debug(f"No resolution detected in prompt")
                except Exception as e:
                    self._debug(f"Failed to extract resolution from prompt: {e}")
                    import traceback

                    self._debug(traceback.format_exc())

            # Try to apply aspect_ratio and/or resolution using ImageConfig per official docs
            if (isinstance(_ar, str) and _ar.strip()) or (
                isinstance(_res, str) and _res.strip()
            ):
                # Check multiple possible locations for ImageConfig
                image_config_class = None

                # Approach 1: types.ImageConfig (per official docs)
                if hasattr(types, "ImageConfig"):
                    image_config_class = types.ImageConfig
                    self._debug(f"Found types.ImageConfig")
                # Approach 2: Check if it's in genai module directly
                elif hasattr(genai, "ImageConfig"):
                    image_config_class = genai.ImageConfig
                    self._debug(f"Found genai.ImageConfig")
                # Approach 3: Check genai.types
                elif hasattr(genai.types, "ImageConfig"):
                    image_config_class = genai.types.ImageConfig
                    self._debug(f"Found genai.types.ImageConfig")
                else:
                    self._debug(
                        f"❌ ImageConfig not found in SDK - version may be outdated"
                    )
                    self._debug(
                        f"SDK version: {getattr(genai, '__version__', 'unknown')}"
                    )
                    self._debug(
                        f"Available image configs: {[a for a in dir(types) if 'ImageConfig' in a]}"
                    )

                if image_config_class:
                    try:
                        # Build ImageConfig with available parameters
                        image_config_kwargs = {}
                        if _ar.strip():
                            image_config_kwargs["aspect_ratio"] = _ar.strip()
                        if _res.strip():
                            # Gemini 3 Pro uses image_size parameter for resolution
                            image_config_kwargs["image_size"] = _res.strip()

                        image_config = image_config_class(**image_config_kwargs)
                        _cfg_kwargs["image_config"] = image_config
                        _cfg_kwargs["response_modalities"] = ["IMAGE"]

                        config_desc = ", ".join(
                            [f"{k}={v}" for k, v in image_config_kwargs.items()]
                        )
                        self._debug(f"✓ Applied ImageConfig with {config_desc}")
                    except Exception as e:
                        self._debug(f"❌ Failed to create ImageConfig: {e}")
                        import traceback

                        self._debug(traceback.format_exc())
                else:
                    self._debug(
                        f"⚠️ Image settings detected but SDK doesn't support ImageConfig"
                    )
                    self._debug(f"Consider upgrading google-genai package")

            generate_content_config = types.GenerateContentConfig(**_cfg_kwargs)
            self._debug(f"Final GenerateContentConfig: {generate_content_config}")

            await self.emit_status(
                __event_emitter__, "info", "Generating image with Gemini..."
            )

            # Generate content using streaming
            result_content = []
            image_count = 0
            text_seen = False
            generated_image_markdowns: List[str] = []

            try:
                for chunk in client.models.generate_content_stream(
                    model=self.valves.model_name,
                    contents=contents,
                    config=generate_content_config,
                ):
                    self._debug(f"Got chunk: {chunk}")

                    prompt_feedback = getattr(chunk, "prompt_feedback", None)
                    blocked_reason = getattr(prompt_feedback, "blocked_reason", None) if prompt_feedback else None
                    if blocked_reason:
                        msg = str(blocked_reason)
                        safety_msg = self._detect_image_safety_filter(msg, phase="prompt")
                        if not safety_msg:
                            safety_msg = self._format_policy_guardrails_message(
                                title="Prompt Blocked",
                                description="Your prompt was blocked by a safety filter. Please rephrase and try again.",
                                phase="prompt",
                            )
                        return await self._emit_policy_block(body, __event_emitter__, safety_msg)

                    # Guard checks like the working tool
                    if (
                        not chunk
                        or chunk.candidates is None
                        or not chunk.candidates
                        or chunk.candidates[0].content is None
                        or chunk.candidates[0].content.parts is None
                    ):
                        self._debug("Skipping empty/invalid chunk")
                        continue

                    parts = chunk.candidates[0].content.parts
                    self._debug(f"Processing {len(parts)} parts in chunk")

                    finish_reason = self._get_finish_reason(chunk.candidates[0])
                    if finish_reason:
                        fr_upper = finish_reason.upper()
                        if fr_upper in {"IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "PROHIBITED_CONTENT", "SAFETY"}:
                            err_text = getattr(chunk, "text", "") or ""
                            safety_msg = self._detect_image_safety_filter(err_text, phase="output")
                            if not safety_msg:
                                safety_msg = self._format_policy_guardrails_message(
                                    title="Generated Image Blocked",
                                    description="The generated output was blocked by a safety filter. Please adjust your prompt and try again.",
                                    phase="output",
                                )
                            return await self._emit_policy_block(body, __event_emitter__, safety_msg)

                    # Process text from chunk.text (like working tool)
                    if getattr(chunk, "text", None):
                        text_seen = True
                        self._debug(f"Chunk text: {chunk.text[:100]}...")
                        result_content.append(chunk.text)

                        if self._is_model_refusal(
                            finish_reason=finish_reason,
                            text=chunk.text,
                            image_count=image_count,
                        ):
                            safety_msg = self._detect_image_safety_filter(chunk.text, phase="prompt")
                            if not safety_msg:
                                safety_msg = self._format_policy_guardrails_message(
                                    title="Model Refused the Request",
                                    description="The model refused to generate an image for this prompt due to safety policies. Please rephrase to remove sensitive or disallowed content and try again.",
                                    phase="prompt",
                                )
                            return await self._emit_policy_block(body, __event_emitter__, safety_msg)

                    # Process images from parts
                    for i, part in enumerate(parts):
                        inline = getattr(part, "inline_data", None)
                        self._debug(f"Part {i} inline_data: {bool(inline)}")
                        if inline and inline.data:
                            mime_type = inline.mime_type or "image/png"
                            self._debug(
                                f"Image found - MIME: {mime_type}, size: {len(inline.data)} bytes"
                            )
                            b64 = base64.b64encode(inline.data).decode("utf-8")
                            data_uri = f"data:{mime_type};base64,{b64}"

                            # Upload image using Open WebUI's system (like the working tool)
                            try:
                                await self.emit_status(
                                    __event_emitter__,
                                    "info",
                                    f"Processing generated image {image_count + 1}...",
                                )
                                image_data, content_type = get_image_data(data_uri)
                                
                                # Sign with C2PA if enabled
                                if self.valves.ENABLE_C2PA_SIGNING:
                                    await self.emit_status(
                                        __event_emitter__,
                                        "info",
                                        f"Signing image {image_count + 1} with C2PA...",
                                    )
                                    image_data = self._sign_image_with_c2pa(
                                        image_data=image_data,
                                        mime_type=mime_type,
                                        prompt=prompt,
                                        user=__user__
                                    )
                                
                                await self.emit_status(
                                    __event_emitter__,
                                    "info",
                                    f"Uploading image {image_count + 1}...",
                                )
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
                                self._debug(f"Image uploaded to: {url}")
                                image_count += 1
                                # Collect for persistence in final message
                                generated_image_markdowns.append(
                                    f"![Generated Image]({url})"
                                )
                                
                                # Send image message via event emitter for immediate UI rendering
                                # This is critical for 2K/4K images to avoid memory/rendering issues
                                if __event_emitter__:
                                    await __event_emitter__(
                                        {
                                            "type": "message",
                                            "data": {
                                                "content": f"![Generated Image]({url})"
                                            },
                                        }
                                    )
                            except Exception as e:
                                self._debug(f"Failed to upload image: {e}")
                                # Fallback to base64 if upload fails
                                image_count += 1
                                # Persist base64 as a last resort so it appears in history
                                generated_image_markdowns.append(
                                    f"![Generated Image]({data_uri})"
                                )
                                
                                # Send base64 image via event emitter for immediate UI rendering
                                if __event_emitter__:
                                    await __event_emitter__(
                                        {
                                            "type": "message",
                                            "data": {
                                                "content": f"![Generated Image]({data_uri})"
                                            },
                                        }
                                    )
            except Exception as e:
                # If streaming fails mid-way, emit partial results if any, else a clear error
                partial_text = "\n\n".join(str(c) for c in result_content if c).strip()
                if generated_image_markdowns:
                    images_block = "\n".join(generated_image_markdowns)
                else:
                    images_block = ""
                tip = " If this persists, please try again later or adjust your prompt."
                if partial_text or images_block:
                    combined = "\n\n".join(
                        [s for s in [partial_text, images_block] if s]
                    )
                    safety_msg = self._detect_image_safety_filter(str(e), phase="prompt")
                    if safety_msg:
                        return await self._emit_policy_block(body, __event_emitter__, safety_msg)
                    error_content = f"⚠️ Partial response before error:\n\n{combined}\n\n❌ Error: {str(e)}.{tip}"
                    body["messages"].append(
                        {
                            "role": "assistant",
                            "content": error_content,
                        }
                    )
                    return error_content
                else:
                    safety_msg = self._detect_image_safety_filter(str(e), phase="prompt")
                    if safety_msg:
                        return await self._emit_policy_block(body, __event_emitter__, safety_msg)
                    error_content = f"❌ Error: {str(e)}.{tip}"
                    body["messages"].append(
                        {
                            "role": "assistant",
                            "content": error_content,
                        }
                    )
                    return error_content

            self._debug(
                f"Generated {image_count} images, {len(result_content)} content items"
            )

            await self.emit_status(__event_emitter__, "info", "Finalizing response...")

            if image_count > 0:
                if not result_content or not any(
                    isinstance(c, str) and c.strip() for c in result_content
                ):
                    result_content.insert(0, "Here's your generated image:")
            elif not result_content:
                result_content = [
                    "I wasn't able to generate any content for your request."
                ]

            # Return in pipeline format
            response_content = "\n\n".join(str(c) for c in result_content if c)
            # Append any generated image links so they're persisted in chat history
            if generated_image_markdowns:
                if response_content:
                    response_content = (
                        response_content + "\n\n" + "\n".join(generated_image_markdowns)
                    )
                else:
                    response_content = "\n".join(generated_image_markdowns)
            self._debug(f"Final response content length: {len(response_content)}")
            self._debug(f"Final response preview: {response_content[:200]}...")

            # Emit completion status BEFORE returning (otherwise it never runs)
            await self.emit_status(
                __event_emitter__, "info", "Image generation complete!", True
            )

            # Append the assistant message for history, but return the content string for inline handling
            # (Following veo_inline.py pattern for proper UI rendering)
            body["messages"].append({"role": "assistant", "content": response_content})
            self._debug(
                f"Appended assistant message with {len(response_content)} chars"
            )
            self._debug(f"Returning response_content string for inline display")

            return response_content  # Return STRING for inline handling, not body dict

        except Exception as e:
            error_msg = f"Error during image generation: {str(e)}"
            self._debug(f"Exception occurred: {error_msg}")
            import traceback

            self._debug(f"Traceback: {traceback.format_exc()}")

            await self.emit_status(__event_emitter__, "error", error_msg, True)

            error_content = f"❌ Error: {error_msg}"
            body["messages"].append({"role": "assistant", "content": error_content})
            return error_content
        finally:
            # Clean up credentials
            if old_creds is not None:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = old_creds
            elif "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
                os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

            if temp_creds_path:
                try:
                    os.unlink(temp_creds_path)
                except:
                    pass

    async def _generate_image_follow_ups(
        self, body: dict, event_emitter: Optional[Any] = None
    ) -> dict:
        """Generate image-specific follow-up prompts using Gemini."""
        try:
            messages = body.get("messages", [])

            # Get the default or custom template
            if self.valves.FOLLOW_UP_PROMPT_TEMPLATE:
                template = self.valves.FOLLOW_UP_PROMPT_TEMPLATE
            else:
                # Default image-optimized template
                template = """### Task:
Suggest 3-5 relevant follow-up image generation prompts that the user might want to create next, based on the image they just generated and the conversation history.

### Guidelines:
- Focus on image-specific concepts: variations, different styles, zoom levels, perspectives, lighting, color schemes, compositions, subjects, seasons, moods
- Suggest prompts that would create compelling visual variations or extensions related to the current image
- Write from the user's perspective as requests to generate images
- Keep prompts concise but descriptive with visual details
- Consider variations in: artistic style, medium (photo, painting, illustration), subject matter, setting, mood, time period
- Suggest creative transformations: same subject in different context, different angle/viewpoint, alternative color palette
- Ensure suggestions are directly related to the discussed topic
- Use the conversation's primary language; default to English if multilingual
- Response must be a JSON array of strings, no extra text

### Output Format:
JSON: { "follow_ups": ["Prompt 1", "Prompt 2", "Prompt 3"] }

### Examples of Good Image Follow-ups:
- "Generate the same subject but in watercolor painting style with pastel colors"
- "Create a close-up portrait version focusing on facial details and expression"
- "Show the scene at sunset with warm golden hour lighting"
- "Transform into a minimalist line art illustration"

### Chat History (last 6 messages):
{{MESSAGES:END:6}}"""

            # Replace template variables
            recent_messages = messages[-6:] if len(messages) > 6 else messages
            messages_text = "\n".join(
                [
                    f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                    for msg in recent_messages
                ]
            )

            prompt = template.replace("{{MESSAGES:END:6}}", messages_text)

            self._debug(
                f"Generating image follow-ups using model: {self.valves.FOLLOW_UP_MODEL}"
            )

            # Use Google GenAI to generate follow-ups
            try:
                if self.valves.use_vertex_ai:
                    project_id = self.valves.project_id or os.getenv(
                        "GOOGLE_CLOUD_PROJECT", ""
                    )
                    location = self.valves.location or os.getenv(
                        "GOOGLE_CLOUD_LOCATION", "us-central1"
                    )

                    import tempfile

                    temp_creds_path = None
                    old_creds = None

                    if self.valves.SERVICE_ACCOUNT_JSON:
                        service_account_json = (
                            self.valves.SERVICE_ACCOUNT_JSON.get_decrypted()
                        )
                        service_account_info = json.loads(service_account_json)
                        with tempfile.NamedTemporaryFile(
                            mode="w", suffix=".json", delete=False
                        ) as f:
                            json.dump(service_account_info, f)
                            temp_creds_path = f.name
                        old_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_creds_path
                    else:
                        old_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
                            self.valves.SERVICE_ACCOUNT_PATH
                        )

                    try:
                        client = genai.Client(
                            vertexai=True, project=project_id, location=location
                        )
                        response = client.models.generate_content(
                            model=self.valves.FOLLOW_UP_MODEL, contents=prompt
                        )
                    finally:
                        if old_creds is not None:
                            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = old_creds
                        elif "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
                            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
                        if temp_creds_path:
                            try:
                                os.unlink(temp_creds_path)
                            except:
                                pass
                else:
                    api_key = self.valves.api_key.get_decrypted()
                    client = genai.Client(api_key=api_key)
                    response = client.models.generate_content(
                        model=self.valves.FOLLOW_UP_MODEL, contents=prompt
                    )

                # Parse the response
                response_text = response.text
                self._debug(f"Follow-up response: {response_text[:200]}...")

                # Extract JSON from response
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1

                if json_start != -1 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    follow_ups_data = json.loads(json_str)
                    follow_ups = follow_ups_data.get("follow_ups", [])

                    self._debug(f"Generated {len(follow_ups)} image follow-up prompts")

                    # Return in OpenAI format expected by the frontend
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps({"follow_ups": follow_ups})
                                }
                            }
                        ]
                    }
                else:
                    self._debug("Could not extract JSON from response")
                    return {
                        "choices": [
                            {"message": {"content": json.dumps({"follow_ups": []})}}
                        ]
                    }

            except Exception as e:
                self._debug(f"Error generating follow-ups with Gemini: {e}")
                return {
                    "choices": [
                        {"message": {"content": json.dumps({"follow_ups": []})}}
                    ]
                }

        except Exception as e:
            self._debug(f"Error in follow-up generation: {e}")
            return {
                "choices": [{"message": {"content": json.dumps({"follow_ups": []})}}]
            }

    def _is_guarded_prompt(self, text: str) -> bool:
        """Heuristics to avoid meta/suggestion/system-like text as a generation prompt."""
        lower = (text or "").strip().lower()
        if lower.startswith("task:"):
            return True
        if lower.startswith("### task"):
            return True
        # Only block if it looks like a system instruction to generate suggestions
        # Allow actual user requests like "Generate a follow-up image showing..."
        if "suggest" in lower and "follow-up" in lower:
            # Block patterns like "suggest follow-up questions/prompts"
            if re.search(r"suggest.*follow[- ]up.*(question|prompt|topic)", lower):
                return True
            # Block if it starts with instructional language
            if re.match(
                r"^(please\s+)?(suggest|provide|generate)\s+.*follow[- ]up", lower
            ):
                return True
        if re.search(r"\bsuggest\s+\d+\s*-?\s*\d*\s*", lower) and (
            "question" in lower or "prompt" in lower
        ):
            return True
        if len(lower.split()) <= 1 and len(lower) < 4:
            return True
        return False

    def _extract_latest_user_prompt_and_image_flag(
        self, messages: List[Dict]
    ) -> tuple[str, bool]:
        """Scan messages from the end to find the latest USER message, extract text and detect image presence.

        Applies guard patterns to the extracted text; if the text is guarded, continues scanning earlier user messages.
        Returns (prompt, user_has_image).
        """
        for msg in reversed(messages or []):
            if (msg or {}).get("role") != "user":
                continue
            content = (msg or {}).get("content", "")
            prompt_parts: List[str] = []
            user_has_image = False
            if isinstance(content, str):
                prompt_parts.append(content.strip())
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text":
                        t = (item.get("text", "") or "").strip()
                        if t:
                            prompt_parts.append(t)
                    elif item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url:
                            user_has_image = True
            elif isinstance(content, dict):
                t = content.get("text") or content.get("content")
                if isinstance(t, str) and t.strip():
                    prompt_parts.append(t.strip())

            prompt = "\n".join(p for p in prompt_parts if p).strip()
            if prompt and self._is_guarded_prompt(prompt):
                # Skip suspicious prompt; keep scanning earlier user messages
                continue
            return prompt, user_has_image

        return "", False

    def _extract_aspect_ratio_from_prompt(self, text: str) -> str:
        """Extract an aspect ratio hint from freeform prompt text.

        Supports Gemini 3 Pro Image aspect ratios:
        - "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"

        Patterns:
        - Numeric: '16:9', '9x16', '4/3', '16 x 9', etc.
        - Keywords: 'square'->1:1, 'portrait/vertical/tall'->9:16,
          'landscape/wide/cinematic'->16:9, 'classic'->4:3, 'ultrawide'->21:9

        Returns normalized form like '16:9' or empty string if none.
        """
        # Gemini 3 Pro Image supported aspect ratios
        VALID_RATIOS = {
            "1:1",
            "2:3",
            "3:2",
            "3:4",
            "4:3",
            "4:5",
            "5:4",
            "9:16",
            "16:9",
            "21:9",
        }

        try:
            if not isinstance(text, str) or not text.strip():
                return ""
            s = text.lower()

            # Keyword mappings first (common quick wins)
            if re.search(r"\bsquare\b", s):
                return "1:1"
            if re.search(r"\bultrawide\b", s):
                return "21:9"
            if re.search(r"\b(portrait|vertical|tall)\b", s):
                return "9:16"
            if re.search(r"\b(landscape|wide|cinematic)\b", s):
                return "16:9"
            if re.search(r"\bclassic\b", s):
                return "4:3"

            # Numeric pattern: allow '16:9', '16 x 9', '16x9', '16 / 9'
            m = re.search(r"(\d{1,3})\s*[:xX\/]\s*(\d{1,3})", s)
            if m:
                a, b = m.group(1), m.group(2)
                try:
                    ai = int(a)
                    bi = int(b)
                    if ai > 0 and bi > 0:
                        # Normalize to simplest terms (e.g., 1080x1920 -> 9:16)
                        from math import gcd

                        g = gcd(ai, bi)
                        ai //= g
                        bi //= g
                        ratio = f"{ai}:{bi}"

                        # Validate against SDK's supported ratios
                        if ratio in VALID_RATIOS:
                            self._debug(f"Validated ratio: {ratio}")
                            return ratio
                        else:
                            # Unsupported ratio, let model use default
                            self._debug(f"Rejected unsupported ratio: {ratio}")
                            return ""
                except Exception:
                    pass
            return ""
        except Exception:
            return ""

    def _extract_resolution_from_prompt(self, text: str) -> str:
        """Extract resolution hint from freeform prompt text.

        Supports Gemini 3 Pro Image resolutions:
        - "1K", "2K", "4K"

        Patterns:
        - Direct: '1K', '2K', '4K' (case insensitive)
        - With whitespace: '2 K', '4 k'

        Returns normalized form like '2K' or empty string if none.
        """
        VALID_RESOLUTIONS = {"1K", "2K", "4K"}

        try:
            if not isinstance(text, str) or not text.strip():
                return ""

            # Look for resolution patterns: 1K, 2K, 4K (case insensitive, optional whitespace)
            m = re.search(r"\b([124])\s*k\b", text, re.IGNORECASE)
            if m:
                res = f"{m.group(1)}K"
                if res in VALID_RESOLUTIONS:
                    self._debug(f"Validated resolution: {res}")
                    return res

            return ""
        except Exception:
            return ""

    def _extract_sticky_settings_from_history(
        self, messages: List[Dict]
    ) -> tuple[str, str]:
        """Extract aspect ratio and resolution from previous messages in conversation history.

        Scans user messages from newest to oldest to find the most recent settings.
        This makes settings "sticky" across follow-up edits.

        Returns: (aspect_ratio, resolution) tuple, either may be empty string
        """
        found_ar = ""
        found_res = ""

        try:
            # Scan messages in reverse (newest first) to find most recent settings
            for msg in reversed(messages):
                if msg.get("role") != "user":
                    continue

                content = msg.get("content", "")
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    # Extract text from content array
                    text_parts = [
                        item.get("text", "")
                        for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ]
                    text = " ".join(text_parts)
                else:
                    continue

                # Look for aspect ratio if not found yet
                if not found_ar:
                    ar = self._extract_aspect_ratio_from_prompt(text)
                    if ar:
                        found_ar = ar

                # Look for resolution if not found yet
                if not found_res:
                    res = self._extract_resolution_from_prompt(text)
                    if res:
                        found_res = res

                # Stop if we found both
                if found_ar and found_res:
                    break

            return found_ar, found_res
        except Exception as e:
            self._debug(f"Error scanning history for sticky settings: {e}")
            return "", ""

    def _build_gemini_context(
        self, messages: List[Dict], current_prompt: str, request: Optional[Any] = None
    ) -> List[types.Content]:
        """Build Gemini content array with conversation history and images.

        Strategy:
        - In iterative mode, we select the most recent images from history (assistant or user images)
          up to max_history_images and include them together with the current prompt in a single
          user message. This generally yields better image edit behavior.
        - In simple mode, we just send the current prompt.
        """
        self._debug(f"Building context for {len(messages)} messages")
        contents: List[types.Content] = []

        if not self.valves.enable_iterative:
            contents.append(
                types.Content(
                    role="user", parts=[types.Part.from_text(text=current_prompt)]
                )
            )
            return contents

        # Gather image URLs from history, most recent first
        collected_images: List[Dict[str, bytes]] = []
        # edit_mode=True: Use only 1 most recent image for consistent editing behavior
        # edit_mode=False: Use up to max_history_images for Gemini 3 Pro's multi-reference capability
        max_images = (
            1 if self.valves.edit_mode else max(0, int(self.valves.max_history_images))
        )
        self._debug(
            f"Image collection: edit_mode={self.valves.edit_mode}, max_images={max_images}"
        )

        def try_add_image(url: str):
            nonlocal collected_images
            if len(collected_images) >= max_images:
                return
            data = self._download_image(url, request)
            if data:
                collected_images.append(data)

        # Walk messages from latest to oldest to prioritize the most recent images
        for msg in reversed(messages):
            if len(collected_images) >= max_images:
                break
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "assistant" and isinstance(content, str):
                for url in self._extract_images_from_content(content):
                    if len(collected_images) >= max_images:
                        break
                    try_add_image(url)

            elif role == "user":
                # User may have provided images via multimodal message format
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            url = item.get("image_url", {}).get("url", "")
                            if url:
                                try_add_image(url)
                            if len(collected_images) >= max_images:
                                break

        # Build a single user content with optional guidance + selected images + current prompt
        parts: List[types.Part] = []
        if self.valves.edit_mode and self.valves.edit_guidance:
            parts.append(types.Part.from_text(text=self.valves.edit_guidance))
        self._debug(
            f"Collected {len(collected_images)} prior image(s) for context (max={max_images})"
        )
        for img in reversed(collected_images):  # keep chronological order for the model
            parts.append(
                types.Part.from_bytes(data=img["data"], mime_type=img["mime_type"])
            )
        parts.append(types.Part.from_text(text=current_prompt))

        contents.append(types.Content(role="user", parts=parts))
        return contents

    def _extract_images_from_content(self, content: str) -> List[str]:
        """Extract image URLs from markdown content"""
        image_urls = []

        # Find markdown images: ![alt](url) — support data URIs, http(s), and site-relative paths
        markdown_pattern = r"!\[.*?\]\((data:image/[^)]+|https?://[^)]+|/[^)]+)\)"
        matches = re.findall(markdown_pattern, content)
        image_urls.extend(matches)

        # Find HTML img tags: <img src="url">
        html_pattern = r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>'
        matches = re.findall(html_pattern, content)
        image_urls.extend(matches)

        return image_urls

    def _download_image(
        self, image_url: str, request: Optional[Any] = None
    ) -> Optional[Dict]:
        """Download image from URL or decode data URI.

        Error-handling strategy:
        - Prefer local file read for /api/v1/files/{id}/content to avoid HTTP where possible.
        - Resolve site-relative URLs using WEBUI_URL/request headers.
        - Use retry/backoff for HTTP fetches to handle transient errors.
        - Return None on failure so the caller can proceed gracefully without the image.
        """
        try:
            if image_url.startswith("data:"):
                # Handle data URI
                header, data = image_url.split(",", 1)
                mime_type = header.split(";")[0].split(":")[1]
                image_data = base64.b64decode(data)
                return {"data": image_data, "mime_type": mime_type}

            # Handle Open WebUI file URLs by ID (bypass HTTP): /api/v1/files/{id}/content
            try:
                m = re.search(
                    r"/api/v1/files/([a-f0-9\-]+)/content", image_url, re.IGNORECASE
                )
                if m:
                    file_id = m.group(1)
                    file_model = FilesDB.get_file_by_id(file_id)
                    if file_model and file_model.path:
                        local_path = Storage.get_file(file_model.path)
                        with open(local_path, "rb") as f:
                            data_bytes = f.read()
                        # Determine mime from stored meta or filename
                        mime_type = None
                        if file_model.meta and isinstance(file_model.meta, dict):
                            mime_type = file_model.meta.get(
                                "content_type"
                            ) or file_model.meta.get("mime_type")
                        if not mime_type:
                            mime_type = (
                                mimetypes.guess_type(file_model.filename or local_path)[
                                    0
                                ]
                                or "image/png"
                            )
                        return {"data": data_bytes, "mime_type": mime_type}
            except Exception as e:
                self._debug(
                    f"Direct file read by ID failed, will try URL resolution: {e}"
                )

            # Resolve site-relative URLs using WEBUI_URL if available
            if image_url.startswith("/"):
                base = os.getenv("WEBUI_URL", "").rstrip("/")
                if not base and request is not None:
                    try:
                        # Prefer proxy headers if present
                        proto = request.headers.get("x-forwarded-proto")
                        host = request.headers.get(
                            "x-forwarded-host"
                        ) or request.headers.get("host")
                        if proto and host:
                            base = f"{proto}://{host}"
                        else:
                            # Fallback to request.base_url
                            base = str(getattr(request, "base_url", "")).rstrip("/")
                    except Exception:
                        base = ""
                if base:
                    image_url = f"{base}{image_url}"
                else:
                    self._debug(
                        f"Cannot resolve relative URL (no WEBUI_URL or request base). Skipping: {image_url}"
                    )
                    return None

            # Prepare headers from incoming request for authenticated endpoints
            headers = {}
            try:
                if request is not None and getattr(request, "headers", None):
                    auth = request.headers.get("authorization")
                    cookie = request.headers.get("cookie")
                    if auth:
                        headers["Authorization"] = auth
                    if cookie:
                        headers["Cookie"] = cookie
            except Exception:
                pass

            # Handle HTTP/HTTPS URL with simple retry/backoff for transient errors
            attempts = max(1, int(getattr(self.valves, "retry_attempts", 3)))
            backoff = 1.0
            for attempt in range(1, attempts + 1):
                try:
                    response = requests.get(
                        image_url,
                        headers=headers or None,
                        timeout=max(
                            1, int(getattr(self.valves, "download_timeout", 20))
                        ),
                    )
                    if (
                        response.status_code in (429,)
                        or 500 <= response.status_code <= 599
                    ):
                        # transient
                        self._debug(
                            f"HTTP transient error {response.status_code}: {response.text[:200]}"
                        )
                        if attempt < attempts:
                            import time as _t

                            _t.sleep(backoff)
                            backoff *= float(
                                getattr(self.valves, "retry_backoff_base", 1.5)
                            )
                            continue
                        response.raise_for_status()
                    response.raise_for_status()
                    break
                except Exception as ex:
                    self._debug(f"Download attempt {attempt} failed: {ex}")
                    if attempt < attempts:
                        import time as _t

                        _t.sleep(backoff)
                        backoff *= float(
                            getattr(self.valves, "retry_backoff_base", 1.5)
                        )
                        continue
                    raise

            # Determine MIME type
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                # Guess from URL
                mime_type = mimetypes.guess_type(image_url)[0]
                if not mime_type or not mime_type.startswith("image/"):
                    mime_type = "image/png"  # Default
            else:
                mime_type = content_type

            return {"data": response.content, "mime_type": mime_type}
        except Exception as e:
            self._debug(f"Failed to download image {image_url}: {e}")
            return None
