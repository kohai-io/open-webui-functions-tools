"""
title: Google Veo-3 Video Generation Pipeline (Inline Display)
author: open-webui
version: 7.1.1
license: MIT
description: A pipeline for generating videos using Google's Veo-3 model via the Google GenAI SDK with Vertex AI support. Supports inline-friendly output, encrypted API key storage, optional image conditioning from chat attachments or file links, and C2PA content provenance signing.
requirements: google-genai>=1.41.0, cryptography, pydantic, google-auth, c2pa-python
"""

import os
import time
import json
import asyncio
import base64
import hashlib
import logging
import re
import aiohttp
import io
from datetime import datetime
from typing import Optional, Callable, Awaitable, Any
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from cryptography.fernet import Fernet, InvalidToken
from pydantic_core import core_schema
import mimetypes

# C2PA imports (optional - will gracefully degrade if not installed)
try:
    from c2pa import Builder as C2paBuilder, C2paSignerInfo, C2paSigningAlg, Signer as C2paSigner
    C2PA_AVAILABLE = True
except ImportError:
    C2PA_AVAILABLE = False

# Google GenAI imports
from google import genai
from google.genai import types

# Optional Open WebUI imports used for resolving images from file storage
try:
    from open_webui.models.files import Files as FilesDB
    from open_webui.storage.provider import Storage
except Exception:  # pragma: no cover - allow pipeline to load even if imports fail during lint/test
    FilesDB = None
    Storage = None


class EncryptedStr(str):
    """A string type that automatically handles encryption/decryption"""

    @classmethod
    def _get_encryption_key(cls) -> Optional[bytes]:
        """Generate encryption key from WEBUI_SECRET_KEY if available"""
        secret = os.getenv("WEBUI_SECRET_KEY")
        if not secret:
            return None
        hashed_key = hashlib.sha256(secret.encode()).digest()
        return base64.urlsafe_b64encode(hashed_key)

    @classmethod
    def encrypt(cls, value: str) -> str:
        """Encrypt a string value if a key is available"""
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
        """Decrypt an encrypted string value if a key is available"""
        if not value or not value.startswith("encrypted:"):
            return value
        key = cls._get_encryption_key()
        if not key:
            return value[len("encrypted:"):]
        try:
            encrypted_part = value[len("encrypted:"):]
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
    """Google Veo-3 Video Generation Pipeline (Inline)"""

    class Valves(BaseModel):
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
        GOOGLE_API_KEY: EncryptedStr = Field(
            default="",
            description="Google API key for accessing Veo-3 (only used if use_vertex_ai=False). Will be encrypted for security.",
        )
        MODEL_ID: str = Field(
            default="veo-3.0-generate-001",
            description="The Veo model identifier. Use 'veo-3.0-generate-001' for the latest Veo 3 model.",
        )
        duration_seconds: int = Field(
            default=8,
            description="Output video duration in seconds. Valid values are: 4, 6, or 8 seconds only.",
        )
        POLL_INTERVAL: int = Field(
            default=20,
            description="Seconds between status checks for video generation (default: 20).",
        )
        MAX_POLL_TIME: int = Field(
            default=600,
            description="Maximum time in seconds to wait for video generation (default: 600 = 10 minutes).",
        )
        DOWNLOAD_TIMEOUT: int = Field(
            default=120,
            description="Total timeout (seconds) for downloading generated videos.",
        )
        EMIT_INTERVAL: float = Field(
            default=0.5, description="Interval in seconds between status emissions"
        )
        ENABLE_STATUS_INDICATOR: bool = Field(
            default=True, description="Enable or disable status indicator emissions"
        )
        PROGRESS_UPDATE_INTERVALS: str = Field(
            default="15,30,60,120",
            description="Comma-separated seconds for progressive status updates (e.g., '15,30,60,120')"
        )
        ENABLE_CUSTOM_FOLLOW_UPS: bool = Field(
            default=True,
            description="Enable custom follow-up generation optimized for video workflows"
        )
        FOLLOW_UP_MODEL: str = Field(
            default="gemini-2.0-flash-exp",
            description="Model to use for generating follow-up prompts (e.g., gemini-2.0-flash-exp)"
        )
        FOLLOW_UP_PROMPT_TEMPLATE: str = Field(
            default="",
            description="Custom prompt template for follow-up generation. Leave empty to use default video-optimized template."
        )
        ENABLE_C2PA_SIGNING: bool = Field(
            default=False,
            description="Enable C2PA content provenance signing for generated videos. Requires c2pa-python library and certificate configuration.",
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
            description="AI training policy for generated videos: 'allowed', 'notAllowed', or 'constrained'. Controls c2pa.training-mining assertion.",
        )

    def __init__(self):
        self.name = "Google Veo-3 Video (Inline)"
        self.valves = self.Valves()
        self.log = logging.getLogger(self.name.replace(" ", "_").lower())
        self.log.setLevel(logging.INFO)
        self.last_emit_time = 0
        self._cached_video_data = None  # Store video data temporarily for inline videos

    async def on_startup(self):
        """Log pipeline startup information including C2PA status"""
        self.log.info(f"Veo-3 Pipeline loaded - version 7.1.0")
        self.log.info(f"Model: {self.valves.MODEL_ID}")
        self.log.info(f"Auth mode: {'Vertex AI' if self.valves.use_vertex_ai else 'API Key'}")
        self.log.info(f"C2PA library: {'Available' if C2PA_AVAILABLE else 'Not installed'}")
        
        if self.valves.ENABLE_C2PA_SIGNING:
            has_cert = bool(self.valves.C2PA_CERT_CONTENT or self.valves.C2PA_CERT_PATH)
            has_key = bool(self.valves.C2PA_KEY_CONTENT or self.valves.C2PA_KEY_PATH)
            cert_source = "content" if self.valves.C2PA_CERT_CONTENT else ("path" if self.valves.C2PA_CERT_PATH else "missing")
            key_source = "content" if self.valves.C2PA_KEY_CONTENT else ("path" if self.valves.C2PA_KEY_PATH else "missing")
            
            if has_cert and has_key:
                self.log.info(f"C2PA signing: Enabled (cert: {cert_source}, key: {key_source})")
            else:
                self.log.warning(f"C2PA signing: Enabled but missing credentials (cert: {cert_source}, key: {key_source})")
        else:
            self.log.info("C2PA signing: Disabled")

    async def on_shutdown(self):
        """Cleanup on pipeline shutdown"""
        self.log.info("Veo-3 Pipeline shutting down")

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
                self.log.warning(f"Failed to emit status: {e}")
    
    async def handle_error(
        self,
        body: dict,
        event_emitter: Optional[Callable[[dict], Awaitable[None]]],
        short_status: str,
        detailed_message: str,
    ) -> str:
        """Handle errors with short status updates and detailed chat messages.
        
        Args:
            body: The request body to append the message to
            event_emitter: Event emitter for status updates
            short_status: Brief error indicator for status (e.g., "Configuration error")
            detailed_message: Full error details for chat response
        
        Returns:
            The detailed error message
        """
        # Emit short status update (limited length)
        await self.emit_status(event_emitter, "error", f"❌ {short_status}", True)
        
        # Log the full error
        self.log.error(detailed_message)
        
        # Add detailed message to chat
        full_response = f"❌ **{short_status}**\n\n{detailed_message}"
        body["messages"].append({"role": "assistant", "content": full_response})
        
        return full_response

    def _create_c2pa_manifest(self, prompt: str, user: dict, duration_seconds: int, image_ref: Optional[str] = None) -> dict:
        """
        Create C2PA manifest for AI-generated videos.
        
        Includes:
        - AI generation assertion
        - Training/mining policy
        - Creator information
        - Generation parameters (prompt, duration, model)
        """
        manifest = {
            "claim_generator": "OpenWebUI/VeoInline",
            "claim_generator_info": [
                {
                    "name": "OpenWebUI - Veo 3 Video Pipeline",
                    "version": "7.1.0"
                }
            ],
            "title": f"AI Generated Video - {datetime.utcnow().isoformat()}",
            "format": "video/mp4",
            "assertions": [
                {
                    "label": "c2pa.ai-generated",
                    "data": {
                        "generator": self.valves.MODEL_ID,
                        "prompt": prompt if prompt else "Not provided",
                        "duration_seconds": duration_seconds,
                        "date": datetime.utcnow().isoformat(),
                        "pipeline": "veo_inline",
                        "conditioning_image": image_ref if image_ref else "None"
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
        if user and user.get("id"):
            manifest["assertions"].append({
                "label": "stds.schema-org.CreativeWork",
                "data": {
                    "@context": "https://schema.org",
                    "@type": "VideoObject",
                    "creator": {
                        "@type": "Person",
                        "name": user.get("name", "Unknown"),
                        "identifier": user.get("id", "unknown")
                    }
                }
            })
        
        return manifest

    def _sign_video_with_c2pa(
        self, 
        video_data: bytes, 
        mime_type: str,
        prompt: str,
        duration_seconds: int,
        user: dict,
        image_ref: Optional[str] = None
    ) -> bytes:
        """
        Sign video data with C2PA manifest.
        
        Args:
            video_data: Raw video bytes
            mime_type: MIME type (e.g., 'video/mp4')
            prompt: Generation prompt
            duration_seconds: Video duration
            user: User information
            image_ref: Optional reference image URL
        
        Returns:
            Signed video bytes (or original if signing fails/disabled)
        """
        
        # Check if C2PA is available and enabled
        if not C2PA_AVAILABLE:
            self.log.debug("C2PA library not installed - skipping signing")
            return video_data
        
        if not self.valves.ENABLE_C2PA_SIGNING:
            self.log.debug("C2PA signing disabled in settings")
            return video_data
        
        # Check for required certificate/key configuration (content OR path)
        has_cert = bool(self.valves.C2PA_CERT_CONTENT or self.valves.C2PA_CERT_PATH)
        has_key = bool(self.valves.C2PA_KEY_CONTENT or self.valves.C2PA_KEY_PATH)
        
        if not has_cert or not has_key:
            self.log.debug("C2PA certificate/key not configured - skipping signing")
            return video_data
        
        try:
            self.log.info(f"Signing video with C2PA (size: {len(video_data)} bytes)")
            
            # Get certificate and key as bytes (content takes precedence)
            if self.valves.C2PA_CERT_CONTENT:
                cert_data = self.valves.C2PA_CERT_CONTENT.get_decrypted().encode('utf-8')
                self.log.debug("Using C2PA certificate from content (encrypted)")
            else:
                cert_path = self.valves.C2PA_CERT_PATH
                if not os.path.exists(cert_path):
                    self.log.warning(f"C2PA certificate not found: {cert_path}")
                    return video_data
                with open(cert_path, "rb") as f:
                    cert_data = f.read()
                self.log.debug(f"Using C2PA certificate from path: {cert_path}")
            
            if self.valves.C2PA_KEY_CONTENT:
                key_data = self.valves.C2PA_KEY_CONTENT.get_decrypted().encode('utf-8')
                self.log.debug("Using C2PA private key from content (encrypted)")
            else:
                key_path = self.valves.C2PA_KEY_PATH
                if not os.path.exists(key_path):
                    self.log.warning(f"C2PA private key not found: {key_path}")
                    return video_data
                with open(key_path, "rb") as f:
                    key_data = f.read()
                self.log.debug(f"Using C2PA private key from path: {key_path}")
            
            # Create manifest JSON
            manifest = self._create_c2pa_manifest(prompt, user, duration_seconds, image_ref)
            manifest_json = json.dumps(manifest)
            
            # Create signer info using the current API
            signer_kwargs = {
                "alg": C2paSigningAlg.ES256,  # Elliptic Curve (most common for ES256 keys)
                "sign_cert": cert_data,
                "private_key": key_data,
            }
            
            # Add timestamp authority only if configured (optional parameter)
            if self.valves.C2PA_TSA_URL:
                signer_kwargs["ta_url"] = self.valves.C2PA_TSA_URL
                self.log.debug(f"Using TSA: {self.valves.C2PA_TSA_URL}")
            else:
                self.log.debug("No TSA configured - signing without timestamp")
            
            signer_info = C2paSignerInfo(**signer_kwargs)
            signer = C2paSigner.from_info(signer_info)
            
            # Sign the video using Builder context manager
            source_stream = io.BytesIO(video_data)
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
            
            self.log.info(f"C2PA signing successful! Manifest size: {len(manifest_bytes)} bytes")
            self.log.info(f"Signed video size: {len(signed_data)} bytes (original: {len(video_data)} bytes)")
            
            return signed_data
            
        except Exception as e:
            self.log.error(f"C2PA signing failed: {e}")
            import traceback
            self.log.error(f"C2PA traceback: {traceback.format_exc()}")
            # Return original unsigned video on error
            return video_data

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __event_call__: Optional[Callable[[dict], Awaitable[dict]]] = None,
    ) -> Optional[dict]:
        """Main pipeline method that generates a video from a text prompt and returns inline-friendly output.
        Returns a simple markdown string as the primary return, and also appends it to body["messages"].
        """
        # Log the incoming request structure
        self.log.info(f"=== Veo Pipeline Called ===")
        self.log.info(f"Body keys: {list(body.keys())}")
        self.log.info(f"Model: {body.get('model', 'NOT_SET')}")
        self.log.info(f"Messages count: {len(body.get('messages', []))}")
        
        # Check if this is a follow-up generation request
        metadata = body.get("metadata", {})
        is_follow_up_task = metadata.get("task") == "FOLLOW_UP_GENERATION"
        
        if is_follow_up_task and self.valves.ENABLE_CUSTOM_FOLLOW_UPS:
            self.log.info("Detected follow-up generation request - using custom video follow-ups")
            return await self._generate_video_follow_ups(body, __event_emitter__)
        
        await self.emit_status(
            __event_emitter__, "info", "Initializing Veo-3 video generation..."
        )

        # Validate authentication configuration
        if self.valves.use_vertex_ai:
            # Check Vertex AI configuration
            project_id = self.valves.project_id or os.getenv("GOOGLE_CLOUD_PROJECT", "")
            if not project_id:
                return await self.handle_error(
                    body, __event_emitter__,
                    "Configuration error",
                    "Google Cloud Project ID not configured.\n\n"
                    "**Solution:** Set `project_id` in pipeline settings or set the `GOOGLE_CLOUD_PROJECT` environment variable."
                )
            
            # Check if service account credentials are provided
            if not self.valves.SERVICE_ACCOUNT_JSON and not self.valves.SERVICE_ACCOUNT_PATH:
                return await self.handle_error(
                    body, __event_emitter__,
                    "Configuration error",
                    "Service account credentials not configured for Vertex AI.\n\n"
                    "**Solutions:**\n"
                    "- Provide `SERVICE_ACCOUNT_JSON` in pipeline settings, or\n"
                    "- Provide `SERVICE_ACCOUNT_PATH` in pipeline settings, or\n"
                    "- Set `use_vertex_ai=False` to use API key authentication instead."
                )
        else:
            # Check API key configuration
            api_key = self.valves.GOOGLE_API_KEY.get_decrypted()
            if not api_key:
                return await self.handle_error(
                    body, __event_emitter__,
                    "Configuration error",
                    "Google API key not configured.\n\n"
                    "**Solution:** Set your `GOOGLE_API_KEY` in the pipeline settings."
                )

        # Extract a valid USER-authored prompt (guarded)
        messages = body.get("messages", [])
        prompt = self._extract_latest_user_prompt(messages)
        if not prompt:
            # Check if this is a system follow-up instruction (should be silently ignored)
            last_user_msg = None
            for msg in reversed(messages or []):
                if msg.get("role") == "user":
                    last_user_msg = msg
                    break
            
            is_system_request = False
            if last_user_msg:
                content = last_user_msg.get("content", "")
                if isinstance(content, str):
                    text = content.lower().strip()
                    # Detect system meta-requests (follow-ups, title generation, etc.)
                    if re.search(r"suggest.*follow[- ]up.*(question|prompt|topic)", text):
                        is_system_request = True
                    elif re.match(r"^(please\s+)?(suggest|provide|generate)\s+\d*\s*follow[- ]up", text):
                        is_system_request = True
                    # Detect title generation requests (multiple patterns for robustness)
                    elif re.search(r"generate.*title.*summar", text):
                        is_system_request = True
                    elif re.search(r"### task.*title", text):
                        is_system_request = True
                    elif re.search(r"^###\s*task:", text):  # Broader task detection
                        is_system_request = True
                    elif "chat history" in text and ("title" in text or "summariz" in text):
                        is_system_request = True
                    elif text.startswith("task:") or text.startswith("###"):
                        # Catch any task-like prompts
                        is_system_request = True
            
            if is_system_request:
                # Silently ignore system meta-requests - don't generate or show error
                self.log.info(f"Ignoring system meta-request. Content preview: {content[:200] if isinstance(content, str) else str(content)[:200]}")
                return ""  # Return empty to skip silently
            else:
                # Real error - no valid prompt found
                self.log.warning(f"No valid prompt found. Messages: {len(messages)}")
                for i, msg in enumerate(messages[-3:]):  # Show last 3 messages
                    role = msg.get('role')
                    content = msg.get('content')
                    content_preview = str(content)[:100] if content else 'None'
                    self.log.warning(f"  Msg {i}: role={role}, content={content_preview}...")
                
                return await self.handle_error(
                    body, __event_emitter__,
                    "Invalid request",
                    "No valid user prompt found for video generation.\n\n"
                    "Please provide a text description of the video you want to create."
                )

        # Attempt to extract an image reference from the latest user message or prior messages
        image_ref: Optional[str] = None
        self.log.info(f"Scanning {len(messages)} messages for image references...")
        if messages:
            for idx, msg in enumerate(reversed(messages)):
                msg_role = (msg or {}).get("role")
                self.log.info(f"Message {idx}: role={msg_role}")
                if msg_role == "user":
                    # Check if there's an images array (alternate format)
                    if "images" in msg:
                        images = msg.get("images", [])
                        self.log.info(f"Found images array with {len(images)} item(s): {images}")
                        if images and len(images) > 0:
                            # Images array might contain file IDs
                            first_image = images[0]
                            if isinstance(first_image, str):
                                # Could be a file ID or URL
                                if first_image.startswith('/api/v1/files/'):
                                    image_ref = first_image
                                    self.log.info(f"✓ Using image from images array: {image_ref}")
                                elif '/' not in first_image:
                                    # Looks like a file ID, construct the path
                                    image_ref = f"/api/v1/files/{first_image}/content"
                                    self.log.info(f"✓ Constructed image path from file ID: {image_ref}")
                    
                    content = (msg or {}).get("content", "")
                    self.log.info(f"User message content type: {type(content)}")
                    if isinstance(content, list):
                        self.log.info(f"Content is a list with {len(content)} items")
                        for item_idx, item in enumerate(content):
                            if not isinstance(item, dict):
                                self.log.info(f"  Item {item_idx}: not a dict, type={type(item)}")
                                continue
                            item_type = item.get("type")
                            self.log.info(f"  Item {item_idx}: type={item_type}, keys={list(item.keys())}")
                            if item_type == "image_url":
                                image_ref = (item.get("image_url") or {}).get("url")
                                self.log.info(f"  ✓ Found image_url: {image_ref}")
                            elif item_type in {"file", "input_image", "input_file"}:
                                meta = item.get("file", {}) or item
                                url = meta.get("url") or (meta.get("image_url") or {}).get("url")
                                mime = meta.get("mime") or meta.get("mime_type")
                                self.log.info(f"  Found file type: url={url}, mime={mime}")
                                if url and (not mime or str(mime).startswith("image/")):
                                    image_ref = url
                                    self.log.info(f"  ✓ Using file URL as image_ref: {image_ref}")
                    elif isinstance(content, str):
                        self.log.info(f"Content is string: {content[:100]}...")
                    else:
                        self.log.info(f"Content is other type: {type(content)}")
                    break

        if not image_ref and messages:
            # Fallback: scan previous messages for image URLs or file links
            for prev in reversed(messages[:-1]):
                c = prev.get("content", "")
                if isinstance(c, str):
                    urls = self._extract_images_from_content(c)
                    if urls:
                        image_ref = urls[0]
                        break
                elif isinstance(c, list):
                    for item in c:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            u = (item.get("image_url") or {}).get("url")
                            if u:
                                image_ref = u
                                break
                    if image_ref:
                        break

        # Setup for credential cleanup
        temp_creds_path = None
        old_creds = None
        
        try:
            self.log.info(f"Starting Veo inline video generation with prompt: {prompt[:100]}...")

            # Create GenAI client based on authentication mode
            if self.valves.use_vertex_ai:
                project_id = self.valves.project_id or os.getenv("GOOGLE_CLOUD_PROJECT", "")
                location = self.valves.location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
                
                self.log.info(f"Using Vertex AI with service account authentication, project={project_id}, location={location}")
                
                # For Vertex AI with service account, set credentials via environment
                # Keep them set for the entire operation duration
                import tempfile
                
                if self.valves.SERVICE_ACCOUNT_JSON:
                    service_account_json = self.valves.SERVICE_ACCOUNT_JSON.get_decrypted()
                    try:
                        service_account_info = json.loads(service_account_json)
                    except json.JSONDecodeError as e:
                        raise json.JSONDecodeError(
                            f"Invalid SERVICE_ACCOUNT_JSON format: {str(e)}",
                            e.doc, e.pos
                        )
                    
                    # Write to temp file and set environment variable
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        json.dump(service_account_info, f)
                        temp_creds_path = f.name
                    
                    # Set environment variable - will be cleaned up in finally block
                    old_creds = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_creds_path
                else:
                    # Use SERVICE_ACCOUNT_PATH directly
                    old_creds = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.valves.SERVICE_ACCOUNT_PATH
                
                client = genai.Client(
                    vertexai=True,
                    project=project_id,
                    location=location,
                )
            else:
                api_key = self.valves.GOOGLE_API_KEY.get_decrypted()
                self.log.info("Using API key authentication")
                client = genai.Client(api_key=api_key)

            # Validate duration early
            valid_durations = [4, 6, 8]
            if self.valves.duration_seconds not in valid_durations:
                return await self.handle_error(
                    body, __event_emitter__,
                    "Invalid configuration",
                    f"Invalid video duration: **{self.valves.duration_seconds}s**\n\n"
                    f"**Valid durations:** {', '.join(map(str, valid_durations))} seconds\n\n"
                    "**Solution:** Update the `duration_seconds` setting in pipeline configuration."
                )

            # Resolve image (optional)
            resolved_image = None
            if image_ref:
                await self.emit_status(
                    __event_emitter__, "info", "Processing reference image..."
                )
                self.log.info(f"Resolving image reference: {image_ref}")
                resolved_image = self._resolve_prompt_image(image_ref)
                if resolved_image:
                    # Log type and size of resolved image
                    if resolved_image.startswith("data:"):
                        mime_match = re.match(r"data:([^;]+);base64,(.+)", resolved_image)
                        if mime_match:
                            mime_type = mime_match.group(1)
                            b64_data = mime_match.group(2)
                            self.log.info(f"✓ Image resolved to data URI: mime={mime_type}, size={len(b64_data)} base64 chars")
                        else:
                            self.log.info(f"✓ Image resolved to data URI (unparsed): {resolved_image[:100]}...")
                    elif resolved_image.startswith("gs://"):
                        self.log.info(f"✓ Image resolved to GCS URI: {resolved_image}")
                    else:
                        self.log.info(f"✓ Image resolved to URL: {resolved_image[:100]}...")
                else:
                    self.log.warning(f"❌ Failed to resolve image reference: {image_ref}")
            else:
                self.log.info("No image reference provided - generating video from prompt only")

            await self.emit_status(
                __event_emitter__, "info", "Starting video generation with Veo..."
            )
            operation = await self._start_video_generation(client, prompt, resolved_image)
            if not operation:
                return await self.handle_error(
                    body, __event_emitter__,
                    "Generation failed",
                    "Failed to start video generation.\n\n"
                    "**Possible causes:**\n"
                    "- Invalid credentials or authentication\n"
                    "- API quota exceeded\n"
                    "- Model unavailable in your region\n\n"
                    "Check the logs for more details."
                )

            await self.emit_status(
                __event_emitter__, "info", "Video generation in progress (this may take several minutes)..."
            )
            video_url = await self._poll_video_status(client, operation, __event_emitter__)

            # Check if this is an error message from polling (must check BEFORE trying to download)
            if video_url and video_url.startswith("ERROR::"):
                # Extract the detailed error message
                response_content = f"❌ {video_url[7:]}"  # Remove "ERROR::" prefix
                body["messages"].append({"role": "assistant", "content": response_content})
                return response_content
            
            if video_url:
                # Check if this is inline video data from REST API, SDK, or a URL
                if video_url == "__REST_INLINE_VIDEO__" or video_url == "__SDK_INLINE_VIDEO__":
                    # Try to get video data from operation first, then from instance variable
                    video_data = None
                    source = "REST API" if video_url == "__REST_INLINE_VIDEO__" else "SDK"
                    
                    if hasattr(operation, '_video_data') and operation._video_data:
                        video_data = operation._video_data
                        self.log.info(f"Using inline video data from {source} (operation): {len(video_data)} bytes")
                    elif self._cached_video_data:
                        video_data = self._cached_video_data
                        self.log.info(f"Using inline video data from {source} (cached): {len(video_data)} bytes")
                        # Clear cache after use
                        self._cached_video_data = None
                    else:
                        self.log.error(f"Inline video marker found but no data available")
                        self.log.error(f"Operation has _video_data: {hasattr(operation, '_video_data')}")
                        self.log.error(f"Cached data available: {self._cached_video_data is not None}")
                else:
                    # Regular URL - download it
                    await self.emit_status(
                        __event_emitter__, "info", "Downloading generated video..."
                    )
                    video_data = await self._download_video(video_url)

                if video_data:
                    await self.emit_status(
                        __event_emitter__, "info", "Saving video file..."
                    )
                    operation_name = operation.name if hasattr(operation, 'name') else str(operation)
                    saved_video_id = await self._save_video_file(video_data, prompt, operation_name, __user__, resolved_image)

                    video_size_mb = len(video_data) / (1024 * 1024)

                    if saved_video_id:
                        await self.emit_status(
                            __event_emitter__, "info", "Veo-3 video generation complete!", True
                        )
                        webui_base = os.getenv("WEBUI_URL", "http://localhost:8080").rstrip("/")
                        content_url = f"{webui_base}/api/v1/files/{saved_video_id}/content"
                        response_content = (
                            f"✅ Veo3 video generated! [Click here to download]({content_url}) "
                            f"({video_size_mb:.1f}MB)"
                        )
                    else:
                        response_content = (
                            f"✅ Veo3 video generated but failed to save. Size: {video_size_mb:.1f}MB"
                        )
                else:
                    await self.emit_status(__event_emitter__, "error", "Download failed", True)
                    response_content = (
                        "❌ **Download failed**\n\n"
                        "Failed to download the generated video.\n\n"
                        "**Possible causes:**\n"
                        "- Network connectivity issues\n"
                        "- Download timeout (increase `DOWNLOAD_TIMEOUT` setting)\n"
                        "- Invalid or expired video URL\n\n"
                        "Please try again."
                    )
            else:
                # Generic error - no video data returned
                response_content = "❌ **Video generation failed**\n\nPlease check the error message above and try again."
                self.log.error("Video generation failed - no video data returned")

            # Append the assistant message for history, but return the content string for inline handling
            body["messages"].append({"role": "assistant", "content": response_content})
            return response_content

        except json.JSONDecodeError as e:
            return await self.handle_error(
                body, __event_emitter__,
                "Configuration error",
                f"Invalid JSON in service account credentials.\n\n"
                f"**Error:** {str(e)}\n\n"
                "**Solution:** Check that `SERVICE_ACCOUNT_JSON` contains valid JSON."
            )
        except Exception as e:
            self.log.exception("Unexpected error during video generation")
            return await self.handle_error(
                body, __event_emitter__,
                "Unexpected error",
                f"An unexpected error occurred during video generation.\n\n"
                f"**Error details:** {str(e)}\n\n"
                f"**Error type:** {type(e).__name__}\n\n"
                "Please check the logs for more information."
            )
        finally:
            # Clean up credentials
            if old_creds is not None:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = old_creds
            elif 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
                os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS', None)
            
            if temp_creds_path:
                try:
                    os.unlink(temp_creds_path)
                except:
                    pass

    def _extract_latest_user_prompt(self, messages: list[dict]) -> str:
        """Return the latest user-authored text prompt that passes guard checks.

        Supports both raw string content and OpenAI-style content parts.
        Applies guard patterns to avoid using meta/suggestion/system-like text.
        """
        for msg in reversed(messages or []):
            if (msg or {}).get("role") != "user":
                continue

            content = (msg or {}).get("content", "")
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                # Join textual parts only
                parts = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text" and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                    elif isinstance(item.get("content"), str):
                        parts.append(item["content"])  # tolerate other shapes
                text = "\n".join(p for p in parts if p)
            elif isinstance(content, dict):
                text = content.get("text") or content.get("content") or json.dumps(content)

            text = (text or "").strip()
            if not text:
                continue

            if self._is_guarded_prompt(text):
                # Skip suspicious prompts (likely assistant meta, suggestions, or tasks copied over)
                self.log.info(f"Guard skipped suspicious prompt: '{text[:100]}...'")
                continue

            return text

        return ""

    async def _generate_video_follow_ups(self, body: dict, event_emitter: Optional[Callable[[dict], Awaitable[None]]] = None) -> dict:
        """Generate video-specific follow-up prompts using Gemini."""
        try:
            messages = body.get("messages", [])
            
            # Get the default or custom template
            if self.valves.FOLLOW_UP_PROMPT_TEMPLATE:
                template = self.valves.FOLLOW_UP_PROMPT_TEMPLATE
            else:
                # Default video-optimized template
                template = """### Task:
Suggest 3-5 relevant follow-up video generation prompts that the user might want to create next, based on the video they just generated and the conversation history.

### Guidelines:
- Focus on video-specific concepts: camera movements, scene transitions, continuations, variations, different angles, lighting changes, time of day, weather, pacing
- Suggest prompts that would create compelling cinematic sequences related to the current video
- Write from the user's perspective as requests to generate videos
- Keep prompts concise but descriptive with cinematic details
- Include camera techniques (tracking shot, crane shot, POV, etc.) where relevant
- Consider narrative progression, mood variations, or alternative perspectives
- Ensure suggestions are directly related to the discussed topic
- Use the conversation's primary language; default to English if multilingual
- Response must be a JSON array of strings, no extra text

### Output Format:
JSON: { "follow_ups": ["Prompt 1", "Prompt 2", "Prompt 3"] }

### Examples of Good Video Follow-ups:
- "Create a continuation showing the same scene from a bird's eye view as the camera slowly rises"
- "Generate a tracking shot following the subject as they move through the environment at golden hour"
- "Show the same scene but during a rainstorm with dramatic lighting"

### Chat History (last 6 messages):
{{MESSAGES:END:6}}"""

            # Replace template variables
            recent_messages = messages[-6:] if len(messages) > 6 else messages
            messages_text = "\n".join([
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                for msg in recent_messages
            ])
            
            prompt = template.replace("{{MESSAGES:END:6}}", messages_text)
            
            self.log.info(f"Generating video follow-ups using model: {self.valves.FOLLOW_UP_MODEL}")
            
            # Use Google GenAI to generate follow-ups
            try:
                if self.valves.use_vertex_ai:
                    project_id = self.valves.project_id or os.getenv("GOOGLE_CLOUD_PROJECT", "")
                    location = self.valves.location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
                    
                    import tempfile
                    temp_creds_path = None
                    old_creds = None
                    
                    if self.valves.SERVICE_ACCOUNT_JSON:
                        service_account_json = self.valves.SERVICE_ACCOUNT_JSON.get_decrypted()
                        try:
                            service_account_info = json.loads(service_account_json)
                        except json.JSONDecodeError as e:
                            self.log.error(f"Invalid SERVICE_ACCOUNT_JSON in follow-up generation: {e}")
                            return {"choices": [{"message": {"content": json.dumps({"follow_ups": []})}}]}
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                            json.dump(service_account_info, f)
                            temp_creds_path = f.name
                        old_creds = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
                        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_creds_path
                    else:
                        old_creds = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
                        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.valves.SERVICE_ACCOUNT_PATH
                    
                    try:
                        client = genai.Client(vertexai=True, project=project_id, location=location)
                        response = client.models.generate_content(
                            model=self.valves.FOLLOW_UP_MODEL,
                            contents=prompt
                        )
                    finally:
                        if old_creds is not None:
                            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = old_creds
                        elif 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
                            os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS', None)
                        if temp_creds_path:
                            try:
                                os.unlink(temp_creds_path)
                            except:
                                pass
                else:
                    api_key = self.valves.GOOGLE_API_KEY.get_decrypted()
                    client = genai.Client(api_key=api_key)
                    response = client.models.generate_content(
                        model=self.valves.FOLLOW_UP_MODEL,
                        contents=prompt
                    )
                
                # Parse the response
                response_text = response.text
                self.log.info(f"Follow-up response: {response_text[:200]}...")
                
                # Extract JSON from response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    try:
                        follow_ups_data = json.loads(json_str)
                        follow_ups = follow_ups_data.get("follow_ups", [])
                        
                        self.log.info(f"Generated {len(follow_ups)} video follow-up prompts")
                        
                        # Return in OpenAI format expected by the frontend
                        return {
                            "choices": [{
                                "message": {
                                    "content": json.dumps({"follow_ups": follow_ups})
                                }
                            }]
                        }
                    except json.JSONDecodeError as e:
                        self.log.warning(f"Invalid JSON from LLM: {e}")
                        return {"choices": [{"message": {"content": json.dumps({"follow_ups": []})}}]}
                else:
                    self.log.warning("Could not extract JSON from response")
                    return {"choices": [{"message": {"content": json.dumps({"follow_ups": []})}}]}
                    
            except Exception as e:
                self.log.error(f"Error generating follow-ups with Gemini: {e}")
                return {"choices": [{"message": {"content": json.dumps({"follow_ups": []})}}]}
                
        except Exception as e:
            self.log.error(f"Error in follow-up generation: {e}")
            return {"choices": [{"message": {"content": json.dumps({"follow_ups": []})}}]}

    def _is_guarded_prompt(self, text: str) -> bool:
        """Heuristics to avoid using meta/suggestion/system-like text as a generation prompt.

        Intentionally conservative; extend as needed.
        """
        lower = (text or "").strip().lower()

        # Obvious meta/task patterns
        if lower.startswith("task:"):
            return True
        if lower.startswith("### task"):
            return True
        # Only block if it looks like a system instruction to generate suggestions
        # Allow actual user requests like "Generate a follow-up video showing..."
        if "suggest" in lower and "follow-up" in lower:
            # Block patterns like "suggest follow-up questions/prompts"
            if re.search(r"suggest.*follow[- ]up.*(question|prompt|topic)", lower):
                return True
            # Block if it starts with instructional language
            if re.match(r"^(please\s+)?(suggest|provide|generate)\s+.*follow[- ]up", lower):
                return True
        if re.search(r"\bsuggest\s+\d+\s*-?\s*\d*\s*", lower) and ("question" in lower or "prompt" in lower):
            return True

        # Very short/low-signal prompts (e.g., single word) - avoid accidental triggers
        if len(lower.split()) <= 1 and len(lower) < 4:
            return True

        return False

    # Safety filter code mappings from Google Cloud documentation
    # https://cloud.google.com/vertex-ai/generative-ai/docs/video/responsible-ai-and-usage-guidelines
    SAFETY_FILTER_CODES = {
        # Violence-related codes
        '58061214': 'violence',
        '29310472': 'violence',
        '64151117': 'violence',
        '42237218': 'violence',
        '62263041': 'violence',
        '57734940': 'violence',
        
        # Sexual content codes
        '15236754': 'sexual',
        '22137204': 'sexual',
        '74803281': 'sexual',
        '29578790': 'sexual',
        
        # Derogatory content codes
        '42876398': 'derogatory',
        '92201652': 'derogatory',
        '89371032': 'derogatory',
        
        # Toxic content codes
        '49114662': 'toxic',
        '72817394': 'toxic',
        '90789179': 'toxic',
        
        # Person generation codes
        '17301594': 'personGeneration',
        '63429089': 'personGeneration',
        '43188360': 'personGeneration',
        '78610348': 'personGeneration',
        '61493863': 'personGeneration',
        '56562880': 'personGeneration',
        '32635315': 'personGeneration',
    }
    
    SAFETY_CATEGORY_GUIDANCE = {
        'violence': {
            'title': 'Violence Content Detected',
            'description': 'Your prompt or input image contains content related to violence.',
            'suggestions': [
                'Avoid references to weapons, fighting, injuries, or violent actions',
                'Remove descriptions of physical harm or dangerous situations',
                'Focus on peaceful, non-threatening scenarios'
            ]
        },
        'sexual': {
            'title': 'Sexual Content Detected',
            'description': 'Your prompt or input image contains sexually suggestive or explicit content.',
            'suggestions': [
                'Remove sexually explicit descriptions or imagery',
                'Avoid suggestive poses, clothing, or scenarios',
                'Keep content appropriate for all audiences'
            ]
        },
        'derogatory': {
            'title': 'Derogatory Content Detected',
            'description': 'Your prompt or input image contains derogatory or hateful content.',
            'suggestions': [
                'Remove language or imagery that targets specific groups',
                'Avoid stereotypes or discriminatory content',
                'Use respectful and inclusive language'
            ]
        },
        'toxic': {
            'title': 'Toxic Content Detected',
            'description': 'Your prompt or input image contains toxic or harmful content.',
            'suggestions': [
                'Remove profanity, insults, or aggressive language',
                'Avoid content that could promote harm',
                'Use constructive and positive descriptions'
            ]
        },
        'personGeneration': {
            'title': 'Person Generation Restriction',
            'description': 'Your request may involve generating realistic depictions of people, which has restrictions.',
            'suggestions': [
                'Try using animated or artistic styles instead of photorealistic people',
                'Focus on environments, objects, or abstract concepts',
                'If people are essential, try describing them in more general terms'
            ]
        }
    }

    def _format_error_message(self, error_data: Any) -> str:
        """Format error data into a user-friendly message with safety filter detection."""
        try:
            # Handle dict errors (REST API format)
            if isinstance(error_data, dict):
                code = error_data.get('code', '')
                message = error_data.get('message', str(error_data))
                
                # Check for safety filter errors
                safety_info = self._detect_safety_filter(message)
                if safety_info:
                    return safety_info
                
                # Extract specific error types
                if 'duration' in message.lower() and 'supported durations' in message.lower():
                    # Parse duration error
                    match = re.search(r'supported durations are \[([^\]]+)\]', message)
                    if match:
                        valid = match.group(1)
                        return f"Invalid video duration. Valid durations are: {valid} seconds. Please update your settings."
                    return "Invalid video duration. Valid durations are: 4, 6, or 8 seconds. Please update your settings."
                
                if 'quota' in message.lower() or 'rate limit' in message.lower():
                    return f"API quota exceeded or rate limit reached. Please try again later. ({message})"
                
                if 'permission' in message.lower() or 'forbidden' in message.lower():
                    return f"Permission denied. Please check your API credentials and project access. ({message})"
                
                # Default format for dict errors
                return message or f"Error code {code}"
            
            # Handle string errors
            error_str = str(error_data)
            
            # Check for safety filter errors in string format
            safety_info = self._detect_safety_filter(error_str)
            if safety_info:
                return safety_info
            
            if 'duration' in error_str.lower():
                return "Invalid video duration. Valid durations are: 4, 6, or 8 seconds. Please update your settings."
            
            return error_str
        except Exception as e:
            self.log.error(f"Error formatting error message: {e}")
            return str(error_data)
    
    def _detect_safety_filter(self, message: str) -> Optional[str]:
        """Detect safety filter errors and return formatted guidance.
        
        Args:
            message: Error message to check for safety filter codes
            
        Returns:
            Formatted error message with safety guidance, or None if not a safety filter error
        """
        if not message or not isinstance(message, str):
            return None
        
        # Check for common safety filter error patterns
        lower_msg = message.lower()
        is_safety_error = (
            'violate' in lower_msg and ('usage guidelines' in lower_msg or 'policies' in lower_msg) or
            'safety' in lower_msg and 'filter' in lower_msg or
            'support code' in lower_msg or
            "couldn't be submitted" in lower_msg or
            'harmful' in lower_msg
        )
        
        if not is_safety_error:
            return None
        
        # Extract support codes from message (format: "Support codes: 15236754" or "code: 15236754")
        support_codes = re.findall(r'(?:support code|code)s?\s*:?\s*(\d{8})', message, re.IGNORECASE)
        
        if support_codes:
            # Get the first matching code's category
            for code in support_codes:
                category = self.SAFETY_FILTER_CODES.get(code)
                if category:
                    guidance = self.SAFETY_CATEGORY_GUIDANCE.get(category, {})
                    
                    # Build formatted message
                    formatted_msg = f"🛡️ **{guidance.get('title', 'Content Safety Filter')}**\n\n"
                    formatted_msg += f"{guidance.get('description', 'Your content triggered a safety filter.')}\n\n"
                    
                    if 'suggestions' in guidance:
                        formatted_msg += "**Suggestions to resolve:**\n"
                        for suggestion in guidance['suggestions']:
                            formatted_msg += f"- {suggestion}\n"
                        formatted_msg += "\n"
                    
                    formatted_msg += f"**Safety Category:** `{category}`\n"
                    formatted_msg += f"**Support Code:** `{code}`\n\n"
                    formatted_msg += "If you believe this is an error, you can report it to Google Cloud support with the code above."
                    
                    return formatted_msg
        
        # Generic safety filter message if no specific code found
        return (
            "🛡️ **Content Safety Filter Triggered**\n\n"
            "Your prompt or input image may contain content that violates safety guidelines.\n\n"
            "**Common issues:**\n"
            "- Violence, weapons, or dangerous content\n"
            "- Sexual or suggestive content\n"
            "- Derogatory or hateful language\n"
            "- Toxic or harmful content\n"
            "- Restrictions on generating realistic people\n\n"
            "**What to try:**\n"
            "- Rephrase your prompt to be more general and appropriate\n"
            "- Remove any potentially sensitive descriptions\n"
            "- Try a different reference image if using one\n\n"
            f"**Original message:** {message[:200]}..."
        )

    async def _start_video_generation(self, client: genai.Client, prompt: str, image_input: Optional[str] = None) -> Optional[Any]:
        """Start the video generation process using GenAI SDK and return the operation.

        If image_input is provided, it can be either a data URI (data:<mime>;base64,<data>)
        or an http(s) URL. The request will include an image field accordingly.
        """
        try:
            # Validate duration
            duration = self.valves.duration_seconds
            valid_durations = [4, 6, 8]
            if duration not in valid_durations:
                self.log.warning(f"Invalid duration {duration}s, must be one of {valid_durations}. Using 8s.")
                duration = 8
            
            # Build config for video generation
            config_kwargs = {
                "number_of_videos": 1,
                "fps": 24,
                "duration_seconds": duration,
                "enhance_prompt": True,
            }
            
            self.log.info(f"Generating {duration}s video with prompt: {prompt[:100]}...")
            
            # Add optional image if provided
            generate_kwargs = {
                "model": self.valves.MODEL_ID,
                "prompt": prompt,
                "config": types.GenerateVideosConfig(**config_kwargs),
            }
            
            # Handle image: SDK only supports GCS URIs, but REST API supports base64
            # For inline images, we'll use direct REST API call
            use_rest_api = False
            rest_image_data = None
            
            if image_input and isinstance(image_input, str) and len(image_input.strip()) > 0:
                if image_input.startswith("data:"):
                    # Parse data URI - need to use REST API for inline base64
                    try:
                        meta, b64data = image_input.split(",", 1)
                        if not b64data or len(b64data.strip()) == 0:
                            self.log.warning("Data URI contains no base64 data")
                        else:
                            mime_match = re.match(r"data:([^;]+);base64", meta)
                            mime_type = mime_match.group(1) if mime_match else "image/png"
                            
                            # SDK doesn't support inline base64, use REST API
                            use_rest_api = True
                            rest_image_data = {
                                "bytesBase64Encoded": b64data,
                                "mimeType": mime_type
                            }
                            self.log.info(f"Using REST API for inline image: {len(b64data)} chars, type: {mime_type}")
                    except Exception as e:
                        self.log.warning(f"Failed to parse image data URI: {e}")
                elif image_input.startswith("gs://"):
                    # GCS URI - SDK supports this
                    generate_kwargs["image"] = types.Image(gcs_uri=image_input, mime_type="image/png")
                    self.log.info(f"Added GCS image URI via SDK: {image_input[:100]}...")
                else:
                    self.log.warning(f"Image input not recognized as data URI or gs:// URL: {image_input[:100]}")
            else:
                self.log.info("No valid image input provided - generating video from text prompt only")

            # Call SDK or REST API depending on whether we have inline image
            loop = asyncio.get_event_loop()
            
            if use_rest_api and rest_image_data:
                # Use direct REST API call for inline base64 images
                operation = await loop.run_in_executor(
                    None,
                    lambda: self._generate_video_rest_api(client, prompt, rest_image_data, config_kwargs)
                )
            else:
                # Use SDK for text-only or GCS images
                operation = await loop.run_in_executor(
                    None,
                    lambda: client.models.generate_videos(**generate_kwargs)
                )
            
            return operation
        except Exception as e:
            self.log.error(f"Failed to start video generation: {e}")
            raise Exception(f"Failed to start video generation: {str(e)}")

    def _generate_video_rest_api(self, client, prompt: str, image_data: dict, config_kwargs: dict):
        """Make direct REST API call for video generation with inline base64 image.
        
        The Python SDK doesn't support inline base64 images, only GCS URIs.
        This method makes a direct HTTP request using google-auth for authentication.
        """
        import json
        import httpx
        from google.auth.transport.requests import Request as GoogleAuthRequest
        import google.auth
        
        # Build request payload per REST API documentation
        request_body = {
            "instances": [{
                "prompt": prompt,
                "image": image_data  # {"bytesBase64Encoded": "...", "mimeType": "..."}
            }],
            "parameters": {
                "sampleCount": config_kwargs.get("number_of_videos", 1),
                "aspectRatio": "16:9",  # Default aspect ratio
                "durationSeconds": config_kwargs.get("duration_seconds", 8),
                "fps": config_kwargs.get("fps", 24),
                "enhancePrompt": config_kwargs.get("enhance_prompt", True),
            }
        }
        
        self.log.info(f"Making REST API call with payload keys: {list(request_body.keys())}")
        self.log.info(f"Image data keys: {list(image_data.keys())}")
        self.log.info(f"Parameters: sampleCount={request_body['parameters']['sampleCount']}, durationSeconds={request_body['parameters']['durationSeconds']}, fps={request_body['parameters']['fps']}")
        
        # Get project and location
        project_id = self.valves.project_id or os.getenv("GOOGLE_CLOUD_PROJECT", "")
        location = self.valves.location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        model_id = self.valves.MODEL_ID
        
        url = f"https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{location}/publishers/google/models/{model_id}:predictLongRunning"
        
        self.log.info(f"REST API URL: {url}")
        
        # Get credentials with proper scopes for Vertex AI
        # Use GOOGLE_APPLICATION_CREDENTIALS environment variable
        credentials, _ = google.auth.default(
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        # Refresh token if needed
        auth_req = GoogleAuthRequest()
        if not credentials.valid:
            credentials.refresh(auth_req)
        
        # Get access token
        access_token = credentials.token
        
        # Make the HTTP request
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=utf-8"
        }
        
        self.log.info("Making authenticated HTTP POST request...")
        
        with httpx.Client(timeout=60.0) as http_client:
            response = http_client.post(
                url,
                headers=headers,
                json=request_body
            )
        
        self.log.info(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            error_body = response.text
            self.log.error(f"REST API error ({response.status_code}): {error_body}")
            # Try to parse and format error
            try:
                error_data = response.json()
                formatted_msg = self._format_error_message(error_data)
            except:
                formatted_msg = error_body[:500]  # Truncate long errors
            raise Exception(f"REST API error ({response.status_code}): {formatted_msg}")
        
        # Parse response to get operation
        response_data = response.json()
        self.log.info(f"Response keys: {list(response_data.keys())}")
        
        # Wrap the response in an operation-like object
        class OperationWrapper:
            def __init__(self, data):
                self.name = data.get('name', '')
                self.done = False
                self._data = data
                self._is_rest_api = True  # Flag to indicate this is from REST API
            
            def __str__(self):
                return f"Operation(name={self.name})"
        
        op = OperationWrapper(response_data)
        self.log.info(f"Created REST operation: {op.name}")
        return op

    def _resolve_prompt_image(self, image_ref: str) -> Optional[str]:
        """Resolve an image reference to either a data URI or an http(s) URL.

        - Supports Open WebUI file links: /api/v1/files/{id}/content
        - Returns data URI for local files to ensure accessibility from Veo API
        - Passes through existing data: URIs and http(s) URLs
        """
        try:
            if not image_ref or not isinstance(image_ref, str):
                return None

            # Already a data URI
            if image_ref.startswith("data:"):
                return image_ref

            # Open WebUI file API path
            m = re.search(r"/api/v1/files/([a-f0-9\-]+)/content", image_ref, re.IGNORECASE)
            if m and FilesDB and Storage:
                file_id = m.group(1)
                file_model = FilesDB.get_file_by_id(file_id)
                if file_model and file_model.path:
                    local_path = Storage.get_file(file_model.path)
                    with open(local_path, "rb") as f:
                        data_bytes = f.read()
                    mime_type = None
                    if file_model.meta and isinstance(file_model.meta, dict):
                        mime_type = file_model.meta.get("content_type") or file_model.meta.get("mime_type")
                    if not mime_type:
                        mime_type = mimetypes.guess_type(file_model.filename or local_path)[0] or "image/png"
                    b64 = base64.b64encode(data_bytes).decode("utf-8")
                    return f"data:{mime_type};base64,{b64}"

            # Direct URL
            if image_ref.startswith("http://") or image_ref.startswith("https://"):
                return image_ref

            self.log.warning(f"Unrecognized image reference format: {image_ref[:100]}")
            return None
        except FileNotFoundError as e:
            self.log.error(f"Image file not found: {e}")
            return None
        except Exception as e:
            self.log.error(f"Failed to resolve prompt image: {type(e).__name__}: {e}")
            return None

    def _extract_images_from_content(self, content: str) -> list[str]:
        """Extract image URLs from markdown or HTML content."""
        urls: list[str] = []
        try:
            # Markdown image syntax
            md = re.findall(r'!\[.*?\]\((data:image/[^)]+|https?://[^)]+|/[^)]+)\)', content)
            urls.extend(md)

            # HTML <img src="...">
            html = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', content, flags=re.IGNORECASE)
            urls.extend(html)

            # Bare Open WebUI file API paths
            file_api = re.findall(r'(?:https?://[^\s)]+)?(/api/v1/files/[a-f0-9\-]+/content)', content, flags=re.IGNORECASE)
            for m in file_api:
                urls.append(m if m.startswith('/') else f'/{m.lstrip("/")}')

            # Absolute image URLs
            abs_img = re.findall(r'https?://[^\s)]+\.(?:png|jpg|jpeg|webp|gif)(?:\?[^\s)]*)?', content, flags=re.IGNORECASE)
            urls.extend(abs_img)

            # Deduplicate
            seen = set()
            deduped = []
            for u in urls:
                if u not in seen:
                    seen.add(u)
                    deduped.append(u)
            return deduped
        except Exception:
            return urls

    async def _poll_video_status(self, client: genai.Client, operation: Any, event_emitter: Optional[Callable[[dict], Awaitable[None]]] = None) -> Optional[str]:
        """Poll the operation status until completion or timeout and return the video URL."""
        start_time = time.time()
        cached_video_data = None  # Store video data to survive operation refresh
        
        # Progressive status update intervals from valve configuration
        try:
            status_intervals = [int(x.strip()) for x in self.valves.PROGRESS_UPDATE_INTERVALS.split(',')]
        except Exception:
            status_intervals = [15, 30, 60, 120]  # Fallback to default
        next_status_index = 0

        loop = asyncio.get_event_loop()
        
        while time.time() - start_time < self.valves.MAX_POLL_TIME:
            elapsed = int(time.time() - start_time)
            
            # Show progress at progressive intervals
            if next_status_index < len(status_intervals):
                if elapsed >= status_intervals[next_status_index]:
                    await self.emit_status(
                        event_emitter, "info", f"Still generating video... ({elapsed}s elapsed)"
                    )
                    next_status_index += 1
            elif elapsed % 120 == 0 and elapsed > 0:
                # After 120s, update every 2 minutes
                await self.emit_status(
                    event_emitter, "info", f"Still generating video... ({elapsed}s elapsed)"
                )
            
            # Check if operation is done
            try:
                # Check if this is a REST API operation
                if hasattr(operation, '_is_rest_api') and operation._is_rest_api:
                    # Use REST API to poll status
                    operation = await loop.run_in_executor(
                        None,
                        lambda: self._poll_rest_operation(operation)
                    )
                    
                    if operation.done:
                        # Check for errors
                        if hasattr(operation, 'error') and operation.error:
                            error_data = operation.error
                            self.log.error(f"Generation error: {error_data}")
                            
                            # Parse error message for user-friendly display
                            error_msg = self._format_error_message(error_data)
                            await self.emit_status(event_emitter, "error", "❌ Generation failed", True)
                            self.log.error(f"Video generation failed: {error_msg}")
                            # Return error message with special prefix so caller can display it
                            return f"ERROR::{error_msg}"
                        
                        # Extract video bytes directly from REST result (no URL, inline base64)
                        if hasattr(operation, '_data') and operation._data:
                            video_data = self._extract_video_data_from_rest(operation._data)
                            if video_data:
                                # Store in instance variable (most reliable)
                                self._cached_video_data = video_data
                                # Also store in operation as backup
                                operation._video_data = video_data
                                self.log.info(f"Stored REST video data: {len(self._cached_video_data)} bytes")
                                # Return special marker to indicate we have inline data
                                return "__REST_INLINE_VIDEO__"
                            else:
                                error_msg = "Failed to extract video data from API response"
                                self.log.error(error_msg)
                                await self.emit_status(event_emitter, "error", "❌ Extraction failed", True)
                                return f"ERROR::{error_msg}"
                        return f"ERROR::Unknown error"
                else:
                    # SDK operation - use SDK polling
                    if operation.done:
                        # Check for errors
                        if hasattr(operation, 'error') and operation.error:
                            error_data = operation.error
                            self.log.error(f"Generation error: {error_data}")
                            
                            # Parse error message for user-friendly display
                            error_msg = self._format_error_message(error_data)
                            await self.emit_status(event_emitter, "error", "❌ Generation failed", True)
                            self.log.error(f"Video generation failed: {error_msg}")
                            # Return error message with special prefix so caller can display it
                            return f"ERROR::{error_msg}"
                        
                        # Extract video URL or bytes from SDK result
                        try:
                            if hasattr(operation, 'result') and operation.result:
                                result = operation.result
                                
                                # Try different possible structures
                                if hasattr(result, 'generated_videos'):
                                    generated_videos = result.generated_videos
                                    self.log.info(f"Found {len(generated_videos) if generated_videos else 0} generated video(s)")
                                    if generated_videos and len(generated_videos) > 0:
                                        video = generated_videos[0].video
                                        
                                        # Check for URL first
                                        if hasattr(video, 'uri') and video.uri:
                                            video_uri = video.uri
                                            self.log.info(f"Extracted video URL from uri: {video_uri}")
                                            return video_uri
                                        elif hasattr(video, 'url') and video.url:
                                            video_uri = video.url
                                            self.log.info(f"Extracted video URL from url: {video_uri}")
                                            return video_uri
                                        elif hasattr(video, 'gcs_uri') and video.gcs_uri:
                                            video_uri = video.gcs_uri
                                            self.log.info(f"Extracted video URL from gcs_uri: {video_uri}")
                                            return video_uri
                                        # Check for inline video bytes
                                        elif hasattr(video, 'video_bytes') and video.video_bytes:
                                            self.log.info(f"Found inline video_bytes: {len(video.video_bytes)} bytes")
                                            # Store in instance variable (most reliable)
                                            self._cached_video_data = video.video_bytes
                                            # Also store in operation as backup
                                            operation._video_data = video.video_bytes
                                            self.log.info(f"Stored video data: {len(self._cached_video_data)} bytes")
                                            return "__SDK_INLINE_VIDEO__"
                                        else:
                                            self.log.error(f"Video object has no valid uri/url/gcs_uri/video_bytes. URI: {getattr(video, 'uri', None)}")
                                elif hasattr(result, 'videos'):
                                    # Alternative structure
                                    videos = result.videos
                                    self.log.info(f"Found {len(videos) if videos else 0} video(s) in alternative structure")
                                    if videos and len(videos) > 0:
                                        video = videos[0]
                                        if hasattr(video, 'uri') and video.uri:
                                            return video.uri
                                        elif hasattr(video, 'url') and video.url:
                                            return video.url
                                        elif hasattr(video, 'gcs_uri') and video.gcs_uri:
                                            return video.gcs_uri
                                        elif hasattr(video, 'video_bytes') and video.video_bytes:
                                            self.log.info(f"Found inline video_bytes (alt): {len(video.video_bytes)} bytes")
                                            self._cached_video_data = video.video_bytes
                                            operation._video_data = video.video_bytes
                                            return "__SDK_INLINE_VIDEO__"
                                else:
                                    self.log.error(f"Result has neither generated_videos nor videos attribute")
                            else:
                                self.log.error("Operation has no result attribute or result is None")
                            
                            error_msg = "Failed to extract video URL from operation result"
                            self.log.error(error_msg)
                            await self.emit_status(event_emitter, "error", "❌ Extraction failed", True)
                            return f"ERROR::{error_msg}"
                        except Exception as e:
                            error_msg = f"Exception extracting video URL: {e}"
                            self.log.error(error_msg)
                            import traceback
                            self.log.error(traceback.format_exc())
                            await self.emit_status(event_emitter, "error", "❌ Extraction error", True)
                            return f"ERROR::{error_msg}"
                    else:
                        # Refresh SDK operation status
                        operation = await loop.run_in_executor(
                            None,
                            lambda: client.operations.get(operation)
                        )
            except Exception as e:
                self.log.error(f"Poll error: {str(e)}")
                # Don't fail immediately, could be transient

            await asyncio.sleep(self.valves.POLL_INTERVAL)

        # Timeout reached
        timeout_msg = (
            f"**Video generation timed out**\n\n"
            f"The video generation exceeded the maximum wait time of {self.valves.MAX_POLL_TIME} seconds.\n\n"
            f"**Possible solutions:**\n"
            f"- Increase the `MAX_POLL_TIME` setting in pipeline configuration\n"
            f"- Try a shorter video duration (4 or 6 seconds instead of 8)\n"
            f"- Simplify your prompt to reduce processing time\n\n"
            f"**Note:** Veo video generation typically takes 2-5 minutes, but complex requests may take longer."
        )
        self.log.error(timeout_msg)
        await self.emit_status(event_emitter, "error", "❌ Timeout", True)
        return f"ERROR::{timeout_msg}"

    def _poll_rest_operation(self, operation):
        """Poll REST API operation status using fetchPredictOperation endpoint."""
        import httpx
        from google.auth.transport.requests import Request as GoogleAuthRequest
        import google.auth
        
        # Get credentials with proper scopes
        credentials, _ = google.auth.default(
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        # Refresh if needed
        auth_req = GoogleAuthRequest()
        if not credentials.valid:
            credentials.refresh(auth_req)
        
        # Build fetch operation request
        project_id = self.valves.project_id or os.getenv("GOOGLE_CLOUD_PROJECT", "")
        location = self.valves.location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        model_id = self.valves.MODEL_ID
        
        url = f"https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{location}/publishers/google/models/{model_id}:fetchPredictOperation"
        
        headers = {
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json; charset=utf-8"
        }
        
        payload = {
            "operationName": operation.name
        }
        
        with httpx.Client() as http_client:
            response = http_client.post(url, headers=headers, json=payload)
        
        if response.status_code != 200:
            self.log.error(f"Poll REST API error: {response.text}")
            return operation  # Return unchanged
        
        data = response.json()
        
        # Update operation wrapper with new data
        operation._data = data
        operation.done = data.get('done', False)
        
        # Check for error
        if 'error' in data:
            operation.error = data['error']
        
        return operation
    
    def _extract_video_data_from_rest(self, data: dict) -> Optional[bytes]:
        """Extract video bytes from REST API response data.
        
        REST API returns base64 encoded video inline, not a URL.
        """
        try:
            # Log full response structure for debugging
            import json
            self.log.info(f"Full REST response (first 2000 chars): {json.dumps(data, indent=2, default=str)[:2000]}")
            
            # REST API response structure
            if 'response' in data:
                response = data['response']
                self.log.info(f"Response keys: {list(response.keys()) if isinstance(response, dict) else 'not a dict'}")
                
                # Extract from videos array with bytesBase64Encoded
                if isinstance(response, dict) and 'videos' in response:
                    videos = response['videos']
                    if videos and len(videos) > 0:
                        video = videos[0]
                        b64_data = video.get('bytesBase64Encoded')
                        if b64_data:
                            self.log.info(f"Extracting base64 video data ({len(b64_data)} chars)")
                            video_bytes = base64.b64decode(b64_data)
                            self.log.info(f"Decoded video: {len(video_bytes)} bytes")
                            return video_bytes
            
            self.log.error(f"Could not find video data in response. Top-level keys: {list(data.keys())}")
            if 'response' in data:
                self.log.error(f"Response content preview: {str(data['response'])[:500]}")
            return None
        except Exception as e:
            self.log.error(f"Failed to extract video data from REST response: {e}")
            import traceback
            self.log.error(traceback.format_exc())
            return None

    async def _download_video(self, video_url: str) -> Optional[bytes]:
        """Download the video bytes with timeout.
        
        The video URL from Veo is typically a signed URL that doesn't require authentication.
        """
        try:
            timeout = aiohttp.ClientTimeout(total=max(1, int(self.valves.DOWNLOAD_TIMEOUT)))
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(video_url) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        err_text = (await response.text())[:200]
                        self.log.error(f"Failed to download video: {response.status} - {err_text}")
                        return None
        except Exception as e:
            self.log.error(f"Error downloading video: {str(e)}")
            return None

    async def _save_video_file(self, video_data: bytes, prompt: str, operation_name: str, __user__: Optional[dict] = None, image_ref: Optional[str] = None) -> Optional[str]:
        """Save video data using Open WebUI's file database and storage provider."""
        try:
            from open_webui.models.files import Files, FileForm
            from open_webui.storage.provider import Storage
            import tempfile
            import uuid

            # Extract user information for C2PA manifest
            user_id = __user__.get("id") if __user__ else None
            user_name = "Unknown"
            if __user__:
                user_name = __user__.get("name") or __user__.get("email") or __user__.get("username") or "Unknown"

            # Sign video with C2PA if enabled (before saving)
            video_data = self._sign_video_with_c2pa(
                video_data=video_data,
                mime_type="video/mp4",
                prompt=prompt,
                duration_seconds=self.valves.duration_seconds,
                user={"id": user_id, "name": user_name},
                image_ref=image_ref
            )

            # Create safe filename
            safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (" ", "-", "_")).rstrip()
            safe_prompt = safe_prompt.replace(" ", "_")
            timestamp = int(time.time())
            filename = f"veo3_inline_{safe_prompt}_{timestamp}.mp4"

            # Temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(video_data)
                temp_file_path = temp_file.name

            try:
                # Upload storage
                with open(temp_file_path, "rb") as f:
                    _file_data, file_path = Storage.upload_file(
                        f,
                        filename,
                        {"content_type": "video/mp4", "source": "veo3_inline_pipeline"},
                    )

                # DB record
                file_id = str(uuid.uuid4())
                record = Files.insert_new_file(
                    user_id or "system",
                    FileForm(
                        id=file_id,
                        filename=filename,
                        path=file_path,
                        meta={
                            "name": filename,
                            "content_type": "video/mp4",
                            "size": len(video_data),
                            "source": "veo3_inline_pipeline",
                            "prompt": prompt[:100],
                            "operation_name": operation_name,
                        },
                    ),
                )

                if record:
                    return record.id
                return None
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        except ImportError as e:
            self.log.error(f"Failed to import Open WebUI components: {str(e)}")
            return None
        except Exception as e:
            self.log.error(f"Error saving video file: {str(e)}")
            return None