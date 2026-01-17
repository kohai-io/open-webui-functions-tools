"""
title: User Dashboard
author: open-webui
version: 1.3.0
description: Displays user statistics dashboard. Admins see all users, regular users see their own stats.
required_open_webui_version: 0.3.9
requirements: cryptography
"""

from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import core_schema
from typing import Optional, Callable, Awaitable, Any
import time
from datetime import datetime, timedelta
import aiohttp
import asyncio
import logging
import hashlib
import base64
import os
from cryptography.fernet import Fernet, InvalidToken

log = logging.getLogger(__name__)


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
        ENABLE_DASHBOARD: bool = Field(
            default=True,
            description="Enable/disable the user dashboard feature"
        )
        TRIGGER_KEYWORDS: str = Field(
            default="dashboard,stats,analytics,metrics",
            description="Comma-separated keywords that trigger the dashboard (case-insensitive)"
        )
        SHOW_ACTIVE_USERS: bool = Field(
            default=True,
            description="Show currently active users in the dashboard"
        )
        SHOW_CHAT_STATS: bool = Field(
            default=True,
            description="Show chat statistics"
        )
        SHOW_FEEDBACK_STATS: bool = Field(
            default=True,
            description="Show feedback/rating statistics"
        )
        SHOW_FILE_STATS: bool = Field(
            default=True,
            description="Show file storage statistics"
        )
        SHOW_MODEL_STATS: bool = Field(
            default=True,
            description="Show model usage breakdown"
        )
        TOP_USERS_COUNT: int = Field(
            default=10,
            description="Number of top users to show in admin dashboard"
        )
        TOP_MODELS_COUNT: int = Field(
            default=10,
            description="Number of top models to show in admin dashboard"
        )
        TOP_GROUPS_COUNT: int = Field(
            default=10,
            description="Number of top groups to show in admin dashboard"
        )
        ENABLE_LITELLM_COSTS: bool = Field(
            default=False,
            description="Enable LiteLLM cost tracking integration (requires LiteLLM proxy)"
        )
        LITELLM_BASE_URL: str = Field(
            default="http://localhost:4000",
            description="LiteLLM proxy base URL for cost API queries"
        )
        LITELLM_API_KEY: EncryptedStr = Field(
            default="",
            description="LiteLLM admin API key for spend queries (required if ENABLE_LITELLM_COSTS=true). Will be encrypted for security."
        )

    def __init__(self):
        self.type = "pipe"
        self.id = "user_dashboard"
        self.name = "User Dashboard"
        self.valves = self.Valves()

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
        """Main pipe function that generates dashboard based on user permissions."""
        
        if not self.valves.ENABLE_DASHBOARD:
            return body
        
        # Check if message triggers dashboard
        user_message = body.get("messages", [])[-1].get("content", "").lower().strip()
        triggers = [kw.strip().lower() for kw in self.valves.TRIGGER_KEYWORDS.split(",")]
        
        if not any(trigger in user_message for trigger in triggers):
            return body
        
        # Get user info
        user_id = __user__.get("id") if __user__ else None
        user_role = __user__.get("role", "user") if __user__ else "user"
        user_name = __user__.get("name", "User") if __user__ else "User"
        
        if not user_id:
            return "❌ Unable to retrieve user information."
        
        try:
            # Import Open WebUI models
            from open_webui.models.users import Users
            from open_webui.models.chats import Chats
            from open_webui.models.feedbacks import Feedbacks
            from open_webui.models.files import Files
            from open_webui.models.models import Models
            from open_webui.models.groups import Groups
            
            # Generate dashboard based on role
            if user_role == "admin":
                dashboard = await self._generate_admin_dashboard(
                    Users, Chats, Feedbacks, Files, Models, Groups, user_name
                )
            else:
                dashboard = self._generate_user_dashboard(
                    Users, Chats, Feedbacks, user_id, user_name
                )
            
            return dashboard
            
        except Exception as e:
            return f"❌ **Error generating dashboard:** {str(e)}"

    async def _fetch_litellm_costs(self, user_ids: list[str]) -> dict:
        """Fetch cost data from LiteLLM for given user IDs.
        
        Tries two endpoints:
        1. /customer/info?end_user_id=<id> - for end-user tracking
        2. /user/info?user_id=<id> - for key-based user tracking
        
        Returns: {user_id: {"spend": float, "error": str|None}}
        """
        if not self.valves.ENABLE_LITELLM_COSTS:
            return {}
        
        if not self.valves.LITELLM_API_KEY:
            log.warning("[DASHBOARD] LiteLLM costs enabled but no API key configured")
            return {}
        
        # Decrypt API key for use
        decrypted_key = EncryptedStr.decrypt(self.valves.LITELLM_API_KEY)
        
        costs = {}
        headers = {"Authorization": f"Bearer {decrypted_key}"}
        
        async with aiohttp.ClientSession() as session:
            for user_id in user_ids:
                try:
                    # Try /customer/info first (for end_user_id tracking)
                    url = f"{self.valves.LITELLM_BASE_URL}/customer/info?end_user_id={user_id}"
                    async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            data = await response.json()
                            spend_value = data.get("spend", 0.0)
                            log.info(f"[DASHBOARD] LiteLLM /customer/info for {user_id[:8]}...: spend={spend_value}, full_response={data}")
                            costs[user_id] = {
                                "spend": spend_value,
                                "error": None
                            }
                            continue
                        elif response.status == 500:
                            # 500 might mean user not found in customer tracking, try /user/info
                            try:
                                error_text = await response.text()
                                log.debug(f"[DASHBOARD] /customer/info failed (500): {error_text[:200]}")
                            except:
                                pass
                        else:
                            error_text = await response.text()
                            log.warning(f"[DASHBOARD] /customer/info returned {response.status} for {user_id}: {error_text[:200]}")
                    
                    # Fallback: Try /user/info (for user_id on keys)
                    url = f"{self.valves.LITELLM_BASE_URL}/user/info?user_id={user_id}"
                    async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            data = await response.json()
                            user_info = data.get("user_info", {})
                            spend_value = user_info.get("spend", 0.0) if isinstance(user_info, dict) else 0.0
                            log.info(f"[DASHBOARD] LiteLLM /user/info for {user_id[:8]}...: spend={spend_value}, full_response={data}")
                            costs[user_id] = {
                                "spend": spend_value,
                                "error": None
                            }
                        else:
                            error_text = await response.text()
                            log.warning(f"[DASHBOARD] Both LiteLLM endpoints failed for {user_id}. /user/info: {response.status}")
                            costs[user_id] = {"spend": 0.0, "error": f"Both endpoints failed (HTTP {response.status})"}
                            
                except asyncio.TimeoutError:
                    log.warning(f"[DASHBOARD] LiteLLM API timeout for user {user_id}")
                    costs[user_id] = {"spend": 0.0, "error": "timeout"}
                except Exception as e:
                    log.warning(f"[DASHBOARD] Failed to fetch LiteLLM cost for user {user_id}: {str(e)}")
                    costs[user_id] = {"spend": 0.0, "error": str(e)}
        
        return costs

    async def _generate_admin_dashboard(self, Users, Chats, Feedbacks, Files, Models, Groups, admin_name: str) -> str:
        """Generate comprehensive dashboard for administrators."""
        
        # Gather system-wide statistics
        all_users_data = Users.get_users()
        total_users = all_users_data.get("total", 0)
        all_users = all_users_data.get("users", [])
        
        # Calculate time-based metrics
        now = int(time.time())
        day_ago = now - 86400
        week_ago = now - 604800
        month_ago = now - 2592000
        
        users_today = sum(1 for u in all_users if u.created_at >= day_ago)
        users_this_week = sum(1 for u in all_users if u.created_at >= week_ago)
        users_this_month = sum(1 for u in all_users if u.created_at >= month_ago)
        
        active_last_24h = sum(1 for u in all_users if u.last_active_at >= day_ago)
        active_last_week = sum(1 for u in all_users if u.last_active_at >= week_ago)
        active_last_month = sum(1 for u in all_users if u.last_active_at >= month_ago)
        active_all_time = sum(1 for u in all_users if u.last_active_at > 0)
        
        # Get currently active users if available
        active_now_count = 0
        active_now_list = []
        if self.valves.SHOW_ACTIVE_USERS:
            try:
                from open_webui.socket.main import get_active_user_ids
                active_user_ids = get_active_user_ids()
                active_now_count = len(active_user_ids)
                active_now_list = [u.name for u in all_users if u.id in active_user_ids][:10]
            except:
                pass
        
        # Role breakdown
        admin_count = sum(1 for u in all_users if u.role == "admin")
        user_count = sum(1 for u in all_users if u.role == "user")
        pending_count = sum(1 for u in all_users if u.role == "pending")
        
        # Global chat statistics
        total_chats_system = 0
        total_chats_archived = 0
        total_chats_pinned = 0
        chat_activity_today = 0
        chat_activity_week = 0
        
        try:
            all_chats = Chats.get_chats()
            total_chats_system = len(all_chats)
            total_chats_archived = sum(1 for c in all_chats if c.archived)
            total_chats_pinned = sum(1 for c in all_chats if c.pinned)
            chat_activity_today = sum(1 for c in all_chats if c.updated_at >= day_ago)
            chat_activity_week = sum(1 for c in all_chats if c.updated_at >= week_ago)
            chat_activity_month = sum(1 for c in all_chats if c.updated_at >= month_ago)
        except Exception as e:
            pass
        
        # File statistics
        file_stats = ""
        if self.valves.SHOW_FILE_STATS:
            try:
                all_files = Files.get_files()
                total_files = len(all_files)
                total_size_bytes = sum(
                    f.meta.get("size", 0) if f.meta else 0 
                    for f in all_files
                )
                total_size_mb = total_size_bytes / (1024 * 1024)
                total_size_gb = total_size_mb / 1024
                
                files_today = sum(1 for f in all_files if f.created_at >= day_ago)
                files_this_week = sum(1 for f in all_files if f.created_at >= week_ago)
                
                # File type breakdown
                file_types = {}
                for f in all_files:
                    content_type = f.meta.get("content_type", "unknown") if f.meta else "unknown"
                    file_types[content_type] = file_types.get(content_type, 0) + 1
                
                top_file_types = sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:5]
                
                size_display = f"{total_size_gb:.2f} GB" if total_size_gb >= 1 else f"{total_size_mb:.2f} MB"
                
                file_stats = f"""\n\n## 📁 File Storage Statistics\n
- **Total Files:** {total_files}
- **Total Storage:** {size_display}
- **Uploaded Today:** {files_today}
- **Uploaded This Week:** {files_this_week}

### Top File Types
"""
                for file_type, count in top_file_types:
                    file_stats += f"- **{file_type}:** {count} files\n"
                
            except Exception as e:
                file_stats = f"\n\n⚠️ File statistics unavailable: {str(e)}\n"
        
        # Model type breakdown (foundational vs user-created vs admin-provided)
        model_type_stats = ""
        try:
            import logging
            log = logging.getLogger(__name__)
            log.info("[DASHBOARD] Starting model registry statistics")
            
            # Get custom models (assistants built on base models)
            log.info("[DASHBOARD] Fetching custom models from database")
            all_custom = Models.get_models()
            log.info(f"[DASHBOARD] Found {len(all_custom)} custom models total")
            
            custom_models = [m for m in all_custom if m.is_active]
            log.info(f"[DASHBOARD] Found {len(custom_models)} active custom models")
            
            # Separate by creator role with safe attribute access
            admin_models = []
            user_created_models = []
            
            for idx, m in enumerate(custom_models):
                try:
                    has_user = hasattr(m, 'user') and m.user is not None
                    user_id = getattr(getattr(m, 'user', None), 'id', None) if has_user else None
                    user_role = getattr(getattr(m, 'user', None), 'role', None) if has_user else None
                    
                    # Only log safe text fields - no binary data like profile_image_url
                    log.info(f"[DASHBOARD] Processing custom model {idx}: id={m.id}, user_id={user_id}, user_role={user_role}")
                    
                    if has_user:
                        if user_role == 'admin':
                            admin_models.append(m)
                        else:
                            user_created_models.append(m)
                    else:
                        log.warning(f"[DASHBOARD]   Model {m.id} has no user attribute")
                        # If no user, assume it's a user-created model
                        user_created_models.append(m)
                except Exception as model_err:
                    log.error(f"[DASHBOARD] Error processing model {idx}: {model_err}", exc_info=True)
                    continue
            
            log.info(f"[DASHBOARD] Categorized: {len(admin_models)} admin, {len(user_created_models)} user")
            
            # For foundational models, we'll count unique base models from database
            log.info("[DASHBOARD] Fetching foundational models")
            all_foundational = Models.get_base_models()
            log.info(f"[DASHBOARD] Found {len(all_foundational)} foundational models total")
            
            foundational_models = [m for m in all_foundational if m.is_active]
            log.info(f"[DASHBOARD] Found {len(foundational_models)} active foundational models")
            
            # Debug: Log sample of active foundational models to backend logs
            all_foundational_ids = sorted([m.id for m in foundational_models]) if foundational_models else []
            if all_foundational_ids:
                sample_size = min(20, len(all_foundational_ids))
                log.info(f"[DASHBOARD] Sample active foundational models (first {sample_size} of {len(all_foundational_ids)}): {', '.join(all_foundational_ids[:sample_size])}")
                if len(all_foundational_ids) > 20:
                    log.info(f"[DASHBOARD] See dashboard output for complete list of all {len(all_foundational_ids)} models")
            
            total_foundational = len(foundational_models)
            total_custom = len(custom_models)
            total_admin = len(admin_models)
            total_user = len(user_created_models)
            total_models = total_foundational + total_custom
            sample_custom_admin = [m.id for m in admin_models[:5]] if admin_models else []
            sample_custom_user = [m.id for m in user_created_models[:5]] if user_created_models else []
            
            # Format foundational models in groups of 10 for readability
            foundational_display = ""
            if all_foundational_ids:
                for i in range(0, len(all_foundational_ids), 10):
                    chunk = all_foundational_ids[i:i+10]
                    foundational_display += f"{i+1:3d}-{min(i+10, len(all_foundational_ids)):3d}: {', '.join(chunk)}\n"
            else:
                foundational_display = "None\n"
            
            model_type_stats = f"""\n\n## 🏗️ Model Registry

### Model Type Breakdown (Database Registry)
- **Total Registered Active Models:** {total_models}
- **Foundational Models:** {total_foundational} ({(total_foundational/total_models*100) if total_models > 0 else 0:.1f}%)
  - Base models from providers (Ollama, OpenAI, etc.)
  - Includes models synced from backends that may no longer be available
- **User-Created Models:** {total_custom} ({(total_custom/total_models*100) if total_models > 0 else 0:.1f}%)
  - Custom assistants/agents built on foundational models
  - Created by admins: {total_admin}
  - Created by users: {total_user}

### Debug Info - Active Foundational Models ({total_foundational} total)
**All {total_foundational} active foundational models in database:**
```
{foundational_display}```

**Cleanup Instructions:**
1. Compare above list with your UI model selector (Settings → Models page)
2. Models listed here but NOT in UI dropdown are orphaned (backend removed)
3. Toggle off orphaned models in Settings → Admin Panel → Settings → Models
4. This will reduce the count from {total_foundational} to ~42 (currently available)

**Sample Admin Custom Models:**
```
{', '.join(sample_custom_admin) if sample_custom_admin else 'None'}
```

**Sample User Custom Models:**
```
{', '.join(sample_custom_user) if sample_custom_user else 'None'}
```

**Note:** This shows all models with `is_active=True` in the database ({total_models} total). The UI model selector may show fewer ({total_foundational} foundational models here vs ~42 typically visible) because it only displays models currently available from active backends (Ollama/OpenAI APIs). Models listed above but not in the UI dropdown are registered but their backends are disconnected or the models were removed from the backend.

**Total in database:** {len(all_foundational)} foundational (including disabled), {len(all_custom)} custom
"""
            
            # Show top admin and user models by name
            if admin_models or user_created_models:
                model_type_stats += "\n### Recent Custom Models\n"
                
                if admin_models:
                    recent_admin = sorted(admin_models, key=lambda m: m.created_at, reverse=True)[:3]
                    model_type_stats += "\n**Admin Models:**\n"
                    for model in recent_admin:
                        model_type_stats += f"- {model.name} (ID: `{model.id}`)\n"
                
                if user_created_models:
                    recent_user = sorted(user_created_models, key=lambda m: m.created_at, reverse=True)[:3]
                    model_type_stats += "\n**User Models:**\n"
                    for model in recent_user:
                        model_type_stats += f"- {model.name} (ID: `{model.id}`)\n"
        
        except Exception as e:
            model_type_stats = f"\n\n⚠️ Model registry statistics unavailable: {str(e)}\n"
        
        # Model usage statistics
        model_stats = ""
        if self.valves.SHOW_MODEL_STATS:
            try:
                all_chats = Chats.get_chats()
                model_usage = {}  # {model_id: {'chats': count, 'prompt_tokens': input, 'completion_tokens': output}}
                
                for chat in all_chats:
                    # Extract model info from chat data
                    chat_data = chat.chat if hasattr(chat, 'chat') else {}
                    
                    # Check various locations where model info might be stored
                    models_used = {}  # {model_id: {'prompt': tokens, 'completion': tokens}}
                    
                    # Check messages for model information and token usage
                    if isinstance(chat_data, dict):
                        messages = chat_data.get("messages", [])
                        if isinstance(messages, list):
                            for msg in messages:
                                if isinstance(msg, dict):
                                    model_id = msg.get("model") or msg.get("modelId")
                                    if model_id:
                                        # Extract separate token counts from message
                                        prompt_tokens = 0
                                        completion_tokens = 0
                                        if "usage" in msg and isinstance(msg["usage"], dict):
                                            prompt_tokens = msg["usage"].get("prompt_tokens", 0)
                                            completion_tokens = msg["usage"].get("completion_tokens", 0)
                                        
                                        if model_id not in models_used:
                                            models_used[model_id] = {'prompt': 0, 'completion': 0}
                                        models_used[model_id]['prompt'] += prompt_tokens
                                        models_used[model_id]['completion'] += completion_tokens
                        
                        # Check top-level model field
                        if chat_data.get("model"):
                            model_id = chat_data.get("model")
                            if model_id not in models_used:
                                models_used[model_id] = {'prompt': 0, 'completion': 0}
                        if chat_data.get("models"):
                            if isinstance(chat_data.get("models"), list):
                                for model_id in chat_data.get("models"):
                                    if model_id not in models_used:
                                        models_used[model_id] = {'prompt': 0, 'completion': 0}
                    
                    # Aggregate model usage
                    for model_id, token_data in models_used.items():
                        if model_id not in model_usage:
                            model_usage[model_id] = {'chats': 0, 'prompt_tokens': 0, 'completion_tokens': 0}
                        model_usage[model_id]['chats'] += 1
                        model_usage[model_id]['prompt_tokens'] += token_data['prompt']
                        model_usage[model_id]['completion_tokens'] += token_data['completion']
                
                if model_usage:
                    # Sort by chat count
                    top_models = sorted(model_usage.items(), key=lambda x: x[1]['chats'], reverse=True)[:self.valves.TOP_MODELS_COUNT]
                    
                    total_chats = sum(data['chats'] for data in model_usage.values())
                    total_prompt_tokens = sum(data['prompt_tokens'] for data in model_usage.values())
                    total_completion_tokens = sum(data['completion_tokens'] for data in model_usage.values())
                    total_tokens = total_prompt_tokens + total_completion_tokens
                    unique_models = len(model_usage)
                    
                    # Format token count nicely
                    def format_tokens(n):
                        if n >= 1_000_000:
                            return f"{n/1_000_000:.1f}M"
                        elif n >= 1_000:
                            return f"{n/1_000:.1f}K"
                        else:
                            return str(n)
                    
                    model_stats = f"""\n\n## 🤖 Model Usage Statistics\n
- **Unique Models Used:** {unique_models}
- **Total Chats:** {total_chats:,}
- **Total Tokens:** {total_tokens:,} ({format_tokens(total_tokens)})
  - Input (Prompt): {total_prompt_tokens:,} ({format_tokens(total_prompt_tokens)})
  - Output (Completion): {total_completion_tokens:,} ({format_tokens(total_completion_tokens)})

### Top Models by Usage
| Rank | Model | Chats | Input Tokens | Output Tokens | Chat % |
|------|-------|-------|--------------|---------------|--------|
"""
                    for idx, (model_id, data) in enumerate(top_models, 1):
                        chat_pct = (data['chats'] / total_chats * 100) if total_chats > 0 else 0
                        model_stats += f"| {idx} | {model_id} | {data['chats']:,} | {format_tokens(data['prompt_tokens'])} | {format_tokens(data['completion_tokens'])} | {chat_pct:.1f}% |\n"
                else:
                    model_stats = "\n\n## 🤖 Model Usage Statistics\n\n*No model usage data available*\n\n*Note: Token counts require usage data in message history. Older chats may show 0 tokens.*\n"
                
            except Exception as e:
                model_stats = f"\n\n⚠️ Model statistics unavailable: {str(e)}\n"
        
        # Chat statistics (per-user breakdown with token usage)
        chat_stats = ""
        if self.valves.SHOW_CHAT_STATS:
            try:
                user_chat_data = []
                for user in all_users:
                    chats = Chats.get_chats_by_user_id(user.id)
                    chat_count = len(chats)
                    
                    if chat_count > 0:
                        # Extract token usage from user's chats
                        prompt_tokens = 0
                        completion_tokens = 0
                        
                        for chat in chats:
                            chat_data = chat.chat if hasattr(chat, 'chat') else {}
                            if isinstance(chat_data, dict):
                                messages = chat_data.get("messages", [])
                                if isinstance(messages, list):
                                    for msg in messages:
                                        if isinstance(msg, dict) and "usage" in msg:
                                            usage = msg.get("usage", {})
                                            if isinstance(usage, dict):
                                                prompt_tokens += usage.get("prompt_tokens", 0)
                                                completion_tokens += usage.get("completion_tokens", 0)
                        
                        user_chat_data.append({
                            'name': user.name,
                            'chats': chat_count,
                            'prompt_tokens': prompt_tokens,
                            'completion_tokens': completion_tokens,
                            'user_id': user.id
                        })
                
                # Sort by chat count
                user_chat_data.sort(key=lambda x: x['chats'], reverse=True)
                top_users = user_chat_data[:self.valves.TOP_USERS_COUNT]
                
                if top_users:
                    # Fetch LiteLLM costs for top users
                    user_costs = {}
                    if self.valves.ENABLE_LITELLM_COSTS:
                        user_ids = [u['user_id'] for u in top_users]
                        user_costs = await self._fetch_litellm_costs(user_ids)
                        log.info(f"[DASHBOARD] Fetched LiteLLM costs for {len(user_costs)} users")
                    
                    # Add costs to user data
                    for user_data in top_users:
                        cost_data = user_costs.get(user_data['user_id'], {})
                        user_data['spend'] = cost_data.get('spend', 0.0)
                    
                    # Calculate totals
                    total_user_chats = sum(u['chats'] for u in user_chat_data)
                    total_user_prompt = sum(u['prompt_tokens'] for u in user_chat_data)
                    total_user_completion = sum(u['completion_tokens'] for u in user_chat_data)
                    total_user_tokens = total_user_prompt + total_user_completion
                    total_user_spend = sum(u['spend'] for u in top_users) if self.valves.ENABLE_LITELLM_COSTS else 0.0
                    
                    # Format tokens helper
                    def format_tokens(n):
                        if n >= 1_000_000:
                            return f"{n/1_000_000:.1f}M"
                        elif n >= 1_000:
                            return f"{n/1_000:.1f}K"
                        else:
                            return str(n)
                    
                    chat_stats = f"""\n\n## 💬 Top Users by Activity

**Total Across All Users:**
- Chats: {total_user_chats:,}
- Tokens: {total_user_tokens:,} ({format_tokens(total_user_tokens)})
  - Input: {total_user_prompt:,} ({format_tokens(total_user_prompt)})
  - Output: {total_user_completion:,} ({format_tokens(total_user_completion)})"""
                    
                    if self.valves.ENABLE_LITELLM_COSTS:
                        chat_stats += f"\n- **Total Spend (LiteLLM):** ${total_user_spend:.4f}"
                    
                    chat_stats += "\n\n"
                    
                    if self.valves.ENABLE_LITELLM_COSTS:
                        chat_stats += "| Rank | User | Chats | Input Tokens | Output Tokens | Total Spend |\n"
                        chat_stats += "|------|------|-------|--------------|---------------|-------------|\n"
                        for idx, data in enumerate(top_users, 1):
                            chat_stats += f"| {idx} | {data['name']} | {data['chats']:,} | {format_tokens(data['prompt_tokens'])} | {format_tokens(data['completion_tokens'])} | ${data['spend']:.4f} |\n"
                    else:
                        chat_stats += "| Rank | User | Chats | Input Tokens | Output Tokens |\n"
                        chat_stats += "|------|------|-------|--------------|---------------|\n"
                        for idx, data in enumerate(top_users, 1):
                            chat_stats += f"| {idx} | {data['name']} | {data['chats']:,} | {format_tokens(data['prompt_tokens'])} | {format_tokens(data['completion_tokens'])} |\n"
            except Exception as e:
                chat_stats = f"\n\n⚠️ Chat statistics unavailable: {str(e)}\n"
        
        # Group statistics (groups with member activity)
        group_stats = ""
        try:
            all_groups = Groups.get_groups()
            
            if all_groups:
                group_data = []
                
                for group in all_groups:
                    member_count = len(group.user_ids) if group.user_ids else 0
                    
                    if member_count > 0:
                        # Calculate chat and token stats for group members
                        group_chats = 0
                        group_prompt_tokens = 0
                        group_completion_tokens = 0
                        
                        for user_id in group.user_ids:
                            chats = Chats.get_chats_by_user_id(user_id)
                            group_chats += len(chats)
                            
                            # Extract token usage from member chats
                            for chat in chats:
                                chat_data = chat.chat if hasattr(chat, 'chat') else {}
                                if isinstance(chat_data, dict):
                                    messages = chat_data.get("messages", [])
                                    if isinstance(messages, list):
                                        for msg in messages:
                                            if isinstance(msg, dict) and "usage" in msg:
                                                usage = msg.get("usage", {})
                                                if isinstance(usage, dict):
                                                    group_prompt_tokens += usage.get("prompt_tokens", 0)
                                                    group_completion_tokens += usage.get("completion_tokens", 0)
                        
                        group_data.append({
                            'name': group.name,
                            'members': member_count,
                            'member_ids': group.user_ids,
                            'chats': group_chats,
                            'prompt_tokens': group_prompt_tokens,
                            'completion_tokens': group_completion_tokens
                        })
                
                # Sort by member count
                group_data.sort(key=lambda x: x['members'], reverse=True)
                top_groups = group_data[:self.valves.TOP_GROUPS_COUNT]
                
                if top_groups:
                    # Fetch LiteLLM costs for all group members
                    if self.valves.ENABLE_LITELLM_COSTS:
                        all_member_ids = set()
                        for group in top_groups:
                            all_member_ids.update(group['member_ids'])
                        member_costs = await self._fetch_litellm_costs(list(all_member_ids))
                        log.info(f"[DASHBOARD] Fetched LiteLLM costs for {len(member_costs)} group members")
                        
                        # Calculate group spend
                        for group in top_groups:
                            group_spend = sum(
                                member_costs.get(uid, {}).get('spend', 0.0)
                                for uid in group['member_ids']
                            )
                            group['spend'] = group_spend
                    else:
                        for group in top_groups:
                            group['spend'] = 0.0
                    
                    # Calculate totals
                    total_groups = len(all_groups)
                    total_active_groups = len(group_data)
                    total_group_chats = sum(g['chats'] for g in group_data)
                    total_group_prompt = sum(g['prompt_tokens'] for g in group_data)
                    total_group_completion = sum(g['completion_tokens'] for g in group_data)
                    total_group_tokens = total_group_prompt + total_group_completion
                    total_group_spend = sum(g['spend'] for g in top_groups) if self.valves.ENABLE_LITELLM_COSTS else 0.0
                    
                    # Format tokens helper
                    def format_tokens(n):
                        if n >= 1_000_000:
                            return f"{n/1_000_000:.1f}M"
                        elif n >= 1_000:
                            return f"{n/1_000:.1f}K"
                        else:
                            return str(n)
                    
                    group_stats = f"""\n\n## 👥 Top User Groups

**Total Groups:** {total_groups} ({total_active_groups} with members)
**Aggregate Activity:**
- Chats: {total_group_chats:,}
- Tokens: {total_group_tokens:,} ({format_tokens(total_group_tokens)})
  - Input: {total_group_prompt:,} ({format_tokens(total_group_prompt)})
  - Output: {total_group_completion:,} ({format_tokens(total_group_completion)})"""
                    
                    if self.valves.ENABLE_LITELLM_COSTS:
                        group_stats += f"\n- **Total Spend (LiteLLM):** ${total_group_spend:.4f}"
                    
                    group_stats += "\n\n"
                    
                    if self.valves.ENABLE_LITELLM_COSTS:
                        group_stats += "| Rank | Group | Members | Chats | Input Tokens | Output Tokens | Total Spend |\n"
                        group_stats += "|------|-------|---------|-------|--------------|---------------|-------------|\n"
                        for idx, data in enumerate(top_groups, 1):
                            group_stats += f"| {idx} | {data['name']} | {data['members']} | {data['chats']:,} | {format_tokens(data['prompt_tokens'])} | {format_tokens(data['completion_tokens'])} | ${data['spend']:.4f} |\n"
                    else:
                        group_stats += "| Rank | Group | Members | Chats | Input Tokens | Output Tokens |\n"
                        group_stats += "|------|-------|---------|-------|--------------|---------------|\n"
                        for idx, data in enumerate(top_groups, 1):
                            group_stats += f"| {idx} | {data['name']} | {data['members']} | {data['chats']:,} | {format_tokens(data['prompt_tokens'])} | {format_tokens(data['completion_tokens'])} |\n"
        except Exception as e:
            group_stats = f"\n\n⚠️ Group statistics unavailable: {str(e)}\n"
        
        # Feedback statistics
        feedback_stats = ""
        if self.valves.SHOW_FEEDBACK_STATS:
            try:
                all_feedbacks = Feedbacks.get_all_feedbacks()
                total_feedbacks = len(all_feedbacks)
                feedbacks_today = sum(1 for f in all_feedbacks if f.created_at >= day_ago)
                feedbacks_this_week = sum(1 for f in all_feedbacks if f.created_at >= week_ago)
                
                feedback_stats = f"\n\n## ⭐ Feedback Statistics\n\n"
                feedback_stats += f"- **Total Feedbacks:** {total_feedbacks}\n"
                feedback_stats += f"- **Today:** {feedbacks_today}\n"
                feedback_stats += f"- **This Week:** {feedbacks_this_week}\n"
            except Exception as e:
                feedback_stats = f"\n\n⚠️ Feedback statistics unavailable: {str(e)}\n"
        
        # Most recent users
        recent_users = sorted(all_users, key=lambda u: u.created_at, reverse=True)[:5]
        recent_users_list = "\n".join([
            f"- **{u.name}** ({u.role}) - {self._format_timestamp(u.created_at)}"
            for u in recent_users
        ])
        
        # Most active users (by last_active_at)
        most_active = sorted(all_users, key=lambda u: u.last_active_at, reverse=True)[:5]
        most_active_list = "\n".join([
            f"- **{u.name}** - {self._format_relative_time(u.last_active_at)}"
            for u in most_active
        ])
        
        # Build dashboard
        dashboard = f"""# 📊 Admin Dashboard

**Generated for:** {admin_name} (Administrator)  
**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 👥 User Statistics

### Overview
- **Total Users:** {total_users}
- **Admins:** {admin_count}
- **Regular Users:** {user_count}
- **Pending:** {pending_count}

### New User Registrations
- **Today:** {users_today}
- **This Week:** {users_this_week}
- **This Month:** {users_this_month}

### Active Users
- **Last 24h:** {active_last_24h}
- **Last Week:** {active_last_week}
- **Last Month:** {active_last_month}
- **All Time:** {active_all_time}

---

## 💬 System-Wide Chat Statistics

- **Total Chat Sessions:** {total_chats_system}
- **Pinned Chats:** {total_chats_pinned}
- **Archived Chats:** {total_chats_archived}

### Chat Updates
- **Today:** {chat_activity_today}
- **This Week:** {chat_activity_week}
- **This Month:** {chat_activity_month}
- **All Time:** {total_chats_system}
"""
        
        if self.valves.SHOW_ACTIVE_USERS and active_now_count > 0:
            dashboard += f"- **Online Now:** {active_now_count}\n"
            if active_now_list:
                dashboard += f"  - {', '.join(active_now_list[:5])}"
                if len(active_now_list) > 5:
                    dashboard += f" (+{len(active_now_list) - 5} more)"
                dashboard += "\n"
        
        dashboard += f"""
---

## 🕐 Recent Activity

### Most Recent Users
{recent_users_list}

### Most Active Users
{most_active_list}
"""
        
        # Cost Analytics (LiteLLM)
        cost_analytics = ""
        if self.valves.ENABLE_LITELLM_COSTS:
            try:
                # Fetch costs for all users
                all_user_ids = [u.id for u in all_users]
                all_user_costs = await self._fetch_litellm_costs(all_user_ids)
                
                total_platform_spend = sum(
                    cost_data.get('spend', 0.0) 
                    for cost_data in all_user_costs.values()
                )
                
                users_with_spend = sum(1 for cost_data in all_user_costs.values() if cost_data.get('spend', 0.0) > 0)
                avg_spend_per_user = total_platform_spend / users_with_spend if users_with_spend > 0 else 0.0
                
                # Find top spenders
                user_spend_list = [
                    (u.name, all_user_costs.get(u.id, {}).get('spend', 0.0))
                    for u in all_users
                ]
                top_spenders = sorted(user_spend_list, key=lambda x: x[1], reverse=True)[:5]
                
                cost_analytics = f"""\n\n## 💰 Cost Analytics (LiteLLM)

**Platform-Wide Spending:**
- **Total Spend:** ${total_platform_spend:.2f}
- **Users with Activity:** {users_with_spend} / {total_users}
- **Average per Active User:** ${avg_spend_per_user:.2f}

**Top 5 Spenders:**
"""
                for idx, (name, spend) in enumerate(top_spenders, 1):
                    if spend > 0:
                        cost_analytics += f"{idx}. **{name}** - ${spend:.4f}\n"
                
                log.info(f"[DASHBOARD] Platform spend: ${total_platform_spend:.2f} across {users_with_spend} users")
                
            except Exception as e:
                cost_analytics = f"\n\n## 💰 Cost Analytics (LiteLLM)\n\n⚠️ Unable to fetch cost data: {str(e)}\n"
                log.error(f"[DASHBOARD] Failed to fetch cost analytics: {str(e)}")
        
        dashboard += file_stats
        dashboard += cost_analytics
        dashboard += model_type_stats
        dashboard += model_stats
        dashboard += chat_stats
        dashboard += group_stats
        dashboard += feedback_stats
        
        dashboard += "\n\n---\n*💡 Tip: Use this dashboard to monitor platform usage and user engagement.*"
        
        return dashboard

    def _generate_user_dashboard(
        self, Users, Chats, Feedbacks, user_id: str, user_name: str
    ) -> str:
        """Generate personal dashboard for regular users."""
        
        # Get user info
        user = Users.get_user_by_id(user_id)
        if not user:
            return "❌ Unable to retrieve user information."
        
        # Calculate account age
        account_age_days = (int(time.time()) - user.created_at) // 86400
        member_since = self._format_timestamp(user.created_at)
        last_active = self._format_relative_time(user.last_active_at)
        
        # Chat statistics
        chat_stats = ""
        if self.valves.SHOW_CHAT_STATS:
            try:
                chats = Chats.get_chats_by_user_id(user_id)
                total_chats = len(chats)
                archived_chats = len([c for c in chats if c.archived])
                pinned_chats = len([c for c in chats if c.pinned])
                
                # Recent chats
                recent_chats = sorted(chats, key=lambda c: c.updated_at, reverse=True)[:5]
                
                # Calculate time-based metrics
                now = int(time.time())
                day_ago = now - 86400
                week_ago = now - 604800
                
                chats_today = sum(1 for c in chats if c.updated_at >= day_ago)
                chats_this_week = sum(1 for c in chats if c.updated_at >= week_ago)
                
                chat_stats = f"""
## 💬 Your Chat Statistics

- **Total Chats:** {total_chats}
- **Pinned:** {pinned_chats}
- **Archived:** {archived_chats}
- **Active Today:** {chats_today}
- **Active This Week:** {chats_this_week}

### Recent Chats
"""
                for chat in recent_chats[:5]:
                    chat_stats += f"- **{chat.title}** - {self._format_relative_time(chat.updated_at)}\n"
                
            except Exception as e:
                chat_stats = f"\n⚠️ Chat statistics unavailable: {str(e)}\n"
        
        # Feedback statistics
        feedback_stats = ""
        if self.valves.SHOW_FEEDBACK_STATS:
            try:
                feedbacks = Feedbacks.get_feedbacks_by_user_id(user_id)
                total_feedbacks = len(feedbacks)
                
                if total_feedbacks > 0:
                    feedback_stats = f"""
## ⭐ Your Feedback Activity

- **Total Feedbacks Submitted:** {total_feedbacks}
- **Most Recent:** {self._format_relative_time(feedbacks[0].created_at) if feedbacks else 'N/A'}
"""
            except Exception as e:
                feedback_stats = f"\n⚠️ Feedback statistics unavailable: {str(e)}\n"
        
        # Build dashboard
        dashboard = f"""# 📊 Your Dashboard

**Welcome back, {user_name}!**  
**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 👤 Account Information

- **Member Since:** {member_since}
- **Account Age:** {account_age_days} days
- **Last Active:** {last_active}
- **Role:** {user.role.capitalize()}
"""
        
        if user.email:
            dashboard += f"- **Email:** {user.email}\n"
        
        dashboard += chat_stats
        dashboard += feedback_stats
        
        dashboard += "\n\n---\n*💡 Tip: Keep engaging with the platform to build your chat history!*"
        
        return dashboard

    def _format_timestamp(self, timestamp: int) -> str:
        """Convert Unix timestamp to readable date."""
        try:
            return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        except:
            return "N/A"

    def _format_relative_time(self, timestamp: int) -> str:
        """Convert Unix timestamp to relative time (e.g., '2 hours ago')."""
        try:
            now = int(time.time())
            diff = now - timestamp
            
            if diff < 60:
                return "just now"
            elif diff < 3600:
                minutes = diff // 60
                return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
            elif diff < 86400:
                hours = diff // 3600
                return f"{hours} hour{'s' if hours != 1 else ''} ago"
            elif diff < 604800:
                days = diff // 86400
                return f"{days} day{'s' if days != 1 else ''} ago"
            elif diff < 2592000:
                weeks = diff // 604800
                return f"{weeks} week{'s' if weeks != 1 else ''} ago"
            else:
                months = diff // 2592000
                return f"{months} month{'s' if months != 1 else ''} ago"
        except:
            return "N/A"
