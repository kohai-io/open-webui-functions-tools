"""
title: Prompt Scheduler
author: open-webui
version: 1.5.6
description: Allows models to create, manage, and monitor scheduled prompts. Enables AI-driven automation of recurring tasks.
required_open_webui_version: 0.3.9
"""

from pydantic import BaseModel, Field
from typing import Optional, List
import logging
import time

log = logging.getLogger(__name__)


class Tools:
    class Valves(BaseModel):
        ALLOW_CREATE: bool = Field(
            default=True,
            description="Allow models to create new scheduled prompts"
        )
        ALLOW_DELETE: bool = Field(
            default=False,
            description="Allow models to delete scheduled prompts (disabled by default for safety)"
        )
        DEFAULT_MODEL_ID: str = Field(
            default="",
            description="Default model ID to use when creating prompts. Leave empty to require explicit model selection."
        )
        MAX_SCHEDULES_PER_USER: int = Field(
            default=50,
            description="Maximum number of scheduled prompts per user"
        )

    def __init__(self):
        self.valves = self.Valves()

    def _format_cron_description(self, cron: str) -> str:
        """Convert cron expression to human-readable description."""
        parts = cron.split()
        if len(parts) != 5:
            return cron
        
        minute, hour, day, month, weekday = parts
        
        # Common patterns
        if cron == "* * * * *":
            return "Every minute"
        if cron == "0 * * * *":
            return "Every hour"
        if cron == "0 0 * * *":
            return "Daily at midnight"
        if cron == "0 9 * * 1-5":
            return "Weekdays at 9:00 AM"
        if cron == "0 9 * * *":
            return "Daily at 9:00 AM"
        if minute == "0" and hour != "*" and day == "*" and month == "*":
            if weekday == "*":
                return f"Daily at {hour}:00"
            elif weekday == "1-5":
                return f"Weekdays at {hour}:00"
            elif weekday == "0,6":
                return f"Weekends at {hour}:00"
        
        return cron

    def _format_timestamp(self, ts: Optional[int]) -> str:
        """Format Unix timestamp to readable string."""
        if not ts:
            return "Never"
        return time.strftime('%Y-%m-%d %H:%M', time.localtime(ts))

    async def list_scheduled_prompts(
        self,
        __user__: dict = None,
        __event_emitter__: callable = None,
    ) -> str:
        """
        List all scheduled prompts for the current user.
        Shows name, schedule, status, and next run time.
        
        :return: List of scheduled prompts with their details
        """
        try:
            from open_webui.models.scheduled_prompts import ScheduledPrompts
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Loading scheduled prompts...", "done": False}
                })
            
            user_id = __user__.get("id") if __user__ else None
            
            if not user_id:
                return "❌ User ID not available."
            
            prompts = ScheduledPrompts.get_scheduled_prompts_by_user_id(user_id)
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": f"Found {len(prompts)} scheduled prompts", "done": True}
                })
            
            if not prompts:
                return "📅 You don't have any scheduled prompts yet.\n\nUse `create_scheduled_prompt()` to create one."
            
            result = "## 📅 Your Scheduled Prompts\n\n"
            result += "| Name | Schedule | Status | Next Run | ID |\n"
            result += "|------|----------|--------|----------|----|\n"
            
            for p in prompts:
                status_icon = "✅" if p.enabled else "⏸️"
                status = "Enabled" if p.enabled else "Paused"
                schedule = self._format_cron_description(p.cron_expression)
                next_run = self._format_timestamp(p.next_run_at) if p.enabled else "—"
                name = p.name[:25] + "..." if len(p.name) > 25 else p.name
                result += f"| {name} | {schedule} | {status_icon} {status} | {next_run} | `{p.id[:8]}` |\n"
            
            return result
            
        except Exception as e:
            log.error(f"[PROMPT SCHEDULER] Error listing prompts: {e}")
            return f"❌ Error listing scheduled prompts: {str(e)}"

    async def get_scheduled_prompt(
        self,
        prompt_id: str,
        __user__: dict = None,
        __event_emitter__: callable = None,
    ) -> str:
        """
        Get detailed information about a specific scheduled prompt.
        
        :param prompt_id: The ID of the scheduled prompt (can be partial, minimum 8 characters)
        :return: Detailed information about the scheduled prompt
        """
        try:
            from open_webui.models.scheduled_prompts import ScheduledPrompts
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Loading prompt details...", "done": False}
                })
            
            user_id = __user__.get("id") if __user__ else None
            
            if not user_id:
                return "❌ User ID not available."
            
            # Support partial ID matching
            prompts = ScheduledPrompts.get_scheduled_prompts_by_user_id(user_id)
            prompt = None
            for p in prompts:
                if p.id.startswith(prompt_id) or p.id == prompt_id:
                    prompt = p
                    break
            
            if not prompt:
                return f"❌ Scheduled prompt not found: {prompt_id}"
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Prompt details loaded", "done": True}
                })
            
            status_icon = "✅" if prompt.enabled else "⏸️"
            last_status_icon = "✅" if prompt.last_status == "success" else "❌" if prompt.last_status == "error" else "—"
            
            result = f"## 📅 {prompt.name}\n\n"
            result += f"**ID:** `{prompt.id}`\n\n"
            result += "### Schedule\n"
            result += f"- **Cron:** `{prompt.cron_expression}` ({self._format_cron_description(prompt.cron_expression)})\n"
            result += f"- **Timezone:** {prompt.timezone}\n"
            result += f"- **Status:** {status_icon} {'Enabled' if prompt.enabled else 'Paused'}\n"
            result += f"- **Next Run:** {self._format_timestamp(prompt.next_run_at)}\n\n"
            
            result += "### Prompt Configuration\n"
            result += f"- **Model:** `{prompt.model_id}`\n"
            result += f"- **Create New Chat:** {'Yes' if prompt.create_new_chat else 'No (append to existing)'}\n"
            if prompt.system_prompt:
                result += f"- **System Prompt:** {prompt.system_prompt[:100]}{'...' if len(prompt.system_prompt) > 100 else ''}\n"
            result += f"- **User Prompt:** {prompt.prompt[:200]}{'...' if len(prompt.prompt) > 200 else ''}\n\n"
            
            result += "### Execution History\n"
            result += f"- **Run Count:** {prompt.run_count}\n"
            result += f"- **Last Run:** {self._format_timestamp(prompt.last_run_at)}\n"
            result += f"- **Last Status:** {last_status_icon} {prompt.last_status or 'Never run'}\n"
            if prompt.last_error:
                result += f"- **Last Error:** {prompt.last_error}\n"
            if prompt.chat_id:
                result += f"- **Chat ID:** `{prompt.chat_id}`\n"
            
            return result
            
        except Exception as e:
            log.error(f"[PROMPT SCHEDULER] Error getting prompt: {e}")
            return f"❌ Error getting scheduled prompt: {str(e)}"

    async def create_scheduled_prompt(
        self,
        name: str,
        cron_expression: str,
        prompt: str,
        model_id: str = "",
        system_prompt: str = "",
        timezone: str = "UTC",
        create_new_chat: bool = True,
        run_once: bool = False,
        tool_ids: list = None,
        use_current_session: bool = True,
        __user__: dict = None,
        __event_emitter__: callable = None,
        __metadata__: dict = None,
        __model__: dict = None,
    ) -> str:
        """
        Create a new scheduled prompt that runs automatically on a schedule.
        
        CRITICAL PROMPT CONSTRUCTION RULES:
        
        1. The 'prompt' parameter should be an AFFIRMATIVE NOTIFICATION, not a request:
           - BAD: "Check your todo list" or "Remind me about X" (sounds like a request)
           - GOOD: "This is your scheduled reminder. Your todo list has these remaining items: Step 3"
           - GOOD: "Reminder: You wanted to be notified about [topic]. Here's the context: [details]"
        
        2. The 'system_prompt' parameter MUST define the assistant's role clearly:
           - ALWAYS set system_prompt to something like:
             "You are delivering a scheduled reminder to the user. Present the reminder content clearly and helpfully. 
              If tools are available (like notes_manager), use them to fetch current data and provide an updated status.
              Do NOT ask the user to schedule anything - this IS the reminder being delivered."
        
        3. Example of a well-constructed reminder:
           - name: "Daily Todo Reminder"
           - prompt: "This is your scheduled reminder about your todo list. Items you wanted to track: Step 3 (pending)"
           - system_prompt: "You are delivering a scheduled reminder. Present the information clearly. Use available tools to fetch current data if relevant. Do not ask about scheduling - this is the reminder delivery."
        
        :param name: A descriptive name for this scheduled prompt (e.g., "Todo List Reminder")
        :param cron_expression: Cron expression for the schedule (e.g., "0 9 * * 1-5" for weekdays at 9am)
        :param prompt: An AFFIRMATIVE notification message - what the reminder says, NOT a request
        :param model_id: Optional. The model ID to use. Leave empty/omit to automatically use the current chat's model (recommended).
        :param system_prompt: IMPORTANT: Set this to define the assistant's role as a reminder delivery agent, NOT a scheduling assistant
        :param timezone: Timezone for the schedule (default: UTC)
        :param create_new_chat: If True, creates a new chat each time. If False, appends to existing chat.
        :param run_once: If True, the prompt runs once then automatically disables (one-off). Default: False (recurring).
        :param tool_ids: List of tool IDs to enable. If not specified and use_current_session is True, uses the tools enabled in the current chat.
        :param use_current_session: If True, automatically use the model and tools from the current chat session. Default: True.
        :return: Success message with the new prompt ID, or error message
        """
        try:
            if not self.valves.ALLOW_CREATE:
                return "❌ Creating scheduled prompts is disabled. Enable ALLOW_CREATE in the tool settings."
            
            from open_webui.models.scheduled_prompts import ScheduledPrompts, ScheduledPromptForm
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Creating scheduled prompt...", "done": False}
                })
            
            user_id = __user__.get("id") if __user__ else None
            
            if not user_id:
                return "❌ User ID not available."
            
            # Validate inputs
            if not name or len(name.strip()) == 0:
                return "❌ Name cannot be empty."
            if len(name) > 100:
                return "❌ Name too long. Maximum 100 characters."
            if not cron_expression or len(cron_expression.strip()) == 0:
                return "❌ Cron expression cannot be empty."
            
            # Determine model_id: explicit > current session (from metadata) > valve default
            # NOTE: __model__ is the model executing the tool (often a smaller model for tool calls)
            # __metadata__['model'] is the actual chat model the user is talking to
            effective_model_id = model_id.strip() if model_id else ""
            log.info(f"[PROMPT SCHEDULER] model_id param: '{model_id}', use_current_session: {use_current_session}")
            
            if not effective_model_id and use_current_session and __metadata__:
                # Get the actual chat model from metadata (not __model__ which is the tool execution model)
                chat_model = __metadata__.get("model")
                log.info(f"[PROMPT SCHEDULER] __metadata__['model']: {chat_model}")
                if chat_model:
                    # chat_model can be a string (model ID) or a dict with 'id' key
                    if isinstance(chat_model, dict):
                        effective_model_id = chat_model.get("id", "")
                    else:
                        effective_model_id = str(chat_model)
                    if effective_model_id:
                        log.info(f"[PROMPT SCHEDULER] Using chat session model from metadata: {effective_model_id}")
            
            if not effective_model_id:
                if self.valves.DEFAULT_MODEL_ID:
                    effective_model_id = self.valves.DEFAULT_MODEL_ID
                    log.info(f"[PROMPT SCHEDULER] Using valve default model: {effective_model_id}")
                else:
                    return "❌ Model ID cannot be empty. Set DEFAULT_MODEL_ID in tool settings or specify a model."
            
            if not prompt or len(prompt.strip()) == 0:
                return "❌ Prompt cannot be empty."
            
            # Validate cron expression
            try:
                from croniter import croniter
                import pytz
                from datetime import datetime
                
                tz = pytz.timezone(timezone)
                now = datetime.now(tz)
                cron = croniter(cron_expression, now)
                next_run = cron.get_next(datetime)
                next_run_ts = int(next_run.timestamp())
            except Exception as e:
                return f"❌ Invalid cron expression or timezone: {str(e)}\n\nExamples:\n- `0 9 * * *` - Daily at 9:00 AM\n- `0 9 * * 1-5` - Weekdays at 9:00 AM\n- `*/30 * * * *` - Every 30 minutes"
            
            # Check rate limit
            count = ScheduledPrompts.count_scheduled_prompts_by_user_id(user_id)
            if count >= self.valves.MAX_SCHEDULES_PER_USER:
                return f"❌ You have reached the maximum number of scheduled prompts ({self.valves.MAX_SCHEDULES_PER_USER})."
            
            # Determine tool_ids: explicit > current session > none
            effective_tool_ids = tool_ids
            log.info(f"[PROMPT SCHEDULER] tool_ids param: {tool_ids}, use_current_session: {use_current_session}")
            log.info(f"[PROMPT SCHEDULER] __metadata__: {type(__metadata__)}, __model__: {type(__model__)}")
            if __metadata__:
                log.info(f"[PROMPT SCHEDULER] __metadata__ keys: {list(__metadata__.keys()) if isinstance(__metadata__, dict) else 'not a dict'}")
                log.info(f"[PROMPT SCHEDULER] __metadata__.tool_ids: {__metadata__.get('tool_ids', 'NOT FOUND')}")
            if __model__:
                log.info(f"[PROMPT SCHEDULER] __model__.id: {__model__.get('id', 'NOT FOUND') if isinstance(__model__, dict) else 'not a dict'}")
            
            if not effective_tool_ids and use_current_session and __metadata__:
                # Get tools from current chat session (keep all tools including prompt_scheduler for chained scheduling)
                session_tools = __metadata__.get("tool_ids", [])
                if session_tools:
                    effective_tool_ids = list(session_tools)
                    log.info(f"[PROMPT SCHEDULER] Capturing current session tools: {effective_tool_ids}")
                else:
                    log.info(f"[PROMPT SCHEDULER] No tools in current session metadata")
            
            # Create the scheduled prompt
            log.info(f"[PROMPT SCHEDULER] Creating with model_id={effective_model_id}, tool_ids={effective_tool_ids}")
            form_data = ScheduledPromptForm(
                name=name.strip(),
                cron_expression=cron_expression.strip(),
                timezone=timezone,
                enabled=True,
                model_id=effective_model_id,
                system_prompt=system_prompt.strip() if system_prompt else None,
                prompt=prompt.strip(),
                create_new_chat=create_new_chat,
                run_once=run_once,
                tool_ids=effective_tool_ids if effective_tool_ids else None,
            )
            log.info(f"[PROMPT SCHEDULER] Form data: model_id={form_data.model_id}, tool_ids={form_data.tool_ids}")
            
            new_prompt = ScheduledPrompts.insert_new_scheduled_prompt(user_id, form_data, next_run_ts)
            log.info(f"[PROMPT SCHEDULER] Saved prompt: model_id={new_prompt.model_id if new_prompt else 'N/A'}, tool_ids={new_prompt.tool_ids if new_prompt else 'N/A'}")
            
            if not new_prompt:
                return "❌ Failed to create scheduled prompt."
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Scheduled prompt created", "done": True}
                })
            
            log.info(f"[PROMPT SCHEDULER] Created prompt {new_prompt.id} '{name}' for user {user_id}")
            
            schedule_type = "one-off" if run_once else "recurring"
            tools_info = f"\n**Tools:** {', '.join(effective_tool_ids)}" if effective_tool_ids else ""
            
            return f"""✅ Scheduled prompt **{name}** created successfully!

**ID:** `{new_prompt.id}`
**Type:** {schedule_type.capitalize()}
**Schedule:** {self._format_cron_description(cron_expression)} ({timezone})
**Next Run:** {self._format_timestamp(next_run_ts)}
**Model:** {effective_model_id}{tools_info}

The prompt will run {"once at the scheduled time then disable" if run_once else "automatically according to the schedule"}. You can manage it in **Workspace → Scheduled Prompts** or use the tool functions."""
            
        except Exception as e:
            log.error(f"[PROMPT SCHEDULER] Error creating prompt: {e}")
            return f"❌ Error creating scheduled prompt: {str(e)}"

    async def toggle_scheduled_prompt(
        self,
        prompt_id: str,
        enabled: bool,
        __user__: dict = None,
        __event_emitter__: callable = None,
    ) -> str:
        """
        Enable or disable a scheduled prompt.
        
        :param prompt_id: The ID of the scheduled prompt (can be partial, minimum 8 characters)
        :param enabled: True to enable, False to disable/pause
        :return: Success or error message
        """
        try:
            from open_webui.models.scheduled_prompts import ScheduledPrompts, ScheduledPromptUpdateForm
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Updating prompt status...", "done": False}
                })
            
            user_id = __user__.get("id") if __user__ else None
            
            if not user_id:
                return "❌ User ID not available."
            
            # Find the prompt (support partial ID)
            prompts = ScheduledPrompts.get_scheduled_prompts_by_user_id(user_id)
            prompt = None
            for p in prompts:
                if p.id.startswith(prompt_id) or p.id == prompt_id:
                    prompt = p
                    break
            
            if not prompt:
                return f"❌ Scheduled prompt not found: {prompt_id}"
            
            # Calculate next run if enabling
            next_run_ts = None
            if enabled:
                try:
                    from croniter import croniter
                    import pytz
                    from datetime import datetime
                    
                    tz = pytz.timezone(prompt.timezone)
                    now = datetime.now(tz)
                    cron = croniter(prompt.cron_expression, now)
                    next_run = cron.get_next(datetime)
                    next_run_ts = int(next_run.timestamp())
                except:
                    pass
            
            form_data = ScheduledPromptUpdateForm(enabled=enabled)
            updated = ScheduledPrompts.update_scheduled_prompt_by_id(prompt.id, form_data, next_run_ts)
            
            if not updated:
                return "❌ Failed to update scheduled prompt."
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Prompt status updated", "done": True}
                })
            
            action = "enabled" if enabled else "paused"
            status_icon = "✅" if enabled else "⏸️"
            
            result = f"{status_icon} Scheduled prompt **{prompt.name}** has been {action}."
            if enabled and next_run_ts:
                result += f"\n\n**Next Run:** {self._format_timestamp(next_run_ts)}"
            
            return result
            
        except Exception as e:
            log.error(f"[PROMPT SCHEDULER] Error toggling prompt: {e}")
            return f"❌ Error updating scheduled prompt: {str(e)}"

    async def delete_scheduled_prompt(
        self,
        prompt_id: str,
        __user__: dict = None,
        __event_emitter__: callable = None,
    ) -> str:
        """
        Delete a scheduled prompt permanently.
        
        This function requires ALLOW_DELETE to be enabled in the tool settings.
        WARNING: This action cannot be undone!
        
        :param prompt_id: The ID of the scheduled prompt to delete (can be partial, minimum 8 characters)
        :return: Success or error message
        """
        try:
            if not self.valves.ALLOW_DELETE:
                return "❌ Deleting scheduled prompts is disabled. Enable ALLOW_DELETE in the tool settings."
            
            from open_webui.models.scheduled_prompts import ScheduledPrompts
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Deleting scheduled prompt...", "done": False}
                })
            
            user_id = __user__.get("id") if __user__ else None
            
            if not user_id:
                return "❌ User ID not available."
            
            # Find the prompt (support partial ID)
            prompts = ScheduledPrompts.get_scheduled_prompts_by_user_id(user_id)
            prompt = None
            for p in prompts:
                if p.id.startswith(prompt_id) or p.id == prompt_id:
                    prompt = p
                    break
            
            if not prompt:
                return f"❌ Scheduled prompt not found: {prompt_id}"
            
            name = prompt.name
            ScheduledPrompts.delete_scheduled_prompt_by_id(prompt.id)
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Scheduled prompt deleted", "done": True}
                })
            
            log.info(f"[PROMPT SCHEDULER] Deleted prompt {prompt.id} '{name}' by user {user_id}")
            
            return f"✅ Scheduled prompt **{name}** has been deleted."
            
        except Exception as e:
            log.error(f"[PROMPT SCHEDULER] Error deleting prompt: {e}")
            return f"❌ Error deleting scheduled prompt: {str(e)}"

    async def update_scheduled_prompt(
        self,
        prompt_id: str,
        name: str = None,
        cron_expression: str = None,
        model_id: str = None,
        prompt: str = None,
        system_prompt: str = None,
        timezone: str = None,
        create_new_chat: bool = None,
        run_once: bool = None,
        tool_ids: list = None,
        __user__: dict = None,
        __event_emitter__: callable = None,
    ) -> str:
        """
        Update an existing scheduled prompt. Only provided fields will be updated.
        
        :param prompt_id: The ID of the scheduled prompt to update (can be partial)
        :param name: New name for the prompt
        :param cron_expression: New cron expression for the schedule
        :param model_id: New model ID
        :param prompt: New user prompt
        :param system_prompt: New system prompt
        :param timezone: New timezone
        :param create_new_chat: Whether to create new chat each time
        :param run_once: If True, run once then disable. If False, recurring.
        :param tool_ids: List of tool IDs to enable (e.g., ["note_manager", "web_search"])
        :return: Success or error message
        """
        try:
            from open_webui.models.scheduled_prompts import ScheduledPrompts, ScheduledPromptUpdateForm
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Updating scheduled prompt...", "done": False}
                })
            
            user_id = __user__.get("id") if __user__ else None
            
            if not user_id:
                return "❌ User ID not available."
            
            # Find the prompt (support partial ID)
            prompts = ScheduledPrompts.get_scheduled_prompts_by_user_id(user_id)
            existing_prompt = None
            for p in prompts:
                if p.id.startswith(prompt_id) or p.id == prompt_id:
                    existing_prompt = p
                    break
            
            if not existing_prompt:
                return f"❌ Scheduled prompt not found: {prompt_id}"
            
            # Validate cron if provided
            next_run_ts = None
            if cron_expression:
                try:
                    from croniter import croniter
                    import pytz
                    from datetime import datetime
                    
                    tz_str = timezone or existing_prompt.timezone
                    tz = pytz.timezone(tz_str)
                    now = datetime.now(tz)
                    cron = croniter(cron_expression, now)
                    next_run = cron.get_next(datetime)
                    next_run_ts = int(next_run.timestamp())
                except Exception as e:
                    return f"❌ Invalid cron expression: {str(e)}"
            
            form_data = ScheduledPromptUpdateForm(
                name=name,
                cron_expression=cron_expression,
                timezone=timezone,
                model_id=model_id,
                system_prompt=system_prompt,
                prompt=prompt,
                create_new_chat=create_new_chat,
                run_once=run_once,
                tool_ids=tool_ids,
            )
            
            updated = ScheduledPrompts.update_scheduled_prompt_by_id(existing_prompt.id, form_data, next_run_ts)
            
            if not updated:
                return "❌ Failed to update scheduled prompt."
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Scheduled prompt updated", "done": True}
                })
            
            log.info(f"[PROMPT SCHEDULER] Updated prompt {existing_prompt.id} by user {user_id}")
            
            # Build update summary
            changes = []
            if name: changes.append(f"name → {name}")
            if cron_expression: changes.append(f"schedule → {self._format_cron_description(cron_expression)}")
            if model_id: changes.append(f"model → {model_id}")
            if prompt: changes.append("prompt updated")
            if system_prompt is not None: changes.append("system prompt updated")
            if timezone: changes.append(f"timezone → {timezone}")
            if create_new_chat is not None: changes.append(f"create_new_chat → {create_new_chat}")
            if run_once is not None: changes.append(f"run_once → {run_once}")
            if tool_ids is not None: changes.append(f"tools → {', '.join(tool_ids) if tool_ids else 'none'}")
            
            result = f"✅ Scheduled prompt **{existing_prompt.name}** updated.\n\n**Changes:**\n"
            for change in changes:
                result += f"- {change}\n"
            
            if next_run_ts:
                result += f"\n**Next Run:** {self._format_timestamp(next_run_ts)}"
            
            return result
            
        except Exception as e:
            log.error(f"[PROMPT SCHEDULER] Error updating prompt: {e}")
            return f"❌ Error updating scheduled prompt: {str(e)}"

    async def get_scheduler_capabilities(
        self,
        __user__: dict = None,
        __event_emitter__: callable = None,
    ) -> str:
        """
        Show available Prompt Scheduler capabilities and usage examples.
        Call this when the user asks what you can do with scheduled prompts or needs help.
        
        :return: Help text describing all available functions
        """
        create_status = "✅ Enabled" if self.valves.ALLOW_CREATE else "❌ Disabled"
        delete_status = "✅ Enabled" if self.valves.ALLOW_DELETE else "❌ Disabled"
        
        return f"""## 📅 Prompt Scheduler - Available Functions

### Viewing Schedules
| Function | Description |
|----------|-------------|
| `list_scheduled_prompts()` | List all your scheduled prompts |
| `get_scheduled_prompt(prompt_id)` | Get detailed info about a specific prompt |

### Creating Schedules
| Function | Description | Status |
|----------|-------------|--------|
| `create_scheduled_prompt(...)` | Create a new scheduled prompt | {create_status} |

### Managing Schedules
| Function | Description |
|----------|-------------|
| `toggle_scheduled_prompt(prompt_id, enabled)` | Enable or pause a scheduled prompt |
| `update_scheduled_prompt(prompt_id, ...)` | Update schedule, prompt, or settings |

### Deleting Schedules
| Function | Description | Status |
|----------|-------------|--------|
| `delete_scheduled_prompt(prompt_id)` | Permanently delete a scheduled prompt | {delete_status} |

---

### 💡 Cron Expression Examples

| Expression | Description |
|------------|-------------|
| `* * * * *` | Every minute |
| `0 * * * *` | Every hour |
| `0 9 * * *` | Daily at 9:00 AM |
| `0 9 * * 1-5` | Weekdays at 9:00 AM |
| `0 0 * * 0` | Every Sunday at midnight |
| `*/30 * * * *` | Every 30 minutes |
| `0 9,17 * * *` | At 9:00 AM and 5:00 PM |

Format: `minute hour day month weekday`

---

### 📝 Usage Examples

**Create a daily summary:**
> "Schedule a daily prompt at 9am to summarize my notes"

**List schedules:**
> "Show me my scheduled prompts"

**Pause a schedule:**
> "Pause the daily summary prompt"

**Update a schedule:**
> "Change my daily summary to run at 8am instead"

---

### ⚙️ Settings
- **ALLOW_CREATE:** {create_status}
- **ALLOW_DELETE:** {delete_status}
- **DEFAULT_MODEL_ID:** {self.valves.DEFAULT_MODEL_ID or '(not set)'}
- **MAX_SCHEDULES_PER_USER:** {self.valves.MAX_SCHEDULES_PER_USER}

*Settings can be changed in Workspace → Tools → Prompt Scheduler → Valves*

**Note:** If a scheduled prompt's model is unavailable, the scheduler will fall back to your default model or the first available model.
"""
