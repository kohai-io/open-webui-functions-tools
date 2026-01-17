"""
title: AutoTool Filter
author: open-webui
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 0.3.0
required_open_webui_version: 0.5.0
description: Intelligently selects custom tools and built-in capabilities (Web Search, Image Generation) based on user queries using LLM analysis
"""

from pydantic import BaseModel, Field
from typing import Callable, Awaitable, Any, Optional, Literal
import json
import re
import logging

# Updated imports
from open_webui.models.users import Users
from open_webui.models.tools import Tools
from open_webui.models.models import Models
from open_webui.utils.chat import generate_chat_completion  # Use the lightweight option
from open_webui.utils.misc import get_last_user_message


logger = logging.getLogger(__name__)


class Filter:
    class Valves(BaseModel):
        template: str = Field(
            default="""Available Tools and Capabilities: {{TOOLS}}
If a tool or capability doesn't match the query, return an empty list []. Otherwise, return a list of matching IDs in the format ["id"]. Select multiple if applicable. Only return the list, no other text. Review the entire chat history to ensure the selection matches the context. If unsure, default to an empty list []. Use conservatively."""
        )
        status: bool = Field(
            default=False,
            description="Show status messages for tool selection (user-facing)",
        )
        debug: bool = Field(
            default=False,
            description="Show detailed debug information in status messages (includes raw LLM output, tool counts, etc.)",
        )
        enable_capabilities: bool = Field(
            default=True,
            description="Allow the filter to trigger built-in capabilities (Web Search, Image Generation, etc.)",
        )
        pass

    def __init__(self):
        self.valves = self.Valves()
        pass

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __request__: Any,  # New requirement in version 0.5
        __user__: Optional[dict] = None,
        __model__: Optional[dict] = None,
    ) -> dict:
        messages = body.get("messages", [])
        user_message = get_last_user_message(messages)

        # Ensure we have a valid user context; if not, skip tool selection
        if not __user__ or "id" not in __user__:
            logger.debug(
                "AutoTool Filter: no valid user context; skipping tool selection"
            )
            if self.valves.debug:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "[DEBUG] No valid user context; skipping tool selection",
                            "done": True,
                        },
                    }
                )
            return body

        if self.valves.status:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Finding the right tools...",
                        "done": False,
                    },
                }
            )

        # Gather custom tools
        all_tools = [
            {"id": tool.id, "description": tool.meta.description}
            for tool in Tools.get_tools()
        ]

        model_info = (__model__ or {}).get("info", {})
        model_meta = model_info.get("meta", {})
        available_tool_ids = model_meta.get("toolIds", [])
        available_tools = [
            tool for tool in all_tools if tool["id"] in available_tool_ids
        ]

        # Gather enabled capabilities
        available_options = list(available_tools)  # Start with custom tools
        capability_map = {}

        if self.valves.enable_capabilities:
            # Check which capabilities are enabled for this model
            # Capabilities might be in meta or at the model_info level
            capabilities = model_meta.get("capabilities", {})
            if not capabilities:
                capabilities = model_info.get("capabilities", {})

            if self.valves.debug:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"[DEBUG] Capabilities detected: {json.dumps(capabilities, ensure_ascii=False)[:200]}",
                            "done": False,
                        },
                    }
                )

            # Map capability IDs to their descriptions and body parameters
            capability_definitions = [
                {
                    "id": "web_search",
                    "description": "Search the web for current information, facts, news, or real-time data",
                    "enabled_key": "web_search",
                    "body_param": "web_search",
                },
                {
                    "id": "image_generation",
                    "description": "Generate, create, or draw images based on text descriptions",
                    "enabled_key": "image_generation",
                    "body_param": "image_generation",
                },
            ]

            for cap in capability_definitions:
                # Check if capability is enabled (handle multiple storage patterns)
                # Try: capabilities dict, direct meta key, camelCase variants
                is_enabled = (
                    capabilities.get(cap["enabled_key"], False)
                    or capabilities.get(cap["enabled_key"].replace("_", ""), False)
                    or model_meta.get(cap["enabled_key"], False)
                    or model_meta.get(cap["enabled_key"].replace("_", ""), False)
                )
                if is_enabled:
                    available_options.append(
                        {"id": cap["id"], "description": cap["description"]}
                    )
                    capability_map[cap["id"]] = cap["body_param"]

        if self.valves.debug:
            tool_names = [t["id"] for t in available_tools]
            cap_names = list(capability_map.keys())
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"[DEBUG] Available: {len(available_tools)} tools ({', '.join(tool_names) if tool_names else 'none'}), {len(cap_names)} capabilities ({', '.join(cap_names) if cap_names else 'none'})",
                        "done": False,
                    },
                }
            )

        # If there are no tools or capabilities available, skip
        if not available_options:
            logger.debug(
                "AutoTool Filter: no tools or capabilities available for this model; skipping"
            )
            if self.valves.debug:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"[DEBUG] No tools or capabilities available (found {len(all_tools)} total tools, {len(capability_map)} capabilities, 0 enabled)",
                            "done": True,
                        },
                    }
                )
            return body

        options_str = json.dumps(available_options, ensure_ascii=False)
        system_prompt = self.valves.template.replace("{{TOOLS}}", options_str)
        prompt = (
            "History:\n"
            + "\n".join(
                [
                    f"{message['role'].upper()}: \"\"\"{message['content']}\"\"\""
                    for message in messages[::-1][:4]
                ]
            )
            + f"\nQuery: {user_message}"
        )
        payload = {
            "model": body.get("model"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }

        try:
            user = Users.get_user_by_id(__user__["id"])
            # Updated to use the direct successor function
            response = await generate_chat_completion(
                request=__request__, form_data=payload, user=user
            )

            # Defensive extraction of completion content
            content = None
            try:
                choices = (
                    response.get("choices", []) if isinstance(response, dict) else []
                )
                if choices and "message" in choices[0]:
                    content = choices[0]["message"].get("content")
            except Exception:
                logger.exception(
                    "AutoTool Filter: error extracting content from response"
                )

            if content is not None:
                logger.debug("AutoTool Filter raw content: %s", content)
                if self.valves.debug:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"[DEBUG] Raw LLM response: {content[:200]}...",
                                "done": False,
                            },
                        }
                    )

                # Attempt to extract the first JSON array from the content
                result = []
                arrays = re.findall(r"\[[^\]]*\]", content, flags=re.DOTALL)
                for candidate in arrays:
                    try:
                        parsed = json.loads(candidate.replace("'", '"'))
                        if isinstance(parsed, list):
                            result = parsed
                            break
                    except json.JSONDecodeError:
                        continue

                # Fallback: try to parse the whole content if no array was found
                if not result:
                    try:
                        parsed = json.loads(content.replace("'", '"'))
                        if isinstance(parsed, list):
                            result = parsed
                    except json.JSONDecodeError:
                        logger.debug(
                            "AutoTool Filter: unable to parse content as JSON list"
                        )
                        if self.valves.debug:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {
                                        "description": f"[DEBUG] Unable to parse LLM output as JSON: {content[:100]}",
                                        "done": False,
                                    },
                                }
                            )

                # Normalize and filter to valid tool/capability IDs
                if isinstance(result, list):
                    original_count = len(result)
                    valid_tool_ids = []
                    enabled_capabilities = []

                    for item_id in result:
                        if not isinstance(item_id, str):
                            continue
                        # Check if it's a custom tool
                        if item_id in available_tool_ids:
                            valid_tool_ids.append(item_id)
                        # Check if it's a capability
                        elif item_id in capability_map:
                            enabled_capabilities.append(item_id)

                    if self.valves.debug and original_count > 0:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": f"[DEBUG] Parsed {original_count} IDs → {len(valid_tool_ids)} tools, {len(enabled_capabilities)} capabilities",
                                    "done": False,
                                },
                            }
                        )

                    # Set custom tools
                    if valid_tool_ids:
                        body["tool_ids"] = valid_tool_ids

                    # Set capabilities in the features dict
                    if enabled_capabilities:
                        # Get existing features or create new dict
                        features = body.get("features", {})
                        for cap_id in enabled_capabilities:
                            body_param = capability_map[cap_id]
                            features[body_param] = True
                            if self.valves.debug:
                                logger.debug(
                                    f"AutoTool Filter: Set features['{body_param}'] = True for capability '{cap_id}'"
                                )
                        body["features"] = features

                        if self.valves.debug:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {
                                        "description": f"[DEBUG] Set features: {json.dumps(features, ensure_ascii=False)}",
                                        "done": False,
                                    },
                                }
                            )

                    result = valid_tool_ids + enabled_capabilities

                if isinstance(result, list) and len(result) > 0:
                    if self.valves.debug:
                        # Show what we actually set in the body
                        relevant_keys = {}
                        if "tool_ids" in body:
                            relevant_keys["tool_ids"] = body["tool_ids"]
                        if "features" in body:
                            relevant_keys["features"] = body["features"]
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": f"[DEBUG] Body after modifications: {json.dumps(relevant_keys, ensure_ascii=False)}",
                                    "done": False,
                                },
                            }
                        )
                    if self.valves.status:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": f"Found matching tools: {', '.join(result)}",
                                    "done": True,
                                },
                            }
                        )
                else:
                    if self.valves.status:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": "No matching tools found.",
                                    "done": True,
                                },
                            }
                        )
        except Exception as e:
            logger.exception("AutoTool Filter: error processing request")
            if self.valves.status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Error processing request while selecting tools.",
                            "done": True,
                        },
                    }
                )
            pass
        return body
