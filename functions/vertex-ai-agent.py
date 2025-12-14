"""
Simplified Vertex AI Agent Pipeline for Open WebUI
No encryption dependencies - uses plain text tokens or gcloud auth
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List, AsyncGenerator
import aiohttp
import subprocess
from pydantic import BaseModel, Field


class Pipe:
    """Simplified Vertex AI Agent Pipeline for Open WebUI."""

    class Valves(BaseModel):
        """Configuration valves for the pipeline."""

        PROJECT_ID: str = Field(
            default="",
            description="Google Cloud Project ID where the agent is deployed",
        )
        LOCATION: str = Field(
            default="europe-west2",
            description="Google Cloud region where the agent is deployed (e.g., us-central1, europe-west2)",
        )
        AGENT_ENGINE_ID: str = Field(
            default="",
            description="The reasoning engine ID from Vertex AI Agent Engine deployment",
        )
        AUTH_METHOD: str = Field(
            default="gcloud",
            description="Authentication method: 'gcloud', 'service_account', or 'manual_token'",
        )
        SERVICE_ACCOUNT_JSON: str = Field(
            default="",
            description="Service account JSON key content (for service_account auth method)",
        )
        SERVICE_ACCOUNT_PATH: str = Field(
            default="",
            description="Path to service account JSON file (for service_account auth method)",
        )
        USE_GCLOUD_AUTH: bool = Field(
            default=True,
            description="Use 'gcloud auth print-access-token' to get authentication token automatically",
        )
        GOOGLE_CLOUD_TOKEN: str = Field(
            default="",
            description="Google Cloud access token (plain text - leave empty to use gcloud auth automatically)",
        )
        TIMEOUT_SECONDS: int = Field(
            default=60, description="Timeout in seconds for agent requests"
        )
        ENABLE_STATUS_INDICATOR: bool = Field(
            default=True, description="Enable or disable status indicator emissions"
        )
        EMIT_INTERVAL: float = Field(
            default=2.0, description="Interval in seconds between status emissions"
        )

    def __init__(self):
        self.name = "Vertex AI Ardoq Agent"
        self.valves = self.Valves()

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger(__name__)

    async def get_access_token(self) -> Optional[str]:
        """Get Google Cloud access token using the configured authentication method."""

        if self.valves.AUTH_METHOD == "service_account":
            return await self._get_service_account_token()
        elif self.valves.AUTH_METHOD == "manual_token":
            return (
                self.valves.GOOGLE_CLOUD_TOKEN
                if self.valves.GOOGLE_CLOUD_TOKEN
                else None
            )
        else:  # gcloud (default)
            return await self._get_gcloud_token()

    async def _get_service_account_token(self) -> Optional[str]:
        """Get access token using service account credentials."""
        try:
            import json
            from google.oauth2 import service_account
            from google.auth.transport.requests import Request

            # Get service account info
            if self.valves.SERVICE_ACCOUNT_JSON:
                # Use JSON content directly
                service_account_info = json.loads(self.valves.SERVICE_ACCOUNT_JSON)
            elif self.valves.SERVICE_ACCOUNT_PATH:
                # Load from file path
                with open(self.valves.SERVICE_ACCOUNT_PATH, "r") as f:
                    service_account_info = json.load(f)
            else:
                self.log.error("No service account credentials provided")
                return None

            # Create credentials
            credentials = service_account.Credentials.from_service_account_info(
                service_account_info,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )

            # Refresh token
            request = Request()
            credentials.refresh(request)

            self.log.info("Successfully authenticated with service account")
            return credentials.token

        except Exception as e:
            self.log.error(f"Service account authentication failed: {e}")
            return None

    async def _get_gcloud_token(self) -> Optional[str]:
        """Get access token using gcloud CLI."""
        try:
            process = await asyncio.create_subprocess_exec(
                "gcloud",
                "auth",
                "print-access-token",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                self.log.info("Successfully authenticated with gcloud")
                return stdout.decode().strip()
            else:
                self.log.error(f"gcloud auth failed: {stderr.decode()}")
                return None
        except Exception as e:
            self.log.error(f"Failed to get gcloud token: {e}")
            return None

    def get_agent_url(self) -> str:
        """Construct the Vertex AI Agent Engine URL."""
        return (
            f"https://{self.valves.LOCATION}-aiplatform.googleapis.com/v1/"
            f"projects/{self.valves.PROJECT_ID}/locations/{self.valves.LOCATION}/"
            f"reasoningEngines/{self.valves.AGENT_ENGINE_ID}:streamQuery?alt=sse"
        )

    async def parse_streaming_response(self, response: aiohttp.ClientResponse) -> str:
        """Parse the streaming response from Vertex AI Agent Engine."""
        final_response = None
        function_calls = []
        response_parts = []
        line_count = 0

        self.log.info("Starting to parse streaming response...")

        try:
            async for line_bytes in response.content:
                line_str = line_bytes.decode("utf-8").strip()
                line_count += 1

                if line_str:
                    self.log.info(
                        f"Line {line_count}: {line_str[:200]}..."
                    )  # Log first 200 chars

                    try:
                        data = json.loads(line_str)
                        self.log.info(f"Successfully parsed JSON for line {line_count}")

                        # Track function calls and responses
                        if "content" in data and "parts" in data["content"]:
                            self.log.info(
                                f"Found content with {len(data['content']['parts'])} parts"
                            )

                            for i, part in enumerate(data["content"]["parts"]):
                                self.log.info(
                                    f"Processing part {i}: {list(part.keys())}"
                                )

                                if "function_call" in part:
                                    function_calls.append(part["function_call"])
                                    func_name = part["function_call"].get(
                                        "name", "unknown"
                                    )
                                    self.log.info(
                                        f"🔧 Function call detected: {func_name}"
                                    )

                                elif "text" in part:
                                    text_content = part["text"]
                                    self.log.info(
                                        f"📝 Text content found: {text_content[:100]}..."
                                    )

                                    # Check if this is a model response (final answer)
                                    role = data.get("content", {}).get("role")
                                    self.log.info(f"Content role: {role}")

                                    if role == "model":
                                        final_response = text_content
                                        self.log.info(
                                            f"✅ Final response set: {final_response}"
                                        )

                                    # Always collect text parts
                                    response_parts.append(text_content)

                                elif "function_response" in part:
                                    func_resp = part["function_response"]
                                    self.log.info(
                                        f"🔄 Function response: {func_resp.get('name', 'unknown')}"
                                    )
                        else:
                            self.log.info(
                                f"No content/parts found in data: {list(data.keys())}"
                            )

                    except json.JSONDecodeError as e:
                        self.log.warning(f"JSON decode error on line {line_count}: {e}")
                        self.log.warning(f"Raw line content: {line_str}")
                        continue
                else:
                    self.log.debug(f"Empty line {line_count}")

        except Exception as e:
            self.log.error(f"Error parsing streaming response: {e}")
            import traceback

            self.log.error(f"Traceback: {traceback.format_exc()}")

        # Log summary
        self.log.info(f"Parsing complete. Lines processed: {line_count}")
        self.log.info(f"Function calls found: {len(function_calls)}")
        self.log.info(f"Response parts found: {len(response_parts)}")
        self.log.info(f"Final response: {final_response}")

        # Return the best available response
        if final_response:
            self.log.info(f"Returning final response: {final_response}")
            return final_response
        elif response_parts:
            combined = " ".join(response_parts)
            self.log.info(f"Returning combined response: {combined}")
            return combined
        else:
            self.log.warning("No response content found - returning error message")
            return "No response received from agent"

    async def query_agent(self, message: str, user_id: str = "openwebui-user") -> str:
        """Send a query to the Vertex AI agent."""
        self.log.info(f"Querying agent: {message}")

        # Get access token
        access_token = await self.get_access_token()
        if not access_token:
            return "❌ Failed to get access token. Please check your authentication."

        # Prepare request
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        url = self.get_agent_url()
        payload = {
            "class_method": "stream_query",
            "input": {"user_id": user_id, "message": message},
        }

        self.log.debug(f"URL: {url}")
        self.log.debug(f"Payload: {json.dumps(payload, indent=2)}")

        # Make request
        session = None
        try:
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.valves.TIMEOUT_SECONDS)
            )

            async with session.post(url, json=payload, headers=headers) as response:
                self.log.info(f"Response status: {response.status}")

                if response.status == 200:
                    result = await self.parse_streaming_response(response)
                    return result
                else:
                    error_text = await response.text()
                    self.log.error(f"Error response: {error_text}")
                    return f"Agent Error: {response.status} - {error_text}"

        except Exception as e:
            self.log.error(f"Request failed: {e}")
            return f"Request failed: {e}"

        finally:
            if session:
                await session.close()

    async def pipe(
        self,
        body: Dict[str, Any],
        __user__: Dict[str, Any],
        __event_emitter__: Optional[Any] = None,
    ) -> AsyncGenerator[str, None]:
        """Main pipeline function called by Open WebUI."""

        # Extract the user's message
        messages = body.get("messages", [])
        if not messages:
            yield "Error: No messages provided"
            return

        # Get the last user message
        user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        if not user_message:
            yield "Error: No user message found"
            return

        # Emit status if enabled
        if self.valves.ENABLE_STATUS_INDICATOR and __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Querying Vertex AI Insight Agent...",
                        "done": False,
                    },
                }
            )

        # Query the agent
        try:
            user_id = __user__.get("id", "openwebui-user")
            response = await self.query_agent(user_message, user_id)

            self.log.info(f"Agent response received: {response}")

            # Emit completion status
            if self.valves.ENABLE_STATUS_INDICATOR and __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Complete", "done": True},
                    }
                )

            # Add the agent response to messages
            messages.append({"role": "assistant", "content": response})

            self.log.info(f"Final messages array: {messages}")

            # Try multiple return formats for Open WebUI compatibility
            result = {"messages": messages}
            self.log.info(f"Returning result: {result}")

            # Stream the response back to Open WebUI
            if len(messages) > 0:
                last_message = messages[-1]
                assistant_response = last_message.get("content", "")
                self.log.info(f"Streaming response: {assistant_response}")

                # Yield the response character by character for streaming effect
                for char in assistant_response:
                    yield char
                    await asyncio.sleep(0.01)  # Small delay for streaming effect
            else:
                self.log.warning("No assistant message to stream")
                yield "No response generated"

        except Exception as e:
            self.log.error(f"Pipeline error: {e}")

            # Emit error status
            if self.valves.ENABLE_STATUS_INDICATOR and __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Error: {str(e)}", "done": True},
                    }
                )

            # Stream error message
            error_msg = f"Pipeline failed: {str(e)}"
            for char in error_msg:
                yield char
                await asyncio.sleep(0.01)
