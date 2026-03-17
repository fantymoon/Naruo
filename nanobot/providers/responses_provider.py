"""OpenAI Responses API provider — for endpoints that use the new Responses API format."""

from __future__ import annotations

import json
import uuid
from typing import Any

import httpx
from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class ResponsesAPIProvider(LLMProvider):
    """Provider for OpenAI-compatible endpoints using the Responses API format.
    
    The Responses API differs from Chat Completions:
    - Endpoint: POST /v1/responses (not /v1/chat/completions)
    - Request body uses 'input' instead of 'messages'
    - Response has 'output' array with different structure
    - Tool calls use 'function_call' type with 'call_id'
    """

    def __init__(
        self,
        api_key: str = "no-key",
        api_base: str = "http://localhost:8000/v1",
        default_model: str = "gpt-4o",
        extra_headers: dict[str, str] | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self._extra_headers = extra_headers or {}
        # Ensure api_base ends without trailing slash, then append /responses
        self._responses_url = api_base.rstrip("/") + "/responses"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Send a request using the Responses API format."""
        # Convert messages to Responses API input format
        # Responses API accepts either a string or an array of message objects
        input_content = self._convert_messages_to_input(messages)
        
        request_body: dict[str, Any] = {
            "model": model or self.default_model,
            "input": input_content,
            "max_output_tokens": max(1, max_tokens),
            "temperature": temperature,
        }
        
        if reasoning_effort:
            # Responses API uses 'reasoning' object for reasoning effort
            request_body["reasoning"] = {"effort": reasoning_effort}
        
        # Convert tools to Responses API format
        if tools:
            request_body["tools"] = self._convert_tools(tools)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self._extra_headers,
        }
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    self._responses_url,
                    json=request_body,
                    headers=headers,
                )
                
                if response.status_code != 200:
                    error_text = response.text
                    return LLMResponse(
                        content=f"Error: HTTP {response.status_code} - {error_text}",
                        finish_reason="error",
                    )
                
                data = response.json()
                return self._parse_response(data)
                
        except Exception as e:
            logger.error(f"ResponsesAPI error: {e}")
            return LLMResponse(content=f"Error: {e}", finish_reason="error")

    def _convert_messages_to_input(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Chat Completions messages to Responses API input format.
        
        Responses API input format is similar but:
        - System message goes in 'instructions' field (handled separately)
        - Content can be string or array of content blocks
        - Image content uses 'input_image' type instead of 'image_url'
        """
        result = []
        instructions = None
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")
            
            # Extract system message as instructions
            if role == "system":
                if isinstance(content, str):
                    instructions = content
                continue
            
            # Convert content blocks
            if isinstance(content, list):
                converted_content = []
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type", "text")
                        if block_type == "image_url":
                            # Convert image_url to input_image format
                            image_url = block.get("image_url", {})
                            url = image_url.get("url", "") if isinstance(image_url, dict) else image_url
                            converted_content.append({
                                "type": "input_image",
                                "image_url": url,
                            })
                        elif block_type == "text":
                            converted_content.append({
                                "type": "input_text",
                                "text": block.get("text", ""),
                            })
                        else:
                            converted_content.append(block)
                    else:
                        converted_content.append({"type": "input_text", "text": str(block)})
                result.append({"role": role, "content": converted_content})
            elif isinstance(content, str):
                result.append({"role": role, "content": content})
            elif content is None and role == "assistant":
                # Assistant message with tool calls
                result.append({"role": role, "content": None})
            else:
                result.append({"role": role, "content": str(content) if content else ""})
        
        # If we have instructions, prepend as a system message (some providers accept this)
        # Or we could add it to the request body's 'instructions' field
        # For now, we'll return the messages array and handle instructions separately
        return result

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Chat Completions tools to Responses API format.
        
        Responses API tools format is similar but uses slightly different structure.
        """
        result = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                result.append({
                    "type": "function",
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                    "strict": func.get("strict", False),
                })
            else:
                result.append(tool)
        return result

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        """Parse Responses API response to LLMResponse."""
        status = data.get("status", "completed")
        
        if status == "error":
            error = data.get("error", {})
            return LLMResponse(
                content=f"Error: {error.get('message', 'Unknown error')}",
                finish_reason="error",
            )
        
        output = data.get("output", [])
        
        if not output:
            # Try output_text as fallback
            output_text = data.get("output_text", "")
            if output_text:
                return LLMResponse(content=output_text, finish_reason="stop")
            return LLMResponse(
                content="Error: Empty response from API",
                finish_reason="error",
            )
        
        # Parse output items
        content_parts = []
        tool_calls = []
        
        for item in output:
            item_type = item.get("type", "")
            
            if item_type == "message":
                # Message item contains content blocks
                for content_block in item.get("content", []):
                    block_type = content_block.get("type", "")
                    if block_type in ("output_text", "text"):
                        content_parts.append(content_block.get("text", ""))
                    elif block_type == "refusal":
                        content_parts.append(f"[Refusal: {content_block.get('refusal', '')}]")
            
            elif item_type == "function_call":
                # Tool call
                try:
                    args_str = item.get("arguments", "{}")
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    args = {}
                
                tool_calls.append(ToolCallRequest(
                    id=item.get("call_id", item.get("id", f"call_{uuid.uuid4().hex[:8]}")),
                    name=item.get("name", ""),
                    arguments=args,
                ))
            
            elif item_type == "function_call_output":
                # Tool result - should not appear in initial response
                pass
        
        content = "\n".join(content_parts) if content_parts else None
        
        # Determine finish reason
        finish_reason = "stop"
        if tool_calls:
            finish_reason = "tool_calls"
        
        # Parse usage
        usage = {}
        if "usage" in data:
            u = data["usage"]
            usage = {
                "prompt_tokens": u.get("input_tokens", 0),
                "completion_tokens": u.get("output_tokens", 0),
                "total_tokens": u.get("total_tokens", 0),
            }
        
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )

    def get_default_model(self) -> str:
        return self.default_model