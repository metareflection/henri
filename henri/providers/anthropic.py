"""Direct Anthropic API provider for Claude models."""

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from anthropic import Anthropic

if TYPE_CHECKING:
    from henri.tools.base import Tool

from henri.config import DEFAULT_ANTHROPIC_MODEL
from henri.messages import Message, ToolCall
from henri.providers.base import Provider, StreamEvent, Usage


class AnthropicProvider(Provider):
    """Direct Anthropic API provider for Claude models."""

    name = "anthropic"

    def __init__(
        self,
        model_id: str = DEFAULT_ANTHROPIC_MODEL,
        **kwargs,
    ):
        self.model_id = model_id
        self.client = Anthropic()

    def _message_to_anthropic(self, msg: Message) -> dict:
        """Convert a Message to Anthropic's format."""
        content = []

        if msg.content:
            content.append({"type": "text", "text": msg.content})

        for tc in msg.tool_calls:
            content.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.args,
            })

        for tr in msg.tool_results:
            content.append({
                "type": "tool_result",
                "tool_use_id": tr.tool_call_id,
                "content": tr.content,
                "is_error": tr.is_error,
            })

        role = "user" if msg.role == "tool" else msg.role
        return {"role": role, "content": content}

    def _tools_to_anthropic(self, tools: list["Tool"]) -> list[dict]:
        """Convert tools to Anthropic's format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    async def stream(
        self,
        messages: list[Message],
        tools: list["Tool"],
        system: str = "",
    ) -> AsyncIterator[StreamEvent]:
        """Stream a response from Claude via the Anthropic API."""
        anthropic_messages = [self._message_to_anthropic(m) for m in messages]

        request = {
            "model": self.model_id,
            "max_tokens": 8192,
            "messages": anthropic_messages,
        }

        if system:
            request["system"] = system

        if tools:
            request["tools"] = self._tools_to_anthropic(tools)

        tool_calls = []
        current_tool_id = None
        current_tool_name = None
        current_tool_input = ""

        with self.client.messages.stream(**request) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        current_tool_id = event.content_block.id
                        current_tool_name = event.content_block.name
                        current_tool_input = ""
                        yield StreamEvent(tool_use_started=True)

                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        yield StreamEvent(text=event.delta.text)
                    elif event.delta.type == "input_json_delta":
                        current_tool_input += event.delta.partial_json

                elif event.type == "content_block_stop":
                    if current_tool_id and current_tool_name:
                        tool_calls.append(ToolCall(
                            id=current_tool_id,
                            name=current_tool_name,
                            args=json.loads(current_tool_input) if current_tool_input else {},
                        ))
                        current_tool_id = None
                        current_tool_name = None

                elif event.type == "message_stop":
                    pass

            # Get final message for stop reason and usage
            final_message = stream.get_final_message()
            stop_reason = final_message.stop_reason if final_message else "end_turn"
            usage = None
            if final_message and final_message.usage:
                usage = Usage(
                    input_tokens=final_message.usage.input_tokens,
                    output_tokens=final_message.usage.output_tokens,
                )

        yield StreamEvent(
            tool_calls=tool_calls if tool_calls else None,
            stop_reason=stop_reason,
            usage=usage,
        )
