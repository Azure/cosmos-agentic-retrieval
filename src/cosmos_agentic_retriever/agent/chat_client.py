"""Send the conversation and available tool descriptions to the language model.

ChatClient.complete returns the model's text and any tool calls it requests.
The agent loop executes those calls and decides whether to ask the model again.
If the request fails after the SDK's retries, or the reply cannot be read,
return an error so the loop can stop.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import structlog
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageFunctionToolCall
from openai.types.completion_usage import CompletionUsage

from cosmos_agentic_retriever.agent.loop import LlmTurn, TokenUsage
from cosmos_agentic_retriever.agent.tool_contracts import InvalidToolCall, Tool, ToolCall

_logger = structlog.get_logger(__name__)


def to_openai_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    """Describe each tool to the model without sending its Python handler."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


def _parse_tool_call(call_id: str, name: str, raw: str | None) -> ToolCall | InvalidToolCall:
    """Decode a tool call's JSON arguments; empty arguments mean no arguments.

    Arguments that are not valid JSON, or not a JSON object, give an
    InvalidToolCall carrying the raw text and the reason.
    """
    raw_arguments = raw or "{}"
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return InvalidToolCall(
            call_id, name, raw_arguments, f"arguments for '{name}' are not valid JSON."
        )
    if not isinstance(arguments, dict):
        return InvalidToolCall(
            call_id,
            name,
            raw_arguments,
            f"arguments for '{name}' must be a JSON object, got "
            f"{type(arguments).__name__}.",
        )
    return ToolCall(call_id, name, arguments)


def _read_usage(usage: CompletionUsage | None) -> TokenUsage:
    """Copy the reply's token counts; endpoints may omit any of them."""
    if usage is None:
        return TokenUsage()
    prompt_details = usage.prompt_tokens_details
    completion_details = usage.completion_tokens_details
    return TokenUsage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cached_prompt_tokens=prompt_details.cached_tokens if prompt_details else None,
        reasoning_tokens=(
            completion_details.reasoning_tokens if completion_details else None
        ),
    )


@dataclass
class ChatClient:
    """Send messages to the configured OpenAI-compatible chat endpoint.

    base_url is the API address; model is the model or deployment name. api_key is
    a key, or a function returning a fresh Entra token. params adds request options
    such as max_completion_tokens and temperature. The client supplies model,
    messages, and tools even if params contains those keys.
    """

    base_url: str
    model: str
    api_key: str | Callable[[], str] = "unused"
    params: dict[str, Any] = field(default_factory=dict)
    timeout: float = 60.0

    def __post_init__(self) -> None:
        """Create the SDK client with the configured endpoint and HTTP timeout."""
        self._client = OpenAI(
            base_url=self.base_url, api_key=self.api_key, timeout=self.timeout
        )

    def complete(self, messages: list[dict[str, Any]], tools: list[Tool]) -> LlmTurn:
        """Request the model's next reply without executing any tools.

        Return its text and requested tool calls to the loop. Set LlmTurn.error
        if the request fails or the response cannot be read.
        """
        request: dict[str, Any] = {
            **self.params,
            "model": self.model,
            "messages": messages,
            "tools": to_openai_tools(tools),
        }
        try:
            response = self._client.chat.completions.create(**request)  # type: ignore[call-overload]
            if not response.choices:
                raise ValueError("Model returned no choices.")
            choice = response.choices[0].message
            message: dict[str, Any] = {
                "role": "assistant",
                "content": choice.content or "",
            }
            function_calls = []
            for call in choice.tool_calls or []:
                if not isinstance(call, ChatCompletionMessageFunctionToolCall):
                    raise TypeError(f"Unsupported tool call type: {call.type!r}.")
                function_calls.append(call)
            tool_calls: list[ToolCall | InvalidToolCall] = []
            if function_calls:
                message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in function_calls
                ]
                tool_calls = [
                    _parse_tool_call(call.id, call.function.name, call.function.arguments)
                    for call in function_calls
                ]
            return LlmTurn(
                message=message,
                content=choice.content,
                tool_calls=tool_calls,
                usage=_read_usage(response.usage),
            )
        except Exception as exc:
            _logger.exception("model_call_failed", model=self.model)
            return LlmTurn(
                message={"role": "assistant", "content": ""},
                error=f"{type(exc).__name__}: {exc}",
            )
