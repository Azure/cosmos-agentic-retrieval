"""Keep the model API's wire format and failure handling outside the agent loop.

The loop works with LlmTurn, ToolCall and TokenUsage rather than OpenAI response
objects. ChatClient makes that translation while preserving the original assistant
message required by later API requests. It never executes tools or owns a question's
conversation; the loop supplies that state and decides what to do next.
The SDK retries an individual model request; final failures pass through
ModelErrorMapper so provider diagnostics do not leak into the service response.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageFunctionToolCall
from openai.types.completion_usage import CompletionUsage

from cosmos_agentic_retriever.agent.loop import LlmTurn, TokenUsage
from cosmos_agentic_retriever.agent.model_errors import ModelErrorMapper
from cosmos_agentic_retriever.agent.tool_contracts import InvalidToolCall, Tool, ToolCall

_logger = structlog.get_logger(__name__)


class ChatClient:
    """Send messages to the configured OpenAI-compatible chat endpoint.

    One service-lifetime SDK client reuses HTTP connections across questions;
    each complete() receives its own transcript, so question history stays with
    the caller rather than on this shared client.

    base_url is the API address; model is the model or deployment name. api_key is
    a key, or an async function returning a fresh Entra token. params adds request
    options such as max_completion_tokens and temperature. The client supplies
    model, messages, and tools even if params contains those keys. Call close()
    when done, to release the HTTP connections.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | Callable[[], Awaitable[str]] = "unused",
        params: dict[str, Any] | None = None,
        timeout: float = 60.0,
        max_retries: int = 6,
    ) -> None:
        self.model = model
        self._params = dict(params or {})
        self._client = AsyncOpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries
        )

    async def complete(self, messages: list[dict[str, Any]], tools: list[Tool]) -> LlmTurn:
        """Request one agent turn without executing tools or restarting the question.

        SDK retries repeat only this request with the same transcript. If they
        are exhausted, return a safe error for the loop and HTTP handler.
        """
        request: dict[str, Any] = {**self._params, "model": self.model, "messages": messages}
        # The API rejects an empty tool list, so a tool-free request omits it.
        if tools:
            request["tools"] = self._to_openai_tools(tools)
        try:
            response = await self._client.chat.completions.create(**request)  # type: ignore[call-overload]
            return self._parse_response(response)
        except Exception as exc:
            error = ModelErrorMapper.from_exception(exc)
            _logger.warning(
                "model_call_failed", error_kind=error.kind,
                error_type=type(exc).__name__, status_code=error.status_code,
            )
            return LlmTurn(
                message={"role": "assistant", "content": ""},
                error=error,
            )

    async def close(self) -> None:
        """Close the SDK client's HTTP connections."""
        await self._client.close()

    @classmethod
    def _parse_response(cls, response: ChatCompletion) -> LlmTurn:
        """Preserve API message structure while exposing typed calls to the loop.

        The next model request needs the assistant's original tool-call IDs and
        argument strings, even when argument parsing fails and the loop reports it.
        """
        if not response.choices:
            raise ValueError("Model returned no choices.")
        first = response.choices[0]
        if first.finish_reason == "content_filter":
            return LlmTurn(
                message={"role": "assistant", "content": ""},
                error=ModelErrorMapper.content_filter(), usage=cls._read_usage(response.usage),
            )
        choice = first.message
        message: dict[str, Any] = {"role": "assistant", "content": choice.content or ""}
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
                cls._parse_tool_call(call.id, call.function.name, call.function.arguments)
                for call in function_calls
            ]
        return LlmTurn(
            message=message, content=choice.content, tool_calls=tool_calls,
            usage=cls._read_usage(response.usage),
        )

    @staticmethod
    def _to_openai_tools(tools: list[Tool]) -> list[dict[str, Any]]:
        """Send model-facing schemas, not Python handlers or other execution state."""
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

    @staticmethod
    def _parse_tool_call(call_id: str, name: str, raw: str | None) -> ToolCall | InvalidToolCall:
        """Let the loop report malformed arguments without executing a bad call.

        Keeping the raw text and reason lets the model correct the call instead
        of silently treating malformed JSON as missing arguments.
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
                call_id, name, raw_arguments,
                f"arguments for '{name}' must be a JSON object, got {type(arguments).__name__}.",
            )
        return ToolCall(call_id, name, arguments)

    @staticmethod
    def _read_usage(usage: CompletionUsage | None) -> TokenUsage:
        """Keep absent counts unknown rather than inventing zero-cost model calls."""
        if usage is None:
            return TokenUsage()
        prompt_details = usage.prompt_tokens_details
        completion_details = usage.completion_tokens_details
        return TokenUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cached_prompt_tokens=prompt_details.cached_tokens if prompt_details else None,
            reasoning_tokens=completion_details.reasoning_tokens if completion_details else None,
        )
