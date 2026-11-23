"""Own one question's reasoning loop without owning model or tool transports.

The loop coordinates transcript updates, tool execution, usage and termination.
The injected complete callable handles the model API; Tool handlers decide where
tools run. This separation keeps HTTP and MCP lifecycle details out of reasoning
and lets tests supply scripted model turns without network access.

Bad tool arguments and tool failures become observations the model can correct.
A final model failure stops the loop and carries its safe error to the endpoint;
the turn limit bounds model turns, not elapsed time or the number of tool calls.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, fields
from typing import Any, Literal

import structlog

from cosmos_agentic_retriever.agent.tool_contracts import (
    InvalidToolCall,
    Tool,
    ToolCall,
    ToolCallResponse,
    execute_tool_call,
)

_logger = structlog.get_logger(__name__)

FINAL_TURN_INSTRUCTION = (
    "You have reached the search limit. Answer now from the documents you have "
    "found, in the format asked for. If they do not answer the question, say so "
    "and give what they do show."
)


@dataclass(frozen=True)
class TokenUsage:
    """Token counts the endpoint reported; None for counts it did not report."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    # Subsets of the two counts above.
    cached_prompt_tokens: int | None = None
    reasoning_tokens: int | None = None

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Sum each count; a sum is unknown if either side is unknown."""
        sums: dict[str, int | None] = {}
        for f in fields(self):
            mine, theirs = getattr(self, f.name), getattr(other, f.name)
            sums[f.name] = None if mine is None or theirs is None else mine + theirs
        return TokenUsage(**sums)


@dataclass(frozen=True)
class ModelError:
    """A model failure safe to return to a caller, without provider response text."""

    kind: Literal[
        "rate_limit", "content_filter", "timeout", "unavailable", "authentication",
        "invalid_request", "invalid_response", "unknown",
    ]
    message: str
    status_code: int
    retry_after_seconds: int | None = None


@dataclass(frozen=True)
class LlmTurn:
    """One model reply: its text, any tool calls, and the message to append.

    ``message`` is the assistant message the loop appends to the running
    transcript verbatim, so threading stays correct without the loop parsing the
    model's wire format. ``error`` is set when the model call itself failed.
    """

    message: dict[str, Any]
    content: str | None = None
    tool_calls: list[ToolCall | InvalidToolCall] = field(default_factory=list)
    error: ModelError | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class AgentResult:
    """The outcome of one episode: the final answer and how the loop ended.

    ``terminal_reason`` is ``"stop"`` (the model answered), ``"max_turns"`` (the
    turn cap was hit), or ``"error"`` (a model call failed). ``messages`` is the
    full transcript, including tool observations, for logging and debugging.
    """

    answer: str
    terminal_reason: str
    turns: int
    messages: list[dict[str, Any]]
    # Summed over every model call in the episode.
    usage: TokenUsage = field(default_factory=TokenUsage)
    # Calls the model made per tool name, including names of unknown tools.
    tool_calls: dict[str, int] = field(default_factory=dict)
    # One response per tool call, in the order the calls ran.
    tool_responses: list[ToolCallResponse] = field(default_factory=list)
    error: ModelError | None = None


async def run_agent_search(
    question: str,
    *,
    complete: Callable[[list[dict[str, Any]], list[Tool]], Awaitable[LlmTurn]],
    tools: list[Tool],
    system_prompt: str,
    max_turns: int,
) -> AgentResult:
    """Run the search loop for one question and return its result.

    Args:
        question: the caller's request; the model decides what to search for.
        complete: gets the model's next reply as an LlmTurn. The service passes
            ChatClient.complete; tests can supply a scripted coroutine.
        tools: the capabilities offered to the model. Names must be unique.
        system_prompt: the instruction that opens the transcript.
        max_turns: the hard cap on model calls; the loop always stops by here.
            The last call offers no tools and asks for an answer from what was found,
            so a capped run still answers; it ends with ``"max_turns"``.

    Returns:
        An `AgentResult` with the final answer, why the loop ended, the number of
        turns taken, and the full transcript.
    """
    if max_turns < 1:
        raise ValueError("max_turns must be >= 1")
    tools_by_name = {tool.name: tool for tool in tools}
    if len(tools_by_name) != len(tools):
        raise ValueError("tool names must be unique")

    def observation(response: ToolCallResponse) -> str:
        return response.content if response.error is None else f"Error: {response.error}"

    async def respond(call: ToolCall | InvalidToolCall) -> ToolCallResponse:
        if isinstance(call, InvalidToolCall):
            return ToolCallResponse(call_id=call.id, content="", error=call.error)
        return await execute_tool_call(tools_by_name, call)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    terminal_reason = "max_turns"
    turn = 0
    usage = TokenUsage(
        prompt_tokens=0, completion_tokens=0, cached_prompt_tokens=0, reasoning_tokens=0
    )
    tool_calls: dict[str, int] = {}
    tool_responses: list[ToolCallResponse] = []
    error: ModelError | None = None
    for turn in range(1, max_turns + 1):
        final = turn == max_turns and max_turns > 1
        if final:
            messages.append({"role": "user", "content": FINAL_TURN_INSTRUCTION})
        reply = await complete(messages, [] if final else tools)
        messages.append(reply.message)
        usage += reply.usage
        if reply.error is not None:
            terminal_reason = "error"
            error = reply.error
            break
        if final:
            break
        if not reply.tool_calls:
            terminal_reason = "stop"
            break
        for call in reply.tool_calls:
            tool_calls[call.name] = tool_calls.get(call.name, 0) + 1
            response = await respond(call)
            if response.error is not None:
                _logger.warning("tool_call_failed", tool=call.name, error=response.error)
            tool_responses.append(response)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": observation(response)}
            )

    answer = ""
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            answer = str(message["content"])
            break
    return AgentResult(
        answer=answer,
        terminal_reason=terminal_reason,
        turns=turn,
        messages=messages,
        usage=usage,
        tool_calls=tool_calls,
        tool_responses=tool_responses,
        error=error,
    )
