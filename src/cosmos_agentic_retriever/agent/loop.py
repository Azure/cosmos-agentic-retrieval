"""Run model requests and tool calls until the model answers or the turn limit is reached.

run_agent_search calls the supplied complete function for the model's next
reply. The service passes ChatClient.complete from chat_client.py. That client
makes the API request and returns the reply and requested tool calls; it does
not execute them.

This loop executes the tool calls, adds their results to the conversation, and
asks the model again. It stops when a reply has no tool calls, a model call fails,
or max_turns model calls have been made.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import Any

import structlog

_logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Tool:
    """A capability the model can call: parsed arguments in, observation text out.

    ``parameters`` is a JSON-Schema object describing the arguments. ``handler``
    receives the parsed arguments and returns the text shown back to the model;
    it should return an error string rather than raise for a bad call, though the
    loop also catches exceptions.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class ToolCall:
    """One tool request the model made this turn."""

    id: str
    name: str
    arguments: dict[str, Any]


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
class LlmTurn:
    """One model reply: its text, any tool calls, and the message to append.

    ``message`` is the assistant message the loop appends to the running
    transcript verbatim, so threading stays correct without the loop parsing the
    model's wire format. ``error`` is set when the model call itself failed.
    """

    message: dict[str, Any]
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    error: str | None = None
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


def _run_tool_call(tools_by_name: dict[str, Tool], call: ToolCall) -> str:
    """Run one tool call, turning an unknown tool or a raised error into text."""
    tool = tools_by_name.get(call.name)
    if tool is None:
        return f"Error: unknown tool {call.name!r}."
    try:
        return tool.handler(call.arguments)
    except Exception as exc:  # a tool bug should not end the episode
        _logger.exception("tool_call_failed", tool=call.name)
        return f"Error: {type(exc).__name__}: {exc}"


def run_agent_search(
    question: str,
    *,
    complete: Callable[[list[dict[str, Any]], list[Tool]], LlmTurn],
    tools: list[Tool],
    system_prompt: str,
    max_turns: int,
) -> AgentResult:
    """Run the search loop for one question and return its result.

    Args:
        question: the caller's request; the model decides what to search for.
        complete: gets the model's next reply as an LlmTurn. The service passes
            ChatClient.complete; tests can supply a scripted callable.
        tools: the capabilities offered to the model (the `full_text_search`
            tool). Names must be unique.
        system_prompt: the instruction that opens the transcript.
        max_turns: the hard cap on model calls; the loop always stops by here.

    Returns:
        An `AgentResult` with the final answer, why the loop ended, the number of
        turns taken, and the full transcript.
    """
    if max_turns < 1:
        raise ValueError("max_turns must be >= 1")
    tools_by_name = {tool.name: tool for tool in tools}
    if len(tools_by_name) != len(tools):
        raise ValueError("tool names must be unique")

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
    for turn in range(1, max_turns + 1):
        reply = complete(messages, tools)
        messages.append(reply.message)
        usage += reply.usage
        if reply.error is not None:
            terminal_reason = "error"
            break
        if not reply.tool_calls:
            terminal_reason = "stop"
            break
        for call in reply.tool_calls:
            tool_calls[call.name] = tool_calls.get(call.name, 0) + 1
            observation = _run_tool_call(tools_by_name, call)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": observation}
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
    )
