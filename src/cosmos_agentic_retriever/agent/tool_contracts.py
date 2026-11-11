"""Tools the agent can call: what a tool is, how it is called, and what it returns.

- `Tool`: an action the model can request. Its handler takes a batch of argument
  dicts (one per call) and returns one result per call, in order.
- `ToolCall`: one request from the model: the tool's name, its arguments, and an
  ID that matches the result to the request.
- `InvalidToolCall`: a request whose arguments could not be parsed into an
  object; it keeps the raw argument text and the parsing error.
- `ToolExecutionResult`: what running a tool produced: the text shown to the
  model, any error, the time taken, and numeric measurements.
- `ToolCallResponse`: an execution result with the ID of the call it answers.

`execute_tool_call` runs one `ToolCall` against tools keyed by name. Unknown tool
names and handler exceptions become errors in the response, never exceptions.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any


@dataclass
class ToolCall:
    """A request identifying a tool and its parsed arguments."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class InvalidToolCall:
    """A request whose raw arguments could not be parsed into an object."""

    id: str
    name: str
    raw_arguments: str
    error: str


@dataclass
class ToolExecutionResult:
    """Execution output, errors, timing, and metrics without a request ID."""

    content: str
    error: str | None = None
    latency_s: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(kw_only=True)
class ToolCallResponse(ToolExecutionResult):
    """Execution output associated with a specific tool request."""

    call_id: str
    tool_implementation: str | None = None


@dataclass
class Tool:
    """A capability the model can invoke during a turn.

    ``parameters`` is a JSON Schema object (OpenAI conventions) describing the
    arguments the model may pass. ``handler`` receives a list of argument dicts
    and returns one string or `ToolExecutionResult` per dict, in order; a single
    call is a batch of one. Only the content is shown to the model.
    ``implementation`` names which implementation of the tool ran, for logs and
    comparisons; it defaults to the tool's name.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[list[dict[str, Any]]], Sequence[str | ToolExecutionResult]]
    implementation: str | None = None

    def __call__(self, **arguments: Any) -> ToolExecutionResult:
        """Run one call and return its output, error, and timing."""
        started_at = time.perf_counter()
        try:
            content = self.handler([arguments])[0]
        except Exception as exc:
            return ToolExecutionResult(
                content="",
                error=f"{type(exc).__name__}: {exc}",
                latency_s=time.perf_counter() - started_at,
            )
        elapsed = time.perf_counter() - started_at
        if isinstance(content, ToolExecutionResult):
            return replace(content, latency_s=elapsed, metrics=dict(content.metrics))
        return ToolExecutionResult(content=content, error=None, latency_s=elapsed)


def execute_tool_call(
    tools_by_name: dict[str, Tool], tool_call: ToolCall
) -> ToolCallResponse:
    """Run the named tool, or return an error response for an unknown name."""
    tool = tools_by_name.get(tool_call.name)
    if tool is None:
        return ToolCallResponse(
            call_id=tool_call.id,
            content="",
            error=f"unknown tool: {tool_call.name}",
            latency_s=0.0,
        )
    result = tool(**tool_call.arguments)
    return ToolCallResponse(
        call_id=tool_call.id,
        tool_implementation=tool.implementation or tool.name,
        content=result.content,
        error=result.error,
        latency_s=result.latency_s,
        metrics=result.metrics,
    )
