"""The tool contract: single calls, batches, errors, and call responses."""

from __future__ import annotations

import pytest

from cosmos_agentic_retriever.agent.tool_contracts import (
    Tool,
    ToolCall,
    ToolExecutionResult,
    execute_tool_call,
)

pytestmark = pytest.mark.anyio


def _tool(handler, implementation: str | None = None) -> Tool:
    return Tool(
        name="echo",
        description="Echo the text.",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        handler=handler,
        implementation=implementation,
    )


async def _echo(calls):
    return [call["text"] for call in calls]


async def test_call_runs_handler_as_batch_of_one():
    seen: list[list[dict]] = []

    async def handler(calls):
        seen.append(calls)
        return [call["text"] for call in calls]

    result = await _tool(handler)(text="hi")

    assert seen == [[{"text": "hi"}]]
    assert result.content == "hi"
    assert result.error is None
    assert result.latency_s >= 0.0


async def test_handler_exception_becomes_error():
    async def handler(_calls):
        raise RuntimeError("kaboom")

    result = await _tool(handler)(text="hi")

    assert result.content == ""
    assert result.error == "RuntimeError: kaboom"


async def test_handler_result_keeps_metrics_and_gets_latency():
    async def handler(calls):
        return [ToolExecutionResult(content="ok", metrics={"documents": 3.0})]

    result = await _tool(handler)(text="hi")

    assert result.content == "ok"
    assert result.metrics == {"documents": 3.0}
    assert result.latency_s >= 0.0


async def test_execute_tool_call_links_response_to_call():
    response = await execute_tool_call(
        {"echo": _tool(_echo)}, ToolCall("c1", "echo", {"text": "hi"})
    )

    assert response.call_id == "c1"
    assert response.content == "hi"
    assert response.error is None
    assert response.tool_implementation == "echo"


async def test_execute_tool_call_reports_implementation():
    tool = _tool(_echo, implementation="echo-v2")

    response = await execute_tool_call({"echo": tool}, ToolCall("c1", "echo", {"text": "x"}))

    assert response.tool_implementation == "echo-v2"


async def test_execute_tool_call_unknown_tool_is_error_response():
    response = await execute_tool_call({}, ToolCall("c1", "missing", {}))

    assert response.call_id == "c1"
    assert response.content == ""
    assert response.error == "unknown tool: missing"
