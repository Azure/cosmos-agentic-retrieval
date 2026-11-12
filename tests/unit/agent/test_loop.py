"""Loop behavior: termination, tool dispatch, and error handling."""

from __future__ import annotations

import json

import pytest

from cosmos_agentic_retriever.agent.loop import (
    LlmTurn,
    TokenUsage,
    run_agent_search,
)
from cosmos_agentic_retriever.agent.tool_contracts import InvalidToolCall, Tool, ToolCall

pytestmark = pytest.mark.anyio


def test_token_usage_sum_is_unknown_when_either_count_is_unknown():
    known = TokenUsage(
        prompt_tokens=10, completion_tokens=2, cached_prompt_tokens=4, reasoning_tokens=1
    )
    partly_known = TokenUsage(prompt_tokens=5, completion_tokens=3)

    assert known + known == TokenUsage(
        prompt_tokens=20, completion_tokens=4, cached_prompt_tokens=8, reasoning_tokens=2
    )
    assert known + partly_known == TokenUsage(prompt_tokens=15, completion_tokens=5)


def _answer(text: str) -> LlmTurn:
    return LlmTurn(message={"role": "assistant", "content": text}, content=text)


def _call(call_id: str, name: str, arguments: dict) -> LlmTurn:
    return LlmTurn(
        message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        },
        content="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
    )


class _ScriptedModel:
    """A stand-in `complete` that returns pre-scripted turns and records calls."""

    def __init__(self, turns: list[LlmTurn]) -> None:
        self._turns = list(turns)
        self.messages_seen: list[list[dict]] = []

    async def __call__(self, messages, tools):
        self.messages_seen.append([dict(message) for message in messages])
        return self._turns.pop(0)


def _echo_tool(record: list[dict]) -> Tool:
    async def handler(calls: list[dict]) -> list[str]:
        record.extend(calls)
        return [f"searched {arguments.get('query')!r}" for arguments in calls]

    return Tool(
        name="full_text_search",
        description="search",
        parameters={"type": "object", "properties": {}},
        handler=handler,
    )


async def test_answers_without_searching_stops():
    model = _ScriptedModel([_answer("done")])
    result = await run_agent_search(
        "q",
        complete=model,
        tools=[_echo_tool([])],
        system_prompt="sys",
        max_turns=5,
    )
    assert result.terminal_reason == "stop"
    assert result.answer == "done"
    assert result.turns == 1


async def test_searches_then_answers():
    calls: list[dict] = []
    model = _ScriptedModel(
        [_call("c1", "full_text_search", {"query": "cats"}), _answer("final")]
    )
    result = await run_agent_search(
        "q",
        complete=model,
        tools=[_echo_tool(calls)],
        system_prompt="sys",
        max_turns=5,
    )
    assert calls == [{"query": "cats"}]
    assert result.terminal_reason == "stop"
    assert result.answer == "final"
    assert result.turns == 2
    roles = [message["role"] for message in result.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    [response] = result.tool_responses
    assert response.call_id == "c1"
    assert response.content == "searched 'cats'"
    assert response.error is None
    assert response.tool_implementation == "full_text_search"


async def test_stops_at_max_turns_when_model_keeps_calling():
    model = _ScriptedModel([_call(f"c{i}", "full_text_search", {"query": "x"}) for i in range(5)])
    result = await run_agent_search(
        "q",
        complete=model,
        tools=[_echo_tool([])],
        system_prompt="sys",
        max_turns=3,
    )
    assert result.terminal_reason == "max_turns"
    assert result.turns == 3


async def test_unknown_tool_yields_error_observation_and_continues():
    model = _ScriptedModel(
        [_call("c1", "does_not_exist", {"query": "x"}), _answer("ok")]
    )
    result = await run_agent_search(
        "q",
        complete=model,
        tools=[_echo_tool([])],
        system_prompt="sys",
        max_turns=5,
    )
    tool_messages = [m for m in result.messages if m["role"] == "tool"]
    assert "unknown tool" in tool_messages[0]["content"]
    assert result.tool_responses[0].error == "unknown tool: does_not_exist"
    assert result.terminal_reason == "stop"
    assert result.answer == "ok"


async def test_invalid_arguments_are_reported_without_running_the_tool():
    calls: list[dict] = []
    invalid = InvalidToolCall("c1", "full_text_search", "broken", "not valid JSON.")
    turn = LlmTurn(
        message={"role": "assistant", "content": "", "tool_calls": []},
        content="",
        tool_calls=[invalid],
    )
    model = _ScriptedModel([turn, _answer("ok")])
    result = await run_agent_search(
        "q",
        complete=model,
        tools=[_echo_tool(calls)],
        system_prompt="sys",
        max_turns=5,
    )
    tool_messages = [m for m in result.messages if m["role"] == "tool"]
    assert calls == []
    assert tool_messages == [
        {"role": "tool", "tool_call_id": "c1", "content": "Error: not valid JSON."}
    ]
    assert result.tool_responses[0].error == "not valid JSON."
    assert result.tool_calls == {"full_text_search": 1}
    assert result.answer == "ok"


async def test_tool_exception_becomes_error_observation():
    async def boom(_calls: list[dict]) -> list[str]:
        raise RuntimeError("kaboom")

    tool = Tool(
        name="full_text_search",
        description="search",
        parameters={"type": "object", "properties": {}},
        handler=boom,
    )
    model = _ScriptedModel(
        [_call("c1", "full_text_search", {"query": "x"}), _answer("recovered")]
    )
    result = await run_agent_search(
        "q", complete=model, tools=[tool], system_prompt="sys", max_turns=5
    )
    tool_messages = [m for m in result.messages if m["role"] == "tool"]
    assert "RuntimeError: kaboom" in tool_messages[0]["content"]
    assert result.terminal_reason == "stop"


async def test_model_error_ends_episode():
    model = _ScriptedModel(
        [LlmTurn(message={"role": "assistant", "content": ""}, error="boom")]
    )
    result = await run_agent_search(
        "q",
        complete=model,
        tools=[_echo_tool([])],
        system_prompt="sys",
        max_turns=5,
    )
    assert result.terminal_reason == "error"
    assert result.turns == 1


async def test_rejects_duplicate_tool_names():
    tool = _echo_tool([])
    with pytest.raises(ValueError, match="unique"):
        await run_agent_search(
            "q",
            complete=_ScriptedModel([_answer("x")]),
            tools=[tool, tool],
            system_prompt="sys",
            max_turns=5,
        )


async def test_rejects_nonpositive_max_turns():
    with pytest.raises(ValueError, match="max_turns"):
        await run_agent_search(
            "q",
            complete=_ScriptedModel([_answer("x")]),
            tools=[_echo_tool([])],
            system_prompt="sys",
            max_turns=0,
        )
