"""Check model requests and replies without calling a model endpoint."""

from unittest.mock import AsyncMock, Mock

import pytest
from openai.types.chat import ChatCompletion
from openai.types.completion_usage import CompletionUsage

from cosmos_agentic_retriever.agent import chat_client
from cosmos_agentic_retriever.agent.loop import TokenUsage
from cosmos_agentic_retriever.agent.tool_contracts import InvalidToolCall, Tool, ToolCall

pytestmark = pytest.mark.anyio


@pytest.fixture
def client(monkeypatch):
    sdk = Mock()
    sdk.chat.completions.create = AsyncMock()
    sdk.close = AsyncMock()
    monkeypatch.setattr(chat_client, "AsyncOpenAI", Mock(return_value=sdk))
    return chat_client.ChatClient(
        base_url="https://model.example.com/v1", model="test-model"
    ), sdk.chat.completions.create


def _reply(*, tool_calls=None):
    return ChatCompletion.model_validate(
        {
            "id": "reply-1",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                    "message": {
                        "role": "assistant",
                        "content": None if tool_calls else "The answer.",
                        "tool_calls": tool_calls,
                    },
                }
            ],
        }
    )


@pytest.mark.parametrize("token_parameter", ["max_tokens", "max_completion_tokens"])
async def test_complete_sends_model_options_and_tool_descriptions(client, token_parameter):
    _, create = client
    chat = chat_client.ChatClient(
        "https://model.example.com/v1",
        "test-model",
        params={token_parameter: 256, "temperature": 0.2, "model": "ignored"},
    )
    create.return_value = _reply()
    messages = [{"role": "user", "content": "Find recycling documents."}]
    handler = Mock()
    tool = Tool(
        name="full_text_search",
        description="Search the configured containers.",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        handler=handler,
    )

    turn = await chat.complete(messages, [tool])

    create.assert_called_once_with(
        **{
            token_parameter: 256,
            "temperature": 0.2,
            "model": "test-model",
            "messages": messages,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
            ],
        }
    )
    handler.assert_not_called()
    assert turn.content == "The answer."
    assert turn.tool_calls == [] and turn.error is None


async def test_complete_returns_token_usage(client):
    chat, create = client
    response = _reply()
    create.return_value = response
    assert (await chat.complete([], [])).usage == TokenUsage()

    response.usage = CompletionUsage(prompt_tokens=120, completion_tokens=8, total_tokens=128)
    assert (await chat.complete([], [])).usage == TokenUsage(
        prompt_tokens=120, completion_tokens=8
    )

    response.usage = CompletionUsage(
        prompt_tokens=120,
        completion_tokens=8,
        total_tokens=128,
        prompt_tokens_details={"cached_tokens": 64},
        completion_tokens_details={"reasoning_tokens": 5},
    )
    turn = await chat.complete([], [])

    assert turn.usage == TokenUsage(
        prompt_tokens=120, completion_tokens=8, cached_prompt_tokens=64, reasoning_tokens=5
    )


@pytest.mark.parametrize(
    "arguments, expected",
    [
        (' {"query": "recycling"} ', ToolCall("call-1", "full_text_search", {"query": "recycling"})),
        ("", ToolCall("call-1", "full_text_search", {})),
        (
            "broken",
            InvalidToolCall(
                "call-1",
                "full_text_search",
                "broken",
                "arguments for 'full_text_search' are not valid JSON.",
            ),
        ),
        (
            "[]",
            InvalidToolCall(
                "call-1",
                "full_text_search",
                "[]",
                "arguments for 'full_text_search' must be a JSON object, got list.",
            ),
        ),
    ],
)
async def test_complete_returns_tool_calls_for_the_loop(client, arguments, expected):
    chat, create = client
    call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "full_text_search", "arguments": arguments},
    }
    create.return_value = _reply(tool_calls=[call])

    turn = await chat.complete([], [])

    assert turn.error is None
    assert turn.tool_calls == [expected]
    assert turn.message["tool_calls"] == [call]


async def test_complete_returns_request_errors_to_the_loop(client):
    chat, create = client
    create.side_effect = RuntimeError("endpoint unavailable")

    turn = await chat.complete([], [])

    assert turn.error == "RuntimeError: endpoint unavailable"
    assert turn.tool_calls == []


async def test_complete_returns_error_for_missing_reply(client):
    chat, create = client
    response = _reply()
    response.choices = []
    create.return_value = response

    turn = await chat.complete([], [])

    assert turn.error == "ValueError: Model returned no choices."
    assert turn.tool_calls == []


async def test_complete_reports_unsupported_tool_calls(client):
    chat, create = client
    create.return_value = _reply(
        tool_calls=[
            {
                "id": "call-1",
                "type": "custom",
                "custom": {"name": "unsupported", "input": "query"},
            }
        ]
    )

    turn = await chat.complete([], [])

    assert turn.error == "TypeError: Unsupported tool call type: 'custom'."
    assert turn.tool_calls == []


async def test_close_closes_the_sdk_client(client):
    chat, _ = client

    await chat.close()

    chat_client.AsyncOpenAI.return_value.close.assert_awaited_once()
