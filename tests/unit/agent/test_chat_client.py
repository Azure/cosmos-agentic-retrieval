"""Check model requests and replies without calling a model endpoint."""

from unittest.mock import AsyncMock, Mock
from datetime import datetime, timezone
from email.utils import format_datetime

import anyio
import httpx
import pytest
from azure.core.exceptions import ClientAuthenticationError
from openai import APIConnectionError, APIStatusError, APITimeoutError
from openai.types.chat import ChatCompletion
from openai.types.completion_usage import CompletionUsage

from cosmos_agentic_retriever.agent import chat_client, model_errors
from cosmos_agentic_retriever.agent.loop import ModelError, TokenUsage
from cosmos_agentic_retriever.agent.model_errors import ModelErrorMapper
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

    assert turn.error == ModelError(
        kind="unknown", message="The search model request failed. Please try again later.",
        status_code=502,
    )
    assert turn.tool_calls == []


async def test_complete_returns_error_for_missing_reply(client):
    chat, create = client
    response = _reply()
    response.choices = []
    create.return_value = response

    turn = await chat.complete([], [])

    assert turn.error.kind == "invalid_response"
    assert "Model returned no choices" not in turn.error.message
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

    assert turn.error.kind == "invalid_response"
    assert "custom" not in turn.error.message
    assert turn.tool_calls == []


async def test_close_closes_the_sdk_client(client):
    chat, _ = client

    await chat.close()

    chat_client.AsyncOpenAI.return_value.close.assert_awaited_once()


@pytest.mark.parametrize("max_retries", [0, 6, 12])
async def test_max_retries_is_passed_to_the_sdk(client, max_retries):
    chat = chat_client.ChatClient(
        "https://model.example.com/v1", "test-model", max_retries=max_retries
    )

    assert chat_client.AsyncOpenAI.call_args.kwargs["max_retries"] == max_retries
    await chat.close()


async def test_default_retry_budget_is_six(client):
    assert chat_client.AsyncOpenAI.call_args.kwargs["max_retries"] == 6


@pytest.mark.parametrize("status, body, kind, expected_status", [
    (429, {"code": "rate_limit_exceeded"}, "rate_limit", 429),
    (400, {"code": "content_filter"}, "content_filter", 400),
    (400, {"error": {"code": "content_filter"}}, "content_filter", 400),
    (400, {"innererror": {"code": "ResponsibleAIPolicyViolation"}}, "content_filter", 400),
    (401, {}, "authentication", 502),
    (403, {}, "authentication", 502),
    (408, {}, "timeout", 504),
    (504, {}, "timeout", 504),
    (500, {}, "unavailable", 503),
    (503, {}, "unavailable", 503),
    (400, {"code": "invalid_parameter"}, "invalid_request", 502),
    (404, {"code": "DeploymentNotFound"}, "invalid_request", 502),
])
async def test_status_errors_are_classified_without_provider_text(
    client, capsys, status, body, kind, expected_status,
):
    chat, create = client
    response = httpx.Response(
        status, headers={"Retry-After": "20"},
        request=httpx.Request("POST", "https://private-url/v1"),
    )
    create.side_effect = APIStatusError(
        "private-key at private-url", response=response, body=body,
    )

    turn = await chat.complete([], [])

    assert turn.error.kind == kind
    assert turn.error.status_code == expected_status
    if kind in {"rate_limit", "timeout", "unavailable"}:
        assert turn.error.retry_after_seconds == 20
    assert "private-" not in repr(turn)
    assert "private-" not in capsys.readouterr().out


@pytest.mark.parametrize("error, kind, status", [
    (APITimeoutError(request=httpx.Request("POST", "https://private-url")), "timeout", 504),
    (APIConnectionError(request=httpx.Request("POST", "https://private-url")), "unavailable", 503),
    (ClientAuthenticationError("private-key private-url"), "authentication", 502),
])
async def test_network_and_credential_errors_are_classified(client, error, kind, status):
    chat, create = client
    create.side_effect = error

    turn = await chat.complete([], [])

    assert turn.error.kind == kind and turn.error.status_code == status
    assert "private-" not in repr(turn)


@pytest.mark.parametrize("headers, expected", [
    ({"Retry-After": "20"}, 20),
    ({"Retry-After": "0"}, 0),
    ({"Retry-After": "2.2"}, 3),
    ({"Retry-After": "180"}, 180),
    ({"retry-after-ms": "1500"}, 2),
    ({"x-ms-retry-after-ms": "1500"}, 2),
    ({"retry-after-ms": "invalid", "Retry-After": "17"}, 17),
    ({"retry-after-ms": "1000", "Retry-After": "60"}, 1),
    ({"Retry-After": "nan"}, None),
    ({"Retry-After": "inf"}, None),
    ({"Retry-After": "-1"}, None),
    ({"Retry-After": "private-key"}, None),
    ({}, None),
])
async def test_retry_after_accepts_only_safe_nonnegative_numbers(headers, expected):
    error = APIStatusError(
        "busy", response=httpx.Response(
            429, headers=headers, request=httpx.Request("POST", "https://model.example.com"),
        ), body=None,
    )
    assert ModelErrorMapper.from_exception(error).retry_after_seconds == expected


@pytest.mark.parametrize("offset, expected", [(30, 30), (-30, None)])
async def test_retry_after_accepts_a_future_http_date(monkeypatch, offset, expected):
    now = 1700000000
    monkeypatch.setattr(model_errors.time, "time", lambda: now)
    date = format_datetime(datetime.fromtimestamp(now + offset, timezone.utc), usegmt=True)

    error = APIStatusError(
        "busy", response=httpx.Response(
            429, headers={"Retry-After": date},
            request=httpx.Request("POST", "https://model.example.com"),
        ), body=None,
    )
    assert ModelErrorMapper.from_exception(error).retry_after_seconds == expected


async def test_content_filter_finish_reason_is_an_error_and_keeps_usage(client):
    chat, create = client
    response = _reply()
    response.choices[0].finish_reason = "content_filter"
    response.usage = CompletionUsage(prompt_tokens=20, completion_tokens=1, total_tokens=21)
    create.return_value = response

    turn = await chat.complete([], [])

    assert turn.error.kind == "content_filter" and turn.error.status_code == 400
    assert turn.content is None and turn.message["content"] == ""
    assert turn.usage == TokenUsage(prompt_tokens=20, completion_tokens=1)


async def test_request_cancellation_is_not_converted_to_a_model_failure(client):
    chat, create = client

    async def blocked(**options):
        await anyio.sleep_forever()

    create.side_effect = blocked
    with anyio.move_on_after(0) as cancelled:
        await chat.complete([], [])

    assert cancelled.cancelled_caught
