"""Agent HTTP requests through in-memory MCP sessions, without cloud calls."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import anyio
import httpx
import pytest
from azure.core.credentials import AccessToken
from fastapi.testclient import TestClient
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from openai import AsyncOpenAI

from cosmos_agentic_retriever import server
from cosmos_agentic_retriever.agent import chat_client
from cosmos_agentic_retriever.agent.chat_client import ChatClient
from cosmos_agentic_retriever.agent.loop import LlmTurn, ModelError, TokenUsage
from cosmos_agentic_retriever.agent.tool_contracts import ToolCall
from cosmos_agentic_retriever.agent.toolkit_client import ToolkitClient
from cosmos_agentic_retriever.config import AgenticSearchConfig
from cosmos_agentic_retriever.server import create_app


class _FakeChat:
    model = "fake-model"

    def __init__(self, turns: list[LlmTurn]) -> None:
        self._turns = list(turns)
        self.tools_seen: list[set[str]] = []
        self.messages_seen: list[list[dict]] = []

    async def complete(self, messages, tools) -> LlmTurn:
        self.tools_seen.append({tool.name for tool in tools})
        self.messages_seen.append(list(messages))
        return self._turns.pop(0)


class _Toolkit:
    def __init__(self) -> None:
        self.server = FastMCP("endpoint-test")
        self.calls: list[tuple[str, dict]] = []
        self.results = {"hi": [{
            "id": "d1", "title": "Evidence", "text": "a" * 1000,
            "embedding": [0.1] * 128, "_rid": "internal",
        }]}

        @self.server.tool()
        async def list_databases() -> list[str]:
            self.calls.append(("list_databases", {}))
            return ["D", "OTHER"]

        @self.server.tool()
        async def list_collections(databaseId: str) -> list[str]:
            self.calls.append(("list_collections", {"databaseId": databaseId}))
            return ["C", "other"]

        @self.server.tool()
        async def get_approximate_schema(databaseId: str, containerId: str) -> dict:
            self.calls.append(("get_approximate_schema", {
                "databaseId": databaseId, "containerId": containerId,
            }))
            return {"sampleSize": 1, "properties": [{"name": "text", "type": "string"}]}

        @self.server.tool()
        async def text_search(
            databaseId: str, containerId: str, property: str, searchPhrase: str, n: int = 10,
        ) -> str:
            self.calls.append(("text_search", {
                "databaseId": databaseId, "containerId": containerId,
                "property": property, "searchPhrase": searchPhrase, "n": n,
            }))
            if searchPhrase == "fail":
                raise RuntimeError("Cosmos busy")
            return json.dumps(self.results.get(searchPhrase, [])[:n])

        @self.server.tool()
        async def find_document_by_id(databaseId: str, containerId: str, id: str) -> dict:
            self.calls.append(("find_document_by_id", {
                "databaseId": databaseId, "containerId": containerId, "id": id,
            }))
            return next(item for items in self.results.values() for item in items if item["id"] == id)


def _settings(**options) -> AgenticSearchConfig:
    options.setdefault("llm_api_key", "test-key")
    options.setdefault("toolkit_app_id", "app-123")
    return AgenticSearchConfig(
        llm_base_url="https://model.example.com/v1",
        llm_model="test-model",
        **options,
    )


def _app(settings: AgenticSearchConfig):
    return create_app(settings)


@pytest.fixture
def http():
    toolkit = _Toolkit()
    with TestClient(_app(_settings())) as client:
        client.app.state.toolkit = ToolkitClient(
            lambda: create_connected_server_and_client_session(toolkit.server)
        )
        yield client, toolkit


def _call(name="text_search", arguments=None, call_id="c1") -> LlmTurn:
    if arguments is None:
        arguments = {
            "databaseId": "D", "containerId": "C", "property": "text", "searchPhrase": "hi",
        }
    return LlmTurn(
        message={"role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]},
        tool_calls=[ToolCall(call_id, name, arguments)],
        usage=TokenUsage(100, 10, 40, 6),
    )


def _answer(text="the answer") -> LlmTurn:
    return LlmTurn(
        message={"role": "assistant", "content": text}, content=text,
        usage=TokenUsage(150, 5, 0, 0),
    )


def test_agent_search_runs_toolkit_tools_and_returns_full_evidence(http):
    client, toolkit = http
    chat = _FakeChat([_call(), _answer("the answer\nSources: [D/C/d1]")])
    client.app.state.chat_client = chat

    response = client.post("/agentic_search", json={"query": "what is up?"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == "the answer\nSources: [D/C/d1]"
    assert body["terminal_reason"] == "stop" and body["turns"] == 2
    assert body["tool_calls"] == {"text_search": 1} and body["model"] == "fake-model"
    assert body["usage"] == {
        "prompt_tokens": 250, "completion_tokens": 15,
        "cached_prompt_tokens": 40, "reasoning_tokens": 6,
    }
    assert body["searched"] == [{"database": "D", "container": "C"}]
    assert body["errors"] == [] and body["partial"] is False
    [document] = body["documents"]
    assert document["database"] == "D" and document["container"] == "C" and document["id"] == "d1"
    assert document["document"]["text"] == "a" * 1000
    assert document["document"]["embedding"] == "[128-dimension vector omitted]"
    assert "_rid" not in document["document"]
    assert "more characters" in chat.messages_seen[-1][-1]["content"]
    assert toolkit.calls[0][0] == "text_search"


@pytest.mark.parametrize("scope, listing", [
    ({}, {"list_databases", "list_collections"}),
    ({"database": "D"}, {"list_collections"}),
    ({"database": "D", "container": "C"}, set()),
])
def test_scope_selects_discovery_tools_and_prompt(http, scope, listing):
    client, _ = http
    chat = _FakeChat([_answer()])
    client.app.state.chat_client = chat

    response = client.post("/agentic_search", json={"query": "question", **scope})

    assert response.status_code == 200
    assert chat.tools_seen[0] & {"list_databases", "list_collections"} == listing
    assert "agentic_search" not in chat.tools_seen[0] and "full_text_search" not in chat.tools_seen[0]
    prompt = chat.messages_seen[0][0]["content"]
    assert "get_approximate_schema" in prompt and "selectProperties must include id" in prompt
    assert "read them in full" in prompt
    assert "[database/container/id]" in prompt


def test_agent_can_discover_then_search_without_scope(http):
    client, toolkit = http
    client.app.state.chat_client = _FakeChat([
        _call("list_databases", {}),
        _call("list_collections", {"databaseId": "OTHER"}, "c2"),
        _call("get_approximate_schema", {"databaseId": "OTHER", "containerId": "other"}, "c3"),
        _call(arguments={
            "databaseId": "OTHER", "containerId": "other", "property": "text", "searchPhrase": "hi",
        }, call_id="c4"),
        _answer(),
    ])

    response = client.post("/agentic_search", json={"query": "find something"})

    assert response.status_code == 200, response.text
    assert response.json()["documents"][0]["database"] == "OTHER"
    assert [name for name, _ in toolkit.calls] == [
        "list_databases", "list_collections", "get_approximate_schema", "text_search",
    ]


@pytest.mark.parametrize("arguments", [
    {"databaseId": "OTHER", "containerId": "C", "property": "text", "searchPhrase": "hi"},
    {"databaseId": "D", "containerId": "other", "property": "text", "searchPhrase": "hi"},
])
def test_out_of_scope_call_never_reaches_toolkit(http, arguments):
    client, toolkit = http
    client.app.state.chat_client = _FakeChat([_call(arguments=arguments), _answer()])

    response = client.post("/agentic_search", json={
        "query": "question", "database": "D", "container": "C",
    })

    assert response.status_code == 500
    assert toolkit.calls == [] and response.json()["errors"]
    assert response.json()["answer"] == ""


@pytest.mark.parametrize("limit, service_cap", [(1, 10), (50, 1), (8, 10)])
def test_evidence_limit_and_citation_priority(http, limit, service_cap):
    client, toolkit = http
    toolkit.results["hi"] = [{"id": f"d{index}", "text": str(index)} for index in range(10)]
    app = _app(_settings(agent_max_documents=service_cap))
    with TestClient(app) as scoped:
        scoped.app.state.toolkit = client.app.state.toolkit
        scoped.app.state.chat_client = _FakeChat([_call(), _answer("Sources: [D/C/d9]")])
        response = scoped.post("/agentic_search", json={"query": "question", "maxDocuments": limit})

    assert response.status_code == 200
    documents = response.json()["documents"]
    assert len(documents) == min(limit, service_cap)
    assert documents[0]["id"] == "d9"


@pytest.mark.parametrize("partial", [False, True])
def test_tool_failures_are_reported_without_claiming_no_matches(http, partial):
    client, _ = http
    failed = _call(arguments={
        "databaseId": "D", "containerId": "C", "property": "text", "searchPhrase": "fail",
    })
    turns = [failed] + ([_call(call_id="c2")] if partial else []) + [_answer()]
    client.app.state.chat_client = _FakeChat(turns)

    response = client.post("/agentic_search", json={"query": "question"})

    assert response.status_code == (200 if partial else 500), response.text
    body = response.json()
    assert body["partial"] is partial and "Cosmos busy" in body["errors"][0]["error"]
    if not partial:
        assert body["answer"] == "" and body["documents"] == []


def test_empty_search_is_not_a_failure(http):
    client, _ = http
    client.app.state.chat_client = _FakeChat([
        _call(arguments={"databaseId": "D", "containerId": "C", "property": "text", "searchPhrase": "none"}),
        _answer("No matching documents found."),
    ])
    response = client.post("/agentic_search", json={"query": "question"})
    assert response.status_code == 200 and response.json()["errors"] == []
    assert response.json()["documents"] == []


def test_model_failure_keeps_evidence_but_not_unfinished_answer(http):
    client, _ = http
    client.app.state.chat_client = _FakeChat([
        _call(), LlmTurn(
            message={"role": "assistant", "content": ""},
            error=ModelError("unavailable", "The search model is temporarily unavailable.", 503),
        ),
    ])
    response = client.post("/agentic_search", json={"query": "question"})
    assert response.status_code == 503
    assert response.json()["error"] == "The search model is temporarily unavailable."
    assert response.json()["error_kind"] == "unavailable"
    assert response.json()["answer"] == "" and response.json()["documents"][0]["id"] == "d1"
    assert "private-" not in response.text


def test_simultaneous_questions_keep_evidence_separate(http):
    client, toolkit = http
    toolkit.results.update({name: [{"id": name, "text": name}] for name in ("one", "two")})

    class ConcurrentChat:
        model = "fake-model"

        async def complete(self, messages, tools):
            question = messages[1]["content"]
            if messages[-1]["role"] == "tool":
                return _answer(f"Sources: [D/C/{question}]")
            return _call(arguments={
                "databaseId": "D", "containerId": "C", "property": "text", "searchPhrase": question,
            })

    client.app.state.chat_client = ConcurrentChat()
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda question: client.post(
            "/agentic_search", json={"query": question, "database": "D", "container": "C"}
        ), ["one", "two"]))
    assert all(response.status_code == 200 for response in responses)
    assert [[document["id"] for document in response.json()["documents"]] for response in responses] == [["one"], ["two"]]


@pytest.mark.parametrize("options, status", [
    ({"query": "   "}, 422), ({"container": "C"}, 400),
    ({"container_filters": {}}, 422), ({"overrides": {}}, 422),
    ({"maxDocuments": 0}, 422), ({"maxDocuments": True}, 422),
])
def test_invalid_request_never_calls_model_or_toolkit(http, options, status):
    client, toolkit = http
    chat = _FakeChat([])
    client.app.state.chat_client = chat
    response = client.post("/agentic_search", json={"query": "question", **options})
    assert response.status_code == status, response.text
    assert chat.messages_seen == [] and toolkit.calls == []


def test_missing_toolkit_returns_unavailable(http):
    client, _ = http
    client.app.state.toolkit = None
    assert client.post("/agentic_search", json={"query": "question"}).status_code == 503
    assert client.get("/health").status_code == 503


def test_configured_llm_builds_chat_client():
    with TestClient(_app(_settings())) as http:
        assert isinstance(http.app.state.chat_client, ChatClient)


class _FakeIdentity:
    def __init__(self) -> None:
        self.scopes: tuple[str, ...] = ()
        self.closed = False

    async def get_token(self, *scopes, **options) -> AccessToken:
        self.scopes = scopes
        return AccessToken("entra-token", int(time.time()) + 3600)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("api_key", ["model-key", None])
@pytest.mark.parametrize("max_retries", [0, 6])
def test_chat_client_signs_in_with_key_or_entra(monkeypatch, api_key, max_retries):
    identity = _FakeIdentity()
    monkeypatch.setattr(server, "AsyncDefaultAzureCredential", lambda: identity)
    sdk_options: dict = {}

    def fake_sdk(**options):
        sdk_options.update(options)
        return Mock(close=AsyncMock())

    monkeypatch.setattr(chat_client, "AsyncOpenAI", fake_sdk)
    with TestClient(_app(_settings(llm_api_key=api_key, llm_max_retries=max_retries))):
        assert sdk_options["max_retries"] == max_retries
        client_key = sdk_options["api_key"]
        if api_key is None:
            assert anyio.run(client_key) == "entra-token"
            assert identity.scopes == (server.LLM_TOKEN_SCOPE,)
        else:
            assert client_key == api_key
    assert identity.closed


def test_toolkit_client_signs_in_with_the_toolkit_app_and_one_credential(monkeypatch):
    created: list[_FakeIdentity] = []

    def new_identity():
        created.append(_FakeIdentity())
        return created[-1]

    monkeypatch.setattr(server, "AsyncDefaultAzureCredential", new_identity)
    opened, closed = [], []

    @asynccontextmanager
    async def fake_sessions(url, get_token):
        opened.append((url, get_token))
        yield "session factory"
        closed.append(True)

    monkeypatch.setattr(server, "toolkit_sessions", fake_sessions)
    with TestClient(_app(_settings(llm_api_key=None))) as http:
        assert isinstance(http.app.state.toolkit, ToolkitClient)
        [(url, get_token)] = opened
        assert url == "http://127.0.0.1:8080/mcp"
        assert anyio.run(get_token) == "entra-token"
        assert created[0].scopes == ("api://app-123/.default",)
    assert closed == [True] and len(created) == 1 and created[0].closed


@pytest.mark.parametrize("max_retries", [0, 2, 6])
def test_exhausted_sdk_retries_report_model_unavailable(monkeypatch, max_retries):
    requests = []

    def unavailable(request):
        requests.append(request)
        return httpx.Response(503, json={"error": {"message": "private endpoint failure"}})

    delay = AsyncMock()
    monkeypatch.setattr("openai._base_client.anyio.sleep", delay)
    transport = httpx.AsyncClient(transport=httpx.MockTransport(unavailable))
    monkeypatch.setattr(chat_client, "AsyncOpenAI", lambda **options: AsyncOpenAI(http_client=transport, **options))
    with TestClient(_app(_settings(llm_max_retries=max_retries))) as http:
        response = http.post("/agentic_search", json={"query": "question"})
    assert len(requests) == max_retries + 1 and delay.await_count == max_retries
    assert response.status_code == 503 and response.json()["error_kind"] == "unavailable"
    assert "private" not in response.text


def _sdk_reply(message, finish_reason="stop"):
    return httpx.Response(200, json={
        "id": "model-reply", "object": "chat.completion", "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
    })


def _mock_model_sdk(monkeypatch, respond):
    delay = AsyncMock()
    monkeypatch.setattr("openai._base_client.anyio.sleep", delay)
    transport = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(
        chat_client, "AsyncOpenAI",
        lambda **options: AsyncOpenAI(http_client=transport, **options),
    )
    return delay


def test_exhausted_rate_limit_reports_retry_hint_and_keeps_evidence(monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        messages = json.loads(request.content)["messages"]
        if messages[-1]["role"] == "user":
            return _sdk_reply(_call().message, "tool_calls")
        return httpx.Response(
            429, headers={"Retry-After": "20"},
            json={"error": {"code": "rate_limit_exceeded", "message": "private-key private-url"}},
        )

    delay = _mock_model_sdk(monkeypatch, respond)
    toolkit = _Toolkit()
    with TestClient(_app(_settings(llm_max_retries=2))) as http:
        http.app.state.toolkit = ToolkitClient(
            lambda: create_connected_server_and_client_session(toolkit.server)
        )
        response = http.post("/agentic_search", json={"query": "question"})

    body = response.json()
    assert response.status_code == 429 and body["error_kind"] == "rate_limit"
    assert "rate limit" in body["error"]
    assert body["retry_after_seconds"] == 20 and response.headers["Retry-After"] == "20"
    assert body["answer"] == "" and body["documents"][0]["id"] == "d1"
    assert body["turns"] == 2 and body["tool_calls"] == {"text_search": 1}
    assert len(requests) == 4 and len(toolkit.calls) == 1
    assert delay.await_count == 2
    delay.assert_any_await(20)
    assert "private-" not in response.text


def test_transient_rate_limits_recover_without_restarting_the_question(monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        if len(requests) <= 2:
            return httpx.Response(
                429, headers={"Retry-After": "0.01"},
                json={"error": {"message": "busy", "code": "rate_limit_exceeded"}},
            )
        messages = json.loads(request.content)["messages"]
        if messages[-1]["role"] == "user":
            return _sdk_reply(_call().message, "tool_calls")
        return _sdk_reply(_answer().message)

    delay = _mock_model_sdk(monkeypatch, respond)
    toolkit = _Toolkit()
    with TestClient(_app(_settings(llm_max_retries=2))) as http:
        http.app.state.toolkit = ToolkitClient(
            lambda: create_connected_server_and_client_session(toolkit.server)
        )
        response = http.post("/agentic_search", json={"query": "question"})

    body = response.json()
    assert response.status_code == 200 and body["answer"] == "the answer"
    assert body["turns"] == 2 and body["tool_calls"] == {"text_search": 1}
    assert "error_kind" not in body and "Retry-After" not in response.headers
    assert len(requests) == 4 and len(toolkit.calls) == 1
    assert delay.await_count == 2
    delay.assert_any_await(0.01)


@pytest.mark.parametrize("status, code, kind, expected_status", [
    (400, "content_filter", "content_filter", 400),
    (401, "Unauthorized", "authentication", 502),
    (403, "Forbidden", "authentication", 502),
])
def test_nonretryable_model_errors_have_safe_messages(monkeypatch, status, code, kind, expected_status):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(status, json={
            "error": {"code": code, "message": "private-key private-url"},
        })

    delay = _mock_model_sdk(monkeypatch, respond)
    with TestClient(_app(_settings())) as http:
        response = http.post("/agentic_search", json={"query": "question"})

    assert response.status_code == expected_status and response.json()["error_kind"] == kind
    assert "private-" not in response.text
    assert "WWW-Authenticate" not in response.headers
    assert len(requests) == 1
    delay.assert_not_awaited()


def test_content_filtered_success_response_does_not_look_like_an_empty_answer(monkeypatch):
    _mock_model_sdk(monkeypatch, lambda request: _sdk_reply(
        {"role": "assistant", "content": ""}, "content_filter"
    ))
    with TestClient(_app(_settings())) as http:
        response = http.post("/agentic_search", json={"query": "question"})

    assert response.status_code == 400
    assert response.json()["error_kind"] == "content_filter"
    assert response.json()["answer"] == ""


@pytest.mark.parametrize("path", ["/search", "/agent_search"])
def test_removed_agent_routes_return_not_found(http, path):
    client, _ = http
    assert client.post(path, json={"query": "question"}).status_code == 404