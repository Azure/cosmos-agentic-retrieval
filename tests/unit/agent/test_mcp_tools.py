"""MCP tools as agent tools, against an in-memory MCP server."""

from __future__ import annotations

from contextlib import asynccontextmanager

import anyio
import pytest
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

from cosmos_agentic_retriever.agent import mcp_tools
from cosmos_agentic_retriever.agent.mcp_tools import open_mcp_tools, signed_in_session
from cosmos_agentic_retriever.agent.tool_contracts import ToolCall, execute_tool_call

pytestmark = pytest.mark.anyio


def _server() -> FastMCP:
    server = FastMCP("fake-toolkit")

    @server.tool(description="Find documents containing a phrase.")
    def text_search(container: str, phrase: str, n: int = 10) -> str:
        return f"{n} results for {phrase!r} in {container}"

    @server.tool(description="Always fails.")
    def broken() -> str:
        raise RuntimeError("Cosmos is down")

    @server.tool(description="The service itself.")
    def agentic_search(query: str) -> str:
        return "should not be offered"

    return server


def _session():
    return create_connected_server_and_client_session(_server())


async def test_offers_server_tools_except_agentic_search():
    async with open_mcp_tools(_session) as tools:
        by_name = {tool.name: tool for tool in tools}

    assert set(by_name) == {"text_search", "broken"}
    search = by_name["text_search"]
    assert search.description == "Find documents containing a phrase."
    assert search.parameters["required"] == ["container", "phrase"]
    assert set(search.parameters["properties"]) == {"container", "phrase", "n"}


async def test_exclude_can_be_changed():
    async with open_mcp_tools(_session, exclude=frozenset({"broken"})) as tools:
        names = {tool.name for tool in tools}

    assert names == {"text_search", "agentic_search"}


async def test_call_returns_the_tool_text():
    async with open_mcp_tools(_session) as tools:
        search = next(tool for tool in tools if tool.name == "text_search")
        result = await search(container="docs", phrase="battery", n=3)

    assert result.error is None
    assert result.content == "3 results for 'battery' in docs"


async def test_batch_runs_each_call_in_order():
    async with open_mcp_tools(_session) as tools:
        search = next(tool for tool in tools if tool.name == "text_search")
        results = await search.handler(
            [{"container": "a", "phrase": "x"}, {"container": "b", "phrase": "y", "n": 1}]
        )

    assert [result.content for result in results] == [
        "10 results for 'x' in a",
        "1 results for 'y' in b",
    ]


@pytest.mark.parametrize(
    "name, arguments, message",
    [("broken", {}, "Cosmos is down"), ("text_search", {"container": "a"}, "phrase")],
)
async def test_server_reported_failure_becomes_error(name, arguments, message):
    async with open_mcp_tools(_session) as tools:
        by_name = {tool.name: tool for tool in tools}
        response = await execute_tool_call(by_name, ToolCall("c1", name, arguments))

    assert response.content == ""
    assert message in (response.error or "")


async def test_tools_fail_after_the_session_closes():
    async with open_mcp_tools(_session) as tools:
        search = next(tool for tool in tools if tool.name == "text_search")

    with anyio.fail_after(5):
        result = await search(container="docs", phrase="battery")

    assert result.content == ""
    assert result.error


async def test_signed_in_session_sends_a_fresh_token_per_session(monkeypatch):
    opened: list[tuple[str, dict]] = []

    @asynccontextmanager
    async def fake_session(url, *, headers):
        opened.append((url, headers))
        yield "session"

    monkeypatch.setattr(mcp_tools, "streamable_http_session", fake_session)
    tokens = iter(["token-1", "token-2"])

    async def get_token() -> str:
        return next(tokens)

    open_session = signed_in_session("http://127.0.0.1:8080/mcp", get_token)
    async with open_session() as first:
        pass
    async with open_session():
        pass

    assert first == "session"
    assert opened == [
        ("http://127.0.0.1:8080/mcp", {"Authorization": "Bearer token-1"}),
        ("http://127.0.0.1:8080/mcp", {"Authorization": "Bearer token-2"}),
    ]
