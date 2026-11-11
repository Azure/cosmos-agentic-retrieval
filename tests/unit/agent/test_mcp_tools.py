"""MCP tools as agent tools, against an in-memory MCP server."""

from __future__ import annotations

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

from cosmos_agentic_retriever.agent.mcp_tools import open_mcp_tools
from cosmos_agentic_retriever.agent.tool_contracts import ToolCall, execute_tool_call


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


def test_offers_server_tools_except_agentic_search():
    with open_mcp_tools(_session) as tools:
        by_name = {tool.name: tool for tool in tools}

    assert set(by_name) == {"text_search", "broken"}
    search = by_name["text_search"]
    assert search.description == "Find documents containing a phrase."
    assert search.parameters["required"] == ["container", "phrase"]
    assert set(search.parameters["properties"]) == {"container", "phrase", "n"}


def test_exclude_can_be_changed():
    with open_mcp_tools(_session, exclude=frozenset({"broken"})) as tools:
        names = {tool.name for tool in tools}

    assert names == {"text_search", "agentic_search"}


def test_call_returns_the_tool_text():
    with open_mcp_tools(_session) as tools:
        search = next(tool for tool in tools if tool.name == "text_search")
        result = search(container="docs", phrase="battery", n=3)

    assert result.error is None
    assert result.content == "3 results for 'battery' in docs"


def test_batch_runs_each_call_in_order():
    with open_mcp_tools(_session) as tools:
        search = next(tool for tool in tools if tool.name == "text_search")
        results = search.handler(
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
def test_server_reported_failure_becomes_error(name, arguments, message):
    with open_mcp_tools(_session) as tools:
        by_name = {tool.name: tool for tool in tools}
        response = execute_tool_call(by_name, ToolCall("c1", name, arguments))

    assert response.content == ""
    assert message in (response.error or "")


def test_tools_fail_after_the_session_closes():
    with open_mcp_tools(_session) as tools:
        search = next(tool for tool in tools if tool.name == "text_search")

    result = search(container="docs", phrase="battery")

    assert result.content == ""
    assert result.error
