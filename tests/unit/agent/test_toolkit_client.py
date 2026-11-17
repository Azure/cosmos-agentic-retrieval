"""The toolkit client: one session per call, isolated failures, time limits, tokens."""

from __future__ import annotations

import socket
import threading
import time
from contextlib import asynccontextmanager

import anyio
import httpx
import pytest
import uvicorn
from mcp import types
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

from cosmos_agentic_retriever.agent.toolkit_client import (
    BearerAuth,
    ToolkitClient,
    toolkit_sessions,
)

pytestmark = pytest.mark.anyio


def _server(**options) -> FastMCP:
    server = FastMCP("fake-toolkit", **options)

    @server.tool(description="Find documents containing a phrase.")
    def text_search(container: str, phrase: str, n: int = 10) -> str:
        return f"{n} results for {phrase!r} in {container}"

    @server.tool(description="Always fails.")
    def broken() -> str:
        raise RuntimeError("Cosmos is down")

    @server.tool(description="Answers after a delay.")
    async def wait(seconds: float) -> str:
        await anyio.sleep(seconds)
        return f"waited {seconds}"

    return server


class _Counted:
    """An in-memory session factory that counts the sessions it opens."""

    def __init__(self) -> None:
        self.opened = 0

    @asynccontextmanager
    async def __call__(self):
        self.opened += 1
        async with create_connected_server_and_client_session(_server()) as session:
            yield session


async def test_calls_a_tool_by_name():
    result = await ToolkitClient(_Counted()).call_tool(
        "text_search", {"container": "docs", "phrase": "battery"}
    )

    assert result.error is None
    assert result.content == "10 results for 'battery' in docs"


async def test_server_reported_failure_becomes_error():
    result = await ToolkitClient(_Counted()).call_tool("broken", {})

    assert result.content == ""
    assert "Cosmos is down" in (result.error or "")


async def test_lists_the_toolkit_tools():
    tools = await ToolkitClient(_Counted()).list_tools()

    assert sorted(tool.name for tool in tools) == ["broken", "text_search", "wait"]


async def test_each_call_opens_its_own_session():
    sessions = _Counted()
    client = ToolkitClient(sessions)

    for phrase in ["a", "b", "c"]:
        await client.call_tool("text_search", {"container": "c", "phrase": phrase})

    assert sessions.opened == 3


async def test_concurrent_calls_get_their_own_replies():
    client = ToolkitClient(_Counted())
    results: dict[int, str] = {}

    async def ask(i: int) -> None:
        reply = await client.call_tool("text_search", {"container": "c", "phrase": f"q{i}"})
        results[i] = reply.content

    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            for i in range(20):
                group.start_soon(ask, i)

    assert results == {i: f"10 results for 'q{i}' in c" for i in range(20)}


async def test_a_call_over_its_time_limit_returns_an_error():
    client = ToolkitClient(_Counted(), call_timeout=0.2)

    with anyio.fail_after(5):
        result = await client.call_tool("wait", {"seconds": 10})

    assert result.content == ""
    assert result.error == "The toolkit did not answer within 0.2 seconds."


async def test_connection_errors_become_readable_errors():
    @asynccontextmanager
    async def refused():
        raise ExceptionGroup("unhandled errors in a TaskGroup", [ConnectionError("refused")])
        yield

    result = await ToolkitClient(refused).call_tool("text_search", {})

    assert result.error == "The toolkit call failed: ConnectionError: refused"


@pytest.fixture(scope="module")
def http_toolkit_url():
    """A stateless MCP server over real HTTP on a free local port."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    app = _server(stateless_http=True, json_response=True).streamable_http_app()
    server = uvicorn.Server(uvicorn.Config(app, port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/mcp"
    server.should_exit = True
    thread.join(5)


async def _no_token() -> str:
    return "test-token"


async def test_over_http_a_timed_out_call_does_not_affect_another(http_toolkit_url):
    results: dict[str, str] = {}
    async with toolkit_sessions(http_toolkit_url, _no_token, timeout=0.6) as open_session:
        client = ToolkitClient(open_session, call_timeout=5)

        async def ask(label: str, seconds: float) -> None:
            reply = await client.call_tool("wait", {"seconds": seconds})
            results[label] = reply.error or reply.content

        with anyio.fail_after(10):
            async with anyio.create_task_group() as group:
                group.start_soon(ask, "slow", 1.5)
                group.start_soon(ask, "quick", 0.3)

        later = await client.call_tool("text_search", {"container": "c", "phrase": "x"})

    assert results["quick"] == "waited 0.3"
    assert "ReadTimeout" in results["slow"]
    assert later.content == "10 results for 'x' in c"


async def test_bearer_auth_gets_a_fresh_token_for_every_request():
    tokens = iter(["token-1", "token-2"])

    async def get_token() -> str:
        return next(tokens)

    seen: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["Authorization"])
        return httpx.Response(200)

    async with httpx.AsyncClient(
        auth=BearerAuth(get_token), transport=httpx.MockTransport(respond)
    ) as http:
        await http.get("http://toolkit/mcp")
        await http.get("http://toolkit/mcp")

    assert seen == ["Bearer token-1", "Bearer token-2"]


async def test_reply_with_non_text_content_is_marked():
    @asynccontextmanager
    async def image_session():
        class Session:
            async def call_tool(self, name, arguments):
                image = types.ImageContent(type="image", data="", mimeType="image/png")
                return types.CallToolResult(content=[image])

        yield Session()

    result = await ToolkitClient(image_session).call_tool("x", {})

    assert result.content == "[image content omitted]"
