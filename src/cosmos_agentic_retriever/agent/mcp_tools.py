"""Offer an MCP server's tools to the agent as `Tool`s.

`open_mcp_tools` opens one MCP session and yields one `Tool` per server tool.
Each tool's handler sends its calls over that session and turns the reply into a
`ToolExecutionResult`. The session stays open until the context exits, so one
agent run uses one session.

Tools named in ``exclude`` are not offered. By default that is ``agentic_search``,
so the agent cannot call the service it runs in.

`streamable_http_session` opens a session to a server over streamable HTTP, such
as the MCP toolkit. `signed_in_session` wraps it so each session signs in with a
fresh bearer token. Tests pass a factory for an in-memory session instead.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager

import httpx
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client

from cosmos_agentic_retriever.agent.tool_contracts import Tool, ToolExecutionResult

SessionFactory = Callable[[], AbstractAsyncContextManager[ClientSession]]

DEFAULT_EXCLUDED_TOOLS = frozenset({"agentic_search"})


@asynccontextmanager
async def streamable_http_session(
    url: str, *, headers: Mapping[str, str] | None = None, timeout: float = 60.0
) -> AsyncIterator[ClientSession]:
    """Open and initialize an MCP session to ``url`` over streamable HTTP."""
    async with (
        httpx.AsyncClient(headers=dict(headers or {}), timeout=timeout) as http,
        streamable_http_client(url, http_client=http) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session


def signed_in_session(url: str, get_token: Callable[[], Awaitable[str]]) -> SessionFactory:
    """A session factory for ``url`` that sends a fresh bearer token with each session."""

    @asynccontextmanager
    async def open_session() -> AsyncIterator[ClientSession]:
        headers = {"Authorization": f"Bearer {await get_token()}"}
        async with streamable_http_session(url, headers=headers) as session:
            yield session

    return open_session


async def _list_all_tools(session: ClientSession) -> list[types.Tool]:
    """Read every page of the server's tool list."""
    tools: list[types.Tool] = []
    cursor: str | None = None
    while True:
        page = await session.list_tools(cursor)
        tools.extend(page.tools)
        cursor = page.nextCursor
        if not cursor:
            return tools


def _to_result(reply: types.CallToolResult) -> ToolExecutionResult:
    """Join the reply's text; a server-reported failure becomes the error."""
    parts = [
        item.text if isinstance(item, types.TextContent) else f"[{item.type} content omitted]"
        for item in reply.content
    ]
    text = "\n".join(parts)
    if reply.isError:
        return ToolExecutionResult(content="", error=text or "The tool call failed.")
    return ToolExecutionResult(content=text)


def _to_tool(session: ClientSession, spec: types.Tool) -> Tool:
    """Wrap one server tool; its handler calls the server once per call, in order."""

    async def handler(calls: list[dict]) -> list[ToolExecutionResult]:
        return [
            _to_result(await session.call_tool(spec.name, arguments)) for arguments in calls
        ]

    return Tool(
        name=spec.name,
        description=spec.description or "",
        parameters=spec.inputSchema,
        handler=handler,
    )


@asynccontextmanager
async def open_mcp_tools(
    open_session: SessionFactory,
    *,
    exclude: frozenset[str] = DEFAULT_EXCLUDED_TOOLS,
) -> AsyncIterator[list[Tool]]:
    """Open an MCP session and yield its tools, except those in ``exclude``.

    ``open_session`` returns an async context manager yielding an initialized
    session, such as ``lambda: streamable_http_session(url, headers=...)``. The
    tools work only inside this context; the session closes when it exits.
    """
    async with open_session() as session:
        specs = await _list_all_tools(session)
        yield [_to_tool(session, spec) for spec in specs if spec.name not in exclude]
