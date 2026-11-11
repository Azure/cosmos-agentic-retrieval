"""Offer an MCP server's tools to the agent as `Tool`s.

The agent loop is synchronous and the MCP client is async, so `open_mcp_tools`
runs one MCP session on a background event loop (an anyio blocking portal) and
returns one `Tool` per server tool. Each tool's handler sends its calls over that
session and turns the reply into a `ToolExecutionResult`. The session stays open
until the context exits, so one agent run uses one session.

Tools named in ``exclude`` are not offered. By default that is ``agentic_search``,
so the agent cannot call the service it runs in.

`streamable_http_session` opens a session to a server over streamable HTTP, such
as the MCP toolkit. Tests pass a factory for an in-memory session instead.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager

import httpx
from anyio.from_thread import BlockingPortal, start_blocking_portal
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


def _to_tool(portal: BlockingPortal, session: ClientSession, spec: types.Tool) -> Tool:
    """Wrap one server tool; its handler calls the server once per call, in order."""

    def handler(calls: list[dict]) -> list[ToolExecutionResult]:
        return [
            _to_result(portal.call(session.call_tool, spec.name, arguments))
            for arguments in calls
        ]

    return Tool(
        name=spec.name,
        description=spec.description or "",
        parameters=spec.inputSchema,
        handler=handler,
    )


@contextmanager
def open_mcp_tools(
    open_session: SessionFactory,
    *,
    exclude: frozenset[str] = DEFAULT_EXCLUDED_TOOLS,
) -> Iterator[list[Tool]]:
    """Open an MCP session and yield its tools, except those in ``exclude``.

    ``open_session`` returns an async context manager yielding an initialized
    session, such as ``lambda: streamable_http_session(url, headers=...)``. The
    tools work only inside this context; the session closes when it exits.
    """
    with (
        start_blocking_portal() as portal,
        portal.wrap_async_context_manager(open_session()) as session,
    ):
        specs = portal.call(_list_all_tools, session)
        yield [_to_tool(portal, session, spec) for spec in specs if spec.name not in exclude]
