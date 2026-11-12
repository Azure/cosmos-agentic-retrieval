"""Call the MCP toolkit's tools, with one MCP session per call.

Each `ToolkitClient.call_tool` opens its own MCP session, makes the call, and
closes the session. The MCP client library ends a whole session when any one
request on it fails or times out, and can leave the session's other calls
waiting forever. With a session per call, a failure only affects that call.
Every call also has a time limit, so none can hang.

`toolkit_sessions` keeps one HTTP client for the life of the service, so the TCP
connections underneath are reused across calls. It asks `get_token` for a fresh
bearer token on every HTTP request. Tests pass a factory for in-memory sessions
instead.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

import anyio
import httpx
import structlog
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client

from cosmos_agentic_retriever.agent.tool_contracts import ToolExecutionResult

SessionFactory = Callable[[], AbstractAsyncContextManager[ClientSession]]

_logger = structlog.get_logger(__name__)


class BearerAuth(httpx.Auth):
    """Add a fresh bearer token from ``get_token`` to every request."""

    def __init__(self, get_token: Callable[[], Awaitable[str]]) -> None:
        self._get_token = get_token

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        request.headers["Authorization"] = f"Bearer {await self._get_token()}"
        yield request


@asynccontextmanager
async def toolkit_sessions(
    url: str, get_token: Callable[[], Awaitable[str]], *, timeout: float = 120.0
) -> AsyncIterator[SessionFactory]:
    """Yield a factory of MCP sessions to ``url`` that share one HTTP client."""
    async with httpx.AsyncClient(auth=BearerAuth(get_token), timeout=timeout) as http:

        @asynccontextmanager
        async def open_session() -> AsyncIterator[ClientSession]:
            async with (
                streamable_http_client(url, http_client=http) as (read, write, _),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                yield session

        yield open_session


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


def _root_cause(exc: BaseException) -> BaseException:
    """The first underlying error inside nested exception groups."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


class ToolkitClient:
    """Call toolkit tools by name, each over its own MCP session, within a time limit."""

    def __init__(self, open_session: SessionFactory, *, call_timeout: float = 120.0) -> None:
        self._open_session = open_session
        self._call_timeout = call_timeout

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolExecutionResult:
        """Call a toolkit tool; a failure or timeout becomes the result's error."""
        try:
            with anyio.fail_after(self._call_timeout):
                async with self._open_session() as session:
                    reply = await session.call_tool(name, arguments)
        except TimeoutError:
            _logger.warning("toolkit_call_timed_out", tool=name, seconds=self._call_timeout)
            return ToolExecutionResult(
                content="", error=f"The toolkit did not answer within {self._call_timeout:g} seconds."
            )
        except Exception as exc:
            cause = _root_cause(exc)
            _logger.warning("toolkit_call_failed", tool=name, error=repr(cause))
            return ToolExecutionResult(
                content="", error=f"The toolkit call failed: {type(cause).__name__}: {cause}"
            )
        return _to_result(reply)

    async def list_tools(self) -> list[types.Tool]:
        """Read every page of the toolkit's tool list."""
        with anyio.fail_after(self._call_timeout):
            async with self._open_session() as session:
                tools: list[types.Tool] = []
                cursor: str | None = None
                while True:
                    page = await session.list_tools(cursor)
                    tools.extend(page.tools)
                    cursor = page.nextCursor
                    if not cursor:
                        return tools
