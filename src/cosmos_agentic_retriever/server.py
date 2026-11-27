"""Answer questions with the MCP toolkit's tools over HTTP."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import asdict
from typing import Literal

import anyio
import structlog
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
from azure.identity.aio import get_bearer_token_provider
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from cosmos_agentic_retriever.agent import (
    ChatClient,
    run_agent_search,
)
from cosmos_agentic_retriever.agent.result_trimming import TrimRules
from cosmos_agentic_retriever.agent.prompts import build_system_prompt
from cosmos_agentic_retriever.agent.toolkit_client import ToolkitClient, toolkit_sessions
from cosmos_agentic_retriever.agent.toolkit_tools import (
    TOOLKIT_TOOLS,
    SearchEvidence,
    SearchScope,
    make_toolkit_tools,
)
from cosmos_agentic_retriever.config import AgenticSearchConfig, get_config

# Entra token audience for Azure OpenAI and Azure AI Foundry model endpoints.
LLM_TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"


def toolkit_token_scope(app_id: str) -> str:
    """Entra scope for tokens to the MCP toolkit whose Entra app has this client ID."""
    return f"api://{app_id}/.default"


def _split_reply(reply: str) -> tuple[str, str]:
    """Split the agent's reply into the answer and explanation the prompt asks for.

    A reply that doesn't follow that format is returned whole as the answer.
    """
    head, heading, explanation = reply.partition("Explanation:")
    head = head.strip()
    if not heading or not head.startswith("Answer:"):
        return reply.strip(), ""
    return head.removeprefix("Answer:").strip(), explanation.strip()


class AgenticSearchRequest(BaseModel):
    """A question with optional discovery scope and per-request settings.

    Settings left out use the service's soft limits; values above a hard limit
    are lowered to it.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    query: str = Field(min_length=1, max_length=4096)
    max_documents: int | None = Field(
        default=None, strict=True, ge=1, le=50, alias="maxDocuments"
    )
    max_turns: int | None = Field(
        default=None, strict=True, ge=1, le=50, alias="maxTurns"
    )
    returned_documents: Literal["cited", "cited_and_ranked"] | None = Field(
        default=None, alias="returnedDocuments"
    )
    database: str | None = Field(default=None, min_length=1, max_length=256)
    container: str | None = Field(default=None, min_length=1, max_length=256)

    @field_validator("query")
    @classmethod
    def _reject_blank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


def create_app(settings: AgenticSearchConfig | None = None) -> FastAPI:
    """Serve agentic search using explicit settings or the environment."""
    resolved = (settings or get_config()).model_copy(deep=True)
    logger = structlog.get_logger("cosmos_agentic_retriever.server")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async_resources = AsyncExitStack()
        identity: AsyncDefaultAzureCredential | None = None

        def azure_identity() -> AsyncDefaultAzureCredential:
            """The app's Azure credential, created once when first needed."""
            nonlocal identity
            if identity is None:
                identity = AsyncDefaultAzureCredential()
                async_resources.push_async_callback(identity.close)
            return identity

        try:
            llm_auth: str | Callable[[], Awaitable[str]]
            if resolved.llm_api_key is not None:
                llm_auth = resolved.llm_api_key.get_secret_value()
            else:
                llm_auth = get_bearer_token_provider(azure_identity(), LLM_TOKEN_SCOPE)
            chat_client = ChatClient(
                base_url=str(resolved.llm_base_url),
                model=resolved.llm_model,
                api_key=llm_auth,
                params=dict(resolved.llm_params),
                max_retries=resolved.llm_max_retries,
            )
            async_resources.push_async_callback(chat_client.close)
            app.state.chat_client = chat_client
            url = str(resolved.toolkit_mcp_url)
            get_token = get_bearer_token_provider(
                azure_identity(), toolkit_token_scope(resolved.toolkit_app_id)
            )
            app.state.toolkit = ToolkitClient(
                await async_resources.enter_async_context(toolkit_sessions(url, get_token))
            )
            yield
        finally:
            app.state.chat_client = None
            app.state.toolkit = None
            with anyio.CancelScope(shield=True):
                await async_resources.aclose()

    app = FastAPI(
        title="Cosmos Retriever",
        version="0.1.0",
        description="Answer questions with the Azure Cosmos DB MCP Toolkit's tools.",
        lifespan=lifespan,
    )
    app.state.chat_client = None
    app.state.toolkit = None

    @app.exception_handler(RequestValidationError)
    async def invalid_request(
        _request: Request, _error: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422, content={"error": "Invalid search request."}
        )

    @app.get("/health")
    async def health() -> JSONResponse:
        ready = app.state.chat_client is not None and app.state.toolkit is not None
        return JSONResponse(
            status_code=200 if ready else 503,
            content={"status": "ok" if ready else "unavailable"},
        )

    @app.post("/agentic_search")
    async def agent_search(request: AgenticSearchRequest) -> JSONResponse:
        """Answer a question using tool queries restricted to the caller's scope."""
        client: ChatClient | None = app.state.chat_client
        toolkit: ToolkitClient | None = app.state.toolkit
        if toolkit is None or client is None:
            return JSONResponse(
                status_code=503, content={"error": "Service is not ready."}
            )

        if request.container is not None and request.database is None:
            return JSONResponse(
                status_code=400, content={"error": "A container requires a database."}
            )
        max_documents = min(
            request.max_documents or resolved.agent_documents_soft_limit,
            resolved.agent_documents_hard_limit,
        )
        max_turns = min(
            request.max_turns or resolved.agent_turns_soft_limit,
            resolved.agent_turns_hard_limit,
        )
        returned_documents = request.returned_documents or resolved.returned_documents
        scope = SearchScope(
            database=request.database,
            containers=(frozenset({request.container}) if request.container is not None else None),
        )
        names = {spec.name for spec in TOOLKIT_TOOLS}
        if scope.database is not None:
            names.remove("list_databases")
        if scope.containers is not None:
            names.remove("list_collections")
        rules = TrimRules(
            text_limit=resolved.tool_result_text_limit,
            min_vector_dim=resolved.tool_result_min_vector_dim,
            dropped_fields=frozenset(resolved.tool_result_dropped_fields),
        )
        evidence = SearchEvidence()
        tools = make_toolkit_tools(
            toolkit, scope=scope, rules=rules, names=names, evidence=evidence,
        )
        try:
            result = await run_agent_search(
                request.query,
                complete=client.complete,
                tools=tools,
                system_prompt=build_system_prompt(scope),
                max_turns=max_turns,
            )
        except Exception:
            logger.exception("agent_search_failed")
            return JSONResponse(
                status_code=500, content={"error": "Agent search failed."}
            )
        if result.error is None:
            reply, cited = evidence.verify_citations(result.answer)
            answer, explanation = _split_reply(reply)
        else:
            answer, explanation, cited = "", "", []
        # Without a finished answer nothing is cited, so return what was found instead.
        include_uncited = (
            returned_documents == "cited_and_ranked" or result.error is not None
        )
        body = {
            "answer": answer,
            "explanation": explanation,
            "documents": evidence.documents(
                cited, limit=max_documents, include_uncited=include_uncited
            ),
            "terminal_reason": result.terminal_reason,
            "turns": result.turns,
            "model": client.model,
            "tool_calls": result.tool_calls,
            "usage": asdict(result.usage),
            "searched": evidence.searched,
            "errors": evidence.errors,
            "partial": bool(evidence.errors) and bool(evidence.searched),
            "settings_used": {
                "max_turns": max_turns,
                "max_documents": max_documents,
                "returned_documents": returned_documents,
            },
        }
        if result.error is not None:
            error = result.error
            body.update(answer="", error=error.message, error_kind=error.kind)
            headers = {}
            if error.retry_after_seconds is not None:
                body["retry_after_seconds"] = error.retry_after_seconds
                headers["Retry-After"] = str(error.retry_after_seconds)
            return JSONResponse(status_code=error.status_code, content=body, headers=headers)
        if evidence.errors and not evidence.searched:
            body.update(
                answer="", explanation="",
                error="Toolkit calls failed; no containers were searched.",
            )
            return JSONResponse(status_code=500, content=body)
        return JSONResponse(content=body)

    return app
