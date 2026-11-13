"""Answer questions with toolkit tools; retain the legacy full-text HTTP endpoint."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager
from dataclasses import asdict
from typing import Any

import anyio
import structlog
from azure.cosmos import CosmosClient
from azure.identity import AzureCliCredential, DefaultAzureCredential
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
from cosmos_agentic_retriever.config import RetrieverConfig, get_config
from cosmos_agentic_retriever.orchestration import (
    ContainerTarget,
    MultiContainerRetriever,
)
from cosmos_agentic_retriever.query_engine import CosmosExecutor
from cosmos_agentic_retriever.query_engine.full_text_terms import tokenize_for_fts
from cosmos_agentic_retriever.query_engine.retriever import CorpusRetriever
from cosmos_agentic_retriever.query_engine.types import (
    FilterExpression,
    QueryCompilationError,
    RetrievalError,
)
from cosmos_agentic_retriever.query_engine.types import (
    SearchRequest as QuerySearchRequest,
)

# Entra token audience for Azure OpenAI and Azure AI Foundry model endpoints.
LLM_TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"


def toolkit_token_scope(app_id: str) -> str:
    """Entra scope for tokens to the MCP toolkit whose Entra app has this client ID."""
    return f"api://{app_id}/.default"


class AgenticSearchRequest(BaseModel):
    """A question with optional discovery scope and a cap on returned evidence."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    query: str = Field(min_length=1, max_length=4096)
    max_documents: int = Field(
        default=20, strict=True, ge=1, le=50, alias="maxDocuments"
    )
    database: str | None = Field(default=None, min_length=1, max_length=256)
    container: str | None = Field(default=None, min_length=1, max_length=256)

    @field_validator("query")
    @classmethod
    def _reject_blank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class SearchRequest(AgenticSearchRequest):
    """Legacy full-text input with operator-configured fields and filters."""

    container_filters: dict[str, list[FilterExpression]] | None = None
    overrides: dict[str, Any] | None = None


def create_app(
    settings: RetrieverConfig | None = None,
    *,
    retrievers: dict[str, CorpusRetriever] | None = None,
) -> FastAPI:
    """Serve configured containers using explicit settings or the environment.

    Injected retrievers must match every configured container's schema and policy.
    Their clients remain caller-owned. Otherwise the app owns its client. No discovery,
    container creation, or index changes are performed.
    """
    resolved = (settings or get_config()).model_copy(deep=True)
    if retrievers is not None:
        if retrievers.keys() != resolved.cosmos_containers.keys():
            raise ValueError("retrievers must match the configured containers")
        for name, retriever in retrievers.items():
            config = resolved.cosmos_containers[name]
            if (
                retriever.schema != config.cosmos_schema
                or retriever.policy != config.partition_policy
            ):
                raise ValueError(
                    "retriever schema and partition policy must match configuration"
                )
        retrievers = dict(retrievers)
    selected_fields = {
        name: [
            str(path)
            for path in config.cosmos_schema.resolve_text_fields(
                config.search_text_fields
            )
        ]
        for name, config in resolved.cosmos_containers.items()
    }
    logger = structlog.get_logger("cosmos_agentic_retriever.server")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resources = ExitStack()
        async_resources = AsyncExitStack()
        identity: AsyncDefaultAzureCredential | None = None

        def azure_identity() -> AsyncDefaultAzureCredential:
            """The app's Azure credential, created once when first needed."""
            nonlocal identity
            if identity is None:
                identity = AsyncDefaultAzureCredential()
                async_resources.push_async_callback(identity.close)
            return identity

        def build_retrievers() -> dict[str, CorpusRetriever]:
            credential: str | AzureCliCredential | DefaultAzureCredential
            if resolved.cosmos_key is not None:
                credential = resolved.cosmos_key.get_secret_value()
            else:
                credential = (
                    DefaultAzureCredential()
                    if resolved.cosmos_credential == "default"
                    else AzureCliCredential()
                )
                resources.callback(credential.close)
            client = CosmosClient(str(resolved.account_uri), credential=credential)
            resources.callback(client.close)
            database = client.get_database_client(resolved.cosmos_database)
            executor = CosmosExecutor(config=resolved.query_engine)
            return {
                name: CorpusRetriever(
                    container=database.get_container_client(name),
                    schema=config.cosmos_schema,
                    executor=executor,
                    partition_policy=config.partition_policy,
                )
                for name, config in resolved.cosmos_containers.items()
            }

        try:
            engines = (
                retrievers
                if retrievers is not None
                else await anyio.to_thread.run_sync(build_retrievers)
            )
            app.state.retriever = MultiContainerRetriever(
                {
                    ContainerTarget(resolved.cosmos_database, name): engine
                    for name, engine in engines.items()
                },
                max_workers=resolved.query_engine.max_concurrency,
            )
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
            app.state.retriever = None
            app.state.chat_client = None
            app.state.toolkit = None
            with anyio.CancelScope(shield=True):
                await async_resources.aclose()
                await anyio.to_thread.run_sync(resources.close)

    app = FastAPI(
        title="Cosmos Retriever",
        version="0.1.0",
        description="Search over explicitly configured Cosmos containers.",
        lifespan=lifespan,
    )
    app.state.retriever = None
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

    def selected_containers(request: SearchRequest) -> list[str]:
        """Validate the caller's scope and return the selected container names."""
        if request.database not in (
            None,
            resolved.cosmos_database,
        ) or (
            request.container is not None
            and request.container not in resolved.cosmos_containers
        ):
            raise RetrievalError(
                "Only the configured database and containers are supported."
            )
        if request.overrides:
            raise RetrievalError(
                "Per-request overrides are not supported. Configure the service instead."
            )
        names = (
            [request.container]
            if request.container is not None
            else list(resolved.cosmos_containers)
        )
        if request.container_filters is not None and set(
            request.container_filters
        ) != set(names):
            raise RetrievalError(
                "container_filters must name every selected container and no others. Use [] for an unfiltered container."
            )
        return names

    @app.post("/full_text_search")
    async def full_text_search(request: SearchRequest) -> JSONResponse:
        """Run one caller-supplied query without involving the language model."""
        active: MultiContainerRetriever | None = app.state.retriever
        if active is None:
            return JSONResponse(
                status_code=503, content={"error": "Service is not ready."}
            )
        try:
            names = selected_containers(request)
            requests = {
                ContainerTarget(resolved.cosmos_database, name): QuerySearchRequest(
                    query=request.query,
                    limit=request.max_documents,
                    text_fields=selected_fields[name].copy(),
                    partition_key=resolved.cosmos_containers[name].partition_key,
                    filters=(
                        request.container_filters[name]
                        if request.container_filters is not None
                        else []
                    ),
                )
                for name in names
            }
            if not tokenize_for_fts(request.query):
                raise QueryCompilationError(
                    "full-text query must contain a searchable term"
                )
            result = await anyio.to_thread.run_sync(
                lambda: active.search(requests, limit=request.max_documents)
            )
            body = {
                "documents": [item.model_dump(mode="json") for item in result.items],
                "searched": [target._asdict() for target in result.searched],
                "errors": [
                    {**target._asdict(), "error": error}
                    for target, error in result.errors.items()
                ],
                "partial": bool(result.errors) and bool(result.searched),
            }
            if not result.searched:
                body["error"] = "Search failed for all selected containers."
                return JSONResponse(status_code=500, content=body)
            return JSONResponse(content=body)
        except RetrievalError as exc:
            return JSONResponse(
                status_code=400, content={"error": str(exc), "type": type(exc).__name__}
            )
        except Exception as exc:
            logger.exception("search_failed", error_type=type(exc).__name__)
            return JSONResponse(status_code=500, content={"error": "Search failed."})

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
        max_documents = min(request.max_documents, resolved.agent_max_documents)
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
        evidence = SearchEvidence(rules)
        tools = make_toolkit_tools(
            toolkit, scope=scope, rules=rules, names=names, evidence=evidence,
        )
        try:
            result = await run_agent_search(
                request.query,
                complete=client.complete,
                tools=tools,
                system_prompt=build_system_prompt(scope),
                max_turns=resolved.agent_max_turns,
            )
        except Exception:
            logger.exception("agent_search_failed")
            return JSONResponse(
                status_code=500, content={"error": "Agent search failed."}
            )
        body = {
            "answer": result.answer,
            "documents": evidence.documents(result.answer, limit=max_documents),
            "terminal_reason": result.terminal_reason,
            "turns": result.turns,
            "model": client.model,
            "tool_calls": result.tool_calls,
            "usage": asdict(result.usage),
            "searched": evidence.searched,
            "errors": evidence.errors,
            "partial": bool(evidence.errors) and bool(evidence.searched),
        }
        if result.terminal_reason == "error":
            body.update(answer="", error="Model request failed.")
            return JSONResponse(status_code=502, content=body)
        if evidence.errors and not evidence.searched:
            body.update(answer="", error="Toolkit calls failed; no containers were searched.")
            return JSONResponse(status_code=500, content=body)
        return JSONResponse(content=body)

    return app
