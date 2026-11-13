"""The toolkit tools the search agent can use, as agent `Tool`s.

Each definition keeps the toolkit tool's name, description and arguments, so the
agent's tools match the toolkit. They are written out here rather than read from
the toolkit at run time, so the agent's tools stay the same until we change them;
`tests/live/test_toolkit_tools.py` checks them against a running toolkit.

Search tools (`text_search`, `vector_search`, `hybrid_search`,
`get_recent_documents`) also take `returnDocuments`: results are trimmed with
`TrimRules` and their text shortened unless it is true. It is not sent to the
toolkit. Other tools return full text.

A handler checks the call against the request's `SearchScope` before calling the
toolkit, and turns the toolkit's ``{"error": ...}`` replies into failed calls.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from cosmos_agentic_retriever.agent.result_trimming import TrimRules, trim_json_text
from cosmos_agentic_retriever.agent.tool_contracts import Tool, ToolExecutionResult
from cosmos_agentic_retriever.agent.toolkit_client import ToolkitClient

_DATABASE = {"type": "string", "description": "Database id containing the container"}
_CONTAINER = {"type": "string", "description": "Container id to query"}
_SELECT = {
    "type": "string",
    "description": "Comma-separated list of specific properties to project in results, "
    "e.g. 'id,title,content'. Cannot use '*' wildcard.",
}
_VECTOR_PROPERTY = {
    "type": "string",
    "description": "Property name where vector embeddings are stored, e.g. 'vector' or 'embeddings'",
}
_TOP_N = {
    "type": "integer",
    "description": "Number of documents to return (1-50, default 10)",
    "default": 10,
}
RETURN_DOCUMENTS = {
    "type": "boolean",
    "description": "Return each document's full text instead of shortened text.",
    "default": False,
}


@dataclass(frozen=True)
class ToolkitToolSpec:
    """One toolkit tool as the agent sees it; ``search`` tools shorten text."""

    name: str
    description: str
    properties: dict[str, Any]
    required: tuple[str, ...]
    search: bool = False


TOOLKIT_TOOLS: tuple[ToolkitToolSpec, ...] = (
    ToolkitToolSpec(
        "list_databases",
        "Lists databases available in the Cosmos DB account.",
        {},
        (),
    ),
    ToolkitToolSpec(
        "list_collections",
        "Lists containers (collections) for the specified database.",
        {"databaseId": {"type": "string", "description": "Database id to list containers from"}},
        ("databaseId",),
    ),
    ToolkitToolSpec(
        "get_approximate_schema",
        "Approximates a container schema by sampling up to 10 documents and returning a "
        "union of top-level properties with inferred types and brief descriptions.",
        {
            "databaseId": _DATABASE,
            "containerId": {"type": "string", "description": "Container id to inspect"},
        },
        ("databaseId", "containerId"),
    ),
    ToolkitToolSpec(
        "text_search",
        "Select TOP N documents where a given property contains the provided search "
        "string. N must be between 1 and 20.",
        {
            "databaseId": _DATABASE,
            "containerId": _CONTAINER,
            "property": {
                "type": "string",
                "description": "Document property to search, e.g. name or profile.name",
            },
            "searchPhrase": {
                "type": "string",
                "description": "Search term to look for within the property",
            },
            "n": {
                "type": "integer",
                "description": "Number of documents to return (1-20, default 10)",
                "default": 10,
            },
        },
        ("databaseId", "containerId", "property", "searchPhrase"),
        search=True,
    ),
    ToolkitToolSpec(
        "vector_search",
        "Performs vector search on Cosmos DB using Azure OpenAI embeddings. Searches for "
        "semantically similar documents based on text input.",
        {
            "databaseId": _DATABASE,
            "containerId": _CONTAINER,
            "searchText": {
                "type": "string",
                "description": "Text to search for semantically similar content",
            },
            "vectorProperty": _VECTOR_PROPERTY,
            "selectProperties": _SELECT,
            "topN": _TOP_N,
        },
        ("databaseId", "containerId", "searchText", "vectorProperty", "selectProperties"),
        search=True,
    ),
    ToolkitToolSpec(
        "hybrid_search",
        "Performs hybrid search combining vector similarity and full-text keyword search "
        "using Reciprocal Rank Fusion (RRF). Requires both a vector index and a full-text "
        "index on the container.",
        {
            "databaseId": _DATABASE,
            "containerId": _CONTAINER,
            "searchText": {
                "type": "string",
                "description": "Text to search for using both semantic similarity and keyword matching",
            },
            "textProperty": {
                "type": "string",
                "description": "Property name that has a full-text index for keyword search, "
                "e.g. 'text' or 'content'",
            },
            "vectorProperty": _VECTOR_PROPERTY,
            "selectProperties": _SELECT,
            "topN": _TOP_N,
        },
        (
            "databaseId",
            "containerId",
            "searchText",
            "textProperty",
            "vectorProperty",
            "selectProperties",
        ),
        search=True,
    ),
    ToolkitToolSpec(
        "find_document_by_id",
        "Find a document by its id in the specified database/container.",
        {
            "databaseId": _DATABASE,
            "containerId": _CONTAINER,
            "id": {"type": "string", "description": "The id of the document to find"},
        },
        ("databaseId", "containerId", "id"),
    ),
    ToolkitToolSpec(
        "get_recent_documents",
        "Gets the most recent N documents ordered by timestamp (_ts DESC) from the "
        "specified database/container. N must be between 1 and 20.",
        {
            "databaseId": _DATABASE,
            "containerId": _CONTAINER,
            "n": {"type": "integer", "description": "Number of documents to return (1-20)"},
        },
        ("databaseId", "containerId", "n"),
        search=True,
    ),
)


@dataclass(frozen=True)
class SearchScope:
    """Where the agent may search. ``None`` allows any database or container."""

    database: str | None = None
    containers: frozenset[str] | None = None

    def problem(self, arguments: dict[str, Any]) -> str | None:
        """Why a call is outside this scope, or ``None`` if it is inside."""
        database = arguments.get("databaseId")
        if self.database is not None and database is not None and database != self.database:
            return f"Only database '{self.database}' may be searched for this request."
        container = arguments.get("containerId")
        if self.containers is not None and container is not None and container not in self.containers:
            allowed = ", ".join(f"'{name}'" for name in sorted(self.containers))
            return f"Only these containers may be searched for this request: {allowed}."
        return None


def _toolkit_error(text: str) -> str | None:
    """The message of a toolkit error reply such as {"error": "...", "statusCode": 429}."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    # Only error-shaped objects, so a document with its own "error" field is not mistaken for one.
    if isinstance(parsed, dict) and "error" in parsed and set(parsed) <= {"error", "statusCode", "hint"}:
        status = parsed.get("statusCode")
        return f"{parsed['error']} (status {status})" if status is not None else str(parsed["error"])
    return None


def _description(spec: ToolkitToolSpec, rules: TrimRules) -> str:
    if not spec.search:
        return spec.description
    return (
        f"{spec.description} Text longer than {rules.text_limit} characters is shortened; "
        "set returnDocuments to true for full text, or read one document by its id."
    )


def _parameters(spec: ToolkitToolSpec) -> dict[str, Any]:
    properties = dict(spec.properties)
    if spec.search:
        properties["returnDocuments"] = RETURN_DOCUMENTS
    return {"type": "object", "properties": properties, "required": list(spec.required)}


def make_toolkit_tools(
    client: ToolkitClient,
    *,
    scope: SearchScope,
    rules: TrimRules,
    names: Iterable[str] | None = None,
) -> list[Tool]:
    """Build agent tools for the toolkit tools in ``names`` (default: all of them)."""
    wanted = None if names is None else set(names)
    unknown = (wanted or set()) - {spec.name for spec in TOOLKIT_TOOLS}
    if unknown:
        raise ValueError(f"unknown toolkit tools: {sorted(unknown)}")

    def make(spec: ToolkitToolSpec) -> Tool:
        async def call(arguments: dict[str, Any]) -> ToolExecutionResult:
            forwarded = dict(arguments)
            full_text = forwarded.pop("returnDocuments", False) is True
            problem = scope.problem(forwarded)
            if problem is not None:
                return ToolExecutionResult(content="", error=problem)
            result = await client.call_tool(spec.name, forwarded)
            if result.error is not None:
                return result
            error = _toolkit_error(result.content)
            if error is not None:
                return ToolExecutionResult(content="", error=error)
            shorten = spec.search and not full_text
            return ToolExecutionResult(content=trim_json_text(result.content, rules, shorten_text=shorten))

        async def handler(calls: list[dict[str, Any]]) -> list[ToolExecutionResult]:
            return [await call(arguments) for arguments in calls]

        return Tool(
            name=spec.name,
            description=_description(spec, rules),
            parameters=_parameters(spec),
            handler=handler,
            implementation="mcp-toolkit",
        )

    return [make(spec) for spec in TOOLKIT_TOOLS if wanted is None or spec.name in wanted]
