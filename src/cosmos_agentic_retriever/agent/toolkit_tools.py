"""The toolkit tools the search agent can use, as agent `Tool`s.

Each definition keeps the toolkit tool's name, description and arguments, so the
agent's tools match the toolkit. They are written out here rather than read from
the toolkit at run time, so the agent's tools stay the same until we change them;
`tests/live/test_toolkit_tools_live.py` checks them against a running toolkit.

Search tools (`text_search`, `vector_search`, `hybrid_search`,
`get_recent_documents`) also take `returnDocuments`: results are trimmed with
`TrimRules` and their text shortened unless it is true. It is not sent to the
toolkit. Other tools return full text.

This is the per-question policy layer, not another query engine. ToolkitClient
owns transport; these handlers enforce request scope, record which documents
were retrieved, and recognize toolkit error envelopes. The agent loop only receives
Tool objects and never needs to know that their handlers call MCP.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from cosmos_agentic_retriever.agent.result_trimming import ResultTrimmer, TrimRules
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
    """Keep the agent's tool contract explicit and independent of a live connection.

    The toolkit's names and arguments stay intact. The search flag identifies
    tools that can return previews and need our local returnDocuments option.
    """

    name: str
    description: str
    properties: dict[str, Any]
    required: tuple[str, ...]
    search: bool = False

    def agent_description(self, rules: TrimRules) -> str:
        """Tell the model how to recover full text when previewing is enabled."""
        if not self.search:
            return self.description
        return (
            f"{self.description} Text longer than {rules.text_limit} characters is shortened; "
            "set returnDocuments to true for full text, or read one document by its id."
        )

    def parameters(self) -> dict[str, Any]:
        """Expose returnDocuments to the agent without adding it to the toolkit contract."""
        properties = dict(self.properties)
        if self.search:
            properties["returnDocuments"] = RETURN_DOCUMENTS
        return {"type": "object", "properties": properties, "required": list(self.required)}


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
    """Enforce the caller's search boundary even when the model chooses bad arguments.

    Prompt instructions alone are not enforcement. None leaves that dimension
    unrestricted; the toolkit's own identity and authorization still govern access.
    """

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


@dataclass
class SearchEvidence:
    """Track which documents the toolkit actually returned for one question.

    Each retrieved document gets a short label, such as D3, the first time a
    result contains it. The agent cites documents by label, because copying long
    ids is error-prone, and the caller gets the same label with each reference.
    The answer's citations are checked against these labels, so the caller only
    gets references to documents that were retrieved. Failures stay visible to
    the caller. Schema and listing results are not supporting documents.
    """

    CITE_AS: ClassVar[str] = "cite_as"
    CITATION: ClassVar[re.Pattern[str]] = re.compile(r"\[(D\d+)\]")
    UNVERIFIED: ClassVar[str] = "[unverified source]"

    errors: list[dict[str, Any]] = field(default_factory=list)
    document_calls: int = 0
    _scores: dict[tuple[str, str, str], float] = field(default_factory=dict)
    _labels: dict[tuple[str, str, str], str] = field(default_factory=dict)
    _searched: dict[tuple[str, str], None] = field(default_factory=dict)

    @property
    def searched(self) -> list[dict[str, str]]:
        return [
            {"database": database, "container": container}
            for database, container in self._searched
        ]

    def record(
        self, name: str, arguments: dict[str, Any], result: ToolExecutionResult
    ) -> None:
        """Capture actual tool results so model-written citations cannot invent evidence."""
        document_tool = name == "find_document_by_id" or any(
            spec.name == name and spec.search for spec in TOOLKIT_TOOLS
        )
        self.document_calls += int(document_tool)
        target = {
            field_name: arguments[argument]
            for field_name, argument in (("database", "databaseId"), ("container", "containerId"))
            if argument in arguments
        }
        if result.error is not None:
            self.errors.append({"tool": name, **target, "error": result.error})
            return
        if not document_tool or set(target) != {"database", "container"}:
            return
        database, container = target["database"], target["container"]
        self._searched[(database, container)] = None
        try:
            parsed = json.loads(result.content)
        except json.JSONDecodeError:
            return
        items = parsed if isinstance(parsed, list) else [parsed]
        seen: set[str] = set()
        for rank, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            item_id = item.get("id")
            if not isinstance(item_id, str) or not item_id or item_id in seen:
                continue
            seen.add(item_id)
            key = (database, container, item_id)
            self._scores[key] = self._scores.get(key, 0.0) + 1.0 / (60 + rank)
            self._labels.setdefault(key, f"D{len(self._labels) + 1}")

    def label_results(self, arguments: dict[str, Any], content: str) -> str:
        """Put each returned document's label first, so the agent can cite it."""
        database, container = arguments.get("databaseId"), arguments.get("containerId")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return content

        def label(item: Any) -> Any:
            item_id = item.get("id") if isinstance(item, dict) else None
            key = (database, container, item_id) if isinstance(item_id, str) else None
            if key not in self._labels:
                return item
            # Ours replaces a document field of the same name, so the agent sees one label.
            fields = {name: value for name, value in item.items() if name != self.CITE_AS}
            return {self.CITE_AS: self._labels[key], **fields}

        labelled = [label(item) for item in parsed] if isinstance(parsed, list) else label(parsed)
        return json.dumps(labelled, ensure_ascii=False, separators=(",", ":"))

    def verify_citations(self, answer: str) -> tuple[str, list[tuple[str, str, str]]]:
        """Return the answer with made-up citations marked, and the real ones in order.

        A citation is real only if a toolkit call returned the labelled document.
        """
        keys = {label: key for key, label in self._labels.items()}
        cited: dict[tuple[str, str, str], None] = {}

        def check(match: re.Match[str]) -> str:
            key = keys.get(match[1])
            if key is None:
                return self.UNVERIFIED
            cited[key] = None
            return match[0]

        return self.CITATION.sub(check, answer), list(cited)

    def documents(
        self, cited: list[tuple[str, str, str]], *, limit: int, include_uncited: bool
    ) -> list[dict[str, Any]]:
        """References to the cited documents, then optionally the best-ranked uncited ones.

        Reciprocal rank uses positions because scores from different searches
        are not directly comparable.
        """
        used = set(cited)
        keys = list(cited)
        if include_uncited:
            keys += sorted(
                (key for key in self._scores if key not in used),
                key=lambda key: -self._scores[key],
            )
        return [
            {self.CITE_AS: self._labels[key], "database": key[0], "container": key[1],
             "id": key[2], "used_in_answer": key in used}
            for key in keys[:limit]
        ]


class ToolkitToolAdapter:
    """Bind one toolkit tool to one question's output policy and search boundary.

    The client can be shared, but scope and evidence must not leak between
    questions. A bound handler keeps that state out of the transport and loop.
    """

    def __init__(
        self,
        client: ToolkitClient,
        spec: ToolkitToolSpec,
        *,
        scope: SearchScope,
        rules: TrimRules,
        evidence: SearchEvidence | None = None,
    ) -> None:
        self._client = client
        self._spec = spec
        self._scope = scope
        self._rules = rules
        self._trimmer = ResultTrimmer(rules)
        self._evidence = evidence

    def as_tool(self) -> Tool:
        """Present the generic Tool interface while keeping MCP behind the handler."""
        return Tool(
            name=self._spec.name,
            description=self._spec.agent_description(self._rules),
            parameters=self._spec.parameters(),
            handler=self.handle,
            implementation="mcp-toolkit",
        )

    async def handle(self, calls: list[dict[str, Any]]) -> list[ToolExecutionResult]:
        return [await self._call(arguments) for arguments in calls]

    async def _call(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        """Check scope before network access and keep evidence before building a preview."""
        forwarded = dict(arguments)
        full_text = forwarded.pop("returnDocuments", False) is True
        problem = self._scope.problem(forwarded)
        if problem is not None:
            result = ToolExecutionResult(content="", error=problem)
        else:
            result = await self._client.call_tool(self._spec.name, forwarded)
            if result.error is None:
                error = self._toolkit_error(result.content)
                if error is not None:
                    result = replace(result, content="", error=error)
        if self._evidence is not None:
            self._evidence.record(self._spec.name, forwarded, result)
        if result.error is not None:
            return result
        content = self._trimmer.trim_json_text(
            result.content, shorten_text=self._spec.search and not full_text
        )
        if self._evidence is not None:
            content = self._evidence.label_results(forwarded, content)
        return replace(result, content=content)

    @staticmethod
    def _toolkit_error(text: str) -> str | None:
        """Promote toolkit failures hidden inside otherwise successful MCP text replies.

        Restrict recognition to error-shaped envelopes: user documents can
        legitimately contain a field named error without describing a failed call.
        """
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict) and "error" in parsed and set(parsed) <= {"error", "statusCode", "hint"}:
            status = parsed.get("statusCode")
            return f"{parsed['error']} (status {status})" if status is not None else str(parsed["error"])
        return None


def make_toolkit_tools(
    client: ToolkitClient,
    *,
    scope: SearchScope,
    rules: TrimRules,
    names: Iterable[str] | None = None,
    evidence: SearchEvidence | None = None,
) -> list[Tool]:
    """Build fresh bound handlers per question, reusing only the transport client.

    names limits the offered tools, for example when a caller has already
    supplied a scope and database discovery is unnecessary. None offers all tools.
    """
    wanted = None if names is None else set(names)
    unknown = (wanted or set()) - {spec.name for spec in TOOLKIT_TOOLS}
    if unknown:
        raise ValueError(f"unknown toolkit tools: {sorted(unknown)}")
    return [
        ToolkitToolAdapter(client, spec, scope=scope, rules=rules, evidence=evidence).as_tool()
        for spec in TOOLKIT_TOOLS
        if wanted is None or spec.name in wanted
    ]
