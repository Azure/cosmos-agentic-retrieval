"""Trim toolkit results before the agent reads them.

The toolkit's tools return whole Cosmos DB documents as JSON, including stored
embeddings, Cosmos DB system fields, and full text. That costs the agent's model
tokens without helping it answer. `trim_tool` wraps a `Tool` so its JSON results
are trimmed:

- A list of at least `min_vector_dim` numbers is an embedding; it is replaced by
  a short marker, so the agent still sees which field holds the vector.
- Fields named in `dropped_fields` are removed wherever they appear. By default
  these are Cosmos DB's internal system fields.
- For search tools, text longer than `text_limit` characters is shortened, unless
  the call sets `return_documents` to true. Other tools, such as reading one
  document by ID, return full text.

Results that are not JSON, and failed calls, pass through unchanged.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from cosmos_agentic_retriever.agent.tool_contracts import Tool, ToolExecutionResult

DEFAULT_DROPPED_FIELDS = ("_rid", "_self", "_etag", "_attachments")

# Toolkit tools that return search results; only these shorten text.
SEARCH_TOOLS = frozenset({"text_search", "vector_search", "hybrid_search", "get_recent_documents"})

_RETURN_DOCUMENTS = {
    "type": "boolean",
    "description": "Return each document's full text instead of shortened text.",
    "default": False,
}


@dataclass(frozen=True)
class TrimRules:
    """How tool results are trimmed; the defaults match the service's settings."""

    text_limit: int = 512
    min_vector_dim: int = 128
    dropped_fields: frozenset[str] = field(default_factory=lambda: frozenset(DEFAULT_DROPPED_FIELDS))


def _is_vector(value: list[Any], rules: TrimRules) -> bool:
    return len(value) >= rules.min_vector_dim and all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
    )


def trim_value(value: Any, rules: TrimRules, *, shorten_text: bool) -> Any:
    """Return a trimmed copy of a parsed JSON value."""
    if isinstance(value, dict):
        return {
            key: trim_value(item, rules, shorten_text=shorten_text)
            for key, item in value.items()
            if key not in rules.dropped_fields
        }
    if isinstance(value, list):
        if _is_vector(value, rules):
            return f"[{len(value)}-dimension vector omitted]"
        return [trim_value(item, rules, shorten_text=shorten_text) for item in value]
    if shorten_text and isinstance(value, str) and len(value) > rules.text_limit:
        hidden = len(value) - rules.text_limit
        return f"{value[: rules.text_limit]}… [{hidden} more characters]"
    return value


def _trim_text(text: str, rules: TrimRules, *, shorten_text: bool) -> str:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    trimmed = trim_value(parsed, rules, shorten_text=shorten_text)
    return json.dumps(trimmed, ensure_ascii=False, separators=(",", ":"))


def _trim_result(
    result: str | ToolExecutionResult, rules: TrimRules, *, shorten_text: bool
) -> str | ToolExecutionResult:
    if isinstance(result, str):
        return _trim_text(result, rules, shorten_text=shorten_text)
    if result.error is not None:
        return result
    content = _trim_text(result.content, rules, shorten_text=shorten_text)
    return ToolExecutionResult(content=content, metrics=dict(result.metrics))


def trim_tool(tool: Tool, rules: TrimRules) -> Tool:
    """Wrap ``tool`` so its results are trimmed; search tools gain `return_documents`."""
    is_search = tool.name in SEARCH_TOOLS
    parameters = copy.deepcopy(tool.parameters)
    description = tool.description
    if is_search:
        parameters.setdefault("properties", {})["return_documents"] = _RETURN_DOCUMENTS
        description = (
            f"{description} Text longer than {rules.text_limit} characters is shortened; "
            "set return_documents to true for full text, or read one document by its ID."
        ).strip()

    async def handler(calls: list[dict[str, Any]]) -> Sequence[str | ToolExecutionResult]:
        forwarded = [dict(arguments) for arguments in calls]
        full_text = [arguments.pop("return_documents", False) is True for arguments in forwarded]
        results = await tool.handler(forwarded)
        return [
            _trim_result(result, rules, shorten_text=is_search and not full)
            for result, full in zip(results, full_text, strict=True)
        ]

    return Tool(
        name=tool.name,
        description=description,
        parameters=parameters,
        handler=handler,
        implementation=tool.implementation,
    )
