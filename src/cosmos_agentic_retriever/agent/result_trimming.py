"""Trim toolkit results before the agent reads them.

The toolkit's tools return whole Cosmos DB documents as JSON, including stored
embeddings, Cosmos DB system fields, and full text. That costs the agent's model
tokens without helping it answer. `trim_json_text` trims a JSON result:

- A list of at least `min_vector_dim` numbers is an embedding; it is replaced by
  a short marker, so the agent still sees which field holds the vector.
- Fields named in `dropped_fields` are removed wherever they appear. By default
  these are Cosmos DB's internal system fields.
- With ``shorten_text``, text longer than `text_limit` characters is shortened.

Text that is not JSON is returned unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

DEFAULT_DROPPED_FIELDS = ("_rid", "_self", "_etag", "_attachments")


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


def trim_json_text(text: str, rules: TrimRules, *, shorten_text: bool) -> str:
    """Trim a JSON result and return it as compact JSON; other text is unchanged."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    trimmed = trim_value(parsed, rules, shorten_text=shorten_text)
    return json.dumps(trimmed, ensure_ascii=False, separators=(",", ":"))
