"""Keep toolkit results from unnecessarily filling the search model's context.

The toolkit serves general-purpose clients and can return entire Cosmos documents,
including large embedding arrays and system fields. Feeding every field into the
search model spends tokens and leaves less room for useful evidence. This module
adapts those results without changing the toolkit or duplicating its search logic.

ToolkitToolAdapter uses short previews for searches; the agent can request full
text with returnDocuments or read by ID. SearchEvidence applies the same cleanup
without shortening text, so the caller can still receive full supporting text.
Vector markers preserve field names for later vector searches. These are output
formatting rules, not access controls; the toolkit still returns the original data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

DEFAULT_DROPPED_FIELDS = ("_rid", "_self", "_etag", "_attachments")


@dataclass(frozen=True)
class TrimRules:
    """Configurable output policy rather than assumptions about one container's schema.

    min_vector_dim is a heuristic: long numeric arrays may be embeddings, while
    short arrays may be useful coordinates or measurements. Operators can adjust
    the threshold and dropped field names to fit their data.
    """

    text_limit: int = 512
    min_vector_dim: int = 128
    dropped_fields: frozenset[str] = field(default_factory=lambda: frozenset(DEFAULT_DROPPED_FIELDS))


class ResultTrimmer:
    """Apply one policy consistently to search previews, full reads and evidence.

    Results are copied, not edited in place, so shortening what the model sees
    cannot accidentally shorten the evidence kept for the caller.
    """

    def __init__(self, rules: TrimRules) -> None:
        self._rules = rules

    def trim_value(self, value: Any, *, shorten_text: bool) -> Any:
        """Clean nested fields without requiring a container-specific field map.

        shorten_text=False retains full text for reads and evidence; configured
        field removal and vector replacement still apply.
        """
        if isinstance(value, dict):
            return {
                key: self.trim_value(item, shorten_text=shorten_text)
                for key, item in value.items()
                if key not in self._rules.dropped_fields
            }
        if isinstance(value, list):
            if self._is_vector(value):
                return f"[{len(value)}-dimension vector omitted]"
            return [self.trim_value(item, shorten_text=shorten_text) for item in value]
        if shorten_text and isinstance(value, str) and len(value) > self._rules.text_limit:
            hidden = len(value) - self._rules.text_limit
            return f"{value[: self._rules.text_limit]}… [{hidden} more characters]"
        return value

    def trim_json_text(self, text: str, *, shorten_text: bool) -> str:
        """Adapt the toolkit's JSON-string replies to the agent's text observations.

        Non-JSON replies pass through so a diagnostic is not lost or disguised
        as an empty search result. Compact JSON avoids unnecessary whitespace.
        """
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return text
        trimmed = self.trim_value(parsed, shorten_text=shorten_text)
        return json.dumps(trimmed, ensure_ascii=False, separators=(",", ":"))

    def _is_vector(self, value: list[Any]) -> bool:
        """Use a size/type heuristic because schema discovery does not label embeddings."""
        return len(value) >= self._rules.min_vector_dim and all(
            isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
        )
