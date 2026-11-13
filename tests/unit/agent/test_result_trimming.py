"""Trimming toolkit results: vectors, system fields and long text."""

from __future__ import annotations

import json

from cosmos_agentic_retriever.agent.result_trimming import TrimRules, trim_json_text, trim_value

RULES = TrimRules(text_limit=10, min_vector_dim=4)


def _document(text: str = "a" * 25) -> dict:
    return {
        "id": "d1",
        "title": "Short",
        "text": text,
        "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
        "point": [1.5, 2.5],
        "_rid": "x",
        "_etag": "y",
        "_ts": 1700000000,
        "meta": {"_self": "z", "tags": ["a", "b"]},
    }


def test_trim_value_drops_vectors_system_fields_and_shortens_text():
    trimmed = trim_value(_document(), RULES, shorten_text=True)

    assert trimmed == {
        "id": "d1",
        "title": "Short",
        "text": "aaaaaaaaaa… [15 more characters]",
        "embedding": "[5-dimension vector omitted]",
        "point": [1.5, 2.5],
        "_ts": 1700000000,
        "meta": {"tags": ["a", "b"]},
    }


def test_trim_value_keeps_full_text_when_not_shortening():
    trimmed = trim_value(_document(), RULES, shorten_text=False)

    assert trimmed["text"] == "a" * 25
    assert trimmed["embedding"] == "[5-dimension vector omitted]"


def test_lists_that_are_not_all_numbers_are_kept():
    value = [1, 2, 3, "four", 5]
    assert trim_value(value, RULES, shorten_text=True) == value
    assert trim_value([True] * 5, RULES, shorten_text=True) == [True] * 5


def test_dropped_fields_can_be_changed():
    rules = TrimRules(text_limit=10, min_vector_dim=4, dropped_fields=frozenset({"title"}))

    trimmed = trim_value(_document(), rules, shorten_text=False)

    assert "title" not in trimmed and trimmed["_rid"] == "x"


def test_trim_json_text_returns_compact_json():
    text = trim_json_text(json.dumps([_document()]), RULES, shorten_text=True)

    assert json.loads(text)[0]["text"] == "aaaaaaaaaa… [15 more characters]"
    assert ", " not in text


def test_trim_json_text_leaves_other_text_unchanged():
    assert trim_json_text("not json at all", RULES, shorten_text=True) == "not json at all"
