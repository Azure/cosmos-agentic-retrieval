"""Trimming toolkit results: vectors, system fields, long text, and the wrapper."""

from __future__ import annotations

import json

import pytest

from cosmos_agentic_retriever.agent.result_trimming import TrimRules, trim_tool, trim_value
from cosmos_agentic_retriever.agent.tool_contracts import Tool, ToolExecutionResult

pytestmark = pytest.mark.anyio

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


def _recording_tool(name: str, reply: str | ToolExecutionResult) -> tuple[Tool, list]:
    seen: list[list[dict]] = []

    async def handler(calls):
        seen.append(calls)
        return [reply for _ in calls]

    tool = Tool(
        name=name,
        description="Search documents.",
        parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        handler=handler,
        implementation="mcp",
    )
    return tool, seen


async def test_search_tool_shortens_text_and_offers_return_documents():
    tool, seen = _recording_tool("text_search", json.dumps([_document()]))

    trimmed = trim_tool(tool, RULES)
    result = await trimmed(q="x")

    assert trimmed.name == "text_search" and trimmed.implementation == "mcp"
    assert "return_documents" in trimmed.parameters["properties"]
    assert "return_documents" not in tool.parameters["properties"]
    assert "10 characters" in trimmed.description
    assert seen == [[{"q": "x"}]]
    [document] = json.loads(result.content)
    assert document["text"] == "aaaaaaaaaa… [15 more characters]"
    assert "_rid" not in document


async def test_return_documents_gives_full_text_and_is_not_forwarded():
    tool, seen = _recording_tool("text_search", json.dumps([_document()]))

    result = await trim_tool(tool, RULES)(q="x", return_documents=True)

    assert seen == [[{"q": "x"}]]
    [document] = json.loads(result.content)
    assert document["text"] == "a" * 25
    assert document["embedding"] == "[5-dimension vector omitted]"


async def test_read_by_id_keeps_full_text_without_a_flag():
    tool, _ = _recording_tool("find_document_by_id", json.dumps(_document()))

    trimmed = trim_tool(tool, RULES)
    result = await trimmed(q="d1")

    assert "return_documents" not in trimmed.parameters["properties"]
    assert trimmed.description == "Search documents."
    document = json.loads(result.content)
    assert document["text"] == "a" * 25 and "_etag" not in document


async def test_batch_keeps_each_calls_own_setting():
    tool, seen = _recording_tool("text_search", json.dumps({"text": "b" * 20}))

    results = await trim_tool(tool, RULES).handler(
        [{"q": "1"}, {"q": "2", "return_documents": True}]
    )

    assert seen == [[{"q": "1"}, {"q": "2"}]]
    assert [json.loads(result)["text"] for result in results] == [
        "bbbbbbbbbb… [10 more characters]",
        "b" * 20,
    ]


@pytest.mark.parametrize(
    "reply",
    ["not json at all", ToolExecutionResult(content="", error="Cosmos is down")],
)
async def test_non_json_and_failed_results_pass_through(reply):
    tool, _ = _recording_tool("text_search", reply)

    result = await trim_tool(tool, RULES)(q="x")

    if isinstance(reply, str):
        assert result.content == reply
    else:
        assert result.error == "Cosmos is down"
