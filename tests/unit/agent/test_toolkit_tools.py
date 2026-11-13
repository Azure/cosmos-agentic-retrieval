"""Toolkit tools for the agent: names, arguments, scope, trimming and toolkit errors."""

from __future__ import annotations

import json

import pytest

from cosmos_agentic_retriever.agent.result_trimming import TrimRules
from cosmos_agentic_retriever.agent.tool_contracts import ToolCall, ToolExecutionResult, execute_tool_call
from cosmos_agentic_retriever.agent.toolkit_tools import SearchScope, make_toolkit_tools

pytestmark = pytest.mark.anyio

RULES = TrimRules(text_limit=10, min_vector_dim=4)
DOCUMENT = {"id": "d1", "text": "a" * 25, "embedding": [0.1, 0.2, 0.3, 0.4], "_rid": "x"}


class _FakeToolkit:
    """Records calls and answers each with a fixed result."""

    def __init__(self, result: ToolExecutionResult | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.result = result or ToolExecutionResult(content=json.dumps([DOCUMENT]))

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self.result


def _tools(toolkit, scope=SearchScope(), **options):
    return {tool.name: tool for tool in make_toolkit_tools(toolkit, scope=scope, rules=RULES, **options)}


def test_offers_the_toolkit_tools_with_their_names():
    tools = _tools(_FakeToolkit())

    assert sorted(tools) == [
        "find_document_by_id",
        "get_approximate_schema",
        "get_recent_documents",
        "hybrid_search",
        "list_collections",
        "list_databases",
        "text_search",
        "vector_search",
    ]
    text_search = tools["text_search"]
    assert text_search.parameters["required"] == ["databaseId", "containerId", "property", "searchPhrase"]
    assert "returnDocuments" in text_search.parameters["properties"]
    assert "10 characters" in text_search.description
    assert "returnDocuments" not in tools["find_document_by_id"].parameters["properties"]
    assert tools["find_document_by_id"].description.startswith("Find a document by its id")


def test_names_choose_which_tools_are_offered():
    assert sorted(_tools(_FakeToolkit(), names=["text_search", "list_databases"])) == [
        "list_databases",
        "text_search",
    ]
    with pytest.raises(ValueError, match="unknown"):
        _tools(_FakeToolkit(), names=["delete_everything"])


async def test_search_forwards_arguments_and_shortens_text():
    toolkit = _FakeToolkit()
    arguments = {"databaseId": "db", "containerId": "c", "property": "text", "searchPhrase": "x"}

    result = await _tools(toolkit)["text_search"](**arguments)

    assert toolkit.calls == [("text_search", arguments)]
    assert json.loads(result.content) == [
        {"id": "d1", "text": "aaaaaaaaaa… [15 more characters]", "embedding": "[4-dimension vector omitted]"}
    ]


async def test_return_documents_gives_full_text_and_is_not_forwarded():
    toolkit = _FakeToolkit()

    result = await _tools(toolkit)["text_search"](databaseId="db", containerId="c", returnDocuments=True)

    assert toolkit.calls == [("text_search", {"databaseId": "db", "containerId": "c"})]
    assert json.loads(result.content)[0]["text"] == "a" * 25


async def test_reading_a_document_keeps_full_text():
    toolkit = _FakeToolkit(ToolExecutionResult(content=json.dumps(DOCUMENT)))

    result = await _tools(toolkit)["find_document_by_id"](databaseId="db", containerId="c", id="d1")

    assert json.loads(result.content) == {
        "id": "d1",
        "text": "a" * 25,
        "embedding": "[4-dimension vector omitted]",
    }


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"databaseId": "other", "containerId": "c"}, "Only database 'db'"),
        ({"databaseId": "db", "containerId": "secret"}, "'a', 'c'"),
    ],
)
async def test_calls_outside_the_scope_are_refused(arguments, message):
    toolkit = _FakeToolkit()
    scope = SearchScope(database="db", containers=frozenset({"c", "a"}))

    result = await _tools(toolkit, scope)["text_search"](**arguments)

    assert toolkit.calls == []
    assert message in (result.error or "")


async def test_calls_inside_the_scope_reach_the_toolkit():
    toolkit = _FakeToolkit()
    scope = SearchScope(database="db", containers=frozenset({"c"}))

    result = await _tools(toolkit, scope)["get_approximate_schema"](databaseId="db", containerId="c")

    assert result.error is None and len(toolkit.calls) == 1


@pytest.mark.parametrize(
    "reply, error",
    [
        ({"error": "Request rate is large", "statusCode": 429}, "Request rate is large (status 429)"),
        ({"error": "Missing required environment variable COSMOS_ENDPOINT."}, "Missing required"),
    ],
)
async def test_toolkit_error_replies_become_failed_calls(reply, error):
    toolkit = _FakeToolkit(ToolExecutionResult(content=json.dumps(reply)))

    response = await execute_tool_call(
        _tools(toolkit), ToolCall("c1", "list_collections", {"databaseId": "db"})
    )

    assert response.content == ""
    assert error in (response.error or "")


async def test_a_document_with_an_error_field_is_not_a_failure():
    document = {"id": "d1", "error": "printer jam", "text": "short"}
    toolkit = _FakeToolkit(ToolExecutionResult(content=json.dumps(document)))

    result = await _tools(toolkit)["find_document_by_id"](databaseId="db", containerId="c", id="d1")

    assert result.error is None
    assert json.loads(result.content) == document


async def test_client_failures_pass_through():
    toolkit = _FakeToolkit(ToolExecutionResult(content="", error="The toolkit call failed: ConnectionError"))

    result = await _tools(toolkit)["list_databases"]()

    assert result.error == "The toolkit call failed: ConnectionError"
