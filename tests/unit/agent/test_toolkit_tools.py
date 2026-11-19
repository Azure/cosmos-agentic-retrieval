"""Toolkit tools for the agent: names, arguments, scope, trimming and toolkit errors."""

from __future__ import annotations

import json

import pytest

from cosmos_agentic_retriever.agent.result_trimming import TrimRules
from cosmos_agentic_retriever.agent.tool_contracts import ToolCall, ToolExecutionResult, execute_tool_call
from cosmos_agentic_retriever.agent.toolkit_tools import SearchEvidence, SearchScope, make_toolkit_tools

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


async def test_evidence_returns_references_without_document_text():
    evidence = SearchEvidence()
    toolkit = _FakeToolkit()

    result = await _tools(toolkit, evidence=evidence)["text_search"](
        databaseId="db", containerId="c", property="text", searchPhrase="x"
    )

    assert "more characters" in result.content
    assert list(json.loads(result.content)[0]) == ["cite_as", "id", "text", "embedding"]
    answer, cited = evidence.verify_citations("It says a [D1].")
    assert answer == "It says a [D1]."
    assert evidence.documents(cited, limit=10, include_uncited=False) == [
        {"cite_as": "D1", "database": "db", "container": "c", "id": "d1", "used_in_answer": True},
    ]
    assert evidence.searched == [{"database": "db", "container": "c"}]


async def test_labels_reach_the_agent_and_stay_the_same_across_calls():
    evidence = SearchEvidence()
    search = _FakeToolkit(ToolExecutionResult(content=json.dumps([{"id": "a"}, {"id": "b", "cite_as": "theirs"}])))
    read = _FakeToolkit(ToolExecutionResult(content=json.dumps({"id": "b", "text": "full"})))

    searched = await _tools(search, evidence=evidence)["text_search"](databaseId="db", containerId="c")
    reread = await _tools(read, evidence=evidence)["find_document_by_id"](databaseId="db", containerId="c", id="b")

    assert json.loads(searched.content) == [{"cite_as": "D1", "id": "a"}, {"cite_as": "D2", "id": "b"}]
    assert json.loads(reread.content) == {"cite_as": "D2", "id": "b", "text": "full"}


def test_citations_keep_answer_order_and_mark_made_up_ones():
    evidence = SearchEvidence()
    evidence.record("text_search", {"databaseId": "db", "containerId": "c"},
                    ToolExecutionResult(content='[{"id":"a"},{"id":"b"}]'))

    answer, cited = evidence.verify_citations(
        "B [D2]. A [D1], again [D2]. C [D9] and [db/c/a]."
    )

    assert answer == "B [D2]. A [D1], again [D2]. C [unverified source] and [db/c/a]."
    assert cited == [("db", "c", "b"), ("db", "c", "a")]
    assert [item["id"] for item in evidence.documents(cited, limit=1, include_uncited=False)] == ["b"]


def test_evidence_deduplicates_by_source_merges_ranks_and_puts_citations_first():
    evidence = SearchEvidence()
    arguments = {"databaseId": "db", "containerId": "c"}
    evidence.record("text_search", arguments, ToolExecutionResult(content=json.dumps([
        {"id": "first"}, {"id": "repeated"}, {"id": "cited"}
    ])))
    evidence.record("hybrid_search", arguments, ToolExecutionResult(
        content='[{"id":"repeated"}]'
    ))
    evidence.record("find_document_by_id", {**arguments, "containerId": "other"},
                    ToolExecutionResult(content='{"id":"first","text":"read"}'))

    ranked = evidence.documents([], limit=2, include_uncited=True)
    assert [item["id"] for item in ranked] == ["repeated", "first"]
    _, cited = evidence.verify_citations("x [D3]")
    assert evidence.documents(cited, limit=10, include_uncited=False) == [
        {"cite_as": "D3", "database": "db", "container": "c", "id": "cited", "used_in_answer": True},
    ]
    documents = evidence.documents(cited, limit=10, include_uncited=True)
    assert [item["used_in_answer"] for item in documents] == [True, False, False, False]
    assert documents[-1]["container"] == "other"


async def test_evidence_records_toolkit_errors_and_does_not_invent_documents():
    evidence = SearchEvidence()
    toolkit = _FakeToolkit(ToolExecutionResult(content='{"error":"busy","statusCode":429}'))

    await _tools(toolkit, evidence=evidence)["text_search"](databaseId="db", containerId="c")

    assert evidence.document_calls == 1
    assert evidence.searched == []
    assert evidence.verify_citations("x [D1]") == ("x [unverified source]", [])
    assert evidence.documents([], limit=5, include_uncited=True) == []
    assert evidence.errors == [{
        "tool": "text_search", "database": "db", "container": "c",
        "error": "busy (status 429)",
    }]


def test_empty_search_is_successful_and_discovery_is_not_evidence():
    evidence = SearchEvidence()
    arguments = {"databaseId": "db", "containerId": "c"}
    evidence.record("get_approximate_schema", arguments,
                    ToolExecutionResult(content='{"id":"not-a-document"}'))
    evidence.record("text_search", arguments, ToolExecutionResult(content="[]"))

    assert evidence.document_calls == 1
    assert evidence.searched == [{"database": "db", "container": "c"}]
    assert evidence.documents([], limit=5, include_uncited=True) == []
