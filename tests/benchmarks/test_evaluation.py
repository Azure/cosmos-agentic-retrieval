"""Offline checks for scoring toolkit-backed search responses by document ID."""

from unittest.mock import Mock

import pytest

pytest.importorskip("huggingface_hub")
pytest.importorskip("pyarrow")
pytest.importorskip("tiktoken")

import dataset_enterprise_rag_bench as erb
from dataset_contracts import Document
from dataset_hipporag2 import HippoRAG2Collection, HippoRAGQuestion
from evaluation_contracts import AgentResponse
from evaluator_enterprise_rag_bench import ERBEvaluator
from evaluator_hipporag2 import HippoRAGEvaluator


@pytest.mark.parametrize(
    "body, expected",
    [
        ({}, AgentResponse("", [], [])),
        (
            {
                "answer": "Old format",
                "documents": [{"database": "db", "container": "corpus", "id": "p1", "document": {}}],
            },
            AgentResponse("Old format", ["p1"], None),
        ),
        (
            {
                "answer": "An answer [db/corpus/passage-1], not [unverified source].",
                "documents": [
                    {
                        "database": "db",
                        "container": "corpus",
                        "id": "passage-1",
                        "used_in_answer": True,
                    },
                    {
                        "database": "db",
                        "container": "corpus",
                        "id": "passage-2",
                        "used_in_answer": False,
                    },
                ],
            },
            AgentResponse("An answer, not.", ["passage-1", "passage-2"], ["passage-1"]),
        ),
    ],
)
def test_response_reads_item_ids_and_removes_citations_before_judging(body, expected):
    assert AgentResponse.from_response_body(body) == expected


@pytest.mark.parametrize("document_id, recall", [("known", 1.0), ("unknown", 0.0)])
def test_hipporag_recall_resolves_titles_from_the_dataset(monkeypatch, document_id, recall):
    monkeypatch.setattr(
        HippoRAG2Collection,
        "read_merged_corpus_documents",
        lambda self: [Document("known", "Gold title", "Evidence", ("test",))],
    )
    judge = Mock()
    evaluator = HippoRAGEvaluator(judge)
    question = HippoRAGQuestion(
        id="q1",
        dataset="test",
        question="Question?",
        answers=("Answer",),
        gold_titles=("Gold title",),
    )

    result = evaluator.evaluate(question, AgentResponse("", [document_id]))

    assert result.scores["gold_title_recall"] == recall
    judge.ask.assert_not_called()


@pytest.mark.parametrize("stored_id", ["doc-with-hyphens", "doc-with-hyphens-copy2"])
def test_erb_passage_ids_round_trip_and_count_each_document_once(monkeypatch, stored_id):
    monkeypatch.setattr(erb, "split_text_into_chunks", lambda *args: ["First", "Second"])
    document = erb.ERBDocument(
        id=stored_id,
        title="Title",
        text="First Second",
        datasets=("enterprise-rag-bench",),
        doc_id="doc-with-hyphens",
        source_type="slack",
    )
    passages = document.get_passages()

    assert all(erb.ERBDocument.read_doc_id(passage.id) == passage.doc_id for passage in passages)
    response = AgentResponse("", [passage.id for passage in passages])
    assert ERBEvaluator(Mock()).read_returned_doc_ids(response) == {document.doc_id}