"""Offline checks for reading benchmark datasets and scoring toolkit-backed search responses by document ID."""

import base64
import hashlib
from dataclasses import asdict
from unittest.mock import Mock

import pytest

pytest.importorskip("huggingface_hub")
pytest.importorskip("pyarrow")
pytest.importorskip("tiktoken")

import dataset_contracts
import dataset_enterprise_rag_bench as erb
from dataset_browsecomp_plus import BrowseCompPlus, BrowseCompPlusDocument
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
                "answer": "An answer [D1], not [unverified source].",
                "documents": [
                    {
                        "cite_as": "D1",
                        "database": "db",
                        "container": "corpus",
                        "id": "passage-1",
                        "used_in_answer": True,
                    },
                    {
                        "cite_as": "D2",
                        "database": "db",
                        "container": "corpus",
                        "id": "passage-2",
                        "used_in_answer": False,
                    },
                ],
            },
            AgentResponse("An answer, not.", ["passage-1", "passage-2"], ["passage-1"]),
        ),
        (
            {
                "answer": "An older answer [db/corpus/passage-1].",
                "documents": [
                    {"database": "db", "container": "corpus", "id": "passage-1", "used_in_answer": True},
                ],
            },
            AgentResponse("An older answer.", ["passage-1"], ["passage-1"]),
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
    monkeypatch.setattr(dataset_contracts, "split_text_into_chunks", lambda *args: ["First", "Second"])
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
    assert ERBEvaluator(Mock()).read_doc_ids([passage.id for passage in passages]) == {document.doc_id}
    # The loaded enterprise-rag-bench-c512-o128 container holds items with exactly these fields.
    assert asdict(passages[1]) == {
        "id": f"{stored_id}-1",
        "title": "Title",
        "text": "Second",
        "datasets": ("enterprise-rag-bench",),
        "doc_id": "doc-with-hyphens",
        "chunk_idx": 1,
        "source_type": "slack",
    }


@pytest.mark.parametrize("cited, recall", [(["gold-1", "gold-0"], 0.5), (["other-0"], 0.0), (None, None)])
def test_erb_cited_recall_counts_only_the_cited_chunks_documents(cited, recall):
    question = erb.ERBQuestion(
        id="q1", dataset="enterprise-rag-bench", question="Q?", category="basic",
        source_types=(), expected_doc_ids=("gold", "missing"), gold_answer="", answer_facts=(),
    )
    response = AgentResponse("", ["gold-0", "gold-1", "other-0"], cited)

    scores = ERBEvaluator(Mock()).evaluate(question, response).scores

    assert scores["returned_document_recall"] == 0.5
    assert scores["cited_document_recall"] == recall


def test_browsecomp_plus_decrypts_the_publishers_obfuscation():
    # Longer than the 32-byte key, and not ASCII, to cover key repetition and UTF-8 decoding.
    plaintext = "Which café in Zürich opened first, and in which year? 答え"
    key = hashlib.sha256(BrowseCompPlus.CANARY.encode()).digest()
    data = plaintext.encode()
    ciphertext = base64.b64encode(bytes(b ^ key[i % len(key)] for i, b in enumerate(data))).decode()

    assert BrowseCompPlus.decrypt(ciphertext) == plaintext


@pytest.mark.parametrize(
    "text, title",
    [
        ("---\ntitle: Page title\ndate: 2002-05-13\n---\nBody", "Page title"),
        ("---\nauthor: A\ntitle: 'Quoted' title\n---\nBody", "'Quoted' title"),
        ("---\ndate: 2002-05-13\n---\nBody", "https://example.com/page"),
        ("---\n---\ntitle: Not front matter", "https://example.com/page"),
        ("Body without front matter", "https://example.com/page"),
    ],
)
def test_browsecomp_plus_titles_pages_from_front_matter_or_url(text, title):
    document = BrowseCompPlusDocument.from_corpus_row("5412", text, "https://example.com/page", "browsecomp-plus")

    assert document.title == title
    assert document.text == text


def test_browsecomp_plus_passages_keep_page_id_and_url(monkeypatch):
    monkeypatch.setattr(dataset_contracts, "split_text_into_chunks", lambda *args: ["First", "Second"])
    document = BrowseCompPlusDocument.from_corpus_row("5412", "First Second", "https://example.com", "browsecomp-plus")

    passages = document.get_passages()

    assert [passage.id for passage in passages] == ["5412-0", "5412-1"]
    assert all(BrowseCompPlusDocument.read_doc_id(passage.id) == "5412" for passage in passages)
    assert all(passage.url == "https://example.com" for passage in passages)