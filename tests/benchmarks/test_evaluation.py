"""Offline checks for reading benchmark datasets and scoring toolkit-backed search responses by document ID."""

import base64
import hashlib
import json
from dataclasses import asdict
from unittest.mock import Mock

import pytest

pytest.importorskip("huggingface_hub")
pytest.importorskip("markitdown")
pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("pypandoc")
pytest.importorskip("tiktoken")

import pandas as pd
import pypandoc
from pptx import Presentation

import dataset_contracts
import dataset_enterprise_rag_bench as erb
from dataset_browsecomp_plus import BrowseCompPlus, BrowseCompPlusDocument, BrowseCompPlusQuestion
from dataset_contracts import Document
from dataset_harvey_firm_knowledge import HarveyCriterion, HarveyDocument, HarveyFirmKnowledge
from dataset_hipporag2 import HippoRAG2Collection, HippoRAGQuestion
from evaluation_contracts import AgentResponse
from evaluator_browsecomp_plus import BrowseCompPlusEvaluator
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


BROWSECOMP_PLUS_QUESTION = BrowseCompPlusQuestion(
    id="q1", dataset="browsecomp-plus", question="Q?", answer="A",
    evidence_doc_ids=("gold", "evidence", "unreturned", "other"), gold_doc_ids=("gold", "unreturned"),
)


@pytest.mark.parametrize(
    "reply, correct",
    [
        ("extracted_final_answer: A\nreasoning: Matches.\ncorrect: yes\nconfidence: 100", 1.0),
        ("**extracted_final_answer:** B\n**reasoning:** It is correct: yes, in part.\n**correct:** no", 0.0),
        ("**correct**: Yes", 1.0),
    ],
)
def test_browsecomp_plus_reads_the_judges_verdict_line(reply, correct):
    judge = Mock()
    judge.ask.return_value = (reply, {"prompt_tokens": 10, "completion_tokens": 5})

    result = BrowseCompPlusEvaluator(judge).evaluate(BROWSECOMP_PLUS_QUESTION, AgentResponse("A", []))

    assert result.scores["correct"] == correct
    assert result.judge_tokens == {"prompt_tokens": 10, "completion_tokens": 5}


def test_browsecomp_plus_rejects_a_judge_reply_without_a_verdict():
    judge = Mock()
    judge.ask.return_value = ("reasoning: unsure", {"prompt_tokens": 1, "completion_tokens": 1})

    with pytest.raises(ValueError, match="correct: yes\\|no"):
        BrowseCompPlusEvaluator(judge).evaluate(BROWSECOMP_PLUS_QUESTION, AgentResponse("A", []))


@pytest.mark.parametrize(
    "cited, cited_evidence, cited_gold",
    [(["gold-1", "gold-0"], 0.25, 0.5), ([], 0.0, 0.0), (None, None, None)],
)
def test_browsecomp_plus_recall_counts_pages_among_returned_and_cited_chunks(cited, cited_evidence, cited_gold):
    judge = Mock()
    response = AgentResponse("", ["gold-0", "gold-1", "evidence-3", "negative-0"], cited)

    scores = BrowseCompPlusEvaluator(judge).evaluate(BROWSECOMP_PLUS_QUESTION, response).scores

    assert scores == {
        "correct": 0.0,
        "returned_evidence_recall": 0.5,
        "returned_gold_recall": 0.5,
        "cited_evidence_recall": cited_evidence,
        "cited_gold_recall": cited_gold,
    }
    judge.ask.assert_not_called()


def test_harvey_reads_each_file_type_as_harvey_lab_does(tmp_path):
    pypandoc.convert_text("# Memo\n\nThe HSR second request arrived.", "docx", format="md", outputfile=str(tmp_path / "memo.docx"))
    with pd.ExcelWriter(tmp_path / "deals.xlsx") as writer:
        pd.DataFrame({"Matter": ["1003-00001"], "Status": ["Second request"]}).to_excel(writer, sheet_name="Deals", index=False)
    deck = Presentation()
    deck.slides.add_slide(deck.slide_layouts[5]).shapes.title.text = "Closing timeline"
    deck.save(tmp_path / "update.pptx")
    (tmp_path / "note.eml").write_text("Subject: Second request\n\nPlease file the response.")
    pandoc = pypandoc.get_pandoc_path()

    def read(name):
        return HarveyFirmKnowledge.read_file_text(tmp_path / name, pandoc)

    assert read("memo.docx").startswith("# Memo\n\nThe HSR second request arrived.")
    assert read("deals.xlsx").splitlines() == ["=== Sheet: Deals ===", "    Matter         Status", "1003-00001 Second request"]
    assert "Closing timeline" in read("update.pptx")
    assert read("note.eml") == "Subject: Second request\n\nPlease file the response."


def test_harvey_tasks_drop_the_output_paragraph(tmp_path, monkeypatch):
    task_dir = tmp_path / "tasks" / "001"
    task_dir.mkdir(parents=True)
    criterion = {"id": "C-001", "title": "Qualifying matter", "match_criteria": "Identifies 1003-00001.", "deliverables": ["response.md"]}
    (task_dir / "task.json").write_text(json.dumps({
        "id": "001", "title": "HSR second requests", "docs_dir": "../../dms", "criteria": [criterion],
        "instructions": "Pull every matter with an HSR second request." + HarveyFirmKnowledge.OUTPUT_PARAGRAPH,
        "deliverables": {"response.md": "response.md"},
    }))
    monkeypatch.setattr(HarveyFirmKnowledge, "download_folder", lambda self, folder: tmp_path)

    [task] = HarveyFirmKnowledge().read_questions()

    assert task.question == "Pull every matter with an HSR second request."
    assert task.criteria == (HarveyCriterion("C-001", "Qualifying matter", "Identifies 1003-00001."),)


def test_harvey_documents_are_stored_under_a_hash_of_their_path(monkeypatch):
    monkeypatch.setattr(dataset_contracts, "split_text_into_chunks", lambda *args: ["First", "Second"])
    # Cosmos DB item IDs can't hold "/", "#" or "?".
    document = HarveyDocument.from_store_path("matters/1001-00001/Deal #2/memo?.docx", "First Second", "harvey-firm-knowledge")

    passages = document.get_passages()

    assert not set("/#?") & set(document.id)
    assert (document.title, document.doc_id, document.matter_id) == (
        "matters/1001-00001/Deal #2/memo?.docx", "matters/1001-00001/Deal #2/memo?.docx", "1001-00001",
    )
    assert all(HarveyDocument.read_doc_id(passage.id) == document.id for passage in passages)
    assert all(passage.matter_id == "1001-00001" for passage in passages)