"""EnterpriseRAG-Bench (Onyx): questions over a synthetic company's documents from nine workplace tools."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass

import pyarrow.parquet as pq

from dataset_contracts import (
    BenchmarkCollection,
    ChunkedDocument,
    ChunkedPassage,
    HFBenchmarkDataset,
    Question,
)


@dataclass(frozen=True)
class ERBPassage(ChunkedPassage):
    """One chunk of a document, with the document's source tool."""

    source_type: str


@dataclass(frozen=True)
class ERBDocument(ChunkedDocument):
    """A document from one workplace tool, e.g. a Slack thread or a Jira ticket, split into chunks when stored.

    A few EnterpriseRAG-Bench IDs are shared by two different documents, so the
    stored ID gets a suffix for every copy after the first.
    """

    source_type: str

    def make_passage(self, chunk_idx: int, text: str) -> ERBPassage:
        return ERBPassage(**asdict(super().make_passage(chunk_idx, text)), source_type=self.source_type)

    @classmethod
    def read_doc_id(cls, passage_id: str) -> str:
        """The ERB document ID of a passage ID made by get_passages, e.g. "abc-copy2-3" -> "abc"."""
        return re.sub(r"-copy\d+$", "", super().read_doc_id(passage_id))


@dataclass(frozen=True)
class ERBQuestion(Question):
    """A question with its reference answer, the facts a complete answer states, and its supporting documents."""

    # ERB's question_type, e.g. "basic" or "info_not_found"; evaluate.py reports scores per category.
    category: str
    source_types: tuple[str, ...]
    expected_doc_ids: tuple[str, ...]
    gold_answer: str
    answer_facts: tuple[str, ...]


class EnterpriseRAGBench(HFBenchmarkDataset):
    """About 512,000 documents and 500 questions in ten categories, some with no answer in the corpus."""

    name = "enterprise-rag-bench"
    repo = "onyx-dot-app/EnterpriseRAG-Bench"
    revision = "69916e31c68aa5963c00248fd7f0bc12d04fd235"

    def read_corpus_documents(self) -> list[ERBDocument]:
        table = pq.read_table(self.download_file("data/documents/test.parquet"))
        columns = {name: table.column(name).to_pylist() for name in ("doc_id", "source_type", "title", "content")}
        copies: Counter[str] = Counter()
        documents = []
        for doc_id, source_type, title, content in zip(*columns.values(), strict=True):
            copies[doc_id] += 1
            documents.append(
                ERBDocument(
                    id=doc_id if copies[doc_id] == 1 else f"{doc_id}-copy{copies[doc_id]}",
                    title=title,
                    text=content,
                    datasets=(self.name,),
                    doc_id=doc_id,
                    source_type=source_type,
                )
            )
        return documents

    def read_questions(self) -> list[ERBQuestion]:
        records = pq.read_table(self.download_file("data/questions/test.parquet")).to_pylist()
        return [
            ERBQuestion(
                id=r["question_id"],
                dataset=self.name,
                question=r["question"],
                category=r["question_type"],
                source_types=tuple(r["source_types"]),
                expected_doc_ids=tuple(r["expected_doc_ids"]),
                gold_answer=r["gold_answer"],
                answer_facts=tuple(r["answer_facts"]),
            )
            for r in records
        ]


class EnterpriseRAGBenchCollection(BenchmarkCollection):
    name = "enterprise-rag-bench"
    datasets = (EnterpriseRAGBench(),)


if __name__ == "__main__":
    collection = EnterpriseRAGBenchCollection()
    documents = collection.read_merged_corpus_documents()
    questions = collection.read_questions()
    print(f"{len(questions)} questions, {len(documents)} documents")
