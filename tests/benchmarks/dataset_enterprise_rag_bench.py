"""EnterpriseRAG-Bench (Onyx): questions over a synthetic company's documents from nine workplace tools."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import ClassVar

import pyarrow.parquet as pq

from chunking import split_text_into_chunks
from dataset_contracts import (
    BenchmarkCollection,
    Document,
    HFBenchmarkDataset,
    Passage,
    Question,
)


@dataclass(frozen=True)
class ERBPassage(Passage):
    """One chunk of a document, with the document's EnterpriseRAG-Bench ID and source tool."""

    doc_id: str
    chunk_idx: int
    source_type: str


@dataclass(frozen=True)
class ERBDocument(Document):
    """A document from one workplace tool, e.g. a Slack thread or a Jira ticket, split into chunks when stored."""

    # Items don't record these, so name the container after them, e.g. enterprise-rag-bench-c512-o128.
    CHUNK_MAX_TOKENS: ClassVar[int] = 512
    CHUNK_OVERLAP_TOKENS: ClassVar[int] = 128

    # EnterpriseRAG-Bench's ID; a few IDs are shared by two different documents,
    # so the stored ID gets a suffix for every copy after the first.
    doc_id: str
    source_type: str

    def get_passages(self) -> list[Passage]:
        return [
            ERBPassage(
                id=f"{self.id}-{index}",
                title=self.title,
                text=chunk,
                datasets=self.datasets,
                doc_id=self.doc_id,
                chunk_idx=index,
                source_type=self.source_type,
            )
            for index, chunk in enumerate(
                split_text_into_chunks(self.text, self.CHUNK_MAX_TOKENS, self.CHUNK_OVERLAP_TOKENS)
            )
        ]


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
    # Tells the service which ERBPassage fields link a chunk to its document.
    chunk_schema: ClassVar[dict[str, str]] = {
        "parent_document_id_path": "/doc_id",
        "chunk_order_path": "/chunk_idx",
    }


if __name__ == "__main__":
    collection = EnterpriseRAGBenchCollection()
    documents = collection.read_merged_corpus_documents()
    questions = collection.read_questions()
    print(f"{len(questions)} questions, {len(documents)} documents")
