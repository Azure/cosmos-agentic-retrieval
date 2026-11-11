"""Contracts for question-answering benchmarks, and loading them into Document, Passage and Question objects.

To add a dataset, subclass `BenchmarkDataset`, or `HFBenchmarkDataset` for one
stored on Hugging Face.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar

from huggingface_hub import hf_hub_download


@dataclass(frozen=True)
class Passage:
    """One searchable unit stored in Cosmos DB, e.g. a Wikipedia paragraph or part of a long document.

    Every field is stored on the passage's Cosmos DB item.
    """

    id: str
    title: str
    text: str
    # Datasets whose corpus contains this passage, e.g. ("hotpotqa", "musique").
    datasets: tuple[str, ...]


@dataclass(frozen=True)
class Document:
    """One corpus document as a dataset provides it.

    By default it is stored as a single passage with the same ID and fields. A
    dataset whose documents are long or carry more fields subclasses this and
    overrides get_passages.
    """

    id: str
    title: str
    text: str
    # Datasets whose corpus contains this document, e.g. ("hotpotqa", "musique").
    datasets: tuple[str, ...]

    @classmethod
    def from_title_and_text(cls, title: str, text: str, dataset: str) -> Document:
        """Build a document whose ID comes from its content, so every dataset and reload agrees on it."""
        document_id = hashlib.sha256(f"{title}\n{text}".encode()).hexdigest()[:32]
        return cls(document_id, title, text, (dataset,))

    def get_passages(self) -> list[Passage]:
        """Return the passages to store for this document."""
        return [Passage(self.id, self.title, self.text, self.datasets)]


@dataclass(frozen=True)
class Question:
    """One benchmark question as sent to the agent.

    Each dataset subclasses this with what its evaluation needs, such as
    reference answers and the documents that support them.
    """

    id: str
    dataset: str
    question: str


class BenchmarkDataset(ABC):
    """A question set and the corpus of documents it is answered from."""

    name: ClassVar[str]

    @abstractmethod
    def read_corpus_documents(self) -> list[Document]:
        """Return every corpus document."""

    @abstractmethod
    def read_questions(self) -> list[Question]:
        """Return every question."""


class HFBenchmarkDataset(BenchmarkDataset):
    """A dataset stored in a Hugging Face dataset repository at a pinned revision."""

    repo: ClassVar[str]
    revision: ClassVar[str]

    def download_file(self, file_name: str) -> Path:
        """Return a local copy of one repository file, using the Hugging Face cache."""
        return Path(
            hf_hub_download(
                repo_id=self.repo,
                filename=file_name,
                repo_type="dataset",
                revision=self.revision,
            )
        )


class BenchmarkCollection:
    """Datasets searched together over one shared corpus."""

    # E.g. "hipporag2"; also names the corpus container in Cosmos DB.
    name: ClassVar[str]
    datasets: ClassVar[tuple[BenchmarkDataset, ...]]

    def read_merged_corpus_documents(self) -> list[Document]:
        """Return every dataset's documents, keeping documents shared by several datasets once."""
        documents: dict[str, Document] = {}
        for dataset in self.datasets:
            for document in dataset.read_corpus_documents():
                existing = documents.get(document.id)
                if existing is None:
                    documents[document.id] = document
                elif dataset.name not in existing.datasets:
                    documents[document.id] = replace(
                        existing, datasets=(*existing.datasets, dataset.name)
                    )
        return list(documents.values())

    def read_questions(self) -> list[Question]:
        """Return every dataset's questions."""
        return [q for dataset in self.datasets for q in dataset.read_questions()]
