"""Contracts for question-answering benchmarks, and loading them into Passage and Question objects.

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
    """One corpus document, e.g. a Wikipedia paragraph."""

    id: str
    title: str
    text: str
    # Datasets whose corpus contains this passage, e.g. ("hotpotqa", "musique").
    datasets: tuple[str, ...]

    @classmethod
    def from_title_and_text(cls, title: str, text: str, dataset: str) -> Passage:
        """Build a passage whose ID comes from its content, so every dataset and reload agrees on it."""
        passage_id = hashlib.sha256(f"{title}\n{text}".encode()).hexdigest()[:32]
        return cls(passage_id, title, text, (dataset,))


@dataclass(frozen=True)
class Question:
    """One benchmark question with its accepted answers."""

    id: str
    dataset: str
    question: str
    answers: tuple[str, ...]
    # Titles of the passages needed to answer, used to score retrieval.
    gold_titles: tuple[str, ...]


class BenchmarkDataset(ABC):
    """A question set and the corpus of passages it is answered from."""

    name: ClassVar[str]

    @abstractmethod
    def read_corpus_passages(self) -> list[Passage]:
        """Return every corpus passage."""

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

    def read_merged_corpus_passages(self) -> list[Passage]:
        """Return every dataset's passages, keeping passages shared by several datasets once."""
        passages: dict[str, Passage] = {}
        for dataset in self.datasets:
            for passage in dataset.read_corpus_passages():
                existing = passages.get(passage.id)
                if existing is None:
                    passages[passage.id] = passage
                elif dataset.name not in existing.datasets:
                    passages[passage.id] = replace(
                        existing, datasets=(*existing.datasets, dataset.name)
                    )
        return list(passages.values())

    def read_questions(self) -> list[Question]:
        """Return every dataset's questions."""
        return [q for dataset in self.datasets for q in dataset.read_questions()]
