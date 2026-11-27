"""Harvey LAB Law Firm Knowledge: research requests over the document store of a fictional law firm."""

from __future__ import annotations

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pypandoc
from markitdown import MarkItDown

from dataset_contracts import (
    BenchmarkCollection,
    ChunkedDocument,
    ChunkedPassage,
    GitHubBenchmarkDataset,
    Question,
)


@dataclass(frozen=True)
class HarveyPassage(ChunkedPassage):
    """One chunk of a firm document, with the matter it is filed under."""

    matter_id: str


@dataclass(frozen=True)
class HarveyDocument(ChunkedDocument):
    """A file from the firm's document store, titled by its path there, e.g. "matters/1001-00001/Antitrust Analysis/hhi.xlsx".

    Some paths hold characters that Cosmos DB item IDs can't, so the stored ID
    is a hash of the path, and doc_id is the path.
    """

    # The firm's matter number, e.g. "1001-00001"; rubric criteria name matters by it.
    matter_id: str

    @classmethod
    def from_store_path(cls, store_path: str, text: str, dataset: str) -> HarveyDocument:
        """Build a document from its path in the document store and its extracted text."""
        return cls(
            id=hashlib.sha256(store_path.encode()).hexdigest()[:32],
            title=store_path,
            text=text,
            datasets=(dataset,),
            doc_id=store_path,
            matter_id=store_path.split("/")[1],
        )

    def make_passage(self, chunk_idx: int, text: str) -> HarveyPassage:
        return HarveyPassage(**asdict(super().make_passage(chunk_idx, text)), matter_id=self.matter_id)


@dataclass(frozen=True)
class HarveyCriterion:
    """One rubric item that a complete answer satisfies, judged pass or fail."""

    id: str
    title: str
    match_criteria: str


@dataclass(frozen=True)
class HarveyTask(Question):
    """A research request, e.g. "pull every antitrust matter with an HSR second request", with its rubric."""

    title: str
    criteria: tuple[HarveyCriterion, ...]


class HarveyFirmKnowledge(GitHubBenchmarkDataset):
    """250 tasks over 9,288 Word, Excel, email and PowerPoint files of 266 matters."""

    name = "harvey-firm-knowledge"
    repo = "harveyai/harvey-labs"
    commit = "7f9b1ee19f9224576a5a5143c4e6f75842ac7020"
    FOLDER: ClassVar[str] = "tasks/firm-knowledge"
    # Every task ends with this paragraph, removed because the agent answers in text, not in a file.
    OUTPUT_PARAGRAPH: ClassVar[str] = (
        "\n\nOutput: write your complete answer to `response.md` — a single self-contained markdown file "
        "with your findings, the matters/documents you relied on, and your reasoning."
    )

    def read_corpus_documents(self) -> list[HarveyDocument]:
        store = self.download_folder(self.FOLDER) / "dms"
        paths = sorted(path for path in store.rglob("*") if path.is_file())
        pandoc = pypandoc.get_pandoc_path()
        # Converting about 8,000 Word files runs one pandoc process per file.
        with ThreadPoolExecutor() as pool:
            texts = list(pool.map(lambda path: self.read_file_text(path, pandoc), paths))
        return [
            HarveyDocument.from_store_path(path.relative_to(store).as_posix(), text, self.name)
            for path, text in zip(paths, texts, strict=True)
        ]

    def read_questions(self) -> list[HarveyTask]:
        tasks = []
        for path in sorted((self.download_folder(self.FOLDER) / "tasks").glob("*/task.json")):
            task = json.loads(path.read_text())
            tasks.append(
                HarveyTask(
                    id=task["id"],
                    dataset=self.name,
                    # question=task["instructions"],
                    question=task["instructions"].removesuffix(self.OUTPUT_PARAGRAPH),
                    title=task["title"],
                    criteria=tuple(
                        HarveyCriterion(c["id"], c["title"], c["match_criteria"]) for c in task["criteria"]
                    ),
                )
            )
        return tasks

    @staticmethod
    def read_file_text(path: Path, pandoc: str) -> str:
        """Convert one file to text as Harvey LAB's agent reads it (lab_core/sandbox/parsers/parse_doc.py).

        Harvey uses pandoc 3.11; the pandoc bundled with pypandoc_binary may be a
        nearby version, whose markdown can differ in markup and list numbers.
        """
        suffix = path.suffix.lower()
        if suffix == ".docx":
            return subprocess.run(
                [pandoc, str(path), "-t", "markdown", "--wrap=none"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, check=True,
            ).stdout
        if suffix == ".xlsx":
            sheets = pd.read_excel(path, sheet_name=None)
            return "\n".join(
                part for sheet, frame in sheets.items() for part in (f"=== Sheet: {sheet} ===", frame.to_string(index=False))
            )
        if suffix == ".pptx":
            return MarkItDown().convert(str(path)).text_content
        # Emails and any other file are read as plain text.
        return path.read_bytes().decode("utf-8", errors="replace")


class HarveyFirmKnowledgeCollection(BenchmarkCollection):
    name = "harvey-firm-knowledge"
    datasets = (HarveyFirmKnowledge(),)


if __name__ == "__main__":
    collection = HarveyFirmKnowledgeCollection()
    documents = collection.read_merged_corpus_documents()
    questions = collection.read_questions()
    print(f"{len(questions)} questions, {len(documents)} documents")
