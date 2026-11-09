"""The five datasets of the HippoRAG 2 benchmark (arXiv:2502.14802), all Wikipedia QA."""

from __future__ import annotations

import json
from abc import abstractmethod

from dataset_contracts import (
    BenchmarkCollection,
    HFBenchmarkDataset,
    Passage,
    Question,
)


def _dedupe_keep_order(values) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


class HippoRAG2Dataset(HFBenchmarkDataset):
    """Each dataset has `<name>.json` (questions) and `<name>_corpus.json` (title/text passages)."""

    repo = "osunlp/HippoRAG_2"
    revision = "5ec05b38deecc3318bb432c69865959c56058990"

    def read_corpus_passages(self) -> list[Passage]:
        records = json.loads(self.download_file(f"{self.name}_corpus.json").read_text())
        return [
            Passage.from_title_and_text(r["title"], r["text"], self.name)
            for r in records
        ]

    def read_questions(self) -> list[Question]:
        records = json.loads(self.download_file(f"{self.name}.json").read_text())
        return [self.parse_question_record(r, i) for i, r in enumerate(records)]

    @abstractmethod
    def parse_question_record(self, record: dict, index: int) -> Question:
        """Turn one question record, and its position in the file, into a Question."""


class HotpotQA(HippoRAG2Dataset):
    name = "hotpotqa"

    def parse_question_record(self, record: dict, index: int) -> Question:
        return Question(
            id=record["_id"],
            dataset=self.name,
            question=record["question"],
            answers=(record["answer"],),
            gold_titles=_dedupe_keep_order(title for title, _ in record["supporting_facts"]),
        )


class TwoWikiMultihopQA(HippoRAG2Dataset):
    name = "2wikimultihopqa"

    def parse_question_record(self, record: dict, index: int) -> Question:
        return Question(
            id=record["_id"],
            dataset=self.name,
            question=record["question"],
            answers=(record["answer"],),
            gold_titles=_dedupe_keep_order(title for title, _ in record["supporting_facts"]),
        )


class MuSiQue(HippoRAG2Dataset):
    name = "musique"

    def parse_question_record(self, record: dict, index: int) -> Question:
        return Question(
            id=record["id"],
            dataset=self.name,
            question=record["question"],
            answers=_dedupe_keep_order([record["answer"], *record["answer_aliases"]]),
            gold_titles=_dedupe_keep_order(
                p["title"] for p in record["paragraphs"] if p["is_supporting"]
            ),
        )


class NaturalQuestions(HippoRAG2Dataset):
    name = "nq_rear"

    def parse_question_record(self, record: dict, index: int) -> Question:
        # This file has no question IDs, so the position in the file is used.
        return Question(
            id=f"{self.name}-{index}",
            dataset=self.name,
            question=record["question"],
            answers=_dedupe_keep_order(record["reference"]),
            gold_titles=_dedupe_keep_order(
                c["title"] for c in record["contexts"] if c["is_supporting"]
            ),
        )


class PopQA(HippoRAG2Dataset):
    name = "popqa"

    def parse_question_record(self, record: dict, index: int) -> Question:
        return Question(
            id=str(record["id"]),
            dataset=self.name,
            question=record["question"],
            # Stored as a JSON-encoded list of strings.
            answers=_dedupe_keep_order(json.loads(record["possible_answers"])),
            gold_titles=_dedupe_keep_order(
                p["title"] for p in record["paragraphs"] if p["is_supporting"]
            ),
        )


class HippoRAG2Collection(BenchmarkCollection):
    name = "hipporag2"
    datasets = (HotpotQA(), TwoWikiMultihopQA(), MuSiQue(), NaturalQuestions(), PopQA())


if __name__ == "__main__":
    collection = HippoRAG2Collection()
    passages = collection.read_merged_corpus_passages()
    questions = collection.read_questions()
    for dataset in collection.datasets:
        count = sum(q.dataset == dataset.name for q in questions)
        print(f"{dataset.name}: {count} questions")
    print(f"{len(passages)} unique passages")
