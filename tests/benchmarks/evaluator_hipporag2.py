"""Scores agentic search answers to HippoRAG 2 questions.

Answer scores follow the standard SQuAD/HotpotQA definitions: lowercase, drop
punctuation and the articles a/an/the, then compare words. Each score takes the
best match over the question's accepted answers. Full-sentence answers score
low on exact match and F1 even when correct, so a judge model scores them too.
"""

import re
import string
from collections import Counter

from dataset_hipporag2 import HippoRAG2Collection, HippoRAGQuestion
from evaluation_contracts import AgentResponse, EvaluationResult, Evaluator, Judge

JUDGE_PROMPT = """Question: {question}
Accepted answers: {answers}
Response: {prediction}

Does the response give an answer that means the same as one of the accepted answers?
Reply with exactly one word: correct or incorrect."""


def normalize_answer(text: str) -> list[str]:
    """Return the answer's words after lowercasing and removing punctuation and articles."""
    text = "".join(ch for ch in text.lower() if ch not in string.punctuation)
    return re.sub(r"\b(a|an|the)\b", " ", text).split()


def compute_exact_match(prediction: str, answers: tuple[str, ...]) -> float:
    """1.0 if the normalized prediction equals any normalized accepted answer, else 0.0."""
    predicted = normalize_answer(prediction)
    return float(any(predicted == normalize_answer(a) for a in answers))


def compute_f1(prediction: str, answers: tuple[str, ...]) -> float:
    """Best word-overlap F1 between the prediction and any accepted answer."""
    predicted = normalize_answer(prediction)
    best = 0.0
    for answer in answers:
        expected = normalize_answer(answer)
        overlap = sum((Counter(predicted) & Counter(expected)).values())
        if overlap:
            precision, recall = overlap / len(predicted), overlap / len(expected)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def compute_gold_title_recall(retrieved_titles: list[str], gold_titles: tuple[str, ...]) -> float:
    """Fraction of the gold passage titles that appear among the retrieved passages."""
    found = set(retrieved_titles)
    return sum(title in found for title in gold_titles) / len(gold_titles)


class HippoRAGEvaluator(Evaluator[HippoRAGQuestion]):
    """Exact match, F1 and judge accuracy of the answer, and recall of the gold passage titles."""

    name = "hipporag2"

    def __init__(self, judge: Judge):
        super().__init__(judge)
        # Each HippoRAG 2 document is stored as one passage with the same ID.
        self.titles_by_id = {
            d.id: d.title for d in HippoRAG2Collection().read_merged_corpus_documents()
        }

    def evaluate(self, question: HippoRAGQuestion, response: AgentResponse) -> EvaluationResult:
        answer = response.answer
        if answer:
            prompt = JUDGE_PROMPT.format(
                question=question.question, answers="; ".join(question.answers), prediction=answer
            )
            verdict, judge_tokens = self.judge.ask(prompt)
            judge_score = float(verdict.strip().lower().startswith("correct"))
        else:
            judge_score, judge_tokens = 0.0, {"prompt_tokens": 0, "completion_tokens": 0}
        titles = [self.titles_by_id[i] for i in response.document_ids if i in self.titles_by_id]
        cited_recall = None
        if response.cited_document_ids is not None:
            cited_titles = [
                self.titles_by_id[i] for i in response.cited_document_ids if i in self.titles_by_id
            ]
            cited_recall = compute_gold_title_recall(cited_titles, question.gold_titles)
        return EvaluationResult(
            scores={
                "exact_match": compute_exact_match(answer, question.answers),
                "f1": compute_f1(answer, question.answers),
                "judge": judge_score,
                "gold_title_recall": compute_gold_title_recall(titles, question.gold_titles),
                "cited_gold_title_recall": cited_recall,
            },
            judge_tokens=judge_tokens,
        )
