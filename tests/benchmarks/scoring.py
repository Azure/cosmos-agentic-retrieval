"""Score an answer and the retrieved passages against a benchmark question.

Answer scores follow the standard SQuAD/HotpotQA definitions: lowercase, drop
punctuation and the articles a/an/the, then compare words. Each score takes the
best match over the question's accepted answers. Full-sentence answers score
low on exact match and F1 even when correct, so containment and a judge model
are reported as well.
"""

import re
import string
from collections import Counter

from openai import OpenAI

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


def compute_answer_containment(prediction: str, answers: tuple[str, ...]) -> float:
    """1.0 if any normalized accepted answer appears, word for word, inside the prediction."""
    predicted = f" {' '.join(normalize_answer(prediction))} "
    return float(
        any(f" {' '.join(normalize_answer(a))} " in predicted for a in answers if normalize_answer(a))
    )


def judge_answer_with_model(
    client: OpenAI, model: str, question: str, answers: tuple[str, ...], prediction: str
) -> tuple[float, dict[str, int | None]]:
    """Score 1.0 if the judge model says the prediction matches an accepted answer, else 0.0.

    Also return the judge call's token counts, None where the endpoint did not report them.
    """
    prompt = JUDGE_PROMPT.format(question=question, answers="; ".join(answers), prediction=prediction)
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    verdict = (response.choices[0].message.content or "").strip().lower()
    usage = response.usage
    tokens = {
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
    }
    return float(verdict.startswith("correct")), tokens


def compute_gold_title_recall(retrieved_titles: list[str], gold_titles: tuple[str, ...]) -> float:
    """Fraction of the gold passage titles that appear among the retrieved passages."""
    found = set(retrieved_titles)
    return sum(title in found for title in gold_titles) / len(gold_titles)
