"""Contracts for scoring agentic search responses against benchmark questions.

To score a collection, subclass `Evaluator` and register it in evaluate.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
from typing import Any, ClassVar, Generic, TypeVar

from openai import OpenAI

from dataset_contracts import Question

QuestionT = TypeVar("QuestionT", bound=Question)


@dataclass(frozen=True)
class AgentResponse:
    """The agent's answer and the Cosmos DB item IDs of the documents it returned for one question."""

    # The service's [database/container/id] citations and its marker for made-up ones.
    CITATION: ClassVar[re.Pattern[str]] = re.compile(
        r"\s*\[(?:[^\[\]/]+/[^\[\]/]+/[^\[\]/]+|unverified source)\]"
    )

    # Citations are removed so the judge scores only the answer.
    answer: str
    # Returned documents are references without text, so evaluators look these
    # IDs up in the collection's data instead.
    document_ids: list[str]

    @classmethod
    def from_response_body(cls, body: dict[str, Any]) -> AgentResponse:
        """Read the answer and document IDs from an /agentic_search response body."""
        return cls(
            answer=cls.CITATION.sub("", body.get("answer") or ""),
            document_ids=[d["id"] for d in body.get("documents", [])],
        )


@dataclass(frozen=True)
class EvaluationResult:
    """One question's scores, None where a score does not apply, and the judge's token counts."""

    scores: dict[str, float | None]
    judge_tokens: dict[str, int | None]
    # Text to read alongside the scores, e.g. the judge's reason for a verdict.
    details: dict[str, str] = field(default_factory=dict)


class Judge:
    """Sends a prompt to the judge model and returns its reply with the call's token counts."""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def ask(self, prompt: str) -> tuple[str, dict[str, int | None]]:
        """Return the reply text and token counts, None where the endpoint did not report them."""
        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}]
        )
        usage = response.usage
        tokens = {
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
        }
        return response.choices[0].message.content or "", tokens


class Evaluator(ABC, Generic[QuestionT]):
    """Scores responses to every question of one benchmark collection."""

    name: str

    def __init__(self, judge: Judge):
        self.judge = judge

    @abstractmethod
    def evaluate(self, question: QuestionT, response: AgentResponse) -> EvaluationResult:
        """Score one response to one question."""
