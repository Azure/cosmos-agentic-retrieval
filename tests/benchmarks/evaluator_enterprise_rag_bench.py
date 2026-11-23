"""Scores agentic search answers to EnterpriseRAG-Bench questions.

Follows ERB's metrics-based evaluation (src/scripts/answer_evaluation/
metrics_based_eval.py) without its gold-correction step, like its
--no-correction mode, and with our own judge model. Scores are therefore not
comparable with the ERB leaderboard. Document recall is scored over the documents
the response returned and, as ERB intends, over the ones its answer cites.

The two judge prompts are copied from ERB's src/prompts/answer_evaluation.py at
commit d36685e. The fact prompt also shows the question, with an instruction to
use it only to resolve what the answer refers to; without it, short correct
answers such as "Marcus Chen volunteered to be captain" were judged to miss
facts that name the context, such as the team. ERB's code is under this notice:

    MIT License
    Copyright (c) 2026 DanswerAI, Inc.
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import json
import re

from dataset_enterprise_rag_bench import ERBDocument, ERBQuestion
from evaluation_contracts import AgentResponse, EvaluationResult, Evaluator

ANSWER_WHOLISTIC_EVALUATION_PROMPT = """
You are a wholistic and detail-oriented answer evaluator. Given a query, a gold answer, and a candidate answer, evaluate if the candidate answer aligned with the gold answer.
Use the following metrics for evaluating the answer:
- The candidate answer must provide loosely the same information as the gold answer. The core aspects directly asked by the query must be addressed in the candidate answer and they must not conflict with the gold answer.
- If there are any specific quantities mentioned in both answers, they must match.
- The candidate answer is not required to contain all of the same details as the gold answer.
- The candidate answer must address the key parts of the query, if it is missing anything critical to the question, it is misaligned.
- The candidate answer may contain more details, richer information, or other helpful relevant information than the gold answer, this is ok.
- The candidate answer may offer up additional loosely related information that adds to the context of the answer, this is ok as long as it does not lead the user to an incorrect conclusion (compared to the gold answer).
- Do not penalize the candidate answer for stylistic differences. If the candidate answer offers follow up questions, asks additional clarifications to the user, or offers additional context, \
this is ok as long as it contains the necessary information to answer the question.

There is a separate check for answer completeness, this is not in scope for this evaluation. However, if there are core parts of the question being left out, this is misaligned.

## Query
```
{query}
```

## Gold Answer
```
{gold_answer}
```

## Candidate Answer
```
{candidate_answer}
```

## Output Format
Output a JSON with "reason" and "aligned" fields. The "reason" field should be a as concise as possible (max 1 sentence) explanation of why the candidate answer is aligned or misaligned with the gold answer. \
The "aligned" field should be a simple "yes" or "no", use only those two strings literally and nothing else.

CRITICAL: Output only a JSON object with the following fields in the order shown below (with no additional text or formatting):
{{
  "reason": "reason for the classification",
  "aligned": "yes or no"
}}
""".strip()


INDIVIDUAL_FACT_VALIDATOR_PROMPT = """
You are an answer validator. Given an answer and a statement, determine if the answer is consistent with and contains the information in the statement. \
The answer may contain more details or richer information than the statement but as long as it does not contradict the statement, this is valid. \
If there are negative statements such as "The answer must not say...", it is valid if the answer mentions the statement with caveats or qualifications. \
It is valid if additional context is shared for completeness however hallucinations are not allowed. \
Output a simple yes or no for if the answer is consistent with and contains the information in the statement.
Use the question only to understand what the answer refers to. The information in the statement must come from the answer; do not credit information that appears only in the question.

## Question
```
{question}
```

## Answer
```
{answer}
```

## Statement
```
{statement}
```

CRITICAL: output only a simple yes if the answer is consistent with the statement or a no if the answer does not contain the information in the statement or contradicts the statement.
""".strip()

# ERB also gives the correctness judge three tries to reply with valid JSON.
CORRECTNESS_ATTEMPTS = 3
YES = re.compile(r"\byes\b", re.IGNORECASE)


class ERBEvaluator(Evaluator[ERBQuestion]):
    """Correctness and completeness of the answer, their product, and recall of the gold documents."""

    name = "enterprise-rag-bench"

    def evaluate(self, question: ERBQuestion, response: AgentResponse) -> EvaluationResult:
        returned_doc_ids = self.read_doc_ids(response.document_ids)
        cited_doc_ids = (
            None if response.cited_document_ids is None else self.read_doc_ids(response.cited_document_ids)
        )
        expected = set(question.expected_doc_ids)
        judge_tokens = {"prompt_tokens": 0, "completion_tokens": 0}
        details = {}
        if response.answer:
            correct, details["correctness_reason"] = self.judge_correctness(question, response.answer, judge_tokens)
            supported = sum(
                self.judge_fact(question.question, response.answer, fact, judge_tokens)
                for fact in question.answer_facts
            )
            completeness = supported / len(question.answer_facts) if question.answer_facts else 1.0
        else:
            correct, completeness = False, 0.0
        return EvaluationResult(
            scores={
                "correct": float(correct),
                "completeness": completeness,
                "overall": completeness if correct else 0.0,
                # Questions without gold documents test answers only.
                "returned_document_recall": (
                    len(returned_doc_ids & expected) / len(expected) if expected else None
                ),
                "returned_invalid_extra_documents": (
                    float(len(returned_doc_ids - expected)) if expected else None
                ),
                # Not scored for responses that don't mark which documents the answer cites.
                "cited_document_recall": (
                    len(cited_doc_ids & expected) / len(expected)
                    if expected and cited_doc_ids is not None
                    else None
                ),
            },
            judge_tokens=judge_tokens,
            details=details,
        )

    def read_doc_ids(self, passage_ids: list[str]) -> set[str]:
        """ERB document IDs of the given chunks; several chunks of one document count once."""
        return {ERBDocument.read_doc_id(passage_id) for passage_id in passage_ids}

    def judge_correctness(
        self, question: ERBQuestion, answer: str, judge_tokens: dict[str, int | None]
    ) -> tuple[bool, str]:
        """Ask whether the answer is aligned with the gold answer; return the verdict and the judge's reason."""
        if not question.gold_answer:
            return True, ""
        prompt = ANSWER_WHOLISTIC_EVALUATION_PROMPT.format(
            query=question.question, gold_answer=question.gold_answer, candidate_answer=answer
        )
        reply = ""
        for _ in range(CORRECTNESS_ATTEMPTS):
            reply = self.ask_judge(prompt, judge_tokens)
            # The judge may wrap the JSON object in other text, such as a code fence.
            start, end = reply.find("{"), reply.rfind("}")
            try:
                verdict = json.loads(reply[start : end + 1])
            except ValueError:
                continue
            if isinstance(verdict, dict) and isinstance(verdict.get("aligned"), str):
                return bool(YES.search(verdict["aligned"])), str(verdict.get("reason", ""))
        raise ValueError(
            f"correctness reply was not the expected JSON after {CORRECTNESS_ATTEMPTS} tries: {reply[:200]!r}"
        )

    def judge_fact(self, question: str, answer: str, fact: str, judge_tokens: dict[str, int | None]) -> bool:
        """Ask whether the answer contains one of the facts a complete answer states."""
        prompt = INDIVIDUAL_FACT_VALIDATOR_PROMPT.format(question=question, answer=answer, statement=fact)
        reply = self.ask_judge(prompt, judge_tokens)
        first_line = reply.strip().splitlines()[0] if reply.strip() else ""
        return bool(YES.search(first_line))

    def ask_judge(self, prompt: str, judge_tokens: dict[str, int | None]) -> str:
        """Ask the judge and add the call's token counts to judge_tokens; an unreported count makes the total unknown."""
        reply, tokens = self.judge.ask(prompt)
        for name, count in tokens.items():
            total = judge_tokens.get(name)
            judge_tokens[name] = None if total is None or count is None else total + count
        return reply
