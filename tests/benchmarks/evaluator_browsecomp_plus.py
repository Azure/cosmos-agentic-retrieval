"""Scores agentic search answers to BrowseComp-Plus questions.

Follows BrowseComp-Plus's evaluation (scripts_evaluation/evaluate_run.py at
commit 4b0df7a) with our own judge model instead of its Qwen3-32B, so accuracy
is not directly comparable with its leaderboard. Its retrieval recall counts
every page the agent saw during the run; ours counts the pages the response
returned, so the two are not comparable. Its citation recall, over the pages
the answer cites, matches ours, except that it averages only over answers with
citations. Its calibration error is not reported, as our agent states no
confidence.

The grader prompt is copied from that script, which took it from BrowseComp's
evaluation in OpenAI's simple-evals (browsecomp_eval.py). Both are under this
notice:

    MIT License
    Copyright (c) 2024 OpenAI
    Copyright (c) 2025 texttron
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

import re

from dataset_browsecomp_plus import BrowseCompPlusDocument, BrowseCompPlusQuestion
from evaluation_contracts import AgentResponse, EvaluationResult, Evaluator

GRADER_TEMPLATE = r"""
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. 

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
""".strip()

# The verdict line, plain or in Markdown bold as the publisher's parser accepts, e.g. "**correct:** yes".
VERDICT = re.compile(r"^[*\s]*correct(?:\*\*)?:(?:\*\*)?\s*(yes|no)\b", re.IGNORECASE | re.MULTILINE)


class BrowseCompPlusEvaluator(Evaluator[BrowseCompPlusQuestion]):
    """Whether the answer is correct, and recall of the evidence and gold pages among returned and cited pages."""

    name = "browsecomp-plus"

    def evaluate(self, question: BrowseCompPlusQuestion, response: AgentResponse) -> EvaluationResult:
        returned_doc_ids = self.read_doc_ids(response.document_ids)
        cited_doc_ids = (
            None if response.cited_document_ids is None else self.read_doc_ids(response.cited_document_ids)
        )
        evidence, gold = set(question.evidence_doc_ids), set(question.gold_doc_ids)
        judge_tokens: dict[str, int | None] = {"prompt_tokens": 0, "completion_tokens": 0}
        details = {}
        correct = False
        if response.answer:
            prompt = GRADER_TEMPLATE.format(
                question=question.question, response=response.answer, correct_answer=question.answer
            )
            reply, judge_tokens = self.judge.ask(prompt)
            verdict = VERDICT.search(reply)
            if verdict is None:
                raise ValueError(f"judge reply has no 'correct: yes|no' line: {reply[:200]!r}")
            correct = verdict.group(1).lower() == "yes"
            details["judge_reply"] = reply
        return EvaluationResult(
            scores={
                "correct": float(correct),
                "returned_evidence_recall": len(returned_doc_ids & evidence) / len(evidence),
                "returned_gold_recall": len(returned_doc_ids & gold) / len(gold),
                # Not scored for responses that don't mark which documents the answer cites.
                "cited_evidence_recall": (
                    None if cited_doc_ids is None else len(cited_doc_ids & evidence) / len(evidence)
                ),
                "cited_gold_recall": None if cited_doc_ids is None else len(cited_doc_ids & gold) / len(gold),
            },
            judge_tokens=judge_tokens,
            details=details,
        )

    def read_doc_ids(self, passage_ids: list[str]) -> set[str]:
        """Page IDs of the given chunks; several chunks of one page count once."""
        return {BrowseCompPlusDocument.read_doc_id(passage_id) for passage_id in passage_ids}
