"""Scores agentic search answers to Harvey LAB Law Firm Knowledge tasks.

Follows Harvey LAB's rubric grading (lab_core/evaluation/scoring.py at commit
7f9b1ee): one judge call per rubric criterion with its prompt, the task title
as the task description, and a task scored correct only when every criterion
passes. Harvey grades a response.md file with its configured judge (Claude by
default); we grade the agent's text answer with our own judge model, so scores
are not directly comparable with Harvey's. Tasks list no gold documents, but their criteria name
matters by number, so matter recall is scored over the documents the response
returned and cited.

The judge prompt is copied from lab_core/evaluation/prompts/rubric_criterion.txt.
Harvey LAB's code is under this notice:

    MIT License
    Copyright (c) 2026 Harvey AI
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

from dataset_harvey_firm_knowledge import HarveyCriterion, HarveyDocument, HarveyFirmKnowledge, HarveyTask
from evaluation_contracts import AgentResponse, EvaluationResult, Evaluator, Judge

RUBRIC_CRITERION_PROMPT = """You are evaluating a legal AI agent's work product against a specific quality criterion.

## Task
{task_description}

## Agent's Output
{agent_output}

## Criterion
**{criterion_title}**

{match_criteria}

## Instructions
Evaluate the agent's output against the criterion above.
- **PASS**: The agent's output satisfies the criterion as described
- **FAIL**: The agent's output does not satisfy the criterion as described

Respond with JSON only:

```json
{{
  "reasoning": "Brief explanation",
  "verdict": "pass" | "fail"
}}
```
"""

# A matter number as the criteria write it, e.g. "1003-00001".
MATTER_ID = re.compile(r"\b\d{4}-\d{5}\b")


class HarveyEvaluator(Evaluator[HarveyTask]):
    """Whether every rubric criterion passes, the share that pass, and recall of the matters the criteria name."""

    name = "harvey-firm-knowledge"

    def __init__(self, judge: Judge):
        super().__init__(judge)
        self.matter_ids_by_document_id = HarveyFirmKnowledge().read_matter_ids_by_document_id()

    def evaluate(self, question: HarveyTask, response: AgentResponse) -> EvaluationResult:
        judge_tokens: dict[str, int | None] = {"prompt_tokens": 0, "completion_tokens": 0}
        # Like Harvey, an empty answer fails every criterion without a judge call.
        verdicts = [
            self.judge_criterion(question, response.answer, criterion, judge_tokens) if response.answer
            else {"id": criterion.id, "verdict": "fail", "reasoning": "no answer"}
            for criterion in question.criteria
        ]
        passed = sum(v["verdict"] == "pass" for v in verdicts)
        named_matters = {
            matter for c in question.criteria for matter in MATTER_ID.findall(f"{c.title} {c.match_criteria}")
        }
        returned_matters = self.read_matter_ids(response.document_ids)
        cited_matters = (
            None if response.cited_document_ids is None else self.read_matter_ids(response.cited_document_ids)
        )
        return EvaluationResult(
            scores={
                "correct": float(passed > 0 and passed == len(verdicts)),
                "criteria_passed": passed / len(verdicts),
                "criteria_with_unreadable_verdict": float(sum(v["verdict"] == "error" for v in verdicts)),
                # Not scored for the few tasks whose criteria name no matter.
                "returned_matter_recall": (
                    len(returned_matters & named_matters) / len(named_matters) if named_matters else None
                ),
                "cited_matter_recall": (
                    len(cited_matters & named_matters) / len(named_matters)
                    if named_matters and cited_matters is not None
                    else None
                ),
            },
            judge_tokens=judge_tokens,
            details={"criteria": json.dumps(verdicts)},
        )

    def judge_criterion(
        self, task: HarveyTask, answer: str, criterion: HarveyCriterion, judge_tokens: dict[str, int | None]
    ) -> dict[str, str]:
        """Ask the judge whether the answer meets one criterion; an unreadable reply is an "error", never a pass."""
        prompt = RUBRIC_CRITERION_PROMPT.format(
            task_description=task.title,
            agent_output=answer,
            criterion_title=criterion.title,
            match_criteria=criterion.match_criteria,
        )
        reply, tokens = self.judge.ask(prompt)
        for name, count in tokens.items():
            total = judge_tokens.get(name)
            judge_tokens[name] = None if total is None or count is None else total + count
        # The judge may wrap the JSON object in other text, such as a code fence.
        try:
            verdict = json.loads(reply[reply.find("{") : reply.rfind("}") + 1])
            return {
                "id": criterion.id,
                "verdict": str(verdict.get("verdict", "fail")).lower(),
                "reasoning": str(verdict.get("reasoning", "")),
            }
        except (ValueError, AttributeError):
            return {"id": criterion.id, "verdict": "error", "reasoning": f"unreadable judge reply: {reply[:200]}"}

    def read_matter_ids(self, passage_ids: list[str]) -> set[str]:
        """Matters of the given chunks' documents."""
        return {
            self.matter_ids_by_document_id[HarveyDocument.read_doc_id(passage_id)] for passage_id in passage_ids
        }
