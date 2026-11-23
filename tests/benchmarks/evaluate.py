"""Score a saved benchmark run with its collection's evaluator.

Reads a run file written by run.py, matches each answered question to the
collection's question by dataset and question ID, and scores the response. The
judge model comes from LLM_BASE_URL, LLM_MODEL and LLM_API_KEY; without
LLM_API_KEY, it is called with an Entra token from `az login`.

Writes one JSON file with the evaluation settings, the mean of each score per
dataset (over the questions where the score applies), also per category for
questions that have one, and each question's scores. A question whose
evaluation fails is recorded with the reason, and the rest continue.

Example:
    python tests/benchmarks/evaluate.py .live-data/benchmark-runs/<run>.json
"""

import argparse
import json
import os
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import OpenAI
from tqdm import tqdm

from cosmos_agentic_retriever.server import LLM_TOKEN_SCOPE
from dataset_contracts import Question
from dataset_enterprise_rag_bench import EnterpriseRAGBenchCollection
from dataset_hipporag2 import HippoRAG2Collection
from evaluation_contracts import AgentResponse, Evaluator, Judge
from evaluator_enterprise_rag_bench import ERBEvaluator
from evaluator_hipporag2 import HippoRAGEvaluator

REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECTIONS = {
    "enterprise-rag-bench": EnterpriseRAGBenchCollection,
    "hipporag2": HippoRAG2Collection,
}
# Each collection's default evaluator; --evaluator picks another.
EVALUATORS = {"enterprise-rag-bench": ERBEvaluator, "hipporag2": HippoRAGEvaluator}


def read_category(question: Question | None) -> str | None:
    """The question's category, e.g. ERB's question type, for datasets whose questions have one."""
    return getattr(question, "category", None)


def evaluate_record(evaluator: Evaluator, question: Question, record: dict[str, Any]) -> dict[str, Any]:
    """Score one answered question, recording any failure instead of raising."""
    result: dict[str, Any] = {"dataset": record["dataset"], "question_id": record["question_id"]}
    if read_category(question) is not None:
        result["category"] = read_category(question)
    try:
        evaluation = evaluator.evaluate(question, AgentResponse.from_response_body(record["response"]))
    except Exception as exc:
        return {**result, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    return {
        **result,
        "status": "ok",
        "error": None,
        "scores": evaluation.scores,
        "judge_tokens": evaluation.judge_tokens,
        "details": evaluation.details,
    }


def summarize_results(results: list[dict[str, Any]], questions_in_run: int) -> dict[str, Any]:
    """Mean of each score over the questions where it applies, plus counts and judge token totals."""
    scored = [r for r in results if r["status"] == "ok"]
    names = list(dict.fromkeys(name for r in scored for name in r["scores"]))
    scores = {}
    for name in names:
        values = [r["scores"][name] for r in scored if r["scores"].get(name) is not None]
        scores[name] = {
            "mean": round(statistics.fmean(values), 3) if values else None,
            "questions": len(values),
        }
    judge_tokens = {
        field: sum(r["judge_tokens"].get(field) or 0 for r in scored)
        for field in ("prompt_tokens", "completion_tokens")
    }
    return {
        "questions_in_run": questions_in_run,
        "evaluated": len(scored),
        "errors": len(results) - len(scored),
        "scores": scores,
        "judge_tokens": judge_tokens,
    }


def write_scores_file(
    path: Path,
    metadata: dict[str, Any],
    run_records: list[dict[str, Any]],
    questions: dict[tuple[str, str], Question],
    results: list[dict[str, Any]],
) -> None:
    """Write the evaluation's settings, per-dataset and per-category summaries and results, replacing the file at once."""
    results = sorted(results, key=lambda r: (r["dataset"], r["question_id"]))
    datasets = sorted({r["dataset"] for r in run_records})
    summary = {
        dataset: summarize_results(
            [r for r in results if r["dataset"] == dataset],
            sum(r["dataset"] == dataset for r in run_records),
        )
        for dataset in datasets
    }
    summary["all"] = summarize_results(results, len(run_records))
    run_categories = [read_category(questions.get((r["dataset"], r["question_id"]))) for r in run_records]
    summary_by_category = {
        category: summarize_results(
            [r for r in results if r.get("category") == category],
            run_categories.count(category),
        )
        for category in sorted({c for c in run_categories if c is not None})
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_suffix(".partial")
    contents = {"metadata": metadata, "summary": summary}
    if summary_by_category:
        contents["summary_by_category"] = summary_by_category
    partial_path.write_text(json.dumps({**contents, "results": results}, indent=2))
    partial_path.replace(path)


def print_summary_table(heading: str, summary: dict[str, Any]) -> None:
    """Print one row per group with its evaluated count and mean scores."""
    names = list(dict.fromkeys(name for values in summary.values() for name in values["scores"]))
    widths = [max(len(n), 6) + 2 for n in names]
    print(f"{heading:<26}{'evaluated':>11}" + "".join(f"{n:>{w}}" for n, w in zip(names, widths)))
    for group, values in summary.items():
        means = [values["scores"].get(n, {}).get("mean") for n in names]
        print(
            f"{group:<26}{values['evaluated']:>5}/{values['questions_in_run']:<5}"
            + "".join(f"{m:>{w}.3f}" if m is not None else f"{'-':>{w}}" for m, w in zip(means, widths))
        )


def print_summary(path: Path) -> None:
    """Print each dataset's, and each category's, evaluated count and mean scores."""
    contents = json.loads(path.read_text())
    print_summary_table("dataset", contents["summary"])
    if "summary_by_category" in contents:
        print()
        print_summary_table("category", contents["summary_by_category"])
    print(f"Scores: {path}")


def read_git_commit() -> str:
    """The evaluator's commit, marked -dirty when there are uncommitted changes."""
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    return git("rev-parse", "--short", "HEAD") + ("-dirty" if git("status", "--porcelain") else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_file", type=Path, help="a run file written by run.py")
    parser.add_argument("--evaluator", choices=sorted(EVALUATORS), help="evaluator to use (default: the run's collection's)")
    parser.add_argument("--output-file", type=Path, help="where to write the scores (default: next to the run file, ending in .scores.json)")
    parser.add_argument("--workers", type=int, default=8, help="number of questions evaluated in parallel (default: %(default)s)")
    args = parser.parse_args()

    run = json.loads(args.run_file.read_text())
    collection_name = run["metadata"]["settings"]["collection"]
    evaluator_name = args.evaluator or collection_name
    output = args.output_file or args.run_file.with_suffix(".scores.json")
    if output.exists():
        parser.error(f"{output} already exists")

    questions = {(q.dataset, q.id): q for q in COLLECTIONS[collection_name]().read_questions()}
    answered = [r for r in run["results"] if r["status"] == "ok"]
    unknown = [r for r in answered if (r["dataset"], r["question_id"]) not in questions]
    if unknown:
        parser.error(
            f"{len(unknown)} run questions are not in {collection_name}, "
            f"e.g. {unknown[0]['dataset']}/{unknown[0]['question_id']}"
        )

    judge_model = os.environ["LLM_MODEL"]
    judge = Judge(
        OpenAI(
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ.get("LLM_API_KEY")
            or get_bearer_token_provider(AzureCliCredential(), LLM_TOKEN_SCOPE),
            # As many retries as the service's default, so rate limits don't fail evaluations.
            max_retries=6,
        ),
        judge_model,
    )
    evaluator = EVALUATORS[evaluator_name](judge)
    metadata = {
        "run_file": str(args.run_file),
        "evaluator": evaluator_name,
        "judge_model": judge_model,
        "evaluator_commit": read_git_commit(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    print(f"{len(answered)} answered questions to evaluate with {evaluator_name}")

    results: list[dict[str, Any]] = []
    pool = ThreadPoolExecutor(args.workers)
    try:
        futures = [
            pool.submit(evaluate_record, evaluator, questions[(r["dataset"], r["question_id"])], r)
            for r in answered
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            results.append(future.result())
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_scores_file(output, metadata, run["results"], questions, results)
        print_summary(output)


if __name__ == "__main__":
    main()
