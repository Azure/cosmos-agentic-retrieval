"""Run a benchmark collection's questions through agentic search and save the responses.

By default the service runs inside this process against the corpus loaded by
load_corpus_to_cosmos.py, signing in with `az login`; its agent model comes from
LLM_BASE_URL, LLM_MODEL and LLM_API_KEY. With --toolkit-url, each question
instead goes to a deployed MCP toolkit's agentic_search tool, and the
deployment's own retriever settings apply. evaluate.py scores the saved run.

Each run writes one JSON file holding its settings, a summary, and one record
per question with the service's response. A failed question is recorded and the
run moves on; on Ctrl+C the finished questions are still written. --resume-file
reruns only the failed or missing questions of an earlier run, with that run's
settings, into a new file.

Examples:
    python tests/benchmarks/run.py --account-uri https://<account>.documents.azure.com:443/ \
        --collection hipporag2 --container hipporag2-full --questions-per-dataset 100
    python tests/benchmarks/run.py --toolkit-url https://<app>.azurecontainerapps.io/mcp \
        --toolkit-app-id <entra-app-client-id> --collection hipporag2 --container hipporag2-full
    python tests/benchmarks/run.py --resume-file .live-data/benchmark-runs/<run>.json

Without --questions-per-dataset, every question is run.
"""

import argparse
import json
import math
import random
import statistics
import subprocess
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anyio
import httpx
from azure.core.credentials import AccessToken
from azure.identity import AzureCliCredential
from fastapi.testclient import TestClient
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent
from tqdm import tqdm

from cosmos_agentic_retriever.config import RetrieverConfig
from cosmos_agentic_retriever.server import create_app
from dataset_contracts import Question
from dataset_enterprise_rag_bench import EnterpriseRAGBenchCollection
from dataset_hipporag2 import HippoRAG2Collection

COLLECTIONS = {
    "enterprise-rag-bench": EnterpriseRAGBenchCollection,
    "hipporag2": HippoRAG2Collection,
}
REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / ".live-data" / "benchmark-runs"
AGENT_TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "cached_prompt_tokens", "reasoning_tokens")
MAX_DOCUMENTS = 10
# Settings that decide which questions run and where; a resumed run reuses them.
RUN_SETTINGS = (
    "collection",
    "container",
    "account_uri",
    "database",
    "questions_per_dataset",
    "seed",
    "toolkit_url",
    "toolkit_app_id",
)
DEFAULTS = {"database": "mcp-bench-v1", "seed": 0}

# Sends one question and returns the /agentic_search response body; failures carry an "error" key.
SearchFn = Callable[[str], dict[str, Any]]


def sample_questions_per_dataset(
    questions: list[Question], per_dataset: int | None, seed: int
) -> list[Question]:
    """Pick the same random questions from each dataset for a given seed; None keeps all."""
    if per_dataset is None:
        return questions
    by_dataset: dict[str, list[Question]] = {}
    for question in questions:
        by_dataset.setdefault(question.dataset, []).append(question)
    rng = random.Random(seed)
    return [
        q for qs in by_dataset.values() for q in rng.sample(qs, min(per_dataset, len(qs)))
    ]


def start_in_process_search(stack: ExitStack, settings: dict[str, Any]) -> SearchFn:
    """Start the service in this process and send questions to its /agentic_search."""
    config = RetrieverConfig(
        account_uri=settings["account_uri"],
        cosmos_database=settings["database"],
        cosmos_credential="azure_cli",
        cosmos_containers={
            settings["container"]: {
                "cosmos_schema": {
                    "item_id_path": "/id",
                    "partition_key_paths": ["/id"],
                    "text_paths": ["/title", "/text"],
                    # Collections stored as chunks define which fields link a chunk to its document.
                    **getattr(COLLECTIONS[settings["collection"]], "chunk_schema", {}),
                },
                "search_text_fields": ["/title", "/text"],
            }
        },
    )
    http = stack.enter_context(TestClient(create_app(config)))

    def search(query: str) -> dict[str, Any]:
        response = http.post(
            "/agentic_search", json={"query": query, "maxDocuments": MAX_DOCUMENTS}
        )
        body = response.json()
        if response.status_code != 200:
            body.setdefault("error", f"HTTP {response.status_code}")
        return body

    return search


def start_toolkit_search(settings: dict[str, Any]) -> SearchFn:
    """Send questions to a deployed toolkit's agentic_search tool, signed in with `az login`.

    Each question opens its own MCP session, so worker threads share no connection.
    """
    credential = AzureCliCredential()
    scope = f"api://{settings['toolkit_app_id']}/.default"
    token: AccessToken | None = None
    lock = threading.Lock()

    def read_bearer_token() -> str:
        nonlocal token
        with lock:
            # Renew early so a slow question does not outlive its token.
            if token is None or token.expires_on - time.time() < 300:
                token = credential.get_token(scope)
            return token.token

    async def call_agentic_search(query: str) -> dict[str, Any]:
        arguments = {
            "query": query,
            "maxDocuments": MAX_DOCUMENTS,
            "database": settings["database"],
            "container": settings["container"],
        }
        async with (
            httpx.AsyncClient(
                headers={"Authorization": f"Bearer {read_bearer_token()}"},
                timeout=httpx.Timeout(60, read=900),
            ) as http,
            streamable_http_client(settings["toolkit_url"], http_client=http) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await session.call_tool("agentic_search", arguments)
        text = next((c.text for c in result.content if isinstance(c, TextContent)), "")
        if result.isError:
            return {"error": text or "The tool call failed."}
        # The toolkit passes the service's response body through as the tool's text.
        return json.loads(text)

    return lambda query: anyio.run(call_agentic_search, query)


def answer_question(search: SearchFn, question: Question) -> dict[str, Any]:
    """Run one question and keep the service's response, recording any failure instead of raising."""
    record: dict[str, Any] = {
        "dataset": question.dataset,
        "question_id": question.id,
        "question": question.question,
    }
    try:
        started = time.monotonic()
        body = search(question.question)
        record["latency_seconds"] = round(time.monotonic() - started, 2)
        if "error" in body:
            return {**record, "status": "error", "error": str(body["error"]), "response": body}
        return {**record, "status": "ok", "error": None, "response": body}
    except Exception as exc:
        return {**record, "status": "error", "error": f"{type(exc).__name__}: {exc}"}


def compute_percentile(sorted_values: list[float], percent: int) -> float:
    """Nearest-rank percentile of already sorted values."""
    return sorted_values[max(0, math.ceil(percent / 100 * len(sorted_values)) - 1)]


def summarize_token_counts(counts: list[dict[str, int | None]], names: tuple[str, ...]) -> dict:
    """Total, mean, p50 and p99 per question of each count, over the questions that reported it."""
    summary = {}
    for name in names:
        known = sorted(c[name] for c in counts if c.get(name) is not None)
        summary[name] = {
            "total": sum(known),
            "mean": round(sum(known) / len(known), 1) if known else None,
            "p50": compute_percentile(known, 50) if known else None,
            "p99": compute_percentile(known, 99) if known else None,
            "unknown": len(counts) - len(known),
        }
    return summary


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Latency, agent behavior and token use over the answered questions."""
    answered = [r for r in records if r["status"] == "ok"]
    summary: dict[str, Any] = {
        "questions": len(records),
        "answered": len(answered),
        "errors": len(records) - len(answered),
    }
    if not answered:
        return summary
    responses = [r["response"] for r in answered]
    latencies = sorted(r["latency_seconds"] for r in answered)
    tool_calls: Counter[str] = Counter()
    for response in responses:
        tool_calls.update(response.get("tool_calls", {}))
    summary.update(
        latency_seconds={
            "mean": round(statistics.fmean(latencies), 2),
            **{f"p{p}": compute_percentile(latencies, p) for p in (50, 90, 95, 99)},
            "max": latencies[-1],
        },
        mean_turns=round(statistics.fmean(r.get("turns") or 0 for r in responses), 2),
        mean_tool_calls={
            name: round(count / len(answered), 2) for name, count in sorted(tool_calls.items())
        },
        terminal_reasons=dict(Counter(r.get("terminal_reason") for r in responses)),
        partial=sum(bool(r.get("partial")) for r in responses),
        agent_tokens=summarize_token_counts(
            # A service without usage reporting leaves every count unknown.
            [r.get("usage") or {} for r in responses],
            AGENT_TOKEN_FIELDS,
        ),
    )
    return summary


def write_run_file(path: Path, metadata: dict[str, Any], records: list[dict[str, Any]]) -> None:
    """Write the run's settings, per-dataset summary and records, replacing the file at once."""
    records = sorted(records, key=lambda r: (r["dataset"], r["question_id"]))
    summary = {
        dataset: summarize_records([r for r in records if r["dataset"] == dataset])
        for dataset in sorted({r["dataset"] for r in records})
    }
    summary["all"] = summarize_records(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_suffix(".partial")
    partial_path.write_text(
        json.dumps({"metadata": metadata, "summary": summary, "results": records}, indent=2)
    )
    partial_path.replace(path)


def print_summary(path: Path) -> None:
    """Print each dataset's answered count, p50/p99 latency and mean turns."""
    summary = json.loads(path.read_text())["summary"]
    print(f"{'dataset':<18}{'answered':>10}{'p50 s':>8}{'p99 s':>8}{'turns':>8}")
    for dataset, values in summary.items():
        latency = values.get("latency_seconds", {})
        print(
            f"{dataset:<18}{values['answered']:>4}/{values['questions']:<5}"
            + "".join(
                f"{latency[p]:>8.1f}" if p in latency else f"{'-':>8}" for p in ("p50", "p99")
            )
            + (f"{values['mean_turns']:>8.2f}" if "mean_turns" in values else f"{'-':>8}")
        )
    print(f"Results: {path}")
    print(f"Score it with: python tests/benchmarks/evaluate.py {path}")


def read_git_commit() -> str:
    """The runner's commit, marked -dirty when there are uncommitted changes."""
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    return git("rev-parse", "--short", "HEAD") + ("-dirty" if git("status", "--porcelain") else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collection", choices=sorted(COLLECTIONS), help="benchmark collection whose questions to run")
    parser.add_argument("--container", help="Cosmos container holding the collection's corpus")
    parser.add_argument("--account-uri", help="Cosmos DB account URI, to run the service in this process")
    parser.add_argument("--database", help=f"database holding the container (default: {DEFAULTS['database']})")
    parser.add_argument("--questions-per-dataset", type=int, help="questions sampled from each dataset (default: all)")
    parser.add_argument("--seed", type=int, help=f"random seed for sampling questions (default: {DEFAULTS['seed']})")
    parser.add_argument("--toolkit-url", help="a deployed MCP toolkit's /mcp URL, to run through the toolkit instead")
    parser.add_argument("--toolkit-app-id", help="client ID of the toolkit's Entra app, used to sign in to the toolkit")
    parser.add_argument("--resume-file", type=Path, help="an earlier run's file; reruns only its failed or missing questions")
    parser.add_argument("--output-file", type=Path, help="where to write this run (default: .live-data/benchmark-runs/<collection>-<time>.json)")
    parser.add_argument("--workers", type=int, default=4, help="number of questions run in parallel (default: %(default)s)")
    args = parser.parse_args()

    if args.resume_file:
        given = [name for name in RUN_SETTINGS if getattr(args, name) is not None]
        if given:
            flags = ", ".join("--" + name.replace("_", "-") for name in given)
            parser.error(f"--resume-file reuses the earlier run's settings; remove {flags}")
        previous = json.loads(args.resume_file.read_text())
        settings = previous["metadata"]["settings"]
        kept = [r for r in previous["results"] if r["status"] == "ok"]
    else:
        settings = {name: getattr(args, name) for name in RUN_SETTINGS}
        for name, value in DEFAULTS.items():
            if settings[name] is None:
                settings[name] = value
        if settings["collection"] is None:
            parser.error("--collection is required")
        if settings["container"] is None:
            parser.error("--container is required")
        if (settings["account_uri"] is None) == (settings["toolkit_url"] is None):
            parser.error("give either --account-uri or --toolkit-url")
        if settings["toolkit_url"] and not settings["toolkit_app_id"]:
            parser.error("--toolkit-url needs --toolkit-app-id")
        kept = []

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_file or RUNS_DIR / f"{settings['collection']}-{stamp}.json"
    if output.exists():
        parser.error(f"{output} already exists")

    collection = COLLECTIONS[settings["collection"]]()
    questions = sample_questions_per_dataset(
        collection.read_questions(), settings["questions_per_dataset"], settings["seed"]
    )
    done = {(r["dataset"], r["question_id"]) for r in kept}
    pending = [q for q in questions if (q.dataset, q.id) not in done]
    metadata = {
        "settings": settings,
        "runner_commit": read_git_commit(),
        "resumed_from": str(args.resume_file) if args.resume_file else None,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    print(f"{len(pending)} questions to run, {len(kept)} kept from the earlier run")

    records = list(kept)
    started = time.monotonic()
    with ExitStack() as stack:
        if settings["toolkit_url"]:
            search = start_toolkit_search(settings)
        else:
            search = start_in_process_search(stack, settings)
        pool = ThreadPoolExecutor(args.workers)
        try:
            futures = [pool.submit(answer_question, search, q) for q in pending]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Answering"):
                records.append(future.result())
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
            metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
            metadata["duration_seconds"] = round(time.monotonic() - started, 1)
            write_run_file(output, metadata, records)
            print_summary(output)


if __name__ == "__main__":
    main()
