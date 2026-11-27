# Benchmarks

Measures agentic search on three collections:
- [HippoRAG 2](https://huggingface.co/datasets/osunlp/HippoRAG_2): 1,000
  questions each from 2WikiMultihopQA, HotpotQA, MuSiQue, Natural Questions and
  PopQA, over one shared corpus of 45,881 passages.
- [EnterpriseRAG-Bench](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench)
  (ERB): 500 questions in 10 categories over 511,962 internal documents of a
  simulated company, from tools such as Slack, Gmail, Jira and Confluence.
- [Harvey LAB Law Firm Knowledge](https://github.com/harveyai/harvey-labs/tree/7f9b1ee19f9224576a5a5143c4e6f75842ac7020/tasks/firm-knowledge)
  (Harvey): 250 research requests over the document store of a fictional law
  firm, 9,288 Word, Excel, email and PowerPoint files across 266 matters.

## Setup

Install the benchmark extras and sign in with `az login`. Then load the corpus
into a Cosmos DB account with full-text and vector search enabled. This creates
the database and container if needed, with full-text indexes on the title and
text and a vector index on the embedding, then embeds and writes the passages
not yet loaded. Your `az login` needs access to call the embedding deployment:

```bash
python -m pip install -e '.[benchmark]'
python tests/benchmarks/load_corpus_to_cosmos.py --subscription <id> \
    --resource-group <rg> --account <account> --collection hipporag2 \
    --container hipporag2-full \
    --embedding-base-url https://<foundry>.services.ai.azure.com/openai/v1 \
    --embedding-deployment text-embedding-3-small --embedding-field embedding_te3_small
```

To load part of a corpus, pass `--document-ids-file` with one document ID per
line. Running again with a longer list, or without one, adds the remaining
documents; passages already in the container are skipped.

ERB documents are long, so each is stored as chunks of up to 512 tokens with
128 tokens of overlap (`ERBDocument.CHUNK_MAX_TOKENS` and
`CHUNK_OVERLAP_TOKENS`); the whole corpus is about 1.73M chunks. Each chunk
keeps its document's ERB `doc_id`. The items don't record the chunk settings,
so name the container after them, e.g. `--collection enterprise-rag-bench
--container enterprise-rag-bench-c512-o128`. Loading it into a serverless
account took about 18 hours with 4 to 8 workers, as Cosmos DB throttles the
writes; a rerun loads only what is missing.

Harvey files are converted to text the way Harvey LAB's agent reads them, and
chunked the same way, e.g. `--collection harvey-firm-knowledge --container
harvey-firm-knowledge-c512-o128`. The benchmark extras include the pandoc used
for Word files. The whole corpus is about 318K chunks and took about 4 hours
to load.

## Run

`run.py` sends every question to agentic search and saves the responses;
`evaluate.py` then scores them. A model endpoint is required: set the model as
environment variables in your shell first; these are the same `LLM_*` variables
the service reads:

```bash
export LLM_BASE_URL='https://<resource>.services.ai.azure.com/openai/v1'
export LLM_MODEL='<deployment-name>'
export LLM_API_KEY='<key>'
```

There are two ways to run the questions.

**In-process** (default). `run.py` starts the service inside its own process. Its
agent searches the corpus container you loaded above through the MCP toolkit at
`TOOLKIT_MCP_URL` (default `http://127.0.0.1:8080/mcp`), signed in with your
`az login`. The model variables are used for the agent:

```bash
python tests/benchmarks/run.py --toolkit-app-id <toolkit-entra-app-client-id> \
    --collection hipporag2 --container hipporag2-full --questions-per-dataset 100 --workers 8
```

**Through a deployed toolkit**. `run.py` calls the toolkit's `agentic_search` MCP
tool, signed in with your `az login`. The agent runs in the deployed retriever
with that deployment's own configuration (model and turn limit); the model
variables are not used:

```bash
python tests/benchmarks/run.py --toolkit-url https://<toolkit-app>/mcp \
    --toolkit-app-id <toolkit-entra-app-client-id> \
    --collection hipporag2 --container hipporag2-full --questions-per-dataset 100 --workers 8
```

Without `--questions-per-dataset`, all 5,000 questions run. For ERB, pass
`--collection enterprise-rag-bench --container enterprise-rag-bench-c512-o128`;
`--questions-per-dataset` then picks that many questions from each of its 10
categories, and without it all 500 run. For Harvey, pass `--collection
harvey-firm-knowledge --container harvey-firm-knowledge-c512-o128`; without
`--questions-per-dataset` all 250 run. Its tasks are long, so use about 4
workers to stay within the model's rate limit.

Each run writes one JSON file under `.live-data/benchmark-runs/` with its
settings, a per-dataset summary (latency, turns, tool calls and agent token
use) and each question's response. A failed question is recorded and the run
continues; `--resume-file <run>.json` reruns only failed or missing questions,
with the earlier run's settings, into a new file.

## Evaluate

`evaluate.py` scores a saved run with the collection's evaluator, using the
model variables as the judge, and writes `<run>.scores.json` next to the run:

```bash
python tests/benchmarks/evaluate.py .live-data/benchmark-runs/<run>.json
```

The HippoRAG 2 evaluator reports:
- **Judge accuracy**: the judge model decides whether the answer matches a
  reference answer.
- **Gold passage recall**: share of the question's supporting passages among
  the returned documents.
- **Exact match** and **F1**: standard SQuAD-style word matching. The agent
  answers in sentences, so these understate correct answers.

The ERB evaluator follows ERB's metrics-based evaluation, with its prompts,
and reports each score per category too:
- **correct**: the judge decides whether the answer is aligned with the
  reference answer.
- **completeness**: share of the question's answer facts the judge finds in
  the answer.
- **overall**: completeness when correct, otherwise 0; ERB's headline score.
- **returned_document_recall**: share of the question's gold documents among
  the returned chunks' documents. Not scored for the 30 questions without gold
  documents.
- **returned_invalid_extra_documents**: number of returned documents that are
  not gold.
- **cited_document_recall**: share of the question's gold documents among the
  documents the answer cites, as ERB intends. Not scored for responses that
  don't mark cited documents.

The scores come from our own evaluator and judge model, without ERB's step
that corrects gold documents and answers. They are not comparable with the ERB
leaderboard.

The Harvey evaluator grades each of a task's rubric criteria with Harvey LAB's
judge prompt, and reports:
- **correct**: every criterion passes; Harvey LAB's headline score.
- **criteria_passed**: share of the task's criteria that pass.
- **returned_matter_recall** and **cited_matter_recall**: share of the matters
  the criteria name among the matters of the returned or cited documents. Not
  scored for the 7 tasks whose criteria name no matter.

These scores are not comparable with Harvey LAB's: our judge model grades the
agent's text answer, where Harvey LAB grades a `response.md` file the agent
writes. Matter recall is our own addition.
