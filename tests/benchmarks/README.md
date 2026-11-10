# Benchmarks

Measures agentic search on the
[HippoRAG 2](https://huggingface.co/datasets/osunlp/HippoRAG_2) collection: 1,000
questions each from 2WikiMultihopQA, HotpotQA, MuSiQue, Natural Questions and
PopQA, over one shared corpus of 45,881 passages. 

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

## Run

`run.py` sends every question to agentic search, then scores the answer with a
judge model. A model endpoint is required: set the model as environment
variables in your shell first; these are the same `LLM_*` variables the service
reads:

```bash
export LLM_BASE_URL='https://<resource>.services.ai.azure.com/openai/v1'
export LLM_MODEL='<deployment-name>'
export LLM_API_KEY='<key>'
```

There are two ways to run the questions.

**In-process** (default). `run.py` starts the service inside its own process,
pointed at the corpus container you loaded above. The model variables are used
for both the agent and the judge, and Cosmos DB is reached with your `az login`:

```bash
python tests/benchmarks/run.py --account-uri https://<account>.documents.azure.com:443/ \
    --collection hipporag2 --container hipporag2-full --questions-per-dataset 100 --workers 8
```

**Through a deployed toolkit**. `run.py` calls the toolkit's `agentic_search` MCP
tool, signed in with your `az login`. The agent runs in the deployed retriever
with that deployment's own configuration (model, turn limit and containers);
the model variables are only used for the judge. The retriever must be
configured to search the corpus container:

```bash
python tests/benchmarks/run.py --toolkit-url https://<toolkit-app>/mcp \
    --toolkit-app-id <toolkit-entra-app-client-id> \
    --collection hipporag2 --container hipporag2-full --questions-per-dataset 100 --workers 8
```

Without `--questions-per-dataset`, all 5,000 questions run.

Each run writes one JSON file under `.live-data/benchmark-runs/` with its
settings, a per-dataset summary and a record per question. A failed question is
recorded and the run continues; `--resume-file <run>.json` reruns only failed or
missing questions, with the earlier run's settings, into a new file.

## Scores

- **Judge accuracy**: the judge model decides whether the answer matches a
  reference answer.
- **Gold passage recall**: share of the question's supporting passages among
  the returned documents.
- **Exact match** and **F1**: standard SQuAD-style word matching. The agent
  answers in sentences, so these understate correct answers.

The summary also reports latency percentiles, turns, tool calls, and agent and
judge token use.
