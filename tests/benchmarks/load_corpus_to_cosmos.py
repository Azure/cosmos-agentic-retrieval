"""Load a benchmark collection's corpus into a Cosmos DB container, with embeddings.

Creates the database and container if they are missing (through the Azure CLI),
with full-text indexes on the title and text and a vector index on the embedding
field, so every MCP toolkit search tool works on it. Then embeds the passages not
yet in the container with the given embedding deployment and writes them. A failed
batch is reported and the load continues; rerunning skips the passages already
loaded. With --force, every passage is embedded and rewritten. With
--document-ids-file, only the listed documents are loaded; a later run with a
longer list, or without one, adds the rest.

Example:
    python tests/benchmarks/load_corpus_to_cosmos.py --subscription <id> \
        --resource-group <rg> --account <cosmos-account> --collection hipporag2 \
        --container hipporag2-full \
        --embedding-base-url https://<foundry>.services.ai.azure.com/openai/v1 \
        --embedding-deployment text-embedding-3-small --embedding-field embedding_te3_small
"""

import argparse
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from azure.cosmos import CosmosClient
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import OpenAI
from tqdm import tqdm

from dataset_contracts import Passage
from dataset_enterprise_rag_bench import EnterpriseRAGBenchCollection
from dataset_hipporag2 import HippoRAG2Collection

COLLECTIONS = {
    "enterprise-rag-bench": EnterpriseRAGBenchCollection,
    "hipporag2": HippoRAG2Collection,
}
TEXT_PATHS = ("/title", "/text")
# Entra token audience for Azure OpenAI and Azure AI Foundry endpoints.
TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"
PASSAGES_PER_EMBEDDING_REQUEST = 100


def run_az_command(arguments: list[str]):
    """Run an Azure CLI command and return its parsed JSON output."""
    result = subprocess.run(
        ["az", *arguments, "--only-show-errors", "--output", "json"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout) if result.stdout.strip() else None


def create_container_if_missing(args, dimensions: int) -> str:
    """Create the database and the full-text and vector indexed container if needed; return the account endpoint."""
    scope = ["--subscription", args.subscription, "--resource-group", args.resource_group]
    account = run_az_command(["cosmosdb", "show", *scope, "--name", args.account])
    scope += ["--account-name", args.account]
    databases = run_az_command(["cosmosdb", "sql", "database", "list", *scope])
    if args.database not in {d["name"] for d in databases}:
        run_az_command(["cosmosdb", "sql", "database", "create", *scope, "--name", args.database])
    containers = run_az_command(
        ["cosmosdb", "sql", "container", "list", *scope, "--database-name", args.database]
    )
    if args.container not in {c["name"] for c in containers}:
        vector_path = f"/{args.embedding_field}"
        full_text_policy = {
            "defaultLanguage": "en-US",
            "fullTextPaths": [{"path": p, "language": "en-US"} for p in TEXT_PATHS],
        }
        vector_policy = {
            "vectorEmbeddings": [
                {
                    "path": vector_path,
                    "dataType": "float32",
                    "dimensions": dimensions,
                    "distanceFunction": "cosine",
                }
            ]
        }
        indexing_policy = {
            "indexingMode": "consistent",
            "automatic": True,
            "includedPaths": [{"path": "/*"}],
            # The vector is searched through its vector index only.
            "excludedPaths": [{"path": '/"_etag"/?'}, {"path": f"{vector_path}/*"}],
            "fullTextIndexes": [{"path": p} for p in TEXT_PATHS],
            "vectorIndexes": [{"path": vector_path, "type": "diskANN"}],
        }
        run_az_command(
            [
                "cosmosdb", "sql", "container", "create", *scope,
                "--database-name", args.database,
                "--name", args.container,
                "--partition-key-path", "/id",
                "--full-text-policy", json.dumps(full_text_policy),
                "--vector-embeddings", json.dumps(vector_policy),
                "--idx", json.dumps(indexing_policy),
            ]
        )
    return account["documentEndpoint"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subscription", required=True, help="Azure subscription ID of the Cosmos DB account")
    parser.add_argument("--resource-group", required=True, help="resource group of the Cosmos DB account")
    parser.add_argument("--account", required=True, help="Cosmos DB account name")
    parser.add_argument("--database", default="mcp-bench-v1", help="database to load into, created if missing (default: %(default)s)")
    parser.add_argument("--collection", choices=sorted(COLLECTIONS), required=True, help="benchmark collection whose corpus to load")
    parser.add_argument("--container", required=True, help="container to load into, created if missing")
    parser.add_argument(
        "--embedding-base-url",
        required=True,
        help="OpenAI-compatible endpoint of the embedding model, e.g. https://<foundry>.services.ai.azure.com/openai/v1; signed in with az login",
    )
    parser.add_argument("--embedding-deployment", required=True, help="embedding model deployment name, e.g. text-embedding-3-small")
    parser.add_argument(
        "--embedding-field",
        required=True,
        help="document field that holds the vector; name it after the model, e.g. embedding_te3_small",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help=f"number of threads working in parallel; each embeds and writes {PASSAGES_PER_EMBEDDING_REQUEST} passages at a time (default: %(default)s)",
    )
    parser.add_argument(
        "--document-ids-file",
        type=Path,
        help="file with one document ID per line; load only those documents (default: the whole corpus)",
    )
    parser.add_argument("--force", action="store_true", help="rewrite passages that already exist")
    args = parser.parse_args()

    collection = COLLECTIONS[args.collection]()
    documents = collection.read_merged_corpus_documents()
    if args.document_ids_file:
        wanted = {line.strip() for line in args.document_ids_file.read_text().splitlines() if line.strip()}
        missing = wanted - {d.id for d in documents}
        if missing:
            raise SystemExit(
                f"{len(missing)} of {len(wanted)} listed document IDs are not in {collection.name}, "
                f"e.g. {sorted(missing)[:5]}"
            )
        documents = [d for d in documents if d.id in wanted]
    passages = [p for d in documents for p in d.get_passages()]
    print(f"{collection.name}: {len(documents)} documents, {len(passages)} passages")

    embedder = OpenAI(
        base_url=args.embedding_base_url,
        api_key=get_bearer_token_provider(AzureCliCredential(), TOKEN_SCOPE),
        # The embedding deployment's tokens-per-minute limit paces the load.
        max_retries=20,
    )

    def embed_passages(batch: list[Passage]) -> list[list[float]]:
        response = embedder.embeddings.create(
            model=args.embedding_deployment, input=[f"{p.title}\n{p.text}" for p in batch]
        )
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

    dimensions = len(embed_passages(passages[:1])[0])
    print(f"{args.embedding_deployment}: {dimensions} dimensions in /{args.embedding_field}")
    endpoint = create_container_if_missing(args, dimensions)
    # Serverless accounts throttle bursts (HTTP 429); allow long retries instead of failing.
    client = CosmosClient(
        endpoint, credential=AzureCliCredential(), retry_total=30, retry_backoff_max=120
    )
    container = client.get_database_client(args.database).get_container_client(args.container)

    corpus_size = len(passages)
    if not args.force:
        existing_ids = set(
            container.query_items("SELECT VALUE c.id FROM c", enable_cross_partition_query=True)
        )
        passages = [p for p in passages if p.id not in existing_ids]
    skipped = corpus_size - len(passages)
    print(f"Writing {len(passages)} passages; skipping {skipped} already in the container")

    def embed_and_write_passages(batch: list[Passage]) -> str | None:
        """Embed and write one batch; return the error instead of raising, so other batches continue."""
        try:
            for passage, embedding in zip(batch, embed_passages(batch), strict=True):
                container.upsert_item({**asdict(passage), args.embedding_field: embedding})
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"[:200]
        return None

    batches = [
        passages[start : start + PASSAGES_PER_EMBEDDING_REQUEST]
        for start in range(0, len(passages), PASSAGES_PER_EMBEDDING_REQUEST)
    ]
    errors: Counter[str] = Counter()
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, tqdm(total=len(passages), desc="Writing passages") as progress:
        for batch, error in zip(batches, pool.map(embed_and_write_passages, batches)):
            progress.update(len(batch))
            if error:
                errors[error] += 1
                failed += len(batch)
                progress.set_postfix(failed_batches=errors.total())

    count = next(iter(container.query_items("SELECT VALUE COUNT(1) FROM c", enable_cross_partition_query=True)))
    # Some passages of a failed batch may have been written before the error.
    print(
        f"Loaded {len(passages) - failed}, skipped {skipped} already in the container, "
        f"failed {failed} ({errors.total()} batches); "
        f"{args.database}/{args.container} now holds {count} items"
    )
    if errors:
        print("Rerun the same command to load the failed passages. Errors:")
        for error, batch_count in errors.most_common():
            print(f"  {batch_count} x {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
