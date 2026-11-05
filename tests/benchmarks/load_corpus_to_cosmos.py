"""Load a benchmark collection's corpus into a Cosmos DB container.

Creates the database and container if they are missing (through the Azure CLI),
then writes the passages not yet in the container. With --force, every passage
is rewritten.

Example:
    python tests/benchmarks/load_corpus_to_cosmos.py --subscription <id> \
        --resource-group <rg> --account <cosmos-account> --collection hipporag2
"""

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

from azure.cosmos import CosmosClient
from azure.identity import AzureCliCredential
from tqdm import tqdm

from dataset_hipporag2 import HippoRAG2Collection

COLLECTIONS = {"hipporag2": HippoRAG2Collection}
TEXT_PATHS = ("/title", "/text")


def run_az_command(arguments: list[str]):
    """Run an Azure CLI command and return its parsed JSON output."""
    result = subprocess.run(
        ["az", *arguments, "--only-show-errors", "--output", "json"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout) if result.stdout.strip() else None


def create_container_if_missing(args, container_name: str) -> str:
    """Create the database and full-text-indexed container if needed; return the account endpoint."""
    scope = ["--subscription", args.subscription, "--resource-group", args.resource_group]
    account = run_az_command(["cosmosdb", "show", *scope, "--name", args.account])
    scope += ["--account-name", args.account]
    databases = run_az_command(["cosmosdb", "sql", "database", "list", *scope])
    if args.database not in {d["name"] for d in databases}:
        run_az_command(["cosmosdb", "sql", "database", "create", *scope, "--name", args.database])
    containers = run_az_command(
        ["cosmosdb", "sql", "container", "list", *scope, "--database-name", args.database]
    )
    if container_name not in {c["name"] for c in containers}:
        full_text_policy = {
            "defaultLanguage": "en-US",
            "fullTextPaths": [{"path": p, "language": "en-US"} for p in TEXT_PATHS],
        }
        indexing_policy = {
            "indexingMode": "consistent",
            "automatic": True,
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": [{"path": '/"_etag"/?'}],
            "fullTextIndexes": [{"path": p} for p in TEXT_PATHS],
        }
        run_az_command(
            [
                "cosmosdb", "sql", "container", "create", *scope,
                "--database-name", args.database,
                "--name", container_name,
                "--partition-key-path", "/id",
                "--full-text-policy", json.dumps(full_text_policy),
                "--idx", json.dumps(indexing_policy),
            ]
        )
    return account["documentEndpoint"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subscription", required=True)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--database", default="mcp-bench-v1")
    parser.add_argument("--collection", choices=sorted(COLLECTIONS), required=True)
    parser.add_argument("--force", action="store_true", help="rewrite passages that already exist")
    args = parser.parse_args()

    collection = COLLECTIONS[args.collection]()
    passages = collection.read_merged_corpus_passages()
    print(f"{collection.name}: {len(passages)} passages")

    endpoint = create_container_if_missing(args, collection.name)
    # Serverless accounts throttle bursts (HTTP 429); allow long retries instead of failing.
    client = CosmosClient(
        endpoint, credential=AzureCliCredential(), retry_total=30, retry_backoff_max=120
    )
    container = client.get_database_client(args.database).get_container_client(collection.name)

    if not args.force:
        existing_ids = set(
            container.query_items("SELECT VALUE c.id FROM c", enable_cross_partition_query=True)
        )
        passages = [p for p in passages if p.id not in existing_ids]
    print(f"Writing {len(passages)} passages")

    def write_passage(passage) -> None:
        container.upsert_item(
            {"id": passage.id, "title": passage.title, "text": passage.text, "datasets": list(passage.datasets)}
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(tqdm(pool.map(write_passage, passages), total=len(passages), desc="Writing passages"))

    count = next(iter(container.query_items("SELECT VALUE COUNT(1) FROM c", enable_cross_partition_query=True)))
    print(f"{args.database}/{collection.name} now holds {count} items")


if __name__ == "__main__":
    main()
