"""Convert Python search arguments into Cosmos DB NoSQL queries.

This module (a) exposes a restricted set of searches methods into python and (b)
converts these to safe, valid database queries. Each method returns a
CompiledCosmosQuery containing the SQL command and its parameter values.  To
execute it, import CosmosExecutor from cosmos_agentic_retriever.query_engine,
construct it with QueryEngineConfig, and pass the compiled query and a Cosmos
container client to run(). Reuse the same executor across calls that need one
combined query limit, including calls to different containers.

The following query methods and filters are supported. The compile_* methods
belong to CosmosQueryCompiler; the filter types are in query_engine.types:
- compile_vector: build a query that ranks items by distance from a supplied
    query vector.
- compile_full_text: build a query that ranks items by relevance to terms in
    specified text fields.
- compile_hybrid: build a query that combines vector and full-text rankings
    using reciprocal rank fusion (RRF).
- compile_structured: build a query that selects items using field filters,
    without vector or full-text ranking.
- compile_document_read: build a query that selects chunks belonging to a
    supplied document ID, up to a specified limit.
- EqualsFilter: require a field to equal a value, for example year = 2020.
- RangeFilter: apply an inclusive lower bound, upper bound, or both, for
    example 2020 <= year <= 2025.
- InFilter: require a field's value to be one of a supplied list, for example
    source in ["report", "article"].
- FilterExpression: the type representing any of the three filter types above.

Pass filter objects through the filters argument of compile_vector,
compile_full_text, compile_hybrid, or compile_structured. Multiple filters are
combined with AND. compile_document_read takes a document ID instead.

Usage Example: 
- Assumes  an existing database "example-db" and container "articles"
with items such as {"id": "article-1", "text": "Battery recycling...", "year": 2024}.
- The container must have a full-text policy and full-text index for /text.
- The caller is signed in (for example via `az login`) with Cosmos data-plane
read access; the account uses Entra identity, not account keys.
- This query searches across partitions and returns up to five ranked rows from 2020
onward. Returned rows use the compiler's aliases: item_id for /id, txt_0 for /text,
and add_0 for /year. The client is closed when the with block exits.

    from azure.cosmos import CosmosClient
    from azure.identity import AzureCliCredential
    from cosmos_agentic_retriever.query_engine import (
        CorpusSchema,
        CosmosExecutor,
        QueryEngineConfig,
    )
    from cosmos_agentic_retriever.query_engine.compiler import CosmosQueryCompiler
    from cosmos_agentic_retriever.query_engine.paths import CosmosPath
    from cosmos_agentic_retriever.query_engine.types import RangeFilter

    schema = CorpusSchema(
        item_id_path="/id",
        text_paths=["/text"],
        additional_return_paths=["/year"],
    )
    compiler = CosmosQueryCompiler(schema)
    compiled = compiler.compile_full_text(
        query="battery recycling",
        text_paths=[CosmosPath.parse("/text")],
        filters=[RangeFilter(path="/year", minimum=2020)],
        limit=5,
        ignored_item_ids=[],
        partition_key=None,
        cross_partition=True,
    )

    config = QueryEngineConfig(max_concurrency=8, slow_query_warning_seconds=4.5)
    executor = CosmosExecutor(config=config)

    with CosmosClient(
        "https://example-db-account.documents.azure.com:443/",
        credential=AzureCliCredential(),
    ) as client:
        database = client.get_database_client("example-db")
        container = database.get_container_client("articles")
        rows = executor.run(compiled, container=container)
        for row in rows:
            print(row["item_id"], row.get("txt_0", ""), row.get("add_0"))
"""

from __future__ import annotations

from cosmos_agentic_retriever.query_engine.executor import CosmosExecutor
from cosmos_agentic_retriever.query_engine.retriever import CorpusRetriever
from cosmos_agentic_retriever.query_engine.schema import CorpusSchema
from cosmos_agentic_retriever.query_engine.types import (
    PartitionQueryPolicy,
    QueryEngineConfig,
    RetrievalError,
    RetrievedItem,
    SearchRequest,
)

__all__ = [
    "CorpusRetriever",
    "CorpusSchema",
    "CosmosExecutor",
    "PartitionQueryPolicy",
    "QueryEngineConfig",
    "RetrievalError",
    "RetrievedItem",
    "SearchRequest",
]
