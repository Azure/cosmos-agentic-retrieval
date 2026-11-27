"""Multi-turn agentic retrieval over Azure Cosmos DB for NoSQL.

This module implements the `cosmos-retriever` service: an agent that answers a
question by calling the Azure Cosmos DB MCP Toolkit's tools (schema lookup,
text, vector and hybrid search, lookup by id) and returns the answer with the
documents it used.

Usage: While the module is designed to be used through the cosmos DB MCP
toolkit, we illustrate the main components with a direct usage pattern here

- Assume a toolkit is running at http://127.0.0.1:8080/mcp, and TOOLKIT_APP_ID
holds the client ID of its Entra app.
- Assume a database "example-db" with an "articles" container whose text field
has a full-text index.
- Assume LLM_BASE_URL and LLM_MODEL select a chat endpoint that supports tool calls.


	import os
	import uvicorn

	from cosmos_agentic_retriever.config import AgenticSearchConfig
	from cosmos_agentic_retriever.server import create_app

	settings = AgenticSearchConfig(
		llm_base_url=os.environ["LLM_BASE_URL"],
		llm_model=os.environ["LLM_MODEL"],
		llm_api_key=os.environ.get("LLM_API_KEY"),
		toolkit_mcp_url="http://127.0.0.1:8080/mcp",
		toolkit_app_id=os.environ["TOOLKIT_APP_ID"],
	)
	app = create_app(settings)
	uvicorn.run(app, host="127.0.0.1", port=9000)

From another terminal, ask a question about one database:

	curl --fail-with-body http://127.0.0.1:9000/agentic_search \
	  -H 'Content-Type: application/json' \
	  -d '{"query":"battery recycling","database":"example-db","maxDocuments":5}'

Add "container":"articles" to the JSON body to search only articles; omit
"database" to let the agent discover where to search. The response contains a
short answer, an explanation of how the agent arrived at it, both citing
documents by label, e.g. [D3], and references to the cited documents
(cite_as label, database, container and id; no text). A citation to a label
the agent was never given becomes [unverified source]. This example sets
maxDocuments to 5; requests can also set maxTurns, and returnedDocuments
("cited" or "cited_and_ranked"). Settings left out use the service's soft
limits, values above its hard limits are lowered to them, and the response
reports the values used in settings_used. If a tool call fails, the response
includes that error.
See the README to run the service and connect the MCP toolkit.
"""
