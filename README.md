# Azure Cosmos DB Agentic Retrieval (Private Preview)

This repository provides an agentic-search service over configured Cosmos DB
containers. Users can ask an agent a natural-language question, which searches
the Cosmos DB containers and returns the answer with its supporting documents.

This feature is exposed as the `agentic_search` tool of the Azure Cosmos DB MCP
Toolkit. This feature is currently in preview.

## Prerequisites

1. The Azure Cosmos DB MCP Toolkit vX.Y.Z or later, deployed to Azure Container
   Apps.
2. Cosmos DB containers whose searched fields have a
   [full-text policy and index](https://learn.microsoft.com/azure/cosmos-db/gen-ai/full-text-search).
3. A chat model deployment in Azure OpenAI or Azure AI Foundry.
4. Tools: the same as the toolkit's deployment (Azure CLI, PowerShell 7+,
   Docker Desktop installed and running).

## Quick Start

### Step 1: Get the code

```powershell
git clone https://github.com/Azure/cosmos-agentic-retrieval.git
cd cosmos-agentic-retrieval
```

### Step 2: Configure 

```powershell
Copy-Item config.example.yaml config.deploy.yaml
```

In `config.deploy.yaml`, fill in your account, database, model, and containers.
The following are required:

```yaml
account_uri: https://YOUR-ACCOUNT.documents.azure.com:443/
cosmos_database: YOUR-DATABASE
llm_base_url: https://YOUR-ENDPOINT/openai/v1
llm_model: YOUR-MODEL
cosmos_containers:
  YOUR-CONTAINER:
    cosmos_schema:
      item_id_path: /id
      partition_key_paths: ["/YOUR-PARTITION-KEY"]
      text_paths: ["/YOUR-TEXT-FIELD"]
```

The service signs in to Cosmos DB and the model with the app's managed identity.
It is built into the service's image; the script refuses a file with
`cosmos_key` or `llm_api_key`.

### Step 3: Add the service to the toolkit app

Use the resource group and Container App name of an existing Azure MCP Tool Kit
app, and the resource ID of the Azure OpenAI or Azure AI Foundry resource that
hosts the model:

```powershell
.\scripts\Add-Retriever-To-Toolkit.ps1 -ResourceGroup <rg> -ContainerAppName <app> `
  -ConfigFile .\config.deploy.yaml -ModelResourceId <model-resource-id>
```

The script builds the service image with the configuration inside, pushes it to
the toolkit's registry, lets the app's managed identity call the model, and adds
the service as a second container in the app. On networks that block PyPI, add
`-PipIndexUrl <mirror>` to install packages from a mirror.

### Step 4: Try it

Connect an MCP client such as VS Code to the toolkit, as described in the
[toolkit's README](https://github.com/AzureCosmosDB/MCPToolKit#vs-code-configuration),
and ask a question that uses `agentic_search`. 


## Updating the configuration

To change the configuration, edit `config.deploy.yaml` and run the script from
Step 3 again. It builds a new image and replaces the service's container.

## Limitations

- *Full-text search only.* Vector and hybrid search are not supported yet.
- *Read-only.* The service never creates or changes databases, containers,
  indexes, or documents.
- *Shares access control with MCP Toolkit.* Only callers allowed to use
`agentic_search` through the MCP Toolkit reach the service, which listens only
inside the app. 

## Run Locally for Testing

Running the service on your own machine is for testing and development only,
not for serving users.

```bash
python -m pip install .
cp config.example.yaml config.local.yaml    # fill in as in Step 2
az login
export COSMOS_RETRIEVER_CONFIG_FILE=./config.local.yaml
python -m cosmos_agentic_retriever serve
```

The service listens on `http://127.0.0.1:9000`. A toolkit running on the same
machine finds it there by default; see the toolkit's
[local development guide](https://github.com/AzureCosmosDB/MCPToolKit/blob/main/LOCAL_DEVELOPMENT.md).

## Benchmarks

Results on the [HippoRAG 2](https://huggingface.co/datasets/osunlp/HippoRAG_2)
collection: 1,000 questions each from five question-answering datasets, over one
shared corpus of 45,881 passages. Run  with `gpt-5.6-luna` as agent and judge,
full-text search only, at most 6 model turns and 10 documents per question. To
run it yourself, see [tests/benchmarks](tests/benchmarks/README.md).

| Dataset | Judge accuracy | Gold passage recall | p50 latency | p99 latency |
|---|---|---|---|---|
| 2WikiMultihopQA | 0.871 | 0.937 | 6.0 s | 15.8 s |
| HotpotQA | 0.903 | 0.908 | 4.9 s | 17.9 s |
| MuSiQue | 0.620 | 0.685 | 10.7 s | 25.2 s |
| Natural Questions | 0.852 | 0.761 | 5.1 s | 18.3 s |
| PopQA | 0.815 | 0.514 | 3.7 s | 14.4 s |
| **All** | **0.812** | **0.761** | **5.6 s** | **21.6 s** |

- **Judge accuracy**: share of answers the judge model accepts as matching a
  reference answer.
- **Gold passage recall**: share of each question's supporting passages among the
  returned documents.
- Per question, the agent used 3.1 model turns, 2.5 searches and about 10k prompt
  and 230 completion tokens on average.
- 7.5% of questions reached the turn limit and returned no answer; this accounts
  for most misses, especially on MuSiQue.
