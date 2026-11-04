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
3. An OpenAI-compatible chat model endpoint reachable from Azure.
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
llm_base_url: https://YOUR-ENDPOINT/v1
llm_model: YOUR-MODEL
llm_api_key: YOUR-KEY            # only if the endpoint needs a key
cosmos_containers:
  YOUR-CONTAINER:
    cosmos_schema:
      item_id_path: /id
      partition_key_paths: ["/YOUR-PARTITION-KEY"]
      text_paths: ["/YOUR-TEXT-FIELD"]
```

### Step 3: Add the service to the toolkit app

Use the resource group and Container App name of an existing Azure MCP Tool Kit App:

```powershell
.\scripts\Add-Retriever-To-Toolkit.ps1 -ResourceGroup <rg> -ContainerAppName <app> `
  -ConfigFile .\config.deploy.yaml
```

The script builds the service image, pushes it to the toolkit's registry,
stores the configuration as a Container App secret, and adds the service as a
second container in the app. 

### Step 4: Try it

Connect an MCP client such as VS Code to the toolkit, as described in the
[toolkit's README](https://github.com/AzureCosmosDB/MCPToolKit#vs-code-configuration),
and ask a question that uses `agentic_search`. 


## Updating the configuration

To change the configuration, update the secret and restart the app:

```powershell
az containerapp secret set --name <app> --resource-group <rg> `
  --secrets "retriever-config=@config.deploy.yaml"
az containerapp revision restart --name <app> --resource-group <rg> `
  --revision (az containerapp show --name <app> --resource-group <rg> --query properties.latestRevisionName -o tsv)
```

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
