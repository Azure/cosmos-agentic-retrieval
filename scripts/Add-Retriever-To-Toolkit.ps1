<#
.SYNOPSIS
    Adds the cosmos-agentic-retrieval service to an existing Azure Cosmos DB MCP Toolkit Container App.

.DESCRIPTION
    This script builds, configures and adds the cosmos-agentic-retriever image
    to an existing Azure Cosmos DB MCP Toolkit Container App. Concretely, it
    creates a new image for the retriever service, pushes it to the toolkit's
    registry, stores the configuration file (keys included) as a Container App
    secret, then adds or updates a "retriever" container in the toolkit's app
    that reads it as a file. The toolkit's own container is not changed.

.EXAMPLE
    .\scripts\Add-Retriever-To-Toolkit.ps1 -ResourceGroup my-rg -ContainerAppName my-mcp-toolkit -ConfigFile .\config.deploy.yaml
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$ResourceGroup,

    [Parameter(Mandatory=$true)]
    [string]$ContainerAppName,

    [Parameter(Mandatory=$true)]
    [string]$ConfigFile,

    [Parameter(Mandatory=$false)]
    [string]$ImageTag = "latest"
)

$ErrorActionPreference = "Stop"

$configPath = (Resolve-Path $ConfigFile).Path
$app = az containerapp show --name $ContainerAppName --resource-group $ResourceGroup | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw "Reading Container App '$ContainerAppName' failed" }

# The toolkit's deployment names its container after the app
if ($ContainerAppName -notin $app.properties.template.containers.name) {
    throw "App '$ContainerAppName' has no container named '$ContainerAppName'. Is it the right name for the MCP toolkit app?"
}

$registry = $app.properties.configuration.registries[0].server
$image = "$registry/cosmos-agentic-retriever:$ImageTag"

az acr login --name $registry.Split('.')[0]
if ($LASTEXITCODE -ne 0) { throw "ACR login failed" }
docker build --platform linux/amd64 -t $image (Split-Path -Parent $PSScriptRoot)
if ($LASTEXITCODE -ne 0) { throw "Docker build failed" }
docker push $image
if ($LASTEXITCODE -ne 0) { throw "Docker push failed" }

# "@path" makes the Azure CLI read the value from the file
az containerapp secret set --name $ContainerAppName --resource-group $ResourceGroup `
    --secrets "retriever-config=@$configPath" --output none
if ($LASTEXITCODE -ne 0) { throw "Setting the configuration secret failed" }

# Mounts the app's secrets as files named after each secret
az containerapp update --name $ContainerAppName --resource-group $ResourceGroup `
    --container-name retriever --image $image --secret-volume-mount /mnt/config `
    --set-env-vars COSMOS_RETRIEVER_CONFIG_FILE=/mnt/config/retriever-config --output none
if ($LASTEXITCODE -ne 0) { throw "Adding the retriever container failed" }

Write-Host "Retriever added to '$ContainerAppName'. To see its logs:"
Write-Host "  az containerapp logs show --name $ContainerAppName --resource-group $ResourceGroup --container retriever --tail 50"
