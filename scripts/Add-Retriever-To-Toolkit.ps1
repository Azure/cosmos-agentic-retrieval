#Requires -Version 7.0
<#
.SYNOPSIS
    Adds the cosmos-agentic-retrieval service to an existing Azure Cosmos DB MCP Toolkit Container App.

.DESCRIPTION
    Adds the agentic retrieval service to an existing Azure Cosmos DB MCP Toolkit
    Container App, as a second container named "retriever". The toolkit's
    agentic_search tool sends its requests to this container.

    The script:
    1. Builds the service's Docker image with the configuration file
       (-ConfigFile) copied inside, and pushes it to the app's container registry.
    2. Gives the app's managed identity the Cognitive Services OpenAI User role
       on the Azure OpenAI or Azure AI Foundry resource (-ModelResourceId). That
       resource hosts the chat model set as llm_model in the configuration, which
       the service uses to plan searches and write answers.
    3. Gives the app's managed identity the Mcp.Tool.Executor role on the
       toolkit's Entra app, so the service can call the toolkit's MCP tools.
       This needs an owner of that Entra app; the toolkit's deployment makes
       whoever deployed it an owner.
    4. Adds the "retriever" container to the app, or updates it if it exists,
       and passes it the toolkit's Entra app ID (TOOLKIT_APP_ID).
    5. Turns on the toolkit's agentic_search tool: if the toolkit container
       does not set ENABLE_AGENTIC_SEARCH, sets it to true. If it is set to
       anything other than true, the script stops before changing anything.

    The service authenticates with the chat model and the toolkit with the
    app's managed identity, so the configuration file must not contain a key
    (llm_api_key).

.EXAMPLE
    .\scripts\Add-Retriever-To-Toolkit.ps1 -ResourceGroup my-rg -ContainerAppName my-mcp-toolkit -ConfigFile .\config.deploy.yaml -ModelResourceId /subscriptions/.../accounts/my-foundry

.PARAMETER ModelResourceId
    Resource ID of the Azure OpenAI or Azure AI Foundry resource that hosts the
    chat model set as llm_model in the configuration, for example
    /subscriptions/<id>/resourceGroups/<rg>/providers/Microsoft.CognitiveServices/accounts/<name>.

.PARAMETER ImageTag
    Tag of the pushed image. Defaults to the current time, so each run deploys
    a new revision.

.PARAMETER PipIndexUrl
    Package index for the image build, for networks that block PyPI and
    provide a mirror instead. Defaults to PyPI.
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$ResourceGroup,

    [Parameter(Mandatory=$true)]
    [string]$ContainerAppName,

    [Parameter(Mandatory=$true)]
    [string]$ConfigFile,

    [Parameter(Mandatory=$true)]
    [string]$ModelResourceId,

    [Parameter(Mandatory=$false)]
    [string]$ImageTag = (Get-Date -Format "yyyyMMdd-HHmmss"),

    [Parameter(Mandatory=$false)]
    [string]$PipIndexUrl = ""
)

$ErrorActionPreference = "Stop"

# Check the configuration has no keys: it is copied into the image, where anyone who can pull the image can read it
$configPath = (Resolve-Path $ConfigFile).Path
if (Select-String -Path $configPath -Pattern '^\s*llm_api_key\s*:' -Quiet) {
    throw "'$ConfigFile' contains llm_api_key. Remove it; the service signs in with the app's managed identity."
}

# Read the toolkit app, and check it is a toolkit app with a managed identity
$app = az containerapp show --name $ContainerAppName --resource-group $ResourceGroup | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw "Reading Container App '$ContainerAppName' failed" }

# The toolkit's deployment names its container after the app
if ($ContainerAppName -notin $app.properties.template.containers.name) {
    throw "App '$ContainerAppName' has no container named '$ContainerAppName'. Is it the right name for the MCP toolkit app?"
}
$principalId = $app.identity.principalId
if (-not $principalId) { throw "App '$ContainerAppName' has no managed identity" }

# The toolkit's deployment stores its Entra app's client ID in the toolkit container's settings
$toolkitContainer = $app.properties.template.containers | Where-Object { $_.name -eq $ContainerAppName }
$toolkitAppId = ($toolkitContainer.env | Where-Object { $_.name -eq "AzureAd__ClientId" }).value
if (-not $toolkitAppId) { throw "The toolkit container has no AzureAd__ClientId setting. Was the toolkit deployed with Entra sign-in?" }

# The toolkit lists agentic_search only when ENABLE_AGENTIC_SEARCH is exactly "true".
# If it is not set, the script sets it below; if it is set to anything else, stop.
$enableSetting = ($toolkitContainer.env | Where-Object { $_.name -eq "ENABLE_AGENTIC_SEARCH" }).value
$searchEnabled = $enableSetting -ceq "true"
if (-not $searchEnabled -and $null -ne $enableSetting) {
    throw "The toolkit container sets ENABLE_AGENTIC_SEARCH to '$enableSetting', which turns agentic_search off. Set it to true, or remove it, then run the script again."
}

# Name the images: the pushed image goes to the app's own registry
$registry = $app.properties.configuration.registries[0].server
$image = "$registry/cosmos-agentic-retriever:$ImageTag"
$repoRoot = Split-Path -Parent $PSScriptRoot
$baseImage = "cosmos-agentic-retriever-base:$ImageTag"

# Build the service image (without configuration) from the repository's Dockerfile
az acr login --name $registry.Split('.')[0]
if ($LASTEXITCODE -ne 0) { throw "ACR login failed" }
$buildArgs = @()
if ($PipIndexUrl) { $buildArgs = @("--build-arg", "PIP_INDEX_URL=$PipIndexUrl") }
docker build --platform linux/amd64 @buildArgs -t $baseImage $repoRoot
if ($LASTEXITCODE -ne 0) { throw "Docker build failed" }

# Copy the configuration into the image (Dockerfile.config), then push the result to the registry
$configContext = Join-Path ([System.IO.Path]::GetTempPath()) "retriever-config-$([guid]::NewGuid())"
New-Item -ItemType Directory -Path $configContext | Out-Null
Copy-Item $configPath (Join-Path $configContext "config.yaml")
docker build --platform linux/amd64 --build-arg "BASE_IMAGE=$baseImage" `
    -f (Join-Path $repoRoot "Dockerfile.config") -t $image $configContext
$buildExitCode = $LASTEXITCODE
Remove-Item -Recurse $configContext
if ($buildExitCode -ne 0) { throw "Adding the configuration to the image failed" }
docker push $image
if ($LASTEXITCODE -ne 0) { throw "Docker push failed" }

# Let the app's managed identity call the chat model, unless it already can
$role = "Cognitive Services OpenAI User"
# Counted here, not with --query "length(@)": on Windows, az runs through cmd.exe, which rejects the parentheses
$assignments = az role assignment list --assignee $principalId --role $role --scope $ModelResourceId --output json | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw "Reading role assignments on '$ModelResourceId' failed" }
if (@($assignments).Count -eq 0) {
    az role assignment create --assignee-object-id $principalId --assignee-principal-type ServicePrincipal `
        --role $role --scope $ModelResourceId --output none
    if ($LASTEXITCODE -ne 0) { throw "Granting '$role' on '$ModelResourceId' failed" }
}

# Let the app's managed identity call the toolkit's MCP endpoint, unless it already can
$toolkitSp = az ad sp show --id $toolkitAppId --output json | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw "Reading the toolkit's Entra app '$toolkitAppId' failed" }
$toolRoleId = ($toolkitSp.appRoles | Where-Object { $_.value -eq "Mcp.Tool.Executor" }).id
if (-not $toolRoleId) { throw "The toolkit's Entra app '$toolkitAppId' has no Mcp.Tool.Executor role" }
$graphUrl = "https://graph.microsoft.com/v1.0/servicePrincipals/$principalId/appRoleAssignments"
$granted = (az rest --method GET --url $graphUrl --output json | ConvertFrom-Json).value
if ($LASTEXITCODE -ne 0) { throw "Reading the app's role assignments failed" }
if (-not ($granted | Where-Object { $_.resourceId -eq $toolkitSp.id -and $_.appRoleId -eq $toolRoleId })) {
    # Sent from a file: on Windows, az runs through cmd.exe, which mangles inline JSON
    $bodyFile = [System.IO.Path]::GetTempFileName()
    @{ principalId = $principalId; resourceId = $toolkitSp.id; appRoleId = $toolRoleId } |
        ConvertTo-Json | Out-File -FilePath $bodyFile -Encoding utf8 -NoNewline
    az rest --method POST --url $graphUrl --headers "Content-Type=application/json" --body "@$bodyFile" --output none
    $grantExitCode = $LASTEXITCODE
    Remove-Item $bodyFile
    if ($grantExitCode -ne 0) {
        throw "Granting Mcp.Tool.Executor on the toolkit's Entra app failed. Run the script as an owner of that app."
    }
}

# Add the "retriever" container to the app, or switch an existing one to the new image
az containerapp update --name $ContainerAppName --resource-group $ResourceGroup `
    --container-name retriever --image $image --cpu 0.25 --memory 0.5Gi `
    --set-env-vars "TOOLKIT_APP_ID=$toolkitAppId" --output none
if ($LASTEXITCODE -ne 0) { throw "Adding the retriever container failed" }

# Turn on the toolkit's agentic_search tool, unless it is already on
if (-not $searchEnabled) {
    az containerapp update --name $ContainerAppName --resource-group $ResourceGroup `
        --container-name $ContainerAppName --set-env-vars "ENABLE_AGENTIC_SEARCH=true" --output none
    if ($LASTEXITCODE -ne 0) { throw "Turning on agentic_search in the toolkit failed" }
}

Write-Host "Retriever added to '$ContainerAppName'. New roles can take a few minutes to take effect. To see the retriever's logs:"
Write-Host "  az containerapp logs show --name $ContainerAppName --resource-group $ResourceGroup --container retriever --tail 50"
