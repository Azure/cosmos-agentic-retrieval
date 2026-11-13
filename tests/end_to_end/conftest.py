"""Check prepared live fixtures read-only. Never provision, seed, or delete here."""

import os
from contextlib import ExitStack
from pathlib import Path

import pytest
from azure.cosmos import CosmosClient
from azure.identity import AzureCliCredential, DefaultAzureCredential
from setup_cosmos_live_tests import (
    DEFAULT_DATABASE,
    read_fixture_items,
    select_data,
    validate_container,
    validate_database_name,
    validate_items,
)


@pytest.fixture(scope="session")
def live_database():
    """Name of the prepared fixture database, after checking its containers and records."""
    archive = os.environ.get("COSMOS_TEST_SCIFACT_ARCHIVE")
    if not archive:
        pytest.fail("COSMOS_TEST_SCIFACT_ARCHIVE is required when RUN_COSMOS_LIVE=1")
    endpoint = os.environ.get("COSMOS_TEST_ENDPOINT")
    if not endpoint:
        pytest.fail("COSMOS_TEST_ENDPOINT is required when RUN_COSMOS_LIVE=1")
    name = validate_database_name(os.environ.get("COSMOS_TEST_DATABASE", DEFAULT_DATABASE))
    credential_name = os.environ.get("COSMOS_TEST_CREDENTIAL", "azure_cli")
    if credential_name not in ("azure_cli", "default"):
        pytest.fail("COSMOS_TEST_CREDENTIAL must be azure_cli or default")
    selected, _ = select_data(Path(archive))
    with ExitStack() as resources:
        credential = (
            AzureCliCredential() if credential_name == "azure_cli" else DefaultAzureCredential()
        )
        resources.callback(credential.close)
        client = CosmosClient(endpoint, credential=credential)
        resources.callback(client.close)
        database = client.get_database_client(name)
        for fixture in selected:
            container = database.get_container_client(fixture.name)
            validate_container(container.read(), fixture)
            validate_items(read_fixture_items(container, fixture), fixture)
    return name
