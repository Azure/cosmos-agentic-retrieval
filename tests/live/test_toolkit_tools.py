"""Check our toolkit tool definitions against a running MCP toolkit.

Our definitions in `agent/toolkit_tools.py` copy the toolkit's tool names and
arguments. This test fails if the toolkit no longer has one of those tools, no
longer takes one of the arguments we offer (or takes it with another type), or
requires an argument we don't offer. Run it when the toolkit changes:

    RUN_TOOLKIT_LIVE=1 TOOLKIT_TEST_URL=https://<app>/mcp TOOLKIT_TEST_APP_ID=<id> \
        python -m pytest tests/live/test_toolkit_tools.py -q

``TOOLKIT_TEST_APP_ID`` is the toolkit's Entra app client ID; the test signs in
with `az login`. Leave it out for a local toolkit running without sign-in.
"""

from __future__ import annotations

import os

import pytest

from cosmos_agentic_retriever.agent.toolkit_client import ToolkitClient, toolkit_sessions
from cosmos_agentic_retriever.agent.toolkit_tools import RETURN_DOCUMENTS, TOOLKIT_TOOLS

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(
        os.environ.get("RUN_TOOLKIT_LIVE") != "1",
        reason="set RUN_TOOLKIT_LIVE=1 and TOOLKIT_TEST_URL to compare with a running toolkit",
    ),
]

OUR_ONLY_ARGUMENTS = {"returnDocuments": RETURN_DOCUMENTS}


async def test_our_definitions_match_the_toolkit():
    url = os.environ["TOOLKIT_TEST_URL"]
    app_id = os.environ.get("TOOLKIT_TEST_APP_ID")
    from azure.identity.aio import AzureCliCredential, get_bearer_token_provider

    async with AzureCliCredential() as credential:
        if app_id:
            get_token = get_bearer_token_provider(credential, f"api://{app_id}/.default")
        else:

            async def get_token() -> str:
                return ""

        async with toolkit_sessions(url, get_token) as open_session:
            toolkit = {tool.name: tool.inputSchema for tool in await ToolkitClient(open_session).list_tools()}

    problems = []
    for spec in TOOLKIT_TOOLS:
        schema = toolkit.get(spec.name)
        if schema is None:
            problems.append(f"{spec.name}: the toolkit has no such tool")
            continue
        theirs = schema.get("properties", {})
        for name, ours in spec.properties.items():
            if name in OUR_ONLY_ARGUMENTS:
                continue
            if name not in theirs:
                problems.append(f"{spec.name}.{name}: the toolkit doesn't take this argument")
            elif theirs[name].get("type") != ours["type"]:
                problems.append(
                    f"{spec.name}.{name}: type {ours['type']} here, {theirs[name].get('type')} in the toolkit"
                )
        for name in schema.get("required", []):
            if name not in spec.properties:
                problems.append(f"{spec.name}.{name}: required by the toolkit but not offered here")
            elif name not in spec.required:
                problems.append(f"{spec.name}.{name}: required by the toolkit but optional here")

    assert not problems, "Toolkit tools differ from agent/toolkit_tools.py:\n" + "\n".join(problems)
