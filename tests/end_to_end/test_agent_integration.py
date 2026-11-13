"""Live agent checks against a real model and the prepared Cosmos containers.

Opt in with RUN_COSMOS_LIVE=1 (after tests/live/setup_cosmos_live_tests.py) and a
real OpenAI-compatible model in COSMOS_TEST_LLM_BASE_URL, COSMOS_TEST_LLM_MODEL,
and optionally COSMOS_TEST_LLM_API_KEY. Set RUN_COSMOS_MCP_LIVE=1 as well to run
the .NET MCP tests through a loopback Python server. Neither test provisions or
changes the stored data.
"""

import os
import shutil
import socket
import subprocess
from pathlib import Path
from threading import Event, Thread
from xml.etree import ElementTree

import pytest
import uvicorn
from fastapi.testclient import TestClient

from cosmos_agentic_retriever.config import RetrieverConfig
from cosmos_agentic_retriever.server import create_app

pytestmark = [
    pytest.mark.cosmos_live,
    pytest.mark.skipif(
        os.environ.get("RUN_COSMOS_LIVE") != "1",
        reason="set RUN_COSMOS_LIVE=1 to use real Cosmos",
    ),
    pytest.mark.skipif(
        not os.environ.get("COSMOS_TEST_LLM_BASE_URL")
        or not os.environ.get("COSMOS_TEST_LLM_MODEL"),
        reason="set COSMOS_TEST_LLM_BASE_URL and COSMOS_TEST_LLM_MODEL to run the agent",
    ),
    pytest.mark.skipif(
        not os.environ.get("TOOLKIT_TEST_URL") or not os.environ.get("TOOLKIT_TEST_APP_ID"),
        reason="set TOOLKIT_TEST_URL and TOOLKIT_TEST_APP_ID for toolkit-backed agent checks",
    ),
]


@pytest.fixture(scope="session")
def agent_settings():
    """The explicitly configured live model and toolkit."""
    return RetrieverConfig(
        llm_base_url=os.environ["COSMOS_TEST_LLM_BASE_URL"],
        llm_model=os.environ["COSMOS_TEST_LLM_MODEL"],
        llm_api_key=os.environ.get("COSMOS_TEST_LLM_API_KEY"),
        toolkit_mcp_url=os.environ["TOOLKIT_TEST_URL"],
        toolkit_app_id=os.environ["TOOLKIT_TEST_APP_ID"],
    )


@pytest.fixture(scope="session")
def agent_http(agent_settings):
    with TestClient(create_app(agent_settings)) as client:
        yield client


def test_agent_search_answers_with_supporting_documents(agent_http, live_database):
    response = agent_http.post(
        "/agentic_search",
        json={
            "query": "What do the documents say about battery recycling?",
            "database": live_database,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["terminal_reason"] in {"stop", "max_turns"}
    assert body["turns"] >= 1
    assert isinstance(body["answer"], str) and body["answer"].strip()
    assert body["documents"], "the agent should return supporting documents"
    assert all("document" in item and "id" in item for item in body["documents"])


@pytest.fixture(scope="session")
def agent_service_url(agent_settings):
    """Serve the real agent on a free loopback port for the .NET test process."""
    app = create_app(agent_settings)
    ready = Event()

    class TestServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            ready.set()

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        server = TestServer(uvicorn.Config(app, log_level="warning", access_log=False))
        thread = Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            assert ready.wait(30) and server.started, "Agent server did not start."
            yield f"http://127.0.0.1:{listener.getsockname()[1]}"
        finally:
            server.should_exit = True
            thread.join(15)
            assert not thread.is_alive(), "Agent server did not shut down."


@pytest.mark.skipif(
    os.environ.get("RUN_COSMOS_MCP_LIVE") != "1",
    reason="set RUN_COSMOS_MCP_LIVE=1 to run the .NET MCP integration tests",
)
def test_agent_search_through_mcp(agent_service_url, live_database):
    """Run real MCP requests through .NET, Python, the model, and Cosmos."""
    dotnet = shutil.which("dotnet")
    assert dotnet is not None, "The .NET 9 SDK must be on PATH."
    package = Path(__file__).resolve().parents[2]
    project = (
        package.parent / "tests" / "AzureCosmosDB.MCP.Toolkit.Tests"
        / "AzureCosmosDB.MCP.Toolkit.Tests.csproj"
    )
    command = [
        dotnet, "test", str(project),
        "--filter", "FullyQualifiedName~AgenticSearchLiveTests",
        "--nologo", "--verbosity", "minimal",
        "--logger", "trx;LogFileName=pr6-mcp-live.trx",
        "--results-directory", str(package / "artifacts"),
        "-p:NuGetAudit=false",
    ]
    toolkit_project = os.environ.get("COSMOS_TEST_MCP_PROJECT")
    if toolkit_project:
        selected_project = Path(toolkit_project).resolve(strict=True)
        command.append(f"-p:ToolkitProject={selected_project}")
    environment = dict(os.environ)
    for name in ("LLM_API_KEY", "COSMOS_TEST_LLM_API_KEY", "OPENAI_API_KEY", "COSMOS_CONNECTION_STRING"):
        environment.pop(name, None)
    environment.update(
        COSMOS_RETRIEVER_URL=agent_service_url,
        COSMOS_RETRIEVER_TIMEOUT_S="240",
        COSMOS_TEST_DATABASE=live_database,
        Logging__LogLevel__Default="Critical",
    )
    result = subprocess.run(
        command, cwd=package.parent, env=environment,
        capture_output=True, text=True, timeout=600, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = ElementTree.parse(package / "artifacts" / "pr6-mcp-live.trx")
    counters = report.find(".//{http://microsoft.com/schemas/VisualStudio/TeamTest/2010}Counters")
    assert counters is not None, "The .NET test report has no result counts."
    assert int(counters.attrib["executed"]) > 0, "No MCP live tests were executed."
    assert counters.attrib["passed"] == counters.attrib["total"], counters.attrib
