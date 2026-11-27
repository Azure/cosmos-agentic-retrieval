"""Offline HTTP contract tests for startup, health and request validation."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from cosmos_agentic_retriever import server
from cosmos_agentic_retriever.config import AgenticSearchConfig
from cosmos_agentic_retriever.server import AgenticSearchRequest, create_app


def _settings(**options):
    return AgenticSearchConfig(
        **{
            "llm_base_url": "https://model.example.com/v1",
            "llm_model": "test-model",
            "toolkit_app_id": "test-toolkit",
            # A model key keeps the model from creating its own Azure credential here.
            "llm_api_key": "test-key",
            **options,
        }
    )


def test_create_app_falls_back_to_get_config(monkeypatch):
    load = Mock(return_value=_settings())
    monkeypatch.setattr(server, "get_config", load)
    app = create_app()
    load.assert_called_once()
    assert app.state.chat_client is None and app.state.toolkit is None


def test_requests_before_startup_are_not_ready():
    client = TestClient(create_app(_settings()))
    for _ in range(2):
        health = client.get("/health")
        assert health.status_code == 503 and health.json() == {"status": "unavailable"}
        response = client.post("/agentic_search", json={"query": "battery"})
        assert response.status_code == 503 and response.json() == {
            "error": "Service is not ready."
        }
        with client:
            assert client.get("/health").status_code == 200


def test_health_and_no_config_routes():
    with TestClient(create_app(_settings())) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/config").status_code == 404
        assert client.patch("/config", json={}).status_code == 404
        assert client.post("/full_text_search", json={"query": "battery"}).status_code == 404


def test_search_request_defaults() -> None:
    request = AgenticSearchRequest(query="q")
    assert request.max_documents is None and request.max_turns is None
    assert request.returned_documents is None
    assert request.database is None and request.container is None


def test_search_request_alias_and_field_name() -> None:
    assert (
        AgenticSearchRequest.model_validate({"query": "q", "maxDocuments": 5}).max_documents
        == 5
    )
    assert AgenticSearchRequest(query="q", max_documents=7).max_documents == 7


@pytest.mark.parametrize("query", ["", "   ", "\n", "q" * 4097])
def test_search_request_query_min_length(query) -> None:
    with pytest.raises(ValidationError):
        AgenticSearchRequest(query=query)


@pytest.mark.parametrize("max_documents", [1, 50])
def test_search_request_max_documents_bounds_ok(max_documents) -> None:
    assert (
        AgenticSearchRequest(query="q", max_documents=max_documents).max_documents
        == max_documents
    )


@pytest.mark.parametrize("value", [0, 51, True, 1.5, "5"])
def test_search_request_max_documents_out_of_range(value) -> None:
    with pytest.raises(ValidationError):
        AgenticSearchRequest(query="q", max_documents=value)


def test_validation_errors_do_not_echo_input():
    with TestClient(create_app(_settings())) as client:
        response = client.post(
            "/agentic_search", json={"query": "battery", "api_key": "private-test-value"}
        )
    assert response.status_code == 422
    assert response.json() == {"error": "Invalid search request."}


@pytest.mark.parametrize("body", ["{", "[]", "{}"])
def test_invalid_json_body_is_rejected(body):
    with TestClient(create_app(_settings())) as client:
        response = client.post(
            "/agentic_search", content=body, headers={"Content-Type": "application/json"}
        )
    assert response.status_code == 422
    assert response.json() == {"error": "Invalid search request."}
