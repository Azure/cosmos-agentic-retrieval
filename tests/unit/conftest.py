"""Keep service tests independent of the developer's Azure configuration."""

import os

import pytest

from cosmos_agentic_retriever.config import AgenticSearchConfig


@pytest.fixture(autouse=True)
def clear_service_environment(monkeypatch):
    names = {name.lower() for name in AgenticSearchConfig.model_fields}
    for name in os.environ:
        if name.lower() in names:
            monkeypatch.delenv(name)
