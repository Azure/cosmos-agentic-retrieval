"""Configure the agentic-search service.

`AgenticSearchConfig` is a pydantic-settings model: each field is read from the
matching uppercased environment variable at startup (`LLM_BASE_URL`,
`LLM_MODEL`, `TOOLKIT_APP_ID`, ...), loaded by `get_config()`.

The service never reads Cosmos DB itself: the agent searches with the MCP
toolkit's tools, and each request names the database and container to search.

Alternatively, set `COSMOS_RETRIEVER_CONFIG_FILE` to a YAML file path to read the
whole configuration from that file instead of many environment variables. Values
in the file win; any field the file omits still falls back to its environment
variable, so secrets like `LLM_API_KEY` can stay in the environment, or be
dropped entirely when signing in with Entra ID.

The agent needs a model: `llm_base_url` and `llm_model` are required and point at
any OpenAI-compatible endpoint. Put per-call model options such as `max_tokens`
(or `max_completion_tokens`) and `temperature` in `llm_params`; they are sent to
the model unchanged.
"""

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import AnyHttpUrl, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from cosmos_agentic_retriever.agent.result_trimming import DEFAULT_DROPPED_FIELDS


class AgenticSearchConfig(BaseSettings):
    """Startup settings for the model, the toolkit, and the HTTP server."""

    model_config = SettingsConfigDict(extra="forbid")

    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=9000, ge=1, le=65535)
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"
    llm_base_url: AnyHttpUrl
    llm_model: str = Field(min_length=1)
    # Without a key, the model is called with an Entra token from DefaultAzureCredential.
    llm_api_key: SecretStr | None = None
    llm_params: dict[str, Any] = Field(default_factory=dict)
    llm_max_retries: int = Field(default=6, ge=0)
    agent_max_turns: int = Field(default=10, ge=1, le=50)
    agent_max_documents: int = Field(default=20, ge=1, le=50)
    # "cited": only documents the answer cites; "cited_and_ranked" adds the
    # best-ranked retrieved documents it didn't cite, e.g. for benchmarks.
    returned_documents: Literal["cited", "cited_and_ranked"] = "cited"
    # Default: the toolkit container in the same Container App.
    toolkit_mcp_url: AnyHttpUrl = AnyHttpUrl("http://127.0.0.1:8080/mcp")
    # Client ID of the toolkit's Entra app; tokens for the toolkit are requested for it.
    toolkit_app_id: str = Field(min_length=1)
    # How toolkit results are trimmed before the agent reads them; see agent/result_trimming.py.
    tool_result_text_limit: int = Field(default=512, ge=1)
    tool_result_min_vector_dim: int = Field(default=128, ge=2)
    tool_result_dropped_fields: list[str] = Field(
        default_factory=lambda: list(DEFAULT_DROPPED_FIELDS)
    )

    @field_validator("toolkit_app_id")
    @classmethod
    def _toolkit_app_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("configure the toolkit's Entra app client ID")
        return value


def get_config() -> AgenticSearchConfig:
    """Build the settings at startup from a YAML file or the environment.

    When `COSMOS_RETRIEVER_CONFIG_FILE` is set, its YAML contents are validated as
    the settings, and any field the file omits falls back to that field's
    environment variable. When the variable is unset, every field comes from the
    environment. Reads happen here at startup, not during import.
    """
    path = os.environ.get("COSMOS_RETRIEVER_CONFIG_FILE")
    if path:
        data = yaml.safe_load(Path(path).read_text()) or {}
        return AgenticSearchConfig.model_validate(data)
    return AgenticSearchConfig()
