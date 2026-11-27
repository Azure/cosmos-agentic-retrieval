"""Validate service configuration without credentials or network access."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from cosmos_agentic_retriever import __main__ as cli
from cosmos_agentic_retriever.config import AgenticSearchConfig, get_config


def _environment(monkeypatch):
    values = {
        "LLM_BASE_URL": "https://model.example.com/v1",
        "LLM_MODEL": "test-model",
        "TOOLKIT_APP_ID": "test-toolkit",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_settings_from_environment(monkeypatch):
    _environment(monkeypatch)
    monkeypatch.setenv("LLM_API_KEY", "test-secret")
    monkeypatch.setenv("LLM_MAX_RETRIES", "4")
    monkeypatch.setenv("HOST", "127.0.0.2")
    monkeypatch.setenv("PORT", "9100")
    monkeypatch.setenv("LOG_LEVEL", "warning")
    settings = get_config()
    assert settings.llm_model == "test-model"
    assert settings.toolkit_app_id == "test-toolkit"
    assert settings.llm_api_key.get_secret_value() == "test-secret"
    assert "test-secret" not in repr(settings)
    assert settings.llm_max_retries == 4
    assert (settings.host, settings.port, settings.log_level) == (
        "127.0.0.2",
        9100,
        "warning",
    )
    monkeypatch.setenv("LLM_MODEL", "changed")
    assert get_config().llm_model == "changed"
    assert settings.llm_model == "test-model"


def test_hard_limits_default_to_soft_limits(monkeypatch):
    _environment(monkeypatch)
    settings = get_config()
    assert (settings.agent_turns_soft_limit, settings.agent_turns_hard_limit) == (10, 10)
    assert (settings.agent_documents_soft_limit, settings.agent_documents_hard_limit) == (20, 20)

    monkeypatch.setenv("AGENT_TURNS_SOFT_LIMIT", "5")
    monkeypatch.setenv("AGENT_DOCUMENTS_HARD_LIMIT", "40")
    settings = get_config()
    assert (settings.agent_turns_soft_limit, settings.agent_turns_hard_limit) == (5, 5)
    assert (settings.agent_documents_soft_limit, settings.agent_documents_hard_limit) == (20, 40)


@pytest.mark.parametrize("name", ["LLM_BASE_URL", "LLM_MODEL", "TOOLKIT_APP_ID"])
def test_required_settings_fail_before_launch(monkeypatch, name):
    _environment(monkeypatch)
    monkeypatch.delenv(name)
    with pytest.raises(ValidationError) as error:
        get_config()
    assert any(
        item["loc"] == (name.lower(),) and item["type"] == "missing"
        for item in error.value.errors()
    )


def test_settings_do_not_implicitly_read_dotenv(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "LLM_BASE_URL=https://model.example.com/v1\nLLM_MODEL=test-model\n"
        "TOOLKIT_APP_ID=test-toolkit\n"
    )
    with pytest.raises(ValidationError):
        get_config()


_YAML_CONFIG = (
    "llm_base_url: https://model.example.com/v1\n"
    "llm_model: file-model\n"
    "toolkit_app_id: test-toolkit\n"
)


def test_get_config_reads_yaml_file(monkeypatch, tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(_YAML_CONFIG)
    monkeypatch.setenv("COSMOS_RETRIEVER_CONFIG_FILE", str(config_file))
    settings = get_config()
    assert settings.llm_model == "file-model"
    assert settings.toolkit_app_id == "test-toolkit"


def test_example_yaml_loads(monkeypatch):
    example = Path(__file__).resolve().parents[2] / "config.example.yaml"
    monkeypatch.setenv("COSMOS_RETRIEVER_CONFIG_FILE", str(example))
    monkeypatch.setenv("TOOLKIT_APP_ID", "test-toolkit")

    settings = get_config()

    assert settings.llm_model == "YOUR-MODEL"
    assert settings.llm_api_key is None


def test_yaml_file_value_beats_environment(monkeypatch, tmp_path):
    _environment(monkeypatch)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(_YAML_CONFIG)
    monkeypatch.setenv("COSMOS_RETRIEVER_CONFIG_FILE", str(config_file))
    assert get_config().llm_model == "file-model"


def test_environment_fills_field_absent_from_yaml(monkeypatch, tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(_YAML_CONFIG)
    monkeypatch.setenv("COSMOS_RETRIEVER_CONFIG_FILE", str(config_file))
    monkeypatch.setenv("LLM_API_KEY", "env-secret")
    assert get_config().llm_api_key.get_secret_value() == "env-secret"


@pytest.mark.parametrize(
    "options",
    [
        {"llm_base_url": "not-a-url"},
        {"llm_model": ""},
        {"port": 0},
        {"port": 65536},
        {"llm_max_retries": -1},
        {"toolkit_app_id": None},
        {"toolkit_app_id": ""},
        {"toolkit_app_id": "   "},
        {"cosmos_database": "D"},
        {"typo": True},
        {"agent_max_turns": 10},
        {"agent_turns_soft_limit": 10, "agent_turns_hard_limit": 9},
        {"agent_documents_soft_limit": 20, "agent_documents_hard_limit": 19},
        {"agent_turns_hard_limit": 51},
        {"agent_documents_hard_limit": 0},
    ],
)
def test_settings_reject_invalid_configuration(options):
    values = {
        "llm_base_url": "https://model.example.com/v1",
        "llm_model": "test-model",
        "toolkit_app_id": "test-toolkit",
    }
    with pytest.raises(ValidationError):
        AgenticSearchConfig(**{**values, **options})


@pytest.mark.parametrize(
    "arguments, host, port",
    [
        (["serve"], "127.0.0.1", 9000),
        (["serve", "--port", "9100"], "127.0.0.1", 9100),
        (["serve", "--host", "0.0.0.0"], "0.0.0.0", 9000),
        (["serve", "--host", "0.0.0.0", "--port", "9100"], "0.0.0.0", 9100),
    ],
)
def test_serve_command(monkeypatch, arguments, host, port):
    settings = AgenticSearchConfig(
        llm_base_url="https://model.example.com/v1",
        llm_model="test-model",
        toolkit_app_id="test-toolkit",
    )
    monkeypatch.setattr(cli, "get_config", Mock(return_value=settings))
    app = object()
    build = Mock(return_value=app)
    run = Mock()
    monkeypatch.setattr(cli, "create_app", build)
    monkeypatch.setattr(cli.uvicorn, "run", run)
    assert cli.main(arguments) == 0
    run.assert_called_once_with(app, host=host, port=port, log_level="info")
    assert build.call_args.args[0].llm_model == "test-model"
    assert settings.host == "127.0.0.1" and settings.port == 9000


def test_invalid_configuration_does_not_start_or_print_values(monkeypatch, capsys):
    monkeypatch.setenv("LLM_BASE_URL", "private-invalid-endpoint")
    monkeypatch.setenv("LLM_API_KEY", "private-key")
    run = Mock()
    monkeypatch.setattr(cli.uvicorn, "run", run)
    assert cli.main(["serve"]) == 1
    run.assert_not_called()
    error = capsys.readouterr().err
    assert "Invalid service configuration" in error
    assert "private-" not in error


@pytest.mark.parametrize(
    "options",
    [
        {"LLM_PARAMS": "private-malformed-json"},
        {"TOOL_RESULT_DROPPED_FIELDS": "private-malformed-json"},
    ],
)
def test_invalid_environment_never_builds_app(monkeypatch, capsys, options):
    _environment(monkeypatch)
    for name, value in options.items():
        monkeypatch.setenv(name, value)
    build = Mock()
    monkeypatch.setattr(cli, "create_app", build)
    assert cli.main(["serve"]) == 1
    build.assert_not_called()
    assert capsys.readouterr().err == (
        "Invalid service configuration. Check the model, toolkit, and server settings.\n"
    )


@pytest.mark.parametrize(
    "arguments", [["--port", "0"], ["--port", "65536"], ["--host", ""]]
)
def test_invalid_cli_overrides_never_build_app(monkeypatch, arguments):
    _environment(monkeypatch)
    build = Mock()
    monkeypatch.setattr(cli, "create_app", build)
    assert cli.main(["serve", *arguments]) == 1
    build.assert_not_called()


def test_cli_overrides_environment_without_changing_other_settings(monkeypatch):
    _environment(monkeypatch)
    monkeypatch.setenv("HOST", "127.0.0.2")
    monkeypatch.setenv("PORT", "9200")
    monkeypatch.setenv("LOG_LEVEL", "warning")
    build, run = Mock(), Mock()
    monkeypatch.setattr(cli, "create_app", build)
    monkeypatch.setattr(cli.uvicorn, "run", run)
    assert cli.main(["serve", "--port", "9100"]) == 0
    run.assert_called_once_with(
        build.return_value, host="127.0.0.2", port=9100, log_level="warning"
    )
    assert build.call_args.args[0].llm_model == "test-model"
