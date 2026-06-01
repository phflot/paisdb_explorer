"""Tests for PAISDB provider routing helpers."""

from __future__ import annotations

from types import SimpleNamespace

from abstracts_explorer.provider_routing import (
    normalize_openai_base_url,
    provider_status_payload,
    resolve_chat_provider,
    resolve_embedding_provider,
    resolve_generation_provider,
    resolve_generation_provider_chain,
)


def _config(**overrides):
    values = {
        "chat_model": "legacy-chat",
        "embedding_model": "legacy-embedding",
        "llm_backend_url": "http://localhost:1234",
        "llm_backend_auth_token": "legacy-token",
        "pais_evidence_brief_model": "",
        "pais_evidence_brief_base_url": "",
        "pais_evidence_brief_auth_token": "",
        "pais_extraction_model": "",
        "pais_extraction_base_url": "",
        "pais_extraction_auth_token": "",
        "pais_embedding_model": "",
        "pais_embedding_base_url": "",
        "pais_embedding_auth_token": "",
        "pais_model_providers_config": "",
        "pais_generation_provider": "",
        "pais_generation_fallbacks": "",
        "pais_chat_provider": "",
        "pais_evidence_brief_provider": "",
        "pais_extraction_provider": "",
    }
    values.update(overrides)
    config = SimpleNamespace(**values)
    config._env = {
        "LLM_BACKEND_AUTH_TOKEN": config.llm_backend_auth_token,
    }
    config._env.update(overrides.pop("_env", {}))
    return config


def test_normalize_openai_base_url_adds_v1_once():
    assert normalize_openai_base_url("http://example.test:18000") == "http://example.test:18000/v1"
    assert normalize_openai_base_url("http://example.test:18000/v1") == "http://example.test:18000/v1"
    assert normalize_openai_base_url("http://example.test:18000/v1/") == "http://example.test:18000/v1"


def test_embedding_provider_prefers_pais_embedding_stage():
    provider = resolve_embedding_provider(
        _config(
            pais_embedding_model="Qwen/Qwen3-Embedding-8B",
            pais_embedding_base_url="http://134.96.118.198:18080/v1",
            pais_embedding_auth_token="pais-token",
        )
    )

    assert provider.model == "Qwen/Qwen3-Embedding-8B"
    assert provider.base_url == "http://134.96.118.198:18080/v1"
    assert provider.auth_token == "pais-token"
    assert provider.source == "pais_embedding"


def test_chat_provider_prefers_evidence_brief_then_extraction():
    provider = resolve_chat_provider(
        _config(
            pais_evidence_brief_model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            pais_evidence_brief_base_url="http://134.96.118.198:18000/v1",
            pais_evidence_brief_auth_token="brief-token",
            pais_extraction_model="extract",
            pais_extraction_base_url="http://extract.test/v1",
        )
    )

    assert provider.model == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    assert provider.base_url == "http://134.96.118.198:18000/v1"
    assert provider.auth_token == "brief-token"
    assert provider.source == "pais_evidence_brief"

    fallback = resolve_chat_provider(
        _config(
            pais_extraction_model="extract",
            pais_extraction_base_url="http://extract.test/v1",
            pais_extraction_auth_token="extract-token",
        )
    )
    assert fallback.model == "extract"
    assert fallback.auth_token == "extract-token"
    assert fallback.source == "pais_extraction"


def test_generation_provider_registry_resolves_selected_provider(tmp_path, monkeypatch):
    config_path = tmp_path / "providers.yaml"
    config_path.write_text(
        """
generation_providers:
  remote_deepseek:
    label: DeepSeek
    model: deepseek-ai/DeepSeek-V4-Pro
    base_url: http://remote.test:18000/v1
    auth_token_env: PAISDB_AI_API_KEY
    enabled: true
    priority: 10
    stages: [chat, evidence_brief, structured_extraction]
  local_qwen:
    label: Qwen
    model: Qwen/Qwen3-Coder-30B-A3B-Instruct
    base_url: http://127.0.0.1:18100
    enabled: true
    priority: 20
    stages: [chat]
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("PAISDB_AI_API_KEY", "secret-token")

    provider = resolve_generation_provider(
        _config(
            pais_model_providers_config=str(config_path),
            pais_generation_provider="remote_deepseek",
        ),
        stage="chat",
    )

    assert provider.provider_id == "remote_deepseek"
    assert provider.model == "deepseek-ai/DeepSeek-V4-Pro"
    assert provider.base_url == "http://remote.test:18000/v1"
    assert provider.auth_token == "secret-token"


def test_generation_provider_reads_auth_token_from_config_env(tmp_path, monkeypatch):
    config_path = tmp_path / "providers.yaml"
    config_path.write_text(
        """
generation_providers:
  remote_deepseek:
    model: deepseek-ai/DeepSeek-V4-Pro
    base_url: http://remote.test:18000/v1
    auth_token_env: PAISDB_AI_API_KEY
    enabled: true
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("PAISDB_AI_API_KEY", raising=False)

    provider = resolve_generation_provider(
        _config(
            pais_model_providers_config=str(config_path),
            pais_generation_provider="remote_deepseek",
            _env={"PAISDB_AI_API_KEY": "dotenv-token"},
        ),
        stage="chat",
    )

    assert provider.auth_token == "dotenv-token"


def test_generation_provider_expands_config_env_with_default(tmp_path):
    config_path = tmp_path / "providers.yaml"
    config_path.write_text(
        """
generation_providers:
  local_qwen:
    model: Qwen/Qwen3-Coder-30B-A3B-Instruct
    base_url: ${PAIS_LOCAL_QWEN_BASE_URL:-http://127.0.0.1:18100/v1}
    enabled: true
""",
        encoding="utf-8",
    )

    default_provider = resolve_generation_provider(
        _config(
            pais_model_providers_config=str(config_path),
            pais_generation_provider="local_qwen",
        ),
        stage="chat",
    )
    container_provider = resolve_generation_provider(
        _config(
            pais_model_providers_config=str(config_path),
            pais_generation_provider="local_qwen",
            _env={"PAIS_LOCAL_QWEN_BASE_URL": "http://host.containers.internal:18100/v1"},
        ),
        stage="chat",
    )

    assert default_provider.base_url == "http://127.0.0.1:18100/v1"
    assert container_provider.base_url == "http://host.containers.internal:18100/v1"


def test_generation_provider_auto_chain_uses_primary_then_fallbacks(tmp_path):
    config_path = tmp_path / "providers.yaml"
    config_path.write_text(
        """
generation_providers:
  primary:
    model: model-primary
    base_url: http://primary.test/v1
    enabled: true
  fallback:
    model: model-fallback
    base_url: http://fallback.test/v1
    enabled: true
""",
        encoding="utf-8",
    )

    chain = resolve_generation_provider_chain(
        _config(
            pais_model_providers_config=str(config_path),
            pais_generation_provider="primary",
            pais_generation_fallbacks="fallback",
        ),
        provider_id="auto",
        stage="chat",
    )

    assert [provider.provider_id for provider in chain] == ["primary", "fallback"]


def test_provider_status_payload_sanitizes_urls(tmp_path):
    config_path = tmp_path / "providers.yaml"
    config_path.write_text(
        """
generation_providers:
  unsafe:
    model: model
    base_url: https://user:pass@example.test/v1?api_key=secret&tenant=public
    enabled: true
""",
        encoding="utf-8",
    )

    payload = provider_status_payload(_config(pais_model_providers_config=str(config_path)))

    assert payload["providers"][0]["base_url"] == "https://example.test/v1?tenant=public"
