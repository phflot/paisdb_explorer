"""Tests for PAISDB provider routing helpers."""

from __future__ import annotations

from types import SimpleNamespace

from abstracts_explorer.provider_routing import (
    normalize_openai_base_url,
    resolve_chat_provider,
    resolve_embedding_provider,
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
    }
    values.update(overrides)
    return SimpleNamespace(**values)


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
