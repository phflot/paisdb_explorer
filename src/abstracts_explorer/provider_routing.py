"""Provider routing helpers for PAISDB web UI integrations."""

from __future__ import annotations

from dataclasses import dataclass


LEGACY_CHAT_MODEL = "diffbot-small-xl-2508"
LEGACY_EMBEDDING_MODEL = "text-embedding-qwen3-embedding-4b"
LEGACY_LLM_BACKEND_URL = "http://localhost:1234"
PAIS_EVIDENCE_COLLECTION = "pais_evidence"


@dataclass(frozen=True)
class ResolvedProvider:
    """Resolved OpenAI-compatible provider settings."""

    model: str
    base_url: str
    auth_token: str = ""
    source: str = "legacy"


def normalize_openai_base_url(base_url: str) -> str:
    """Return an OpenAI SDK base URL with exactly one trailing /v1 path."""
    cleaned = (base_url or "").rstrip("/")
    if not cleaned:
        return cleaned
    if cleaned.endswith("/v1"):
        return cleaned
    return f"{cleaned}/v1"


def resolve_embedding_provider(config) -> ResolvedProvider:
    """Resolve the embedding provider, preferring PAISDB stage config."""
    if config.pais_embedding_model and config.pais_embedding_base_url:
        return ResolvedProvider(
            model=config.pais_embedding_model,
            base_url=config.pais_embedding_base_url,
            auth_token=config.pais_embedding_auth_token or config.llm_backend_auth_token,
            source="pais_embedding",
        )
    return ResolvedProvider(
        model=config.embedding_model,
        base_url=config.llm_backend_url,
        auth_token=config.llm_backend_auth_token,
        source="legacy_embedding",
    )


def resolve_chat_provider(config) -> ResolvedProvider:
    """Resolve the RAG/chat provider, preferring PAISDB hosted enrichment config."""
    if config.pais_evidence_brief_model and config.pais_evidence_brief_base_url:
        return ResolvedProvider(
            model=config.pais_evidence_brief_model,
            base_url=config.pais_evidence_brief_base_url,
            auth_token=config.pais_evidence_brief_auth_token or config.llm_backend_auth_token,
            source="pais_evidence_brief",
        )
    if config.pais_extraction_model and config.pais_extraction_base_url:
        return ResolvedProvider(
            model=config.pais_extraction_model,
            base_url=config.pais_extraction_base_url,
            auth_token=config.pais_extraction_auth_token or config.llm_backend_auth_token,
            source="pais_extraction",
        )
    return ResolvedProvider(
        model=config.chat_model,
        base_url=config.llm_backend_url,
        auth_token=config.llm_backend_auth_token,
        source="legacy_chat",
    )
