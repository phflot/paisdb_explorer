"""Provider routing helpers for PAISDB web UI integrations."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import yaml


LEGACY_CHAT_MODEL = "diffbot-small-xl-2508"
LEGACY_EMBEDDING_MODEL = "text-embedding-qwen3-embedding-4b"
LEGACY_LLM_BACKEND_URL = "http://localhost:1234"
PAIS_EVIDENCE_COLLECTION = "pais_evidence"
AUTO_PROVIDER_ID = "auto"
ENV_VAR_PATTERN = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}|\$([A-Za-z_][A-Za-z0-9_]*)"
)


@dataclass(frozen=True)
class ResolvedProvider:
    """Resolved OpenAI-compatible provider settings."""

    model: str
    base_url: str
    auth_token: str = ""
    source: str = "legacy"
    provider_id: str = ""
    label: str = ""


@dataclass(frozen=True)
class GenerationProvider:
    """Configured OpenAI-compatible generation provider."""

    provider_id: str
    label: str
    model: str
    base_url: str
    auth_token: str = ""
    auth_token_env: str = ""
    enabled: bool = True
    priority: int = 100
    stages: tuple[str, ...] = ("chat", "evidence_brief", "structured_extraction")


def normalize_openai_base_url(base_url: str) -> str:
    """Return an OpenAI SDK base URL with exactly one trailing /v1 path."""
    cleaned = (base_url or "").strip().rstrip("/")
    if not cleaned:
        return cleaned
    parts = urlsplit(cleaned)
    path = parts.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def sanitize_url(url: str) -> str:
    """Remove credentials and secret query parameters from a URL."""
    if not url:
        return ""
    parts = urlsplit(url)
    netloc = parts.hostname or ""
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    safe_query = urlencode(
        [
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if key.lower() not in {"api_key", "access_token", "authorization", "token", "key"}
        ]
    )
    return urlunsplit((parts.scheme, netloc, parts.path, safe_query, parts.fragment))


def load_generation_providers(config) -> dict[str, GenerationProvider]:
    """Load configured generation providers from YAML."""
    raw = _load_provider_file(config)
    providers: dict[str, GenerationProvider] = {}
    for provider_id, item in (raw.get("generation_providers") or {}).items():
        if not isinstance(item, dict):
            continue
        model = _expand_env(str(item.get("model") or ""), config)
        base_url = _expand_env(str(item.get("base_url") or ""), config)
        if not model or not base_url:
            continue
        auth_token_env = str(item.get("auth_token_env") or "")
        explicit_token = _expand_env(str(item.get("auth_token") or ""), config)
        providers[str(provider_id)] = GenerationProvider(
            provider_id=str(provider_id),
            label=str(item.get("label") or provider_id),
            model=model,
            base_url=normalize_openai_base_url(base_url),
            auth_token=explicit_token or _get_config_env(config, auth_token_env),
            auth_token_env=auth_token_env,
            enabled=bool(item.get("enabled", True)),
            priority=int(item.get("priority", 100)),
            stages=tuple(
                str(stage)
                for stage in item.get("stages", ["chat", "evidence_brief", "structured_extraction"])
            ),
        )
    return providers


def resolve_generation_provider(
    config,
    provider_id: str | None = None,
    stage: str = "chat",
) -> ResolvedProvider:
    """Resolve one generation provider for a stage."""
    chain = resolve_generation_provider_chain(config, provider_id=provider_id, stage=stage)
    if chain:
        return chain[0]
    return _legacy_stage_provider(config, stage)


def resolve_generation_provider_chain(
    config,
    provider_id: str | None = None,
    stage: str = "chat",
) -> list[ResolvedProvider]:
    """Resolve provider fallback order for a requested stage."""
    providers = load_generation_providers(config)
    if not providers:
        return []

    requested = _stage_provider_id(config, provider_id, stage)
    if requested and requested != AUTO_PROVIDER_ID:
        provider = providers.get(requested)
        if provider is None:
            raise ValueError(f"Unknown generation provider '{requested}'")
        if not provider.enabled:
            raise ValueError(f"Generation provider '{requested}' is disabled")
        if stage not in provider.stages:
            raise ValueError(f"Generation provider '{requested}' is not allowed for stage '{stage}'")
        return [_as_resolved(provider)]

    ids: list[str] = []
    primary = getattr(config, "pais_generation_provider", "") or ""
    if primary and primary != AUTO_PROVIDER_ID:
        ids.append(primary)
    ids.extend(_split_provider_ids(getattr(config, "pais_generation_fallbacks", "")))
    if not ids:
        ids = [
            provider.provider_id
            for provider in sorted(providers.values(), key=lambda item: (item.priority, item.provider_id))
        ]

    resolved: list[ResolvedProvider] = []
    seen: set[str] = set()
    for item_id in ids:
        if item_id in seen:
            continue
        seen.add(item_id)
        provider = providers.get(item_id)
        if provider and provider.enabled and stage in provider.stages:
            resolved.append(_as_resolved(provider))
    return resolved


def provider_status_payload(config, include_tokens: bool = False) -> dict[str, Any]:
    """Return frontend-safe generation provider metadata."""
    providers = load_generation_providers(config)
    selected = _stage_provider_id(config, None, "chat") or AUTO_PROVIDER_ID
    return {
        "selected_provider_id": selected,
        "default_provider_id": getattr(config, "pais_generation_provider", "") or AUTO_PROVIDER_ID,
        "fallback_provider_ids": _split_provider_ids(getattr(config, "pais_generation_fallbacks", "")),
        "providers": [
            {
                "id": provider.provider_id,
                "label": provider.label,
                "model": provider.model,
                "base_url": sanitize_url(provider.base_url),
                "enabled": provider.enabled,
                "priority": provider.priority,
                "stages": list(provider.stages),
                "auth_configured": bool(provider.auth_token) if include_tokens else bool(provider.auth_token_env),
            }
            for provider in sorted(providers.values(), key=lambda item: (item.priority, item.provider_id))
        ],
    }


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


def _legacy_stage_provider(config, stage: str) -> ResolvedProvider:
    if stage in {"chat", "evidence_brief"}:
        return resolve_chat_provider(config)
    if stage == "structured_extraction" and config.pais_extraction_model and config.pais_extraction_base_url:
        return ResolvedProvider(
            model=config.pais_extraction_model,
            base_url=config.pais_extraction_base_url,
            auth_token=config.pais_extraction_auth_token or config.llm_backend_auth_token,
            source="pais_extraction",
            provider_id="legacy_pais_extraction",
            label="Legacy PAIS extraction",
        )
    return resolve_chat_provider(config)


def _load_provider_file(config) -> dict[str, Any]:
    raw_path = getattr(config, "pais_model_providers_config", "") or ""
    if not raw_path:
        return {}
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
        if not path.exists():
            path = Path(__file__).resolve().parents[2] / raw_path
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _expand_env(value: str, config=None) -> str:
    """Expand $VAR and ${VAR:-default} using process env plus loaded .env values."""

    if "$" not in value:
        return value.strip()

    config_env = dict(getattr(config, "_env", {}) or {})
    merged_env = {**config_env, **os.environ}

    def replace(match: re.Match[str]) -> str:
        key = match.group(1) or match.group(3)
        default = match.group(2) or ""
        env_value = merged_env.get(key, "")
        return env_value if env_value else default

    return ENV_VAR_PATTERN.sub(replace, value).strip()


def _get_config_env(config, key: str) -> str:
    if not key:
        return ""
    config_env = getattr(config, "_env", {})
    value = config_env.get(key) or os.environ.get(key, "")
    if value:
        return value
    if key == "PAISDB_AI_API_KEY":
        return config_env.get("LLM_BACKEND_AUTH_TOKEN") or os.environ.get("LLM_BACKEND_AUTH_TOKEN", "")
    return ""


def _split_provider_ids(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _stage_provider_id(config, requested: str | None, stage: str) -> str:
    if requested:
        return requested
    if stage == "chat" and getattr(config, "pais_chat_provider", ""):
        return config.pais_chat_provider
    if stage == "evidence_brief" and getattr(config, "pais_evidence_brief_provider", ""):
        return config.pais_evidence_brief_provider
    if stage == "structured_extraction" and getattr(config, "pais_extraction_provider", ""):
        return config.pais_extraction_provider
    return getattr(config, "pais_generation_provider", "") or AUTO_PROVIDER_ID


def _as_resolved(provider: GenerationProvider) -> ResolvedProvider:
    return ResolvedProvider(
        model=provider.model,
        base_url=provider.base_url,
        auth_token=provider.auth_token,
        source="pais_generation_provider",
        provider_id=provider.provider_id,
        label=provider.label,
    )
