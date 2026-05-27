"""OpenAI-compatible clients for PAISDB model stages."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests
from pydantic import BaseModel, ValidationError

from abstracts_explorer.config import get_config
from abstracts_explorer.pais_prompts import canonical_json
from abstracts_explorer.pais_schemas import PaisStage


@dataclass(frozen=True)
class PaisStageConfig:
    """Resolved model routing for one PAIS stage."""

    stage: str
    backend: str
    model: str
    base_url: str
    auth_token: str = ""
    endpoint_id: Optional[str] = None


@dataclass
class PaisLLMResult:
    """Normalized result of a model invocation."""

    raw_output: str
    parsed_json: Optional[dict[str, Any]]
    valid: bool
    elapsed_s: float
    backend: str
    model_id: str
    endpoint_id: Optional[str]
    structured_output_used: bool
    error_kind: Optional[str] = None
    error_message: Optional[str] = None
    model_version: Optional[str] = None


class OpenAICompatiblePaisClient:
    """Small OpenAI-compatible client for PAIS screen and extraction stages."""

    def __init__(self, timeout_s: float = 120.0):
        self.config = get_config()
        self.timeout_s = timeout_s

    def resolve_stage_config(self, stage: str) -> PaisStageConfig:
        """Resolve backend/model/base URL for a PAIS stage from config."""
        if stage == PaisStage.BENCHMARK_SCREEN.value:
            model = self.config.pais_screen_model
            base_url = self.config.pais_screen_base_url
            token = self.config.pais_screen_auth_token
        elif stage == PaisStage.EVIDENCE_BRIEF.value:
            model = self.config.pais_evidence_brief_model
            base_url = self.config.pais_evidence_brief_base_url
            token = self.config.pais_evidence_brief_auth_token
        elif stage == PaisStage.STRUCTURED_EXTRACTION.value:
            model = self.config.pais_extraction_model
            base_url = self.config.pais_extraction_base_url
            token = self.config.pais_extraction_auth_token
        else:
            raise ValueError(f"Unknown PAIS stage: {stage}")

        return PaisStageConfig(
            stage=stage,
            backend="openai_compatible",
            model=model,
            base_url=base_url,
            auth_token=token or self.config.llm_backend_auth_token,
            endpoint_id=_endpoint_id(base_url),
        )

    def complete_text(
        self,
        messages: list[dict[str, str]],
        model: str,
        base_url: str,
        auth_token: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> PaisLLMResult:
        """Call an OpenAI-compatible chat completion endpoint and return text."""
        stage_config = PaisStageConfig(
            stage="text",
            backend="openai_compatible",
            model=model,
            base_url=base_url,
            auth_token=auth_token,
            endpoint_id=_endpoint_id(base_url),
        )
        return self._chat_completion(
            stage_config=stage_config,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            schema_model=None,
            structured=False,
        )

    def complete_json(
        self,
        messages: list[dict[str, str]],
        schema_model: type[BaseModel],
        stage: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        structured: Optional[bool] = None,
    ) -> PaisLLMResult:
        """Call a stage model and validate JSON against the supplied schema."""
        stage_config = self.resolve_stage_config(stage)
        structured_mode = self.config.pais_structured_output_mode
        structured_output = structured if structured is not None else structured_mode == "json_schema"
        return self._chat_completion(
            stage_config=stage_config,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            schema_model=schema_model,
            structured=structured_output,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts through the configured OpenAI-compatible embeddings endpoint."""
        if not texts:
            return []
        started = time.monotonic()
        url = _api_url(self.config.pais_embedding_base_url, "embeddings")
        payload = {"model": self.config.pais_embedding_model, "input": texts}
        headers = _headers(self.config.pais_embedding_auth_token or self.config.llm_backend_auth_token)
        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        response.raise_for_status()
        data = response.json()
        embeddings = [item["embedding"] for item in data.get("data", [])]
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Embedding endpoint returned {len(embeddings)} embeddings for {len(texts)} input texts "
                f"after {time.monotonic() - started:.3f}s"
            )
        return embeddings

    def _chat_completion(
        self,
        stage_config: PaisStageConfig,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        schema_model: Optional[type[BaseModel]],
        structured: bool,
    ) -> PaisLLMResult:
        started = time.monotonic()
        payload: dict[str, Any] = {
            "model": stage_config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if structured and schema_model is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_model.__name__,
                    "schema": schema_model.model_json_schema(),
                    "strict": True,
                },
            }

        try:
            response = requests.post(
                _api_url(stage_config.base_url, "chat/completions"),
                headers=_headers(stage_config.auth_token),
                json=payload,
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            data = response.json()
            raw_output = _extract_message_content(data)
            parsed_json = None
            valid = True
            error_kind = None
            error_message = None
            if schema_model is not None:
                try:
                    parsed = parse_json_object(raw_output)
                    parsed_json = schema_model.model_validate(parsed).model_dump(mode="json")
                except (ValueError, ValidationError, TypeError) as exc:
                    valid = False
                    error_kind = "validation_error"
                    error_message = str(exc)
            return PaisLLMResult(
                raw_output=raw_output,
                parsed_json=parsed_json,
                valid=valid,
                elapsed_s=time.monotonic() - started,
                backend=stage_config.backend,
                model_id=stage_config.model,
                endpoint_id=stage_config.endpoint_id,
                structured_output_used=structured,
                error_kind=error_kind,
                error_message=error_message,
            )
        except Exception as exc:
            return PaisLLMResult(
                raw_output="",
                parsed_json=None,
                valid=False,
                elapsed_s=time.monotonic() - started,
                backend=stage_config.backend,
                model_id=stage_config.model,
                endpoint_id=stage_config.endpoint_id,
                structured_output_used=structured,
                error_kind=exc.__class__.__name__,
                error_message=str(exc),
            )


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse one JSON object from a model response."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _strip_markdown_fence(stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("No JSON object found in model output")
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model output JSON must be an object")
    return parsed


def _strip_markdown_fence(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_message_content(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return canonical_json(response_json)
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return canonical_json(message)


def _headers(auth_token: str = "") -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    return headers


def _api_url(base_url: str, endpoint: str) -> str:
    normalized = base_url.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return f"{normalized}/{endpoint.lstrip('/')}"


def _endpoint_id(base_url: str) -> str:
    return base_url.rstrip("/")
