"""LLM clients for PAISDB model stages."""

from __future__ import annotations

import ast
import json
import os
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
    base_url: str = ""
    auth_token: str = ""
    endpoint_id: Optional[str] = None
    revision: str = ""
    hf_home: str = ""
    local_files_only: bool = True
    cuda_visible_devices: str = ""
    max_new_tokens: int = 300


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
    """Small PAIS stage client with OpenAI-compatible and local HF backends."""

    def __init__(self, timeout_s: float = 120.0):
        self.config = get_config()
        self.timeout_s = timeout_s
        self._hf_models: dict[tuple[Any, ...], tuple[Any, Any]] = {}

    def resolve_stage_config(self, stage: str) -> PaisStageConfig:
        """Resolve backend/model/base URL for a PAIS stage from config."""
        if stage == PaisStage.BENCHMARK_SCREEN.value:
            backend = self.config.pais_screen_backend
            model = self.config.pais_screen_model
            base_url = self.config.pais_screen_base_url
            token = self.config.pais_screen_auth_token
            model_var = "PAIS_SCREEN_MODEL"
            base_url_var = "PAIS_SCREEN_BASE_URL"
        elif stage == PaisStage.EVIDENCE_BRIEF.value:
            model = self.config.pais_evidence_brief_model
            base_url = self.config.pais_evidence_brief_base_url
            token = self.config.pais_evidence_brief_auth_token
            model_var = "PAIS_EVIDENCE_BRIEF_MODEL"
            base_url_var = "PAIS_EVIDENCE_BRIEF_BASE_URL"
        elif stage == PaisStage.STRUCTURED_EXTRACTION.value:
            model = self.config.pais_extraction_model
            base_url = self.config.pais_extraction_base_url
            token = self.config.pais_extraction_auth_token
            model_var = "PAIS_EXTRACTION_MODEL"
            base_url_var = "PAIS_EXTRACTION_BASE_URL"
        else:
            raise ValueError(f"Unknown PAIS stage: {stage}")

        if stage == PaisStage.BENCHMARK_SCREEN.value and backend == "hf_transformers":
            _require_model_config(stage, model, model_var)
            return PaisStageConfig(
                stage=stage,
                backend=backend,
                model=model,
                base_url="",
                auth_token="",
                endpoint_id=_hf_endpoint_id(
                    model=model,
                    revision=self.config.pais_screen_revision,
                    hf_home=self.config.pais_screen_hf_home,
                ),
                revision=self.config.pais_screen_revision,
                hf_home=self.config.pais_screen_hf_home,
                local_files_only=self.config.pais_screen_local_files_only,
                cuda_visible_devices=self.config.pais_screen_cuda_visible_devices,
                max_new_tokens=self.config.pais_screen_max_new_tokens,
            )
        if stage == PaisStage.BENCHMARK_SCREEN.value and backend != "openai_compatible":
            raise ValueError(
                f"Unsupported PAIS screen backend '{backend}'. "
                "Use 'hf_transformers' for local Mistral or 'openai_compatible' for a local server."
            )

        _require_stage_config(stage, model, base_url, model_var, base_url_var)
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

    def complete_json_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        schema_model: type[BaseModel],
        stage: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        structured: Optional[bool] = None,
    ) -> list[PaisLLMResult]:
        """Call a stage model for many independent JSON prompts.

        The local HF backend uses real tensor batching. OpenAI-compatible
        endpoints are intentionally left as a serial fallback here; callers
        that want server-side batching should submit concurrent requests.
        """
        if not messages_batch:
            return []
        stage_config = self.resolve_stage_config(stage)
        structured_mode = self.config.pais_structured_output_mode
        structured_output = structured if structured is not None else structured_mode == "json_schema"
        if stage_config.backend == "hf_transformers":
            return self._hf_completion_batch(
                stage_config=stage_config,
                messages_batch=messages_batch,
                schema_model=schema_model,
                structured=structured_output,
            )
        return [
            self._chat_completion(
                stage_config=stage_config,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                schema_model=schema_model,
                structured=structured_output,
            )
            for messages in messages_batch
        ]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts through the configured OpenAI-compatible embeddings endpoint."""
        if not texts:
            return []
        _require_stage_config(
            "embeddings",
            self.config.pais_embedding_model,
            self.config.pais_embedding_base_url,
            "PAIS_EMBEDDING_MODEL",
            "PAIS_EMBEDDING_BASE_URL",
        )
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
        if stage_config.backend == "hf_transformers":
            return self._hf_completion(
                stage_config=stage_config,
                messages=messages,
                schema_model=schema_model,
                structured=structured,
            )

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
                    parsed_json = _validate_or_normalize_schema(schema_model, parsed)
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

    def _hf_completion(
        self,
        stage_config: PaisStageConfig,
        messages: list[dict[str, str]],
        schema_model: Optional[type[BaseModel]],
        structured: bool,
    ) -> PaisLLMResult:
        started = time.monotonic()
        try:
            raw_output = self._hf_generate(_messages_to_prompt(messages), stage_config)
            parsed_json = None
            valid = True
            error_kind = None
            error_message = None
            if schema_model is not None:
                try:
                    parsed = parse_json_object(raw_output)
                    parsed_json = _validate_or_normalize_schema(schema_model, parsed)
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
                model_version=stage_config.revision or None,
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
                model_version=stage_config.revision or None,
            )

    def _hf_completion_batch(
        self,
        stage_config: PaisStageConfig,
        messages_batch: list[list[dict[str, str]]],
        schema_model: Optional[type[BaseModel]],
        structured: bool,
    ) -> list[PaisLLMResult]:
        started = time.monotonic()
        try:
            prompts = [_messages_to_prompt(messages) for messages in messages_batch]
            raw_outputs = self._hf_generate_batch(prompts, stage_config)
            elapsed_s = time.monotonic() - started
            per_item_elapsed = elapsed_s / max(len(raw_outputs), 1)
            results = []
            for raw_output in raw_outputs:
                parsed_json = None
                valid = True
                error_kind = None
                error_message = None
                if schema_model is not None:
                    try:
                        parsed = parse_json_object(raw_output)
                        parsed_json = _validate_or_normalize_schema(schema_model, parsed)
                    except (ValueError, ValidationError, TypeError) as exc:
                        valid = False
                        error_kind = "validation_error"
                        error_message = str(exc)
                results.append(
                    PaisLLMResult(
                        raw_output=raw_output,
                        parsed_json=parsed_json,
                        valid=valid,
                        elapsed_s=per_item_elapsed,
                        backend=stage_config.backend,
                        model_id=stage_config.model,
                        endpoint_id=stage_config.endpoint_id,
                        structured_output_used=structured,
                        error_kind=error_kind,
                        error_message=error_message,
                        model_version=stage_config.revision or None,
                    )
                )
            return results
        except Exception as exc:
            elapsed_s = time.monotonic() - started
            per_item_elapsed = elapsed_s / max(len(messages_batch), 1)
            return [
                PaisLLMResult(
                    raw_output="",
                    parsed_json=None,
                    valid=False,
                    elapsed_s=per_item_elapsed,
                    backend=stage_config.backend,
                    model_id=stage_config.model,
                    endpoint_id=stage_config.endpoint_id,
                    structured_output_used=structured,
                    error_kind=exc.__class__.__name__,
                    error_message=str(exc),
                    model_version=stage_config.revision or None,
                )
                for _ in messages_batch
            ]

    def _hf_generate(self, prompt: str, stage_config: PaisStageConfig) -> str:
        return self._hf_generate_batch([prompt], stage_config)[0]

    def _hf_generate_batch(self, prompts: list[str], stage_config: PaisStageConfig) -> list[str]:
        tokenizer, model = self._load_hf_model(stage_config)

        import torch

        tokenized = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
        inputs = {key: value.to("cuda" if torch.cuda.is_available() else "cpu") for key, value in tokenized.items()}
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=stage_config.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        decoded_inputs = tokenizer.batch_decode(tokenized["input_ids"], skip_special_tokens=True)
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generations = []
        for decoded_input, decoded_output in zip(decoded_inputs, decoded_outputs):
            if decoded_output.startswith(decoded_input):
                generations.append(decoded_output[len(decoded_input) :].strip())
            else:
                generations.append(decoded_output.split("Answer:", 1)[-1].strip())
        return generations

    def _load_hf_model(self, stage_config: PaisStageConfig) -> tuple[Any, Any]:
        cache_key = (
            stage_config.model,
            stage_config.revision,
            stage_config.hf_home,
            stage_config.local_files_only,
            stage_config.cuda_visible_devices,
        )
        if cache_key in self._hf_models:
            return self._hf_models[cache_key]

        _configure_hf_environment(stage_config)

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, logging

        logging.set_verbosity_error()
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA device is visible for local Mistral screening.")

        tokenizer = AutoTokenizer.from_pretrained(
            stage_config.model,
            local_files_only=stage_config.local_files_only,
            trust_remote_code=True,
            revision=stage_config.revision or None,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            stage_config.model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            local_files_only=stage_config.local_files_only,
            trust_remote_code=True,
            revision=stage_config.revision or None,
        )
        model.eval()
        self._hf_models[cache_key] = (tokenizer, model)
        return tokenizer, model


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse one JSON object from a model response."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _strip_markdown_fence(stripped)
    try:
        parsed = _parse_dict_literal(stripped)
    except ValueError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("No JSON object found in model output")
        parsed = _parse_dict_literal(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model output JSON must be an object")
    return parsed


def _parse_dict_literal(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError("Model output is not valid JSON or Python dict syntax") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Model output JSON must be an object")
    return parsed


def _validate_or_normalize_schema(schema_model: type[BaseModel], parsed: dict[str, Any]) -> dict[str, Any]:
    try:
        return schema_model.model_validate(parsed).model_dump(mode="json")
    except ValidationError:
        normalized = _normalize_schema_payload(schema_model, parsed)
        if normalized is None:
            raise
        return schema_model.model_validate(normalized).model_dump(mode="json")


def _normalize_schema_payload(schema_model: type[BaseModel], parsed: dict[str, Any]) -> Optional[dict[str, Any]]:
    if schema_model.__name__ == "PAISEvidenceBriefResult":
        return _normalize_brief_payload(parsed)
    if schema_model.__name__ == "PAISEvidenceExtractionResult":
        return _normalize_extraction_payload(parsed)
    return None


def _normalize_brief_payload(parsed: dict[str, Any]) -> Optional[dict[str, Any]]:
    if "embedding_text" not in parsed:
        return None

    key_entities = parsed.get("key_entities") if isinstance(parsed.get("key_entities"), dict) else {}
    if not key_entities:
        key_entities = {
            "pathogen": _entity_name(parsed.get("pathogen")),
            "disease_or_phenotype": _entity_name(parsed.get("disease_or_phenotype")),
            "host": _string_or_unknown(parsed.get("host")),
            "organism_or_model": _string_or_unknown(parsed.get("organism_or_model")),
            "tissue_or_sample": _string_or_unknown(parsed.get("tissue_or_sample")),
        }

    source_quotes = parsed.get("source_span_quotes")
    if not isinstance(source_quotes, list):
        source_quotes = parsed.get("source_support") or parsed.get("source_spans") or []
    source_quotes = [_source_text(item) for item in source_quotes if _source_text(item)]

    quality_flags = parsed.get("brief_quality_flags")
    if not isinstance(quality_flags, list):
        quality_flags = []
    quality_flags = [str(flag) for flag in quality_flags]

    expected = {"embedding_text", "key_entities", "brief_quality_flags", "source_span_quotes", "uncertainty_notes"}
    extra_keys = sorted(set(parsed).difference(expected))
    if extra_keys:
        quality_flags.append("brief_schema_normalized_from_extra_fields")

    return {
        "embedding_text": str(parsed["embedding_text"]),
        "key_entities": key_entities,
        "brief_quality_flags": sorted(set(quality_flags)),
        "source_span_quotes": source_quotes,
        "uncertainty_notes": parsed.get("uncertainty_notes")
        or parsed.get("limitations_or_uncertainty")
        or parsed.get("uncertainty"),
    }


def _normalize_extraction_payload(parsed: dict[str, Any]) -> Optional[dict[str, Any]]:
    if isinstance(parsed.get("pais_classification"), dict) and isinstance(parsed.get("evidence"), dict):
        return _sanitize_extraction_enums(parsed)

    evidence = parsed.get("evidence_extraction")
    if not isinstance(evidence, dict):
        return None

    article = parsed.get("article") or parsed.get("article_metadata") or {}
    pathogen = parsed.get("pathogen") or parsed.get("pathogen_candidate") or {}
    disease = parsed.get("disease_or_phenotype") or parsed.get("disease_or_phenotype_candidate") or {}
    findings = evidence.get("key_findings") if isinstance(evidence.get("key_findings"), list) else []
    source_spans = []
    finding_summaries = []
    for finding in findings:
        if isinstance(finding, dict):
            if finding.get("finding"):
                finding_summaries.append(str(finding["finding"]))
            spans = finding.get("source_spans") if isinstance(finding.get("source_spans"), list) else []
            source_spans.extend({"text": str(span), "source": "abstract"} for span in spans if str(span).strip())
        elif str(finding).strip():
            finding_summaries.append(str(finding))

    if not source_spans:
        source_spans = [
            {"text": item, "source": "abstract"} for item in _as_string_list(evidence.get("source_spans"))
        ]

    molecular_modalities = _as_string_list(evidence.get("molecular_data"))
    limitations = _as_string_list(evidence.get("limitations"))
    quality_flags = _as_string_list(parsed.get("quality_flags"))
    quality_flags.append("extraction_schema_normalized_from_alternative_shape")

    return {
        "article": {
            "pmid": article.get("pmid") if isinstance(article, dict) else None,
            "doi": article.get("doi") if isinstance(article, dict) else None,
            "title": article.get("title", "unknown") if isinstance(article, dict) else "unknown",
            "publication_year": article.get("publication_year") if isinstance(article, dict) else None,
        },
        "pathogen": _normalize_entity_extraction(pathogen),
        "disease_or_phenotype": _normalize_entity_extraction(disease),
        "host_context": {
            "host_name": evidence.get("host"),
            "host_taxid": None,
            "host_type": _host_type_from_text(evidence.get("host") or evidence.get("organism_or_model")),
            "species": evidence.get("organism_or_model"),
            "tissue_or_sample": evidence.get("tissue_or_sample"),
            "cohort_or_model_description": evidence.get("study_design") or evidence.get("extraction_source"),
        },
        "relationship": {
            "relation_type": _relation_type_from_text(evidence.get("relation_type") or evidence.get("finding")),
            "timing_after_infection": evidence.get("timing") or evidence.get("timing_after_infection"),
            "statement": "; ".join(finding_summaries) or str(evidence.get("finding") or ""),
        },
        "pais_classification": {
            "pais_category": _pais_category_from_text(
                evidence.get("pais_category") or evidence.get("pa_is_category")
            ),
            "rationale": str(evidence.get("pais_category") or evidence.get("pa_is_category") or ""),
        },
        "evidence": {
            "evidence_type": _evidence_type_from_text(evidence.get("evidence_type") or evidence.get("study_design")),
            "disease_phenotypes": _as_string_list(evidence.get("disease_phenotypes")),
            "pathogen_details": _as_string_list(evidence.get("pathogen_details")),
            "summary": "; ".join(finding_summaries) or str(evidence.get("finding") or ""),
        },
        "mechanism": {"mechanism_summary": evidence.get("mechanism")},
        "molecular_data": {
            "molecular_data_summary": (
                ", ".join(molecular_modalities) if molecular_modalities else evidence.get("molecular_data")
            ),
            "molecular_modalities": molecular_modalities,
        },
        "source_spans": source_spans,
        "confidence": {
            "confidence": _confidence_from_text(evidence.get("confidence") or parsed.get("confidence")),
            "rationale": str(evidence.get("confidence_rationale") or ""),
        },
        "limitations": limitations,
        "disagreement_with_screen": bool(parsed.get("disagreement_with_screen", False)),
        "quality_flags": sorted(set(quality_flags)),
    }


def _sanitize_extraction_enums(parsed: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(parsed)
    normalized["host_context"] = dict(normalized.get("host_context") or {})
    normalized["relationship"] = dict(normalized.get("relationship") or {})
    normalized["pais_classification"] = dict(normalized.get("pais_classification") or {})
    normalized["evidence"] = dict(normalized.get("evidence") or {})
    normalized["confidence"] = dict(normalized.get("confidence") or {})

    normalized["host_context"]["host_type"] = _host_type_from_text(normalized["host_context"].get("host_type"))
    normalized["relationship"]["relation_type"] = _relation_type_from_text(
        normalized["relationship"].get("relation_type") or normalized["relationship"].get("statement")
    )
    normalized["pais_classification"]["pais_category"] = _pais_category_from_text(
        normalized["pais_classification"].get("pais_category")
    )
    normalized["evidence"]["evidence_type"] = _evidence_type_from_text(normalized["evidence"].get("evidence_type"))
    normalized["confidence"]["confidence"] = _confidence_from_text(normalized["confidence"].get("confidence"))

    quality_flags = _as_string_list(normalized.get("quality_flags"))
    quality_flags.append("extraction_schema_normalized_enum_values")
    normalized["quality_flags"] = sorted(set(quality_flags))
    return normalized


def _normalize_entity_extraction(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        identifiers = {}
        for key in ("ncbi_taxid", "doid", "hpo_id", "mondo_id"):
            if value.get(key):
                identifiers[key] = value[key]
        return {
            "name": value.get("name") or value.get("normalized_name") or "unknown",
            "normalized_name": value.get("normalized_name") or value.get("name") or "unknown",
            "identifiers": identifiers,
        }
    text = _string_or_unknown(value)
    return {"name": text, "normalized_name": text, "identifiers": {}}


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _host_type_from_text(value: Any) -> str:
    text = _string_or_unknown(value).casefold()
    if text in {"human", "animal_model", "cell_line", "organoid", "in_vitro", "mixed", "unknown"}:
        return text
    if "human" in text or "patient" in text:
        return "human"
    if "mouse" in text or "mice" in text or "animal" in text:
        return "animal_model"
    if "cell" in text:
        return "cell_line"
    if "organoid" in text:
        return "organoid"
    if "vitro" in text:
        return "in_vitro"
    return "unknown"


def _relation_type_from_text(value: Any) -> str:
    text = _string_or_unknown(value).casefold()
    if text in {
        "causes",
        "associated_with",
        "increases_risk",
        "worsens",
        "protective",
        "no_significant_association",
        "mentions_only",
        "unclear",
    }:
        return text
    if "no significant" in text:
        return "no_significant_association"
    if "risk" in text:
        return "increases_risk"
    if "worsen" in text:
        return "worsens"
    if "protect" in text:
        return "protective"
    if text != "unknown":
        return "associated_with"
    return "unclear"


def _pais_category_from_text(value: Any) -> str:
    text = _string_or_unknown(value).casefold()
    if text in {
        "true_pais",
        "post_acute_sequela",
        "acute_infection",
        "chronic_active_infection",
        "persistent_pathogen_or_antigen",
        "adjacent_mechanism",
        "non_pais",
        "unclear",
    }:
        return text
    if "post" in text and ("acute" in text or "sequela" in text):
        return "post_acute_sequela"
    if "acute" in text:
        return "acute_infection"
    if "chronic" in text:
        return "chronic_active_infection"
    if "persistent" in text or "antigen" in text:
        return "persistent_pathogen_or_antigen"
    if "non" in text and "pais" in text:
        return "non_pais"
    if "mechanism" in text or "host" in text or "pathogen" in text:
        return "adjacent_mechanism"
    return "unclear"


def _evidence_type_from_text(value: Any) -> str:
    text = _string_or_unknown(value).casefold()
    if text in {
        "clinical_cohort",
        "case_control",
        "longitudinal_cohort",
        "case_report",
        "animal_model",
        "cell_model",
        "molecular_assay",
        "omics_study",
        "review",
        "database_or_mining",
        "unclear",
    }:
        return text
    if "longitudinal" in text or "prospective" in text or "follow" in text:
        return "longitudinal_cohort"
    if "case control" in text or "case-control" in text:
        return "case_control"
    if "case report" in text:
        return "case_report"
    if "animal" in text or "mouse" in text or "mice" in text:
        return "animal_model"
    if "cell" in text:
        return "cell_model"
    if "omic" in text:
        return "omics_study"
    if "assay" in text or "molecular" in text:
        return "molecular_assay"
    if "review" in text:
        return "review"
    if text != "unknown":
        return "clinical_cohort"
    return "unclear"


def _confidence_from_text(value: Any) -> str:
    text = _string_or_unknown(value).casefold()
    if text in {"high", "medium", "low", "unknown"}:
        return text
    return "unknown"


def _entity_name(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("name") or value.get("normalized_name")
    return _string_or_unknown(value)


def _string_or_unknown(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip()
    return text or "unknown"


def _source_text(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("text") or value.get("quote") or value.get("span")
    if value is None:
        return ""
    return str(value).strip()


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


def _hf_endpoint_id(model: str, revision: str = "", hf_home: str = "") -> str:
    bits = [f"hf:{model}"]
    if revision:
        bits.append(f"revision:{revision}")
    if hf_home:
        bits.append(f"hf_home:{hf_home}")
    return "|".join(bits)


def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
    if len(messages) == 1 and messages[0].get("role") == "user":
        return messages[0].get("content", "")
    return "\n\n".join(f"{message.get('role', 'user')}: {message.get('content', '')}" for message in messages)


def _configure_hf_environment(stage_config: PaisStageConfig) -> None:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if stage_config.cuda_visible_devices:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", stage_config.cuda_visible_devices)
    if stage_config.hf_home:
        os.environ.setdefault("HF_HOME", stage_config.hf_home)
    if stage_config.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _require_model_config(stage: str, model: str, model_var: str) -> None:
    if not model:
        raise ValueError(
            f"PAIS stage '{stage}' is not configured. Set {model_var} before running model calls. "
            "Database initialization does not require PAIS model configuration."
        )


def _require_stage_config(
    stage: str,
    model: str,
    base_url: str,
    model_var: str,
    base_url_var: str,
) -> None:
    missing = []
    if not model:
        missing.append(model_var)
    if not base_url:
        missing.append(base_url_var)
    if missing:
        raise ValueError(
            f"PAIS stage '{stage}' is not configured. Set {', '.join(missing)} before running model calls. "
            "Database initialization does not require PAIS model configuration."
        )
