"""Versioned prompt templates for PAISDB model stages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from abstracts_explorer.pais_schemas import (
    BenchmarkScreenResult,
    PAISEvidenceBriefResult,
    PAISEvidenceExtractionResult,
    PaisCandidateInput,
)


@dataclass(frozen=True)
class PromptSpec:
    """A versioned prompt template with provenance helpers."""

    name: str
    version: str
    system: str
    user: str

    @property
    def sha256(self) -> str:
        return sha256_text(f"{self.name}\n{self.version}\n{self.system}\n{self.user}")


BENCHMARK_SCREEN_PROMPT = PromptSpec(
    name="paper_zero_shot_v1",
    version="paisdb2-paper-zero-shot-v1",
    system="",
    user=(
        "I seek assistance with a systematic review focused on the direct relationship between pathogens and "
        "diseases, specifically {disease}. I’ll provide the title and abstract of a particular journal article "
        "and would appreciate an assessment for its inclusion based on the following criteria:\n\n"
        "1. The title or abstract provides sufficient evidence of a direct relationship between the disease "
        "({disease}) and the pathogen ({pathogen}).\n"
        "2. The title or abstract investigates the Pathogen ({pathogen}) and reports evidence for the Disease "
        "({disease}).\n"
        "3. The title or abstract investigates the Disease ({disease}) and reports evidence for the Pathogen "
        "({pathogen}).\n"
        "4. The title or abstract states the association between the Pathogen ({pathogen}) and the Disease "
        "({disease}), but does not focus on it.\n"
        "5. The title and abstract present data or findings supporting this association.\n\n"
        "Exclusion criteria:\n"
        "1. The title and abstract do not provide sufficient evidence of a direct relationship between the "
        "disease ({disease}) and the pathogen ({pathogen}).\n\n"
        "Please provide the assessment in the following dictionary format:\n"
        '{{"relationship": 1, "unrelated": 0}} if there is a relationship, or '
        '{{"relationship": 0, "unrelated": 1}} if the study should be excluded.\n\n'
        "Note: only one value can be 1 at a time.\n\n"
        "Title: {title}\n\n"
        "Abstract: {abstract}\n\n"
        "You are required to classify a journal article based solely on the given title and abstract. Do not "
        "use any external knowledge or assumptions beyond the text provided. Your decision must be strictly "
        "based on the information within the title and abstract.\n\n"
        "Respond only in the dictionary format with no explanation.\n\n"
        "Answer:"
    ),
)


EVIDENCE_BRIEF_PROMPT = PromptSpec(
    name="pais_evidence_brief",
    version="2026-05-27-v2",
    system=(
        "Return one valid JSON object only. Use only supplied biomedical source text. "
        "Never write markdown, comments, explanations, or text after the JSON object."
    ),
    user=(
        "Article metadata:\n{article_json}\n\n"
        "Pathogen candidate:\n{pathogen_json}\n\n"
        "Disease or phenotype candidate:\n{disease_json}\n\n"
        "Title:\n{title}\n\n"
        "Abstract:\n{abstract}\n\n"
        "Create a compact PAIS evidence brief. Use only title/abstract. Do not infer external facts. "
        "Use unknown when host, timing, mechanism, or molecular data are not stated. Keep every string short. "
        "Do not create extra top-level keys. Use double quotes for all JSON keys and strings. "
        "Do not use trailing commas. Output must be parseable by json.loads.\n\n"
        "The embedding_text should follow this shape:\n"
        "PAIS evidence brief. Article: <title or PMID>. Candidate relation: <pathogen> -> <disease>. "
        "Host/model: <...>. Timing: <...>. Evidence type: <...>. Finding: <...>. "
        "PAIS category: <...>. Mechanism: <...>. Molecular data: <...>. "
        "Limitations/uncertainty: <...>. Source support: <...>.\n\n"
        "Constraints: embedding_text <= 900 characters; source_span_quotes <= 2 items; "
        "each quote <= 180 characters.\n\n"
        "Return exactly this minified JSON shape:\n"
        '{{"embedding_text":"...","key_entities":{{"pathogen":"...","disease_or_phenotype":"...",'
        '"host":"unknown","organism_or_model":"unknown","tissue_or_sample":"unknown"}},'
        '"brief_quality_flags":[],"source_span_quotes":[],"uncertainty_notes":null}}'
    ),
)


STRUCTURED_EXTRACTION_PROMPT = PromptSpec(
    name="pais_structured_extraction",
    version="2026-05-27-v2",
    system=(
        "Return one valid JSON object only. Use only supplied biomedical source text and evidence brief. "
        "Never write markdown, comments, explanations, or text after the JSON object."
    ),
    user=(
        "Article metadata:\n{article_json}\n\n"
        "Pathogen candidate:\n{pathogen_json}\n\n"
        "Disease or phenotype candidate:\n{disease_json}\n\n"
        "Evidence brief:\n{brief_json}\n\n"
        "Title:\n{title}\n\n"
        "Abstract:\n{abstract}\n\n"
        "Fill PAISEvidenceExtractionResult. Keep it source-grounded and concise. "
        "Do not invent values that are not supported by the supplied text. "
        "Use double quotes for all JSON keys and strings. Do not use trailing commas. "
        "Output must be parseable by json.loads.\n\n"
        "Allowed enum values:\n"
        "host_type: human, animal_model, cell_line, organoid, in_vitro, mixed, unknown\n"
        "relation_type: causes, associated_with, increases_risk, worsens, protective, "
        "no_significant_association, mentions_only, unclear\n"
        "pais_category: true_pais, post_acute_sequela, acute_infection, chronic_active_infection, "
        "persistent_pathogen_or_antigen, adjacent_mechanism, non_pais, unclear\n"
        "evidence_type: clinical_cohort, case_control, longitudinal_cohort, case_report, animal_model, "
        "cell_model, molecular_assay, omics_study, review, database_or_mining, unclear\n"
        "confidence: high, medium, low, unknown\n\n"
        "If no allowed enum fits, use unclear for relation_type/pais_category/evidence_type and unknown for "
        "host_type/confidence. Never create enum values such as biomarker, transmission, prospective study, "
        "or vaccination strategy.\n\n"
        "Length limits: statement <= 240 characters; summary <= 360 characters; rationale <= 180 characters; "
        "source_spans <= 2 items; each source_spans text <= 180 characters; limitations <= 3 items.\n\n"
        "Return exactly this minified JSON shape:\n"
        '{{"article":{{"pmid":null,"doi":null,"title":"...","publication_year":null}},'
        '"pathogen":{{"name":"...","normalized_name":"...","identifiers":{{}}}},'
        '"disease_or_phenotype":{{"name":"...","normalized_name":"...","identifiers":{{}}}},'
        '"host_context":{{"host_name":null,"host_taxid":null,"host_type":"unknown","species":null,'
        '"tissue_or_sample":null,"cohort_or_model_description":null}},'
        '"relationship":{{"relation_type":"unclear","timing_after_infection":null,"statement":"..."}},'
        '"pais_classification":{{"pais_category":"unclear","rationale":"..."}},'
        '"evidence":{{"evidence_type":"unclear","disease_phenotypes":[],"pathogen_details":[],"summary":"..."}},'
        '"mechanism":{{"mechanism_summary":null}},'
        '"molecular_data":{{"molecular_data_summary":null,"molecular_modalities":[]}},'
        '"source_spans":[{{"text":"...","source":"abstract"}}],'
        '"confidence":{{"confidence":"unknown","rationale":"..."}},'
        '"limitations":[],"disagreement_with_screen":false,"quality_flags":[]}}'
    ),
)


def sha256_text(text: str) -> str:
    """Return the SHA-256 hex digest for text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(data: Any) -> str:
    """Serialize data for stable hashing and prompt interpolation."""
    if isinstance(data, BaseModel):
        data = data.model_dump(mode="json")
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def schema_sha256(schema_model: type[BaseModel]) -> str:
    """Return a stable hash of a Pydantic JSON schema."""
    return sha256_text(canonical_json(schema_model.model_json_schema()))


def build_benchmark_screen_messages(candidate: PaisCandidateInput) -> list[dict[str, str]]:
    """Build chat messages for the benchmark screen stage."""
    user = BENCHMARK_SCREEN_PROMPT.user.format(
        pathogen=candidate.pathogen.name,
        disease=candidate.disease.name,
        title=candidate.article.title,
        abstract=candidate.article.abstract,
    )
    messages = []
    if BENCHMARK_SCREEN_PROMPT.system:
        messages.append({"role": "system", "content": BENCHMARK_SCREEN_PROMPT.system})
    messages.append({"role": "user", "content": user})
    return messages


def build_evidence_brief_messages(
    candidate: PaisCandidateInput, screen_result: BenchmarkScreenResult
) -> list[dict[str, str]]:
    """Build chat messages for the evidence brief stage."""
    user = EVIDENCE_BRIEF_PROMPT.user.format(
        article_json=canonical_json(candidate.article),
        pathogen_json=canonical_json(candidate.pathogen),
        disease_json=canonical_json(candidate.disease),
        title=candidate.article.title,
        abstract=candidate.article.abstract,
    )
    return [
        {"role": "system", "content": EVIDENCE_BRIEF_PROMPT.system},
        {"role": "user", "content": user},
    ]


def build_structured_extraction_messages(
    candidate: PaisCandidateInput,
    screen_result: BenchmarkScreenResult,
    brief_result: PAISEvidenceBriefResult,
) -> list[dict[str, str]]:
    """Build chat messages for the structured extraction stage."""
    user = STRUCTURED_EXTRACTION_PROMPT.user.format(
        article_json=canonical_json(candidate.article),
        pathogen_json=canonical_json(candidate.pathogen),
        disease_json=canonical_json(candidate.disease),
        brief_json=canonical_json(brief_result),
        title=candidate.article.title,
        abstract=candidate.article.abstract,
    )
    return [
        {"role": "system", "content": STRUCTURED_EXTRACTION_PROMPT.system},
        {"role": "user", "content": user},
    ]


def prompt_for_stage(stage: str) -> PromptSpec:
    """Return the prompt spec for a PAIS stage."""
    if stage == "benchmark_screen":
        return BENCHMARK_SCREEN_PROMPT
    if stage == "evidence_brief":
        return EVIDENCE_BRIEF_PROMPT
    if stage == "structured_extraction":
        return STRUCTURED_EXTRACTION_PROMPT
    raise ValueError(f"Unknown PAIS stage: {stage}")


def schema_for_stage(stage: str) -> type[BaseModel]:
    """Return the Pydantic schema for a PAIS stage."""
    if stage == "benchmark_screen":
        return BenchmarkScreenResult
    if stage == "evidence_brief":
        return PAISEvidenceBriefResult
    if stage == "structured_extraction":
        return PAISEvidenceExtractionResult
    raise ValueError(f"Unknown PAIS stage: {stage}")
