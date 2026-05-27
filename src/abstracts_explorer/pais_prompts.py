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
    name="pais_benchmark_screen",
    version="2026-05-27-v1",
    system=(
        "You are a biomedical relation-classification assistant for PAISDB. "
        "Use only the provided title and abstract. Do not use external knowledge. "
        "Your task is to decide whether the article supports a relationship between "
        "the specified pathogen and disease/phenotype candidate. Return only the requested JSON object."
    ),
    user=(
        "Pathogen candidate:\n{name_pathogen}\n\n"
        "Disease or phenotype candidate:\n{name_disease}\n\n"
        "Title:\n{title}\n\n"
        "Abstract:\n{abstract}\n\n"
        "Question:\n"
        "Does this title/abstract support a relationship between the specified pathogen and the specified "
        "disease/phenotype?\n\n"
        "Decision rules:\n"
        "1. Return relationship=1 and unrelated=0 if the text investigates or states evidence connecting "
        "this pathogen to this disease/phenotype.\n"
        "2. Return relationship=0 and unrelated=1 if the text only co-mentions them, lists them without "
        "relation evidence, or does not discuss the specified pair.\n"
        "3. Do not infer a relation from substring overlap. If the disease term appears only inside a pathogen "
        "name, such as Japanese encephalitis virus containing encephalitis, this is not sufficient.\n"
        "4. Use only the title and abstract. Do not add outside knowledge.\n"
        "5. Include short evidence-span quotes from the title/abstract when possible.\n"
        "6. Keep the rationale short.\n\n"
        "Return JSON matching the BenchmarkScreenResult schema."
    ),
)


EVIDENCE_BRIEF_PROMPT = PromptSpec(
    name="pais_evidence_brief",
    version="2026-05-27-v1",
    system=(
        "You create compact, source-grounded PAIS evidence briefs for embedding and retrieval. "
        "Use only the supplied article text and benchmark screen result. Return only JSON."
    ),
    user=(
        "Article metadata:\n{article_json}\n\n"
        "Pathogen candidate:\n{pathogen_json}\n\n"
        "Disease or phenotype candidate:\n{disease_json}\n\n"
        "Benchmark screen result:\n{screen_json}\n\n"
        "Title:\n{title}\n\n"
        "Abstract:\n{abstract}\n\n"
        "Create a compact PAIS evidence brief. Include pathogen, disease/phenotype, host, timing after "
        "infection, study design, evidence type, mechanism, molecular data, and uncertainty if available. "
        "Do not invent missing host, mechanism, modality, or timing. Explicitly state unknown when relevant "
        "information is not in the source text. Keep source quotes short.\n\n"
        "The embedding_text should follow this shape:\n"
        "PAIS evidence brief. Article: <title or PMID>. Candidate relation: <pathogen> -> <disease>. "
        "Benchmark screen: <positive/uncertain; confidence>. Host/model: <...>. Timing: <...>. "
        "Evidence type: <...>. Finding: <...>. PAIS category: <...>. Mechanism: <...>. "
        "Molecular data: <...>. Limitations/uncertainty: <...>. Source support: <...>.\n\n"
        "Return JSON matching the PAISEvidenceBriefResult schema."
    ),
)


STRUCTURED_EXTRACTION_PROMPT = PromptSpec(
    name="pais_structured_extraction",
    version="2026-05-27-v1",
    system=(
        "You convert source-grounded PAIS evidence into a strict database record. "
        "Do not use external knowledge. Use unknown rather than inventing values. Return only JSON."
    ),
    user=(
        "Article metadata:\n{article_json}\n\n"
        "Pathogen candidate:\n{pathogen_json}\n\n"
        "Disease or phenotype candidate:\n{disease_json}\n\n"
        "Benchmark screen result:\n{screen_json}\n\n"
        "Evidence brief:\n{brief_json}\n\n"
        "Title:\n{title}\n\n"
        "Abstract:\n{abstract}\n\n"
        "Fill PAISEvidenceExtractionResult. Use enum values from the schema. Keep the extraction "
        "source-grounded. Source spans should be snippets from the title, abstract, or brief. If the "
        "extraction disagrees with the benchmark screen, set disagreement_with_screen=true and add a "
        "quality flag. Do not overwrite the benchmark screen decision.\n\n"
        "Return JSON matching the PAISEvidenceExtractionResult schema."
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
        name_pathogen=_entity_display(candidate.pathogen.model_dump(mode="json")),
        name_disease=_entity_display(candidate.disease.model_dump(mode="json")),
        title=candidate.article.title,
        abstract=candidate.article.abstract,
    )
    return [
        {"role": "system", "content": BENCHMARK_SCREEN_PROMPT.system},
        {"role": "user", "content": user},
    ]


def build_evidence_brief_messages(
    candidate: PaisCandidateInput, screen_result: BenchmarkScreenResult
) -> list[dict[str, str]]:
    """Build chat messages for the evidence brief stage."""
    user = EVIDENCE_BRIEF_PROMPT.user.format(
        article_json=canonical_json(candidate.article),
        pathogen_json=canonical_json(candidate.pathogen),
        disease_json=canonical_json(candidate.disease),
        screen_json=canonical_json(screen_result),
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
        screen_json=canonical_json(screen_result),
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


def _entity_display(entity: dict[str, Any]) -> str:
    name = entity.get("name", "")
    identifiers = {key: value for key, value in entity.items() if key != "synonyms" and value}
    return f"{name}\n{canonical_json(identifiers)}"
