"""Pydantic schemas for PAISDB evidence extraction."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrEnum(str, Enum):
    """String enum base class with readable values."""

    def __str__(self) -> str:
        return self.value


class PaisStage(StrEnum):
    BENCHMARK_SCREEN = "benchmark_screen"
    EVIDENCE_BRIEF = "evidence_brief"
    STRUCTURED_EXTRACTION = "structured_extraction"


class ScreenStatus(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNCERTAIN = "uncertain"
    INVALID = "invalid"
    ERROR = "error"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class ExclusionReason(StrEnum):
    NO_RELATION = "no_relation"
    CO_MENTION_ONLY = "co_mention_only"
    DISEASE_TERM_ONLY_INSIDE_PATHOGEN_NAME = "disease_term_only_inside_pathogen_name"
    PATHOGEN_TERM_ONLY_INSIDE_DISEASE_NAME = "pathogen_term_only_inside_disease_name"
    ACUTE_ONLY_NOT_POST_ACUTE = "acute_only_not_post_acute"
    NO_PATHOGEN_EVIDENCE = "no_pathogen_evidence"
    NO_DISEASE_EVIDENCE = "no_disease_evidence"
    NEGATED_OR_NO_SIGNIFICANT_ASSOCIATION = "negated_or_no_significant_association"
    UNCLEAR = "unclear"


class HostType(StrEnum):
    HUMAN = "human"
    ANIMAL_MODEL = "animal_model"
    CELL_LINE = "cell_line"
    ORGANOID = "organoid"
    IN_VITRO = "in_vitro"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class PaisCategory(StrEnum):
    TRUE_PAIS = "true_pais"
    POST_ACUTE_SEQUELA = "post_acute_sequela"
    ACUTE_INFECTION = "acute_infection"
    CHRONIC_ACTIVE_INFECTION = "chronic_active_infection"
    PERSISTENT_PATHOGEN_OR_ANTIGEN = "persistent_pathogen_or_antigen"
    ADJACENT_MECHANISM = "adjacent_mechanism"
    NON_PAIS = "non_pais"
    UNCLEAR = "unclear"


class RelationType(StrEnum):
    CAUSES = "causes"
    ASSOCIATED_WITH = "associated_with"
    INCREASES_RISK = "increases_risk"
    WORSENS = "worsens"
    PROTECTIVE = "protective"
    NO_SIGNIFICANT_ASSOCIATION = "no_significant_association"
    MENTIONS_ONLY = "mentions_only"
    UNCLEAR = "unclear"


class EvidenceType(StrEnum):
    CLINICAL_COHORT = "clinical_cohort"
    CASE_CONTROL = "case_control"
    LONGITUDINAL_COHORT = "longitudinal_cohort"
    CASE_REPORT = "case_report"
    ANIMAL_MODEL = "animal_model"
    CELL_MODEL = "cell_model"
    MOLECULAR_ASSAY = "molecular_assay"
    OMICS_STUDY = "omics_study"
    REVIEW = "review"
    DATABASE_OR_MINING = "database_or_mining"
    UNCLEAR = "unclear"


class EmbeddingStatus(StrEnum):
    PENDING = "pending"
    EMBEDDED = "embedded"
    FAILED = "failed"


class ArticleInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pmid: Optional[str] = None
    doi: Optional[str] = None
    title: str
    abstract: str
    journal: Optional[str] = None
    publication_year: Optional[int] = None
    publication_date: Optional[str] = None
    publication_type: Optional[str] = None
    source: Optional[str] = None
    source_url: Optional[str] = None


class EntityInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    normalized_name: Optional[str] = None
    synonyms: list[str] = Field(default_factory=list)


class PathogenInput(EntityInput):
    ncbi_taxid: Optional[str] = None
    taxonomic_rank: Optional[str] = None
    strain_or_variant: Optional[str] = None


class DiseasePhenotypeInput(EntityInput):
    doid: Optional[str] = None
    hpo_id: Optional[str] = None
    mondo_id: Optional[str] = None


class PaisCandidateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    article: ArticleInput
    pathogen: PathogenInput
    disease: DiseasePhenotypeInput


class BenchmarkScreenResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relationship: Literal[0, 1]
    unrelated: Literal[0, 1]
    confidence: Confidence = Confidence.UNKNOWN
    decision_rationale_short: str = ""
    evidence_span_quotes: list[str] = Field(default_factory=list)
    exclusion_reason: Optional[ExclusionReason] = None
    quality_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_inverse_labels(self) -> "BenchmarkScreenResult":
        if self.relationship == self.unrelated:
            raise ValueError("relationship and unrelated must be inverse binary labels")
        return self


class KeyEntities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pathogen: str = "unknown"
    disease_or_phenotype: str = "unknown"
    host: str = "unknown"
    organism_or_model: str = "unknown"
    tissue_or_sample: str = "unknown"


class PAISEvidenceBriefResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding_text: str
    key_entities: KeyEntities = Field(default_factory=KeyEntities)
    brief_quality_flags: list[str] = Field(default_factory=list)
    source_span_quotes: list[str] = Field(default_factory=list)
    uncertainty_notes: Optional[str] = None


class ArticleExtraction(BaseModel):
    model_config = ConfigDict(extra="allow")

    pmid: Optional[str] = None
    doi: Optional[str] = None
    title: str = "unknown"
    publication_year: Optional[int] = None


class EntityExtraction(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = "unknown"
    normalized_name: str = "unknown"
    identifiers: dict[str, Any] = Field(default_factory=dict)


class HostContextExtraction(BaseModel):
    model_config = ConfigDict(extra="allow")

    host_name: Optional[str] = None
    host_taxid: Optional[str] = None
    host_type: HostType = HostType.UNKNOWN
    species: Optional[str] = None
    tissue_or_sample: Optional[str] = None
    cohort_or_model_description: Optional[str] = None


class RelationshipExtraction(BaseModel):
    model_config = ConfigDict(extra="allow")

    relation_type: RelationType = RelationType.UNCLEAR
    timing_after_infection: Optional[str] = None
    statement: str = ""


class PaisClassificationExtraction(BaseModel):
    model_config = ConfigDict(extra="allow")

    pais_category: PaisCategory = PaisCategory.UNCLEAR
    rationale: str = ""


class EvidenceExtraction(BaseModel):
    model_config = ConfigDict(extra="allow")

    evidence_type: EvidenceType = EvidenceType.UNCLEAR
    disease_phenotypes: list[str] = Field(default_factory=list)
    pathogen_details: list[str] = Field(default_factory=list)
    summary: str = ""


class MechanismExtraction(BaseModel):
    model_config = ConfigDict(extra="allow")

    mechanism_summary: Optional[str] = None


class MolecularDataExtraction(BaseModel):
    model_config = ConfigDict(extra="allow")

    molecular_data_summary: Optional[str] = None
    molecular_modalities: list[str] = Field(default_factory=list)


class SourceSpan(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str
    source: str = "abstract"


class ConfidenceExtraction(BaseModel):
    model_config = ConfigDict(extra="allow")

    confidence: Confidence = Confidence.UNKNOWN
    rationale: str = ""


class PAISEvidenceExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    article: ArticleExtraction = Field(default_factory=ArticleExtraction)
    pathogen: EntityExtraction = Field(default_factory=EntityExtraction)
    disease_or_phenotype: EntityExtraction = Field(default_factory=EntityExtraction)
    host_context: HostContextExtraction = Field(default_factory=HostContextExtraction)
    relationship: RelationshipExtraction = Field(default_factory=RelationshipExtraction)
    pais_classification: PaisClassificationExtraction = Field(default_factory=PaisClassificationExtraction)
    evidence: EvidenceExtraction = Field(default_factory=EvidenceExtraction)
    mechanism: MechanismExtraction = Field(default_factory=MechanismExtraction)
    molecular_data: MolecularDataExtraction = Field(default_factory=MolecularDataExtraction)
    source_spans: list[SourceSpan] = Field(default_factory=list)
    confidence: ConfidenceExtraction = Field(default_factory=ConfidenceExtraction)
    limitations: list[str] = Field(default_factory=list)
    disagreement_with_screen: bool = False
    quality_flags: list[str] = Field(default_factory=list)
