"""PAISDB candidate screening and evidence-building pipeline."""

from __future__ import annotations

import json
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from abstracts_explorer.database import DatabaseError, DatabaseManager
from abstracts_explorer.db_models import (
    Article,
    CandidateRelation,
    DiseasePhenotype,
    EmbeddingRecord,
    HostContext,
    ModelRun,
    PAISEvidenceRecord,
    Pathogen,
)
from abstracts_explorer.pais_llm import OpenAICompatiblePaisClient, PaisLLMResult
from abstracts_explorer.pais_prompts import (
    build_benchmark_screen_messages,
    build_evidence_brief_messages,
    build_structured_extraction_messages,
    canonical_json,
    prompt_for_stage,
    schema_for_stage,
    schema_sha256,
    sha256_text,
)
from abstracts_explorer.pais_schemas import (
    BenchmarkScreenResult,
    Confidence,
    EmbeddingStatus,
    HostType,
    PAISEvidenceBriefResult,
    PAISEvidenceExtractionResult,
    PaisCandidateInput,
    PaisStage,
    ScreenStatus,
)


def run_candidate_pipeline(
    candidate_data: PaisCandidateInput | dict[str, Any],
    database: DatabaseManager,
    llm_client: Optional[OpenAICompatiblePaisClient] = None,
    structured: Optional[bool] = None,
    initial_quality_flags: Optional[list[str]] = None,
    generation_provider_id: Optional[str] = None,
) -> dict[str, Any]:
    """Run the full PAIS candidate pipeline and persist provenance."""
    candidate = PaisCandidateInput.model_validate(candidate_data)
    client = llm_client or OpenAICompatiblePaisClient()
    database.create_tables()
    session = _session(database)

    try:
        article = _upsert_article(session, candidate)
        pathogen = _upsert_pathogen(session, candidate)
        disease = _upsert_disease(session, candidate)
        relation = _upsert_candidate_relation(session, article, pathogen, disease)
        if initial_quality_flags:
            _merge_relation_quality_flags(relation, initial_quality_flags)

        screen_call, screen_result, screen_messages = screen_candidate(candidate, client, structured=structured)
        screen_run = _persist_model_run(
            session=session,
            stage=PaisStage.BENCHMARK_SCREEN.value,
            call=screen_call,
            article_id=article.id,
            candidate_relation_id=relation.id,
            evidence_record_id=None,
            input_payload={"candidate": candidate.model_dump(mode="json"), "messages": screen_messages},
        )
        screen_status = _apply_screen_result(relation, screen_call, screen_result)
        session.flush()

        model_run_ids = [screen_run.id]
        summary = {
            "candidate_relation_id": relation.id,
            "benchmark_relationship": relation.benchmark_relationship,
            "benchmark_unrelated": relation.benchmark_unrelated,
            "screen_status": screen_status.value,
            "screen_confidence": relation.screen_confidence,
            "server2_called": False,
            "evidence_record_id": None,
            "embedding_record_id": None,
            "model_run_ids": model_run_ids,
            "hosted_disagreement_flag": bool(relation.hosted_disagreement_flag),
            "quality_flags": _json_loads(relation.quality_flags_json, default=[]),
            "generation_provider_id": generation_provider_id,
        }

        allow_invalid_adjudication = bool(
            getattr(getattr(client, "config", None), "pais_allow_adjudication_on_invalid_screen", False)
        )
        if not _should_call_hosted(screen_status, screen_result, allow_invalid_adjudication):
            session.commit()
            return summary

        screen_for_hosted = screen_result or _synthetic_invalid_screen(screen_call)
        summary["server2_called"] = True

        brief_call, brief_result, brief_messages = build_evidence_brief(
            candidate,
            screen_for_hosted,
            client,
            structured=structured,
            generation_provider_id=generation_provider_id,
        )
        brief_run = _persist_model_run(
            session=session,
            stage=PaisStage.EVIDENCE_BRIEF.value,
            call=brief_call,
            article_id=article.id,
            candidate_relation_id=relation.id,
            evidence_record_id=None,
            input_payload={
                "candidate": candidate.model_dump(mode="json"),
                "screen": screen_for_hosted.model_dump(mode="json"),
                "messages": brief_messages,
            },
        )
        model_run_ids.append(brief_run.id)
        if brief_result is None:
            session.commit()
            return summary

        extraction_call, extraction_result, extraction_messages = extract_evidence_record(
            candidate,
            screen_for_hosted,
            brief_result,
            client,
            structured=structured,
            generation_provider_id=generation_provider_id,
        )
        extraction_run = _persist_model_run(
            session=session,
            stage=PaisStage.STRUCTURED_EXTRACTION.value,
            call=extraction_call,
            article_id=article.id,
            candidate_relation_id=relation.id,
            evidence_record_id=None,
            input_payload={
                "candidate": candidate.model_dump(mode="json"),
                "screen": screen_for_hosted.model_dump(mode="json"),
                "brief": brief_result.model_dump(mode="json"),
                "messages": extraction_messages,
            },
        )
        model_run_ids.append(extraction_run.id)
        if extraction_result is None:
            session.commit()
            return summary

        host_context = _create_host_context(session, article, extraction_result)
        evidence = _create_evidence_record(session, relation, host_context, brief_result, extraction_result)
        embedding = _create_pending_embedding(session, evidence)
        extraction_run.evidence_record_id = evidence.id
        brief_run.evidence_record_id = evidence.id
        if extraction_result.disagreement_with_screen:
            relation.hosted_disagreement_flag = True
            flags = set(_json_loads(relation.quality_flags_json, default=[]))
            flags.add("hosted_disagreement_with_screen")
            relation.quality_flags_json = _json_dumps(sorted(flags))
        session.flush()
        session.commit()

        summary.update(
            {
                "evidence_record_id": evidence.id,
                "embedding_record_id": embedding.id,
                "hosted_disagreement_flag": bool(relation.hosted_disagreement_flag),
                "quality_flags": _json_loads(relation.quality_flags_json, default=[]),
            }
        )
        return summary
    except Exception:
        session.rollback()
        raise


def screen_candidate(
    candidate: PaisCandidateInput,
    llm_client: OpenAICompatiblePaisClient,
    structured: Optional[bool] = None,
) -> tuple[PaisLLMResult, Optional[BenchmarkScreenResult], list[dict[str, str]]]:
    """Run the benchmark screen stage."""
    messages = build_benchmark_screen_messages(candidate)
    call = llm_client.complete_json(
        messages=messages,
        schema_model=BenchmarkScreenResult,
        stage=PaisStage.BENCHMARK_SCREEN.value,
        temperature=0.0,
        max_tokens=1024,
        structured=False,
    )
    result = BenchmarkScreenResult.model_validate(call.parsed_json) if call.valid and call.parsed_json else None
    return call, result, messages


def build_evidence_brief(
    candidate: PaisCandidateInput,
    screen_result: BenchmarkScreenResult,
    llm_client: OpenAICompatiblePaisClient,
    structured: Optional[bool] = None,
    generation_provider_id: Optional[str] = None,
) -> tuple[PaisLLMResult, Optional[PAISEvidenceBriefResult], list[dict[str, str]]]:
    """Run the hosted evidence brief stage."""
    messages = build_evidence_brief_messages(candidate, screen_result)
    kwargs = {"generation_provider_id": generation_provider_id} if generation_provider_id else {}
    call = llm_client.complete_json(
        messages=messages,
        schema_model=PAISEvidenceBriefResult,
        stage=PaisStage.EVIDENCE_BRIEF.value,
        temperature=0.0,
        max_tokens=1024,
        structured=structured,
        **kwargs,
    )
    result = PAISEvidenceBriefResult.model_validate(call.parsed_json) if call.valid and call.parsed_json else None
    return call, result, messages


def extract_evidence_record(
    candidate: PaisCandidateInput,
    screen_result: BenchmarkScreenResult,
    brief_result: PAISEvidenceBriefResult,
    llm_client: OpenAICompatiblePaisClient,
    structured: Optional[bool] = None,
    generation_provider_id: Optional[str] = None,
) -> tuple[PaisLLMResult, Optional[PAISEvidenceExtractionResult], list[dict[str, str]]]:
    """Run the hosted structured extraction stage."""
    messages = build_structured_extraction_messages(candidate, screen_result, brief_result)
    kwargs = {"generation_provider_id": generation_provider_id} if generation_provider_id else {}
    call = llm_client.complete_json(
        messages=messages,
        schema_model=PAISEvidenceExtractionResult,
        stage=PaisStage.STRUCTURED_EXTRACTION.value,
        temperature=0.0,
        max_tokens=3072,
        structured=structured,
        **kwargs,
    )
    result = (
        PAISEvidenceExtractionResult.model_validate(call.parsed_json) if call.valid and call.parsed_json else None
    )
    return call, result, messages


def render_embedding_text_from_brief(brief: PAISEvidenceBriefResult) -> str:
    """Return the LLM-generated evidence brief text in normalized form."""
    return " ".join(brief.embedding_text.split())


def render_embedding_text_from_extraction(extraction: PAISEvidenceExtractionResult) -> str:
    """Render deterministic embedding text from a structured extraction."""
    spans = "; ".join(span.text for span in extraction.source_spans[:3]) or "unknown"
    modalities = ", ".join(extraction.molecular_data.molecular_modalities) or "unknown"
    phenotypes = ", ".join(extraction.evidence.disease_phenotypes) or extraction.disease_or_phenotype.name
    pathogen_details = ", ".join(extraction.evidence.pathogen_details) or extraction.pathogen.name
    parts = [
        "PAIS evidence brief.",
        f"Article: {extraction.article.title}.",
        f"Candidate relation: {extraction.pathogen.name} -> {extraction.disease_or_phenotype.name}.",
        f"Host/model: {extraction.host_context.host_type.value}; "
        f"{extraction.host_context.cohort_or_model_description or 'unknown'}.",
        f"Timing: {extraction.relationship.timing_after_infection or 'unknown'}.",
        f"Evidence type: {extraction.evidence.evidence_type.value}.",
        f"Finding: {extraction.evidence.summary or extraction.relationship.statement or 'unknown'}.",
        f"PAIS category: {extraction.pais_classification.pais_category.value}.",
        f"Mechanism: {extraction.mechanism.mechanism_summary or 'unknown'}.",
        f"Molecular data: {extraction.molecular_data.molecular_data_summary or modalities}.",
        f"Disease phenotypes: {phenotypes}.",
        f"Pathogen details: {pathogen_details}.",
        f"Limitations/uncertainty: {'; '.join(extraction.limitations) or 'unknown'}.",
        f"Source support: {spans}.",
    ]
    return " ".join(part.strip() for part in parts if part.strip())


def embed_pending_records(
    database: DatabaseManager,
    llm_client: Optional[OpenAICompatiblePaisClient] = None,
    limit: int = 100,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Embed pending PAIS evidence texts and update EmbeddingRecord metadata."""
    client = llm_client or OpenAICompatiblePaisClient()
    database.create_tables()
    session = _session(database)
    records = (
        session.execute(
            select(EmbeddingRecord)
            .where(EmbeddingRecord.status == EmbeddingStatus.PENDING.value)
            .order_by(EmbeddingRecord.id)
            .limit(limit)
        )
        .scalars()
        .all()
    )
    embedded = 0
    failed = 0
    try:
        batch_size = max(1, batch_size)
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            texts = []
            text_records = []
            for record in batch:
                evidence = session.get(PAISEvidenceRecord, record.evidence_record_id)
                if evidence is None:
                    record.status = EmbeddingStatus.FAILED.value
                    failed += 1
                    continue
                texts.append(evidence.embedding_text)
                text_records.append((record, evidence))
            if not texts:
                continue
            try:
                vectors = client.embed_texts(texts)
                for (record, evidence), vector in zip(text_records, vectors):
                    record.embedding_model = client.config.pais_embedding_model
                    record.embedding_dim = len(vector)
                    record.vector_db = client.config.pais_embedding_base_url
                    record.vector_collection = "pais_evidence"
                    record.vector_id = f"pais-evidence-{evidence.id}-{record.text_sha256[:12]}"
                    record.status = EmbeddingStatus.EMBEDDED.value
                    embedded += 1
            except Exception:
                for record, _evidence in text_records:
                    record.status = EmbeddingStatus.FAILED.value
                    failed += 1
        session.commit()
    except Exception:
        session.rollback()
        raise
    return {"processed": len(records), "embedded": embedded, "failed": failed}


def _session(database: DatabaseManager) -> Session:
    if database._session is None:
        raise DatabaseError("Not connected to database")
    return database._session


def _upsert_article(session: Session, candidate: PaisCandidateInput) -> Article:
    article_data = candidate.article
    existing = None
    if article_data.pmid:
        existing = session.execute(select(Article).where(Article.pmid == article_data.pmid)).scalar_one_or_none()
    if existing is None and article_data.doi:
        existing = session.execute(select(Article).where(Article.doi == article_data.doi)).scalar_one_or_none()
    if existing is None:
        existing = (
            session.execute(
                select(Article).where(Article.title == article_data.title, Article.abstract == article_data.abstract)
            )
            .scalars()
            .first()
        )
    article = existing or Article(title=article_data.title, abstract=article_data.abstract)
    for field, value in article_data.model_dump(mode="json").items():
        if value is not None:
            setattr(article, field, value)
    session.add(article)
    session.flush()
    return article


def _upsert_pathogen(session: Session, candidate: PaisCandidateInput) -> Pathogen:
    data = candidate.pathogen
    normalized = _normalized_name(data.normalized_name or data.name)
    existing = None
    if data.ncbi_taxid:
        existing = session.execute(
            select(Pathogen).where(Pathogen.ncbi_taxid == data.ncbi_taxid)
        ).scalar_one_or_none()
    if existing is None:
        existing = session.execute(select(Pathogen).where(Pathogen.normalized_name == normalized)).scalars().first()
    pathogen = existing or Pathogen(name=data.name, normalized_name=normalized)
    pathogen.name = data.name
    pathogen.normalized_name = normalized
    pathogen.ncbi_taxid = data.ncbi_taxid
    pathogen.taxonomic_rank = data.taxonomic_rank
    pathogen.strain_or_variant = data.strain_or_variant
    pathogen.synonyms_json = _json_dumps(data.synonyms)
    session.add(pathogen)
    session.flush()
    return pathogen


def _upsert_disease(session: Session, candidate: PaisCandidateInput) -> DiseasePhenotype:
    data = candidate.disease
    normalized = _normalized_name(data.normalized_name or data.name)
    existing = None
    for column_name, value in (("doid", data.doid), ("hpo_id", data.hpo_id), ("mondo_id", data.mondo_id)):
        if value:
            column = getattr(DiseasePhenotype, column_name)
            existing = session.execute(select(DiseasePhenotype).where(column == value)).scalar_one_or_none()
            if existing is not None:
                break
    if existing is None:
        existing = (
            session.execute(select(DiseasePhenotype).where(DiseasePhenotype.normalized_name == normalized))
            .scalars()
            .first()
        )
    disease = existing or DiseasePhenotype(name=data.name, normalized_name=normalized)
    disease.name = data.name
    disease.normalized_name = normalized
    disease.doid = data.doid
    disease.hpo_id = data.hpo_id
    disease.mondo_id = data.mondo_id
    disease.synonyms_json = _json_dumps(data.synonyms)
    session.add(disease)
    session.flush()
    return disease


def _upsert_candidate_relation(
    session: Session, article: Article, pathogen: Pathogen, disease: DiseasePhenotype
) -> CandidateRelation:
    key = sha256_text(
        canonical_json({"article_id": article.id, "pathogen_id": pathogen.id, "disease_id": disease.id})
    )
    existing = session.execute(
        select(CandidateRelation).where(CandidateRelation.candidate_key == key)
    ).scalar_one_or_none()
    relation = existing or CandidateRelation(
        article_id=article.id,
        pathogen_id=pathogen.id,
        disease_id=disease.id,
        candidate_key=key,
    )
    session.add(relation)
    session.flush()
    return relation


def _persist_model_run(
    session: Session,
    stage: str,
    call: PaisLLMResult,
    article_id: Optional[int],
    candidate_relation_id: Optional[int],
    evidence_record_id: Optional[int],
    input_payload: dict[str, Any],
) -> ModelRun:
    prompt = prompt_for_stage(stage)
    schema = schema_for_stage(stage)
    run = ModelRun(
        stage=stage,
        article_id=article_id,
        candidate_relation_id=candidate_relation_id,
        evidence_record_id=evidence_record_id,
        backend=call.backend,
        model_id=call.model_id,
        model_version=call.model_version,
        endpoint_id=call.endpoint_id,
        structured_output_used=call.structured_output_used,
        prompt_name=prompt.name,
        prompt_version=prompt.version,
        prompt_sha256=prompt.sha256,
        schema_name=schema.__name__,
        schema_version="pydantic-v1",
        schema_sha256=schema_sha256(schema),
        input_sha256=sha256_text(canonical_json(input_payload)),
        raw_output=call.raw_output,
        parsed_json=_json_dumps(call.parsed_json),
        valid=call.valid,
        error_kind=call.error_kind,
        error_message=call.error_message,
        elapsed_s=call.elapsed_s,
    )
    session.add(run)
    session.flush()
    return run


def _apply_screen_result(
    relation: CandidateRelation,
    call: PaisLLMResult,
    screen_result: Optional[BenchmarkScreenResult],
) -> ScreenStatus:
    if screen_result is None:
        status = (
            ScreenStatus.ERROR if call.error_kind and call.error_kind != "validation_error" else ScreenStatus.INVALID
        )
        relation.screen_status = status.value
        relation.screen_confidence = Confidence.UNKNOWN.value
        flags = set(_json_loads(relation.quality_flags_json, default=[]))
        flags.add("invalid_benchmark_screen")
        relation.quality_flags_json = _json_dumps(sorted(flags))
        return status

    if screen_result.relationship == 1:
        status = ScreenStatus.POSITIVE
    elif screen_result.confidence == Confidence.LOW:
        status = ScreenStatus.UNCERTAIN
    else:
        status = ScreenStatus.NEGATIVE

    relation.benchmark_relationship = screen_result.relationship
    relation.benchmark_unrelated = screen_result.unrelated
    relation.screen_status = status.value
    relation.screen_confidence = screen_result.confidence.value
    relation.screen_exclusion_reason = (
        screen_result.exclusion_reason.value if screen_result.exclusion_reason else None
    )
    flags = set(_json_loads(relation.quality_flags_json, default=[]))
    flags.update(screen_result.quality_flags)
    relation.quality_flags_json = _json_dumps(sorted(flags))
    return status


def _merge_relation_quality_flags(relation: CandidateRelation, new_flags: list[str]) -> None:
    flags = set(_json_loads(relation.quality_flags_json, default=[]))
    flags.update(flag for flag in new_flags if flag)
    relation.quality_flags_json = _json_dumps(sorted(flags))


def _should_call_hosted(
    status: ScreenStatus,
    screen_result: Optional[BenchmarkScreenResult],
    allow_invalid_adjudication: bool,
) -> bool:
    if status in (ScreenStatus.POSITIVE, ScreenStatus.UNCERTAIN):
        return True
    if status in (ScreenStatus.INVALID, ScreenStatus.ERROR):
        return allow_invalid_adjudication
    if screen_result and screen_result.confidence == Confidence.LOW:
        return True
    return False


def _synthetic_invalid_screen(call: PaisLLMResult) -> BenchmarkScreenResult:
    message = call.error_message or "Benchmark screen did not return valid JSON."
    return BenchmarkScreenResult(
        relationship=0,
        unrelated=1,
        confidence=Confidence.UNKNOWN,
        decision_rationale_short=message[:500],
        evidence_span_quotes=[],
        exclusion_reason=None,
        quality_flags=["invalid_screen_adjudication"],
    )


def _create_host_context(
    session: Session,
    article: Article,
    extraction: PAISEvidenceExtractionResult,
) -> Optional[HostContext]:
    host = extraction.host_context
    has_host_signal = any(
        [
            host.host_name,
            host.host_taxid,
            host.species,
            host.tissue_or_sample,
            host.cohort_or_model_description,
            host.host_type != HostType.UNKNOWN,
        ]
    )
    if not has_host_signal:
        return None
    row = HostContext(
        article_id=article.id,
        host_name=host.host_name,
        host_taxid=host.host_taxid,
        host_type=host.host_type.value,
        species=host.species,
        tissue_or_sample=host.tissue_or_sample,
        cohort_or_model_description=host.cohort_or_model_description,
    )
    session.add(row)
    session.flush()
    return row


def _create_evidence_record(
    session: Session,
    relation: CandidateRelation,
    host_context: Optional[HostContext],
    brief: PAISEvidenceBriefResult,
    extraction: PAISEvidenceExtractionResult,
) -> PAISEvidenceRecord:
    embedding_text = render_embedding_text_from_brief(brief)
    record = PAISEvidenceRecord(
        candidate_relation_id=relation.id,
        host_context_id=host_context.id if host_context else None,
        pais_category=extraction.pais_classification.pais_category.value,
        relation_type=extraction.relationship.relation_type.value,
        evidence_type=extraction.evidence.evidence_type.value,
        timing_after_infection=extraction.relationship.timing_after_infection,
        mechanism_summary=extraction.mechanism.mechanism_summary,
        molecular_data_summary=extraction.molecular_data.molecular_data_summary,
        molecular_modalities_json=_json_dumps(extraction.molecular_data.molecular_modalities),
        disease_phenotypes_json=_json_dumps(extraction.evidence.disease_phenotypes),
        pathogen_details_json=_json_dumps(extraction.evidence.pathogen_details),
        source_evidence_spans_json=_json_dumps([span.model_dump(mode="json") for span in extraction.source_spans]),
        llm_summary=render_embedding_text_from_brief(brief),
        embedding_text=embedding_text,
        confidence=extraction.confidence.confidence.value,
        limitations="; ".join(extraction.limitations) if extraction.limitations else None,
    )
    session.add(record)
    session.flush()
    return record


def _create_pending_embedding(session: Session, evidence: PAISEvidenceRecord) -> EmbeddingRecord:
    record = EmbeddingRecord(
        evidence_record_id=evidence.id,
        text_sha256=sha256_text(evidence.embedding_text),
        status=EmbeddingStatus.PENDING.value,
    )
    session.add(record)
    session.flush()
    return record


def _normalized_name(name: str) -> str:
    return " ".join(name.casefold().split())


def _json_dumps(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value: Optional[str], default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default
