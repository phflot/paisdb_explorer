"""Import the PAIS benchmark rows as a first PAISDB Explorer data batch."""

from __future__ import annotations

import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import select

from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.db_models import Article, CandidateRelation, ModelRun, PAISEvidenceRecord
from abstracts_explorer.pais_llm import OpenAICompatiblePaisClient, PaisLLMResult
from abstracts_explorer.pais_pipeline import (
    _apply_screen_result,
    _create_evidence_record,
    _create_host_context,
    _create_pending_embedding,
    _json_loads,
    _merge_relation_quality_flags,
    _persist_model_run,
    _session,
    _should_call_hosted,
    _synthetic_invalid_screen,
    _upsert_article,
    _upsert_candidate_relation,
    _upsert_disease,
    _upsert_pathogen,
    build_evidence_brief,
    embed_pending_records,
    extract_evidence_record,
    run_candidate_pipeline,
)
from abstracts_explorer.pais_prompts import build_benchmark_screen_messages
from abstracts_explorer.pais_schemas import (
    BenchmarkScreenResult,
    PAISEvidenceExtractionResult,
    PAISEvidenceBriefResult,
    PaisCandidateInput,
    PaisStage,
    ScreenStatus,
)

REQUIRED_BENCHMARK_COLUMNS = {
    "pmid",
    "pathogen_term",
    "disease_term",
    "title_process",
    "abstract_process",
    "Relationship",
}


@dataclass(frozen=True)
class BenchmarkItem:
    """Prepared benchmark row with persisted candidate identifiers."""

    row_number: int
    row: dict[str, str]
    candidate: PaisCandidateInput
    gold: Optional[int]
    article_id: int
    relation_id: int
    screen_messages: list[dict[str, str]]


@dataclass(frozen=True)
class BriefWorkItem:
    """Evidence-brief work item ready for extraction."""

    item: BenchmarkItem
    screen_result: BenchmarkScreenResult
    brief_result: PAISEvidenceBriefResult
    brief_run_id: int


def default_benchmark_input_path() -> Path:
    """Return the local paisdb2 benchmark input path when this monorepo layout is present."""
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root.parent / "paisdb_local" / "relation_extraction" / "src" / "final_test_dataset_12_03_2025.csv"


def ingest_benchmark_dataset(
    database: DatabaseManager,
    input_path: Path | str | None = None,
    llm_client: Optional[OpenAICompatiblePaisClient] = None,
    limit: Optional[int] = None,
    structured: Optional[bool] = None,
) -> dict[str, Any]:
    """Run benchmark CSV rows through the production PAISDB Explorer pipeline."""
    path = Path(input_path) if input_path else default_benchmark_input_path()
    rows = list(iter_benchmark_rows(path, limit=limit))
    summary: dict[str, Any] = {
        "input_path": str(path),
        "rows_seen": len(rows),
        "processed": 0,
        "failed": 0,
        "screen_status_counts": {},
        "evidence_records": 0,
        "embedding_records": 0,
        "gold_agreement": {"available": 0, "matches": 0, "mismatches": 0, "missing_prediction": 0},
        "errors": [],
    }
    for row_number, row in rows:
        try:
            candidate = candidate_from_benchmark_row(row)
            gold = gold_relationship_value(row.get("Relationship"))
            run_summary = run_candidate_pipeline(
                candidate,
                database=database,
                llm_client=llm_client,
                structured=structured,
                initial_quality_flags=benchmark_quality_flags(row_number, row, gold),
            )
            summary["processed"] += 1
            status = run_summary.get("screen_status") or "unknown"
            summary["screen_status_counts"][status] = summary["screen_status_counts"].get(status, 0) + 1
            if run_summary.get("evidence_record_id") is not None:
                summary["evidence_records"] += 1
            if run_summary.get("embedding_record_id") is not None:
                summary["embedding_records"] += 1
            _update_gold_agreement(summary["gold_agreement"], gold, run_summary.get("benchmark_relationship"))
        except Exception as exc:
            summary["failed"] += 1
            if len(summary["errors"]) < 10:
                summary["errors"].append({"row_number": row_number, "error": str(exc)})
    return summary


def ingest_benchmark_dataset_batched(
    database: DatabaseManager,
    input_path: Path | str | None = None,
    llm_client: Optional[OpenAICompatiblePaisClient] = None,
    limit: Optional[int] = None,
    structured: Optional[bool] = None,
    screen_batch_size: int = 8,
    hosted_concurrency: int = 8,
    hosted_chunk_size: int = 32,
    screen_only: bool = False,
    resume: bool = True,
    embed: bool = False,
    embedding_batch_size: int = 64,
    fallback_from_brief: bool = False,
) -> dict[str, Any]:
    """Run benchmark CSV rows through a stage-batched PAISDB pipeline."""
    path = Path(input_path) if input_path else default_benchmark_input_path()
    rows = list(iter_benchmark_rows(path, limit=limit))
    client = llm_client or OpenAICompatiblePaisClient()
    database.create_tables()
    started = time.monotonic()
    summary: dict[str, Any] = {
        "mode": "batched",
        "input_path": str(path),
        "rows_seen": len(rows),
        "processed": 0,
        "failed": 0,
        "screened": 0,
        "screen_skipped_existing": 0,
        "screen_batch_size": screen_batch_size,
        "hosted_concurrency": hosted_concurrency,
        "hosted_chunk_size": hosted_chunk_size,
        "embedding_batch_size": embedding_batch_size,
        "screen_only": screen_only,
        "embed": embed,
        "fallback_from_brief": fallback_from_brief,
        "screen_status_counts": {},
        "enrichment_candidates": 0,
        "brief_runs": 0,
        "extraction_runs": 0,
        "fallback_records": 0,
        "evidence_records": 0,
        "embedding_records": 0,
        "gold_agreement": {"available": 0, "matches": 0, "mismatches": 0, "missing_prediction": 0},
        "errors": [],
        "timings_s": {},
    }

    items = _prepare_items(database, rows, summary)
    summary["processed"] = len(items)

    screen_started = time.monotonic()
    _run_screen_batches(
        database=database,
        client=client,
        items=items,
        summary=summary,
        structured=structured,
        batch_size=screen_batch_size,
        resume=resume,
    )
    summary["timings_s"]["screen"] = round(time.monotonic() - screen_started, 3)

    if not screen_only:
        enrich_started = time.monotonic()
        _run_hosted_enrichment(
            database=database,
            client=client,
            items=items,
            summary=summary,
            structured=structured,
            hosted_concurrency=hosted_concurrency,
            hosted_chunk_size=hosted_chunk_size,
            resume=resume,
            fallback_from_brief=fallback_from_brief,
        )
        summary["timings_s"]["hosted_enrichment"] = round(time.monotonic() - enrich_started, 3)

    if embed and not screen_only:
        embed_started = time.monotonic()
        summary["embedding_summary"] = embed_pending_records(
            database=database,
            llm_client=client,
            limit=max(len(items), 1),
            batch_size=embedding_batch_size,
        )
        summary["timings_s"]["embedding"] = round(time.monotonic() - embed_started, 3)

    _finalize_batched_summary(database, items, summary)
    summary["timings_s"]["total"] = round(time.monotonic() - started, 3)
    if items:
        summary["average_s_per_row"] = round(summary["timings_s"]["total"] / len(items), 3)
    return summary


def iter_benchmark_rows(path: Path, limit: Optional[int] = None) -> list[tuple[int, dict[str, str]]]:
    """Read benchmark rows and validate the expected PAISDB columns."""
    if not path.exists():
        raise FileNotFoundError(f"Benchmark input not found: {path}")
    if limit is not None and limit <= 0:
        return []
    rows: list[tuple[int, dict[str, str]]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_BENCHMARK_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Benchmark input is missing required columns: {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, start=2):
            rows.append((row_number, row))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def candidate_from_benchmark_row(row: dict[str, str]) -> PaisCandidateInput:
    """Convert one benchmark CSV row into the normal production candidate input."""
    title = _first_value(row, "title_process", "title")
    abstract = _first_value(row, "abstract_process", "abstract")
    pathogen = _first_value(row, "pathogen_term", "pathogen")
    disease = _first_value(row, "disease_term", "disease")
    article = {
        "pmid": _optional_value(row.get("pmid")),
        "doi": _optional_value(row.get("doi")),
        "title": title,
        "abstract": abstract,
        "journal": _optional_value(row.get("journal")),
        "publication_year": _optional_int(row.get("publication_year")),
        "publication_date": _optional_value(row.get("publication_date_pubmed")),
        "publication_type": _optional_value(row.get("publication_type")),
        "source": "paisdb2_benchmark_1000",
        "source_url": _optional_value(row.get("query")),
    }
    return PaisCandidateInput.model_validate(
        {
            "article": article,
            "pathogen": {"name": pathogen, "normalized_name": _normalize_name(pathogen)},
            "disease": {"name": disease, "normalized_name": _normalize_name(disease)},
        }
    )


def benchmark_quality_flags(row_number: int, row: dict[str, str], gold: Optional[int]) -> list[str]:
    """Encode benchmark reference metadata as non-prompt quality/provenance flags."""
    flags = ["source:paisdb2_benchmark_1000", f"benchmark_csv_row:{row_number}"]
    if row.get("pmid"):
        flags.append(f"benchmark_pmid:{row['pmid']}")
    if gold is not None:
        flags.append(f"benchmark_gold_relationship:{gold}")
    return flags


def gold_relationship_value(value: Optional[str]) -> Optional[int]:
    """Normalize the benchmark gold label to 0/1 when present."""
    if value is None:
        return None
    normalized = value.strip().casefold()
    if normalized in {"yes", "1", "true", "relationship"}:
        return 1
    if normalized in {"no", "0", "false", "unrelated"}:
        return 0
    return None


def _update_gold_agreement(stats: dict[str, int], gold: Optional[int], prediction: Any) -> None:
    if gold is None:
        return
    stats["available"] += 1
    if prediction is None:
        stats["missing_prediction"] += 1
        return
    predicted = int(prediction)
    if predicted == gold:
        stats["matches"] += 1
    else:
        stats["mismatches"] += 1


def _prepare_items(
    database: DatabaseManager,
    rows: list[tuple[int, dict[str, str]]],
    summary: dict[str, Any],
) -> list[BenchmarkItem]:
    session = _session(database)
    items = []
    try:
        for row_number, row in rows:
            try:
                candidate = candidate_from_benchmark_row(row)
                gold = gold_relationship_value(row.get("Relationship"))
                article = _upsert_article(session, candidate)
                pathogen = _upsert_pathogen(session, candidate)
                disease = _upsert_disease(session, candidate)
                relation = _upsert_candidate_relation(session, article, pathogen, disease)
                _merge_relation_quality_flags(relation, benchmark_quality_flags(row_number, row, gold))
                items.append(
                    BenchmarkItem(
                        row_number=row_number,
                        row=row,
                        candidate=candidate,
                        gold=gold,
                        article_id=article.id,
                        relation_id=relation.id,
                        screen_messages=build_benchmark_screen_messages(candidate),
                    )
                )
            except Exception as exc:
                summary["failed"] += 1
                _append_error(summary, row_number, exc)
        session.commit()
    except Exception:
        session.rollback()
        raise
    return items


def _run_screen_batches(
    database: DatabaseManager,
    client: OpenAICompatiblePaisClient,
    items: list[BenchmarkItem],
    summary: dict[str, Any],
    structured: Optional[bool],
    batch_size: int,
    resume: bool,
) -> None:
    session = _session(database)
    pending = []
    for item in items:
        relation = session.get(CandidateRelation, item.relation_id)
        if relation is None:
            continue
        if resume and _relation_has_screen(relation):
            summary["screen_skipped_existing"] += 1
            continue
        pending.append(item)

    pending.sort(key=lambda item: sum(len(message["content"]) for message in item.screen_messages))
    batch_size = max(1, batch_size)
    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        try:
            calls = _complete_json_batch(
                client=client,
                messages_batch=[item.screen_messages for item in chunk],
                schema_model=BenchmarkScreenResult,
                stage=PaisStage.BENCHMARK_SCREEN.value,
                max_tokens=1024,
                structured=False,
            )
        except Exception as exc:
            calls = [
                PaisLLMResult(
                    raw_output="",
                    parsed_json=None,
                    valid=False,
                    elapsed_s=0.0,
                    backend="unknown",
                    model_id="unknown",
                    endpoint_id=None,
                    structured_output_used=False,
                    error_kind=exc.__class__.__name__,
                    error_message=str(exc),
                )
                for _item in chunk
            ]
        try:
            for item, call in zip(chunk, calls):
                relation = session.get(CandidateRelation, item.relation_id)
                if relation is None:
                    continue
                screen_result = (
                    BenchmarkScreenResult.model_validate(call.parsed_json)
                    if call.valid and call.parsed_json
                    else None
                )
                _persist_model_run(
                    session=session,
                    stage=PaisStage.BENCHMARK_SCREEN.value,
                    call=call,
                    article_id=item.article_id,
                    candidate_relation_id=item.relation_id,
                    evidence_record_id=None,
                    input_payload={
                        "candidate": item.candidate.model_dump(mode="json"),
                        "messages": item.screen_messages,
                    },
                )
                _apply_screen_result(relation, call, screen_result)
                summary["screened"] += 1
            session.commit()
        except Exception:
            session.rollback()
            raise


def _run_hosted_enrichment(
    database: DatabaseManager,
    client: OpenAICompatiblePaisClient,
    items: list[BenchmarkItem],
    summary: dict[str, Any],
    structured: Optional[bool],
    hosted_concurrency: int,
    hosted_chunk_size: int,
    resume: bool,
    fallback_from_brief: bool,
) -> None:
    session = _session(database)
    allow_invalid_adjudication = bool(
        getattr(getattr(client, "config", None), "pais_allow_adjudication_on_invalid_screen", False)
    )
    enrichment_items: list[tuple[BenchmarkItem, BenchmarkScreenResult]] = []
    for item in items:
        relation = session.get(CandidateRelation, item.relation_id)
        if relation is None:
            continue
        if resume and _relation_has_evidence(session, item.relation_id):
            continue
        call = _latest_screen_call(session, item.relation_id)
        screen_result = _screen_result_from_relation(relation, call)
        status = ScreenStatus(relation.screen_status or ScreenStatus.INVALID.value)
        if _should_call_hosted(status, screen_result, allow_invalid_adjudication):
            enrichment_items.append((item, screen_result or _synthetic_invalid_screen(call)))
    summary["enrichment_candidates"] = len(enrichment_items)
    if not enrichment_items:
        return

    max_workers = max(1, hosted_concurrency)
    chunk_size = max(1, hosted_chunk_size)
    brief_items: list[BriefWorkItem] = []
    brief_jobs: list[tuple[BenchmarkItem, BenchmarkScreenResult]] = []
    for item, screen_result in enrichment_items:
        if resume:
            existing = _latest_valid_brief(session, item.relation_id)
            if existing is not None:
                brief_items.append(
                    BriefWorkItem(
                        item=item,
                        screen_result=screen_result,
                        brief_result=existing[0],
                        brief_run_id=existing[1],
                    )
                )
                summary["brief_reused_existing"] = summary.get("brief_reused_existing", 0) + 1
                continue
        brief_jobs.append((item, screen_result))

    for chunk in _chunks(brief_jobs, chunk_size):
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(build_evidence_brief, item.candidate, screen_result, client, structured): (
                    item,
                    screen_result,
                )
                for item, screen_result in chunk
            }
            for future in as_completed(futures):
                item, screen_result = futures[future]
                try:
                    brief_call, brief_result, brief_messages = future.result()
                    brief_run = _persist_model_run(
                        session=session,
                        stage=PaisStage.EVIDENCE_BRIEF.value,
                        call=brief_call,
                        article_id=item.article_id,
                        candidate_relation_id=item.relation_id,
                        evidence_record_id=None,
                        input_payload={
                            "candidate": item.candidate.model_dump(mode="json"),
                            "screen": screen_result.model_dump(mode="json"),
                            "messages": brief_messages,
                        },
                    )
                    summary["brief_runs"] += 1
                    if brief_result is not None:
                        brief_items.append(
                            BriefWorkItem(
                                item=item,
                                screen_result=screen_result,
                                brief_result=brief_result,
                                brief_run_id=brief_run.id,
                            )
                        )
                except Exception as exc:
                    summary["failed"] += 1
                    _append_error(summary, item.row_number, exc)
            session.commit()

    if not brief_items:
        return

    for chunk in _chunks(brief_items, chunk_size):
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    extract_evidence_record,
                    brief_item.item.candidate,
                    brief_item.screen_result,
                    brief_item.brief_result,
                    client,
                    structured,
                ): brief_item
                for brief_item in chunk
                if not (resume and _relation_has_evidence(session, brief_item.item.relation_id))
            }
            for future in as_completed(futures):
                brief_item = futures[future]
                item = brief_item.item
                try:
                    extraction_call, extraction_result, extraction_messages = future.result()
                    extraction_run = _persist_model_run(
                        session=session,
                        stage=PaisStage.STRUCTURED_EXTRACTION.value,
                        call=extraction_call,
                        article_id=item.article_id,
                        candidate_relation_id=item.relation_id,
                        evidence_record_id=None,
                        input_payload={
                            "candidate": item.candidate.model_dump(mode="json"),
                            "screen": brief_item.screen_result.model_dump(mode="json"),
                            "brief": brief_item.brief_result.model_dump(mode="json"),
                            "messages": extraction_messages,
                        },
                    )
                    summary["extraction_runs"] += 1
                    if extraction_result is None:
                        if fallback_from_brief:
                            fallback_result = _fallback_extraction_from_brief(
                                brief_item.item.candidate,
                                brief_item.screen_result,
                                brief_item.brief_result,
                            )
                            fallback_run = _persist_model_run(
                                session=session,
                                stage=PaisStage.STRUCTURED_EXTRACTION.value,
                                call=_fallback_call(fallback_result),
                                article_id=item.article_id,
                                candidate_relation_id=item.relation_id,
                                evidence_record_id=None,
                                input_payload={
                                    "candidate": item.candidate.model_dump(mode="json"),
                                    "screen": brief_item.screen_result.model_dump(mode="json"),
                                    "brief": brief_item.brief_result.model_dump(mode="json"),
                                    "fallback_reason": extraction_call.error_message
                                    or extraction_call.error_kind
                                    or "invalid_structured_extraction",
                                },
                            )
                            article = session.get(Article, item.article_id)
                            relation = session.get(CandidateRelation, item.relation_id)
                            if article is not None and relation is not None:
                                host_context = _create_host_context(session, article, fallback_result)
                                evidence = _create_evidence_record(
                                    session,
                                    relation,
                                    host_context,
                                    brief_item.brief_result,
                                    fallback_result,
                                )
                                embedding = _create_pending_embedding(session, evidence)
                                fallback_run.evidence_record_id = evidence.id
                                brief_run = session.get(ModelRun, brief_item.brief_run_id)
                                if brief_run is not None:
                                    brief_run.evidence_record_id = evidence.id
                                summary["fallback_records"] += 1
                                summary["evidence_records"] += 1
                                summary["embedding_records"] += 1 if embedding.id is not None else 0
                        continue

                    article = session.get(Article, item.article_id)
                    relation = session.get(CandidateRelation, item.relation_id)
                    if article is None or relation is None:
                        continue
                    host_context = _create_host_context(session, article, extraction_result)
                    evidence = _create_evidence_record(
                        session,
                        relation,
                        host_context,
                        brief_item.brief_result,
                        extraction_result,
                    )
                    embedding = _create_pending_embedding(session, evidence)
                    extraction_run.evidence_record_id = evidence.id
                    brief_run = session.get(ModelRun, brief_item.brief_run_id)
                    if brief_run is not None:
                        brief_run.evidence_record_id = evidence.id
                    if extraction_result.disagreement_with_screen:
                        relation.hosted_disagreement_flag = True
                        flags = set(_json_loads(relation.quality_flags_json, default=[]))
                        flags.add("hosted_disagreement_with_screen")
                        relation.quality_flags_json = json_dumps_for_flags(flags)
                    summary["evidence_records"] += 1
                    summary["embedding_records"] += 1 if embedding.id is not None else 0
                except Exception as exc:
                    summary["failed"] += 1
                    _append_error(summary, item.row_number, exc)
            session.commit()


def _complete_json_batch(
    client: OpenAICompatiblePaisClient,
    messages_batch: list[list[dict[str, str]]],
    schema_model: type[BenchmarkScreenResult],
    stage: str,
    max_tokens: int,
    structured: bool,
) -> list[PaisLLMResult]:
    if hasattr(client, "complete_json_batch"):
        return client.complete_json_batch(
            messages_batch=messages_batch,
            schema_model=schema_model,
            stage=stage,
            temperature=0.0,
            max_tokens=max_tokens,
            structured=structured,
        )
    return [
        client.complete_json(
            messages=messages,
            schema_model=schema_model,
            stage=stage,
            temperature=0.0,
            max_tokens=max_tokens,
            structured=structured,
        )
        for messages in messages_batch
    ]


def _relation_has_screen(relation: CandidateRelation) -> bool:
    return relation.screen_status is not None and (
        relation.benchmark_relationship is not None or relation.screen_status in {"invalid", "error"}
    )


def _relation_has_evidence(session, relation_id: int) -> bool:
    return (
        session.execute(
            select(PAISEvidenceRecord.id).where(PAISEvidenceRecord.candidate_relation_id == relation_id).limit(1)
        ).first()
        is not None
    )


def _latest_screen_call(session, relation_id: int) -> PaisLLMResult:
    run = (
        session.execute(
            select(ModelRun)
            .where(
                ModelRun.candidate_relation_id == relation_id,
                ModelRun.stage == PaisStage.BENCHMARK_SCREEN.value,
            )
            .order_by(ModelRun.id.desc())
            .limit(1)
        )
        .scalars()
        .first()
    )
    if run is None:
        return PaisLLMResult(
            raw_output="",
            parsed_json=None,
            valid=False,
            elapsed_s=0.0,
            backend="unknown",
            model_id="unknown",
            endpoint_id=None,
            structured_output_used=False,
            error_kind="missing_screen_run",
            error_message="No persisted benchmark_screen ModelRun found.",
        )
    return PaisLLMResult(
        raw_output=run.raw_output,
        parsed_json=_json_loads(run.parsed_json, default=None),
        valid=bool(run.valid),
        elapsed_s=run.elapsed_s or 0.0,
        backend=run.backend or "unknown",
        model_id=run.model_id or "unknown",
        endpoint_id=run.endpoint_id,
        structured_output_used=bool(run.structured_output_used),
        error_kind=run.error_kind,
        error_message=run.error_message,
        model_version=run.model_version,
    )


def _latest_valid_brief(session, relation_id: int) -> Optional[tuple[PAISEvidenceBriefResult, int]]:
    run = (
        session.execute(
            select(ModelRun)
            .where(
                ModelRun.candidate_relation_id == relation_id,
                ModelRun.stage == PaisStage.EVIDENCE_BRIEF.value,
                ModelRun.valid == True,  # noqa: E712
            )
            .order_by(ModelRun.id.desc())
            .limit(1)
        )
        .scalars()
        .first()
    )
    if run is None:
        return None
    parsed = _json_loads(run.parsed_json, default=None)
    if not isinstance(parsed, dict):
        return None
    try:
        return PAISEvidenceBriefResult.model_validate(parsed), run.id
    except Exception:
        return None


def _screen_result_from_relation(
    relation: CandidateRelation,
    call: PaisLLMResult,
) -> Optional[BenchmarkScreenResult]:
    if relation.benchmark_relationship is None or relation.benchmark_unrelated is None:
        return None
    try:
        flags = _json_loads(relation.quality_flags_json, default=[])
        return BenchmarkScreenResult(
            relationship=relation.benchmark_relationship,
            unrelated=relation.benchmark_unrelated,
            confidence=relation.screen_confidence or "unknown",
            decision_rationale_short="Persisted benchmark screen result.",
            evidence_span_quotes=[],
            exclusion_reason=relation.screen_exclusion_reason,
            quality_flags=[flag for flag in flags if isinstance(flag, str)],
        )
    except Exception:
        return _synthetic_invalid_screen(call)


def _finalize_batched_summary(
    database: DatabaseManager,
    items: list[BenchmarkItem],
    summary: dict[str, Any],
) -> None:
    session = _session(database)
    summary["screen_status_counts"] = {}
    summary["gold_agreement"] = {"available": 0, "matches": 0, "mismatches": 0, "missing_prediction": 0}
    for item in items:
        relation = session.get(CandidateRelation, item.relation_id)
        if relation is None:
            continue
        status = relation.screen_status or "unknown"
        summary["screen_status_counts"][status] = summary["screen_status_counts"].get(status, 0) + 1
        _update_gold_agreement(summary["gold_agreement"], item.gold, relation.benchmark_relationship)


def _append_error(summary: dict[str, Any], row_number: int, exc: Exception) -> None:
    if len(summary["errors"]) < 10:
        summary["errors"].append({"row_number": row_number, "error": str(exc)})


def _fallback_extraction_from_brief(
    candidate: PaisCandidateInput,
    screen_result: BenchmarkScreenResult,
    brief: PAISEvidenceBriefResult,
) -> PAISEvidenceExtractionResult:
    key_entities = brief.key_entities
    host = _none_if_unknown(key_entities.host)
    organism = _none_if_unknown(key_entities.organism_or_model)
    tissue = _none_if_unknown(key_entities.tissue_or_sample)
    quotes = [_clip_text(quote, 180) for quote in brief.source_span_quotes[:2] if quote.strip()]
    if not quotes:
        quotes = [_clip_text(candidate.article.title, 180)]
    summary = _clip_text(brief.embedding_text, 700)
    return PAISEvidenceExtractionResult.model_validate(
        {
            "article": {
                "pmid": candidate.article.pmid,
                "doi": candidate.article.doi,
                "title": candidate.article.title,
                "publication_year": candidate.article.publication_year,
            },
            "pathogen": {
                "name": candidate.pathogen.name,
                "normalized_name": candidate.pathogen.normalized_name or _normalize_name(candidate.pathogen.name),
                "identifiers": {"ncbi_taxid": candidate.pathogen.ncbi_taxid} if candidate.pathogen.ncbi_taxid else {},
            },
            "disease_or_phenotype": {
                "name": candidate.disease.name,
                "normalized_name": candidate.disease.normalized_name or _normalize_name(candidate.disease.name),
                "identifiers": {
                    key: value
                    for key, value in {
                        "doid": candidate.disease.doid,
                        "hpo_id": candidate.disease.hpo_id,
                        "mondo_id": candidate.disease.mondo_id,
                    }.items()
                    if value
                },
            },
            "host_context": {
                "host_name": host,
                "host_taxid": None,
                "host_type": _fallback_host_type(host or organism),
                "species": organism,
                "tissue_or_sample": tissue,
                "cohort_or_model_description": organism,
            },
            "relationship": {
                "relation_type": "associated_with" if screen_result.relationship == 1 else "unclear",
                "timing_after_infection": None,
                "statement": _clip_text(summary, 360),
            },
            "pais_classification": {
                "pais_category": "unclear",
                "rationale": "Deterministic fallback from a validated evidence brief after invalid structured extraction.",
            },
            "evidence": {
                "evidence_type": "unclear",
                "disease_phenotypes": [candidate.disease.name],
                "pathogen_details": [candidate.pathogen.name],
                "summary": summary,
            },
            "mechanism": {"mechanism_summary": None},
            "molecular_data": {"molecular_data_summary": None, "molecular_modalities": []},
            "source_spans": [{"text": quote, "source": "abstract"} for quote in quotes],
            "confidence": {
                "confidence": "low",
                "rationale": "Structured extraction fallback preserves a validated brief but has reduced confidence.",
            },
            "limitations": [
                "structured_extraction_invalid",
                "fallback_from_valid_evidence_brief",
            ],
            "disagreement_with_screen": False,
            "quality_flags": [
                "fallback_from_valid_evidence_brief",
                "structured_extraction_invalid",
            ],
        }
    )


def _fallback_call(extraction: PAISEvidenceExtractionResult) -> PaisLLMResult:
    payload = extraction.model_dump(mode="json")
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return PaisLLMResult(
        raw_output=raw,
        parsed_json=payload,
        valid=True,
        elapsed_s=0.0,
        backend="deterministic_fallback",
        model_id="brief_to_extraction_fallback",
        endpoint_id=None,
        structured_output_used=False,
        error_kind=None,
        error_message=None,
    )


def _none_if_unknown(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped.casefold() in {"unknown", "not stated", "not specified", "none"}:
        return None
    return stripped


def _fallback_host_type(value: Optional[str]) -> str:
    text = (value or "").casefold()
    if any(term in text for term in ("human", "patient", "children", "adult", "cohort")):
        return "human"
    if any(term in text for term in ("mouse", "mice", "rat", "swine", "pig", "animal")):
        return "animal_model"
    if any(term in text for term in ("cell", "vero", "hela")):
        return "cell_line"
    if "in vitro" in text:
        return "in_vitro"
    return "unknown"


def _clip_text(value: str, limit: int) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def json_dumps_for_flags(flags: set[str]) -> str:
    return json.dumps(sorted(flags), ensure_ascii=False, sort_keys=True)


def _first_value(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = _optional_value(row.get(key))
        if value:
            return value
    raise ValueError(f"Missing required value in columns: {', '.join(keys)}")


def _optional_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _optional_int(value: Optional[str]) -> Optional[int]:
    value = _optional_value(value)
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None

    if not parsed.is_integer():
        return None

    integer = int(parsed)
    if 1900 <= integer <= 2100:
        return integer

    # Some benchmark exports encoded years as YYYY0.0, e.g. 20220.0.
    if integer % 10 == 0:
        normalized = integer // 10
        if 1900 <= normalized <= 2100:
            return normalized

    return None


def _normalize_name(value: str) -> str:
    return " ".join(value.casefold().split())
