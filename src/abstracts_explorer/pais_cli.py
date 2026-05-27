"""CLI commands for PAISDB evidence-building workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import requests
from sqlalchemy import select

from abstracts_explorer.config import get_config
from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.db_models import EmbeddingRecord, PAISEvidenceRecord
from abstracts_explorer.pais_benchmark_ingest import (
    default_benchmark_input_path,
    ingest_benchmark_dataset,
    ingest_benchmark_dataset_batched,
)
from abstracts_explorer.pais_examples import EXAMPLES, get_example
from abstracts_explorer.pais_llm import _api_url
from abstracts_explorer.pais_pipeline import embed_pending_records, run_candidate_pipeline
from abstracts_explorer.pais_schemas import PaisCandidateInput


def add_pais_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register PAIS subcommands on the main CLI parser."""
    pais_parser = subparsers.add_parser(
        "pais",
        help="PAISDB candidate screening and evidence-building commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pais_subparsers = pais_parser.add_subparsers(dest="pais_command", help="PAIS sub-commands")

    pais_subparsers.add_parser("init-db", help="Create PAIS tables in the configured database")

    run_parser = pais_subparsers.add_parser(
        "run-candidate",
        help="Run one article-pathogen-disease candidate through the PAIS pipeline",
    )
    _add_candidate_args(run_parser)
    run_parser.add_argument("--output", type=str, default=None, help="Write summary JSON to this path")
    run_parser.add_argument("--no-structured", action="store_true", help="Disable provider-native JSON schema mode")

    example_parser = pais_subparsers.add_parser("run-example", help="Run a built-in PAIS example")
    example_parser.add_argument("name", choices=sorted(EXAMPLES), help="Example fixture name")
    example_parser.add_argument("--output", type=str, default=None, help="Write summary JSON to this path")
    example_parser.add_argument(
        "--no-structured",
        action="store_true",
        help="Disable provider-native JSON schema mode",
    )

    export_parser = pais_subparsers.add_parser("export-embedding-texts", help="Export PAIS embedding texts as JSONL")
    export_parser.add_argument("--output", type=str, default=None, help="Output JSONL path; defaults to stdout")
    export_parser.add_argument("--limit", type=int, default=None, help="Maximum number of records to export")

    embed_parser = pais_subparsers.add_parser("embed-pending", help="Embed pending PAIS evidence records")
    embed_parser.add_argument("--limit", type=int, default=100, help="Maximum pending records to process")
    embed_parser.add_argument("--batch-size", type=int, default=64, help="Embedding texts per API request")

    ingest_parser = pais_subparsers.add_parser(
        "ingest-benchmark",
        help="Ingest the local PAIS benchmark rows as PAISDB Explorer candidate/evidence records",
    )
    ingest_parser.add_argument(
        "--input",
        type=str,
        default=str(default_benchmark_input_path()),
        help="Benchmark CSV path",
    )
    ingest_parser.add_argument("--limit", type=int, default=None, help="Maximum rows to ingest")
    ingest_parser.add_argument("--output", type=str, default=None, help="Write ingestion summary JSON to this path")
    ingest_parser.add_argument(
        "--no-structured",
        action="store_true",
        help="Disable provider-native JSON schema mode for enrichment stages",
    )
    ingest_parser.add_argument(
        "--batched",
        action="store_true",
        help="Use staged batched execution: local Mistral tensor batches, concurrent hosted calls",
    )
    ingest_parser.add_argument(
        "--screen-only",
        action="store_true",
        help="Only run/persist the local benchmark screen stage",
    )
    ingest_parser.add_argument("--screen-batch-size", type=int, default=8, help="Local Mistral screen batch size")
    ingest_parser.add_argument(
        "--hosted-concurrency",
        type=int,
        default=8,
        help="Concurrent hosted evidence/extraction requests",
    )
    ingest_parser.add_argument(
        "--hosted-chunk-size",
        type=int,
        default=32,
        help="Hosted items to submit before committing results",
    )
    ingest_parser.add_argument(
        "--embed",
        action="store_true",
        help="Embed pending PAIS evidence texts after enrichment",
    )
    ingest_parser.add_argument(
        "--fallback-from-brief",
        action="store_true",
        help="Create flagged deterministic evidence records from valid briefs when structured extraction is invalid",
    )
    ingest_parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="Embedding texts per API request when --embed is set",
    )
    ingest_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip rows that already have persisted screen/evidence records",
    )

    smoke_parser = pais_subparsers.add_parser("smoke", help="Inspect configured PAIS model endpoints")
    smoke_parser.add_argument(
        "--no-network",
        action="store_true",
        help="Only print resolved config; do not call endpoints",
    )


def pais_command(args: argparse.Namespace) -> int:
    """Dispatch PAIS CLI subcommands."""
    if not getattr(args, "pais_command", None):
        print("Missing PAIS sub-command", file=sys.stderr)
        return 1

    if args.pais_command == "init-db":
        return _init_db_command()
    if args.pais_command == "run-candidate":
        candidate = _candidate_from_args(args)
        return _run_candidate_command(candidate, output=args.output, structured=not args.no_structured)
    if args.pais_command == "run-example":
        return _run_candidate_command(get_example(args.name), output=args.output, structured=not args.no_structured)
    if args.pais_command == "export-embedding-texts":
        return _export_embedding_texts_command(output=args.output, limit=args.limit)
    if args.pais_command == "embed-pending":
        return _embed_pending_command(limit=args.limit, batch_size=args.batch_size)
    if args.pais_command == "ingest-benchmark":
        return _ingest_benchmark_command(
            input_path=args.input,
            limit=args.limit,
            output=args.output,
            structured=not args.no_structured,
            batched=args.batched,
            screen_only=args.screen_only,
            screen_batch_size=args.screen_batch_size,
            hosted_concurrency=args.hosted_concurrency,
            hosted_chunk_size=args.hosted_chunk_size,
            embed=args.embed,
            fallback_from_brief=args.fallback_from_brief,
            embedding_batch_size=args.embedding_batch_size,
            resume=not args.no_resume,
        )
    if args.pais_command == "smoke":
        return _smoke_command(no_network=args.no_network)

    print(f"Unknown PAIS sub-command: {args.pais_command}", file=sys.stderr)
    return 1


def _add_candidate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-json", type=str, default=None, help="Path to candidate JSON")
    parser.add_argument("--title", type=str, default=None, help="Article title")
    parser.add_argument("--abstract", type=str, default=None, help="Article abstract")
    parser.add_argument("--pmid", type=str, default=None, help="PubMed ID")
    parser.add_argument("--doi", type=str, default=None, help="DOI")
    parser.add_argument("--pathogen", type=str, default=None, help="Pathogen candidate name")
    parser.add_argument("--disease", type=str, default=None, help="Disease/phenotype candidate name")


def _candidate_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.input_json:
        return json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    missing = [name for name in ("title", "abstract", "pathogen", "disease") if not getattr(args, name, None)]
    if missing:
        raise SystemExit(f"Missing required candidate fields: {', '.join(missing)}")
    return {
        "article": {
            "pmid": args.pmid,
            "doi": args.doi,
            "title": args.title,
            "abstract": args.abstract,
            "source": "cli",
        },
        "pathogen": {"name": args.pathogen},
        "disease": {"name": args.disease},
    }


def _run_candidate_command(candidate_data: dict[str, Any], output: Optional[str], structured: bool) -> int:
    candidate = PaisCandidateInput.model_validate(candidate_data)
    with DatabaseManager() as database:
        summary = run_candidate_pipeline(candidate, database=database, structured=structured)
    _write_json(summary, output)
    return 0


def _init_db_command() -> int:
    with DatabaseManager() as database:
        database.create_tables()
    print("PAIS tables are ready.")
    return 0


def _export_embedding_texts_command(output: Optional[str], limit: Optional[int]) -> int:
    with DatabaseManager() as database:
        database.create_tables()
        if database._session is None:
            raise RuntimeError("Database session is not connected")
        stmt = select(PAISEvidenceRecord).order_by(PAISEvidenceRecord.id)
        if limit is not None:
            stmt = stmt.limit(limit)
        rows = database._session.execute(stmt).scalars().all()
        lines = [
            json.dumps(
                {
                    "evidence_record_id": row.id,
                    "candidate_relation_id": row.candidate_relation_id,
                    "text_sha256": _text_hash_for_record(database, row),
                    "embedding_text": row.embedding_text,
                },
                ensure_ascii=False,
            )
            for row in rows
        ]
    text = "\n".join(lines) + ("\n" if lines else "")
    if output:
        Path(output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


def _embed_pending_command(limit: int, batch_size: int) -> int:
    with DatabaseManager() as database:
        summary = embed_pending_records(database=database, limit=limit, batch_size=batch_size)
    _write_json(summary, None)
    return 0


def _ingest_benchmark_command(
    input_path: str,
    limit: Optional[int],
    output: Optional[str],
    structured: bool,
    batched: bool,
    screen_only: bool,
    screen_batch_size: int,
    hosted_concurrency: int,
    hosted_chunk_size: int,
    embed: bool,
    fallback_from_brief: bool,
    embedding_batch_size: int,
    resume: bool,
) -> int:
    with DatabaseManager() as database:
        if batched or screen_only:
            summary = ingest_benchmark_dataset_batched(
                database=database,
                input_path=input_path,
                limit=limit,
                structured=structured,
                screen_batch_size=screen_batch_size,
                hosted_concurrency=hosted_concurrency,
                hosted_chunk_size=hosted_chunk_size,
                screen_only=screen_only,
                resume=resume,
                embed=embed,
                fallback_from_brief=fallback_from_brief,
                embedding_batch_size=embedding_batch_size,
            )
        else:
            summary = ingest_benchmark_dataset(
                database=database,
                input_path=input_path,
                limit=limit,
                structured=structured,
            )
    _write_json(summary, output)
    return 0


def _smoke_command(no_network: bool) -> int:
    config = get_config()
    screen_configured = bool(config.pais_screen_model)
    if config.pais_screen_backend == "openai_compatible":
        screen_configured = bool(config.pais_screen_model and config.pais_screen_base_url)
    report: dict[str, Any] = {
        "screen": {
            "backend": config.pais_screen_backend,
            "base_url": config.pais_screen_base_url,
            "configured": screen_configured,
            "hf_home": config.pais_screen_hf_home,
            "local_files_only": config.pais_screen_local_files_only,
            "model": config.pais_screen_model,
            "revision": config.pais_screen_revision,
        },
        "evidence_brief": {
            "base_url": config.pais_evidence_brief_base_url,
            "configured": bool(config.pais_evidence_brief_model and config.pais_evidence_brief_base_url),
            "model": config.pais_evidence_brief_model,
        },
        "structured_extraction": {
            "base_url": config.pais_extraction_base_url,
            "configured": bool(config.pais_extraction_model and config.pais_extraction_base_url),
            "model": config.pais_extraction_model,
        },
        "embeddings": {
            "base_url": config.pais_embedding_base_url,
            "configured": bool(config.pais_embedding_model and config.pais_embedding_base_url),
            "model": config.pais_embedding_model,
        },
    }
    if not no_network:
        screen_check = (
            {"ok": True, "skipped": True, "reason": "local hf_transformers backend"}
            if config.pais_screen_backend == "hf_transformers"
            else _check_models(
                config.pais_screen_base_url,
                config.pais_screen_auth_token or config.llm_backend_auth_token,
            )
        )
        report["endpoint_checks"] = {
            "screen_models": screen_check,
            "evidence_models": _check_models(
                config.pais_evidence_brief_base_url,
                config.pais_evidence_brief_auth_token or config.llm_backend_auth_token,
            ),
            "embedding_models": _check_models(
                config.pais_embedding_base_url,
                config.pais_embedding_auth_token or config.llm_backend_auth_token,
            ),
        }
    _write_json(report, None)
    return 0


def _check_models(base_url: str, auth_token: str) -> dict[str, Any]:
    if not base_url:
        return {"ok": False, "skipped": True, "error": "Base URL is not configured"}
    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
    try:
        response = requests.get(_api_url(base_url, "models"), headers=headers, timeout=10)
        return {"ok": response.ok, "status_code": response.status_code}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _text_hash_for_record(database: DatabaseManager, row: PAISEvidenceRecord) -> Optional[str]:
    if database._session is None:
        return None
    record = (
        database._session.execute(select(EmbeddingRecord).where(EmbeddingRecord.evidence_record_id == row.id))
        .scalars()
        .first()
    )
    return record.text_sha256 if record else None


def _write_json(data: dict[str, Any], output: Optional[str]) -> None:
    text = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if output:
        Path(output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
