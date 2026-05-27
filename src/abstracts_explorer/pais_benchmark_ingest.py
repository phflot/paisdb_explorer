"""Import the PAIS benchmark rows as a first PAISDB Explorer data batch."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional

from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.pais_llm import OpenAICompatiblePaisClient
from abstracts_explorer.pais_pipeline import run_candidate_pipeline
from abstracts_explorer.pais_schemas import PaisCandidateInput

REQUIRED_BENCHMARK_COLUMNS = {
    "pmid",
    "pathogen_term",
    "disease_term",
    "title_process",
    "abstract_process",
    "Relationship",
}


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
        return int(value)
    except ValueError:
        return None


def _normalize_name(value: str) -> str:
    return " ".join(value.casefold().split())
