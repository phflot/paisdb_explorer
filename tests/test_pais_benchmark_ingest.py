"""Tests for importing PAIS benchmark rows into PAISDB Explorer."""

from __future__ import annotations

import csv
import json

from sqlalchemy import select

from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.db_models import CandidateRelation
from abstracts_explorer.pais_benchmark_ingest import candidate_from_benchmark_row, ingest_benchmark_dataset
from abstracts_explorer.pais_llm import PaisLLMResult
from tests.conftest import set_test_db


class FakeScreenClient:
    def complete_json(self, messages, schema_model, stage, temperature=0.0, max_tokens=2048, structured=None):
        return PaisLLMResult(
            raw_output='{"relationship": 0, "unrelated": 1}',
            parsed_json={"relationship": 0, "unrelated": 1},
            valid=True,
            elapsed_s=0.01,
            backend="fake",
            model_id="fake-screen",
            endpoint_id="fake",
            structured_output_used=bool(structured),
        )


def test_candidate_from_benchmark_row_maps_source_columns():
    row = _benchmark_row()
    candidate = candidate_from_benchmark_row(row)

    assert candidate.article.pmid == "123"
    assert candidate.article.title == "Example title"
    assert candidate.article.abstract == "Example abstract"
    assert candidate.article.source == "paisdb2_benchmark_1000"
    assert candidate.pathogen.name == "Giardia lamblia"
    assert candidate.disease.name == "chronic fatigue syndrome"


def test_ingest_benchmark_dataset_stores_gold_as_quality_metadata(tmp_path):
    set_test_db(tmp_path / "pais.db")
    csv_path = tmp_path / "benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_benchmark_row()))
        writer.writeheader()
        writer.writerow(_benchmark_row())

    with DatabaseManager() as database:
        summary = ingest_benchmark_dataset(
            database=database,
            input_path=csv_path,
            llm_client=FakeScreenClient(),
            structured=True,
        )
        session = database._session
        assert session is not None
        relation = session.execute(select(CandidateRelation)).scalar_one()
        flags = json.loads(relation.quality_flags_json)

    assert summary["processed"] == 1
    assert summary["screen_status_counts"] == {"negative": 1}
    assert summary["gold_agreement"] == {
        "available": 1,
        "matches": 0,
        "mismatches": 1,
        "missing_prediction": 0,
    }
    assert "source:paisdb2_benchmark_1000" in flags
    assert "benchmark_gold_relationship:1" in flags


def _benchmark_row():
    return {
        "pmid": "123",
        "pathogen_term": "Giardia lamblia",
        "disease_term": "chronic fatigue syndrome",
        "title_process": "Example title",
        "abstract_process": "Example abstract",
        "Relationship": "yes",
        "doi": "",
        "journal": "Example Journal",
        "publication_date_pubmed": "2025",
        "publication_type": "Journal Article",
        "publication_year": "2025",
        "query": "Giardia AND fatigue",
    }
