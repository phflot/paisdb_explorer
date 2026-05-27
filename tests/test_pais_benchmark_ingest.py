"""Tests for importing PAIS benchmark rows into PAISDB Explorer."""

from __future__ import annotations

import csv
import json

from sqlalchemy import select

from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.db_models import CandidateRelation, EmbeddingRecord, ModelRun, PAISEvidenceRecord
from abstracts_explorer.pais_benchmark_ingest import (
    candidate_from_benchmark_row,
    ingest_benchmark_dataset,
    ingest_benchmark_dataset_batched,
)
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


class FakeBatchScreenClient(FakeScreenClient):
    def __init__(self):
        self.batch_sizes = []

    def complete_json_batch(
        self,
        messages_batch,
        schema_model,
        stage,
        temperature=0.0,
        max_tokens=2048,
        structured=None,
    ):
        self.batch_sizes.append(len(messages_batch))
        return [
            self.complete_json(messages, schema_model, stage, temperature, max_tokens, structured)
            for messages in messages_batch
        ]


class FakeFallbackClient:
    def complete_json_batch(
        self,
        messages_batch,
        schema_model,
        stage,
        temperature=0.0,
        max_tokens=2048,
        structured=None,
    ):
        return [
            PaisLLMResult(
                raw_output='{"relationship": 1, "unrelated": 0}',
                parsed_json={"relationship": 1, "unrelated": 0},
                valid=True,
                elapsed_s=0.01,
                backend="fake",
                model_id="fake-screen",
                endpoint_id="fake",
                structured_output_used=False,
            )
            for _messages in messages_batch
        ]

    def complete_json(self, messages, schema_model, stage, temperature=0.0, max_tokens=2048, structured=None):
        if stage == "evidence_brief":
            parsed = {
                "embedding_text": (
                    "PAIS evidence brief. Candidate relation: Giardia lamblia -> chronic fatigue syndrome. "
                    "Host/model: human cohort. Finding: The abstract links Giardia to fatigue."
                ),
                "key_entities": {
                    "pathogen": "Giardia lamblia",
                    "disease_or_phenotype": "chronic fatigue syndrome",
                    "host": "human",
                    "organism_or_model": "human cohort",
                    "tissue_or_sample": "unknown",
                },
                "brief_quality_flags": [],
                "source_span_quotes": ["Example abstract"],
                "uncertainty_notes": None,
            }
            return PaisLLMResult(
                raw_output=json.dumps(parsed),
                parsed_json=parsed,
                valid=True,
                elapsed_s=0.01,
                backend="fake",
                model_id="fake-brief",
                endpoint_id="fake",
                structured_output_used=bool(structured),
            )
        return PaisLLMResult(
            raw_output="not json",
            parsed_json=None,
            valid=False,
            elapsed_s=0.01,
            backend="fake",
            model_id="fake-extraction",
            endpoint_id="fake",
            structured_output_used=bool(structured),
            error_kind="validation_error",
            error_message="invalid extraction",
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


def test_batched_ingest_screens_rows_in_one_batch_without_enrichment(tmp_path):
    set_test_db(tmp_path / "pais.db")
    csv_path = tmp_path / "benchmark.csv"
    first = _benchmark_row()
    second = {**_benchmark_row(), "pmid": "124", "title_process": "Second title"}
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerow(second)

    fake = FakeBatchScreenClient()
    with DatabaseManager() as database:
        summary = ingest_benchmark_dataset_batched(
            database=database,
            input_path=csv_path,
            llm_client=fake,
            structured=True,
            screen_batch_size=2,
            screen_only=True,
        )
        session = database._session
        assert session is not None
        runs = session.execute(select(ModelRun).order_by(ModelRun.id)).scalars().all()

    assert fake.batch_sizes == [2]
    assert summary["mode"] == "batched"
    assert summary["screened"] == 2
    assert summary["screen_status_counts"] == {"negative": 2}
    assert summary["evidence_records"] == 0
    assert [run.stage for run in runs] == ["benchmark_screen", "benchmark_screen"]


def test_batched_ingest_can_fallback_from_valid_brief(tmp_path):
    set_test_db(tmp_path / "pais.db")
    csv_path = tmp_path / "benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_benchmark_row()))
        writer.writeheader()
        writer.writerow(_benchmark_row())

    with DatabaseManager() as database:
        summary = ingest_benchmark_dataset_batched(
            database=database,
            input_path=csv_path,
            llm_client=FakeFallbackClient(),
            structured=True,
            screen_batch_size=1,
            hosted_concurrency=1,
            fallback_from_brief=True,
        )
        session = database._session
        assert session is not None
        evidence = session.execute(select(PAISEvidenceRecord)).scalar_one()
        embedding = session.execute(select(EmbeddingRecord)).scalar_one()
        fallback_run = (
            session.execute(select(ModelRun).where(ModelRun.backend == "deterministic_fallback")).scalars().one()
        )

    assert summary["fallback_records"] == 1
    assert summary["evidence_records"] == 1
    assert evidence.confidence == "low"
    assert embedding.status == "pending"
    assert fallback_run.valid is True


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
