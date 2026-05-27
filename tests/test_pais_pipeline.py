"""Tests for PAISDB screening and evidence pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import select

from abstracts_explorer.database import DatabaseManager
from abstracts_explorer.db_models import CandidateRelation, EmbeddingRecord, ModelRun, PAISEvidenceRecord
from abstracts_explorer.pais_llm import PaisLLMResult
from abstracts_explorer.pais_pipeline import render_embedding_text_from_extraction, run_candidate_pipeline
from abstracts_explorer.pais_prompts import build_benchmark_screen_messages
from abstracts_explorer.pais_schemas import BenchmarkScreenResult, PAISEvidenceExtractionResult, PaisCandidateInput
from tests.conftest import set_test_db


class FakePaisClient:
    """Fake stage client returning predeclared PAIS LLM results."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []
        self.structured_by_stage = {}

    def complete_json(self, messages, schema_model, stage, temperature=0.0, max_tokens=2048, structured=None):
        self.calls.append(stage)
        self.structured_by_stage[stage] = structured
        response = self.responses[stage]
        if isinstance(response, PaisLLMResult):
            return response
        return PaisLLMResult(
            raw_output=json.dumps(response),
            parsed_json=response,
            valid=True,
            elapsed_s=0.01,
            backend="fake",
            model_id=f"fake-{stage}",
            endpoint_id="fake-endpoint",
            structured_output_used=bool(structured),
        )


def test_benchmark_screen_result_requires_inverse_labels():
    with pytest.raises(ValueError):
        BenchmarkScreenResult(relationship=1, unrelated=1)
    with pytest.raises(ValueError):
        BenchmarkScreenResult(relationship=0, unrelated=0)


def test_benchmark_screen_prompt_matches_paisdb2_paper_zero_shot_prompt():
    prompt_file = Path(__file__).resolve().parents[2] / "paisdb2" / "src" / "paisdb2" / "prompts.py"
    namespace = {}
    exec(prompt_file.read_text(encoding="utf-8"), namespace)
    candidate = _candidate()
    expected = namespace["PAPER_ZERO_SHOT_V1"].format(
        pathogen=candidate["pathogen"]["name"],
        disease=candidate["disease"]["name"],
        title=candidate["article"]["title"],
        abstract=candidate["article"]["abstract"],
    )

    assert build_benchmark_screen_messages(PaisCandidateInput.model_validate(candidate)) == [
        {"role": "user", "content": expected}
    ]


def test_high_confidence_negative_stops_after_screen(tmp_path):
    set_test_db(tmp_path / "pais.db")
    fake = FakePaisClient(
        {
            "benchmark_screen": {
                "relationship": 0,
                "unrelated": 1,
                "confidence": "high",
                "decision_rationale_short": "Only a co-mention.",
                "evidence_span_quotes": [],
                "exclusion_reason": "co_mention_only",
                "quality_flags": [],
            }
        }
    )

    with DatabaseManager() as db:
        summary = run_candidate_pipeline(_candidate(), db, llm_client=fake, structured=False)
        session = db._session
        assert session is not None
        assert session.execute(select(ModelRun)).scalars().all()[0].valid is True
        assert session.execute(select(PAISEvidenceRecord)).scalars().all() == []
        assert session.execute(select(EmbeddingRecord)).scalars().all() == []

    assert summary["screen_status"] == "negative"
    assert summary["server2_called"] is False
    assert fake.calls == ["benchmark_screen"]
    assert fake.structured_by_stage["benchmark_screen"] is False


def test_positive_candidate_creates_evidence_and_pending_embedding(tmp_path):
    set_test_db(tmp_path / "pais.db")
    fake = FakePaisClient(
        {
            "benchmark_screen": _positive_screen(),
            "evidence_brief": _brief(),
            "structured_extraction": _extraction(),
        }
    )

    with DatabaseManager() as db:
        summary = run_candidate_pipeline(_candidate(), db, llm_client=fake, structured=False)
        session = db._session
        assert session is not None
        relation = session.execute(select(CandidateRelation)).scalar_one()
        runs = session.execute(select(ModelRun).order_by(ModelRun.id)).scalars().all()
        evidence = session.execute(select(PAISEvidenceRecord)).scalar_one()
        embedding = session.execute(select(EmbeddingRecord)).scalar_one()

        assert relation.benchmark_relationship == 1
        assert relation.benchmark_unrelated == 0
        assert relation.hosted_disagreement_flag is False
        assert [run.stage for run in runs] == ["benchmark_screen", "evidence_brief", "structured_extraction"]
        assert evidence.pais_category == "true_pais"
        assert evidence.embedding_text == " ".join(_brief()["embedding_text"].split())
        assert embedding.status == "pending"

    assert summary["server2_called"] is True
    assert summary["evidence_record_id"] is not None
    assert summary["embedding_record_id"] is not None
    assert fake.calls == ["benchmark_screen", "evidence_brief", "structured_extraction"]


def test_invalid_screen_is_persisted_without_hosted_calls(tmp_path):
    set_test_db(tmp_path / "pais.db")
    fake = FakePaisClient(
        {
            "benchmark_screen": PaisLLMResult(
                raw_output="not json",
                parsed_json=None,
                valid=False,
                elapsed_s=0.01,
                backend="fake",
                model_id="fake-screen",
                endpoint_id="fake",
                structured_output_used=False,
                error_kind="validation_error",
                error_message="invalid json",
            )
        }
    )

    with DatabaseManager() as db:
        summary = run_candidate_pipeline(_candidate(), db, llm_client=fake, structured=False)
        session = db._session
        assert session is not None
        run = session.execute(select(ModelRun)).scalar_one()
        relation = session.execute(select(CandidateRelation)).scalar_one()
        assert run.valid is False
        assert relation.screen_status == "invalid"

    assert summary["screen_status"] == "invalid"
    assert summary["server2_called"] is False
    assert fake.calls == ["benchmark_screen"]


def test_hosted_disagreement_sets_candidate_quality_flag(tmp_path):
    set_test_db(tmp_path / "pais.db")
    extraction = _extraction()
    extraction["disagreement_with_screen"] = True
    extraction["quality_flags"] = ["hosted_output_disagrees"]
    fake = FakePaisClient(
        {
            "benchmark_screen": _positive_screen(),
            "evidence_brief": _brief(),
            "structured_extraction": extraction,
        }
    )

    with DatabaseManager() as db:
        summary = run_candidate_pipeline(_candidate(), db, llm_client=fake, structured=False)
        session = db._session
        assert session is not None
        relation = session.execute(select(CandidateRelation)).scalar_one()
        assert relation.hosted_disagreement_flag is True
        assert "hosted_disagreement_with_screen" in json.loads(relation.quality_flags_json)

    assert summary["hosted_disagreement_flag"] is True


def test_deterministic_embedding_renderer_is_stable():
    extraction = PAISEvidenceExtractionResult.model_validate(_extraction())
    first = render_embedding_text_from_extraction(extraction)
    second = render_embedding_text_from_extraction(extraction)
    assert first == second
    assert "Giardia lamblia -> chronic fatigue syndrome" in first


def _candidate():
    return {
        "article": {
            "title": "Chronic fatigue syndrome after Giardia enteritis",
            "abstract": "A cohort after Giardia enteritis reported chronic fatigue after infection.",
            "source": "test",
        },
        "pathogen": {"name": "Giardia lamblia"},
        "disease": {"name": "chronic fatigue syndrome"},
    }


def _positive_screen():
    return {
        "relationship": 1,
        "unrelated": 0,
        "confidence": "high",
        "decision_rationale_short": "The abstract links Giardia enteritis to chronic fatigue.",
        "evidence_span_quotes": ["after Giardia enteritis reported chronic fatigue"],
        "exclusion_reason": None,
        "quality_flags": [],
    }


def _brief():
    return {
        "embedding_text": (
            "PAIS evidence brief. Article: Chronic fatigue syndrome after Giardia enteritis. "
            "Candidate relation: Giardia lamblia -> chronic fatigue syndrome. "
            "Benchmark screen: positive; high. Host/model: human; cohort. Timing: after infection."
        ),
        "key_entities": {
            "pathogen": "Giardia lamblia",
            "disease_or_phenotype": "chronic fatigue syndrome",
            "host": "human",
            "organism_or_model": "human cohort",
            "tissue_or_sample": "unknown",
        },
        "brief_quality_flags": [],
        "source_span_quotes": ["after Giardia enteritis reported chronic fatigue"],
        "uncertainty_notes": None,
    }


def _extraction():
    return {
        "article": {"title": "Chronic fatigue syndrome after Giardia enteritis"},
        "pathogen": {"name": "Giardia lamblia", "normalized_name": "giardia lamblia"},
        "disease_or_phenotype": {
            "name": "chronic fatigue syndrome",
            "normalized_name": "chronic fatigue syndrome",
        },
        "host_context": {
            "host_name": "Homo sapiens",
            "host_type": "human",
            "species": "human",
            "cohort_or_model_description": "human cohort",
        },
        "relationship": {
            "relation_type": "associated_with",
            "timing_after_infection": "after infection",
            "statement": "Giardia enteritis was followed by chronic fatigue.",
        },
        "pais_classification": {"pais_category": "true_pais", "rationale": "Post-infectious symptoms."},
        "evidence": {
            "evidence_type": "clinical_cohort",
            "disease_phenotypes": ["chronic fatigue syndrome"],
            "pathogen_details": ["Giardia enteritis"],
            "summary": "The source reports chronic fatigue after Giardia enteritis.",
        },
        "mechanism": {"mechanism_summary": None},
        "molecular_data": {"molecular_data_summary": None, "molecular_modalities": []},
        "source_spans": [{"text": "after Giardia enteritis reported chronic fatigue", "source": "abstract"}],
        "confidence": {"confidence": "high", "rationale": "Explicit source statement."},
        "limitations": [],
        "disagreement_with_screen": False,
        "quality_flags": [],
    }
