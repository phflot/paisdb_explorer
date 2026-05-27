"""Web API tests for PAISDB endpoints."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from abstracts_explorer.web_ui.app import app as flask_app
from tests.conftest import set_test_db


def _pais_config(**overrides):
    values = {
        "pais_screen_model": "",
        "pais_screen_base_url": "",
        "pais_screen_auth_token": "",
        "pais_evidence_brief_model": "",
        "pais_evidence_brief_base_url": "",
        "pais_evidence_brief_auth_token": "",
        "pais_extraction_model": "",
        "pais_extraction_base_url": "",
        "pais_extraction_auth_token": "",
        "pais_embedding_model": "",
        "pais_embedding_base_url": "",
        "pais_embedding_auth_token": "",
        "pais_structured_output_mode": "json_schema",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_pais_run_candidate_endpoint_validates_and_returns_summary(tmp_path):
    set_test_db(tmp_path / "pais.db")
    payload = {
        "article": {"title": "Title", "abstract": "Abstract"},
        "pathogen": {"name": "Pathogen"},
        "disease": {"name": "Disease"},
    }
    expected = {
        "candidate_relation_id": 1,
        "screen_status": "negative",
        "server2_called": False,
        "model_run_ids": [1],
    }
    flask_app.config["TESTING"] = True
    with patch("abstracts_explorer.pais_pipeline.run_candidate_pipeline", return_value=expected):
        with flask_app.test_client() as client:
            response = client.post("/api/pais/run-candidate", json=payload)

    assert response.status_code == 200
    assert response.get_json() == expected


def test_pais_run_candidate_endpoint_rejects_missing_body():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        response = client.post("/api/pais/run-candidate")

    assert response.status_code == 400
    assert response.get_json()["error"] == "JSON body is required"


def test_pais_status_endpoint_returns_sanitized_configuration():
    flask_app.config["TESTING"] = True
    config = _pais_config(
        pais_screen_model="screen-model",
        pais_screen_base_url="https://user:secret@example.test/v1?api_key=secret-key&tenant=public",
        pais_evidence_brief_model="brief-model",
        pais_evidence_brief_base_url="https://brief.example.test/openai?access_token=secret-token",
        pais_extraction_model="extract-model",
        pais_extraction_base_url="https://extract.example.test/v1?region=eu",
        pais_embedding_model="embed-model",
        pais_embedding_base_url="https://embed.example.test/v1?authorization=Bearer%20secret",
        pais_structured_output_mode="json_schema",
    )

    with patch("abstracts_explorer.web_ui.app.get_config", return_value=config):
        with flask_app.test_client() as client:
            response = client.get("/api/pais/status")

    assert response.status_code == 200
    data = response.get_json()
    assert data["pais_screen_configured"] is True
    assert data["pais_evidence_brief_configured"] is True
    assert data["pais_extraction_configured"] is True
    assert data["pais_embedding_configured"] is True
    assert data["configured_models"] == {
        "screen": "screen-model",
        "evidence_brief": "brief-model",
        "extraction": "extract-model",
        "embedding": "embed-model",
    }
    assert data["configured_base_urls"]["screen"] == "https://example.test/v1?tenant=public"
    assert data["configured_base_urls"]["evidence_brief"] == "https://brief.example.test/openai"
    assert data["configured_base_urls"]["extraction"] == "https://extract.example.test/v1?region=eu"
    assert data["configured_base_urls"]["embedding"] == "https://embed.example.test/v1"
    assert data["pais_structured_output_mode"] == "json_schema"

    serialized = json.dumps(data)
    assert "secret" not in serialized
    assert "api_key" not in serialized
    assert "access_token" not in serialized
    assert "authorization" not in serialized


def test_pais_status_endpoint_marks_unconfigured_stages():
    flask_app.config["TESTING"] = True
    config = _pais_config(pais_screen_model="screen-only")

    with patch("abstracts_explorer.web_ui.app.get_config", return_value=config):
        with flask_app.test_client() as client:
            response = client.get("/api/pais/status")

    assert response.status_code == 200
    data = response.get_json()
    assert data["pais_screen_configured"] is False
    assert data["pais_evidence_brief_configured"] is False
    assert data["pais_extraction_configured"] is False
    assert data["pais_embedding_configured"] is False
