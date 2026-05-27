"""Web API tests for PAISDB endpoints."""

from __future__ import annotations

from unittest.mock import patch

from abstracts_explorer.web_ui.app import app as flask_app
from tests.conftest import set_test_db


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
