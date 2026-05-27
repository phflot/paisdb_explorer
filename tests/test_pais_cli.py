"""CLI tests for PAISDB commands."""

from __future__ import annotations

import sys
import json
from unittest.mock import patch

from abstracts_explorer.cli import main
from tests.conftest import set_test_db


def test_pais_smoke_no_network(capsys, monkeypatch):
    for key in (
        "PAIS_SCREEN_MODEL",
        "PAIS_SCREEN_BASE_URL",
        "PAIS_EVIDENCE_BRIEF_MODEL",
        "PAIS_EVIDENCE_BRIEF_BASE_URL",
        "PAIS_EXTRACTION_MODEL",
        "PAIS_EXTRACTION_BASE_URL",
        "PAIS_EMBEDDING_MODEL",
        "PAIS_EMBEDDING_BASE_URL",
    ):
        monkeypatch.setenv(key, "")
    with patch.object(sys, "argv", ["abstracts-explorer", "pais", "smoke", "--no-network"]):
        assert main() == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["screen"]["configured"] is False
    assert report["evidence_brief"]["configured"] is False
    assert report["structured_extraction"]["configured"] is False
    assert report["embeddings"]["configured"] is False


def test_pais_init_db(tmp_path, capsys):
    set_test_db(tmp_path / "pais.db")
    with patch.object(sys, "argv", ["abstracts-explorer", "pais", "init-db"]):
        assert main() == 0
    captured = capsys.readouterr()
    assert "PAIS tables are ready" in captured.out
