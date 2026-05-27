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


def test_pais_ingest_benchmark_command(capsys):
    expected = {"processed": 1, "screen_status_counts": {"negative": 1}}
    with patch("abstracts_explorer.pais_cli.ingest_benchmark_dataset", return_value=expected) as ingest:
        with patch.object(
            sys,
            "argv",
            ["abstracts-explorer", "pais", "ingest-benchmark", "--input", "benchmark.csv", "--limit", "1"],
        ):
            assert main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == expected
    assert ingest.call_args.kwargs["input_path"] == "benchmark.csv"
    assert ingest.call_args.kwargs["limit"] == 1
