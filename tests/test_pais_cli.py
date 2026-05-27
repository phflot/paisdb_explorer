"""CLI tests for PAISDB commands."""

from __future__ import annotations

import sys
import json
from unittest.mock import patch

from abstracts_explorer.cli import main
from abstracts_explorer.config import get_config
from tests.conftest import set_test_db


def test_pais_smoke_no_network(capsys, monkeypatch):
    for key in (
        "PAIS_SCREEN_MODEL",
        "PAIS_SCREEN_BACKEND",
        "PAIS_SCREEN_BASE_URL",
        "PAIS_SCREEN_REVISION",
        "PAIS_SCREEN_HF_HOME",
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


def test_pais_smoke_hf_screen_does_not_require_base_url(capsys, monkeypatch):
    monkeypatch.setenv("PAIS_SCREEN_BACKEND", "hf_transformers")
    monkeypatch.setenv("PAIS_SCREEN_MODEL", "mistralai/Mistral-Small-Instruct-2409")
    monkeypatch.setenv("PAIS_SCREEN_BASE_URL", "")
    monkeypatch.setenv("PAIS_SCREEN_REVISION", "test-revision")
    monkeypatch.setenv("PAIS_SCREEN_HF_HOME", "/tmp/hf-cache")
    get_config(reload=True)

    with patch.object(sys, "argv", ["abstracts-explorer", "pais", "smoke", "--no-network"]):
        assert main() == 0

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["screen"]["backend"] == "hf_transformers"
    assert report["screen"]["configured"] is True
    assert report["screen"]["base_url"] == ""
    assert report["screen"]["revision"] == "test-revision"


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


def test_pais_ingest_benchmark_batched_command(capsys):
    expected = {"mode": "batched", "processed": 2}
    with patch("abstracts_explorer.pais_cli.ingest_benchmark_dataset_batched", return_value=expected) as ingest:
        with patch.object(
            sys,
            "argv",
            [
                "abstracts-explorer",
                "pais",
                "ingest-benchmark",
                "--input",
                "benchmark.csv",
                "--limit",
                "2",
                "--batched",
                "--screen-only",
                "--screen-batch-size",
                "4",
                "--hosted-concurrency",
                "3",
            ],
        ):
            assert main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == expected
    assert ingest.call_args.kwargs["input_path"] == "benchmark.csv"
    assert ingest.call_args.kwargs["limit"] == 2
    assert ingest.call_args.kwargs["screen_batch_size"] == 4
    assert ingest.call_args.kwargs["hosted_concurrency"] == 3
    assert ingest.call_args.kwargs["screen_only"] is True
