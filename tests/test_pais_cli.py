"""CLI tests for PAISDB commands."""

from __future__ import annotations

import sys
from unittest.mock import patch

from abstracts_explorer.cli import main
from tests.conftest import set_test_db


def test_pais_smoke_no_network(capsys):
    with patch.object(sys, "argv", ["abstracts-explorer", "pais", "smoke", "--no-network"]):
        assert main() == 0
    captured = capsys.readouterr()
    assert "mistralai/Mistral-Small-Instruct-2409" in captured.out
    assert "Qwen/Qwen3-Coder-30B-A3B-Instruct" in captured.out


def test_pais_init_db(tmp_path, capsys):
    set_test_db(tmp_path / "pais.db")
    with patch.object(sys, "argv", ["abstracts-explorer", "pais", "init-db"]):
        assert main() == 0
    captured = capsys.readouterr()
    assert "PAIS tables are ready" in captured.out
