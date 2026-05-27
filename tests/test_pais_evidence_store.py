"""Tests for PAIS evidence search/clustering helpers."""

from __future__ import annotations

import numpy as np

from abstracts_explorer.pais_evidence_store import (
    _chroma_where,
    _cluster_labels,
    _paper_from_metadata,
    _score_labels,
    get_pais_available_filters,
)


def test_paper_from_metadata_returns_authors_array():
    paper = _paper_from_metadata(
        {
            "paper_uid": "pais_article_1",
            "title": "Title",
            "authors": "Example Journal",
            "source": "paisdb2_benchmark_1000",
            "publication_year": 2022,
        },
        "Evidence text",
    )

    assert paper["authors"] == ["Example Journal"]
    assert paper["conference"] == "paisdb2_benchmark_1000"
    assert paper["source"] == "paisdb2_benchmark_1000"


def test_chroma_where_combines_source_and_year_with_and():
    where = _chroma_where(years=[2022], sources=["paisdb2_benchmark_1000"])

    assert where == {
        "$and": [
            {"year": {"$in": [2022]}},
            {"source": {"$in": ["paisdb2_benchmark_1000"]}},
        ]
    }


def test_cluster_labels_names_noise_as_outliers():
    labels = _cluster_labels(
        [
            {"pathogen_name": "A", "disease_name": "B", "pais_category": "true_pais"},
            {"pathogen_name": "C", "disease_name": "D", "pais_category": "unclear"},
        ],
        [0, -1],
    )

    assert labels["-1"] == "Outliers / mixed PAIS evidence"
    assert labels["0"] == "A / B / true_pais"


def test_score_labels_penalizes_too_few_clusters():
    matrix = np.asarray([[0.0, 0.0], [0.1, 0.1], [2.0, 2.0]])
    score, stats = _score_labels(matrix, matrix, [0, 0, 0], requested_clusters=2)

    assert score < 0
    assert stats["cluster_selection_reason"] == "too_few_clusters"


def test_pais_available_filters_default_to_all_years():
    class FakeDatabase:
        def query(self, sql):
            return [
                {"source": "paisdb2_benchmark_1000", "year": 2024},
                {"source": "paisdb2_benchmark_1000", "year": 2022},
            ]

    filters = get_pais_available_filters(FakeDatabase())

    assert filters["years"] == [2024, 2022]
    assert filters["default_year"] is None
    assert filters["allow_all_years"] is True
