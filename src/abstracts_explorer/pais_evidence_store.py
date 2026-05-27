"""PAIS evidence vector search and clustering helpers."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

try:  # scikit-learn >=1.3
    from sklearn.cluster import HDBSCAN
except ImportError:  # pragma: no cover - compatibility for older sklearn installs
    HDBSCAN = None  # type: ignore[assignment]

from abstracts_explorer.embeddings import EmbeddingsManager


PAIS_CLUSTER_CONFERENCE = "PAISDB"
PAIS_CLUSTER_REDUCTION = "umap"
PAIS_CLUSTER_METHOD = "umap_auto"
PAIS_CLUSTER_VERSION = 2
PAIS_METADATA_VERSION = 2

_UMAP_N_NEIGHBORS = (15, 30, 50, 75)
_UMAP_MIN_DIST = (0.0, 0.05, 0.1)


def fetch_pais_evidence_records(
    database,
    years: Optional[list[int]] = None,
    sources: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Fetch positive/uncertain PAIS evidence records used for RAG and clustering."""
    params: list[Any] = []
    filters = ["er.embedding_text is not null", "trim(er.embedding_text) != ''"]

    if years:
        clean_years = [int(year) for year in years]
        placeholders = ",".join("?" for _ in clean_years)
        filters.append(f"a.publication_year in ({placeholders})")
        params.extend(clean_years)

    if sources:
        clean_sources = [str(source) for source in sources if str(source).strip()]
        if clean_sources:
            placeholders = ",".join("?" for _ in clean_sources)
            filters.append(f"a.source in ({placeholders})")
            params.extend(clean_sources)

    where_clause = " and ".join(filters)
    rows = database.query(
        f"""
        select
            er.id as evidence_id,
            er.candidate_relation_id,
            er.pais_category,
            er.relation_type,
            er.evidence_type,
            er.timing_after_infection,
            er.mechanism_summary,
            er.molecular_data_summary,
            er.llm_summary,
            er.embedding_text,
            er.confidence as evidence_confidence,
            er.limitations,
            cr.article_id,
            cr.screen_status,
            cr.screen_confidence,
            cr.benchmark_relationship,
            cr.benchmark_unrelated,
            a.pmid,
            a.doi,
            a.title,
            a.abstract,
            a.journal,
            a.publication_year,
            a.source,
            a.source_url,
            p.name as pathogen_name,
            d.name as disease_name,
            papers.uid as paper_uid
        from pais_evidence_records er
        join pais_candidate_relations cr on cr.id = er.candidate_relation_id
        join pais_articles a on a.id = cr.article_id
        join pais_pathogens p on p.id = cr.pathogen_id
        join pais_disease_phenotypes d on d.id = cr.disease_id
        left join papers on papers.original_id = a.pmid
        where {where_clause}
        order by er.id
        """,
        tuple(params),
    )
    return [_normalize_record(row) for row in rows]


def get_pais_available_filters(database) -> dict[str, Any]:
    """Return PAIS-native source/year filters for the web UI."""
    rows = database.query(
        """
        select distinct
            coalesce(nullif(a.source, ''), 'PAISDB') as source,
            a.publication_year as year
        from pais_evidence_records er
        join pais_candidate_relations cr on cr.id = er.candidate_relation_id
        join pais_articles a on a.id = cr.article_id
        where er.embedding_text is not null
          and trim(er.embedding_text) != ''
        order by source, year desc
        """
    )

    source_years: dict[str, list[int]] = {}
    sources: set[str] = set()
    years: set[int] = set()
    for row in rows:
        source = str(row.get("source") or "PAISDB")
        sources.add(source)
        year = row.get("year")
        if year is None:
            source_years.setdefault(source, [])
            continue
        year_int = int(year)
        years.add(year_int)
        source_years.setdefault(source, []).append(year_int)

    sorted_sources = sorted(sources)
    source_years = {
        source: sorted(set(source_years.get(source, [])), reverse=True) for source in sorted_sources
    }
    default_source = sorted_sources[0] if sorted_sources else ""
    default_years = source_years.get(default_source, [])
    return {
        "sources": sorted_sources,
        "years": sorted(years, reverse=True),
        "source_years": source_years,
        "default_source": default_source,
        # PAISDB should open on the full evidence set; a latest-year default
        # can shrink clustering to only a handful of records.
        "default_year": None,
        "allow_all_years": bool(default_years),
    }


def get_pais_stats(
    database,
    years: Optional[list[int]] = None,
    sources: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Return PAIS evidence stats for the top-line UI counters."""
    params: list[Any] = []
    filters = ["er.embedding_text is not null", "trim(er.embedding_text) != ''"]
    if years:
        clean_years = [int(year) for year in years]
        placeholders = ",".join("?" for _ in clean_years)
        filters.append(f"a.publication_year in ({placeholders})")
        params.extend(clean_years)
    if sources:
        clean_sources = [str(source) for source in sources if str(source).strip()]
        if clean_sources:
            placeholders = ",".join("?" for _ in clean_sources)
            filters.append(f"a.source in ({placeholders})")
            params.extend(clean_sources)

    rows = database.query(
        f"""
        select
            count(distinct a.id) as total_articles,
            count(distinct er.id) as total_evidence_records,
            count(distinct cr.id) as total_candidate_relations
        from pais_evidence_records er
        join pais_candidate_relations cr on cr.id = er.candidate_relation_id
        join pais_articles a on a.id = cr.article_id
        where {" and ".join(filters)}
        """,
        tuple(params),
    )
    row = rows[0] if rows else {}
    total_evidence = int(row.get("total_evidence_records") or 0)
    return {
        "total_papers": total_evidence,
        "total_articles": int(row.get("total_articles") or 0),
        "total_evidence_records": total_evidence,
        "total_candidate_relations": int(row.get("total_candidate_relations") or 0),
        "conference": sources[0] if sources and len(sources) == 1 else None,
        "year": years[0] if years and len(years) == 1 else None,
    }


def ensure_pais_evidence_collection(
    database,
    embeddings_manager: EmbeddingsManager,
    batch_size: int = 32,
    years: Optional[list[int]] = None,
    sources: Optional[list[str]] = None,
) -> int:
    """Ensure the Chroma collection contains PAIS evidence vectors."""
    records = fetch_pais_evidence_records(database, years=years, sources=sources)
    if not records:
        return 0

    ids = [_evidence_vector_id(record) for record in records]
    existing_by_id = _existing_metadata_by_id(embeddings_manager, ids)
    to_add: list[dict[str, Any]] = []
    stale_ids: list[str] = []
    for record in records:
        vector_id = _evidence_vector_id(record)
        expected_sha = _text_sha(record["embedding_text"])
        existing = existing_by_id.get(vector_id)
        if (
            existing
            and existing.get("text_sha256") == expected_sha
            and int(existing.get("metadata_version") or 0) == PAIS_METADATA_VERSION
        ):
            continue
        if existing:
            stale_ids.append(vector_id)
        to_add.append(record)

    if stale_ids:
        embeddings_manager.collection.delete(ids=stale_ids)

    for batch in _batched(to_add, batch_size):
        texts = [record["embedding_text"] for record in batch]
        embeddings = embeddings_manager.generate_embeddings(texts)
        embeddings_manager.collection.add(
            ids=[_evidence_vector_id(record) for record in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[_metadata_for_record(record) for record in batch],
        )

    return len(records)


def search_pais_evidence_semantic(
    query: str,
    database,
    embeddings_manager: EmbeddingsManager,
    limit: int = 10,
    distance_threshold: Optional[float] = None,
    years: Optional[list[int]] = None,
    sources: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Search PAIS evidence vectors and return paper-like records for the UI/RAG."""
    ensure_pais_evidence_collection(database, embeddings_manager, years=years, sources=sources)
    n_results = max(limit * 2, limit)
    results = _query_pais_collection(
        embeddings_manager,
        query,
        n_results=n_results,
        years=years,
        sources=sources,
    )
    papers = format_pais_evidence_search_results(results)
    if distance_threshold is not None:
        filtered = [paper for paper in papers if paper.get("distance", 0.0) <= distance_threshold]
        if filtered:
            papers = filtered
    return papers[:limit]


def count_pais_evidence_within_distance(
    query: str,
    database,
    embeddings_manager: EmbeddingsManager,
    distance_threshold: float,
    years: Optional[list[int]] = None,
    sources: Optional[list[str]] = None,
) -> int:
    """Count PAIS evidence records within an embedding distance."""
    ensure_pais_evidence_collection(database, embeddings_manager, years=years, sources=sources)
    total = embeddings_manager.collection.count()
    if total <= 0:
        return 0
    results = _query_pais_collection(
        embeddings_manager,
        query,
        n_results=min(total, 1000),
        years=years,
        sources=sources,
    )
    distances = results.get("distances", [[]])[0] if results else []
    return sum(1 for distance in distances if distance <= distance_threshold)


def format_pais_evidence_search_results(search_results: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert Chroma PAIS evidence hits into paper-like dictionaries."""
    ids = search_results.get("ids", [[]])[0]
    metadatas = search_results.get("metadatas", [[]])[0]
    documents = search_results.get("documents", [[]])[0]
    distances = search_results.get("distances", [[]])[0]
    papers: list[dict[str, Any]] = []
    for idx, _vector_id in enumerate(ids):
        metadata = metadatas[idx] if idx < len(metadatas) else {}
        document = documents[idx] if idx < len(documents) else ""
        distance = distances[idx] if idx < len(distances) else None
        paper = _paper_from_metadata(metadata, document)
        if distance is not None:
            paper["distance"] = distance
            paper["similarity"] = max(0.0, 1.0 - distance)
            paper["relevance_score"] = paper["similarity"]
        papers.append(paper)
    return papers


def _query_pais_collection(
    embeddings_manager: EmbeddingsManager,
    query: str,
    n_results: int,
    years: Optional[list[int]] = None,
    sources: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Query PAIS evidence Chroma data without applying legacy paper metadata parsing."""
    total = embeddings_manager.collection.count()
    if total <= 0:
        return {"ids": [[]], "metadatas": [[]], "documents": [[]], "distances": [[]]}
    query_embedding = embeddings_manager.generate_embedding(query)
    kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": max(1, min(n_results, total)),
    }
    where = _chroma_where(years=years, sources=sources)
    if where:
        kwargs["where"] = where
    return embeddings_manager.collection.query(**kwargs)


def compute_pais_evidence_clusters(
    database,
    embeddings_manager: EmbeddingsManager,
    years: Optional[list[int]] = None,
    sources: Optional[list[str]] = None,
    n_clusters: Optional[int] = None,
) -> dict[str, Any]:
    """Compute or load cached PAIS evidence clusters."""
    records = fetch_pais_evidence_records(database, years=years, sources=sources)
    if not records:
        return {
            "points": [],
            "cluster_centers": {},
            "cluster_labels": {},
            "statistics": {"n_papers": 0, "n_clusters": 0},
        }

    ensure_pais_evidence_collection(database, embeddings_manager, years=years, sources=sources)
    requested_clusters = n_clusters or _default_cluster_count(len(records))
    params = _cluster_cache_params(records, years=years, sources=sources, requested_clusters=requested_clusters)
    cached = database.get_clustering_cache(
        embedding_model=embeddings_manager.model_name,
        reduction_method=PAIS_CLUSTER_REDUCTION,
        n_components=2,
        clustering_method=PAIS_CLUSTER_METHOD,
        n_clusters=requested_clusters,
        clustering_params=params,
        conference=PAIS_CLUSTER_CONFERENCE,
        year=None,
    )
    if cached:
        cached.setdefault("metadata", {})["cache_hit"] = True
        return cached

    vectors = _load_vectors_for_records(embeddings_manager, records)
    matrix = np.asarray(vectors, dtype=float)
    selection = _select_umap_clustering(matrix, requested_clusters)
    coords = selection["coords"]
    labels = [int(label) for label in selection["labels"]]

    points = []
    for record, coord, cluster_id in zip(records, coords, labels):
        source = str(record.get("source") or "PAISDB")
        points.append(
            {
                "id": record["paper_uid"],
                "evidence_id": record["evidence_id"],
                "candidate_relation_id": record["candidate_relation_id"],
                "title": record["title"],
                "x": float(coord[0]),
                "y": float(coord[1]),
                "cluster": int(cluster_id),
                "year": record.get("publication_year"),
                "conference": source,
                "source": source,
                "session": record.get("pais_category") or "",
                "pathogen": record.get("pathogen_name") or "",
                "disease": record.get("disease_name") or "",
                "evidence_type": record.get("evidence_type") or "",
            }
        )

    cluster_centers = _cluster_centers(points)
    cluster_labels = _cluster_labels(records, labels)
    n_noise = sum(1 for label in labels if label == -1)
    n_real_clusters = len({label for label in labels if label != -1})
    results = {
        "points": points,
        "cluster_centers": cluster_centers,
        "cluster_labels": cluster_labels,
        "cluster_keywords": {str(k): v.split(" / ") for k, v in cluster_labels.items()},
        "statistics": {
            "n_papers": len(points),
            "n_clusters": n_real_clusters,
            "n_noise": n_noise,
            "embedding_model": embeddings_manager.model_name,
            "source": "pais_evidence_records",
            "reduction_method": PAIS_CLUSTER_REDUCTION,
            "clustering_method": selection["method"],
        },
        "metadata": {"cache_hit": False, **params, **selection["metadata"]},
    }
    database.save_clustering_cache(
        embedding_model=embeddings_manager.model_name,
        reduction_method=PAIS_CLUSTER_REDUCTION,
        n_components=2,
        clustering_method=PAIS_CLUSTER_METHOD,
        n_clusters=requested_clusters,
        clustering_params=params,
        conference=PAIS_CLUSTER_CONFERENCE,
        year=None,
        results=results,
    )
    return results


def _normalize_record(row: dict[str, Any]) -> dict[str, Any]:
    record = dict(row)
    record["paper_uid"] = record.get("paper_uid") or f"pais_article_{int(record['article_id']):012d}"
    record["embedding_text"] = str(record.get("embedding_text") or "").strip()
    return record


def _existing_metadata_by_id(embeddings_manager: EmbeddingsManager, ids: list[str]) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    for batch in _batched(ids, 500):
        try:
            result = embeddings_manager.collection.get(ids=batch, include=["metadatas"])
        except Exception:
            continue
        for vector_id, metadata in zip(result.get("ids", []), result.get("metadatas", [])):
            existing[vector_id] = metadata or {}
    return existing


def _load_vectors_for_records(
    embeddings_manager: EmbeddingsManager, records: list[dict[str, Any]]
) -> list[list[float]]:
    vectors_by_id: dict[str, list[float]] = {}
    ids = [_evidence_vector_id(record) for record in records]
    for batch in _batched(ids, 500):
        result = embeddings_manager.collection.get(ids=batch, include=["embeddings"])
        embeddings = result.get("embeddings", [])
        for vector_id, embedding in zip(result.get("ids", []), embeddings):
            vectors_by_id[vector_id] = list(embedding)
    missing = [vector_id for vector_id in ids if vector_id not in vectors_by_id]
    if missing:
        raise RuntimeError(f"Missing PAIS evidence vectors after bootstrap: {missing[:5]}")
    return [vectors_by_id[vector_id] for vector_id in ids]


def _metadata_for_record(record: dict[str, Any]) -> dict[str, Any]:
    source = str(record.get("source") or "PAISDB")
    return {
        "metadata_version": PAIS_METADATA_VERSION,
        "uid": record["paper_uid"],
        "paper_uid": record["paper_uid"],
        "evidence_id": int(record["evidence_id"]),
        "candidate_relation_id": int(record["candidate_relation_id"]),
        "article_id": int(record["article_id"]),
        "pmid": str(record.get("pmid") or ""),
        "doi": str(record.get("doi") or ""),
        "title": str(record.get("title") or ""),
        "abstract": str(record.get("abstract") or ""),
        "journal": str(record.get("journal") or ""),
        "year": int(record["publication_year"]) if record.get("publication_year") else "",
        "publication_year": int(record["publication_year"]) if record.get("publication_year") else "",
        "conference": source,
        "source": source,
        "session": str(record.get("pais_category") or ""),
        "authors": str(record.get("journal") or source),
        "pathogen": str(record.get("pathogen_name") or ""),
        "disease": str(record.get("disease_name") or ""),
        "pais_category": str(record.get("pais_category") or ""),
        "relation_type": str(record.get("relation_type") or ""),
        "evidence_type": str(record.get("evidence_type") or ""),
        "screen_status": str(record.get("screen_status") or ""),
        "screen_confidence": str(record.get("screen_confidence") or ""),
        "evidence_confidence": str(record.get("evidence_confidence") or ""),
        "keywords": _keywords(record),
        "text_sha256": _text_sha(record["embedding_text"]),
    }


def _paper_from_metadata(metadata: dict[str, Any], document: str) -> dict[str, Any]:
    title = str(metadata.get("title") or "Untitled PAIS evidence")
    pathogen = str(metadata.get("pathogen") or "")
    disease = str(metadata.get("disease") or "")
    source = str(metadata.get("source") or metadata.get("conference") or "PAISDB")
    return {
        "uid": str(metadata.get("paper_uid") or metadata.get("uid") or ""),
        "evidence_id": metadata.get("evidence_id"),
        "candidate_relation_id": metadata.get("candidate_relation_id"),
        "title": title,
        "authors": _authors_from_metadata(metadata),
        "abstract": document or str(metadata.get("abstract") or ""),
        "source_abstract": str(metadata.get("abstract") or ""),
        "year": metadata.get("publication_year") or metadata.get("year") or "",
        "conference": source,
        "source": source,
        "session": str(metadata.get("pais_category") or ""),
        "keywords": str(metadata.get("keywords") or ""),
        "pathogen": pathogen,
        "disease": disease,
        "pais_category": str(metadata.get("pais_category") or ""),
        "evidence_type": str(metadata.get("evidence_type") or ""),
        "relation_type": str(metadata.get("relation_type") or ""),
        "url": "",
        "paper_pdf_url": "",
        "poster_image_url": "",
    }


def _authors_from_metadata(metadata: dict[str, Any]) -> list[str]:
    authors = metadata.get("authors")
    if isinstance(authors, list):
        return [str(author) for author in authors if str(author).strip()]
    text = str(authors or metadata.get("journal") or "PAISDB").strip()
    if not text:
        return ["PAISDB"]
    if ";" in text:
        parts = [part.strip() for part in text.split(";")]
    else:
        parts = [text]
    return [part for part in parts if part] or ["PAISDB"]


def _chroma_where(
    years: Optional[list[int]] = None,
    sources: Optional[list[str]] = None,
) -> Optional[dict[str, Any]]:
    filters: list[dict[str, Any]] = []
    if years:
        filters.append({"year": {"$in": [int(year) for year in years]}})
    if sources:
        clean_sources = [str(source) for source in sources if str(source).strip()]
        if clean_sources:
            filters.append({"source": {"$in": clean_sources}})
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def _select_umap_clustering(matrix: np.ndarray, requested_clusters: int) -> dict[str, Any]:
    if len(matrix) == 1:
        return {
            "coords": np.asarray([[0.0, 0.0]]),
            "labels": [0],
            "method": "single",
            "metadata": {"selected_umap_n_neighbors": None, "selected_umap_min_dist": None},
        }

    if len(matrix) < 4:
        coords = _linear_coords(matrix)
        labels = list(range(len(matrix)))
        return {
            "coords": coords,
            "labels": labels,
            "method": "trivial",
            "metadata": {"selected_umap_n_neighbors": None, "selected_umap_min_dist": None},
        }

    best: Optional[dict[str, Any]] = None
    for n_neighbors in _valid_umap_neighbors(len(matrix)):
        for min_dist in _UMAP_MIN_DIST:
            coords = _fit_umap(matrix, 2, n_neighbors=n_neighbors, min_dist=min_dist)
            clusterable = _fit_umap(matrix, min(15, len(matrix) - 1), n_neighbors=n_neighbors, min_dist=min_dist)
            for candidate in _candidate_clusterings(clusterable, coords, requested_clusters):
                candidate["coords"] = coords
                candidate["metadata"].update(
                    {
                        "selected_umap_n_neighbors": n_neighbors,
                        "selected_umap_min_dist": min_dist,
                    }
                )
                if best is None or candidate["score"] > best["score"]:
                    best = candidate

    if best is None:
        coords = _fit_umap(
            matrix,
            2,
            n_neighbors=min(max(2, len(matrix) - 1), _UMAP_N_NEIGHBORS[0]),
            min_dist=0.1,
        )
        labels = _fit_kmeans(coords, requested_clusters)
        best = {
            "coords": coords,
            "labels": labels,
            "method": "kmeans",
            "score": 0.0,
            "metadata": {
                "selected_umap_n_neighbors": min(max(2, len(matrix) - 1), _UMAP_N_NEIGHBORS[0]),
                "selected_umap_min_dist": 0.1,
                "cluster_selection_score": 0.0,
                "cluster_selection_reason": "fallback",
            },
        }

    best["metadata"]["cluster_selection_score"] = float(best.pop("score"))
    return best


def _valid_umap_neighbors(n_records: int) -> list[int]:
    max_neighbors = max(2, n_records - 1)
    return sorted(
        {
            min(max_neighbors, neighbors)
            for neighbors in _UMAP_N_NEIGHBORS
            if min(max_neighbors, neighbors) >= 2
        }
    )


def _fit_umap(matrix: np.ndarray, n_components: int, n_neighbors: int, min_dist: float) -> np.ndarray:
    cache_dir = Path(
        os.environ.setdefault(
            "NUMBA_CACHE_DIR",
            os.environ.get(
                "ABSTRACTS_EXPLORER_NUMBA_CACHE_DIR",
                str(Path(tempfile.gettempdir()) / "abstracts-explorer-numba-cache"),
            ),
        )
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    import umap

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=0,
        n_jobs=1,
    )
    return reducer.fit_transform(matrix)


def _candidate_clusterings(
    clusterable: np.ndarray,
    coords: np.ndarray,
    requested_clusters: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    if HDBSCAN is not None and len(clusterable) >= 12:
        for min_cluster_size in _hdbscan_cluster_sizes(len(clusterable)):
            for min_samples in (1, 3, 5):
                labels = HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min(min_samples, min_cluster_size),
                    metric="euclidean",
                ).fit_predict(clusterable)
                score, stats = _score_labels(clusterable, coords, labels, requested_clusters)
                if not _hdbscan_guardrails(labels):
                    score -= 5.0
                    stats["cluster_selection_reason"] = "hdbscan_guardrail_penalty"
                candidates.append(
                    {
                        "labels": [int(label) for label in labels],
                        "method": "hdbscan",
                        "score": score,
                        "metadata": {
                            **stats,
                            "hdbscan_min_cluster_size": min_cluster_size,
                            "hdbscan_min_samples": min(min_samples, min_cluster_size),
                        },
                    }
                )

    for n_clusters in _kmeans_cluster_counts(len(clusterable), requested_clusters):
        labels = _fit_kmeans(clusterable, n_clusters)
        score, stats = _score_labels(clusterable, coords, labels, requested_clusters)
        candidates.append(
            {
                "labels": labels,
                "method": "kmeans",
                "score": score,
                "metadata": {**stats, "kmeans_n_clusters": n_clusters},
            }
        )

    return candidates


def _score_labels(
    clusterable: np.ndarray,
    coords: np.ndarray,
    labels: Iterable[int],
    requested_clusters: int,
) -> tuple[float, dict[str, Any]]:
    labels_array = np.asarray([int(label) for label in labels])
    non_noise = labels_array != -1
    unique = {int(label) for label in labels_array[non_noise]}
    n_clusters = len(unique)
    noise_fraction = 1.0 - (float(non_noise.sum()) / float(len(labels_array)))
    if n_clusters < 2 or non_noise.sum() <= n_clusters:
        return -100.0, {
            "silhouette_clusterable": None,
            "silhouette_2d": None,
            "davies_bouldin": None,
            "noise_fraction": noise_fraction,
            "cluster_selection_reason": "too_few_clusters",
        }

    try:
        sil_clusterable = float(silhouette_score(clusterable[non_noise], labels_array[non_noise]))
    except Exception:
        sil_clusterable = -1.0
    try:
        sil_2d = float(silhouette_score(coords[non_noise], labels_array[non_noise]))
    except Exception:
        sil_2d = -1.0
    try:
        dbi = float(davies_bouldin_score(clusterable[non_noise], labels_array[non_noise]))
    except Exception:
        dbi = 10.0

    counts = Counter(int(label) for label in labels_array if int(label) != -1)
    tiny_fraction = sum(count for count in counts.values() if count < 5) / float(len(labels_array))
    cluster_count_penalty = abs(n_clusters - requested_clusters) / max(1.0, float(requested_clusters))
    score = (
        sil_clusterable
        + 0.5 * sil_2d
        - 0.05 * dbi
        - 0.8 * noise_fraction
        - 0.4 * tiny_fraction
        - 0.15 * cluster_count_penalty
    )
    return score, {
        "silhouette_clusterable": sil_clusterable,
        "silhouette_2d": sil_2d,
        "davies_bouldin": dbi,
        "noise_fraction": noise_fraction,
        "tiny_cluster_fraction": tiny_fraction,
        "cluster_count_penalty": cluster_count_penalty,
        "cluster_selection_reason": "scored",
    }


def _hdbscan_guardrails(labels: Iterable[int]) -> bool:
    labels_list = [int(label) for label in labels]
    non_noise = [label for label in labels_list if label != -1]
    n_clusters = len(set(non_noise))
    noise_fraction = 1.0 - (len(non_noise) / float(len(labels_list)))
    counts = Counter(non_noise)
    min_cluster = min(counts.values()) if counts else 0
    return 4 <= n_clusters <= 24 and noise_fraction <= 0.25 and min_cluster >= 5


def _hdbscan_cluster_sizes(n_records: int) -> list[int]:
    candidates = {5, 8, 12, max(5, int(math.sqrt(n_records))), max(5, n_records // 25)}
    return sorted(size for size in candidates if 2 <= size <= max(2, n_records // 2))


def _kmeans_cluster_counts(n_records: int, requested_clusters: int) -> list[int]:
    max_clusters = min(24, max(2, n_records - 1))
    candidates = {requested_clusters, 8, 10, 12, 14, 16}
    return sorted(min(max_clusters, max(2, count)) for count in candidates)


def _fit_kmeans(matrix: np.ndarray, n_clusters: int) -> list[int]:
    n_clusters = min(max(1, n_clusters), len(matrix))
    if n_clusters <= 1:
        return [0 for _ in matrix]
    model = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    return [int(label) for label in model.fit_predict(matrix)]


def _linear_coords(matrix: np.ndarray) -> np.ndarray:
    if len(matrix) == 1:
        return np.asarray([[0.0, 0.0]])
    return np.column_stack([np.arange(len(matrix), dtype=float), np.zeros(len(matrix), dtype=float)])


def _cluster_centers(points: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    centers: dict[str, dict[str, float]] = {}
    clusters = sorted({int(point["cluster"]) for point in points if int(point["cluster"]) != -1})
    for cluster_id in clusters:
        cluster_points = [point for point in points if int(point["cluster"]) == cluster_id]
        centers[str(cluster_id)] = {
            "x": sum(float(point["x"]) for point in cluster_points) / len(cluster_points),
            "y": sum(float(point["y"]) for point in cluster_points) / len(cluster_points),
        }
    return centers


def _cluster_labels(records: list[dict[str, Any]], labels: list[int]) -> dict[str, str]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record, label in zip(records, labels):
        grouped.setdefault(int(label), []).append(record)
    cluster_labels = {}
    for cluster_id, cluster_records in grouped.items():
        if cluster_id == -1:
            cluster_labels[str(cluster_id)] = "Outliers / mixed PAIS evidence"
            continue
        pathogen = _most_common(cluster_records, "pathogen_name")
        disease = _most_common(cluster_records, "disease_name")
        category = _most_common(cluster_records, "pais_category")
        parts = [part for part in (pathogen, disease, category) if part]
        cluster_labels[str(cluster_id)] = " / ".join(parts[:3]) or f"PAIS evidence {cluster_id}"
    return cluster_labels


def _most_common(records: list[dict[str, Any]], key: str) -> str:
    values = [str(record.get(key) or "").strip() for record in records]
    values = [value for value in values if value and value.lower() != "unknown"]
    if not values:
        return ""
    return Counter(values).most_common(1)[0][0]


def _default_cluster_count(n_records: int) -> int:
    if n_records <= 1:
        return 1
    return min(12, max(2, round(math.sqrt(n_records / 2))))


def _cluster_cache_params(
    records: list[dict[str, Any]],
    years: Optional[list[int]] = None,
    sources: Optional[list[str]] = None,
    requested_clusters: Optional[int] = None,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["evidence_id"]).encode("utf-8"))
        digest.update(b":")
        digest.update(_text_sha(record["embedding_text"]).encode("utf-8"))
        digest.update(b"\n")
    return {
        "source": "pais_evidence_records",
        "record_count": len(records),
        "evidence_hash": digest.hexdigest(),
        "algorithm_version": PAIS_CLUSTER_VERSION,
        "filter_years": [int(year) for year in years] if years else [],
        "filter_sources": [str(source) for source in sources] if sources else [],
        "requested_clusters": requested_clusters,
    }


def _evidence_vector_id(record: dict[str, Any]) -> str:
    return f"pais-evidence-{record['evidence_id']}"


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _keywords(record: dict[str, Any]) -> str:
    values = [
        record.get("pathogen_name"),
        record.get("disease_name"),
        record.get("pais_category"),
        record.get("evidence_type"),
        record.get("relation_type"),
    ]
    return ", ".join(str(value) for value in values if value)


def _batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
