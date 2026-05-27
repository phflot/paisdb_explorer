"""PAIS evidence vector search and clustering helpers."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any, Iterable, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from abstracts_explorer.embeddings import EmbeddingsManager


PAIS_CLUSTER_CONFERENCE = "PAISDB"
PAIS_CLUSTER_REDUCTION = "pca"
PAIS_CLUSTER_METHOD = "kmeans"
PAIS_CLUSTER_VERSION = 1


def fetch_pais_evidence_records(database, years: Optional[list[int]] = None) -> list[dict[str, Any]]:
    """Fetch positive/uncertain PAIS evidence records used for RAG and clustering."""
    params: list[Any] = []
    year_filter = ""
    if years:
        placeholders = ",".join("?" for _ in years)
        year_filter = f" and a.publication_year in ({placeholders})"
        params.extend(int(year) for year in years)

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
        where er.embedding_text is not null
          and trim(er.embedding_text) != ''
          {year_filter}
        order by er.id
        """,
        tuple(params),
    )
    return [_normalize_record(row) for row in rows]


def ensure_pais_evidence_collection(
    database,
    embeddings_manager: EmbeddingsManager,
    batch_size: int = 32,
    years: Optional[list[int]] = None,
) -> int:
    """Ensure the Chroma collection contains PAIS evidence vectors."""
    records = fetch_pais_evidence_records(database, years=years)
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
        if existing and existing.get("text_sha256") == expected_sha:
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
) -> list[dict[str, Any]]:
    """Search PAIS evidence vectors and return paper-like records for the UI/RAG."""
    ensure_pais_evidence_collection(database, embeddings_manager, years=years)
    n_results = max(limit * 2, limit)
    results = _query_pais_collection(embeddings_manager, query, n_results=n_results)
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
) -> int:
    """Count PAIS evidence records within an embedding distance."""
    ensure_pais_evidence_collection(database, embeddings_manager, years=years)
    total = embeddings_manager.collection.count()
    if total <= 0:
        return 0
    results = _query_pais_collection(embeddings_manager, query, n_results=min(total, 1000))
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
) -> dict[str, Any]:
    """Query PAIS evidence Chroma data without applying legacy paper metadata parsing."""
    total = embeddings_manager.collection.count()
    if total <= 0:
        return {"ids": [[]], "metadatas": [[]], "documents": [[]], "distances": [[]]}
    query_embedding = embeddings_manager.generate_embedding(query)
    return embeddings_manager.collection.query(
        query_embeddings=[query_embedding],
        n_results=max(1, min(n_results, total)),
    )


def compute_pais_evidence_clusters(
    database,
    embeddings_manager: EmbeddingsManager,
    years: Optional[list[int]] = None,
    n_clusters: Optional[int] = None,
) -> dict[str, Any]:
    """Compute or load cached PAIS evidence clusters."""
    records = fetch_pais_evidence_records(database, years=years)
    if not records:
        return {
            "points": [],
            "cluster_centers": {},
            "cluster_labels": {},
            "statistics": {"n_papers": 0, "n_clusters": 0},
        }

    ensure_pais_evidence_collection(database, embeddings_manager, years=years)
    n_clusters = n_clusters or _default_cluster_count(len(records))
    params = _cluster_cache_params(records)
    cached = database.get_clustering_cache(
        embedding_model=embeddings_manager.model_name,
        reduction_method=PAIS_CLUSTER_REDUCTION,
        n_components=2,
        clustering_method=PAIS_CLUSTER_METHOD,
        n_clusters=n_clusters,
        clustering_params=params,
        conference=PAIS_CLUSTER_CONFERENCE,
        year=None,
    )
    if cached:
        cached.setdefault("metadata", {})["cache_hit"] = True
        return cached

    vectors = _load_vectors_for_records(embeddings_manager, records)
    matrix = np.asarray(vectors, dtype=float)
    coords = _reduce_to_2d(matrix)
    labels = _cluster_vectors(matrix, n_clusters)

    points = []
    for record, coord, cluster_id in zip(records, coords, labels):
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
                "conference": "PAISDB",
                "session": record.get("pais_category") or "",
                "pathogen": record.get("pathogen_name") or "",
                "disease": record.get("disease_name") or "",
                "evidence_type": record.get("evidence_type") or "",
            }
        )

    cluster_centers = _cluster_centers(points)
    cluster_labels = _cluster_labels(records, labels)
    results = {
        "points": points,
        "cluster_centers": cluster_centers,
        "cluster_labels": cluster_labels,
        "cluster_keywords": {str(k): v.split(" / ") for k, v in cluster_labels.items()},
        "statistics": {
            "n_papers": len(points),
            "n_clusters": len(cluster_labels),
            "embedding_model": embeddings_manager.model_name,
            "source": "pais_evidence_records",
            "reduction_method": PAIS_CLUSTER_REDUCTION,
            "clustering_method": PAIS_CLUSTER_METHOD,
        },
        "metadata": {"cache_hit": False, **params},
    }
    database.save_clustering_cache(
        embedding_model=embeddings_manager.model_name,
        reduction_method=PAIS_CLUSTER_REDUCTION,
        n_components=2,
        clustering_method=PAIS_CLUSTER_METHOD,
        n_clusters=n_clusters,
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
    return {
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
        "conference": "PAISDB",
        "session": str(record.get("pais_category") or ""),
        "authors": str(record.get("journal") or "PAISDB"),
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
    return {
        "uid": str(metadata.get("paper_uid") or metadata.get("uid") or ""),
        "evidence_id": metadata.get("evidence_id"),
        "candidate_relation_id": metadata.get("candidate_relation_id"),
        "title": title,
        "authors": str(metadata.get("authors") or "PAISDB"),
        "abstract": document or str(metadata.get("abstract") or ""),
        "source_abstract": str(metadata.get("abstract") or ""),
        "year": metadata.get("publication_year") or metadata.get("year") or "",
        "conference": "PAISDB",
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


def _reduce_to_2d(matrix: np.ndarray) -> np.ndarray:
    if len(matrix) == 1:
        return np.asarray([[0.0, 0.0]])
    n_components = min(2, matrix.shape[0], matrix.shape[1])
    coords = PCA(n_components=n_components, random_state=0).fit_transform(matrix)
    if n_components == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])
    return coords


def _cluster_vectors(matrix: np.ndarray, n_clusters: int) -> list[int]:
    if len(matrix) == 1:
        return [0]
    n_clusters = min(max(1, n_clusters), len(matrix))
    model = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    return [int(label) for label in model.fit_predict(matrix)]


def _cluster_centers(points: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    centers: dict[str, dict[str, float]] = {}
    clusters = sorted({int(point["cluster"]) for point in points})
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


def _cluster_cache_params(records: list[dict[str, Any]]) -> dict[str, Any]:
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
