"""
Database Models
===============

This module defines SQLAlchemy ORM models for the database tables.
These models support both SQLite and PostgreSQL backends.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    DateTime,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


class Paper(Base):
    """
    Paper model representing a research paper.

    This uses the lightweight schema from LightweightPaper model.

    Attributes
    ----------
    uid : str
        Unique identifier (hash-based, primary key).
    original_id : str, optional
        Original ID from the source (e.g., OpenReview ID).
    title : str
        Paper title.
    authors : str, optional
        Semicolon-separated list of author names.
    abstract : str, optional
        Paper abstract.
    session : str, optional
        Conference session name.
    poster_position : str, optional
        Poster position identifier.
    paper_pdf_url : str, optional
        URL to paper PDF.
    poster_image_url : str, optional
        URL to poster image.
    url : str, optional
        General URL for the paper.
    room_name : str, optional
        Room name for presentation.
    keywords : str, optional
        Comma-separated keywords.
    starttime : str, optional
        Start time of presentation.
    endtime : str, optional
        End time of presentation.
    award : str, optional
        Award received (e.g., "Best Paper").
    year : int, optional
        Publication year.
    conference : str, optional
        Conference name (e.g., "NeurIPS", "ICLR").
    created_at : datetime
        Timestamp when record was created.
    """

    __tablename__ = "papers"

    uid = Column(String(16), primary_key=True, index=True)
    original_id = Column(String, nullable=True, index=True)
    title = Column(Text, nullable=False, index=True)
    authors = Column(Text, nullable=True)
    abstract = Column(Text, nullable=True)
    session = Column(String, nullable=True, index=True)
    poster_position = Column(String, nullable=True)
    paper_pdf_url = Column(String, nullable=True)
    poster_image_url = Column(String, nullable=True)
    url = Column(String, nullable=True)
    room_name = Column(String, nullable=True)
    keywords = Column(Text, nullable=True)
    starttime = Column(String, nullable=True)
    endtime = Column(String, nullable=True)
    award = Column(String, nullable=True)
    year = Column(Integer, nullable=True, index=True)
    conference = Column(String, nullable=True, index=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()
    )

    def __repr__(self) -> str:
        """String representation of Paper."""
        return f"<Paper(uid='{self.uid}', title='{self.title[:50]}...')>"


class EmbeddingsMetadata(Base):
    """
    Embeddings metadata model.

    Tracks which embedding model was used for the vector embeddings.

    Attributes
    ----------
    id : int
        Auto-incrementing primary key.
    embedding_model : str
        Name of the embedding model used.
    created_at : datetime
        Timestamp when record was created.
    updated_at : datetime
        Timestamp when record was last updated.
    """

    __tablename__ = "embeddings_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    embedding_model = Column(String, nullable=False)
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        """String representation of EmbeddingsMetadata."""
        return f"<EmbeddingsMetadata(id={self.id}, model='{self.embedding_model}')>"


class ClusteringCache(Base):
    """
    Clustering cache model.

    Stores cached clustering results including visualization coordinates.
    When only the dimensionality reduction method changes, the clustering
    results (assignments, labels, hierarchy) are reused and only the reduction
    is re-applied, avoiding expensive re-clustering.

    Attributes
    ----------
    id : int
        Auto-incrementing primary key.
    embedding_model : str
        Name of the embedding model used.
    conference : str, optional
        Conference name this cache entry is scoped to (e.g., 'NeurIPS').
    year : int, optional
        Conference year this cache entry is scoped to.
    reduction_method : str
        Dimensionality reduction method used (e.g., 'pca', 'tsne').
    n_components : int
        Number of dimensions after reduction.
    clustering_method : str
        Clustering algorithm used (e.g., 'kmeans', 'dbscan').
    n_clusters : int, optional
        Actual number of clusters in the cached results.
    clustering_params : str
        JSON string of additional clustering parameters.
    results_json : str
        JSON string containing full clustering results including points
        with visualization coordinates.
    created_at : datetime
        Timestamp when cache was created.
    """

    __tablename__ = "clustering_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    embedding_model = Column(String, nullable=False, index=True)
    conference = Column(String, nullable=True, index=True)
    year = Column(Integer, nullable=True, index=True)
    reduction_method = Column(String, nullable=False)
    n_components = Column(Integer, nullable=False)
    clustering_method = Column(String, nullable=False, index=True)
    n_clusters = Column(Integer, nullable=True)
    clustering_params = Column(Text, nullable=True)
    results_json = Column(Text, nullable=False)
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()
    )

    def __repr__(self) -> str:
        """String representation of ClusteringCache."""
        return (
            f"<ClusteringCache(id={self.id}, conference='{self.conference}', "
            f"year={self.year}, method='{self.clustering_method}', n_clusters={self.n_clusters})>"
        )


class HierarchicalLabelCache(Base):
    """
    Hierarchical label cache model.

    Stores cached hierarchical cluster labels for agglomerative clustering.
    Labels are independent of the number of clusters or distance threshold and
    are reused for all agglomerative clustering settings that share the same
    embedding model and linkage method.

    Attributes
    ----------
    id : int
        Auto-incrementing primary key.
    embedding_model : str
        Name of the embedding model used.
    linkage : str
        Linkage method used in agglomerative clustering (e.g., 'ward').
    labels_json : str
        JSON string mapping node IDs to their generated labels.
    created_at : datetime
        Timestamp when cache was created.
    """

    __tablename__ = "hierarchical_label_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    embedding_model = Column(String, nullable=False, index=True)
    linkage = Column(String, nullable=False, default="ward")
    labels_json = Column(Text, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        """String representation of HierarchicalLabelCache."""
        return f"<HierarchicalLabelCache(id={self.id}, model='{self.embedding_model}', linkage='{self.linkage}')>"


class ValidationData(Base):
    """
    Validation data model.

    Stores anonymized user-donated data about interesting papers
    for validation and service improvement purposes.

    Attributes
    ----------
    id : int
        Auto-incrementing primary key.
    paper_uid : str
        Paper UID reference (anonymized - no direct user identification).
    priority : int
        User-assigned priority/rating (1-5).
    search_term : str, optional
        Search term or context associated with this paper.
    donated_at : datetime
        Timestamp when data was donated.
    """

    __tablename__ = "validation_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_uid = Column(String(16), nullable=False, index=True)
    priority = Column(Integer, nullable=False)
    search_term = Column(String, nullable=True)
    donated_at = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()
    )

    def __repr__(self) -> str:
        """String representation of ValidationData."""
        return f"<ValidationData(id={self.id}, paper_uid='{self.paper_uid}', priority={self.priority})>"


class ChatDonation(Base):
    """
    Chat donation model.

    Stores anonymized user-donated chat transcripts with thumbs up/down
    feedback for improving the chat system.

    Attributes
    ----------
    id : int
        Auto-incrementing primary key.
    rating : str
        User feedback rating ('up' or 'down').
    transcript : str
        JSON string containing the chat transcript (list of messages).
    donated_at : datetime
        Timestamp when data was donated.
    """

    __tablename__ = "chat_donations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    rating = Column(String, nullable=False)
    transcript = Column(Text, nullable=False)
    donated_at = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()
    )

    def __repr__(self) -> str:
        """String representation of ChatDonation."""
        return f"<ChatDonation(id={self.id}, rating='{self.rating}')>"


class EvalQAPair(Base):
    """
    Evaluation query/answer pair.

    Stores queries and their expected answers for automatic evaluation of the
    RAG system. Supports multi-turn conversations via ``conversation_id`` and
    ``turn_number``.

    Attributes
    ----------
    id : int
        Auto-incrementing primary key.
    conversation_id : str
        Groups related queries in a conversation. All turns in the same
        conversation share this ID.
    turn_number : int
        Position within the conversation (0 = initial query, 1+ = follow-ups).
    query : str
        The user query text.
    expected_answer : str
        The expected/reference answer.
    tool_name : str, optional
        The MCP tool expected to be invoked for this query.
    verified : int
        Verification status: 0 = unverified, 1 = verified/approved,
        -1 = rejected/deleted.
    source_info : str, optional
        JSON string with metadata about how the pair was generated
        (e.g. paper UIDs used, generation model).
    created_at : datetime
        Timestamp when the pair was created.
    updated_at : datetime
        Timestamp when the pair was last modified.
    """

    __tablename__ = "eval_qa_pairs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String, nullable=False, index=True)
    turn_number = Column(Integer, nullable=False, default=0)
    query = Column(Text, nullable=False)
    expected_answer = Column(Text, nullable=False)
    tool_name = Column(String, nullable=True, index=True)
    verified = Column(Integer, nullable=False, default=0)
    source_info = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        """String representation of EvalQAPair."""
        return (
            f"<EvalQAPair(id={self.id}, conv='{self.conversation_id}', "
            f"turn={self.turn_number}, tool='{self.tool_name}')>"
        )


class EvalResult(Base):
    """
    Evaluation run result for a single Q/A pair.

    Stores the actual output from the RAG system when evaluated against a
    stored :class:`EvalQAPair`, together with scoring metrics.

    Attributes
    ----------
    id : int
        Auto-incrementing primary key.
    run_id : str
        Identifier grouping results from the same evaluation run.
    qa_pair_id : int
        ID of the :class:`EvalQAPair` that was evaluated.
    actual_answer : str, optional
        The answer produced by the RAG system.
    actual_tool_name : str, optional
        The MCP tool actually invoked by the RAG system.
    answer_score : float, optional
        LLM-judged quality score (1–5 scale).
    tool_correct : int, optional
        Whether the correct tool was used (1 = yes, 0 = no).
    latency_ms : int, optional
        Wall-clock time for the query in milliseconds.
    error : str, optional
        Error message if the query failed.
    judge_reasoning : str, optional
        The LLM judge's reasoning for the assigned score.
    created_at : datetime
        Timestamp when the result was recorded.
    """

    __tablename__ = "eval_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, nullable=False, index=True)
    qa_pair_id = Column(Integer, nullable=False, index=True)
    actual_answer = Column(Text, nullable=True)
    actual_tool_name = Column(String, nullable=True)
    answer_score = Column(Float, nullable=True)
    tool_correct = Column(Integer, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    error = Column(Text, nullable=True)
    judge_reasoning = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        """String representation of EvalResult."""
        return (
            f"<EvalResult(id={self.id}, run='{self.run_id}', "
            f"qa_pair={self.qa_pair_id}, score={self.answer_score})>"
        )


class Article(Base):
    """PubMed/article-level source text for PAIS evidence building."""

    __tablename__ = "pais_articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pmid = Column(String, nullable=True, index=True)
    doi = Column(String, nullable=True, index=True)
    title = Column(Text, nullable=False)
    abstract = Column(Text, nullable=False)
    journal = Column(Text, nullable=True)
    publication_year = Column(Integer, nullable=True, index=True)
    publication_date = Column(String, nullable=True)
    publication_type = Column(String, nullable=True)
    source = Column(String, nullable=True)
    source_url = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        """String representation of Article."""
        return f"<Article(id={self.id}, pmid='{self.pmid}', title='{self.title[:50]}...')>"


class Pathogen(Base):
    """Pathogen candidate entity."""

    __tablename__ = "pais_pathogens"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False, index=True)
    normalized_name = Column(Text, nullable=True, index=True)
    ncbi_taxid = Column(String, nullable=True, index=True)
    taxonomic_rank = Column(String, nullable=True)
    strain_or_variant = Column(Text, nullable=True)
    synonyms_json = Column(Text, nullable=True)

    def __repr__(self) -> str:
        """String representation of Pathogen."""
        return f"<Pathogen(id={self.id}, name='{self.name}')>"


class DiseasePhenotype(Base):
    """Disease or phenotype candidate entity."""

    __tablename__ = "pais_disease_phenotypes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False, index=True)
    normalized_name = Column(Text, nullable=True, index=True)
    doid = Column(String, nullable=True, index=True)
    hpo_id = Column(String, nullable=True, index=True)
    mondo_id = Column(String, nullable=True, index=True)
    synonyms_json = Column(Text, nullable=True)

    def __repr__(self) -> str:
        """String representation of DiseasePhenotype."""
        return f"<DiseasePhenotype(id={self.id}, name='{self.name}')>"


class CandidateRelation(Base):
    """Article-pathogen-disease candidate and benchmark screen decision."""

    __tablename__ = "pais_candidate_relations"
    __table_args__ = (UniqueConstraint("candidate_key", name="uq_pais_candidate_key"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    article_id = Column(Integer, ForeignKey("pais_articles.id"), nullable=False, index=True)
    pathogen_id = Column(Integer, ForeignKey("pais_pathogens.id"), nullable=False, index=True)
    disease_id = Column(Integer, ForeignKey("pais_disease_phenotypes.id"), nullable=False, index=True)
    candidate_key = Column(String(64), nullable=False, index=True)
    benchmark_relationship = Column(Integer, nullable=True)
    benchmark_unrelated = Column(Integer, nullable=True)
    screen_status = Column(String, nullable=True, index=True)
    screen_confidence = Column(String, nullable=True)
    screen_exclusion_reason = Column(String, nullable=True)
    hosted_disagreement_flag = Column(Boolean, nullable=False, default=False, server_default="0")
    quality_flags_json = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        """String representation of CandidateRelation."""
        return f"<CandidateRelation(id={self.id}, status='{self.screen_status}', key='{self.candidate_key[:12]}')>"


class HostContext(Base):
    """Host/model context extracted for a PAIS evidence record."""

    __tablename__ = "pais_host_contexts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    article_id = Column(Integer, ForeignKey("pais_articles.id"), nullable=False, index=True)
    host_name = Column(Text, nullable=True)
    host_taxid = Column(String, nullable=True)
    host_type = Column(String, nullable=False, default="unknown", server_default="unknown")
    species = Column(Text, nullable=True)
    tissue_or_sample = Column(Text, nullable=True)
    cohort_or_model_description = Column(Text, nullable=True)

    def __repr__(self) -> str:
        """String representation of HostContext."""
        return f"<HostContext(id={self.id}, host_type='{self.host_type}')>"


class PAISEvidenceRecord(Base):
    """Source-grounded PAIS evidence record suitable for later database filling."""

    __tablename__ = "pais_evidence_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    candidate_relation_id = Column(Integer, ForeignKey("pais_candidate_relations.id"), nullable=False, index=True)
    host_context_id = Column(Integer, ForeignKey("pais_host_contexts.id"), nullable=True, index=True)
    pais_category = Column(String, nullable=False, default="unclear", server_default="unclear")
    relation_type = Column(String, nullable=False, default="unclear", server_default="unclear")
    evidence_type = Column(String, nullable=False, default="unclear", server_default="unclear")
    timing_after_infection = Column(Text, nullable=True)
    mechanism_summary = Column(Text, nullable=True)
    molecular_data_summary = Column(Text, nullable=True)
    molecular_modalities_json = Column(Text, nullable=True)
    disease_phenotypes_json = Column(Text, nullable=True)
    pathogen_details_json = Column(Text, nullable=True)
    source_evidence_spans_json = Column(Text, nullable=True)
    llm_summary = Column(Text, nullable=True)
    embedding_text = Column(Text, nullable=False)
    confidence = Column(String, nullable=False, default="unknown", server_default="unknown")
    limitations = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        """String representation of PAISEvidenceRecord."""
        return f"<PAISEvidenceRecord(id={self.id}, category='{self.pais_category}')>"


class ModelRun(Base):
    """Provenance for one PAIS model invocation."""

    __tablename__ = "pais_model_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stage = Column(String, nullable=False, index=True)
    article_id = Column(Integer, ForeignKey("pais_articles.id"), nullable=True, index=True)
    candidate_relation_id = Column(Integer, ForeignKey("pais_candidate_relations.id"), nullable=True, index=True)
    evidence_record_id = Column(Integer, ForeignKey("pais_evidence_records.id"), nullable=True, index=True)
    backend = Column(String, nullable=True)
    model_id = Column(Text, nullable=True)
    model_version = Column(Text, nullable=True)
    endpoint_id = Column(Text, nullable=True)
    structured_output_used = Column(Boolean, nullable=False, default=False, server_default="0")
    prompt_name = Column(String, nullable=False)
    prompt_version = Column(String, nullable=False)
    prompt_sha256 = Column(String(64), nullable=False)
    schema_name = Column(String, nullable=True)
    schema_version = Column(String, nullable=True)
    schema_sha256 = Column(String(64), nullable=True)
    input_sha256 = Column(String(64), nullable=False)
    raw_output = Column(Text, nullable=False)
    parsed_json = Column(Text, nullable=True)
    valid = Column(Boolean, nullable=False, default=False, server_default="0")
    error_kind = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    elapsed_s = Column(Float, nullable=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()
    )

    def __repr__(self) -> str:
        """String representation of ModelRun."""
        return f"<ModelRun(id={self.id}, stage='{self.stage}', valid={self.valid})>"


class EmbeddingRecord(Base):
    """Embedding lifecycle record for PAIS evidence text."""

    __tablename__ = "pais_embedding_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    evidence_record_id = Column(Integer, ForeignKey("pais_evidence_records.id"), nullable=False, index=True)
    text_sha256 = Column(String(64), nullable=False, index=True)
    embedding_model = Column(Text, nullable=True)
    embedding_dim = Column(Integer, nullable=True)
    vector_db = Column(Text, nullable=True)
    vector_collection = Column(Text, nullable=True)
    vector_id = Column(Text, nullable=True)
    status = Column(String, nullable=False, default="pending", server_default="pending", index=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        """String representation of EmbeddingRecord."""
        return f"<EmbeddingRecord(id={self.id}, status='{self.status}')>"
