# PAISDB Evidence Pipeline

This repository now includes a PAISDB evidence-building prototype for article-level pathogen-disease screening and extraction.
The initial database-fill batch is the local 1000-row PAIS benchmark dataset from `paisdb2`/`paisdb_local`.

## Architecture

The pipeline input is one article plus one pathogen candidate plus one disease or phenotype candidate. Candidate terms are hypotheses from benchmark rows, PubMed query provenance, metadata, or dictionary/ontology matching; they are not ground truth.

The first model call is always the benchmark-compatible PAIS screen. That screen uses the original PAIS zero-shot prompt and preserves the historical output shape:

```json
{"relationship": 1, "unrelated": 0}
```

or:

```json
{"relationship": 0, "unrelated": 1}
```

The screen result is stored on `pais_candidate_relations` and as a `pais_model_runs` provenance row. Hosted enrichment models cannot overwrite this screen decision.

Negative screens stop after storing the candidate relation and model run. Positive, uncertain, low-confidence, or explicitly adjudicated invalid screens continue to evidence brief generation and structured extraction.

Benchmark gold labels, when available, are stored only as QC/provenance metadata and used for agreement summaries. They are never included in model prompts.

## Model Routing

Database initialization does not require model settings:

```bash
abstracts-explorer pais init-db
```

Before running candidate screening, define the local benchmark-screen model and the hosted enrichment/extraction models in `.env` or the process environment:

```bash
PAIS_SCREEN_MODEL=<local-screen-model>
PAIS_SCREEN_BASE_URL=<local-openai-compatible-base-url>

PAIS_EVIDENCE_BRIEF_MODEL=<hosted-brief-model>
PAIS_EVIDENCE_BRIEF_BASE_URL=<hosted-generation-base-url>

PAIS_EXTRACTION_MODEL=<hosted-extraction-model>
PAIS_EXTRACTION_BASE_URL=<hosted-generation-base-url>

PAIS_EMBEDDING_MODEL=<embedding-model>
PAIS_EMBEDDING_BASE_URL=<embedding-base-url>
```

Use `abstracts-explorer pais smoke --no-network` to confirm which stages are configured without calling any model endpoint.

## Database Tables

The PAIS tables are created through the existing SQLAlchemy `Base.metadata.create_all()` flow:

- `pais_articles`
- `pais_pathogens`
- `pais_disease_phenotypes`
- `pais_candidate_relations`
- `pais_host_contexts`
- `pais_evidence_records`
- `pais_model_runs`
- `pais_embedding_records`

JSON/list fields are stored as JSON-encoded text for compatibility with the existing SQLite/Postgres code style.

## CLI

Initialize tables:

```bash
abstracts-explorer pais init-db
```

Inspect configured endpoints without network calls:

```bash
abstracts-explorer pais smoke --no-network
```

Run a candidate from JSON:

```bash
abstracts-explorer pais run-candidate --input-json examples/pais/giardia-positive.json
```

Run a built-in fixture:

```bash
abstracts-explorer pais run-example giardia-positive
```

Ingest the local PAIS benchmark rows as the first database batch:

```bash
abstracts-explorer pais ingest-benchmark
```

Limit the run while testing model wiring:

```bash
abstracts-explorer pais ingest-benchmark --limit 10
```

Export evidence embedding texts. These texts are the stage-2 PAIS evidence brief text, not a deterministic re-rendering of the structured extraction:

```bash
abstracts-explorer pais export-embedding-texts --output pais_embedding_texts.jsonl
```

Materialize pending embedding metadata through the configured embedding endpoint:

```bash
abstracts-explorer pais embed-pending --limit 100
```

## Web API

The Flask app exposes:

```text
POST /api/pais/run-candidate
```

The request body follows the `PaisCandidateInput` schema:

```json
{
  "article": {
    "title": "Example title",
    "abstract": "Example abstract"
  },
  "pathogen": {
    "name": "Giardia lamblia"
  },
  "disease": {
    "name": "chronic fatigue syndrome"
  }
}
```

The response includes `candidate_relation_id`, `screen_status`, whether Server 2 was called, model run IDs, and evidence/embedding IDs when created.

## Provenance

Every model call stores:

- stage, backend, model, endpoint
- prompt name/version/hash
- schema name/version/hash
- input hash
- raw output
- parsed JSON
- validity and error details
- elapsed time

Every evidence record is traceable to its candidate relation and model runs. Every embedding record tracks the source text hash and embedding lifecycle state.
