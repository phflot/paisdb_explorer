# PAISDB Evidence Pipeline

This repository now includes a PAISDB evidence-building prototype for article-level pathogen-disease screening and extraction.

## Architecture

The pipeline input is one article plus one pathogen candidate plus one disease or phenotype candidate. The first model call is always the benchmark-compatible PAIS screen. That screen preserves the historical output shape:

```json
{"relationship": 1, "unrelated": 0}
```

or:

```json
{"relationship": 0, "unrelated": 1}
```

The screen result is stored on `pais_candidate_relations` and as a `pais_model_runs` provenance row. Hosted enrichment models cannot overwrite this benchmark decision.

High-confidence negative screens stop after storing the candidate relation and model run. Positive, uncertain, low-confidence, or explicitly adjudicated invalid screens continue to evidence brief generation and structured extraction.

## Model Routing

Defaults are configured in `.env.example`:

```bash
PAIS_SCREEN_MODEL=mistralai/Mistral-Small-Instruct-2409
PAIS_SCREEN_BASE_URL=http://localhost:8000/v1

PAIS_EVIDENCE_BRIEF_MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct
PAIS_EVIDENCE_BRIEF_BASE_URL=http://134.96.118.198:18000/v1

PAIS_EXTRACTION_MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct
PAIS_EXTRACTION_BASE_URL=http://134.96.118.198:18000/v1

PAIS_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
PAIS_EMBEDDING_BASE_URL=http://134.96.118.198:18080/v1
```

The Server 2 endpoints were smoke-tested in `llm_server/paisdb_model_host/results/smoke_20260527T163230Z.json`.

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

Export evidence embedding texts:

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
