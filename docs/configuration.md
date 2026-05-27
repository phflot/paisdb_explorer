# Configuration

Abstracts Explorer uses a flexible configuration system that supports environment variables, `.env` files, and command-line arguments.

## Configuration Priority

Settings are loaded in the following priority order (later overrides earlier):

1. Built-in defaults
2. `.env` file in the current directory
3. Environment variables
4. Command-line arguments (when applicable)

## Configuration File

Create a `.env` file in your project directory:

```bash
# Copy the example file
cp .env.example .env

# Edit with your preferences
nano .env
```

## Available Settings

### Chat/Language Model Settings

- **CHAT_MODEL**: The LLM model to use for RAG chat (default: `gemma-3-4b-it-qat`)
- **CHAT_TEMPERATURE**: Temperature for LLM responses, 0.0-2.0 (default: `0.7`)
- **CHAT_MAX_TOKENS**: Maximum tokens in LLM responses (default: `1000`)

### Embedding Model Settings

- **EMBEDDING_MODEL**: The embedding model to use (default: `text-embedding-qwen3-embedding-4b`)

### LLM Backend Configuration

- **LLM_BACKEND_URL**: URL of the LLM backend server (default: `http://localhost:1234`)
- **LLM_BACKEND_AUTH_TOKEN**: Optional authentication token for LLM backend (default: empty)

### PAISDB Evidence Pipeline Settings

- **PAIS_SCREEN_BACKEND**: Benchmark gatekeeper backend. Use `hf_transformers` for local Mistral, or `openai_compatible` only if you run a local Mistral server.
- **PAIS_SCREEN_MODEL**: Benchmark gatekeeper model. Required for candidate screening, not for `pais init-db`.
- **PAIS_SCREEN_REVISION**: Optional Hugging Face snapshot revision for local Mistral reproducibility.
- **PAIS_SCREEN_HF_HOME**: Hugging Face cache root for local Mistral. In this workspace, prefer the copied `paisdb_local/.cache`.
- **PAIS_SCREEN_LOCAL_FILES_ONLY**: Whether local Mistral must use only cached files.
- **PAIS_SCREEN_CUDA_VISIBLE_DEVICES**: Optional GPU selector for local Mistral. Existing scheduler values are preserved.
- **PAIS_SCREEN_MAX_NEW_TOKENS**: Maximum generated tokens for local Mistral screening.
- **PAIS_SCREEN_BASE_URL**: OpenAI-compatible screen endpoint. Required only when `PAIS_SCREEN_BACKEND=openai_compatible`.
- **PAIS_SCREEN_AUTH_TOKEN**: Optional screen endpoint token
- **PAIS_EVIDENCE_BRIEF_MODEL**: Evidence brief model. Required only when hosted evidence enrichment is run.
- **PAIS_EVIDENCE_BRIEF_BASE_URL**: Evidence brief endpoint. Required only when hosted evidence enrichment is run.
- **PAIS_EVIDENCE_BRIEF_AUTH_TOKEN**: Optional evidence brief endpoint token
- **PAIS_EXTRACTION_MODEL**: Structured extraction model. Required only when structured extraction is run.
- **PAIS_EXTRACTION_BASE_URL**: Structured extraction endpoint. Required only when structured extraction is run.
- **PAIS_EXTRACTION_AUTH_TOKEN**: Optional extraction endpoint token
- **PAIS_EMBEDDING_MODEL**: PAIS evidence embedding model. Required only for `pais embed-pending`.
- **PAIS_EMBEDDING_BASE_URL**: PAIS evidence embedding endpoint. Required only for `pais embed-pending`.
- **PAIS_EMBEDDING_AUTH_TOKEN**: Optional embedding endpoint token
- **PAIS_STRUCTURED_OUTPUT_MODE**: Structured output mode (`json_schema` or fallback JSON prompt mode)
- **PAIS_ALLOW_ADJUDICATION_ON_INVALID_SCREEN**: Allow hosted stages after invalid/error screen output (default: `false`)

Benchmark/database-fill CLI tuning:

- **`--batched`**: Use tensor-batched local Mistral screening, chunked hosted enrichment, and optional batch embeddings.
- **`--screen-batch-size`**: Local Mistral batch size. `8` worked on the H100 in this workspace.
- **`--hosted-concurrency`** / **`--hosted-chunk-size`**: Remote Qwen concurrency and commit chunk size. Lower concurrency improved JSON validity.
- **`--fallback-from-brief`**: Explicitly create flagged low-confidence evidence records from validated evidence briefs when hosted structured extraction is invalid.
- **`--embed`** / **`--embedding-batch-size`**: Embed pending evidence texts at the end of the ingest run.

### Data Directory

- **DATA_DIR**: Base directory for data files (default: `data`)

### Database Configuration

#### Paper Database

- **PAPER_DB**: Database connection for papers. Can be either:
  - **PostgreSQL URL**: `postgresql://user:password@host:port/database`
  - **SQLite file path**: `abstracts.db` (relative to DATA_DIR) or `/absolute/path/to/abstracts.db`
  - Default: `abstracts.db`

The configuration automatically detects the database type based on the format:
- URLs starting with `postgresql://`, `sqlite://`, or other database schemes are treated as database URLs
- Other values are treated as SQLite file paths (relative to DATA_DIR unless absolute)

#### Embedding Database

- **EMBEDDING_DB**: ChromaDB location. Can be either:
  - **HTTP URL**: `http://chromadb:8000` for remote ChromaDB service (Docker deployments)
  - **File path**: `chroma_db` (relative to DATA_DIR) or `/absolute/path/to/chroma_db` for local ChromaDB
  - Default: `chroma_db`

The configuration automatically detects the type based on the format:
- URLs starting with `http://` or `https://` are treated as remote ChromaDB services
- Other values are treated as local file paths (relative to DATA_DIR unless absolute)

### RAG Settings

- **COLLECTION_NAME**: ChromaDB collection name (default: `papers`)
- **MAX_CONTEXT_PAPERS**: Number of papers to include in RAG context (default: `5`)

### Registry Settings

These settings are used by the `abstracts-explorer registry` commands for sharing data via OCI-compatible container registries (e.g. GitHub Container Registry).

- **GITHUB_TOKEN**: Personal Access Token for authenticating with the container registry. Requires `read:packages` scope for downloads and `write:packages` scope for uploads.
- **REGISTRY_REPOSITORY**: Default OCI repository for registry operations (e.g. `ghcr.io/thawn/abstracts-data`). Can be overridden per command with `-r`.

```bash
# .env file — registry settings
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
REGISTRY_REPOSITORY=ghcr.io/thawn/abstracts-data
```

See [Registry documentation](registry.md) for complete usage instructions.

## Example Configurations

### Local Development (SQLite)

```bash
# .env file for local development

# Base directory for data files
DATA_DIR=data

CHAT_MODEL=diffbot-small-xl-2508
CHAT_TEMPERATURE=0.7
CHAT_MAX_TOKENS=1000

EMBEDDING_MODEL=text-embedding-qwen3-embedding-4b

LLM_BACKEND_URL=http://localhost:1234
LLM_BACKEND_AUTH_TOKEN=

# SQLite database (relative to DATA_DIR - will resolve to data/abstracts.db)
PAPER_DB=abstracts.db

# Local ChromaDB (relative to DATA_DIR - will resolve to data/chroma_db)
EMBEDDING_DB=chroma_db

COLLECTION_NAME=papers
MAX_CONTEXT_PAPERS=5
```

### Production/Docker (PostgreSQL)

```bash
# .env file for production with PostgreSQL

DATA_DIR=data

CHAT_MODEL=diffbot-small-xl-2508
CHAT_TEMPERATURE=0.7
CHAT_MAX_TOKENS=1000

EMBEDDING_MODEL=text-embedding-qwen3-embedding-4b

LLM_BACKEND_URL=http://localhost:1234
LLM_BACKEND_AUTH_TOKEN=

# PostgreSQL database URL
PAPER_DB=postgresql://abstracts:password@postgres:5432/abstracts

# Remote ChromaDB HTTP service
EMBEDDING_DB=http://chromadb:8000

COLLECTION_NAME=papers
MAX_CONTEXT_PAPERS=5
```

### Alternative: Absolute Paths

```bash
# Using absolute paths for both databases
PAPER_DB=/var/data/abstracts.db
EMBEDDING_DB=/var/data/chroma_db
```

## Using Configuration in Code

```python
from abstracts_explorer.config import get_config

# Get the singleton configuration instance
config = get_config()

# Access configuration values
print(f"Chat model: {config.chat_model}")
print(f"Backend URL: {config.llm_backend_url}")
print(f"Database URL: {config.database_url}")  # SQLAlchemy-compatible URL

# ChromaDB location (automatically detected as URL or path)
print(f"ChromaDB: {config.embedding_db}")
```

## Environment Variables

You can also set configuration via environment variables:

```bash
export CHAT_MODEL=llama-3.2-3b-instruct
export LLM_BACKEND_URL=http://localhost:8080
abstracts-explorer chat
```

## Security Best Practices

- Never commit `.env` files to version control
- Use `.env.example` as a template without sensitive data
- Keep authentication tokens secure
- Use environment variables in production environments
