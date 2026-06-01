# Abstracts Explorer

A package to download conference data and search it with LLM-based semantic search including document retrieval and question answering.

## Features

- 📥 Download conference data from various sources (NeurIPS, ICLR, ICML, ML4PS)
- 💾 Store data in SQL database (SQLite or PostgreSQL) with efficient indexing
- 🔍 Search abstracts by keywords, track, and other attributes
- 🤖 Generate text embeddings for semantic search
- 🔎 Find similar abstracts using AI-powered semantic similarity
- 💬 Interactive RAG chat to ask questions about abstracts and conferences
- 🎨 Cluster and visualize abstract embeddings with interactive plots
- 🌐 Web interface for browsing and searching abstracts
- 🔌 MCP server for LLM-based cluster analysis
- 🗄️ Multi-database backend support (SQLite and PostgreSQL)
- ⚙️ Environment-based configuration with `.env` file support
- 🧬 PAISDB candidate screening and article-level evidence extraction prototype

## Installation

### Quick Start with Docker/Podman 🐳

The easiest way to get started with a complete stack (PostgreSQL + ChromaDB):

First create a `.env` file with your [blablador token](https://sdlaml.pages.jsc.fz-juelich.de/ai/guides/blablador_api_access/):

```bash
LLM_BACKEND_AUTH_TOKEN=your_blablador_token_here
```

Then download [`docker-compose.yml`](https://raw.githubusercontent.com/thawn/abstracts-explorer/refs/heads/main/docker-compose.yml):

```bash
curl -o docker-compose.yml https://raw.githubusercontent.com/thawn/abstracts-explorer/refs/heads/main/docker-compose.yml
```

Then start the services with:

```bash
# Using Podman (recommended)
podman-compose up -d

# Or using Docker
docker-compose up -d

# Access at http://localhost:5000
```

The Docker Compose setup includes:
- **Web UI** on port 5000 (exposed)
- **PostgreSQL** for paper metadata (internal only)
- **ChromaDB** for semantic search (internal only)

📖 **[Complete Docker/Podman Guide](docs/docker.md)**

**Note:** The container images use pre-built static vendor files. Node.js is only needed for local development if you want to rebuild CSS/JS libraries.

### Traditional Installation

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/) package manager, Node.js 14+ (for web UI development)

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/thawn/neurips-abstracts.git
cd abstracts-explorer

# Install dependencies
uv sync --all-extras

# Install Node.js dependencies for web UI
npm install
npm run install:vendor
```

📖 **[Full Installation Guide](docs/installation.md)**

## PAISDB Model Routing

PAISDB generation providers are configured in `config/model_providers.yaml`.
The same registry drives the web UI dropdown and the database-build workflow.
The benchmark screen stays fixed to local `mistralai/Mistral-Small-Instruct-2409`;
the dropdown controls only chat, evidence brief, and structured extraction.

```bash
export PAISDB_AI_API_KEY=...
export PAIS_MODEL_PROVIDERS_CONFIG=config/model_providers.yaml
export PAIS_GENERATION_PROVIDER=remote_deepseek_v4_pro
export PAIS_GENERATION_FALLBACKS=local_qwen3_coder_30b,remote_mistral_medium35
export PAIS_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
export PAIS_EMBEDDING_BASE_URL=http://127.0.0.1:18180/v1
```

Start the local one-GPU fallback services from this repo. The scripts reuse the
tested `llm_server/paisdb_model_host` vLLM environment and shared HF cache when
available; `bootstrap_vllm_env.sh` only creates a repo-local fallback env if
that shared environment is missing or `PAIS_LOCAL_VLLM_ENV` is set. Runtime
home/cache paths are redirected under `.cache/local_models/`.

```bash
scripts/local_models/bootstrap_vllm_env.sh
CUDA_VISIBLE_DEVICES=0 scripts/local_models/start_embedding_qwen8b.sh
CUDA_VISIBLE_DEVICES=1 scripts/local_models/start_generation_qwen30b.sh
scripts/local_models/smoke_local_models.sh
```

Run the database build with an explicit provider:

```bash
abstracts-explorer pais ingest-benchmark --batched --generation-provider remote_deepseek_v4_pro
abstracts-explorer pais ingest-benchmark --batched --generation-provider local_qwen3_coder_30b
abstracts-explorer pais ingest-benchmark --batched --generation-provider auto
```

## Configuration

Create a `.env` file to customize settings:

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

📖 **[Configuration Guide](docs/configuration.md)** - Complete list of settings and options

### Database Backend

Abstracts Explorer supports both SQLite and PostgreSQL backends:

```bash
# Option 1: SQLite (default, no additional setup required)
PAPER_DB=data/abstracts.db

# Option 2: PostgreSQL (requires PostgreSQL server)
PAPER_DB=postgresql://user:password@localhost/abstracts
```

**PostgreSQL Setup:**

```bash
# Install PostgreSQL support
uv sync --extra postgres

# Create database
createdb abstracts

# Configure in .env
PAPER_DB=postgresql://user:password@localhost/abstracts
```

📖 See [Configuration Guide](docs/configuration.md) for more database options

### PAISDB Evidence Pipeline

PAISDB candidate screening and article-level evidence extraction is available under the `pais` CLI group:

```bash
abstracts-explorer pais init-db
abstracts-explorer pais smoke --no-network
abstracts-explorer pais run-example giardia-positive
abstracts-explorer pais ingest-benchmark --limit 10
```

The local Mistral-Small-Instruct-2409 benchmark screen is the gatekeeper. The 1000-row PAIS benchmark can be ingested as the first database-fill batch; its gold labels are used for QC summaries, not as model prompt input. Hosted Server 2 models are used only for evidence enrichment and structured extraction after a positive or uncertain screen.
`pais init-db` does not require model configuration; candidate runs require the `PAIS_*_MODEL` and `PAIS_*_BASE_URL` settings to be supplied first.

📖 See [PAISDB Evidence Pipeline](docs/pais_pipeline.md) for model routing, schemas, provenance, and run commands.

## Quick Start

### Download Conference Data

```bash
# Download NeurIPS 2025 abstracts
abstracts-explorer download --year 2025
```

### Generate Embeddings for Semantic Search

```bash
# Requires LM Studio running with embedding model loaded
abstracts-explorer create-embeddings
```

### Cluster and Visualize Embeddings

```bash
# Cluster embeddings using K-Means (PCA reduction)
abstracts-explorer cluster-embeddings --n-clusters 8 --output clusters.json

# Cluster using t-SNE and DBSCAN
abstracts-explorer cluster-embeddings \
  --reduction-method tsne \
  --clustering-method dbscan \
  --eps 0.5 \
  --min-samples 5 \
  --output clusters.json

# Cluster using Agglomerative with distance threshold
abstracts-explorer cluster-embeddings \
  --clustering-method agglomerative \
  --distance-threshold 5.0 \
  --output clusters.json

# Cluster using Spectral clustering
abstracts-explorer cluster-embeddings \
  --clustering-method spectral \
  --n-clusters 10 \
  --output clusters.json

# The web UI includes an interactive cluster visualization tab!
```

### Start MCP Server for Cluster Analysis

```bash
# Start MCP server for LLM-based cluster analysis
abstracts-explorer mcp-server

# The MCP server provides tools to analyze clustered abstracts, such as:
# - Get most frequently mentioned topics
# - Analyze topic evolution over years
# - Find recent developments in topics
# - Generate cluster visualizations
```

**NEW**: MCP clustering tools are now **automatically integrated** into the RAG chat! The LLM will automatically use clustering tools when appropriate to answer questions about topics, trends, and developments. No need to run a separate MCP server for RAG chat usage.

### Start Web Interface

```bash
abstracts-explorer web-ui
# Open http://127.0.0.1:5000 in your browser
```

📖 **[Usage Guide](docs/usage.md)** - Detailed examples and workflows  
📖 **[CLI Reference](docs/cli_reference.md)** - Complete command-line documentation  
📖 **[API Reference](docs/api/modules.md)** - Python API documentation

## Web Interface

The web UI provides an intuitive interface for browsing and searching abstracts with powerful features:

- 🔍 **Search**: Keyword and AI-powered semantic search  
- 💬 **Chat**: Interactive RAG chat with query rewriting
- ⭐ **Ratings**: Save and organize interesting abstracts
- 📊 **Filters**: Filter by track, decision, event type, and more
- 🎨 **Clusters**: Interactive visualization of paper embeddings (NEW!)

```bash
abstracts-explorer web-ui
# Open http://127.0.0.1:5000
```

![Web UI Screenshot](docs/images/screenshot.png)
*The web interface provides an intuitive way to search and explore conference abstracts*

## Python API Examples

### Download and Search Abstracts

```python
from abstracts_explorer.plugins import get_plugin
from abstracts_explorer import DatabaseManager

# Download abstracts using the NeurIPS plugin
neurips_plugin = get_plugin('neurips')
abstracts_data = neurips_plugin.download(year=2025)

# Load into database and search
with DatabaseManager() as db:
    db.create_tables()
    db.add_papers(abstracts_data)
    
    # Search abstracts
    abstracts = db.search_papers(keyword="deep learning", limit=5)
    for abstract in abstracts:
        print(f"{abstract['title']} by {abstract['authors']}")
```

### Semantic Search with Embeddings

```python
from abstracts_explorer import EmbeddingsManager

with EmbeddingsManager() as em:
    em.create_collection()
    em.embed_from_database()
    
    # Find similar abstracts using semantic search
    results = em.search_similar(
        "transformers for natural language processing",
        n_results=5
    )
```

### Cluster and Visualize Embeddings

```python
from abstracts_explorer.clustering import perform_clustering

# Perform complete clustering pipeline
results = perform_clustering(
    reduction_method="tsne",      # or "pca"
    n_components=2,
    clustering_method="kmeans",    # or "dbscan", "agglomerative", "spectral", "fuzzy_cmeans"
    n_clusters=8,
    output_path="clusters.json"
)

# Access clustering results
print(f"Found {results['statistics']['n_clusters']} clusters")
for point in results['points']:
    print(f"Paper: {point['title']} -> Cluster {point['cluster']}")
```

📖 **[Complete Usage Guide](docs/usage.md)** - More examples and workflows

## Documentation

📚 **[Full Documentation](docs/index.md)** - Complete documentation built with Sphinx

### Quick Links

- **[Installation Guide](docs/installation.md)** - Detailed installation instructions
- **[Docker/Podman Guide](docs/docker.md)** - Container deployment with Docker and Podman
- **[Usage Guide](docs/usage.md)** - Examples and workflows  
- **[Configuration Guide](docs/configuration.md)** - Environment variables and settings
- **[CLI Reference](docs/cli_reference.md)** - Command-line interface documentation
- **[Plugins Guide](docs/plugins.md)** - Plugin system and conference downloaders
- **[API Reference](docs/api/modules.md)** - Python API documentation
- **[Contributing Guide](docs/contributing.md)** - Development setup and guidelines

## Development

```bash
# Install with development dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Run linters
ruff check src/ tests/
mypy src/ --ignore-missing-imports
```

📖 **[Contributing Guide](docs/contributing.md)** - Complete development documentation

## Contributing

Contributions are welcome! Please read our [Contributing Guide](docs/contributing.md) for details on:

- Development setup
- Running tests and linters  
- Code style and conventions
- Submitting pull requests

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## Support

For issues, questions, or contributions:
- 🐛 [Report issues](https://github.com/thawn/neurips-abstracts/issues)
- 💬 [Discussions](https://github.com/thawn/neurips-abstracts/discussions)
- 📧 Contact the maintainers
