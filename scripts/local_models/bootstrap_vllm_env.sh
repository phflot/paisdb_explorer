#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SHARED_ENV="/share/runs/2026/04-23-paisdb-phflot/llm_server/paisdb_model_host/.mamba/envs/paisdb_model_host"
ENV_PREFIX="${PAIS_LOCAL_VLLM_ENV:-$SHARED_ENV}"
CACHE_ROOT="${PAIS_LOCAL_CACHE:-$ROOT/.cache/local_models}"

if [ -x "$ENV_PREFIX/bin/vllm" ]; then
  echo "Using existing vLLM environment: $ENV_PREFIX"
  exit 0
fi

if [ -z "${PAIS_LOCAL_VLLM_ENV:-}" ]; then
  ENV_PREFIX="$ROOT/.conda/envs/paisdb-explorer-vllm"
fi

export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export CONDA_PKGS_DIRS="$CACHE_ROOT/conda_pkgs"
mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"

if command -v micromamba >/dev/null 2>&1; then
  micromamba create -y -p "$ENV_PREFIX" -c conda-forge python=3.12 pip
elif command -v mamba >/dev/null 2>&1; then
  mamba create -y -p "$ENV_PREFIX" -c conda-forge python=3.12 pip
else
  conda create -y -p "$ENV_PREFIX" -c conda-forge python=3.12 pip
fi

"$ENV_PREFIX/bin/python" -m pip install -U pip
"$ENV_PREFIX/bin/python" -m pip install -U "vllm>=0.10.2" "transformers>=4.55.2"

echo "Created local vLLM environment: $ENV_PREFIX"
