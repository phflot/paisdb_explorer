#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SHARED_ENV="/share/runs/2026/04-23-paisdb-phflot/llm_server/paisdb_model_host/.mamba/envs/paisdb_model_host"
LOCAL_ENV="$ROOT/.conda/envs/paisdb-explorer-vllm"
ENV_PREFIX="${PAIS_LOCAL_VLLM_ENV:-$SHARED_ENV}"
if [ -z "${PAIS_LOCAL_VLLM_ENV:-}" ] && [ ! -x "$ENV_PREFIX/bin/vllm" ] && [ -x "$LOCAL_ENV/bin/vllm" ]; then
  ENV_PREFIX="$LOCAL_ENV"
fi
HF_HOME_DEFAULT="/share/runs/2026/04-23-paisdb-phflot/llm_server/paisdb_model_host/.cache/huggingface"
CACHE_ROOT="${PAIS_LOCAL_CACHE:-$ROOT/.cache/local_models}"
LOG_DIR="$ROOT/logs/local_models"
PID_DIR="$ROOT/pids/local_models"
PORT="${PAIS_LOCAL_EMBED_PORT:-18180}"
HOST="${PAIS_LOCAL_HOST:-127.0.0.1}"
MODEL="${PAIS_LOCAL_EMBED_MODEL:-Qwen/Qwen3-Embedding-8B}"

mkdir -p "$LOG_DIR" "$PID_DIR"
mkdir -p "$CACHE_ROOT/home" "$CACHE_ROOT/torch" "$CACHE_ROOT/triton" "$CACHE_ROOT/vllm"
export HOME="${PAIS_LOCAL_HOME:-$CACHE_ROOT/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$CACHE_ROOT/vllm}"
export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN_VLLM_V1}"

if [ -z "${PAISDB_AI_API_KEY:-}" ] && [ -f "$ROOT/.env" ]; then
  PAISDB_AI_API_KEY="$(
    awk -F= '
      $1 == "PAISDB_AI_API_KEY" && length($2) { print substr($0, index($0, "=") + 1); found=1; exit }
      $1 == "LLM_BACKEND_AUTH_TOKEN" && !fallback { fallback=substr($0, index($0, "=") + 1) }
      END { if (!found && fallback) print fallback }
    ' "$ROOT/.env"
  )"
  export PAISDB_AI_API_KEY
fi

if [ ! -x "$ENV_PREFIX/bin/vllm" ]; then
  echo "Missing vLLM binary at $ENV_PREFIX/bin/vllm. Run scripts/local_models/bootstrap_vllm_env.sh first." >&2
  exit 1
fi

LOG="$LOG_DIR/qwen3_embedding_8b_${PORT}.log"
PID_FILE="$PID_DIR/qwen3_embedding_8b_${PORT}.pid"
AUTH_ARGS=()
if [ -n "${PAISDB_AI_API_KEY:-}" ]; then
  AUTH_ARGS=(--api-key "$PAISDB_AI_API_KEY")
fi

setsid -f "$ENV_PREFIX/bin/vllm" serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$MODEL" \
  --runner pooling \
  --max-model-len "${PAIS_LOCAL_EMBED_MAX_MODEL_LEN:-32768}" \
  --gpu-memory-utilization "${PAIS_LOCAL_EMBED_GPU_MEMORY_UTILIZATION:-0.85}" \
  "${AUTH_ARGS[@]}" \
  >"$LOG" 2>&1 &

launcher_pid="$!"
sleep 2
actual_pid="$(
  pgrep -f "[v]llm serve $MODEL .*--port $PORT" | head -n 1 || true
)"
echo "${actual_pid:-$launcher_pid}" >"$PID_FILE"
echo "Started $MODEL on http://$HOST:$PORT/v1 pid=$(cat "$PID_FILE") log=$LOG"
