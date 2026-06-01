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
PORT="${PAIS_LOCAL_QWEN_PORT:-18100}"
HOST="${PAIS_LOCAL_HOST:-127.0.0.1}"
MODEL="${PAIS_LOCAL_QWEN_MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"

mkdir -p "$LOG_DIR" "$PID_DIR"
mkdir -p "$CACHE_ROOT/home" "$CACHE_ROOT/torch" "$CACHE_ROOT/triton" "$CACHE_ROOT/vllm"
export HOME="${PAIS_LOCAL_HOME:-$CACHE_ROOT/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$CACHE_ROOT/vllm}"
export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN_VLLM_V1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

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

LOG="$LOG_DIR/qwen3_coder_30b_${PORT}.log"
PID_FILE="$PID_DIR/qwen3_coder_30b_${PORT}.pid"
AUTH_ARGS=()
if [ -n "${PAISDB_AI_API_KEY:-}" ]; then
  AUTH_ARGS=(--api-key "$PAISDB_AI_API_KEY")
fi
PARALLEL_ARGS=()
if [ -n "${PAIS_LOCAL_QWEN_TENSOR_PARALLEL_SIZE:-}" ]; then
  PARALLEL_ARGS+=(--tensor-parallel-size "$PAIS_LOCAL_QWEN_TENSOR_PARALLEL_SIZE")
fi
if [ -n "${PAIS_LOCAL_QWEN_PIPELINE_PARALLEL_SIZE:-}" ]; then
  PARALLEL_ARGS+=(--pipeline-parallel-size "$PAIS_LOCAL_QWEN_PIPELINE_PARALLEL_SIZE")
fi
if [ -n "${PAIS_LOCAL_QWEN_DISTRIBUTED_EXECUTOR_BACKEND:-}" ]; then
  PARALLEL_ARGS+=(--distributed-executor-backend "$PAIS_LOCAL_QWEN_DISTRIBUTED_EXECUTOR_BACKEND")
elif [ -n "${PAIS_LOCAL_QWEN_TENSOR_PARALLEL_SIZE:-}${PAIS_LOCAL_QWEN_PIPELINE_PARALLEL_SIZE:-}" ]; then
  PARALLEL_ARGS+=(--distributed-executor-backend mp)
fi

setsid -f "$ENV_PREFIX/bin/vllm" serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$MODEL" \
  --max-model-len "${PAIS_LOCAL_QWEN_MAX_MODEL_LEN:-8192}" \
  --gpu-memory-utilization "${PAIS_LOCAL_QWEN_GPU_MEMORY_UTILIZATION:-0.85}" \
  "${PARALLEL_ARGS[@]}" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --trust-remote-code \
  "${AUTH_ARGS[@]}" \
  >"$LOG" 2>&1 &

launcher_pid="$!"
sleep 2
actual_pid="$(
  pgrep -f "[v]llm serve $MODEL .*--port $PORT" | head -n 1 || true
)"
echo "${actual_pid:-$launcher_pid}" >"$PID_FILE"
echo "Started $MODEL on http://$HOST:$PORT/v1 pid=$(cat "$PID_FILE") log=$LOG"
