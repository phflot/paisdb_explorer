#!/usr/bin/env bash
set -euo pipefail

GEN_URL="${PAIS_LOCAL_QWEN_BASE_URL:-http://127.0.0.1:18100/v1}"
GEN_MODEL="${PAIS_LOCAL_QWEN_MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
EMB_URL="${PAIS_LOCAL_EMBED_BASE_URL:-http://127.0.0.1:18180/v1}"
EMB_MODEL="${PAIS_LOCAL_EMBED_MODEL:-Qwen/Qwen3-Embedding-8B}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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

AUTH_ARGS=()
if [ -n "${PAISDB_AI_API_KEY:-}" ]; then
  AUTH_ARGS=(-H "Authorization: Bearer $PAISDB_AI_API_KEY")
fi

curl -fsS "${AUTH_ARGS[@]}" "$GEN_URL/models" >/dev/null
gen_response="$(curl -fsS "${AUTH_ARGS[@]}" -H "Content-Type: application/json" \
  "$GEN_URL/chat/completions" \
  -d "{\"model\":\"$GEN_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Return exactly: paisdb-local-qwen-ok\"}],\"temperature\":0,\"max_tokens\":16}")"
python -c 'import json,sys; p=json.loads(sys.stdin.read()); print("generation model=%s content=%s" % (p.get("model"), p["choices"][0]["message"].get("content", "")))' <<<"$gen_response"

curl -fsS "${AUTH_ARGS[@]}" "$EMB_URL/models" >/dev/null
emb_response="$(curl -fsS "${AUTH_ARGS[@]}" -H "Content-Type: application/json" \
  "$EMB_URL/embeddings" \
  -d "{\"model\":\"$EMB_MODEL\",\"input\":[\"PAISDB local embedding smoke\"]}")"
python -c 'import json,sys; p=json.loads(sys.stdin.read()); vectors=p.get("data", []); dim=len(vectors[0].get("embedding", [])) if vectors else 0; print("embedding model=%s vectors=%d dim=%d" % (p.get("model"), len(vectors), dim))' <<<"$emb_response"
