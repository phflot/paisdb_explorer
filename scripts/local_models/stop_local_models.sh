#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PID_DIR="$ROOT/pids/local_models"
PORTS=("${PAIS_LOCAL_QWEN_PORT:-18100}" "${PAIS_LOCAL_EMBED_PORT:-18180}")

if [ -d "$PID_DIR" ]; then
  for pid_file in "$PID_DIR"/*.pid; do
    [ -e "$pid_file" ] || continue
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" || true
    fi
    rm -f "$pid_file"
  done
fi

for port in "${PORTS[@]}"; do
  while read -r pid; do
    [ -n "$pid" ] || continue
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" || true
    fi
  done < <(pgrep -f "[v]llm serve .* --port $port" || true)
done

for _ in 1 2 3 4 5; do
  running=0
  for port in "${PORTS[@]}"; do
    if pgrep -f "[v]llm serve .* --port $port" >/dev/null 2>&1; then
      running=1
    fi
  done
  if [ "$running" -eq 0 ]; then
    exit 0
  fi
  sleep 1
done
