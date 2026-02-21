#!/bin/sh
set -eu

SCALE_TARGET="${SCALE_TARGET:-executor}"
UP="${SCALE_UP_THRESHOLD:-0.80}"
DOWN="${SCALE_DOWN_THRESHOLD:-0.50}"
METRICS_URL="${METRICS_URL:-http://executor:8002/metrics}"

while true; do
  RAW="$(wget -qO- "$METRICS_URL" 2>/dev/null || true)"
  LOAD="$(printf '%s' "$RAW" | sed -n 's/.*"load_ratio"[[:space:]]*:[[:space:]]*\([0-9.]*\).*/\1/p')"
  if [ -n "$LOAD" ]; then
    awk "BEGIN {exit !($LOAD > $UP)}" && docker compose up -d --scale "$SCALE_TARGET"=2 || true
    awk "BEGIN {exit !($LOAD < $DOWN)}" && docker compose up -d --scale "$SCALE_TARGET"=1 || true
  fi
  sleep 20
done
