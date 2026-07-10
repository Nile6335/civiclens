#!/usr/bin/env bash
# Ensure an Ollama server is reachable and the configured model is pulled.
# Prefers a native install (tiny binary, Metal acceleration on macOS); falls back to
# the docker compose "full" profile service if native ollama is unavailable.
set -euo pipefail

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-$(grep -E '^OLLAMA_BASE_URL=' .env 2>/dev/null | cut -d= -f2- || true)}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-$(grep -E '^OLLAMA_MODEL=' .env 2>/dev/null | cut -d= -f2- || true)}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:8b}"

reachable() { curl -sf -m 3 "${OLLAMA_BASE_URL}/api/version" >/dev/null 2>&1; }

if ! reachable; then
  if command -v ollama >/dev/null 2>&1; then
    echo "starting native ollama server..."
    nohup ollama serve >/tmp/civiclens-ollama.log 2>&1 &
    for _ in $(seq 1 30); do reachable && break; sleep 1; done
  elif command -v brew >/dev/null 2>&1; then
    echo "installing ollama via homebrew..."
    brew install ollama
    nohup ollama serve >/tmp/civiclens-ollama.log 2>&1 &
    for _ in $(seq 1 30); do reachable && break; sleep 1; done
  else
    echo "no native ollama; starting docker compose ollama (large image)..."
    docker compose -f infra/docker-compose.yml --profile full up -d ollama
    for _ in $(seq 1 60); do reachable && break; sleep 2; done
  fi
fi

if ! reachable; then
  echo "ERROR: could not reach an Ollama server at ${OLLAMA_BASE_URL}" >&2
  exit 1
fi

echo "ollama reachable at ${OLLAMA_BASE_URL}; ensuring model ${OLLAMA_MODEL} is present..."
if ! curl -sf "${OLLAMA_BASE_URL}/api/tags" | grep -q "\"${OLLAMA_MODEL}\""; then
  curl -sf -X POST "${OLLAMA_BASE_URL}/api/pull" -d "{\"name\": \"${OLLAMA_MODEL}\"}" \
    | while read -r line; do
        status=$(echo "$line" | sed -n 's/.*"status":"\([^"]*\)".*/\1/p')
        [ -n "$status" ] && printf '\r%-60s' "$status"
      done
  echo ""
fi
echo "model ${OLLAMA_MODEL} ready."
