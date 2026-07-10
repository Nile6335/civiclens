#!/usr/bin/env bash
# Launch API + UI for the demo and open the browser. Idempotent: kills stale instances.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

mkdir -p .demo
pkill -f "uvicorn api.main:app" 2>/dev/null || true
pkill -f "streamlit run ui/app.py" 2>/dev/null || true
sleep 1

echo "starting api on :8000 ..."
nohup uv run uvicorn api.main:app --host 127.0.0.1 --port 8000 > .demo/api.log 2>&1 &
echo "starting ui on :8501 ..."
nohup uv run streamlit run ui/app.py --server.port 8501 --server.headless true > .demo/ui.log 2>&1 &

# generous: first boot imports torch/sentence-transformers, slow on small machines
for _ in $(seq 1 90); do
  curl -sf -m 2 http://localhost:8000/health >/dev/null 2>&1 && break
  sleep 1
done
curl -sf -m 2 http://localhost:8000/health >/dev/null || {
  echo "ERROR: api failed to start; last log lines:" >&2; tail -20 .demo/api.log >&2; exit 1;
}

echo ""
echo "CivicLens demo is running:"
echo "  UI:       http://localhost:8501"
echo "  API:      http://localhost:8000/docs"
echo "  Langfuse: http://localhost:3001  (admin@civiclens.local / civiclens-admin)"
echo ""
if [ "$(uname)" = "Darwin" ] && [ -z "${CI:-}" ]; then
  open http://localhost:8501 || true
fi
