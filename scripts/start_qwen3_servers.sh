#!/usr/bin/env bash
# Start local Qwen3 embedding + reranker services for OpenViking on macOS.
# Uses the lightweight torch/transformers server in serve_qwen3_openai.py.
#
# Fixed ports (keep in sync with ~/.openviking/ov.conf):
#   Embedding  -> http://localhost:8834/v1          (model: qwen3-embedding-0.6b)
#   Reranker   -> http://localhost:8835/v1/rank     (model: qwen3-reranker-0.6b)

set -euo pipefail

VENV="${HOME}/.venv-vllm-metal"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
EMBED_PORT=8834
RERANK_PORT=8835

cd "$(dirname "$0")/.."

if [[ ! -f "${VENV}/bin/activate" ]]; then
    echo "ERROR: venv not found at ${VENV}" >&2
    exit 1
fi

source "${VENV}/bin/activate"
export HF_ENDPOINT

mkdir -p /tmp

stop_existing() {
    local port="$1"
    local pids
    pids=$(lsof -ti :"${port}" 2>/dev/null || true)
    if [[ -n "${pids}" ]]; then
        echo "Stopping existing service on port ${port}: ${pids}"
        kill ${pids} 2>/dev/null || true
        sleep 2
    fi
}

stop_existing "${EMBED_PORT}"
stop_existing "${RERANK_PORT}"

echo "Starting Qwen3-Embedding-0.6B on port ${EMBED_PORT} ..."
nohup python3 scripts/serve_qwen3_openai.py \
    --task embed \
    --model Qwen/Qwen3-Embedding-0.6B \
    --port "${EMBED_PORT}" \
    --served-model-name qwen3-embedding-0.6b \
    > /tmp/qwen3-embed-server.log 2>&1 &
echo $! > /tmp/qwen3-embed-server.pid

echo "Starting Qwen3-Reranker-0.6B on port ${RERANK_PORT} ..."
nohup python3 scripts/serve_qwen3_openai.py \
    --task rerank \
    --model Qwen/Qwen3-Reranker-0.6B \
    --port "${RERANK_PORT}" \
    --served-model-name qwen3-reranker-0.6b \
    > /tmp/qwen3-rerank-server.log 2>&1 &
echo $! > /tmp/qwen3-rerank-server.pid

echo "Servers starting. Logs:"
echo "  /tmp/qwen3-embed-server.log"
echo "  /tmp/qwen3-rerank-server.log"
echo ""
echo "Wait ~30-60s for model downloads, then test with:"
echo "  curl http://localhost:${EMBED_PORT}/health"
echo "  curl http://localhost:${RERANK_PORT}/health"
