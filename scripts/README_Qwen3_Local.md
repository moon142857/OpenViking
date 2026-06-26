# Local Qwen3 Embedding / Reranker Services for OpenViking (macOS)

This directory contains a lightweight replacement for `vllm-metal` on Apple Silicon.
Because the official `vllm-metal` build requires CMake ≥ 3.26 plus a source build of
vLLM 0.23.0, which is blocked in this environment, we run the Qwen3 models directly
with `torch` + `transformers` behind an OpenAI-compatible FastAPI server.

## Fixed Ports

| Service | Port | Endpoint | Model name seen by OpenViking |
|---------|------|----------|------------------------------|
| Embedding | **8834** | `http://localhost:8834/v1` | `qwen3-embedding-0.6b` |
| Reranker | **8835** | `http://localhost:8835/v1/rerank` | `qwen3-reranker-0.6b` |

These ports are also recorded in `~/.openviking/ov.conf`.

## Quick Start

```bash
cd /Users/xx/repo/OpenViking
./scripts/start_qwen3_servers.sh
```

Then wait ~30-60 seconds for the models to load and test:

```bash
curl http://localhost:8834/health
curl http://localhost:8835/health
```

## Manual Start

```bash
source ~/.venv-vllm-metal/bin/activate
export HF_ENDPOINT=https://hf-mirror.com

python3 scripts/serve_qwen3_openai.py \
  --task embed \
  --model Qwen/Qwen3-Embedding-0.6B \
  --port 8834 \
  --served-model-name qwen3-embedding-0.6b &

python3 scripts/serve_qwen3_openai.py \
  --task rerank \
  --model Qwen/Qwen3-Reranker-0.6B \
  --port 8835 \
  --served-model-name qwen3-reranker-0.6b &
```

## Files

- `serve_qwen3_openai.py` — FastAPI server, supports `--task embed` and `--task rerank`.
- `start_qwen3_servers.sh` — one-shot launcher that kills old processes on 8834/8835 and restarts both services.

## Notes

- `HF_ENDPOINT=https://hf-mirror.com` is used to accelerate HuggingFace downloads.
- The embedding model outputs **1024-dimensional** vectors (matches `ov.conf`).
- The reranker returns 0-1 probability scores via sigmoid over the yes/no token logits.
