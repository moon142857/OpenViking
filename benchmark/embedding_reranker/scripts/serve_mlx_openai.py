#!/usr/bin/env python3
"""OpenAI-compatible MLX server for Qwen3 embedding + reranker (0.6B).

Serves BOTH endpoints from one process using the MLX backends in this repo:
  POST /v1/embeddings  -> benchmark_embedding.embed_texts  (Qwen3-Embedding-0.6B-4bit)
  POST /v1/rerank      -> benchmark_reranker.rerank_scores  (Qwen3-Reranker-0.6B-4bit,
                          cross-encoder, with the suffix-preserving / last-token fixes)

Shapes match what OpenViking's OpenAIDenseEmbedder / OpenAIRerankClient expect, so it
is a drop-in local backend for `ov.conf` (embedding.dense + rerank).

Non-symmetric embeddings: Qwen3-Embedding adds an instruction prefix to QUERIES only.
The request may carry `input_type` ("query" | "document"); OpenViking sends it via the
`query_param` / `document_param` config (merged into the request body). When input_type
is "query" we apply the model's query_prefix+instruction; otherwise we embed as a
document (no instruction).

Usage:
  source ~/mlx-env/bin/activate
  export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  python scripts/serve_mlx_openai.py \
      --emb-config qwen3_embedding_0.6b_4bit_dwq \
      --rerank-config qwen3_reranker_0.6b_4bit \
      --port 11455 --max-length 512
"""

import argparse
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from benchmark_embedding import embed_texts, load_model as load_embedding_model
from benchmark_reranker import load_model as load_reranker_model, rerank_scores
from config.loader import resolve_model_config

logger = logging.getLogger("serve_mlx_openai")

# MLX/Metal calls are serialized: one GPU context, not thread-safe under FastAPI's
# threadpool. Set OpenViking embedding.max_concurrent low too.
_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Request / response schemas (match OpenViking OpenAI clients)
# --------------------------------------------------------------------------- #
class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # tolerate extra_body keys
    model: str
    input: Union[str, List[str]]
    encoding_format: str = "float"
    dimensions: Optional[int] = None
    input_type: Optional[str] = None  # "query" | "document" (from query/document_param)


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


class RerankDocument(BaseModel):
    text: str


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    query: str
    documents: List[Union[str, RerankDocument]]
    top_n: Optional[int] = None
    return_documents: bool = True


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: Optional[RerankDocument] = None


class RerankResponse(BaseModel):
    results: List[RerankResult]


# --------------------------------------------------------------------------- #
# Engines
# --------------------------------------------------------------------------- #
class EmbeddingEngine:
    def __init__(self, config_name: str, max_length: int):
        raw = resolve_model_config(config_name)
        self.model_id = raw["model_id"]
        self.served_name = raw.get("display_name", self.model_id)
        self.loader = raw.get("loader", "mlx_embeddings")
        self.max_length = max_length
        self.instruction = raw.get("instruction", "")
        self.query_prefix = raw.get("query_prefix")
        self.document_prefix = raw.get("document_prefix", "{text}")
        logger.info("Loading embedding %s (loader=%s) ...", self.model_id, self.loader)
        self.model, self.tokenizer, t = load_embedding_model(self.model_id, self.loader)
        self.tokenizer._config = raw.get("embedding", {"pooling": "last", "normalize": True})
        logger.info("Embedding ready in %.1fs", t)

    def encode(self, texts: List[str], is_query: bool) -> np.ndarray:
        prefix = self.query_prefix if is_query else self.document_prefix
        instruction = self.instruction if is_query else ""
        with _LOCK:
            return embed_texts(
                texts, self.model, self.tokenizer, self.max_length,
                self.loader, prefix, instruction,
            )


class RerankEngine:
    def __init__(self, config_name: str, max_length: int):
        raw = resolve_model_config(config_name)
        self.served_name = raw.get("display_name", raw["model_id"])
        self.cfg = {
            "model_id": raw["model_id"],
            "display_name": self.served_name,
            "loader": raw.get("loader", "mlx_lm"),
            "instruction": raw.get(
                "instruction",
                "Given a web search query, retrieve relevant passages that answer the query",
            ),
            "query_prefix": raw.get("query_prefix"),
            "document_prefix": raw.get("document_prefix"),
            "max_length": max_length,
            "reranker": raw.get("reranker", {"mode": "cross_encoder"}),
        }
        logger.info("Loading reranker %s (loader=%s) ...", self.cfg["model_id"], self.cfg["loader"])
        self.model, self.tokenizer, t, self.use_mlx_embeddings = load_reranker_model(self.cfg)
        logger.info("Reranker ready in %.1fs (use_mlx_embeddings=%s)", t, self.use_mlx_embeddings)

    def score(self, query: str, docs: List[str]) -> List[float]:
        if not docs:
            return []
        with _LOCK:
            scores = rerank_scores(
                [query] * len(docs), docs, self.model, self.tokenizer,
                self.cfg, self.use_mlx_embeddings,
            )
        return [float(s) for s in scores]


def _approx_tokens(texts: List[str]) -> int:
    return max(1, sum(len(t) for t in texts) // 4)


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def create_app(emb: EmbeddingEngine, rnk: RerankEngine) -> FastAPI:
    app = FastAPI(title="Qwen3-0.6B MLX OpenAI server")

    @app.exception_handler(Exception)
    async def _err(request: Request, exc: Exception):
        logger.exception("Unhandled error")
        return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.post("/v1/embeddings", response_model=EmbeddingResponse)
    def embeddings(req: EmbeddingRequest):
        texts = [req.input] if isinstance(req.input, str) else list(req.input)
        is_query = (req.input_type or "").lower() == "query"
        embs = emb.encode(texts, is_query)
        data = [EmbeddingData(index=i, embedding=embs[i].tolist()) for i in range(len(texts))]
        n = _approx_tokens(texts)
        return EmbeddingResponse(
            data=data, model=emb.served_name, usage=EmbeddingUsage(prompt_tokens=n, total_tokens=n)
        )

    @app.post("/v1/rerank", response_model=RerankResponse)
    def rerank(req: RerankRequest):
        docs = [d if isinstance(d, str) else d.text for d in req.documents]
        scores = rnk.score(req.query, docs)
        top_n = req.top_n if req.top_n is not None else len(docs)
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:top_n]
        results = [
            RerankResult(
                index=i,
                relevance_score=scores[i],
                document=RerankDocument(text=docs[i]) if req.return_documents else None,
            )
            for i in order
        ]
        return RerankResponse(results=results)

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [
            {"id": emb.served_name, "object": "model"},
            {"id": rnk.served_name, "object": "model"},
        ]}

    @app.get("/health")
    def health():
        return {"status": "ok", "embedding": emb.served_name, "reranker": rnk.served_name}

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb-config", default="qwen3_embedding_0.6b_4bit_dwq")
    p.add_argument("--rerank-config", default="qwen3_reranker_0.6b_4bit")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=11455)
    p.add_argument("--max-length", type=int, default=512)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    emb = EmbeddingEngine(args.emb_config, args.max_length)
    rnk = RerankEngine(args.rerank_config, args.max_length)
    app = create_app(emb, rnk)
    logger.info("Serving on http://%s:%d  (emb=%s, rerank=%s, max_length=%d)",
                args.host, args.port, emb.served_name, rnk.served_name, args.max_length)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
