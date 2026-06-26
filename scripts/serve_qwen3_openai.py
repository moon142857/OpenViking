#!/usr/bin/env python3
"""Lightweight OpenAI-compatible server for Qwen3 embedding/reranker models.

Runs on Apple Silicon (or any torch device) using transformers.
Designed as a drop-in replacement for vLLM --task embed / --task score when
vllm-metal cannot be installed.

Examples:
  # Embedding
  python scripts/serve_qwen3_openai.py --task embed \\
      --model mlx-community/Qwen3-Embedding-0.6B-8bit --port 8000

  # Reranker
  python scripts/serve_qwen3_openai.py --task rerank \\
      --model Qwen/Qwen3-Reranker-0.6B --port 8001

Environment:
  HF_ENDPOINT    HuggingFace mirror, e.g. https://hf-mirror.com
"""

import argparse
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, List, Optional, Union

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    encoding_format: str = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


class RerankDocument(BaseModel):
    text: str


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: List[Union[str, RerankDocument]]
    top_n: Optional[int] = None
    return_documents: bool = True


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: Optional[Union[str, RerankDocument]] = None


class RerankResponse(BaseModel):
    results: List[RerankResult]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

class Embedder:
    def __init__(self, model_name: str, device: str, served_name: str):
        self.model_name = served_name or model_name
        logger.info("Loading embedding model %s on %s ...", model_name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device == "mps" else torch.float32,
        )
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.hidden_size = self.model.config.hidden_size
        logger.info("Embedding model ready (dim=%d)", self.hidden_size)

    @torch.inference_mode()
    def encode(self, texts: List[str]) -> tuple[torch.Tensor, int]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        outputs = self.model(**encoded, output_hidden_states=False)
        # Last-token pooling using attention mask.
        hidden = outputs.last_hidden_state  # [batch, seq, dim]
        attention_mask = encoded["attention_mask"]
        seq_indices = attention_mask.sum(dim=1) - 1  # last real token index
        embeddings = hidden[torch.arange(hidden.size(0)), seq_indices]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.float(), int(attention_mask.sum())


class Reranker:
    # Qwen3 reranker uses the LM head to choose between "yes" / "no" tokens.
    TRUE_TOKEN_ID = 9693
    FALSE_TOKEN_ID = 2152

    def __init__(self, model_name: str, device: str, served_name: str):
        self.model_name = served_name or model_name
        logger.info("Loading reranker model %s on %s ...", model_name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device == "mps" else torch.float32,
        )
        self.model.to(device)
        self.model.eval()
        self.device = device
        logger.info("Reranker model ready")

    @torch.inference_mode()
    def score(self, query: str, docs: List[str]) -> List[float]:
        # Build the same chat prompt CrossEncoder uses. The model answers
        # "yes"/"no" based on the provided query, document and instruction.
        texts = []
        for doc in docs:
            messages = [
                {"role": "system", "content": "Given a web search query, retrieve relevant passages that answer the query"},
                {"role": "query", "content": query},
                {"role": "document", "content": doc},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(prompt)

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        outputs = self.model(**encoded)
        logits = outputs.logits[:, -1, :]  # logits for the next token after the prompt
        yes = logits[:, self.TRUE_TOKEN_ID]
        no = logits[:, self.FALSE_TOKEN_ID]
        # 0-1 probability score; mirrors sentence-transformers with Sigmoid activation.
        scores = torch.sigmoid(yes - no).float().tolist()
        return scores


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

state: dict[str, Any] = {"engine": None}


def create_app(task: str, engine: Any) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["engine"] = engine
        yield
        state["engine"] = None

    app = FastAPI(title=f"Qwen3 {task} server", lifespan=lifespan)

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error")
        return JSONResponse(status_code=500, content={"error": str(exc)})

    if task == "embed":
        @app.post("/v1/embeddings", response_model=EmbeddingResponse)
        async def embeddings(req: EmbeddingRequest):
            engine: Embedder = state["engine"]
            if isinstance(req.input, str):
                texts = [req.input]
            else:
                texts = req.input
            embs, n_tokens = engine.encode(texts)
            data = [
                EmbeddingData(
                    index=i,
                    embedding=embs[i].tolist(),
                )
                for i in range(len(texts))
            ]
            return EmbeddingResponse(
                data=data,
                model=engine.model_name,
                usage=EmbeddingUsage(prompt_tokens=n_tokens, total_tokens=n_tokens),
            )

        @app.get("/v1/models")
        async def list_models():
            engine: Embedder = state["engine"]
            return {
                "object": "list",
                "data": [{"id": engine.model_name, "object": "model"}],
            }

    elif task == "rerank":
        @app.post("/v1/rerank", response_model=RerankResponse)
        async def rerank(req: RerankRequest):
            engine: Reranker = state["engine"]
            docs: List[str] = []
            for d in req.documents:
                if isinstance(d, str):
                    docs.append(d)
                else:
                    docs.append(d.text)

            scores = engine.score(req.query, docs)
            top_n = req.top_n if req.top_n is not None else len(docs)
            indexed = sorted(
                [{"index": i, "score": s, "doc": docs[i]} for i, s in enumerate(scores)],
                key=lambda x: x["score"],
                reverse=True,
            )[:top_n]

            results = []
            for item in indexed:
                doc_field = RerankDocument(text=item["doc"]) if req.return_documents else None
                results.append(
                    RerankResult(
                        index=item["index"],
                        relevance_score=item["score"],
                        document=doc_field,
                    )
                )
            return RerankResponse(results=results)

        @app.get("/v1/models")
        async def list_models():
            engine: Reranker = state["engine"]
            return {
                "object": "list",
                "data": [{"id": engine.model_name, "object": "model"}],
            }

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["embed", "rerank"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--hf-endpoint", default=os.getenv("HF_ENDPOINT"))
    args = parser.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
        logger.info("Using HF_ENDPOINT=%s", args.hf_endpoint)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.device is None:
        args.device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info("Using device: %s", args.device)

    if args.task == "embed":
        engine = Embedder(args.model, args.device, args.served_model_name)
    else:
        engine = Reranker(args.model, args.device, args.served_model_name)

    app = create_app(args.task, engine)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
