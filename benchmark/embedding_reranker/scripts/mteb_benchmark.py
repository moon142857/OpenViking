#!/usr/bin/env python3
"""Lightweight MTEB benchmark for MLX embedding/reranker models.

Usage:
    # Embedding model
    python scripts/mteb_benchmark.py --config qwen3_embedding_0.6b_4bit_dwq --tasks STSBenchmark,SciFact

    # Reranker model (bi-encoder similarity, matching official mlx-embeddings usage)
    python scripts/mteb_benchmark.py --config qwen3_reranker_0.6b_mxfp8 --tasks AskUbuntuDupQuestions,SciDocsRR
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import mlx.core as mx
import mteb
import numpy as np
from config.loader import (
    format_document,
    format_query,
    resolve_model_config,
)
from mteb import get_task
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType as MTEBPromptType
from sklearn.metrics.pairwise import cosine_similarity

try:
    from mlx_embeddings import generate as mlx_embeddings_generate
    from mlx_embeddings import load as mlx_embeddings_load

    HAS_MLX_EMBEDDINGS = True
except ImportError:
    HAS_MLX_EMBEDDINGS = False

try:
    from mlx_lm import load as mlx_lm_load
    from mlx_lm.utils import _download, load_model as _load_model_raw, load_tokenizer

    HAS_MLX_LM = True
except ImportError:
    HAS_MLX_LM = False


DEFAULT_EMBEDDING_TASKS = [
    "STSBenchmark",
    "SICK-R",
    "SciFact",
]

DEFAULT_RERANKER_TASKS = [
    "AskUbuntuDupQuestions",
    "SciDocsRR",
]


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _normalize_embeddings(embs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / (norms + 1e-12)


class MLXEmbeddingBackend:
    """Low-level MLX embedding backend shared by encoder and bi-encoder reranker."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self._load()

    def _load(self):
        loader = self.cfg["loader"]
        model_id = self.cfg["model_id"]
        print(f"Loading MLX embedding backend: {model_id} (loader={loader})")
        t0 = time.time()
        if loader == "mlx_embeddings":
            if not HAS_MLX_EMBEDDINGS:
                raise RuntimeError("pip install mlx-embeddings")
            self.model, self.tokenizer = mlx_embeddings_load(model_id)
        elif loader == "mlx_lm":
            if not HAS_MLX_LM:
                raise RuntimeError("pip install mlx-lm")
            try:
                self.model, self.tokenizer = mlx_lm_load(model_id)
            except ValueError as e:
                if "lm_head.weight" in str(e):
                    local_path = _download(model_id)
                    self.model, config = _load_model_raw(local_path, strict=False)
                    self.tokenizer = load_tokenizer(
                        local_path, {}, eos_token_ids=config.get("eos_token_id", None)
                    )
                else:
                    raise
        else:
            raise ValueError(f"Unknown loader: {loader}")
        print(f"  Loaded in {time.time() - t0:.2f}s")

    def _embed_with_mlx_lm(self, texts: List[str], max_length: int) -> np.ndarray:
        self.tokenizer._tokenizer.padding_side = "left"
        inputs = self.tokenizer._tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="np",
            max_length=max_length,
        )
        input_ids = mx.array(inputs["input_ids"].astype(np.int32))
        attention_mask = mx.array(inputs["attention_mask"].astype(np.float32))

        outputs = self.model.model(input_ids)
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        batch_size = hidden.shape[0]
        sequence_lengths = (attention_mask.sum(axis=1) - 1).astype(mx.int32)
        embeddings = hidden[mx.arange(batch_size), sequence_lengths, :]
        embeddings = embeddings / mx.sqrt(
            mx.sum(embeddings * embeddings, axis=1, keepdims=True) + 1e-12
        )
        return np.array(embeddings.astype(mx.float32))

    def _embed_with_mlx_embeddings(self, texts: List[str], max_length: int) -> np.ndarray:
        output = mlx_embeddings_generate(
            self.model,
            self.tokenizer,
            texts=texts,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return np.array(output.text_embeds.astype(mx.float32))

    def encode_sentences(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        max_length = self.cfg.get("max_length", 512)
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            if self.cfg["loader"] == "mlx_embeddings":
                embs = self._embed_with_mlx_embeddings(batch, max_length)
            else:
                embs = self._embed_with_mlx_lm(batch, max_length)
            all_embs.append(embs)
        return np.vstack(all_embs)


class MLXEncoderForMTEB:
    """MTEB-compatible encoder wrapping an MLX embedding model."""

    def __init__(self, cfg: Dict, model_meta: ModelMeta | None = None):
        self.cfg = cfg
        self.backend = MLXEmbeddingBackend(cfg)
        self._model_meta = model_meta

    def encode(
        self,
        inputs,
        *,
        task_metadata=None,
        hf_split=None,
        hf_subset=None,
        prompt_type=None,
        **kwargs: Any,
    ) -> np.ndarray:
        batch_size = kwargs.get("batch_size", 32)
        texts = []
        for batch in inputs:
            # MTEB text batches are TypedDicts with a "text" key (list[str])
            if isinstance(batch, dict):
                texts.extend(batch.get("text", []))
            else:
                texts.extend(getattr(batch, "text", []))

        instruction = self.cfg.get("instruction", "")
        query_prefix = self.cfg.get("query_prefix")
        document_prefix = self.cfg.get("document_prefix")

        # Determine which prefix to apply based on prompt_type.
        # STS tasks typically perform best without a retrieval-style instruction,
        # so we only apply the query instruction when prompt_type indicates a query.
        if prompt_type is not None:
            if isinstance(prompt_type, str):
                prompt_type = MTEBPromptType(prompt_type)
            is_query = prompt_type == MTEBPromptType.query
        else:
            is_query = kwargs.get("apply_instruction", False)

        prefix = query_prefix if is_query else document_prefix
        input_texts = [format_query(t, instruction, prefix) if prefix else t for t in texts]
        embs = self.backend.encode_sentences(input_texts, batch_size=batch_size)
        return _normalize_embeddings(embs)

    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        return cosine_similarity(embeddings1, embeddings2)

    def similarity_pairwise(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        sim_matrix = cosine_similarity(embeddings1, embeddings2)
        return np.diag(sim_matrix)

    # mteb sets this attribute after loading the model
    mteb_model_meta: ModelMeta | None = None


class MLXBiEncoderRankerForMTEB:
    """MTEB-compatible bi-encoder reranker (embedding similarity).

    This matches the official mlx-embeddings usage for Qwen3-Reranker-*-mxfp8 models.
    """

    def __init__(self, cfg: Dict, model_meta: ModelMeta | None = None):
        self.cfg = cfg
        self.backend = MLXEmbeddingBackend(cfg)
        self._model_meta = model_meta

    def _texts_from_loader(self, inputs) -> List[str]:
        texts = []
        for batch in inputs:
            if isinstance(batch, dict):
                texts.extend(batch.get("text", []))
            else:
                texts.extend(getattr(batch, "text", []))
        return texts

    def predict(
        self,
        inputs1,
        inputs2,
        *,
        task_metadata=None,
        hf_split=None,
        hf_subset=None,
        prompt_type=None,
        **kwargs: Any,
    ) -> np.ndarray:
        batch_size = kwargs.get("batch_size", 32)
        queries = self._texts_from_loader(inputs1)
        docs = self._texts_from_loader(inputs2)

        instruction = self.cfg.get("instruction", "")
        q_prefix = self.cfg.get("query_prefix")
        d_prefix = self.cfg.get("document_prefix")

        q_texts = [format_query(q, instruction, q_prefix) for q in queries]
        d_texts = [format_document(d, d_prefix) for d in docs]

        q_embs = _normalize_embeddings(self.backend.encode_sentences(q_texts, batch_size=batch_size))
        d_embs = _normalize_embeddings(self.backend.encode_sentences(d_texts, batch_size=batch_size))
        # MTEB's SearchCrossEncoderWrapper expects a 1-D score per (query, doc) pair.
        return np.sum(q_embs * d_embs, axis=1)

    # mteb sets this attribute after loading the model
    mteb_model_meta: ModelMeta | None = None


def _extract_main_score(task_result) -> float | None:
    if hasattr(task_result, "scores") and task_result.scores:
        scores = task_result.scores
        for split, split_scores in scores.items():
            if isinstance(split_scores, dict):
                for subset, metrics_list in split_scores.items():
                    if metrics_list and isinstance(metrics_list, list):
                        return metrics_list[0].get("main_score")
    if hasattr(task_result, "main_score"):
        return task_result.main_score
    return None


def _parse_mteb_result(result) -> dict:
    """Parse mteb.evaluate() result into a JSON-serializable summary."""
    summary = {}
    if hasattr(result, "to_dict"):
        data = result.to_dict()
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    summary[key] = {
                        "main_score": value.get("main_score"),
                        "scores": value,
                    }
                else:
                    summary[key] = {
                        "main_score": _extract_main_score(value),
                        "scores": str(value),
                    }
            return summary
    try:
        for res in result:
            name = getattr(res, "task_name", getattr(res, "task", str(res)))
            summary[name] = {
                "main_score": _extract_main_score(res),
                "scores": getattr(res, "scores", str(res)),
            }
    except TypeError:
        summary["result"] = {
            "main_score": getattr(result, "main_score", None),
            "scores": getattr(result, "scores", str(result)),
        }
    return summary


def build_model_meta(cfg: Dict, is_reranker: bool) -> ModelMeta:
    def loader(model_name: str, revision: str | None = None, **kwargs):
        if is_reranker:
            return MLXBiEncoderRankerForMTEB(cfg, model_meta=meta)
        return MLXEncoderForMTEB(cfg, model_meta=meta)

    meta = ModelMeta(
        name=cfg["model_id"],
        loader=loader,
        loader_kwargs={},
        languages=["eng-Latn"],
        open_weights=True,
        revision="main",
        release_date=None,
        n_parameters=None,
        n_active_parameters_override=None,
        n_embedding_parameters=None,
        memory_usage_mb=None,
        max_tokens=cfg.get("max_length", 512),
        embed_dim=None,
        license="apache-2.0",
        framework=["PyTorch"],
        public_training_code=None,
        public_training_data=None,
        reference=None,
        similarity_fn_name="cosine",
        use_instructions=True,
        training_datasets=set(),
        adapted_from=None,
        superseded_by=None,
        modalities=["text"],
        model_type=["dense"],
        citation=None,
        contacts=None,
        experiment_kwargs={},
        output_dtypes="float16",
        extra_requirements_groups=[],
    )
    return meta


def run_embedding_benchmark(cfg: Dict, tasks: List[str], output_path: Path):
    meta = build_model_meta(cfg, is_reranker=False)
    task_objs = [get_task(t) for t in tasks]
    results = mteb.evaluate(
        meta,
        task_objs,
        encode_kwargs={"batch_size": 128},
        raise_error=False,
        show_progress_bar=True,
        overwrite_strategy="always",
    )
    return _parse_mteb_result(results)


def run_reranker_benchmark(cfg: Dict, tasks: List[str], output_path: Path):
    meta = build_model_meta(cfg, is_reranker=True)
    task_objs = [get_task(t) for t in tasks]
    results = mteb.evaluate(
        meta,
        task_objs,
        encode_kwargs={"batch_size": 128},
        raise_error=False,
        show_progress_bar=True,
        overwrite_strategy="always",
    )
    return _parse_mteb_result(results)


def build_config(args) -> tuple[Dict, bool]:
    raw = resolve_model_config(args.config)
    is_reranker = "reranker" in raw
    cfg = {
        "model_id": args.model if args.model else raw["model_id"],
        "display_name": raw.get("display_name", raw["model_id"]),
        "loader": raw.get("loader", "mlx_lm"),
        "instruction": raw.get(
            "instruction",
            "Given a web search query, retrieve relevant passages that answer the query",
        ),
        "query_prefix": raw.get("query_prefix"),
        "document_prefix": raw.get("document_prefix"),
        "max_length": args.max_length,
        "reranker": raw.get("reranker", {"mode": "cross_encoder"}),
    }
    return cfg, is_reranker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Model config name (e.g. qwen3_embedding_0.6b_4bit_dwq) or YAML path",
    )
    parser.add_argument("--model", help="Override model_id or local path")
    parser.add_argument(
        "--tasks",
        help="Comma-separated MTEB task names (default: lightweight core set)",
    )
    parser.add_argument("--output", default="../results/mteb_results.json")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    cfg, is_reranker = build_config(args)

    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")]
    elif is_reranker:
        tasks = DEFAULT_RERANKER_TASKS
    else:
        tasks = DEFAULT_EMBEDDING_TASKS

    print(f"Running MTEB tasks: {tasks}")
    print(f"Model: {cfg['model_id']} (reranker={is_reranker})")
    output_path = Path(args.output)

    t0 = time.time()
    if is_reranker:
        summary = run_reranker_benchmark(cfg, tasks, output_path)
    else:
        summary = run_embedding_benchmark(cfg, tasks, output_path)
    elapsed = time.time() - t0

    result = {
        "model": cfg["model_id"],
        "display_name": cfg["display_name"],
        "tasks": tasks,
        "elapsed_s": round(elapsed, 2),
        "results": summary,
    }
    save_json(output_path, result)
    print(f"\nMTEB benchmark complete in {elapsed:.2f}s")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
