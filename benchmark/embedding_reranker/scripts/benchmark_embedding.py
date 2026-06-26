#!/usr/bin/env python3
"""Embedding model functional benchmark (config-driven).

Evaluates embedding models on:
- Semantic Textual Similarity (Spearman correlation)
- Paraphrase detection (Accuracy / F1)
- Retrieval recall (Recall@k, MRR)
- Clustering (V-measure)

Model configs live in config/models/*.yaml and specify:
  - model_id / local path
  - loader: mlx_lm | mlx_embeddings
  - instruction, query_prefix, document_prefix
  - pooling, normalize

Usage:
    # Use a named config
    python scripts/benchmark_embedding.py --config qwen3_embedding_8b_mxfp8

    # Override model_id/path from command line
    python scripts/benchmark_embedding.py --config qwen3_embedding_8b_mxfp8 \
        --model /path/to/local/model

    # Raw model_id (auto-detects loader)
    python scripts/benchmark_embedding.py --model mlx-community/Qwen3-Embedding-4B-4bit-DWQ
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import mlx.core as mx
import numpy as np
from config.loader import (
    format_document,
    format_query,
    load_model_config,
    resolve_model_config,
)
from mlx_lm import load as mlx_lm_load
from mlx_lm.utils import _download, load_model as _load_model_raw, load_tokenizer
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, v_measure_score
from sklearn.metrics.pairwise import cosine_similarity

try:
    from mlx_embeddings import generate as mlx_embeddings_generate
    from mlx_embeddings import load as mlx_embeddings_load

    HAS_MLX_EMBEDDINGS = True
except ImportError:
    HAS_MLX_EMBEDDINGS = False


def load_jsonl(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def last_token_pool(hidden_states: mx.array, attention_mask: mx.array) -> mx.array:
    """Pool using the last non-padding token, compatible with left/right padding."""
    batch_size, seq_len, _ = hidden_states.shape
    last_token_attn = attention_mask[:, -1]
    left_padded = bool(last_token_attn.sum().item() == batch_size)

    if left_padded:
        return hidden_states[:, -1, :]
    else:
        sequence_lengths = attention_mask.sum(axis=1) - 1
        return hidden_states[mx.arange(batch_size), sequence_lengths, :]


def _embed_with_mlx_lm(
    texts: List[str],
    model,
    tokenizer,
    max_length: int,
    pooling: str,
    normalize: bool,
) -> np.ndarray:
    tokenizer._tokenizer.padding_side = "left"
    inputs = tokenizer._tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="np",
        max_length=max_length,
    )
    input_ids = mx.array(inputs["input_ids"].astype(np.int32))
    attention_mask = mx.array(inputs["attention_mask"].astype(np.float32))

    outputs = model.model(input_ids)
    hidden = outputs[0] if isinstance(outputs, tuple) else outputs

    if pooling == "last":
        embeddings = last_token_pool(hidden, attention_mask)
    elif pooling == "mean":
        mask = attention_mask.astype(hidden.dtype)
        sum_embeddings = (hidden * mask[:, :, None]).sum(axis=1)
        embeddings = sum_embeddings / mx.maximum(mask.sum(axis=1)[:, None], 1e-9)
    else:
        raise ValueError(f"Unknown pooling: {pooling}")

    if normalize:
        embeddings = embeddings / mx.sqrt(
            mx.sum(embeddings * embeddings, axis=1, keepdims=True) + 1e-12
        )

    return np.array(embeddings.astype(mx.float32))


def _embed_with_mlx_embeddings(
    texts: List[str],
    model,
    processor,
    max_length: int,
) -> np.ndarray:
    output = mlx_embeddings_generate(
        model,
        processor,
        texts=texts,
        max_length=max_length,
        padding=True,
        truncation=True,
    )
    return np.array(output.text_embeds.astype(mx.float32))


def load_model(model_id: str, loader: str) -> Tuple[object, object, float]:
    t0 = time.time()
    if loader == "mlx_embeddings":
        if not HAS_MLX_EMBEDDINGS:
            raise RuntimeError(
                "mlx-embeddings is required for this model. "
                "Install it with: pip install mlx-embeddings"
            )
        model, processor = mlx_embeddings_load(model_id)
        return model, processor, time.time() - t0

    if loader != "mlx_lm":
        raise ValueError(f"Unknown loader: {loader}")

    try:
        model, tokenizer = mlx_lm_load(model_id)
    except ValueError as e:
        if "lm_head.weight" in str(e):
            print("  Standard mlx_lm load failed due to missing lm_head (embedding model). "
                  "Retrying with strict=False...")
            local_path = _download(model_id)
            model, config = _load_model_raw(local_path, strict=False)
            tokenizer = load_tokenizer(
                local_path, {}, eos_token_ids=config.get("eos_token_id", None)
            )
        else:
            raise
    return model, tokenizer, time.time() - t0


def embed_texts(
    texts: List[str],
    model,
    tokenizer,
    max_length: int,
    loader: str,
    prefix_template: str,
    instruction: str,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0))

    input_texts = [
        format_query(t, instruction, prefix_template) if prefix_template else t
        for t in texts
    ]

    if loader == "mlx_embeddings":
        return _embed_with_mlx_embeddings(input_texts, model, tokenizer, max_length)

    config = getattr(tokenizer, "_config", None)
    pooling = config.get("pooling", "last") if config else "last"
    normalize = config.get("normalize", True) if config else True
    return _embed_with_mlx_lm(input_texts, model, tokenizer, max_length, pooling, normalize)


def benchmark_sts(dataset_path: Path, model, tokenizer, cfg: Dict) -> Dict:
    data = load_jsonl(dataset_path)
    s1 = [d["sentence1"] for d in data]
    s2 = [d["sentence2"] for d in data]
    labels = np.array([d["score"] for d in data])

    emb1 = embed_texts(s1, model, tokenizer, cfg["max_length"], cfg["loader"], cfg["query_prefix"], cfg["instruction"])
    emb2 = embed_texts(s2, model, tokenizer, cfg["max_length"], cfg["loader"], cfg["query_prefix"], cfg["instruction"])
    preds = cosine_similarity(emb1, emb2).diagonal() * 5.0

    corr, _ = spearmanr(labels, preds)
    return {
        "metric": "spearman_correlation",
        "value": round(float(corr), 4),
        "samples": len(data),
    }


def benchmark_paraphrase(dataset_path: Path, model, tokenizer, cfg: Dict) -> Dict:
    data = load_jsonl(dataset_path)
    s1 = [d["sentence1"] for d in data]
    s2 = [d["sentence2"] for d in data]
    labels = np.array([d["label"] for d in data])

    emb1 = embed_texts(s1, model, tokenizer, cfg["max_length"], cfg["loader"], cfg["query_prefix"], cfg["instruction"])
    emb2 = embed_texts(s2, model, tokenizer, cfg["max_length"], cfg["loader"], cfg["query_prefix"], cfg["instruction"])
    sims = cosine_similarity(emb1, emb2).diagonal()

    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.3, 0.9, 0.02):
        preds = (sims > thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    preds = (sims > best_thresh).astype(int)
    accuracy = float(np.mean(preds == labels))
    return {
        "metric": "paraphrase_detection",
        "accuracy": round(accuracy, 4),
        "f1": round(float(f1_score(labels, preds, zero_division=0)), 4),
        "best_threshold": round(float(best_thresh), 3),
        "samples": len(data),
    }


def benchmark_retrieval(corpus_path: Path, queries_path: Path, model, tokenizer, cfg: Dict) -> Dict:
    corpus = {d["doc_id"]: d["text"] for d in load_jsonl(corpus_path)}
    queries = load_jsonl(queries_path)

    doc_ids = list(corpus.keys())
    doc_texts = [format_document(corpus[did], cfg["document_prefix"]) for did in doc_ids]
    doc_embeddings = embed_texts(
        doc_texts, model, tokenizer, cfg["max_length"], cfg["loader"], cfg["query_prefix"], cfg["instruction"]
    )

    results = {"recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0}
    total = 0

    for q in queries:
        q_text = format_query(q["text"], cfg["instruction"], cfg["query_prefix"])
        q_emb = embed_texts(
            [q_text], model, tokenizer, cfg["max_length"], cfg["loader"], cfg["query_prefix"], cfg["instruction"]
        )
        scores = cosine_similarity(q_emb, doc_embeddings)[0]
        ranked = [doc_ids[i] for i in np.argsort(scores)[::-1]]
        relevant = set(q["relevant_docs"])

        total += 1
        for k in [1, 5, 10]:
            if any(r in ranked[:k] for r in relevant):
                results[f"recall@{k}"] += 1

        for rank, did in enumerate(ranked, start=1):
            if did in relevant:
                results["mrr"] += 1.0 / rank
                break

    return {k: round(v / total, 4) if total else 0.0 for k, v in results.items()} | {"samples": total}


def benchmark_clustering(dataset_path: Path, model, tokenizer, cfg: Dict) -> Dict:
    data = load_jsonl(dataset_path)
    texts = [format_query(d["text"], cfg["instruction"], cfg["query_prefix"]) for d in data]
    labels = np.array([d["cluster"] for d in data])

    embeddings = embed_texts(texts, model, tokenizer, cfg["max_length"], cfg["loader"], cfg["query_prefix"], cfg["instruction"])
    unique_labels = list(set(labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    centers = np.zeros((len(unique_labels), embeddings.shape[1]))
    counts = np.zeros(len(unique_labels))
    for emb, lab in zip(embeddings, labels):
        idx = label_to_idx[lab]
        centers[idx] += emb
        counts[idx] += 1
    centers /= np.maximum(counts[:, None], 1)

    sims = cosine_similarity(embeddings, centers)
    pred_labels = np.array([unique_labels[i] for i in sims.argmax(axis=1)])

    v_measure = v_measure_score(labels, pred_labels)
    return {
        "metric": "clustering_v_measure",
        "value": round(float(v_measure), 4),
        "samples": len(data),
        "num_clusters": len(unique_labels),
    }


def build_config(args) -> Dict:
    """Build runtime config from YAML config + command-line overrides."""
    if args.config:
        raw = resolve_model_config(args.config)
    else:
        raw = resolve_model_config(args.model)

    cfg = {
        "model_id": args.model if args.model else raw["model_id"],
        "display_name": raw.get("display_name", raw["model_id"]),
        "loader": raw.get("loader", "mlx_lm"),
        "instruction": raw.get("instruction", "Given a web search query, retrieve relevant passages that answer the query"),
        "query_prefix": raw.get("query_prefix", "Instruct: {instruction}\\nQuery:{text}"),
        "document_prefix": raw.get("document_prefix", "{text}"),
        "max_length": args.max_length,
        "embedding": raw.get("embedding", {"pooling": "last", "normalize": True}),
        "notes": raw.get("notes", ""),
    }

    if args.instruction is not None:
        cfg["instruction"] = args.instruction
    if args.pooling is not None:
        cfg["embedding"]["pooling"] = args.pooling
    if args.normalize is not None:
        cfg["embedding"]["normalize"] = args.normalize

    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="Model config name (e.g. qwen3_embedding_8b_mxfp8) or path to a YAML file",
    )
    parser.add_argument(
        "--model",
        help="Override model_id or local path from the config",
    )
    parser.add_argument("--datasets", default="../datasets", help="Datasets directory")
    parser.add_argument("--output", default="../results/embedding_results.json")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--instruction", help="Override task instruction")
    parser.add_argument("--pooling", choices=["last", "mean"], help="Override pooling (mlx_lm path only)")
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        help="Override L2 normalization (mlx_lm path only)",
    )
    args = parser.parse_args()

    if not args.config and not args.model:
        parser.error("One of --config or --model is required.")

    cfg = build_config(args)
    print(f"Loading embedding model: {cfg['model_id']}")
    print(f"  config={args.config or 'auto'}, loader={cfg['loader']}, "
          f"pooling={cfg['embedding']['pooling']}, normalize={cfg['embedding']['normalize']}, "
          f"instruction={cfg['instruction']!r}")
    if cfg["notes"]:
        print(f"  note: {cfg['notes']}")

    model, tokenizer, load_time = load_model(cfg["model_id"], cfg["loader"])
    # Attach embedding config to tokenizer so embed_texts can read it (mlx_lm path)
    tokenizer._config = cfg["embedding"]
    print(f"Model loaded in {load_time:.2f}s")

    ds_dir = Path(args.datasets)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "model": cfg["model_id"],
        "display_name": cfg["display_name"],
        "load_time_s": round(load_time, 2),
        "max_length": cfg["max_length"],
        "loader": cfg["loader"],
        "pooling": cfg["embedding"]["pooling"] if cfg["loader"] != "mlx_embeddings" else "last",
        "normalize": cfg["embedding"]["normalize"] if cfg["loader"] != "mlx_embeddings" else True,
        "instruction": cfg["instruction"],
        "tasks": {},
    }

    print("\n[STS] Evaluating semantic textual similarity...")
    results["tasks"]["sts"] = benchmark_sts(ds_dir / "sts_sample.jsonl", model, tokenizer, cfg)
    print(f"  Spearman correlation: {results['tasks']['sts']['value']}")

    print("\n[Paraphrase] Evaluating paraphrase detection...")
    results["tasks"]["paraphrase"] = benchmark_paraphrase(
        ds_dir / "paraphrase_sample.jsonl", model, tokenizer, cfg
    )
    print(f"  Accuracy: {results['tasks']['paraphrase']['accuracy']}")
    print(f"  F1: {results['tasks']['paraphrase']['f1']}")

    print("\n[Retrieval] Evaluating retrieval recall...")
    results["tasks"]["retrieval"] = benchmark_retrieval(
        ds_dir / "retrieval_corpus.jsonl",
        ds_dir / "retrieval_queries.jsonl",
        model,
        tokenizer,
        cfg,
    )
    print(f"  Recall@1: {results['tasks']['retrieval']['recall@1']}")
    print(f"  Recall@5: {results['tasks']['retrieval']['recall@5']}")
    print(f"  MRR: {results['tasks']['retrieval']['mrr']}")

    print("\n[Clustering] Evaluating clustering...")
    results["tasks"]["clustering"] = benchmark_clustering(
        ds_dir / "clustering_sample.jsonl", model, tokenizer, cfg
    )
    print(f"  V-measure: {results['tasks']['clustering']['value']}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
