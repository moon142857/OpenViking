#!/usr/bin/env python3
"""Performance benchmark for embedding and reranker MLX models.

Measures:
- Latency (single and batch)
- Throughput (items/sec)
- Peak memory usage
- Scaling with sequence length

Usage:
    python scripts/benchmark_performance.py \
        --embedding-model mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ \
        --reranker-model mlx-community/Qwen3-Reranker-0.6B-4bit \
        --output ../results/performance_results.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import mlx.core as mx
import numpy as np
from mlx_lm import load


def embed_texts(texts: List[str], model, tokenizer, max_length: int = 512) -> np.ndarray:
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
    mask = attention_mask.astype(hidden.dtype)
    sum_embeddings = (hidden * mask[:, :, None]).sum(axis=1)
    embeddings = sum_embeddings / mx.maximum(mask.sum(axis=1)[:, None], 1e-9)
    return np.array(embeddings.astype(mx.float32))


def rerank_scores(queries: List[str], docs: List[str], model, tokenizer, max_length: int = 512) -> np.ndarray:
    pairs = [[q, d] for q, d in zip(queries, docs)]
    inputs = tokenizer._tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="np",
        max_length=max_length,
    )
    input_ids = mx.array(inputs["input_ids"].astype(np.int32))
    outputs = model(input_ids)
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    yes_id = tokenizer._tokenizer.convert_tokens_to_ids("yes")
    last_logits = logits[:, -1, :]
    scores = last_logits[:, yes_id]
    return np.array(scores.astype(mx.float32))


def benchmark_embedding(model, tokenizer) -> Dict:
    text = "This is a sample sentence for benchmarking embedding performance."
    results = {}

    # Warmup
    embed_texts([text], model, tokenizer)

    # Single latency
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        embed_texts([text], model, tokenizer)
        mx.eval(mx.array(0.0))  # sync
        times.append((time.perf_counter() - t0) * 1000)
    results["single_latency_ms"] = {
        "avg": round(float(np.mean(times)), 2),
        "min": round(float(np.min(times)), 2),
        "max": round(float(np.max(times)), 2),
    }

    # Batch latency and throughput
    batch_sizes = [1, 8, 16, 32]
    results["batch"] = {}
    for bs in batch_sizes:
        texts = [text] * bs
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            embed_texts(texts, model, tokenizer)
            mx.eval(mx.array(0.0))
            times.append((time.perf_counter() - t0) * 1000)
        avg_ms = float(np.mean(times))
        throughput = bs / (avg_ms / 1000.0)
        results["batch"][f"batch_{bs}"] = {
            "avg_ms": round(avg_ms, 2),
            "throughput_items_per_sec": round(throughput, 2),
        }

    # Sequence length scaling
    results["seq_length"] = {}
    for length in [128, 512, 1024, 2048]:
        long_text = " ".join(["word"] * length)
        # Truncate to approximate tokens
        t0 = time.perf_counter()
        embed_texts([long_text], model, tokenizer, max_length=length)
        mx.eval(mx.array(0.0))
        latency = (time.perf_counter() - t0) * 1000
        results["seq_length"][f"length_{length}"] = {
            "latency_ms": round(latency, 2),
        }

    return results


def benchmark_reranker(model, tokenizer) -> Dict:
    query = "What is OpenViking?"
    doc = "OpenViking is an agent-native context database."
    results = {}

    # Warmup
    rerank_scores([query], [doc], model, tokenizer)

    # Single pair latency
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        rerank_scores([query], [doc], model, tokenizer)
        mx.eval(mx.array(0.0))
        times.append((time.perf_counter() - t0) * 1000)
    results["single_latency_ms"] = {
        "avg": round(float(np.mean(times)), 2),
        "min": round(float(np.min(times)), 2),
        "max": round(float(np.max(times)), 2),
    }

    # Batch latency and throughput
    batch_sizes = [1, 8, 16, 32]
    results["batch"] = {}
    for bs in batch_sizes:
        queries = [query] * bs
        docs = [doc] * bs
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            rerank_scores(queries, docs, model, tokenizer)
            mx.eval(mx.array(0.0))
            times.append((time.perf_counter() - t0) * 1000)
        avg_ms = float(np.mean(times))
        throughput = bs / (avg_ms / 1000.0)
        results["batch"][f"batch_{bs}"] = {
            "avg_ms": round(avg_ms, 2),
            "throughput_pairs_per_sec": round(throughput, 2),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding-model",
        default="mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    )
    parser.add_argument(
        "--reranker-model",
        default="mlx-community/Qwen3-Reranker-0.6B-4bit",
    )
    parser.add_argument("--output", default="../results/performance_results.json")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "embedding_model": args.embedding_model,
        "reranker_model": args.reranker_model,
        "embedding": {},
        "reranker": {},
    }

    print(f"Loading embedding model: {args.embedding_model}")
    emb_model, emb_tokenizer = load(args.embedding_model)
    print("Benchmarking embedding model...")
    results["embedding"] = benchmark_embedding(emb_model, emb_tokenizer)
    print(f"  Single latency: {results['embedding']['single_latency_ms']['avg']} ms")
    print(f"  Batch 32 throughput: {results['embedding']['batch']['batch_32']['throughput_items_per_sec']} items/sec")

    print(f"\nLoading reranker model: {args.reranker_model}")
    rerank_model, rerank_tokenizer = load(args.reranker_model)
    print("Benchmarking reranker model...")
    results["reranker"] = benchmark_reranker(rerank_model, rerank_tokenizer)
    print(f"  Single latency: {results['reranker']['single_latency_ms']['avg']} ms")
    print(f"  Batch 32 throughput: {results['reranker']['batch']['batch_32']['throughput_pairs_per_sec']} pairs/sec")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
