#!/usr/bin/env python3
"""Embedding model functional benchmark.

Evaluates embedding models on:
- Semantic Textual Similarity (Spearman correlation)
- Paraphrase detection (Accuracy / F1)
- Retrieval recall (Recall@k, MRR)
- Clustering (V-measure)

Usage:
    python scripts/benchmark_embedding.py \
        --model mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ \
        --datasets ../datasets \
        --output ../results/embedding_results.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import mlx.core as mx
import numpy as np
from mlx_lm import load
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, v_measure_score
from sklearn.metrics.pairwise import cosine_similarity


def load_jsonl(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def embed_texts(texts: List[str], model, tokenizer, max_length: int = 512) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0))
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


def benchmark_sts(dataset_path: Path, model, tokenizer) -> Dict:
    data = load_jsonl(dataset_path)
    s1 = [d["sentence1"] for d in data]
    s2 = [d["sentence2"] for d in data]
    labels = np.array([d["score"] for d in data])

    emb1 = embed_texts(s1, model, tokenizer)
    emb2 = embed_texts(s2, model, tokenizer)
    preds = cosine_similarity(emb1, emb2).diagonal() * 5.0  # scale to 0-5

    corr, _ = spearmanr(labels, preds)
    return {
        "metric": "spearman_correlation",
        "value": round(float(corr), 4),
        "samples": len(data),
    }


def benchmark_paraphrase(dataset_path: Path, model, tokenizer) -> Dict:
    data = load_jsonl(dataset_path)
    s1 = [d["sentence1"] for d in data]
    s2 = [d["sentence2"] for d in data]
    labels = np.array([d["label"] for d in data])

    emb1 = embed_texts(s1, model, tokenizer)
    emb2 = embed_texts(s2, model, tokenizer)
    sims = cosine_similarity(emb1, emb2).diagonal()

    # Find optimal threshold
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


def benchmark_retrieval(
    corpus_path: Path, queries_path: Path, model, tokenizer
) -> Dict:
    corpus = {d["doc_id"]: d["text"] for d in load_jsonl(corpus_path)}
    queries = load_jsonl(queries_path)

    doc_ids = list(corpus.keys())
    doc_texts = [corpus[did] for did in doc_ids]
    doc_embeddings = embed_texts(doc_texts, model, tokenizer)

    results = {"recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0}
    total = 0

    for q in queries:
        q_emb = embed_texts([q["text"]], model, tokenizer)
        scores = cosine_similarity(q_emb, doc_embeddings)[0]
        ranked = [doc_ids[i] for i in np.argsort(scores)[::-1]]
        relevant = set(q["relevant_docs"])

        total += 1
        for k in [1, 5, 10]:
            if any(r in ranked[:k] for r in relevant):
                results[f"recall@{k}"] += 1

        # MRR
        for rank, did in enumerate(ranked, start=1):
            if did in relevant:
                results["mrr"] += 1.0 / rank
                break

    return {
        k: round(v / total, 4) if total else 0.0
        for k, v in results.items()
    } | {"samples": total}


def benchmark_clustering(dataset_path: Path, model, tokenizer) -> Dict:
    data = load_jsonl(dataset_path)
    texts = [d["text"] for d in data]
    labels = np.array([d["cluster"] for d in data])

    embeddings = embed_texts(texts, model, tokenizer)
    # Simple argmax over cosine similarity to cluster centers
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        help="Embedding model ID or local path",
    )
    parser.add_argument("--datasets", default="../datasets", help="Datasets directory")
    parser.add_argument("--output", default="../results/embedding_results.json")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    print(f"Loading embedding model: {args.model}")
    t0 = time.time()
    model, tokenizer = load(args.model)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.2f}s")

    ds_dir = Path(args.datasets)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "model": args.model,
        "load_time_s": round(load_time, 2),
        "max_length": args.max_length,
        "tasks": {},
    }

    print("\n[STS] Evaluating semantic textual similarity...")
    results["tasks"]["sts"] = benchmark_sts(ds_dir / "sts_sample.jsonl", model, tokenizer)
    print(f"  Spearman correlation: {results['tasks']['sts']['value']}")

    print("\n[Paraphrase] Evaluating paraphrase detection...")
    results["tasks"]["paraphrase"] = benchmark_paraphrase(
        ds_dir / "paraphrase_sample.jsonl", model, tokenizer
    )
    print(f"  Accuracy: {results['tasks']['paraphrase']['accuracy']}")
    print(f"  F1: {results['tasks']['paraphrase']['f1']}")

    print("\n[Retrieval] Evaluating retrieval recall...")
    results["tasks"]["retrieval"] = benchmark_retrieval(
        ds_dir / "retrieval_corpus.jsonl",
        ds_dir / "retrieval_queries.jsonl",
        model,
        tokenizer,
    )
    print(f"  Recall@1: {results['tasks']['retrieval']['recall@1']}")
    print(f"  Recall@5: {results['tasks']['retrieval']['recall@5']}")
    print(f"  MRR: {results['tasks']['retrieval']['mrr']}")

    print("\n[Clustering] Evaluating clustering...")
    results["tasks"]["clustering"] = benchmark_clustering(
        ds_dir / "clustering_sample.jsonl", model, tokenizer
    )
    print(f"  V-measure: {results['tasks']['clustering']['value']}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
