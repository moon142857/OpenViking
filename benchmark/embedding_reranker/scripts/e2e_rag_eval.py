#!/usr/bin/env python3
"""End-to-end RAG evaluation: Embedding retrieval + Reranker reranking.

Usage:
    # Balanced suite on SciFact
    export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python scripts/e2e_rag_eval.py \
        --embedding-config qwen3_embedding_0.6b_4bit_dwq \
        --reranker-config qwen3_reranker_0.6b_4bit \
        --task SciFact \
        --max-length 256 \
        --output results/e2e_rag_scifact_0.6b_4bit.json

    # Use bi-encoder reranker (mxfp8)
    python scripts/e2e_rag_eval.py \
        --embedding-config qwen3_embedding_0.6b_mxfp8 \
        --reranker-config qwen3_reranker_8b_mxfp8 \
        --task SciFact \
        --max-length 256 \
        --output results/e2e_rag_scifact_mxfp8.json
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from config.loader import resolve_model_config
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity

from benchmark_embedding import load_model as load_embedding_model
from benchmark_reranker import load_model as load_reranker_model, rerank_scores


def load_mteb_retrieval_data(task_name: str, max_length: int = None):
    """Load corpus, queries, and qrels from MTEB cached datasets."""
    corpus_ds = load_dataset(f"mteb/{task_name.lower()}", "corpus", split="corpus")
    queries_ds = load_dataset(f"mteb/{task_name.lower()}", "queries", split="queries")
    qrels_ds = load_dataset(f"mteb/{task_name.lower()}", "default", split="test")

    corpus = {}
    for doc in corpus_ds:
        did = doc["_id"]
        title = doc.get("title", "")
        text = doc.get("text", "")
        corpus[did] = f"{title}\n\n{text}".strip() if title else text

    queries = {q["_id"]: q["text"] for q in queries_ds}

    qrels = defaultdict(dict)
    for item in qrels_ds:
        qid = item["query-id"]
        did = item["corpus-id"]
        score = float(item["score"])
        qrels[qid][did] = score

    return corpus, queries, dict(qrels)


def encode_corpus(corpus: Dict[str, str], model, tokenizer, cfg: Dict, batch_size: int = 32) -> Tuple[List[str], np.ndarray]:
    """Encode all corpus documents."""
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[did] for did in doc_ids]

    from benchmark_embedding import embed_texts

    max_length = cfg.get("max_length", 512)
    all_embs = []
    n_batches = (len(doc_texts) + batch_size - 1) // batch_size
    t0 = time.time()
    for i in range(0, len(doc_texts), batch_size):
        batch = doc_texts[i : i + batch_size]
        bt0 = time.time()
        embs = embed_texts(
            batch, model, tokenizer, max_length, cfg["loader"],
            cfg.get("document_prefix", "{text}"), cfg.get("instruction", "")
        )
        all_embs.append(embs)
        if (i // batch_size + 1) % 10 == 0 or i // batch_size + 1 == n_batches:
            print(f"  encoded {min(i + batch_size, len(doc_texts))}/{len(doc_texts)} docs "
                  f"({time.time() - t0:.1f}s elapsed, {time.time() - bt0:.2f}s/batch)")

    return doc_ids, np.vstack(all_embs)


def encode_queries(queries: Dict[str, str], model, tokenizer, cfg: Dict, batch_size: int = 32) -> Dict[str, np.ndarray]:
    """Encode all queries."""
    from benchmark_embedding import embed_texts

    max_length = cfg.get("max_length", 512)
    qids = list(queries.keys())
    all_embs = {}
    n_batches = (len(qids) + batch_size - 1) // batch_size
    t0 = time.time()

    for i in range(0, len(qids), batch_size):
        batch_qids = qids[i : i + batch_size]
        batch_texts = [queries[qid] for qid in batch_qids]
        embs = embed_texts(
            batch_texts, model, tokenizer, max_length, cfg["loader"],
            cfg.get("query_prefix", "Instruct: {instruction}\\nQuery:{text}"), cfg.get("instruction", "")
        )
        for qid, emb in zip(batch_qids, embs):
            all_embs[qid] = emb
        if (i // batch_size + 1) % 10 == 0 or i // batch_size + 1 == n_batches:
            print(f"  encoded {min(i + batch_size, len(qids))}/{len(qids)} queries "
                  f"({time.time() - t0:.1f}s elapsed)")

    return all_embs


def retrieve_top_k(
    query_emb: np.ndarray,
    doc_ids: List[str],
    doc_embeddings: np.ndarray,
    k: int,
) -> List[Tuple[str, float]]:
    """Return top-k (doc_id, score) by cosine similarity."""
    scores = cosine_similarity(query_emb.reshape(1, -1), doc_embeddings)[0]
    top_idx = np.argsort(scores)[::-1][:k]
    return [(doc_ids[idx], float(scores[idx])) for idx in top_idx]


def rerank_top_k(
    query: str,
    candidates: List[Tuple[str, float]],
    corpus: Dict[str, str],
    model,
    tokenizer,
    cfg: Dict,
    use_mlx_embeddings: bool,
    k: int,
) -> List[Tuple[str, float]]:
    """Rerank candidates with the reranker model (single query)."""
    docs = [corpus[did] for did, _ in candidates]
    queries = [query] * len(docs)
    scores = rerank_scores(queries, docs, model, tokenizer, cfg, use_mlx_embeddings)

    sorted_idx = np.argsort(scores)[::-1][:k]
    return [(candidates[i][0], float(scores[i])) for i in sorted_idx]


def rerank_queries(
    query_candidates: Dict[str, List[Tuple[str, float]]],
    queries: Dict[str, str],
    corpus: Dict[str, str],
    model,
    tokenizer,
    cfg: Dict,
    use_mlx_embeddings: bool,
    k: int,
    batch_queries: int = 4,
) -> Dict[str, List[Tuple[str, float]]]:
    """Rerank candidates for multiple queries in larger batches.

    Processes `batch_queries` queries at a time, flattening all their candidates
    into one reranker batch. This is much faster than one query per batch.
    """
    results = {}
    qids = list(query_candidates.keys())
    t0 = time.time()
    n_batches = (len(qids) + batch_queries - 1) // batch_queries

    for b_idx in range(n_batches):
        batch_qids = qids[b_idx * batch_queries : (b_idx + 1) * batch_queries]
        flat_queries = []
        flat_candidates = []
        offsets = {}
        offset = 0
        for qid in batch_qids:
            cands = query_candidates[qid]
            flat_queries.extend([queries[qid]] * len(cands))
            flat_candidates.extend(cands)
            offsets[qid] = (offset, offset + len(cands))
            offset += len(cands)

        docs = [corpus[did] for did, _ in flat_candidates]
        scores = rerank_scores(flat_queries, docs, model, tokenizer, cfg, use_mlx_embeddings)

        for qid in batch_qids:
            start, end = offsets[qid]
            q_scores = scores[start:end]
            q_cands = flat_candidates[start:end]
            sorted_idx = np.argsort(q_scores)[::-1][:k]
            results[qid] = [(q_cands[i][0], float(q_scores[i])) for i in sorted_idx]

        if (b_idx + 1) % 10 == 0 or b_idx + 1 == n_batches:
            print(f"  reranked {min((b_idx + 1) * batch_queries, len(qids))}/{len(qids)} queries "
                  f"({time.time() - t0:.1f}s elapsed)")

    return results


def dcg(relevances: List[float], k: int) -> float:
    relevances = relevances[:k]
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def compute_metrics(
    ranked_lists: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, float]],
    k_values: List[int] = (10,),
) -> Dict:
    """Compute nDCG@k, MAP, Recall@k, MRR@k."""
    metrics = {}

    for k in k_values:
        ndcg_scores = []
        recall_scores = []
        mrr_scores = []
        ap_scores = []

        for qid, ranked in ranked_lists.items():
            rels = qrels.get(qid, {})
            if not rels:
                continue

            # nDCG@k
            ranked_rels = [rels.get(did, 0.0) for did, _ in ranked[:k]]
            ideal_rels = sorted(rels.values(), reverse=True)[:k]
            idcg_val = dcg(ideal_rels, k)
            ndcg_scores.append(dcg(ranked_rels, k) / idcg_val if idcg_val > 0 else 0.0)

            # Recall@k
            n_relevant = sum(1 for score in rels.values() if score > 0)
            retrieved_relevant = sum(1 for did, _ in ranked[:k] if rels.get(did, 0) > 0)
            recall_scores.append(retrieved_relevant / n_relevant if n_relevant > 0 else 0.0)

            # MRR@k
            rr = 0.0
            for rank, (did, _) in enumerate(ranked[:k], start=1):
                if rels.get(did, 0) > 0:
                    rr = 1.0 / rank
                    break
            mrr_scores.append(rr)

            # AP (for MAP)
            ap = 0.0
            n_correct = 0
            for rank, (did, _) in enumerate(ranked, start=1):
                if rels.get(did, 0) > 0:
                    n_correct += 1
                    ap += n_correct / rank
            ap_scores.append(ap / n_relevant if n_relevant > 0 else 0.0)

        metrics[f"nDCG@{k}"] = round(float(np.mean(ndcg_scores)), 4)
        metrics[f"Recall@{k}"] = round(float(np.mean(recall_scores)), 4)
        metrics[f"MRR@{k}"] = round(float(np.mean(mrr_scores)), 4)

    metrics["MAP"] = round(float(np.mean(ap_scores)), 4)
    return metrics


def build_config(config_name: str, max_length: int) -> Dict:
    raw = resolve_model_config(config_name)
    cfg = {
        "model_id": raw["model_id"],
        "display_name": raw.get("display_name", raw["model_id"]),
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
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-config", required=True, help="Embedding model config name")
    parser.add_argument("--reranker-config", required=True, help="Reranker model config name")
    parser.add_argument("--task", default="SciFact", help="MTEB retrieval task name")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--top-k-retrieve", type=int, default=100)
    parser.add_argument("--top-k-rerank", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="results/e2e_rag_results.json")
    args = parser.parse_args()

    emb_cfg = build_config(args.embedding_config, args.max_length)
    rnk_cfg = build_config(args.reranker_config, args.max_length)

    print(f"Task: {args.task}")
    print(f"Embedding: {emb_cfg['model_id']}")
    print(f"Reranker: {rnk_cfg['model_id']}")
    print(f"max_length={args.max_length}, top_k_retrieve={args.top_k_retrieve}, top_k_rerank={args.top_k_rerank}")

    # Load data
    print("\nLoading MTEB retrieval data...")
    t0 = time.time()
    corpus, queries, qrels = load_mteb_retrieval_data(args.task, args.max_length)
    print(f"  Corpus: {len(corpus)} docs, Queries: {len(queries)}, Qrels: {len(qrels)} ({time.time()-t0:.2f}s)")

    # Load embedding model
    print("\nLoading embedding model...")
    t0 = time.time()
    emb_model, emb_tokenizer, emb_load_time = load_embedding_model(emb_cfg["model_id"], emb_cfg["loader"])
    # Attach embedding config to tokenizer for mlx_lm path
    emb_tokenizer._config = emb_cfg.get("embedding", {"pooling": "last", "normalize": True})
    print(f"  Loaded in {time.time()-t0:.2f}s")

    # Encode corpus
    print("\nEncoding corpus...")
    t0 = time.time()
    doc_ids, doc_embeddings = encode_corpus(corpus, emb_model, emb_tokenizer, emb_cfg, args.batch_size)
    print(f"  Done in {time.time()-t0:.2f}s, shape={doc_embeddings.shape}")

    # Encode queries (only those with qrels to save time)
    eval_qids = set(qrels.keys())
    eval_queries = {qid: queries[qid] for qid in eval_qids if qid in queries}
    print(f"\nEncoding {len(eval_queries)} evaluation queries...")
    t0 = time.time()
    query_embeddings = encode_queries(eval_queries, emb_model, emb_tokenizer, emb_cfg, args.batch_size)
    print(f"  Done in {time.time()-t0:.2f}s")

    # Retrieval only
    print("\n[Phase 1] Embedding retrieval...")
    t0 = time.time()
    retrieval_results = {}
    for qid, q_emb in query_embeddings.items():
        retrieval_results[qid] = retrieve_top_k(q_emb, doc_ids, doc_embeddings, args.top_k_rerank)
    print(f"  Done in {time.time()-t0:.2f}s")

    retrieval_metrics = compute_metrics(retrieval_results, qrels, k_values=[1, 5, 10, 100])
    print("  Retrieval metrics:")
    for k, v in retrieval_metrics.items():
        print(f"    {k}: {v}")

    # Load reranker model
    print("\nLoading reranker model...")
    t0 = time.time()
    rnk_model, rnk_tokenizer, rnk_load_time, use_mlx_embeddings = load_reranker_model(rnk_cfg)
    print(f"  Loaded in {time.time()-t0:.2f}s (use_mlx_embeddings={use_mlx_embeddings})")

    # Rerank
    print(f"\n[Phase 2] Reranker reranking top-{args.top_k_retrieve} → top-{args.top_k_rerank}...")
    t0 = time.time()
    query_candidates = {}
    for qid, q_emb in query_embeddings.items():
        query_candidates[qid] = retrieve_top_k(q_emb, doc_ids, doc_embeddings, args.top_k_retrieve)
    rerank_results = rerank_queries(
        query_candidates, eval_queries, corpus,
        rnk_model, rnk_tokenizer, rnk_cfg, use_mlx_embeddings,
        args.top_k_rerank,
    )
    print(f"  Done in {time.time()-t0:.2f}s")

    rerank_metrics = compute_metrics(rerank_results, qrels, k_values=[1, 5, 10])
    print("  Retrieval + Reranker metrics:")
    for k, v in rerank_metrics.items():
        print(f"    {k}: {v}")

    # Save
    result = {
        "task": args.task,
        "embedding_model": emb_cfg["model_id"],
        "reranker_model": rnk_cfg["model_id"],
        "max_length": args.max_length,
        "top_k_retrieve": args.top_k_retrieve,
        "top_k_rerank": args.top_k_rerank,
        "corpus_size": len(corpus),
        "num_queries": len(queries),
        "num_eval_queries": len(eval_queries),
        "retrieval_only": retrieval_metrics,
        "with_reranker": rerank_metrics,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
