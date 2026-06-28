#!/usr/bin/env python3
"""Complete 0.6B Qwen3 MLX benchmark battery vs official MTEB, with report.

Runs ONLY the 0.6B models (embedding 4bit/mxfp8, reranker 4bit cross-encoder /
mxfp8 bi-encoder), compares to official Qwen3-Embedding-0.6B MTEB scores, and
writes a Markdown report. Each step is resumable: a step whose output JSON
already exists is skipped, so the battery can be re-run to fill gaps.

Usage:
    source ~/mlx-env/bin/activate
    export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python scripts/run_0.6b_report.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "report_0.6b" / "raw"
REPORT = ROOT / "report_0.6b" / "REPORT.md"
PY = sys.executable

E_4BIT = "qwen3_embedding_0.6b_4bit_dwq"
E_MXFP8 = "qwen3_embedding_0.6b_mxfp8"
R_4BIT = "qwen3_reranker_0.6b_4bit"      # cross-encoder (fixed)
R_MXFP8 = "qwen3_reranker_0.6b_mxfp8"    # bi-encoder

# Official Qwen3-Embedding-0.6B MTEB scores (embeddings-benchmark/results).
# main_score per task: STS -> spearman, Retrieval -> nDCG@10.
OFFICIAL = {
    "STSBenchmark": 0.9113,
    "SciFact": 0.6972,
    "NFCorpus": 0.3671,
}

# (output_name, config, task, max_length)
EMBED_RUNS = [
    ("mteb_e_4bit_stsb.json",     E_4BIT,  "STSBenchmark", None),
    ("mteb_e_4bit_scifact.json",  E_4BIT,  "SciFact",      256),
    ("mteb_e_4bit_nfcorpus.json", E_4BIT,  "NFCorpus",     256),
    ("mteb_e_mxfp8_stsb.json",    E_MXFP8, "STSBenchmark", None),
    ("mteb_e_mxfp8_scifact.json", E_MXFP8, "SciFact",      256),
    ("mteb_e_mxfp8_nfcorpus.json",E_MXFP8, "NFCorpus",     256),
]

# (output_name, embedding_config, reranker_config, task)
E2E_RUNS = [
    ("e2e_scifact_4bit_cross.json",  E_4BIT, R_4BIT,  "SciFact"),
    ("e2e_nfcorpus_4bit_cross.json", E_4BIT, R_4BIT,  "NFCorpus"),
    ("e2e_scifact_4bit_bi.json",     E_4BIT, R_MXFP8, "SciFact"),
]


def run(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    print(f"  -> rc={rc} in {time.time()-t0:.1f}s", flush=True)
    return rc


def run_embedding():
    for out, cfg, task, max_len in EMBED_RUNS:
        path = RAW / out
        if path.exists():
            print(f"[skip] {out} exists", flush=True)
            continue
        cmd = [PY, str(HERE / "mteb_benchmark.py"),
               "--config", cfg, "--tasks", task, "--output", str(path)]
        if max_len:
            cmd += ["--max-length", str(max_len)]
        run(cmd)


def run_e2e():
    for out, emb, rnk, task in E2E_RUNS:
        path = RAW / out
        if path.exists():
            print(f"[skip] {out} exists", flush=True)
            continue
        cmd = [PY, str(HERE / "e2e_rag_eval.py"),
               "--embedding-config", emb, "--reranker-config", rnk,
               "--task", task, "--max-length", "256",
               "--top-k-retrieve", "20", "--top-k-rerank", "10",
               "--output", str(path)]
        run(cmd)


def _load(name):
    p = RAW / name
    return json.loads(p.read_text()) if p.exists() else None


def _main_score(doc, task):
    if not doc:
        return None
    return doc.get("results", {}).get(task, {}).get("main_score")


def gen_report():
    def pct(local, off):
        if local is None or not off:
            return "—"
        return f"{(local-off)/off*100:+.1f}%"

    def fmt(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) else "—"

    lines = []
    lines.append("# Qwen3-0.6B MLX 本地评测 vs 官方 MTEB 完整报告\n")
    lines.append("> 仅测试 0.6B 模型（Embedding 4bit/mxfp8，Reranker 4bit cross-encoder / mxfp8 bi-encoder）。")
    lines.append("> 本地环境：Mac, MLX, max_length=256（Retrieval）；官方：fp16, max_length=8192, 任务特定 instruction。\n")

    # --- Embedding table ---
    lines.append("## 1. Embedding：本地 vs 官方 0.6B\n")
    lines.append("| 模型 | 任务 | 本地 | 官方 | 差距 | 指标 |")
    lines.append("|------|------|------|------|------|------|")
    embed_map = [
        ("0.6B 4bit",  "STSBenchmark", "mteb_e_4bit_stsb.json",     "spearman"),
        ("0.6B 4bit",  "SciFact",      "mteb_e_4bit_scifact.json",  "nDCG@10"),
        ("0.6B 4bit",  "NFCorpus",     "mteb_e_4bit_nfcorpus.json", "nDCG@10"),
        ("0.6B mxfp8", "STSBenchmark", "mteb_e_mxfp8_stsb.json",    "spearman"),
        ("0.6B mxfp8", "SciFact",      "mteb_e_mxfp8_scifact.json", "nDCG@10"),
        ("0.6B mxfp8", "NFCorpus",     "mteb_e_mxfp8_nfcorpus.json","nDCG@10"),
    ]
    for label, task, fname, metric in embed_map:
        s = _main_score(_load(fname), task)
        off = OFFICIAL.get(task)
        lines.append(f"| {label} | {task} | {fmt(s)} | {fmt(off)} | {pct(s, off)} | {metric} |")

    # --- Reranker / e2e table ---
    lines.append("\n## 2. Reranker 端到端 RAG（修复后）\n")
    lines.append("> Embedding 召回 top-20 → Reranker 重排 top-10。关注 +Reranker 相对 Retrieval-only 的变化。\n")
    lines.append("| 配置 | 任务 | Retrieval-only nDCG@10 | +Reranker nDCG@10 | 变化 |")
    lines.append("|------|------|------|------|------|")
    e2e_map = [
        ("4bit Emb + 4bit Reranker (cross)", "SciFact",  "e2e_scifact_4bit_cross.json"),
        ("4bit Emb + 4bit Reranker (cross)", "NFCorpus", "e2e_nfcorpus_4bit_cross.json"),
        ("4bit Emb + mxfp8 Reranker (bi)",   "SciFact",  "e2e_scifact_4bit_bi.json"),
    ]
    for label, task, fname in e2e_map:
        d = _load(fname)
        if not d:
            lines.append(f"| {label} | {task} | — | — | — |")
            continue
        ro = d["retrieval_only"]["nDCG@10"]
        wr = d["with_reranker"]["nDCG@10"]
        chg = f"{(wr-ro)/ro*100:+.1f}%" if ro else "—"
        lines.append(f"| {label} | {task} | {fmt(ro)} | {fmt(wr)} | {chg} |")

    lines.append("\n## 3. 原始结果文件\n")
    for p in sorted(RAW.glob("*.json")):
        lines.append(f"- `raw/{p.name}`")

    REPORT.write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {REPORT}", flush=True)


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    print("=== Embedding MTEB runs ===", flush=True)
    run_embedding()
    print("\n=== End-to-end RAG runs ===", flush=True)
    run_e2e()
    print("\n=== Generating report ===", flush=True)
    gen_report()
    print("DONE", flush=True)
