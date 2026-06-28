# MTEB 轻量评测报告

**结果目录:** `results`

## 模型任务得分

| Model | AskUbuntuDupQuestions | NFCorpus | STS12 | STSBenchmark | SciFact | Total elapsed (s) |
|---|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B-4bit-DWQ | - | 0.3430 | 0.7711 | 0.8451 | 0.6867 | 2116.9 |
| Qwen3-Embedding-0.6B-mxfp8 | - | 0.3469 | - | 0.8453 | 0.6849 | 1585.7 |
| Qwen3-Embedding-4B-4bit-DWQ | - | - | - | 0.8692 | - | 344.9 |
| Qwen3-Reranker-0.6B-mxfp8 (bi-encoder) | 0.4275 | - | - | - | - | 402.8 |

## 说明

- 使用 `scripts/mteb_benchmark.py` 在本地 MLX (macOS Apple Silicon) 上运行。
- 分数为对应任务的 `main_score`（通常为 cosine_spearman）。
- 受限于本地算力，当前仅完成轻量 STS 任务；Retrieval / Reranking 任务耗时较长，建议后续在 GPU 环境或分批运行。