# Embedding / Reranker 多模型对比报告：default

**生成时间：** 2026-06-29 09:28:55
**测试环境：** macOS Apple Silicon, MLX 本地推理

---

## 1. 评测方法

- Embedding 脚本：`scripts/benchmark_embedding.py`（配置驱动）
- Reranker 脚本：`scripts/benchmark_reranker.py`（配置驱动）
- 数据集：`datasets/` 内置小样本数据集
- Reranker 统一使用官方 Qwen3-Reranker prompt 和 yes/no softmax 概率打分

---

## 2. Embedding 结果对比

| 组合 | 模型 | Loader | STS Spearman | Paraphrase Acc | Paraphrase F1 | Recall@1 | Recall@5 | MRR | V-measure |
|------|------|--------|--------------|----------------|---------------|----------|----------|-----|-----------|
| low_memory_4bit | Qwen3-Embedding-0.6B-4bit-DWQ | mlx_embeddings | 0.9409 | 0.9000 | 0.9167 | 0.9333 | 1.0000 | 0.9500 | 1.0000 |

---

## 3. Reranker 结果对比

| 组合 | 模型 | Loader | Mode | Pairwise Acc | NDCG@10 | Spearman |
|------|------|--------|------|--------------|---------|----------|
| low_memory_4bit | Qwen3-Reranker-0.6B-4bit (cross-encoder) | mlx_lm | cross_encoder | 1.0000 | 1.0000 | 0.9101 |
| balanced | Qwen3-Reranker-0.6B-4bit (cross-encoder) | mlx_lm | cross_encoder | 1.0000 | 1.0000 | 0.9101 |

---

## 4. 模型配置说明

| 组合 | Embedding 配置 | Reranker 配置 | 说明 |
|------|----------------|---------------|------|
| low_memory_4bit | `qwen3_embedding_0.6b_4bit_dwq` | `qwen3_reranker_0.6b_4bit` | Low-memory 4-bit combination |
| low_memory_mxfp8 | `qwen3_embedding_0.6b_mxfp8` | `qwen3_reranker_0.6b_mxfp8` | Low-memory mxfp8 combination |
| balanced | `qwen3_embedding_4b_4bit_dwq` | `qwen3_reranker_0.6b_4bit` | Balanced 4B embedding + 0.6B reranker |
| high_quality | `qwen3_embedding_8b_mxfp8` | `qwen3_reranker_8b_mxfp8` | Highest-quality 8B mxfp8 combination |

---

## 5. 结论与注意事项

- 所有 `mlx-community/Qwen3-Embedding-*` 模型都应使用 `mlx-embeddings` 加载器；
  使用 `mlx_lm` 会导致 MTEB 等标准评测上的 embedding 输出接近随机（~0.09）。
- `mlx-community/Qwen3-Reranker-*-4bit` 是官方 cross-encoder，使用 `mlx_lm` 加载并以 yes/no logit 打分；
  `mlx-community/Qwen3-Reranker-*-mxfp8` 按官方 mlx-embeddings 用法为 bi-encoder，
  使用 query/document embedding 的 cosine similarity 打分。
- 实测 `Qwen3-Reranker-0.6B-mxfp8` 在 bi-encoder 模式下表现较差；
  `Qwen3-Reranker-8B-mxfp8` 可用但通常仍弱于 0.6B-4bit cross-encoder。
- 内置数据集样本量小，结论仅用于快速横向对比，生产决策请用真实业务数据或 MTEB 官方数据集。

---

*报告生成脚本：* `benchmark/embedding_reranker/scripts/run_suite.py`