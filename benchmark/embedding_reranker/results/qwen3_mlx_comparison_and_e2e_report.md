# Qwen3 MLX Embedding/Reranker 短期对比、端到端 RAG 与官方基准对比报告

> 生成时间：2026-06-27  
> 测试环境：Mac M4 (32GB), macOS, MLX 本地推理  
> 对比基准：Qwen 官方 MTEB  leaderboard 结果（来自 [embeddings-benchmark/results](https://github.com/embeddings-benchmark/results)）

---

## 1. 执行摘要

本报告在修复脚本 bug 的基础上，完成了：
1. **短期模型对比**：0.6B 4bit / 0.6B mxfp8 / 4B 4bit 在 STSBenchmark、SciFact、NFCorpus 上的横向评测。
2. **端到端 RAG 评测**：实现 `scripts/e2e_rag_eval.py`，在 SciFact 上验证 Embedding → Reranker 串联效果。
3. **官方基准对比**：将本地分数与 Qwen 官方 MTEB 结果对比，分析差距原因。

**核心结论**：
- 本地 0.6B/4B Embedding 分数普遍低于官方 5%–10%，主要受 **max_length=256**、**通用 instruction**、**量化精度** 和 **MLX 本地推理效率** 影响。
- **SciFact 上差距最小（-1.5%）**，说明短文档检索对 256 length 不敏感；**STS 差距最大（-7%）**，说明长文本和任务指令对 STS 影响更大。
- **端到端 RAG 的 retrieval-only 阶段与 MTEB 一致（nDCG@10=0.6838）**，但加入 cross-encoder reranker 后分数异常下降，原因是 `benchmark_reranker.py` 在批处理不同长度序列时未传入 `attention_mask`，导致 padding token 干扰分数。

---

## 2. 官方基准数据来源

从 [embeddings-benchmark/results](https://github.com/embeddings-benchmark/results) 拉取的 Qwen 官方 MTEB 结果：

| 模型 | 来源 commit |
|------|-------------|
| Qwen3-Embedding-0.6B | `b22da495047858cce924d27d76261e96be6febc0` |
| Qwen3-Embedding-4B | `636cd9bf47d976946cdbb2b0c3ca0cb2f8eea5ff` |
| Qwen3-Embedding-8B | `4e423935c619ae4df87b646a3ce949610c66241c` |

官方评测配置（来自 Qwen GitHub `evaluation/run_mteb.sh`）：
- `max_length=8192`
- `precision=fp16`
- `pooler_type=last`
- `do_norm=true`
- `use_instruction=true`
- 任务特定 instruction（`task_prompts.json`）
- `batch_size=8`

---

## 3. Embedding 模型：本地结果 vs 官方基准

### 3.1 测试结果汇总

| 模型 | 任务 | 本地分数 | 官方分数 | 差距 | 本地耗时 |
|------|------|----------|----------|------|----------|
| **0.6B 4bit** | STSBenchmark | 0.8451 | **0.9113** | -7.28% | ~29s |
| **0.6B 4bit** | SciFact | 0.6867 | **0.6972** | -1.51% | ~10m50s |
| **0.6B 4bit** | NFCorpus | 0.3430 | **0.3671** | -6.56% | ~7m20s |
| **0.6B mxfp8** | STSBenchmark | 0.8453 | **0.9113** | -7.24% | ~2m |
| **0.6B mxfp8** | SciFact | 0.6849 | **0.6972** | -1.76% | ~13m45s |
| **0.6B mxfp8** | NFCorpus | 0.3469 | **0.3671** | -5.50% | ~10m45s |
| **4B 4bit** | STSBenchmark | 0.8692 | **0.9370** | -7.24% | ~5m45s |
| **4B 4bit** | SciFact | — | **0.7833** | — | >45m（卡住） |
| **4B 4bit** | NFCorpus | — | **0.4110** | — | 未跑 |
| **8B mxfp8** | STSBenchmark | — | **0.9360** | — | — |
| **8B mxfp8** | SciFact | — | **0.7846** | — | — |
| **8B mxfp8** | NFCorpus | — | **0.4145** | — | — |

### 3.2 横向模型对比

| 任务 | 0.6B 4bit | 0.6B mxfp8 | 4B 4bit | 8B 官方 |
|------|-----------|------------|---------|---------|
| STSBenchmark | 0.8451 | 0.8453 | 0.8692 | 0.9360 |
| SciFact | 0.6867 | 0.6849 | — | 0.7846 |
| NFCorpus | 0.3430 | 0.3469 | — | 0.4145 |

**观察**：
1. **0.6B 4bit 与 mxfp8 几乎没有差异**：STSBenchmark 0.8451 vs 0.8453，SciFact 0.6867 vs 0.6849，NFCorpus 0.3430 vs 0.3469。4bit 更快，是低内存首选。
2. **4B 在 STS 上明显优于 0.6B**：0.8692 vs ~0.845，提升约 2.4 个百分点，与调研文档中 "balanced 配置显著提升" 一致。
3. **官方 4B/8B 在检索任务上大幅领先 0.6B**：SciFact 0.7833/0.7846 vs 0.6972，NFCorpus 0.4110/0.4145 vs 0.3671。说明大模型在专业检索上有明显优势。

### 3.3 与官方差距分析

| 因素 | 本地设置 | 官方设置 | 对分数的影响 |
|------|----------|----------|--------------|
| **max_length** | 256 | 8192 | 官方可保留完整长文本；本地对长文档截断，STS/长检索影响大 |
| **Instruction** | 通用 `"Given a web search query..."` | 任务特定（如 SciFact 用科学声明检索指令） | 官方任务指令更精准，Qwen 博客称可提升 1%–5% |
| **精度** | 4bit-DWQ / mxfp8 | fp16 | 量化带来轻微精度损失，但对 0.6B/4B 影响预计 <2% |
| **推理后端** | MLX / Metal | PyTorch / CUDA | 数值实现差异、padding/attention 处理差异 |
| **MTEB 版本** | 2.16.1 | 官方使用与模型同时期的 MTEB | API、数据集格式、指标计算可能存在微小差异 |

**分项解释**：

- **SciFact 差距最小（-1.5%）**：SciFact 文档是短摘要，256 length 足够覆盖；任务本身对指令不敏感，因此本地与官方接近。
- **NFCorpus 差距中等（-5.5% ~ -6.6%）**：医学问题检索，部分文档/问题较长，截断会丢失关键信息；任务特定指令也有帮助。
- **STSBenchmark 差距最大（-7.2% ~ -7.3%）**：STS 需要捕捉句子级细微语义，官方 fp16 + 任务指令 + 更长上下文共同作用，使分数明显高于本地量化版本。
- **4B STSBenchmark 本地 0.8692 vs 官方 0.9370**：即使模型更大，本地仍落后 6.8 个百分点，说明 max_length/instruction/后端的综合影响大于模型容量提升。

### 3.4 本地速度问题

实测本地 Retrieval 速度远低于调研文档预估：

| 模型 | max_length=256 | max_length=512 |
|------|----------------|----------------|
| 0.6B 4bit | ~10–11 docs/s | ~5 docs/s |
| 0.6B mxfp8 | ~8–9 docs/s | ~4 docs/s |
| 4B 4bit | <3 docs/s（且易卡顿/内存不足） | 未测 |

**原因分析**：
- 调研文档估算基于序列长度 128、理想 Metal 利用率；实际长文档（256–512）下吞吐量下降明显。
- 4B/8B 模型在 32GB Mac 上运行 Retrieval 时，模型加载 + 向量索引 + 系统开销接近内存上限，出现交换和卡顿。
- 结论：**4B/8B 全量 Retrieval 任务不适合在当前 Mac 上跑**，应在 GPU/云端执行。

---

## 4. 端到端 RAG 评测

### 4.1 实现说明

新增脚本 `scripts/e2e_rag_eval.py`，流程：
1. 加载 MTEB Retrieval 数据集（corpus / queries / qrels）。
2. Embedding 模型编码全部文档。
3. 对每个查询：编码 → cosine 相似度召回 Top-K。
4. Reranker 模型对 Top-K 候选重新排序，输出 Top-10。
5. 分别计算 **Retrieval-only** 和 **Retrieval + Reranker** 的 nDCG@k、MAP、Recall@k、MRR@k。

为加速 reranker，脚本实现了跨查询 batch（默认 4 个查询一起处理），并对 SciFact 只处理有标注的 300 个查询。

### 4.2 SciFact 结果

配置：`Qwen3-Embedding-0.6B-4bit-DWQ` + `Qwen3-Reranker-0.6B-4bit`，max_length=256，top_k_retrieve=20，top_k_rerank=10。

| 阶段 | nDCG@10 | MAP | Recall@10 | MRR@10 |
|------|---------|-----|-----------|--------|
| **Retrieval-only** | **0.6838** | **0.6344** | **0.8209** | **0.6457** |
| **+ Cross-Encoder Reranker** | 0.3044 | 0.2098 | 0.5985 | 0.2208 |
| 变化 | **-55.5%** | **-66.9%** | -27.1% | **-65.8%** |

### 4.3 结果分析

**Retrieval-only 与 MTEB 一致**：MTEB SciFact 0.6B 4bit 为 0.6867，端到端 retrieval-only 为 0.6838，差异仅 0.4%，证明编码和召回逻辑正确。

**加入 Reranker 后分数异常下降**，这与预期相反。经排查，原因是 `benchmark_reranker.py` 的 cross-encoder 实现存在 **padding/batching 缺陷**：

```python
# benchmark_reranker.py: rerank_scores_cross_encoder
input_ids = mx.array(inputs["input_ids"].astype(np.int32))
outputs = model(input_ids)  # 未传入 attention_mask！
```

- `mlx_lm` 加载的 Qwen3-Reranker 模型不接受 `attention_mask` 参数。
- 当多个 `(query, doc)` 对拼接成 batch 时，tokenizer 会按 batch 内最长序列 padding；模型因无 attention_mask，会把 padding token 当作正常 token 参与 attention，导致分数被批次内其他样本的长度/内容干扰。
- 实验验证：同一查询、同一文档，单独打分与 batch 打分结果差异显著（例如无关文档分数从 0.00001 被抬高到 0.1366）。

**这不是模型本身的问题**，而是本地 cross-encoder 推理实现的问题。Qwen 官方使用 `transformers` + `flash_attention_2`，正确支持 attention_mask，因此 reranker 在官方评测中表现正常（MTEB-R 0.6B reranker 65.80）。

### 4.4 修复建议

1. **短期**：在 `benchmark_reranker.py` 中提供 `batch_size=1` 模式，每个 `(query, doc)` 对单独前向，避免 padding 干扰；代价是速度极慢。
2. **中期**：换用支持 `attention_mask` 的模型加载方式（如直接使用 `transformers` + PyTorch，或等待 mlx-lm 更新）。
3. **长期**：使用官方 Qwen 的 evaluation 脚本在 GPU 上跑端到端 RAG，获得可靠基准。

---

## 5. 数据集与资源状态

### 5.1 已下载/可用数据集

| 数据集 | 状态 | 说明 |
|--------|------|------|
| SciFact | ✅ 已缓存 + 已跑通 | corpus/queries/default 三配置完整 |
| NFCorpus | ✅ 已缓存 + 已跑通 | 通过 HF Mirror 刷新为完整格式 |
| FiQA2018 | ✅ 已下载 | corpus 57,638 / queries 6,648，未跑全量（预计单模型 90 分钟） |
| ArguAna | ⚠️ 旧缓存不兼容 | 仅 default 配置，运行会报错 |
| SCIDOCS | ⚠️ 旧缓存不兼容 | 同上 |
| STSBenchmark | ✅ 已缓存 + 已跑通 | — |
| AskUbuntuDupQuestions | ✅ 已缓存 + 已跑通 | Reranker bi-encoder 测试 |

### 5.2 资源约束

- **磁盘**：55GB 可用，足够下载所有推荐数据集。
- **内存**：32GB 总容量，当前 29GB 已用。8B + 8B high_quality 组合加载后接近上限，4B Retrieval 任务已出现卡顿。
- **时间**：0.6B Retrieval 单数据集 7–14 分钟；4B/8B 不适合本地全量跑。

---

## 6. 综合结论

### 6.1 模型选型建议

| 场景 | 推荐配置 | 理由 |
|------|----------|------|
| **低内存 / 高吞吐原型** | 0.6B 4bit Embedding + 0.6B 4bit Reranker | 速度快、内存小、精度可接受 |
| **平衡配置（教育 RAG）** | 4B 4bit Embedding + 0.6B 4bit Reranker | STS 上明显优于 0.6B；检索任务需在 GPU 跑全量验证 |
| **最高精度** | 8B mxfp8 Embedding + 8B/4B Reranker | 官方分数最高，但本地 Mac 无法高效运行 |

### 6.2 关键发现

1. **量化影响小**：0.6B 4bit 与 mxfp8 在 SciFact/NFCorpus/STS 上差距均 <1%，4bit 更快更省内存。
2. **模型容量影响大**：4B 在 STS 上比 0.6B 高约 2.4%；官方 4B/8B 在检索任务上比 0.6B 高 8%–13%。
3. **Instruction 修复显著提升 Retrieval**：SciFact 从 0.6590 提升到 0.6867（+2.77%），证明 query instruction 对检索至关重要。
4. **本地 cross-encoder Reranker 当前不可靠**：由于 attention_mask 缺失，batch 内 padding 干扰分数，端到端 Reranker 阶段分数异常。
5. **本地 Retrieval 速度是主要瓶颈**：0.6B 约 10 docs/s，4B/8B 更慢； FiQA2018 全量单模型需约 90 分钟，大模型不适合本地跑。

### 6.3 后续行动建议

**高优先级**：
1. 修复 `benchmark_reranker.py` cross-encoder 的 padding 问题（支持单条推理或换用支持 attention_mask 的 backend）。
2. 使用更长 max_length（512/8192）和任务特定 instruction 重跑核心数据集，缩小与官方差距。
3. 在 GPU/云端运行 4B/8B 的 SciFact/FiQA2018/NFCorpus 全量测试。

**中优先级**：
1. 实现 embedding 向量缓存，避免每次 e2e RAG 重复编码语料库。
2. 增加 Matryoshka 维度评测（0.6B 输出 1024 维，4B 2560 维，8B 4096 维，可测 256/512/1024 等截断）。
3. 补充中文 C-MTEB 数据集（T2Retrieval、MMarcoReranking）。

**低优先级**：
1. 恢复性能基准脚本 `benchmark_performance.py`。
2. 跑 TRECCOVID、Banking77 等 P1/P2 数据集。

---

## 7. 附录：关键文件

- `scripts/e2e_rag_eval.py` — 新增端到端 RAG 评测脚本
- `results/e2e_rag_scifact_low_memory_4bit.json` — SciFact 端到端 RAG 结果
- `results/mteb_report_final_v2.md` — MTEB 分数汇总表
- `results/qwen3_mlx_evaluation_summary.md` — 前期问题分析与测试结果汇总
- `results/qwen3_mlx_gap_analysis.md` — 调研文档与实现差距分析

---

## 8. 数据来源

- Qwen 官方博客：[Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models](https://qwenlm.github.io/blog/qwen3-embedding/)
- Qwen GitHub 仓库：[QwenLM/Qwen3-Embedding](https://github.com/QwenLM/Qwen3-Embedding)
- MTEB Leaderboard 结果仓库：[embeddings-benchmark/results](https://github.com/embeddings-benchmark/results)
- 本地 MLX 模型来源：[mlx-community](https://huggingface.co/mlx-community)
