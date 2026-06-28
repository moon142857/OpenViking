# Qwen3-0.6B MLX 本地评测 vs 官方 MTEB 完整报告

> **范围**：仅 0.6B 模型 —— Embedding（4bit-DWQ / mxfp8）、Reranker（4bit cross-encoder / mxfp8 bi-encoder）。
> **生成时间**：2026-06-28　**环境**：Mac, MLX (Metal), `~/mlx-env`
> **本地配置**：max_length=256（Retrieval）、量化权重、通用 instruction。
> **官方配置**：fp16、max_length=8192、任务特定 instruction（来源 [embeddings-benchmark/results](https://github.com/embeddings-benchmark/results)）。
> **原始数据**：见 `raw/*.json`。

---

## 0. 执行摘要

1. **Embedding 与官方趋势完全一致，绝对分数低 1.5%–7.3%**，差距来源单一可解释（max_length=256、通用 instruction、量化），其中 SciFact 几乎对齐（-1.5%）。
2. **4bit 与 mxfp8 精度几乎无差别（<1%）**，4bit 更快更省内存 → 0.6B 场景首选 4bit。
3. **修复了 Reranker 的致命 bug**：cross-encoder 之前会让端到端 nDCG 暴跌 55%，修复后**转为正向提升**（SciFact +6.6%，NFCorpus +4.0%）。
4. **cross-encoder 远胜 bi-encoder**：同一检索结果，4bit cross-reranker 提升 +6.6%，而 mxfp8 bi-encoder 当 reranker 用会把 nDCG 砸掉 -66.5%。**Reranker 必须用 cross-encoder（0.6B-4bit）**。

---

## 1. Embedding：本地 vs 官方 0.6B

| 模型 | 任务 | 本地 | 官方 | 差距 | 指标 |
|------|------|------|------|------|------|
| 0.6B 4bit | STSBenchmark | 0.8451 | 0.9113 | -7.3% | spearman |
| 0.6B 4bit | SciFact | 0.6867 | 0.6972 | **-1.5%** | nDCG@10 |
| 0.6B 4bit | NFCorpus | 0.3430 | 0.3671 | -6.6% | nDCG@10 |
| 0.6B mxfp8 | STSBenchmark | 0.8453 | 0.9113 | -7.2% | spearman |
| 0.6B mxfp8 | SciFact | 0.6849 | 0.6972 | -1.8% | nDCG@10 |
| 0.6B mxfp8 | NFCorpus | 0.3469 | 0.3671 | -5.5% | nDCG@10 |

### 差距分析

| 因素 | 本地 | 官方 | 影响 |
|------|------|------|------|
| max_length | 256 | 8192 | 长文本/长查询被截断 → STS、长文档检索影响最大 |
| Instruction | 通用 web-search | 任务特定 | 官方 +1%–5%，对 Retrieval 尤甚 |
| 精度 | 4bit-DWQ / mxfp8 | fp16 | 量化损失 <2%（4bit vs mxfp8 差 <1% 已佐证） |
| 后端 | MLX/Metal | PyTorch/CUDA | 数值实现差异，量级小 |

- **SciFact 差距最小（-1.5%）**：文档是短摘要，256 足够覆盖，对 instruction 不敏感。
- **STSBenchmark 差距最大（-7.3%）**：句子级细微语义最吃 fp16 + 长上下文 + 任务指令的综合收益。
- **4bit vs mxfp8**：三项任务差距均 <1%（STS +0.02%、SciFact -0.18%、NFCorpus +0.39%），量化对 0.6B 几乎无损 → **选 4bit**（更快、更省内存）。

---

## 2. Reranker 端到端 RAG

> 流程：Embedding 召回 top-20 → Reranker 重排 top-10。关注 **+Reranker 相对 Retrieval-only 的变化**（无官方端到端基准，以"是否提升检索"为准）。

| 配置 | 任务 | Retrieval-only nDCG@10 | +Reranker nDCG@10 | 变化 | MAP 变化 |
|------|------|------|------|------|------|
| 4bit Emb + **4bit Reranker (cross)** | SciFact | 0.6838 | **0.7288** | **+6.6%** | 0.6344→0.6843 (+7.9%) |
| 4bit Emb + **4bit Reranker (cross)** | NFCorpus | 0.3441 | **0.3577** | **+4.0%** | 0.1273→0.1359 (+6.8%) |
| 4bit Emb + mxfp8 Reranker (bi) | SciFact | 0.6838 | 0.2294 | **-66.5%** | 0.6344→0.1489 (-76.5%) |

**结论**：
- 修复后的 **cross-encoder reranker 在两个数据集上都稳定提升检索**（nDCG +6.6% / +4.0%，MAP 同步提升），符合一个正常 reranker 的预期。
- **mxfp8 bi-encoder 不能用作 reranker**：把 query/doc 各自嵌入再算 cosine，对已经召回的近邻区分度极差，重排后 nDCG 砸掉 2/3。它仅在独立 Reranking 任务（如 AskUbuntu，bi-encoder MAP@20≈0.43）下勉强可用。
- **推荐**：教育 RAG 用 **0.6B-4bit Embedding + 0.6B-4bit cross-encoder Reranker**。

---

## 3. Reranker Bug 修复说明（关键）

之前的报告把 reranker 失效归因于"批处理缺 attention_mask 的 padding 污染"。排查发现那只是次要因素，**真正的元凶是截断截掉了打分位**：

| # | 问题 | 现象 | 修复 |
|---|------|------|------|
| ① | 取 `logits[:, -1, :]` 但右 padding | 短序列的 -1 是 PAD token，分数被同批样本污染 | 按 `attention_mask.sum-1` 取真实末 token |
| ② | **整串 prompt 截断到 256** | 长文档 prompt>256 时，`</think>` assistant **决策后缀被砍掉**，打分位落在文档中间，logits 是 `the`/`ine` 等词而非 yes/no | **prefix/suffix 单独 tokenize，只截断文档正文**（对齐官方实现），保证 yes/no 决策位永远保留 |

**修复前后（40 query × 真实 SciFact，相关文档 vs 8 个随机文档）**：

| 指标 | 修复前 | 修复后 |
|------|-------|-------|
| 相关文档排第 1 | 25% | **100%** |
| 相关文档分数中位数 | 0.26 | **0.98** |

端到端 SciFact 也从 +Reranker nDCG 0.30（坏）→ **0.7288**（好）。

> 诊断铁证：修复前对长文档，模型在打分位预测的 top token 是 `the / a / e / ine` 等词片段，而非 `yes/no` —— 直接证明后缀被截断。

代码：`scripts/benchmark_reranker.py::rerank_scores_cross_encoder`。

---

## 4. 方法与复现

```bash
cd benchmark/embedding_reranker
source ~/mlx-env/bin/activate
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/run_0.6b_report.py     # 断点续跑：已存在的 raw/*.json 会跳过
```

- 驱动脚本：`scripts/run_0.6b_report.py`（串行跑 6 个 Embedding MTEB + 3 个 e2e，自动生成本报告）。
- Embedding MTEB：`scripts/mteb_benchmark.py`（main_score = STS→spearman / Retrieval→nDCG@10）。
- 端到端 RAG：`scripts/e2e_rag_eval.py`（top_k_retrieve=20, top_k_rerank=10）。
- 数据集：SciFact（corpus 5183 / eval-q 300）、NFCorpus（eval-q 323）、STSBenchmark。

---

## 5. 已知限制

- **本地 Retrieval 慢**：0.6B≈10–11 docs/s（max_length=256），全 battery 串行约 2 小时；GPU-bound，不宜并行多进程。
- **官方端到端 RAG 无公开基准**：Reranker 一节用相对提升而非绝对对标。
- **未提高 max_length / 未用任务特定 instruction**：若要进一步逼近官方 Embedding 分数，需提到 512/8192 并接入 `task_prompts.json`。
- **bi-encoder reranker 不纳入推荐**，仅作对照。

---

## 6. 原始结果文件

| 文件 | 内容 |
|------|------|
| `raw/mteb_e_4bit_stsb.json` / `..._scifact.json` / `..._nfcorpus.json` | 0.6B 4bit Embedding MTEB |
| `raw/mteb_e_mxfp8_stsb.json` / `..._scifact.json` / `..._nfcorpus.json` | 0.6B mxfp8 Embedding MTEB |
| `raw/e2e_scifact_4bit_cross.json` | SciFact e2e（cross-reranker，修复后） |
| `raw/e2e_nfcorpus_4bit_cross.json` | NFCorpus e2e（cross-reranker，修复后） |
| `raw/e2e_scifact_4bit_bi.json` | SciFact e2e（bi-encoder reranker，对照） |
