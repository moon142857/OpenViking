# Qwen3 MLX Embedding/Reranker 本地评测资源与测试方法手册

> 本手册汇总当前 Mac 本地已下载的模型、数据集、测试脚本及常用命令，方便复现和扩展评测。  
> 生成时间：2026-06-27  
> 工作目录：`benchmark/embedding_reranker/`

---

## 1. 环境信息

- **机器**：Mac M4 (32GB 统一内存)
- **系统**：macOS (Darwin 25.3.0)
- **Python 环境**：`~/mlx-env`
- **核心依赖**：
  - `mlx` (Apple Silicon 推理)
  - `mlx-lm`
  - `mlx-embeddings`
  - `mteb==2.16.1`
  - `datasets`
  - `transformers`, `scikit-learn`, `scipy`, `numpy`, `pyyaml`
- **模型缓存**：`~/.cache/huggingface/hub/`
- **数据集缓存**：`~/.cache/huggingface/datasets/`

### 1.1 磁盘与内存现状

- **系统磁盘**：460G 总计 / 371G 已用 / **55G 可用**
- **模型缓存**：~18.4G
- **数据集缓存**：~3.6G（MTEB 相关）
- **内存**：32G 总容量；跑 8B + 8B 组合时接近上限，建议关闭其他应用

---

## 2. 已下载模型

所有模型来自 `mlx-community`，已缓存于 `~/.cache/huggingface/hub/`。

| 配置名 | HuggingFace ID | 大小 | 角色 | 加载器 | 输出维度 | 用途 |
|--------|----------------|------|------|--------|----------|------|
| `qwen3_embedding_0.6b_4bit_dwq` | `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` | 335M | Embedding | `mlx_embeddings` | 1024 | low_memory_4bit / balanced |
| `qwen3_embedding_0.6b_mxfp8` | `mlx-community/Qwen3-Embedding-0.6B-mxfp8` | 600M | Embedding | `mlx_embeddings` | 1024 | low_memory_mxfp8 |
| `qwen3_embedding_4b_4bit_dwq` | `mlx-community/Qwen3-Embedding-4B-4bit-DWQ` | 2.1G | Embedding | `mlx_embeddings` | 2560 | balanced |
| `qwen3_embedding_8b_mxfp8` | `mlx-community/Qwen3-Embedding-8B-mxfp8` | 7.3G | Embedding | `mlx_embeddings` | 4096 | high_quality |
| `qwen3_reranker_0.6b_4bit` | `mlx-community/Qwen3-Reranker-0.6B-4bit` | 331M | Reranker | `mlx_lm` (cross-encoder) | — | low_memory_4bit / balanced |
| `qwen3_reranker_0.6b_mxfp8` | `mlx-community/Qwen3-Reranker-0.6B-mxfp8` | 600M | Reranker | `mlx_embeddings` (bi-encoder) | — | low_memory_mxfp8 |
| `qwen3_reranker_8b_mxfp8` | `mlx-community/Qwen3-Reranker-8B-mxfp8` | 7.3G | Reranker | `mlx_embeddings` (bi-encoder) | — | high_quality |

### 2.1 模型选型速查

| 场景 | Embedding | Reranker | 备注 |
|------|-----------|----------|------|
| 低内存 / 高吞吐原型 | `qwen3_embedding_0.6b_4bit_dwq` | `qwen3_reranker_0.6b_4bit` | 速度最快，内存最小 |
| 低内存 + 更好 embedding | `qwen3_embedding_0.6b_mxfp8` | `qwen3_reranker_0.6b_4bit` | mxfp8 精度略优，但更慢 |
| 平衡配置（教育 RAG 推荐） | `qwen3_embedding_4b_4bit_dwq` | `qwen3_reranker_0.6b_4bit` | 精度提升明显，内存可控 |
| 最高精度（实验性） | `qwen3_embedding_8b_mxfp8` | `qwen3_reranker_8b_mxfp8` | 内存压力大，本地难跑全量 |

---

## 3. 已下载数据集

### 3.1 MTEB 官方数据集

已缓存于 `~/.cache/huggingface/datasets/`，前缀为 `mteb___`。

| 数据集 | 缓存大小 | 类型 | MTEB 任务名 | 本地状态 | 说明 |
|--------|----------|------|-------------|----------|------|
| **SciFact** | 30M | Retrieval | `SciFact` | ✅ 可跑 | corpus/queries/default 三配置完整 |
| **NFCorpus** | 27M | Retrieval | `NFCorpus` | ✅ 可跑 | 已通过 HF Mirror 刷新为完整格式 |
| **FiQA2018** | 44M | Retrieval | `FiQA2018` | ✅ 可跑 | corpus 57,638 / queries 6,648；未跑全量 |
| **ArguAna** | 224K | Retrieval | `ArguAna` | ⚠️ 需刷新 | 旧缓存仅 default，运行报 `config 'corpus' not found` |
| **SCIDOCS** | 18M | Retrieval | `SCIDOCS` | ⚠️ 需刷新 | 旧缓存仅 default，需重新下载 |
| **SciDocsRR** | 18M | Reranking | `SciDocsRR` | ✅ 可跑 | 仅 default 配置，适合 Reranking |
| **AskUbuntuDupQuestions** | 444K | Reranking | `AskUbuntuDupQuestions` | ✅ 可跑 | Reranking 任务 |
| **StackOverflowDupQuestions** | 33M | Reranking | `StackOverflowDupQuestions` | ✅ 可跑 | 已缓存 |
| **STSBenchmark** | 1.5M | STS | `STSBenchmark` | ✅ 可跑 | — |
| **STS12** | 972K | STS | `STS12` | ✅ 可跑 | — |
| **SICK-R** | 2.1M | STS | `SICK-R` | ✅ 可跑 | — |
| **MS MARCO** | 15M | Retrieval | `MSMARCO` | ✅ 已缓存 | 数据量大，非 P0 |
| **MindSmallReranking** | 1.1G | Reranking | `MindSmallReranking` | ✅ 已缓存 | 数据量大 |
| **AmazonPolarity** | 1.7G | Classification | `AmazonPolarity` | ✅ 已缓存 | 非核心 |
| **ArXivClusteringP2P** | 716M | Clustering | `ArXivHierarchicalClusteringP2P` 等 | ✅ 已缓存 | 非核心 |
| ** Banking77** | ❌ 未下载 | Classification | `Banking77Classification` | ❌ | P2 优先级，暂不跑 |
| **TRECCOVID** | ❌ 未下载 | Retrieval | `TRECCOVID` | ❌ | 数据量大，P1 可选 |

### 3.2 内置小样本数据集

位于 `benchmark/embedding_reranker/datasets/`，用于快速冒烟测试。

| 文件 | 行数 | 用途 | 说明 |
|------|------|------|------|
| `sts_sample.jsonl` | 20 | STS Spearman | 句子对 + 人工相似度分数 |
| `paraphrase_sample.jsonl` | 20 | 复述检测 | 句子对 + 是否复述标签 |
| `retrieval_corpus.jsonl` | 20 | 检索召回 | 文档库 |
| `retrieval_queries.jsonl` | 15 | 检索召回 | 查询 + 相关文档标注 |
| `rerank_pairs.jsonl` | 24 | Reranker 排序 | query-doc 对 + 相关度 |
| `clustering_sample.jsonl` | 20 | 聚类 | 文本 + 类别标签 |

---

## 4. 测试脚本说明

所有脚本位于 `benchmark/embedding_reranker/scripts/`。

### 4.1 脚本总览

| 脚本 | 功能 | 输出 |
|------|------|------|
| `benchmark_embedding.py` | Embedding 功能测试（内置数据集） | `results/embedding_*.json` |
| `benchmark_reranker.py` | Reranker 功能测试（内置数据集） | `results/reranker_*.json` |
| `run_suite.py` | 批量跑多模型组合（内置数据集） | `results/comparison_report_*.md` |
| `mteb_benchmark.py` | MTEB 官方数据集轻量评测 | `results/mteb_*.json` |
| `summarize_mteb.py` | 汇总 MTEB JSON 结果为 Markdown | `results/mteb_report.md` |
| `e2e_rag_eval.py` | 端到端 RAG 评测（Retrieval + Reranker） | `results/e2e_rag_*.json` |

### 4.2 配置系统

- 模型配置：`scripts/config/models/*.yaml`
- 组合配置：`scripts/config/suites/default.yaml`
- 支持通过 `--config <config_name>` 或 `--model <model_id>` 调用

---

## 5. 常用测试命令

### 5.1 环境准备

```bash
cd benchmark/embedding_reranker
source ~/mlx-env/bin/activate

# 离线运行 MTEB 时必须设置
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 下载缺失数据集时建议用 HF Mirror
export HF_ENDPOINT=https://hf-mirror.com
```

### 5.2 内置数据集功能测试

```bash
# Embedding 单模型测试
python scripts/benchmark_embedding.py \
  --config qwen3_embedding_0.6b_4bit_dwq \
  --datasets ./datasets \
  --output results/embedding_0.6b_4bit.json

# Reranker 单模型测试
python scripts/benchmark_reranker.py \
  --config qwen3_reranker_0.6b_4bit \
  --dataset ./datasets/rerank_pairs.jsonl \
  --output results/reranker_0.6b_4bit.json

# 批量跑默认 4 个 suite
python scripts/run_suite.py --suite default --output-dir ./results
```

### 5.3 MTEB 官方数据集测试

```bash
# Embedding: SciFact（检索）
python scripts/mteb_benchmark.py \
  --config qwen3_embedding_0.6b_4bit_dwq \
  --tasks SciFact \
  --max-length 256 \
  --output results/mteb_embedding_0.6b_4bit_scifact.json

# Embedding: NFCorpus（检索）
python scripts/mteb_benchmark.py \
  --config qwen3_embedding_0.6b_4bit_dwq \
  --tasks NFCorpus \
  --max-length 256 \
  --output results/mteb_embedding_0.6b_4bit_nfcorpus.json

# Embedding: STSBenchmark（STS）
python scripts/mteb_benchmark.py \
  --config qwen3_embedding_0.6b_4bit_dwq \
  --tasks STSBenchmark \
  --output results/mteb_embedding_0.6b_4bit_stsb.json

# Reranker: AskUbuntuDupQuestions（bi-encoder）
python scripts/mteb_benchmark.py \
  --config qwen3_reranker_0.6b_mxfp8 \
  --tasks AskUbuntuDupQuestions \
  --output results/mteb_reranker_0.6b_mxfp8_askubuntu.json

# 汇总 MTEB 结果
python scripts/summarize_mteb.py \
  --results-dir ./results \
  --output results/mteb_report.md
```

### 5.4 端到端 RAG 评测

```bash
python scripts/e2e_rag_eval.py \
  --embedding-config qwen3_embedding_0.6b_4bit_dwq \
  --reranker-config qwen3_reranker_0.6b_4bit \
  --task SciFact \
  --max-length 256 \
  --top-k-retrieve 20 \
  --top-k-rerank 10 \
  --output results/e2e_rag_scifact_low_memory_4bit.json
```

**参数说明**：
- `--top-k-retrieve`：Embedding 召回候选数
- `--top-k-rerank`：Reranker 最终返回数
- `--max-length`：Embedding 模型编码最大长度

### 5.5 下载缺失数据集

```bash
export HF_ENDPOINT=https://hf-mirror.com

python - <<'PY'
from datasets import load_dataset

# 刷新 ArguAna / SCIDOCS 为 MTEB 2.16 所需格式
for task in ['arguana', 'scidocs']:
    load_dataset(f'mteb/{task}', 'corpus')
    load_dataset(f'mteb/{task}', 'queries')

# 下载 Banking77 / TRECCOVID（如需要）
load_dataset('mteb/banking77')
load_dataset('mteb/trec-covid', 'corpus')
load_dataset('mteb/trec-covid', 'queries')
PY
```

---

## 6. 已产生的结果文件

### 6.1 测试报告

| 文件 | 内容 |
|------|------|
| `results/comparison_report_default.md` | 内置数据集 4 个 suite 对比报告 |
| `results/mteb_report_final_v2.md` | MTEB 得分汇总表 |
| `results/qwen3_mlx_gap_analysis.md` | 调研文档与现状差距分析 |
| `results/qwen3_mlx_evaluation_summary.md` | 测试结果、问题清单与后续计划 |
| `results/qwen3_mlx_comparison_and_e2e_report.md` | **与官方基准对比 + 端到端 RAG 分析报告** |

### 6.2 关键 JSON 结果

| 文件 | 内容 |
|------|------|
| `mteb_embedding_0.6b_4bit_scifact_m256_v2.json` | 0.6B 4bit SciFact nDCG@10=0.6867 |
| `mteb_embedding_0.6b_mxfp8_scifact_m256.json` | 0.6B mxfp8 SciFact nDCG@10=0.6849 |
| `mteb_embedding_0.6b_4bit_nfcorpus_m256.json` | 0.6B 4bit NFCorpus nDCG@10=0.3430 |
| `mteb_embedding_0.6b_mxfp8_nfcorpus_m256.json` | 0.6B mxfp8 NFCorpus nDCG@10=0.3469 |
| `mteb_embedding_0.6b_4bit_stsb_fixed.json` | 0.6B 4bit STSBenchmark Spearman=0.8451 |
| `mteb_embedding_0.6b_mxfp8_stsb.json` | 0.6B mxfp8 STSBenchmark Spearman=0.8453 |
| `mteb_embedding_4b_4bit_stsb.json` | 4B 4bit STSBenchmark Spearman=0.8692 |
| `mteb_reranker_0.6b_mxfp8_askubuntu.json` | 0.6B mxfp8 bi-encoder reranker AskUbuntu MAP@20=0.4275 |
| `e2e_rag_scifact_low_memory_4bit.json` | 端到端 RAG：retrieval nDCG@10=0.6838，+reranker 后 0.3044（见报告分析） |

---

## 7. 已知限制与注意事项

### 7.1 模型/速度

- **0.6B 4bit** 是本地最实用的配置：Retrieval 约 10–11 docs/s（max_length=256）。
- **4B 4bit** 跑 STS 可行（~5–6 分钟），但跑 Retrieval 时本地 Mac 容易卡顿/内存不足。
- **8B mxfp8** 未在本地跑全量 Retrieval，预计非常慢。

### 7.2 数据集兼容性

- MTEB 2.16 的 Retrieval 任务需要数据集同时包含 `corpus`、`queries`、`default` 三个配置。
- SciFact、NFCorpus、FiQA2018 已刷新为完整格式。
- ArguAna、SCIDOCS 等旧缓存只有 `default`，需要删除后重新下载。

### 7.3 Reranker 实现问题

- `Qwen3-Reranker-0.6B-4bit` 是官方 cross-encoder，效果应最好。
- 但当前 `benchmark_reranker.py` 在批处理不同长度序列时未传入 `attention_mask`，导致 padding token 干扰分数。
- **端到端 RAG 中 cross-encoder 阶段分数异常下降**，建议在修复前谨慎使用；bi-encoder reranker 无此问题但精度较低。

### 7.4 离线运行

- 所有 MTEB 测试必须设置 `HF_DATASETS_OFFLINE=1`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。
- 首次下载数据集需稳定网络，建议使用 `HF_ENDPOINT=https://hf-mirror.com`。

---

## 8. 快速复现清单

```bash
# 1. 进入目录并激活环境
cd benchmark/embedding_reranker
source ~/mlx-env/bin/activate

# 2. 设置离线环境变量
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 3. 跑一个快速冒烟测试
python scripts/benchmark_embedding.py \
  --config qwen3_embedding_0.6b_4bit_dwq \
  --datasets ./datasets \
  --output results/embedding_0.6b_4bit_smoke.json

# 4. 跑一个 MTEB Retrieval 测试（约 10 分钟）
python scripts/mteb_benchmark.py \
  --config qwen3_embedding_0.6b_4bit_dwq \
  --tasks SciFact \
  --max-length 256 \
  --output results/mteb_scifact_0.6b_4bit.json

# 5. 汇总结果
python scripts/summarize_mteb.py --results-dir ./results --output results/mteb_report.md
```

---

## 9. 参考链接

- Qwen3 Embedding 官方博客：https://qwenlm.github.io/blog/qwen3-embedding/
- Qwen3 Embedding GitHub：https://github.com/QwenLM/Qwen3-Embedding
- MTEB Leaderboard 结果仓库：https://github.com/embeddings-benchmark/results
- mlx-embeddings GitHub：https://github.com/Blaizzy/mlx-embeddings
