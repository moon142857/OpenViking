# Embedding / Reranker 模型基准测试

本目录包含一套可复用的 Embedding 和 Reranker 模型评测方案，基于 MLX 本地推理，用于对 Qwen3 Embedding / Reranker 系列（0.6B / 4B / 8B）进行功能与性能测试。

> 设计目标：作为基准（baseline），后续更换更大模型时，可用同一套方法做公平对比。脚本已通过 loader 自动识别模型格式；其中所有 `mlx-community/Qwen3-Embedding-*` 模型统一使用 `mlx-embeddings` 加载以获得正确的 pooling 输出。`mlx_lm` 仅用于 `Qwen3-Reranker-*-4bit` cross-encoder。

---

## 目录结构

```
benchmark/embedding_reranker/
├── README.md                       # 本说明文档
├── requirements.txt                # Python 依赖
├── datasets/                       # 内置测试数据集
│   ├── sts_sample.jsonl            # 语义相似度（STS）
│   ├── paraphrase_sample.jsonl     # 复述检测
│   ├── retrieval_corpus.jsonl      # 检索语料
│   ├── retrieval_queries.jsonl     # 检索查询
│   ├── rerank_pairs.jsonl          # 重排序 query-doc 对
│   └── clustering_sample.jsonl     # 聚类
├── scripts/                        # 测试脚本
│   ├── benchmark_embedding.py      # Embedding 功能测试（配置驱动）
│   ├── benchmark_reranker.py       # Reranker 功能测试（配置驱动）
│   ├── run_suite.py                # 批量运行多模型组合并生成对比报告
│   ├── mteb_benchmark.py           # MTEB 官方数据集轻量评测
│   ├── summarize_mteb.py           # 汇总 MTEB JSON 结果生成报告
│   └── config/                     # 模型与测试组合配置
│       ├── models/                 # 单个模型配置
│       │   ├── qwen3_embedding_0.6b_4bit_dwq.yaml
│       │   ├── qwen3_embedding_0.6b_mxfp8.yaml
│       │   ├── qwen3_embedding_4b_4bit_dwq.yaml
│       │   ├── qwen3_embedding_8b_mxfp8.yaml
│       │   ├── qwen3_reranker_0.6b_4bit.yaml
│       │   ├── qwen3_reranker_0.6b_mxfp8.yaml
│       │   └── qwen3_reranker_8b_mxfp8.yaml
│       └── suites/                 # 测试组合配置
│           └── default.yaml
└── results/                        # 测试结果
    ├── embedding_*.json
    ├── reranker_*.json
    ├── performance_results.json
    └── comparison_report_*.md
```

---

## 1. 主流评测方法调研

### 1.1 Embedding 模型主流评测

| 评测方向 | 代表基准 | 常用指标 | 说明 |
|----------|---------|---------|------|
| 语义文本相似度 | STS-B, SICK-R, STS12-16 | Spearman correlation | 衡量向量余弦相似度与人类打分的一致性 |
| 复述检测 | Quora Question Pairs, PAWS | Accuracy / F1 | 判断两个句子是否同义 |
| 信息检索 | MS MARCO, BEIR, NQ | Recall@k, MRR, nDCG@k | 用向量召回相关文档 |
| 聚类 | 20 Newsgroups, StackExchange | V-measure, NMI | 按主题聚类 |
| 分类 | Amazon Reviews, Banking77 | Accuracy / F1 | 句子分类 |
| 跨语言 | XTREME, LaBSE eval | 多语言对齐指标 | 多语言语义对齐 |

### 1.2 Reranker 模型主流评测

| 评测方向 | 代表基准 | 常用指标 | 说明 |
|----------|---------|---------|------|
| passage 重排序 | MS MARCO passage ranking | MRR@10, nDCG@10 | 对检索结果重新排序 |
| 成对排序 | MTEB Reranking | MAP, NDCG | query-doc 对相关性排序 |
| 相关性校准 | 人工标注数据集 | Spearman / Pearson | 模型分数与人类标注的相关性 |

### 1.3 本方案设计思路

为了兼顾**可复现性**和**本地运行效率**，本方案没有直接跑完整的 MTEB/BEIR（下载和运行成本高），而是：

1. 内置小样本数据集（中英双语），覆盖主流任务类型；
2. 使用与 OpenViking 服务一致的 `mlx_lm` / `mlx-embeddings` 推理代码；
3. 输出结构化 JSON，方便后续与更大模型做横向对比；
4. 性能测试覆盖单条延迟、batch 吞吐、序列长度扩展。

> 如需更权威的评测，可将本脚本中的数据集替换为 MTEB/BEIR 子集，指标计算逻辑保持不变。

---

## 2. 环境准备

```bash
# 进入目录
cd benchmark/embedding_reranker

# 激活 MLX 环境（与 OpenViking 部署共用）
source ~/mlx-env/bin/activate

# 安装依赖
pip install -r requirements.txt

# 额外安装 mlx-embeddings（测试 mxfp8 模型必需）
pip install mlx-embeddings
```

> 国内用户可设置 HF 镜像加速模型下载：
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```
>
> 如果模型已经通过 OpenViking 部署时下载到本地（例如 `~/.cache/huggingface/hub/models--mlx-community--Qwen3-Embedding-8B-mxfp8`），可以直接传入本地路径，避免重复从 Hub 下载。

---

## 3. 运行测试

### 3.1 批量运行多模型对比（推荐）

```bash
python scripts/run_suite.py
```

默认读取 `scripts/config/suites/default.yaml` 中的 4 组组合，依次运行 Embedding 和 Reranker 评测，最终生成：
- `results/embedding_<suite>.json`
- `results/reranker_<suite>.json`
- `results/comparison_report_default.md`

### 3.2 单独运行 Embedding 评测（配置驱动）

```bash
python scripts/benchmark_embedding.py --config qwen3_embedding_8b_mxfp8
```

或直接用 model_id（自动检测 loader）：

```bash
python scripts/benchmark_embedding.py \
  --model mlx-community/Qwen3-Embedding-4B-4bit-DWQ \
  --datasets ./datasets \
  --output ./results/embedding_results.json \
  --pooling last \
  --normalize
```

关键参数：

- `--config`：指定 `config/models/` 下的模型配置名（如 `qwen3_embedding_8b_mxfp8`）或 YAML 文件路径
- `--model`：直接传入 model_id 或本地路径（脚本自动从路径推断 loader）
- `--pooling last`：使用官方推荐的 **last-token pooling**（默认）
- `--pooling mean`：使用 mean pooling（旧版行为，便于对比；仅 mlx_lm 路径有效）
- `--normalize` / `--no-normalize`：是否做 L2 归一化（官方推荐归一化；mlx-embeddings 路径总是归一化）
- `--instruction`：query 前的任务指令，默认 `"Given a web search query, retrieve relevant passages that answer the query"`

不同模型的加载器选择（由配置决定）：

| 模型 | 转换工具 | 推荐加载器 | 脚本行为 |
|------|----------|------------|----------|
| `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` | mlx-lm | **mlx-embeddings** | auto |
| `mlx-community/Qwen3-Embedding-4B-4bit-DWQ` | mlx-lm | **mlx-embeddings** | auto |
| `mlx-community/Qwen3-Embedding-0.6B-mxfp8` | mlx-embeddings | mlx-embeddings | auto |
| `mlx-community/Qwen3-Embedding-8B-mxfp8` | mlx-embeddings | mlx-embeddings | auto |
| `mlx-community/Qwen3-Reranker-0.6B-4bit` | mlx-lm | mlx_lm | auto |
| `mlx-community/Qwen3-Reranker-*-mxfp8` | mlx-embeddings | mlx-embeddings | auto |

示例：

```bash
# 推荐：使用配置名
python scripts/benchmark_embedding.py --config qwen3_embedding_8b_mxfp8

# 0.6B 低内存
python scripts/benchmark_embedding.py \
  --model mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ \
  --datasets ./datasets \
  --output ./results/embedding_results_0.6b.json \
  --pooling last --normalize

# 4B 平衡
python scripts/benchmark_embedding.py \
  --model mlx-community/Qwen3-Embedding-4B-4bit-DWQ \
  --datasets ./datasets \
  --output ./results/embedding_results_4b.json \
  --pooling last --normalize

# 8B mxfp8 最强（注意：mlx-embeddings 路径总是 last-token + 归一化）
python scripts/benchmark_embedding.py \
  --model mlx-community/Qwen3-Embedding-8B-mxfp8 \
  --datasets ./datasets \
  --output ./results/embedding_results_8b.json
```

### 3.4 单独运行 Reranker 评测（配置驱动）

```bash
python scripts/benchmark_reranker.py \
  --model mlx-community/Qwen3-Reranker-0.6B-4bit \
  --dataset ./datasets/rerank_pairs.jsonl \
  --output ./results/reranker_results.json
```

脚本默认使用**官方 Qwen3-Reranker prompt** 和 `yes/no` softmax 概率打分，无需手动选择模式：

```text
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {instruction}
<Query>: {query}
<Document>: {doc}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

#### Reranker 评测模式

脚本根据配置中的 `reranker.mode` 选择打分方式：

| 模式 | 适用模型 | 说明 |
|------|----------|------|
| `cross_encoder` | `Qwen3-Reranker-0.6B-4bit` | 官方用法：构造 Qwen3-Reranker prompt，取最后一个 token 的 `no/yes` logit 做 softmax |
| `embedding_similarity` | `Qwen3-Reranker-*-mxfp8` | 官方 mlx-embeddings 用法：把 query 和 document 分别 embed，用 cosine similarity 作为相关性分数 |

> ⚠️ `mlx-community/Qwen3-Reranker-*-mxfp8` 系列在 mlx-embeddings 转换后变成了 bi-encoder 形态，不再保留原版的 cross-encoder `lm_head`。因此不能按 `no/yes` logit 打分，必须按官方示例使用 embedding similarity。

示例：

```bash
# 推荐：使用配置名
python scripts/benchmark_reranker.py --config qwen3_reranker_0.6b_4bit

# 0.6B 低内存
python scripts/benchmark_reranker.py \
  --model mlx-community/Qwen3-Reranker-0.6B-4bit \
  --dataset ./datasets/rerank_pairs.jsonl \
  --output ./results/reranker_results_0.6b.json

# 8B mxfp8（会自动 fallback 到 mlx-embeddings）
python scripts/benchmark_reranker.py \
  --model mlx-community/Qwen3-Reranker-8B-mxfp8 \
  --dataset ./datasets/rerank_pairs.jsonl \
  --output ./results/reranker_results_8b.json
```

### 3.5 单独运行性能评测

> ⚠️ 性能脚本 `benchmark_performance.py` 已暂存到 `bak/`，目前仍使用旧版硬编码逻辑（`mlx_lm` + mean pooling），对 Qwen3 Embedding 模型会得出不可靠结果。待后续更新为配置驱动 + `mlx-embeddings` 后再移回 `scripts/`。

### 3.6 运行 MTEB 官方数据集轻量评测

```bash
# Embedding 模型（以 STSBenchmark 为例）
python scripts/mteb_benchmark.py \
  --config qwen3_embedding_0.6b_4bit_dwq \
  --tasks STSBenchmark \
  --output ./results/mteb_embedding_0.6b_4bit.json

# Reranker 模型（bi-encoder，以 AskUbuntuDupQuestions 为例）
python scripts/mteb_benchmark.py \
  --config qwen3_reranker_0.6b_mxfp8 \
  --tasks AskUbuntuDupQuestions \
  --output ./results/mteb_reranker_0.6b_mxfp8_askubuntu.json
```

批量跑完多个 `results/mteb_*.json` 后，汇总为 Markdown：

```bash
python scripts/summarize_mteb.py \
  --results-dir ./results \
  --output ./results/mteb_report.md
```

> 注意：MTEB 任务在 Apple Silicon 本地运行较慢（一个 STS 任务约 1–4 分钟，一个 Reranking 任务约 5–10 分钟），大模型 / Retrieval 任务建议 GPU 环境或分批运行。

---

## 4. 测试方法详解

### 4.1 Embedding 功能测试

本脚本按照官方 Qwen3-Embedding 推荐用法实现：

1. **Query 加 instruction**：`Instruct: {task}\nQuery:{text}`
2. **Document 不加 instruction**（仅检索任务）
3. **Tokenizer `padding_side='left'`**（mlx_lm 路径）
4. **Last-token pooling**：取最后一个有效 token 的 hidden state
5. **L2 归一化**：`embeddings = embeddings / ||embeddings||_2`
6. **相似度**：归一化后向量 cosine（等价于点积）

> 旧版脚本使用 mean pooling 且未加 instruction，与官方用法不一致，会导致分数偏低。

#### STS（Semantic Textual Similarity）

- 输入：句子对 + 人工相似度分数（0-5）
- 方法：两个句子都加 instruction 后编码，计算余弦相似度，缩放到 0-5
- 指标：**Spearman correlation**

#### 复述检测（Paraphrase Detection）

- 输入：句子对 + 是否复述标签（0/1）
- 方法：两个句子都加 instruction 后编码，用余弦相似度做二分类
- 指标：**Accuracy / F1**

#### 检索召回（Retrieval Recall）

- 输入：语料库 + 查询 + 相关文档标注
- 方法：查询加 instruction，文档不加 instruction，分别编码后按余弦相似度排序
- 指标：**Recall@1 / Recall@5 / Recall@10 / MRR**

#### 聚类（Clustering）

- 输入：带类别标签的句子
- 方法：加 instruction 后编码，再用类别中心做最近邻分类
- 指标：**V-measure**

### 4.2 Reranker 功能测试

#### 成对排序准确率（Pairwise Accuracy）

- 输入：同一查询下的相关文档和无关文档
- 方法：比较 reranker 分数，看相关文档是否更高
- 指标：**Accuracy**

#### NDCG@10

- 输入：同一查询下的多个 query-doc 对 + 分级相关度（0-3）
- 方法：按 reranker 分数排序，计算 DCG / IDCG
- 指标：**NDCG@10**

#### 相关性校准（Correlation）

- 输入：所有 query-doc 对 + 分级相关度
- 方法：计算 reranker 分数与人工标签的 Spearman 相关
- 指标：**Spearman correlation**

> 本方案对 `cross_encoder` 模型使用 `yes` 与 `no` token 的 logit softmax 概率作为相关性分数。
> 对 `embedding_similarity` 模型使用 query 与 document embedding 的 cosine similarity 作为相关性分数。

### 4.3 性能测试

> ⚠️ 性能脚本 `benchmark_performance.py` 尚未更新为配置驱动 + `mlx-embeddings`，当前结果不可靠。下表描述的是脚本目标，而非当前可用状态。

| 测试项 | Embedding | Reranker |
|--------|-----------|----------|
| 单条延迟 | 1 条句子 | 1 个 query-doc 对 |
| Batch 吞吐 | batch=1,8,16,32 | batch=1,8,16,32 |
| 序列长度扩展 | 128/512/1024/2048 | - |

---

## 5. 推荐的模型组合

可直接在 `scripts/config/suites/default.yaml` 中增删组合，或用 `--config` 参数单独指定模型配置。

| 场景 | Embedding 配置 | Reranker 配置 | 说明 |
|------|----------------|---------------|------|
| 低内存 / 高吞吐（推荐） | `qwen3_embedding_0.6b_4bit_dwq` | `qwen3_reranker_0.6b_4bit` | 速度快、内存小，reranker 为官方 cross-encoder |
| 低内存 + 更高 embedding 精度 | `qwen3_embedding_0.6b_mxfp8` | `qwen3_reranker_0.6b_4bit` | mxfp8 embedding 精度优于 4bit，reranker 仍用 4bit cross-encoder |
| 平衡 | `qwen3_embedding_4b_4bit_dwq` | `qwen3_reranker_0.6b_4bit` | 4B embedding + 轻量 reranker |
| 最强精度（实验性） | `qwen3_embedding_8b_mxfp8` | `qwen3_reranker_8b_mxfp8` | 8B reranker 为 bi-encoder，分数仅供参考 |

> ⚠️ `qwen3_reranker_0.6b_mxfp8` 在 bi-encoder 模式下实测表现很差（内置数据集 Spearman 为负），不推荐在生产场景使用。建议低内存 reranker 仍选 `qwen3_reranker_0.6b_4bit`。

---

## 6. 已知问题

1. **所有 mlx-community/Qwen3-Embedding-* 模型必须用 mlx-embeddings 加载**
   - 若用 `mlx_lm` 直接加载 4bit-DWQ 或 mxfp8 模型，权重映射/输出异常，在 MTEB STSBenchmark 上 Spearman 会掉到 ~0.09。
   - 脚本已自动识别并切换加载器；配置文件中 `loader` 也已统一为 `mlx_embeddings`。

2. **mlx-community/Qwen3-Reranker-*-mxfp8 是 bi-encoder 形态**
   - 官方 mlx-embeddings 用法是分别 embed query/document 再算相似度，不是 cross-encoder 的 `no/yes` logit 打分。
   - 脚本已按官方用法实现 `embedding_similarity` 模式。
   - 实测 `Qwen3-Reranker-0.6B-mxfp8` 在此模式下表现较差，可能是转换问题；`Qwen3-Reranker-8B-mxfp8` 可用但 Spearman 仍弱于 0.6B-4bit cross-encoder。

3. **小样本数据集限制**
   - 内置数据集仅 20-30 条样本，用于快速冒烟和横向对比，不代表真实业务性能。

---

## 7. 扩展建议

1. **扩展 MTEB 任务：** 当前已通过 `scripts/mteb_benchmark.py` 接入 MTEB；可继续补充 SciFact、FiQA2018、SciDocsRR 等任务，大模型建议在 GPU 环境运行。
2. **增加中文评测：** 当前数据集已包含中英双语，可进一步补充 CLUE、LCQMC 等中文任务。
3. **增加内存监控：** 在性能脚本中加入 `mx.metal.get_peak_memory()` 或 `torch.mps` 内存统计。
4. **校准 Reranker 分数：** 对更大模型可尝试温度缩放或 Platt scaling 改善 Spearman correlation。

---

*最后更新: 2026-06-26（新增 MTEB 脚本；统一 Qwen3 Embedding 使用 mlx-embeddings loader；清理旧结果到 bak/）*
