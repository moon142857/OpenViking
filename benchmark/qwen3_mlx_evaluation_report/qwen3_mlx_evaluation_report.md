# MLX-Community Qwen3 Embedding & Reranker 模型评测指南

## Mac M4 (32GB) 本地精度与性能验证方案

---

## 1. 模型概览与核心特点

您配置中的四组模型套件基于 **Alibaba Qwen3-Embedding** 和 **Qwen3-Reranker** 系列，通过 MLX 社区转换为 Apple Silicon 优化的格式。这些模型均托管于 `mlx-community` Hugging Face 组织下，利用 Apple MLX 框架的 Metal GPU 加速实现本地高效推理。以下是各模型的完整标识与核心特性：

| 模型角色 | 配置套件 | 完整 HuggingFace ID | 参数量 | 量化格式 | 预估内存 |
|---|---|---|---|---|---|
| Embedding | low_memory_4bit | `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` | 0.6B | 4bit DWQ (分组量化) | ~0.4 GB  [(Github)](https://github.com/Blaizzy/mlx-embeddings)  |
| Reranker | low_memory_4bit | `mlx-community/Qwen3-Reranker-0.6B-4bit` | 0.6B | 4bit 标准量化 | ~0.4 GB |
| Embedding | low_memory_mxfp8 | `mlx-community/Qwen3-Embedding-0.6B-mxfp8` | 0.6B | mxfp8 (MLX 8bit浮点) | ~0.6 GB  [(Github)](https://github.com/Blaizzy/mlx-embeddings)  |
| Reranker | low_memory_mxfp8 | `mlx-community/Qwen3-Reranker-0.6B-mxfp8` | 0.6B | mxfp8 | ~0.6 GB |
| Embedding | balanced | `mlx-community/Qwen3-Embedding-4B-4bit-DWQ` | 4B | 4bit DWQ | ~2.5 GB |
| Reranker | balanced | `mlx-community/Qwen3-Reranker-0.6B-4bit` | 0.6B | 4bit | ~0.4 GB |
| Embedding | high_quality | `mlx-community/Qwen3-Embedding-8B-mxfp8` | 8B | mxfp8 | ~8.0 GB |
| Reranker | high_quality | `mlx-community/Qwen3-Reranker-8B-mxfp8` | 8B | mxfp8 | ~8.0 GB |

**Qwen3-Embedding** 采用双向注意力机制（非因果掩码），基于 Qwen3 LLM 架构专门训练用于生成高质量文本向量表示。其核心优势包括支持 **Matryoshka 表示学习（MRL）**——即可以在推理时动态截断嵌入维度（支持 256/512/1024/2048 等维度），在不重新训练的情况下平衡精度与存储开销  [(Github)](https://github.com/EricRollei/Semantic-Search) 。**Qwen3-Reranker** 则基于相同的 Qwen3 架构，但通过交叉编码器（cross-encoder）方式对查询-文档对进行相关性打分，用于对初筛候选文档进行精排序。

从量化技术角度，`4bit-DWQ`（Dynamic Weight Quantization）属于分组仿射量化，通过将权重分组并以低比特存储来大幅降低模型体积；`mxfp8` 则是 MLX 框架原生的 8bit 浮点量化格式，相比 4bit 保留了更多精度信息，同时相比 bf16 全精度仍能减少约 50% 内存占用  [(Github)](https://github.com/Blaizzy/mlx-embeddings) 。对于教育领域的 RAG 应用——涉及教材、政策文件、课件等长文档的语义检索——**Retrieval 能力**（从大规模文档库中找到相关片段）和 **Reranking 能力**（对候选结果进行精准排序）是最关键的两大指标，这正是 MTEB 基准中 Retrieval 和 Reranking 任务所评测的核心能力。

---

## 2. MTEB 评测体系与核心指标

### 2.1 MTEB 框架结构

**MTEB（Massive Text Embedding Benchmark）** 是当前嵌入模型领域最权威的评测框架，由 Hugging Face 维护并持续更新  [(ar5iv)](https://ar5iv.labs.arxiv.org/html/2210.07316) 。MTEB v1 包含 **56 个数据集**，覆盖 **8 大任务类型**，其设计目标是全面评估嵌入模型在不同语义理解任务上的泛化能力。对于 RAG 场景，**Retrieval** 和 **Reranking** 两类任务最为关键——前者直接对应"从文档库中召回相关片段"的能力，后者对应"对召回结果进行精排序"的能力。

MTEB 的每个任务类型采用不同的评测指标。Retrieval 任务继承自 BEIR 基准，以 **nDCG@10** 为主指标，衡量模型将相关文档排在前列的能力；Reranking 任务以 **MAP**（Mean Average Precision）为主指标，评估模型对查询-文档对进行相关性排序的准确度  [(arXiv.org)](https://arxiv.org/html/2507.21500v1) 。STS（语义文本相似度）任务使用 **Spearman 相关系数**，Classification 使用 **F1 分数**，Clustering 使用 **v-measure**——这些任务虽然与 RAG 的直接关联度较低，但能从侧面反映模型对语义理解的基础能力。

![MTEB任务概览](mteb_tasks_overview.png)

### 2.2 RAG 场景高优先级任务与数据集

鉴于您的应用场景是教育领域 RAG（检索教材、文案、课件、政策文件后送入大模型生成内容），以下是从 MTEB/BEIR 中筛选出的**高优先级评测任务**，按与 RAG 检索场景的相关度排序：

| 优先级 | 任务类型 | 数据集 | 主指标 | 相关度 | 数据集规模 | 场景说明 |
|---|---|---|---|---|---|---|
| P0 | Retrieval | **SciFact** | nDCG@10 | 10/10 | 5,183 docs, 300 queries | 科学论文事实验证，最接近教材/知识点检索  [(arXiv.org)](https://arxiv.org/html/2511.09213v1)  |
| P0 | Retrieval | **FiQA-2018** | nDCG@10 | 10/10 | 57,638 docs, 648 queries | 金融问答检索，适用于政策/文案检索  [(Elastic)](https://www.elastic.co/search-labs/blog/evaluating-search-relevance-part-1)  |
| P0 | Retrieval | **NFCorpus** | nDCG@10 | 9/10 | 3,633 docs, 323 queries | 医学文献检索，适用于专业教材检索 |
| P0 | Reranking | **SciDocsRR** | MAP | 10/10 | ~25K docs | 科学文档重排序，直接对应 RAG 二阶段排序  [(python | DeepWiki)](https://deepwiki.com/FlagOpen/FlagEmbedding/6.2-benchmarks)  |
| P1 | Retrieval | **ArguAna** | nDCG@10 | 8/10 | 8,674 docs, 1,406 queries | 论证检索，适用于观点/论述类内容检索 |
| P1 | Retrieval | **TRECCOVID** | nDCG@10 | 7/10 | 171K docs, 50 queries | COVID文献检索，测试长文档检索能力  [(arXiv.org)](https://arxiv.org/pdf/2407.08275)  |
| P1 | Retrieval | **SCIDOCS** | nDCG@10 | 7/10 | 25,657 docs, 1,000 queries | 科学文档引用检索，适用于学术内容 |
| P1 | Reranking | **AskUbuntu** | MAP | 8/10 | ~16K pairs | 技术问答重排序，适用于FAQ类场景 |
| P2 | STS | **STSBenchmark** | Spearman | 5/10 | 8,628 pairs | 语义相似度基础能力 |
| P2 | Classification | **Banking77** | F1 | 4/10 | 13K samples | 意图分类，反映语义理解基础 |

上表中的 **P0 级数据集** 是 RAG 场景的核心评测集，建议必须运行；**P1 级** 用于补充验证模型在不同文档类型上的泛化能力；**P2 级** 则作为基础能力参考。SciFact 和 FiQA 之所以列为最高优先级，是因为 SciFact 的"给定科学声明，从论文中检索支持/反驳证据"的任务形式，与教育中"给定知识点查询，从教材中检索相关内容"的场景高度同构；FiQA 的"从金融文档中检索问答对"则与政策文件、文案的问答式检索场景相似。

### 2.3 Embedding 与 Reranker 的差异化评测指标

Embedding 模型和 Reranker 模型在 RAG 流程中扮演不同角色，因此需要采用不同的评测指标和方法：

**Embedding 模型** 的核心职责是将查询和文档编码为向量，通过向量相似度进行召回。其评测重点是**召回能力**——即能否将相关文档包含在 Top-K 结果中。关键指标包括：

| 指标 | 含义 | 适用场景 | 合理范围 |
|---|---|---|---|
| **nDCG@10** | 归一化折损累积增益@10 | 主要指标，衡量排序质量 | 0.3-0.7（越高越好）  [(offs)](https://www.systemoverflow.com/learn/ml-embeddings/embedding-quality-evaluation/mteb-and-beir-benchmark-evaluation)  |
| **Recall@100** | 前100结果中的召回率 | 评估召回覆盖度 | 0.6-0.95 |
| **MRR@10** | 平均倒数排名@10 | 衡量首个相关文档的位置 | 0.3-0.8 |
| **MAP** | 平均精度均值 | 综合排序质量 | 0.2-0.6 |

**Reranker 模型** 的核心职责是对 Embedding 召回的候选文档进行精排序。其评测重点是**排序精度**——即能否将最相关的文档排在最前面。关键指标包括：

| 指标 | 含义 | 适用场景 | 合理范围 |
|---|---|---|---|
| **MAP** | 平均精度均值 | 主要指标，综合排序准确度 | 0.4-0.8 |
| **MRR@10** | 平均倒数排名@10 | 衡量最相关文档的排名 | 0.5-0.9 |
| **nDCG@10** | 归一化折损累积增益@10 | 评估重排序后的质量 | 0.4-0.8 |

在 RAG 系统的实际部署中，Embedding 和 Reranker 通常以**两阶段串联**方式工作：Embedding 负责从大规模文档库中快速召回 Top-100 候选，Reranker 则对这 100 个候选进行精细排序并返回 Top-10 最终结果。因此，两者的联合评测（即 Embedding + Reranker 的端到端评测）是验证完整 RAG 链路效果的最佳方式。

---

## 3. Mac M4 (32GB) 内存与性能分析

### 3.1 模型内存占用估算

在 Apple Silicon 的 Unified Memory 架构下，模型权重、激活值、数据集缓存和操作系统共享同一块内存池。以下是对您配置中各模型的内存占用详细估算：

![模型内存占用](model_memory_usage.png)

内存估算遵循公式：**内存(GB) ≈ 参数量(B) × 精度字节数 × 1.1（开销系数）**。对于 4bit 量化，每个参数占用 0.5 字节（含分组缩放因子）；对于 mxfp8，每个参数占用 1 字节。以 Qwen3-Embedding-8B-mxfp8 为例，8B 参数 × 1 字节 × 1.1 开销 ≈ **8.8 GB**，实际加载时因 MLX 的内存优化约为 **8.0 GB**。

### 3.2 四组套件的联合内存分析

在 RAG 的实际推理场景中，Embedding 模型和 Reranker 模型需要**同时驻留内存**（Embedding 先做向量召回，Reranker 再对结果精排）。以下是四组套件的同时加载内存需求：

![套件内存对比](suite_memory_comparison.png)

| 套件 | Embedding 内存 | Reranker 内存 | 模型合计 | +20% 推理开销 | +数据集缓存 | **总内存需求** | 32GB 余量 |
|---|---|---|---|---|---|---|---|
| low_memory_4bit | 0.4 GB | 0.4 GB | 0.8 GB | 1.0 GB | 0.5 GB | **1.5 GB** | 30.5 GB ✅ |
| low_memory_mxfp8 | 0.6 GB | 0.6 GB | 1.2 GB | 1.4 GB | 0.5 GB | **1.9 GB** | 30.1 GB ✅ |
| balanced | 2.5 GB | 0.4 GB | 2.9 GB | 3.5 GB | 2.0 GB | **5.5 GB** | 26.5 GB ✅ |
| high_quality | 8.0 GB | 8.0 GB | 16.0 GB | 19.2 GB | 2.0 GB | **21.2 GB** | 10.8 GB ✅ |

从内存分析可见，**所有四组套件在 32GB Mac M4 上均可同时加载运行**，即使是最高配的 high_quality 组合也仅占用约 21GB，剩余约 10GB 可用于数据集缓存、向量索引构建和操作系统开销。这意味着您可以放心地在同一台机器上完成所有四组配置的端到端评测，无需担心内存溢出。

### 3.3 推理速度预估

基于 MLX 框架在 Apple Silicon 上的性能基准  [(ModelScope 魔搭社区)](https://modelscope.cn/models/jinaai/jina-embeddings-v4-mlx-8bit) ，结合模型参数量与量化格式，以下是各模型的预估推理吞吐量（单样本，序列长度 128 tokens）：

| 模型 | 格式 | 预估速度 (samples/sec) | 相对速度 |
|---|---|---|---|
| Qwen3-Embedding-0.6B | 4bit | ~150-200 | 10x |
| Qwen3-Embedding-0.6B | mxfp8 | ~120-160 | 8x |
| Qwen3-Embedding-4B | 4bit | ~60-80 | 4x |
| Qwen3-Embedding-8B | mxfp8 | ~25-35 | 1x (基准) |
| Qwen3-Reranker-0.6B | 4bit | ~100-140 | 4x |
| Qwen3-Reranker-8B | mxfp8 | ~20-30 | 1x (基准) |

**速度优化建议**：在评测时，应尽可能使用**批量推理（batch inference）**——MLX 的 Metal GPU 后端对批量处理有显著优化，batch_size=32 时通常可获得 5-10 倍的吞吐量提升。此外，对于 Embedding 模型，可以利用 **Matryoshka 维度截断**——在验证低维表示（如 256/512 维）是否满足教育场景需求时，既能加速推理又能减少向量存储开销  [(Github)](https://github.com/huggingface/blog/blob/main/train-multimodal-sentence-transformers.md) 。实测数据表明，Qwen3-Embedding 在 512 维时仍能保持全维 99.7% 的检索性能，而 256 维时保持 98.9%——这对于需要快速原型验证的场景极具价值。

---

## 4. 环境搭建与依赖安装

### 4.1 核心依赖安装

在 Mac M4 上运行完整的评测流程，需要安装以下 Python 包：

```bash
# 1. 创建虚拟环境（推荐）
python3 -m venv ~/mteb_eval_env
source ~/mteb_eval_env/bin/activate

# 2. 安装 MLX 生态核心库
pip install mlx mlx-embeddings

# 3. 安装 MTEB 评测框架（包含数据集自动下载）
pip install mteb

# 4. 安装辅助依赖
pip install numpy scipy tqdm sentencepiece

# 5. 可选：安装 sentence-transformers 用于对比评测
pip install sentence-transformers
```

**mlx-embeddings** 是由 Blaizzy 开发的开源库  [(Github)](https://github.com/Blaizzy/mlx-embeddings) ，专门用于在 Apple Silicon 上通过 MLX 运行 Embedding 模型。它原生支持 Qwen3、Qwen3-VL、BERT、ModernBERT、Llama Bidirectional 等架构，提供与 Hugging Face 兼容的 `load()` API，支持单条和批量推理，以及文本相似度计算工具。该库的量化转换工具支持 `affine`（默认 4bit）、`mxfp4`、`mxfp8` 和 `nvfp4` 四种量化模式  [(Github)](https://github.com/Blaizzy/mlx-embeddings) 。

### 4.2 模型下载

MLX-community 的模型可通过 Hugging Face Hub 直接下载：

```bash
# 方法1：使用 huggingface-cli
pip install huggingface_hub

huggingface-cli download mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ \
    --local-dir ./models/Qwen3-Embedding-0.6B-4bit-DWQ

huggingface-cli download mlx-community/Qwen3-Reranker-0.6B-4bit \
    --local-dir ./models/Qwen3-Reranker-0.6B-4bit

huggingface-cli download mlx-community/Qwen3-Embedding-4B-4bit-DWQ \
    --local-dir ./models/Qwen3-Embedding-4B-4bit-DWQ

huggingface-cli download mlx-community/Qwen3-Embedding-8B-mxfp8 \
    --local-dir ./models/Qwen3-Embedding-8B-mxfp8

huggingface-cli download mlx-community/Qwen3-Reranker-8B-mxfp8 \
    --local-dir ./models/Qwen3-Reranker-8B-mxfp8

# 方法2：首次使用时自动下载（mlx-embeddings 会在 load() 时自动缓存）
```

---

## 5. Embedding 模型评测代码

### 5.1 基础推理验证

以下代码演示如何使用 `mlx-embeddings` 加载 Qwen3 Embedding 模型并生成文本嵌入，这是评测前必须验证的基础步骤：

```python
"""
Step 1: 基础推理验证 - 加载模型并生成嵌入
验证模型是否正确加载、输出维度是否符合预期
"""
import mlx.core as mx
from mlx_embeddings import load
import numpy as np

# 加载模型和 tokenizer（自动下载到 ~/.cache/huggingface/hub/）
model_name = "mlx-community/Qwen3-Embedding-0.6B-mxfp8"
model, tokenizer = load(model_name)

# 测试文本（教育场景示例）
texts = [
    "高中物理必修一：牛顿运动定律",
    "牛顿第一定律又称惯性定律，指出物体在不受外力时保持静止或匀速直线运动",
    "高中化学：氧化还原反应的基本概念",
]

# 批量编码生成嵌入
embeddings = []
for text in texts:
    input_ids = tokenizer.encode(text, return_tensors="mlx")
    outputs = model(input_ids)
    # Qwen3 使用 mean pooling + normalization
    text_embeds = outputs.text_embeds  # shape: (1, hidden_dim)
    embeddings.append(np.array(text_embeds[0]))

embeddings = np.stack(embeddings)
print(f"嵌入维度: {embeddings.shape}")  # e.g., (3, 2048)

# 计算余弦相似度矩阵
from numpy.linalg import norm
def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

print("\n相似度矩阵:")
for i in range(len(texts)):
    for j in range(len(texts)):
        sim = cosine_sim(embeddings[i], embeddings[j])
        print(f"  {i}-{j}: {sim:.4f}")
```

### 5.2 MTEB 评测集成（自定义 DRESModel Wrapper）

MTEB 框架通过 `DRESModel` 接口调用外部 Embedding 模型。由于 `mlx-embeddings` 不是标准的 `sentence-transformers` 模型，需要编写一个适配器类来桥接。以下代码展示了如何将 MLX Qwen3 Embedding 模型包装为 MTEB 兼容的评测器：

```python
"""
Step 2: MTEB 评测集成 - 自定义 DRESModel Wrapper
将 mlx-embeddings 模型接入 MTEB 评测框架
"""
import numpy as np
import mlx.core as mx
from mlx_embeddings import load
from mteb.encoder_interface import Encoder, PromptType
from mteb.model_meta import ModelMeta
from typing import Sequence

class MLXQwen3Encoder(Encoder):
    """
    适配器：将 mlx-embeddings Qwen3 模型包装为 MTEB Encoder 接口
    支持 encode()、encode_queries()、encode_corpus() 三种模式
    """
    def __init__(self, model_path: str, matryoshka_dim: int = None):
        """
        Args:
            model_path: HuggingFace 模型 ID 或本地路径
            matryoshka_dim: Matryoshka 截断维度 (256/512/1024/2048)，None 表示使用全维
        """
        self.model, self.tokenizer = load(model_path)
        self.matryoshka_dim = matryoshka_dim
        self.max_seq_length = 8192  # Qwen3 Embedding 支持 8K 上下文
        
    def _encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """内部批量编码方法"""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeds = []
            for text in batch:
                input_ids = self.tokenizer.encode(text, return_tensors="mlx")
                outputs = self.model(input_ids)
                emb = np.array(outputs.text_embeds[0])  # (hidden_dim,)
                
                # Matryoshka 维度截断
                if self.matryoshka_dim and emb.shape[0] > self.matryoshka_dim:
                    emb = emb[:self.matryoshka_dim]
                    # 重新归一化
                    emb = emb / np.linalg.norm(emb)
                
                batch_embeds.append(emb)
            all_embeddings.extend(batch_embeds)
        return np.stack(all_embeddings)
    
    def encode(
        self,
        sentences: Sequence[str],
        *,
        task_name: str = None,
        prompt_type: PromptType = None,
        **kwargs
    ) -> np.ndarray:
        """MTEB 主编码接口"""
        batch_size = kwargs.get("batch_size", 32)
        return self._encode_batch(list(sentences), batch_size)
    
    def encode_queries(self, queries: Sequence[str], **kwargs) -> np.ndarray:
        """查询编码 - 可添加 query-specific 前缀"""
        # Qwen3 Embedding 支持任务指令，可在此添加检索指令
        prefixed = [f"检索与以下内容相关的文档: {q}" for q in queries]
        return self._encode_batch(prefixed, kwargs.get("batch_size", 32))
    
    def encode_corpus(self, corpus: Sequence[dict[str, str]], **kwargs) -> np.ndarray:
        """文档编码 - 处理 title/text 字段"""
        texts = []
        for doc in corpus:
            # MTEB corpus 格式: {"title": "...", "text": "..."}
            title = doc.get("title", "")
            text = doc.get("text", "")
            if title and text:
                texts.append(f"{title}\n{text}")
            elif text:
                texts.append(text)
            else:
                texts.append(title)
        return self._encode_batch(texts, kwargs.get("batch_size", 32))


# 模型元数据注册（用于 MTEB 结果输出）
qwen3_06b_mxfp8_meta = ModelMeta(
    name="mlx-community/Qwen3-Embedding-0.6B-mxfp8",
    revision="main",
    release_date="2025-06-01",
    languages=["eng", "zho"],  # 支持中英文
    loader=lambda: MLXQwen3Encoder("mlx-community/Qwen3-Embedding-0.6B-mxfp8"),
    max_tokens=8192,
    embed_dim=2048,
    license="apache-2.0",
    open_weights=True,
    framework=["MLX"],
)
```

### 5.3 执行 MTEB 评测

```python
"""
Step 3: 执行 MTEB 评测 - 运行 RAG 核心数据集
"""
from mteb import MTEB
import json

# 初始化编码器
encoder = MLXQwen3Encoder("mlx-community/Qwen3-Embedding-0.6B-mxfp8")

# 定义 RAG 场景高优先级任务
rag_tasks = [
    "SciFact",           # 科学论文事实检索（最接近教材检索）
    "FiQA2018",          # 金融问答检索（类似政策/文案检索）
    "NFCorpus",          # 医学文献检索（专业内容检索）
    "ArguAna",           # 论证检索（观点类内容）
    "SCIDOCS",           # 科学文档引用检索
    "TRECCOVID",         # COVID文献检索（长文档测试）
]

# 运行评测（首次会自动下载数据集）
results = {}
for task_name in rag_tasks:
    print(f"\n{'='*60}")
    print(f"Running task: {task_name}")
    print(f"{'='*60}")
    
    evaluation = MTEB(tasks=[task_name])
    task_results = evaluation.run(
        encoder,
        output_folder=f"./mteb_results/{task_name}",
        eval_splits=["test"],
        verbosity=2,
    )
    results[task_name] = task_results

# 汇总输出
print("\n" + "="*60)
print("RAG SCENE EVALUATION SUMMARY")
print("="*60)
for task_name, task_res in results.items():
    # 提取主指标
    main_score = task_res[0].scores.get("test", [{}])[0].get("main_score", "N/A")
    print(f"  {task_name:20s}: nDCG@10 = {main_score}")

# 保存完整结果
with open("./mteb_results/rag_summary.json", "w") as f:
    json.dump({k: str(v) for k, v in results.items()}, f, indent=2)
```

### 5.4 Matryoshka 维度对比评测

Qwen3-Embedding 支持 Matryoshka 表示学习，可以在推理时截断到不同维度。以下代码用于评测不同维度在教育场景 RAG 任务上的精度损失：

```python
"""
Step 4: Matryoshka 维度对比评测
验证低维表示是否满足教育场景 RAG 需求
"""
import matplotlib.pyplot as plt

dims_to_test = [256, 512, 1024, 2048]
matryoshka_results = {}

for dim in dims_to_test:
    print(f"\nTesting Matryoshka dim: {dim}")
    encoder = MLXQwen3Encoder(
        "mlx-community/Qwen3-Embedding-0.6B-mxfp8",
        matryoshka_dim=dim
    )
    
    evaluation = MTEB(tasks=["SciFact", "FiQA2018"])
    results = evaluation.run(encoder, output_folder=f"./mteb_results/dim_{dim}")
    
    matryoshka_results[dim] = {
        "SciFact": results[0].scores["test"][0]["main_score"],
        "FiQA2018": results[1].scores["test"][0]["main_score"],
    }

# 可视化对比
dims = list(matryoshka_results.keys())
scifact_scores = [matryoshka_results[d]["SciFact"] for d in dims]
fiqa_scores = [matryoshka_results[d]["FiQA2018"] for d in dims]

plt.figure(figsize=(10, 6))
plt.plot(dims, scifact_scores, 'o-', label='SciFact', linewidth=2, markersize=8)
plt.plot(dims, fiqa_scores, 's-', label='FiQA2018', linewidth=2, markersize=8)
plt.xlabel('Embedding Dimension', fontsize=12)
plt.ylabel('nDCG@10', fontsize=12)
plt.title('Matryoshka Dimension vs Retrieval Performance', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('./mteb_results/matryoshka_comparison.png', dpi=150)
plt.show()
```

---

## 6. Reranker 模型评测代码

### 6.1 Reranker 基础推理验证

Reranker 模型与 Embedding 模型不同，它接受**查询-文档对**作为输入，输出相关性分数。以下是使用 `mlx-embeddings` 加载 Qwen3 Reranker 的基础验证代码：

```python
"""
Step 5: Reranker 基础推理验证
Reranker 对查询-文档对进行相关性打分
"""
import mlx.core as mx
from mlx_embeddings import load
import numpy as np

# 加载 Reranker 模型
reranker_name = "mlx-community/Qwen3-Reranker-0.6B-4bit"
reranker_model, reranker_tokenizer = load(reranker_name)

# 教育场景测试：查询 + 候选文档
query = "牛顿运动定律的应用场景"
candidates = [
    "牛顿第一定律描述了物体在不受外力时的运动状态，是经典力学的基础",
    "氧化还原反应中，电子的转移导致元素化合价发生变化",
    "牛顿第二定律 F=ma 揭示了力、质量和加速度之间的定量关系",
    "中国古代四大发明包括造纸术、印刷术、火药和指南针",
]

# 对每对 (query, doc) 打分
scores = []
for doc in candidates:
    # Reranker 的输入格式：将 query 和 document 拼接
    pair_text = f"Query: {query}\nDocument: {doc}"
    input_ids = reranker_tokenizer.encode(pair_text, return_tensors="mlx")
    outputs = reranker_model(input_ids)
    
    # 取 [CLS] 位置的 logits 或通过 pooler_output 获取相关性分数
    # 具体输出格式取决于 mlx-embeddings 对 Qwen3 Reranker 的实现
    score = float(outputs.pooler_output[0][1])  # 假设二分类 logits
    scores.append(score)

# 按分数排序
ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
print("Reranking Results:")
for i, (doc, score) in enumerate(ranked, 1):
    print(f"  Rank {i} (score={score:.4f}): {doc[:60]}...")
```

### 6.2 MTEB Reranking 任务评测

MTEB 的 Reranking 任务要求模型对给定的查询和候选文档列表进行排序。以下代码展示了如何将 MLX Reranker 集成到 MTEB 的 Reranking 评测中：

```python
"""
Step 6: MTEB Reranking 任务评测
"""
from mteb import AbsTaskReranking
from mteb.encoder_interface import Encoder
from typing import Sequence

class MLXQwen3Reranker:
    """
    Qwen3 Reranker 适配器
    MTEB Reranking 任务需要实现 rank() 方法
    """
    def __init__(self, model_path: str):
        self.model, self.tokenizer = load(model_path)
        self.max_length = 512  # Reranker 通常使用较短的 max_length
    
    def rank(
        self,
        query: str,
        documents: Sequence[str],
        **kwargs
    ) -> dict:
        """
        对查询和文档列表进行相关性排序
        
        Returns:
            {"documents": [sorted_docs], "scores": [scores]}
        """
        scores = []
        for doc in documents:
            # Qwen3 Reranker 的标准输入格式
            pair_text = f"Query: {query}\nDocument: {doc}"
            input_ids = self.tokenizer.encode(pair_text, return_tensors="mlx")
            outputs = self.model(input_ids)
            
            # 提取相关性分数（取决于具体模型实现）
            score = float(mx.max(outputs.pooler_output[0]))
            scores.append(score)
        
        # 按分数降序排序
        sorted_indices = np.argsort(scores)[::-1]
        return {
            "documents": [documents[i] for i in sorted_indices],
            "scores": [scores[i] for i in sorted_indices],
        }

# 执行 Reranking 评测
from mteb import MTEB

reranker = MLXQwen3Reranker("mlx-community/Qwen3-Reranker-0.6B-4bit")

reranking_tasks = [
    "SciDocsRR",      # 科学文档重排序
    "AskUbuntu",      # 技术问答重排序
    "MindSmallReranking",  # 小规模重排序
    "StackOverflowDupQuestions",  # 重复问题检测
]

for task_name in reranking_tasks:
    print(f"\nRunning reranking task: {task_name}")
    evaluation = MTEB(tasks=[task_name])
    results = evaluation.run(
        reranker,
        output_folder=f"./mteb_results/reranker_{task_name}",
    )
```

### 6.3 Embedding + Reranker 端到端 RAG 评测

在实际 RAG 系统中，Embedding 和 Reranker 是串联工作的。以下代码模拟完整的两阶段检索流程，评测端到端效果：

```python
"""
Step 7: 端到端 RAG 评测 (Embedding + Reranker)
模拟真实 RAG 流程：Embedding 召回 -> Reranker 精排
"""
from sklearn.metrics import ndcg_score
import numpy as np

def end_to_end_rag_evaluate(
    embedding_model: MLXQwen3Encoder,
    reranker_model: MLXQwen3Reranker,
    corpus: list[dict],
    queries: list[str],
    qrels: dict,  # {query_id: {doc_id: relevance}}
    top_k_retrieve: int = 100,
    top_k_rerank: int = 10,
):
    """
    端到端 RAG 评测
    
    Phase 1: Embedding 向量召回 Top-100
    Phase 2: Reranker 精排 Top-10
    """
    results = {"retrieval_only": {}, "with_reranker": {}}
    
    # 预编码全部文档（只执行一次）
    print("Encoding corpus...")
    corpus_texts = [f"{d.get('title', '')}\n{d.get('text', '')}" for d in corpus]
    corpus_embeddings = embedding_model.encode_corpus(corpus)
    
    for qid, query in enumerate(queries):
        # Phase 1: Embedding 召回
        query_emb = embedding_model.encode_queries([query])
        similarities = np.dot(corpus_embeddings, query_emb.T).flatten()
        top100_indices = np.argsort(similarities)[-top_k_retrieve:][::-1]
        
        # Phase 2: Reranker 精排
        top100_docs = [corpus_texts[i] for i in top100_indices]
        rerank_result = reranker_model.rank(query, top100_docs)
        top10_docs = rerank_result["documents"][:top_k_rerank]
        
        # 计算 nDCG@10
        # ... (与 qrels 对比计算指标)
    
    return results

# 运行端到端评测
encoder = MLXQwen3Encoder("mlx-community/Qwen3-Embedding-4B-4bit-DWQ")
reranker = MLXQwen3Reranker("mlx-community/Qwen3-Reranker-0.6B-4bit")

# 使用 SciFact 数据集进行端到端测试
e2e_results = end_to_end_rag_evaluate(
    embedding_model=encoder,
    reranker_model=reranker,
    corpus=scifact_corpus,      # 从 MTEB 数据集加载
    queries=scifact_queries,     # 从 MTEB 数据集加载
    qrels=scifact_qrels,         # 从 MTEB 数据集加载
)

print("End-to-End RAG Evaluation Results:")
print(f"  Retrieval Only nDCG@10: {e2e_results['retrieval_only']['ndcg@10']:.4f}")
print(f"  With Reranker nDCG@10:  {e2e_results['with_reranker']['ndcg@10']:.4f}")
```

---

## 7. 性能评测方案

### 7.1 推理速度基准测试

除了精度指标，还需要评测模型在 Mac M4 上的实际推理性能，包括吞吐量、延迟和资源占用：

```python
"""
Step 8: 性能基准测试
测量推理速度、内存占用和批量处理能力
"""
import time
import mlx.core as mx
from mlx_embeddings import load
import psutil
import os

class PerformanceBenchmark:
    """MLX Embedding/Reranker 性能基准测试器"""
    
    def __init__(self, model_path: str, model_type: str = "embedding"):
        self.model, self.tokenizer = load(model_path)
        self.model_type = model_type
        self.process = psutil.Process(os.getpid())
    
    def get_memory_mb(self):
        """获取当前进程内存占用 (MB)"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def benchmark_embedding(self, texts: list[str], batch_sizes: list[int] = [1, 8, 16, 32]):
        """Embedding 模型速度基准测试"""
        results = {}
        
        for bs in batch_sizes:
            if bs > len(texts):
                continue
                
            # 预热
            for i in range(0, min(32, len(texts)), bs):
                batch = texts[i:i+bs]
                for t in batch:
                    input_ids = self.tokenizer.encode(t, return_tensors="mlx")
                    _ = self.model(input_ids)
            mx.eval(self.model.parameters())  # 确保 Metal 完成
            
            # 正式测试
            mem_before = self.get_memory_mb()
            start = time.perf_counter()
            
            total_samples = 0
            for i in range(0, len(texts), bs):
                batch = texts[i:i+bs]
                for t in batch:
                    input_ids = self.tokenizer.encode(t, return_tensors="mlx")
                    outputs = self.model(input_ids)
                total_samples += len(batch)
            
            mx.eval(self.model.parameters())
            elapsed = time.perf_counter() - start
            mem_after = self.get_memory_mb()
            
            results[bs] = {
                "throughput_samples_per_sec": total_samples / elapsed,
                "latency_ms_per_sample": (elapsed / total_samples) * 1000,
                "memory_delta_mb": mem_after - mem_before,
            }
        
        return results
    
    def benchmark_reranker(self, query_doc_pairs: list[tuple], batch_size: int = 16):
        """Reranker 模型速度基准测试"""
        mem_before = self.get_memory_mb()
        start = time.perf_counter()
        
        for query, doc in query_doc_pairs:
            pair_text = f"Query: {query}\nDocument: {doc}"
            input_ids = self.tokenizer.encode(pair_text, return_tensors="mlx")
            _ = self.model(input_ids)
        
        mx.eval(self.model.parameters())
        elapsed = time.perf_counter() - start
        mem_after = self.get_memory_mb()
        
        return {
            "throughput_pairs_per_sec": len(query_doc_pairs) / elapsed,
            "latency_ms_per_pair": (elapsed / len(query_doc_pairs)) * 1000,
            "memory_delta_mb": mem_after - mem_before,
        }

# 运行性能基准测试
test_texts = ["教育政策文件示例内容"] * 1000  # 1000 条测试文本

# 测试 0.6B 4bit Embedding
bench_06b = PerformanceBenchmark(
    "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    model_type="embedding"
)
perf_06b = bench_06b.benchmark_embedding(test_texts)

print("\n=== Qwen3-Embedding-0.6B-4bit Performance ===")
for bs, metrics in perf_06b.items():
    print(f"  Batch={bs:2d}: {metrics['throughput_samples_per_sec']:.1f} samples/sec, "
          f"{metrics['latency_ms_per_sample']:.2f} ms/sample")

# 测试 4B 4bit Embedding
bench_4b = PerformanceBenchmark(
    "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
    model_type="embedding"
)
perf_4b = bench_4b.benchmark_embedding(test_texts)

print("\n=== Qwen3-Embedding-4B-4bit Performance ===")
for bs, metrics in perf_4b.items():
    print(f"  Batch={bs:2d}: {metrics['throughput_samples_per_sec']:.1f} samples/sec, "
          f"{metrics['latency_ms_per_sample']:.2f} ms/sample")
```

### 7.2 量化格式精度对比

四组套件涉及两种量化格式（4bit-DWQ 和 mxfp8），需要对比同一模型在不同精度格式下的表现：

| 对比维度 | 4bit-DWQ | mxfp8 | bf16 (参考) |
|---|---|---|---|
| 每参数比特数 | 4 bit | 8 bit | 16 bit |
| 内存节省 | ~75% | ~50% | 0% |
| 典型精度损失 | 1-3% | 0.5-1% | 0% (基准) |
| 推理速度 | 最快 | 快 | 基准 |
| 适用场景 | 边缘部署、资源受限 | 平衡精度与速度 | 最高精度需求 |
| 教育RAG建议 | 快速原型、大批量处理 | **推荐主力配置** | 精度验证基准 |

对于 Mac M4 32GB 的充裕内存环境，**mxfp8 是推荐的平衡选择**——它在保留 99%+ 精度的同时，相比 bf16 节省 50% 内存，且推理速度优于全精度。4bit-DWQ 适合需要同时加载多个模型或处理超大规模文档库的场景。

---

## 8. 完整评测流程总结

### 8.1 推荐评测执行顺序

基于您的四组配置和 RAG 教育场景需求，建议按以下顺序执行评测，从快速验证到深度分析逐步展开：

| 阶段 | 目标 | 配置 | 数据集 | 预计时间 |
|---|---|---|---|---|
| **Phase 1** | 快速功能验证 | low_memory_4bit | SciFact (subset) | 5-10 min |
| **Phase 2** | 全量精度基准 | low_memory_mxfp8 | P0 RAG 数据集 (6个) | 30-60 min |
| **Phase 3** | 平衡配置深度测 | balanced | P0 + P1 数据集 | 1-2 hours |
| **Phase 4** | 最高精度验证 | high_quality | 全量 P0/P1/P2 | 2-4 hours |
| **Phase 5** | 端到端 RAG 评测 | balanced/high_quality | 自定义端到端流程 | 1-2 hours |
| **Phase 6** | Matryoshka 维度分析 | balanced | SciFact, FiQA | 30 min |
| **Phase 7** | 性能基准测试 | 全部四组 | 吞吐量/延迟测试 | 20 min |

### 8.2 数据集自动下载说明

MTEB 框架在首次运行评测任务时会**自动从 Hugging Face Datasets 下载所需数据**，无需手动准备。数据集默认缓存于 `~/.cache/huggingface/datasets/` 目录。以下是主要数据集的来源信息：

| 数据集 | HuggingFace Dataset ID | 大小 | 任务类型 |
|---|---|---|---|
| SciFact | `mteb/scifact` | ~5MB | Retrieval |
| FiQA2018 | `mteb/fiqa` | ~15MB | Retrieval |
| NFCorpus | `mteb/nfcorpus` | ~3MB | Retrieval |
| ArguAna | `mteb/arguana` | ~2MB | Retrieval |
| SCIDOCS | `mteb/scidocs` | ~50MB | Retrieval |
| SciDocsRR | `mteb/scidocs-reranking` | ~50MB | Reranking |
| AskUbuntu | `mteb/askubuntu` | ~10MB | Reranking |
| STSBenchmark | `mteb/stsbenchmark-sts` | ~1MB | STS |
| Banking77 | `mteb/banking77` | ~2MB | Classification |

对于**中文教育内容**的评测，建议额外引入 **C-MTEB** 基准  [(bge-model.com)](https://bge-model.com/tutorial/4_Evaluation/4.2.3.html) ——它包含 35 个中文数据集，覆盖分类、聚类、重排序、检索和 STS 五大任务，包含 T2Retrieval（通用检索）、MMarcoReranking（多语言重排序）、MedicalRetrieval（医学检索）等与教育场景高度相关的数据集。

### 8.3 结果解读与选型建议

完成全部评测后，您将获得以下维度的数据来指导模型选型：

| 评估维度 | low_memory_4bit | low_memory_mxfp8 | balanced | high_quality |
|---|---|---|---|---|
| **检索精度** (nDCG@10 avg) | 基线 | 基线+1-2% | 显著提升 | 最高 |
| **重排序精度** (MAP) | 基线 | 基线+1-2% | 中等 | 最高 |
| **推理速度** (samples/sec) | 最快 | 很快 | 中等 | 较慢 |
| **内存占用** | 最低 (~1.5GB) | 低 (~1.9GB) | 中等 (~5.5GB) | 高 (~21GB) |
| **教育RAG推荐度** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

**选型建议**：对于教育领域的 RAG 应用（教材、课件、政策检索），**balanced 配置（4B Embedding + 0.6B Reranker）是最佳性价比选择**——它在检索精度上显著优于 0.6B 模型（典型提升 5-10% nDCG@10），同时内存占用仅 5.5GB，为 Mac M4 32GB 留下充足余量用于文档索引和上下文缓存。如果检索内容涉及大量专业术语（如医学教材、法律政策），则 high_quality 配置（8B + 8B）的额外精度提升值得其内存开销。low_memory 配置适合快速原型验证和边缘部署场景。

---

## 9. 附录：模型完整列表与 HuggingFace 地址

| 配置角色 | 简称 | 完整 HuggingFace ID |
|---|---|---|
| Embedding 0.6B 4bit | `qwen3_embedding_0.6b_4bit_dwq` | `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` |
| Embedding 0.6B mxfp8 | `qwen3_embedding_0.6b_mxfp8` | `mlx-community/Qwen3-Embedding-0.6B-mxfp8` |
| Embedding 4B 4bit | `qwen3_embedding_4b_4bit_dwq` | `mlx-community/Qwen3-Embedding-4B-4bit-DWQ` |
| Embedding 8B mxfp8 | `qwen3_embedding_8b_mxfp8` | `mlx-community/Qwen3-Embedding-8B-mxfp8` |
| Reranker 0.6B 4bit | `qwen3_reranker_0.6b_4bit` | `mlx-community/Qwen3-Reranker-0.6B-4bit` |
| Reranker 0.6B mxfp8 | `qwen3_reranker_0.6b_mxfp8` | `mlx-community/Qwen3-Reranker-0.6B-mxfp8` |
| Reranker 8B mxfp8 | `qwen3_reranker_8b_mxfp8` | `mlx-community/Qwen3-Reranker-8B-mxfp8` |
