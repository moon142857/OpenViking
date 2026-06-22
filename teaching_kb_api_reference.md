# 教学知识库系统 - 完整接口参考文档

**基于**: OpenViking 上下文数据库 v1.x + Qwen3 MLX 本地推理服务
**文档日期**: 2026-05-18

---

## 一、OpenViking 核心 API

Base URL: `http://127.0.0.1:1933/api/v1`

### 1.1 文件系统操作

---

#### `POST /fs/mkdir` — 创建目录

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `uri` | string | 是 | 目录 URI，如 `viking://resources/teaching_kb/public` |
| `description` | string | 否 | 目录描述 |

**响应**:
```json
{"status": "ok", "result": {"uri": "viking://resources/...", "created": true}}
```

**使用场景**: 创建知识库目录结构（如按年级/学科分目录）。

---

#### `GET /fs/ls` — 列出目录内容

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `uri` | string | 是 | 目录 URI（需 URL 编码） |

**响应**:
```json
{
  "status": "ok",
  "result": [
    {"uri": "viking://resources/.../policies", "type": "dir"},
    {"uri": "viking://resources/.../textbooks", "type": "dir"}
  ]
}
```

**使用场景**: 浏览知识库目录结构；验证导入完整性。

---

### 1.2 内容读写

---

#### `POST /content/write` — 写入/更新内容

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `uri` | string | 是 | 目标文件 URI |
| `content` | string | 是 | 文件内容（Markdown/纯文本） |
| `mode` | string | 是 | `"create"` 新建 / `"overwrite"` 覆盖 |
| `wait` | bool | 否 | `true` 等待向量化完成（慢）；`false` 异步处理 |

**响应**:
```json
{"status": "ok", "result": {"uri": "viking://...", "written": true}}
```

**使用场景**:
- 批量导入教案/政策文档
- 教师生成教案后回写到知识库
- 学校专有库内容写入

**注意**: `content/write` 不走资源导入流水线，不会自动触发向量化。导入后需调用 `/content/reindex`。

---

#### `GET /content/abstract` — L0 摘要（~100 字）

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `uri` | string | 是 | 文件 URI（需 URL 编码） |

**使用场景**: 搜索结果列表页快速预览。

---

#### `GET /content/overview` — L1 概述（~500 字）

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `uri` | string | 是 | 文件 URI（需 URL 编码） |

**使用场景**: 详情页首屏展示。

---

#### `GET /content/read` — L2 完整内容

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `uri` | string | 是 | 文件 URI（需 URL 编码） |
| `offset` | int | 否 | 起始字符偏移 |
| `limit` | int | 否 | 最大返回字符数 |

**使用场景**: 教案全文阅读、下载。

---

#### `POST /content/reindex` — 触发向量化

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `uri` | string | 是 | 目录或文件 URI |
| `wait` | bool | 否 | `true` 等待完成 |

**使用场景**: `content/write` 批量写入后，手动触发向量索引生成。

---

### 1.3 语义搜索

---

#### `POST /search/find` — 语义检索

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | 是 | 搜索查询文本 |
| `target_uri` | string | 否 | 限定搜索范围（目录 URI），为空则全局搜索 |
| `limit` | int | 否 | 返回结果数量上限，默认 10 |
| `level` | int | 否 | `0`=L0摘要 / `1`=L1概述 / `2`=L2全文；不填则返回所有级别 |
| `filter` | object | 否 | 元数据过滤 DSL |

**响应**:
```json
{
  "status": "ok",
  "result": {
    "resources": [
      {
        "uri": "viking://resources/.../分数初步认识-教案1.md",
        "score": 0.95,
        "level": 2,
        "abstract": "..."
      }
    ]
  }
}
```

**使用场景**:
- 教师搜索教案：`target_uri` 限定到 `lesson_plans/小学三年级/数学`
- 政策查询：`target_uri` 限定到 `policies`
- 跨库统一检索：并行调用三次（公共库/学校库/个人库）后合并

**搜索机制**: Dense Embedding + Sparse Embedding + Rerank（Hybrid Search）。

---

### 1.4 资源导入（完整流水线）

---

#### `POST /resources` — 资源导入

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `uri` | string | 是 | 目标 URI |
| `path` | string | 否 | 本地文件路径（**不支持**） |
| `wait` | bool | 否 | 是否等待处理完成 |

**使用场景**: 单文件导入，走完整处理流水线（解析→摘要→向量化）。

**注意**: `/resources` 不接受本地文件系统路径，需先用 `/resources/temp_upload` 上传。

---

#### `POST /resources/temp_upload` — 临时文件上传

**使用场景**: 上传本地文件后获取临时路径，再调用 `/resources` 导入。

---

### 1.5 会话管理

---

#### `POST /sessions` — 创建会话

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `session_id` | string | 否 | 自定义会话 ID |
| `user` | object | 否 | `{"account_id": "...", "user_id": "...", "agent_id": "..."}` |

**响应**:
```json
{
  "status": "ok",
  "result": {
    "session_id": "teacher_2048_20250518_math",
    "user": {"account_id": "teaching_platform", "user_id": "teacher_2048", "agent_id": "lesson_plan_assistant"}
  }
}
```

**使用场景**: 教师登录后创建专属备课会话。

---

#### `POST /sessions/{session_id}/messages` — 追加消息

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `role` | string | 是 | `"user"` / `"assistant"` / `"system"` |
| `content` | string | 否 | 文本内容（`role=user` 时使用） |
| `parts` | array | 否 | 多模态内容（`role=assistant` 时使用） |

**parts 结构**:
```json
[
  {"type": "text", "text": "生成的教案正文..."},
  {"type": "context", "uri": "viking://resources/...", "context_type": "resource"}
]
```

**使用场景**: 记录教师输入和 AI 生成结果（含检索上下文引用）。

---

#### `POST /sessions/{session_id}/commit` — 提交会话

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `keep_recent_count` | int | 否 | 保留最近 N 条消息继续对话 |

**执行流程**:
1. **Phase 1（同步）**: 归档会话历史到 `viking://session/{session_id}/history/`
2. **Phase 2（异步）**: 提取长期记忆写入教师个人空间

**使用场景**: 备课结束后沉淀经验，生成长期记忆。

---

### 1.6 系统与调试

---

#### `POST /system/wait` — 等待后台队列处理

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `timeout` | int | 否 | 等待超时秒数 |

**响应**:
```json
{"status": "ok"}  // 队列为空时返回
```

**使用场景**: 批量导入后等待 Embedding/Semantic 索引处理完成。

---

#### `GET /debug/vector/count` — 向量数量统计（调试）

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `uri` | string | 是 | URI 前缀（需 URL 编码） |

**注意**: 该接口统计的索引路径可能与实际搜索索引不同，`count=0` 不代表搜索不可用。以实际 `find` 功能为准。

---

## 二、MLX Embedding/Rerank 服务 API

Base URL: `http://localhost:11436`

### 2.1 Embedding 接口

---

#### `POST /v1/embeddings` — 文本嵌入

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `input` | string/array | 是 | 输入文本 |
| `instruction` | string | 否 | 嵌入指令（如 `"Given a web search query, retrieve relevant passages..."`） |
| `model` | string | 否 | 模型名称 |

**响应**:
```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "embedding": [0.1, 0.2, ...], "index": 0}
  ],
  "model": "Qwen3-Embedding-8B-4bit-DWQ",
  "usage": {"prompt_tokens": 45, "total_tokens": 45}
}
```

**使用场景**: OpenViking 内部调用；也可独立用于自定义向量化需求。

**配置映射**（`ov.conf`）:
```yaml
embedding:
  api_base: "http://localhost:11436/v1"
  model: "Qwen3-Embedding-8B"
  dimension: 4096
```

---

### 2.2 Rerank 接口

---

#### `POST /v1/rerank` — 重排序

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | 是 | 查询文本 |
| `documents` | array | 是 | 候选文档字符串列表 |
| `instruction` | string | 否 | 排序指令 |

**响应**:
```json
{
  "results": [
    {"index": 2, "score": 0.95},
    {"index": 0, "score": 0.82}
  ]
}
```

**使用场景**: OpenViking `find` 接口内部调用，对向量检索结果做精排。

**配置映射**（`ov.conf`）:
```yaml
rerank:
  api_base: "http://localhost:11436/v1/rerank"
  model: "Qwen3-Reranker-8B"
  threshold: 0.1
```

---

## 三、教学知识库系统应用场景映射

### 3.1 知识库初始化流程

| 步骤 | 接口 | 说明 |
|------|------|------|
| 1. 创建目录结构 | `POST /fs/mkdir` | 按 `policies/textbooks/lesson_plans/年级/学科` 建目录 |
| 2. 写入文档 | `POST /content/write` | 逐个写入 Markdown 文件，`wait=false` |
| 3. 等待处理 | `POST /system/wait` | 等待后台 Embedding 队列完成 |
| 4. 触发向量化 | `POST /content/reindex` | 对整个知识库目录重建索引 |

### 3.2 教师检索流程

| 步骤 | 接口 | 说明 |
|------|------|------|
| 1. 搜索资源 | `POST /search/find` | 指定 `target_uri` 和 `query` |
| 2. 展示摘要 | `GET /content/abstract` | 搜索结果列表展示 L0 摘要 |
| 3. 查看详情 | `GET /content/overview` / `GET /content/read` | 详情页展示 L1/L2 内容 |

### 3.3 教案生成流程

| 步骤 | 接口/服务 | 说明 |
|------|-----------|------|
| 1. 检索上下文 | `POST /search/find` | 同时检索政策+教材+参考教案 |
| 2. 加载内容 | `GET /content/read` | 读取 Top-K 结果的全文 |
| 3. LLM 生成 | Claude/GPT-4/Kimi API | 外部 LLM 生成教案 |
| 4. 存储结果 | `POST /content/write` | 写入 `viking://user/teacher_{id}/lesson_plans/` |
| 5. 经验沉淀 | `POST /sessions/{id}/commit` | 提交会话，提取长期记忆 |

### 3.4 教案评价流程

| 步骤 | 接口/服务 | 说明 |
|------|-----------|------|
| 1. 检索标准 | `POST /search/find` | 检索课标要求和优秀教案范例 |
| 2. LLM 评价 | Claude/GPT-4/Kimi API | 按 5 维度（教学目标/内容/方法/互动/反思）打分 |
| 3. 存储评价 | `POST /content/write` | 评价结果写入教师个人空间 |

### 3.5 多租户隔离方案

| 层级 | URI 前缀 | Scope | 访问规则 | 接口使用方式 |
|------|---------|-------|---------|-------------|
| 公共知识库 | `viking://resources/teaching_kb/public/` | `resources` | 所有认证用户可读 | 直接调用 `find`，`target_uri` 指向公共库 |
| 学校专有库 | `viking://user/school_{id}/kb/` | `user` | 仅本校可见 | 后端代理，Header 切换 `X-OpenViking-User: school_{id}` |
| 教师个人空间 | `viking://user/teacher_{id}/` | `user` | 仅本人可见 | 后端代理，Header 切换 `X-OpenViking-User: teacher_{id}` |

**后端代理请求头**:
```python
headers = {
    "X-API-Key": ROOT_KEY,           # ADMIN/ROOT 密钥
    "X-OpenViking-Account": "teaching_platform",
    "X-OpenViking-User": f"school_{school_id}",  # 或 teacher_{teacher_id}
}
```

---

## 四、性能基准参考

| 操作 | MLX 本地环境 | 生产环境预估（云 API） |
|------|-------------|---------------------|
| Embedding | 0.5-4s/条 | 200-500ms |
| Rerank | 0.7-9s/文档 | 100-300ms |
| 单次 `find` 总延迟 | 25-57s | 1-3s |
| `fs/ls` | 2-5ms | 2-5ms |
| `content/read` | 5-10ms | 5-10ms |

---

## 五、关键注意事项

1. **`resources` scope 全局可读** — 学校保密内容必须放在 `user` scope 下
2. **`content/write` 不自动向量化** — 写入后必须调用 `/content/reindex`
3. **MLX 服务单线程** — Metal GPU 不支持并发，MLX Server 使用 `HTTPServer` + `threading.Lock`
4. **`session` 是临时存储** — 必须通过 `commit` 才能沉淀为长期记忆
5. **后端是安全边界** — 教师终端不持有 OpenViking API Key，所有请求经后端代理

---

*文档生成时间: 2026-05-18*
*关联文档: teaching_kb_assessment_report.md, teaching_kb_multi_tenant_architecture.md*
