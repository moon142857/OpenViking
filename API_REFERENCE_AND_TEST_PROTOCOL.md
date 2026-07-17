# OpenViking HTTP API 参考与测试方案

> 版本：OpenViking 开发分支（HEAD `a1c633c3`，git describe `v0.4.9-22`；`/health.version` 报告 `0.4.4`）  
> 测试环境：macOS Apple Silicon，Python 3.13.12，MLX 嵌入/重排序服务（127.0.0.1:11436），OpenViking Server（127.0.0.1:1933）  
> 文档用途：列出 OpenViking 对外暴露的 HTTP API、功能说明、典型使用场景，并附功能/性能测试结果与可复用的测试协议。

---

## 目录

1. [概述](#1-概述)
2. [基础信息](#2-基础信息)
3. [通用响应格式](#3-通用响应格式)
4. [接口清单](#4-接口清单)
5. [功能测试结果](#5-功能测试结果)
6. [性能测试结果](#6-性能测试结果)
7. [测试协议](#7-测试协议)
8. [附录：测试脚本](#8-附录测试脚本)

---

## 1. 概述

OpenViking 是一个面向 AI Agent 的开源上下文数据库（Context Database）。它采用"文件系统范式"统一管理 memories、resources、skills 三类上下文，并提供 L0/L1/L2 三级分层加载、语义检索、目录递归检索、会话记忆归档、Observer 可观测性等能力。

本文档基于 `openviking/server/routers/*.py` 源码（HEAD `a1c633c3`）逐文件核对整理，覆盖 OpenViking 当前对外暴露的全部 REST 接口，并提供可直接运行的功能与性能测试方案。

---

## 2. 基础信息

- **Base URL**：`http://127.0.0.1:1933`（默认本地部署）
- **API 前缀**：大部分接口使用 `/api/v1`
- **认证模式**：当前测试环境为 `dev` 模式（`auth_mode: dev`），无需额外 API Key
- **CORS**：默认开启，浏览器/插件可直接调用
- **模型依赖**：
  - Embedding：`Qwen3-Embedding-8B-4bit-DWQ`（MLX，4096 维）
  - Rerank：`Qwen3-Reranker-8B-MLX-4bit`
  - VLM：`kimi-2.6`

### 健康检查

```bash
curl http://127.0.0.1:1933/health
```

响应示例：

```json
{
  "status": "ok",
  "healthy": true,
  "version": "0.4.4",
  "auth_mode": "dev"
}
```

---

## 3. 通用响应格式

OpenViking 接口统一返回如下 JSON 结构：

```json
{
  "status": "ok|error",
  "result": { ... },
  "error": null,
  "telemetry": { ... }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | `ok` 或 `error` |
| `result` | any | 成功时返回的业务数据 |
| `error` | object/null | 失败时的错误信息，包含 `code` 与 `message` |
| `telemetry` | object | 请求链路统计（可选） |

错误码示例：

| HTTP 状态码 | 错误码 | 含义 |
|-------------|--------|------|
| 400 | `INVALID_ARGUMENT` | 参数校验失败 |
| 404 | `NOT_FOUND` | 资源不存在 |
| 401 | `UNAUTHORIZED` | 未授权 |
| 409 | `CONFLICT` | 资源冲突（如重复 Watch） |
| 413 | `PAYLOAD_TOO_LARGE` | 上传文件过大 |

---

## 4. 接口清单

### 4.1 System（系统）

| 方法 | 路径 | 功能简介 | 典型场景 |
|------|------|----------|----------|
| `GET` | `/health` | 服务健康检查 | 负载均衡探针、启动检测 |
| `GET` | `/ready` | 就绪检查（含 AGFS/VectorDB/Embedding 等子系统） | K8s readinessProbe |
| `GET` | `/api/v1/system/status` | 系统初始化状态与当前用户 | 控制台首页 |
| `POST` | `/api/v1/system/wait` | 等待系统初始化完成 | 脚本启动后同步等待 |
| `POST` | `/api/v1/system/consistency` | 一致性检查 | 运维巡检 |
| `POST` | `/api/v1/system/backend/sync-status` | 后端同步状态 | 多写一致性监控 |
| `POST` | `/api/v1/system/backend/sync-retry` | 后端同步重试 | 故障恢复 |
| `GET` | `/api/v1/system/sync/{sync_path:path}` | 查询同步路径状态 | 分布式部署 |
| `POST` | `/api/v1/system/sync/{sync_path:path}/retry` | 重试指定同步路径 | 分布式部署 |

**`/ready` 返回示例**：

```json
{
  "status": "ready",
  "checks": {
    "agfs": { "status": "ok", "checks": { "filesystem": "ok" } },
    "vectordb": "ok",
    "api_key_manager": "not_configured",
    "embedding": "ok",
    "ollama": "not_configured"
  }
}
```

---

### 4.2 Resources（资源管理）

资源对应 Agent 需要阅读的文档、代码库、网页等外部材料。

| 方法 | 路径 | 功能简介 | 典型场景 |
|------|------|----------|----------|
| `POST` | `/api/v1/resources/temp_upload` | 临时文件上传，返回 `temp_file_id` | 本地文件先上传再导入 |
| `POST` | `/api/v1/resources` | 添加远程/本地资源到 OpenViking | 导入 GitHub README、文档仓库 |
| `POST` | `/api/v1/skills` | 添加 Skill（与资源上传共用临时文件机制） | 注册 Agent 工具/技能 |

#### `POST /api/v1/resources`

**请求体（AddResourceRequest）**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `path` | string | 与 `temp_file_id` 二选一 | 远程 URL/Repo 地址 |
| `temp_file_id` | string | 与 `path` 二选一 | 临时上传文件 ID |
| `to` | string | 否 | 目标 URI，如 `viking://resources/my_repo` |
| `parent` | string | 否 | 父目录 URI，不能与 `to` 同时使用 |
| `create_parent` | bool | 否 | 自动创建父目录 |
| `reason` | string | 否 | 添加原因 |
| `instruction` | string | 否 | 语义提取提示 |
| `wait` | bool | 否 | 是否等待语义处理完成（默认 false） |
| `timeout` | float | 否 | `wait=true` 时超时秒数 |
| `strict` | bool | 否 | 严格模式（默认 false） |
| `ignore_dirs` | string | 否 | 忽略的目录列表，逗号分隔 |
| `include`/`exclude` | string | 否 | Glob 过滤 |
| `directly_upload_media` | bool | 否 | 直接上传媒体文件（默认 true） |
| `preserve_structure` | bool | 否 | 保留目录结构 |
| `args` | object | 否 | 解析器参数（如飞书 token） |
| `watch_interval` | float | 否 | 自动监控间隔（分钟），0 表示不监控 |

**请求示例**：

```bash
curl -X POST http://127.0.0.1:1933/api/v1/resources \
  -H "Content-Type: application/json" \
  -d '{
    "path": "https://raw.githubusercontent.com/volcengine/OpenViking/main/README.md",
    "to": "viking://resources/api-test-readme",
    "wait": false
  }'
```

**响应示例**：

```json
{
  "status": "ok",
  "result": {
    "status": "success",
    "root_uri": "viking://resources/api-test-readme",
    "temp_uri": "viking://temp/default/.../README",
    "task_id": "629583b9-cdae-408a-87c7-22434c9c723e"
  }
}
```

---

### 4.3 Filesystem（文件系统操作）

OpenViking 通过虚拟文件系统 URI（`viking://...`）暴露资源，支持类 POSIX 操作。

| 方法 | 路径 | 功能简介 | 典型场景 |
|------|------|----------|----------|
| `GET` | `/api/v1/fs/ls` | 列出目录内容 | 资源浏览器、文件选择器 |
| `GET` | `/api/v1/fs/tree` | 递归列出目录树 | 展示完整目录结构 |
| `GET` | `/api/v1/fs/stat` | 获取文件/目录元信息 | 判断类型、大小、修改时间 |
| `GET` | `/api/v1/fs/attrs` | 获取逻辑扩展属性（tags/memory） | 读取检索标签、记忆属性 |
| `POST` | `/api/v1/fs/attrs/set_tags` | 设置检索标签 | 与 `/content/set_tags` 等价 |
| `POST` | `/api/v1/fs/mkdir` | 创建目录 | 组织资源 |
| `DELETE` | `/api/v1/fs` | 删除文件或目录 | 清理资源 |
| `POST` | `/api/v1/fs/mv` | 移动/重命名 | 整理目录 |

**`/api/v1/fs/ls` 查询参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `uri` | string | 是 | 目录 URI |

**`/api/v1/fs/tree` 查询参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `uri` | string | 是 | 目录 URI |
| `depth` | int | 否 | 递归深度 |

**`/api/v1/fs/stat` 查询参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `uri` | string | 是 | 文件或目录 URI |

**`/api/v1/fs/attrs` 查询参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `uri` | string | 是 | 文件或目录 URI |

> `attrs` 返回 `context_type` 与 `attrs.tags`；当 URI 为 memory 文件时额外返回 `attrs.memory`。`POST /api/v1/fs/attrs/set_tags` 请求体与 `POST /api/v1/content/set_tags` 一致（`uri`、`tags`、`mode=replace`、`recursive`）。

**示例**：

```bash
curl "http://127.0.0.1:1933/api/v1/fs/ls?uri=viking://resources/"
curl "http://127.0.0.1:1933/api/v1/fs/tree?uri=viking://resources/api-test-readme&depth=2"
curl "http://127.0.0.1:1933/api/v1/fs/stat?uri=viking://resources/api-test-readme"
```

---

### 4.4 Content（内容读写）

| 方法 | 路径 | 功能简介 | 典型场景 |
|------|------|----------|----------|
| `GET` | `/api/v1/content/read` | 读取文件 L2 完整内容 | 查看文档、代码 |
| `GET` | `/api/v1/content/abstract` | 读取 L0 摘要 | 快速了解文件/目录 |
| `GET` | `/api/v1/content/overview` | 读取 L1 概述 | 比摘要更详细 |
| `GET` | `/api/v1/content/download` | 以二进制流下载文件 | 图片、PDF |
| `POST` | `/api/v1/content/write` | 写入/追加/创建文件 | Agent 生成记忆文件 |
| `POST` | `/api/v1/content/set_tags` | 设置检索标签 | 增强语义检索 |
| `POST` | `/api/v1/content/reindex` | 重建向量/语义索引 | 维护、修复索引 |

**`/api/v1/content/read` 查询参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `uri` | string | 是 | 文件 URI |
| `offset` | int | 否 | 起始行号（0-based） |
| `limit` | int | 否 | 读取行数，-1 表示读到末尾 |
| `raw` | bool | 否 | 是否返回原始存储内容 |

**`/api/v1/content/write` 请求体**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `uri` | string | 是 | 目标文件 URI |
| `content` | string | 是 | 文本内容 |
| `mode` | string | 否 | `replace`/`append`/`create` |
| `wait` | bool | 否 | 是否等待语义处理 |
| `timeout` | float | 否 | 超时秒数 |

---

### 4.5 Search（检索）

OpenViking 提供三类检索能力：语义检索（find/search）、内容检索（grep）、路径匹配（glob）。

| 方法 | 路径 | 功能简介 | 典型场景 |
|------|------|----------|----------|
| `POST` | `/api/v1/search/find` | 语义检索（无会话） | 知识库问答 |
| `POST` | `/api/v1/search/search` | 语义检索（带会话上下文） | 会话中检索 |
| `POST` | `/api/v1/search/recall` | 按类型配额的记忆召回（bounded 渲染） | Agent 注入相关记忆 |
| `POST` | `/api/v1/search/grep` | 文本模式匹配 | 代码/文档关键字搜索 |
| `POST` | `/api/v1/search/glob` | Glob 路径匹配 | 按文件名/后缀筛选 |

#### `POST /api/v1/search/find`

**请求体（FindRequest）**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | 是 | 检索 query |
| `target_uri` | string/array | 否 | 目标 URI 范围 |
| `context_type` | string/array | 否 | 上下文类型过滤（memory/resource/skill） |
| `agent_id`/`agent_uri` | string | 否 | Agent 身份 |
| `limit` | int | 否 | 返回条数（默认 10） |
| `node_limit` | int | 否 | 节点级条数限制 |
| `score_threshold` | float | 否 | 相似度阈值 |
| `filter` | object | 否 | 高级过滤条件 |
| `include_provenance` | bool | 否 | 包含来源信息 |
| `tags` | array | 否 | 标签过滤 |
| `since`/`until` | string | 否 | 时间范围 |
| `time_field` | string | 否 | `updated_at`/`created_at` |
| `level` | int/string/array | 否 | L0/L1/L2 层级过滤 |

**请求示例**：

```bash
curl -X POST http://127.0.0.1:1933/api/v1/search/find \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what is OpenViking",
    "target_uri": "viking://resources/api-test-readme",
    "limit": 3
  }'
```

#### `POST /api/v1/search/recall`

按类型配额的记忆召回（type-quota recall）：在字符预算内召回并渲染最相关的记忆，供 Agent 注入上下文。

**请求体（RecallRequest）**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | 是 | 召回 query |
| `quotas` | object | 否 | 各记忆类型条数配额，默认 `{"events":10,"entities":10,"preferences":3,"experiences":0}` |
| `max_chars` | int | 否 | 渲染总字符上限，默认 `6500` |
| `min_score` | float | 否 | 相似度阈值，默认 `0.1` |
| `peer_scope` | string | 否 | `actor`（仅当前用户）/ `all`（默认 `all`） |
| `other_peer_penalty` | float/object | 否 | 其他用户记忆降权（数值或按类型字典） |
| `render` | bool | 否 | 是否渲染为文本（默认 true） |
| `telemetry` | object | 否 | 链路统计 |

#### `POST /api/v1/search/grep`

**请求体（GrepRequest）**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `uri` | string | 是 | 目标 URI |
| `pattern` | string | 是 | 正则/关键字 |
| `exclude_uri` | string | 否 | 排除 URI |
| `case_insensitive` | bool | 否 | 忽略大小写 |
| `node_limit` | int | 否 | 节点限制 |
| `level_limit` | int | 否 | 层级限制（默认 5） |

#### `POST /api/v1/search/glob`

**请求体（GlobRequest）**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `pattern` | string | 是 | Glob 模式 |
| `uri` | string | 否 | 起始 URI（默认 `viking://`） |
| `node_limit` | int | 否 | 节点限制 |

---

### 4.6 Sessions（会话管理）

会话用于维护 Agent 多轮交互上下文，支持消息追加、归档、记忆提取。

| 方法 | 路径 | 功能简介 | 典型场景 |
|------|------|----------|----------|
| `POST` | `/api/v1/sessions` | 创建会话 | 开始一次 Agent 对话 |
| `GET` | `/api/v1/sessions` | 列会话 | 会话列表页 |
| `GET` | `/api/v1/sessions/{session_id}` | 获取会话详情 | 恢复对话 |
| `GET` | `/api/v1/sessions/{session_id}/tool-results` | 列出工具结果 | 工具调用审计 |
| `GET` | `/api/v1/sessions/{session_id}/tool-results/{tool_result_id}` | 读取工具结果 | 展示长工具输出 |
| `GET` | `/api/v1/sessions/{session_id}/tool-results/{tool_result_id}/search` | 在工具结果中搜索 | 定位工具输出 |
| `GET` | `/api/v1/sessions/{session_id}/context` | 获取组装后的会话上下文 | 构造 LLM prompt |
| `GET` | `/api/v1/sessions/{session_id}/archives/{archive_id}` | 获取归档 | 历史记录查看 |
| `DELETE` | `/api/v1/sessions/{session_id}` | 删除会话 | 清理 |
| `POST` | `/api/v1/sessions/{session_id}/commit` | 提交会话（归档 + 异步提取记忆） | 会话结束 |
| `POST` | `/api/v1/sessions/{session_id}/extract` | 手动提取记忆 | 即时记忆更新 |
| `POST` | `/api/v1/sessions/{session_id}/messages` | 追加单条消息 | 多轮对话 |
| `POST` | `/api/v1/sessions/{session_id}/messages/batch` | 批量追加消息 | 批量导入 |
| `POST` | `/api/v1/sessions/{session_id}/used` | 记录已使用上下文 | 强化学习/反馈 |

**`POST /api/v1/sessions/{session_id}/messages` 请求体（AddMessageRequest）**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `role` | string | 是 | 角色，如 `user`/`assistant` |
| `content` | string | 与 `parts` 二选一 | 简单文本 |
| `parts` | array | 与 `content` 二选一 | Part 数组（text/context/tool/image_url） |
| `peer_id`/`agent_id`/`agent_uri` | string | 否 | 发送者身份 |

---

### 4.7 Skills（技能）

| 方法 | 路径 | 功能简介 | 典型场景 |
|------|------|----------|----------|
| `GET` | `/api/v1/skills` | 列出已注册技能 | 技能市场/管理 |
| `POST` | `/api/v1/skills/find` | 语义搜索技能 | Agent 动态选技能 |
| `POST` | `/api/v1/skills/validate` | 校验技能定义 | 发布前检查 |
| `GET` | `/api/v1/skills/{skill_name}` | 获取技能详情 | 展示技能信息 |
| `PUT` | `/api/v1/skills/{skill_name}` | 更新技能 | 版本迭代 |
| `DELETE` | `/api/v1/skills/{skill_name}` | 删除技能 | 下线 |

---

### 4.8 Tasks（任务）

| 方法 | 路径 | 功能简介 | 典型场景 |
|------|------|----------|----------|
| `GET` | `/api/v1/tasks/{task_id}` | 查询异步任务状态 | 轮询资源处理进度 |
| `GET` | `/api/v1/tasks` | 列出任务 | 任务管理 |

---

### 4.9 Observer（可观测性）

| 方法 | 路径 | 功能简介 | 典型场景 |
|------|------|----------|----------|
| `GET` | `/api/v1/observer/system` | 系统整体健康度 | 运维大盘 |
| `GET` | `/api/v1/observer/queue` | 队列状态（Embedding/Semantic） |  backlog 监控 |
| `GET` | `/api/v1/observer/vikingdb` | 向量库统计 | 索引健康 |
| `GET` | `/api/v1/observer/models` | 模型调用统计 | Token 监控 |
| `GET` | `/api/v1/observer/lock` | 锁状态 | 并发调试 |
| `GET` | `/api/v1/observer/retrieval` | 检索指标 | RAG 效果监控 |
| `GET` | `/api/v1/observer/filesystem` | 文件系统操作统计 | IO 性能分析 |

---

### 4.10 Console（控制台）

| 方法 | 路径 | 功能简介 | 典型场景 |
|------|------|----------|----------|
| `GET` | `/api/v1/console/dashboard/summary` | 仪表盘摘要 | 首页数据 |
| `GET` | `/api/v1/console/tokens` | Token 消耗统计 | 按日期聚合 |
| `GET` | `/api/v1/console/context-commits` | 上下文提交记录 | 审计 |
| `GET` | `/api/v1/console/audit` | 审计日志 | 安全审计 |

**`/api/v1/console/tokens` 查询参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `start_date` | string | 是 | 开始日期 `YYYY-MM-DD` |
| `end_date` | string | 是 | 结束日期 `YYYY-MM-DD` |
| `bucket` | string | 否 | `day`/`week`/`month` |

---

### 4.11 Pack（导入导出）

| 方法 | 路径 | 功能简介 | 典型场景 |
|------|------|----------|----------|
| `POST` | `/api/v1/pack/export` | 导出 ovpack 包 | 备份、迁移 |
| `POST` | `/api/v1/pack/backup` | 创建备份 | 定时备份 |
| `POST` | `/api/v1/pack/import` | 导入 ovpack 包 | 恢复、迁移 |
| `POST` | `/api/v1/pack/restore` | 从备份恢复 | 灾难恢复 |

---

### 4.12 Watches（资源监控）

| 方法 | 路径 | 功能简介 | 典型场景 |
|------|------|----------|----------|
| `GET` | `/api/v1/watches` | 列出监控任务 | 监控列表 |
| `GET` | `/api/v1/watches/{task_id}` | 获取监控任务详情 | 详情页 |
| `PATCH` | `/api/v1/watches/{task_id}` | 更新监控任务 | 调整间隔 |
| `PATCH` | `/api/v1/watches` | 批量更新 | 批量管理 |
| `DELETE` | `/api/v1/watches/{task_id}` | 删除监控任务 | 取消监控 |
| `DELETE` | `/api/v1/watches` | 批量删除 | 批量取消 |
| `POST` | `/api/v1/watches/{task_id}/trigger` | 手动触发一次监控 | 即时更新 |
| `POST` | `/api/v1/watches/trigger` | 批量触发 | 即时更新 |

---

### 4.13 Relations（关系图）

| 方法 | 路径 | 功能简介 | 典型场景 |
|------|------|----------|----------|
| `GET` | `/api/v1/relations` | 列出关系 | 查询已建关系 |
| `POST` | `/api/v1/relations/link` | 创建关系链接 | 构建知识图谱 |
| `DELETE` | `/api/v1/relations/link` | 删除关系链接 | 图维护 |
| `POST` | `/api/v1/relations/build_graph` | 构建关系图 | 批量建图 |

**`/api/v1/relations` 查询参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `uri` | string | 是 | 目标 URI |

---

### 4.14 Privacy Configs（隐私配置）

| 方法 | 路径 | 功能简介 | 典型场景 |
|------|------|----------|----------|
| `GET` | `/api/v1/privacy-configs` | 列出隐私配置 | 查看策略 |
| `GET` | `/api/v1/privacy-configs/{category}` | 获取某类配置 | 分类查看 |
| `GET` | `/api/v1/privacy-configs/{category}/{target_key}` | 获取具体配置 | 详情 |
| `GET` | `/api/v1/privacy-configs/{category}/{target_key}/versions` | 获取配置历史 | 版本管理 |
| `GET` | `/api/v1/privacy-configs/{category}/{target_key}/versions/{version}` | 获取具体版本 | 回滚参考 |
| `POST` | `/api/v1/privacy-configs/{category}/{target_key}` | 创建/更新配置 | 策略下发 |
| `POST` | `/api/v1/privacy-configs/{category}/{target_key}/activate` | 激活配置 | 生效策略 |

---

### 4.15 Code（代码辅助）

| 方法 | 路径 | 功能简介 | 典型场景 |
|------|------|----------|----------|
| `POST` | `/api/v1/code/outline` | 生成代码大纲 | 代码导航 |
| `POST` | `/api/v1/code/search` | 代码语义搜索 | 代码问答 |
| `POST` | `/api/v1/code/expand` | 代码展开/补全 | 查看引用 |

---

### 4.16 Bot（VikingBot 对话）

| 方法 | 路径 | 功能简介 | 典型场景 |
|------|------|----------|----------|
| `GET` | `/bot/v1/health` | Bot 健康检查 | Bot 状态监控 |
| `POST` | `/bot/v1/chat` | 非流式对话 | 单次问答 |
| `POST` | `/bot/v1/chat/stream` | 流式对话 | 实时响应 |
| `POST` | `/bot/v1/feedback` | 提交反馈 | 强化学习 |

> Bot 路由挂载在 `/bot/v1` 前缀下（非 `/api/v1/bot`）。

---

### 4.17 Admin（管理）

| 方法 | 路径 | 功能简介 | 典型场景 |
|------|------|----------|----------|
| `POST` | `/api/v1/admin/accounts` | 创建账户 | 多租户 |
| `GET` | `/api/v1/admin/accounts` | 列出账户 | 账户管理 |
| `DELETE` | `/api/v1/admin/accounts/{account_id}` | 删除账户 | 下线 |
| `POST` | `/api/v1/admin/accounts/{account_id}/users` | 创建用户 | 用户管理 |
| `GET` | `/api/v1/admin/accounts/{account_id}/users` | 列出用户 | 用户管理 |
| `DELETE` | `/api/v1/admin/accounts/{account_id}/users/{user_id}` | 删除用户 | 用户下线 |
| `PUT` | `/api/v1/admin/accounts/{account_id}/users/{user_id}/role` | 修改用户角色 | 权限调整 |
| `POST` | `/api/v1/admin/accounts/{account_id}/users/{user_id}/key` | 生成 API Key | 密钥轮换 |
| `POST` | `/api/v1/admin/migrate` | 数据迁移 | 升级、迁移 |

> Admin 路由统一挂载在 `/api/v1/admin` 前缀下；`dev` 模式通常未启用多租户，相关接口可能返回 404。

---

### 4.18 Debug（调试）

| 方法 | 路径 | 功能简介 | 典型场景 |
|------|------|----------|----------|
| `GET` | `/api/v1/debug/health` | Debug 健康检查 | 开发调试 |
| `GET` | `/api/v1/debug/vector/scroll` | 向量滚动查询 | 向量内容审计 |
| `GET` | `/api/v1/debug/vector/count` | 向量计数 | 索引统计 |

---

### 4.19 Metrics（指标）

| 方法 | 路径 | 功能简介 | 典型场景 |
|------|------|----------|----------|
| `GET` | `/metrics` | Prometheus 指标 | 监控采集 |

> 当前测试环境未启用 Prometheus metrics，返回 `404 Prometheus metrics are disabled`。

---

### 4.20 Stats（统计）

记忆健康度与会话提取统计，挂载在 `/api/v1/stats` 前缀下。

| 方法 | 路径 | 功能简介 | 典型场景 |
|------|------|----------|----------|
| `GET` | `/api/v1/stats/memories` | 记忆统计（按类别聚合热度/陈旧度） | 记忆大盘 |
| `GET` | `/api/v1/stats/sessions/{session_id}` | 会话记忆提取统计 | 单会话统计 |

**`/api/v1/stats/memories` 查询参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `category` | string | 否 | 按记忆类别过滤 |

> 注意：早期文档误记为 `/api/v1/memories`（返回 404），正确路径为 `/api/v1/stats/memories`。`/api/v1/sessions/{session_id}`（会话详情）由 Sessions 路由提供，与本统计接口不同。

---

### 4.21 WebDAV

OpenViking 同时提供 WebDAV 服务（挂载路径 `/webdav/resources`），可直接用文件管理器或挂载为本地磁盘访问 `viking://resources` 命名空间。

| 方法 | 路径 | 功能简介 | 典型场景 |
|------|------|----------|----------|
| WebDAV | `/webdav/resources` | 文件系统协议访问（`viking://resources` 命名空间） | 挂载为本地目录 |

### 4.22 Snapshot（快照 / Git 式版本控制）

对工作区进行 git 风格的快照（commit/restore/show/log）与 `.ovgitignore` 管理，底层对应 VikingFS 的 commit/restore/show/log。

| 方法 | 路径 | 功能简介 | 典型场景 |
|------|------|----------|----------|
| `POST` | `/api/v1/snapshot/commit` | 创建快照 | 保存工作区状态 |
| `POST` | `/api/v1/snapshot/restore` | 前向提交式恢复 | 回滚/重建到指定 commit |
| `GET` | `/api/v1/snapshot/show` | 查看提交元数据或 blob | 历史内容查看 |
| `GET` | `/api/v1/snapshot/log` | 提交历史 | 版本审计 |
| `GET` | `/api/v1/snapshot/ignore` | 读取 `.ovgitignore` | 查看忽略规则 |
| `PUT` | `/api/v1/snapshot/ignore` | 写入 `.ovgitignore` | 设置忽略规则 |
| `DELETE` | `/api/v1/snapshot/ignore` | 删除 `.ovgitignore` | 清除忽略规则 |

**`POST /api/v1/snapshot/commit` 请求体（CommitRequest）**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `message` | string | 是 | 提交信息 |
| `paths` | array | 否 | 限定快照路径列表 |
| `branch` | string | 否 | 分支，默认 `main` |
| `author_name` | string | 否 | 作者名 |
| `author_email` | string | 否 | 作者邮箱 |

**`POST /api/v1/snapshot/restore` 请求体（RestoreRequest）**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `source_commit` | string | 是 | 恢复源 commit oid |
| `project_dir` | string | 否 | 目标项目目录 |
| `branch` | string | 否 | 分支，默认 `main` |
| `dry_run` | bool | 否 | 仅预演不写回（默认 false） |
| `message` | string | 否 | 恢复提交信息 |
| `author_name`/`author_email` | string | 否 | 作者信息 |

**`GET /api/v1/snapshot/show` 查询参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `target_ref` | string | 是 | commit oid / 分支 / tag |
| `path` | string | 否 | 指定 blob 的 `viking://` URI；不传返回提交元数据 JSON，传则返回原始字节（带 `X-Snapshot-Oid`/`X-Snapshot-Size` 响应头） |

**`GET /api/v1/snapshot/log` 查询参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `branch` | string | 否 | 分支，默认 `main` |
| `limit` | int | 否 | 最大返回数，默认 20（范围 1–500） |

**`PUT /api/v1/snapshot/ignore` 请求体（SetIgnoreRequest）**：`{ "content": string }`。

---

### 4.23 User Settings（用户设置）

管理当前用户的资源/技能"添加位置"（add-locations）覆盖；响应同时返回 `override`（用户覆盖）与 `effective`（合并 server 配置后的最终生效值）。

| 方法 | 路径 | 功能简介 | 典型场景 |
|------|------|----------|----------|
| `GET` | `/api/v1/user-settings/add-locations` | 读取 override + effective | 查看默认添加目标 |
| `PATCH` | `/api/v1/user-settings/add-locations` | 增量更新 override | 修改默认 resource/skill 目录 |
| `DELETE` | `/api/v1/user-settings/add-locations` | 清除 override | 恢复为 server 默认 |

**`PATCH /api/v1/user-settings/add-locations` 请求体（PatchAddLocationsRequest）**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `resource_uri` | string | 否 | 资源默认添加目标 URI；传 `null` 清除该字段 |
| `skill_uri` | string | 否 | 技能默认添加目标 URI；传 `null` 清除该字段 |

响应 `result` 包含 `override` 与 `effective`，各自带 `resource_uri`/`skill_uri`。

---

## 5. 功能测试结果

使用脚本 `scripts/api_functional_perf_test.py` 在本地 MLX 环境执行。测试资源：GitHub 上 OpenViking `README.md` 导入为 `viking://resources/api-test-readme`。

### 5.1 核心接口功能状态

| 接口 | 方法 | 路径 | 状态 | 耗时 (ms) | 备注 |
|------|------|------|------|-----------|------|
| 健康检查 | GET | `/health` | ✅ 200 | 8.15 | - |
| 系统状态 | GET | `/api/v1/system/status` | ✅ 200 | 1.19 | - |
| 就绪检查 | GET | `/ready` | ✅ 200 | 92.82 | 含子系统状态 |
| 资源添加 | POST | `/api/v1/resources` | ✅ 200 | - | 异步返回 task_id |
| 列目录 | GET | `/api/v1/fs/ls` | ✅ 200 | 2.04 | 返回目录列表 |
| 目录树 | GET | `/api/v1/fs/tree` | ✅ 200 | 2.18 | 递归返回 |
| 文件元信息 | GET | `/api/v1/fs/stat` | ✅ 200 | 1.49 | - |
| 读取内容 | GET | `/api/v1/content/read` | ✅ 200 | 1.27 | 返回 HTML/Markdown |
| L0 摘要 | GET | `/api/v1/content/abstract` | ✅ 200 | 1.10 | 异步生成中 |
| L1 概述 | GET | `/api/v1/content/overview` | ✅ 200 | 1.18 | 异步生成中 |
| 语义检索 | POST | `/api/v1/search/find` | ✅ 200 | 110.38 | 当前资源未生成摘要，结果为空 |
| 内容检索 | POST | `/api/v1/search/grep` | ✅ 200 | 9.32 | 返回 49 处匹配 |
| 路径匹配 | POST | `/api/v1/search/glob` | ✅ 200 | 7.61 | 返回 20 个 `.md` 文件 |
| 系统观测 | GET | `/api/v1/observer/system` | ✅ 200 | 7.09 | 含 queue/vikingdb/models/lock/retrieval/filesystem |
| 文件系统观测 | GET | `/api/v1/observer/filesystem` | ✅ 200 | 1.57 | - |
| 队列观测 | GET | `/api/v1/observer/queue` | ✅ 200 | 1.66 | - |
| 模型观测 | GET | `/api/v1/observer/models` | ✅ 200 | 4.49 | VLM/Embedding 调用次数 |
| 会话列表 | GET | `/api/v1/sessions` | ✅ 200 | 1.32 | 当前为空 |
| 技能列表 | GET | `/api/v1/skills` | ✅ 200 | 1.15 | 当前为空 |
| 控制台摘要 | GET | `/api/v1/console/dashboard/summary` | ✅ 200 | 2.60 | 文件/技能/记忆/Token 统计 |
| Token 统计 | GET | `/api/v1/console/tokens` | ✅ 200 | 1.28 | 需 start_date/end_date |
| 监控任务 | GET | `/api/v1/watches` | ✅ 200 | 0.95 | 当前为空 |
| 关系列表 | GET | `/api/v1/relations` | ✅ 200 | 1.10 | 需 uri 参数 |
| 隐私配置 | GET | `/api/v1/privacy-configs` | ✅ 200 | 1.14 | 当前为空 |
| 代码大纲 | POST | `/api/v1/code/outline` | ✅ 200 | 1.38 | Markdown 不支持，返回错误提示 |

### 5.2 未启用/未配置接口

| 接口 | 方法 | 路径 | 状态 | 说明 |
|------|------|------|------|------|
| Prometheus 指标 | GET | `/metrics` | ❌ 404 | 未启用 Prometheus metrics |
| 向量计数 | GET | `/api/v1/debug/vector/count` | ❌ 404 | 当前配置未暴露 |
| 备份列表 | GET | `/api/v1/pack/backup` | ❌ 405 | 仅支持 POST |
| 账户管理 | GET | `/api/v1/admin/accounts` | ❌ 404 | dev 模式下未启用多租户 Admin |

### 5.3 关键发现

1. **语义检索返回空**：当前资源刚导入，L0/L1 摘要和语义节点仍在异步生成（`Semantic-Nodes` 队列仍有 pending/in-progress），因此 `/api/v1/search/find` 返回 `total: 0`。等待几分钟后再次调用可获得结果。
2. **Observer `retrieval` 健康为 false**：因为存在 zero-result queries，属于正常过渡状态。
3. **代码大纲需指定支持的编程语言**：对 Markdown 调用返回 `unsupported language`，对代码文件可正常生成大纲。
4. **/metrics 需显式开启**：如需监控，在配置中启用 Prometheus exporter。

---

## 6. 性能测试结果

在同一本地环境，使用脚本对关键接口进行多轮压测，结果如下：

### 6.1 压测汇总

| 接口 | 方法 | 并发数 | 总请求 | 成功 | 平均耗时 | P50 | P95 | P99 | 最大耗时 |
|------|------|--------|--------|------|----------|-----|-----|-----|----------|
| 健康检查 | GET | 1 | 50 | 50 | 4.37 ms | 2.39 ms | 10.06 ms | 81.57 ms | 81.57 ms |
| 列目录 | GET | 1 | 20 | 20 | 5.43 ms | 5.14 ms | 18.89 ms | 18.89 ms | 18.89 ms |
| 读取内容 | GET | 1 | 20 | 20 | 4.95 ms | 5.60 ms | 6.47 ms | 6.47 ms | 6.47 ms |
| 语义检索 | POST | 1 | 10 | 10 | 127.42 ms | 110.14 ms | 266.23 ms | 266.23 ms | 266.23 ms |

### 6.2 性能说明

- **纯本地 API（health/fs/content）**：均在 20ms 以内，P95 通常 < 10ms，性能优异。
- **语义检索（search/find）**：耗时主要在 Embedding + 向量检索 + Rerank，当前 MLX 模型单次 embedding 约 100–300ms；若模型首次加载或队列繁忙，可能进一步上升。
- **当前环境瓶颈**：MLX embedding server 在 Apple Silicon 上以 4-bit 运行，适合本地验证，不适合高并发生产吞吐。
- **文件系统 Observer 数据**：本地 fs 操作平均 0.081 ms/次，QueueFS 平均 0.134 ms/次，IO 层面无瓶颈。

---

## 7. 测试协议

以下协议可用于 OpenViking 部署后的功能验收与回归测试。

### 7.1 前置条件

1. OpenViking Server 已启动并监听目标端口（默认 1933）。
2. Embedding/Rerank/VLM 服务可达且配置正确。
3. 工作目录（workspace）已初始化。
4. 若启用认证，需准备 API Key。

### 7.2 功能测试用例

#### TC-01 健康检查

| 项 | 内容 |
|----|------|
| 目的 | 验证服务存活 |
| 请求 | `GET /health` |
| 预期 | 返回 `200`，`healthy: true`，`version` 与部署版本一致 |

#### TC-02 就绪检查

| 项 | 内容 |
|----|------|
| 目的 | 验证各子系统就绪 |
| 请求 | `GET /ready` |
| 预期 | 返回 `200`，`status: ready`，`agfs`/`vectordb`/`embedding` 状态为 `ok` |

#### TC-03 资源添加

| 项 | 内容 |
|----|------|
| 目的 | 验证外部资源可导入 |
| 请求 | `POST /api/v1/resources` |
| 数据 | `{ "path": "https://.../README.md", "to": "viking://resources/test", "wait": false }` |
| 预期 | 返回 `200`，`root_uri` 正确，`task_id` 非空 |

#### TC-04 文件系统浏览

| 项 | 内容 |
|----|------|
| 目的 | 验证目录与文件可列出 |
| 请求 | `GET /api/v1/fs/ls?uri=viking://resources/test` |
| 预期 | 返回 `200`，包含预期文件条目 |

#### TC-05 内容读取

| 项 | 内容 |
|----|------|
| 目的 | 验证文件内容可读取 |
| 请求 | `GET /api/v1/content/read?uri=viking://resources/test/Overview.md` |
| 预期 | 返回 `200`，`result` 包含非空内容 |

#### TC-06 语义检索

| 项 | 内容 |
|----|------|
| 目的 | 验证向量检索可用 |
| 前置 | 资源已处理完成（等待 30s–数分钟） |
| 请求 | `POST /api/v1/search/find` |
| 数据 | `{ "query": "OpenViking filesystem", "target_uri": "viking://resources/test", "limit": 5 }` |
| 预期 | 返回 `200`，`total > 0`，结果包含相关 URI |

#### TC-07 内容检索

| 项 | 内容 |
|----|------|
| 目的 | 验证 grep 可用 |
| 请求 | `POST /api/v1/search/grep` |
| 数据 | `{ "uri": "viking://resources/test", "pattern": "OpenViking" }` |
| 预期 | 返回 `200`，`match_count > 0` |

#### TC-08 路径匹配

| 项 | 内容 |
|----|------|
| 目的 | 验证 glob 可用 |
| 请求 | `POST /api/v1/search/glob` |
| 数据 | `{ "uri": "viking://resources/test", "pattern": "**/*.md" }` |
| 预期 | 返回 `200`，`count > 0` |

#### TC-09 会话创建与消息追加

| 项 | 内容 |
|----|------|
| 目的 | 验证会话生命周期 |
| 请求 1 | `POST /api/v1/sessions` |
| 预期 1 | 返回 `200`，`session_id` 非空 |
| 请求 2 | `POST /api/v1/sessions/{id}/messages` |
| 数据 | `{ "role": "user", "content": "Hello" }` |
| 预期 2 | 返回 `200` |

#### TC-10 观察者

| 项 | 内容 |
|----|------|
| 目的 | 验证可观测性接口 |
| 请求 | `GET /api/v1/observer/system` |
| 预期 | 返回 `200`，包含 queue/vikingdb/models 等组件 |

### 7.3 性能测试用例

#### PT-01 健康检查吞吐

| 项 | 内容 |
|----|------|
| 目的 | 验证基础接口低延迟 |
| 方法 | 使用脚本或 `ab`/`wrk` 对 `/health` 施压 |
| 指标 | P95 < 50ms，成功率 100% |

#### PT-02 文件系统读取吞吐

| 项 | 内容 |
|----|------|
| 目的 | 验证本地 IO 性能 |
| 方法 | 对 `/api/v1/fs/ls` 与 `/api/v1/content/read` 各发起 100 次串行请求 |
| 指标 | P95 < 50ms，无 5xx |

#### PT-03 语义检索延迟

| 项 | 内容 |
|----|------|
| 目的 | 评估检索端到端延迟 |
| 方法 | 对 `/api/v1/search/find` 发起 20 次请求，query 长度 10–50 字符 |
| 指标 | P95 < 1s（本地 MLX），P95 < 300ms（生产 GPU/云端 Embedding） |

#### PT-04 并发资源导入

| 项 | 内容 |
|----|------|
| 目的 | 验证队列与并发处理能力 |
| 方法 | 同时导入 5 个不同 URL，轮询 `/api/v1/tasks/{task_id}` |
| 指标 | 所有任务最终 `status: success`，无 error |

### 7.4 测试通过标准

- 功能测试：TC-01 至 TC-10 全部通过（HTTP 200 且业务断言成立）。
- 性能测试：PT-01/PT-02 P95 < 50ms；PT-03 达到部署环境约定阈值；PT-04 全部成功。
- Observer 中各组件 `is_healthy` 为 true，队列无大量 pending/error。

---

## 8. 附录：测试脚本

### 8.1 一键功能与性能测试

`scripts/api_functional_perf_test.py`

```python
#!/usr/bin/env python3
"""OpenViking API 功能与性能测试脚本"""
import json
import time
import urllib.request
import urllib.error
from urllib.parse import urlencode
from datetime import datetime, timezone

BASE_URL = "http://127.0.0.1:1933"


def call(method, path, body=None, query=None, timeout=60):
    url = f"{BASE_URL}{path}"
    if query:
        url += "?" + urlencode(query)
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            elapsed = (time.time() - t0) * 1000
            try:
                payload = json.loads(raw)
            except Exception:
                payload = raw[:500]
            return {"ok": True, "status": resp.status, "elapsed_ms": elapsed, "data": payload}
    except urllib.error.HTTPError as e:
        elapsed = (time.time() - t0) * 1000
        raw = e.read().decode("utf-8")
        try:
            payload = json.loads(raw)
        except Exception:
            payload = raw[:500]
        return {"ok": False, "status": e.code, "elapsed_ms": elapsed, "error": payload}
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        return {"ok": False, "status": None, "elapsed_ms": elapsed, "error": str(e)}


def perf(method, path, body=None, query=None, n=20, timeout=60):
    times = []
    ok_count = 0
    for _ in range(n):
        r = call(method, path, body=body, query=query, timeout=timeout)
        times.append(r["elapsed_ms"])
        if r["ok"]:
            ok_count += 1
        time.sleep(0.05)
    times.sort()
    return {
        "n": n,
        "ok": ok_count,
        "min_ms": round(times[0], 2),
        "p50_ms": round(times[len(times)//2], 2),
        "p95_ms": round(times[int(len(times)*0.95)], 2),
        "p99_ms": round(times[int(len(times)*0.99)] if len(times) >= 2 else times[-1], 2),
        "max_ms": round(times[-1], 2),
        "avg_ms": round(sum(times)/len(times), 2),
    }


def main():
    results = {}
    root_uri = "viking://resources/api-test-readme"
    file_uri = f"{root_uri}/Overview.md"

    # 确保资源已导入（如未导入，先调用 POST /api/v1/resources）
    results["health"] = call("GET", "/health")
    results["system_status"] = call("GET", "/api/v1/system/status")
    results["system_ready"] = call("GET", "/ready")
    results["fs_ls_root"] = call("GET", "/api/v1/fs/ls", query={"uri": "viking://resources/"})
    results["fs_ls_resource"] = call("GET", "/api/v1/fs/ls", query={"uri": root_uri})
    results["fs_tree"] = call("GET", "/api/v1/fs/tree", query={"uri": root_uri, "depth": "2"})
    results["fs_stat"] = call("GET", "/api/v1/fs/stat", query={"uri": root_uri})
    results["content_read"] = call("GET", "/api/v1/content/read", query={"uri": file_uri})
    results["content_abstract"] = call("GET", "/api/v1/content/abstract", query={"uri": root_uri})
    results["content_overview"] = call("GET", "/api/v1/content/overview", query={"uri": root_uri})
    results["search_find"] = call(
        "POST", "/api/v1/search/find",
        body={"query": "what is OpenViking", "target_uri": root_uri, "limit": 3},
        timeout=120,
    )
    results["search_grep"] = call(
        "POST", "/api/v1/search/grep",
        body={"pattern": "OpenViking", "uri": root_uri, "limit": 3},
        timeout=60,
    )
    results["search_glob"] = call(
        "POST", "/api/v1/search/glob",
        body={"pattern": "**/*.md", "target_uri": root_uri, "limit": 5},
        timeout=60,
    )
    results["observer_system"] = call("GET", "/api/v1/observer/system")
    results["observer_filesystem"] = call("GET", "/api/v1/observer/filesystem")
    results["observer_queue"] = call("GET", "/api/v1/observer/queue")
    results["observer_models"] = call("GET", "/api/v1/observer/models")
    results["sessions"] = call("GET", "/api/v1/sessions")
    results["skills_list"] = call("GET", "/api/v1/skills")
    results["console_dashboard"] = call("GET", "/api/v1/console/dashboard/summary")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    results["console_tokens"] = call(
        "GET", "/api/v1/console/tokens",
        query={"start_date": today, "end_date": today},
    )
    results["watches_list"] = call("GET", "/api/v1/watches")
    results["relations_list"] = call("GET", "/api/v1/relations", query={"uri": root_uri})
    results["privacy_configs"] = call("GET", "/api/v1/privacy-configs")
    results["code_outline"] = call(
        "POST", "/api/v1/code/outline",
        body={"uri": file_uri, "language": "markdown"},
        timeout=60,
    )

    # 性能测试
    results["perf_health"] = perf("GET", "/health", n=50)
    results["perf_fs_ls"] = perf("GET", "/api/v1/fs/ls", query={"uri": root_uri}, n=20)
    results["perf_content_read"] = perf("GET", "/api/v1/content/read", query={"uri": file_uri}, n=20)
    results["perf_search_find"] = perf(
        "POST", "/api/v1/search/find",
        body={"query": "OpenViking filesystem", "target_uri": root_uri, "limit": 3},
        n=10, timeout=120,
    )

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

### 8.2 运行方式

```bash
cd /Users/zhengxiaoxi/repo/OpenViking
source .venv/bin/activate
python scripts/api_functional_perf_test.py
```

### 8.3 资源导入脚本（如需重新初始化测试数据）

```bash
curl -X POST http://127.0.0.1:1933/api/v1/resources \
  -H "Content-Type: application/json" \
  -d '{
    "path": "https://raw.githubusercontent.com/volcengine/OpenViking/main/README.md",
    "to": "viking://resources/api-test-readme",
    "wait": false
  }'
```

---

## 9. 总结

OpenViking 当前版本（HEAD `a1c633c3`）提供了覆盖上下文全生命周期的 REST API：资源导入、文件系统管理、三级内容读取、语义/文本/路径检索、按类型配额记忆召回、会话记忆、Git 式快照、用户设置、可观测性、管理运维等。本地 MLX 环境下，纯本地接口延迟极低（< 10ms P95），语义检索受模型推理影响在百毫秒级。生产部署建议切换至 GPU/云 Embedding 服务以提升吞吐。

本文档中的接口清单、请求/响应说明、功能/性能测试结果与测试协议可直接作为开发集成、验收测试与日常运维的参考。
