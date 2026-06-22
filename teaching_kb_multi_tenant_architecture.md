# 教学知识库多租户架构设计

**设计日期**: 2026-05-18
**基于**: OpenViking 上下文数据库 v1.x
**场景**: 面向多学校的教学知识库系统（公共知识库 + 学校专有知识库 + 教师个人空间）

---

## 1. 整体架构设计

```
┌──────────────────────────────────────────────────────────────┐
│                    教学知识库应用后端                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ 学校/教师管理 │  │ 权限控制网关  │  │ 教案生成/评价引擎    │  │
│  │ (用户体系)   │  │ (API Key代理)│  │ (LLM编排层)         │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
└─────────┼────────────────┼────────────────────┼─────────────┘
          │                │                    │
          ▼                ▼                    ▼
┌──────────────────────────────────────────────────────────────┐
│              OpenViking 上下文数据库层                         │
│  ┌─────────────────┐ ┌─────────────────┐ ┌──────────────┐   │
│  │ resources/公共库 │ │ user/学校专有库  │ │ user/教师个人 │   │
│  │ (全量只读)       │ │ (学校内共享)     │ │ (仅本人可见)  │   │
│  └─────────────────┘ └─────────────────┘ └──────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ session/教师会话 (临时) → commit → 长期记忆提取        │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. 三层知识库隔离方案

OpenViking 的访问控制逻辑（`openviking/core/namespace.py:234-266`）决定了隔离策略：

| Scope | 访问规则 | 适用场景 |
|-------|---------|---------|
| `resources` | **所有认证用户可读** | 公共知识库（政策、通用教材） |
| `user` | **仅匹配 user_id 可访问** | 学校专有库、教师个人数据 |
| `agent` | **仅匹配 agent_id 可访问** | 可选：按学科/年级隔离 |

### 2.1 公共知识库（全局只读）

```python
# URI: viking://resources/teaching_kb/public/
# 权限：所有学校/教师均可检索
POST /api/v1/search/find
{
    "query": "双减政策作业设计",
    "target_uri": "viking://resources/teaching_kb/public/policies",
    "limit": 5
}
```

### 2.2 学校专有知识库（学校内共享，跨校隔离）

**核心问题**：OpenViking 没有原生"学校"这一隔离层级，`resources` 又是全局可读的。因此学校专有库必须放在 **`user` scope** 下，用一个**伪用户**代表学校：

```python
# 为每个学校创建专属 user namespace
school_user_id = f"school_{school_id}"  # 如 school_10086

# 学校专有库 URI 结构
viking://user/school_10086/kb/policies/       # 校本政策
viking://user/school_10086/kb/textbooks/      # 校本教材
viking://user/school_10086/kb/lesson_plans/   # 校内优秀教案
```

**访问控制**：教学应用后端持有 OpenViking 的 **ADMIN 或 ROOT API Key**，代表学校发起请求。教师不直接调用 OpenViking，所有请求由后端代理，后端根据登录教师的 `school_id` 拼接对应的 `target_uri`。

```python
# 后端代理搜索学校专有库
headers = {
    "X-API-Key": ROOT_KEY,
    "X-OpenViking-Account": "teaching_platform",
    "X-OpenViking-User": f"school_{school_id}",  # 切换到学校的 namespace
}

POST /api/v1/search/find
{
    "query": "分数初步认识教案",
    "target_uri": f"viking://user/school_{school_id}/kb/lesson_plans",
    "limit": 5
}
```

> **为什么不用 `resources` 存学校库？**
> `is_accessible()` 源码显示 `resources` scope 对所有认证用户返回 `True`，无法做数据隔离。`user` scope 会校验 `owner_user_id == ctx.user.user_id`，是真正的隔离边界。

### 2.3 教师个人空间（仅本人可见）

```python
teacher_user_id = f"teacher_{teacher_id}"  # 如 teacher_2048

# 个人数据 URI
viking://user/teacher_2048/memories/          # 长期记忆（commit 提取）
viking://user/teacher_2048/lesson_plans/      # 个人生成的教案
viking://user/teacher_2048/preferences/       # 教学偏好
```

---

## 3. 教师会话与经验沉淀（Session → Commit → Memory）

OpenViking 的 `session` + `commit` 机制非常适合记录教师的备课对话并沉淀经验。

### 3.1 每个教师独立 Session

```python
# 教师登录后，后端为其创建/获取专属 session
POST /api/v1/sessions
{
    "session_id": "teacher_2048_20250518_math"
}

# 返回
{
    "session_id": "teacher_2048_20250518_math",
    "user": {
        "account_id": "teaching_platform",
        "user_id": "teacher_2048",
        "agent_id": "lesson_plan_assistant"
    }
}
```

### 3.2 备课对话记录

```python
# 记录教师输入
POST /api/v1/sessions/{session_id}/messages
{
    "role": "user",
    "content": "帮我生成一个小学三年级分数初步认识的教案，要求融入双减政策"
}

# 记录生成结果（含检索上下文）
POST /api/v1/sessions/{session_id}/messages
{
    "role": "assistant",
    "parts": [
        {"type": "text", "text": "生成的教案正文..."},
        {"type": "context", "uri": "viking://resources/.../双减政策解读.md", "context_type": "resource"},
        {"type": "context", "uri": "viking://resources/.../分数初步认识-背景知识.md", "context_type": "resource"}
    ]
}
```

### 3.3 经验沉淀：Commit 提取长期记忆

```python
# 备课结束后，commit session
POST /api/v1/sessions/{session_id}/commit
{
    "keep_recent_count": 5   # 保留最近5条消息继续下一轮对话
}
```

Commit 的底层行为（`openviking/server/routers/sessions.py:252-276`）：
1. **Phase 1（同步）**：归档会话历史到 `viking://session/{session_id}/history/`
2. **Phase 2（异步）**：提取长期记忆，写入教师个人空间：
   - `viking://user/teacher_2048/memories/preferences/` — 教学偏好（如"喜欢情境导入"）
   - `viking://user/teacher_2048/memories/entities/` — 关注的实体（如"分数初步认识"）
   - `viking://user/teacher_2048/memories/events/` — 备课事件记录

### 3.4 优秀教案回写知识库

教师确认生成的教案质量合格后，后端将其写入**学校专有库**（供本校其他教师参考）：

```python
# 写入学校专有教案库
POST /api/v1/content/write
{
    "uri": "viking://user/school_10086/kb/lesson_plans/小学三年级/数学/分数初步认识-generated-2048.md",
    "content": "生成的教案内容...",
    "mode": "create"
}

# 触发向量化
POST /api/v1/content/reindex
{
    "uri": "viking://user/school_10086/kb/lesson_plans",
    "wait": true
}
```

---

## 4. 跨库检索策略

教师搜索时，后端需要**并行查询三层知识库**，合并去重后返回：

```python
async def search_for_teacher(query, teacher_id, school_id):
    # 1. 公共知识库
    public_results = await search(
        query=query,
        target_uri="viking://resources/teaching_kb/public"
    )

    # 2. 学校专有库（切换 X-OpenViking-User 为 school_xxx）
    school_results = await search(
        query=query,
        target_uri=f"viking://user/school_{school_id}/kb"
    )

    # 3. 教师个人库
    personal_results = await search(
        query=query,
        target_uri=f"viking://user/teacher_{teacher_id}"
    )

    # 4. 合并、去重、按分数排序
    return merge_results(public_results, school_results, personal_results)
```

**检索优先级建议**：
- 教师个人库 > 学校专有库 > 公共知识库
- 个人库命中表明教师之前备过类似课程，优先呈现
- 学校库体现校本特色
- 公共库提供课标和政策基准

---

## 5. OpenViking 能否满足？评估总结

| 需求 | OpenViking 支持度 | 实现方式 |
|------|------------------|---------|
| **公共知识库** | 完全满足 | `viking://resources/` 全局共享 |
| **学校专有库隔离** | 可满足（需设计） | `user` scope + 伪用户 + 后端代理 |
| **教师个人隔离** | 完全满足 | `user` scope 原生隔离 |
| **教师独立 Session** | 完全满足 | `/api/v1/sessions/*` 全套 API |
| **经验沉淀到知识库** | 完全满足 | `commit` 自动提取记忆 + `content/write` 回写教案 |
| **跨库统一检索** | 完全满足 | 后端并行调用 `find` + 合并结果 |

---

## 6. 关键配置建议

### 6.1 学校隔离配置

如果希望进一步增强隔离，可启用 `AccountNamespacePolicy`：

```json
// ov.conf 中每个 school account 配置
{
  "account_namespace_policy": {
    "isolate_user_scope_by_agent": true,
    "isolate_agent_scope_by_user": false
  }
}
```

这会让 user namespace 进一步按 agent 隔离：`viking://user/school_10086/agent/{agent_id}/...`

### 6.2 API Key 管理

```python
# 建议 Key 层级
- ROOT Key: 平台超级管理员（仅运维持有）
- ADMIN Key: 每个学校一个（学校管理员持有，可管理本校教师）
- USER Key: 每个教师一个（仅用于个人空间，无法跨校访问）
```

### 6.3 注意点

1. **`resources` 不可用于敏感数据** — 任何认证用户都可读取，学校保密内容必须放 `user` scope
2. **`session` 是临时存储** — 必须通过 `commit` 才能沉淀为长期记忆，否则会话结束后数据仍在但不会被提取为结构化记忆
3. **后端是安全边界** — 教师终端不持有 OpenViking API Key，所有请求经后端代理，后端负责校验教师身份和所属学校

---

## 7. URI 命名规范建议

```
# 公共知识库
viking://resources/teaching_kb/public/
├── policies/                    # 国家/省级政策（全局共享）
├── textbooks/                   # 通用教材背景知识
└── lesson_plans/                # 平台精选优秀教案

# 学校专有知识库（按 school_id 隔离）
viking://user/school_{school_id}/kb/
├── policies/                    # 校本政策、校内规定
├── textbooks/                   # 校本教材、校本课程
└── lesson_plans/                # 校内教师共享教案

# 教师个人空间（按 teacher_id 隔离）
viking://user/teacher_{teacher_id}/
├── memories/
│   ├── preferences/             # 教学风格偏好
│   ├── entities/                # 关注的知识点/班级
│   └── events/                  # 备课/授课记录
├── lesson_plans/                # 个人生成的教案
└── sessions/                    # 会话归档（由 commit 自动生成）
```

---

*文档生成时间: 2026-05-18*
*关联文档: teaching_kb_assessment_report.md, teaching_kb_test_plan.md*
