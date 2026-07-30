# VikingBot 接口集成参考

> 给另一个前后端集成这套 bot 用。生成于 2026-07-30。

## 一、服务地址(经 nginx HTTPS 暴露)

bot 不直接暴露,经 OpenViking 后端代理。**两个库各自一套**:

| 库 | bot 基础地址 | 用途 |
|---|---|---|
| 教材库 | `https://82.156.210.246:8443/bot/v1` | 教材知识 + Kimi 联网 |
| 课标库 | `https://82.156.210.246:8444/bot/v1` | 课标知识 + Kimi 联网 |

> 内部链路:nginx(8443/8444)→ 后端(1933/1934)→ bot gateway(18791/18790,loopback)。
> 后端把 `/bot/v1/*` 代理给 bot,你只需调后端地址即可,不用管 bot 端口。

## 二、鉴权(关键)

**复用 OpenViking 的 API Key**,二选一:
- `X-API-Key: <viewer 或 admin key>`
- `Authorization: Bearer <viewer 或 admin key>`

```
VIEWER_TEXTBOOKS   = ZGVmYXVsdA.dmlld2Vy.MjYzYTVkNWZmMjEwZjM1NDhjNzYwYWEyMDY3NGU2YWUyMTFhMTBjMGU1NjVmNTBlNjYwYjY5OWVjM2IwNWJjYg   # 教材库 viewer(只读)
VIEWER_CURRICULUM  = ZGVmYXVsdA.dmlld2Vy.MzRiZTkxOWEzY2RiODE2MDU4MWM0MTI2YjViNDk1YTRlOTQwYjJmZDMwMzljM2MyZTAyMmJlZTdjN2VlZDM5Zg   # 课标库 viewer(只读)
```
- viewer key:能对话/检索/联网搜索(只读),**够前端用**
- admin key:额外能写记忆/写库(见 `~/.openviking/.keys.env`)
- ⚠️ 教材库和课标库是**不同的 key**,别填串。教材库(8443)用 `VIEWER_TEXTBOOKS`,课标库(8444)用 `VIEWER_CURRICULUM`。

## 三、接口清单(全部在 /bot/v1 下)

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/bot/v1/health` | 健康检查 |
| POST | `/bot/v1/chat` | **对话(同步,最常用)** |
| POST | `/bot/v1/chat/stream` | 对话(SSE 流式,前端推荐) |
| POST | `/bot/v1/chat/channel` | 对话(指定 bot 频道,同步) |
| POST | `/bot/v1/chat/channel/stream` | 对话(指定频道,流式) |
| GET | `/bot/v1/sessions` | 列出会话 |
| POST | `/bot/v1/sessions` | 创建会话 |
| GET | `/bot/v1/sessions/{id}` | 会话详情(含消息历史) |
| DELETE | `/bot/v1/sessions/{id}` | 删除会话 |
| POST | `/bot/v1/feedback` | 对某条回复提交反馈 |

## 四、核心接口:POST /bot/v1/chat(同步)

**请求体**(JSON,只有 `message` 必填):
```jsonc
{
  "message": "小学三年级分数怎么教?结合课标",   // 必填,用户消息
  "session_id": "可选,传了就接着这个会话聊",       // 可选,不传则新建
  "user_id": "可选,用户标识",                      // 可选
  "stream": false,                                 // 这里用同步,设 true 走流式
  "context": null,                                  // 可选,附加历史消息
  "need_reply": true,                              // 可选,默认 true
  "channel_id": null,                              // 可选,多频道路由
  "disabled_tools": []                              // 可选,本次禁用的工具名
}
```

**响应体**:
```jsonc
{
  "session_id": "01b3f1ed-...",          // 会话 ID(下次对话传回接着聊)
  "response_id": "1d1c947a...",         // 回复 ID(给 feedback 用)
  "message": "分数是表示整体被等分...",   // 最终答案(已融合库内检索+联网)
  "events": [                            // 中间过程(思考/工具调用),可能为 null(简单问题)
    {"type": "tool_call", "data": {...}},
    {"type": "reasoning", "data": "..."}
  ],
  "relevant_memories": null,             // 检索到的相关记忆
  "token_usage": {                       // token 统计
    "prompt_tokens": 5021,
    "completion_tokens": 107,
    "total_tokens": 5128
  },
  "timestamp": "2026-07-30T13:25:56.058721"
}
```

**curl 示例**:
```bash
curl -X POST https://82.156.210.246:8444/bot/v1/chat \
  -H "Authorization: Bearer ZGVmYXVsdA.dmlld2Vy.MzRi..." \
  -H "Content-Type: application/json" \
  -k \   # 自签证书,客户端需跳过校验或导入 CA
  -d '{"message":"什么是课程标准","chat_id":"x"}'
```
> 注:`chat_id` 不是标准字段,用 `session_id`。第一次不传,拿到响应里的 `session_id` 后续传回。

## 五、流式接口:POST /bot/v1/chat/stream(前端推荐)

**SSE(Server-Sent Events)**,`Content-Type: text/event-stream`。请求体同 /chat(可省略 `stream` 字段,会自动设 true)。

**事件类型**(每条 `data:` 后是 JSON `{"event","data","timestamp}`):
| event | 含义 | 前端怎么用 |
|---|---|---|
| `iteration` | agent 循环第 N 轮 | 显示"思考中...第N步" |
| `reasoning` | 思考内容(thinking) | 折叠展示推理过程 |
| `reasoning_delta` | 思考内容增量 | 流式拼接推理文本 |
| `tool_call` | 要调工具了 | 显示"正在搜索知识库/联网..." |
| `tool_result` | 工具结果 | 可选展示检索到的片段 |
| `content_delta` | 答案文本增量 | **流式拼接到对话气泡** |
| `response` | 最终完整答案(含 `response_id`) | 收尾,确认完整文本 |
| `no_reply` | 无需回复 | 结束 |

> 注意字段名是 **`event`**(不是 `type`)。`response` 事件的 `data` 是 `{"content": "...", "response_id": "..."}`。简单问题(如算术)可能只发一个 `response` 事件就结束;复杂问题才有完整 iteration/tool_call 链。

**前端解析(伪代码)**:
```javascript
const es = new EventSource(...) // 或用 fetch + ReadableStream(因为有自定义 header,EventSource 不支持,用 fetch)
// 实际:用 fetch 流式读
const resp = await fetch('https://82.156.210.246:8444/bot/v1/chat/stream', {
  method: 'POST',
  headers: {'Authorization':'Bearer <key>', 'Content-Type':'application/json'},
  body: JSON.stringify({message:'...', session_id:'...'})
});
const reader = resp.body.getReader();
// 逐行解析 "data: {...}\n\n",按 type 分发:content_delta 追加到答案区,tool_call 显示状态...
```

> ⚠️ 流式接口要带 `Authorization` 头,**不能用浏览器原生 `EventSource`**(它不支持自定义头)。用 `fetch` + `ReadableStream` 解析 SSE,或用 `@microsoft/fetch-event-source` 库。

## 六、会话管理(多轮对话)

bot 支持有状态多轮对话。流程:
1. 第一次:`POST /bot/v1/chat`,`message` 里不传 `session_id`。响应给 `session_id`。
2. 后续:把 `session_id` 传回去,bot 记得上下文。

```
# 列会话
GET /bot/v1/sessions

# 创建空会话(可选)
POST /bot/v1/sessions  body: {"user_id":"可选","metadata":{}}

# 看某会话历史
GET /bot/v1/sessions/{session_id}

# 删会话
DELETE /bot/v1/sessions/{session_id}
```

## 七、反馈接口(可选,质量优化)

```bash
POST /bot/v1/feedback
body: {
  "response_id": "上面 chat 响应里的 response_id",
  "feedback": "positive" / "negative",
  "comment": "可选文字"
}
```

## 八、前端集成要点

1. **CORS**:后端 `cors_origins` 已收紧。如果你的另一个前端域名不在白名单,会跨域。需在对应 ov.conf 的 `server.cors_origins` 加你的前端域名,重启服务。
2. **自签证书**:前端调 `https://82.156.210.246:8443` 会证书报错。要么客户端导入 CA(见 ACCESS.md 第四节),要么开发期跳过校验。
3. **超时**:thinking + 多轮 agent 可能较慢(几十秒)。前端请求超时设长(如 180s),或用流式接口避免长时间空白等待。
4. **两个库二选一**:教材问题调 8443,课标问题调 8444。如果要"一个问题同时查两个库",得前端分别调两个再自己合并(后端不提供跨库)。
5. **工具透明**:bot 内部会自动调 `openviking_search`(查库)/ `web_search`(kimi 联网)/ `openviking_read` 等,你不用管,只管发 message 收 message。流式时可通过 `tool_call` 事件展示"正在查..."。

## 九、快速验证(集成前先测通)

```bash
# 1. 健康检查
curl -k https://82.156.210.246:8444/bot/v1/health -H "Authorization: Bearer <课标viewer_key>"

# 2. 同步对话
curl -k -X POST https://82.156.210.246:8444/bot/v1/chat \
  -H "Authorization: Bearer <课标viewer_key>" -H "Content-Type: application/json" \
  -d '{"message":"小学科学电路单元的核心知识点"}'

# 3. 流式对话(看 SSE 事件)
curl -k -N -X POST https://82.156.210.246:8444/bot/v1/chat/stream \
  -H "Authorization: Bearer <课标viewer_key>" -H "Content-Type: application/json" \
  -d '{"message":"结合课标分析电路教学"}'
```

## 十、相关文件(代码层)

| 文件 | 内容 |
|---|---|
| `bot/vikingbot/channels/openapi.py` | 所有路由定义 + 鉴权(`verify_gateway_request`) |
| `bot/vikingbot/channels/openapi_models.py` | 请求/响应模型(ChatRequest 等) |
| `bot/vikingbot/bus/events.py` | SSE 事件类型枚举(OutboundEventType) |
| `bot/vikingbot/agent/loop.py` | agent ReAct 循环(思考+工具) |
| `bot/vikingbot/agent/tools/` | 工具实现(search/read/glob/web 等) |
