"""Base LLM provider interface."""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Literal


@dataclass
class ToolCallRequest:
    """A tool call request from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]
    tokens: int


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    reasoning_content: str | None = None  # Kimi, DeepSeek-R1 etc.

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


@dataclass
class LLMStreamEvent:
    """Streaming event emitted by an LLM provider."""

    type: Literal["content_delta", "reasoning_delta", "response"]
    content: str | None = None
    response: LLMResponse | None = None


def stream_delta_value(delta: Any, name: str) -> str:
    value = getattr(delta, name, None)
    return value if isinstance(value, str) else ""


def merge_stream_tool_call_delta(
    raw_tool_calls: dict[int, dict[str, Any]],
    delta_tool_call: Any,
    fallback_index: int | None = None,
) -> None:
    index = getattr(delta_tool_call, "index", None)
    if index is None:
        index = fallback_index if fallback_index is not None else len(raw_tool_calls)
    entry = raw_tool_calls.setdefault(
        int(index),
        {"id": "", "name": "", "arguments": ""},
    )
    tool_call_id = getattr(delta_tool_call, "id", None)
    if tool_call_id:
        entry["id"] = tool_call_id
    function = getattr(delta_tool_call, "function", None)
    if function is None:
        return
    name = getattr(function, "name", None)
    if name:
        entry["name"] += name
    arguments = getattr(function, "arguments", None)
    if arguments:
        entry["arguments"] += arguments


# 抢救出的参数 dict 上打的截断标记：工具（如 write_file）据此在结果里
# 回告模型"内容被截断，请分段重写"，而不是静默写出一个残缺文件还报成功。
TRUNCATED_MARKER = "__ov_truncated__"

_JSON_ESCAPES = {
    '"': '"', "\\": "\\", "/": "/",
    "n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f",
}


def _try_parse_json(text: str) -> Any:
    """先严格解析；失败再用宽松模式（容忍字面控制字符）重试一次。"""
    for strict in (True, False):
        try:
            return json.loads(text, strict=strict)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _salvage_truncated_object(text: str) -> dict[str, Any]:
    """从被上游截断的扁平 JSON 对象字符串里抢救字符串字段。

    工具参数的实际形状是 `{"path": "...", "content": "...<截断>` 这类扁平对象：
    完整闭合的字段正常解码，最后一个未闭合的字段取已收到的前缀。抠不出任何
    字段时返回 {}（调用方回退到原始参数，校验照常报 missing required）。
    """
    out: dict[str, Any] = {}
    i, n = 0, len(text)
    while i < n:
        m = re.search(r'"([^"\\]+)"\s*:\s*"', text[i:])
        if not m:
            break
        key = m.group(1)
        j = i + m.end()  # 值内容起点（开引号之后）
        chars: list[str] = []
        closed = False
        while j < n:
            c = text[j]
            if c == "\\" and j + 1 < n:
                nxt = text[j + 1]
                if nxt == "u" and j + 5 < n:
                    try:
                        chars.append(chr(int(text[j + 2:j + 6], 16)))
                    except ValueError:
                        break  # 截断在 \\uXXXX 中间
                    j += 6
                    continue
                if nxt in _JSON_ESCAPES:
                    chars.append(_JSON_ESCAPES[nxt])
                    j += 2
                    continue
                break  # 无法识别的转义：保守截断
            if c == '"':
                closed = True
                j += 1
                break
            chars.append(c)
            j += 1
        if key not in out:
            out[key] = "".join(chars)
        if not closed:
            break  # 后面的字段必然不完整
        i = j
    return out


def unwrap_tool_arguments(arguments: Any) -> dict[str, Any]:
    """展开部分模型/网关在工具参数较长时产生的套娃结构。

    某些 provider（如 Kimi/moonshot）在工具参数 payload 较长时，会把真实参数整体
    JSON 转义后塞进单个 `raw_arguments`（或 `raw`）字符串字段，导致顶层缺少工具
    真正需要的参数（如 `path`/`content`/`command`），校验时报 "missing required ..."。

    本函数在解析出 dict 后调用：若该 dict 恰好只含 `raw_arguments`/`raw` 一个键、
    且其值为可继续解析的 JSON 字符串，则用内层解析结果替换；否则原样返回。

    补充：上游（Kimi 网关）在参数约 6KB 以上时，内层字符串本身是截断的
    （Unterminated string）。此时尽力抢救已收到的字段并打 TRUNCATED_MARKER
    标记，让工具回告模型分段重写，而不是全部丢失误报成参数缺失。
    """
    if not isinstance(arguments, dict) or len(arguments) != 1:
        return arguments if isinstance(arguments, dict) else {}
    (key, value), = arguments.items()
    if key not in ("raw_arguments", "raw") or not isinstance(value, str):
        return arguments
    inner = _try_parse_json(value)
    if isinstance(inner, dict):
        return inner
    salvaged = _salvage_truncated_object(value)
    if salvaged:
        salvaged[TRUNCATED_MARKER] = True
        return salvaged
    return arguments


def build_stream_response(
    *,
    content: str,
    reasoning_content: str,
    raw_tool_calls: dict[int, dict[str, Any]],
    finish_reason: str,
    usage: dict[str, int] | None = None,
    token_counter: Callable[[str, str], int] | None = None,
) -> LLMResponse:
    tool_calls: list[ToolCallRequest] = []
    for index in sorted(raw_tool_calls):
        raw_tool_call = raw_tool_calls[index]
        name = str(raw_tool_call.get("name") or "")
        if not name:
            continue
        raw_arguments = str(raw_tool_call.get("arguments") or "")
        arguments: dict[str, Any]
        if raw_arguments:
            try:
                parsed_arguments = json.loads(raw_arguments)
                arguments = (
                    unwrap_tool_arguments(parsed_arguments)
                    if isinstance(parsed_arguments, dict)
                    else {}
                )
            except json.JSONDecodeError:
                arguments = {"raw": raw_arguments}
        else:
            arguments = {}
        tokens = token_counter(name, raw_arguments) if token_counter else 0
        tool_calls.append(
            ToolCallRequest(
                id=str(raw_tool_call.get("id") or f"tool_call_{index}"),
                name=name,
                arguments=arguments,
                tokens=tokens,
            )
        )

    return LLMResponse(
        content=content or None,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage or {},
        reasoning_content=reasoning_content or None,
    )


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Implementations should handle the specifics of each provider's API
    while maintaining a consistent interface.
    """

    def __init__(self, api_key: str | None = None, api_base: str | None = None):
        self.api_key = api_key
        self.api_base = api_base

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        session_id: str | None = None,
    ) -> LLMResponse:
        """
        Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions.
            model: Model identifier (provider-specific).
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
            session_id: Optional session ID for tracing.

        Returns:
            LLMResponse with content and/or tool calls.
        """
        pass

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        session_id: str | None = None,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Stream a chat completion request.

        Providers without native streaming fall back to a single final response event.
        """
        response = await self.chat(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            session_id=session_id,
        )
        yield LLMStreamEvent(type="response", response=response)

    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        pass
