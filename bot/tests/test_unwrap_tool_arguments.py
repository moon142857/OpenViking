"""unwrap_tool_arguments 截断抢救测试。

背景：Kimi 网关在工具参数约 6KB 以上时，把真实参数转义塞进 raw_arguments
且内层字符串是截断的（Unterminated string）。旧行为直接原样返回，校验报
"missing required path/content"，模型反复重试同一大参数全部失败。
新行为：尽力抢救已收到字段并打 TRUNCATED_MARKER 标记，由工具回告模型分段重写。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vikingbot.providers.base import TRUNCATED_MARKER, unwrap_tool_arguments


def test_normal_wrap_still_unwraps():
    wrapped = {"raw_arguments": json.dumps({"path": "/a.md", "content": "hi"})}
    assert unwrap_tool_arguments(wrapped) == {"path": "/a.md", "content": "hi"}


def test_unrelated_input_passthrough():
    assert unwrap_tool_arguments({"a": 1, "b": 2}) == {"a": 1, "b": 2}
    assert unwrap_tool_arguments("not-a-dict") == {}
    assert unwrap_tool_arguments(None) == {}
    assert unwrap_tool_arguments({"raw_arguments": 123}) == {"raw_arguments": 123}


def test_truncated_path_first_salvaged():
    # 真实失败形态：{"path": "...", "content": "...<截断>
    inner = (
        '{"path": "/workspace/shared/汇编.md", "content": "# 标题\\n\\n> 正文'
        + "很长" * 100  # 截断在中途，无闭合引号
    )
    out = unwrap_tool_arguments({"raw_arguments": inner})
    assert out[TRUNCATED_MARKER] is True
    assert out["path"] == "/workspace/shared/汇编.md"
    assert out["content"].startswith("# 标题\n\n> 正文")
    assert len(out["content"]) > 200


def test_truncated_content_first_keeps_content_only():
    # 真实失败形态：content 在前、path 还没出现就被截断 → 只有 content，
    # 校验照常报 missing path（不能凭空造路径）
    inner = '{"content": "# 文档\\n\\n' + "甲" * 500
    out = unwrap_tool_arguments({"raw_arguments": inner})
    assert out[TRUNCATED_MARKER] is True
    assert "path" not in out
    assert out["content"].startswith("# 文档\n\n")


def test_escapes_decoded_and_trailing_escape_safe():
    # 转义序列正确解码；截断点落在 \uXXXX 中间时安全截断不抛异常
    inner = '{"path": "a.md", "content": "引号\\"换行\\n制表\\t统一码\\u4e2d\\u5'
    out = unwrap_tool_arguments({"raw_arguments": inner})
    assert out[TRUNCATED_MARKER] is True
    assert out["path"] == "a.md"
    assert out["content"] == '引号"换行\n制表\t统一码中'


def test_garbage_falls_back_to_original():
    wrapped = {"raw_arguments": "not json at all"}
    assert unwrap_tool_arguments(wrapped) == wrapped


def test_raw_key_with_broken_outer_also_handled():
    # build_stream_response 外层解析失败时包成 {"raw": <原始字符串>}：
    # 抢救不出字段则原样返回（行为与旧版一致）
    wrapped = {"raw": '{"raw_arguments": "{\\"path\\": \\"a.md\\", \\"content\\": \\"x'}
    out = unwrap_tool_arguments(wrapped)
    assert "path" not in out  # 双层转义不再深入，保持回退
