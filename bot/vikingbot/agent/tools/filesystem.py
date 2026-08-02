"""File system tools: read, write, edit."""

from typing import TYPE_CHECKING, Any

from vikingbot.agent.tools.base import Tool

if TYPE_CHECKING:
    from vikingbot.agent.tools.base import ToolContext


class ReadFileTool(Tool):
    """Tool to read file contents."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file at the given path."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "The file path to read"}},
            "required": ["path"],
        }

    async def execute(self, tool_context: "ToolContext", path: str, **kwargs: Any) -> str:
        try:
            sandbox = await tool_context.sandbox_manager.get_sandbox(tool_context.session_key)
            content = await sandbox.read_file(path)
            return content
        except FileNotFoundError as e:
            return f"Error: {e}"
        except IOError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error reading file: {str(e)}"


class WriteFileTool(Tool):
    """Tool to write content to a file."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file at the given path. Creates parent directories if needed."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to write to"},
                "content": {"type": "string", "description": "The content to write"},
            },
            "required": ["path", "content"],
        }

    async def execute(
        self, tool_context: "ToolContext", path: str, content: str, **kwargs: Any
    ) -> str:
        try:
            sandbox = await tool_context.sandbox_manager.get_sandbox(tool_context.session_key)
            await sandbox.write_file(path, content)
            # __ov_truncated__：providers/base.py 从被上游截断的参数里抢救出的部分内容。
            # 必须明确回告模型"文档不完整"，否则它以为写成功，用户拿到残缺文件。
            if kwargs.get("__ov_truncated__"):
                return (
                    f"Warning: 工具参数在传输中被上游网关截断，内容不完整，"
                    f"仅前 {len(content)} 个字符已写入 {path}。"
                    f"请把剩余内容拆小（每段不超过约 2000 字符），用 edit_file 逐段追加补齐"
                    f"（old_text 取文件末尾几个字，new_text 为末尾原文+续写内容）。"
                )
            return f"Successfully wrote {len(content)} bytes to {path}"
        except IOError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error writing file: {str(e)}"


class EditFileTool(Tool):
    """Tool to edit a file by replacing text."""

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return "Edit a file by replacing old_text with new_text. The old_text must exist exactly in the file."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to edit"},
                "old_text": {"type": "string", "description": "The exact text to find and replace"},
                "new_text": {"type": "string", "description": "The text to replace with"},
            },
            "required": ["path", "old_text", "new_text"],
        }

    async def execute(
        self, tool_context: "ToolContext", path: str, old_text: str, new_text: str, **kwargs: Any
    ) -> str:
        try:
            sandbox = await tool_context.sandbox_manager.get_sandbox(tool_context.session_key)
            content = await sandbox.read_file(path)

            if old_text not in content:
                return "Error: old_text not found in file. Make sure it matches exactly."

            count = content.count(old_text)
            if count > 1:
                return f"Warning: old_text appears {count} times. Please provide more context to make it unique."

            new_content = content.replace(old_text, new_text, 1)
            await sandbox.write_file(path, new_content)

            return f"Successfully edited {path}"
        except FileNotFoundError as e:
            return f"Error: {e}"
        except IOError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error editing file: {str(e)}"


class ListDirTool(Tool):
    """Tool to list directory contents."""

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "List the contents of a directory."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "The directory path to list"}},
            "required": ["path"],
        }

    async def execute(self, tool_context: "ToolContext", path: str, **kwargs: Any) -> str:
        try:
            sandbox = await tool_context.sandbox_manager.get_sandbox(tool_context.session_key)
            items = await sandbox.list_dir(path)

            if not items:
                return f"Directory {path} is empty"

            formatted_items = []
            for name, is_dir in items:
                prefix = "📁 " if is_dir else "📄 "
                formatted_items.append(f"{prefix}{name}")

            return "\n".join(formatted_items)
        except FileNotFoundError as e:
            return f"Error: {e}"
        except IOError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error listing directory: {str(e)}"
