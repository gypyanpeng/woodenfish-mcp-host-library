"""MCP 主机的错误定义。"""

from typing import Any


class MCPHostError(Exception):
    """MCP 主机错误的基类。"""


class ThreadNotFoundError(MCPHostError):
    """线程未找到时抛出的异常。"""

    def __init__(self, thread_id: str) -> None:
        """初始化错误。

        参数：
            thread_id: 未找到的线程ID。
        """
        self.thread_id = thread_id
        super().__init__(f"Thread {thread_id} not found")


class ThreadQueryError(MCPHostError):
    """查询无效时抛出的异常。"""

    def __init__(
        self,
        query: Any,
        state_values: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        """初始化错误。

        参数：
            query: 无效的查询内容。
            state_values: 线程状态值。
            error: 发生的异常。
        """
        self.query = query
        self.state_values = state_values
        self.error = error
        super().__init__(f"Error in query, {error}")


class GraphNotCompiledError(MCPHostError):
    """图未编译时抛出的异常。"""

    def __init__(self, thread_id: str | None = None) -> None:
        """初始化错误。

        参数：
            thread_id: 未找到的线程ID。
        """
        self.thread_id = thread_id
        super().__init__(f"Graph not compiled for thread {thread_id}")


class MessageTypeError(MCPHostError, ValueError):
    """消息类型不正确时抛出的异常。"""

    def __init__(self, msg: str | None = None) -> None:
        """初始化错误。"""
        if msg is None:
            msg = "Message is not the correct type"
        super().__init__(msg)


class InvalidMcpServerError(MCPHostError, ValueError):
    """MCP 服务器无效时抛出的异常。"""

    def __init__(self, mcp_server: str, reason: str | None = None) -> None:
        """初始化错误。"""
        if reason is None:
            reason = "Invalid MCP server"
        super().__init__(f"{mcp_server}: {reason}")


class McpSessionGroupError(MCPHostError, ValueError, BaseExceptionGroup):
    """MCP 会话错误的异常组。"""


class McpSessionNotInitializedError(MCPHostError):
    """MCP 会话未初始化时抛出的异常。"""

    def __init__(self, mcp_server: str) -> None:
        """初始化错误。"""
        super().__init__(f"MCP session not initialized for {mcp_server}")


class McpSessionClosedOrFailedError(MCPHostError):
    """MCP 会话关闭或失败时抛出的异常。"""

    def __init__(self, mcp_server: str, state: str) -> None:
        """初始化错误。"""
        super().__init__(f"MCP session {state} for {mcp_server}")


class LogBufferNotFoundError(MCPHostError):
    """日志缓冲区未找到时抛出的异常。"""

    def __init__(self, name: str) -> None:
        """初始化错误。"""
        super().__init__(f"Log buffer {name} not found")
