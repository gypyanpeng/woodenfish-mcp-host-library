"""MCP 服务器的日志管理模块。

每个 MCP 服务器都有一个 `LogBuffer` 用于收集重要日志，
并注册到一个 `LogManager`，后者负责将日志写入文件，并提供 `listen_log` 方法供用户监听日志更新。

                      ┌────────────────────┐
                      │    API 或其他       │
                      └────────▲───────────┘
                               │
                           listen_log (监听日志)
                        ┌──────┼─────┐           ┌─────────────┐
                        │ LogManager ├───────────► 写到文件    │
                        └──────▲─────┘           └─────────────┘
                               │
                               │
                               │  register_buffer (注册缓冲区)
                               └──────────────────┐
                                                  │
┌─────────────────────────────────────────────────┼─────┐
│                          McpServer              │     │
│                                                 │     │
│                                                 │     │
│                                                 │     │
│┌─────────────────┐      ┌────────┐              │     │
││MCP 服务器 stdio ├──────►LogProxy (日志代理)├─────┐        │     │
│└─────────────────┘      └────────┘     │        │     │
│                                        │        │     │
│┌────────────────────────┐              │  ┌─────┼───┐ │
││MCP 客户端会话错误 ├──────────────┼──►LogBuffer (日志缓冲区)│ │
│└────────────────────────┘              │  └─────────┘ │
│                                        │              │
│┌────────────────────────┐              │              │
││MCP 客户端状态变更 ├──────────────┘              │
│└────────────────────────┘                             │
│                                                       │
└───────────────────────────────────────────────────────┘

# 使用 https://asciiflow.com/ 绘制
"""

import asyncio
import sys
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from enum import StrEnum
from logging import INFO, getLogger
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from traceback import format_exception
from typing import TextIO

from pydantic import BaseModel, Field

from woodenfish_mcp_host.host.errors import LogBufferNotFoundError
from woodenfish_mcp_host.host.tools.model_types import ClientState

logger = getLogger(__name__)


class LogEvent(StrEnum):
    """日志事件类型。"""

    STATUS_CHANGE = "status_change"  # 状态变更
    STDERR = "stderr"  # 标准错误输出
    STDOUT = "stdout"  # 标准输出
    SESSION_ERROR = "session_error" # 会话错误

    # API 相关的额外事件
    STREAMING_ERROR = "streaming_error" # 流式传输错误


class LogMsg(BaseModel):
    """日志结构。"""

    event: LogEvent # 事件类型
    body: str # 日志内容
    mcp_server_name: str # MCP 服务器名称
    client_state: ClientState | None = None # 客户端状态
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC)) # 时间戳


class LogBuffer:
    """具有大小限制的日志缓冲区，支持添加监听器以观察新日志。

    向缓冲区添加日志：
        使用 `push_logs` 或其他特定方法
        （例如 `push_session_error`、`push_state_change`）。

    观察日志更新：
        使用 `add_listener` 上下文管理器向缓冲区添加 `listener`。
        Listener 是一个异步函数，每当新日志添加到缓冲区时都会被调用。
        当首次添加监听器时，它会（逐个）处理缓冲区中的所有日志。

    示例：
        ```python
        # 创建一个日志缓冲区
        buffer = LogBuffer(size=1000, name="mcp_server")

        # 推送一条日志到缓冲区
        msg = LogMsg(event=LogEvent.STDERR, body="hello", mcp_server_name="mcp_server")
        await buffer.push_log(msg)


        async def listener(log: LogMsg) -> None:
            print(log)


        # listener 是一个上下文管理器，
        # 用户可以决定它监听缓冲区多久。
        async with buffer.add_listener(listener):
            await asyncio.sleep(10)
        ```
    """

    def __init__(self, size: int = 1000, name: str = "") -> None:
        """初始化日志缓冲区。"""
        self._size = size
        self._logs: list[LogMsg] = []
        self._listeners: list[Callable[[LogMsg], Coroutine[None, None, None]]] = []
        self._name = name
        self._client_state: ClientState = ClientState.INIT

    @property
    def name(self) -> str:
        """获取缓冲区名称。"""
        return self._name

    def _listener_wrapper(
        self,
        listener: Callable[[LogMsg], Coroutine[None, None, None]],
    ) -> Callable[[LogMsg], Coroutine[None, None, None]]:
        """包装监听器以处理异常。"""

        async def _wrapper(msg: LogMsg) -> None:
            try:
                await listener(msg)
            except Exception:
                logger.exception("listener exception")

        return _wrapper

    async def push_session_error(
        self,
        inpt: BaseExceptionGroup | BaseException,
    ) -> None:
        """将会话错误推送到日志缓冲区。"""
        msg = LogMsg(
            event=LogEvent.SESSION_ERROR,
            body="".join(format_exception(inpt)),
            mcp_server_name=self.name,
            client_state=self._client_state,
        )
        await self.push_log(msg)

    async def push_state_change(
        self,
        inpt: str,
        state: ClientState,
    ) -> None:
        """将客户端状态变更推送到日志缓冲区。"""
        self._client_state = state
        msg = LogMsg(
            event=LogEvent.STATUS_CHANGE,
            body=inpt,
            mcp_server_name=self.name,
            client_state=self._client_state,
        )
        await self.push_log(msg)

    async def push_stdout(
        self,
        inpt: str,
    ) -> None:
        """将标准输出日志推送到日志缓冲区。"""
        msg = LogMsg(
            event=LogEvent.STDOUT,
            body=inpt,
            mcp_server_name=self.name,
            client_state=self._client_state,
        )
        await self.push_log(msg)

    async def push_stderr(
        self,
        inpt: str,
    ) -> None:
        """将标准错误日志推送到日志缓冲区。"""
        msg = LogMsg(
            event=LogEvent.STDERR,
            body=inpt,
            mcp_server_name=self.name,
            client_state=self._client_state,
        )
        await self.push_log(msg)

    async def push_log(self, log: LogMsg) -> None:
        """向缓冲区添加一条日志，所有监听函数都将被调用。"""
        self._logs.append(log)
        if len(self._logs) > self._size:
            self._logs.pop(0)

        async with asyncio.TaskGroup() as group:
            for listener in self._listeners:
                group.create_task(listener(log))

    def get_logs(self) -> list[LogMsg]:
        """检索所有日志。"""
        return self._logs

    @asynccontextmanager
    async def add_listener(
        self, listener: Callable[[LogMsg], Coroutine[None, None, None]]
    ) -> AsyncIterator[None]:
        """向缓冲区添加一个监听器。

        读取缓冲区中的所有日志并监听新日志。
        该监听器是一个上下文管理器，
        用户可以决定它监听缓冲区多久。

        示例：
            ```python
            async def listener(log: LogMsg) -> None:
                print(log)


            async with buffer.add_listener(listener):
                await asyncio.sleep(10)
            ```
        """
        for i in self._logs:
            await listener(i)
        _listener = self._listener_wrapper(listener)
        self._listeners.append(_listener)
        try:
            yield
        except Exception:
            logger.exception("add listener error")
        finally:
            self._listeners.remove(_listener)


class LogProxy:
    """代理标准错误输出日志。"""

    def __init__(
        self,
        callback: Callable[[str], Coroutine[None, None, None]],
        mcp_server_name: str,
        stdio: TextIO = sys.stderr,
    ) -> None:
        """初始化代理。"""
        self._stdio = stdio
        self._callback = callback
        self._mcp_server_name = mcp_server_name

    async def write(self, s: str) -> None:
        """写入日志。"""
        await self._callback(s)
        self._stdio.write(s)

    async def flush(self) -> None:
        """刷新日志。"""
        self._stdio.flush()


class _LogFile:
    """按天轮转的日志文件。"""

    def __init__(self, name: str, log_dir: Path, rotation_files: int = 5) -> None:
        self._name = f"{name}.log"
        self._path = log_dir / self._name
        self._logger = getLogger(self._name)
        self._logger.setLevel(INFO)
        self._logger.propagate = False
        handler = TimedRotatingFileHandler(
            self._path,
            when="D",
            interval=1,
            backupCount=rotation_files,
            encoding="utf-8",
        )
        self._logger.addHandler(handler)

    async def __call__(self, log: LogMsg) -> None:
        self._logger.info(log.model_dump_json())


class LogManager:
    """MCP 服务器的日志管理器。

    注册到日志管理器的 `LogBuffers` 的日志将被写入文件。

    用户可以通过调用 `listen_log` 来监听特定 MCP 服务器的日志更新。
    该方法类似于 `LogBuffer` 的 `add_listener` 方法，是一个上下文管理器，
    在用户退出上下文之前会一直监听日志更新。

    示例：
        ```python
        log_dir = Path("/var/log/woodenfish_mcp_host")
        dummy_log = LogMsg(
            event=LogEvent.STDERR,
            body="hello",
            mcp_server_name="mcp_server",
        )

        # 创建一个日志管理器
        log_manager = LogManager(log_dir=log_dir)

        # 创建一个日志缓冲区
        buffer = LogBuffer(name="mcp_server")


        async def listener(log: LogMsg) -> None:
            print(log)


        # 注册缓冲区并监听日志更新
        async with (
            log_manager.register_buffer(buffer),
            log_manager.listen_log(buffer.name, listener),
        ):
            await buffer.push_log(dummy_log)
            await buffer.push_log(dummy_log)

            await asyncio.sleep(10)
        ```
    """

    def __init__(self, log_dir: Path, rotation_files: int = 5) -> None:
        """初始化日志管理器。

        参数：
            log_dir: 存储日志的目录。
            rotation_files: 每个 mcp 服务器的轮转文件数。
        """
        self._log_dir = log_dir / "mcp_logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._buffers: dict[str, LogBuffer] = {}
        self._rotation_files = rotation_files

    @asynccontextmanager
    async def register_buffer(self, buffer: LogBuffer) -> AsyncGenerator[None, None]:
        """将缓冲区注册到日志管理器。

        管理器会将日志写入文件。
        """
        self._buffers[buffer.name] = buffer
        log_file = _LogFile(buffer.name, self._log_dir, self._rotation_files)
        async with buffer.add_listener(log_file):
            try:
                yield
            except Exception:
                logger.exception("register buffer error")
            finally:
                if buffer.name in self._buffers:
                    self._buffers.pop(buffer.name)
                else:
                    logger.warning(f"LogBuffer '{buffer.name}' not found during unregistration, possibly already removed.")

    @asynccontextmanager
    async def listen_log(
        self,
        name: str,
        listener: Callable[[LogMsg], Coroutine[None, None, None]],
    ) -> AsyncGenerator[None, None]:
        """监听来自特定 MCP 服务器的日志更新。

        该监听器是一个上下文管理器，
        用户可以决定它监听缓冲区多久。

        只有注册到日志管理器的缓冲区才能被监听。
        如果缓冲区未注册，将引发 `LogBufferNotFoundError`。

        示例：
            ```python
            async def listener(log: LogMsg) -> None:
                print(log)


            async with log_manager.listen_log(buffer.name, listener):
                await asyncio.sleep(10)
            ```
        """
        buffer = self._buffers.get(name)
        if buffer is None:
            raise LogBufferNotFoundError(name)
        async with buffer.add_listener(listener):
            yield
