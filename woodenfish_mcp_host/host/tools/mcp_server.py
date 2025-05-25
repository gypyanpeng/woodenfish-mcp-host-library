"""MCP 服务器的数据模型。"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from json import JSONDecodeError
from json import loads as json_loads
from logging import getLogger
from traceback import format_exception
from typing import TYPE_CHECKING, Any, Literal, Self

import anyio
import httpx
from langchain_core.tools import BaseTool, ToolException
from mcp import ClientSession, McpError, StdioServerParameters, types
from mcp.client.sse import sse_client
from mcp.client.websocket import websocket_client
from pydantic import BaseModel, ConfigDict, SecretStr
from pydantic_core import to_json

from woodenfish_mcp_host.host.errors import (
    InvalidMcpServerError,
    McpSessionClosedOrFailedError,
    McpSessionGroupError,
    McpSessionNotInitializedError,
)
from woodenfish_mcp_host.host.helpers.context import ContextProtocol
from woodenfish_mcp_host.host.tools.log import (
    LogBuffer,
    LogProxy,
)
from woodenfish_mcp_host.host.tools.model_types import ClientState
from woodenfish_mcp_host.host.tools.stdio_server import stdio_client

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable

    from woodenfish_mcp_host.host.conf import (
        ServerConfig,
    )
    from woodenfish_mcp_host.host.tools import (
        ReadStreamType,
        StreamContextType,
        WriteStreamType,
    )

logger = getLogger(__name__)


class McpServerInfo(BaseModel):
    """MCP 服务器的能力和工具列表。"""

    name: str
    """MCP 服务器名称。"""
    tools: list[types.Tool]
    """MCP 服务器提供的工具列表。"""
    initialize_result: types.InitializeResult | None
    """initialize 方法的返回结果。

    initialize_result.capabilities: 服务器能力。
    initialize_result.instructions: 服务器指令。
    """

    error: BaseException | BaseExceptionGroup | None
    """MCP 服务器发生的错误。"""

    client_status: ClientState
    """客户端状态：RUNNING、CLOSED、RESTARTING 或 INIT。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def error_str(self) -> str | None:
        """打印完整的错误信息。"""
        if self.error is None:
            return None
        return "\n".join(format_exception(self.error))


class McpServer(ContextProtocol):
    """表示一个 MCP 服务器及其相关工具集。

    后台会运行一个任务持续监控客户端连接状态。如果客户端断开，任务会尝试重连。
    当 McpServer 状态变为 CLOSED 时，监控任务停止。
    """

    RETRY_LIMIT: int = 3
    KEEP_ALIVE_INTERVAL: float = 60
    RESTART_INTERVAL: float = 3

    def __init__(
        self,
        name: str,
        config: ServerConfig,
        log_buffer_length: int = 1000,
    ) -> None:
        """初始化 McpToolKit。"""
        self.name = name
        self.config = config
        self._log_buffer = LogBuffer(name=name, size=log_buffer_length)
        self._stderr_log_proxy = LogProxy(
            callback=self._log_buffer.push_stderr,
            mcp_server_name=self.name,
            stdio=sys.stderr,
        )
        self._stdout_log_proxy = LogProxy(
            callback=self._log_buffer.push_stdout,
            mcp_server_name=self.name,
            stdio=sys.stdout,
        )
        self._cond = asyncio.Condition()
        """用于同步访问共享变量的条件变量。"""
        self._client_status: ClientState = ClientState.INIT
        self._task: asyncio.Task[Awaitable[None]] | None = None
        self._session: ClientSession | None = None
        self._tool_results: types.ListToolsResult | None = None
        self._initialize_result: types.InitializeResult | None = None
        self._session_count: int = 0
        self._exception: BaseException | BaseExceptionGroup | None = None
        self._mcp_tools: list[McpTool] = []
        self._retries: int = 0
        self._last_active: float = 0

    async def _keep_alive(self, session: ClientSession) -> None:
        """定期向服务器发送 ping 消息以保持连接活跃。"""
        if not self.config.keep_alive:
            return

        while True:
            if self._client_status != ClientState.RUNNING:
                logger.debug(
                    "Keep-alive stopped for %s: client status is %s",
                    self.name,
                    self._client_status,
                )
                return
            try:
                await asyncio.sleep(self.config.keep_alive)
                await session.send_ping()
                self._last_active = time.time()
            except asyncio.CancelledError:
                logger.debug("Keep-alive cancelled for %s", self.name)
                return
            except BaseException as e:
                logger.error("Keep-alive error for %s: %s", self.name, e)
                async with self._cond:
                    await self.__change_state(
                        ClientState.RESTARTING, [ClientState.RUNNING], e
                    )
                raise
            await asyncio.sleep(self.KEEP_ALIVE_INTERVAL)

    async def _client_watcher(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """监控客户端连接状态并根据需要重启客户端。

        只有此监控器有权将客户端状态设置为 RUNNING/FAILED。
        """
        keep_alive_task: asyncio.Task[None] | None = None
        should_break = False
        while True:
            try:
                logger.debug("Attempting to initialize client %s", self.name)
                async with (
                    self._get_client() as streams,
                    ClientSession(*streams) as session,
                ):
                    async with asyncio.timeout(10):
                        # When using stdio, the initialize call may block indefinitely
                        self._initialize_result = await session.initialize()
                    tool_results = await session.list_tools()
                    self._last_active = time.time()
                    mcp_tools = [
                        McpTool(
                            toolkit_name=self.name,
                            name=tool.name,
                            description=tool.description or "",
                            mcp_server=self,
                            kwargs_arg="kwargs" in tool.inputSchema,
                            args_schema=tool.inputSchema,
                        )
                        for tool in tool_results.tools
                    ]
                    async with self._cond:
                        self._session = session
                        self._tool_results = tool_results
                        self._mcp_tools = mcp_tools
                        self._exception = None
                        self._retries = 0
                        await self.__change_state(ClientState.RUNNING, None, None)
                        logger.debug(
                            "Client %s initialized successfully with %d tools",
                            self.name,
                            len(mcp_tools),
                        )

                        # Start keep-alive task if configured
                        keep_alive_task = (
                            asyncio.create_task(self._keep_alive(session))
                            if self.config.keep_alive
                            else None
                        )

                        await self._cond.wait_for(
                            lambda: self._client_status
                            in [ClientState.CLOSED, ClientState.RESTARTING],
                        )
                        logger.debug(
                            "client watcher %s exited. status: %s",
                            self.name,
                            self._client_status,
                        )
            # cannot launch local sse server
            except* InvalidMcpServerError as e:
                self._exception = e
                should_break = True
            except* ProcessLookupError as eg:
                # 当 stdio 进程退出且 initialize 调用超时时抛出此异常
                err_msg = f"ProcessLookupError for {self.name}: {eg.exceptions}"
                logger.exception(err_msg)
                self._exception = McpSessionGroupError(
                    err_msg,
                    eg.exceptions,
                )
                should_break = True
            except* (
                FileNotFoundError,
                PermissionError,
                McpError,
                httpx.ConnectError,
                httpx.InvalidURL,
                httpx.TooManyRedirects,
            ) as eg:
                err_msg = (
                    f"Client initialization error for {self.name}: {eg.exceptions}"
                )
                logger.exception(err_msg)
                self._exception = McpSessionGroupError(err_msg, eg.exceptions)
                should_break = True
            except* httpx.HTTPStatusError as eg:
                err_msg = f"Client http error for {self.name}: {eg.exceptions}"
                logger.exception(err_msg)
                self._exception = McpSessionGroupError(err_msg, eg.exceptions)
                for e in eg.exceptions:
                    if (
                        isinstance(e, httpx.HTTPStatusError)
                        and e.response.status_code < 500  # noqa: PLR2004
                        and e.response.status_code != 429  # noqa: PLR2004
                    ):
                        should_break = True
                        break
            except* asyncio.CancelledError as e:
                should_break = True
                logger.debug("Client watcher cancelled for %s", self.name)
            except* BaseException as eg:
                err_msg = (
                    f"Client initialization error for {self.name}: {eg.exceptions}"
                )
                logger.exception(err_msg)
                self._exception = McpSessionGroupError(err_msg, eg.exceptions)

            if self._exception:
                await self._log_buffer.push_session_error(self._exception)

            self._retries += 1
            self._session = None
            if keep_alive_task:
                with suppress(Exception):
                    keep_alive_task.cancel()
                    await keep_alive_task
                    logger.debug("Keep-alive task awaited for %s", self.name)
            if self._client_status == ClientState.CLOSED:
                logger.info("Client %s closed, stopping watcher", self.name)
                return
            if self._retries >= self.RETRY_LIMIT or should_break:
                logger.warning(
                    "client for [%s] failed after %d retries %s",
                    self.name,
                    self._retries,
                    self._exception,
                )
                async with self._cond:
                    if self._client_status != ClientState.CLOSED:
                        await self.__change_state(ClientState.FAILED, None, False)
                return
            logger.debug(
                "Retrying client initialization for %s (attempt %d/%d)",
                self.name,
                self._retries,
                self.RETRY_LIMIT,
            )
            await asyncio.sleep(self.RESTART_INTERVAL)

    async def _run_in_context(self) -> AsyncGenerator[Self, None]:
        """获取 MCP 服务器的 langchain 工具。"""
        task = asyncio.create_task(self._client_watcher())
        async with self._cond:
            await self._cond.wait_for(
                lambda: self._client_status
                in [ClientState.RUNNING, ClientState.CLOSED, ClientState.FAILED]
            )
        try:
            yield self
        finally:
            logger.debug("mcp server shutting down %s", self.name)
            async with self._cond:
                self._session = None
                await self.__change_state(
                    ClientState.CLOSED, [ClientState.INIT, ClientState.RUNNING], False
                )
                logger.debug(
                    "%s: wait all sessions to be closed. now is %s",
                    self.name,
                    self._session_count,
                )
            async with self._cond, asyncio.timeout(30):
                try:
                    await self._cond.wait_for(
                        lambda: self._session_count == 0,
                    )
                except TimeoutError:
                    logger.warning(
                        "Timeout to wait %d sessions to be closed",
                        self._session_count,
                    )
            with suppress(Exception):
                async with asyncio.timeout(10):
                    task.cancel()
                    await task
                logger.debug("MCP server %s exited", self.name)

    def _get_client(
        self,
    ) -> StreamContextType:
        """根据配置创建一个新的客户端（stdio, sse 或 websocket）。

        此方法仅由客户端监控器调用。
        """
        if self.config.command:
            env = os.environ.copy()
            env.update(self.config.env)
            if self.config.transport in ("stdio", None):
                return stdio_client(
                    server=StdioServerParameters(
                        command=self.config.command,
                        args=self.config.args,
                        env=env,
                    ),
                    errlog=self._stderr_log_proxy,
                )
            if self.config.transport in ("sse", "websocket") and self.config.url:
                return local_mcp_net_server_client(
                    config=self.config,
                    command=self.config.command,
                    headers=self.config.headers,
                    args=self.config.args,
                    env=env,
                    stderrlog=self._stderr_log_proxy,
                    stdoutlog=self._stdout_log_proxy,
                )
            raise InvalidMcpServerError(
                self.name, "Only stdio is supported for command."
            )
        if self.config.url:
            if self.config.transport in ("sse", None):
                return sse_client(
                    url=self.config.url,
                    headers={
                        key: value.get_secret_value()
                        for key, value in self.config.headers.items()
                    },
                )
            if self.config.transport == "websocket":
                return websocket_client(
                    url=self.config.url,
                )
            raise InvalidMcpServerError(
                self.name, "Only sse and websocket are supported for url."
            )
        raise InvalidMcpServerError(self.name, "No url or command provided.")

    @property
    def mcp_tools(self) -> list[McpTool]:
        """获取工具列表。"""
        if self._client_status == ClientState.RUNNING:
            return self._mcp_tools
        return []

    async def __change_state(
        self,
        new_state: ClientState,
        orig_state: list[ClientState] | None,
        e: BaseException | None | Literal[False],
    ) -> None:
        """更改客户端状态并通知等待者。

        调用者必须持有 self._cond 锁。
        只有状态实际发生变化时才会通知条件变量。

        参数：
            new_state: 新的状态。
            orig_state: 原始状态。
              如果 orig_state 为 None 或 self._client_status == orig_state，则切换到 new_state。
            e: 发生的异常。
              如果 e 不是 False，则将 self._exception 设为 e。
        """
        if orig_state is None or self._client_status in orig_state:
            if e is not False:
                self._exception = e
            self._client_status = new_state
            log_msg = f"client status changed, {self.name} {new_state}, error: {e}"
            logger.debug(log_msg)
            await self._log_buffer.push_state_change(inpt=log_msg, state=new_state)
            self._cond.notify_all()

    async def _wait_for_session(self) -> ClientSession:
        """等待并返回一个可用的客户端会话。

        此方法仅由 session 上下文管理器调用，负责处理重试和等待连接建立。
        """
        for retried in range(self.RETRY_LIMIT):
            async with self._cond:
                if self._client_status in (ClientState.RESTARTING, ClientState.INIT):
                    logger.debug(
                        "Waiting for session initialization, %s status: %s",
                        self.name,
                        self._client_status,
                    )
                    await self._cond.wait_for(
                        lambda: self._client_status
                        in [
                            ClientState.CLOSED,
                            ClientState.RUNNING,
                            ClientState.FAILED,
                        ],
                    )
            if self._client_status in [ClientState.FAILED, ClientState.CLOSED]:
                logger.error(
                    "Session failed or closed for %s: %s",
                    self.name,
                    self._client_status,
                )
                raise McpSessionClosedOrFailedError(self.name, self._client_status.name)
            now = time.time()
            if (
                self._client_status == ClientState.RUNNING
                and self._session
                and (now - self._last_active > self.KEEP_ALIVE_INTERVAL)
            ):
                # check if the session is still active
                try:
                    logger.debug(
                        "Checking session health for %s (inactive for %.1f seconds)",
                        self.name,
                        now - self._last_active,
                    )
                    async with asyncio.timeout(10):
                        await self._session.send_ping()
                        self._last_active = time.time()
                except Exception as e:  # noqa: BLE001
                    logger.error(
                        "Keep-alive error for %s: %s",
                        self.name,
                        e,
                        extra={
                            "mcp_server": self.name,
                            "client_status": self._client_status,
                        },
                    )
                    async with self._cond:
                        await self.__change_state(
                            ClientState.RESTARTING, [ClientState.RUNNING], e
                        )
            if self._client_status == ClientState.RUNNING and self._session:
                return self._session
            if retried < self.RETRY_LIMIT - 1:
                logger.warning(
                    "Session not initialized, retrying, %s status: %s (attempt %d/%d)",
                    self.name,
                    self._client_status,
                    retried + 1,
                    self.RETRY_LIMIT,
                    extra={
                        "mcp_server": self.name,
                        "client_status": self._client_status,
                    },
                )
                await asyncio.sleep(self.RESTART_INTERVAL)
        logger.error(
            "Session not initialized after %d attempts, %s status: %s",
            self.RETRY_LIMIT,
            self.name,
            self._client_status,
            extra={"mcp_server": self.name, "client_status": self._client_status},
        )
        raise McpSessionNotInitializedError(self.name)

    def session(self) -> AbstractAsyncContextManager[ClientSession]:
        """获取一个 MCP 会话的异步上下文管理器。"""

        @asynccontextmanager
        async def session_ctx() -> AsyncGenerator[ClientSession, None]:
            """会话上下文管理器的实现。"""
            session = await self._wait_for_session()
            self._session_count += 1
            try:
                yield session
            # TODO: 并非所有异常都需要重启服务器
            # 应该针对不同异常类型做不同处理
            # 例如连接错误可能需要重启，应用级错误只需抛出即可
            except (ToolException, McpError) as e:
                logger.error("Tool exception for %s: %s", self.name, e)
                raise
            except (
                httpx.HTTPError,
                httpx.StreamError,
                httpx.TimeoutException,
                httpx.TooManyRedirects,
                anyio.BrokenResourceError,
                anyio.ClosedResourceError,
                anyio.EndOfStream,
                Exception,  # Before we know the exception type
            ) as e:
                async with self._cond:
                    await self.__change_state(
                        ClientState.RESTARTING, [ClientState.RUNNING], e
                    )
                logger.warning(
                    "mcp server %s failed, restarting, %s",
                    self.name,
                    e,
                    extra={
                        "mcp_server": self.name,
                        "client_status": self._client_status,
                    },
                )
                raise
            finally:
                async with self._cond:
                    self._session_count -= 1
                    self._cond.notify_all()

        return session_ctx()

    @property
    def log_buffer(self) -> LogBuffer:
        """获取 MCP 服务器的日志缓冲区。"""
        return self._log_buffer

    @property
    def server_info(self) -> McpServerInfo:
        """获取 MCP 服务器的当前信息（能力、工具、状态等）。"""
        return McpServerInfo(
            name=self.name,
            tools=[tool.mcp_tool_info for tool in self._mcp_tools],
            initialize_result=self._initialize_result,
            error=self._exception,
            client_status=self._client_status,
        )

    async def wait(self, states: list[ClientState]) -> bool:
        """等待客户端状态变为指定状态列表中的一个。"""
        async with self._cond:
            await self._cond.wait_for(
                lambda: self._client_status
                in [
                    *states,
                    ClientState.FAILED,
                    ClientState.CLOSED,
                ],
            )
            return self._client_status in states


class McpTool(BaseTool):
    """封装 MCP 服务器提供的工具，使其可被 langchain 调用。"""

    toolkit_name: str
    description: str = ""
    mcp_server: McpServer
    kwargs_arg: bool = False

    def _run(self, **kwargs: dict[str, Any]) -> str:
        """运行该工具。"""
        return asyncio.run(self._arun(**kwargs))

    async def _arun(self, **kwargs: dict[str, Any]) -> str:
        """异步运行该工具。"""
        if not self.kwargs_arg and len(kwargs) == 1 and "kwargs" in kwargs:
            if isinstance(kwargs["kwargs"], str):
                with suppress(JSONDecodeError):
                    kwargs = json_loads(kwargs["kwargs"])
            else:
                kwargs = kwargs["kwargs"]
        logger.debug(
            "Executing tool %s.%s with args: %s", self.toolkit_name, self.name, kwargs
        )
        async with self.mcp_server.session() as session:
            result = await session.call_tool(self.name, arguments=kwargs)
        content = to_json(result.content).decode()
        if result.isError:
            logger.error(
                "Tool execution failed for %s.%s: %s",
                self.toolkit_name,
                self.name,
                content,
            )
        logger.debug("Tool %s.%s executed successfully", self.toolkit_name, self.name)
        return content


@asynccontextmanager
async def local_mcp_net_server_client(  # noqa: C901, PLR0913, PLR0915
    config: ServerConfig,
    stderrlog: LogProxy,
    stdoutlog: LogProxy,
    command: str | None = None,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    max_connection_retries: int = 10,
    headers: dict[str, Any] | None = None,
) -> AsyncGenerator[tuple[ReadStreamType, WriteStreamType], None]:
    """创建本地 MCP 服务器客户端。

    参数：
        config: MCP 服务器的配置。
        command: 启动 MCP 服务器的命令。如果为 None，则使用 config.command。
        args: 启动 MCP 服务器的参数。如果为 None，则使用 config.args。
        env: 启动 MCP 服务器的环境变量。如果为 None，则使用 config.env。
        max_connection_retries: 最大连接重试次数。
        headers: 发送给 MCP 服务器的请求头。如果为 None，则使用 config.headers。
        stderrlog: 子进程 stderr 的日志代理。
        stdoutlog: 子进程 stdout 的日志代理。
    """
    command = command or config.command
    args = args or config.args
    env = env or config.env
    headers = (headers or config.headers).copy()
    assert config.url is not None, "url is required"

    def _sse_client(
        url: str,
    ) -> AbstractAsyncContextManager[tuple[ReadStreamType, WriteStreamType]]:
        """创建 SSE 客户端上下文管理器。"""
        for key in headers:
            value = headers[key]
            if isinstance(value, SecretStr):
                headers[key] = value.get_secret_value()
        return sse_client(url=url, headers=headers)

    get_client = _sse_client if config.transport == "sse" else websocket_client
    logger.debug("Starting local MCP server %s with command: %s", config.name, command)
    if not (
        subprocess := await asyncio.create_subprocess_exec(
            command,
            *args,
            env=env,
            stderr=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
    ):
        logger.error("Failed to start subprocess for %s", config.name)
        raise RuntimeError("failed to start subprocess")
    retried = 0

    # it tooks time to start the server, so we need to retry
    async def _read_stdout(
        stream: asyncio.StreamReader | None,
    ) -> None:
        """读取标准输出流并推送到日志缓冲区。"""
        if not stream:
            return

        async for line in stream:
            await stdoutlog.write(line.decode())
            await stdoutlog.flush()

    async def _read_stderr(
        stream: asyncio.StreamReader | None,
    ) -> None:
        """读取标准错误流并推送到日志缓冲区。"""
        if not stream:
            return

        async for line in stream:
            await stderrlog.write(line.decode())
            await stderrlog.flush()

    read_stderr_task = asyncio.create_task(
        _read_stderr(subprocess.stderr),
        name="read-stderr",
    )
    read_stdout_task = asyncio.create_task(
        _read_stdout(subprocess.stdout),
        name="read-stdout",
    )

    try:
        while retried < max_connection_retries:
            await asyncio.sleep(0.3 if retried == 0 else 1)
            logger.debug(
                "Attempting to connect to server %s (attempt %d/%d)",
                config.name,
                retried + 1,
                max_connection_retries,
            )
            with suppress(TimeoutError, httpx.HTTPError):
                async with (
                    get_client(url=config.url) as streams,
                    ClientSession(*streams) as session,
                ):
                    async with asyncio.timeout(10):
                        await session.initialize()
                        logger.info("Successfully connected to server %s", config.name)
                        break
            retried += 1
        else:
            raise InvalidMcpServerError(config.name)
        logger.info(
            "Connected to the server %s after %d attempts", config.name, retried
        )
        async with get_client(url=config.url) as streams:
            yield streams
    finally:
        with suppress(TimeoutError):
            logger.debug("Terminating subprocess for %s", config.name)
            read_stderr_task.cancel()
            read_stdout_task.cancel()
            subprocess.terminate()
            if sys.platform != "win32":
                subprocess.send_signal(signal.SIGINT)
            await asyncio.wait_for(subprocess.wait(), timeout=10)
            await read_stderr_task
            await read_stdout_task
            subprocess = None
        if subprocess:
            logger.info("Timeout to terminate mcp-server %s. Kill it.", config.name)
            with suppress(TimeoutError):
                read_stderr_task.cancel()
                read_stdout_task.cancel()
                subprocess.kill()
                await asyncio.wait_for(subprocess.wait(), timeout=10)
                await read_stderr_task
                await read_stdout_task
