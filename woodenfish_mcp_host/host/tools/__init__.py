"""MCP 服务器的模型相关定义和工具管理器。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager
from itertools import chain
from typing import TYPE_CHECKING, Self

from mcp import types

from woodenfish_mcp_host.host.conf import (
    LogConfig,
    ServerConfig,
)
from woodenfish_mcp_host.host.helpers.context import ContextProtocol
from woodenfish_mcp_host.host.tools.log import (
    LogManager,
)
from woodenfish_mcp_host.host.tools.mcp_server import McpServer, McpServerInfo, McpTool

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Iterable, Mapping

    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream


type ReadStreamType = MemoryObjectReceiveStream[types.JSONRPCMessage | Exception]
type WriteStreamType = MemoryObjectSendStream[types.JSONRPCMessage]
type StreamContextType = AbstractAsyncContextManager[
    tuple[ReadStreamType, WriteStreamType]
]


logger = logging.getLogger(__name__)


class ToolManager(ContextProtocol):
    """MCP 服务器的管理器。

    负责初始化、启动、停止和重载 MCP 服务器及其提供的工具。

    示例：
        ```python
        # 使用 MCP 服务器配置初始化 ToolManager
        config = HostConfig(...)
        async with ToolManager(config.mcp_servers) as tool_manager:
            # 获取 langchain 格式的工具列表
            tools = tool_manager.langchain_tools()
            # 使用工具进行操作...
        ```
    """

    def __init__(
        self,
        configs: dict[str, ServerConfig],
        log_config: LogConfig = LogConfig(),
    ) -> None:
        """初始化 ToolManager。

        参数：
            configs: MCP 服务器配置字典，键为服务器名称，值为 ServerConfig。
            log_config: 日志配置。
        """
        self._configs = configs
        self._log_config = log_config
        self._log_manager = LogManager(
            log_dir=log_config.log_dir, rotation_files=log_config.rotation_files
        )
        self._mcp_servers = dict[str, McpServer]()
        self._mcp_servers_task = dict[str, tuple[asyncio.Task, asyncio.Event]]()
        self._lock = asyncio.Lock()
        self._initialized_event = asyncio.Event()

        self._mcp_servers = {
            name: McpServer(
                name=name,
                config=config,
                log_buffer_length=log_config.buffer_length,
            )
            for name, config in self._configs.items()
            if config.enabled
        }

    def langchain_tools(
        self,
        tool_filter: Callable[[McpServer], bool] = lambda _: True,
    ) -> list[McpTool]:
        """获取 MCP 服务器提供的 langchain 格式的工具列表。

        参数：
            tool_filter: 用于过滤 MCP 服务器的函数，返回 True 表示包含该服务器的工具。

        返回：
            合并后的所有符合条件的 MCP 服务器提供的工具列表。
        """
        return list(
            chain.from_iterable(
                [i.mcp_tools for i in self._mcp_servers.values() if tool_filter(i)],
            ),
        )

    async def _launch_tools(self, servers: Mapping[str, McpServer]) -> None:
        async def tool_process(
            server: McpServer, exit_signal: asyncio.Event, ready: asyncio.Event
        ) -> None:
            async with self._log_manager.register_buffer(server.log_buffer), server:
                ready.set()
                await exit_signal.wait()
            logger.debug("Tool process %s exited", server.name)

        async def _launch_task(name: str, server: McpServer) -> None:
            event = asyncio.Event()
            ready = asyncio.Event()
            task = asyncio.create_task(tool_process(server, event, ready))
            await ready.wait()
            self._mcp_servers_task[name] = (task, event)

        async with self._lock, asyncio.TaskGroup() as tg:
            for name, server in servers.items():
                tg.create_task(_launch_task(name, server))

        self._initialized_event.set()

    async def _shutdown_tools(self, servers: Iterable[str]) -> None:
        async def _shutdown_task(name: str) -> None:
            task, event = self._mcp_servers_task.pop(name, (None, None))
            if not (task and event):
                logger.warning(
                    "task or event not found for %s. %s %s", name, task, event
                )
                return
            event.set()
            logger.debug("ToolManager shutting down %s", name)
            await task
            del self._mcp_servers[name]

        async with self._lock, asyncio.TaskGroup() as tg:
            for name in servers:
                tg.create_task(_shutdown_task(name))
        logger.debug("ToolManager shutdown complete")

    async def reload(
        self,
        new_configs: dict[str, ServerConfig], force: bool = False
    ) -> None:
        """使用新的配置重载 MCP 服务器。

        Args:
            new_configs: 新的 MCP 服务器配置字典。
            force: 如果为 True，即使配置未更改也强制重载所有 MCP 服务器。
        """
        logger.debug("Reloading MCP servers, force: %s", force)

        if not force:
            to_shutdown = set(self._configs.keys()) - set(new_configs.keys())
            to_launch = set(new_configs.keys()) - set(self._configs.keys())

            # check if the config has changed
            for key in set(self._configs) - to_shutdown:
                if self._configs[key] != new_configs[key]:
                    to_shutdown.add(key)
                    to_launch.add(key)
        else:
            to_shutdown = set(self._configs.keys())
            to_launch = set(new_configs.keys())

        self._configs = new_configs

        await self._shutdown_tools(to_shutdown)

        launch_servers = {}
        for l_key in to_launch:
            new_server = McpServer(
                name=l_key,
                config=new_configs[l_key],
                log_buffer_length=self._log_config.buffer_length,
            )
            launch_servers[l_key] = new_server
            self._mcp_servers[l_key] = new_server
        await self._launch_tools(launch_servers)

    async def _run_in_context(self) -> AsyncGenerator[Self, None]:
        """作为异步上下文管理器运行 ToolManager，负责 MCP 服务器的启动和关闭。"""
        # 我们可以操作堆栈来添加或移除工具
        launch_tools_task = asyncio.create_task(
            self._launch_tools(self._mcp_servers),
            name="init-launch-tools",
        )
        try:
            yield self
        finally:
            await self._shutdown_tools(list(self._mcp_servers.keys()))
            launch_tools_task.cancel()
            await launch_tools_task

    @property
    def mcp_server_info(self) -> dict[str, McpServerInfo]:
        """获取所有 MCP 服务器的当前信息（能力、工具、状态等）。

        返回：
            一个字典，键为服务器名称，值为对应的 McpServerInfo。
            如果 MCP 服务器尚未初始化成功，对应的值可能不完整或为 None（取决于 McpServerInfo 的定义）。
        """
        return {name: i.server_info for name, i in self._mcp_servers.items()}

    @property
    def initialized_event(self) -> asyncio.Event:
        """获取 ToolManager 初始化完成的事件。

        此事件仅在 ToolManager 初次启动并成功初始化所有 MCP 服务器时触发，重载时不会触发。
        """
        return self._initialized_event

    @property
    def log_manager(self) -> LogManager:
        """获取日志管理器实例。"""
        return self._log_manager
