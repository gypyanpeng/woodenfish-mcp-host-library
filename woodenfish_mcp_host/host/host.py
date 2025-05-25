import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import AsyncExitStack
from copy import deepcopy
from typing import TYPE_CHECKING, Self

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langgraph.graph.message import MessagesState
from langgraph.prebuilt.tool_node import ToolNode

from woodenfish_mcp_host.host.agents import AgentFactory, get_chat_agent_factory
from woodenfish_mcp_host.host.chat import Chat
from woodenfish_mcp_host.host.conf import HostConfig
from woodenfish_mcp_host.host.errors import ThreadNotFoundError
from woodenfish_mcp_host.host.helpers.checkpointer import get_checkpointer
from woodenfish_mcp_host.host.helpers.context import ContextProtocol
from woodenfish_mcp_host.host.tools import McpServerInfo, ToolManager
from woodenfish_mcp_host.host.tools.log import LogManager
from woodenfish_mcp_host.models import load_model

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


logger = logging.getLogger(__name__)


class woodenfishMcpHost(ContextProtocol):
    """The Model Context Protocol (MCP) Host.

    该 woodenfishMcpHost 类提供了一个异步上下文管理接口，用于管理通过模型上下文协议（MCP）与语言模型进行交互。它处理模型实例的初始化和清理、管理服务器连接，并为代理聊天提供统一的接口。

    MCP 使得工具和模型能够以标准化的方式进行通信，允许在不同的模型实现之间保持一致的交互模式。

    示例：
        ```python
        # 使用配置初始化主机
        config = HostConfig(...)
        chat_id = ""
        async with woodenfishMcpHost(config) as host:
            # 发送消息并获取响应
            async with host.chat() as chat:
                while query := input("输入消息： "):
                    if query == "exit":
                        nonlocal chat_id # 修改此处以匹配变量名
                        # 保存 thread_id 以便后续恢复
                        chat_id = chat.chat_id
                        break
                    async for response in await chat.query(query):
                        print(response)
        ...
        # 恢复对话
        async with woodenfishMcpHost(config) as host:
            # 传入 chat_id 以恢复对话
            async with host.chat(chat_id=chat_id) as chat:
                ...
        ```

    该主机必须用作异步上下文管理器，以确保资源管理，包括模型初始化和清理。
    """

    def __init__(
        self,
        config: HostConfig,
    ) -> None:
        """初始化主机。

        参数：
            config: 主机配置。
        """
        self._config = config
        self._model: BaseChatModel | None = None
        self._checkpointer: BaseCheckpointSaver[str] | None = None
        self._tool_manager: ToolManager = ToolManager(
            configs=self._config.mcp_servers,
            log_config=self.config.log_config,
        )
        self._exit_stack: AsyncExitStack | None = None

    async def _run_in_context(self) -> AsyncGenerator[Self, None]:
        async with AsyncExitStack() as stack:
            self._exit_stack = stack
            await self._init_models()
            if self._config.checkpointer:
                checkpointer = get_checkpointer(str(self._config.checkpointer.uri))
                self._checkpointer = await stack.enter_async_context(checkpointer)
                await self._checkpointer.setup()
            await stack.enter_async_context(self._tool_manager)
            try:
                yield self
            except Exception as e:
                raise e

    async def _init_models(self) -> None:
        """初始化 LLM 模型。"""
        if not self._config.llm:
            return

        provider_to_load = self._config.llm.model_provider
        # Map custom provider names to what Langchain's load_model expects
        if provider_to_load == "openai_compatible":
            logger.info("Mapping 'openai_compatible' to 'openai' for Langchain's load_model.")
            provider_to_load = "openai"
        elif provider_to_load == "zhipu_glm4": # Added mapping for zhipu_glm4
            logger.info("Mapping 'zhipu_glm4' to 'openai' for Langchain's load_model as it's OpenAI compatible.")
            provider_to_load = "openai"
        # Add other custom mappings here if needed in the future, e.g., "moonshot_v1": "openai"

        model = load_model(
            provider_to_load,  # Use the potentially mapped provider
            self._config.llm.model,
            **self._config.llm.to_load_model_kwargs(),
        )
        if not isinstance(model, BaseChatModel):
            raise RuntimeError("Model is not a BaseChatModel")
        self._model = model

    def chat[T: MessagesState](  # noqa: PLR0913 是否有更好的方法？
        self,
        *,
        chat_id: str | None = None,
        user_id: str = "default",
        tools: Sequence[BaseTool] | None = None,
        get_agent_factory_method: Callable[
            [BaseChatModel, Sequence[BaseTool] | ToolNode, bool],
            AgentFactory[T],
        ] = get_chat_agent_factory,
        system_prompt: str | Callable[[T], list[BaseMessage]] | None = None,
        disable_default_system_prompt: bool = False,
        tools_in_prompt: bool | None = None,
        volatile: bool = False,
    ) -> Chat[T]:
        """开始或恢复对话。

        参数：
            chat_id: 对话使用的 ID。
            user_id: 对话使用的用户 ID。
            tools: 对话使用的工具。
            system_prompt: 对话使用的自定义系统提示词。
            get_agent_factory_method: 获取代理工厂的方法。
            volatile: 如果为 True，则对话不会被保存。
            disable_default_system_prompt: 禁用默认系统提示词。
            tools_in_prompt: 如果为 True，工具将被传递到提示词中。

        如果未提供对话 ID，将创建一个新的对话。
        自定义代理工厂以使用不同的模型或工具。
        如果未提供工具，主机将使用在主机中初始化的工具。
        """
        if self._model is None:
            raise RuntimeError("Model not initialized")
        
        logger.info(f"woodenfishMcpHost.chat called with chat_id: {chat_id}, user_id: {user_id}, volatile: {volatile}")
        logger.info(f"woodenfishMcpHost internal checkpointer state: {self._checkpointer}")

        if tools is None:
            tools = self._tool_manager.langchain_tools()
        if tools_in_prompt is None:
            tools_in_prompt = self._config.llm.tools_in_prompt
        agent_factory = get_agent_factory_method(
            self._model,
            tools,
            tools_in_prompt,
        )
        return Chat(
            model=self._model,
            agent_factory=agent_factory,
            system_prompt=system_prompt,
            disable_default_system_prompt=disable_default_system_prompt,
            chat_id=chat_id,
            user_id=user_id,
            checkpointer=None if volatile else self._checkpointer,
        )

    async def reload(
        self,
        new_config: HostConfig,
        reloader: Callable[[], Awaitable[None]] | None = None,
        force_mcp: bool = False,
    ) -> None:
        """使用新配置重新加载主机。

        参数：
            new_config: 新的主机配置。
            reloader: 重载时调用的函数。
            force_mcp: 如果为 True，即使 MCP 服务器未变也强制重载所有 MCP 服务器。

        reloader 函数会在主机准备好重载时被调用。这意味着所有正在进行的对话已完成，且没有新的请求在处理。
        reloader 应负责按需停止和重启相关服务。
        重载后可通过相同的 chat_id 恢复对话。
        """
        # 注意：有正在进行的请求时不要重启 MCP 服务器。
        if self._exit_stack is None:
            raise RuntimeError("Host not initialized")

        # Update config
        old_config = self._config
        self._config = new_config

        try:
            # Reload model if needed
            if old_config.llm != new_config.llm:
                self._model = None
                await self._init_models()

            await self._tool_manager.reload(
                new_configs=new_config.mcp_servers, force=force_mcp
            )

            # Reload checkpointer if needed
            if old_config.checkpointer != new_config.checkpointer:
                if self._checkpointer is not None:
                    await self._exit_stack.aclose()
                    self._checkpointer = None

                if new_config.checkpointer:
                    checkpointer = get_checkpointer(str(new_config.checkpointer.uri))
                    self._checkpointer = await self._exit_stack.enter_async_context(
                        checkpointer
                    )
                    await self._checkpointer.setup()

            # Call the reloader function to handle service restart
            if reloader:
                await reloader()

        except Exception as e:
            # Restore old config if reload fails
            self._config = old_config
            logging.error("Failed to reload host: %s", e)
            raise

    @property
    def config(self) -> HostConfig:
        """当前主机配置的副本。

        注意：不要修改返回的配置。如需更改配置请使用 `reload` 方法。
        """
        return deepcopy(self._config)

    @property
    def tools_initialized_event(self) -> asyncio.Event:
        """获取工具初始化事件。

        仅在初次启动时有用，重载时无效。
        """
        return self._tool_manager.initialized_event

    @property
    def tools(self) -> Sequence[BaseTool]:
        """主机的 ACTIVE 工具。

        此属性为只读。调用 `reload` 以更改工具。
        """
        return self._tool_manager.langchain_tools()

    @property
    def mcp_server_info(self) -> dict[str, McpServerInfo]:
        """获取活动 MCP 服务器的信息。

        返回：
            一个字典，将服务器名称映射到其功能和工具。
            对于任何尚未完成初始化的服务器，该值为 None。
        """
        return self._tool_manager.mcp_server_info

    @property
    def model(self) -> BaseChatModel:
        """主机的模型。"""
        if self._model is None:
            raise RuntimeError("Model not initialized")
        return self._model

    async def get_messages(self, thread_id: str, user_id: str) -> list[BaseMessage]:
        """获取特定线程的消息。

        参数：
            thread_id: 要检索消息的线程 ID。
            user_id: 要检索消息的用户 ID。

        返回：
            消息列表。

        异常：
            ThreadNotFoundError: 如果未找到线程。
        """
        if self._checkpointer is None:
            return []

        if ckp := await self._checkpointer.aget(
            {
                "configurable": {
                    "thread_id": thread_id,
                    "user_id": user_id,
                }
            }
        ):
            return ckp["channel_values"].get("messages", [])
        raise ThreadNotFoundError(thread_id)

    async def delete_thread(self, thread_id: str) -> None:
        """删除线程。

        参数：
            thread_id: 要删除的线程 ID。
        """
        if self._checkpointer:
            await self._checkpointer.adelete_thread(thread_id)

    @property
    def log_manager(self) -> LogManager:
        """获取日志管理器。"""
        return self._tool_manager.log_manager
