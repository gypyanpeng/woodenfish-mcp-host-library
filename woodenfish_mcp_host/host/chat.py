import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from typing import Any, Self, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    RemoveMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import MessagesState
from langgraph.store.base import BaseStore
from langgraph.types import StreamMode

from woodenfish_mcp_host.host.agents import AgentFactory, V
from woodenfish_mcp_host.host.errors import (
    GraphNotCompiledError,
    MessageTypeError,
    ThreadNotFoundError,
    ThreadQueryError,
)
from woodenfish_mcp_host.host.helpers.context import ContextProtocol
from woodenfish_mcp_host.host.prompt import default_system_prompt

logger = logging.getLogger(__name__)


class Chat[STATE_TYPE: MessagesState](ContextProtocol):
    """与语言模型的对话会话。

    示例：
        ```python
        # 创建一个 Chat 实例
        chat = Chat(model=my_model, agent_factory=my_agent_factory)

        # 运行对话上下文
        async with chat.run():
            # 发送查询并流式获取响应
            async for response in chat.query("你好！"):
                print(response)
        ```
    """

    def __init__(  # noqa: PLR0913, too many arguments
        self,
        model: BaseChatModel,
        agent_factory: AgentFactory[STATE_TYPE],
        *,
        system_prompt: str | Callable[[STATE_TYPE], list[BaseMessage]] | None = None,
        chat_id: str | None = None,
        user_id: str = "default",
        store: BaseStore | None = None,
        checkpointer: BaseCheckpointSaver[V] | None = None,
        disable_default_system_prompt: bool = False,
    ) -> None:
        """初始化对话会话。

        参数：
            model: 用于对话的语言模型。
            agent_factory: 用于对话的代理工厂。
            system_prompt: 对话使用的系统提示词。
            chat_id: 对话的唯一ID（langgraph 线程ID）。
            user_id: 对话使用的用户ID。
            store: 对话使用的数据存储。
            checkpointer: 对话使用的 langgraph 检查点管理器。
            disable_default_system_prompt: 是否禁用默认系统提示词。

        agent_factory 只会被调用一次用于编译代理。
        """
        self._chat_id: str = chat_id if chat_id else uuid.uuid4().hex
        self._user_id: str = user_id
        self._store = store
        self._checkpointer = checkpointer
        self._model = model
        self._system_prompt = system_prompt
        self._agent: CompiledGraph | None = None
        self._agent_factory: AgentFactory[STATE_TYPE] = agent_factory
        self._abort_signal: asyncio.Event | None = None
        self._disable_default_system_prompt = disable_default_system_prompt

    @property
    def active_agent(self) -> CompiledGraph:
        """当前对话的激活代理。"""
        if self._agent is None:
            raise GraphNotCompiledError(self._chat_id)
        return self._agent

    @property
    def chat_id(self) -> str:
        """当前对话的唯一ID。"""
        return self._chat_id

    def abort(self) -> None:
        """中止正在进行的查询。"""
        if self._abort_signal is None:
            return
        self._abort_signal.set()

    async def _run_in_context(self) -> AsyncGenerator[Self, None]:
        if self._system_prompt is None and not self._disable_default_system_prompt:
            system_prompt = default_system_prompt()
        else:
            system_prompt = self._system_prompt

        if callable(system_prompt):
            prompt = system_prompt
        else:
            prompt = (
                self._agent_factory.create_prompt(system_prompt=system_prompt)
                if system_prompt
                else ""
            )

        # 可以在这里对 prompt 进行进一步处理
        self._agent = self._agent_factory.create_agent(
            prompt=prompt,
            checkpointer=self._checkpointer,
            store=self._store,
        )
        try:
            yield self
        finally:
            self._agent = None

    async def _get_updates_for_resend(
        self,
        resend: list[BaseMessage],
        update: list[BaseMessage],
    ) -> list[BaseMessage]:
        if not self._checkpointer:
            return update
        resend_map = {msg.id: msg for msg in resend}
        to_update = [i for i in update if i.id not in resend_map]
        if state := await self.active_agent.aget_state(
            RunnableConfig(
                configurable={
                    "thread_id": self._chat_id,
                    "user_id": self._user_id,
                },
            )
        ):
            drop_after = False
            for msg in cast(MessagesState, state.values)["messages"]:
                assert msg.id is not None  # 代理生成的所有消息都必须有ID
                if msg.id in resend_map:
                    drop_after = True
                elif drop_after:
                    to_update.append(RemoveMessage(msg.id))
            return to_update
        raise ThreadNotFoundError(self._chat_id)

    def query(
        self,
        query: str | HumanMessage | list[BaseMessage] | None,
        *,
        stream_mode: list[StreamMode] | StreamMode | None = "messages",
        modify: list[BaseMessage] | None = None,
        is_resend: bool = False,
    ) -> AsyncIterator[dict[str, Any] | Any]:
        """对话查询。

        参数：
            query: 向对话提问的内容，可以是字符串、HumanMessage 或消息列表。
                如果需要重发消息，请将要重发的消息列表传递到这里。
            stream_mode: 响应的流式模式，可以是单个模式或模式列表。
            modify: 需要修改的消息列表，用于无需重发时修改对话状态（例如确认工具调用参数）。
            is_resend: 如果为 True，表示 query 包含需要重发的消息。此时，query 及其后续消息会被移除，modify 中出现在 query 中的消息会被忽略。

        返回：
            异步生成器，生成响应内容。

        异常：
            MessageTypeError: 如果要修改的消息类型无效，则抛出此异常。
        """

        async def _stream_response() -> AsyncGenerator[dict[str, Any] | Any, None]:
            query_msgs = _convert_query_to_messages(query)
            if is_resend and query_msgs:
                if len(query_msgs) == 0 or not all(
                    isinstance(msg, BaseMessage) and msg.id for msg in query_msgs
                ):
                    raise MessageTypeError("Resending messages must has an ID")
                query_msgs += await self._get_updates_for_resend(
                    query_msgs, modify or []
                )
            elif modify:
                query_msgs = [*query_msgs, *modify]
            signal = asyncio.Event()
            self._abort_signal = signal
            if query_msgs:
                init_state = self._agent_factory.create_initial_state(query=query_msgs)
            else:
                init_state = None
            logger.debug("init_state: %s", query_msgs)
            config = self._agent_factory.create_config(
                user_id=self._user_id,
                thread_id=self._chat_id,
            )
            try:
                async for response in self.active_agent.astream(
                    input=init_state,
                    stream_mode=stream_mode,
                    config=config,
                ):
                    if signal.is_set():
                        break
                    yield response
            except Exception as e:
                raise ThreadQueryError(
                    query, state_values=await self.dump_values(), error=e
                ) from e

        return _stream_response()

    async def dump_values(self) -> dict[str, Any] | None:
        """导出当前对话状态的所有值。"""
        if self._checkpointer is None:
            return None
        try:
            if state := await self.active_agent.aget_state(
                RunnableConfig(
                    configurable={"thread_id": self._chat_id, "user_id": self._user_id},
                )
            ):
                return state.values
        except Exception:
            logger.exception("导出状态值失败")
            return None


def _convert_query_to_messages(
    query: str | HumanMessage | list[BaseMessage] | None,
) -> list[BaseMessage]:
    # 将查询内容转换为消息列表
    if isinstance(query, BaseMessage):
        return [query]
    if isinstance(query, str):
        return [HumanMessage(content=query)]
    if isinstance(query, list):
        return [
            i if isinstance(i, BaseMessage) else HumanMessage(content=i) for i in query
        ]
    return []


class MessageChunkHolder:
    """消息分片持有器。"""

    def __init__(self) -> None:
        """初始化消息分片持有器。"""
        self._merged: dict[str, BaseMessage] = {}

    def feed[T: BaseMessage | BaseMessageChunk](self, chunk: T) -> T | None:
        """输入一个分片，如果完成则返回合并后的消息。"""
        if isinstance(chunk, BaseMessageChunk) and chunk.id:
            m = cast(T, s + chunk) if (s := self._merged.get(chunk.id)) else chunk
            self._merged[chunk.id] = m
            if m.response_metadata.keys() & {
                "finish_reason",
                "stop_reason",
                "done",
            }:
                return m
            return None
        return chunk

    def partial_merged[T: BaseMessage](self, chunk: T) -> T:
        """返回部分合并的消息。"""
        if isinstance(chunk, BaseMessageChunk) and chunk.id:
            m = cast(T, s + chunk) if (s := self._merged.get(chunk.id)) else chunk
            self._merged[chunk.id] = m
            return m
        return chunk
