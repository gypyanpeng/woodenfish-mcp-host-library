from typing import Literal, Protocol

from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, V
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import MessagesState
from langgraph.store.base import BaseStore

from woodenfish_mcp_host.host.prompt import PromptType


# XXX 是否有更好的方法？
class AgentFactory[T: MessagesState](Protocol):
    """创建代理的工厂。

    实现此协议以创建您自己的自定义代理。
    将工厂传递给主机以创建对话代理。
    """

    def create_agent(
        self,
        *,
        prompt: PromptType | ChatPromptTemplate,
        checkpointer: BaseCheckpointSaver[V] | None = None,
        store: BaseStore | None = None,
        debug: bool = False,
    ) -> CompiledGraph:
        """创建一个代理。

        参数：
            prompt: 用于代理的提示词。
            checkpointer: 用于保存代理状态的 langgraph 检查点管理器。
            store: 用于长期记忆的 langgraph 存储。
            debug: 是否为代理启用调试模式。

        返回：
            编译后的代理。
        """
        ...

    def create_config(
        self,
        *,
        user_id: str,
        thread_id: str,
        max_input_tokens: int | None = None,
        oversize_policy: Literal["window"] | None = None,
    ) -> RunnableConfig | None:
        """为代理创建一个配置。

        重写此方法以自定义代理的配置。
        默认实现返回以下配置：
        {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
            },
            "recursion_limit": 100,
        }
        """
        return {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
                "max_input_tokens": max_input_tokens,
                "oversize_policy": oversize_policy,
            },
            "recursion_limit": 102,
        }

    def create_initial_state(
        self,
        *,
        query: str | HumanMessage | list[BaseMessage],
    ) -> T:
        """为查询创建一个初始状态。"""
        ...

    def state_type(
        self,
    ) -> type[T]:
        """获取状态的类型。"""
        ...

    def create_prompt(
        self,
        *,
        system_prompt: str,
    ) -> ChatPromptTemplate:
        """为代理创建一个提示词。

        重写此方法以自定义代理的提示词。
        默认实现返回一个带有消息占位符的提示词。
        """
        return ChatPromptTemplate.from_messages(  # type: ignore[arg-type]
            [
                ("system", system_prompt),
                ("placeholder", "{messages}"),
            ],
        )


def initial_messages(
    query: str | HumanMessage | list[AnyMessage | BaseMessage],
) -> list[AnyMessage]:
    """为您的状态创建初始消息。

    状态必须包含一个键为 'messages' 且类型为 list[BaseMessage] 的值。
    此实用函数可将查询转换为 list[BaseMessage]，无论查询是字符串还是 BaseMessage。

    参数：
        query: 用于创建初始消息的查询。

    返回：
        HumanMessage 对象的列表。

    """
    if isinstance(query, list):
        messages = []
        for q in query:
            messages.append(
                q if isinstance(q, BaseMessage) else HumanMessage(content=q)
            )
        return messages
    return [query] if isinstance(query, BaseMessage) else [HumanMessage(content=query)]
