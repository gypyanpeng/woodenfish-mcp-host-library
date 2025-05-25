"""此模块包含主机的 ChatAgentFactory。

它使用 langgraph.prebuilt.create_react_agent 来创建代理。
"""

from collections.abc import Sequence
from typing import Literal, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver, V
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import MessagesState
from langgraph.managed import IsLastStep, RemainingSteps
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.store.base import BaseStore
from langgraph.utils.runnable import RunnableCallable
from pydantic import BaseModel

from woodenfish_mcp_host.host.agents.agent_factory import AgentFactory, initial_messages
from woodenfish_mcp_host.host.agents.tools_in_prompt import (
    convert_messages,
    extract_tool_calls,
)
from woodenfish_mcp_host.host.helpers import today_datetime
from woodenfish_mcp_host.host.prompt import PromptType, tools_prompt

StructuredResponse = dict | BaseModel
StructuredResponseSchema = dict | type[BaseModel]


class AgentState(MessagesState):
    """代理的状态。"""

    is_last_step: IsLastStep
    today_datetime: str
    remaining_steps: RemainingSteps
    structured_response: StructuredResponse


MINIMUM_STEPS_TOOL_CALL_REQUIRED = 2
# 需要调用工具的最小步骤数

PROMPT_RUNNABLE_NAME = "Prompt"
# 提示词可运行对象的名称


# from langgraph.prebuilt
def get_prompt_runnable(prompt: PromptType | ChatPromptTemplate | None) -> Runnable:
    """获取提示词可运行对象。"""
    prompt_runnable: Runnable
    if prompt is None:
        prompt_runnable = RunnableCallable(
            lambda state: state if isinstance(state, list) else (state.get("messages", None) or []), name=PROMPT_RUNNABLE_NAME
        )
    elif isinstance(prompt, str):
        _system_message: BaseMessage = SystemMessage(content=prompt)

        def _func(state: AgentState | ChatPromptValue | list[BaseMessage]) -> list[BaseMessage]:
            if isinstance(state, ChatPromptValue):
                return [_system_message, *state.to_messages()]
            if isinstance(state, list):
                return [_system_message, *state]
            return [_system_message, *(state.get("messages", None) or [])]

        prompt_runnable = RunnableCallable(_func, name=PROMPT_RUNNABLE_NAME)
    elif isinstance(prompt, SystemMessage):
        prompt_runnable = RunnableCallable(
            lambda state: [prompt, *(state if isinstance(state, list) else (state.get("messages", None) or []))],
            name=PROMPT_RUNNABLE_NAME,
        )
    elif callable(prompt):
        prompt_runnable = RunnableCallable(
            prompt,
            name=PROMPT_RUNNABLE_NAME,
        )
    elif isinstance(prompt, Runnable):
        prompt_runnable = prompt
    else:
        raise ValueError(f"Got unexpected type for `prompt`: {type(prompt)}")

    return prompt_runnable


class ChatAgentFactory(AgentFactory[AgentState]):
    """A factory for ChatAgents."""

    def __init__(
        self,
        model: BaseChatModel,
        tools: Sequence[BaseTool] | ToolNode,
        tools_in_prompt: bool = False,
    ) -> None:
        """Initialize the chat agent factory."""
        self._model = model
        self._model_class = type(model).__name__
        self._tools = tools
        self._tools_in_prompt = tools_in_prompt
        self._response_format: (
            StructuredResponseSchema | tuple[str, StructuredResponseSchema] | None
        ) = None

        # 在调用 self._build_graph 时改变
        self._tool_classes: list[BaseTool] = []
        self._should_return_direct: set[str] = set()
        self._graph: StateGraph | None = None

        # 在调用 self.create_agent 时改变
        self._prompt: Runnable = get_prompt_runnable(None)
        self._tool_prompt: Runnable = get_prompt_runnable(None)

        # Initialize the tool prompt
        if self._tools_in_prompt:
            if isinstance(self._tools, ToolNode):
                tools = list(self._tools.tools_by_name.values())
                self._tool_prompt = get_prompt_runnable(tools_prompt(tools))
            else:
                self._tool_prompt = get_prompt_runnable(tools_prompt(self._tools))

        self._build_graph()

    def _check_more_steps_needed(
        self, state: AgentState, response: BaseMessage
    ) -> bool:
        """检查是否需要更多步骤。"""
        has_tool_calls = (
            isinstance(response, AIMessage) and response.tool_calls is not None
        )
        all_tools_return_direct = (
            all(
                call["name"] in self._should_return_direct
                for call in response.tool_calls
            )
            if isinstance(response, AIMessage)
            else False
        )
        remaining_steps = state.get("remaining_steps", None)
        is_last_step = state.get("is_last_step", False)

        return (
            (remaining_steps is None and is_last_step and has_tool_calls)
            or (
                remaining_steps is not None
                and remaining_steps < 1
                and all_tools_return_direct
            )
            or (
                remaining_steps is not None
                and remaining_steps < MINIMUM_STEPS_TOOL_CALL_REQUIRED
                and has_tool_calls
            )
        )

    def _call_model(self, state: AgentState, config: RunnableConfig) -> AgentState:
        # TODO: _validate_chat_history 验证聊天记录
        if not self._tools_in_prompt:
            model = self._model.bind_tools(self._tool_classes)
            model_runnable = self._prompt | drop_empty_messages | model
        else:
            model_runnable = (
                self._prompt
                | self._tool_prompt
                | convert_messages
                | drop_empty_messages
                | self._model
            )

        response = model_runnable.invoke(state, config)
        if isinstance(response, AIMessage):
            response = extract_tool_calls(response)
        if self._check_more_steps_needed(state, response):
            response = AIMessage(
                id=response.id,
                content="Sorry, need more steps to process this request.",
            )
        return cast(AgentState, {"messages": [response]})

    def _generate_structured_response(
        self, state: AgentState, config: RunnableConfig
    ) -> AgentState:
        """生成结构化响应。"""
        messages = state["messages"][:-1]
        if isinstance(self._response_format, tuple):
            system_prompt, structured_response_schema = self._response_format
            messages = [SystemMessage(content=system_prompt), *list(messages)]

        model_with_structured_output = self._model.with_structured_output(
            cast(StructuredResponseSchema, structured_response_schema)
        )

        response = model_with_structured_output.invoke(messages, config)
        return cast(AgentState, {"structured_response": response})

    def _before_agent(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """在代理运行前执行的节点。"""
        configurable = config.get("configurable", {})
        max_input_tokens: int | None = configurable.get("max_input_tokens")
        oversize_policy: Literal["window"] | None = configurable.get("oversize_policy")
        if max_input_tokens is None or oversize_policy is None:
            return cast(AgentState, {"messages": []})
        if oversize_policy == "window":
            messages: list[BaseMessage] = trim_messages(
                state["messages"],
                max_tokens=max_input_tokens,
                token_counter=count_tokens_approximately,
            )
            remove_messages = [
                RemoveMessage(id=m.id)  # type: ignore
                for m in state["messages"]
                if m not in messages
            ]
            return cast(AgentState, {"messages": remove_messages})

        return cast(AgentState, {"messages": []})

    def _after_agent(self, state: AgentState) -> str:
        """在代理运行后执行的节点。"""
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return (
                END if self._response_format is None else "generate_structured_response"
            )
        return "tools"

    def _after_tools(self, state: AgentState) -> str:
        """在工具运行后执行的节点。"""
        for m in reversed(state["messages"]):
            if not isinstance(m, ToolMessage):
                break
            if m.name in self._should_return_direct:
                return END
        return "before_agent"

    def _build_graph(self) -> None:
        """构建代理图。"""
        graph = StateGraph(AgentState)

        graph.add_node("before_agent", self._before_agent)
        graph.set_entry_point("before_agent")

        # 创建代理节点
        graph.add_node("agent", self._call_model)
        graph.add_edge("before_agent", "agent")

        tool_node = (
            self._tools if isinstance(self._tools, ToolNode) else ToolNode(self._tools)
        )
        self._tool_classes = list(tool_node.tools_by_name.values())
        graph.add_node("tools", tool_node)
        self._should_return_direct = {
            t.name for t in self._tool_classes if t.return_direct
        }

        if self._response_format:
            graph.add_node(
                "generate_structured_response", self._generate_structured_response
            )
            graph.add_edge("generate_structured_response", END)
            next_node = ["tools", "generate_structured_response"]
        else:
            next_node = ["tools", END]

        graph.add_conditional_edges(
            "agent",
            self._after_agent,
            next_node,
        )

        # one of the tools should return direct
        if self._should_return_direct:
            graph.add_conditional_edges("tools", self._after_tools)
        else:
            graph.add_edge("tools", "before_agent")

        self._graph = graph

    def create_agent(
        self,
        *,
        prompt: PromptType | ChatPromptTemplate,
        checkpointer: BaseCheckpointSaver[V] | None = None,
        store: BaseStore | None = None,
        debug: bool = False,
    ) -> CompiledGraph:
        """创建一个 react 代理。"""
        self._prompt = get_prompt_runnable(prompt)
        if self._graph is None:
            raise ValueError("Graph is not built")
        return self._graph.compile(checkpointer=checkpointer, store=store, debug=debug)

    def create_initial_state(
        self,
        *,
        query: str | HumanMessage | list[BaseMessage],
    ) -> AgentState:
        """为查询创建一个初始状态。"""
        return AgentState(
            messages=initial_messages(query),
            is_last_step=False,
            today_datetime=today_datetime(),
            remaining_steps=100,
        )  # type: ignore

    def state_type(
        self,
    ) -> type[AgentState]:
        """获取状态类型。"""
        return AgentState


def get_chat_agent_factory(
    model: BaseChatModel,
    tools: Sequence[BaseTool] | ToolNode,
    tools_in_prompt: bool = False,
) -> ChatAgentFactory:
    """获取代理工厂。"""
    return ChatAgentFactory(model, tools, tools_in_prompt)


@RunnableCallable
def drop_empty_messages(inpt: ChatPromptValue | list[BaseMessage]) -> list[BaseMessage]:
    """删除空消息。"""
    messages = inpt.to_messages() if isinstance(inpt, ChatPromptValue) else inpt

    result = []
    for message in messages:
        # AIMessage 有更多约束
        if isinstance(message, AIMessage):
            if (
                not message.content
                and not message.tool_calls
                and not message.invalid_tool_calls
            ):
                continue
        # ToolMessage, SystemMessage, HumanMessage 需要有内容
        elif not message.content:
            continue
        result.append(message)
    return result
