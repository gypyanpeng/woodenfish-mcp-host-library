import uuid
from collections.abc import Callable, Sequence
from time import sleep
from typing import Any

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field


def default_responses() -> list[AIMessage]:
    """假模型的默认响应。"""
    return [
        AIMessage(content="I am a fake model."),
    ]


class FakeMessageToolModel(BaseChatModel):
    """一个假工具模型。

    使用此模型测试工具调用。

    示例：
        responses = [
            AIMessage(
                content="我是一个假模型。",
                tool_calls=[ToolCall(name="fake_tool", args={"arg": "arg"}, id="id")],
            ),
            AIMessage(
                content="最终的 AI 消息",
            ),
        ]
        model = FakeMessageToolModel(responses=responses)
    """

    responses: list[AIMessage] = Field(default_factory=default_responses)
    query_history: list[BaseMessage] = Field(default_factory=list)
    sleep: float | None = None
    i: int = 0

    def _generate(
        self,
        messages: list[BaseMessage],
        _stop: list[str] | None = None,
        _run_manager: CallbackManagerForLLMRun | None = None,
        **_kwargs: Any,
    ) -> ChatResult:
        self.query_history.extend(messages)
        response = self.responses[self.i].model_copy()
        if response.id is None:
            response.id = str(uuid.uuid4())
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        if self.sleep is not None and self.sleep > 0:
            sleep(self.sleep)
        generation = ChatGeneration(message=response)
        return ChatResult(generations=[generation])

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """将工具绑定到模型。"""
        formatted_tools = [convert_to_openai_tool(tool, strict=False) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "fake-model"


def load_model(
    *,
    responses: list[AIMessage] | None = None,
    **_kwargs: dict,
) -> FakeMessageToolModel:
    """加载假模型。"""
    return FakeMessageToolModel(responses=responses or default_responses())
