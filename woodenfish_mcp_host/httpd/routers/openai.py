import asyncio
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Literal

from fastapi import APIRouter, Depends, Request
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from woodenfish_mcp_host.httpd.conf.prompt import PromptKey
from woodenfish_mcp_host.httpd.dependencies import get_app
from woodenfish_mcp_host.httpd.routers.models import ResultResponse, StreamMessage
from woodenfish_mcp_host.httpd.routers.utils import ChatProcessor, EventStreamContextManager
from woodenfish_mcp_host.httpd.server import woodenfishHostAPI

openai = APIRouter(tags=["openai"])


class OpenaiModel(BaseModel):
    """表示具有基本属性的 OpenAI 模型。"""

    id: str
    type: str
    owned_by: str


class ModelsResult(ResultResponse):
    """列出可用 OpenAI 模型的响应模型。"""

    models: list[OpenaiModel]


class CompletionsMessage(BaseModel):
    """OpenAI completions API 中的消息。"""

    role: str
    content: str


class CompletionsMessageResp(CompletionsMessage):
    """OpenAI completions API 中的消息。"""

    refusal: None = None
    role: str | None = None


class CompletionsArgs(BaseModel):
    """OpenAI completions API 的参数。"""

    messages: list[CompletionsMessage]
    stream: bool
    tool_choice: Literal["none", "auto"]


class CompletionsUsage(BaseModel):
    """OpenAI completions API 的使用信息。"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionsChoice(BaseModel):
    """OpenAI completions API 中的一个选择。"""

    index: int
    message: CompletionsMessageResp | None = None
    logprobs: None = None
    delta: CompletionsMessageResp | dict | None = None
    finish_reason: str | None = None


class CompletionsResult(BaseModel):
    """OpenAI completions API 的结果。"""

    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list  # of what
    usage: CompletionsUsage | None = None
    system_fingerprint: str = "fp_woodenfish"


class CompletionEventStreamContextManager(EventStreamContextManager):
    """OpenAI completions API 的上下文管理器。"""

    chat_id: str
    model: str
    abort_signal: asyncio.Event

    def __init__(self, chat_id: str, model: str) -> None:
        """初始化 completions 事件流上下文管理器。"""
        self.chat_id = chat_id
        self.model = model
        self.abort_signal = asyncio.Event()
        super().__init__()

    async def write(self, data: str | StreamMessage | CompletionsResult) -> None:
        """将数据写入流。"""
        if isinstance(data, StreamMessage) and data.type == "text":
            data = CompletionsResult(
                id=f"chatcmpl-{self.chat_id}",
                model=self.model,
                object="chat.completion.chunk",
                choices=[
                    CompletionsChoice(
                        index=0,
                        delta=CompletionsMessageResp(content=str(data.content))
                        if data.content
                        else CompletionsMessageResp(role="assistant", content=""),
                    )
                ],
            ).model_dump_json()
            await super().write(data)
        elif isinstance(data, CompletionsResult):
            await super().write(data.model_dump_json())

    async def _generate(self) -> AsyncGenerator[str, None]:
        """生成流。"""
        try:
            async for chunk in super()._generate():
                yield chunk
        finally:
            self.abort_signal.set()


@openai.get("/")
async def get_openai() -> ResultResponse:
    """返回 woodenfish Compatible API 的欢迎消息。

    返回：
        ResultResponse: 带有欢迎消息的成功响应。
    """
    return ResultResponse(success=True, message="Welcome to woodenfish Compatible API! 🚀")


@openai.get("/models")
async def list_models(app: woodenfishHostAPI = Depends(get_app)) -> ModelsResult:
    """列出所有可用的 OpenAI 兼容模型。

    返回：
        ModelsResult: 可用模型的列表。
    """
    return ModelsResult(
        success=True,
        models=[
            OpenaiModel(
                id=m.config.llm.model,
                type="model",
                owned_by=m.config.llm.model_provider,
            )
            for m in app.woodenfish_host.values()
        ],
    )


@openai.post("/chat/completions")
async def create_chat_completion(
    request: Request,
    params: CompletionsArgs,
    app: woodenfishHostAPI = Depends(get_app),
) -> object:  # idk what this actual do...
    """使用 OpenAI 兼容 API 创建聊天完成。

    返回：
        聊天完成结果。
    """
    has_system_message = False
    messages = []
    for message in params.messages:
        if message.role == "system":
            has_system_message = True
            messages.append(SystemMessage(content=message.content))
        elif message.role == "assistant":
            messages.append(AIMessage(content=message.content))
        else:
            messages.append(HumanMessage(content=message.content))

    if not has_system_message:
        disable_woodenfish_system_prompt = (
            app.model_config_manager.full_config.disable_woodenfish_system_prompt
            if app.model_config_manager.full_config
            else False
        )

        if disable_woodenfish_system_prompt:
            system_prompt = app.prompt_config_manager.get_prompt(PromptKey.CUSTOM)
        else:
            system_prompt = app.prompt_config_manager.get_prompt(PromptKey.SYSTEM)

        if system_prompt:
            messages.insert(
                0,
                SystemMessage(content=system_prompt),
            )

    woodenfish_host = app.woodenfish_host["default"]

    chat_id = str(uuid.uuid4())
    model_name = woodenfish_host._config.llm.model  # noqa: SLF001
    stream = CompletionEventStreamContextManager(chat_id, model_name)

    async def abort_handler() -> None:
        await stream.abort_signal.wait()
        await app.abort_controller.abort(chat_id)

    async def process() -> tuple[CompletionsMessageResp, CompletionsUsage]:
        async with stream:
            task = asyncio.create_task(abort_handler())
            processor = ChatProcessor(app, request.state, stream)
            result, usage = await processor.handle_chat_with_history(
                chat_id,
                None,
                messages,
                [] if params.tool_choice != "auto" else None,
            )

            await stream.write(
                CompletionsResult(
                    id=f"chatcmpl-{chat_id}",
                    model=model_name,
                    object="chat.completion.chunk",
                    choices=[
                        CompletionsChoice(
                            index=0,
                            delta={},
                            finish_reason="stop",
                        )
                    ],
                )
            )
            task.cancel()

            return (
                CompletionsMessageResp(role="assistant", content=result),
                CompletionsUsage(
                    prompt_tokens=usage.total_input_tokens,
                    completion_tokens=usage.total_output_tokens,
                    total_tokens=usage.total_tokens,
                ),
            )

    if params.stream:
        response = stream.get_response()
        stream.add_task(process)
        return response

    result, usage = await process()
    return CompletionsResult(
        id=f"chatcmpl-{chat_id}",
        model=model_name,
        choices=[
            CompletionsChoice(
                index=0,
                message=result,
                finish_reason="stop",
            )
        ],
        usage=usage,
    )
