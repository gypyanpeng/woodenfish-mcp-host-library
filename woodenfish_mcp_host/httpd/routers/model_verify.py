import asyncio
import json
from collections.abc import Callable, Generator
from contextlib import AsyncExitStack, contextmanager
from enum import StrEnum
from logging import getLogger
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.pregel.io import AddableUpdatesDict
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from woodenfish_mcp_host.host.conf import HostConfig, LogConfig
from woodenfish_mcp_host.host.conf.llm import LLMConfig, LLMConfigTypes
from woodenfish_mcp_host.host.errors import ThreadQueryError
from woodenfish_mcp_host.host.host import woodenfishMcpHost
from woodenfish_mcp_host.host.tools.misc import TestTool
from woodenfish_mcp_host.httpd.dependencies import get_app
from woodenfish_mcp_host.httpd.routers.utils import EventStreamContextManager
from woodenfish_mcp_host.httpd.server import woodenfishHostAPI

logger = getLogger(__name__)

model_verify = APIRouter(tags=["model_verify"])


VERIFY_SUBJECTS = Literal["connection", "tools", "tools_in_prompt"]
DEFAULT_VERIFY_SUBJECTS: list[VERIFY_SUBJECTS] = [
    "connection",
    "tools",
    "tools_in_prompt",
]


class ToolVerifyState(StrEnum):
    """工具验证状态。"""

    TOOL_NOT_USED = "TOOL_NOT_USED"
    TOOL_CALLED = "TOOL_CALLED"
    TOOL_RESPONDED = "TOOL_RESPONDED"
    ERROR = "ERROR"

    # Tool use is successful, no need to check tool in prompt
    SKIPPED = "SKIPPED"


class ToolVerifyResult(BaseModel):
    """工具验证结果。"""

    success: bool = False
    final_state: ToolVerifyState | None = None
    error_msg: str | None = None


class ConnectionVerifyState(StrEnum):
    """连接验证状态。"""

    SKIPPED = "SKIPPED"
    CONNECTED = "CONNECTED"
    ERROR = "ERROR"


class ConnectionVerifyResult(BaseModel):
    """连接验证结果。"""

    success: bool = False
    final_state: ConnectionVerifyState | None = None
    error_msg: str | None = None


class ModelVerifyResult(BaseModel):
    """模型验证结果。"""

    success: bool = False
    connecting: ConnectionVerifyResult = Field(default_factory=ConnectionVerifyResult)
    support_tools: ToolVerifyResult = Field(default_factory=ToolVerifyResult)
    support_tools_in_prompt: ToolVerifyResult = Field(default_factory=ToolVerifyResult)

    model_config = ConfigDict(
        alias_generator=to_camel,
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
    )


class ModelVerifyProgress(BaseModel):
    """模型验证进度。"""

    type: Literal["progress"] = "progress"
    step: int
    model_name: str
    test_type: VERIFY_SUBJECTS
    ok: bool
    final_state: ToolVerifyState | ConnectionVerifyState | None
    error: str | None

    model_config = ConfigDict(
        alias_generator=to_camel,
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
    )


class ModelVerifyService:
    """模型验证服务。"""

    _abort_signal: asyncio.Event = asyncio.Event()
    _stream_progress: Callable | None = None

    def __init__(
        self,
        stream_progress: Callable | None = None,
        original_host_config: HostConfig | None = None,
    ) -> None:
        """Initialize the model verify service.

        Args:
            stream_progress (Callable): The stream progress callback.
            original_host_config (HostConfig): The original host config.
        """
        self._abort_signal = asyncio.Event()
        self._stream_progress = stream_progress
        self._original_host_config = original_host_config

    async def test_models(
        self,
        llm_configs: list[LLMConfigTypes],
    ) -> dict[str, ModelVerifyResult]:
        """Test the models.

        Args:
            llm_configs: The LLM configurations.

        Returns:
            dict[str, ModelVerifyResult]: The results.
        """
        results = {}
        for llm_config in llm_configs:
            if self._abort_signal.is_set():
                break
            verify_subjects = get_verify_subjects(llm_config)
            results[llm_config.model] = await self.test_model(
                llm_config,
                verify_subjects,
                llm_configs.index(llm_config) * len(DEFAULT_VERIFY_SUBJECTS),
            )
        return results

    def abort(self) -> None:
        """Abort the test."""
        self._abort_signal.set()

    @contextmanager
    def _handle_abort(self, abort_func: Callable) -> Generator[None, None, None]:
        async def wait_for_abort() -> None:
            await self._abort_signal.wait()
            abort_func()

        task = asyncio.create_task(wait_for_abort())
        try:
            yield
        finally:
            task.cancel()

    async def _report_progress(  # noqa: PLR0913
        self,
        step: int,
        model_name: str,
        test_type: VERIFY_SUBJECTS,
        ok: bool,
        final_state: ToolVerifyState | ConnectionVerifyState | None,
        error: str | None,
    ) -> None:
        if not self._stream_progress:
            return
        await self._stream_progress(
            ModelVerifyProgress(
                step=step,
                model_name=model_name,
                test_type=test_type,
                ok=ok,
                final_state=final_state,
                error=error,
            ).model_dump()
        )

    async def test_model(
        self,
        llm_config: LLMConfigTypes,
        subjects: list[VERIFY_SUBJECTS],
        steps: int,
    ) -> ModelVerifyResult:
        """Run the model.

        Args:
            llm_config: The LLM configuration.
            subjects: Subjects to verify.
            steps: The steps to verify.

        Returns:
            ModelVerifyResult
        """
        if self._original_host_config:
            log_config = self._original_host_config.log_config
        else:
            log_config = LogConfig(log_dir=Path.cwd() / "logs")

        host = woodenfishMcpHost(
            HostConfig(
                llm=llm_config,
                mcp_servers={},
                log_config=log_config,
            )
        )
        async with host:
            con_ok = False
            con_state = ConnectionVerifyState.SKIPPED
            con_error = None

            tools_ok = False
            tools_state = ToolVerifyState.SKIPPED
            tools_error = None

            tools_in_prompt_ok = False
            tools_in_prompt_state = ToolVerifyState.SKIPPED
            tools_in_prompt_error = None

            n_step = steps

            if "connection" in subjects:
                con_ok, con_error, con_state = await self._check_connection(host)

            n_step += 1
            await self._report_progress(
                step=n_step,
                model_name=llm_config.model,
                test_type="connection",
                ok=con_ok,
                final_state=con_state,
                error=con_error,
            )

            if "tools" in subjects:
                tools_ok, tools_error, tools_state = await self._check_tools(host)

            n_step += 1
            await self._report_progress(
                step=n_step,
                model_name=llm_config.model,
                test_type="tools",
                ok=tools_ok,
                final_state=tools_state,
                error=tools_error,
            )

            if not tools_ok and "tools_in_prompt" in subjects:
                (
                    tools_in_prompt_ok,
                    tools_in_prompt_error,
                    tools_in_prompt_state,
                ) = await self._check_tools_in_prompt(host)

            n_step += 1
            await self._report_progress(
                step=n_step,
                model_name=llm_config.model,
                test_type="tools_in_prompt",
                ok=tools_in_prompt_ok,
                final_state=tools_in_prompt_state,
                error=tools_in_prompt_error,
            )

            return ModelVerifyResult(
                success=True,
                connecting=ConnectionVerifyResult(
                    success=con_ok,
                    final_state=con_state,
                    error_msg=con_error,
                ),
                support_tools=ToolVerifyResult(
                    success=tools_ok,
                    final_state=tools_state,
                    error_msg=tools_error,
                ),
                support_tools_in_prompt=ToolVerifyResult(
                    success=tools_in_prompt_ok,
                    final_state=tools_in_prompt_state,
                    error_msg=tools_in_prompt_error,
                ),
            )

    async def _check_connection(
        self, host: woodenfishMcpHost
    ) -> tuple[bool, str | None, ConnectionVerifyState]:
        """检查模型是否连接。"""
        logger.debug("Checking connection, llm: %s", host.config.llm)
        try:
            chat = host.chat(volatile=True)
            async with AsyncExitStack() as stack:
                await stack.enter_async_context(chat)
                stack.enter_context(self._handle_abort(chat.abort))
                _responses = [
                    response
                    async for response in chat.query(
                        "Only return 'Hi' strictly", stream_mode=["updates"]
                    )
                ]
            return True, None, ConnectionVerifyState.CONNECTED
        except ThreadQueryError as e:
            logger.exception("检查连接失败")
            return False, str(e.error), ConnectionVerifyState.ERROR
        except Exception as e:
            logger.exception("检查连接失败")
            return False, str(e), ConnectionVerifyState.ERROR

    async def _check_tools(
        self, host: woodenfishMcpHost
    ) -> tuple[bool, str | None, ToolVerifyState | None]:
        """检查模型是否支持工具。"""
        logger.debug("Checking tools, llm: %s", host.config.llm)
        try:
            state = ToolVerifyState.TOOL_NOT_USED

            test_tool = TestTool()
            chat = host.chat(
                volatile=True,
                tools=[test_tool.weather_tool],
                tools_in_prompt=False,
            )

            async with AsyncExitStack() as stack:
                await stack.enter_async_context(chat)
                stack.enter_context(self._handle_abort(chat.abort))
                async for _, data in chat.query(
                    "使用工具检查台北的天气",
                    stream_mode=["updates"],
                ):
                    if isinstance(data, AddableUpdatesDict):
                        for _, v in data.items():
                            for msg in v.get("messages", []):
                                if isinstance(msg, AIMessage) and msg.tool_calls:
                                    state = ToolVerifyState.TOOL_CALLED
                                if isinstance(msg, ToolMessage):
                                    state = ToolVerifyState.TOOL_RESPONDED

            return test_tool.called, None, state
        except ThreadQueryError as e:
            logger.exception("检查工具失败")
            return False, str(e.error), ToolVerifyState.ERROR
        except Exception as e:
            logger.exception("检查工具失败")
            return False, str(e), ToolVerifyState.ERROR

    async def _check_tools_in_prompt(
        self, host: woodenfishMcpHost
    ) -> tuple[bool, str | None, ToolVerifyState | None]:
        """检查模型是否支持在提示中使用工具。"""
        logger.debug("Checking tools in prompt, llm: %s", host.config.llm)
        try:
            state = ToolVerifyState.TOOL_NOT_USED

            test_tool = TestTool()
            chat = host.chat(
                volatile=True,
                tools=[test_tool.weather_tool],
                tools_in_prompt=True,
            )

            async with AsyncExitStack() as stack:
                await stack.enter_async_context(chat)
                stack.enter_context(self._handle_abort(chat.abort))
                async for _, data in chat.query(
                    "使用工具检查台北的天气",
                    stream_mode=["updates"],
                ):
                    if isinstance(data, AddableUpdatesDict):
                        for _, v in data.items():
                            for msg in v.get("messages", []):
                                if isinstance(msg, AIMessage) and msg.tool_calls:
                                    state = ToolVerifyState.TOOL_CALLED
                                if isinstance(msg, ToolMessage):
                                    state = ToolVerifyState.TOOL_RESPONDED

            return test_tool.called, None, state
        except ThreadQueryError as e:
            logger.exception("检查提示中的工具失败")
            return False, str(e.error), ToolVerifyState.ERROR
        except Exception as e:
            logger.exception("检查提示中的工具失败")
            return False, str(e), ToolVerifyState.ERROR


class ModelVerifyRequest(BaseModel):
    """模型验证请求。"""

    model_settings: LLMConfig | None = Field(alias="modelSettings", default=None)


def get_verify_subjects(llm_config: LLMConfigTypes) -> list[VERIFY_SUBJECTS]:
    """获取验证主题。"""
    if llm_config.model_provider == "ollama":
        logger.info(
            "检测到 Ollama 提供程序，仅检查连接和提示中的工具"
        )
        return ["connection", "tools_in_prompt"]
    return DEFAULT_VERIFY_SUBJECTS


@model_verify.post("")
async def do_verify_model(
    app: woodenfishHostAPI = Depends(get_app),
    settings: ModelVerifyRequest | None = None,
) -> ModelVerifyResult:
    """验证模型是否支持流式传输功能。

    Returns: # 返回：
        ModelVerifyResult
    """
    woodenfish_host = app.woodenfish_host["default"]

    llm_config = settings.model_settings if settings else None

    if not llm_config:
        llm_config = woodenfish_host._config.llm  # noqa: SLF001

    test_service = ModelVerifyService(
        original_host_config=woodenfish_host.config,
    )
    verify_subjects = get_verify_subjects(llm_config)
    return await test_service.test_model(llm_config, verify_subjects, 0)


@model_verify.post("/streaming")
async def verify_model(
    request: Request,
    app: woodenfishHostAPI = Depends(get_app),
    settings: dict[str, list[LLMConfigTypes]] | None = None,
) -> StreamingResponse:
    """验证模型是否支持流式传输功能。

    Returns: # 返回：
        CompletionEventStreamContextManager
    """
    woodenfish_host = app.woodenfish_host["default"]

    llm_configs = settings.get("modelSettings") if settings else None
    if not llm_configs:
        llm_configs = [woodenfish_host.config.llm]
    stream = EventStreamContextManager()
    test_service = ModelVerifyService(
        stream_progress=lambda x: stream.write(json.dumps(x)),
        original_host_config=woodenfish_host.config,
    )
    response = stream.get_response()

    @contextmanager
    def handle_connection() -> Generator[None, None, None]:
        async def abort_func() -> None:
            while not await request.is_disconnected():
                await asyncio.sleep(1)
            test_service.abort()

        task = asyncio.create_task(abort_func())
        try:
            yield
        finally:
            task.cancel()

    async def process() -> None:
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(stream)
            stack.enter_context(handle_connection())
            results = await test_service.test_models(llm_configs)
            await stream.write(
                json.dumps(
                    {
                        "type": "final",
                        "results": [
                            {
                                "modelName": n,
                                "connection": {
                                    "ok": r.connecting.success,
                                    "finalState": r.connecting.final_state,
                                    "error": r.connecting.error_msg,
                                },
                                "tools": {
                                    "ok": r.support_tools.success,
                                    "finalState": r.support_tools.final_state,
                                    "error": r.support_tools.error_msg,
                                },
                                "toolsInPrompt": {
                                    "ok": r.support_tools_in_prompt.success,
                                    "finalState": r.support_tools_in_prompt.final_state,
                                    "error": r.support_tools_in_prompt.error_msg,
                                },
                            }
                            for n, r in results.items()
                        ],
                    }
                )
            )

    stream.add_task(process)
    return response
