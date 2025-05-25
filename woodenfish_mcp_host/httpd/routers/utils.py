import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine
from contextlib import AsyncExitStack, suppress
from typing import TYPE_CHECKING, Any, Self
from uuid import uuid4

from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
from starlette.datastructures import State

from woodenfish_mcp_host.host.errors import LogBufferNotFoundError
from woodenfish_mcp_host.host.tools.log import LogEvent, LogManager, LogMsg
from woodenfish_mcp_host.host.tools.model_types import ClientState
from woodenfish_mcp_host.httpd.conf.prompt import PromptKey
from woodenfish_mcp_host.httpd.database.models import (
    Message,
    NewMessage,
    QueryInput,
    ResourceUsage,
    Role,
)
from woodenfish_mcp_host.httpd.routers.models import (
    ChatInfoContent,
    MessageInfoContent,
    StreamMessage,
    TokenUsage,
    ToolCallsContent,
    ToolResultContent,
)
from woodenfish_mcp_host.httpd.server import woodenfishHostAPI
from woodenfish_mcp_host.httpd.store.store import SUPPORTED_IMAGE_EXTENSIONS, Store
from woodenfish_mcp_host.log import TRACE

if TYPE_CHECKING:
    from woodenfish_mcp_host.host.host import woodenfishMcpHost
    from woodenfish_mcp_host.httpd.middlewares.general import woodenfishUser

title_prompt = """您是根据用户输入生成标题的工具。
您的唯一任务是根据用户输入生成一个简短的标题。
重要：
- 只输出标题
- 不要尝试回答或解决用户输入的问题。
- 不要尝试使用任何工具生成标题
- 没有思考、推理、解释、引用或额外文本
- 结尾没有标点符号
- 如果输入只有 URL，输出 URL 的描述，例如"xxx 网站的 URL"
- 如果输入包含繁体中文，标题使用繁体中文。
- 对于所有其他语言，标题使用与输入相同的语言。"""  # noqa: E501


logger = logging.getLogger(__name__)


class EventStreamContextManager:
    """事件流的上下文管理器。"""

    task: asyncio.Task | None = None
    done: bool = False
    response: StreamingResponse | None = None
    _exit_message: str | None = None

    def __init__(self) -> None:
        """初始化事件流上下文管理器。"""
        self.queue = asyncio.Queue()

    def add_task(
        self, func: Callable[[], Coroutine[Any, Any, Any]], *args: Any, **kwargs: Any
    ) -> None:
        """向事件流添加任务。"""
        self.task = asyncio.create_task(func(*args, **kwargs))

    async def __aenter__(self) -> Self:
        """进入上下文管理器。"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """退出上下文管理器。"""
        if exc_val:
            import traceback

            logger.error(traceback.format_exception(exc_type, exc_val, exc_tb))
            self._exit_message = StreamMessage(
                type="error",
                content=f"<thread-query-error>{exc_val}</thread-query-error>",
            ).model_dump_json(by_alias=True)

        self.done = True
        await self.queue.put(None)  # Signal completion

    async def write(self, data: str | StreamMessage) -> None:
        """将数据写入事件流。

        参数：
            data (str): 要写入流的数据。
        """
        if isinstance(data, BaseModel):
            data = json.dumps({"message": data.model_dump_json(by_alias=True)})
        await self.queue.put(data)

    async def _generate(self) -> AsyncGenerator[str, None]:
        """生成事件流内容。"""
        while not self.done or not self.queue.empty():
            chunk = await self.queue.get()
            if chunk is None:  # End signal
                continue
            yield "data: " + chunk + "\n\n"
        if self._exit_message:
            yield "data: " + json.dumps({"message": self._exit_message}) + "\n\n"
        yield "data: [DONE]\n\n"

    def get_response(self) -> StreamingResponse:
        """获取流式响应。"""
        self.response = StreamingResponse(
            content=self._generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
        return self.response


class ChatError(Exception):
    """聊天错误。"""

    def __init__(self, message: str) -> None:
        """初始化聊天错误。"""
        self.message = message


class ChatProcessor:
    """聊天处理器。"""

    def __init__(
        self,
        app: woodenfishHostAPI,
        request_state: State,
        stream: EventStreamContextManager,
    ) -> None:
        """初始化聊天处理器。"""
        self.app = app
        self.request_state = request_state
        self.stream = stream
        self.store: Store = app.store
        self.woodenfish_host: woodenfishMcpHost = app.woodenfish_host["default"]
        self._str_output_parser = StrOutputParser()
        self.disable_woodenfish_system_prompt = (
            app.model_config_manager.full_config.disable_woodenfish_system_prompt
            if app.model_config_manager.full_config
            else False
        )

    async def handle_chat(  # noqa: C901, PLR0912, PLR0915
        self,
        chat_id: str | None,
        query_input: QueryInput | None,
        regenerate_message_id: str | None,
    ) -> tuple[str, TokenUsage]:
        """处理聊天。"""
        chat_id = chat_id if chat_id else str(uuid4())
        woodenfish_user: woodenfishUser = self.request_state.woodenfish_user
        title = "New Chat"
        title_await = None
        result = ""

        if isinstance(query_input, QueryInput) and query_input.text:
            async with self.app.db_sessionmaker() as session:
                db = self.app.msg_store(session)
                if not await db.check_chat_exists(chat_id, woodenfish_user["user_id"]):
                    title_await = asyncio.create_task(
                        self._generate_title(query_input.text)
                    )

        await self.stream.write(
            StreamMessage(
                type="chat_info",
                content=ChatInfoContent(id=chat_id, title=title),
            )
        )

        start = time.time()
        if regenerate_message_id:
            if query_input:
                query_message = await self._query_input_to_message(
                    query_input, message_id=regenerate_message_id
                )
            else:
                query_message = await self._get_history_user_input(
                    chat_id, regenerate_message_id
                )
        elif query_input:
            query_message = await self._query_input_to_message(
                query_input, message_id=str(uuid4())
            )
        else:
            query_message = None
        user_message, ai_message, current_messages = await self._process_chat(
            chat_id,
            query_message,
            is_resend=regenerate_message_id is not None,
        )
        end = time.time()
        if ai_message is None:
            if title_await:
                title_await.cancel()
            return "", TokenUsage()
        assert user_message.id
        assert ai_message.id

        if title_await:
            title = await title_await

        async with self.app.db_sessionmaker() as session:
            db = self.app.msg_store(session)
            if not await db.check_chat_exists(chat_id, woodenfish_user["user_id"]):
                await db.create_chat(
                    chat_id, title, woodenfish_user["user_id"], woodenfish_user["user_type"]
                )

            if regenerate_message_id and query_message:
                await db.delete_messages_after(chat_id, query_message.id)  # type: ignore
                if query_input:
                    await db.update_message_content(
                        query_message.id,  # type: ignore
                        QueryInput(
                            text=query_input.text or "",
                            images=query_input.images or [],
                            documents=query_input.documents or [],
                            tool_calls=query_input.tool_calls,
                        ),
                    )

            for message in current_messages:
                assert message.id
                if isinstance(message, HumanMessage):
                    if not query_input or regenerate_message_id:
                        continue
                    await db.create_message(
                        NewMessage(
                            chatId=chat_id,
                            role=Role.USER,
                            messageId=message.id,
                            content=query_input.text or "",  # type: ignore
                            files=(
                                (query_input.images or [])
                                + (query_input.documents or [])
                            ),
                        ),
                    )
                elif isinstance(message, AIMessage):
                    if (
                        message.usage_metadata is None
                        or (duration := message.usage_metadata.get("total_duration"))
                        is None
                    ):
                        duration = 0 if message.id == ai_message.id else end - start
                    resource_usage = ResourceUsage(
                        model=message.response_metadata.get("model")
                        or message.response_metadata.get("model_name")
                        or "",
                        total_input_tokens=message.usage_metadata["input_tokens"]
                        if message.usage_metadata
                        else 0,
                        total_output_tokens=message.usage_metadata["output_tokens"]
                        if message.usage_metadata
                        else 0,
                        total_run_time=duration,
                    )
                    result = (
                        self._str_output_parser.invoke(message)
                        if message.content
                        else ""
                    )
                    await db.create_message(
                        NewMessage(
                            chatId=chat_id,
                            role=Role.ASSISTANT,
                            messageId=message.id,
                            content=result,
                            toolCalls=message.tool_calls,
                            resource_usage=resource_usage,
                        ),
                    )
                elif isinstance(message, ToolMessage):
                    if isinstance(message.content, list):
                        content = json.dumps(message.content)
                    elif isinstance(message.content, str):
                        content = message.content
                    else:
                        raise ValueError(
                            f"got unknown type: {type(message.content)}, "
                            f"data: {message.content}"
                        )
                    await db.create_message(
                        NewMessage(
                            chatId=chat_id,
                            role=Role.TOOL_RESULT,
                            messageId=message.id,
                            content=content,
                        ),
                    )

            await session.commit()

        logger.log(TRACE, "usermessage.id: %s", user_message.id)
        await self.stream.write(
            StreamMessage(
                type="message_info",
                content=MessageInfoContent(
                    userMessageId=user_message.id,
                    assistantMessageId=ai_message.id,
                ),
            )
        )

        await self.stream.write(
            StreamMessage(
                type="chat_info",
                content=ChatInfoContent(id=chat_id, title=title),
            )
        )

        token_usage = TokenUsage(
            totalInputTokens=ai_message.usage_metadata["input_tokens"]
            if ai_message.usage_metadata
            else 0,
            totalOutputTokens=ai_message.usage_metadata["output_tokens"]
            if ai_message.usage_metadata
            else 0,
            totalTokens=ai_message.usage_metadata["total_tokens"]
            if ai_message.usage_metadata
            else 0,
        )

        return result, token_usage

    async def handle_chat_with_history(
        self,
        chat_id: str,
        query_input: BaseMessage | None,
        history: list[BaseMessage],
        tools: list | None = None,
    ) -> tuple[str, TokenUsage]:
        """处理包含历史记录的聊天。

        参数：
            chat_id (str): 聊天 ID。
            query_input (BaseMessage | None): 查询输入。
            history (list[BaseMessage]): 历史记录。
            tools (list | None): 工具。

        返回：
            tuple[str, TokenUsage]: 结果和令牌使用情况。
        """
        _, ai_message, _ = await self._process_chat(
            chat_id, query_input, history, tools
        )
        usage = TokenUsage()
        if ai_message.usage_metadata:
            usage.total_input_tokens = ai_message.usage_metadata["input_tokens"]
            usage.total_output_tokens = ai_message.usage_metadata["output_tokens"]
            usage.total_tokens = ai_message.usage_metadata["total_tokens"]

        return str(ai_message.content), usage

    async def _process_chat(
        self,
        chat_id: str | None,
        query_input: str | QueryInput | BaseMessage | None,
        history: list[BaseMessage] | None = None,
        tools: list | None = None,
        is_resend: bool = False,
    ) -> tuple[HumanMessage, AIMessage, list[BaseMessage]]:
        messages = [*history] if history else []

        # if retry input is empty
        if query_input:
            if isinstance(query_input, str):
                messages.append(HumanMessage(content=query_input))
            elif isinstance(query_input, QueryInput):
                messages.append(await self._query_input_to_message(query_input))
            else:
                messages.append(query_input)

        woodenfish_user: woodenfishUser = self.request_state.woodenfish_user

        def _prompt_cb(_: Any) -> list[BaseMessage]:
            return messages

        prompt: str | Callable[..., list[BaseMessage]] | None = None
        if any(isinstance(m, SystemMessage) for m in messages):
            prompt = _prompt_cb
        elif self.disable_woodenfish_system_prompt and (
            custom_prompt := self.app.prompt_config_manager.get_prompt(PromptKey.CUSTOM)
        ):
            prompt = custom_prompt
        elif system_prompt := self.app.prompt_config_manager.get_prompt(
            PromptKey.SYSTEM
        ):
            prompt = system_prompt

        chat = self.woodenfish_host.chat(
            chat_id=chat_id,
            user_id=woodenfish_user.get("user_id") or "default",
            tools=tools,
            system_prompt=prompt,
            disable_default_system_prompt=self.disable_woodenfish_system_prompt,
        )
        async with AsyncExitStack() as stack:
            if chat_id:
                await stack.enter_async_context(
                    self.app.abort_controller.abort_signal(chat_id, chat.abort)
                )
            await stack.enter_async_context(chat)
            response_generator = chat.query(
                messages,
                stream_mode=["messages", "values", "updates"],
                is_resend=is_resend,
            )
            return await self._handle_response(response_generator)

        raise RuntimeError("Unreachable")

    async def _stream_text_msg(self, message: AIMessage) -> None:
        content = self._str_output_parser.invoke(message)
        if content:
            await self.stream.write(StreamMessage(type="text", content=content))
        if message.response_metadata.get("stop_reason") == "max_tokens":
            await self.stream.write(
                StreamMessage(
                    type="error",
                    content="stop_reason: max_tokens",
                )
            )

    async def _stream_tool_calls_msg(self, message: AIMessage) -> None:
        await self.stream.write(
            StreamMessage(
                type="tool_calls",
                content=[
                    ToolCallsContent(name=c["name"], arguments=c["args"])
                    for c in message.tool_calls
                ],
            )
        )

    async def _stream_tool_result_msg(self, message: ToolMessage) -> None:
        result = message.content
        with suppress(json.JSONDecodeError):
            if isinstance(result, list):
                result = [json.loads(r) if isinstance(r, str) else r for r in result]
            else:
                result = json.loads(result)
        await self.stream.write(
            StreamMessage(
                type="tool_result",
                content=ToolResultContent(name=message.name or "", result=result),
            )
        )

    async def _handle_response(  # noqa: C901, PLR0912
        self, response: AsyncIterator[dict[str, Any] | Any]
    ) -> tuple[HumanMessage | Any, AIMessage | Any, list[BaseMessage]]:
        """处理响应。

        返回：
            tuple[HumanMessage | Any, AIMessage | Any, list[BaseMessage]]:
            用户消息、AI 消息和当前查询的所有消息。
        """
        user_message = None
        ai_message = None
        values_messages: list[BaseMessage] = []
        current_messages: list[BaseMessage] = []
        async for res_type, res_content in response:
            if res_type == "messages":
                message, _ = res_content
                if isinstance(message, AIMessage):
                    logger.log(TRACE, "got AI message: %s", message.model_dump_json())
                    if message.content:
                        await self._stream_text_msg(message)
                elif isinstance(message, ToolMessage):
                    logger.log(TRACE, "got tool message: %s", message.model_dump_json())
                    await self._stream_tool_result_msg(message)
                else:
                    # idk what is this
                    logger.warning("Unknown message type: %s", message)
            elif res_type == "values" and len(res_content["messages"]) >= 2:  # type: ignore  # noqa: PLR2004
                values_messages = res_content["messages"]  # type: ignore
            elif res_type == "updates":
                # Get tool call message
                if not isinstance(res_content, dict):
                    continue

                for value in res_content.values():
                    if not isinstance(value, dict):
                        continue

                    msgs = value.get("messages", [])
                    for msg in msgs:
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            logger.log(
                                TRACE,
                                "got tool call message: %s",
                                msg.model_dump_json(),
                            )
                            await self._stream_tool_calls_msg(msg)

        # Find the most recent user and AI messages from newest to oldest
        user_message = next(
            (msg for msg in reversed(values_messages) if isinstance(msg, HumanMessage)),
            None,
        )
        ai_message = next(
            (msg for msg in reversed(values_messages) if isinstance(msg, AIMessage)),
            None,
        )
        if user_message:
            current_messages = values_messages[values_messages.index(user_message) :]

        return user_message, ai_message, current_messages

    async def _generate_title(self, query: str) -> str:
        """生成标题。"""
        chat = self.woodenfish_host.chat(
            tools=[],  # do not use tools
            system_prompt=title_prompt,
            volatile=True,
        )
        try:
            async with chat:
                response = await chat.active_agent.ainvoke(
                    {"messages": [HumanMessage(content=query)]}
                )
                if isinstance(response["messages"][-1], AIMessage):
                    return strip_title(
                        self._str_output_parser.invoke(response["messages"][-1])
                    )
        except Exception as e:
            logger.exception("Error generating title: %s", e)
        return "New Chat"

    async def _process_history_messages(
        self, history_messages: list[Message], history: list[BaseMessage]
    ) -> list[BaseMessage]:
        """处理历史消息。"""
        for message in history_messages:
            files: list[str] = message.files
            if not files:
                message_content = message.content.strip()
                if message.role == Role.USER:
                    history.append(
                        HumanMessage(content=message_content, id=message.message_id)
                    )
                else:
                    history.append(
                        AIMessage(content=message_content, id=message.message_id)
                    )
            else:
                content = []
                if message.content:
                    content.append(
                        {
                            "type": "text",
                            "text": message.content,
                        }
                    )

                for file_path in files:
                    local_path = file_path
                    if any(
                        local_path.endswith(suffix)
                        for suffix in SUPPORTED_IMAGE_EXTENSIONS
                    ):
                        base64_image = await self.store.get_image(local_path)

                        content.append(
                            {
                                "type": "text",
                                "text": f"![Image]({base64_image})",
                            }
                        )
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": base64_image,
                                },
                            }
                        )
                    else:
                        base64_document, _ = await self.store.get_document(local_path)
                        content.append(
                            {
                                "type": "text",
                                "text": f"source: {local_path}, content: {base64_document}",  # noqa: E501
                            },
                        )

                if message.role == Role.ASSISTANT:
                    history.append(AIMessage(content=content, id=message.message_id))
                else:
                    history.append(HumanMessage(content=content, id=message.message_id))

        return history

    async def _query_input_to_message(
        self, query_input: QueryInput, message_id: str | None = None
    ) -> HumanMessage:
        """将查询输入转换为消息。"""
        content = []

        if query_input.text:
            content.append(
                {
                    "type": "text",
                    "text": query_input.text,
                }
            )

        for image in query_input.images or []:
            local_path = image
            base64_image = await self.store.get_image(local_path)
            content.append(
                {
                    "type": "text",
                    "text": f"![Image]({base64_image})",
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image,
                    },
                }
            )

        for document in query_input.documents or []:
            local_path = document
            base64_document, _ = await self.store.get_document(local_path)
            content.append(
                {
                    "type": "text",
                    "text": f"source: {local_path}, content: {base64_document}",
                },
            )

        return HumanMessage(content=content, id=message_id)

    async def _get_history_user_input(
        self, chat_id: str, message_id: str
    ) -> BaseMessage:
        """从历史记录中获取最后一条用户输入消息。"""
        woodenfish_user: woodenfishUser = self.request_state.woodenfish_user
        async with self.app.db_sessionmaker() as session:
            db = self.app.msg_store(session)
            chat = await db.get_chat_with_messages(chat_id, woodenfish_user["user_id"])
            if chat is None:
                raise ChatError("chat not found")
            message = None
            for i in chat.messages:
                if i.role == Role.USER:
                    message = i
                if i.message_id == message_id:
                    break
            else:
                message = None
            if message is None:
                raise ChatError("message not found")

            return (
                await self._process_history_messages(
                    [message],
                    [],
                )
            )[0]


class LogStreamHandler:
    """处理日志流。"""

    def __init__(
        self,
        stream: EventStreamContextManager,
        log_manager: LogManager,
        stream_until: ClientState | None = None,
        stop_on_notfound: bool = True,
        max_retries: int = 10,
    ) -> None:
        """初始化日志处理器。"""
        self._stream = stream
        self._log_manager = log_manager
        self._end_event = asyncio.Event()
        self._stop_on_notfound = stop_on_notfound
        self._max_retries = max_retries

        self._stream_until: set[ClientState] = {
            ClientState.CLOSED,
            ClientState.FAILED,
        }
        if stream_until:
            self._stream_until.add(stream_until)

    async def _log_listener(self, msg: LogMsg) -> None:
        await self._stream.write(msg.model_dump_json())
        if msg.client_state in self._stream_until:
            self._end_event.set()

    async def stream_logs(self, server_name: str) -> None:
        """从特定 MCP 服务器流式传输日志。

        保持连接打开，直到客户端断开连接或达到客户端状态。

        如果 self._stop_on_notfound 为 False，它将继续重试，直到找到日志缓冲区或达到最大重试次数。
        """
        while self._max_retries > 0:
            self._max_retries -= 1

            try:
                async with self._log_manager.listen_log(
                    name=server_name,
                    listener=self._log_listener,
                ):
                    with suppress(asyncio.CancelledError):
                        await self._end_event.wait()
                        break
            except LogBufferNotFoundError as e:
                logger.warning(
                    "Log buffer not found for server %s, retries left: %d",
                    server_name,
                    self._max_retries,
                )

                msg = LogMsg(
                    event=LogEvent.STREAMING_ERROR,
                    body=f"Error streaming logs: {e}",
                    mcp_server_name=server_name,
                )
                await self._stream.write(msg.model_dump_json())

                if self._stop_on_notfound or self._max_retries == 0:
                    break

                await asyncio.sleep(1)

            except Exception as e:
                logger.exception("Error in log streaming for server %s", server_name)
                msg = LogMsg(
                    event=LogEvent.STREAMING_ERROR,
                    body=f"Error streaming logs: {e}",
                    mcp_server_name=server_name,
                )
                await self._stream.write(msg.model_dump_json())
                break


def strip_title(title: str) -> str:
    """剥离标题。"""
    title = re.sub(r"\s*<.+>.*?</.+>\s*", "", title, flags=re.DOTALL)
    return " ".join(title.split())
