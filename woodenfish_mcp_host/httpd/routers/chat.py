from typing import TYPE_CHECKING, Annotated, TypeVar

from fastapi import APIRouter, Body, Depends, File, Form, Request, UploadFile
from fastapi.responses import StreamingResponse

from woodenfish_mcp_host.httpd.database.models import (
    Chat,
    ChatMessage,
    QueryInput,
)
from woodenfish_mcp_host.httpd.dependencies import get_app, get_woodenfish_user
from woodenfish_mcp_host.httpd.routers.models import (
    ResultResponse,
    SortBy,
    UserInputError,
)
from woodenfish_mcp_host.httpd.routers.utils import ChatProcessor, EventStreamContextManager
from woodenfish_mcp_host.httpd.server import woodenfishHostAPI

if TYPE_CHECKING:
    from woodenfish_mcp_host.httpd.middlewares.general import woodenfishUser

chat = APIRouter(tags=["chat"])

T = TypeVar("T")


class DataResult[T](ResultResponse):
    """扩展 ResultResponse，带有 data 字段的通用结果。"""

    data: T | None


@chat.get("/list")
async def list_chat(
    app: woodenfishHostAPI = Depends(get_app),
    woodenfish_user: "woodenfishUser" = Depends(get_woodenfish_user),
    sort_by: SortBy = SortBy.CHAT,
) -> DataResult[list[Chat]]:
    """列出所有可用聊天。

    参数：
        app (woodenfishHostAPI): woodenfishHostAPI 实例。
        woodenfish_user (woodenfishUser): woodenfishUser 实例。
        sort_by (SortBy):
            - 'chat': 按聊天创建时间排序。
            - 'msg': 按消息创建时间排序。
            默认值: 'chat'

    返回：
        DataResult[list[Chat]]: 可用聊天列表。
    """
    async with app.db_sessionmaker() as session:
        chats = await app.msg_store(session).get_all_chats(
            woodenfish_user["user_id"],
            sort_by=sort_by,
        )
    return DataResult(success=True, message=None, data=chats)


@chat.post("")
async def create_chat(  # noqa: PLR0913
    request: Request,
    app: woodenfishHostAPI = Depends(get_app),
    chat_id: Annotated[str | None, Form(alias="chatId")] = None,
    message: Annotated[str | None, Form()] = None,
    files: Annotated[list[UploadFile] | None, File()] = None,
    filepaths: Annotated[list[str] | None, Form()] = None,
) -> StreamingResponse:
    """创建新聊天。

    参数：
        request (Request): 请求对象。
        app (woodenfishHostAPI): woodenfishHostAPI 实例。
        chat_id (str | None): 要创建的聊天 ID。
        message (str | None): 要发送的消息。
        files (list[UploadFile] | None): 要上传的文件。
        filepaths (list[str] | None): 要上传的文件路径。
    """
    if files is None:
        files = []

    if filepaths is None:
        filepaths = []

    images, documents = await app.store.upload_files(files, filepaths)

    stream = EventStreamContextManager()
    response = stream.get_response()
    query_input = QueryInput(text=message, images=images, documents=documents)

    async def process() -> None:
        async with stream:
            processor = ChatProcessor(app, request.state, stream)
            await processor.handle_chat(chat_id, query_input, None)

    stream.add_task(process)
    return response


@chat.post("/edit")
async def edit_chat(  # noqa: PLR0913
    request: Request,
    app: woodenfishHostAPI = Depends(get_app),
    chat_id: Annotated[str | None, Form(alias="chatId")] = None,
    message_id: Annotated[str | None, Form(alias="messageId")] = None,
    content: Annotated[str | None, Form()] = None,
    files: Annotated[list[UploadFile] | None, File()] = None,
    filepaths: Annotated[list[str] | None, Form()] = None,
) -> StreamingResponse:
    """编辑聊天中的消息并再次查询。

    参数：
        request (Request): 请求对象。
        app (woodenfishHostAPI): woodenfishHostAPI 实例。
        chat_id (str | None): 要编辑的聊天 ID。
        message_id (str | None): 要编辑的消息 ID。
        content (str | None): 要发送的内容。
        files (list[UploadFile] | None): 要上传的文件。
        filepaths (list[str] | None): 要上传的文件路径。
    """
    if chat_id is None or message_id is None:
        raise UserInputError("Chat ID and Message ID are required")

    if files is None:
        files = []

    if filepaths is None:
        filepaths = []

    images, documents = await app.store.upload_files(files, filepaths)

    stream = EventStreamContextManager()
    response = stream.get_response()
    query_input = QueryInput(text=content, images=images, documents=documents)

    async def process() -> None:
        async with stream:
            processor = ChatProcessor(app, request.state, stream)
            await processor.handle_chat(chat_id, query_input, message_id)

    stream.add_task(process)
    return response


@chat.post("/retry")
async def retry_chat(
    request: Request,
    app: woodenfishHostAPI = Depends(get_app),
    chat_id: Annotated[str | None, Body(alias="chatId")] = None,
    message_id: Annotated[str | None, Body(alias="messageId")] = None,
) -> StreamingResponse:
    """重试聊天。

    参数：
        request (Request): 请求对象。
        app (woodenfishHostAPI): woodenfishHostAPI 实例。
        chat_id (str | None): 要重试的聊天 ID。
        message_id (str | None): 要重试的消息 ID。
    """
    if chat_id is None or message_id is None:
        raise UserInputError("Chat ID and Message ID are required")

    stream = EventStreamContextManager()
    response = stream.get_response()

    async def process() -> None:
        async with stream:
            processor = ChatProcessor(app, request.state, stream)
            await processor.handle_chat(chat_id, None, message_id)

    stream.add_task(process)
    return response


@chat.get("/{chat_id}")
async def get_chat(
    chat_id: str,
    app: woodenfishHostAPI = Depends(get_app),
    woodenfish_user: "woodenfishUser" = Depends(get_woodenfish_user),
) -> DataResult[ChatMessage]:
    """按 ID 获取特定聊天及其消息。

    参数：
        chat_id (str): 要检索的聊天 ID。
        app (woodenfishHostAPI): woodenfishHostAPI 实例。
        woodenfish_user (woodenfishUser): woodenfishUser 实例。

    返回：
        DataResult[ChatMessage]: 聊天及其消息。
    """
    async with app.db_sessionmaker() as session:
        chat = await app.msg_store(session).get_chat_with_messages(
            chat_id=chat_id,
            user_id=woodenfish_user["user_id"],
        )
    return DataResult(success=True, message=None, data=chat)


@chat.delete("/{chat_id}")
async def delete_chat(
    chat_id: str,
    app: woodenfishHostAPI = Depends(get_app),
    woodenfish_user: "woodenfishUser" = Depends(get_woodenfish_user),
) -> ResultResponse:
    """按 ID 删除特定聊天。

    参数：
        chat_id (str): 要删除的聊天 ID。
        app (woodenfishHostAPI): woodenfishHostAPI 实例。
        woodenfish_user (woodenfishUser): woodenfishUser 实例。

    返回：
        ResultResponse: 删除操作的结果。
    """
    async with app.db_sessionmaker() as session:
        await app.msg_store(session).delete_chat(
            chat_id=chat_id,
            user_id=woodenfish_user["user_id"],
        )
        await session.commit()
    await app.woodenfish_host["default"].delete_thread(chat_id)
    return ResultResponse(success=True, message=None)


@chat.post("/{chat_id}/abort")
async def abort_chat(
    chat_id: str,
    app: woodenfishHostAPI = Depends(get_app),
) -> ResultResponse:
    """中止正在进行的聊天操作。

    参数：
        chat_id (str): 要中止的聊天 ID。
        app (woodenfishHostAPI): woodenfishHostAPI 实例。

    返回：
        ResultResponse: 中止操作的结果。
    """
    abort_controller = app.abort_controller
    ok = await abort_controller.abort(chat_id)
    if not ok:
        raise UserInputError("Chat not found")

    return ResultResponse(success=True, message="Chat abort signal sent successfully")
