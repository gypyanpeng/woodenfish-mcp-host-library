import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import delete, desc, exists, func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from woodenfish_mcp_host.httpd.database.models import (
    Chat,
    ChatMessage,
    Message,
    NewMessage,
    QueryInput,
    ResourceUsage,
    Role,
)
from woodenfish_mcp_host.httpd.database.orm_models import Chat as ORMChat
from woodenfish_mcp_host.httpd.database.orm_models import Message as ORMMessage
from woodenfish_mcp_host.httpd.database.orm_models import (
    ResourceUsage as ORMResourceUsage,
)
from woodenfish_mcp_host.httpd.routers.models import SortBy

from .abstract import AbstractMessageStore

if TYPE_CHECKING:
    from collections.abc import Sequence


class BaseMessageStore(AbstractMessageStore):
    """消息存储基类。

    包含可在 SQLite 和 PostgreSQL 中使用的查询
    """

    def __init__(self, session: AsyncSession) -> None:
        """初始化消息存储。

        参数：
            session: SQLAlchemy 异步会话。
        """
        self._session = session

    async def get_all_chats(
        self,
        user_id: str | None = None,
        sort_by: SortBy = SortBy.CHAT,
    ) -> list[Chat]:
        """从数据库中检索所有聊天。

        参数：
            user_id: 用户 ID 或指纹，取决于前缀。
            sort_by: 排序方式。
                - 'chat': 按聊天创建时间排序。
                - 'msg': 按消息创建时间排序。
                默认值: 'chat'

        返回：
            聊天对象列表。
        """
        if sort_by == SortBy.MESSAGE:
            query = (
                select(
                    ORMChat,
                    func.coalesce(
                        func.max(ORMMessage.created_at), ORMChat.created_at
                    ).label("last_message_at"),
                )
                .outerjoin(ORMMessage, ORMChat.id == ORMMessage.chat_id)
                .group_by(
                    ORMChat.id,
                    ORMChat.title,
                    ORMChat.created_at,
                    ORMChat.user_id,
                )
                .where(ORMChat.user_id == user_id)
                .order_by(desc("last_message_at"))
            )
            result = await self._session.execute(query)
            chats: Sequence[ORMChat] = result.scalars().all()

        elif sort_by == SortBy.CHAT:
            query = (
                select(ORMChat)
                .where(ORMChat.user_id == user_id)
                .order_by(desc(ORMChat.created_at))
            )
            result = await self._session.scalars(query)
            chats: Sequence[ORMChat] = result.all()

        else:
            raise ValueError(f"Invalid sort_by value: {sort_by}")

        return [
            Chat(
                id=chat.id,
                title=chat.title,
                createdAt=chat.created_at,
                user_id=chat.user_id,
            )
            for chat in chats
        ]

    async def get_chat_with_messages(
        self,
        chat_id: str,
        user_id: str | None = None,
    ) -> ChatMessage | None:
        """检索包含所有消息的聊天。

        参数：
            chat_id: 聊天的唯一标识符。
            user_id: 用户 ID 或指纹，取决于前缀。

        返回：
            ChatMessage 对象或 None（如果未找到）。
        """
        query = (
            select(ORMChat)
            .options(
                selectinload(ORMChat.messages).selectinload(ORMMessage.resource_usage),
            )
            .where(ORMChat.user_id == user_id)
            .where(ORMChat.id == chat_id)
            .order_by(ORMChat.created_at.desc())
        )
        data = await self._session.scalar(query)
        if data is None:
            return None

        chat = Chat(
            id=data.id,
            title=data.title,
            createdAt=data.created_at,
            user_id=data.user_id,
        )
        messages: list[Message] = []
        for msg in data.messages:
            resource_usage = (
                ResourceUsage.model_validate(
                    msg.resource_usage,
                    from_attributes=True,
                )
                if msg.resource_usage is not None
                else None
            )
            messages.append(
                Message(
                    id=msg.id,
                    createdAt=msg.created_at,
                    content=msg.content,
                    role=Role(msg.role),
                    chatId=msg.chat_id,
                    messageId=msg.message_id,
                    files=json.loads(msg.files) if msg.files else [],
                    tool_calls=msg.tool_calls or [],
                    resource_usage=resource_usage,
                ),
            )
        return ChatMessage(chat=chat, messages=messages)

    async def create_chat(
        self,
        chat_id: str,
        title: str,
        user_id: str | None = None,
        user_type: str | None = None,
    ) -> Chat | None:
        """创建一个新聊天。

        参数：
            chat_id: 聊天的唯一标识符。
            title: 聊天标题。
            user_id: 用户 ID 或指纹，取决于前缀。
            user_type: 可选的用户类型

        返回：
            创建的 Chat 对象或 None（如果创建失败）。
        """
        raise NotImplementedError(
            "The implementation of the method varies on different database.",
        )

    async def create_message(self, message: NewMessage) -> Message:
        """创建一个新消息。

        参数：
            message: 包含消息数据的 NewMessage 对象。

        返回：
            创建的 Message 对象。
        """
        query = (
            insert(ORMMessage)
            .values(
                {
                    "created_at": datetime.now(UTC),
                    "content": message.content,
                    "role": message.role,
                    "chat_id": message.chat_id,
                    "message_id": message.message_id,
                    "files": json.dumps(message.files),
                    "tool_calls": message.tool_calls,
                },
            )
            .returning(ORMMessage)
        )
        new_msg = await self._session.scalar(query)
        if new_msg is None:
            raise Exception(f"Create message failed: {message}")

        # NOTE: Only LLM messages will have resource usage
        new_resource_usage = None
        if message.role == Role.ASSISTANT and message.resource_usage is not None:
            query = (
                insert(ORMResourceUsage)
                .values(
                    {
                        "message_id": message.message_id,
                        "model": message.resource_usage.model,
                        "total_input_tokens": message.resource_usage.total_input_tokens,
                        "total_output_tokens": message.resource_usage.total_output_tokens,  # noqa: E501
                        "total_run_time": message.resource_usage.total_run_time,
                    },
                )
                .returning(ORMResourceUsage)
            )
            new_resource_usage = await self._session.scalar(query)
            if new_resource_usage is None:
                raise Exception(f"Create resource usage failed: {message}")

        resource_usage = (
            ResourceUsage.model_validate(
                new_resource_usage,
                from_attributes=True,
            )
            if new_resource_usage is not None
            else None
        )
        return Message(
            id=new_msg.id,
            createdAt=new_msg.created_at,
            content=new_msg.content,
            role=Role(new_msg.role),
            chatId=new_msg.chat_id,
            messageId=new_msg.message_id,
            files=json.loads(new_msg.files),
            tool_calls=new_msg.tool_calls or [],
            resource_usage=resource_usage,
        )

    async def check_chat_exists(
        self,
        chat_id: str,
        user_id: str | None = None,
    ) -> bool:
        """检查聊天是否存在于数据库中。"""
        query = (
            exists(ORMChat)
            .where(ORMChat.id == chat_id)
            .where(ORMChat.user_id == user_id)
            .select()
        )
        exist = await self._session.scalar(query)
        return bool(exist)

    async def delete_chat(self, chat_id: str, user_id: str | None = None) -> None:
        """从数据库中删除聊天。"""
        query = (
            delete(ORMChat)
            .where(ORMChat.id == chat_id)
            .where(ORMChat.user_id == user_id)
        )
        await self._session.execute(query)

    async def delete_messages_after(
        self,
        chat_id: str,
        message_id: str,
    ) -> None:
        """删除聊天中特定消息之后的所有消息。"""
        query = (
            delete(ORMMessage)
            .where(ORMMessage.chat_id == chat_id)
            .where(
                ORMMessage.created_at
                > (
                    select(ORMMessage.created_at)
                    .where(ORMMessage.chat_id == chat_id)
                    .where(ORMMessage.message_id == message_id)
                    .scalar_subquery()
                )
            )
        )
        await self._session.execute(query)

    async def update_message_content(
        self,
        message_id: str,
        data: QueryInput,
        user_id: str | None = None,
    ) -> Message:
        """更新消息的内容。"""
        # Prepare files list
        files = []
        if data.images:
            files.extend(data.images)
        if data.documents:
            files.extend(data.documents)

        # Update the message content and files with a single query
        query = (
            update(ORMMessage)
            .where(
                ORMMessage.message_id == message_id,
                ORMMessage.chat_id == ORMChat.id,
                ORMChat.user_id == user_id,
            )
            .values(
                content=data.text or "",
                files=json.dumps(files) if files else "",
                tool_calls=data.tool_calls,
            )
            .returning(ORMMessage)
            .options(selectinload(ORMMessage.resource_usage))
        )
        updated_message = await self._session.scalar(query)
        if updated_message is None:
            raise ValueError(f"Message {message_id} not found or access denied")

        resource_usage = (
            ResourceUsage.model_validate(
                updated_message.resource_usage,
                from_attributes=True,
            )
            if updated_message.resource_usage is not None
            else None
        )
        return Message(
            id=updated_message.id,
            createdAt=updated_message.created_at,
            content=updated_message.content,
            role=Role(updated_message.role),
            chatId=updated_message.chat_id,
            messageId=updated_message.message_id,
            files=json.loads(updated_message.files) if updated_message.files else [],
            tool_calls=updated_message.tool_calls or [],
            resource_usage=resource_usage,
        )

    async def get_next_ai_message(
        self,
        chat_id: str,
        message_id: str,
    ) -> Message:
        """获取特定消息之后的下一条 AI 消息。"""
        query = (
            select(ORMMessage)
            .where(ORMMessage.message_id == message_id)
            .where(ORMMessage.role == Role.USER)
        )
        user_message = await self._session.scalar(query)
        if user_message is None:
            raise ValueError("Can only get next AI message for user messages")

        query = (
            select(ORMMessage)
            .options(
                selectinload(ORMMessage.resource_usage),
            )
            .where(ORMMessage.chat_id == chat_id)
            .where(ORMMessage.id > user_message.id)
            .where(ORMMessage.role == Role.ASSISTANT)
            .limit(1)
        )
        message = await self._session.scalar(query)
        if not message:
            raise ValueError(
                f"No AI message found after user message ${message_id}."
                "This indicates a data integrity issue.",
            )

        resource_usage = (
            ResourceUsage.model_validate(
                message.resource_usage,
                from_attributes=True,
            )
            if message.resource_usage is not None
            else None
        )
        return Message(
            id=message.id,
            createdAt=message.created_at,
            content=message.content,
            role=Role(message.role),
            chatId=message.chat_id,
            messageId=message.message_id,
            files=json.loads(message.files) if message.files else [],
            tool_calls=message.tool_calls or [],
            resource_usage=resource_usage,
        )
