# new abstraction for database

from abc import ABC, abstractmethod

from woodenfish_mcp_host.httpd.database.models import (
    Chat,
    ChatMessage,
    Message,
    NewMessage,
    QueryInput,
)


class AbstractMessageStore(ABC):
    """数据库操作的抽象基类。"""

    @abstractmethod
    async def get_all_chats(
        self,
        user_id: str | None = None,
    ) -> list[Chat]:
        """从数据库中检索所有聊天。

        参数：
            user_id: 用户 ID 或指纹，取决于前缀。

        返回：
            聊天对象列表。
        """

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    async def create_message(
        self,
        message: NewMessage,
    ) -> Message:
        """创建一个新消息。

        参数：
            message: 包含消息数据的 NewMessage 对象。
            user_id: 用户 ID 或指纹，取决于前缀。

        返回：
            创建的 Message 对象。
        """

    @abstractmethod
    async def check_chat_exists(
        self,
        chat_id: str,
        user_id: str | None = None,
    ) -> bool:
        """检查聊天是否存在于数据库中。

        参数：
            chat_id: 聊天的唯一标识符。
            user_id: 用户 ID 或指纹，取决于前缀。

        返回：
            如果聊天存在则为 True，否则为 False。
        """

    @abstractmethod
    async def delete_chat(
        self,
        chat_id: str,
        user_id: str | None = None,
    ) -> None:
        """从数据库中删除聊天。

        参数：
            chat_id: 聊天的唯一标识符。
            user_id: 用户 ID 或指纹，取决于前缀。
        """

    @abstractmethod
    async def delete_messages_after(
        self,
        chat_id: str,
        message_id: str,
    ) -> None:
        """删除聊天中特定消息之后的所有消息。

        参数：
            chat_id: 聊天的唯一标识符。
            message_id: 将在其后删除所有消息的消息 ID。
            user_id: 用户 ID 或指纹，取决于前缀。
        """

    # NOTE: Might change, currently not used
    @abstractmethod
    async def update_message_content(
        self,
        message_id: str,
        data: QueryInput,
        user_id: str | None = None,
    ) -> Message:
        """更新消息的内容。

        参数：
            message_id: 消息的唯一标识符。
            data: 消息的新内容。
            user_id: 用户 ID 或指纹，取决于前缀。

        返回：
            更新的 Message 对象。
        """

    @abstractmethod
    async def get_next_ai_message(
        self,
        chat_id: str,
        message_id: str,
    ) -> Message:
        """获取特定消息之后的下一条 AI 消息。

        参数：
            chat_id: 聊天的唯一标识符。
            message_id: 将在其后查找下一条 AI 消息的消息 ID。

        返回：
            下一条 AI 消息对象。
        """
