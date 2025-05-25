from datetime import UTC, datetime

from sqlalchemy.dialects.sqlite import insert

from woodenfish_mcp_host.httpd.database.models import Chat
from woodenfish_mcp_host.httpd.database.msg_store.base import BaseMessageStore
from woodenfish_mcp_host.httpd.database.orm_models import Chat as ORMChat
from woodenfish_mcp_host.httpd.database.orm_models import Users as ORMUsers


class SQLiteMessageStore(BaseMessageStore):
    """SQLite 的消息存储。"""

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
        if user_id is not None:
            query = (
                insert(ORMUsers)
                .values(
                    {
                        "id": user_id,
                        "user_type": user_type,
                    }
                )
                .on_conflict_do_nothing()
            )
            await self._session.execute(query)

        query = (
            insert(ORMChat)
            .values(
                {
                    "id": chat_id,
                    "title": title,
                    "created_at": datetime.now(UTC),
                    "user_id": user_id,
                },
            )
            .on_conflict_do_nothing()
            .returning(ORMChat)
        )
        chat = await self._session.scalar(query)
        if chat is None:
            return None
        return Chat(
            id=chat.id,
            title=chat.title,
            createdAt=chat.created_at,
            user_id=chat.user_id,
        )
