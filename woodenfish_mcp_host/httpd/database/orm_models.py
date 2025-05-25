from datetime import datetime

from langchain_core.messages import ToolCall
from sqlalchemy import (
    CHAR,
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB as PGJSONB
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON  # noqa: N811
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """所有 ORM 模型的基础类。"""


class Users(Base):
    """用户模型。

    属性：
        id: 用户 ID 或指纹，取决于前缀。
    """

    __tablename__ = "users"
    id: Mapped[str] = mapped_column(Text(), primary_key=True)
    user_type: Mapped[str | None] = mapped_column(CHAR(10))

    chats: Mapped[list["Chat"]] = relationship(
        back_populates="user",
        passive_deletes=True,
        uselist=True,
    )


# sqlite> PRAGMA table_info("chats");
# +-----+------------+------+---------+------------+----+
# | cid |    name    | type | notnull | dflt_value | pk |
# +-----+------------+------+---------+------------+----+
# | 0   | id         | TEXT | 1       |            | 1  |
# | 1   | title      | TEXT | 1       |            | 0  |
# | 2   | created_at | TEXT | 1       |            | 0  |
# +-----+------------+------+---------+------------+----+


class Chat(Base):
    """聊天模型。

    属性：
        id: 聊天 ID。
        title: 聊天标题。
        created_at: 聊天创建时间戳。
        user_id: 用户 ID 或指纹，取决于前缀。
    """

    __tablename__ = "chats"
    __table_args__ = (Index("idx_chats_user_id", "user_id", postgresql_using="hash"),)
    id: Mapped[str] = mapped_column(Text(), primary_key=True)
    title: Mapped[str] = mapped_column(Text())
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True).with_variant(Text(), "sqlite"),
    )
    user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="chat",
        passive_deletes=True,
        uselist=True,
    )
    user: Mapped["Users"] = relationship(
        foreign_keys=user_id,
        back_populates="chats",
        passive_deletes=True,
    )


# sqlite> PRAGMA table_info("messages");
# +-----+------------+---------+---------+------------+----+
# | cid |    name    |  type   | notnull | dflt_value | pk |
# +-----+------------+---------+---------+------------+----+
# | 0   | id         | INTEGER | 1       |            | 1  |
# | 1   | content    | TEXT    | 1       |            | 0  |
# | 2   | role       | TEXT    | 1       |            | 0  |
# | 3   | chat_id    | TEXT    | 1       |            | 0  |
# | 4   | message_id | TEXT    | 1       |            | 0  |
# | 5   | created_at | TEXT    | 1       |            | 0  |
# | 6   | files      | TEXT    | 1       |            | 0  |
# +-----+------------+---------+---------+------------+----+


class Message(Base):
    """消息模型。

    属性：
        id: 消息 ID。
        created_at: 消息创建时间戳。
        content: 消息内容。
        role: 消息角色。
        chat_id: 聊天 ID。
        message_id: 消息 ID。
        files: 消息文件。
        tool_calls: 消息工具调用。
    """

    __tablename__ = "messages"
    __table_args__ = (
        Index("messages_message_id_index", "message_id", postgresql_using="hash"),
        Index("idx_messages_chat_id", "chat_id", postgresql_using="hash"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer(), "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True).with_variant(Text(), "sqlite"),
    )
    content: Mapped[str] = mapped_column(Text())
    role: Mapped[str] = mapped_column(Text())
    chat_id: Mapped[str] = mapped_column(ForeignKey("chats.id", ondelete="CASCADE"))
    message_id: Mapped[str] = mapped_column(Text(), unique=True)
    files: Mapped[str] = mapped_column(Text())
    tool_calls: Mapped[list[ToolCall] | None] = mapped_column(
        PGJSONB().with_variant(SQLiteJSON(), "sqlite"), default=[]
    )

    chat: Mapped["Chat"] = relationship(
        foreign_keys=chat_id,
        back_populates="messages",
        passive_deletes=True,
    )
    resource_usage: Mapped["ResourceUsage"] = relationship(
        back_populates="message",
        passive_deletes=True,
    )


class ResourceUsage(Base):
    """资源使用模型。

    属性：
        id: 资源使用 ID。
        message_id: 消息 ID。
        model: 模型名称。
        total_input_tokens: 总输入令牌。
        total_output_tokens: 总输出令牌。
        total_run_time: 总运行时间。
    """

    __tablename__ = "resource_usage"
    __table_args__ = (Index("idx_resource_usage_message_id", "message_id"),)
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer(), "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.message_id", ondelete="CASCADE"),
    )
    model: Mapped[str] = mapped_column(Text())
    total_input_tokens: Mapped[int] = mapped_column(BigInteger())
    total_output_tokens: Mapped[int] = mapped_column(BigInteger())
    total_run_time: Mapped[float] = mapped_column(Float())

    message: Mapped["Message"] = relationship(
        foreign_keys=message_id,
        back_populates="resource_usage",
        passive_deletes=True,
    )
