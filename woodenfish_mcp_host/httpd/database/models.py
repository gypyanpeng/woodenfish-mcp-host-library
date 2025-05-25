from datetime import datetime
from enum import StrEnum

from langchain_core.messages import ToolCall
from pydantic import BaseModel, ConfigDict, Field


class ResourceUsage(BaseModel):
    """表示语言模型的用量统计信息。"""

    model: str
    total_input_tokens: int
    total_output_tokens: int
    total_run_time: float


class QueryInput(BaseModel):
    """带有文本、图片和文档的查询用户输入。"""

    text: str | None
    images: list[str] | None
    documents: list[str] | None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class Chat(BaseModel):
    """表示具有基本属性的聊天对话。"""

    id: str
    title: str
    created_at: datetime = Field(alias="createdAt")
    user_id: str | None


class Role(StrEnum):
    """消息的角色。"""

    ASSISTANT = "assistant"
    USER = "user"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class NewMessage(BaseModel):
    """表示聊天对话中的一条消息。"""

    content: str
    role: Role
    chat_id: str = Field(alias="chatId")
    message_id: str = Field(alias="messageId")
    resource_usage: ResourceUsage | None = None
    files: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list, alias="toolCalls")

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)


class Message(BaseModel):
    """表示聊天对话中的一条消息。"""

    id: int
    create_at: datetime = Field(alias="createdAt")
    content: str
    role: Role
    chat_id: str = Field(alias="chatId")
    message_id: str = Field(alias="messageId")
    resource_usage: ResourceUsage | None = None
    files: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list, alias="toolCalls")

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)


class ChatMessage(BaseModel):
    """将聊天与相关消息组合。"""

    chat: Chat
    messages: list[Message]
