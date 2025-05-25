from enum import StrEnum
from typing import Annotated, Any, Literal, Self, TypeVar

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    RootModel,
    SecretStr,
    field_serializer,
    model_validator,
)
from pydantic.alias_generators import to_camel

from woodenfish_mcp_host.host.conf.llm import (
    LLMConfigTypes,
    LLMConfiguration,
    get_llm_config_type,
)

T = TypeVar("T")


class ResultResponse(BaseModel):
    """带有成功状态和消息的通用响应模型。"""

    success: bool
    message: str | None = None


Transport = Literal["stdio", "sse", "websocket"]


class McpServerConfig(BaseModel):
    """带有传输和连接设置的 MCP 服务器配置。"""

    transport: Annotated[
        Transport, BeforeValidator(lambda v: "stdio" if v == "command" else v)
    ]
    enabled: bool | None
    command: str | None = None
    args: list[str] | None = Field(default_factory=list)
    env: dict[str, str] | None = Field(default_factory=dict)
    url: str | None = None
    headers: dict[str, SecretStr] | None = Field(default_factory=dict)

    def model_post_init(self, _: Any) -> None:
        """后初始化钩子。"""
        if self.transport in ["sse", "websocket"]:
            if self.url is None:
                raise ValueError("url is required for sse and websocket transport")
        elif self.transport == "stdio" and self.command is None:
            raise ValueError("command is required for stdio transport")

    @field_serializer("headers", when_used="json")
    def dump_headers(
        self, headers: dict[str, SecretStr] | None
    ) -> dict[str, str] | None:
        """将 headers 字段序列化为纯文本。"""
        if not headers:
            return None
        return {k: v.get_secret_value() for k, v in headers.items()}


class McpServers(BaseModel):
    """MCP 服务器配置集合。"""

    mcp_servers: dict[str, McpServerConfig] = Field(
        alias="mcpServers", default_factory=dict
    )


class McpServerError(BaseModel):
    """表示来自 MCP 服务器的错误。"""

    server_name: str = Field(alias="serverName")
    error: Any  # any


class ModelType(StrEnum):
    """模型类型。"""

    OLLAMA = "ollama"
    MISTRAL = "mistralai"
    BEDROCK = "bedrock"
    DEEPSEEK = "deepseek"
    OTHER = "other"

    @classmethod
    def get_model_type(cls, llm_config: LLMConfigTypes) -> "ModelType":
        """从模型名称获取模型类型。"""
        # Direct mapping for known providers
        try:
            return cls(llm_config.model_provider)
        except ValueError:
            pass
        # Special case for deepseek
        if "deepseek" in llm_config.model.lower():
            return cls.DEEPSEEK

        return cls.OTHER


class ModelSettingsProperty(BaseModel):
    """定义具有类型信息和元数据的模型设置属性。"""

    type: Literal["string", "number"]
    description: str
    required: bool
    default: Any | None = None
    placeholder: Any | None = None


class ModelSettingsDefinition(ModelSettingsProperty):
    """具有嵌套属性的模型设置定义。"""

    type: Literal["string", "number", "object"]
    properties: dict[str, ModelSettingsProperty] | None = None


class ModelInterfaceDefinition(BaseModel):
    """定义模型设置的接口。"""

    model_settings: dict[str, ModelSettingsDefinition]


class SimpleToolInfo(BaseModel):
    """表示具有属性和元数据的 MCP 工具。"""

    name: str
    description: str


class McpTool(BaseModel):
    """表示具有属性和元数据的 MCP 工具。"""

    name: str
    tools: list[SimpleToolInfo]
    description: str
    enabled: bool
    icon: str
    error: str | None = None


class ToolsCache(RootModel[dict[str, McpTool]]):
    """工具缓存。"""

    root: dict[str, McpTool]


class ToolCallsContent(BaseModel):
    """工具调用内容。"""

    name: str
    arguments: Any


class ToolResultContent(BaseModel):
    """工具结果内容。"""

    name: str
    result: Any


class ChatInfoContent(BaseModel):
    """聊天信息。"""

    id: str
    title: str


class MessageInfoContent(BaseModel):
    """消息信息。"""

    user_message_id: str = Field(alias="userMessageId")
    assistant_message_id: str = Field(alias="assistantMessageId")


class StreamMessage(BaseModel):
    """流消息。"""

    type: Literal[
        "text", "tool_calls", "tool_result", "error", "chat_info", "message_info"
    ]
    content: (
        str
        | list[ToolCallsContent]
        | ToolResultContent
        | ChatInfoContent
        | MessageInfoContent
    )


class TokenUsage(BaseModel):
    """令牌使用情况。"""

    total_input_tokens: int = Field(default=0, alias="totalInputTokens")
    total_output_tokens: int = Field(default=0, alias="totalOutputTokens")
    total_tokens: int = Field(default=0, alias="totalTokens")


class ModelSingleConfig(BaseModel):
    """模型单配置。"""

    model_provider: str
    model: str
    max_tokens: int | None = None
    api_key: SecretStr | None = None
    configuration: LLMConfiguration | None = None
    azure_endpoint: str | None = None
    azure_deployment: str | None = None
    api_version: str | None = None
    active: bool = Field(default=True)
    checked: bool = Field(default=False)
    tools_in_prompt: bool = Field(default=False)

    model_config = ConfigDict(
        alias_generator=to_camel,
        arbitrary_types_allowed=True,
        validate_by_name=True,
        validate_by_alias=True,
        extra="allow",
    )

    @model_validator(mode="after")
    def post_validate(self) -> Self:
        """通过转换为 LLMConfigTypes 验证模型配置。"""
        get_llm_config_type(self.model_provider).model_validate(self.model_dump())

        # ollama doesn't work well with normal bind tools
        if self.model_provider == "ollama":
            self.tools_in_prompt = True

        return self

    @field_serializer("api_key", when_used="json")
    def dump_api_key(self, v: SecretStr | None) -> str | None:
        """将 api_key 字段序列化为纯文本。"""
        return v.get_secret_value() if v else None


class ModelFullConfigs(BaseModel):
    """模型的配置。"""

    active_provider: str
    enable_tools: bool
    configs: dict[str, ModelSingleConfig] = Field(default_factory=dict)

    disable_woodenfish_system_prompt: bool = False
    # If True, custom rules will be used directly without extra system prompt from woodenfish.

    model_config = ConfigDict(
        alias_generator=to_camel,
        arbitrary_types_allowed=True,
        validate_by_name=True,
        validate_by_alias=True,
    )


class UserInputError(Exception):
    """用户输入错误。"""


class SortBy(StrEnum):
    """排序方式。"""

    CHAT = "chat"
    MESSAGE = "msg"
