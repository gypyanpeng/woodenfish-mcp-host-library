from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_serializer,
    model_validator,
)
from pydantic.alias_generators import to_camel, to_snake

SpecialProvider = Literal["woodenfish", "__load__"]
"""特殊提供者：
- woodenfish：使用 woodenfish_mcp_host.models 中的模型
- __load__：从配置中加载模型
"""


def to_snake_dict(d: dict[str, str]) -> dict[str, str]:
    """将字典转换为蛇形命名法。"""
    return {to_snake(k): v for k, v in d.items()}


pydantic_model_config = ConfigDict(
    alias_generator=to_camel,
    validate_by_name=True,
    validate_assignment=True,
    validate_by_alias=True,
)


class Credentials(BaseModel):
    """LLM 模型的凭据。"""

    access_key_id: SecretStr = Field(default_factory=lambda: SecretStr(""))
    secret_access_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    session_token: SecretStr = Field(default_factory=lambda: SecretStr(""))
    credentials_profile_name: str = ""

    model_config = pydantic_model_config

    @field_serializer("access_key_id", when_used="json")
    def dump_access_key_id(self, v: SecretStr | None) -> str | None:
        """将 access_key_id 字段序列化为纯文本。"""
        return v.get_secret_value() if v else None

    @field_serializer("secret_access_key", when_used="json")
    def dump_secret_access_key(self, v: SecretStr | None) -> str | None:
        """将 secret_access_key 字段序列化为纯文本。"""
        return v.get_secret_value() if v else None

    @field_serializer("session_token", when_used="json")
    def dump_session_token(self, v: SecretStr | None) -> str | None:
        """将 session_token 字段序列化为纯文本。"""
        return v.get_secret_value() if v else None


class BaseLLMConfig(BaseModel):
    """LLM 模型的基本配置。"""

    model: str = "gpt-4o"
    model_provider: str | SpecialProvider = Field(default="openai")
    streaming: bool | None = True
    max_tokens: int | None = Field(default=None)
    tools_in_prompt: bool = Field(default=False)
    """教模型在提示词中使用工具。"""

    model_config = pydantic_model_config


class LLMConfiguration(BaseModel):
    """LLM 模型的配置。"""

    base_url: str | None = Field(default=None, alias="baseURL")
    temperature: float | None = Field(default=0)
    top_p: float | None = Field(default=None)

    model_config = pydantic_model_config

    def to_load_model_kwargs(self) -> dict:
        """将 LLM 配置转换为 load_model 的 kwargs。"""
        kwargs = {}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.temperature:
            kwargs["temperature"] = self.temperature
        if self.top_p:
            kwargs["top_p"] = self.top_p
        return kwargs


class LLMConfig(BaseLLMConfig):
    """通用 LLM 模型的配置。"""

    api_key: SecretStr | None = Field(default=None)
    configuration: LLMConfiguration | None = Field(default=None)

    model_config = pydantic_model_config

    def to_load_model_kwargs(self: LLMConfig) -> dict:
        """将 LLM 配置转换为 load_model 的 kwargs。"""
        exclude = {
            "configuration",
            "model_provider",
            "model",
            "streaming",
            "tools_in_prompt",
        }
        if self.model_provider == "anthropic" and self.max_tokens is None:
            exclude.add("max_tokens")
        kwargs = self.model_dump(
            exclude=exclude,
            exclude_none=True,
        )
        if self.configuration:
            kwargs.update(self.configuration.to_load_model_kwargs())
        remove_keys = []
        if self.model_provider == "openai" and self.model == "o3-mini":
            remove_keys.extend(["temperature", "top_p"])
        for key in remove_keys:
            kwargs.pop(key, None)
        return to_snake_dict(kwargs)

    @field_serializer("api_key", when_used="json")
    def dump_api_key(self, v: SecretStr | None) -> str | None:
        """将 api_key 字段序列化为纯文本。"""
        return v.get_secret_value() if v else None


class LLMBedrockConfig(BaseLLMConfig):
    """Bedrock LLM 模型的配置。"""

    model_provider: Literal["bedrock"] = "bedrock"
    region: str = "us-east-1"
    credentials: Credentials

    model_config = pydantic_model_config

    def to_load_model_kwargs(self) -> dict:
        """将 LLM 配置转换为 load_model 的 kwargs。"""
        model_kwargs = {}
        model_kwargs["aws_access_key_id"] = self.credentials.access_key_id
        model_kwargs["aws_secret_access_key"] = self.credentials.secret_access_key
        model_kwargs["credentials_profile_name"] = (
            self.credentials.credentials_profile_name
        )
        model_kwargs["aws_session_token"] = self.credentials.session_token
        model_kwargs["region_name"] = self.region
        model_kwargs["streaming"] = True if self.streaming is None else self.streaming
        return model_kwargs


class LLMAzureConfig(LLMConfig):
    """Azure LLM 模型的配置。"""

    model_provider: Literal["azure_openai"] = "azure_openai"
    api_version: str
    azure_endpoint: str
    azure_deployment: str
    configuration: LLMConfiguration | None = Field(default=None)

    model_config = pydantic_model_config

    def to_load_model_kwargs(self) -> dict:
        """将 LLM 配置转换为 load_model 的 kwargs。

        忽略 LLMConfig 中的 base_url。
        """
        kwargs = super().to_load_model_kwargs()
        if "base_url" in kwargs:
            del kwargs["base_url"]
        return kwargs


class LLMAnthropicConfig(LLMConfig):
    """Anthropic 模型的配置。"""

    model_provider: Literal["anthropic"] = "anthropic"
    max_tokens: int | None = None
    """Anthropic Claude 3.x 的内容窗口。"""
    default_headers: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def update_max_tokens(self) -> Self:
        """更新大令牌的默认请求头。"""
        if self.max_tokens is None:
            if self.model.startswith("claude-3-7"):
                self.max_tokens = 128000
            elif self.model.startswith("claude-3-5"):
                self.max_tokens = 8129
            else:
                self.max_tokens = 4096
        if self.max_tokens > 64000 and "anthropic-beta" not in self.default_headers:  # noqa: PLR2004
            self.default_headers["anthropic-beta"] = "output-128k-2025-02-19"
        return self


type LLMConfigTypes = Annotated[
    LLMAnthropicConfig | LLMAzureConfig | LLMBedrockConfig | LLMConfig,
    Field(union_mode="left_to_right"),
]


model_provider_map: dict[str, type[LLMConfigTypes]] = {
    "anthropic": LLMAnthropicConfig,
    "azure_openai": LLMAzureConfig,
    "bedrock": LLMBedrockConfig,
}


def get_llm_config_type(model_provider: str) -> type[LLMConfigTypes]:
    """获取给定模型提供者的模型配置。"""
    return model_provider_map.get(model_provider, LLMConfig)
