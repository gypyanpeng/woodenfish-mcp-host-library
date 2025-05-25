from pathlib import Path
from typing import Annotated, Literal

from pydantic import AnyUrl, BaseModel, Field, SecretStr, UrlConstraints

from woodenfish_mcp_host.host.conf.llm import LLMConfigTypes


class CheckpointerConfig(BaseModel):
    """检查点管理器的配置。"""

    # more parameters in the future. like pool size, etc.
    uri: Annotated[
        AnyUrl,
        UrlConstraints(allowed_schemes=["sqlite", "postgres", "postgresql"]),
    ]


class ServerConfig(BaseModel):
    """MCP 服务器的配置。"""

    name: str
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
    exclude_tools: list[str] = Field(default_factory=list)
    url: str | None = None
    keep_alive: float | None = None
    transport: Literal["stdio", "sse", "websocket"]
    headers: dict[str, SecretStr] = Field(default_factory=dict)


class LogConfig(BaseModel):
    """MCP 服务器日志配置。

    属性：
        log_dir: 日志文件的基本目录。
        rotation_files: 每个 MCP 服务器的最大日志轮换文件数。
        buffer_length: 日志缓冲区中的日志条目数量。
    """

    log_dir: Path = Field(default_factory=lambda: Path.cwd() / "logs")
    rotation_files: int = 5
    buffer_length: int = 1000


class HostConfig(BaseModel):
    """MCP 主机的配置。"""

    llm: LLMConfigTypes
    checkpointer: CheckpointerConfig | None = None
    mcp_servers: dict[str, ServerConfig]
    log_config: LogConfig = Field(default_factory=LogConfig)


class AgentConfig(BaseModel):
    """MCP 代理的配置。"""

    model: str
