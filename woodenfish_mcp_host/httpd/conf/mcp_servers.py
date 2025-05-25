import json
import logging
import os
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field, SecretStr, field_serializer

from woodenfish_mcp_host.httpd.conf.misc import WOODENFISH_CONFIG_DIR, write_then_replace


# Define necessary types for configuration
class MCPServerConfig(BaseModel):
    """MCP 服务器配置模型。"""

    transport: (
        Annotated[
            Literal["stdio", "sse", "websocket"],
            BeforeValidator(lambda v: "stdio" if v == "command" else v),
        ]
        | None
    ) = "stdio"
    enabled: bool = True
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None
    url: str | None = None
    headers: dict[str, SecretStr] | None = None

    @field_serializer("headers", when_used="json")
    def dump_headers(
        self, headers: dict[str, SecretStr] | None
    ) -> dict[str, str] | None:
        """将 headers 字段序列化为纯文本。"""
        if not headers:
            return None
        return {k: v.get_secret_value() for k, v in headers.items()}


class Config(BaseModel):
    """mcp_config.json 的模型。"""

    mcp_servers: dict[str, MCPServerConfig] = Field(alias="mcpServers")


# Logger setup
logger = logging.getLogger(__name__)


class MCPServerManager:
    """MCP 服务器管理器，用于处理配置。"""

    def __init__(self, config_path: str | None = None) -> None:
        """初始化 MCPServerManager。

        参数：
            config_path: 可选的配置文件路径。
                如果未提供，则默认为当前工作目录下的 "config.json"。
        """
        self._config_path: str = config_path or str(WOODENFISH_CONFIG_DIR / "mcp_config.json")
        self._current_config: Config | None = None

    @property
    def config_path(self) -> str:
        """获取配置路径。"""
        return self._config_path

    @property
    def current_config(self) -> Config | None:
        """获取当前配置。"""
        return self._current_config

    def initialize(self) -> None:
        """初始化 MCPServerManager。

        返回：
            成功返回 True，否则返回 False。
        """
        logger.info("Initializing MCPServerManager from %s", self._config_path)
        env_config = os.environ.get("WOODENFISH_MCP_CONFIG_CONTENT")

        if env_config:
            config_content = env_config
        elif Path(self._config_path).exists():
            with Path(self._config_path).open(encoding="utf-8") as f:
                config_content = f.read()
        else:
            logger.warning("MCP server configuration not found")
            return

        config_dict = json.loads(config_content)
        self._current_config = Config(**config_dict)

    def get_enabled_servers(self) -> dict[str, MCPServerConfig]:
        """获取已启用服务器的列表。

        返回：
            启用服务器名称及其配置的字典。
        """
        if not self._current_config:
            return {}

        return {
            server_name: config
            for server_name, config in self._current_config.mcp_servers.items()
            if config.enabled
        }

    def update_all_configs(self, new_config: Config) -> bool:
        """替换所有配置。

        参数：
            new_config: 新的配置。

        返回：
            成功返回 True，否则返回 False。
        """
        write_then_replace(
            Path(self._config_path),
            new_config.model_dump_json(by_alias=True),
        )

        self._current_config = new_config
        return True
