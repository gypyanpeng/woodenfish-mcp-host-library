import logging
import os
from pathlib import Path

from pydantic import Field, RootModel

from woodenfish_mcp_host.httpd.conf.misc import WOODENFISH_CONFIG_DIR

logger = logging.getLogger(__name__)


class CommandAliasConfig(RootModel[dict[str, str]]):
    """命令别名配置模型。"""

    root: dict[str, str] = Field(default_factory=dict)


class CommandAliasManager:
    """命令别名管理器，用于处理配置。"""

    def __init__(self, config_path: str | None = None) -> None:
        """初始化 CommandAliasManager。

        参数：
            config_path: 可选的配置文件路径。
                如果未提供，则默认为当前工作目录下的 "config.json"。
        """
        self._config_path: str = config_path or str(
            WOODENFISH_CONFIG_DIR / "command_alias.json"
        )
        self._current_config: dict[str, str] | None = None

    @property
    def config_path(self) -> str:
        """获取配置路径。"""
        return self._config_path

    @property
    def current_config(self) -> dict[str, str] | None:
        """获取当前配置。"""
        return self._current_config

    def initialize(self) -> None:
        """初始化 CommandAliasManager。

        返回：
            成功返回 True，否则返回 False。
        """
        logger.info("Initializing CommandAliasManager from %s", self._config_path)
        env_config = os.environ.get("WOODENFISH_COMMAND_ALIAS_CONTENT")

        if env_config:
            config_content = env_config
        else:
            with Path(self._config_path).open(encoding="utf-8") as f:
                config_content = f.read()

        config_dict = CommandAliasConfig.model_validate_json(config_content)
        self._current_config = config_dict.root
