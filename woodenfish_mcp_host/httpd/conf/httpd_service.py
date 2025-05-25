import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import make_url

from woodenfish_mcp_host.host.conf import CheckpointerConfig, LogConfig
from woodenfish_mcp_host.httpd.conf.arguments import StrPath
from woodenfish_mcp_host.httpd.conf.misc import WOODENFISH_CONFIG_DIR, RESOURCE_DIR

logger = logging.getLogger(__name__)


class DBConfig(BaseModel):
    """数据库配置。"""

    uri: str = Field(default="sqlite:///db.sqlite")
    pool_size: int = 5
    pool_recycle: int = 60
    max_overflow: int = 10
    echo: bool = False
    pool_pre_ping: bool = True
    migrate: bool = True

    @property
    def async_uri(self) -> str:
        """获取异步URI。"""
        url = make_url(self.uri)

        if url.get_backend_name() == "sqlite":
            url = url.set(drivername="sqlite+aiosqlite")
        elif url.get_backend_name() == "postgresql":
            url = url.set(drivername="postgresql+asyncpg")
        else:
            raise ValueError(f"Unsupported database: {url.get_backend_name()}")

        return str(url)


class ConfigLocation(BaseModel):
    """配置文件路径。"""

    mcp_server_config_path: str | None = None
    model_config_path: str | None = None
    prompt_config_path: str | None = None
    command_alias_config_path: str | None = None


class ServiceConfig(BaseModel):
    """服务配置。"""

    db: DBConfig = Field(default_factory=DBConfig)
    checkpointer: CheckpointerConfig
    resource_dir: Path = RESOURCE_DIR
    local_file_cache_prefix: str = "woodenfish_mcp_host"
    config_location: ConfigLocation = Field(default_factory=ConfigLocation)
    cors_origin: str | None = None
    mcp_server_log: LogConfig = Field(default_factory=LogConfig)

    logging_config: dict[str, Any] = {
        "disable_existing_loggers": False,
        "version": 1,
        "handlers": {
            "default": {"class": "logging.StreamHandler", "formatter": "default"}
        },
        "formatters": {
            "default": {
                "format": "%(levelname)s %(name)s:%(funcName)s:%(lineno)d :: %(message)s"  # noqa: E501
            }
        },
        "root": {"level": "INFO", "handlers": ["default"]},
        "loggers": {"woodenfish_mcp_host": {"level": "DEBUG"}},
    }


class ServiceManager:
    """服务管理器。"""

    def __init__(self, config_path: str | None = None) -> None:
        """初始化 ServiceManager。"""
        self._config_path: str = config_path or str(WOODENFISH_CONFIG_DIR / "woodenfish_httpd.json")
        self._current_setting: ServiceConfig | None = None

    def initialize(self) -> bool:
        """初始化 ServiceManager。"""
        # from env
        if env_config := os.environ.get("WOODENFISH_SERVICE_CONFIG_CONTENT"):
            config_content = env_config
        # from file
        else:
            with Path(self._config_path).open(encoding="utf-8") as f:
                config_content = f.read()

        if not config_content:
            logger.error("Service configuration not found")
            return False

        self._current_setting = ServiceConfig.model_validate_json(config_content)
        return True

    def overwrite_paths(
        self,
        config_location: ConfigLocation,
        resource_dir: Path = RESOURCE_DIR,
        log_dir: StrPath | None = None,
    ) -> None:
        """覆盖路径。"""
        if self._current_setting is None:
            raise ValueError("Service configuration not found")
        self._current_setting.config_location = config_location
        self._current_setting.resource_dir = resource_dir
        if log_dir:
            self._current_setting.mcp_server_log.log_dir = Path(log_dir)

    @property
    def current_setting(self) -> ServiceConfig | None:
        """获取当前配置。"""
        return self._current_setting

    @property
    def config_path(self) -> str:
        """获取配置路径。"""
        return self._config_path
