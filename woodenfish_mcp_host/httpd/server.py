import os
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import UTC, datetime
from logging import getLogger
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from woodenfish_mcp_host.host.conf import HostConfig, ServerConfig
from woodenfish_mcp_host.host.conf.llm import LLMConfig
from woodenfish_mcp_host.host.host import woodenfishMcpHost
from woodenfish_mcp_host.httpd.abort_controller import AbortController
from woodenfish_mcp_host.httpd.conf.command_alias import CommandAliasManager
from woodenfish_mcp_host.httpd.conf.httpd_service import ServiceManager
from woodenfish_mcp_host.httpd.conf.mcp_servers import MCPServerManager
from woodenfish_mcp_host.httpd.conf.models import ModelManager
from woodenfish_mcp_host.httpd.conf.prompt import PromptManager
from woodenfish_mcp_host.httpd.database.migrate import db_migration
from woodenfish_mcp_host.httpd.database.msg_store.base import BaseMessageStore
from woodenfish_mcp_host.httpd.database.msg_store.sqlite import SQLiteMessageStore
from woodenfish_mcp_host.httpd.store.cache import LocalFileCache
from woodenfish_mcp_host.httpd.store.local import LocalStore

logger = getLogger(__name__)


class Listen(BaseModel):
    """woodenfishHostAPI 的监听地址。"""

    ip: str | None = None
    port: int | None = None


class Server(BaseModel):
    """woodenfishHostAPI 的服务器信息。"""

    listen: Listen = Field(default_factory=Listen)


class Status(BaseModel):
    """woodenfishHostAPI 的状态。"""

    state: Literal["UP", "FAILED"]
    last_error: str | None = None
    error_code: str | None = None


class ReportStatus(BaseModel):
    """报告 woodenfishHostAPI 的状态。"""

    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    server: Server = Field(default_factory=Server)
    status: Status


class woodenfishHostAPI(FastAPI):
    """woodenfishHostAPI 是用于托管 woodenfishHost API 的 FastAPI 应用程序。"""

    woodenfish_host: dict[str, woodenfishMcpHost]  # 初始化时应包含 "default" 键

    def __init__(
        self,
        service_config_manager: ServiceManager,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """初始化 woodenfishHostAPI。"""
        super().__init__(*args, **kwargs)
        self._service_config_manager = service_config_manager

        self._listen_ip: str | None = None
        self._listen_port: int | None = None
        self._report_status_file: str | None = None
        self._report_status_fd: int | None = None

        self._engine: AsyncEngine | None = None

        if self._service_config_manager.current_setting is None:
            raise ValueError("Service manager is not initialized")

        self._mcp_server_config_manager = MCPServerManager(
            self._service_config_manager.current_setting.config_location.mcp_server_config_path
        )
        self._model_config_manager = ModelManager(
            self._service_config_manager.current_setting.config_location.model_config_path
        )
        self._prompt_config_manager = PromptManager(
            self._service_config_manager.current_setting.config_location.prompt_config_path
        )
        self._command_alias_config_manager = CommandAliasManager(
            self._service_config_manager.current_setting.config_location.command_alias_config_path
        )

        self._abort_controller = AbortController()

    def _load_configs(self) -> None:
        """加载所有配置。"""
        logger.info("Loading configs")
        self._mcp_server_config_manager.initialize()
        self._model_config_manager.initialize()
        self._prompt_config_manager.initialize()
        self._command_alias_config_manager.initialize()
        logger.info("Configs loaded")

    @asynccontextmanager
    async def prepare(self) -> AsyncGenerator[None, None]:
        """设置 woodenfishHostAPI。"""
        logger.info("Server Prepare")
        self._load_configs()

        # ================================================
        # 数据库
        # ================================================
        if self._service_config_manager.current_setting is None:
            raise ValueError("Service manager is not initialized")

        if self._service_config_manager.current_setting.db.migrate:
            db_migration(uri=self._service_config_manager.current_setting.db.uri)

        self._engine = create_async_engine(
            self._service_config_manager.current_setting.db.async_uri,
            echo=self._service_config_manager.current_setting.db.echo,
            # 使用前检查连接
            pool_pre_ping=self._service_config_manager.current_setting.db.pool_pre_ping,
            # 最大连接数
            pool_size=self._service_config_manager.current_setting.db.pool_size,
            # 60 秒后关闭连接
            pool_recycle=self._service_config_manager.current_setting.db.pool_recycle,
            # 突发连接数
            max_overflow=self._service_config_manager.current_setting.db.max_overflow,
        )
        self._db_sessionmaker = async_sessionmaker(self._engine, class_=AsyncSession)
        self._msg_store = SQLiteMessageStore

        # ================================================
        # 存储
        # ================================================
        self._store = LocalStore(
            root_dir=self._service_config_manager.current_setting.resource_dir
        )
        self._local_file_cache = LocalFileCache(
            root_dir=self._service_config_manager.current_setting.resource_dir
        )

        # ================================================
        # woodenfish Host
        # ================================================
        config = self.load_host_config()
        async with AsyncExitStack() as stack:
            default_host = woodenfishMcpHost(config)
            await stack.enter_async_context(default_host)
            self.woodenfish_host = {"default": default_host}

            logger.info("Server Prepare Complete")
            yield

    def load_host_config(self) -> HostConfig:
        """生成所有主机配置。"""
        model_setting = self._model_config_manager.current_setting
        if model_setting is None:
            model_setting = LLMConfig(
                model_provider="woodenfish",
                model="fake",
            )

        if self._service_config_manager.current_setting is None:
            raise ValueError("MCPServer config manager is not initialized")

        if self._command_alias_config_manager.current_config is None:
            raise ValueError("Command alias config manager is not initialized")

        mcp_servers: dict[str, ServerConfig] = {}
        for (
            server_name,
            server_config,
        ) in self._mcp_server_config_manager.get_enabled_servers().items():
            if not server_config.enabled:
                continue

            # 应用命令别名
            if server_config.command:
                command = self._command_alias_config_manager.current_config.get(
                    server_config.command, server_config.command
                )
            else:
                command = ""

            mcp_servers[server_name] = ServerConfig(
                name=server_name,
                command=command,
                args=server_config.args or [],
                env=server_config.env or {},
                enabled=server_config.enabled,
                url=server_config.url or None,
                transport=server_config.transport or "stdio",
                headers=server_config.headers or {},
            )

        return HostConfig(
            llm=model_setting,
            checkpointer=self._service_config_manager.current_setting.checkpointer,
            mcp_servers=mcp_servers,
            log_config=self._service_config_manager.current_setting.mcp_server_log,
        )

    async def ready(self) -> bool:
        """准备好 woodenfishHostAPI。"""
        try:
            # check db connection
            async with self._db_sessionmaker() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception:
            logger.exception("Server not ready")
            return False

    async def cleanup(self) -> None:
        """清理 woodenfishHostAPI。"""
        logger.info("Server Cleanup")
        if self._engine:
            await self._engine.dispose()
        logger.info("Server Cleanup Complete")

    def set_status_report_info(
        self,
        listen: str,
        report_status_file: str | None = None,
        report_status_fd: int | None = None,
    ) -> None:
        """设置状态报告信息。"""
        self._listen_ip = listen
        self._report_status_file = report_status_file
        self._report_status_fd = report_status_fd

    def set_listen_port(self, port: int) -> None:
        """设置监听端口。"""
        self._listen_port = port

    def report_status(self, error: str | None = None) -> None:
        """报告 woodenfishHostAPI 的状态。"""
        if error:
            msg = ReportStatus(
                status=Status(state="FAILED", last_error=error),
            )

        elif self._listen_ip and self._listen_port:
            msg = ReportStatus(
                server=Server(
                    listen=Listen(ip=self._listen_ip, port=self._listen_port)
                ),
                status=Status(state="UP"),
            )
        else:
            raise ValueError("Status info not complete")

        if self._report_status_file:
            logger.info("Report status to file: %s", self._report_status_file)
            with Path(self._report_status_file).open("w") as f:
                f.write(msg.model_dump_json())

        elif self._report_status_fd:
            logger.info("Report status to fd: %s", self._report_status_fd)
            os.write(
                self._report_status_fd,
                f"{msg.model_dump_json()}\r\n".encode(),
            )
        else:
            logger.info("No fd or file to report status")

    @property
    def db_sessionmaker(self) -> async_sessionmaker[AsyncSession]:
        """获取数据库 sessionmaker。"""
        return self._db_sessionmaker

    @property
    def msg_store(self) -> type[BaseMessageStore]:
        """获取消息存储类型。"""
        return self._msg_store

    @property
    def store(self) -> LocalStore:
        """获取存储。"""
        return self._store

    @property
    def local_file_cache(self) -> LocalFileCache:
        """获取本地文件缓存。"""
        return self._local_file_cache

    @property
    def service_manager(self) -> ServiceManager:
        """获取服务管理器。"""
        return self._service_config_manager

    @property
    def mcp_server_config_manager(self) -> MCPServerManager:
        """获取 MCP 服务器配置管理器。"""
        return self._mcp_server_config_manager

    @property
    def model_config_manager(self) -> ModelManager:
        """获取模型配置管理器。"""
        return self._model_config_manager

    @property
    def prompt_config_manager(self) -> PromptManager:
        """获取提示词配置管理器。"""
        return self._prompt_config_manager

    @property
    def abort_controller(self) -> AbortController:
        """获取中止控制器。"""
        return self._abort_controller
