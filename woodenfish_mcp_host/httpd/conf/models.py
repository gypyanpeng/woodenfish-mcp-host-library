import json
import logging
import os
from pathlib import Path

from pydantic import ValidationError

from woodenfish_mcp_host.host.conf.llm import LLMConfigTypes, get_llm_config_type
from woodenfish_mcp_host.httpd.conf.misc import WOODENFISH_CONFIG_DIR, write_then_replace
from woodenfish_mcp_host.httpd.routers.models import ModelFullConfigs, ModelSingleConfig

# Logger setup
logger = logging.getLogger(__name__)


class ModelManager:
    """模型管理器。"""

    def __init__(self, config_path: str | None = None) -> None:
        """初始化 ModelManager。

        参数：
            config_path: 可选的模型配置文件路径。
                如果未提供，则默认为当前工作目录下的 "modelConfig.json"。
        """
        self._config_path: str = config_path or str(
            WOODENFISH_CONFIG_DIR / "model_config.json"
        )
        self._current_setting: LLMConfigTypes | None = None
        self._full_config: ModelFullConfigs | None = None

    def initialize(self) -> bool:
        """初始化 ModelManager。"""
        logger.info("Initializing ModelManager from %s", self._config_path)
        if env_config := os.environ.get("WOODENFISH_MODEL_CONFIG_CONTENT"):
            config_content = env_config
        elif Path(self._config_path).exists():
            with Path(self._config_path).open(encoding="utf-8") as f:
                config_content = f.read()
        else:
            logger.warning("Model configuration not found")
            return False

        config_dict = json.loads(config_content)
        if not config_dict:
            logger.error("Model configuration not found")
            return False
        try:
            self._full_config = ModelFullConfigs.model_validate(config_dict)
            if model_config := (
                self._full_config.configs.get(self._full_config.active_provider)
            ):
                self._current_setting = get_llm_config_type(
                    model_config.model_provider
                ).model_validate(model_config.model_dump())
            else:
                self._current_setting = None
        except ValidationError as e:
            logger.error("Error parsing model settings: %s", e)
            return False

        return True

    @property
    def current_setting(self) -> LLMConfigTypes | None:
        """获取当前激活的模型设置。

        返回：
            如果未找到配置或激活提供者，则返回 None。
        """
        return self._current_setting

    @property
    def full_config(self) -> ModelFullConfigs | None:
        """获取完整的模型配置。

        返回：
            如果未找到配置，则返回 None。
        """
        return self._full_config

    @property
    def config_path(self) -> str:
        """获取配置路径。"""
        return self._config_path

    def get_settings_by_provider(self, provider: str) -> ModelSingleConfig | None:
        """根据提供者获取模型设置。

        参数：
            provider: 模型提供者名称。
        """
        if not self._full_config:
            return None
        return self._full_config.configs.get(provider, None)

    def save_single_settings(
        self,
        provider: str,
        upload_model_settings: ModelSingleConfig,
        enable_tools: bool = True,
    ) -> None:
        """保存单个模型配置。

        参数：
            provider: 模型提供者名称。
            upload_model_settings: 要上传的模型设置。
            enable_tools: 是否启用工具。
        """
        if not self._full_config:
            self._full_config = ModelFullConfigs(
                active_provider=provider,
                enable_tools=enable_tools,
                configs={provider: upload_model_settings},
            )
        else:
            self._full_config.active_provider = provider
            self._full_config.configs[provider] = upload_model_settings
            self._full_config.enable_tools = enable_tools

        write_then_replace(
            Path(self._config_path),
            self._full_config.model_dump_json(by_alias=True, exclude_none=True),
        )

    def replace_all_settings(
        self,
        upload_model_settings: ModelFullConfigs,
    ) -> None:
        """替换所有模型配置。

        参数：
            upload_model_settings: 要上传的模型设置。

        返回：
            成功返回 True。
        """
        self._full_config = upload_model_settings
        write_then_replace(
            Path(self._config_path),
            upload_model_settings.model_dump_json(by_alias=True, exclude_none=True),
        )
