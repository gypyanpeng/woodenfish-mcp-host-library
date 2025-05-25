import logging
import os
from enum import StrEnum
from pathlib import Path

from woodenfish_mcp_host.httpd.conf.misc import WOODENFISH_CONFIG_DIR, write_then_replace
from woodenfish_mcp_host.httpd.conf.system_prompt import system_prompt

# Logger setup
logger = logging.getLogger(__name__)


class PromptKey(StrEnum):
    """提示词键枚举。"""

    SYSTEM = "system"
    CUSTOM = "custom"


class PromptManager:
    """提示词管理器，用于处理系统提示词和自定义规则。"""

    def __init__(self, custom_rules_path: str | None = None) -> None:
        """初始化 PromptManager。

        系统提示词的设置优先级如下：
        1. 如果存在环境变量 WOODENFISH_CUSTOM_RULES_CONTENT，则优先使用
        2. 如果设置了 custom_rules_path 参数，则使用该文件
        3. 如果当前工作目录下存在 .customrules 文件，则使用该文件
        4. 如果都没有，则默认为空字符串

        参数：
            custom_rules_path: 可选的自定义规则文件路径。
        """
        self.prompts: dict[str, str] = {}
        self.custom_rules_path = custom_rules_path or str(
            WOODENFISH_CONFIG_DIR / "custom_rules"
        )

    def initialize(self) -> None:
        """初始化 PromptManager。"""
        logger.info("Initializing PromptManager from %s", self.custom_rules_path)
        if custom_rules := os.environ.get("WOODENFISH_CUSTOM_RULES_CONTENT"):
            self.prompts[PromptKey.SYSTEM] = system_prompt(custom_rules)
            self.prompts[PromptKey.CUSTOM] = custom_rules
        elif (path := Path(self.custom_rules_path)) and path.exists():
            self.prompts[PromptKey.SYSTEM] = system_prompt(path.read_text("utf-8"))
            self.prompts[PromptKey.CUSTOM] = path.read_text("utf-8")
        else:
            self.prompts[PromptKey.SYSTEM] = system_prompt("")
            self.prompts[PromptKey.CUSTOM] = ""

    def set_prompt(self, key: str, prompt: str) -> None:
        """根据 key 设置提示词。

        参数：
            key: 存储提示词的键。
            prompt: 提示词内容。
        """
        self.prompts[key] = prompt

    def get_prompt(self, key: str) -> str | None:
        """根据 key 获取提示词。

        参数：
            key: 要获取的提示词键。

        返回：
            提示词内容，如果未找到则返回 None。
        """
        return self.prompts.get(key)

    def write_custom_rules(self, prompt: str) -> None:
        """将自定义规则写入文件。

        参数：
            prompt: 提示词内容。
        """
        write_then_replace(Path(self.custom_rules_path), prompt)

    def load_custom_rules(self) -> str:
        """从文件或环境变量加载自定义规则。

        返回：
            自定义规则内容。
        """
        try:
            return os.environ.get("WOODENFISH_CUSTOM_RULES_CONTENT") or Path(
                self.custom_rules_path
            ).read_text(encoding="utf-8")
        except OSError as error:
            logger.warning("Cannot read %s: %s", self.custom_rules_path, error)
            return ""

    def update_prompts(self) -> None:
        """用当前自定义规则更新系统提示词。"""
        custom_rules = self.load_custom_rules()
        self.prompts[PromptKey.SYSTEM] = system_prompt(custom_rules)
        self.prompts[PromptKey.CUSTOM] = custom_rules
