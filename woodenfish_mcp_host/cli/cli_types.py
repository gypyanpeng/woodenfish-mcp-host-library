"""woodenfish MCP Host 命令行参数类型。"""

from dataclasses import dataclass


@dataclass
class CLIArgs:
    """命令行参数。

    参数：
        chat_id: 要继续的线程ID。
        query: 输入的查询内容。
        config_path: 配置文件路径。
        prompt_file: 系统提示词文件路径。
    """

    chat_id: str | None
    query: list
    config_path: str
    prompt_file: str | None
