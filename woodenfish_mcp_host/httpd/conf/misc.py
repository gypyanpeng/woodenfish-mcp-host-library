from os import getenv
from pathlib import Path

RESOURCE_DIR = Path(getenv("RESOURCE_DIR", Path.cwd()))
WOODENFISH_CONFIG_DIR = Path(getenv("WOODENFISH_CONFIG_DIR", Path.cwd()))


def write_then_replace(path: Path, content: str) -> None:
    """将内容写入临时文件，然后替换目标文件。"""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        f.write(content)
    tmp_path.replace(path)
