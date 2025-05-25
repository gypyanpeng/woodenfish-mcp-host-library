from enum import StrEnum
from logging import getLogger
from pathlib import Path

from woodenfish_mcp_host.httpd.conf.misc import RESOURCE_DIR

logger = getLogger(__name__)


class CacheKeys(StrEnum):
    """MCP 主机的缓存键。"""

    LIST_TOOLS = "list_tools"


class LocalFileCache:
    """MCP 主机的本地文件缓存。"""

    def __init__(
        self,
        root_dir: Path = RESOURCE_DIR,
        cache_file_prefix: str = "woodenfish_mcp_host",
    ) -> None:
        """初始化本地文件缓存。

        参数：
            root_dir: 配置的根目录
            cache_file_prefix: 缓存文件前缀。
        """
        self._cache_file_prefix = cache_file_prefix
        self._cache_dir = root_dir / "cache"

        logger.info("本地缓存目录: %s", self._cache_dir)
        logger.info("本地缓存文件前缀: %s", self._cache_file_prefix)
        self._cache_dir.mkdir(mode=0o744, parents=True, exist_ok=True)
        logger.info("本地缓存目录已准备好")

    def get_cache_file_path(self, key: CacheKeys, extension: str = "json") -> Path:
        """获取缓存文件路径。

        参数：
            key: 缓存键。
            extension: 缓存文件扩展名。

        返回：
            缓存文件路径。
        """
        return self._cache_dir / f"{self._cache_file_prefix}_{key.value}.{extension}"

    @property
    def cache_dir(self) -> Path:
        """获取缓存目录。"""
        return self._cache_dir

    def get(self, key: CacheKeys, extension: str = "json") -> str | None:
        """获取指定键的值。

        参数：
            key: 缓存键。
            extension: 缓存文件扩展名。

        返回：
            键对应的值。
        """
        cache_file_path = self.get_cache_file_path(key, extension)
        if not cache_file_path.exists():
            return None
        with cache_file_path.open("r") as f:
            return f.read()

    def set(self, key: CacheKeys, value: str, extension: str = "json") -> None:
        """设置指定键的值。

        参数：
            key: 缓存键。
            value: 要设置的值。
            extension: 缓存文件扩展名。

        返回：
            无
        """
        cache_file_path = self.get_cache_file_path(key, extension)
        with cache_file_path.open("w") as f:
            f.write(value)

    def delete(self, key: CacheKeys, extension: str = "json") -> None:
        """删除缓存文件。

        参数：
            key: 缓存键。
            extension: 缓存文件扩展名。

        返回：
            无
        """
        cache_file_path = self.get_cache_file_path(key, extension)
        if cache_file_path.exists():
            cache_file_path.unlink()
