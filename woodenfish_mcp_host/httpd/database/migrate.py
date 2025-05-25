"""数据库迁移脚本。"""

from logging import getLogger

from alembic import command
from alembic.config import Config

logger = getLogger(__name__)


def db_migration(
    uri: str,
    migrations_dir: str = "woodenfish_mcp_host:httpd/database/migrations",
) -> Config:
    """运行数据库迁移。

    参数：
        uri: 数据库 URI。
        migrations_dir: 迁移目录。

    返回：
        Alembic 配置。
    """
    logger.info("running migration")
    config = Config()
    config.set_main_option("script_location", migrations_dir)
    config.set_main_option("sqlalchemy.url", uri)
    command.upgrade(config, "head")
    logger.info("finish migration")
    return config
