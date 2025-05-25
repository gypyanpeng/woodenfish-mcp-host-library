from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from woodenfish_mcp_host.httpd.database.orm_models import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.

DATABASE_URL = (
    config.get_main_option("sqlalchemy.url")
    if config.get_main_option("sqlalchemy.url")
    else "sqlite:///dummy.db"
)


def run_migrations_offline() -> None:
    """在“离线”模式下运行迁移。

    这将仅使用 URL 配置上下文，
    而不是 Engine，尽管此处也接受 Engine。
    通过跳过 Engine 创建，我们甚至不需要可用的 DBAPI。

    此处对 context.execute() 的调用会将给定字符串输出到脚本输出。

    """
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在“在线”模式下运行迁移。

    在这种情况下，我们需要创建 Engine
    并将连接与上下文关联。

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        url=DATABASE_URL,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
