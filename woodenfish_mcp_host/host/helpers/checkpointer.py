from contextlib import AbstractAsyncContextManager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


def get_checkpointer(
    uri: str,
) -> AbstractAsyncContextManager[AsyncSqliteSaver | AsyncPostgresSaver]:
    """根据数据库连接字符串获取合适的异步检查点管理器。

    参数：
        uri (str): 数据库连接字符串，以 'sqlite' 或 'postgres' 开头

    异常：
        ValueError: 如果连接字符串中的数据库类型不受支持

    返回：
        AsyncIterator[BaseCheckpointSaver[V]]: 指定数据库的异步检查点管理器实例
    """
    if uri.startswith("sqlite"):
        path = uri.removeprefix("sqlite:///")
        return AsyncSqliteSaver.from_conn_string(path)
    if uri.startswith("postgres"):
        return AsyncPostgresSaver.from_conn_string(uri)
    raise ValueError(f"Unsupported database: {uri}")
