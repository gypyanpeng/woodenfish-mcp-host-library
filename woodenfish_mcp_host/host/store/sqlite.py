"""langgraph-checkpoint-postgres 的 SQLite 版本。

使用 sqlite-vector-store 进行向量搜索。
"""

from langgraph.store.base.batch import AsyncBatchedBaseStore


class AsyncSQLiteStore(AsyncBatchedBaseStore):
    """异步 SQLite 存储。

    从 langgraph.store.sqlite 导入 SQLiteStore

    conn_string = "sqlite:///./db.sqlite"

    async with AsyncSQLiteStore.from_conn_string(conn_string) as store:
        await store.setup()

        # Store and retrieve data
        await store.aput(("users", "123"), "prefs", {"theme": "dark"})
        item = await store.aget(("users", "123"), "prefs")

    """
