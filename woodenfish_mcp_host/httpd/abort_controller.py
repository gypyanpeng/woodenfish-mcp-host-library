from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager


class AbortController:
    """AbortController 是一个允许您中止任务的类。"""

    def __init__(self) -> None:
        """初始化 AbortController。"""
        self._mapped_events: dict[str, Callable[[], None]] = {}

    @asynccontextmanager
    async def abort_signal(
        self, key: str, func: Callable[[], None]
    ) -> AsyncGenerator[None, None]:
        """获取给定键的中止信号。"""
        self._mapped_events[key] = func
        try:
            yield
        finally:
            del self._mapped_events[key]

    async def abort(self, key: str) -> bool:
        """中止给定键的任务。"""
        if func := self._mapped_events.get(key):
            func()
            return True
        return False
