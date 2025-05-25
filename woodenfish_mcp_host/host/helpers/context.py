from collections.abc import AsyncGenerator
from types import TracebackType
from typing import Protocol, Self


class ContextProtocol(Protocol):
    """表示一个异步上下文。

    ContextProtocol 定义了一个用于管理提供对话上下文的异步上下文管理器的接口。它确保了资源的正确管理并防止了重入使用。

    这个协议被需要提供对话上下文的类使用，例如数据库连接、模型实例或服务器连接。

    示例：
        ```python
        class DatabaseContext(ContextProtocol):
            async def _run_in_context(self) -> AsyncGenerator[Self, None]:
                # 设置数据库连接
                try:
                    await self.connect()
                    yield self
                finally:
                    await self.disconnect()

        async with DatabaseContext() as db:
            # 在这里使用 db
            pass
        ```
    """

    async def _run_in_context(self) -> AsyncGenerator[Self, None]: ...

    __gen: AsyncGenerator[Self, None] | None = None

    async def __aenter__(self) -> Self:
        if self.__gen is not None:
            raise RuntimeError("No Reentrant usage of ContextProtocol")
        self.__gen = self._run_in_context()
        return await anext(self.__gen)

    async def __aexit__(  # noqa: C901, PLR0912 A copy of asynccontextmanager.__aexit__.
        self,
        typ: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if self.__gen is None:
            raise RuntimeError("No context to exit")
        gen = self.__gen
        self.__gen = None
        if typ is None:
            try:
                await anext(gen)
            except StopAsyncIteration:
                return False
            else:
                try:
                    raise RuntimeError("generator didn't stop")
                finally:
                    await gen.aclose()
        else:
            if value is None:
                value = typ()
            try:
                await gen.athrow(value)
            except StopAsyncIteration as exc:
                return exc is not value
            except RuntimeError as exc:
                if exc is value:
                    exc.__traceback__ = traceback
                    return False
                if (
                    isinstance(value, StopIteration | StopAsyncIteration)
                    and exc.__cause__ is value
                ):
                    value.__traceback__ = traceback
                    return False
                raise
            except BaseException as exc:
                if exc is not value:
                    raise
                exc.__traceback__ = traceback
                return False
            try:
                raise RuntimeError("generator didn't stop after athrow()")
            finally:
                await gen.aclose()
