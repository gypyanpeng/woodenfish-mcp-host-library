from collections.abc import Callable, Coroutine, Mapping
from typing import Any

from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp


class KwargsMiddleware(BaseHTTPMiddleware):
    """向请求添加 kwargs 的中间件。"""

    kwargs_func: Mapping[
        str,
        Callable[[Request], Coroutine[Any, Any, Mapping[str, Any]]],
    ]

    def __init__(
        self,
        app: ASGIApp,
        kwargs_func: Mapping[
            str,
            Callable[[Request], Coroutine[Any, Any, Mapping[str, Any]]],
        ],
    ) -> None:
        """初始化中间件。"""
        self.kwargs_func = kwargs_func
        super().__init__(app, self.insert_kwargs)

    async def insert_kwargs(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """分派请求。"""

        async def get_kwargs(key: str) -> Mapping[str, Any]:
            if func := self.kwargs_func.get(key):
                return await func(request)
            return {}

        request.state.get_kwargs = get_kwargs
        return await call_next(request)
