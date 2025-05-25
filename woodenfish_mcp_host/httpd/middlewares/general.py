from collections.abc import Callable
from typing import TypedDict

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from woodenfish_mcp_host.httpd.routers.models import ResultResponse, UserInputError


async def error_handler(_: Request, exc: Exception) -> Response:
    """错误处理中间件。

    参数：
        request (Request): 请求对象。
        exc (Exception): 要处理的异常。

    返回：
        ResultResponse: 响应对象。
    """
    msg = ResultResponse(success=False, message=str(exc)).model_dump(
        mode="json",
        by_alias=True,
    )

    if isinstance(exc, UserInputError):
        return JSONResponse(
            status_code=400,
            content=msg,
        )

    return JSONResponse(
        status_code=500,
        content=msg,
    )


class woodenfishUser(TypedDict):
    """与用户相关的状态存储。

    此状态可由所有中间件和处理程序访问。
    """

    user_id: str | None
    user_name: str | None
    user_type: str | None
    token_spent: int
    """用户在此期间花费的令牌数量。"""
    token_limit: int
    """用户在此期间可使用的令牌数量。"""
    token_increased: int
    """在此请求中增加的令牌数量。"""


async def default_state(request: Request, call_next: Callable) -> Response:
    """预填充默认状态。"""
    request.state.woodenfish_user = woodenfishUser(
        user_id=None,
        user_name=None,
        user_type=None,
        token_spent=0,
        token_limit=0,
        token_increased=0,
    )
    return await call_next(request)
