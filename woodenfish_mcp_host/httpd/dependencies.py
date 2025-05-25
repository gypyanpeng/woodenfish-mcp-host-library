"""MCP 主机的依赖项。"""

from typing import TYPE_CHECKING

from fastapi import Request

if TYPE_CHECKING:
    from woodenfish_mcp_host.httpd.middlewares.general import woodenfishUser
    from woodenfish_mcp_host.httpd.server import woodenfishHostAPI


def get_app(request: Request) -> "woodenfishHostAPI":
    """获取 woodenfishHostAPI 实例。"""
    return request.app


def get_woodenfish_user(
    request: Request,
) -> "woodenfishUser":
    """获取 woodenfishUser 实例。"""
    return request.state.woodenfish_user
