"""一个用于测试的简单 echo mcp 服务器。"""

import asyncio
from argparse import ArgumentParser
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

Instructions = """回显消息。"""

mcp = FastMCP(name="echo", instructions=Instructions)

ECHO_DESCRIPTION = """一个简单的 echo 工具，用于验证 MCP 服务器是否正常工作。
它会返回包含输入消息的特征性响应。"""  # noqa: E501 描述过长

IGNORE_DESCRIPTION = """不执行任何操作。"""


@mcp.tool(
    name="echo",
    description=ECHO_DESCRIPTION,
)
async def echo(
    message: Annotated[str, Field(description="Message to be echoed back")],
    delay_ms: Annotated[
        int | None,
        Field(description="Optional delay in milliseconds before responding"),
    ] = None,
) -> str:
    """回显消息。"""
    if delay_ms and delay_ms > 0:
        await asyncio.sleep(delay_ms / 1000)
    return message


@mcp.tool(name="ignore", description=IGNORE_DESCRIPTION)
async def ignore(
    message: Annotated[str, Field(description="The message I should ignore.")],  # noqa: ARG001
) -> None:
    """不执行任何操作。"""
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--transport", type=str, default="stdio")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8800)

    args = parser.parse_args()
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "sse":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="sse")
    else:
        raise ValueError(f"Invalid transport: {args.transport}")
