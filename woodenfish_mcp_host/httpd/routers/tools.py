from logging import getLogger

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from woodenfish_mcp_host.host.tools.model_types import ClientState
from woodenfish_mcp_host.httpd.dependencies import get_app
from woodenfish_mcp_host.httpd.routers.models import (
    McpTool,
    ResultResponse,
    SimpleToolInfo,
    ToolsCache,
)
from woodenfish_mcp_host.httpd.routers.utils import (
    EventStreamContextManager,
    LogStreamHandler,
)
from woodenfish_mcp_host.httpd.server import woodenfishHostAPI
from woodenfish_mcp_host.httpd.store.cache import CacheKeys

logger = getLogger(__name__)
tools = APIRouter(tags=["tools"])


class ToolsResult(ResultResponse):
    """列出可用 MCP 工具的响应模型。"""

    tools: list[McpTool]


@tools.get("/initialized")
async def initialized(
    app: woodenfishHostAPI = Depends(get_app),
) -> ResultResponse:
    """检查初始设置是否完成。

    仅在初次启动时有用，重新加载时无效。
    """
    await app.woodenfish_host["default"].tools_initialized_event.wait()
    return ResultResponse(success=True, message=None)


@tools.get("/")
async def list_tools(
    app: woodenfishHostAPI = Depends(get_app),
) -> ToolsResult:
    """列出所有可用的 MCP 工具。

    返回：
        ToolsResult: 可用工具的列表。
    """
    result: dict[str, McpTool] = {}

    # get full list of servers from config
    if app.mcp_server_config_manager.current_config is not None:
        all_servers = set(
            app.mcp_server_config_manager.current_config.mcp_servers.keys()
        )
    else:
        all_servers = set()

    # get tools from woodenfish host
    for server_name, server_info in app.woodenfish_host["default"].mcp_server_info.items():
        result[server_name] = McpTool(
            name=server_name,
            tools=[
                SimpleToolInfo(name=tool.name, description=tool.description or "")
                for tool in server_info.tools
            ],
            description="",
            enabled=True,
            icon="",
            error=server_info.error_str,
        )
    logger.debug("active mcp servers: %s", result.keys())

    # find missing servers
    missing_servers = all_servers - set(result.keys())
    logger.debug("disabled mcp servers: %s", missing_servers)

    # get missing from local cache
    if missing_servers:
        raw_cached_tools = app.local_file_cache.get(CacheKeys.LIST_TOOLS)
        cached_tools = ToolsCache(root={})
        if raw_cached_tools is not None:
            try:
                cached_tools = ToolsCache.model_validate_json(raw_cached_tools)
            except ValidationError as e:
                logger.warning(
                    "Failed to validate cached tools: %s %s", e, raw_cached_tools
                )
        for server_name in missing_servers:
            if server_info := cached_tools.root.get(server_name, None):
                server_info.enabled = False
                result[server_name] = server_info
            else:
                logger.warning("Server %s not found in cached tools", server_name)
                result[server_name] = McpTool(
                    name=server_name,
                    tools=[],
                    description="",
                    enabled=False,
                    icon="",
                    error=None,
                )

    # update local cache
    app.local_file_cache.set(
        CacheKeys.LIST_TOOLS,
        ToolsCache(root=result).model_dump_json(),
    )
    return ToolsResult(success=True, message=None, tools=list(result.values()))


@tools.get("/{server_name}/logs/stream")
async def stream_server_logs(
    server_name: str,
    stream_until: ClientState | None = None,
    stop_on_notfound: bool = True,
    max_retries: int = 10,
    app: woodenfishHostAPI = Depends(get_app),
) -> StreamingResponse:
    """从特定 MCP 服务器流式传输日志。

    参数：
        server_name (str): 要流式传输日志的 MCP 服务器名称。
        stream_until (ClientState | None): 流式传输直到达到客户端状态。
        stop_on_notfound (bool): 如果为 True，则如果找不到服务器，则停止流式传输。
        max_retries (int): 流式传输日志的最大重试次数。
        app (woodenfishHostAPI): woodenfishHostAPI 实例。

    返回：
        StreamingResponse: 服务器日志的流式响应。
        保持流式传输直到客户端断开连接。
    """
    log_manager = app.woodenfish_host["default"].log_manager
    stream = EventStreamContextManager()
    response = stream.get_response()

    async def process() -> None:
        async with stream:
            processor = LogStreamHandler(
                stream=stream,
                log_manager=log_manager,
                stream_until=stream_until,
                stop_on_notfound=stop_on_notfound,
                max_retries=max_retries,
            )
            await processor.stream_logs(server_name)

    stream.add_task(process)
    return response
