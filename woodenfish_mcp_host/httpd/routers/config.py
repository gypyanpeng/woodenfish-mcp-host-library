from logging import getLogger

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from woodenfish_mcp_host.httpd.conf.mcp_servers import Config
from woodenfish_mcp_host.httpd.dependencies import get_app
from woodenfish_mcp_host.httpd.server import woodenfishHostAPI

from .models import (
    McpServerError,
    McpServers,
    ModelFullConfigs,
    ModelInterfaceDefinition,
    ModelSettingsDefinition,
    ModelSingleConfig,
    ResultResponse,
)

logger = getLogger(__name__)

config = APIRouter(tags=["config"])


class ConfigResult[T](ResultResponse):
    """扩展 ResultResponse，带有 config 字段的通用配置结果。"""

    config: T | None


class SaveConfigResult(ResultResponse):
    """保存配置的结果，包括发生的任何错误。"""

    errors: list[McpServerError]


class InterfaceResult(ResultResponse):
    """包含模型接口定义的结果。"""

    interface: ModelInterfaceDefinition


class RulesResult(ResultResponse):
    """包含自定义规则作为字符串的结果。"""

    rules: str


class SaveModelSettingsRequest(BaseModel):
    """保存模型设置的请求模型。"""

    provider: str
    model_settings: ModelSingleConfig = Field(alias="modelSettings")
    enable_tools: bool = Field(alias="enableTools")


@config.get("/mcpserver")
async def get_mcp_server(
    app: woodenfishHostAPI = Depends(get_app),
) -> ConfigResult[McpServers]:
    """获取 MCP 服务器配置。

    返回：
        ConfigResult[McpServers]: MCP 服务器的配置。
    """
    if app.mcp_server_config_manager.current_config is None:
        logger.warning("MCP server configuration not found")
        return ConfigResult(
            success=True,
            config=McpServers(),
        )

    config = McpServers.model_validate(
        app.mcp_server_config_manager.current_config.model_dump(by_alias=True)
    )
    return ConfigResult(
        success=True,
        config=config,
    )


@config.post("/mcpserver")
async def post_mcp_server(
    servers: McpServers,
    app: woodenfishHostAPI = Depends(get_app),
    force: bool = False,
) -> SaveConfigResult:
    """保存 MCP 服务器配置。

    参数：
        servers (McpServers): 要保存的服务器配置。
        app (woodenfishHostAPI): woodenfishHostAPI 实例。
        force (bool): 如果为 True，即使未更改也重新加载所有 mcp 服务器。

    返回：
        SaveConfigResult: 保存操作的结果以及任何错误。
    """
    # Update conifg
    new_config = Config.model_validate(servers.model_dump(by_alias=True))
    if not app.mcp_server_config_manager.update_all_configs(new_config):
        raise ValueError("Failed to update MCP server configurations")

    # Reload host
    await app.woodenfish_host["default"].reload(
        new_config=app.load_host_config(), force_mcp=force
    )

    # Get failed MCP servers
    failed_servers: list[McpServerError] = []
    for server_name, server_info in app.woodenfish_host["default"].mcp_server_info.items():
        if server_info.error is not None:
            failed_servers.append(
                McpServerError(
                    serverName=server_name,
                    error=str(server_info.error),
                )
            )

    return SaveConfigResult(
        success=True,
        errors=failed_servers,
    )


@config.get("/model")
async def get_model(
    app: woodenfishHostAPI = Depends(get_app),
) -> ConfigResult["ModelFullConfigs"]:
    """获取当前模型配置。

    返回：
        ConfigResult[ModelConfig]: 当前模型配置。
    """
    if app.model_config_manager.full_config is None:
        logger.warning("Model configuration not found")
        return ConfigResult(
            success=True,
            config=None,
        )

    return ConfigResult(success=True, config=app.model_config_manager.full_config)


@config.post("/model")
async def post_model(
    model_settings: SaveModelSettingsRequest,
    app: woodenfishHostAPI = Depends(get_app),
) -> ResultResponse:
    """保存特定提供者的模型设置。

    参数：
        model_settings (SaveModelSettingsRequest): 要保存的模型设置。
        app (woodenfishHostAPI): woodenfishHostAPI 实例。

    返回：
        ResultResponse: 保存操作的结果。
    """
    app.model_config_manager.save_single_settings(
        provider=model_settings.provider,
        upload_model_settings=model_settings.model_settings,
        enable_tools=model_settings.enable_tools,
    )

    # Reload model config
    if not app.model_config_manager.initialize():
        raise ValueError("Failed to reload model configuration")

    # Reload host
    await app.woodenfish_host["default"].reload(new_config=app.load_host_config())

    return ResultResponse(success=True)


@config.post("/model/replaceAll")
async def post_model_replace_all(
    model_config: "ModelFullConfigs",
    app: woodenfishHostAPI = Depends(get_app),
) -> ResultResponse:
    """替换所有模型配置。

    参数：
        model_config (ModelConfig): 要使用的完整模型配置。
        app (woodenfishHostAPI): woodenfishHostAPI 实例。

    返回：
        ResultResponse: 替换操作的结果。
    """
    app.model_config_manager.replace_all_settings(model_config)
    if not app.model_config_manager.initialize():
        raise ValueError("Failed to reload model configuration")

    # Reload host
    await app.woodenfish_host["default"].reload(new_config=app.load_host_config())

    return ResultResponse(success=True)


@config.get("/model/interface")
async def get_model_interface() -> InterfaceResult:
    """获取模型接口定义。

    返回：
        InterfaceResult: 模型接口定义。
    """
    return InterfaceResult(
        success=True,
        interface=ModelInterfaceDefinition(
            model_settings={
                "modelProvider": ModelSettingsDefinition(
                    type="string",
                    description="The provider sdk of the model",
                    required=True,
                    default="",
                    placeholder="openai",
                ),
                "model": ModelSettingsDefinition(
                    type="string",
                    description="The model's name to use",
                    required=True,
                    default="gpt-4o-mini",
                ),
                "apiKey": ModelSettingsDefinition(
                    type="string",
                    description="The Model Provider API key",
                    required=False,
                    default="",
                    placeholder="YOUR_API_KEY",
                ),
                "baseURL": ModelSettingsDefinition(
                    type="string",
                    description="The model's base URL",
                    required=False,
                    default="",
                    placeholder="",
                ),
            },
        ),
    )


@config.get("/customrules")
async def get_custom_rules(app: woodenfishHostAPI = Depends(get_app)) -> RulesResult:
    """获取自定义规则配置。

    返回：
        RulesResult: 自定义规则作为字符串。
    """
    custom_rules = app.prompt_config_manager.load_custom_rules()
    return RulesResult(success=True, rules=custom_rules)


@config.post("/customrules")
async def post_custom_rules(
    request: Request,
    app: woodenfishHostAPI = Depends(get_app),
) -> ResultResponse:
    """保存自定义规则配置。

    返回：
        ResultResponse: 保存操作的结果。
    """
    raw_rules = await request.body()
    rules = raw_rules.decode("utf-8")
    app.prompt_config_manager.write_custom_rules(rules)
    app.prompt_config_manager.update_prompts()
    return ResultResponse(success=True)
