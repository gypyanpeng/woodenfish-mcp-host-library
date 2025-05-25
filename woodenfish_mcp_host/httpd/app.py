from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from logging import getLogger

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

from woodenfish_mcp_host.httpd.conf.httpd_service import ServiceManager
from woodenfish_mcp_host.httpd.middlewares import default_state, error_handler
from woodenfish_mcp_host.httpd.routers.chat import chat
from woodenfish_mcp_host.httpd.routers.config import config
from woodenfish_mcp_host.httpd.routers.model_verify import model_verify
from woodenfish_mcp_host.httpd.routers.openai import openai
from woodenfish_mcp_host.httpd.routers.tools import tools
from woodenfish_mcp_host.httpd.server import woodenfishHostAPI

logger = getLogger(__name__)


@asynccontextmanager
async def lifespan(app: woodenfishHostAPI) -> AsyncGenerator[None, None]:
    """FastAPI 应用的生命周期管理。"""
    try:
        async with app.prepare():
            app.report_status()
            yield
    except Exception as e:
        logger.exception("Error in lifespan")
        app.report_status(error=str(e))
        yield
    finally:
        await app.cleanup()


def create_app(
    service_config_manager: ServiceManager,
) -> woodenfishHostAPI:
    """创建 FastAPI 应用。"""
    app = woodenfishHostAPI(
        lifespan=lifespan,
        service_config_manager=service_config_manager,
    )
    app.add_exception_handler(Exception, error_handler)

    service_setting = service_config_manager.current_setting
    if service_setting and service_setting.cors_origin:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[service_setting.cors_origin],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=[
                "Origin",
                "X-Requested-With",
                "Content-Type",
                "Accept",
                "Authorization",
            ],
        )
    app.add_middleware(BaseHTTPMiddleware, dispatch=default_state)
    app.include_router(openai, prefix="/v1/openai")
    app.include_router(chat, prefix="/api/chat")
    app.include_router(tools, prefix="/api/tools")
    app.include_router(config, prefix="/api/config")
    app.include_router(model_verify, prefix="/model_verify")

    # remote endpoints
    app.include_router(chat, prefix="/api/v1/mcp")

    return app
