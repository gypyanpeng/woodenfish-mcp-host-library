from argparse import ArgumentParser
from os import environ
from pathlib import Path
from typing import Annotated, Self

from pydantic import AfterValidator, BaseModel, Field, model_validator

from woodenfish_mcp_host.httpd.conf.misc import WOODENFISH_CONFIG_DIR

type StrPath = str | Path


def config_file(env_name: str, file_name: str) -> Path:
    """获取配置文件路径。"""
    return Path(environ.get(env_name, Path.cwd())).joinpath(file_name)


def _convert_path(x: str | None) -> StrPath | None:
    return Path(x) if x else x


class Arguments(BaseModel):
    """woodenfish_httpd 的命令行参数。"""

    httpd_config: Annotated[StrPath, AfterValidator(_convert_path)] = Field(
        alias="config",
        default="",
        description="主服务配置文件。",
    )
    llm_config: Annotated[StrPath, AfterValidator(_convert_path)] = Field(
        alias="model_config",
        default="",
        description="模型配置文件。",
    )

    mcp_config: Annotated[StrPath, AfterValidator(_convert_path)] = Field(
        description="MCP配置文件。",
        default="",
    )

    custom_rules: Annotated[StrPath | None, AfterValidator(_convert_path)] = Field(
        description="LLM的自定义规则。",
        default="",
    )

    command_alias_config: Annotated[StrPath | None, AfterValidator(_convert_path)] = (
        Field(
            description="命令别名的配置。",
            default="",
        )
    )

    listen: str = Field(
        default="127.0.0.1",
        description="绑定服务器到网络接口。",
    )

    port: int | None = Field(
        default=None,
        description="要监听的TCP端口号。使用0进行自动端口选择。",
    )

    auto_reload: bool = Field(
        default=False,
        description="自动重新加载配置文件，当检测到更改时。",
    )

    working_dir: Annotated[StrPath | None, AfterValidator(_convert_path)] = Field(
        default=None,
        description="服务器操作的基本目录。",
    )

    report_status_file: Annotated[StrPath | None, AfterValidator(_convert_path)] = (
        Field(
            default=None,
            description="写入服务器状态信息的文件路径。",
        )
    )

    report_status_fd: int | None = Field(
        default=None,
        description="写入服务器状态信息的文件描述符。",
    )

    auto_generate_configs: bool = Field(
        default=True,
        description="如果配置文件不存在，则自动生成配置文件。",
    )

    cors_origin: str | None = Field(
        default=None,
        description="允许的CORS来源。",
    )

    log_dir: Annotated[StrPath | None, AfterValidator(_convert_path)] = Field(
        default=None,
        description="写入日志文件的目录。",
    )

    log_level: str = Field(
        default="INFO",
        description="要使用的日志级别。",
    )

    @model_validator(mode="after")
    def rewrite_default_path(self) -> Self:
        """根据工作目录重写默认配置文件路径。"""
        cwd = Path(self.working_dir) if self.working_dir else WOODENFISH_CONFIG_DIR
        if not self.httpd_config:
            self.httpd_config = cwd.joinpath("woodenfish_httpd.json")
        if not self.llm_config:
            self.llm_config = cwd.joinpath("model_config.json")
        if not self.mcp_config:
            self.mcp_config = cwd.joinpath("mcp_config.json")
        if not self.command_alias_config:
            self.command_alias_config = cwd.joinpath("command_alias.json")
        if not self.log_dir:
            self.log_dir = cwd.joinpath("logs")
        return self

    @classmethod
    def parse_args(cls, args: list[str] | None = None) -> Self:
        """根据参数模型创建 argumentparser。"""
        parser = ArgumentParser(prog="woodenfish_httpd", exit_on_error=False)
        for name, field in cls.model_fields.items():
            kw = {}
            if field.is_required():
                kw["required"] = True
            arg_name = field.alias or name
            kw["help"] = field.description
            kw["default"] = field.get_default(call_default_factory=True)
            # Handle different field types for argument parsing
            # Convert field names with underscores to dashes for CLI arguments
            if field.annotation is int or field.annotation == int | None:
                kw["type"] = int
            elif field.annotation is bool:
                kw["action"] = (
                    "store_false" if kw.get("default", True) else "store_true"
                )
                if "default" in kw:
                    kw.pop("default")
            elif (
                field.annotation is str
                or field.annotation == StrPath
                or field.annotation == StrPath | None
                or field.annotation is Path
            ):
                kw["type"] = str
            parser.add_argument(f"--{arg_name}", dest=arg_name, **kw)

        return cls.model_validate(vars(parser.parse_args(args)))
