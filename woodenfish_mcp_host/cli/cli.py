"""woodenfish MCP Host 命令行工具。"""

import argparse
from pathlib import Path

from langchain_core.messages import HumanMessage

from woodenfish_mcp_host.cli.cli_types import CLIArgs
from woodenfish_mcp_host.host.conf import HostConfig
from woodenfish_mcp_host.host.host import woodenfishMcpHost


def parse_query(args: type[CLIArgs]) -> HumanMessage:
    """从命令行参数解析查询内容。"""
    query = " ".join(args.query)
    return HumanMessage(content=query)


def setup_argument_parser() -> type[CLIArgs]:
    """设置参数解析器。"""
    parser = argparse.ArgumentParser(description="woodenfish MCP Host 命令行工具")
    parser.add_argument(
        "query",
        nargs="*",
        default=[],
        help="The input query.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="The path to the configuration file.",
        dest="config_path",
    )
    parser.add_argument(
        "-c",
        type=str,
        default=None,
        help="Continue from given CHAT_ID.",
        dest="chat_id",
    )
    parser.add_argument(
        "-p",
        type=str,
        default=None,
        help="With given system prompt in the file.",
        dest="prompt_file",
    )
    return parser.parse_args(namespace=CLIArgs)


def load_config(config_path: str) -> HostConfig:
    """加载配置文件。"""
    with Path(config_path).open("r") as f:
        return HostConfig.model_validate_json(f.read())


async def run() -> None:
    """woodenfish_mcp_host 命令行入口。"""
    args = setup_argument_parser()
    query = parse_query(args)
    config = load_config(args.config_path)

    current_chat_id: str | None = args.chat_id
    system_prompt = None
    if args.prompt_file:
        with Path(args.prompt_file).open("r") as f:
            system_prompt = f.read()

    async with woodenfishMcpHost(config) as mcp_host:
        print("Waiting for tools to initialize...")
        await mcp_host.tools_initialized_event.wait()
        print("Tools initialized")
        chat = mcp_host.chat(chat_id=current_chat_id, system_prompt=system_prompt)
        current_chat_id = chat.chat_id
        async with chat:
            async for response in chat.query(query, stream_mode="messages"):
                print(response[0].content, end="")  # type: ignore

    print()
    print(f"Chat ID: {current_chat_id}")
