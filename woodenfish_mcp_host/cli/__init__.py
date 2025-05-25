"""woodenfish MCP Host CLI."""

import asyncio

from woodenfish_mcp_host.cli.cli import run


def main() -> None:
    """woodenfish_mcp_host CLI entrypoint."""
    asyncio.run(run())
