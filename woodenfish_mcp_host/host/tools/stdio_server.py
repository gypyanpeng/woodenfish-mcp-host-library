"""复制自 mcp.client.stdio.stdio_client。"""

import logging
import subprocess
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import anyio.abc
import anyio.lowlevel
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from anyio.streams.text import TextReceiveStream
from mcp import types
from mcp.client.stdio import StdioServerParameters, get_default_environment
from mcp.client.stdio.win32 import (
    get_windows_executable_command,
    terminate_windows_process,
)
from mcp.shared.message import SessionMessage

from woodenfish_mcp_host.host.tools.log import LogProxy

logger = logging.getLogger(__name__)

# 默认继承的环境变量
DEFAULT_INHERITED_ENV_VARS = (
    [
        "APPDATA",
        "HOMEDRIVE",
        "HOMEPATH",
        "LOCALAPPDATA",
        "PATH",
        "PROCESSOR_ARCHITECTURE",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "USERNAME",
        "USERPROFILE",
    ]
    if sys.platform == "win32"
    else ["HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER"]
)


@asynccontextmanager
async def stdio_client(  # noqa: C901, PLR0915
    server: StdioServerParameters,
    errlog: LogProxy,
) -> AsyncGenerator[
    tuple[
        MemoryObjectReceiveStream[types.JSONRPCMessage | Exception],
        MemoryObjectSendStream[types.JSONRPCMessage],
    ],
    None,
]:
    """Copy of mcp.client.stdio.stdio_client."""
    read_stream: MemoryObjectReceiveStream[types.JSONRPCMessage | Exception]
    read_stream_writer: MemoryObjectSendStream[types.JSONRPCMessage | Exception]

    write_stream: MemoryObjectSendStream[types.JSONRPCMessage]
    write_stream_reader: MemoryObjectReceiveStream[types.JSONRPCMessage]

    read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(0)

    command = _get_executable_command(server.command)

    # Open process with stderr piped for capture
    process = await _create_platform_compatible_process(
        command=command,
        args=server.args,
        env=(
            {**get_default_environment(), **server.env}
            if server.env is not None
            else get_default_environment()
        ),
        cwd=server.cwd,
    )

    async def stderr_reader() -> None:
        try:
            assert process.stderr, "Opened process is missing stderr"
            async for line in TextReceiveStream(
                process.stderr,
                encoding=server.encoding,
                errors=server.encoding_error_handler,
            ):
                await errlog.write(line)
                await errlog.flush()
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()
        finally:
            logger.debug("stderr_pipe closed")

    async def stdout_reader() -> None:
        assert process.stdout, "Opened process is missing stdout"

        try:
            async with read_stream_writer:
                buffer = ""
                async for chunk in TextReceiveStream(
                    process.stdout,
                    encoding=server.encoding,
                    errors=server.encoding_error_handler,
                ):
                    lines = (buffer + chunk).split("\n")
                    buffer = lines.pop()
                    for line in lines:
                        try:
                            parsed_msg = types.JSONRPCMessage.model_validate_json(line)
                        except Exception as exc:  # noqa: BLE001
                            logger.error("Error validating message: %s, %s", exc, line)
                            await read_stream_writer.send(exc)
                            continue

                        await read_stream_writer.send(SessionMessage(message=parsed_msg))
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()
        finally:
            logger.debug("stdout_reader closed")

    async def stdin_writer() -> None:
        assert process.stdin, "Opened process is missing stdin"

        try:
            async with write_stream_reader:
                async for message in write_stream_reader:
                    # 假设 Heartbeat 可能是一种 message.message 类型（JSONRPCMessage）
                    # 暂时假设 Heartbeat 检查在其他地方处理或不关键
                    json = message.message.model_dump_json(by_alias=True, exclude_none=True)
                    await process.stdin.send(
                        (json + "\n").encode(
                            encoding=server.encoding,
                            errors=server.encoding_error_handler,
                        )
                    )
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()
        finally:
            logger.debug("stdin_writer closed")

    async with (
        anyio.create_task_group() as tg,
        process,
    ):
        tg.start_soon(stdout_reader)
        tg.start_soon(stdin_writer)
        tg.start_soon(stderr_reader)
        try:
            yield read_stream, write_stream
        except Exception as exc:  # noqa: BLE001
            # 确保 exc_info=True 以记录完整的堆栈跟踪，特别是对于 ExceptionGroup
            logger.error(
                "Error during stdio_client operation or closing for process %s: %s",
                process.pid if process else "N/A",
                exc,
                exc_info=True,  # Crucial for ExceptionGroup details
            )
        finally:
            if process and hasattr(process, 'pid') and process.pid:  #  检查进程是否存在且有 pid
                logger.info("Terminated process %s", process.pid)
                if sys.platform == "win32":
                    await terminate_windows_process(process)
                else:
                    try:  # 为 kill 添加 try-except
                        process.kill()
                    except ProcessLookupError:
                        logger.warning("Process %s already terminated.", process.pid)
                    except Exception as kill_exc:
                        logger.error("Error killing process %s: %s", process.pid, kill_exc, exc_info=True)

                try:  # 为 wait 添加 try-except
                    status = await process.wait()
                    logger.info("Process %s exited with status %s", process.pid, status)
                except Exception as wait_exc:
                    logger.error("Error waiting for process %s: %s", process.pid, wait_exc, exc_info=True)
            else:
                logger.warning("Process was not available or PID not found in finally block.")
    # logger.error("Process %s closed", "xx") # Removed less informative log
    logger.info("Exiting stdio_client context manager.")


def _get_executable_command(command: str) -> str:
    """复制自 mcp.client.stdio._get_executable_command。"""
    if sys.platform == "win32":
        return get_windows_executable_command(command)
    return command


async def _create_platform_compatible_process(
    command: str,
    args: list[str],
    env: dict[str, str] | None = None,
    cwd: Path | str | None = None,
) -> anyio.abc.Process:
    """复制自 mcp.client.stdio._create_platform_compatible_process。"""
    if sys.platform == "win32" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        creationflags = subprocess.CREATE_NO_WINDOW
    else:
        creationflags = 0
    process = await anyio.open_process(
        [command, *args], creationflags=creationflags, env=env, cwd=cwd
    )
    logger.info("launched process: %s, pid: %s", command, process.pid)

    return process
