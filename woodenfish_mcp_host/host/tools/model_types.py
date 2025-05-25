from enum import StrEnum


class ClientState(StrEnum):
    """客户端状态。

    状态和转换：
    """

    INIT = "init"
    RUNNING = "running"
    CLOSED = "closed"
    RESTARTING = "restarting"
    FAILED = "failed"
