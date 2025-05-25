"""此模块包含主机的代理。

请参考 ChatAgentFactory 的设计来实现其他代理。
"""

from woodenfish_mcp_host.host.agents.agent_factory import AgentFactory, V
from woodenfish_mcp_host.host.agents.chat_agent import (
    ChatAgentFactory,
    get_chat_agent_factory,
)

__all__ = [
    "AgentFactory",
    "ChatAgentFactory",
    "V",
    "get_chat_agent_factory",
]
