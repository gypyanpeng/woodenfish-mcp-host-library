"""主机的提示词模块。"""

import json
from collections.abc import Callable, Sequence

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import BaseTool

from woodenfish_mcp_host.host.helpers import today_datetime

PromptType = SystemMessage | str | Callable[..., list[BaseMessage]]

SYSTEM_PROMPT = """你是一名帮助软件工程师的 AI 助手。
你的用户是一位专业的软件工程师，参与各种编程项目。
今天的日期是 {today_datetime}。我的目标是提供清晰、准确、有帮助的回答，重点关注软件开发最佳实践。

我在沟通风格上应直接、技术化、实用。
在进行 git diff 操作时，请务必检查 README.md 文件，这样你可以更好地结合项目上下文理解变更。
"""


def default_system_prompt() -> str:
    """默认系统提示词。"""
    return SYSTEM_PROMPT.format(today_datetime=today_datetime())


def tools_definition(tools: Sequence[BaseTool]) -> str:
    """工具描述信息。"""
    return "\n".join(
        f"""
<tool>
  <name>{tool.name}</name>
  <description>{tool.description}</description>
  <arguments>{json.dumps(tool.args)}</arguments>
</tool>"""
        for tool in tools
    )


def tools_prompt(tools: Sequence[BaseTool]) -> str:
    """工具在提示词中的示例与说明。"""
    return f"""
## 工具使用示例
---
用户：埃菲尔铁塔是什么时候建成的？

助手：我可以使用搜索工具查找埃菲尔铁塔的信息。
<tool_call>
  <name>search</name>
  <arguments>{{\"query\": \"When was the Eiffel Tower built\"}}</arguments>
</tool_call>

ToolMessage: <tool_call_result>
  <name>search</name>
  <result>埃菲尔铁塔建于1887年至1889年，并于1889年3月31日正式开放。</result>
</tool_call_result>

助手：埃菲尔铁塔建于1887年至1889年，并于1889年3月31日正式开放。

---
用户：“85.50美元的15%加5美元手续费是多少？”

助手：我可以用计算器工具计算85.50美元的15%。
<tool_call>
  <name>calculator</name>
  <arguments>{{\"expression\": \"85.50 * 0.15\"}}</arguments>
</tool_call>

ToolMessage: <tool_call_result>
  <name>calculator</name>
  <result>12.83</result>
</tool_call_result>

助手：现在我加上5美元手续费。
<tool_call>
  <name>calculator</name>
  <arguments>{{\"expression\": \"12.83 + 5\"}}</arguments>
</tool_call>

ToolMessage: <tool_call_result>
  <name>calculator</name>
  <result>17.83</result>
</tool_call_result>

助手：85.50美元的15%是12.83美元，加上5美元手续费，总共是17.83美元。

---
用户：“东京的人口是多少，与纽约市相比如何？”

助手：我可以用搜索工具查找东京的人口。
<tool_call>
  <name>search</name>
  <arguments>{{\"query\": 123131}}</arguments>
</tool_call>

ToolMessage: <tool_call_failed>
  <name>search</name>
  <error>输入应为字符串</error>
</tool_call_failed>

助手：哦，我发现参数需要是字符串，让我再试一次。
<tool_call>
  <name>search</name>
  <arguments>{{\"query\": \"Population of Tokyo\"}}</arguments>
</tool_call>

ToolMessage: <tool_call_result>
  <name>search</name>
  <result>东京人口约为1396万（2023年估算）。</result>
</tool_call_result>

助手：我也可以用搜索工具查找纽约市的人口。
<tool_call>
  <name>search</name>
  <arguments>{{\"query\": \"Population of New York City\"}}</arguments>
</tool_call>

ToolMessage: <tool_call_result>
  <name>search</name>
  <result>纽约市人口约为880万（2022年估算）。</result>
</tool_call_result>

助手：东京人口约为1396万，而纽约市约为880万。东京比纽约市多约516万人。
---
## 可用工具
<tools>
{tools_definition(tools)}
</tools>

---
## 重要的工具使用规则
- 不要在回答中提及“工具使用示例”里的例子，这些只是帮助你理解如何用工具。
- 只要有机会就使用“可用工具”部分列出的工具。
- 实际调用工具时要返回 <tool_call> 标签。
- <tool_call> 的结构必须与“工具使用示例”完全一致，采用 XML 格式，包含 <name> 和 <arguments> 标签。
- 虽然 <tool_call> 标签本身是 XML 格式，但 <arguments> 标签内的内容是 JSON。
- 只能用 <tool_call>，其他与工具相关的标签不要使用。
- 所有 XML 标签都必须有闭合标签，例如：<tool_call><name>tool_name</name><arguments>{{}}</arguments></tool_call>。

请尽可能好地使用工具。
---
"""  # noqa: E501
