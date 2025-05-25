# woodenfish-mcp-host 核心库

此仓库包含 woodenfish 项目的后端核心库，即模型上下文协议 (MCP) 主机。它包括了智能体逻辑、工具管理、模型加载和会话管理等核心功能。

这个库旨在集成到其他 Python 应用中（例如，定制的 HTTP 服务、CLI 工具、桌面应用等），以提供 woodenfish 的核心 AI 能力。

## 1. 库的功能

`woodenfish-mcp-host` 库提供以下核心功能：

-   **智能体 (Intelligent Agent)**: 使用 **LangGraph** 管理 AI 的推理过程，包括多轮对话和自动化工具调用。
-   **工具管理 (Tool Management)**: 加载、管理并与各种外部 MCP 工具（脚本或进程）对接，这些工具扩展了 AI 的能力（例如，文件访问、网页搜索、计算等）。
-   **模型加载 (Model Loading)**: 根据配置处理不同大型语言模型 (LLM) 的加载和管理。
-   **会话管理 (Chat Session Management)**: 管理单独的聊天会话，维护对话历史和智能体状态。
-   **配置管理 (Configuration Management)**: 处理 LLM 和工具配置的加载和应用。

这个库**不包含**内置的 Web 服务器、图形界面或 CLI 入口（尽管它为这些提供了基础构建块）。集成此库的开发者负责构建自己的界面和服务层。

## 2. 安装

要在你的 Python 项目中使用此库，你可以使用 pip 从本地目录直接安装：

1.  导航到你克隆或放置 `woodenfish-mcp-host-library` 文件夹的目录。
2.  运行以下命令：

    ```bash
    pip install ./woodenfish-mcp-host-library
    ```

此命令将安装 `woodenfish_mcp_host` 包及其依赖项（在 `pyproject.toml` 中指定）到你的 Python 环境中。`./` 表示 pip 应从本地目录安装。

如果你希望以可编辑模式安装以便开发（库代码的更改无需重新安装即可立即反映），请使用 `-e` 标志：

```bash
pip install -e ./woodenfish-mcp-host-library
```

**重要提示**：

-   这个库的核心功能依赖于 **LangChain** 和 **LangGraph**。如果你想深入理解其内部工作原理，建议查阅这两个库的官方文档。
-   此库使用了 Python 的**异步编程 (asyncio)**。这意味着调用库中大部分核心方法时，你需要在 `async def` 函数中使用 `await` 关键字，并在程序的入口使用 `asyncio.run()` 来运行异步主函数。如果你不熟悉异步编程，建议先学习相关基础知识。

## 3. 配置

`mcp-host` 库的行为由一个配置对象（`HostConfig`）控制。在初始化核心 `woodenfishMcpHost` 类时，你需要提供此配置。

我们提供了示例配置文件：

-   `model_config.example.json`: 大模型 (LLM) 的配置示例。
-   `mcp_config.example.json`: MCP 工具的配置示例。

**使用这些示例**：

1.  将示例文件复制到你项目的配置目录中，并重命名（例如，`model_config.json`, `mcp_config.json`）。
2.  编辑复制的配置文件以匹配你的环境和需求（例如，添加你的 LLM API 密钥、指定本地模型路径、定义你的 MCP 工具）。
3.  在你的应用中加载这些配置文件，并用它们创建一个 `HostConfig` 对象。

配置加载的具体实现取决于你的项目结构和偏好（例如，从 JSON、YAML 文件或环境变量加载）。请参考下面的使用示例，它包含一个加载示例配置的简单实现。

## 3.1 模型配置 (model_config.example.json)

`model_config.example.json` 文件用于配置库可以使用的大型语言模型 (LLM)。它定义了不同的模型提供商和模型设置。

以下是 `model_config.example.json` 的示例内容和详细注释：

```json
{
  "activeProvider": "ollama", // 【必填】当前激活的模型提供商的名称。必须是 "configs" 中定义的一个 key。
  "configs": { // 【必填】一个字典，包含了各种模型提供商的具体配置。
    "ollama": { // 以 ollama 为例，这是一个具体的模型提供商配置。
      "model": "qwen3:1.7b", // 【必填】模型名称。具体名称取决于你使用的提供商和安装的模型。
      "model_provider": "ollama", // 【必填】模型提供商的内部标识符，用于库识别如何与该模型交互。
      "base_url": "http://localhost:11434", // 【选填】模型服务的基地址。本地模型或某些 API 可能需要指定。
      "streaming": true, // 【选填】是否启用流式传输。设置为 true 可以逐字接收 AI 回复，提供更好的用户体验。
      "max_tokens": 4096, // 【选填】模型生成回复的最大 token 数。
      "tools_in_prompt": true, // 【选填】是否将工具 schema 包含在发送给模型的 prompt 中。对于支持 Function Calling 的模型通常设为 true。
      "configuration": { // 【选填】一个字典，包含传递给底层 LangChain 模型对象的额外配置参数。具体参数取决于 model_provider。
        "temperature": 0.6 // 例如，ollama 提供商的 temperature 参数。
      }
    },
    "zhipu_glm4": { // 以智谱 GLM4 为例
      "model": "GLM-4-Flash", // 【必填】模型名称
      "model_provider": "openai_compatible", // 【必填】如果模型提供商兼容 OpenAI API，使用此标识符。
      "streaming": true, // 【选填】是否启用流式传输。
      "max_tokens": 4096, // 【选填】最大 token 数。
      "tools_in_prompt": false, // 【选填】是否将工具 schema 包含在 prompt 中。智谱 GLM-4-Flash 支持工具调用，这里设为 false 可能是因为采用其他方式传递工具信息或是一个示例。
      "api_key": "your api key", // 【必填】调用该模型所需的 API 密钥。请替换为你的真实密钥。
      "configuration": { // 【选填】额外配置参数。
        "base_url": "https://open.bigmodel.cn/api/paas/v4/", // 智谱的模型 API 地址。
        "temperature": 0.6 // 温度参数。
      },
      "default_headers": {} // 【选填】调用 API 时需要添加的额外 HTTP 头。
    },
    "openrouter": { // 以 OpenRouter 为例，它聚合了多种模型。
      "model": "deepseek/deepseek-r1:free", // 【必填】在 OpenRouter 上选择的模型名称。
      "model_provider": "openai_compatible", // 【必填】OpenRouter 兼容 OpenAI API。
      "streaming": true, // 【选填】是否启用流式传输。
      "max_tokens": 4096, // 【选填】最大 token 数。
      "tools_in_prompt": false, // 【选填】是否将工具 schema 包含在 prompt 中。
      "api_key": "your api key", // 【必填】OpenRouter API 密钥。
      "configuration": { // 【选填】额外配置参数。
        "base_url": "https://openrouter.ai/api/v1/", // OpenRouter API 地址。
        "temperature": 0.6 // 温度参数。
      },
      "default_headers": {} // 【选填】额外 HTTP 头。
    },
    "modelscope": { // 以 ModelScope 为例
      "model": "Qwen/Qwen3-30B-A3B", // 【必填】在 ModelScope 上选择的模型名称。
      "model_provider": "openai_compatible", // 【必填】ModelScope 兼容 OpenAI API。
      "streaming": true, // 【选填】是否启用流式传输。
      "max_tokens": 4096, // 【选填】最大 token 数。
      "tools_in_prompt": true, // 【选填】是否将工具 schema 包含在 prompt 中。
      "api_key": "your api key", // 【必填】ModelScope API 密钥。
      "configuration": { // 【选填】额外配置参数。
        "base_url": "https://api-inference.modelscope.cn/v1/", // ModelScope API 地址。
        "temperature": 0.6 // 温度参数。
      }
    }
  },
  "enable_tools": true // 【选填】全局开关，是否启用工具调用功能。设为 false 会禁用所有工具的使用。
}
```

## 3.2 工具配置 (mcp_config.example.json)

`mcp_config.example.json` 文件用于配置库可以调用的外部 MCP 工具服务器。这些工具通过标准的协议与 `mcp-host` 库通信。

以下是 `mcp_config.example.json` 的示例内容和详细注释：

```json
{
  "mcpServers": { // 【必填】一个字典，定义了不同的 MCP 工具服务器。
    "tavily-mcp-server": { // 以 tavily-mcp-server 为例，这是一个工具服务器的配置。
      "enabled": false, // 【必填】是否启用该工具服务器。设为 true 时库才能调用其提供的工具。
      "transport": "stdio", // 【必填】通信方式。例如 "stdio" 表示通过标准输入输出通信。
      "description": "tavily-mcp", // 【选填】工具服务器的描述。
      "registryUrl": "", // 【选填】工具注册表的 URL，用于发现工具（目前可能未使用或用于特定场景）。
      "command": "npx", // 【必填】启动工具服务器进程的命令。
      "args": [ // 【选填】启动命令的参数列表。
        "-y", // 例如 npx 的参数
        "tavily-mcp@0.1.4" // 要运行的 npm 包。
      ],
      "env": { // 【选填】启动工具进程时需要设置的环境变量。
        "TAVILY_API_KEY": "tvly-dev-……" // 例如，Tavily 工具需要的 API Key。请替换为你的真实密钥。
      },
      "disabled": false, // 【已废弃/不推荐使用】请优先使用 enabled 字段。
      "autoApprove": [] // 【选填】一个列表，列出哪些工具调用可以自动批准，无需用户确认。
    },
    "sequential-thinking": { // 以 sequential-thinking 工具服务器为例。
      "command": "npx", // 【必填】启动命令。
      "transport": "stdio", // 【必填】通信方式。
      "args": [ // 【选填】启动命令的参数。
        "-y",
        "@modelcontextprotocol/server-sequential-thinking"
      ],
      "enabled": true // 【必填】是否启用。
    },
    "MCP-HUB": { // 以 MCP-HUB 工具服务器为例，可能是一种集成了多个工具的服务。
      "description": "test", // 【选填】描述。
      "transport": "sse", // 【必填】通信方式，例如 "sse" (Server-Sent Events)。
      "enabled": false, // 【必填】是否启用。
      "timeout": 60, // 【选填】请求超时时间（秒）。
      "url": "http://localhost:3000/sse", // 【必填】工具服务器的 URL。
      "headers": {} // 【选填】调用 URL 时需要添加的额外 HTTP 头。
    }
  }
}
```

配置这些文件时，你需要确保填入正确的值，特别是 API 密钥和本地服务的地址。如果某个工具服务器的 `enabled` 设为 `false`，那么库将不会尝试调用该工具服务器下的任何工具。

## 4. 使用方法 (含配置加载和聊天示例)

使用 `mcp-host` 库的主要入口是 `woodenfishMcpHost` 类。你需要使用你的配置对其进行初始化，然后调用其方法与 AI 进行交互。

下面详细讲解核心 API 的使用：

### `woodenfishMcpHost` 类

这是整个库的**主入口点**。它的主要作用是管理整个 Host 的生命周期、配置以及创建聊天会话。你需要通过传入一个 `HostConfig` 对象来初始化它：

```python
from woodenfish_mcp_host.host.host import woodenfishMcpHost
# 假设 my_config 是一个已经加载好的 HostConfig 对象
host = woodenfishMcpHost(config=my_config)
# 这里的 my_config 包含了 LLM 和 MCP 工具的配置。
```

`woodenfishMcpHost` 类的生命周期管理（初始化和清理）可能需要额外的异步调用，具体取决于其内部实现（例如，是否使用了异步上下文管理器 `async with`）。请参考下方提供的完整示例代码中的注释部分，以及库本身的源代码（如果需要深入理解）。

### 聊天会话对象 (通过 `host.chat()` 获取)

通过调用 `woodenfishMcpHost` 实例的 `.chat()` 方法，你可以获取一个用于管理单个聊天会话的对象。这是你与智能体进行实际对话的接口。

```python
# 从 woodenfishMcpHost 实例获取聊天会话对象
chat_session = host.chat() # 不传入 chat_id 则创建新会话
# 或者，如果你想恢复一个已有的会话，可以传入会话 ID：
# chat_session = host.chat(chat_id="已存在的会话ID") # 恢复现有会话
```

这个 `chat_session` 对象负责维护当前会话的对话历史和状态，确保智能体能够理解上下文并进行多轮对话。

### 发送消息与处理响应 (`chat_session.astream()`)

与智能体交互的核心方法是 `chat_session.astream()`。这是一个**异步生成器**方法，用于发送用户消息并以**流式**方式接收智能体的响应。流式处理对于实时显示 AI 生成的内容非常有用，特别是在处理长时间运行的智能体思考或工具调用时。

调用 `astream()` 方法时，你需要传入用户的查询文本：

```python
user_message = "你好，你能做什么？" # 用户的输入或指令
print(f"用户: {user_message}")

print("AI: ", end="") # 为了流式显示，先打印提示符，不换行
# 使用 async for 迭代异步生成器，接收每个数据块 (chunk)
async for chunk in chat_session.astream(query=user_message):
     # 处理每一个接收到的数据块 (chunk)
     # chunk 的具体结构取决于底层的智能体实现（LangGraph 的状态机输出）
     # 通常，chunk 是一个字典，包含了智能体状态的变化信息
     # 你需要检查 chunk 的内容来提取 AI 生成的文本、工具调用提示、工具执行结果等
     # 最常见的场景是提取消息列表中的文本内容
     if isinstance(chunk, dict):
         if 'messages' in chunk and isinstance(chunk['messages'], list):
              # 从消息列表中找到最新的消息，通常是 AI 的回复
              for msg in reversed(chunk['messages']): # 从后往前找最新的消息
                  if hasattr(msg, 'content') and msg.content is not None:
                       # 如果是文本内容，打印出来（end="" 保持流式效果）
                       print(msg.content, end="")
                       break # 找到最新消息就退出内层循环，处理下一个 chunk
                  # 如果需要处理工具调用信息 (msg.tool_calls) 或工具消息 (msg.tool_message)，请在此添加逻辑
     # 如果 chunk 不是字典或不包含消息列表，可能是其他类型的状态更新，可以按需处理或忽略

print("\n(AI 响应结束)") # 响应结束后换行并打印提示
```

由于 `astream` 是一个异步生成器，你必须在 `async def` 函数中使用 `await` 关键字来迭代它，并且整个包含 `await` 调用的异步函数（例如上面示例中的 `run_chat_example`）需要通过 `asyncio.run()` 来作为程序的入口来运行。

你需要根据 `chunk` 的结构，从流中提取出文本内容或者其他类型的信息（如工具调用提示、工具执行结果等），并在你的应用中进行相应的展示或处理。理解 LangGraph 的输出结构有助于更精确地解析 `chunk`。

### 完整使用示例 (含配置加载和聊天)

下面的 Python 脚本是一个**完整、可直接运行**的示例，展示了如何加载示例配置、初始化 Host 以及开始一个基本的聊天会话并处理流式响应。这个示例整合了上面的概念。

```python
import asyncio
import json
import os
import sys
from woodenfish_mcp_host.host.host import woodenfishMcpHost
from woodenfish_mcp_host.host.conf import HostConfig, LLMConfig, McpServerConfig
# 根据需要导入 woodenfish_mcp_host 中的其他必要类，如各种消息类型 BaseMessage 等

# --- 示例：加载配置 ---
def load_example_config(library_dir: str) -> HostConfig:
    """加载示例配置文件并创建 HostConfig 对象"""
    llm_cfg = {}
    mcp_servers = {}

    # 尝试加载 LLM 示例配置
    llm_example_path = os.path.join(library_dir, "model_config.example.json")
    if os.path.exists(llm_example_path):
        try:
            with open(llm_example_path, "r", encoding='utf-8') as f:
                 llm_cfg_data = json.load(f)
                 llm_cfg = HostConfig.LLMConfig(**llm_cfg_data)
            print(f"已加载示例 LLM 配置：{llm_example_path}", file=sys.stderr)
        except Exception as e:
             print(f"警告：加载示例 LLM 配置失败 ({llm_example_path}): {e}", file=sys.stderr)
             llm_cfg = HostConfig.LLMConfig(provider="placeholder", model_name="placeholder")
    else:
         llm_cfg = HostConfig.LLMConfig(provider="placeholder", model_name="placeholder")
         print(f"未找到示例 LLM 配置：{llm_example_path}. 使用占位符配置。", file=sys.stderr)

    # 尝试加载 MCP 示例配置
    mcp_example_path = os.path.join(library_dir, "mcp_config.example.json")
    if os.path.exists(mcp_example_path):
        try:
            with open(mcp_example_path, "r", encoding='utf-8') as f:
                 mcp_cfg_data = json.load(f)
                 mcp_servers = {name: HostConfig.McpServerConfig(**cfg) for name, cfg in mcp_cfg_data.items()}
            print(f"已加载示例 MCP 配置：{mcp_example_path}", file=sys.stderr)
        except Exception as e:
            print(f"警告：加载示例 MCP 配置失败 ({mcp_example_path}): {e}", file=sys.stderr)
            mcp_servers = {}
    else:
        mcp_servers = {}
        print(f"未找到示例 MCP 配置：{mcp_example_path}. 使用空配置。", file=sys.stderr)

    # 创建主要的 HostConfig 对象
    # 注意：实际使用时，您可能需要加载 log_config 等其他重要配置
    host_config = HostConfig(
        llm=llm_cfg,
        mcp_servers=mcp_servers,
        log_config={} # 请根据需要提供实际的 LogConfig
    )
    return host_config

# --- 示例：运行聊天流程 ---
async def run_chat_example():
    """一个完整的聊天流程示例"""

    # 假设当前脚本位于 woodenfish-mcp-host-library 目录中
    # 如果脚本在其他位置，请调整 library_dir 变量
    library_dir = "."

    # 1. 加载配置
    my_config = load_example_config(library_dir)
    print("\n配置加载完成。", file=sys.stderr)

    # 2. 初始化 woodenfishMcpHost
    # 推荐使用异步上下文管理器，如果 Host 支持的话
    # （检查 Host 类定义是否有 async __aenter__ / __aexit__ 方法）
    # 如果 Host 没有实现 async context manager，您可能需要手动调用 init/shutdown 方法（如果存在）
    host = woodenfishMcpHost(config=my_config)
    # 如果 Host 有显式的 async init 方法：
    # await host.init()
    print("woodenfishMcpHost 初始化完成。", file=sys.stderr)

    try:
        # 3. 开始或恢复一个聊天会话
        # 您可以传入 chat_id 来恢复现有会话，不传则创建新会话
        chat_session = host.chat()
        print(f"聊天会话已创建，会话 ID: {chat_session._chat_id}", file=sys.stderr)

        # 4. 发送消息并处理响应（流式处理）
        user_message = "你好，你能做什么？"
        print(f"\n用户: {user_message}")

        print("AI:", end="") # 使用 end="" 避免换行，以便和流式输出连起来
        # astream 方法返回一个异步生成器
        async for chunk in chat_session.astream(query=user_message):
             # 处理流式接收到的 chunk
             # chunk 的类型和结构取决于 LangGraph 智能体的状态定义
             # 典型的 chunk 是一个包含状态更新的字典
             if isinstance(chunk, dict):
                 # 示例：如果 chunk 中包含消息列表
                 if 'messages' in chunk and isinstance(chunk['messages'], list):
                      # 找到最新的 AI 消息或工具调用消息并打印其内容
                      for msg in reversed(chunk['messages']): # 从后往前找最新的消息
                          if hasattr(msg, 'content') and msg.content is not None:
                               print(msg.content, end="") # 打印消息内容
                               break # 找到最新消息就退出内层循环
                          # 如果需要处理工具调用信息 (msg.tool_calls) 或工具消息 (msg.tool_message)，请在此添加逻辑

        print("\n(AI 响应结束)\n", file=sys.stderr)

        # 在同一会话中发送另一条消息的示例
        # user_message_2 = "使用工具读取文件 /tmp/test.txt 的内容"
        # print(f"用户: {user_message_2}")
        # print("AI:", end="")
        # async for chunk in chat_session.astream(query=user_message_2):
        #      if isinstance(chunk, dict) and 'messages' in chunk:
        #           for msg in reversed(chunk['messages']):
        #                if hasattr(msg, 'content') and msg.content is not None:
        #                     print(msg.content, end="")
        #                     break
        # print("\n(AI 响应结束)\n", file=sys.stderr)


    finally:
        # 5. 清理 Host 实例
        # 如果 Host 类实现了异步上下文管理器 (__aenter__ 和 __aexit__)，
        # 推荐使用 "async with woodenfishMcpHost(...) as host:" 的方式，
        # 这样可以在 Host 生命周期结束时自动进行清理。
        print("\n正在关闭 woodenfishMcpHost...", file=sys.stderr)
        # 如果 Host 有显式的 async shutdown 方法：
        # await host.shutdown()
        pass # 如果使用了 async with 则此处通常无需额外清理


# --- 运行主函数 ---
if __name__ == "__main__":
   # 使用 asyncio.run() 运行异步主函数
   asyncio.run(run_chat_example())
```

## 5. 集成 MCP 工具

`mcp-host` 库被设计为与外部 MCP 工具进程协同工作。

1.  **定义工具**: 你需要将你的 MCP 工具定义为独立的脚本或可执行文件，它们遵循标准的输入/输出 JSON 协议（从标准输入读取，输出到标准输出）。
2.  **配置工具**: 使用 `mcp_config.json` 文件（或你选择的配置方法）告诉 `mcp-host` 库关于你的工具的信息，包括它们的名称、运行命令和 schema（输入参数）。
3.  **确保可执行性**: 确保运行 `mcp-host` 库的进程可以访问并执行工具脚本/可执行文件。

有关 MCP 工具协议和实现示例（如 `echo.py`）的更多详细信息，请参阅 `woodenfish_mcp_host/host/tools/` 目录中的文档或代码。

## 6. 进一步开发

-   **构建你的服务层**: 将此库集成到你的应用程序中，使用如 FastAPI、Flask 或自定义 asyncio 服务器等框架，通过 HTTP 或其他协议暴露功能。
-   **实现前端**: 开发一个与你的服务层交互的用户界面。
-   **定制智能体和工具**: 扩展 `woodenfish_mcp_host` 库或创建新的 MCP 工具，以添加自定义的 AI 行为和能力。

---