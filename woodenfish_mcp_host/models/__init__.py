"""MCP 的附加模型。"""

import logging
from importlib import import_module
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger("woodenfish_mcp_host.models")


def load_model(
    provider: str,
    model_name: str,
    *args: Any,
    **kwargs: Any,
) -> BaseChatModel:
    """从 models 目录加载模型。

    参数：
        provider: 提供者名称。支持两种特殊提供者：
            - "woodenfish": 使用 woodenfish_mcp_host.models 中的模型
            - "__load__": 从配置中加载模型
        model_name: 要加载的模型名称。
        args: 传递给模型的附加参数。
        kwargs: 传递给模型的附加关键字参数。

    返回：
        加载的模型。

    如果提供者是 "woodenfish"，它应该是这样的：
        import woodenfish_mcp_host.models.model_name_in_lower_case as model_module
        model = model_module.load_model(*args, **kwargs)
    如果提供者是 "__load__"，model_name 是模型的类名。
    例如，model_name="package.module:ModelClass" 时，它将是这样的：
        import package.module as model_module
        model = model_module.ModelClass(*args, **kwargs)
    如果提供者既不是 "woodenfish" 也不是 "__load__"，它将从 langchain 加载模型。
    """
    # XXX 将配置/参数传递给模型

    logger.debug(
        "Loading model %s with provider %s, kwargs: %s",
        model_name,
        provider,
        kwargs,
    )
    if provider == "woodenfish":
        model_name_lower = model_name.replace("-", "_").replace(".", "_").lower()
        model_module = import_module(
            f"woodenfish_mcp_host.models.{model_name_lower}",
        )
        model = model_module.load_model(*args, **kwargs)
    elif provider == "__load__":
        module_path, class_name = model_name.rsplit(":", 1)
        model_module = import_module(module_path)
        class_ = getattr(model_module, class_name)
        model = class_(*args, **kwargs)
    else:
        if len(args) > 0:
            raise ValueError(
                f"Additional arguments are not supported for {provider} provider.",
            )
        model = init_chat_model(model=model_name, model_provider=provider, **kwargs)
    return model
