from langchain_core.tools import tool
from langchain_core.tools.base import BaseTool


class TestTool:
    """测试工具。"""

    def __init__(self) -> None:
        """初始化测试状态。"""
        self._called: bool = False

    @property
    def called(self) -> bool:
        """工具是否已被调用。"""
        return self._called

    @property
    def weather_tool(self) -> BaseTool:
        """天气工具。"""

        @tool
        def check_weather_location(city: str) -> str:
            """获取当前天气信息。"""
            self._called = True
            return f"The weather in {city} is sunny."

        return check_weather_location
