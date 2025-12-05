"""
Tool emulator for testing agents with mock tools.
"""

import inspect
import json
import logging
from collections.abc import Callable
from typing import Any


logger = logging.getLogger(__name__)


class ToolEmulator:
    """
    A utility to create and manage mock tools for agent testing.
    Allows defining tools with specific behaviors, return values, and side effects.
    """

    def __init__(self):
        self.tools: dict[str, Callable] = {}
        self.calls: list[dict[str, Any]] = []

    def register_tool(self, name: str, func: Callable, description: str | None = None):
        """
        Register a python function as a tool.

        Args:
            name: Name of the tool
            func: The function to execute
            description: Optional description (defaults to docstring)
        """
        self.tools[name] = func
        # Store metadata if needed for schema generation

    def mock_tool(self, name: str, return_value: Any = None, side_effect: Callable | None = None):
        """
        Create a mock tool that returns a static value or executes a side effect.

        Args:
            name: Name of the tool
            return_value: Value to return when called
            side_effect: Function to call instead of returning a value
        """

        def mock_func(*args, **kwargs):
            self.calls.append({"tool": name, "args": args, "kwargs": kwargs})

            if side_effect:
                return side_effect(*args, **kwargs)
            return return_value

        self.tools[name] = mock_func

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """
        Generate OpenAI-compatible tool definitions for registered tools.
        """
        definitions = []
        for name, func in self.tools.items():
            # Basic schema generation - can be enhanced with Pydantic
            sig = inspect.signature(func)
            doc = inspect.getdoc(func) or f"Tool {name}"

            parameters = {"type": "object", "properties": {}, "required": []}

            for param_name, param in sig.parameters.items():
                param_type = "string"  # Default
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == list:
                    param_type = "array"
                elif param.annotation == dict:
                    param_type = "object"

                parameters["properties"][param_name] = {
                    "type": param_type,
                    "description": f"Parameter {param_name}",
                }

                if param.default == inspect.Parameter.empty:
                    parameters["required"].append(param_name)

            definitions.append(
                {
                    "type": "function",
                    "function": {"name": name, "description": doc, "parameters": parameters},
                }
            )

        return definitions

    def call_tool(self, name: str, arguments: str | dict[str, Any]) -> Any:
        """
        Execute a registered tool.

        Args:
            name: Tool name
            arguments: Tool arguments (dict or JSON string)
        """
        if name not in self.tools:
            msg = f"Tool {name} not found"
            raise ValueError(msg)

        if isinstance(arguments, str):
            try:
                kwargs = json.loads(arguments)
            except json.JSONDecodeError:
                # Maybe it's just a string argument?
                kwargs = {"arg": arguments}
        else:
            kwargs = arguments

        try:
            return self.tools[name](**kwargs)
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            raise
