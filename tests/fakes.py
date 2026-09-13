"""Test doubles for the chat model."""

from __future__ import annotations

from langchain_core.messages import AIMessage


class _StructuredRunnable:
    """Stands in for ``llm.with_structured_output(Schema)``."""

    def __init__(self, schema, responses: dict) -> None:
        self._schema = schema
        self._responses = responses
        self.calls: list = []

    def invoke(self, messages, **_kwargs):
        self.calls.append(messages)
        value = self._responses.get(self._schema.__name__)
        if value is None:
            raise AssertionError(f"FakeChatModel has no scripted {self._schema.__name__}")
        if callable(value):
            value = value(messages)
        return value


class _ToolRunnable:
    """Stands in for ``llm.bind_tools(tools)``."""

    def __init__(self, script: list[AIMessage]) -> None:
        self._script = list(script)
        self.invocations = 0

    def invoke(self, messages, **_kwargs):
        self.invocations += 1
        if self._script:
            return self._script.pop(0)
        return AIMessage(content="Done.")


class FakeChatModel:
    """A duck-typed chat model exposing only what the nodes call."""

    def __init__(
        self,
        structured: dict | None = None,
        tool_script: list[AIMessage] | None = None,
        text: str = "Reply text.",
    ) -> None:
        self.structured = structured or {}
        self.tool_script = tool_script or []
        self.text = text
        self.tool_runnable: _ToolRunnable | None = None

    def with_structured_output(self, schema, **_kwargs):
        return _StructuredRunnable(schema, self.structured)

    def bind_tools(self, _tools, **_kwargs):
        self.tool_runnable = _ToolRunnable(self.tool_script)
        return self.tool_runnable

    def invoke(self, _messages, **_kwargs):
        return AIMessage(content=self.text)
