import asyncio
import json
from contextlib import AsyncExitStack
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


MCP_SERVER_URL = "https://mcp.openmbti.org/mcp"


def create_mcp_http_client(**kwargs) -> httpx.AsyncClient:
    """Create an httpx client that uses system proxy settings."""
    return httpx.AsyncClient(**kwargs)


class OpenMBTIClient:
    def __init__(self, server_url: str = MCP_SERVER_URL, call_timeout: float = 60.0):
        self.server_url = server_url
        self.session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._lock = asyncio.Lock()
        self._call_timeout = call_timeout

    async def connect(self) -> None:
        self._exit_stack = AsyncExitStack()
        streams = await self._exit_stack.enter_async_context(
            streamablehttp_client(
                url=self.server_url,
                timeout=30,
                sse_read_timeout=300,
                httpx_client_factory=create_mcp_http_client,
            )
        )
        self.session = await self._exit_stack.enter_async_context(
            ClientSession(streams[0], streams[1])
        )
        await self.session.initialize()

    async def disconnect(self) -> None:
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self.session = None

    async def __aenter__(self) -> "OpenMBTIClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

    async def _call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        async with self._lock:
            try:
                result = await asyncio.wait_for(
                    self.session.call_tool(name, arguments or {}),
                    timeout=self._call_timeout,
                )
            except asyncio.TimeoutError:
                raise TimeoutError(f"MCP call '{name}' timed out after {self._call_timeout}s")

        if result.content:
            for content in result.content:
                if hasattr(content, "text"):
                    return json.loads(content.text)
        return None

    async def get_questions(self) -> list[dict[str, Any]]:
        result = await self._call_tool("get_questions")
        if result and "questions" in result:
            return result["questions"]
        return result or []

    async def quick_test(self, answers: dict[str, int]) -> dict[str, Any]:
        """Submit answers and get MBTI result.

        Args:
            answers: Dict mapping question IDs ("1"-"32") to answers (1-5)
        """
        result = await self._call_tool("quick_test", {"answers": answers})
        return result or {}
