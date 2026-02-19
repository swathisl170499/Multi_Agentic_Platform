from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: list[str]


class MCPService:
    """Minimal MCP client service to list tools from configured MCP servers."""

    def __init__(self, servers: list[MCPServerConfig]) -> None:
        self._servers = servers

    def configured_servers(self) -> list[dict[str, Any]]:
        return [{"name": s.name, "command": s.command, "args": s.args} for s in self._servers]

    async def list_tools(self, server_name: str) -> list[dict[str, Any]]:
        server = next((s for s in self._servers if s.name == server_name), None)
        if server is None:
            raise ValueError(f"Unknown MCP server: {server_name}")

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise ImportError("mcp package is required. Install: pip install mcp") from exc

        params = StdioServerParameters(command=server.command, args=server.args)

        async with stdio_client(params) as (read, write):
            session = ClientSession(read, write)
            await session.initialize()
            tools = await session.list_tools()

        # `tools` format depends on server/sdk version; normalize to dict list.
        payload: list[dict[str, Any]] = []
        for tool in getattr(tools, "tools", tools):
            name = getattr(tool, "name", None) if not isinstance(tool, dict) else tool.get("name")
            description = (
                getattr(tool, "description", "") if not isinstance(tool, dict) else tool.get("description", "")
            )
            payload.append({"name": name or "unknown", "description": description})
        return payload
