"""Barebones MCP client that connects to the HTTP server."""
import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


SERVER_URL = "http://127.0.0.1:8000/mcp"


async def main() -> None:
    async with streamablehttp_client(SERVER_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List tools
            tools = await session.list_tools()
            print("Tools:")
            for t in tools.tools:
                print(f"  - {t.name}: {t.description}")

            # Call the `add` tool
            result = await session.call_tool("add", {"a": 2, "b": 3})
            print("\nadd(2, 3) ->", result.content[0].text)

            # Call the `echo` tool
            result = await session.call_tool("echo", {"message": "hello mcp"})
            print("echo('hello mcp') ->", result.content[0].text)

            # Read a resource
            result = await session.read_resource("greeting://world")
            first = result.contents[0] if result.contents else None
            text = getattr(first, "text", None) if first else "<empty>"
            print("\ngreeting://world ->", text)


if __name__ == "__main__":
    asyncio.run(main())
