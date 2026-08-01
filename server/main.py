"""Barebones MCP server using streamable HTTP transport."""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("barebones-server", host="127.0.0.1", port=8000)


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@mcp.tool()
def echo(message: str) -> str:
    """Echo a message back."""
    return f"Echo: {message}"


@mcp.resource("greeting://{name}")
def greeting(name: str) -> str:
    """Return a greeting for the given name."""
    return f"Hello, {name}!"


if __name__ == "__main__":
    # Streamable HTTP transport (modern MCP HTTP transport)
    mcp.run(transport="streamable-http")
