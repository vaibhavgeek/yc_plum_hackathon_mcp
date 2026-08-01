# MCP Barebones

Minimal Model Context Protocol server + client over HTTP (streamable transport).

## Structure

```
server/main.py   # FastMCP server with add, echo tools + greeting resource
client/main.py   # Client that connects, lists tools, calls them
pyproject.toml   # Deps: mcp[cli], uvicorn, starlette
```

## Setup

```bash
# Using uv (recommended)
uv sync

# Or with pip
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Run

Terminal 1 — start the server (listens on http://127.0.0.1:8000/mcp):

```bash
uv run python -m server.main
```

Terminal 2 — run the client:

```bash
uv run python -m client.main
```

Expected output:

```
Tools:
  - add: Add two numbers.
  - echo: Echo a message back.

add(2, 3) -> 5
echo('hello mcp') -> Echo: hello mcp

greeting://world -> Hello, world!
```

## Extend

- Add a tool: decorate a function with `@mcp.tool()` in `server/main.py`.
- Add a resource: decorate with `@mcp.resource("scheme://{param}")`.
- Add a prompt: decorate with `@mcp.prompt()`.
