# MCP Voice Client

Barebones MCP server + a web voice client. Tap a mic button in your browser, speak a command, and see the raw MCP tool response.

```
Voice → Web Speech API → transcript → OpenAI (function calling)
      → picks MCP tool + args → MCP server → response → UI
```

## Structure

```
server/main.py          # FastMCP server: add, echo tools + greeting resource
client/main.py          # CLI client (original)
client/web.py           # FastAPI web client backend
client/static/index.html  # Voice UI (mic button + response cards)
pyproject.toml
.env.example
```

## Setup

```bash
uv sync
cp .env.example .env
# edit .env and paste a fresh OPENAI_API_KEY
```

## Run

Terminal 1 — MCP server:

```bash
uv run python -m server.main
```

Terminal 2 — web voice client:

```bash
uv run python -m client.web
```

Open http://127.0.0.1:5173 in Chrome (Web Speech API requires Chromium-based browsers). Tap the mic, say a command:

- "add two and three"
- "echo hello world"

Response card shows: transcript, tool chosen, arguments, and the raw MCP response.

## CLI variant (no UI)

```bash
uv run python -m client.main
```

## Security note

`.env` is gitignored. Never commit or paste your API key in chat — rotate it at https://platform.openai.com/api-keys if you do.

## Extend

- Add a tool: decorate a function with `@mcp.tool()` in `server/main.py`. The web UI picks it up automatically on the next request (tools are listed fresh each call).
- Swap the LLM model: change `gpt-4o-mini` in `client/web.py`.
