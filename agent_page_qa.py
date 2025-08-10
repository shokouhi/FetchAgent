# agent_page_qa.py
# Async LangChain agent that calls MCP tools with token-disciplined outputs.
# Provider toggle: OpenAI or Gemini (set in .env or environment)
#
# Requirements:
#   pip install python-dotenv "mcp[cli]" anyio
#   pip install langchain langchain-core langchain-community
#   pip install langchain-openai openai
#   pip install langchain-google-genai google-generativeai
#
# Server expected alongside this file:
#   mcp_web_server.py  (the MCP server this client spawns)

from dotenv import load_dotenv
load_dotenv()  # read .env from current directory, if present

import os
import sys
import json
import asyncio
from typing import Dict, Any, List, Union

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool  # decorator

# --- LLM provider selection --------------------------------------------------

PROVIDER = os.getenv("PROVIDER", "openai").lower().strip()

if PROVIDER == "openai":
    from langchain_openai import ChatOpenAI
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
elif PROVIDER == "gemini":
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
else:
    raise RuntimeError(
        f"Unsupported PROVIDER='{PROVIDER}'. Use 'openai' or 'gemini'. "
        "Set PROVIDER in your .env or environment."
    )

# Friendly checks for API keys
if PROVIDER == "openai" and not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env or export it.")
if PROVIDER == "gemini" and not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env or export it.")

# --- MCP client (stdio) ------------------------------------------------------
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent  # mcp>=1.x

SERVER_CMD = [sys.executable, "-u", "mcp_web_server.py"]

class MCPClient:
    def __init__(self, cmd: list[str]):
        self.cmd = cmd
        self._stdio_ctx = None
        self.read = None
        self.write = None
        self.session: ClientSession | None = None

    async def start(self):
        params = StdioServerParameters(command=self.cmd[0], args=self.cmd[1:])
        # Keep ctx so we can __aexit__ on the SAME task (fixes AnyIO cancel-scope issue)
        self._stdio_ctx = stdio_client(params)
        self.read, self.write = await self._stdio_ctx.__aenter__()
        self.session = ClientSession(self.read, self.write)
        await self.session.__aenter__()
        await self.session.initialize()
        print("MCP client started", file=sys.stderr, flush=True)

    async def stop(self):
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
        finally:
            self.session = None
            if self._stdio_ctx:
                try:
                    await self._stdio_ctx.__aexit__(None, None, None)
                finally:
                    self._stdio_ctx = None
            if self.read:
                try: await self.read.aclose()
                except Exception: pass
            self.read = None
            if self.write:
                try: await self.write.aclose()
                except Exception: pass
            self.write = None

    async def call_tool(self, tool_name: str, args: dict):
        result = await self.session.call_tool(tool_name, arguments=args)

        if getattr(result, "isError", False):
            err = getattr(result, "error", None) or "Tool returned error"
            raise RuntimeError(f"{tool_name} failed: {err}")

        # mcp>=1.x uses camelCase 'structuredContent'
        if getattr(result, "structuredContent", None) is not None:
            return result.structuredContent

        # Fallback: join TextContent blocks
        texts = []
        for c in (result.content or []):
            if isinstance(c, TextContent):
                texts.append(c.text)
        if texts:
            return {"ok": True, "text": "\n".join(texts)}

        return {"ok": False, "error": f"{tool_name} returned no content"}

mcp_client = MCPClient(SERVER_CMD)

# --- Tiny sanitizer to avoid context blow-ups --------------------------------

# Keep any single string reasonably small before passing to the LLM.
# (We already keep snippets small in the server; this is belt-and-suspenders.)
MAX_STR_LEN = 6000  # characters; approx few K tokens worst case

def _shorten(s: str) -> str:
    if len(s) <= MAX_STR_LEN:
        return s
    # Keep head + tail to preserve value & citations
    head = s[: int(MAX_STR_LEN * 0.7)]
    tail = s[-int(MAX_STR_LEN * 0.2):]
    return head + "\n…\n" + tail

def compact_payload(obj: Union[Dict, List, str, int, float, None]) -> Union[Dict, List, str, int, float, None]:
    """Recursively trim oversized strings in tool results."""
    if isinstance(obj, dict):
        return {k: compact_payload(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [compact_payload(x) for x in obj]
    if isinstance(obj, str):
        return _shorten(obj)
    return obj

# --- Wrap MCP tools as LangChain async tools ---------------------------------

@tool("get_relevant")
async def lc_get_relevant(url: str, question: str) -> Dict[str, Any]:
    """
    Route to the most relevant subpage (based on `question`) and return small snippets.
    """
    res = await mcp_client.call_tool("get_relevant", {"url": url, "question": question})
    return compact_payload(res) or {"ok": False, "url": url, "error": "empty tool response"}

@tool("fetch_snippets")
async def lc_fetch_snippets(url: str, question: str = "") -> Dict[str, Any]:
    """
    Fetch a page and return question-focused snippets (no raw HTML).
    """
    args = {"url": url}
    if question:
        args["question"] = question
    res = await mcp_client.call_tool("fetch_snippets", args)
    return compact_payload(res) or {"ok": False, "url": url, "error": "empty tool response"}

@tool("fetch_page")
async def lc_fetch_page(url: str) -> Dict[str, Any]:
    """
    Fetch a page and return compact generic summary snippets.
    """
    res = await mcp_client.call_tool("fetch_page", {"url": url})
    return compact_payload(res) or {"ok": False, "url": url, "error": "empty tool response"}

@tool("extract_links")
async def lc_extract_links(url: str, query: str = "") -> Dict[str, Any]:
    """
    Extract likely-relevant links (ranked by overlap with `query` if provided).
    """
    args = {"url": url}
    if query:
        args["query"] = query
    res = await mcp_client.call_tool("extract_links", args)
    return compact_payload(res) or {"ok": False, "url": url, "links": [], "error": "empty tool response"}

TOOLS = [lc_get_relevant, lc_fetch_snippets, lc_fetch_page, lc_extract_links]

# --- Prompt & Agent ----------------------------------------------------------

SYSTEM = """You are a careful browsing QA agent.

Use the tools to read only what is needed. Prefer precision over breadth.
Workflow:
1) First try `get_relevant(url, question)` which may jump to a better subpage and return small snippets.
2) If needed, call `fetch_snippets(url, question)` to extract question-focused snippets from a specific page.
3) If you just need a quick overview, `fetch_page(url)` returns a tiny generic summary.
4) You may use `extract_links(url, query)` to choose a better page, but keep to ONE hop.
5) Keep token usage small. Do not paste large blocks. Quote only short snippets for evidence.
6) Do not fabricate. If unknown, say so and suggest which link texts might help next.
Return a concise answer with 1–2 brief snippets and the page title/URL when useful.
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "Target URL: {url}\nQuestion: {question}\nAnswer using only fetched snippets."),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm, TOOLS, PROMPT)
# Verbose=True is handy while iterating. Switch to False to cut console noise.
executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=True)

# --- App entrypoint ----------------------------------------------------------

async def amain():
    url = input("Enter the page URL: ").strip()
    question = input("Enter your question about this page: ").strip()

    await mcp_client.start()
    try:
        result = await executor.ainvoke({"url": url, "question": question, "chat_history": []})
        print("\n=== ANSWER ===")
        print(result.get("output", ""))
    finally:
        await mcp_client.stop()

if __name__ == "__main__":
    asyncio.run(amain())
