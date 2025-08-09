# agent_page_qa.py
# Async LangChain agent that calls MCP tools (fetch_page, extract_links)
# Provider toggle: OpenAI or Gemini (set in .env or environment)
#
# Requirements:
#   pip install python-dotenv "mcp[cli]" anyio
#   pip install langchain langchain-core langchain-community
#   pip install langchain-openai openai
#   pip install langchain-google-genai google-generativeai
#
# Also have the MCP server file in the same folder:
#   mcp_web_server.py  (this script auto-spawns it)

from dotenv import load_dotenv
load_dotenv()  # read .env from current directory, if present

import os
import sys
import json
import asyncio
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool  # <-- use decorator

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
from mcp.types import TextContent  # available in mcp>=1.x

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
        # hold onto the ctx so we can __aexit__ it on the SAME task
        self._stdio_ctx = stdio_client(params)
        self.read, self.write = await self._stdio_ctx.__aenter__()
        self.session = ClientSession(self.read, self.write)
        await self.session.__aenter__()
        await self.session.initialize()
        print("MCP client started", file=sys.stderr, flush=True)

    async def stop(self):
        # tear down in the reverse order they were created
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
        finally:
            self.session = None
            # Make sure to exit the stdio_client context manager
            if self._stdio_ctx:
                try:
                    await self._stdio_ctx.__aexit__(None, None, None)
                finally:
                    self._stdio_ctx = None
            # These are usually closed by __aexit__, but guard anyway
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

        # If the tool errored, surface that
        if getattr(result, "isError", False):
            # newer SDKs may use `error` field/string
            err = getattr(result, "error", None) or "Tool returned error"
            raise RuntimeError(f"{tool_name} failed: {err}")

        # NEW: use the correct attribute name
        if getattr(result, "structuredContent", None) is not None:
            return result.structuredContent

        # Fallback: concatenate any text content blocks
        texts = []
        for c in (result.content or []):
            if isinstance(c, TextContent):
                texts.append(c.text)
        if texts:
            return {"ok": True, "text": "\n".join(texts)}

        # Nothing useful came back
        return {"ok": False, "error": f"{tool_name} returned no content"}

mcp_client = MCPClient(SERVER_CMD)

# --- Wrap MCP tools as LangChain async tools via decorator -------------------

@tool("fetch_page")
async def lc_fetch_page(url: str) -> Dict[str, Any]:
    """Fetch a webpage via MCP and return {ok,url,title,text,html,status,error?}."""
    res = await mcp_client.call_tool("fetch_page", {"url": url})
    print(res)
    # Always return something; the model can decide what to do with an error.
    return res or {"ok": False, "url": url, "error": "empty tool response"}

@tool("extract_links")
async def lc_extract_links(url: str) -> Dict[str, Any]:
    """Extract absolute links via MCP and return {ok,url,links,status,error?}."""
    res = await mcp_client.call_tool("extract_links", {"url": url})
    return res or {"ok": False, "url": url, "links": [], "error": "empty tool response"}


TOOLS = [lc_fetch_page, lc_extract_links]

# --- Prompt & Agent ----------------------------------------------------------

SYSTEM = """You are a careful browsing QA agent.
Rules:
- Use tools to fetch pages.
- First read the seed URL; if answer not present, follow ONE hop via extract_links -> fetch_page on the best link (prefer same domain/subdomains).
- If unknown, say you couldn’t find it and suggest likely link labels to click next.
- Provide a concise answer and include 1–2 brief snippets as evidence when possible.
- Do not fabricate content; answer only from fetched pages.
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "Target URL: {url}\nQuestion: {question}\nAnswer using the page(s)."),
        MessagesPlaceholder("agent_scratchpad"),  # REQUIRED
    ]
)

agent = create_tool_calling_agent(llm, TOOLS, PROMPT)
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
