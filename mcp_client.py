import os
import re
import json
import argparse
from typing import Dict

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import httpx

# ---------------- Config ----------------
MODEL = os.getenv("MODEL", "gpt-4o-mini")
MCP_URL = os.getenv("MCP_WEBQA_URL", "http://127.0.0.1:8765")


def _coerce_json(text: str) -> Dict:
    try:
        return json.loads(text)
    except Exception:
        pass
    import re as _re
    m = _re.search(r"\{.*\}", text, flags=_re.DOTALL)
    if not m:
        raise ValueError(f"Model did not return JSON:{text}")
    return json.loads(m.group(0))


# ---------------- Tool: delegate browsing to MCP (server) ----------------
@tool("read_page", return_direct=False)
def read_page(url: str, question: str, max_urls: int = 5) -> str:
    """Call MCP server's `read_page` so the server performs the web-enabled
    OpenAI Responses call. Returns JSON string with keys: answer, urls.

    Why this change: per your request to keep logic identical but move
    network/browsing out of the client, we proxy to MCP. No new parsing logic
    or heuristics are introduced on the client.
    """
    import anyio

    async def _call():
        async with httpx.AsyncClient(timeout=40.0) as client:
            r = await client.post(
                f"{MCP_URL}/tools/read_page",
                json={"arguments": {"url": url, "question": question, "max_urls": max_urls}},
            )
            r.raise_for_status()
            payload = r.json()
            content = payload.get("content", {})
            return json.dumps(
                {"answer": content.get("answer", ""), "urls": content.get("urls", [])},
                ensure_ascii=False,
            )

    return anyio.run(_call)


# ---------------- Agent-only path ----------------

def _build_agent(max_hops: int):
    llm = ChatOpenAI(model=MODEL, temperature=0)
    tools = [read_page]

    system_msg = (
        "You are a concise web QA agent. You may only browse via the provided tools.\n"
        "Goal: Answer the user's question from the seed page if possible. "
        "If not found, choose the most promising next link from the tool's returned `urls` "
        f"and try again, up to {max_hops} total tool calls.\n"
        "On success, STOP and produce final JSON:\n"
        '{{"answer": "<text>", "url": "<source url or empty>", "hops": ["<visited urls in order>"]}}\n'
        "If nothing is found after the budget, return:\n"
        '{{"answer": "", "url": "", "hops": ["<visited urls in order>"]}}'
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "Seed URL: {seed_url}\nQuestion: {question}\n\nStart by calling the tool once on the seed URL."),
        MessagesPlaceholder("agent_scratchpad"),
    ])


    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def answer_with_agent(seed_url: str, question: str, max_hops: int = 3) -> Dict:
    executor = _build_agent(max_hops=max_hops)
    out = executor.invoke({"seed_url": seed_url, "question": question})
    try:
        return _coerce_json(out["output"])
    except Exception:
        return {"answer": "", "url": "", "hops": [seed_url]}


# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web QA (MCP-backed): agent-only")
    parser.add_argument("url", help="Seed URL")
    parser.add_argument("question", help="Question about the page")
    parser.add_argument("--max-hops", type=int, default=int(os.getenv("MAX_HOPS", "3")))
    args = parser.parse_args()

    result = answer_with_agent(args.url, args.question, max_hops=args.max_hops)
    print(json.dumps(result, indent=2, ensure_ascii=False))