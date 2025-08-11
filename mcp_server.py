import os
import re
import json
from typing import Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()

# MCP server over FastAPI HTTP
from mcp.server import FastMCP
from openai import OpenAI

# NEW: FastAPI shim
from fastapi import FastAPI
from pydantic import BaseModel

MODEL = os.getenv("WEB_MODEL", os.getenv("MODEL", "gpt-4o-mini"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

mcp = FastMCP()  # MCP app

# ---------------- Prompt helpers (unchanged) ----------------
def _page_prompt(url: str, question: str, max_urls: int = 5) -> str:
    return (
        "You are a careful, no-hallucination web QA assistant. "
        "Browse ONLY the given URL, read what's on the page, and try to answer the question. "
        "If the answer is not clearly on the page, DO NOT GUESS. "
        f"Instead, extract up to {max_urls} promising, fully-qualified links (http/https) from THIS page only "
        "that are most likely to contain the answer (e.g., help/contact/returns/policy pages). "
        "Return STRICT JSON as: "
        '{"answer": "<string or empty if not found>", "urls": ["<link1>", "..."]} '
        f"URL: {url} "
        f"QUESTION: {question} "
        "Remember: ONLY this page is allowed; do not follow links yourself."
    )

def _coerce_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Model did not return JSON:{text}")
    return json.loads(m.group(0))

# ---------------- MCP tool (as before) ----------------
@mcp.tool()
async def read_page(url: str, question: str, max_urls: int = 5) -> Dict[str, Any]:
    """Use OpenAI's web-enabled Responses API to *only* read the given URL and
    either answer the question or return follow-up links."""
    prompt = _page_prompt(url, question, max_urls=max_urls)

    resp = client.responses.create(
        model=MODEL,
        input=prompt,
        temperature=0,
        max_output_tokens=1200,
    )

    text = resp.output_text
    data = _coerce_json(text)
    return {"answer": data.get("answer", "") or "", "urls": data.get("urls", [])}

# ---------------- FastAPI shim (NEW) ----------------
class ReadPageEnvelope(BaseModel):
    url: Optional[str] = None
    question: Optional[str] = None
    max_urls: Optional[int] = 5
    # support the client's {"arguments": {...}} shape
    arguments: Optional[Dict[str, Any]] = None

api = FastAPI()

@api.get("/")
async def root():
    return {"status": "ok", "message": "MCP Web QA server"}

@api.post("/tools/read_page")
async def http_read_page(body: ReadPageEnvelope):
    # Accept either direct fields or {"arguments": {...}}
    url = body.url
    question = body.question
    max_urls = body.max_urls if body.max_urls is not None else 5

    if body.arguments:
        url = body.arguments.get("url", url)
        question = body.arguments.get("question", question)
        max_urls = body.arguments.get("max_urls", max_urls)

    if not url or not question:
        return {"error": "Missing 'url' or 'question'."}

    result = await read_page(url=url, question=question, max_urls=max_urls)
    # Match the client's expectation: payload["content"] -> {answer, urls}
    return {"content": result}

# Mount the MCP HTTP app under /mcp (optional but useful)
api.mount("/mcp", mcp.streamable_http_app)

# ---------------- Run the server ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="127.0.0.1", port=8765)
