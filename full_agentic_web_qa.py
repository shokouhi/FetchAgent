# full_agentic_web_qa.py
from dotenv import load_dotenv
import os, re, json, sys
from urllib.parse import urlparse

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# Agent imports
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# ---------------- Config ----------------
MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI client for the web-enabled Responses API (does the browsing)
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------- Prompt helpers for the web tool ----------------
def _page_prompt(url: str, question: str, max_urls: int = 5) -> str:
    return (
        "You are a careful, no-hallucination web QA assistant. "
        "Browse ONLY the given URL, read what's on the page, and try to answer the question. "
        "If the answer is not clearly on the page, DO NOT GUESS. "
        f"Instead, extract up to {max_urls} promising, fully-qualified links (http/https) from THIS page only "
        "that are most likely to contain the answer (e.g., help/contact/returns/policy pages). "
        "Return STRICT JSON as:\n"
        '{"answer": "<string or empty if not found>", "urls": ["<link1>", "..."]}\n\n'
        f"URL: {url}\n"
        f"QUESTION: {question}\n"
        "Remember: ONLY this page is allowed; do not follow links yourself."
    )


def _coerce_json(text: str) -> dict:
    # Be lenient about stray text around JSON
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Model did not return JSON:\n{text}")
    return json.loads(m.group(0))

def _normalize_urls(urls):
    clean, seen = [], set()
    for u in urls or []:
        try:
            pu = urlparse(u)
            if pu.scheme in ("http", "https") and u not in seen:
                seen.add(u)
                clean.append(u)
        except Exception:
            continue
    return clean

# ---------------- Tool: delegate actual browsing to OpenAI web_search ----------------
@tool("read_page", return_direct=False)
def read_page(url: str, question: str, max_urls: int = 5) -> str:
    """Fetch ONLY the given URL using OpenAI's web tool and answer the question, or return follow-up links.
    Returns a JSON string with keys: answer (string) and urls (array of strings)."""
    prompt = _page_prompt(url, question, max_urls=max_urls)
    # full_agentic_web_qa.py -> read_page
    resp = client.responses.create(
        model=MODEL,
        input=prompt,          # ok for Responses
        temperature=0,
        max_output_tokens=1200,
        # response_format={"type": "json_object"},  # <-- delete this line
    )



    # Extract text output
    text = resp.output_text
    data = _coerce_json(text)

    return json.dumps(data, ensure_ascii=False)

# ---------------- Public API (same signatures) ----------------
def ask_page_once(url: str, question: str, link_cap: int = 5) -> dict:
    # Call the tool function directly (not as a LangChain tool)
    raw = read_page(url, question, link_cap)
    data = json.loads(raw)
    return {"answer": data.get("answer", "") or "", "urls": data.get("urls", [])}

def find_answer_recursive(seed_url: str, question: str, follow_limit: int = 3) -> dict:
    """
    1) Try seed page.
    2) If no answer, take up to `follow_limit` links from seed and try each once.
    3) If still no answer, return empty answer + the candidate links we tried.
    """
    root = ask_page_once(seed_url, question, link_cap=follow_limit)
    if root["answer"]:
        return {"answer": root["answer"], "url": seed_url, "hops": [seed_url]}

    candidates = root["urls"][:follow_limit]
    hops = [seed_url]
    for u in candidates:
        hops.append(u)
        print("crawling", u)
        sub = ask_page_once(u, question, link_cap=5)
        if sub["answer"]:
            return {"answer": sub["answer"], "url": u, "hops": hops}

    return {"answer": "", "url": "", "hops": hops}

# ---------------- Full Agent path (uses the same @tool) ----------------
def _build_agent(max_hops: int):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
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



def answer_with_agent(seed_url: str, question: str, max_hops: int = 3) -> dict:
    """
    Lets the agent decide when to call `read_page` and which link to try next.
    Returns a dict with keys: answer, url, hops
    """
    executor = _build_agent(max_hops=max_hops)
    out = executor.invoke({
        "seed_url": seed_url,
        "question": question,
    })
    # Agent returns text in `out["output"]` per the prompt. Parse safely.
    try:
        return _coerce_json(out["output"])
    except Exception:
        # Fallback to empty shape
        return {"answer": "", "url": "", "hops": [seed_url]}

# ---------------- CLI ----------------
def _interactive_inputs():
    try:
        url = input("Enter the page URL: ").strip()
        q = input("Enter your question about this page: ").strip()
        return url, q
    except EOFError:
        # If piped, fall back to args handling
        return None, None

def _cli():
    # Priority: argv if provided, else interactive prompts
    # Usage:
    #   python full_agentic_web_qa.py <url> "<question>" [--max-hops N] [--no-agent]
    # or interactive:
    #   python full_agentic_web_qa.py  (then it will prompt)
    import argparse
    parser = argparse.ArgumentParser(description="Web QA: seed page + follow links if needed.")
    parser.add_argument("url", nargs="?", help="Seed URL")
    parser.add_argument("question", nargs="?", help="Question about the page")
    parser.add_argument("--max-hops", type=int, default=int(os.getenv("MAX_HOPS", "3")),
                        help="Max tool calls/hops (default 3)")
    parser.add_argument("--no-agent", action="store_true",
                        help="Disable agent; use direct recursive calls")
    args = parser.parse_args()

    seed_url, q = args.url, args.question
    if not seed_url or not q:
        iu, iq = _interactive_inputs()
        if iu and iq:
            seed_url, q = iu, iq
        else:
            parser.error("URL and question are required.")

    if args.no_agent:
        result = find_answer_recursive(seed_url, q, follow_limit=args.max_hops)
    else:
        result = answer_with_agent(seed_url, q, max_hops=args.max_hops)

    print(json.dumps(result, indent=2, ensure_ascii=False))

# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    _cli()
