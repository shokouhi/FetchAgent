from dotenv import load_dotenv
import os, re, json
from urllib.parse import urlparse

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.tools import tool

load_dotenv()

# ---------------- Config ----------------
MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI client for the web-enabled Responses API (does the browsing)
client = OpenAI(api_key=OPENAI_API_KEY)

# LLM (kept in case you want to expand later; not used in this minimal flow)
llm = ChatOpenAI(model=MODEL, temperature=0.0, api_key=OPENAI_API_KEY)

# ---------------- Shared helpers (same behavior as your original) ----------------

def _page_prompt(url: str, question: str, max_urls: int = 5) -> str:
    return f"""
You are a precise web research assistant.

1) Fetch and read ONLY this page: {url}
2) Answer this question using ONLY that page:
"{question}"

STRICT OUTPUT CONTRACT (return JSON only, no extra text):
- Keys: "answer" (string), "urls" (array of strings).
- If the answer IS found on THIS page: return {{"answer": "<the answer>", "urls": []}}.
- If NOT found on THIS page:
  - return {{"answer": "", "urls": [up to {max_urls} fully-qualified http/https links FROM THIS PAGE that are most likely to contain the answer]}}.
  - Select by link text/title/surrounding context. No fragments, anchors, mailto, javascript, or duplicates.
- No reasoning or explanation in output. JSON object only.

Examples:
{{"answer": "12345", "urls": []}}
{{"answer": "", "urls": ["https://example.com/a", "https://example.com/b"]}}
""".strip()


def _extract_json(text: str) -> dict:
    text = text.strip()
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
    resp = client.responses.create(
        model=MODEL,
        tools=[{"type": "web_search"}],
        input=prompt,
        temperature=0.0,
    )
    # Prefer resp.output_text when available; otherwise stitch from parts
    output_text = getattr(resp, "output_text", None)
    if not output_text:
        parts = []
        for item in getattr(resp, "output", [])[:1]:
            for c in getattr(item, "content", []):
                if getattr(c, "type", "") == "output_text":
                    parts.append(getattr(c, "text", ""))
        output_text = "".join(parts).strip()
    try:
        data = _extract_json(output_text)
    except Exception:
        # Be defensive—return empty answer, no urls
        data = {"answer": "", "urls": []}
    # Normalize URLs and re-dump to string for the caller
    data["urls"] = _normalize_urls(data.get("urls", []))
    return json.dumps(data, ensure_ascii=False)

# ---------------- Public API (same signatures) ----------------

def ask_page_once(url: str, question: str, link_cap: int = 5) -> dict:
    # Delegate to read_page tool directly to mimic your original behavior.
    raw = read_page.invoke({"url": url, "question": question, "max_urls": link_cap})
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
        return {"answer": root["answer"], "urls": []}

    candidates = root["urls"][:follow_limit]
    for u in candidates:
        print("crawling", u)
        sub = ask_page_once(u, question, link_cap=5)
        if sub["answer"]:
            return {"answer": sub["answer"], "url": u, "candidates": candidates}

    return {"answer": "", "urls": candidates}

# ---------------- Example ----------------
if __name__ == "__main__":
    url = "https://www.nike.com/t/nikecourt-mens-dri-fit-tennis-t-shirt-VT4PMW/HJ3470-100"
    question = "how to contact the customer service?"
    result = find_answer_recursive(url, question, follow_limit=3)
    print(json.dumps(result, indent=2, ensure_ascii=False))
