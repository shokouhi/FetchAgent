from openai import OpenAI
from dotenv import load_dotenv
import json
import os
import re
from urllib.parse import urlparse

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"
TOOLS = [{"type": "web_search"}]

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
    """Tolerant JSON grab: find the first {...} block and parse it."""
    text = text.strip()
    # If it's already pure JSON, try directly
    try:
        return json.loads(text)
    except Exception:
        pass
    # Otherwise extract first balanced {...}
    m = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Model did not return JSON:\n{text}")
    j = m.group(0)
    return json.loads(j)

def _normalize_urls(urls):
    clean = []
    seen = set()
    for u in urls or []:
        try:
            pu = urlparse(u)
            if pu.scheme in ("http", "https") and u not in seen:
                seen.add(u)
                clean.append(u)
        except Exception:
            continue
    return clean

def ask_page_once(url: str, question: str, link_cap: int = 5) -> dict:
    prompt = _page_prompt(url, question, max_urls=link_cap)
    resp = client.responses.create(
        model=MODEL,
        tools=TOOLS,
        input=prompt,
        temperature=0.0
    )
    # Prefer .output_text if present; otherwise fall back to assembling text
    output_text = getattr(resp, "output_text", None)
    if not output_text:
        # Fallback: concatenate text parts from the first item
        parts = []
        for item in getattr(resp, "output", [])[:1]:
            for c in getattr(item, "content", []):
                if getattr(c, "type", "") == "output_text":
                    parts.append(getattr(c, "text", ""))
        output_text = "".join(parts).strip()
    data = _extract_json(output_text)
    return {
        "answer": data.get("answer", "") or "",
        "urls": _normalize_urls(data.get("urls", []))
    }

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
            return {"answer": sub["answer"],"url": u, "candidates": candidates}

    return {"answer": "", "urls": candidates}

# ------------------ Example usage ------------------
if __name__ == "__main__":
    url = "https://www.nike.com/t/nikecourt-mens-dri-fit-tennis-t-shirt-VT4PMW/HJ3470-100?nikemt=true&cp=83188699188_search_--x-20404158439---c-1015707943-00197862434590&dplnk=member&gclsrc=aw.ds&gad_source=1&gad_campaignid=20404160074&gbraid=0AAAAADy86kO2BUnAwBH0berwn0uGUYNAm&gclid=CjwKCAjw49vEBhAVEiwADnMbbKDErEtSY60Mheb-MM4nlBd7n4YkuS0A_LthF4Z8_nXB7sKrzhx5QBoCoO0QAvD_BwE"
    question = "how to contact the customer service?"
    result = find_answer_recursive(url, question, follow_limit=3)
    print(json.dumps(result, indent=2, ensure_ascii=False))
