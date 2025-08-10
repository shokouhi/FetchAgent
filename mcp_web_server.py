# mcp_web_server.py
# Generic, token-disciplined MCP tools server:
# - No raw HTML to the LLM
# - Question-driven snippet extraction
# - Link discovery biased by the question (optional)
# Requires: pip install mcp fastmcp beautifulsoup4 lxml requests
# Optional: pip install trafilatura

from __future__ import annotations

import re
import textwrap
from typing import Dict, List, Tuple, Optional

import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup

try:
    import trafilatura  # optional, better boilerplate removal
    HAS_TRAFILATURA = True
except Exception:
    HAS_TRAFILATURA = False

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("WebScraper")

# ---- HTTP session with retries & browsery headers ---------------------------
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
})
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[403, 429, 500, 502, 503, 504])
SESSION.mount("http://", HTTPAdapter(max_retries=retries))
SESSION.mount("https://", HTTPAdapter(max_retries=retries))
TIMEOUT = 25

# ---- Token-discipline limits (characters, not tokens, for simplicity) -------
SNIPPET_WINDOW_CHARS = 450     # approx local context around a match
SNIPPET_MAX = 4                # max snippets returned
SNIPPET_CLIP = 1800            # max length of each merged snippet
GENERAL_SUMMARY_PARAS = 3      # for generic fetch without a question
MAX_LINKS_RETURNED = 8         # for extract_links
ANCHOR_LEN_BONUS_RANGE = (3, 60)

# ---- Helpers ----------------------------------------------------------------
def safe_fetch(url: str) -> Tuple[str, str, int]:
    """Fetch a URL; return (final_url, html, status). Raises on HTTPError for non-2xx."""
    r = SESSION.get(url, timeout=TIMEOUT, allow_redirects=True)
    r.raise_for_status()
    return r.url, r.text, r.status_code

def soupify(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")

def bs4_text(soup: BeautifulSoup) -> str:
    # Strip obvious noise
    for t in soup(["script", "style", "noscript", "svg", "canvas", "img", "video", "audio"]):
        t.decompose()
    for t in soup.find_all(["nav", "footer", "form", "aside"]):
        t.decompose()
    return " ".join(soup.get_text(" ").split())

def extract_main_text(html: str) -> str:
    """Prefer trafilatura for main-content extraction; fallback to bs4."""
    if HAS_TRAFILATURA:
        try:
            txt = trafilatura.extract(html, include_tables=False, include_comments=False) or ""
            txt = " ".join(txt.split())
            if txt.strip():
                return txt
        except Exception:
            pass
    # fallback
    return bs4_text(soupify(html))

def sent_split(text: str) -> List[str]:
    # simple heuristic sentence splitter
    parts = re.split(r'(?<=[\.\!\?])\s+', text)
    return [p.strip() for p in parts if p.strip()]

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()

def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]{3,}", normalize(s))

def text_windows_around_matches(text: str, patterns: List[str], window: int) -> List[Tuple[int,int]]:
    spans: List[Tuple[int,int]] = []
    for pat in patterns:
        for m in re.finditer(re.escape(pat), text, flags=re.I):
            s = max(0, m.start() - window)
            e = min(len(text), m.end() + window)
            spans.append((s, e))
    return spans

def merge_spans(spans: List[Tuple[int,int]], pad: int = 80) -> List[Tuple[int,int]]:
    if not spans:
        return []
    spans.sort()
    merged: List[Tuple[int,int]] = []
    cs, ce = spans[0]
    for s, e in spans[1:]:
        if s <= ce + pad:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    return merged

def clip_snippet(text: str) -> str:
    return textwrap.shorten(text, width=SNIPPET_CLIP, placeholder="…")

def anchor_score(text: str, query_terms: List[str]) -> int:
    t = (text or "").strip().lower()
    score = 0
    for qt in query_terms:
        if qt in t:
            score += 10
    if ANCHOR_LEN_BONUS_RANGE[0] <= len(t) <= ANCHOR_LEN_BONUS_RANGE[1]:
        score += 1
    return score

def absolutize(base: str, href: str) -> str:
    return requests.compat.urljoin(base, href)

# ---- Core extraction logic ---------------------------------------------------
def make_question_snippets(text: str, question: Optional[str]) -> List[str]:
    """Return compact, question-focused snippets. If no question, return intro paragraphs."""
    if not text:
        return []
    if question:
        q_terms = tokenize(question)
        if not q_terms:
            # fallback to generic summary
            paras = [p for p in text.split("\n") if p.strip()]
            head = " ".join(paras) if paras else text
            sents = sent_split(head)[: 12]  # cap small
            para = " ".join(sents)
            return [clip_snippet(para)]
        # Build windows around question terms (as plain substrings)
        spans = text_windows_around_matches(text, q_terms, SNIPPET_WINDOW_CHARS)
        merged = merge_spans(spans)
        snippets = [clip_snippet(text[s:e]) for s, e in merged[:SNIPPET_MAX]]
        # If nothing matched (e.g., different wording), fall back to top paragraphs
        if not snippets:
            paras = [p for p in text.split("\n") if p.strip()]
            sents = sent_split(" ".join(paras[:3]))
            return [clip_snippet(" ".join(sents))]
        return snippets
    else:
        # Generic summary: take first few paragraphs/sentences
        paras = [p for p in text.split("\n") if p.strip()]
        if not paras:
            sents = sent_split(text)[:12]
            return [clip_snippet(" ".join(sents))]
        head = " ".join(paras[:GENERAL_SUMMARY_PARAS])
        sents = sent_split(head)[:20]
        return [clip_snippet(" ".join(sents))]

def find_relevant_links(url: str, html: str, question: Optional[str], max_links: int) -> List[Dict[str,str]]:
    soup = soupify(html)
    query_terms = tokenize(question or "")
    # Boost footer anchors
    candidates: List[Tuple[int, str, str]] = []
    footers = soup.find_all("footer")
    footer_anchors = [a for f in footers for a in f.find_all("a", href=True)]
    for a in footer_anchors:
        href = a.get("href")
        txt = (a.get_text() or "").strip()
        sc = anchor_score(txt, query_terms) + (5 if query_terms else 0)
        if sc > 0:
            candidates.append((sc, txt, absolutize(url, href)))
    # All anchors (lower weight)
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        txt = (a.get_text() or "").strip()
        sc = anchor_score(txt, query_terms)
        if sc > 0:
            candidates.append((sc, txt, absolutize(url, href)))
    # Dedup by URL, keep best score
    best = {}
    for sc, txt, absu in candidates:
        if absu not in best or sc > best[absu][0]:
            best[absu] = (sc, txt)
    ranked = sorted(((sc, txt, u) for u, (sc, txt) in best.items()), key=lambda x: x[0], reverse=True)
    # If no query, still return a compact set (top anchors by heuristic length)
    if not ranked and not query_terms:
        anchors = []
        for a in soup.find_all("a", href=True):
            txt = (a.get_text() or "").strip()
            if ANCHOR_LEN_BONUS_RANGE[0] <= len(txt) <= ANCHOR_LEN_BONUS_RANGE[1]:
                anchors.append((1, txt, absolutize(url, a["href"])))
        ranked = anchors[:max_links]
    return [{"url": u, "text": txt} for sc, txt, u in ranked[:max_links]]

# ---- Tools ------------------------------------------------------------------
@mcp.tool()
def fetch_page(url: str) -> Dict:
    """
    Fetch a URL and return compact, generic summary snippets (no raw HTML).
    Returns:
      { ok, url, status, title, snippets: [str], note }
    """
    try:
        final_url, html, status = safe_fetch(url)
        soup = soupify(html)
        title = soup.title.get_text(" ").strip() if soup.title else None
        text = extract_main_text(html)
        snippets = make_question_snippets(text, question=None)  # generic summary
        return {
            "ok": True,
            "url": final_url,
            "status": status,
            "title": title,
            "snippets": snippets[:SNIPPET_MAX],
            "note": "Generic summary snippets only (no raw HTML) to keep token usage low.",
        }
    except requests.HTTPError as e:
        return {"ok": False, "url": url, "error": f"HTTP {getattr(e.response,'status_code','?')}: {e}"}
    except Exception as e:
        return {"ok": False, "url": url, "error": str(e)}

@mcp.tool()
def fetch_snippets(url: str, question: Optional[str] = None) -> Dict:
    """
    Fetch a URL and return compact snippets relevant to `question`.
    If `question` is None, returns a short generic summary.
    Returns:
      { ok, url, status, title, snippets: [str], note }
    """
    try:
        final_url, html, status = safe_fetch(url)
        soup = soupify(html)
        title = soup.title.get_text(" ").strip() if soup.title else None
        text = extract_main_text(html)
        snippets = make_question_snippets(text, question=question)
        return {
            "ok": True,
            "url": final_url,
            "status": status,
            "title": title,
            "snippets": snippets[:SNIPPET_MAX],
            "note": ("Question-focused snippets." if question else "Generic summary snippets.") +
                    " No raw HTML returned.",
        }
    except requests.HTTPError as e:
        return {"ok": False, "url": url, "error": f"HTTP {getattr(e.response,'status_code','?')}: {e}"}
    except Exception as e:
        return {"ok": False, "url": url, "error": str(e)}

@mcp.tool()
def extract_links(url: str, query: Optional[str] = None) -> Dict:
    """
    Return a compact, ranked list of links likely relevant to `query`.
    If `query` is None, returns a small set of reasonable anchors.
    Returns:
      { ok, url, status, title, candidates: [{url, text}], note }
    """
    try:
        final_url, html, status = safe_fetch(url)
        soup = soupify(html)
        title = soup.title.get_text(" ").strip() if soup.title else None
        cands = find_relevant_links(final_url, html, question=query, max_links=MAX_LINKS_RETURNED)
        return {
            "ok": True,
            "url": final_url,
            "status": status,
            "title": title,
            "candidates": cands,
            "note": ("Candidates ranked by anchor/query overlap." if query
                     else "Generic candidate anchors (compact)."),
        }
    except requests.HTTPError as e:
        return {"ok": False, "url": url, "error": f"HTTP {getattr(e.response,'status_code','?')}: {e}"}
    except Exception as e:
        return {"ok": False, "url": url, "error": str(e)}

@mcp.tool()
def get_relevant(url: str, question: str) -> Dict:
    """
    One-shot helper for generic questions:
      1) Discover likely subpages based on `question`.
      2) If any found, fetch the best candidate and extract question-focused snippets.
      3) Else, fall back to extracting snippets from the given page.
    Returns:
      { ok, source, url, candidate_list?, status?, title?, snippets?, note?, error? }
    """
    try:
        # Find candidates first
        link_res = extract_links(url, query=question)
        if not link_res.get("ok"):
            page = fetch_snippets(url, question=question)
            page["source"] = "fallback_direct"
            return page

        candidates = link_res.get("candidates", [])
        if candidates:
            top = candidates[0]["url"]
            page = fetch_snippets(top, question=question)
            page["source"] = "candidate"
            page["candidate_list"] = candidates
            return page

        page = fetch_snippets(url, question=question)
        page["source"] = "no_candidates"
        page["candidate_list"] = []
        return page
    except Exception as e:
        return {"ok": False, "url": url, "error": str(e)}

# ---- Entry point ------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()
