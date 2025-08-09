# mcp_web_server.py
import re, sys, time
from typing import Dict, List, Optional
import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

mcp = FastMCP("WebScraper")

# Use a real browser UA and retries; many retail sites block default clients
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
TIMEOUT = 20

def safe_fetch(url: str) -> Dict:
    try:
        r = SESSION.get(url, timeout=TIMEOUT, allow_redirects=True)
        ct = (r.headers.get("content-type") or "").lower()
        if "text/html" not in ct and "application/xhtml+xml" not in ct:
            return {"ok": False, "error": f"Unsupported content-type: {ct}", "status": r.status_code}
        if r.status_code >= 400:
            return {"ok": False, "error": f"HTTP {r.status_code}", "status": r.status_code}
        return {"ok": True, "html": r.text, "status": r.status_code}
    except requests.RequestException as e:
        return {"ok": False, "error": f"Request failed: {e.__class__.__name__}: {e}"}

def extract_text(html: str) -> str:
    if HAS_TRAFILATURA:
        try:
            x = trafilatura.extract(html, include_tables=True, favor_recall=True)
            if x:
                return x
        except Exception:
            pass
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    txt = soup.get_text("\n")
    return re.sub(r"\n{3,}", "\n\n", txt).strip()

@mcp.tool()
def fetch_page(url: str) -> Dict:
    """Fetch a webpage and return {ok, url, title, text, html, status, error?}."""
    res = safe_fetch(url)
    if not res.get("ok"):
        return {"ok": False, "url": url, "status": res.get("status"), "error": res.get("error")}
    html = res["html"]
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else url
    return {
        "ok": True,
        "url": url,
        "status": res.get("status"),
        "title": title,
        "text": extract_text(html)[:20000],
        "html": html[:200000],
    }

@mcp.tool()
def extract_links(url: str) -> Dict:
    """Extract absolute links and anchors from a page. Returns {ok, url, links, status, error?}."""
    res = safe_fetch(url)
    if not res.get("ok"):
        return {"ok": False, "url": url, "status": res.get("status"), "error": res.get("error"), "links": []}
    html = res["html"]
    soup = BeautifulSoup(html, "html.parser")
    out: List[Dict] = []
    for a in soup.find_all("a", href=True):
        abs_url = requests.compat.urljoin(url, a["href"])
        anchor = (a.get_text() or "").strip()
        out.append({"url": abs_url, "anchor": anchor})
    return {"ok": True, "url": url, "status": res.get("status"), "links": out}

if __name__ == "__main__":
    print("MCP WebScraper server starting on stdio…", file=sys.stderr, flush=True)
    mcp.run()
