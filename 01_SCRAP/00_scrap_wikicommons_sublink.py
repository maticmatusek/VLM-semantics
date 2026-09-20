#!/usr/bin/env python3
"""
Scrape Commons media-viewer links for multiple Slovenia-related pages.

For each search term (e.g., "Mount Triglav", "Ljubljana", ...):
  - Find best-matching Commons page (article or category) via API search.
  - Fetch that page.
  - Collect links like "#/media/File:*.jpg" (and fallback "/wiki/File:*").
  - Resolve to "File:*" titles and fetch image URLs + descriptions via API.
Output JSON: commons_slovenia_sublinks.json
"""

import html
import json
import re
import time
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup

API = "https://commons.wikimedia.org/w/api.php"
BASE = "https://commons.wikimedia.org"
OUTFILE = "commons_slovenia_sublinks.json"
THUMB_WIDTH = 1280
REQUESTS_SLEEP = 0.2

# 👉 Add or edit your targets here:
SEED_TERMS = [
    "Mount Triglav",
    "Ljubljana",     # (fixes 'Ljubjana')
    "Maribor",
    "Celje",
    "Olm",           # Proteus anguinus
    "Lake Bohinj",
    "Nanos",
    "France Prešeren",
    "Kranj",
    "Nova Gorica",
    "Novo mesto",
    "Koper",
    "Piran",
    "Ptuj",
    "Krško",
    "Jesenice",
    "France Prešeren",
    "Anton Martin Slomšek",
    "Jožef Štefan",
    "Ivan Grohar",
    "Ivana Kobilica",
    "Lojze Peterle",
    "Tina Maze",
    "Predjama Castle",
]

session = requests.Session()
session.headers.update({"User-Agent": "commons-media-viewer-multi/1.0 (educational use)"})


# ---------- API helpers ----------
def mw_get(params: Dict) -> Dict:
    p = {"format": "json", "formatversion": "2"}
    p.update(params)
    for attempt in range(5):
        try:
            r = session.get(API, params=p, timeout=30)
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise RuntimeError(str(data["error"]))
            return data
        except Exception:
            if attempt == 4:
                raise
            time.sleep(0.7 * (attempt + 1))


# ---------- Utils ----------
def to_plain_text(html_str: Optional[str]) -> Optional[str]:
    if not html_str:
        return None
    text = re.sub(r"<\s*br\s*/?>", "\n", html_str, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def normalize_file_title(raw: str) -> Optional[str]:
    """
    Accepts 'File:Name.jpg', '/media/File:Name.jpg', '/wiki/File:Name.jpg', or '#/media/File:Name.jpg'
    Returns 'File:Name with spaces.jpg'
    """
    if not raw:
        return None
    s = raw.strip()

    # strip leading '#', then leading slashes
    if s.startswith("#"):
        s = s[1:]
    s = s.lstrip("/")

    # strip 'media/' or 'wiki/' prefixes
    if s.lower().startswith("media/"):
        s = s[6:]
    if s.lower().startswith("wiki/"):
        s = s[5:]

    if not s.lower().startswith("file:"):
        return None

    s = unquote(s).replace("_", " ")
    if s.lower().startswith("file:"):
        s = "File:" + s[5:]
    return s


def collect_file_titles_from_page(url: str) -> List[Dict]:
    """
    From a single Commons page, collect media viewer file links.
    Returns a list of {title (File:*), media_viewer_url}.
    """
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    items: List[Dict] = []
    seen: Set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = urljoin(url, href)
        parsed = urlparse(full)

        title = None
        # Prefer fragment-based media links like '#/media/File:...'
        if parsed.fragment:
            frag = parsed.fragment
            if frag.lower().startswith("/media/") or frag.lower().startswith("media/"):
                title = normalize_file_title(frag)

        # Fallback: direct links like '/wiki/File:...'
        if not title:
            path = parsed.path
            if path.lower().startswith("/wiki/file:"):
                title = normalize_file_title(path)

        if title and title not in seen:
            items.append({"title": title, "media_viewer_url": full})
            seen.add(title)

    return items


def fetch_imageinfo(file_titles: List[str]) -> Dict[str, Dict]:
    """
    For given File: titles, return mapping:
      title -> { image_url, thumbnail_url, description_html, description_text, extmetadata }
    """
    info: Dict[str, Dict] = {}
    for i in range(0, len(file_titles), 50):
        chunk = file_titles[i:i + 50]
        params = {
            "action": "query",
            "titles": "|".join(chunk),
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": str(THUMB_WIDTH),
            "formatversion": "2",
        }
        data = mw_get(params)
        for p in data.get("query", {}).get("pages", []):
            title = p.get("title")
            ii = (p.get("imageinfo") or [{}])[0]
            image_url = ii.get("url")
            thumb_url = ii.get("thumburl")
            extmetadata = ii.get("extmetadata") or {}
            desc_html = None
            if isinstance(extmetadata.get("ImageDescription"), dict):
                desc_html = extmetadata["ImageDescription"].get("value")
            desc_text = to_plain_text(desc_html)
            info[title] = {
                "image_url": image_url,
                "thumbnail_url": thumb_url,
                "description_html": desc_html,
                "description_text": desc_text,
                "extmetadata": extmetadata,
            }
        time.sleep(REQUESTS_SLEEP)
    return info


def find_best_commons_page(term: str) -> Optional[str]:
    """
    Use MediaWiki search to find the most relevant page for the term.
    Prefer mainspace and Category pages. Return canonical title (e.g., 'Category:Triglav' or 'Mount Triglav').
    """
    # Try exact page first
    for candidate in (term, term.title()):
        data = mw_get({"action": "query", "titles": candidate, "prop": "info"})
        pages = data.get("query", {}).get("pages", [])
        if pages and pages[0].get("missing") is None:
            return pages[0]["title"]

    # Use search across ns 0 (main) and 14 (Category)
    data = mw_get({
        "action": "query",
        "list": "search",
        "srsearch": term,
        "srnamespace": "0|14",
        "srlimit": 10,
    })
    results = data.get("query", {}).get("search", [])

    # Heuristic: prefer exact-ish matches, then "Category:..." matches
    if not results:
        return None

    # Score results
    best = None
    best_score = -1
    tnorm = term.lower()
    for r in results:
        title = r.get("title", "")
        ns = r.get("ns", 0)
        t = title.lower().replace("_", " ")
        score = r.get("score", 0)
        # small bonuses for name containment and category relevance
        if tnorm in t:
            score += 10
        if ns == 14:  # Category
            score += 3
        if score > best_score:
            best = title
            best_score = score

    return best


def main():
    all_records = []
    summary = []

    for term in SEED_TERMS:
        try:
            resolved_title = find_best_commons_page(term)
            if not resolved_title:
                summary.append({"term": term, "status": "no_page"})
                continue

            page_url = f"{BASE}/wiki/{resolved_title.replace(' ', '_')}"
            items = collect_file_titles_from_page(page_url)
            titles = [it["title"] for it in items]
            print(f"[{term}] → {resolved_title}: {len(titles)} media file(s)")

            meta = fetch_imageinfo(titles) if titles else {}

            # Build records
            for it in items:
                t = it["title"]
                file_page = f"{BASE}/wiki/{t.replace(' ', '_')}"
                m = meta.get(t, {})
                all_records.append({
                    "topic": term,
                    "resolved_page_title": resolved_title,
                    "page_url": page_url,
                    "title": t,
                    "media_viewer_url": it["media_viewer_url"],
                    "file_page": file_page,
                    "image_url": m.get("image_url"),
                    "thumbnail_url": m.get("thumbnail_url"),
                    "description_html": m.get("description_html"),
                    "description_text": m.get("description_text"),
                    "extmetadata": m.get("extmetadata", {}),
                })

            summary.append({
                "term": term,
                "resolved_page_title": resolved_title,
                "page_url": page_url,
                "files_found": len(titles),
            })

        except Exception as e:
            summary.append({"term": term, "status": f"error: {e}"})

        time.sleep(REQUESTS_SLEEP)

    # Save JSON
    out = {
        "summary": summary,
        "records": all_records,
    }
    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_records)} records to {OUTFILE}")
    print("Summary:")
    for s in summary:
        print(" -", s)

if __name__ == "__main__":
    main()
    