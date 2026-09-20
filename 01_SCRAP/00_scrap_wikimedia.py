#!/usr/bin/env python3
"""
Extract images + descriptions from Media Viewer links on:
  https://commons.wikimedia.org/wiki/Slovenija

What it does:
- Parse the page and collect all links like "#/media/File:Something.jpg"
  (and also direct "/wiki/File:..." links as fallback).
- Normalize to "File:..." titles.
- Use Commons API to fetch:
    * image_url (original)
    * thumbnail_url (configurable width)
    * description_html + description_text
    * raw extmetadata (author, license, etc.)
- Save JSON to commons_slovenija_media.json
"""

import html
import json
import re
import time
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup

START_URL = "https://commons.wikimedia.org/wiki/Slovenija"
API = "https://commons.wikimedia.org/w/api.php"
OUTFILE = "commons_slovenija_media.json"
THUMB_WIDTH = 1280
REQUESTS_SLEEP = 0.2

# ---------- HTTP ----------
session = requests.Session()
session.headers.update({"User-Agent": "commons-media-viewer-scraper/1.0 (educational use)"})


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
        except Exception as e:
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
    Input like 'File:Tina_Maze.jpg' or '/media/File:Tina_Maze.jpg' or '/wiki/File:Tina_Maze.jpg'
    Return normalized 'File:Tina Maze.jpg' (spaces allowed; Commons handles both).
    """
    if not raw:
        return None
    s = raw.strip()

    # Remove leading paths like '/media/' or '/wiki/'
    s = s.lstrip("#/")  # strip starting '#', '/'
    if s.lower().startswith("media/"):
        s = s[6:]
    if s.lower().startswith("wiki/"):
        s = s[5:]

    # Now s should start with 'File:' or 'file:'
    if not s.lower().startswith("file:"):
        return None

    # Decode percent-encoding; Commons accepts spaces
    s = unquote(s)
    # Replace underscores with spaces (optional)
    s = s.replace("_", " ")
    # Ensure 'File:' capitalized
    if s.lower().startswith("file:"):
        s = "File:" + s[5:]
    return s


def collect_file_titles_from_page(url: str) -> List[Dict]:
    """
    Return a list of dicts with:
      - title: 'File:...'
      - media_viewer_url: original href (resolved)
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

        # Prefer fragment-based media links like '#/media/File:...'
        title = None
        if parsed.fragment:
            frag = parsed.fragment
            # Common forms: '/media/File:Name.jpg' or 'media/File:Name.jpg'
            if frag.lower().startswith("/media/") or frag.lower().startswith("media/"):
                title = normalize_file_title(frag)

        # Fallback to direct /wiki/File:... links
        if not title:
            path = parsed.path  # e.g., '/wiki/File:Tina_Maze.jpg'
            if path.lower().startswith("/wiki/file:"):
                title = normalize_file_title(path)

        if title and title not in seen:
            items.append({"title": title, "media_viewer_url": full})
            seen.add(title)

    return items


def fetch_imageinfo(file_titles: List[str]) -> Dict[str, Dict]:
    """
    For given File: titles, return a dict title -> metadata dict:
      { "image_url", "thumbnail_url", "description_html", "description_text", "extmetadata" }
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


def main():
    # 1) Collect all File: titles from media-viewer links on the page
    items = collect_file_titles_from_page(START_URL)
    titles = [it["title"] for it in items]
    print(f"Found {len(titles)} file titles from media-viewer links")

    if not titles:
        print("No media-viewer file links found. Check the page or adjust selectors.")
        return

    # 2) Fetch metadata for those files
    meta_by_title = fetch_imageinfo(titles)

    # 3) Build output records
    records = []
    for it in items:
        title = it["title"]
        meta = meta_by_title.get(title, {})
        file_page = f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}"
        records.append({
            "title": title,
            "media_viewer_url": it["media_viewer_url"],
            "file_page": file_page,
            "image_url": meta.get("image_url"),
            "thumbnail_url": meta.get("thumbnail_url"),
            "description_html": meta.get("description_html"),
            "description_text": meta.get("description_text"),
            "extmetadata": meta.get("extmetadata", {}),
        })

    # 4) Save JSON
    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(records)} records to {OUTFILE}")


if __name__ == "__main__":
    main()
