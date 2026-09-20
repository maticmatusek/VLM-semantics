#!/usr/bin/env python3
"""
Scrape VisMet thumbnails + lightbox pages:
- Start at http://www.vismet.org/VisMet/display.php
- Find all <a href="...showimage.php?...&id=..."> (class 'lightbox' or not)
- For each, grab:
    id, thumb_url (from <img>), title (from <a title>), page_url (showimage URL),
    full_image_url (from the showimage page),
    description (dict of label:value parsed from the showimage page)
- Write to vismet.json
"""

import json
import re
import time
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse, parse_qs, urlencode

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "http://www.vismet.org/VisMet/"
INDEX_URL = urljoin(BASE, "display.php")
SHOW_URL = urljoin(BASE, "showimage.php")
OUTFILE = "vismet.json"

# Canonical label normalization for known fields (we still keep *all* fields)
CANON_LABELS = {
    "notes": "Notes",
    "content conceptualization": "Content Conceptualization",
    "context": "Context",
    "content expression": "Content Expression",
    "expression realization": "Expression Realization",
    "linguistic expression": "Linguistic Expression",
    "content": "Content",  # just in case
    "conceptualization": "Conceptualization",
}

def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5, connect=5, read=5,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    s.headers.update({
        "User-Agent": "vismet-scraper/1.1 (+for research/educational use)"
    })
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def normalize_label(lbl: str) -> str:
    key = clean_text(lbl).lower().strip(" :")
    if key in CANON_LABELS:
        return CANON_LABELS[key]
    # punctuation-insensitive match
    key2 = re.sub(r"[^\w\s]", "", key)
    for k, v in CANON_LABELS.items():
        if re.sub(r"[^\w\s]", "", k) == key2:
            return v
    return clean_text(lbl).strip(" :\u00a0")

def extract_id_from_href(href: str) -> Optional[str]:
    try:
        q = parse_qs(urlparse(href).query)
        vals = q.get("id") or q.get("ID")
        if vals and vals[0]:
            return vals[0]
    except Exception:
        pass
    return None

def canonical_show_url(raw_href: str) -> str:
    """
    Remove lightbox[...] params; keep only the id, so we fetch clean content:
    showimage.php?id=XYZ
    """
    u = urlparse(urljoin(BASE, raw_href))
    q = parse_qs(u.query)
    item_id = (q.get("id") or q.get("ID") or [""])[0]
    q_clean = {}
    if item_id:
        q_clean["id"] = item_id
    clean_qs = urlencode(q_clean)
    return f"{u.scheme}://{u.netloc}{u.path}" + (f"?{clean_qs}" if clean_qs else "")

def find_all_lightbox_links(session: requests.Session, start_url: str) -> List[dict]:
    """
    On display.php (and simple pagination if present), collect all showimage.php links.
    Also store associated thumbnail URL and title from the <a>.
    """
    seen_pages = set()
    to_visit = [start_url]
    items = []

    while to_visit:
        url = to_visit.pop(0)
        if url in seen_pages:
            continue
        seen_pages.add(url)

        r = session.get(url, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # collect showimage links
        for a in soup.select("a[href*='showimage.php']"):
            href = a.get("href") or ""
            sid = extract_id_from_href(href)
            if not sid:
                continue
            show_url = canonical_show_url(href)
            title = clean_text(a.get("title") or "")

            # associated thumbnail (if the anchor wraps an <img>)
            thumb_url = None
            img = a.find("img")
            if img and img.get("src"):
                thumb_url = urljoin(BASE, img["src"])

            items.append({
                "id": sid,
                "page_url": show_url,
                "title": title or None,
                "thumb_url": thumb_url,
            })

        # try to follow naive pagination (links that point back to display.php with page/offset)
        for pag in soup.select("a[href*='display.php']"):
            href = pag.get("href") or ""
            full = urljoin(BASE, href)
            if "display.php" in full and full not in seen_pages and full not in to_visit:
                # Heuristic: only queue if it looks like a pager (has page/offset/start params)
                q = parse_qs(urlparse(full).query)
                if any(k.lower() in {"page", "p", "start", "offset"} for k in q.keys()):
                    to_visit.append(full)

        # Be polite
        time.sleep(0.3)

    # De-duplicate by (id, page_url) keeping first occurrence (preserves first-found thumb/title)
    dedup = {}
    for it in items:
        key = (it["id"], it["page_url"])
        if key not in dedup:
            dedup[key] = it
    return list(dedup.values())

def parse_showimage_page(html: str) -> dict:
    """
    From the showimage page HTML, extract:
      - full_image_url (best guess = largest <img>)
      - description dict from tables, DLs, or bold-label patterns
    """
    soup = BeautifulSoup(html, "html.parser")

    # Full image: choose largest on the page (by width*height if provided)
    best_img = None
    best_area = -1
    for img in soup.find_all("img"):
        src = img.get("src")
        if not src:
            continue
        # Prefer site-hosted images
        if "http" in src and "vismet.org" not in src:
            continue
        try:
            w = int(img.get("width") or 0)
            h = int(img.get("height") or 0)
            area = w * h
        except Exception:
            area = 0
        if area > best_area:
            best_area = area
            best_img = img

    full_image_url = urljoin(BASE, best_img["src"]) if (best_img and best_img.get("src")) else None

    # Description extraction
    desc = {}

    # (a) Two-column tables (label/value)
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            if len(cells) >= 2:
                label_raw = clean_text(cells[0].get_text(" "))
                value_raw = clean_text(cells[1].get_text(" "))
                if label_raw and value_raw:
                    label = normalize_label(label_raw)
                    desc[label] = (desc.get(label, "") + " " + value_raw).strip()

    # (b) Definition lists
    for dl in soup.find_all("dl"):
        dts = dl.find_all("dt")
        dds = dl.find_all("dd")
        for dt, dd in zip(dts, dds):
            label_raw = clean_text(dt.get_text(" "))
            value_raw = clean_text(dd.get_text(" "))
            if label_raw and value_raw:
                label = normalize_label(label_raw)
                desc[label] = (desc.get(label, "") + " " + value_raw).strip()

    # (c) Bold labels followed by text (Label: value)
    for b in soup.find_all(["b", "strong"]):
        label_raw = clean_text(b.get_text(" "))
        if not label_raw:
            continue
        if label_raw.endswith(":") or label_raw.lower().strip(":") in CANON_LABELS:
            parts = []
            sib = b.next_sibling
            steps = 0
            while sib and steps < 3:
                if isinstance(sib, str):
                    parts.append(sib)
                else:
                    parts.append(sib.get_text(" "))
                sib = sib.next_sibling
                steps += 1
            value_raw = clean_text(" ".join(parts))
            if value_raw:
                label = normalize_label(label_raw)
                desc[label] = (desc.get(label, "") + " " + value_raw).strip()

    # Cleanup
    desc = {k: clean_text(v) for k, v in desc.items() if clean_text(v)}

    return {"full_image_url": full_image_url, "description": desc}

def main():
    session = make_session()

    print("Scanning display.php for lightbox items…")
    items = find_all_lightbox_links(session, INDEX_URL)
    print(f"Found {len(items)} lightbox links")

    results = []
    for i, it in enumerate(items, 1):
        show_url = it["page_url"]
        try:
            resp = session.get(show_url, timeout=30)
            resp.raise_for_status()
            parsed = parse_showimage_page(resp.text)

            results.append({
                "id": it["id"],
                "title": it.get("title"),
                "thumb_url": it.get("thumb_url"),
                "photo_url": parsed["full_image_url"],     # main/full image
                "description": parsed["description"],      # dict
                "page_url": show_url,
            })

        except Exception as e:
            # Keep partial record to help debugging, but don't crash the run
            results.append({
                "id": it["id"],
                "title": it.get("title"),
                "thumb_url": it.get("thumb_url"),
                "photo_url": None,
                "description": {},
                "page_url": show_url,
                "error": str(e),
            })

        if i % 25 == 0:
            print(f"Processed {i}/{len(items)}…")
        time.sleep(0.3)  # be polite

    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} records to {OUTFILE}")

if __name__ == "__main__":
    main()
