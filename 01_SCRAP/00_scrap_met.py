#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Met Museum scraper – offset pagination + de-dup by object_id + URL normalization
Keeps artworks only if description block exists/non-empty.
Merges localized dupes (/en/ vs canonical) and filters junk images.
"""

import json
import re
import time
from html import unescape
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://www.metmuseum.org"
START_URL = f"{BASE}/art/collection/search?showOnly=withImage"
OUTFILE = "met_with_desc_images.json"

REQUEST_DELAY = 0.25
PAGE_SIZE = 40
MAX_PAGES = 1000

# --------------- HTTP ---------------
def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=6, connect=6, read=6,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; metmuseum-scraper/1.3; +educational)"
    })
    return s

# --------------- utils ---------------
def clean_text(x: str | None) -> str:
    if not x:
        return ""
    x = unescape(x)
    return re.sub(r"\s+", " ", x).strip()

def soup_for(sess: requests.Session, url: str) -> BeautifulSoup:
    r = sess.get(url, timeout=30); r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def set_offset(url: str, offset: int) -> str:
    parts = list(urlparse(url))
    q = parse_qs(parts[4], keep_blank_values=True)
    q["offset"] = [str(offset)]
    parts[4] = urlencode({k: v[-1] for k, v in q.items()})
    return urlunparse(parts)

def extract_object_id_from_url(url: str) -> str | None:
    m = re.search(r"/art/collection/search/(\d+)", url)
    return m.group(1) if m else None

def normalize_object_url(url: str) -> str:
    """
    Canonicalize artwork URL:
      - remove '/en' language segment
      - remove trailing slash
    """
    if not url.startswith("http"):
        url = urljoin(BASE, url)
    url = re.sub(r"^(https?://www\.metmuseum\.org)/en(/.+)$", r"\1\2", url, flags=re.I)
    parts = urlparse(url)
    norm_path = re.sub(r"//+", "/", parts.path).rstrip("/")
    return urlunparse((parts.scheme, parts.netloc, norm_path, "", parts.query, ""))

def filter_good_images(urls: list[str]) -> list[str]:
    bad_patterns = [
        r"/Rodan/dist/svg/no-image-image-related\.svg",
        r"/iiif/.*/preview$",
    ]
    out = []
    seen = set()
    for u in urls:
        if any(re.search(p, u) for p in bad_patterns):
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

# --------------- pagination ---------------
def discover_artwork_links(sess: requests.Session, start_url: str) -> list[str]:
    all_links = []
    seen = set()
    page_idx = 0

    while page_idx < MAX_PAGES:
        offset = page_idx * PAGE_SIZE
        url = set_offset(start_url, offset) if offset else start_url
        soup = soup_for(sess, url)

        page_links = []
        for a in soup.select("a[href*='/art/collection/search/']"):
            href = a.get("href") or ""
            full = urljoin(url, href)
            full = normalize_object_url(full)
            if re.search(r"/art/collection/search/\d+$", full):
                oid = extract_object_id_from_url(full)
                if oid and oid not in seen:
                    seen.add(oid)
                    page_links.append(full)

        print(f"offset {offset}: +{len(page_links)} objects (total {len(seen)})")

        if not page_links:
            break

        all_links.extend(page_links)
        if len(page_links) < PAGE_SIZE:
            break

        page_idx += 1
        time.sleep(REQUEST_DELAY)

    return all_links

# --------------- per-artwork ---------------
def extract_images_from_page(soup: BeautifulSoup, page_url: str) -> list[str]:
    imgs = []

    og = soup.find("meta", attrs={"property": "og:image"})
    if og and og.get("content"):
        imgs.append(urljoin(page_url, og["content"]))

    for sel in [
        "img#artwork__image",
        "img.artwork__image",
        ".artwork__image img",
        ".artwork__images img",
        "figure img",
    ]:
        for img in soup.select(sel):
            src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
            if not src:
                srcset = img.get("srcset")
                if srcset:
                    cands = [c.strip().split(" ")[0] for c in srcset.split(",") if c.strip()]
                    if cands:
                        src = cands[-1]
            if src:
                imgs.append(urljoin(page_url, src))

    imgs = filter_good_images(imgs)
    return imgs

def parse_artwork(sess: requests.Session, url: str) -> dict | None:
    url = normalize_object_url(url)
    soup = soup_for(sess, url)

    desc = soup.select_one("div.artwork__intro__desc.js-artwork__intro__desc[itemprop='description']")
    if not desc:
        return None
    desc_text = clean_text(desc.get_text(" "))
    if not desc_text:
        return None

    desc_html = str(desc)

    title = None
    title_el = soup.find(["h1", "h2"], class_=re.compile(r"(artwork|title|object)", re.I)) or soup.find("h2")
    if title_el:
        title = clean_text(title_el.get_text(" "))

    artist = None
    a_el = soup.select_one(".artwork__intro__artist, [itemprop='creator'], .artwork__artist, .artwork__artist__name")
    if a_el:
        artist = clean_text(a_el.get_text(" "))

    date = None
    date_el = soup.select_one(".artwork__intro__date, [itemprop='dateCreated'], .artwork__date")
    if date_el:
        date = clean_text(date_el.get_text(" "))

    image_urls = extract_images_from_page(soup, url)
    if not image_urls:
        return None

    object_id = extract_object_id_from_url(url)

    return {
        "object_url": url,
        "object_id": object_id,
        "title": title,
        "artist": artist,
        "date": date,
        "description_html": desc_html,
        "description_text": desc_text,
        "image_urls": image_urls,
    }

# --------------- main ---------------
def main():
    sess = make_session()

    print("→ Discovering artwork links …")
    artwork_links = discover_artwork_links(sess, START_URL)
    print(f"Total objects discovered (unique by ID): {len(artwork_links)}")

    by_id = {}
    found_with_desc = 0  # <-- live counter of qualifying artworks

    for i, obj_url in enumerate(artwork_links, 1):
        try:
            rec = parse_artwork(sess, obj_url)
            if not rec:
                # not counted: no description / no images
                pass
            else:
                oid = rec.get("object_id")
                if oid:
                    if oid not in by_id:
                        by_id[oid] = rec
                        found_with_desc += 1
                        # print every 10 finds to keep output tidy
                        if found_with_desc % 10 == 0:
                            print(f"  ✓ Found with description so far: {found_with_desc} (at {i}/{len(artwork_links)})")
                    else:
                        # merge duplicate variants (rare with normalization, but safe)
                        cur = by_id[oid]
                        for k in ["title", "artist", "date", "description_html", "description_text"]:
                            if not cur.get(k) and rec.get(k):
                                cur[k] = rec[k]
                        if "/en/" in cur["object_url"] and "/en/" not in rec["object_url"]:
                            cur["object_url"] = rec["object_url"]
                        merged_imgs = []
                        seen_imgs = set()
                        for u in (cur.get("image_urls") or []) + (rec.get("image_urls") or []):
                            if u not in seen_imgs:
                                seen_imgs.add(u); merged_imgs.append(u)
                        cur["image_urls"] = filter_good_images(merged_imgs)

        except Exception as e:
            by_id[f"ERROR::{obj_url}"] = {"object_url": obj_url, "error": str(e)}

        if i % 25 == 0:
            print(f"Processed {i}/{len(artwork_links)} – found with description so far: {found_with_desc}")
        time.sleep(REQUEST_DELAY)

    results = [v for k, v in by_id.items() if not str(k).startswith("ERROR::")]
    errors  = [v for k, v in by_id.items() if str(k).startswith("ERROR::")]

    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✓ Saved {len(results)} unique artworks with images & descriptions → {OUTFILE}")
    print(f"✓ Total with description found: {found_with_desc}")
    if errors:
        with open("met_with_desc_images_errors.json", "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"⚠ Logged {len(errors)} errors to met_with_desc_images_errors.json")

if __name__ == "__main__":
    main()
