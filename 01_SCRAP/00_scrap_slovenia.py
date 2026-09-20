#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scrape https://slovenia.si/sl/umetnost-in-kulturna-dediscina
- Zbere vse članke (vključno s paginacijo).
- Na vsaki strani članka pobere vse slike in njihove napise/opise.
- Zraven vsake slike shrani tudi celoten članek (HTML + tekst) in metapodatke članka.

Izhod: slovenia_si_art_kultura.json
"""

import json
import re
import time
from html import unescape
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://slovenia.si"
START_URL = f"{BASE}/sl/umetnost-in-kulturna-dediscina"
OUTFILE = "slovenia_si_art_kultura.json"
REQUEST_DELAY = 0.2  # bodi vljuden

# ------------- HTTP session -------------
def make_session():
    s = requests.Session()
    retries = Retry(
        total=6, connect=6, read=6,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; slovenia-si-scraper/1.0; +educational)"
    })
    return s

# ------------- utils -------------
def clean_text(x: str | None) -> str:
    if not x:
        return ""
    x = unescape(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def get_soup(sess: requests.Session, url: str) -> BeautifulSoup:
    r = sess.get(url, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def absolutize(page_url: str, raw: str | None) -> str | None:
    if not raw:
        return None
    return urljoin(page_url, raw)

# ------------- discovery: list all article links -------------
def discover_article_links(sess: requests.Session, start_url: str) -> list[str]:
    """
    Najde vse članke v kategoriji. Prehodi paginacijo prek:
      - <a rel="next"> ali gumbov z 'page', 'stran', '?page='...
    """
    seen_pages = set()
    to_visit = [start_url]
    article_links = set()

    while to_visit:
        url = to_visit.pop(0)
        if url in seen_pages:
            continue
        seen_pages.add(url)

        soup = get_soup(sess, url)

        # 1) članki v gridu/listi (WordPress / custom):
        # poskusi tipične selektorje za kartice
        for a in soup.select(
            "article a[href], .post a[href], .entry a[href], .card a[href], .c-articles a[href], .listing a[href]"
        ):
            href = a.get("href") or ""
            full = urljoin(url, href)
            # heuristika: članki so pod /sl/ in niso datoteke / sidra / kategorije
            if not full.startswith(f"{BASE}/"):
                continue
            if any(full.lower().endswith(ext) for ext in [".pdf", ".jpg", ".jpeg", ".png", ".webp", ".gif"]):
                continue
            if "#comments" in full or full.endswith("#"):
                continue
            # izločimo očitne navigacije kategorij
            if re.search(r"/(kategorija|oznaka|tag|category)/", full, flags=re.I):
                continue
            # članki imajo pogosto /sl/.... (ne /en/, /de/...)
            if "/sl/" in full and "/sl/umetnost-in-kulturna-dediscina" not in full:
                article_links.add(full)
            # včasih so članki tudi v isti poti, poskusi še to:
            elif re.search(r"/sl/.+/.+", full):
                article_links.add(full)

        # 2) paginacija – poskusi rel="next"
        next_link = soup.find("a", rel=lambda v: v and "next" in v.lower())
        if next_link and next_link.get("href"):
            nxt = urljoin(url, next_link["href"])
            if nxt not in seen_pages:
                to_visit.append(nxt)

        # 3) dodatne paginacije po vzorcu '?page=' ali '/page/2'
        for a in soup.select("a[href*='page='], a[href*='/page/']"):
            nxt = urljoin(url, a.get("href", ""))
            # mora ostati v isti kategoriji
            if nxt.startswith(START_URL) and nxt not in seen_pages:
                to_visit.append(nxt)

        time.sleep(REQUEST_DELAY)

    # deduplikacija in stabilno sortiranje
    links = sorted(article_links)
    return links

# ------------- parse a single article -------------
def extract_article_meta_and_content(soup: BeautifulSoup, page_url: str) -> dict:
    """
    Vrne:
      - title, date, author
      - article_html, article_text
      - images: seznam slovarjev {image_url, caption}
    """
    # naslov
    title_el = soup.find(["h1", "h2"], class_=re.compile(r"(entry|post|article|title)", re.I)) or soup.find("h1")
    title = clean_text(title_el.get_text(" ")) if title_el else None

    # datum in avtor (poskusi več lokacij)
    date = author = None

    # tipične WP sheme:
    meta_candidates = [
        soup.select_one("time[datetime]"),
        soup.select_one(".entry-date"),
        soup.select_one(".post-date"),
        soup.select_one("meta[property='article:published_time']"),
    ]
    for m in meta_candidates:
        if not m:
            continue
        if m.name == "time" and m.get("datetime"):
            date = m["datetime"]
            break
        if m.name == "meta" and m.get("content"):
            date = m["content"]
            break
        txt = clean_text(m.get_text(" "))
        if txt:
            date = txt
            break

    author_candidates = [
        soup.select_one(".author a"),
        soup.select_one(".post-author a"),
        soup.select_one("meta[name='author']"),
        soup.find(attrs={"itemprop": "author"}),
    ]
    for a in author_candidates:
        if not a:
            continue
        if a.name == "meta" and a.get("content"):
            author = a["content"]
            break
        txt = clean_text(a.get_text(" "))
        if txt:
            author = txt
            break

    # glavno vsebinsko področje
    content = (
        soup.select_one("article .entry-content")
        or soup.select_one(".entry-content")
        or soup.select_one("article")
        or soup.select_one("main")
        or soup.find("article")
        or soup.find("main")
        or soup.find("body")
    )

    article_html = str(content) if content else None
    article_text = clean_text(content.get_text(" ")) if content else None

    # slike + napisi
    images = []

    # Najprej figure/figcaption (najzanesljivejše)
    for fig in soup.select("figure"):
        # poišči img (ali source/srcset)
        img = fig.find("img")
        if not img:
            continue
        src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
        # če je srcset, vzemi prvi URL (ali največjega, če želiš)
        if not src:
            srcset = img.get("srcset")
            if srcset:
                # vzemi največjo različico (zadnja ponavadi najširša)
                candidates = [c.strip().split(" ")[0] for c in srcset.split(",") if c.strip()]
                if candidates:
                    src = candidates[-1]
        if not src:
            continue
        image_url = absolutize(page_url, src)
        # caption
        cap = fig.find("figcaption")
        caption = clean_text(cap.get_text(" ")) if cap else (clean_text(img.get("alt")) or None)
        images.append({"image_url": image_url, "caption": caption})

    # Nato še “samostojne” slike v vsebini (če niso že zajete)
    if content:
        for img in content.select("img"):
            src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
            if not src:
                srcset = img.get("srcset")
                if srcset:
                    candidates = [c.strip().split(" ")[0] for c in srcset.split(",") if c.strip()]
                    if candidates:
                        src = candidates[-1]
            if not src:
                continue
            image_url = absolutize(page_url, src)
            # poišči napis v bližini (npr. div.wp-caption-text, ali naslednji/parent figcaption)
            caption = None
            # najprej parent figure
            pf = img.find_parent("figure")
            if pf:
                fc = pf.find("figcaption")
                if fc:
                    caption = clean_text(fc.get_text(" "))
            # wp-caption
            if not caption:
                wp_cap = img.find_parent(class_=re.compile(r"wp-caption", re.I))
                if wp_cap:
                    t = wp_cap.find(class_=re.compile(r"wp-caption-text", re.I))
                    if t:
                        caption = clean_text(t.get_text(" "))
            # alt fallback
            if not caption:
                caption = clean_text(img.get("alt")) or None

            # dodaj, če še ni v images
            if image_url and all(image_url != it["image_url"] for it in images):
                images.append({"image_url": image_url, "caption": caption})

    # deduplikacija (stabilno)
    dedup = []
    seen = set()
    for it in images:
        if it["image_url"] not in seen:
            dedup.append(it)
            seen.add(it["image_url"])
    images = dedup

    return {
        "title": title,
        "date": date,
        "author": author,
        "article_html": article_html,
        "article_text": article_text,
        "images": images,
    }

# ------------- main -------------
def main():
    sess = make_session()

    print("→ Iščem članke …")
    article_links = discover_article_links(sess, START_URL)
    print(f"Najdenih člankov: {len(article_links)}")

    records = []
    for i, url in enumerate(article_links, 1):
        try:
            soup = get_soup(sess, url)
            meta = extract_article_meta_and_content(soup, url)

            # za vsako sliko naredimo svoj zapis z “zraven priloženim” člankom
            for img in meta["images"]:
                records.append({
                    "article_url": url,
                    "article_title": meta["title"],
                    "article_date": meta["date"],
                    "article_author": meta["author"],
                    "article_html": meta["article_html"],
                    "article_text": meta["article_text"],
                    "image_url": img["image_url"],
                    "image_caption": img["caption"],
                })

        except Exception as e:
            records.append({
                "article_url": url,
                "error": str(e),
            })

        if i % 10 == 0:
            print(f"  … {i}/{len(article_links)}")
        time.sleep(REQUEST_DELAY)

    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"✓ Shranjenih zapisov (slik): {len(records)} → {OUTFILE}")

if __name__ == "__main__":
    main()
