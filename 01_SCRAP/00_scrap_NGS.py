#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NG - stalna zbirka (fix: normalizacija detail_url/author_url in čist meta/inventory)
"""

import json
import re
import time
import unicodedata
from html import unescape
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://www.ng-slo.si"
COLLECTION_BASE = f"{BASE}/si/stalna-zbirka"
MEDIA_PREFIX = f"{BASE}/si/"
OUTFILE = "ng_stalna_zbirka_structured.json"
REQUEST_DELAY = 0.25

PERIOD_LABELS = [
    "1200–1600","1600–1700","1700–1800","1800–1820",
    "1820–1870","1870–1900","1900–1918","Od 1918 dalje",
    "Zoran Mušič","Znamenite umetnine",
]

def session_with_retries():
    s = requests.Session()
    retries = Retry(total=6, connect=6, read=6, backoff_factor=0.7,
                    status_forcelist=(429,500,502,503,504),
                    allowed_methods=frozenset(["GET","HEAD"]), raise_on_status=False)
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent":"Mozilla/5.0 (compatible; NG-StalnaZbirkaScraper/1.6)"})
    return s

def clean_text(x: str | None) -> str:
    if not x: return ""
    x = unescape(x)
    return re.sub(r"\s+", " ", x).strip()

def slugify(label: str) -> str:
    s = label.strip().lower().replace("–","-").replace("—","-")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s\-]", "", s)
    return re.sub(r"\s+", "-", s).strip("-")

def period_candidate_urls(label: str) -> list[str]:
    s = slugify(label); base = f"{COLLECTION_BASE}/"
    cand = [base + s]
    specials = {
        "od-1918-dalje": ["od-1918-dalje","od-1918-naprej"],
        "zoran-music": ["zoran-music"],
        "znamenite-umetnine": ["znamenite-umetnine","znamenite"],
        "1200-1600":["1200-1600"],"1600-1700":["1600-1700"],"1700-1800":["1700-1800"],
        "1800-1820":["1800-1820"],"1820-1870":["1820-1870"],"1870-1900":["1870-1900"],
        "1900-1918":["1900-1918"],
    }
    if s in specials: cand = [base+x for x in specials[s]] + cand
    if re.match(r"^\d{3,4}-\d{2,4}$", s): cand.append(base + s.replace("-", "–"))
    seen, out = set(), []
    for u in cand:
        if u not in seen: seen.add(u); out.append(u)
    return out

def get_soup(sess: requests.Session, url: str) -> BeautifulSoup:
    r = sess.get(url, timeout=30); r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def resolve_period_url(sess, label):
    first_ok=None
    for u in period_candidate_urls(label):
        try:
            soup = get_soup(sess, u)
            if first_ok is None: first_ok=u
            if soup.select_one(".collections-gallery a[href], .gallery a[href], a[href*='workId=']"):
                return u
        except Exception:
            pass
        time.sleep(REQUEST_DELAY)
    return first_ok

# ---------- URL normalizacija ----------
def normalize_site_url(raw: str) -> str:
    """
    - zagotovi, da pot vsebuje '/si/' (če manjka na začetku),
    - odstrani podvojeni segment '/stalna-zbirka/stalna-zbirka/',
    - naredi absolutni URL pod https://www.ng-slo.si
    """
    if not raw: return raw
    if re.match(r"^https?://", raw, flags=re.I):
        url = raw
    else:
        # absolutna pot
        if raw.startswith("/"):
            url = BASE + raw
        else:
            # relativna pot, predpostavi '/si/'
            if raw.startswith("si/"):
                url = f"{BASE}/{raw}"
            elif raw.startswith("stalna-zbirka/"):
                url = f"{BASE}/si/{raw}"
            else:
                url = f"{BASE}/si/{raw.lstrip('/')}"
    # popravi dvojni 'stalna-zbirka'
    url = url.replace("/stalna-zbirka/stalna-zbirka/", "/stalna-zbirka/")
    # če je '/stalna-zbirka' brez '/si', ga vstavi
    url = url.replace(f"{BASE}/stalna-zbirka/", f"{BASE}/si/stalna-zbirka/")
    return url

def absolutize_media_url(page_url: str, raw_url: str) -> str:
    if not raw_url: return raw_url
    raw_url = raw_url.strip()
    if re.match(r"^https?://", raw_url, flags=re.I):
        # popravi zlepljeno varianto
        if "/stalna-zbirka/" in raw_url and "/imagelib/" in raw_url:
            fixed = re.sub(r"/si/stalna-zbirka/[^/]+/(.*?/)?(?=imagelib/)", "", raw_url)
            fixed = re.sub(r"https?://www\.ng-slo\.si/(?:si/)?(?:.*?/)?(?=imagelib/)", MEDIA_PREFIX, fixed)
            return fixed
        return raw_url
    if raw_url.startswith("/si/imagelib/"): return urljoin(BASE, raw_url)
    if raw_url.startswith("imagelib/"): return MEDIA_PREFIX + raw_url
    joined = urljoin(page_url, raw_url)
    if "/stalna-zbirka/" in joined and "/imagelib/" in joined:
        fixed = re.sub(r"/si/stalna-zbirka/[^/]+/(.*?/)?(?=imagelib/)", "", joined)
        fixed = re.sub(r"https?://www\.ng-slo\.si/(?:si/)?(?:.*?/)?(?=imagelib/)", MEDIA_PREFIX, fixed)
        return fixed
    return joined

# ---------- galerija ----------
def extract_gallery_item_links(sess, period_url: str) -> list[str]:
    soup = get_soup(sess, period_url)
    links=set()
    for a in soup.select(".collections-gallery a[href], .gallery a[href], .items a[href], .grid a[href], a[href*='/stalna-zbirka/']"):
        href = a.get("href") or ""
        if not href: continue
        full = urljoin(period_url, href)
        # normaliziraj in filtriraj
        full = normalize_site_url(full)
        if full.rstrip("/") == period_url.rstrip("/") or full.endswith(("#","/#")):
            continue
        if "/stalna-zbirka/" in full:
            links.add(full)
    for a in soup.select("a[href*='workId=']"):
        links.add(normalize_site_url(a.get("href", "")))
    return sorted(links)

# ---------- parsanje detajlov ----------
def parse_author_block(soup: BeautifulSoup) -> dict:
    head = soup.select_one(".collections-details-rightBlockHead")
    author_name = author_url = author_bio = None
    if head:
        a = head.select_one("h2 a")
        if a:
            author_name = clean_text(a.get_text(" "))
            if a.get("href"):
                author_url = normalize_site_url(a["href"])
        h3 = head.find("h3")
        if h3:
            author_bio = clean_text(h3.get_text(" "))
    return {"author_name": author_name or None, "author_url": author_url or None, "author_bio": author_bio or None}

def _desc_lines_from_block(desc_div: Tag) -> list[str]:
    """
    Preberi vrstice iz .collections-details-imageDescription z upoštevanjem <br>.
    Brez HTML-a, samo čisti tekst.
    """
    lines=[]; buf=[]
    def flush():
        nonlocal buf
        txt = clean_text("".join(buf))
        if txt: lines.append(txt)
        buf=[]
    for node in desc_div.children:
        if isinstance(node, NavigableString):
            buf.append(str(node))
        elif isinstance(node, Tag):
            if node.name.lower()=="br":
                flush()
            else:
                buf.append(node.get_text(" "))
    flush()
    # odstrani prazne
    return [ln for ln in lines if ln]

def parse_left_block(soup: BeautifulSoup, page_url: str) -> dict:
    left = soup.select_one(".collections-details-leftBlock")
    images=[]; work_title=None; work_meta_line=None; work_inventory_line=None
    desc_html=desc_text=None
    if left:
        for a in left.select(".image a[href]"):
            images.append(absolutize_media_url(page_url, a.get("href")))
        for img in left.select("img[data-original], img[src]"):
            raw = img.get("data-original") or img.get("src")
            if raw: images.append(absolutize_media_url(page_url, raw))
        desc_div = left.select_one(".collections-details-imageDescription")
        if desc_div:
            desc_html = str(desc_div)
            desc_text = clean_text(desc_div.get_text(" "))
            # naslov v <b>
            b = desc_div.find("b")
            if b: work_title = clean_text(b.get_text(" "))
            # vrstice po <br>
            lines = _desc_lines_from_block(desc_div)
            # odstrani vrstico, ki je identična naslovu
            lines_wo_title = [ln for ln in lines if not (work_title and ln == work_title)]
            if lines_wo_title:
                work_meta_line = lines_wo_title[0]
                if len(lines_wo_title) > 1:
                    work_inventory_line = lines_wo_title[1]
    images = sorted({u for u in images if u})
    return {
        "work_images": images,
        "work_title": work_title or None,
        "work_meta_line": work_meta_line or None,
        "work_inventory_line": work_inventory_line or None,
        "work_image_description_html": desc_html,
        "work_image_description_text": desc_text,
    }

def parse_description_block(soup: BeautifulSoup) -> dict:
    d = soup.select_one(".collections-details-rightBlock-text")
    if not d: return {"description_html": None, "description_text": None}
    return {"description_html": str(d), "description_text": clean_text(d.get_text(" "))}

def extract_work_id(url: str) -> str | None:
    try:
        q = parse_qs(urlparse(url).query); vals = q.get("workId")
        if vals: return vals[0]
    except Exception: pass
    return None

def scrape_detail_page(sess, period_label: str, period_url: str, detail_url: str) -> dict:
    detail_url = normalize_site_url(detail_url)  # <-- ključni fix
    soup = get_soup(sess, detail_url)
    author = parse_author_block(soup)
    left = parse_left_block(soup, detail_url)
    desc = parse_description_block(soup)
    h = soup.find(["h1","h2"])
    page_heading = clean_text(h.get_text(" ")) if h else None
    return {
        "period_label": period_label,
        "period_url": period_url,
        "detail_url": detail_url,
        "page_heading": page_heading,
        "work_id": extract_work_id(detail_url),
        **author, **left, **desc,
    }

def main():
    sess = session_with_retries()
    resolved=[]
    print("→ Resolving period URLs …")
    for label in PERIOD_LABELS:
        url = resolve_period_url(sess, label)
        print(f"  {label}: {url or 'NI NAJDENO'}")
        if url: resolved.append((label, url))
        time.sleep(REQUEST_DELAY)
    if not resolved: raise SystemExit("Noben URL za obdobja ni bil uspešno razrešen.")

    all_records=[]
    for i,(label,purl) in enumerate(resolved,1):
        print(f"[{i}/{len(resolved)}] {label}")
        try:
            item_links = extract_gallery_item_links(sess, purl)
        except Exception as e:
            print(f"  [warn] galerija: {e}"); item_links=[]
        print(f"  → galerijskih povezav: {len(item_links)}")
        for j,durl in enumerate(item_links,1):
            try:
                rec = scrape_detail_page(sess, label, purl, durl)
                all_records.append(rec)
            except Exception as e:
                all_records.append({
                    "period_label": label, "period_url": purl,
                    "detail_url": normalize_site_url(durl), "error": str(e),
                })
            if j % 10 == 0: print(f"    … {j}/{len(item_links)}")
            time.sleep(REQUEST_DELAY)
        time.sleep(REQUEST_DELAY)

    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    print(f"✓ Shranjeno {len(all_records)} zapisov v {OUTFILE}")

if __name__ == "__main__":
    main()
