#!/usr/bin/env python3
# save_random_images.py
# Run:  python save_random_images.py

import json, pathlib, mimetypes, subprocess, sys, random
from urllib.parse import urlparse, urljoin
from typing import Optional, List

# ========= CONFIG =========
JSON_FILE     = "slovenia_si_art_kultura.json"       # input JSON (list of objects, or a single object)
OUTPUT_DIR    = "SLO_images"      # output folder
START_INDEX   = 835               # first index to assign
SAMPLE_SIZE   = 250               # how many random entries to attempt
TIMEOUT_SECS  = 30
RANDOM_SEED   = None              # set e.g. 42 for reproducible sampling
IMAGE_KEY     = "image_url"       # primary image field key to use
# ==========================

# Ensure 'requests' is available
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

def make_session() -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=[403, 408, 429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    sess.mount("http://", HTTPAdapter(max_retries=retry))
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    sess.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/127 Safari/537.36 RandomImageSaver/1.0",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": "https://slovenia.si/",
    })
    return sess

def ensure_list(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError("JSON must be a list of objects or a single object.")

def get_base_url(item: dict) -> Optional[str]:
    # Use article_url as base for relative image paths, if present
    a = item.get("article_url")
    if isinstance(a, str) and a.startswith(("http://", "https://")):
        parts = urlparse(a)
        return f"{parts.scheme}://{parts.netloc}"
    return None

def build_image_url(item: dict) -> str:
    v = item.get(IMAGE_KEY)
    if not isinstance(v, str) or not v:
        raise KeyError(f"No '{IMAGE_KEY}' field or empty.")
    if v.startswith(("http://", "https://")):
        return v
    # relative path -> join with base (prefer article_url’s domain)
    base = get_base_url(item) or "https://slovenia.si"
    return urljoin(base, v)

def guess_ext_from_url(url: str) -> Optional[str]:
    p = pathlib.PurePosixPath(url.split("?", 1)[0])
    ext = p.suffix.lower()
    if ext in (".jpeg", ".jpe"):
        return ".jpg"
    return ext or None

def guess_ext_from_headers(content_type: Optional[str]) -> Optional[str]:
    if not content_type:
        return None
    ct = content_type.split(";")[0].strip().lower()
    ext = mimetypes.guess_extension(ct)
    if ext in (".jpeg", ".jpe"):
        return ".jpg"
    return ext

def sniff_ext_from_bytes(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    return ".jpg"

def main():
    in_path = pathlib.Path(JSON_FILE)
    out_dir = pathlib.Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = ensure_list(data)
    n = len(items)
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    # Choose up to SAMPLE_SIZE distinct random indices
    sample_indices = random.sample(range(n), k=min(SAMPLE_SIZE, n))

    sess = make_session()

    results = []          # only successful entries will be appended
    next_index = START_INDEX
    failures = 0
    attempted = 0

    for i in sample_indices:
        attempted += 1
        item = items[i]
        try:
            img_url = build_image_url(item)
            resp = sess.get(img_url, timeout=TIMEOUT_SECS)
            if not resp.ok:
                # skip this one entirely (no indexing)
                failures += 1
                continue

            content = resp.content
            ext = (
                guess_ext_from_url(img_url)
                or guess_ext_from_headers(resp.headers.get("Content-Type"))
                or sniff_ext_from_bytes(content)
            )

            filename = f"{next_index:05d}{ext}"
            (out_dir / filename).write_bytes(content)

            rec = dict(item)
            rec["index"] = next_index
            rec["saved_image"] = str(out_dir / filename)
            results.append(rec)

            next_index += 1  # increment ONLY for successful saves

        except Exception:
            failures += 1
            # do not index, do not save, just skip

    # Write only successful ones to indexed.json (always as a list)
    out_json = in_path.with_name("indexed.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Tried: {attempted} | Saved: {len(results)} | Skipped (errors): {failures}")
    print(f"Saved images to: {out_dir.resolve()}")
    print(f"Wrote indexed JSON to: {out_json.resolve()}")

if __name__ == "__main__":
    main()
