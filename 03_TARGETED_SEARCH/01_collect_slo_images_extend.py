import os
import re
import io
import json
import time
import base64
import random
import hashlib
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Set
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup
from PIL import Image

from google import genai
from google.genai import types


# ==================================================
# CONFIG
# ==================================================
TAXONOMY_PATH = "ocm-slo-clean.json"
OUT_DIR = "SLO_images"
OUT_JSONL = "slo_images_manifest.jsonl"
START_ID = 20000

PROMPTS_PER_NODE = 25
TARGET_GOOD_PER_PROMPT = 2

# Bing scraping
BING_COUNT = 250
BING_PAGE_STEP = 30
SLEEP_BING = 0.1
MAX_BING_RETRIES = 2

# Gemini
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_EMPTY_RETRIES = 4
GEMINI_ROUND_MAX = 6

# Ollama / Gemma
OLLAMA_PORT = ""
OLLAMA_URL = os.getenv("OLLAMA_URL", OLLAMA_PORT )
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:27b")
SLEEP_BETWEEN_GEMMA_CALLS = 0.0

# Accept only strict good
ACCEPT_LABELS = {"good"}

# Verbose printing
DEBUG = True
PRINT_URLS = True
PRINT_URLS_N = 10
PRINT_SKIP_REASONS = True
PRINT_PROMPTS = True
PRINT_GEMMA_RAW = False

NEGATIVE_CONTROL_EVERY = 0


# ==================================================
# KEYS
# ==================================================
GEMINI_API_KEY = GEMINI_API_KEY
BING_SEARCH_KEY = os.getenv("BING_SEARCH_KEY", "")
BING_SEARCH_ENDPOINT = os.getenv("BING_SEARCH_ENDPOINT", "https://api.bing.microsoft.com")


# ==================================================
# PRINT HELPERS
# ==================================================
def log(msg: str) -> None:
    print(msg, flush=True)

def dbg(msg: str) -> None:
    if DEBUG:
        print(f"[DEBUG] {msg}", flush=True)

def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)

def skip(msg: str) -> None:
    if PRINT_SKIP_REASONS:
        print(f"[SKIP] {msg}", flush=True)


# ==================================================
# UTIL
# ==================================================
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def safe_filename(idx: int) -> str:
    return f"{idx}.jpg"

def encode_image_bytes_to_base64(jpg_bytes: bytes) -> str:
    return base64.b64encode(jpg_bytes).decode("utf-8")


# ==================================================
# TAXONOMY
# ==================================================
def load_taxonomy(path: str) -> Dict[str, Any]:
    dbg(f"Loading taxonomy from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        tax = json.load(f)
    dbg(f"Taxonomy loaded. Roots={len(list(tax.keys()))}")
    return tax

def children_of(node: Any) -> List[str]:
    return list(node.keys()) if isinstance(node, dict) else []

def get_node_by_path(tax: Dict[str, Any], path: List[str]) -> Any:
    cur = tax
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

def bfs_all_category_paths(tax: Dict[str, Any]) -> List[List[str]]:
    all_paths = []
    q = [[root] for root in tax.keys()]
    while q:
        p = q.pop(0)
        all_paths.append(p)
        node = get_node_by_path(tax, p)
        for c in children_of(node):
            q.append(p + [c])
    return all_paths

def path_to_category_str(path: List[str]) -> str:
    return " > ".join(path)


# ==================================================
# GEMINI PROMPTS
# ==================================================
@dataclass
class PromptItem:
    prompt: str
    description: str

PROMPT_GEN_SYSTEM = """
Si generator iskalnih pozivov (promptov) v slovenščini za zbiranje fotografij, kulturno povezanih s Slovenijo.
Pozivi naj bodo konkretni in primerni za Bing Image Search (realne fotografije).
Izogibaj se politični propagandi, logotipom in avtorsko zaščitenim likom.
Vrni SAMO veljaven JSON (brez markdowna, brez ``` ograj, brez razlage).
""".strip()

def _extract_text_from_genai_response(resp) -> str:
    t = getattr(resp, "text", None)
    if isinstance(t, str) and t.strip():
        return t
    try:
        cands = getattr(resp, "candidates", None) or []
        if not cands:
            return ""
        c0 = cands[0]
        content = getattr(c0, "content", None)
        parts = getattr(content, "parts", None) or []
        chunks = []
        for p in parts:
            pt = getattr(p, "text", None)
            if isinstance(pt, str) and pt.strip():
                chunks.append(pt)
        return "\n".join(chunks).strip()
    except Exception:
        return ""

def _safe_json_load(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    s2 = re.sub(r"^\s*```(?:json)?\s*", "", s.strip(), flags=re.IGNORECASE)
    s2 = re.sub(r"\s*```\s*$", "", s2).strip()
    try:
        return json.loads(s2)
    except Exception:
        return None

def _gemini_generate_once(client: genai.Client, category_label: str, n: int, avoid: List[str]) -> List[Dict[str, str]]:
    avoid_block = ""
    if avoid:
        sample = avoid[:60]
        avoid_block = "\n\nNE PONAVLJAJ (ali zelo podobno) teh promptov:\n- " + "\n- ".join(sample)

    user = f"""
Kategorija: {category_label}

Naloga:
Ustvari {n} različnih iskalnih pozivov (promptov) v slovenščini, kulturno povezanih s Slovenijo.
Vsak prompt naj bo primeren za iskanje fotografij (kraj/dogodek/predmet/običaj).

Izhodna JSON shema:
{{
  "items": [
    {{
      "prompt": "slovenski iskalni niz",
      "description": "2–4 stavki v slovenščini: kaj prompt predstavlja in kaj pomeni za Slovenijo."
    }}
  ]
}}

Pomembno:
- Generiraj prompte, ki so povezani s kategorijo.
- Ne uporabljaj dejanskih novih vrstic znotraj description (naj bo en odstavek).
- Vrni SAMO JSON.
{avoid_block}
""".strip()

    for attempt in range(1, GEMINI_EMPTY_RETRIES + 1):
        t0 = time.time()
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[PROMPT_GEN_SYSTEM, user],
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=65536,
                response_mime_type="application/json",
            ),
        )
        dt = time.time() - t0
        raw = _extract_text_from_genai_response(resp)
        dbg(f"Gemini attempt {attempt}/{GEMINI_EMPTY_RETRIES} in {dt:.2f}s, chars={len(raw)}")

        if not raw.strip():
            time.sleep(0.8 * attempt)
            continue

        data = _safe_json_load(raw)
        if not data or "items" not in data:
            warn("Gemini output not parseable as JSON.")
            if DEBUG:
                dbg("RAW HEAD:\n" + raw[:400])
            return []

        out = []
        for it in (data.get("items") or []):
            p = normalize_text(it.get("prompt", ""))
            d = normalize_text(it.get("description", ""))
            if p and d:
                out.append({"prompt": p, "description": d})
        return out

    warn("Gemini returned empty extracted text after retries.")
    return []

def gemini_generate_prompts(client: genai.Client, category_label: str, target_n: int = PROMPTS_PER_NODE) -> List[PromptItem]:
    dbg(f"Gemini -> generate prompts for: {category_label} target={target_n}")

    got: List[PromptItem] = []
    seen = set()

    for round_idx in range(1, GEMINI_ROUND_MAX + 1):
        remaining = target_n - len(got)
        if remaining <= 0:
            break

        batch = _gemini_generate_once(client, category_label, remaining, [x.prompt for x in got])

        added = 0
        for it in batch:
            key = it["prompt"].lower()
            if key in seen:
                continue
            seen.add(key)
            got.append(PromptItem(prompt=it["prompt"], description=it["description"]))
            added += 1
            if len(got) >= target_n:
                break

        dbg(f"Round {round_idx}: batch={len(batch)} added={added} total={len(got)}/{target_n}")

        if added == 0 and round_idx >= 2:
            warn("Gemini not adding new prompts; stopping early.")
            break

    if len(got) < target_n:
        warn(f"Only obtained {len(got)}/{target_n} prompts for '{category_label}'.")

    if PRINT_PROMPTS:
        log(f"--- PROMPTS for category: {category_label} ({len(got)}/{target_n}) ---")
        for i, it in enumerate(got, 1):
            log(f"  [{i:02d}] {it.prompt}")

    return got


# ==================================================
# BING SEARCH
# ==================================================
def get_bing_image_urls(query: str, limit: int) -> List[str]:
    log(f"    🔍 Bing search: '{query}'")
    headers = {"User-Agent": "Mozilla/5.0"}
    urls: List[str] = []

    for start in range(1, limit + 120, BING_PAGE_STEP):
        if len(urls) >= limit:
            break

        q = quote_plus(query)
        url = f"https://www.bing.com/images/search?q={q}&first={start}"

        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            found_this_page = 0
            for a in soup.find_all("a", class_="iusc"):
                if len(urls) >= limit:
                    break
                try:
                    m = json.loads(a.get("m", "{}"))
                    u = m.get("murl")
                    if u and u not in urls:
                        urls.append(u)
                        found_this_page += 1
                except Exception:
                    continue

            dbg(f"      page first={start}: +{found_this_page} new urls (total={len(urls)})")
            time.sleep(SLEEP_BING)

        except Exception as e:
            warn(f"      ⚠️ Bing error: {e}")
            break

    log(f"    ↳ found {len(urls)} urls")
    if PRINT_URLS and DEBUG and urls:
        for i, u in enumerate(urls[:PRINT_URLS_N], 1):
            dbg(f"      url[{i}]: {u}")
        if len(urls) > PRINT_URLS_N:
            dbg(f"      ... (+{len(urls)-PRINT_URLS_N} more)")
    return urls


# ==================================================
# IMAGE DOWNLOAD
# ==================================================
def download_jpg(url: str) -> Optional[bytes]:
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))

        if img.mode == "P":
            img = img.convert("RGBA")
        if img.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")

        w, h = img.size
        if min(w, h) < 256:
            skip(f"image too small: {w}x{h}")
            return None

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return buf.getvalue()

    except Exception as e:
        skip(f"download_jpg failed: {type(e).__name__}: {e}")
        return None


# ==================================================
# GEMMA JUDGE
# ==================================================
GEMMA_JUDGE_SYSTEM = """
You are a strict image relevance judge for dataset collection.

You will be given:
- category in slovene
- Slovene prompt
- Slovene commentary describing what the prompt should depict / why it matters in Slovenia
- an image

Return ONLY valid JSON:
{"label":"good|medium|bad","reason":"one short sentence"}

Guidelines:
- good: clearly matches BOTH prompt and category; looks like a real photograph of the described thing/place/event/object.
- medium: partially related/ambiguous/too generic/wrong context.
- bad: unrelated, wrong country/culture, meme/graphic/screenshot, product listing, lots of text, illustration instead of photo.

Hard rules:
- IMPORTANT: Reject if category does not match the photo even if prompt and comentary match => bad
- posters/memes/screenshots/infographics or lots of text => bad
- product listing photos (marketplace) => medium or bad (prefer bad if mostly product shot)
""".strip()

def gemma_judge_image(category: str, prompt: str, description: str, jpg_bytes: bytes) -> Tuple[str, str]:
    img_b64 = encode_image_bytes_to_base64(jpg_bytes)

    user = f"""Category: {category}
Prompt (SL): {prompt}
Prompt commentary (SL): {description}

Return JSON: {{"label":"good|medium|bad","reason":"..."}}.
"""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": GEMMA_JUDGE_SYSTEM},
            {"role": "user", "content": user, "images": [img_b64]},
        ],
        "stream": False,
    }

    try:
        t0 = time.time()
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        js = r.json()
        dt = time.time() - t0

        raw = (js.get("message", {}).get("content") or "").strip()
        if PRINT_GEMMA_RAW and DEBUG:
            dbg(f"Gemma raw='{raw}' ({dt:.2f}s)")

        raw2 = re.sub(r"^\s*```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
        raw2 = re.sub(r"\s*```\s*$", "", raw2).strip()

        try:
            obj = json.loads(raw2)
            label = (obj.get("label") or "").strip().lower()
            reason = normalize_text(obj.get("reason", ""))
        except Exception:
            m = re.search(r"\b(good|medium|bad)\b", raw.lower())
            label = m.group(1) if m else "bad"
            reason = "non-json response"

        if label not in ("good", "medium", "bad"):
            label = "bad"
        if not reason:
            reason = "no reason"

        dbg(f"Gemma label={label} ({dt:.2f}s) reason={reason}")
        return label, reason

    except Exception as e:
        warn(f"Gemma judge failed: {type(e).__name__}: {e} -> bad")
        return "bad", "judge failed"


# ==================================================
# MANIFEST
# ==================================================
def load_existing_manifest(path: str) -> Tuple[Set[str], int, Set[str]]:
    seen_url_hashes = set()
    done_categories = set()
    max_id = START_ID - 1

    if not os.path.exists(path):
        dbg("No existing manifest found; starting fresh.")
        return seen_url_hashes, START_ID, done_categories

    dbg(f"Loading existing manifest: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)

                u = obj.get("source_url", "")
                if u:
                    seen_url_hashes.add(sha1(u))

                i = obj.get("id")
                if isinstance(i, int):
                    max_id = max(max_id, i)

                cat = normalize_text(obj.get("category", ""))
                if cat:
                    done_categories.add(cat)

            except Exception:
                continue

    next_id = max(max_id + 1, START_ID)
    dbg(
        f"Manifest seen_urls={len(seen_url_hashes)}, "
        f"done_categories={len(done_categories)}, max_id={max_id}, next_id={next_id}"
    )
    return seen_url_hashes, next_id, done_categories

def append_manifest(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


# ==================================================
# COLLECTION FOR ONE NEW CATEGORY
# ==================================================
def collect_for_new_category(
    category_label: str,
    gemini_client: genai.Client,
    seen_url_hashes: Set[str],
    next_id: int,
) -> int:
    ensure_dir(OUT_DIR)

    log("\n==============================")
    log(f"[NEW CATEGORY] {category_label}")
    log(f"[PLAN] Will try to collect {TARGET_GOOD_PER_PROMPT} GOOD images per prompt")
    log("==============================")

    subcategory_label = category_label.split(">")[-1].strip()
    prompts = gemini_generate_prompts(gemini_client, subcategory_label, target_n=PROMPTS_PER_NODE)
    if not prompts:
        warn(f"No prompts generated -> skip category: {category_label}")
        return next_id

    random.shuffle(prompts)

    for p_idx, item in enumerate(prompts, 1):
        log(f"\n[PROMPT {p_idx}/{len(prompts)}] {item.prompt}")
        dbg(f"Opis: {item.description}")

        kept_for_prompt = 0

        urls: List[str] = []
        for attempt in range(1, MAX_BING_RETRIES + 1):
            try:
                dbg(f"Scrape Bing attempt {attempt}/{MAX_BING_RETRIES} ...")
                urls = get_bing_image_urls(item.prompt, limit=BING_COUNT)
                break
            except Exception as e:
                warn(f"Bing scrape failed (attempt {attempt}): {type(e).__name__}: {e}")
                time.sleep(1.0)

        if not urls:
            warn("No Bing URLs -> skipping prompt")
            continue

        for u_idx, url in enumerate(urls, 1):
            if kept_for_prompt >= TARGET_GOOD_PER_PROMPT:
                break

            uh = sha1(url)
            if uh in seen_url_hashes:
                skip(f"already seen url (hash) url_idx={u_idx}")
                continue

            dbg(f"[URL {u_idx}/{len(urls)}] downloading...")
            jpg = download_jpg(url)
            if not jpg:
                continue

            dbg("Judging with Gemma ...")
            label, reason = gemma_judge_image(category_label, item.prompt, item.description, jpg)
            time.sleep(SLEEP_BETWEEN_GEMMA_CALLS)

            if label not in ACCEPT_LABELS:
                skip(f"gemma label '{label}' not accepted reason={reason}")
                continue

            img_id = next_id
            next_id += 1

            local_path = os.path.join(OUT_DIR, safe_filename(img_id))
            with open(local_path, "wb") as f:
                f.write(jpg)

            append_manifest(OUT_JSONL, {
                "id": img_id,
                "source_url": url,
                "local_path": local_path.replace("\\", "/"),
                "category": category_label,
                "description": item.description,
                "prompt": item.prompt,
                "gemma_label": label,
                "gemma_reason": reason,
            })

            seen_url_hashes.add(uh)
            kept_for_prompt += 1

            log(
                f"[SAVED] id={img_id} label={label} reason={reason} "
                f"kept_for_prompt={kept_for_prompt}/{TARGET_GOOD_PER_PROMPT} path={local_path}"
            )

        log(
            f"[PROMPT DONE] kept {kept_for_prompt}/{TARGET_GOOD_PER_PROMPT} good images "
            f"for prompt: {item.prompt}"
        )

    log(f"[DONE CATEGORY] {category_label}")
    return next_id


# ==================================================
# MAIN
# ==================================================
def main():
    log("========== SLO IMAGE COLLECTION EXTENSION START ==========")

    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY env var.")

    log(f"Gemini model: {GEMINI_MODEL}")
    log(f"Ollama URL: {OLLAMA_URL}")
    log(f"Ollama model: {OLLAMA_MODEL}")
    log(f"Output dir: {OUT_DIR}")
    log(f"Manifest: {OUT_JSONL}")
    log(f"Start ID: {START_ID}")

    ensure_dir(OUT_DIR)
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    tax = load_taxonomy(TAXONOMY_PATH)
    all_paths = bfs_all_category_paths(tax)
    all_categories = [path_to_category_str(p) for p in all_paths]

    log(f"Total taxonomy nodes: {len(all_categories)}")

    seen_url_hashes, next_id, done_categories = load_existing_manifest(OUT_JSONL)
    log(f"Resume: next_id={next_id}, seen_urls={len(seen_url_hashes)}")
    log(f"Already done categories/subcategories in manifest: {len(done_categories)}")

    remaining_categories = [c for c in all_categories if c not in done_categories]
    log(f"Remaining categories/subcategories to extend: {len(remaining_categories)}")

    for idx, category_label in enumerate(remaining_categories, 1):
        log("\n\n#############################################")
        log(f"[CATEGORY {idx}/{len(remaining_categories)}] {category_label}")
        log("#############################################")
        next_id = collect_for_new_category(
            category_label=category_label,
            gemini_client=gemini_client,
            seen_url_hashes=seen_url_hashes,
            next_id=next_id,
        )

    log("\n========== EXTENSION DONE ==========")
    log(f"Images folder: {OUT_DIR}")
    log(f"Manifest JSONL: {OUT_JSONL}")


if __name__ == "__main__":
    main()
