from __future__ import annotations

import os
import json
import time
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# pip install google-generativeai
import google.generativeai as genai


# =========================
# Config (edit if needed)
# =========================
TRAIN_JSON = Path("fine_tune_train.json")
TEST_JSON  = Path("fine_tune_test.json")
OCM_JSON   = Path("ocm-slo-clean.json")

# If your JSON references local images like "SLO_images\\00001.JPG",
# set this base directory so (LOCAL_IMAGE_BASE / that_path) exists.
LOCAL_IMAGE_BASE = Path("")

# Gemini model name (must be enabled for your API key)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Cache to avoid repeated calls for same image
CACHE_PATH = Path("ocm-slo-clean.json")

# Networking / retries
HTTP_TIMEOUT = 30
MAX_RETRIES = 5
SLEEP_BETWEEN_CALLS_SEC = 0.2  # throttle a bit

# Output field added to each item in both JSONs
OUT_FIELD = "ocm_top_categories"


# =========================
# JSON helpers
# =========================
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def backup_file(path: Path) -> Path:
    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    return bak


# =========================
# Category extraction
# =========================
def extract_top_level_categories(ocm_obj: Dict[str, Any]) -> List[str]:
    # Top-level categories = top-level keys
    return sorted(list(ocm_obj.keys()))

def build_slovenian_prompt(top_categories: List[str]) -> str:
    cats = json.dumps(top_categories, ensure_ascii=False)
    return (
        "Opravi klasifikacijo slike po kategorijah OCM.\n\n"
        "Navodila:\n"
        "1) Izberi VSE ustrezne kategorije iz spodnjega seznama.\n"
        "2) Uporabi IZKLJUČNO kategorije iz seznama (ne izmišljaj novih).\n"
        "3) Če nobena kategorija ne ustreza, vrni prazen seznam [].\n"
        "4) Odgovori IZKLJUČNO kot veljaven JSON seznam nizov, npr. "
        "[\"geografija\", \"zgodovina in kulturne spremembe\"].\n\n"
        f"Seznam dovoljenih (vrhnjih) kategorij:\n{cats}\n"
    )

def parse_categories_from_response(text: str, allowed: set) -> List[str]:
    """
    Expects JSON list of strings. Strictly filters to allowed categories.
    """
    text = (text or "").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract first bracketed list
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Response is not JSON list: {text[:200]}")
        obj = json.loads(text[start:end + 1])

    if not isinstance(obj, list):
        raise ValueError(f"Expected JSON list, got: {type(obj)}")

    cleaned: List[str] = []
    for x in obj:
        if isinstance(x, str):
            x2 = x.strip()
            if x2 in allowed and x2 not in cleaned:
                cleaned.append(x2)
    return cleaned


# =========================
# Image loading
# =========================
def normalize_local_image_path(p: Any) -> Optional[Path]:
    """
    Handles cases where 'image' is string or list like ["SLO_images\\00003.jpg"].
    Returns absolute Path under LOCAL_IMAGE_BASE if possible.
    """
    if p is None:
        return None
    if isinstance(p, list) and len(p) > 0:
        p = p[0]
    if not isinstance(p, str) or not p.strip():
        return None

    p_norm = p.replace("\\", "/")
    return (LOCAL_IMAGE_BASE / p_norm)

def load_image_from_url(url: str) -> Image.Image:
    r = requests.get(url, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")

def load_image_from_path(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


# =========================
# Cache
# =========================
def cache_load() -> Dict[str, List[str]]:
    if CACHE_PATH.exists():
        try:
            data = load_json(CACHE_PATH)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}

def cache_save(cache: Dict[str, List[str]]) -> None:
    save_json(CACHE_PATH, cache)


# =========================
# Gemini call
# =========================
def gemini_classify_image(model: genai.GenerativeModel, prompt: str, img: Image.Image) -> str:
    resp = model.generate_content([prompt, img])
    return (resp.text or "").strip()


# =========================
# Logging helpers
# =========================
def build_item_label(item: Dict[str, Any]) -> str:
    # Try common fields; fall back to a short hash-like representation
    parts = []
    for k in ["id", "subquestion", "question_id", "name", "uid"]:
        if k in item and item.get(k) not in (None, ""):
            parts.append(f"{k}={item.get(k)}")
    return " ".join(parts) if parts else "item"

def build_img_ref(image_url: Optional[str], local_path: Optional[Path]) -> str:
    if isinstance(image_url, str) and image_url.strip():
        return image_url.strip()
    if local_path is not None:
        return str(local_path)
    return "(no image)"


# =========================
# Main processing
# =========================
def process_file(
    json_path: Path,
    model: genai.GenerativeModel,
    prompt: str,
    allowed_categories: set,
    cache: Dict[str, List[str]],
) -> Tuple[int, int]:
    """
    LOCAL-ONLY version:
    - Ignores any image_url
    - Loads images only from item["image"] (string or [string]) under LOCAL_IMAGE_BASE
    - Writes OUT_FIELD with chosen top-level categories
    - Prints chosen categories for each image
    - Uses cache by local path

    Returns: (updated_count, skipped_count)
    """
    data = load_json(json_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {json_path}, got {type(data)}")

    updated = 0
    skipped = 0

    for item in tqdm(data, desc=f"Processing {json_path.name}"):
        if not isinstance(item, dict):
            skipped += 1
            continue

        label = build_item_label(item)

        # Resolve local image path from item["image"]
        local_img_path = normalize_local_image_path(item.get("image"))
        img_ref = str(local_img_path) if local_img_path is not None else "(no image)"

        # If already has categories, skip but print
        if OUT_FIELD in item and isinstance(item[OUT_FIELD], list) and len(item[OUT_FIELD]) > 0:
            print(f"[{json_path.name}] {label} | image={img_ref} -> categories={item[OUT_FIELD]} (ALREADY PRESENT)")
            skipped += 1
            continue

        # Cache key (local path)
        cache_key = f"path:{str(local_img_path)}" if local_img_path is not None else None

        # Cache hit
        if cache_key and cache_key in cache:
            cats = cache[cache_key]
            item[OUT_FIELD] = cats
            print(f"[{json_path.name}] {label} | image={img_ref} -> categories={cats} (CACHED)")
            updated += 1
            continue

        # No path available
        if local_img_path is None:
            item[OUT_FIELD] = []
            if cache_key:
                cache[cache_key] = []
            print(f"[{json_path.name}] {label} | image={img_ref} -> categories=[] (NO PATH)")
            updated += 1
            continue

        # File missing
        if not local_img_path.exists():
            item[OUT_FIELD] = []
            if cache_key:
                cache[cache_key] = []
            print(f"[{json_path.name}] {label} | image={img_ref} -> categories=[] (FILE NOT FOUND)")
            updated += 1
            continue

        # Load local image
        try:
            img = load_image_from_path(local_img_path)
        except Exception as e:
            item[OUT_FIELD] = []
            if cache_key:
                cache[cache_key] = []
            print(f"[{json_path.name}] {label} | image={img_ref} -> categories=[] (LOAD FAILED: {type(e).__name__}: {e})")
            updated += 1
            continue

        # Call Gemini with retries
        last_err = None
        cats: List[str] = []
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                time.sleep(SLEEP_BETWEEN_CALLS_SEC)
                text = gemini_classify_image(model, prompt, img)
                cats = parse_categories_from_response(text, allowed_categories)
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 20))

        if last_err is not None:
            cats = []
            print(f"[{json_path.name}] {label} | image={img_ref} -> categories=[] (GEMINI FAILED: {type(last_err).__name__}: {last_err})")
        else:
            print(f"[{json_path.name}] {label} | image={img_ref} -> categories={cats}")

        item[OUT_FIELD] = cats
        if cache_key:
            cache[cache_key] = cats
        updated += 1

    bak = backup_file(json_path)
    save_json(json_path, data)
    print(f"[{json_path.name}] saved. Backup: {bak.name}")
    return updated, skipped



def main():
    api_key = ""
    if not api_key:
        raise RuntimeError(
            "Missing GOOGLE_API_KEY environment variable.\n"
            "Set it like: export GOOGLE_API_KEY='...'\n"
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    ocm = load_json(OCM_JSON)
    if not isinstance(ocm, dict):
        raise ValueError(f"Expected dict in {OCM_JSON}, got {type(ocm)}")

    top_categories = extract_top_level_categories(ocm)
    allowed = set(top_categories)
    prompt = build_slovenian_prompt(top_categories)

    cache = cache_load()

    print(f"Using Gemini model: {GEMINI_MODEL}")
    print(f"Top-level categories count: {len(top_categories)}")
    print(f"Cache entries: {len(cache)}")
    print(f"Writing field: {OUT_FIELD}")
    print("-" * 60)

    upd1, skip1 = process_file(TRAIN_JSON, model, prompt, allowed, cache)
    cache_save(cache)
    print(f"[TRAIN] updated={upd1} skipped={skip1} cache={len(cache)}")
    print("-" * 60)

    upd2, skip2 = process_file(TEST_JSON, model, prompt, allowed, cache)
    cache_save(cache)
    print(f"[TEST ] updated={upd2} skipped={skip2} cache={len(cache)}")
    print("-" * 60)

    print("Done.")
    print("Backups saved as .bak next to original JSON files.")
    print(f"Cache saved to: {CACHE_PATH}")


if __name__ == "__main__":
    main()
