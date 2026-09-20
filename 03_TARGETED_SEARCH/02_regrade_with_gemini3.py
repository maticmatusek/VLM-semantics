import os
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, Set, DefaultDict, List

import PIL.Image
import google.generativeai as genai


# =========================
# CONFIG
# =========================
DEFAULT_IN_MANIFEST = "slo_images_manifest.jsonl"
DEFAULT_OUT_MANIFEST = "slo_images_manifest_graded.jsonl"
DEFAULT_TAXONOMY_PATH = "ocm-slo-clean.json"
JUDGE_MODEL = "gemini-2.5-flash"
TARGET_GOOD_PER_TOP_CATEGORY = 60
ONLY_INPUT_GEMMA_LABEL = "good"

MAX_ATTEMPTS = 8
INITIAL_BACKOFF_S = 0.2
MAX_BACKOFF_S = 60.0

# Safety caps. Set to 0 to disable.
MAX_GEMINI_CALLS_PER_TOP_PER_RUN = 1000
MAX_GEMINI_CALLS_PER_RUN = 10000

DEBUG = True
PRINT_EVERY = 1
PRINT_SKIP_REASONS = True
PRINT_CATEGORY_SUMMARY_EVERY = 1



# =========================
# HELPERS
# =========================
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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_path(p: str) -> str:
    return (p or "").replace("\\", "/").strip()


def normalize_text(s: str) -> str:
    return " ".join((s or "").strip().split())


def safe_json_load(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    s2 = s.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(s2)
    except Exception:
        return None


def load_taxonomy_roots_in_order(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Taxonomy file is not a JSON object: {path}")
    roots = list(data.keys())
    log(f"Loaded taxonomy roots from {path}: {len(roots)} top categories")
    return roots


def get_top_category(category: str) -> str:
    return normalize_text((category or "").split(">")[0])


def try_open_image(path: str) -> Optional[PIL.Image.Image]:
    try:
        img = PIL.Image.open(path)
        img.load()
        return img
    except Exception as e:
        warn(f"Failed to open image '{path}': {type(e).__name__}: {e}")
        return None


# =========================
# OUTPUT RESUME STATE
# =========================
def load_output_state(path: str) -> Tuple[Set[int], DefaultDict[str, int], DefaultDict[str, int]]:
    seen: Set[int] = set()
    good_per_top: DefaultDict[str, int] = defaultdict(int)
    graded_per_top: DefaultDict[str, int] = defaultdict(int)

    if not os.path.exists(path):
        return seen, good_per_top, graded_per_top

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            rec_id = obj.get("id")
            if isinstance(rec_id, int) and "final_grade" in obj:
                seen.add(rec_id)

            top = get_top_category(obj.get("category", ""))
            if top:
                graded_per_top[top] += 1
                if str(obj.get("final_grade", "")).strip().lower() == "good":
                    good_per_top[top] += 1

    return seen, good_per_top, graded_per_top


# =========================
# PRE-SCAN INPUT CANDIDATES
# =========================
def prescan_input_candidates(
    in_manifest: str,
    already_done: Set[int],
    top_categories: Set[str],
) -> Tuple[DefaultDict[str, int], Dict[str, int]]:
    eligible_remaining_by_top: DefaultDict[str, int] = defaultdict(int)
    prescan_stats = {
        "lines": 0,
        "bad_json": 0,
        "unknown_top": 0,
        "already_done": 0,
        "wrong_gemma_label": 0,
        "eligible": 0,
    }

    with open(in_manifest, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            prescan_stats["lines"] += 1

            try:
                rec = json.loads(line)
            except Exception:
                prescan_stats["bad_json"] += 1
                continue

            rec_id = rec.get("id")
            category = normalize_text(rec.get("category", ""))
            top_category = get_top_category(category)
            gemma_label = str(rec.get("gemma_label", "")).strip().lower()

            if not top_category or top_category not in top_categories:
                prescan_stats["unknown_top"] += 1
                continue

            if isinstance(rec_id, int) and rec_id in already_done:
                prescan_stats["already_done"] += 1
                continue

            if gemma_label != ONLY_INPUT_GEMMA_LABEL:
                prescan_stats["wrong_gemma_label"] += 1
                continue

            eligible_remaining_by_top[top_category] += 1
            prescan_stats["eligible"] += 1

    return eligible_remaining_by_top, prescan_stats


# =========================
# JUDGE PROMPT
# =========================
JUDGE_INSTRUCTIONS = """
You are a strict dataset quality judge.

You will be given:
- Category (taxonomy path): PRIMARY criterion
- Prompt (Slovene): secondary
- Commentary (Slovene): secondary
- An image

You MUST separate two decisions:
A) category_match (PRIMARY): Interpret the category label broadly but only within its domain; do not accept unrelated topics.
If the category fit is weak or speculative, answer "no".
B) prompt_match: if category_match=yes, does it also match the prompt+commentary? (yes/no)

Scoring rules:
- bad: category_match = no
- good: category_match = yes AND prompt_match = yes
- medium: category_match = yes AND prompt_match = no  (category-only match)

Also return:
- category_only_match: true if category_match=yes and prompt_match=no, else false

Return ONLY valid JSON:
{
  "final_grade": "good|medium|bad",
  "category_match": "yes|no",
  "prompt_match": "yes|no",
  "category_only_match": true|false,
  "reason": "one short sentence"
}

Hard rules:
- posters/memes/screenshots/infographics or lots of text => category_match=no
- product listing / marketplace photos => usually category_match=no unless category is explicitly about products/listings
""".strip()


def judge_one(model, rec: Dict[str, Any], img: PIL.Image.Image) -> Tuple[str, str, bool, str, str]:
    category = rec.get("category", "")
    prompt = rec.get("prompt", "")
    desc = rec.get("description", "")

    text = f"""{JUDGE_INSTRUCTIONS}

Category: {category}
Prompt (SL): {prompt}
Commentary (SL): {desc}
"""

    delay = INITIAL_BACKOFF_S
    last_err = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = model.generate_content([text, img])
            raw = (getattr(resp, "text", "") or "").strip()
            data = safe_json_load(raw)

            if not data:
                warn("Judge returned non-JSON; fallback=medium.")
                return "medium", "parse_failed", False, "yes", "no"

            final_grade = str(data.get("final_grade", "")).strip().lower()
            reason = str(data.get("reason", "")).strip()
            category_match = str(data.get("category_match", "")).strip().lower()
            prompt_match = str(data.get("prompt_match", "")).strip().lower()
            category_only_match = bool(data.get("category_only_match", False))

            if category_match not in ("yes", "no"):
                category_match = "yes" if final_grade != "bad" else "no"
            if prompt_match not in ("yes", "no"):
                prompt_match = "yes" if final_grade == "good" else "no"

            if category_match == "no":
                final_grade = "bad"
                category_only_match = False
            else:
                if prompt_match == "yes":
                    final_grade = "good"
                    category_only_match = False
                else:
                    final_grade = "medium"
                    category_only_match = True

            if final_grade not in ("good", "medium", "bad"):
                final_grade = "medium"
            if not reason:
                reason = "no reason"

            return final_grade, reason, category_only_match, category_match, prompt_match

        except Exception as e:
            last_err = e
            msg = str(e)

            if "503" in msg or "UNAVAILABLE" in msg or "high demand" in msg:
                warn(f"Gemini overloaded. attempt {attempt}/{MAX_ATTEMPTS}. sleep {delay:.1f}s")
                time.sleep(delay)
                delay = min(delay * 1.8, MAX_BACKOFF_S)
                continue

            if "Unable to process input image" in msg or "INVALID_ARGUMENT" in msg:
                return "bad", "image_upload_failed", False, "no", "no"

            warn(f"Gemini judge error attempt {attempt}/{MAX_ATTEMPTS}: {type(e).__name__}: {e}. sleep {delay:.1f}s")
            time.sleep(delay)
            delay = min(delay * 1.8, MAX_BACKOFF_S)

    warn(f"Gemini judge failed after retries: {last_err}")
    return "medium", "judge_failed", False, "yes", "no"


# =========================
# MAIN
# =========================
def main():
    api_key = GEMINI_API_KEY
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY env var.")

    in_manifest = os.environ.get("IN_MANIFEST", DEFAULT_IN_MANIFEST)
    out_manifest = os.environ.get("OUT_MANIFEST", DEFAULT_OUT_MANIFEST)
    taxonomy_path = os.environ.get("TAXONOMY_PATH", DEFAULT_TAXONOMY_PATH)

    log("========== RE-GRADE START ==========")
    log(f"Judge model: {JUDGE_MODEL}")
    log(f"Input manifest: {in_manifest}")
    log(f"Output manifest: {out_manifest}")
    log(f"Taxonomy path: {taxonomy_path}")
    log(f"Target good per top category: {TARGET_GOOD_PER_TOP_CATEGORY}")
    log(f"Only sending records with gemma_label == '{ONLY_INPUT_GEMMA_LABEL}' to Gemini")
    log(f"MAX_GEMINI_CALLS_PER_TOP_PER_RUN={MAX_GEMINI_CALLS_PER_TOP_PER_RUN} (0 means disabled)")
    log(f"MAX_GEMINI_CALLS_PER_RUN={MAX_GEMINI_CALLS_PER_RUN} (0 means disabled)")

    top_categories_in_order = load_taxonomy_roots_in_order(taxonomy_path)
    top_categories = set(top_categories_in_order)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(JUDGE_MODEL)

    already_done, good_per_top, graded_per_top = load_output_state(out_manifest)
    if already_done:
        log(f"Resume: {len(already_done)} items already graded in output.")

    remaining_candidates_by_top, prescan_stats = prescan_input_candidates(
        in_manifest=in_manifest,
        already_done=already_done,
        top_categories=top_categories,
    )

    log("\n========== PRE-SCAN SUMMARY ==========")
    for k, v in prescan_stats.items():
        log(f"{k}: {v}")

    impossible_announced: Set[str] = set()
    exhausted_announced: Set[str] = set()
    target_announced: Set[str] = set()
    capped_top_announced: Set[str] = set()

    log("\n========== STARTING TOP-CATEGORY STATUS ==========")
    for top in top_categories_in_order:
        current_good = good_per_top[top]
        pending = remaining_candidates_by_top[top]
        max_possible = current_good + pending
        if current_good >= TARGET_GOOD_PER_TOP_CATEGORY:
            log(
                f"[ALREADY DONE] {top}: current_good={current_good}/{TARGET_GOOD_PER_TOP_CATEGORY}, "
                f"pending_candidates={pending}"
            )
            target_announced.add(top)
        elif pending == 0:
            log(
                f"[NO MORE CANDIDATES AT START] {top}: current_good={current_good}/{TARGET_GOOD_PER_TOP_CATEGORY}, "
                f"pending_candidates=0"
            )
            exhausted_announced.add(top)
            if current_good < TARGET_GOOD_PER_TOP_CATEGORY:
                log(
                    f"[CANNOT REACH 50 FROM CURRENT POOL] {top}: current_good={current_good}, "
                    f"remaining_candidates=0, max_possible={max_possible}"
                )
                impossible_announced.add(top)
        elif max_possible < TARGET_GOOD_PER_TOP_CATEGORY:
            log(
                f"[CANNOT REACH 50 FROM CURRENT POOL] {top}: current_good={current_good}, "
                f"remaining_candidates={pending}, max_possible={max_possible}"
            )
            impossible_announced.add(top)
        else:
            log(
                f"[CAN STILL REACH 50] {top}: current_good={current_good}, "
                f"remaining_candidates={pending}, max_possible={max_possible}"
            )

    total_lines = 0
    total_considered = 0
    total_sent_to_gemini = 0
    sent_to_gemini_per_top: DefaultDict[str, int] = defaultdict(int)

    skipped_already_done = 0
    skipped_not_gemma_good = 0
    skipped_unknown_top_category = 0
    skipped_target_already_reached = 0
    skipped_top_run_cap = 0
    skipped_global_run_cap = 0
    newly_graded = 0

    with open(in_manifest, "r", encoding="utf-8") as fin, open(out_manifest, "a", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total_lines += 1

            try:
                rec = json.loads(line)
            except Exception:
                warn("Bad JSONL line; skipping.")
                continue

            rec_id = rec.get("id", None)
            category = normalize_text(rec.get("category", ""))
            top_category = get_top_category(category)
            gemma_label = str(rec.get("gemma_label", "")).strip().lower()

            dbg(
                f"[SCAN] line={total_lines} id={rec_id} top='{top_category}' "
                f"category='{category}' gemma_label='{gemma_label}'"
            )

            if not top_category or top_category not in top_categories:
                skipped_unknown_top_category += 1
                skip(f"id={rec_id} unknown top category '{top_category}'")
                continue

            if isinstance(rec_id, int) and rec_id in already_done:
                skipped_already_done += 1
                skip(f"id={rec_id} already graded in output")
                continue

            if gemma_label != ONLY_INPUT_GEMMA_LABEL:
                skipped_not_gemma_good += 1
                skip(f"id={rec_id} gemma_label='{gemma_label}' != '{ONLY_INPUT_GEMMA_LABEL}'")
                continue

            # This record was counted in prescan as still-eligible. We are now consuming it from the remaining pool,
            # regardless of whether we end up grading it in this run.
            if remaining_candidates_by_top[top_category] > 0:
                remaining_candidates_by_top[top_category] -= 1

            if good_per_top[top_category] >= TARGET_GOOD_PER_TOP_CATEGORY:
                skipped_target_already_reached += 1
                if top_category not in target_announced:
                    log(
                        f"[STOP TOP CATEGORY] '{top_category}' already reached target with "
                        f"{good_per_top[top_category]} good images."
                    )
                    target_announced.add(top_category)
                skip(
                    f"id={rec_id} top category '{top_category}' already has "
                    f"{good_per_top[top_category]}/{TARGET_GOOD_PER_TOP_CATEGORY} good images"
                )
                continue

            if MAX_GEMINI_CALLS_PER_RUN > 0 and total_sent_to_gemini >= MAX_GEMINI_CALLS_PER_RUN:
                skipped_global_run_cap += 1
                skip(
                    f"id={rec_id} skipped because global Gemini run cap reached: "
                    f"{total_sent_to_gemini}/{MAX_GEMINI_CALLS_PER_RUN}"
                )
                continue

            if (
                MAX_GEMINI_CALLS_PER_TOP_PER_RUN > 0
                and sent_to_gemini_per_top[top_category] >= MAX_GEMINI_CALLS_PER_TOP_PER_RUN
            ):
                skipped_top_run_cap += 1
                if top_category not in capped_top_announced:
                    log(
                        f"[TOP RUN CAP REACHED] {top_category}: sent_to_gemini_this_run="
                        f"{sent_to_gemini_per_top[top_category]}/{MAX_GEMINI_CALLS_PER_TOP_PER_RUN}. "
                        f"Further eligible records in this top category will be skipped for this run."
                    )
                    capped_top_announced.add(top_category)
                skip(
                    f"id={rec_id} skipped because top-category Gemini run cap reached for '{top_category}'"
                )
                continue

            total_considered += 1
            log("\n--------------------------------------------------")
            log(
                f"[PROCESS] id={rec_id} | top='{top_category}' | current good in top="
                f"{good_per_top[top_category]}/{TARGET_GOOD_PER_TOP_CATEGORY} | "
                f"remaining_candidates_after_this={remaining_candidates_by_top[top_category]}"
            )
            log(f"[PROCESS] category={category}")
            log(f"[PROCESS] prompt={rec.get('prompt', '')}")
            log(f"[PROCESS] local_path={rec.get('local_path', '')}")

            local_path = normalize_path(rec.get("local_path", ""))
            if local_path and not os.path.exists(local_path):
                alt = os.path.join(os.getcwd(), local_path)
                if os.path.exists(alt):
                    local_path = alt

            if not local_path or not os.path.exists(local_path):
                warn(f"Missing local image for id={rec_id}: path='{rec.get('local_path','')}'. Marking bad.")
                rec["final_grade"] = "bad"
                rec["final_reason"] = "missing_local_image"
                rec["category_match"] = "no"
                rec["prompt_match"] = "no"
                rec["category_only_match"] = False
                rec["final_judge_model"] = JUDGE_MODEL
                rec["final_judged_at"] = utc_now_iso()
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                newly_graded += 1
                graded_per_top[top_category] += 1
                if isinstance(rec_id, int):
                    already_done.add(rec_id)
                log(f"[WRITE] id={rec_id} final_grade=bad reason=missing_local_image")
            else:
                img = try_open_image(local_path)
                if img is None:
                    rec["final_grade"] = "bad"
                    rec["final_reason"] = "failed_to_open_image"
                    rec["category_match"] = "no"
                    rec["prompt_match"] = "no"
                    rec["category_only_match"] = False
                    rec["final_judge_model"] = JUDGE_MODEL
                    rec["final_judged_at"] = utc_now_iso()
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    newly_graded += 1
                    graded_per_top[top_category] += 1
                    if isinstance(rec_id, int):
                        already_done.add(rec_id)
                    log(f"[WRITE] id={rec_id} final_grade=bad reason=failed_to_open_image")
                else:
                    total_sent_to_gemini += 1
                    sent_to_gemini_per_top[top_category] += 1
                    log(
                        f"[GEMINI] sending id={rec_id} to judge model ... "
                        f"global_sent={total_sent_to_gemini} top_sent={sent_to_gemini_per_top[top_category]}"
                    )
                    final_grade, reason, category_only_match, category_match, prompt_match = judge_one(model, rec, img)

                    rec["final_grade"] = final_grade
                    rec["final_reason"] = reason
                    rec["category_match"] = category_match
                    rec["prompt_match"] = prompt_match
                    rec["category_only_match"] = category_only_match
                    rec["final_judge_model"] = JUDGE_MODEL
                    rec["final_judged_at"] = utc_now_iso()

                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()

                    newly_graded += 1
                    graded_per_top[top_category] += 1
                    if final_grade == "good":
                        good_per_top[top_category] += 1

                    if isinstance(rec_id, int):
                        already_done.add(rec_id)

                    if PRINT_EVERY and (newly_graded % PRINT_EVERY == 0):
                        log(
                            f"[GRADED] id={rec_id} top='{top_category}' final_grade={final_grade} "
                            f"category_only_match={category_only_match} reason={reason}"
                        )

            if PRINT_CATEGORY_SUMMARY_EVERY:
                log(
                    f"[TOP CATEGORY STATUS] {top_category}: good={good_per_top[top_category]}/"
                    f"{TARGET_GOOD_PER_TOP_CATEGORY}, graded={graded_per_top[top_category]}, "
                    f"remaining_candidates={remaining_candidates_by_top[top_category]}, "
                    f"max_possible={good_per_top[top_category] + remaining_candidates_by_top[top_category]}"
                )

            if good_per_top[top_category] >= TARGET_GOOD_PER_TOP_CATEGORY and top_category not in target_announced:
                log(
                    f"[STOP TOP CATEGORY] '{top_category}' reached target with "
                    f"{good_per_top[top_category]} good images. All further records in this top category will be skipped."
                )
                target_announced.add(top_category)

            max_possible_now = good_per_top[top_category] + remaining_candidates_by_top[top_category]
            if (
                good_per_top[top_category] < TARGET_GOOD_PER_TOP_CATEGORY
                and max_possible_now < TARGET_GOOD_PER_TOP_CATEGORY
                and top_category not in impossible_announced
            ):
                log(
                    f"[CANNOT REACH 50 FROM CURRENT POOL] {top_category}: current_good={good_per_top[top_category]}, "
                    f"remaining_candidates={remaining_candidates_by_top[top_category]}, max_possible={max_possible_now}"
                )
                impossible_announced.add(top_category)

            if (
                remaining_candidates_by_top[top_category] == 0
                and good_per_top[top_category] < TARGET_GOOD_PER_TOP_CATEGORY
                and top_category not in exhausted_announced
            ):
                log(
                    f"[NO MORE CANDIDATES] {top_category}: finished all available eligible records with "
                    f"good={good_per_top[top_category]}/{TARGET_GOOD_PER_TOP_CATEGORY}"
                )
                exhausted_announced.add(top_category)

    log("\n========== RE-GRADE DONE ==========")
    log(f"Total lines read: {total_lines}")
    log(f"Eligible after prefilters: {total_considered}")
    log(f"Sent to Gemini: {total_sent_to_gemini}")
    log(f"Newly graded: {newly_graded}")
    log(f"Skipped already graded: {skipped_already_done}")
    log(f"Skipped gemma_label != good: {skipped_not_gemma_good}")
    log(f"Skipped unknown top category: {skipped_unknown_top_category}")
    log(f"Skipped because top category already has enough good images: {skipped_target_already_reached}")
    log(f"Skipped because top-category run cap reached: {skipped_top_run_cap}")
    log(f"Skipped because global run cap reached: {skipped_global_run_cap}")

    log("\n========== FINAL TOP-CATEGORY STATUS ==========")
    for top in top_categories_in_order:
        current_good = good_per_top[top]
        remaining = remaining_candidates_by_top[top]
        max_possible = current_good + remaining
        if current_good >= TARGET_GOOD_PER_TOP_CATEGORY:
            status = "DONE_TARGET_REACHED"
        elif remaining == 0:
            status = "DONE_NO_MORE_CANDIDATES"
        elif max_possible < TARGET_GOOD_PER_TOP_CATEGORY:
            status = "NOT_DONE_IMPOSSIBLE_FROM_CURRENT_POOL"
        else:
            status = "NOT_DONE_CAN_CONTINUE"

        log(
            f"  - {top}: status={status}, good={current_good}, graded={graded_per_top[top]}, "
            f"remaining_candidates={remaining}, max_possible={max_possible}, "
            f"sent_to_gemini_this_run={sent_to_gemini_per_top[top]}"
        )

    log(f"Output: {out_manifest}")


if __name__ == "__main__":
    main()
