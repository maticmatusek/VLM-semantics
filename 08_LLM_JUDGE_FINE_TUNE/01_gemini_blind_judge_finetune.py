#!/usr/bin/env python3
"""Blind Gemini judge for base Gemma vs fine-tuned Gemma.

The script consumes the judge-ready JSONL produced by
predict_gemma3_testset_dual_gpu.py. Gemini sees the image, question, golden
reference, and two neutrally labelled candidates (answer_A and answer_B). It
never receives model names or base/fine-tuned labels. Candidate order is
deterministic and exactly balanced across the input set to reduce position bias.

Results are appended to a resume-safe JSONL after every successful judgment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    PROJECT
    / "out_jsons_final"
    / "judge_ready_gemma3-12b-slo-all_vs_gemma-3-12b-it.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT
    / "out_jsons_final"
    / "gemini_blind_judge_gemma3-12b-slo-all_vs_base.jsonl"
)
DEFAULT_SUMMARY = (
    PROJECT
    / "out_jsons_final"
    / "gemini_blind_judge_gemma3-12b-slo-all_vs_base.csv"
)

MODEL_NAME = "gemini-2.5-pro"
PROMPT_VERSION = "blind-two-candidate-v1"
SCORE_FIELDS = (
    "fulfilment",
    "visual_grounding",
    "explanation",
    "hallucination_control",
)
CANDIDATE_IDS = ("answer_A", "answer_B")


# The evaluation criteria and wording are retained from the supplied five-model
# prompt. Only the number of candidates and their anonymous IDs are changed.
PROMPT_TEMPLATE = """
Deluješ kot strog, a pošten evalvator odgovorov na vprašanja o SLIKAH (LLM-as-a-judge).

DOBIŠ:
- opis naloge in vprašanje o sliki,
- sliko, na katero se navezujejo vprašanja in odgovori,
- referenčni "golden standard" odgovor,
- 2 anonimna kandidatna odgovora na isto vprašanje.

Kandidatna odgovora sta namenoma označena samo kot answer_A in answer_B. Nimaš
informacij o njunem izvoru. Ocenjuj samo njuno vsebino in ne ugibaj, kateri
model je ustvaril posamezen odgovor.

TVOJA NALOGA:
1. Vsak kandidatni odgovor (answer_A in answer_B) oceni po 4 dimenzijah (0–2 točk vsaka):
   1. IZPOLNITEV NALOGE (fulfilment)
      - 0: zgreši bistvo ali odgovori na drugo temo;
      - 1: delno; manjkajo ključni deli ali je preveč ohlapen;
      - 2: v celoti odgovori na zahtevo.
   2. VIZUALNA UTEMELJENOST & SPECIFIČNOST (visual_grounding)
      - 0: generičen opis, sklicevanje na nevidne stvari;
      - 1: nekaj konkretnih sklicev na vidne elemente;
      - 2: jasne, specifične podrobnosti (položaj, velikost, barve, ostrina, prekrivanje).
   3. RAZLAGA & KOHERENCA (explanation)
      - 0: nelogično, protislovno, skokovito;
      - 1: razumljivo, a tanko ali delno nedosledno;
      - 2: jasna, povezovalna razlaga, ki podpira trditve.
   4. NADZOR HALUCINACIJ (hallucination_control)
      - 0: izmišljuje objekte/odnose/barve, ki niso na sliki;
      - 1: manjše pretiravanje ali ugibanje brez jasne označitve;
      - 2: brez halucinacij; negotovost je ustrezno označena.

2. HALUCINACIJE – POSEBNO PRAVILO:
   - Če je pri kateremkoli odgovoru prisotna "velika halucinacija"
     (npr. glavni subjekt slike je napačen, izmišljene dominantne barve ali simboli),
     to jasno zabeleži v polju "major_hallucination": true.
   - V tem primeru pri tem odgovoru NE DVIGUJ ocen previsoko.

3. IZBIRA NAJBOLJŠEGA ODGOVORA:
   - Upoštevaj oba kandidatna odgovora.
   - Uporabi 4 dimenzije in sliko.
   - Izberi en odgovor kot objektivno najboljši kompromis med:
     izpolnitvijo naloge, vizualno utemeljenostjo, koherenco in minimalnimi halucinacijami.
   - Golden standard odgovor služi samo kot REFERENCA in NI med kandidati za "best_answer".
   - Vrni ID kandidata: "answer_A" ali "answer_B".

4. IZHODNA OBLIKA – POMEMBNO:
   - VRNI IZKLJUČNO VELJAVEN JSON (brez razlage, brez komentarjev, brez dodatnega teksta).
   - Struktura je natanko:

{{
  "best_answer_id": "<answer_A ali answer_B>",
  "scores": [
    {{
      "answer_id": "answer_A",
      "fulfilment": ...,
      "visual_grounding": ...,
      "explanation": ...,
      "hallucination_control": ...,
      "major_hallucination": ...
    }},
    {{
      "answer_id": "answer_B",
      "fulfilment": ...,
      "visual_grounding": ...,
      "explanation": ...,
      "hallucination_control": ...,
      "major_hallucination": ...
    }}
  ]
}}

--------------------------------------------------
KONKRETNA NALOGA:

ID slike: {example_id}
Vprašanje ({question_key}): {question_text}

Golden standard odgovor:
{gold_answer}

Kandidatna odgovora:
- answer_A: {answer_A}
- answer_B: {answer_B}
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--image-root", type=Path, default=PROJECT / "data")
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument(
        "--api-key-env",
        default="GEMINI_API_KEY",
        help="Environment variable containing the Gemini API key.",
    )
    parser.add_argument(
        "--blind-seed",
        type=int,
        default=20250811,
        help="Fixed seed for deterministic, balanced A/B assignment.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Judge only the first N input rows; useful for a smoke test.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete this script's output checkpoint before judging.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=240,
        help="Characters shown for questions/answers in progress output.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {path}")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    required = (
        "task_id",
        "image_id",
        "image_path",
        "question_type",
        "question",
        "golden_answer",
        "benchmark_answer",
        "finetuned_answer",
    )
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc

            missing = [key for key in required if key not in row]
            if missing:
                raise ValueError(
                    f"Missing fields at {path}:{line_number}: {', '.join(missing)}"
                )
            task_id = str(row["task_id"])
            if task_id in seen:
                raise ValueError(f"Duplicate task_id in input: {task_id}")
            seen.add(task_id)
            for answer_key in ("question", "golden_answer", "benchmark_answer", "finetuned_answer"):
                if not str(row[answer_key]).strip():
                    raise ValueError(f"Empty {answer_key} for task_id={task_id}")
            rows.append(row)

    if not rows:
        raise ValueError(f"Input JSONL is empty: {path}")
    print(f"[load] Validated {len(rows):,} unique judge-ready tasks from {path}", flush=True)
    return rows


def make_blind_assignments(
    rows: list[dict[str, Any]], seed: int
) -> dict[str, bool]:
    """Return task_id -> base_is_answer_A, balanced over the full input."""
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    base_is_a_indices = set(indices[: len(indices) // 2])
    assignments = {
        str(row["task_id"]): index in base_is_a_indices
        for index, row in enumerate(rows)
    }
    base_as_a = sum(assignments.values())
    print(
        f"[blind] seed={seed}; base is A for {base_as_a:,} tasks and B for "
        f"{len(rows) - base_as_a:,} tasks",
        flush=True,
    )
    return assignments


def blinded_candidates(
    row: dict[str, Any], base_is_a: bool
) -> tuple[dict[str, str], dict[str, str]]:
    """Return candidate text and hidden source mapping for one request."""
    base = str(row["benchmark_answer"]).strip()
    finetuned = str(row["finetuned_answer"]).strip()
    if base_is_a:
        return (
            {"answer_A": base, "answer_B": finetuned},
            {"answer_A": "base", "answer_B": "finetuned"},
        )
    return (
        {"answer_A": finetuned, "answer_B": base},
        {"answer_A": "finetuned", "answer_B": "base"},
    )


def build_prompt(row: dict[str, Any], candidates: dict[str, str]) -> str:
    # Only task content, the golden reference, and anonymous answer text enter
    # this prompt. Model labels and base/fine-tuned metadata are never inserted.
    return PROMPT_TEMPLATE.format(
        example_id=row["image_id"],
        question_key=row["question_type"],
        question_text=str(row["question"]).strip(),
        gold_answer=str(row["golden_answer"]).strip(),
        answer_A=candidates["answer_A"],
        answer_B=candidates["answer_B"],
    )


def resolve_image_path(row: dict[str, Any], image_root: Path) -> Path:
    raw = Path(str(row["image_path"]).replace("\\", "/")).expanduser()
    root = image_root.expanduser().resolve()
    candidates = [
        raw,
        PROJECT / raw,
        root / raw,
        root / "SLO_images" / raw.name,
    ]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Image not found for task_id={row['task_id']}: image_path={row['image_path']}; "
        f"image_root={root}"
    )


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def validate_judgment(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Gemini response is not a JSON object")
    best = value.get("best_answer_id")
    if best not in CANDIDATE_IDS:
        raise ValueError(f"Invalid best_answer_id: {best!r}")
    scores = value.get("scores")
    if not isinstance(scores, list) or len(scores) != 2:
        raise ValueError("scores must be a list containing exactly two entries")

    normalized: dict[str, dict[str, Any]] = {}
    for entry in scores:
        if not isinstance(entry, dict):
            raise ValueError("Every scores entry must be an object")
        answer_id = entry.get("answer_id")
        if answer_id not in CANDIDATE_IDS:
            raise ValueError(f"Invalid answer_id: {answer_id!r}")
        if answer_id in normalized:
            raise ValueError(f"Duplicate score entry for {answer_id}")

        clean: dict[str, Any] = {"answer_id": answer_id}
        for field in SCORE_FIELDS:
            score = entry.get(field)
            if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 2:
                raise ValueError(f"{answer_id}.{field} must be an integer from 0 to 2")
            clean[field] = score
        major = entry.get("major_hallucination")
        if not isinstance(major, bool):
            raise ValueError(f"{answer_id}.major_hallucination must be boolean")
        clean["major_hallucination"] = major
        clean["total"] = sum(clean[field] for field in SCORE_FIELDS)
        normalized[answer_id] = clean

    if set(normalized) != set(CANDIDATE_IDS):
        raise ValueError("scores must contain answer_A and answer_B exactly once")
    return {
        "best_answer_id": best,
        "scores": [normalized["answer_A"], normalized["answer_B"]],
    }


def call_gemini(
    client: Any,
    model: str,
    prompt: str,
    image_path: Path,
    max_retries: int,
    retry_base_seconds: float,
) -> dict[str, Any]:
    from google.genai import types

    with Image.open(image_path) as opened:
        image = opened.convert("RGB").copy()

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        print(f"[gemini] Calling {model}; attempt {attempt}/{max_retries}", flush=True)
        try:
            response = client.models.generate_content(
                model=model,
                contents=[prompt, image],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0,
                ),
            )
            raw_text = (response.text or "").strip()
            print(
                f"[gemini] Received {len(raw_text):,} characters; preview="
                f"{raw_text[:300]!r}",
                flush=True,
            )
            if not raw_text:
                raise ValueError("Gemini returned empty response text")
            parsed = json.loads(strip_code_fence(raw_text))
            return validate_judgment(parsed)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last_error = exc
            print(
                f"[retry] attempt {attempt}/{max_retries} failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            if attempt < max_retries:
                delay = retry_base_seconds * (2 ** (attempt - 1))
                print(f"[retry] Waiting {delay:.1f} seconds", flush=True)
                time.sleep(delay)

    assert last_error is not None
    raise RuntimeError(f"Gemini failed after {max_retries} attempts") from last_error


def input_fingerprint(
    row: dict[str, Any], base_is_a: bool, blind_seed: int
) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "blind_seed": blind_seed,
        "base_is_a": base_is_a,
        "task_id": row["task_id"],
        "image_path": row["image_path"],
        "question": row["question"],
        "golden_answer": row["golden_answer"],
        "benchmark_answer": row["benchmark_answer"],
        "finetuned_answer": row["finetuned_answer"],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid output checkpoint at {path}:{line_number}. "
                    "Remove only a visibly incomplete final line, then resume."
                ) from exc
            task_id = str(row.get("task_id", ""))
            if not task_id:
                raise ValueError(f"Missing task_id at {path}:{line_number}")
            if task_id in completed:
                raise ValueError(f"Duplicate task_id in output checkpoint: {task_id}")
            completed[task_id] = row
    return completed


def preview(value: Any, limit: int) -> str:
    text = " ".join(str(value).split())
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + "..."


def score_map(judgment: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["answer_id"]: entry for entry in judgment["scores"]}


def build_output_row(
    row: dict[str, Any],
    judgment: dict[str, Any],
    source_by_candidate: dict[str, str],
    candidates: dict[str, str],
    args: argparse.Namespace,
    fingerprint: str,
) -> dict[str, Any]:
    by_candidate = score_map(judgment)
    candidate_by_source = {source: candidate for candidate, source in source_by_candidate.items()}
    best_candidate = judgment["best_answer_id"]
    base_candidate = candidate_by_source["base"]
    finetuned_candidate = candidate_by_source["finetuned"]

    return {
        "task_id": row["task_id"],
        "image_id": row["image_id"],
        "image_path": row["image_path"],
        "question_type": row["question_type"],
        "question": row["question"],
        "golden_answer": row["golden_answer"],
        "base_model": row.get("benchmark_model", "base_gemma"),
        "base_answer": row["benchmark_answer"],
        "finetuned_model": row.get("finetuned_model", "finetuned_gemma"),
        "finetuned_answer": row["finetuned_answer"],
        "text_image_correlation": row.get("text_image_correlation"),
        "suitability": row.get("suitability"),
        "judge_model": args.model,
        "prompt_version": PROMPT_VERSION,
        "blind_seed": args.blind_seed,
        "input_fingerprint": fingerprint,
        "blind_mapping": source_by_candidate,
        "blind_answers": candidates,
        "best_answer_id": best_candidate,
        "winner": source_by_candidate[best_candidate],
        "base_score": by_candidate[base_candidate],
        "finetuned_score": by_candidate[finetuned_candidate],
        "blind_scores": judgment["scores"],
    }


def print_result(result: dict[str, Any]) -> None:
    mapping = result["blind_mapping"]
    base = result["base_score"]
    finetuned = result["finetuned_score"]
    print(
        f"[result] Gemini chose {result['best_answer_id']}; hidden mapping revealed "
        f"after judgment: A={mapping['answer_A']}, B={mapping['answer_B']}",
        flush=True,
    )
    print(
        f"[result] winner={result['winner']}; "
        f"base: F={base['fulfilment']} VG={base['visual_grounding']} "
        f"E={base['explanation']} H={base['hallucination_control']} "
        f"major={base['major_hallucination']} total={base['total']}; "
        f"finetuned: F={finetuned['fulfilment']} "
        f"VG={finetuned['visual_grounding']} E={finetuned['explanation']} "
        f"H={finetuned['hallucination_control']} "
        f"major={finetuned['major_hallucination']} total={finetuned['total']}",
        flush=True,
    )


def write_summary_csv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id",
        "image_id",
        "question_type",
        "source",
        "model",
        "blind_answer_id",
        *SCORE_FIELDS,
        "major_hallucination",
        "total",
        "best",
        "winner",
        "suitability",
        "text_image_correlation",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in sorted(results, key=lambda item: str(item["task_id"])):
            candidate_by_source = {
                source: candidate
                for candidate, source in result["blind_mapping"].items()
            }
            for source, score_key, model_key in (
                ("base", "base_score", "base_model"),
                ("finetuned", "finetuned_score", "finetuned_model"),
            ):
                score = result[score_key]
                writer.writerow(
                    {
                        "task_id": result["task_id"],
                        "image_id": result["image_id"],
                        "question_type": result["question_type"],
                        "source": source,
                        "model": result[model_key],
                        "blind_answer_id": candidate_by_source[source],
                        **{field: score[field] for field in SCORE_FIELDS},
                        "major_hallucination": score["major_hallucination"],
                        "total": score["total"],
                        "best": result["winner"] == source,
                        "winner": result["winner"],
                        "suitability": result.get("suitability"),
                        "text_image_correlation": result.get("text_image_correlation"),
                    }
                )
    print(f"[save] Summary CSV: {path} ({len(results) * 2:,} rows)", flush=True)


def print_aggregate(results: list[dict[str, Any]]) -> None:
    if not results:
        print("[summary] No completed judgments", flush=True)
        return
    winners = Counter(result["winner"] for result in results)
    base_total = sum(result["base_score"]["total"] for result in results)
    finetuned_total = sum(result["finetuned_score"]["total"] for result in results)
    count = len(results)
    print(
        f"[summary] judgments={count:,}; winners: "
        f"base={winners['base']:,}, finetuned={winners['finetuned']:,}; "
        f"mean total: base={base_total / count:.3f}, "
        f"finetuned={finetuned_total / count:.3f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.sleep_seconds < 0:
        raise ValueError("--sleep-seconds cannot be negative")
    if args.max_retries < 1:
        raise ValueError("--max-retries must be at least 1")
    if args.retry_base_seconds < 0:
        raise ValueError("--retry-base-seconds cannot be negative")

    input_path = args.input_jsonl.expanduser().resolve()
    output_path = args.output_jsonl.expanduser().resolve()
    summary_path = args.summary_csv.expanduser().resolve()
    if output_path == input_path:
        raise ValueError("--output-jsonl must differ from --input-jsonl")

    rows = load_jsonl(input_path)
    assignments = make_blind_assignments(rows, args.blind_seed)
    selected = rows[: args.limit] if args.limit is not None else rows

    if args.overwrite and output_path.exists():
        output_path.unlink()
        print(f"[overwrite] Removed old checkpoint: {output_path}", flush=True)
    completed = load_completed(output_path)
    input_ids = {str(row["task_id"]) for row in rows}
    unknown = set(completed) - input_ids
    if unknown:
        raise ValueError(f"Output contains unknown task IDs: {sorted(unknown)[:5]}")

    rows_by_id = {str(row["task_id"]): row for row in rows}
    for task_id, existing in completed.items():
        row = rows_by_id[task_id]
        expected_fingerprint = input_fingerprint(
            row, assignments[task_id], args.blind_seed
        )
        if existing.get("input_fingerprint") != expected_fingerprint:
            raise ValueError(
                f"Checkpoint configuration/input mismatch for {task_id}. "
                "Use a different output path, or --overwrite intentionally."
            )
        if existing.get("judge_model") != args.model:
            raise ValueError(
                f"Checkpoint model mismatch for {task_id}: "
                f"{existing.get('judge_model')!r} != {args.model!r}"
            )

    selected_complete = sum(str(row["task_id"]) in completed for row in selected)
    print(
        f"[resume] selected={len(selected):,}; already complete={selected_complete:,}; "
        f"remaining={len(selected) - selected_complete:,}",
        flush=True,
    )

    pending = [row for row in selected if str(row["task_id"]) not in completed]
    if pending:
        api_key = os.environ.get(args.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"Environment variable {args.api_key_env} is empty. "
                f"Set it before running; do not paste the key into this script."
            )
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "The Google GenAI SDK is missing. Install it with: "
                "python -m pip install -U google-genai"
            ) from exc
        client = genai.Client(api_key=api_key)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        failed: list[str] = []

        with output_path.open("a", encoding="utf-8") as checkpoint:
            for run_index, row in enumerate(pending, start=1):
                task_id = str(row["task_id"])
                overall_complete = selected_complete + run_index
                candidates, source_by_candidate = blinded_candidates(
                    row, assignments[task_id]
                )
                prompt = build_prompt(row, candidates)
                image_path = resolve_image_path(row, args.image_root)

                print("\n" + "=" * 100, flush=True)
                print(
                    f"[task] {overall_complete:,}/{len(selected):,}; "
                    f"task_id={task_id}; image={image_path}",
                    flush=True,
                )
                print(f"[task] question: {preview(row['question'], args.preview_chars)}", flush=True)
                print(
                    f"[task] golden:   {preview(row['golden_answer'], args.preview_chars)}",
                    flush=True,
                )
                # Deliberately print only blind IDs before the API call.
                print(
                    f"[blind] answer_A ({len(candidates['answer_A']):,} chars): "
                    f"{preview(candidates['answer_A'], args.preview_chars)}",
                    flush=True,
                )
                print(
                    f"[blind] answer_B ({len(candidates['answer_B']):,} chars): "
                    f"{preview(candidates['answer_B'], args.preview_chars)}",
                    flush=True,
                )
                print(
                    f"[blind] prompt={len(prompt):,} chars; model identities omitted",
                    flush=True,
                )

                try:
                    judgment = call_gemini(
                        client=client,
                        model=args.model,
                        prompt=prompt,
                        image_path=image_path,
                        max_retries=args.max_retries,
                        retry_base_seconds=args.retry_base_seconds,
                    )
                except Exception as exc:
                    print(
                        f"[error] task_id={task_id} not checkpointed: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    failed.append(task_id)
                    continue

                fingerprint = input_fingerprint(
                    row, assignments[task_id], args.blind_seed
                )
                result = build_output_row(
                    row=row,
                    judgment=judgment,
                    source_by_candidate=source_by_candidate,
                    candidates=candidates,
                    args=args,
                    fingerprint=fingerprint,
                )
                checkpoint.write(json.dumps(result, ensure_ascii=False) + "\n")
                checkpoint.flush()
                os.fsync(checkpoint.fileno())
                completed[task_id] = result
                print_result(result)

                elapsed = time.monotonic() - started
                rate = run_index / elapsed if elapsed else 0.0
                remaining = len(pending) - run_index
                print(
                    f"[progress] this run={run_index:,}/{len(pending):,}; "
                    f"rate={rate:.3f} judgments/s; "
                    f"ETA={(remaining / rate / 3600) if rate else 0.0:.2f} h; "
                    f"checkpointed={len(completed):,}/{len(rows):,}",
                    flush=True,
                )
                if args.sleep_seconds:
                    time.sleep(args.sleep_seconds)

        if failed:
            print(
                f"[warning] {len(failed):,} tasks failed and remain resumable; "
                f"first IDs: {failed[:10]}",
                flush=True,
            )

    results = list(completed.values())
    write_summary_csv(summary_path, results)
    print_aggregate(results)
    print(f"[save] Resume-safe judgments: {output_path}", flush=True)
    selected_done = sum(str(row["task_id"]) in completed for row in selected)
    if len(completed) == len(rows):
        print(f"[done] All {len(rows):,} tasks judged successfully", flush=True)
    elif args.limit is not None and selected_done == len(selected):
        print(
            f"[partial] Limit run selected {len(selected):,} of {len(rows):,} tasks; "
            "rerun without --limit to continue",
            flush=True,
        )
    else:
        print(
            f"[partial] {len(completed):,}/{len(rows):,} tasks checkpointed; rerun to resume",
            flush=True,
        )
    if selected_done != len(selected):
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[stopped] Checkpoint is safe; rerun the same command to resume", flush=True)
        sys.exit(130)
