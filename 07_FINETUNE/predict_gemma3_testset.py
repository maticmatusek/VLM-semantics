#!/usr/bin/env python3
"""Generate fine-tuned Gemma 3 answers for every golden-test question.

Inputs:
  * golden_final_test.json: human/golden Answer* fields
  * gemma-3-12b-it_golden_final_test.json: base-Gemma Answer* fields

Outputs:
  * a wide JSON file with the same schema as the inputs, where Answer* fields
    contain predictions from the fine-tuned adapter;
  * a resume-safe, judge-ready JSONL file containing the golden, benchmark,
    and fine-tuned answers side by side (one row per question).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any


PROJECT = Path.home() / "VLM-semantics"
QUESTION_TYPES = ("SLO", "F1", "F2", "F3", "I1", "I2", "I3", "A1", "A2", "A3")
DEFAULT_INSTRUCTION = (
    "Odgovori v slovenščini na vprašanje o sliki. Odgovori jasno in stvarno."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-json",
        type=Path,
        default=PROJECT / "data" / "gemma-3-12b-it_golden_final_test.json",
        help="JSON containing questions and base google/gemma-3-12b-it answers.",
    )
    parser.add_argument(
        "--golden-json",
        type=Path,
        default=PROJECT / "data" / "golden_final_test.json",
        help="JSON containing the golden Answer* fields.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=PROJECT / "data",
        help="Directory containing SLO_images.",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=PROJECT / "outputs" / "gemma3-12b-slo-all",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=(
            PROJECT
            / "out_jsons_final"
            / "gemma3-12b-slo-all_golden_final_test.json"
        ),
        help="Final wide prediction JSON with fine-tuned Answer* fields.",
    )
    parser.add_argument(
        "--judge-jsonl",
        type=Path,
        default=(
            PROJECT
            / "out_jsons_final"
            / "judge_ready_gemma3-12b-slo-all_vs_gemma-3-12b-it.jsonl"
        ),
        help="Resume checkpoint and judge-ready long-format JSONL.",
    )
    parser.add_argument("--base-model", default="google/gemma-3-12b-it")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--gpu", default="5", help="Physical GPU used for inference.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 is deterministic greedy decoding; positive values enable sampling.",
    )
    parser.add_argument("--pan-and-scan", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        help="Generate only the first N unanswered questions (for a smoke test).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing judge JSONL instead of resuming it.",
    )
    return parser.parse_args()


def load_json_list(path: Path, label: str) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} JSON not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{label} JSON must be a list of objects: {path}")
    return value


def validate_and_index(
    benchmark: list[dict[str, Any]], golden: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    def make_index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for position, row in enumerate(rows):
            image_id = str(row.get("image_id", "")).strip()
            if not image_id:
                raise ValueError(f"{label} row {position} has no image_id")
            if image_id in result:
                raise ValueError(f"{label} contains duplicate image_id={image_id}")
            result[image_id] = row
        return result

    base_by_id = make_index(benchmark, "benchmark")
    gold_by_id = make_index(golden, "golden")
    if set(base_by_id) != set(gold_by_id):
        missing_gold = sorted(set(base_by_id) - set(gold_by_id))[:10]
        missing_base = sorted(set(gold_by_id) - set(base_by_id))[:10]
        raise ValueError(
            "Benchmark/golden image IDs do not match; "
            f"missing from golden={missing_gold}, missing from benchmark={missing_base}"
        )

    for image_id, base_row in base_by_id.items():
        gold_row = gold_by_id[image_id]
        for key in ("image_path", *QUESTION_TYPES):
            if base_row.get(key) != gold_row.get(key):
                raise ValueError(f"Mismatch for image_id={image_id}, field={key}")
        for question_type in QUESTION_TYPES:
            if not str(base_row.get(question_type, "")).strip():
                raise ValueError(f"Empty {question_type} question for image_id={image_id}")
            if not str(base_row.get("Answer" + question_type, "")).strip():
                raise ValueError(
                    f"Empty benchmark Answer{question_type} for image_id={image_id}"
                )
            if not str(gold_row.get("Answer" + question_type, "")).strip():
                raise ValueError(
                    f"Empty golden Answer{question_type} for image_id={image_id}"
                )
    return gold_by_id


def build_image_index(image_root: Path) -> tuple[Path, dict[str, Path]]:
    root = image_root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Image root not found: {root}")
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_key = path.relative_to(root).as_posix().casefold()
        if relative_key in index and index[relative_key] != path:
            raise ValueError(f"Case-insensitive duplicate image path: {relative_key}")
        index[relative_key] = path
    return root, index


def resolve_image(image_root: Path, image_index: dict[str, Path], raw_path: str) -> Path:
    normalized = raw_path.replace("\\", "/").lstrip("/")
    candidate = (image_root / normalized).resolve()
    try:
        candidate.relative_to(image_root)
    except ValueError as exc:
        raise ValueError(f"image_path escapes --image-root: {raw_path}") from exc
    if candidate.is_file():
        return candidate
    case_insensitive = image_index.get(normalized.casefold())
    if case_insensitive is not None:
        return case_insensitive
    raise FileNotFoundError(f"Image not found under {image_root}: {raw_path}")


def task_id(image_id: str, question_type: str) -> str:
    return f"{image_id}:{question_type}"


def load_completed(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed, 0
    lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid resume JSONL at {path}:{line_number}. "
                    "Remove only the incomplete final line, then rerun."
                ) from exc
            identifier = str(row.get("task_id", ""))
            if not identifier:
                raise ValueError(f"Resume row {line_number} has no task_id")
            if identifier in completed:
                raise ValueError(f"Duplicate task_id in resume file: {identifier}")
            completed[identifier] = row
            lines += 1
    return completed, lines


def write_wide_output(
    output_path: Path,
    benchmark: list[dict[str, Any]],
    completed: dict[str, dict[str, Any]],
) -> None:
    result: list[dict[str, Any]] = []
    for source_row in benchmark:
        row = dict(source_row)
        image_id = str(row["image_id"])
        for question_type in QUESTION_TYPES:
            identifier = task_id(image_id, question_type)
            prediction = completed.get(identifier)
            if prediction is not None:
                row["Answer" + question_type] = prediction["finetuned_answer"]
        result.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, output_path)


class ImageCache:
    def __init__(self, capacity: int = 8):
        self.capacity = capacity
        self.values: OrderedDict[Path, Any] = OrderedDict()

    def get(self, path: Path, image_class: Any) -> Any:
        if path in self.values:
            image = self.values.pop(path)
            self.values[path] = image
            return image
        image = image_class.open(path).convert("RGB")
        self.values[path] = image
        while len(self.values) > self.capacity:
            _, old_image = self.values.popitem(last=False)
            old_image.close()
        return image


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    # Set this before CUDA libraries are imported: visible cuda:0 becomes the
    # requested physical GPU.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from peft import PeftModel
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration

    benchmark_path = args.benchmark_json.expanduser().resolve()
    golden_path = args.golden_json.expanduser().resolve()
    adapter_dir = args.adapter_dir.expanduser().resolve()
    output_path = args.output_json.expanduser().resolve()
    judge_path = args.judge_jsonl.expanduser().resolve()

    print(f"[setup] Benchmark JSON: {benchmark_path}", flush=True)
    print(f"[setup] Golden JSON:    {golden_path}", flush=True)
    print(f"[setup] Adapter:        {adapter_dir}", flush=True)
    print(f"[setup] Wide output:    {output_path}", flush=True)
    print(f"[setup] Judge/resume:   {judge_path}", flush=True)

    benchmark = load_json_list(benchmark_path, "benchmark")
    golden = load_json_list(golden_path, "golden")
    golden_by_id = validate_and_index(benchmark, golden)
    image_root, image_index = build_image_index(args.image_root)
    print(
        f"[setup] Validated {len(benchmark):,} images and "
        f"{len(benchmark) * len(QUESTION_TYPES):,} aligned questions",
        flush=True,
    )

    if not adapter_dir.is_dir() or not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Valid LoRA adapter directory not found: {adapter_dir}")

    judge_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and judge_path.exists():
        judge_path.unlink()
    completed, completed_lines = load_completed(judge_path)
    print(f"[resume] Loaded {completed_lines:,} completed predictions", flush=True)

    tasks: list[dict[str, Any]] = []
    known_task_ids: set[str] = set()
    for record_index, base_row in enumerate(benchmark):
        image_id = str(base_row["image_id"])
        gold_row = golden_by_id[image_id]
        image_path = resolve_image(image_root, image_index, str(base_row["image_path"]))
        for question_type in QUESTION_TYPES:
            identifier = task_id(image_id, question_type)
            known_task_ids.add(identifier)
            if identifier in completed:
                prior = completed[identifier]
                expected_values = {
                    "image_id": image_id,
                    "question_type": question_type,
                    "question": str(base_row[question_type]).strip(),
                    "golden_answer": str(gold_row["Answer" + question_type]).strip(),
                    "benchmark_answer": str(base_row["Answer" + question_type]).strip(),
                    "benchmark_model": args.base_model,
                    "finetuned_model": adapter_dir.name,
                }
                for field, expected in expected_values.items():
                    if prior.get(field) != expected:
                        raise ValueError(
                            f"Resume data mismatch for {identifier}, field={field}. "
                            "Use a different --judge-jsonl or pass --overwrite."
                        )
                if not str(prior.get("finetuned_answer", "")).strip():
                    raise ValueError(f"Resume data has an empty answer for {identifier}")
                continue
            suitability_key = question_type + "_suitable"
            tasks.append(
                {
                    "task_id": identifier,
                    "record_index": record_index,
                    "image_id": image_id,
                    "image_path": str(base_row["image_path"]),
                    "resolved_image_path": image_path,
                    "image_url": base_row.get("image_url"),
                    "question_type": question_type,
                    "question": str(base_row[question_type]).strip(),
                    "golden_answer": str(gold_row["Answer" + question_type]).strip(),
                    "benchmark_answer": str(base_row["Answer" + question_type]).strip(),
                    "text_image_correlation": base_row.get("text_image_correlation"),
                    "suitability": base_row.get(suitability_key),
                }
            )

    unknown_completed = set(completed) - known_task_ids
    if unknown_completed:
        raise ValueError(
            "Resume file contains task IDs absent from the current inputs: "
            f"{sorted(unknown_completed)[:10]}"
        )

    total_questions = len(benchmark) * len(QUESTION_TYPES)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        if len(completed) == total_questions:
            print("[done] Nothing left to generate", flush=True)
            write_wide_output(output_path, benchmark, completed)
            print(f"[done] Fine-tuned wide JSON: {output_path}", flush=True)
        else:
            print(
                f"[partial] No tasks selected, but only {len(completed):,}/"
                f"{total_questions:,} predictions are complete",
                flush=True,
            )
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(0)
    print(
        f"[model] Physical GPU {args.gpu} -> visible cuda:0 "
        f"({torch.cuda.get_device_name(0)})",
        flush=True,
    )
    print("[model] Loading saved processor", flush=True)
    processor = AutoProcessor.from_pretrained(adapter_dir)
    processor.tokenizer.padding_side = "left"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    print(f"[model] Loading 4-bit base model {args.base_model}", flush=True)
    base_model = Gemma3ForConditionalGeneration.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    print("[model] Attaching trained LoRA adapter", flush=True)
    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        is_trainable=False,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(
        f"[model] Ready; allocated={torch.cuda.memory_allocated(0) / 1024**3:.2f} GiB",
        flush=True,
    )

    eos_token_ids = [processor.tokenizer.eos_token_id]
    vocabulary = processor.tokenizer.get_vocab()
    for token in ("<end_of_turn>", "<turn|>"):
        token_id = vocabulary.get(token)
        if token_id is not None and token_id not in eos_token_ids:
            eos_token_ids.append(token_id)

    generation_args: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "eos_token_id": eos_token_ids,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "cache_implementation": "static",
        "disable_compile": True,
    }
    if args.temperature > 0:
        generation_args.update(temperature=args.temperature, top_p=0.9)

    cache = ImageCache()
    started = time.monotonic()
    generated_this_run = 0
    cursor = 0
    current_batch_size = args.batch_size
    instruction = args.instruction.strip()

    with judge_path.open("a", encoding="utf-8") as checkpoint:
        while cursor < len(tasks):
            batch_tasks = tasks[cursor : cursor + current_batch_size]
            conversations = []
            for task in batch_tasks:
                image = cache.get(task["resolved_image_path"], Image)
                user_text = (
                    f"{instruction}\n\n{task['question']}"
                    if instruction
                    else task["question"]
                )
                conversations.append(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": user_text},
                            ],
                        }
                    ]
                )

            try:
                inputs = processor.apply_chat_template(
                    conversations,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True,
                    processor_kwargs={"do_pan_and_scan": args.pan_and_scan},
                ).to("cuda:0")
                prompt_length = inputs["input_ids"].shape[1]
                with torch.inference_mode():
                    generated_ids = model.generate(**inputs, **generation_args)
                answer_ids = generated_ids[:, prompt_length:]
                answers = processor.batch_decode(
                    answer_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                del inputs, generated_ids, answer_ids
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if current_batch_size == 1:
                    raise
                current_batch_size = max(1, current_batch_size // 2)
                print(
                    f"[memory] CUDA OOM; retrying with batch size {current_batch_size}",
                    flush=True,
                )
                continue

            if len(answers) != len(batch_tasks):
                raise RuntimeError(
                    f"Decoded {len(answers)} answers for {len(batch_tasks)} prompts"
                )

            for task, answer in zip(batch_tasks, answers):
                output_row = {
                    "task_id": task["task_id"],
                    "record_index": task["record_index"],
                    "image_id": task["image_id"],
                    "image_path": task["image_path"],
                    "image_url": task["image_url"],
                    "question_type": task["question_type"],
                    "question": task["question"],
                    "golden_answer": task["golden_answer"],
                    "benchmark_model": args.base_model,
                    "benchmark_answer": task["benchmark_answer"],
                    "finetuned_model": adapter_dir.name,
                    "finetuned_answer": answer.strip(),
                    "text_image_correlation": task["text_image_correlation"],
                    "suitability": task["suitability"],
                }
                checkpoint.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                completed[task["task_id"]] = output_row
            checkpoint.flush()
            os.fsync(checkpoint.fileno())

            cursor += len(batch_tasks)
            generated_this_run += len(batch_tasks)
            elapsed = time.monotonic() - started
            rate = generated_this_run / elapsed if elapsed else 0.0
            remaining = len(tasks) - cursor
            eta_seconds = remaining / rate if rate else 0.0
            overall_done = len(completed)
            print(
                f"[progress] {overall_done:,}/{total_questions:,} "
                f"({overall_done / total_questions:.1%}); "
                f"this run {generated_this_run:,}/{len(tasks):,}; "
                f"{rate:.2f} questions/s; ETA {eta_seconds / 3600:.2f} h",
                flush=True,
            )

    if len(completed) == total_questions:
        write_wide_output(output_path, benchmark, completed)
        print(f"[done] All {total_questions:,} predictions completed", flush=True)
        print(f"[done] Fine-tuned wide JSON: {output_path}", flush=True)
    else:
        print(
            f"[partial] Saved {len(completed):,}/{total_questions:,} predictions; "
            "rerun the same command to resume",
            flush=True,
        )
    print(f"[done] Judge-ready JSONL:    {judge_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[stopped] Interrupted safely; completed batches remain in the JSONL", flush=True)
        sys.exit(130)
