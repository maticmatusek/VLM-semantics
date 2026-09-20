#!/usr/bin/env python3
"""Resume-safe dual-GPU inference for the Slovenian Gemma 3 test set.

The parent launches two independent inference workers, one on each requested
physical GPU. Each worker uses batch size 1 to avoid the Gemma 3 batched
multimodal token-index failure, while the two workers still generate two
answers concurrently. Worker checkpoints are merged into one judge-ready JSONL
and, after all 4,090 tasks finish, one wide prediction JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from predict_gemma3_testset import (
    DEFAULT_INSTRUCTION,
    PROJECT,
    QUESTION_TYPES,
    ImageCache,
    build_image_index,
    load_json_list,
    resolve_image,
    task_id,
    validate_and_index,
    write_wide_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-json",
        type=Path,
        default=PROJECT / "data" / "gemma-3-12b-it_golden_final_test.json",
    )
    parser.add_argument(
        "--golden-json",
        type=Path,
        default=PROJECT / "data" / "golden_final_test.json",
    )
    parser.add_argument("--image-root", type=Path, default=PROJECT / "data")
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
    )
    parser.add_argument(
        "--judge-jsonl",
        type=Path,
        default=(
            PROJECT
            / "out_jsons_final"
            / "judge_ready_gemma3-12b-slo-all_vs_gemma-3-12b-it.jsonl"
        ),
    )
    parser.add_argument("--base-model", default="google/gemma-3-12b-it")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument(
        "--gpus",
        nargs=2,
        default=("5", "6"),
        metavar=("GPU_A", "GPU_B"),
        help="Two physical GPUs used as independent inference workers.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Generation ceiling per answer; must be at least 1000.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 gives deterministic greedy decoding.",
    )
    parser.add_argument("--pan-and-scan", action="store_true")
    parser.add_argument(
        "--cuda-launch-blocking",
        action="store_true",
        help="Synchronous CUDA errors for a short diagnostic run; slower.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Select only the first N total tasks; intended for diagnostics.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete canonical and GPU-worker checkpoints before starting.",
    )

    # Parent-to-worker arguments are intentionally hidden from normal help.
    parser.add_argument("--worker-index", type=int, choices=(0, 1), help=argparse.SUPPRESS)
    parser.add_argument("--physical-gpu", help=argparse.SUPPRESS)
    parser.add_argument("--worker-jsonl", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--resume-jsonl",
        type=Path,
        action="append",
        default=[],
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def worker_path(canonical: Path, gpu: str) -> Path:
    return canonical.with_name(f"{canonical.stem}.gpu{gpu}{canonical.suffix}")


def load_rows(paths: list[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for source in paths:
        path = source.expanduser().resolve()
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL at {path}:{line_number}. Remove only the "
                        "incomplete final line and rerun."
                    ) from exc
                identifier = str(row.get("task_id", ""))
                if not identifier:
                    raise ValueError(f"Missing task_id at {path}:{line_number}")
                previous = rows.get(identifier)
                if previous is not None and previous != row:
                    raise ValueError(
                        f"Conflicting checkpoint records for task_id={identifier}"
                    )
                rows[identifier] = row
    return rows


def ordered_task_ids(benchmark: list[dict[str, Any]]) -> list[str]:
    return [
        task_id(str(row["image_id"]), question_type)
        for row in benchmark
        for question_type in QUESTION_TYPES
    ]


def atomically_write_jsonl(
    path: Path, rows: dict[str, dict[str, Any]], ordered_ids: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for identifier in ordered_ids:
            row = rows.get(identifier)
            if row is not None:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def validate_completed(
    completed: dict[str, dict[str, Any]],
    benchmark: list[dict[str, Any]],
    golden_by_id: dict[str, dict[str, Any]],
    adapter_name: str,
    base_model: str,
) -> None:
    expected_ids = set(ordered_task_ids(benchmark))
    unknown = set(completed) - expected_ids
    if unknown:
        raise ValueError(f"Unknown task IDs in checkpoints: {sorted(unknown)[:10]}")
    base_by_id = {str(row["image_id"]): row for row in benchmark}
    for identifier, row in completed.items():
        image_id, question_type = identifier.rsplit(":", 1)
        base = base_by_id[image_id]
        gold = golden_by_id[image_id]
        expected = {
            "image_id": image_id,
            "question_type": question_type,
            "question": str(base[question_type]).strip(),
            "golden_answer": str(gold["Answer" + question_type]).strip(),
            "benchmark_answer": str(base["Answer" + question_type]).strip(),
            "benchmark_model": base_model,
            "finetuned_model": adapter_name,
        }
        for field, value in expected.items():
            if row.get(field) != value:
                raise ValueError(
                    f"Checkpoint mismatch for {identifier}, field={field}. "
                    "Use another output path or --overwrite."
                )
        if not str(row.get("finetuned_answer", "")).strip():
            raise ValueError(f"Empty fine-tuned answer in checkpoint: {identifier}")


def build_tasks(
    benchmark: list[dict[str, Any]],
    golden_by_id: dict[str, dict[str, Any]],
    image_root: Path,
    image_index: dict[str, Path],
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for record_index, base in enumerate(benchmark):
        image_id = str(base["image_id"])
        gold = golden_by_id[image_id]
        resolved_image = resolve_image(image_root, image_index, str(base["image_path"]))
        for question_index, question_type in enumerate(QUESTION_TYPES):
            tasks.append(
                {
                    "global_index": len(tasks),
                    "task_id": task_id(image_id, question_type),
                    "record_index": record_index,
                    "question_index": question_index,
                    "image_id": image_id,
                    "image_path": str(base["image_path"]),
                    "resolved_image_path": resolved_image,
                    "image_url": base.get("image_url"),
                    "question_type": question_type,
                    "question": str(base[question_type]).strip(),
                    "golden_answer": str(gold["Answer" + question_type]).strip(),
                    "benchmark_answer": str(base["Answer" + question_type]).strip(),
                    "text_image_correlation": base.get("text_image_correlation"),
                    "suitability": base.get(question_type + "_suitable"),
                }
            )
    return tasks


def common_command_args(args: argparse.Namespace) -> list[str]:
    values = [
        "--benchmark-json",
        str(args.benchmark_json.expanduser().resolve()),
        "--golden-json",
        str(args.golden_json.expanduser().resolve()),
        "--image-root",
        str(args.image_root.expanduser().resolve()),
        "--adapter-dir",
        str(args.adapter_dir.expanduser().resolve()),
        "--base-model",
        args.base_model,
        "--instruction",
        args.instruction,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
    ]
    if args.pan_and_scan:
        values.append("--pan-and-scan")
    if args.cuda_launch_blocking:
        values.append("--cuda-launch-blocking")
    if args.limit is not None:
        values.extend(("--limit", str(args.limit)))
    return values


def run_parent(args: argparse.Namespace) -> None:
    canonical = args.judge_jsonl.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    adapter_dir = args.adapter_dir.expanduser().resolve()
    gpu_paths = [worker_path(canonical, gpu) for gpu in args.gpus]

    if args.overwrite:
        for path in (canonical, *gpu_paths, output_json):
            if path.exists():
                path.unlink()

    benchmark = load_json_list(args.benchmark_json, "benchmark")
    golden = load_json_list(args.golden_json, "golden")
    golden_by_id = validate_and_index(benchmark, golden)
    all_ids = ordered_task_ids(benchmark)
    selected_ids = all_ids[: args.limit] if args.limit is not None else all_ids

    completed = load_rows([canonical, *gpu_paths])
    validate_completed(
        completed, benchmark, golden_by_id, adapter_dir.name, args.base_model
    )
    selected_complete = sum(identifier in completed for identifier in selected_ids)
    print(
        f"[parent] Validated {len(benchmark):,} images and {len(all_ids):,} tasks; "
        f"selected={len(selected_ids):,}, already complete={selected_complete:,}",
        flush=True,
    )

    if selected_complete == len(selected_ids):
        atomically_write_jsonl(canonical, completed, all_ids)
        if len(completed) == len(all_ids):
            write_wide_output(output_json, benchmark, completed)
            print(f"[parent] All predictions complete: {output_json}", flush=True)
        else:
            print("[parent] Selected diagnostic range is already complete", flush=True)
        return

    commands = []
    for worker_index, (gpu, output) in enumerate(zip(args.gpus, gpu_paths)):
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            *common_command_args(args),
            "--worker-index",
            str(worker_index),
            "--physical-gpu",
            gpu,
            "--worker-jsonl",
            str(output),
        ]
        for resume_path in (canonical, *gpu_paths):
            command.extend(("--resume-jsonl", str(resume_path)))
        commands.append(command)

    print(
        f"[parent] Launching independent workers on physical GPUs "
        f"{args.gpus[0]} and {args.gpus[1]}; batch size 1 each",
        flush=True,
    )
    processes: list[subprocess.Popen[Any]] = []
    interrupted = False
    try:
        for command in commands:
            processes.append(subprocess.Popen(command))
        return_codes = [process.wait() for process in processes]
    except KeyboardInterrupt:
        interrupted = True
        print("\n[parent] Interrupt received; stopping both workers safely", flush=True)
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        return_codes = [process.wait() for process in processes]

    completed = load_rows([canonical, *gpu_paths])
    validate_completed(
        completed, benchmark, golden_by_id, adapter_dir.name, args.base_model
    )
    atomically_write_jsonl(canonical, completed, all_ids)
    print(
        f"[parent] Merged checkpoint: {len(completed):,}/{len(all_ids):,} answers "
        f"in {canonical}",
        flush=True,
    )

    if len(completed) == len(all_ids):
        write_wide_output(output_json, benchmark, completed)
        print(f"[parent] Judge-ready JSONL: {canonical}", flush=True)
        print(f"[parent] Fine-tuned wide JSON: {output_json}", flush=True)
    elif not interrupted and any(code != 0 for code in return_codes):
        raise RuntimeError(f"Worker exit codes: {return_codes}; rerun to resume")
    else:
        print("[parent] Partial results are safe; rerun the same command to resume", flush=True)

    if interrupted:
        raise KeyboardInterrupt


def run_worker(args: argparse.Namespace) -> None:
    assert args.worker_index is not None
    assert args.physical_gpu is not None
    assert args.worker_jsonl is not None

    # The child process sees exactly one GPU, mapped to visible cuda:0.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.physical_gpu
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.cuda_launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    import torch
    from peft import PeftModel
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration

    prefix = f"[gpu {args.physical_gpu}]"
    benchmark = load_json_list(args.benchmark_json, "benchmark")
    golden = load_json_list(args.golden_json, "golden")
    golden_by_id = validate_and_index(benchmark, golden)
    image_root, image_index = build_image_index(args.image_root)
    adapter_dir = args.adapter_dir.expanduser().resolve()
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Adapter not found: {adapter_dir}")

    completed = load_rows(args.resume_jsonl)
    validate_completed(
        completed, benchmark, golden_by_id, adapter_dir.name, args.base_model
    )
    tasks = build_tasks(benchmark, golden_by_id, image_root, image_index)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    tasks = [
        task
        for task in tasks
        if task["global_index"] % 2 == args.worker_index
        and task["task_id"] not in completed
    ]
    print(f"{prefix} Assigned {len(tasks):,} unanswered tasks", flush=True)
    if not tasks:
        return

    if not torch.cuda.is_available():
        raise RuntimeError(f"{prefix} CUDA unavailable")
    torch.cuda.set_device(0)
    print(f"{prefix} Device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"{prefix} Loading processor", flush=True)
    processor = AutoProcessor.from_pretrained(adapter_dir)
    processor.tokenizer.padding_side = "left"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    print(f"{prefix} Loading 4-bit {args.base_model}", flush=True)
    base_model = Gemma3ForConditionalGeneration.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    print(f"{prefix} Attaching LoRA adapter", flush=True)
    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        is_trainable=False,
        low_cpu_mem_usage=True,
    )
    model.eval()
    embedding_count = model.get_input_embeddings().num_embeddings
    core_vocab_size = int(base_model.model.vocab_size)
    image_token_id_value = getattr(
        model.config,
        "image_token_id",
        getattr(model.config, "image_token_index", None),
    )
    if image_token_id_value is None:
        raise ValueError("Gemma model config has no image token ID")
    image_token_id = int(image_token_id_value)
    print(
        f"{prefix} Ready; memory={torch.cuda.memory_allocated(0) / 1024**3:.2f} GiB; "
        f"embeddings={embedding_count:,}; core_vocab={core_vocab_size:,}; "
        f"image_token={image_token_id:,}",
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
        # The default/dynamic cache avoids a Gemma 3 static-cache indexing
        # failure observed during multimodal prefill with this installation.
        "cache_implementation": "dynamic",
        "disable_compile": True,
    }
    if args.temperature > 0:
        generation_args.update(temperature=args.temperature, top_p=0.9)

    output_path = args.worker_jsonl.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = ImageCache()
    instruction = args.instruction.strip()
    started = time.monotonic()

    with output_path.open("a", encoding="utf-8") as checkpoint:
        for position, task in enumerate(tasks, start=1):
            image = cache.get(task["resolved_image_path"], Image)
            user_text = (
                f"{instruction}\n\n{task['question']}"
                if instruction
                else task["question"]
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_text},
                    ],
                }
            ]

            # Split formatting from multimodal encoding. This avoids the
            # apply_chat_template batch path that produced the CUDA index assert.
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor(
                text=[prompt],
                images=[image],
                text_kwargs={"padding": True, "return_tensors": "pt"},
                images_kwargs={
                    "do_pan_and_scan": args.pan_and_scan,
                    "return_tensors": "pt",
                },
            )

            input_ids_cpu = inputs["input_ids"]
            invalid = (input_ids_cpu < 0) | (
                (input_ids_cpu >= embedding_count) & (input_ids_cpu != image_token_id)
            )
            if invalid.any():
                bad_ids = sorted(set(input_ids_cpu[invalid].tolist()))
                raise ValueError(
                    f"{prefix} Invalid token IDs before CUDA for {task['task_id']}: "
                    f"{bad_ids}; embedding_count={embedding_count}"
                )

            if position == 1:
                tensor_shapes = {
                    key: tuple(value.shape)
                    for key, value in inputs.items()
                    if hasattr(value, "shape")
                }
                print(
                    f"{prefix} First task={task['task_id']}; "
                    f"image_size={image.size}; input_ids="
                    f"[{int(input_ids_cpu.min())}, {int(input_ids_cpu.max())}]; "
                    f"image_tokens={int((input_ids_cpu == image_token_id).sum())}; "
                    f"shapes={tensor_shapes}",
                    flush=True,
                )

            inputs = inputs.to("cuda:0")
            prompt_length = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                generated = model.generate(**inputs, **generation_args)
            answer = processor.decode(
                generated[0, prompt_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            if not answer:
                raise RuntimeError(f"{prefix} Empty answer for {task['task_id']}")

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
                "finetuned_answer": answer,
                "text_image_correlation": task["text_image_correlation"],
                "suitability": task["suitability"],
            }
            checkpoint.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            checkpoint.flush()
            os.fsync(checkpoint.fileno())

            del inputs, generated
            elapsed = time.monotonic() - started
            rate = position / elapsed if elapsed else 0.0
            remaining = len(tasks) - position
            if position == 1 or position % 10 == 0 or position == len(tasks):
                print(
                    f"{prefix} {position:,}/{len(tasks):,}; "
                    f"{rate:.3f} questions/s; ETA "
                    f"{(remaining / rate / 3600) if rate else 0.0:.2f} h; "
                    f"last={task['task_id']}",
                    flush=True,
                )

    print(f"{prefix} Worker finished successfully", flush=True)


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1000:
        raise ValueError("--max-new-tokens must be at least 1000")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if len(set(args.gpus)) != 2:
        raise ValueError("--gpus must name two different physical GPUs")

    if args.worker_index is None:
        run_parent(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[stopped] Checkpoints are safe; rerun the same command to resume", flush=True)
        sys.exit(130)
