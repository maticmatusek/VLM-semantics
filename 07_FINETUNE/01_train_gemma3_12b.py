#!/usr/bin/env python3
"""Two-GPU Gemma 3 12B vision QLoRA trainer for physical GPUs 5 and 6.

Expected default layout:
    ~/VLM-semantics/data/SLO_images/
    ~/VLM-semantics/cleaned_gemma_data_all/train.jsonl
    ~/VLM-semantics/cleaned_gemma_data_all/validation.jsonl

With that layout, start training with no path arguments:
    python ~/VLM-semantics/train_gemma3_12b_gpu56.py

The parent process relaunches this file under torchrun with two workers. Inside
the workers, physical GPUs 5 and 6 are visible as local CUDA devices 0 and 1.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def launch_on_gpu_5_and_6() -> None:
    """Relaunch once with torchrun; child processes have LOCAL_RANK set."""
    if "LOCAL_RANK" in os.environ:
        return
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    print("[launcher] Selecting physical GPUs 5 and 6", flush=True)
    print("[launcher] Starting two torchrun workers", flush=True)
    print(f"[launcher] Command: {' '.join(command)}", flush=True)
    os.execvp(sys.executable, command)


launch_on_gpu_5_and_6()

# Import CUDA libraries only after CUDA_VISIBLE_DEVICES has been set.
import torch
from datasets import Image as DatasetImage
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
    TrainerCallback,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


DEFAULT_INSTRUCTION = "Odgovori v slovenščini na vprašanje o sliki. Odgovori jasno in stvarno."
DEFAULT_PROJECT = Path.home() / "VLM-semantics"
DEFAULT_JSONL_DIR = DEFAULT_PROJECT / "cleaned_gemma_data_all"
DEFAULT_IMAGE_ROOT = DEFAULT_PROJECT / "data"


def progress(message: str, *, all_ranks: bool = False) -> None:
    rank = int(os.environ.get("RANK", "0"))
    if all_ranks or rank == 0:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{stamp}] [rank {rank}] {message}", flush=True)


def print_cuda_memory(label: str) -> None:
    device = torch.cuda.current_device()
    gib = 1024**3
    allocated = torch.cuda.memory_allocated(device) / gib
    reserved = torch.cuda.memory_reserved(device) / gib
    peak = torch.cuda.max_memory_allocated(device) / gib
    progress(
        f"{label}: cuda:{device} allocated={allocated:.2f} GiB, "
        f"reserved={reserved:.2f} GiB, peak={peak:.2f} GiB",
        all_ranks=True,
    )


def package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


class ProgressPrinter(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        progress(f"Training started; planned optimizer steps: {state.max_steps:,}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            progress(f"step {state.global_step:,}: {json.dumps(logs, default=str, sort_keys=True)}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        progress(
            f"Evaluation finished at step {state.global_step:,}: "
            f"{json.dumps(metrics or {}, default=str, sort_keys=True)}"
        )

    def on_save(self, args, state, control, **kwargs):
        progress(f"Checkpoint saved at step {state.global_step:,} under {args.output_dir}")

    def on_train_end(self, args, state, control, **kwargs):
        progress(f"Training loop finished at step {state.global_step:,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=DEFAULT_JSONL_DIR / "train.jsonl",
    )
    parser.add_argument(
        "--validation-jsonl",
        type=Path,
        default=DEFAULT_JSONL_DIR / "validation.jsonl",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=DEFAULT_IMAGE_ROOT,
        help="Parent of SLO_images; defaults to ~/VLM-semantics/data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROJECT / "outputs" / "gemma3-12b-slo-vqa-lora",
    )
    parser.add_argument("--model-id", default="google/gemma-3-12b-it")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument(
        "--correlations",
        nargs="+",
        default=["dobro"],
        help="Keep only these correlation ratings. Pass 'all' to disable filtering.",
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Positive value overrides epochs; useful for a 50-step smoke test.",
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer",
        choices=("adamw_torch_fused", "adamw_torch", "paged_adamw_8bit"),
        default="adamw_torch_fused",
        help="Fused AdamW is the stable default on A100; 4-bit model quantization remains enabled.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Default global batch: 1 per GPU x 2 GPUs x 4 accumulation = 8.",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--data-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2"),
        default="sdpa",
        help="Use flash_attention_2 only after installing a compatible flash-attn build.",
    )
    parser.add_argument(
        "--pan-and-scan",
        action="store_true",
        help="Better detail for wide/tall images, but materially higher VRAM use.",
    )
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id")
    return parser.parse_args()


def secure_image_path(image_root: Path, relative_path: str) -> Path:
    root = image_root.expanduser().resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"image_path escapes --image-root: {relative_path}") from exc
    return candidate


def validate_images(raw_dataset, image_root: Path, split_name: str) -> None:
    unique_paths = sorted(set(raw_dataset["image_path"]))
    progress(f"{split_name}: checking {len(unique_paths):,} unique image paths")
    missing = [path for path in unique_paths if not secure_image_path(image_root, path).is_file()]
    if missing:
        preview = "\n".join(f"  - {item}" for item in missing[:20])
        raise FileNotFoundError(
            f"{split_name}: {len(missing):,}/{len(unique_paths):,} unique images are missing under "
            f"{image_root.resolve()}. First missing paths:\n{preview}"
        )
    progress(f"{split_name}: all {len(unique_paths):,} unique images are present")


def build_dataset(
    jsonl_path: Path,
    image_root: Path,
    instruction: str,
    correlations: list[str],
    data_workers: int,
    split_name: str,
):
    progress(f"{split_name}: loading JSONL from {jsonl_path.resolve()}")
    raw = load_dataset("json", data_files=str(jsonl_path), split="train")
    required = {"image_path", "question", "answer", "text_image_correlation"}
    missing_columns = required - set(raw.column_names)
    if missing_columns:
        raise ValueError(f"{jsonl_path} is missing columns: {sorted(missing_columns)}")

    original_count = len(raw)
    progress(f"{split_name}: loaded {original_count:,} rows")
    allowed_lower = {value.lower() for value in correlations}
    if "all" not in allowed_lower:
        allowed = set(correlations)
        raw = raw.filter(
            lambda row: row["text_image_correlation"] in allowed,
            num_proc=data_workers,
            desc=f"Filter {split_name} correlations",
        )
        progress(
            f"{split_name}: correlation filter {sorted(allowed)} retained "
            f"{len(raw):,}/{original_count:,} rows"
        )
    if not len(raw):
        raise ValueError(f"{split_name} has no rows after correlation filtering")

    validate_images(raw, image_root, split_name)
    root = image_root.expanduser().resolve()
    prefix = instruction.strip()

    def format_row(row):
        question = row["question"].strip()
        user_text = f"{prefix}\n\n{question}" if prefix else question
        return {
            "image": str(secure_image_path(root, row["image_path"])),
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_text},
                    ],
                }
            ],
            "completion": [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": row["answer"].strip()}],
                }
            ],
        }

    formatted = raw.map(
        format_row,
        remove_columns=raw.column_names,
        num_proc=data_workers,
        desc=f"Format {split_name}",
    )
    formatted = formatted.cast_column("image", DatasetImage(decode=True))
    progress(f"{split_name}: formatting complete; using {len(formatted):,} examples")
    return formatted


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the torchrun worker")
    torch.cuda.set_device(local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support bfloat16")
    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"Expected exactly two visible GPUs (physical 5 and 6), found {torch.cuda.device_count()}"
        )
    if args.data_workers < 1:
        raise ValueError("--data-workers must be at least 1")

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    progress(
        f"Worker ready on visible cuda:{local_rank} ({torch.cuda.get_device_name(local_rank)}); "
        f"physical GPU is {'5' if local_rank == 0 else '6'}",
        all_ranks=True,
    )
    print_cuda_memory("startup")

    if not args.train_jsonl.is_file():
        raise FileNotFoundError(f"Training JSONL not found: {args.train_jsonl.resolve()}")
    if not args.skip_eval and not args.validation_jsonl.is_file():
        raise FileNotFoundError(f"Validation JSONL not found: {args.validation_jsonl.resolve()}")
    if not args.image_root.is_dir():
        raise NotADirectoryError(f"Image root not found: {args.image_root.resolve()}")

    progress("Resolved configuration:")
    progress(f"  train_jsonl={args.train_jsonl.resolve()}")
    progress(f"  validation_jsonl={args.validation_jsonl.resolve()}")
    progress(f"  image_root={args.image_root.resolve()}")
    progress(f"  expected image folder={args.image_root.resolve() / 'SLO_images'}")
    progress(f"  output_dir={args.output_dir.resolve()}")
    progress(f"  model={args.model_id}")
    progress(
        f"  attention={args.attn_implementation}; optimizer={args.optimizer}; "
        f"pan_and_scan={args.pan_and_scan}"
    )
    progress(
        "  package_versions="
        + ", ".join(
            f"{package}={package_version(package)}"
            for package in (
                "torch",
                "transformers",
                "datasets",
                "trl",
                "peft",
                "accelerate",
                "bitsandbytes",
            )
        )
    )

    if local_rank == 0:
        global_batch = args.batch_size * args.gradient_accumulation_steps * 2
        print(f"GPU 5 -> visible cuda:0: {torch.cuda.get_device_name(0)}")
        print(f"GPU 6 -> visible cuda:1: {torch.cuda.get_device_name(1)}")
        print(f"Effective global batch size: {global_batch}")

    train_dataset = build_dataset(
        args.train_jsonl,
        args.image_root,
        args.instruction,
        args.correlations,
        args.data_workers,
        "train",
    )
    validation_dataset = None
    if not args.skip_eval:
        validation_dataset = build_dataset(
            args.validation_jsonl,
            args.image_root,
            args.instruction,
            args.correlations,
            args.data_workers,
            "validation",
        )

    progress(f"Loading processor for {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "right"
    if hasattr(processor, "image_processor"):
        processor.image_processor.do_pan_and_scan = args.pan_and_scan
    progress("Processor loaded")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    progress("Loading Gemma 3 12B in 4-bit NF4; this is the longest startup step")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map={"": local_rank},
    )
    progress("Base model loaded", all_ranks=True)
    print_cuda_memory("after base-model load")
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    progress("Preparing the quantized model for gradient checkpointing and QLoRA")
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Tune language attention and MLP layers; keep the pretrained vision encoder frozen.
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=(
            r"model\.language_model\..*\."
            r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
        ),
    )
    progress("Attaching LoRA adapters to language attention and MLP layers")
    model = get_peft_model(model, peft_config)
    if local_rank == 0:
        model.print_trainable_parameters()
    print_cuda_memory("after LoRA attachment")

    world_size = int(os.environ.get("WORLD_SIZE", "2"))
    update_steps_per_epoch = math.ceil(
        len(train_dataset)
        / (args.batch_size * world_size * args.gradient_accumulation_steps)
    )
    planned_steps = (
        args.max_steps
        if args.max_steps > 0
        else math.ceil(update_steps_per_epoch * args.epochs)
    )
    warmup_steps = max(0, round(planned_steps * args.warmup_ratio))
    progress(
        f"Scheduler plan: approximately {planned_steps:,} optimizer steps, "
        f"including {warmup_steps:,} warmup steps ({args.warmup_ratio:.1%})"
    )

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        optim=args.optimizer,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=None,
        packing=False,
        completion_only_loss=True,
        eval_strategy="no" if args.skip_eval else "epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        log_level="info",
        report_to="tensorboard",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        dataloader_num_workers=args.data_workers,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        seed=args.seed,
        data_seed=args.seed,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    progress("Creating SFTTrainer and multimodal data collator")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=processor,
        callbacks=[ProgressPrinter()],
    )
    progress("Trainer ready; entering trainer.train()")
    print_cuda_memory("immediately before training")
    trainer.train(
        resume_from_checkpoint=str(args.resume_from_checkpoint)
        if args.resume_from_checkpoint
        else None
    )
    print_cuda_memory("after training")
    progress(f"Saving final LoRA adapter to {args.output_dir.resolve()}")
    trainer.save_model(str(args.output_dir))
    if trainer.is_world_process_zero():
        processor.save_pretrained(args.output_dir)
        progress(f"Finished successfully. Final files are in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
