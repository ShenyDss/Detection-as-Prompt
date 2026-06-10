from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dap.config.loader import read_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA training for correction-oriented Qwen3-VL reviewer tuning.")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--train-data", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the training plan only.")
    parser.add_argument("--check-vram", action="store_true", help="Load model+LoRA and run one forward pass before training.")
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_yaml(PROJECT_ROOT / args.config)
    plan = build_training_plan(config, args)
    output_dir = Path(plan["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("Dry run complete. No model training was started.")
        return

    missing = [name for name in plan["requires"] if importlib.util.find_spec(name) is None]
    if missing:
        raise ModuleNotFoundError(f"Missing packages for real LoRA training: {missing}")
    if not plan["checkpoint_exists"]:
        raise FileNotFoundError(f"Checkpoint not found: {plan['checkpoint']}")

    trainer, model = build_trainer(plan)
    if args.check_vram:
        run_vram_check(trainer, model, output_dir)
        print("VRAM check complete. No training was started.")
        return

    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    trainer.processing_class.save_pretrained(str(output_dir / "final_processor"))
    print(f"Saved LoRA adapter to {output_dir / 'final'}")


def build_training_plan(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    train_path = _resolve_project_path(args.train_data or config.get("instruction_data"))
    checkpoint = args.checkpoint or config.get("model_dir") or config.get("base_model")
    output_dir = _resolve_project_path(args.output_dir or config.get("output_dir"))
    if not train_path.exists():
        raise FileNotFoundError(f"Instruction data not found: {train_path}")

    optimization = dict(config.get("optimization", {}))
    lora = dict(config.get("lora", {}))
    runtime = dict(config.get("runtime", {}))
    data_summary = _summarize_instruction_data(train_path)
    max_train_samples = args.max_train_samples
    if max_train_samples is None:
        max_train_samples = runtime.get("max_train_samples")

    return {
        "train_data": str(train_path),
        "sample_count": data_summary["sample_count"],
        "data_summary": data_summary,
        "checkpoint": str(checkpoint),
        "checkpoint_exists": Path(str(checkpoint)).exists(),
        "output_dir": str(output_dir),
        "lora": {
            "rank": args.lora_rank if args.lora_rank is not None else int(lora.get("rank", 16)),
            "alpha": int(lora.get("alpha", 32)),
            "dropout": float(lora.get("dropout", 0.05)),
            "target_modules": list(lora.get("target_modules", ["q_proj", "v_proj"])),
        },
        "optimization": {
            "learning_rate": args.learning_rate if args.learning_rate is not None else float(optimization.get("learning_rate", 2e-4)),
            "epochs": args.epochs if args.epochs is not None else int(optimization.get("epochs", 1)),
            "per_device_train_batch_size": int(optimization.get("per_device_train_batch_size", 1)),
            "gradient_accumulation_steps": int(optimization.get("gradient_accumulation_steps", 8)),
            "max_length": int(optimization.get("max_length", 2048)),
            "logging_steps": int(optimization.get("logging_steps", 5)),
            "save_steps": int(optimization.get("save_steps", 100)),
            "save_total_limit": int(optimization.get("save_total_limit", 2)),
        },
        "runtime": {
            "device": runtime.get("device", "auto"),
            "precision": runtime.get("precision", "bf16"),
            "quantization": runtime.get("quantization"),
            "gradient_checkpointing": bool(runtime.get("gradient_checkpointing", True)),
            "dataloader_num_workers": int(runtime.get("dataloader_num_workers", 0)),
            "max_train_samples": max_train_samples,
        },
        "requires": ["transformers", "peft", "datasets", "accelerate"],
        "notes": [
            "For an 8GB GPU, use batch=1, gradient accumulation, bf16, and gradient checkpointing.",
            "bitsandbytes is not installed, so this plan does not use 4-bit quantization.",
            "The KG fields are temporary unless an expert-curated KG replaces the generated class-map KG.",
        ],
    }


def build_trainer(plan: dict[str, Any]):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

    precision = plan["runtime"]["precision"]
    dtype = torch.bfloat16 if precision == "bf16" and torch.cuda.is_available() else torch.float16 if precision == "fp16" else torch.float32
    processor = AutoProcessor.from_pretrained(plan["checkpoint"], trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        plan["checkpoint"],
        trust_remote_code=True,
        dtype=dtype,
        device_map="auto",
    )
    if plan["runtime"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    lora_config = LoraConfig(
        r=int(plan["lora"]["rank"]),
        lora_alpha=int(plan["lora"]["alpha"]),
        lora_dropout=float(plan["lora"]["dropout"]),
        target_modules=list(plan["lora"]["target_modules"]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = InstructionDataset(
        Path(plan["train_data"]),
        max_samples=plan["runtime"].get("max_train_samples"),
    )
    collator = QwenVLInstructionCollator(
        processor=processor,
        max_length=int(plan["optimization"]["max_length"]),
    )
    args = TrainingArguments(
        output_dir=str(Path(plan["output_dir"]) / "trainer"),
        num_train_epochs=int(plan["optimization"]["epochs"]),
        per_device_train_batch_size=int(plan["optimization"]["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(plan["optimization"]["gradient_accumulation_steps"]),
        learning_rate=float(plan["optimization"]["learning_rate"]),
        bf16=precision == "bf16" and torch.cuda.is_available(),
        fp16=precision == "fp16" and torch.cuda.is_available(),
        logging_steps=int(plan["optimization"]["logging_steps"]),
        save_steps=int(plan["optimization"]["save_steps"]),
        save_total_limit=int(plan["optimization"]["save_total_limit"]),
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=int(plan["runtime"]["dataloader_num_workers"]),
        gradient_checkpointing=bool(plan["runtime"]["gradient_checkpointing"]),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=processor,
    )
    return trainer, model


def run_vram_check(trainer, model, output_dir: Path) -> None:
    import torch

    report: dict[str, Any] = {"cuda_available": torch.cuda.is_available(), "status": "unknown"}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        report["before_train_step"] = _cuda_memory_report()
    batch = next(iter(trainer.get_train_dataloader()))
    device = next(model.parameters()).device
    batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
    model.train()
    try:
        output = model(**batch)
        report["loss"] = float(output.loss.detach().cpu()) if output.loss is not None else None
        if torch.cuda.is_available():
            report["after_forward"] = _cuda_memory_report()
        output.loss.backward()
        report["status"] = "ready"
        if torch.cuda.is_available():
            report["after_backward"] = _cuda_memory_report()
    except RuntimeError as exc:
        message = str(exc)
        report["status"] = "oom" if "out of memory" in message.lower() else "failed"
        report["error"] = message[:1000]
        if torch.cuda.is_available():
            report["after_error"] = _cuda_memory_report()
        (output_dir / "vram_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise
    finally:
        model.zero_grad(set_to_none=True)
    (output_dir / "vram_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


class InstructionDataset:
    def __init__(self, path: Path, max_samples: int | None = None):
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        self.rows = rows[:max_samples] if max_samples else rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class QwenVLInstructionCollator:
    def __init__(self, processor, max_length: int):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        encoded_features = [self._encode_feature(feature) for feature in features]
        return self._pad_features(encoded_features)

    def _encode_feature(self, feature: dict[str, Any]) -> dict[str, Any]:
        image_path = feature.get("region_path") or feature.get("image_path")
        target_text = json.dumps(feature["target"], ensure_ascii=False)
        user_message = {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": feature["prompt"]},
            ],
        }
        full_messages = [user_message, {"role": "assistant", "content": [{"type": "text", "text": target_text}]}]
        prompt_inputs = self.processor.apply_chat_template(
            [user_message],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        full_inputs = self.processor.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        item = {}
        for key, value in full_inputs.items():
            if key in {"input_ids", "attention_mask"}:
                item[key] = value.squeeze(0)
            else:
                item[key] = value
        labels = item["input_ids"].clone()
        prompt_len = int(prompt_inputs["input_ids"].shape[-1])
        labels[:prompt_len] = -100
        if labels.shape[0] > self.max_length:
            labels = labels[-self.max_length :]
            item = {
                key: value[-self.max_length :] if key in {"input_ids", "attention_mask"} else value
                for key, value in item.items()
            }
        item["labels"] = labels
        return item

    def _pad_features(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        tokenizer = self.processor.tokenizer
        input_ids = [feature["input_ids"] for feature in features]
        attention_mask = [feature["attention_mask"] for feature in features]
        labels = [feature["labels"] for feature in features]
        batch = {
            "input_ids": torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0),
            "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100),
        }
        for key in features[0]:
            if key in batch or key in {"labels", "input_ids", "attention_mask"}:
                continue
            values = [feature[key] for feature in features if key in feature]
            if not values:
                continue
            if values[0].dim() == 0:
                batch[key] = torch.stack(values)
            else:
                batch[key] = torch.cat(values, dim=0)
        return batch


def _resolve_project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _summarize_instruction_data(path: Path) -> dict:
    status_counts: Counter[str] = Counter()
    pred_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    missing_images = 0
    missing_regions = 0
    empty_fields: Counter[str] = Counter()
    sample_count = 0
    max_prompt_chars = 0
    max_target_chars = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample_count += 1
        target = dict(row.get("target", {}))
        metadata = dict(row.get("metadata", {}))
        status_counts[str(target.get("hypothesis_status"))] += 1
        pred_counts[str(metadata.get("pred_class"))] += 1
        target_counts[str(target.get("corrected_class") or "__reject__")] += 1
        image_path = row.get("image_path")
        region_path = row.get("region_path")
        if not image_path or not Path(str(image_path)).exists():
            missing_images += 1
        if not region_path or not Path(str(region_path)).exists():
            missing_regions += 1
        for key in ("visual_evidence", "uncertainty_evidence", "graph_path", "causes", "actions"):
            value = target.get(key)
            if value in (None, "", []) or value == {}:
                empty_fields[key] += 1
        max_prompt_chars = max(max_prompt_chars, len(str(row.get("prompt", ""))))
        max_target_chars = max(max_target_chars, len(json.dumps(target, ensure_ascii=False)))
    return {
        "sample_count": sample_count,
        "status_counts": dict(status_counts),
        "pred_class_counts": dict(pred_counts),
        "target_class_counts": dict(target_counts),
        "missing_image_path": missing_images,
        "missing_region_path": missing_regions,
        "empty_target_fields": dict(empty_fields),
        "max_prompt_chars": max_prompt_chars,
        "max_target_chars": max_target_chars,
    }


def _cuda_memory_report() -> dict[str, float]:
    import torch

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "total_gb": round(total_bytes / 1024**3, 3),
        "free_gb": round(free_bytes / 1024**3, 3),
        "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }


if __name__ == "__main__":
    main()
