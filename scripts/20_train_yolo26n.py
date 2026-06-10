from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dap.config.loader import read_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO26n on the converted Labelme YOLO dataset.")
    parser.add_argument("--config", default="configs/yolo26n_train.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and print train kwargs without training.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    config = dict(read_yaml(config_path))
    _apply_overrides(config, args)
    _normalize_paths(config)
    _prepare_ultralytics_config_dir()

    report = build_training_report(config)
    report_path = PROJECT_ROOT / "outputs/yolo_runs/yolo26n_training_plan.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install ultralytics before YOLO training.") from exc

    model = YOLO(str(config.pop("model")))
    results = model.train(**config)
    print(results)


def build_training_report(config: dict[str, Any]) -> dict[str, Any]:
    data_path = Path(str(config["data"]))
    model_path = str(config["model"])
    dataset_summary = _summarize_yolo_dataset(data_path)
    return {
        "status": "ready" if data_path.exists() else "blocked",
        "model": model_path,
        "train_kwargs": config,
        "dataset": dataset_summary,
        "notes": [
            "YOLO_CONFIG_DIR is redirected to outputs/ultralytics_config to avoid user-profile permission issues.",
            "The dataset.yaml uses test: val/images as requested.",
            "For an 8GB GPU, start with batch=8 at imgsz=640. Increase batch only after confirming stable memory usage.",
        ],
    }


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    for key in ("epochs", "batch", "imgsz", "device", "model"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value


def _normalize_paths(config: dict[str, Any]) -> None:
    for key in ("data", "project"):
        value = Path(str(config[key]))
        if not value.is_absolute():
            value = PROJECT_ROOT / value
        config[key] = str(value)


def _prepare_ultralytics_config_dir() -> None:
    config_dir = PROJECT_ROOT / "outputs/ultralytics_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))


def _summarize_yolo_dataset(dataset_yaml: Path) -> dict[str, Any]:
    if not dataset_yaml.exists():
        return {"dataset_yaml": str(dataset_yaml), "exists": False}
    dataset_root = _read_dataset_root(dataset_yaml)
    summary: dict[str, Any] = {"dataset_yaml": str(dataset_yaml), "exists": True, "root": str(dataset_root)}
    for split in ("train", "val"):
        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"
        images = [path for path in image_dir.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
        labels = list(label_dir.glob("*.txt"))
        objects = 0
        empty = 0
        for label in labels:
            lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            objects += len(lines)
            if not lines:
                empty += 1
        summary[split] = {
            "images": len(images),
            "labels": len(labels),
            "objects": objects,
            "empty_labels": empty,
        }
    return summary


def _read_dataset_root(dataset_yaml: Path) -> Path:
    text = dataset_yaml.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip().startswith("path:"):
            raw_path = line.split(":", 1)[1].strip()
            return Path(raw_path)
    return dataset_yaml.parent


if __name__ == "__main__":
    main()
