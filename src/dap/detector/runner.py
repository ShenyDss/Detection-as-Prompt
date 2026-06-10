from __future__ import annotations

from pathlib import Path
from typing import Any

from dap.detector.mock_converter import convert_mock_predictions
from dap.detector.mock_detector import run_mock_detector
from dap.detector.yolo_adapter import run_ultralytics_yolo
from dap.schemas.core import DefectHypothesis
from dap.utils.jsonl import read_jsonl, write_jsonl


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def collect_image_paths(image_dir: str | Path, limit: int | None = None) -> list[Path]:
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")
    image_paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if limit is not None:
        image_paths = image_paths[:limit]
    return image_paths


def run_detector_to_hypotheses(
    *,
    dataset_config: dict[str, Any],
    detector_config: dict[str, Any],
    image_dir: str | Path | None = None,
    limit: int | None = None,
) -> tuple[list[DefectHypothesis], list[dict[str, Any]]]:
    mode = str(detector_config.get("mode", "mock")).lower()
    resolved_image_dir = Path(image_dir or dataset_config["image_dir"])
    class_names = list(dataset_config.get("class_names", []))

    if mode == "mock":
        image_paths = collect_image_paths(resolved_image_dir, limit=limit)
        if not image_paths:
            image_paths = _mock_image_paths(class_names)
        else:
            image_paths = [_display_path(path) for path in image_paths]
        raw_predictions = run_mock_detector(image_paths, class_names)
    elif mode in {"ultralytics", "yolo", "real"}:
        image_paths = collect_image_paths(resolved_image_dir, limit=limit)
        raw_predictions = run_ultralytics_yolo(
            checkpoint=str(detector_config.get("checkpoint") or ""),
            image_paths=image_paths,
            class_names=class_names,
            confidence_threshold=float(detector_config.get("confidence_threshold", 0.25)),
            iou_threshold=float(detector_config.get("iou_threshold", 0.5)),
        )
    elif mode == "from_jsonl":
        input_path = Path(detector_config["input_path"])
        raw_predictions = list(read_jsonl(input_path))
    else:
        raise ValueError(f"Unsupported detector mode: {mode}")

    hypotheses = convert_mock_predictions(
        raw_predictions,
        top_k_confusions=int(detector_config.get("top_k_confusions", 3)),
        score_threshold=float(detector_config.get("confidence_threshold", 0.0)),
    )
    return hypotheses, raw_predictions


def write_raw_predictions(path: str | Path, rows: list[dict[str, Any]]) -> None:
    write_jsonl(path, rows)


def _mock_image_paths(class_names: list[str]) -> list[Path]:
    count = max(2, min(len(class_names), 4))
    return [Path(f"mock_image_{index:04d}.png") for index in range(count)]


def _display_path(path: Path) -> Path:
    try:
        return path.relative_to(Path.cwd())
    except ValueError:
        return path
