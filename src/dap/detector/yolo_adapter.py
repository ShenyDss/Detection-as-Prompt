from __future__ import annotations

from pathlib import Path
from typing import Any


def run_ultralytics_yolo(
    *,
    checkpoint: str,
    image_paths: list[Path],
    class_names: list[str],
    confidence_threshold: float,
    iou_threshold: float,
) -> list[dict[str, Any]]:
    """Run an Ultralytics YOLO model and return detector-like prediction rows."""

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ultralytics is not installed in the active environment. "
            "Install it before using detector.mode='ultralytics'."
        ) from exc

    if not checkpoint:
        raise ValueError("detector.checkpoint must be set when mode='ultralytics'.")

    model = YOLO(checkpoint)
    rows: list[dict[str, Any]] = []

    for image_path in image_paths:
        results = model.predict(
            source=str(image_path),
            conf=confidence_threshold,
            iou=iou_threshold,
            verbose=False,
        )
        detections: list[dict[str, Any]] = []
        for result in results:
            names = result.names
            boxes = result.boxes
            if boxes is None:
                continue
            for box_index, box in enumerate(boxes):
                class_index = int(box.cls.item())
                pred_class = str(names[class_index])
                score = float(box.conf.item())
                bbox = [float(value) for value in box.xyxy[0].tolist()]
                detections.append(
                    {
                        "hypothesis_id": f"{image_path.stem}_{box_index:03d}",
                        "pred_class": pred_class,
                        "bbox": bbox,
                        "score": score,
                        "class_scores": _fallback_class_scores(class_names, pred_class, score),
                    }
                )

        rows.append(
            {
                "image_id": image_path.stem,
                "image_path": str(image_path),
                "source": "ultralytics_yolo",
                "detections": detections,
            }
        )

    return rows


def _fallback_class_scores(class_names: list[str], pred_class: str, score: float) -> dict[str, float]:
    scores = {class_name: 0.0 for class_name in class_names}
    scores[pred_class] = score
    return scores
