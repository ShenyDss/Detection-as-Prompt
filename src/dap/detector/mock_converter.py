from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from dap.schemas.core import BBox, DefectHypothesis


def convert_mock_predictions(
    predictions: Iterable[dict[str, Any]],
    *,
    top_k_confusions: int = 3,
    score_threshold: float = 0.0,
) -> list[DefectHypothesis]:
    """Convert detector-like dictionaries into paper-style hypotheses.

    Expected input row shape:
    {
      "image_id": "img_001",
      "detections": [
        {
          "pred_class": "broken_weft",
          "bbox": [x1, y1, x2, y2],
          "score": 0.72,
          "class_scores": {"broken_weft": 0.72, "loose_weft": 0.18}
        }
      ]
    }
    """

    hypotheses: list[DefectHypothesis] = []
    for row in predictions:
        image_id = str(row["image_id"])
        detections = row.get("detections", [])
        if not isinstance(detections, list):
            raise ValueError(f"Detections must be a list for image_id={image_id}")

        for index, detection in enumerate(detections):
            score = float(detection["score"])
            if score < score_threshold:
                continue

            pred_class = str(detection["pred_class"])
            class_scores = dict(detection.get("class_scores", {}))
            confusion_candidates = _top_confusion_candidates(
                class_scores=class_scores,
                pred_class=pred_class,
                top_k=top_k_confusions,
            )
            hypothesis_id = str(detection.get("hypothesis_id", f"{image_id}_{index:03d}"))

            hypotheses.append(
                DefectHypothesis(
                    image_id=image_id,
                    hypothesis_id=hypothesis_id,
                    pred_class=pred_class,
                    bbox=BBox.from_list(detection["bbox"]),
                    score=score,
                    confusion_candidates=confusion_candidates,
                    detector_meta={
                        "source": row.get("source", "mock_detector"),
                        "detection_index": index,
                        "class_scores": class_scores,
                    },
                )
            )

    return hypotheses


def _top_confusion_candidates(
    *,
    class_scores: dict[str, Any],
    pred_class: str,
    top_k: int,
) -> list[str]:
    if top_k <= 0 or not class_scores:
        return []

    ranked = sorted(
        ((str(class_name), float(score)) for class_name, score in class_scores.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    return [class_name for class_name, _ in ranked if class_name != pred_class][:top_k]
