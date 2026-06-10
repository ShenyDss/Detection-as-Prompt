from __future__ import annotations

from pathlib import Path


def run_mock_detector(image_paths: list[Path], class_names: list[str]) -> list[dict]:
    """Generate deterministic detector-like predictions for pipeline tests."""

    if not class_names:
        class_names = ["defect"]

    rows: list[dict] = []
    for image_index, image_path in enumerate(image_paths):
        first_class = class_names[image_index % len(class_names)]
        second_class = class_names[(image_index + 1) % len(class_names)]
        third_class = class_names[(image_index + 2) % len(class_names)]

        rows.append(
            {
                "image_id": image_path.stem,
                "image_path": str(image_path),
                "source": "mock_detector",
                "detections": [
                    {
                        "pred_class": first_class,
                        "bbox": [96 + 12 * image_index, 128, 260 + 12 * image_index, 288],
                        "score": 0.82,
                        "class_scores": _class_scores(class_names, first_class, second_class, 0.82, 0.11),
                    },
                    {
                        "pred_class": second_class,
                        "bbox": [420, 360 + 10 * image_index, 590, 520 + 10 * image_index],
                        "score": 0.42,
                        "class_scores": _class_scores(class_names, second_class, third_class, 0.42, 0.35),
                    },
                ],
            }
        )

    return rows


def _class_scores(
    class_names: list[str],
    pred_class: str,
    confusion_class: str,
    pred_score: float,
    confusion_score: float,
) -> dict[str, float]:
    scores = {class_name: 0.01 for class_name in class_names}
    scores[pred_class] = pred_score
    if confusion_class != pred_class:
        scores[confusion_class] = confusion_score
    return scores
