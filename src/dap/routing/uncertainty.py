from __future__ import annotations

from typing import Any

from dap.schemas.core import BBox, DefectHypothesis
from dap.routing.schema import UncertaintySignals


def compute_uncertainty_signals(hypothesis: DefectHypothesis) -> UncertaintySignals:
    class_scores = _class_scores(hypothesis)
    top_score, second_score = _top_two_scores(class_scores, hypothesis.score)
    margin = max(0.0, top_score - second_score)
    bbox_quality = _bbox_quality(hypothesis.bbox, hypothesis.detector_meta)
    return UncertaintySignals(
        confidence=float(hypothesis.score),
        class_margin=margin,
        bbox_quality=bbox_quality,
        top_class_score=top_score,
        second_class_score=second_score,
    )


def _class_scores(hypothesis: DefectHypothesis) -> dict[str, float]:
    raw_scores = dict(hypothesis.detector_meta.get("class_scores", {}))
    scores: dict[str, float] = {}
    for class_name, score in raw_scores.items():
        try:
            scores[str(class_name)] = float(score)
        except (TypeError, ValueError):
            continue
    if hypothesis.pred_class not in scores:
        scores[hypothesis.pred_class] = float(hypothesis.score)
    return scores


def _top_two_scores(scores: dict[str, float], fallback_score: float) -> tuple[float, float]:
    if not scores:
        return float(fallback_score), 0.0
    ranked = sorted(scores.values(), reverse=True)
    top = float(ranked[0])
    second = float(ranked[1]) if len(ranked) > 1 else 0.0
    return top, second


def _bbox_quality(bbox: BBox, detector_meta: dict[str, Any]) -> float:
    if "bbox_quality" in detector_meta:
        return _clip01(float(detector_meta["bbox_quality"]))

    width = max(0.0, bbox.x2 - bbox.x1)
    height = max(0.0, bbox.y2 - bbox.y1)
    if width <= 0 or height <= 0:
        return 0.0

    aspect = width / max(height, 1e-6)
    if aspect > 12.0 or aspect < 1.0 / 12.0:
        return 0.35

    area = width * height
    if area < 16:
        return 0.25
    return 0.85


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))
