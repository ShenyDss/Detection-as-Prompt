from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from dap.routing.schema import ReviewMode, RouteDecision
from dap.routing.uncertainty import compute_uncertainty_signals
from dap.schemas.core import DefectHypothesis


def route_hypotheses(
    hypotheses: Iterable[DefectHypothesis],
    *,
    route_config: dict[str, Any],
) -> list[RouteDecision]:
    return [route_single_hypothesis(hypothesis, route_config=route_config) for hypothesis in hypotheses]


def route_single_hypothesis(
    hypothesis: DefectHypothesis,
    *,
    route_config: dict[str, Any],
) -> RouteDecision:
    signals = compute_uncertainty_signals(hypothesis)
    high_conf = float(route_config.get("high_confidence_threshold", 0.8))
    low_conf = float(route_config.get("low_confidence_threshold", 0.35))
    ambiguous_margin = float(route_config.get("ambiguity_margin_threshold", 0.15))
    low_box_quality = float(route_config.get("low_box_quality_threshold", 0.5))

    reasons: list[str] = []
    if signals.bbox_quality < low_box_quality:
        mode = ReviewMode.BOX_REFINEMENT
        reasons.append(f"bbox_quality={signals.bbox_quality:.3f} < {low_box_quality:.3f}")
    elif signals.confidence < low_conf:
        mode = ReviewMode.FALSE_ALARM_CHECKING
        reasons.append(f"confidence={signals.confidence:.3f} < {low_conf:.3f}")
    elif signals.class_margin < ambiguous_margin:
        mode = ReviewMode.CLASS_COMPARISON
        reasons.append(f"class_margin={signals.class_margin:.3f} < {ambiguous_margin:.3f}")
    elif signals.confidence >= high_conf:
        mode = ReviewMode.LIGHT_VERIFICATION
        reasons.append(f"confidence={signals.confidence:.3f} >= {high_conf:.3f}")
    else:
        mode = ReviewMode.CLASS_COMPARISON
        reasons.append("medium-confidence hypothesis requires class comparison")

    return RouteDecision(
        image_id=hypothesis.image_id,
        hypothesis_id=hypothesis.hypothesis_id,
        review_mode=mode,
        uncertainty=signals,
        reasons=reasons,
    )
