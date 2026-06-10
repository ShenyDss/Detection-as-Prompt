from __future__ import annotations

import json
from typing import Any

from dap.prompt.schema import DetectionPrompt


class MockVLMReviewer:
    """Deterministic reviewer used for local pipeline development."""

    def generate(self, prompt: DetectionPrompt) -> str:
        fields = prompt.prompt_fields
        detector = fields["detector_hypothesis"]
        kg = fields["knowledge_context"]
        routing = fields.get("uncertainty_routing") or {}
        route = prompt.route_hint or routing.get("review_mode") or "class_comparison"

        status = _status_from_route(route)
        corrected_class = detector["category"]
        refined_bbox = fields["candidate_region"]["bbox_xyxy"]

        if status == "reject":
            corrected_class = None
            refined_bbox = None

        cause = _first(kg.get("causes", []))
        action = _first(kg.get("actions", []))
        risk = _first(kg.get("risk_notes", []))
        visual = _first(kg.get("visual_attributes", []), default="candidate region contains possible defect evidence")
        uncertainty = _uncertainty_sentence(routing)

        output: dict[str, Any] = {
            "hypothesis_status": status,
            "visual_evidence": visual,
            "uncertainty_evidence": uncertainty,
            "graph_path": [item for item in [corrected_class, cause, action] if item],
            "corrected_class": corrected_class,
            "refined_bbox": refined_bbox,
            "causes": [cause] if cause else [],
            "actions": [action] if action else [],
            "risks": [risk] if risk else [],
        }
        return json.dumps(output, ensure_ascii=False)


def _status_from_route(route: str) -> str:
    if route == "false_alarm_checking":
        return "reject"
    if route in {"class_comparison", "box_refinement"}:
        return "revise"
    return "verify"


def _first(values: list[Any], default: str = "") -> str:
    return str(values[0]) if values else default


def _uncertainty_sentence(routing: dict[str, Any]) -> str:
    if not routing:
        return "No explicit routing signal was provided."
    uncertainty = routing.get("uncertainty", {})
    reasons = routing.get("reasons", [])
    return (
        f"Route={routing.get('review_mode')}; "
        f"confidence={uncertainty.get('confidence')}; "
        f"class_margin={uncertainty.get('class_margin')}; "
        f"bbox_quality={uncertainty.get('bbox_quality')}; "
        f"reasons={reasons}"
    )
