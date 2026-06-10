from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from dap.kg import DefectKnowledgeGraph, retrieve_category_context
from dap.prompt.cropper import crop_hypothesis_region
from dap.prompt.schema import DetectionPrompt
from dap.routing.schema import RouteDecision
from dap.schemas.core import DefectHypothesis


def build_detection_prompts(
    hypotheses: Iterable[DefectHypothesis],
    *,
    graph: DefectKnowledgeGraph,
    image_lookup: dict[str, str | None] | None = None,
    route_lookup: dict[str, RouteDecision] | None = None,
    region_output_dir: str | Path = "outputs/regions",
    kg_retrieval_config: dict[str, Any] | None = None,
    enabled_modules: dict[str, bool] | None = None,
) -> list[DetectionPrompt]:
    prompts: list[DetectionPrompt] = []
    for hypothesis in hypotheses:
        prompts.append(
            build_detection_prompt(
                hypothesis,
                graph=graph,
                image_path=(image_lookup or {}).get(hypothesis.image_id),
                route_decision=(route_lookup or {}).get(hypothesis.hypothesis_id),
                region_output_dir=region_output_dir,
                kg_retrieval_config=kg_retrieval_config,
                enabled_modules=enabled_modules,
            )
        )
    return prompts


def build_detection_prompt(
    hypothesis: DefectHypothesis,
    *,
    graph: DefectKnowledgeGraph,
    image_path: str | None = None,
    route_decision: RouteDecision | None = None,
    region_output_dir: str | Path = "outputs/regions",
    kg_retrieval_config: dict[str, Any] | None = None,
    route_hint: str | None = None,
    enabled_modules: dict[str, bool] | None = None,
) -> DetectionPrompt:
    modules = _normalize_modules(enabled_modules)
    retrieval_config = kg_retrieval_config or {}
    if modules["kg_constraint"]:
        kg_context = retrieve_category_context(
            graph,
            hypothesis.pred_class,
            max_visual_attributes=int(retrieval_config.get("max_visual_attributes", 6)),
            max_confusion_classes=int(retrieval_config.get("max_confusion_classes", 3)),
            max_causes=int(retrieval_config.get("max_causes", 5)),
            max_actions=int(retrieval_config.get("max_actions", 5)),
        ).to_dict()
    else:
        kg_context = {
            "class_name": hypothesis.pred_class,
            "visual_attributes": [],
            "confusion_classes": [],
            "causes": [],
            "actions": [],
            "risk_notes": [],
            "cause_action_paths": [],
        }
    region_path = crop_hypothesis_region(
        image_path=image_path,
        bbox=hypothesis.bbox,
        output_dir=region_output_dir,
        output_name=hypothesis.hypothesis_id,
    )

    fields = {
        "candidate_region": {
            "bbox_xyxy": hypothesis.bbox.to_list() if modules["detection_prompt"] else None,
            "region_path": str(region_path) if region_path else None,
        },
        "detector_hypothesis": {
            "category": hypothesis.pred_class if modules["detection_prompt"] else None,
            "confidence": hypothesis.score if modules["detection_prompt"] else None,
            "confusion_candidates": hypothesis.confusion_candidates if modules["confusion_candidates"] else [],
            "detector_meta": hypothesis.detector_meta,
        },
        "uncertainty_routing": route_decision.to_dict() if route_decision and modules["uncertainty_routing"] else None,
        "knowledge_context": kg_context,
        "enabled_modules": modules,
        "required_output_schema": {
            "hypothesis_status": "verify | reject | revise",
            "visual_evidence": "short visual evidence grounded in candidate region",
            "uncertainty_evidence": "how confidence, ambiguity, or box quality affects review",
            "graph_path": ["defect_class", "cause", "action"],
            "corrected_class": "corrected defect class or null",
            "refined_bbox": "[x1, y1, x2, y2] or null",
            "causes": ["graph-grounded cause"],
            "actions": ["graph-grounded action"],
            "risks": ["risk note"],
        },
    }
    prompt_text = _render_prompt(
        image_id=hypothesis.image_id,
        hypothesis_id=hypothesis.hypothesis_id,
        fields=fields,
        route_hint=route_hint
        or (route_decision.review_mode.value if route_decision and modules["uncertainty_routing"] else None),
    )
    return DetectionPrompt(
        image_id=hypothesis.image_id,
        hypothesis_id=hypothesis.hypothesis_id,
        image_path=image_path,
        region_path=str(region_path) if region_path else None,
        route_hint=route_hint
        or (route_decision.review_mode.value if route_decision and modules["uncertainty_routing"] else None),
        prompt_text=prompt_text,
        prompt_fields=fields,
    )


def _render_prompt(
    *,
    image_id: str,
    hypothesis_id: str,
    fields: dict[str, Any],
    route_hint: str | None,
) -> str:
    detector = fields["detector_hypothesis"]
    region = fields["candidate_region"]
    routing = fields.get("uncertainty_routing")
    knowledge = fields["knowledge_context"]
    modules = fields["enabled_modules"]
    schema = fields["required_output_schema"]

    if modules["detection_prompt"]:
        detector_text = (
            "You are an auditable industrial defect reviewer.\n"
            f"Image id: {image_id}.\n"
            f"Hypothesis id: {hypothesis_id}.\n"
            f"Candidate region bbox xyxy: {region['bbox_xyxy']}.\n"
            f"Detector hypothesis: category={detector['category']}, confidence={detector['confidence']:.4f}.\n"
        )
    else:
        detector_text = (
            "You are an auditable industrial defect reviewer.\n"
            f"Image id: {image_id}.\n"
            f"Hypothesis id: {hypothesis_id}.\n"
            "Detector hypothesis: disabled for this ablation.\n"
        )
    detector_text += (
        f"Confusion candidates: {detector['confusion_candidates']}."
        if modules["confusion_candidates"]
        else "Confusion candidates: disabled for this ablation."
    )

    if routing:
        uncertainty = routing["uncertainty"]
        route_line = (
            f"\nRoute hint: {route_hint}."
            f"\nUncertainty signals: confidence={uncertainty['confidence']:.4f}, "
            f"class_margin={uncertainty['class_margin']:.4f}, "
            f"bbox_quality={uncertainty['bbox_quality']:.4f}."
            f"\nRouting reasons: {routing['reasons']}."
        )
    else:
        route_line = f"\nRoute hint: {route_hint}." if route_hint else ""

    if modules["kg_constraint"]:
        kg_text = (
            "Knowledge context:\n"
            f"- Visual attributes: {knowledge['visual_attributes']}\n"
            f"- Confusion classes: {knowledge['confusion_classes']}\n"
            f"- Possible causes: {knowledge['causes']}\n"
            f"- Candidate actions: {knowledge['actions']}\n"
            f"- Risk notes: {knowledge['risk_notes']}\n"
            f"- Valid cause-action paths: {knowledge['cause_action_paths']}\n\n"
        )
    else:
        kg_text = "Knowledge context: disabled for this ablation.\n\n"

    return (
        detector_text
        + f"{route_line}\n\n"
        + kg_text
        + "Tasks:\n"
        + "1. Verify whether the detector hypothesis is visually supported.\n"
        + "2. Reject the hypothesis if the region is likely a false alarm.\n"
        + "3. Revise the category or bbox if the defect exists but the detector output is wrong.\n"
        + "4. Generate only graph-grounded causes, actions, and risks.\n\n"
        + "Return a JSON object with this schema:\n"
        + f"{schema}\n"
    )


def _normalize_modules(enabled_modules: dict[str, bool] | None) -> dict[str, bool]:
    defaults = {
        "detection_prompt": True,
        "self_verification": True,
        "uncertainty_routing": True,
        "confusion_candidates": True,
        "kg_constraint": True,
        "instruction_tuning": False,
    }
    if enabled_modules:
        defaults.update({key: bool(value) for key, value in enabled_modules.items()})
    return defaults
