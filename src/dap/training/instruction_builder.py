from __future__ import annotations

from dap.evaluation.ground_truth import GroundTruthDecision
from dap.prompt.schema import DetectionPrompt
from dap.schemas.core import DefectHypothesis, HypothesisStatus, VLMDecision
from dap.training.schema import InstructionSample


def build_instruction_samples(
    *,
    prompts: list[DetectionPrompt],
    hypotheses: list[DefectHypothesis],
    decisions: list[VLMDecision],
    ground_truth: dict[str, GroundTruthDecision],
) -> list[InstructionSample]:
    hypothesis_lookup = {item.hypothesis_id: item for item in hypotheses}
    decision_lookup = {item.hypothesis_id: item for item in decisions}
    samples: list[InstructionSample] = []

    for prompt in prompts:
        hypothesis = hypothesis_lookup.get(prompt.hypothesis_id)
        if hypothesis is None:
            continue
        target = _target_from_ground_truth_or_decision(
            prompt.hypothesis_id,
            decision_lookup.get(prompt.hypothesis_id),
            ground_truth.get(prompt.hypothesis_id),
        )
        samples.append(
            InstructionSample(
                sample_id=f"{prompt.hypothesis_id}_instruction",
                image_id=prompt.image_id,
                hypothesis_id=prompt.hypothesis_id,
                image_path=prompt.image_path,
                region_path=prompt.region_path,
                prompt=prompt.prompt_text,
                target=target,
                metadata={
                    "pred_class": hypothesis.pred_class,
                    "detector_score": hypothesis.score,
                    "bbox": hypothesis.bbox.to_list(),
                    "confusion_candidates": hypothesis.confusion_candidates,
                    "source": "ground_truth" if prompt.hypothesis_id in ground_truth else "model_decision",
                },
            )
        )
    return samples


def _target_from_ground_truth_or_decision(
    hypothesis_id: str,
    decision: VLMDecision | None,
    target: GroundTruthDecision | None,
) -> dict:
    if target is not None:
        status = target.status.value
        corrected_class = target.corrected_class
        return {
            "hypothesis_status": status,
            "visual_evidence": _template_visual_evidence(status, corrected_class),
            "uncertainty_evidence": _template_uncertainty_evidence(status),
            "graph_path": [item for item in [corrected_class, *target.causes[:1], *target.actions[:1]] if item],
            "corrected_class": corrected_class,
            "refined_bbox": target.bbox.to_list() if target.bbox else None,
            "causes": target.causes,
            "actions": target.actions,
            "risks": [],
        }
    if decision is not None:
        return {
            "hypothesis_status": decision.hypothesis_status.value,
            "visual_evidence": decision.evidence.visual,
            "uncertainty_evidence": decision.evidence.uncertainty,
            "graph_path": decision.evidence.graph_path,
            "corrected_class": decision.corrected_class,
            "refined_bbox": decision.refined_bbox.to_list() if decision.refined_bbox else None,
            "causes": decision.decision.causes,
            "actions": decision.decision.actions,
            "risks": decision.decision.risks,
        }
    return {
        "hypothesis_status": HypothesisStatus.REVISE.value,
        "visual_evidence": "",
        "uncertainty_evidence": "",
        "graph_path": [],
        "corrected_class": None,
        "refined_bbox": None,
        "causes": [],
        "actions": [],
        "risks": [],
    }


def _template_visual_evidence(status: str, corrected_class: str | None) -> str:
    if status == HypothesisStatus.REJECT.value:
        return "The detector hypothesis is not sufficiently supported by the matched ground-truth region."
    if status == HypothesisStatus.REVISE.value and corrected_class:
        return f"The region overlaps a defect annotation, but the detector category should be revised to {corrected_class}."
    if corrected_class:
        return f"The candidate region overlaps the matched {corrected_class} annotation."
    return "The candidate region is reviewed against the matched annotation."


def _template_uncertainty_evidence(status: str) -> str:
    if status == HypothesisStatus.REJECT.value:
        return "No ground-truth box reaches the matching IoU threshold, so the hypothesis is treated as a false alarm."
    if status == HypothesisStatus.REVISE.value:
        return "The localization matches a ground-truth defect, but the predicted class differs from the annotation."
    return "The localization and class agree with the matched ground-truth annotation."
