from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from dap.evaluation.box_ops import bbox_iou
from dap.evaluation.ground_truth import GroundTruthDecision
from dap.kg import DefectKnowledgeGraph, check_graph_consistency
from dap.schemas.core import DefectHypothesis, HypothesisStatus, VLMDecision


@dataclass
class EvaluationReport:
    status: str
    metrics: dict[str, float | None]
    counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_predictions(
    *,
    hypotheses: list[DefectHypothesis],
    decisions: list[VLMDecision],
    graph: DefectKnowledgeGraph,
    ground_truth: dict[str, GroundTruthDecision] | None = None,
) -> EvaluationReport:
    gt = ground_truth or {}
    decision_by_id = {decision.hypothesis_id: decision for decision in decisions}
    paired = [(hypothesis, decision_by_id.get(hypothesis.hypothesis_id)) for hypothesis in hypotheses]
    paired = [(hypothesis, decision) for hypothesis, decision in paired if decision is not None]

    notes: list[str] = []
    if not gt:
        notes.append("No ground-truth file was provided; supervised metrics are reported as placeholder null values.")

    kg_results = [
        check_graph_consistency(
            graph,
            corrected_class=decision.corrected_class or hypothesis.pred_class,
            causes=decision.decision.causes,
            actions=decision.decision.actions,
        )
        for hypothesis, decision in paired
    ]
    graph_path_consistency = _mean([1.0 if result.is_consistent else 0.0 for result in kg_results])
    hallucination_rate = _mean([0.0 if result.is_consistent else 1.0 for result in kg_results])
    evidence_consistency = _mean([_evidence_present(decision) for _, decision in paired])

    metrics: dict[str, float | None] = {
        "Defect-F1": None,
        "Corr-Cls": _corrected_class_accuracy(paired, gt) if gt else None,
        "MCR": _misclassification_correction_rate(paired, gt) if gt else None,
        "FPRed": _false_positive_reduction(paired, gt) if gt else None,
        "Ref-IoU": _refinement_iou(paired, gt) if gt else None,
        "HSA": _status_accuracy(paired, gt) if gt else None,
        "FRA": _false_alarm_rejection_accuracy(paired, gt) if gt else None,
        "Rev-P": _revision_precision(paired, gt) if gt else None,
        "Rev-R": _revision_recall(paired, gt) if gt else None,
        "CMA": _cause_matching_accuracy(paired, gt) if gt else None,
        "ARA": _action_recommendation_accuracy(paired, gt) if gt else None,
        "GPC": graph_path_consistency,
        "EC": evidence_consistency,
        "HR": hallucination_rate,
    }
    return EvaluationReport(
        status="ready" if gt else "partial",
        metrics=metrics,
        counts={
            "hypotheses": len(hypotheses),
            "decisions": len(decisions),
            "paired": len(paired),
            "ground_truth": len(gt),
        },
        notes=notes,
    )


def _status_accuracy(paired, gt: dict[str, GroundTruthDecision]) -> float | None:
    values = []
    for _, decision in paired:
        target = gt.get(decision.hypothesis_id)
        if target:
            values.append(1.0 if decision.hypothesis_status == target.status else 0.0)
    return _mean(values)


def _corrected_class_accuracy(paired, gt: dict[str, GroundTruthDecision]) -> float | None:
    values = []
    for hypothesis, decision in paired:
        target = gt.get(decision.hypothesis_id)
        if target and target.corrected_class:
            predicted = decision.corrected_class or hypothesis.pred_class
            values.append(1.0 if predicted == target.corrected_class else 0.0)
    return _mean(values)


def _misclassification_correction_rate(paired, gt: dict[str, GroundTruthDecision]) -> float | None:
    values = []
    for hypothesis, decision in paired:
        target = gt.get(decision.hypothesis_id)
        if target and target.corrected_class and hypothesis.pred_class != target.corrected_class:
            values.append(1.0 if decision.corrected_class == target.corrected_class else 0.0)
    return _mean(values)


def _false_positive_reduction(paired, gt: dict[str, GroundTruthDecision]) -> float | None:
    values = []
    for _, decision in paired:
        target = gt.get(decision.hypothesis_id)
        if target and target.status == HypothesisStatus.REJECT:
            values.append(1.0 if decision.hypothesis_status == HypothesisStatus.REJECT else 0.0)
    return _mean(values)


def _refinement_iou(paired, gt: dict[str, GroundTruthDecision]) -> float | None:
    values = []
    for hypothesis, decision in paired:
        target = gt.get(decision.hypothesis_id)
        if target and target.bbox:
            predicted_box = decision.refined_bbox or hypothesis.bbox
            values.append(bbox_iou(predicted_box, target.bbox))
    return _mean(values)


def _false_alarm_rejection_accuracy(paired, gt: dict[str, GroundTruthDecision]) -> float | None:
    return _false_positive_reduction(paired, gt)


def _revision_precision(paired, gt: dict[str, GroundTruthDecision]) -> float | None:
    values = []
    for _, decision in paired:
        if decision.hypothesis_status == HypothesisStatus.REVISE:
            target = gt.get(decision.hypothesis_id)
            if target:
                values.append(1.0 if target.status == HypothesisStatus.REVISE else 0.0)
    return _mean(values)


def _revision_recall(paired, gt: dict[str, GroundTruthDecision]) -> float | None:
    values = []
    for _, decision in paired:
        target = gt.get(decision.hypothesis_id)
        if target and target.status == HypothesisStatus.REVISE:
            values.append(1.0 if decision.hypothesis_status == HypothesisStatus.REVISE else 0.0)
    return _mean(values)


def _cause_matching_accuracy(paired, gt: dict[str, GroundTruthDecision]) -> float | None:
    values = []
    for _, decision in paired:
        target = gt.get(decision.hypothesis_id)
        if target and target.causes:
            values.append(1.0 if set(decision.decision.causes) & set(target.causes) else 0.0)
    return _mean(values)


def _action_recommendation_accuracy(paired, gt: dict[str, GroundTruthDecision]) -> float | None:
    values = []
    for _, decision in paired:
        target = gt.get(decision.hypothesis_id)
        if target and target.actions:
            values.append(1.0 if set(decision.decision.actions) & set(target.actions) else 0.0)
    return _mean(values)


def _evidence_present(decision: VLMDecision) -> float:
    fields = [
        bool(decision.evidence.visual),
        bool(decision.evidence.uncertainty),
        bool(decision.evidence.graph_path),
    ]
    return sum(1 for item in fields if item) / len(fields)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
