from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dap.evaluation.ground_truth import GroundTruthDecision, load_ground_truth
from dap.kg.schema import DefectKnowledgeGraph
from dap.schemas.core import BBox, DefectHypothesis, HypothesisStatus
from dap.schemas.hypothesis_io import load_hypotheses
from dap.utils.jsonl import read_jsonl


@dataclass
class DataValidationIssue:
    level: str
    code: str
    message: str
    item_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataValidationReport:
    status: str
    counts: dict[str, int] = field(default_factory=dict)
    issues: list[DataValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "counts": self.counts,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def validate_real_data(
    *,
    project_root: Path,
    graph: DefectKnowledgeGraph,
    hypotheses_path: Path,
    raw_predictions_path: Path,
    annotations_path: Path,
    image_dir: Path,
) -> DataValidationReport:
    issues: list[DataValidationIssue] = []
    hypotheses = _load_hypotheses_safe(hypotheses_path, issues)
    ground_truth = _load_ground_truth_safe(annotations_path, issues)
    raw_rows = _load_raw_predictions_safe(raw_predictions_path, issues)
    image_lookup = _build_image_lookup(project_root, raw_rows, image_dir)
    class_names = set(graph.classes)

    _check_images(image_dir, image_lookup, issues)
    _check_hypotheses(hypotheses, class_names, image_lookup, issues)
    _check_ground_truth(ground_truth, hypotheses, class_names, graph, issues)

    error_count = sum(1 for issue in issues if issue.level == "error")
    warning_count = sum(1 for issue in issues if issue.level == "warning")
    status = "ready" if error_count == 0 else "blocked"
    return DataValidationReport(
        status=status,
        counts={
            "images": len({path for path in image_lookup.values() if path is not None}),
            "raw_prediction_rows": len(raw_rows),
            "hypotheses": len(hypotheses),
            "ground_truth": len(ground_truth),
            "errors": error_count,
            "warnings": warning_count,
        },
        issues=issues,
    )


def _load_hypotheses_safe(path: Path, issues: list[DataValidationIssue]) -> list[DefectHypothesis]:
    if not path.exists():
        issues.append(_issue("error", "missing_hypotheses", f"Hypothesis file not found: {path}"))
        return []
    try:
        return load_hypotheses(path)
    except Exception as exc:  # noqa: BLE001
        issues.append(_issue("error", "invalid_hypotheses", f"Cannot parse hypotheses: {exc}"))
        return []


def _load_ground_truth_safe(
    path: Path,
    issues: list[DataValidationIssue],
) -> dict[str, GroundTruthDecision]:
    if not path.exists():
        issues.append(_issue("warning", "missing_annotations", f"Annotation file not found: {path}"))
        return {}
    try:
        return load_ground_truth(path)
    except Exception as exc:  # noqa: BLE001
        issues.append(_issue("error", "invalid_annotations", f"Cannot parse annotations: {exc}"))
        return {}


def _load_raw_predictions_safe(path: Path, issues: list[DataValidationIssue]) -> list[dict[str, Any]]:
    if not path.exists():
        issues.append(_issue("warning", "missing_raw_predictions", f"Raw prediction file not found: {path}"))
        return []
    try:
        return list(read_jsonl(path))
    except Exception as exc:  # noqa: BLE001
        issues.append(_issue("error", "invalid_raw_predictions", f"Cannot parse raw predictions: {exc}"))
        return []


def _build_image_lookup(
    project_root: Path,
    raw_rows: list[dict[str, Any]],
    image_dir: Path,
) -> dict[str, Path | None]:
    lookup: dict[str, Path | None] = {}
    for row in raw_rows:
        image_id = str(row.get("image_id", ""))
        raw_path = row.get("image_path")
        lookup[image_id] = _resolve_path(project_root, raw_path) if raw_path else None

    for image_path in image_dir.glob("*"):
        if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            lookup.setdefault(image_path.stem, image_path.resolve())
    return lookup


def _check_images(
    image_dir: Path,
    image_lookup: dict[str, Path | None],
    issues: list[DataValidationIssue],
) -> None:
    if not image_dir.exists():
        issues.append(_issue("error", "missing_image_dir", f"Image directory not found: {image_dir}"))
        return
    if not any(image_dir.glob("*")):
        issues.append(_issue("warning", "empty_image_dir", f"No image files found in: {image_dir}"))
    for image_id, image_path in image_lookup.items():
        if image_path is None or not image_path.exists():
            issues.append(_issue("warning", "missing_image_file", f"Image path not found for {image_id}", image_id))


def _check_hypotheses(
    hypotheses: list[DefectHypothesis],
    class_names: set[str],
    image_lookup: dict[str, Path | None],
    issues: list[DataValidationIssue],
) -> None:
    seen: set[str] = set()
    for hypothesis in hypotheses:
        item_id = hypothesis.hypothesis_id
        if item_id in seen:
            issues.append(_issue("error", "duplicate_hypothesis_id", f"Duplicate hypothesis_id: {item_id}", item_id))
        seen.add(item_id)
        if hypothesis.pred_class not in class_names:
            issues.append(_issue("error", "unknown_pred_class", f"Class not found in KG: {hypothesis.pred_class}", item_id))
        for candidate in hypothesis.confusion_candidates:
            if candidate not in class_names:
                issues.append(_issue("warning", "unknown_confusion_class", f"Confusion class not in KG: {candidate}", item_id))
        if not 0.0 <= hypothesis.score <= 1.0:
            issues.append(_issue("error", "invalid_score", f"Score must be in [0, 1], got {hypothesis.score}", item_id))
        _check_bbox(hypothesis.bbox, item_id, issues)
        image_path = image_lookup.get(hypothesis.image_id)
        if image_path is None:
            issues.append(_issue("warning", "hypothesis_image_unresolved", f"No image path resolved for image_id={hypothesis.image_id}", item_id))


def _check_ground_truth(
    ground_truth: dict[str, GroundTruthDecision],
    hypotheses: list[DefectHypothesis],
    class_names: set[str],
    graph: DefectKnowledgeGraph,
    issues: list[DataValidationIssue],
) -> None:
    hypothesis_ids = {hypothesis.hypothesis_id for hypothesis in hypotheses}
    for hypothesis_id in hypothesis_ids:
        if hypothesis_id not in ground_truth:
            issues.append(_issue("warning", "missing_ground_truth", f"No annotation for hypothesis_id={hypothesis_id}", hypothesis_id))

    for item_id, decision in ground_truth.items():
        if item_id not in hypothesis_ids:
            issues.append(_issue("error", "orphan_ground_truth", f"Annotation has no matching hypothesis: {item_id}", item_id))
        if decision.status in {HypothesisStatus.VERIFY, HypothesisStatus.REVISE} and not decision.corrected_class:
            issues.append(_issue("error", "missing_corrected_class", "verify/revise annotations require corrected_class", item_id))
        if decision.corrected_class and decision.corrected_class not in class_names:
            issues.append(_issue("error", "unknown_corrected_class", f"Corrected class not found in KG: {decision.corrected_class}", item_id))
        if decision.status in {HypothesisStatus.VERIFY, HypothesisStatus.REVISE} and decision.bbox is None:
            issues.append(_issue("warning", "missing_refined_bbox", "verify/revise annotations should include bbox or refined_bbox", item_id))
        if decision.bbox is not None:
            _check_bbox(decision.bbox, item_id, issues)
        _check_graph_labels(decision, graph, issues)


def _check_graph_labels(
    decision: GroundTruthDecision,
    graph: DefectKnowledgeGraph,
    issues: list[DataValidationIssue],
) -> None:
    if not decision.corrected_class or decision.corrected_class not in graph.classes:
        return
    node = graph.classes[decision.corrected_class]
    for cause in decision.causes:
        if cause not in node.causes:
            issues.append(_issue("warning", "cause_not_in_class_kg", f"Cause not listed for {decision.corrected_class}: {cause}", decision.hypothesis_id))
    for action in decision.actions:
        if action not in node.actions:
            issues.append(_issue("warning", "action_not_in_class_kg", f"Action not listed for {decision.corrected_class}: {action}", decision.hypothesis_id))


def _check_bbox(bbox: BBox, item_id: str, issues: list[DataValidationIssue]) -> None:
    if bbox.x1 < 0 or bbox.y1 < 0:
        issues.append(_issue("error", "negative_bbox", f"BBox has negative coordinate: {bbox.to_list()}", item_id))
    if bbox.x2 <= bbox.x1 or bbox.y2 <= bbox.y1:
        issues.append(_issue("error", "invalid_bbox_order", f"BBox must satisfy x2>x1 and y2>y1: {bbox.to_list()}", item_id))


def _resolve_path(project_root: Path, raw_path: Any) -> Path:
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _issue(level: str, code: str, message: str, item_id: str | None = None) -> DataValidationIssue:
    return DataValidationIssue(level=level, code=code, message=message, item_id=item_id)
