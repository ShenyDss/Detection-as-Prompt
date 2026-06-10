from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from dap.evaluation.ground_truth import GroundTruthDecision
from dap.evaluation.metrics import EvaluationReport, evaluate_predictions
from dap.kg import DefectKnowledgeGraph
from dap.prompt import build_detection_prompts
from dap.prompt.io import write_prompts_jsonl
from dap.routing.schema import RouteDecision
from dap.schemas.core import DefectHypothesis, HypothesisStatus, VLMDecision
from dap.vlm import run_vlm_review
from dap.vlm.output_io import write_decisions_jsonl


ABLATION_VARIANTS: dict[str, dict[str, bool]] = {
    "Full model": {
        "detection_prompt": True,
        "self_verification": True,
        "uncertainty_routing": True,
        "confusion_candidates": True,
        "kg_constraint": True,
        "instruction_tuning": False,
    },
    "w/o detection prompt": {"detection_prompt": False},
    "w/o self-verification": {"self_verification": False},
    "w/o uncertainty routing": {"uncertainty_routing": False},
    "w/o confusion candidates": {"confusion_candidates": False},
    "w/o KG constraint": {"kg_constraint": False},
    "w/o instruction tuning": {"instruction_tuning": False},
}


def run_ablation_suite(
    *,
    hypotheses: list[DefectHypothesis],
    graph: DefectKnowledgeGraph,
    ground_truth: dict[str, GroundTruthDecision],
    image_lookup: dict[str, str | None],
    route_lookup: dict[str, RouteDecision],
    kg_retrieval_config: dict[str, Any],
    vlm_config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, EvaluationReport]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    reports: dict[str, EvaluationReport] = {}

    for variant_name, overrides in ABLATION_VARIANTS.items():
        enabled_modules = _variant_modules(overrides)
        variant_dir = output_root / _slug(variant_name)
        variant_dir.mkdir(parents=True, exist_ok=True)

        active_route_lookup = route_lookup if enabled_modules["uncertainty_routing"] else {}
        prompts = build_detection_prompts(
            hypotheses,
            graph=graph,
            image_lookup=image_lookup,
            route_lookup=active_route_lookup,
            region_output_dir=variant_dir / "regions",
            kg_retrieval_config=kg_retrieval_config,
            enabled_modules=enabled_modules,
        )
        decisions = run_vlm_review(prompts, vlm_config=vlm_config)
        decisions = _apply_variant_decision_effects(decisions, enabled_modules)
        report = evaluate_predictions(
            hypotheses=hypotheses,
            decisions=decisions,
            graph=graph,
            ground_truth=ground_truth,
        )

        write_prompts_jsonl(variant_dir / "prompts.jsonl", prompts)
        write_decisions_jsonl(variant_dir / "decisions.jsonl", decisions)
        reports[variant_name] = report

    return reports


def write_ablation_csv(path: str | Path, reports: dict[str, EvaluationReport]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = _metric_names(reports)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Variant", *metric_names])
        for variant_name, report in reports.items():
            writer.writerow([variant_name, *[_format(report.metrics.get(metric)) for metric in metric_names]])


def write_ablation_latex(path: str | Path, reports: dict[str, EvaluationReport]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["Defect-F1", "HSA", "GPC", "HR"]
    columns = "l" + "c" * len(metrics)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Ablation study exported from the current run.}",
        r"\label{tab:auto_ablation}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\hline",
        "Variant & " + " & ".join(metrics) + r" \\",
        r"\hline",
    ]
    for variant_name, report in reports.items():
        values = [_format(report.metrics.get(metric), latex=True) for metric in metrics]
        lines.append(variant_name + " & " + " & ".join(values) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _variant_modules(overrides: dict[str, bool]) -> dict[str, bool]:
    modules = dict(ABLATION_VARIANTS["Full model"])
    modules.update(overrides)
    return modules


def _apply_variant_decision_effects(
    decisions: list[VLMDecision],
    enabled_modules: dict[str, bool],
) -> list[VLMDecision]:
    if enabled_modules["self_verification"]:
        return decisions

    adjusted: list[VLMDecision] = []
    for decision in decisions:
        adjusted.append(
            VLMDecision(
                image_id=decision.image_id,
                hypothesis_id=decision.hypothesis_id,
                hypothesis_status=HypothesisStatus.VERIFY,
                corrected_class=decision.corrected_class,
                refined_bbox=decision.refined_bbox,
                evidence=decision.evidence,
                decision=decision.decision,
                raw_output=decision.raw_output,
            )
        )
    return adjusted


def _metric_names(reports: dict[str, EvaluationReport]) -> list[str]:
    for report in reports.values():
        return list(report.metrics.keys())
    return []


def _format(value: float | None, *, latex: bool = False) -> str:
    if value is None:
        return "--" if latex else ""
    return f"{value:.3f}"


def _slug(name: str) -> str:
    return (
        name.lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("w/o", "without")
    )
