from __future__ import annotations

import csv
import json
from pathlib import Path

from dap.evaluation.metrics import EvaluationReport


MAIN_METRICS = ["Defect-F1", "Corr-Cls", "MCR", "FPRed", "Ref-IoU", "HSA", "FRA", "Rev-P", "Rev-R"]
DECISION_METRICS = ["CMA", "ARA", "GPC", "EC", "HR"]


def write_metrics_json(path: str | Path, report: EvaluationReport) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report.to_dict(), file, ensure_ascii=False, indent=2)


def write_metrics_csv(path: str | Path, report: EvaluationReport, *, method_name: str = "Full model") -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "metric", "value"])
        for metric, value in report.metrics.items():
            writer.writerow([method_name, metric, _format_metric(value)])


def write_latex_tables(path: str | Path, report: EvaluationReport, *, method_name: str = "Full model") -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n\n".join(
        [
            _latex_table(
                caption="Main comparison metrics exported from the current run.",
                label="tab:auto_main_results",
                method_name=method_name,
                metrics=MAIN_METRICS,
                report=report,
            ),
            _latex_table(
                caption="Decision quality metrics exported from the current run.",
                label="tab:auto_decision_quality",
                method_name=method_name,
                metrics=DECISION_METRICS,
                report=report,
            ),
        ]
    )
    output_path.write_text(text, encoding="utf-8")


def _latex_table(
    *,
    caption: str,
    label: str,
    method_name: str,
    metrics: list[str],
    report: EvaluationReport,
) -> str:
    header = "Method & " + " & ".join(metrics) + r" \\"
    values = [method_name] + [_format_metric(report.metrics.get(metric), latex=True) for metric in metrics]
    row = " & ".join(values) + r" \\"
    columns = "l" + "c" * len(metrics)
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            rf"\begin{{tabular}}{{{columns}}}",
            r"\hline",
            header,
            r"\hline",
            row,
            r"\hline",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )


def _format_metric(value: float | None, *, latex: bool = False) -> str:
    if value is None:
        return "--" if latex else ""
    return f"{value:.3f}"
