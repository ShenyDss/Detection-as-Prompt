from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dap.config import load_app_config
from dap.evaluation import evaluate_predictions, write_latex_tables, write_metrics_csv
from dap.evaluation.ground_truth import load_ground_truth
from dap.evaluation.table_export import write_metrics_json
from dap.kg import load_defect_kg
from dap.schemas.hypothesis_io import load_hypotheses
from dap.vlm.output_io import load_decisions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Detection-as-Prompt predictions.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--hypotheses", default="data/detector_predictions/hypotheses.jsonl")
    parser.add_argument("--decisions", default="outputs/predictions/vlm_decisions.jsonl")
    parser.add_argument("--ground-truth", default="data/annotations/test_decisions.jsonl")
    parser.add_argument("--kg-path", default=None, help="Override the knowledge graph path from configs/kg.yaml.")
    parser.add_argument("--metrics-json", default="outputs/metrics/metrics.json")
    parser.add_argument("--metrics-csv", default="outputs/metrics/summary.csv")
    parser.add_argument("--latex-table", default="outputs/tables/evaluation_tables.tex")
    parser.add_argument("--method-name", default="Full model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(PROJECT_ROOT / args.config_dir)
    kg_path = args.kg_path or config.kg["path"]
    graph = load_defect_kg(PROJECT_ROOT / kg_path)
    hypotheses = load_hypotheses(PROJECT_ROOT / args.hypotheses)
    decisions = load_decisions(PROJECT_ROOT / args.decisions)
    ground_truth = load_ground_truth(PROJECT_ROOT / args.ground_truth)

    report = evaluate_predictions(
        hypotheses=hypotheses,
        decisions=decisions,
        graph=graph,
        ground_truth=ground_truth,
    )
    write_metrics_json(PROJECT_ROOT / args.metrics_json, report)
    write_metrics_csv(PROJECT_ROOT / args.metrics_csv, report, method_name=args.method_name)
    write_latex_tables(PROJECT_ROOT / args.latex_table, report, method_name=args.method_name)

    print(f"Evaluation status: {report.status}")
    print(f"Wrote metrics JSON to {PROJECT_ROOT / args.metrics_json}")
    print(f"Wrote metrics CSV to {PROJECT_ROOT / args.metrics_csv}")
    print(f"Wrote LaTeX tables to {PROJECT_ROOT / args.latex_table}")
    print("Metrics:")
    for metric, value in report.metrics.items():
        print(f"  {metric}: {'--' if value is None else f'{value:.3f}'}")


if __name__ == "__main__":
    main()
