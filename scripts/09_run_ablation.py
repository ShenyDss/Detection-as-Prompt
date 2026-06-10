from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dap.ablation.runner import run_ablation_suite, write_ablation_csv, write_ablation_latex
from dap.config import load_app_config
from dap.evaluation.ground_truth import load_ground_truth
from dap.kg import load_defect_kg
from dap.routing.io import load_route_lookup
from dap.schemas.hypothesis_io import load_hypotheses
from dap.utils.jsonl import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Detection-as-Prompt ablation variants.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--hypotheses", default="data/detector_predictions/hypotheses.jsonl")
    parser.add_argument("--raw-predictions", default="data/detector_predictions/raw_detector_predictions.jsonl")
    parser.add_argument("--routes", default="outputs/predictions/routes.jsonl")
    parser.add_argument("--ground-truth", default="data/annotations/test_decisions.jsonl")
    parser.add_argument("--output-dir", default="outputs/ablations")
    parser.add_argument("--csv", default="outputs/metrics/ablation_results.csv")
    parser.add_argument("--latex", default="outputs/tables/ablation_results.tex")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(PROJECT_ROOT / args.config_dir)
    graph = load_defect_kg(PROJECT_ROOT / config.kg["path"])
    hypotheses = load_hypotheses(PROJECT_ROOT / args.hypotheses)
    ground_truth = load_ground_truth(PROJECT_ROOT / args.ground_truth)
    route_lookup = load_route_lookup(PROJECT_ROOT / args.routes) if (PROJECT_ROOT / args.routes).exists() else {}
    image_lookup = _load_image_lookup(PROJECT_ROOT / args.raw_predictions)

    reports = run_ablation_suite(
        hypotheses=hypotheses,
        graph=graph,
        ground_truth=ground_truth,
        image_lookup=image_lookup,
        route_lookup=route_lookup,
        kg_retrieval_config=config.kg.get("retrieval", {}),
        vlm_config=config.vlm,
        output_dir=PROJECT_ROOT / args.output_dir,
    )
    write_ablation_csv(PROJECT_ROOT / args.csv, reports)
    write_ablation_latex(PROJECT_ROOT / args.latex, reports)

    print(f"Wrote ablation CSV to {PROJECT_ROOT / args.csv}")
    print(f"Wrote ablation LaTeX table to {PROJECT_ROOT / args.latex}")
    for variant_name, report in reports.items():
        summary = ", ".join(
            f"{metric}={'--' if value is None else f'{value:.3f}'}"
            for metric, value in report.metrics.items()
            if metric in {"HSA", "GPC", "HR"}
        )
        print(f"{variant_name}: {summary}")


def _load_image_lookup(raw_predictions_path: Path) -> dict[str, str | None]:
    if not raw_predictions_path.exists():
        return {}
    lookup: dict[str, str | None] = {}
    for row in read_jsonl(raw_predictions_path):
        lookup[str(row["image_id"])] = row.get("image_path")
    return lookup


if __name__ == "__main__":
    main()
