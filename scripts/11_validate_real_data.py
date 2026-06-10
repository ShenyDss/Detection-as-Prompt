from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dap.config import load_app_config
from dap.data_validation import validate_real_data
from dap.kg import load_defect_kg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate real data before Detection-as-Prompt experiments.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--hypotheses", default="data/detector_predictions/hypotheses.jsonl")
    parser.add_argument("--raw-predictions", default="data/detector_predictions/raw_detector_predictions.jsonl")
    parser.add_argument("--annotations", default="data/annotations/test_decisions.jsonl")
    parser.add_argument("--image-dir", default="data/images")
    parser.add_argument("--kg", default=None, help="Optional KG path overriding configs/kg.yaml.")
    parser.add_argument("--output", default="outputs/metrics/data_validation_report.json")
    parser.add_argument("--fail-on-warning", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(PROJECT_ROOT / args.config_dir)
    graph = load_defect_kg(PROJECT_ROOT / (args.kg or config.kg["path"]))
    report = validate_real_data(
        project_root=PROJECT_ROOT,
        graph=graph,
        hypotheses_path=PROJECT_ROOT / args.hypotheses,
        raw_predictions_path=PROJECT_ROOT / args.raw_predictions,
        annotations_path=PROJECT_ROOT / args.annotations,
        image_dir=PROJECT_ROOT / args.image_dir,
    )

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Validation status: {report.status}")
    print(f"Wrote report to {output_path}")
    for key, value in report.counts.items():
        print(f"  {key}: {value}")
    for issue in report.issues[:30]:
        print(f"[{issue.level}] {issue.code}: {issue.message}")
    if len(report.issues) > 30:
        print(f"... {len(report.issues) - 30} more issues in report")

    if report.counts.get("errors", 0) > 0 or (args.fail_on_warning and report.counts.get("warnings", 0) > 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
