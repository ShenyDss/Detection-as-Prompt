from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dap.schemas.hypothesis_io import load_hypotheses
from dap.utils.jsonl import read_jsonl
from dap.visualization import render_case_visualizations
from dap.vlm.output_io import load_decisions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render case visualizations with boxes and decision status.")
    parser.add_argument("--hypotheses", default="data/detector_predictions/hypotheses.jsonl")
    parser.add_argument("--decisions", default="outputs/predictions/vlm_decisions.jsonl")
    parser.add_argument("--raw-predictions", default="data/detector_predictions/raw_detector_predictions.jsonl")
    parser.add_argument("--output-dir", default="outputs/visualizations/cases")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hypotheses = load_hypotheses(PROJECT_ROOT / args.hypotheses)
    decisions = load_decisions(PROJECT_ROOT / args.decisions)
    image_lookup = {
        str(row["image_id"]): str(PROJECT_ROOT / row["image_path"]) if row.get("image_path") else None
        for row in read_jsonl(PROJECT_ROOT / args.raw_predictions)
    }
    count = render_case_visualizations(
        hypotheses=hypotheses,
        decisions=decisions,
        image_lookup=image_lookup,
        output_dir=PROJECT_ROOT / args.output_dir,
    )
    print(f"Wrote {count} visualizations to {PROJECT_ROOT / args.output_dir}")


if __name__ == "__main__":
    main()
