from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dap.config import load_app_config
from dap.detector import convert_mock_predictions
from dap.schemas.hypothesis_io import write_hypotheses_jsonl
from dap.utils.jsonl import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert mock detector predictions into Detection-as-Prompt hypotheses."
    )
    parser.add_argument(
        "--input",
        default="data/detector_predictions/mock_detector_predictions.jsonl",
        help="Input mock detector predictions JSONL.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output hypotheses JSONL. Defaults to detector.output_path in configs/detector.yaml.",
    )
    parser.add_argument("--config-dir", default="configs", help="Directory containing project YAML configs.")
    parser.add_argument(
        "--write-demo-input",
        action="store_true",
        help="Write a small demo mock prediction file before conversion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(PROJECT_ROOT / args.config_dir)
    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / (args.output or config.detector["output_path"])

    if args.write_demo_input:
        write_jsonl(input_path, _demo_predictions())

    if not input_path.exists():
        raise FileNotFoundError(
            f"Mock prediction file not found: {input_path}. "
            "Run with --write-demo-input to create a small example."
        )

    hypotheses = convert_mock_predictions(
        read_jsonl(input_path),
        top_k_confusions=int(config.detector.get("top_k_confusions", 3)),
        score_threshold=float(config.detector.get("confidence_threshold", 0.0)),
    )
    write_hypotheses_jsonl(output_path, hypotheses)
    print(f"Wrote {len(hypotheses)} hypotheses to {output_path}")


def _demo_predictions() -> list[dict]:
    return [
        {
            "image_id": "fabric_0001",
            "source": "mock_detector",
            "detections": [
                {
                    "pred_class": "broken_weft",
                    "bbox": [118, 204, 286, 330],
                    "score": 0.86,
                    "class_scores": {
                        "broken_weft": 0.86,
                        "loose_weft": 0.09,
                        "coarse_weft": 0.04,
                        "yarn_irregularity": 0.01,
                    },
                },
                {
                    "pred_class": "coarse_weft",
                    "bbox": [620, 410, 780, 560],
                    "score": 0.41,
                    "class_scores": {
                        "coarse_weft": 0.41,
                        "loose_weft": 0.34,
                        "broken_weft": 0.14,
                        "yarn_irregularity": 0.11,
                    },
                },
            ],
        },
        {
            "image_id": "fabric_0002",
            "source": "mock_detector",
            "detections": [
                {
                    "pred_class": "loose_weft",
                    "bbox": [92, 144, 240, 260],
                    "score": 0.31,
                    "class_scores": {
                        "loose_weft": 0.31,
                        "yarn_irregularity": 0.28,
                        "coarse_weft": 0.24,
                        "broken_weft": 0.17,
                    },
                }
            ],
        },
    ]


if __name__ == "__main__":
    main()
