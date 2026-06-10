from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dap.config import load_app_config
from dap.detector.runner import run_detector_to_hypotheses, write_raw_predictions
from dap.schemas.hypothesis_io import write_hypotheses_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detector and export Detection-as-Prompt hypotheses.")
    parser.add_argument("--config-dir", default="configs", help="Directory containing project YAML configs.")
    parser.add_argument("--mode", choices=["mock", "ultralytics", "yolo", "from_jsonl"], default=None)
    parser.add_argument("--checkpoint", default=None, help="YOLO checkpoint path for ultralytics mode.")
    parser.add_argument("--dataset-yaml", default=None, help="Optional YOLO dataset.yaml used to override class names.")
    parser.add_argument("--image-dir", default=None, help="Image directory. Defaults to dataset.image_dir.")
    parser.add_argument("--output", default=None, help="Hypotheses JSONL output path.")
    parser.add_argument(
        "--raw-output",
        default="data/detector_predictions/raw_detector_predictions.jsonl",
        help="Raw detector-like predictions JSONL output path.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional image limit for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(PROJECT_ROOT / args.config_dir)
    detector_config = dict(config.detector)
    dataset_config = dict(config.dataset)

    if args.mode:
        detector_config["mode"] = args.mode
    if args.checkpoint:
        detector_config["checkpoint"] = args.checkpoint
    if args.dataset_yaml:
        dataset_config["class_names"] = _load_yolo_class_names(PROJECT_ROOT / args.dataset_yaml)

    hypotheses, raw_predictions = run_detector_to_hypotheses(
        dataset_config=dataset_config,
        detector_config=detector_config,
        image_dir=PROJECT_ROOT / args.image_dir if args.image_dir else PROJECT_ROOT / dataset_config["image_dir"],
        limit=args.limit,
    )

    output_path = PROJECT_ROOT / (args.output or detector_config["output_path"])
    raw_output_path = PROJECT_ROOT / args.raw_output
    write_raw_predictions(raw_output_path, raw_predictions)
    write_hypotheses_jsonl(output_path, hypotheses)

    print(f"Detector mode: {detector_config.get('mode')}")
    print(f"Wrote {len(raw_predictions)} raw prediction rows to {raw_output_path}")
    print(f"Wrote {len(hypotheses)} hypotheses to {output_path}")


def _load_yolo_class_names(path: Path) -> list[str]:
    import yaml

    yaml_path = Path(path)
    with yaml_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    return [str(name) for name in names]


if __name__ == "__main__":
    main()
