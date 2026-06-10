from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dap.config import load_app_config
from dap.kg import load_defect_kg
from dap.prompt import build_detection_prompts
from dap.prompt.io import write_prompts_jsonl
from dap.routing.io import load_route_lookup
from dap.schemas.hypothesis_io import load_hypotheses
from dap.utils.jsonl import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Detection-as-Prompt structured prompts.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--hypotheses", default="data/detector_predictions/hypotheses.jsonl")
    parser.add_argument("--raw-predictions", default="data/detector_predictions/raw_detector_predictions.jsonl")
    parser.add_argument("--routes", default="outputs/predictions/routes.jsonl")
    parser.add_argument("--output", default="outputs/predictions/detection_prompts.jsonl")
    parser.add_argument("--region-output-dir", default="outputs/regions")
    parser.add_argument("--kg", default=None, help="Optional KG path overriding configs/kg.yaml.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(PROJECT_ROOT / args.config_dir)
    graph = load_defect_kg(PROJECT_ROOT / (args.kg or config.kg["path"]))
    hypotheses = load_hypotheses(PROJECT_ROOT / args.hypotheses)
    image_lookup = _load_image_lookup(PROJECT_ROOT / args.raw_predictions)
    route_lookup = load_route_lookup(PROJECT_ROOT / args.routes) if (PROJECT_ROOT / args.routes).exists() else {}

    prompts = build_detection_prompts(
        hypotheses,
        graph=graph,
        image_lookup=image_lookup,
        route_lookup=route_lookup,
        region_output_dir=PROJECT_ROOT / args.region_output_dir,
        kg_retrieval_config=config.kg.get("retrieval", {}),
    )
    output_path = PROJECT_ROOT / args.output
    write_prompts_jsonl(output_path, prompts)
    print(f"Wrote {len(prompts)} prompts to {output_path}")
    if prompts:
        print("First prompt preview:")
        print(prompts[0].prompt_text[:900])


def _load_image_lookup(raw_predictions_path: Path) -> dict[str, str | None]:
    if not raw_predictions_path.exists():
        return {}
    lookup: dict[str, str | None] = {}
    for row in read_jsonl(raw_predictions_path):
        image_id = str(row["image_id"])
        image_path = row.get("image_path")
        if image_path:
            candidate = Path(image_path)
            lookup[image_id] = str(candidate)
        else:
            lookup[image_id] = None
    return lookup


if __name__ == "__main__":
    main()
