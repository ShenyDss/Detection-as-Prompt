from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dap.config import load_app_config
from dap.routing import route_hypotheses
from dap.routing.io import write_routes_jsonl
from dap.schemas.hypothesis_io import load_hypotheses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route detector hypotheses by uncertainty signals.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--hypotheses", default="data/detector_predictions/hypotheses.jsonl")
    parser.add_argument("--output", default="outputs/predictions/routes.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(PROJECT_ROOT / args.config_dir)
    hypotheses = load_hypotheses(PROJECT_ROOT / args.hypotheses)
    routes = route_hypotheses(hypotheses, route_config=config.experiment.get("routes", {}))
    output_path = PROJECT_ROOT / args.output
    write_routes_jsonl(output_path, routes)

    counts = Counter(route.review_mode.value for route in routes)
    print(f"Wrote {len(routes)} route decisions to {output_path}")
    print("Route summary:")
    for mode, count in sorted(counts.items()):
        print(f"  {mode}: {count}")


if __name__ == "__main__":
    main()
