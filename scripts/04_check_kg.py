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
from dap.kg import check_graph_consistency, load_defect_kg, retrieve_category_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect defect KG context and consistency.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--class-name", default="broken_weft")
    parser.add_argument("--cause", default="yarn breakage")
    parser.add_argument("--action", default="stop and repair broken yarn")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(PROJECT_ROOT / args.config_dir)
    kg_path = PROJECT_ROOT / config.kg["path"]
    graph = load_defect_kg(kg_path)

    context = retrieve_category_context(
        graph,
        args.class_name,
        max_visual_attributes=int(config.kg["retrieval"]["max_visual_attributes"]),
        max_confusion_classes=int(config.kg["retrieval"]["max_confusion_classes"]),
        max_causes=int(config.kg["retrieval"]["max_causes"]),
        max_actions=int(config.kg["retrieval"]["max_actions"]),
    )
    result = check_graph_consistency(
        graph,
        corrected_class=args.class_name,
        causes=[args.cause],
        actions=[args.action],
        require_valid_class=bool(config.kg["consistency"]["require_valid_class"]),
        require_valid_cause_action_path=bool(config.kg["consistency"]["require_valid_cause_action_path"]),
    )

    print("KG context:")
    print(json.dumps(context.to_dict(), ensure_ascii=False, indent=2))
    print("\nConsistency:")
    print(json.dumps(result.__dict__ | {"score": result.score()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
