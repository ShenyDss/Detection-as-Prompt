from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a temporary KG from a class-map JSON.")
    parser.add_argument("--class-map", default="configs/class_map.sm_nzb.json")
    parser.add_argument("--output", default="data/imported/sm_nzb_kg.generated.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_map_path = PROJECT_ROOT / args.class_map
    output_path = PROJECT_ROOT / args.output
    class_map = json.loads(class_map_path.read_text(encoding="utf-8"))
    classes = sorted(set(str(value) for value in class_map.values()))
    kg = {
        "name": "sm_nzb_labelme_generated_kg",
        "metadata": {
            "version": "generated_from_class_map",
            "description": "Temporary KG generated from observed Labelme labels. Replace with expert-curated knowledge before formal experiments.",
        },
        "classes": [
            {
                "name": class_name,
                "visual_attributes": [
                    f"visual pattern annotated as {class_name}",
                    "localized abnormal texture region",
                    "requires expert confirmation because imported labels may be incomplete",
                ],
                "confusion_classes": [other for other in classes if other != class_name][:5],
                "causes": [f"unknown production cause for {class_name}"],
                "actions": [f"inspect and confirm {class_name}"],
                "risk_notes": ["temporary generated KG entry for pipeline testing"],
            }
            for class_name in classes
        ],
        "cause_action_edges": {
            f"unknown production cause for {class_name}": [f"inspect and confirm {class_name}"]
            for class_name in classes
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(kg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote temporary KG with {len(classes)} classes to {output_path}")


if __name__ == "__main__":
    main()
