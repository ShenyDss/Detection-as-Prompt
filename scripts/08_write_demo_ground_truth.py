from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dap.vlm.output_io import load_decisions
from dap.utils.jsonl import write_jsonl


def main() -> None:
    decisions = load_decisions(PROJECT_ROOT / "outputs/predictions/vlm_decisions.jsonl")
    rows = []
    for decision in decisions:
        rows.append(
            {
                "image_id": decision.image_id,
                "hypothesis_id": decision.hypothesis_id,
                "hypothesis_status": decision.hypothesis_status.value,
                "corrected_class": decision.corrected_class,
                "bbox": decision.refined_bbox.to_list() if decision.refined_bbox else None,
                "causes": decision.decision.causes,
                "actions": decision.decision.actions,
            }
        )
    output_path = PROJECT_ROOT / "data/annotations/test_decisions.jsonl"
    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} demo ground-truth rows to {output_path}")


if __name__ == "__main__":
    main()
