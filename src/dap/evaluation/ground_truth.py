from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dap.schemas.core import BBox, HypothesisStatus
from dap.utils.jsonl import read_jsonl


@dataclass
class GroundTruthDecision:
    image_id: str
    hypothesis_id: str
    status: HypothesisStatus
    corrected_class: str | None
    bbox: BBox | None
    causes: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)


def load_ground_truth(path: str | Path) -> dict[str, GroundTruthDecision]:
    gt_path = Path(path)
    if not gt_path.exists():
        return {}

    items: dict[str, GroundTruthDecision] = {}
    for row in read_jsonl(gt_path):
        item = ground_truth_from_dict(row)
        items[item.hypothesis_id] = item
    return items


def ground_truth_from_dict(data: dict[str, Any]) -> GroundTruthDecision:
    bbox = data.get("bbox", data.get("refined_bbox"))
    return GroundTruthDecision(
        image_id=str(data["image_id"]),
        hypothesis_id=str(data["hypothesis_id"]),
        status=HypothesisStatus(str(data.get("hypothesis_status", data.get("status", "verify")))),
        corrected_class=data.get("corrected_class"),
        bbox=BBox.from_list(bbox) if isinstance(bbox, list) else None,
        causes=[str(item) for item in data.get("causes", [])],
        actions=[str(item) for item in data.get("actions", [])],
    )
