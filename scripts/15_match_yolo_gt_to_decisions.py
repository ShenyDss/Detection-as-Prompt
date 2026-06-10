from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dap.evaluation.box_ops import bbox_iou
from dap.schemas.core import BBox, DefectHypothesis
from dap.schemas.hypothesis_io import load_hypotheses
from dap.utils.jsonl import write_jsonl


@dataclass
class GroundTruthBox:
    class_name: str
    bbox: BBox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match YOLO predictions to YOLO-format ground truth and create verify/reject/revise labels.")
    parser.add_argument("--hypotheses", default="outputs/imported/yolo26n_val/hypotheses.jsonl")
    parser.add_argument("--dataset-yaml", default="data/yolo/sm_nzb/dataset.yaml")
    parser.add_argument("--label-dir", default="data/yolo/sm_nzb/val/labels")
    parser.add_argument("--image-dir", default="data/yolo/sm_nzb/val/images")
    parser.add_argument("--output", default="outputs/imported/yolo26n_val/test_decisions.jsonl")
    parser.add_argument("--report", default="outputs/imported/yolo26n_val/match_report.json")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_names = _load_yolo_class_names(PROJECT_ROOT / args.dataset_yaml)
    image_sizes = _load_image_sizes(PROJECT_ROOT / args.image_dir)
    gt_lookup = _load_gt_lookup(
        label_dir=PROJECT_ROOT / args.label_dir,
        image_sizes=image_sizes,
        class_names=class_names,
    )
    hypotheses = load_hypotheses(PROJECT_ROOT / args.hypotheses)
    rows, report = match_hypotheses_to_ground_truth(
        hypotheses=hypotheses,
        gt_lookup=gt_lookup,
        iou_threshold=args.iou_threshold,
    )
    output_path = PROJECT_ROOT / args.output
    report_path = PROJECT_ROOT / args.report
    write_jsonl(output_path, rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote matched decisions to {output_path}")


def match_hypotheses_to_ground_truth(
    *,
    hypotheses: list[DefectHypothesis],
    gt_lookup: dict[str, list[GroundTruthBox]],
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    status_counts = Counter()
    class_counts = Counter()
    best_ious: list[float] = []

    for hypothesis in hypotheses:
        gt_boxes = gt_lookup.get(hypothesis.image_id, [])
        best_gt, best_iou = _best_match(hypothesis.bbox, gt_boxes)
        best_ious.append(best_iou)
        if best_gt is None or best_iou < iou_threshold:
            status = "reject"
            corrected_class = None
            bbox = None
            causes: list[str] = []
            actions: list[str] = []
        elif best_gt.class_name == hypothesis.pred_class:
            status = "verify"
            corrected_class = hypothesis.pred_class
            bbox = best_gt.bbox.to_list()
            causes = [f"unknown production cause for {corrected_class}"]
            actions = [f"inspect and confirm {corrected_class}"]
        else:
            status = "revise"
            corrected_class = best_gt.class_name
            bbox = best_gt.bbox.to_list()
            causes = [f"unknown production cause for {corrected_class}"]
            actions = [f"inspect and confirm {corrected_class}"]

        status_counts[status] += 1
        class_counts[corrected_class or "__reject__"] += 1
        rows.append(
            {
                "image_id": hypothesis.image_id,
                "hypothesis_id": hypothesis.hypothesis_id,
                "hypothesis_status": status,
                "corrected_class": corrected_class,
                "bbox": bbox,
                "causes": causes,
                "actions": actions,
                "match_iou": best_iou,
                "pred_class": hypothesis.pred_class,
                "note": "Automatically matched from YOLO prediction and val YOLO ground truth.",
            }
        )

    unmatched_gt = _count_unmatched_gt(hypotheses, gt_lookup, iou_threshold)
    report = {
        "status": "ready",
        "hypotheses": len(hypotheses),
        "ground_truth_images": len(gt_lookup),
        "ground_truth_boxes": sum(len(items) for items in gt_lookup.values()),
        "status_counts": dict(status_counts),
        "corrected_class_counts": dict(class_counts),
        "mean_best_iou": sum(best_ious) / len(best_ious) if best_ious else None,
        "iou_threshold": iou_threshold,
        "unmatched_ground_truth_boxes": unmatched_gt,
        "notes": [
            "verify: IoU passes threshold and predicted class matches ground truth.",
            "revise: IoU passes threshold but predicted class differs from ground truth.",
            "reject: no ground-truth box reaches the IoU threshold.",
            "Unmatched ground-truth boxes are detector misses and are not hypothesis-level decisions unless a proposal exists.",
        ],
    }
    return rows, report


def _best_match(pred_box: BBox, gt_boxes: list[GroundTruthBox]) -> tuple[GroundTruthBox | None, float]:
    best_gt: GroundTruthBox | None = None
    best_iou = 0.0
    for gt in gt_boxes:
        score = bbox_iou(pred_box, gt.bbox)
        if score > best_iou:
            best_iou = score
            best_gt = gt
    return best_gt, best_iou


def _count_unmatched_gt(
    hypotheses: list[DefectHypothesis],
    gt_lookup: dict[str, list[GroundTruthBox]],
    iou_threshold: float,
) -> int:
    hypotheses_by_image: dict[str, list[DefectHypothesis]] = {}
    for hypothesis in hypotheses:
        hypotheses_by_image.setdefault(hypothesis.image_id, []).append(hypothesis)

    missed = 0
    for image_id, gt_boxes in gt_lookup.items():
        image_hypotheses = hypotheses_by_image.get(image_id, [])
        for gt in gt_boxes:
            if not any(bbox_iou(hypothesis.bbox, gt.bbox) >= iou_threshold for hypothesis in image_hypotheses):
                missed += 1
    return missed


def _load_gt_lookup(
    *,
    label_dir: Path,
    image_sizes: dict[str, tuple[int, int]],
    class_names: list[str],
) -> dict[str, list[GroundTruthBox]]:
    lookup: dict[str, list[GroundTruthBox]] = {}
    for label_path in sorted(label_dir.glob("*.txt")):
        image_id = label_path.stem
        if image_id not in image_sizes:
            continue
        width, height = image_sizes[image_id]
        boxes = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cls_id_text, x_text, y_text, w_text, h_text = line.split()
            cls_id = int(cls_id_text)
            x_center = float(x_text) * width
            y_center = float(y_text) * height
            box_width = float(w_text) * width
            box_height = float(h_text) * height
            bbox = BBox(
                x1=_clamp(x_center - box_width / 2, 0.0, float(width)),
                y1=_clamp(y_center - box_height / 2, 0.0, float(height)),
                x2=_clamp(x_center + box_width / 2, 0.0, float(width)),
                y2=_clamp(y_center + box_height / 2, 0.0, float(height)),
            )
            boxes.append(GroundTruthBox(class_name=class_names[cls_id], bbox=bbox))
        lookup[image_id] = boxes
    return lookup


def _load_image_sizes(image_dir: Path) -> dict[str, tuple[int, int]]:
    from PIL import Image

    sizes: dict[str, tuple[int, int]] = {}
    for image_path in image_dir.glob("*"):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue
        with Image.open(image_path) as image:
            sizes[image_path.stem] = image.size
    return sizes


def _load_yolo_class_names(path: Path) -> list[str]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    return [str(name) for name in names]


def _clamp(value: float, lower: float, upper: float) -> float:
    if -1e-5 < value < lower:
        value = lower
    return min(max(value, lower), upper)


if __name__ == "__main__":
    main()
