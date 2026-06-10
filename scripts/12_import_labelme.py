from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dap.schemas.core import BBox, DefectHypothesis
from dap.schemas.hypothesis_io import write_hypotheses_jsonl
from dap.utils.jsonl import write_jsonl

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import Labelme image+json data for Detection-as-Prompt tests.")
    parser.add_argument("--labelme-dir", default="SM_norm/norm/images")
    parser.add_argument("--json-dir", default=None, help="Optional separate directory containing Labelme JSON files.")
    parser.add_argument("--image-dir", default=None, help="Optional separate directory containing source images.")
    parser.add_argument("--output-dir", default="data/imported/sm_norm")
    parser.add_argument("--class-map", default=None, help="Optional JSON mapping from Labelme labels to canonical classes.")
    parser.add_argument("--score", type=float, default=0.95, help="Pseudo-detector confidence for Labelme boxes.")
    parser.add_argument("--include-empty", action="store_true", help="Record empty Labelme files in manifest as normal images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labelme_dir = (PROJECT_ROOT / args.labelme_dir).resolve()
    json_dir = (PROJECT_ROOT / args.json_dir).resolve() if args.json_dir else labelme_dir
    image_dir = (PROJECT_ROOT / args.image_dir).resolve() if args.image_dir else labelme_dir
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    class_map = _load_class_map(PROJECT_ROOT / args.class_map) if args.class_map else {}

    import_result = import_labelme_dir(
        labelme_dir=json_dir,
        image_dir=image_dir,
        output_dir=output_dir,
        class_map=class_map,
        score=args.score,
        include_empty=args.include_empty,
    )
    print(json.dumps(import_result, ensure_ascii=False, indent=2))


def import_labelme_dir(
    *,
    labelme_dir: Path,
    image_dir: Path | None = None,
    output_dir: Path,
    class_map: dict[str, str],
    score: float,
    include_empty: bool,
) -> dict[str, Any]:
    if not labelme_dir.exists():
        raise FileNotFoundError(f"Labelme directory not found: {labelme_dir}")
    image_dir = image_dir or labelme_dir
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    hypotheses: list[DefectHypothesis] = []
    raw_rows: list[dict[str, Any]] = []
    annotation_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    labels = Counter()
    canonical_labels = Counter()
    shape_types = Counter()
    warnings: list[str] = []

    for json_path in sorted(labelme_dir.glob("*.json")):
        data = _load_json(json_path)
        image_path = _resolve_image_path(json_path, data, image_dir=image_dir)
        image_id = json_path.stem
        shapes = data.get("shapes", [])
        if not isinstance(shapes, list):
            warnings.append(f"{json_path.name}: shapes is not a list")
            shapes = []

        if not shapes and include_empty:
            manifest_rows.append(
                {
                    "image_id": image_id,
                    "image_path": _relative_or_absolute(image_path),
                    "labelme_json": _relative_or_absolute(json_path),
                    "is_empty": True,
                    "note": "No Labelme shapes. Treat as normal/empty image for detector testing.",
                }
            )

        image_predictions: list[dict[str, Any]] = []
        for shape_index, shape in enumerate(shapes):
            label = str(shape.get("label", "unknown")).strip() or "unknown"
            canonical = class_map.get(label, label)
            labels[label] += 1
            canonical_labels[canonical] += 1
            shape_types[str(shape.get("shape_type", "unknown"))] += 1

            bbox = _bbox_from_shape(shape)
            if bbox is None:
                warnings.append(f"{json_path.name}:{shape_index}: unsupported or invalid shape")
                continue
            hypothesis_id = f"{image_id}_{shape_index:03d}"
            confusion = _confusion_candidates(canonical, class_map, canonical_labels)
            hypothesis = DefectHypothesis(
                image_id=image_id,
                hypothesis_id=hypothesis_id,
                pred_class=canonical,
                bbox=bbox,
                score=score,
                confusion_candidates=confusion,
                detector_meta={
                    "source": "labelme_pseudo_detector",
                    "labelme_json": _relative_or_absolute(json_path),
                    "raw_label": label,
                    "shape_index": shape_index,
                    "shape_type": shape.get("shape_type"),
                    "note": "Imported Labelme box used as a pseudo detector hypothesis for pipeline testing.",
                },
            )
            hypotheses.append(hypothesis)
            image_predictions.append(
                {
                    "class": canonical,
                    "bbox": bbox.to_list(),
                    "score": score,
                    "raw_label": label,
                }
            )
            annotation_rows.append(
                {
                    "image_id": image_id,
                    "hypothesis_id": hypothesis_id,
                    "hypothesis_status": "verify",
                    "corrected_class": canonical,
                    "bbox": bbox.to_list(),
                    "causes": [],
                    "actions": [],
                }
            )

        if image_predictions or include_empty:
            raw_rows.append(
                {
                    "image_id": image_id,
                    "image_path": _relative_or_absolute(image_path),
                    "predictions": image_predictions,
                    "labelme_json": _relative_or_absolute(json_path),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_hypotheses_jsonl(output_dir / "hypotheses.jsonl", hypotheses)
    write_jsonl(output_dir / "raw_detector_predictions.jsonl", raw_rows)
    write_jsonl(output_dir / "test_decisions.jsonl", annotation_rows)
    if manifest_rows:
        write_jsonl(output_dir / "normal_manifest.jsonl", manifest_rows)
    _write_generated_kg(output_dir / "defect_kg.generated.json", canonical_labels)

    report = {
        "status": "ready" if hypotheses else "empty",
        "labelme_dir": str(labelme_dir),
        "image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "json_files": len(list(labelme_dir.glob("*.json"))),
        "hypotheses": len(hypotheses),
        "raw_prediction_rows": len(raw_rows),
        "annotation_rows": len(annotation_rows),
        "raw_labels": dict(labels),
        "canonical_labels": dict(canonical_labels),
        "shape_types": dict(shape_types),
        "warnings": warnings[:100],
        "notes": [
            "Labelme boxes are imported as pseudo detector hypotheses for pipeline smoke tests.",
            "Empty Labelme files are normal images, not reject hypotheses, unless a detector later produces false alarms on them.",
            "If labels are incomplete, update the generated KG or provide a class-map JSON before formal experiments.",
        ],
    }
    (output_dir / "import_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Labelme JSON must be an object: {path}")
    return data


def _load_class_map(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Class map must be a JSON object: {path}")
    return {str(key): str(value) for key, value in data.items()}


def _resolve_image_path(json_path: Path, data: dict[str, Any], *, image_dir: Path | None = None) -> Path | None:
    image_dir = image_dir or json_path.parent
    candidates: list[Path] = []
    image_path = data.get("imagePath")
    if image_path:
        candidates.append(image_dir / str(image_path))
        candidates.append(json_path.parent / str(image_path))
    for suffix in IMAGE_SUFFIXES:
        candidates.append(image_dir / f"{json_path.stem}{suffix}")
        candidates.append(json_path.with_suffix(suffix))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _bbox_from_shape(shape: dict[str, Any]) -> BBox | None:
    points = shape.get("points")
    if not isinstance(points, list) or len(points) < 2:
        return None
    xs = [float(point[0]) for point in points if isinstance(point, list) and len(point) >= 2]
    ys = [float(point[1]) for point in points if isinstance(point, list) and len(point) >= 2]
    if not xs or not ys:
        return None
    x1, x2 = _clean_coord(min(xs)), _clean_coord(max(xs))
    y1, y2 = _clean_coord(min(ys)), _clean_coord(max(ys))
    if x2 <= x1 or y2 <= y1:
        return None
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _clean_coord(value: float) -> float:
    if -1e-6 < value < 0:
        return 0.0
    return value


def _confusion_candidates(
    current_class: str,
    class_map: dict[str, str],
    observed_labels: Counter[str],
) -> list[str]:
    candidates = sorted({*class_map.values(), *observed_labels.keys()} - {current_class})
    return candidates[:3]


def _write_generated_kg(path: Path, class_counts: Counter[str]) -> None:
    classes = []
    for class_name in sorted(class_counts):
        classes.append(
            {
                "name": class_name,
                "visual_attributes": [
                    f"visual pattern annotated as {class_name}",
                    "localized abnormal texture region",
                    "requires expert confirmation because imported labels may be incomplete",
                ],
                "confusion_classes": sorted(set(class_counts) - {class_name}),
                "causes": [f"unknown production cause for {class_name}"],
                "actions": [f"inspect and confirm {class_name}"],
                "risk_notes": ["temporary generated KG entry for pipeline testing"],
            }
        )
    cause_action_edges = {
        f"unknown production cause for {class_name}": [f"inspect and confirm {class_name}"]
        for class_name in class_counts
    }
    kg = {
        "name": "sm_norm_labelme_generated_kg",
        "metadata": {
            "version": "generated_from_labelme",
            "description": "Temporary KG generated from observed Labelme labels. Replace or edit before paper-grade experiments.",
        },
        "classes": classes,
        "cause_action_edges": cause_action_edges,
    }
    path.write_text(json.dumps(kg, ensure_ascii=False, indent=2), encoding="utf-8")


def _relative_or_absolute(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    main()
