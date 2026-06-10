from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class YoloObject:
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]


@dataclass
class ImageRecord:
    dataset: str
    image_id: str
    image_path: Path
    json_path: Path | None
    width: int
    height: int
    objects: list[YoloObject]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Labelme datasets to YOLO train/val format.")
    parser.add_argument("--class-map", default="configs/class_map.sm_nzb.json")
    parser.add_argument("--output-dir", default="data/yolo/sm_nzb")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-unlabeled-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_map = _load_class_map(PROJECT_ROOT / args.class_map)
    sources = [
        {
            "name": "sm_norm",
            "json_dir": PROJECT_ROOT / "SM_norm/norm/images",
            "image_dir": PROJECT_ROOT / "SM_norm/norm/images",
        },
        {
            "name": "nzb_dataset",
            "json_dir": PROJECT_ROOT / "NZB_dataset/json",
            "image_dir": PROJECT_ROOT / "NZB_dataset/images",
        },
    ]
    records, warnings = collect_labelme_records(
        sources=sources,
        class_map=class_map,
        include_unlabeled_images=args.include_unlabeled_images,
    )
    output_dir = PROJECT_ROOT / args.output_dir
    result = write_yolo_dataset(
        records=records,
        output_dir=output_dir,
        class_names=sorted(set(class_map.values())),
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    result["warnings"] = warnings[:200]
    (output_dir / "conversion_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def collect_labelme_records(
    *,
    sources: list[dict[str, Path | str]],
    class_map: dict[str, str],
    include_unlabeled_images: bool,
) -> tuple[list[ImageRecord], list[str]]:
    records: list[ImageRecord] = []
    warnings: list[str] = []
    seen_output_ids: set[str] = set()
    for source in sources:
        dataset = str(source["name"])
        json_dir = Path(source["json_dir"])
        image_dir = Path(source["image_dir"])
        json_files = sorted(json_dir.glob("*.json"))
        json_image_names: set[str] = set()

        for json_path in json_files:
            data = _load_json(json_path)
            image_path = _resolve_image_path(json_path, data, image_dir=image_dir)
            if image_path is None:
                warnings.append(f"{dataset}/{json_path.name}: matching image not found")
                continue
            image_id = _unique_id(f"{dataset}_{json_path.stem}", seen_output_ids)
            width, height = _image_size(image_path, data)
            objects = _objects_from_labelme(data, class_map, width=width, height=height, warnings=warnings, item_id=f"{dataset}/{json_path.name}")
            records.append(
                ImageRecord(
                    dataset=dataset,
                    image_id=image_id,
                    image_path=image_path,
                    json_path=json_path,
                    width=width,
                    height=height,
                    objects=objects,
                )
            )
            json_image_names.add(image_path.name.lower())

        if include_unlabeled_images:
            for image_path in sorted(image_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                if image_path.name.lower() in json_image_names:
                    continue
                image_id = _unique_id(f"{dataset}_{image_path.stem}", seen_output_ids)
                width, height = _image_size(image_path, {})
                records.append(
                    ImageRecord(
                        dataset=dataset,
                        image_id=image_id,
                        image_path=image_path,
                        json_path=None,
                        width=width,
                        height=height,
                        objects=[],
                    )
                )
    return records, warnings


def write_yolo_dataset(
    *,
    records: list[ImageRecord],
    output_dir: Path,
    class_names: list[str],
    val_ratio: float,
    seed: int,
) -> dict[str, Any]:
    class_to_id = {name: index for index, name in enumerate(class_names)}
    split_lookup = split_records(records, val_ratio=val_ratio, seed=seed)
    for split in ("train", "val"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    label_counts = Counter()
    split_counts = Counter()
    empty_counts = Counter()
    for record in records:
        split = split_lookup[record.image_id]
        split_counts[split] += 1
        if not record.objects:
            empty_counts[split] += 1
        image_dst = output_dir / split / "images" / f"{record.image_id}{record.image_path.suffix.lower()}"
        label_dst = output_dir / split / "labels" / f"{record.image_id}.txt"
        shutil.copy2(record.image_path, image_dst)
        lines = []
        for obj in record.objects:
            label_counts[obj.class_name] += 1
            lines.append(_yolo_line(class_to_id[obj.class_name], obj.bbox_xyxy, record.width, record.height))
        label_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    dataset_yaml = output_dir / "dataset.yaml"
    names_block = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve().as_posix()}",
                "train: train/images",
                "val: val/images",
                "test: val/images",
                "names:",
                names_block,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "status": "ready" if records else "empty",
        "output_dir": str(output_dir.resolve()),
        "dataset_yaml": str(dataset_yaml.resolve()),
        "classes": class_names,
        "images": len(records),
        "objects": sum(len(record.objects) for record in records),
        "split_images": dict(split_counts),
        "empty_images": dict(empty_counts),
        "label_counts": dict(label_counts),
        "val_ratio": val_ratio,
        "seed": seed,
        "note": "YOLO test split intentionally points to val/images as requested.",
    }


def split_records(records: list[ImageRecord], *, val_ratio: float, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        key = record.objects[0].class_name if record.objects else "__empty__"
        grouped[key].append(record)

    split_lookup: dict[str, str] = {}
    for key, group in grouped.items():
        group = list(group)
        rng.shuffle(group)
        if key == "__empty__":
            n_val = round(len(group) * val_ratio)
        elif len(group) < 5:
            n_val = 0
        else:
            n_val = max(1, round(len(group) * val_ratio))
        val_ids = {record.image_id for record in group[:n_val]}
        for record in group:
            split_lookup[record.image_id] = "val" if record.image_id in val_ids else "train"
    return split_lookup


def _objects_from_labelme(
    data: dict[str, Any],
    class_map: dict[str, str],
    *,
    width: int,
    height: int,
    warnings: list[str],
    item_id: str,
) -> list[YoloObject]:
    objects: list[YoloObject] = []
    shapes = data.get("shapes", [])
    if not isinstance(shapes, list):
        warnings.append(f"{item_id}: shapes is not a list")
        return objects
    for index, shape in enumerate(shapes):
        raw_label = str(shape.get("label", "")).strip()
        if raw_label not in class_map:
            warnings.append(f"{item_id}:{index}: unknown label {raw_label!r}, skipped")
            continue
        bbox = _bbox_from_shape(shape, width=width, height=height)
        if bbox is None:
            warnings.append(f"{item_id}:{index}: invalid bbox, skipped")
            continue
        objects.append(YoloObject(class_name=class_map[raw_label], bbox_xyxy=bbox))
    return objects


def _bbox_from_shape(
    shape: dict[str, Any],
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    points = shape.get("points")
    if not isinstance(points, list) or len(points) < 2:
        return None
    xs = [float(point[0]) for point in points if isinstance(point, list) and len(point) >= 2]
    ys = [float(point[1]) for point in points if isinstance(point, list) and len(point) >= 2]
    if not xs or not ys:
        return None
    x1 = _clamp(min(xs), 0.0, float(width))
    x2 = _clamp(max(xs), 0.0, float(width))
    y1 = _clamp(min(ys), 0.0, float(height))
    y2 = _clamp(max(ys), 0.0, float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _yolo_line(class_id: int, bbox: tuple[float, float, float, float], width: int, height: int) -> str:
    x1, y1, x2, y2 = bbox
    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"{class_id} {x_center:.8f} {y_center:.8f} {box_width:.8f} {box_height:.8f}"


def _resolve_image_path(json_path: Path, data: dict[str, Any], *, image_dir: Path) -> Path | None:
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


def _image_size(path: Path, data: dict[str, Any]) -> tuple[int, int]:
    width = data.get("imageWidth")
    height = data.get("imageHeight")
    if width and height:
        return int(width), int(height)
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Pillow is required to read image sizes for YOLO conversion.") from exc
    with Image.open(path) as image:
        return image.size


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return data


def _load_class_map(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Class map must be an object: {path}")
    return {str(key): str(value) for key, value in data.items()}


def _unique_id(base: str, seen: set[str]) -> str:
    candidate = base
    index = 1
    while candidate in seen:
        index += 1
        candidate = f"{base}__dup{index}"
    seen.add(candidate)
    return candidate


def _clamp(value: float, lower: float, upper: float) -> float:
    if -1e-6 < value < lower:
        value = lower
    return min(max(value, lower), upper)


if __name__ == "__main__":
    main()
