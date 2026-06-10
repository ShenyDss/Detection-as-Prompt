from __future__ import annotations

from pathlib import Path

from dap.schemas.core import BBox


def crop_hypothesis_region(
    *,
    image_path: str | Path | None,
    bbox: BBox,
    output_dir: str | Path,
    output_name: str,
    padding: int = 8,
) -> Path | None:
    """Crop a candidate region when the source image exists.

    Returns None for mock hypotheses or missing files, allowing the prompt
    pipeline to run before real image assets are available.
    """

    if image_path is None:
        return None

    source = Path(image_path)
    if not source.exists():
        return None

    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Pillow is required for region cropping.") from exc

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / f"{output_name}.png"

    with Image.open(source) as image:
        width, height = image.size
        x1 = max(0, int(round(bbox.x1)) - padding)
        y1 = max(0, int(round(bbox.y1)) - padding)
        x2 = min(width, int(round(bbox.x2)) + padding)
        y2 = min(height, int(round(bbox.y2)) + padding)
        if x2 <= x1 or y2 <= y1:
            return None
        image.crop((x1, y1, x2, y2)).save(destination)

    return destination
