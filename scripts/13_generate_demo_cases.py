from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate small demo images for pipeline visualization.")
    parser.add_argument("--output-dir", default="data/images")
    parser.add_argument("--count", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#d9480f", "#2f9e44", "#1971c2", "#9c36b5"]
    for index in range(args.count):
        image = Image.new("RGB", (768, 768), "#f3f0e8")
        draw = ImageDraw.Draw(image)
        for y in range(0, 768, 18):
            draw.line((0, y, 768, y), fill="#d6d0c4", width=1)
        for x in range(0, 768, 28):
            draw.line((x, 0, x, 768), fill="#e7e1d7", width=1)
        x1, y1 = 96 + index * 12, 128
        x2, y2 = 260 + index * 12, 288
        draw.rectangle((x1, y1, x2, y2), outline=colors[index % len(colors)], width=5)
        draw.line((x1, (y1 + y2) // 2, x2, (y1 + y2) // 2), fill=colors[index % len(colors)], width=4)
        image.save(output_dir / f"mock_image_{index:04d}.png")
    print(f"Wrote {args.count} demo images to {output_dir}")


if __name__ == "__main__":
    main()
