from __future__ import annotations

import argparse
import csv
from pathlib import Path
from PIL import Image, ImageDraw


COLORS = [
    ("red", (220, 60, 60)),
    ("green", (70, 160, 90)),
    ("blue", (60, 100, 220)),
    ("yellow", (220, 200, 60)),
    ("purple", (160, 80, 180)),
]

SHAPES = ["square", "circle", "triangle", "diamond"]


def draw_shape(draw: ImageDraw.ImageDraw, shape: str, fill: tuple[int, int, int], box):
    if shape == "square":
        draw.rectangle(box, fill=fill)
    elif shape == "circle":
        draw.ellipse(box, fill=fill)
    elif shape == "triangle":
        x0, y0, x1, y1 = box
        draw.polygon([(x0 + (x1 - x0) // 2, y0), (x0, y1), (x1, y1)], fill=fill)
    elif shape == "diamond":
        x0, y0, x1, y1 = box
        draw.polygon([(x0 + (x1 - x0) // 2, y0), (x1, (y0 + y1) // 2), (x0 + (x1 - x0) // 2, y1), (x0, (y0 + y1) // 2)], fill=fill)
    else:
        draw.rectangle(box, fill=fill)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw/sample")
    parser.add_argument("--manifest", default="data/raw/captions.csv")
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx in range(args.count):
        color_name, color_rgb = COLORS[idx % len(COLORS)]
        shape = SHAPES[idx % len(SHAPES)]
        bg = (245, 245, 245)
        img = Image.new("RGB", (224, 224), bg)
        draw = ImageDraw.Draw(img)
        inset = 36 + (idx % 4) * 8
        offset = (idx % 3) * 6
        box = (inset + offset, inset, 224 - inset + offset - 8, 224 - inset)
        draw_shape(draw, shape, color_rgb, box)
        filename = f"sample_{idx:03d}.png"
        path = image_dir / filename
        img.save(path)
        caption = f"a {color_name} {shape} on a light background"
        rows.append({"image_id": f"sample_{idx:03d}", "image_path": str(path), "caption": caption})

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "image_path", "caption"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
