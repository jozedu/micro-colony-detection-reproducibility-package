#!/usr/bin/env python3
"""
convert_coco_to_yolo.py
=======================
Convert one COCO split JSON into YOLO labels/images layout.

This implements the conversion logic used for the paper YOLO datasets:
- class ids are remapped by sorted COCO category ids (0-indexed)
- bbox conversion: [x, y, w, h] -> [cx, cy, w, h] normalized to [0, 1]
- one label file per image (empty file for images with no annotations)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a COCO split to YOLO format")
    parser.add_argument("--coco-json", type=Path, required=True, help="Path to COCO JSON file")
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="AGAR raw root (contains dataset/images or images)",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output YOLO dataset root")
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        required=True,
        help="Split name under images/<split> and labels/<split>",
    )
    parser.add_argument(
        "--image-mode",
        choices=("copy", "symlink", "none"),
        default="copy",
        help="How to materialize images in output (default: copy)",
    )
    parser.add_argument(
        "--strict-images",
        action="store_true",
        help="Fail if any source image is missing",
    )
    return parser.parse_args()


def find_image_roots(raw_root: Path) -> list[Path]:
    candidates = [
        raw_root / "dataset" / "images",
        raw_root / "images",
        raw_root / "dataset",
        raw_root,
    ]
    return [p for p in candidates if p.exists() and p.is_dir()]


def resolve_image_path(file_name: str, image_roots: list[Path]) -> Path | None:
    for root in image_roots:
        candidate = root / file_name
        if candidate.exists():
            return candidate
    return None


def write_data_yaml(
    dataset_dir: Path,
    class_names: list[str],
    train_rel: str,
    val_rel: str,
    test_rel: str,
) -> Path:
    out_path = dataset_dir / "data.yaml"
    lines = [
        f"path: {dataset_dir}",
        f"train: {train_rel}",
        f"val: {val_rel}",
        f"test: {test_rel}",
        f"nc: {len(class_names)}",
        "names:",
    ]
    lines.extend([f"  - {name}" for name in class_names])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def convert_split(
    coco_json_path: Path,
    raw_root: Path,
    out_dir: Path,
    split: str,
    image_mode: str = "copy",
    strict_images: bool = False,
) -> dict[str, Any]:
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    out_labels_dir = out_dir / "labels" / split
    out_images_dir = out_dir / "images" / split
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    out_images_dir.mkdir(parents=True, exist_ok=True)

    # Class id is index in sorted category ids.
    categories = sorted(coco["categories"], key=lambda c: c["id"])
    cat_ids = [c["id"] for c in categories]
    cat_map = {cid: idx for idx, cid in enumerate(cat_ids)}
    class_names = [c["name"] for c in categories]

    img_lookup = {img["id"]: img for img in coco["images"]}
    anns_by_img: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    image_roots = find_image_roots(raw_root)
    if not image_roots and image_mode != "none":
        raise FileNotFoundError(
            f"No image directories found under raw root: {raw_root}. "
            "Expected dataset/images or images."
        )

    n_images = 0
    n_missing_images = 0

    for img_id, img_info in img_lookup.items():
        img_w = float(img_info["width"])
        img_h = float(img_info["height"])
        file_name = str(img_info["file_name"])
        stem = Path(file_name).stem

        label_path = out_labels_dir / f"{stem}.txt"
        with open(label_path, "w", encoding="utf-8") as lf:
            for ann in anns_by_img.get(img_id, []):
                x, y, w, h = ann["bbox"]

                cx = (float(x) + float(w) / 2.0) / img_w
                cy = (float(y) + float(h) / 2.0) / img_h
                nw = float(w) / img_w
                nh = float(h) / img_h

                # Clamp values to [0, 1].
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))

                class_id = cat_map[ann["category_id"]]
                lf.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        if image_mode != "none":
            src = resolve_image_path(file_name, image_roots)
            dst = out_images_dir / Path(file_name).name
            if src is None:
                n_missing_images += 1
                if strict_images:
                    raise FileNotFoundError(f"Image not found for {file_name}")
            elif not dst.exists():
                if image_mode == "copy":
                    shutil.copy2(src, dst)
                else:
                    os.symlink(src.resolve(), dst)

        n_images += 1

    return {
        "images_processed": n_images,
        "missing_images": n_missing_images,
        "class_names": class_names,
        "labels_dir": str(out_labels_dir),
        "images_dir": str(out_images_dir),
    }


def main() -> None:
    args = parse_args()
    summary = convert_split(
        coco_json_path=args.coco_json,
        raw_root=args.raw_root,
        out_dir=args.out_dir,
        split=args.split,
        image_mode=args.image_mode,
        strict_images=args.strict_images,
    )

    # For single-file conversions, create a yaml that points all keys to this split.
    yaml_path = write_data_yaml(
        dataset_dir=args.out_dir.resolve(),
        class_names=summary["class_names"],
        train_rel=f"images/{args.split}",
        val_rel=f"images/{args.split}",
        test_rel=f"images/{args.split}",
    )

    print(f"Converted {summary['images_processed']} images")
    print(f"Missing images: {summary['missing_images']}")
    print(f"Labels dir: {summary['labels_dir']}")
    print(f"Images dir: {summary['images_dir']}")
    print(f"data.yaml: {yaml_path}")

    if args.strict_images and summary["missing_images"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
