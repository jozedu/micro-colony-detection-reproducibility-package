#!/usr/bin/env python3
"""
reproduce_yolo_datasets.py
==========================
Build YOLO-formatted datasets from reproduced COCO split JSON files.

This reproduces the two YOLO conversion workflows used for the paper:
1) Baseline conversion
   - Convert total train/val/test to YOLO dataset + data.yaml
2) Cross-subset conversion
   - Convert subset test splits (bright, vague, lowres, dark) + per-subset data.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from convert_coco_to_yolo import convert_split, write_data_yaml


SUBSETS_CROSS = ("bright", "vague", "lowres", "dark")


def split_path(repro_splits_dir: Path, split: str, group: str) -> Path:
    candidates = [
        repro_splits_dir / f"{group}_{split}_coco.json",
        repro_splits_dir / f"{group}_{split}.json",
        repro_splits_dir / f"{split}_{group}100.json",
        repro_splits_dir / f"{split}_annotated_{group}100.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce YOLO datasets from reproduced COCO splits")
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="AGAR raw root (contains dataset/images or images)",
    )
    parser.add_argument(
        "--repro-splits",
        type=Path,
        default=Path("reproduced_splits"),
        help="Directory created by reproduce_splits.py --mode reconstruct",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reproduced_yolo"),
        help="Output base directory",
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "cross-subset", "all"),
        default="all",
        help="Which conversion workflow to run",
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
        help="Fail on missing images",
    )
    return parser.parse_args()


def ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required split file: {path}")


def run_baseline(args: argparse.Namespace) -> None:
    """
    Baseline paper workflow:
    - Convert total train/val/test
    - Create one data.yaml with train/val/test paths
    """
    print("=" * 70)
    print("Baseline conversion (total train/val/test)")
    print("=" * 70)

    yolo_dataset_dir = args.out_dir / "yolo_agar_total"
    splits = ("train", "val", "test")
    class_names = None

    for split in splits:
        coco_json = split_path(args.repro_splits, split, "total")
        ensure_file(coco_json)
        summary = convert_split(
            coco_json_path=coco_json,
            raw_root=args.raw_root,
            out_dir=yolo_dataset_dir,
            split=split,
            image_mode=args.image_mode,
            strict_images=args.strict_images,
        )
        class_names = summary["class_names"]
        print(
            f"{split}: converted {summary['images_processed']} images "
            f"(missing images: {summary['missing_images']})"
        )

    assert class_names is not None
    yaml_path = write_data_yaml(
        dataset_dir=yolo_dataset_dir.resolve(),
        class_names=class_names,
        train_rel="images/train",
        val_rel="images/val",
        test_rel="images/test",
    )
    print(f"Saved: {yaml_path}")


def run_cross_subset(args: argparse.Namespace) -> None:
    """
    Cross-subset paper workflow:
    - Convert test split only for bright/vague/lowres/dark
    - Create one data.yaml per subset with train=val=test=images/test
    """
    print("=" * 70)
    print("Cross-subset conversion (subset test only)")
    print("=" * 70)

    yolo_subsets_dir = args.out_dir / "yolo_subsets"

    # Paper workflow: class names are read from total train.
    total_train_json = split_path(args.repro_splits, "train", "total")
    ensure_file(total_train_json)
    with open(total_train_json, "r", encoding="utf-8") as f:
        total_train = json.load(f)
    class_names_from_total = [
        c["name"] for c in sorted(total_train["categories"], key=lambda c: c["id"])
    ]

    for subset in SUBSETS_CROSS:
        test_json = split_path(args.repro_splits, "test", subset)
        ensure_file(test_json)
        subset_dir = yolo_subsets_dir / subset
        summary = convert_split(
            coco_json_path=test_json,
            raw_root=args.raw_root,
            out_dir=subset_dir,
            split="test",
            image_mode=args.image_mode,
            strict_images=args.strict_images,
        )

        yaml_path = write_data_yaml(
            dataset_dir=subset_dir.resolve(),
            class_names=class_names_from_total,
            train_rel="images/test",
            val_rel="images/test",
            test_rel="images/test",
        )
        print(
            f"{subset}: converted {summary['images_processed']} test images "
            f"(missing images: {summary['missing_images']})"
        )
        print(f"  Saved: {yaml_path}")


def main() -> None:
    args = parse_args()
    args.repro_splits = args.repro_splits.resolve()
    args.raw_root = args.raw_root.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("baseline", "all"):
        run_baseline(args)
    if args.mode in ("cross-subset", "all"):
        run_cross_subset(args)

    print("\nDone.")


if __name__ == "__main__":
    main()
