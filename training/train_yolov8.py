#!/usr/bin/env python3
"""
train_yolov8.py
===============
Train YOLOv8 models for the review-process experiments:
1) AGAR total dataset (100 epochs, patience 20 by default)
2) Curated dataset (1000 epochs, patience 200 by default)

Datasets must be in YOLO format and referenced by data.yaml.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 on AGAR total and/or curated datasets.",
    )
    parser.add_argument(
        "--mode",
        choices=("agar", "curated", "both"),
        default="both",
        help="Which training runs to execute.",
    )
    parser.add_argument(
        "--agar-data",
        type=Path,
        default=Path("reproduced_yolo/yolo_agar_total/data.yaml"),
        help=(
            "Path to AGAR total YOLO data.yaml. "
            "Default expects output from dataset/reproduce_yolo_datasets.py --mode baseline."
        ),
    )
    parser.add_argument(
        "--curated-data",
        type=Path,
        help="Path to curated YOLO data.yaml.",
    )
    parser.add_argument("--agar-epochs", type=int, default=100)
    parser.add_argument("--curated-epochs", type=int, default=1000)
    parser.add_argument("--agar-patience", type=int, default=20)
    parser.add_argument("--curated-patience", type=int, default=200)
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics model or checkpoint path (e.g., yolov8n.pt, yolov8s.pt).",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None, help="Device id or 'cpu'.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs_yolov8"),
        help="Base directory where Ultralytics training outputs are written.",
    )
    parser.add_argument(
        "--run-prefix",
        default="yolov8_review",
        help="Prefix used in run directory names.",
    )
    parser.add_argument(
        "--eval-test-after-train",
        action="store_true",
        help="Run model.val(..., split='test') after each training run.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        print(f"ERROR: {label} not found: {path}")
        sys.exit(1)
    if not path.is_file():
        print(f"ERROR: {label} is not a file: {path}")
        sys.exit(1)
    return path


def run_training(
    *,
    dataset_tag: str,
    data_yaml: Path,
    epochs: int,
    patience: int,
    args: argparse.Namespace,
    yolo_cls: type,
) -> None:
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    run_name = f"{args.run_prefix}_{dataset_tag}_{timestamp}"

    print("=" * 72)
    print(f"Starting YOLOv8 training: {dataset_tag}")
    print("=" * 72)
    print(f"model: {args.model}")
    print(f"data: {data_yaml}")
    print(f"epochs: {epochs}")
    print(f"patience: {patience}")
    print(f"imgsz: {args.imgsz}")
    print(f"batch_size: {args.batch_size}")
    print(f"device: {args.device if args.device else 'auto'}")
    print(f"workers: {args.workers}")
    print(f"seed: {args.seed}")
    print(f"output_root: {args.output_root.resolve()}")
    print(f"run_name: {run_name}")

    model = yolo_cls(args.model)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        patience=patience,
        imgsz=args.imgsz,
        batch=args.batch_size,
        project=str(args.output_root.resolve()),
        name=run_name,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
    )

    if args.eval_test_after_train:
        print(f"Running test evaluation for {dataset_tag}...")
        model.val(data=str(data_yaml), split="test")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print("ERROR: ultralytics is required for YOLOv8 training.")
        print("Install with: python -m pip install ultralytics")
        print(f"Import failure: {exc}")
        sys.exit(1)

    run_agar = args.mode in ("agar", "both")
    run_curated = args.mode in ("curated", "both")

    agar_yaml = require_file(args.agar_data.resolve(), "AGAR data.yaml") if run_agar else None

    curated_yaml = None
    if run_curated:
        if args.curated_data is None:
            print("ERROR: --curated-data is required when --mode is 'curated' or 'both'.")
            sys.exit(1)
        curated_yaml = require_file(args.curated_data.resolve(), "curated data.yaml")

    if run_agar:
        run_training(
            dataset_tag="agar_total",
            data_yaml=agar_yaml,
            epochs=args.agar_epochs,
            patience=args.agar_patience,
            args=args,
            yolo_cls=YOLO,
        )

    if run_curated:
        run_training(
            dataset_tag="curated",
            data_yaml=curated_yaml,
            epochs=args.curated_epochs,
            patience=args.curated_patience,
            args=args,
            yolo_cls=YOLO,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
