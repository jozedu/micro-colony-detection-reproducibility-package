#!/usr/bin/env python3
"""
benchmark_inference_speed.py
============================
Benchmark end-to-end inference speed for Detectron2 and YOLO models.

This script is intentionally standalone and paper-oriented:
- it auto-discovers models inside a reviewer payload root
- it benchmarks on one or more named image directories
- it writes one flat CSV with per-model, per-benchmark timing results

Timing policy:
- image paths are selected first
- images are decoded on demand during warmup/timed passes
- warmup passes are excluded
- reported latency is wall-clock prediction time per image
- on CUDA, timings are synchronized before and after each inference call
"""

from __future__ import annotations

import argparse
import csv
import gc
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODEL_ZOO_BY_TYPE = {
    ("faster_rcnn", "R_50"): "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    ("faster_rcnn", "R_101"): "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
    ("retinanet", "R_50"): "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
    ("retinanet", "R_101"): "COCO-Detection/retinanet_R_101_FPN_3x.yaml",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class BenchmarkSpec:
    name: str
    image_dir: Path


@dataclass
class ModelSpec:
    framework: str
    model_name: str
    raw_model_name: str
    payload_group: str
    model_dir: Path
    weights_path: Path
    model_family: str
    backbone: str
    trained_on: str
    num_classes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark inference speed for Detectron2 and YOLO reviewer payloads.",
    )
    parser.add_argument(
        "--payload-root",
        type=Path,
        required=True,
        help="Root directory containing benchmarked model directories (e.g. TESE/outputs_detectron2).",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        required=True,
        help="Named benchmark image directory in the form name=/path/to/images.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--framework-filter",
        default="all",
        choices=("all", "detectron2", "yolo"),
        help="Framework filter.",
    )
    parser.add_argument(
        "--model-name-filter",
        default="",
        help="Optional case-insensitive substring filter applied to raw model names.",
    )
    parser.add_argument(
        "--warmup-images",
        type=int,
        default=20,
        help="Number of warmup images per model/benchmark.",
    )
    parser.add_argument(
        "--timed-images",
        type=int,
        default=100,
        help="Number of timed images per model/benchmark.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device string for both frameworks, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--yolo-imgsz",
        type=int,
        default=640,
        help="YOLO inference image size.",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.001,
        help="YOLO confidence threshold.",
    )
    parser.add_argument(
        "--yolo-iou",
        type=float,
        default=0.6,
        help="YOLO NMS IoU threshold.",
    )
    parser.add_argument(
        "--yolo-max-det",
        type=int,
        default=100,
        help="YOLO max detections per image.",
    )
    parser.add_argument(
        "--detectron2-score-thresh",
        type=float,
        default=0.0,
        help="Detectron2 score threshold.",
    )
    parser.add_argument(
        "--detectron2-dets-per-image",
        type=int,
        default=100,
        help="Detectron2 max detections per image.",
    )
    return parser.parse_args()


def require_dir(path: Path, label: str) -> Path:
    if not path.exists() or not path.is_dir():
        print(f"ERROR: {label} not found: {path}")
        sys.exit(1)
    return path


def parse_benchmarks(values: list[str]) -> list[BenchmarkSpec]:
    out: list[BenchmarkSpec] = []
    for value in values:
        if "=" not in value:
            print(f"ERROR: invalid --benchmark value: {value}")
            print("Use: --benchmark name=/path/to/images")
            sys.exit(1)
        name, raw_path = value.split("=", 1)
        name = name.strip()
        path = Path(raw_path.strip()).expanduser().resolve()
        if not name:
            print(f"ERROR: empty benchmark name in: {value}")
            sys.exit(1)
        require_dir(path, f"benchmark image dir ({name})")
        out.append(BenchmarkSpec(name=name, image_dir=path))
    return out


def normalize_model_name(name: str) -> str:
    return re.sub(
        r"_(?:\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2}|\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$",
        "",
        name,
    )


def detect_detectron2_type(model_dir: Path) -> tuple[str, str]:
    name = model_dir.name.lower()
    if "faster" in name:
        family = "faster_rcnn"
    elif "retina" in name:
        family = "retinanet"
    else:
        raise ValueError(f"Could not infer Detectron2 model family from {model_dir.name}")

    if "r_101" in name:
        backbone = "R_101"
    elif "r_50" in name:
        backbone = "R_50"
    else:
        raise ValueError(f"Could not infer Detectron2 backbone from {model_dir.name}")
    return family, backbone


def detect_subset_tag(model_dir: Path) -> str:
    parts = [model_dir.name.lower(), model_dir.parent.name.lower()]
    merged = " ".join(parts)
    for subset in ("curated", "total", "bright", "dark", "vague", "lowres"):
        if subset in merged:
            return subset
    return "unknown"


def discover_models(payload_root: Path, framework_filter: str, model_name_filter: str) -> list[ModelSpec]:
    payload_root = require_dir(payload_root, "payload root")
    name_filter = model_name_filter.strip().lower()
    models: list[ModelSpec] = []

    if framework_filter in ("all", "detectron2"):
        for weights in sorted(payload_root.rglob("model_final.pth")):
            model_dir = weights.parent
            raw_name = model_dir.name
            if name_filter and name_filter not in raw_name.lower():
                continue
            family, backbone = detect_detectron2_type(model_dir)
            trained_on = detect_subset_tag(model_dir)
            num_classes = 4 if trained_on == "curated" else 3
            models.append(
                ModelSpec(
                    framework="detectron2",
                    model_name=normalize_model_name(raw_name),
                    raw_model_name=raw_name,
                    payload_group=model_dir.parent.name,
                    model_dir=model_dir,
                    weights_path=weights,
                    model_family=family,
                    backbone=backbone,
                    trained_on=trained_on,
                    num_classes=num_classes,
                )
            )

    if framework_filter in ("all", "yolo"):
        for weights in sorted(payload_root.rglob("weights/best.pt")):
            model_dir = weights.parents[1]
            raw_name = model_dir.name
            if name_filter and name_filter not in raw_name.lower():
                continue
            trained_on = detect_subset_tag(model_dir)
            models.append(
                ModelSpec(
                    framework="yolo",
                    model_name=normalize_model_name(raw_name),
                    raw_model_name=raw_name,
                    payload_group=model_dir.parent.name,
                    model_dir=model_dir,
                    weights_path=weights,
                    model_family="yolov8",
                    backbone=normalize_model_name(raw_name),
                    trained_on=trained_on,
                    num_classes=4 if trained_on == "curated" else 3,
                )
            )

    return models


def collect_image_paths(image_dir: Path) -> list[Path]:
    paths = [
        p for p in sorted(image_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    if not paths:
        raise ValueError(f"No images found under {image_dir}")
    return paths


def load_image_cv2(path: Path) -> Any:
    try:
        import cv2
    except ImportError as exc:
        print(f"ERROR: opencv-python is required. {exc}")
        sys.exit(1)
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")
    return image


def split_warmup_timed_paths(image_paths: list[Path], warmup_n: int, timed_n: int) -> tuple[list[Path], list[Path]]:
    total_needed = max(1, warmup_n + timed_n)
    if len(image_paths) >= total_needed:
        selected = image_paths[:total_needed]
    else:
        repeats: list[Path] = []
        idx = 0
        while len(repeats) < total_needed:
            repeats.append(image_paths[idx % len(image_paths)])
            idx += 1
        selected = repeats
    warmup = selected[:warmup_n]
    timed = selected[warmup_n:warmup_n + timed_n]
    return warmup, timed


def load_detectron2_runtime() -> tuple[Any, Any, Any]:
    try:
        from PIL import Image
        if not hasattr(Image, "LINEAR") and hasattr(Image, "BILINEAR"):
            Image.LINEAR = Image.BILINEAR
        import torch
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
    except ImportError as exc:
        print(f"ERROR: torch + detectron2 are required for Detectron2 benchmarking. {exc}")
        sys.exit(1)
    return torch, model_zoo, get_cfg, DefaultPredictor


def load_yolo_runtime() -> tuple[Any, Any]:
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"ERROR: torch + ultralytics are required for YOLO benchmarking. {exc}")
        sys.exit(1)
    return torch, YOLO


def build_detectron2_cfg(
    *,
    get_cfg: Any,
    model_zoo: Any,
    model_cfg: str,
    weights_path: Path,
    model_family: str,
    num_classes: int,
    device: str,
    score_thresh: float,
    dets_per_image: int,
) -> Any:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_cfg))
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.DEVICE = device
    cfg.TEST.DETECTIONS_PER_IMAGE = dets_per_image
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATASETS.TEST = ()
    if model_family == "faster_rcnn":
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    else:
        cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_thresh
    return cfg


def maybe_sync_cuda(torch_mod: Any, device: str) -> None:
    if device.startswith("cuda") and torch_mod.cuda.is_available():
        torch_mod.cuda.synchronize()


def benchmark_detectron2_model(
    *,
    model: ModelSpec,
    warmup_paths: list[Path],
    timed_paths: list[Path],
    device: str,
    score_thresh: float,
    dets_per_image: int,
) -> dict[str, Any]:
    torch_mod, model_zoo, get_cfg, DefaultPredictor = load_detectron2_runtime()
    model_cfg = MODEL_ZOO_BY_TYPE.get((model.model_family, model.backbone))
    if model_cfg is None:
        raise ValueError(f"No model zoo config for {(model.model_family, model.backbone)}")

    cfg = build_detectron2_cfg(
        get_cfg=get_cfg,
        model_zoo=model_zoo,
        model_cfg=model_cfg,
        weights_path=model.weights_path,
        model_family=model.model_family,
        num_classes=model.num_classes,
        device=device,
        score_thresh=score_thresh,
        dets_per_image=dets_per_image,
    )
    predictor = DefaultPredictor(cfg)

    for path in warmup_paths:
        image = load_image_cv2(path)
        _ = predictor(image)
    maybe_sync_cuda(torch_mod, device)

    times_ms: list[float] = []
    for path in timed_paths:
        image = load_image_cv2(path)
        maybe_sync_cuda(torch_mod, device)
        start = time.perf_counter()
        _ = predictor(image)
        maybe_sync_cuda(torch_mod, device)
        times_ms.append((time.perf_counter() - start) * 1000.0)

    del predictor
    gc.collect()
    if torch_mod.cuda.is_available():
        torch_mod.cuda.empty_cache()

    return summarize_times(times_ms)


def benchmark_yolo_model(
    *,
    model: ModelSpec,
    warmup_paths: list[Path],
    timed_paths: list[Path],
    device: str,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
) -> dict[str, Any]:
    torch_mod, YOLO = load_yolo_runtime()
    yolo = YOLO(str(model.weights_path))

    for path in warmup_paths:
        image = load_image_cv2(path)
        _ = yolo.predict(
            source=image,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
            verbose=False,
            save=False,
            stream=False,
        )
    maybe_sync_cuda(torch_mod, device)

    times_ms: list[float] = []
    for path in timed_paths:
        image = load_image_cv2(path)
        maybe_sync_cuda(torch_mod, device)
        start = time.perf_counter()
        _ = yolo.predict(
            source=image,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
            verbose=False,
            save=False,
            stream=False,
        )
        maybe_sync_cuda(torch_mod, device)
        times_ms.append((time.perf_counter() - start) * 1000.0)

    del yolo
    gc.collect()
    if torch_mod.cuda.is_available():
        torch_mod.cuda.empty_cache()

    return summarize_times(times_ms)


def summarize_times(times_ms: list[float]) -> dict[str, float]:
    if not times_ms:
        raise ValueError("No timed inferences were recorded")
    ordered = sorted(times_ms)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    mean = sum(times_ms) / len(times_ms)
    fps = 1000.0 / mean if mean > 0 else 0.0
    return {
        "latency_ms_mean": mean,
        "latency_ms_median": median,
        "latency_ms_min": min(times_ms),
        "latency_ms_max": max(times_ms),
        "images_per_second": fps,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "framework",
        "model_name",
        "raw_model_name",
        "payload_group",
        "model_family",
        "backbone",
        "trained_on",
        "weights_path",
        "benchmark_name",
        "image_dir",
        "warmup_images",
        "timed_images",
        "device",
        "latency_ms_mean",
        "latency_ms_median",
        "latency_ms_min",
        "latency_ms_max",
        "images_per_second",
        "status",
        "error_message",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    payload_root = require_dir(args.payload_root.resolve(), "payload root")
    benchmarks = parse_benchmarks(args.benchmark)
    models = discover_models(payload_root, args.framework_filter, args.model_name_filter)

    print(f"Discovered {len(models)} models under {payload_root}")
    if not models:
        print("ERROR: no models discovered")
        sys.exit(1)

    benchmark_cache: dict[str, tuple[list[Path], list[Path], int]] = {}
    for spec in benchmarks:
        image_paths = collect_image_paths(spec.image_dir)
        warmup_paths, timed_paths = split_warmup_timed_paths(
            image_paths,
            warmup_n=max(0, args.warmup_images),
            timed_n=max(1, args.timed_images),
        )
        benchmark_cache[spec.name] = (warmup_paths, timed_paths, len(image_paths))
        print(
            f"[benchmark] {spec.name}: {len(image_paths)} image paths "
            f"(warmup={len(warmup_paths)}, timed={len(timed_paths)})"
        )

    rows: list[dict[str, Any]] = []
    for model in models:
        for spec in benchmarks:
            warmup_paths, timed_paths, decoded_count = benchmark_cache[spec.name]
            print(f"[run] {model.framework} | {model.model_name} | benchmark={spec.name}")
            row = {
                "framework": model.framework,
                "model_name": model.model_name,
                "raw_model_name": model.raw_model_name,
                "payload_group": model.payload_group,
                "model_family": model.model_family,
                "backbone": model.backbone,
                "trained_on": model.trained_on,
                "weights_path": str(model.weights_path),
                "benchmark_name": spec.name,
                "image_dir": str(spec.image_dir),
                "warmup_images": len(warmup_paths),
                "timed_images": len(timed_paths),
                "device": args.device,
                "status": "ok",
                "error_message": "",
            }
            try:
                if model.framework == "detectron2":
                    metrics = benchmark_detectron2_model(
                        model=model,
                        warmup_paths=warmup_paths,
                        timed_paths=timed_paths,
                        device=args.device,
                        score_thresh=args.detectron2_score_thresh,
                        dets_per_image=args.detectron2_dets_per_image,
                    )
                else:
                    metrics = benchmark_yolo_model(
                        model=model,
                        warmup_paths=warmup_paths,
                        timed_paths=timed_paths,
                        device=args.device,
                        imgsz=args.yolo_imgsz,
                        conf=args.yolo_conf,
                        iou=args.yolo_iou,
                        max_det=args.yolo_max_det,
                    )
                row.update(metrics)
                print(
                    f"  [ok] mean={metrics['latency_ms_mean']:.2f} ms/img | "
                    f"fps={metrics['images_per_second']:.2f}"
                )
            except Exception as exc:
                row["status"] = "fail"
                row["error_message"] = str(exc)
                print(f"  [error] {exc}")
            rows.append(row)

    out_csv = args.out_csv.resolve()
    write_rows(out_csv, rows)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
