#!/usr/bin/env python3
"""
run_stress_test_detectron2.py
=============================
Stress-test pipeline for high-density AGAR images using Detectron2 models.

Pipeline stages:
1) Build extended cohort (no max annotation cap).
2) Build stress subsets by annotation density bins.
3) Leakage check against train/val splits.
4) Evaluate selected Detectron2 models on stress subsets.
5) Export long/wide CSV reports.

An additional reconstruction mode can rebuild CSV reports from existing
stress-test evaluation folders.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import traceback
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any


MODEL_ZOO_BY_TYPE = {
    ("faster_rcnn", "R_50"): "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    ("faster_rcnn", "R_101"): "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
    ("retinanet", "R_50"): "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
    ("retinanet", "R_101"): "COCO-Detection/retinanet_R_101_FPN_3x.yaml",
}

# AGAR curation constants (aligned with reproducibility_package/dataset/reproduce_splits.py)
SRC_KEEP_CAT_IDS = {0, 2, 3}
CAT_REMAP = {0: 0, 2: 1, 3: 2}
SPLIT_CATEGORIES = [
    {"supercategory": "microbes", "id": 0, "name": "S.aureus"},
    {"supercategory": "microbes", "id": 1, "name": "P.aeruginosa"},
    {"supercategory": "microbes", "id": 2, "name": "E.coli"},
]
COUNTABLE_ID_RANGES = [
    (309, 1302),
    (2712, 8709),
    (11761, 12617),
    (12994, 17417),
]


torch = None
detectron2 = None
COCOEvaluator = None
inference_on_dataset = None
build_detection_test_loader = None
DatasetCatalog = None
register_coco_instances = None
get_cfg = None
model_zoo = None
DefaultPredictor = None
PILLOW_LINEAR_SHIM_APPLIED = False


def load_runtime_dependencies() -> None:
    global torch
    global detectron2
    global COCOEvaluator
    global inference_on_dataset
    global build_detection_test_loader
    global DatasetCatalog
    global register_coco_instances
    global get_cfg
    global model_zoo
    global DefaultPredictor
    global PILLOW_LINEAR_SHIM_APPLIED

    try:
        from PIL import Image

        if not hasattr(Image, "LINEAR") and hasattr(Image, "BILINEAR"):
            Image.LINEAR = Image.BILINEAR
            PILLOW_LINEAR_SHIM_APPLIED = True

        import torch as _torch
        import detectron2 as _d2
        from detectron2 import model_zoo as _model_zoo
        from detectron2.config import get_cfg as _get_cfg
        from detectron2.data import DatasetCatalog as _DatasetCatalog
        from detectron2.data import build_detection_test_loader as _build_detection_test_loader
        from detectron2.data.datasets import register_coco_instances as _register_coco_instances
        from detectron2.engine import DefaultPredictor as _DefaultPredictor
        from detectron2.evaluation import COCOEvaluator as _COCOEvaluator
        from detectron2.evaluation import inference_on_dataset as _inference_on_dataset
    except ImportError as exc:
        print("ERROR: torch + detectron2 are required for stress-test evaluation mode.")
        print(f"Import failure: {exc}")
        sys.exit(1)

    torch = _torch
    detectron2 = _d2
    model_zoo = _model_zoo
    get_cfg = _get_cfg
    DatasetCatalog = _DatasetCatalog
    build_detection_test_loader = _build_detection_test_loader
    register_coco_instances = _register_coco_instances
    DefaultPredictor = _DefaultPredictor
    COCOEvaluator = _COCOEvaluator
    inference_on_dataset = _inference_on_dataset


def parse_float_list(value: str) -> list[float]:
    parts = [x.strip() for x in value.split(",") if x.strip()]
    if not parts:
        raise ValueError("empty float list")
    return [float(x) for x in parts]


def parse_bins(value: str) -> list[tuple[str, int, int]]:
    bins: list[tuple[str, int, int]] = []
    parts = [x.strip() for x in value.split(",") if x.strip()]
    if not parts:
        raise ValueError("empty bins")

    for part in parts:
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if not m:
            raise ValueError(f"invalid bin format: {part}. expected like 101-150")
        lo = int(m.group(1))
        hi = int(m.group(2))
        if lo > hi:
            raise ValueError(f"invalid bin range: {part}")
        bins.append((f"{lo}-{hi}", lo, hi))
    return bins


def parse_indices(value: str | None) -> list[int]:
    if not value:
        return []
    out: list[int] = []
    for p in value.split(","):
        s = p.strip()
        if not s:
            continue
        out.append(int(s))
    return out


def require_file(path: Path, label: str) -> Path:
    if not path.exists() or not path.is_file():
        print(f"ERROR: {label} not found: {path}")
        sys.exit(1)
    return path


def require_dir(path: Path, label: str) -> Path:
    if not path.exists() or not path.is_dir():
        print(f"ERROR: {label} not found: {path}")
        sys.exit(1)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Detectron2 stress-test pipeline.")
    parser.add_argument("--mode", choices=("run", "reconstruct"), default="run")

    parser.add_argument("--full-coco-json", type=Path, help="Original AGAR COCO annotations.json")
    parser.add_argument("--image-dir", type=Path, help="AGAR image root directory")
    parser.add_argument("--train-coco-json", type=Path, help="Train split COCO JSON for format/leakage checks")
    parser.add_argument("--val-coco-json", type=Path, help="Val split COCO JSON for leakage checks")

    parser.add_argument("--runs-root", type=Path, required=True, help="Root with Detectron2 model run folders")
    parser.add_argument("--out-dir", type=Path, default=Path("stress_test_results"), help="Output directory")

    parser.add_argument("--thresholds", default="0.0,0.25,0.5,0.75", help="Comma-separated score thresholds")
    parser.add_argument("--dets-per-image", type=int, default=300, help="TEST.DETECTIONS_PER_IMAGE")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--bins", default="101-150,151-300", help="Comma-separated bins like 101-150,151-300")

    parser.add_argument("--evaluate-all", action="store_true", help="Evaluate all discovered models")
    parser.add_argument("--selected-indices", help="Comma-separated model indices from discovery list")
    parser.add_argument("--filter-family", choices=("faster_rcnn", "retinanet"), help="Optional model family filter")
    parser.add_argument("--filter-backbone", choices=("R_50", "R_101"), help="Optional backbone filter")
    parser.add_argument(
        "--filter-subset",
        default="none",
        help="Optional training-subset filter (bright,dark,vague,lowres,total,curated,all,none)",
    )
    parser.add_argument("--run-name-contains", help="Optional substring filter on run folder name")

    parser.add_argument("--resume", action="store_true", help="Enable resume logic")
    parser.add_argument("--resume-model-index", type=int, default=0, help="Resume model index")
    parser.add_argument("--resume-test-set", default="101-150", help="Resume test-set name")
    parser.add_argument("--resume-threshold", type=float, default=0.0, help="Resume threshold")

    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--skip-env-capture", action="store_true")
    parser.add_argument(
        "--eval-dir-glob",
        default="*stress_test_eval_*",
        help="Eval directory glob for reconstruction mode",
    )

    return parser.parse_args()


def is_in_ranges(sample_id: int, ranges: list[tuple[int, int]]) -> bool:
    return any(lo <= sample_id <= hi for lo, hi in ranges)


def generate_extended_cohort(
    *,
    full_coco_json: Path,
    out_dir: Path,
    overwrite: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    extended_coco_path = out_dir / "extended_cohort_no_max.json"
    extended_counts_path = out_dir / "extended_cohort_counts.json"

    if extended_coco_path.exists() and not overwrite:
        with open(extended_coco_path, "r", encoding="utf-8") as f:
            extended_coco = json.load(f)
        counts = {}
        if extended_counts_path.exists():
            with open(extended_counts_path, "r", encoding="utf-8") as f:
                counts = json.load(f)
        print(f"[OK] Extended cohort loaded: {extended_coco_path}")
        return extended_coco, counts

    with open(full_coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)
    countable_ranges = COUNTABLE_ID_RANGES

    keep_count: Counter[int] = Counter()
    nonkeep_count: Counter[int] = Counter()
    ann_per_img: Counter[int] = Counter()

    for ann in coco["annotations"]:
        iid = int(ann["image_id"])
        ann_per_img[iid] += 1
        if int(ann["category_id"]) in SRC_KEEP_CAT_IDS:
            keep_count[iid] += 1
        else:
            nonkeep_count[iid] += 1

    img_by_id = {int(img["id"]): img for img in coco["images"]}

    curated_nonempty: set[int] = set()
    for sid in img_by_id:
        if not is_in_ranges(sid, countable_ranges):
            continue
        kc = keep_count.get(sid, 0)
        nkc = nonkeep_count.get(sid, 0)
        if nkc == 0 and kc >= 1:
            curated_nonempty.add(sid)

    curated_empty: set[int] = set()
    for sid, img in img_by_id.items():
        if int(img.get("items_count", 0)) == 0 and ann_per_img.get(sid, 0) == 0:
            curated_empty.add(sid)

    curated_ids = curated_nonempty | curated_empty

    curated_annotations: list[dict[str, Any]] = []
    new_ann_id = 1
    for ann in coco["annotations"]:
        iid = int(ann["image_id"])
        cid = int(ann["category_id"])
        if iid in curated_ids and cid in SRC_KEEP_CAT_IDS:
            new_ann = copy.deepcopy(ann)
            new_ann["id"] = new_ann_id
            new_ann["category_id"] = CAT_REMAP[cid]
            curated_annotations.append(new_ann)
            new_ann_id += 1

    output_categories = copy.deepcopy(SPLIT_CATEGORIES)

    curated_images = sorted([copy.deepcopy(img_by_id[sid]) for sid in curated_ids], key=lambda x: x["id"])
    items_lookup = Counter(int(ann["image_id"]) for ann in curated_annotations)
    for img in curated_images:
        img["items_count"] = items_lookup.get(int(img["id"]), 0)

    curated_annotations.sort(key=lambda x: x["id"])

    extended_coco: dict[str, Any] = {
        "images": curated_images,
        "annotations": curated_annotations,
        "categories": output_categories,
    }
    for key in coco:
        if key not in ("images", "annotations", "categories"):
            extended_coco[key] = coco[key]

    with open(extended_coco_path, "w", encoding="utf-8") as f:
        json.dump(extended_coco, f)

    keep_ann_distribution = Counter(keep_count[iid] for iid in curated_nonempty)
    counts_report = {
        "n_images_total": len(curated_ids),
        "n_images_nonempty": len(curated_nonempty),
        "n_images_empty": len(curated_empty),
        "n_annotations_total": len(curated_annotations),
        "keep_annotation_distribution": dict(keep_ann_distribution.most_common()),
        "max_annotations": max(keep_ann_distribution.keys()) if keep_ann_distribution else 0,
        "min_annotations": min(keep_ann_distribution.keys()) if keep_ann_distribution else 0,
    }
    with open(extended_counts_path, "w", encoding="utf-8") as f:
        json.dump(counts_report, f, indent=2)

    print(f"[OK] Extended cohort generated: {extended_coco_path}")
    return extended_coco, counts_report


def validate_category_format(extended_coco: dict[str, Any], train_coco_json: Path) -> None:
    with open(train_coco_json, "r", encoding="utf-8") as f:
        target = json.load(f)

    target_cats = sorted(target["categories"], key=lambda x: x["id"])
    extended_cats = sorted(extended_coco["categories"], key=lambda x: x["id"])

    errors: list[str] = []
    if len(target_cats) != len(extended_cats):
        errors.append(f"category count mismatch: {len(extended_cats)} vs {len(target_cats)}")
    else:
        for tc, ec in zip(target_cats, extended_cats):
            if int(tc["id"]) != int(ec["id"]) or str(tc["name"]) != str(ec["name"]):
                errors.append(f"category mismatch: {ec} vs {tc}")

    if errors:
        print("[ERROR] Format validation failed:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("[OK] Format validation passed")


def build_stress_subsets(
    *,
    extended_coco: dict[str, Any],
    bins: list[tuple[str, int, int]],
    out_dir: Path,
) -> tuple[dict[str, Path], Counter[int], set[int]]:
    stress_test_dir = out_dir / "stress_test_subsets"
    stress_test_dir.mkdir(parents=True, exist_ok=True)

    ann_counts: Counter[int] = Counter()
    for ann in extended_coco["annotations"]:
        ann_counts[int(ann["image_id"])] += 1

    stress_test_paths: dict[str, Path] = {}
    all_gt100_images: set[int] = set()

    print("Building stress test subsets:")
    for bin_name, low, high in bins:
        bin_images = {iid for iid in ann_counts if low <= ann_counts[iid] <= high}
        all_gt100_images.update(bin_images)

        subset_images = [img for img in extended_coco["images"] if int(img["id"]) in bin_images]
        subset_annotations = [ann for ann in extended_coco["annotations"] if int(ann["image_id"]) in bin_images]

        subset_coco = {
            "images": sorted(subset_images, key=lambda x: x["id"]),
            "annotations": sorted(subset_annotations, key=lambda x: x["id"]),
            "categories": extended_coco["categories"],
        }
        for key in extended_coco:
            if key not in ("images", "annotations", "categories"):
                subset_coco[key] = extended_coco[key]

        output_path = stress_test_dir / f"stress_test_{bin_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(subset_coco, f)

        stress_test_paths[bin_name] = output_path
        print(f"  {bin_name}: {len(subset_images)} images, {len(subset_annotations)} annotations")

    gt100_images = [img for img in extended_coco["images"] if int(img["id"]) in all_gt100_images]
    gt100_annotations = [ann for ann in extended_coco["annotations"] if int(ann["image_id"]) in all_gt100_images]

    gt100_coco = {
        "images": sorted(gt100_images, key=lambda x: x["id"]),
        "annotations": sorted(gt100_annotations, key=lambda x: x["id"]),
        "categories": extended_coco["categories"],
    }
    for key in extended_coco:
        if key not in ("images", "annotations", "categories"):
            gt100_coco[key] = extended_coco[key]

    gt100_path = stress_test_dir / "stress_test_gt100_combined.json"
    with open(gt100_path, "w", encoding="utf-8") as f:
        json.dump(gt100_coco, f)
    stress_test_paths["gt100"] = gt100_path

    print(f"  gt100: {len(gt100_images)} images, {len(gt100_annotations)} annotations")
    print(f"[OK] Stress subsets saved in {stress_test_dir}")

    return stress_test_paths, ann_counts, all_gt100_images


def write_density_stats(
    *,
    full_coco_json: Path,
    extended_coco: dict[str, Any],
    ann_counts: Counter[int],
    bins: list[tuple[str, int, int]],
    all_gt100_images: set[int],
    out_dir: Path,
) -> None:
    with open(full_coco_json, "r", encoding="utf-8") as f:
        original_coco = json.load(f)

    original_ann_counts: Counter[int] = Counter()
    for ann in original_coco["annotations"]:
        original_ann_counts[int(ann["image_id"])] += 1

    original_gt100_ids = [iid for iid, cnt in original_ann_counts.items() if cnt > 100]
    original_gt100 = len(original_gt100_ids)

    extended_gt100 = len(all_gt100_images)
    extended_gt100_annotations = sum(1 for ann in extended_coco["annotations"] if int(ann["image_id"]) in all_gt100_images)

    stats = {
        "original_total_images": len(original_coco["images"]),
        "original_gt100_images": original_gt100,
        "original_gt100_annotations": sum(original_ann_counts[iid] for iid in original_gt100_ids),
        "extended_total_images": len(extended_coco["images"]),
        "extended_gt100_images": extended_gt100,
        "extended_gt100_annotations": extended_gt100_annotations,
        "retention_rate_pct": (extended_gt100 / original_gt100 * 100.0) if original_gt100 else 0.0,
        "bins": {
            name: {
                "low": low,
                "high": high,
                "n_images": len([iid for iid in ann_counts if low <= ann_counts[iid] <= high]),
            }
            for name, low, high in bins
        },
    }

    stats_path = out_dir / "density_stats_gt100.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"[OK] Density stats saved: {stats_path}")


def run_leakage_check(
    *,
    train_coco_json: Path,
    val_coco_json: Path,
    stress_test_paths: dict[str, Path],
    out_dir: Path,
) -> dict[str, Any]:
    with open(train_coco_json, "r", encoding="utf-8") as f:
        train = json.load(f)
    with open(val_coco_json, "r", encoding="utf-8") as f:
        val = json.load(f)

    train_ids = {int(img["id"]) for img in train["images"]}
    val_ids = {int(img["id"]) for img in val["images"]}
    trainval_ids = train_ids | val_ids

    leakage_report: dict[str, Any] = {}
    leakage_detected = False

    for subset_name, subset_path in stress_test_paths.items():
        with open(subset_path, "r", encoding="utf-8") as f:
            subset = json.load(f)
        subset_ids = {int(img["id"]) for img in subset["images"]}
        id_overlap = subset_ids & trainval_ids

        leakage_report[subset_name] = {
            "n_images": len(subset_ids),
            "leaked_ids": len(id_overlap),
            "has_leakage": len(id_overlap) > 0,
        }

        if id_overlap:
            leakage_detected = True
            print(f"  [ERROR] {subset_name}: {len(id_overlap)} leaked IDs")
        else:
            print(f"  [OK] {subset_name}: no leakage")

    leakage_report["overall"] = {"has_leakage": leakage_detected}
    leakage_report_path = out_dir / "leakage_check_report.json"
    with open(leakage_report_path, "w", encoding="utf-8") as f:
        json.dump(leakage_report, f, indent=2)

    if leakage_detected:
        print("[ERROR] Leakage detected")
    else:
        print("[OK] Leakage check passed")

    return leakage_report


def detect_model_family_backbone(name_lower: str) -> tuple[str, str]:
    if "faster" in name_lower:
        family = "faster_rcnn"
    elif "retina" in name_lower:
        family = "retinanet"
    else:
        family = "unknown"

    if "r_101" in name_lower:
        backbone = "R_101"
    elif "r_50" in name_lower:
        backbone = "R_50"
    else:
        backbone = "unknown"

    return family, backbone


def detect_training_subset(name_lower: str) -> str:
    if "final" in name_lower:
        return "curated"
    if "bright" in name_lower:
        return "bright"
    if "dark" in name_lower:
        return "dark"
    if "vague" in name_lower:
        return "vague"
    if "lowres" in name_lower:
        return "lowres"
    if "total" in name_lower:
        return "total"
    return "unknown"


def discover_models(runs_root: Path) -> list[dict[str, Any]]:
    checkpoints = sorted(runs_root.rglob("model_final.pth"))
    models: list[dict[str, Any]] = []

    for idx, ckpt in enumerate(checkpoints):
        model_dir = ckpt.parent
        model_name = model_dir.name
        name_lower = model_name.lower()
        family, backbone = detect_model_family_backbone(name_lower)
        subset = detect_training_subset(name_lower)

        models.append(
            {
                "index": idx,
                "name": model_name,
                "path": ckpt,
                "family": family,
                "backbone": backbone,
                "subset": subset,
            }
        )

    print(f"[OK] Found {len(models)} models")
    for model in models:
        print(
            f"  [{model['index']:>3}] {model['name']} "
            f"| {model['family']} | {model['backbone']} | subset={model['subset']}"
        )
    return models


def resolve_subset_filter(value: str | None) -> str | None:
    if value is None:
        return None
    val = value.strip().lower()
    if val in ("", "none", "all"):
        return None
    return val


def select_models(available_models: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.evaluate_all:
        selected = list(available_models)
        print("[OK] Selected all models")
        return selected

    subset_filter = resolve_subset_filter(args.filter_subset)

    filtered = list(available_models)
    if args.filter_family:
        filtered = [m for m in filtered if m["family"] == args.filter_family]
    if args.filter_backbone:
        filtered = [m for m in filtered if m["backbone"] == args.filter_backbone]
    if subset_filter:
        filtered = [m for m in filtered if m["subset"] == subset_filter]
    if args.run_name_contains:
        filtered = [m for m in filtered if args.run_name_contains in m["name"]]

    selected_indices = parse_indices(args.selected_indices)
    if selected_indices:
        selected = [available_models[i] for i in selected_indices if 0 <= i < len(available_models)]
        print(f"[OK] Selected by indices: {selected_indices}")
    else:
        selected = filtered
        print("[OK] Selected by filters")

    selected_by_path: dict[str, dict[str, Any]] = {}
    for m in selected:
        selected_by_path[str(m["path"].resolve())] = m
    final_selected = list(selected_by_path.values())

    print(f"[OK] Final selected models: {len(final_selected)}")
    for m in final_selected:
        print(f"  - {m['name']}")
    return final_selected


def capture_environment_info(out_dir: Path) -> dict[str, Any]:
    env_info = {
        "python_version": "unknown",
        "torch_version": "unknown",
        "detectron2_version": "unknown",
        "cuda_version": "unknown",
        "cudnn_version": "unknown",
        "device": "unknown",
        "timestamp_utc": dt.datetime.utcnow().isoformat(),
    }

    try:
        env_info["python_version"] = platform.python_version()
        env_info["torch_version"] = torch.__version__
        env_info["detectron2_version"] = detectron2.__version__
        env_info["cuda_version"] = torch.version.cuda if hasattr(torch.version, "cuda") else "N/A"
        env_info["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.enabled else "N/A"
        env_info["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception as exc:
        print(f"[warn] Could not capture full environment: {exc}")

    with open(out_dir / "env_versions.json", "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)

    try:
        freeze_output = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        with open(out_dir / "requirements_freeze.txt", "w", encoding="utf-8") as f:
            f.write(freeze_output)
    except Exception as exc:
        print(f"[warn] Could not capture pip freeze: {exc}")

    print(
        f"[env] Python {env_info['python_version']} | Torch {env_info['torch_version']} | "
        f"Detectron2 {env_info['detectron2_version']} | CUDA {env_info['cuda_version']}"
    )
    return env_info


def ensure_dataset(dataset_name: str, json_path: Path, images_dir: Path) -> None:
    if dataset_name in DatasetCatalog.list():
        return
    register_coco_instances(dataset_name, {}, str(json_path), str(images_dir))


def build_eval_cfg(
    *,
    model_cfg: str,
    weights_path: Path,
    dataset_name: str,
    num_classes: int,
    model_family: str,
    threshold: float,
    dets_per_image: int,
    num_workers: int,
) -> Any:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_cfg))
    cfg.DATASETS.TEST = (dataset_name,)
    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.TEST.DETECTIONS_PER_IMAGE = dets_per_image
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if model_family == "faster_rcnn":
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = max(10000, dets_per_image * 50)
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = max(5000, dets_per_image * 20)
    else:
        cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold
        cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST = max(5000, dets_per_image * 50)

    return cfg


def extract_metrics_from_results(results: dict[str, Any], dets_per_image: int) -> dict[str, Any]:
    bbox = results.get("bbox", results)
    ar_key = f"AR@{dets_per_image}"

    metrics = {
        "AP": bbox.get("AP"),
        "AP50": bbox.get("AP50"),
        "AP75": bbox.get("AP75"),
        "APs": bbox.get("APs"),
        "APm": bbox.get("APm"),
        "APl": bbox.get("APl"),
        f"AR{dets_per_image}": bbox.get(ar_key, bbox.get("AR@100", bbox.get("AR100", bbox.get("AR")))),
        "AP_E_coli": bbox.get("AP-E.coli", bbox.get("AP-E_coli")),
        "AP_P_aeruginosa": bbox.get("AP-P.aeruginosa", bbox.get("AP-P_aeruginosa")),
        "AP_S_aureus": bbox.get("AP-S.aureus", bbox.get("AP-S_aureus")),
    }
    return metrics


def extract_ar_from_eval_text(eval_text: str, dets_per_image: int) -> float | None:
    pattern = (
        r"Average Recall\s+\(AR\)\s+@\[\s*IoU=0.50:0.95\s*\|\s*area=\s*all\s*"
        r"\|\s*maxDets=\s*" + str(dets_per_image) + r"\s*\]\s*=\s*([\d.]+)"
    )
    m = re.search(pattern, eval_text)
    if not m:
        return None
    try:
        return float(m.group(1)) * 100.0
    except ValueError:
        return None


def generate_run_id(run_dir: Path) -> str:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    h = hashlib.md5(str(run_dir).encode("utf-8")).hexdigest()[:8]
    return f"{ts}_{h}"


def write_run_meta_json(
    *,
    eval_dir: Path,
    run_id: str,
    model_info: dict[str, Any],
    test_set_name: str,
    env_info: dict[str, Any],
    thresholds: list[float],
    dets_per_image: int,
) -> None:
    meta_path = eval_dir / f"run_meta_{test_set_name}.json"
    meta = {
        "run_id": run_id,
        "timestamp_utc": dt.datetime.utcnow().isoformat(),
        "model_name": model_info["name"],
        "model_family": model_info["family"],
        "backbone": model_info["backbone"],
        "weights_path": str(model_info["path"]),
        "test_set": test_set_name,
        "dataset_tag": "agar_stress_test",
        "classes": ["E.coli", "P.aeruginosa", "S.aureus"],
        "num_classes": 3,
        "thresholds": thresholds,
        "dets_per_image": dets_per_image,
        "versions": env_info,
        "eval_invocation": "stress_test/run_stress_test_detectron2.py",
        "notes": "Stress test evaluation on high-density subsets (>100 annotations)",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def evaluate_selected_models(
    *,
    selected_models: list[dict[str, Any]],
    stress_test_paths: dict[str, Path],
    image_dir: Path,
    thresholds: list[float],
    dets_per_image: int,
    out_dir: Path,
    overwrite: bool,
    num_workers: int,
    resume: bool,
    resume_model_index: int,
    resume_test_set: str,
    resume_threshold: float,
    env_info: dict[str, Any],
) -> list[dict[str, Any]]:
    csv_rows: list[dict[str, Any]] = []
    eval_timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print("=" * 80)
    print(f"Starting stress evaluation: {eval_timestamp}")
    print("=" * 80)

    test_sets_list = list(stress_test_paths.items())

    for model_idx, model_info in enumerate(selected_models):
        if resume and model_idx < resume_model_index:
            print(f"\n[{model_idx+1}/{len(selected_models)}] Skipping (resume): {model_info['name']}")
            continue

        print(f"\n[{model_idx+1}/{len(selected_models)}] Evaluating: {model_info['name']}")

        model_path = model_info["path"]
        model_dir = model_path.parent
        model_family = model_info["family"]
        backbone = model_info["backbone"]

        model_cfg = MODEL_ZOO_BY_TYPE.get((model_family, backbone))
        if model_cfg is None:
            print("  [SKIP] Skipping unknown model family/backbone")
            continue

        eval_dir = model_dir / f"{dets_per_image}stress_test_eval_{eval_timestamp}"
        eval_dir.mkdir(parents=True, exist_ok=True)

        run_id = generate_run_id(model_dir)

        test_set_start_idx = 0
        if resume and model_idx == resume_model_index:
            for idx, (name, _) in enumerate(test_sets_list):
                if name == resume_test_set:
                    test_set_start_idx = idx
                    break
            print(f"  [RESUME] Test set start: {resume_test_set}")

        per_model_summary: list[str] = []

        for test_idx, (test_name, test_json) in enumerate(test_sets_list):
            if resume and model_idx == resume_model_index and test_idx < test_set_start_idx:
                print(f"\n  Test set: {test_name} (skipped - resume)")
                continue

            print(f"\n  Test set: {test_name}")

            write_run_meta_json(
                eval_dir=eval_dir,
                run_id=run_id,
                model_info=model_info,
                test_set_name=test_name,
                env_info=env_info,
                thresholds=thresholds,
                dets_per_image=dets_per_image,
            )

            dataset_name = (
                f"stress_{test_name}_{model_dir.name.replace('-', '_')}_"
                f"{abs(hash(str(model_dir))) % 100000}"
            )
            ensure_dataset(dataset_name, test_json, image_dir)

            threshold_start_idx = 0
            if resume and model_idx == resume_model_index and test_name == resume_test_set:
                for idx, thr in enumerate(thresholds):
                    if abs(thr - resume_threshold) < 1e-6:
                        threshold_start_idx = idx
                        break
                print(f"    [RESUME] Threshold start: {resume_threshold}")

            for thr_idx, threshold in enumerate(thresholds):
                if resume and model_idx == resume_model_index and test_name == resume_test_set and thr_idx < threshold_start_idx:
                    print(f"    Threshold {threshold:.2f} (skipped - resume)")
                    continue

                print(f"    Threshold {threshold:.2f}...", end=" ")
                thr_tag = f"{threshold}"

                out_json = eval_dir / f"coco_instances_{test_name}_thr{thr_tag}.json"
                out_metrics = eval_dir / f"metrics_{test_name}_thr{thr_tag}.json"
                out_test = eval_dir / f"eval_output_{test_name}_thr{thr_tag}.txt"
                out_pth = eval_dir / f"instances_predictions_{test_name}_thr{thr_tag}.pth"
                cfg_yaml = eval_dir / f"config_inference_{test_name}_thr{thr_tag}.yaml"

                base_row = {
                    "model_name": model_info["name"],
                    "model_family": model_family,
                    "backbone": backbone,
                    "subset": model_info["subset"],
                    "test_set": test_name,
                    "threshold": threshold,
                    "pred_json_path": str(out_json),
                    "metrics_json_path": str(out_metrics),
                    "eval_txt_path": str(out_test),
                    "config_yaml_path": str(cfg_yaml),
                    "timestamp_utc": dt.datetime.utcnow().isoformat(),
                }

                if out_json.exists() and out_metrics.exists() and not overwrite:
                    try:
                        with open(out_metrics, "r", encoding="utf-8") as f:
                            results = json.load(f)
                        metrics = extract_metrics_from_results(results, dets_per_image)

                        if metrics.get(f"AR{dets_per_image}") is None and out_test.exists():
                            eval_text = out_test.read_text(encoding="utf-8")
                            ar_val = extract_ar_from_eval_text(eval_text, dets_per_image)
                            if ar_val is not None:
                                metrics[f"AR{dets_per_image}"] = ar_val

                        row = {
                            **base_row,
                            **metrics,
                            "status": "ok",
                            "error_message": "",
                        }
                        csv_rows.append(row)
                        print("[OK] (cached)")
                    except Exception:
                        print("[WARN] (cache read error)")
                    continue

                tmp_dir = eval_dir / f"tmp_{test_name}_thr{str(threshold).replace('.', 'p')}"
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
                tmp_dir.mkdir(parents=True, exist_ok=True)

                try:
                    cfg = build_eval_cfg(
                        model_cfg=model_cfg,
                        weights_path=model_path,
                        dataset_name=dataset_name,
                        num_classes=3,
                        model_family=model_family,
                        threshold=threshold,
                        dets_per_image=dets_per_image,
                        num_workers=num_workers,
                    )

                    with open(cfg_yaml, "w", encoding="utf-8") as f:
                        f.write(cfg.dump())

                    predictor = DefaultPredictor(cfg)
                    evaluator = COCOEvaluator(dataset_name, output_dir=str(tmp_dir), max_dets_per_image=dets_per_image)
                    val_loader = build_detection_test_loader(cfg, dataset_name)

                    buf = io.StringIO()
                    with redirect_stdout(buf):
                        results = inference_on_dataset(predictor.model, val_loader, evaluator)
                    eval_text = buf.getvalue()

                    coco_src = tmp_dir / "coco_instances_results.json"
                    metrics_src = tmp_dir / "metrics.json"
                    pth_src = tmp_dir / "instances_predictions.pth"

                    if coco_src.exists():
                        shutil.move(str(coco_src), str(out_json))
                    if metrics_src.exists():
                        shutil.move(str(metrics_src), str(out_metrics))
                    else:
                        with open(out_metrics, "w", encoding="utf-8") as f:
                            json.dump(results, f, indent=2)
                    if pth_src.exists():
                        shutil.move(str(pth_src), str(out_pth))

                    with open(out_test, "w", encoding="utf-8") as f:
                        f.write(eval_text)

                    metrics = extract_metrics_from_results(results, dets_per_image)
                    ar_key = f"AR{dets_per_image}"
                    if metrics.get(ar_key) is None:
                        parsed_ar = extract_ar_from_eval_text(eval_text, dets_per_image)
                        if parsed_ar is not None:
                            metrics[ar_key] = parsed_ar
                            metrics_data = results.copy() if isinstance(results, dict) else {}
                            if "bbox" not in metrics_data:
                                metrics_data["bbox"] = {}
                            metrics_data["bbox"][f"AR@{dets_per_image}"] = parsed_ar
                            with open(out_metrics, "w", encoding="utf-8") as f:
                                json.dump(metrics_data, f, indent=2)

                    row = {
                        **base_row,
                        **metrics,
                        "status": "ok",
                        "error_message": "",
                    }
                    csv_rows.append(row)
                    per_model_summary.append(f"\n=== {test_name} | threshold {thr_tag} ===\n{eval_text}")

                    print(f"[OK] AP={metrics.get('AP', 0):.2f} {ar_key}={metrics.get(ar_key, 0):.2f}")
                except Exception as exc:
                    print(f"[FAIL] Error: {exc}")
                    traceback.print_exc()
                    row = {
                        **base_row,
                        "status": "fail",
                        "error_message": str(exc),
                    }
                    csv_rows.append(row)
                finally:
                    if tmp_dir.exists():
                        shutil.rmtree(tmp_dir)

        if per_model_summary:
            summary_path = eval_dir / "summary_all_tests.txt"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("\n".join(per_model_summary))
            print(f"  [OK] Summary: {summary_path}")

    print("=" * 80)
    print(f"Evaluation completed with {len(csv_rows)} result rows")
    print("=" * 80)
    return csv_rows


def reconstruct_from_existing(
    *,
    runs_root: Path,
    eval_dir_glob: str,
    dets_per_image: int,
) -> list[dict[str, Any]]:
    print(f"Reconstructing from existing eval dirs under {runs_root}")
    eval_dirs = sorted(runs_root.rglob(eval_dir_glob))
    print(f"Found {len(eval_dirs)} evaluation directories")

    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"metrics_(.+)_thr([\d.]+)\.json")

    for eval_dir in eval_dirs:
        model_dir = eval_dir.parent
        model_name = model_dir.name
        name_lower = model_name.lower()
        family, backbone = detect_model_family_backbone(name_lower)
        subset = detect_training_subset(name_lower)

        metrics_files = sorted(eval_dir.glob("metrics_*.json"))
        print(f"  {model_name}: {len(metrics_files)} metric files")

        for metrics_file in metrics_files:
            m = pattern.fullmatch(metrics_file.name)
            if not m:
                continue
            test_set = m.group(1)
            threshold = float(m.group(2))
            thr_tag = f"{threshold}"

            out_json = eval_dir / f"coco_instances_{test_set}_thr{thr_tag}.json"
            out_test = eval_dir / f"eval_output_{test_set}_thr{thr_tag}.txt"
            cfg_yaml = eval_dir / f"config_inference_{test_set}_thr{thr_tag}.yaml"

            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
                metrics = extract_metrics_from_results(results, dets_per_image)

                ar_key = f"AR{dets_per_image}"
                if metrics.get(ar_key) is None and out_test.exists():
                    eval_text = out_test.read_text(encoding="utf-8")
                    parsed_ar = extract_ar_from_eval_text(eval_text, dets_per_image)
                    if parsed_ar is not None:
                        metrics[ar_key] = parsed_ar

                row = {
                    "model_name": model_name,
                    "model_family": family,
                    "backbone": backbone,
                    "subset": subset,
                    "test_set": test_set,
                    "threshold": threshold,
                    **metrics,
                    "pred_json_path": str(out_json),
                    "metrics_json_path": str(metrics_file),
                    "eval_txt_path": str(out_test),
                    "config_yaml_path": str(cfg_yaml),
                    "status": "ok",
                    "error_message": "",
                    "timestamp_utc": dt.datetime.utcfromtimestamp(metrics_file.stat().st_mtime).isoformat(),
                }
                rows.append(row)
            except Exception as exc:
                rows.append(
                    {
                        "model_name": model_name,
                        "model_family": family,
                        "backbone": backbone,
                        "subset": subset,
                        "test_set": test_set,
                        "threshold": threshold,
                        "status": "fail",
                        "error_message": str(exc),
                        "timestamp_utc": dt.datetime.utcnow().isoformat(),
                    }
                )

    print(f"[OK] Reconstructed {len(rows)} rows")
    return rows


def export_results_csv(
    *,
    rows: list[dict[str, Any]],
    out_dir: Path,
    dets_per_image: int,
) -> tuple[Path, Path | None]:
    csv_output_dir = out_dir / "csv_reports"
    csv_output_dir.mkdir(parents=True, exist_ok=True)

    key_to_row: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for row in rows:
        key = (row.get("model_name"), row.get("test_set"), row.get("threshold"))
        key_to_row[key] = row
    deduplicated_rows = list(key_to_row.values())

    ar_key = f"AR{dets_per_image}"
    fieldnames = [
        "model_name",
        "model_family",
        "backbone",
        "subset",
        "test_set",
        "threshold",
        "AP",
        "AP50",
        "AP75",
        "APs",
        "APm",
        "APl",
        ar_key,
        "AP_E_coli",
        "AP_P_aeruginosa",
        "AP_S_aureus",
        "pred_json_path",
        "metrics_json_path",
        "eval_txt_path",
        "config_yaml_path",
        "status",
        "error_message",
        "timestamp_utc",
    ]

    global_csv_path = csv_output_dir / f"{dets_per_image}_stress_test_results_all.csv"
    with open(global_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(deduplicated_rows)

    models_exported: set[str] = set()
    for row in deduplicated_rows:
        model_name = str(row.get("model_name", "")).strip()
        if not model_name or model_name in models_exported:
            continue
        model_rows = [r for r in deduplicated_rows if r.get("model_name") == model_name]
        model_csv_path = csv_output_dir / f"{model_name}_stress_test_long.csv"
        with open(model_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(model_rows)
        models_exported.add(model_name)

    wide_csv_path: Path | None = None
    try:
        import pandas as pd

        if deduplicated_rows:
            df = pd.DataFrame(deduplicated_rows)
            wide_rows: list[dict[str, Any]] = []
            thr_suffixes = {0.0: "t0", 0.25: "t025", 0.5: "t05", 0.75: "t075"}
            metric_keys = [
                "AP",
                "AP50",
                "AP75",
                "APs",
                "APm",
                "APl",
                ar_key,
                "AP_E_coli",
                "AP_P_aeruginosa",
                "AP_S_aureus",
            ]

            for (model_name, test_set), group in df.groupby(["model_name", "test_set"]):
                wide_row = {
                    "model_name": model_name,
                    "model_family": group.iloc[0].get("model_family"),
                    "backbone": group.iloc[0].get("backbone"),
                    "subset": group.iloc[0].get("subset"),
                    "test_set": test_set,
                }
                for _, crow in group.iterrows():
                    thr = crow.get("threshold")
                    suffix = thr_suffixes.get(thr, f"t{str(thr).replace('.', 'p')}")
                    for metric in metric_keys:
                        if metric in crow:
                            wide_row[f"{metric}_{suffix}"] = crow.get(metric)
                wide_rows.append(wide_row)

            if wide_rows:
                wide_csv_path = csv_output_dir / "stress_test_results_wide.csv"
                pd.DataFrame(wide_rows).to_csv(wide_csv_path, index=False)
    except Exception as exc:
        print(f"[warn] Wide CSV export skipped: {exc}")

    print(f"[OK] Global long CSV: {global_csv_path}")
    if wide_csv_path:
        print(f"[OK] Wide CSV: {wide_csv_path}")
    print(f"[OK] Rows exported: {len(deduplicated_rows)}")

    return global_csv_path, wide_csv_path


def save_run_manifest(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    stress_test_paths: dict[str, Path] | None,
    selected_models: list[dict[str, Any]] | None,
) -> None:
    manifest = {
        "generated_at": dt.datetime.utcnow().isoformat(),
        "mode": args.mode,
        "runs_root": str(args.runs_root),
        "out_dir": str(out_dir),
        "thresholds": args.thresholds,
        "dets_per_image": args.dets_per_image,
        "overwrite": int(args.overwrite),
        "bins": args.bins,
        "evaluate_all": int(args.evaluate_all),
        "selected_indices": args.selected_indices or "",
        "filter_family": args.filter_family or "",
        "filter_backbone": args.filter_backbone or "",
        "filter_subset": args.filter_subset or "",
        "run_name_contains": args.run_name_contains or "",
        "resume": int(args.resume),
        "resume_model_index": args.resume_model_index,
        "resume_test_set": args.resume_test_set,
        "resume_threshold": args.resume_threshold,
        "stress_test_paths": {k: str(v) for k, v in (stress_test_paths or {}).items()},
        "selected_models": [
            {
                "index": m.get("index"),
                "name": m.get("name"),
                "path": str(m.get("path")),
                "family": m.get("family"),
                "backbone": m.get("backbone"),
                "subset": m.get("subset"),
            }
            for m in (selected_models or [])
        ],
    }

    with open(out_dir / "stress_test_run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    args = parse_args()

    args.runs_root = require_dir(args.runs_root.resolve(), "runs root")
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = parse_float_list(args.thresholds)
    bins = parse_bins(args.bins)

    if args.mode == "run":
        if args.full_coco_json is None or args.image_dir is None:
            print("ERROR: run mode requires --full-coco-json and --image-dir")
            sys.exit(1)
        if args.train_coco_json is None or args.val_coco_json is None:
            print("ERROR: run mode requires --train-coco-json and --val-coco-json")
            sys.exit(1)

        args.full_coco_json = require_file(args.full_coco_json.resolve(), "full coco json")
        args.image_dir = require_dir(args.image_dir.resolve(), "image dir")
        args.train_coco_json = require_file(args.train_coco_json.resolve(), "train coco json")
        args.val_coco_json = require_file(args.val_coco_json.resolve(), "val coco json")

        extended_coco, _ = generate_extended_cohort(
            full_coco_json=args.full_coco_json,
            out_dir=args.out_dir,
            overwrite=args.overwrite,
        )

        validate_category_format(extended_coco, args.train_coco_json)

        stress_test_paths, ann_counts, all_gt100_images = build_stress_subsets(
            extended_coco=extended_coco,
            bins=bins,
            out_dir=args.out_dir,
        )

        write_density_stats(
            full_coco_json=args.full_coco_json,
            extended_coco=extended_coco,
            ann_counts=ann_counts,
            bins=bins,
            all_gt100_images=all_gt100_images,
            out_dir=args.out_dir,
        )

        run_leakage_check(
            train_coco_json=args.train_coco_json,
            val_coco_json=args.val_coco_json,
            stress_test_paths=stress_test_paths,
            out_dir=args.out_dir,
        )

        available_models = discover_models(args.runs_root)
        selected_models = select_models(available_models, args)

        if not selected_models:
            print("No models selected. Exiting run mode.")
            save_run_manifest(
                out_dir=args.out_dir,
                args=args,
                stress_test_paths=stress_test_paths,
                selected_models=selected_models,
            )
            return

        load_runtime_dependencies()
        if PILLOW_LINEAR_SHIM_APPLIED and args.num_workers > 0:
            print(
                "[warn] Pillow compatibility shim active; forcing --num-workers 0 "
                "to avoid worker import failures."
            )
            args.num_workers = 0

        env_info = {
            "python_version": "unknown",
            "torch_version": "unknown",
            "detectron2_version": "unknown",
            "cuda_version": "unknown",
            "cudnn_version": "unknown",
            "device": "unknown",
        }
        if not args.skip_env_capture:
            env_info = capture_environment_info(args.out_dir)

        rows = evaluate_selected_models(
            selected_models=selected_models,
            stress_test_paths=stress_test_paths,
            image_dir=args.image_dir,
            thresholds=thresholds,
            dets_per_image=args.dets_per_image,
            out_dir=args.out_dir,
            overwrite=args.overwrite,
            num_workers=args.num_workers,
            resume=args.resume,
            resume_model_index=args.resume_model_index,
            resume_test_set=args.resume_test_set,
            resume_threshold=args.resume_threshold,
            env_info=env_info,
        )

        export_results_csv(rows=rows, out_dir=args.out_dir, dets_per_image=args.dets_per_image)
        save_run_manifest(
            out_dir=args.out_dir,
            args=args,
            stress_test_paths=stress_test_paths,
            selected_models=selected_models,
        )

    else:
        rows = reconstruct_from_existing(
            runs_root=args.runs_root,
            eval_dir_glob=args.eval_dir_glob,
            dets_per_image=args.dets_per_image,
        )
        export_results_csv(rows=rows, out_dir=args.out_dir, dets_per_image=args.dets_per_image)
        save_run_manifest(
            out_dir=args.out_dir,
            args=args,
            stress_test_paths=None,
            selected_models=None,
        )


if __name__ == "__main__":
    main()
