#!/usr/bin/env python3
"""
evaluate_detectron2_outputs.py
==============================
Detectron2 output evaluation workflow.

Supports:
1) Automated evaluation of all checkpoints under a runs root at multiple thresholds.
2) Cross-subset evaluation for one selected checkpoint at one threshold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import traceback
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AGAR_SUBSETS = ("total", "bright", "dark", "vague", "lowres")
DEFAULT_THRESHOLDS = (0.0, 0.25, 0.5, 0.75)
AGAR_NUM_CLASSES = 3
CURATED_NUM_CLASSES = 4
MODEL_FAMILIES = ("faster_rcnn", "retinanet")
BACKBONES = ("R_50", "R_101")

MODEL_ZOO_BY_TYPE = {
    ("faster_rcnn", "R_50"): "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    ("faster_rcnn", "R_101"): "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
    ("retinanet", "R_50"): "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
    ("retinanet", "R_101"): "COCO-Detection/retinanet_R_101_FPN_3x.yaml",
}


# Runtime imports are loaded lazily to make --help work without detectron2 installed.
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
        print("ERROR: torch + detectron2 are required for evaluation.")
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
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        raise ValueError("empty float list")
    return [float(x) for x in items]


def parse_subset_list(value: str) -> list[str]:
    items = [x.strip().lower() for x in value.split(",") if x.strip()]
    if not items:
        raise ValueError("empty subset list")
    return items


def resolve_filter_selection(filter_value: str) -> set[str]:
    """
    Resolve user filter input into an explicit subset set.

    Accepted values:
    - one subset: total|bright|dark|vague|lowres|curated
    - group alias: agar
    - all subsets: all
    - custom list: bright,dark,vague
    """
    known = set(AGAR_SUBSETS) | {"curated"}
    value = filter_value.strip().lower()

    if value == "all":
        return set(known)
    if value == "agar":
        return set(AGAR_SUBSETS)

    parts = [x.strip().lower() for x in value.split(",") if x.strip()]
    if not parts:
        raise ValueError("empty filter value")

    unknown = [p for p in parts if p not in known]
    if unknown:
        raise ValueError(
            f"unknown subset filter(s): {', '.join(unknown)}. "
            f"Allowed: {', '.join(sorted(known))}, agar, all"
        )
    return set(parts)


def resolve_model_family_filter(value: str) -> set[str]:
    token = value.strip().lower()
    if not token:
        raise ValueError("empty model family filter")
    if token == "all":
        return set(MODEL_FAMILIES)

    alias = {
        "faster": "faster_rcnn",
        "faster_rcnn": "faster_rcnn",
        "retina": "retinanet",
        "retinanet": "retinanet",
    }
    out: set[str] = set()
    unknown: list[str] = []
    for part in [x.strip().lower() for x in token.split(",") if x.strip()]:
        mapped = alias.get(part)
        if mapped is None:
            unknown.append(part)
        else:
            out.add(mapped)
    if unknown:
        raise ValueError(
            f"unknown model family filter(s): {', '.join(unknown)}. "
            "Allowed: faster, retina, all"
        )
    if not out:
        raise ValueError("empty model family filter")
    return out


def resolve_backbone_filter(value: str) -> set[str]:
    token = value.strip().lower()
    if not token:
        raise ValueError("empty backbone filter")
    if token == "all":
        return set(BACKBONES)

    alias = {
        "50": "R_50",
        "r50": "R_50",
        "r_50": "R_50",
        "101": "R_101",
        "r101": "R_101",
        "r_101": "R_101",
    }
    out: set[str] = set()
    unknown: list[str] = []
    for part in [x.strip().lower() for x in token.split(",") if x.strip()]:
        mapped = alias.get(part)
        if mapped is None:
            unknown.append(part)
        else:
            out.add(mapped)
    if unknown:
        raise ValueError(
            f"unknown backbone filter(s): {', '.join(unknown)}. "
            "Allowed: 50, 101, all"
        )
    if not out:
        raise ValueError("empty backbone filter")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Detectron2 checkpoints from training outputs.",
    )
    parser.add_argument(
        "--mode",
        choices=("evaluate-runs", "cross-subset"),
        default="evaluate-runs",
        help="Evaluation mode.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        required=True,
        help="Root directory containing model run folders with model_final.pth.",
    )
    parser.add_argument(
        "--repro-splits",
        type=Path,
        required=True,
        help="Directory containing reproduced split JSON files.",
    )
    parser.add_argument(
        "--agar-images-dir",
        type=Path,
        required=True,
        help="AGAR image root used for AGAR subset evaluation.",
    )
    parser.add_argument(
        "--curated-test-json",
        type=Path,
        help="Curated test COCO JSON (required when curated models are evaluated).",
    )
    parser.add_argument(
        "--curated-images-dir",
        type=Path,
        help="Curated image root (required when curated models are evaluated).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("eval_reports"),
        help="Output directory for global CSV and environment info.",
    )
    parser.add_argument(
        "--thresholds",
        default="0.0,0.25,0.5,0.75",
        help="Comma-separated score thresholds for evaluate-runs mode.",
    )
    parser.add_argument(
        "--filter-subset",
        help=(
            "Required for evaluate-runs mode. "
            "Accepts one subset (e.g. bright), a group alias (agar), "
            "all subsets (all), or a comma list (e.g. bright,dark,vague)."
        ),
    )
    parser.add_argument(
        "--filter-model-family",
        default="all",
        help=(
            "Model family filter for evaluate-runs mode. "
            "Accepts faster, retina, all, or a comma list (e.g. faster,retina)."
        ),
    )
    parser.add_argument(
        "--filter-backbone",
        default="all",
        help=(
            "Backbone filter for evaluate-runs mode. "
            "Accepts 50, 101, all, or a comma list (e.g. 50,101)."
        ),
    )
    parser.add_argument(
        "--csv-filename",
        help="Optional global CSV filename override.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing evaluation outputs.",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Use fixed eval_v2 folder name instead of eval_v2_<timestamp>.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--skip-env-capture",
        action="store_true",
        help="Skip env_versions.json and requirements_freeze.txt generation.",
    )
    parser.add_argument(
        "--force-workers-zero-with-pillow-shim",
        action="store_true",
        help=(
            "Force --num-workers 0 when the Pillow compatibility shim is active. "
            "By default, the script keeps the user-defined --num-workers value."
        ),
    )

    # Cross-subset mode args
    parser.add_argument(
        "--target-model",
        help="Model folder name fragment (or full checkpoint path) for cross-subset mode.",
    )
    parser.add_argument(
        "--cross-threshold",
        type=float,
        default=0.5,
        help="Single threshold for cross-subset mode.",
    )
    parser.add_argument(
        "--cross-subsets",
        default="bright,dark,vague,lowres",
        help="Comma-separated AGAR subsets for cross-subset mode.",
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


def require_dir(path: Path, label: str) -> Path:
    if not path.exists():
        print(f"ERROR: {label} not found: {path}")
        sys.exit(1)
    if not path.is_dir():
        print(f"ERROR: {label} is not a directory: {path}")
        sys.exit(1)
    return path


def build_agar_test_json_map(repro_splits: Path) -> dict[str, Path]:
    def resolve_subset_test_json(subset: str) -> Path:
        candidates = [
            repro_splits / f"{subset}_test_coco.json",
            repro_splits / f"{subset}_test.json",
            repro_splits / f"test_{subset}100.json",
            repro_splits / f"test_annotated_{subset}100.json",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate
        print(f"ERROR: test split json ({subset}) not found. Tried:")
        for candidate in candidates:
            print(f"  - {candidate}")
        sys.exit(1)

    mapping: dict[str, Path] = {}
    for subset in AGAR_SUBSETS:
        mapping[subset] = resolve_subset_test_json(subset)
    return mapping


def capture_environment_info(report_dir: Path) -> dict[str, Any]:
    py_version = "unknown"
    torch_version = "unknown"
    d2_version = "unknown"
    cuda_version = "unknown"
    cudnn_version = "unknown"
    device = "unknown"
    try:
        import platform

        py_version = platform.python_version()
        torch_version = torch.__version__
        d2_version = detectron2.__version__
        cuda_version = torch.version.cuda if hasattr(torch.version, "cuda") else "N/A"
        cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.enabled else "N/A"
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception as exc:
        print(f"[warn] Could not capture full environment: {exc}")

    env_info = {
        "python_version": py_version,
        "torch_version": torch_version,
        "detectron2_version": d2_version,
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "device": device,
    }

    with open(report_dir / "env_versions.json", "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)

    try:
        freeze_output = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        with open(report_dir / "requirements_freeze.txt", "w", encoding="utf-8") as f:
            f.write(freeze_output)
    except Exception as exc:
        print(f"[warn] Could not capture pip freeze: {exc}")

    print(
        f"[env] Python {py_version} | Torch {torch_version} | "
        f"Detectron2 {d2_version} | CUDA {cuda_version} | Device {device}"
    )
    return env_info


def detect_subset(weights_path: Path, known_subsets: list[str]) -> str | None:
    folder_name = weights_path.parent.name.lower()
    if "curated" in folder_name or "final" in folder_name:
        return "curated"
    for subset in known_subsets:
        if subset in folder_name:
            return subset
    return None


def detect_model_type(weights_path: Path) -> tuple[str | None, str | None]:
    name = weights_path.as_posix().lower()
    if "faster" in name:
        family = "faster_rcnn"
    elif "retina" in name:
        family = "retinanet"
    else:
        family = None

    if "r_101" in name:
        backbone = "R_101"
    elif "r_50" in name:
        backbone = "R_50"
    else:
        backbone = None
    return family, backbone


TRANSFER_SOURCE_SUBSETS = {
    "total",
    "bright",
    "dark",
    "vague",
    "lowres",
    "highres",
    "curated",
}


def detect_transfer_source(run_dir: Path) -> str:
    name = run_dir.name.lower()
    if "transfer" not in name:
        return ""

    tokens = [t for t in re.split(r"[^a-z0-9]+", name) if t]
    for idx, tok in enumerate(tokens):
        if tok.startswith("transfer"):
            for nxt in tokens[idx + 1 :]:
                if nxt in TRANSFER_SOURCE_SUBSETS:
                    return nxt
            return "total"
    return "total"


def ensure_dataset(dataset_name: str, json_path: Path, images_dir: Path) -> None:
    if dataset_name in DatasetCatalog.list():
        return
    register_coco_instances(dataset_name, {}, str(json_path), str(images_dir))


def build_cfg(
    model_cfg: str,
    weights_path: Path,
    dataset_name: str,
    num_classes: int,
    model_family: str,
    num_workers: int,
) -> Any:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_cfg))
    cfg.DATASETS.TEST = (dataset_name,)
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if model_family == "faster_rcnn":
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    else:
        cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
    return cfg


def extract_metrics_from_results(results: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if "bbox" in results:
        bbox = results["bbox"]
    else:
        bbox = results

    metrics["AP"] = bbox.get("AP")
    metrics["AP50"] = bbox.get("AP50")
    metrics["AP75"] = bbox.get("AP75")
    metrics["APs"] = bbox.get("APs")
    metrics["APm"] = bbox.get("APm")
    metrics["APl"] = bbox.get("APl")
    metrics["AR100"] = bbox.get("AR100")

    for key, value in bbox.items():
        if key.startswith("AP-"):
            class_name = key[3:]
            # Preserve both historical key styles:
            # - AP_Ecoli / AP_Paeruginosa / AP_Saureus
            # - AP_E_coli / AP_P_aeruginosa / AP_S_aureus
            class_name_underscore = class_name.replace(" ", "_")
            safe_key_compact = f"AP_{class_name_underscore.replace('.', '')}"
            safe_key_split = f"AP_{class_name_underscore.replace('.', '_')}"
            metrics[safe_key_compact] = value
            metrics[safe_key_split] = value
    return metrics


def extract_ar100_from_eval_text(eval_text: str) -> float | None:
    pattern = (
        r"Average Recall\s+\(AR\)\s+@\[\s*IoU=0.50:0.95\s*\|\s*area=\s*all\s*\|"
        r"\s*maxDets=\s*100\s*\]\s*=\s*([\d.]+)"
    )
    match = re.search(pattern, eval_text)
    if not match:
        return None
    try:
        return float(match.group(1)) * 100.0
    except ValueError:
        return None


def generate_run_id(run_dir: Path) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    h = hashlib.md5(str(run_dir).encode("utf-8")).hexdigest()[:8]
    return f"{ts}_{h}"


def get_training_config_if_exists(model_dir: Path) -> Path | None:
    for config_name in ("config.yaml", "cfg.yaml", "config_train.yaml"):
        cfg_path = model_dir / config_name
        if cfg_path.exists():
            return cfg_path
    return None


def write_run_meta_json(
    eval_dir: Path,
    run_id: str,
    run_dir: Path,
    subset: str,
    model_family: str,
    backbone: str,
    weights_path: Path,
    img_root: Path,
    subset_info: dict[str, Any],
    num_classes: int,
    thresholds: list[float],
    env_info: dict[str, Any],
    overwrite: bool,
) -> None:
    meta_path = eval_dir / "run_meta.json"
    if meta_path.exists() and not overwrite:
        return

    dataset_tag = "curated" if subset == "curated" else "agar100"
    classes = None if subset == "curated" else ["E.coli", "P.aeruginosa", "S.aureus"]

    meta = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "eval_dir": str(eval_dir),
        "subset": subset,
        "dataset_tag": dataset_tag,
        "model_family": model_family,
        "backbone": backbone,
        "weights_path": str(weights_path),
        "img_root": str(img_root),
        "split_identifiers": subset_info,
        "classes": classes,
        "num_classes": num_classes,
        "thresholds": thresholds,
        "dets_per_image": 100,
        "versions": env_info,
        "eval_invocation": "evaluation/evaluate_detectron2_outputs.py",
        "notes": "",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def write_inference_config_yaml(cfg: Any, eval_dir: Path, thr: float) -> Path:
    out = eval_dir / f"config_inference_thr{thr}.yaml"
    with open(out, "w", encoding="utf-8") as f:
        f.write(cfg.dump())
    return out


def csv_fieldnames() -> list[str]:
    return [
        "subset",
        "dataset_tag",
        "model_family",
        "backbone",
        "transfer",
        "run_dir",
        "eval_dir",
        "threshold",
        "AP",
        "AP50",
        "AP75",
        "APs",
        "APm",
        "APl",
        "AR100",
        "AP_E_coli",
        "AP_P_aeruginosa",
        "AP_S_aureus",
        "pred_json_path",
        "metrics_json_path",
        "test_txt_path",
        "summary_txt_path",
        "config_yaml_path",
        "ci_ready",
        "status",
        "error_message",
        "timestamp_utc",
        "detectron2_version",
        "torch_version",
        "cuda_version",
    ]


def append_to_csv(
    csv_path: Path,
    row_dict: dict[str, Any],
    env_info: dict[str, Any],
) -> None:
    row = row_dict.copy()
    row["detectron2_version"] = env_info.get("detectron2_version", "unknown")
    row["torch_version"] = env_info.get("torch_version", "unknown")
    row["cuda_version"] = env_info.get("cuda_version", "unknown")

    fields = csv_fieldnames()
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def aggregate_subset_csvs(
    subset_csv_paths: dict[str, Path],
    aggregate_csv_path: Path,
) -> int:
    fields = csv_fieldnames()
    total_rows = 0
    aggregate_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(aggregate_csv_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fields, restval="")
        writer.writeheader()
        for subset in sorted(subset_csv_paths.keys()):
            subset_csv = subset_csv_paths[subset]
            if not subset_csv.exists() or subset_csv.stat().st_size == 0:
                continue
            with open(subset_csv, "r", newline="", encoding="utf-8") as in_f:
                reader = csv.DictReader(in_f)
                for row in reader:
                    writer.writerow({k: row.get(k, "") for k in fields})
                    total_rows += 1
    return total_rows


def write_per_run_csvs(eval_dir: Path, per_threshold_rows: list[dict[str, Any]], overwrite: bool) -> None:
    if not per_threshold_rows:
        return

    long_csv_path = eval_dir / "eval_long.csv"
    if overwrite or not long_csv_path.exists():
        fields = list(per_threshold_rows[0].keys())
        with open(long_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, restval="")
            writer.writeheader()
            writer.writerows(per_threshold_rows)

    wide_csv_path = eval_dir / "eval_wide.csv"
    if overwrite or not wide_csv_path.exists():
        wide_row: dict[str, Any] = {}
        common_keys = ["subset", "dataset_tag", "model_family", "backbone", "transfer", "run_dir", "eval_dir"]
        for key in common_keys:
            wide_row[key] = per_threshold_rows[0].get(key)

        metric_keys = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR100"]
        for key in per_threshold_rows[0]:
            if key.startswith("AP_") and key not in metric_keys:
                metric_keys.append(key)

        thr_suffixes = {0.0: "t0", 0.25: "t025", 0.5: "t05", 0.75: "t075"}
        for row in per_threshold_rows:
            thr = float(row.get("threshold"))
            suffix = thr_suffixes.get(thr, f"t{str(thr).replace('.', 'p')}")
            for metric in metric_keys:
                if metric in row:
                    wide_row[f"{metric}_{suffix}"] = row[metric]

        with open(wide_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(wide_row.keys()), restval="")
            writer.writeheader()
            writer.writerow(wide_row)


def evaluate_one_model(
    *,
    weights_path: Path,
    subset: str,
    gt_json: Path,
    images_dir: Path,
    model_family: str,
    backbone: str,
    eval_timestamp: str,
    env_info: dict[str, Any],
    subset_info: dict[str, Any],
    subset_csv_path: Path,
    thresholds: list[float],
    overwrite: bool,
    use_timestamp: bool,
    num_workers: int,
) -> None:
    model_cfg = MODEL_ZOO_BY_TYPE.get((model_family, backbone))
    if model_cfg is None:
        print(f"[skip] Unknown model type for {weights_path}")
        return

    model_dir = weights_path.parent
    transfer = detect_transfer_source(model_dir)
    eval_root = model_dir / (f"eval_v2_{eval_timestamp}" if use_timestamp else "eval_v2")
    eval_root.mkdir(parents=True, exist_ok=True)

    run_id = generate_run_id(model_dir)
    num_classes = CURATED_NUM_CLASSES if subset == "curated" else AGAR_NUM_CLASSES
    write_run_meta_json(
        eval_dir=eval_root,
        run_id=run_id,
        run_dir=model_dir,
        subset=subset,
        model_family=model_family,
        backbone=backbone,
        weights_path=weights_path,
        img_root=images_dir,
        subset_info=subset_info,
        num_classes=num_classes,
        thresholds=thresholds,
        env_info=env_info,
        overwrite=overwrite,
    )

    safe_model = re.sub(r"[^A-Za-z0-9_]+", "_", model_dir.name)
    suffix = abs(hash(str(model_dir))) % 100000
    dataset_name = f"{subset}_test_{safe_model}_{suffix}"
    ensure_dataset(dataset_name, gt_json, images_dir)

    summary_blocks: list[str] = []
    per_threshold_rows: list[dict[str, Any]] = []

    for thr in thresholds:
        thr_tag = f"{thr}"
        out_json = eval_root / f"coco_instances_results_thr{thr_tag}.json"
        out_metrics = eval_root / f"metrics_thr{thr_tag}.json"
        out_test = eval_root / f"test_thr{thr_tag}.txt"
        out_pth = eval_root / f"instances_predictions_thr{thr_tag}.pth"

        if not overwrite and out_json.exists() and out_metrics.exists() and out_test.exists():
            print(f"[skip] {model_dir.name} thr={thr_tag} already done in {eval_root.name}")
            continue

        tmp_dir = eval_root / f"tmp_thr_{str(thr).replace('.', 'p')}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            cfg = build_cfg(
                model_cfg=model_cfg,
                weights_path=weights_path,
                dataset_name=dataset_name,
                num_classes=num_classes,
                model_family=model_family,
                num_workers=num_workers,
            )
            if model_family == "faster_rcnn":
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thr
            else:
                cfg.MODEL.RETINANET.SCORE_THRESH_TEST = thr

            cfg_yaml = write_inference_config_yaml(cfg, eval_root, thr)

            print(f"[run] subset={subset} model={model_dir.name} thr={thr_tag} -> {eval_root.name}/")
            predictor = DefaultPredictor(cfg)
            evaluator = COCOEvaluator(dataset_name, output_dir=str(tmp_dir), max_dets_per_image=100)
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
            summary_blocks.append(f"\n=== threshold {thr_tag} ===\n{eval_text}")

            metrics = extract_metrics_from_results(results)
            ar100_parsed = extract_ar100_from_eval_text(eval_text)
            if ar100_parsed is not None:
                metrics["AR100"] = ar100_parsed
                metrics_data = results.copy() if isinstance(results, dict) else {}
                if "bbox" not in metrics_data:
                    metrics_data["bbox"] = {}
                metrics_data["bbox"]["AR100"] = ar100_parsed
                with open(out_metrics, "w", encoding="utf-8") as f:
                    json.dump(metrics_data, f, indent=2)

            csv_row: dict[str, Any] = {
                "subset": subset,
                "dataset_tag": "curated" if subset == "curated" else "agar100",
                "model_family": model_family,
                "backbone": backbone,
                "transfer": transfer,
                "run_dir": str(model_dir),
                "eval_dir": str(eval_root),
                "threshold": thr,
                "AP": metrics.get("AP"),
                "AP50": metrics.get("AP50"),
                "AP75": metrics.get("AP75"),
                "APs": metrics.get("APs"),
                "APm": metrics.get("APm"),
                "APl": metrics.get("APl"),
                "AR100": metrics.get("AR100"),
                "AP_E_coli": metrics.get("AP_E_coli") or metrics.get("AP_Ecoli"),
                "AP_P_aeruginosa": metrics.get("AP_P_aeruginosa") or metrics.get("AP_Paeruginosa"),
                "AP_S_aureus": metrics.get("AP_S_aureus") or metrics.get("AP_Saureus"),
                "pred_json_path": str(out_json),
                "metrics_json_path": str(out_metrics),
                "test_txt_path": str(out_test),
                "summary_txt_path": str(eval_root / "summary_all_thresholds.txt"),
                "config_yaml_path": str(cfg_yaml),
                "ci_ready": 1 if out_json.exists() else 0,
                "status": "ok",
                "error_message": "",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            per_threshold_rows.append(csv_row)
            append_to_csv(subset_csv_path, csv_row, env_info)
        except Exception as exc:
            print(f"[error] Threshold {thr_tag} failed: {exc}")
            traceback.print_exc()
            fail_row = {
                "subset": subset,
                "dataset_tag": "curated" if subset == "curated" else "agar100",
                "model_family": model_family,
                "backbone": backbone,
                "transfer": transfer,
                "run_dir": str(model_dir),
                "eval_dir": str(eval_root),
                "threshold": thr,
                "status": "fail",
                "error_message": str(exc),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            append_to_csv(subset_csv_path, fail_row, env_info)
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)

    if summary_blocks:
        summary_path = eval_root / "summary_all_thresholds.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_blocks))
        print(f"[done] {model_dir.name} -> {summary_path}")

    train_cfg = get_training_config_if_exists(model_dir)
    if train_cfg:
        try:
            shutil.copy(str(train_cfg), str(eval_root / "config_train_original.yaml"))
        except Exception as exc:
            print(f"[warn] Could not copy training config: {exc}")

    write_per_run_csvs(eval_root, per_threshold_rows, overwrite=overwrite)


def run_evaluate_runs(args: argparse.Namespace, env_info: dict[str, Any]) -> None:
    thresholds = parse_float_list(args.thresholds)
    use_timestamp = not args.no_timestamp
    if not args.filter_subset:
        print("ERROR: --filter-subset is required for --mode evaluate-runs")
        sys.exit(1)
    try:
        selected_subsets = resolve_filter_selection(args.filter_subset)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    try:
        selected_families = resolve_model_family_filter(args.filter_model_family)
        selected_backbones = resolve_backbone_filter(args.filter_backbone)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    test_json_by_subset = build_agar_test_json_map(args.repro_splits)
    subset_csv_paths = {
        subset: args.report_dir / f"detectron2_eval_long_{subset}.csv"
        for subset in sorted(selected_subsets)
    }
    csv_filename = args.csv_filename if args.csv_filename else "detectron2_eval_long.csv"
    aggregate_csv_path = args.report_dir / csv_filename

    eval_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoints = sorted(args.runs_root.rglob("model_final.pth"))

    print(f"\n{'='*70}")
    print(f"Starting evaluation run: {eval_timestamp}")
    print(f"Filter input: {args.filter_subset}")
    print(f"Resolved subsets: {sorted(selected_subsets)}")
    print(f"Resolved model families: {sorted(selected_families)}")
    print(f"Resolved backbones: {sorted(selected_backbones)}")
    print(f"Subset CSVs dir: {args.report_dir}")
    print(f"Aggregate CSV: {aggregate_csv_path}")
    print(f"Found {len(checkpoints)} checkpoints under {args.runs_root}")
    print(f"{'='*70}\n")

    if "curated" in selected_subsets and (args.curated_test_json is None or args.curated_images_dir is None):
        print(
            "ERROR: curated is included in filter, so both "
            "--curated-test-json and --curated-images-dir are required."
        )
        sys.exit(1)

    for ckpt in checkpoints:
        subset = detect_subset(ckpt, list(test_json_by_subset.keys()))
        if subset is None:
            print(f"[skip] Cannot detect subset for {ckpt}")
            continue

        if subset not in selected_subsets:
            print(f"[skip] {ckpt.parent.name} (subset={subset}, filter={sorted(selected_subsets)})")
            continue

        if subset == "curated":
            if args.curated_test_json is None or args.curated_images_dir is None:
                print(
                    "[skip] Curated checkpoint found but --curated-test-json/--curated-images-dir "
                    "were not provided."
                )
                continue
            gt_json = require_file(args.curated_test_json, "curated test json")
            images_dir = require_dir(args.curated_images_dir, "curated images dir")
            subset_info = {"test_json": str(gt_json)}
        else:
            gt_json = test_json_by_subset[subset]
            images_dir = args.agar_images_dir
            subset_info = {
                "test_json": str(test_json_by_subset[subset]),
                "val_json": None,
                "train_json": None,
            }

        model_family, backbone = detect_model_type(ckpt)
        if model_family is None or backbone is None:
            print(f"[skip] Cannot detect model type for {ckpt}")
            continue
        if model_family not in selected_families or backbone not in selected_backbones:
            print(
                f"[skip] {ckpt.parent.name} (family={model_family}, backbone={backbone}, "
                f"family_filter={sorted(selected_families)}, backbone_filter={sorted(selected_backbones)})"
            )
            continue

        evaluate_one_model(
            weights_path=ckpt,
            subset=subset,
            gt_json=gt_json,
            images_dir=images_dir,
            model_family=model_family,
            backbone=backbone,
            eval_timestamp=eval_timestamp,
            env_info=env_info,
            subset_info=subset_info,
            subset_csv_path=subset_csv_paths[subset],
            thresholds=thresholds,
            overwrite=args.overwrite,
            use_timestamp=use_timestamp,
            num_workers=args.num_workers,
        )

    total_rows = aggregate_subset_csvs(subset_csv_paths, aggregate_csv_path)

    print(f"\n{'='*70}")
    print(f"Evaluation run completed: {eval_timestamp}")
    print(f"Subset CSVs: {', '.join(str(p) for p in subset_csv_paths.values())}")
    print(f"Aggregate CSV ({total_rows} rows): {aggregate_csv_path}")
    print(f"Env info: {args.report_dir / 'env_versions.json'}")
    print(f"{'='*70}")


def find_target_checkpoint(runs_root: Path, target_model: str) -> Path | None:
    candidate = Path(target_model)
    if candidate.exists() and candidate.is_file():
        return candidate
    checkpoints = sorted(runs_root.rglob("model_final.pth"))
    for ckpt in checkpoints:
        if target_model in str(ckpt):
            return ckpt
    return None


def run_cross_subset(args: argparse.Namespace) -> None:
    if not args.target_model:
        print("ERROR: --target-model is required for --mode cross-subset")
        sys.exit(1)

    test_json_by_subset = build_agar_test_json_map(args.repro_splits)
    cross_subsets = parse_subset_list(args.cross_subsets)
    for subset in cross_subsets:
        if subset not in test_json_by_subset:
            print(f"ERROR: unknown subset in --cross-subsets: {subset}")
            sys.exit(1)

    target_ckpt = find_target_checkpoint(args.runs_root, args.target_model)
    checkpoints = sorted(args.runs_root.rglob("model_final.pth"))

    print(f"\n{'='*70}")
    print("CROSS-SUBSET EVALUATION MODE")
    print(f"Model selector: {args.target_model}")
    print(f"Threshold: {args.cross_threshold}")
    print(f"Testing on subsets: {cross_subsets}")
    print(f"{'='*70}\n")

    if target_ckpt is None:
        print(f"Model not found: {args.target_model}")
        print("\nAvailable models:")
        for ckpt in checkpoints:
            print(f"  - {ckpt.parent.name}")
        return

    print(f"Found model: {target_ckpt.parent.name}\n")
    model_family, backbone = detect_model_type(target_ckpt)
    if model_family is None or backbone is None:
        print(f"ERROR: cannot detect model type for {target_ckpt}")
        return

    model_dir = target_ckpt.parent
    print(f"Model family: {model_family}")
    print(f"Backbone: {backbone}")
    transfer = detect_transfer_source(model_dir)

    model_cfg = MODEL_ZOO_BY_TYPE.get((model_family, backbone))
    if model_cfg is None:
        print(f"ERROR: unknown model config for {model_family}/{backbone}")
        return

    eval_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cross_eval_root = model_dir / f"cross_subset_eval_{eval_timestamp}"
    cross_eval_root.mkdir(parents=True, exist_ok=True)

    cross_csv_path = cross_eval_root / "detectron2_cross_subset_eval.csv"
    cross_rows: list[dict[str, Any]] = []
    summary_texts: list[str] = []

    print(f"\n{'='*70}")
    print(f"Evaluating on {len(cross_subsets)} subsets at threshold={args.cross_threshold}")
    print(f"{'='*70}\n")

    for subset in cross_subsets:
        print(f"[{subset}] Evaluating...")
        gt_json = test_json_by_subset[subset]
        images_dir = args.agar_images_dir
        subset_eval_dir = cross_eval_root / subset
        subset_eval_dir.mkdir(parents=True, exist_ok=True)

        safe_model = re.sub(r"[^A-Za-z0-9_]+", "_", model_dir.name)
        suffix = abs(hash(str(model_dir) + subset)) % 100000
        dataset_name = f"{subset}_cross_test_{safe_model}_{suffix}"
        ensure_dataset(dataset_name, gt_json, images_dir)

        cfg = build_cfg(
            model_cfg=model_cfg,
            weights_path=target_ckpt,
            dataset_name=dataset_name,
            num_classes=AGAR_NUM_CLASSES,
            model_family=model_family,
            num_workers=args.num_workers,
        )
        if model_family == "faster_rcnn":
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.cross_threshold
        else:
            cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.cross_threshold

        cfg_yaml = subset_eval_dir / f"config_inference_thr{args.cross_threshold}.yaml"
        with open(cfg_yaml, "w", encoding="utf-8") as f:
            f.write(cfg.dump())

        out_json = subset_eval_dir / "coco_instances_results.json"
        out_metrics = subset_eval_dir / "metrics.json"
        out_test = subset_eval_dir / "test_output.txt"
        out_pth = subset_eval_dir / "instances_predictions.pth"

        tmp_dir = subset_eval_dir / "tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            predictor = DefaultPredictor(cfg)
            evaluator = COCOEvaluator(dataset_name, output_dir=str(tmp_dir), max_dets_per_image=100)
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
            summary_texts.append(
                f"\n{'='*70}\n{subset.upper()} TEST SET (threshold={args.cross_threshold})\n"
                f"{'='*70}\n{eval_text}"
            )

            metrics = extract_metrics_from_results(results)
            ar100_parsed = extract_ar100_from_eval_text(eval_text)
            if ar100_parsed is not None:
                metrics["AR100"] = ar100_parsed
                metrics_data = results.copy() if isinstance(results, dict) else {}
                if "bbox" not in metrics_data:
                    metrics_data["bbox"] = {}
                metrics_data["bbox"]["AR100"] = ar100_parsed
                with open(out_metrics, "w", encoding="utf-8") as f:
                    json.dump(metrics_data, f, indent=2)

            row = {
                "subset": subset,
                "dataset_tag": "agar100",
                "model_name": model_dir.name,
                "model_family": model_family,
                "backbone": backbone,
                "transfer": transfer,
                "run_dir": str(model_dir),
                "eval_dir": str(subset_eval_dir),
                "threshold": args.cross_threshold,
                "AP": metrics.get("AP"),
                "AP50": metrics.get("AP50"),
                "AP75": metrics.get("AP75"),
                "APs": metrics.get("APs"),
                "APm": metrics.get("APm"),
                "APl": metrics.get("APl"),
                "AR100": metrics.get("AR100"),
                "AP_E_coli": metrics.get("AP_Ecoli") or metrics.get("AP_E_coli"),
                "AP_P_aeruginosa": metrics.get("AP_Paeruginosa") or metrics.get("AP_P_aeruginosa"),
                "AP_S_aureus": metrics.get("AP_Saureus") or metrics.get("AP_S_aureus"),
                "pred_json_path": str(out_json),
                "metrics_json_path": str(out_metrics),
                "test_txt_path": str(out_test),
                "summary_txt_path": str(cross_eval_root / "summary_all_subsets.txt"),
                "config_yaml_path": str(cfg_yaml),
                "ci_ready": 1 if out_json.exists() else 0,
                "status": "ok",
                "error_message": "",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            cross_rows.append(row)
            print(f"  AP={metrics.get('AP', 0):.2f} | AP50={metrics.get('AP50', 0):.2f}")
        except Exception as exc:
            print(f"  Error: {exc}")
            traceback.print_exc()
            cross_rows.append(
                {
                    "subset": subset,
                    "dataset_tag": "agar100",
                    "model_name": model_dir.name,
                    "model_family": model_family,
                    "backbone": backbone,
                    "transfer": transfer,
                    "run_dir": str(model_dir),
                    "eval_dir": str(subset_eval_dir),
                    "threshold": args.cross_threshold,
                    "status": "fail",
                    "error_message": str(exc),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)

    if summary_texts:
        summary_path = cross_eval_root / "summary_all_subsets.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_texts))
        print(f"\nSummary saved: {summary_path}")

    if cross_rows:
        fields = list(cross_rows[0].keys())
        with open(cross_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, restval="")
            writer.writeheader()
            writer.writerows(cross_rows)
        print(f"\n{'='*70}")
        print("Cross-subset evaluation complete")
        print(f"Results: {cross_csv_path}")
        print(f"Per-subset outputs: {cross_eval_root}")
        print(f"{'='*70}\n")


def main() -> None:
    args = parse_args()

    args.runs_root = require_dir(args.runs_root.resolve(), "runs root")
    args.repro_splits = require_dir(args.repro_splits.resolve(), "repro splits")
    args.agar_images_dir = require_dir(args.agar_images_dir.resolve(), "AGAR images dir")
    args.report_dir = args.report_dir.resolve()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    if args.curated_test_json is not None:
        args.curated_test_json = require_file(args.curated_test_json.resolve(), "curated test json")
    if args.curated_images_dir is not None:
        args.curated_images_dir = require_dir(args.curated_images_dir.resolve(), "curated images dir")

    load_runtime_dependencies()
    if PILLOW_LINEAR_SHIM_APPLIED and args.num_workers > 0:
        if args.force_workers_zero_with_pillow_shim:
            print(
                "[warn] Pillow compatibility shim active; forcing --num-workers 0 "
                "because --force-workers-zero-with-pillow-shim was set."
            )
            args.num_workers = 0
        else:
            print(
                "[warn] Pillow compatibility shim active; keeping user-defined "
                f"--num-workers {args.num_workers}. "
                "If worker imports fail, rerun with --force-workers-zero-with-pillow-shim "
                "or set --num-workers 0."
            )

    env_info = {
        "python_version": "unknown",
        "torch_version": "unknown",
        "detectron2_version": "unknown",
        "cuda_version": "unknown",
        "cudnn_version": "unknown",
        "device": "unknown",
    }
    if not args.skip_env_capture:
        env_info = capture_environment_info(args.report_dir)

    if args.mode == "evaluate-runs":
        run_evaluate_runs(args, env_info)
    else:
        run_cross_subset(args)


if __name__ == "__main__":
    main()
