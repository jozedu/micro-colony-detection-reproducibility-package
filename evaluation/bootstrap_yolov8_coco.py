#!/usr/bin/env python3
"""
bootstrap_yolov8_coco.py
========================
Bootstrap COCO metrics for YOLO evaluation outputs.

This script follows the YOLO bootstrap evaluation flow:
- AGAR bootstrap with image-resampling over prediction/GT overlap
- curated bootstrap with GT parent-class handling and image-id remapping
- fast in-memory COCOeval execution
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AGAR_SUBSETS = ("total", "bright", "dark", "vague", "lowres")
ALL_SUBSETS = AGAR_SUBSETS + ("curated",)
SUBSET_GROUPS = {
    "agar": list(AGAR_SUBSETS),
    "all": list(ALL_SUBSETS),
}

np = None


def load_runtime_dependencies() -> None:
    global np
    try:
        import numpy as _np
        from pycocotools.coco import COCO as _COCO  # noqa: F401
        from pycocotools.cocoeval import COCOeval as _COCOeval  # noqa: F401
    except ImportError as exc:
        print("ERROR: numpy and pycocotools are required.")
        print("Install with: python -m pip install numpy pycocotools")
        print(f"Import failure: {exc}")
        sys.exit(1)
    np = _np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap YOLO COCO metrics from evaluation CSV.")
    parser.add_argument(
        "--eval-csv",
        type=Path,
        required=True,
        help="Evaluation CSV produced by evaluation/evaluate_yolov8_coco.py.",
    )
    parser.add_argument(
        "--repro-splits",
        type=Path,
        required=True,
        help="Directory containing reproduced COCO split JSON files.",
    )
    parser.add_argument(
        "--curated-gt-json",
        type=Path,
        help="Curated COCO test JSON (required if curated subset is selected).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("bootstrap_reports"),
        help="Output directory for bootstrap reports.",
    )
    parser.add_argument(
        "--subset-filter",
        default="agar",
        help=(
            "Subset selection: one subset, group alias (agar), all, "
            "or comma list (e.g. bright,dark,curated)."
        ),
    )
    parser.add_argument(
        "--model-filter",
        default="all",
        help=(
            "Model selection by YOLO variant/run token: all, yolov8, yolov8m, "
            "or comma list (e.g. yolov8m,yolov8l)."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.001,
        help="Evaluate rows at this threshold (default: 0.001).",
    )
    parser.add_argument(
        "--status-filter",
        default="ok",
        help="CSV status filter (default: ok).",
    )
    parser.add_argument(
        "--run-name-contains",
        help="Optional substring filter applied to run directory name.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=500,
        help="Number of bootstrap replicates (default: 500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Bootstrap random seed (default: 12345).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files in out-dir.",
    )
    return parser.parse_args()


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


def resolve_subset_filter(value: str) -> set[str]:
    token = value.strip().lower()
    if not token:
        print("ERROR: empty --subset-filter")
        sys.exit(1)

    if token in SUBSET_GROUPS:
        return set(SUBSET_GROUPS[token])

    parts = [x.strip().lower() for x in token.split(",") if x.strip()]
    if not parts:
        print("ERROR: empty --subset-filter")
        sys.exit(1)

    bad = [p for p in parts if p not in ALL_SUBSETS]
    if bad:
        print(
            f"ERROR: invalid subset filter(s): {bad}. "
            f"Allowed: {', '.join(ALL_SUBSETS)}, agar, all"
        )
        sys.exit(1)
    return set(parts)


def normalize_model_token(token: str) -> str:
    t = token.strip().lower().replace("-", "_")
    t = re.sub(r"_+", "_", t)
    return t


def resolve_model_filter(value: str) -> list[str]:
    token = normalize_model_token(value)
    if not token:
        print("ERROR: empty --model-filter")
        sys.exit(1)
    if token == "all":
        return ["all"]
    return [normalize_model_token(x) for x in token.split(",") if x.strip()]


def detect_csv_delimiter(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
    return ";" if first.count(";") > first.count(",") else ","


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def read_eval_rows(csv_path: Path) -> list[dict[str, Any]]:
    delimiter = detect_csv_delimiter(csv_path)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        out: list[dict[str, Any]] = []
        for row in reader:
            out.append({str(k).strip(): v for k, v in row.items()})
    return out


def infer_model_key(row: dict[str, Any]) -> str:
    text = " ".join(
        [
            str(row.get("backbone", "")),
            str(row.get("model_family", "")),
            Path(str(row.get("run_dir", ""))).name,
        ]
    ).lower()
    m = re.search(r"(yolov\d+[nslmx])", text)
    if m:
        return m.group(1)
    m = re.search(r"(yolo\d+[nslmx])", text)
    if m:
        return m.group(1)
    m = re.search(r"(yolov\d+)", text)
    if m:
        return m.group(1)
    if "yolo" in text:
        return "yolo"
    return "unknown"


def model_matches(row: dict[str, Any], filters: list[str]) -> bool:
    if filters == ["all"]:
        return True
    key = normalize_model_token(str(row.get("_model_key", "")))
    run_name = normalize_model_token(Path(str(row.get("run_dir", ""))).name)
    backbone = normalize_model_token(str(row.get("backbone", "")))

    for token in filters:
        if token == "yolo":
            if key.startswith("yolo"):
                return True
        elif token == "yolov8":
            if key.startswith("yolov8"):
                return True
        elif token.endswith("*"):
            prefix = token[:-1]
            if key.startswith(prefix) or run_name.startswith(prefix) or backbone.startswith(prefix):
                return True
        elif token == key or token == run_name or token == backbone:
            return True
        elif token in run_name or token in backbone:
            return True
    return False


def build_subset_gt_map(repro_splits: Path, curated_gt: Path | None) -> dict[str, Path]:
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
        print(f"ERROR: GT json for {subset} not found. Tried:")
        for candidate in candidates:
            print(f"  - {candidate}")
        sys.exit(1)

    out: dict[str, Path] = {}
    for subset in AGAR_SUBSETS:
        out[subset] = resolve_subset_test_json(subset)
    if curated_gt is not None:
        out["curated"] = require_file(curated_gt, "curated GT json")
    return out


def _coco_from_dict(dataset_dict: dict[str, Any]) -> Any:
    from pycocotools.coco import COCO

    coco = COCO()
    coco.dataset = dataset_dict
    coco.createIndex()
    return coco


def _extract_metrics(coco_eval: Any, cat_ids: list[int], cat_names: list[str]) -> dict[str, float]:
    precision = coco_eval.eval["precision"]  # [T, R, K, A, M]
    results: dict[str, float] = {
        "AP": float(coco_eval.stats[0]) * 100.0,
        "AP50": float(coco_eval.stats[1]) * 100.0,
        "APs": float(coco_eval.stats[3]) * 100.0,
    }
    for k_idx, cat_name in enumerate(cat_names):
        p = precision[:, :, k_idx, 0, 2]
        valid = p[p > -1]
        ap_cat = float(np.mean(valid)) * 100.0 if len(valid) > 0 else 0.0
        results[f"AP_{cat_name}"] = ap_cat
    return results


def _run_one_replicate(
    boot_gt_dict: dict[str, Any],
    boot_preds: list[dict[str, Any]],
    cat_ids: list[int],
    cat_names: list[str],
) -> dict[str, float] | None:
    from pycocotools.cocoeval import COCOeval

    if not boot_preds:
        return None

    old_stdout = sys.stdout
    devnull = open(os.devnull, "w", encoding="utf-8")
    sys.stdout = devnull
    try:
        coco_gt = _coco_from_dict(boot_gt_dict)
        coco_dt = coco_gt.loadRes(boot_preds)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    finally:
        sys.stdout = old_stdout
        devnull.close()
    return _extract_metrics(ev, cat_ids, cat_names)


def _build_boot_sample(
    orig_img_ids: np.ndarray,
    coco_gt: Any,
    pred_by_image: dict[int, list[dict[str, Any]]],
    rng: np.random.Generator,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sampled_ids = rng.choice(orig_img_ids, size=len(orig_img_ids), replace=True)

    boot_images: list[dict[str, Any]] = []
    boot_anns: list[dict[str, Any]] = []
    boot_preds: list[dict[str, Any]] = []
    new_id = 1

    for oid_raw in sampled_ids:
        oid = int(oid_raw)
        img_info_list = coco_gt.loadImgs(oid)
        if not img_info_list:
            continue

        img = img_info_list[0].copy()
        img["id"] = new_id
        boot_images.append(img)

        ann_ids = coco_gt.getAnnIds(imgIds=oid)
        for ann in coco_gt.loadAnns(ann_ids):
            new_ann = ann.copy()
            new_ann["image_id"] = new_id
            boot_anns.append(new_ann)

        for pred in pred_by_image.get(oid, []):
            new_pred = pred.copy()
            new_pred["image_id"] = new_id
            boot_preds.append(new_pred)

        new_id += 1

    for i, ann in enumerate(boot_anns):
        ann["id"] = i + 1

    return {
        "images": boot_images,
        "annotations": boot_anns,
        "categories": coco_gt.dataset["categories"],
    }, boot_preds


def align_prediction_categories(pred_list: list[dict[str, Any]], valid_cat_ids: set[int]) -> tuple[list[dict[str, Any]], str]:
    if not pred_list:
        return pred_list, "none"

    present = {int(p["category_id"]) for p in pred_list if "category_id" in p}
    if present and present.issubset(valid_cat_ids):
        return pred_list, "none"

    shifted_minus = {cid - 1 for cid in present if cid > 0}
    if shifted_minus and shifted_minus.issubset(valid_cat_ids):
        for p in pred_list:
            if int(p["category_id"]) > 0:
                p["category_id"] = int(p["category_id"]) - 1
        return pred_list, "minus_1"

    shifted_plus = {cid + 1 for cid in present}
    if shifted_plus and shifted_plus.issubset(valid_cat_ids):
        for p in pred_list:
            p["category_id"] = int(p["category_id"]) + 1
        return pred_list, "plus_1"

    return pred_list, "none"


def bootstrap_coco_eval_agar_fast(
    gt_json_path: Path,
    pred_json_path: Path,
    n_boot: int,
    seed: int,
) -> tuple[dict[str, dict[str, float]] | None, str]:
    from pycocotools.coco import COCO

    old_stdout = sys.stdout
    devnull = open(os.devnull, "w", encoding="utf-8")
    sys.stdout = devnull
    try:
        coco_gt = COCO(str(gt_json_path))
    finally:
        sys.stdout = old_stdout
        devnull.close()

    with open(pred_json_path, "r", encoding="utf-8") as f:
        pred_list = json.load(f)
    if not pred_list:
        return None, "none"

    valid_cat_ids = set(coco_gt.getCatIds())
    pred_list, fix_mode = align_prediction_categories(pred_list, valid_cat_ids)

    gt_img_ids = set(coco_gt.getImgIds())
    pred_img_ids = {int(p["image_id"]) for p in pred_list if isinstance(p.get("image_id"), (int, float))}
    overlap = gt_img_ids & pred_img_ids
    if not overlap:
        return None, fix_mode

    cat_ids = coco_gt.getCatIds()
    cat_names = [c["name"].replace(" ", "_").replace(".", "") for c in coco_gt.loadCats(cat_ids)]

    pred_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in pred_list:
        if isinstance(p.get("image_id"), (int, float)):
            pred_by_image[int(p["image_id"])].append(p)

    rng = np.random.default_rng(seed)
    orig_img_ids = np.array(sorted(overlap))
    accum: dict[str, list[float]] = defaultdict(list)
    n_success = 0

    for i in range(n_boot):
        boot_gt_dict, boot_preds = _build_boot_sample(orig_img_ids, coco_gt, pred_by_image, rng)
        if not boot_gt_dict["images"] or not boot_preds:
            continue
        metrics = _run_one_replicate(boot_gt_dict, boot_preds, cat_ids, cat_names)
        if metrics is None:
            continue
        for k, v in metrics.items():
            accum[k].append(v)
        n_success += 1
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{n_boot} replicates done ({n_success} successful)", end="\r")
    print(f"    {n_boot}/{n_boot} replicates done ({n_success} successful)    ")

    if n_success == 0:
        return None, fix_mode

    return {
        metric: {
            "mean": float(np.mean(vals)),
            "ci_low": float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5)),
        }
        for metric, vals in accum.items()
    }, fix_mode


def bootstrap_coco_eval_curated_fast(
    gt_json_path: Path,
    pred_json_path: Path,
    n_boot: int,
    seed: int,
) -> tuple[dict[str, dict[str, float]] | None, str]:
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    cat_info = {int(c["id"]): str(c["name"]) for c in gt_data.get("categories", [])}
    if 0 in cat_info and cat_info[0].lower() in ("colonies", "colony"):
        gt_data["categories"] = [c for c in gt_data["categories"] if int(c["id"]) != 0]
        gt_data["annotations"] = [a for a in gt_data["annotations"] if int(a["category_id"]) != 0]

    with open(pred_json_path, "r", encoding="utf-8") as f:
        pred_list = json.load(f)
    if not pred_list:
        return None, "none"

    gt_img_ids = {int(img["id"]) for img in gt_data.get("images", [])}
    pred_img_ids = {
        int(p["image_id"]) for p in pred_list
        if isinstance(p.get("image_id"), (int, float))
    }
    overlap = gt_img_ids & pred_img_ids

    if len(overlap) < max(1, len(pred_img_ids) // 2):
        fname_to_id: dict[str, int] = {}
        for img in gt_data.get("images", []):
            fname = str(img.get("file_name", ""))
            iid = int(img["id"])
            fname_to_id[fname] = iid
            fname_to_id[Path(fname).stem] = iid

        remapped = 0
        for p in pred_list:
            iid = p.get("image_id")
            if iid in fname_to_id:
                p["image_id"] = fname_to_id[iid]
                remapped += 1
            elif isinstance(iid, str):
                stem = Path(iid).stem
                if stem in fname_to_id:
                    p["image_id"] = fname_to_id[stem]
                    remapped += 1
        if remapped > 0:
            pred_img_ids = {
                int(p["image_id"]) for p in pred_list
                if isinstance(p.get("image_id"), (int, float))
            }
            overlap = gt_img_ids & pred_img_ids

    if not overlap:
        return None, "none"

    coco_gt = _coco_from_dict(gt_data)
    valid_cat_ids = set(coco_gt.getCatIds())
    pred_list, fix_mode = align_prediction_categories(pred_list, valid_cat_ids)

    cat_ids = coco_gt.getCatIds()
    cat_names = [c["name"].replace(" ", "_").replace(".", "") for c in coco_gt.loadCats(cat_ids)]

    pred_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in pred_list:
        if isinstance(p.get("image_id"), (int, float)):
            pred_by_image[int(p["image_id"])].append(p)

    rng = np.random.default_rng(seed)
    orig_img_ids = np.array(sorted(overlap))
    accum: dict[str, list[float]] = defaultdict(list)
    n_success = 0

    for i in range(n_boot):
        boot_gt_dict, boot_preds = _build_boot_sample(orig_img_ids, coco_gt, pred_by_image, rng)
        if not boot_gt_dict["images"] or not boot_preds:
            continue
        metrics = _run_one_replicate(boot_gt_dict, boot_preds, cat_ids, cat_names)
        if metrics is None:
            continue
        for k, v in metrics.items():
            accum[k].append(v)
        n_success += 1
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{n_boot} replicates done ({n_success} successful)", end="\r")
    print(f"    {n_boot}/{n_boot} replicates done ({n_success} successful)    ")

    if n_success == 0:
        return None, fix_mode

    return {
        metric: {
            "mean": float(np.mean(vals)),
            "ci_low": float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5)),
        }
        for metric, vals in accum.items()
    }, fix_mode


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def to_wide_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        key = (
            str(r.get("subset")),
            str(r.get("run_dir")),
            str(r.get("model_key")),
        )
        out = grouped.setdefault(
            key,
            {
                "subset": r.get("subset"),
                "dataset_tag": r.get("dataset_tag"),
                "model_family": r.get("model_family"),
                "backbone": r.get("backbone"),
                "model_key": r.get("model_key"),
                "run_dir": r.get("run_dir"),
                "eval_dir": r.get("eval_dir"),
                "threshold": r.get("threshold"),
                "n_boot": r.get("n_boot"),
                "seed": r.get("seed"),
                "category_fix_mode": r.get("category_fix_mode"),
            },
        )
        metric = str(r.get("metric"))
        out[f"{metric}_mean"] = r.get("mean")
        out[f"{metric}_ci_low"] = r.get("ci_low")
        out[f"{metric}_ci_high"] = r.get("ci_high")
    return list(grouped.values())


def main() -> None:
    args = parse_args()
    load_runtime_dependencies()

    args.eval_csv = require_file(args.eval_csv.resolve(), "eval csv")
    args.repro_splits = require_dir(args.repro_splits.resolve(), "repro splits")
    if args.curated_gt_json is not None:
        args.curated_gt_json = require_file(args.curated_gt_json.resolve(), "curated GT json")
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.n_boot <= 0:
        print("ERROR: --n-boot must be > 0")
        sys.exit(1)

    subset_selection = resolve_subset_filter(args.subset_filter)
    model_filters = resolve_model_filter(args.model_filter)
    gt_map = build_subset_gt_map(args.repro_splits, args.curated_gt_json)

    if "curated" in subset_selection and "curated" not in gt_map:
        print("ERROR: curated subset selected but --curated-gt-json was not provided.")
        sys.exit(1)

    rows = read_eval_rows(args.eval_csv)
    print(f"Loaded {len(rows)} rows from {args.eval_csv}")

    selected: list[dict[str, Any]] = []
    for row in rows:
        subset = str(row.get("subset", "")).strip().lower()
        status = str(row.get("status", "")).strip().lower()
        thr = parse_float(row.get("threshold"))
        pred_json = Path(str(row.get("pred_json_path", "")).strip())
        run_dir = str(row.get("run_dir", ""))

        row["_model_key"] = infer_model_key(row)

        if subset not in subset_selection:
            continue
        if args.status_filter and status != args.status_filter.lower():
            continue
        if thr is None or abs(thr - args.threshold) > 1e-9:
            continue
        if not model_matches(row, model_filters):
            continue
        if args.run_name_contains and args.run_name_contains not in Path(run_dir).name:
            continue
        if not pred_json.exists():
            continue

        row["_pred_json_path"] = pred_json
        selected.append(row)

    dedup: dict[str, dict[str, Any]] = {}
    for row in selected:
        dedup[str(row["_pred_json_path"].resolve())] = row
    selected = list(dedup.values())

    print(
        f"Selected {len(selected)} rows after filters | subsets={sorted(subset_selection)} "
        f"| model_filter={model_filters} | threshold={args.threshold}"
    )
    if not selected:
        print("No matching rows. Nothing to bootstrap.")
        return

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_csv_long = args.out_dir / f"bootstrap_yolo_long_{ts}.csv"
    out_csv_wide = args.out_dir / f"bootstrap_yolo_wide_{ts}.csv"
    out_json = args.out_dir / f"bootstrap_yolo_manifest_{ts}.json"

    if not args.overwrite:
        for p in (out_csv_long, out_csv_wide, out_json):
            if p.exists():
                print(f"ERROR: output exists and --overwrite not set: {p}")
                sys.exit(1)

    boot_rows: list[dict[str, Any]] = []
    for i, row in enumerate(selected, start=1):
        subset = str(row.get("subset", "")).strip().lower()
        gt_json = gt_map.get(subset)
        pred_json: Path = row["_pred_json_path"]
        model_family = str(row.get("model_family", "yolov8"))
        backbone = str(row.get("backbone", ""))
        run_dir = str(row.get("run_dir", ""))
        eval_dir = str(row.get("eval_dir", ""))
        model_key = str(row.get("_model_key", "unknown"))

        print(f"\n[{i}/{len(selected)}] subset={subset} | model={model_key}")
        print(f"  GT:   {gt_json}")
        print(f"  Pred: {pred_json}")

        if gt_json is None or not gt_json.exists():
            for metric in ("AP", "AP50", "APs"):
                boot_rows.append(
                    {
                        "subset": subset,
                        "dataset_tag": row.get("dataset_tag", "curated" if subset == "curated" else "agar100"),
                        "model_family": model_family,
                        "backbone": backbone,
                        "model_key": model_key,
                        "run_dir": run_dir,
                        "eval_dir": eval_dir,
                        "threshold": args.threshold,
                        "metric": metric,
                        "mean": "",
                        "ci_low": "",
                        "ci_high": "",
                        "n_boot": args.n_boot,
                        "seed": args.seed,
                        "gt_json_path": str(gt_json) if gt_json else "",
                        "pred_json_path": str(pred_json),
                        "category_fix_mode": "none",
                        "status": "fail",
                        "error_message": "GT json not found",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
            continue

        try:
            if subset == "curated":
                result, fix_mode = bootstrap_coco_eval_curated_fast(gt_json, pred_json, args.n_boot, args.seed)
            else:
                result, fix_mode = bootstrap_coco_eval_agar_fast(gt_json, pred_json, args.n_boot, args.seed)

            if result is None:
                for metric in ("AP", "AP50", "APs"):
                    boot_rows.append(
                        {
                            "subset": subset,
                            "dataset_tag": row.get("dataset_tag", "curated" if subset == "curated" else "agar100"),
                            "model_family": model_family,
                            "backbone": backbone,
                            "model_key": model_key,
                            "run_dir": run_dir,
                            "eval_dir": eval_dir,
                            "threshold": args.threshold,
                            "metric": metric,
                            "mean": "",
                            "ci_low": "",
                            "ci_high": "",
                            "n_boot": args.n_boot,
                            "seed": args.seed,
                            "gt_json_path": str(gt_json),
                            "pred_json_path": str(pred_json),
                            "category_fix_mode": fix_mode,
                            "status": "fail",
                            "error_message": "bootstrap returned no result",
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                continue

            for metric, vals in result.items():
                boot_rows.append(
                    {
                        "subset": subset,
                        "dataset_tag": row.get("dataset_tag", "curated" if subset == "curated" else "agar100"),
                        "model_family": model_family,
                        "backbone": backbone,
                        "model_key": model_key,
                        "run_dir": run_dir,
                        "eval_dir": eval_dir,
                        "threshold": args.threshold,
                        "metric": metric,
                        "mean": vals["mean"],
                        "ci_low": vals["ci_low"],
                        "ci_high": vals["ci_high"],
                        "n_boot": args.n_boot,
                        "seed": args.seed,
                        "gt_json_path": str(gt_json),
                        "pred_json_path": str(pred_json),
                        "category_fix_mode": fix_mode,
                        "status": "ok",
                        "error_message": "",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )

            if "AP" in result:
                ap = result["AP"]
                print(f"  AP: {ap['mean']:.2f} [{ap['ci_low']:.2f}, {ap['ci_high']:.2f}] | fix={fix_mode}")

        except Exception as exc:
            err = str(exc)
            print(f"  [error] bootstrap failed: {err}")
            for metric in ("AP", "AP50", "APs"):
                boot_rows.append(
                    {
                        "subset": subset,
                        "dataset_tag": row.get("dataset_tag", "curated" if subset == "curated" else "agar100"),
                        "model_family": model_family,
                        "backbone": backbone,
                        "model_key": model_key,
                        "run_dir": run_dir,
                        "eval_dir": eval_dir,
                        "threshold": args.threshold,
                        "metric": metric,
                        "mean": "",
                        "ci_low": "",
                        "ci_high": "",
                        "n_boot": args.n_boot,
                        "seed": args.seed,
                        "gt_json_path": str(gt_json),
                        "pred_json_path": str(pred_json),
                        "category_fix_mode": "none",
                        "status": "fail",
                        "error_message": err,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )

    write_csv(out_csv_long, boot_rows)
    write_csv(out_csv_wide, to_wide_summary(boot_rows))

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eval_csv": str(args.eval_csv),
        "repro_splits": str(args.repro_splits),
        "curated_gt_json": str(args.curated_gt_json) if args.curated_gt_json else "",
        "subset_filter": args.subset_filter,
        "resolved_subsets": sorted(subset_selection),
        "model_filter": args.model_filter,
        "resolved_model_tokens": model_filters,
        "threshold": args.threshold,
        "status_filter": args.status_filter,
        "run_name_contains": args.run_name_contains or "",
        "n_boot": args.n_boot,
        "seed": args.seed,
        "selected_rows": len(selected),
        "output_long_csv": str(out_csv_long),
        "output_wide_csv": str(out_csv_wide),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    ok_count = sum(1 for r in boot_rows if r.get("status") == "ok")
    fail_count = sum(1 for r in boot_rows if r.get("status") == "fail")
    print(f"\nSaved: {out_csv_long}")
    print(f"Saved: {out_csv_wide}")
    print(f"Saved: {out_json}")
    print(f"Rows: {len(boot_rows)} | ok={ok_count} | fail={fail_count}")


if __name__ == "__main__":
    main()
