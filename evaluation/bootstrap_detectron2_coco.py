#!/usr/bin/env python3
"""
bootstrap_detectron2_coco.py
============================
Bootstrap COCO metrics for Detectron2 evaluation outputs.

This script uses the optimized in-memory bootstrap logic and supports selecting:
- subsets: one / group / all
- models: one / group / all
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

MODEL_KEYS = (
    "faster_rcnn_r50",
    "faster_rcnn_r101",
    "retinanet_r50",
    "retinanet_r101",
)
MODEL_GROUPS = {
    "all": list(MODEL_KEYS),
    "faster": ["faster_rcnn_r50", "faster_rcnn_r101"],
    "faster_rcnn": ["faster_rcnn_r50", "faster_rcnn_r101"],
    "retina": ["retinanet_r50", "retinanet_r101"],
    "retinanet": ["retinanet_r50", "retinanet_r101"],
    "r50": ["faster_rcnn_r50", "retinanet_r50"],
    "r_50": ["faster_rcnn_r50", "retinanet_r50"],
    "r101": ["faster_rcnn_r101", "retinanet_r101"],
    "r_101": ["faster_rcnn_r101", "retinanet_r101"],
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
    parser = argparse.ArgumentParser(description="Bootstrap Detectron2 COCO metrics from evaluation CSV.")
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=Path("eval_reports/eval_v2_index_long.csv"),
        help=(
            "Evaluation CSV produced by evaluation/evaluate_detectron2_outputs.py "
            "(default: eval_reports/eval_v2_index_long.csv)."
        ),
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
        default="all",
        help=(
            "Subset selection: one subset, group alias (agar), all, "
            "or comma list (e.g. bright,dark,curated)."
        ),
    )
    parser.add_argument(
        "--model-filter",
        default="all",
        help=(
            "Model selection: one key (faster_rcnn_r50, faster_rcnn_r101, "
            "retinanet_r50, retinanet_r101), group (faster, retinanet, r50, r101), "
            "all, or comma list."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Evaluate rows at this confidence threshold (default: 0.0).",
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


def normalize_model_token(token: str) -> str:
    t = token.strip().lower().replace("-", "_")
    t = re.sub(r"_+", "_", t)
    t = t.replace("_r_50", "_r50").replace("_r_101", "_r101")
    if t in ("fasterrcnn_r50",):
        return "faster_rcnn_r50"
    if t in ("fasterrcnn_r101",):
        return "faster_rcnn_r101"
    return t


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


def resolve_model_filter(value: str) -> set[str]:
    token = normalize_model_token(value)
    if not token:
        print("ERROR: empty --model-filter")
        sys.exit(1)

    if token in MODEL_GROUPS:
        return set(MODEL_GROUPS[token])

    parts = [normalize_model_token(x) for x in token.split(",") if x.strip()]
    if not parts:
        print("ERROR: empty --model-filter")
        sys.exit(1)

    resolved: set[str] = set()
    invalid: list[str] = []
    for p in parts:
        if p in MODEL_GROUPS:
            resolved.update(MODEL_GROUPS[p])
        elif p in MODEL_KEYS:
            resolved.add(p)
        else:
            invalid.append(p)
    if invalid:
        print(
            f"ERROR: invalid model filter(s): {invalid}. "
            "Allowed keys: faster_rcnn_r50, faster_rcnn_r101, retinanet_r50, retinanet_r101; "
            "groups: faster, retinanet, r50, r101, all"
        )
        sys.exit(1)
    return resolved


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
        rows = []
        for row in reader:
            clean = {str(k).strip(): v for k, v in row.items()}
            rows.append(clean)
    return rows


def parse_model_key(row: dict[str, Any]) -> str | None:
    family_raw = str(row.get("model_family", "")).strip().lower()
    backbone_raw = str(row.get("backbone", "")).strip().upper()
    run_dir = str(row.get("run_dir", "")).lower()

    family = None
    if "faster" in family_raw or "faster" in run_dir:
        family = "faster_rcnn"
    if "retina" in family_raw or "retina" in run_dir:
        family = "retinanet"

    backbone = None
    if "R_50" in backbone_raw or "r_50" in run_dir:
        backbone = "r50"
    if "R_101" in backbone_raw or "r_101" in run_dir:
        backbone = "r101"

    if family and backbone:
        return f"{family}_{backbone}"
    return None


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


# ---------------------------------------------------------------------------
# Fast bootstrap helpers (adapted from old/bootstrap_coco_eval_fast.py)
# ---------------------------------------------------------------------------

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
    for k_idx, (_, cat_name) in enumerate(zip(cat_ids, cat_names)):
        p = precision[:, :, k_idx, 0, 2]
        valid = p[p > -1]
        ap_cat = float(np.mean(valid)) * 100.0 if len(valid) > 0 else 0.0
        results[f"AP_{cat_name}"] = ap_cat
    return results


def _run_one_replicate(boot_gt_dict: dict[str, Any], boot_preds: list[dict[str, Any]], cat_ids: list[int], cat_names: list[str]) -> dict[str, float] | None:
    from pycocotools.cocoeval import COCOeval

    if not boot_preds:
        return None
    # Suppress pycocotools verbose output
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


def _build_boot_sample(orig_img_ids: np.ndarray, coco_gt: Any, pred_by_image: dict[int, list[dict[str, Any]]], rng: np.random.Generator) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sampled_ids = rng.choice(orig_img_ids, size=len(orig_img_ids), replace=True)

    boot_images: list[dict[str, Any]] = []
    boot_anns: list[dict[str, Any]] = []
    boot_preds: list[dict[str, Any]] = []
    new_id = 1

    for orig_id in sampled_ids:
        oid = int(orig_id)
        img_info_list = coco_gt.loadImgs(oid)
        if not img_info_list:
            continue

        new_img = img_info_list[0].copy()
        new_img["id"] = new_id

        ann_ids = coco_gt.getAnnIds(imgIds=oid)
        for ann in coco_gt.loadAnns(ann_ids):
            new_ann = ann.copy()
            new_ann["image_id"] = new_id
            boot_anns.append(new_ann)

        for pred in pred_by_image.get(oid, []):
            new_pred = pred.copy()
            new_pred["image_id"] = new_id
            boot_preds.append(new_pred)

        boot_images.append(new_img)
        new_id += 1

    for i, ann in enumerate(boot_anns):
        ann["id"] = i + 1

    boot_gt_dict = {
        "images": boot_images,
        "annotations": boot_anns,
        "categories": coco_gt.dataset["categories"],
    }
    return boot_gt_dict, boot_preds


def bootstrap_coco_eval_fast(gt_json_path: Path, pred_json_path: Path, n_boot: int, seed: int) -> dict[str, dict[str, float]] | None:
    from pycocotools.coco import COCO

    # Suppress pycocotools index output
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
        return None

    gt_img_ids = set(coco_gt.getImgIds())
    pred_img_ids = {int(p["image_id"]) for p in pred_list}
    overlap = gt_img_ids & pred_img_ids
    if not overlap:
        return None

    valid_ids = list(overlap)
    cat_ids = coco_gt.getCatIds()
    cat_names = [c["name"].replace(" ", "_").replace(".", "") for c in coco_gt.loadCats(cat_ids)]

    pred_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in pred_list:
        pred_by_image[int(p["image_id"])].append(p)

    rng = np.random.default_rng(seed)
    orig_img_ids = np.array(valid_ids)

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
        return None

    return {
        metric: {
            "mean": float(np.mean(vals)),
            "ci_low": float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5)),
        }
        for metric, vals in accum.items()
    }


def bootstrap_coco_eval_curated_fast(gt_json_path: Path, pred_json_path: Path, n_boot: int, seed: int) -> dict[str, dict[str, float]] | None:
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # Curated fix: remove parent class if present.
    cat_info = {int(c["id"]): str(c["name"]) for c in gt_data.get("categories", [])}
    if 0 in cat_info and cat_info[0].lower() in ("colonies", "colony"):
        gt_data["categories"] = [c for c in gt_data["categories"] if int(c["id"]) != 0]
        gt_data["annotations"] = [a for a in gt_data["annotations"] if int(a["category_id"]) != 0]

    with open(pred_json_path, "r", encoding="utf-8") as f:
        pred_list = json.load(f)
    if not pred_list:
        return None

    gt_img_ids = {int(img["id"]) for img in gt_data.get("images", [])}
    pred_img_ids = {int(p["image_id"]) for p in pred_list if isinstance(p.get("image_id"), (int, float))}
    overlap = gt_img_ids & pred_img_ids
    if len(overlap) < max(1, len(pred_img_ids) // 2):
        # Remap filename IDs to integer IDs if needed.
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
            pred_img_ids = {int(p["image_id"]) for p in pred_list if isinstance(p.get("image_id"), (int, float))}
            overlap = gt_img_ids & pred_img_ids
    if not overlap:
        return None

    coco_gt = _coco_from_dict(gt_data)
    cat_ids = coco_gt.getCatIds()
    cat_names = [c["name"].replace(" ", "_").replace(".", "") for c in coco_gt.loadCats(cat_ids)]
    valid_ids = list(overlap)

    pred_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in pred_list:
        if isinstance(p.get("image_id"), (int, float)):
            pred_by_image[int(p["image_id"])].append(p)

    rng = np.random.default_rng(seed)
    orig_img_ids = np.array(valid_ids)

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
        return None

    return {
        metric: {
            "mean": float(np.mean(vals)),
            "ci_low": float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5)),
        }
        for metric, vals in accum.items()
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def to_wide_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        key = (
            str(r.get("subset")),
            str(r.get("model_family")),
            str(r.get("backbone")),
            str(r.get("run_dir")),
        )
        row = grouped.setdefault(
            key,
            {
                "subset": r.get("subset"),
                "model_family": r.get("model_family"),
                "backbone": r.get("backbone"),
                "run_dir": r.get("run_dir"),
                "eval_dir": r.get("eval_dir"),
                "threshold": r.get("threshold"),
                "n_boot": r.get("n_boot"),
                "seed": r.get("seed"),
            },
        )
        metric = str(r.get("metric"))
        row[f"{metric}_mean"] = r.get("mean")
        row[f"{metric}_ci_low"] = r.get("ci_low")
        row[f"{metric}_ci_high"] = r.get("ci_high")
    return list(grouped.values())


def _safe_subset_name(value: str) -> str:
    name = value.strip().lower()
    if not name:
        name = "unknown"
    name = re.sub(r"[^a-z0-9_\\-]+", "_", name)
    return name


def write_bootstrap_outputs(
    out_dir: Path,
    ts: str,
    boot_rows: list[dict[str, Any]],
) -> tuple[Path, Path, dict[str, dict[str, str | int]]]:
    out_csv_long = out_dir / f"bootstrap_detectron2_long_{ts}.csv"
    out_csv_wide = out_dir / f"bootstrap_detectron2_wide_{ts}.csv"

    write_csv(out_csv_long, boot_rows)
    write_csv(out_csv_wide, to_wide_summary(boot_rows))

    subset_outputs: dict[str, dict[str, str | int]] = {}
    by_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in boot_rows:
        subset = str(row.get("subset", "")).strip().lower()
        by_subset[subset].append(row)

    for subset, rows in sorted(by_subset.items()):
        subset_tag = _safe_subset_name(subset)
        subset_long = out_dir / f"bootstrap_detectron2_long_{subset_tag}_{ts}.csv"
        subset_wide = out_dir / f"bootstrap_detectron2_wide_{subset_tag}_{ts}.csv"
        write_csv(subset_long, rows)
        write_csv(subset_wide, to_wide_summary(rows))
        subset_outputs[subset] = {
            "rows": len(rows),
            "long_csv": str(subset_long),
            "wide_csv": str(subset_wide),
        }

    return out_csv_long, out_csv_wide, subset_outputs


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
    model_selection = resolve_model_filter(args.model_filter)
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
        pred_json_path = Path(str(row.get("pred_json_path", "")).strip())
        run_dir = str(row.get("run_dir", ""))
        model_key = parse_model_key(row)

        if subset not in subset_selection:
            continue
        if args.status_filter and status != args.status_filter.lower():
            continue
        if thr is None or abs(thr - args.threshold) > 1e-9:
            continue
        if model_key is None or model_key not in model_selection:
            continue
        if args.run_name_contains and args.run_name_contains not in Path(run_dir).name:
            continue
        if not pred_json_path.exists():
            continue

        row["_model_key"] = model_key
        row["_pred_json_path"] = pred_json_path
        selected.append(row)

    # Deduplicate by pred path to avoid repeated CSV rows for the same evaluation file.
    dedup: dict[str, dict[str, Any]] = {}
    for row in selected:
        dedup[str(row["_pred_json_path"].resolve())] = row
    selected = list(dedup.values())

    print(
        f"Selected {len(selected)} rows after filters | subsets={sorted(subset_selection)} "
        f"| models={sorted(model_selection)} | threshold={args.threshold}"
    )
    if not selected:
        print("No matching rows. Nothing to bootstrap.")
        return

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_csv_long = args.out_dir / f"bootstrap_detectron2_long_{ts}.csv"
    out_csv_wide = args.out_dir / f"bootstrap_detectron2_wide_{ts}.csv"
    out_json = args.out_dir / f"bootstrap_detectron2_manifest_{ts}.json"

    if not args.overwrite:
        for p in (out_csv_long, out_csv_wide, out_json):
            if p.exists():
                print(f"ERROR: output exists and --overwrite not set: {p}")
                sys.exit(1)

    boot_rows: list[dict[str, Any]] = []
    subset_outputs: dict[str, dict[str, str | int]] = {}
    for idx, row in enumerate(selected, start=1):
        subset = str(row["subset"]).strip().lower()
        model_family = str(row.get("model_family", ""))
        backbone = str(row.get("backbone", ""))
        run_dir = str(row.get("run_dir", ""))
        eval_dir = str(row.get("eval_dir", ""))
        pred_json_path: Path = row["_pred_json_path"]
        gt_json_path = gt_map.get(subset)

        print(f"\n[{idx}/{len(selected)}] subset={subset} | model={model_family}/{backbone}")
        print(f"  GT:   {gt_json_path}")
        print(f"  Pred: {pred_json_path}")

        if gt_json_path is None or not gt_json_path.exists():
            for metric in ("AP", "AP50", "APs"):
                boot_rows.append(
                    {
                        "subset": subset,
                        "dataset_tag": row.get("dataset_tag", "curated" if subset == "curated" else "agar100"),
                        "model_family": model_family,
                        "backbone": backbone,
                        "model_key": row.get("_model_key"),
                        "run_dir": run_dir,
                        "eval_dir": eval_dir,
                        "threshold": args.threshold,
                        "metric": metric,
                        "mean": "",
                        "ci_low": "",
                        "ci_high": "",
                        "n_boot": args.n_boot,
                        "seed": args.seed,
                        "gt_json_path": str(gt_json_path) if gt_json_path else "",
                        "pred_json_path": str(pred_json_path),
                        "status": "fail",
                        "error_message": "GT json not found",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
            continue

        try:
            if subset == "curated":
                result = bootstrap_coco_eval_curated_fast(gt_json_path, pred_json_path, args.n_boot, args.seed)
            else:
                result = bootstrap_coco_eval_fast(gt_json_path, pred_json_path, args.n_boot, args.seed)

            if result is None:
                for metric in ("AP", "AP50", "APs"):
                    boot_rows.append(
                        {
                            "subset": subset,
                            "dataset_tag": row.get("dataset_tag", "curated" if subset == "curated" else "agar100"),
                            "model_family": model_family,
                            "backbone": backbone,
                            "model_key": row.get("_model_key"),
                            "run_dir": run_dir,
                            "eval_dir": eval_dir,
                            "threshold": args.threshold,
                            "metric": metric,
                            "mean": "",
                            "ci_low": "",
                            "ci_high": "",
                            "n_boot": args.n_boot,
                            "seed": args.seed,
                            "gt_json_path": str(gt_json_path),
                            "pred_json_path": str(pred_json_path),
                            "status": "fail",
                            "error_message": "bootstrap returned no result",
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                continue

            for metric, metric_values in result.items():
                boot_rows.append(
                    {
                        "subset": subset,
                        "dataset_tag": row.get("dataset_tag", "curated" if subset == "curated" else "agar100"),
                        "model_family": model_family,
                        "backbone": backbone,
                        "model_key": row.get("_model_key"),
                        "run_dir": run_dir,
                        "eval_dir": eval_dir,
                        "threshold": args.threshold,
                        "metric": metric,
                        "mean": metric_values["mean"],
                        "ci_low": metric_values["ci_low"],
                        "ci_high": metric_values["ci_high"],
                        "n_boot": args.n_boot,
                        "seed": args.seed,
                        "gt_json_path": str(gt_json_path),
                        "pred_json_path": str(pred_json_path),
                        "status": "ok",
                        "error_message": "",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
            if "AP" in result:
                ap = result["AP"]
                print(f"  AP: {ap['mean']:.2f} [{ap['ci_low']:.2f}, {ap['ci_high']:.2f}]")

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
                        "model_key": row.get("_model_key"),
                        "run_dir": run_dir,
                        "eval_dir": eval_dir,
                        "threshold": args.threshold,
                        "metric": metric,
                        "mean": "",
                        "ci_low": "",
                        "ci_high": "",
                        "n_boot": args.n_boot,
                        "seed": args.seed,
                        "gt_json_path": str(gt_json_path),
                        "pred_json_path": str(pred_json_path),
                        "status": "fail",
                        "error_message": err,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
        finally:
            out_csv_long, out_csv_wide, subset_outputs = write_bootstrap_outputs(
                args.out_dir,
                ts,
                boot_rows,
            )

    out_csv_long, out_csv_wide, subset_outputs = write_bootstrap_outputs(
        args.out_dir,
        ts,
        boot_rows,
    )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eval_csv": str(args.eval_csv),
        "repro_splits": str(args.repro_splits),
        "curated_gt_json": str(args.curated_gt_json) if args.curated_gt_json else "",
        "subset_filter": args.subset_filter,
        "resolved_subsets": sorted(subset_selection),
        "model_filter": args.model_filter,
        "resolved_models": sorted(model_selection),
        "threshold": args.threshold,
        "status_filter": args.status_filter,
        "run_name_contains": args.run_name_contains or "",
        "n_boot": args.n_boot,
        "seed": args.seed,
        "selected_rows": len(selected),
        "output_long_csv": str(out_csv_long),
        "output_wide_csv": str(out_csv_wide),
        "subset_outputs": subset_outputs,
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
