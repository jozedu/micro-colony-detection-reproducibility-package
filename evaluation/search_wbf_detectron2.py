#!/usr/bin/env python3
"""
search_wbf_detectron2.py
========================
Exhaustive WBF search for prediction files (Detectron2 and/or YOLO).

For each selected subset, this script:
1) loads all eligible prediction JSON files from an evaluation CSV,
2) evaluates all model combinations across WBF threshold grids,
3) writes one per-subset CSV with metrics (including per-class AP),
4) writes one merged CSV with all subset results.
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import itertools
import json
import math
import re
import sys
from contextlib import redirect_stdout
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
COCO = None
COCOeval = None
weighted_boxes_fusion = None


def load_runtime_dependencies() -> None:
    global np
    global COCO
    global COCOeval
    global weighted_boxes_fusion
    try:
        import numpy as _np
        from pycocotools.coco import COCO as _COCO
        from pycocotools.cocoeval import COCOeval as _COCOeval
        from ensemble_boxes import weighted_boxes_fusion as _weighted_boxes_fusion
    except ImportError as exc:
        print("ERROR: numpy, pycocotools, and ensemble_boxes are required.")
        print("Install with: python -m pip install numpy pycocotools ensemble-boxes")
        print(f"Import failure: {exc}")
        sys.exit(1)

    np = _np
    COCO = _COCO
    COCOeval = _COCOeval
    weighted_boxes_fusion = _weighted_boxes_fusion


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def parse_float_list(value: str) -> list[float]:
    out: list[float] = []
    for token in [x.strip() for x in value.split(",") if x.strip()]:
        val = parse_float(token)
        if val is None:
            raise ValueError(f"invalid float value: {token}")
        out.append(val)
    if not out:
        raise ValueError("empty float list")
    return out


def parse_threshold_list(value: str) -> list[float]:
    return parse_float_list(value)


def detect_csv_delimiter(path: Path) -> str:
    with open(path, "r", encoding="utf-8-sig") as f:
        first = f.readline()
    return ";" if first.count(";") > first.count(",") else ","


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


def resolve_subset_filter(value: str) -> list[str]:
    token = value.strip().lower()
    if not token:
        print("ERROR: empty --subset-filter")
        sys.exit(1)

    if token in SUBSET_GROUPS:
        wanted = set(SUBSET_GROUPS[token])
        return [s for s in ALL_SUBSETS if s in wanted]

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

    wanted = set(parts)
    return [s for s in ALL_SUBSETS if s in wanted]


def resolve_model_filter(value: str) -> set[str] | None:
    token = value.strip().lower()
    if token == "all":
        return None

    alias = {
        "faster": "faster_rcnn",
        "faster_rcnn": "faster_rcnn",
        "retina": "retinanet",
        "retinanet": "retinanet",
    }
    out: set[str] = set()
    for part in [x.strip().lower() for x in token.split(",") if x.strip()]:
        mapped = alias.get(part)
        if mapped is None:
            print(
                f"ERROR: invalid model family token: {part}. "
                "Allowed: faster, retina, all, comma-lists."
            )
            sys.exit(1)
        out.add(mapped)
    if not out:
        print("ERROR: empty --model-family-filter")
        sys.exit(1)
    return out


def resolve_backbone_filter(value: str) -> set[str] | None:
    token = value.strip().lower()
    if token == "all":
        return None

    alias = {
        "50": "R_50",
        "r50": "R_50",
        "r_50": "R_50",
        "101": "R_101",
        "r101": "R_101",
        "r_101": "R_101",
    }
    out: set[str] = set()
    for part in [x.strip().lower() for x in token.split(",") if x.strip()]:
        mapped = alias.get(part)
        if mapped is None:
            print(
                f"ERROR: invalid backbone token: {part}. "
                "Allowed: 50, 101, all, comma-lists."
            )
            sys.exit(1)
        out.add(mapped)
    if not out:
        print("ERROR: empty --backbone-filter")
        sys.exit(1)
    return out


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


def read_eval_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    delim = detect_csv_delimiter(csv_path)
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            clean = {str(k).strip(): v for k, v in row.items() if k is not None}
            clean["_source_csv"] = str(csv_path.resolve())
            rows.append(clean)
    return rows


def parse_extra_csv_specs(value: str) -> list[str]:
    specs = [x.strip() for x in value.split(",") if x.strip()]
    return specs


def expand_csv_spec(spec: str) -> list[Path]:
    if any(ch in spec for ch in "*?[]"):
        matches = [Path(p).resolve() for p in sorted(glob.glob(spec))]
        return [p for p in matches if p.exists() and p.is_file()]
    p = Path(spec).expanduser()
    if p.exists() and p.is_file():
        return [p.resolve()]
    return []


def collect_eval_rows(primary_csv: Path, extra_specs: list[str]) -> tuple[list[dict[str, Any]], list[Path]]:
    csv_paths: list[Path] = [primary_csv.resolve()]
    for spec in extra_specs:
        expanded = expand_csv_spec(spec)
        if not expanded:
            print(f"WARNING: no files matched extra CSV spec: {spec}")
            continue
        csv_paths.extend(expanded)

    # Deduplicate while preserving order.
    seen: set[str] = set()
    uniq_paths: list[Path] = []
    for p in csv_paths:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq_paths.append(p)

    all_rows: list[dict[str, Any]] = []
    for p in uniq_paths:
        rows = read_eval_rows(p)
        all_rows.extend(rows)
    return all_rows, uniq_paths


def normalize_class_name(name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", name.strip())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "class"


def resolve_pred_path(raw_path: str, csv_path: Path) -> Path:
    candidate = Path(raw_path.strip())
    if candidate.exists():
        return candidate.resolve()
    if not candidate.is_absolute():
        alt = (csv_path.parent / candidate).resolve()
        if alt.exists():
            return alt
    return candidate.resolve()


def load_predictions_by_image(pred_json_path: Path) -> dict[int, list[dict[str, Any]]]:
    with open(pred_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"prediction json must be a list: {pred_json_path}")

    out: dict[int, list[dict[str, Any]]] = {}
    for ann in payload:
        image_id_raw = ann.get("image_id")
        bbox = ann.get("bbox")
        score = parse_float(ann.get("score"))
        cat = ann.get("category_id")

        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        if score is None:
            score = 0.0
        try:
            image_id = int(image_id_raw)
            category_id = int(cat)
            x, y, w, h = [float(v) for v in bbox]
        except Exception:
            continue
        if w <= 0 or h <= 0:
            continue

        out.setdefault(image_id, []).append(
            {
                "bbox": [x, y, w, h],
                "score": float(score),
                "category_id": category_id,
            }
        )
    return out


def evaluate_subset_candidates(
    eval_rows: list[dict[str, Any]],
    subset: str,
    source_thresholds: list[float],
    status_filter: str,
    model_family_filter: set[str] | None,
    backbone_filter: set[str] | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    dedup: set[str] = set()

    for row in eval_rows:
        row_subset = str(row.get("subset", "")).strip().lower()
        status = str(row.get("status", "")).strip().lower()
        threshold = parse_float(row.get("threshold"))
        if row_subset != subset:
            continue
        if status_filter and status != status_filter:
            continue
        if threshold is None:
            continue
        if not any(abs(threshold - thr) <= 1e-9 for thr in source_thresholds):
            continue

        model_family = str(row.get("model_family", "")).strip().lower()
        backbone = str(row.get("backbone", "")).strip().upper()
        if model_family_filter is not None and model_family not in model_family_filter:
            continue
        if backbone_filter is not None and backbone not in backbone_filter:
            continue

        pred_raw = str(row.get("pred_json_path", "")).strip()
        if not pred_raw:
            continue
        source_csv = Path(str(row.get("_source_csv", "")))
        pred_path = resolve_pred_path(pred_raw, source_csv)
        if not pred_path.exists():
            continue

        pred_key = str(pred_path)
        if pred_key in dedup:
            continue
        dedup.add(pred_key)

        run_dir = str(row.get("run_dir", "")).strip()
        run_name = Path(run_dir).name if run_dir else pred_path.parent.name
        ap_value = parse_float(row.get("AP"))

        selected.append(
            {
                "subset": subset,
                "dataset_tag": str(row.get("dataset_tag", "")).strip(),
                "model_family": model_family,
                "backbone": backbone,
                "run_dir": run_dir,
                "run_name": run_name,
                "pred_json_path": pred_path,
                "ap_value": ap_value,
                "trained_on": str(row.get("trained_on", "")).strip(),
                "source_csv": str(source_csv),
                "threshold": threshold,
            }
        )
    return selected


def fuse_predictions_for_image(
    image_id: int,
    width: float,
    height: float,
    model_entries: list[dict[str, Any]],
    combo_idx: tuple[int, ...],
    weights: list[float],
    iou_thr: float,
    skip_box_thr: float,
) -> list[dict[str, Any]]:
    boxes_list: list[list[list[float]]] = []
    scores_list: list[list[float]] = []
    labels_list: list[list[int]] = []

    for idx in combo_idx:
        model = model_entries[idx]
        anns = model["pred_by_image"].get(image_id, [])
        model_boxes: list[list[float]] = []
        model_scores: list[float] = []
        model_labels: list[int] = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

            x1 /= width
            x2 /= width
            y1 /= height
            y2 /= height

            x1 = min(1.0, max(0.0, x1))
            x2 = min(1.0, max(0.0, x2))
            y1 = min(1.0, max(0.0, y1))
            y2 = min(1.0, max(0.0, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            model_boxes.append([x1, y1, x2, y2])
            model_scores.append(float(ann["score"]))
            model_labels.append(int(ann["category_id"]))

        boxes_list.append(model_boxes)
        scores_list.append(model_scores)
        labels_list.append(model_labels)

    if not any(len(x) > 0 for x in boxes_list):
        return []

    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=weights,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
    )

    out: list[dict[str, Any]] = []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [float(v) for v in box]
        x1 = min(1.0, max(0.0, x1))
        x2 = min(1.0, max(0.0, x2))
        y1 = min(1.0, max(0.0, y1))
        y2 = min(1.0, max(0.0, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        abs_x1 = x1 * width
        abs_x2 = x2 * width
        abs_y1 = y1 * height
        abs_y2 = y2 * height
        abs_w = abs_x2 - abs_x1
        abs_h = abs_y2 - abs_y1
        if abs_w <= 0 or abs_h <= 0:
            continue

        out.append(
            {
                "image_id": int(image_id),
                "category_id": int(round(float(label))),
                "bbox": [abs_x1, abs_y1, abs_w, abs_h],
                "score": float(score),
            }
        )
    return out


def run_wbf_for_combo(
    image_ids: list[int],
    image_sizes: dict[int, tuple[float, float]],
    model_entries: list[dict[str, Any]],
    combo_idx: tuple[int, ...],
    weights: list[float],
    iou_thr: float,
    skip_box_thr: float,
) -> list[dict[str, Any]]:
    fused: list[dict[str, Any]] = []
    for image_id in image_ids:
        width, height = image_sizes[image_id]
        fused.extend(
            fuse_predictions_for_image(
                image_id=image_id,
                width=width,
                height=height,
                model_entries=model_entries,
                combo_idx=combo_idx,
                weights=weights,
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr,
            )
        )
    return fused


def extract_coco_metrics(coco_eval: Any, cat_ids: list[int], cat_names: list[str]) -> dict[str, float]:
    stats = coco_eval.stats
    out: dict[str, float] = {
        "AP": float(stats[0]) * 100.0,
        "AP50": float(stats[1]) * 100.0,
        "AP75": float(stats[2]) * 100.0,
        "APs": float(stats[3]) * 100.0,
        "APm": float(stats[4]) * 100.0,
        "APl": float(stats[5]) * 100.0,
        "AR1": float(stats[6]) * 100.0,
        "AR10": float(stats[7]) * 100.0,
        "AR100": float(stats[8]) * 100.0,
    }

    precision = coco_eval.eval["precision"]  # [T, R, K, A, M]
    for k_idx, cat_name in enumerate(cat_names):
        p = precision[:, :, k_idx, 0, 2]
        valid = p[p > -1]
        ap_cat = float(np.mean(valid)) * 100.0 if len(valid) > 0 else 0.0
        out[f"AP_{cat_name}"] = ap_cat

    return out


def evaluate_fused_predictions(
    coco_gt: Any,
    fused_preds: list[dict[str, Any]],
    cat_ids: list[int],
    cat_names: list[str],
) -> dict[str, float] | None:
    if not fused_preds:
        return None

    with redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(fused_preds)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.maxDets = [1, 10, 100]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    return extract_coco_metrics(coco_eval, cat_ids, cat_names)


def collect_fieldnames(rows: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    if not rows:
        return preferred[:]

    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())

    out: list[str] = [k for k in preferred if k in all_keys]
    extras = sorted(k for k in all_keys if k not in out)
    out.extend(extras)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], preferred: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = collect_fieldnames(rows, preferred)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def read_plain_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    delim = detect_csv_delimiter(path)
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            rows.append({str(k).strip(): v for k, v in row.items() if k is not None})
    return rows


def _fmt_key_float(value: Any) -> str:
    val = parse_float(value)
    if val is None:
        return str(value)
    return f"{float(val):.12g}"


def build_row_key(row: dict[str, Any]) -> str:
    payload = {
        "subset": str(row.get("subset", "")).strip().lower(),
        "run_names": str(row.get("run_names", "")).strip(),
        "wbf_iou_thr": _fmt_key_float(row.get("wbf_iou_thr", "")),
        "wbf_skip_box_thr": _fmt_key_float(row.get("wbf_skip_box_thr", "")),
        "weight_hypothesis": str(row.get("weight_hypothesis", "")).strip().lower(),
        "weights": str(row.get("weights", "")).strip(),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def best_ok_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_ap: float | None = None
    for row in rows:
        if str(row.get("status", "")).strip().lower() != "ok":
            continue
        ap = parse_float(row.get("AP"))
        if ap is None:
            continue
        if best_ap is None or ap > best_ap:
            best = row
            best_ap = ap
    return best


def parse_weight_hypotheses(value: str) -> list[str]:
    allowed = {"uniform", "ap", "rank", "grid"}
    parts = [x.strip().lower() for x in value.split(",") if x.strip()]
    if not parts:
        raise ValueError("empty --weight-hypotheses")
    bad = [x for x in parts if x not in allowed]
    if bad:
        raise ValueError(
            f"invalid weight hypothesis value(s): {bad}. "
            "Allowed: uniform, ap, rank, grid"
        )
    # Keep stable order and drop duplicates.
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def build_weight_configs_for_combo(
    combo_idx: tuple[int, ...],
    model_entries: list[dict[str, Any]],
    hypotheses: list[str],
    grid_values: list[float],
    max_grid_configs: int,
    rng_seed: int,
) -> list[tuple[str, list[float]]]:
    n = len(combo_idx)
    configs: list[tuple[str, list[float]]] = []

    if "uniform" in hypotheses:
        configs.append(("uniform", [1.0] * n))

    if "ap" in hypotheses:
        ap_weights: list[float] = []
        for idx in combo_idx:
            ap_val = model_entries[idx].get("ap_value")
            if isinstance(ap_val, (int, float)) and ap_val > 0:
                ap_weights.append(float(ap_val))
            else:
                ap_weights.append(1.0)
        configs.append(("ap", ap_weights))

    if "rank" in hypotheses:
        # Highest AP model gets weight N, then N-1, ..., lowest gets 1.
        order = list(range(n))
        order.sort(
            key=lambda pos: (
                parse_float(model_entries[combo_idx[pos]].get("ap_value")) is not None,
                parse_float(model_entries[combo_idx[pos]].get("ap_value")) or float("-inf"),
                str(model_entries[combo_idx[pos]].get("run_name", "")),
            ),
            reverse=True,
        )
        rank_weights = [1.0] * n
        for rank_pos, pos in enumerate(order, start=1):
            rank_weights[pos] = float(n - rank_pos + 1)
        configs.append(("rank", rank_weights))

    if "grid" in hypotheses:
        grid_tuples = list(itertools.product(grid_values, repeat=n))
        if max_grid_configs > 0 and len(grid_tuples) > max_grid_configs:
            rng = np.random.default_rng(rng_seed)
            sampled_idx = rng.choice(len(grid_tuples), size=max_grid_configs, replace=False)
            sampled_idx = sorted(int(i) for i in sampled_idx)
            grid_tuples = [grid_tuples[i] for i in sampled_idx]

        for w in grid_tuples:
            configs.append(("grid", [float(x) for x in w]))

    # Deduplicate exact vectors.
    unique: list[tuple[str, list[float]]] = []
    seen: set[tuple[float, ...]] = set()
    for label, weights in configs:
        key = tuple(round(float(x), 12) for x in weights)
        if key in seen:
            continue
        seen.add(key)
        unique.append((label, weights))
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exhaustive WBF search for prediction files by subset.",
    )
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=Path("eval_reports/eval_v2_index_long.csv"),
        help="Primary evaluation CSV containing prediction paths.",
    )
    parser.add_argument(
        "--extra-eval-csvs",
        default="",
        help=(
            "Additional CSV specs (comma-separated). "
            "Each item can be a file path or a glob pattern."
        ),
    )
    parser.add_argument(
        "--repro-splits",
        type=Path,
        required=True,
        help="Directory with reproduced split COCO JSON files.",
    )
    parser.add_argument(
        "--curated-gt-json",
        type=Path,
        help="Curated test COCO JSON (required when curated subset is selected).",
    )
    parser.add_argument(
        "--subset-filter",
        default="all",
        help="Subset selection: one subset, agar, all, or comma list.",
    )
    parser.add_argument(
        "--source-thresholds",
        default="0,0.001",
        help="Comma list of input detection thresholds accepted from source CSV rows.",
    )
    parser.add_argument(
        "--status-filter",
        default="ok",
        help="Status filter from eval CSV (default: ok).",
    )
    parser.add_argument(
        "--model-family-filter",
        default="all",
        help="Model-family filter: faster, retina, all, or comma list.",
    )
    parser.add_argument(
        "--backbone-filter",
        default="all",
        help="Backbone filter: 50,101,all or comma list.",
    )
    parser.add_argument(
        "--wbf-iou-thrs",
        default="0.75",
        help="Comma list of WBF IoU thresholds.",
    )
    parser.add_argument(
        "--wbf-skip-box-thrs",
        default="0.01,0.05",
        help="Comma list of WBF skip_box_thr values.",
    )
    parser.add_argument(
        "--weight-hypotheses",
        default="uniform,ap,grid",
        help="Weight search hypotheses: uniform,ap,rank,grid (comma list).",
    )
    parser.add_argument(
        "--weight-grid-values",
        default="1,2,3,5,7",
        help="Comma list of weight values used when grid search is enabled.",
    )
    parser.add_argument(
        "--max-grid-configs-per-combo",
        type=int,
        default=0,
        help=(
            "Limit number of grid weight vectors per combo (0 = use all). "
            "If limited, vectors are sampled deterministically."
        ),
    )
    parser.add_argument(
        "--weight-seed",
        type=int,
        default=42,
        help="Seed used when sampling grid weight vectors.",
    )
    parser.add_argument(
        "--min-models",
        type=int,
        default=2,
        help="Minimum number of models in a combination (default: 2).",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        help="Maximum number of models in a combination (default: all available).",
    )
    parser.add_argument(
        "--combo-mode",
        choices=("topk-prefix", "exhaustive"),
        default="topk-prefix",
        help=(
            "Combination strategy: "
            "topk-prefix tests only top-2, top-3, ... top-k by AP ranking; "
            "exhaustive tests all combinations for each size."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("wbf_reports"),
        help="Output directory.",
    )
    parser.add_argument(
        "--save-best-fused-json",
        action="store_true",
        help="Save fused predictions JSON for the best row of each subset.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help=(
            "Write partial checkpoint CSVs every N newly computed rows "
            "(default: 25, set 0 to disable periodic checkpoint writes)."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Resume from existing partial/final subset CSVs in out-dir "
            "(default: enabled). Use --no-resume to force recompute."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files when they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_runtime_dependencies()

    args.eval_csv = require_file(args.eval_csv.resolve(), "eval csv")
    args.repro_splits = require_dir(args.repro_splits.resolve(), "repro splits")
    if args.curated_gt_json is not None:
        args.curated_gt_json = require_file(args.curated_gt_json.resolve(), "curated GT json")
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    subset_selection = resolve_subset_filter(args.subset_filter)
    model_family_filter = resolve_model_filter(args.model_family_filter)
    backbone_filter = resolve_backbone_filter(args.backbone_filter)

    if "curated" in subset_selection and args.curated_gt_json is None:
        print("ERROR: curated subset selected but --curated-gt-json was not provided.")
        sys.exit(1)

    try:
        source_thresholds = parse_threshold_list(args.source_thresholds)
        iou_grid = parse_float_list(args.wbf_iou_thrs)
        skip_grid = parse_float_list(args.wbf_skip_box_thrs)
        weight_hypotheses = parse_weight_hypotheses(args.weight_hypotheses)
        weight_grid_values = parse_float_list(args.weight_grid_values)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if args.min_models <= 0:
        print("ERROR: --min-models must be >= 1")
        sys.exit(1)
    if args.max_models is not None and args.max_models <= 0:
        print("ERROR: --max-models must be >= 1")
        sys.exit(1)
    if args.max_models is not None and args.max_models < args.min_models:
        print("ERROR: --max-models cannot be smaller than --min-models")
        sys.exit(1)
    if args.max_grid_configs_per_combo < 0:
        print("ERROR: --max-grid-configs-per-combo must be >= 0")
        sys.exit(1)
    if args.checkpoint_every < 0:
        print("ERROR: --checkpoint-every must be >= 0")
        sys.exit(1)

    gt_map = build_subset_gt_map(args.repro_splits, args.curated_gt_json)
    extra_specs = parse_extra_csv_specs(args.extra_eval_csvs)
    eval_rows, eval_csv_paths = collect_eval_rows(args.eval_csv, extra_specs)
    print(f"Loaded {len(eval_rows)} rows from {len(eval_csv_paths)} CSV files")
    for p in eval_csv_paths:
        print(f"  - {p}")

    preferred_columns = [
        "subset",
        "dataset_tag",
        "combo_id",
        "weight_config_id",
        "combo_mode",
        "n_models",
        "run_names",
        "model_families",
        "backbones",
        "trained_on",
        "source_csvs",
        "source_thresholds",
        "wbf_iou_thr",
        "wbf_skip_box_thr",
        "weight_hypothesis",
        "weights",
        "AP",
        "AP50",
        "AP75",
        "APs",
        "APm",
        "APl",
        "AR1",
        "AR10",
        "AR100",
        "status",
        "error_message",
        "timestamp_utc",
    ]

    merged_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    merged_partial_csv = args.out_dir / "wbf_search_merged_partial.csv"
    best_partial_csv = args.out_dir / "wbf_search_best_by_subset_partial.csv"

    for subset in subset_selection:
        print("\n" + "=" * 72)
        print(f"Subset: {subset}")
        print("=" * 72)
        subset_csv = args.out_dir / f"wbf_search_{subset}.csv"
        subset_partial_csv = args.out_dir / f"wbf_search_{subset}_partial.csv"

        if args.resume and not args.overwrite and subset_csv.exists():
            done_rows = read_plain_csv_rows(subset_csv)
            for row in done_rows:
                row["row_key"] = build_row_key(row)
            merged_rows.extend(done_rows)
            best_done = best_ok_row(done_rows)
            if best_done is not None:
                best_rows.append(best_done)
            print(f"  resume: loaded completed subset CSV ({len(done_rows)} rows) and skipped recompute")
            continue

        gt_json = gt_map.get(subset)
        if gt_json is None or not gt_json.exists():
            print(f"  [skip] GT json not found for subset={subset}")
            continue

        candidates = evaluate_subset_candidates(
            eval_rows=eval_rows,
            subset=subset,
            source_thresholds=source_thresholds,
            status_filter=args.status_filter.strip().lower(),
            model_family_filter=model_family_filter,
            backbone_filter=backbone_filter,
        )

        print(f"  candidates: {len(candidates)} prediction files")
        if not candidates:
            continue

        coco_gt = COCO(str(gt_json))
        image_ids = sorted(coco_gt.getImgIds())
        imgs = coco_gt.loadImgs(image_ids)
        image_sizes: dict[int, tuple[float, float]] = {}
        for img in imgs:
            iid = int(img["id"])
            width = float(img["width"])
            height = float(img["height"])
            image_sizes[iid] = (width, height)

        cat_ids = coco_gt.getCatIds()
        cat_names = [
            normalize_class_name(c["name"])
            for c in coco_gt.loadCats(cat_ids)
        ]

        model_entries: list[dict[str, Any]] = []
        for candidate in candidates:
            pred_by_image = load_predictions_by_image(candidate["pred_json_path"])
            model_entries.append(
                {
                    **candidate,
                    "pred_by_image": pred_by_image,
                }
            )

        n_models_available = len(model_entries)

        # Rank candidates by source AP (highest first) for topk-prefix strategy.
        ranked_indices = list(range(n_models_available))
        ranked_indices.sort(
            key=lambda i: (
                parse_float(model_entries[i].get("ap_value")) is not None,
                parse_float(model_entries[i].get("ap_value")) or float("-inf"),
                str(model_entries[i].get("run_name", "")),
            ),
            reverse=True,
        )

        max_models = args.max_models if args.max_models is not None else n_models_available
        max_models = min(max_models, n_models_available)
        min_models = min(args.min_models, max_models)

        if min_models > max_models:
            print(
                f"  [skip] no valid combination size for subset={subset} "
                f"(min={args.min_models}, max={args.max_models}, available={n_models_available})"
            )
            continue

        combo_count = 0
        if args.combo_mode == "topk-prefix":
            combo_count = max_models - min_models + 1
        else:
            for r in range(min_models, max_models + 1):
                combo_count += math.comb(n_models_available, r)
        # Estimate jobs including weight hypotheses, accounting for combo size.
        total_jobs = 0
        for r in range(min_models, max_models + 1):
            if args.combo_mode == "topk-prefix":
                combos_r = 1
            else:
                combos_r = math.comb(n_models_available, r)
            weight_cfg_r = 0
            if "uniform" in weight_hypotheses:
                weight_cfg_r += 1
            if "ap" in weight_hypotheses:
                weight_cfg_r += 1
            if "rank" in weight_hypotheses:
                weight_cfg_r += 1
            if "grid" in weight_hypotheses:
                grid_count = len(weight_grid_values) ** r
                if args.max_grid_configs_per_combo > 0:
                    grid_count = min(grid_count, args.max_grid_configs_per_combo)
                weight_cfg_r += grid_count
            total_jobs += combos_r * len(iou_grid) * len(skip_grid) * max(weight_cfg_r, 1)
        print(
            f"  combinations: {combo_count} ({args.combo_mode}) | thresholds per combo: {len(iou_grid) * len(skip_grid)} "
            f"| total jobs: {total_jobs}"
        )

        subset_rows: list[dict[str, Any]] = []
        best_row: dict[str, Any] | None = None
        best_fused: list[dict[str, Any]] | None = None
        processed_keys: set[str] = set()
        newly_computed_rows = 0

        if args.resume and not args.overwrite and subset_partial_csv.exists():
            subset_rows = read_plain_csv_rows(subset_partial_csv)
            for row in subset_rows:
                row["row_key"] = build_row_key(row)
                processed_keys.add(row["row_key"])
            best_row = best_ok_row(subset_rows)
            print(f"  resume: loaded partial subset checkpoint ({len(subset_rows)} rows)")

        job_idx = 0
        combo_id = 0

        for r in range(min_models, max_models + 1):
            if args.combo_mode == "topk-prefix":
                combo_iter = [tuple(ranked_indices[:r])]
            else:
                combo_iter = itertools.combinations(range(n_models_available), r)

            for combo_idx in combo_iter:
                combo_id += 1
                run_names = [model_entries[i]["run_name"] for i in combo_idx]
                families = [model_entries[i]["model_family"] for i in combo_idx]
                backbones = [model_entries[i]["backbone"] for i in combo_idx]
                trained_on_values = [
                    str(model_entries[i].get("trained_on", "")).strip() for i in combo_idx
                ]
                source_csv_values = [str(model_entries[i].get("source_csv", "")).strip() for i in combo_idx]

                weight_configs = build_weight_configs_for_combo(
                    combo_idx=combo_idx,
                    model_entries=model_entries,
                    hypotheses=weight_hypotheses,
                    grid_values=weight_grid_values,
                    max_grid_configs=args.max_grid_configs_per_combo,
                    rng_seed=args.weight_seed + combo_id,
                )

                for iou_thr in iou_grid:
                    for skip_thr in skip_grid:
                        for weight_config_id, (weight_hyp, combo_weights) in enumerate(weight_configs, start=1):
                            job_idx += 1
                            print(
                                f"  [{job_idx}/{total_jobs}] combo={combo_id} size={len(combo_idx)} "
                                f"iou={iou_thr} skip={skip_thr} weights={weight_hyp}#{weight_config_id}",
                                end="\r",
                            )

                            row_base: dict[str, Any] = {
                                "subset": subset,
                                "dataset_tag": model_entries[combo_idx[0]].get("dataset_tag", ""),
                                "combo_id": combo_id,
                                "weight_config_id": weight_config_id,
                                "combo_mode": args.combo_mode,
                                "n_models": len(combo_idx),
                                "run_names": "|".join(run_names),
                                "model_families": "|".join(families),
                                "backbones": "|".join(backbones),
                                "trained_on": "|".join(trained_on_values),
                                "source_csvs": "|".join(source_csv_values),
                                "source_thresholds": ",".join(f"{t:.6g}" for t in source_thresholds),
                                "wbf_iou_thr": iou_thr,
                                "wbf_skip_box_thr": skip_thr,
                                "weight_hypothesis": weight_hyp,
                                "weights": "|".join(f"{float(w):.6g}" for w in combo_weights),
                                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            }
                            row_key = build_row_key(row_base)
                            if row_key in processed_keys:
                                continue

                            try:
                                fused_preds = run_wbf_for_combo(
                                    image_ids=image_ids,
                                    image_sizes=image_sizes,
                                    model_entries=model_entries,
                                    combo_idx=combo_idx,
                                    weights=combo_weights,
                                    iou_thr=iou_thr,
                                    skip_box_thr=skip_thr,
                                )
                                metrics = evaluate_fused_predictions(
                                    coco_gt=coco_gt,
                                    fused_preds=fused_preds,
                                    cat_ids=cat_ids,
                                    cat_names=cat_names,
                                )
                                if metrics is None:
                                    subset_rows.append(
                                        {
                                            **row_base,
                                            "status": "fail",
                                            "error_message": "empty fused predictions or no overlap",
                                        }
                                    )
                                    continue

                                row_ok = {
                                    **row_base,
                                    **metrics,
                                    "row_key": row_key,
                                    "status": "ok",
                                    "error_message": "",
                                }
                                subset_rows.append(
                                    row_ok
                                )
                                processed_keys.add(row_key)
                                newly_computed_rows += 1

                                ap = row_ok.get("AP")
                                if isinstance(ap, (int, float)):
                                    if best_row is None or ap > best_row.get("AP", float("-inf")):
                                        best_row = row_ok
                                        best_fused = fused_preds

                            except Exception as exc:
                                subset_rows.append(
                                    {
                                        **row_base,
                                        "row_key": row_key,
                                        "status": "fail",
                                        "error_message": str(exc),
                                    }
                                )
                                processed_keys.add(row_key)
                                newly_computed_rows += 1

                            if args.checkpoint_every > 0 and newly_computed_rows >= args.checkpoint_every:
                                write_csv(subset_partial_csv, subset_rows, preferred_columns)
                                write_csv(merged_partial_csv, merged_rows + subset_rows, preferred_columns)
                                current_best_rows = best_rows + ([best_row] if best_row is not None else [])
                                write_csv(best_partial_csv, current_best_rows, preferred_columns)
                                print(
                                    f"\n  checkpoint: saved partial CSVs "
                                    f"(subset rows={len(subset_rows)})"
                                )
                                newly_computed_rows = 0

        print()
        if subset_csv.exists() and not args.overwrite:
            print(f"ERROR: output exists and --overwrite not set: {subset_csv}")
            sys.exit(1)
        write_csv(subset_csv, subset_rows, preferred_columns)
        print(f"  saved: {subset_csv} ({len(subset_rows)} rows)")
        write_csv(subset_partial_csv, subset_rows, preferred_columns)

        if best_row is not None:
            best_rows.append(best_row)
            print(
                f"  best: AP={best_row.get('AP'):.4f} | "
                f"combo={best_row.get('run_names')} | "
                f"iou={best_row.get('wbf_iou_thr')} skip={best_row.get('wbf_skip_box_thr')}"
            )
            if args.save_best_fused_json and best_fused is not None:
                best_json = args.out_dir / f"wbf_best_{subset}_predictions.json"
                if best_json.exists() and not args.overwrite:
                    print(f"ERROR: output exists and --overwrite not set: {best_json}")
                    sys.exit(1)
                with open(best_json, "w", encoding="utf-8") as f:
                    json.dump(best_fused, f)
                print(f"  saved: {best_json}")
        else:
            print("  best: none (no successful rows)")

        merged_rows.extend(subset_rows)
        write_csv(merged_partial_csv, merged_rows, preferred_columns)
        write_csv(best_partial_csv, best_rows, preferred_columns)

    merged_csv = args.out_dir / "wbf_search_merged.csv"
    if merged_csv.exists() and not args.overwrite:
        print(f"ERROR: output exists and --overwrite not set: {merged_csv}")
        sys.exit(1)
    write_csv(merged_csv, merged_rows, preferred_columns)
    print(f"\nSaved merged CSV: {merged_csv} ({len(merged_rows)} rows)")

    best_csv = args.out_dir / "wbf_search_best_by_subset.csv"
    if best_csv.exists() and not args.overwrite:
        print(f"ERROR: output exists and --overwrite not set: {best_csv}")
        sys.exit(1)
    write_csv(best_csv, best_rows, preferred_columns)
    print(f"Saved best-by-subset CSV: {best_csv} ({len(best_rows)} rows)")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eval_csv": str(args.eval_csv),
        "extra_eval_csvs": extra_specs,
        "resolved_eval_csvs": [str(p) for p in eval_csv_paths],
        "repro_splits": str(args.repro_splits),
        "curated_gt_json": str(args.curated_gt_json) if args.curated_gt_json else "",
        "subset_filter": args.subset_filter,
        "resolved_subsets": subset_selection,
        "source_thresholds": source_thresholds,
        "status_filter": args.status_filter,
        "model_family_filter": args.model_family_filter,
        "backbone_filter": args.backbone_filter,
        "wbf_iou_thrs": iou_grid,
        "wbf_skip_box_thrs": skip_grid,
        "weight_hypotheses": weight_hypotheses,
        "weight_grid_values": weight_grid_values,
        "max_grid_configs_per_combo": args.max_grid_configs_per_combo,
        "weight_seed": args.weight_seed,
        "checkpoint_every": args.checkpoint_every,
        "resume": args.resume,
        "combo_mode": args.combo_mode,
        "min_models": args.min_models,
        "max_models": args.max_models,
        "merged_csv": str(merged_csv),
        "best_csv": str(best_csv),
        "merged_partial_csv": str(merged_partial_csv),
        "best_partial_csv": str(best_partial_csv),
        "total_rows": len(merged_rows),
        "best_rows": len(best_rows),
    }
    manifest_path = args.out_dir / "wbf_search_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        print(f"ERROR: output exists and --overwrite not set: {manifest_path}")
        sys.exit(1)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
