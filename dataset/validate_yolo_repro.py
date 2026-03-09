#!/usr/bin/env python3
"""
validate_yolo_repro.py
======================
Validate YOLO datasets produced by reproduce_yolo_datasets.py.

Checks:
- required directories/files exist
- one YOLO label file per COCO image
- each label line matches COCO annotation conversion logic
- class ids and normalized bbox values are valid
- data.yaml contains expected split pointers and class metadata
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUBSETS_CROSS = ("bright", "vague", "lowres", "dark")
SPLITS_BASELINE = ("train", "val", "test")


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: dict[str, Any]


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
    parser = argparse.ArgumentParser(description="Validate reproduced YOLO datasets")
    parser.add_argument(
        "--repro-splits",
        type=Path,
        default=Path("reproduced_splits"),
        help="Directory with reconstructed COCO splits",
    )
    parser.add_argument(
        "--repro-yolo",
        type=Path,
        default=Path("reproduced_yolo"),
        help="Directory with reproduced YOLO datasets",
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "cross-subset", "all"),
        default="all",
        help="Which validation workflow to run",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("yolo_validation_output"),
        help="Directory for validation reports",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_expected_label_lines(coco: dict[str, Any]) -> dict[str, list[str]]:
    categories = sorted(coco["categories"], key=lambda c: c["id"])
    cat_map = {c["id"]: i for i, c in enumerate(categories)}
    img_lookup = {img["id"]: img for img in coco["images"]}

    anns_by_img: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    expected: dict[str, list[str]] = {}
    for img_id, img_info in img_lookup.items():
        img_w = float(img_info["width"])
        img_h = float(img_info["height"])
        stem = Path(str(img_info["file_name"])).stem
        lines: list[str] = []
        for ann in anns_by_img.get(img_id, []):
            x, y, w, h = ann["bbox"]
            cx = (float(x) + float(w) / 2.0) / img_w
            cy = (float(y) + float(h) / 2.0) / img_h
            nw = float(w) / img_w
            nh = float(h) / img_h
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))
            class_id = cat_map[ann["category_id"]]
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        expected[f"{stem}.txt"] = lines
    return expected


def parse_label_line(line: str) -> tuple[int, float, float, float, float] | None:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])
        return cls, cx, cy, w, h
    except ValueError:
        return None


def validate_yaml_content(yaml_path: Path, expected_lines: list[str]) -> tuple[bool, list[str]]:
    if not yaml_path.exists():
        return False, [f"missing file: {yaml_path}"]
    text = yaml_path.read_text(encoding="utf-8")
    errors = []
    for line in expected_lines:
        if line not in text:
            errors.append(f"data.yaml missing line: {line}")
    return len(errors) == 0, errors


def validate_one_coco_to_yolo(
    coco_json_path: Path,
    labels_dir: Path,
    images_dir: Path,
) -> CheckResult:
    name = f"{coco_json_path.name} -> {labels_dir}"
    if not coco_json_path.exists():
        return CheckResult(name=name, ok=False, details={"error": f"missing COCO file: {coco_json_path}"})
    if not labels_dir.exists():
        return CheckResult(name=name, ok=False, details={"error": f"missing labels dir: {labels_dir}"})

    coco = load_json(coco_json_path)
    expected = build_expected_label_lines(coco)
    expected_label_files = set(expected.keys())
    actual_label_files = {p.name for p in labels_dir.glob("*.txt")}

    missing_label_files = sorted(expected_label_files - actual_label_files)
    extra_label_files = sorted(actual_label_files - expected_label_files)

    bad_line_format = 0
    bad_value_range = 0
    line_mismatches = 0
    mismatched_examples: list[str] = []

    for label_name, expected_lines in expected.items():
        label_path = labels_dir / label_name
        if not label_path.exists():
            continue
        actual_lines = label_path.read_text(encoding="utf-8").splitlines()

        # Format/range checks independent of exact equality.
        for line in actual_lines:
            parsed = parse_label_line(line)
            if parsed is None:
                bad_line_format += 1
                continue
            _, cx, cy, w, h = parsed
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                bad_value_range += 1

        if actual_lines != expected_lines:
            line_mismatches += 1
            if len(mismatched_examples) < 5:
                mismatched_examples.append(label_name)

    # Image presence check is optional: if directory has files, compare names.
    expected_image_files = {img["file_name"] for img in coco["images"]}
    image_checks: dict[str, Any] = {"checked": False}
    if images_dir.exists():
        actual_images = {p.name for p in images_dir.iterdir() if p.is_file() or p.is_symlink()}
        if len(actual_images) > 0:
            image_checks = {
                "checked": True,
                "missing_images": sorted(expected_image_files - actual_images),
                "extra_images": sorted(actual_images - expected_image_files),
            }
        else:
            image_checks = {"checked": False, "reason": "images directory empty (likely --image-mode none)"}

    ok = (
        len(missing_label_files) == 0
        and len(extra_label_files) == 0
        and bad_line_format == 0
        and bad_value_range == 0
        and line_mismatches == 0
        and (
            not image_checks.get("checked", False)
            or (
                len(image_checks.get("missing_images", [])) == 0
                and len(image_checks.get("extra_images", [])) == 0
            )
        )
    )

    details = {
        "coco_json": str(coco_json_path),
        "labels_dir": str(labels_dir),
        "images_dir": str(images_dir),
        "n_expected_label_files": len(expected_label_files),
        "n_actual_label_files": len(actual_label_files),
        "missing_label_files": missing_label_files[:20],
        "extra_label_files": extra_label_files[:20],
        "bad_line_format_count": bad_line_format,
        "bad_value_range_count": bad_value_range,
        "line_mismatch_count": line_mismatches,
        "line_mismatch_examples": mismatched_examples,
        "image_checks": image_checks,
    }
    return CheckResult(name=name, ok=ok, details=details)


def run_baseline_validation(repro_splits: Path, repro_yolo: Path) -> list[CheckResult]:
    results: list[CheckResult] = []
    yolo_root = repro_yolo / "yolo_agar_total"

    for split in SPLITS_BASELINE:
        coco_path = split_path(repro_splits, split, "total")
        labels_dir = yolo_root / "labels" / split
        images_dir = yolo_root / "images" / split
        results.append(validate_one_coco_to_yolo(coco_path, labels_dir, images_dir))

    # data.yaml checks for baseline
    train_coco = split_path(repro_splits, "train", "total")
    data_yaml = yolo_root / "data.yaml"
    expected_yaml = [
        "train: images/train",
        "val: images/val",
        "test: images/test",
    ]
    if train_coco.exists():
        coco = load_json(train_coco)
        categories = sorted(coco["categories"], key=lambda c: c["id"])
        expected_yaml.append(f"nc: {len(categories)}")
        expected_yaml.extend([f"  - {c['name']}" for c in categories])
    ok, errors = validate_yaml_content(data_yaml, expected_yaml)
    results.append(
        CheckResult(
            name="baseline data.yaml",
            ok=ok,
            details={"data_yaml": str(data_yaml), "errors": errors},
        )
    )
    return results


def run_cross_subset_validation(repro_splits: Path, repro_yolo: Path) -> list[CheckResult]:
    results: list[CheckResult] = []
    yolo_root = repro_yolo / "yolo_subsets"

    total_train = split_path(repro_splits, "train", "total")
    class_names: list[str] = []
    if total_train.exists():
        coco = load_json(total_train)
        class_names = [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])]

    for subset in SUBSETS_CROSS:
        coco_path = split_path(repro_splits, "test", subset)
        labels_dir = yolo_root / subset / "labels" / "test"
        images_dir = yolo_root / subset / "images" / "test"
        results.append(validate_one_coco_to_yolo(coco_path, labels_dir, images_dir))

        data_yaml = yolo_root / subset / "data.yaml"
        expected_yaml = [
            "train: images/test",
            "val: images/test",
            "test: images/test",
            f"nc: {len(class_names)}",
        ]
        expected_yaml.extend([f"  - {name}" for name in class_names])
        ok, errors = validate_yaml_content(data_yaml, expected_yaml)
        results.append(
            CheckResult(
                name=f"{subset} data.yaml",
                ok=ok,
                details={"data_yaml": str(data_yaml), "errors": errors},
            )
        )
    return results


def write_reports(out_dir: Path, results: list[CheckResult], mode: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_json = out_dir / "yolo_validation_report.json"
    report_md = out_dir / "yolo_validation_report.md"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "total_checks": len(results),
        "passed_checks": sum(1 for r in results if r.ok),
        "failed_checks": sum(1 for r in results if not r.ok),
        "checks": [
            {
                "name": r.name,
                "ok": r.ok,
                "details": r.details,
            }
            for r in results
        ],
    }
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# YOLO Reproducibility Validation Report",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Mode: `{mode}`",
        f"- Total checks: **{payload['total_checks']}**",
        f"- Passed: **{payload['passed_checks']}**",
        f"- Failed: **{payload['failed_checks']}**",
        "",
        "## Checks",
        "",
    ]
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        lines.append(f"- [{status}] `{r.name}`")
        if not r.ok:
            details = r.details
            if "error" in details:
                lines.append(f"  - Error: `{details['error']}`")
            elif "errors" in details and details["errors"]:
                for err in details["errors"][:8]:
                    lines.append(f"  - {err}")
            else:
                lines.append(f"  - Details: `{json.dumps(details)[:400]}`")
    lines.append("")

    report_md.write_text("\n".join(lines), encoding="utf-8")
    return report_json, report_md


def main() -> None:
    args = parse_args()
    repro_splits = args.repro_splits.resolve()
    repro_yolo = args.repro_yolo.resolve()
    out_dir = args.out_dir.resolve()

    if not repro_splits.exists():
        print(f"ERROR: Missing repro splits dir: {repro_splits}")
        sys.exit(1)
    if not repro_yolo.exists():
        print(f"ERROR: Missing repro yolo dir: {repro_yolo}")
        sys.exit(1)

    results: list[CheckResult] = []
    if args.mode in ("baseline", "all"):
        results.extend(run_baseline_validation(repro_splits, repro_yolo))
    if args.mode in ("cross-subset", "all"):
        results.extend(run_cross_subset_validation(repro_splits, repro_yolo))

    n_fail = sum(1 for r in results if not r.ok)
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        print(f"[{status}] {r.name}")
    print(f"\nSummary: {len(results) - n_fail}/{len(results)} checks passed")

    report_json, report_md = write_reports(out_dir, results, args.mode)
    print(f"Report JSON: {report_json}")
    print(f"Report MD:   {report_md}")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
