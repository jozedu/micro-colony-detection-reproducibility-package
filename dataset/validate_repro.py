#!/usr/bin/env python3
"""
validate_repro.py
=================
Validate reproduced COCO split JSONs for internal consistency and
correctness against the embedded split assignments.

This script is intended for reviewers who have:
1. Downloaded the AGAR dataset themselves.
2. Run ``reproduce_splits.py --mode reconstruct`` to generate the splits.
3. Want to verify the generated files are correct.

Checks performed
----------------
- Image / annotation counts per split.
- Val and test image ids match ``configs/split_assignments.json``.
- Train ids = curated group members − val − test.
- No train / val / test overlap within each group.
- Category remapping is correct (0 → S.aureus, 1 → P.aeruginosa, 2 → E.coli).
- Annotation ids match source annotations.json after curation + remap.
- Per-class annotation distributions.

Outputs
-------
validation_report.json   – machine-readable results
validation_report.md     – human-readable Markdown report

Usage
-----
    python scripts/validate_repro.py \\
        --raw-root /data/AGAR \\
        --repro-splits reproduced_splits \\
        --out-dir validation_output
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── Constants ────────────────────────────────────────────────────────
SPLIT_NAMES = ("train", "val", "test")
GROUP_NAMES = ("total", "bright", "dark", "vague", "lowres", "highres")

BG_TO_GROUP = {0: "bright", 1: "dark", 2: "vague", 3: "lowres"}

SRC_KEEP_CAT_IDS = {0, 2, 3}
SRC_EXCLUDE_CAT_IDS = {1, 4, 5, 6}
CAT_REMAP = {0: 0, 2: 1, 3: 2}
MAX_ANNOTATIONS = 100
DEFAULT_ASSIGNMENTS_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "split_assignments.json"
)

_RANGES = {
    "countable": [(309, 1302), (2712, 8709), (11761, 12617), (12994, 17417)],
    "uncountable": [(1303, 2088), (8710, 11737), (12618, 12731), (17418, 18000)],
}


def _is_countable(img_id: int) -> bool:
    return any(lo <= img_id <= hi for lo, hi in _RANGES["countable"])


def _is_uncountable(img_id: int) -> bool:
    return any(lo <= img_id <= hi for lo, hi in _RANGES["uncountable"])


def compute_curated_group_members(source: dict[str, Any]) -> tuple[set[int], dict[str, set[int]]]:
    """Recompute curated image ids and group membership from source annotations."""
    img_map = {img["id"]: img for img in source["images"]}
    src_ann_by_img: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in source["annotations"]:
        src_ann_by_img[ann["image_id"]].append(ann)

    curated_ids: set[int] = set()
    group_members: dict[str, set[int]] = {g: set() for g in GROUP_NAMES}

    for img_id in sorted(img_map):
        img = img_map[img_id]
        src_anns = src_ann_by_img.get(img_id, [])

        if img["items_count"] == 0 and len(src_anns) == 0:
            curated_ids.add(img_id)
            sub_group = BG_TO_GROUP[img["background_category_id"]]
            group_members["total"].add(img_id)
            group_members[sub_group].add(img_id)
            if sub_group != "lowres":
                group_members["highres"].add(img_id)
            continue
        if _is_uncountable(img_id):
            continue
        if not _is_countable(img_id):
            continue

        keep_anns = [a for a in src_anns if a["category_id"] in SRC_KEEP_CAT_IDS]
        exclude_anns = [a for a in src_anns if a["category_id"] in SRC_EXCLUDE_CAT_IDS]
        if exclude_anns:
            continue
        if len(keep_anns) == 0 or len(keep_anns) > MAX_ANNOTATIONS:
            continue

        curated_ids.add(img_id)
        sub_group = BG_TO_GROUP[img["background_category_id"]]
        group_members["total"].add(img_id)
        group_members[sub_group].add(img_id)
        if sub_group != "lowres":
            group_members["highres"].add(img_id)

    return curated_ids, group_members


def parse_split_filename(fname: str) -> tuple[str, str] | None:
    if fname.startswith("._") or not fname.endswith(".json"):
        return None
    stem = fname.removesuffix(".json")
    m = re.match(
        r"^(?P<group>total|bright|dark|vague|lowres|highres)_(?P<split>train|val|test)_coco$",
        stem,
    )
    if m:
        return m.group("split"), m.group("group")

    m = re.match(
        r"^(?P<group>total|bright|dark|vague|lowres|highres)_(?P<split>train|val|test)$",
        stem,
    )
    if m:
        return m.group("split"), m.group("group")

    m = re.match(
        r"^(?P<split>train|val|test)_(?P<group>total|bright|dark|vague|lowres|highres)100$",
        stem,
    )
    if m:
        return m.group("split"), m.group("group")

    m = re.match(
        r"^(?P<split>train|val|test)_annotated_(?P<group>total|bright|dark|vague|lowres|highres)100$",
        stem,
    )
    if m:
        return m.group("split"), m.group("group")

    return None


def load_splits(split_dir: Path) -> dict[tuple[str, str], dict]:
    splits: dict[tuple[str, str], dict] = {}
    for fname in sorted(os.listdir(split_dir)):
        parsed = parse_split_filename(fname)
        if parsed is None:
            continue
        split_name, group_name = parsed
        with open(split_dir / fname) as f:
            splits[(group_name, split_name)] = json.load(f)
    return splits


def compute_split_stats(data: dict) -> dict[str, Any]:
    """Compute statistics for a single split COCO dict."""
    img_ids = sorted(img["id"] for img in data["images"])
    ann_count = len(data["annotations"])

    cat_map = {c["id"]: c["name"] for c in data["categories"]}
    ann_per_class: Counter = Counter()
    for ann in data["annotations"]:
        ann_per_class[cat_map.get(ann["category_id"], str(ann["category_id"]))] += 1

    ann_per_img: Counter = Counter()
    for ann in data["annotations"]:
        ann_per_img[ann["image_id"]] += 1
    max_ann = max(ann_per_img.values()) if ann_per_img else 0

    return {
        "images": len(img_ids),
        "annotations": ann_count,
        "max_annotations_per_image": max_ann,
        "annotations_per_class": dict(sorted(ann_per_class.items())),
        "image_ids": img_ids,
        "annotation_ids": sorted(ann["id"] for ann in data["annotations"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate reproduced COCO splits.",
    )
    parser.add_argument("--raw-root", type=Path, required=True,
                        help="Path to AGAR dataset root.")
    parser.add_argument("--repro-splits", type=Path, required=True,
                        help="Path to reproduced splits directory.")
    parser.add_argument("--assignments", type=Path,
                        default=DEFAULT_ASSIGNMENTS_PATH,
                        help="Path to split_assignments.json (default: reproducibility_package/configs/split_assignments.json).")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("validation_output"),
                        help="Where to write reports.")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load assignments
    assignments_path = args.assignments.resolve()
    if not assignments_path.exists():
        print(f"ERROR: assignments file not found: {assignments_path}")
        sys.exit(1)
    with open(assignments_path) as f:
        assignments = json.load(f)

    # Load reproduced splits
    repro_dir = args.repro_splits.resolve()
    if not repro_dir.exists():
        print(f"ERROR: reproduced splits directory not found: {repro_dir}")
        print("  Run reproduce_splits.py --mode reconstruct first.")
        sys.exit(1)

    print("Loading reproduced splits ...")
    repro = load_splits(repro_dir)
    if len(repro) == 0:
        print(f"ERROR: no split files found in {repro_dir}")
        sys.exit(1)

    # Load source for cross-referencing
    raw_root = args.raw_root.resolve()
    print("Loading source annotations.json ...")
    src_path = raw_root / "dataset" / "annotations.json"
    if not src_path.exists():
        src_path = raw_root / "annotations.json"
    with open(src_path) as f:
        source = json.load(f)

    src_ann_by_img: dict[int, list[dict]] = defaultdict(list)
    for ann in source["annotations"]:
        src_ann_by_img[ann["image_id"]].append(ann)
    curated_ids, curated_group_members = compute_curated_group_members(source)

    errors: list[str] = []
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(raw_root),
        "repro_splits_dir": str(repro_dir),
        "assignments_file": str(assignments_path),
        "splits": {},
    }

    # ── File presence ────────────────────────────────────────────────
    expected_keys = {(g, s) for g in GROUP_NAMES for s in SPLIT_NAMES}
    missing_files = expected_keys - repro.keys()
    extra_files = repro.keys() - expected_keys
    if missing_files:
        for g, s in sorted(missing_files):
            errors.append(f"Missing: {g}_{s}_coco.json")
    if extra_files:
        for g, s in sorted(extra_files):
            errors.append(f"Extra: {g}_{s}_coco.json")

    # ── Per-split checks ─────────────────────────────────────────────
    print(f"\n{'Group':10s}  {'Split':6s}  {'Imgs':>6s}  {'Anns':>7s}  "
          f"{'MaxAnn':>6s}  {'IDMatch':>8s}  {'AnnMatch':>9s}")
    print("-" * 70)

    for group in GROUP_NAMES:
        for split in SPLIT_NAMES:
            key = (group, split)
            if key not in repro:
                continue

            stats = compute_split_stats(repro[key])

            # Check image membership against assignments
            repro_img_set = set(stats["image_ids"])

            # Derive expected membership from source curation + assignments
            val_assigned = set(assignments[group]["val"])
            test_assigned = set(assignments[group]["test"])
            group_members = curated_group_members[group]

            if split == "val":
                expected_ids = val_assigned
            elif split == "test":
                expected_ids = test_assigned
            else:  # train
                expected_ids = group_members - val_assigned - test_assigned

            id_match = repro_img_set == expected_ids
            if not id_match:
                missing = expected_ids - repro_img_set
                extra = repro_img_set - expected_ids
                errors.append(f"{group}/{split}: id mismatch "
                              f"(missing={len(missing)}, extra={len(extra)})")

            # Check annotations against source
            expected_ann_ids: set[int] = set()
            for img_id in repro_img_set:
                for ann in src_ann_by_img.get(img_id, []):
                    if ann["category_id"] in SRC_KEEP_CAT_IDS:
                        expected_ann_ids.add(ann["id"])
            repro_ann_set = set(stats["annotation_ids"])
            ann_match = repro_ann_set == expected_ann_ids
            if not ann_match:
                missing_a = expected_ann_ids - repro_ann_set
                extra_a = repro_ann_set - expected_ann_ids
                errors.append(f"{group}/{split}: annotation mismatch "
                              f"(missing={len(missing_a)}, extra={len(extra_a)})")

            id_status = "[OK]" if id_match else "[FAIL]"
            ann_status = "[OK]" if ann_match else "[FAIL]"
            print(f"  {group:10s}  {split:6s}  {stats['images']:>6,}  "
                  f"{stats['annotations']:>7,}  "
                  f"{stats['max_annotations_per_image']:>6}  "
                  f"{id_status:>8s}  {ann_status:>9s}")

            report["splits"][f"{group}/{split}"] = {
                "images": stats["images"],
                "annotations": stats["annotations"],
                "max_annotations_per_image": stats["max_annotations_per_image"],
                "annotations_per_class": stats["annotations_per_class"],
                "image_membership_match": id_match,
                "annotation_membership_match": ann_match,
            }

    # Ensure assignments are valid for this source annotations.json
    for group in GROUP_NAMES:
        val_assigned = set(assignments[group]["val"])
        test_assigned = set(assignments[group]["test"])
        not_curated = (val_assigned | test_assigned) - curated_ids
        not_in_group = (val_assigned | test_assigned) - curated_group_members[group]
        if not_curated:
            errors.append(
                f"{group}: assignments reference {len(not_curated)} ids outside curated set"
            )
        if not_in_group:
            errors.append(
                f"{group}: assignments reference {len(not_in_group)} ids outside group membership"
            )

    # ── No-overlap checks ────────────────────────────────────────────
    print(f"\n── No-overlap checks ──────────────────────────────────────")
    for group in GROUP_NAMES:
        id_sets: dict[str, set[int]] = {}
        for split in SPLIT_NAMES:
            key = (group, split)
            if key in repro:
                id_sets[split] = {img["id"] for img in repro[key]["images"]}
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            if a in id_sets and b in id_sets:
                overlap = id_sets[a] & id_sets[b]
                if overlap:
                    errors.append(f"Overlap {group} {a}∩{b}: {len(overlap)}")
                status = "[OK]" if not overlap else "[FAIL]"
                print(f"  {group} {a}∩{b}: {len(overlap)} overlap  {status}")

    # ── Category remap check ─────────────────────────────────────────
    print(f"\n── Category remap check ───────────────────────────────────")
    expected_cats = [
        {"supercategory": "microbes", "id": 0, "name": "S.aureus"},
        {"supercategory": "microbes", "id": 1, "name": "P.aeruginosa"},
        {"supercategory": "microbes", "id": 2, "name": "E.coli"},
    ]
    cat_ok = True
    for key, data in repro.items():
        if data["categories"] != expected_cats:
            group, split = key
            errors.append(f"{group}/{split}: wrong categories")
            cat_ok = False
    print(f"  All files have correct categories: {'[OK]' if cat_ok else '[FAIL]'}")

    # ── Finalize ─────────────────────────────────────────────────────
    report["errors"] = errors
    report["passed"] = len(errors) == 0

    # Write JSON report
    json_path = out_dir / "validation_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Write Markdown report
    md_path = out_dir / "validation_report.md"
    with open(md_path, "w") as f:
        f.write("# Split Reproduction Validation Report\n\n")
        f.write(f"Generated: {report['generated_at']}\n\n")
        f.write(f"- **Raw root**: `{report['raw_root']}`\n")
        f.write(f"- **Reproduced splits**: `{report['repro_splits_dir']}`\n")
        f.write(f"- **Assignments file**: `{report['assignments_file']}`\n\n")

        f.write("## Per-split results\n\n")
        f.write("| Group | Split | Images | Annotations | Max Ann/Img | "
                "ID Match | Ann Match |\n")
        f.write("|-------|-------|-------:|------------:|------------:|"
                ":---------:|:---------:|\n")
        for key_str, stats in sorted(report["splits"].items()):
            group, split = key_str.split("/")
            im = "[OK]" if stats["image_membership_match"] else "[FAIL]"
            am = "[OK]" if stats["annotation_membership_match"] else "[FAIL]"
            f.write(f"| {group} | {split} | {stats['images']:,} | "
                    f"{stats['annotations']:,} | "
                    f"{stats['max_annotations_per_image']} | {im} | {am} |\n")

        f.write("\n## Annotation counts per class (total group)\n\n")
        f.write("| Split | S.aureus | P.aeruginosa | E.coli | Total |\n")
        f.write("|-------|--------:|-----------:|------:|------:|\n")
        grand_total: Counter = Counter()
        for split_name in SPLIT_NAMES:
            key_str = f"total/{split_name}"
            if key_str in report["splits"]:
                s = report["splits"][key_str]
                apc = s["annotations_per_class"]
                sa = apc.get("S.aureus", 0)
                pa = apc.get("P.aeruginosa", 0)
                ec = apc.get("E.coli", 0)
                total = sa + pa + ec
                f.write(f"| {split_name} | {sa:,} | {pa:,} | {ec:,} | {total:,} |\n")
                for cls, cnt in apc.items():
                    grand_total[cls] += cnt
        gt = sum(grand_total.values())
        f.write(f"| **Total** | **{grand_total.get('S.aureus', 0):,}** | "
                f"**{grand_total.get('P.aeruginosa', 0):,}** | "
                f"**{grand_total.get('E.coli', 0):,}** | **{gt:,}** |\n")

        f.write("\n## Errors\n\n")
        if errors:
            for e in errors:
                f.write(f"- [ERROR] {e}\n")
        else:
            f.write("None — all checks passed [OK]\n")

        f.write(f"\n---\n**Result: {'PASS [OK]' if not errors else 'FAIL [ERROR]'}**\n")

    print(f"\n  Reports written to:")
    print(f"    {json_path}")
    print(f"    {md_path}")

    if errors:
        print(f"\n[ERROR]  {len(errors)} error(s):")
        for e in errors:
            print(f"    {e}")
        sys.exit(1)
    else:
        print(f"\n[OK]  VALIDATION PASSED – reproduced splits are correct")


if __name__ == "__main__":
    main()
