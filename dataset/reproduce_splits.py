#!/usr/bin/env python3
"""
reproduce_splits.py
===================
Reproduce the exact train / val / test COCO split JSON files used in the
paper, given a user's local copy of the original AGAR dataset.

The script ships with ``configs/split_assignments.json``, a compact file
(~127 KB) that records which image ids belong to **val** and **test** for
each of the 6 background groups.  Train membership is derived as the
complement (all curated images in the group minus val minus test).

The script does **not** guess random seeds.  It deterministically
rebuilds the COCO JSONs from the user's ``annotations.json`` using the
embedded split membership.

Modes
-----
reconstruct  – (default) write 18 COCO split JSONs identical to the ones
               used in the paper.
verify       – dry-run that checks the user's annotations.json is
               compatible with the assignments; no files are written.

Image key
---------
Each image is keyed by its **integer id** (== filename stem, e.g.
``315`` ↔ ``315.jpg``).  This is path-independent.

Curation rules (applied automatically)
---------------------------------------
1. *Uncountable* id-ranges → excluded.
2. Images with annotations from excluded categories (B. subtilis,
   C. albicans, Defect, Contamination) → excluded.
3. Images with > 100 keep-category annotations → excluded.
4. Empty plates (``items_count == 0`` AND 0 total annotations) → included
   in splits as valid images with zero annotations.
5. Remaining images with 1–100 keep-category annotations → included.

Usage
-----
    python dataset/reproduce_splits.py \\
        --raw-root /data/AGAR \\
        --out-dir reproduced_splits \\
        --mode reconstruct
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ── Constants ────────────────────────────────────────────────────────
SPLIT_NAMES = ("train", "val", "test")
GROUP_NAMES = ("total", "bright", "dark", "vague", "lowres", "highres")

# Source category ids → split category ids
SRC_KEEP_CAT_IDS = {0, 2, 3}
SRC_EXCLUDE_CAT_IDS = {1, 4, 5, 6}
CAT_REMAP = {0: 0, 2: 1, 3: 2}

SPLIT_CATEGORIES = [
    {"supercategory": "microbes", "id": 0, "name": "S.aureus"},
    {"supercategory": "microbes", "id": 1, "name": "P.aeruginosa"},
    {"supercategory": "microbes", "id": 2, "name": "E.coli"},
]

# ID ranges by sample type
_RANGES = {
    "countable": [(309, 1302), (2712, 8709), (11761, 12617), (12994, 17417)],
    "uncountable": [(1303, 2088), (8710, 11737), (12618, 12731), (17418, 18000)],
}

# Background category id → group name
BG_TO_GROUP = {0: "bright", 1: "dark", 2: "vague", 3: "lowres"}

MAX_ANNOTATIONS = 100
DEFAULT_ASSIGNMENTS_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "split_assignments.json"
)


# ── Helpers ──────────────────────────────────────────────────────────
def _is_countable(img_id: int) -> bool:
    return any(lo <= img_id <= hi for lo, hi in _RANGES["countable"])


def _is_uncountable(img_id: int) -> bool:
    return any(lo <= img_id <= hi for lo, hi in _RANGES["uncountable"])


def make_split_filename(split_name: str, group_name: str) -> str:
    return f"{group_name}_{split_name}_coco.json"


def load_source(raw_root: Path) -> dict:
    for candidate in [raw_root / "dataset" / "annotations.json",
                      raw_root / "annotations.json"]:
        if candidate.exists():
            print(f"  annotations.json: {candidate}")
            with open(candidate) as f:
                return json.load(f)
    print(f"ERROR: Cannot find annotations.json under {raw_root}")
    print(f"  Tried: {raw_root / 'dataset' / 'annotations.json'}")
    print(f"         {raw_root / 'annotations.json'}")
    sys.exit(1)


def load_assignments(assignments_path: Path) -> dict[str, dict[str, list[int]]]:
    if not assignments_path.exists():
        print(f"ERROR: Split assignments file not found: {assignments_path}")
        print(f"  This file ships with the repository under configs/.")
        sys.exit(1)
    with open(assignments_path) as f:
        return json.load(f)


def curate(source: dict) -> tuple[set[int], dict[int, dict], dict[int, list[dict]]]:
    """
    Apply curation rules.  Returns:
    - curated_ids: set of image ids that pass curation (includes empty plates)
    - img_map: id → image metadata (all 18 000)
    - ann_by_img: id → list of *kept* annotations (category ids remapped)
    """
    img_map = {img["id"]: img for img in source["images"]}
    src_ann_by_img: dict[int, list[dict]] = defaultdict(list)
    for ann in source["annotations"]:
        src_ann_by_img[ann["image_id"]].append(ann)

    curated_ids: set[int] = set()
    ann_by_img: dict[int, list[dict]] = {}

    for img_id in sorted(img_map):
        img = img_map[img_id]
        src_anns = src_ann_by_img.get(img_id, [])

        # Legacy split definition includes empty plates as valid split members.
        if img["items_count"] == 0 and len(src_anns) == 0:
            curated_ids.add(img_id)
            ann_by_img[img_id] = []
            continue

        # Rule: uncountable range → excluded
        if _is_uncountable(img_id):
            continue

        # Rule: must be in countable range
        if not _is_countable(img_id):
            continue

        # Collect keep / exclude annotations
        keep_anns = [a for a in src_anns if a["category_id"] in SRC_KEEP_CAT_IDS]
        exclude_anns = [a for a in src_anns if a["category_id"] in SRC_EXCLUDE_CAT_IDS]

        # Rule: any excluded-category annotation → excluded
        if exclude_anns:
            continue

        # Rule: 0 keep annotations (for non-empty images) → excluded
        if len(keep_anns) == 0:
            continue

        # Rule: > 100 keep annotations → excluded
        if len(keep_anns) > MAX_ANNOTATIONS:
            continue

        # Passed all rules — remap category ids
        remapped: list[dict] = []
        for a in keep_anns:
            ra = copy.deepcopy(a)
            ra["category_id"] = CAT_REMAP[a["category_id"]]
            remapped.append(ra)
        remapped.sort(key=lambda a: a["id"])

        curated_ids.add(img_id)
        ann_by_img[img_id] = remapped

    return curated_ids, img_map, ann_by_img


def assign_group_membership(
    curated_ids: set[int],
    img_map: dict[int, dict],
) -> dict[str, set[int]]:
    """Return group_name → set of curated image ids in that group."""
    groups: dict[str, set[int]] = {g: set() for g in GROUP_NAMES}

    for img_id in curated_ids:
        bg = img_map[img_id]["background_category_id"]
        sub_group = BG_TO_GROUP[bg]

        groups["total"].add(img_id)
        groups[sub_group].add(img_id)
        if sub_group != "lowres":
            groups["highres"].add(img_id)

    return groups


def derive_full_assignments(
    assignments: dict[str, dict[str, list[int]]],
    group_members: dict[str, set[int]],
) -> dict[str, dict[str, set[int]]]:
    """
    From the compact assignments (val + test ids only) derive complete
    {group → {train, val, test} → set[int]} mapping.
    Train = group members − val − test.
    """
    full: dict[str, dict[str, set[int]]] = {}
    for group in GROUP_NAMES:
        val_ids = set(assignments[group]["val"])
        test_ids = set(assignments[group]["test"])
        train_ids = group_members[group] - val_ids - test_ids
        full[group] = {"train": train_ids, "val": val_ids, "test": test_ids}
    return full


def build_coco(
    img_ids: set[int],
    img_map: dict[int, dict],
    ann_by_img: dict[int, list[dict]],
) -> dict[str, Any]:
    """Build a COCO-format dict for the given image ids."""
    sorted_ids = sorted(img_ids)
    images: list[dict] = []
    annotations: list[dict] = []

    for img_id in sorted_ids:
        src_img = img_map[img_id]
        anns = ann_by_img.get(img_id, [])
        images.append({
            "id": img_id,
            "license": src_img.get("license", 1),
            "file_name": f"{img_id}.jpg",
            "width": src_img["width"],
            "height": src_img["height"],
            "items_count": len(anns),
            "background_category_id": src_img["background_category_id"],
        })
        annotations.extend(anns)

    return {
        "images": images,
        "annotations": annotations,
        "categories": copy.deepcopy(SPLIT_CATEGORIES),
    }


# ── Verify mode ──────────────────────────────────────────────────────
def mode_verify(
    raw_root: Path,
    assignments_path: Path,
) -> bool:
    print("=" * 70)
    print("MODE: verify")
    print("=" * 70)

    print("\nLoading source data ...")
    source = load_source(raw_root)
    assignments = load_assignments(assignments_path)

    print("\nApplying curation rules ...")
    curated_ids, img_map, ann_by_img = curate(source)
    group_members = assign_group_membership(curated_ids, img_map)

    total_anns = sum(len(ann_by_img.get(iid, [])) for iid in curated_ids)
    print(f"  Curated images           : {len(curated_ids):,}")
    print(f"  Curated annotations      : {total_anns:,}")
    for g in GROUP_NAMES:
        print(f"    {g:10s}: {len(group_members[g]):>5,}")

    errors: list[str] = []

    # Derive full split assignments
    full = derive_full_assignments(assignments, group_members)

    # Check that every val/test id is a curated member of the correct group
    print(f"\n── Membership checks ──────────────────────────────────────")
    for group in GROUP_NAMES:
        for split in ("val", "test"):
            assigned = set(assignments[group][split])
            not_in_group = assigned - group_members[group]
            not_curated = assigned - curated_ids
            if not_in_group:
                errors.append(f"{group}/{split}: {len(not_in_group)} assigned ids "
                              f"not in group membership")
            if not_curated:
                errors.append(f"{group}/{split}: {len(not_curated)} assigned ids "
                              f"not in curated set")

        # No overlap
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            overlap = full[group][a] & full[group][b]
            if overlap:
                errors.append(f"{group}: {a}∩{b} = {len(overlap)} ids")

        # Union = group membership
        union = full[group]["train"] | full[group]["val"] | full[group]["test"]
        if union != group_members[group]:
            diff = union.symmetric_difference(group_members[group])
            errors.append(f"{group}: union of splits ≠ group membership "
                          f"({len(diff)} diff)")

    # Per-group/split counts
    print(f"\n  {'Group':10s}  {'Train':>6s}  {'Val':>6s}  {'Test':>6s}  {'Total':>6s}  {'Status'}")
    print(f"  {'-' * 50}")
    for group in GROUP_NAMES:
        tr = len(full[group]["train"])
        va = len(full[group]["val"])
        te = len(full[group]["test"])
        tot = tr + va + te
        ok = tot == len(group_members[group])
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {group:10s}  {tr:>6,}  {va:>6,}  {te:>6,}  {tot:>6,}  {status}")

    # Summary
    print(f"\n{'=' * 70}")
    if errors:
        print(f"[ERROR]  {len(errors)} error(s):")
        for e in errors:
            print(f"    {e}")
        return False
    else:
        print("[OK]  VERIFICATION PASSED")
        print("    All assignment ids map to curated images in correct groups.")
        print("    No overlaps.  Union of splits = group membership.")
        return True


# ── Reconstruct mode ────────────────────────────────────────────────
def mode_reconstruct(
    raw_root: Path,
    assignments_path: Path,
    out_dir: Path,
) -> None:
    print("=" * 70)
    print("MODE: reconstruct")
    print("=" * 70)

    # Step 1: verify
    print("\nStep 1/3: Verifying source data and assignments ...")
    ok = mode_verify(raw_root, assignments_path)
    if not ok:
        print("\nERROR: Verification failed. Cannot reconstruct.")
        sys.exit(1)

    # Step 2: curate + derive full assignments
    print(f"\nStep 2/3: Curating and computing split assignments ...")
    source = load_source(raw_root)
    assignments = load_assignments(assignments_path)
    curated_ids, img_map, ann_by_img = curate(source)
    group_members = assign_group_membership(curated_ids, img_map)
    full = derive_full_assignments(assignments, group_members)

    # Step 3: write COCO JSONs
    print(f"\nStep 3/3: Writing COCO split JSONs → {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for group in GROUP_NAMES:
        for split in SPLIT_NAMES:
            ids = full[group][split]
            coco = build_coco(ids, img_map, ann_by_img)
            fname = make_split_filename(split, group)
            out_path = out_dir / fname
            with open(out_path, "w") as f:
                json.dump(coco, f)

            n_img = len(coco["images"])
            n_ann = len(coco["annotations"])
            print(f"  {fname:45s}  imgs={n_img:>5,}  anns={n_ann:>6,}")

    print(f"\n[OK]  Wrote {len(GROUP_NAMES) * len(SPLIT_NAMES)} split files to {out_dir}/")


# ── CLI ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the exact AGAR paper split JSONs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--raw-root", type=Path, required=True,
        help="Path to user's local AGAR dataset root (contains dataset/).",
    )
    parser.add_argument(
        "--assignments", type=Path,
        default=DEFAULT_ASSIGNMENTS_PATH,
        help="Path to split_assignments.json (default: reproducibility_package/configs/split_assignments.json).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("reproduced_splits"),
        help="Where to write reconstructed JSONs (default: reproduced_splits).",
    )
    parser.add_argument(
        "--mode", choices=("verify", "reconstruct"), default="reconstruct",
        help="Operating mode (default: reconstruct).",
    )
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    assignments_path = args.assignments.resolve()
    out_dir = args.out_dir.resolve()

    if not raw_root.exists():
        print(f"ERROR: --raw-root does not exist: {raw_root}")
        sys.exit(1)

    if args.mode == "verify":
        ok = mode_verify(raw_root, assignments_path)
        sys.exit(0 if ok else 1)
    elif args.mode == "reconstruct":
        mode_reconstruct(raw_root, assignments_path, out_dir)


if __name__ == "__main__":
    main()
