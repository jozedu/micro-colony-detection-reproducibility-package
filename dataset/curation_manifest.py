#!/usr/bin/env python3
"""
curation_manifest.py
====================
Produce a per-image curation manifest and summary for the AGAR dataset.

Outputs
-------
curation_manifest.jsonl
    One JSON object per line, one line per image in annotations.json.
    Each record contains:
      - image_id, file_name, background, sample_type
      - included (bool), reasons (list of strings)
      - annotation counts per category (source and curated)

curation_summary.json
    Aggregate counts: total images, included/excluded by reason,
    label distributions, per-background breakdowns.

Usage
-----
    python scripts/curation_manifest.py \\
        --raw-root /data/AGAR \\
        --out-dir curation_output
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── ID ranges ────────────────────────────────────────────────────────
_RANGES = {
    "empty": [(1, 308), (2089, 2711), (11738, 11760), (12732, 12993)],
    "countable": [(309, 1302), (2712, 8709), (11761, 12617), (12994, 17417)],
    "uncountable": [(1303, 2088), (8710, 11737), (12618, 12731), (17418, 18000)],
}

KEEP_CAT_IDS = {0, 2, 3}      # S.aureus, P.aeruginosa, E.coli
EXCLUDE_CAT_IDS = {1, 4, 5, 6}  # B.subtilis, C.albicans, Defect, Contamination

CAT_NAMES = {
    0: "S.aureus", 1: "B.subtilis", 2: "P.aeruginosa",
    3: "E.coli", 4: "C.albicans", 5: "Defect", 6: "Contamination",
}


def classify_sample_type(sid: int) -> str:
    for stype, ranges in _RANGES.items():
        for lo, hi in ranges:
            if lo <= sid <= hi:
                return stype
    return "unknown"


def load_source(raw_root: Path) -> dict:
    for candidate in [raw_root / "dataset" / "annotations.json",
                      raw_root / "annotations.json"]:
        if candidate.exists():
            with open(candidate) as f:
                return json.load(f)
    print(f"ERROR: annotations.json not found under {raw_root}")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Produce curation manifest.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("curation_output"))
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading annotations.json from {raw_root} ...")
    source = load_source(raw_root)

    bg_map = {bg["id"]: bg["name"] for bg in source.get("background_categories", [])}
    img_meta = {img["id"]: img for img in source["images"]}

    # Count annotations per image per category
    ann_counts: dict[int, Counter] = defaultdict(Counter)
    for ann in source["annotations"]:
        ann_counts[ann["image_id"]][ann["category_id"]] += 1

    # ── Build manifest ───────────────────────────────────────────────
    manifest_path = out_dir / "curation_manifest.jsonl"
    summary_counters: dict[str, int] = Counter()
    reason_counters: Counter = Counter()
    included_per_bg: Counter = Counter()
    excluded_per_bg: Counter = Counter()
    included_ann_dist: Counter = Counter()  # category_name → count
    included_images = 0
    excluded_images = 0

    print(f"Writing {manifest_path} ...")
    with open(manifest_path, "w") as f:
        for img_id in sorted(img_meta.keys()):
            img = img_meta[img_id]
            bg_name = bg_map.get(img["background_category_id"], "unknown")
            sample_type = classify_sample_type(img_id)
            cats = ann_counts.get(img_id, Counter())
            keep_count = sum(cats.get(c, 0) for c in KEEP_CAT_IDS)
            exclude_count = sum(cats.get(c, 0) for c in EXCLUDE_CAT_IDS)
            total_count = sum(cats.values())

            # Determine inclusion and reasons
            reasons: list[str] = []
            included = False

            # Empty rule is checked first regardless of sample_type:
            # items_count == 0 AND 0 total annotations → empty plate
            if img["items_count"] == 0 and total_count == 0:
                included = True
                reasons.append("empty_plate")
            elif sample_type == "uncountable":
                reasons.append("uncountable_sample")
            elif sample_type == "countable":
                if exclude_count > 0:
                    reasons.append("has_excluded_categories")
                    for cid in EXCLUDE_CAT_IDS:
                        if cats.get(cid, 0) > 0:
                            reasons.append(f"has_{CAT_NAMES[cid].replace('.', '_')}")
                elif keep_count == 0:
                    reasons.append("no_keep_annotations")
                elif keep_count > 100:
                    reasons.append("too_many_annotations")
                else:
                    included = True
                    reasons.append("curated_non_empty")
            else:
                reasons.append("unknown_sample_type")

            record = {
                "image_id": img_id,
                "file_name": f"{img_id}.jpg",
                "background": bg_name,
                "sample_type": sample_type,
                "included": included,
                "reasons": reasons,
                "items_count": img["items_count"],
                "total_annotations": total_count,
                "keep_annotations": keep_count,
                "exclude_annotations": exclude_count,
                "annotations_per_category": {
                    CAT_NAMES.get(cid, str(cid)): cnt
                    for cid, cnt in sorted(cats.items())
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Update summary counters
            if included:
                included_images += 1
                included_per_bg[bg_name] += 1
                for cid in KEEP_CAT_IDS:
                    included_ann_dist[CAT_NAMES[cid]] += cats.get(cid, 0)
            else:
                excluded_images += 1
                excluded_per_bg[bg_name] += 1

            for r in reasons:
                reason_counters[r] += 1

    # ── Build summary ────────────────────────────────────────────────
    summary = {
        "total_source_images": len(img_meta),
        "total_source_annotations": len(source["annotations"]),
        "source_categories": {str(cid): name for cid, name in sorted(CAT_NAMES.items())},
        "curation_rules": {
            "keep_categories": [CAT_NAMES[c] for c in sorted(KEEP_CAT_IDS)],
            "exclude_categories": [CAT_NAMES[c] for c in sorted(EXCLUDE_CAT_IDS)],
            "min_annotations": 1,
            "max_annotations": 100,
            "empty_rule": "items_count == 0 AND total_annotations == 0",
        },
        "included_images": included_images,
        "excluded_images": excluded_images,
        "included_non_empty": reason_counters.get("curated_non_empty", 0),
        "included_empty": reason_counters.get("empty_plate", 0),
        "exclusion_reasons": {
            reason: count
            for reason, count in reason_counters.most_common()
            if reason not in ("curated_non_empty", "empty_plate")
        },
        "included_annotations_per_category": dict(sorted(included_ann_dist.items())),
        "included_total_annotations": sum(included_ann_dist.values()),
        "included_per_background": dict(sorted(included_per_bg.items())),
        "excluded_per_background": dict(sorted(excluded_per_bg.items())),
    }

    summary_path = out_dir / "curation_summary.json"
    print(f"Writing {summary_path} ...")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n── Curation Summary ───────────────────────────────────────")
    print(f"  Source images      : {summary['total_source_images']:>7,}")
    print(f"  Source annotations : {summary['total_source_annotations']:>7,}")
    print(f"  Included images    : {summary['included_images']:>7,}")
    print(f"    non-empty        : {summary['included_non_empty']:>7,}")
    print(f"    empty            : {summary['included_empty']:>7,}")
    print(f"  Excluded images    : {summary['excluded_images']:>7,}")
    print(f"  Included annotations: {summary['included_total_annotations']:>7,}")
    print(f"\n  Exclusion reasons:")
    for reason, count in sorted(summary["exclusion_reasons"].items(),
                                key=lambda x: -x[1]):
        print(f"    {reason:40s}: {count:>5,}")

    print(f"\n[OK]  Manifest written to {out_dir}/")


if __name__ == "__main__":
    main()
