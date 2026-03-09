#!/usr/bin/env python3
"""
bbox_anchor_analysis.py
=======================
Analyze bounding-box size distributions against Detectron2 RetinaNet default
anchor sizes.

Pipeline:
1) Load COCO annotations for AGAR and curated datasets.
2) Compute bbox statistics (overall and per class).
3) Compute anchor coverage (% boxes below each anchor threshold).
4) Generate plots:
   - fig_area_histogram.png
   - fig_width_height_scatter.png
   - fig_area_cdf.png
   - fig_area_boxplot.png
5) Export CSV tables.
6) Print summary and suggested anchor sizes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    import matplotlib

    if "MPLBACKEND" not in os.environ:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import seaborn as sns
except ImportError:
    sns = None


# Detectron2 RetinaNet default anchor configuration
ANCHOR_SIZES = [32, 64, 128, 256, 512]
ANCHOR_ASPECT_RATIOS = [0.5, 1.0, 2.0]
ANCHOR_AREAS = [size**2 for size in ANCHOR_SIZES]

CLASS_COLORS = ["#e74c3c", "#3498db", "#2ecc71"]


def require_dependencies() -> None:
    missing: list[str] = []
    if pd is None:
        missing.append("pandas")
    if np is None:
        missing.append("numpy")
    if plt is None:
        missing.append("matplotlib")
    if sns is None:
        missing.append("seaborn")
    if missing:
        raise ImportError(
            "Missing dependencies: "
            + ", ".join(missing)
            + ". Install with: python -m pip install pandas numpy matplotlib seaborn"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze COCO bbox distributions vs Detectron2 RetinaNet anchors."
    )
    parser.add_argument(
        "--agar-json",
        type=Path,
        required=True,
        help="COCO JSON for AGAR test set (for example reproduced_splits/total_test_coco.json).",
    )
    parser.add_argument(
        "--curated-json",
        type=Path,
        required=True,
        help="COCO JSON for curated test set.",
    )
    parser.add_argument(
        "--agar-label",
        default="AGAR Total",
        help="Display label for AGAR dataset.",
    )
    parser.add_argument(
        "--curated-label",
        default="Curated Test",
        help="Display label for curated dataset.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis_results/bbox_anchor_analysis"),
        help="Output directory for figures and CSVs.",
    )
    parser.add_argument(
        "--n-suggested-anchors",
        type=int,
        default=5,
        help="Number of suggested anchors based on percentiles.",
    )
    return parser.parse_args()


def ensure_file(path: Path, label: str) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def sanitize_label(label: str) -> str:
    return (
        label.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "")
        .replace("/", "_")
    )


def configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-paper")
    sns.set_palette("husl")
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["legend.fontsize"] = 9


def load_bbox_data(coco_json_path: Path, dataset_name: str) -> pd.DataFrame:
    """
    Load COCO annotations and extract bbox statistics.

    Returns columns:
    - dataset, category_id, category_name
    - bbox_x, bbox_y, bbox_w, bbox_h, bbox_area
    - img_w, img_h, img_area
    - bbox_w_rel, bbox_h_rel, bbox_area_rel
    - aspect_ratio
    """
    print(f"\nLoading {dataset_name}: {coco_json_path.name}")

    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    cat_lookup = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    img_lookup = {img["id"]: img for img in coco_data["images"]}

    rows: list[dict[str, float | int | str]] = []
    for ann in coco_data["annotations"]:
        image = img_lookup[ann["image_id"]]
        img_w = float(image["width"])
        img_h = float(image["height"])
        img_area = img_w * img_h

        x, y, w, h = ann["bbox"]
        bbox_area = float(w) * float(h)

        rows.append(
            {
                "dataset": dataset_name,
                "category_id": int(ann["category_id"]),
                "category_name": str(cat_lookup[ann["category_id"]]),
                "bbox_x": float(x),
                "bbox_y": float(y),
                "bbox_w": float(w),
                "bbox_h": float(h),
                "bbox_area": bbox_area,
                "img_w": img_w,
                "img_h": img_h,
                "img_area": img_area,
                "bbox_w_rel": float(w) / img_w if img_w > 0 else 0.0,
                "bbox_h_rel": float(h) / img_h if img_h > 0 else 0.0,
                "bbox_area_rel": bbox_area / img_area if img_area > 0 else 0.0,
                "aspect_ratio": float(w) / float(h) if h > 0 else 0.0,
            }
        )

    df = pd.DataFrame(rows)

    print(f"  Total annotations: {len(df):,}")
    print(f"  Categories: {df['category_name'].nunique()}")
    print("  Per-class counts:")
    for cat_name, count in df["category_name"].value_counts().items():
        print(f"    {cat_name}: {count:,}")

    return df


def compute_bbox_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute detailed bbox statistics for overall and per class.
    """
    metrics = [
        "bbox_w",
        "bbox_h",
        "bbox_area",
        "bbox_w_rel",
        "bbox_h_rel",
        "bbox_area_rel",
        "aspect_ratio",
    ]

    stats_rows: list[dict[str, float | int | str]] = []

    for metric in metrics:
        values = df[metric]
        stats_rows.append(
            {
                "class": "Overall",
                "metric": metric,
                "count": len(values),
                "min": values.min(),
                "p05": values.quantile(0.05),
                "p25": values.quantile(0.25),
                "median": values.median(),
                "mean": values.mean(),
                "p75": values.quantile(0.75),
                "p95": values.quantile(0.95),
                "max": values.max(),
                "std": values.std(),
            }
        )

    for class_name in sorted(df["category_name"].unique()):
        class_df = df[df["category_name"] == class_name]
        for metric in metrics:
            values = class_df[metric]
            stats_rows.append(
                {
                    "class": class_name,
                    "metric": metric,
                    "count": len(values),
                    "min": values.min(),
                    "p05": values.quantile(0.05),
                    "p25": values.quantile(0.25),
                    "median": values.median(),
                    "mean": values.mean(),
                    "p75": values.quantile(0.75),
                    "p95": values.quantile(0.95),
                    "max": values.max(),
                    "std": values.std(),
                }
            )

    return pd.DataFrame(stats_rows)


def compute_anchor_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute percentage of boxes below each anchor area threshold.
    """
    coverage_rows: list[dict[str, float | int | str]] = []

    for anchor_size, anchor_area in zip(ANCHOR_SIZES, ANCHOR_AREAS):
        n_below = int((df["bbox_area"] < anchor_area).sum())
        coverage_rows.append(
            {
                "class": "Overall",
                "anchor_size": anchor_size,
                "anchor_area": anchor_area,
                "n_below": n_below,
                "n_total": len(df),
                "pct_below": 100.0 * n_below / len(df),
            }
        )

    for class_name in sorted(df["category_name"].unique()):
        class_df = df[df["category_name"] == class_name]
        for anchor_size, anchor_area in zip(ANCHOR_SIZES, ANCHOR_AREAS):
            n_below = int((class_df["bbox_area"] < anchor_area).sum())
            coverage_rows.append(
                {
                    "class": class_name,
                    "anchor_size": anchor_size,
                    "anchor_area": anchor_area,
                    "n_below": n_below,
                    "n_total": len(class_df),
                    "pct_below": 100.0 * n_below / len(class_df),
                }
            )

    return pd.DataFrame(coverage_rows)


def print_statistics_tables(stats_a: pd.DataFrame, stats_b: pd.DataFrame, label_b: str) -> None:
    display_cols = ["class", "metric", "count", "min", "p05", "p25", "median", "mean", "p75", "p95", "max"]
    absolute_metrics = ["bbox_w", "bbox_h", "bbox_area"]
    relative_metrics = ["bbox_w_rel", "bbox_h_rel", "bbox_area_rel", "aspect_ratio"]

    print("=" * 70)
    print("AGAR DATASET - BOUNDING BOX STATISTICS (Pixels)")
    print("=" * 70)
    print(stats_a[stats_a["metric"].isin(absolute_metrics)][display_cols].to_string(index=False))

    print("\n" + "=" * 70)
    print(f"{label_b.upper()} - BOUNDING BOX STATISTICS (Pixels)")
    print("=" * 70)
    print(stats_b[stats_b["metric"].isin(absolute_metrics)][display_cols].to_string(index=False))

    print("\n" + "=" * 70)
    print("AGAR DATASET - RELATIVE SIZE STATISTICS (Ratio to Image)")
    print("=" * 70)
    print(stats_a[stats_a["metric"].isin(relative_metrics)][display_cols].to_string(index=False))

    print("\n" + "=" * 70)
    print(f"{label_b.upper()} - RELATIVE SIZE STATISTICS (Ratio to Image)")
    print("=" * 70)
    print(stats_b[stats_b["metric"].isin(relative_metrics)][display_cols].to_string(index=False))


def print_anchor_tables(coverage_a: pd.DataFrame, coverage_b: pd.DataFrame, label_b: str) -> None:
    print("=" * 80)
    print("AGAR DATASET - ANCHOR COVERAGE ANALYSIS")
    print("Percentage of bounding boxes BELOW each anchor size threshold")
    print("=" * 80)
    print(coverage_a.to_string(index=False))

    print("\n" + "=" * 80)
    print(f"{label_b.upper()} - ANCHOR COVERAGE ANALYSIS")
    print("Percentage of bounding boxes BELOW each anchor size threshold")
    print("=" * 80)
    print(coverage_b.to_string(index=False))


def save_area_histogram(df_a: pd.DataFrame, label_a: str, df_b: pd.DataFrame, label_b: str, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    datasets = [(df_a, label_a), (df_b, label_b)]

    for ax, (df, name) in zip(axes, datasets):
        class_names_sorted = sorted(df["category_name"].unique())
        for i, class_name in enumerate(class_names_sorted):
            class_data = df[df["category_name"] == class_name]["bbox_area"]
            color = CLASS_COLORS[i % len(CLASS_COLORS)]
            ax.hist(
                class_data,
                bins=50,
                alpha=0.6,
                label=class_name,
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )

        for anchor_size, anchor_area in zip(ANCHOR_SIZES, ANCHOR_AREAS):
            ax.axvline(anchor_area, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
            ax.text(
                anchor_area,
                ax.get_ylim()[1] * 0.95,
                f"{anchor_size}^2",
                rotation=90,
                va="top",
                ha="right",
                fontsize=8,
                color="red",
            )

        ax.axvspan(0, ANCHOR_AREAS[0], alpha=0.15, color="red", label="Below smallest anchor")
        ax.set_xlabel("Bounding Box Area (pixels^2)")
        ax.set_ylabel("Count")
        ax.set_title(f"{name}\nBBox Area Distribution vs Anchor Sizes")
        ax.set_xscale("log")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / "fig_area_histogram.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {fig_path}")


def save_width_height_scatter(
    df_a: pd.DataFrame, label_a: str, df_b: pd.DataFrame, label_b: str, out_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    datasets = [(df_a, label_a), (df_b, label_b)]

    for ax, (df, name) in zip(axes, datasets):
        class_names_sorted = sorted(df["category_name"].unique())
        for i, class_name in enumerate(class_names_sorted):
            class_data = df[df["category_name"] == class_name]
            color = CLASS_COLORS[i % len(CLASS_COLORS)]
            ax.scatter(
                class_data["bbox_w"],
                class_data["bbox_h"],
                alpha=0.4,
                s=10,
                label=class_name,
                color=color,
            )

        for anchor_size in ANCHOR_SIZES:
            for aspect_ratio in ANCHOR_ASPECT_RATIOS:
                w = anchor_size * np.sqrt(aspect_ratio)
                h = anchor_size / np.sqrt(aspect_ratio)
                ax.plot(w, h, "rx", markersize=6, alpha=0.5, markeredgewidth=1.5)

        max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, linewidth=1, label="AR=1")

        ax.set_xlabel("Bounding Box Width (pixels)")
        ax.set_ylabel("Bounding Box Height (pixels)")
        ax.set_title(f"{name}\nBBox Width vs Height\n(Red crosses: anchor boxes at different aspect ratios)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    plt.tight_layout()
    fig_path = out_dir / "fig_width_height_scatter.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {fig_path}")


def save_area_cdf(df_a: pd.DataFrame, label_a: str, df_b: pd.DataFrame, label_b: str, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    datasets = [(df_a, label_a), (df_b, label_b)]

    for ax, (df, name) in zip(axes, datasets):
        class_names_sorted = sorted(df["category_name"].unique())
        for i, class_name in enumerate(class_names_sorted):
            class_data = df[df["category_name"] == class_name]["bbox_area"].sort_values()
            y = np.arange(1, len(class_data) + 1) / len(class_data) * 100.0
            color = CLASS_COLORS[i % len(CLASS_COLORS)]
            ax.plot(class_data, y, label=class_name, color=color, linewidth=2)

        overall_data = df["bbox_area"].sort_values()
        y = np.arange(1, len(overall_data) + 1) / len(overall_data) * 100.0
        ax.plot(overall_data, y, label="Overall", color="black", linewidth=2, linestyle=":")

        for anchor_size, anchor_area in zip(ANCHOR_SIZES, ANCHOR_AREAS):
            ax.axvline(anchor_area, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
            pct_below = (df["bbox_area"] < anchor_area).sum() / len(df) * 100.0
            ax.text(anchor_area, pct_below - 5, f"{pct_below:.1f}%", fontsize=8, color="red", ha="left")

        ax.set_xlabel("Bounding Box Area (pixels^2)")
        ax.set_ylabel("Cumulative Percentage (%)")
        ax.set_title(f"{name}\nCumulative Distribution of BBox Areas")
        ax.set_xscale("log")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])

    plt.tight_layout()
    fig_path = out_dir / "fig_area_cdf.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {fig_path}")


def save_area_boxplot(df_a: pd.DataFrame, label_a: str, df_b: pd.DataFrame, label_b: str, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    datasets = [(df_a, label_a), (df_b, label_b)]

    for ax, (df, name) in zip(axes, datasets):
        class_names_sorted = sorted(df["category_name"].unique())
        data_for_boxplot = [df[df["category_name"] == cn]["bbox_area"].values for cn in class_names_sorted]

        try:
            bp = ax.boxplot(
                data_for_boxplot,
                tick_labels=class_names_sorted,
                patch_artist=True,
                boxprops={"facecolor": "lightblue", "alpha": 0.7},
                medianprops={"color": "red", "linewidth": 2},
                whiskerprops={"linewidth": 1.5},
                capprops={"linewidth": 1.5},
            )
        except TypeError:
            bp = ax.boxplot(
                data_for_boxplot,
                labels=class_names_sorted,
                patch_artist=True,
                boxprops={"facecolor": "lightblue", "alpha": 0.7},
                medianprops={"color": "red", "linewidth": 2},
                whiskerprops={"linewidth": 1.5},
                capprops={"linewidth": 1.5},
            )

        for patch, color in zip(bp["boxes"], CLASS_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        for anchor_size, anchor_area in zip(ANCHOR_SIZES, ANCHOR_AREAS):
            ax.axhline(anchor_area, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
            ax.text(len(class_names_sorted) + 0.3, anchor_area, f"{anchor_size}^2", va="center", fontsize=8, color="red")

        ax.set_ylabel("Bounding Box Area (pixels^2)")
        ax.set_title(f"{name}\nPer-Class BBox Area Distribution")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticklabels(class_names_sorted, rotation=15, ha="right")

    plt.tight_layout()
    fig_path = out_dir / "fig_area_boxplot.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {fig_path}")


def print_summary(df: pd.DataFrame, coverage_df: pd.DataFrame, dataset_name: str) -> None:
    """
    Print formatted summary of anchor coverage analysis.
    """
    print(f"\n{'=' * 70}")
    print(f"{dataset_name.upper()} - SUMMARY")
    print(f"{'=' * 70}")

    median_area = df["bbox_area"].median()
    mean_area = df["bbox_area"].mean()
    smallest_anchor = ANCHOR_AREAS[0]

    overall_coverage = coverage_df[
        (coverage_df["class"] == "Overall") & (coverage_df["anchor_size"] == ANCHOR_SIZES[0])
    ].iloc[0]
    pct_below_smallest = overall_coverage["pct_below"]

    print(f"\nTotal annotations: {len(df):,}")
    print(
        f"Median colony area: {median_area:.1f} px^2 "
        f"({100.0 * median_area / smallest_anchor:.1f}% of smallest anchor)"
    )
    print(f"Mean colony area: {mean_area:.1f} px^2 ({100.0 * mean_area / smallest_anchor:.1f}% of smallest anchor)")
    print(
        f"\n{pct_below_smallest:.1f}% of all annotations have area below the smallest "
        f"RetinaNet anchor ({smallest_anchor} px^2)"
    )

    print("\nPer-class breakdown (% below smallest anchor):")
    for class_name in sorted(df["category_name"].unique()):
        class_coverage = coverage_df[
            (coverage_df["class"] == class_name) & (coverage_df["anchor_size"] == ANCHOR_SIZES[0])
        ]
        if len(class_coverage) > 0:
            pct = class_coverage.iloc[0]["pct_below"]
            class_median = df[df["category_name"] == class_name]["bbox_area"].median()
            print(f"  {class_name:20s}: {pct:5.1f}%  (median area: {class_median:6.1f} px^2)")

    print("\nCoverage by anchor level (% of boxes below each threshold):")
    for anchor_size, anchor_area in zip(ANCHOR_SIZES, ANCHOR_AREAS):
        level_coverage = coverage_df[
            (coverage_df["class"] == "Overall") & (coverage_df["anchor_size"] == anchor_size)
        ].iloc[0]
        pct = level_coverage["pct_below"]
        print(f"  {anchor_size}x{anchor_size} ({anchor_area:6} px^2): {pct:5.1f}%")


def suggest_optimal_anchors(df: pd.DataFrame, dataset_name: str, n_anchors: int = 5) -> list[int]:
    """
    Suggest anchor sizes based on bbox-area percentiles.
    """
    print(f"\n{'=' * 70}")
    print(f"{dataset_name.upper()} - SUGGESTED OPTIMAL ANCHOR SIZES")
    print(f"{'=' * 70}")

    areas = df["bbox_area"].values
    percentiles = np.linspace(10, 90, n_anchors)

    suggested_areas = [np.percentile(areas, p) for p in percentiles]
    suggested_sizes = [int(np.sqrt(area)) for area in suggested_areas]

    print(f"\nCurrent Detectron2 defaults: {ANCHOR_SIZES}")
    print(f"Suggested anchors (based on {percentiles[0]:.0f}th-{percentiles[-1]:.0f}th percentiles):")

    for i, (size, area, pct) in enumerate(zip(suggested_sizes, suggested_areas, percentiles), start=1):
        n_below = int((df["bbox_area"] < area).sum())
        pct_below = 100.0 * n_below / len(df)
        print(f"  Anchor {i}: {size:3d}x{size:3d} = {area:6.0f} px^2  (p{pct:.0f}, covers {pct_below:.1f}% of boxes)")

    print("\nKey observations:")
    print(f"  - Smallest suggested anchor: {suggested_sizes[0]}x{suggested_sizes[0]} = {suggested_areas[0]:.0f} px^2")
    print(f"  - Default smallest anchor: {ANCHOR_SIZES[0]}x{ANCHOR_SIZES[0]} = {ANCHOR_AREAS[0]} px^2")
    print(f"  - Reduction factor: {ANCHOR_AREAS[0] / suggested_areas[0]:.1f}x smaller anchors needed")

    median_area = df["bbox_area"].median()
    median_size = int(np.sqrt(median_area))
    print(f"  - Median colony size: {median_size}x{median_size} = {median_area:.0f} px^2")

    return suggested_sizes


def export_tables(
    stats_agar: pd.DataFrame,
    stats_curated: pd.DataFrame,
    coverage_agar: pd.DataFrame,
    coverage_curated: pd.DataFrame,
    curated_label: str,
    out_dir: Path,
) -> None:
    stats_agar_path = out_dir / "bbox_stats_agar.csv"
    stats_agar.to_csv(stats_agar_path, index=False)
    print(f"[OK] Saved: {stats_agar_path}")

    curated_slug = sanitize_label(curated_label)
    stats_curated_path = out_dir / f"bbox_stats_{curated_slug}.csv"
    stats_curated.to_csv(stats_curated_path, index=False)
    print(f"[OK] Saved: {stats_curated_path}")

    coverage_agar_path = out_dir / "anchor_coverage_agar.csv"
    coverage_agar.to_csv(coverage_agar_path, index=False)
    print(f"[OK] Saved: {coverage_agar_path}")

    coverage_curated_path = out_dir / f"anchor_coverage_{curated_slug}.csv"
    coverage_curated.to_csv(coverage_curated_path, index=False)
    print(f"[OK] Saved: {coverage_curated_path}")

    print(f"\n{'=' * 70}")
    print("ALL OUTPUTS SAVED")
    print(f"{'=' * 70}")
    print(f"Directory: {out_dir}")
    print("\nFiles:")
    for file_path in sorted(out_dir.iterdir()):
        print(f"  - {file_path.name}")


def main() -> None:
    args = parse_args()
    require_dependencies()
    args.agar_json = ensure_file(args.agar_json.resolve(), "AGAR JSON")
    args.curated_json = ensure_file(args.curated_json.resolve(), "Curated JSON")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_plot_style()

    print(f"Output directory: {out_dir}")
    print(f"Anchor sizes: {ANCHOR_SIZES} pixels")
    print(f"Anchor areas: {ANCHOR_AREAS} px^2")
    print(f"Smallest anchor: {ANCHOR_SIZES[0]}x{ANCHOR_SIZES[0]} = {ANCHOR_AREAS[0]} px^2")

    df_agar = load_bbox_data(args.agar_json, args.agar_label)
    df_curated = load_bbox_data(args.curated_json, args.curated_label)
    print("\n" + "=" * 70)
    print("DATASETS LOADED SUCCESSFULLY")
    print("=" * 70)

    print("Computing statistics for AGAR dataset...")
    stats_agar = compute_bbox_statistics(df_agar)
    print("Computing statistics for curated dataset...")
    stats_curated = compute_bbox_statistics(df_curated)
    print("\n[OK] Statistics computed")

    print_statistics_tables(stats_agar, stats_curated, args.curated_label)

    print("Computing anchor coverage analysis...")
    coverage_agar = compute_anchor_coverage(df_agar)
    coverage_curated = compute_anchor_coverage(df_curated)
    print("\n[OK] Anchor coverage computed")

    print_anchor_tables(coverage_agar, coverage_curated, args.curated_label)

    save_area_histogram(df_agar, args.agar_label, df_curated, args.curated_label, out_dir)
    save_width_height_scatter(df_agar, args.agar_label, df_curated, args.curated_label, out_dir)
    save_area_cdf(df_agar, args.agar_label, df_curated, args.curated_label, out_dir)
    save_area_boxplot(df_agar, args.agar_label, df_curated, args.curated_label, out_dir)

    print_summary(df_agar, coverage_agar, args.agar_label)
    print_summary(df_curated, coverage_curated, args.curated_label)

    suggest_optimal_anchors(df_agar, args.agar_label, args.n_suggested_anchors)
    suggest_optimal_anchors(df_curated, args.curated_label, args.n_suggested_anchors)

    export_tables(
        stats_agar=stats_agar,
        stats_curated=stats_curated,
        coverage_agar=coverage_agar,
        coverage_curated=coverage_curated,
        curated_label=args.curated_label,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
