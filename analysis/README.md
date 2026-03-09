# Bounding Box Anchor Analysis

## Script

- `analysis/bbox_anchor_analysis.py`

## Purpose

Analyze COCO ground-truth bounding box distributions against Detectron2 RetinaNet default anchors:

- Anchor sizes: `32, 64, 128, 256, 512`
- Anchor areas: `1024, 4096, 16384, 65536, 262144`

The script reproduces the notebook flow:

1. Load AGAR + curated COCO test annotations.
2. Compute bbox statistics (overall and per class).
3. Compute anchor coverage (`%` below each anchor threshold).
4. Generate 4 figures.
5. Export CSV tables.
6. Print textual summaries and suggested anchor sizes.

## Environment

From repository root:

```bash
cd reproducibility_package
python -m pip install -r requirements.txt
```

## Run

```bash
python analysis/bbox_anchor_analysis.py \
  --agar-json reproduced_splits/total_test_coco.json \
  --curated-json /path/to/curated/test/_annotations.coco.json \
  --agar-label "AGAR Total" \
  --curated-label "Curated Test" \
  --out-dir analysis_results/bbox_anchor_analysis
```

Important:
- Use the reproduced COCO split files (`*_test_coco.json`) for AGAR input.

## Outputs

In `--out-dir`:

- `fig_area_histogram.png`
- `fig_width_height_scatter.png`
- `fig_area_cdf.png`
- `fig_area_boxplot.png`
- `bbox_stats_agar.csv`
- `bbox_stats_<curated_label>.csv`
- `anchor_coverage_agar.csv`
- `anchor_coverage_<curated_label>.csv`
