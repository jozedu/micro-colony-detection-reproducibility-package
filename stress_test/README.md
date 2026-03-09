# Detectron2 Stress Test

## Purpose

Run high-density AGAR stress evaluation for Detectron2 models using COCOeval metrics.

Pipeline:

1. Build extended curated cohort without the `<=100` annotation cap.
2. Build stress subsets by density bins (`101-150`, `151-300`) plus combined `gt100`.
3. Check leakage against reproduced train/val splits.
4. Evaluate selected Detectron2 runs on stress subsets.
5. Export long and wide CSV reports.

## Script

- `stress_test/run_stress_test_detectron2.py`

## Environment

From repo root:

```bash
cd reproducibility_package
```

Install dependencies for this stage:

```bash
python -m pip install -r requirements.txt
```

Detectron2 evaluation requires the same `torch` + `detectron2` environment used for Detectron2 training/evaluation scripts.

## Inputs

Required in `--mode run`:

- `--full-coco-json`: AGAR source `annotations.json`
- `--image-dir`: AGAR images folder
- `--train-coco-json`: reproduced train COCO JSON (format + leakage reference)
- `--val-coco-json`: reproduced val COCO JSON (leakage reference)
- `--runs-root`: root folder containing Detectron2 run folders with `model_final.pth`

## Run Stress Evaluation

```bash
python stress_test/run_stress_test_detectron2.py \
  --mode run \
  --full-coco-json /path/to/AGAR/dataset/annotations.json \
  --image-dir /path/to/AGAR/dataset/images \
  --train-coco-json reproduced_splits/total_train_coco.json \
  --val-coco-json reproduced_splits/total_val_coco.json \
  --runs-root outputs_detectron2 \
  --out-dir stress_test_results \
  --thresholds 0.0,0.25,0.5,0.75 \
  --dets-per-image 300 \
  --evaluate-all
```

Defaults aligned with the stress-test runs:

- `--thresholds 0.0,0.25,0.5,0.75`
- `--dets-per-image 300`
- `--bins 101-150,151-300`

## Model Selection Options

Evaluate a specific list from discovered indices:

```bash
--selected-indices 0,1,2,3
```

Filter by model metadata parsed from run-folder names:

```bash
--filter-family faster_rcnn|retinanet
--filter-backbone R_50|R_101
--filter-subset bright|dark|vague|lowres|total|curated
--run-name-contains <substring>
```

## Resume Interrupted Evaluations

```bash
--resume --resume-model-index 2 --resume-test-set 151-300 --resume-threshold 0.5
```

## Reconstruct CSV Reports from Existing Stress Eval Folders

```bash
python stress_test/run_stress_test_detectron2.py \
  --mode reconstruct \
  --runs-root outputs_detectron2 \
  --out-dir stress_test_results \
  --dets-per-image 300
```

This scans `*stress_test_eval_*` directories and rebuilds CSV reports.

## Outputs

Under `--out-dir`:

- `extended_cohort_no_max.json`
- `extended_cohort_counts.json`
- `density_stats_gt100.json`
- `leakage_check_report.json`
- `stress_test_subsets/stress_test_101-150.json`
- `stress_test_subsets/stress_test_151-300.json`
- `stress_test_subsets/stress_test_gt100_combined.json`
- `csv_reports/300_stress_test_results_all.csv`
- `csv_reports/stress_test_results_wide.csv`
- `stress_test_run_manifest.json`

Per model run folder:

- `<model_dir>/300stress_test_eval_<timestamp>/`
- `metrics_<subset>_thr<thr>.json`
- `coco_instances_<subset>_thr<thr>.json`
- `eval_output_<subset>_thr<thr>.txt`
- `config_inference_<subset>_thr<thr>.yaml`
