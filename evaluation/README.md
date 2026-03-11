# Evaluation Scripts

Companion archived model/evaluation payload:
- Zenodo DOI: <https://doi.org/10.5281/zenodo.18922895>

Use the Zenodo record to inspect the archived paper checkpoints and stored
evaluation outputs. Use the commands in this README when you want to rerun the
evaluation workflows locally.

## Files

- `evaluation/evaluate_detectron2_outputs.py`: Detectron2 evaluation workflow.
- `evaluation/evaluate_yolov8_coco.py`: YOLOv8 COCOeval workflow using `model.val(..., save_json=True)` (lower RAM pressure, aligned with the notebook workflow).
- `evaluation/benchmark_inference_speed.py`: inference-speed benchmark for Detectron2 and YOLO reviewer payloads.
- `stress_test/run_stress_test_detectron2.py`: Detectron2 high-density stress-test workflow.

## Environment

From repository root:

```bash
cd reproducibility_package
```

Install dependencies in one environment:

```bash
python -m pip install -r requirements.txt
```

For Detectron2 evaluation, use the same `torch` + `detectron2` versions as training.

## Detectron2: Evaluate All Runs

```bash
python evaluation/evaluate_detectron2_outputs.py \
  --mode evaluate-runs \
  --runs-root outputs_detectron2 \
  --repro-splits reproduced_splits \
  --agar-images-dir /path/to/AGAR/dataset/images \
  --curated-test-json /path/to/curated/test/_annotations.coco.json \
  --curated-images-dir /path/to/curated/test \
  --report-dir eval_reports \
  --filter-subset all \
  --thresholds 0.0,0.25,0.5,0.75
```

`--filter-subset` is required in `evaluate-runs` mode.

```bash
--filter-subset bright              # one subset
--filter-subset agar                # group: total, bright, dark, vague, lowres
--filter-subset all                 # all subsets including curated
--filter-subset bright,dark,vague   # custom subset group
```

Model filters in `evaluate-runs` mode:

```bash
--filter-model-family faster        # one family
--filter-model-family retina        # one family
--filter-model-family faster,retina # both families
--filter-model-family all           # default

--filter-backbone 50                # one backbone
--filter-backbone 101               # one backbone
--filter-backbone 50,101            # both backbones
--filter-backbone all               # default
```

Worker behavior:

```bash
--num-workers 2                                  # default behavior: keep requested value
--force-workers-zero-with-pillow-shim            # optional safety fallback to force workers=0
```

## Detectron2: Cross-Subset

```bash
python evaluation/evaluate_detectron2_outputs.py \
  --mode cross-subset \
  --runs-root outputs_detectron2 \
  --repro-splits reproduced_splits \
  --agar-images-dir /path/to/AGAR/dataset/images \
  --target-model total_100_faster_rcnn_R_101_FPN_3x \
  --cross-threshold 0.0 \
  --cross-subsets bright,dark,vague,lowres
```

## YOLOv8 + COCOeval: Evaluate Runs

This script runs YOLO inference, converts predictions to COCO-format detections,
applies the category-id fix when needed, and evaluates with `pycocotools.COCOeval`.

Default run filter is `--seeds 42` (matches the evaluation runs reported in the paper). To evaluate all seeds, pass `--seeds ""`.

AGAR example:

```bash
python evaluation/evaluate_yolov8_coco.py \
  --mode evaluate-runs \
  --runs-root outputs_yolov8 \
  --data-yaml reproduced_yolo/yolo_agar_total/data.yaml \
  --gt-json reproduced_splits/total_test_coco.json \
  --subset-label total \
  --dataset-tag agar100 \
  --report-dir eval_reports
```

Curated example:

```bash
python evaluation/evaluate_yolov8_coco.py \
  --mode evaluate-runs \
  --runs-root outputs_yolov8 \
  --data-yaml /path/to/curated_yolo/data.yaml \
  --gt-json /path/to/curated_yolo/test/_annotations.coco.json \
  --subset-label curated \
  --dataset-tag curated \
  --report-dir eval_reports
```

## YOLOv8 + COCOeval: Cross-Subset

```bash
python evaluation/evaluate_yolov8_coco.py \
  --mode cross-subset \
  --runs-root outputs_yolov8 \
  --subsets-root reproduced_yolo/yolo_subsets \
  --repro-splits reproduced_splits \
  --subsets agar \
  --cross-dataset-tag agar100 \
  --trained-on total \
  --report-dir eval_reports
```

`--subsets` supports:

```bash
--subsets bright              # one subset
--subsets agar                # group: bright,dark,vague,lowres
--subsets all                 # same as agar for cross-subset mode
--subsets bright,dark,vague   # custom subset group
```

## Inference Speed Benchmark

`benchmark_inference_speed.py` benchmarks end-to-end inference latency for
Detectron2 and YOLO models discovered under a payload root. It selects image
paths first, decodes them on demand, excludes warmup passes from the reported
numbers, and writes one flat CSV with per-model, per-benchmark timing results.

Example:

```bash
python evaluation/benchmark_inference_speed.py \
  --payload-root /path/to/model_weights_and_eval_payload \
  --benchmark agar_total=/path/to/agar_total/images/test \
  --benchmark curated_test=/path/to/curated_yolo/test/images \
  --out-csv inference_speed_summary.csv \
  --framework-filter all \
  --warmup-images 20 \
  --timed-images 100 \
  --device cuda
```

Useful filters and controls:

```bash
--framework-filter detectron2     # only Detectron2 models
--framework-filter yolo           # only YOLO models
--model-name-filter yolov8m       # case-insensitive substring match
--device cpu                      # benchmark on CPU
--yolo-imgsz 640                  # YOLO inference size
--detectron2-dets-per-image 100   # Detectron2 max detections
```

The output CSV includes the benchmark name, framework, normalized model name,
training subset tag, timing configuration, mean latency in milliseconds per
image, and images per second.

## Detectron2: Bootstrap CIs

`bootstrap_detectron2_coco.py` computes bootstrap confidence intervals from Detectron2
evaluation predictions (`pred_json_path` in the eval CSV), using the fast in-memory COCOeval flow.

Run on all subsets and all 4 Detectron2 model variants:

```bash
python evaluation/bootstrap_detectron2_coco.py \
  --eval-csv eval_reports/detectron2_eval_long.csv \
  --repro-splits reproduced_splits \
  --curated-gt-json /path/to/curated/test/_annotations.coco.json \
  --subset-filter all \
  --model-filter all \
  --threshold 0.0 \
  --n-boot 300 \
  --seed 12345 \
  --out-dir bootstrap_reports
```

Examples:

```bash
# One subset + one model
--subset-filter bright --model-filter faster_rcnn_r101

# Subset group + model group
--subset-filter agar --model-filter faster

# Custom lists
--subset-filter bright,dark,lowres --model-filter faster_rcnn_r50,retinanet_r50
```

Selector options:

```bash
# subset filter
one: bright | dark | vague | lowres | total | curated
group: agar
all: all

# model filter
one: faster_rcnn_r50 | faster_rcnn_r101 | retinanet_r50 | retinanet_r101
group: faster | retinanet | r50 | r101
all: all
```

## Detectron2: WBF Ensemble Search

`search_wbf_detectron2.py` runs an exhaustive Weighted Boxes Fusion search per subset.
It evaluates all model combinations for each selected subset over WBF threshold grids.

This workflow also requires:

- `numpy`
- `pycocotools`
- `ensemble-boxes`

Example (multi-source candidate pool: Detectron2 + YOLO + cross-subset CSVs):

```bash
python evaluation/search_wbf_detectron2.py \
  --eval-csv eval_reports/detectron2_eval_long.csv \
  --extra-eval-csvs "eval_reports_detectron2_cross/*.csv,eval_reports_yolo/yolo_cocoeval_*.csv,eval_reports_yolo_cross/*.csv" \
  --repro-splits reproduced_splits \
  --curated-gt-json /path/to/curated/test/_annotations.coco.json \
  --subset-filter all \
  --source-thresholds 0,0.001 \
  --wbf-iou-thrs 0.75 \
  --wbf-skip-box-thrs 0.01,0.05 \
  --min-models 2 \
  --combo-mode topk-prefix \
  --weight-hypotheses uniform,ap,grid \
  --weight-grid-values 1,2,3,5,7 \
  --checkpoint-every 25 \
  --out-dir wbf_reports \
  --overwrite
```

Useful selectors:

```bash
# one subset / group / all / list
--subset-filter bright
--subset-filter agar
--subset-filter all
--subset-filter bright,dark,vague

# optional model filters
--model-family-filter faster
--model-family-filter retina
--model-family-filter all

--backbone-filter 50
--backbone-filter 101
--backbone-filter all
```

Combination mode:

```bash
# default: rank models by source AP and evaluate:
# top-2, top-3, ..., top-k (prefix sets)
--combo-mode topk-prefix

# evaluate all combinations for each size (slower)
--combo-mode exhaustive
```

Weight search controls:

```bash
# choose which weight hypotheses to test
--weight-hypotheses uniform,ap,grid

# grid values used when "grid" is enabled (all tuples are tested)
--weight-grid-values 1,2,3,5,7

# optional cap for very large combos (0 = use all grid tuples)
--max-grid-configs-per-combo 0
```

Checkpoint/resume behavior:

```bash
# periodic partial writes (safe for long Colab runs)
--checkpoint-every 25

# resume from existing wbf_search_<subset>_partial.csv and completed subset CSVs
# (default enabled)
--resume

# disable resume and recompute from scratch (useful with changed settings)
--no-resume
```

Output files:

- `wbf_search_<subset>.csv` for each processed subset.
- `wbf_search_<subset>_partial.csv` checkpoint file for each subset.
- `wbf_search_merged.csv` with all subset rows.
- `wbf_search_merged_partial.csv` merged checkpoint across finished/partial subsets.
- `wbf_search_best_by_subset.csv` with one best row per subset.
- `wbf_search_best_by_subset_partial.csv` checkpoint of best rows during run.
- `wbf_search_manifest.json` with run configuration metadata.

Each CSV row includes:

- model combination (`run_names`, `n_models`)
- source metadata (`trained_on`, `source_csvs`, `source_thresholds`)
- WBF settings (`wbf_iou_thr`, `wbf_skip_box_thr`, `weight_hypothesis`, `weights`)
- COCO metrics (`AP`, `AP50`, `AP75`, `APs`, `APm`, `APl`, `AR1`, `AR10`, `AR100`)
- per-class AP columns (`AP_<class_name>`)

## YOLOv8: Bootstrap CIs

`bootstrap_yolov8_coco.py` computes bootstrap confidence intervals from YOLO COCOeval
prediction files (`pred_json_path` in the YOLO evaluation CSV).

AGAR bootstrap (flow starting from AGAR evaluation outputs):

```bash
python evaluation/bootstrap_yolov8_coco.py \
  --eval-csv eval_reports/yolo_cocoeval_total.csv \
  --repro-splits reproduced_splits \
  --subset-filter agar \
  --model-filter all \
  --threshold 0.001 \
  --n-boot 300 \
  --seed 12345 \
  --out-dir bootstrap_reports
```

Curated bootstrap:

```bash
python evaluation/bootstrap_yolov8_coco.py \
  --eval-csv eval_reports/yolo_cocoeval_curated.csv \
  --repro-splits reproduced_splits \
  --curated-gt-json /path/to/curated/test/_annotations.coco.json \
  --subset-filter curated \
  --model-filter all \
  --threshold 0.001 \
  --n-boot 300 \
  --seed 12345 \
  --out-dir bootstrap_reports
```

Model and subset selectors:

```bash
# subset filter
one: total | bright | dark | vague | lowres | curated
group: agar
all: all

# model filter
all: all
group: yolov8
one/list: yolov8m or yolov8m,yolov8l
```

## Outputs

Detectron2:

- Per-run folders under each run (`eval_v2_<timestamp>/`).
- Per-subset CSVs in `--report-dir`: `detectron2_eval_long_<subset>.csv`.
- Aggregate CSV in `--report-dir`: `detectron2_eval_long.csv` (all subset CSV rows combined).
- Cross-subset aggregate CSV in each cross-eval run dir: `detectron2_cross_subset_eval.csv`.
- Global/per-run CSVs include `transfer`:
  - empty string when run folder name has no transfer token
  - `<subset>` when transfer source subset is present in the run folder name after transfer token
  - `total` when transfer token exists but source subset is not explicit in name

YOLO COCOeval:

- Per-run folders under each run (`eval_<subset>_cocoeval/`) with:
  - `predictions_coco.json`
  - `metrics_cocoeval.json`
  - `summary_cocoeval.txt`
- Global CSV in `--report-dir`:
  - `yolo_cocoeval_<subset>.csv` for `evaluate-runs`
  - `yolo_cross_subset_cocoeval.csv` for `cross-subset`



Detectron2 bootstrap:

- Long-format CSV in `--out-dir`: `bootstrap_detectron2_long_<timestamp>.csv`
- Wide summary CSV in `--out-dir`: `bootstrap_detectron2_wide_<timestamp>.csv`
- Run manifest JSON in `--out-dir`: `bootstrap_detectron2_manifest_<timestamp>.json`
- Per-subset long CSVs in `--out-dir`: `bootstrap_detectron2_long_<subset>_<timestamp>.csv`
- Per-subset wide CSVs in `--out-dir`: `bootstrap_detectron2_wide_<subset>_<timestamp>.csv`
- Outputs are refreshed during processing, so partial results remain available if runtime stops.

YOLO bootstrap:

- Long-format CSV in `--out-dir`: `bootstrap_yolo_long_<timestamp>.csv`
- Wide summary CSV in `--out-dir`: `bootstrap_yolo_wide_<timestamp>.csv`
- Run manifest JSON in `--out-dir`: `bootstrap_yolo_manifest_<timestamp>.json`

Stress test:

- `reproducibility_package/stress_test/README.md` contains run/reconstruct commands.
