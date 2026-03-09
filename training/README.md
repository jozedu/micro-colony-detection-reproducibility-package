# Training Workflows

This folder contains training scripts separated from dataset-reproduction scripts.

Companion archived model payload:
- Zenodo DOI: <https://doi.org/10.5281/zenodo.18922895>

Use the Zenodo record to inspect the archived paper checkpoints and compact
evaluation outputs. Use the commands in this README when you want to retrain the
models from reproduced AGAR splits or from the curated dataset.

## Files

- `training/train_detectron2.py`: baseline + transfer-learning training entrypoint
- `training/train_yolov8.py`: YOLOv8 training entrypoint for review-process runs

## 1) Environment

From repository root:

```bash
cd reproducibility_package
```

### 1.1 Historical Environment Used for Reported Artifacts

The paper artifacts were produced with a Google Colab environment using:

- `nvcc`: CUDA 11.8
- `torch`: `2.0` (`cu118`)
- `detectron2`: `0.6`
- `pyyaml`: `5.1`

### 1.2 Pinned Setup (Closest Match)

Use these pins when recreating the training environment:

```bash
python -m pip install -r requirements.txt
python -m pip install pyyaml==5.1
python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.0.1 torchvision==0.15.2
python -m pip install "git+https://github.com/facebookresearch/detectron2.git@v0.6"
```

### 1.3 Colab Version Drift Note

Current Colab runtimes may resolve different package builds than the historical environment above.
If package resolution differs, prioritize reproducing the pinned versions listed in this section.

## 2) Models Used in the Paper

The 4 baseline models are:

1. `COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml`
2. `COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml`
3. `COCO-Detection/retinanet_R_50_FPN_3x.yaml`
4. `COCO-Detection/retinanet_R_101_FPN_3x.yaml`

## 3) AGAR Training from Reproduced Splits

Paper-aligned subsets used for training:

1. `total`
2. `bright`
3. `dark`
4. `vague`
5. `lowres`

### 3.1 Single run template

```bash
python training/train_detectron2.py \
  --repro-splits reproduced_splits \
  --group <total|bright|dark|vague|lowres> \
  --images-root /path/to/AGAR/dataset/images \
  --model-config <MODEL_CONFIG> \
  --run-name <RUN_NAME> \
  --epochs 10 \
  --batch-size 8 \
  --base-lr 0.005 \
  --lr-step-epochs 3 \
  --output-root outputs_detectron2
```

### 3.2 Run all 4 models across all 5 subsets

```bash
MODELS=(
  COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml
  COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml
  COCO-Detection/retinanet_R_50_FPN_3x.yaml
  COCO-Detection/retinanet_R_101_FPN_3x.yaml
)

SUBSETS=(total bright dark vague lowres)

for subset in "${SUBSETS[@]}"; do
  for model in "${MODELS[@]}"; do
    python training/train_detectron2.py \
      --repro-splits reproduced_splits \
      --group "$subset" \
      --images-root /path/to/AGAR/dataset/images \
      --model-config "$model" \
      --run-name "agar_${subset}" \
      --epochs 10 \
      --batch-size 8 \
      --base-lr 0.005 \
      --lr-step-epochs 3 \
      --output-root outputs_detectron2
  done
done
```

## 4) AGAR Transfer Learning

Per manuscript logic, transfer learning starts from a stronger pretrained run and fine-tunes target subsets.

Example (fine-tune bright from a previously trained total Faster R-CNN R101 checkpoint):

```bash
python training/train_detectron2.py \
  --repro-splits reproduced_splits \
  --group bright \
  --images-root /path/to/AGAR/dataset/images \
  --model-config COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml \
  --weights /path/to/outputs_detectron2/total_.../model_final.pth \
  --run-name agar_bright_transfer \
  --epochs 10 \
  --batch-size 8 \
  --base-lr 0.005 \
  --lr-step-epochs 3 \
  --output-root outputs_detectron2
```

Run transfer learning for paper-target subsets (`bright`, `vague`, `lowres`):

```bash
TARGET_SUBSETS=(bright vague lowres)
SOURCE_CKPT=/path/to/outputs_detectron2/total_.../model_final.pth

for subset in "${TARGET_SUBSETS[@]}"; do
  python training/train_detectron2.py \
    --repro-splits reproduced_splits \
    --group "$subset" \
    --images-root /path/to/AGAR/dataset/images \
    --model-config COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml \
    --weights "$SOURCE_CKPT" \
    --run-name "agar_${subset}_transfer" \
    --epochs 10 \
    --batch-size 8 \
    --base-lr 0.005 \
    --lr-step-epochs 3 \
    --output-root outputs_detectron2
done
```

## 5) Curated Dataset from Zenodo

### 5.1 Download and extract

Use the dataset script in `dataset/download_curated_dataset.py`:

```bash
python3 dataset/download_curated_dataset.py \
  --package coco \
  --out-dir curated_dataset \
  --force
```

Full dataset download details are documented in `dataset/README.md`.

### 5.2 Train on curated dataset

```bash
python training/train_detectron2.py \
  --curated-root /path/to/curated_dataset_root \
  --model-config COCO-Detection/retinanet_R_50_FPN_3x.yaml \
  --run-name curated_retinanet_r50 \
  --epochs 100 \
  --batch-size 8 \
  --base-lr 0.005 \
  --lr-step-ratio 0.3 \
  --output-root outputs_detectron2
```

If the validation folder is `val/` instead of `valid/`, add:

```bash
--curated-val-split val
```

## 6) What Gets Saved

`train_detectron2.py` saves:

1. Detectron2 training artifacts (`metrics.json`, checkpoints, `model_final.pth`)
2. Optional `test_metrics.json` when `--eval-test-after-train` is used

For the archived paper checkpoints and compact evaluation exports, see the
model weights Zenodo payload above.

## 7) Learning Rate and Iterations

- Use `--base-lr 0.005` for paper-aligned training runs.

For a fixed iteration schedule, set `--iterations` explicitly. Example:

```bash
--iterations 2800 --lr-step-ratio 0.3
```

## 8) Full CLI Reference

```bash
python training/train_detectron2.py --help
python training/train_yolov8.py --help
```

## 9) YOLOv8 Review Training

This workflow is for the review-process YOLOv8 experiments:

1. AGAR total dataset (`100` epochs, `20` patience)
2. Curated dataset (`1000` epochs, `200` patience)

Seed used for these YOLOv8 runs: `42`.

Install dependency:

```bash
python -m pip install ultralytics
```

Prepare AGAR total YOLO dataset:

```bash
python dataset/reproduce_yolo_datasets.py \
  --raw-root /path/to/AGAR \
  --repro-splits reproduced_splits \
  --out-dir reproduced_yolo \
  --mode baseline
```

Run both trainings in one command:

```bash
python training/train_yolov8.py \
  --mode both \
  --agar-data reproduced_yolo/yolo_agar_total/data.yaml \
  --curated-data /path/to/curated_yolo/data.yaml \
  --agar-epochs 100 \
  --agar-patience 20 \
  --curated-epochs 1000 \
  --curated-patience 200 \
  --seed 42 \
  --model yolov8n.pt \
  --output-root outputs_yolov8
```

Run only one dataset:

```bash
python training/train_yolov8.py --mode agar --agar-data reproduced_yolo/yolo_agar_total/data.yaml --agar-epochs 100 --agar-patience 20 --seed 42
python training/train_yolov8.py --mode curated --curated-data /path/to/curated_yolo/data.yaml --curated-epochs 1000 --curated-patience 200 --seed 42
```

By default, YOLO training outputs are written under:

- `outputs_yolov8`

Keep that path consistent with the evaluation examples unless you override
`--output-root`.
