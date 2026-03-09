# Reproducibility Package

This package contains the scripts used to reproduce the paper workflows for:

- AGAR split reconstruction
- curated dataset download
- Detectron2 and YOLO training
- Detectron2 and YOLO evaluation
- bootstrap confidence intervals
- ensemble search with Weighted Boxes Fusion
- stress-test evaluation
- bounding box / anchor analysis

This package is the code-side companion to the reviewer artifacts and dataset releases.

Related external resources:
- curated dataset Zenodo release
- model weights and evaluation payload Zenodo release
- paper repository / GitHub project hosting this package

## Installation

Use one environment for the full package, then add a Detectron2-compatible
`torch`/`detectron2` stack if you need Detectron2 training, Detectron2
evaluation, or the stress-test workflow.

From the repository root:

```bash
cd reproducibility_package
python -m pip install -r requirements.txt
```

For Detectron2-backed workflows, install `torch`, `torchvision`, and
`detectron2` versions that match your platform and accelerator. The historical
paper-aligned pins are documented in `training/README.md`.

## Structure

- `configs/`: static configuration files used by the scripts
- `dataset/`: split reproduction, curated dataset download, COCO-to-YOLO conversion, validation
- `training/`: Detectron2 and YOLO training entrypoints
- `evaluation/`: evaluation, bootstrap, and WBF search
- `stress_test/`: high-density Detectron2 stress-test workflow
- `analysis/`: bbox and anchor analysis

## Recommended Reading Order

1. `dataset/README.md`
   Split reconstruction, curated dataset download, conversion, and validation.
2. `training/README.md`
   Model training entrypoints and expected environments.
3. `evaluation/README.md`
   Evaluation, bootstrap, and WBF ensemble search.
4. `stress_test/README.md`
   Stress-test workflow for Detectron2.
5. `analysis/README.md`
   Bounding box and anchor analysis.

## Minimal Workflow

From the repository root:

```bash
cd reproducibility_package
```

1. Reproduce the AGAR COCO splits

```bash
python dataset/reproduce_splits.py \
  --raw-root /path/to/AGAR \
  --out-dir reproduced_splits
```

2. Optionally build YOLO datasets from the reproduced splits

```bash
python dataset/reproduce_yolo_datasets.py \
  --raw-root /path/to/AGAR \
  --repro-splits reproduced_splits \
  --out-dir reproduced_yolo \
  --mode all
```

3. Download the curated dataset release when needed

```bash
python dataset/download_curated_dataset.py \
  --package coco \
  --out-dir curated_dataset \
  --force
```

4. Train or evaluate with the task-specific scripts documented in:
- `training/README.md`
- `evaluation/README.md`
- `stress_test/README.md`
- `analysis/README.md`

## Notes

- The package does not ship AGAR raw data.
- AGAR-derived files are reproduced from the user's own AGAR dataset copy.
- The curated dataset is obtained from the separate curated dataset release.
- Model weights and compact evaluation payloads are distributed separately from this package.
- `requirements.txt` covers the shared Python dependencies in this package.
- Detectron2-specific workflows still require a compatible `torch` + `detectron2`
  install on top of the shared dependencies.
