# Dataset Guide

This folder contains the dataset-side workflows used in the paper:

- AGAR split reconstruction
- AGAR split validation
- AGAR COCO-to-YOLO conversion
- YOLO dataset validation
- curated dataset download from Zenodo
- curation manifest generation

The AGAR-derived files are not shipped here. They are reproduced from the
user's own AGAR download.

## Scope

Scripts in this folder:

- `dataset/reproduce_splits.py`
- `dataset/validate_repro.py`
- `dataset/reproduce_yolo_datasets.py`
- `dataset/validate_yolo_repro.py`
- `dataset/convert_coco_to_yolo.py`
- `dataset/curation_manifest.py`
- `dataset/download_curated_dataset.py`

## Prerequisites

| Requirement | Details |
|---|---|
| Python | >= 3.10 |
| AGAR dataset | Download from <https://agar.neurosys.com/> |
| AGAR annotations | You need `dataset/annotations.json` from your AGAR copy |
| Repository files | `reproducibility_package/configs/split_assignments.json` and the scripts in `reproducibility_package/dataset/` |

Expected layout:

```text
<REPO_ROOT>/
├── reproducibility_package/
│   ├── configs/
│   │   └── split_assignments.json
│   └── dataset/
│       ├── reproduce_splits.py
│       ├── validate_repro.py
│       ├── reproduce_yolo_datasets.py
│       ├── validate_yolo_repro.py
│       ├── convert_coco_to_yolo.py
│       ├── curation_manifest.py
│       ├── download_curated_dataset.py
│       └── README.md
└── <AGAR_DOWNLOAD>/
    └── dataset/
        └── annotations.json
```

## Quick Start

Run from the package root:

```bash
cd reproducibility_package
```

### 1. Reconstruct AGAR COCO splits

```bash
python dataset/reproduce_splits.py \
  --raw-root /path/to/AGAR \
  --out-dir reproduced_splits
```

This writes files such as:

```text
reproduced_splits/
├── total_train_coco.json
├── total_val_coco.json
├── total_test_coco.json
├── bright_train_coco.json
├── bright_val_coco.json
├── bright_test_coco.json
└── ...
```

Split files are written as `<group>_<split>_coco.json`.

### 2. Validate the reconstructed AGAR splits

```bash
python dataset/validate_repro.py \
  --raw-root /path/to/AGAR \
  --repro-splits reproduced_splits \
  --out-dir validation_output
```

### 3. Build YOLO datasets from the reproduced splits

```bash
python dataset/reproduce_yolo_datasets.py \
  --raw-root /path/to/AGAR \
  --repro-splits reproduced_splits \
  --out-dir reproduced_yolo \
  --mode all
```

### 4. Validate the YOLO reproduction

```bash
python dataset/validate_yolo_repro.py \
  --repro-splits reproduced_splits \
  --repro-yolo reproduced_yolo \
  --mode all \
  --out-dir yolo_validation_output
```

### 5. Generate a curation manifest

```bash
python dataset/curation_manifest.py \
  --raw-root /path/to/AGAR \
  --out-dir curation_output
```

### 6. Download the curated dataset release

Zenodo DOI: `https://doi.org/10.5281/zenodo.18505210`

Download curated COCO:

```bash
python dataset/download_curated_dataset.py \
  --package coco \
  --out-dir curated_dataset \
  --force
```

Download curated YOLO:

```bash
python dataset/download_curated_dataset.py \
  --package yolo \
  --out-dir curated_dataset \
  --force
```

The download script performs SHA-256 verification automatically for the curated packages.

## How AGAR Split Reconstruction Works

### Curation rules

The reconstruction pipeline loads the AGAR source annotations and applies the
paper curation rules deterministically.

Main effect:

- uncountable images are excluded
- images with excluded categories are excluded
- images with more than 100 kept annotations are excluded
- empty plates are retained
- curated images with 1-100 kept annotations are retained


### Group assignment

Each curated image is assigned to background groups using
`background_category_id`:

| `background_category_id` | Group | Also in |
|:---:|---|---|
| 0 | `bright` | `highres`, `total` |
| 1 | `dark` | `highres`, `total` |
| 2 | `vague` | `highres`, `total` |
| 3 | `lowres` | `total` |

### Split assignment

Val/test membership is read from:

- `configs/split_assignments.json`

Train membership is computed as the curated group complement of val and test.

This assignments file is the authoritative record for the paper splits.

## Script Reference

### `dataset/reproduce_splits.py`

Main AGAR split reconstruction entrypoint.

Modes:

- `reconstruct`: verifies inputs and writes the COCO split JSON files
- `verify`: checks compatibility only and writes nothing

Main arguments:

- `--raw-root`
- `--assignments`
- `--out-dir`
- `--mode`

### `dataset/validate_repro.py`

Validates that the reconstructed COCO splits match the assignments and source
data.

Outputs:

- `validation_report.json`
- `validation_report.md`

### `dataset/curation_manifest.py`

Writes a per-image log describing curation inclusion and exclusion decisions.

Outputs:

- `curation_manifest.jsonl`
- `curation_summary.json`

### `dataset/convert_coco_to_yolo.py`

Converts one COCO split file into YOLO labels/images structure.

Example:

```bash
python dataset/convert_coco_to_yolo.py \
  --raw-root /path/to/AGAR \
  --coco-json reproduced_splits/bright_test_coco.json \
  --out-dir reproduced_yolo/bright_single \
  --split test
```

### `dataset/reproduce_yolo_datasets.py`

Builds the YOLO datasets used in the paper.

Modes:

- `baseline`
- `cross-subset`
- `all`

### `dataset/validate_yolo_repro.py`

Validates YOLO labels, optional copied/symlinked images, and generated
`data.yaml` files against the reconstructed COCO splits.

Outputs:

- `yolo_validation_report.json`
- `yolo_validation_report.md`

## Organization Notes

Use this README as the dataset entrypoint for the package. The root package
README links here first because AGAR split reconstruction underpins the rest of
the AGAR workflows.
