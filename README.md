# Ovarian Tumor Ultrasound Classification

Deep learning research project for the binary classification (benign / malignant) of ovarian tumors in 2D ultrasound images.

This repository contains the source code, experimental notebooks, configurations, and reproducible workflows developed as part of an undergraduate thesis project.

## Research Scope

The project studies the benign / malignant classification of ovarian tumors from 2D ultrasound images and compares, under a common training and evaluation protocol, different input representations for the classifier:

- **A. Full image:** the original ultrasound image.
- **B. Predicted ROI:** a region of interest cropped using the mask predicted by a segmentation model (U-Net baseline).
- **C. Reference ROI:** a region of interest cropped using the reference segmentation mask of the dataset, used as an idealized condition.

Segmentation is used as a means to obtain the regions of interest for conditions B and C. It is not a mandatory stage before every classification.

### Objective 1 – Experimental dataset

Objective 1 is to build the experimental dataset.

- **R1 – Data preparation pipeline.** A reproducible, documented pipeline for searching, selecting, verifying, characterizing and preparing, and organizing 2D ultrasound images. The pipeline keeps the original files unchanged: images and masks keep their original dimensions and format, and no technical normalization is applied to the dataset. Input size adaptation is defined later according to each architecture.
- **R2 – Selected, characterized and organized dataset.** The MMOTU / OTU_2D dataset with traceable binary labels, verified reference segmentation masks, and a fixed experimental partition into `train`, `validation` and `test`.

## Dataset

This project uses the OTU_2D subset of the MMOTU (Multi-Modality Ovarian Tumor Ultrasound) dataset. The original medical images are not redistributed through this repository.

To reproduce the data preparation workflow, obtain the dataset from its official source and place the OTU_2D files under:

```text
data/raw/MMOTU/OTU_2D/
```

The contents of `data/raw/` are treated as read-only and are never modified by the pipeline.

### Verified composition

| Property | Value |
|---|---|
| Samples | 1469 (1469 unique `image_id`) |
| Benign | 1312 |
| Malignant | 157 |
| Image–mask correspondences | 1469 / 1469 |
| Orphan files | 0 |
| Exact duplicate images (overall / across partitions) | 0 / 0 |

The binary label is derived from the eight original MMOTU categories (`src/pipeline/config.py`). The original category is kept alongside the binary label.

### Experimental partition

MMOTU distributes OTU_2D as 1000 `train` and 469 `val` images. In this project:

- the 469 original `val` images form the **`test`** set, reserved for final evaluation;
- the 1000 original `train` images are split into **`train`** and **`validation`**.

| split | benign | malignant | total |
|---|---|---|---|
| train | 713 | 87 | 800 |
| validation | 178 | 22 | 200 |
| test | 421 | 48 | 469 |
| **total** | 1312 | 157 | 1469 |

The membership of each split is **frozen** and must not be regenerated. The code that originally produced the 800 / 200 subdivision is not part of this repository, so no sampling method or seed is claimed for it. The class proportions of the frozen partition were verified afterwards.

The partition is formalized and verified in `notebooks/R2_dataset/06_generacion_split_experimental.ipynb`.

### Split manifests

| File | Role |
|---|---|
| `configs/splits/mmotu_experimental_split.csv` | Canonical manifest: identity, labels, original split and experimental split of every sample. |
| `configs/splits/legacy/mmotu_experimental_split_legacy.csv` | Previous manifest, preserved unchanged (`train_internal`, `val_internal`, `final_validation`). |
| `configs/splits/mmotu_experimental_split_standard.csv` | Compatibility file referenced by experiments already executed (R4 / R5). Same membership as the canonical manifest. |

See `configs/splits/README.md` for details. Technical properties recalculated from the experimental copy of the dataset are stored in `results/dataset/mmotu_experimental_technical_metadata.csv`.

## Project Structure

```text
.
├── configs/
│   ├── classification/      # CNN training configurations
│   └── splits/              # experimental partition manifests
├── data/                    # not versioned
│   ├── raw/                 # original MMOTU / OTU_2D files (read-only)
│   ├── interim/
│   └── processed/           # organized experimental copy and materialized ROIs
├── models/                  # trained checkpoints (not versioned)
├── notebooks/
│   ├── R2_dataset/          # inspection, pipeline, characterization, integrity, experimental partition
│   ├── R4_segmentation/     # U-Net baseline and ROI procedure
│   └── R5_classification/   # input conditions A/B/C and CNN experiments
├── results/
│   ├── dataset/
│   ├── segmentation/
│   └── classification/
├── src/
│   ├── pipeline/            # data preparation pipeline (R1)
│   ├── segmentation/        # U-Net and ROI extraction
│   └── classification/      # datasets, models, training and evaluation
├── tests/
└── README.md
```

## Current Status

- **R1 / R2:** data preparation pipeline implemented and validated; experimental dataset characterized, verified and organized into the frozen `train` / `validation` / `test` partition.
- **R4:** U-Net baseline segmentation model trained and evaluated; ROI extraction procedure defined.
- **R5:** input conditions A / B / C materialized for `train` and `validation`; formal runs of three CNN architectures (ResNet50, DenseNet121, ConvNeXt-Tiny) under a common protocol.
- **R6:** comparison of architectures documented.
- **R7 and final evaluation on `test`:** pending.

The repository is being aligned with the current formulation of the thesis methodology. Results from experiments already executed are preserved as historical records.

## Repository Status

This repository is currently maintained as a private research repository during thesis development.
