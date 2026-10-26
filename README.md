# Ovarian Tumor Ultrasound Classification

Deep learning research project for the **binary classification of ovarian tumors (benign/malignant)** using 2D ultrasound images, developed as part of an undergraduate thesis.

The project evaluates three convolutional neural network architectures and investigates how different image representations affect classification performance.

## Research Overview

Three input representations are compared:

- **A - Full image:** original ultrasound image.
- **B - Predicted ROI:** cropped region using a U-Net-predicted segmentation mask.
- **C - Reference ROI:** cropped region using the dataset's reference mask, as an idealized comparison.

The evaluated architectures are **ResNet50, DenseNet121 and ConvNeXt-Tiny**. Segmentation is an auxiliary component for ROI extraction; it is not required for full-image classification.

## Dataset

The project uses the **OTU_2D subset of MMOTU**, containing 1,469 ultrasound images with corresponding segmentation masks.

| Split | Images |
|---|---:|
| Train | 800 |
| Validation | 200 |
| Test | 469 |
| **Total** | **1469** |

The original MMOTU images are not distributed in this repository. Original data are treated as read-only, and the experimental partition is frozen.

See [Dataset splits](configs/splits/README.md) for class distribution, manifest definitions, partition provenance and limitations.

## Project Structure

```text
.
├── configs/                   # Training configurations and frozen splits
├── data/                      # Raw and processed datasets
├── models/                    # Model checkpoints
├── notebooks/
│   ├── R2_dataset/            # Dataset preparation and verification
│   ├── R4_benchmark/          # CNN benchmark analysis
│   ├── R5_input_representation/ # U-Net, ROI and A/B/C analyses
│   ├── R6_model_selection/    # Model selection
│   └── historico/             # Preliminary experiments
├── results/
│   ├── dataset/               # Dataset characterization
│   ├── segmentation/          # U-Net and ROI results
│   └── classification/        # CNN experiments and comparisons
├── src/                       # Data pipeline, segmentation and classification
└── tests/                     # Automated tests
```

## Research Progress

| Stage | Description | Status |
|---|---|---|
| R1–R2 | Data pipeline, characterization and experimental partition | Completed |
| R3 | Bibliographic selection of CNN architectures | Completed; documented separately |
| R4 | CNN benchmark: 3 architectures × 3 seeds, input A | Completed |
| R5 | U-Net, ROI generation and comparison of A/B/C | Completed |
| R6 | Model selection: ConvNeXt-Tiny with input A (mean validation AP, per approved IOV) | Completed |
| R7–R9 | Optimization and subsequent research stages | Pending |

R4 and R5 comprise **27 formal classification runs**, using train and validation. The final formal classification evaluation on test remains pending. Earlier exploratory evaluations accessed test; these are separate from the formal classification runs.

## Documentation

- [Configuration guide](configs/README.md)
- [Experimental dataset splits](configs/splits/README.md)
- [R5 workflow](notebooks/R5_input_representation/README.md)
- [Historical experiments](notebooks/historico/README.md)

Experiment-specific settings and metrics are recorded under `results/classification/`.