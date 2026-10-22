# Experimental Dataset Splits

This directory contains the frozen experimental partition of the **MMOTU / OTU_2D** dataset used in the thesis.

## Dataset partition

| Split | Benign | Malignant | Total | MMOTU origin |
|---|---:|---:|---:|---|
| Train | 713 | 87 | 800 | Original train |
| Validation | 178 | 22 | 200 | Original train |
| Test | 421 | 48 | 469 | Original val |
| **Total** | **1312** | **157** | **1469** | |

The original MMOTU training subset (1,000 images) was divided into 800 training and 200 validation images. Its original validation subset (469 images) became the experimental test set.

**The partition is frozen and must not be regenerated or modified.**

### Partition provenance

Historical code suggests a stratified 80/20 split with `random_state=42`, but the exact frozen membership could not be reproduced because the original input order is unavailable.

Therefore, the sampling procedure and seed cannot be confirmed as the origin of the current partition. The versioned manifests are the authoritative records of split membership.

## Manifest files

| File | Purpose |
|---|---|
| `mmotu_experimental_split.csv` | **Canonical reference:** 11 columns documenting sample identity, labels, MMOTU origin, experimental partition, and image/mask paths |
| `mmotu_experimental_split_standard.csv` | **Operational manifest:** 7-column version used by training scripts, notebooks, tests and experiment configurations |
| `legacy/mmotu_experimental_split_legacy.csv` | **Historical manifest:** original project partition names; retained as input to the R2 documentation notebook |

All three manifests contain the same 1,469 samples, with identical labels, ordering and split membership.

The legacy partition names correspond to:

| Legacy | Current |
|---|---|
| `train_internal` | `train` |
| `val_internal` | `validation` |
| `final_validation` | `test` |

The canonical manifest is generated and verified by `notebooks/R2_dataset/06_generacion_split_experimental.ipynb`, using the legacy manifest and additional sample metadata.

## Reproducibility and limitations

- **Do not modify, rename or regenerate** any of the three manifests.
- The operational `_standard.csv` is referenced by the training pipeline and all 42 recorded classification runs.
- The 27 formal R4/R5 classification runs used only train and validation. Test was previously accessed in preliminary classifier and U-Net evaluations; it was not used to train or select those formal classification checkpoints.
- No patient identifiers are available, so independence between patients across splits cannot be verified.

Technical image and mask metadata are available in `results/dataset/mmotu_experimental_technical_metadata.csv`.