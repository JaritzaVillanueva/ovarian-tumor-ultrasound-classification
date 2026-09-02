# Ovarian Tumor Ultrasound Classification

Deep learning research project for ovarian tumor analysis and binary classification using 2D ultrasound images.

This repository contains the source code, experimental notebooks, and reproducible workflows developed as part of an undergraduate thesis project.

The methodology includes dataset preparation, ovarian lesion segmentation, and deep learning-based classification of ovarian tumors as benign or malignant.

## Project Structure

```text
.
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
├── src/
└── README.md
```

## Dataset

This project uses the OTU_2D subset of the MMOTU (Multi-Modality Ovarian Tumor Ultrasound) dataset.

The original medical images are not redistributed through this repository.

To reproduce the data preparation workflow, obtain the dataset from its official source and place the OTU_2D files under:
```cmd
data/raw/MMOTU/OTU_2D/
```

## Current Status

The dataset preparation, characterization, and integrity verification pipeline has been implemented and validated.

Segmentation and classification experiments are under development.

## Research Scope

The project focuses on the automated classification of ovarian tumors from 2D ultrasound images. A segmentation stage is incorporated to delimit the region of interest before the classification stage.

## Repository Status

This repository is currently maintained as a private research repository during thesis development.