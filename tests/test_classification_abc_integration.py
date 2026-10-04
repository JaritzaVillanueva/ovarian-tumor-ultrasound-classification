"""
Tests de integración para las condiciones A/B/C materializadas. Verifican conteos, 
correspondencia de etiquetas, shapes de batch y class weights.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "configs" / "splits" / "mmotu_experimental_split_standard.csv"
DIR_A = PROJECT_ROOT / "data" / "processed" / "MMOTU_OTU_2D_experimental"
DIR_B = PROJECT_ROOT / "data" / "processed" / "MMOTU_OTU_2D_roi_pred"
DIR_C = PROJECT_ROOT / "data" / "processed" / "MMOTU_OTU_2D_roi_gt"

SKIP_REASON = "Datos materializados A/B/C no encontrados en disco"
_data_available = DIR_A.exists() and DIR_B.exists() and DIR_C.exists() and CSV_PATH.exists()


def _make_dataset(image_dir, split, condition, transform=None):
    from src.classification.dataset import MMOTUClassificationDataset
    return MMOTUClassificationDataset(
        csv_file=str(CSV_PATH),
        image_dir=str(image_dir),
        split=split,
        transform=transform,
        condition=condition,
    )

# Clases de prueba para verificar la integridad de los datos materializados A/B/C
@pytest.mark.skipif(not _data_available, reason=SKIP_REASON) # Skip tests si no se encuentran los datos materializados
class TestConditionCounts:

    def test_condition_a_train_count_800(self):
        ds = _make_dataset(DIR_A, "train", "A")
        assert len(ds) == 800

    def test_condition_b_train_count_800(self):
        ds = _make_dataset(DIR_B, "train", "B")
        assert len(ds) == 800

    def test_condition_c_train_count_800(self):
        ds = _make_dataset(DIR_C, "train", "C")
        assert len(ds) == 800

    def test_all_conditions_validation_count_200(self):
        for dir_, cond in [(DIR_A, "A"), (DIR_B, "B"), (DIR_C, "C")]:
            ds = _make_dataset(dir_, "validation", cond)
            assert len(ds) == 200, f"Condition {cond} validation: expected 200, got {len(ds)}"

# Clases de prueba para verificar la correspondencia de etiquetas entre las condiciones A/B/C
@pytest.mark.skipif(not _data_available, reason=SKIP_REASON)
class TestLabelCorrespondence:

    def test_labels_identical_across_conditions_by_image_id(self):
        ds_a = _make_dataset(DIR_A, "train", "A")
        ds_b = _make_dataset(DIR_B, "train", "B")
        ds_c = _make_dataset(DIR_C, "train", "C")

        labels_a = dict(zip(ds_a.df["image_name"].apply(lambda x: Path(x).stem), ds_a.df["binary_label"]))
        labels_b = dict(zip(ds_b.df["image_name"].apply(lambda x: Path(x).stem), ds_b.df["binary_label"]))
        labels_c = dict(zip(ds_c.df["image_name"].apply(lambda x: Path(x).stem), ds_c.df["binary_label"]))

        assert set(labels_a.keys()) == set(labels_b.keys()) == set(labels_c.keys())
        for image_id in labels_a:
            assert labels_a[image_id] == labels_b[image_id] == labels_c[image_id], \
                f"Label mismatch for image_id {image_id}: A={labels_a[image_id]}, B={labels_b[image_id]}, C={labels_c[image_id]}"

# Clases de prueba para verificar la forma de los batches y la correspondencia de class weights entre las condiciones A/B/C
@pytest.mark.skipif(not _data_available, reason=SKIP_REASON)
class TestBatchShape:

    def test_batch_shape_224x224_all_conditions(self):
        from src.classification.transforms import get_train_transforms
        from src.classification.dataloader import create_dataloader

        for dir_, cond in [(DIR_A, "A"), (DIR_B, "B"), (DIR_C, "C")]:
            ds = _make_dataset(dir_, "train", cond, transform=get_train_transforms())
            loader = create_dataloader(ds, batch_size=32, shuffle=False, num_workers=0)
            images, labels = next(iter(loader))
            assert images.shape == torch.Size([32, 3, 224, 224]), \
                f"Condition {cond}: expected [32, 3, 224, 224], got {list(images.shape)}"

# Clases de prueba para verificar la correspondencia de class weights entre las condiciones A/B/C
@pytest.mark.skipif(not _data_available, reason=SKIP_REASON)
class TestClassWeights:

    def test_class_weights_identical_across_conditions(self):
        from src.classification.utils import compute_class_weights

        ds_a = _make_dataset(DIR_A, "train", "A")
        ds_b = _make_dataset(DIR_B, "train", "B")
        ds_c = _make_dataset(DIR_C, "train", "C")

        weights_a = compute_class_weights(ds_a.df["binary_label"].values)
        weights_b = compute_class_weights(ds_b.df["binary_label"].values)
        weights_c = compute_class_weights(ds_c.df["binary_label"].values)

        assert torch.equal(weights_a, weights_b), f"Weights A != B: {weights_a} vs {weights_b}"
        assert torch.equal(weights_a, weights_c), f"Weights A != C: {weights_a} vs {weights_c}"
