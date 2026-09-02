from pathlib import Path
import cv2
import pandas as pd

from .config import TRAIN_FILE, VAL_FILE
from .dataset import read_split_file


def validate_dataset_index(dataset_df: pd.DataFrame) -> dict:
    """Verifica integridad básica del índice del dataset."""

    results = {}

    # Duplicados por ID
    results["duplicate_ids"] = int(dataset_df["image_id"].duplicated().sum())

    # Archivos faltantes
    results["missing_images"] = int(
        sum(
            not Path(path).exists()
            for path in dataset_df["image_path"]
        )
    )

    results["missing_masks"] = int(
        sum(
            not Path(path).exists()
            for path in dataset_df["mask_path"]
        )
    )

    # Particiones
    train_ids = set(read_split_file(TRAIN_FILE))
    val_ids = set(read_split_file(VAL_FILE))

    results["train_val_overlap"] = len(train_ids.intersection(val_ids))
    results["total_samples"] = len(dataset_df)

    return results


def validate_sample(row) -> tuple[bool, str]:
    """
    Verifica que una muestra pueda ser procesada correctamente.
    """

    image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)

    if image is None:
        return False, "image_not_readable"

    mask = cv2.imread(row["mask_path"], cv2.IMREAD_UNCHANGED)

    if mask is None:
        return False, "mask_not_readable"

    if image.shape[:2] != mask.shape[:2]:
        return False, "dimension_mismatch"

    unique_values = set(
        int(value)
        for value in set(mask.flatten())
    )

    if unique_values != {0, 1}:
        return False, "invalid_binary_mask"

    return True, "valid"