from pathlib import Path
import pandas as pd

from .config import (
    TRAIN_FILE,
    VAL_FILE,
    TRAIN_CLS_FILE,
    VAL_CLS_FILE,
    IMAGES_DIR,
    ANNOTATIONS_DIR,
    BINARY_LABEL_MAP,
    BINARY_CLASS_NAMES,
)


def read_split_file(file_path: Path) -> list[str]:
    """Lee los identificadores de un archivo de partición."""

    with open(file_path, "r", encoding="utf-8") as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]


def read_classification_file(file_path: Path, split: str) -> pd.DataFrame:
    """
    Lee las etiquetas originales de MMOTU y genera adicionalmente
    la etiqueta binaria utilizada por la investigación.
    """

    records = []

    with open(file_path, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            image_name, original_class = line.split()

            image_id = Path(image_name).stem
            original_class = int(original_class)

            binary_label = BINARY_LABEL_MAP[original_class]

            records.append({
                "image_id": image_id,
                "image_name": image_name,
                "split": split,
                "original_class": original_class,
                "binary_label": binary_label,
                "binary_class": BINARY_CLASS_NAMES[binary_label],

                "image_path": str(IMAGES_DIR / image_name),
                "mask_path": str(ANNOTATIONS_DIR / f"{image_id}_binary.PNG"),
            })

    return pd.DataFrame(records)


def load_dataset_index() -> pd.DataFrame:
    """Construye el índice completo de OTU_2D."""

    train_df = read_classification_file(TRAIN_CLS_FILE,split="train")
    val_df = read_classification_file(VAL_CLS_FILE, split="validation")
    
    dataset_df = pd.concat([train_df, val_df], ignore_index=True)

    return dataset_df