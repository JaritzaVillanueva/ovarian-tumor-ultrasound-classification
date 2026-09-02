from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = ( PROJECT_ROOT / "data" / "raw" / "MMOTU" / "OTU_2D")

IMAGES_DIR = RAW_DATA_DIR / "images"
ANNOTATIONS_DIR = RAW_DATA_DIR / "annotations"

TRAIN_FILE = RAW_DATA_DIR / "train.txt"
VAL_FILE = RAW_DATA_DIR / "val.txt"
TRAIN_CLS_FILE = RAW_DATA_DIR / "train_cls.txt"
VAL_CLS_FILE = RAW_DATA_DIR / "val_cls.txt"

PROCESSED_DATA_DIR = ( PROJECT_ROOT / "data" / "processed" / "MMOTU_OTU_2D")

# Etiqueta derivada para la tesis
# 0 = benigno
# 1 = maligno
BINARY_LABEL_MAP = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 1,
    7: 1,
}

BINARY_CLASS_NAMES = {
    0: "benign",
    1: "malignant",
}

# Directorio para almacenar los resultados de la validación del pipeline.
PIPELINE_VALIDATION_DIR = (PROJECT_ROOT / "data" / "interim" / "pipeline_validation")