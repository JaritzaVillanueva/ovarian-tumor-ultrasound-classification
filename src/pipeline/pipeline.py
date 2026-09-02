from pathlib import Path
import shutil

import cv2
import numpy as np
import pandas as pd

from .config import (PROJECT_ROOT, PROCESSED_DATA_DIR,)
from .validation import validate_sample
from .metadata import (build_included_record, build_exclusion_record, records_to_dataframe,)


def to_project_relative(path: str | Path) -> str:
    """
    Convierte una ruta a una representación relativa respecto a la raíz del proyecto cuando sea posible.
    """

    path = Path(path).resolve()
    project_root = PROJECT_ROOT.resolve()

    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def prepare_row_for_metadata(row: pd.Series) -> pd.Series:
    """
    Genera una copia de la muestra utilizando rutas relativas para evitar dependencias con rutas absolutas del 
    equipo local.
    """

    metadata_row = row.copy()
    metadata_row["image_path"] = to_project_relative(row["image_path"])
    metadata_row["mask_path"] = to_project_relative(row["mask_path"])

    return metadata_row

def get_channel_count(array: np.ndarray) -> int:
    """
    Obtiene el número de canales de una imagen o máscara.
    """

    if array.ndim == 2:
        return 1

    return array.shape[2]

def extract_technical_info(row: pd.Series,) -> dict:
    """
    Obtiene las propiedades técnicas de la imagen y la máscara sin modificar los archivos originales.
    """

    image_path = Path(row["image_path"])
    mask_path = Path(row["mask_path"])

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED,)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED,)

    if image is None:
        raise ValueError("image_not_readable")

    if mask is None:
        raise ValueError("mask_not_readable")

    image_height, image_width = image.shape[:2]
    mask_height, mask_width = mask.shape[:2]

    mask_values = np.unique(mask).tolist()

    return {
        "image_width": image_width,
        "image_height": image_height,
        "image_channels": get_channel_count(image),
        "image_dtype": str(image.dtype),
        "image_format": image_path.suffix,

        "mask_width": mask_width,
        "mask_height": mask_height,
        "mask_channels": get_channel_count(mask),
        "mask_dtype": str(mask.dtype),
        "mask_format": mask_path.suffix,
        "mask_values": mask_values,

        "same_dimensions": (image_height == mask_height and image_width == mask_width),
    }



def build_output_paths(row: pd.Series, output_root: str | Path,) -> tuple[Path, Path]:
    """
    Construye las rutas de organización de la imagen y máscara conservando sus nombres y formatos originales.
    """

    output_root = Path(output_root)
    split = str(row["split"])

    image_name = Path(row["image_path"]).name
    mask_name = Path(row["mask_path"]).name

    organized_image_path = (output_root / split / "images" / image_name)
    organized_mask_path = (output_root / split / "masks" / mask_name)

    return (organized_image_path, organized_mask_path,)


def copy_original_files(
    row: pd.Series,
    organized_image_path: Path,
    organized_mask_path: Path,
) -> None:
    """
    Copia la imagen y máscara originales al conjunto organizado sin recodificarlas ni modificar sus píxeles.
    """

    source_image_path = Path(row["image_path"])
    source_mask_path = Path(row["mask_path"])

    organized_image_path.parent.mkdir(parents=True, exist_ok=True,)
    organized_mask_path.parent.mkdir(parents=True, exist_ok=True,)

    try:
        shutil.copy2(source_image_path, organized_image_path,)
        shutil.copy2(source_mask_path, organized_mask_path,)

    except Exception:

        # Evitar archivos parciales en caso de error
        if organized_image_path.exists():
            organized_image_path.unlink()

        if organized_mask_path.exists():
            organized_mask_path.unlink()

        raise


def process_sample(
    row: pd.Series,
    output_root: str | Path = PROCESSED_DATA_DIR,
    save_outputs: bool = True,
) -> dict:
    """
    Ejecuta las operaciones de preparación y organización
    correspondientes a una muestra.
    """

    metadata_row = prepare_row_for_metadata(row)

    # 1. Validación de integridad
    is_valid, reason = validate_sample(row)

    if not is_valid:
        return build_exclusion_record(row=metadata_row, reason=reason,)

    # 2. Caracterización técnica
    try:
        technical_info = extract_technical_info(row)

    except ValueError as error:
        return build_exclusion_record(row=metadata_row, reason=str(error),)

    # 3. Construcción de rutas de organización
    (organized_image_path, organized_mask_path,) = build_output_paths(row=row, output_root=output_root,)

    # 4. Organización física de archivos
    if save_outputs:
        try:
            copy_original_files(
                row=row,
                organized_image_path=organized_image_path,
                organized_mask_path=organized_mask_path,
            )

        except Exception:
            return build_exclusion_record(
                row=metadata_row,
                reason="output_copy_error",
            )

    # 5. Generación del registro de metadatos
    return build_included_record(
        row=metadata_row,
        technical_info=technical_info,
        organized_image_path=to_project_relative(organized_image_path),
        organized_mask_path=to_project_relative(organized_mask_path),
    )


def run_pipeline(
    dataset_df: pd.DataFrame,
    output_root: str | Path = PROCESSED_DATA_DIR,
    save_outputs: bool = True,
) -> pd.DataFrame:
    """
    Ejecuta el pipeline sobre las muestras recibidas. Los errores de una muestra se registran como exclusiones
    sin detener el procesamiento de las demás.
    """

    records = []

    for _, row in dataset_df.iterrows():
        try:
            record = process_sample(
                row=row,
                output_root=output_root,
                save_outputs=save_outputs,
            )

        except Exception as error:
            metadata_row = prepare_row_for_metadata(row)

            record = build_exclusion_record(
                row=metadata_row,
                reason=(
                    f"processing_error:"
                    f"{type(error).__name__}"
                ),
            )

        records.append(record)

    return records_to_dataframe(records)


def save_pipeline_reports(metadata_df: pd.DataFrame, output_root: str | Path,) -> dict:
    """
    Guarda el registro completo de metadatos y el registro de exclusiones.
    """

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True,)

    metadata_path = output_root / "metadata.csv"
    exclusions_path = output_root / "exclusions.csv"

    metadata_df.to_csv(metadata_path, index=False, encoding="utf-8",)

    exclusions_df = metadata_df[metadata_df["status"] == "excluded"].copy()
    exclusions_df.to_csv(exclusions_path, index=False, encoding="utf-8",)

    return {"metadata_path": metadata_path, "exclusions_path": exclusions_path,}