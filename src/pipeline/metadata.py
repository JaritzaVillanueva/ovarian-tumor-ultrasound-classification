import pandas as pd


METADATA_COLUMNS = [
    "image_id",
    "image_name",
    "split",

    "original_class",
    "binary_label",
    "binary_class",

    "source_image_path",
    "source_mask_path",

    "organized_image_path",
    "organized_mask_path",

    "image_width",
    "image_height",
    "image_channels",
    "image_dtype",
    "image_format",

    "mask_width",
    "mask_height",
    "mask_channels",
    "mask_dtype",
    "mask_format",
    "mask_values",

    "same_dimensions",

    "status",
    "exclusion_reason",
]

def serialize_mask_values(values) -> str:
    """
    Convierte los valores únicos de una máscara en una representación adecuada para el archivo CSV.
    """

    if values is None:
        return ""

    return ",".join(str(int(value)) for value in values)


def build_included_record(
    row,
    technical_info: dict,
    organized_image_path: str,
    organized_mask_path: str,
) -> dict:
    """
    Genera el registro de metadatos correspondiente a una muestra incluida en el conjunto preparado.
    """

    return {
        "image_id": row["image_id"],
        "image_name": row["image_name"],
        "split": row["split"],

        "original_class": int(row["original_class"]),
        "binary_label": int(row["binary_label"]),
        "binary_class": row["binary_class"],

        "source_image_path": row["image_path"],
        "source_mask_path": row["mask_path"],

        "organized_image_path": organized_image_path,
        "organized_mask_path": organized_mask_path,

        "image_width": technical_info["image_width"],
        "image_height": technical_info["image_height"],
        "image_channels": technical_info["image_channels"],
        "image_dtype": technical_info["image_dtype"],
        "image_format": technical_info["image_format"],

        "mask_width": technical_info["mask_width"],
        "mask_height": technical_info["mask_height"],
        "mask_channels": technical_info["mask_channels"],
        "mask_dtype": technical_info["mask_dtype"],
        "mask_format": technical_info["mask_format"],

        "mask_values": serialize_mask_values(
            technical_info["mask_values"]
        ),

        "same_dimensions": technical_info["same_dimensions"],

        "status": "included",
        "exclusion_reason": "",
    }


def build_exclusion_record(row, reason: str,) -> dict:
    """
    Genera el registro correspondiente a una muestra excluida del conjunto preparado.
    """

    return {
        "image_id": row["image_id"],
        "image_name": row["image_name"],
        "split": row["split"],

        "original_class": int(row["original_class"]),
        "binary_label": int(row["binary_label"]),
        "binary_class": row["binary_class"],

        "source_image_path": row["image_path"],
        "source_mask_path": row["mask_path"],

        "organized_image_path": "",
        "organized_mask_path": "",

        "image_width": None,
        "image_height": None,
        "image_channels": None,
        "image_dtype": None,
        "image_format": "",

        "mask_width": None,
        "mask_height": None,
        "mask_channels": None,
        "mask_dtype": None,
        "mask_format": "",
        "mask_values": "",

        "same_dimensions": None,

        "status": "excluded",
        "exclusion_reason": reason,
    }


def records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    """
    Convierte una lista de registros en un DataFrame utilizando una estructura de columnas uniforme.
    """
    return pd.DataFrame(records, columns=METADATA_COLUMNS)