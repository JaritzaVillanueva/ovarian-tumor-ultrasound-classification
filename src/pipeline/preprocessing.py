import cv2
import numpy as np

# original -> scale_factor -> redimencionar -> padding -> final

"""
Operaciones de transformación de imágenes.

Las funciones de este módulo se encuentran implementadas y validadas como alternativas de preprocesamiento. 
No se aplican de forma obligatoria durante la constitución del conjunto de datos.

Su configuración dependerá de los requerimientos de las arquitecturas seleccionadas posteriormente.
"""

def resize_with_padding(image: np.ndarray, mask: np.ndarray, target_size: tuple[int, int],):
    """
    Redimensiona imagen y máscara preservando la relación de aspecto y completa el espacio 
    restante mediante padding.
    """

    target_width, target_height = target_size
    original_height, original_width = image.shape[:2]

    # Factor de escala que permite preservar la relación de aspecto
    scale_factor = min(target_width / original_width, target_height / original_height)

    new_width = max(1, round(original_width * scale_factor))
    new_height = max(1, round(original_height * scale_factor))

    # Evitar que el redondeo exceda el tamaño objetivo
    new_width = min(new_width, target_width)
    new_height = min(new_height, target_height)

    # Interpolación de la imagen
    image_interpolation = (
        cv2.INTER_AREA
        if scale_factor < 1
        else cv2.INTER_LINEAR
    )

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=image_interpolation)

    # La máscara usa nearest-neighbor para conservar {0, 1}
    resized_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # Padding necesario
    remaining_width = target_width - new_width
    remaining_height = target_height - new_height
    padding_left = remaining_width // 2
    padding_right = remaining_width - padding_left
    padding_top = remaining_height // 2
    padding_bottom = remaining_height - padding_top

    processed_image = cv2.copyMakeBorder(
        resized_image,
        padding_top,
        padding_bottom,
        padding_left,
        padding_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    processed_mask = cv2.copyMakeBorder(
        resized_mask,
        padding_top,
        padding_bottom,
        padding_left,
        padding_right,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )

    transform_info = {
        "original_width": original_width,
        "original_height": original_height,
        "resized_width": new_width,
        "resized_height": new_height,
        "target_width": target_width,
        "target_height": target_height,
        "scale_factor": scale_factor,
        "padding_left": padding_left,
        "padding_right": padding_right,
        "padding_top": padding_top,
        "padding_bottom": padding_bottom,
    }

    return (processed_image, processed_mask, transform_info)