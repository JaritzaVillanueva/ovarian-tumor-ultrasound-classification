from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


BBox = Tuple[int, int, int, int]

MIN_COMPONENT_AREA_FRACTION: float = 0.001
"""
Fracción mínima del área de la imagen original que debe ocupar la mayor componente conexa 
predicha para considerarla válida. Si el valor es inferior, se aplica fallback a imagen completa.
"""


@dataclass(frozen=True)
class ROIResult:
    """
    Resultado de la extracción de una región de interés (ROI).
    """

    roi: Image.Image        # Recorte RGB de la imagen original
    binary_mask: np.ndarray # Máscara binaria para calcular el ROI
    bbox: BBox              # Bounding box final en formato (left, top, right, bottom), 
                            # con right y bottom exclusivos.
    used_fallback: bool     # True: máscara vacía, se utilizó imagen completa como fallback.


def _to_binary_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convierte una máscara a uint8 binaria {0,1}.
    """
    array = np.asarray(mask)

    if array.ndim == 3:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError( f"Se esperaba una máscara 2D, pero se recibió shape={array.shape}." )

    return (array >= threshold).astype(np.uint8)


def keep_largest_component(binary_mask: np.ndarray) -> np.ndarray:
    """
    Conserva únicamente la componente conexa de mayor área.
    Si la máscara está vacía, devuelve una máscara vacía del mismo tamaño.
    """
    mask = _to_binary_mask(binary_mask, threshold=0.5)

    if mask.sum() == 0:
        return mask

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats( mask, connectivity=8, )

    # La etiqueta 0 corresponde al fondo.
    if n_labels <= 1:
        return mask

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(component_areas))

    return (labels == largest_label).astype(np.uint8)


def mask_to_bbox(binary_mask: np.ndarray) -> Optional[BBox]:
    """
    Obtiene el bounding box mínimo que contiene todos los píxeles positivos.
    """
    mask = _to_binary_mask(binary_mask, threshold=0.5)
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    left = int(xs.min())
    top = int(ys.min())
    right = int(xs.max()) + 1
    bottom = int(ys.max()) + 1

    return left, top, right, bottom


def expand_bbox( bbox: BBox, image_size: Tuple[int, int], margin_ratio: float = 0.10, ) -> BBox:
    """
    Expande un bounding box con un margen proporcional al tamaño de la ROI.
    """
    if margin_ratio < 0:
        raise ValueError("margin_ratio debe ser >= 0.")

    image_width, image_height = image_size
    left, top, right, bottom = bbox

    bbox_width = max(1, right - left)
    bbox_height = max(1, bottom - top)

    margin_x = int(round(bbox_width * margin_ratio))
    margin_y = int(round(bbox_height * margin_ratio))

    expanded_left = max(0, left - margin_x)
    expanded_top = max(0, top - margin_y)
    expanded_right = min(image_width, right + margin_x)
    expanded_bottom = min(image_height, bottom + margin_y)

    return ( expanded_left, expanded_top, expanded_right, expanded_bottom, )


def resize_mask_to_image( binary_mask: np.ndarray, image_size: Tuple[int, int], ) -> np.ndarray:
    """
    Redimensiona una máscara binaria a la resolución de la imagen original utilizando 
    interpolación nearest-neighbor.
    """
    mask = _to_binary_mask(binary_mask, threshold=0.5)
    width, height = image_size
    resized = cv2.resize( mask, dsize=(width, height), interpolation=cv2.INTER_NEAREST, )

    return (resized > 0).astype(np.uint8)


def _is_degenerate_mask( mask: np.ndarray, image_size: Tuple[int, int], min_area_fraction: float, ) -> bool:
    """
    Determina si la mayor componente conexa de una máscara es demasiado pequeña para 
    producir una ROI útil.
    """
    if mask.sum() == 0:
        return True

    image_area = image_size[0] * image_size[1]
    if image_area == 0:
        return True

    component_area = int(mask.sum())
    return (component_area / image_area) < min_area_fraction


def extract_roi_from_mask(
    image: Image.Image,
    binary_mask: np.ndarray,
    margin_ratio: float = 0.10,
    keep_largest: bool = True,
    fallback_to_full_image: bool = True,
    min_component_area_fraction: Optional[float] = None,
) -> ROIResult:
    """
    Extrae una ROI de la imagen original a partir de una máscara binaria.

    Fallback a imagen completa cuando:
    - la máscara está vacía, o
    - la mayor componente conexa ocupa menos de min_component_area_fraction de la imagen original.
    """
    if min_component_area_fraction is None:
        min_component_area_fraction = MIN_COMPONENT_AREA_FRACTION

    image = image.convert("RGB")
    mask = resize_mask_to_image( binary_mask, image_size=image.size, )

    if keep_largest:
        mask = keep_largest_component(mask)

    degenerate = _is_degenerate_mask( mask, image_size=image.size, min_area_fraction=min_component_area_fraction, )

    if degenerate:
        if not fallback_to_full_image:
            raise ValueError( "La máscara es vacía o degenerada y " "fallback_to_full_image=False." )
        full_bbox = (0, 0, image.width, image.height)

        return ROIResult( roi=image.copy(), binary_mask=mask, bbox=full_bbox, used_fallback=True, )

    bbox = mask_to_bbox(mask)
    expanded = expand_bbox( bbox=bbox, image_size=image.size, margin_ratio=margin_ratio, )
    roi = image.crop(expanded)

    return ROIResult( roi=roi, binary_mask=mask, bbox=expanded, used_fallback=False, )


def predict_binary_mask(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    image_size: Tuple[int, int] = (256, 256),
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ejecuta inferencia con el U-Net baseline sobre una imagen original.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold debe estar en [0,1].")

    image_rgb = image.convert("RGB")
    resized = image_rgb.resize( image_size, resample=Image.Resampling.BILINEAR, )

    array = np.asarray( resized, dtype=np.float32, ) / 255.0
    array = np.transpose(array, (2, 0, 1))
    array = np.ascontiguousarray(array)

    tensor = ( torch.from_numpy(array) .unsqueeze(0) .to(device) )
    model.eval()

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.sigmoid(logits)

    probability_map = (
        probabilities
        .squeeze()
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    binary_mask = ( probability_map >= threshold ).astype(np.uint8)

    return binary_mask, probability_map


def predict_and_extract_roi(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    image_size: Tuple[int, int] = (256, 256),
    threshold: float = 0.5,
    margin_ratio: float = 0.10,
    keep_largest: bool = True,
    fallback_to_full_image: bool = True,
    min_component_area_fraction: Optional[float] = None,
) -> Tuple[ROIResult, np.ndarray]:
    """
    Pipeline completo: imagen original -> U-Net -> máscara predicha -> ROI.
    """
    binary_mask, probability_map = predict_binary_mask(
        model=model,
        image=image,
        device=device,
        image_size=image_size,
        threshold=threshold,
    )

    roi_result = extract_roi_from_mask(
        image=image,
        binary_mask=binary_mask,
        margin_ratio=margin_ratio,
        keep_largest=keep_largest,
        fallback_to_full_image=fallback_to_full_image,
        min_component_area_fraction=min_component_area_fraction,
    )

    return roi_result, probability_map
