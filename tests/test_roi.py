"""
Verifica el procedimiento ROI:
- keep_largest_component solo para predicciones
- min_component_area_fraction = 0.001 como criterio de fallback
- margin_ratio = 0.10
- GT multicomponente conserva todas sus componentes.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.segmentation.roi import (
    MIN_COMPONENT_AREA_FRACTION,
    ROIResult,
    _is_degenerate_mask,
    _to_binary_mask,
    expand_bbox,
    extract_roi_from_mask,
    keep_largest_component,
    mask_to_bbox,
    resize_mask_to_image,
)

def _make_image(width: int, height: int) -> Image.Image:
    """Crea una imagen RGB de prueba."""
    array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")

def _make_mask(height: int, width: int, regions: list[tuple]) -> np.ndarray:
    """
    Crea una máscara binaria uint8 con regiones rectangulares activas.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for top, left, bottom, right in regions:
        mask[top:bottom, left:right] = 1
    return mask


# Constante del módulo
class TestModuleConstant:

    def test_min_component_area_fraction_value(self):
        assert MIN_COMPONENT_AREA_FRACTION == 0.001


# 1. Máscara normal -> ROI 
class TestNormalMaskProducesROI:

    def test_single_component_produces_roi(self):
        image = _make_image(100, 100)
        mask = _make_mask(100, 100, [(20, 30, 60, 70)])

        result = extract_roi_from_mask( image, mask, margin_ratio=0.10, keep_largest=True, )

        assert isinstance(result, ROIResult)
        assert not result.used_fallback
        assert result.roi.size[0] > 0
        assert result.roi.size[1] > 0

    def test_large_component_not_degenerate(self):
        mask = _make_mask(100, 100, [(10, 10, 30, 30)])
        assert not _is_degenerate_mask(mask, (100, 100), 0.001)


# 2. Máscara vacía -> fallback
class TestEmptyMaskFallback:

    def test_empty_mask_triggers_fallback(self):
        image = _make_image(100, 100)
        mask = np.zeros((100, 100), dtype=np.uint8)

        result = extract_roi_from_mask( image, mask, margin_ratio=0.10, keep_largest=True, )

        assert result.used_fallback
        assert result.bbox == (0, 0, 100, 100)
        assert result.roi.size == (100, 100)

    def test_empty_mask_raises_when_no_fallback(self):
        image = _make_image(100, 100)
        mask = np.zeros((100, 100), dtype=np.uint8)

        with pytest.raises(ValueError, match="vacía o degenerada"):
            extract_roi_from_mask( image, mask, margin_ratio=0.10, keep_largest=True, fallback_to_full_image=False, )

    def test_empty_mask_is_degenerate(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        assert _is_degenerate_mask(mask, (100, 100), 0.001)


# 3. Componente < 0.001 -> fallback 
class TestTinyComponentFallback:

    def test_tiny_component_triggers_fallback(self):
        image = _make_image(1000, 1000)
        mask = _make_mask(1000, 1000, [(0, 0, 1, 1)])

        result = extract_roi_from_mask( image, mask, margin_ratio=0.10, keep_largest=True, )

        assert result.used_fallback
        assert result.bbox == (0, 0, 1000, 1000)

    def test_tiny_component_is_degenerate(self):
        mask = _make_mask(1000, 1000, [(0, 0, 1, 1)])
        area_fraction = 1 / (1000 * 1000)
        assert area_fraction < MIN_COMPONENT_AREA_FRACTION
        assert _is_degenerate_mask(mask, (1000, 1000), MIN_COMPONENT_AREA_FRACTION)

    def test_fraction_just_below_threshold(self):
        mask = _make_mask(100, 100, [(0, 0, 1, 1)])
        assert _is_degenerate_mask(mask, (100, 100), 0.001)

    def test_custom_threshold_respected(self):
        image = _make_image(100, 100)
        mask = _make_mask(100, 100, [(0, 0, 2, 2)])

        result_strict = extract_roi_from_mask( image, mask, margin_ratio=0.10, keep_largest=True, min_component_area_fraction=0.05, )
        assert result_strict.used_fallback

        result_lax = extract_roi_from_mask( image, mask, margin_ratio=0.10, keep_largest=True, min_component_area_fraction=0.0001, )
        assert not result_lax.used_fallback


# 4. Componente válida >= 0.001 -> ROI
class TestValidComponentProducesROI:

    def test_component_at_threshold_produces_roi(self):
        image = _make_image(100, 100)
        mask = _make_mask(100, 100, [(0, 0, 1, 10)])
        area_fraction = 10 / 10000
        assert area_fraction >= MIN_COMPONENT_AREA_FRACTION

        result = extract_roi_from_mask( image, mask, margin_ratio=0.10, keep_largest=True, )
        assert not result.used_fallback

    def test_component_above_threshold_not_degenerate(self):
        mask = _make_mask(100, 100, [(0, 0, 5, 5)])
        assert not _is_degenerate_mask(mask, (100, 100), MIN_COMPONENT_AREA_FRACTION)


# 5. Clipping correcto del margen
class TestMarginClipping:

    def test_margin_clips_to_image_bounds(self):
        bbox = (0, 0, 10, 10)
        expanded = expand_bbox(bbox, image_size=(50, 50), margin_ratio=0.10)
        left, top, right, bottom = expanded

        assert left >= 0
        assert top >= 0
        assert right <= 50
        assert bottom <= 50

    def test_margin_clips_at_bottom_right_edge(self):
        bbox = (90, 90, 100, 100)
        expanded = expand_bbox(bbox, image_size=(100, 100), margin_ratio=0.20)
        left, top, right, bottom = expanded

        assert right == 100
        assert bottom == 100

    def test_margin_clips_at_top_left_edge(self):
        bbox = (0, 0, 10, 10)
        expanded = expand_bbox(bbox, image_size=(100, 100), margin_ratio=0.20)
        left, top, right, bottom = expanded

        assert left == 0
        assert top == 0

    def test_full_roi_respects_image_bounds(self):
        image = _make_image(50, 50)
        mask = _make_mask(50, 50, [(0, 0, 50, 50)])

        result = extract_roi_from_mask( image, mask, margin_ratio=0.20, keep_largest=True, )
        left, top, right, bottom = result.bbox
        assert left >= 0 and top >= 0
        assert right <= 50 and bottom <= 50

    def test_margin_ratio_zero(self):
        bbox = (10, 20, 30, 40)
        expanded = expand_bbox(bbox, image_size=(100, 100), margin_ratio=0.0)
        assert expanded == (10, 20, 30, 40)


# 6. GT multicomponente conserva todas las componentes
class TestGTMulticomponent:

    def test_keep_largest_false_preserves_all_components(self):
        image = _make_image(100, 100)
        mask = _make_mask(100, 100, [
            (10, 10, 20, 20),
            (60, 60, 70, 70),
        ])

        total_pixels_before = int(mask.sum())
        assert total_pixels_before == 200

        result = extract_roi_from_mask( image, mask, margin_ratio=0.10, keep_largest=False, )

        assert not result.used_fallback
        assert int(result.binary_mask.sum()) == total_pixels_before

    def test_keep_largest_true_removes_smaller_component(self):
        mask = _make_mask(100, 100, [
            (10, 10, 30, 30),
            (60, 60, 65, 65),
        ])

        largest = keep_largest_component(mask)

        assert int(largest.sum()) < int(mask.sum())
        assert int(largest.sum()) == 20 * 20

    def test_gt_roi_bbox_covers_all_components(self):
        image = _make_image(100, 100)
        mask = _make_mask(100, 100, [
            (10, 10, 20, 20),
            (80, 80, 90, 90),
        ])

        result = extract_roi_from_mask( image, mask, margin_ratio=0.0, keep_largest=False, )

        left, top, right, bottom = result.bbox
        assert left <= 10
        assert top <= 10
        assert right >= 90
        assert bottom >= 90


# Helpers
class TestResizeMask:

    def test_resize_preserves_binary(self):
        mask = _make_mask(10, 10, [(2, 2, 8, 8)])
        resized = resize_mask_to_image(mask, image_size=(50, 50))

        assert resized.dtype == np.uint8
        assert set(np.unique(resized)).issubset({0, 1})
        assert resized.shape == (50, 50)

    def test_resize_identity(self):
        mask = _make_mask(100, 100, [(10, 10, 20, 20)])
        resized = resize_mask_to_image(mask, image_size=(100, 100))

        np.testing.assert_array_equal(mask, resized)
