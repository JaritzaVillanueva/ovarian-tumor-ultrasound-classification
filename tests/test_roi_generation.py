"""
Tests para la generación de ROIs y extensión del dataset con condition.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image


# Helpers

# Crea una imagen aleatoria y la guarda en el directorio dado.
def _create_image(directory: Path, image_id: str, ext: str, height: int = 100, width: int = 150):
    directory.mkdir(parents=True, exist_ok=True)
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    img.save(directory / f"{image_id}{ext}")

def _create_mask(directory: Path, image_id: str, suffix: str, height: int, width: int, regions=None):
    """
    Crea una máscara binaria. Para sufijos con 'pred', usa valores {0, 255}; para GT (_binary), usa valores {0, 1}.
    """
    directory.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((height, width), dtype=np.uint8)
    fill_value = 255 if "pred" in suffix else 1
    if regions:
        for top, left, bottom, right in regions:
            mask[top:bottom, left:right] = fill_value
    Image.fromarray(mask, mode="L").save(directory / f"{image_id}{suffix}")


def _create_csv(csv_path: Path, image_ids: list[str], split_name: str):
    rows = [
        {
            "image_id": iid,
            "image_name": f"{iid}.JPG",
            "source_split": "train",
            "original_class": "benign",
            "binary_label": 0,
            "binary_class": "benign",
            "experimental_split": split_name,
        }
        for iid in image_ids
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)


# Dataset con condition
class TestDatasetCondition:

    def test_condition_a_loads_jpg(self, tmp_path):
        from src.classification.dataset import MMOTUClassificationDataset

        _create_image(tmp_path / "train" / "images", "100", ".JPG")
        _create_csv(tmp_path / "s.csv", ["100"], "train")

        ds = MMOTUClassificationDataset(str(tmp_path / "s.csv"), str(tmp_path), "train", condition="A")
        img, label = ds[0]
        assert isinstance(img, Image.Image)

    def test_condition_b_loads_png(self, tmp_path):
        from src.classification.dataset import MMOTUClassificationDataset

        _create_image(tmp_path / "train" / "images", "100", ".png")
        _create_csv(tmp_path / "s.csv", ["100"], "train")

        ds = MMOTUClassificationDataset(str(tmp_path / "s.csv"), str(tmp_path), "train", condition="B")
        img, label = ds[0]
        assert isinstance(img, Image.Image)

    def test_condition_c_loads_png(self, tmp_path):
        from src.classification.dataset import MMOTUClassificationDataset

        _create_image(tmp_path / "train" / "images", "100", ".png")
        _create_csv(tmp_path / "s.csv", ["100"], "train")

        ds = MMOTUClassificationDataset(str(tmp_path / "s.csv"), str(tmp_path), "train", condition="C")
        img, _ = ds[0]
        assert isinstance(img, Image.Image)

    def test_default_condition_is_a(self, tmp_path):
        from src.classification.dataset import MMOTUClassificationDataset

        _create_image(tmp_path / "train" / "images", "100", ".JPG")
        _create_csv(tmp_path / "s.csv", ["100"], "train")

        ds = MMOTUClassificationDataset(str(tmp_path / "s.csv"), str(tmp_path), "train")
        assert ds.condition == "A"

    def test_condition_stored_as_attribute(self, tmp_path):
        from src.classification.dataset import MMOTUClassificationDataset

        _create_image(tmp_path / "train" / "images", "100", ".png")
        _create_csv(tmp_path / "s.csv", ["100"], "train")

        ds = MMOTUClassificationDataset(str(tmp_path / "s.csv"), str(tmp_path), "train", condition="B")
        assert ds.condition == "B"

    def test_label_unchanged_across_conditions(self, tmp_path):
        from src.classification.dataset import MMOTUClassificationDataset

        _create_image(tmp_path / "a" / "train" / "images", "100", ".JPG")
        _create_image(tmp_path / "b" / "train" / "images", "100", ".png")
        csv_path = tmp_path / "s.csv"
        _create_csv(csv_path, ["100"], "train")

        ds_a = MMOTUClassificationDataset(str(csv_path), str(tmp_path / "a"), "train", condition="A")
        ds_b = MMOTUClassificationDataset(str(csv_path), str(tmp_path / "b"), "train", condition="B")

        _, label_a = ds_a[0]
        _, label_b = ds_b[0]
        assert label_a == label_b


# Generación de ROIs 
class TestROIGenerationB:
    def test_generates_rois_normal_mask(self, tmp_path):
        from src.classification.generate_rois import generate_rois_for_split

        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        output_dir = tmp_path / "output"

        _create_image(img_dir, "100", ".JPG", 200, 300)
        _create_image(img_dir, "101", ".JPG", 200, 300)
        _create_mask(mask_dir, "100", "_pred.png", 256, 256, [(50, 50, 150, 150)])
        _create_mask(mask_dir, "101", "_pred.png", 256, 256, [(30, 30, 200, 200)])

        manifest = generate_rois_for_split(
            condition="B", split="train",
            image_dir=img_dir, mask_dir=mask_dir, output_dir=output_dir,
            mask_pattern="{stem}_pred.png",
        )

        assert len(manifest) == 2
        for row in manifest:
            assert Path(row["output_path"]).exists()
            assert row["condition"] == "B"
            assert not row["used_fallback"]

    def test_fallback_on_empty_mask(self, tmp_path):
        from src.classification.generate_rois import generate_rois_for_split

        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        output_dir = tmp_path / "output"

        _create_image(img_dir, "100", ".JPG", 200, 300)
        _create_mask(mask_dir, "100", "_pred.png", 256, 256, [])

        manifest = generate_rois_for_split(
            condition="B", split="train",
            image_dir=img_dir, mask_dir=mask_dir, output_dir=output_dir,
            mask_pattern="{stem}_pred.png",
        )

        assert manifest[0]["used_fallback"] is True
        assert manifest[0]["fallback_reason"] == "empty_mask"
        assert manifest[0]["roi_width"] == 300
        assert manifest[0]["roi_height"] == 200

    def test_fallback_on_tiny_component(self, tmp_path):
        from src.classification.generate_rois import generate_rois_for_split

        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        output_dir = tmp_path / "output"

        _create_image(img_dir, "100", ".JPG", 500, 500)
        _create_mask(mask_dir, "100", "_pred.png", 256, 256, [(0, 0, 1, 1)])

        manifest = generate_rois_for_split(
            condition="B", split="train",
            image_dir=img_dir, mask_dir=mask_dir, output_dir=output_dir,
            mask_pattern="{stem}_pred.png",
        )

        assert manifest[0]["used_fallback"] is True
        assert manifest[0]["fallback_reason"] == "tiny_component"


class TestROIGenerationC:
    def test_generates_rois_from_gt(self, tmp_path):
        from src.classification.generate_rois import generate_rois_for_split

        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        output_dir = tmp_path / "output"

        _create_image(img_dir, "100", ".JPG", 200, 300)
        _create_mask(mask_dir, "100", "_binary.PNG", 200, 300, [(30, 40, 100, 200)])

        manifest = generate_rois_for_split(
            condition="C", split="train",
            image_dir=img_dir, mask_dir=mask_dir, output_dir=output_dir,
            mask_pattern="{stem}_binary.PNG",
        )

        assert len(manifest) == 1
        assert manifest[0]["condition"] == "C"
        assert not manifest[0]["used_fallback"]
        assert manifest[0]["fallback_reason"] == ""

    def test_raises_on_empty_gt(self, tmp_path):
        from src.classification.generate_rois import generate_rois_for_split

        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        output_dir = tmp_path / "output"

        _create_image(img_dir, "100", ".JPG", 200, 300)
        _create_mask(mask_dir, "100", "_binary.PNG", 200, 300, [])

        with pytest.raises(ValueError, match="integrity error"):
            generate_rois_for_split(
                condition="C", split="train",
                image_dir=img_dir, mask_dir=mask_dir, output_dir=output_dir,
                mask_pattern="{stem}_binary.PNG",
            )

    def test_keep_largest_false_preserves_multicomponent(self, tmp_path):
        from src.classification.generate_rois import generate_rois_for_split

        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        output_dir = tmp_path / "output"

        _create_image(img_dir, "100", ".JPG", 200, 300)
        _create_mask(mask_dir, "100", "_binary.PNG", 200, 300, [
            (10, 10, 30, 30),
            (150, 250, 190, 290),
        ])

        manifest = generate_rois_for_split(
            condition="C", split="train",
            image_dir=img_dir, mask_dir=mask_dir, output_dir=output_dir,
            mask_pattern="{stem}_binary.PNG",
        )

        roi = Image.open(manifest[0]["output_path"])
        # El bbox debe cubrir ambas componentes -> ancho/alto mayor que cada componente individual
        assert roi.size[0] > 50
        assert roi.size[1] > 50


# Formato y manifest 
class TestROIFormat:
    def test_roi_saved_as_png(self, tmp_path):
        from src.classification.generate_rois import generate_rois_for_split

        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        output_dir = tmp_path / "output"

        _create_image(img_dir, "100", ".JPG", 200, 300)
        _create_mask(mask_dir, "100", "_pred.png", 256, 256, [(50, 50, 150, 150)])

        manifest = generate_rois_for_split(
            condition="B", split="train",
            image_dir=img_dir, mask_dir=mask_dir, output_dir=output_dir,
            mask_pattern="{stem}_pred.png",
        )

        output_path = Path(manifest[0]["output_path"])
        assert output_path.suffix == ".png"
        roi = Image.open(output_path)
        assert roi.format == "PNG"

    def test_roi_is_rgb(self, tmp_path):
        from src.classification.generate_rois import generate_rois_for_split

        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        output_dir = tmp_path / "output"

        _create_image(img_dir, "100", ".JPG", 200, 300)
        _create_mask(mask_dir, "100", "_pred.png", 256, 256, [(50, 50, 150, 150)])

        manifest = generate_rois_for_split(
            condition="B", split="train",
            image_dir=img_dir, mask_dir=mask_dir, output_dir=output_dir,
            mask_pattern="{stem}_pred.png",
        )

        roi = Image.open(manifest[0]["output_path"])
        assert roi.mode == "RGB"


class TestManifest:
    def test_has_all_required_columns(self, tmp_path):
        from src.classification.generate_rois import generate_rois_for_split

        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        output_dir = tmp_path / "output"

        _create_image(img_dir, "100", ".JPG", 200, 300)
        _create_mask(mask_dir, "100", "_pred.png", 256, 256, [(50, 50, 150, 150)])

        manifest = generate_rois_for_split(
            condition="B", split="train",
            image_dir=img_dir, mask_dir=mask_dir, output_dir=output_dir,
            mask_pattern="{stem}_pred.png",
        )

        required = [
            "image_id", "split", "condition", "source_image", "source_mask",
            "roi_width", "roi_height", "roi_area_fraction",
            "used_fallback", "fallback_reason", "output_path",
        ]
        for col in required:
            assert col in manifest[0], f"Falta columna: {col}"

    def test_save_manifest_creates_csv(self, tmp_path):
        from src.classification.generate_rois import save_manifest

        rows = [{
            "image_id": "100", "split": "train", "condition": "B",
            "source_image": "/a/100.JPG", "source_mask": "/b/100_pred.png",
            "roi_width": 150, "roi_height": 100, "roi_area_fraction": 0.25,
            "used_fallback": False, "fallback_reason": "", "output_path": "/c/100.png",
        }]
        csv_path = tmp_path / "manifest.csv"
        save_manifest(rows, csv_path)

        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert list(df.columns) == list(rows[0].keys())

    def test_roi_area_fraction_correct(self, tmp_path):
        from src.classification.generate_rois import generate_rois_for_split

        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        output_dir = tmp_path / "output"

        _create_image(img_dir, "100", ".JPG", 200, 300)
        _create_mask(mask_dir, "100", "_pred.png", 256, 256, [])

        manifest = generate_rois_for_split(
            condition="B", split="train",
            image_dir=img_dir, mask_dir=mask_dir, output_dir=output_dir,
            mask_pattern="{stem}_pred.png",
        )

        # Fallback -> ROI = imagen completa -> area_fraction = 1.0
        assert manifest[0]["roi_area_fraction"] == pytest.approx(1.0)

    def test_missing_mask_raises(self, tmp_path):
        from src.classification.generate_rois import generate_rois_for_split

        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        output_dir = tmp_path / "output"

        _create_image(img_dir, "100", ".JPG", 200, 300)
        mask_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(FileNotFoundError, match="Máscara no encontrada"):
            generate_rois_for_split(
                condition="B", split="train",
                image_dir=img_dir, mask_dir=mask_dir, output_dir=output_dir,
                mask_pattern="{stem}_pred.png",
            )


# ── Validación de máscaras predichas ──────────────────────────────────────────

class TestValidatePredictedMasks:

    def test_all_present_passes(self, tmp_path):
        from src.classification.generate_rois import validate_predicted_masks

        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        img_dir.mkdir(); mask_dir.mkdir()
        for i in ["100", "101"]:
            _create_image(img_dir, i, ".JPG")
            (mask_dir / f"{i}_pred.png").touch()

        validate_predicted_masks(img_dir, mask_dir)  # no debe lanzar

    def test_missing_mask_raises_file_not_found(self, tmp_path):
        from src.classification.generate_rois import validate_predicted_masks

        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        img_dir.mkdir(); mask_dir.mkdir()
        _create_image(img_dir, "100", ".JPG")
        # no se crea la máscara

        with pytest.raises(FileNotFoundError, match="faltante"):
            validate_predicted_masks(img_dir, mask_dir)

    def test_error_message_names_missing_file(self, tmp_path):
        from src.classification.generate_rois import validate_predicted_masks

        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        img_dir.mkdir(); mask_dir.mkdir()
        _create_image(img_dir, "42", ".JPG")

        with pytest.raises(FileNotFoundError, match="42_pred.png"):
            validate_predicted_masks(img_dir, mask_dir)


# ── Protección de re-ejecución ─────────────────────────────────────────────────

class TestReExecution:

    def test_roi_not_overwritten_by_default(self, tmp_path):
        from src.classification.generate_rois import generate_rois_for_split

        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        output_dir = tmp_path / "output"

        _create_image(img_dir, "100", ".JPG", 200, 300)
        _create_mask(mask_dir, "100", "_pred.png", 256, 256, [(50, 50, 150, 150)])

        generate_rois_for_split(
            condition="B", split="train",
            image_dir=img_dir, mask_dir=mask_dir, output_dir=output_dir,
            mask_pattern="{stem}_pred.png",
        )

        roi_path = output_dir / "100.png"
        sentinel = b"SENTINEL_CONTENT"
        roi_path.write_bytes(sentinel)

        generate_rois_for_split(
            condition="B", split="train",
            image_dir=img_dir, mask_dir=mask_dir, output_dir=output_dir,
            mask_pattern="{stem}_pred.png", overwrite=False,
        )

        assert roi_path.read_bytes() == sentinel

    def test_roi_overwritten_with_flag(self, tmp_path):
        from src.classification.generate_rois import generate_rois_for_split

        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        output_dir = tmp_path / "output"

        _create_image(img_dir, "100", ".JPG", 200, 300)
        _create_mask(mask_dir, "100", "_pred.png", 256, 256, [(50, 50, 150, 150)])

        generate_rois_for_split(
            condition="B", split="train",
            image_dir=img_dir, mask_dir=mask_dir, output_dir=output_dir,
            mask_pattern="{stem}_pred.png",
        )

        roi_path = output_dir / "100.png"
        roi_path.write_bytes(b"SENTINEL_CONTENT")

        generate_rois_for_split(
            condition="B", split="train",
            image_dir=img_dir, mask_dir=mask_dir, output_dir=output_dir,
            mask_pattern="{stem}_pred.png", overwrite=True,
        )

        assert roi_path.read_bytes() != b"SENTINEL_CONTENT"

    def test_manifest_row_count_unchanged_on_rerun(self, tmp_path):
        from src.classification.generate_rois import generate_rois_for_split

        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        output_dir = tmp_path / "output"

        for i in ["100", "101"]:
            _create_image(img_dir, i, ".JPG", 200, 300)
            _create_mask(mask_dir, i, "_pred.png", 256, 256, [(50, 50, 150, 150)])

        rows1 = generate_rois_for_split(
            condition="B", split="train",
            image_dir=img_dir, mask_dir=mask_dir, output_dir=output_dir,
            mask_pattern="{stem}_pred.png",
        )
        rows2 = generate_rois_for_split(
            condition="B", split="train",
            image_dir=img_dir, mask_dir=mask_dir, output_dir=output_dir,
            mask_pattern="{stem}_pred.png",
        )

        assert len(rows1) == len(rows2) == 2


# ── Salto de máscaras predichas existentes ────────────────────────────────────

class TestPredictedMaskSkipping:

    class _DummyModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3])

    def test_existing_mask_skipped_by_default(self, tmp_path):
        from src.classification.generate_rois import generate_predicted_masks

        img_dir = tmp_path / "images"
        output_dir = tmp_path / "output"
        _create_image(img_dir, "100", ".JPG", 32, 32)
        output_dir.mkdir()

        # Pre-crear la máscara con contenido centinela
        sentinel_path = output_dir / "100_pred.png"
        Image.fromarray(
            np.zeros((8, 8), dtype=np.uint8), "L"
        ).save(sentinel_path)
        sentinel_mtime = sentinel_path.stat().st_mtime

        generated, skipped = generate_predicted_masks(
            model=self._DummyModel(),
            device=torch.device("cpu"),
            image_dir=img_dir,
            output_dir=output_dir,
            overwrite=False,
        )

        assert generated == 0
        assert skipped == 1
        assert sentinel_path.stat().st_mtime == sentinel_mtime
