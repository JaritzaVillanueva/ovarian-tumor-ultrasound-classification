"""
Materialización de ROIs para condiciones B (máscara predicha) y C (máscara GT).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.segmentation.roi import (
    MIN_COMPONENT_AREA_FRACTION,
    extract_roi_from_mask,
    predict_binary_mask,
)


def validate_predicted_masks(
    image_dir: Path,
    mask_dir: Path,
    mask_suffix: str = "_pred.png",
) -> None:
    """
    Verifica que cada imagen en image_dir tiene su máscara predicha en mask_dir.
    Lanza FileNotFoundError con lista de faltantes si alguna no existe.
    """
    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file()
    )
    missing = [
        mask_dir / f"{p.stem}{mask_suffix}"
        for p in image_paths
        if not (mask_dir / f"{p.stem}{mask_suffix}").exists()
    ]
    if missing:
        names = "\n  ".join(m.name for m in missing[:10])
        extra = f" (y {len(missing) - 10} más)" if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"{len(missing)} máscara(s) predicha(s) faltante(s) en {mask_dir}:\n  {names}{extra}"
        )


def generate_predicted_masks(
    model: torch.nn.Module,
    device: torch.device,
    image_dir: Path,
    output_dir: Path,
    image_size: tuple[int, int] = (256, 256),
    threshold: float = 0.5,
    overwrite: bool = False,
) -> tuple[int, int]:
    """
    Ejecuta inferencia con el U-Net sobre todas las imágenes de un directorio.
    Guarda máscaras como {stem}_pred.png (256×256, valores {0, 255}).
    Si overwrite=False, omite archivos ya existentes.
    Retorna (generated, skipped).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file()
    )
    generated = 0
    skipped = 0

    for image_path in image_paths:
        out_path = output_dir / f"{image_path.stem}_pred.png"
        if not overwrite and out_path.exists():
            skipped += 1
            continue
        image = Image.open(image_path).convert("RGB")
        binary_mask, _ = predict_binary_mask(
            model=model,
            image=image,
            device=device,
            image_size=image_size,
            threshold=threshold,
        )
        mask_255 = (binary_mask * 255).astype(np.uint8)
        Image.fromarray(mask_255, mode="L").save(out_path)
        generated += 1

    return generated, skipped


def generate_rois_for_split(
    condition: str,
    split: str,
    image_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    mask_pattern: str,
    margin_ratio: float = 0.10,
    overwrite: bool = False,
) -> list[dict]:
    """
    Genera ROIs recortadas para un split bajo una condición.
    Si overwrite=False, reutiliza ROIs ya existentes sin sobrescribirlas.
    Siempre retorna el manifest completo (re-procesa máscaras para extraer metadatos).
    """
    if condition == "B":
        keep_largest = True
        fallback_to_full_image = True
        min_component_area_fraction = MIN_COMPONENT_AREA_FRACTION
    elif condition == "C":
        keep_largest = False
        fallback_to_full_image = False
        min_component_area_fraction = 0.0
    else:
        raise ValueError(f"condition debe ser 'B' o 'C', recibido: '{condition}'")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file()
    )
    manifest_rows: list[dict] = []

    for image_path in image_paths:
        image_id = image_path.stem
        mask_name = mask_pattern.format(stem=image_id)
        mask_path = mask_dir / mask_name

        if not mask_path.exists():
            raise FileNotFoundError(f"Máscara no encontrada: {mask_path}")

        image = Image.open(image_path).convert("RGB")
        mask_array = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        mask_binary = (mask_array > 0).astype(np.uint8)

        if condition == "C" and mask_binary.sum() == 0:
            raise ValueError(
                f"La máscara GT está vacía para imagen {image_id} "
                f"— integrity error en el dataset."
            )

        roi_result = extract_roi_from_mask(
            image=image,
            binary_mask=mask_binary,
            margin_ratio=margin_ratio,
            keep_largest=keep_largest,
            fallback_to_full_image=fallback_to_full_image,
            min_component_area_fraction=min_component_area_fraction,
        )

        output_path = output_dir / f"{image_id}.png"
        if not output_path.exists() or overwrite:
            roi_result.roi.save(output_path, format="PNG")

        roi_w, roi_h = roi_result.roi.size
        img_w, img_h = image.size
        img_area = img_w * img_h
        roi_area_fraction = (roi_w * roi_h) / img_area if img_area > 0 else 0.0

        fallback_reason = ""
        if roi_result.used_fallback:
            if roi_result.binary_mask.sum() == 0:
                fallback_reason = "empty_mask"
            else:
                fallback_reason = "tiny_component"

        manifest_rows.append({
            "image_id": image_id,
            "split": split,
            "condition": condition,
            "source_image": str(image_path),
            "source_mask": str(mask_path),
            "roi_width": roi_w,
            "roi_height": roi_h,
            "roi_area_fraction": round(roi_area_fraction, 6),
            "used_fallback": roi_result.used_fallback,
            "fallback_reason": fallback_reason,
            "output_path": str(output_path),
        })

    return manifest_rows


def save_manifest(rows: list[dict], output_path: Path) -> None:
    """Guarda el manifest de trazabilidad como CSV (sobreescribe si existe)."""
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera ROIs para condiciones B y C.")
    parser.add_argument(
        "--project-root", type=Path, default=None,
        help="Raíz del proyecto. Si no se proporciona, se detecta automáticamente.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Sobreescribe archivos ya generados. Por defecto, reutiliza existentes.",
    )
    args = parser.parse_args()

    if args.project_root:
        project_root = args.project_root.resolve()
    else:
        project_root = Path(__file__).resolve().parents[2]

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    data_dir = project_root / "data" / "processed" / "MMOTU_OTU_2D_experimental"
    results_seg = project_root / "results" / "segmentation" / "unet_baseline"
    model_path = project_root / "models" / "segmentation" / "unet_baseline_best.pth"
    roi_pred_dir = project_root / "data" / "processed" / "MMOTU_OTU_2D_roi_pred"
    roi_gt_dir = project_root / "data" / "processed" / "MMOTU_OTU_2D_roi_gt"
    manifest_dir = project_root / "results" / "classification" / "roi_generation"

    val_pred_mask_dir = results_seg / "validation" / "predicted_masks"
    train_pred_mask_dir = results_seg / "train" / "predicted_masks"
    train_image_dir = data_dir / "train" / "images"

    # ── Verificar máscaras de validation (nunca re-inferir) ───────────────
    print("Verificando máscaras predichas de validation (R4)...")
    validate_predicted_masks(
        image_dir=data_dir / "validation" / "images",
        mask_dir=val_pred_mask_dir,
    )
    n_val = len(list(val_pred_mask_dir.glob("*_pred.png")))
    print(f"  OK — {n_val} máscaras verificadas.")

    # ── Generar máscaras predichas de train (solo las faltantes) ─────────
    train_images = sorted(
        p for p in train_image_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file()
    )
    missing_train_masks = [
        p for p in train_images
        if not (train_pred_mask_dir / f"{p.stem}_pred.png").exists()
    ]

    if missing_train_masks or args.overwrite:
        n_to_generate = len(train_images) if args.overwrite else len(missing_train_masks)
        print(f"Generando máscaras predichas para train ({n_to_generate} a generar)...")
        from src.segmentation.unet import UNet

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        image_size = tuple(checkpoint.get("image_size", (256, 256)))
        threshold = float(checkpoint.get("threshold", 0.5))

        model = UNet(in_channels=3, out_channels=1).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        generated, skipped = generate_predicted_masks(
            model=model, device=device,
            image_dir=train_image_dir,
            output_dir=train_pred_mask_dir,
            image_size=image_size, threshold=threshold,
            overwrite=args.overwrite,
        )
        print(f"  {generated} nuevas, {skipped} reutilizadas.")
    else:
        print("Todas las máscaras de train ya existen, saltando inferencia.")

    # Validar que todos los IDs de train tienen máscara (IDs coinciden con imágenes)
    print("Validando cobertura de máscaras de train...")
    validate_predicted_masks(
        image_dir=train_image_dir,
        mask_dir=train_pred_mask_dir,
    )
    n_train = len(list(train_pred_mask_dir.glob("*_pred.png")))
    print(f"  OK — {n_train} máscaras verificadas.")

    all_manifest_rows: list[dict] = []

    # ── Condición B ───────────────────────────────────────────────────────
    for split, mask_dir in [("train", train_pred_mask_dir), ("validation", val_pred_mask_dir)]:
        print(f"Generando ROIs condición B — {split}...")
        rows = generate_rois_for_split(
            condition="B", split=split,
            image_dir=data_dir / split / "images",
            mask_dir=mask_dir,
            output_dir=roi_pred_dir / split / "images",
            mask_pattern="{stem}_pred.png",
            overwrite=args.overwrite,
        )
        all_manifest_rows.extend(rows)
        print(f"  {len(rows)} ROIs procesadas.")

    # ── Condición C ───────────────────────────────────────────────────────
    for split in ["train", "validation"]:
        print(f"Generando ROIs condición C — {split}...")
        rows = generate_rois_for_split(
            condition="C", split=split,
            image_dir=data_dir / split / "images",
            mask_dir=data_dir / split / "masks",
            output_dir=roi_gt_dir / split / "images",
            mask_pattern="{stem}_binary.PNG",
            overwrite=args.overwrite,
        )
        all_manifest_rows.extend(rows)
        print(f"  {len(rows)} ROIs procesadas.")

    # ── Manifests (sobreescriben en cada ejecución, sin duplicados) ───────
    manifest_b = [r for r in all_manifest_rows if r["condition"] == "B"]
    manifest_c = [r for r in all_manifest_rows if r["condition"] == "C"]

    save_manifest(manifest_b, manifest_dir / "manifest_B_roi_pred.csv")
    save_manifest(manifest_c, manifest_dir / "manifest_C_roi_gt.csv")
    save_manifest(all_manifest_rows, manifest_dir / "manifest_all.csv")

    print(f"\nManifests guardados en {manifest_dir}/")
    n_fallback = sum(1 for r in manifest_b if r["used_fallback"])
    print(f"Condición B: {len(manifest_b)} ROIs, {n_fallback} con fallback.")
    print(f"Condición C: {len(manifest_c)} ROIs.")


if __name__ == "__main__":
    main()
