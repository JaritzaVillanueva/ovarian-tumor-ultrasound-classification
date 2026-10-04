"""
Punto de entrada para entrenamientos formales. Ejecuta una corrida parametrizada por arquitectura y condición.
Los hiperparámetros se leen del YAML correspondiente en configs/classification/.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.classification.dataset import MMOTUClassificationDataset
from src.classification.dataloader import create_dataloader
from src.classification.evaluate import predict, evaluate_predictions
from src.classification.models import (
    create_resnet50_classifier,
    create_densenet121_classifier,
    create_convnext_tiny_classifier,
)
from src.classification.train import set_seed, train_model
from src.classification.transforms import get_train_transforms, get_val_transforms
from src.classification.utils import compute_class_weights

ARCH_CONFIGS = {
    "resnet50": "configs/classification/resnet50_baseline.yaml",
    "densenet121": "configs/classification/densenet121_baseline.yaml",
    "convnext_tiny": "configs/classification/convnext_tiny_baseline.yaml",
}

ARCH_CREATORS = {
    "resnet50": create_resnet50_classifier,
    "densenet121": create_densenet121_classifier,
    "convnext_tiny": create_convnext_tiny_classifier,
}

CONDITION_DIRS = {
    "A": "data/processed/MMOTU_OTU_2D_experimental",
    "B": "data/processed/MMOTU_OTU_2D_roi_pred",
    "C": "data/processed/MMOTU_OTU_2D_roi_gt",
}

CSV_REL = "configs/splits/mmotu_experimental_split_standard.csv"


def main() -> None:
    parser = argparse.ArgumentParser( description="Entrenamiento formal: architecture × condition.", )
    parser.add_argument(
        "--arch",
        required=True,
        choices=list(ARCH_CONFIGS.keys()),
        help="Arquitectura CNN.",
    )
    parser.add_argument(
        "--condition",
        required=True,
        choices=["A", "B", "C"],
        help="Condición experimental: A (full), B (ROI pred), C (ROI GT).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Limpia artefactos existentes antes de ejecutar.",
    )
    args = parser.parse_args()

    project_root = Path.cwd()
    csv_path = project_root / CSV_REL
    image_dir = project_root / CONDITION_DIRS[args.condition]
    config_path = project_root / ARCH_CONFIGS[args.arch]
    output_dir = project_root / "results" / "classification" / "formal" / args.arch / args.condition

    # Validaciones previas
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV no encontrado: {csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Directorio de imágenes no encontrado: {image_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"YAML de configuración no encontrado: {config_path}")

    # Protección de corridas formales
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            print(f"Error: {output_dir} ya contiene artefactos.")
            print("Use --overwrite para limpiar y re-ejecutar.")
            return
        print(f"--overwrite: limpiando {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar config desde YAML
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    epochs = cfg["training"]["epochs"]
    batch_size = cfg["training"]["batch_size"]
    lr = cfg["training"]["learning_rate"]
    weight_decay = cfg["training"]["weight_decay"]
    use_amp = cfg["training"]["use_amp"]
    patience = cfg["early_stopping"]["patience"]
    min_delta = cfg["early_stopping"]["min_delta"]
    pretrained = cfg["model"]["pretrained"]
    num_classes = cfg["model"]["num_classes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*60}")
    print(f"ENTRENAMIENTO FORMAL")
    print(f"  Arquitectura: {args.arch}")
    print(f"  Condición:    {args.condition}")
    print(f"  Device:       {device}")
    print(f"  Output:       {output_dir}")
    print(f"{'='*60}")

    # Datasets
    set_seed(seed)

    ds_train = MMOTUClassificationDataset(
        csv_file=str(csv_path),
        image_dir=str(image_dir),
        split="train",
        transform=get_train_transforms(),
        condition=args.condition,
    )
    ds_val = MMOTUClassificationDataset(
        csv_file=str(csv_path),
        image_dir=str(image_dir),
        split="validation",
        transform=get_val_transforms(),
        condition=args.condition,
    )
    print(f"  Train: {len(ds_train)} | Validation: {len(ds_val)}")

    # Dataloaders
    train_loader = create_dataloader( ds_train, batch_size=batch_size, shuffle=True, num_workers=4, seed=seed, )
    val_loader = create_dataloader( ds_val, batch_size=batch_size, shuffle=False, num_workers=4, seed=seed, )

    # Class weights (desde train) 
    train_labels = ds_train.df["binary_label"].values
    class_weights = compute_class_weights(train_labels).to(device)
    print(f"  Class weights: {class_weights}")

    # Modelo 
    set_seed(seed)
    create_model = ARCH_CREATORS[args.arch]
    model = create_model( pretrained=pretrained, num_classes=num_classes, freeze_backbone=False, ).to(device)

    # Criterion y optimizer 
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Entrenamiento 
    save_path = str(output_dir / "best.pth")

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        save_path=save_path,
        patience=patience,
        min_delta=min_delta,
        seed=seed,
        use_amp=use_amp,
    )

    # Guardar training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Recargar best checkpoint para evaluación formal
    print("\nRecargando best.pth para evaluación formal en validation...")
    eval_model = create_model(
        pretrained=False,
        num_classes=num_classes,
        freeze_backbone=False,
    ).to(device)
    eval_model.load_state_dict( torch.load(save_path, map_location=device, weights_only=True), )

    y_true, y_pred, y_prob = predict(eval_model, val_loader, device)
    metrics = evaluate_predictions(y_true, y_pred, y_prob)

    # Guardar validation metrics
    val_metrics = {
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "precision_malignant": metrics["precision_malignant"],
        "recall_malignant": metrics["recall_malignant"],
        "specificity": metrics["specificity"],
        "f1_malignant": metrics["f1_malignant"],
        "roc_auc": metrics["roc_auc"],
        "average_precision": metrics["average_precision"],
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
    }
    with open(output_dir / "validation_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=2)

    print(f"\n  Métricas de validation (best checkpoint):")
    for k, v in val_metrics.items():
        if k != "confusion_matrix":
            print(f"    {k}: {v:.4f}")
    print(f"    confusion_matrix: {val_metrics['confusion_matrix']}")

    # Leer best_meta para run_config 
    meta_path = output_dir / "best_meta.json"
    with open(meta_path) as f:
        best_meta = json.load(f)

    # Guardar run_config con trazabilidad completa 
    epochs_run = len(history["train_loss"])
    early_stopped = epochs_run < epochs

    run_config = {
        "architecture": args.arch,
        "condition": args.condition,
        "seed": seed,
        "csv_path": CSV_REL,
        "image_dir": CONDITION_DIRS[args.condition],
        "train_samples": len(ds_train),
        "validation_samples": len(ds_val),
        "class_weights": class_weights.cpu().tolist(),
        "pretrained": pretrained,
        "freeze_backbone": False,
        "num_classes": num_classes,
        "batch_size": batch_size,
        "epochs_max": epochs,
        "epochs_run": epochs_run,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "optimizer": "AdamW",
        "use_amp": use_amp,
        "patience": patience,
        "min_delta": min_delta,
        "threshold": 0.5,
        "scheduler": None,
        "checkpoint_criterion": "min val_weighted_loss (absolute minimum)",
        "epoch_best": best_meta["epoch"],
        "val_weighted_loss_best": best_meta["val_weighted_loss"],
        "val_unweighted_loss_best": best_meta["val_unweighted_loss"],
        "val_acc_best": best_meta["val_acc"],
        "early_stopped": early_stopped,
        "early_stopped_at_epoch": epochs_run if early_stopped else None,
    }
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # Verificar consistencia best_meta y run_config 
    assert run_config["epoch_best"] == best_meta["epoch"], \
        f"Inconsistencia epoch_best: run_config={run_config['epoch_best']} vs best_meta={best_meta['epoch']}"
    assert run_config["val_weighted_loss_best"] == best_meta["val_weighted_loss"], \
        f"Inconsistencia val_weighted_loss"

    print(f"\n  Época del mejor checkpoint: {run_config['epoch_best']}")
    print(f"  Early stopped: {run_config['early_stopped']}", end="")
    if early_stopped:
        print(f" (época {run_config['early_stopped_at_epoch']})")
    else:
        print()

    print(f"\nArtefactos guardados en {output_dir}/:")
    for p in sorted(output_dir.iterdir()):
        print(f"  {p.name} ({p.stat().st_size / 1024:.0f} KB)")

    print(f"\n{'='*60}")
    print(f"COMPLETADO: {args.arch} — Condición {args.condition}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
