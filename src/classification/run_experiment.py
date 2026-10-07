"""
Punto de entrada para entrenamientos formales. Ejecuta una corrida parametrizada por arquitectura y condición.
Los hiperparámetros se leen del YAML correspondiente en configs/classification/.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import SequentialSampler
from sklearn.metrics import roc_auc_score

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


def resolve_seed(yaml_seed, cli_seed):
    """
    Devuelve (seed efectiva, origen). --seed sustituye únicamente la seed del YAML.
    """
    if cli_seed is None:
        return yaml_seed, "yaml"
    return cli_seed, "cli_override"


def save_selected_predictions(output_path, epoch, val_loader, y_true, y_prob):
    """
    Guarda las probabilidades de malignidad de validation del checkpoint seleccionado, por image_id.

    Verifica que el orden de image_id corresponde al orden efectivo del dataset evaluado:
    val_loader con shuffle=False, mismo número de predicciones y etiquetas evaluadas idénticas a
    las del dataset en ese orden.
    """
    dataset_df = val_loader.dataset.df
    image_ids = dataset_df["image_id"].tolist()
    labels = dataset_df["binary_label"].to_numpy()

    if not isinstance(val_loader.sampler, SequentialSampler):
        raise ValueError("val_loader debe usar shuffle=False para asociar las predicciones con image_id.")
    if not (len(image_ids) == len(y_true) == len(y_prob)):
        raise ValueError("El número de image_id no coincide con el número de predicciones.")
    if not np.array_equal(np.asarray(y_true), labels):
        raise ValueError("Las etiquetas evaluadas no coinciden con el orden de image_id del dataset de validation.")

    pd.DataFrame({
        "epoch": epoch,
        "image_id": image_ids,
        "y_true": np.asarray(y_true).astype(int),
        "probability_malignant": np.asarray(y_prob, dtype=float),
    }).to_csv(output_path, index=False)


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
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override opcional del learning rate del YAML. Requiere --experiment distinto de 'formal'.",
    )
    parser.add_argument(
        "--experiment",
        default="formal",
        help="Nombre del experimento; define results/classification/<experiment>/. Por defecto: formal.",
    )
    parser.add_argument(
        "--save-both-checkpoints",
        action="store_true",
        default=False,
        help=(
            "Diagnóstico: guarda best_loss.pth (min val_weighted_loss) y best_auc.pth (max val_roc_auc) "
            "de la misma trayectoria, y las probabilidades de validation por época. El early stopping "
            "no cambia. Requiere --experiment distinto de 'formal'."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Override opcional de la seed del YAML. La salida se guarda en .../<cond>/seed_<seed>/. "
            "Requiere --experiment distinto de 'formal'."
        ),
    )
    parser.add_argument(
        "--checkpoint-metric",
        choices=["val_loss", "val_roc_auc"],
        default="val_loss",
        help=(
            "Criterio del checkpoint oficial: val_loss (min val_weighted_loss, por defecto) o "
            "val_roc_auc (max val_roc_auc). El early stopping sigue basado en val_weighted_loss. "
            "val_roc_auc requiere --experiment distinto de 'formal'."
        ),
    )
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.experiment):
        parser.error("--experiment solo admite letras, números, '_', '-' y '.'.")
    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            parser.error("--learning-rate debe ser positivo.")
        if args.experiment == "formal":
            parser.error("--learning-rate requiere --experiment distinto de 'formal' para no mezclar corridas formales.")
    if args.save_both_checkpoints and args.experiment == "formal":
        parser.error("--save-both-checkpoints requiere --experiment distinto de 'formal' para no mezclar corridas formales.")
    if args.seed is not None and args.experiment == "formal":
        parser.error("--seed requiere --experiment distinto de 'formal' para no mezclar corridas formales.")
    if args.checkpoint_metric != "val_loss" and args.experiment == "formal":
        parser.error("--checkpoint-metric val_roc_auc requiere --experiment distinto de 'formal'.")
    if args.checkpoint_metric != "val_loss" and args.save_both_checkpoints:
        parser.error("--save-both-checkpoints solo se usa con --checkpoint-metric val_loss (modo diagnóstico).")

    project_root = Path.cwd()
    csv_path = project_root / CSV_REL
    image_dir = project_root / CONDITION_DIRS[args.condition]
    config_path = project_root / ARCH_CONFIGS[args.arch]
    output_dir = project_root / "results" / "classification" / args.experiment / args.arch / args.condition
    if args.seed is not None:
        output_dir = output_dir / f"seed_{args.seed}"

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

    seed, seed_source = resolve_seed(cfg["seed"], args.seed)
    epochs = cfg["training"]["epochs"]
    batch_size = cfg["training"]["batch_size"]
    lr = cfg["training"]["learning_rate"]
    lr_source = "yaml"
    if args.learning_rate is not None:
        lr = args.learning_rate
        lr_source = "cli_override"
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
    print(f"  Experimento:  {args.experiment}")
    print(f"  Learning rate: {lr} ({lr_source})")
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
    diagnostic = args.save_both_checkpoints
    select_by_auc = args.checkpoint_metric == "val_roc_auc"
    if diagnostic:
        checkpoint_name = "best_loss"
    elif select_by_auc:
        checkpoint_name = "best_auc"
    else:
        checkpoint_name = "best"
    save_path = str(output_dir / f"{checkpoint_name}.pth")

    diagnostic_kwargs = {}
    if diagnostic:
        # El orden de los image_id se toma del mismo dataset que recorre val_loader (shuffle=False);
        # train_model verifica el sampler, la longitud y las etiquetas antes de escribir el CSV.
        diagnostic_kwargs = dict(
            auc_save_path=str(output_dir / "best_auc.pth"),
            predictions_path=str(output_dir / "val_predictions.csv"),
            val_image_ids=val_loader.dataset.df["image_id"].tolist(),
            val_expected_labels=val_loader.dataset.df["binary_label"].to_numpy(),
        )

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
        checkpoint_metric=args.checkpoint_metric,
        **diagnostic_kwargs,
    )

    # Guardar training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Recargar checkpoint(s) para evaluación formal en validation
    evaluated_predictions = {}

    def evaluate_checkpoint(checkpoint_path):
        eval_model = create_model(
            pretrained=False,
            num_classes=num_classes,
            freeze_backbone=False,
        ).to(device)
        eval_model.load_state_dict( torch.load(checkpoint_path, map_location=device, weights_only=True), )

        y_true, y_pred, y_prob = predict(eval_model, val_loader, device)
        metrics = evaluate_predictions(y_true, y_pred, y_prob)
        evaluated_predictions[checkpoint_path] = (y_true, y_prob)

        return {
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

    if diagnostic:
        checkpoints_to_evaluate = {
            "best_loss": (save_path, "validation_metrics_best_loss.json"),
            "best_auc": (diagnostic_kwargs["auc_save_path"], "validation_metrics_best_auc.json"),
        }
    else:
        checkpoints_to_evaluate = {"best": (save_path, "validation_metrics.json")}

    evaluated_metrics = {}
    for tag, (checkpoint_path, metrics_filename) in checkpoints_to_evaluate.items():
        print(f"\nRecargando {Path(checkpoint_path).name} para evaluación formal en validation...")
        val_metrics = evaluate_checkpoint(checkpoint_path)
        evaluated_metrics[tag] = val_metrics

        with open(output_dir / metrics_filename, "w") as f:
            json.dump(val_metrics, f, indent=2)

        print(f"\n  Métricas de validation ({tag}):")
        for k, v in val_metrics.items():
            if k != "confusion_matrix":
                print(f"    {k}: {v:.4f}")
        print(f"    confusion_matrix: {val_metrics['confusion_matrix']}")

    # Leer best_meta para run_config 
    meta_path = output_dir / f"{checkpoint_name}_meta.json"
    with open(meta_path) as f:
        best_meta = json.load(f)

    # Probabilidades por imagen del checkpoint seleccionado (modo checkpoint por AUC)
    selected_predictions_name = None
    if select_by_auc:
        selected_predictions_name = "val_predictions_best_auc.csv"
        sel_y_true, sel_y_prob = evaluated_predictions[save_path]
        save_selected_predictions(
            output_dir / selected_predictions_name, best_meta["epoch"], val_loader, sel_y_true, sel_y_prob,
        )

    # Con checkpoint por AUC, los valores de pérdida y accuracy no son mínimos/máximos de esas métricas:
    # se registran como valores en la época del checkpoint seleccionado.
    if select_by_auc:
        checkpoint_value_keys = (
            "val_weighted_loss_at_best_checkpoint",
            "val_unweighted_loss_at_best_checkpoint",
            "val_acc_at_best_checkpoint",
        )
    else:
        checkpoint_value_keys = ("val_weighted_loss_best", "val_unweighted_loss_best", "val_acc_best")

    # Guardar run_config con trazabilidad completa 
    epochs_run = len(history["train_loss"])
    early_stopped = epochs_run < epochs

    run_config = {
        "architecture": args.arch,
        "condition": args.condition,
        "experiment": args.experiment,
        "seed": seed,
        "seed_source": seed_source,
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
        "learning_rate_source": lr_source,
        "weight_decay": weight_decay,
        "optimizer": "AdamW",
        "use_amp": use_amp,
        "patience": patience,
        "min_delta": min_delta,
        "threshold": 0.5,
        "scheduler": None,
        "checkpoint_criterion": "min val_weighted_loss (absolute minimum)",
        "epoch_best": best_meta["epoch"],
        checkpoint_value_keys[0]: best_meta["val_weighted_loss"],
        checkpoint_value_keys[1]: best_meta["val_unweighted_loss"],
        checkpoint_value_keys[2]: best_meta["val_acc"],
        "early_stopped": early_stopped,
        "early_stopped_at_epoch": epochs_run if early_stopped else None,
        "history_metrics": list(history.keys()),
    }
    if select_by_auc:
        run_config["checkpoint_criterion"] = "max val_roc_auc (first epoch among ties)"
        run_config["checkpoint_file"] = f"{checkpoint_name}.pth"
        run_config["val_roc_auc_best"] = best_meta["val_roc_auc"]
        run_config["early_stopping_criterion"] = (
            f"val_weighted_loss (patience={patience}, min_delta={min_delta})"
        )
        run_config["val_predictions"] = selected_predictions_name
    auc_meta = None
    if diagnostic:
        with open(output_dir / "best_auc_meta.json") as f:
            auc_meta = json.load(f)

        run_config["checkpoint_diagnostic"] = {
            "early_stopping_criterion": f"val_weighted_loss (patience={patience}, min_delta={min_delta})",
            "best_loss": {
                "epoch": best_meta["epoch"],
                "value": best_meta["val_weighted_loss"],
                "metric": "val_weighted_loss",
                "checkpoint": "best_loss.pth",
            },
            "best_auc": {
                "epoch": auc_meta["epoch"],
                "value": auc_meta["val_roc_auc"],
                "metric": "val_roc_auc",
                "checkpoint": "best_auc.pth",
            },
            "best_auc_is_last_epoch": auc_meta["epoch"] == epochs_run,
            "val_predictions": "val_predictions.csv",
            "validation_metrics": {
                "best_loss": "validation_metrics_best_loss.json",
                "best_auc": "validation_metrics_best_auc.json",
            },
        }

    with open(output_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # Verificar consistencia best_meta y run_config 
    assert run_config["epoch_best"] == best_meta["epoch"], \
        f"Inconsistencia epoch_best: run_config={run_config['epoch_best']} vs best_meta={best_meta['epoch']}"
    assert run_config[checkpoint_value_keys[0]] == best_meta["val_weighted_loss"], \
        f"Inconsistencia val_weighted_loss"

    # Verificar que el checkpoint por AUC corresponde a la época declarada
    if select_by_auc:
        assert best_meta["criterion"] == "max val_roc_auc", "best_auc_meta.json no declara el criterio AUC"
        assert best_meta["val_roc_auc"] == max(history["val_roc_auc"]), \
            "best_auc no coincide con el máximo de val_roc_auc del historial"
        assert best_meta["epoch"] == int(np.argmax(history["val_roc_auc"])) + 1, \
            "La época de best_auc no es la primera época con el máximo de val_roc_auc"
        assert abs(evaluated_metrics["best"]["roc_auc"] - best_meta["val_roc_auc"]) < 1e-4, (
            f"La AUC del checkpoint recargado ({evaluated_metrics['best']['roc_auc']:.6f}) no coincide con "
            f"la registrada en su época ({best_meta['val_roc_auc']:.6f})"
        )
        print("\n  Verificación del checkpoint por AUC: OK")

    # Verificar que ambos checkpoints corresponden a la época declarada de la misma trayectoria
    if diagnostic:
        predictions_df = pd.read_csv(output_dir / "val_predictions.csv")
        expected_rows = epochs_run * len(ds_val)
        assert len(predictions_df) == expected_rows, \
            f"val_predictions.csv: {len(predictions_df)} filas, se esperaban {expected_rows}"

        assert auc_meta["val_roc_auc"] == max(history["val_roc_auc"]), \
            "best_auc no coincide con el máximo de val_roc_auc del historial"
        assert history["val_roc_auc"][auc_meta["epoch"] - 1] == auc_meta["val_roc_auc"], \
            "val_roc_auc del historial no coincide con best_auc_meta"
        assert best_meta["val_weighted_loss"] == min(history["val_loss"]), \
            "best_loss no coincide con el mínimo de val_loss del historial"

        for tag, epoch_selected in (("best_loss", best_meta["epoch"]), ("best_auc", auc_meta["epoch"])):
            epoch_rows = predictions_df[predictions_df["epoch"] == epoch_selected]
            csv_auc = roc_auc_score(epoch_rows["y_true"], epoch_rows["probability_malignant"])
            reloaded_auc = evaluated_metrics[tag]["roc_auc"]
            assert abs(csv_auc - reloaded_auc) < 1e-4, (
                f"{tag}: la AUC del checkpoint recargado ({reloaded_auc:.6f}) no coincide con la AUC de "
                f"val_predictions.csv en la época {epoch_selected} ({csv_auc:.6f})"
            )
        print("\n  Verificación de ambos checkpoints contra val_predictions.csv: OK")

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
