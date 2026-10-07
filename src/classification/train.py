from __future__ import annotations

import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import SequentialSampler

from src.classification.evaluate import evaluate_predictions

# Métricas de validation registradas por época (clave del historial -> clave de evaluate_predictions).
EPOCH_VALIDATION_METRICS = {
    "val_roc_auc": "roc_auc",
    "val_average_precision": "average_precision",
    "val_f1_malignant": "f1_malignant",
    "val_recall_malignant": "recall_malignant",
    "val_specificity": "specificity",
    "val_balanced_accuracy": "balanced_accuracy",
}


def set_seed(seed: int) -> None:
    """
    Establece la semilla para reproducibilidad.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """
    Ejecuta una época de entrenamiento.
    """
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total


def validate_one_epoch(model, dataloader, criterion, device, return_predictions=False):
    """
    Ejecuta una época de validación.

    Con return_predictions=True devuelve además las etiquetas y las probabilidades de la clase
    maligna obtenidas en el mismo forward de validación (no se ejecuta un segundo pase).
    """
    model.eval()
    unweighted_criterion = nn.CrossEntropyLoss()

    running_weighted_loss = 0.0
    running_unweighted_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_prob = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            weighted_loss = criterion(outputs, labels)
            unweighted_loss = unweighted_criterion(outputs, labels)

            running_weighted_loss += weighted_loss.item() * images.size(0)
            running_unweighted_loss += unweighted_loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if return_predictions:
                y_true.extend(labels.cpu().numpy())
                y_prob.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

    results = ( running_weighted_loss / total, running_unweighted_loss / total, correct / total, )

    if return_predictions:
        return results + ( np.array(y_true), np.array(y_prob), )

    return results


def compute_epoch_validation_metrics(y_true, y_prob):
    """
    Calcula las métricas de validation registradas por época (umbral 0.5, igual que evaluate.predict).

    Si validation contiene una sola clase, las métricas no están definidas y se devuelven como None.
    """
    if len(np.unique(y_true)) < 2:
        return {key: None for key in EPOCH_VALIDATION_METRICS}

    y_pred = (y_prob >= 0.5).astype(int)
    metrics = evaluate_predictions(y_true, y_pred, y_prob)

    return {key: float(metrics[source]) for key, source in EPOCH_VALIDATION_METRICS.items()}


def _save_checkpoint_meta(save_path, epoch, val_weighted_loss, val_unweighted_loss, val_acc, extra=None):
    """
    Guarda un archivo JSON con metadatos del checkpoint.

    extra: campos adicionales opcionales (solo se usan en el modo de diagnóstico de checkpoints).
    """
    meta_path = os.path.splitext(save_path)[0] + "_meta.json"
    meta = {
        "epoch": epoch,
        "val_weighted_loss": val_weighted_loss,
        "val_unweighted_loss": val_unweighted_loss,
        "val_acc": val_acc,
    }
    if extra:
        meta.update(extra)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


VAL_PREDICTIONS_COLUMNS = ["epoch", "image_id", "y_true", "probability_malignant"]


def _check_val_prediction_order(val_loader, val_image_ids, val_expected_labels):
    """
    Verifica que el orden de las predicciones de validation corresponde al de val_image_ids.

    - val_loader debe recorrer el dataset en orden secuencial (shuffle=False).
    - val_image_ids y val_expected_labels deben tener un elemento por muestra del dataset y los
      image_id no deben repetirse.
    """
    if not isinstance(val_loader.sampler, SequentialSampler):
        raise ValueError(
            "val_loader debe usar shuffle=False (SequentialSampler) para asociar las predicciones "
            "con val_image_ids."
        )

    n_samples = len(val_loader.dataset)
    if len(val_image_ids) != n_samples or len(val_expected_labels) != n_samples:
        raise ValueError(
            f"val_image_ids ({len(val_image_ids)}) y val_expected_labels ({len(val_expected_labels)}) "
            f"deben tener el mismo número de elementos que el dataset de validation ({n_samples})."
        )

    if len(set(val_image_ids)) != len(val_image_ids):
        raise ValueError("val_image_ids contiene identificadores repetidos.")


def _init_val_predictions_file(predictions_path):
    """
    Crea el CSV de probabilidades de validation por época (solo encabezado).
    """
    with open(predictions_path, "w", newline="") as f:
        csv.writer(f).writerow(VAL_PREDICTIONS_COLUMNS)


def _append_val_predictions(predictions_path, epoch, image_ids, y_true, y_prob):
    """
    Añade al CSV las probabilidades de malignidad de validation de una época (epoch empieza en 1).
    """
    with open(predictions_path, "a", newline="") as f:
        writer = csv.writer(f)
        for image_id, label, probability in zip(image_ids, y_true, y_prob):
            writer.writerow([epoch, image_id, int(label), float(probability)])


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    save_path,
    patience: int = 5, 
    min_delta: float = 0.001,
    seed: int = 42,
    use_amp: bool = True,
    auc_save_path=None,
    predictions_path=None,
    val_image_ids=None,
    val_expected_labels=None,
    checkpoint_metric: str = "val_loss",
):
    """
    Entrena un modelo de clasificación.

    checkpoint_metric define el checkpoint oficial guardado en save_path:
        "val_loss"    -> min val_weighted_loss (comportamiento original, por defecto);
        "val_roc_auc" -> max val_roc_auc (primera época en caso de empate).
    El early stopping se basa siempre en val_weighted_loss, sea cual sea checkpoint_metric.

    Opciones de diagnóstico (opt-in; por defecto no cambian nada):
        auc_save_path: guarda además un segundo checkpoint seleccionado por max val_roc_auc, dentro
            de la misma ejecución y trayectoria. No interviene en el early stopping.
        predictions_path: CSV con epoch, image_id, y_true y probability_malignant de validation por
            época. Requiere val_image_ids y val_expected_labels (en el orden del dataset de
            validation) y un val_loader con shuffle=False.
    """
    if checkpoint_metric not in ("val_loss", "val_roc_auc"):
        raise ValueError(f"checkpoint_metric no soportado: {checkpoint_metric}")
    if checkpoint_metric == "val_roc_auc" and auc_save_path is not None:
        raise ValueError("auc_save_path solo se usa con checkpoint_metric='val_loss' (modo diagnóstico).")

    if predictions_path is not None:
        if val_image_ids is None or val_expected_labels is None:
            raise ValueError("predictions_path requiere val_image_ids y val_expected_labels.")
        _check_val_prediction_order(val_loader, val_image_ids, val_expected_labels)
        _init_val_predictions_file(predictions_path)

    set_seed(seed)

    _use_amp = use_amp and torch.cuda.is_available()
    scaler = GradScaler() if _use_amp else None

    # best_checkpoint_loss: mínimo absoluto de val_weighted_loss
    # early_stop_reference: solo se actualiza cuando la mejora supera min_delta
    best_checkpoint_loss = float("inf")
    best_checkpoint_auc = float("-inf")  # se usa con checkpoint_metric="val_roc_auc" o auc_save_path
    early_stop_reference = float("inf")
    patience_counter = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_unweighted_loss": [],
        "val_acc": [],
    }
    for key in EPOCH_VALIDATION_METRICS:
        history[key] = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler=scaler
        )
        val_weighted_loss, val_unweighted_loss, val_acc, val_y_true, val_y_prob = validate_one_epoch(
            model, val_loader, criterion, device, return_predictions=True
        )
        epoch_metrics = compute_epoch_validation_metrics(val_y_true, val_y_prob)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_weighted_loss)
        history["val_unweighted_loss"].append(val_unweighted_loss)
        history["val_acc"].append(val_acc)
        for key, value in epoch_metrics.items():
            history[key].append(value)

        if predictions_path is not None:
            if not np.array_equal(np.asarray(val_y_true), np.asarray(val_expected_labels)):
                raise ValueError(
                    "Las etiquetas de validation evaluadas no coinciden con val_expected_labels; "
                    "el orden de val_image_ids no corresponde al del dataset evaluado."
                )
            _append_val_predictions(predictions_path, epoch + 1, val_image_ids, val_y_true, val_y_prob)

        print(
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss(w): {val_weighted_loss:.4f} | Val Loss(uw): {val_unweighted_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Checkpoint por pérdida: cualquier mínimo absoluto guarda el modelo.
        if checkpoint_metric == "val_loss" and val_weighted_loss < best_checkpoint_loss:
            best_checkpoint_loss = val_weighted_loss
            torch.save(model.state_dict(), save_path)
            _save_checkpoint_meta(
                save_path=save_path,
                epoch=epoch + 1,
                val_weighted_loss=val_weighted_loss,
                val_unweighted_loss=val_unweighted_loss,
                val_acc=val_acc,
                extra=(
                    {"criterion": "min val_weighted_loss", "val_roc_auc": epoch_metrics["val_roc_auc"]}
                    if auc_save_path is not None else None
                ),
            )
            print("  Modelo guardado")

        # Checkpoint por AUC: oficial (checkpoint_metric="val_roc_auc") o diagnóstico (auc_save_path).
        # No interviene en el early stopping.
        auc_target_path = save_path if checkpoint_metric == "val_roc_auc" else auc_save_path
        if auc_target_path is not None:
            val_auc = epoch_metrics["val_roc_auc"]
            if val_auc is not None and val_auc > best_checkpoint_auc:
                best_checkpoint_auc = val_auc
                torch.save(model.state_dict(), auc_target_path)
                _save_checkpoint_meta(
                    save_path=auc_target_path,
                    epoch=epoch + 1,
                    val_weighted_loss=val_weighted_loss,
                    val_unweighted_loss=val_unweighted_loss,
                    val_acc=val_acc,
                    extra={"criterion": "max val_roc_auc", "val_roc_auc": val_auc},
                )
                print("  Modelo guardado (best_auc)")

        # Early stopping: patience solo se resetea cuando la mejora supera min_delta.
        if val_weighted_loss < (early_stop_reference - min_delta):
            early_stop_reference = val_weighted_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping en época {epoch + 1} (patience={patience})")
                break

    return history
