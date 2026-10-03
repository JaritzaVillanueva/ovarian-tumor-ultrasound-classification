from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast


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


def validate_one_epoch(model, dataloader, criterion, device):
    """
    Ejecuta una época de validación.
    """
    model.eval()
    unweighted_criterion = nn.CrossEntropyLoss()

    running_weighted_loss = 0.0
    running_unweighted_loss = 0.0
    correct = 0
    total = 0

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

    return ( running_weighted_loss / total, running_unweighted_loss / total, correct / total, )


def _save_checkpoint_meta(save_path, epoch, val_weighted_loss, val_unweighted_loss, val_acc):
    """
    Guarda un archivo JSON con metadatos del checkpoint.
    """
    meta_path = os.path.splitext(save_path)[0] + "_meta.json"
    meta = {
        "epoch": epoch,
        "val_weighted_loss": val_weighted_loss,
        "val_unweighted_loss": val_unweighted_loss,
        "val_acc": val_acc,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


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
):
    """
    Entrena un modelo de clasificación.
    """
    set_seed(seed)

    _use_amp = use_amp and torch.cuda.is_available()
    scaler = GradScaler() if _use_amp else None

    # best_checkpoint_loss: mínimo absoluto de val_weighted_loss
    # early_stop_reference: solo se actualiza cuando la mejora supera min_delta
    best_checkpoint_loss = float("inf")
    early_stop_reference = float("inf")
    patience_counter = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_unweighted_loss": [],
        "val_acc": [],
    }

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler=scaler
        )
        val_weighted_loss, val_unweighted_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_weighted_loss)
        history["val_unweighted_loss"].append(val_unweighted_loss)
        history["val_acc"].append(val_acc)

        print(
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss(w): {val_weighted_loss:.4f} | Val Loss(uw): {val_unweighted_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Checkpoint: cualquier mínimo absoluto guarda el modelo.
        if val_weighted_loss < best_checkpoint_loss:
            best_checkpoint_loss = val_weighted_loss
            torch.save(model.state_dict(), save_path)
            _save_checkpoint_meta(
                save_path=save_path,
                epoch=epoch + 1,
                val_weighted_loss=val_weighted_loss,
                val_unweighted_loss=val_unweighted_loss,
                val_acc=val_acc,
            )
            print("  Modelo guardado")

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
