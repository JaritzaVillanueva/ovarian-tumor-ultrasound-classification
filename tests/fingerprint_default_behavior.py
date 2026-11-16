"""
Huella del comportamiento por defecto del pipeline de clasificación (no es una prueba de pytest).

Solo usa la API que existía antes de los argumentos del brazo del agente (R7, Fase 4), de modo que el mismo archivo
se ejecuta contra el código anterior y contra el nuevo, y las dos salidas deben ser idénticas byte a byte:

    python tests/fingerprint_default_behavior.py > huella.json

Cubre, con semillas fijas y en CPU: las transformaciones de train y validation, el modelo ConvNeXt-Tiny (pesos
iniciales, salida en modo evaluación y probabilidades de stochastic depth), las pérdidas, el optimizador y un
entrenamiento diminuto completo con train_model (historial y checkpoint).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import tempfile

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.classification.losses import build_criterion
from src.classification.models import create_convnext_tiny_classifier
from src.classification.train import set_seed, train_model
from src.classification.transforms import get_train_transforms, get_val_transforms


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_digest(state_dict):
    h = hashlib.sha256()
    for key, value in state_dict.items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class _Tiny(Dataset):
    def __init__(self, n, seed):
        generator = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 4, generator=generator)
        self.y = ((self.x[:, 0] + 0.8 * torch.randn(n, generator=generator)) > 0).long()
        self.y[0], self.y[1] = 0, 1

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], int(self.y[idx])


def transforms_fingerprint():
    array = np.random.RandomState(0).randint(0, 256, (200, 300, 3), dtype=np.uint8)
    image = Image.fromarray(array, mode="RGB")
    out = {}
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        random.seed(seed)
        out[f"train_seed{seed}"] = digest(get_train_transforms()(image))
    out["val"] = digest(get_val_transforms()(image))
    out["val_shape"] = list(get_val_transforms()(image).shape)
    return out


def model_fingerprint():
    set_seed(0)
    model = create_convnext_tiny_classifier(pretrained=False, num_classes=2, freeze_backbone=False)
    probabilities = [m.p for m in model.modules() if type(m).__name__ == "StochasticDepth"]
    x = torch.randn(1, 3, 64, 64, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        output = model.eval()(x)
    return {
        "state_dict": state_digest(model.state_dict()),
        "eval_output": [repr(v) for v in output.flatten().tolist()],
        "stochastic_depth_p": [repr(p) for p in probabilities],
        "n_parameters": sum(p.numel() for p in model.parameters()),
    }


def criterion_fingerprint():
    weight = torch.tensor([0.5610098242759705, 4.597701072692871])
    generator = torch.Generator().manual_seed(1)
    logits = torch.randn(32, 2, generator=generator)
    target = torch.randint(0, 2, (32,), generator=generator)
    return {
        "cross_entropy": repr(build_criterion("cross_entropy", weight, 2.0)(logits, target).item()),
        "focal": repr(build_criterion("focal", weight, 2.0)(logits, target).item()),
    }


def optimizer_fingerprint():
    model = create_convnext_tiny_classifier(pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    return {
        "n_groups": len(optimizer.param_groups),
        "lr": [g["lr"] for g in optimizer.param_groups],
        "weight_decay": [g["weight_decay"] for g in optimizer.param_groups],
        "extra_group_keys": sorted(set(optimizer.param_groups[0]) - {"params", "lr", "betas", "eps", "weight_decay", "amsgrad",
                                                                       "foreach", "maximize", "capturable", "differentiable", "fused"}),
        "n_tensors": len(optimizer.param_groups[0]["params"]),
    }


def training_fingerprint():
    out = {}
    for metric in ("val_loss", "val_roc_auc"):
        with tempfile.TemporaryDirectory() as directory:
            set_seed(99)
            model = nn.Linear(4, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
            criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.6, 1.4]))
            train_loader = DataLoader(_Tiny(64, 1), batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(7))
            val_loader = DataLoader(_Tiny(40, 2), batch_size=16, shuffle=False)
            save_path = os.path.join(directory, "best.pth")
            history = train_model(
                model, train_loader, val_loader, criterion, optimizer, device=torch.device("cpu"), epochs=8,
                save_path=save_path, patience=3, min_delta=0.001, seed=42, use_amp=False, checkpoint_metric=metric,
            )
            with open(os.path.splitext(save_path)[0] + "_meta.json") as f:
                meta = json.load(f)
            out[metric] = {
                "history_keys": list(history.keys()),
                "history": {k: [repr(v) for v in values] for k, values in history.items()},
                "meta": meta,
                "checkpoint": state_digest(torch.load(save_path, map_location="cpu", weights_only=True)),
                "final_model": state_digest(model.state_dict()),
                "final_lr": [g["lr"] for g in optimizer.param_groups],
                "param_group_keys": sorted(optimizer.param_groups[0].keys()),
            }
    return out


if __name__ == "__main__":
    fingerprint = {
        "torch": torch.__version__,
        "transforms": transforms_fingerprint(),
        "model": model_fingerprint(),
        "criterion": criterion_fingerprint(),
        "optimizer": optimizer_fingerprint(),
        "training": training_fingerprint(),
    }
    print(json.dumps(fingerprint, indent=1, sort_keys=True))
