"""
Opciones de entrenamiento opt-in del brazo del agente de optimización (R7):
calendario de learning rate, decaimiento del learning rate por capa y promedio exponencial de pesos (EMA).

Ninguna se activa por defecto; con los valores por defecto de run_experiment.py no se ejecuta nada de este módulo.
Ninguna cambia el criterio de checkpoint.
"""
from __future__ import annotations

import copy
import math

import torch

LR_SCHEDULES = ("constant", "warmup_cosine")


def lr_multiplier_fn(peak_lr, steps_per_epoch, warmup_epochs, cosine_epochs, lr_min):
    """
    Devuelve una función paso -> multiplicador del learning rate pico (el paso es global, por iteración, desde 0).

    Calentamiento lineal de `warmup_epochs` épocas y después decaimiento coseno de `cosine_epochs` épocas hasta
    `lr_min`, que se mantiene hasta el final:
        paso < W:            lr = pico * (paso + 1) / W
        W <= paso < W + C:   lr = lr_min + (pico - lr_min) * 0.5 * (1 + cos(pi * (paso - W) / C))
        paso >= W + C:       lr = lr_min
    con W = warmup_epochs * steps_per_epoch y C = cosine_epochs * steps_per_epoch (redondeados a pasos enteros).
    El multiplicador es lr / pico; cada grupo de parámetros lo aplica a su propio learning rate base.
    """
    if peak_lr <= 0:
        raise ValueError("el learning rate pico debe ser positivo.")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs debe ser mayor o igual que 0.")
    if cosine_epochs <= 0:
        raise ValueError("cosine_epochs debe ser mayor que 0.")
    if not 0 <= lr_min < peak_lr:
        raise ValueError(f"lr_min debe estar en [0, learning rate pico); se recibió {lr_min} con pico {peak_lr}.")

    warmup_steps = int(round(warmup_epochs * steps_per_epoch))
    cosine_steps = int(round(cosine_epochs * steps_per_epoch))
    if cosine_steps <= 0:
        raise ValueError("cosine_epochs * pasos por época debe ser al menos 1.")
    floor = lr_min / peak_lr

    def multiplier(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / cosine_steps)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return multiplier


def build_layer_decay_param_groups(model, lr, weight_decay, layer_decay):
    """
    Grupos de parámetros con decaimiento del learning rate por capa para ConvNeXt de torchvision.

    Un grupo por módulo de `model.features` (8 en ConvNeXt-Tiny, índices 0 a 7) y uno para la cabeza
    (`model.classifier`: normalización y capa lineal). El módulo i recibe lr * layer_decay ** (n - i), con
    n = len(model.features); la cabeza recibe lr. El weight decay es el mismo en todos los grupos.
    Verifica que cada parámetro del modelo quede en exactamente un grupo.
    """
    if not 0.0 < layer_decay <= 1.0:
        raise ValueError(f"layer_decay debe estar en (0, 1]; se recibió {layer_decay}.")
    if not (hasattr(model, "features") and hasattr(model, "classifier")):
        raise ValueError("layer_decay solo está definido para modelos con 'features' y 'classifier' (ConvNeXt).")

    n_modules = len(model.features)
    groups = []
    for index, module in enumerate(model.features):
        scale = layer_decay ** (n_modules - index)
        groups.append({"params": list(module.parameters()), "lr": lr * scale, "weight_decay": weight_decay,
                       "lr_scale": scale, "name": f"features.{index}"})
    groups.append({"params": list(model.classifier.parameters()), "lr": lr, "weight_decay": weight_decay,
                   "lr_scale": 1.0, "name": "classifier"})

    assigned = [id(p) for g in groups for p in g["params"]]
    if len(assigned) != len(set(assigned)):
        raise ValueError("un parámetro está en más de un grupo.")
    missing = [name for name, p in model.named_parameters() if id(p) not in set(assigned)]
    if missing:
        raise ValueError(f"parámetros sin grupo de learning rate: {missing[:5]}")
    return groups


class ModelEMA:
    """
    Promedio exponencial de pesos: ema = decay * ema + (1 - decay) * pesos, una vez por iteración de entrenamiento.

    La EMA se inicializa con los pesos de partida del modelo (sin corrección de sesgo ni calentamiento). Contiene
    una copia del modelo (`module`, en modo evaluación y sin gradientes) que es la que se usa en validation y la
    que se guarda como checkpoint. Los tensores que no son de coma flotante se copian.
    """

    def __init__(self, model, decay):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"ema decay debe estar en (0, 1); se recibió {decay}.")
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        model_state = model.state_dict()
        for key, ema_value in self.module.state_dict().items():
            value = model_state[key].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(value)
