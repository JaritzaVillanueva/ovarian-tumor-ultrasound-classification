import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFocalLoss(nn.Module):
    """
    Focal Loss multiclase con pesos de clase: FL = -w_y * (1 - p_t)^gamma * log(p_t).

    p_t es la probabilidad que el modelo asigna a la clase verdadera y se obtiene de log_softmax,
    sin pasar por una entropía cruzada ya ponderada. La reducción es la de nn.CrossEntropyLoss con
    pesos y reduction="mean": suma de las pérdidas ponderadas dividida por la suma de los pesos de
    las muestras del batch. Con gamma = 0 coincide con esa entropía cruzada ponderada.
    """

    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma debe ser >= 0; se recibió {gamma}.")
        self.gamma = float(gamma)
        self.register_buffer("weight", None if weight is None else torch.as_tensor(weight, dtype=torch.float32))

    def forward(self, logits, target):
        log_pt = F.log_softmax(logits.float(), dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
        one_minus_pt = -torch.expm1(log_pt)  # 1 - p_t sin perder precisión cuando p_t ≈ 1

        sample_weight = torch.ones_like(log_pt) if self.weight is None else self.weight[target]
        loss = -sample_weight * one_minus_pt.pow(self.gamma) * log_pt

        return loss.sum() / sample_weight.sum()


def build_criterion(loss_type, class_weights, focal_gamma=2.0):
    """
    Devuelve la función de pérdida del entrenamiento: entropía cruzada ponderada (por defecto,
    protocolo R4-R6) o Focal Loss ponderada (R7-H3). Ambas reciben los mismos pesos de clase.
    """
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    if loss_type == "focal":
        return WeightedFocalLoss(weight=class_weights, gamma=focal_gamma)
    raise ValueError(f"loss_type no soportado: {loss_type}")
