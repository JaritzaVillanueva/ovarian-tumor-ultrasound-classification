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


class SmoothedWeightedCrossEntropy(nn.Module):
    """
    Entropía cruzada con label smoothing uniforme y pesos de clase como peso por muestra (enmienda E1 del
    protocolo del agente, a06).

    Objetivo suavizado q: (1 - e) + e/C sobre la clase verdadera y e/C en las demás (con C = 2 y e = 0.1:
    0.95 y 0.05). Cada muestra pesa w[y_n] según su clase verdadera y la reducción es la media ponderada:

        L = sum_n w[y_n] * ( -sum_c q_c * log p_c ) / sum_n w[y_n]

    Con e = 0 es la entropía cruzada ponderada de siempre (nn.CrossEntropyLoss con weight). A diferencia de
    nn.CrossEntropyLoss(weight=w, label_smoothing=e), los pesos NO multiplican el término suavizado por clase
    (eso desplaza el óptimo de una benigna a P(maligna) ≈ 0.30), de modo que el óptimo de cada muestra es q.
    """

    def __init__(self, weight, label_smoothing=0.1):
        super().__init__()
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing debe estar en [0, 1); se recibió {label_smoothing}.")
        self.label_smoothing = float(label_smoothing)
        weight = torch.as_tensor(weight)
        self.register_buffer("weight", weight if weight.dtype.is_floating_point else weight.float())

    def forward(self, logits, target):
        # float32 como mínimo (con AMP los logits pueden ser float16); float64 se conserva
        log_p = F.log_softmax(logits.to(torch.promote_types(logits.dtype, torch.float32)), dim=1)
        n_classes = log_p.shape[1]

        smoothed_target = torch.full_like(log_p, self.label_smoothing / n_classes)
        smoothed_target.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing + self.label_smoothing / n_classes)

        sample_weight = self.weight.to(log_p.dtype)[target]
        sample_loss = -(smoothed_target * log_p).sum(dim=1)

        return (sample_weight * sample_loss).sum() / sample_weight.sum()


def build_criterion(loss_type, class_weights, focal_gamma=2.0, label_smoothing=0.0):
    """
    Devuelve la función de pérdida del entrenamiento: entropía cruzada ponderada (por defecto,
    protocolo R4-R6) o Focal Loss ponderada (R7-H3). Ambas reciben los mismos pesos de clase.

    label_smoothing (R7, brazo del agente, enmienda E1) solo se admite con cross_entropy y usa
    SmoothedWeightedCrossEntropy: suavizado uniforme con los pesos de clase como peso por muestra. Con 0.0
    (por defecto) la llamada es la de siempre: nn.CrossEntropyLoss(weight=...).
    """
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError(f"label_smoothing debe estar en [0, 1); se recibió {label_smoothing}.")
    if loss_type == "cross_entropy":
        if label_smoothing > 0.0:
            return SmoothedWeightedCrossEntropy(weight=class_weights, label_smoothing=label_smoothing)
        return nn.CrossEntropyLoss(weight=class_weights)
    if loss_type == "focal":
        if label_smoothing > 0.0:
            raise ValueError("label_smoothing no se combina con la Focal Loss.")
        return WeightedFocalLoss(weight=class_weights, gamma=focal_gamma)
    raise ValueError(f"loss_type no soportado: {loss_type}")


def build_validation_criterion(class_weights):
    """
    Pérdida ponderada de validation SIN suavizado: entropía cruzada con los pesos de clase de entrenamiento.
    Se usa cuando el entrenamiento aplica label smoothing, para que val_loss (y con él el early stopping) se
    calcule como en B0 y B2 (enmienda E1: la pérdida de entrenamiento es SmoothedWeightedCrossEntropy).
    """
    return nn.CrossEntropyLoss(weight=class_weights)
