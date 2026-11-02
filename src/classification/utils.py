import torch
import numpy as np


def compute_class_weights(labels, alpha=1.0):
    """
    Calcula pesos inversamente proporcionales a la frecuencia de cada clase.

    alpha es el exponente de la ponderación (R7): el peso de la clase c es proporcional a
    frecuencia_c ** (-alpha) y se normaliza para que el peso medio por muestra sea 1.
        alpha = 1.0 -> frecuencia inversa (comportamiento original, por defecto).
        alpha = 0.5 -> ponderación atenuada (raíz cuadrada de la frecuencia inversa).
        alpha = 0.0 -> sin ponderación (todos los pesos valen 1).
    Los pesos se devuelven en el orden de las etiquetas ordenadas (0 = benigna, 1 = maligna).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha debe estar entre 0 y 1; se recibió {alpha}.")

    classes, counts = np.unique(
        labels,
        return_counts=True
    )

    total = counts.sum()

    weights = []

    if alpha == 1.0:
        for count in counts:
            weight = total / (
                len(classes) * count
            )

            weights.append(weight)
    else:
        smoothed = counts.astype(float) ** (-alpha)
        weights = total * smoothed / (counts * smoothed).sum()


    return torch.tensor(
        weights,
        dtype=torch.float32
    )