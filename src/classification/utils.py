import torch
import numpy as np


def compute_class_weights(labels):
    """
    Calcula pesos inversamente proporcionales a la frecuencia de cada clase.
    """

    classes, counts = np.unique(
        labels,
        return_counts=True
    )

    total = counts.sum()

    weights = []

    for count in counts:
        weight = total / (
            len(classes) * count
        )

        weights.append(weight)


    return torch.tensor(
        weights,
        dtype=torch.float32
    )