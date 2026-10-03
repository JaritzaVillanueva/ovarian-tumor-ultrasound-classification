import random

import numpy as np
import torch
from torch.utils.data import DataLoader


def _worker_init_fn(worker_id):
    """
    Inicializa el generador de números aleatorios para cada worker.
    """
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def create_dataloader(dataset, batch_size=32, shuffle=False, num_workers=4, seed=42):
    """
    Crea un DataLoader de PyTorch.
    """
    generator = torch.Generator().manual_seed(seed) if shuffle else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
        generator=generator,
    )

    return loader
