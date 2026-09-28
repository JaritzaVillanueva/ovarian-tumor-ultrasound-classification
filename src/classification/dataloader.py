from torch.utils.data import DataLoader


def create_dataloader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
):
    """
    Crea un DataLoader de PyTorch.

    Args:
        dataset:
            Dataset PyTorch.
        batch_size:
            Número de imágenes por lote.
        shuffle:
            Mezclar muestras antes de cada época.
        num_workers:
            Procesos paralelos de carga.

    Returns:
        DataLoader
    """

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return loader