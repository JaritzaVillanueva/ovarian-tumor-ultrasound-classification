"""Dataset de G2 para clasificación binaria de lesiones (mapeo v2).

A diferencia del `dataset.py` histórico (que lee `binary_label` y el manifiesto mixto), este lee `label_v2` de la partición
`phash_le8_v1` y resuelve cada ruta con la guarda K4 (`resolver_imagen`: solo `train/images` o `validation/images` de desarrollo;
sin componente `test`). Importa torch y Pillow al cargarse: solo se importa de forma diferida desde el ejecutor [ND localmente].
Está a nivel de módulo (no anidado) para que sea serializable con DataLoader y `num_workers > 0` en cualquier plataforma.
"""

from __future__ import annotations

from PIL import Image
from torch.utils.data import Dataset

from .particion import resolver_imagen


class DatasetLesiones(Dataset):
    def __init__(self, root, filas, transform=None):
        self.items = [(resolver_imagen(root, f), int(f["label_v2"])) for f in filas]   # K4 sobre todas las rutas, sin abrirlas
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ruta, etiqueta = self.items[i]
        imagen = Image.open(ruta).convert("RGB")
        return (self.transform(imagen) if self.transform else imagen), etiqueta
