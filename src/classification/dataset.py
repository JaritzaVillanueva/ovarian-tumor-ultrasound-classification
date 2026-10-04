from pathlib import Path

import pandas as pd
from PIL import Image

from torch.utils.data import Dataset


class MMOTUClassificationDataset(Dataset):
    """
    Dataset PyTorch para clasificación binaria de imágenes ecográficas MMOTU.
    """
    def __init__(
        self,
        csv_file,
        image_dir,
        split,
        transform=None,
        condition="A",
    ):

        self.df = pd.read_csv(csv_file)
        self.df = self.df[ self.df["experimental_split"] == split ].reset_index(drop=True)

        self.image_dir = Path(image_dir)
        self.split = split
        self.transform = transform
        self.condition = condition

    def __len__(self):
        return len(self.df)

    # Devuelve una dupla (imagen, etiqueta) para el índice dado.
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row["image_name"]

        if self.condition in ("B", "C"):
            image_name = Path(image_name).stem + ".png"

        label = int( row["binary_label"] )

        image_path = ( self.image_dir / self.split / "images" / image_name )
        image = Image.open( image_path ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
