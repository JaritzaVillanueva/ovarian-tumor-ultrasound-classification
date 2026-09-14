from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",}


def get_files_by_id(
    directory: Path,
    is_mask: bool = False,
    extensions=IMAGE_EXTENSIONS,
):
    """
    Indexa los archivos utilizando el ID de la muestra.

    Convención:
        Imagen:  1000.JPG
        Máscara: 1000_binary.PNG
    """

    directory = Path(directory)

    files = {}

    for path in directory.iterdir():

        if not path.is_file():
            continue

        if path.suffix.lower() not in extensions:
            continue

        file_id = path.stem

        if is_mask:
            suffix = "_binary"

            if not file_id.endswith(suffix):
                raise ValueError(f"Nombre inesperado de máscara: {path.name}")

            file_id = file_id[:-len(suffix)]

        if file_id in files:
            raise ValueError(f"ID duplicado '{file_id}' en {directory}")

        files[file_id] = path

    return files


class OvarianTumorSegmentationDataset(Dataset):
    """
    Dataset para segmentación binaria de tumores ováricos.

    Imagen:
        RGB -> [3, H, W]
        float32 en [0,1]

    Máscara:
        binaria -> [1, H, W]
        float32 en {0,1}
    """

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        image_size=(256, 256),
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size

        self.images = get_files_by_id(self.images_dir, is_mask=False,)
        self.masks = get_files_by_id(self.masks_dir, is_mask=True,)

        image_ids = set(self.images)
        mask_ids = set(self.masks)

        missing_masks = image_ids - mask_ids
        missing_images = mask_ids - image_ids

        if missing_masks:
            raise ValueError(f"{len(missing_masks)} imágenes no tienen máscara.")
        if missing_images:
            raise ValueError(f"{len(missing_images)} máscaras no tienen imagen.")

        self.ids = sorted(image_ids, key=int,)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):

        image_id = self.ids[index]
        image_path = self.images[image_id]
        mask_path = self.masks[image_id]

        # Imagen

        with Image.open(image_path) as image:

            image = image.convert("RGB")
            image = image.resize(self.image_size, resample=Image.Resampling.BILINEAR,)
            image = np.asarray(image, dtype=np.float32,)

        image = image / 255.0
        image = np.transpose(image, (2, 0, 1),)
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image)

        # Máscara

        with Image.open(mask_path) as mask:

            mask = mask.convert("L")
            mask = mask.resize(self.image_size, resample=Image.Resampling.NEAREST,)
            mask = np.asarray(mask, dtype=np.float32,)

        mask = np.expand_dims(mask, axis=0,)

        mask = np.ascontiguousarray(mask)
        mask = torch.from_numpy(mask)

        return {
            "image": image,
            "mask": mask,
            "image_id": image_id,
        }