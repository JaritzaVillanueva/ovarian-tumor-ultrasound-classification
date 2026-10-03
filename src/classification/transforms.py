from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image


class ResizeWithPadding:
    """
    Redimensiona una imagen manteniendo la relación de aspecto y agrega padding para obtener un tamaño cuadrado.
    """

    def __init__(self, target_size=224, fill=0):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img):
        width, height = img.size

        scale = min(self.target_size / width, self.target_size / height)

        new_width = int(width * scale)
        new_height = int(height * scale)

        img = img.resize((new_width, new_height), Image.BILINEAR)

        delta_width = self.target_size - new_width
        delta_height = self.target_size - new_height

        padding = (
            delta_width // 2,
            delta_height // 2,
            delta_width - (delta_width // 2),
            delta_height - (delta_height // 2)
        )

        img = transforms.functional.pad(img, padding, fill=self.fill)

        return img

# Función para obtener las transformaciones de entrenamiento
def get_train_transforms():
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1),
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        ),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ResizeWithPadding(target_size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Función para obtener las transformaciones de validación
def get_val_transforms():
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        ResizeWithPadding(target_size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# Alias para compatibilidad con notebooks existentes.
def get_classification_transforms():
    return get_val_transforms()
