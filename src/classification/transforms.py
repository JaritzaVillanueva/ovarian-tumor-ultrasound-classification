from torchvision import transforms
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

def get_classification_transforms():

    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        ResizeWithPadding(target_size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])