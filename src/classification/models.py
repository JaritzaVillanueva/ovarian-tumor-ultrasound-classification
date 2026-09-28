import torch.nn as nn
from torchvision import models


def create_resnet50_classifier(
    pretrained=True,
    num_classes=2,
    freeze_backbone=False
):
    """
    Crea un clasificador basado en ResNet50.

    Args:
        pretrained:
            Utiliza pesos preentrenados.
        num_classes:
            Número de clases de salida.
        freeze_backbone:
            Congela las capas convolucionales.

    Returns:
        Modelo ResNet50 adaptado a clasificación binaria.
    """

    if pretrained:
        weights = models.ResNet50_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

def create_densenet121_classifier(
    pretrained=True,
    num_classes=2,
    freeze_backbone=False
):
    """
    Crea un clasificador DenseNet121
    adaptado para clasificación binaria.
    """

    model = models.densenet121(
        weights="DEFAULT" if pretrained else None
    )


    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False


    model.classifier = nn.Linear(
        model.classifier.in_features,
        num_classes
    )


    return model

def create_convnext_tiny_classifier(
    pretrained=True,
    num_classes=2,
    freeze_backbone=False
):
    """
    Crea un clasificador ConvNeXt-Tiny adaptado para clasificación binaria.
    """

    model = models.convnext_tiny(
        weights="DEFAULT" if pretrained else None
    )


    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False


    model.classifier[2] = nn.Linear(
        model.classifier[2].in_features,
        num_classes
    )


    return model