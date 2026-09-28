# ResNet50 Baseline

## Configuración

Modelo:
ResNet50

Inicialización:
Pesos preentrenados ImageNet

Entrada:
224x224 RGB

Loss:
Weighted CrossEntropyLoss

Optimizer:
AdamW

Learning rate:
1e-4

Epochs:
30


## Resultado test

Accuracy:
0.90

Balanced Accuracy:
0.568

ROC-AUC:
0.753


## Clase maligna

Precision:
0.64

Recall:
0.15

F1-score:
0.24


## Observación

El modelo presenta una alta exactitud global, influenciada por la distribución mayoritaria de la clase benigna. Sin embargo, la detección de muestras malignas presenta limitaciones, evidenciadas por el bajo recall de dicha clase.