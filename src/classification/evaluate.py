import numpy as np
import torch

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def predict(model, dataloader, device):
    """
    Obtiene predicciones del modelo.
    """
    model.eval()

    y_true = []
    y_prob = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_prob.extend(probabilities[:, 1].cpu().numpy())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int) # umbral de 0.5 para clasificar

    return y_true, y_pred, y_prob


def evaluate_predictions(y_true, y_pred, y_prob):
    """
    Evalúa las predicciones del modelo y devuelve métricas de rendimiento.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    results = {}
    results["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    results["roc_auc"] = roc_auc_score(y_true, y_prob)
    results["average_precision"] = average_precision_score(y_true, y_prob)

    results["precision_malignant"] = precision_score( y_true, y_pred, pos_label=1, zero_division=0 )
    results["recall_malignant"] = recall_score( y_true, y_pred, pos_label=1, zero_division=0 )
    results["f1_malignant"] = f1_score( y_true, y_pred, pos_label=1, zero_division=0 )
    results["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    results["accuracy"] = float((tp + tn) / (tp + tn + fp + fn))
    results["confusion_matrix"] = cm
    results["classification_report"] = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=["benign", "malignant"],
        zero_division=0,
    )

    return results
