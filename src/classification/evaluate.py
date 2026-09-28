import torch


def predict(
    model,
    dataloader,
    device
):
    """
    Obtiene predicciones del modelo.

    Returns:
        etiquetas reales
        predicciones
        probabilidades
    """

    model.eval()

    y_true = []
    y_pred = []
    y_prob = []


    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)


            outputs = model(images)


            probabilities = torch.softmax(
                outputs,
                dim=1
            )


            predictions = torch.argmax(
                probabilities,
                dim=1
            )


            y_true.extend(
                labels.cpu().numpy()
            )

            y_pred.extend(
                predictions.cpu().numpy()
            )

            y_prob.extend(
                probabilities[:,1]
                .cpu()
                .numpy()
            )


    return (
        y_true,
        y_pred,
        y_prob
    )

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    roc_auc_score
)


def evaluate_predictions(
    y_true,
    y_pred,
    y_prob
):

    results = {}


    results["balanced_accuracy"] = (
        balanced_accuracy_score(
            y_true,
            y_pred
        )
    )


    results["roc_auc"] = (
        roc_auc_score(
            y_true,
            y_prob
        )
    )


    results["confusion_matrix"] = (
        confusion_matrix(
            y_true,
            y_pred
        )
    )


    results["classification_report"] = (
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "benign",
                "malignant"
            ]
        )
    )


    return results