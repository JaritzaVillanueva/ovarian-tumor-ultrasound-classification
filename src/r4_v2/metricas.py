"""Métricas de discriminación con pesos enteros por imagen (numpy; sin sklearn en ejecución).

Un peso `w_i` equivale a repetir `w_i` veces la imagen (remuestreo del bootstrap), con empates de puntuación tratados como en
sklearn: AP escalonado (no interpolado) sobre umbrales distintos y ROC-AUC con empates a 0.5. Las pruebas comparan estas
funciones con `sklearn.metrics` y con una implementación ingenua independiente.

Convención de clase positiva: `p >= umbral` es lesión maligna (label_v2 = 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NIVELES_ESPECIFICIDAD = (0.90, 0.95)
UMBRAL_CLASIFICACION = 0.5          # continuidad con el informe; NO es el umbral calibrado de G4


@dataclass(frozen=True)
class Preparado:
    """Puntuaciones ordenadas de mayor a menor con sus grupos de empate (se calcula una vez por (arquitectura, semilla))."""
    orden: np.ndarray
    y: np.ndarray            # etiquetas en el orden descendente
    inicios: np.ndarray      # inicio de cada grupo de empate
    pred_05: np.ndarray      # (score >= 0.5) en el orden descendente


def preparar(y, score) -> Preparado:
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if y.shape != score.shape or y.ndim != 1:
        raise ValueError("y y score deben ser vectores del mismo tamaño")
    if not np.isfinite(score).all():
        raise ValueError("las puntuaciones deben ser finitas")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y debe ser binaria (0/1)")
    orden = np.argsort(-score, kind="stable")
    s = score[orden]
    inicios = np.flatnonzero(np.concatenate(([True], s[1:] != s[:-1])))
    return Preparado(orden=orden, y=y[orden], inicios=inicios, pred_05=s >= UMBRAL_CLASIFICACION)


def calcular(prep: Preparado, pesos=None) -> dict:
    """Métricas con pesos `pesos` (vector (n,) o matriz (B, n) de enteros, en el orden ORIGINAL de las imágenes).

    Devuelve arrays de forma (B,) (o escalares si `pesos` es un vector). Si un remuestreo no tiene una clase, las métricas
    que la requieren son NaN (se cuentan aparte en el bootstrap).
    """
    n = prep.y.size
    w = np.ones(n) if pesos is None else np.asarray(pesos, dtype=np.float64)
    escalar = w.ndim == 1
    w = np.atleast_2d(w)[:, prep.orden]
    pos_w = w * prep.y
    neg_w = w * (1 - prep.y)
    P = np.add.reduceat(pos_w, prep.inicios, axis=1)
    N = np.add.reduceat(neg_w, prep.inicios, axis=1)
    tp, fp = np.cumsum(P, axis=1), np.cumsum(N, axis=1)
    tot_p, tot_n = tp[:, -1:], fp[:, -1:]

    with np.errstate(divide="ignore", invalid="ignore"):
        den = tp + fp
        precision = np.where(den > 0, tp / np.where(den > 0, den, 1.0), 0.0)
        recall = tp / tot_p
        previo = np.concatenate([np.zeros((w.shape[0], 1)), recall[:, :-1]], axis=1)
        ap = np.sum((recall - previo) * precision, axis=1)
        # ROC-AUC: pares (positiva, negativa) con la positiva puntuada más alto + 0.5 de los empates
        auc = np.sum(P * ((tot_n - fp) + 0.5 * N), axis=1) / (tot_p[:, 0] * tot_n[:, 0])
        tpr, fpr = recall, fp / tot_n
        sens_esp = {}
        for esp in NIVELES_ESPECIFICIDAD:
            validos = fpr <= (1.0 - esp) + 1e-12
            sens_esp[f"sensibilidad_esp_{esp:.2f}"] = np.where(tot_p[:, 0] > 0, np.max(np.where(validos, tpr, 0.0), axis=1), np.nan)
    ap = np.where(tot_p[:, 0] > 0, ap, np.nan)
    auc = np.where((tot_p[:, 0] > 0) & (tot_n[:, 0] > 0), auc, np.nan)

    pred = prep.pred_05
    tp05 = pos_w[:, pred].sum(axis=1)
    fp05 = neg_w[:, pred].sum(axis=1)
    fn05 = pos_w[:, ~pred].sum(axis=1)
    tn05 = neg_w[:, ~pred].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sens = np.where(tot_p[:, 0] > 0, tp05 / tot_p[:, 0], np.nan)
        espec = np.where(tot_n[:, 0] > 0, tn05 / tot_n[:, 0], np.nan)
    out = {"ap": ap, "roc_auc": auc, "sensibilidad_05": sens, "especificidad_05": espec, "tp_05": tp05, "fp_05": fp05,
           "fn_05": fn05, "tn_05": tn05, **sens_esp}
    return {k: (v[0] if escalar else v) for k, v in out.items()}


def metricas(y, score) -> dict:
    """Métricas sin pesos de un único vector de predicciones (floats de Python)."""
    return {k: float(v) for k, v in calcular(preparar(y, score)).items()}


def ap(y, score) -> float:
    return metricas(y, score)["ap"]


def roc_auc(y, score) -> float:
    return metricas(y, score)["roc_auc"]


def ap_ingenuo(y, score) -> float:
    """Referencia independiente y simple (estilo `robustez_e2_analysis.py`): AP escalonado por umbrales distintos."""
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    n_pos = int(y.sum())
    previo_recall, ap_total = 0.0, 0.0
    for umbral in sorted(set(score.tolist()), reverse=True):
        sel = score >= umbral
        tp = int((y[sel] == 1).sum())
        precision = tp / int(sel.sum())
        recall = tp / n_pos
        ap_total += (recall - previo_recall) * precision
        previo_recall = recall
    return ap_total


def curva_pr(y, score):
    """(recall, precision) en cada umbral distinto, de mayor a menor (curva escalonada de F4; no interpolada)."""
    prep = preparar(y, score)
    pos = np.add.reduceat(prep.y.astype(float), prep.inicios)
    neg = np.add.reduceat((1 - prep.y).astype(float), prep.inicios)
    tp, fp = np.cumsum(pos), np.cumsum(neg)
    return tp / tp[-1], tp / (tp + fp)
