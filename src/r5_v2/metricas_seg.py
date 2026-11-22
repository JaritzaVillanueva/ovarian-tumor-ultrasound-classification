"""Dice/IoU, selección de época, ROI (solo caja y cobertura; NO genera recortes) y resumen de métricas de segmentación.

Sin interpretación: `resumen_metricas_seg` calcula estadísticos descriptivos de un CSV por imagen, sin IC (el bootstrap agrupado
con los índices `b` de G2 es posterior y está pendiente).
"""

from __future__ import annotations

import csv
import math

import numpy as np

from .comun import GuardError
from .mascaras import areas_de_componentes, binarizar, leer_mascara_binaria

COLUMNAS_METRICAS = ["image_id", "fold", "original_class", "label_v2", "group_id", "dice", "iou", "area_gt_256", "area_pred_256",
                     "n_componentes_gt", "n_componentes_pred", "gt_vacia", "pred_vacia", "cobertura", "fallback_pred", "fallback_gt"]


# ---------------------------------------------------------------- Dice / IoU / época
def dice_iou(pred, gt, eps: float = 1e-7):
    """Dice e IoU de UNA imagen, como `dice_iou_scores` del notebook histórico: (2I+eps)/(P+T+eps) y (I+eps)/(U+eps).

    Con ambas máscaras vacías el resultado es 1.0 (convención heredada); con GT vacía y predicción no vacía es ~0.
    """
    p, g = binarizar(pred), binarizar(gt)
    if p.shape != g.shape:
        raise GuardError(f"formas distintas: {p.shape} y {g.shape}")
    inter, pa, ga = float(np.logical_and(p, g).sum()), float(p.sum()), float(g.sum())
    return (2.0 * inter + eps) / (pa + ga + eps), (inter + eps) / (pa + ga - inter + eps)


def epoca_maxima_dice(val_dice) -> int:
    """Época (1-based) del MÁXIMO ESTRICTO del Dice medio de validación interna: un empate conserva la más temprana."""
    mejor, epoca = -math.inf, None
    for i, v in enumerate(val_dice, start=1):
        if not math.isfinite(v):
            raise GuardError(f"Dice de validación no finito en la época {i}")
        if v > mejor:
            mejor, epoca = v, i
    if epoca is None:
        raise GuardError("historial de Dice vacío")
    return epoca


# ---------------------------------------------------------------- ROI (solo caja) y cobertura
def caja_roi(mascara, tam_original, roi_cfg):
    """Caja ROI con `src/segmentation/roi.py::extract_roi_from_mask` (sin editarlo; requiere cv2 y torch). NO guarda recortes.

    `mascara`: máscara {0,1} a cualquier resolución (la predicha 256x256 o la de referencia a resolución original); `tam_original` =
    (ancho, alto) de la imagen original. Devuelve ((izq, arr, der, abj), usó_fallback). Misma función para predicha y referencia.
    """
    from PIL import Image
    from src.segmentation import roi as R
    res = R.extract_roi_from_mask(Image.new("RGB", tuple(tam_original)), np.asarray(mascara).astype(np.uint8),
                                  margin_ratio=roi_cfg["margen"], keep_largest=roi_cfg["keep_largest"],
                                  fallback_to_full_image=roi_cfg["fallback_a_imagen_completa"],
                                  min_component_area_fraction=roi_cfg["min_component_area_fraction"])
    return tuple(int(v) for v in res.bbox), bool(res.used_fallback)


def cobertura(gt_original, caja):
    """|GT ∩ caja| / |GT| a resolución original; None si la GT está vacía. `caja` = (izq, arr, der, abj), der/abj exclusivos."""
    g = binarizar(gt_original)
    total = int(g.sum())
    if total == 0:
        return None
    izq, arr, der, abj = caja
    return float(g[arr:abj, izq:der].sum()) / total


def fila_metricas(fila, pred256, gt_original, gt256, roi_cfg, calcular_caja, eps: float) -> dict:
    """Fila de `metricas_seg_por_imagen.csv` (Dice/IoU/áreas/componentes a 256x256; cobertura a resolución original)."""
    d, i = dice_iou(pred256, gt256, eps)
    alto, ancho = np.asarray(gt_original).shape
    caja_p, fb_p = calcular_caja(pred256, (ancho, alto), roi_cfg)
    caja_g, fb_g = calcular_caja(gt_original, (ancho, alto), roi_cfg)
    cob = cobertura(gt_original, caja_p)
    ap, ag = int(binarizar(pred256).sum()), int(binarizar(gt256).sum())
    return {"image_id": fila["image_id"], "fold": fila["fold"], "original_class": fila["original_class"],
            "label_v2": fila["label_v2"], "group_id": fila["group_id"], "dice": repr(float(d)), "iou": repr(float(i)),
            "area_gt_256": ag, "area_pred_256": ap,
            "n_componentes_gt": len(areas_de_componentes(binarizar(gt256))) if ag else 0,
            "n_componentes_pred": len(areas_de_componentes(binarizar(pred256))) if ap else 0,
            "gt_vacia": ag == 0, "pred_vacia": ap == 0, "cobertura": "" if cob is None else repr(cob),
            "fallback_pred": fb_p, "fallback_gt": fb_g}


def metricas_de_fold(filas_prediccion, rutas_mascaras, rutas_pred, tam, roi_cfg, calcular_caja, eps: float) -> list:
    """Métricas por imagen del fold k. Se leen las máscaras de referencia DESPUÉS de haber guardado las predicciones."""
    out = []
    for fila in filas_prediccion:
        gt_orig = leer_mascara_binaria(rutas_mascaras[fila["image_id"]])
        gt256 = leer_mascara_binaria(rutas_mascaras[fila["image_id"]], tam)
        pred = leer_mascara_binaria(rutas_pred[fila["image_id"]])
        if pred.shape != tuple(reversed(tam)):
            raise GuardError(f"{fila['image_id']}: la máscara predicha no es {tam}")
        out.append(fila_metricas(fila, pred, gt_orig, gt256, roi_cfg, calcular_caja, eps))
    return out


# ---------------------------------------------------------------- resumen (sin IC, sin interpretación)
def leer_metricas_csv(path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _flt(x):
    return None if x in ("", None) else float(x)


def _bool(x):
    return x is True or x == "True"


def _estad(valores):
    a = np.asarray(valores, dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "media": None, "mediana": None}
    return {"n": int(a.size), "media": float(a.mean()), "mediana": float(np.median(a))}


def _cobertura(valores, umbral):
    a = np.asarray(valores, dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "media": None, "minimo": None, "n_menor_umbral": 0}
    return {"n": int(a.size), "media": float(a.mean()), "minimo": float(a.min()), "n_menor_umbral": int((a < umbral).sum())}


def resumen_metricas_seg(filas, umbral_dice: float = 0.1, umbral_cobertura: float = 0.5) -> dict:
    """Descriptivos de `metricas_seg_por_imagen.csv`: Dice/IoU (media, mediana, por fold, n con Dice < umbral), cobertura (media,
    mínimo, n < umbral) y tasa de fallback. Las filas con fallback (cobertura 1 por construcción) se informan APARTE.
    """
    if not filas:
        raise GuardError("no hay filas de métricas")
    dice, iou = [_flt(f["dice"]) for f in filas], [_flt(f["iou"]) for f in filas]
    fb_p, fb_g = [_bool(f["fallback_pred"]) for f in filas], [_bool(f["fallback_gt"]) for f in filas]
    cob = [_flt(f["cobertura"]) for f in filas]
    n = len(filas)
    por_fold = {}
    for k in sorted({int(f["fold"]) for f in filas}):
        idx = [i for i, f in enumerate(filas) if int(f["fold"]) == k]
        por_fold[str(k)] = {"dice": _estad([dice[i] for i in idx]), "iou": _estad([iou[i] for i in idx]),
                            "n_dice_menor_umbral": sum(1 for i in idx if dice[i] < umbral_dice),
                            "n_fallback_pred": sum(1 for i in idx if fb_p[i]),
                            "n_cobertura_menor_umbral_sin_fallback": sum(1 for i in idx if not fb_p[i] and cob[i] is not None and cob[i] < umbral_cobertura)}
    sin_fb = [c for c, fb in zip(cob, fb_p) if not fb and c is not None]
    return {
        "n": n, "umbral_dice": umbral_dice, "umbral_cobertura": umbral_cobertura,
        "dice": _estad(dice), "iou": _estad(iou), "n_dice_menor_umbral": sum(1 for d in dice if d < umbral_dice),
        "por_fold": por_fold,
        "cobertura_sin_fallback_pred": _cobertura(sin_fb, umbral_cobertura),
        "cobertura_todas": _cobertura([c for c in cob if c is not None], umbral_cobertura),
        "n_cobertura_indefinida_gt_vacia": sum(1 for c in cob if c is None),
        "fallback_pred": {"n": sum(fb_p), "tasa": sum(fb_p) / n}, "fallback_gt": {"n": sum(fb_g), "tasa": sum(fb_g) / n},
        "n_pred_vacia": sum(1 for f in filas if _bool(f["pred_vacia"])), "n_gt_vacia": sum(1 for f in filas if _bool(f["gt_vacia"])),
        "sin_intervalos_de_confianza": True,
    }
