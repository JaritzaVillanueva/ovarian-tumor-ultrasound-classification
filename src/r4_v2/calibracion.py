"""Función de calibración del umbral y comprobación de transferencia (protocolo §10.1; solo especificada e implementada en G2).

G2 NO invoca estas funciones sobre las predicciones OOF reales: la sensibilidad objetivo `S*` (propuesta 0.90 de CLAUDE.md §5) y el
techo de 75 corridas NO están congelados (en la configuración son `null`). G4 ejecuta la calibración y congela el umbral para R8.

Convención: una imagen es positiva (lesión maligna) si `p >= umbral`. `t̂` es el MAYOR umbral cuya sensibilidad no baja de
`S*`: la k-ésima mayor puntuación positiva con `k = ceil(S* · n⁺)` en aritmética entera (S* como fracción exacta, nunca float).
Los empates NO se rompen: todas las imágenes con `p = t̂` son positivas, por lo que la sensibilidad realizada puede superar `S*`.
"""

from __future__ import annotations

import csv
import math
from fractions import Fraction
from numbers import Integral

from .common import GuardError, escribir_csv_nuevo


def como_fraccion(s_star) -> Fraction:
    """S* como fracción exacta (`Fraction`, `str` como "0.90" o entero); un `float` se rechaza a propósito."""
    if isinstance(s_star, bool) or isinstance(s_star, float):
        raise GuardError("S* debe expresarse como fracción exacta (Fraction o str), no como float")
    try:
        s = Fraction(s_star)
    except (ValueError, TypeError, ZeroDivisionError) as exc:
        raise GuardError(f"S* no es una fracción válida: {s_star!r}") from exc
    if not (0 < s <= 1):
        raise GuardError("S* debe cumplir 0 < S* <= 1")
    return s


def k_objetivo(s_star, n_pos: int) -> int:
    """k = ceil(S*·n⁺) con aritmética entera: k = -(-num·n⁺ // den); 1 <= k <= n⁺."""
    s = como_fraccion(s_star)
    if not isinstance(n_pos, Integral) or n_pos < 1:
        raise GuardError("n⁺ debe ser un entero >= 1")
    k = -(-s.numerator * int(n_pos) // s.denominator)
    if not 1 <= k <= n_pos:
        raise GuardError(f"k={k} fuera de [1, {n_pos}]")
    return k


def _validar(y, scores):
    if len(y) != len(scores):
        raise GuardError("y y las puntuaciones deben tener el mismo tamaño")
    ys, ps = [], []
    for a, p in zip(y, scores):
        if isinstance(a, bool) or int(a) != a or int(a) not in (0, 1):
            raise GuardError("las etiquetas deben ser 0 o 1")
        p = float(p)
        if not math.isfinite(p):
            raise GuardError("las puntuaciones deben ser finitas")
        ys.append(int(a))
        ps.append(p)
    return ys, ps


def evaluar_en_umbral(y, scores, umbral: float) -> dict:
    """Matriz de confusión con la MISMA regla `p >= umbral`, con las puntuaciones tal cual (sin redondear)."""
    ys, ps = _validar(y, scores)
    tp = sum(1 for a, p in zip(ys, ps) if a == 1 and p >= umbral)
    fn = sum(1 for a, p in zip(ys, ps) if a == 1 and p < umbral)
    fp = sum(1 for a, p in zip(ys, ps) if a == 0 and p >= umbral)
    tn = sum(1 for a, p in zip(ys, ps) if a == 0 and p < umbral)
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "sensibilidad": Fraction(tp, tp + fn) if tp + fn else None,
            "especificidad": Fraction(tn, tn + fp) if tn + fp else None}


def calibrar_umbral(y, scores, s_star) -> dict:
    """Procedimiento de §10.1 (pasos 1-6) sobre una fuente con etiquetas `y` y puntuaciones `scores`."""
    s = como_fraccion(s_star)                                              # paso 1
    ys, ps = _validar(y, scores)
    positivas = sorted((p for a, p in zip(ys, ps) if a == 1), reverse=True)
    n_pos = len(positivas)
    if n_pos < 1:
        raise GuardError("la fuente no tiene positivas (n⁺ = 0)")
    k = k_objetivo(s, n_pos)                                               # paso 2
    t = positivas[k - 1]                                                   # pasos 3-4: t̂ = s_(k)
    ev = evaluar_en_umbral(ys, ps, t)                                      # paso 5
    return {"umbral": t, "k": k, "n_pos": n_pos, "m": ev["tp"], "s_star": str(s),
            "sensibilidad_real": ev["sensibilidad"], "tp": ev["tp"], "fn": ev["fn"], "fp": ev["fp"], "tn": ev["tn"],
            "especificidad": ev["especificidad"],                            # paso 6
            "n_pos_empatadas": sum(1 for p in positivas if p == t),
            "n_neg_empatadas": sum(1 for a, p in zip(ys, ps) if a == 0 and p == t)}


def comprobar_transferencia(y, scores, folds, s_star) -> dict:
    """`t̂` calculado con los folds distintos de k y aplicado al fold k (un fold fuera).

    Una sensibilidad >= S* sobre las mismas predicciones que fijan `t̂` se cumple por construcción; esta comprobación no.
    """
    ys, ps = _validar(y, scores)
    if len(folds) != len(ys):
        raise GuardError("folds debe tener un elemento por imagen")
    por_fold, tp_total, pos_total = {}, 0, 0
    for k in sorted(set(folds)):
        fuera = [i for i, f in enumerate(folds) if f != k]
        dentro = [i for i, f in enumerate(folds) if f == k]
        cal = calibrar_umbral([ys[i] for i in fuera], [ps[i] for i in fuera], s_star)
        ev = evaluar_en_umbral([ys[i] for i in dentro], [ps[i] for i in dentro], cal["umbral"])
        por_fold[k] = {"umbral": cal["umbral"], **ev}
        tp_total += ev["tp"]
        pos_total += ev["tp"] + ev["fn"]
    return {"s_star": str(como_fraccion(s_star)), "por_fold": por_fold, "tp": tp_total, "n_pos": pos_total,
            "sensibilidad_agrupada": Fraction(tp_total, pos_total) if pos_total else None}


# ---------------------------------------------------------------- almacenamiento sin pérdida de las puntuaciones
def guardar_puntuaciones(path, ids, y, scores) -> None:
    """CSV con `repr(float)`: ida y vuelta exacta (un float32 convertido a float64 se conserva sin pérdida)."""
    filas = [{"image_id": i, "y_true": int(a), "probability_malignant": repr(float(p))} for i, a, p in zip(ids, y, scores)]
    escribir_csv_nuevo(path, filas, ["image_id", "y_true", "probability_malignant"])


def cargar_puntuaciones(path):
    with open(path, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    return ([r["image_id"] for r in filas], [int(r["y_true"]) for r in filas],
            [float(r["probability_malignant"]) for r in filas])
