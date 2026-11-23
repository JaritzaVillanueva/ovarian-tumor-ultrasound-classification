"""R8 (mapeo v2): procedimiento estadístico aprobado, como funciones puras sin acceso a datos (numpy/scipy; sin torch).

* Evaluación primaria única del sistema congelado, con umbral FIJO `t_hat` y regla `p >= t_hat` (el umbral nunca se calcula ni se ajusta aquí).
* AP y ROC-AUC puntuales con las funciones de G2 (`r4_v2.metricas`: empates como sklearn) e IC 95 % por **bootstrap percentil estratificado por imagen**: 10 000 remuestreos,
  semilla 20261123; en cada remuestreo se extraen con reemplazo tantas imágenes positivas y negativas como las observadas (imágenes ordenadas por `image_id` dentro de cada clase;
  primero la matriz de posiciones positivas (B, P) y después la de negativas (B, N), con `numpy.random.default_rng(semilla).integers`). Supone independencia entre imágenes: el test no tiene
  `group_id` ni identificadores de paciente, por lo que el IC puede ser optimista (limitación declarada).
* Sensibilidad y especificidad en `t_hat` con intervalos binomiales exactos de Clopper–Pearson (95 %), condicionales al número de positivas y de negativas.
* Matriz de confusión, FP, FN y VPP (con la prevalencia del test; no transferible a otras poblaciones). Umbral 0.5: solo descripción secundaria, sin IC ni decisión.
No hay análisis por número de modelos ni comparación B2 − B0.
"""

from __future__ import annotations

import hashlib

import numpy as np

from src.r4_v2 import metricas as M

SEMILLA_BOOTSTRAP = 20261123
N_BOOTSTRAP = 10000
NIVEL_IC = 0.95
UMBRAL_SECUNDARIO = 0.5


def clopper_pearson(k: int, n: int, nivel: float = NIVEL_IC) -> tuple:
    """Intervalo binomial exacto (Clopper–Pearson) bilateral para k éxitos en n ensayos."""
    from scipy.stats import beta
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("Clopper–Pearson exige 0 <= k <= n y n > 0")
    a = (1.0 - nivel) / 2.0
    bajo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    alto = 1.0 if k == n else float(beta.ppf(1.0 - a, k + 1, n - k))
    return bajo, alto


def _validar(y, score):
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if y.ndim != 1 or y.shape != score.shape or y.size == 0:
        raise ValueError("y y score deben ser vectores no vacíos del mismo tamaño")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError("y debe ser 0/1 y las puntuaciones finitas en [0, 1]")
    if y.sum() == 0 or y.sum() == y.size:
        raise ValueError("se necesitan imágenes de ambas clases")
    return y, score


def bootstrap_estratificado(y, score, B: int = N_BOOTSTRAP, semilla: int = SEMILLA_BOOTSTRAP) -> dict:
    """AP y ROC-AUC en B remuestreos estratificados por clase. `y` y `score` deben venir ordenados por `image_id` (el llamador lo garantiza)."""
    y, score = _validar(y, score)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(semilla)
    sel_pos = pos[rng.integers(0, pos.size, size=(B, pos.size))]
    sel_neg = neg[rng.integers(0, neg.size, size=(B, neg.size))]
    conteos = np.zeros((B, y.size), dtype=np.float64)
    for b in range(B):
        conteos[b] = np.bincount(sel_pos[b], minlength=y.size) + np.bincount(sel_neg[b], minlength=y.size)
    res = M.calcular(M.preparar(y, score), conteos)
    return {"ap": np.asarray(res["ap"], dtype=np.float64), "roc_auc": np.asarray(res["roc_auc"], dtype=np.float64), "B": B, "semilla": semilla, "numpy": np.__version__}


def _percentiles(v: np.ndarray) -> dict:
    v = v[np.isfinite(v)]
    p = np.percentile(v, [2.5, 97.5])
    return {"p2_5": float(p[0]), "p97_5": float(p[1]), "n_validos": int(v.size), "sha256_replicas": hashlib.sha256(np.ascontiguousarray(v, dtype="<f8").tobytes()).hexdigest()}


def matriz(y, score, umbral: float) -> dict:
    """Matriz de confusión con la regla `score >= umbral` (los empates con el umbral son positivos)."""
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(score, dtype=np.float64) >= umbral
    tp, fn = int(np.sum(pred & (y == 1))), int(np.sum(~pred & (y == 1)))
    fp, tn = int(np.sum(pred & (y == 0))), int(np.sum(~pred & (y == 0)))
    return {"umbral": float(umbral), "tp": tp, "fp": fp, "tn": tn, "fn": fn, "positivos_predichos": tp + fp,
            "sensibilidad": tp / (tp + fn), "especificidad": tn / (tn + fp), "vpp": (tp / (tp + fp)) if (tp + fp) > 0 else None}


def calcular_resultados(y, score, t_hat: float, *, B: int = N_BOOTSTRAP, semilla: int = SEMILLA_BOOTSTRAP) -> dict:
    y, score = _validar(y, score)
    n_pos, n_neg = int(y.sum()), int(y.size - y.sum())
    prim = matriz(y, score, t_hat)
    prim["sensibilidad_ic95_clopper_pearson"] = clopper_pearson(prim["tp"], n_pos)
    prim["especificidad_ic95_clopper_pearson"] = clopper_pearson(prim["tn"], n_neg)
    prim["vpp_nota"] = "VPP con la prevalencia del test; no transferible a otras poblaciones; sin IC aprobado"
    boot = bootstrap_estratificado(y, score, B, semilla)
    return {
        "n": int(y.size), "n_positivas": n_pos, "n_negativas": n_neg, "prevalencia_test": n_pos / y.size,
        "puntuales": {"ap": M.ap(y, score), "roc_auc": M.roc_auc(y, score)},
        "ic95_bootstrap_percentil_estratificado": {"ap": _percentiles(boot["ap"]), "roc_auc": _percentiles(boot["roc_auc"]), "B": B, "semilla": semilla,
                                                   "numpy": boot["numpy"], "unidad": "imagen (sin group_id en test; independencia no demostrada)"},
        "umbral_primario": prim,
        "umbral_secundario_0_5_descriptivo": matriz(y, score, UMBRAL_SECUNDARIO)}
