"""Bootstrap agrupado por `group_id` (protocolo §8.1).

* Unidad = grupo pHash ≤ 8 (nunca se divide); cada grupo tiene una sola clase. NO son pacientes.
* Estratificado por la clase del grupo: en cada remuestreo `b` se extraen con reemplazo tantos grupos negativos y positivos como
  los observados; cada imagen entra tantas veces como su grupo. Sensibilidad: bootstrap simple de todos los grupos.
* UN único `b` para todas las arquitecturas y semillas (las semillas no se concatenan como imágenes ni se remuestrean); la
  métrica por arquitectura es la media entre semillas dentro de cada `b`, y ΔAP usa el mismo `b`.
* IC95 por percentiles 2.5 y 97.5. Los remuestreos sin una clase se cuentan y se excluyen de los percentiles.
* Los conteos (B x grupos) se guardan o se regeneran con la misma semilla y la misma versión de numpy (se registra).
"""

from __future__ import annotations

import itertools

import numpy as np

from . import metricas as M

METRICAS_IC = ("ap", "roc_auc", "sensibilidad_05", "especificidad_05")
METRICAS_DELTA = ("ap", "roc_auc")


def indexar_grupos(group_ids, y):
    """(grupo_por_imagen, clase_de_grupo, nombres): índice entero de grupo por imagen y la clase única de cada grupo."""
    nombres = sorted(set(group_ids))
    pos = {g: i for i, g in enumerate(nombres)}
    grupo_por_imagen = np.array([pos[g] for g in group_ids], dtype=np.int64)
    y = np.asarray(y)
    clase = np.full(len(nombres), -1, dtype=np.int64)
    for i, g in enumerate(grupo_por_imagen):
        if clase[g] not in (-1, y[i]):
            raise ValueError(f"el grupo {nombres[g]!r} mezcla clases")
        clase[g] = y[i]
    return grupo_por_imagen, clase, nombres


def generar_conteos(clase_de_grupo, B: int, semilla: int, estratificado: bool = True) -> np.ndarray:
    """Matriz (B, G) uint16: cuántas veces entra cada grupo en cada remuestreo `b`. Determinista dado (semilla, numpy)."""
    clase = np.asarray(clase_de_grupo)
    G = clase.size
    rng = np.random.default_rng(semilla)

    def contar(cols):                                                # cols: (B, m) índices de grupo
        plano = (np.arange(B)[:, None] * G + cols).ravel()
        return np.bincount(plano, minlength=B * G).reshape(B, G)

    if estratificado:
        conteos = np.zeros((B, G), dtype=np.int64)
        for c in (0, 1):                                              # primero negativos, luego positivos (orden fijo)
            miembros = np.flatnonzero(clase == c)
            if miembros.size == 0:
                continue
            elegidos = miembros[rng.integers(0, miembros.size, size=(B, miembros.size))]
            conteos += contar(elegidos)
    else:
        conteos = contar(rng.integers(0, G, size=(B, G)))
    return conteos.astype(np.uint16)


def distribuciones(preps: dict, conteos: np.ndarray, grupo_por_imagen: np.ndarray, bloque: int = 500) -> dict:
    """{(arq, semilla): {métrica: array (B,)}} con el MISMO `b` para todas las claves."""
    B = conteos.shape[0]
    salida = {k: {} for k in preps}
    for i in range(0, B, bloque):
        pesos = conteos[i:i + bloque][:, grupo_por_imagen]
        for k, prep in preps.items():
            for nombre, v in M.calcular(prep, pesos).items():
                salida[k].setdefault(nombre, []).append(v)
    return {k: {n: np.concatenate(v) for n, v in d.items()} for k, d in salida.items()}


def percentiles(x, nivel: float = 0.95):
    """(inferior, superior, n_validos) por percentiles; ignora NaN (remuestreos sin una clase)."""
    x = np.asarray(x, dtype=float)
    ok = x[~np.isnan(x)]
    if ok.size == 0:
        return float("nan"), float("nan"), 0
    a = (1.0 - nivel) / 2 * 100
    inf, sup = np.percentile(ok, [a, 100 - a])
    return float(inf), float(sup), int(ok.size)


def media_entre_semillas(dist: dict, arq: str, semillas, metrica: str) -> np.ndarray:
    return np.mean([dist[(arq, s)][metrica] for s in semillas], axis=0)


def analizar(preps: dict, arquitecturas, semillas, conteos, grupo_por_imagen, nivel: float = 0.95) -> dict:
    """Estimación puntual (media entre semillas), IC95 y ΔAP/ΔAUC emparejadas entre arquitecturas con el mismo `b`."""
    dist = distribuciones(preps, conteos, grupo_por_imagen)
    puntual = {k: M.calcular(p) for k, p in preps.items()}
    por_arq = {}
    for a in arquitecturas:
        por_arq[a] = {}
        for m in METRICAS_IC:
            medias = media_entre_semillas(dist, a, semillas, m)
            inf, sup, n_ok = percentiles(medias, nivel)
            por_arq[a][m] = {"estimacion": float(np.mean([puntual[(a, s)][m] for s in semillas])),
                             "ic_inf": inf, "ic_sup": sup, "n_remuestreos_validos": n_ok}
    deltas = {}
    for a, b in itertools.combinations(arquitecturas, 2):            # orden de la configuración; Δ = M_a - M_b
        deltas[f"{a}__menos__{b}"] = {}
        for m in METRICAS_DELTA:
            d = media_entre_semillas(dist, a, semillas, m) - media_entre_semillas(dist, b, semillas, m)
            inf, sup, n_ok = percentiles(d, nivel)
            est = np.mean([puntual[(a, s)][m] for s in semillas]) - np.mean([puntual[(b, s)][m] for s in semillas])
            deltas[f"{a}__menos__{b}"][m] = {"estimacion": float(est), "ic_inf": inf, "ic_sup": sup,
                                              "n_remuestreos_validos": n_ok, "ic_excluye_cero": bool(inf > 0 or sup < 0)}
    primera = next(iter(dist.values()))["ap"]
    return {"por_arquitectura": por_arq, "deltas": deltas, "n_remuestreos": int(primera.size),
            "remuestreos_sin_una_clase": int(np.isnan(primera).sum())}


def diferencia_maxima_extremos(a: dict, b: dict) -> float:
    """Mayor diferencia absoluta entre los extremos del IC de dos corridas del bootstrap (se REPORTA; sin umbral)."""
    d = []
    for arq, ms in a["por_arquitectura"].items():
        for m, v in ms.items():
            w = b["por_arquitectura"][arq][m]
            d += [abs(v["ic_inf"] - w["ic_inf"]), abs(v["ic_sup"] - w["ic_sup"])]
    for par, ms in a["deltas"].items():
        for m, v in ms.items():
            w = b["deltas"][par][m]
            d += [abs(v["ic_inf"] - w["ic_inf"]), abs(v["ic_sup"] - w["ic_sup"])]
    return float(np.nanmax(d))


def guardar_indices(path, conteos, *, semilla: int, estratificado: bool, nombres_de_grupo) -> None:
    """Guarda los conteos de grupo por remuestreo (archivo nuevo; no sobrescribe)."""
    with open(path, "xb") as f:
        np.savez_compressed(f, conteos=conteos, semilla=semilla, estratificado=estratificado,
                            grupos=np.array(nombres_de_grupo), numpy_version=np.__version__)


def cargar_indices(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
