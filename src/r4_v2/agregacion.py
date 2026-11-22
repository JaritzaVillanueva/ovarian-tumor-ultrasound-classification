"""Agregación del AP OOF y métricas (protocolo §7), con la comprobación K7 de integridad de la agregación.

Primaria: AP sobre las 820 predicciones OOF de cada (arquitectura, semilla) y media entre semillas. Complementarias: AP por fold
(media entre semillas) y AP del ensamble de semillas por fold. Solo lee corridas `completada` del modo `completa`; las de `humo`
y `referencia` no entran (protocolo §4.7). Las métricas por fold son descriptivas (7-8 positivas por fold).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import bootstrap as B
from . import metricas as M
from .common import GuardError, CONFIG_REL
from .entrenamiento import corrida_completada, id_corrida, leer_prob_por_epoca
from .particion import Particion


@dataclass
class OOF:
    image_ids: list
    y: np.ndarray
    fold: np.ndarray
    group_ids: list
    prob: dict = field(default_factory=dict)        # (arq, semilla) -> vector de 820 (float64 con valores float32 exactos)
    run_configs: dict = field(default_factory=dict)  # (arq, semilla, fold) -> run_config


def cargar_corridas(raiz_completa, config, particion: Particion) -> dict:
    """{(arq, semilla, fold): (directorio, run_config)} de las 45 corridas `completa` completadas; falla si falta alguna."""
    d = config["diseno"]
    corridas, faltan = {}, []
    for a in d["arquitecturas"]:
        for s in d["semillas"]:
            for k in d["folds"]:
                intento = corrida_completada(Path(raiz_completa) / id_corrida(config, a, s, k))
                if intento is None:
                    faltan.append(id_corrida(config, a, s, k))
                    continue
                rc = json.loads((intento / "run_config.json").read_text(encoding="utf-8"))
                if rc["modo"] != "completa" or rc["invalida_integridad"] or rc["estado"] != "completada":
                    raise GuardError(f"{intento.parent.name}: no es una corrida completa válida")
                corridas[(a, s, k)] = (intento, rc)
    if faltan:
        raise GuardError(f"K7: faltan {len(faltan)} corridas completadas (p. ej. {faltan[:3]})")
    _verificar_comparabilidad(corridas, particion)
    return corridas


def _verificar_comparabilidad(corridas, particion) -> None:
    """Clave de comparabilidad (§12): misma partición, misma configuración y misma referencia de pesos en todas las corridas."""
    claves = set()
    for (_, rc) in corridas.values():
        cfg_blob = rc["verificacion_inicio"]["archivos"][CONFIG_REL]["sha256_contenido_blob"]
        ref = rc.get("referencia_pesos") or {}
        claves.add((rc["sha256_bytes_archivo_particion"], cfg_blob, ref.get("sha256_contenido_blob"), rc["mapeo"], rc["particion"]))
    if len(claves) != 1:
        raise GuardError("las corridas no son comparables: difieren la partición, la configuración o la referencia de pesos")
    if next(iter(claves))[0] != particion.sha256_bytes_archivo:
        raise GuardError("las corridas se hicieron con una partición distinta de la cargada (sha256_bytes_archivo)")


def ensamblar_oof(corridas, particion: Particion, config, *, epoca_alternativa: str | None = None) -> OOF:
    """Vector de 820 predicciones por (arq, semilla). K7: cada imagen exactamente una vez y `y_true` = `label_v2`.

    `epoca_alternativa` (p. ej. 'epoca_perdida_ponderada_minima') toma esa época de la trayectoria (§6.3, descriptivo).
    """
    ids = [r["image_id"] for r in particion.filas]
    y = np.array([r["label_v2"] for r in particion.filas], dtype=np.int64)
    fold = np.array([r["fold"] for r in particion.filas], dtype=np.int64)
    pos = {i: n for n, i in enumerate(ids)}
    oof = OOF(image_ids=ids, y=y, fold=fold, group_ids=[r["group_id"] for r in particion.filas])
    for a in config["diseno"]["arquitecturas"]:
        for s in config["diseno"]["semillas"]:
            v = np.full(len(ids), np.nan)
            visto = np.zeros(len(ids), dtype=int)
            for k in config["diseno"]["folds"]:
                intento, rc = corridas[(a, s, k)]
                oof.run_configs[(a, s, k)] = rc
                if epoca_alternativa is None:
                    d = _leer_oof_csv(intento / "predicciones_oof.csv")
                else:
                    epoca = rc["seleccion"]["sensibilidad_checkpoint"][epoca_alternativa]
                    d = leer_prob_por_epoca(intento / "probabilidades_por_epoca.csv")[epoca]
                for i, yy, p in zip(d["image_id"], d["y"], d["p"]):
                    if i not in pos:
                        raise GuardError(f"K7: image_id {i!r} fuera de la partición")
                    if y[pos[i]] != yy or fold[pos[i]] != k:
                        raise GuardError(f"K7: la imagen {i} no corresponde al fold {k} o su y_true difiere de label_v2")
                    v[pos[i]] = p
                    visto[pos[i]] += 1
            if not (visto == 1).all():
                raise GuardError(f"K7: ({a}, s{s}) no tiene cada imagen exactamente una vez")
            oof.prob[(a, s)] = v
    return oof


def _leer_oof_csv(path) -> dict:
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    return {"image_id": [r["image_id"] for r in filas], "y": np.array([int(r["y_true"]) for r in filas]),
            "p": np.array([float(r["probability_malignant"]) for r in filas], dtype=np.float32)}


# ---------------------------------------------------------------- métricas
def metricas_por_semilla(oof: OOF, config) -> dict:
    """{(arq, semilla): métricas sobre las 820 predicciones}."""
    return {k: M.metricas(oof.y, p) for k, p in oof.prob.items()}


def media_entre_semillas(por_semilla: dict, config, metrica: str) -> dict:
    return {a: float(np.mean([por_semilla[(a, s)][metrica] for s in config["diseno"]["semillas"]]))
            for a in config["diseno"]["arquitecturas"]}


def ap_por_fold(oof: OOF, config) -> dict:
    """{(arq, semilla, fold): (AP, n_positivas)} — descriptivo."""
    salida = {}
    for (a, s), p in oof.prob.items():
        for k in config["diseno"]["folds"]:
            m = oof.fold == k
            salida[(a, s, k)] = (M.ap(oof.y[m], p[m]), int(oof.y[m].sum()))
    return salida


def ap_ensamble_por_fold(oof: OOF, config) -> dict:
    """{arq: {"agregado": AP sobre las 820 de q̄, "por_fold": {fold: AP}}} con q̄ = media de semillas (OOF de un ensamble POR FOLD,
    no de un sistema final; §7 nota)."""
    salida = {}
    for a in config["diseno"]["arquitecturas"]:
        q = np.mean([oof.prob[(a, s)] for s in config["diseno"]["semillas"]], axis=0)
        salida[a] = {"agregado": M.ap(oof.y, q), "por_fold": {k: M.ap(oof.y[oof.fold == k], q[oof.fold == k])
                                                              for k in config["diseno"]["folds"]}}
    return salida


def ap_sensibilidad_checkpoint(corridas, particion, config) -> dict:
    """§6.3: AP agregado por arquitectura (media entre semillas) con la época alternativa de cada regla; descriptivo."""
    salida = {}
    for alt in ("epoca_perdida_ponderada_minima", "epoca_roc_auc_maximo"):
        o = ensamblar_oof(corridas, particion, config, epoca_alternativa=alt)
        ps = metricas_por_semilla(o, config)
        salida[alt] = media_entre_semillas(ps, config, "ap")
    return salida


# ---------------------------------------------------------------- bootstrap
def preparar_bootstrap(oof: OOF):
    preps = {k: M.preparar(oof.y, p) for k, p in oof.prob.items()}
    grupo_por_imagen, clase, nombres = B.indexar_grupos(oof.group_ids, oof.y)
    return preps, grupo_por_imagen, clase, nombres


def ejecutar_bootstrap(oof: OOF, config, *, n_remuestreos: int | None = None) -> dict:
    """Bootstrap primario (estratificado), sensibilidad no estratificada y repetición con la segunda semilla (§8.1).

    `n_remuestreos` solo se reduce en pruebas; en G2 es `bootstrap.n_remuestreos` (10 000).
    """
    cfg = config["bootstrap"]
    n_b = int(n_remuestreos or cfg["n_remuestreos"])
    archs, semillas = config["diseno"]["arquitecturas"], config["diseno"]["semillas"]
    preps, gpi, clase, nombres = preparar_bootstrap(oof)
    out = {"n_remuestreos": n_b, "numpy_version": np.__version__, "n_grupos": len(nombres),
           "n_grupos_positivos": int((clase == 1).sum()), "n_grupos_negativos": int((clase == 0).sum())}
    conteos = {}
    for etiqueta, semilla, estrat in (("estratificado", cfg["semilla"], True), ("no_estratificado", cfg["semilla"], False),
                                      ("estratificado_semilla_comprobacion", cfg["semilla_de_comprobacion"], True)):
        c = B.generar_conteos(clase, n_b, semilla, estratificado=estrat)
        conteos[etiqueta] = (c, semilla, estrat)
        out[etiqueta] = B.analizar(preps, archs, semillas, c, gpi, nivel=cfg["intervalo"]["nivel"])
        out[etiqueta]["semilla"] = semilla
    out["diferencia_maxima_extremos_ic_entre_semillas"] = B.diferencia_maxima_extremos(
        out["estratificado"], out["estratificado_semilla_comprobacion"])
    out["_conteos"] = conteos
    out["_nombres_de_grupo"] = nombres
    return out
