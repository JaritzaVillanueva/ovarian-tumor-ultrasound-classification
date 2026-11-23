"""Reporte de G4 `g4_optimizacion_v1` (modos `fase_a`, `fase_b`, `fase_c`) desde predicciones guardadas; solo desarrollo; sin texto interpretativo.

Dos ejes por configuración frente a la base A|densenet121 (misma partición y mismos índices `b` de G2):
  (1) discriminación: delta AP emparejada (primaria, sin redondear; ROC-AUC secundaria), IC95 % estratificado y no estratificado, tres niveles de evidencia y
      multiplicidad (IC ajustado de nivel 1 - 0.05/K, K = configuraciones efectivamente evaluadas, incluidas las del agente);
  (2) punto de operación con S* = 0.90 (transferencia entre folds: umbral calibrado con `calibracion.calibrar_umbral` en 4 folds y evaluado en el 5.º; puntuación =
      media de las semillas disponibles): sensibilidad, especificidad, FP, FN, VPP con IC95 % bootstrap agrupado y umbral RECALCULADO en cada remuestreo
      (vectorizado, sin reducir B), y comparación pareada (config - base) cuyo deterioro respaldado (límite superior < 0 en sensibilidad O especificidad) es el VETO.
El veto formal solo aplica con 3 semillas completas. Solo con corridas pedidas completas (exit 0/3/4); directorio nuevo (K5); integridad al inicio y al fin.
Partición provisional (phash_le8_v1, exploratoria, post hoc); conformidad académica pendiente. Viabilidad clínica: no determinada.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.r4_v2 import agregacion as AG
from src.r4_v2 import bootstrap as BS
from src.r4_v2 import calibracion as CAL
from src.r4_v2 import metricas as M
from src.r4_v2 import pesos as W
from src.r4_v2.common import cargar_config as cargar_config_g2
from src.r4_v2.common import directorio_nuevo, escribir_json_nuevo, exigir_salida_segura, sha256_archivo
from src.r4_v2.entrenamiento import corrida_completada
from src.r4_v2.particion import cargar_particion
from src.r5_v2.comun import comparar_instantaneas
from src.r5_v2.g3b_reporte_condiciones import media_dist, verificar_indices_g2

from . import g4_agente as AGT
from . import g4_comun as C
from . import g4_presupuesto as P

GuardError = C.GuardError
CODIGO_SALIDA_GUARDA, CODIGO_SALIDA_INTEGRIDAD_INVALIDA = 3, 4
MODOS = ("fase_a", "fase_b", "fase_c")
BASE = "BASE"
NIVELES = ("candidata prometedora", "mejora mejor respaldada", "sin mejora suficientemente respaldada")
ENCABEZADO = (f"# mapeo v2 | phash_le8_v1 (provisional) | {C.PROTOCOLO_ID} | sin uso de test | {C.LEYENDA} | " + " | ".join(C.LEYENDAS_FIJAS))
NOTA_VPP = ("VPP calculado con la prevalencia del conjunto de desarrollo ({pos}/{n} = {prev:.1%}); solo válido para esa prevalencia; "
            "no transferible a otras poblaciones")
NOTA_PARCIAL = "parciales exploratorios provisionales; no se declara la mejor de las seis"


# ============================================================================ reglas puras (probadas con entradas sintéticas)
def ic(x, nivel: float = C.NIVEL_IC) -> dict:
    inf, sup, n = BS.percentiles(x, nivel)
    return {"ic_inf": inf, "ic_sup": sup, "n_remuestreos_validos": n}


def clasificar_punto_operacion(d_sens: dict, d_esp: dict, n_semillas: int) -> dict:
    """Clasificación del punto de operación a partir de los IC95 % pareados (config - base) de sensibilidad y especificidad transferidas.

    deterioro respaldado: límite superior < 0 en cualquiera de las dos (PREVALECE sobre una mejora en la otra); mejora respaldada: IC completamente positivo en
    al menos una y sin deterioro en la otra; no distinguible: lo demás. El veto formal solo con 3 semillas; con menos se rotula «exploratoria; sin veto formal».
    """
    def neg(d): return d["n_remuestreos_validos"] > 0 and d["ic_sup"] < 0
    def pos(d): return d["n_remuestreos_validos"] > 0 and d["ic_inf"] > 0
    det = [n for n, d in (("sensibilidad", d_sens), ("especificidad", d_esp)) if neg(d)]
    mej = [n for n, d in (("sensibilidad", d_sens), ("especificidad", d_esp)) if pos(d)]
    clase = "deterioro respaldado" if det else ("mejora respaldada" if mej else "no distinguible")
    formal = n_semillas == C.SEMILLAS_VETO_FORMAL
    return {"clasificacion": clase, "deterioro_en": det, "mejora_en": mej, "veto_formal": formal, "veto_aplicado": bool(formal and det),
            "veto_prevalece_sobre_mejora": bool(det and mej), "rotulo": None if formal else "exploratoria; sin veto formal", "n_semillas": n_semillas}


def nivel_evidencia(delta_por_semilla: dict, ic_est: dict, ic_nos: dict, semillas) -> dict:
    """Tres niveles de evidencia en AP. (i) candidata prometedora: delta AP de la semilla 42 > 0. (ii) mejora mejor respaldada: 3 semillas completas; (a) delta
    medio > 0 y > 0 en cada semilla; (b) IC95 % que excluye 0 (por arriba) en el bootstrap estratificado Y en el no estratificado. (iii) lo demás."""
    d42 = delta_por_semilla.get(42)
    completo = sorted(semillas) == [42, 43, 44]
    vals = list(delta_por_semilla.values())
    cumple_a = bool(completo and float(np.mean(vals)) > 0 and all(v > 0 for v in vals))
    cumple_b = bool(completo and ic_est["n_remuestreos_validos"] > 0 and ic_nos["n_remuestreos_validos"] > 0 and ic_est["ic_inf"] > 0 and ic_nos["ic_inf"] > 0)
    candidata = bool(d42 is not None and d42 > 0)
    nivel = NIVELES[1] if (cumple_a and cumple_b) else (NIVELES[0] if candidata else NIVELES[2])
    return {"nivel": nivel, "candidata_prometedora": candidata, "cumple_a": cumple_a, "cumple_b": cumple_b, "nivel_ii_evaluable": completo}


def decidir_reemplazo(nivel: dict, clasif: dict) -> dict:
    """Una configuración solo reemplaza a DenseNet121 + A si cumple (i) mejora mejor respaldada en AP y (ii) NO tiene deterioro respaldado."""
    i = nivel["nivel"] == NIVELES[1]
    ii = clasif["clasificacion"] != "deterioro respaldado"
    if i and ii:
        return {"decision": "puede reemplazar a la base", "cumple_i": True, "cumple_ii": True, "nota": None}
    if i:
        return {"decision": "conserva la base", "cumple_i": True, "cumple_ii": False, "nota": "mejora de discriminación con deterioro del punto de operación"}
    return {"decision": "conserva la base", "cumple_i": False, "cumple_ii": ii, "nota": None}


def atribucion_agente(nivel_vs_conv: dict, clasif_vs_conv: dict, nivel_vs_base: dict) -> dict:
    """La mejora adicional atribuible al agente solo se afirma si, frente a la candidata convencional, cumple (i) y no tiene deterioro respaldado."""
    atrib = bool(nivel_vs_conv["nivel"] == NIVELES[1] and clasif_vs_conv["clasificacion"] != "deterioro respaldado")
    nota = None
    if not atrib:
        nota = ("superar solo la base NO basta: la comparación con la candidata convencional de su familia no respalda una mejora adicional atribuible al agente"
                if nivel_vs_base["nivel"] == NIVELES[1] else "sin mejora respaldada frente a la candidata convencional")
        if not nivel_vs_conv["nivel_ii_evaluable"]:
            nota += "; semillas insuficientes para evaluar (3 semillas requeridas)"
    return {"atribuible_al_agente": atrib, "nota": nota}


# ============================================================================ transferencia con umbral recalculado, vectorizada
def transferencia_ponderada(y, q, folds, conteos, gpi, s_star: str = C.S_STAR) -> dict:
    """Transferencia entre folds con pesos enteros por imagen (`conteos[b, gpi[i]]`): para cada remuestreo `b` y cada fold k se recalcula el umbral con los
    otros folds (k-ésima mayor puntuación positiva con k = ceil(S*·n⁺_ponderado), empates sin romper, regla `p >= umbral`) y se evalúa en el fold k.
    Equivale a `calibracion.calibrar_umbral` sobre las filas repetidas. Devuelve arrays (B,): tp, fn, fp, tn, sens, esp, vpp (NaN si el remuestreo no tiene
    positivas en el entrenamiento de algún fold)."""
    s = CAL.como_fraccion(s_star)
    y, q, folds, gpi = np.asarray(y), np.asarray(q, dtype=np.float64), np.asarray(folds), np.asarray(gpi)
    B = conteos.shape[0]
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    orden = pos[np.argsort(-q[pos], kind="stable")]
    sp, fp_ = q[orden], folds[orden]
    wp = conteos[:, gpi[orden]].astype(np.int64)
    tp, fn, fpc, tn = (np.zeros(B) for _ in range(4))
    ok = np.ones(B, dtype=bool)
    for k in sorted(set(folds.tolist())):
        ent = fp_ != k
        wt = wp * ent[None, :]
        n_w = wt.sum(axis=1)
        kk = -(-(s.numerator * n_w) // s.denominator)                                  # ceil(S*·n⁺) con aritmética entera
        ok &= n_w >= 1
        t = sp[(np.cumsum(wt, axis=1) >= kk[:, None]).argmax(axis=1)]                   # k-ésima mayor puntuación positiva (con multiplicidad)
        ev = ~ent
        wpe = wp[:, ev]
        tpk = (wpe * (sp[ev][None, :] >= t[:, None])).sum(axis=1)
        tp += tpk
        fn += wpe.sum(axis=1) - tpk
        ineg = neg[folds[neg] == k]
        wn = conteos[:, gpi[ineg]].astype(np.int64)
        fpk = (wn * (q[ineg][None, :] >= t[:, None])).sum(axis=1)
        fpc += fpk
        tn += wn.sum(axis=1) - fpk
    with np.errstate(divide="ignore", invalid="ignore"):
        sens, esp, vpp = tp / (tp + fn), tn / (tn + fpc), tp / (tp + fpc)
    nan = np.where(ok, 1.0, np.nan)
    return {"tp": tp * nan, "fn": fn * nan, "fp": fpc * nan, "tn": tn * nan, "sens": sens * nan, "esp": esp * nan, "vpp": vpp * nan}


def punto_operacion_puntual(y, q, folds, s_star: str = C.S_STAR) -> dict:
    """Estimación puntual con `calibracion.comprobar_transferencia` (la función de referencia) y su control con la versión vectorizada (pesos 1)."""
    y, q, folds = np.asarray(y), np.asarray(q, dtype=np.float64), np.asarray(folds)
    ref = CAL.comprobar_transferencia(y.tolist(), q.tolist(), folds.tolist(), s_star)
    pf = ref["por_fold"]
    tp, fn, fp, tn = (sum(v[c] for v in pf.values()) for c in ("tp", "fn", "fp", "tn"))
    vec = transferencia_ponderada(y, q, folds, np.ones((1, y.size), dtype=np.uint16), np.arange(y.size), s_star)
    if (vec["tp"][0], vec["fn"][0], vec["fp"][0], vec["tn"][0]) != (tp, fn, fp, tn):
        raise GuardError("la transferencia vectorizada no coincide con calibracion.comprobar_transferencia (pesos 1): se detiene")
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "n_malignas": tp + fn, "n_no_malignas": tn + fp,
            "sens": tp / (tp + fn) if tp + fn else float("nan"), "esp": tn / (tn + fp) if tn + fp else float("nan"),
            "vpp": tp / (tp + fp) if tp + fp else float("nan"), "prevalencia": (tp + fn) / (tp + fn + tn + fp), "s_star": ref["s_star"],
            "por_fold": {int(k): {"umbral": repr(float(v["umbral"])), "tp": v["tp"], "fn": v["fn"], "fp": v["fp"], "tn": v["tn"]} for k, v in pf.items()}}


def resumir_dist(puntual: dict, dist: dict) -> dict:
    return {m: {"estimacion": puntual[m], **ic(dist[m])} for m in ("sens", "esp", "fp", "fn", "vpp")}


def delta_punto_operacion(pun_x: dict, dist_x: dict, pun_y: dict, dist_y: dict, n_semillas: int) -> dict:
    """Comparación PAREADA x - y (mismos `b`): diferencias de sensibilidad, especificidad, FP, FN y VPP con IC95 % percentil SIN ajustar."""
    out = {}
    for m, clave in (("sens", "sens"), ("esp", "esp"), ("fp", "fp"), ("fn", "fn"), ("vpp", "vpp")):
        out[m] = {"estimacion": pun_x[clave] - pun_y[clave], **ic(dist_x[clave] - dist_y[clave])}
    out["clasificacion"] = clasificar_punto_operacion(out["sens"], out["esp"], n_semillas)
    return out


# ============================================================================ carga (solo si no falta ninguna corrida pedida)
def cargar_oof(root, config, config_g2, particion, estrategia, semillas):
    """OOF de una estrategia (o de la base si `estrategia is None`) para esas semillas; falla si falta alguna corrida y verifica la comparabilidad."""
    if estrategia is None:
        g, raiz = C.config_g2_base(config_g2, config, semillas), C.raiz_base(root, config_g2)
        cid = BASE
        plan = C.plan_de_corridas(g, cid)
    else:
        g, raiz, cid = C.config_g2_g4(config_g2, config, estrategia, semillas), C.raiz_corridas(root, config, estrategia["id"], "completa"), estrategia["id"]
        plan = C.plan_de_corridas(g, cid)
    faltan = [p["id"] for p in plan if corrida_completada(raiz / p["id"]) is None]
    if faltan:
        raise GuardError(f"faltan {len(faltan)} de {len(plan)} corridas de {cid} (p. ej. {faltan[:3]}): no se generan métricas parciales")
    corr = AG.cargar_corridas(raiz, g, particion)
    if estrategia is not None:
        for _, rc in corr.values():
            C.exigir_corrida_comparable(rc, config_g2, estrategia, particion)
        for (_, _, _), (d, _) in corr.items():
            if not (Path(d) / "sello_g4.json").is_file():
                raise GuardError(f"{Path(d).parent.name}: sin sello_g4.json (no es una corrida de G4 con procedencia)")
    return AG.ensamblar_oof(corr, particion, g), corr


def contar_configuraciones_evaluadas(root, config) -> int:
    """K de la multiplicidad: configuraciones con al menos una corrida `completa` completada (incluidas las propuestas del agente)."""
    raiz = C.raiz_g4(root, config)
    n = 0
    for d in sorted(raiz.glob("*/completa")) if raiz.is_dir() else []:
        if C.ID_CONFIG.fullmatch(d.parent.name) and any(corrida_completada(r) is not None for r in d.iterdir() if r.is_dir()):
            n += 1
    return n


# ============================================================================ análisis
def analizar(oofs: dict, pares: list, arq: str, K: int, n_b: int, conteos_est, conteos_nos, gpi, tiempos: dict) -> dict:
    """`oofs`: {cid: (OOF, semillas)}; `pares`: [(x, y, semillas)]. Mismos `b` para todo; devuelve las comparaciones y los puntos de operación."""
    y, fold = oofs[BASE][0].y, oofs[BASE][0].fold
    preps = {(cid, arq, s): M.preparar(y, oof.prob[(arq, s)]) for cid, (oof, sems) in oofs.items() for s in sems}
    dist_e = BS.distribuciones(preps, conteos_est, gpi)
    dist_n = BS.distribuciones(preps, conteos_nos, gpi)
    puntual = {k: M.calcular(p) for k, p in preps.items()}
    ops = {}

    def op(cid, sems):
        clave = (cid, tuple(sems))
        if clave not in ops:
            t0 = time.perf_counter()
            q = np.mean([oofs[cid][0].prob[(arq, s)] for s in sems], axis=0)
            pun = punto_operacion_puntual(y, q, fold)
            dist = transferencia_ponderada(y, q, fold, conteos_est, gpi)
            ops[clave] = {"puntual": pun, "dist": dist}
            tiempos.setdefault("punto_operacion_s", []).append(time.perf_counter() - t0)
        return ops[clave]

    comps = []
    nivel_adj = 1.0 - C.ALFA_MULTIPLICIDAD / max(1, K)
    for x, yy, sems in pares:
        por_semilla = {s: float(puntual[(x, arq, s)]["ap"] - puntual[(yy, arq, s)]["ap"]) for s in sems}
        d_e = media_dist(dist_e, x, arq, sems, "ap") - media_dist(dist_e, yy, arq, sems, "ap")
        d_n = media_dist(dist_n, x, arq, sems, "ap") - media_dist(dist_n, yy, arq, sems, "ap")
        auc_e = media_dist(dist_e, x, arq, sems, "roc_auc") - media_dist(dist_e, yy, arq, sems, "roc_auc")
        auc_pt = float(np.mean([puntual[(x, arq, s)]["roc_auc"] - puntual[(yy, arq, s)]["roc_auc"] for s in sems]))
        media = float(np.mean(list(por_semilla.values())))
        ic_e, ic_n = ic(d_e), ic(d_n)
        niv = nivel_evidencia(por_semilla, ic_e, ic_n, sems)
        adj = {"nivel": nivel_adj, "k": K, **ic(d_e, nivel_adj)}
        sub = None
        if {43, 44} <= set(sems):                                                       # informativo; no es puerta
            sub = {"estimacion": float(np.mean([por_semilla[43], por_semilla[44]])),
                   **ic(media_dist(dist_e, x, arq, [43, 44], "ap") - media_dist(dist_e, yy, arq, [43, 44], "ap")), "informativo": True}
        ox, oy = op(x, sems), op(yy, sems)
        po = delta_punto_operacion(ox["puntual"], ox["dist"], oy["puntual"], oy["dist"], len(sems))
        comps.append({"x": x, "y": yy, "semillas": list(sems), "delta_ap": {"estimacion": media, "por_semilla": {str(k): v for k, v in por_semilla.items()}, **ic_e},
                      "delta_ap_no_estratificado": ic_n, "delta_roc_auc": {"estimacion": auc_pt, **ic(auc_e)}, "delta_ap_semillas_43_44": sub,
                      "multiplicidad": {**adj, "etiqueta": C.ETIQUETA_MULTIPLICIDAD,
                                        "respaldada_tras_ajuste_por_multiplicidad": bool(niv["nivel"] == NIVELES[1] and adj["n_remuestreos_validos"] > 0 and adj["ic_inf"] > 0)},
                      "nivel_evidencia": niv, "punto_operacion_pareado": po, "clasificacion_punto_operacion": po["clasificacion"],
                      "decision_reemplazo": decidir_reemplazo(niv, po["clasificacion"])})
    return {"comparaciones": comps, "puntos_de_operacion": {f"{c}|{'+'.join(map(str, s))}": {"config_id": c, "semillas": list(s), "puntual": v["puntual"],
                                                                                            "intervalos": resumir_dist(v["puntual"], v["dist"])}
                                                          for (c, s), v in ops.items()},
            "ap_por_semilla": {f"{c}|{s}": {"ap": float(puntual[(c, a, s)]["ap"]), "roc_auc": float(puntual[(c, a, s)]["roc_auc"])} for (c, a, s) in puntual}}


# ============================================================================ tablas
def _f(x) -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else repr(float(x))


def escribir_tabla(path, filas, columnas, extra: str = "") -> None:
    with open(path, "x", encoding="utf-8", newline="") as fh:
        fh.write(ENCABEZADO + (f" | {extra}" if extra else "") + "\n")
        w = csv.DictWriter(fh, fieldnames=columnas, lineterminator="\n")
        w.writeheader()
        w.writerows(filas)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def tablas(res: dict, resumen_ledger: dict, maximo: int, estr_por_id: dict) -> dict:
    def etiqueta(c): return c
    t1 = [{"config_id": k.split("|")[0], "semilla": k.split("|")[1], "ap": _f(v["ap"]), "roc_auc": _f(v["roc_auc"])} for k, v in sorted(res["ap_por_semilla"].items())]
    t2, t4, t6 = [], [], []
    for c in res["comparaciones"]:
        d, n = c["delta_ap"], c["delta_ap_no_estratificado"]
        s = c["delta_ap_semillas_43_44"]
        t2.append({"comparacion": f"{c['x']} - {c['y']}", "semillas": " ".join(map(str, c["semillas"])), "delta_ap": _f(d["estimacion"]),
                   "delta_ap_por_semilla": " ".join(f"{k}:{_f(v)}" for k, v in d["por_semilla"].items()), "ic95_estrat_inf": _f(d["ic_inf"]), "ic95_estrat_sup": _f(d["ic_sup"]),
                   "ic95_no_estrat_inf": _f(n["ic_inf"]), "ic95_no_estrat_sup": _f(n["ic_sup"]), "delta_roc_auc": _f(c["delta_roc_auc"]["estimacion"]),
                   "delta_roc_auc_ic95_inf": _f(c["delta_roc_auc"]["ic_inf"]), "delta_roc_auc_ic95_sup": _f(c["delta_roc_auc"]["ic_sup"]),
                   "delta_ap_semillas_43_44_informativo": _f(s["estimacion"]) if s else "", "ic95_43_44_inf": _f(s["ic_inf"]) if s else "", "ic95_43_44_sup": _f(s["ic_sup"]) if s else "",
                   "ic_ajustado_nivel": _f(c["multiplicidad"]["nivel"]), "k": c["multiplicidad"]["k"], "ic_ajustado_inf": _f(c["multiplicidad"]["ic_inf"]),
                   "ic_ajustado_sup": _f(c["multiplicidad"]["ic_sup"]), "respaldada_tras_ajuste_por_multiplicidad": c["multiplicidad"]["respaldada_tras_ajuste_por_multiplicidad"],
                   "etiqueta_multiplicidad": C.ETIQUETA_MULTIPLICIDAD, "nivel_evidencia": c["nivel_evidencia"]["nivel"], "cumple_a": c["nivel_evidencia"]["cumple_a"],
                   "cumple_b": c["nivel_evidencia"]["cumple_b"]})
        po = c["punto_operacion_pareado"]
        cl = po["clasificacion"]
        t4.append({"comparacion": f"{c['x']} - {c['y']}", "semillas": " ".join(map(str, c["semillas"])), "n_semillas": cl["n_semillas"],
                   **{f"{m}_{p}": _f(po[m][p]) for m in ("sens", "esp") for p in ("estimacion", "ic_inf", "ic_sup")},
                   **{f"{m}_descriptivo_{p}": _f(po[m][p]) for m in ("fp", "fn", "vpp") for p in ("estimacion", "ic_inf", "ic_sup")},
                   "clasificacion": cl["clasificacion"], "veto_formal": cl["veto_formal"], "veto_aplicado": cl["veto_aplicado"],
                   "veto_prevalece_sobre_mejora": cl["veto_prevalece_sobre_mejora"], "rotulo": cl["rotulo"] or ""})
        t6.append({"comparacion": f"{c['x']} - {c['y']}", "nivel_evidencia": c["nivel_evidencia"]["nivel"], "clasificacion_punto_operacion": cl["clasificacion"],
                   "veto_aplicado": cl["veto_aplicado"], "decision": c["decision_reemplazo"]["decision"], "nota": c["decision_reemplazo"]["nota"] or "",
                   "rotulo_veto": cl["rotulo"] or ""})
    t3 = []
    for v in res["puntos_de_operacion"].values():
        p, iv = v["puntual"], v["intervalos"]
        fila = {"config_id": v["config_id"], "semillas": " ".join(map(str, v["semillas"])), "n_semillas": len(v["semillas"]), "s_star": C.S_STAR,
                "rotulo": "" if len(v["semillas"]) == C.SEMILLAS_VETO_FORMAL else "exploratoria; sin veto formal",
                "tp": p["tp"], "n_malignas": p["n_malignas"], "tn": p["tn"], "n_no_malignas": p["n_no_malignas"], "fp": p["fp"], "fn": p["fn"],
                "fp_sobre_no_malignas": f"{p['fp']}/{p['n_no_malignas']}", "fn_sobre_malignas": f"{p['fn']}/{p['n_malignas']}", "prevalencia": _f(p["prevalencia"])}
        for m in ("sens", "esp", "fp", "fn", "vpp"):
            fila[f"{m}_ic95_inf"], fila[f"{m}_ic95_sup"] = _f(iv[m]["ic_inf"]), _f(iv[m]["ic_sup"])
        fila["sens"], fila["esp"], fila["vpp"] = _f(p["sens"]), _f(p["esp"]), _f(p["vpp"])
        t3.append(fila)
    t5 = [{"concepto": k, "valor": json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v} for k, v in resumen_ledger.items() if not k.startswith("_")]
    t5.append({"concepto": "presupuesto_de_C (75 - consumidos, acotado a 25)", "valor": resumen_ledger["_presupuesto_c"]})
    t5.append({"concepto": "maximo_propuestas_del_agente (5n + 10 <= presupuesto de C; n <= 3)", "valor": maximo})
    return {"T-G4-1_ap_por_semilla": (t1, list(t1[0])), "T-G4-2_delta_ap": (t2, list(t2[0])), "T-G4-3_punto_operacion": (t3, list(t3[0])),
            "T-G4-4_delta_punto_operacion": (t4, list(t4[0])), "T-G4-5_presupuesto_ledger": (t5, ["concepto", "valor"]), "T-G4-6_decision": (t6, list(t6[0]))}


# ============================================================================ orquestación
def _estrategias_pedidas(root, config, config_g2, particion, modo, hasta):
    """(estrategias de la comparación principal, semillas, info) por modo; el conjunto pedido debe estar COMPLETO."""
    raiz = C.raiz_g4(root, config)
    todas = {e["id"]: e for e in C.estrategias_fase_a(config)}
    info = {"fase_a_completa": None, "promocion": None}
    if modo == "fase_a":
        sel = C.seleccionar_fase_a(config, hasta)
        info["fase_a_completa"] = len(sel) == len(todas)
        return sel, [config["diseno"]["fase_a_semilla"]], info
    ruta = C.ultimo_archivo(raiz, "promocion")
    prom = json.loads(ruta.read_text(encoding="utf-8")) if ruta else None
    if prom:
        info["fase_a_completa"], info["promocion"] = prom["fase_a_completa"], {"archivo": ruta.name, "sha256_bytes_archivo": sha256_archivo(ruta),
                                                                            "promocion_provisional": prom["promocion_provisional"]}
    if modo == "fase_b":
        if prom is None:
            raise GuardError("el reporte fase_b exige un promocion.json previo")
        nueva = C.calcular_promocion(root, config, config_g2, particion)
        if nueva["promovidas"] != prom["promovidas"]:
            raise GuardError("la promoción registrada no coincide con la recalculada desde las predicciones; repetir el modo promocion")
        if not prom["promovidas"]:
            raise GuardError("sin candidata promovida: no hay fase B que reportar")
        return [todas[i] for i in prom["promovidas"]], [42, 43, 44], info
    return None, [42], info


def generar_reporte(root, config, config_g2, particion, *, modo, hasta=None, etiqueta=None, verificar, referencia=None, salida=None, n_remuestreos=None) -> dict:
    if modo not in MODOS:
        raise GuardError(f"modo del reporte desconocido: {modo!r} (válidos: {MODOS})")
    root = Path(root)
    C.validar_config(config)
    C.comprobar_coherencia(config, config_g2)
    C.validar_reglas(config)
    sha_protocolo = C.exigir_protocolo_coherente(root)
    inicio = C.exigir_integridad(verificar())
    r6 = C.verificar_base_r6(root, config)                                              # la ausencia bloquea
    referencia = referencia or W.cargar_referencia(root)
    nombre = f"reporte_{modo}" + (f"_{etiqueta}" if etiqueta else "")
    salida = Path(salida) if salida else exigir_salida_segura(root, C.raiz_g4(root, config) / nombre, config["diseno"]["raiz_resultados"].strip("/"))
    if salida.exists():
        raise GuardError(f"K5: el directorio del reporte ya existe y no se sobrescribe: {salida}")
    arq = config["base"]["arquitectura"]
    estr, semillas, info = _estrategias_pedidas(root, config, config_g2, particion, modo, hasta)
    eventos = P.leer_ledger(C.ruta_ledger(root, config))
    oofs, pares, conv_info, props = {}, [], None, []
    if modo == "fase_c":
        ag = config["agente"]
        raiz = C.raiz_g4(root, config)
        evid = AGT.leer_evidencia_fase_b(raiz / ag["evidencia_fase_b"])
        fam = AGT.familia_seleccionada(evid)
        props = AGT.validar_propuestas(root, raiz / ag["carpeta_propuestas"], fam["familia"], max_propuestas=ag["max_propuestas"])
        if not props:
            raise GuardError("no hay propuestas del agente que reportar")
        conv = next((e for e in C.estrategias_fase_a(config) if e["id"] == fam["config_id"]), None)
        if conv is None:
            raise GuardError(f"la candidata convencional {fam['config_id']} no es una de las seis predefinidas")
        conv_info = {"config_id": conv["id"], "familia": conv["familia"], "evidencia_fase_b": {"archivo": ag["evidencia_fase_b"],
                                                                                          "sha256_bytes_archivo": sha256_archivo(raiz / ag["evidencia_fase_b"])}}
        oofs[conv["id"]] = (cargar_oof(root, config, config_g2, particion, conv, [42, 43, 44])[0], [42, 43, 44])
        pares.append((conv["id"], BASE, [42, 43, 44]))
        for p in props:
            e = p["estrategia"]
            conf = [s for s in (43, 44) if all(corrida_completada(C.raiz_corridas(root, config, e["id"], "completa") / r["id"]) is not None
                                              for r in C.plan_de_corridas(C.config_g2_g4(config_g2, config, e, [s]), e["id"]))]
            if conf not in ([], [43, 44]):
                raise GuardError(f"{e['id']}: confirmación incompleta (semillas completas {conf}); se requieren ambas (43 y 44) o ninguna")
            sems = [42] + conf
            oofs[e["id"]] = (cargar_oof(root, config, config_g2, particion, e, sems)[0], sems)
            pares += [(e["id"], BASE, sems), (e["id"], conv["id"], sems)]
    else:
        for e in estr:
            oofs[e["id"]] = (cargar_oof(root, config, config_g2, particion, e, semillas)[0], semillas)
            pares.append((e["id"], BASE, semillas))
    todas = sorted({s for _, sems in oofs.values() for s in sems})
    oofs[BASE] = (cargar_oof(root, config, config_g2, particion, None, todas)[0], todas)
    comp_base = C.comparar_con_base(root, config, config_g2, particion, referencia, todas)
    ref_oof = oofs[BASE][0]
    for cid, (o, _) in oofs.items():
        if o.image_ids != ref_oof.image_ids or not np.array_equal(o.y, ref_oof.y) or o.group_ids != ref_oof.group_ids:
            raise GuardError(f"{cid}: orden de imágenes, etiquetas o grupos distintos de la base")
    gpi, clase, nombres = BS.indexar_grupos(ref_oof.group_ids, ref_oof.y)
    cfg_b = config_g2["bootstrap"]
    n_b = int(n_remuestreos or cfg_b["n_remuestreos"])
    t0 = time.perf_counter()
    cont = {s: BS.generar_conteos(clase, n_b, s, estratificado=True) for s in (cfg_b["semilla"], cfg_b["semilla_de_comprobacion"])}
    cont_nos = BS.generar_conteos(clase, n_b, cfg_b["semilla"], estratificado=False)
    indices = (verificar_indices_g2(root, config, config_g2, cont, nombres) if n_remuestreos is None
               else [{"estado": "no verificado: n_remuestreos reducido (solo pruebas)"}])
    K = contar_configuraciones_evaluadas(root, config)
    tiempos = {"generar_indices_s": time.perf_counter() - t0}
    res = analizar(oofs, pares, arq, K, n_b, cont[cfg_b["semilla"]], cont_nos, gpi, tiempos)
    ops_s = tiempos.get("punto_operacion_s", [0.0])
    tiempos["estimacion"] = {"n_remuestreos": n_b, "segundos_medios_por_punto_de_operacion": float(np.mean(ops_s)),
                             "nota": "vectorizado; el costo es lineal en B y en el número de configuraciones; B no se reduce"}
    rl = P.resumen(eventos)
    rl["_presupuesto_c"] = P.presupuesto_fase_c(eventos)
    maximo = P.maximo_propuestas(eventos)
    partial = info["fase_a_completa"] is False
    # decisión sobre la base y atribución del agente
    orden = C.orden_de_configuraciones(props)
    decision = C.seleccionar_configuracion_final(res["comparaciones"], orden)             # determinista: mayor delta AP medio; empate: orden predefinido; sin elegibles: base
    agente = None
    if modo == "fase_c":
        agente = []
        for p in props:
            i = p["estrategia"]["id"]
            vb = next(c for c in res["comparaciones"] if c["x"] == i and c["y"] == BASE)
            vc = next(c for c in res["comparaciones"] if c["x"] == i and c["y"] == conv_info["config_id"])
            agente.append({"config_id": i, "propuesta": p["archivo"], "sha256_bytes_archivo": p["sha256_bytes_archivo"], "semillas": vb["semillas"],
                           "nivel_vs_base": vb["nivel_evidencia"]["nivel"], "clasificacion_vs_base": vb["clasificacion_punto_operacion"]["clasificacion"],
                           "decision_vs_base": vb["decision_reemplazo"], "nivel_vs_candidata_convencional": vc["nivel_evidencia"]["nivel"],
                           "clasificacion_vs_candidata_convencional": vc["clasificacion_punto_operacion"]["clasificacion"],
                           "delta_ap_semilla_42_vs_base": vb["delta_ap"]["por_semilla"]["42"], "delta_ap_semilla_42_vs_candidata_convencional": vc["delta_ap"]["por_semilla"]["42"],
                           **atribucion_agente(vc["nivel_evidencia"], vc["clasificacion_punto_operacion"], vb["nivel_evidencia"])})
        d42b = {a_["config_id"]: a_["delta_ap_semilla_42_vs_base"] for a_ in agente}
        d42c = {a_["config_id"]: a_["delta_ap_semilla_42_vs_candidata_convencional"] for a_ in agente}
        promo = C.elegir_confirmacion(d42b, d42c, [p["estrategia"]["id"] for p in props])
        for a_ in agente:
            a_["promocion_exploratoria_a_confirmacion"] = a_["config_id"] in promo["autorizadas"]
            a_["seleccionada_para_confirmacion"] = a_["config_id"] == promo["elegida"]
        agente = {"promocion_a_confirmacion": promo, "propuestas": agente}
    meta_fase = {"fase_a_completa": info["fase_a_completa"], "rotulo_fase_a": NOTA_PARCIAL if partial else None, "no_se_declara_mejor_de_las_seis": True if partial else None,
                 "promocion": info["promocion"]}
    salida = directorio_nuevo(salida)                                                   # K5: solo tras validar y calcular
    extra = NOTA_PARCIAL if partial else ""
    if modo == "fase_c":
        extra = (extra + " | " if extra else "") + "promoción del agente a confirmación: regla exploratoria, no evidencia confirmatoria"
    pun = ref_oof
    prev = float(pun.y.mean())
    for nom, (filas, cols) in tablas(res, rl, maximo, {}).items():
        escribir_tabla(salida / f"{nom}.csv", filas, cols, (extra + (" | " if extra else "") + NOTA_VPP.format(pos=int(pun.y.sum()), n=int(pun.y.size), prev=prev))
                       if nom in ("T-G4-3_punto_operacion", "T-G4-4_delta_punto_operacion") else extra)
    filas_sel = ([{"rango": n, "config_id": c_["config_id"], "delta_ap_medio": _f(c_["delta_ap_medio"]), "orden_predefinido": c_["orden"], "seleccionada": n == 1,
                   "criterio_desempate": decision["criterio_desempate_usado"] or ""} for n, c_ in enumerate(decision["cumplen_ordenadas"], 1)]
                 or [{"rango": "", "config_id": decision["configuracion_final_seleccionada"], "delta_ap_medio": "", "orden_predefinido": "", "seleccionada": True,
                      "criterio_desempate": "sin elegibles: se conserva la base"}])
    escribir_tabla(salida / "T-G4-7_seleccion_final.csv", filas_sel, list(filas_sel[0]), (extra + " | " if extra else "") + C.REGLA_SELECCION)
    for s, c in cont.items():
        BS.guardar_indices(salida / f"indices_b_estratificado_s{s}.npz", c, semilla=s, estratificado=True, nombres_de_grupo=nombres)
    evidencia = None
    if modo == "fase_b":
        evidencia = {"esquema": AGT.ESQUEMA_EVIDENCIA, "protocolo_id": C.PROTOCOLO_ID, "leyenda": C.LEYENDA, "git_commit_head": inicio["git_commit_head"],
                     "fase_a_completa": info["fase_a_completa"], "configuraciones": [
                         {"config_id": c["x"], "familia": next(e["familia"] for e in estr if e["id"] == c["x"]), "semillas": c["semillas"],
                          "cumple_a": bool(c["nivel_evidencia"]["cumple_a"]), "delta_ap_medio": float(c["delta_ap"]["estimacion"]), "nivel": c["nivel_evidencia"]["nivel"],
                          "clasificacion_punto_operacion": c["clasificacion_punto_operacion"]["clasificacion"]} for c in res["comparaciones"] if c["y"] == BASE]}
        escribir_json_nuevo(salida / "evidencia_fase_b.json", json.loads(json.dumps(evidencia, default=_json_default)))
    resultados = {"modo": modo, "protocolo_id": C.PROTOCOLO_ID, "leyenda": C.LEYENDA, "leyendas_fijas": list(C.LEYENDAS_FIJAS),
                  "viabilidad_clinica": "no determinada", "s_star": C.S_STAR, "semillas": todas, "k_configuraciones_evaluadas": K, **meta_fase,
                  "prevalencia_desarrollo": prev, "vpp_nota": NOTA_VPP.format(pos=int(pun.y.sum()), n=int(pun.y.size), prev=prev),
                  "decision_base": decision, "configuracion_final_seleccionada": decision["configuracion_final_seleccionada"], "agente": agente, "candidata_convencional": conv_info, "presupuesto": {k: v for k, v in rl.items()},
                  "maximo_propuestas_del_agente": maximo, "resultados": res}
    escribir_json_nuevo(salida / "resultados_g4.json", json.loads(json.dumps(resultados, default=_json_default)))
    fin = verificar()
    dif = comparar_instantaneas(inicio, fin)
    meta = {"protocolo": C.PROTOCOLO_ID, "modo": modo, "leyenda": C.LEYENDA, "fecha": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit_head": inicio["git_commit_head"], "rama": inicio["rama"], "sha256_bytes_archivo_particion": particion.sha256_bytes_archivo,
            "sha256_bytes_archivo_protocolo": sha_protocolo, "base_r6": r6, "comparabilidad_con_base": comp_base, "indices_b_verificacion": indices,
            "n_remuestreos": n_b, "numpy_version": np.__version__, "tiempos": tiempos, "k": K, "diferencias_inicio_fin": dif,
            "invalida_integridad": bool(dif) or bool(fin["bloqueos"]), "reporte_aceptado": not (bool(dif) or bool(fin["bloqueos"])),
            "sha256_bytes_archivo": {p.name: sha256_archivo(p) for p in sorted(salida.iterdir()) if p.suffix in (".csv", ".json", ".npz")}}
    escribir_json_nuevo(salida / "reporte_metadatos.json", json.loads(json.dumps(meta, default=_json_default)))
    return {**meta, "resultados": resultados, "dir": salida, "evidencia": evidencia}


def main(argv=None, *, root=None, config=None, config_g2=None, verificar=None, referencia=None, n_remuestreos=None) -> int:
    ap = argparse.ArgumentParser(description="Reporte G4 (solo desarrollo; partición provisional; sin interpretación)")
    ap.add_argument("--modo", required=True, choices=MODOS)
    ap.add_argument("--hasta", type=int, default=None, help="fase_a: configuraciones del orden fijo a reportar (todas pedidas deben estar completas)")
    ap.add_argument("--etiqueta", default=None, help="sufijo del directorio (el reporte no sobrescribe: un directorio nuevo por ejecución)")
    args = ap.parse_args(argv)
    root = Path(root or C.RAIZ_REPO).resolve()
    try:
        config = config or C.cargar_config_g4(root / C.CONFIG_REL)
        config_g2 = config_g2 or cargar_config_g2(root / config["referencia_g2"]["config"])
        verificar = verificar or (lambda: C.verificar_integridad_g4(root, config))
        particion = cargar_particion(root, config_g2)
        meta = generar_reporte(root, config, config_g2, particion, modo=args.modo, hasta=args.hasta, etiqueta=args.etiqueta, verificar=verificar,
                               referencia=referencia, n_remuestreos=n_remuestreos)
        print(json.dumps({"reporte_aceptado": meta["reporte_aceptado"], "dir": str(meta["dir"].relative_to(root)), "tiempos": meta["tiempos"]["estimacion"]},
                         ensure_ascii=False, default=_json_default))
        if meta["invalida_integridad"]:
            print("REPORTE NO ACEPTADO: la integridad final difiere de la inicial o tiene bloqueos; no debe usarse.", file=sys.stderr)
            return CODIGO_SALIDA_INTEGRIDAD_INVALIDA
        return 0
    except GuardError as exc:
        print(f"DETENIDO: {exc}", file=sys.stderr)
        return CODIGO_SALIDA_GUARDA


if __name__ == "__main__":
    sys.exit(main())
