"""Reporte descriptivo entre condiciones A, B (B_hibrida / B_oof) y C a partir de predicciones guardadas (solo desarrollo; sin interpretación).

* Solo se genera si NO falta ninguna corrida de lo solicitado (condición x arquitectura x semilla x fold): nunca métricas parciales.
* A se lee de G2 en solo lectura con `r4_v2.agregacion` (45/45 corridas). B y C se leen igual (misma estructura de corrida).
* Bootstrap agrupado por `group_id`, estratificado por clase del grupo, B = `bootstrap.n_remuestreos` de G2, con los MISMOS índices `b` de G2: se
  regeneran con `r4_v2.bootstrap.generar_conteos` (semillas de la configuración de G2) y, si existen los `indices_b_*.npz` de G2, se verifica igualdad.
* ΔAP emparejado por semilla y por `b`; con una sola semilla (p. ej. C) el contraste es «condicional a la semilla 42».
* Salidas: T-B1 AP (y secundarias) por condición x arquitectura, T-B2 ΔAP (+ identidad C-A = (C-B) + (B-A), descriptiva), T-B3 segmentación por fold
  (G3a, si existe), T-B4 trazabilidad, F-B1 ΔAP con IC95. Cada tabla lleva el encabezado y la leyenda. Sin texto interpretativo.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from src.r4_v2 import agregacion as AG
from src.r4_v2 import bootstrap as BS
from src.r4_v2 import metricas as M
from src.r4_v2.common import cargar_config as cargar_config_g2
from src.r4_v2.common import directorio_nuevo, escribir_json_nuevo
from src.r4_v2.particion import Particion, cargar_particion

from . import g3b_comun as G
from .comun import cargar_config_g3a, comparar_instantaneas

GuardError = G.GuardError
CODIGO_SALIDA_GUARDA, CODIGO_SALIDA_INTEGRIDAD_INVALIDA = 3, 4
ENCABEZADO = f"# mapeo v2 | phash_le8_v1 (provisional) | {G.PROTOCOLO_ID} | sin uso de test | {G.LEYENDA}"
ORDEN_CONDICIONES = ("B_hibrida", "B_oof", "C")


def f(x, nd=6):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{float(x):.{nd}f}"


def escribir_tabla(path, filas, columnas) -> None:
    with open(path, "x", encoding="utf-8", newline="") as fh:
        fh.write(ENCABEZADO + "\n")
        w = csv.DictWriter(fh, fieldnames=columnas, lineterminator="\n")
        w.writeheader()
        w.writerows(filas)


def etiqueta_semillas(semillas, semillas_g2) -> str:
    s = sorted(semillas)
    if set(s) == set(semillas_g2):
        return f"media de las semillas {s}"
    return f"condicional a la semilla {s[0]}" if len(s) == 1 else f"condicional a las semillas {s}"


# ---------------------------------------------------------------- carga (solo si no falta ninguna corrida)
def cargar_condicion(root, config, config_g2, particion: Particion, cond: str, archs, sems):
    g2d = G.config_g2_derivada(config_g2, config, cond, archs, sems)
    raiz = G.raiz_condicion(root, config, cond, "completa")
    plan = G.plan_de_corridas(g2d)
    faltan = [p["id"] for p in plan if G.corrida_completada(raiz / p["id"]) is None]
    if faltan:
        raise GuardError(f"faltan {len(faltan)} de {len(plan)} corridas de {cond} (p. ej. {faltan[:3]}): no se generan métricas parciales")
    corridas = AG.cargar_corridas(raiz, g2d, particion)                                   # K7 + comparabilidad interna
    huella = G.huella_config_efectiva(config, cond)
    manifiestos = set()
    for (a, s, k), (d, rc) in corridas.items():
        pr_path = Path(d) / "procedencia_g3b.json"
        if rc.get("condicion") != cond or not pr_path.is_file():
            raise GuardError(f"{rc.get('id_corrida')}: no es una corrida de {cond} con procedencia_g3b.json")
        pr = json.loads(pr_path.read_text(encoding="utf-8"))
        if pr["condicion"] != cond or pr["huella_config_efectiva"] != huella or pr["fold"] != k:
            raise GuardError(f"{rc['id_corrida']}: la procedencia no corresponde a la condición/configuración efectiva vigente")
        manifiestos.add(pr["sha256_bytes_manifiesto_recortes"])
    if len(manifiestos) != 1:
        raise GuardError(f"las corridas de {cond} usan más de un manifiesto de recortes")
    return AG.ensamblar_oof(corridas, particion, g2d), corridas, g2d


def cargar_A(root, config_g2, particion: Particion):
    raiz = Path(root) / config_g2["diseno"]["raiz_resultados"].strip("/") / "completa"
    corridas = AG.cargar_corridas(raiz, config_g2, particion)                             # 45/45 o falla
    return AG.ensamblar_oof(corridas, particion, config_g2), corridas


def exigir_comparables(corridas_A, corridas_c, cond: str) -> None:
    ka = {G.clave_comparabilidad(rc) for _, rc in corridas_A.values()}
    kc = {G.clave_comparabilidad(rc) for _, rc in corridas_c.values()}
    if len(ka) != 1 or kc != ka:
        raise GuardError(f"{cond} no es comparable con A: difieren partición, receta numérica o referencia de pesos (la condición es la única variable declarada)")


# ---------------------------------------------------------------- bootstrap y contrastes (extiende la lógica de r4_v2.bootstrap entre condiciones)
def verificar_indices_g2(root, config, config_g2, conteos_por_semilla: dict, nombres) -> list:
    """Compara los índices regenerados con los `indices_b_estratificado*_s<semilla>.npz` de G2 si existen; discrepancia = GuardError; ausentes = [ND]."""
    base = Path(root) / config["referencia_g2"]["indices_b_dir"].strip("/")
    registros = []
    for etiqueta, semilla in (("estratificado", config_g2["bootstrap"]["semilla"]), ("estratificado_semilla_comprobacion", config_g2["bootstrap"]["semilla_de_comprobacion"])):
        ruta = base / f"indices_b_{etiqueta}_s{semilla}.npz"
        c = conteos_por_semilla[semilla]
        if not ruta.is_file():
            registros.append({"etiqueta": etiqueta, "semilla": semilla, "estado": "[ND] no existe el .npz de G2 en este clon", "ruta": ruta.name})
            continue
        z = BS.cargar_indices(ruta)
        if z["conteos"].shape != c.shape or not np.array_equal(z["conteos"], c) or not np.array_equal(np.asarray(z["grupos"]).astype(str), np.asarray(nombres).astype(str)):
            raise GuardError(f"los índices b regenerados (semilla {semilla}) NO coinciden con {ruta.name} de G2; sin índices idénticos el emparejamiento con A no es válido")
        registros.append({"etiqueta": etiqueta, "semilla": semilla, "estado": "verificado: igualdad exacta con el .npz de G2", "ruta": ruta.name,
                          "numpy_g2": str(z["numpy_version"]), "numpy_actual": np.__version__})
    return registros


def media_dist(dist, cond, arq, semillas, metrica) -> np.ndarray:
    return np.mean([dist[(cond, arq, s)][metrica] for s in semillas], axis=0)


def media_puntual(puntual, cond, arq, semillas, metrica) -> float:
    return float(np.mean([puntual[(cond, arq, s)][metrica] for s in semillas]))


def delta_ap(dist, puntual, x, y, arq, semillas, nivel=0.95) -> dict:
    d = media_dist(dist, x, arq, semillas, "ap") - media_dist(dist, y, arq, semillas, "ap")
    inf, sup, n_ok = BS.percentiles(d, nivel)
    return {"estimacion": media_puntual(puntual, x, arq, semillas, "ap") - media_puntual(puntual, y, arq, semillas, "ap"),
            "ic_inf": inf, "ic_sup": sup, "n_remuestreos_validos": n_ok, "ic_excluye_cero": bool(inf > 0 or sup < 0), "d": d}


def analizar_condiciones(oofs: dict, semillas_c: dict, archs, semillas_g2, conteos, gpi, nivel=0.95) -> dict:
    """`oofs`: {'A': OOF, cond: OOF}; `semillas_c`: {cond: [semillas]}. Mismo `b` para todas las claves (condición, arquitectura, semilla)."""
    s_a = set(semillas_g2)
    usadas = {c: sorted(set(s) & s_a) for c, s in semillas_c.items()}
    if any(len(u) != len(s) for u, s in zip(usadas.values(), semillas_c.values())):
        raise GuardError("hay semillas de una condición sin su homóloga A (G2)")
    y = oofs["A"].y
    preps = {}
    needed_a = sorted(set().union(*map(set, usadas.values())))
    for cond, oof in oofs.items():
        if not np.array_equal(oof.y, y) or oof.image_ids != oofs["A"].image_ids:
            raise GuardError(f"{cond}: orden de imágenes o etiquetas distintos de A")
        for a in archs:
            for s in (needed_a if cond == "A" else semillas_c[cond]):
                preps[(cond, a, s)] = M.preparar(y, oof.prob[(a, s)])
    dist = BS.distribuciones(preps, conteos, gpi)
    puntual = {k: M.calcular(p) for k, p in preps.items()}
    out = {"semillas": {c: usadas[c] for c in usadas}, "por_condicion": {}, "deltas": [], "identidad": [], "n_remuestreos": int(conteos.shape[0]),
           "remuestreos_sin_una_clase": int(np.isnan(next(iter(dist.values()))["ap"]).sum())}
    conj = [("A", s) for s in sorted({tuple(u) for u in usadas.values()})] + [(c, tuple(semillas_c[c])) for c in semillas_c]
    for cond, sems in conj:
        for a in archs:
            fila = {"condicion": cond, "arquitectura": a, "semillas": list(sems)}
            for m in ("ap", "roc_auc", "sensibilidad_05", "especificidad_05"):
                inf, sup, n_ok = BS.percentiles(media_dist(dist, cond, a, sems, m), nivel)
                fila[m] = {"estimacion": media_puntual(puntual, cond, a, sems, m), "ic_inf": inf, "ic_sup": sup, "n_remuestreos_validos": n_ok}
            out["por_condicion"].setdefault(cond, []).append(fila)
    pares = [(c, "A", False) for c in semillas_c] + [("C", b, True) for b in ("B_hibrida", "B_oof") if "C" in semillas_c and b in semillas_c]
    for x, y_, descriptivo in pares:
        base = set(semillas_c[x]) if y_ == "A" else set(semillas_c[x]) & set(semillas_c[y_])
        sems = sorted(base & s_a)
        if not sems:
            continue                                                                       # sin semillas comunes no hay contraste emparejado
        for a in archs:
            d = delta_ap(dist, puntual, x, y_, a, sems, nivel)
            d.pop("d")
            out["deltas"].append({"comparacion": f"{x}-{y_}", "arquitectura": a, "semillas": sems, "descriptivo": descriptivo, **d})
    if "C" in semillas_c and "B_hibrida" in semillas_c:
        comunes = sorted(set(semillas_c["C"]) & set(semillas_c["B_hibrida"]) & s_a)
        for a in archs:
            ca, cb, ba = (delta_ap(dist, puntual, "C", "A", a, comunes), delta_ap(dist, puntual, "C", "B_hibrida", a, comunes),
                          delta_ap(dist, puntual, "B_hibrida", "A", a, comunes))
            res = np.abs(ca["d"] - (cb["d"] + ba["d"]))
            out["identidad"].append({"arquitectura": a, "semillas": comunes, "C-A": ca["estimacion"], "C-B": cb["estimacion"], "B-A": ba["estimacion"],
                                     "residuo_puntual": abs(ca["estimacion"] - (cb["estimacion"] + ba["estimacion"])),
                                     "residuo_maximo_por_remuestreo": float(np.nanmax(res)) if np.isfinite(res).any() else float("nan")})
    return out


# ---------------------------------------------------------------- tablas
COLUMNAS_TB1 = ["condicion", "arquitectura", "semillas", "etiqueta_semillas", "ap", "ap_ic95_inf", "ap_ic95_sup", "roc_auc", "roc_auc_ic95_inf",
                "roc_auc_ic95_sup", "sensibilidad_0.5", "sensibilidad_0.5_ic95_inf", "sensibilidad_0.5_ic95_sup", "especificidad_0.5",
                "especificidad_0.5_ic95_inf", "especificidad_0.5_ic95_sup", "prevalencia"]


def tabla_tb1(res, semillas_g2, prevalencia) -> list:
    filas = []
    nombres = {"ap": "ap", "roc_auc": "roc_auc", "sensibilidad_05": "sensibilidad_0.5", "especificidad_05": "especificidad_0.5"}
    for cond in ("A", *[c for c in ORDEN_CONDICIONES if c in res["por_condicion"]]):
        for r in res["por_condicion"][cond]:
            fila = {"condicion": cond, "arquitectura": r["arquitectura"], "semillas": " ".join(map(str, r["semillas"])),
                    "etiqueta_semillas": etiqueta_semillas(r["semillas"], semillas_g2), "prevalencia": f(prevalencia)}
            for m, n in nombres.items():
                fila[n], fila[n + "_ic95_inf"], fila[n + "_ic95_sup"] = f(r[m]["estimacion"]), f(r[m]["ic_inf"]), f(r[m]["ic_sup"])
            filas.append(fila)
    return filas


def tabla_tb1_por_semilla(oofs, semillas_c, archs, semillas_g2) -> list:
    filas = []
    for cond, oof in oofs.items():
        sems = sorted({s for c in semillas_c.values() for s in c}) if cond == "A" else semillas_c[cond]
        for a in archs:
            for s in sems:
                m = M.metricas(oof.y, oof.prob[(a, s)])
                filas.append({"condicion": cond, "arquitectura": a, "semilla": s, "ap": f(m["ap"]), "roc_auc": f(m["roc_auc"]),
                              "sensibilidad_0.5": f(m["sensibilidad_05"]), "especificidad_0.5": f(m["especificidad_05"]), "fp": int(m["fp_05"]), "fn": int(m["fn_05"])})
    return filas


COLUMNAS_TB2 = ["comparacion", "descriptivo", "arquitectura", "semillas", "etiqueta_semillas", "delta_ap", "ic95_inf", "ic95_sup", "n_remuestreos_validos", "ic95_excluye_cero"]


def tabla_tb2(res, semillas_g2) -> list:
    return [{"comparacion": d["comparacion"], "descriptivo": d["descriptivo"], "arquitectura": d["arquitectura"], "semillas": " ".join(map(str, d["semillas"])),
             "etiqueta_semillas": etiqueta_semillas(d["semillas"], semillas_g2), "delta_ap": f(d["estimacion"]), "ic95_inf": f(d["ic_inf"]),
             "ic95_sup": f(d["ic_sup"]), "n_remuestreos_validos": d["n_remuestreos_validos"], "ic95_excluye_cero": d["ic_excluye_cero"]} for d in res["deltas"]]


def tabla_identidad(res) -> list:
    return [{"arquitectura": r["arquitectura"], "semillas": " ".join(map(str, r["semillas"])), "C-A": f(r["C-A"]), "C-B": f(r["C-B"]), "B-A": f(r["B-A"]),
             "residuo_puntual": f"{r['residuo_puntual']:.3e}", "residuo_maximo_por_remuestreo": f"{r['residuo_maximo_por_remuestreo']:.3e}"} for r in res["identidad"]]


def tabla_tb3(root, config_g3a, folds) -> list:
    """Dice/IoU/cobertura/fallback por fold desde `metricas_seg_por_imagen.csv` de G3a (solo lectura); [ND] si no existe."""
    ruta = Path(root) / config_g3a["diseno"]["raiz_resultados"].strip("/") / "completa" / "metricas_seg_por_imagen.csv"
    if not ruta.is_file():
        return [{"fold": "[ND]", "n": "", "dice_media": "", "iou_media": "", "cobertura_media_sin_fallback": "", "n_fallback_pred": "", "n_fallback_gt": "",
                 "n_pred_vacia": "", "nota": f"[ND] no existe {ruta.name} de G3a en este clon"}]
    with open(ruta, newline="", encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
    out = []
    for k in [*folds, None]:
        sel = [r for r in filas if k is None or int(r["fold"]) == k]
        cob = [float(r["cobertura"]) for r in sel if r["cobertura"] != "" and r["fallback_pred"] != "True"]
        out.append({"fold": "total" if k is None else k, "n": len(sel), "dice_media": f(np.mean([float(r["dice"]) for r in sel]) if sel else None),
                    "iou_media": f(np.mean([float(r["iou"]) for r in sel]) if sel else None), "cobertura_media_sin_fallback": f(np.mean(cob) if cob else None),
                    "n_fallback_pred": sum(r["fallback_pred"] == "True" for r in sel), "n_fallback_gt": sum(r["fallback_gt"] == "True" for r in sel),
                    "n_pred_vacia": sum(r["pred_vacia"] == "True" for r in sel), "nota": "G3a: descriptivo"})
    return out


COLUMNAS_TB4 = ["condicion", "modo", "id_corrida", "intento", "estado", "motivo", "git_commit_head", "sha256_bytes_archivo_checkpoint", "invalida_integridad",
                "segmentador_fold_k", "sha256_bytes_checkpoint_unet_fold_k", "sha256_bytes_manifiesto_recortes", "n_fallback_validacion", "comparabilidad_con_A"]


def tabla_tb4(root, config, condiciones) -> list:
    filas = []
    for cond in condiciones:
        for modo in ("humo", "completa"):
            base = G.raiz_condicion(root, config, cond, modo)
            if not base.exists():
                continue
            for intento in sorted(base.glob("*/intento_*")):
                rc = json.loads((intento / "run_config.json").read_text(encoding="utf-8")) if (intento / "run_config.json").is_file() else {}
                pr = json.loads((intento / "procedencia_g3b.json").read_text(encoding="utf-8")) if (intento / "procedencia_g3b.json").is_file() else {}
                sello = json.loads((intento / "sello_g3b.json").read_text(encoding="utf-8")) if (intento / "sello_g3b.json").is_file() else {}
                u = pr.get("unet_fold_k") or {}
                filas.append({"condicion": cond, "modo": modo, "id_corrida": intento.parent.name, "intento": intento.name, "estado": rc.get("estado", "fallo_tecnico"),
                              "motivo": rc.get("motivo") or "", "git_commit_head": rc.get("git_commit_head", ""),
                              "sha256_bytes_archivo_checkpoint": (rc.get("checkpoint") or {}).get("sha256_bytes_archivo", ""),
                              "invalida_integridad": rc.get("invalida_integridad", ""), "segmentador_fold_k": u.get("segmentador_fold_k") or "",
                              "sha256_bytes_checkpoint_unet_fold_k": u.get("sha256_bytes_checkpoint_unet_fold_k") or "",
                              "sha256_bytes_manifiesto_recortes": pr.get("sha256_bytes_manifiesto_recortes", ""),
                              "n_fallback_validacion": (pr.get("validacion") or {}).get("n_fallback", ""),
                              "comparabilidad_con_A": (sello.get("comparabilidad_con_A") or {}).get("estado", "")})
    return filas


def tabla_unet(g3a_info) -> list:
    return [{"fold": k, "segmentador": g3a_info["segmentadores"][k], "sha256_bytes_checkpoint": g3a_info["checkpoints"][k],
             "git_commit_head_g3a": g3a_info["git_commit_head_g3a"]} for k in sorted(g3a_info["checkpoints"])]


# ---------------------------------------------------------------- figura (matplotlib diferido; una figura por propósito)
def figura_delta(salida, res, archs, semillas_g2) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise GuardError("[ND] matplotlib no está instalado: las tablas se generaron, la figura no") from exc
    comps = [c for c in dict.fromkeys(d["comparacion"] for d in res["deltas"] if not d["descriptivo"])]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ancho = 0.8 / max(1, len(comps))
    for j, c in enumerate(comps):
        for i, a in enumerate(archs):
            d = next(x for x in res["deltas"] if x["comparacion"] == c and x["arquitectura"] == a)
            ax.errorbar([i + (j - (len(comps) - 1) / 2) * ancho], [d["estimacion"]], yerr=[[d["estimacion"] - d["ic_inf"]], [d["ic_sup"] - d["estimacion"]]],
                        fmt="o", capsize=4, c=f"C{j}", label=f"{c} ({etiqueta_semillas(d['semillas'], semillas_g2)})" if i == 0 else None)
    ax.axhline(0, c="gray", ls="--")
    ax.set_xticks(range(len(archs)), archs)
    ax.set_ylabel("ΔAP con IC95 (mismo remuestreo b)")
    ax.legend(fontsize=6)
    fig.text(0.01, 0.005, f"mapeo v2; phash_le8_v1 (provisional); {G.PROTOCOLO_ID}; sin uso de test; conformidad académica pendiente", fontsize=6)
    nombre = "F-B1_delta_ap.png"
    fig.savefig(Path(salida) / nombre, dpi=150, metadata={"Software": None})
    plt.close(fig)
    return nombre


# ---------------------------------------------------------------- orquestación
def generar_reporte(root, config, config_g2, config_g3a, particion: Particion, *, condiciones, arquitecturas=None, semillas_por_condicion=None,
                    verificar, salida=None, n_remuestreos=None, con_figuras=True, g3a_info=None) -> dict:
    """Genera el reporte SOLO si no falta ninguna corrida solicitada. `verificar()` se aplica antes de leer predicciones y al terminar."""
    inicio = G.exigir_integridad(verificar())
    salida = Path(salida) if salida else G.raiz_reporte(root, config)
    if salida.exists():
        raise GuardError(f"K5: el directorio del reporte ya existe y no se sobrescribe: {salida}")
    if not condiciones or len(set(condiciones)) != len(condiciones):
        raise GuardError("condiciones vacías o repetidas")
    archs = list(arquitecturas) if arquitecturas else list(config["diseno"]["arquitecturas"])
    semillas_c, oofs, corridas_c = {}, {}, {}
    for cond in condiciones:
        _, sems = G.seleccionar_diseno(config, config_g2, cond, archs, (semillas_por_condicion or {}).get(cond))
        semillas_c[cond] = sems
    for cond in condiciones:                                                              # 1) que no falte ninguna corrida, antes de leer métricas
        oofs[cond], corridas_c[cond], _ = cargar_condicion(root, config, config_g2, particion, cond, archs, semillas_c[cond])
    oofs["A"], corridas_A = cargar_A(root, config_g2, particion)
    for cond in condiciones:
        exigir_comparables(corridas_A, corridas_c[cond], cond)
    oofs = {"A": oofs["A"], **{c: oofs[c] for c in condiciones}}
    sg2 = config_g2["diseno"]["semillas"]
    gpi, clase, nombres = BS.indexar_grupos(oofs["A"].group_ids, oofs["A"].y)
    n_b = int(n_remuestreos or config_g2["bootstrap"]["n_remuestreos"])
    cfg_b = config_g2["bootstrap"]
    conteos = {s: BS.generar_conteos(clase, n_b, s, estratificado=True) for s in (cfg_b["semilla"], cfg_b["semilla_de_comprobacion"])}
    indices = verificar_indices_g2(root, config, config_g2, conteos, nombres) if n_remuestreos is None else \
        [{"estado": "no verificado: n_remuestreos reducido (solo pruebas)"}]
    res = analizar_condiciones(oofs, semillas_c, archs, sg2, conteos[cfg_b["semilla"]], gpi, nivel=cfg_b["intervalo"]["nivel"])
    salida = directorio_nuevo(salida)                                                     # K5: solo tras validar y calcular
    escribir_tabla(salida / "T-B1_ap_por_condicion_arquitectura.csv", tabla_tb1(res, sg2, float(oofs["A"].y.mean())), COLUMNAS_TB1)
    t1s = tabla_tb1_por_semilla(oofs, semillas_c, archs, sg2)
    escribir_tabla(salida / "T-B1_por_semilla.csv", t1s, list(t1s[0]))
    escribir_tabla(salida / "T-B2_delta_ap.csv", tabla_tb2(res, sg2), COLUMNAS_TB2)
    ident = tabla_identidad(res)
    if ident:
        escribir_tabla(salida / "T-B2_identidad_C-A.csv", ident, list(ident[0]))
    t3 = tabla_tb3(root, config_g3a, config["diseno"]["folds"])
    escribir_tabla(salida / "T-B3_segmentacion_por_fold.csv", t3, list(t3[0]))
    t4 = tabla_tb4(root, config, condiciones)
    escribir_tabla(salida / "T-B4_trazabilidad.csv", t4, COLUMNAS_TB4)
    if g3a_info is not None:
        tu = tabla_unet(g3a_info)
        escribir_tabla(salida / "T-B4_unet_por_fold.csv", tu, list(tu[0]))
    for s, c in conteos.items():
        BS.guardar_indices(salida / f"indices_b_estratificado_s{s}.npz", c, semilla=s, estratificado=True, nombres_de_grupo=nombres)
    figura, nota_fig = None, None
    if con_figuras:
        try:
            figura = figura_delta(salida, res, archs, sg2)
        except GuardError as exc:
            nota_fig = str(exc)
    fin = verificar()
    dif = comparar_instantaneas(inicio, fin)
    resumen = {k: v for k, v in res.items()}
    escribir_json_nuevo(salida / "bootstrap_resumen.json", json.loads(json.dumps(resumen, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))))
    meta = {"protocolo": G.PROTOCOLO_ID, "leyenda": G.LEYENDA, "git_commit_head": inicio["git_commit_head"], "rama": inicio["rama"],
            "condiciones": list(condiciones), "arquitecturas": archs, "semillas": semillas_c, "n_remuestreos": n_b,
            "indices_b_verificacion": indices, "numpy_version": np.__version__,
            "commits_corridas": {c: sorted({rc["git_commit_head"] for _, rc in corridas_c[c].values()}) for c in condiciones},
            "commits_corridas_A": sorted({rc["git_commit_head"] for _, rc in corridas_A.values()}),
            "sha256_bytes_archivo_particion": particion.sha256_bytes_archivo, "figura": figura, "nota_figura": nota_fig,
            "diferencias_inicio_fin": dif, "invalida_integridad": bool(dif) or bool(fin["bloqueos"]),
            "reporte_aceptado": not (bool(dif) or bool(fin["bloqueos"])),
            "sha256_bytes_archivo": {p.name: G.sha256_archivo(p) for p in sorted(salida.iterdir()) if p.suffix in (".csv", ".json", ".npz", ".png")}}
    escribir_json_nuevo(salida / "reporte_condiciones_metadatos.json", meta)
    return {**meta, "resultados": res}


def _parsear_semillas(pares) -> dict:
    out = {}
    for p in pares or []:
        try:
            c, v = p.split("=")
            out[c] = [int(x) for x in v.split(",")]
        except ValueError:
            raise GuardError(f"--semillas debe ser COND=42,43,44 (recibido {p!r})") from None
    return out


def main(argv=None, *, root=None, verificar=None) -> int:
    ap = argparse.ArgumentParser(description="Reporte entre condiciones A/B/C de G3b (solo desarrollo; partición provisional)")
    ap.add_argument("--condiciones", nargs="+", default=["B_hibrida", "C"], choices=sorted(G.SEMANTICA))
    ap.add_argument("--arquitecturas", nargs="+", default=None)
    ap.add_argument("--semillas", nargs="*", default=None, help="COND=42,43,44 (por defecto las de la configuración)")
    ap.add_argument("--sin-figuras", action="store_true")
    args = ap.parse_args(argv)
    root = Path(root or G.RAIZ_REPO).resolve()
    try:
        config = G.cargar_config_g3b(root / G.CONFIG_REL)
        config_g2 = cargar_config_g2(root / config["referencia_g2"]["config"])
        config_g3a = cargar_config_g3a(root / config["referencia_g3a"]["config"])
        G.comprobar_coherencia(config, config_g2, config_g3a)
        verificar = verificar or (lambda: G.verificar_integridad_g3b(root, config))
        particion = cargar_particion(root, config_g2)
        try:
            g3a_info = G.verificar_consolidacion_g3a(root, config_g3a, particion)
        except GuardError:
            g3a_info = None                                                                  # T-B3/T-B4 de U-Net quedan [ND]; no bloquea el reporte
        meta = generar_reporte(root, config, config_g2, config_g3a, particion, condiciones=args.condiciones, arquitecturas=args.arquitecturas,
                               semillas_por_condicion=_parsear_semillas(args.semillas), verificar=verificar, con_figuras=not args.sin_figuras, g3a_info=g3a_info)
        print(json.dumps({"reporte_aceptado": meta["reporte_aceptado"], "figura": meta["figura"], "nota_figura": meta["nota_figura"]}, ensure_ascii=False))
        if meta["invalida_integridad"]:
            print("REPORTE NO ACEPTADO: la integridad final difiere de la inicial o tiene bloqueos; no debe usarse.", file=sys.stderr)
            return CODIGO_SALIDA_INTEGRIDAD_INVALIDA
        return 0
    except GuardError as exc:
        print(f"DETENIDO: {exc}", file=sys.stderr)
        return CODIGO_SALIDA_GUARDA


if __name__ == "__main__":
    sys.exit(main())
