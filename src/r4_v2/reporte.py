"""Reporte determinista T1-T6 y F1-F4 (protocolo §9) generado por script desde las predicciones guardadas.

La lógica de las tablas solo usa numpy y la biblioteca estándar; las figuras (una por propósito) usan matplotlib con
importación diferida. Cada tabla lleva una primera línea de comentario con los identificadores y la leyenda obligatoria.
No usa test ni las 180 normales, no calcula ningún umbral y no interpreta resultados (la interpretación la escribe el autor
después de disponer de resultados reales).

Uso:  python -B -X pycache_prefix=<dir> -m src.r4_v2.reporte --salida results/r4_v2/phash_le8_v1/g2_baseline_v1/reporte

Códigos de salida de `main`: 0 = reporte generado y ACEPTADO; 3 = detenido por una guarda (K5, K6, K7, partición...; no se genera
nada); 4 = el reporte se generó pero la verificación de integridad §4.5 FINAL no pasó o difiere de la inicial
(`invalida_integridad`): los archivos se conservan, `reporte_metadatos.json` dice `reporte_aceptado: false` y NO debe usarse.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from . import agregacion as AG
from . import bootstrap as BS
from . import integridad as I
from . import metricas as M
from .common import (CONFIG_REL, GuardError, LEYENDA_PROVISIONAL, MAPEO_ID, PARTICION_ID, PROTOCOLO_ID, RAIZ_REPO,
                     cargar_config, directorio_nuevo, escribir_json_nuevo, sha256_archivo)
from .particion import Particion, cargar_particion

CODIGO_SALIDA_GUARDA = 3
CODIGO_SALIDA_INTEGRIDAD_INVALIDA = 4

ENCABEZADO = (f"# {MAPEO_ID} | {PARTICION_ID} (provisional) | {PROTOCOLO_ID} | 45 corridas | sin uso de test | "
              f"{LEYENDA_PROVISIONAL}")


def f(x, nd=6):
    """Formato fijo (determinista); NaN y None se escriben vacíos."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{float(x):.{nd}f}"


def escribir_tabla(path, filas, columnas) -> None:
    with open(path, "x", encoding="utf-8", newline="") as fh:
        fh.write(ENCABEZADO + "\n")
        w = csv.DictWriter(fh, fieldnames=columnas, lineterminator="\n")
        w.writeheader()
        w.writerows(filas)


# ---------------------------------------------------------------- tablas
def tabla_t1(part: Particion, config) -> list:
    filas = []
    for k in config["diseno"]["folds"] + [None]:
        sel = [r for r in part.filas if k is None or r["fold"] == k]
        grupos = {}
        for r in sel:
            grupos.setdefault(r["group_id"], 0)
            grupos[r["group_id"]] += 1
        filas.append({"fold": "total" if k is None else k, "n": len(sel), "positivas": sum(r["label_v2"] for r in sel),
                      "grupos": len(grupos), "grupo_mayor": max(grupos.values())})
    return filas


def tabla_t2(oof, por_semilla, boot, config) -> list:
    d = config["diseno"]
    filas = []
    est = boot["estratificado"]["por_arquitectura"]
    for a in d["arquitecturas"]:
        def media(m):
            return float(np.mean([por_semilla[(a, s)][m] for s in d["semillas"]]))
        fila = {"arquitectura": a, "prevalencia": f(oof.y.mean())}
        for m, nombre in (("ap", "ap"), ("roc_auc", "roc_auc"), ("sensibilidad_05", "sensibilidad_0.5"),
                          ("especificidad_05", "especificidad_0.5")):
            fila[nombre] = f(est[a][m]["estimacion"])
            fila[nombre + "_ic95_inf"], fila[nombre + "_ic95_sup"] = f(est[a][m]["ic_inf"]), f(est[a][m]["ic_sup"])
        fila["fp_media"] = f(media("fp_05"), 2)
        fila["fn_media"] = f(media("fn_05"), 2)
        for nivel in M.NIVELES_ESPECIFICIDAD:         # puntos DESCRIPTIVOS de la ROC; no son S* ni un umbral
            fila[f"sensibilidad_en_especificidad_{nivel:.2f}"] = f(media(f"sensibilidad_esp_{nivel:.2f}"))
        filas.append(fila)
    return filas


def tabla_t3(oof, por_semilla, por_fold, config) -> tuple:
    d = config["diseno"]
    semillas = [{"arquitectura": a, "semilla": s, "ap": f(por_semilla[(a, s)]["ap"]), "roc_auc": f(por_semilla[(a, s)]["roc_auc"]),
                 "sensibilidad_0.5": f(por_semilla[(a, s)]["sensibilidad_05"]),
                 "especificidad_0.5": f(por_semilla[(a, s)]["especificidad_05"]), "fp": int(por_semilla[(a, s)]["fp_05"]),
                 "fn": int(por_semilla[(a, s)]["fn_05"])} for a in d["arquitecturas"] for s in d["semillas"]]
    folds = [{"arquitectura": a, "semilla": s, "fold": k, "n_positivas": por_fold[(a, s, k)][1], "ap": f(por_fold[(a, s, k)][0])}
             for a in d["arquitecturas"] for s in d["semillas"] for k in d["folds"]]
    return semillas, folds


def tabla_t4(boot, config) -> list:
    filas = []
    est, nes = boot["estratificado"]["deltas"], boot["no_estratificado"]["deltas"]
    for par, ms in est.items():
        for m, v in ms.items():
            w = nes[par][m]
            ambos = v["ic_excluye_cero"] and w["ic_excluye_cero"]
            filas.append({"comparacion": par, "metrica": m, "delta": f(v["estimacion"]), "ic95_inf": f(v["ic_inf"]),
                          "ic95_sup": f(v["ic_sup"]), "no_estratificado_ic95_inf": f(w["ic_inf"]),
                          "no_estratificado_ic95_sup": f(w["ic_sup"]),
                          "lectura": "diferencia observada en AP" if (m == "ap" and ambos) else "no concluyente"})
    return filas


def tabla_t5(oof, boot, ap_alt, por_semilla, por_fold, ens, config) -> list:
    d = config["diseno"]
    filas = []
    for a in d["arquitecturas"]:
        n = boot["no_estratificado"]["por_arquitectura"][a]["ap"]
        filas.append({"arquitectura": a, "sensibilidad": "bootstrap_no_estratificado_ap", "valor": f(n["estimacion"]),
                      "detalle": f"IC95 [{f(n['ic_inf'])}, {f(n['ic_sup'])}]; remuestreos validos {n['n_remuestreos_validos']}"})
        for alt, ap in ap_alt.items():
            filas.append({"arquitectura": a, "sensibilidad": alt, "valor": f(ap[a]), "detalle": "AP agregado; media entre semillas (§6.3)"})
        filas.append({"arquitectura": a, "sensibilidad": "ap_por_fold_media_entre_semillas", "valor": "",
                      "detalle": "; ".join(f"fold {k}: {f(np.mean([por_fold[(a, s, k)][0] for s in d['semillas']]))}"
                                           f" (n+={por_fold[(a, d['semillas'][0], k)][1]})" for k in d["folds"])})
        filas.append({"arquitectura": a, "sensibilidad": "ap_ensamble_de_semillas", "valor": f(ens[a]["agregado"]),
                      "detalle": "; ".join(f"fold {k}: {f(v)}" for k, v in ens[a]["por_fold"].items())})
    filas.append({"arquitectura": "todas", "sensibilidad": "bootstrap_segunda_semilla", "valor": f(boot["diferencia_maxima_extremos_ic_entre_semillas"]),
                  "detalle": "diferencia máxima entre extremos del IC (se reporta; sin umbral de aceptación)"})
    filas.append({"arquitectura": "todas", "sensibilidad": "remuestreos_sin_una_clase",
                  "valor": boot["estratificado"]["remuestreos_sin_una_clase"],
                  "detalle": f"no estratificado: {boot['no_estratificado']['remuestreos_sin_una_clase']} de {boot['n_remuestreos']}"})
    return filas


def tabla_t6(raiz_resultados) -> list:
    """Todos los intentos (referencia, humo y completa), incluidos los fallidos; humo/referencia solo figuran como intentos."""
    filas = []
    for modo in ("referencia", "humo", "completa"):
        base = Path(raiz_resultados) / modo
        if not base.exists():
            continue
        for intento in sorted(base.glob("intento_*")) + sorted(base.glob("*/intento_*")):
            rc_path = intento / "run_config.json"
            rc = json.loads(rc_path.read_text(encoding="utf-8")) if rc_path.is_file() else {}
            sel, rec, ck = rc.get("seleccion") or {}, rc.get("recarga") or {}, rc.get("checkpoint") or {}
            filas.append({"modo": modo, "id_corrida": rc.get("id_corrida", ""), "intento": intento.name,
                          "estado": rc.get("estado", "fallo_tecnico"), "motivo": rc.get("motivo") or "",
                          "epoca_elegida": sel.get("epoca_elegida", ""), "epocas_entrenadas": sel.get("epocas_entrenadas", ""),
                          "sha256_bytes_archivo_checkpoint": ck.get("sha256_bytes_archivo", ""),
                          "dif_recarga_max": f(rec.get("diferencia_maxima_absoluta")),
                          "cambios_de_clase_recarga": rec.get("cambios_de_clase_a_0_5", ""),
                          "alerta_recarga": rc.get("alerta_recarga", ""), "invalida_integridad": rc.get("invalida_integridad", ""),
                          "git_commit_head": rc.get("git_commit_head", "")})
    return filas


# ---------------------------------------------------------------- figuras (matplotlib diferido; una figura por propósito)
def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise GuardError("[ND] matplotlib no está instalado: las tablas se generaron, las figuras no") from exc
    return plt


def figuras(salida, oof, por_semilla, boot, por_fold, config) -> list:
    plt = _plt()
    d = config["diseno"]
    archs, semillas, folds = d["arquitecturas"], d["semillas"], d["folds"]
    est = boot["estratificado"]["por_arquitectura"]
    pie = f"{MAPEO_ID}; {PARTICION_ID} (provisional); {PROTOCOLO_ID}; sin uso de test; conformidad académica pendiente"
    hechos = []

    def guardar(fig, nombre):
        ruta = Path(salida) / nombre
        fig.text(0.01, 0.005, pie, fontsize=6)
        fig.savefig(ruta, dpi=150, metadata={"Software": None})
        plt.close(fig)
        hechos.append(nombre)

    fig, ax = plt.subplots(figsize=(6, 4))                                              # F1
    x = np.arange(len(archs))
    ap = [est[a]["ap"]["estimacion"] for a in archs]
    err = [[est[a]["ap"]["estimacion"] - est[a]["ap"]["ic_inf"] for a in archs], [est[a]["ap"]["ic_sup"] - est[a]["ap"]["estimacion"] for a in archs]]
    ax.errorbar(x, ap, yerr=err, fmt="o", capsize=4, label="media entre semillas e IC95 (bootstrap agrupado)")
    for i, a in enumerate(archs):
        ax.scatter([i] * len(semillas), [por_semilla[(a, s)]["ap"] for s in semillas], marker="x", c="k", label="semilla" if i == 0 else None)
    ax.axhline(oof.y.mean(), ls="--", c="gray", label="prevalencia")
    ax.set_xticks(x, archs)
    ax.set_ylabel("AP OOF (820 predicciones)")
    ax.legend(fontsize=7)
    guardar(fig, "F1_ap_por_arquitectura.png")

    fig, ax = plt.subplots(figsize=(6, 3.5))                                            # F2
    pares = list(boot["estratificado"]["deltas"])
    for i, par in enumerate(pares):
        v = boot["estratificado"]["deltas"][par]["ap"]
        ax.errorbar([v["estimacion"]], [i], xerr=[[v["estimacion"] - v["ic_inf"]], [v["ic_sup"] - v["estimacion"]]], fmt="o", capsize=4)
    ax.axvline(0, c="gray", ls="--")
    ax.set_yticks(range(len(pares)), [p.replace("__menos__", " - ") for p in pares])
    ax.set_xlabel("ΔAP con IC95 (mismo remuestreo b)")
    guardar(fig, "F2_delta_ap.png")

    fig, ax = plt.subplots(figsize=(6, 4))                                              # F3 (sin barras de error)
    for a in archs:
        ax.plot(folds, [np.mean([por_fold[(a, s, k)][0] for s in semillas]) for k in folds], marker="o", label=a)
    ax.set_xlabel("fold")
    ax.set_ylabel("AP (media entre semillas; 7-8 positivas por fold)")
    ax.legend(fontsize=7)
    guardar(fig, "F3_ap_por_fold.png")

    fig, ax = plt.subplots(figsize=(6, 4))                                              # F4
    colores = dict(zip(archs, ("C0", "C1", "C2")))
    for a in archs:
        for s in semillas:
            r, p = M.curva_pr(oof.y, oof.prob[(a, s)])
            ax.step(r, p, where="post", c=colores[a], alpha=0.7, label=a if s == semillas[0] else None)
    ax.axhline(oof.y.mean(), ls="--", c="gray")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.legend(fontsize=7)
    guardar(fig, "F4_curva_pr_agregada.png")
    return hechos


# ---------------------------------------------------------------- orquestación
def generar_reporte(root, config, particion: Particion, raiz_resultados, salida, *, verificar, n_remuestreos=None,
                    con_figuras=True) -> dict:
    """Genera T1-T6 y F1-F4. `verificar()` (obligatorio, sin valor por defecto) devuelve la verificación §4.5 y se aplica ANTES de
    leer predicciones o escribir nada (protocolo §3: todo resultado lleva el commit del código); se repite al terminar."""
    inicio = verificar()
    I.exigir_integridad(inicio)                                                          # K6 antes de generar el reporte
    if Path(salida).exists():
        raise GuardError(f"K5: el directorio del reporte ya existe y no se sobrescribe: {salida}")
    corridas = AG.cargar_corridas(Path(raiz_resultados) / "completa", config, particion)
    oof = AG.ensamblar_oof(corridas, particion, config)
    por_semilla = AG.metricas_por_semilla(oof, config)
    por_fold = AG.ap_por_fold(oof, config)
    ens = AG.ap_ensamble_por_fold(oof, config)
    ap_alt = AG.ap_sensibilidad_checkpoint(corridas, particion, config)
    boot = AG.ejecutar_bootstrap(oof, config, n_remuestreos=n_remuestreos)
    salida = directorio_nuevo(salida)                                                   # K5: solo tras validar K7 y calcular

    escribir_tabla(salida / "T1_particion.csv", tabla_t1(particion, config), ["fold", "n", "positivas", "grupos", "grupo_mayor"])
    t2 = tabla_t2(oof, por_semilla, boot, config)
    escribir_tabla(salida / "T2_arquitecturas.csv", t2, list(t2[0]))
    sem, fol = tabla_t3(oof, por_semilla, por_fold, config)
    escribir_tabla(salida / "T3_por_semilla.csv", sem, list(sem[0]))
    escribir_tabla(salida / "T3_ap_por_fold.csv", fol, list(fol[0]))
    t4 = tabla_t4(boot, config)
    escribir_tabla(salida / "T4_delta_ap.csv", t4, list(t4[0]))
    t5 = tabla_t5(oof, boot, ap_alt, por_semilla, por_fold, ens, config)
    escribir_tabla(salida / "T5_sensibilidades.csv", t5, ["arquitectura", "sensibilidad", "valor", "detalle"])
    t6 = tabla_t6(raiz_resultados)
    escribir_tabla(salida / "T6_trazabilidad.csv", t6, list(t6[0]) if t6 else ["modo"])
    for etiqueta, (conteos, semilla, estrat) in boot["_conteos"].items():             # índices `b` reutilizables en G3/G4
        BS.guardar_indices(salida / f"indices_b_{etiqueta}_s{semilla}.npz", conteos, semilla=semilla, estratificado=estrat,
                           nombres_de_grupo=boot["_nombres_de_grupo"])
    resumen = {k: v for k, v in boot.items() if not k.startswith("_")}
    escribir_json_nuevo(salida / "bootstrap_resumen.json", resumen)
    figs = []
    nota_figuras = None
    if con_figuras:
        try:
            figs = figuras(salida, oof, por_semilla, boot, por_fold, config)
        except GuardError as exc:
            nota_figuras = str(exc)
    fin = verificar()
    diferencias = I.comparar_instantaneas(inicio, fin)
    commits_corridas = sorted({rc["git_commit_head"] for _, rc in corridas.values()})
    meta = {"protocolo": PROTOCOLO_ID, "mapeo": MAPEO_ID, "particion": PARTICION_ID, "leyenda": LEYENDA_PROVISIONAL,
            "git_commit_head": inicio["git_commit_head"], "rama": inicio["rama"],
            "protocolo_versionado": inicio["protocolo_versionado"], "entrada": inicio["entrada"],
            "integridad": {rel: {c: r[c] for c in ("git_blob_id", "sha256_contenido_blob", "sha256_bytes_archivo")}
                           for rel, r in sorted(inicio["archivos"].items())},
            "diferencias_inicio_fin": diferencias, "invalida_integridad": bool(diferencias) or bool(fin["bloqueos"]),
            "reporte_aceptado": not (bool(diferencias) or bool(fin["bloqueos"])),
            "git_commit_head_de_las_corridas": commits_corridas,
            "commits_de_las_corridas_difieren_del_reporte": commits_corridas != [inicio["git_commit_head"]],
            "sha256_bytes_archivo_particion": particion.sha256_bytes_archivo, "n_corridas": len(corridas),
            "figuras": figs, "nota_figuras": nota_figuras,
            "sha256_bytes_archivo": {p.name: sha256_archivo(p) for p in sorted(salida.iterdir()) if p.suffix in (".csv", ".json", ".npz")}}
    escribir_json_nuevo(salida / "reporte_metadatos.json", meta)
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Reporte T1-T6 y F1-F4 de G2 (solo desarrollo; partición provisional)")
    ap.add_argument("--salida", required=True, help="directorio NUEVO, relativo al repositorio, bajo results/r4_v2/...")
    ap.add_argument("--sin-figuras", action="store_true")
    ap.add_argument("--notebook", default=None, help="ruta relativa del notebook orquestador, si el reporte se lanzó desde uno")
    args = ap.parse_args(argv)
    try:
        config = cargar_config(RAIZ_REPO / CONFIG_REL)
        base = config["diseno"]["raiz_resultados"].strip("/")
        destino = (RAIZ_REPO / args.salida).resolve()
        if (RAIZ_REPO / base).resolve() not in destino.parents:
            raise GuardError(f"K5: la salida debe estar dentro de {base}/")
        def verificar():
            return I.verificar_integridad(RAIZ_REPO, config, modo="completa", notebook=args.notebook)

        meta = generar_reporte(RAIZ_REPO, config, cargar_particion(RAIZ_REPO, config), RAIZ_REPO / base, destino,
                               verificar=verificar, con_figuras=not args.sin_figuras)
        print(json.dumps({"salida": args.salida, "figuras": meta["figuras"], "nota_figuras": meta["nota_figuras"],
                          "reporte_aceptado": meta["reporte_aceptado"]}, ensure_ascii=False))
        if meta["invalida_integridad"]:
            print("REPORTE NO ACEPTADO: la verificación de integridad final (§4.5) falló o difiere de la inicial "
                  f"({meta['diferencias_inicio_fin'] or 'bloqueos finales'}). Los archivos se conservan en {args.salida}, "
                  "pero no deben usarse.", file=sys.stderr)
            return CODIGO_SALIDA_INTEGRIDAD_INVALIDA
        return 0
    except GuardError as exc:
        print(f"DETENIDO: {exc}", file=sys.stderr)
        return CODIGO_SALIDA_GUARDA


if __name__ == "__main__":
    sys.exit(main())
