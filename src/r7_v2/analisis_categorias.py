"""R7 `analisis_categorias_v1`: análisis DESCRIPTIVO por categoría original {0, 1, 2, 3, 4, 6, 7} sobre q_barra y el umbral de transferencia (solo desarrollo).

* Lee `q_barra_oof.csv` de `umbral_sistema_v1` verificando su SHA-256 contra `trazabilidad.json`, y `original_class`/`fold`/`group_id` de la partición (K1-K4).
* La categoría 5 (ovario normal) y cualquier clase fuera de {0, 1, 2, 3, 4, 6, 7} BLOQUEAN.
* Umbral por imagen = t̂ de su fold calculado con los OTROS cuatro folds (`calibrar_umbral`, S* = 0.90; transferencia). Para la categoría 7 la proporción por encima del umbral es la
  sensibilidad; para las demás, la tasa de falsos positivos (conteos x/n). AUC de cada categoría no maligna contra la 7 con IC95 % bootstrap agrupado (mismos `b` de G2) SOLO si n_c >= 10.
* Categoría 6: n, FP/n y diferencia de proporción frente al resto de no malignas, SIN p-valores. NO se reevalúa como positiva; no altera etiquetas, entrenamiento ni selección;
  R7.3 NO se declara cumplido (discrepancia de alcance). Sin independencia por paciente.

CLI: python -B -X pycache_prefix=<dir> -m src.r7_v2.analisis_categorias --autorizacion CONFIRMADO:D-NNN  (solo la autora, después de `umbral_sistema` y del commit de congelamiento).
Partición provisional (phash_le8_v1, exploratoria, post hoc); conformidad académica pendiente.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from src.r4_v2 import calibracion as CAL
from src.r4_v2 import metricas as M
from src.r4_v2.common import RAIZ_REPO, GuardError, directorio_nuevo, escribir_json_nuevo, exigir_ruta_no_prohibida, sha256_archivo, sha256_bytes
from src.r4_v2.common import cargar_config as cargar_config_g2
from src.r4_v2.particion import cargar_particion
from src.r5_v2.comun import comparar_instantaneas
from src.r7_v2 import g4_reporte as R
from src.r7_v2 import umbral_sistema as U

CATEGORIAS = (0, 1, 2, 3, 4, 6, 7)
NO_MALIGNAS = (0, 1, 2, 3, 4, 6)
N_MIN_AUC = 10
LEYENDAS = (
    "análisis DESCRIPTIVO por categoría original: no altera etiquetas, entrenamiento ni selección (el mapeo v2 no cambia)",
    "la categoría 6 NO se reevalúa como positiva",
    "R7.3 no se declara cumplido (discrepancia de alcance: el IOV se definió para el mapeo anterior)",
    "umbral de transferencia por fold (S* = 0.90, calculado con los otros cuatro folds): estimación en desarrollo, no validación independiente; no es el umbral de R8",
    "sin p-valores; sin independencia por paciente (grupos pHash ≤ 8 exploratorios); viabilidad clínica: no determinada",
    "no se afirma independencia absoluta del test histórico")
ENCABEZADO = (f"# mapeo v2 | phash_le8_v1 (provisional) | analisis_categorias_v1 | sin uso de test | {U.LEYENDA} | " + " | ".join(LEYENDAS))


def exigir_categorias(filas) -> None:
    """La categoría 5 y cualquier clase fuera de {0, 1, 2, 3, 4, 6, 7} bloquean; label_v2 debe ser 1 solo para la 7."""
    for r in filas:
        c = int(r["original_class"])
        if c == 5:
            raise GuardError(f"imagen {r['image_id']}: categoría 5 (ovario normal) presente; excluida de todo análisis de lesiones")
        if c not in CATEGORIAS:
            raise GuardError(f"imagen {r['image_id']}: categoría {c} fuera de {list(CATEGORIAS)}")
        if int(r["label_v2"]) != (1 if c == 7 else 0):
            raise GuardError(f"imagen {r['image_id']}: label_v2 no coincide con el mapeo v2 (solo la categoría 7 es maligna)")


def digest_etiquetas(filas) -> str:
    return sha256_bytes("\n".join(f"{r['image_id']}:{r['original_class']}:{r['label_v2']}:{r['fold']}" for r in filas).encode("utf-8"))


def leer_q_barra(origen: Path, filas) -> np.ndarray:
    """q_barra verificado contra la partición (mismas imágenes, orden, fold y label_v2) y contra el SHA-256 registrado en `trazabilidad.json`."""
    import json
    tz, qc = origen / "trazabilidad.json", origen / "q_barra_oof.csv"
    if not tz.is_file() or not qc.is_file():
        raise GuardError("falta q_barra_oof.csv o trazabilidad.json de umbral_sistema_v1 (se ejecuta primero `umbral_sistema`)")
    traz = json.loads(tz.read_text(encoding="utf-8"))
    if sha256_archivo(qc) != (traz.get("sha256_bytes_archivo") or {}).get("q_barra_oof.csv"):
        raise GuardError("q_barra_oof.csv no coincide con el SHA-256 de trazabilidad.json; no se usa")
    lineas = qc.read_text(encoding="utf-8").splitlines()
    datos = list(csv.DictReader(lineas[1:] if lineas and lineas[0].startswith("#") else lineas))
    if [d["image_id"] for d in datos] != [r["image_id"] for r in filas] or [int(d["fold"]) for d in datos] != [r["fold"] for r in filas] \
            or [int(d["label_v2"]) for d in datos] != [r["label_v2"] for r in filas]:
        raise GuardError("q_barra_oof.csv no corresponde a la partición (imágenes, orden, fold o label_v2 distintos)")
    return np.array([float(d["q_barra"]) for d in datos], dtype=np.float64), traz


def auc_vs_7(orig, q, conteos, gpi, categoria: int, n_min: int = N_MIN_AUC) -> dict:
    """AUC de una categoría no maligna contra la 7; IC95 % bootstrap agrupado (mismos `b`) solo si n_c >= n_min; si no, «n insuficiente» sin IC."""
    n_c, n_7 = int((orig == categoria).sum()), int((orig == 7).sum())
    if n_c < n_min or n_7 == 0:
        return {"categoria": categoria, "n_categoria": n_c, "n_categoria_7": n_7, "auc": None, "ic": None, "estado": "n insuficiente"}
    idx = np.flatnonzero((orig == categoria) | (orig == 7))
    prep = M.preparar((orig[idx] == 7).astype(np.int64), q[idx])
    auc = float(M.calcular(prep)["roc_auc"])
    boot = M.calcular(prep, conteos[:, gpi[idx]])["roc_auc"]
    return {"categoria": categoria, "n_categoria": n_c, "n_categoria_7": n_7, "auc": auc, "ic": R.ic(boot), "estado": "con IC95 % bootstrap agrupado (mismos b de G2)"}


def analizar_categorias(y, fold, orig, q, gpi, conteos) -> dict:
    """Todo el cálculo (sin E/S). `orig`: original_class por imagen."""
    y, fold, orig, q = np.asarray(y), np.asarray(fold), np.asarray(orig), np.asarray(q, dtype=np.float64)
    pf = CAL.comprobar_transferencia(y.tolist(), q.tolist(), fold.tolist(), U.S_PRIMARIO)["por_fold"]
    umbral_fold = {int(k): float(v["umbral"]) for k, v in pf.items()}
    t_i = np.array([umbral_fold[int(k)] for k in fold])
    sobre = q >= t_i                                                                    # regla p >= umbral
    cats = []
    for c in CATEGORIAS:
        m = orig == c
        n = int(m.sum())
        x = int(sobre[m].sum())
        q1, med, q3 = (np.percentile(q[m], [25, 50, 75]) if n else (np.nan, np.nan, np.nan))
        cats.append({"categoria": c, "n": n, "n_por_fold": {int(k): int((m & (fold == k)).sum()) for k in sorted(set(fold.tolist()))}, "mediana": float(med), "q1": float(q1),
                     "q3": float(q3), "iqr": float(q3 - q1), "sobre_umbral": x, "proporcion": (x / n) if n else float("nan"),
                     "lectura": "sensibilidad (TP/n)" if c == 7 else "tasa de falsos positivos (FP/n)"})
    aucs = [auc_vs_7(orig, q, conteos, gpi, c) for c in NO_MALIGNAS]
    m6, resto = orig == 6, np.isin(orig, [0, 1, 2, 3, 4])
    fp6, n6, fpr, nr = int(sobre[m6].sum()), int(m6.sum()), int(sobre[resto].sum()), int(resto.sum())
    p6, pr = (fp6 / n6 if n6 else None), (fpr / nr if nr else None)
    foco = {"n_6": n6, "fp_6": fp6, "fp_6_sobre_n_6": f"{fp6}/{n6}", "proporcion_6": p6, "n_resto_no_malignas": nr, "fp_resto": fpr, "fp_resto_sobre_n_resto": f"{fpr}/{nr}",
            "proporcion_resto": pr, "diferencia_de_proporciones_6_menos_resto": (p6 - pr) if (p6 is not None and pr is not None) else None,
            "p_valores": "no se calculan", "reevaluada_como_positiva": False}
    return {"umbral_transferencia_por_fold": umbral_fold, "categorias": cats, "auc_vs_7": aucs, "foco_categoria_6": foco}


def ejecutar(root, config, config_g2, *, autorizacion, verificar=None, n_remuestreos=None, origen=None, destino=None) -> dict:
    token = U.exigir_autorizacion(autorizacion)                                          # antes de leer o escribir nada
    root = Path(root)
    U.validar_config(config)
    U.comprobar_coherencia(config, config_g2)
    sha_protocolo = U.exigir_protocolo_coherente(root)
    verificar = verificar or (lambda: U.verificar_integridad(root, config))
    inicio = U.exigir_integridad(verificar())
    r6 = U.verificar_r6(root, config)
    destino = U.destino_nuevo(root, config["salidas"]["categorias"], destino)
    origen = Path(origen) if origen else Path(root) / config["salidas"]["umbral"]
    exigir_ruta_no_prohibida(str(origen.relative_to(root)) if origen.is_relative_to(root) else str(origen))
    particion = cargar_particion(root, config_g2)                                        # K1-K4
    filas = particion.filas
    exigir_categorias(filas)
    antes = digest_etiquetas(filas)
    q, traz = leer_q_barra(origen, filas)
    if traz.get("sha256_bytes_archivo_particion") != particion.sha256_bytes_archivo:
        raise GuardError("q_barra se calculó con otra partición (sha256_bytes_archivo distinto)")
    y = np.array([r["label_v2"] for r in filas])
    fold = np.array([r["fold"] for r in filas])
    orig = np.array([r["original_class"] for r in filas])
    gpi, conteos, indices, n_b = U.indices_b(root, config, config_g2, [r["group_id"] for r in filas], y, n_remuestreos)
    res = analizar_categorias(y, fold, orig, q, gpi, conteos)
    despues = digest_etiquetas(filas)
    if antes != despues:
        raise GuardError("las etiquetas cambiaron durante el análisis: se detiene")
    destino = directorio_nuevo(destino)                                                  # K5
    U.escribir_tabla_con(destino / "categorias.csv", [{
        "categoria": c["categoria"], "n": c["n"], "n_por_fold": " ".join(f"f{k}:{v}" for k, v in c["n_por_fold"].items()), "mediana_q_barra": U.f(c["mediana"]),
        "q1": U.f(c["q1"]), "q3": U.f(c["q3"]), "iqr": U.f(c["iqr"]), "sobre_umbral_x": c["sobre_umbral"], "sobre_umbral_n": c["n"],
        "sobre_umbral_x_sobre_n": f"{c['sobre_umbral']}/{c['n']}", "proporcion": U.f(c["proporcion"]), "lectura": c["lectura"]} for c in res["categorias"]],
        ENCABEZADO, "umbral = t̂ de transferencia del fold de cada imagen (S* = 0.90, otros cuatro folds)")
    U.escribir_tabla_con(destino / "auc_vs_7.csv", [{
        "categoria": a["categoria"], "n_categoria": a["n_categoria"], "n_categoria_7": a["n_categoria_7"], "auc_vs_7": U.f(a["auc"]), "ic95_inf": U.f(a["ic"]["ic_inf"]) if a["ic"] else "",
        "ic95_sup": U.f(a["ic"]["ic_sup"]) if a["ic"] else "", "estado": a["estado"]} for a in res["auc_vs_7"]], ENCABEZADO, f"IC solo si n_c >= {N_MIN_AUC}; mismos b de G2")
    escribir_json_nuevo(destino / "foco_categoria_6.json", U.a_json({"leyendas": list(LEYENDAS), "foco_categoria_6": res["foco_categoria_6"]}))
    fin = verificar()
    dif = comparar_instantaneas(inicio, fin)
    escribir_json_nuevo(destino / "resumen_categorias.json", U.a_json({
        "esquema": "resumen_categorias_v1/1", "protocolo_id": U.PROTOCOLO_ID, "leyendas": list(LEYENDAS), "leyenda": U.LEYENDA, "autorizacion": token, "fecha": U.ahora(),
        "git_commit_head": inicio["git_commit_head"], "base_r6": r6, "sha256_bytes_archivo_protocolo": sha_protocolo, "sha256_bytes_archivo_particion": particion.sha256_bytes_archivo,
        "entrada": {"q_barra_oof.csv": traz["sha256_bytes_archivo"]["q_barra_oof.csv"], "verificado_contra": "trazabilidad.json de umbral_sistema_v1",
                    "git_commit_head_umbral_sistema": traz.get("git_commit_head")},
        "etiquetas_sin_alterar": {"sha256_antes": antes, "sha256_despues": despues, "iguales": antes == despues},
        "categoria_6_reevaluada_como_positiva": False, "r7_3_declarado_cumplido": False, "umbral_de_transferencia": "S* = 0.90 por fold; estimación en desarrollo; no congelado",
        "umbral_transferencia_por_fold": res["umbral_transferencia_por_fold"], "n_remuestreos": n_b, "indices_b_verificacion": indices, "n_minimo_auc": N_MIN_AUC,
        "viabilidad_clinica": "no determinada", "limites": list(U.LIMITES), "diferencias_inicio_fin": dif, "invalida_integridad": bool(dif) or bool(fin["bloqueos"]),
        "reporte_aceptado": not (bool(dif) or bool(fin["bloqueos"])), "numpy_version": np.__version__,
        "sha256_bytes_archivo": {p.name: sha256_archivo(p) for p in sorted(destino.iterdir()) if p.suffix in (".csv", ".json")}}))
    return {"dir": destino, "resultados": res, "reporte_aceptado": not (bool(dif) or bool(fin["bloqueos"])), "diferencias_inicio_fin": dif}


def main(argv=None, *, root=None, config=None, config_g2=None, verificar=None, n_remuestreos=None, origen=None, destino=None) -> int:
    ap = argparse.ArgumentParser(description="analisis_categorias_v1 (R7): análisis descriptivo por categoría original; solo desarrollo; R7.3 no se declara cumplido")
    ap.add_argument("--autorizacion", default="", help="CONFIRMADO:D-NNN de ejecución; sin ella no se lee ni se escribe nada")
    args = ap.parse_args(argv)
    root = Path(root or RAIZ_REPO).resolve()
    try:
        U.exigir_autorizacion(args.autorizacion)
        config = config or U.cargar_config(root / U.CONFIG_REL)
        config_g2 = config_g2 or cargar_config_g2(root / config["referencia_g2"]["config"])
        r = ejecutar(root, config, config_g2, autorizacion=args.autorizacion, verificar=verificar, n_remuestreos=n_remuestreos, origen=origen, destino=destino)
        print(f"{U.LEYENDA}\nanálisis descriptivo; R7.3 no se declara cumplido\nsalida: {r['dir']}")
        if not r["reporte_aceptado"]:
            print("SALIDA NO ACEPTADA: la integridad final difiere de la inicial; no debe usarse.", file=sys.stderr)
            return U.CODIGO_SALIDA_INTEGRIDAD_INVALIDA
        return 0
    except GuardError as exc:
        print(f"DETENIDO: {exc}", file=sys.stderr)
        return U.CODIGO_SALIDA_GUARDA


if __name__ == "__main__":
    sys.exit(main())
