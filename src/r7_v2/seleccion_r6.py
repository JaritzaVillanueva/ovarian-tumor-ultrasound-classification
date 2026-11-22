"""R6: regla de selección de configuración (arquitectura x {A, B_hibrida}) aplicada UNA vez a tablas de AP OOF ya aceptadas.

Regla (ver `protocolo_r6_v1.md`): AP_media = media entre semillas del AP OOF por semilla; se redondea a 4 decimales SOLO para detectar
empates exactos; elegida = máxima; desempate: (i) A sobre B_hibrida, (ii) mayor ROC-AUC media, (iii) orden fijo de arquitecturas. Sin
tolerancias, sensibilidad, umbral ni IC. C no participa. Solo importa `src.r4_v2.common` (stdlib): sin torch ni numpy; no abre test.
Integridad sencilla (HEAD, árbol limpio en `src/r7_v2`, SHA-256): NO es K6.

CLI: python -B -X pycache_prefix=<dir> -m src.r7_v2.seleccion_r6 --autorizacion CONFIRMADO:D-NNN [--g2-t3 R] [--g3b-tb1 R] [--g2-t4 R]
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.r4_v2.common import (RAIZ_REPO, GuardError, directorio_nuevo, escribir_json_nuevo, exigir_ruta_no_prohibida,
                              exigir_salida_segura, git, git_ok, git_texto, sha256_archivo, sha256_bytes)

PROTOCOLO_ID = "seleccion_r6_v1"
PROTOCOLO_REL = "src/r7_v2/protocolo_r6_v1.md"
CODIGO_REL = "src/r7_v2/seleccion_r6.py"
SIN_CAMBIOS = ("src/r7_v2",)
SALIDA_REL = "results/r7_v2/phash_le8_v1/seleccion_r6"
T3_REL = "results/r4_v2/phash_le8_v1/g2_baseline_v1/reporte/T3_por_semilla.csv"
TB1_REL = "results/r5_v2/phash_le8_v1/g3b_condiciones_v1/reporte_condiciones/T-B1_por_semilla.csv"
LEYENDA = "partición provisional (phash_le8_v1, exploratoria, post hoc); conformidad académica pendiente"
AUTORIZACION = re.compile(r"CONFIRMADO:D-[0-9]{3}")
ARQUITECTURAS = ("resnet50", "densenet121", "convnext_tiny")          # el orden es el criterio (iii)
CONDICIONES = ("A", "B_hibrida")                                       # el orden es el criterio (i)
IGNORADAS = ("C", "B_oof")                                             # no desplegables / fuera de R6: se registran, no participan
SEMILLAS_ESPERADAS = 3
DESCRIPTIVAS = ("desc_sensibilidad_0.5_media", "desc_especificidad_0.5_media")
COLUMNAS = ["rango", "elegible", "condicion", "arquitectura", "n_semillas", "semillas", "ap_por_semilla", "ap_media", "ap_media_4dec", "roc_auc_media",
            "ap_rango", "ap_desv_est", "ap_semillas_comunes", *DESCRIPTIVAS, "elegida"]
ENCABEZADO = (f"# mapeo v2 | phash_le8_v1 (provisional) | {PROTOCOLO_ID} | sin uso de test | {LEYENDA} | columnas desc_* y ap_semillas_comunes: "
              "descriptivas, no participan en la selección")
NOTA = ("R6.2 pide la media entre semillas: solo las combinaciones con n_semillas == 3 participan en la selección formal, que se fundamenta en A "
        "con tres semillas. Una combinación con menos semillas (p. ej. B de una semilla) no se compara con la media de A como estimación equivalente: "
        "se conserva en la tabla con elegible=false. La comparación emparejada por semilla común es complementaria y descriptiva. La regla es "
        "determinista y sin inferencia: no incluye IC, sensibilidad/especificidad, estabilidad entre semillas ni umbral.")


def ahora() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _id(cond, arq) -> str:
    return f"{cond}|{arq}"


# ---------------------------------------------------------------- regla pura
def _num(valor, nombre, *, obligatorio, rango01=False):
    if valor is None or (isinstance(valor, str) and not valor.strip()):
        if obligatorio:
            raise GuardError(f"{nombre}: valor ausente")
        return None
    try:
        x = float(valor)
    except (TypeError, ValueError):
        raise GuardError(f"{nombre}: no numérico ({valor!r})") from None
    if not math.isfinite(x) or (rango01 and not 0.0 <= x <= 1.0):
        raise GuardError(f"{nombre}: fuera de rango o no finito ({valor!r})")
    return x


def _media(xs):
    return math.fsum(xs) / len(xs) if xs else None


def aplicar_regla(filas) -> dict:
    """`filas`: dicts con condicion, arquitectura, semilla, ap, roc_auc y opcionales `sensibilidad_0.5`/`especificidad_0.5`.
    Falla si hay filas duplicadas, valores inválidos, condiciones desconocidas o falta alguna de las 6 combinaciones."""
    ignoradas, por_combo, vistos = {}, {}, set()
    for f in filas:
        cond, arq = f.get("condicion"), f.get("arquitectura")
        if cond in IGNORADAS:
            ignoradas[cond] = ignoradas.get(cond, 0) + 1
            continue
        if cond not in CONDICIONES or arq not in ARQUITECTURAS:
            raise GuardError(f"condición/arquitectura desconocida: {cond!r}/{arq!r}")
        try:
            semilla = int(f["semilla"])
        except (KeyError, TypeError, ValueError):
            raise GuardError(f"{_id(cond, arq)}: semilla inválida ({f.get('semilla')!r})") from None
        if (cond, arq, semilla) in vistos:
            raise GuardError(f"fila duplicada: {_id(cond, arq)} semilla {semilla}")
        vistos.add((cond, arq, semilla))
        por_combo.setdefault((cond, arq), {})[semilla] = {
            "ap": _num(f.get("ap"), f"{_id(cond, arq)} s{semilla} ap", obligatorio=True, rango01=True),
            "roc_auc": _num(f.get("roc_auc"), f"{_id(cond, arq)} s{semilla} roc_auc", obligatorio=True, rango01=True),
            "sens": _num(f.get("sensibilidad_0.5"), "sensibilidad_0.5", obligatorio=False, rango01=True),
            "esp": _num(f.get("especificidad_0.5"), "especificidad_0.5", obligatorio=False, rango01=True)}
    faltan = [_id(c, a) for c in CONDICIONES for a in ARQUITECTURAS if (c, a) not in por_combo]
    if faltan:
        raise GuardError(f"faltan combinaciones arquitectura x condición: {faltan} (las 6 deben estar presentes)")
    comunes = sorted(set.intersection(*[set(v) for v in por_combo.values()]))
    tabla = []
    for (cond, arq), por_s in por_combo.items():
        sems = sorted(por_s)
        aps = [por_s[s]["ap"] for s in sems]
        ap_m = _media(aps)
        desc = {}
        for nombre, clave in zip(DESCRIPTIVAS, ("sens", "esp")):
            vals = [por_s[s][clave] for s in sems if por_s[s][clave] is not None]
            desc[nombre] = _media(vals) if len(vals) == len(sems) else None
        tabla.append({"condicion": cond, "arquitectura": arq, "n_semillas": len(sems), "semillas": sems, "ap_por_semilla": {str(s): por_s[s]["ap"] for s in sems},
                      "ap_media": ap_m, "ap_media_4dec": f"{round(ap_m, 4):.4f}", "roc_auc_media": _media([por_s[s]["roc_auc"] for s in sems]),
                      "ap_rango": max(aps) - min(aps), "ap_desv_est": statistics.stdev(aps) if len(aps) > 1 else None,
                      "ap_semillas_comunes": _media([por_s[s]["ap"] for s in comunes]) if comunes else None, **desc})
    for t in tabla:
        t["elegible"] = t["n_semillas"] == SEMILLAS_ESPERADAS
    ap_ineleg = [_id(t["condicion"], t["arquitectura"]) for t in tabla if t["condicion"] == "A" and not t["elegible"]]
    if ap_ineleg or not any(t["elegible"] for t in tabla):
        raise GuardError(f"A debe tener {SEMILLAS_ESPERADAS} semillas en las 3 arquitecturas (sin ellas no hay selección formal): {ap_ineleg}")
    clave = lambda t: (-round(t["ap_media"], 4), CONDICIONES.index(t["condicion"]), -t["roc_auc_media"], ARQUITECTURAS.index(t["arquitectura"]))  # noqa: E731
    ranked = sorted((t for t in tabla if t["elegible"]), key=clave)
    resto = sorted((t for t in tabla if not t["elegible"]), key=lambda t: (CONDICIONES.index(t["condicion"]), ARQUITECTURAS.index(t["arquitectura"])))
    for i, t in enumerate(ranked, 1):
        t["rango"] = i
    for t in resto:
        t["rango"] = None
    tabla = ranked + resto                              # las no elegibles se conservan, sin rango, fuera del ranking, del margen y del empate
    ap4 = round(ranked[0]["ap_media"], 4)
    empatados = [t for t in ranked if round(t["ap_media"], 4) == ap4]
    paso = None
    if len(empatados) > 1:                              # el desempate solo interviene con empate exacto a 4 decimales; `paso` = último paso necesario
        pool = empatados
        if len({t["condicion"] for t in pool}) > 1:
            pool, paso = [t for t in pool if t["condicion"] == "A"], "i"
        if len(pool) > 1:
            mejor = max(t["roc_auc_media"] for t in pool)
            pool, paso = [t for t in pool if t["roc_auc_media"] == mejor], "ii"
        if len(pool) > 1:
            pool, paso = [min(pool, key=lambda t: ARQUITECTURAS.index(t["arquitectura"]))], "iii"
        if pool[0] is not ranked[0]:                     # la clave de ordenación aplica (i), (ii), (iii) en ese orden
            raise GuardError("inconsistencia interna: el desempate no coincide con el orden de la tabla")
    elegida = ranked[0]
    for t in tabla:
        t["elegida"] = t is elegida
    incompletas = [_id(t["condicion"], t["arquitectura"]) for t in tabla if t["n_semillas"] < SEMILLAS_ESPERADAS]
    return {"tabla": tabla, "ranking": [_id(t["condicion"], t["arquitectura"]) for t in ranked],
            "margen_1_2_ap_media": elegida["ap_media"] - ranked[1]["ap_media"],
            "seleccion_formal_solo_entre": [_id(t["condicion"], t["arquitectura"]) for t in ranked],
            "b_excluida_de_la_seleccion_por_semillas_insuficientes": any(t["condicion"] == "B_hibrida" and not t["elegible"] for t in tabla),
            "comparacion_emparejada_semilla_comun": emparejada(por_combo),
            "elegida": {"condicion": elegida["condicion"], "arquitectura": elegida["arquitectura"]},
            "empatados_en_el_maximo": [_id(t["condicion"], t["arquitectura"]) for t in empatados],
            "desempate_usado": paso is not None, "paso_desempate": paso,
            "r6_2_media_entre_semillas_completa": not incompletas, "combinaciones_con_menos_de_3_semillas": incompletas,
            "semillas_homogeneas_entre_combinaciones": len({tuple(t["semillas"]) for t in tabla}) == 1, "semillas_comunes": comunes,
            "condiciones_ignoradas": ignoradas, "nota_limitacion": NOTA}


def emparejada(por_combo) -> list:
    """Solo descriptivo (no participa en la selección): por arquitectura, AP de A y de B_hibrida en las semillas comunes; diferencia B-A sin IC."""
    out = []
    for a in ARQUITECTURAS:
        pa, pb = por_combo[("A", a)], por_combo[("B_hibrida", a)]
        sc = sorted(set(pa) & set(pb))
        if not sc:
            continue
        ap_a, ap_b = [pa[s]["ap"] for s in sc], [pb[s]["ap"] for s in sc]
        out.append({"arquitectura": a, "semillas_comunes": sc, "ap_A": dict(zip(map(str, sc), ap_a)), "ap_B_hibrida": dict(zip(map(str, sc), ap_b)),
                    "ap_A_media": _media(ap_a), "ap_B_hibrida_media": _media(ap_b), "diferencia_B_menos_A": _media(ap_b) - _media(ap_a),
                    "etiqueta": f"condicional a la(s) semilla(s) común(es) {sc}; sin IC (los IC emparejados están en T-B2 de G3b)"})
    return out


# ---------------------------------------------------------------- lectura de CSV aceptados
def leer_csv_comentado(path, requeridas) -> list:
    """Salta la línea inicial de comentario `# ...` (estilo `escribir_tabla` de G2/G3b) y exige las columnas requeridas."""
    with open(path, newline="", encoding="utf-8") as f:
        lineas = f.read().splitlines()
    if lineas and lineas[0].startswith("#"):
        lineas = lineas[1:]
    filas = list(csv.DictReader(lineas))
    campos = set(filas[0]) if filas else set()
    if not filas or not set(requeridas) <= campos:
        raise GuardError(f"{Path(path).name}: vacío o sin columnas {sorted(set(requeridas) - campos)}")
    return filas


def leer_insumos(t3, tb1) -> tuple:
    """(filas para la regla, comprobación A: T3 de G2 frente a T-B1 de G3b). A sale de T3; B_hibrida (y C/B_oof, ignoradas) de T-B1."""
    a = [{**f, "condicion": "A"} for f in leer_csv_comentado(t3, ("arquitectura", "semilla", "ap", "roc_auc"))]
    b = leer_csv_comentado(tb1, ("condicion", "arquitectura", "semilla", "ap", "roc_auc"))
    ref = {(f["arquitectura"], int(f["semilla"])): f for f in a}
    comparadas = 0
    for f in b:
        if f["condicion"] != "A":
            continue
        otro = ref.get((f["arquitectura"], int(f["semilla"])))
        if otro is None:
            continue
        for col in ("ap", "roc_auc", "sensibilidad_0.5", "especificidad_0.5"):
            if f.get(col) not in (None, "") and otro.get(col) not in (None, "") and float(f[col]) != float(otro[col]):
                raise GuardError(f"A {f['arquitectura']} s{f['semilla']}: {col} difiere entre T3 de G2 ({otro[col]}) y T-B1 de G3b ({f[col]})")
        comparadas += 1
    return a + [f for f in b if f["condicion"] != "A"], {"n_comparaciones_A": comparadas, "coincide": True}


def t4_descriptivo(path) -> list:
    filas = leer_csv_comentado(path, ("comparacion", "metrica", "delta"))
    return [{"comparacion": f["comparacion"], "delta_ap": f["delta"]} for f in filas if f["metrica"] == "ap"]


# ---------------------------------------------------------------- integridad sencilla (≠ K6)
def huella(root, rel) -> dict:
    ruta = Path(root) / rel
    r = git(root, "cat-file", "blob", f"HEAD:{rel}")
    return {"sha256_bytes_archivo": sha256_archivo(ruta) if ruta.is_file() else None,
            "sha256_contenido_blob": sha256_bytes(r.stdout) if r.returncode == 0 else None}


def verificar_integridad(root) -> dict:
    head = git_texto(root, "rev-parse", "HEAD", obligatorio=True)
    estado = git_ok(root, "status", "--porcelain", "--untracked-files=all", "--", *SIN_CAMBIOS).decode("utf-8", "replace")
    status = [l for l in estado.splitlines() if l.strip()]
    bloqueos = [f"arbol_sucio: {l}" for l in status]
    archivos = {rel: huella(root, rel) for rel in (PROTOCOLO_REL, CODIGO_REL)}
    for rel, h in archivos.items():
        if h["sha256_bytes_archivo"] is None or h["sha256_contenido_blob"] is None:
            bloqueos.append(f"archivo_ausente_o_no_versionado: {rel}")
    if bloqueos:
        raise GuardError("integridad: " + "; ".join(bloqueos[:6]) + ". No se genera la selección")
    return {"git_commit_head": head, "rutas_sin_cambios": list(SIN_CAMBIOS), "status_porcelain": status, "archivos": archivos,
            "tipo": "sencilla (HEAD + árbol limpio + SHA-256); no es K6"}


def _insumo(root, ruta) -> Path:
    """K4: ninguna ruta con componente `test` (ni la dada ni la resuelta) y debe estar dentro de `results/` del repositorio."""
    p = Path(ruta)
    p = p if p.is_absolute() else Path(root) / p
    exigir_ruta_no_prohibida(str(ruta))
    exigir_salida_segura(root, p, "results")
    if not p.is_file():
        raise GuardError(f"insumo ausente: {p}")
    return p.resolve()


# ---------------------------------------------------------------- orquestación
def ejecutar(root, autorizacion, g2_t3=T3_REL, g3b_tb1=TB1_REL, g2_t4=None) -> Path:
    if not isinstance(autorizacion, str) or not AUTORIZACION.fullmatch(autorizacion):
        raise GuardError(f"--autorizacion exige el formato {AUTORIZACION.pattern!r}; no se otorga ni se infiere")
    root = Path(root)
    t3, tb1 = _insumo(root, g2_t3), _insumo(root, g3b_tb1)
    t4 = _insumo(root, g2_t4) if g2_t4 else None
    destino = exigir_salida_segura(root, root / SALIDA_REL, "results/r7_v2")
    if destino.exists():
        raise GuardError(f"K5: la salida ya existe y no se sobrescribe: {destino}")
    integridad = verificar_integridad(root)
    filas, comprobacion_a = leer_insumos(t3, tb1)
    res = aplicar_regla(filas)
    insumos = {"T3_G2": t3, "T-B1_G3b": tb1, **({"T4_G2": t4} if t4 else {})}
    salida = {"protocolo": PROTOCOLO_ID, "leyenda": LEYENDA, "fecha_utc": ahora(), "autorizacion": autorizacion, "integridad": integridad,
              "insumos_sha256": {k: {"ruta": v.relative_to(root.resolve()).as_posix(), "sha256_bytes_archivo": sha256_archivo(v)} for k, v in insumos.items()},
              "comprobacion_A_t3_vs_tb1": comprobacion_a, "columnas_descriptivas_no_participan_en_la_seleccion": [*DESCRIPTIVAS, "ap_semillas_comunes"],
              "t4_descriptivo_no_participa": t4_descriptivo(t4) if t4 else None, **res}
    tabla_csv = [{**{c: t.get(c) for c in COLUMNAS}, "semillas": " ".join(map(str, t["semillas"])),
                  "ap_por_semilla": " ".join(f"{s}:{v:.6f}" for s, v in t["ap_por_semilla"].items()),
                  **{c: ("" if t[c] is None else f"{t[c]:.6f}") for c in ("ap_media", "roc_auc_media", "ap_rango", "ap_desv_est", "ap_semillas_comunes", *DESCRIPTIVAS)},
                  "elegida": str(t["elegida"]).lower(), "elegible": str(t["elegible"]).lower()} for t in res["tabla"]]
    destino = directorio_nuevo(destino)                                  # K5: solo tras validar y calcular
    escribir_json_nuevo(destino / "seleccion_r6.json", salida)
    with open(destino / "tabla_r6.csv", "x", encoding="utf-8", newline="") as fh:
        fh.write(ENCABEZADO + "\n")
        w = csv.DictWriter(fh, fieldnames=COLUMNAS, lineterminator="\n")
        w.writeheader()
        w.writerows(tabla_csv)
    return destino


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--autorizacion", required=True)
    p.add_argument("--g2-t3", default=T3_REL)
    p.add_argument("--g3b-tb1", default=TB1_REL)
    p.add_argument("--g2-t4", default=None)
    a = p.parse_args(argv)
    try:
        destino = ejecutar(RAIZ_REPO, a.autorizacion, a.g2_t3, a.g3b_tb1, a.g2_t4)
    except GuardError as exc:
        print(f"BLOQUEADO: {exc}", file=sys.stderr)
        return 3
    print(f"{LEYENDA}\nselección R6 escrita en {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
