"""R7 `umbral_sistema_v1`: calibración EXPLORATORIA del umbral del sistema provisional DenseNet121 + A sobre q_barra (solo desarrollo).

* Entradas: SOLO las 15 corridas A|densenet121 de G2 (`r4_v2.agregacion`, con su comparabilidad) y la partición (K1-K4). No usa otras arquitecturas, test ni normales.
* q_barra = promedio de las 3 predicciones OOF por imagen (semillas 42, 43, 44): OOF exacta de un ensamble de TRES modelos del mismo fold. No existe OOF del sistema de 15
  modelos (12 de los 15 la vieron en entrenamiento): la calibración sobre q_barra NO es equivalente ni validada para él (`discrepancia_A_vs_B`).
* S* = 0.90 primario (objetivo exploratorio, no garantía); 0.85 y 0.95 solo descriptivos (sin seleccionar). `calibracion.calibrar_umbral` / `comprobar_transferencia` /
  `evaluar_en_umbral` sin modificar; regla `p >= umbral`; empates sin romper; k = ceil(S*·n⁺) en aritmética entera. El umbral calculado NO se congela y no es el de R8.
* Bootstrap agrupado por `group_id`, estratificado, con los MISMOS índices `b` de G2 (verificados con `verificar_indices_g2`), umbral recalculado en cada remuestreo.

CLI (solo la autora, con D-NNN de ejecución y el commit de congelamiento; un agente no lo ejecuta):
    python -B -X pycache_prefix=<dir> -m src.r7_v2.umbral_sistema --autorizacion CONFIRMADO:D-NNN
Este módulo también aloja las guardas comunes que importa `analisis_categorias`. Partición provisional (phash_le8_v1, exploratoria, post hoc); conformidad académica pendiente.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import sys
from datetime import datetime, timezone
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import numpy as np

from src.r4_v2 import agregacion as AG
from src.r4_v2 import bootstrap as BS
from src.r4_v2 import calibracion as CAL
from src.r4_v2 import metricas as M
from src.r4_v2.common import (RAIZ_REPO, GuardError, directorio_nuevo, escribir_json_nuevo, exigir_ruta_no_prohibida,  # noqa: F401
                              exigir_salida_segura, git_ok, git_texto, sha256_archivo, sha256_bytes)
from src.r4_v2.common import cargar_config as cargar_config_g2
from src.r4_v2.entrenamiento import corrida_completada, id_corrida
from src.r4_v2.particion import cargar_particion
from src.r5_v2.comun import comparar_instantaneas, huella_archivo
from src.r5_v2.g3b_reporte_condiciones import verificar_indices_g2
from src.r7_v2 import g4_reporte as R

PROTOCOLO_ID = "umbral_sistema_v1"
ESQUEMA_CONFIG = "umbral_sistema_v1_config/1"
CONFIG_REL = "configs/r7_v2/umbral_sistema_v1.yaml"
PROTOCOLO_REL = "src/r7_v2/protocolo_umbral_v1.md"
SALIDA_UMBRAL_REL = "results/r7_v2/phash_le8_v1/umbral_sistema_v1"
SALIDA_CATEGORIAS_REL = "results/r7_v2/phash_le8_v1/analisis_categorias_v1"
AUTORIZACION = re.compile(r"CONFIRMADO:D-[0-9]{3}")
CODIGO_SALIDA_GUARDA, CODIGO_SALIDA_INTEGRIDAD_INVALIDA = 3, 4
S_PRIMARIO, S_DESCRIPTIVOS = "0.90", ("0.85", "0.95")
UMBRAL_05 = 0.5
CONTROL_SENS_05_G2 = 0.447368
NO_CONGELADO = "calculado en desarrollo; NO congelado; no es el umbral de R8"
TRANSFERENCIA = "estimación de transferencia en desarrollo, no validación independiente"
NOTA_VPP = "VPP: solo válido para la prevalencia del desarrollo; no transferible a otras poblaciones"
LEYENDA = "partición provisional (phash_le8_v1, exploratoria, post hoc); conformidad académica pendiente"
LEYENDAS = (
    NO_CONGELADO,
    "S* = 0.90 es el objetivo primario de calibración: objetivo exploratorio (ya usado en G4), no una garantía de sensibilidad ni una validación clínica; "
    "S* = 0.85 y 0.95 son solo descriptivos (sensibilidad al criterio): el código no selecciona entre ellos",
    "sistema final PROVISIONAL (no validado): DenseNet121, representación A, 15 checkpoints (3 semillas × 5 folds), promedio aritmético simple; su umbral numérico no se congela en este hito",
    TRANSFERENCIA, NOTA_VPP,
    "viabilidad clínica: no determinada; sin mínimos absolutos de especificidad ni máximos de FP",
    "no se afirma independencia absoluta del test histórico")
DISCREPANCIA_A_VS_B = {
    "A": "q_barra = promedio de las tres predicciones OOF por imagen (semillas 42, 43, 44): OOF exacta de un ensamble de TRES modelos del mismo fold (G2 §10.2 fila 3)",
    "B": "sistema final = promedio de 15 modelos de inferencia; para una imagen de desarrollo 12 de los 15 modelos la vieron en entrenamiento: no existe OOF de B",
    "consecuencia": "la calibración sobre A no se presenta como equivalente ni validada para B; el desajuste no es cuantificable con este diseño",
    "criterio_numerico_de_discrepancia": None}
AFIRMACIONES_POSIBLES = (
    "lo calculado sobre q_barra de tres semillas (ensamble de 3 modelos del mismo fold), con su incertidumbre bootstrap",
    "cómo cambian AP, ROC-AUC, t̂ y el punto de operación con m = 1, 2, 3 modelos promediados (descripción, sin criterio de decisión)",
    "la transferencia entre folds como estimación en desarrollo (no validación independiente)")
AFIRMACIONES_NO_PERMITIDAS = (
    "que t̂ garantice sensibilidad en el ensamble de 15 modelos o en el test",
    "que el sistema de 15 modelos esté validado",
    "extrapolar numéricamente de m <= 3 a 15 modelos",
    "utilidad clínica del sistema o del umbral",
    "que mover el umbral mejore el AP o la ROC-AUC (mover el umbral desplaza el punto sobre la misma curva)",
    "independencia absoluta del test histórico")
LIMITES = ("38 positivas (≈ 7–8 por fold: incertidumbre alta por fold)", "partición provisional (pHash ≤ 8 exploratoria y post hoc; riesgo residual de fuga)",
           "desarrollo reutilizado en toda la cadena de decisiones", "accesos históricos al test declarados (P335); este análisis no abre el test")
PROCEDIMIENTO = {
    "s_star_primario": "0.90", "s_star_descriptivos": ["0.85", "0.95"], "regla_positiva": "p >= umbral",
    "empates": "no se rompen; se reportan n_pos_empatadas y n_neg_empatadas", "k": "ceil(S* * n_pos) en aritmetica entera",
    "sistema_provisional": {"arquitectura": "densenet121", "representacion": "A", "semillas": [42, 43, 44], "folds": [0, 1, 2, 3, 4], "n_checkpoints": 15,
                            "agregacion": "promedio aritmetico simple de probabilidades"},
    "fuente_de_calibracion": "q_barra: promedio de las 3 predicciones OOF por imagen (ensamble de 3 modelos del mismo fold)",
    "m_agregaciones": [1, 2, 3],
    "bootstrap": {"n_remuestreos": 10000, "nivel_ic": 0.95, "percentiles": [2.5, 50, 97.5], "indices_b": "los de G2 (verificar_indices_g2)"},
    "umbral_de_comparacion": 0.5, "criterio_numerico_de_discrepancia": None, "umbral_congelado": False, "n_minimo_auc_por_categoria": 10,
    "categorias_permitidas": [0, 1, 2, 3, 4, 6, 7], "categoria_6_reevaluada_como_positiva": False, "r7_3_declarado_cumplido": False,
    "control_sensibilidad_media_0_5_g2": 0.447368}
FRASES_PROTOCOLO = ("S* = 0.90", "NO congelado", "discrepancia_A_vs_B", "12 de los 15", "no existe OOF de B", "R7.3", "no se declara cumplido", "categoría 6",
                    "viabilidad clínica: no determinada", "p >= umbral", "ceil(S*·n⁺)", "0.85 y 0.95", "estimación de transferencia en desarrollo, no validación independiente",
                    "no se afirma independencia absoluta del test histórico", "no se selecciona")
ENCABEZADO = (f"# mapeo v2 | phash_le8_v1 (provisional) | {PROTOCOLO_ID} | sin uso de test | {LEYENDA} | {NO_CONGELADO} | S* = 0.90 objetivo exploratorio, no garantía | "
              f"{TRANSFERENCIA} | viabilidad clínica: no determinada")


# ============================================================================ configuración, autorización y guardas comunes
def ahora() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def exigir_autorizacion(valor) -> str:
    if not isinstance(valor, str) or not AUTORIZACION.fullmatch(valor):
        raise GuardError(f"se exige --autorizacion con formato {AUTORIZACION.pattern!r}; no se otorga ni se infiere (sin ella no se lee ni se escribe nada)")
    return valor


def cargar_config(path) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise GuardError("Se requiere PyYAML para leer configs/r7_v2/umbral_sistema_v1.yaml") from exc
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    validar_config(config)
    return config


def validar_config(config) -> None:
    """Invariantes que el YAML no puede cambiar: procedimiento (contra constantes), salidas, R6, test cerrado y huellas del propio código."""
    if not isinstance(config, dict) or config.get("esquema") != ESQUEMA_CONFIG:
        raise GuardError(f"Configuración con esquema distinto de {ESQUEMA_CONFIG!r}")
    if config["protocolo"]["id"] != PROTOCOLO_ID or config["protocolo"]["ruta"] != PROTOCOLO_REL:
        raise GuardError("La configuración no corresponde a umbral_sistema_v1")
    if config["estudio"]["particion"] != "phash_le8_v1" or config["estudio"]["mapeo"] != "v2":
        raise GuardError("La configuración no corresponde a phash_le8_v1 / mapeo v2")
    if (config.get("acceso_test") or {}).get("estado") != "cerrado":
        raise GuardError("acceso_test debe ser 'cerrado'")
    if config.get("procedimiento") != PROCEDIMIENTO:
        raise GuardError("procedimiento del YAML distinto de las constantes del código (S*, regla, empates, sistema, bootstrap, umbral no congelado, categorías): "
                         "se bloquea; cualquier cambio es una enmienda previa al congelamiento")
    if config.get("salidas") != {"umbral": SALIDA_UMBRAL_REL, "categorias": SALIDA_CATEGORIAS_REL}:
        raise GuardError("salidas del YAML distintas de results/r7_v2/phash_le8_v1/{umbral_sistema_v1,analisis_categorias_v1}")
    for ruta in config["salidas"].values():
        exigir_ruta_no_prohibida(ruta)
    if config["referencia_r6"]["elegida"] != {"condicion": "A", "arquitectura": "densenet121"}:
        raise GuardError("referencia_r6.elegida debe ser A|densenet121")
    huella = set(config["integridad"]["archivos_con_huella"])
    for rel in (CONFIG_REL, PROTOCOLO_REL, "src/r7_v2/umbral_sistema.py", "src/r7_v2/analisis_categorias.py", config["referencia_g2"]["config"]):
        if rel not in huella:
            raise GuardError(f"integridad.archivos_con_huella debe incluir {rel}")


def comprobar_coherencia(config, config_g2) -> None:
    sp = config["procedimiento"]["sistema_provisional"]
    if [int(f) for f in config_g2["particion"]["verificaciones_bloqueantes"]["folds"]] != sp["folds"] or config["estudio"]["particion"] != config_g2["estudio"]["particion"]:
        raise GuardError("los folds o la partición no coinciden con los de G2")
    if sp["arquitectura"] not in config_g2["diseno"]["arquitecturas"] or not set(sp["semillas"]) <= set(config_g2["diseno"]["semillas"]):
        raise GuardError("el sistema provisional no está dentro del diseño de G2")


def exigir_protocolo_coherente(root) -> str:
    ruta = Path(root) / PROTOCOLO_REL
    if not ruta.is_file():
        raise GuardError(f"{PROTOCOLO_REL} no existe: no se puede validar la coherencia del procedimiento")
    texto = ruta.read_text(encoding="utf-8")
    faltan = [f for f in FRASES_PROTOCOLO if f not in texto]
    if faltan or re.search(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", texto):
        raise GuardError(f"el protocolo no es coherente con el procedimiento del código (faltan frases {faltan}) o contiene un hash de 64 hex (no debe tener hash propio)")
    return sha256_archivo(ruta)


def verificar_r6(root, config) -> dict:
    """GUARDA de R6: `seleccion_r6.json` debe existir y elegir A|densenet121; la ausencia o una elegida distinta BLOQUEA (sin bandera que la debilite)."""
    rel = config["referencia_r6"]["salida"]
    exigir_ruta_no_prohibida(rel)
    ruta = Path(root) / rel
    if not ruta.is_file():
        raise GuardError(f"{rel} no existe: se exige la selección de R6 con elegida A|densenet121 (la ausencia bloquea)")
    try:
        elegida = json.loads(ruta.read_text(encoding="utf-8"))["elegida"]
    except (OSError, ValueError, KeyError):
        raise GuardError(f"{rel}: ilegible o sin 'elegida'") from None
    if elegida != config["referencia_r6"]["elegida"]:
        raise GuardError(f"R6 eligió {elegida.get('condicion')}|{elegida.get('arquitectura')}: este análisis es de A|densenet121; se bloquea")
    return {"estado": "verificada", "ruta": rel, "sha256_bytes_archivo": sha256_archivo(ruta)}


def verificar_integridad(root, config) -> dict:
    """HEAD, árbol limpio en `rutas_sin_cambios`, SHA-256 de `archivos_con_huella`, pin del YAML de G2 y PROTOCOLO CONGELADO: el YAML debe registrar un commit y el
    SHA-256 del protocolo, que debe coincidir con el blob de HEAD y con el del commit registrado. NO es K6. Un error de Git lanza GuardError."""
    ig = config["integridad"]
    head = git_texto(root, "rev-parse", "HEAD", obligatorio=True)
    estado = git_ok(root, "status", "--porcelain", "--untracked-files=all", "--", *ig["rutas_sin_cambios"]).decode("utf-8", "replace")
    status = [l for l in estado.splitlines() if l.strip()]
    bloqueos = [{"codigo": "arbol_sucio", "detalle": l} for l in status]
    archivos = {}
    for rel in ig["archivos_con_huella"]:
        reg = huella_archivo(root, rel)
        archivos[rel] = reg
        if reg["sha256_bytes_archivo"] is None:
            bloqueos.append({"codigo": "archivo_ausente", "detalle": rel})
        elif reg["sha256_contenido_blob"] is None:
            bloqueos.append({"codigo": "archivo_no_versionado", "detalle": rel})
    ref = config["referencia_g2"]
    if archivos[ref["config"]]["sha256_contenido_blob"] != ref["sha256_contenido_blob"]:
        bloqueos.append({"codigo": "yaml_referencia_g2_distinto_del_fijado", "detalle": ref["config"]})
    p = config["protocolo"]
    blob_head = archivos[PROTOCOLO_REL]["sha256_contenido_blob"]
    if not p.get("git_commit") or not p.get("sha256_contenido_blob"):
        bloqueos.append({"codigo": "protocolo_no_congelado", "detalle": "el YAML no registra commit y SHA-256 del protocolo (commit 2 pendiente)"})
    else:
        if blob_head != p["sha256_contenido_blob"]:
            bloqueos.append({"codigo": "protocolo_distinto_del_registrado", "detalle": f"blob de HEAD {blob_head} != YAML"})
        try:
            en_commit = sha256_bytes(git_ok(root, "cat-file", "blob", f"{p['git_commit']}:{PROTOCOLO_REL}"))
        except GuardError:
            en_commit = None
        if en_commit != p["sha256_contenido_blob"]:
            bloqueos.append({"codigo": "protocolo_del_commit_registrado_distinto", "detalle": f"commit {p['git_commit']}"})
    return {"git_commit_head": head, "rama": git_texto(root, "rev-parse", "--abbrev-ref", "HEAD"), "rutas_sin_cambios": list(ig["rutas_sin_cambios"]),
            "status_porcelain": status, "archivos": archivos, "bloqueos": bloqueos}


def exigir_integridad(res: dict) -> dict:
    if res["bloqueos"]:
        raise GuardError("integridad: " + "; ".join(f"{b['codigo']}: {b.get('detalle')}" for b in res["bloqueos"][:6]) + ". No se lee ni se escribe nada más")
    return res


# ============================================================================ utilidades de salida
def _json_default(o):
    if isinstance(o, Fraction):
        return float(o)
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def a_json(obj):
    return json.loads(json.dumps(obj, default=_json_default))


def f(x) -> str:
    """`repr(float)`: ida y vuelta exacta; vacío si no hay valor."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return repr(float(x))


def escribir_tabla(path, filas, columnas, extra: str = "") -> None:
    with open(path, "x", encoding="utf-8", newline="") as fh:
        fh.write(ENCABEZADO + (f" | {extra}" if extra else "") + "\n")
        w = csv.DictWriter(fh, fieldnames=columnas, lineterminator="\n")
        w.writeheader()
        w.writerows(filas)


def escribir_tabla_con(path, filas, encabezado: str, extra: str = "") -> None:
    """Como `escribir_tabla` con un encabezado `# ...` propio (lo usa `analisis_categorias`); columnas = claves de la primera fila."""
    with open(path, "x", encoding="utf-8", newline="") as fh:
        fh.write(encabezado + (f" | {extra}" if extra else "") + chr(10))
        w = csv.DictWriter(fh, fieldnames=list(filas[0]), lineterminator=chr(10))
        w.writeheader()
        w.writerows(filas)


def destino_nuevo(root, rel, destino=None) -> Path:
    d = exigir_salida_segura(root, Path(destino) if destino else Path(root) / rel, "results/r7_v2")
    if d.exists():
        raise GuardError(f"K5: el directorio de salida ya existe y no se sobrescribe: {d}")
    return d


def indices_b(root, config, config_g2, group_ids, y, n_remuestreos=None):
    """Mismos índices `b` de G2: se regeneran con sus semillas y se verifican contra los .npz de G2 (con B reducido solo en pruebas: no se verifican)."""
    gpi, clase, nombres = BS.indexar_grupos(group_ids, y)
    cfg = config_g2["bootstrap"]
    n_b = int(n_remuestreos or cfg["n_remuestreos"])
    cont = {s: BS.generar_conteos(clase, n_b, s, estratificado=True) for s in (cfg["semilla"], cfg["semilla_de_comprobacion"])}
    ver = verificar_indices_g2(root, config, config_g2, cont, nombres) if n_remuestreos is None else [{"estado": "no verificado: n_remuestreos reducido (solo pruebas)"}]
    return gpi, cont[cfg["semilla"]], ver, n_b


# ============================================================================ cálculo puro
def umbral_ponderado(y, q, conteos, gpi, s_star=S_PRIMARIO) -> dict:
    """t̂ recalculado en CADA remuestreo: con los pesos enteros del remuestreo, k = ceil(S*·n⁺ ponderado) y t̂ = k-ésima mayor puntuación positiva (con multiplicidad).
    Equivale a `calibrar_umbral` sobre las filas repetidas. Devuelve arrays (B,): t, k, n_pos, tp, fp, tn, sens, esp (NaN si el remuestreo no tiene positivas)."""
    s = CAL.como_fraccion(s_star)
    y, q, gpi = np.asarray(y), np.asarray(q, dtype=np.float64), np.asarray(gpi)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    orden = pos[np.argsort(-q[pos], kind="stable")]
    sp = q[orden]
    wp = conteos[:, gpi[orden]].astype(np.int64)
    n_w = wp.sum(axis=1)
    kk = -(-(s.numerator * n_w) // s.denominator)
    t = sp[(np.cumsum(wp, axis=1) >= kk[:, None]).argmax(axis=1)]
    tp = (wp * (sp[None, :] >= t[:, None])).sum(axis=1)
    wn = conteos[:, gpi[neg]].astype(np.int64)
    fp = (wn * (q[neg][None, :] >= t[:, None])).sum(axis=1)
    tn = wn.sum(axis=1) - fp
    ok = np.where(n_w >= 1, 1.0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        sens, esp = tp / n_w, tn / (tn + fp)
    return {"t": t * ok, "k": kk * ok, "n_pos": n_w * ok, "tp": tp * ok, "fp": fp * ok, "tn": tn * ok, "sens": sens * ok, "esp": esp * ok}


def percentiles_226(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    ok = x[~np.isnan(x)]
    if ok.size == 0:
        return {"p2_5": float("nan"), "p50": float("nan"), "p97_5": float("nan"), "n_remuestreos_validos": 0}
    a, b, c = np.percentile(ok, [2.5, 50, 97.5])
    return {"p2_5": float(a), "p50": float(b), "p97_5": float(c), "n_remuestreos_validos": int(ok.size)}


def metricas_en_umbral(y, q, umbral) -> dict:
    ev = CAL.evaluar_en_umbral(np.asarray(y).tolist(), np.asarray(q).tolist(), umbral)
    tp, fp = ev["tp"], ev["fp"]
    return {"umbral": float(umbral), "tp": tp, "fn": ev["fn"], "fp": fp, "tn": ev["tn"], "sens": float(ev["sensibilidad"]), "esp": float(ev["especificidad"]),
            "vpp": tp / (tp + fp) if tp + fp else float("nan")}


def calibracion_primaria(y, q, s_star=S_PRIMARIO) -> dict:
    """`calibrar_umbral` sin modificar. S* se pasa como cadena/Fraction (un float se rechaza dentro de `como_fraccion`)."""
    c = CAL.calibrar_umbral(np.asarray(y).tolist(), np.asarray(q).tolist(), s_star)
    tp, fp = c["tp"], c["fp"]
    return {"s_star": c["s_star"], "umbral": float(c["umbral"]), "umbral_repr": repr(float(c["umbral"])), "k": c["k"], "n_pos": c["n_pos"], "m_tp": c["m"], "tp": tp,
            "fn": c["fn"], "fp": fp, "tn": c["tn"], "sens_real": float(c["sensibilidad_real"]), "esp": float(c["especificidad"]),
            "vpp": tp / (tp + fp) if tp + fp else float("nan"), "n_pos_empatadas": c["n_pos_empatadas"], "n_neg_empatadas": c["n_neg_empatadas"]}


def resumen_bootstrap_umbral(d: dict) -> dict:
    return {"t_hat": percentiles_226(d["t"]), "especificidad": percentiles_226(d["esp"]), "fp": percentiles_226(d["fp"]), "sensibilidad": percentiles_226(d["sens"])}


def descripcion_scores(y, q) -> dict:
    y, q = np.asarray(y), np.asarray(q, dtype=np.float64)
    out = {}
    for nombre, m in (("positivas", y == 1), ("negativas", y == 0)):
        a, b = np.percentile(q[m], [25, 75])
        out[nombre] = {"q1": float(a), "q3": float(b), "iqr": float(b - a)}
    return out


def vectores_m(probs: dict) -> dict:
    """{etiqueta: (m, miembros, vector)}: m=1 cada semilla, m=2 los tres pares, m=3 q_barra; promedio aritmético simple de probabilidades."""
    sem = sorted(probs)
    out = {}
    for s in sem:
        out[f"m1_s{s}"] = (1, [s], np.asarray(probs[s], dtype=np.float64))
    for a, b in combinations(sem, 2):
        out[f"m2_s{a}+s{b}"] = (2, [a, b], np.mean([probs[a], probs[b]], axis=0))
    out["m3_" + "+".join(f"s{s}" for s in sem)] = (3, sem, np.mean([probs[s] for s in sem], axis=0))
    return out


def analizar_sistema(y, fold, gpi, probs: dict, conteos) -> dict:
    """Todo el cálculo (sin E/S): agregaciones, umbral primario, tabla frente a 0.5, transferencia, S* descriptivos y m = 1, 2, 3."""
    y, fold = np.asarray(y), np.asarray(fold)
    vec = vectores_m(probs)
    q = next(v for (m, _, v) in vec.values() if m == 3)
    # b) AP y ROC-AUC con IC (mismos b)
    preps = {k: M.preparar(y, v) for k, (_, _, v) in vec.items()}
    dist = BS.distribuciones(preps, conteos, gpi)
    puntual = {k: M.calcular(p) for k, p in preps.items()}
    agreg = []
    for m in (1, 2, 3):
        claves = [k for k, (mm, _, _) in vec.items() if mm == m]
        for k in claves:
            agreg.append({"m": m, "miembros": vec[k][1], "etiqueta": k, "ap": float(puntual[k]["ap"]), "roc_auc": float(puntual[k]["roc_auc"]),
                          "ap_ic": R.ic(dist[k]["ap"]), "roc_auc_ic": R.ic(dist[k]["roc_auc"])})
        if m < 3:
            agreg.append({"m": m, "miembros": "media de las combinaciones", "etiqueta": f"m{m}_media", "ap": float(np.mean([puntual[k]["ap"] for k in claves])),
                          "roc_auc": float(np.mean([puntual[k]["roc_auc"] for k in claves])),
                          "ap_ic": R.ic(np.mean([dist[k]["ap"] for k in claves], axis=0)), "roc_auc_ic": R.ic(np.mean([dist[k]["roc_auc"] for k in claves], axis=0))})
    # c) umbral primario + bootstrap de t̂
    prim = calibracion_primaria(y, q, S_PRIMARIO)
    boot = resumen_bootstrap_umbral(umbral_ponderado(y, q, conteos, gpi, S_PRIMARIO))
    # d) tabla frente a 0.5 (q_barra y cada semilla a 0.5)
    en05, en_t = metricas_en_umbral(y, q, UMBRAL_05), metricas_en_umbral(y, q, prim["umbral"])
    diffs = {k: en_t[k] - en05[k] for k in ("sens", "esp", "fp", "fn", "vpp")}
    por_semilla_05 = {s: metricas_en_umbral(y, probs[s], UMBRAL_05) for s in sorted(probs)}
    sens_media = float(np.mean([v["sens"] for v in por_semilla_05.values()]))
    control = {"sensibilidad_media_a_0_5_calculada": sens_media, "valor_reportado_g2": CONTROL_SENS_05_G2, "coincide_a_6_decimales": round(sens_media, 6) == CONTROL_SENS_05_G2,
               "nota": "comprobación de coherencia, no es una entrada; una discrepancia se reporta y no corrige nada"}
    prevalencia = float(y.mean())
    # e) transferencia sobre q_barra (leave-one-fold-out) con t̂ recalculado en cada remuestreo
    tr_p = R.punto_operacion_puntual(y, q, fold, S_PRIMARIO)
    tr_d = R.transferencia_ponderada(y, q, fold, conteos, gpi, S_PRIMARIO)
    umbrales = np.array([float(v["umbral"]) for v in tr_p["por_fold"].values()])
    transf = {"puntual": tr_p, "intervalos": R.resumir_dist(tr_p, tr_d),
              "dispersion_t_hat_entre_folds": {"min": float(umbrales.min()), "max": float(umbrales.max()), "rango": float(np.ptp(umbrales)), "media": float(umbrales.mean()),
                                               "desv_est": float(umbrales.std(ddof=1)) if umbrales.size > 1 else float("nan")}}
    # f) S* descriptivos (orden fijo, sin ranking)
    desc = []
    for s in S_DESCRIPTIVOS:
        c = calibracion_primaria(y, q, s)
        tp_ = R.punto_operacion_puntual(y, q, fold, s)
        desc.append({"s_star": s, "calibracion": c, "bootstrap_t_hat": resumen_bootstrap_umbral(umbral_ponderado(y, q, conteos, gpi, s)),
                     "transferencia": {"puntual": tp_, "intervalos": R.resumir_dist(tp_, R.transferencia_ponderada(y, q, fold, conteos, gpi, s))}})
    # g) m = 1, 2, 3 a S* = 0.90
    filas_m = []
    for k, (m, miembros, v) in vec.items():
        c = calibracion_primaria(y, v, S_PRIMARIO)
        tp_ = R.punto_operacion_puntual(y, v, fold, S_PRIMARIO)
        filas_m.append({"etiqueta": k, "m": m, "miembros": miembros, "t_hat": c["umbral"], "k": c["k"], "n_pos": c["n_pos"], "esp_en_t_hat": c["esp"], "fp_en_t_hat": c["fp"],
                        "sens_real_en_t_hat": c["sens_real"], "sens_transferencia_agrupada": tp_["sens"], "esp_transferencia_agrupada": tp_["esp"],
                        "fp_transferencia": tp_["fp"], "fn_transferencia": tp_["fn"], "ap": float(puntual[k]["ap"]), "roc_auc": float(puntual[k]["roc_auc"]),
                        "iqr_scores": descripcion_scores(y, v)})
    return {"q_barra": q, "agregaciones": agreg, "umbral_primario": {"calibracion": prim, "bootstrap_t_hat": boot, "rotulo": NO_CONGELADO},
            "tabla_vs_0_5": {"q_barra_a_0_5": en05, "q_barra_a_t_hat": en_t, "diferencias_t_hat_menos_0_5": diffs, "por_semilla_a_0_5": por_semilla_05,
                             "prevalencia": prevalencia, "n_pos": int(y.sum()), "n": int(y.size), "nota_vpp": NOTA_VPP, "control_coherencia": control},
            "transferencia": transf, "s_star_descriptivos": desc, "m_agregaciones": filas_m}


# ============================================================================ carga de las 15 corridas A|densenet121 de G2
def config_g2_sistema(config_g2, config) -> dict:
    sp = config["procedimiento"]["sistema_provisional"]
    c = copy.deepcopy(config_g2)
    c["diseno"]["arquitecturas"], c["diseno"]["semillas"], c["diseno"]["folds"] = [sp["arquitectura"]], list(sp["semillas"]), list(sp["folds"])
    c["diseno"]["n_corridas"] = len(sp["semillas"]) * len(sp["folds"])
    return c


def cargar_entradas(root, config, config_g2, particion) -> dict:
    """SOLO las 15 corridas A|densenet121: falta o duplicado bloquea (K7); trazabilidad de cada corrida (SHA-256 de predicciones, run_config y checkpoint)."""
    g = config_g2_sistema(config_g2, config)
    raiz = Path(root) / config_g2["diseno"]["raiz_resultados"].strip("/") / "completa"
    sp = config["procedimiento"]["sistema_provisional"]
    esperadas = [id_corrida(g, sp["arquitectura"], s, k) for s in sp["semillas"] for k in sp["folds"]]
    if len(esperadas) != 15 or len(set(esperadas)) != 15:
        raise GuardError("el sistema provisional debe ser exactamente 15 corridas (3 semillas × 5 folds)")
    corridas = AG.cargar_corridas(raiz, g, particion)
    if len(corridas) != 15:
        raise GuardError(f"se esperaban 15 corridas A|densenet121 y hay {len(corridas)}")
    traz = {}
    for (a, s, k), (d, rc) in sorted(corridas.items()):
        if a != sp["arquitectura"] or rc.get("arquitectura") != sp["arquitectura"] or rc.get("condicion") != "A":
            raise GuardError(f"{rc.get('id_corrida')}: no es una corrida de la condición A con DenseNet121")
        pred = Path(d) / "predicciones_oof.csv"
        sha_pred = sha256_archivo(pred)
        if rc.get("sha256_bytes_archivo_predicciones_oof") not in (None, sha_pred):
            raise GuardError(f"{rc['id_corrida']}: predicciones_oof.csv no coincide con el SHA-256 registrado en su run_config")
        traz[rc["id_corrida"]] = {"intento": Path(d).name, "sha256_bytes_archivo_predicciones_oof": sha_pred, "sha256_bytes_archivo_run_config": sha256_archivo(Path(d) / "run_config.json"),
                                  "sha256_bytes_archivo_checkpoint_registrado": (rc.get("checkpoint") or {}).get("sha256_bytes_archivo"), "git_commit_head_corrida": rc.get("git_commit_head")}
    oof = AG.ensamblar_oof(corridas, particion, g)                                       # K7: cada imagen exactamente una vez por semilla
    probs = {s: np.asarray(oof.prob[(sp["arquitectura"], s)], dtype=np.float64) for s in sp["semillas"]}
    return {"oof": oof, "probs": probs, "trazabilidad": traz, "filas": particion.filas}


# ============================================================================ orquestación
def ejecutar(root, config, config_g2, *, autorizacion, verificar=None, n_remuestreos=None, destino=None) -> dict:
    token = exigir_autorizacion(autorizacion)                                            # antes de leer o escribir nada
    root = Path(root)
    validar_config(config)
    comprobar_coherencia(config, config_g2)
    sha_protocolo = exigir_protocolo_coherente(root)
    verificar = verificar or (lambda: verificar_integridad(root, config))
    inicio = exigir_integridad(verificar())
    r6 = verificar_r6(root, config)
    destino = destino_nuevo(root, config["salidas"]["umbral"], destino)
    particion = cargar_particion(root, config_g2)                                        # K1-K4
    ent = cargar_entradas(root, config, config_g2, particion)
    oof = ent["oof"]
    gpi, conteos, indices, n_b = indices_b(root, config, config_g2, oof.group_ids, oof.y, n_remuestreos)
    res = analizar_sistema(oof.y, oof.fold, gpi, ent["probs"], conteos)
    q = res["q_barra"]
    destino = directorio_nuevo(destino)                                                  # K5: solo tras validar y calcular
    sem = sorted(ent["probs"])
    filas_q = [{"image_id": i, "fold": int(fo), "label_v2": int(yy), **{f"p_s{s}": repr(float(ent["probs"][s][n])) for s in sem}, "q_barra": repr(float(q[n]))}
               for n, (i, fo, yy) in enumerate(zip(oof.image_ids, oof.fold, oof.y))]
    escribir_tabla(destino / "q_barra_oof.csv", filas_q, ["image_id", "fold", "label_v2", *[f"p_s{s}" for s in sem], "q_barra"],
                   "q_barra = promedio simple de las 3 predicciones OOF por imagen; ensamble de 3 modelos del mismo fold")
    sha_q = sha256_archivo(destino / "q_barra_oof.csv")
    escribir_json_nuevo(destino / "trazabilidad.json", a_json({
        "esquema": "trazabilidad_umbral_sistema_v1/1", "protocolo_id": PROTOCOLO_ID, "leyenda": LEYENDA, "fecha": ahora(), "autorizacion": token,
        "git_commit_head": inicio["git_commit_head"], "rama": inicio["rama"], "sha256_bytes_archivo_particion": particion.sha256_bytes_archivo,
        "sha256_bytes_archivo_protocolo": sha_protocolo, "base_r6": r6, "numpy_version": np.__version__, "corridas": ent["trazabilidad"],
        "sha256_bytes_archivo": {"q_barra_oof.csv": sha_q}, "un_registro_por_imagen_y_semilla": True}))
    _escribir_tablas(destino, res, sem)
    _escribir_json_umbral(destino, res)
    fin = verificar()
    dif = comparar_instantaneas(inicio, fin)
    escribir_json_nuevo(destino / "resumen_umbral.json", a_json({
        "esquema": "resumen_umbral_v1/1", "protocolo_id": PROTOCOLO_ID, "leyendas": list(LEYENDAS), "leyenda": LEYENDA, "estado_del_umbral": NO_CONGELADO,
        "s_star": {"primario": S_PRIMARIO, "estatus": "objetivo exploratorio (ya usado en G4); no es garantía de sensibilidad ni validación clínica",
                   "descriptivos_solo": list(S_DESCRIPTIVOS), "seleccion_entre_ellos": "ninguna; el código no los ordena ni destaca el más favorable"},
        "regla_positiva": "p >= umbral", "empates": "no se rompen", "sistema_provisional": PROCEDIMIENTO["sistema_provisional"],
        "discrepancia_A_vs_B": DISCREPANCIA_A_VS_B, "afirmaciones_posibles": list(AFIRMACIONES_POSIBLES), "afirmaciones_no_permitidas": list(AFIRMACIONES_NO_PERMITIDAS),
        "limites": list(LIMITES), "viabilidad_clinica": "no determinada", "criterio_numerico_de_discrepancia": None, "umbral_congelado": False,
        "n_remuestreos": n_b, "indices_b_verificacion": indices, "t_hat_primario": res["umbral_primario"]["calibracion"]["umbral_repr"],
        "control_coherencia_sens_05": res["tabla_vs_0_5"]["control_coherencia"], "diferencias_inicio_fin": dif, "invalida_integridad": bool(dif) or bool(fin["bloqueos"]),
        "reporte_aceptado": not (bool(dif) or bool(fin["bloqueos"])), "numpy_version": np.__version__,
        "sha256_bytes_archivo": {p.name: sha256_archivo(p) for p in sorted(destino.iterdir()) if p.suffix in (".csv", ".json") and p.name != "trazabilidad.json"}}))
    return {"dir": destino, "resultados": res, "reporte_aceptado": not (bool(dif) or bool(fin["bloqueos"])), "diferencias_inicio_fin": dif}


def _escribir_tablas(destino, res, sem) -> None:
    filas = []
    for a in res["agregaciones"]:
        filas.append({"m": a["m"], "miembros": " ".join(map(str, a["miembros"])) if isinstance(a["miembros"], list) else a["miembros"], "etiqueta": a["etiqueta"],
                      "ap": f(a["ap"]), "ap_ic95_inf": f(a["ap_ic"]["ic_inf"]), "ap_ic95_sup": f(a["ap_ic"]["ic_sup"]), "roc_auc": f(a["roc_auc"]),
                      "roc_auc_ic95_inf": f(a["roc_auc_ic"]["ic_inf"]), "roc_auc_ic95_sup": f(a["roc_auc_ic"]["ic_sup"])})
    escribir_tabla(destino / "agregaciones_ap_auc.csv", filas, list(filas[0]), "IC95 % bootstrap agrupado estratificado con los índices b de G2; sin afirmar mejora")
    u = res["umbral_primario"]
    c, b = u["calibracion"], u["bootstrap_t_hat"]
    fila = {"s_star": c["s_star"], "t_hat": c["umbral_repr"], "k": c["k"], "n_pos": c["n_pos"], "m_tp": c["m_tp"], "sens_real": f(c["sens_real"]), "tp": c["tp"], "fn": c["fn"],
            "fp": c["fp"], "tn": c["tn"], "esp": f(c["esp"]), "n_pos_empatadas": c["n_pos_empatadas"], "n_neg_empatadas": c["n_neg_empatadas"], "rotulo": NO_CONGELADO}
    for nom, d in b.items():
        for p in ("p2_5", "p50", "p97_5"):
            fila[f"boot_{nom}_{p}"] = f(d[p])
    escribir_tabla(destino / "umbral_primario.csv", [fila], list(fila), "t̂ recalculado en cada remuestreo (k y t̂)")
    t = res["tabla_vs_0_5"]
    filas = []
    for nombre, d in (("q_barra a 0.5", t["q_barra_a_0_5"]), ("q_barra a t_hat", t["q_barra_a_t_hat"]), *[(f"semilla {s} a 0.5", v) for s, v in t["por_semilla_a_0_5"].items()]):
        filas.append({"fila": nombre, "umbral": repr(d["umbral"]), "sens": f(d["sens"]), "esp": f(d["esp"]), "fp": d["fp"], "fn": d["fn"], "vpp": f(d["vpp"]),
                      "prevalencia": f(t["prevalencia"]), "vpp_nota": NOTA_VPP})
    dd = t["diferencias_t_hat_menos_0_5"]
    filas.append({"fila": "diferencia (t_hat - 0.5)", "umbral": "", "sens": f(dd["sens"]), "esp": f(dd["esp"]), "fp": dd["fp"], "fn": dd["fn"], "vpp": f(dd["vpp"]),
                  "prevalencia": f(t["prevalencia"]), "vpp_nota": NOTA_VPP})
    escribir_tabla(destino / "tabla_vs_0_5.csv", filas, list(filas[0]), f"prevalencia del desarrollo {t['n_pos']}/{t['n']}; {NOTA_VPP}; mover el umbral desplaza el punto sobre la misma curva")
    tr = res["transferencia"]
    filas = []
    for k, v in tr["puntual"]["por_fold"].items():
        filas.append({"fila": f"fold {k}", "t_hat_fold": v["umbral"], "tp": v["tp"], "fn": v["fn"], "fp": v["fp"], "tn": v["tn"],
                      "sens": f(v["tp"] / (v["tp"] + v["fn"]) if v["tp"] + v["fn"] else float("nan")), "esp": f(v["tn"] / (v["tn"] + v["fp"]) if v["tn"] + v["fp"] else float("nan")),
                      "vpp": f(v["tp"] / (v["tp"] + v["fp"]) if v["tp"] + v["fp"] else float("nan"))})
    p, iv = tr["puntual"], tr["intervalos"]
    ag = {"fila": "agrupada", "t_hat_fold": "", "tp": p["tp"], "fn": p["fn"], "fp": p["fp"], "tn": p["tn"], "sens": f(p["sens"]), "esp": f(p["esp"]), "vpp": f(p["vpp"])}
    for m in ("sens", "esp", "fp", "fn", "vpp"):
        ag[f"{m}_ic95_inf"], ag[f"{m}_ic95_sup"] = f(iv[m]["ic_inf"]), f(iv[m]["ic_sup"])
    filas.append(ag)
    cols = list(ag)
    escribir_tabla(destino / "transferencia.csv", [{k: r.get(k, "") for k in cols} for r in filas], cols,
                   f"{TRANSFERENCIA}; IC95 % con t̂ recalculado en cada remuestreo; {NOTA_VPP}")
    filas = []
    for d in res["s_star_descriptivos"]:
        c, b, tp_, iv = d["calibracion"], d["bootstrap_t_hat"], d["transferencia"]["puntual"], d["transferencia"]["intervalos"]
        filas.append({"s_star": d["s_star"], "t_hat": c["umbral_repr"], "k": c["k"], "n_pos": c["n_pos"], "sens_real": f(c["sens_real"]), "tp": c["tp"], "fn": c["fn"], "fp": c["fp"],
                      "tn": c["tn"], "esp": f(c["esp"]), "n_pos_empatadas": c["n_pos_empatadas"], "n_neg_empatadas": c["n_neg_empatadas"], "boot_t_hat_p2_5": f(b["t_hat"]["p2_5"]),
                      "boot_t_hat_p50": f(b["t_hat"]["p50"]), "boot_t_hat_p97_5": f(b["t_hat"]["p97_5"]), "boot_esp_p50": f(b["especificidad"]["p50"]), "boot_fp_p50": f(b["fp"]["p50"]),
                      "transf_sens": f(tp_["sens"]), "transf_sens_ic95_inf": f(iv["sens"]["ic_inf"]), "transf_sens_ic95_sup": f(iv["sens"]["ic_sup"]), "transf_esp": f(tp_["esp"]),
                      "transf_esp_ic95_inf": f(iv["esp"]["ic_inf"]), "transf_esp_ic95_sup": f(iv["esp"]["ic_sup"]), "transf_fp": tp_["fp"], "transf_fn": tp_["fn"]})
    escribir_tabla(destino / "sensibilidad_S_star.csv", filas, list(filas[0]), "S* = 0.85 y 0.95 SOLO descriptivos (sensibilidad al criterio); sin ranking, sin selección, sin destacar")
    filas = [{"etiqueta": r["etiqueta"], "m": r["m"], "miembros": " ".join(map(str, r["miembros"])), "t_hat_s090": f(r["t_hat"]), "k": r["k"], "n_pos": r["n_pos"],
              "esp_en_t_hat": f(r["esp_en_t_hat"]), "fp_en_t_hat": r["fp_en_t_hat"], "sens_real_en_t_hat": f(r["sens_real_en_t_hat"]),
              "sens_transferencia_agrupada": f(r["sens_transferencia_agrupada"]), "esp_transferencia_agrupada": f(r["esp_transferencia_agrupada"]),
              "ap": f(r["ap"]), "roc_auc": f(r["roc_auc"]), "iqr_positivas": f(r["iqr_scores"]["positivas"]["iqr"]), "q1_positivas": f(r["iqr_scores"]["positivas"]["q1"]),
              "q3_positivas": f(r["iqr_scores"]["positivas"]["q3"]), "iqr_negativas": f(r["iqr_scores"]["negativas"]["iqr"]), "q1_negativas": f(r["iqr_scores"]["negativas"]["q1"]),
              "q3_negativas": f(r["iqr_scores"]["negativas"]["q3"])} for r in res["m_agregaciones"]]
    escribir_tabla(destino / "m_agregaciones.csv", filas, list(filas[0]), "m = 1, 2, 3: descripción de cómo cambia con el número de modelos promediados; sin criterio de decisión numérico")


def _escribir_json_umbral(destino, res) -> None:
    escribir_json_nuevo(destino / "umbral_primario.json", a_json({"leyenda": LEYENDA, "rotulo": NO_CONGELADO, "umbral_primario": res["umbral_primario"],
                                                                 "tabla_vs_0_5": res["tabla_vs_0_5"], "umbral_congelado": False}))
    escribir_json_nuevo(destino / "transferencia.json", a_json({"leyenda": LEYENDA, "rotulo": TRANSFERENCIA, "transferencia": res["transferencia"], "nota_vpp": NOTA_VPP}))


def main(argv=None, *, root=None, config=None, config_g2=None, verificar=None, n_remuestreos=None, destino=None) -> int:
    ap = argparse.ArgumentParser(description="umbral_sistema_v1 (R7): calibración exploratoria; solo desarrollo; partición provisional; umbral NO congelado")
    ap.add_argument("--autorizacion", default="", help="CONFIRMADO:D-NNN de ejecución; sin ella no se lee ni se escribe nada")
    args = ap.parse_args(argv)
    root = Path(root or RAIZ_REPO).resolve()
    try:
        exigir_autorizacion(args.autorizacion)
        config = config or cargar_config(root / CONFIG_REL)
        config_g2 = config_g2 or cargar_config_g2(root / config["referencia_g2"]["config"])
        r = ejecutar(root, config, config_g2, autorizacion=args.autorizacion, verificar=verificar, n_remuestreos=n_remuestreos, destino=destino)
        print(f"{LEYENDA}\n{NO_CONGELADO}\nsalida: {r['dir'].relative_to(root).as_posix() if r['dir'].is_relative_to(root) else r['dir']}")
        if not r["reporte_aceptado"]:
            print("SALIDA NO ACEPTADA: la integridad final difiere de la inicial; no debe usarse.", file=sys.stderr)
            return CODIGO_SALIDA_INTEGRIDAD_INVALIDA
        return 0
    except GuardError as exc:
        print(f"DETENIDO: {exc}", file=sys.stderr)
        return CODIGO_SALIDA_GUARDA


if __name__ == "__main__":
    sys.exit(main())
