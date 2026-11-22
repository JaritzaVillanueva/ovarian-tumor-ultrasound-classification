"""Utilidades comunes de G3b `g3_condiciones_v1`: configuración, integridad propia, control de calidad de G3a, comparabilidad con A.

Solo IMPORTA de G2 (`src.r4_v2`) y de G3a (`src.r5_v2.comun/mascaras`); no los modifica. La receta de clasificación es la de G2
(YAML de G2 en solo lectura, fijado por blob): este módulo no define ningún hiperparámetro de entrenamiento. La única variable
declarada entre A, B y C es el origen de la imagen (imagen completa / ROI de la máscara predicha OOF / ROI de la máscara de referencia).
"""

from __future__ import annotations

import copy
import csv
import json
import re
import sys
from pathlib import Path

from src.r4_v2.entrenamiento import corrida_completada, estado_de_intento, id_corrida  # noqa: F401
from src.r4_v2.modos import FORMATO_ATESTACION  # noqa: F401

from .comun import (GuardError, LEYENDA, RAIZ_REPO, ahora, exigir_autorizacion, exigir_ruta_no_prohibida,  # noqa: F401
                    exigir_salida_segura, git_ok, git_texto, huella_archivo, ruta_menciona_test, sha256_archivo, sha256_bytes,
                    verificar_prediccion_unica)

PROTOCOLO_ID = "g3_condiciones_v1"
ESQUEMA_CONFIG = "g3_condiciones_v1_config/1"
CONFIG_REL = "configs/r5_v2/g3_condiciones_v1.yaml"
PROTOCOLO_REL = "src/r5_v2/protocolo_g3b_v1.md"
MODOS = ("recortes", "humo", "completa")
FUENTES = ("gt", "oof_pred")
# Semántica fija por nombre de condición: (fuente en entrenamiento, fuente en validación/predicción del fold k)
SEMANTICA = {"B_hibrida": ("gt", "oof_pred"), "C": ("gt", "gt"), "B_oof": ("oof_pred", "oof_pred")}
FORMATO_ACEPTACION = re.compile(r"CONFIRMADO:D-[0-9]{3}")
ARCHIVOS_CONSOLIDADOS_G3A = ("manifiesto_mascaras_oof.csv", "metricas_seg_por_imagen.csv", "resumen_metricas_seg.json")
COLUMNAS_RECORTES = ["image_id", "fold", "fuente", "ruta_recorte", "sha256_recorte", "izq", "arr", "der", "abj", "ancho_imagen", "alto_imagen",
                     "fallback", "n_componentes", "keep_largest", "margen", "ruta_mascara_origen", "sha256_mascara_origen",
                     "segmentador", "sha256_bytes_checkpoint_unet"]


# ---------------------------------------------------------------- configuración
def cargar_config_g3b(path) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise GuardError("Se requiere PyYAML para leer configs/r5_v2/g3_condiciones_v1.yaml") from exc
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    validar_config(config)
    return config


def validar_config(config) -> None:
    if not isinstance(config, dict) or config.get("esquema") != ESQUEMA_CONFIG:
        raise GuardError(f"Configuración con esquema distinto de {ESQUEMA_CONFIG!r}")
    if config["protocolo"]["id"] != PROTOCOLO_ID:
        raise GuardError("La configuración no corresponde a g3_condiciones_v1")
    if config["estudio"]["particion"] != "phash_le8_v1" or config["estudio"]["mapeo"] != "v2":
        raise GuardError("La configuración no corresponde a phash_le8_v1 / mapeo v2")
    d = config["diseno"]
    folds = d["folds"]
    if folds != list(range(len(folds))) or len(folds) < 3:
        raise GuardError(f"diseno.folds debe ser [0..n-1] con n >= 3 (recibido {folds})")
    for clave in ("raiz_resultados", "raiz_datos_derivados"):
        exigir_ruta_no_prohibida(d[clave])
    if not str(d["raiz_resultados"]).startswith("results/r5_v2/") or not str(d["raiz_datos_derivados"]).startswith("data/processed/r5_v2/"):
        raise GuardError("Las salidas de G3b deben estar bajo results/r5_v2/ y data/processed/r5_v2/ (rutas nuevas; sin reutilizar históricos ni G2/G3a)")
    if set(config["condiciones"]) != set(SEMANTICA):
        raise GuardError(f"condiciones debe ser exactamente {sorted(SEMANTICA)}")
    for nombre, c in config["condiciones"].items():
        if (c["fuente_roi_entrenamiento"], c["fuente_roi_validacion"]) != SEMANTICA[nombre]:
            raise GuardError(f"condición {nombre}: las fuentes de ROI deben ser {SEMANTICA[nombre]} (el nombre fija la semántica; otra "
                             "combinación exige otra condición con su protocolo)")
        if not c["semillas"] or any(not isinstance(s, int) for s in c["semillas"]) or not isinstance(c["habilitada"], bool):
            raise GuardError(f"condición {nombre}: semillas enteras no vacías y `habilitada` booleana")
    roi = config["roi"]
    if roi["umbral"] != 0.5 or roi["fallback_a_imagen_completa"] is not True or not isinstance(roi["keep_largest"], bool) or roi["margen"] < 0:
        raise GuardError("roi: umbral 0.5 (fijado en roi.py), fallback a imagen completa activo, keep_largest booleano y margen >= 0")
    huella = set(config["integridad"]["archivos_con_huella"])
    for rel in (CONFIG_REL, PROTOCOLO_REL, config["referencia_g2"]["config"], config["referencia_g3a"]["config"]):
        if rel not in huella:
            raise GuardError(f"integridad.archivos_con_huella debe incluir {rel}")


def comprobar_coherencia(config, config_g2, config_g3a) -> None:
    """Folds, partición, arquitecturas y semillas de G3b deben ser un subconjunto exacto de lo que fijan G2 y G3a (solo lectura)."""
    d, g2 = config["diseno"], config_g2["diseno"]
    if [int(f) for f in config_g2["particion"]["verificaciones_bloqueantes"]["folds"]] != d["folds"] or config_g3a["diseno"]["folds"] != d["folds"]:
        raise GuardError("los folds de G3b no coinciden con los de G2/G3a")
    if config["estudio"]["particion"] != config_g2["estudio"]["particion"]:
        raise GuardError("la partición de G3b no coincide con la de G2")
    if not set(d["arquitecturas"]) <= set(g2["arquitecturas"]):
        raise GuardError("arquitecturas de G3b fuera de las de G2")
    for n, c in config["condiciones"].items():
        if not set(c["semillas"]) <= set(g2["semillas"]):
            raise GuardError(f"condición {n}: semillas fuera de las de G2 {g2['semillas']}")
    if config["recortes"]["segmentador_semilla"] != config_g3a["receta"]["semilla"]:
        raise GuardError("recortes.segmentador_semilla no coincide con la semilla del U-Net de G3a")


def huella_config_efectiva(config, condicion: str) -> str:
    """SHA-256 de lo que define el insumo de una condición (no de las variables del plan: arquitecturas/semillas). Invariante ante
    cambios de plan; cambia si cambia el ROI, las fuentes, los folds, los pines de G2/G3a o el protocolo."""
    c = config["condiciones"][condicion]
    nucleo = {"condicion": condicion, "fuente_roi_entrenamiento": c["fuente_roi_entrenamiento"], "fuente_roi_validacion": c["fuente_roi_validacion"],
              "roi": config["roi"], "folds": config["diseno"]["folds"], "referencia_g2": config["referencia_g2"]["sha256_contenido_blob"],
              "referencia_g3a": config["referencia_g3a"]["sha256_contenido_blob"], "segmentador_semilla": config["recortes"]["segmentador_semilla"],
              "id_corrida": config["diseno"]["id_corrida"]}
    return sha256_bytes(json.dumps(nucleo, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def seleccionar_diseno(config, config_g2, condicion: str, arquitecturas=None, semillas=None):
    """(arquitecturas, semillas) del plan: por defecto las de la configuración; los argumentos solo pueden restringir/ampliar dentro de G2."""
    if condicion not in config["condiciones"]:
        raise GuardError(f"condición desconocida: {condicion!r} (válidas: {sorted(config['condiciones'])})")
    spec = config["condiciones"][condicion]
    if not spec["habilitada"]:
        raise GuardError(f"la condición {condicion} está deshabilitada en la configuración (`habilitada: false`); la autora decide habilitarla")
    archs = list(arquitecturas) if arquitecturas else list(config["diseno"]["arquitecturas"])
    sems = [int(s) for s in semillas] if semillas else list(spec["semillas"])
    if not set(archs) <= set(config_g2["diseno"]["arquitecturas"]) or len(set(archs)) != len(archs):
        raise GuardError(f"arquitecturas {archs} fuera del diseño de G2 o repetidas")
    if not set(sems) <= set(config_g2["diseno"]["semillas"]) or len(set(sems)) != len(sems):
        raise GuardError(f"semillas {sems} fuera de {config_g2['diseno']['semillas']} o repetidas")
    return archs, sems


def config_g2_derivada(config_g2, config, condicion: str, archs, semillas) -> dict:
    """Copia de la configuración de G2 con la condición y el diseño del plan; la receta y la partición son EXACTAMENTE las de G2."""
    c = copy.deepcopy(config_g2)
    c["estudio"]["condicion"] = condicion
    c["diseno"]["arquitecturas"], c["diseno"]["semillas"] = list(archs), list(semillas)
    c["diseno"]["folds"] = list(config["diseno"]["folds"])
    c["diseno"]["n_corridas"] = len(archs) * len(semillas) * len(config["diseno"]["folds"])
    return c


def plan_de_corridas(config_g2d) -> list:
    d = config_g2d["diseno"]
    return [{"arq": a, "semilla": s, "fold": k, "id": id_corrida(config_g2d, a, s, k)} for a in d["arquitecturas"] for s in d["semillas"] for k in d["folds"]]


# ---------------------------------------------------------------- rutas de salida
def raiz_condicion(root, config, condicion: str, modo: str) -> Path:
    base = config["diseno"]["raiz_resultados"].strip("/")
    return exigir_salida_segura(root, Path(root) / base / condicion / modo, base)


def raiz_recortes(root, config) -> Path:
    base = config["diseno"]["raiz_datos_derivados"].strip("/")
    return exigir_salida_segura(root, Path(root) / base, base)


def raiz_reporte(root, config) -> Path:
    base = config["diseno"]["raiz_resultados"].strip("/")
    return exigir_salida_segura(root, Path(root) / base / config["reporte"]["carpeta"], base)


# ---------------------------------------------------------------- aceptación de máscaras de G3a (decisión de la autora; sin umbrales)
def exigir_aceptacion_mascaras(valor) -> str:
    if not isinstance(valor, str) or not FORMATO_ACEPTACION.fullmatch(valor):
        raise GuardError("el modo recortes exige --aceptar-mascaras-g3a CONFIRMADO:D-NNN (la autora decide la calidad de las máscaras OOF tras ver "
                         "Dice/IoU/cobertura; el código no fija umbrales ni la otorga)")
    return valor


# ---------------------------------------------------------------- integridad propia (estilo G3a, con campos compatibles con r4_v2.entrenamiento)
def verificar_integridad_g3b(root, config) -> dict:
    """HEAD, árbol limpio en `integridad.rutas_sin_cambios`, SHA-256 de `archivos_con_huella` y pines de los YAML de G2/G3a.

    El diccionario es compatible con `r4_v2.entrenamiento.ejecutar_corrida` (campos `bloqueos`, `archivos[*]` con los tres hashes,
    `protocolo_versionado`, `entrada`, `rama`). NO es K6 (sin lista cerrada, sin lista dinámica de módulos; los controles de bytecode solo se
    registran). Un error de Git lanza GuardError.
    """
    ig = config["integridad"]
    head = git_texto(root, "rev-parse", "HEAD", obligatorio=True)
    rama = git_texto(root, "rev-parse", "--abbrev-ref", "HEAD")
    estado = git_ok(root, "status", "--porcelain", "--untracked-files=all", "--", *ig["rutas_sin_cambios"]).decode("utf-8", "replace")
    status = [l for l in estado.splitlines() if l.strip()]
    bloqueos = [{"codigo": "arbol_sucio", "ruta": l[3:], "detalle": l} for l in status]
    archivos = {}
    for rel in ig["archivos_con_huella"]:
        reg = huella_archivo(root, rel)
        reg["git_blob_id"] = git_texto(root, "rev-parse", f"HEAD:{rel}")
        archivos[rel] = reg
        if reg["sha256_bytes_archivo"] is None:
            bloqueos.append({"codigo": "archivo_ausente", "ruta": rel, "detalle": rel})
        elif reg["sha256_contenido_blob"] is None:
            bloqueos.append({"codigo": "archivo_no_versionado", "ruta": rel, "detalle": rel})
    for clave in ("referencia_g2", "referencia_g3a"):
        ref = config[clave]
        if archivos[ref["config"]]["sha256_contenido_blob"] != ref["sha256_contenido_blob"]:
            bloqueos.append({"codigo": f"yaml_{clave}_distinto_del_fijado", "ruta": ref["config"],
                             "detalle": f"{ref['config']}: {archivos[ref['config']]['sha256_contenido_blob']}"})
    principal = sys.modules.get("__main__")
    entrada = None if principal is None or not getattr(principal, "__file__", None) else ("modulo" if getattr(principal, "__spec__", None) else "script")
    return {"git_commit_head": head, "rama": rama, "rutas_sin_cambios": list(ig["rutas_sin_cambios"]), "status_porcelain": status, "status_aux": status,
            "archivos": archivos, "bloqueos": bloqueos, "protocolo_versionado": archivos[PROTOCOLO_REL]["sha256_contenido_blob"] is not None,
            "entrada": entrada, "modulos_en_repo": [], "modulos_virtuales_tolerados": [],
            "python": {"version": sys.version.split()[0], "dont_write_bytecode": bool(sys.flags.dont_write_bytecode),
                       "pycache_prefix": getattr(sys, "pycache_prefix", None)}}


def exigir_integridad(res: dict) -> dict:
    if res["bloqueos"]:
        raise GuardError("integridad: " + "; ".join(f"{b['codigo']}: {b.get('detalle')}" for b in res["bloqueos"][:6])
                         + (" ..." if len(res["bloqueos"]) > 6 else "") + ". No se genera ni se entrena")
    return res


# ---------------------------------------------------------------- control de la consolidación de G3a (sin escribir nada en G3a)
def _csv(path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verificar_consolidacion_g3a(root, config_g3a, particion) -> dict:
    """La consolidación de G3a `completa` debe existir, con los 5 folds `completada` válidos, 820 máscaras OOF con SHA-256 coincidente, cada
    imagen una vez y con el U-Net de su propio fold. SOLO LEE; no fija umbrales de calidad (la autora acepta con la bandera explícita)."""
    raiz = Path(root) / config_g3a["diseno"]["raiz_resultados"].strip("/") / "completa"
    ausentes = [n for n in ARCHIVOS_CONSOLIDADOS_G3A if not (raiz / n).is_file()]
    if ausentes:
        raise GuardError(f"la consolidación de G3a 'completa' no existe o está incompleta (faltan {ausentes} en {raiz.relative_to(Path(root)).as_posix()}); "
                         "no se generan recortes")
    folds = config_g3a["diseno"]["folds"]
    rcs, por_fold, ids_vistos = {}, {}, {}
    for k in folds:
        d = corrida_completada(raiz / f"fold{k}")
        if d is None:
            raise GuardError(f"G3a: el fold {k} no tiene un intento `completada`")
        rc = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
        sid = config_g3a["diseno"]["id_segmentador"].format(seed=config_g3a["receta"]["semilla"], fold=k)
        if (rc.get("estado") != "completada" or rc.get("invalida_integridad") or rc.get("modo") != "completa" or rc.get("segmentador") != sid
                or rc.get("sha256_bytes_archivo_particion") != particion.sha256_bytes_archivo):
            raise GuardError(f"G3a: el fold {k} no es una corrida `completa` válida del segmentador {sid} con la partición cargada")
        rcs[k] = rc
        por_fold[k] = _csv(d / f"manifiesto_mascaras_fold{k}.csv")
    if len({rc["git_commit_head"] for rc in rcs.values()}) != 1 or len({rc["epocas"] for rc in rcs.values()}) != 1:
        raise GuardError("G3a: los folds se obtuvieron con commits o números de época distintos")
    verificar_prediccion_unica(particion.filas, por_fold)
    consolidado = _csv(raiz / "manifiesto_mascaras_oof.csv")
    if len(consolidado) != len(particion.filas) or {m["image_id"] for m in consolidado} != {r["image_id"] for r in particion.filas}:
        raise GuardError("G3a: el manifiesto consolidado no cubre exactamente las elegibles una vez cada una")
    base = Path(root).resolve()
    manifiesto = {}
    for m in consolidado:
        k = int(m["fold"])
        if m["image_id"] in manifiesto or m not in por_fold[k] or m["sha256_bytes_checkpoint"] != rcs[k]["checkpoint"]["sha256_bytes_archivo"]:
            raise GuardError(f"G3a: {m['image_id']}: el manifiesto consolidado difiere del de su fold o del checkpoint registrado")
        ruta = (base / m["ruta_mascara"]).resolve()
        if ruta_menciona_test(m["ruta_mascara"]) or base not in ruta.parents or not ruta.is_file():
            raise GuardError(f"G3a: {m['image_id']}: ruta de máscara inválida o ausente (K4)")
        if sha256_archivo(ruta) != m["sha256_mascara"]:
            raise GuardError(f"G3a: {m['image_id']}: el SHA-256 de la máscara OOF no coincide con el manifiesto")
        manifiesto[m["image_id"]] = {**m, "ruta_abs": ruta}
    resumen = json.loads((raiz / "resumen_metricas_seg.json").read_text(encoding="utf-8"))
    sha_man = sha256_archivo(raiz / "manifiesto_mascaras_oof.csv")
    if resumen.get("sha256_texto_manifiesto_oof") != sha_man:
        raise GuardError("G3a: el resumen consolidado no corresponde al manifiesto OOF actual")
    return {"manifiesto": manifiesto, "sha256_manifiesto_oof": sha_man, "sha256_resumen": sha256_archivo(raiz / "resumen_metricas_seg.json"),
            "sha256_metricas_por_imagen": sha256_archivo(raiz / "metricas_seg_por_imagen.csv"),
            "git_commit_head_g3a": next(iter(rcs.values()))["git_commit_head"],
            "checkpoints": {k: rcs[k]["checkpoint"]["sha256_bytes_archivo"] for k in folds},
            "segmentadores": {k: rcs[k]["segmentador"] for k in folds}, "raiz": raiz}


# ---------------------------------------------------------------- comparabilidad con A (G2, solo lectura)
def _canon(x) -> str:
    return json.dumps(x, sort_keys=True, ensure_ascii=False)


def clave_comparabilidad(rc: dict) -> tuple:
    return (rc["sha256_bytes_archivo_particion"], (rc.get("referencia_pesos") or {}).get("sha256_contenido_blob"), _canon(rc["receta"]),
            rc["epocas_max"], rc["mapeo"], rc["particion"])


def comparar_con_A(root, config_g2, particion, referencia, plan, raiz_A, *, exigir: bool) -> dict:
    """Cada corrida del plan debe tener su homóloga A (misma arquitectura, semilla y fold) con la MISMA partición (`sha256_bytes_archivo`),
    la MISMA receta numérica (la del YAML de G2) y la MISMA referencia de pesos. La condición es la única variable declarada.
    Con A ausente: bloquea si `exigir`; si no, devuelve `no_disponible` (el humo puede correr sin los resultados de G2 a mano)."""
    esperado = (particion.sha256_bytes_archivo, referencia["sha256_contenido_blob"], _canon(config_g2["receta"]))
    faltan, n, blobs = [], 0, set()
    for p in plan:
        d = corrida_completada(Path(raiz_A) / id_corrida(config_g2, p["arq"], p["semilla"], p["fold"]))
        if d is None:
            faltan.append(p["id"])
            continue
        rc = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
        if rc.get("modo") != "completa" or rc.get("condicion") != "A" or rc.get("invalida_integridad") or rc.get("estado") != "completada":
            raise GuardError(f"{p['id']}: la corrida A de G2 no es una corrida `completa` válida de la condición A")
        k = clave_comparabilidad(rc)
        if k[:3] != esperado or k[3] != config_g2["receta"]["epochs_max"]:
            dif = [n_ for n_, a, b in zip(("particion", "referencia_pesos", "receta"), k[:3], esperado) if a != b]
            raise GuardError(f"{p['id']}: A no es comparable ({dif or ['epocas_max']}): misma partición, receta numérica y referencia de pesos son "
                             "condiciones de comparabilidad; la condición es la única variable declarada")
        blobs.add(rc["verificacion_inicio"]["archivos"].get("configs/r4_v2/g2_baseline_v1.yaml", {}).get("sha256_contenido_blob"))
        n += 1
    if faltan:
        if exigir:
            raise GuardError(f"faltan {len(faltan)} corridas A homólogas en G2 (p. ej. {faltan[:3]}); `completa` exige comparar con A")
        return {"estado": "no_disponible", "n_comparadas": n, "faltan": len(faltan)}
    return {"estado": "ok", "n_comparadas": n, "yaml_g2_blob_de_las_corridas_A": sorted(b for b in blobs if b),
            "sha256_bytes_archivo_particion": esperado[0], "sha256_contenido_blob_referencia_pesos": esperado[1]}


# ---------------------------------------------------------------- humo previo (misma condición, commit, YAML y partición)
def exigir_humo_previo(raiz_humo, *, condicion, sha256_bytes_archivo_particion, sha256_contenido_blob_config, git_commit_head) -> Path:
    for rc_path in sorted(Path(raiz_humo).glob("*/intento_*/run_config.json")):
        rc = json.loads(rc_path.read_text(encoding="utf-8"))
        if (rc.get("modo") == "humo" and rc.get("estado") == "completada" and not rc["invalida_integridad"] and rc.get("condicion") == condicion
                and rc["git_commit_head"] == git_commit_head and rc["sha256_bytes_archivo_particion"] == sha256_bytes_archivo_particion
                and rc["verificacion_inicio"]["archivos"][CONFIG_REL]["sha256_contenido_blob"] == sha256_contenido_blob_config):
            return rc_path.parent
    raise GuardError(f"el modo 'completa' de {condicion} exige un 'humo' completado previo de la misma condición (mismo commit, YAML y partición)")
