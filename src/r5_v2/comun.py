"""Utilidades comunes de G3a: configuración, roles por fold, integridad sencilla, autorización y rutas de salida.

Reutiliza `src.r4_v2` (solo importa): `GuardError`, guardas K4/K5, hashes, Git de solo lectura, escritura que no sobrescribe y los
intentos por directorio. NO reutiliza `r4_v2.integridad.verificar_integridad` (su lista cerrada es de G2): la comprobación de
esta puerta es `verificar_integridad_g3a` (ver diferencias con K6 en su docstring).
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from src.r4_v2.common import (RAIZ_REPO, GuardError, directorio_nuevo, escribir_csv_nuevo, escribir_json_nuevo,  # noqa: F401
                              exigir_ruta_no_prohibida, exigir_salida_segura, git, git_ok, git_texto, ruta_menciona_test,
                              sha256_archivo, sha256_bytes)
from src.r4_v2.entrenamiento import corrida_completada, crear_intento, estado_de_intento, intentos_de  # noqa: F401
from src.r4_v2.modos import FORMATO_ATESTACION

PROTOCOLO_ID = "g3_segmentacion_v1"
ESQUEMA_CONFIG = "g3_segmentacion_v1_config/1"
CONFIG_REL = "configs/r5_v2/g3_segmentacion_v1.yaml"
PROTOCOLO_REL = "src/r5_v2/protocolo_g3a_v1.md"
PARTICION_ID = "phash_le8_v1"
MAPEO_ID = "mapeo v2"
MODOS = ("verificar_datos", "humo", "completa")
LEYENDA = "partición provisional (phash_le8_v1, exploratoria, post hoc); conformidad académica pendiente"
ROL_ENTRENAMIENTO, ROL_VALIDACION_INTERNA, ROL_PREDICCION = "entrenamiento", "validacion_interna", "prediccion_fold_k"


def ahora() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- configuración
def cargar_config_g3a(path) -> dict:
    """Lee el YAML de G3a y valida esquema, identificadores y los parámetros que el código no puede cambiar."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise GuardError("Se requiere PyYAML para leer configs/r5_v2/g3_segmentacion_v1.yaml") from exc
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    validar_config(config)
    return config


def validar_config(config) -> None:
    if not isinstance(config, dict) or config.get("esquema") != ESQUEMA_CONFIG:
        raise GuardError(f"Configuración con esquema distinto de {ESQUEMA_CONFIG!r}")
    if config["protocolo"]["id"] != PROTOCOLO_ID:
        raise GuardError("La configuración no corresponde a g3_segmentacion_v1")
    if config["estudio"]["particion"] != PARTICION_ID or config["estudio"]["mapeo"] != "v2":
        raise GuardError("La configuración no corresponde a phash_le8_v1 / mapeo v2")
    d, r, roi = config["diseno"], config["receta"], config["roi"]
    folds = d["folds"]
    if folds != list(range(len(folds))) or len(folds) < 3:
        raise GuardError(f"diseno.folds debe ser [0..n-1] con n >= 3 (recibido {folds})")
    for clave in ("raiz_resultados", "raiz_datos_derivados", "imagenes_dir"):
        exigir_ruta_no_prohibida(d[clave])
    if not str(d["raiz_resultados"]).startswith("results/r5_v2/") or not str(d["raiz_datos_derivados"]).startswith("data/processed/r5_v2/"):
        raise GuardError("Las rutas de salida de G3a deben estar bajo results/r5_v2/ y data/processed/r5_v2/ (rutas nuevas; sin reutilizar históricos)")
    if r["binarizacion_referencia"] != "> 0" or not isinstance(r["semilla"], int):
        raise GuardError("receta: binarización '> 0' y semilla entera obligatorias")
    if roi["umbral"] != 0.5:
        raise GuardError("roi.umbral debe ser 0.5: src/segmentation/roi.py lo fija y no se edita")
    if CONFIG_REL not in config["integridad"]["archivos_con_huella"] or PROTOCOLO_REL not in config["integridad"]["archivos_con_huella"]:
        raise GuardError("integridad.archivos_con_huella debe incluir la configuración y el protocolo de G3a")
    h = d["modos"]["humo"]
    if not 1 <= h["epocas_min"] <= h["epocas_max"]:
        raise GuardError("diseno.modos.humo: rango de épocas inválido")


def comprobar_coherencia_con_g2(config, config_g2) -> None:
    """La partición y las carpetas de imágenes de G3a son exactamente las de G2 (solo lectura del YAML congelado de G2)."""
    vb = config_g2["particion"]["verificaciones_bloqueantes"]
    if [int(f) for f in vb["folds"]] != list(config["diseno"]["folds"]):
        raise GuardError("los folds de G3a no coinciden con los de G2")
    if config["diseno"]["imagenes_dir"].strip("/") != config_g2["diseno"]["imagenes_dir"].strip("/"):
        raise GuardError("imagenes_dir de G3a no coincide con el de G2")
    if config["estudio"]["particion"] != config_g2["estudio"]["particion"]:
        raise GuardError("la partición de G3a no coincide con la de G2")


def raiz_del_modo(root, config, modo: str) -> Path:
    base = config["diseno"]["raiz_resultados"].strip("/")
    return exigir_salida_segura(root, Path(root) / base / modo, base)


def raiz_datos_del_modo(root, config, modo: str) -> Path:
    base = config["diseno"]["raiz_datos_derivados"].strip("/")
    return exigir_salida_segura(root, Path(root) / base / modo, base)


def segmentador_id(config, fold: int) -> str:
    return config["diseno"]["id_segmentador"].format(seed=config["receta"]["semilla"], fold=fold)


# ---------------------------------------------------------------- roles por fold (sin fuga)
def fold_validacion_interna(folds, k: int) -> int:
    folds = sorted(folds)
    if k not in folds:
        raise GuardError(f"fold {k} fuera de {folds}")
    return folds[(folds.index(k) + 1) % len(folds)]


def roles_de_fold(filas, k: int, folds) -> dict:
    """Entrenamiento = folds fuera de {k, v}; validación interna = fold v = (k+1) mod n (solo elige checkpoint); predicción = fold k."""
    v = fold_validacion_interna(folds, k)
    roles = {"fold": k, "fold_validacion_interna": v,
             "entrenamiento": [r for r in filas if r["fold"] not in (k, v)],
             "validacion_interna": [r for r in filas if r["fold"] == v],
             "prediccion": [r for r in filas if r["fold"] == k]}
    verificar_roles(roles, filas)
    return roles


def verificar_roles(roles: dict, filas) -> None:
    """Invariantes de no-fuga: (entrenamiento ∪ validación interna) ∩ fold k = ∅; roles disjuntos; cobertura exacta."""
    k, v = roles["fold"], roles["fold_validacion_interna"]
    ent = [r["image_id"] for r in roles["entrenamiento"]]
    val = [r["image_id"] for r in roles["validacion_interna"]]
    pred = [r["image_id"] for r in roles["prediccion"]]
    if not ent or not val or not pred:
        raise GuardError(f"fold {k}: algún rol está vacío")
    for nombre, ids in (("entrenamiento", ent), ("validacion_interna", val), ("prediccion", pred)):
        if len(set(ids)) != len(ids):
            raise GuardError(f"fold {k}: image_id repetidos en {nombre}")
    if k == v:
        raise GuardError(f"fold {k}: el fold interno coincide con el fold k")
    if set(ent) & set(val) or (set(ent) | set(val)) & set(pred):
        raise GuardError(f"fold {k}: fuga de información: solapamiento entre entrenamiento/validación interna y fold {k}")
    if set(ent) | set(val) | set(pred) != {r["image_id"] for r in filas}:
        raise GuardError(f"fold {k}: los roles no cubren exactamente las imágenes de la partición")
    if any(r["fold"] in (k, v) for r in roles["entrenamiento"]) or any(r["fold"] != v for r in roles["validacion_interna"]) \
            or any(r["fold"] != k for r in roles["prediccion"]):
        raise GuardError(f"fold {k}: filas con fold incoherente con su rol")


def verificar_prediccion_unica(filas, manifiestos_por_fold: dict) -> None:
    """Cada imagen recibe exactamente una predicción, del segmentador de su propio fold."""
    visto = {}
    por_id = {r["image_id"]: r for r in filas}
    for k, manifiesto in manifiestos_por_fold.items():
        for m in manifiesto:
            i = m["image_id"]
            if i in visto:
                raise GuardError(f"{i}: predicha más de una vez (folds {visto[i]} y {k})")
            if i not in por_id:
                raise GuardError(f"{i}: predicha pero no pertenece a las elegibles")
            if por_id[i]["fold"] != int(k) or int(m["fold"]) != int(k):
                raise GuardError(f"{i}: predicha por el segmentador del fold {k} pero pertenece al fold {por_id[i]['fold']}")
            visto[i] = k
    if set(visto) != set(por_id):
        raise GuardError(f"faltan predicciones para {len(set(por_id) - set(visto))} imágenes")


def hash_de_ids(ids) -> str:
    return sha256_bytes("\n".join(sorted(str(i) for i in ids)).encode("utf-8"))


# ---------------------------------------------------------------- autorización
def exigir_autorizacion(valor) -> str:
    """Mismo formato que G2 (`r4_v2.modos.FORMATO_ATESTACION['autorizacion_ejecucion']`): CONFIRMADO:AUT-NNN o CONFIRMADO:D-NNN."""
    patron = FORMATO_ATESTACION["autorizacion_ejecucion"]
    if not isinstance(valor, str) or not patron.fullmatch(valor):
        raise GuardError(f"humo y completa exigen --autorizacion-ejecucion con formato {patron.pattern!r}; no se otorga ni se infiere")
    return valor


def validar_auditoria_opcional(valor):
    """Atestación de auditoría independiente: OPCIONAL en G3a (la especificación solo exige la autorización); si se da, formato G2."""
    if valor in (None, ""):
        return None
    patron = FORMATO_ATESTACION["auditoria_independiente"]
    if not isinstance(valor, str) or not patron.fullmatch(valor):
        raise GuardError(f"--auditoria-independiente, si se da, debe cumplir {patron.pattern!r}")
    return valor


# ---------------------------------------------------------------- integridad sencilla (≠ K6 de G2)
def huella_archivo(root, rel: str) -> dict:
    ruta = Path(root) / rel
    reg = {"sha256_bytes_archivo": None, "sha256_contenido_blob": None}
    if ruta.is_file():
        reg["sha256_bytes_archivo"] = sha256_bytes(ruta.read_bytes())
    r = git(root, "cat-file", "blob", f"HEAD:{rel}")
    if r.returncode == 0:
        reg["sha256_contenido_blob"] = sha256_bytes(r.stdout)
    return reg


def verificar_integridad_g3a(root, config) -> dict:
    """Comprobación propia y sencilla. Registra `git rev-parse HEAD`, exige árbol limpio (`git status --porcelain` vacío) en las
    rutas de `integridad.rutas_sin_cambios` y registra SHA-256 de los archivos de `integridad.archivos_con_huella`.

    Diferencias con K6 de G2: sin lista cerrada ni lista dinámica de módulos (`sys.modules`); sin comparación archivo a archivo con
    el blob de HEAD (la limpieza del árbol la sustituye); sin tolerancia explícita CRLF; sin control de banderas de bytecode (solo se
    registran); sin control de protocolo versionado/excepción. Bloquea además si un archivo con huella no existe o no está en HEAD, o
    si el YAML de G2 no coincide con el SHA-256 de su blob fijado en la configuración. Un error de Git lanza GuardError.
    """
    ig = config["integridad"]
    head = git_texto(root, "rev-parse", "HEAD", obligatorio=True)
    rama = git_texto(root, "rev-parse", "--abbrev-ref", "HEAD")
    estado = git_ok(root, "status", "--porcelain", "--untracked-files=all", "--", *ig["rutas_sin_cambios"]).decode("utf-8", "replace")
    status = [l for l in estado.splitlines() if l.strip()]
    bloqueos = [{"codigo": "arbol_sucio", "detalle": l} for l in status]
    archivos = {}
    for rel in ig["archivos_con_huella"]:
        archivos[rel] = huella_archivo(root, rel)
        if archivos[rel]["sha256_bytes_archivo"] is None:
            bloqueos.append({"codigo": "archivo_ausente", "detalle": rel})
        elif archivos[rel]["sha256_contenido_blob"] is None:
            bloqueos.append({"codigo": "archivo_no_versionado", "detalle": rel})
    ref = config["referencia_g2"]
    g2 = archivos.get(ref["config"]) or huella_archivo(root, ref["config"])
    if g2["sha256_contenido_blob"] != ref["sha256_contenido_blob"]:
        bloqueos.append({"codigo": "yaml_g2_distinto_del_fijado", "detalle": f"{ref['config']}: {g2['sha256_contenido_blob']}"})
    return {"git_commit_head": head, "rama": rama, "rutas_sin_cambios": list(ig["rutas_sin_cambios"]), "status_porcelain": status,
            "archivos": archivos, "bloqueos": bloqueos,
            "python": {"version": sys.version.split()[0], "dont_write_bytecode": bool(sys.flags.dont_write_bytecode),
                       "pycache_prefix": getattr(sys, "pycache_prefix", None)}}


def exigir_integridad(res: dict) -> dict:
    if res["bloqueos"]:
        raise GuardError("integridad: " + "; ".join(f"{b['codigo']}: {b['detalle']}" for b in res["bloqueos"][:6])
                         + (" ..." if len(res["bloqueos"]) > 6 else "") + ". No se entrena")
    return res


def comparar_instantaneas(inicio: dict, fin: dict) -> list:
    dif = []
    if inicio["git_commit_head"] != fin["git_commit_head"]:
        dif.append("git_commit_head")
    if inicio["status_porcelain"] != fin["status_porcelain"] or fin["bloqueos"]:
        dif.append("status_porcelain")
    for rel, reg in inicio["archivos"].items():
        if fin["archivos"].get(rel, {}).get("sha256_bytes_archivo") != reg["sha256_bytes_archivo"]:
            dif.append(f"archivo:{rel}")
    return dif
