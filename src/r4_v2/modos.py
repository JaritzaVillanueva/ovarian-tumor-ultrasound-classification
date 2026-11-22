"""Modos de ejecución (protocolo §4.7): `referencia` -> `humo` -> `completa` (orden fijo).

Las precondiciones que el código NO puede comprobar (conformidad real del asesor, auditoría independiente, autorización de
ejecución del autor) se exigen como ATESTACIONES explícitas que se registran en `run_config`; el código no las verifica ni las
otorga. La precondición (2) y la partición (3) no se exigen a `referencia` (propuesta [P] del protocolo; X10: el autor puede exigirlas).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from . import integridad as I
from . import pesos as W
from .common import (GuardError, LEYENDA_PROVISIONAL, MAPEO_ID, MODOS, PARTICION_ID, PROTOCOLO_ID, CONFIG_REL,
                     escribir_json_nuevo, exigir_salida_segura)
from .entrenamiento import crear_intento, estado_de_intento, id_corrida

# Formato de una atestación POSITIVA del autor (documentado aquí y probado): `CONFIRMADO:<PREFIJO>-<NNN>`.
#   conformidad_asesor      -> CONFIRMADO:ASE-NNN  (registro de la conformidad REAL del asesor académico)
#   auditoria_independiente -> CONFIRMADO:AUD-NNN  (registro de la auditoría independiente de la implementación)
#   autorizacion_ejecucion  -> CONFIRMADO:AUT-NNN  o CONFIRMADO:D-NNN (autorización explícita del autor / decisión de trazabilidad)
# NNN = exactamente tres dígitos. Sensible a mayúsculas, sin espacios ni normalización: «pendiente», «PENDIENTE», «no», «sí», vacíos,
# `CONFIRMADO` sin identificador y cualquier otra cadena se rechazan. El código NO puede verificar que el registro exista ni que la
# conformidad sea real: solo exige y registra la atestación del autor; la comprobación del registro es del autor y del auditor.
FORMATO_ATESTACION = {
    "conformidad_asesor": re.compile(r"CONFIRMADO:ASE-[0-9]{3}"),
    "auditoria_independiente": re.compile(r"CONFIRMADO:AUD-[0-9]{3}"),
    "autorizacion_ejecucion": re.compile(r"CONFIRMADO:(?:AUT|D)-[0-9]{3}"),
}

ATESTACIONES = {
    "referencia": ("auditoria_independiente", "autorizacion_ejecucion"),
    "humo": ("conformidad_asesor", "auditoria_independiente", "autorizacion_ejecucion"),
    "completa": ("conformidad_asesor", "auditoria_independiente", "autorizacion_ejecucion"),
}


def exigir_atestaciones(modo: str, atestaciones: dict) -> dict:
    if modo not in MODOS:
        raise GuardError(f"modo desconocido: {modo!r} (válidos: {MODOS})")
    invalidas = []
    for a in ATESTACIONES[modo]:
        valor = (atestaciones or {}).get(a)
        if not isinstance(valor, str) or not FORMATO_ATESTACION[a].fullmatch(valor):
            invalidas.append(a)
    if invalidas:
        formatos = {a: FORMATO_ATESTACION[a].pattern for a in invalidas}
        raise GuardError(f"precondiciones del modo '{modo}' sin atestación positiva válida del autor: {invalidas}. Formato exigido "
                         f"{formatos}. No se otorgan ni se infieren (la conformidad del asesor sigue PENDIENTE hasta que conste de "
                         "forma real)")
    return {a: atestaciones[a] for a in ATESTACIONES[modo]}


def raiz_del_modo(root, config, modo: str, crear: bool = True) -> Path:
    """`results/r4_v2/phash_le8_v1/g2_baseline_v1/<modo>/` (nunca bajo results/classification; K5)."""
    base = config["diseno"]["raiz_resultados"].strip("/")
    destino = exigir_salida_segura(root, Path(root) / base / modo, base)
    if "classification" in destino.parts:
        raise GuardError("K5: los resultados de G2 no se escriben bajo results/classification/")
    if crear:
        destino.mkdir(parents=True, exist_ok=True)
    return destino


def plan_de_corridas(modo: str, config, *, humo_arq=None, humo_semilla=None, humo_fold=None) -> list:
    """Corridas del modo. `referencia`: ninguna (no entrena). `humo`: una arquitectura, una semilla y un fold. `completa`: 45."""
    d = config["diseno"]
    if modo == "referencia":
        return []
    if modo == "humo":
        a = humo_arq or d["arquitecturas"][0]
        s = humo_semilla if humo_semilla is not None else d["semillas"][0]
        k = humo_fold if humo_fold is not None else d["folds"][0]
        if a not in d["arquitecturas"] or s not in d["semillas"] or k not in d["folds"]:
            raise GuardError("el humo debe usar una arquitectura, semilla y fold del diseño")
        return [{"arq": a, "semilla": s, "fold": k, "id": id_corrida(config, a, s, k)}]
    if modo == "completa":
        plan = [{"arq": a, "semilla": s, "fold": k, "id": id_corrida(config, a, s, k)}
                for a in d["arquitecturas"] for s in d["semillas"] for k in d["folds"]]
        if len(plan) != d["n_corridas"]:
            raise GuardError(f"el plan tiene {len(plan)} corridas (configuración: {d['n_corridas']})")
        return plan
    raise GuardError(f"modo desconocido: {modo!r}")


def epocas_humo(config, solicitadas) -> int:
    h = config["diseno"]["modos"]["humo"]
    n = int(solicitadas)
    if not h["epocas_min"] <= n <= h["epocas_max"]:
        raise GuardError(f"el humo admite {h['epocas_min']}-{h['epocas_max']} épocas (recibido {n})")
    return n


def exigir_humo_previo(raiz_humo, *, sha256_bytes_archivo_particion, sha256_contenido_blob_config, sha256_contenido_blob_referencia) -> Path:
    """`completa` exige al menos un `humo` sin fallo técnico previo, con la misma partición, configuración y referencia [P]."""
    for rc_path in sorted(Path(raiz_humo).glob("*/intento_*/run_config.json")):
        rc = json.loads(rc_path.read_text(encoding="utf-8"))
        if (rc.get("modo") == "humo" and estado_de_intento(rc_path.parent) == "completada" and not rc["invalida_integridad"]
                and rc["sha256_bytes_archivo_particion"] == sha256_bytes_archivo_particion
                and rc["verificacion_inicio"]["archivos"][CONFIG_REL]["sha256_contenido_blob"] == sha256_contenido_blob_config
                and (rc.get("referencia_pesos") or {}).get("sha256_contenido_blob") == sha256_contenido_blob_referencia):
            return rc_path.parent
    raise GuardError("el modo 'completa' exige un 'humo' previo sin fallo técnico (misma partición, configuración y referencia)")


# ---------------------------------------------------------------- modo referencia (sin entrenamiento, sin datos)
def ejecutar_referencia(root, config, *, verificar, instanciar, resolver, versiones, raiz_modo, atestaciones=None) -> dict:
    """Instancia las tres arquitecturas, registra su procedencia (§4.6) y escribe el archivo CANDIDATO de referencia.

    NO lee imágenes ni el CSV de partición, NO crea optimizador, pérdida ni DataLoader, NO produce probabilidades, checkpoints ni
    métricas y NO escribe en `configs/` (el autor verifica el candidato y lo copia byte a byte).
    """
    inicio = verificar()
    I.exigir_integridad(inicio)
    registros, estado, motivo = {}, "completada", None
    dir_intento = crear_intento(raiz_modo)
    try:
        for arq in config["diseno"]["arquitecturas"]:
            modelo = instanciar(arq)
            registros[arq] = W.procedencia_tras_instanciar(arq, modelo, pretrained=config["receta"]["pretrained"], resolver=resolver)
            del modelo
        v = versiones()
        candidato = W.construir_candidato(registros, v, datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        escribir_json_nuevo(dir_intento / "referencia_pesos_preentrenados_v1.candidato.json", candidato)
    except GuardError as exc:
        estado, motivo, v = ("detenida_k8" if str(exc).startswith("K8") else "fallo_tecnico"), str(exc), {}
    fin = verificar()
    dif = I.comparar_instantaneas(inicio, fin)
    if (dif or fin["bloqueos"]) and estado == "completada":
        estado = "invalida_integridad"
    try:                                                      # solo se registra si ya existe una referencia (K8 no compara aquí)
        existente = W.cargar_referencia(root)
        coincide = all(existente["datos"]["arquitecturas"][a][c] == registros[a][c]
                       for a in registros for c in ("enum_pesos", "url", "tamano_bytes", "sha256_bytes_archivo"))
    except GuardError:
        coincide = None
    escribir_json_nuevo(dir_intento / "run_config.json", {
        "esquema": "run_config_g2_baseline_v1/1", "protocolo_id": PROTOCOLO_ID, "mapeo": MAPEO_ID, "particion": PARTICION_ID,
        "leyenda": LEYENDA_PROVISIONAL, "modo": "referencia", "estado": estado, "motivo": motivo,
        "git_commit_head": inicio["git_commit_head"], "rama": inicio["rama"], "protocolo_versionado": inicio["protocolo_versionado"],
        "excepcion_protocolo": inicio["excepcion_protocolo"], "entrada": inicio["entrada"],
        "pesos_preentrenados": registros, **v,
        "referencia_existente_coincide": coincide, "atestaciones_del_autor": atestaciones or {},
        "verificacion_inicio": inicio, "verificacion_fin": fin, "diferencias_inicio_fin": dif,
        "invalida_integridad": bool(dif or fin["bloqueos"]), "sin_metricas": True})
    return {"estado": estado, "motivo": motivo, "dir": dir_intento}
