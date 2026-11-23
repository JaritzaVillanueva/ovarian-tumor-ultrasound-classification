"""Mecanismo del agente de la fase C de G4 (preparado; NO ejecutable con datos reales sin una D-NNN específica).

El agente solo se activa si `evidencia_fase_b.json` (emitido por el reporte; contrato abajo) contiene una técnica con 3 semillas que cumple la condición (a)
(`cumple_a`). Refina EXCLUSIVAMENTE la familia de esa técnica (S1: alpha en [0, 1]; S2: gamma en [0, 3]); no combina alpha y gamma. Cada propuesta es un
archivo `propuestas_agente/propuesta_NN.json` (NN = 01..03) con familia, valor, justificación escrita, fecha, referencia al resultado previo en el que se
basa y un hash propio. Este módulo SOLO valida y registra (el agente no ejecuta nada ni accede al test): `registro_propuestas.jsonl` (append-only) anota el
SHA-256 de cada archivo; un archivo alterado tras su registro bloquea. `conclusion_sin_mejora.json` permite detenerse sin error.

Contrato de `evidencia_fase_b.json` (lo emite el reporte de G4, PENDIENTE de actualizar): {"esquema": "evidencia_fase_b_g4/1", "configuraciones":
[{"config_id", "familia", "semillas": [42, 43, 44], "cumple_a": bool, "delta_ap_medio": float}, ...]}.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from src.r4_v2.common import GuardError, exigir_salida_segura, sha256_archivo, sha256_bytes

from . import g4_comun as C
from .g4_presupuesto import ahora

ESQUEMA_EVIDENCIA = "evidencia_fase_b_g4/1"
NOMBRE_PROPUESTA = re.compile(r"propuesta_(0[1-9]|[1-9][0-9])\.json")
CAMPOS_PROPUESTA = ("familia", "valor", "justificacion", "fecha", "referencia_resultado_previo", "sha256_propuesta")
CONCLUSION = "conclusion_sin_mejora.json"
REGISTRO = "registro_propuestas.jsonl"
SEMILLAS_REQUERIDAS = [42, 43, 44]


def _canon(d) -> bytes:
    return json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")


def hash_de_propuesta(d: dict) -> str:
    """SHA-256 del contenido canónico SIN el campo `sha256_propuesta` (el agente lo calcula con esta misma función)."""
    return sha256_bytes(_canon({k: v for k, v in d.items() if k != "sha256_propuesta"}))


# ---------------------------------------------------------------- activación
def leer_evidencia_fase_b(path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise GuardError(f"el agente no se activa: falta {path.name} (resultado de la fase B con 3 semillas)")
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        assert d["esquema"] == ESQUEMA_EVIDENCIA
        for c in d["configuraciones"]:
            assert c["familia"] in C.LIMITES_FAMILIA and isinstance(c["cumple_a"], bool) and isinstance(c["semillas"], list)
            float(c["delta_ap_medio"])
    except (ValueError, KeyError, TypeError, AssertionError):
        raise GuardError(f"{path.name}: no cumple el contrato {ESQUEMA_EVIDENCIA}") from None
    return d


def familia_seleccionada(evidencia: dict) -> dict:
    """Técnica con 3 semillas que cumple (a): mayor ΔAP medio; empate por el orden de la lista. Sin ninguna, el agente NO se activa."""
    aptas = [(i, c) for i, c in enumerate(evidencia["configuraciones"]) if c["cumple_a"] and sorted(c["semillas"]) == SEMILLAS_REQUERIDAS]
    if not aptas:
        raise GuardError("ninguna técnica cumple (a) con 3 semillas: el agente no se activa (se conserva la base)")
    i, c = min(aptas, key=lambda t: (-float(t[1]["delta_ap_medio"]), t[0]))
    return {"config_id": c["config_id"], "familia": c["familia"], "delta_ap_medio": float(c["delta_ap_medio"])}


# ---------------------------------------------------------------- propuestas
def leer_registro(carpeta) -> list:
    p = Path(carpeta) / REGISTRO
    if not p.is_file():
        return []
    try:
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    except ValueError:
        raise GuardError(f"{REGISTRO} corrupto") from None


def _anotar(carpeta, entrada: dict) -> None:
    with open(Path(carpeta) / REGISTRO, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entrada, ensure_ascii=False, sort_keys=True) + "\n")


def _validar_referencia(root, ref) -> None:
    if not isinstance(ref, dict) or not isinstance(ref.get("archivo"), str) or not re.fullmatch(r"[0-9a-f]{64}", str(ref.get("sha256_bytes_archivo", ""))):
        raise GuardError("referencia_resultado_previo debe ser {archivo, sha256_bytes_archivo} con SHA-256 de 64 hex en minúsculas")
    ruta = exigir_salida_segura(root, Path(root) / ref["archivo"], "results")           # K4: bajo results/, sin componente test
    if not ruta.is_file() or sha256_archivo(ruta) != ref["sha256_bytes_archivo"]:
        raise GuardError(f"referencia_resultado_previo: {ref['archivo']} no existe o su SHA-256 no coincide")


def validar_propuestas(root, carpeta, familia: str, *, max_propuestas: int = 3) -> list:
    """Valida TODAS las propuestas de `carpeta` sin escribir nada. Devuelve [{archivo, estrategia, sha256_bytes_archivo, datos}] en orden NN.

    Reglas: <= 3, numeración contigua desde 01, familia = la seleccionada, valor dentro de límites y de 2 decimales, no duplicado (ni con la base,
    ni con las seis de la fase A, ni entre sí, ni con otra propuesta ya registrada), justificación escrita, hash propio válido y archivo sin cambios
    desde su registro. No se combinan familias.
    """
    carpeta = Path(carpeta)
    archivos = sorted(p for p in carpeta.glob("propuesta_*.json")) if carpeta.is_dir() else []
    nombres = [p.name for p in archivos]
    if any(not NOMBRE_PROPUESTA.fullmatch(n) for n in nombres):
        raise GuardError(f"nombres de propuesta inválidos (se espera propuesta_NN.json): {nombres}")
    if len(archivos) > max_propuestas:
        raise GuardError(f"hay {len(archivos)} propuestas (máximo {max_propuestas})")
    if nombres != [f"propuesta_{i:02d}.json" for i in range(1, len(nombres) + 1)]:
        raise GuardError(f"la numeración debe ser contigua desde 01: {nombres}")
    registro = {r["archivo"]: r for r in leer_registro(carpeta) if r.get("evento") == "propuesta"}
    ausentes = sorted(set(registro) - set(nombres))
    if ausentes:
        raise GuardError(f"propuestas registradas que ya no están en la carpeta: {ausentes}; se bloquea (el registro es append-only)")
    vistos = {e[0] for e in C.ESTRATEGIAS_FASE_A}
    salida = []
    for p in archivos:
        sha = sha256_archivo(p)
        previo = registro.get(p.name)
        if previo is not None and previo["sha256_bytes_archivo"] != sha:
            raise GuardError(f"{p.name}: el archivo cambió tras su registro (SHA-256 distinto del anotado); se bloquea")
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            raise GuardError(f"{p.name}: no es JSON") from None
        faltan = [c for c in CAMPOS_PROPUESTA if c not in d]
        if faltan or set(d) - set(CAMPOS_PROPUESTA):
            raise GuardError(f"{p.name}: campos faltantes {faltan} o no permitidos {sorted(set(d) - set(CAMPOS_PROPUESTA))} (no se combinan familias ni parámetros)")
        if d["familia"] != familia:
            raise GuardError(f"{p.name}: la familia {d['familia']!r} no es la seleccionada ({familia}); solo se refina esa familia")
        if not isinstance(d["justificacion"], str) or len(d["justificacion"].strip()) < 20:
            raise GuardError(f"{p.name}: falta la justificación escrita (>= 20 caracteres)")
        try:
            datetime.strptime(d["fecha"], "%Y-%m-%d")
        except (ValueError, TypeError):
            raise GuardError(f"{p.name}: fecha inválida (YYYY-MM-DD)") from None
        if d["sha256_propuesta"] != hash_de_propuesta(d):
            raise GuardError(f"{p.name}: sha256_propuesta no coincide con el contenido")
        _validar_referencia(root, d["referencia_resultado_previo"])
        estr = C.estrategia_de(familia, d["valor"])                                        # límites de la familia y 2 decimales
        if d["valor"] == C.VALOR_DE_LA_BASE[familia]:
            raise GuardError(f"{p.name}: el valor {d['valor']} equivale a la base (duplicado)")
        if estr["id"] in vistos:
            raise GuardError(f"{p.name}: {estr['id']} duplicado respecto a lo ya definido o evaluado")
        vistos.add(estr["id"])
        salida.append({"archivo": p.name, "estrategia": estr, "sha256_bytes_archivo": sha, "datos": d, "registrada": previo is not None})
    return salida


def registrar_propuestas(carpeta, validadas, *, head, autorizacion) -> int:
    """Anota (append-only) las propuestas aún no registradas con su SHA-256. Devuelve cuántas anotó."""
    n = 0
    for v in validadas:
        if not v["registrada"]:
            _anotar(carpeta, {"evento": "propuesta", "fecha": ahora(), "archivo": v["archivo"], "sha256_bytes_archivo": v["sha256_bytes_archivo"],
                              "config_id": v["estrategia"]["id"], "familia": v["estrategia"]["familia"], "valor": v["datos"]["valor"],
                              "head": head, "autorizacion": autorizacion})
            n += 1
    return n


def leer_conclusion(root, carpeta):
    """`conclusion_sin_mejora.json` (si existe): {fecha, justificacion, referencia_resultado_previo}. Anotarla no es un error."""
    p = Path(carpeta) / CONCLUSION
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        datetime.strptime(d["fecha"], "%Y-%m-%d")
        assert isinstance(d["justificacion"], str) and len(d["justificacion"].strip()) >= 20
    except (ValueError, KeyError, TypeError, AssertionError):
        raise GuardError(f"{CONCLUSION}: requiere fecha (YYYY-MM-DD), justificación escrita y referencia_resultado_previo") from None
    _validar_referencia(root, d.get("referencia_resultado_previo"))
    return {"archivo": p.name, "sha256_bytes_archivo": sha256_archivo(p), "datos": d}


def anotar_conclusion(carpeta, conclusion, *, head, autorizacion) -> bool:
    if any(r.get("evento") == "conclusion_sin_mejora" and r["sha256_bytes_archivo"] == conclusion["sha256_bytes_archivo"] for r in leer_registro(carpeta)):
        return False
    _anotar(carpeta, {"evento": "conclusion_sin_mejora", "fecha": ahora(), "archivo": conclusion["archivo"],
                      "sha256_bytes_archivo": conclusion["sha256_bytes_archivo"], "head": head, "autorizacion": autorizacion})
    return True
