"""Libro mayor (ledger) append-only y guardas del presupuesto de G4 (`g4_optimizacion_v1`).

Una línea JSON por evento: fecha, fase, config_id, id_corrida, intento, evento, segundos, dispositivo, gpu, head, autorizacion, nota.
Eventos: `planificada` (corrida prevista, una vez por (config_id, id_corrida)); `reintento` (marca previa a un intento n>1 de una corrida);
`iniciada` (se registra tras pasar integridad y autorización, inmediatamente antes de crear el directorio del intento); `completada`; `fallida`.

Intentos consumidos = eventos `iniciada` (incluye fallidos, reintentos y el humo). El tope duro cuenta INTENTOS, no corridas planificadas.
Un bloqueo antes de `iniciada` (autorización, integridad, presupuesto) no cuenta ni crea directorio. Solo importa stdlib y `src.r4_v2.common`.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.r4_v2.common import GuardError

TOPE_INTENTOS = 75
PLANIFICADAS_POR_FASE = {"A": 30, "B": 20, "C": 25}
EVENTOS = ("planificada", "iniciada", "completada", "fallida", "reintento")
FASES = ("humo", "A", "B", "C")
CODIGO_SALIDA_PRESUPUESTO = 5


class PresupuestoAgotado(GuardError):
    """El tope duro de intentos (o lo posible dentro de él) rechaza el intento ANTES de crear ningún directorio."""


def ahora() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def leer_ledger(path) -> list:
    path = Path(path)
    if not path.is_file():
        return []
    eventos = []
    with open(path, encoding="utf-8") as f:
        for n, linea in enumerate(f, start=1):
            if not linea.strip():
                continue
            try:
                e = json.loads(linea)
            except ValueError:
                raise GuardError(f"ledger corrupto: línea {n} no es JSON ({path.name})") from None
            if e.get("evento") not in EVENTOS or e.get("fase") not in FASES:
                raise GuardError(f"ledger corrupto: línea {n} con evento o fase desconocidos")
            eventos.append(e)
    return eventos


def registrar(path, *, fase, config_id, id_corrida, intento, evento, head, autorizacion, segundos=None, dispositivo=None, gpu=None, nota=None) -> dict:
    """Agrega UNA línea (modo 'a', con fsync). Nunca reescribe ni borra."""
    if evento not in EVENTOS or fase not in FASES:
        raise GuardError(f"evento/fase inválidos: {evento!r}/{fase!r}")
    e = {"fecha": ahora(), "fase": fase, "config_id": config_id, "id_corrida": id_corrida, "intento": intento, "evento": evento,
         "segundos": segundos, "dispositivo": dispositivo, "gpu": gpu, "head": head, "autorizacion": autorizacion, "nota": nota}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return e


def consumidos(eventos) -> int:
    return sum(1 for e in eventos if e["evento"] == "iniciada")


def intentos_previos(eventos, config_id, id_corrida) -> int:
    return sum(1 for e in eventos if e["evento"] == "iniciada" and e["config_id"] == config_id and e["id_corrida"] == id_corrida)


def planificadas_unicas(eventos, fase=None) -> set:
    return {(e["fase"], e["config_id"], e["id_corrida"]) for e in eventos if e["evento"] == "planificada" and (fase is None or e["fase"] == fase)}


def resumen(eventos, tope: int = TOPE_INTENTOS) -> dict:
    """Resumen DERIVADO del ledger (no se almacena): planificadas, completadas, fallidas, reintentos, consumidos (= iniciadas), restante, cómputo."""
    plan = planificadas_unicas(eventos)
    iniciadas = [e for e in eventos if e["evento"] == "iniciada"]
    cerr = [e for e in eventos if e["evento"] in ("completada", "fallida")]
    humo = [e for e in iniciadas if e["fase"] == "humo"]
    c = len(iniciadas)
    return {"planificadas": len(plan), "planificadas_por_fase": {f: sum(1 for p in plan if p[0] == f) for f in ("A", "B", "C")},
            "completadas": sum(1 for e in eventos if e["evento"] == "completada" and e["fase"] != "humo"),
            "intentos_fallidos": sum(1 for e in eventos if e["evento"] == "fallida"), "reintentos": sum(1 for e in eventos if e["evento"] == "reintento"),
            "humo_consumidos": len(humo), "iniciadas_sin_cierre": c - len(cerr), "intentos_consumidos": c, "tope_intentos": tope,
            "restante": tope - c, "consumo_segundos": float(sum(e["segundos"] or 0 for e in cerr)),
            "gpus": sorted({e["gpu"] for e in cerr if e.get("gpu")}), "dispositivos": sorted({e["dispositivo"] for e in cerr if e.get("dispositivo")})}


def guardia(eventos, pendientes: int, *, tope: int = TOPE_INTENTOS, reserva: int = 0) -> None:
    """Rechaza SIN crear nada si consumidos >= tope o si consumidos + pendientes (+ reserva) excede lo posible.

    `pendientes` = corridas de la fase aún sin completar (incluida la actual); cada una necesita al menos un intento.
    """
    c = consumidos(eventos)
    if c >= tope:
        raise PresupuestoAgotado(f"tope duro de intentos alcanzado: {c} consumidos de {tope}; no se amplía sin enmienda")
    if c + pendientes + reserva > tope:
        raise PresupuestoAgotado(f"presupuesto insuficiente: {c} consumidos + {pendientes} corridas pendientes"
                                 + (f" + {reserva} reservadas" if reserva else "") + f" exceden el tope {tope} (restan {tope - c} intentos)")


def presupuesto_fase_c(eventos, tope: int = TOPE_INTENTOS) -> int:
    """Regla pre-fijada de reparto: presupuesto de C = tope - intentos consumidos al terminar B, acotado por lo planificado para C (25)."""
    return min(PLANIFICADAS_POR_FASE["C"], tope - consumidos(eventos))


def maximo_propuestas(eventos, tope: int = TOPE_INTENTOS, max_agente: int = 3) -> int:
    """Máximo efectivo de propuestas del agente: n tal que 5n + 10 <= presupuesto de C (5 folds por propuesta + 10 de confirmación), con n <= 3.
    Con A (30) + B (20) + 1 humo consumidos quedan 24 para C: máximo 2."""
    return max(0, min(max_agente, (presupuesto_fase_c(eventos, tope) - 10) // 5))


def planificar(path, eventos, fase, items, *, head, autorizacion) -> list:
    """Registra `planificada` una sola vez por (fase, config_id, id_corrida); falla si superaría lo planificado de la fase. Devuelve las nuevas."""
    previas = planificadas_unicas(eventos)
    nuevas = [(fase, i["config_id"], i["id"]) for i in items if (fase, i["config_id"], i["id"]) not in previas]
    total = sum(1 for p in previas if p[0] == fase) + len(nuevas)
    if total > PLANIFICADAS_POR_FASE[fase]:
        raise PresupuestoAgotado(f"la fase {fase} superaría sus corridas planificadas ({total} > {PLANIFICADAS_POR_FASE[fase]})")
    for f, c, r in nuevas:
        registrar(path, fase=f, config_id=c, id_corrida=r, intento=None, evento="planificada", head=head, autorizacion=autorizacion)
    return nuevas
