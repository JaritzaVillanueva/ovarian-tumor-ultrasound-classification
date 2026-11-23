"""Ejecutor de G4 `g4_optimizacion_v1` por fases: humo, fase_a, promocion, fase_b, fase_c, fase_c_confirmacion.

Uso (solo por la autora, en Falcon; el agente que lo escribió NO lo ejecuta; NINGÚN modo se lanza sin una D-NNN específica por fase):

    python -B -X pycache_prefix=<directorio nuevo y vacío fuera del repositorio> -m src.r7_v2.g4_ejecutar --modo humo --autorizacion-ejecucion CONFIRMADO:D-NNN [--humo-config S2_g200]
    ... --modo fase_a [--hasta N] | promocion | fase_b [--aceptar-fase-a-incompleta CONFIRMADO:D-NNN] | fase_c | fase_c_confirmacion

Cada modo exige `--autorizacion-ejecucion CONFIRMADO:D-NNN` ANTES de leer o escribir nada. Entrenamiento = `r4_v2.entrenamiento.ejecutar_corrida` (sin editar) con el
entrenador de `g4_entrenador` y los pesos efectivos; integridad propia (HEAD, árbol limpio, SHA-256, pin del YAML de G2), K1-K4 y K8 de G2 y comparabilidad con
la base A|densenet121 de G2 (solo lectura). Presupuesto: ledger append-only y tope duro de 75 INTENTOS (`g4_presupuesto`). No consulta el test oficial ni las
180 normales. Códigos de salida: 0 ok; 2 un intento no se completó (queda conservado); 3 guarda; 5 presupuesto. Partición provisional (phash_le8_v1,
exploratoria, post hoc); conformidad académica pendiente.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.r4_v2 import pesos as W
from src.r4_v2.common import cargar_config as cargar_config_g2
from src.r4_v2.common import escribir_json_nuevo, sha256_archivo
from src.r4_v2.entrenamiento import corrida_completada, ejecutar_corrida, id_corrida, intentos_de
from src.r4_v2.particion import cargar_particion

from . import g4_agente as AGT
from . import g4_comun as C
from . import g4_presupuesto as P
from .g4_entrenador import entrenar_fold_g4, pesos_efectivos, precargar_modulos_g4

GuardError = C.GuardError
FASE_DE_MODO = {"fase_a": "A", "fase_b": "B", "fase_c": "C", "fase_c_confirmacion": "C"}


class Ctx:
    """Dependencias del ejecutor (inyectables en pruebas)."""

    def __init__(self, root, config, config_g2, particion, referencia, verificar, entrenar, token, head, ledger, comp, r6, tope=P.TOPE_INTENTOS):
        self.root, self.config, self.config_g2, self.particion, self.referencia = root, config, config_g2, particion, referencia
        self.verificar, self.entrenar, self.token, self.head, self.ledger, self.comp, self.r6, self.tope = verificar, entrenar, token, head, ledger, comp, r6, tope


# ---------------------------------------------------------------- piezas
def items_de(config, estrategias, semillas) -> list:
    return [{"estrategia": e, "semilla": s, "fold": k} for e in estrategias for s in semillas for k in config["diseno"]["folds"]]


def _g2d(E, it):
    return C.config_g2_g4(E.config_g2, E.config, it["estrategia"], [it["semilla"]])


def _dir_corrida(E, it, modo="completa") -> Path:
    g = _g2d(E, it)
    return C.raiz_corridas(E.root, E.config, it["estrategia"]["id"], modo) / id_corrida(g, E.config["base"]["arquitectura"], it["semilla"], it["fold"])


def _item_plan(E, it, modo="completa") -> dict:
    return {"config_id": it["estrategia"]["id"], "id": _dir_corrida(E, it, modo).name}


def _coherente_con_previo(E, it, dir_previo) -> None:
    """Una corrida ya completada solo se omite si es de la misma estrategia, partición y receta; si no, la autora decide."""
    rc = json.loads((Path(dir_previo) / "run_config.json").read_text(encoding="utf-8"))
    if (rc["sha256_bytes_archivo_particion"] != E.particion.sha256_bytes_archivo or rc["receta"] != _g2d(E, it)["receta"]
            or rc.get("condicion") != f"A+{it['estrategia']['id']}"):
        raise GuardError(f"{Path(dir_previo).parent.name}: la corrida ya completada tiene otra partición, receta o estrategia; no se mezcla (decide la autora)")


def exigir_humo_previo(root, config, *, sha256_bytes_archivo_particion, sha256_contenido_blob_config, git_commit_head) -> Path:
    for rc_path in sorted(C.raiz_g4(root, config).glob("*/humo/*/intento_*/run_config.json")):
        rc = json.loads(rc_path.read_text(encoding="utf-8"))
        if (rc.get("modo") == "humo" and rc.get("estado") == "completada" and not rc["invalida_integridad"] and rc["git_commit_head"] == git_commit_head
                and rc["sha256_bytes_archivo_particion"] == sha256_bytes_archivo_particion
                and rc["verificacion_inicio"]["archivos"][C.CONFIG_REL]["sha256_contenido_blob"] == sha256_contenido_blob_config):
            return rc_path.parent
    raise GuardError("fase_a exige un 'humo' completado previo (mismo commit, YAML y partición); el humo consume un intento del tope")


def exigir_orden_fase_a(E, seleccionadas) -> None:
    """No se salta ni se reordena: si la configuración i no está completa (5 folds), ninguna posterior puede tener intentos."""
    todas = C.estrategias_fase_a(E.config)
    estado = []
    for e in todas:
        its = items_de(E.config, [e], [E.config["diseno"]["fase_a_semilla"]])
        n_ok = sum(1 for it in its if corrida_completada(_dir_corrida(E, it)) is not None)
        hay = any(intentos_de(_dir_corrida(E, it)) for it in its)
        estado.append((e["id"], n_ok == len(its), hay))
    for i, (cid, completa, _) in enumerate(estado):
        if not completa and any(h for _, _, h in estado[i + 1:]):
            raise GuardError(f"orden fijo violado: {cid} no está completa y hay intentos de una configuración posterior; no se salta ni se reordena")
    ids_sel = [e["id"] for e in seleccionadas]
    if ids_sel != [e["id"] for e in todas][:len(ids_sel)]:
        raise GuardError("la selección de la fase A debe ser un prefijo del orden fijo")


def _sello(E, d, *, fase, it, extras, humo=False) -> None:
    rc = Path(d) / "run_config.json"
    e = it["estrategia"]
    escribir_json_nuevo(Path(d) / "sello_g4.json", {
        "esquema": "sello_g4_v1/1", "protocolo_id": C.PROTOCOLO_ID, "leyenda": C.LEYENDA, "fase": "humo" if humo else fase, "humo": humo,
        "config_id": e["id"], "familia": e["familia"], "alpha": e["alpha"], "gamma": e["gamma"], "perdida_entrenamiento": C.perdida_de(e),
        "semilla": it["semilla"], "fold": it["fold"], "id_propuesta": (extras or {}).get("propuesta", {}).get("archivo"),
        "autorizacion_ejecucion": E.token, "git_commit_head": E.head, "fecha": P.ahora(), "base_r6": E.r6, "comparabilidad_con_base": E.comp,
        "sha256_bytes_run_config": sha256_archivo(rc) if rc.is_file() else None, **(extras or {})})


def correr(E, fase, items, *, modo="completa", epocas=None, extras=None, humo=False) -> int:
    """Ejecuta `items` EN ORDEN. Guarda previa a CADA intento (sin crear directorio si se rechaza); cada intento iniciado se anota en el ledger."""
    n = len(items)
    for i, it in enumerate(items):
        d_run = _dir_corrida(E, it, modo)
        previo = corrida_completada(d_run)
        rotulo = f"{it['estrategia']['id']} {d_run.name}"
        if previo is not None:
            _coherente_con_previo(E, it, previo)
            print(f"{rotulo}: ya completada; se omite (nunca se repite una corrida válida)")
            continue
        C.exigir_integridad(E.verificar())                                             # bloqueo previo: no cuenta ni crea nada
        eventos = P.leer_ledger(E.ledger)
        pendientes = sum(1 for x in items[i:] if corrida_completada(_dir_corrida(E, x, modo)) is None)
        P.guardia(eventos, pendientes, tope=E.tope)
        intento = len(intentos_de(d_run)) + 1
        comun = dict(fase="humo" if humo else fase, config_id=it["estrategia"]["id"], id_corrida=d_run.name, head=E.head, autorizacion=E.token,
                     nota="humo" if humo else None)
        if intento > 1:
            P.registrar(E.ledger, intento=intento, evento="reintento", **comun)
        P.registrar(E.ledger, intento=intento, evento="iniciada", **comun)            # tras integridad y autorización, justo antes de crear el directorio
        t0, g = time.monotonic(), _g2d(E, it)
        try:
            with pesos_efectivos():
                r = ejecutar_corrida(E.root, g, E.particion, modo=modo, arq=E.config["base"]["arquitectura"], semilla=it["semilla"], fold=it["fold"],
                                     raiz_modo=C.raiz_corridas(E.root, E.config, it["estrategia"]["id"], modo), verificar=E.verificar,
                                     entrenar=E.entrenar, referencia=E.referencia, atestaciones={"autorizacion_ejecucion": E.token}, epocas_max=epocas)
        except GuardError as exc:                                                       # G2 no llegó a crear el intento: sigue contando (conservador)
            P.registrar(E.ledger, intento=intento, evento="fallida", segundos=round(time.monotonic() - t0, 3), **{**comun, "nota": f"sin_directorio: {exc}"})
            raise
        seg = round(time.monotonic() - t0, 3)
        rc = json.loads((Path(r["dir"]) / "run_config.json").read_text(encoding="utf-8"))
        ent = rc.get("entorno") or {}
        P.registrar(E.ledger, intento=r["intento"], evento="completada" if r["estado"] == "completada" else "fallida", segundos=seg,
                    dispositivo=ent.get("dispositivo"), gpu=ent.get("gpu"), **{**comun, "nota": comun["nota"] if r["estado"] == "completada" else f"{r['estado']}: {r['motivo']}"})
        _sello(E, r["dir"], fase=fase, it=it, extras=extras, humo=humo)
        print(f"{rotulo} intento {r['intento']}: {r['estado']}" + (f" ({r['motivo']})" if r["motivo"] else "") + f" [{i + 1}/{n}]")
        if r["estado"] != "completada":
            return 2                                                                   # el intento queda conservado; decide la autora
    return 0


def _planificar_y_guardar(E, fase, items, *, reserva=0) -> None:
    """Guarda de fase (sin escribir si se rechaza) y luego `planificada` por corrida."""
    eventos = P.leer_ledger(E.ledger)
    pendientes = sum(1 for it in items if corrida_completada(_dir_corrida(E, it)) is None)
    P.guardia(eventos, pendientes, tope=E.tope, reserva=reserva)
    P.planificar(E.ledger, eventos, fase, [_item_plan(E, it) for it in items], head=E.head, autorizacion=E.token)


def _escribir_json_g4(E, prefijo, datos) -> Path:
    raiz = C.raiz_g4(E.root, E.config)
    raiz.mkdir(parents=True, exist_ok=True)
    destino = C.siguiente_archivo_nuevo(raiz, prefijo)
    escribir_json_nuevo(destino, datos)
    return destino


def _huellas_runs(E, estrategia, semilla) -> dict:
    return {_dir_corrida(E, it).name: sha256_archivo(corrida_completada(_dir_corrida(E, it)) / "run_config.json")
            for it in items_de(E.config, [estrategia], [semilla]) if corrida_completada(_dir_corrida(E, it)) is not None}


# ---------------------------------------------------------------- núcleo
def ejecutar(root, config, config_g2, *, modo, autorizacion="", humo_config=None, hasta=None, aceptar_incompleta="", verificar=None, entrenar=None,
             referencia=None, tope=P.TOPE_INTENTOS) -> int:
    if modo not in C.MODOS:
        raise GuardError(f"modo desconocido: {modo!r} (válidos: {C.MODOS})")
    token = C.exigir_autorizacion(autorizacion)                                       # antes de leer o escribir nada
    if (hasta is not None and modo != "fase_a") or (humo_config is not None and modo != "humo"):
        raise GuardError("--hasta solo aplica a fase_a y --humo-config solo a humo")
    C.validar_config(config)
    if not isinstance(tope, int) or not 1 <= tope <= P.TOPE_INTENTOS:
        raise GuardError(f"el tope de intentos no puede superar {P.TOPE_INTENTOS} (sin enmienda)")
    C.comprobar_coherencia(config, config_g2)
    verificar = verificar or (lambda: C.verificar_integridad_g4(root, config))
    particion = cargar_particion(root, config_g2)                                     # K1-K4
    referencia = referencia or W.cargar_referencia(root)                              # K8
    inicial = C.exigir_integridad(verificar())
    r6 = C.verificar_base_r6(root, config)
    root = Path(root)
    E = Ctx(root, config, config_g2, particion, referencia, verificar, entrenar or entrenar_fold_g4, token, inicial["git_commit_head"],
            C.ruta_ledger(root, config), None, r6, tope)
    semilla_a = config["diseno"]["fase_a_semilla"]
    todas = C.estrategias_fase_a(config)

    if modo == "humo":
        h = config["diseno"]["humo"]
        cid = humo_config or h["config_por_defecto"]
        est = next((e for e in todas if e["id"] == cid), None)
        if est is None:
            raise GuardError(f"--humo-config debe ser una de {[e['id'] for e in todas]}")
        E.comp = C.comparar_con_base(root, config, config_g2, particion, referencia, [h["semilla"]], exigir=False)
        it = {"estrategia": est, "semilla": h["semilla"], "fold": h["fold"]}
        return correr(E, "humo", [it], modo="humo", epocas=h["epocas"], humo=True)

    if modo == "promocion":
        E.comp = C.comparar_con_base(root, config, config_g2, particion, referencia, [semilla_a])
        res = C.calcular_promocion(root, config, config_g2, particion)
        res["huellas_run_config"] = {e["id"]: _huellas_runs(E, e, semilla_a) for e in todas}
        destino = _escribir_json_g4(E, "promocion", {"esquema": "promocion_g4_v1/1", "protocolo_id": C.PROTOCOLO_ID, "leyenda": C.LEYENDA,
                                                     "fecha": P.ahora(), "autorizacion_ejecucion": token, "git_commit_head": E.head,
                                                     "criterio": config["promocion"], "comparabilidad_con_base": E.comp, **res})
        print(f"promoción escrita en {destino.relative_to(root.resolve()).as_posix()}: promovidas {res['promovidas']}"
              + (" (PROVISIONAL: fase A incompleta; no se declara la mejor de las seis)" if res["promocion_provisional"] else ""))
        return 0

    if modo == "fase_a":
        sel = C.seleccionar_fase_a(config, hasta)
        exigir_humo_previo(root, config, sha256_bytes_archivo_particion=particion.sha256_bytes_archivo,
                           sha256_contenido_blob_config=inicial["archivos"][C.CONFIG_REL]["sha256_contenido_blob"], git_commit_head=E.head)
        E.comp = C.comparar_con_base(root, config, config_g2, particion, referencia, [semilla_a])
        exigir_orden_fase_a(E, sel)
        completos = items_de(config, todas, [semilla_a])
        eventos = P.leer_ledger(E.ledger)
        pend = sum(1 for it in items_de(config, sel, [semilla_a]) if corrida_completada(_dir_corrida(E, it)) is None)
        P.guardia(eventos, pend, tope=tope)
        P.planificar(E.ledger, eventos, "A", [_item_plan(E, it) for it in completos], head=E.head, autorizacion=token)   # las 30 de la fase
        return correr(E, "A", items_de(config, sel, [semilla_a]))

    raiz = C.raiz_g4(root, config)
    if modo == "fase_b":
        ruta = C.ultimo_archivo(raiz, "promocion")
        if ruta is None:
            raise GuardError("fase_b exige un promocion.json previo (modo promocion)")
        prom = json.loads(ruta.read_text(encoding="utf-8"))
        nueva = C.calcular_promocion(root, config, config_g2, particion)
        if nueva["promovidas"] != prom["promovidas"] or nueva["promocion_provisional"] != prom["promocion_provisional"]:
            raise GuardError("la promoción registrada no coincide con la recalculada desde las predicciones (¿se añadieron configuraciones?); repetir el modo promocion")
        if not prom["promovidas"]:
            raise GuardError("sin candidata con ΔAP > 0 en la fase A: no hay fase B ni fase C (se conserva la base)")
        if prom["promocion_provisional"]:
            C.exigir_autorizacion(aceptar_incompleta, "--aceptar-fase-a-incompleta")
        E.comp = C.comparar_con_base(root, config, config_g2, particion, referencia, config["diseno"]["semillas_confirmacion"])
        por_id = {e["id"]: e for e in todas}
        items = items_de(config, [por_id[i] for i in prom["promovidas"]], config["diseno"]["semillas_confirmacion"])
        _planificar_y_guardar(E, "B", items)
        extras = {"promocion": {"archivo": ruta.name, "sha256_bytes_archivo": sha256_archivo(ruta), "provisional": prom["promocion_provisional"],
                                "aceptacion_fase_a_incompleta": aceptar_incompleta or None}}
        return correr(E, "B", items, extras=extras)

    # fase_c y fase_c_confirmacion: mecanismo del agente (preparado)
    ag = config["agente"]
    ruta_ev = raiz / ag["evidencia_fase_b"]
    familia = AGT.familia_seleccionada(AGT.leer_evidencia_fase_b(ruta_ev))
    carpeta = raiz / ag["carpeta_propuestas"]
    conclusion = AGT.leer_conclusion(root, carpeta)
    props = AGT.validar_propuestas(root, carpeta, familia["familia"], max_propuestas=ag["max_propuestas"])
    if conclusion and props:
        raise GuardError("hay propuestas y conclusion_sin_mejora.json a la vez; el agente debe elegir una")
    ev = {"evidencia_fase_b": {"archivo": ag["evidencia_fase_b"], "sha256_bytes_archivo": sha256_archivo(ruta_ev)}, "familia_seleccionada": familia}
    if conclusion:
        carpeta.mkdir(parents=True, exist_ok=True)
        nuevo = AGT.anotar_conclusion(carpeta, conclusion, head=E.head, autorizacion=token)
        print("el agente concluyó SIN mejora" + (" (anotado en el registro)" if nuevo else " (ya anotado)") + ": no se ejecuta la fase C")
        return 0
    if not props:
        raise GuardError("el agente activado no dejó propuestas ni conclusion_sin_mejora.json")
    E.comp = C.comparar_con_base(root, config, config_g2, particion, referencia, [ag["semilla_propuestas"]] if modo == "fase_c" else ag["semillas_confirmacion"])
    folds = config["diseno"]["folds"]
    if modo == "fase_c":
        items = items_de(config, [p["estrategia"] for p in props], [ag["semilla_propuestas"]])
        reserva = len(folds) * len(ag["semillas_confirmacion"])
        _planificar_y_guardar(E, "C", items, reserva=reserva)                         # confirmación posible: 5n + 10 <= restante y <= 25
        AGT.registrar_propuestas(carpeta, props, head=E.head, autorizacion=token)
        por_archivo = {p["estrategia"]["id"]: p for p in props}
        rc_ = 0
        for e in [p["estrategia"] for p in props]:
            p = por_archivo[e["id"]]
            rc_ = correr(E, "C", items_de(config, [e], [ag["semilla_propuestas"]]),
                         extras={**ev, "propuesta": {"archivo": p["archivo"], "sha256_bytes_archivo": p["sha256_bytes_archivo"]}})
            if rc_ != 0:
                return rc_
        print(f"presupuesto restante de C: {P.presupuesto_fase_c(P.leer_ledger(E.ledger), tope)} intentos")
        return 0
    # fase_c_confirmacion: promoción EXPLORATORIA (no evidencia confirmatoria): delta AP de la semilla 42 > 0 frente a la base Y frente a la candidata convencional;
    # con varias, UNA sola: mayor delta frente a la candidata; empate exacto: menor número de propuesta
    s42 = ag["semilla_propuestas"]
    conv = next((e for e in todas if e["id"] == familia["config_id"]), None)
    ap_conv = None if conv is None else C.ap_estrategia(root, config, config_g2, particion, conv, s42)
    if ap_conv is None:
        raise GuardError(f"la candidata convencional {familia['config_id']} no tiene su semilla 42 completa: no se puede evaluar la promoción")
    base42 = C.ap_base(root, config, config_g2, particion, s42)
    d_base, d_conv = {}, {}
    for p in props:
        ap = C.ap_estrategia(root, config, config_g2, particion, p["estrategia"], s42)
        if ap is None:
            raise GuardError(f"{p['estrategia']['id']}: la fase C (semilla 42) no está completa; no se elige la propuesta a confirmar")
        d_base[p["estrategia"]["id"]], d_conv[p["estrategia"]["id"]] = ap - base42, ap - ap_conv
    sel = C.elegir_confirmacion(d_base, d_conv, [p["estrategia"]["id"] for p in props])
    if sel["elegida"] is None:
        print("ninguna propuesta tiene delta AP > 0 en la semilla 42 frente a la base Y frente a la candidata convencional: no se confirma ninguna (promoción exploratoria)")
        return 0
    mejor = next(p for p in props if p["estrategia"]["id"] == sel["elegida"])
    items = items_de(config, [mejor["estrategia"]], ag["semillas_confirmacion"])
    _planificar_y_guardar(E, "C", items)                                                # presupuesto insuficiente: se rechaza sin crear nada
    return correr(E, "C", items, extras={**ev, "propuesta": {"archivo": mejor["archivo"], "sha256_bytes_archivo": mejor["sha256_bytes_archivo"]},
                                         "promocion_exploratoria": {**sel, "delta_ap_semilla_42_vs_base": d_base[sel["elegida"]],
                                                                    "delta_ap_semilla_42_vs_candidata_convencional": d_conv[sel["elegida"]],
                                                                    "candidata_convencional": familia["config_id"]}})


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="G4 g4_optimizacion_v1 (mapeo v2, phash_le8_v1; partición provisional; conformidad académica pendiente)")
    p.add_argument("--modo", required=True, choices=C.MODOS)
    p.add_argument("--autorizacion-ejecucion", default="", help="CONFIRMADO:D-NNN específica de la fase; sin ella no se lee ni se escribe nada")
    p.add_argument("--humo-config", default=None, help="humo: una de S1_a050 S2_g200 S1_a025 S2_g100 S1_a075 S2_g050 (por defecto la del YAML)")
    p.add_argument("--hasta", type=int, default=None, help="fase_a: detenerse tras la N-ésima configuración del orden fijo")
    p.add_argument("--aceptar-fase-a-incompleta", default="", help="fase_b con promoción provisional: CONFIRMADO:D-NNN")
    return p


def main(argv=None, *, root=None, config=None, config_g2=None, precargar=None, **inyecciones) -> int:
    args = construir_parser().parse_args(argv)
    root = Path(root or C.RAIZ_REPO).resolve()
    try:
        C.exigir_autorizacion(args.autorizacion_ejecucion)                            # antes de leer nada
        config = config or C.cargar_config_g4(root / C.CONFIG_REL)
        config_g2 = config_g2 or cargar_config_g2(root / config["referencia_g2"]["config"])
        if args.modo != "promocion":
            (precargar or precargar_modulos_g4)()                                     # torch y módulos de G2 antes de verificar la integridad
        return ejecutar(root, config, config_g2, modo=args.modo, autorizacion=args.autorizacion_ejecucion, humo_config=args.humo_config, hasta=args.hasta,
                        aceptar_incompleta=args.aceptar_fase_a_incompleta, **inyecciones)
    except P.PresupuestoAgotado as exc:
        print(f"PRESUPUESTO: {exc}", file=sys.stderr)
        return P.CODIGO_SALIDA_PRESUPUESTO
    except GuardError as exc:
        print(f"DETENIDO: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
