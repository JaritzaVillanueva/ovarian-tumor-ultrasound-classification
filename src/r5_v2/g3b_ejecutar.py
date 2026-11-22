"""Ejecutor de G3b `g3_condiciones_v1` (modos `recortes`, `humo`, `completa`).

Uso (solo por la autora, en Falcon; el agente que lo escribió NO lo ejecuta; no antes de que G3a `completa` termine y la autora acepte las máscaras):

    python -B -X pycache_prefix=<directorio nuevo y vacío fuera del repositorio> -m src.r5_v2.g3b_ejecutar --modo recortes --aceptar-mascaras-g3a CONFIRMADO:D-NNN
    python -B -X pycache_prefix=<...> -m src.r5_v2.g3b_ejecutar --modo humo --condicion B_hibrida --autorizacion-ejecucion CONFIRMADO:D-NNN
    python -B -X pycache_prefix=<...> -m src.r5_v2.g3b_ejecutar --modo completa --condicion B_hibrida [--semillas 42 43 44] --autorizacion-ejecucion CONFIRMADO:D-NNN

`recortes` no entrena (CPU): exige la bandera de aceptación y la consolidación de G3a verificada. `humo` y `completa` exigen la autorización explícita,
árbol Git limpio, la partición verificada (K1-K4 de G2), la referencia de pesos (K8), los recortes verificados y la comparabilidad con A;
`completa` exige un `humo` completado de la MISMA condición (mismo commit, YAML y partición) y omite las corridas ya completadas.
El entrenamiento es el de G2 (`r4_v2.entrenamiento.ejecutar_corrida`, sin editar): solo cambia el origen de la imagen. No consulta el test
oficial ni las 180 normales. Partición provisional (phash_le8_v1, exploratoria, post hoc); conformidad académica pendiente.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from src.r4_v2 import pesos as W
from src.r4_v2.common import cargar_config as cargar_config_g2
from src.r4_v2.common import escribir_json_nuevo
from src.r4_v2.entrenamiento import ejecutar_corrida, precargar_modulos
from src.r4_v2.particion import cargar_particion

from . import g3b_comun as G
from . import g3b_recortes as R
from .comun import cargar_config_g3a

GuardError = G.GuardError


def _coherente_con_previo(dir_intento, *, sha_particion, sha_manifiesto, huella_efectiva, id_corrida) -> None:
    """Una corrida ya completada solo se mezcla con las nuevas si comparte partición, recortes y configuración efectiva (no el commit: R6 añade semillas después)."""
    d = Path(dir_intento)
    rc = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
    pr = json.loads((d / "procedencia_g3b.json").read_text(encoding="utf-8")) if (d / "procedencia_g3b.json").is_file() else {}
    if (rc["sha256_bytes_archivo_particion"] != sha_particion or pr.get("sha256_bytes_manifiesto_recortes") != sha_manifiesto
            or pr.get("huella_config_efectiva") != huella_efectiva):
        raise GuardError(f"{id_corrida}: la corrida ya completada se obtuvo con otra partición, otros recortes u otra configuración efectiva; "
                         "no se mezclan (la autora decide)")


def _sello(dir_intento, condicion, comparabilidad) -> None:
    d = Path(dir_intento)
    rc, pr = d / "run_config.json", d / "procedencia_g3b.json"
    escribir_json_nuevo(d / "sello_g3b.json", {
        "esquema": "sello_g3b_v1/1", "protocolo_id": G.PROTOCOLO_ID, "leyenda": G.LEYENDA, "condicion": condicion, "fecha": G.ahora(),
        "sha256_bytes_run_config": G.sha256_archivo(rc) if rc.is_file() else None,
        "sha256_bytes_procedencia": G.sha256_archivo(pr) if pr.is_file() else None, "comparabilidad_con_A": comparabilidad})


def ejecutar(root, config, config_g2, config_g3a, *, modo, condicion=None, autorizacion="", aceptacion="", arquitecturas=None, semillas=None,
             humo_arquitectura=None, humo_semilla=None, humo_fold=None, epocas_humo=None, verificar=None, entrenar_base=None,
             referencia=None, extraer=None, redimensionar=None) -> int:
    """Núcleo con inyección de dependencias (las pruebas sustituyen Git, torch y ROI). Códigos: 0 ok; 2 fallo de una corrida (intento conservado)."""
    if modo not in G.MODOS:
        raise GuardError(f"modo desconocido: {modo!r} (válidos: {G.MODOS})")
    G.comprobar_coherencia(config, config_g2, config_g3a)
    verificar = verificar or (lambda: G.verificar_integridad_g3b(root, config))
    if modo == "recortes":
        G.exigir_aceptacion_mascaras(aceptacion)                                         # antes de leer nada
        if extraer is None:                                                              # fallar ANTES de crear nada si roi.py no es importable
            try:
                importlib.import_module("src.segmentation.roi")
            except ImportError as exc:
                raise GuardError(f"src/segmentation/roi.py no es importable ({exc}); se requieren torch y cv2") from exc
        particion = cargar_particion(root, config_g2)                                    # K1-K4
        g3a = G.verificar_consolidacion_g3a(root, config_g3a, particion)
        r = R.generar_recortes(root, config, config_g3a, particion, g3a, aceptacion=aceptacion, verificar=verificar, extraer=extraer,
                               redimensionar=redimensionar)
        c = r["sentinela"]
        print(f"recortes generados: {r['n']} (por fuente {c['n_por_fuente']}; fallback {c['n_fallback_por_fuente']}) -> "
              f"{r['dir'].relative_to(Path(root).resolve()).as_posix()}")
        return 0

    atest = {"autorizacion_ejecucion": G.exigir_autorizacion(autorizacion)}              # antes de leer nada
    if condicion is None:
        raise GuardError("humo y completa exigen --condicion")
    d = config["diseno"]
    if modo == "humo":
        h = d["modos"]["humo"]
        arq = humo_arquitectura or h["arquitectura"]
        sem = int(humo_semilla if humo_semilla is not None else h["semilla"])
        fold = int(humo_fold if humo_fold is not None else h["fold"])
        epocas = int(epocas_humo if epocas_humo is not None else h["epocas"])
        if not h["epocas_min"] <= epocas <= h["epocas_max"] or fold not in d["folds"]:
            raise GuardError(f"humo: fold en {d['folds']} y {h['epocas_min']}-{h['epocas_max']} épocas (recibido {fold}, {epocas})")
        if condicion not in config["condiciones"] or not config["condiciones"][condicion]["habilitada"]:
            raise GuardError(f"condición {condicion!r} desconocida o deshabilitada")
        g2d = G.config_g2_derivada(config_g2, config, condicion, [arq], [sem])
        g2d["diseno"]["folds"] = [fold]
        if arq not in config_g2["diseno"]["arquitecturas"] or sem not in config_g2["diseno"]["semillas"]:
            raise GuardError("el humo debe usar una arquitectura y semilla del diseño de G2")
    else:
        archs, sems = G.seleccionar_diseno(config, config_g2, condicion, arquitecturas, semillas)
        g2d = G.config_g2_derivada(config_g2, config, condicion, archs, sems)
        epocas = None
    plan = G.plan_de_corridas(g2d)
    particion = cargar_particion(root, config_g2)                                        # K1-K4
    referencia = referencia or W.cargar_referencia(root)                                 # K8
    g3a = G.verificar_consolidacion_g3a(root, config_g3a, particion)
    indice = R.cargar_indice_recortes(root, config, particion, g3a)
    inicial = G.exigir_integridad(verificar())
    raiz_A = Path(root) / config_g2["diseno"]["raiz_resultados"].strip("/") / "completa"
    comp = G.comparar_con_A(root, config_g2, particion, referencia, plan, raiz_A, exigir=(modo == "completa"))
    print(f"comparabilidad con A: {comp['estado']} ({comp['n_comparadas']} corridas A homólogas)")
    raiz_modo = G.raiz_condicion(root, config, condicion, modo)
    if modo == "completa":
        G.exigir_humo_previo(G.raiz_condicion(root, config, condicion, "humo"), condicion=condicion,
                             sha256_bytes_archivo_particion=particion.sha256_bytes_archivo,
                             sha256_contenido_blob_config=inicial["archivos"][G.CONFIG_REL]["sha256_contenido_blob"],
                             git_commit_head=inicial["git_commit_head"])
    entrenar = R.crear_entrenador(root, config, indice, condicion, entrenar_base=entrenar_base)
    huella = G.huella_config_efectiva(config, condicion)
    for c in plan:
        previo = G.corrida_completada(raiz_modo / c["id"])
        if previo is not None:
            _coherente_con_previo(previo, sha_particion=particion.sha256_bytes_archivo, sha_manifiesto=indice.sha256_manifiesto,
                                  huella_efectiva=huella, id_corrida=c["id"])
            print(f"{c['id']}: ya completada; se omite (nunca se repite una corrida válida)")
            continue
        r = ejecutar_corrida(root, g2d, particion, modo=modo, arq=c["arq"], semilla=c["semilla"], fold=c["fold"], raiz_modo=raiz_modo,
                             verificar=verificar, entrenar=entrenar, referencia=referencia, atestaciones=atest, epocas_max=epocas)
        _sello(r["dir"], condicion, comp)
        print(f"{condicion} {r['id_corrida']} intento {r['intento']}: {r['estado']}" + (f" ({r['motivo']})" if r["motivo"] else ""))
        if r["estado"] != "completada":
            return 2                                                                     # el intento queda conservado; decide la autora
    return 0


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="G3b g3_condiciones_v1 (mapeo v2, phash_le8_v1; partición provisional; conformidad académica pendiente)")
    p.add_argument("--modo", required=True, choices=G.MODOS)
    p.add_argument("--condicion", default=None, choices=sorted(G.SEMANTICA), help="humo y completa: B_hibrida | C (B_oof solo si la autora la habilita)")
    p.add_argument("--autorizacion-ejecucion", default="", help="humo y completa: CONFIRMADO:AUT-NNN o CONFIRMADO:D-NNN (mismo formato que G2/G3a)")
    p.add_argument("--aceptar-mascaras-g3a", default="", help="recortes: CONFIRMADO:D-NNN (decisión de la autora sobre la calidad de las máscaras OOF)")
    p.add_argument("--arquitecturas", nargs="+", default=None)
    p.add_argument("--semillas", nargs="+", type=int, default=None, help="por defecto las de la configuración de la condición (B_hibrida y C: 42)")
    p.add_argument("--humo-arquitectura", default=None)
    p.add_argument("--humo-semilla", type=int, default=None)
    p.add_argument("--humo-fold", type=int, default=None)
    p.add_argument("--epocas-humo", type=int, default=None)
    return p


def main(argv=None, *, root=None, config=None, config_g2=None, config_g3a=None, precargar=None, **inyecciones) -> int:
    args = construir_parser().parse_args(argv)
    root = Path(root or G.RAIZ_REPO).resolve()
    try:
        config = config or G.cargar_config_g3b(root / G.CONFIG_REL)
        config_g2 = config_g2 or cargar_config_g2(root / config["referencia_g2"]["config"])
        config_g3a = config_g3a or cargar_config_g3a(root / config["referencia_g3a"]["config"])
        if args.modo != "recortes":
            (precargar or precargar_modulos)()                                           # torch y módulos de G2 antes de verificar la integridad
        return ejecutar(root, config, config_g2, config_g3a, modo=args.modo, condicion=args.condicion, autorizacion=args.autorizacion_ejecucion,
                        aceptacion=args.aceptar_mascaras_g3a, arquitecturas=args.arquitecturas, semillas=args.semillas,
                        humo_arquitectura=args.humo_arquitectura, humo_semilla=args.humo_semilla, humo_fold=args.humo_fold,
                        epocas_humo=args.epocas_humo, **inyecciones)
    except GuardError as exc:
        print(f"DETENIDO: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
