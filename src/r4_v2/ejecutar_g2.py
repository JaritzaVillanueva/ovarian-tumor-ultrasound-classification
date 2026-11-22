"""Ejecutor de G2 `g2_baseline_v1` (puntos de entrada de los tres modos del protocolo §4.7).

Uso (solo por el autor, en el entorno autorizado, con las banderas de bytecode del protocolo §4.5.6):

    python -B -X pycache_prefix=<directorio nuevo y vacío fuera del repositorio> -m src.r4_v2.ejecutar_g2 --modo referencia ...

NO entrena por sí solo ni sin autorización: cada modo exige las atestaciones del autor (auditoría independiente y autorización de ejecución; se registran, no se verifican; la
conformidad del asesor ya NO se exige, enmienda 2, y sigue PENDIENTE), la
integridad §4.5 (K6) y, en `humo`/`completa`, la partición verificada (K1-K4) y la referencia de pesos (K8). Un agente no ejecuta esto.
No consulta el test oficial ni las 180 normales; no invoca la calibración del umbral (la ejecuta G4).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import integridad as I
from . import modos as MO
from . import pesos as W
from .common import CONFIG_REL, GuardError, MODOS, RAIZ_REPO, cargar_config, validar_sha256
from .entrenamiento import corrida_completada, ejecutar_corrida, entrenar_fold_torch, precargar_modulos
from .particion import cargar_particion


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="G2 g2_baseline_v1 (mapeo v2, phash_le8_v1; partición provisional; conformidad "
                                            "académica pendiente)")
    p.add_argument("--modo", required=True, choices=MODOS)
    p.add_argument("--autorizacion-ejecucion", default="", help="atestación positiva: CONFIRMADO:AUT-NNN o CONFIRMADO:D-NNN")
    p.add_argument("--auditoria-independiente", default="", help="atestación positiva: CONFIRMADO:AUD-NNN")
    p.add_argument("--conformidad-asesor", default="", help="OPCIONAL, informativa: vacía = se ignora; si se da, CONFIRMADO:ASE-NNN. No habilita nada (la conformidad sigue PENDIENTE)")
    p.add_argument("--notebook", default=None, help="ruta (relativa al repositorio) del notebook orquestador, si se lanzó desde uno")
    p.add_argument("--excepcion-protocolo-no-versionado", default=None, help="solo referencia: motivo documentado por el autor")
    p.add_argument("--sha256-borrador-aprobado", default=None, help="sha256_bytes_archivo del borrador aprobado: 64 hex en minúsculas (con la excepción)")
    p.add_argument("--humo-arquitectura", default=None)
    p.add_argument("--humo-semilla", type=int, default=None)
    p.add_argument("--humo-fold", type=int, default=None)
    p.add_argument("--epocas-humo", type=int, default=1)
    return p


def main(argv=None, *, root=None, entrenar=None, precargar=None, instanciar=None, resolver=None, versiones=None,
         entorno=None) -> int:
    """Los parámetros con nombre permiten inyectar sustitutos sintéticos en las pruebas; en producción usan torch."""
    args = construir_parser().parse_args(argv)
    root = Path(root or RAIZ_REPO).resolve()
    try:
        config = cargar_config(root / CONFIG_REL)
        atest = MO.exigir_atestaciones(args.modo, {"conformidad_asesor": args.conformidad_asesor,
                                                    "auditoria_independiente": args.auditoria_independiente,
                                                    "autorizacion_ejecucion": args.autorizacion_ejecucion})
        excepcion = None
        if args.excepcion_protocolo_no_versionado or args.sha256_borrador_aprobado:
            if not (args.excepcion_protocolo_no_versionado or "").strip():
                raise GuardError("--sha256-borrador-aprobado requiere --excepcion-protocolo-no-versionado con el motivo")
            excepcion = {"motivo": args.excepcion_protocolo_no_versionado,
                         "sha256_bytes_archivo_borrador": validar_sha256(args.sha256_borrador_aprobado, "--sha256-borrador-aprobado")}
        (precargar or precargar_modulos)()                                       # importaciones ANTES de la verificación §4.5

        def verificar():
            return I.verificar_integridad(root, config, modo=args.modo, entorno=entorno() if entorno else I.EntornoPython.actual(),
                                          notebook=args.notebook, excepcion_protocolo=excepcion)

        raiz_modo = MO.raiz_del_modo(root, config, args.modo, crear=False)   # se crea al escribir el primer intento
        if args.modo == "referencia":
            def _instanciar(arq):
                from src.classification import models as m
                return {"resnet50": m.create_resnet50_classifier, "densenet121": m.create_densenet121_classifier,
                        "convnext_tiny": m.create_convnext_tiny_classifier}[arq](pretrained=True, num_classes=2)
            r = MO.ejecutar_referencia(root, config, verificar=verificar, instanciar=instanciar or _instanciar,
                                       resolver=resolver or W.resolver_pesos_torchvision, versiones=versiones or W.versiones_torch,
                                       raiz_modo=raiz_modo, atestaciones=atest)
            print(f"referencia: {r['estado']} -> {r['dir'].relative_to(root).as_posix()} (candidato; el autor lo verifica y versiona)")
            return 0 if r["estado"] == "completada" else 2

        referencia = W.cargar_referencia(root)                                   # K8: ausente/no rastreada/modificada -> se detiene
        particion = cargar_particion(root, config)                               # K1-K4
        if args.modo == "completa":
            inicial = I.exigir_integridad(verificar())
            MO.exigir_humo_previo(MO.raiz_del_modo(root, config, "humo", crear=False),
                                  sha256_bytes_archivo_particion=particion.sha256_bytes_archivo,
                                  sha256_contenido_blob_config=inicial["archivos"][CONFIG_REL]["sha256_contenido_blob"],
                                  sha256_contenido_blob_referencia=referencia["sha256_contenido_blob"])
        plan = MO.plan_de_corridas(args.modo, config, humo_arq=args.humo_arquitectura, humo_semilla=args.humo_semilla,
                                   humo_fold=args.humo_fold)
        epocas = MO.epocas_humo(config, args.epocas_humo) if args.modo == "humo" else None
        for c in plan:
            if corrida_completada(raiz_modo / c["id"]) is not None:
                print(f"{c['id']}: ya completada; se omite (nunca se repite una corrida válida)")
                continue
            r = ejecutar_corrida(root, config, particion, modo=args.modo, arq=c["arq"], semilla=c["semilla"], fold=c["fold"],
                                 raiz_modo=raiz_modo, verificar=verificar, entrenar=entrenar or entrenar_fold_torch,
                                 referencia=referencia, atestaciones=atest, epocas_max=epocas)
            print(f"{r['id_corrida']} intento {r['intento']}: {r['estado']}" + (f" ({r['motivo']})" if r["motivo"] else ""))
            if r["estado"] != "completada":
                return 2                                                          # el autor decide; el intento queda conservado
        return 0
    except GuardError as exc:
        print(f"DETENIDO: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
