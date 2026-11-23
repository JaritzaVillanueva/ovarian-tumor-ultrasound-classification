"""CLI sin pantalla de la interfaz de R9: clasificación de una imagen, verificación del sistema congelado e integración con imágenes de DESARROLLO (para Falcon).

  python -B -m src.gui.cli clasificar --imagen RUTA [--json]
  python -B -m src.gui.cli verificar --autorizacion CONFIRMADO:D-NNN
  python -B -m src.gui.cli integracion-desarrollo --autorizacion CONFIRMADO:D-NNN [--n-por-fold 2]
  python -B -m src.gui.cli verificar-checkpoints --autorizacion CONFIRMADO:D-NNN [--sha256-declaracion 64HEX]   # solo SHA-256 de los 15 checkpoints; sin torch, sin inferencia

`integracion-desarrollo` comprueba, SOLO con imágenes de desarrollo y sin etiquetas, que la ruta de la interfaz (validación, RGB, preprocesamiento de G2, softmax) reproduce las probabilidades guardadas
de cada miembro en su propio pliegue. NO calcula métricas, NO usa etiquetas, NO evalúa el ensamble sobre desarrollo y NO sirve para reajustar nada. Nunca abre el test.
Códigos de salida: 0 correcto; 2 sistema no disponible (modo demostración); 3 imagen inválida o compuerta; 5 integración fuera de tolerancia.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.r4_v2.common import RAIZ_REPO, GuardError, directorio_nuevo, escribir_json_nuevo, exigir_ruta_no_prohibida, exigir_salida_segura, sha256_archivo
from src.r4_v2.common import cargar_config as cargar_config_g2
from src.r4_v2.particion import cargar_particion, filas_de_fold, resolver_imagen
from src.r7_v2 import sistema_final as SF
from src.r7_v2 import umbral_sistema as U
from src.r8_v2 import congelamiento as C
from src.r8_v2 import declaracion as DC
from src.gui import textos as T
from src.gui.sistema import SHA256_DECLARACION, SistemaNoDisponible, cargar_sistema
from src.gui.validacion_imagen import ImagenInvalida, validar_y_cargar

SALIDA_REL = "results/r9_gui/integracion_desarrollo"
TOLERANCIA_INTEGRACION = 1e-4                                                              # más estricta que la alerta de recarga de G2 (0.01); el lote de G2 es de 32 y aquí de 1


def integracion_desarrollo(root, config_g2, particion, miembros_filas, probabilidad_miembro, *, n_por_fold=2, tolerancia=TOLERANCIA_INTEGRACION) -> dict:
    """Cada miembro, sobre las primeras `n_por_fold` imágenes (por `image_id`) de su PROPIO pliegue de desarrollo, a través de la ruta de la interfaz, frente a su `predicciones_oof.csv`.
    No lee ni informa etiquetas; no agrega miembros; no calcula métricas."""
    root = Path(root)
    salida = []
    for m in miembros_filas:
        _, val = filas_de_fold(particion, int(m["fold"]))
        elegidas = sorted(val, key=lambda f: str(f["image_id"]))[:n_por_fold]
        pred_csv = root / config_g2["diseno"]["raiz_resultados"].strip("/") / "completa" / m["id_corrida"] / m["intento"] / "predicciones_oof.csv"
        ids, guardadas = SF.leer_probabilidades_guardadas(pred_csv)
        por_id = dict(zip(ids, guardadas))
        difs = []
        for f in elegidas:
            imagen, _ = validar_y_cargar(resolver_imagen(root, f))                           # K4: solo train/validation de desarrollo; la validación rechaza además `test`
            difs.append(abs(float(probabilidad_miembro(m["id_corrida"], imagen)) - float(por_id[str(f["image_id"])])))
        salida.append({"id_corrida": m["id_corrida"], "fold": int(m["fold"]), "n_imagenes": len(elegidas), "diferencia_maxima_absoluta": max(difs) if difs else None})
    maxima = max((s["diferencia_maxima_absoluta"] for s in salida if s["diferencia_maxima_absoluta"] is not None), default=None)
    return {"miembros": salida, "n_por_fold": n_por_fold, "diferencia_maxima_global": maxima, "tolerancia": tolerancia, "ok": maxima is not None and maxima <= tolerancia,
            "metricas_calculadas": False, "etiquetas_usadas": False, "ensamble_evaluado_sobre_desarrollo": False}


def tabla_checkpoints(root, miembros) -> list[dict]:
    """SHA-256 recalculado de cada checkpoint frente al del manifiesto versionado. Informa los 15 (no se detiene en el primero); no carga modelos ni infiere."""
    filas = []
    for m in miembros:
        exigir_ruta_no_prohibida(m["ruta_checkpoint"])
        ruta = Path(root) / m["ruta_checkpoint"]
        esperado = m["sha256_bytes_archivo_checkpoint"]
        observado = sha256_archivo(ruta) if ruta.is_file() else None
        estado = "AUSENTE" if observado is None else ("OK" if observado == esperado else "DISTINTO")
        filas.append({"id_corrida": m["id_corrida"], "estado": estado, "sha256_esperado": esperado, "sha256_observado": observado,
                      "tamano_bytes": ruta.stat().st_size if observado is not None else None, "ruta": m["ruta_checkpoint"]})
    return filas


def _verificar_checkpoints(root, token: str, sha256_declaracion: str) -> int:
    DC.exigir_declaracion(root, sha256_declaracion, comprobar_checkpoints=False, comprobar_reporte=False)          # declaración, manifiesto y código de inferencia; los checkpoints se tabulan abajo
    config = C.cargar_config(root / C.CONFIG_REL)
    filas = tabla_checkpoints(root, C.validar_miembros(C.leer_tabla(root / C.MIEMBROS_REL), config))
    for f in filas:
        print(f"{f['estado']:<8} {f['id_corrida']:<24} {f['tamano_bytes'] if f['tamano_bytes'] is not None else '-':>10}  esperado {f['sha256_esperado']}  observado {f['sha256_observado']}")
    buenos = sum(f["estado"] == "OK" for f in filas)
    print(f"CHECKPOINTS ({token}): {buenos}/{len(filas)} con SHA-256 igual al del manifiesto versionado. Declaración verificada (SHA-256 {sha256_declaracion[:12]}…). Sin carga de modelos ni inferencia.")
    if buenos != len(filas):
        print("DETENIDO: no se debe ejecutar la interfaz con checkpoints ausentes o distintos de los congelados.", file=sys.stderr)
        return 3
    return 0


def _imprimir_resultado(r, como_json: bool) -> None:
    if como_json:
        print(json.dumps({"clase": r.clase, "puntuacion": r.puntuacion, "umbral": r.umbral_texto, "regla": r.regla, "n_miembros": r.n_miembros, "sha256_declaracion": r.sha256_declaracion,
                          "sistema": r.sistema, "version_interfaz": r.version_interfaz, "probabilidades_por_miembro": r.probabilidades_por_miembro,
                          "advertencias": T.advertencias(), "advertencias_entorno": list(r.advertencias_entorno)}, indent=2, ensure_ascii=False))
        return
    from src.gui.controlador import Controlador
    print(*Controlador.lineas_resultado(r), sep="\n")
    print("\nAVISOS", *T.advertencias(), sep="\n- ")


def main(argv=None, *, root=None, sistema=None) -> int:
    ap = argparse.ArgumentParser(description="R9: interfaz demostrativa (modo sin pantalla). Prototipo académico; sin validez diagnóstica ni uso clínico.")
    sub = ap.add_subparsers(dest="modo", required=True)
    c = sub.add_parser("clasificar")
    c.add_argument("--imagen", required=True)
    c.add_argument("--json", action="store_true")
    for nombre in ("verificar", "integracion-desarrollo"):
        s = sub.add_parser(nombre)
        s.add_argument("--autorizacion", default="", help="CONFIRMADO:D-NNN (decisión de ejecución)")
        if nombre != "verificar":
            s.add_argument("--n-por-fold", type=int, default=2)
    vc = sub.add_parser("verificar-checkpoints")
    vc.add_argument("--autorizacion", default="", help="CONFIRMADO:D-NNN (decisión de ejecución)")
    vc.add_argument("--sha256-declaracion", default=SHA256_DECLARACION, help="SHA-256 de la declaración D-060 (vía externa a ella; por defecto, el fijado en la interfaz)")
    args = ap.parse_args(argv)
    if callable(getattr(sys.stdout, "reconfigure", None)):                          # consolas de Windows en cp1252: los textos llevan tildes y símbolos (≥)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    root = Path(root or RAIZ_REPO).resolve()
    try:
        if args.modo != "clasificar":
            token = U.exigir_autorizacion(args.autorizacion)
        if args.modo == "verificar-checkpoints":
            return _verificar_checkpoints(root, token, args.sha256_declaracion)
        sistema = sistema or cargar_sistema(root, exigir_reporte=(args.modo == "verificar"))
        if args.modo == "verificar":
            print(f"SISTEMA VERIFICADO ({token}): {sistema.t_hat_texto}, {len(sistema.miembros)} miembros, declaración y reporte D-058 comprobados.")
            return 0
        if args.modo == "clasificar":
            imagen, info = validar_y_cargar(args.imagen)
            _imprimir_resultado(sistema.clasificar(imagen), args.json)
            return 0
        config = C.cargar_config(root / C.CONFIG_REL)
        config_g2 = cargar_config_g2(root / config["referencia_g2"]["config"])
        filas = [f for f in C.leer_tabla(root / C.MIEMBROS_REL)]
        res = integracion_desarrollo(root, config_g2, cargar_particion(root, config_g2), filas, sistema.probabilidad_miembro, n_por_fold=args.n_por_fold)
        destino = directorio_nuevo(exigir_salida_segura(root, root / SALIDA_REL / token.split(":")[1], "results/r9_gui"))
        escribir_json_nuevo(destino / "resumen_integracion.json", U.a_json({**res, "autorizacion": token, "fecha": U.ahora(), "version_interfaz": T.VERSION_PROTOTIPO,
                                                                         "entorno_efectivo": SF._versiones(), "advertencia": "sin métricas de rendimiento; solo equivalencia de la ruta de inferencia"}))
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0 if res["ok"] else 5
    except SistemaNoDisponible as exc:
        print(f"{T.MODO_MAQUETA}. " + T.MAQUETA_DETALLE.format(motivo=exc), file=sys.stderr)
        return 2
    except (ImagenInvalida, GuardError) as exc:
        print(f"DETENIDO: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
