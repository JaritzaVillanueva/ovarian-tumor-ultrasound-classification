"""R9 (D-077): dos ejemplos REALES de DESARROLLO para demostrar la interfaz, con selección objetiva fijada ANTES de consultar cualquier predicción.

Regla (fijada antes de aplicarla; no usa predicciones, puntuaciones ni métricas):
  universo   filas de la partición v2 de desarrollo (`train|validation`, K4; jamás test);
  orden      `str(image_id)` (orden de texto, igual que D-072, D-074 y D-076);
  válida     el archivo existe, está dentro de `train|validation/images`, no menciona `test` y pasa la validación de la interfaz (PNG/JPG, dimensiones);
  Caso A     primera imagen válida con categoría original 0, 1, 2, 3 o 4 (clase v2 0, lesión no maligna; se excluye la 6 por indicación de la autora);
  Caso B     primera imagen válida con categoría original 7 (clase v2 1, lesión maligna).
`CASOS` fija los identificadores y los SHA-256 de los archivos; `verificar` re-aplica la regla sobre la partición v2 y se DETIENE si el resultado difiere de lo fijado.
«Caso A/B» describe la clase v2 de la imagen, NO la predicción: la salida de la interfaz puede coincidir o no, y se documenta tal cual.

  python -B -m src.gui.ejemplos_demostracion --autorizacion CONFIRMADO:D-077

No infiere, no carga modelos, no calcula métricas, no abre test. Solo lee la partición v2 verificada por SHA-256 (K1–K4) y calcula el SHA-256 de los dos archivos.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.r4_v2.common import RAIZ_REPO, GuardError, sha256_archivo
from src.r4_v2.common import cargar_config as cargar_config_g2
from src.r4_v2.particion import cargar_particion, resolver_imagen
from src.r7_v2 import umbral_sistema as U
from src.gui.validacion_imagen import ImagenInvalida, validar_y_cargar

TOKEN_DECISION = "D-077"
CATEGORIAS = {"A": (0, 1, 2, 3, 4), "B": (7,)}
CLASE_V2 = {"A": 0, "B": 1}
CONFIG_G2_REL = "configs/r4_v2/g2_baseline_v1.yaml"
CASOS = {                                                                                  # fijados (D-077) a partir de la regla; SHA-256 de los archivos de desarrollo del clon local
    "A": {"image_id": "1", "image_name": "1.JPG", "original_class": 4, "e1_split": "train", "sha256": "1dbb74893c09b2deafccc692a786fbb6385936ecdd491bdf2c707e809a58bc17"},
    "B": {"image_id": "1417", "image_name": "1417.JPG", "original_class": 7, "e1_split": "train", "sha256": "e8637ee3a341fac13a1e3468ff76cfb4856c0980a7a126428d28e02dbbe197b3"},
}


def seleccionar(filas, ruta_de) -> dict:
    """Aplica la regla a `filas` (dicts de la partición); `ruta_de(fila)` devuelve la ruta resuelta de desarrollo (lanza GuardError si no es válida). Devuelve {'A': fila, 'B': fila}."""
    orden = sorted(filas, key=lambda f: str(f["image_id"]))
    elegidas = {}
    for caso, categorias in CATEGORIAS.items():
        for f in orden:
            if int(f["original_class"]) not in categorias:
                continue
            try:
                validar_y_cargar(ruta_de(f))
            except (GuardError, ImagenInvalida):
                continue                                                                   # no válida: se pasa a la siguiente por orden
            elegidas[caso] = f
            break
        else:
            raise GuardError(f"no hay ninguna imagen válida para el caso {caso}")
    return elegidas


def verificar(root, particion) -> list[dict]:
    """Re-aplica la regla sobre la partición v2 y exige que coincida con `CASOS` (identificadores, categoría, split, clase v2 y SHA-256 del archivo)."""
    sel = seleccionar(particion.filas, lambda f: resolver_imagen(root, f))
    salida = []
    for caso, f in sel.items():
        fijo, ruta = CASOS[caso], resolver_imagen(root, f)
        sha = sha256_archivo(ruta)
        if (str(f["image_id"]), f["image_name"], int(f["original_class"]), f["e1_split"], sha) != (fijo["image_id"], fijo["image_name"], fijo["original_class"], fijo["e1_split"], fijo["sha256"]):
            raise GuardError(f"la selección del caso {caso} difiere de la fijada en D-077 (regla → {f['image_id']}, {f['image_name']}, categoría {f['original_class']}, sha {sha[:12]}…): detener y decidir; no se cambia la selección")
        if int(f["label_v2"]) != CLASE_V2[caso]:
            raise GuardError(f"caso {caso}: la clase v2 {f['label_v2']} no es la esperada {CLASE_V2[caso]}")
        salida.append({"caso": caso, "image_id": str(f["image_id"]), "image_name": f["image_name"], "original_class": int(f["original_class"]), "label_v2": int(f["label_v2"]),
                       "class_v2": f.get("class_v2"), "e1_split": f["e1_split"], "fold": int(f["fold"]), "group_id": f["group_id"], "group_size": f.get("group_size"),
                       "ruta": f["image_path"].replace("\\", "/"), "sha256": sha, "bytes": ruta.stat().st_size})
    return salida


def main(argv=None, *, root=None) -> int:
    ap = argparse.ArgumentParser(description="R9 (D-077): verifica la selección fijada de dos ejemplos reales de desarrollo. Sin inferencia; sin test.")
    ap.add_argument("--autorizacion", default="")
    args = ap.parse_args(argv)
    if callable(getattr(sys.stdout, "reconfigure", None)):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    root = Path(root or RAIZ_REPO).resolve()
    try:
        token = U.exigir_autorizacion(args.autorizacion)
        if token.split(":")[1] != TOKEN_DECISION:
            raise GuardError(f"esta herramienta solo se ejecuta bajo CONFIRMADO:{TOKEN_DECISION}")
        filas = verificar(root, cargar_particion(root, cargar_config_g2(root / CONFIG_G2_REL)))
    except GuardError as exc:
        print(f"DETENIDO: {exc}", file=sys.stderr)
        return 3
    for f in filas:
        print(f"Caso {f['caso']} (clase v2 {f['label_v2']}, {f['class_v2']}): image_id={f['image_id']} image_name={f['image_name']} categoría_original={f['original_class']} "
              f"split={f['e1_split']} fold={f['fold']} group_id={f['group_id']} group_size={f['group_size']}\n    ruta={f['ruta']}  bytes={f['bytes']}  sha256={f['sha256']}")
    print(f"SELECCIÓN VERIFICADA ({token}): coincide con la fijada. Sin inferencia, sin métricas, sin test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
