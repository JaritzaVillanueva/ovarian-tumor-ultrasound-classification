"""Validación y carga de la imagen de entrada de la interfaz de R9 (PNG/JPG; errores claros en español; nunca rutas de test).

La imagen se convierte a RGB igual que el lector de G2 (`Image.open(...).convert("RGB")`); el preprocesamiento de inferencia lo aplica `sistema` con las piezas de G2.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

from src.r4_v2.common import ruta_menciona_test

EXTENSIONES = (".png", ".jpg", ".jpeg")
FORMATOS = ("PNG", "JPEG")
BYTES_MAX = 25 * 1024 * 1024
LADO_MIN, LADO_MAX = 32, 8192


class ImagenInvalida(ValueError):
    """La imagen no se puede usar; el mensaje (en español) se muestra tal cual al usuario."""


def validar_y_cargar(ruta):
    """Devuelve (imagen PIL en RGB, información). Lanza `ImagenInvalida` con un mensaje claro ante cualquier problema."""
    from PIL import Image
    p = Path(str(ruta))
    try:
        absoluta = p.resolve()
    except (OSError, RuntimeError):
        raise ImagenInvalida("La ruta de la imagen no es válida.") from None
    if ruta_menciona_test(str(p)) or ruta_menciona_test(str(absoluta)):
        raise ImagenInvalida("Ruta rechazada: contiene un componente «test». Esta interfaz no abre imágenes del conjunto de prueba oficial.")
    if not p.is_file():
        raise ImagenInvalida("El archivo no existe o no es un archivo regular.")
    if p.suffix.lower() not in EXTENSIONES:
        raise ImagenInvalida("Formato no admitido: solo se aceptan imágenes PNG o JPG.")
    tam = p.stat().st_size
    if tam == 0:
        raise ImagenInvalida("El archivo está vacío.")
    if tam > BYTES_MAX:
        raise ImagenInvalida(f"El archivo supera el tamaño máximo permitido ({BYTES_MAX // (1024 * 1024)} MB).")
    datos = p.read_bytes()
    try:
        with Image.open(io.BytesIO(datos)) as im:
            formato = im.format
            im.verify()
    except Exception:                                                                       # noqa: BLE001 - cualquier fallo de lectura es «imagen dañada»
        raise ImagenInvalida("No se pudo leer el archivo como imagen: está dañado o no es una imagen válida.") from None
    if formato not in FORMATOS:
        raise ImagenInvalida("Formato no admitido: el contenido del archivo no es PNG ni JPG.")
    try:
        im = Image.open(io.BytesIO(datos))
        ancho, alto = im.size
        if min(ancho, alto) < LADO_MIN or max(ancho, alto) > LADO_MAX:
            raise ImagenInvalida(f"Dimensiones no admitidas ({ancho}×{alto}): cada lado debe estar entre {LADO_MIN} y {LADO_MAX} píxeles.")
        im.load()
        modo = im.mode
        rgb = im.convert("RGB")
    except ImagenInvalida:
        raise
    except Exception:                                                                       # noqa: BLE001
        raise ImagenInvalida("No se pudo decodificar la imagen (archivo dañado, truncado o demasiado grande).") from None
    return rgb, {"archivo": p.name, "formato": formato, "ancho": ancho, "alto": alto, "modo_original": modo, "bytes": tam, "sha256_bytes_archivo": hashlib.sha256(datos).hexdigest()}
