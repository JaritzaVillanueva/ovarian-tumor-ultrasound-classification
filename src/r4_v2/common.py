"""Utilidades comunes de G2 (`g2_baseline_v1`): constantes, guardas de rutas, hashes, Git (solo lectura) y escritura
que no sobrescribe.

Vocabulario de hashes (protocolo §4.2), sin sinónimos: `git_blob_id` (SHA-1 de objeto Git, NO es SHA-256),
`sha256_contenido_blob` (SHA-256 de los bytes del blob de Git) y `sha256_bytes_archivo` (SHA-256 de los bytes
del archivo en disco).
"""

from __future__ import annotations

import csv
import fnmatch
import hashlib
import io
import json
import os
import re
import subprocess
from pathlib import Path, PurePath

RAIZ_REPO = Path(__file__).resolve().parents[2]

PROTOCOLO_ID = "g2_baseline_v1"
PARTICION_ID = "phash_le8_v1"
MAPEO_ID = "mapeo v2"
LEYENDA_PROVISIONAL = "partición provisional; conformidad académica pendiente"
ESQUEMA_CONFIG = "g2_baseline_v1_config/1"
CONFIG_REL = "configs/r4_v2/g2_baseline_v1.yaml"
PROTOCOLO_REL = "src/r4_v2/protocolo_g2_baseline_v1.md"
CAMPOS_HASH = ("git_blob_id", "sha256_contenido_blob", "sha256_bytes_archivo")
MODOS = ("referencia", "humo", "completa")
DIRECTORIOS_PROHIBIDOS_EN_REPO = {".claude", "agent-memory", "reportes", "version-informes"}  # pertenecen a Proyecto/


class GuardError(RuntimeError):
    """Fallo de una guarda metodológica: el proceso debe detenerse sin producir resultados."""


# ---------------------------------------------------------------- hashes
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_archivo(path) -> str:
    """`sha256_bytes_archivo`: bytes del archivo en disco, sin normalizar."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


# ---------------------------------------------------------------- rutas (K4)
def componentes(path) -> tuple:
    return PurePath(str(path).replace("\\", "/")).parts


def ruta_menciona_test(path) -> bool:
    """True si algún componente de la ruta es 'test' (sin distinguir mayúsculas)."""
    return any(p.lower() == "test" for p in componentes(path))


def exigir_ruta_no_prohibida(path) -> None:
    """K4: rechaza componentes `test`, los manifiestos mixtos y `cv_folds_e2.csv` (mapeo histórico)."""
    if ruta_menciona_test(path):
        raise GuardError(f"K4: la ruta contiene un componente 'test': {path}")
    nombre = componentes(path)[-1].lower() if componentes(path) else ""
    if fnmatch.fnmatch(nombre, "mmotu_experimental_split*.csv"):
        raise GuardError(f"K4: manifiesto mixto (contiene filas de test) rechazado: {path}")
    if nombre == "cv_folds_e2.csv":
        raise GuardError(f"K4: cv_folds_e2.csv (mapeo y folds históricos) rechazado: {path}")


def exigir_salida_segura(root, destino, prefijo) -> Path:
    """La salida debe quedar dentro de `root/prefijo` (ya resuelta), sin carpetas de agentes/reportes del proyecto."""
    root = Path(root).resolve()
    destino = Path(destino).resolve()
    permitido = (root / prefijo).resolve()
    if destino != permitido and permitido not in destino.parents:
        raise GuardError(f"K5: la salida debe estar dentro de {prefijo}/ (recibido: {destino})")
    malas = [p for p in destino.relative_to(root).parts if p.lower() in DIRECTORIOS_PROHIBIDOS_EN_REPO]
    if malas:
        raise GuardError(f"K5: la salida no puede usar carpetas de agentes o reportes del proyecto: {malas}")
    if ruta_menciona_test(destino.relative_to(root)):
        raise GuardError("K5: la salida contiene un componente 'test'")
    return destino


def directorio_nuevo(path) -> Path:
    """K5: crea un directorio que NO existía; nunca reutiliza ni sobrescribe."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError:
        raise GuardError(f"K5: el directorio de salida ya existe y no se sobrescribe: {path}") from None
    return path


# ---------------------------------------------------------------- escritura que no sobrescribe
def escribir_json_nuevo(path, datos) -> None:
    """Escribe JSON determinista (claves ordenadas) en un archivo nuevo; falla si existe."""
    texto = json.dumps(datos, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    try:
        with open(path, "x", encoding="utf-8", newline="\n") as f:
            f.write(texto)
    except FileExistsError:
        raise GuardError(f"K5: el archivo ya existe y no se sobrescribe: {path}") from None


def texto_csv(filas, columnas) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(columnas), lineterminator="\n", extrasaction="raise")
    w.writeheader()
    w.writerows(filas)
    return buf.getvalue()


def escribir_csv_nuevo(path, filas, columnas) -> None:
    try:
        with open(path, "x", encoding="utf-8", newline="") as f:
            f.write(texto_csv(filas, columnas))
    except FileExistsError:
        raise GuardError(f"K5: el archivo ya existe y no se sobrescribe: {path}") from None


# ---------------------------------------------------------------- configuración
def cargar_config(path) -> dict:
    """Lee la configuración congelada (YAML) y comprueba esquema e identificadores."""
    try:
        import yaml  # dependencia de la configuración; el entorno de Falcon la usa en los YAML históricos
    except ImportError as exc:  # pragma: no cover
        raise GuardError("Se requiere PyYAML para leer configs/r4_v2/g2_baseline_v1.yaml") from exc
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or config.get("esquema") != ESQUEMA_CONFIG:
        raise GuardError(f"Configuración con esquema distinto de {ESQUEMA_CONFIG!r}")
    if config["protocolo"]["id"] != PROTOCOLO_ID or config["estudio"]["particion"] != PARTICION_ID:
        raise GuardError("La configuración no corresponde a g2_baseline_v1 / phash_le8_v1")
    return config


# ---------------------------------------------------------------- Git (solo lectura)
def git(root, *args, entrada: bytes | None = None) -> subprocess.CompletedProcess:
    """Ejecuta `git -C root ...` y devuelve la salida en BYTES (sin recodificar). Solo se usan órdenes de lectura."""
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, input=entrada)


def git_texto(root, *args, obligatorio: bool = False):
    """Salida de texto sin espacios finales; None si la orden falla, salvo `obligatorio=True` (entonces GuardError)."""
    r = git(root, *args)
    if r.returncode != 0:
        if obligatorio:
            raise GuardError(f"K6: error de Git en 'git {' '.join(args[:3])}' (código {r.returncode}): "
                             f"{r.stderr.decode('utf-8', 'replace').strip()[:200]}")
        return None
    return r.stdout.decode("utf-8", "replace").strip()


def git_ok(root, *args) -> bytes:
    """Salida binaria de una orden Git de lectura; si falla lanza GuardError (los controles NUNCA pasan en silencio)."""
    r = git(root, *args)
    if r.returncode != 0:
        raise GuardError(f"K6: error de Git en 'git {' '.join(args[:3])}' (código {r.returncode}): "
                         f"{r.stderr.decode('utf-8', 'replace').strip()[:200]}")
    return r.stdout


def git_lista_z(root, *args) -> list:
    """Salida separada por NUL (rutas) como lista de cadenas. Un error de Git lanza GuardError: nunca devuelve [] en silencio."""
    return [p.decode("utf-8", "replace") for p in git_ok(root, *args).split(b"\0") if p]


# ---------------------------------------------------------------- formato de hashes recibidos por CLI
_SHA256 = re.compile(r"[0-9a-f]{64}")


def validar_sha256(valor, nombre: str = "sha256") -> str:
    """64 caracteres hexadecimales en MINÚSCULAS, sin espacios ni normalización (un valor en mayúsculas se rechaza)."""
    if not isinstance(valor, str) or not _SHA256.fullmatch(valor):
        raise GuardError(f"{nombre} debe ser un SHA-256 de 64 caracteres hexadecimales en minúsculas (sin espacios)")
    return valor
