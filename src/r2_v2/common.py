"""Utilidades comunes de G1 / R2 v2: fuente única de desarrollo, guardas contra el test,
publicación segura de artefactos y procedencia.

Fuente única: configs/splits/cv_folds_e2.csv (1000 imágenes del train oficial, source_split == "train").
Este módulo NO abre los manifiestos mixtos (los que contienen filas del test oficial).
De la fuente solo se usan image_id, image_name, source_split, original_class y e1_split.
binary_label, binary_class y fold pertenecen al mapeo histórico y se ignoran al decidir.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePath

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]

# ---------------------------------------------------------------- fuente y rutas
SOURCE_REL = "configs/splits/cv_folds_e2.csv"
DATA_REL = "data/processed/MMOTU_OTU_2D_experimental"
DEV_SOURCE_SPLIT = "train"
DEV_E1_SPLITS = ("train", "validation")  # carpetas físicas del desarrollo oficial (E1)
SOURCE_COLUMNS = ("image_id", "image_name", "source_split", "original_class", "e1_split")
HISTORICAL_COLUMNS = ("binary_label", "binary_class", "fold")  # mapeo histórico: nunca se usan para decidir

# ---------------------------------------------------------------- mapeo vigente (v2)
LABEL_MAP_V2 = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 6: 0, 7: 1}
EXCLUDED_CLASSES = {5}  # ovario normal: fuera de la tarea principal
KNOWN_CLASSES = set(LABEL_MAP_V2) | EXCLUDED_CLASSES
CLASS_NAMES_V2 = {0: "lesion_no_maligna", 1: "lesion_maligna"}

# Recuentos ESPERADOS para verificar contra los datos de desarrollo (no son un resultado).
EXPECTED_TOTAL = 1000
EXPECTED = {"eligible": 820, "non_malignant": 782, "malignant": 38, "normal": 180}

REVIEW_FIELDS = ("revision", "revisor", "nota")


class GuardError(RuntimeError):
    """Fallo de una guarda metodológica: el proceso debe detenerse sin escribir salidas."""


# ---------------------------------------------------------------- lectura de la fuente
def path_mentions_test(path) -> bool:
    """True si algún componente de la ruta (normalizada) es 'test', sin distinguir mayúsculas."""
    parts = PurePath(str(path).replace("\\", "/")).parts
    return any(p.lower() == "test" for p in parts)


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_csv_with_header(path):
    """(columnas, filas); también devuelve las columnas cuando no hay filas."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def check_image_name(name):
    """Solo un nombre de archivo simple: sin separadores, '..', rutas absolutas ni unidades."""
    norm = name.replace("\\", "/")
    if (not name or norm != PurePath(norm).name or ".." in name or ":" in name
            or name.startswith(".") or path_mentions_test(name)):
        raise GuardError(f"image_name no seguro: {name!r}")


def label_from_class(original_class):
    """Etiqueta v2 desde original_class: 0/1, o None si la categoría está excluida (5)."""
    if original_class in EXCLUDED_CLASSES:
        return None
    if original_class in LABEL_MAP_V2:
        return LABEL_MAP_V2[original_class]
    raise GuardError(f"Categoría desconocida: {original_class}")


def load_development_source(path, expected_total=EXPECTED_TOTAL):
    """Lee SOLO la fuente de desarrollo y la valida de forma estricta.

    Devuelve filas con exactamente SOURCE_COLUMNS (original_class como int). Las columnas históricas
    y cualquier otra columna se descartan sin leerse. Cualquier fila que no sea train aborta.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in SOURCE_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise GuardError(f"La fuente no tiene las columnas requeridas: {missing}")
        rows = []
        for raw in reader:
            row = {c: (raw.get(c) or "").strip() for c in SOURCE_COLUMNS}
            who = f"image_id {row['image_id']!r}"
            if row["source_split"] != DEV_SOURCE_SPLIT:
                raise GuardError(f"{who}: source_split {row['source_split']!r} (solo se admite 'train')")
            if row["e1_split"] not in DEV_E1_SPLITS:
                raise GuardError(f"{who}: e1_split {row['e1_split']!r} fuera de {list(DEV_E1_SPLITS)}")
            if not row["image_id"].isdigit():
                raise GuardError(f"image_id no numérico: {row['image_id']!r}")
            check_image_name(row["image_name"])
            try:
                row["original_class"] = int(row["original_class"])
            except ValueError:
                raise GuardError(f"{who}: original_class no entero: {row['original_class']!r}") from None
            if row["original_class"] not in KNOWN_CLASSES:
                raise GuardError(f"{who}: categoría desconocida {row['original_class']}")
            rows.append(row)
    ids = [r["image_id"] for r in rows]
    names = [r["image_name"] for r in rows]
    errors = []
    if len(rows) != expected_total:
        errors.append(f"{len(rows)} filas de desarrollo (esperado {expected_total})")
    if len(set(ids)) != len(ids):
        errors.append("image_id repetidos")
    if len(set(names)) != len(names):
        errors.append("image_name repetidos")
    if errors:
        raise GuardError("Integridad de la fuente: " + "; ".join(errors))
    return rows


def dev_image_relpath(e1_split, image_name):
    return f"{DATA_REL}/{e1_split}/images/{image_name}"


def resolve_dev_image(root, e1_split, image_name):
    """Ruta resuelta de una imagen de desarrollo, verificada contra los directorios autorizados.

    Los únicos directorios autorizados son <DATA_REL>/{train,validation}/images. La ruta resuelta
    (siguiendo enlaces simbólicos) debe seguir en ese mismo directorio. No abre el archivo.
    """
    if e1_split not in DEV_E1_SPLITS:
        raise GuardError(f"e1_split no autorizado: {e1_split!r}")
    check_image_name(image_name)
    data_root = (Path(root) / DATA_REL).resolve()
    allowed = data_root / e1_split / "images"
    allowed_resolved = allowed.resolve()
    if os.path.normcase(str(allowed_resolved)) != os.path.normcase(str(allowed)):
        raise GuardError(f"El directorio autorizado está redirigido (enlace simbólico): {e1_split}/images")
    resolved = (allowed_resolved / image_name).resolve()
    if (os.path.normcase(str(resolved.parent)) != os.path.normcase(str(allowed_resolved))
            or os.path.normcase(resolved.name) != os.path.normcase(image_name)):
        raise GuardError(f"La ruta resuelta de {image_name!r} sale del directorio autorizado {e1_split}/images")
    if path_mentions_test(resolved.relative_to(data_root)):
        raise GuardError(f"Ruta de test detectada: {resolved}")
    if not resolved.is_file():
        raise GuardError(f"Imagen de desarrollo no encontrada: {dev_image_relpath(e1_split, image_name)}")
    return resolved


# ---------------------------------------------------------------- hashes y CSV
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_text(rows, columns):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def hojas_digest(hojas):
    """Huella del conjunto de hojas de la planilla ciega, independiente de las anotaciones de revisión."""
    return hashlib.sha256("\n".join(sorted(hojas)).encode("utf-8")).hexdigest()


def as_bool(value):
    return str(value).strip().lower() == "true"


# ---------------------------------------------------------------- protección y publicación segura
def has_review_annotations(path):
    """True si el CSV tiene alguna anotación humana (revision/revisor/nota). Ante duda, True."""
    p = Path(path)
    if not p.exists():
        return False
    try:
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if any((r.get(k) or "").strip() for k in REVIEW_FIELDS):
                    return True
    except Exception:
        return True  # ilegible: se protege
    return False


FORBIDDEN_OUT_DIRS = {".claude", "agent-memory", "reportes", "version-informes"}  # pertenecen a Proyecto/, no al repo


def require_safe_out(out, root, allowed_prefix, allow_equal=False, forbidden_paths=()):
    """Restringe dónde puede escribir (y con --force, reemplazar) un script.

    `out` debe quedar dentro de `root/allowed_prefix` (ya resuelto, sin salir por enlaces), no puede ser
    la raíz, `data`, `configs/splits` u otro ancestro, ni contener o ser alguna de `forbidden_paths`
    (p. ej. el directorio de la auditoría revisada), ni usar nombres de directorios de agentes o reportes.
    """
    out, root = Path(out).resolve(), Path(root).resolve()
    allowed = (root / allowed_prefix).resolve()
    try:
        rel = out.relative_to(root)
    except ValueError:
        raise GuardError(f"--out debe estar dentro del repositorio ({root}): {out}") from None
    bad = [p for p in rel.parts if p.lower() in FORBIDDEN_OUT_DIRS]
    if bad:
        raise GuardError(f"--out no puede usar directorios de agentes o reportes ({bad}): esos pertenecen a Proyecto/, "
                         f"no al repositorio.")
    inside = out == allowed if allow_equal else False
    inside = inside or allowed in out.parents
    if not inside:
        raise GuardError(f"--out debe estar {'en' if allow_equal else 'dentro de'} {allowed_prefix}/ "
                         f"(recibido: {rel.as_posix() or '.'})")
    for other in forbidden_paths:
        other = Path(other).resolve()
        if out == other or other in out.parents or out in other.parents:
            raise GuardError(f"--out no puede coincidir con ni contener a {other}")


def prepare_output(out, protected_names, force):
    """Comprueba que se puede publicar en `out`.

    Los archivos con anotaciones humanas NUNCA se sobrescriben, ni siquiera con force.
    Si `out` ya tiene contenido, solo se reemplaza con force (y el contenido anterior se conserva).
    """
    out = Path(out)
    if not out.exists():
        return
    if not out.is_dir():
        raise GuardError(f"La ruta de salida existe y no es un directorio: {out}")
    reviewed = [n for n in protected_names if has_review_annotations(out / n)]
    if reviewed:
        raise GuardError("Hay archivos con revisión humana; nunca se sobrescriben, ni con --force: "
                         + ", ".join(reviewed) + ". Archívalos manualmente o elige otra ruta de salida.")
    if any(out.iterdir()) and not force:
        raise GuardError(f"Ya existen salidas en {out}; no se sobrescriben sin --force.")


def new_staging(out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.parent / f".{out.name}.staging-{os.getpid()}"
    if staging.exists():
        raise GuardError(f"Existe un directorio temporal previo: {staging}")
    staging.mkdir()
    return staging


def discard_staging(staging):
    staging = Path(staging)
    if staging.exists() and staging.name.startswith(".") and ".staging-" in staging.name:
        shutil.rmtree(staging)


def publish_staging(staging, out):
    """Publica el directorio temporal ya validado. El contenido anterior se conserva renombrado."""
    out = Path(out)
    backup = None
    if out.exists():
        if any(out.iterdir()):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup = out.with_name(f"{out.name}.reemplazado-{stamp}")
            n = 1
            while backup.exists():
                n += 1
                backup = out.with_name(f"{out.name}.reemplazado-{stamp}-{n}")
            os.replace(out, backup)
        else:
            out.rmdir()
    try:
        os.replace(staging, out)
    except Exception:
        if backup is not None:
            os.replace(backup, out)
        raise
    return backup


# ---------------------------------------------------------------- procedencia
def _git(root, *args):
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def package_code_files():
    """Archivos .py del paquete (el código que se ejecuta), relativos a la raíz del repositorio."""
    return [p.relative_to(PROJECT_ROOT).as_posix() for p in sorted(PACKAGE_DIR.glob("*.py"))]


def code_state(git_root, rel_files, runner=None):
    """Verifica que cada archivo está versionado y sin cambios respecto de HEAD (blob idéntico)."""
    git = runner or _git
    head = git(git_root, "rev-parse", "HEAD")
    files, ok = {}, head is not None
    for rel in rel_files:
        in_head = git(git_root, "rev-parse", f"HEAD:{rel}") if head else None
        in_work = git(git_root, "hash-object", "--", rel)
        committed = in_head is not None and in_work is not None and in_head == in_work
        path = Path(git_root) / rel
        # `blob` (id de git) no depende de la plataforma (saltos de línea); `sha256` es del archivo tal cual.
        files[rel] = {"sha256": sha256_file(path) if path.is_file() else None, "blob": in_work,
                      "en_commit": committed}
        ok = ok and committed
    return {"commit": head, "en_commit": bool(ok), "archivos": files}


def provenance(module, argv, runner=None):
    """Procedencia de una ejecución: commit, estado del código y de la fuente, versiones y comando."""
    git = runner or _git
    code = code_state(PROJECT_ROOT, package_code_files(), runner)
    source = code_state(PROJECT_ROOT, [SOURCE_REL], runner)
    status = git(PROJECT_ROOT, "status", "--porcelain")
    versions = {"python": platform.python_version()}
    for mod in ("numpy", "scipy", "PIL", "sklearn"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    return {
        "commit": code["commit"],
        "codigo_en_commit": code["en_commit"],
        "fuente_en_commit": source["en_commit"],
        "archivos_codigo": code["archivos"],
        "arbol_con_cambios": bool(status) if status is not None else None,
        "archivos_sin_commit": status.splitlines()[:50] if status else [],
        "comando": " ".join([Path(sys.executable).name, "-m", module, *argv]),
        "versiones": versions,
    }


def require_versioned_code(prov, allow_uncommitted):
    """Aborta si el código o la fuente no están versionados y sin cambios, salvo opt-in explícito."""
    problems = []
    if not prov["codigo_en_commit"]:
        bad = [k for k, v in prov["archivos_codigo"].items() if not v["en_commit"]]
        problems.append("código sin commit o modificado respecto de HEAD: " + (", ".join(bad) or "git no disponible"))
    if not prov["fuente_en_commit"]:
        problems.append(f"{SOURCE_REL} sin commit o modificada respecto de HEAD")
    if problems and not allow_uncommitted:
        raise GuardError("; ".join(problems) + ". Haz commit antes de ejecutar (o usa --permitir-codigo-sin-commit "
                         "solo para pruebas: el resultado NO sirve como evidencia de G1).")
    return not problems


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
