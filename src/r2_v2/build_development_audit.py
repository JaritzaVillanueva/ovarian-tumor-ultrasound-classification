"""G1 / R2 v2: manifiesto de desarrollo con el mapeo vigente y auditoría de duplicados.

Fuente única: configs/splits/cv_folds_e2.csv (1000 imágenes del train oficial). No abre manifiestos mixtos.
Usa image_id, image_name, source_split, original_class y e1_split; ignora binary_label, binary_class y fold.
Las etiquetas v2 salen exclusivamente de original_class: 0,1,2,3,4,6 -> 0 (no maligna); 7 -> 1 (maligna); 5 excluida.
Las rutas se construyen desde e1_split y se verifican, ya resueltas, contra data/processed/MMOTU_OTU_2D_experimental/
{train,validation}/images. Nunca abre archivos de test.

Salidas (en --out, por defecto results/r2_v2/auditoria_desarrollo/), publicadas de una sola vez tras validarlas:
  development_eligible_v2.csv      lesiones elegibles (esperado: 820 = 782 no malignas + 38 malignas)
  normal_development_controls.csv  ovarios normales (esperado: 180; solo control auxiliar, no se entrenan)
  image_hashes.csv                 SHA-256 y pHash (64 bits) de las imágenes de desarrollo
  exact_duplicates.csv             grupos con SHA-256 idéntico
  revision_ciega.csv               PLANILLA DE REVISIÓN: solo hoja, revision, revisor y nota. No revela distancia,
                                   categoría original ni etiqueta. Es lo único que edita el revisor.
  candidate_pairs.csv              CORRESPONDENCIA (no se edita): pares elegibles con distancia pHash <= --cutoff
  calibration_pairs.csv            CORRESPONDENCIA (no se edita): muestra reproducible justo por encima del corte
  review_sheets/                   hojas par_XXXX.png (orden aleatorio; la imagen no muestra distancia ni origen);
                                   no se versionan.
  resumen_auditoria.json           recuentos, parámetros, procedencia y hashes de las salidas

El revisor trabaja solo con la planilla y las hojas. La correspondencia hoja -> pareja queda en los CSV de pares,
cuyo hash exacto se verifica antes de usar las revisiones para generar folds.

Uso (desde la raíz del repositorio, en Falcon, con el código ya en un commit):
  python -m src.r2_v2.build_development_audit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dct

from . import common as C

MODULE = "src.r2_v2.build_development_audit"
DEFAULT_OUT_REL = "results/r2_v2/auditoria_desarrollo"
ALLOWED_OUT_PREFIX = "results/r2_v2"  # --out debe quedar DENTRO de este directorio (nunca igual ni fuera)
REVIEW_SHEET_FILE = "revision_ciega.csv"
PROTECTED_FILES = (REVIEW_SHEET_FILE,)  # lleva la revisión humana: nunca se sobrescribe, ni con --force
KEY_FILES = ("candidate_pairs.csv", "calibration_pairs.csv")  # correspondencia hoja -> pareja (no se edita)

# Aprobados por el autor para la AUDITORÍA INICIAL (cribado, no prueba de identidad): corte <= 12 y banda 13-18.
# No congelan la partición: 5 folds, semilla 42 y mínimo de 4 positivas siguen siendo propuesta (make_folds_v2).
DEFAULT_CUTOFF = 12
CALIBRATION_BANDS = [(13, 14), (15, 16), (17, 18)]
CALIBRATION_PER_BAND = 20
SAMPLING_SEED = 42

ELIGIBLE_COLUMNS = ["image_id", "image_name", "original_class", "label_v2", "class_v2", "e1_split", "image_path"]
NORMAL_COLUMNS = ["image_id", "image_name", "original_class", "e1_split", "image_path"]
HASH_COLUMNS = ["image_id", "original_class", "eligible", "sha256", "phash_hex", "width", "height"]
DUP_COLUMNS = ["sha256", "image_ids", "n", "clases_originales", "categorias_originales_distintas",
               "n_elegibles", "mezcla_normal_y_lesion", "etiquetas_binarias_contradictorias"]
KEY_COLUMNS = ["image_id_a", "image_id_b", "distance", "original_class_a", "original_class_b",
               "label_a", "label_b", "categorias_originales_distintas", "etiquetas_contradictorias", "hoja"]
BLIND_COLUMNS = ["hoja", "revision", "revisor", "nota"]  # EXACTAMENTE esto ve y edita el revisor
SHEETS_DIR = "review_sheets"

OPENED: set = set()  # rutas resueltas de imágenes realmente abiertas en esta ejecución (medido, no declarado)


# ---------------------------------------------------------------- manifiestos y recuentos
def build_manifests(dev_rows):
    """Elegibles (con label_v2 desde original_class) y normales. Ignora cualquier columna histórica."""
    eligible, normals = [], []
    for r in dev_rows:
        c = int(r["original_class"])
        base = {"image_id": r["image_id"], "image_name": r["image_name"], "original_class": c,
                "e1_split": r["e1_split"], "image_path": C.dev_image_relpath(r["e1_split"], r["image_name"])}
        label = C.label_from_class(c)
        if label is None:
            normals.append(base)
        else:
            eligible.append({**base, "label_v2": label, "class_v2": C.CLASS_NAMES_V2[label]})
    eligible.sort(key=lambda r: int(r["image_id"]))
    normals.sort(key=lambda r: int(r["image_id"]))
    return eligible, normals


def check_counts(eligible, normals, expected=C.EXPECTED):
    got = {"eligible": len(eligible),
           "non_malignant": sum(1 for r in eligible if r["label_v2"] == 0),
           "malignant": sum(1 for r in eligible if r["label_v2"] == 1),
           "normal": len(normals)}
    if got != expected:
        raise C.GuardError(f"Recuentos inesperados: {got} (esperado {expected})")
    return got


# ---------------------------------------------------------------- hashing perceptual
def phash(image, hash_size=8, highfreq_factor=4):
    """pHash DCT de 64 bits (variante de Zauner, 2010): gris, 32x32, DCT 2D, bloque 8x8, mediana sin el DC."""
    size = hash_size * highfreq_factor
    img = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(img, dtype=np.float64)
    coeffs = dct(dct(pixels, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = coeffs[:hash_size, :hash_size]
    med = np.median(low.flatten()[1:])  # excluye el coeficiente DC
    bits = (low > med).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return (a ^ b).bit_count()


def pairwise_hamming(hashes):
    """Matriz de distancias de Hamming entre hashes de 64 bits."""
    arr = np.array(hashes, dtype=np.uint64)
    x = arr[:, None] ^ arr[None, :]
    bytes_view = x.view(np.uint8).reshape(len(arr), len(arr), 8)
    return np.unpackbits(bytes_view, axis=2).sum(axis=2)


def all_pairs(ids, hashes):
    d = pairwise_hamming(hashes)
    i, j = np.triu_indices(len(ids), k=1)
    return [(ids[a], ids[b], int(d[a, b])) for a, b in zip(i, j)]


# ---------------------------------------------------------------- imágenes (solo rutas ya validadas)
def open_image(path):
    OPENED.add(str(path))
    return Image.open(path)


def compute_hashes(root, rows):
    """SHA-256 y pHash de cada imagen de desarrollo, abriendo solo rutas resueltas y autorizadas."""
    hash_rows, by_id = [], {}
    for r in sorted(rows, key=lambda r: int(r["image_id"])):
        resolved = C.resolve_dev_image(root, r["e1_split"], r["image_name"])
        with open_image(resolved) as im:
            ph = phash(im)
            w, h = im.size
        row = {"image_id": r["image_id"], "original_class": r["original_class"],
               "eligible": int(r["original_class"]) not in C.EXCLUDED_CLASSES,
               "sha256": C.sha256_file(resolved), "phash_hex": f"{ph:016x}", "width": w, "height": h}
        hash_rows.append(row)
        by_id[r["image_id"]] = (ph, resolved)
    return hash_rows, by_id


def find_exact_duplicates(hash_rows, label_of):
    """Grupos con SHA-256 idéntico. `categorias_originales_distintas` es informativo y se mantiene
    independiente de `etiquetas_binarias_contradictorias` (que usa solo label_v2 de los elegibles)."""
    groups = {}
    for row in hash_rows:
        groups.setdefault(row["sha256"], []).append(row)
    out = []
    for sha, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        ids = [m["image_id"] for m in members]
        classes = [int(m["original_class"]) for m in members]
        elig_labels = {label_of[i] for i in ids if i in label_of}
        out.append({"sha256": sha, "image_ids": ";".join(ids), "n": len(ids),
                    "clases_originales": ";".join(str(c) for c in classes),
                    "categorias_originales_distintas": len(set(classes)) > 1,
                    "n_elegibles": sum(1 for i in ids if i in label_of),
                    "mezcla_normal_y_lesion": any(c in C.EXCLUDED_CLASSES for c in classes)
                    and any(c not in C.EXCLUDED_CLASSES for c in classes),
                    "etiquetas_binarias_contradictorias": len(elig_labels) > 1})
    return out


def make_review_sheet(path_a, path_b, out_path, height=256):
    imgs = []
    for p in (path_a, path_b):
        with open_image(p) as im:
            im = im.convert("RGB")
            w = max(1, int(im.width * height / im.height))
            imgs.append(im.resize((w, height)))
    sheet = Image.new("RGB", (imgs[0].width + imgs[1].width + 10, height), "white")
    sheet.paste(imgs[0], (0, 0))
    sheet.paste(imgs[1], (imgs[0].width + 10, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


# ---------------------------------------------------------------- publicación segura
def validate_staging(staging, expected, expected_total, with_sheets):
    """Valida los artefactos preparados ANTES de publicarlos. Lanza GuardError ante cualquier inconsistencia."""
    staging = Path(staging)
    need = ["development_eligible_v2.csv", "normal_development_controls.csv", "image_hashes.csv",
            "exact_duplicates.csv", *KEY_FILES, *PROTECTED_FILES, "resumen_auditoria.json"]
    absent = [n for n in need if not (staging / n).is_file()]
    if absent:
        raise C.GuardError(f"Faltan artefactos preparados: {absent}")
    summary = json.loads((staging / "resumen_auditoria.json").read_text(encoding="utf-8"))
    for name, digest in summary["salidas_sha256"].items():
        if C.sha256_file(staging / name) != digest:
            raise C.GuardError(f"El hash de {name} no coincide con el resumen")
    eligible = C.read_csv_rows(staging / "development_eligible_v2.csv")
    normals = C.read_csv_rows(staging / "normal_development_controls.csv")
    if any(int(r["original_class"]) in C.EXCLUDED_CLASSES for r in eligible):
        raise C.GuardError("Hay categoría excluida entre los elegibles")
    if any(int(r["label_v2"]) != C.label_from_class(int(r["original_class"])) for r in eligible):
        raise C.GuardError("label_v2 no coincide con el mapeo desde original_class")
    if any(int(r["original_class"]) not in C.EXCLUDED_CLASSES for r in normals):
        raise C.GuardError("Hay lesiones entre los normales")
    check_counts([{**r, "label_v2": int(r["label_v2"])} for r in eligible], normals, expected)
    if len(C.read_csv_rows(staging / "image_hashes.csv")) != expected_total:
        raise C.GuardError("image_hashes.csv no cubre todas las imágenes de desarrollo")
    key_hojas = []
    for name in KEY_FILES:
        header, rows = C.read_csv_with_header(staging / name)
        if header != KEY_COLUMNS:
            raise C.GuardError(f"{name}: columnas inesperadas")
        key_hojas += [r["hoja"] for r in rows]
    header, sheet_rows = C.read_csv_with_header(staging / REVIEW_SHEET_FILE)
    if header != BLIND_COLUMNS:  # la planilla no puede revelar distancia, categoría ni etiqueta
        raise C.GuardError(f"{REVIEW_SHEET_FILE}: columnas {header}; deben ser exactamente {BLIND_COLUMNS}")
    sheet_hojas = [r["hoja"] for r in sheet_rows]
    if len(set(sheet_hojas)) != len(sheet_hojas) or sorted(sheet_hojas) != sorted(key_hojas) or "" in sheet_hojas:
        raise C.GuardError(f"{REVIEW_SHEET_FILE} no corresponde uno a uno con los pares de la correspondencia")
    if any(r["revision"] or r["revisor"] or r["nota"] for r in sheet_rows):
        raise C.GuardError(f"{REVIEW_SHEET_FILE} debe publicarse sin anotaciones")
    if with_sheets and any(not (staging / h).is_file() for h in sheet_hojas):
        raise C.GuardError("Falta alguna hoja de revisión")
    for p in staging.rglob("*"):
        if p.is_file() and C.path_mentions_test(p.relative_to(staging)):
            raise C.GuardError(f"Artefacto con ruta de test: {p}")
    return summary


def main(argv=None, project_root=C.PROJECT_ROOT, expected=None, expected_total=None):
    """`expected`/`expected_total` existen solo para pruebas sintéticas; la CLI usa los valores de G1."""
    expected = C.EXPECTED if expected is None else expected
    expected_total = C.EXPECTED_TOTAL if expected_total is None else expected_total
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None, help=f"Directorio de salida (por defecto {DEFAULT_OUT_REL}).")
    ap.add_argument("--cutoff", type=int, default=DEFAULT_CUTOFF)
    ap.add_argument("--no-sheets", action="store_true", help="No genera las hojas de revisión.")
    ap.add_argument("--force", action="store_true",
                    help="Reemplaza salidas previas SIN revisión humana (se conservan renombradas). "
                         "Nunca sobrescribe archivos revisados.")
    ap.add_argument("--permitir-codigo-sin-commit", action="store_true",
                    help="Solo para pruebas: el resultado no sirve como evidencia de G1.")
    args = ap.parse_args(argv)

    root = Path(project_root).resolve()
    out = (args.out if args.out is not None else root / DEFAULT_OUT_REL)
    out = (out if out.is_absolute() else root / out).resolve()
    staging = None
    OPENED.clear()
    try:
        C.require_safe_out(out, root, ALLOWED_OUT_PREFIX)
        C.prepare_output(out, PROTECTED_FILES, args.force)
        prov = C.provenance(MODULE, argv)
        versioned = C.require_versioned_code(prov, args.permitir_codigo_sin_commit)

        source_path = root / C.SOURCE_REL
        if not source_path.is_file():
            raise C.GuardError(f"No existe la fuente única: {C.SOURCE_REL}")
        dev = C.load_development_source(source_path, expected_total)
        eligible, normals = build_manifests(dev)
        counts = check_counts(eligible, normals, expected)

        hash_rows, by_id = compute_hashes(root, eligible + normals)
        label_of = {r["image_id"]: r["label_v2"] for r in eligible}
        ocls = {r["image_id"]: r["original_class"] for r in eligible}
        dups = find_exact_duplicates(hash_rows, label_of)

        el_ids = [r["image_id"] for r in eligible]
        pairs_all = all_pairs(el_ids, [by_id[i][0] for i in el_ids])

        def pair_row(a, b, d):
            return {"image_id_a": a, "image_id_b": b, "distance": d,
                    "original_class_a": ocls[a], "original_class_b": ocls[b],
                    "label_a": label_of[a], "label_b": label_of[b],
                    "categorias_originales_distintas": ocls[a] != ocls[b],
                    "etiquetas_contradictorias": label_of[a] != label_of[b], "hoja": ""}

        cands = [pair_row(a, b, d) for a, b, d in pairs_all if d <= args.cutoff]
        cands.sort(key=lambda r: (r["distance"], int(r["image_id_a"]), int(r["image_id_b"])))
        rng = random.Random(SAMPLING_SEED)
        calib = []
        for lo, hi in CALIBRATION_BANDS:
            band = sorted((p for p in pairs_all if lo <= p[2] <= hi and p[2] > args.cutoff),
                          key=lambda p: (p[2], int(p[0]), int(p[1])))
            calib += [pair_row(*p) for p in rng.sample(band, min(CALIBRATION_PER_BAND, len(band)))]

        # Hojas numeradas al azar: la planilla no revela la distancia ni si el par es candidato o de calibración.
        combined = cands + calib
        order = list(range(len(combined)))
        random.Random(SAMPLING_SEED + 1).shuffle(order)
        width = max(4, len(str(len(combined))))
        sheets = []
        for number, idx in enumerate(order, 1):
            row = combined[idx]
            row["hoja"] = f"{SHEETS_DIR}/par_{number:0{width}d}.png"  # identificador; el PNG solo existe sin --no-sheets
            if not args.no_sheets:
                sheets.append((row["hoja"], by_id[row["image_id_a"]][1], by_id[row["image_id_b"]][1]))
        blind = [{"hoja": r["hoja"], "revision": "", "revisor": "", "nota": ""}
                 for r in sorted(combined, key=lambda r: r["hoja"])]

        hist = {}
        for _, _, d in pairs_all:
            if d <= 20:
                hist[d] = hist.get(d, 0) + 1

        # ---- preparación temporal: nada llega a `out` hasta validar todo
        staging = C.new_staging(out)
        files = {
            "development_eligible_v2.csv": C.csv_text(eligible, ELIGIBLE_COLUMNS),
            "normal_development_controls.csv": C.csv_text(normals, NORMAL_COLUMNS),
            "image_hashes.csv": C.csv_text(hash_rows, HASH_COLUMNS),
            "exact_duplicates.csv": C.csv_text(dups, DUP_COLUMNS),
            "candidate_pairs.csv": C.csv_text(cands, KEY_COLUMNS),
            "calibration_pairs.csv": C.csv_text(calib, KEY_COLUMNS),
        }
        for name, text in files.items():
            (staging / name).write_text(text, encoding="utf-8", newline="")
        (staging / REVIEW_SHEET_FILE).write_text(C.csv_text(blind, BLIND_COLUMNS), encoding="utf-8", newline="")
        if sheets:  # el .gitignore va PRIMERO: un directorio temporal huérfano no deja ecografías sin ignorar
            (staging / SHEETS_DIR).mkdir()
            (staging / SHEETS_DIR / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
        for rel, path_a, path_b in sheets:
            make_review_sheet(path_a, path_b, staging / rel)

        sheet_digest = hashlib.sha256("\n".join(
            f"{rel},{C.sha256_file(staging / rel)}" for rel, _, _ in sorted(sheets)).encode()).hexdigest()
        summary = {
            "etapa": "G1 / R2 v2 - auditoria de desarrollo",
            "estado": ("Auditoria para revision manual. Folds NO congelados." if versioned else
                       "NO VALIDA COMO EVIDENCIA DE G1: codigo o fuente sin commit (ejecucion de prueba)."),
            "procedencia": prov,
            "codigo_en_commit": prov["codigo_en_commit"],
            "fuente": {"ruta": C.SOURCE_REL, "sha256": C.sha256_file(root / C.SOURCE_REL),
                       "columnas_usadas": list(C.SOURCE_COLUMNS),
                       "columnas_historicas_ignoradas": list(C.HISTORICAL_COLUMNS),
                       "filas": len(dev), "source_split_solo_train": True},
            "mapeo_v2": {str(k): v for k, v in C.LABEL_MAP_V2.items()},
            "categorias_excluidas": sorted(C.EXCLUDED_CLASSES),
            "recuentos": counts,
            "duplicados_exactos_grupos": len(dups),
            "duplicados_exactos_con_categorias_originales_distintas":
                sum(1 for d in dups if d["categorias_originales_distintas"]),
            "duplicados_exactos_con_etiquetas_binarias_contradictorias":
                sum(1 for d in dups if d["etiquetas_binarias_contradictorias"]),
            "duplicados_exactos_con_mezcla_normal_y_lesion": sum(1 for d in dups if d["mezcla_normal_y_lesion"]),
            "pares_candidatos_con_categorias_originales_distintas":
                sum(1 for c in cands if c["categorias_originales_distintas"]),
            "pares_candidatos_con_etiquetas_contradictorias": sum(1 for c in cands if c["etiquetas_contradictorias"]),
            "phash": {"bits": 64, "corte_cribado": args.cutoff, "bandas_calibracion": CALIBRATION_BANDS,
                      "estado_parametros": "aprobados por el autor para la auditoria inicial (no congelan la particion)",
                      "pares_candidatos": len(cands), "pares_calibracion": len(calib),
                      "histograma_distancias_0_20": {str(k): hist[k] for k in sorted(hist)}},
            "planilla_ciega": {"archivo": REVIEW_SHEET_FILE, "columnas": BLIND_COLUMNS, "n_hojas": len(blind),
                               "hojas_sha256": C.hojas_digest(r["hoja"] for r in blind),
                               "correspondencia": list(KEY_FILES)},
            "hojas_revision": {"n": len(sheets), "huella_sha256": sheet_digest, "versionar": False},
            "acceso_a_datos": {
                "imagenes_distintas_abiertas": len(OPENED),
                "directorios_autorizados": [f"{C.DATA_REL}/{s}/images" for s in C.DEV_E1_SPLITS],
                "rutas_resueltas_verificadas": True,
                "manifiestos_mixtos_abiertos": False,  # por diseño: el código solo referencia la fuente única
            },
            "salidas_sha256": {n: C.sha256_file(staging / n) for n in sorted(files)},
            "nota": "pHash detecta duplicados y casi-duplicados visuales; no garantiza independencia por paciente. "
                    "calibration_pairs.csv es la muestra de control fuera del corte. La planilla revision_ciega.csv "
                    "no revela distancia, categoria original ni etiqueta; la correspondencia esta en los CSV de pares.",
        }
        C.write_json(staging / "resumen_auditoria.json", summary)
        validate_staging(staging, expected, expected_total, with_sheets=not args.no_sheets)

        C.prepare_output(out, PROTECTED_FILES, args.force)  # el destino pudo cambiar durante el cálculo
        backup = C.publish_staging(staging, out)
        staging = None
        print(json.dumps({k: summary[k] for k in ("estado", "recuentos", "duplicados_exactos_grupos", "phash")},
                         indent=2, ensure_ascii=False))
        if backup:
            print(f"Salidas anteriores conservadas en: {backup}")
        return 0
    except C.GuardError as e:
        print(f"ERROR (no se publicó ninguna salida): {e}", file=sys.stderr)
        return 1
    finally:
        if staging is not None:
            C.discard_staging(staging)


if __name__ == "__main__":
    sys.exit(main())
