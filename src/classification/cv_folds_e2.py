"""
Esquema de evaluación E2: validación cruzada estratificada de 5 particiones sobre las 1000 imágenes
del train oficial de MMOTU (las 800 de train y las 200 de validation del esquema E1). Test no entra.

Es estratificada por etiqueta binaria y NO está agrupada por paciente (no hay identificadores de paciente).
Por eso puede sobreestimar el desempeño y solo sirve como chequeo de robustez y selección de
configuraciones, no como estimación final.

Solo usa la biblioteca estándar (sin pandas ni torch), de modo que el archivo generado es el mismo en
cualquier máquina.

Uso (desde la raíz del repositorio):
    python -m src.classification.cv_folds_e2            # genera configs/splits/cv_folds_e2.csv (no sobrescribe)
    python -m src.classification.cv_folds_e2 --verify   # verifica el CSV versionado y lo regenera en memoria
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from pathlib import Path

SEED_E2 = 42  # semilla fija de la partición E2 (no es la semilla de entrenamiento)
N_FOLDS = 5
EXPECTED_N_IMAGES = 1000

MANIFEST_REL = "configs/splits/mmotu_experimental_split_standard.csv"
OUTPUT_REL = "configs/splits/cv_folds_e2.csv"
FOLD_MANIFEST_NAME = "fold_manifest.csv"

OFFICIAL_TRAIN_SPLITS = ("train", "validation")  # particiones E1 que forman el train oficial de MMOTU
TEST_SPLIT = "test"

FOLD_COLUMNS = [
    "image_id", "image_name", "source_split", "original_class", "binary_label", "binary_class",
    "e1_split", "fold",
]
# Manifiesto derivado de un fold: mismas columnas que el manifiesto operativo + e1_split y fold.
# `experimental_split` (train/validation) se redefine por fold; `e1_split` conserva la carpeta física
# de la imagen (data/processed/.../<e1_split>/images/).
DERIVED_COLUMNS = [
    "image_id", "image_name", "source_split", "original_class", "binary_label", "binary_class",
    "experimental_split", "e1_split", "fold",
]


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path, rows, columns):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_official_train(manifest_rows):
    """
    Devuelve las filas del train oficial de MMOTU (train + validation de E1). Excluye test.
    """
    rows = [r for r in manifest_rows if r["experimental_split"] in OFFICIAL_TRAIN_SPLITS]
    not_official = [r["image_id"] for r in rows if r["source_split"] != "train"]
    if not_official:
        raise ValueError(f"Filas fuera del train oficial de MMOTU (source_split != 'train'): {not_official[:5]}")
    return rows


def assign_folds(rows, seed=SEED_E2, n_folds=N_FOLDS):
    """
    Asigna cada imagen a un fold, estratificando por binary_label.

    Las imágenes se ordenan por image_id (independiente del orden del archivo), se barajan dentro de cada
    clase con random.Random(seed) y se reparten por turnos. El turno continúa entre clases (positivas
    primero), de modo que los restos de cada clase caen en folds distintos y los folds quedan de igual
    tamaño cuando es posible. Devuelve filas con las columnas de FOLD_COLUMNS, ordenadas por image_id.
    """
    rng = random.Random(seed)
    by_label = {}
    for row in sorted(rows, key=lambda r: int(r["image_id"])):
        by_label.setdefault(int(row["binary_label"]), []).append(row)

    fold_of = {}
    turn = 0
    for label in sorted(by_label, reverse=True):  # clase positiva (1) primero
        group = list(by_label[label])
        rng.shuffle(group)
        for row in group:
            fold_of[row["image_id"]] = turn % n_folds
            turn += 1

    out = []
    for row in sorted(rows, key=lambda r: int(r["image_id"])):
        out.append({
            "image_id": row["image_id"],
            "image_name": row["image_name"],
            "source_split": row["source_split"],
            "original_class": row["original_class"],
            "binary_label": row["binary_label"],
            "binary_class": row["binary_class"],
            "e1_split": row["experimental_split"],
            "fold": fold_of[row["image_id"]],
        })
    return out


def verify_folds(fold_rows, manifest_rows, n_folds=N_FOLDS, expected_n=EXPECTED_N_IMAGES, verbose=True):
    """
    Verifica la partición E2 contra el manifiesto operativo y, si verbose, imprime el resumen.
    Lanza ValueError ante cualquier incumplimiento. Devuelve {fold: (n, positivos)}.
    """
    ids = [str(r["image_id"]) for r in fold_rows]
    manifest_by_id = {r["image_id"]: r for r in manifest_rows}
    official_ids = {r["image_id"] for r in manifest_rows if r["experimental_split"] in OFFICIAL_TRAIN_SPLITS}
    test_ids = {r["image_id"] for r in manifest_rows if r["experimental_split"] == TEST_SPLIT}

    repeated = len(ids) - len(set(ids))
    in_test = len(set(ids) & test_ids)
    if verbose:
        print(f"Imágenes en la partición E2: {len(ids)} (esperadas: {expected_n})")
        print(f"Imágenes repetidas: {repeated}")
        print(f"Imágenes de test incluidas: {in_test}")

    if len(ids) != expected_n:
        raise ValueError(f"Se esperaban {expected_n} imágenes y hay {len(ids)}.")
    if repeated:
        raise ValueError(f"Hay {repeated} imágenes repetidas.")
    if in_test:
        raise ValueError(f"Hay {in_test} imágenes de test en la partición E2.")
    if set(ids) != official_ids:
        raise ValueError("Las imágenes de E2 no coinciden con el train oficial (train + validation de E1).")

    for row in fold_rows:
        ref = manifest_by_id[str(row["image_id"])]
        if str(row["binary_label"]) != ref["binary_label"] or row["e1_split"] != ref["experimental_split"]:
            raise ValueError(f"image_id {row['image_id']}: etiqueta o e1_split distintos del manifiesto operativo.")
        if int(row["fold"]) not in range(n_folds):
            raise ValueError(f"image_id {row['image_id']}: fold fuera de rango ({row['fold']}).")

    stats = {}
    for fold in range(n_folds):
        members = [r for r in fold_rows if int(r["fold"]) == fold]
        stats[fold] = (len(members), sum(int(r["binary_label"]) for r in members))
        if stats[fold][0] == 0 or stats[fold][1] in (0, stats[fold][0]):
            raise ValueError(f"fold {fold}: no contiene ambas clases.")

    sizes = [n for n, _ in stats.values()]
    positives = [p for _, p in stats.values()]
    if max(sizes) - min(sizes) > 1 or max(positives) - min(positives) > 1:
        raise ValueError(f"Partición no balanceada: tamaños {sizes}, positivos {positives}.")

    if verbose:
        print(f"\n{'fold':>4} {'n':>5} {'positivos':>10} {'negativos':>10} {'% positivos':>12}")
        for fold, (n, pos) in stats.items():
            print(f"{fold:>4} {n:>5} {pos:>10} {n - pos:>10} {100 * pos / n:>11.1f}%")
        total_pos = sum(positives)
        print(f"{'total':>4} {sum(sizes):>5} {total_pos:>10} {sum(sizes) - total_pos:>10} "
              f"{100 * total_pos / sum(sizes):>11.1f}%")
    return stats


def derive_fold_rows(fold_rows, fold, n_folds=N_FOLDS):
    """
    Manifiesto de una corrida: las imágenes del fold `fold` son `validation`; el resto, `train`.
    Conserva e1_split (carpeta física de la imagen) y fold.
    """
    if fold not in range(n_folds):
        raise ValueError(f"fold debe estar entre 0 y {n_folds - 1}; se recibió {fold}.")
    derived = []
    for row in fold_rows:
        derived_row = dict(row)
        derived_row["experimental_split"] = "validation" if int(row["fold"]) == fold else "train"
        derived.append(derived_row)
    return derived


def check_train_validation_disjoint(train_ids, validation_ids, expected_n=EXPECTED_N_IMAGES):
    """
    train y validation de una corrida E2: sin imágenes repetidas, sin solapamiento y que suman las 1000.
    """
    train_ids, validation_ids = [str(i) for i in train_ids], [str(i) for i in validation_ids]
    if len(set(train_ids)) != len(train_ids) or len(set(validation_ids)) != len(validation_ids):
        raise ValueError("Hay imágenes repetidas dentro de train o de validation.")
    overlap = set(train_ids) & set(validation_ids)
    if overlap:
        raise ValueError(f"{len(overlap)} imágenes están en train y en validation a la vez.")
    if len(train_ids) + len(validation_ids) != expected_n:
        raise ValueError(f"train + validation = {len(train_ids) + len(validation_ids)}, se esperaban {expected_n}.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Genera o verifica la partición E2 (5 folds estratificados).")
    parser.add_argument("--verify", action="store_true",
                        help="Verifica el CSV existente y comprueba que coincide con la regeneración en memoria.")
    parser.add_argument("--manifest", default=MANIFEST_REL, help="Manifiesto operativo de entrada.")
    parser.add_argument("--output", default=OUTPUT_REL, help="CSV de la partición E2.")
    args = parser.parse_args(argv)

    root = Path.cwd()
    manifest_rows = read_csv_rows(root / args.manifest)
    official_rows = select_official_train(manifest_rows)
    regenerated = assign_folds(official_rows)
    output_path = root / args.output

    print(f"Semilla E2: {SEED_E2} | folds: {N_FOLDS} | manifiesto: {args.manifest}")
    if args.verify:
        if not output_path.exists():
            print(f"No existe {args.output}.")
            return 1
        committed = read_csv_rows(output_path)
        verify_folds(committed, manifest_rows)
        key = lambda rows: [(r["image_id"], str(r["fold"])) for r in rows]
        same = key(committed) == key(regenerated)
        print(f"\nCoincide con la regeneración (semilla {SEED_E2}): {'sí' if same else 'NO'}")
        print(f"SHA-256 de {args.output}: {sha256_of(output_path)}")
        return 0 if same else 1

    if output_path.exists():
        print(f"{args.output} ya existe: la partición E2 está congelada y no se sobrescribe.")
        return 1
    verify_folds(regenerated, manifest_rows)
    write_csv_rows(output_path, regenerated, FOLD_COLUMNS)
    print(f"\nEscrito: {args.output}")
    print(f"SHA-256: {sha256_of(output_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
