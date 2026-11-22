"""G1 / R2 v2: agrupamiento AUTOMÁTICO por similitud visual (pHash <= 8) y asignación determinista de 5 folds.

Protocolo: src/r2_v2/protocolo_agrupamiento_phash_v1.md (identificador phash_le8_v1). Aprobación técnica PROVISIONAL del
autor (D-025/D-026); conformidad del asesor académico PENDIENTE. Es una PROPUESTA de folds, no una partición congelada.

Qué es y qué NO es:
  - Medida de mitigación de fuga de información entre folds por imágenes visualmente casi idénticas.
  - NO demuestra identidad ni independencia por paciente (no hay IDs de paciente).
  - El umbral <= 8 es EXPLORATORIO: se eligió después de observar datos de desarrollo y usó sus etiquetas.
  - NO hay validación humana de los grupos: parejas con distancia mayor que 8 pueden quedar en folds distintos.

Solo lee artefactos de la auditoría (CSV y JSON), verificados por SHA-256 contra su resumen. No abre imágenes ni test,
no lee ni modifica revision_ciega.csv, no escribe en el directorio de la auditoría y no INVOCA scikit-learn (importar
make_folds_v2 requiere que scikit-learn esté instalado: es una dependencia de importación, no de cálculo).

Raíz de confianza: el SHA-256 de resumen_auditoria.json se ancla con --resumen-sha256 (el valor registrado en la
trazabilidad, D-024); sin él, una falsificación coherente de resumen y artefactos no se detectaría. image_hashes.csv no
es verificable sin imágenes: solo se comprueba su coherencia con los demás artefactos.

Se detiene, sin publicar nada, si: falta o difiere un artefacto de la auditoría; las distancias recalculadas no
reproducen candidate_pairs.csv ni calibration_pairs.csv; aparece un grupo con etiquetas mixtas (no se divide ni se
corrige); algún fold queda con menos de 4 positivas; falta --aprobacion; o el código/protocolo no están en el commit.

Uso (desde la raíz del repositorio, en Falcon, con el código y el protocolo ya en un commit):
  python -m src.r2_v2.make_folds_phash_v1 --aprobacion "D-026 2026-11-21" --resumen-sha256 <SHA-256 del resumen>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import common as C
from . import make_folds_v2 as mf  # reutiliza verify/build_groups/exact_duplicate_edges; no invoca scikit-learn (solo se importa)

MODULE = "src.r2_v2.make_folds_phash_v1"
VERSION = "phash_le8_v1"
PHASH_THRESHOLD = 8          # EXPLORATORIO, post hoc, no validado. Fijo: no se expone por línea de comandos.
N_FOLDS = mf.N_FOLDS         # 5 (propuesto)
DEFAULT_AUDIT_REL = mf.DEFAULT_AUDIT_REL
DEFAULT_OUT_REL = "configs/splits/v2_auto"
ALLOWED_OUT_PREFIX = "configs/splits/v2_auto"
PROTOCOL_REL = "src/r2_v2/protocolo_agrupamiento_phash_v1.md"
FOLDS_FILE = "cv_folds_phash_le8_v1.csv"
EDGES_FILE = "aristas_phash_le8_v1.csv"
SUMMARY_FILE = "resumen_phash_le8_v1.json"
# Artefactos de la auditoría que se leen y cuyo hash debe estar declarado y coincidir. revision_ciega.csv NO se lee.
INPUT_FILES = ("development_eligible_v2.csv", "image_hashes.csv", "exact_duplicates.csv",
               "candidate_pairs.csv", "calibration_pairs.csv")
FOLD_EXTRA_COLUMNS = ["group_id", "group_size", "fold"]
EDGE_COLUMNS = ["image_id_a", "image_id_b", "distance", "origen"]

LIMITATIONS = [
    "El agrupamiento por similitud visual NO equivale a separación por paciente: no hay identificadores de paciente; "
    "imagenes de la misma paciente con pHash distinto pueden quedar en folds distintos.",
    "El umbral pHash <= 8 es exploratorio: se eligio despues de observar datos de desarrollo y su criterio uso las "
    "etiquetas de desarrollo; no esta validado externamente.",
    "No hay validacion humana de los grupos ni se ha medido la precision de pHash <= 8 como detector de casi duplicados.",
    "Riesgo residual: parejas con distancia pHash mayor que 8 (p. ej. 9-12 y 13-18) pueden quedar en folds distintos; "
    "las estimaciones de rendimiento pueden ser optimistas en una magnitud desconocida.",
    "Con 38 positivas y 5 folds hay unas 7-8 por fold: incertidumbre alta por fold. Los limites numericos por fold son "
    "reglas operativas propuestas, no prueba de validez estadistica.",
    "pHash es una variante propia (mediana sin el coeficiente DC; Zauner 2010 [CITA POR VERIFICAR]).",
]


# ---------------------------------------------------------------- cálculo puro
def parse_phash(hex_value, image_id):
    if len(hex_value) != 16:
        raise C.GuardError(f"image_id {image_id}: phash_hex de longitud inválida ({len(hex_value)})")
    try:
        return int(hex_value, 16)
    except ValueError:
        raise C.GuardError(f"image_id {image_id}: phash_hex no hexadecimal") from None


def close_pairs(ids, phash_of, max_distance):
    """(a, b, d) para cada pareja con distancia de Hamming <= max_distance; a precede a b en `ids`."""
    out = []
    for i in range(len(ids)):
        hi = phash_of[ids[i]]
        for j in range(i + 1, len(ids)):
            d = (hi ^ phash_of[ids[j]]).bit_count()
            if d <= max_distance:
                out.append((ids[i], ids[j], d))
    return out


def pair_key(a, b):
    return tuple(sorted((a, b), key=int))


def cross_check_with_audit(recomputed, cand_rows, calib_rows, cutoff, phash_of):
    """Reproducibilidad: las distancias recalculadas deben reproducir EXACTAMENTE los CSV de la auditoría."""
    for name, rows in (("candidate_pairs.csv", cand_rows), ("calibration_pairs.csv", calib_rows)):
        for r in rows:
            if r["image_id_a"] not in phash_of or r["image_id_b"] not in phash_of:
                raise C.GuardError(f"{name}: pareja {r['image_id_a']}-{r['image_id_b']} con imagen no elegible")
    expected = {pair_key(a, b): d for a, b, d in recomputed}                    # todas las parejas con d <= corte
    got = {pair_key(r["image_id_a"], r["image_id_b"]): int(r["distance"]) for r in cand_rows}
    missing = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))
    differ = sorted(k for k in set(expected) & set(got) if expected[k] != got[k])
    if missing or extra or differ:
        raise C.GuardError(
            "Las distancias recalculadas desde image_hashes.csv NO reproducen candidate_pairs.csv: "
            f"{len(missing)} parejas ausentes, {len(extra)} sobrantes, {len(differ)} con distancia distinta "
            f"(p. ej. {(missing + extra + differ)[:3]}). Explica la discrepancia antes de continuar.")
    for r in calib_rows:
        a, b = r["image_id_a"], r["image_id_b"]
        d = (phash_of[a] ^ phash_of[b]).bit_count()
        if d != int(r["distance"]) or d <= cutoff:
            raise C.GuardError(f"calibration_pairs.csv: pareja {a}-{b} no reproducible (distancia recalculada {d}, "
                               f"registrada {r['distance']}, corte {cutoff}).")


def find_mixed_groups(ids, labels, groups):
    members = {}
    for i, y in zip(ids, labels):
        members.setdefault(groups[i], []).append((i, y))
    return [(g, ms) for g, ms in sorted(members.items(), key=lambda kv: int(kv[0]))
            if len({y for _, y in ms}) > 1]


def assign_folds_deterministic(ids, labels, groups, n_folds=N_FOLDS):
    """Asigna cada grupo a un fold de forma determinista (sin aleatoriedad; no invoca scikit-learn).

    Grupos con positivas: orden (positivas desc, tamaño desc, menor image_id asc) -> fold con menos positivas
    (desempate: menos imágenes, menor índice). Grupos sin positivas: orden (tamaño desc, menor image_id asc) -> fold con
    menos imágenes (desempate: menor índice). El resultado no depende del orden de entrada ni de ninguna versión de
    biblioteca. Nunca separa un grupo.
    """
    members = {}
    for i, y in zip(ids, labels):
        entry = members.setdefault(groups[i], [0, []])
        entry[0] += y
        entry[1].append(i)

    def min_id(ms):
        return min(int(i) for i in ms)

    with_pos = sorted((kv for kv in members.items() if kv[1][0] > 0),
                      key=lambda kv: (-kv[1][0], -len(kv[1][1]), min_id(kv[1][1])))
    no_pos = sorted((kv for kv in members.items() if kv[1][0] == 0),
                    key=lambda kv: (-len(kv[1][1]), min_id(kv[1][1])))
    pos, size, fold_of = [0] * n_folds, [0] * n_folds, {}
    for _, (p, ms) in with_pos:
        k = min(range(n_folds), key=lambda f: (pos[f], size[f], f))
        pos[k] += p
        size[k] += len(ms)
        fold_of.update({i: k for i in ms})
    for _, (_, ms) in no_pos:
        k = min(range(n_folds), key=lambda f: (size[f], f))
        size[k] += len(ms)
        fold_of.update({i: k for i in ms})
    return [fold_of[i] for i in ids]


def edges_split_across_folds(edges, fold_of):
    return [(a, b) for a, b in edges if fold_of[a] != fold_of[b]]


def residual_risk_counts(pairs_between, fold_of):
    """Transparencia (NO se usa para ajustar nada): cuántas parejas por encima del umbral quedan en folds distintos."""
    total = len(pairs_between)
    split = sum(1 for a, b in pairs_between if fold_of[a] != fold_of[b])
    return {"parejas": total, "en_folds_distintos": split}


# ---------------------------------------------------------------- entradas
def load_audit_inputs(audit_dir, expected, root, allow_uncommitted, resumen_sha256):
    audit_dir = Path(audit_dir)
    needed = [*INPUT_FILES, "resumen_auditoria.json"]
    missing = [n for n in needed if not (audit_dir / n).is_file()]
    if missing:
        raise C.GuardError(f"Faltan archivos de la auditoría en {audit_dir}: {missing}")
    if C.sha256_file(audit_dir / "resumen_auditoria.json") != resumen_sha256:
        raise C.GuardError("El SHA-256 de resumen_auditoria.json no coincide con el anclado (--resumen-sha256): la "
                           "auditoría no es la registrada en la trazabilidad.")
    summary = json.loads((audit_dir / "resumen_auditoria.json").read_text(encoding="utf-8"))
    if summary.get("recuentos") != expected:
        raise C.GuardError(f"Recuentos de la auditoría {summary.get('recuentos')} != esperado {expected}")
    declared = summary.get("salidas_sha256", {})
    undeclared = [n for n in INPUT_FILES if n not in declared]
    if undeclared:
        raise C.GuardError(f"El resumen de la auditoría no declara el hash de: {undeclared}")
    for name in INPUT_FILES:
        if C.sha256_file(audit_dir / name) != declared[name]:
            raise C.GuardError(f"{name} fue alterado respecto de la auditoría (hash distinto)")
    source = Path(root) / C.SOURCE_REL
    if not source.is_file() or C.sha256_file(source) != summary.get("fuente", {}).get("sha256"):
        raise C.GuardError(f"{C.SOURCE_REL} difiere de la fuente que se auditó (o no existe)")
    prov = summary.get("procedencia", {})
    code = prov.get("archivos_codigo") or {}
    if not allow_uncommitted:
        unversioned = [k for k, v in code.items() if not v.get("en_commit")]
        if not code or unversioned or not prov.get("fuente_en_commit"):
            raise C.GuardError("La auditoría se ejecutó con código o fuente sin commit (según su propia procedencia "
                               f"{unversioned or 'vacía'}): no es evidencia válida de G1.")
    cutoff = summary.get("phash", {}).get("corte_cribado")
    if not isinstance(cutoff, int) or PHASH_THRESHOLD > cutoff:
        raise C.GuardError(f"El corte de cribado de la auditoría ({cutoff}) no cubre el umbral {PHASH_THRESHOLD}.")

    eligible = C.read_csv_rows(audit_dir / "development_eligible_v2.csv")
    for r in eligible:  # las etiquetas se recomputan SOLO desde original_class
        if C.label_from_class(int(r["original_class"])) != int(r["label_v2"]):
            raise C.GuardError(f"image_id {r['image_id']}: label_v2 no coincide con el mapeo desde original_class")
    ids = [r["image_id"] for r in eligible]
    if len(set(ids)) != len(ids):
        raise C.GuardError("image_id repetidos en development_eligible_v2.csv")
    n_pos = sum(1 for r in eligible if int(r["label_v2"]) == 1)
    got = {"eligible": len(eligible), "non_malignant": len(eligible) - n_pos, "malignant": n_pos}
    wanted = {k: expected[k] for k in got}
    if got != wanted:
        raise C.GuardError(f"Recuentos recalculados desde development_eligible_v2.csv {got} != esperado {wanted}")
    # La fuente de desarrollo (solo CSV, sin imágenes) debe reproducir los elegibles y los normales.
    src_rows = C.load_development_source(source, expected["eligible"] + expected["normal"])
    src_elig = {r["image_id"]: r["original_class"] for r in src_rows if r["original_class"] not in C.EXCLUDED_CLASSES}
    if (src_elig != {r["image_id"]: int(r["original_class"]) for r in eligible}
            or len(src_rows) - len(src_elig) != expected["normal"]):
        raise C.GuardError(f"development_eligible_v2.csv no coincide con la fuente {C.SOURCE_REL} "
                           "(identificadores o categorías distintos).")
    hash_rows = {r["image_id"]: r for r in C.read_csv_rows(audit_dir / "image_hashes.csv")}
    phash_of = {}
    for r in eligible:
        h = hash_rows.get(r["image_id"])
        if h is None:
            raise C.GuardError(f"image_id {r['image_id']} sin fila en image_hashes.csv")
        if int(h["original_class"]) != int(r["original_class"]):
            raise C.GuardError(f"image_id {r['image_id']}: categoría distinta entre image_hashes.csv y los elegibles")
        phash_of[r["image_id"]] = parse_phash(h["phash_hex"], r["image_id"])
    return {"summary": summary, "eligible": eligible, "phash_of": phash_of, "cutoff": cutoff,
            "dup_rows": C.read_csv_rows(audit_dir / "exact_duplicates.csv"),
            "cand_rows": C.read_csv_rows(audit_dir / "candidate_pairs.csv"),
            "calib_rows": C.read_csv_rows(audit_dir / "calibration_pairs.csv")}


# ---------------------------------------------------------------- validación de la salida
def validate_staging(staging, eligible, expected):
    """Relee los archivos preparados y repite las comprobaciones críticas ANTES de publicar."""
    staging = Path(staging)
    folds_header, rows = C.read_csv_with_header(staging / FOLDS_FILE)
    ids = [r["image_id"] for r in eligible]
    if [r["image_id"] for r in rows] != ids:
        raise C.GuardError(f"{FOLDS_FILE} no contiene exactamente las lesiones elegibles")
    if any(c in folds_header for c in ("distance", "phash_hex", "sha256")):
        raise C.GuardError(f"{FOLDS_FILE} no debe llevar distancias ni hashes")
    labels = [C.label_from_class(int(r["original_class"])) for r in rows]
    if [int(r["label_v2"]) for r in rows] != labels:
        raise C.GuardError(f"{FOLDS_FILE}: label_v2 no coincide con el mapeo desde original_class")
    fold_of = {r["image_id"]: int(r["fold"]) for r in rows}
    if any(f not in range(N_FOLDS) for f in fold_of.values()):
        raise C.GuardError(f"{FOLDS_FILE}: fold fuera de rango")
    edge_rows = C.read_csv_rows(staging / EDGES_FILE)
    edges = [(r["image_id_a"], r["image_id_b"]) for r in edge_rows]
    regroup = mf.build_groups(ids, edges)
    if {r["image_id"]: r["group_id"] for r in rows} != {i: str(regroup[i]) for i in ids}:
        raise C.GuardError(f"{FOLDS_FILE}: los grupos no corresponden a las aristas")
    if edges_split_across_folds(edges, fold_of):
        raise C.GuardError("Hay aristas con sus extremos en folds distintos")
    _, problems, _ = mf.verify(ids, labels, regroup, [fold_of[i] for i in ids])
    if problems:
        raise C.GuardError("Validación de los folds preparados: " + "; ".join(problems))
    if C.path_mentions_test(FOLDS_FILE) or any(C.path_mentions_test(r["image_path"]) for r in rows):
        raise C.GuardError("Ruta de test en la salida")
    summary = json.loads((staging / SUMMARY_FILE).read_text(encoding="utf-8"))
    for name, digest in summary["salidas_sha256"].items():
        if C.sha256_file(staging / name) != digest:
            raise C.GuardError(f"El hash de {name} no coincide con el resumen")
    return summary


# ---------------------------------------------------------------- ejecución
def main(argv=None, project_root=C.PROJECT_ROOT, expected=None):
    """`expected` existe solo para pruebas sintéticas; la CLI usa los recuentos de G1."""
    expected = C.EXPECTED if expected is None else expected
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--aprobacion", default="", help='Referencia a la aprobación provisional, p. ej. "D-026 2026-11-21".')
    ap.add_argument("--resumen-sha256", default="", help="SHA-256 de resumen_auditoria.json registrado en la "
                    "trazabilidad (D-024): ancla la raíz de confianza. Obligatorio.")
    ap.add_argument("--force", action="store_true", help="Reemplaza una propuesta previa (se conserva renombrada).")
    ap.add_argument("--permitir-codigo-sin-commit", action="store_true",
                    help="Solo para pruebas: el resultado no sirve como evidencia de G1.")
    args = ap.parse_args(argv)

    root = Path(project_root).resolve()

    def resolve(p, default):
        p = default if p is None else p
        return (p if p.is_absolute() else root / p).resolve()

    audit_dir = resolve(args.audit_dir, Path(DEFAULT_AUDIT_REL))
    out = resolve(args.out, Path(DEFAULT_OUT_REL))
    staging = None
    try:
        if not args.aprobacion.strip():
            raise C.GuardError("Falta --aprobacion: se requiere una referencia a la aprobación registrada.")
        pinned = args.resumen_sha256.strip().lower()
        if len(pinned) != 64 or any(c not in "0123456789abcdef" for c in pinned):
            raise C.GuardError("Falta --resumen-sha256 (64 caracteres hexadecimales): anota en el comando el SHA-256 de "
                               "resumen_auditoria.json registrado en la trazabilidad (D-024).")
        C.require_safe_out(out, root, ALLOWED_OUT_PREFIX, allow_equal=True, forbidden_paths=[audit_dir])
        C.prepare_output(out, (), args.force)
        prov = C.provenance(MODULE, argv)
        versioned = C.require_versioned_code(prov, args.permitir_codigo_sin_commit)
        protocol_path = C.PROJECT_ROOT / PROTOCOL_REL
        if not protocol_path.is_file():
            raise C.GuardError(f"Falta el protocolo versionado {PROTOCOL_REL}.")
        protocol = C.code_state(C.PROJECT_ROOT, [PROTOCOL_REL])
        if not protocol["en_commit"] and not args.permitir_codigo_sin_commit:
            raise C.GuardError(f"El protocolo {PROTOCOL_REL} no está en el commit o difiere de él: versiónalo antes "
                               "de la primera corrida.")

        data = load_audit_inputs(audit_dir, expected, root, args.permitir_codigo_sin_commit, pinned)
        eligible, phash_of, cutoff = data["eligible"], data["phash_of"], data["cutoff"]
        ids = [r["image_id"] for r in eligible]
        labels = [int(r["label_v2"]) for r in eligible]
        label_of = dict(zip(ids, labels))

        recomputed = close_pairs(ids, phash_of, cutoff)
        cross_check_with_audit(recomputed, data["cand_rows"], data["calib_rows"], cutoff, phash_of)

        close = [(a, b, d) for a, b, d in recomputed if d <= PHASH_THRESHOLD]
        dup_edges = mf.exact_duplicate_edges(data["dup_rows"], label_of)   # aborta si un duplicado une etiquetas distintas
        phash_edges = [(a, b) for a, b, _ in close]
        covered = {pair_key(a, b) for a, b in phash_edges}
        edge_rows = [{"image_id_a": a, "image_id_b": b, "distance": d, "origen": f"phash<={PHASH_THRESHOLD}"}
                     for a, b, d in close]
        edge_rows += [{"image_id_a": a, "image_id_b": b, "distance": 0, "origen": "sha256_identico"}
                      for a, b in dup_edges if pair_key(a, b) not in covered]
        all_edges = phash_edges + dup_edges
        groups = mf.build_groups(ids, all_edges)

        mixed = find_mixed_groups(ids, labels, groups)
        if mixed:
            detail = "; ".join(f"grupo {g}: imágenes {[i for i, _ in ms]} con etiquetas {[y for _, y in ms]}"
                               for g, ms in mixed[:10])
            raise C.GuardError(f"{len(mixed)} grupo(s) con etiquetas mixtas: {detail}. No se divide ningún grupo ni se "
                               "corrige ninguna etiqueta: reporta el caso y decide por escrito antes de continuar.")

        folds = assign_folds_deterministic(ids, labels, groups)
        per_fold, problems, warnings = mf.verify(ids, labels, groups, folds)
        fold_of = dict(zip(ids, folds))
        if edges_split_across_folds(all_edges, fold_of):
            problems.append("hay aristas con sus extremos en folds distintos")
        if problems:
            raise C.GuardError("Partición no viable:\n  - " + "\n  - ".join(problems) +
                               "\nDecide por escrito antes de continuar; no se publicó ningún archivo.")
        for w in warnings:
            print("ADVERTENCIA:", w)

        sizes = {}
        for g in groups.values():
            sizes[g] = sizes.get(g, 0) + 1
        histogram = {}
        for s in sizes.values():
            histogram[s] = histogram.get(s, 0) + 1
        between = [(a, b) for a, b, d in recomputed if PHASH_THRESHOLD < d <= cutoff]

        staging = C.new_staging(out)
        cols = list(eligible[0].keys()) + FOLD_EXTRA_COLUMNS
        out_rows = [{**r, "group_id": groups[r["image_id"]], "group_size": sizes[groups[r["image_id"]]],
                     "fold": fold_of[r["image_id"]]} for r in eligible]
        (staging / FOLDS_FILE).write_text(C.csv_text(out_rows, cols), encoding="utf-8", newline="")
        (staging / EDGES_FILE).write_text(C.csv_text(edge_rows, EDGE_COLUMNS), encoding="utf-8", newline="")

        audit_summary = data["summary"]
        summary = {
            "etapa": "G1 / R2 v2 - agrupamiento automatico por pHash y folds (propuesta provisional)",
            "version_protocolo": VERSION,
            "estado": ("PROPUESTA PROVISIONAL - aprobacion tecnica provisional del autor; conformidad del asesor "
                       "PENDIENTE; no congelada" if (versioned and protocol["en_commit"]) else
                       "NO VALIDA COMO EVIDENCIA DE G1: codigo, fuente o protocolo sin commit (ejecucion de "
                       "prueba)."),
            "protocolo": {"archivo": PROTOCOL_REL, "sha256": C.sha256_file(protocol_path),
                          "blob": protocol["archivos"][PROTOCOL_REL]["blob"], "en_commit": protocol["en_commit"]},
            "criterio": {"tipo": "componentes conexas de parejas con distancia pHash <= umbral, mas duplicados exactos",
                         "umbral_phash": PHASH_THRESHOLD, "exploratorio": True, "post_hoc": True,
                         "uso_de_etiquetas_en_la_seleccion": True, "validado_externamente": False,
                         "equivale_a_separacion_por_paciente": False},
            "validacion_humana": {"realizada": False,
                                  "nota": "Sin revision humana de los grupos ni de las parejas; revision_ciega.csv no se "
                                          "lee ni se modifica."},
            "limitaciones": LIMITATIONS,
            "aprobacion_parametros": args.aprobacion.strip(),
            "n_folds": N_FOLDS, "asignador": "determinista, sin aleatoriedad; no invoca scikit-learn",
            "criterios_propuestos": {"min_positivas_por_fold_error": mf.MIN_POS_HARD,
                                     "rango_positivas_advertencia": list(mf.WARN_POS_RANGE),
                                     "desviacion_tamano_advertencia": mf.MAX_SIZE_DEVIATION},
            "recuentos": audit_summary["recuentos"],
            "agrupamiento_observado": {   # INFORMATIVO: no son criterios de aceptación
                "aristas_phash": len(phash_edges), "aristas_sha256_adicionales": len(edge_rows) - len(phash_edges),
                "grupos_multiimagen": sum(1 for s in sizes.values() if s > 1),
                "imagenes_en_grupos_multiimagen": sum(s for s in sizes.values() if s > 1),
                "tamano_maximo_de_grupo": max(sizes.values()) if sizes else 0,
                "histograma_tamanos_de_grupo": {str(k): histogram[k] for k in sorted(histogram)},
                "grupos_mixtos": 0},
            "folds": per_fold, "advertencias": warnings,
            "diagnostico_riesgo_residual": {   # transparencia; NO se usa para ajustar nada
                "parejas_entre_umbral_y_corte_de_cribado": residual_risk_counts(between, fold_of),
                "parejas_de_calibracion_fuera_del_corte": residual_risk_counts(
                    [(r["image_id_a"], r["image_id_b"]) for r in data["calib_rows"]], fold_of),
                "umbral": PHASH_THRESHOLD, "corte_de_cribado_de_la_auditoria": cutoff,
                "nota": "Las parejas de calibracion son una MUESTRA (hasta 60), no la poblacion de parejas con "
                        "distancia 13-18; las parejas entre el umbral y el corte son todas las del cribado."},
            "entradas_sha256": {n: C.sha256_file(audit_dir / n) for n in (*INPUT_FILES, "resumen_auditoria.json")},
            "resumen_auditoria_sha256_anclado": pinned,
            "auditoria_commit": audit_summary["procedencia"]["commit"],
            "procedencia": prov, "codigo_en_commit": prov["codigo_en_commit"],
            "salidas_sha256": {n: C.sha256_file(staging / n) for n in (FOLDS_FILE, EDGES_FILE)},
            "nota": "La agrupacion por similitud visual NO equivale a separacion por paciente. El umbral es exploratorio "
                    "y post hoc. No hay validacion humana. Hay riesgo residual de similitudes no agrupadas.",
        }
        C.write_json(staging / SUMMARY_FILE, summary)
        validate_staging(staging, eligible, expected)

        C.prepare_output(out, (), args.force)  # el destino pudo cambiar durante el cálculo
        backup = C.publish_staging(staging, out)
        staging = None
        print(json.dumps({k: summary[k] for k in ("estado", "version_protocolo", "agrupamiento_observado", "folds")},
                         indent=2, ensure_ascii=False))
        if backup:
            print(f"Propuesta anterior conservada en: {backup}")
        return 0
    except C.GuardError as e:
        print(f"ERROR (no se publicó ninguna salida): {e}", file=sys.stderr)
        return 1
    finally:
        if staging is not None:
            C.discard_staging(staging)


if __name__ == "__main__":
    sys.exit(main())
