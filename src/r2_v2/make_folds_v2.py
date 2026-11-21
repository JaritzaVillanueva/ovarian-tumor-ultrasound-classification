"""G1 / R2 v2: PROPUESTA de folds de validación cruzada sobre las lesiones elegibles de desarrollo.

Solo se ejecuta cuando la revisión humana está COMPLETA. Se detiene, sin escribir nada, si:
  - falta algún archivo de la auditoría, la columna `revision`, o el resumen de la auditoría;
  - algún par queda sin revisar o en `duda`, o un par revisado no tiene `revisor`;
  - algún artefacto de la auditoría, incluida la CORRESPONDENCIA hoja -> pareja (candidate_pairs.csv y
    calibration_pairs.csv), fue alterado (hash exacto), o la planilla ciega revision_ciega.csv no corresponde
    uno a uno con ella (hojas añadidas, quitadas, repetidas o renombradas);
  - un mismo par tiene revisiones distintas en candidate_pairs.csv y calibration_pairs.csv;
  - un duplicado exacto o un par `misma` une imágenes con etiquetas v2 distintas;
  - falta --aprobacion (referencia a la aprobación de los parámetros propuestos);
  - el código o la fuente no están en el commit registrado (salvo --permitir-codigo-sin-commit, solo pruebas).

Grupos: los duplicados exactos (SHA-256 idéntico) se unen automáticamente; los pares `misma` se unen por
revisión humana; `distinta` no une nada. Nunca se separa un grupo para equilibrar clases.

Uso (desde la raíz del repositorio, en Falcon, tras la revisión y su commit):
  python -m src.r2_v2.make_folds_v2 --aprobacion "D-0XX 2026-11-21"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from . import common as C

MODULE = "src.r2_v2.make_folds_v2"
DEFAULT_AUDIT_REL = "results/r2_v2/auditoria_desarrollo"
DEFAULT_OUT_REL = "configs/splits/v2"
PAIR_FILES = ("candidate_pairs.csv", "calibration_pairs.csv")  # correspondencia hoja -> pareja (no se edita)
REVIEW_SHEET_FILE = "revision_ciega.csv"                       # planilla: hoja, revision, revisor, nota
AUDIT_FILES = ("development_eligible_v2.csv", "normal_development_controls.csv", "image_hashes.csv",
               "exact_duplicates.csv", *PAIR_FILES, REVIEW_SHEET_FILE, "resumen_auditoria.json")
# La planilla lleva la revisión humana: se comprueba su correspondencia con los pares, no su hash.
HASHED_AUDIT_FILES = tuple(n for n in AUDIT_FILES if n not in (REVIEW_SHEET_FILE, "resumen_auditoria.json"))
ALLOWED_OUT_PREFIX = "configs/splits/v2"  # --out puede ser este directorio o quedar dentro de él
KEY_REQUIRED_COLUMNS = ("image_id_a", "image_id_b", "distance", "label_a", "label_b", "hoja")
SHEET_REQUIRED_COLUMNS = ("hoja", "revision", "revisor", "nota")

N_FOLDS = 5
SEED = 42
VALID_REVIEW = {"misma", "distinta", "duda"}
# Valores PROPUESTOS (no validados estadísticamente; requieren aprobación registrada en la trazabilidad).
# Nunca se rompe un grupo para equilibrar: StratifiedGroupKFold respeta los grupos.
# Error (detiene): algún fold con menos de MIN_POS_HARD positivas.
# Advertencia (se reporta, no detiene): positivas fuera de [6, 9] o tamaño desviado > 15 % (820/5 = 164).
MIN_POS_HARD = 4
WARN_POS_RANGE = (6, 9)
MAX_SIZE_DEVIATION = 0.15


class UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb, key=int)] = min(ra, rb, key=int)


# ---------------------------------------------------------------- revisión humana
def confirmed_edges(pair_rows, valid_ids):
    """Aristas `misma` y recuento de revisiones. No bloquea: ver require_complete_review."""
    edges = []
    stats = {"revisados": 0, "misma": 0, "distinta": 0, "duda": 0, "sin_revisar": 0, "sin_revisor": 0}
    for r in pair_rows:
        rev = (r.get("revision") or "").strip().lower()
        if rev not in VALID_REVIEW | {""}:
            raise C.GuardError(f"Valor de revisión inválido: {rev!r} ({r['image_id_a']}-{r['image_id_b']})")
        if rev == "":
            stats["sin_revisar"] += 1
            continue
        stats["revisados"] += 1
        stats[rev] += 1
        if not (r.get("revisor") or "").strip():
            stats["sin_revisor"] += 1
        if rev == "misma":
            a, b = r["image_id_a"], r["image_id_b"]
            if a not in valid_ids or b not in valid_ids:
                raise C.GuardError(f"Par con imagen no elegible: {a}-{b}")
            edges.append((a, b))
    return edges, stats


def require_complete_review(stats):
    problems = []
    if stats["sin_revisar"]:
        problems.append(f"{stats['sin_revisar']} pares sin revisar")
    if stats["duda"]:
        problems.append(f"{stats['duda']} pares en 'duda' (resuélvelos a misma/distinta)")
    if stats["sin_revisor"]:
        problems.append(f"{stats['sin_revisor']} pares revisados sin `revisor`")
    if problems:
        raise C.GuardError("Revisión manual incompleta: " + "; ".join(problems))


def merge_pair_reviews(rows_by_file):
    """Une los pares de los dos CSV sin duplicar. Un mismo par con revisiones distintas aborta."""
    merged = {}
    for name, rows in rows_by_file.items():
        for r in rows:
            key = tuple(sorted((r["image_id_a"], r["image_id_b"]), key=int))
            rev = (r.get("revision") or "").strip().lower()
            if key in merged and merged[key]["revision"] != rev:
                raise C.GuardError(f"Par {key[0]}-{key[1]} con revisiones distintas "
                                   f"({merged[key]['revision']!r} en {merged[key]['origen']}, {rev!r} en {name})")
            merged.setdefault(key, {**r, "revision": rev, "origen": name})
    return list(merged.values())


def exact_duplicate_edges(dup_rows, label_of):
    """Aristas por SHA-256 idéntico entre elegibles. Aborta si el grupo une etiquetas v2 distintas."""
    edges = []
    for d in dup_rows:
        ids = [i for i in d["image_ids"].split(";") if i in label_of]
        if len(ids) < 2:
            continue  # los normales no entran en los folds
        if len({label_of[i] for i in ids}) > 1:
            raise C.GuardError(f"Duplicado exacto con etiquetas contradictorias: {d['image_ids']} "
                               f"(resuélvelo por escrito antes de generar folds)")
        edges += [(ids[0], i) for i in ids[1:]]
    return edges


def build_groups(ids, edges):
    uf = UnionFind(ids)
    for a, b in edges:
        uf.union(a, b)
    return {i: uf.find(i) for i in ids}


def assign_folds(ids, labels, groups, n_folds=N_FOLDS, seed=SEED):
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y = np.array(labels)
    g = np.array([groups[i] for i in ids])
    fold = np.full(len(ids), -1)
    for k, (_, val_idx) in enumerate(sgkf.split(np.zeros(len(ids)), y, g)):
        fold[val_idx] = k
    return fold.tolist()


def verify(ids, labels, groups, folds, n_folds=N_FOLDS):
    """Devuelve (por_fold, errores, advertencias)."""
    problems, warnings = [], []
    if len(set(ids)) != len(ids):
        problems.append("ids repetidos")
    if -1 in folds:
        problems.append("imágenes sin fold")
    group_folds = {}
    for i, f in zip(ids, folds):
        group_folds.setdefault(groups[i], set()).add(f)
    crossing = [g for g, fs in group_folds.items() if len(fs) > 1]
    if crossing:
        problems.append(f"{len(crossing)} grupos en más de un fold")
    target = len(ids) / n_folds  # 820 / 5 = 164
    per_fold = []
    for k in range(n_folds):
        n = sum(1 for f in folds if f == k)
        pos = sum(1 for f, y in zip(folds, labels) if f == k and y == 1)
        per_fold.append({"fold": k, "n": n, "positivas": pos})
        if pos < MIN_POS_HARD:
            problems.append(f"fold {k}: solo {pos} positivas (mínimo {MIN_POS_HARD})")
        elif not WARN_POS_RANGE[0] <= pos <= WARN_POS_RANGE[1]:
            warnings.append(f"fold {k}: {pos} positivas fuera de {list(WARN_POS_RANGE)}")
        if abs(n - target) / target > MAX_SIZE_DEVIATION:
            warnings.append(f"fold {k}: tamaño {n} se desvía más de {MAX_SIZE_DEVIATION:.0%} de {target:.0f}")
    return per_fold, problems, warnings


# ---------------------------------------------------------------- entradas de la auditoría
def load_audit(audit_dir, expected, root=C.PROJECT_ROOT, allow_uncommitted=False):
    """Lee la auditoría y comprueba que sus artefactos no fueron alterados (salvo las anotaciones de revisión).

    El resumen NO es una raíz de confianza por sí mismo: se exige que declare el hash de CADA artefacto
    esperado (no solo los que él decide listar), que la fuente actual coincida con la auditada y que la
    procedencia registrada muestre código y fuente versionados (no basta un `codigo_en_commit` autodeclarado).
    Devuelve (resumen, elegibles, pares por archivo, advertencias).
    """
    missing = [n for n in AUDIT_FILES if not (audit_dir / n).is_file()]
    if missing:
        raise C.GuardError(f"Faltan archivos de la auditoría en {audit_dir}: {missing}")
    summary = json.loads((audit_dir / "resumen_auditoria.json").read_text(encoding="utf-8"))
    if summary.get("recuentos") != expected:
        raise C.GuardError(f"Recuentos de la auditoría {summary.get('recuentos')} != esperado {expected}")
    hashed = summary.get("salidas_sha256", {})
    unhashed = [n for n in HASHED_AUDIT_FILES if n not in hashed]
    if unhashed:
        raise C.GuardError(f"El resumen de la auditoría no declara el hash de: {unhashed}")
    for name in HASHED_AUDIT_FILES:  # los CSV de pares llevan la revisión humana: se comprueba su conjunto de pares
        if C.sha256_file(audit_dir / name) != hashed[name]:
            raise C.GuardError(f"{name} fue alterado respecto de la auditoría (hash distinto)")
    source = root / C.SOURCE_REL
    if not source.is_file() or C.sha256_file(source) != summary.get("fuente", {}).get("sha256"):
        raise C.GuardError(f"{C.SOURCE_REL} difiere de la fuente que se auditó (o no existe)")
    prov = summary.get("procedencia", {})
    code = prov.get("archivos_codigo") or {}
    if not allow_uncommitted:
        unversioned = [k for k, v in code.items() if not v.get("en_commit")]
        if not code or unversioned or not prov.get("fuente_en_commit"):
            raise C.GuardError("La auditoría se ejecutó con código o fuente sin commit (según su propia procedencia "
                               f"{unversioned or 'vacía'}): no es evidencia válida de G1.")
    warnings = []
    current = {k: v for k, v in C.code_state(C.PROJECT_ROOT, list(code)).get("archivos", {}).items()}
    changed = sorted(k for k, v in code.items()
                     if k != "src/r2_v2/make_folds_v2.py" and v.get("blob") and current.get(k, {}).get("blob") != v["blob"])
    if changed:
        warnings.append("El código de la auditoría cambió desde que se ejecutó: " + ", ".join(changed))
    key_rows = {}  # correspondencia: ya verificada por hash exacto arriba
    for name in PAIR_FILES:
        header, rows = C.read_csv_with_header(audit_dir / name)
        absent = [c for c in KEY_REQUIRED_COLUMNS if c not in header]
        if absent:
            raise C.GuardError(f"{name}: faltan columnas {absent}")
        key_rows[name] = rows
    header, sheet = C.read_csv_with_header(audit_dir / REVIEW_SHEET_FILE)
    absent = [c for c in SHEET_REQUIRED_COLUMNS if c not in header]
    if absent:
        raise C.GuardError(f"{REVIEW_SHEET_FILE}: faltan columnas {absent}")
    hojas = [r["hoja"] for r in sheet]
    expected_hojas = [r["hoja"] for rows in key_rows.values() for r in rows]
    if (len(set(hojas)) != len(hojas) or sorted(hojas) != sorted(expected_hojas)
            or C.hojas_digest(hojas) != summary.get("planilla_ciega", {}).get("hojas_sha256")):
        raise C.GuardError(f"{REVIEW_SHEET_FILE} no corresponde uno a uno con la correspondencia auditada "
                           "(se añadieron, quitaron, repitieron o renombraron hojas)")
    by_hoja = {r["hoja"]: r for r in sheet}
    pairs = {name: [{**k, **{f: by_hoja[k["hoja"]][f] for f in ("revision", "revisor", "nota")}} for k in rows]
             for name, rows in key_rows.items()}
    eligible = C.read_csv_rows(audit_dir / "development_eligible_v2.csv")
    for r in eligible:  # las etiquetas se recomputan SOLO desde original_class
        if C.label_from_class(int(r["original_class"])) != int(r["label_v2"]):
            raise C.GuardError(f"image_id {r['image_id']}: label_v2 no coincide con el mapeo desde original_class")
    return summary, eligible, pairs, warnings


def validate_staging(staging, ids, labels, groups):
    """Re-lee los archivos preparados y repite las comprobaciones críticas antes de publicar."""
    out_csv = Path(staging) / "cv_folds_v2.csv"
    rows = C.read_csv_rows(out_csv)
    if [r["image_id"] for r in rows] != list(ids):
        raise C.GuardError("cv_folds_v2.csv no contiene exactamente las imágenes elegibles")
    folds = [int(r["fold"]) for r in rows]
    if [int(r["label_v2"]) for r in rows] != list(labels):
        raise C.GuardError("cv_folds_v2.csv: etiquetas alteradas")
    regroup = {r["image_id"]: r["group_id"] for r in rows}
    if regroup != {i: str(groups[i]) for i in ids}:
        raise C.GuardError("cv_folds_v2.csv: grupos alterados")
    _, problems, _ = verify(ids, labels, regroup, folds)
    if problems:
        raise C.GuardError("Validación de los folds preparados: " + "; ".join(problems))
    summary = json.loads((Path(staging) / "resumen_folds_v2.json").read_text(encoding="utf-8"))
    if summary["salida_sha256"] != C.sha256_file(out_csv):
        raise C.GuardError("El hash de cv_folds_v2.csv no coincide con el resumen")


def main(argv=None, project_root=C.PROJECT_ROOT, expected=None):
    """`expected` existe solo para pruebas sintéticas; la CLI usa los recuentos de G1."""
    expected = C.EXPECTED if expected is None else expected
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--aprobacion", default="", help='Referencia a la aprobación de los parámetros propuestos, '
                                                      'p. ej. "D-0XX 2026-11-21".')
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
            raise C.GuardError("Falta --aprobacion: los parámetros propuestos requieren una aprobación registrada.")
        C.require_safe_out(out, root, ALLOWED_OUT_PREFIX, allow_equal=True, forbidden_paths=[audit_dir])
        C.prepare_output(out, (), args.force)
        prov = C.provenance(MODULE, argv)
        versioned = C.require_versioned_code(prov, args.permitir_codigo_sin_commit)

        audit_summary, eligible, pairs_by_file, audit_warnings = load_audit(
            audit_dir, expected, root, args.permitir_codigo_sin_commit)
        ids = [r["image_id"] for r in eligible]
        labels = [int(r["label_v2"]) for r in eligible]
        label_of = dict(zip(ids, labels))

        merged = merge_pair_reviews(pairs_by_file)
        edges_pairs, stats = confirmed_edges(merged, set(ids))
        require_complete_review(stats)
        dup_rows = C.read_csv_rows(audit_dir / "exact_duplicates.csv")
        edges_dup = exact_duplicate_edges(dup_rows, label_of)
        for a, b in edges_pairs:
            if label_of[a] != label_of[b]:
                raise C.GuardError(f"Par `misma` {a}-{b} une etiquetas v2 distintas: resuélvelo por escrito")

        groups = build_groups(ids, edges_pairs + edges_dup)
        folds = assign_folds(ids, labels, groups)
        per_fold, problems, warnings = verify(ids, labels, groups, folds)
        if problems:
            raise C.GuardError("Partición no viable:\n  - " + "\n  - ".join(problems) +
                               "\nDecide por escrito (menos folds o limitación declarada) antes de entrenar.")
        warnings = audit_warnings + warnings
        for w in warnings:
            print("ADVERTENCIA:", w)
        conflict_dup = sum(1 for r in merged if r["revision"] == "distinta"
                           and groups.get(r["image_id_a"]) is not None
                           and groups[r["image_id_a"]] == groups.get(r["image_id_b"]))

        staging = C.new_staging(out)
        cols = ["image_id", "image_name", "original_class", "label_v2", "class_v2", "e1_split", "image_path",
                "group_id", "fold"]
        out_rows = [{**{k: r[k] for k in cols[:7]}, "group_id": groups[r["image_id"]], "fold": fo}
                    for r, fo in zip(eligible, folds)]
        (staging / "cv_folds_v2.csv").write_text(C.csv_text(out_rows, cols), encoding="utf-8", newline="")

        sizes = {}
        for g in groups.values():
            sizes[g] = sizes.get(g, 0) + 1
        multi = [s for s in sizes.values() if s > 1]
        cand_stats = confirmed_edges(pairs_by_file["candidate_pairs.csv"], set(ids))[1]
        summary = {
            "etapa": "G1 / R2 v2 - folds",
            "estado": ("PROPUESTA - no congelada hasta revision y visto bueno del autor y del asesor" if versioned
                       else "NO VALIDA COMO EVIDENCIA DE G1: codigo o fuente sin commit (ejecucion de prueba)."),
            "modo": "agrupado_por_similitud" if (edges_pairs or edges_dup) else "por_imagen",
            "n_folds": N_FOLDS, "semilla": SEED,
            "criterios_propuestos": {"min_positivas_por_fold_error": MIN_POS_HARD,
                                     "rango_positivas_advertencia": list(WARN_POS_RANGE),
                                     "desviacion_tamano_advertencia": MAX_SIZE_DEVIATION},
            "aprobacion_parametros": args.aprobacion.strip(),
            "advertencias": warnings,
            "revision_pares": stats,
            "revisiones_distinta_en_grupo_por_duplicado_exacto": conflict_dup,
            "precision_cribado_entre_candidatos_revisados":
                (cand_stats["misma"] / cand_stats["revisados"]) if cand_stats["revisados"] else None,
            "aristas_confirmadas_por_revision": len(edges_pairs),
            "aristas_por_duplicado_exacto": len(edges_dup),
            "grupos_multiimagen": len(multi), "imagenes_en_grupos_multiimagen": sum(multi),
            "folds": per_fold,
            "procedencia": prov,
            "codigo_en_commit": prov["codigo_en_commit"],
            "entradas_sha256": {
                "resumen_auditoria.json": C.sha256_file(audit_dir / "resumen_auditoria.json"),
                **{n: C.sha256_file(audit_dir / n) for n in AUDIT_FILES if n != "resumen_auditoria.json"}},
            "auditoria_commit": audit_summary["procedencia"]["commit"],
            "salida_sha256": C.sha256_file(staging / "cv_folds_v2.csv"),
            "nota": "Se controla la dependencia visual detectable (duplicados y casi-duplicados confirmados). "
                    "Los grupos no son pacientes: dos imágenes de la misma paciente pueden tener pHash muy distinto. "
                    "La independencia por paciente dentro del desarrollo no está verificada.",
        }
        C.write_json(staging / "resumen_folds_v2.json", summary)
        validate_staging(staging, ids, labels, groups)

        C.prepare_output(out, (), args.force)  # el destino pudo cambiar durante el cálculo
        backup = C.publish_staging(staging, out)
        staging = None
        print(json.dumps({k: summary[k] for k in ("estado", "modo", "aristas_confirmadas_por_revision",
                                                  "aristas_por_duplicado_exacto", "folds")},
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
