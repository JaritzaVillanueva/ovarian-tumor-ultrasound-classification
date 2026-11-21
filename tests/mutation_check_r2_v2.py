"""Control de mutaciones de G1 / R2 v2 (opcional; NO se recolecta con pytest).

Rompe a propósito UNA guarda por vez en una COPIA del paquete y comprueba que la suite sintética falla.
Una mutación que sobrevive indica una guarda sin prueba que la proteja.

Uso:
  python tests/mutation_check_r2_v2.py <raiz_del_repositorio> <directorio_de_trabajo_FUERA_del_repositorio> [filtro ...]
  (los filtros opcionales son subcadenas de la descripción: solo corren las mutaciones que las contienen)

El directorio de trabajo se borra y se recrea en cada mutación: por seguridad el script se niega a usar
uno que esté dentro del repositorio o que no sea nuevo/vacío. Tarda unos minutos (una corrida de pytest por
mutación). No toca el repositorio ni lee imágenes ni datos de test: usa solo datos sintéticos y
configs/splits/cv_folds_e2.csv (fuente de desarrollo).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# (archivo, texto original, texto mutado, descripción)
MUTATIONS = [
    ("make_folds_v2.py", "        require_complete_review(stats)", "        pass", "folds sin bloqueo por revisión incompleta"),
    ("make_folds_v2.py", "        if problems:\n            raise C.GuardError(\"Partición no viable:", "        if False:\n            raise C.GuardError(\"Partición no viable:", "[no viable] folds con menos de 4 positivas se publican"),
    ("make_folds_v2.py", "edges_pairs + edges_dup", "edges_pairs", "duplicados exactos no se unen"),
    ("make_folds_v2.py", "if len({label_of[i] for i in ids}) > 1:", "if False:", "duplicado exacto contradictorio no aborta"),
    ("make_folds_v2.py", "if label_of[a] != label_of[b]:", "if False:", "par misma contradictorio no aborta"),
    ("make_folds_v2.py", "    if stats[\"duda\"]:", "    if False:", "duda no bloquea"),
    ("make_folds_v2.py", "if not args.aprobacion.strip():", "if False:", "sin --aprobacion"),
    ("make_folds_v2.py", "if n not in (REVIEW_SHEET_FILE, \"resumen_auditoria.json\"))", "if n not in (REVIEW_SHEET_FILE, \"resumen_auditoria.json\", *PAIR_FILES))", "[ciega] la correspondencia hoja->pareja no se verifica por hash"),
    ("make_folds_v2.py", "if (len(set(hojas)) != len(hojas) or", "if False and (len(set(hojas)) != len(hojas) or", "[ciega] la planilla puede no corresponder uno a uno con la correspondencia"),
    ("build_development_audit.py", "    if header != BLIND_COLUMNS:", "    if False:", "[ciega] la planilla puede revelar columnas"),
    ("build_development_audit.py", "if any(r[\"revision\"] or r[\"revisor\"] or r[\"nota\"] for r in sheet_rows):", "if False:", "[ciega] la planilla puede publicarse con anotaciones"),
    ("build_development_audit.py", "PROTECTED_FILES = (REVIEW_SHEET_FILE,)", "PROTECTED_FILES = ()", "[ciega] la planilla revisada no está protegida"),
    ("build_development_audit.py", "BLIND_COLUMNS = [\"hoja\", \"revision\", \"revisor\", \"nota\"]", "BLIND_COLUMNS = [\"hoja\", \"revision\", \"revisor\", \"nota\", \"distance\"]", "[ciega] la planilla incluye la distancia"),
    ("make_folds_v2.py", "    if unhashed:", "    if False:", "el resumen puede omitir hashes de artefactos"),
    ("make_folds_v2.py", "or C.sha256_file(source) != summary.get(\"fuente\", {}).get(\"sha256\"):", "or False:", "fuente distinta de la auditada no se detecta"),
    ("make_folds_v2.py", "if not code or unversioned or not prov.get(\"fuente_en_commit\"):", "if False:", "se confía en el commit autodeclarado de la auditoría"),
    ("make_folds_v2.py", "forbidden_paths=[audit_dir]", "forbidden_paths=[]", "--out puede coincidir con la auditoría revisada"),
    ("common.py", "    if reviewed:", "    if False:", "revisados sobrescribibles con --force"),
    ("common.py", "if any(out.iterdir()) and not force:", "if False:", "sobrescritura sin --force"),
    ("common.py", "row[\"source_split\"] != DEV_SOURCE_SPLIT", "False", "acepta filas que no son train"),
    ("common.py", "if os.path.normcase(str(allowed_resolved)) != os.path.normcase(str(allowed)):", "if False:", "directorio autorizado redirigido no se detecta"),
    ("common.py", "if not prov[\"codigo_en_commit\"]:", "if False:", "código sin commit no se detecta"),
    ("common.py", "committed = in_head is not None and in_work is not None and in_head == in_work", "committed = True", "archivos modificados se dan por versionados"),
    ("common.py", "    inside = inside or allowed in out.parents", "    inside = True", "--out sin restricción de directorio"),
    ("common.py", "    if bad:", "    if False:", "--out admite directorios de agentes o reportes"),
    ("build_development_audit.py", "validate_staging(staging, expected, expected_total, with_sheets=not args.no_sheets)", "pass", "no se valida antes de publicar"),
    ("build_development_audit.py", "        C.prepare_output(out, PROTECTED_FILES, args.force)  # el destino pudo cambiar", "        pass  # ", "no se re-comprueba el destino antes de publicar"),
    ("build_development_audit.py", "\"categorias_originales_distintas\": len(set(classes)) > 1,", "\"categorias_originales_distintas\": False,", "indicador de categorías siempre falso"),
    ("build_development_audit.py", "\"etiquetas_binarias_contradictorias\": len(elig_labels) > 1}", "\"etiquetas_binarias_contradictorias\": len(set(classes)) > 1}", "contradicción binaria = categorías distintas"),
    ("build_development_audit.py", "label = C.label_from_class(c)", "label = int(c in (6, 7))", "etiqueta con el mapeo histórico (6 y 7 positivos)"),
    ("build_development_audit.py", "resolved = C.resolve_dev_image(root, r[\"e1_split\"], r[\"image_name\"])", "resolved = Path(root) / C.DATA_REL / r[\"e1_split\"] / \"images\" / r[\"image_name\"]", "se omite la verificación de ruta resuelta"),
]


def run_suite(work):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "tests/test_r2_v2.py", "-q", "-x"],
                       cwd=work, capture_output=True, text=True, env=env)
    return r.returncode, (r.stdout.strip().splitlines() or [""])[-1]


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    only = argv[3:]  # opcional: subcadenas; solo corre las mutaciones cuya descripción contenga alguna
    repo, work = Path(argv[1]).resolve(), Path(argv[2]).resolve()
    if work == repo or repo in work.parents or work in repo.parents:
        print("ERROR: el directorio de trabajo no puede estar dentro (ni contener) al repositorio.")
        return 2
    if work.exists() and any(work.iterdir()):
        print(f"ERROR: {work} no está vacío; usa un directorio nuevo (se borra en cada mutación).")
        return 2
    survivors = 0
    selected = [m for m in MUTATIONS if not only or any(s in m[3] for s in only)]
    if not selected:
        print("ERROR: ninguna mutación coincide con el filtro.")
        return 2
    for fname, old, new, label in selected:
        if work.exists():
            shutil.rmtree(work)
        (work / "src").mkdir(parents=True)
        (work / "tests").mkdir()
        (work / "configs" / "splits").mkdir(parents=True)
        shutil.copytree(repo / "src" / "r2_v2", work / "src" / "r2_v2", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(repo / "tests" / "test_r2_v2.py", work / "tests")
        (work / "tests" / "__init__.py").write_text("", encoding="utf-8")
        shutil.copy(repo / "configs" / "splits" / "cv_folds_e2.csv", work / "configs" / "splits")
        target = work / "src" / "r2_v2" / fname
        text = target.read_text(encoding="utf-8")
        if old not in text:
            print(f"[PATRON NO ENCONTRADO] {fname}: {label}")
            survivors += 1
            continue
        target.write_text(text.replace(old, new, 1), encoding="utf-8")
        code, tail = run_suite(work)
        survivors += code == 0
        print(f"[{'DETECTADA ' if code else 'SOBREVIVE  '}] {fname}: {label}  -> {tail}")
    shutil.rmtree(work, ignore_errors=True)
    print(f"\nMutaciones no detectadas o inválidas: {survivors} de {len(selected)} ejecutadas ({len(MUTATIONS)} definidas)")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
