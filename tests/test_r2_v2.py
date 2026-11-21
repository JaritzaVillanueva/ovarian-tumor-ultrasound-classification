"""Pruebas de G1 / R2 v2 con datos SINTÉTICOS en directorios temporales.

No abren imágenes reales ni el test oficial. Única lectura de datos reales: la fuente de desarrollo
configs/splits/cv_folds_e2.csv (solo CSV; 1000 filas train) en test_real_source_counts_without_images.
"""

import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageEnhance

from src.r2_v2 import build_development_audit as audit
from src.r2_v2 import common as C
from src.r2_v2 import make_folds_v2 as mf

# Proyecto sintético de 100 imágenes: 90 elegibles (65 no malignas + 25 malignas) y 10 normales.
SMALL_COUNTS = {0: 20, 1: 15, 2: 15, 3: 8, 4: 4, 6: 3, 7: 25, 5: 10}
SMALL_EXPECTED = {"eligible": 90, "non_malignant": 65, "malignant": 25, "normal": 10}
SMALL_TOTAL = 100
SOURCE_HEADER = ["image_id", "image_name", "source_split", "original_class", "binary_label", "binary_class",
                 "e1_split", "fold"]
BASE_PLANTS = [("near", (1, 0), (1, 1)), ("exact", (0, 0), (6, 0)), ("exact", (2, 0), (5, 0))]


# ---------------------------------------------------------------- utilidades de prueba
def make_project(root, historical="honest", plants=()):
    root = Path(root)
    rng = random.Random(7)
    classes = [c for c, n in SMALL_COUNTS.items() for _ in range(n)]
    rng.shuffle(classes)
    base = root / C.DATA_REL
    for s in ("train", "validation", "test"):
        (base / s / "images").mkdir(parents=True)
    info, e1_of, rows = {}, {}, []
    for pos, c in enumerate(classes):
        i = pos + 1
        e1 = "train" if pos < 80 else "validation"
        info[i], e1_of[i] = c, e1
        arr = (np.random.default_rng(1000 + i).random((48, 64)) * 255).astype(np.uint8)
        Image.fromarray(arr).save(base / e1 / "images" / f"{i}.JPG", "JPEG", quality=95)
        hist = ({"binary_label": int(c in (6, 7)), "binary_class": "hist", "fold": i % 5} if historical == "honest"
                else {"binary_label": 1 - int(c in (6, 7)), "binary_class": "basura", "fold": 99})
        rows.append({"image_id": i, "image_name": f"{i}.JPG", "source_split": "train", "original_class": c,
                     "e1_split": e1, **hist})
    sentinel = (np.random.default_rng(5).random((48, 64)) * 255).astype(np.uint8)
    Image.fromarray(sentinel).save(base / "test" / "images" / "9999.JPG", "JPEG")
    (root / "configs" / "splits").mkdir(parents=True)
    (root / C.SOURCE_REL).write_text(C.csv_text(rows, SOURCE_HEADER), encoding="utf-8", newline="")

    def pick(cls, k):
        return sorted(i for i, c in info.items() if c == cls)[k]

    for kind, (ca, ka), (cb, kb) in plants:
        a, b = pick(ca, ka), pick(cb, kb)
        src, dst = base / e1_of[a] / "images" / f"{a}.JPG", base / e1_of[b] / "images" / f"{b}.JPG"
        if kind == "exact":
            shutil.copyfile(src, dst)
        else:
            with Image.open(src) as im:
                ImageEnhance.Brightness(im.convert("L")).enhance(1.05).save(dst, "JPEG", quality=95)
    return {"root": root, "classes": info, "pick": pick}


def adir(root):
    return Path(root) / audit.DEFAULT_OUT_REL


def run_audit(root, *extra):
    return audit.main([*extra, "--permitir-codigo-sin-commit"], project_root=root,
                      expected=SMALL_EXPECTED, expected_total=SMALL_TOTAL)


def run_folds(root, *extra, aprobacion="D-TEST"):
    args = ["--permitir-codigo-sin-commit", *extra]
    if aprobacion:
        args += ["--aprobacion", aprobacion]
    return mf.main(args, project_root=root, expected=SMALL_EXPECTED)


def pair_of_hoja(root):
    """hoja -> pareja (ids ordenados), leyendo la CORRESPONDENCIA que el revisor no ve."""
    out = {}
    for name in mf.PAIR_FILES:
        for r in C.read_csv_rows(adir(root) / name):
            out[r["hoja"]] = tuple(sorted((r["image_id_a"], r["image_id_b"]), key=int))
    return out


def annotate(root, misma=(), default="distinta", revisor="T", skip_revisor=False):
    """Anota la PLANILLA CIEGA (hoja, revision, revisor, nota); `misma` se indica por pareja vía la correspondencia."""
    mapping = pair_of_hoja(root)
    p = adir(root) / audit.REVIEW_SHEET_FILE
    header, rows = C.read_csv_with_header(p)
    for r in rows:
        r["revision"] = "misma" if mapping[r["hoja"]] in misma else default
        r["revisor"] = "" if skip_revisor else revisor
    p.write_text(C.csv_text(rows, header), encoding="utf-8", newline="")


def fake_git(committed, blob="aaa"):
    def run(root, *args):
        if args[:2] == ("rev-parse", "HEAD"):
            return "deadbeef"
        if args[0] == "rev-parse":
            return blob
        if args[0] == "hash-object":
            return blob if committed else "bbb"
        if args[0] == "status":
            return "" if committed else " M src/r2_v2/common.py"
        return None
    return run


def make_dir_link(link, target):
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        if sys.platform != "win32":
            pytest.skip("no se pueden crear enlaces simbólicos")
        res = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True)
        if res.returncode != 0:
            pytest.skip("no se pueden crear enlaces simbólicos ni uniones")


# Referencias a las implementaciones REALES, capturadas antes de que los fixtures las sustituyan.
REAL_GIT = C._git
REAL_ASSIGN = mf.assign_folds


def stable_assign_folds(ids, labels, groups, n_folds=5, seed=42):
    """Asignador DETERMINISTA y sin scikit-learn, solo para pruebas de extremo a extremo.

    StratifiedGroupKFold es una heurística cuyo reparto de positivas cambia entre versiones de scikit-learn
    (p. ej. 1.7.2 frente a 1.9.1): no se puede depender de su distribución exacta en pruebas de la lógica del
    programa. Este asignador respeta los grupos y reparte las positivas de forma equilibrada. El `assign_folds`
    real se prueba aparte, solo por sus invariantes (ver test_real_assign_folds_*).
    """
    members = {}
    for i, y in zip(ids, labels):
        entry = members.setdefault(groups[i], [0, []])
        entry[0] += y
        entry[1].append(i)
    pos, size, fold_of = [0] * n_folds, [0] * n_folds, {}
    with_pos = sorted((kv for kv in members.items() if kv[1][0] > 0), key=lambda kv: (-kv[1][0], -len(kv[1][1]), str(kv[0])))
    no_pos = sorted((kv for kv in members.items() if kv[1][0] == 0), key=lambda kv: (-len(kv[1][1]), str(kv[0])))
    for g, (p, ms) in with_pos:
        k = min(range(n_folds), key=lambda f: (pos[f], size[f], f))
        pos[k] += p
        size[k] += len(ms)
        fold_of.update({i: k for i in ms})
    for g, (p, ms) in no_pos:
        k = min(range(n_folds), key=lambda f: (size[f], f))
        size[k] += len(ms)
        fold_of.update({i: k for i in ms})
    return [fold_of[i] for i in ids]


@pytest.fixture(autouse=True)
def _isolated_from_external_state(monkeypatch):
    """Cada prueba parte de un estado controlado: sin depender del Git real ni de la versión de scikit-learn."""
    monkeypatch.setattr(C, "_git", fake_git(False))               # código "sin commit", salvo que la prueba diga otra cosa
    monkeypatch.setattr(mf, "assign_folds", stable_assign_folds)  # las pruebas del algoritmo real usan REAL_ASSIGN


@pytest.fixture(scope="module")
def base_info(tmp_path_factory):
    root = tmp_path_factory.mktemp("base")
    info = make_project(root, plants=BASE_PLANTS)
    with pytest.MonkeyPatch.context() as mp:                      # la auditoría base no depende del Git real
        mp.setattr(C, "_git", fake_git(False))
        assert run_audit(root) == 0
    return info


@pytest.fixture
def project(base_info, tmp_path):
    shutil.copytree(base_info["root"], tmp_path / "p")
    return tmp_path / "p"


def near_pair(info):
    return tuple(sorted((str(info["pick"](1, 0)), str(info["pick"](1, 1))), key=int))


# ---------------------------------------------------------------- fuente única y mapeo
def test_real_source_counts_without_images():
    src = C.PROJECT_ROOT / C.SOURCE_REL
    if not src.is_file():
        pytest.skip("fuente no disponible en este clon")
    eligible, normals = audit.build_manifests(C.load_development_source(src))
    assert audit.check_counts(eligible, normals) == C.EXPECTED
    assert {r["original_class"] for r in normals} == {5}
    assert all(r["label_v2"] == (1 if r["original_class"] == 7 else 0) for r in eligible)
    assert all(r["label_v2"] == 0 for r in eligible if r["original_class"] == 6)


def test_label_mapping():
    assert [C.label_from_class(c) for c in (0, 1, 2, 3, 4, 6)] == [0] * 6
    assert C.label_from_class(7) == 1
    assert C.label_from_class(5) is None
    with pytest.raises(C.GuardError):
        C.label_from_class(9)


def test_source_columns_missing_are_rejected(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("image_id,image_name,original_class\n1,1.JPG,0\n", encoding="utf-8")
    with pytest.raises(C.GuardError, match="columnas"):
        C.load_development_source(p, expected_total=1)


def _source(tmp_path, rows):
    p = tmp_path / "s.csv"
    p.write_text(C.csv_text(rows, SOURCE_HEADER), encoding="utf-8", newline="")
    return p


def _row(i=1, **kw):
    r = {"image_id": str(i), "image_name": f"{i}.JPG", "source_split": "train", "original_class": "0",
         "binary_label": "0", "binary_class": "x", "e1_split": "train", "fold": "0"}
    r.update(kw)
    return r


def test_source_rejects_non_train_rows(tmp_path):
    for bad in ("validation", "test", "Train", ""):
        with pytest.raises(C.GuardError, match="source_split"):
            C.load_development_source(_source(tmp_path, [_row(source_split=bad)]), expected_total=1)


def test_source_rejects_unknown_e1_split_and_class(tmp_path):
    with pytest.raises(C.GuardError, match="e1_split"):
        C.load_development_source(_source(tmp_path, [_row(e1_split="test")]), expected_total=1)
    with pytest.raises(C.GuardError, match="categoría"):
        C.load_development_source(_source(tmp_path, [_row(original_class="9")]), expected_total=1)


@pytest.mark.parametrize("name", ["../x.JPG", "a/b.JPG", "a\\b.JPG", "/abs.JPG", "C:x.JPG", ".h.JPG", ""])
def test_source_rejects_unsafe_image_names(tmp_path, name):
    with pytest.raises(C.GuardError):
        C.load_development_source(_source(tmp_path, [_row(image_name=name)]), expected_total=1)


def test_source_integrity_total_and_duplicates(tmp_path):
    with pytest.raises(C.GuardError, match="esperado 2"):
        C.load_development_source(_source(tmp_path, [_row(1)]), expected_total=2)
    with pytest.raises(C.GuardError, match="repetidos"):
        C.load_development_source(_source(tmp_path, [_row(1), _row(1)]), expected_total=2)


def test_loader_drops_historical_columns(tmp_path):
    rows = C.load_development_source(_source(tmp_path, [_row(binary_label="1", fold="4")]), expected_total=1)
    assert set(rows[0]) == set(C.SOURCE_COLUMNS)


def test_code_never_references_mixed_manifests_or_historical_labels():
    import ast
    historical = {"binary_label", "binary_class"}
    for p in C.PACKAGE_DIR.glob("*.py"):
        text = p.read_text(encoding="utf-8")
        for forbidden in ("mmotu_experimental_split", "experimental_image_path", "original_split"):
            assert forbidden not in text, f"{p.name} contiene {forbidden}"
        for node in ast.walk(ast.parse(text)):  # ningún acceso r["binary_label"] ni r.get("binary_label")
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                assert node.slice.value not in historical, f"{p.name}:{node.lineno} indexa {node.slice.value}"
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                    and node.args and isinstance(node.args[0], ast.Constant)):
                assert node.args[0].value not in historical, f"{p.name}:{node.lineno} lee {node.args[0].value}"


# ---------------------------------------------------------------- rutas resueltas y test
def test_path_mentions_test_is_case_and_separator_insensitive():
    assert C.path_mentions_test("data/x/test/images/1.JPG")
    assert C.path_mentions_test("data\\x\\Test\\images\\1.JPG")
    assert C.path_mentions_test("TEST/a")
    assert not C.path_mentions_test("data/x/train/images/1.JPG")
    assert not C.path_mentions_test("data/contest/train")


def test_resolve_dev_image_ok_and_rejections(tmp_path):
    make_project(tmp_path)
    p = C.resolve_dev_image(tmp_path, "train", "1.JPG")
    assert p.is_file() and p.parent.name == "images"
    with pytest.raises(C.GuardError):
        C.resolve_dev_image(tmp_path, "test", "9999.JPG")        # e1_split fuera de lo autorizado
    with pytest.raises(C.GuardError):
        C.resolve_dev_image(tmp_path, "train", "../../test/images/9999.JPG")
    with pytest.raises(C.GuardError, match="no encontrada"):
        C.resolve_dev_image(tmp_path, "train", "no_existe.JPG")


def test_resolve_rejects_redirected_authorized_directory(tmp_path):
    make_project(tmp_path)
    images = tmp_path / C.DATA_REL / "validation" / "images"
    shutil.rmtree(images)
    make_dir_link(images, tmp_path / C.DATA_REL / "test" / "images")
    with pytest.raises(C.GuardError, match="redirigido"):
        C.resolve_dev_image(tmp_path, "validation", "9999.JPG")


# ---------------------------------------------------------------- pHash
def _img(seed, size=(120, 90)):
    rng = np.random.default_rng(seed)
    base = rng.random((size[1] // 10, size[0] // 10))
    return Image.fromarray((np.kron(base, np.ones((10, 10))) * 255).astype(np.uint8))


def test_phash_deterministic_and_robust():
    a = _img(0)
    assert audit.phash(a) == audit.phash(a.copy())
    brighter = ImageEnhance.Brightness(a).enhance(1.1)
    assert audit.hamming(audit.phash(a), audit.phash(brighter)) <= 6
    assert audit.hamming(audit.phash(a), audit.phash(_img(1))) > 12


def test_pairwise_matches_scalar_hamming():
    hs = [audit.phash(_img(s)) for s in range(5)]
    d = audit.pairwise_hamming(hs)
    for i in range(5):
        for j in range(5):
            assert d[i, j] == audit.hamming(hs[i], hs[j])


# ---------------------------------------------------------------- código versionado
def test_code_state_detects_committed_modified_untracked_and_no_git():
    files = ["src/r2_v2/common.py"]
    assert C.code_state(C.PROJECT_ROOT, files, fake_git(True))["en_commit"] is True
    assert C.code_state(C.PROJECT_ROOT, files, fake_git(False))["en_commit"] is False

    def untracked(root, *a):
        return None if a[0] == "rev-parse" and a[1] != "HEAD" else fake_git(True)(root, *a)
    assert C.code_state(C.PROJECT_ROOT, files, untracked)["en_commit"] is False
    assert C.code_state(C.PROJECT_ROOT, files, lambda *a: None)["en_commit"] is False


def test_real_git_runner_on_real_repository():
    """Solo comprueba propiedades que no dependen de si el código ya está o no en un commit."""
    if REAL_GIT(C.PROJECT_ROOT, "rev-parse", "HEAD") is None or not (C.PROJECT_ROOT / C.SOURCE_REL).is_file():
        pytest.skip("git no disponible, no es un clon o falta la fuente")
    state = C.code_state(C.PROJECT_ROOT, [C.SOURCE_REL, "src/r2_v2/__no_existe__.py"], REAL_GIT)
    assert state["archivos"]["src/r2_v2/__no_existe__.py"]["en_commit"] is False   # no versionado ni existente
    assert state["en_commit"] is False
    assert state["archivos"][C.SOURCE_REL]["blob"]                                  # id de git, independiente del SO
    assert state["commit"] == REAL_GIT(C.PROJECT_ROOT, "rev-parse", "HEAD")


def test_code_state_records_platform_independent_blob():
    s = C.code_state(C.PROJECT_ROOT, ["src/r2_v2/common.py"], fake_git(True))
    assert s["archivos"]["src/r2_v2/common.py"]["blob"] == "aaa" and s["archivos"]["src/r2_v2/common.py"]["sha256"]


def test_provenance_records_flags_versions_and_command(monkeypatch):
    monkeypatch.setattr(C, "_git", fake_git(True))
    prov = C.provenance("src.r2_v2.x", ["--no-sheets"])
    assert prov["codigo_en_commit"] is True and prov["fuente_en_commit"] is True
    assert prov["commit"] == "deadbeef" and prov["arbol_con_cambios"] is False
    assert prov["comando"].endswith("-m src.r2_v2.x --no-sheets")
    assert prov["versiones"]["python"] and "sklearn" in prov["versiones"]
    assert set(prov["archivos_codigo"]) == set(C.package_code_files())


def test_require_versioned_code(monkeypatch):
    monkeypatch.setattr(C, "_git", fake_git(False))
    prov = C.provenance("m", [])
    with pytest.raises(C.GuardError, match="sin commit"):
        C.require_versioned_code(prov, allow_uncommitted=False)
    assert C.require_versioned_code(prov, allow_uncommitted=True) is False


def test_code_and_source_versioning_are_checked_independently(monkeypatch):
    def only(committed_paths):
        def run(root, *args):
            if args[:2] == ("rev-parse", "HEAD"):
                return "deadbeef"
            if args[0] == "rev-parse":
                return "aaa"
            if args[0] == "hash-object":
                return "aaa" if args[-1] in committed_paths else "bbb"
            return ""
        return run
    monkeypatch.setattr(C, "_git", only({C.SOURCE_REL}))           # fuente OK, código modificado
    prov = C.provenance("m", [])
    assert prov["codigo_en_commit"] is False and prov["fuente_en_commit"] is True
    with pytest.raises(C.GuardError, match="código sin commit"):
        C.require_versioned_code(prov, allow_uncommitted=False)
    monkeypatch.setattr(C, "_git", only(set(C.package_code_files())))   # código OK, fuente modificada
    prov = C.provenance("m", [])
    assert prov["codigo_en_commit"] is True and prov["fuente_en_commit"] is False
    with pytest.raises(C.GuardError, match=C.SOURCE_REL):
        C.require_versioned_code(prov, allow_uncommitted=False)


def test_audit_refuses_unversioned_code_and_writes_nothing(tmp_path, monkeypatch, capsys):
    make_project(tmp_path)
    monkeypatch.setattr(C, "_git", fake_git(False))
    rc = audit.main([], project_root=tmp_path, expected=SMALL_EXPECTED, expected_total=SMALL_TOTAL)
    assert rc == 1 and "sin commit" in capsys.readouterr().err
    assert not adir(tmp_path).exists() and not list(adir(tmp_path).parent.glob(".*staging*"))


# ---------------------------------------------------------------- auditoría de extremo a extremo
def test_audit_outputs_counts_and_sheets(project):
    out = adir(project)
    for name in ("development_eligible_v2.csv", "normal_development_controls.csv", "image_hashes.csv",
                 "exact_duplicates.csv", "candidate_pairs.csv", "calibration_pairs.csv", "revision_ciega.csv",
                 "resumen_auditoria.json"):
        assert (out / name).is_file()
    s = json.loads((out / "resumen_auditoria.json").read_text(encoding="utf-8"))
    assert s["recuentos"] == SMALL_EXPECTED
    assert s["fuente"]["source_split_solo_train"] is True
    assert s["fuente"]["columnas_historicas_ignoradas"] == ["binary_label", "binary_class", "fold"]
    assert "codigo_en_commit" in s and s["acceso_a_datos"]["manifiestos_mixtos_abiertos"] is False
    assert s["acceso_a_datos"]["imagenes_distintas_abiertas"] >= SMALL_TOTAL
    assert "NO VALIDA" in s["estado"]  # se ejecutó con --permitir-codigo-sin-commit
    elig = C.read_csv_rows(out / "development_eligible_v2.csv")
    assert all(r["image_path"].startswith(C.DATA_REL) and "/test/" not in r["image_path"] for r in elig)
    pairs = C.read_csv_rows(out / "candidate_pairs.csv") + C.read_csv_rows(out / "calibration_pairs.csv")
    sheets = sorted(p.name for p in (out / audit.SHEETS_DIR).glob("par_*.png"))
    assert sheets and len(sheets) == len(pairs) == s["hojas_revision"]["n"]
    assert all(r["hoja"] and (out / r["hoja"]).is_file() for r in pairs)
    assert all(r["hoja"].startswith("review_sheets/par_") for r in pairs)  # sin distancia ni origen
    assert (out / audit.SHEETS_DIR / ".gitignore").read_text(encoding="utf-8").startswith("*")


def test_audit_sha256_in_summary_match_files(project):
    out = adir(project)
    s = json.loads((out / "resumen_auditoria.json").read_text(encoding="utf-8"))
    assert s["salidas_sha256"] and all(C.sha256_file(out / n) == h for n, h in s["salidas_sha256"].items())
    assert set(audit.KEY_FILES) <= set(s["salidas_sha256"])               # la correspondencia se verifica por hash exacto
    assert audit.REVIEW_SHEET_FILE not in s["salidas_sha256"]            # la planilla la edita el revisor
    assert s["planilla_ciega"]["hojas_sha256"] == C.hojas_digest(r["hoja"] for r in C.read_csv_rows(out / "revision_ciega.csv"))
    assert s["phash"]["corte_cribado"] == 12 and s["phash"]["bandas_calibracion"] == [[13, 14], [15, 16], [17, 18]]


def test_audit_never_opens_test_images(tmp_path, monkeypatch):
    make_project(tmp_path, plants=BASE_PLANTS)
    opened = []
    real_open = Image.open

    def spy(fp, *a, **k):
        opened.append(str(fp).replace("\\", "/"))
        return real_open(fp, *a, **k)
    monkeypatch.setattr(Image, "open", spy)
    assert run_audit(tmp_path) == 0
    assert opened and not any("/test/" in p.lower() for p in opened)
    allowed = tuple(f"{C.DATA_REL}/{s}/images/" for s in C.DEV_E1_SPLITS)
    assert all(any(a in p for a in allowed) for p in opened)


def test_audit_ignores_historical_columns(tmp_path):
    outputs = []
    for hist in ("honest", "garbage"):
        root = tmp_path / hist
        make_project(root, historical=hist, plants=BASE_PLANTS)
        assert run_audit(root, "--no-sheets") == 0
        out = adir(root)
        outputs.append({n: (out / n).read_bytes() for n in
                        ("development_eligible_v2.csv", "normal_development_controls.csv", "image_hashes.csv",
                         "exact_duplicates.csv", "candidate_pairs.csv", "calibration_pairs.csv",
                         "revision_ciega.csv")})
    assert outputs[0] == outputs[1]


def test_audit_exact_duplicates_and_category_indicator(project, base_info):
    pick = base_info["pick"]
    dups = C.read_csv_rows(adir(project) / "exact_duplicates.csv")
    assert len(dups) == 2
    by_ids = {frozenset(d["image_ids"].split(";")): d for d in dups}
    # clase 0 vs 6: categorías distintas pero la MISMA etiqueta binaria (indicadores independientes)
    d06 = by_ids[frozenset({str(pick(0, 0)), str(pick(6, 0))})]
    assert d06["categorias_originales_distintas"] == "True"
    assert d06["etiquetas_binarias_contradictorias"] == "False"
    assert d06["mezcla_normal_y_lesion"] == "False"
    # normal (5) vs lesión (2): se informa la mezcla; no hay contradicción binaria entre elegibles
    d52 = by_ids[frozenset({str(pick(2, 0)), str(pick(5, 0))})]
    assert d52["mezcla_normal_y_lesion"] == "True" and d52["n_elegibles"] == "1"
    assert d52["etiquetas_binarias_contradictorias"] == "False"
    s = json.loads((adir(project) / "resumen_auditoria.json").read_text(encoding="utf-8"))
    assert s["duplicados_exactos_con_categorias_originales_distintas"] == 2
    assert s["duplicados_exactos_con_etiquetas_binarias_contradictorias"] == 0
    assert s["duplicados_exactos_con_mezcla_normal_y_lesion"] == 1


def test_audit_pairs_report_category_indicator_independent_of_binary_contradiction(project, base_info):
    pairs = C.read_csv_rows(adir(project) / "candidate_pairs.csv")
    exact06 = next(r for r in pairs if {r["image_id_a"], r["image_id_b"]} ==
                   {str(base_info["pick"](0, 0)), str(base_info["pick"](6, 0))})
    assert exact06["distance"] == "0"
    assert exact06["categorias_originales_distintas"] == "True" and exact06["etiquetas_contradictorias"] == "False"
    near = next(r for r in pairs if (r["image_id_a"], r["image_id_b"]) == near_pair(base_info))
    assert near["categorias_originales_distintas"] == "False"
    assert all("revision" not in r and "revisor" not in r for r in pairs)   # la correspondencia no se anota


def test_audit_exact_duplicate_with_contradictory_binary_labels(tmp_path):
    info = make_project(tmp_path, plants=[("exact", (0, 0), (7, 0))])
    assert run_audit(tmp_path, "--no-sheets") == 0
    d = C.read_csv_rows(adir(tmp_path) / "exact_duplicates.csv")[0]
    assert d["etiquetas_binarias_contradictorias"] == "True" and d["categorias_originales_distintas"] == "True"
    s = json.loads((adir(tmp_path) / "resumen_auditoria.json").read_text(encoding="utf-8"))
    assert s["duplicados_exactos_con_etiquetas_binarias_contradictorias"] == 1
    assert info["classes"][info["pick"](7, 0)] == 7


def test_audit_wrong_counts_abort_without_writing(tmp_path, capsys):
    make_project(tmp_path)
    bad = dict(SMALL_EXPECTED, malignant=24)
    rc = audit.main(["--permitir-codigo-sin-commit"], project_root=tmp_path, expected=bad,
                    expected_total=SMALL_TOTAL)
    assert rc == 1 and "Recuentos inesperados" in capsys.readouterr().err
    assert not adir(tmp_path).exists()


def test_audit_missing_image_fails_before_writing(tmp_path, capsys):
    make_project(tmp_path)
    (tmp_path / C.DATA_REL / "train" / "images" / "3.JPG").unlink()
    assert run_audit(tmp_path) == 1
    assert "no encontrada" in capsys.readouterr().err
    assert not adir(tmp_path).exists() and not list(adir(tmp_path).parent.glob(".*staging*"))


def test_audit_failure_mid_preparation_leaves_nothing(tmp_path, monkeypatch):
    make_project(tmp_path, plants=BASE_PLANTS)

    def boom(*a, **k):
        raise RuntimeError("fallo simulado al escribir una hoja")
    monkeypatch.setattr(audit, "make_review_sheet", boom)
    with pytest.raises(RuntimeError):
        run_audit(tmp_path)
    assert not adir(tmp_path).exists() and not list(adir(tmp_path).parent.glob(".*staging*"))


def test_audit_failed_validation_leaves_nothing(tmp_path, monkeypatch, capsys):
    make_project(tmp_path)
    monkeypatch.setattr(audit, "validate_staging", lambda *a, **k: (_ for _ in ()).throw(C.GuardError("inválido")))
    assert run_audit(tmp_path) == 1 and "inválido" in capsys.readouterr().err
    assert not adir(tmp_path).exists() and not list(adir(tmp_path).parent.glob(".*staging*"))


def test_validate_staging_detects_tampering(project):
    stage = project / "stage"
    shutil.copytree(adir(project), stage)
    assert audit.validate_staging(stage, SMALL_EXPECTED, SMALL_TOTAL, with_sheets=True)
    (stage / "image_hashes.csv").write_text("x\n", encoding="utf-8")
    with pytest.raises(C.GuardError, match="hash"):
        audit.validate_staging(stage, SMALL_EXPECTED, SMALL_TOTAL, with_sheets=True)


# ---------------------------------------------------------------- no sobrescritura
def test_audit_refuses_existing_outputs_without_force(project, capsys):
    before = (adir(project) / "resumen_auditoria.json").read_bytes()
    assert run_audit(project) == 1 and "--force" in capsys.readouterr().err
    assert (adir(project) / "resumen_auditoria.json").read_bytes() == before


def test_audit_force_replaces_unreviewed_and_keeps_previous(project):
    assert run_audit(project, "--force") == 0
    backups = list(adir(project).parent.glob("auditoria_desarrollo.reemplazado-*"))
    assert len(backups) == 1 and (backups[0] / "resumen_auditoria.json").is_file()
    assert (adir(project) / "resumen_auditoria.json").is_file()


@pytest.mark.parametrize("name", audit.PROTECTED_FILES)
def test_reviewed_files_are_never_overwritten_even_with_force(project, capsys, name):
    annotate(project)
    before = (adir(project) / name).read_bytes()
    assert run_audit(project, "--force") == 1
    assert "nunca se sobrescriben" in capsys.readouterr().err
    assert (adir(project) / name).read_bytes() == before
    assert not list(adir(project).parent.glob("auditoria_desarrollo.reemplazado-*"))


def test_single_note_counts_as_review(project):
    p = adir(project) / "revision_ciega.csv"
    header, rows = C.read_csv_with_header(p)
    rows[0]["nota"] = "mirar"
    p.write_text(C.csv_text(rows, header), encoding="utf-8", newline="")
    assert C.has_review_annotations(p)


def test_unreadable_pair_file_is_protected(tmp_path):
    p = tmp_path / "candidate_pairs.csv"
    p.write_bytes(b"\xff\xfe\x00bad")
    assert C.has_review_annotations(p)


def test_publish_restores_previous_output_if_replace_fails(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.txt").write_text("viejo", encoding="utf-8")
    stage = C.new_staging(out)
    (stage / "b.txt").write_text("nuevo", encoding="utf-8")
    real = os.replace

    def flaky(src, dst):
        if Path(src) == stage:
            raise OSError("fallo simulado")
        return real(src, dst)
    monkeypatch.setattr(C.os, "replace", flaky)
    with pytest.raises(OSError):
        C.publish_staging(stage, out)
    assert (out / "a.txt").read_text(encoding="utf-8") == "viejo"
    C.discard_staging(stage)
    assert not stage.exists()


# ---------------------------------------------------------------- revisión y grupos
def test_confirmed_edges_only_misma_and_counts():
    pairs = [{"image_id_a": "1", "image_id_b": "2", "revision": "misma", "revisor": "T"},
             {"image_id_a": "2", "image_id_b": "3", "revision": "distinta", "revisor": "T"},
             {"image_id_a": "3", "image_id_b": "4", "revision": "", "revisor": ""}]
    edges, stats = mf.confirmed_edges(pairs, {"1", "2", "3", "4"})
    assert edges == [("1", "2")]
    assert stats["revisados"] == 2 and stats["sin_revisar"] == 1


def test_invalid_review_value_raises():
    with pytest.raises(C.GuardError):
        mf.confirmed_edges([{"image_id_a": "1", "image_id_b": "2", "revision": "si"}], {"1", "2"})


def test_require_complete_review_blocks_unreviewed_duda_and_missing_revisor():
    ok = {"revisados": 3, "misma": 1, "distinta": 2, "duda": 0, "sin_revisar": 0, "sin_revisor": 0}
    mf.require_complete_review(ok)
    for key in ("sin_revisar", "duda", "sin_revisor"):
        with pytest.raises(C.GuardError, match="incompleta"):
            mf.require_complete_review({**ok, key: 1})


def test_merge_pair_reviews_dedups_and_detects_conflicts():
    a = {"image_id_a": "1", "image_id_b": "2", "revision": "misma"}
    same = {"image_id_a": "2", "image_id_b": "1", "revision": "misma"}
    assert len(mf.merge_pair_reviews({"c": [a], "k": [same]})) == 1
    with pytest.raises(C.GuardError, match="revisiones distintas"):
        mf.merge_pair_reviews({"c": [a], "k": [{**same, "revision": "distinta"}]})


def test_exact_duplicate_edges_and_contradiction():
    label = {"1": 0, "2": 0, "3": 1}
    dup_ok = [{"image_ids": "1;2;9"}, {"image_ids": "9;3"}]  # 9 no es elegible (normal)
    assert mf.exact_duplicate_edges(dup_ok, label) == [("1", "2")]
    with pytest.raises(C.GuardError, match="contradictorias"):
        mf.exact_duplicate_edges([{"image_ids": "2;3"}], label)


def test_groups_transitive():
    g = mf.build_groups(["1", "2", "3", "4"], [("1", "2"), ("2", "3")])
    assert g["1"] == g["2"] == g["3"] != g["4"]


def _groups_820():
    ids = [str(i) for i in range(820)]
    labels = [1 if i < 38 else 0 for i in range(820)]
    edges = [(str(i), str(i + 1)) for i in range(100, 300, 2)] + [("0", "1")]
    return ids, labels, mf.build_groups(ids, edges)


def test_real_assign_folds_invariants_do_not_depend_on_the_sklearn_balance():
    """Solo lo que StratifiedGroupKFold garantiza en cualquier versión: no el reparto exacto de positivas."""
    ids, labels, groups = _groups_820()
    folds = REAL_ASSIGN(ids, labels, groups)
    assert len(folds) == 820 and set(folds) == {0, 1, 2, 3, 4}            # todas asignadas, los cinco folds usados
    by_group = {}
    for i, f in zip(ids, folds):
        by_group.setdefault(groups[i], set()).add(f)
    assert all(len(fs) == 1 for fs in by_group.values())                  # NUNCA se separa un grupo
    assert folds[0] == folds[1]
    assert folds == REAL_ASSIGN(ids, labels, groups)                      # determinista con la misma versión y entradas
    per_fold, problems, _ = mf.verify(ids, labels, groups, folds)
    assert sum(f["positivas"] for f in per_fold) == 38
    assert not any("más de un fold" in p for p in problems)               # y verify lo comprueba


def test_verify_floor_of_four_positives_is_kept_and_applies_to_any_assigner():
    ids = [str(i) for i in range(100)]
    labels = [1 if i < 12 else 0 for i in range(100)]
    folds = [i % 5 for i in range(100)]                                   # 12 positivas -> 3,3,2,2,2 por fold
    _, problems, _ = mf.verify(ids, labels, {i: i for i in ids}, folds)
    assert sum("solo" in p and "positivas" in p for p in problems) == 5   # los cinco folds quedan por debajo de 4


def test_stable_test_assigner_respects_groups_and_balances_positives():
    ids, labels, groups = _groups_820()
    folds = stable_assign_folds(ids, labels, groups)
    per_fold, problems, _ = mf.verify(ids, labels, groups, folds)
    assert problems == [] and folds == stable_assign_folds(ids, labels, groups)
    assert max(f["positivas"] for f in per_fold) - min(f["positivas"] for f in per_fold) <= 1


def test_fold_with_too_few_positives_is_error_not_warning():
    ids = [str(i) for i in range(100)]
    labels = [1 if i < 10 else 0 for i in range(100)]
    folds = [0 if i < 9 else (i % 4) + 1 for i in range(100)]
    _, problems, _ = mf.verify(ids, labels, {i: i for i in ids}, folds)
    assert any("positivas" in p for p in problems)


def test_large_group_is_never_split_by_the_real_assigner():
    ids = [str(i) for i in range(820)]
    labels = [1 if i < 38 else 0 for i in range(820)]
    groups = mf.build_groups(ids, [(str(i), str(i + 1)) for i in range(0, 11)])   # 12 positivas en un grupo
    folds = REAL_ASSIGN(ids, labels, groups)
    assert len({folds[i] for i in range(12)}) == 1                        # invariante garantizada; no el reparto


def test_range_warning_for_positives_is_a_warning_not_an_error():
    ids, labels, folds = _sized_folds([164] * 5, [12, 7, 7, 6, 6])        # un fold con 12 positivas (fuera de 6-9)
    _, problems, warnings = mf.verify(ids, labels, {i: i for i in ids}, folds)
    assert problems == []
    assert any("12 positivas fuera de [6, 9]" in w for w in warnings)


def test_verify_detects_group_split_across_folds():
    ids, labels = ["1", "2", "3", "4", "5"], [1, 1, 1, 1, 1]
    _, problems, _ = mf.verify(ids, labels, {i: "g" for i in ids}, [0, 1, 0, 1, 0], n_folds=2)
    assert any("más de un fold" in p for p in problems)


def _sized_folds(sizes, positives):
    folds, labels = [], []
    for k, (n, p) in enumerate(zip(sizes, positives)):
        folds += [k] * n
        labels += [1] * p + [0] * (n - p)
    return [str(i) for i in range(len(folds))], labels, folds


def test_size_warning_threshold_is_15_percent_of_164():
    pos = [8, 8, 8, 7, 7]
    ids, labels, folds = _sized_folds([190, 157, 157, 158, 158], pos)   # |190-164|/164 = 15.9 %
    _, problems, warnings = mf.verify(ids, labels, {i: i for i in ids}, folds)
    assert problems == [] and any("fold 0" in w and "tamaño" in w for w in warnings)
    ids, labels, folds = _sized_folds([188, 158, 158, 158, 158], pos)   # 14.6 %: sin advertencia
    _, problems, warnings = mf.verify(ids, labels, {i: i for i in ids}, folds)
    assert problems == [] and not any("tamaño" in w for w in warnings)


# ---------------------------------------------------------------- folds de extremo a extremo
def test_folds_refuse_without_aprobacion(project, capsys):
    annotate(project)
    assert run_folds(project, aprobacion="") == 1 and "--aprobacion" in capsys.readouterr().err
    assert not (project / mf.DEFAULT_OUT_REL).exists()


def test_folds_blocked_when_nothing_reviewed(project, capsys):
    assert run_folds(project) == 1 and "sin revisar" in capsys.readouterr().err
    assert not (project / mf.DEFAULT_OUT_REL).exists()


def test_folds_blocked_by_duda_missing_revisor_and_partial_review(project, capsys, base_info):
    annotate(project, default="duda")
    assert run_folds(project) == 1 and "duda" in capsys.readouterr().err
    annotate(project, skip_revisor=True)
    assert run_folds(project) == 1 and "revisor" in capsys.readouterr().err
    annotate(project)
    p = adir(project) / "revision_ciega.csv"
    header, rows = C.read_csv_with_header(p)
    rows[0]["revision"] = ""
    p.write_text(C.csv_text(rows, header), encoding="utf-8", newline="")
    assert run_folds(project) == 1 and "sin revisar" in capsys.readouterr().err
    assert not (project / mf.DEFAULT_OUT_REL).exists()


def test_folds_blocked_by_missing_files_or_column(project, capsys):
    annotate(project)
    (adir(project) / "calibration_pairs.csv").unlink()
    assert run_folds(project) == 1 and "Faltan archivos" in capsys.readouterr().err
    (adir(project) / "resumen_auditoria.json").unlink()
    assert run_folds(project) == 1 and "resumen_auditoria.json" in capsys.readouterr().err
    assert not (project / mf.DEFAULT_OUT_REL).exists()


def test_folds_blocked_when_revision_column_is_missing(project, capsys):
    p = adir(project) / "revision_ciega.csv"
    header, rows = C.read_csv_with_header(p)
    header = [h for h in header if h != "revision"]
    p.write_text(C.csv_text(rows, header), encoding="utf-8", newline="")
    assert run_folds(project) == 1 and "faltan columnas" in capsys.readouterr().err


def test_folds_blocked_when_correspondence_or_artifacts_are_tampered(project, capsys):
    annotate(project)
    for name in mf.PAIR_FILES:                      # la correspondencia hoja -> pareja es inmutable (hash exacto)
        p = adir(project) / name
        original = p.read_text(encoding="utf-8")
        header, rows = C.read_csv_with_header(p)
        if not rows:
            continue
        p.write_text(C.csv_text(rows[1:], header), encoding="utf-8", newline="")      # se elimina un par
        assert run_folds(project) == 1 and "alterado" in capsys.readouterr().err
        rows[0]["image_id_b"] = rows[0]["image_id_a"]                                  # se cambia una pareja
        p.write_text(C.csv_text(rows, header), encoding="utf-8", newline="")
        assert run_folds(project) == 1 and "alterado" in capsys.readouterr().err
        p.write_text(original, encoding="utf-8", newline="")
    (adir(project) / "development_eligible_v2.csv").write_text("x\n", encoding="utf-8")
    assert run_folds(project) == 1 and "alterado" in capsys.readouterr().err
    assert not (project / mf.DEFAULT_OUT_REL).exists()


def test_folds_generate_with_groups_and_exact_duplicates(project, base_info):
    pick = base_info["pick"]
    annotate(project, misma={near_pair(base_info)})  # el par exacto 0-vs-6 se marca 'distinta' a propósito
    assert run_folds(project) == 0
    out = project / mf.DEFAULT_OUT_REL
    rows = C.read_csv_rows(out / "cv_folds_v2.csv")
    fold = {r["image_id"]: r["fold"] for r in rows}
    assert len(rows) == SMALL_EXPECTED["eligible"] and set(fold.values()) == {"0", "1", "2", "3", "4"}
    a, b = near_pair(base_info)
    assert fold[a] == fold[b]                                              # par `misma`
    assert fold[str(pick(0, 0))] == fold[str(pick(6, 0))]                  # duplicado exacto: unión automática
    assert all(r["label_v2"] == str(C.label_from_class(int(r["original_class"]))) for r in rows)
    s = json.loads((out / "resumen_folds_v2.json").read_text(encoding="utf-8"))
    assert s["modo"] == "agrupado_por_similitud" and s["aristas_por_duplicado_exacto"] == 1
    assert s["aprobacion_parametros"] == "D-TEST" and s["revisiones_distinta_en_grupo_por_duplicado_exacto"] >= 1
    assert s["estado"].startswith("NO VALIDA")      # se usó --permitir-codigo-sin-commit
    assert s["codigo_en_commit"] is False
    assert s["salida_sha256"] == C.sha256_file(out / "cv_folds_v2.csv")
    assert "sklearn" in s["procedencia"]["versiones"] and s["entradas_sha256"]["resumen_auditoria.json"]
    cand = C.read_csv_rows(adir(project) / "candidate_pairs.csv")
    assert s["precision_cribado_entre_candidatos_revisados"] == pytest.approx(1 / len(cand))


def test_folds_refuse_overwrite_without_force_and_keep_previous(project, capsys):
    annotate(project)
    assert run_folds(project) == 0
    out = project / mf.DEFAULT_OUT_REL
    before = (out / "cv_folds_v2.csv").read_bytes()
    assert run_folds(project) == 1 and "--force" in capsys.readouterr().err
    assert (out / "cv_folds_v2.csv").read_bytes() == before
    assert run_folds(project, "--force") == 0
    assert len(list(out.parent.glob("v2.reemplazado-*"))) == 1


def test_folds_failed_validation_writes_nothing(project, monkeypatch, capsys):
    annotate(project)
    monkeypatch.setattr(mf, "validate_staging", lambda *a, **k: (_ for _ in ()).throw(C.GuardError("inválido")))
    assert run_folds(project) == 1 and "inválido" in capsys.readouterr().err
    out = project / mf.DEFAULT_OUT_REL
    assert not out.exists() and not list(out.parent.glob(".*staging*"))


def test_folds_require_versioned_audit_and_code(project, monkeypatch, capsys):
    annotate(project)
    monkeypatch.setattr(C, "_git", fake_git(False))
    assert mf.main(["--aprobacion", "D-T"], project_root=project, expected=SMALL_EXPECTED) == 1
    assert "sin commit" in capsys.readouterr().err
    monkeypatch.setattr(C, "_git", fake_git(True))   # código versionado, pero la auditoría no lo estaba
    assert mf.main(["--aprobacion", "D-T"], project_root=project, expected=SMALL_EXPECTED) == 1
    assert "auditoría se ejecutó con código o fuente sin commit" in capsys.readouterr().err


def test_full_flow_with_versioned_code_marks_proposal(tmp_path, monkeypatch):
    info = make_project(tmp_path, plants=BASE_PLANTS)
    monkeypatch.setattr(C, "_git", fake_git(True))
    assert audit.main([], project_root=tmp_path, expected=SMALL_EXPECTED, expected_total=SMALL_TOTAL) == 0
    s = json.loads((adir(tmp_path) / "resumen_auditoria.json").read_text(encoding="utf-8"))
    assert s["codigo_en_commit"] is True and s["estado"].startswith("Auditoria para revision manual")
    annotate(tmp_path, misma={near_pair(info)})
    assert mf.main(["--aprobacion", "D-T"], project_root=tmp_path, expected=SMALL_EXPECTED) == 0
    f = json.loads((tmp_path / mf.DEFAULT_OUT_REL / "resumen_folds_v2.json").read_text(encoding="utf-8"))
    assert f["estado"].startswith("PROPUESTA") and f["codigo_en_commit"] is True
    assert f["auditoria_commit"] == "deadbeef"


def test_folds_abort_on_misma_pair_with_contradictory_labels(tmp_path, capsys):
    info = make_project(tmp_path, plants=[("near", (0, 2), (7, 0))])
    assert run_audit(tmp_path, "--no-sheets") == 0
    pair = tuple(sorted((str(info["pick"](0, 2)), str(info["pick"](7, 0))), key=int))
    cands = C.read_csv_rows(adir(tmp_path) / "candidate_pairs.csv")
    assert any((r["image_id_a"], r["image_id_b"]) == pair and r["etiquetas_contradictorias"] == "True"
               for r in cands)
    annotate(tmp_path, misma={pair})
    assert run_folds(tmp_path) == 1 and "une etiquetas v2 distintas" in capsys.readouterr().err


def test_folds_abort_on_exact_duplicate_with_contradictory_labels(tmp_path, capsys):
    make_project(tmp_path, plants=[("exact", (0, 0), (7, 0))])
    assert run_audit(tmp_path, "--no-sheets") == 0
    annotate(tmp_path)
    assert run_folds(tmp_path) == 1 and "Duplicado exacto con etiquetas contradictorias" in capsys.readouterr().err
    assert not (tmp_path / mf.DEFAULT_OUT_REL).exists()


# ---------------------------------------------------------------- --out restringido y directorios de agentes
@pytest.mark.parametrize("bad", [".", "configs/splits", "data/processed", "results", "results/r2_v2", "Reportes",
                                 ".claude/agent-memory/x", "../fuera", "results/r2_v2/.claude",
                                 "results/r2_v2/Version-Informes"])
def test_audit_out_must_be_strictly_inside_results_r2_v2(tmp_path, bad, capsys):
    make_project(tmp_path)
    assert run_audit(tmp_path, "--out", bad, "--force") == 1
    assert "--out" in capsys.readouterr().err
    assert (tmp_path / C.SOURCE_REL).is_file() and (tmp_path / C.DATA_REL).is_dir()   # nada se movió
    assert not list(tmp_path.rglob("*.reemplazado-*")) and not list(tmp_path.rglob(".*staging*"))
    for name in (".claude", "Reportes", "Version-Informes", "agent-memory"):
        assert not list(tmp_path.rglob(name))


def test_audit_accepts_nested_out_inside_results_r2_v2(tmp_path):
    make_project(tmp_path)
    assert run_audit(tmp_path, "--out", "results/r2_v2/otra_auditoria", "--no-sheets") == 0
    assert (tmp_path / "results/r2_v2/otra_auditoria/resumen_auditoria.json").is_file()


@pytest.mark.parametrize("bad", ["configs/splits", "configs", ".", "results/r2_v2/auditoria_desarrollo",
                                 ".claude/x", "configs/splits/v2/Reportes"])
def test_folds_out_is_restricted(project, bad, capsys):
    annotate(project)
    assert run_folds(project, "--out", bad, "--force") == 1 and "--out" in capsys.readouterr().err
    assert (adir(project) / "resumen_auditoria.json").is_file() and (project / C.SOURCE_REL).is_file()
    assert not list(project.rglob("*.reemplazado-*"))


def test_folds_out_cannot_contain_the_reviewed_audit_directory(project, capsys):
    annotate(project)
    nested = project / "configs" / "splits" / "v2" / "aud"
    nested.parent.mkdir(parents=True)
    shutil.copytree(adir(project), nested)
    assert run_folds(project, "--audit-dir", str(nested)) == 1
    assert "no puede coincidir" in capsys.readouterr().err
    assert (nested / "candidate_pairs.csv").is_file()


def test_require_safe_out_unit(tmp_path):
    (tmp_path / "configs/splits/v2").mkdir(parents=True)
    C.require_safe_out(tmp_path / "configs/splits/v2", tmp_path, "configs/splits/v2", allow_equal=True)
    with pytest.raises(C.GuardError):
        C.require_safe_out(tmp_path / "configs/splits/v2", tmp_path, "configs/splits/v2", allow_equal=False)
    with pytest.raises(C.GuardError, match="agentes"):
        C.require_safe_out(tmp_path / "configs/splits/v2/.claude", tmp_path, "configs/splits/v2", allow_equal=True)


# ---------------------------------------------------------------- el resumen de auditoría no es raíz de confianza
def _edit_summary(project, fn):
    p = adir(project) / "resumen_auditoria.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    fn(s)
    p.write_text(json.dumps(s), encoding="utf-8")


def test_folds_require_every_artifact_hash_to_be_declared(project, capsys):
    annotate(project)
    _edit_summary(project, lambda s: s["salidas_sha256"].pop("image_hashes.csv"))
    assert run_folds(project) == 1 and "no declara el hash" in capsys.readouterr().err
    assert not (project / mf.DEFAULT_OUT_REL).exists()


def test_folds_require_the_blind_sheet_digest_to_be_declared(project, capsys):
    annotate(project)
    _edit_summary(project, lambda s: s.pop("planilla_ciega"))
    assert run_folds(project) == 1 and "no corresponde uno a uno" in capsys.readouterr().err


def test_folds_detect_a_source_different_from_the_audited_one(project, capsys):
    annotate(project)
    src = project / C.SOURCE_REL
    src.write_text(src.read_text(encoding="utf-8") + "\n", encoding="utf-8", newline="")
    assert run_folds(project) == 1 and "difiere de la fuente" in capsys.readouterr().err


def test_self_declared_codigo_en_commit_is_not_trusted(project, monkeypatch, capsys):
    annotate(project)
    _edit_summary(project, lambda s: s.update(codigo_en_commit=True))   # autodeclarado, pero la procedencia dice que no
    monkeypatch.setattr(C, "_git", fake_git(True))
    assert mf.main(["--aprobacion", "D-T"], project_root=project, expected=SMALL_EXPECTED) == 1
    assert "sin commit" in capsys.readouterr().err


def test_folds_warn_when_audit_code_changed_after_the_audit(tmp_path, monkeypatch, capsys):
    info = make_project(tmp_path, plants=BASE_PLANTS)
    monkeypatch.setattr(C, "_git", fake_git(True, blob="aaa"))
    assert audit.main([], project_root=tmp_path, expected=SMALL_EXPECTED, expected_total=SMALL_TOTAL) == 0
    annotate(tmp_path, misma={near_pair(info)})
    monkeypatch.setattr(C, "_git", fake_git(True, blob="ccc"))          # el código cambió tras la auditoría
    assert mf.main(["--aprobacion", "D-T"], project_root=tmp_path, expected=SMALL_EXPECTED) == 0
    assert "cambió desde que se ejecutó" in capsys.readouterr().out
    s = json.loads((tmp_path / mf.DEFAULT_OUT_REL / "resumen_folds_v2.json").read_text(encoding="utf-8"))
    assert any("cambió desde que se ejecutó" in w for w in s["advertencias"])
    assert all(v["blob"] == "aaa" for v in json.loads(
        (adir(tmp_path) / "resumen_auditoria.json").read_text(encoding="utf-8"))["procedencia"]["archivos_codigo"].values())


# ---------------------------------------------------------------- publicación
def test_review_sheets_gitignore_is_written_before_any_sheet(tmp_path, monkeypatch):
    make_project(tmp_path, plants=BASE_PLANTS)
    seen, real = [], audit.make_review_sheet

    def spy(a, b, out_path, *r, **k):
        seen.append((out_path.parent / ".gitignore").is_file())
        return real(a, b, out_path, *r, **k)
    monkeypatch.setattr(audit, "make_review_sheet", spy)
    assert run_audit(tmp_path) == 0 and seen and all(seen)


def test_destination_is_rechecked_right_before_publishing(tmp_path, monkeypatch, capsys):
    make_project(tmp_path)
    real = audit.validate_staging

    def sneaky(*a, **k):                      # alguien anota un archivo en el destino mientras se calculaba
        out = adir(tmp_path)
        out.mkdir(parents=True)
        (out / "revision_ciega.csv").write_text(
            "hoja,revision,revisor,nota\nreview_sheets/par_0001.png,misma,T,\n", encoding="utf-8")
        return real(*a, **k)
    monkeypatch.setattr(audit, "validate_staging", sneaky)
    assert run_audit(tmp_path, "--force") == 1 and "nunca se sobrescriben" in capsys.readouterr().err
    assert "misma" in (adir(tmp_path) / "revision_ciega.csv").read_text(encoding="utf-8")
    assert not list(adir(tmp_path).parent.glob(".*staging*"))


# ---------------------------------------------------------------- revisión ciega
def test_review_sheet_is_blind_and_correspondence_is_separate(project):
    out = adir(project)
    header, sheet = C.read_csv_with_header(out / "revision_ciega.csv")
    assert header == ["hoja", "revision", "revisor", "nota"]            # exactamente lo que ve y edita el revisor
    assert sheet and all(r["revision"] == r["revisor"] == r["nota"] == "" for r in sheet)
    text = (out / "revision_ciega.csv").read_text(encoding="utf-8")
    assert not any(w in text for w in ("distance", "original_class", "label", "categorias", "contradictorias"))
    key_hojas = []
    for name in audit.KEY_FILES:                                         # la correspondencia vive aparte
        key_header, key_rows = C.read_csv_with_header(out / name)
        assert key_header == audit.KEY_COLUMNS and "distance" in key_header and "label_a" in key_header
        key_hojas += [r["hoja"] for r in key_rows]
    assert len(set(key_hojas)) == len(key_hojas) and sorted(key_hojas) == sorted(r["hoja"] for r in sheet)


def test_no_sheets_run_still_produces_a_consistent_blind_sheet(tmp_path):
    make_project(tmp_path, plants=BASE_PLANTS)
    assert run_audit(tmp_path, "--no-sheets") == 0
    out = adir(tmp_path)
    assert not (out / audit.SHEETS_DIR).exists()
    sheet = C.read_csv_rows(out / "revision_ciega.csv")
    keys = [r["hoja"] for n in audit.KEY_FILES for r in C.read_csv_rows(out / n)]
    assert sheet and sorted(keys) == sorted(r["hoja"] for r in sheet)


def test_validate_staging_rejects_a_sheet_that_reveals_or_is_prefilled(project):
    stage = project / "stage"
    shutil.copytree(adir(project), stage)
    p = stage / "revision_ciega.csv"
    header, rows = C.read_csv_with_header(p)
    for r in rows:
        r["distance"] = "0"
    p.write_text(C.csv_text(rows, header + ["distance"]), encoding="utf-8", newline="")
    with pytest.raises(C.GuardError, match="exactamente"):
        audit.validate_staging(stage, SMALL_EXPECTED, SMALL_TOTAL, with_sheets=True)
    for r in rows:
        r.pop("distance")
    rows[0]["revision"] = "misma"
    p.write_text(C.csv_text(rows, header), encoding="utf-8", newline="")
    with pytest.raises(C.GuardError, match="sin anotaciones"):
        audit.validate_staging(stage, SMALL_EXPECTED, SMALL_TOTAL, with_sheets=True)
    p.write_text(C.csv_text(rows[1:], header), encoding="utf-8", newline="")
    with pytest.raises(C.GuardError, match="uno a uno"):
        audit.validate_staging(stage, SMALL_EXPECTED, SMALL_TOTAL, with_sheets=True)


def test_folds_reject_sheet_with_added_removed_repeated_or_renamed_hojas(project, capsys):
    annotate(project)
    p = adir(project) / "revision_ciega.csv"
    header, rows = C.read_csv_with_header(p)
    original = p.read_text(encoding="utf-8")
    variants = {"quitada": rows[1:],
                "anadida": rows + [{"hoja": "review_sheets/par_9999.png", "revision": "distinta", "revisor": "T", "nota": ""}],
                "repetida": rows + [dict(rows[0])],
                "renombrada": [{**rows[0], "hoja": "review_sheets/par_9998.png"}] + rows[1:]}
    for label, variant in variants.items():
        p.write_text(C.csv_text(variant, header), encoding="utf-8", newline="")
        assert run_folds(project) == 1, label
        assert "no corresponde uno a uno" in capsys.readouterr().err, label
    assert not (project / mf.DEFAULT_OUT_REL).exists()
    p.write_text(original, encoding="utf-8", newline="")                 # restaurada: la integridad vuelve a cumplirse
    assert run_folds(project) == 0


def test_folds_translate_sheet_reviews_through_the_correspondence(project, base_info):
    mapping = pair_of_hoja(project)
    target = near_pair(base_info)
    hoja = next(h for h, pair in mapping.items() if pair == target)
    annotate(project)
    p = adir(project) / "revision_ciega.csv"
    header, rows = C.read_csv_with_header(p)
    next(r for r in rows if r["hoja"] == hoja)["revision"] = "misma"     # el revisor solo ve la hoja
    p.write_text(C.csv_text(rows, header), encoding="utf-8", newline="")
    assert run_folds(project) == 0
    fold = {r["image_id"]: r["fold"] for r in C.read_csv_rows(project / mf.DEFAULT_OUT_REL / "cv_folds_v2.csv")}
    assert fold[target[0]] == fold[target[1]]
    s = json.loads((project / mf.DEFAULT_OUT_REL / "resumen_folds_v2.json").read_text(encoding="utf-8"))
    assert s["aristas_confirmadas_por_revision"] == 1
    assert s["entradas_sha256"]["revision_ciega.csv"] == C.sha256_file(adir(project) / "revision_ciega.csv")


# ---------------------------------------------------------------- generación de folds con el asignador REAL
def test_folds_with_the_real_assigner_either_succeed_validly_or_block_safely(project, base_info, monkeypatch, capsys):
    """Contrato, no distribución: con StratifiedGroupKFold real el resultado depende de la versión de scikit-learn;
    lo que nunca puede ocurrir es una salida publicada que viole el mínimo de positivas o separe un grupo."""
    monkeypatch.setattr(mf, "assign_folds", REAL_ASSIGN)
    annotate(project, misma={near_pair(base_info)})
    rc = run_folds(project)
    out = project / mf.DEFAULT_OUT_REL
    if rc == 0:
        rows = C.read_csv_rows(out / "cv_folds_v2.csv")
        per_fold = {}
        for r in rows:
            per_fold[r["fold"]] = per_fold.get(r["fold"], 0) + int(r["label_v2"])
        assert len(per_fold) == 5 and min(per_fold.values()) >= mf.MIN_POS_HARD
        by_group = {}
        for r in rows:
            by_group.setdefault(r["group_id"], set()).add(r["fold"])
        assert all(len(fs) == 1 for fs in by_group.values())
    else:
        assert rc == 1 and "Partición no viable" in capsys.readouterr().err
        assert not out.exists() and not list(out.parent.glob(".*staging*"))   # bloqueo seguro: no se publica nada


def test_folds_block_deterministically_when_a_fold_has_fewer_than_four_positives(project, base_info, monkeypatch, capsys):
    """El bloqueo 'Partición no viable' se prueba sin depender de scikit-learn: un asignador simulado deja un fold con 2 positivas."""
    def too_few_positives(ids, labels, groups, n_folds=5, seed=42):
        folds = stable_assign_folds(ids, labels, groups, n_folds, seed)
        positives_in_0 = [k for k, (f, y) in enumerate(zip(folds, labels)) if f == 0 and y == 1]
        for k in positives_in_0[2:]:                   # las positivas son singletons: mover una no separa ningún grupo
            folds[k] = 1
        return folds
    monkeypatch.setattr(mf, "assign_folds", too_few_positives)
    annotate(project, misma={near_pair(base_info)})
    assert run_folds(project) == 1
    err = capsys.readouterr().err
    assert "Partición no viable" in err and "fold 0: solo 2 positivas (mínimo 4)" in err
    out = project / mf.DEFAULT_OUT_REL
    assert not out.exists() and not list(out.parent.glob(".*staging*"))   # no se publica nada
