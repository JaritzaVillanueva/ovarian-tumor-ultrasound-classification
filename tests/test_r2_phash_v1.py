"""Pruebas SINTÉTICAS del agrupamiento automático por pHash <= 8 (phash_le8_v1) y su asignador determinista.

No abren imágenes reales ni el test oficial. Son independientes del estado de Git (`_git` simulado por defecto) y de
scikit-learn (el procedimiento no lo INVOCA; una prueba lo comprueba; importarlo sí requiere tenerlo instalado). Los datos observados en el diagnóstico exploratorio
(69 pares, grupo máximo 4, 0 mixtos) NO se usan como criterios de aceptación en ninguna prueba.
"""

import builtins
import io
import json
import random
import shutil
from pathlib import Path

import pytest
from PIL import Image

from src.r2_v2 import common as C
from src.r2_v2 import make_folds_phash_v1 as ph
from src.r2_v2 import make_folds_v2 as mf
from tests.test_r2_v2 import BASE_PLANTS, SMALL_EXPECTED, SMALL_TOTAL, adir, fake_git, make_project, run_audit


@pytest.fixture(autouse=True)
def _isolated_git(monkeypatch):
    monkeypatch.setattr(C, "_git", fake_git(False))


def pin(root):
    """SHA-256 del resumen de la auditoría tal como está AHORA (lo que el autor anotaría de la trazabilidad)."""
    return C.sha256_file(adir(root) / "resumen_auditoria.json")


def run_ph(root, *extra, aprobacion="D-026 prueba", pinned="auto"):
    args = ["--permitir-codigo-sin-commit", *extra]
    if aprobacion:
        args += ["--aprobacion", aprobacion]
    if pinned == "auto":
        pinned = pin(root)
    if pinned:
        args += ["--resumen-sha256", pinned]
    return ph.main(args, project_root=root, expected=SMALL_EXPECTED)


def out_dir(root):
    return Path(root) / ph.DEFAULT_OUT_REL


def snapshot(directory):
    return {p.relative_to(directory).as_posix(): C.sha256_file(p) for p in sorted(Path(directory).rglob("*")) if p.is_file()}


def edit_summary(root, fn):
    p = adir(root) / "resumen_auditoria.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    fn(s)
    p.write_text(json.dumps(s), encoding="utf-8")


def refresh_declared_hash(root, name):
    """Simula artefactos y resumen FALSIFICADOS de forma coherente entre sí (el hash declarado se actualiza)."""
    edit_summary(root, lambda s: s["salidas_sha256"].__setitem__(name, C.sha256_file(adir(root) / name)))


@pytest.fixture(scope="module")
def base_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("phash_base")
    make_project(root, plants=BASE_PLANTS)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(C, "_git", fake_git(False))
        assert run_audit(root, "--no-sheets") == 0
    return root


@pytest.fixture
def project(base_root, tmp_path):
    shutil.copytree(base_root, tmp_path / "p")
    return tmp_path / "p"


# ---------------------------------------------------------------- asignador determinista (unidad)
def _scenario(seed, k_pairs, n=820, npos=38):
    rng = random.Random(seed)
    ids = [str(i) for i in range(n)]
    pos = set(rng.sample(range(n), npos))
    labels = [1 if i in pos else 0 for i in range(n)]
    used, edges = set(), []
    while len(edges) < k_pairs:
        a, b = rng.sample(range(n), 2)
        if a in used or b in used:
            continue
        used |= {a, b}
        edges.append((str(a), str(b)))
    return ids, labels, edges


def test_assigner_is_deterministic_and_independent_of_input_order():
    ids, labels, edges = _scenario(1, 60)
    groups = mf.build_groups(ids, edges)
    folds = ph.assign_folds_deterministic(ids, labels, groups)
    assert folds == ph.assign_folds_deterministic(ids, labels, groups)
    order = list(range(len(ids)))
    random.Random(5).shuffle(order)
    sids, slabels = [ids[i] for i in order], [labels[i] for i in order]
    sfolds = ph.assign_folds_deterministic(sids, slabels, groups)
    assert {i: f for i, f in zip(sids, sfolds)} == dict(zip(ids, folds))


def test_assigner_never_splits_groups_and_keeps_chains_together():
    ids, labels, edges = _scenario(2, 100)
    chain = [("700", "701"), ("701", "702"), ("702", "703")]            # cadena transitiva
    groups = mf.build_groups(ids, [e for e in edges if not {e[0], e[1]} & {"700", "701", "702", "703"}] + chain)
    folds = ph.assign_folds_deterministic(ids, labels, groups)
    fold_of = dict(zip(ids, folds))
    assert len({fold_of[i] for i in ("700", "701", "702", "703")}) == 1
    by_group = {}
    for i in ids:
        by_group.setdefault(groups[i], set()).add(fold_of[i])
    assert all(len(fs) == 1 for fs in by_group.values())
    assert set(folds) == {0, 1, 2, 3, 4}


def test_assigner_reaches_the_floor_on_synthetic_scenarios_of_the_real_shape():
    for t in range(40):
        ids, labels, edges = _scenario(1000 + t, random.Random(t).choice([0, 10, 30, 60, 100, 150]))
        groups = mf.build_groups(ids, edges)
        folds = ph.assign_folds_deterministic(ids, labels, groups)
        per_fold, problems, _ = mf.verify(ids, labels, groups, folds)
        assert not any("positivas" in p and "mínimo" in p for p in problems), (t, per_fold)
        assert sum(f["positivas"] for f in per_fold) == 38


def test_close_pairs_and_cross_check_are_exact():
    phash_of = {"1": 0b0000, "2": 0b0001, "3": 0b0111, "4": 0xFFFF}
    pairs = ph.close_pairs(["1", "2", "3", "4"], phash_of, 3)
    assert {(a, b): d for a, b, d in pairs} == {("1", "2"): 1, ("1", "3"): 3, ("2", "3"): 2}
    cand = [{"image_id_a": a, "image_id_b": b, "distance": str(d)} for a, b, d in pairs]
    ph.cross_check_with_audit(pairs, cand, [], 3, phash_of)
    with pytest.raises(C.GuardError, match="NO reproducen"):
        ph.cross_check_with_audit(pairs, cand[1:], [], 3, phash_of)
    with pytest.raises(C.GuardError, match="NO reproducen"):
        ph.cross_check_with_audit(pairs, [{**cand[0], "distance": "2"}] + cand[1:], [], 3, phash_of)


def test_residual_risk_counts():
    fold_of = {"1": 0, "2": 0, "3": 1, "4": 2}
    assert ph.residual_risk_counts([("1", "2"), ("1", "3"), ("3", "4")], fold_of) == {"parejas": 3, "en_folds_distintos": 2}


# ---------------------------------------------------------------- ejecución de extremo a extremo
def test_generates_folds_with_groups_intact_and_documented_limits(project):
    before = snapshot(adir(project))
    assert run_ph(project) == 0
    out = out_dir(project)
    header, rows = C.read_csv_with_header(out / ph.FOLDS_FILE)
    assert len(rows) == SMALL_EXPECTED["eligible"] and header[-3:] == ["group_id", "group_size", "fold"]
    assert not any(c in header for c in ("distance", "phash_hex", "sha256"))
    fold_of = {r["image_id"]: r["fold"] for r in rows}
    edges = C.read_csv_rows(out / ph.EDGES_FILE)
    assert all(fold_of[e["image_id_a"]] == fold_of[e["image_id_b"]] for e in edges)       # aristas nunca separadas
    assert all(int(e["distance"]) <= ph.PHASH_THRESHOLD for e in edges)
    per_fold = {}
    for r in rows:
        per_fold[r["fold"]] = per_fold.get(r["fold"], 0) + int(r["label_v2"])
    assert len(per_fold) == 5 and min(per_fold.values()) >= mf.MIN_POS_HARD               # el mínimo de 4 se conserva
    s = json.loads((out / ph.SUMMARY_FILE).read_text(encoding="utf-8"))
    assert s["version_protocolo"] == "phash_le8_v1" and s["estado"].startswith("NO VALIDA")  # se usó el flag de pruebas
    assert s["criterio"] == {"tipo": s["criterio"]["tipo"], "umbral_phash": 8, "exploratorio": True, "post_hoc": True,
                             "uso_de_etiquetas_en_la_seleccion": True, "validado_externamente": False,
                             "equivale_a_separacion_por_paciente": False}
    assert s["validacion_humana"]["realizada"] is False and s["asignador"].startswith("determinista")
    assert any("paciente" in t for t in s["limitaciones"]) and any("Riesgo residual" in t for t in s["limitaciones"])
    assert s["aprobacion_parametros"] == "D-026 prueba" and s["agrupamiento_observado"]["grupos_mixtos"] == 0
    assert s["salidas_sha256"][ph.FOLDS_FILE] == C.sha256_file(out / ph.FOLDS_FILE)
    assert snapshot(adir(project)) == before                                              # la auditoría queda byte a byte intacta


def test_exact_duplicates_and_near_duplicates_share_a_fold(project, base_root):
    assert run_ph(project) == 0
    fold_of = {r["image_id"]: r["fold"] for r in C.read_csv_rows(out_dir(project) / ph.FOLDS_FILE)}
    cands = C.read_csv_rows(adir(project) / "candidate_pairs.csv")
    close = [(r["image_id_a"], r["image_id_b"]) for r in cands if int(r["distance"]) <= ph.PHASH_THRESHOLD]
    assert close, "la fixture sintética debe tener al menos una pareja <= 8 (duplicado exacto con distancia 0)"
    assert all(fold_of[a] == fold_of[b] for a, b in close)


def test_diagnostic_of_residual_risk_matches_the_audit_pairs(project):
    assert run_ph(project) == 0
    out = out_dir(project)
    s = json.loads((out / ph.SUMMARY_FILE).read_text(encoding="utf-8"))
    fold_of = {r["image_id"]: r["fold"] for r in C.read_csv_rows(out / ph.FOLDS_FILE)}
    cands = C.read_csv_rows(adir(project) / "candidate_pairs.csv")
    between = [(r["image_id_a"], r["image_id_b"]) for r in cands if int(r["distance"]) > ph.PHASH_THRESHOLD]
    calib = [(r["image_id_a"], r["image_id_b"]) for r in C.read_csv_rows(adir(project) / "calibration_pairs.csv")]
    d = s["diagnostico_riesgo_residual"]
    assert d["parejas_entre_umbral_y_corte_de_cribado"] == ph.residual_risk_counts(between, fold_of)
    assert d["parejas_de_calibracion_fuera_del_corte"] == ph.residual_risk_counts(calib, fold_of)


def test_observed_diagnostic_values_are_not_acceptance_criteria(tmp_path):
    """Sin parejas plantadas (0 aristas) y con una cadena de 6 casi duplicados (grupo de 6): ambos se aceptan."""
    make_project(tmp_path)
    assert run_audit(tmp_path, "--no-sheets") == 0
    assert run_ph(tmp_path) == 0
    s = json.loads((out_dir(tmp_path) / ph.SUMMARY_FILE).read_text(encoding="utf-8"))
    assert s["agrupamiento_observado"]["aristas_phash"] == 0 and s["agrupamiento_observado"]["tamano_maximo_de_grupo"] == 1
    chain = tmp_path / "chain"
    make_project(chain, plants=[("near", (1, k), (1, k + 1)) for k in range(5)])      # 1,0 -> 1,1 -> ... -> 1,5
    assert run_audit(chain, "--no-sheets") == 0
    assert run_ph(chain) == 0
    s = json.loads((out_dir(chain) / ph.SUMMARY_FILE).read_text(encoding="utf-8"))
    assert s["agrupamiento_observado"]["tamano_maximo_de_grupo"] >= 5                  # un grupo mayor que 4 no bloquea


def test_two_runs_produce_identical_outputs(project):
    assert run_ph(project, "--out", "configs/splits/v2_auto/a") == 0
    assert run_ph(project, "--out", "configs/splits/v2_auto/b") == 0
    a, b = project / "configs/splits/v2_auto/a", project / "configs/splits/v2_auto/b"
    assert (a / ph.FOLDS_FILE).read_bytes() == (b / ph.FOLDS_FILE).read_bytes()
    assert (a / ph.EDGES_FILE).read_bytes() == (b / ph.EDGES_FILE).read_bytes()


def test_runs_without_sklearn_images_or_the_blind_sheet(project, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no debe usar scikit-learn ni abrir imágenes")
    monkeypatch.setattr(mf, "StratifiedGroupKFold", boom)
    monkeypatch.setattr(Image, "open", boom)
    opened, real_open = [], io.open

    def spy(file, *a, **k):
        opened.append(str(file).replace("\\", "/"))
        return real_open(file, *a, **k)
    monkeypatch.setattr(io, "open", spy)
    monkeypatch.setattr(builtins, "open", spy)
    assert run_ph(project) == 0
    assert opened and not any("revision_ciega" in p or "review_sheets" in p or "/test/" in p.lower() for p in opened)
    assert not any(p.lower().endswith((".jpg", ".png")) for p in opened)


def test_input_files_never_include_the_blind_sheet():
    assert "revision_ciega.csv" not in ph.INPUT_FILES and "resumen_auditoria.json" not in ph.INPUT_FILES


# ---------------------------------------------------------------- bloqueos
def test_mixed_group_stops_generation_and_nothing_is_divided_or_corrected(tmp_path, capsys):
    info = make_project(tmp_path, plants=[("near", (0, 2), (7, 0))])         # casi duplicado entre una no maligna y una maligna
    assert run_audit(tmp_path, "--no-sheets") == 0
    a, b = str(info["pick"](0, 2)), str(info["pick"](7, 0))
    pair = {tuple(sorted((r["image_id_a"], r["image_id_b"]), key=int)): int(r["distance"])
            for r in C.read_csv_rows(adir(tmp_path) / "candidate_pairs.csv")}
    assert pair[tuple(sorted((a, b), key=int))] <= ph.PHASH_THRESHOLD       # precondición de la prueba
    before = snapshot(adir(tmp_path))
    assert run_ph(tmp_path) == 1
    err = capsys.readouterr().err
    assert "etiquetas mixtas" in err and a in err and b in err and "No se divide ningún grupo" in err
    assert not out_dir(tmp_path).exists() and not list(out_dir(tmp_path).parent.glob(".*staging*"))
    assert snapshot(adir(tmp_path)) == before


def test_exact_duplicate_with_contradictory_labels_stops(tmp_path, capsys):
    make_project(tmp_path, plants=[("exact", (0, 0), (7, 0))])
    assert run_audit(tmp_path, "--no-sheets") == 0
    assert run_ph(tmp_path) == 1 and "Duplicado exacto con etiquetas contradictorias" in capsys.readouterr().err
    assert not out_dir(tmp_path).exists()


def test_floor_of_four_positives_blocks_publication(project, monkeypatch, capsys):
    real = ph.assign_folds_deterministic

    def too_few(ids, labels, groups, n_folds=5):
        folds = real(ids, labels, groups, n_folds)
        zero = [k for k, (f, y) in enumerate(zip(folds, labels)) if f == 0 and y == 1]
        for k in zero[2:]:                      # las positivas del proyecto sintético son singletons: no separa grupos
            folds[k] = 1
        return folds
    monkeypatch.setattr(ph, "assign_folds_deterministic", too_few)
    assert run_ph(project) == 1
    err = capsys.readouterr().err
    assert "Partición no viable" in err and "fold 0: solo 2 positivas (mínimo 4)" in err
    assert not out_dir(project).exists() and not list(out_dir(project).parent.glob(".*staging*"))


def test_requires_approval_reference(project, capsys):
    assert run_ph(project, aprobacion="") == 1 and "--aprobacion" in capsys.readouterr().err
    assert not out_dir(project).exists()


@pytest.mark.parametrize("name", ph.INPUT_FILES)
def test_tampered_audit_inputs_are_blocked(project, capsys, name):
    with open(adir(project) / name, "a", encoding="utf-8") as f:
        f.write("\n")
    assert run_ph(project) == 1 and "alterado" in capsys.readouterr().err
    assert not out_dir(project).exists()


def test_summary_must_declare_every_input_hash(project, capsys):
    edit_summary(project, lambda s: s["salidas_sha256"].pop("image_hashes.csv"))
    assert run_ph(project) == 1 and "no declara el hash" in capsys.readouterr().err


def test_source_different_from_the_audited_one_is_blocked(project, capsys):
    src = project / C.SOURCE_REL
    src.write_text(src.read_text(encoding="utf-8") + "\n", encoding="utf-8", newline="")
    assert run_ph(project) == 1 and "difiere de la fuente" in capsys.readouterr().err


def test_forged_but_self_consistent_candidate_pairs_are_detected_by_recomputation(project, capsys):
    p = adir(project) / "candidate_pairs.csv"
    header, rows = C.read_csv_with_header(p)
    rows[0]["distance"] = str(int(rows[0]["distance"]) + 1)
    p.write_text(C.csv_text(rows, header), encoding="utf-8", newline="")
    refresh_declared_hash(project, "candidate_pairs.csv")                 # el hash declarado coincide: solo la recomputación lo ve
    assert run_ph(project) == 1 and "NO reproducen candidate_pairs.csv" in capsys.readouterr().err
    assert not out_dir(project).exists()


def test_forged_calibration_pairs_are_detected_by_recomputation(project, capsys):
    p = adir(project) / "calibration_pairs.csv"
    header, rows = C.read_csv_with_header(p)
    ids = [r["image_id"] for r in C.read_csv_rows(adir(project) / "development_eligible_v2.csv")][:2]
    rows.append({**{h: "" for h in header}, "image_id_a": ids[0], "image_id_b": ids[1], "distance": "13"})  # distancia inventada
    p.write_text(C.csv_text(rows, header), encoding="utf-8", newline="")
    refresh_declared_hash(project, "calibration_pairs.csv")
    assert run_ph(project) == 1 and "calibration_pairs.csv" in capsys.readouterr().err


def test_forged_image_hashes_are_detected(project, capsys):
    p = adir(project) / "image_hashes.csv"
    header, rows = C.read_csv_with_header(p)
    rows[0]["phash_hex"] = "zz"
    p.write_text(C.csv_text(rows, header), encoding="utf-8", newline="")
    refresh_declared_hash(project, "image_hashes.csv")
    assert run_ph(project) == 1 and "phash_hex" in capsys.readouterr().err


def test_wrong_audit_counts_are_blocked(project, capsys):
    bad = dict(SMALL_EXPECTED, malignant=24)
    assert ph.main(["--permitir-codigo-sin-commit", "--aprobacion", "x", "--resumen-sha256", pin(project)],
                   project_root=project, expected=bad) == 1
    assert "Recuentos de la auditoría" in capsys.readouterr().err


def test_unversioned_code_or_protocol_blocks_the_run(project, monkeypatch, capsys):
    assert ph.main(["--aprobacion", "D-026", "--resumen-sha256", pin(project)], project_root=project,
                   expected=SMALL_EXPECTED) == 1     # Git simulado: sin commit
    assert "sin commit" in capsys.readouterr().err
    base = fake_git(True)

    def protocol_modified(root, *args):
        return "zzz" if args[0] == "hash-object" and args[-1] == ph.PROTOCOL_REL else base(root, *args)
    monkeypatch.setattr(C, "_git", protocol_modified)
    assert ph.main(["--aprobacion", "D-026", "--resumen-sha256", pin(project)], project_root=project,
                   expected=SMALL_EXPECTED) == 1
    assert "protocolo" in capsys.readouterr().err and not out_dir(project).exists()


def test_uncommitted_audit_is_not_valid_evidence(project, monkeypatch, capsys):
    monkeypatch.setattr(C, "_git", fake_git(True))                         # el código actual SÍ está en commit...
    assert ph.main(["--aprobacion", "D-026", "--resumen-sha256", pin(project)], project_root=project,
                   expected=SMALL_EXPECTED) == 1
    assert "La auditoría se ejecutó con código o fuente sin commit" in capsys.readouterr().err   # ...pero la auditoría no


def test_full_flow_with_versioned_code_marks_a_provisional_proposal(tmp_path, monkeypatch):
    make_project(tmp_path, plants=BASE_PLANTS)
    monkeypatch.setattr(C, "_git", fake_git(True))
    from src.r2_v2 import build_development_audit as audit
    assert audit.main(["--no-sheets"], project_root=tmp_path, expected=SMALL_EXPECTED, expected_total=SMALL_TOTAL) == 0
    assert ph.main(["--aprobacion", "D-026 2026-11-21", "--resumen-sha256", pin(tmp_path)], project_root=tmp_path,
                   expected=SMALL_EXPECTED) == 0
    s = json.loads((out_dir(tmp_path) / ph.SUMMARY_FILE).read_text(encoding="utf-8"))
    assert s["estado"].startswith("PROPUESTA PROVISIONAL") and s["codigo_en_commit"] is True
    assert s["protocolo"]["en_commit"] is True and s["protocolo"]["archivo"] == ph.PROTOCOL_REL
    assert s["protocolo"]["sha256"] == C.sha256_file(C.PROJECT_ROOT / ph.PROTOCOL_REL)


# ---------------------------------------------------------------- salidas: protección y validación previa a publicar
@pytest.mark.parametrize("bad", ["configs/splits", "configs", ".", "results/r2_v2/auditoria_desarrollo",
                                 ".claude/x", "configs/splits/v2_auto/Reportes", "configs/splits/v2"])
def test_out_is_restricted(project, bad, capsys):
    assert run_ph(project, "--out", bad, "--force") == 1 and "--out" in capsys.readouterr().err
    assert (project / C.SOURCE_REL).is_file() and not list(project.rglob("*.reemplazado-*"))


def test_refuses_overwrite_without_force_and_keeps_previous(project, capsys):
    assert run_ph(project) == 0
    before = (out_dir(project) / ph.FOLDS_FILE).read_bytes()
    assert run_ph(project) == 1 and "--force" in capsys.readouterr().err
    assert (out_dir(project) / ph.FOLDS_FILE).read_bytes() == before
    assert run_ph(project, "--force") == 0 and len(list(out_dir(project).parent.glob("v2_auto.reemplazado-*"))) == 1


def test_failed_validation_publishes_nothing(project, monkeypatch, capsys):
    monkeypatch.setattr(ph, "validate_staging", lambda *a, **k: (_ for _ in ()).throw(C.GuardError("inválido")))
    assert run_ph(project) == 1 and "inválido" in capsys.readouterr().err
    assert not out_dir(project).exists() and not list(out_dir(project).parent.glob(".*staging*"))


def test_validate_staging_detects_tampering(project):
    assert run_ph(project) == 0
    eligible = C.read_csv_rows(adir(project) / "development_eligible_v2.csv")
    stage = project / "stage"
    shutil.copytree(out_dir(project), stage)
    assert ph.validate_staging(stage, eligible, SMALL_EXPECTED)
    header, rows = C.read_csv_with_header(stage / ph.FOLDS_FILE)
    edges = C.read_csv_rows(stage / ph.EDGES_FILE)
    multi = [r for r in rows if int(r["group_size"]) > 1]
    assert multi
    other = str((int(multi[0]["fold"]) + 1) % 5)
    moved = [dict(r, fold=other) if r["image_id"] == multi[0]["image_id"] else r for r in rows]    # separa un grupo
    (stage / ph.FOLDS_FILE).write_text(C.csv_text(moved, header), encoding="utf-8", newline="")
    with pytest.raises(C.GuardError):
        ph.validate_staging(stage, eligible, SMALL_EXPECTED)
    (stage / ph.FOLDS_FILE).write_text(C.csv_text(rows, header), encoding="utf-8", newline="")
    (stage / ph.EDGES_FILE).write_text(C.csv_text(edges[1:], ph.EDGE_COLUMNS), encoding="utf-8", newline="")   # arista de menos
    with pytest.raises(C.GuardError, match="grupos no corresponden"):
        ph.validate_staging(stage, eligible, SMALL_EXPECTED)


# ---------------------------------------------------------------- raíz de confianza anclada y recuentos recalculados
def test_pinned_summary_hash_is_required_and_must_match(project, capsys):
    assert run_ph(project, pinned="") == 1 and "Falta --resumen-sha256" in capsys.readouterr().err
    assert run_ph(project, pinned="0" * 64) == 1 and "no coincide con el anclado" in capsys.readouterr().err
    anchored = pin(project)                                      # el hash que el autor anotó de la auditoría real
    edit_summary(project, lambda s: s.__setitem__("estado", "falsificado"))   # resumen alterado DESPUÉS de anotarlo
    assert run_ph(project, pinned=anchored) == 1 and "no coincide con el anclado" in capsys.readouterr().err
    assert not out_dir(project).exists()


def test_the_anchored_hash_is_recorded_in_the_summary(project):
    assert run_ph(project) == 0
    s = json.loads((out_dir(project) / ph.SUMMARY_FILE).read_text(encoding="utf-8"))
    assert s["resumen_auditoria_sha256_anclado"] == pin(project) == s["entradas_sha256"]["resumen_auditoria.json"]


def test_eligible_counts_are_recomputed_from_the_rows(project, capsys):
    p = adir(project) / "development_eligible_v2.csv"
    header, rows = C.read_csv_with_header(p)
    p.write_text(C.csv_text(rows[:-1], header), encoding="utf-8", newline="")          # una fila menos
    refresh_declared_hash(project, "development_eligible_v2.csv")
    assert run_ph(project) == 1 and "Recuentos recalculados" in capsys.readouterr().err


def test_eligible_rows_must_match_the_development_source(project, capsys):
    p = adir(project) / "development_eligible_v2.csv"
    header, rows = C.read_csv_with_header(p)
    row = next(r for r in rows if r["original_class"] == "0")
    row["original_class"] = "2"                                  # misma etiqueta binaria (0): solo la fuente lo delata
    p.write_text(C.csv_text(rows, header), encoding="utf-8", newline="")
    refresh_declared_hash(project, "development_eligible_v2.csv")
    assert run_ph(project) == 1 and "no coincide con la fuente" in capsys.readouterr().err


def test_audit_cutoff_must_cover_the_threshold(project, capsys):
    edit_summary(project, lambda s: s["phash"].__setitem__("corte_cribado", 6))
    assert run_ph(project) == 1 and "no cubre el umbral" in capsys.readouterr().err


def test_missing_or_inconsistent_image_hash_rows_are_blocked(project, capsys):
    p = adir(project) / "image_hashes.csv"
    header, rows = C.read_csv_with_header(p)
    eligible_id = C.read_csv_rows(adir(project) / "development_eligible_v2.csv")[0]["image_id"]
    p.write_text(C.csv_text([r for r in rows if r["image_id"] != eligible_id], header), encoding="utf-8", newline="")
    refresh_declared_hash(project, "image_hashes.csv")
    assert run_ph(project) == 1 and "sin fila en image_hashes.csv" in capsys.readouterr().err


def test_a_pair_with_a_non_eligible_image_is_a_guard_error_not_a_crash(project, capsys):
    p = adir(project) / "calibration_pairs.csv"
    header, rows = C.read_csv_with_header(p)
    rows.append({**{h: "" for h in header}, "image_id_a": "99998", "image_id_b": "99999", "distance": "20"})
    p.write_text(C.csv_text(rows, header), encoding="utf-8", newline="")
    refresh_declared_hash(project, "calibration_pairs.csv")
    assert run_ph(project) == 1 and "con imagen no elegible" in capsys.readouterr().err


def test_state_is_not_a_valid_proposal_when_only_the_protocol_is_unversioned(tmp_path, monkeypatch):
    make_project(tmp_path, plants=BASE_PLANTS)
    monkeypatch.setattr(C, "_git", fake_git(True))
    from src.r2_v2 import build_development_audit as audit
    assert audit.main(["--no-sheets"], project_root=tmp_path, expected=SMALL_EXPECTED, expected_total=SMALL_TOTAL) == 0
    base = fake_git(True)
    monkeypatch.setattr(C, "_git", lambda root, *a: "zzz" if a[0] == "hash-object" and a[-1] == ph.PROTOCOL_REL
                        else base(root, *a))
    assert run_ph(tmp_path) == 0                                     # con --permitir-codigo-sin-commit
    s = json.loads((out_dir(tmp_path) / ph.SUMMARY_FILE).read_text(encoding="utf-8"))
    assert s["estado"].startswith("NO VALIDA") and s["protocolo"]["en_commit"] is False


# ---------------------------------------------------------------- asignador: casos límite
def test_assigner_edge_cases():
    ids = [str(i) for i in range(30)]
    one_group = mf.build_groups(ids, [(ids[i], ids[i + 1]) for i in range(29)])               # un solo grupo
    labels = [1 if i < 6 else 0 for i in range(30)]
    folds = ph.assign_folds_deterministic(ids, labels, one_group)
    assert len(set(folds)) == 1                                                                # nunca se separa
    no_edges = mf.build_groups(ids, [])
    folds = ph.assign_folds_deterministic(ids, labels, no_edges)
    assert sorted(folds.count(k) for k in range(5)) == [6, 6, 6, 6, 6]                         # equilibra tamaños
    multi = mf.build_groups(ids, [("0", "1"), ("1", "2"), ("3", "4")])                         # grupos con varias positivas
    folds = ph.assign_folds_deterministic(ids, labels, multi)
    assert len({folds[0], folds[1], folds[2]}) == 1 and folds[3] == folds[4]
    big = [str(i) for i in range(400)]
    big_labels = [1 if i < 20 else 0 for i in range(400)]
    big_groups = mf.build_groups(big, [(str(i), str(i + 1)) for i in range(100, 299)])         # grupo negativo de 200
    folds = ph.assign_folds_deterministic(big, big_labels, big_groups)
    assert len({folds[i] for i in range(100, 300)}) == 1
    assert [folds[:20].count(k) for k in range(5)] == [4, 4, 4, 4, 4]                          # positivas equilibradas


def test_assigner_has_a_fixed_known_result():
    """Resultado fijo (regresión): grupos de tamaños y positivas conocidos."""
    ids = [str(i) for i in range(1, 11)]
    labels = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    groups = mf.build_groups(ids, [("1", "2"), ("4", "5"), ("5", "6")])
    # {1,2} (2 pos) -> fold 0; {3} (1 pos) -> fold 1; {4,5,6} -> fold 2 (tamaño 0, menor índice); luego 7 -> 3, 8 -> 4, 9 -> 1, 10 -> 3
    assert ph.assign_folds_deterministic(ids, labels, groups) == [0, 0, 1, 2, 2, 2, 3, 4, 1, 3]
