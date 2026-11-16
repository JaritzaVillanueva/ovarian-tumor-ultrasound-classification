"""
Pruebas del análisis de la fase B (src/classification/agente_fase_b_analysis.py). Solo biblioteca estándar.

Las 45 corridas nuevas (α = 0, 0.25 y 0.75) y las 10 condicionales se SIMULAN en una carpeta temporal (datos de prueba, no
resultados). Las 30 reutilizadas de B0 y B2 son las corridas reales (robustez_e2 y confirmación), copiadas al árbol
temporal; las pruebas que las necesitan se omiten si no están en este entorno. No se lee ninguna corrida real de α ≠ 1.0 y 0.5.
"""
from __future__ import annotations

import json
import random
import shutil
import statistics
from pathlib import Path

import pytest

from src.classification import agente_confirmacion_analysis as cf
from src.classification import agente_etapa1_analysis as a1
from src.classification import agente_fase_b_analysis as fb
from src.classification import robustez_e2_analysis as e2a
from src.classification.cv_folds_e2 import OUTPUT_REL, read_csv_rows
from tests.test_agente_confirmacion_analysis import _patch_confirmation_config
from tests.test_agente_etapa1_analysis import _write_synthetic_run

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE_B_PROTOCOL = PROJECT_ROOT / fb.PHASE_B_REL / fb.PROTOCOL_NAME
E2_DIR = PROJECT_ROOT / e2a.E2_REL
CONFIRMATION_DIR = PROJECT_ROOT / cf.CONFIRMATION_REL
HAVE_PROTOCOL = PHASE_B_PROTOCOL.exists()
HAVE_REUSED = (HAVE_PROTOCOL and (E2_DIR / "corridas" / "convnext_a_alpha05" / "fold_4" / "validation_metrics.json").exists()
               and (CONFIRMATION_DIR / "b2" / "seed_44" / "fold_4" / "validation_metrics.json").exists()
               and (PROJECT_ROOT / fb.FINGERPRINT_REL).exists())
skip_without_protocol = pytest.mark.skipif(not HAVE_PROTOCOL, reason="falta protocolo_fase_b.json (git pull)")
skip_without_reused = pytest.mark.skipif(not HAVE_REUSED, reason="faltan las corridas reutilizadas de B0 y B2 o la huella (git pull)")


def _protocol():
    return json.loads(PHASE_B_PROTOCOL.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------------------
# Árbol sintético: protocolos y corridas reutilizadas reales; corridas nuevas y condicionales sintéticas
# ---------------------------------------------------------------------------------------------------------
def _write_run(root, run, folds_rows, quality, rng, compared_keys, b0_config_by_fold):
    _write_synthetic_run(root, run, folds_rows, quality, rng)
    folder = Path(root) / run["dir_rel"]
    _patch_confirmation_config(folder, run)
    path = folder / "run_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in compared_keys:  # lo que run_experiment.py registra igual que en B0
        config[key] = b0_config_by_fold[run["fold"]][key]
    path.write_text(json.dumps(config), encoding="utf-8")


def build_tree(root, qualities, fingerprint="real", conditional_quality=None, protocol_edit=None):
    """
    qualities: α -> señal de las corridas nuevas. fingerprint: 'real' (copia el veredicto versionado), 'falla' o None (sin archivo).
    conditional_quality: si no es None, se simulan también las 10 condicionales con esa señal.
    """
    root = Path(root)
    protocol = _protocol()
    if protocol_edit:
        protocol_edit(protocol)
    (root / fb.PHASE_B_REL / "verificacion_equivalencia").mkdir(parents=True)
    (root / fb.PHASE_B_REL / fb.PROTOCOL_NAME).write_text(json.dumps(protocol), encoding="utf-8")
    shutil.copy(PROJECT_ROOT / a1.AGENT_REL / a1.PROTOCOL_NAME, root / a1.AGENT_REL / a1.PROTOCOL_NAME)
    if fingerprint == "real":
        shutil.copy(PROJECT_ROOT / fb.FINGERPRINT_REL, root / fb.FINGERPRINT_REL)
    elif fingerprint == "falla":
        text = (PROJECT_ROOT / fb.FINGERPRINT_REL).read_text(encoding="utf-8").replace("byte a byte: True", "byte a byte: False")
        (root / fb.FINGERPRINT_REL).write_text(text, encoding="utf-8")
    (root / e2a.E2_REL).mkdir(parents=True)
    shutil.copy(E2_DIR / e2a.PROTOCOL_NAME, root / e2a.E2_REL / e2a.PROTOCOL_NAME)
    shutil.copytree(E2_DIR / "corridas", root / e2a.E2_REL / "corridas", ignore=shutil.ignore_patterns("*.pth"))
    for config in ("b0", "b2"):
        shutil.copytree(CONFIRMATION_DIR / config, root / cf.CONFIRMATION_REL / config, ignore=shutil.ignore_patterns("*.pth"))
    shutil.copytree(PROJECT_ROOT / cf.STAGE1_ANALYSIS_REL, root / cf.STAGE1_ANALYSIS_REL)
    (root / OUTPUT_REL).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(PROJECT_ROOT / OUTPUT_REL, root / OUTPUT_REL)

    folds_rows = read_csv_rows(root / OUTPUT_REL)
    keys = fb.compared_config_keys(protocol)
    b0_config = {k: json.loads((root / e2a.E2_REL / "corridas" / "convnext_a_alpha1" / f"fold_{k}" / "run_config.json").read_text(encoding="utf-8"))
                 for k in range(5)}
    rng = random.Random(0)
    for run in fb.planned_new_runs(protocol):
        _write_run(root, run, folds_rows, qualities.get(run["alpha"], 0.0), rng, keys, b0_config)
    if conditional_quality is not None:
        for run in fb.planned_conditional_runs(protocol):
            _write_run(root, run, folds_rows, conditional_quality, rng, keys, b0_config)
    return protocol


STRONG_ZERO = {0.0: 5.0, 0.25: 0.0, 0.75: 0.0}  # α = 0 muy por encima de B0 (AP ≈ 1); 0.25 y 0.75 muy por debajo


def _run_dir(root, config_id="alpha_025", seed=43, fold=0):
    return Path(root) / fb.PHASE_B_REL / "corridas" / config_id / f"seed_{seed}" / f"fold_{fold}"


def _out(root):
    return Path(root) / fb.PHASE_B_REL / fb.ANALYSIS_NAME


# ---------------------------------------------------------------------------------------------------------
# Protocolo: lecturas del texto, comandos y huella
# ---------------------------------------------------------------------------------------------------------
@skip_without_protocol
class TestProtocol:
    def test_design_and_thresholds_are_read_from_the_text(self):
        protocol = _protocol()
        assert fb.design(protocol) == {"alphas": [0.0, 0.25, 0.5, 0.75, 1.0], "seeds": [42, 43, 44], "folds": [0, 1, 2, 3, 4]}
        assert fb.tie_threshold(protocol) == 0.005 and fb.consistency_minimum(protocol) == 4
        keys = fb.compared_config_keys(protocol)
        assert {"architecture", "learning_rate", "history_metrics", "checkpoint_criterion", "val_predictions"} <= set(keys)
        assert not {"seed", "experiment", "class_weight_alpha", "class_weights"} & set(keys)

    def test_protocol_commands_match_the_template(self):
        assert all(r["ok"] for r in fb.verify_protocol_commands(_protocol()))

    @pytest.mark.parametrize(
        "edit",
        [
            lambda p: p["comandos"]["corridas"][3].update(comando=p["comandos"]["corridas"][3]["comando"].replace("--fold 3", "--fold 2")),
            lambda p: p["comandos"]["corridas"][2].update(comando=p["comandos"]["corridas"][2]["comando"].replace("alpha 0.0 ", "alpha 0.1 ")),
            lambda p: p["comandos"]["corridas"].pop(),
            lambda p: p["comandos"]["corridas"][0].update(comando=p["comandos"]["corridas"][0]["comando"] + " --overwrite"),
            lambda p: p["comandos"]["condicionales"]["corridas"][0].update(comando=p["comandos"]["condicionales"]["corridas"][0]["comando"].replace("--seed 42", "--seed 43")),
            lambda p: p["comandos"]["condicionales"]["corridas"][6].update(comando=p["comandos"]["condicionales"]["corridas"][6]["comando"].replace(" --class-weight-alpha 0.5", "")),
        ],
        ids=["fold", "alpha", "falta_una", "argumento_extra", "condicional_semilla", "condicional_sin_alpha"],
    )
    def test_tampered_protocol_commands_are_detected(self, edit):
        protocol = _protocol()
        edit(protocol)
        assert not all(r["ok"] for r in fb.verify_protocol_commands(protocol))

    @skip_without_reused
    def test_fingerprint_verdict(self, tmp_path):
        protocol = _protocol()
        assert fb.fingerprint_accepted(PROJECT_ROOT, protocol)[0] is True
        (tmp_path / fb.FINGERPRINT_REL).parent.mkdir(parents=True)
        assert fb.fingerprint_accepted(tmp_path, protocol)[0] is None  # no se aporta
        text = (PROJECT_ROOT / fb.FINGERPRINT_REL).read_text(encoding="utf-8")
        (tmp_path / fb.FINGERPRINT_REL).write_text(text.replace("rutas de src distintas: True", "rutas de src distintas: False"), encoding="utf-8")
        assert fb.fingerprint_accepted(tmp_path, protocol)[0] is False


# ---------------------------------------------------------------------------------------------------------
# Lecturas, piso, etiqueta, mejor α (funciones puras)
# ---------------------------------------------------------------------------------------------------------
def _stats(per_alpha):
    """per_alpha: α -> semilla -> lista de 5 AP (ROC-AUC = AP + 0.3)."""
    return {a: {s: {k: {"ap": v, "auc": v + 0.3, "epoca": 3, "epocas": 8} for k, v in enumerate(values)} for s, values in seeds.items()}
            for a, seeds in per_alpha.items()}


B0_FLAT = {42: [0.5] * 5, 43: [0.5] * 5, 44: [0.5] * 5}


class TestDeltaReadings:
    def test_three_readings(self):
        stats = _stats({1.0: B0_FLAT, 0.0: {42: [0.6, 0.6, 0.6, 0.6, 0.4], 43: [0.6, 0.6, 0.6, 0.5, 0.5], 44: [0.4, 0.6, 0.6, 0.6, 0.6]}})
        r = fb.delta_readings(stats, 0.0, [42, 43, 44], 4)
        assert r["a"][42]["folds_positivos"] == 4 and r["a"][43]["folds_cero"] == 2 and r["a"][44]["folds_negativos"] == 1
        # lectura b: Δ por fold sobre el promedio de las 3 semillas
        assert all(abs(a - b) < 1e-6 for a, b in zip(r["b"]["deltas"], [1 / 30, 0.1, 0.1, 2 / 30, 0.0]))
        assert r["b"]["veredicto"] == "mejor_consistente"
        assert r["c"] == {"n": 15, "positivos": 11, "negativos": 2, "ceros": 2}

    @pytest.mark.parametrize("values, verdict", [([0.4, 0.4, 0.4, 0.4, 0.9], "peor_consistente"), ([0.6, 0.6, 0.4, 0.4, 0.5], "no_consistente"),
                                                  ([0.6, 0.6, 0.6, 0.5, 0.5], "no_consistente")])
    def test_rule_in_both_directions_and_zero_counts_in_neither(self, values, verdict):
        stats = _stats({1.0: B0_FLAT, 0.25: {s: values for s in (42, 43, 44)}})
        assert fb.delta_readings(stats, 0.25, [42, 43, 44], 4)["b"]["veredicto"] == verdict


class TestNoiseFloor:
    def test_floor_is_the_largest_absolute_mean_difference_between_seeds(self):
        stats = _stats({1.0: {42: [0.5] * 5, 43: [0.55] * 5, 44: [0.5] * 5}, 0.5: {42: [0.5] * 5, 43: [0.5] * 5, 44: [0.42] * 5}})
        rows, piso, top, consistent = fb.noise_floor(stats, [42, 43, 44], 4)
        assert piso == pytest.approx(0.08) and top["configuracion"] == "B2" and top["semillas"] in ("42->44", "43->44")
        pair = next(r for r in rows if r["tipo"] == "par_de_semillas" and r["configuracion"] == "B0" and r["semillas"] == "42->43" and r["metrica"] == "AP")
        assert pair["valor"] == pytest.approx(0.05)  # dirección s2 - s1
        assert consistent == 4  # B0 42->43, 43->44; B2 42->44, 43->44 (cinco folds del mismo signo); los pares sin diferencia no cuentan
        sd = [r for r in rows if r["tipo"] == "de_entre_semillas" and r["configuracion"] == "B0" and r["metrica"] == "AP"]
        assert len(sd) == 5 and sd[0]["valor"] == pytest.approx(statistics.stdev([0.5, 0.55, 0.5]))


class TestNoiseLabel:
    @staticmethod
    def _readings(seed_means, b_mean, verdict):
        return {"a": {s: {"media": m} for s, m in zip((42, 43, 44), seed_means)}, "b": {"media": b_mean, "veredicto": verdict}}

    @pytest.mark.parametrize(
        "seed_means, b_mean, verdict, label",
        [
            ((0.12, 0.10, 0.11), 0.11, "mejor_consistente", fb.LABEL_ABOVE),
            ((-0.12, -0.10, -0.11), -0.11, "peor_consistente", fb.LABEL_ABOVE),
            ((0.12, 0.10, 0.11), 0.11, "no_consistente", fb.LABEL_WITHIN),  # (i)
            ((0.20, -0.01, 0.14), 0.11, "mejor_consistente", fb.LABEL_WITHIN),  # (ii) signos distintos
            ((0.20, 0.0, 0.13), 0.11, "mejor_consistente", fb.LABEL_WITHIN),  # (ii) un Δ medio exactamente 0
            ((0.10, 0.09, 0.095), 0.095, "mejor_consistente", fb.LABEL_WITHIN),  # (iii) por debajo del piso
            ((0.10, 0.10, 0.10), 0.1, "mejor_consistente", fb.LABEL_WITHIN),  # (iii) igual al piso: no es mayor
        ],
        ids=["arriba_positivo", "arriba_negativo", "b_no_consistente", "signos_mixtos", "semilla_en_cero", "bajo_el_piso", "igual_al_piso"],
    )
    def test_label(self, seed_means, b_mean, verdict, label):
        assert fb.noise_label(self._readings(seed_means, b_mean, verdict), 0.1)["etiqueta"] == label


class TestBestAlpha:
    @staticmethod
    def _curve(*rows):
        return [{"alpha": a, "ap_medio": ap, "roc_auc_medio": auc} for a, ap, auc in rows]

    def test_highest_mean_ap(self):
        assert fb.best_alpha(self._curve((0.0, 0.55, 0.9), (0.5, 0.60, 0.8), (1.0, 0.58, 0.9)), 0.005)["alpha"] == 0.5

    def test_tie_below_0005_goes_to_higher_auc(self):
        best = fb.best_alpha(self._curve((0.0, 0.600, 0.85), (0.5, 0.597, 0.90), (1.0, 0.590, 0.95)), 0.005)
        assert best["alpha"] == 0.5 and best["en_empate"] == [0.0, 0.5] and best["el_desempate_cambia_el_resultado"] is True

    def test_difference_of_0005_or_more_is_not_a_tie(self):
        assert fb.best_alpha(self._curve((0.0, 0.606, 0.85), (0.5, 0.600, 0.99)), 0.005)["en_empate"] == [0.0]

    def test_remaining_tie_goes_to_the_alpha_closest_to_one(self):
        assert fb.best_alpha(self._curve((0.25, 0.60, 0.9), (0.75, 0.60, 0.9), (0.5, 0.60, 0.9)), 0.005)["alpha"] == 0.75


class TestShapeAndMultiplicity:
    @pytest.mark.parametrize("values, shape", [([0.1, 0.2, 0.3], "monotona_creciente"), ([0.3, 0.2, 0.1], "monotona_decreciente"),
                                                ([0.1, 0.3, 0.2], "no_monotona"), ([0.1, 0.1, 0.2], "no_monotona")])
    def test_curve_shape(self, values, shape):
        assert fb.curve_shape(values) == shape

    def test_multiplicity_matches_the_protocol(self):
        m = fb.multiplicity(4, 4)
        assert m["p_una"] == pytest.approx(6 / 32) and m["p_al_menos_una"] == pytest.approx(1 - 0.8125 ** 4) and round(m["p_al_menos_una"], 2) == 0.56


# ---------------------------------------------------------------------------------------------------------
# Análisis completo
# ---------------------------------------------------------------------------------------------------------
@skip_without_reused
class TestRunAnalysis:
    def test_full_analysis_with_reused_runs(self, tmp_path):
        protocol = build_tree(tmp_path, STRONG_ZERO)
        code, results = fb.run_analysis(tmp_path, verbose=False)
        out = _out(tmp_path)
        assert code == 0 and all(c["ok"] for c in results["checks"])
        assert {p.name for p in out.iterdir()} == set(fb.OUTPUT_FILES.values())

        # las cifras del protocolo sobre B0 y B2 se reproducen con las corridas reutilizadas
        assert any(c["comprobacion"] == "valor del protocolo: piso_ap" for c in results["checks"])
        assert results["piso"] == pytest.approx(0.0949, abs=5e-5)

        registry = read_csv_rows(out / fb.OUTPUT_FILES["registro"])
        assert list(registry[0].keys()) == protocol["registro"]["columnas"] and len(registry) == 75
        origins = [r["origen"] for r in registry]
        assert origins.count("nueva") == 45 and sum(o.startswith("reutilizada:") for o in origins) == 30
        assert {r["config_id"] for r in registry} == {"alpha_000", "alpha_025", "alpha_050", "alpha_075", "alpha_100"}

        labels = results["labels"]
        assert set(labels) == {0.0, 0.25, 0.5, 0.75}
        assert labels[0.0]["etiqueta"] == fb.LABEL_ABOVE and labels[0.5]["etiqueta"] == fb.LABEL_WITHIN  # B2 frente a B0 real: dentro del piso
        assert results["best"]["alpha"] == 0.0
        payload = json.loads((out / fb.OUTPUT_FILES["best_alpha"]).read_text(encoding="utf-8"))
        assert payload["rotulo_obligatorio"].startswith("exploratorio") and payload["multiplicidad"]["comparaciones"] == 4
        assert payload["piso_de_ruido"]["semilla_42"] == "reutilizada" and payload["piso_de_ruido"]["pares_de_una_misma_configuracion_que_cumplen_4_de_5"] == 3

        summary = read_csv_rows(out / fb.OUTPUT_FILES["delta_summary"])
        assert {r["lectura"] for r in summary} == {"a_por_semilla", "b_promedio_3_semillas", "c_15_pares"}
        assert len(summary) == 4 * 2 * (3 + 1 + 1)
        assert not any(("ic" in key.split("_")) or "p_valor" in key for key in summary[0])
        assert len(read_csv_rows(out / fb.OUTPUT_FILES["delta"])) == 4 * 2 * 15
        assert len(read_csv_rows(out / fb.OUTPUT_FILES["by_fold"])) == 5 * 5

        with pytest.raises(FileExistsError):
            fb.run_analysis(tmp_path, verbose=False)

    def test_a_missing_run_stops_without_writing(self, tmp_path):
        build_tree(tmp_path, STRONG_ZERO)
        shutil.rmtree(_run_dir(tmp_path, "alpha_075", 44, 4))
        code, results = fb.run_analysis(tmp_path, verbose=False)
        assert code == 1 and results is None and not _out(tmp_path).exists()

    def test_an_incomplete_new_run_marks_its_alpha_as_incomplete(self, tmp_path):
        build_tree(tmp_path, STRONG_ZERO)
        (_run_dir(tmp_path, "alpha_025", 43, 2) / "validation_metrics.json").unlink()
        code, results = fb.run_analysis(tmp_path, verbose=False)
        assert code == 0 and results["payload"]["alphas_incompletos"] == [0.25]
        assert 0.25 not in results["labels"] and 0.25 not in [c["alpha"] for c in results["curve"]]
        failed = [r for r in results["registry"] if r["estado"] == "fallida"]
        assert len(failed) == 1 and failed[0]["config_id"] == "alpha_025" and failed[0]["ap"] == "" and len(results["registry"]) == 75

    @pytest.mark.parametrize(
        "tamper",
        [
            ("alpha_025", 42, 0, "run_config.json", lambda d: d.update(class_weight_alpha=0.5)),
            ("alpha_000", 43, 1, "run_config.json", lambda d: d.update(class_weights=[0.56, 4.6])),  # pesos de α = 1 con α = 0
            ("alpha_075", 44, 2, "run_config.json", lambda d: d.update(seed=42)),
            ("alpha_000", 42, 3, "run_config.json", lambda d: d.update(fold=4)),
            ("alpha_025", 43, 4, "run_config.json", lambda d: d.update(epoch_best=40)),
            ("alpha_075", 42, 1, "run_config.json", lambda d: d.update(batch_size=16)),
            ("alpha_000", 44, 0, "run_config.json", lambda d: d.update(lr_schedule={"type": "warmup_cosine"})),
            ("alpha_025", 44, 1, "training_history.json", lambda d: d["val_loss"].__setitem__(1, float("nan"))),
            ("alpha_075", 43, 3, "validation_metrics.json", lambda d: d.update(average_precision=d["average_precision"] + 0.01)),
        ],
        ids=["alpha_registrado", "pesos", "semilla", "fold", "epoca_fuera_de_rango", "clave_distinta_de_b0", "clave_fase_a", "nan", "ap_no_reproducible"],
    )
    def test_integrity_failures_stop_without_writing(self, tmp_path, tamper):
        config_id, seed, fold, filename, edit = tamper
        build_tree(tmp_path, STRONG_ZERO)
        path = _run_dir(tmp_path, config_id, seed, fold) / filename
        data = json.loads(path.read_text(encoding="utf-8"))
        edit(data)
        path.write_text(json.dumps(data), encoding="utf-8")
        code, results = fb.run_analysis(tmp_path, verbose=False)
        assert code == 1 and results is None and not _out(tmp_path).exists()

    def test_a_different_fold_manifest_stops_without_writing(self, tmp_path):
        build_tree(tmp_path, STRONG_ZERO)
        path = _run_dir(tmp_path, "alpha_000", 42, 2) / "fold_manifest.csv"
        path.write_bytes(path.read_bytes() + b"\n")  # mismo contenido de filas, otro hash
        code, _ = fb.run_analysis(tmp_path, verbose=False)
        assert code == 1 and not _out(tmp_path).exists()

    def test_tampered_protocol_commands_stop_without_writing(self, tmp_path):
        edit = lambda p: p["comandos"]["corridas"][5].update(comando=p["comandos"]["corridas"][5]["comando"].replace("--seed 42", "--seed 45"))
        build_tree(tmp_path, {}, protocol_edit=edit)
        assert fb.run_analysis(tmp_path, verbose=False)[0] == 1 and not _out(tmp_path).exists()

    @pytest.mark.parametrize("fingerprint", ["falla", None], ids=["huella_falla", "huella_ausente"])
    def test_without_the_fingerprint_the_conditional_runs_are_required(self, tmp_path, fingerprint):
        build_tree(tmp_path / "sin", STRONG_ZERO, fingerprint=fingerprint)
        assert fb.run_analysis(tmp_path / "sin", verbose=False)[0] == 1  # faltan las 10 condicionales

        build_tree(tmp_path / "con", STRONG_ZERO, fingerprint=fingerprint, conditional_quality=1.0)
        code, results = fb.run_analysis(tmp_path / "con", verbose=False)
        assert code == 0 and results["payload"]["piso_de_ruido"]["semilla_42"] == "condicional"
        assert not any(c["comprobacion"].startswith("valor del protocolo") for c in results["checks"])  # el piso se recalcula
        seed42 = [r for r in results["registry"] if r["seed"] == 42 and r["config_id"] in ("alpha_100", "alpha_050")]
        assert len(seed42) == 10 and all(r["origen"] == "nueva" and "condicional" in r["nota"] for r in seed42)

    def test_conditional_runs_with_an_accepted_fingerprint_stop_the_analysis(self, tmp_path):
        build_tree(tmp_path, STRONG_ZERO, conditional_quality=1.0)
        assert fb.run_analysis(tmp_path, verbose=False)[0] == 1 and not _out(tmp_path).exists()

    def test_the_test_split_is_never_read(self, tmp_path):
        build_tree(tmp_path, STRONG_ZERO)
        manifests = list((tmp_path / fb.PHASE_B_REL / "corridas").glob("*/seed_*/fold_*/fold_manifest.csv"))
        assert len(manifests) == 45
        for manifest in manifests:
            rows = read_csv_rows(manifest)
            assert {r["e1_split"] for r in rows} <= {"train", "validation"} and len(rows) == 1000
