"""
Pruebas del análisis de la confirmación del agente (src/classification/agente_confirmacion_analysis.py). Solo biblioteca estándar.

Las 30 corridas de la confirmación se SIMULAN en una carpeta temporal (datos de prueba, no resultados) con los mismos archivos que
una corrida real. Las referencias B0 y B2 (semilla 42) son las corridas reales de robustez_e2; las pruebas se omiten si no están.
Una prueba adicional corre el análisis sobre las 30 corridas reales si están en este clon (git pull) y comprueba el resultado
contra un recálculo independiente desde los validation_metrics.json, sin fijar ninguna cifra.
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
from src.classification import robustez_e2_analysis as e2a
from src.classification.cv_folds_e2 import OUTPUT_REL, read_csv_rows
from tests.test_agente_etapa1_analysis import _write_synthetic_run

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_PROTOCOL = PROJECT_ROOT / a1.AGENT_REL / a1.PROTOCOL_NAME
E2_DIR = PROJECT_ROOT / e2a.E2_REL
CONFIRMATION_DIR = PROJECT_ROOT / cf.CONFIRMATION_REL
HAVE_REFERENCES = AGENT_PROTOCOL.exists() and (E2_DIR / "corridas" / "convnext_a_alpha1" / "fold_4" / "validation_metrics.json").exists()
HAVE_REAL_CONFIRMATION = (HAVE_REFERENCES and (PROJECT_ROOT / cf.STAGE1_ANALYSIS_REL / cf.COMMANDS_NAME).exists()
                          and (CONFIRMATION_DIR / "b2" / "seed_44" / "fold_4" / "validation_metrics.json").exists())
skip_without_files = pytest.mark.skipif(not HAVE_REFERENCES, reason="faltan el protocolo del agente o las corridas de robustez_e2 (git pull)")
skip_without_real_runs = pytest.mark.skipif(not HAVE_REAL_CONFIRMATION, reason="faltan las 30 corridas reales de la confirmación (git pull)")

FINALIST = "a03_cosine"


def _protocol():
    return json.loads(AGENT_PROTOCOL.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------------------
# Árbol sintético
# ---------------------------------------------------------------------------------------------------------
def _patch_confirmation_config(folder, run):
    """Una corrida de confirmación: semilla del comando y pesos de clase con el α del comando (B2 = 0.5 y lo registra)."""
    path = folder / "run_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["seed"] = run["seed"]
    alpha = float(run["args"]["--class-weight-alpha"]) if "--class-weight-alpha" in run["args"] else 1.0
    train = [r for r in read_csv_rows(folder / "fold_manifest.csv") if r["experimental_split"] == "train"]
    config["class_weights"] = e2a.expected_class_weights([int(r["binary_label"]) for r in train], alpha)
    if "--class-weight-alpha" in run["args"]:
        config["class_weight_alpha"] = alpha
    path.write_text(json.dumps(config), encoding="utf-8")


def build_tree(root, qualities, protocol_edit=None):
    """Protocolo, referencias reales, archivos de la etapa 1 (finalista y comandos) y 30 corridas sintéticas. qualities: config -> señal."""
    root = Path(root)
    (root / a1.AGENT_REL).mkdir(parents=True)
    protocol = _protocol()
    if protocol_edit:
        protocol_edit(protocol)
    (root / a1.AGENT_REL / a1.PROTOCOL_NAME).write_text(json.dumps(protocol), encoding="utf-8")
    (root / e2a.E2_REL).mkdir(parents=True)
    shutil.copy(E2_DIR / e2a.PROTOCOL_NAME, root / e2a.E2_REL / e2a.PROTOCOL_NAME)
    shutil.copytree(E2_DIR / "corridas", root / e2a.E2_REL / "corridas", ignore=shutil.ignore_patterns("*.pth"))
    (root / OUTPUT_REL).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(PROJECT_ROOT / OUTPUT_REL, root / OUTPUT_REL)

    candidates = a1.candidate_table(protocol)
    commands = a1.generate_confirmation_commands(protocol, candidates, FINALIST)
    stage1 = root / cf.STAGE1_ANALYSIS_REL
    stage1.mkdir(parents=True)
    (stage1 / cf.COMMANDS_NAME).write_text(json.dumps({"finalista": FINALIST, "flags_del_finalista": candidates[FINALIST]["flags"],
                                                       "total": 30, "comandos": commands}), encoding="utf-8")
    (stage1 / cf.SELECTION_NAME).write_text(json.dumps({"finalista": FINALIST, "conteo": {"corridas_etapa_1": 45, "fallidas": 0}}), encoding="utf-8")

    folds_rows = read_csv_rows(root / OUTPUT_REL)
    rng = random.Random(0)
    flags = dict(a1.confirmation_entries(protocol, candidates, FINALIST))
    for c in commands:
        run = {"id": c["id"], "config_id": c["config_id"], "fold": c["fold"], "seed": c["seed"], "command": c["comando"],
               "args": e2a.parse_command(c["comando"]), "dir_rel": c["salida"], "flags": flags[c["config_id"]]}
        _write_synthetic_run(root, run, folds_rows, qualities.get(c["config_id"], 0.0), rng)
        _patch_confirmation_config(root / c["salida"], run)
    return protocol


WINNER = {FINALIST: 5.0, "b0": 1.0, "b2": 1.0}  # AP y AUC del finalista muy por encima de B0 y B2
TIE = {FINALIST: 1.0, "b0": 1.0, "b2": 1.0}


def _run_dir(root, config="a03_cosine", seed=43, fold=0):
    return Path(root) / cf.CONFIRMATION_REL / config / f"seed_{seed}" / f"fold_{fold}"


# ---------------------------------------------------------------------------------------------------------
# Reglas leídas del protocolo
# ---------------------------------------------------------------------------------------------------------
@skip_without_files
class TestProtocolRules:
    def test_design_and_decision_rule_are_read_from_the_protocol_text(self):
        protocol = _protocol()
        design, rule = cf.read_confirmation_design(protocol), cf.read_decision_rule(protocol)
        assert design == {"seeds": [43, 44], "folds": [0, 1, 2, 3, 4], "total": 30, "configs": ["finalista", "B0", "B2"]}
        assert [(c["kind"], c["reference"], c["threshold"]) for c in rule] == [
            ("delta_ap_medio", "B0", 0.04), ("delta_ap_medio", "B2", 0.04), ("consistencia_ap", "B0", 4),
            ("consistencia_ap", "B2", 4), ("delta_auc_medio", "B0", -0.02), ("delta_auc_medio", "B2", -0.02)]

    def test_thresholds_follow_the_text_and_malformed_rules_are_rejected(self):
        protocol = _protocol()
        texts = protocol["regla_de_eleccion"]["la_finalista_es_ganadora_si_cumple_todas"]
        texts[0] = texts[0].replace("+0.04", "+0.05")
        assert cf.read_decision_rule(protocol)[0]["threshold"] == 0.05
        texts[1] = "ΔAP medio frente a B2 mayor que cuatro centésimas"
        with pytest.raises(ValueError):
            cf.read_decision_rule(protocol)
        protocol = _protocol()
        del protocol["regla_de_eleccion"]["la_finalista_es_ganadora_si_cumple_todas"][5]
        with pytest.raises(ValueError):
            cf.read_decision_rule(protocol)


# ---------------------------------------------------------------------------------------------------------
# regla_de_eleccion y estadísticos descriptivos (funciones puras)
# ---------------------------------------------------------------------------------------------------------
def _values(finalist_ap, b0_ap, b2_ap, finalist_auc=(0.9,) * 5, b0_auc=(0.9,) * 5, b2_auc=(0.9,) * 5):
    return {FINALIST: {"ap": list(finalist_ap), "auc": list(finalist_auc)}, "b0": {"ap": list(b0_ap), "auc": list(b0_auc)},
            "b2": {"ap": list(b2_ap), "auc": list(b2_auc)}}


@skip_without_files
class TestEvaluateRule:
    RULE = cf.read_decision_rule(_protocol()) if HAVE_REFERENCES else None

    def _evaluate(self, values):
        return cf.evaluate_rule(values, FINALIST, self.RULE)

    def test_all_conditions_met_is_winner(self):
        result = self._evaluate(_values([0.65] * 5, [0.55] * 5, [0.50] * 5))
        assert result["veredicto"] == "ganadora" and result["incumplidas"] == []

    def test_mean_below_margin_fails_only_the_mean_conditions(self):
        result = self._evaluate(_values([0.58] * 5, [0.55] * 5, [0.50] * 5))  # ΔAP = +0.03 frente a B0: < 0.04; +0.08 frente a B2
        assert result["veredicto"] == "sin_ganadora" and result["incumplidas"] == ["C1"]

    def test_each_reference_limits_independently(self):
        result = self._evaluate(_values([0.65] * 5, [0.55] * 5, [0.63] * 5))  # frente a B2 solo +0.02
        assert result["incumplidas"] == ["C2"]

    def test_four_of_five_folds_is_enough_and_three_is_not(self):
        four = self._evaluate(_values([0.80, 0.80, 0.80, 0.80, 0.30], [0.5] * 5, [0.5] * 5))
        assert four["incumplidas"] == [] and [c["valor"] for c in four["condiciones"] if c["id"] in ("C3", "C4")] == [4, 4]
        three = self._evaluate(_values([0.80, 0.80, 0.80, 0.40, 0.40], [0.5] * 5, [0.5] * 5))  # ΔAP medio +0.14 (cumple) pero solo 3 folds positivos
        assert three["incumplidas"] == ["C3", "C4"]
        assert all(c["valor"] == 3 for c in three["condiciones"] if c["id"] in ("C3", "C4"))

    def test_a_zero_delta_counts_in_neither_sign(self):
        values = _values([0.9, 0.9, 0.9, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5])  # 3 positivos y 2 ceros
        result = self._evaluate(values)
        assert [c["valor"] for c in result["condiciones"] if c["id"] in ("C3", "C4")] == [3, 3] and result["incumplidas"] == ["C3", "C4"]

    @pytest.mark.parametrize("drop, ok", [(0.019, True), (0.021, False)])
    def test_auc_floor_is_applied_to_the_mean_delta_against_each_reference(self, drop, ok):
        values = _values([0.65] * 5, [0.55] * 5, [0.50] * 5, finalist_auc=(0.9 - drop,) * 5)
        result = self._evaluate(values)
        assert result["incumplidas"] == ([] if ok else ["C5", "C6"])

    def test_margin_is_reported_without_rounding(self):
        result = self._evaluate(_values([0.65] * 5, [0.55] * 5, [0.50] * 5))
        first = result["condiciones"][0]
        assert first["margen"] == pytest.approx(first["valor"] - 0.04) and first["cumple"] is True


class TestDescribeDeltas:
    def test_sample_standard_deviation_and_both_directions(self):
        better = cf.describe_deltas([0.1, 0.1, 0.1, 0.1, -0.2], 4)
        assert better["veredicto"] == "mejor_consistente" and better["desv_est_muestral"] == pytest.approx(statistics.stdev([0.1, 0.1, 0.1, 0.1, -0.2]))
        assert cf.describe_deltas([-0.1, -0.1, -0.1, -0.1, 0.2], 4)["veredicto"] == "peor_consistente"
        assert cf.describe_deltas([0.1, 0.1, 0.0, 0.0, -0.1], 4)["veredicto"] == "no_consistente"

    def test_no_confidence_interval_or_p_value_is_computed(self):
        described = cf.describe_deltas([0.1, 0.2, 0.3, 0.4, 0.5], 4)
        assert not any(("ic" in key.lower().split("_")) or "p_valor" in key or "intervalo" in key for key in described)

    def test_seed_average_is_the_plain_mean_of_the_requested_seeds_only(self):
        seed_stats = {s: {k: {"ap": float(s) + k} for k in range(5)} for s in (42, 43, 44)}
        assert cf.seed_average(seed_stats, [43, 44], "ap") == [43.5 + k for k in range(5)]  # la semilla 42 no entra


class TestContrastB2VsB0:
    @staticmethod
    def _stats():
        make = lambda base: {k: {"ap": base[k], "auc": 0.9} for k in range(5)}
        confirmation = {"b2": {43: make([0.5] * 5), 44: make([0.6] * 5)}, "b0": {43: make([0.6] * 5), 44: make([0.5] * 5)}}
        seed42 = {"b2": make([0.4] * 5), "b0": make([0.5, 0.5, 0.5, 0.5, 0.3])}
        return confirmation, seed42

    def test_three_readings_are_reported_and_none_is_chosen(self):
        confirmation, seed42 = self._stats()
        deltas, summaries = cf.contrast_b2_vs_b0(confirmation, seed42, [43, 44], 4)
        ap = {(r["lectura"], r["semillas"]): r for r in summaries if r["metrica"] == "AP"}
        assert set(ap) == {(cf.READING_A, "42,43,44"), (cf.READING_B, "42"), (cf.READING_B, "43"), (cf.READING_B, "44"), (cf.READING_C, "43,44")}
        assert ap[(cf.READING_C, "43,44")]["media"] == pytest.approx(0.0)  # 43: -0.1, 44: +0.1
        assert ap[(cf.READING_B, "43")]["veredicto"] == "peor_consistente" and ap[(cf.READING_B, "44")]["veredicto"] == "mejor_consistente"
        assert ap[(cf.READING_B, "42")]["folds_negativos"] == 4 and ap[(cf.READING_B, "42")]["folds_positivos"] == 1
        # lectura A: B2 = 0.5 en cada fold; B0 = 0.5333 en los folds 0-3 y 0.4667 en el 4 -> Δ = -0.0333 (x4) y +0.0333, media -0.02
        assert ap[(cf.READING_A, "42,43,44")]["media"] == pytest.approx(-0.02) and ap[(cf.READING_A, "42,43,44")]["veredicto"] == "peor_consistente"
        assert len(deltas) == 5 * 2 * 5  # 5 lecturas x 2 métricas x 5 folds

    def test_reading_a_averages_the_three_seeds_per_fold(self):
        confirmation, seed42 = self._stats()
        deltas, _ = cf.contrast_b2_vs_b0(confirmation, seed42, [43, 44], 4)
        fold0 = [r for r in deltas if r["lectura"] == cf.READING_A and r["metrica"] == "AP" and r["fold"] == 0][0]
        assert fold0["valor_b2"] == pytest.approx((0.4 + 0.5 + 0.6) / 3) and fold0["valor_b0"] == pytest.approx((0.5 + 0.6 + 0.5) / 3)


# ---------------------------------------------------------------------------------------------------------
# Análisis completo sobre corridas sintéticas
# ---------------------------------------------------------------------------------------------------------
@skip_without_files
class TestRunAnalysis:
    def test_winner_scenario_writes_every_output_and_applies_the_rule(self, tmp_path):
        build_tree(tmp_path, WINNER)
        code, results = cf.run_analysis(tmp_path, verbose=False)
        out = tmp_path / cf.CONFIRMATION_REL / cf.ANALYSIS_NAME

        assert code == 0 and all(c["ok"] for c in results["checks"])
        assert {p.name for p in out.iterdir()} == set(cf.OUTPUT_FILES.values())
        verdict = json.loads((out / cf.OUTPUT_FILES["verdict"]).read_text(encoding="utf-8"))
        assert verdict["finalista"] == FINALIST and verdict["veredicto_mecanico"] == "ganadora" and verdict["condiciones_incumplidas"] == []
        assert [c["id"] for c in verdict["condiciones"]] == ["C1", "C2", "C3", "C4", "C5", "C6"]
        assert verdict["consecuencia"] == _protocol()["regla_de_eleccion"]["si_cumple"]
        assert "intervalo_de_confianza_95" in verdict["no_calculado"] and len(verdict["puntos_pendientes_de_decision"]) == 3

        registry = read_csv_rows(out / cf.OUTPUT_FILES["registro"])
        assert list(registry[0].keys()) == _protocol()["registro"]["columnas"] and len(registry) == 30
        assert {r["etapa"] for r in registry} == {"confirmacion"} and {r["estado"] for r in registry} == {"completada"}
        assert {(r["config_id"], r["seed"], r["fold"]) for r in registry} == {(c, s, k) for c in (FINALIST, "b0", "b2") for s in ("43", "44") for k in map(str, range(5))}
        by_fold = read_csv_rows(out / cf.OUTPUT_FILES["by_fold"])
        assert len(by_fold) == 15 and {"ap_semilla_43", "ap_semilla_44", "ap_promedio_semillas"} <= set(by_fold[0])
        deltas = read_csv_rows(out / cf.OUTPUT_FILES["delta_by_fold"])
        assert len(deltas) == 2 * 2 * 5
        assert not any("ic" in key.split("_") or "p_valor" in key for key in deltas[0])

        with pytest.raises(FileExistsError):
            cf.run_analysis(tmp_path, verbose=False)

    def test_equal_configurations_give_no_winner(self, tmp_path):
        build_tree(tmp_path, TIE)
        code, results = cf.run_analysis(tmp_path, verbose=False)
        assert code == 0 and results["decision"]["veredicto"] == "sin_ganadora" and "C1" in results["decision"]["incumplidas"]
        verdict = results["verdict"]
        assert verdict["consecuencia"] == _protocol()["regla_de_eleccion"]["sin_ganadora"]

    def test_fold_values_are_the_mean_of_seeds_43_and_44(self, tmp_path):
        build_tree(tmp_path, WINNER)
        _, results = cf.run_analysis(tmp_path, verbose=False)
        for config in (FINALIST, "b0", "b2"):
            for fold in range(5):
                per_seed = [json.loads((_run_dir(tmp_path, config, s, fold) / "validation_metrics.json").read_text(encoding="utf-8"))["average_precision"] for s in (43, 44)]
                assert results["values"][config]["ap"][fold] == pytest.approx(statistics.mean(per_seed))

    def test_a_run_not_executed_stops_without_writing(self, tmp_path):
        build_tree(tmp_path, WINNER)
        shutil.rmtree(_run_dir(tmp_path, "b2", 44, 3))
        code, results = cf.run_analysis(tmp_path, verbose=False)
        assert code == 1 and results is None and not (tmp_path / cf.CONFIRMATION_REL / cf.ANALYSIS_NAME).exists()

    def test_an_incomplete_run_stops_without_writing(self, tmp_path):
        build_tree(tmp_path, WINNER)
        (_run_dir(tmp_path, FINALIST, 43, 2) / "validation_metrics.json").unlink()
        code, results = cf.run_analysis(tmp_path, verbose=False)
        assert code == 1 and results is None and not (tmp_path / cf.CONFIRMATION_REL / cf.ANALYSIS_NAME).exists()

    @pytest.mark.parametrize(
        "tamper",
        [
            ("a03_cosine", 43, 0, "run_config.json", lambda d: d.update(seed=42)),  # semilla distinta de la del comando
            ("b0", 44, 1, "run_config.json", lambda d: d.update(fold=2)),
            ("b2", 43, 2, "run_config.json", lambda d: d.pop("class_weight_alpha")),  # B2 sin su exponente
            ("b0", 43, 3, "run_config.json", lambda d: d.update(class_weight_alpha=0.5)),  # B0 con exponente de B2
            ("a03_cosine", 44, 4, "run_config.json", lambda d: d.pop("lr_schedule")),  # el finalista sin su calendario
            ("b0", 44, 0, "run_config.json", lambda d: d.update(checkpoint_criterion="min val_weighted_loss (absolute minimum)")),
            ("b2", 44, 1, "run_config.json", lambda d: d.update(epoch_best=40)),  # época fuera de rango
            ("a03_cosine", 43, 2, "run_config.json", lambda d: d.update(csv_path="configs/splits/otra.csv")),
            ("b0", 43, 4, "validation_metrics.json", lambda d: d.update(average_precision=d["average_precision"] + 0.01)),
            ("b2", 43, 0, "training_history.json", lambda d: d["val_loss"].__setitem__(1, float("nan"))),  # NaN en el historial
            ("a03_cosine", 44, 3, "training_history.json", lambda d: d["val_roc_auc"].__setitem__(0, 2.0)),  # otro máximo de AUC
        ],
        ids=["semilla", "fold", "b2_sin_alpha", "b0_con_alpha", "sin_calendario", "checkpoint", "epoca_fuera_de_rango", "particion",
             "ap_no_reproducible", "nan", "epoca_del_checkpoint"],
    )
    def test_integrity_failures_stop_the_analysis_without_writing(self, tmp_path, tamper):
        config, seed, fold, filename, edit = tamper
        build_tree(tmp_path, WINNER)
        path = _run_dir(tmp_path, config, seed, fold) / filename
        data = json.loads(path.read_text(encoding="utf-8"))
        edit(data)
        path.write_text(json.dumps(data), encoding="utf-8")
        code, results = cf.run_analysis(tmp_path, verbose=False)
        assert code == 1 and results is None and not (tmp_path / cf.CONFIRMATION_REL / cf.ANALYSIS_NAME).exists()

    @pytest.mark.parametrize(
        "edit",
        [
            lambda p: p["comandos"][4].update(comando=p["comandos"][4]["comando"].replace("--fold 4", "--fold 3")),
            lambda p: p["comandos"][12].update(comando=p["comandos"][12]["comando"] + " --overwrite"),
            lambda p: p.update(finalista="a01_lr5e5"),
            lambda p: p.update(comandos=p["comandos"][:-1]),
        ],
        ids=["fold_cambiado", "argumento_extra", "finalista_distinto", "faltan_comandos"],
    )
    def test_tampered_commands_file_is_rejected(self, tmp_path, edit):
        build_tree(tmp_path, WINNER)
        path = tmp_path / cf.STAGE1_ANALYSIS_REL / cf.COMMANDS_NAME
        payload = json.loads(path.read_text(encoding="utf-8"))
        edit(payload)
        path.write_text(json.dumps(payload), encoding="utf-8")
        code, results = cf.run_analysis(tmp_path, verbose=False)
        assert code == 1 and results is None and not (tmp_path / cf.CONFIRMATION_REL / cf.ANALYSIS_NAME).exists()

    def test_finalist_must_match_the_stage_1_selection(self, tmp_path):
        build_tree(tmp_path, WINNER)
        path = tmp_path / cf.STAGE1_ANALYSIS_REL / cf.SELECTION_NAME
        path.write_text(json.dumps({"finalista": "a01_lr5e5", "conteo": {"corridas_etapa_1": 45, "fallidas": 0}}), encoding="utf-8")
        assert cf.run_analysis(tmp_path, verbose=False)[0] == 1

    def test_protocol_without_the_amendment_is_rejected(self, tmp_path):
        build_tree(tmp_path, WINNER, protocol_edit=lambda p: p.update(enmiendas=[]))
        assert cf.run_analysis(tmp_path, verbose=False)[0] == 1

    def test_the_test_split_is_never_read(self, tmp_path):
        build_tree(tmp_path, WINNER)
        manifests = list((tmp_path / cf.CONFIRMATION_REL).glob("*/seed_*/fold_*/fold_manifest.csv"))
        assert len(manifests) == 30
        for manifest in manifests:
            rows = read_csv_rows(manifest)
            assert {r["e1_split"] for r in rows} <= {"train", "validation"} and len(rows) == 1000


# ---------------------------------------------------------------------------------------------------------
# Corridas reales (si están en este clon)
# ---------------------------------------------------------------------------------------------------------
@skip_without_real_runs
class TestRealConfirmationRuns:
    def test_real_runs_pass_every_check_and_match_an_independent_recomputation(self, tmp_path):
        code, results = cf.run_analysis(PROJECT_ROOT, output_dir=tmp_path / "salida", verbose=False)
        assert code == 0 and all(c["ok"] for c in results["checks"])
        finalist = results["verdict"]["finalista"]

        def ap(config, seed, fold):
            path = CONFIRMATION_DIR / config / f"seed_{seed}" / f"fold_{fold}" / "validation_metrics.json"
            return json.loads(path.read_text(encoding="utf-8"))["average_precision"]

        mean_delta = {ref: statistics.mean(statistics.mean(ap(finalist, s, k) for s in (43, 44)) - statistics.mean(ap(ref, s, k) for s in (43, 44))
                                           for k in range(5)) for ref in ("b0", "b2")}
        conditions = {c["id"]: c for c in results["verdict"]["condiciones"]}
        assert conditions["C1"]["valor"] == pytest.approx(mean_delta["b0"]) and conditions["C2"]["valor"] == pytest.approx(mean_delta["b2"])
        assert results["verdict"]["veredicto_mecanico"] == ("ganadora" if all(c["cumple"] for c in conditions.values()) else "sin_ganadora")
