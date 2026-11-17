"""
Pruebas del análisis de la fase C (src/classification/agente_fase_c_analysis.py).

Solo datos SINTÉTICOS: una partición pequeña (5 folds de 8 imágenes), 75 métricas de la fase B y 30 corridas de B0 y B2
escritas en una carpeta temporal con el protocolo real de la fase C. No se lee ninguna corrida real.
Las pruebas que necesitan scikit-learn (las funciones del pipeline) usan pytest.importorskip("sklearn").
"""
from __future__ import annotations

import csv
import json
import math
import random
import shutil
import statistics
from pathlib import Path

import pytest

from src.classification import agente_fase_b_analysis as fb
from src.classification import agente_fase_c_analysis as fc
from src.classification import robustez_e2_analysis as e2a

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / fc.PHASE_C_REL / fc.PROTOCOL_NAME
skip_without_protocol = pytest.mark.skipif(not PROTOCOL_PATH.exists(), reason="falta protocolo_fase_c.json (git pull)")

ALPHAS, SEEDS, FOLDS = [0.0, 0.25, 0.5, 0.75, 1.0], [42, 43, 44], [0, 1, 2, 3, 4]


def stdlib_metrics(y, p):
    """Equivalente de biblioteca estándar de AP y ROC-AUC (robustez_e2_analysis), para las pruebas sin scikit-learn."""
    return {"ap": e2a.average_precision(y, p), "roc_auc": e2a.roc_auc(y, p)}


def _write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------------------------------------
# Árbol sintético
# ---------------------------------------------------------------------------------------------------------
def build_tree(root):
    """Protocolo real de la fase C; partición, fase B y 30 corridas de B0 y B2 sintéticas y coherentes entre sí."""
    root = Path(root)
    rng = random.Random(0)
    (root / fc.PHASE_C_REL).mkdir(parents=True)
    shutil.copy(PROTOCOL_PATH, root / fc.PHASE_C_REL / fc.PROTOCOL_NAME)

    # partición: 5 folds de 8 imágenes (3 positivas por fold)
    partition = [{"image_id": str(100 + 8 * k + i), "fold": k, "binary_label": 1 if i < 3 else 0} for k in FOLDS for i in range(8)]
    _write_csv(root / fc.CV_FOLDS_REL, partition, ["image_id", "fold", "binary_label"])

    # fase B: 75 métricas, cada una respaldada por un validation_metrics.json en la ruta del registro
    phase_b = root / fc.PHASE_B_ANALYSIS_REL
    runs, registry, values = [], [], {}
    for alpha in ALPHAS:
        for seed in SEEDS:
            for fold in FOLDS:
                ap, auc = 0.5 + 0.1 * rng.random(), 0.85 + 0.1 * rng.random()
                values[(alpha, seed, fold)] = (ap, auc)
                ruta = f"{fb.PHASE_B_REL.as_posix()}/sinteticas/a{alpha}/seed_{seed}/fold_{fold}"
                (root / ruta).mkdir(parents=True)
                (root / ruta / fc.METRICS_NAME).write_text(json.dumps({"average_precision": ap, "roc_auc": auc}), encoding="utf-8")
                runs.append({"alpha": alpha, "seed": seed, "fold": fold, "ap": repr(ap), "roc_auc": repr(auc)})
                registry.append({"alpha": alpha, "seed": seed, "fold": fold, "estado": "completada", "ruta": ruta})
    _write_csv(phase_b / fb.OUTPUT_FILES["runs"], runs, ["alpha", "seed", "fold", "ap", "roc_auc"])
    _write_csv(phase_b / fb.OUTPUT_FILES["registro"], registry, ["alpha", "seed", "fold", "estado", "ruta"])
    by_fold = [{"alpha": a, "fold": k, "ap_de_entre_semillas": repr(statistics.stdev(values[(a, s, k)][0] for s in SEEDS)),
                "roc_auc_de_entre_semillas": repr(statistics.stdev(values[(a, s, k)][1] for s in SEEDS))} for a in ALPHAS for k in FOLDS]
    _write_csv(phase_b / fb.OUTPUT_FILES["by_fold"], by_fold, ["alpha", "fold", "ap_de_entre_semillas", "roc_auc_de_entre_semillas"])
    (phase_b / fb.OUTPUT_FILES["best_alpha"]).write_text(json.dumps({"piso_de_ruido": {"piso_ap": 0.0949}}), encoding="utf-8")

    # parte 2: 30 corridas en las rutas del protocolo
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    for run in fc.planned_runs(protocol):
        folder = root / run["dir_rel"]
        folder.mkdir(parents=True)
        fold_rows = [r for r in partition if r["fold"] == run["fold"]]
        y = [r["binary_label"] for r in fold_rows]
        p = [min(1.0, max(0.0, 0.35 * label + 0.65 * rng.random())) for label in y]
        predictions = [{"epoch": 4, "image_id": r["image_id"], "y_true": label, "probability_malignant": repr(prob)}
                       for r, label, prob in zip(fold_rows, y, p)]
        _write_csv(folder / fc.PREDICTIONS_NAME, predictions, list(fc.PREDICTION_COLUMNS))
        metrics = stdlib_metrics(y, p)
        (folder / fc.METRICS_NAME).write_text(json.dumps({"average_precision": metrics["ap"], "roc_auc": metrics["roc_auc"]}), encoding="utf-8")
        config = {"seed": run["seed"], "fold": run["fold"], "epoch_best": 4}
        if run["config"] == "B2":
            config["class_weight_alpha"] = 0.5
        (folder / fc.CONFIG_NAME).write_text(json.dumps(config), encoding="utf-8")
    return protocol


def _run_dir(root, config="B0", seed=43, fold=0):
    if seed == 42:
        name = "convnext_a_alpha1" if config == "B0" else "convnext_a_alpha05"
        return Path(root) / "results/classification/r7_optimizacion/robustez_e2/corridas" / name / f"fold_{fold}"
    return Path(root) / "results/classification/r7_optimizacion/agente/confirmacion" / config.lower() / f"seed_{seed}" / f"fold_{fold}"


def _edit_predictions(path, edit):
    rows = _read_csv(path)
    edit(rows)
    _write_csv(path, rows, list(fc.PREDICTION_COLUMNS))


def _out(root):
    return Path(root) / fc.PHASE_C_REL / fc.ANALYSIS_NAME


def _assert_stopped_without_outputs(root):
    code, results = fc.run_analysis(root, verbose=False)
    assert code == 1 and results is None
    assert not _out(root).exists()  # ninguna salida, ni parcial


# ---------------------------------------------------------------------------------------------------------
# Rutas del protocolo
# ---------------------------------------------------------------------------------------------------------
class TestRoutes:
    def test_expand_braces(self):
        assert fc.expand_braces("a/fold_{0..2}/") == ["a/fold_0/", "a/fold_1/", "a/fold_2/"]
        assert fc.expand_braces("b/seed_{43,44}/fold_{0..1}") == ["b/seed_43/fold_0", "b/seed_43/fold_1", "b/seed_44/fold_0", "b/seed_44/fold_1"]

    @skip_without_protocol
    def test_planned_runs_from_the_protocol(self):
        runs = fc.planned_runs(json.loads(PROTOCOL_PATH.read_text(encoding="utf-8")))
        assert len(runs) == 30 and len({r["dir_rel"] for r in runs}) == 30
        assert {(r["config"], r["seed"], r["fold"]) for r in runs} == {(c, s, k) for c in ("B0", "B2") for s in SEEDS for k in FOLDS}
        b0_42 = [r["dir_rel"] for r in runs if r["config"] == "B0" and r["seed"] == 42]
        assert b0_42[0] == "results/classification/r7_optimizacion/robustez_e2/corridas/convnext_a_alpha1/fold_0"
        b2_44 = [r["dir_rel"] for r in runs if r["config"] == "B2" and r["seed"] == 44]
        assert b2_44[4] == "results/classification/r7_optimizacion/agente/confirmacion/b2/seed_44/fold_4"


# ---------------------------------------------------------------------------------------------------------
# Parte 1: reparto de la varianza con resultado conocido
# ---------------------------------------------------------------------------------------------------------
FOLD_EFFECT = [-0.02, -0.01, 0.0, 0.01, 0.02]
ALPHA_EFFECT = [-0.04, -0.02, 0.0, 0.02, 0.04]
SEED_PATTERN = [-1.0, 0.0, 1.0]  # suma cero dentro de cada α
R = [1.0, -2.0, 1.0]  # residual r_s · q_k: suma cero sobre semillas y sobre folds
Q = [1.0, -1.0, 0.0, 0.0, 0.0]


def _known_values(seed_scale=0.01, residual_scale=0.001):
    values = {}
    for i, alpha in enumerate(ALPHAS):
        for j, seed in enumerate(SEEDS):
            for k in FOLDS:
                values[(alpha, seed, k)] = (0.6 + FOLD_EFFECT[k] + ALPHA_EFFECT[i] + seed_scale * (i + 1) * SEED_PATTERN[j]
                                            + residual_scale * R[j] * Q[k])
    return values


class TestVarianceDecomposition:
    def test_sums_of_squares_degrees_of_freedom_and_components_match_the_construction(self):
        rows = {r["fuente"]: r for r in fc.variance_decomposition(_known_values())}
        a, s, k = 5, 3, 5
        ss_fold = a * s * sum(x ** 2 for x in FOLD_EFFECT)
        ss_alpha = s * k * sum(x ** 2 for x in ALPHA_EFFECT)
        ss_seed = k * sum((0.01 * (i + 1) * p) ** 2 for i in range(a) for p in SEED_PATTERN)
        ss_res = a * sum((0.001 * r * q) ** 2 for r in R for q in Q)
        assert rows["fold"]["SS"] == pytest.approx(ss_fold, abs=1e-12)
        assert rows["alpha"]["SS"] == pytest.approx(ss_alpha, abs=1e-12)
        assert rows["semilla_dentro_de_alpha"]["SS"] == pytest.approx(ss_seed, abs=1e-12)
        assert rows["residual"]["SS"] == pytest.approx(ss_res, abs=1e-12)
        assert rows["total"]["SS"] == pytest.approx(ss_fold + ss_alpha + ss_seed + ss_res, abs=1e-12)
        assert [rows[n]["gl"] for n in ("fold", "alpha", "semilla_dentro_de_alpha", "residual", "total")] == [4, 4, 10, 56, 74]

        ms = {n: rows[n]["SS"] / rows[n]["gl"] for n in fc.SOURCES}
        assert rows["residual"]["componente"] == pytest.approx(ms["residual"], abs=1e-15)
        assert rows["semilla_dentro_de_alpha"]["componente"] == pytest.approx((ms["semilla_dentro_de_alpha"] - ms["residual"]) / k, abs=1e-15)
        assert rows["fold"]["componente"] == pytest.approx((ms["fold"] - ms["residual"]) / (a * s), abs=1e-15)
        assert rows["alpha"]["componente"] == pytest.approx((ms["alpha"] - ms["semilla_dentro_de_alpha"]) / (s * k), abs=1e-15)
        assert sum(rows[n]["porcentaje_SS"] for n in fc.SOURCES) == pytest.approx(100.0)
        assert sum(rows[n]["porcentaje_componente"] for n in fc.SOURCES) == pytest.approx(100.0)
        assert not any(rows[n]["negativa"] for n in fc.SOURCES)

    def test_a_negative_component_is_reported_untruncated_and_counts_as_zero_in_percentages(self):
        rows = {r["fuente"]: r for r in fc.variance_decomposition(_known_values(seed_scale=0.0, residual_scale=0.01))}
        seed = rows["semilla_dentro_de_alpha"]
        assert seed["SS"] == pytest.approx(0.0, abs=1e-15) and seed["componente"] < 0 and seed["negativa"] is True
        assert seed["componente_truncada"] == 0.0 and seed["porcentaje_componente"] == 0.0
        assert sum(rows[n]["porcentaje_componente"] for n in fc.SOURCES) == pytest.approx(100.0)

    def test_an_unbalanced_design_is_rejected(self):
        values = _known_values()
        del values[(0.5, 43, 2)]
        with pytest.raises(fc.IntegrityError):
            fc.variance_decomposition(values)


class TestBetweenSeeds:
    def test_sd_by_fold_pooled_variance_and_sd_b(self):
        values = {(a, s, k): 0.5 for a in ALPHAS for s in SEEDS for k in FOLDS}
        values[(0.0, 42, 0)], values[(0.0, 43, 0)], values[(0.0, 44, 0)] = 0.4, 0.5, 0.6  # DE = 0.1 en (α = 0, fold 0)
        result = fc.between_seed_sd(values)
        assert result["sd"][(0.0, 0)] == pytest.approx(0.1) and result["sd"][(0.5, 3)] == 0.0
        assert result["media_por_alpha"][0.0] == pytest.approx(0.1 / 5)
        assert result["s2_p"] == pytest.approx(0.01 / 25)  # una varianza de 0.01 y 24 nulas
        assert result["sd_b"] == pytest.approx(math.sqrt(2 * 0.01 / 25 / 3))


# ---------------------------------------------------------------------------------------------------------
# Parte 2: verificaciones, alineación y ensamble a mano
# ---------------------------------------------------------------------------------------------------------
def _rows(ids, labels, probs, epoch=4):
    return [{"epoch": str(epoch), "image_id": i, "y_true": str(y), "probability_malignant": repr(p)} for i, y, p in zip(ids, labels, probs)]


IDS, Y = ["a", "b", "c", "d"], [1, 0, 1, 0]
P42, P43, P44 = [0.9, 0.6, 0.5, 0.1], [0.6, 0.3, 0.9, 0.4], [0.6, 0.3, 0.1, 0.4]


class TestPredictionChecks:
    COLUMNS = list(fc.PREDICTION_COLUMNS)

    def test_valid_rows_pass(self):
        assert fc.check_prediction_rows(_rows(IDS, Y, P42), self.COLUMNS, 4) == []

    @pytest.mark.parametrize(
        "rows, columns, epoch_best",
        [
            (_rows(IDS, Y, [0.9, 1.2, 0.5, 0.1]), list(fc.PREDICTION_COLUMNS), 4),  # probabilidad > 1
            (_rows(IDS, Y, [0.9, -0.1, 0.5, 0.1]), list(fc.PREDICTION_COLUMNS), 4),  # probabilidad < 0
            (_rows(IDS, Y, [0.9, float("nan"), 0.5, 0.1]), list(fc.PREDICTION_COLUMNS), 4),  # no finita
            (_rows(IDS, [1, 0, 2, 0], P42), list(fc.PREDICTION_COLUMNS), 4),  # y_true fuera de {0, 1}
            (_rows(IDS, Y, P42, epoch=3), list(fc.PREDICTION_COLUMNS), 4),  # epoch distinto de epoch_best
            (_rows(IDS, Y, P42), ["epoch", "image_id", "y_true", "prob"], 4),  # columna renombrada
        ],
        ids=["mayor_que_1", "menor_que_0", "nan", "y_true_2", "epoch", "columna"],
    )
    def test_invalid_rows_are_reported(self, rows, columns, epoch_best):
        assert fc.check_prediction_rows(rows, columns, epoch_best)


class TestAlignment:
    def test_different_order_is_aligned_by_image_id(self):
        shuffled = list(reversed(_rows(IDS, Y, P43)))
        result = fc.align_seed_predictions({42: _rows(IDS, Y, P42), 43: shuffled, 44: _rows(IDS, Y, P44)})
        assert result["orden_distinto"] is True and result["image_ids"] == IDS and result["y_true"] == Y
        assert result["probabilities"][43] == P43  # reordenada al orden de la semilla 42

    @pytest.mark.parametrize(
        "seed_43_rows, message",
        [
            (_rows(["a", "b", "c", "z"], Y, P43), "image_id distintos"),
            (_rows(["a", "b", "c", "c"], Y, P43), "repetidos"),
            (_rows(IDS, [1, 0, 0, 0], P43), "y_true distinto"),
        ],
        ids=["image_id_distintos", "duplicados", "y_true_distinto"],
    )
    def test_integrity_errors(self, seed_43_rows, message):
        with pytest.raises(fc.IntegrityError, match=message):
            fc.align_seed_predictions({42: _rows(IDS, Y, P42), 43: seed_43_rows, 44: _rows(IDS, Y, P44)})


class TestEnsembleByHand:
    """
    p_ens = [(0.9+0.6+0.6)/3, (0.6+0.3+0.3)/3, (0.5+0.9+0.1)/3, (0.1+0.4+0.4)/3] = [0.7, 0.4, 0.5, 0.3]: positivas (a, c)
    por encima de las negativas -> AP = 1 y ROC-AUC = 1. Individuales: semilla 42 AP = 5/6, AUC = 3/4; semilla 43 AP = 1,
    AUC = 1; semilla 44 AP = 3/4, AUC = 1/2. Δ AP = 1 - (5/6 + 1 + 3/4)/3 = 5/36; Δ ROC-AUC = 1 - 3/4 = 1/4.
    """

    @staticmethod
    def _check(result):
        assert all(abs(x - expected) < 1e-12 for x, expected in zip(result["p_ens"], [0.7, 0.4, 0.5, 0.3]))
        assert result["ensamble"]["ap"] == pytest.approx(1.0) and result["ensamble"]["roc_auc"] == pytest.approx(1.0)
        assert result["individual"][42]["ap"] == pytest.approx(5 / 6) and result["individual"][44]["roc_auc"] == pytest.approx(0.5)
        assert result["delta_ap"] == pytest.approx(5 / 36) and result["delta_roc_auc"] == pytest.approx(0.25)

    def test_ensemble_with_the_stdlib_equivalent(self):
        self._check(fc.ensemble_fold(Y, {42: P42, 43: P43, 44: P44}, stdlib_metrics))

    def test_ensemble_with_the_pipeline_functions(self):
        pytest.importorskip("sklearn")
        self._check(fc.ensemble_fold(Y, {42: P42, 43: P43, 44: P44}, fc.pipeline_metrics))

    def test_summary_of_deltas_has_signs_and_counts_only(self):
        s = fc.summarize_deltas([0.1, -0.05, 0.0, 0.2, 0.05])
        assert s["signos"] == ["+", "-", "0", "+", "+"] and (s["positivos"], s["negativos"], s["ceros"]) == (3, 1, 1)
        assert s["media"] == pytest.approx(0.06) and s["minimo"] == -0.05 and s["maximo"] == 0.2
        assert not any(k in s for k in ("veredicto", "mejor_consistente", "p_valor", "ic95"))


# ---------------------------------------------------------------------------------------------------------
# Análisis completo sobre el árbol sintético
# ---------------------------------------------------------------------------------------------------------
@skip_without_protocol
class TestRunAnalysisStops:
    """Fallas que detienen el análisis antes de necesitar scikit-learn: corren sin él."""

    def test_part_1_value_that_does_not_match_validation_metrics(self, tmp_path):
        build_tree(tmp_path)
        path = tmp_path / fb.PHASE_B_REL / "sinteticas" / "a0.25" / "seed_43" / "fold_2" / fc.METRICS_NAME
        data = json.loads(path.read_text(encoding="utf-8"))
        data["average_precision"] += 0.01
        path.write_text(json.dumps(data), encoding="utf-8")
        _assert_stopped_without_outputs(tmp_path)

    def test_part_1_missing_row(self, tmp_path):
        build_tree(tmp_path)
        path = tmp_path / fc.PHASE_B_ANALYSIS_REL / fb.OUTPUT_FILES["runs"]
        rows = _read_csv(path)[:-1]
        _write_csv(path, rows, ["alpha", "seed", "fold", "ap", "roc_auc"])
        _assert_stopped_without_outputs(tmp_path)

    @pytest.mark.parametrize(
        "config, seed, fold, edit",
        [
            ("B0", 44, 1, lambda rows: rows[0].update(image_id="999")),  # image_id distintos
            ("B2", 43, 2, lambda rows: rows[1].update(image_id=rows[0]["image_id"])),  # duplicado
            ("B0", 43, 3, lambda rows: rows[0].update(y_true="0" if rows[0]["y_true"] == "1" else "1")),  # y_true distinto
            ("B2", 42, 4, lambda rows: rows[2].update(probability_malignant="1.5")),  # probabilidad fuera de [0, 1]
            ("B0", 42, 0, lambda rows: rows[3].update(probability_malignant="nan")),  # no finita
            ("B2", 44, 0, lambda rows: [r.update(epoch="3") for r in rows]),  # epoch distinto de epoch_best
        ],
        ids=["image_id_distintos", "duplicados", "y_true_distinto", "probabilidad_fuera_de_rango", "nan", "epoch"],
    )
    def test_part_2_prediction_failures(self, tmp_path, config, seed, fold, edit):
        build_tree(tmp_path)
        _edit_predictions(_run_dir(tmp_path, config, seed, fold) / fc.PREDICTIONS_NAME, edit)
        _assert_stopped_without_outputs(tmp_path)

    @pytest.mark.parametrize(
        "config, seed, fold, edit",
        [
            ("B0", 43, 1, lambda d: d.update(class_weight_alpha=0.5)),  # B0 con α registrado
            ("B2", 44, 2, lambda d: d.pop("class_weight_alpha")),  # B2 sin α
            ("B0", 44, 3, lambda d: d.update(seed=42)),
            ("B2", 43, 4, lambda d: d.update(fold=0)),
        ],
        ids=["b0_con_alpha", "b2_sin_alpha", "semilla", "fold"],
    )
    def test_part_2_config_failures(self, tmp_path, config, seed, fold, edit):
        build_tree(tmp_path)
        path = _run_dir(tmp_path, config, seed, fold) / fc.CONFIG_NAME
        data = json.loads(path.read_text(encoding="utf-8"))
        edit(data)
        path.write_text(json.dumps(data), encoding="utf-8")
        _assert_stopped_without_outputs(tmp_path)

    def test_part_2_missing_folder(self, tmp_path):
        build_tree(tmp_path)
        shutil.rmtree(_run_dir(tmp_path, "B2", 43, 1))
        _assert_stopped_without_outputs(tmp_path)

    def test_existing_outputs_are_not_overwritten(self, tmp_path):
        build_tree(tmp_path)
        _out(tmp_path).mkdir(parents=True)
        (_out(tmp_path) / fc.OUTPUT_FILES["summary"]).write_text("{}", encoding="utf-8")
        with pytest.raises(FileExistsError):
            fc.run_analysis(tmp_path, verbose=False)
        assert [p.name for p in _out(tmp_path).iterdir()] == [fc.OUTPUT_FILES["summary"]]


@skip_without_protocol
class TestRunAnalysisWithPipeline:
    """Análisis completo y verificación de las métricas recalculadas: necesitan scikit-learn."""

    def test_full_analysis_writes_the_seven_outputs(self, tmp_path):
        pytest.importorskip("sklearn")
        build_tree(tmp_path)
        code, results = fc.run_analysis(tmp_path, verbose=True)
        out = _out(tmp_path)
        assert code == 0 and all(c["ok"] for c in results["checks"])
        assert {p.name for p in out.iterdir()} == set(fc.OUTPUT_FILES.values())

        assert len(_read_csv(out / fc.OUTPUT_FILES["components"])) == 2 * 5  # AP y ROC-AUC × 4 fuentes + total
        by_fold = _read_csv(out / fc.OUTPUT_FILES["ensemble_by_fold"])
        assert len(by_fold) == 10 and {"ap_semilla_42", "ap_media_individual", "ap_ensamble", "delta_ap", "signo_ap", "delta_roc_auc"} <= set(by_fold[0])
        summary = _read_csv(out / fc.OUTPUT_FILES["ensemble_summary"])
        assert len(summary) == 4 and set(summary[0]) == {"configuracion", "metrica", "media", "minimo", "maximo", "positivos", "negativos", "ceros"}
        for row in by_fold:  # Δ = ensamble - media de las 3 semillas
            assert float(row["delta_ap"]) == pytest.approx(float(row["ap_ensamble"]) - statistics.mean(float(row[f"ap_semilla_{s}"]) for s in SEEDS))

        payload = json.loads((out / fc.OUTPUT_FILES["summary"]).read_text(encoding="utf-8"))
        assert payload["rotulo_obligatorio"].startswith("exploratorio")
        assert set(payload["parte_2"]["delta_por_fold"]) == {"B0", "B2"}
        assert all(len(payload["parte_2"]["delta_por_fold"][c][m]["deltas"]) == 5 for c in ("B0", "B2") for m in ("AP", "ROC-AUC"))
        noise = json.loads((out / fc.OUTPUT_FILES["noise_b"]).read_text(encoding="utf-8"))
        assert noise["AP"]["cociente_piso_ap_sobre_sd_b"] == pytest.approx(0.0949 / noise["AP"]["sd_b"])
        text = json.dumps(payload) + "".join((out / name).read_text(encoding="utf-8") for name in fc.OUTPUT_FILES.values() if name.endswith(".csv"))
        assert "p_valor" not in text and "mejor_consistente" not in text and "ic95" not in text.lower()

    def test_different_order_between_seeds_does_not_stop_the_analysis(self, tmp_path):
        pytest.importorskip("sklearn")
        build_tree(tmp_path)
        _edit_predictions(_run_dir(tmp_path, "B0", 44, 2) / fc.PREDICTIONS_NAME, lambda rows: rows.reverse())
        code, results = fc.run_analysis(tmp_path, verbose=False)
        assert code == 0 and results["summary"]["parte_2"]["folds_con_orden_distinto"] == ["B0_fold2"]

    def test_recomputed_ap_that_does_not_match_stops_without_outputs(self, tmp_path):
        pytest.importorskip("sklearn")
        build_tree(tmp_path)
        path = _run_dir(tmp_path, "B2", 44, 3) / fc.METRICS_NAME
        data = json.loads(path.read_text(encoding="utf-8"))
        data["average_precision"] += 1e-4  # por encima de la tolerancia de 1e-6
        path.write_text(json.dumps(data), encoding="utf-8")
        _assert_stopped_without_outputs(tmp_path)  # la última verificación: tampoco deja salidas parciales
