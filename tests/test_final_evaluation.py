"""
Tests unitarios del evaluador final de R8 (src/classification/final_evaluation.py).

Usan datos sintéticos y los archivos de resultados ya existentes (solo lectura). Ningún test abre el conjunto test.
"""
from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.classification import final_evaluation as fe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (PROJECT_ROOT / fe.PROTOCOL_REL).is_file(), reason="protocolo congelado de R7 no disponible")


@pytest.fixture(scope="module")
def protocol():
    return fe.load_protocol(PROJECT_ROOT)


def _run_config(protocol, config, seed):
    return json.loads(fe.run_paths(PROJECT_ROOT, protocol, config, seed)["run_config"].read_text(encoding="utf-8"))


def _synthetic(n_pos=12, n_neg=60, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([1] * n_pos + [0] * n_neg)
    prob = np.clip(np.where(y == 1, rng.beta(4, 3, y.size), rng.beta(2, 6, y.size)), 1e-4, 1 - 1e-4)
    return y, prob


class TestProtocol:
    def test_frozen_protocol_is_valid(self, protocol):
        assert fe.validate_protocol(protocol) == []
        assert protocol["umbral"] == 0.5 and protocol["modelos"]["semillas"] == [42, 43, 44]

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda p: p.update(umbral=0.4),
            lambda p: p["modelos"].update(semillas=[42, 43]),
            lambda p: p["modelos"].update(ensambles=True),
            lambda p: p["comparacion_b2_vs_b0"].update(n_bootstrap=1000),
            lambda p: p["comparacion_b2_vs_b0"].update(semilla_bootstrap=1),
            lambda p: p["datos"].update(augmentation=True),
        ],
    )
    def test_modified_protocol_is_rejected(self, protocol, mutate):
        changed = copy.deepcopy(protocol)
        mutate(changed)
        assert fe.validate_protocol(changed) != []

    def test_run_config_of_all_six_runs_matches_protocol(self, protocol):
        for config, seed in fe.iter_runs(protocol):
            assert fe.check_run_config(_run_config(protocol, config, seed), protocol, config, seed) == [], (config, seed)

    @pytest.mark.parametrize("key, value", [("seed", 99), ("threshold", 0.4), ("class_weights", [1.0, 1.0]), ("condition", "B"), ("csv_path", "otro.csv")])
    def test_incompatible_run_config_is_reported(self, protocol, key, value):
        run_config = _run_config(protocol, "B2", 42)
        run_config[key] = value
        assert fe.check_run_config(run_config, protocol, "B2", 42) != []

    def test_b2_weights_cannot_be_checked_as_b0(self, protocol):
        assert fe.check_run_config(_run_config(protocol, "B2", 42), protocol, "B0", 42) != []

    def test_b0_and_b2_share_hyperparameters_and_differences_are_detected(self, protocol):
        b0, b2 = _run_config(protocol, "B0", 42), _run_config(protocol, "B2", 42)
        assert fe.check_frozen_hyperparameters(b0, b2) == []
        b2["learning_rate"] = 1e-3
        assert fe.check_frozen_hyperparameters(b0, b2) != []

    def test_checkpoint_meta_consistency(self, protocol):
        run_config = _run_config(protocol, "B0", 42)
        meta = json.loads(fe.run_paths(PROJECT_ROOT, protocol, "B0", 42)["meta"].read_text(encoding="utf-8"))
        assert fe.check_meta(meta, run_config) == []
        assert fe.check_meta({**meta, "epoch": meta["epoch"] + 1}, run_config) != []
        assert fe.check_meta({**meta, "criterion": "min val_loss"}, run_config) != []

    def test_precheck_has_no_failures_with_existing_files(self, protocol):
        report = fe.precheck(PROJECT_ROOT, protocol, with_hashes=False)
        assert report["estado"] in ("completo", "pendiente")  # 'pendiente' si faltan checkpoints en este entorno
        assert not [item for item in report["comprobaciones"] if item["estado"] == "falla"]


class TestComparison:
    def _frames(self):
        y, prob = _synthetic()
        saved = pd.DataFrame({"epoch": 7, "image_id": np.arange(100, 100 + y.size), "y_true": y, "probability_malignant": prob})
        return saved.copy(), saved

    def test_identical_predictions_are_approved(self):
        new, saved = self._frames()
        result = fe.compare_predictions(new, saved)
        assert result["aprobada"] and result["max_abs_diff"] == 0.0 and result["cambios_de_clase_al_umbral_0.5"] == 0

    def test_alignment_is_by_image_id_not_row_order(self):
        new, saved = self._frames()
        shuffled = new.sample(frac=1.0, random_state=1).reset_index(drop=True)
        result = fe.compare_predictions(shuffled, saved)
        assert result["aprobada"] and result["max_abs_diff"] == 0.0

    def test_difference_above_tolerance_is_reported_not_hidden(self):
        new, saved = self._frames()
        new.loc[3, "probability_malignant"] = np.clip(new.loc[3, "probability_malignant"] + 1e-3, 0, 1)
        result = fe.compare_predictions(new, saved)
        assert not result["aprobada"] and result["max_abs_diff"] > fe.PROB_TOLERANCE and result["n_sobre_tolerancia"] == 1
        assert result["image_id_max_diff"] == int(saved.loc[3, "image_id"])

    def test_small_difference_within_tolerance_is_approved_and_still_reported(self):
        new, saved = self._frames()
        new["probability_malignant"] = np.clip(new["probability_malignant"] + 1e-6, 0, 1)
        result = fe.compare_predictions(new, saved)
        assert result["aprobada"] and 0 < result["max_abs_diff"] <= fe.PROB_TOLERANCE

    def test_class_flip_at_threshold_fails_even_with_tiny_difference(self):
        new, saved = self._frames()
        saved.loc[0, "probability_malignant"] = 0.5 - 1e-9
        new.loc[0, "probability_malignant"] = 0.5
        result = fe.compare_predictions(new, saved, tolerance=1e-3)
        assert result["cambios_de_clase_al_umbral_0.5"] == 1 and not result["aprobada"]

    def test_different_image_sets_and_labels_fail(self):
        new, saved = self._frames()
        assert not fe.compare_predictions(new.iloc[:-1], saved)["aprobada"]
        flipped = new.copy()
        flipped.loc[0, "y_true"] = 1 - flipped.loc[0, "y_true"]
        assert not fe.compare_predictions(flipped, saved)["aprobada"]

    def test_duplicated_image_id_is_an_error(self):
        new, saved = self._frames()
        new.loc[1, "image_id"] = new.loc[0, "image_id"]
        with pytest.raises(ValueError):
            fe.compare_predictions(new, saved)

    def test_saved_metrics_of_b0_are_reproduced_from_saved_predictions(self, protocol):
        paths = fe.run_paths(PROJECT_ROOT, protocol, "B0", 42)
        frame = pd.read_csv(paths["predictions"])
        saved = json.loads(paths["metrics"].read_text(encoding="utf-8"))
        metrics = fe.compute_metrics(frame["y_true"], frame["probability_malignant"])
        assert metrics["average_precision"] == pytest.approx(saved["average_precision"], abs=1e-9)
        assert metrics["roc_auc"] == pytest.approx(saved["roc_auc"], abs=1e-9)
        assert metrics["sensibilidad"] == pytest.approx(saved["recall_malignant"], abs=1e-9)
        assert metrics["especificidad"] == pytest.approx(saved["specificity"], abs=1e-9)
        assert [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]] == saved["confusion_matrix"]


class TestMetricsAndBootstrap:
    def test_threshold_metrics_on_a_known_case(self):
        y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        prob = np.array([0.9, 0.5, 0.4, 0.1, 0.6, 0.2, 0.2, 0.1, 0.1, 0.0])
        m = fe.compute_metrics(y, prob)
        assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 2, 1, 5)  # 0.5 cuenta como positivo
        assert m["sensibilidad"] == 0.5 and m["especificidad"] == pytest.approx(5 / 6)
        assert m["precision"] == pytest.approx(2 / 3) and m["balanced_accuracy"] == pytest.approx((0.5 + 5 / 6) / 2)
        assert m["f1"] == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))

    def test_bootstrap_is_reproducible_and_identical_models_give_zero(self):
        y, prob = _synthetic()
        probs = np.stack([prob, np.clip(prob * 0.9, 0, 1), np.clip(prob + 0.01, 0, 1)])
        first = fe.paired_bootstrap_delta_ap(y, probs, probs, n_boot=50, seed=7)
        again = fe.paired_bootstrap_delta_ap(y, probs, probs, n_boot=50, seed=7)
        assert first == again
        assert first["delta_ap_medio_observado"] == 0 and first["ic_inf"] == 0 and first["ic_sup"] == 0

    def test_bootstrap_detects_a_clearly_better_model_and_depends_on_seed(self):
        y, prob = _synthetic(seed=3)
        rng = np.random.default_rng(5)
        b0 = np.stack([rng.uniform(size=y.size) for _ in range(3)])  # sin información
        b2 = np.stack([prob, prob, prob])
        result = fe.paired_bootstrap_delta_ap(y, b0, b2, n_boot=200, seed=1)
        assert result["delta_ap_medio_observado"] > 0 and result["ic_inf"] > 0 and result["fraccion_replicas_delta_positivo"] == 1.0
        assert len(result["delta_ap_por_semilla"]) == 3
        assert fe.paired_bootstrap_delta_ap(y, b0, b2, n_boot=200, seed=2)["ic_inf"] != result["ic_inf"]

    def test_bootstrap_rejects_inconsistent_shapes(self):
        y, prob = _synthetic()
        with pytest.raises(ValueError):
            fe.paired_bootstrap_delta_ap(y, np.stack([prob, prob]), np.stack([prob]), n_boot=5, seed=1)


class _FakeDataset(Dataset):
    """Dataset sintético con la interfaz que usa predict_dataset: .df con image_id y binary_label."""

    def __init__(self, n=10):
        rng = np.random.default_rng(0)
        self.x = torch.tensor(rng.normal(size=(n, 3, 4, 4)), dtype=torch.float32)
        self.df = pd.DataFrame({"image_id": np.arange(500, 500 + n)[::-1], "binary_label": (np.arange(n) % 2)})

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.x[idx], int(self.df["binary_label"].iloc[idx])


class TestInference:
    def test_predict_dataset_keeps_image_ids_and_valid_probabilities(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Flatten(), nn.Linear(48, 2))
        dataset = _FakeDataset()
        frame = fe.predict_dataset(model, dataset, batch_size=4, device=torch.device("cpu"))
        assert frame["image_id"].tolist() == dataset.df["image_id"].tolist()
        assert frame["y_true"].tolist() == dataset.df["binary_label"].tolist()
        assert frame["probability_malignant"].between(0, 1).all() and len(frame) == len(dataset)

    def test_predict_dataset_is_deterministic_in_eval_mode(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(48, 2))
        dataset = _FakeDataset()
        first = fe.predict_dataset(model, dataset, 4, torch.device("cpu"))
        second = fe.predict_dataset(model, dataset, 3, torch.device("cpu"))
        np.testing.assert_allclose(first["probability_malignant"], second["probability_malignant"], atol=1e-6)

    def test_validation_split_uses_validation_transforms_without_augmentation(self):
        pytest.importorskip("torchvision")
        dataset = fe.build_dataset(PROJECT_ROOT, "validation")
        names = [type(t).__name__ for t in dataset.transform.transforms]
        assert "RandomHorizontalFlip" not in names and "RandomAffine" not in names and "ColorJitter" not in names
        assert set(dataset.df["experimental_split"]) == {"validation"} and len(dataset) == 200

    def test_dataset_uses_the_manifest_of_training(self):
        pytest.importorskip("torchvision")
        run_experiment = pytest.importorskip("src.classification.run_experiment")
        assert fe.PARTITION_REL == run_experiment.CSV_REL
        assert fe.IMAGE_DIR_REL == run_experiment.CONDITION_DIRS["A"]


class TestTestModeProtection:
    def test_test_split_cannot_be_built_without_explicit_permission(self):
        with pytest.raises(PermissionError):
            fe.build_dataset(PROJECT_ROOT, "test")
        with pytest.raises(ValueError):
            fe.build_dataset(PROJECT_ROOT, "otro")

    def test_validation_verification_code_never_references_test(self):
        source = inspect.getsource(fe.verify_validation) + inspect.getsource(fe.precheck)
        assert '"test"' not in source and "'test'" not in source and "allow_test" not in source

    def test_gates_refuse_without_confirmation_registro_and_verification(self, protocol):
        with pytest.raises(PermissionError) as error:
            fe.assert_test_gates(PROJECT_ROOT, protocol, None, None, None)
        message = str(error.value)
        assert "confirmación explícita" in message and "registro previo" in message and "verificación de validation" in message

    def test_wrong_confirmation_text_is_refused(self, protocol, tmp_path):
        with pytest.raises(PermissionError, match="confirmación explícita"):
            fe.assert_test_gates(PROJECT_ROOT, protocol, "si", None, None)

    def test_partial_verification_does_not_enable_test(self, protocol, tmp_path):
        verification = tmp_path / "verificacion.json"
        verification.write_text(json.dumps({"estado": "parcial", "tolerancia_estandar": True}), encoding="utf-8")
        registro = tmp_path / "registro.json"
        registro.write_text(json.dumps(fe.build_registro_previo(PROJECT_ROOT, protocol, with_hashes=False)), encoding="utf-8")
        with pytest.raises(PermissionError) as error:
            fe.assert_test_gates(PROJECT_ROOT, protocol, fe.TEST_CONFIRMATION, registro, verification)
        assert "no está superada" in str(error.value)

    def test_nonstandard_tolerance_does_not_enable_test(self, protocol, tmp_path):
        verification = tmp_path / "verificacion.json"
        verification.write_text(json.dumps({"estado": "superada", "tolerancia_estandar": False}), encoding="utf-8")
        with pytest.raises(PermissionError, match="tolerancia no estándar"):
            fe.assert_test_gates(PROJECT_ROOT, protocol, fe.TEST_CONFIRMATION, None, verification)

    def test_registro_mismatch_is_detected(self, protocol):
        current = fe.build_registro_previo(PROJECT_ROOT, protocol, with_hashes=False)
        frozen = copy.deepcopy(current)
        frozen["git"]["commit"] = "0" * 40
        frozen["sha256_particiones"]["operativa"]["sha256"] = "0" * 64
        problems = fe.compare_registros(frozen, current)
        assert any("commit" in p for p in problems) and any("partición" in p for p in problems)

    def test_existing_test_output_folder_blocks_a_second_run(self, protocol, tmp_path, monkeypatch):
        (tmp_path / fe.R8_RESULTS_REL / "test").mkdir(parents=True)
        with pytest.raises(PermissionError, match="una sola vez"):
            fe.assert_test_gates(tmp_path, protocol, fe.TEST_CONFIRMATION, None, None)

    def test_prepare_output_dir_never_overwrites(self, tmp_path):
        target = tmp_path / "salida"
        fe.prepare_output_dir(target)
        (target / "archivo.txt").write_text("x", encoding="utf-8")
        with pytest.raises(FileExistsError):
            fe.prepare_output_dir(target)

    def test_cli_test_mode_requires_explicit_confirmation(self, capsys):
        code = fe.main(["--project-root", str(PROJECT_ROOT), "test", "--registro-previo", "no_existe.json", "--verificacion-validation", "no_existe.json"])
        assert code == 2
        assert "Test NO habilitado" in capsys.readouterr().err
        assert not (PROJECT_ROOT / fe.R8_RESULTS_REL / "test").exists()

    def test_cli_verify_validation_requires_a_label(self):
        with pytest.raises(SystemExit):
            fe.main(["--project-root", str(PROJECT_ROOT), "verify-validation"])


class TestRunTestFlow:
    """Flujo completo del modo test con datos sintéticos y sin abrir el test real (se sustituyen dataset y modelos)."""

    @pytest.fixture
    def synthetic_project(self, protocol, tmp_path, monkeypatch):
        for config, seed in fe.iter_runs(protocol):
            folder = fe.run_paths(tmp_path, protocol, config, seed)["dir"]
            folder.mkdir(parents=True)
            (folder / "run_config.json").write_text(json.dumps({"batch_size": 8}), encoding="utf-8")
        y, prob = _synthetic(n_pos=15, n_neg=60, seed=11)
        offsets = {("B0", 42): 0.0, ("B0", 43): 0.02, ("B0", 44): -0.02, ("B2", 42): 0.03, ("B2", 43): 0.05, ("B2", 44): 0.01}

        def fake_predict(model, dataset, batch_size, device, num_workers=0):
            noise = np.random.default_rng(abs(hash(model)) % 1000).normal(scale=0.05, size=y.size)
            return pd.DataFrame({"image_id": np.arange(y.size), "y_true": y, "probability_malignant": np.clip(prob + offsets[model] + noise, 0, 1)})

        monkeypatch.setattr(fe, "assert_test_gates", lambda *args, **kwargs: None)
        monkeypatch.setattr(fe, "build_dataset", lambda root, split, allow_test=False: object())
        monkeypatch.setattr(fe, "load_model", lambda architecture, path, device: (path.parent.parent.name == "class_weights" and "B2" or "B0", int(path.parent.name.split("_")[1])))
        monkeypatch.setattr(fe, "predict_dataset", fake_predict)
        monkeypatch.setattr(fe, "build_registro_previo", lambda *args, **kwargs: {"git": {"commit": "x"}})
        monkeypatch.setattr(fe, "environment_info", lambda device: {})
        monkeypatch.setattr(fe, "_device", lambda: torch.device("cpu"))
        return tmp_path

    def test_outputs_are_written_once_and_never_overwritten(self, protocol, synthetic_project):
        result = fe.run_test(synthetic_project, protocol, fe.TEST_CONFIRMATION, None, None)
        out_dir = synthetic_project / fe.R8_RESULTS_REL / "test"
        assert (out_dir / "resultados_test.json").is_file() and (out_dir / "metricas_por_semilla_test.csv").is_file()
        assert len(list((out_dir / "predicciones").glob("*.csv"))) == 6
        metrics = pd.read_csv(out_dir / "metricas_por_semilla_test.csv")
        assert {"average_precision", "roc_auc", "sensibilidad", "especificidad", "precision", "f1", "balanced_accuracy", "tn", "fp", "fn", "tp"} <= set(metrics.columns)
        assert len(metrics) == 6 and result["comparacion_b2_vs_b0"]["n_bootstrap"] == 2000 and result["comparacion_b2_vs_b0"]["semilla_bootstrap"] == 20260101
        assert set(result["resumen_entre_semillas"]) == {"B0", "B2"}
        with pytest.raises(FileExistsError):  # segunda ejecución: la carpeta ya existe
            fe.run_test(synthetic_project, protocol, fe.TEST_CONFIRMATION, None, None)
