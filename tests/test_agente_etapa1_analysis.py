"""
Pruebas del análisis de la etapa 1 del agente (src/classification/agente_etapa1_analysis.py). Solo biblioteca estándar.

Las 45 corridas del agente se SIMULAN en una carpeta temporal (datos de prueba, no resultados): cada corrida sintética
tiene los mismos archivos que una real y un run_config.json que registra sus flags como lo hace run_experiment.py.
Las referencias B0 y B2 son las corridas reales de robustez_e2 (se omiten las pruebas si no están en este entorno).
"""
from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path

import pytest

from src.classification import agente_etapa1_analysis as an
from src.classification import robustez_e2_analysis as e2a
from src.classification.cv_folds_e2 import DERIVED_COLUMNS, N_FOLDS, OUTPUT_REL, derive_fold_rows, read_csv_rows, write_csv_rows

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_PROTOCOL = PROJECT_ROOT / an.AGENT_REL / an.PROTOCOL_NAME
E2_DIR = PROJECT_ROOT / e2a.E2_REL
HAVE_REAL_FILES = AGENT_PROTOCOL.exists() and (E2_DIR / "corridas" / "convnext_a_alpha1" / "fold_4" / "validation_metrics.json").exists()
skip_without_files = pytest.mark.skipif(not HAVE_REAL_FILES, reason="faltan el protocolo del agente o las corridas de robustez_e2 (git pull)")

PREDICTION_COLUMNS = ["epoch", "image_id", "y_true", "probability_malignant"]


# ---------------------------------------------------------------------------------------------------------
# Corridas sintéticas
# ---------------------------------------------------------------------------------------------------------
def _recorded_flags(args, lr):
    """Lo que run_experiment.py registra en run_config.json para cada flag del brazo del agente."""
    cfg = {}
    if "--lr-schedule" in args:
        schedule = {"type": args["--lr-schedule"]}
        if args["--lr-schedule"] == "warmup_cosine":
            schedule.update(update="per_iteration (global step)", peak_lr=lr, warmup_epochs=float(args.get("--warmup-epochs", 0.0)),
                            cosine_epochs=float(args["--cosine-epochs"]), lr_min=float(args.get("--lr-min", 0.0)),
                            steps_per_epoch=25, after_cosine="lr_min constant")
        cfg["lr_schedule"] = schedule
    if "--layer-decay" in args:
        decay = float(args["--layer-decay"])
        groups = {f"features.{i}": decay ** (8 - i) for i in range(8)}
        groups["classifier"] = 1.0
        cfg["layer_decay"] = {"decay": decay, "groups": groups, "weight_decay": "same in all groups"}
    if "--drop-path-rate" in args:
        cfg["drop_path_rate"] = float(args["--drop-path-rate"])
    if "--label-smoothing" in args:
        cfg["label_smoothing"] = float(args["--label-smoothing"])
        cfg["label_smoothing_scope"] = "training loss only; validation loss (early stopping) is the weighted cross entropy without smoothing"
        cfg["label_smoothing_type"] = "uniform (1-e+e/C on the true class, e/C elsewhere); class weights as per-sample weight by true class (protocol amendment E1)"
    if "--augmentation" in args:
        cfg["augmentation"] = {"name": args["--augmentation"], **an.MODERATE_AUGMENTATION}
    if "--image-size" in args:
        cfg["image_size"] = int(args["--image-size"])
    if "--ema-decay" in args:
        cfg["ema_decay"] = float(args["--ema-decay"])
    return cfg


def _write_synthetic_run(root, run, folds_rows, quality, rng):
    folder = Path(root) / run["dir_rel"]
    folder.mkdir(parents=True)
    args, fold = run["args"], run["fold"]
    derived = derive_fold_rows(folds_rows, fold)
    write_csv_rows(folder / "fold_manifest.csv", derived, DERIVED_COLUMNS)
    validation = [r for r in derived if r["experimental_split"] == "validation"]
    train = [r for r in derived if r["experimental_split"] == "train"]

    labels = [int(r["binary_label"]) for r in validation]
    probabilities = [1.0 / (1.0 + math.exp(-(quality * (2 * y - 1) + rng.gauss(0, 1)))) for y in labels]
    ap, auc = e2a.average_precision(labels, probabilities), e2a.roc_auc(labels, probabilities)
    epochs_run, best = 6, 4

    prediction_rows = [{"epoch": best, "image_id": r["image_id"], "y_true": y, "probability_malignant": p}
                       for r, y, p in zip(validation, labels, probabilities)]
    write_csv_rows(folder / "val_predictions_best_auc.csv", prediction_rows, PREDICTION_COLUMNS)

    def curve(peak):  # primer máximo exactamente en la época `best`
        return [peak - 0.05 - 0.01 * i if i != best - 1 else peak for i in range(epochs_run)]

    lr = float(args.get("--learning-rate", 1e-4))
    history = {
        "train_loss": [1.0 - 0.1 * i for i in range(epochs_run)], "train_acc": [0.8] * epochs_run,
        "val_loss": [0.7 - 0.05 * i for i in range(epochs_run)], "val_unweighted_loss": [0.5] * epochs_run, "val_acc": [0.8] * epochs_run,
        "val_roc_auc": curve(auc), "val_average_precision": curve(ap), "val_f1_malignant": [0.3] * epochs_run,
        "val_recall_malignant": [0.3] * epochs_run, "val_specificity": [0.9] * epochs_run, "val_balanced_accuracy": [0.6] * epochs_run,
    }
    if "--lr-schedule" in args:
        history["lr"] = [lr * (0.9 ** i) for i in range(epochs_run)]

    config = {
        "architecture": "convnext_tiny", "condition": "A", "experiment": "r7_agente", "seed": 42, "seed_source": "cli_override",
        "csv_path": "configs/splits/cv_folds_e2.csv", "image_dir": "data/processed/MMOTU_OTU_2D_experimental",
        "train_samples": len(train), "validation_samples": len(validation),
        "class_weights": e2a.expected_class_weights([int(r["binary_label"]) for r in train], 1.0),
        "pretrained": True, "freeze_backbone": False, "num_classes": 2, "batch_size": 32, "epochs_max": 30, "epochs_run": epochs_run,
        "learning_rate": lr, "learning_rate_source": "cli_override" if "--learning-rate" in args else "yaml", "weight_decay": 1e-4,
        "optimizer": "AdamW", "use_amp": True, "patience": 5, "min_delta": 0.001, "threshold": 0.5, "scheduler": None,
        "checkpoint_criterion": "max val_roc_auc (first epoch among ties)", "epoch_best": best, "early_stopped": True,
        "early_stopping_criterion": "val_weighted_loss (patience=5, min_delta=0.001)",
        "evaluation_scheme": "E2", "fold": fold, "fold_manifest": "fold_manifest.csv",
        **_recorded_flags(args, lr),
    }
    (folder / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    (folder / "training_history.json").write_text(json.dumps(history), encoding="utf-8")
    (folder / "validation_metrics.json").write_text(json.dumps({"roc_auc": auc, "average_precision": ap}), encoding="utf-8")
    (folder / "best_auc_meta.json").write_text(json.dumps({"epoch": best, "criterion": "max val_roc_auc", "val_roc_auc": auc}), encoding="utf-8")


def build_tree(root, qualities, protocol_edit=None):
    """Árbol temporal con el protocolo, las referencias reales y 45 corridas sintéticas. qualities: candidata -> señal."""
    root = Path(root)
    (root / an.AGENT_REL).mkdir(parents=True)
    protocol = json.loads(AGENT_PROTOCOL.read_text(encoding="utf-8"))
    if protocol_edit:
        protocol_edit(protocol)
    (root / an.AGENT_REL / an.PROTOCOL_NAME).write_text(json.dumps(protocol), encoding="utf-8")
    (root / e2a.E2_REL).mkdir(parents=True)
    shutil.copy(E2_DIR / e2a.PROTOCOL_NAME, root / e2a.E2_REL / e2a.PROTOCOL_NAME)
    shutil.copytree(E2_DIR / "corridas", root / e2a.E2_REL / "corridas", ignore=shutil.ignore_patterns("*.pth"))
    (root / OUTPUT_REL).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(PROJECT_ROOT / OUTPUT_REL, root / OUTPUT_REL)

    folds_rows = read_csv_rows(root / OUTPUT_REL)
    rng = random.Random(0)
    for run in an.stage1_runs(protocol):
        _write_synthetic_run(root, run, folds_rows, qualities.get(run["config_id"], 0.0), rng)
    return protocol


STRONG = {"a03_cosine": 3.0, "a04_llrd08": 3.0}  # AUC muy superior a la de B0 (≈ 0.90): elegibles


# ---------------------------------------------------------------------------------------------------------
# Selección del finalista y consistencia (funciones puras)
# ---------------------------------------------------------------------------------------------------------
THRESHOLDS = {"delta_auc_minimo": -0.02, "empate_ap": 0.005}


def _candidate(cid, ap, auc, complete=True):
    return {"id": cid, "complete": complete, "ap_mean": ap if complete else None, "auc_mean": auc if complete else None}


class TestSelectFinalist:
    def test_highest_ap_among_eligible_and_auc_floor(self):
        result = an.select_finalist([_candidate("a01", 0.62, 0.91), _candidate("a02", 0.70, 0.80), _candidate("a03", 0.55, 0.90)], 0.90, THRESHOLDS)
        by_id = {r["id"]: r for r in result["candidatas"]}
        assert by_id["a02"]["elegible"] is False  # ΔAUC = -0.10 < -0.02, aunque tenga el mayor AP
        assert result["elegibles"] == ["a01", "a03"] and result["finalista"] == "a01"
        assert result["el_desempate_cambia_el_finalista"] is False

    @pytest.mark.parametrize("auc, eligible", [(0.881, True), (0.879, False)])
    def test_auc_floor_is_applied_to_the_mean_delta_against_b0(self, auc, eligible):
        result = an.select_finalist([_candidate("a01", 0.6, auc)], 0.90, THRESHOLDS)
        assert result["candidatas"][0]["elegible"] is eligible

    def test_tie_within_0005_goes_to_the_higher_auc(self):
        result = an.select_finalist([_candidate("a01", 0.603, 0.90), _candidate("a02", 0.600, 0.95), _candidate("a03", 0.590, 0.99)], 0.90, THRESHOLDS)
        assert result["en_empate"] == ["a01", "a02"]  # a03 difiere 0.013 del máximo
        assert result["finalista"] == "a02" and result["el_desempate_cambia_el_finalista"] is True

    def test_difference_of_0005_or_more_is_not_a_tie(self):
        result = an.select_finalist([_candidate("a01", 0.607, 0.90), _candidate("a02", 0.600, 0.99)], 0.90, THRESHOLDS)
        assert result["en_empate"] == ["a01"] and result["finalista"] == "a01"

    def test_remaining_tie_goes_to_the_lowest_id(self):
        result = an.select_finalist([_candidate("a05", 0.60, 0.92), _candidate("a02", 0.60, 0.92)], 0.90, THRESHOLDS)
        assert result["finalista"] == "a02"

    def test_discarded_candidates_are_not_eligible_and_no_eligible_means_no_finalist(self):
        result = an.select_finalist([_candidate("a01", None, None, complete=False), _candidate("a02", 0.4, 0.5)], 0.90, THRESHOLDS)
        assert result["candidatas"][0]["descartada"] is True and result["candidatas"][0]["elegible"] is False
        assert result["sin_elegibles"] is True and result["finalista"] is None and result["elegibles"] == []


class TestDeltaSummary:
    @staticmethod
    def _stats(first_ap, second_ap):
        make = lambda aps: {k: {"ap": ap, "auc": 0.9, "epoca": 3, "epocas": 6} for k, ap in enumerate(aps)}
        return {"x": make(first_ap), "b0": make(second_ap), "b2": make(second_ap)}

    @pytest.mark.parametrize(
        "first, verdict",
        [([0.6, 0.6, 0.6, 0.6, 0.4], "mejor_consistente"), ([0.4, 0.4, 0.4, 0.4, 0.6], "peor_consistente"),
         ([0.6, 0.6, 0.4, 0.4, 0.5], "no_consistente"), ([0.6, 0.6, 0.6, 0.5, 0.5], "no_consistente")],
    )
    def test_consistency_in_both_directions_uses_only_ap(self, first, verdict):
        stats = self._stats(first, [0.5] * 5)
        pairs = [("x__vs_b0", "x", "b0")]
        summaries = {r["metrica"]: r for r in an.summary_rows(an.delta_rows(stats, pairs), pairs)}
        assert summaries["AP"]["veredicto"] == verdict
        assert summaries["ROC-AUC"]["veredicto"] == "" and summaries["ROC-AUC"]["mejor_consistente"] == ""  # solo AP decide
        assert summaries["AP"]["folds_positivos"] + summaries["AP"]["folds_negativos"] + summaries["AP"]["folds_cero"] == 5
        assert summaries["AP"]["desv_est_muestral"] >= 0

    def test_incomplete_candidates_get_no_comparison(self):
        stats = {"b0": self._stats([0.5] * 5, [0.5] * 5)["b0"], "b2": self._stats([0.5] * 5, [0.5] * 5)["b2"], "x": {0: {"ap": 0.5, "auc": 0.9, "epoca": 1, "epocas": 2}}}
        assert [name for name, *_ in an.comparison_list(["x"], stats)] == ["b2__vs_b0"]


# ---------------------------------------------------------------------------------------------------------
# Plantilla de confirmación
# ---------------------------------------------------------------------------------------------------------
@skip_without_files
class TestConfirmationCommands:
    def test_thirty_commands_by_mechanical_substitution(self):
        protocol = json.loads(AGENT_PROTOCOL.read_text(encoding="utf-8"))
        candidates = an.candidate_table(protocol)
        commands = an.generate_confirmation_commands(protocol, candidates, "a03_cosine")
        checks = an.verify_confirmation_commands(commands, protocol, candidates, "a03_cosine")

        assert len(commands) == 30 and all(c["ok"] for c in checks)
        by_id = {}
        for c in commands:
            by_id.setdefault(c["config_id"], []).append(c)
        assert {k: len(v) for k, v in by_id.items()} == {"a03_cosine": 10, "b0": 10, "b2": 10}
        assert "--lr-schedule warmup_cosine --warmup-epochs 1 --cosine-epochs 15 --lr-min 1e-6" in by_id["a03_cosine"][0]["comando"]
        assert all("--class-weight-alpha 0.5" in c["comando"] for c in by_id["b2"])
        assert all("--checkpoint-metric val_roc_auc --split-csv" in c["comando"] and "  " not in c["comando"] for c in by_id["b0"])
        assert {c["seed"] for c in commands} == {43, 44} and {c["fold"] for c in commands} == set(range(5))

    def test_tampered_commands_fail_the_verification(self):
        protocol = json.loads(AGENT_PROTOCOL.read_text(encoding="utf-8"))
        candidates = an.candidate_table(protocol)
        commands = an.generate_confirmation_commands(protocol, candidates, "a01_lr5e5")
        for edit in (lambda c: c["comando"].replace("--fold 3", "--fold 4"), lambda c: c["comando"].replace("--seed 43", "--seed 42"),
                     lambda c: c["comando"].replace("val_roc_auc ", "val_roc_auc  "), lambda c: c["comando"] + " --overwrite"):
            tampered = [dict(c) for c in commands]
            tampered[3]["comando"] = edit(tampered[3])
            assert not all(r["ok"] for r in an.verify_confirmation_commands(tampered, protocol, candidates, "a01_lr5e5"))
        assert not all(r["ok"] for r in an.verify_confirmation_commands(commands[:-1], protocol, candidates, "a01_lr5e5"))

    def test_protocol_commands_and_thresholds_are_consistent_with_the_candidates(self):
        protocol = json.loads(AGENT_PROTOCOL.read_text(encoding="utf-8"))
        candidates, runs = an.candidate_table(protocol), an.stage1_runs(protocol)
        assert len(runs) == 45 and len(candidates) == 9
        assert all(f"--checkpoint-metric val_roc_auc {candidates[r['config_id']]['flags']} --split-csv" in r["command"] for r in runs)
        assert an.selection_thresholds(protocol) == THRESHOLDS
        assert [e["id"] for e in protocol["enmiendas"]] == ["E1"]


# ---------------------------------------------------------------------------------------------------------
# Análisis completo sobre corridas sintéticas
# ---------------------------------------------------------------------------------------------------------
@skip_without_files
class TestRunAnalysis:
    def test_with_a_finalist_writes_everything_and_the_thirty_commands(self, tmp_path):
        build_tree(tmp_path, STRONG)
        code, results = an.run_analysis(tmp_path, verbose=False)
        out = tmp_path / an.AGENT_REL / an.ANALYSIS_NAME

        assert code == 0 and all(c["ok"] for c in results["checks"])
        assert {p.name for p in out.iterdir()} == set(an.OUTPUT_FILES.values())

        registry = read_csv_rows(out / an.OUTPUT_FILES["registro"])
        protocol = json.loads(AGENT_PROTOCOL.read_text(encoding="utf-8"))
        assert list(registry[0].keys()) == protocol["registro"]["columnas"]
        assert sum(r["etapa"] == "etapa_1" for r in registry) == 45 and sum(r["etapa"] == "referencia" for r in registry) == 10
        assert {r["estado"] for r in registry} == {"completada"}

        selection = json.loads((out / an.OUTPUT_FILES["selection"]).read_text(encoding="utf-8"))
        assert selection["finalista"] in STRONG and set(selection["elegibles"]) == set(STRONG)
        assert selection["protocolo"]["enmiendas"] == ["E1"] and selection["conteo"]["candidatas_probadas"] == 9
        eligible_by_flag = {r["id"]: r["elegible"] for r in selection["candidatas"]}
        assert [cid for cid, ok in eligible_by_flag.items() if ok] == sorted(STRONG)

        commands = json.loads((out / an.OUTPUT_FILES["commands"]).read_text(encoding="utf-8"))
        assert commands["total"] == len(commands["comandos"]) == 30 and commands["finalista"] == selection["finalista"]
        assert all(commands["verificacion_contra_la_plantilla"].values())

        metrics = read_csv_rows(out / an.OUTPUT_FILES["metrics"])
        assert len(metrics) == 9 * 5 + 2 * 5  # candidatas y referencias, 5 folds cada una
        summaries = read_csv_rows(out / an.OUTPUT_FILES["delta_summary"])
        assert {r["comparacion"] for r in summaries} >= {"a01_lr5e5__vs_b0", "a01_lr5e5__vs_b2", "b2__vs_b0"}

        with pytest.raises(FileExistsError):
            an.run_analysis(tmp_path, verbose=False)

    def test_without_eligible_candidates_there_is_no_finalist_and_no_commands(self, tmp_path):
        build_tree(tmp_path, {})  # todas con señal nula: ROC-AUC ≈ 0.5, muy por debajo de B0
        code, _ = an.run_analysis(tmp_path, verbose=False)
        out = tmp_path / an.AGENT_REL / an.ANALYSIS_NAME
        selection = json.loads((out / an.OUTPUT_FILES["selection"]).read_text(encoding="utf-8"))

        assert code == 0 and selection["sin_elegibles"] is True and selection["finalista"] is None
        assert selection["consecuencia"].startswith("sin_elegibles")
        assert not (out / an.OUTPUT_FILES["commands"]).exists()

    def test_a_failed_run_is_registered_and_discards_its_candidate(self, tmp_path):
        build_tree(tmp_path, STRONG)
        (tmp_path / an.AGENT_REL / "corridas" / "a04_llrd08" / "fold_2" / "validation_metrics.json").unlink()
        code, _ = an.run_analysis(tmp_path, verbose=False)
        out = tmp_path / an.AGENT_REL / an.ANALYSIS_NAME
        registry = read_csv_rows(out / an.OUTPUT_FILES["registro"])
        selection = json.loads((out / an.OUTPUT_FILES["selection"]).read_text(encoding="utf-8"))

        failed = [r for r in registry if r["estado"] == "fallida"]
        assert code == 0 and len(registry) == 55 and len(failed) == 1
        assert failed[0]["config_id"] == "a04_llrd08" and failed[0]["fold"] == "2" and failed[0]["ap"] == "" and "validation_metrics.json" in failed[0]["nota"]
        assert all("descartada" in r["nota"] for r in registry if r["config_id"] == "a04_llrd08" and r["estado"] == "completada")
        by_id = {r["id"]: r for r in selection["candidatas"]}
        assert by_id["a04_llrd08"]["descartada"] is True and by_id["a04_llrd08"]["elegible"] is False
        assert selection["finalista"] == "a03_cosine"  # la única elegible que completó los 5 folds

    def test_runs_not_executed_stop_the_analysis_without_writing(self, tmp_path):
        build_tree(tmp_path, STRONG)
        shutil.rmtree(tmp_path / an.AGENT_REL / "corridas" / "a09_res288" / "fold_1")
        code, results = an.run_analysis(tmp_path, verbose=False)
        assert code == 1 and results is None and not (tmp_path / an.AGENT_REL / an.ANALYSIS_NAME).exists()

    @pytest.mark.parametrize(
        "tamper",
        [
            ("a05_dp03", 0, "run_config.json", lambda d: d.update(drop_path_rate=0.1)),  # flag no registrado como en el comando
            ("a06_ls01", 1, "run_config.json", lambda d: d.update(label_smoothing_type="PyTorch weight= semantics")),  # sin la enmienda E1
            ("a01_lr5e5", 2, "run_config.json", lambda d: d.update(learning_rate=1e-4, learning_rate_source="yaml")),
            ("a02_lr2e4", 3, "run_config.json", lambda d: d.update(checkpoint_criterion="min val_weighted_loss (absolute minimum)")),
            ("a03_cosine", 4, "run_config.json", lambda d: d.pop("lr_schedule")),
            ("a08_augmod", 0, "run_config.json", lambda d: d.update(fold=1)),
            ("a09_res288", 0, "run_config.json", lambda d: d.update(image_size=224)),
            ("a10_ema099", 0, "run_config.json", lambda d: d.update(experiment="r7_otro")),
            ("a05_dp03", 1, "validation_metrics.json", lambda d: d.update(average_precision=d["average_precision"] + 0.01)),
            ("a04_llrd08", 1, "training_history.json", lambda d: d["val_roc_auc"].__setitem__(0, 2.0)),  # otro máximo de AUC (supera cualquier AUC real)
        ],
        ids=["drop_path", "label_smoothing_E1", "learning_rate", "checkpoint", "calendario", "fold", "image_size", "experimento", "ap_no_reproducible", "epoca_del_checkpoint"],
    )
    def test_integrity_failures_stop_the_analysis_without_writing(self, tmp_path, tamper):
        config_id, fold, filename, edit = tamper
        build_tree(tmp_path, STRONG)
        path = tmp_path / an.AGENT_REL / "corridas" / config_id / f"fold_{fold}" / filename
        data = json.loads(path.read_text(encoding="utf-8"))
        edit(data)
        path.write_text(json.dumps(data), encoding="utf-8")

        code, results = an.run_analysis(tmp_path, verbose=False)
        assert code == 1 and results is None and not (tmp_path / an.AGENT_REL / an.ANALYSIS_NAME).exists()

    def test_protocol_without_the_amendment_or_with_other_reference_values_is_rejected(self, tmp_path):
        build_tree(tmp_path / "a", STRONG, protocol_edit=lambda p: p.update(enmiendas=[]))
        assert an.run_analysis(tmp_path / "a", verbose=False)[0] == 1

        build_tree(tmp_path / "b", STRONG, protocol_edit=lambda p: p["base_y_referencias"]["valores_de_referencia_semilla_42"]["B0"].update(ap_medio=0.9))
        assert an.run_analysis(tmp_path / "b", verbose=False)[0] == 1

    def test_the_test_split_is_never_read(self, tmp_path):
        """Ninguna salida ni corrida referencia imágenes de test: el manifiesto de cada corrida solo tiene train y validation."""
        build_tree(tmp_path, STRONG)
        for manifest in (tmp_path / an.AGENT_REL / "corridas").glob("*/fold_*/fold_manifest.csv"):
            rows = read_csv_rows(manifest)
            assert {r["e1_split"] for r in rows} <= {"train", "validation"} and len(rows) == 1000
