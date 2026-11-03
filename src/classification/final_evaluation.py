"""
Evaluador final (R8) de ConvNeXt-Tiny con representación A: B0 y B2 × semillas 42, 43 y 44.

Aplica el protocolo congelado en el cierre de R7
(results/classification/r7_optimizacion/cierre_seleccion/protocolo_r8_congelado.json). No entrena, no
recalibra, no ajusta umbrales ni selecciona modelos: solo carga los checkpoints best_auc.pth ya existentes.

Subcomandos:
    precheck             Comprobaciones de archivos, configuraciones, hashes y Git (no toca imágenes).
    verify-validation    Regenera las probabilidades de VALIDATION con cada checkpoint y las compara con
                         val_predictions_best_auc.csv. Nunca abre el conjunto test.
    test                 Evaluación final en test. Protegida: requiere confirmación explícita, un registro
                         previo con hashes y commit, una verificación de validation superada y una carpeta
                         de salida nueva. Se ejecuta una sola vez.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.classification.evaluate import evaluate_predictions, predict

PROTOCOL_REL = Path("results/classification/r7_optimizacion/cierre_seleccion/protocolo_r8_congelado.json")
PARTITION_REL = "configs/splits/mmotu_experimental_split_standard.csv"  # manifiesto usado en el entrenamiento
CANONICAL_PARTITION_REL = "configs/splits/mmotu_experimental_split.csv"
IMAGE_DIR_REL = "data/processed/MMOTU_OTU_2D_experimental"  # representación A (imagen completa)
R8_RESULTS_REL = Path("results/classification/r8_evaluacion_final")
CHECKPOINT_NAME = "best_auc.pth"
PREDICTIONS_NAME = "val_predictions_best_auc.csv"
THRESHOLD = 0.5
PROB_TOLERANCE = 1e-4  # diferencia máxima admitida entre las probabilidades regeneradas y las guardadas
TEST_CONFIRMATION = "EJECUTAR_TEST_UNA_SOLA_VEZ"
MODEL_FACTORIES = {"convnext_tiny": "create_convnext_tiny_classifier"}

# Hiperparámetros que deben ser idénticos entre B0 y B2 (solo difieren los pesos de clase y la etiqueta).
FROZEN_KEYS = [
    "architecture", "condition", "csv_path", "image_dir", "train_samples", "validation_samples", "pretrained", "freeze_backbone",
    "num_classes", "batch_size", "epochs_max", "learning_rate", "weight_decay", "optimizer", "use_amp", "patience", "min_delta",
    "threshold", "scheduler", "checkpoint_criterion",
]


# --------------------------------------------------------------------------------------------------
# Protocolo y rutas
# --------------------------------------------------------------------------------------------------
def load_protocol(project_root) -> dict:
    protocol = json.loads((Path(project_root) / PROTOCOL_REL).read_text(encoding="utf-8"))
    problems = validate_protocol(protocol)
    if problems:
        raise ValueError("Protocolo congelado inválido o modificado: " + "; ".join(problems))
    return protocol


def validate_protocol(protocol: dict) -> list[str]:
    problems = []
    models = protocol.get("modelos", {})
    comparison = protocol.get("comparacion_b2_vs_b0", {})
    if protocol.get("umbral") != THRESHOLD:
        problems.append("umbral distinto de 0.5")
    if models.get("semillas") != [42, 43, 44]:
        problems.append("semillas distintas de 42, 43 y 44")
    if models.get("total_modelos") != 6 or models.get("ensambles") is not False:
        problems.append("debe evaluar seis modelos sin ensambles")
    if models.get("arquitectura") not in MODEL_FACTORIES or models.get("representacion") != "A":
        problems.append("arquitectura o representación no admitidas")
    if set(models.get("configuraciones", {})) != {"B0", "B2"}:
        problems.append("las configuraciones deben ser B0 y B2")
    if comparison.get("n_bootstrap") != 2000 or comparison.get("semilla_bootstrap") != 20260101:
        problems.append("bootstrap distinto de 2000 remuestreos y semilla 20260101")
    if protocol.get("datos", {}).get("augmentation") is not False:
        problems.append("la evaluación no admite augmentation")
    return problems


def run_paths(project_root, protocol: dict, config: str, seed: int) -> dict:
    folder = Path(project_root) / protocol["modelos"]["configuraciones"][config]["carpeta"] / f"seed_{seed}"
    return {
        "dir": folder,
        "checkpoint": folder / CHECKPOINT_NAME,
        "run_config": folder / "run_config.json",
        "meta": folder / "best_auc_meta.json",
        "predictions": folder / PREDICTIONS_NAME,
        "metrics": folder / "validation_metrics.json",
    }


def iter_runs(protocol: dict, only=None):
    for config in protocol["modelos"]["configuraciones"]:
        if only and config not in only:
            continue
        for seed in protocol["modelos"]["semillas"]:
            yield config, seed


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_output_dir(path) -> Path:
    """Crea la carpeta de salida; no admite carpetas con contenido (no se sobrescribe nada)."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"La carpeta de salida ya existe y no está vacía: {path}. Use otra etiqueta; no se sobrescribe.")
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------------------------------
# Compatibilidad de configuraciones guardadas con el protocolo
# --------------------------------------------------------------------------------------------------
def check_run_config(run_config: dict, protocol: dict, config: str, seed: int) -> list[str]:
    models = protocol["modelos"]
    spec = models["configuraciones"][config]
    problems = []

    def expect(key, expected):
        if run_config.get(key) != expected:
            problems.append(f"{key}: {run_config.get(key)!r} != {expected!r}")

    expect("architecture", models["arquitectura"])
    expect("condition", models["representacion"])
    expect("seed", seed)
    expect("threshold", protocol["umbral"])
    expect("csv_path", PARTITION_REL)
    expect("image_dir", IMAGE_DIR_REL)
    expect("num_classes", 2)
    expect("freeze_backbone", False)
    expect("checkpoint_file", CHECKPOINT_NAME)
    if not str(run_config.get("checkpoint_criterion", "")).startswith("max val_roc_auc"):
        problems.append(f"checkpoint_criterion: {run_config.get('checkpoint_criterion')!r}")
    weights = run_config.get("class_weights")
    if weights is None or not np.allclose(weights, spec["class_weights"], atol=1e-6):
        problems.append(f"class_weights: {weights!r} != {spec['class_weights']!r}")
    if "class_weight_alpha" in spec:
        expect("class_weight_alpha", spec["class_weight_alpha"])
    elif run_config.get("class_weight_alpha", 1.0) != 1.0:
        problems.append("B0 no debe registrar un class_weight_alpha distinto de 1.0")
    return problems


def check_frozen_hyperparameters(run_config_b0: dict, run_config_b2: dict) -> list[str]:
    return [f"{key}: B0={run_config_b0.get(key)!r} B2={run_config_b2.get(key)!r}" for key in FROZEN_KEYS if run_config_b0.get(key) != run_config_b2.get(key)]


def check_meta(meta: dict, run_config: dict) -> list[str]:
    problems = []
    if meta.get("criterion") != "max val_roc_auc":
        problems.append(f"criterio del checkpoint: {meta.get('criterion')!r}")
    if meta.get("epoch") != run_config.get("epoch_best"):
        problems.append(f"época del checkpoint {meta.get('epoch')} != epoch_best {run_config.get('epoch_best')}")
    return problems


# --------------------------------------------------------------------------------------------------
# Git, hashes y registro previo
# --------------------------------------------------------------------------------------------------
def git_state(project_root) -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=project_root, capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {"commit": run("rev-parse", "HEAD"), "rama": run("rev-parse", "--abbrev-ref", "HEAD"), "cambios_en_archivos_versionados": run("status", "--porcelain", "--untracked-files=no") != ""}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"commit": None, "rama": None, "cambios_en_archivos_versionados": None, "error": str(error)}


def build_registro_previo(project_root, protocol: dict, with_hashes=True) -> dict:
    root = Path(project_root)
    checkpoints = {}
    for config, seed in iter_runs(protocol):
        path = run_paths(root, protocol, config, seed)["checkpoint"]
        checkpoints[f"{config}_seed{seed}"] = {
            "ruta": path.relative_to(root).as_posix(),
            "existe": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if with_hashes and path.is_file() else None,
        }
    partitions = {}
    for name, rel in (("operativa", PARTITION_REL), ("canonica", CANONICAL_PARTITION_REL)):
        path = root / rel
        partitions[name] = {"ruta": rel, "sha256": sha256_file(path) if with_hashes and path.is_file() else None}
    return {
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(root),
        "sha256_protocolo": sha256_file(root / PROTOCOL_REL),
        "sha256_particiones": partitions,
        "checkpoints": checkpoints,
    }


def compare_registros(frozen: dict, current: dict) -> list[str]:
    """Diferencias entre el registro previo congelado y el estado actual (vacía si coinciden)."""
    problems = []
    if frozen.get("git", {}).get("commit") != current["git"]["commit"]:
        problems.append("el commit de Git actual no coincide con el registrado")
    if current["git"].get("cambios_en_archivos_versionados") is not False:
        problems.append("hay cambios en archivos versionados (o no se pudo comprobar el estado de Git)")
    if frozen.get("sha256_protocolo") != current["sha256_protocolo"]:
        problems.append("el protocolo congelado cambió")
    for name, item in current["sha256_particiones"].items():
        if item["sha256"] is None or frozen.get("sha256_particiones", {}).get(name, {}).get("sha256") != item["sha256"]:
            problems.append(f"el SHA-256 de la partición {name} no coincide con el registrado")
    for key, item in current["checkpoints"].items():
        recorded = frozen.get("checkpoints", {}).get(key, {}).get("sha256")
        if not item["existe"]:
            problems.append(f"falta el checkpoint {key}")
        elif recorded is None or recorded != item["sha256"]:
            problems.append(f"el SHA-256 del checkpoint {key} no coincide con el registrado")
    return problems


def precheck(project_root, protocol: dict, with_hashes=True) -> dict:
    """Comprobaciones previas. 'pendiente' = no verificable en este entorno (p. ej., checkpoint ausente)."""
    root = Path(project_root)
    items = []

    def add(name, state, detail=""):
        items.append({"comprobacion": name, "estado": state, "detalle": detail})

    configs = {}
    for config, seed in iter_runs(protocol):
        paths = run_paths(root, protocol, config, seed)
        label = f"{config} semilla {seed}"
        missing = [name for name in ("run_config", "meta", "predictions", "metrics") if not paths[name].is_file()]
        if missing:
            add(f"{label}: archivos de resultados", "falla", "faltan " + ", ".join(missing))
            continue
        run_config = json.loads(paths["run_config"].read_text(encoding="utf-8"))
        configs[(config, seed)] = run_config
        problems = check_run_config(run_config, protocol, config, seed)
        add(f"{label}: run_config compatible con el protocolo", "falla" if problems else "cumple", "; ".join(problems))
        problems = check_meta(json.loads(paths["meta"].read_text(encoding="utf-8")), run_config)
        add(f"{label}: best_auc_meta coherente", "falla" if problems else "cumple", "; ".join(problems))
        if paths["checkpoint"].is_file():
            add(f"{label}: checkpoint presente", "cumple", f"{paths['checkpoint'].stat().st_size} bytes")
        else:
            add(f"{label}: checkpoint presente", "pendiente", "no está en este entorno: verificar en el entorno donde se entrenó")

    for seed in protocol["modelos"]["semillas"]:
        if ("B0", seed) in configs and ("B2", seed) in configs:
            problems = check_frozen_hyperparameters(configs[("B0", seed)], configs[("B2", seed)])
            add(f"semilla {seed}: B0 y B2 comparten hiperparámetros", "falla" if problems else "cumple", "; ".join(problems))

    for name, rel in (("operativa", PARTITION_REL), ("canónica", CANONICAL_PARTITION_REL)):
        add(f"partición {name} presente", "cumple" if (root / rel).is_file() else "falla", rel)
    git = git_state(root)
    add("estado de Git", "cumple" if git["commit"] else "falla", f"{git.get('rama')} @ {git['commit']}; cambios en archivos versionados: {git['cambios_en_archivos_versionados']}")
    states = {item["estado"] for item in items}
    return {"estado": "falla" if "falla" in states else ("pendiente" if "pendiente" in states else "completo"), "comprobaciones": items, "registro_previo": build_registro_previo(root, protocol, with_hashes)}


# --------------------------------------------------------------------------------------------------
# Métricas, comparación con las predicciones guardadas y bootstrap
# --------------------------------------------------------------------------------------------------
def compute_metrics(y_true, y_prob) -> dict:
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    results = evaluate_predictions(y_true, (y_prob >= THRESHOLD).astype(int), y_prob)  # misma función que el entrenamiento
    tn, fp, fn, tp = (int(v) for v in results["confusion_matrix"].ravel())
    return {
        "average_precision": float(results["average_precision"]),
        "roc_auc": float(results["roc_auc"]),
        "sensibilidad": float(results["recall_malignant"]),
        "especificidad": float(results["specificity"]),
        "precision": float(results["precision_malignant"]),
        "f1": float(results["f1_malignant"]),
        "balanced_accuracy": float(results["balanced_accuracy"]),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def compare_predictions(new: pd.DataFrame, saved: pd.DataFrame, tolerance=PROB_TOLERANCE) -> dict:
    """Compara probabilidades regeneradas y guardadas alineando por image_id; no usa el orden de las filas."""
    columns = ["image_id", "y_true", "probability_malignant"]
    for name, frame in (("regeneradas", new), ("guardadas", saved)):
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Predicciones {name}: faltan columnas {sorted(missing)}")
        if not frame["image_id"].is_unique:
            raise ValueError(f"Predicciones {name}: image_id repetidos")
    a, b = new.set_index("image_id").sort_index(), saved.set_index("image_id").sort_index()
    same_ids = a.index.equals(b.index)
    result = {"mismo_conjunto_de_image_id": bool(same_ids), "n_regeneradas": len(a), "n_guardadas": len(b), "tolerancia": tolerance}
    if not same_ids:
        result.update({"mismas_etiquetas": None, "aprobada": False, "motivo": "los conjuntos de image_id difieren"})
        return result
    same_labels = bool(np.array_equal(a["y_true"].to_numpy(), b["y_true"].to_numpy()))
    diff = np.abs(a["probability_malignant"].to_numpy() - b["probability_malignant"].to_numpy())
    flips = int(((a["probability_malignant"].to_numpy() >= THRESHOLD) != (b["probability_malignant"].to_numpy() >= THRESHOLD)).sum())
    metrics_new = compute_metrics(a["y_true"], a["probability_malignant"])
    metrics_saved = compute_metrics(b["y_true"], b["probability_malignant"])
    result.update(
        {
            "mismas_etiquetas": same_labels,
            "max_abs_diff": float(diff.max()),
            "image_id_max_diff": a.index[int(diff.argmax())].item(),
            "mean_abs_diff": float(diff.mean()),
            "n_sobre_tolerancia": int((diff > tolerance).sum()),
            "cambios_de_clase_al_umbral_0.5": flips,
            "max_abs_diff_metricas": {key: abs(metrics_new[key] - metrics_saved[key]) for key in ("average_precision", "roc_auc", "sensibilidad", "especificidad", "precision", "f1", "balanced_accuracy")},
            "metricas_regeneradas": metrics_new,
            "aprobada": bool(same_labels and diff.max() <= tolerance and flips == 0),
        }
    )
    return result


def paired_bootstrap_delta_ap(y, probs_b0, probs_b2, n_boot, seed, ci_level=95) -> dict:
    """
    Diferencia media de AP (B2 - B0) entre semillas con IC percentil por bootstrap pareado y estratificado.

    probs_b0 y probs_b2 tienen forma (semillas, imágenes). En cada réplica se remuestrea con reemplazo dentro de cada
    clase y los MISMOS índices se aplican a todas las semillas y a ambas configuraciones.
    """
    y, probs_b0, probs_b2 = np.asarray(y), np.asarray(probs_b0), np.asarray(probs_b2)
    if probs_b0.shape != probs_b2.shape or probs_b0.shape[1] != len(y):
        raise ValueError("probs_b0 y probs_b2 deben tener forma (semillas, imágenes) con las mismas imágenes que y")
    rng = np.random.default_rng(seed)
    positives, negatives = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(positives, positives.size), rng.choice(negatives, negatives.size)])
        yb = y[idx]
        deltas[b] = np.mean([average_precision_score(yb, p2[idx]) - average_precision_score(yb, p0[idx]) for p0, p2 in zip(probs_b0, probs_b2)])
    low, high = np.percentile(deltas, [(100 - ci_level) / 2, 100 - (100 - ci_level) / 2])
    observed_by_seed = [average_precision_score(y, p2) - average_precision_score(y, p0) for p0, p2 in zip(probs_b0, probs_b2)]
    return {
        "delta_ap_medio_observado": float(np.mean(observed_by_seed)),
        "delta_ap_por_semilla": [float(v) for v in observed_by_seed],
        "ic_inf": float(low), "ic_sup": float(high), "nivel_ic": ci_level,
        "ee_bootstrap": float(deltas.std(ddof=1)),
        "fraccion_replicas_delta_positivo": float((deltas > 0).mean()),
        "n_bootstrap": n_boot, "semilla_bootstrap": seed,
    }


# --------------------------------------------------------------------------------------------------
# Inferencia
# --------------------------------------------------------------------------------------------------
def build_dataset(project_root, split: str, allow_test=False):
    """Dataset con las transformaciones de validation (sin augmentation). El split 'test' exige allow_test=True."""
    if split not in ("validation", "test"):
        raise ValueError(f"split no admitido: {split!r}")
    if split == "test" and not allow_test:
        raise PermissionError("El conjunto test solo puede abrirse desde el modo 'test' protegido.")
    from src.classification.dataset import MMOTUClassificationDataset
    from src.classification.transforms import get_val_transforms

    return MMOTUClassificationDataset(
        csv_file=str(Path(project_root) / PARTITION_REL),
        image_dir=str(Path(project_root) / IMAGE_DIR_REL),
        split=split,
        transform=get_val_transforms(),
        condition="A",
    )


def load_model(architecture: str, checkpoint_path, device):
    import torch
    from src.classification import models

    model = getattr(models, MODEL_FACTORIES[architecture])(pretrained=False, num_classes=2, freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    return model


def predict_dataset(model, dataset, batch_size, device, num_workers=0) -> pd.DataFrame:
    """Probabilidades de la clase positiva por image_id, con el mismo predict() del entrenamiento (fp32, eval, sin augmentation)."""
    from src.classification.dataloader import create_dataloader

    loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    y_true, _, y_prob = predict(model, loader, device)
    frame = dataset.df
    if not (len(frame) == len(y_true) == len(y_prob)) or not np.array_equal(frame["binary_label"].to_numpy(), y_true):
        raise ValueError("Las predicciones no corresponden al orden de image_id del dataset.")
    return pd.DataFrame({"image_id": frame["image_id"].to_numpy(), "y_true": y_true.astype(int), "probability_malignant": y_prob.astype(float)})


def environment_info(device) -> dict:
    import torch

    info = {"python": platform.python_version(), "torch": torch.__version__, "device": str(device), "sistema": platform.platform()}
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except ImportError:
        info["torchvision"] = None
    return info


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------------------------------
# Verificación sobre validation (nunca abre test)
# --------------------------------------------------------------------------------------------------
def verify_validation(project_root, protocol, label, tolerance=PROB_TOLERANCE, num_workers=0, only=None) -> dict:
    root = Path(project_root)
    if any(run_paths(root, protocol, c, s)["checkpoint"].is_file() for c, s in iter_runs(protocol, only)):
        import torchvision  # noqa: F401  (falla antes de crear la carpeta de salida si el entorno no puede ejecutar el modelo)
    out_dir = prepare_output_dir(root / R8_RESULTS_REL / f"verificacion_validation_{label}")
    (out_dir / "predicciones_regeneradas").mkdir()
    device = _device()
    pre = precheck(root, protocol)
    dataset = None
    runs, rows = {}, []

    for config, seed in iter_runs(protocol, only):
        paths = run_paths(root, protocol, config, seed)
        key = f"{config}_seed{seed}"
        if not paths["checkpoint"].is_file():
            runs[key] = {"estado": "no_ejecutado", "motivo": "checkpoint ausente en este entorno"}
            continue
        if dataset is None:
            dataset = build_dataset(root, "validation")
        run_config = json.loads(paths["run_config"].read_text(encoding="utf-8"))
        model = load_model(protocol["modelos"]["arquitectura"], paths["checkpoint"], device)
        regenerated = predict_dataset(model, dataset, run_config["batch_size"], device, num_workers)
        regenerated.to_csv(out_dir / "predicciones_regeneradas" / f"{key}.csv", index=False)
        comparison = compare_predictions(regenerated, pd.read_csv(paths["predictions"]), tolerance)
        saved_metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
        comparison["max_abs_diff_vs_validation_metrics_json"] = {
            "average_precision": abs(comparison["metricas_regeneradas"]["average_precision"] - saved_metrics["average_precision"]),
            "roc_auc": abs(comparison["metricas_regeneradas"]["roc_auc"] - saved_metrics["roc_auc"]),
        } if comparison.get("metricas_regeneradas") else None
        comparison["sha256_checkpoint"] = sha256_file(paths["checkpoint"])
        comparison["estado"] = "aprobada" if comparison["aprobada"] else "no_aprobada"
        runs[key] = comparison
        rows.append({"corrida": key, "estado": comparison["estado"], **{k: comparison.get(k) for k in ("max_abs_diff", "mean_abs_diff", "n_sobre_tolerancia", "cambios_de_clase_al_umbral_0.5", "mismas_etiquetas", "image_id_max_diff")}, **{f"dif_{k}": v for k, v in (comparison.get("max_abs_diff_metricas") or {}).items()}})

    executed = [r for r in runs.values() if r["estado"] != "no_ejecutado"]
    all_expected = len(list(iter_runs(protocol))) == 6 and only is None
    if any(r["estado"] == "no_aprobada" for r in executed) or pre["estado"] == "falla":
        status = "no_superada"
    elif len(executed) == 6 and all_expected:
        status = "superada"
    else:
        status = "parcial"  # no se marca como superada hasta ejecutar las seis corridas
    report = {
        "estado": status,
        "tolerancia_prob": tolerance,
        "tolerancia_estandar": tolerance == PROB_TOLERANCE,
        "corridas_ejecutadas": len(executed),
        "corridas_esperadas": 6,
        "corridas": runs,
        "precheck": {"estado": pre["estado"], "comprobaciones": pre["comprobaciones"]},
        "registro_previo": pre["registro_previo"],
        "entorno": environment_info(device),
        "split_usado": "validation",
    }
    (out_dir / "verificacion_validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_dir / "diferencias_validation.csv", index=False)
    return report


# --------------------------------------------------------------------------------------------------
# Modo test (protegido)
# --------------------------------------------------------------------------------------------------
def assert_test_gates(project_root, protocol, confirmation, registro_path, verificacion_path) -> dict:
    """Todas las condiciones previas a abrir test. Lanza PermissionError con la lista completa de fallos."""
    root = Path(project_root)
    failures = []
    if confirmation != TEST_CONFIRMATION:
        failures.append(f"falta la confirmación explícita (--confirmar-test {TEST_CONFIRMATION})")
    test_dir = root / R8_RESULTS_REL / "test"
    if test_dir.exists():
        failures.append(f"ya existe {test_dir.relative_to(root).as_posix()}: test se evalúa una sola vez y no se sobrescribe")
    registro = None
    if not registro_path or not Path(registro_path).is_file():
        failures.append("falta el registro previo (--registro-previo) con commit y SHA-256")
    else:
        registro = json.loads(Path(registro_path).read_text(encoding="utf-8"))
        registro = registro.get("registro_previo", registro)
        failures += compare_registros(registro, build_registro_previo(root, protocol))
    if not verificacion_path or not Path(verificacion_path).is_file():
        failures.append("falta la verificación de validation (--verificacion-validation)")
    else:
        verification = json.loads(Path(verificacion_path).read_text(encoding="utf-8"))
        if verification.get("estado") != "superada":
            failures.append(f"la verificación de validation no está superada (estado: {verification.get('estado')!r})")
        if verification.get("tolerancia_estandar") is not True:
            failures.append("la verificación de validation usó una tolerancia no estándar")
        if registro is not None:
            for key, item in verification.get("registro_previo", {}).get("checkpoints", {}).items():
                if item.get("sha256") != registro.get("checkpoints", {}).get(key, {}).get("sha256"):
                    failures.append(f"el checkpoint {key} de la verificación no coincide con el registro previo")
    if failures:
        raise PermissionError("Test NO habilitado: " + "; ".join(failures))
    return registro


def run_test(project_root, protocol, confirmation, registro_path, verificacion_path, num_workers=0) -> dict:
    root = Path(project_root)
    assert_test_gates(root, protocol, confirmation, registro_path, verificacion_path)
    device = _device()
    dataset = build_dataset(root, "test", allow_test=True)
    predictions = {}
    for config, seed in iter_runs(protocol):
        paths = run_paths(root, protocol, config, seed)
        run_config = json.loads(paths["run_config"].read_text(encoding="utf-8"))
        model = load_model(protocol["modelos"]["arquitectura"], paths["checkpoint"], device)
        predictions[(config, seed)] = predict_dataset(model, dataset, run_config["batch_size"], device, num_workers)

    reference = predictions[("B0", protocol["modelos"]["semillas"][0])].set_index("image_id").sort_index()
    for (config, seed), frame in predictions.items():  # mismas imágenes y etiquetas en las seis corridas
        aligned = frame.set_index("image_id").sort_index()
        if not (aligned.index.equals(reference.index) and np.array_equal(aligned["y_true"].to_numpy(), reference["y_true"].to_numpy())):
            raise ValueError(f"{config} semilla {seed}: no evalúa las mismas imágenes y etiquetas que las demás corridas")
    y = reference["y_true"].to_numpy()
    probs = {cfg: np.stack([predictions[(cfg, s)].set_index("image_id").sort_index()["probability_malignant"].to_numpy() for s in protocol["modelos"]["semillas"]]) for cfg in ("B0", "B2")}

    rows = [{"config": cfg, "seed": seed, **compute_metrics(predictions[(cfg, seed)]["y_true"], predictions[(cfg, seed)]["probability_malignant"])} for cfg, seed in predictions]
    metrics = pd.DataFrame(rows)
    metric_columns = [c for c in metrics.columns if c not in ("config", "seed", "tn", "fp", "fn", "tp")]
    summary = {cfg: {m: {"media": float(g[m].mean()), "sd_muestral": float(g[m].std(ddof=1))} for m in metric_columns} for cfg, g in metrics.groupby("config")}
    comparison = protocol["comparacion_b2_vs_b0"]
    bootstrap = paired_bootstrap_delta_ap(y, probs["B0"], probs["B2"], comparison["n_bootstrap"], comparison["semilla_bootstrap"])

    out_dir = root / R8_RESULTS_REL / "test"
    out_dir.mkdir(parents=True, exist_ok=False)  # una sola ejecución: falla si ya existe
    (out_dir / "predicciones").mkdir()
    for (cfg, seed), frame in predictions.items():
        frame.to_csv(out_dir / "predicciones" / f"{cfg}_seed{seed}.csv", index=False)
    metrics.to_csv(out_dir / "metricas_por_semilla_test.csv", index=False)
    result = {
        "resumen_entre_semillas": summary,
        "comparacion_b2_vs_b0": bootstrap,
        "umbral": THRESHOLD,
        "n_imagenes_test": int(len(y)),
        "registro_previo": build_registro_previo(root, protocol),
        "entorno": environment_info(device),
        "nota": "Incertidumbre por remuestreo de imágenes (IC bootstrap) y variabilidad entre semillas (DE) son fuentes distintas y no se combinan.",
    }
    (out_dir / "resultados_test.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluador final R8 (protocolo congelado en el cierre de R7).")
    parser.add_argument("--project-root", default=".", help="Raíz del repositorio (por defecto, el directorio actual).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser("precheck", help="Comprobaciones de archivos, configuraciones, hashes y Git (no abre imágenes).")
    p_pre.add_argument("--write-registro-previo", metavar="ARCHIVO", help="Guarda el registro previo (commit y SHA-256) en una carpeta nueva de r8_evaluacion_final.")
    p_pre.add_argument("--sin-hash", action="store_true", help="No calcula SHA-256 (solo vista rápida).")

    p_val = sub.add_parser("verify-validation", help="Reproduce validation con cada checkpoint y compara con val_predictions_best_auc.csv.")
    p_val.add_argument("--label", required=True, help="Etiqueta de la carpeta de salida (verificacion_validation_<label>), p. ej. 'local' o 'remoto'.")
    p_val.add_argument("--tolerancia", type=float, default=PROB_TOLERANCE, help=f"Diferencia máxima admitida (por defecto {PROB_TOLERANCE}); un valor distinto se registra como no estándar.")
    p_val.add_argument("--configs", nargs="+", choices=["B0", "B2"], help="Solo para preparación local: limita las configuraciones; la verificación queda 'parcial'.")
    p_val.add_argument("--num-workers", type=int, default=0)

    p_test = sub.add_parser("test", help="Evaluación final en test (protegida; se ejecuta una sola vez).")
    p_test.add_argument("--confirmar-test", default=None, help=f"Debe ser exactamente {TEST_CONFIRMATION}.")
    p_test.add_argument("--registro-previo", required=True, help="Registro previo congelado (commit y SHA-256).")
    p_test.add_argument("--verificacion-validation", required=True, help="verificacion_validation.json con estado 'superada'.")
    p_test.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args(argv)
    root = Path(args.project_root).resolve()
    protocol = load_protocol(root)

    if args.command == "precheck":
        report = precheck(root, protocol, with_hashes=not args.sin_hash)
        for item in report["comprobaciones"]:
            print(f"[{item['estado']:>9}] {item['comprobacion']}" + (f" -> {item['detalle']}" if item["detalle"] else ""))
        print(f"Estado general: {report['estado']}")
        if args.write_registro_previo:
            target = root / R8_RESULTS_REL / args.write_registro_previo
            if target.exists():
                raise FileExistsError(f"{target} ya existe; no se sobrescribe.")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report["registro_previo"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"Registro previo guardado en {target.relative_to(root).as_posix()}")
        return 0 if report["estado"] != "falla" else 1

    if args.command == "verify-validation":
        report = verify_validation(root, protocol, args.label, args.tolerancia, args.num_workers, args.configs)
        print(f"Verificación de validation: {report['estado']} ({report['corridas_ejecutadas']} de {report['corridas_esperadas']} corridas ejecutadas)")
        for key, run in report["corridas"].items():
            if "motivo" in run:
                detail = run["motivo"]
            elif run.get("max_abs_diff") is None:
                detail = "los conjuntos de image_id no coinciden"
            else:
                detail = f"max_abs_diff={run['max_abs_diff']:.3e}, cambios de clase={run['cambios_de_clase_al_umbral_0.5']}, sobre tolerancia={run['n_sobre_tolerancia']}"
            print(f"  {key}: {run['estado']} ({detail})")
        return 0 if report["estado"] in ("superada", "parcial") else 1

    if args.command == "test":
        try:
            result = run_test(root, protocol, args.confirmar_test, args.registro_previo, args.verificacion_validation, args.num_workers)
        except PermissionError as error:
            print(error, file=sys.stderr)
            return 2
        print("Test evaluado; resultados en", (R8_RESULTS_REL / "test").as_posix(), f"({result['n_imagenes_test']} imágenes)")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
