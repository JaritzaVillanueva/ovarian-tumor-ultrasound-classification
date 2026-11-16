"""
Análisis de la etapa 1 del brazo del agente de optimización (R7), aplicando literalmente
results/classification/r7_optimizacion/agente/protocolo_agente.json (con la enmienda E1). No añade reglas: las
candidatas, los comandos, la regla de consistencia y la selección del finalista se leen y aplican como están escritos.

Lee (solo archivos de validation; no entrena, no carga checkpoints y no usa test):
    - las 45 corridas de agente/corridas/<candidata>/fold_<k>/ (semilla 42);
    - las referencias B0 y B2 de robustez_e2/corridas/ (semilla 42, ya existentes).
Escribe en agente/analisis_etapa1/ sin sobrescribir:
    registro_intentos.csv (columnas del protocolo), verificacion_corridas.csv, metricas_por_candidata_y_fold.csv,
    delta_por_fold.csv, resumen_delta.csv, seleccion_finalista.json y, solo si hay finalista,
    comandos_confirmacion.json (30 comandos por sustitución mecánica de la plantilla, verificados contra ella).

Solo usa la biblioteca estándar.

Uso (desde la raíz del repositorio):
    python -m src.classification.agente_etapa1_analysis
"""
from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
from pathlib import Path

from src.classification import checkpoint_sensitivity as cs
from src.classification import robustez_e2_analysis as e2a
from src.classification.cv_folds_e2 import N_FOLDS, OUTPUT_REL as CV_FOLDS_REL, read_csv_rows

AGENT_REL = Path("results") / "classification" / "r7_optimizacion" / "agente"
PROTOCOL_NAME = "protocolo_agente.json"
ANALYSIS_NAME = "analisis_etapa1"
PROTOCOL_COMMIT = "7d48f86"  # commit del protocolo con la enmienda E1

REQUIRED_FILES = (
    "validation_metrics.json", "run_config.json", "training_history.json", "best_auc_meta.json",
    "fold_manifest.csv", "val_predictions_best_auc.csv",
)
# Parámetros del aumento "moderate" (protocolo a08)
MODERATE_AUGMENTATION = {"degrees": 30, "brightness": 0.2, "contrast": 0.2}

OUTPUT_FILES = {
    "registro": "registro_intentos.csv",
    "verification": "verificacion_corridas.csv",
    "metrics": "metricas_por_candidata_y_fold.csv",
    "delta_by_fold": "delta_por_fold.csv",
    "delta_summary": "resumen_delta.csv",
    "selection": "seleccion_finalista.json",
    "commands": "comandos_confirmacion.json",
}

STATUS_COMPLETED, STATUS_FAILED, STATUS_NOT_RUN = "completada", "fallida", "no_ejecutada"


# ---------------------------------------------------------------------------------------------------------
# Protocolo
# ---------------------------------------------------------------------------------------------------------
def load_agent_protocol(root):
    path = Path(root) / AGENT_REL / PROTOCOL_NAME
    return e2a.read_json(path), path


def file_matches_commit(root, relative_path, commit, path):
    """True/False si el archivo es idéntico al del commit; None si git no está disponible."""
    try:
        result = subprocess.run(["git", "show", f"{commit}:{Path(relative_path).as_posix()}"], cwd=root,
                                capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    normalize = lambda b: b.replace(b"\r\n", b"\n")
    return normalize(result.stdout) == normalize(Path(path).read_bytes())


def candidate_table(protocol):
    """id -> {'flags': texto, 'args': {'--flag': valor}}."""
    return {c["id"]: {"flags": c["flags"], "args": e2a.parse_command(c["flags"])} for c in protocol["candidatas"]}


def stage1_runs(protocol):
    """Las corridas de la etapa 1 tal como están escritas en el protocolo."""
    runs = []
    for item in protocol["comandos"]["etapa_1"]:
        config_id, fold = item["id"].rsplit("_fold", 1)
        args = e2a.parse_command(item["comando"])
        runs.append({"id": item["id"], "config_id": config_id, "fold": int(fold), "seed": int(args["--seed"]),
                     "command": item["comando"], "args": args, "dir_rel": args["--output-dir"]})
    return runs


def selection_thresholds(protocol):
    """Umbrales de seleccion_del_finalista leídos del texto del protocolo (no se fijan aquí)."""
    rules = protocol["metodo_de_busqueda"]["seleccion_del_finalista"]
    auc = re.search(r">=\s*(-?\d+(?:\.\d+)?)", rules["elegibles"])
    tie = re.search(r"diferencia\s*<\s*(\d+(?:\.\d+)?)", rules["finalista"])
    if not auc or not tie:
        raise ValueError("No se pudieron leer los umbrales de seleccion_del_finalista del protocolo.")
    return {"delta_auc_minimo": float(auc.group(1)), "empate_ap": float(tie.group(1))}


# ---------------------------------------------------------------------------------------------------------
# Carga y verificación de las corridas del agente
# ---------------------------------------------------------------------------------------------------------
def load_agent_run(root, run):
    """Estado de una corrida: no ejecutada (sin carpeta), fallida (carpeta sin todos los archivos) o completada."""
    folder = Path(root) / run["dir_rel"]
    data = dict(run, dir=folder, status=STATUS_COMPLETED, missing=[])
    if not folder.exists():
        data.update(status=STATUS_NOT_RUN)
        return data
    missing = [name for name in REQUIRED_FILES if not (folder / name).exists()]
    if missing:
        data.update(status=STATUS_FAILED, missing=missing)
        return data
    data.update(
        metrics=e2a.read_json(folder / "validation_metrics.json"),
        config=e2a.read_json(folder / "run_config.json"),
        history=e2a.read_json(folder / "training_history.json"),
        meta=e2a.read_json(folder / "best_auc_meta.json"),
        manifest=read_csv_rows(folder / "fold_manifest.csv"),
        predictions=read_csv_rows(folder / "val_predictions_best_auc.csv"),
    )
    return data


def _flag_checks(cfg, history, args):
    """Cada flag del comando debe quedar registrado en run_config.json, y solo los que se pasaron."""
    checks = []
    lr = float(args["--learning-rate"]) if "--learning-rate" in args else e2a.EXPECTED_TRAINING["learning_rate"]
    checks.append(("flag: learning rate efectivo", cfg["learning_rate"] == lr and
                   cfg["learning_rate_source"] == ("cli_override" if "--learning-rate" in args else "yaml"),
                   f"lr={cfg['learning_rate']} ({cfg['learning_rate_source']}), esperado {lr}"))

    if "--lr-schedule" in args:
        schedule = cfg.get("lr_schedule", {})
        ok = schedule.get("type") == args["--lr-schedule"] and "lr" in history and len(history["lr"]) == cfg["epochs_run"]
        detail = str(schedule)
        if args["--lr-schedule"] == "warmup_cosine" and ok:
            warmup = float(args.get("--warmup-epochs", 0.0))
            cosine, floor = float(args["--cosine-epochs"]), float(args.get("--lr-min", 0.0))
            ok = (schedule.get("warmup_epochs") == warmup and schedule.get("cosine_epochs") == cosine
                  and schedule.get("lr_min") == floor and schedule.get("peak_lr") == lr)
            values = history["lr"]
            in_range = all(floor * (1 - 1e-9) <= v <= lr * (1 + 1e-9) for v in values)
            after_warmup = values[int(-(-warmup // 1)):]  # épocas posteriores al calentamiento
            ok = ok and in_range and all(a >= b - 1e-15 for a, b in zip(after_warmup, after_warmup[1:]))
        checks.append(("flag: calendario del learning rate", ok, detail))
    else:
        checks.append(("flag: sin calendario", "lr_schedule" not in cfg and "lr" not in history, ""))

    if "--layer-decay" in args:
        decay = float(args["--layer-decay"])
        groups = cfg.get("layer_decay", {}).get("groups", {})
        ok = (cfg.get("layer_decay", {}).get("decay") == decay and len(groups) == 9
              and abs(groups.get("features.7", -1) - decay) < 1e-12 and abs(groups.get("features.0", -1) - decay ** 8) < 1e-12
              and groups.get("classifier") == 1.0)
        checks.append(("flag: layer decay", ok, str(cfg.get("layer_decay"))))
    else:
        checks.append(("flag: sin layer decay", "layer_decay" not in cfg, ""))

    if "--drop-path-rate" in args:
        checks.append(("flag: drop path", cfg.get("drop_path_rate") == float(args["--drop-path-rate"]), str(cfg.get("drop_path_rate"))))
    else:
        checks.append(("flag: sin drop path", "drop_path_rate" not in cfg, ""))

    if "--label-smoothing" in args:
        smoothing_type = cfg.get("label_smoothing_type", "")
        checks.append(("flag: label smoothing uniforme (enmienda E1), validación sin suavizado",
                       cfg.get("label_smoothing") == float(args["--label-smoothing"]) and "uniform" in smoothing_type
                       and "E1" in smoothing_type and "without smoothing" in cfg.get("label_smoothing_scope", ""), smoothing_type))
    else:
        checks.append(("flag: sin label smoothing", "label_smoothing" not in cfg, ""))

    if "--augmentation" in args:
        augmentation = cfg.get("augmentation", {})
        expected = MODERATE_AUGMENTATION if args["--augmentation"] == "moderate" else {}
        checks.append(("flag: aumento", augmentation.get("name") == args["--augmentation"]
                       and all(augmentation.get(k) == v for k, v in expected.items()), str(augmentation)))
    else:
        checks.append(("flag: sin aumento moderado", "augmentation" not in cfg, ""))

    if "--image-size" in args:
        checks.append(("flag: tamaño de imagen", cfg.get("image_size") == int(args["--image-size"]), str(cfg.get("image_size"))))
    else:
        checks.append(("flag: sin cambio de tamaño", "image_size" not in cfg, ""))

    if "--ema-decay" in args:
        checks.append(("flag: EMA", cfg.get("ema_decay") == float(args["--ema-decay"]), str(cfg.get("ema_decay"))))
    else:
        checks.append(("flag: sin EMA", "ema_decay" not in cfg, ""))
    return checks


def verify_agent_run(run, candidate, folds_rows):
    """Comprobaciones de una corrida completada del agente. Filas (corrida, comprobacion, ok, detalle)."""
    cfg, args, fold = run["config"], run["args"], run["fold"]
    rows = []

    def add(name, ok, detail=""):
        rows.append({"corrida": run["id"], "comprobacion": name, "ok": bool(ok), "detalle": detail})

    add("comando = flags de la candidata", f"--checkpoint-metric val_roc_auc {candidate['flags']} --split-csv" in run["command"],
        candidate["flags"])
    add("config: esquema E2 y fold", cfg.get("evaluation_scheme") == "E2" and cfg.get("fold") == fold,
        f"evaluation_scheme={cfg.get('evaluation_scheme')}, fold={cfg.get('fold')}")
    add("config: arquitectura, condición, experimento, semilla",
        cfg["architecture"] == args["--arch"] and cfg["condition"] == args["--condition"]
        and cfg["experiment"] == args["--experiment"] and cfg["seed"] == int(args["--seed"]),
        f"{cfg['architecture']}/{cfg['condition']}/{cfg['experiment']}/seed {cfg['seed']}")
    add("config: partición", cfg["csv_path"] == args["--split-csv"] == CV_FOLDS_REL, cfg["csv_path"])
    add("config: pesos de clase con α = 1.0 (no se busca)", "class_weight_alpha" not in cfg, str(cfg.get("class_weight_alpha")))

    lr = float(args["--learning-rate"]) if "--learning-rate" in args else e2a.EXPECTED_TRAINING["learning_rate"]
    expected_training = dict(e2a.EXPECTED_TRAINING, learning_rate=lr)
    differing = {k: cfg[k] for k in expected_training if cfg[k] != expected_training[k]}
    add("config: hiperparámetros fijos del protocolo", not differing, str(differing) if differing else "")
    add("config: criterio de checkpoint max val_roc_auc (ninguna candidata lo cambia)",
        cfg["checkpoint_criterion"].startswith("max val_roc_auc"), cfg["checkpoint_criterion"])
    add("config: early stopping activo (patience 5, min_delta 0.001)",
        cfg["patience"] == 5 and cfg["min_delta"] == 0.001 and "disabled" not in cfg.get("early_stopping_criterion", ""), "")
    add("config: 800 train y 200 validation", cfg["train_samples"] == 800 and cfg["validation_samples"] == 200,
        f"{cfg['train_samples']}/{cfg['validation_samples']}")
    for name, ok, detail in _flag_checks(cfg, run["history"], args):
        add(name, ok, detail)

    manifest = run["manifest"]
    fold_ids = {r["image_id"] for r in folds_rows if int(r["fold"]) == fold}
    validation = [r for r in manifest if r["experimental_split"] == "validation"]
    train = [r for r in manifest if r["experimental_split"] == "train"]
    add("manifiesto: 1000 imágenes, ninguna de test",
        len(manifest) == 1000 and {r["e1_split"] for r in manifest} <= {"train", "validation"}
        and len({r["image_id"] for r in manifest}) == 1000, f"{len(manifest)} filas")
    add("manifiesto: validation = fold de cv_folds_e2.csv",
        {r["image_id"] for r in validation} == fold_ids and len(validation) == 200, f"{len(validation)} imágenes")
    expected_weights = e2a.expected_class_weights([int(r["binary_label"]) for r in train], 1.0)
    add("pesos de clase = recalculados con las etiquetas de train del fold (α = 1.0)",
        all(abs(a - b) <= 1e-5 * max(1.0, abs(b)) for a, b in zip(cfg["class_weights"], expected_weights)), str(cfg["class_weights"]))

    predictions = run["predictions"]
    y_true = [int(r["y_true"]) for r in predictions]
    y_prob = [float(r["probability_malignant"]) for r in predictions]
    add("predicciones: mismas imágenes y orden que validation del manifiesto, época del checkpoint",
        [r["image_id"] for r in predictions] == [r["image_id"] for r in validation]
        and y_true == [int(r["binary_label"]) for r in validation]
        and {int(r["epoch"]) for r in predictions} == {cfg["epoch_best"]}, f"{len(predictions)} filas")
    for label, (metric_key, history_key) in e2a.METRICS.items():
        recomputed = e2a.roc_auc(y_true, y_prob) if metric_key == "roc_auc" else e2a.average_precision(y_true, y_prob)
        add(f"{label}: validation_metrics.json = recalculado desde las predicciones",
            abs(recomputed - run["metrics"][metric_key]) <= e2a.TOLERANCE_RECOMPUTED,
            f"json={run['metrics'][metric_key]:.8f}, recalculado={recomputed:.8f}")

    history = run["history"]
    n_epochs = len(history["val_loss"])
    add("historial: longitud = épocas ejecutadas y AP por época presente",
        n_epochs == cfg["epochs_run"] and all(len(history[k]) == n_epochs for k in ("val_roc_auc", "val_average_precision"))
        and None not in history["val_average_precision"] + history["val_roc_auc"], f"{n_epochs} épocas")
    epoch_auc = cs.select_epoch(history["val_roc_auc"], max)
    add("checkpoint: época = primer máximo de val_roc_auc del historial (config, meta e historial coinciden)",
        epoch_auc == cfg["epoch_best"] == run["meta"]["epoch"], f"historial={epoch_auc}, run_config={cfg['epoch_best']}, meta={run['meta']['epoch']}")
    for label, (metric_key, history_key) in e2a.METRICS.items():
        add(f"{label}: historial en la época del checkpoint = validation_metrics.json",
            abs(history[history_key][cfg["epoch_best"] - 1] - run["metrics"][metric_key]) <= e2a.TOLERANCE_HISTORY,
            f"historial={history[history_key][cfg['epoch_best'] - 1]:.6f}, json={run['metrics'][metric_key]:.6f}")
    return rows


def verify_reference_values(protocol, reference_stats):
    """Los valores de referencia escritos en el protocolo coinciden con los de las corridas leídas (4 decimales)."""
    expected = protocol["base_y_referencias"]["valores_de_referencia_semilla_42"]
    rows = []
    for name in ("B0", "B2"):
        stats = reference_stats[name.lower()]
        folds = [stats[k] for k in range(N_FOLDS)]
        observed = {
            "ap_medio": round(statistics.mean(f["ap"] for f in folds), 4),
            "roc_auc_medio": round(statistics.mean(f["auc"] for f in folds), 4),
            "ap_por_fold": [round(f["ap"], 4) for f in folds],
        }
        for key, value in observed.items():
            rows.append({"corrida": f"referencia {name}", "comprobacion": f"valor del protocolo: {key}",
                         "ok": value == expected[name][key], "detalle": f"leído={value}, protocolo={expected[name][key]}"})
    return rows


# ---------------------------------------------------------------------------------------------------------
# Tablas, comparaciones y selección del finalista
# ---------------------------------------------------------------------------------------------------------
def registry_rows(protocol, candidates, runs, reference_runs):
    """registro_intentos.csv: una fila por configuración, fold y semilla probados (incluidas las fallidas)."""
    columns = protocol["registro"]["columnas"]
    failed_by_config = {}
    for run in runs:
        if run["status"] == STATUS_FAILED:
            failed_by_config.setdefault(run["config_id"], []).append(run["fold"])

    rows = []
    for run in runs:
        completed = run["status"] == STATUS_COMPLETED
        notes = []
        if run["status"] == STATUS_FAILED:
            notes.append("faltan: " + ", ".join(run["missing"]))
        if run["config_id"] in failed_by_config and run["status"] != STATUS_FAILED:
            notes.append(f"configuración descartada (cuenta como probada): fold(s) fallido(s) {sorted(failed_by_config[run['config_id']])}")
        rows.append({
            "etapa": "etapa_1", "config_id": run["config_id"], "fold": run["fold"], "seed": run["seed"],
            "flags": candidates[run["config_id"]]["flags"],
            "ap": run["metrics"]["average_precision"] if completed else "",
            "roc_auc": run["metrics"]["roc_auc"] if completed else "",
            "epoca_elegida": run["config"]["epoch_best"] if completed else "",
            "epocas_ejecutadas": run["config"]["epochs_run"] if completed else "",
            "estado": run["status"], "ruta": run["dir_rel"], "nota": "; ".join(notes),
        })
    for ref in reference_runs:
        rows.append({
            "etapa": "referencia", "config_id": ref["config_id"], "fold": ref["fold"], "seed": ref["seed"], "flags": ref["flags"],
            "ap": ref["metrics"]["average_precision"], "roc_auc": ref["metrics"]["roc_auc"],
            "epoca_elegida": ref["config"]["epoch_best"], "epocas_ejecutadas": ref["config"]["epochs_run"],
            "estado": STATUS_COMPLETED, "ruta": ref["dir_rel"], "nota": "corrida existente de robustez_e2 (no se repite)",
        })
    return [{column: row[column] for column in columns} for row in rows]


def build_stats(runs, reference_runs):
    """config_id -> fold -> {ap, auc, epoca, epocas} solo con corridas completadas y verificadas."""
    stats = {}
    for run in list(runs) + list(reference_runs):
        if run.get("status", STATUS_COMPLETED) != STATUS_COMPLETED:
            continue
        stats.setdefault(run["config_id"], {})[run["fold"]] = {
            "ap": run["metrics"]["average_precision"], "auc": run["metrics"]["roc_auc"],
            "epoca": run["config"]["epoch_best"], "epocas": run["config"]["epochs_run"],
        }
    return stats


def is_complete(stats, config_id):
    return config_id in stats and sorted(stats[config_id]) == list(range(N_FOLDS))


def metrics_rows(stats):
    return [{"config_id": config, "fold": fold, "ap": v["ap"], "roc_auc": v["auc"], "epoca_elegida": v["epoca"],
             "epocas_ejecutadas": v["epocas"]}
            for config in stats for fold, v in sorted(stats[config].items())]


def comparison_list(candidate_ids, stats):
    """Candidatas completas frente a B0 y B2 y, como referencia, B2 frente a B0."""
    pairs = []
    for config in candidate_ids:
        if is_complete(stats, config):
            pairs += [(f"{config}__vs_b0", config, "b0"), (f"{config}__vs_b2", config, "b2")]
    pairs.append(("b2__vs_b0", "b2", "b0"))
    return pairs


def delta_rows(stats, pairs):
    out = []
    for name, first, second in pairs:
        for label, key in (("AP", "ap"), ("ROC-AUC", "auc")):
            for fold in range(N_FOLDS):
                a, b = stats[first][fold][key], stats[second][fold][key]
                out.append({"comparacion": name, "primera": first, "segunda": second, "metrica": label, "fold": fold,
                            "valor_primera": a, "valor_segunda": b, "delta": a - b, "signo": e2a.sign_of(a - b)})
    return out


def summary_rows(delta_table, pairs):
    """media, desviación estándar muestral (n-1, descriptiva), mínimo, máximo, signos y consistencia (solo AP)."""
    out = []
    for name, first, second in pairs:
        for label in ("AP", "ROC-AUC"):
            values = [r["delta"] for r in delta_table if r["comparacion"] == name and r["metrica"] == label]
            summary = e2a.summarize(values)
            row = {"comparacion": name, "primera": first, "segunda": second, "metrica": label,
                   "media": summary["media"], "desv_est_muestral": statistics.stdev(values), "minimo": summary["minimo"],
                   "maximo": summary["maximo"], "folds_positivos": summary["folds_positivos"],
                   "folds_negativos": summary["folds_negativos"], "folds_cero": summary["folds_cero"],
                   "mejor_consistente": "", "peor_consistente": "", "veredicto": ""}
            if label == "AP":  # el veredicto de consistencia se basa solo en AP
                favored = e2a.consistency(summary)["favorece"]
                row["mejor_consistente"] = favored == "primera"
                row["peor_consistente"] = favored == "segunda"
                row["veredicto"] = {"primera": "mejor_consistente", "segunda": "peor_consistente"}.get(favored, "no_consistente")
            out.append(row)
    return out


def select_finalist(candidates, b0_auc_mean, thresholds):
    """
    Aplicación mecánica de seleccion_del_finalista. `candidates`: lista de dicts con id, complete, ap_mean, auc_mean
    (en el orden del protocolo). Elegibles: ΔROC-AUC medio frente a B0 >= umbral. Finalista: la elegible con mayor AP
    medio; empate (las elegibles cuyo AP medio difiere del máximo en menos de `empate_ap`): mayor ROC-AUC medio; luego
    menor id.
    """
    table = []
    for c in candidates:
        row = {"id": c["id"], "folds_completos": c["complete"], "descartada": not c["complete"]}
        if c["complete"]:
            row.update(ap_medio=c["ap_mean"], roc_auc_medio=c["auc_mean"], delta_auc_medio_vs_b0=c["auc_mean"] - b0_auc_mean)
            row["elegible"] = row["delta_auc_medio_vs_b0"] >= thresholds["delta_auc_minimo"]
            row["motivo"] = "" if row["elegible"] else f"ΔROC-AUC medio frente a B0 < {thresholds['delta_auc_minimo']}"
        else:
            row.update(ap_medio=None, roc_auc_medio=None, delta_auc_medio_vs_b0=None, elegible=False,
                       motivo="descartada: no completó los 5 folds (cuenta como probada)")
        table.append(row)

    eligible = sorted((r for r in table if r["elegible"]), key=lambda r: (-r["ap_medio"], r["id"]))
    result = {"candidatas": table, "elegibles": [r["id"] for r in eligible], "sin_elegibles": not eligible,
              "ap_maximo": None, "en_empate": [], "finalista": None, "el_desempate_cambia_el_finalista": None}
    if eligible:
        top = max(r["ap_medio"] for r in eligible)
        tied = [r for r in eligible if top - r["ap_medio"] < thresholds["empate_ap"]]
        winner = sorted(tied, key=lambda r: (-r["roc_auc_medio"], r["id"]))[0]
        result.update(ap_maximo=top, en_empate=[r["id"] for r in tied], finalista=winner["id"],
                      el_desempate_cambia_el_finalista=winner["id"] != eligible[0]["id"])
    return result


# ---------------------------------------------------------------------------------------------------------
# comandos_confirmacion.json: sustitución mecánica de la plantilla y verificación contra ella
# ---------------------------------------------------------------------------------------------------------
def confirmation_entries(protocol, candidates, finalist_id):
    """(id, flags) en el orden del protocolo: finalista, B0, B2."""
    flags = {item["id"]: item["flags"] for item in protocol["comandos"]["confirmacion"]["sustituciones"]["id_y_flags"]}
    return [(finalist_id, candidates[finalist_id]["flags"]), ("b0", ""), ("b2", flags["b2"])]


def generate_confirmation_commands(protocol, candidates, finalist_id):
    confirmation = protocol["comandos"]["confirmacion"]
    template, subs = confirmation["plantilla"], confirmation["sustituciones"]
    if " {flags}" not in template:
        raise ValueError("La plantilla no tiene el campo ' {flags}'.")
    commands = []
    for config_id, flags in confirmation_entries(protocol, candidates, finalist_id):
        text = template.replace(" {flags}", f" {flags}" if flags else "")  # sin flags: se omite el campo (sin espacios dobles)
        for seed in subs["seed"]:
            for fold in subs["k"]:
                command = text.format(seed=seed, k=fold, id=config_id)
                commands.append({"id": f"{config_id}_seed{seed}_fold{fold}", "config_id": config_id, "seed": seed, "fold": fold,
                                 "comando": command, "salida": e2a.parse_command(command)["--output-dir"]})
    return commands


def template_regex(template):
    """Expresión regular de la plantilla: cada campo es un grupo (las repeticiones usan retroreferencias)."""
    pattern, seen = "", set()
    parts = re.split(r"(\{\w+\})", template)
    for index, part in enumerate(parts):
        match = re.fullmatch(r"\{(\w+)\}", part)
        if not match:
            pattern += re.escape(part)
            continue
        name = match.group(1)
        if name in seen:
            pattern += f"(?P={name})"
        elif name == "flags":
            pattern = pattern[:-len(re.escape(" "))] if pattern.endswith(re.escape(" ")) else pattern
            pattern += r"(?: (?P<flags>\S.*?))?"  # el espacio y los flags son opcionales (B0 no tiene flags)
        else:
            pattern += rf"(?P<{name}>\d+|[A-Za-z0-9_]+)"
        seen.add(name)
    return re.compile(pattern)


def verify_confirmation_commands(commands, protocol, candidates, finalist_id):
    """Comprueba que los 30 comandos coinciden con la plantilla y con lo esperado para cada (id, semilla, fold)."""
    confirmation = protocol["comandos"]["confirmacion"]
    subs = confirmation["sustituciones"]
    regex = template_regex(confirmation["plantilla"])
    expected_flags = dict(confirmation_entries(protocol, candidates, finalist_id))
    rows = []

    def add(name, ok, detail=""):
        rows.append({"corrida": "comandos_confirmacion", "comprobacion": name, "ok": bool(ok), "detalle": detail})

    expected_total = protocol["metodo_de_busqueda"]["confirmacion"]["corridas"]
    add("30 comandos (3 configuraciones x 2 semillas x 5 folds)", len(commands) == expected_total == 30, str(len(commands)))
    add("comandos y salidas únicos", len({c["comando"] for c in commands}) == len(commands) == len({c["salida"] for c in commands}), "")
    add("sin espacios dobles ni espacios al final", all("  " not in c["comando"] and c["comando"] == c["comando"].strip() for c in commands), "")
    combos = {(c["config_id"], c["seed"], c["fold"]) for c in commands}
    add("todas las combinaciones id x semilla x fold",
        combos == {(i, s, k) for i in expected_flags for s in subs["seed"] for k in subs["k"]}, "")

    mismatches = []
    for c in commands:
        match = regex.fullmatch(c["comando"])
        if not match:
            mismatches.append(f"{c['id']}: no coincide con la plantilla")
            continue
        g = match.groupdict()
        flags = g.get("flags") or ""
        if (g["id"], int(g["seed"]), int(g["k"]), flags) != (c["config_id"], c["seed"], c["fold"], expected_flags[c["config_id"]]):
            mismatches.append(f"{c['id']}: valores {g} distintos de los esperados")
    add("cada comando coincide con la plantilla y con los flags de su configuración", not mismatches, "; ".join(mismatches[:3]))
    return rows


# ---------------------------------------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------------------------------------
def load_reference_runs(root, protocol, e2_protocol):
    """Las corridas de B0 y B2 (robustez_e2) según las rutas que indica el protocolo del agente."""
    references = protocol["base_y_referencias"]["referencias"]
    folders = {name.lower(): text.split(" ")[0] for name, text in references.items()}
    by_folder = {}
    for run in e2a.load_runs(root, e2_protocol):
        by_folder[(Path(run["dir"]).parent.as_posix(), run["fold"])] = run
    flags = {item["id"]: item["flags"] for item in protocol["comandos"]["confirmacion"]["sustituciones"]["id_y_flags"]}
    loaded = []
    for name, folder in folders.items():
        for fold in range(N_FOLDS):
            run = by_folder[((Path(root) / folder).as_posix(), fold)]
            loaded.append(dict(run, config_id=name, seed=int(run["args"]["--seed"]), flags="" if name == "b0" else flags.get(name, ""),
                               dir_rel=f"{folder}/fold_{fold}", status=STATUS_COMPLETED))
    return loaded


def run_analysis(root, output_dir=None, verbose=True):
    """
    Verifica las corridas y escribe las tablas. Devuelve (código de salida, resultados).
    No escribe nada si faltan corridas por ejecutar, si alguna comprobación falla o si algún archivo de salida existe.
    """
    root = Path(root)
    protocol, protocol_path = load_agent_protocol(root)
    e2_protocol, _ = e2a.load_protocol(root)
    candidates = candidate_table(protocol)
    folds_rows = read_csv_rows(root / CV_FOLDS_REL)
    out_dir = Path(output_dir) if output_dir else root / AGENT_REL / ANALYSIS_NAME

    runs = [load_agent_run(root, run) for run in stage1_runs(protocol)]
    not_run = [r["id"] for r in runs if r["status"] == STATUS_NOT_RUN]
    if verbose:
        print(f"Corridas de la etapa 1 en el protocolo: {len(runs)} | completadas: "
              f"{sum(r['status'] == STATUS_COMPLETED for r in runs)} | fallidas: {sum(r['status'] == STATUS_FAILED for r in runs)} "
              f"| sin carpeta: {len(not_run)}")
    if not_run:
        print(f"Faltan corridas por ejecutar (sin carpeta de salida; ¿falta el git pull?): {not_run[:5]}{' ...' if len(not_run) > 5 else ''}")
        return 1, None

    checks = [
        {"corrida": "protocolo", "comprobacion": "incluye la enmienda E1",
         "ok": any(e["id"] == "E1" for e in protocol["enmiendas"]), "detalle": str([e["id"] for e in protocol["enmiendas"]])},
        {"corrida": "protocolo", "comprobacion": "45 corridas = 9 candidatas x 5 folds, ids únicos",
         "ok": len(runs) == 45 == len({r["id"] for r in runs}) == len(candidates) * N_FOLDS, "detalle": str(len(runs))},
    ]
    identical = file_matches_commit(root, AGENT_REL / PROTOCOL_NAME, PROTOCOL_COMMIT, protocol_path)
    checks.append({"corrida": "protocolo", "comprobacion": f"idéntico al del commit {PROTOCOL_COMMIT}", "ok": identical is not False,
                   "detalle": "no verificable (git no disponible)" if identical is None else str(identical)})
    for run in runs:
        if run["status"] == STATUS_FAILED:
            checks.append({"corrida": run["id"], "comprobacion": "corrida fallida (registrada; no entra en la selección)", "ok": True,
                           "detalle": "faltan: " + ", ".join(run["missing"])})
        else:
            checks += verify_agent_run(run, candidates[run["config_id"]], folds_rows)

    reference_runs = load_reference_runs(root, protocol, e2_protocol)
    for ref in reference_runs:
        checks += e2a.verify_run(ref, e2_protocol, folds_rows)
    reference_stats = build_stats([], reference_runs)
    checks += verify_reference_values(protocol, reference_stats)

    failed = [c for c in checks if not c["ok"]]
    if verbose:
        print(f"Comprobaciones: {len(checks)} | fallidas: {len(failed)}")
    if failed:
        for c in failed:
            print(f"  FALLA {c['corrida']}: {c['comprobacion']} -> {c['detalle']}")
        return 1, None

    stats = build_stats(runs, reference_runs)
    thresholds = selection_thresholds(protocol)
    b0_auc = statistics.mean(stats["b0"][k]["auc"] for k in range(N_FOLDS))
    candidate_stats = [
        {"id": cid, "complete": is_complete(stats, cid),
         "ap_mean": statistics.mean(stats[cid][k]["ap"] for k in range(N_FOLDS)) if is_complete(stats, cid) else None,
         "auc_mean": statistics.mean(stats[cid][k]["auc"] for k in range(N_FOLDS)) if is_complete(stats, cid) else None}
        for cid in candidates
    ]
    selection = select_finalist(candidate_stats, b0_auc, thresholds)
    for row in selection["candidatas"]:
        row["flags"] = candidates[row["id"]]["flags"]
        if row["folds_completos"]:
            row["delta_ap_medio_vs_b0"] = row["ap_medio"] - statistics.mean(stats["b0"][k]["ap"] for k in range(N_FOLDS))

    pairs = comparison_list(list(candidates), stats)
    delta_table = delta_rows(stats, pairs)
    summaries = summary_rows(delta_table, pairs)
    registry = registry_rows(protocol, candidates, runs, reference_runs)

    commands_payload = None
    if selection["finalista"] is not None:
        commands = generate_confirmation_commands(protocol, candidates, selection["finalista"])
        command_checks = verify_confirmation_commands(commands, protocol, candidates, selection["finalista"])
        if any(not c["ok"] for c in command_checks):
            for c in command_checks:
                if not c["ok"]:
                    print(f"  FALLA {c['comprobacion']} -> {c['detalle']}")
            return 1, None
        checks += command_checks
        commands_payload = {
            "estado": "generado por sustitución mecánica de la plantilla del protocolo; pendiente de que el autor verifique que coinciden con ella antes de correr",
            "protocolo": f"{AGENT_REL.as_posix()}/{PROTOCOL_NAME} (commit {PROTOCOL_COMMIT}, con la enmienda E1)",
            "finalista": selection["finalista"], "flags_del_finalista": candidates[selection["finalista"]]["flags"],
            "plantilla": protocol["comandos"]["confirmacion"]["plantilla"], "total": len(commands),
            "verificacion_contra_la_plantilla": {c["comprobacion"]: c["ok"] for c in command_checks},
            "comandos": commands,
        }

    existing = [out_dir / name for name in OUTPUT_FILES.values() if (out_dir / name).exists()]
    if existing:
        raise FileExistsError("Ya existen archivos de salida; no se sobrescribe ninguno:\n" + "\n".join(str(p) for p in existing))

    selection_payload = {
        "protocolo": {"archivo": f"{AGENT_REL.as_posix()}/{PROTOCOL_NAME}", "commit_de_referencia": PROTOCOL_COMMIT,
                      "idéntico_al_commit": identical, "enmiendas": [e["id"] for e in protocol["enmiendas"]], "estado": protocol["estado"]},
        "regla": {"texto": protocol["metodo_de_busqueda"]["seleccion_del_finalista"], "umbrales_leidos_del_texto": thresholds,
                  "interpretacion_del_empate": "se consideran empatadas las elegibles cuyo AP medio difiere del AP medio máximo en menos del umbral; entre ellas decide la mayor ROC-AUC media y luego el menor id"},
        "referencias": {name: {"ap_medio": statistics.mean(stats[name][k]["ap"] for k in range(N_FOLDS)),
                               "roc_auc_medio": statistics.mean(stats[name][k]["auc"] for k in range(N_FOLDS))} for name in ("b0", "b2")},
        "conteo": {"corridas_etapa_1": len(runs), "completadas": sum(r["status"] == STATUS_COMPLETED for r in runs),
                   "fallidas": sum(r["status"] == STATUS_FAILED for r in runs), "candidatas_probadas": len(candidates),
                   "referencias": 2},
        **selection,
        "consecuencia": ("hay finalista: se generan los 30 comandos de confirmación (comandos_confirmacion.json)"
                         if selection["finalista"] else
                         "sin_elegibles: " + protocol["metodo_de_busqueda"]["seleccion_del_finalista"]["sin_elegibles"]),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    cs.write_csv(out_dir / OUTPUT_FILES["registro"], registry)
    cs.write_csv(out_dir / OUTPUT_FILES["verification"], checks)
    cs.write_csv(out_dir / OUTPUT_FILES["metrics"], metrics_rows(stats))
    cs.write_csv(out_dir / OUTPUT_FILES["delta_by_fold"], delta_table)
    cs.write_csv(out_dir / OUTPUT_FILES["delta_summary"], summaries)
    e2a.write_json(out_dir / OUTPUT_FILES["selection"], selection_payload)
    if commands_payload is not None:
        e2a.write_json(out_dir / OUTPUT_FILES["commands"], commands_payload)

    if verbose:
        print(f"Elegibles: {selection['elegibles']} | en empate: {selection['en_empate']} | finalista: {selection['finalista']}")
        print("Escrito en", out_dir)
    return 0, {"checks": checks, "registry": registry, "stats": stats, "deltas": delta_table, "summaries": summaries,
               "selection": selection_payload, "commands": commands_payload}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Análisis de la etapa 1 del brazo del agente según protocolo_agente.json.")
    parser.add_argument("--output-dir", default=None, help=f"Carpeta de salida (por defecto {AGENT_REL.as_posix()}/{ANALYSIS_NAME}/).")
    args = parser.parse_args(argv)
    code, _ = run_analysis(Path.cwd(), args.output_dir)
    return code


if __name__ == "__main__":
    sys.exit(main())
