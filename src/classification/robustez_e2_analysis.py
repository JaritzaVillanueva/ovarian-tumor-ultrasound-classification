"""
Análisis de la verificación de robustez E2 (15 corridas), aplicando literalmente las reglas de lectura de
results/classification/r7_optimizacion/robustez_e2/protocolo_robustez_e2.json. No cambia ninguna regla: las
configuraciones, las comparaciones y las corridas se leen del propio protocolo.

Solo lee archivos de las corridas E2 (validation_metrics.json, run_config.json, training_history.json,
best_auc_meta.json, fold_manifest.csv, val_predictions_best_auc.csv). No entrena, no carga checkpoints y no usa test.
Solo usa la biblioteca estándar.

Uso (desde la raíz del repositorio):
    python -m src.classification.robustez_e2_analysis
Escribe en robustez_e2/analisis/ y no sobrescribe archivos existentes.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

from src.classification import checkpoint_sensitivity as cs
from src.classification.cv_folds_e2 import N_FOLDS, OUTPUT_REL as CV_FOLDS_REL, read_csv_rows

E2_REL = Path("results") / "classification" / "r7_optimizacion" / "robustez_e2"
PROTOCOL_NAME = "protocolo_robustez_e2.json"
ANALYSIS_NAME = "analisis"
PROTOCOL_COMMIT = "6de5349"  # commit en el que se congeló el protocolo

# Métrica -> (clave en validation_metrics.json, clave en training_history.json)
METRICS = {"AP": ("average_precision", "val_average_precision"), "ROC-AUC": ("roc_auc", "val_roc_auc")}
PRIMARY_METRIC = "AP"

# Hiperparámetros declarados en el protocolo ("entrenamiento"): YAML sin cambios + early stopping de E1.
EXPECTED_TRAINING = {
    "learning_rate": 1e-4, "weight_decay": 1e-4, "batch_size": 32, "epochs_max": 30, "patience": 5,
    "min_delta": 0.001, "use_amp": True, "pretrained": True,
}
TOLERANCE_RECOMPUTED = 1e-6  # métricas recalculadas desde val_predictions frente a validation_metrics.json
TOLERANCE_HISTORY = 1e-4  # historial por época frente al checkpoint recargado (misma tolerancia que run_experiment)

OUTPUT_FILES = {
    "verification": "verificacion_corridas.csv",
    "metrics": "metricas_por_configuracion_y_fold.csv",
    "delta_by_fold": "delta_por_fold.csv",
    "delta_summary": "resumen_delta.csv",
    "rules": "aplicacion_de_reglas.json",
    "sens_by_fold": "sensibilidad_checkpoint_por_fold.csv",
    "sens_delta_by_fold": "sensibilidad_checkpoint_delta_por_fold.csv",
    "sens_summary": "sensibilidad_checkpoint_resumen.csv",
}


# ---------------------------------------------------------------------------------------------------------
# Métricas recalculadas desde las probabilidades guardadas (biblioteca estándar)
# ---------------------------------------------------------------------------------------------------------
def roc_auc(y_true, y_score):
    pos = [s for y, s in zip(y_true, y_score) if y == 1]
    neg = [s for y, s in zip(y_true, y_score) if y == 0]
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def average_precision(y_true, y_score):
    """AP como en scikit-learn: suma de (R_n - R_{n-1}) * P_n sobre los umbrales distintos."""
    order = sorted(range(len(y_true)), key=lambda i: -y_score[i])
    n_pos = sum(y_true)
    tp = fp = 0
    previous_recall = 0.0
    ap = 0.0
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and y_score[order[j]] == y_score[order[i]]:
            j += 1
        for k in range(i, j):
            if y_true[order[k]] == 1:
                tp += 1
            else:
                fp += 1
        recall, precision = tp / n_pos, tp / (tp + fp)
        ap += (recall - previous_recall) * precision
        previous_recall = recall
        i = j
    return ap


# ---------------------------------------------------------------------------------------------------------
# Protocolo y carga de corridas
# ---------------------------------------------------------------------------------------------------------
def load_protocol(root):
    path = Path(root) / E2_REL / PROTOCOL_NAME
    return json.loads(path.read_text(encoding="utf-8")), path


def parse_command(command):
    """Argumentos '--clave valor' de un comando del protocolo."""
    tokens = command.split()
    return {tokens[i]: tokens[i + 1] for i in range(len(tokens) - 1) if tokens[i].startswith("--")}


def parse_comparisons(protocol):
    """Comparaciones del protocolo: delta = primera - segunda, a partir del texto 'AP(a) - AP(b)'."""
    out = []
    for comparison in protocol["comparaciones"]:
        match = re.search(r"AP\((\w+)\)\s*-\s*AP\((\w+)\)", comparison["delta"])
        if not match:
            raise ValueError(f"No se pudo leer el delta de {comparison['id']}: {comparison['delta']}")
        out.append({"id": comparison["id"], "nombre": comparison["nombre"],
                    "first": match.group(1), "second": match.group(2)})
    return out


def protocol_matches_commit(root, protocol_path, commit=PROTOCOL_COMMIT):
    """True/False si el protocolo es idéntico al del commit; None si git no está disponible."""
    relative = (Path(E2_REL) / PROTOCOL_NAME).as_posix()
    try:
        result = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=root, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    normalize = lambda b: b.replace(b"\r\n", b"\n")
    return normalize(result.stdout) == normalize(Path(protocol_path).read_bytes())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_runs(root, protocol):
    """Una entrada por corrida del protocolo, con sus archivos leídos."""
    runs = []
    for item in protocol["comandos_para_falcon"]["corridas"]:
        folder = Path(root) / item["salida"]
        args = parse_command(item["comando"])
        runs.append({
            "id": item["id"], "configuracion": item["configuracion"], "fold": item["fold"], "args": args,
            "dir": folder,
            "metrics": read_json(folder / "validation_metrics.json"),
            "config": read_json(folder / "run_config.json"),
            "history": read_json(folder / "training_history.json"),
            "meta": read_json(folder / "best_auc_meta.json"),
            "manifest": read_csv_rows(folder / "fold_manifest.csv"),
            "predictions": read_csv_rows(folder / "val_predictions_best_auc.csv"),
        })
    return runs


def expected_class_weights(labels, alpha):
    counts = [labels.count(0), labels.count(1)]
    total = sum(counts)
    if alpha == 1.0:
        return [total / (2 * c) for c in counts]
    smoothed = [c ** (-alpha) for c in counts]
    denominator = sum(c * s for c, s in zip(counts, smoothed))
    return [total * s / denominator for s in smoothed]


def verify_run(run, protocol, folds_rows):
    """Comprobaciones de una corrida. Devuelve filas (corrida, comprobacion, ok, detalle)."""
    cfg, args, fold = run["config"], run["args"], run["fold"]
    checks = []

    def add(name, ok, detail=""):
        checks.append({"corrida": run["id"], "comprobacion": name, "ok": bool(ok), "detalle": detail})

    alpha = protocol["configuraciones"][run["configuracion"]]["class_weight_alpha"]
    add("config: esquema E2 y fold", cfg.get("evaluation_scheme") == "E2" and cfg.get("fold") == fold,
        f"evaluation_scheme={cfg.get('evaluation_scheme')}, fold={cfg.get('fold')}")
    add("config: arquitectura, condición, experimento, semilla",
        cfg["architecture"] == args["--arch"] and cfg["condition"] == args["--condition"]
        and cfg["experiment"] == args["--experiment"] and cfg["seed"] == int(args["--seed"]),
        f"{cfg['architecture']}/{cfg['condition']}/{cfg['experiment']}/seed {cfg['seed']}")
    add("config: partición", cfg["csv_path"] == args["--split-csv"] == CV_FOLDS_REL, cfg["csv_path"])
    add("config: pesos de clase (exponente)", cfg.get("class_weight_alpha", 1.0) == alpha, f"alpha={cfg.get('class_weight_alpha', 1.0)}")
    differing = {k: cfg[k] for k in EXPECTED_TRAINING if cfg[k] != EXPECTED_TRAINING[k]}
    add("config: hiperparámetros del protocolo", not differing, str(differing) if differing else "")
    add("config: criterio de checkpoint max val_roc_auc", cfg["checkpoint_criterion"].startswith("max val_roc_auc"), cfg["checkpoint_criterion"])
    add("config: early stopping activo", cfg["patience"] == 5 and cfg["min_delta"] == 0.001 and "disabled" not in cfg.get("early_stopping_criterion", ""), "")
    add("config: 800 train y 200 validation", cfg["train_samples"] == 800 and cfg["validation_samples"] == 200,
        f"{cfg['train_samples']}/{cfg['validation_samples']}")

    manifest = run["manifest"]
    fold_ids = {r["image_id"] for r in folds_rows if int(r["fold"]) == fold}
    validation = [r for r in manifest if r["experimental_split"] == "validation"]
    train = [r for r in manifest if r["experimental_split"] == "train"]
    labels_train = [int(r["binary_label"]) for r in train]
    add("manifiesto: 1000 imágenes, ninguna de test",
        len(manifest) == 1000 and {r["e1_split"] for r in manifest} <= {"train", "validation"}
        and len({r["image_id"] for r in manifest}) == 1000, f"{len(manifest)} filas")
    add("manifiesto: validation = fold de cv_folds_e2.csv",
        {r["image_id"] for r in validation} == fold_ids and len(validation) == 200, f"{len(validation)} imágenes")
    add("pesos de clase = recalculados con las etiquetas de train del fold",
        all(abs(a - b) <= 1e-5 * max(1.0, abs(b)) for a, b in zip(cfg["class_weights"], expected_class_weights(labels_train, alpha))),
        str(cfg["class_weights"]))

    predictions = run["predictions"]
    y_true = [int(r["y_true"]) for r in predictions]
    y_prob = [float(r["probability_malignant"]) for r in predictions]
    add("predicciones: mismas imágenes y orden que validation del manifiesto, época del checkpoint",
        [r["image_id"] for r in predictions] == [r["image_id"] for r in validation]
        and y_true == [int(r["binary_label"]) for r in validation]
        and {int(r["epoch"]) for r in predictions} == {cfg["epoch_best"]}, f"{len(predictions)} filas")
    for label, (metric_key, _) in METRICS.items():
        recomputed = roc_auc(y_true, y_prob) if metric_key == "roc_auc" else average_precision(y_true, y_prob)
        add(f"{label}: validation_metrics.json = recalculado desde las predicciones",
            abs(recomputed - run["metrics"][metric_key]) <= TOLERANCE_RECOMPUTED,
            f"json={run['metrics'][metric_key]:.8f}, recalculado={recomputed:.8f}")

    history = run["history"]
    n_epochs = len(history["val_loss"])
    add("historial: longitud = épocas ejecutadas y AP por época presente",
        n_epochs == cfg["epochs_run"] and all(len(history[k]) == n_epochs for k in ("val_roc_auc", "val_average_precision"))
        and None not in history["val_average_precision"] + history["val_roc_auc"], f"{n_epochs} épocas")
    epoch_auc = cs.select_epoch(history["val_roc_auc"], max)
    add("checkpoint: época = primer máximo de val_roc_auc del historial (config, meta e historial coinciden)",
        epoch_auc == cfg["epoch_best"] == run["meta"]["epoch"], f"historial={epoch_auc}, run_config={cfg['epoch_best']}, meta={run['meta']['epoch']}")
    for label, (metric_key, history_key) in METRICS.items():
        add(f"{label}: historial en la época del checkpoint = validation_metrics.json",
            abs(history[history_key][cfg["epoch_best"] - 1] - run["metrics"][metric_key]) <= TOLERANCE_HISTORY,
            f"historial={history[history_key][cfg['epoch_best'] - 1]:.6f}, json={run['metrics'][metric_key]:.6f}")
    return checks


# ---------------------------------------------------------------------------------------------------------
# Tablas y reglas
# ---------------------------------------------------------------------------------------------------------
def sign_of(value):
    return "+" if value > 0 else "-" if value < 0 else "0"


def metrics_rows(runs):
    rows = []
    for run in sorted(runs, key=lambda r: (r["configuracion"], r["fold"])):
        validation = [r for r in run["manifest"] if r["experimental_split"] == "validation"]
        rows.append({
            "configuracion": run["configuracion"], "fold": run["fold"],
            "ap": run["metrics"]["average_precision"], "roc_auc": run["metrics"]["roc_auc"],
            "epoca_elegida": run["config"]["epoch_best"], "epocas_ejecutadas": run["config"]["epochs_run"],
            "n_validation": len(validation), "n_positivas": sum(int(r["binary_label"]) for r in validation),
        })
    return rows


def delta_rows(metric_rows, comparisons):
    """Delta por fold (primera - segunda) para AP y ROC-AUC, con el valor sin redondear."""
    by_key = {(r["configuracion"], r["fold"]): r for r in metric_rows}
    out = []
    for comparison in comparisons:
        for label, column in (("AP", "ap"), ("ROC-AUC", "roc_auc")):
            for fold in range(N_FOLDS):
                first, second = by_key[(comparison["first"], fold)][column], by_key[(comparison["second"], fold)][column]
                delta = first - second
                out.append({"comparacion": comparison["id"], "metrica": label, "fold": fold,
                            "primera": comparison["first"], "segunda": comparison["second"],
                            "valor_primera": first, "valor_segunda": second, "delta": delta, "signo": sign_of(delta)})
    return out


def summarize(deltas):
    """media, mínimo, máximo y conteo de signos de una lista de delta (cinco folds)."""
    return {
        "n_folds": len(deltas), "media": statistics.mean(deltas), "minimo": min(deltas), "maximo": max(deltas),
        "folds_positivos": sum(d > 0 for d in deltas), "folds_negativos": sum(d < 0 for d in deltas),
        "folds_cero": sum(d == 0 for d in deltas),
    }


def summary_rows(delta_table, comparisons):
    out = []
    for comparison in comparisons:
        for label in METRICS:
            values = [r["delta"] for r in delta_table if r["comparacion"] == comparison["id"] and r["metrica"] == label]
            out.append({"comparacion": comparison["id"], "metrica": label, **summarize(values)})
    return out


def consistency(summary, minimum=4):
    """Consistente si el mismo signo estricto aparece en al menos `minimum` folds (el 0 no cuenta)."""
    positive, negative = summary["folds_positivos"], summary["folds_negativos"]
    favored = "primera" if positive >= minimum else "segunda" if negative >= minimum else None
    return {"consistente": favored is not None, "favorece": favored}


def apply_rules(summaries, comparisons, protocol, protocol_check):
    """Aplicación literal de las reglas de lectura. Las frases se construyen a partir del texto del protocolo."""
    rules = protocol["reglas_de_lectura"]
    names = {c["id"]: c for c in comparisons}
    by_key = {(s["comparacion"], s["metrica"]): s for s in summaries}
    result = {"protocolo": {"archivo": f"{E2_REL.as_posix()}/{PROTOCOL_NAME}", "commit_de_referencia": PROTOCOL_COMMIT,
                            "idéntico_al_commit": protocol_check, "estado": protocol["estado"]},
              "metrica_del_veredicto": PRIMARY_METRIC, "comparaciones": {}}

    for cid, comparison in names.items():
        s = by_key[(cid, PRIMARY_METRIC)]
        verdict = consistency(s)
        first, second = comparison["first"], comparison["second"]
        counts = f"+{s['folds_positivos']} / -{s['folds_negativos']} / 0:{s['folds_cero']} en {s['n_folds']} folds"
        if verdict["consistente"]:
            winner, loser = (first, second) if verdict["favorece"] == "primera" else (second, first)
            phrase = (f"Comparación {cid} consistente en AP ({counts}): puede declararse que {winner} es mejor que {loser} "
                      f"en AP en esta partición, con el AP medio y el rango como únicos estadísticos.")
        else:
            phrase = (f"Comparación {cid} no consistente en AP ({counts}): no se declara que {first} sea mejor que {second} "
                      f"ni lo contrario.")
        roc = by_key[(cid, "ROC-AUC")]
        result["comparaciones"][cid] = {
            "nombre": comparison["nombre"], "delta": f"{first} - {second}", "regla_consistencia": rules["consistencia"],
            "regla_declarar_mejor": rules["declarar_mejor"], "resumen_ap": s, **verdict,
            "frase_del_veredicto": phrase, "nota_frase": "formulación derivada de las reglas 'consistencia' y 'declarar_mejor'; el protocolo no fija su texto",
            "roc_auc_descriptivo": {**roc, "nota": "metrica_secundaria: se reporta sin veredicto de consistencia"},
        }

    c1 = by_key[("C1", PRIMARY_METRIC)]
    b0_not_robust = c1["folds_negativos"] >= 4
    result["regla_robustez_de_b0"] = {
        "texto_del_protocolo": rules["regla_robustez_de_b0"],
        "condicion": "delta C1 (ConvNeXt-A - DenseNet-A) en AP estrictamente negativo en al menos 4 de 5 folds",
        "folds_negativos": c1["folds_negativos"], "se_cumple": b0_not_robust,
        "frase_del_veredicto": (
            "La elección de B0 no es robusta a esta partición: DenseNet-A supera a ConvNeXt-A en AP en "
            f"{c1['folds_negativos']} de 5 folds. Esto no modifica R8: B0 y B2 están congelados."
            if b0_not_robust else
            f"La regla no se cumple (DenseNet-A supera a ConvNeXt-A en AP en {c1['folds_negativos']} de 5 folds): "
            "el informe no concluye que B0 sea robusto y reporta media, rango y signos."),
    }
    result["regla_c2"] = {
        "existe_en_el_protocolo": "regla_c2" in rules,
        "nota": ("El protocolo no define una regla específica para C2: se lee con 'consistencia' y 'declarar_mejor' (ver comparaciones.C2). "
                 "Añadir una regla ahora, con los resultados ya conocidos, contradiría 'no_se_hace: cambiar estas reglas después de ver los resultados'."
                 if "regla_c2" not in rules else "El protocolo define regla_c2; esta versión del análisis no la implementa."),
    }
    result["otras_reglas"] = {"p_valor": rules["p_valor"], "metrica_secundaria": rules["metrica_secundaria"],
                              "no_se_hace": rules["no_se_hace"]}
    return result


# ---------------------------------------------------------------------------------------------------------
# Sensibilidad del criterio de checkpoint por fold (descriptiva; no modifica el veredicto)
# ---------------------------------------------------------------------------------------------------------
def sensitivity_rows(runs):
    rows = []
    for run in sorted(runs, key=lambda r: (r["configuracion"], r["fold"])):
        history = run["history"]
        for criterion, (key, direction) in cs.CRITERIA.items():
            epoch = cs.select_epoch(history[key], direction)
            rows.append({
                "configuracion": run["configuracion"], "fold": run["fold"], "criterio": criterion,
                "epoca_elegida": epoch, "epocas_ejecutadas": len(history["val_loss"]),
                "val_average_precision": history["val_average_precision"][epoch - 1],
                "val_roc_auc": history["val_roc_auc"][epoch - 1], "val_weighted_loss": history["val_loss"][epoch - 1],
                "epoca_vigente_run_config": run["config"]["epoch_best"],
                "auc_max_coincide_con_run_config": (epoch == run["config"]["epoch_best"]) if criterion == cs.CURRENT_CRITERION else "",
            })
    return rows


def sensitivity_deltas(sens_rows, comparisons):
    """Delta de AP por fold (primera - segunda) con la época que elegiría cada criterio."""
    by_key = {(r["configuracion"], r["fold"], r["criterio"]): r for r in sens_rows}
    out = []
    for comparison in comparisons:
        for criterion in cs.CRITERIA:
            for fold in range(N_FOLDS):
                first = by_key[(comparison["first"], fold, criterion)]["val_average_precision"]
                second = by_key[(comparison["second"], fold, criterion)]["val_average_precision"]
                out.append({"comparacion": comparison["id"], "criterio": criterion, "fold": fold,
                            "ap_primera": first, "ap_segunda": second, "delta_ap": first - second, "signo": sign_of(first - second)})
    return out


def sensitivity_summary(delta_table, comparisons):
    out = []
    for comparison in comparisons:
        reference = [r for r in delta_table if r["comparacion"] == comparison["id"] and r["criterio"] == cs.CURRENT_CRITERION]
        ref_summary = summarize([r["delta_ap"] for r in reference])
        ref_consistent = consistency(ref_summary)
        for criterion in cs.CRITERIA:
            rows = [r for r in delta_table if r["comparacion"] == comparison["id"] and r["criterio"] == criterion]
            s = summarize([r["delta_ap"] for r in rows])
            verdict = consistency(s)
            out.append({
                "comparacion": comparison["id"], "criterio": criterion, **s,
                "consistente_4_de_5": verdict["consistente"], "favorece": verdict["favorece"] or "ninguna",
                "folds_con_signo_distinto_a_auc_max": sum(a["signo"] != b["signo"] for a, b in zip(rows, reference)),
                "signo_de_la_media_distinto_a_auc_max": sign_of(s["media"]) != sign_of(ref_summary["media"]),
                "veredicto_de_consistencia_distinto_a_auc_max": (verdict["consistente"], verdict["favorece"]) != (ref_consistent["consistente"], ref_consistent["favorece"]),
            })
    return out


# ---------------------------------------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------------------------------------
def write_json(path, data):
    if Path(path).exists():
        raise FileExistsError(f"{path} ya existe; no se sobrescribe.")
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_analysis(root, output_dir=None, verbose=True):
    """
    Verifica las 15 corridas y escribe las tablas. Devuelve (código de salida, resultados).
    Si alguna comprobación falla no escribe nada. Si algún archivo de salida ya existe, no escribe nada.
    """
    root = Path(root)
    protocol, protocol_path = load_protocol(root)
    comparisons = parse_comparisons(protocol)
    folds_rows = read_csv_rows(root / CV_FOLDS_REL)
    runs = load_runs(root, protocol)
    out_dir = Path(output_dir) if output_dir else root / E2_REL / ANALYSIS_NAME

    checks = []
    for run in runs:
        checks += verify_run(run, protocol, folds_rows)
    protocol_check = protocol_matches_commit(root, protocol_path)
    checks.append({"corrida": "protocolo", "comprobacion": f"idéntico al del commit {PROTOCOL_COMMIT}",
                   "ok": protocol_check is not False, "detalle": "no verificable (git no disponible)" if protocol_check is None else str(protocol_check)})
    failed = [c for c in checks if not c["ok"]]
    if verbose:
        print(f"Corridas: {len(runs)} | comprobaciones: {len(checks)} | fallidas: {len(failed)}")
    if failed:
        for c in failed:
            print(f"  FALLA {c['corrida']}: {c['comprobacion']} -> {c['detalle']}")
        return 1, None

    existing = [out_dir / name for name in OUTPUT_FILES.values() if (out_dir / name).exists()]
    if existing:
        raise FileExistsError("Ya existen archivos de salida; no se sobrescribe ninguno:\n" + "\n".join(str(p) for p in existing))

    metric_table = metrics_rows(runs)
    delta_table = delta_rows(metric_table, comparisons)
    summaries = summary_rows(delta_table, comparisons)
    rules = apply_rules(summaries, comparisons, protocol, protocol_check)
    sens = sensitivity_rows(runs)
    sens_deltas = sensitivity_deltas(sens, comparisons)
    sens_summary = sensitivity_summary(sens_deltas, comparisons)

    out_dir.mkdir(parents=True, exist_ok=True)
    cs.write_csv(out_dir / OUTPUT_FILES["verification"], checks)
    cs.write_csv(out_dir / OUTPUT_FILES["metrics"], metric_table)
    cs.write_csv(out_dir / OUTPUT_FILES["delta_by_fold"], delta_table)
    cs.write_csv(out_dir / OUTPUT_FILES["delta_summary"], summaries)
    write_json(out_dir / OUTPUT_FILES["rules"], rules)
    cs.write_csv(out_dir / OUTPUT_FILES["sens_by_fold"], sens)
    cs.write_csv(out_dir / OUTPUT_FILES["sens_delta_by_fold"], sens_deltas)
    cs.write_csv(out_dir / OUTPUT_FILES["sens_summary"], sens_summary)

    if verbose:
        for s in summaries:
            print(f"  {s['comparacion']} {s['metrica']:<8} media {s['media']:+.4f} | rango [{s['minimo']:+.4f}, {s['maximo']:+.4f}] "
                  f"| +{s['folds_positivos']} -{s['folds_negativos']} 0:{s['folds_cero']}")
        for cid, block in rules["comparaciones"].items():
            print(f"  {block['frase_del_veredicto']}")
        print(f"  {rules['regla_robustez_de_b0']['frase_del_veredicto']}")
        print(f"  regla_c2 existe en el protocolo: {rules['regla_c2']['existe_en_el_protocolo']}")
        print(f"Escrito en {out_dir}")
    return 0, {"checks": checks, "metrics": metric_table, "deltas": delta_table, "summaries": summaries, "rules": rules,
               "sensitivity": sens, "sensitivity_summary": sens_summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Análisis de la robustez E2 según protocolo_robustez_e2.json.")
    parser.add_argument("--output-dir", default=None, help=f"Carpeta de salida (por defecto {E2_REL.as_posix()}/{ANALYSIS_NAME}/).")
    args = parser.parse_args(argv)
    code, _ = run_analysis(Path.cwd(), args.output_dir)
    return code


if __name__ == "__main__":
    sys.exit(main())
