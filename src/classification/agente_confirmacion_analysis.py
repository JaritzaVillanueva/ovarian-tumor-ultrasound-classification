"""
Análisis de la CONFIRMACIÓN del brazo del agente de optimización (R7), aplicando literalmente
results/classification/r7_optimizacion/agente/protocolo_agente.json (con la enmienda E1): sección
metodo_de_busqueda.confirmacion, reglas_de_lectura y regla_de_eleccion. No cambia ningún umbral ni criterio: los
umbrales se leen del texto del protocolo y los puntos que el protocolo no define se informan como lecturas
alternativas, sin elegir una.

Lee (solo archivos de validation; no entrena, no carga checkpoints y no usa test):
    - las 30 corridas de agente/confirmacion/<config>/seed_<s>/fold_<k>/ (finalista, B0 y B2; semillas 43 y 44);
    - los comandos de agente/analisis_etapa1/comandos_confirmacion.json y el finalista de seleccion_finalista.json;
    - las referencias B0 y B2 de robustez_e2/corridas/ (semilla 42), solo para el contraste B2 frente a B0.
Escribe en agente/confirmacion/analisis/ sin sobrescribir:
    registro_confirmacion.csv, verificacion_corridas.csv, metricas_por_corrida.csv, metricas_por_fold.csv,
    delta_por_fold.csv, resumen_delta.csv, contraste_b2_vs_b0_delta.csv, contraste_b2_vs_b0_resumen.csv y
    veredicto_confirmacion.json (veredicto mecánico de regla_de_eleccion).

No calcula intervalos de confianza ni p-valores: reglas_de_lectura.p_valor lo excluye con 5 folds.
Solo usa la biblioteca estándar.

Uso (desde la raíz del repositorio):
    python -m src.classification.agente_confirmacion_analysis
"""
from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from pathlib import Path

from src.classification import agente_etapa1_analysis as a1
from src.classification import checkpoint_sensitivity as cs
from src.classification import robustez_e2_analysis as e2a
from src.classification.cv_folds_e2 import N_FOLDS, OUTPUT_REL as CV_FOLDS_REL, read_csv_rows

AGENT_REL = a1.AGENT_REL
CONFIRMATION_REL = AGENT_REL / "confirmacion"
ANALYSIS_NAME = "analisis"
STAGE1_ANALYSIS_REL = AGENT_REL / a1.ANALYSIS_NAME
COMMANDS_NAME = a1.OUTPUT_FILES["commands"]
SELECTION_NAME = a1.OUTPUT_FILES["selection"]

OUTPUT_FILES = {
    "registro": "registro_confirmacion.csv",
    "verification": "verificacion_corridas.csv",
    "runs": "metricas_por_corrida.csv",
    "by_fold": "metricas_por_fold.csv",
    "delta_by_fold": "delta_por_fold.csv",
    "delta_summary": "resumen_delta.csv",
    "contrast_delta": "contraste_b2_vs_b0_delta.csv",
    "contrast_summary": "contraste_b2_vs_b0_resumen.csv",
    "verdict": "veredicto_confirmacion.json",
}

METRIC_KEYS = (("AP", "ap"), ("ROC-AUC", "auc"))
READING_A, READING_B, READING_C = "A_promedio_por_fold_de_las_3_semillas", "B_por_semilla", "C_promedio_por_fold_de_las_semillas_43_y_44"


# ---------------------------------------------------------------------------------------------------------
# Reglas del protocolo (se leen del texto; no se fijan aquí)
# ---------------------------------------------------------------------------------------------------------
def read_confirmation_design(protocol):
    """Semillas, folds y total de la confirmación tal como están escritos en el protocolo."""
    design = protocol["metodo_de_busqueda"]["confirmacion"]
    return {"seeds": [int(s) for s in design["semillas"]], "folds": [int(k) for k in design["folds"]],
            "total": int(design["corridas"]), "configs": list(design["configuraciones"])}


def read_decision_rule(protocol):
    """
    Las seis condiciones de regla_de_eleccion.la_finalista_es_ganadora_si_cumple_todas, con sus umbrales leídos del texto.
    Orden esperado: ΔAP medio frente a B0 y B2, mejor_consistente frente a B0 y B2, ΔAUC medio frente a B0 y B2.
    """
    texts = protocol["regla_de_eleccion"]["la_finalista_es_ganadora_si_cumple_todas"]
    if len(texts) != 6:
        raise ValueError(f"Se esperaban 6 condiciones en regla_de_eleccion y hay {len(texts)}.")
    kinds = ("delta_ap_medio", "delta_ap_medio", "consistencia_ap", "consistencia_ap", "delta_auc_medio", "delta_auc_medio")
    references = ("B0", "B2", "B0", "B2", "B0", "B2")
    conditions = []
    for index, (text, kind, reference) in enumerate(zip(texts, kinds, references)):
        if f"frente a {reference}" not in text:
            raise ValueError(f"La condición {index + 1} no menciona 'frente a {reference}': {text}")
        if kind == "delta_ap_medio":
            match = re.search(r"ΔAP medio frente a B\d\s*>=\s*\+?(-?\d+(?:\.\d+)?)", text)
            threshold, unit = (float(match.group(1)), "ΔAP medio") if match else (None, None)
        elif kind == "consistencia_ap":
            match = re.search(r"ΔAP\s*>\s*0 en al menos (\d+) de (\d+) folds", text)
            threshold, unit = (int(match.group(1)), "folds con ΔAP > 0") if match else (None, None)
            if match and int(match.group(2)) != N_FOLDS:
                raise ValueError(f"La condición {index + 1} no es sobre {N_FOLDS} folds: {text}")
        else:
            match = re.search(r"ΔAUC medio >=\s*(-?\d+(?:\.\d+)?)", text)
            threshold, unit = (float(match.group(1)), "ΔROC-AUC medio") if match else (None, None)
        if threshold is None:
            raise ValueError(f"No se pudo leer el umbral de la condición {index + 1}: {text}")
        conditions.append({"id": f"C{index + 1}", "kind": kind, "reference": reference, "threshold": threshold,
                           "magnitude": unit, "text": text})
    return conditions


# ---------------------------------------------------------------------------------------------------------
# Comandos del autor (comandos_confirmacion.json) y finalista
# ---------------------------------------------------------------------------------------------------------
def load_confirmation_commands(root, protocol, candidates):
    """
    Lee comandos_confirmacion.json y seleccion_finalista.json (etapa 1) y verifica los 30 comandos contra la plantilla del
    protocolo y contra una regeneración mecánica. Devuelve (finalista, corridas, comprobaciones).
    """
    payload = e2a.read_json(Path(root) / STAGE1_ANALYSIS_REL / COMMANDS_NAME)
    selection = e2a.read_json(Path(root) / STAGE1_ANALYSIS_REL / SELECTION_NAME)
    finalist = payload["finalista"]
    commands = payload["comandos"]
    checks = []

    def add(name, ok, detail=""):
        checks.append({"corrida": "comandos_confirmacion", "comprobacion": name, "ok": bool(ok), "detalle": detail})

    add("el finalista de comandos_confirmacion.json = el de seleccion_finalista.json de la etapa 1",
        finalist == selection.get("finalista"), f"{finalist} / {selection.get('finalista')}")
    add("el finalista es una candidata del protocolo", finalist in candidates, finalist)
    if finalist not in candidates:
        return finalist, [], checks
    add("los flags del finalista = los de la candidata en el protocolo",
        payload.get("flags_del_finalista") == candidates[finalist]["flags"], str(payload.get("flags_del_finalista")))
    expected_ids = {finalist, "b0", "b2"}
    add("las configuraciones de los comandos son el finalista, B0 y B2", {c["config_id"] for c in commands} == expected_ids,
        str(sorted({c["config_id"] for c in commands})))
    if {c["config_id"] for c in commands} != expected_ids:
        return finalist, [], checks
    checks += a1.verify_confirmation_commands(commands, protocol, candidates, finalist)
    regenerated = a1.generate_confirmation_commands(protocol, candidates, finalist)
    add("los 30 comandos = los regenerados por sustitución mecánica de la plantilla (mismo orden y texto)",
        [(c["id"], c["comando"], c["salida"]) for c in commands] == [(c["id"], c["comando"], c["salida"]) for c in regenerated], "")

    flags = dict(a1.confirmation_entries(protocol, candidates, finalist))
    runs = []
    for c in commands:
        runs.append({"id": c["id"], "config_id": c["config_id"], "fold": int(c["fold"]), "seed": int(c["seed"]), "command": c["comando"],
                     "args": e2a.parse_command(c["comando"]), "dir_rel": c["salida"], "flags": flags.get(c["config_id"], "")})
    return finalist, runs, checks


# ---------------------------------------------------------------------------------------------------------
# Verificación de cada corrida
# ---------------------------------------------------------------------------------------------------------
def _finite(values):
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)


def verify_confirmation_run(run, folds_rows, design):
    """Comprobaciones de una corrida completada de la confirmación. Filas (corrida, comprobacion, ok, detalle)."""
    cfg, args, fold, seed, config_id = run["config"], run["args"], run["fold"], run["seed"], run["config_id"]
    rows = []

    def add(name, ok, detail=""):
        rows.append({"corrida": run["id"], "comprobacion": name, "ok": bool(ok), "detalle": detail})

    expected_dir = f"{CONFIRMATION_REL.as_posix()}/{config_id}/seed_{seed}/fold_{fold}"
    add("comando: semilla, fold y carpeta de salida coinciden con el id y con la carpeta leída",
        int(args["--seed"]) == seed and int(args["--fold"]) == fold and args["--output-dir"] == run["dir_rel"] == expected_dir,
        f"seed={args.get('--seed')}, fold={args.get('--fold')}, salida={args.get('--output-dir')}")
    add("semilla en {43, 44} y fold en 0..4 (según el protocolo)", seed in design["seeds"] and fold in design["folds"], f"seed={seed}, fold={fold}")
    add("comando = flags de la configuración", (f"--checkpoint-metric val_roc_auc {run['flags']} --split-csv" if run["flags"]
                                                else "--checkpoint-metric val_roc_auc --split-csv") in run["command"], run["flags"])
    add("config: esquema E2 y fold", cfg.get("evaluation_scheme") == "E2" and cfg.get("fold") == fold,
        f"evaluation_scheme={cfg.get('evaluation_scheme')}, fold={cfg.get('fold')}")
    add("config: arquitectura, condición, experimento, semilla",
        cfg["architecture"] == args["--arch"] and cfg["condition"] == args["--condition"]
        and cfg["experiment"] == args["--experiment"] and cfg["seed"] == seed,
        f"{cfg['architecture']}/{cfg['condition']}/{cfg['experiment']}/seed {cfg['seed']}")
    add("config: partición", cfg["csv_path"] == args["--split-csv"] == CV_FOLDS_REL, cfg["csv_path"])

    alpha = float(args["--class-weight-alpha"]) if "--class-weight-alpha" in args else 1.0
    add("config: exponente de los pesos de clase registrado como en el comando (B2 = 0.5; el resto no lo registra, α = 1.0)",
        ("class_weight_alpha" in cfg) == ("--class-weight-alpha" in args) and cfg.get("class_weight_alpha", 1.0) == alpha,
        f"alpha={cfg.get('class_weight_alpha', 'no registrado')}")

    lr = float(args["--learning-rate"]) if "--learning-rate" in args else e2a.EXPECTED_TRAINING["learning_rate"]
    expected_training = dict(e2a.EXPECTED_TRAINING, learning_rate=lr)
    differing = {k: cfg[k] for k in expected_training if cfg[k] != expected_training[k]}
    add("config: hiperparámetros fijos del protocolo", not differing, str(differing) if differing else "")
    add("config: criterio de checkpoint max val_roc_auc (ninguna configuración lo cambia)",
        cfg["checkpoint_criterion"].startswith("max val_roc_auc"), cfg["checkpoint_criterion"])
    add("config: early stopping activo (patience 5, min_delta 0.001)",
        cfg["patience"] == 5 and cfg["min_delta"] == 0.001 and "disabled" not in cfg.get("early_stopping_criterion", ""), "")
    add("config: 800 train y 200 validation", cfg["train_samples"] == 800 and cfg["validation_samples"] == 200,
        f"{cfg['train_samples']}/{cfg['validation_samples']}")
    for name, ok, detail in a1._flag_checks(cfg, run["history"], args):
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
    expected_weights = e2a.expected_class_weights([int(r["binary_label"]) for r in train], alpha)
    add(f"pesos de clase = recalculados con las etiquetas de train del fold (α = {alpha})",
        all(abs(a - b) <= 1e-5 * max(1.0, abs(b)) for a, b in zip(cfg["class_weights"], expected_weights)), str(cfg["class_weights"]))

    predictions = run["predictions"]
    y_true = [int(r["y_true"]) for r in predictions]
    y_prob = [float(r["probability_malignant"]) for r in predictions]
    add("predicciones: mismas imágenes y orden que validation del manifiesto, época del checkpoint",
        [r["image_id"] for r in predictions] == [r["image_id"] for r in validation]
        and y_true == [int(r["binary_label"]) for r in validation]
        and {int(r["epoch"]) for r in predictions} == {cfg["epoch_best"]}, f"{len(predictions)} filas")

    history = run["history"]
    numeric_history = [v for key in ("train_loss", "val_loss", "val_unweighted_loss", "val_roc_auc", "val_average_precision", "lr")
                       for v in history.get(key, [])]
    add("sin NaN ni valores no finitos: métricas, historial y probabilidades",
        _finite([run["metrics"]["roc_auc"], run["metrics"]["average_precision"]]) and _finite(numeric_history)
        and _finite(y_prob) and all(0.0 <= p <= 1.0 for p in y_prob), "")
    add("época del checkpoint dentro del rango (1 <= época <= épocas ejecutadas <= épocas máximas)",
        1 <= cfg["epoch_best"] <= cfg["epochs_run"] <= cfg["epochs_max"],
        f"epoch_best={cfg['epoch_best']}, epochs_run={cfg['epochs_run']}, epochs_max={cfg['epochs_max']}")

    for label, (metric_key, _) in e2a.METRICS.items():
        recomputed = e2a.roc_auc(y_true, y_prob) if metric_key == "roc_auc" else e2a.average_precision(y_true, y_prob)
        add(f"{label}: validation_metrics.json = recalculado desde las predicciones",
            abs(recomputed - run["metrics"][metric_key]) <= e2a.TOLERANCE_RECOMPUTED,
            f"json={run['metrics'][metric_key]:.8f}, recalculado={recomputed:.8f}")
    n_epochs = len(history["val_loss"])
    add("historial: longitud = épocas ejecutadas y AP por época presente",
        n_epochs == cfg["epochs_run"] and all(len(history[k]) == n_epochs for k in ("val_roc_auc", "val_average_precision"))
        and None not in history["val_average_precision"] + history["val_roc_auc"], f"{n_epochs} épocas")
    epoch_auc = cs.select_epoch(history["val_roc_auc"], max)
    add("checkpoint: época = primer máximo de val_roc_auc del historial (config, meta e historial coinciden)",
        epoch_auc == cfg["epoch_best"] == run["meta"]["epoch"], f"historial={epoch_auc}, run_config={cfg['epoch_best']}, meta={run['meta']['epoch']}")
    index = cfg["epoch_best"] - 1
    for label, (metric_key, history_key) in e2a.METRICS.items():
        in_history = 0 <= index < len(history[history_key])  # una época fuera de rango se informa como falla, no como error
        add(f"{label}: historial en la época del checkpoint = validation_metrics.json",
            in_history and abs(history[history_key][index] - run["metrics"][metric_key]) <= e2a.TOLERANCE_HISTORY,
            f"historial={history[history_key][index]:.6f}, json={run['metrics'][metric_key]:.6f}" if in_history
            else f"época {cfg['epoch_best']} fuera del historial ({len(history[history_key])} épocas)")
    return rows


# ---------------------------------------------------------------------------------------------------------
# Métricas por corrida y por fold
# ---------------------------------------------------------------------------------------------------------
def build_run_stats(runs):
    """config_id -> semilla -> fold -> {ap, auc, epoca, epocas}, solo con corridas completadas."""
    stats = {}
    for run in runs:
        if run["status"] != a1.STATUS_COMPLETED:
            continue
        stats.setdefault(run["config_id"], {}).setdefault(run["seed"], {})[run["fold"]] = {
            "ap": run["metrics"]["average_precision"], "auc": run["metrics"]["roc_auc"],
            "epoca": run["config"]["epoch_best"], "epocas": run["config"]["epochs_run"]}
    return stats


def seed_average(seed_stats, seeds, key):
    """Valor por fold (lista de 5): promedio simple de las semillas indicadas en ese fold."""
    return [statistics.mean(seed_stats[s][k][key] for s in seeds) for k in range(N_FOLDS)]


def per_fold_values(stats, configs, seeds):
    """config_id -> {'ap': [5], 'auc': [5]} con el promedio de las semillas en cada fold."""
    return {c: {key: seed_average(stats[c], seeds, key) for _, key in METRIC_KEYS} for c in configs}


def describe_deltas(deltas, minimum):
    """media, desviación estándar muestral (n-1; solo descriptiva), mínimo, máximo, signos y consistencia (4 de 5)."""
    summary = e2a.summarize(deltas)
    summary["desv_est_muestral"] = statistics.stdev(deltas)
    favored = e2a.consistency(summary, minimum)["favorece"]
    summary["veredicto"] = {"primera": "mejor_consistente", "segunda": "peor_consistente"}.get(favored, "no_consistente")
    return summary


def delta_table(values, pairs):
    """Delta por fold (primera - segunda) para AP y ROC-AUC. `values`: config -> {'ap': [5], 'auc': [5]}."""
    rows = []
    for name, first, second in pairs:
        for label, key in METRIC_KEYS:
            for k in range(N_FOLDS):
                a, b = values[first][key][k], values[second][key][k]
                rows.append({"comparacion": name, "primera": first, "segunda": second, "metrica": label, "fold": k,
                             "valor_primera": a, "valor_segunda": b, "delta": a - b, "signo": e2a.sign_of(a - b)})
    return rows


def summary_table(rows, pairs, minimum):
    out = []
    for name, first, second in pairs:
        for label, _ in METRIC_KEYS:
            deltas = [r["delta"] for r in rows if r["comparacion"] == name and r["metrica"] == label]
            s = describe_deltas(deltas, minimum)
            out.append({"comparacion": name, "primera": first, "segunda": second, "metrica": label, "media": s["media"],
                        "desv_est_muestral": s["desv_est_muestral"], "minimo": s["minimo"], "maximo": s["maximo"],
                        "folds_positivos": s["folds_positivos"], "folds_negativos": s["folds_negativos"], "folds_cero": s["folds_cero"],
                        # el veredicto de consistencia se basa solo en AP
                        "veredicto": s["veredicto"] if label == "AP" else ""})
    return out


# ---------------------------------------------------------------------------------------------------------
# regla_de_eleccion: aplicación literal
# ---------------------------------------------------------------------------------------------------------
def evaluate_rule(values, finalist, rule):
    """
    `values`: config -> {'ap': [5], 'auc': [5]} (promedio por fold de las semillas 43 y 44). Devuelve las seis condiciones
    con su valor, umbral, cumplimiento y margen, y el veredicto mecánico. Comparaciones sin redondear.
    """
    key_of = {"delta_ap_medio": "ap", "consistencia_ap": "ap", "delta_auc_medio": "auc"}
    results = []
    for condition in rule:
        reference = condition["reference"].lower()
        key = key_of[condition["kind"]]
        deltas = [values[finalist][key][k] - values[reference][key][k] for k in range(N_FOLDS)]
        if condition["kind"] == "consistencia_ap":
            value = sum(d > 0 for d in deltas)  # un Δ exactamente 0 no cuenta
            ok = value >= condition["threshold"]
        else:
            value = statistics.mean(deltas)
            ok = value >= condition["threshold"]
        results.append({"id": condition["id"], "condicion": condition["text"], "magnitud": condition["magnitude"],
                        "referencia": condition["reference"], "valor": value, "umbral": condition["threshold"],
                        "cumple": bool(ok), "margen": value - condition["threshold"], "delta_por_fold": deltas})
    unmet = [r["id"] for r in results if not r["cumple"]]
    return {"condiciones": results, "incumplidas": unmet, "veredicto": "sin_ganadora" if unmet else "ganadora"}


def contrast_b2_vs_b0(stats_confirmation, stats_seed42, decision_seeds, minimum):
    """
    B2 frente a B0 con tres semillas (42 de las referencias existentes, más 43 y 44). El protocolo no define cómo combinar
    las semillas: se calculan tres lecturas y no se elige ninguna.
        A: promedio por fold de las tres semillas;  B: cada semilla por separado;  C: promedio de 43 y 44 (base de la decisión).
    """
    def values_for(seeds):
        out = {}
        for config in ("b2", "b0"):
            out[config] = {}
            for _, key in METRIC_KEYS:
                per_seed = {s: [(stats_seed42[config][k][key] if s == 42 else stats_confirmation[config][s][k][key]) for k in range(N_FOLDS)]
                            for s in seeds}
                out[config][key] = [statistics.mean(per_seed[s][k] for s in seeds) for k in range(N_FOLDS)]
        return out

    readings = [(READING_A, "42,43,44", [42, *decision_seeds]), *[(READING_B, str(s), [s]) for s in (42, *decision_seeds)],
                (READING_C, ",".join(map(str, decision_seeds)), list(decision_seeds))]
    delta_rows, summary_rows = [], []
    for reading, label, seeds in readings:
        values = values_for(seeds)
        for metric, key in METRIC_KEYS:
            deltas = [values["b2"][key][k] - values["b0"][key][k] for k in range(N_FOLDS)]
            for k, d in enumerate(deltas):
                delta_rows.append({"lectura": reading, "semillas": label, "metrica": metric, "fold": k, "valor_b2": values["b2"][key][k],
                                   "valor_b0": values["b0"][key][k], "delta": d, "signo": e2a.sign_of(d)})
            s = describe_deltas(deltas, minimum)
            summary_rows.append({"lectura": reading, "semillas": label, "metrica": metric, "media": s["media"],
                                 "desv_est_muestral": s["desv_est_muestral"], "minimo": s["minimo"], "maximo": s["maximo"],
                                 "folds_positivos": s["folds_positivos"], "folds_negativos": s["folds_negativos"],
                                 "folds_cero": s["folds_cero"],
                                 # primera = B2: mejor_consistente = B2 supera a B0; peor_consistente = B0 supera a B2
                                 "veredicto": s["veredicto"] if metric == "AP" else ""})
    return delta_rows, summary_rows


# ---------------------------------------------------------------------------------------------------------
# Registro
# ---------------------------------------------------------------------------------------------------------
def registry_rows(protocol, runs):
    """Una fila por corrida de la confirmación (columnas de protocolo.registro), incluidas las fallidas."""
    columns = protocol["registro"]["columnas"]
    rows = []
    for run in runs:
        completed = run["status"] == a1.STATUS_COMPLETED
        rows.append({
            "etapa": "confirmacion", "config_id": run["config_id"], "fold": run["fold"], "seed": run["seed"], "flags": run["flags"],
            "ap": run["metrics"]["average_precision"] if completed else "", "roc_auc": run["metrics"]["roc_auc"] if completed else "",
            "epoca_elegida": run["config"]["epoch_best"] if completed else "",
            "epocas_ejecutadas": run["config"]["epochs_run"] if completed else "",
            "estado": run["status"], "ruta": run["dir_rel"],
            "nota": "faltan: " + ", ".join(run["missing"]) if run["status"] == a1.STATUS_FAILED else "",
        })
    return [{column: row[column] for column in columns} for row in rows]


# ---------------------------------------------------------------------------------------------------------
# Puntos que el protocolo no define: lecturas alternativas (no se elige ninguna)
# ---------------------------------------------------------------------------------------------------------
def open_points(rule, design):
    return [
        {"punto": "B2 frente a B0 con tres semillas: cómo combinar las semillas 42, 43 y 44",
         "lo_que_dice_el_protocolo": "regla_de_eleccion.sin_ganadora pide 'los resultados de B2 frente a B0 con tres semillas'; reglas_de_lectura solo define "
                                     "la métrica por fold como el promedio de las semillas 43 y 44 de la confirmación",
         "lecturas": {READING_A: "Δ por fold sobre el promedio de las tres semillas; consistencia 4/5 sobre esos 5 Δ",
                      READING_B: "Δ por fold dentro de cada semilla; una consistencia 4/5 por semilla",
                      READING_C: "Δ por fold sobre el promedio de 43 y 44 (la misma base de la regla de elección); no incluye la semilla 42"},
         "donde_ver_el_resultado": "contraste_b2_vs_b0_resumen.csv (una fila por lectura, semilla y métrica)"},
        {"punto": "Corrida fallida o ausente en la confirmación",
         "lo_que_dice_el_protocolo": "metodo_de_busqueda.fallos: se repite una vez; si vuelve a fallar, la configuración se registra como descartada y cuenta como "
                                     "probada. No dice cómo promediar semillas con una corrida faltante ni qué ocurre si la descartada es B0, B2 o el finalista",
         "lecturas": {"detener": "no calcular veredicto y esperar la decisión del autor (lo que hace este script)",
                      "descartar": "el finalista descartado = sin_ganadora; B0 o B2 descartados = la regla no es evaluable"},
         "estado_en_este_análisis": "no ocurre: se exige que las 30 corridas estén completas y verificadas"},
        {"punto": "Margen exacto sobre un umbral",
         "lo_que_dice_el_protocolo": "los umbrales (+0.04, 4 de 5 folds, -0.02) no indican redondeo",
         "lecturas": {"sin_redondear": "se comparan los valores en doble precisión (lo que hace este script; el margen está en veredicto_confirmacion.json)",
                      "redondeado_a_4_decimales": "como en las tablas del protocolo; solo difiere si el margen es menor que 0.00005"}},
    ]


# ---------------------------------------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------------------------------------
def run_analysis(root, output_dir=None, verbose=True):
    """
    Verifica las 30 corridas y escribe las tablas y el veredicto. Devuelve (código de salida, resultados).
    No escribe nada si falta alguna corrida, si alguna comprobación falla o si algún archivo de salida ya existe.
    """
    root = Path(root)
    protocol, protocol_path = a1.load_agent_protocol(root)
    e2_protocol, _ = e2a.load_protocol(root)
    candidates = a1.candidate_table(protocol)
    design = read_confirmation_design(protocol)
    rule = read_decision_rule(protocol)
    folds_rows = read_csv_rows(root / CV_FOLDS_REL)
    out_dir = Path(output_dir) if output_dir else root / CONFIRMATION_REL / ANALYSIS_NAME

    finalist, command_runs, checks = load_confirmation_commands(root, protocol, candidates)
    command_failures = [c for c in checks if not c["ok"]]
    if command_failures:
        for c in command_failures:
            print(f"  FALLA {c['corrida']}: {c['comprobacion']} -> {c['detalle']}")
        return 1, None
    runs = [a1.load_agent_run(root, run) for run in command_runs]
    not_run = [r["id"] for r in runs if r["status"] == a1.STATUS_NOT_RUN]
    failed_runs = [r for r in runs if r["status"] == a1.STATUS_FAILED]
    if verbose:
        print(f"Finalista: {finalist} | corridas de la confirmación: {len(runs)} | completadas: "
              f"{sum(r['status'] == a1.STATUS_COMPLETED for r in runs)} | fallidas: {len(failed_runs)} | sin carpeta: {len(not_run)}")
    if not_run:
        print(f"Faltan corridas por ejecutar (sin carpeta de salida; ¿falta el git pull?): {not_run[:5]}{' ...' if len(not_run) > 5 else ''}")
        return 1, None
    if failed_runs:
        print("Hay corridas incompletas: el protocolo no define cómo evaluar la regla sin ellas; decide el autor "
              f"(ver veredicto_confirmacion.json, puntos_pendientes_de_decision): {[r['id'] for r in failed_runs]}")
        return 1, None

    checks.insert(0, {"corrida": "protocolo", "comprobacion": "incluye la enmienda E1",
                      "ok": any(e["id"] == "E1" for e in protocol["enmiendas"]), "detalle": str([e["id"] for e in protocol["enmiendas"]])})
    identical = a1.file_matches_commit(root, AGENT_REL / a1.PROTOCOL_NAME, a1.PROTOCOL_COMMIT, protocol_path)
    checks.insert(1, {"corrida": "protocolo", "comprobacion": f"idéntico al del commit {a1.PROTOCOL_COMMIT}", "ok": identical is not False,
                      "detalle": "no verificable (git no disponible)" if identical is None else str(identical)})
    checks.insert(2, {"corrida": "protocolo", "comprobacion": "30 corridas = 3 configuraciones x semillas 43 y 44 x folds 0..4, sin repetir",
                      "ok": len(runs) == design["total"] == 30 and len({(r["config_id"], r["seed"], r["fold"]) for r in runs}) == 30
                      and {r["config_id"] for r in runs} == {finalist, "b0", "b2"}
                      and {r["seed"] for r in runs} == set(design["seeds"]) == {43, 44} and {r["fold"] for r in runs} == set(design["folds"]) == set(range(5)),
                      "detalle": str(len(runs))})
    for run in runs:
        checks += verify_confirmation_run(run, folds_rows, design)

    reference_runs = a1.load_reference_runs(root, protocol, e2_protocol)
    for ref in reference_runs:
        checks += e2a.verify_run(ref, e2_protocol, folds_rows)
    stats_seed42 = a1.build_stats([], reference_runs)
    checks += a1.verify_reference_values(protocol, stats_seed42)

    failed = [c for c in checks if not c["ok"]]
    if verbose:
        print(f"Comprobaciones: {len(checks)} | fallidas: {len(failed)}")
    if failed:
        for c in failed:
            print(f"  FALLA {c['corrida']}: {c['comprobacion']} -> {c['detalle']}")
        return 1, None

    existing = [out_dir / name for name in OUTPUT_FILES.values() if (out_dir / name).exists()]
    if existing:
        raise FileExistsError("Ya existen archivos de salida; no se sobrescribe ninguno:\n" + "\n".join(str(p) for p in existing))

    stats = build_run_stats(runs)
    configs = [finalist, "b0", "b2"]
    values = per_fold_values(stats, configs, design["seeds"])
    minimum = next(c["threshold"] for c in rule if c["kind"] == "consistencia_ap")
    pairs = [(f"{finalist}__vs_b0", finalist, "b0"), (f"{finalist}__vs_b2", finalist, "b2")]
    deltas = delta_table(values, pairs)
    summaries = summary_table(deltas, pairs, minimum)
    decision = evaluate_rule(values, finalist, rule)
    contrast_deltas, contrast_summaries = contrast_b2_vs_b0(stats, {c: stats_seed42[c] for c in ("b0", "b2")}, design["seeds"], minimum)

    by_fold_rows = []
    for config in configs:
        for k in range(N_FOLDS):
            row = {"config_id": config, "fold": k}
            for seed in design["seeds"]:
                row[f"ap_semilla_{seed}"] = stats[config][seed][k]["ap"]
            row["ap_promedio_semillas"] = values[config]["ap"][k]
            for seed in design["seeds"]:
                row[f"roc_auc_semilla_{seed}"] = stats[config][seed][k]["auc"]
            row["roc_auc_promedio_semillas"] = values[config]["auc"][k]
            by_fold_rows.append(row)
    run_rows = [{"config_id": c, "semilla": s, "fold": k, "ap": v["ap"], "roc_auc": v["auc"], "epoca_elegida": v["epoca"],
                 "epocas_ejecutadas": v["epocas"]}
                for c in configs for s in sorted(stats[c]) for k, v in sorted(stats[c][s].items())]

    previous = e2a.read_json(root / STAGE1_ANALYSIS_REL / SELECTION_NAME)
    verdict_payload = {
        "protocolo": {"archivo": f"{AGENT_REL.as_posix()}/{a1.PROTOCOL_NAME}", "commit_de_referencia": a1.PROTOCOL_COMMIT,
                      "idéntico_al_commit": identical, "enmiendas": [e["id"] for e in protocol["enmiendas"]]},
        "finalista": finalist, "flags_del_finalista": candidates[finalist]["flags"],
        "base_de_la_decision": protocol["regla_de_eleccion"]["se_evalua_con"],
        "regla": {"texto": protocol["regla_de_eleccion"]["la_finalista_es_ganadora_si_cumple_todas"],
                  "umbrales_leidos_del_texto": [{"id": c["id"], "magnitud": c["magnitude"], "referencia": c["reference"], "umbral": c["threshold"]} for c in rule]},
        "condiciones": decision["condiciones"], "condiciones_incumplidas": decision["incumplidas"],
        "veredicto_mecanico": decision["veredicto"],
        "consecuencia": protocol["regla_de_eleccion"]["si_cumple"] if decision["veredicto"] == "ganadora" else protocol["regla_de_eleccion"]["sin_ganadora"],
        "metricas_por_fold": {c: {"ap": values[c]["ap"], "roc_auc": values[c]["auc"]} for c in configs},
        "resumen_delta": summaries,
        "conteo": {"configuraciones_nuevas_probadas": len(candidates), "configuraciones_de_referencia": 2,
                   "corridas_etapa_1": previous["conteo"]["corridas_etapa_1"], "corridas_confirmacion": len(runs),
                   "corridas_fallidas_o_descartadas": len(failed_runs) + previous["conteo"]["fallidas"]},
        "no_calculado": {"intervalo_de_confianza_95": protocol["reglas_de_lectura"]["p_valor"],
                         "nota": "el pedido incluía un IC95 de los Δ como descripción; el protocolo lo excluye con 5 folds y no se calcula (discrepancia: gana el protocolo)"},
        "contraste_b2_vs_b0": {"nota": "tres lecturas posibles; el protocolo no define cuál usar y no se elige ninguna",
                               "resumen": contrast_summaries},
        "puntos_pendientes_de_decision": open_points(rule, design),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    cs.write_csv(out_dir / OUTPUT_FILES["registro"], registry_rows(protocol, runs))
    cs.write_csv(out_dir / OUTPUT_FILES["verification"], checks)
    cs.write_csv(out_dir / OUTPUT_FILES["runs"], run_rows)
    cs.write_csv(out_dir / OUTPUT_FILES["by_fold"], by_fold_rows)
    cs.write_csv(out_dir / OUTPUT_FILES["delta_by_fold"], deltas)
    cs.write_csv(out_dir / OUTPUT_FILES["delta_summary"], summaries)
    cs.write_csv(out_dir / OUTPUT_FILES["contrast_delta"], contrast_deltas)
    cs.write_csv(out_dir / OUTPUT_FILES["contrast_summary"], contrast_summaries)
    e2a.write_json(out_dir / OUTPUT_FILES["verdict"], verdict_payload)

    if verbose:
        for c in decision["condiciones"]:
            print(f"  {c['id']} {c['magnitud']} frente a {c['referencia']}: {c['valor']:.4f} (umbral {c['umbral']}) -> {'cumple' if c['cumple'] else 'NO cumple'}")
        print(f"Veredicto mecánico: {decision['veredicto']} | incumplidas: {decision['incumplidas']}")
        print("Escrito en", out_dir)
    return 0, {"checks": checks, "values": values, "deltas": deltas, "summaries": summaries, "decision": decision,
               "contrast_summaries": contrast_summaries, "verdict": verdict_payload}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Análisis de la confirmación del brazo del agente según protocolo_agente.json.")
    parser.add_argument("--output-dir", default=None, help=f"Carpeta de salida (por defecto {CONFIRMATION_REL.as_posix()}/{ANALYSIS_NAME}/).")
    args = parser.parse_args(argv)
    code, _ = run_analysis(Path.cwd(), args.output_dir)
    return code


if __name__ == "__main__":
    sys.exit(main())
