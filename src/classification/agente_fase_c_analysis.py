"""
Análisis de la fase C (exploratoria) del brazo del agente de optimización (R7): estudio de ruido y ensamble de semillas,
aplicando literalmente results/classification/r7_optimizacion/agente/fase_c_ruido_ensamble/protocolo_fase_c.json.
Es descriptivo: sin p-valores, intervalos de confianza, pruebas, reglas con etiqueta ni ganador.

Parte 1 (solo biblioteca estándar): reparto de la variación del AP (y del ROC-AUC, descriptivo) de las 75 corridas de la
fase B en fold, α, semilla dentro de α y residual, con sumas de cuadrados de un diseño balanceado; DE entre semillas por
fold, varianza agrupada s²_p y sd_b = sqrt(2·s²_p / 3), comparada con el piso de AP de la fase B.
Parte 2: ensamble de las semillas 42, 43 y 44 de B0 (hipótesis primaria) y B2 (secundaria) por fold, promediando
probability_malignant por image_id; AP y ROC-AUC con las funciones del pipeline (scikit-learn, como evaluate.py).

Lee solo archivos de validation ya versionados; no entrena, no carga checkpoints y no usa test. Si una verificación de
integridad falla, se detiene sin escribir ningún archivo. No sobrescribe salidas existentes.

Uso (desde la raíz del repositorio):
    python -m src.classification.agente_fase_c_analysis
"""
from __future__ import annotations

import argparse
import itertools
import math
import re
import statistics
import sys
from pathlib import Path

from src.classification import agente_etapa1_analysis as a1
from src.classification import agente_fase_b_analysis as fb
from src.classification import checkpoint_sensitivity as cs
from src.classification import robustez_e2_analysis as e2a
from src.classification.cv_folds_e2 import OUTPUT_REL as CV_FOLDS_REL, read_csv_rows

AGENT_REL = a1.AGENT_REL
PHASE_C_REL = AGENT_REL / "fase_c_ruido_ensamble"
PROTOCOL_NAME = "protocolo_fase_c.json"
PROTOCOL_COMMIT = "cc768aa"  # commit del protocolo de la fase C
ANALYSIS_NAME = "analisis"
PHASE_B_ANALYSIS_REL = fb.PHASE_B_REL / fb.ANALYSIS_NAME

OUTPUT_FILES = {
    "components": "componentes_varianza.csv",
    "between_seeds": "de_entre_semillas_por_fold.csv",
    "noise_b": "ruido_lectura_b.json",
    "ensemble_by_fold": "ensamble_por_fold.csv",
    "ensemble_summary": "resumen_ensamble.csv",
    "verification": "verificacion.csv",
    "summary": "resumen_fase_c.json",
}

METRICS = (("AP", "ap", "average_precision"), ("ROC-AUC", "roc_auc", "roc_auc"))  # etiqueta, columna, clave en validation_metrics.json
CONFIG_ALPHA = {"B0": 1.0, "B2": 0.5}
PREDICTION_COLUMNS = ("epoch", "image_id", "y_true", "probability_malignant")
PREDICTIONS_NAME, METRICS_NAME, CONFIG_NAME = "val_predictions_best_auc.csv", "validation_metrics.json", "run_config.json"
TOLERANCE_METRICS = 1e-6  # métricas frente a validation_metrics.json
TOLERANCE_PHASE_B = 1e-9  # DE entre semillas frente a metricas_por_fold.csv de la fase B
SOURCES = ("fold", "alpha", "semilla_dentro_de_alpha", "residual")


class IntegrityError(ValueError):
    """Una verificación de integridad del protocolo no se cumple."""


# ---------------------------------------------------------------------------------------------------------
# Protocolo y rutas
# ---------------------------------------------------------------------------------------------------------
def load_protocol(root):
    path = Path(root) / PHASE_C_REL / PROTOCOL_NAME
    return e2a.read_json(path), path


def expand_braces(text):
    """Expande los grupos '{a..b}' y '{x,y}' de una ruta del protocolo (por ejemplo fold_{0..4} o seed_{43,44})."""
    parts = re.split(r"(\{[^}]*\})", text)
    options = []
    for part in parts:
        match = re.fullmatch(r"\{(-?\d+)\.\.(-?\d+)\}", part)
        if match:
            options.append([str(i) for i in range(int(match.group(1)), int(match.group(2)) + 1)])
        elif part.startswith("{") and part.endswith("}"):
            options.append([x.strip() for x in part[1:-1].split(",")])
        else:
            options.append([part])
    return ["".join(choice) for choice in itertools.product(*options)]


def planned_runs(protocol):
    """Las 30 corridas de entradas.parte_2.rutas: configuración, α, semilla, fold y carpeta."""
    runs = []
    for key, text in protocol["entradas"]["parte_2"]["rutas"].items():
        config = key.split("_")[0]
        if config not in CONFIG_ALPHA:
            raise ValueError(f"Clave de ruta no reconocida en el protocolo: {key}")
        seed_in_key = re.search(r"semilla_(\d+)$", key)
        for path in expand_braces(text):
            seed_match, fold_match = re.search(r"seed_(\d+)", path), re.search(r"fold_(\d+)", path)
            if fold_match is None or (seed_match is None and seed_in_key is None):
                raise ValueError(f"No se pudo leer semilla y fold de la ruta {path}")
            seed = int(seed_match.group(1)) if seed_match else int(seed_in_key.group(1))
            runs.append({"config": config, "alpha": CONFIG_ALPHA[config], "seed": seed, "fold": int(fold_match.group(1)),
                         "dir_rel": path.rstrip("/"), "id": f"{config}_seed{seed}_fold{fold_match.group(1)}"})
    return sorted(runs, key=lambda r: (r["config"], r["seed"], r["fold"]))


# ---------------------------------------------------------------------------------------------------------
# Parte 1: descomposición de la variación (funciones puras)
# ---------------------------------------------------------------------------------------------------------
def _levels(values):
    alphas = sorted({a for a, _, _ in values})
    seeds = sorted({s for _, s, _ in values})
    folds = sorted({k for _, _, k in values})
    expected = {(a, s, k) for a in alphas for s in seeds for k in folds}
    if set(values) != expected:
        raise IntegrityError("El diseño no es balanceado: falta alguna combinación (α, semilla, fold) o sobra alguna.")
    return alphas, seeds, folds


def variance_decomposition(values):
    """
    values: (α, semilla, fold) -> métrica. Modelo y = μ + F_k + A_a + S_s(a) + e, una observación por celda.
    Devuelve una fila por fuente (fold, alpha, semilla_dentro_de_alpha, residual, total) con SS, gl, MS, componente por
    método de momentos (sin truncar), componente truncada en 0, marca de negativa y porcentajes.
    """
    alphas, seeds, folds = _levels(values)
    a, s, k = len(alphas), len(seeds), len(folds)
    grand = statistics.mean(values.values())
    fold_mean = {f: statistics.mean(values[(x, y, f)] for x in alphas for y in seeds) for f in folds}
    alpha_mean = {x: statistics.mean(values[(x, y, f)] for y in seeds for f in folds) for x in alphas}
    cell_mean = {(x, y): statistics.mean(values[(x, y, f)] for f in folds) for x in alphas for y in seeds}

    ss = {
        "fold": a * s * sum((fold_mean[f] - grand) ** 2 for f in folds),
        "alpha": s * k * sum((alpha_mean[x] - grand) ** 2 for x in alphas),
        "semilla_dentro_de_alpha": k * sum((cell_mean[(x, y)] - alpha_mean[x]) ** 2 for x in alphas for y in seeds),
    }
    ss_total = sum((v - grand) ** 2 for v in values.values())
    ss["residual"] = ss_total - sum(ss.values())
    df = {"fold": k - 1, "alpha": a - 1, "semilla_dentro_de_alpha": a * (s - 1)}
    df_total = a * s * k - 1
    df["residual"] = df_total - sum(df.values())
    ms = {name: ss[name] / df[name] for name in SOURCES}

    component = {
        "residual": ms["residual"],
        "semilla_dentro_de_alpha": (ms["semilla_dentro_de_alpha"] - ms["residual"]) / k,
        "fold": (ms["fold"] - ms["residual"]) / (a * s),
        "alpha": (ms["alpha"] - ms["semilla_dentro_de_alpha"]) / (s * k),  # α fijo: magnitud descriptiva
    }
    truncated = {name: max(0.0, value) for name, value in component.items()}
    total_component = sum(truncated.values())
    rows = []
    for name in SOURCES:
        rows.append({"fuente": name, "SS": ss[name], "gl": df[name], "MS": ms[name], "componente": component[name],
                     "componente_truncada": truncated[name], "negativa": component[name] < 0,
                     "porcentaje_SS": 100.0 * ss[name] / ss_total if ss_total > 0 else None,
                     "porcentaje_componente": 100.0 * truncated[name] / total_component if total_component > 0 else None})
    rows.append({"fuente": "total", "SS": ss_total, "gl": df_total, "MS": ss_total / df_total, "componente": "",
                 "componente_truncada": "", "negativa": "", "porcentaje_SS": 100.0 if ss_total > 0 else None, "porcentaje_componente": ""})
    return rows


def between_seed_sd(values):
    """DE muestral entre semillas por (α, fold), su media por α, s²_p (media de las varianzas) y sd_b = sqrt(2·s²_p / n_semillas)."""
    alphas, seeds, folds = _levels(values)
    sd = {(x, f): statistics.stdev([values[(x, y, f)] for y in seeds]) for x in alphas for f in folds}
    mean_by_alpha = {x: statistics.mean(sd[(x, f)] for f in folds) for x in alphas}
    s2_p = statistics.mean(v ** 2 for v in sd.values())
    return {"sd": sd, "media_por_alpha": mean_by_alpha, "s2_p": s2_p, "sd_b": math.sqrt(2.0 * s2_p / len(seeds)), "n_semillas": len(seeds)}


# ---------------------------------------------------------------------------------------------------------
# Parte 2: verificación, alineación y ensamble (funciones puras)
# ---------------------------------------------------------------------------------------------------------
def check_prediction_rows(rows, columns, epoch_best):
    """Significado de la probabilidad: columnas, valores finitos en [0, 1], y_true ∈ {0, 1}, epoch único = epoch_best."""
    problems = []
    missing = [c for c in PREDICTION_COLUMNS if c not in columns]
    if missing:
        return [f"faltan columnas: {missing}"]
    probabilities = []
    for row in rows:
        try:
            probabilities.append(float(row["probability_malignant"]))
        except (TypeError, ValueError):
            probabilities.append(float("nan"))
    bad = [p for p in probabilities if not (math.isfinite(p) and 0.0 <= p <= 1.0)]
    if bad:
        problems.append(f"{len(bad)} probabilidades no finitas o fuera de [0, 1] (por ejemplo {bad[0]})")
    labels = {row["y_true"] for row in rows}
    if not labels <= {"0", "1"}:
        problems.append(f"y_true fuera de {{0, 1}}: {sorted(labels - {'0', '1'})}")
    epochs = {row["epoch"] for row in rows}
    if len(epochs) != 1 or int(float(next(iter(epochs)))) != int(epoch_best):
        problems.append(f"epoch {sorted(epochs)} distinto de epoch_best={epoch_best}")
    return problems


def align_seed_predictions(predictions_by_seed):
    """
    predictions_by_seed: semilla -> filas (image_id, y_true, probability_malignant). Alinea por image_id con el orden de la
    primera semilla. Lanza IntegrityError si hay image_id repetidos, si los conjuntos de image_id difieren o si, después de
    alinear, el y_true de alguna imagen difiere entre semillas. Un orden distinto entre semillas no es un error.
    """
    seeds = sorted(predictions_by_seed)
    by_seed = {}
    for seed in seeds:
        ids = [row["image_id"] for row in predictions_by_seed[seed]]
        if len(ids) != len(set(ids)):
            repeated = sorted({i for i in ids if ids.count(i) > 1})
            raise IntegrityError(f"semilla {seed}: image_id repetidos {repeated[:5]}")
        by_seed[seed] = {row["image_id"]: row for row in predictions_by_seed[seed]}
    reference_ids = [row["image_id"] for row in predictions_by_seed[seeds[0]]]
    for seed in seeds[1:]:
        if set(by_seed[seed]) != set(reference_ids):
            only_ref = sorted(set(reference_ids) - set(by_seed[seed]))
            only_seed = sorted(set(by_seed[seed]) - set(reference_ids))
            raise IntegrityError(f"semilla {seed}: image_id distintos de la semilla {seeds[0]} (solo en {seeds[0]}: {only_ref[:5]}; solo en {seed}: {only_seed[:5]})")
    y_true = []
    for image_id in reference_ids:
        labels = {by_seed[s][image_id]["y_true"] for s in seeds}
        if len(labels) != 1:
            raise IntegrityError(f"image_id {image_id}: y_true distinto entre semillas después de alinear ({sorted(labels)})")
        y_true.append(int(next(iter(labels))))
    order_differs = any([row["image_id"] for row in predictions_by_seed[s]] != reference_ids for s in seeds[1:])
    probabilities = {s: [float(by_seed[s][i]["probability_malignant"]) for i in reference_ids] for s in seeds}
    return {"image_ids": reference_ids, "y_true": y_true, "probabilities": probabilities, "orden_distinto": order_differs}


def ensemble_fold(y_true, probabilities, metric_fn):
    """
    Ensamble de un fold: p_ens = media aritmética de las probabilidades de las semillas (alineadas). metric_fn(y, p) ->
    {'ap': ..., 'roc_auc': ...}. Δ = métrica(ensamble) - media de las métricas de las semillas.
    """
    seeds = sorted(probabilities)
    ensemble = [statistics.mean(probabilities[s][i] for s in seeds) for i in range(len(y_true))]
    individual = {s: metric_fn(y_true, probabilities[s]) for s in seeds}
    combined = metric_fn(y_true, ensemble)
    out = {"p_ens": ensemble, "individual": individual, "ensamble": combined}
    for _, key, _ in METRICS:
        mean_individual = statistics.mean(individual[s][key] for s in seeds)
        out[f"{key}_media_individual"] = mean_individual
        out[f"delta_{key}"] = combined[key] - mean_individual
    return out


def pipeline_metrics(y_true, y_prob):
    """AP y ROC-AUC con las funciones del pipeline (src/classification/evaluate.py: scikit-learn)."""
    from sklearn.metrics import average_precision_score, roc_auc_score  # importación diferida: solo la parte 2 la necesita

    return {"ap": float(average_precision_score(y_true, y_prob)), "roc_auc": float(roc_auc_score(y_true, y_prob))}


def summarize_deltas(deltas):
    s = e2a.summarize(deltas)
    return {"deltas": list(deltas), "signos": [e2a.sign_of(d) for d in deltas], "media": s["media"], "minimo": s["minimo"],
            "maximo": s["maximo"], "positivos": s["folds_positivos"], "negativos": s["folds_negativos"], "ceros": s["folds_cero"]}


# ---------------------------------------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------------------------------------
def _check(rows, run_id, name, ok, detail=""):
    rows.append({"corrida": run_id, "comprobacion": name, "ok": bool(ok), "detalle": detail})


def _stop(checks, verbose=True):
    failed = [c for c in checks if not c["ok"]]
    print(f"Comprobaciones: {len(checks)} | fallidas: {len(failed)}")
    for c in failed:
        print(f"  FALLA {c['corrida']}: {c['comprobacion']} -> {c['detalle']}")
    print("Análisis detenido: no se escribió ningún archivo de salida.")
    return 1, None


def part1_checks_and_values(root, checks):
    """Verificaciones de la parte 1 y valores (α, semilla, fold) -> {'ap', 'roc_auc'}."""
    root = Path(root)
    rows = read_csv_rows(root / PHASE_B_ANALYSIS_REL / fb.OUTPUT_FILES["runs"])
    registry = read_csv_rows(root / PHASE_B_ANALYSIS_REL / fb.OUTPUT_FILES["registro"])
    key_of = lambda r: (float(r["alpha"]), int(r["seed"]), int(r["fold"]))
    keys = [key_of(r) for r in rows]
    _check(checks, "parte_1", "metricas_por_corrida.csv: 75 filas, cada (α, semilla, fold) una sola vez",
           len(rows) == 75 and len(set(keys)) == 75, f"{len(rows)} filas, {len(set(keys))} distintas")
    registry_by_key = {}
    for r in registry:
        registry_by_key.setdefault(key_of(r), []).append(r)
    values = {}
    for row in rows:
        key = key_of(row)
        values[key] = {"ap": float(row["ap"]), "roc_auc": float(row["roc_auc"])}
        run_id = f"alpha_{key[0]}_seed{key[1]}_fold{key[2]}"
        matches = registry_by_key.get(key, [])
        if len(matches) != 1:
            _check(checks, run_id, "fila única en registro_intentos_fase_b.csv", False, f"{len(matches)} filas")
            continue
        entry = matches[0]
        _check(checks, run_id, "registro_intentos_fase_b.csv: estado 'completada'", entry["estado"] == "completada", entry["estado"])
        metrics_path = root / entry["ruta"] / METRICS_NAME
        if not metrics_path.exists():
            _check(checks, run_id, "validation_metrics.json de la ruta del registro existe", False, entry["ruta"])
            continue
        saved = e2a.read_json(metrics_path)
        for label, column, json_key in METRICS:
            _check(checks, run_id, f"{label} de metricas_por_corrida.csv = validation_metrics.json de la ruta (1e-6)",
                   abs(values[key][column] - saved[json_key]) <= TOLERANCE_METRICS, f"csv={values[key][column]:.8f}, json={saved[json_key]:.8f}")
    return values


def run_analysis(root, output_dir=None, verbose=True):
    """
    Verifica, calcula y escribe las salidas del protocolo. Devuelve (código, resultados). Si alguna verificación falla,
    imprime las fallas y devuelve (1, None) sin escribir nada. Si algún archivo de salida existe, lanza FileExistsError.
    """
    root = Path(root)
    protocol, protocol_path = load_protocol(root)
    out_dir = Path(output_dir) if output_dir else root / PHASE_C_REL / ANALYSIS_NAME
    existing = [out_dir / name for name in OUTPUT_FILES.values() if (out_dir / name).exists()]
    if existing:
        raise FileExistsError("Ya existen archivos de salida; no se sobrescribe ninguno:\n" + "\n".join(str(p) for p in existing))

    checks = []
    identical = a1.file_matches_commit(root, PHASE_C_REL / PROTOCOL_NAME, PROTOCOL_COMMIT, protocol_path)
    _check(checks, "protocolo", f"idéntico al del commit {PROTOCOL_COMMIT}", identical is not False,
           "no verificable (git no disponible)" if identical is None else str(identical))

    # ---- Parte 1: verificación ----
    values = part1_checks_and_values(root, checks)
    if any(not c["ok"] for c in checks):
        return _stop(checks, verbose)
    components, noise = [], {}
    for label, column, _ in METRICS:
        metric_values = {key: v[column] for key, v in values.items()}
        try:
            for row in variance_decomposition(metric_values):
                components.append(dict({"metrica": label}, **row))
            noise[label] = between_seed_sd(metric_values)
        except IntegrityError as error:
            _check(checks, "parte_1", "diseño balanceado 5 α × 3 semillas × 5 folds", False, str(error))
            return _stop(checks, verbose)
    by_fold_b = read_csv_rows(root / PHASE_B_ANALYSIS_REL / fb.OUTPUT_FILES["by_fold"])
    for label, column, _ in METRICS:
        sd = noise[label]["sd"]
        reference = {(float(r["alpha"]), int(r["fold"])): float(r[f"{column}_de_entre_semillas"]) for r in by_fold_b}
        differing = [key for key in sd if key not in reference or abs(sd[key] - reference[key]) > TOLERANCE_PHASE_B]
        _check(checks, "parte_1", f"DE entre semillas de {label} = {column}_de_entre_semillas de la fase B (1e-9)",
               not differing and len(reference) == len(sd), str(differing[:3]))
    if any(not c["ok"] for c in checks):
        return _stop(checks, verbose)
    piso_ap = e2a.read_json(root / PHASE_B_ANALYSIS_REL / fb.OUTPUT_FILES["best_alpha"])["piso_de_ruido"]["piso_ap"]

    # ---- Parte 2: verificación estructural (sin scikit-learn) ----
    runs = planned_runs(protocol)
    _check(checks, "parte_2", "30 corridas declaradas: B0 y B2 × semillas 42, 43, 44 × folds 0..4",
           len(runs) == 30 and {(r["config"], r["seed"], r["fold"]) for r in runs}
           == {(c, s, k) for c in CONFIG_ALPHA for s in (42, 43, 44) for k in range(5)}, str(len(runs)))
    partition = read_csv_rows(root / CV_FOLDS_REL)
    fold_ids = {k: {r["image_id"] for r in partition if int(r["fold"]) == k} for k in range(5)}
    loaded = {}
    for run in runs:
        folder = root / run["dir_rel"]
        missing = [name for name in (PREDICTIONS_NAME, METRICS_NAME, CONFIG_NAME) if not (folder / name).exists()]
        _check(checks, run["id"], "carpeta con val_predictions_best_auc.csv, validation_metrics.json y run_config.json", not missing,
               run["dir_rel"] + (f" (faltan {missing})" if missing else ""))
        if missing:
            continue
        cfg = e2a.read_json(folder / CONFIG_NAME)
        with open(folder / PREDICTIONS_NAME, encoding="utf-8") as handle:
            header = handle.readline().strip().split(",")
        rows = read_csv_rows(folder / PREDICTIONS_NAME)
        problems = check_prediction_rows(rows, header, cfg.get("epoch_best"))
        _check(checks, run["id"], "probability_malignant finita en [0, 1], y_true ∈ {0, 1}, epoch único = epoch_best", not problems, "; ".join(problems))
        alpha_ok = ("class_weight_alpha" not in cfg) if run["alpha"] == 1.0 else cfg.get("class_weight_alpha") == run["alpha"]
        _check(checks, run["id"], "class_weight_alpha: ausente en B0, 0.5 en B2", alpha_ok, str(cfg.get("class_weight_alpha", "no registrado")))
        _check(checks, run["id"], "semilla y fold de run_config.json = los de la carpeta", cfg.get("seed") == run["seed"] and cfg.get("fold") == run["fold"],
               f"seed={cfg.get('seed')}, fold={cfg.get('fold')}")
        loaded[run["id"]] = dict(run, rows=rows, metrics=e2a.read_json(folder / METRICS_NAME))
    if any(not c["ok"] for c in checks):
        return _stop(checks, verbose)

    aligned = {}
    for config in CONFIG_ALPHA:
        for k in range(5):
            group = {r["seed"]: loaded[r["id"]]["rows"] for r in runs if r["config"] == config and r["fold"] == k}
            name = f"{config}_fold{k}"
            try:
                result = align_seed_predictions(group)
            except IntegrityError as error:
                _check(checks, name, "mismas image_id, sin repetidos y mismo y_true entre las 3 semillas (alineadas por image_id)", False, str(error))
                continue
            _check(checks, name, "mismas image_id, sin repetidos y mismo y_true entre las 3 semillas (alineadas por image_id)", True,
                   "orden distinto entre semillas: alineado por image_id" if result["orden_distinto"] else "mismo orden")
            _check(checks, name, "image_id = validation del fold en cv_folds_e2.csv", set(result["image_ids"]) == fold_ids[k],
                   f"{len(result['image_ids'])} imágenes, {len(fold_ids[k])} en la partición")
            aligned[(config, k)] = result
    for k in range(5):
        if ("B0", k) in aligned and ("B2", k) in aligned:
            _check(checks, f"fold{k}", "mismas image_id en B0 y B2", set(aligned[("B0", k)]["image_ids"]) == set(aligned[("B2", k)]["image_ids"]), "")
    if any(not c["ok"] for c in checks):
        return _stop(checks, verbose)

    # ---- Parte 2: métricas recalculadas con las funciones del pipeline ----
    recomputed = {}
    for run_id, run in loaded.items():
        y = [int(r["y_true"]) for r in run["rows"]]
        p = [float(r["probability_malignant"]) for r in run["rows"]]
        recomputed[run_id] = pipeline_metrics(y, p)
        for label, column, json_key in METRICS:
            _check(checks, run_id, f"{label} recalculado desde el CSV = validation_metrics.json (1e-6)",
                   abs(recomputed[run_id][column] - run["metrics"][json_key]) <= TOLERANCE_METRICS,
                   f"recalculado={recomputed[run_id][column]:.8f}, json={run['metrics'][json_key]:.8f}")
    if any(not c["ok"] for c in checks):
        return _stop(checks, verbose)

    # ---- Cálculos de la parte 2 ----
    ensemble_rows, ensemble_summary, deltas = [], [], {}
    for config in CONFIG_ALPHA:
        per_metric = {key: [] for _, key, _ in METRICS}
        for k in range(5):
            result = aligned[(config, k)]
            fold = ensemble_fold(result["y_true"], result["probabilities"], pipeline_metrics)
            row = {"configuracion": config, "alpha": CONFIG_ALPHA[config], "fold": k, "orden_distinto_entre_semillas": result["orden_distinto"]}
            for label, key, _ in METRICS:
                for seed in sorted(result["probabilities"]):
                    row[f"{key}_semilla_{seed}"] = fold["individual"][seed][key]
                row[f"{key}_media_individual"] = fold[f"{key}_media_individual"]
                row[f"{key}_ensamble"] = fold["ensamble"][key]
                row[f"delta_{key}"] = fold[f"delta_{key}"]
                row[f"signo_{key}"] = e2a.sign_of(fold[f"delta_{key}"])
                per_metric[key].append(fold[f"delta_{key}"])
            ensemble_rows.append(row)
        deltas[config] = {}
        for label, key, _ in METRICS:
            summary = summarize_deltas(per_metric[key])
            deltas[config][label] = summary
            ensemble_summary.append({"configuracion": config, "metrica": label, "media": summary["media"], "minimo": summary["minimo"],
                                     "maximo": summary["maximo"], "positivos": summary["positivos"], "negativos": summary["negativos"], "ceros": summary["ceros"]})

    # ---- Salidas (solo con todas las verificaciones superadas) ----
    between_rows = []
    for label, _, _ in METRICS:
        n = noise[label]
        for (alpha, fold), value in sorted(n["sd"].items()):
            between_rows.append({"metrica": label, "tipo": "de_por_fold", "alpha": alpha, "fold": fold, "valor": value})
        for alpha, value in sorted(n["media_por_alpha"].items()):
            between_rows.append({"metrica": label, "tipo": "media_de_por_alpha", "alpha": alpha, "fold": "", "valor": value})
        between_rows.append({"metrica": label, "tipo": "varianza_agrupada_s2_p", "alpha": "", "fold": "", "valor": n["s2_p"]})
    sd_b_ap = noise["AP"]["sd_b"]
    ratio = piso_ap / sd_b_ap if sd_b_ap > 0 else None
    noise_payload = {
        "AP": {"s2_p": noise["AP"]["s2_p"], "sd_b": sd_b_ap, "piso_ap_fase_b": piso_ap, "cociente_piso_ap_sobre_sd_b": ratio,
               "fuente_del_piso": f"{PHASE_B_ANALYSIS_REL.as_posix()}/{fb.OUTPUT_FILES['best_alpha']} (piso_de_ruido.piso_ap)"},
        "ROC-AUC": {"s2_p": noise["ROC-AUC"]["s2_p"], "sd_b": noise["ROC-AUC"]["sd_b"], "nota": "descriptiva"},
        "definicion": protocol["parte_1_estudio_de_ruido"]["ruido_del_delta_en_la_lectura_b"]["sd_b"],
        "no_es": protocol["parte_1_estudio_de_ruido"]["ruido_del_delta_en_la_lectura_b"]["no_es"],
    }
    summary_payload = {
        "rotulo_obligatorio": protocol["rotulo_obligatorio"],
        "protocolo": {"archivo": f"{PHASE_C_REL.as_posix()}/{PROTOCOL_NAME}", "commit_de_referencia": PROTOCOL_COMMIT, "idéntico_al_commit": identical},
        "entradas": {"parte_1": [f"{PHASE_B_ANALYSIS_REL.as_posix()}/{fb.OUTPUT_FILES[name]}" for name in ("runs", "registro", "by_fold", "best_alpha")],
                     "parte_2": [r["dir_rel"] for r in runs], "particion": CV_FOLDS_REL},
        "parte_1": {"componentes": components, "ruido_lectura_b": noise_payload},
        "parte_2": {"hipotesis": protocol["parte_2_ensamble_de_semillas"]["hipotesis"], "delta_por_fold": deltas,
                    "folds_con_orden_distinto": [f"{c}_fold{k}" for (c, k), r in sorted(aligned.items()) if r["orden_distinto"]]},
        "no_se_hace": protocol["parte_2_ensamble_de_semillas"]["se_reporta_siempre"]["no_se_hace"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    cs.write_csv(out_dir / OUTPUT_FILES["components"], components)
    cs.write_csv(out_dir / OUTPUT_FILES["between_seeds"], between_rows)
    e2a.write_json(out_dir / OUTPUT_FILES["noise_b"], noise_payload)
    cs.write_csv(out_dir / OUTPUT_FILES["ensemble_by_fold"], ensemble_rows)
    cs.write_csv(out_dir / OUTPUT_FILES["ensemble_summary"], ensemble_summary)
    cs.write_csv(out_dir / OUTPUT_FILES["verification"], checks)
    e2a.write_json(out_dir / OUTPUT_FILES["summary"], summary_payload)

    if verbose:
        failed = sum(not c["ok"] for c in checks)
        print(f"Comprobaciones: {len(checks)} | fallidas: {failed}")
        for config in CONFIG_ALPHA:
            for label, _, _ in METRICS:
                d = deltas[config][label]
                listed = ", ".join(f"{value:+.4f}" for value in d["deltas"])
                print(f"{config} {label}: Δ por fold (ensamble - media individual) = [{listed}] | media {d['media']:+.4f} "
                      f"| positivos {d['positivos']}, negativos {d['negativos']}, ceros {d['ceros']}")
        ratio_text = "no definido (sd_b = 0)" if ratio is None else f"{ratio:.3f}"
        print(f"Lectura b (AP): sd_b = {sd_b_ap:.4f} | piso_ap fase B = {piso_ap:.4f} | piso_ap / sd_b = {ratio_text}")
        print("Escrito en", out_dir)
    return 0, {"checks": checks, "components": components, "noise": noise, "noise_payload": noise_payload, "ensemble_rows": ensemble_rows,
               "deltas": deltas, "summary": summary_payload}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Análisis exploratorio de la fase C (ruido y ensamble de semillas) según protocolo_fase_c.json.")
    parser.add_argument("--output-dir", default=None, help=f"Carpeta de salida (por defecto {PHASE_C_REL.as_posix()}/{ANALYSIS_NAME}/).")
    args = parser.parse_args(argv)
    code, _ = run_analysis(Path.cwd(), args.output_dir)
    return code


if __name__ == "__main__":
    sys.exit(main())
