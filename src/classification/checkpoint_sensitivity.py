"""
Sensibilidad del criterio de checkpoint (sin reentrenar y sin usar test).

Para cada corrida de R4 a R7 se lee el training_history.json (métricas de validation por época) y se calcula
qué época habría elegido cada criterio:
    auc_max  : máx val_roc_auc (criterio vigente; primera época en caso de empate)
    ap_max   : máx val_average_precision (primera época en caso de empate)
    loss_min : mín val_loss, la pérdida ponderada de validation (primera época en caso de empate)
y cuáles serían el AP y la ROC-AUC de validation en esa época. Solo usa valores guardados en el historial;
no reconstruye nada con otros datos. La trayectoria de entrenamiento no depende del criterio de checkpoint
(el early stopping se basa siempre en val_loss), por lo que el historial es válido para los tres criterios.

Uso (desde la raíz del repositorio; solo biblioteca estándar):
    python -m src.classification.checkpoint_sensitivity
No sobrescribe archivos existentes.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

RESULTS_REL = Path("results") / "classification"
OUTPUT_REL = RESULTS_REL / "r7_optimizacion" / "diagnosticos"
PER_RUN_NAME = "sensibilidad_criterio_checkpoint_por_corrida.csv"
PER_CONFIG_NAME = "sensibilidad_criterio_checkpoint_por_configuracion.csv"

# criterio -> (clave del historial, sentido)
CRITERIA = {
    "auc_max": ("val_roc_auc", max),
    "ap_max": ("val_average_precision", max),
    "loss_min": ("val_loss", min),
}
CURRENT_CRITERION = "auc_max"

FAMILY_R45 = "R4-R5 arquitectura x representacion"
FAMILY_R7 = "R7 B0-B3 (ConvNeXt-Tiny A)"
R7_CONFIGS = {"early_stopping": "B1", "class_weights": "B2", "focal_loss": "B3"}


def select_epoch(values, direction):
    """Época (desde 1) elegida: primera en caso de empate, ignorando valores None."""
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if not valid:
        raise ValueError("el historial no tiene valores válidos para el criterio")
    best = direction(v for _, v in valid)
    return next(i for i, v in valid if v == best) + 1


def discover_runs(results_root):
    """Lista de corridas R4-R7: (familias, configuración, experimento, arquitectura, condición, semilla, carpeta)."""
    runs = []
    for path in sorted((results_root / "r4_benchmark").glob("*/A/seed_*/training_history.json")):
        arch, cond, seed = path.parts[-4], path.parts[-3], int(path.parts[-2].split("_")[1])
        families = [FAMILY_R45] + ([FAMILY_R7] if arch == "convnext_tiny" else [])
        config = f"{arch}-{cond}" if arch != "convnext_tiny" else "convnext_tiny-A (B0)"
        runs.append((families, config, "r4_benchmark", arch, cond, seed, path.parent))
    for path in sorted((results_root / "r5_input_representation").glob("*/*/seed_*/training_history.json")):
        arch, cond, seed = path.parts[-4], path.parts[-3], int(path.parts[-2].split("_")[1])
        runs.append(([FAMILY_R45], f"{arch}-{cond}", "r5_input_representation", arch, cond, seed, path.parent))
    for folder, label in R7_CONFIGS.items():
        for path in sorted((results_root / "r7_optimizacion" / folder).glob("seed_*/training_history.json")):
            seed = int(path.parts[-2].split("_")[1])
            runs.append(([FAMILY_R7], f"convnext_tiny-A ({label})", f"r7_optimizacion/{folder}", "convnext_tiny", "A",
                         seed, path.parent))
    return runs


def run_rows(run, history, run_config):
    """Filas (una por criterio) de una corrida."""
    families, config, experiment, arch, cond, seed, _ = run
    n_epochs = len(history["val_loss"])
    for key in ("val_roc_auc", "val_average_precision", "val_loss"):
        if key not in history or len(history[key]) != n_epochs:
            raise ValueError(f"{experiment}/{arch}/{cond}/seed_{seed}: falta '{key}' o su longitud no coincide.")
    rows = []
    for criterion, (key, direction) in CRITERIA.items():
        epoch = select_epoch(history[key], direction)
        rows.append({
            "configuracion": config, "experimento": experiment, "arquitectura": arch, "condicion": cond,
            "semilla": seed, "epocas_ejecutadas": n_epochs, "criterio": criterion, "epoca_elegida": epoch,
            "val_roc_auc": history["val_roc_auc"][epoch - 1],
            "val_average_precision": history["val_average_precision"][epoch - 1],
            "val_weighted_loss": history["val_loss"][epoch - 1],
            "epoca_vigente_run_config": run_config["epoch_best"],
            "auc_max_coincide_con_run_config": (epoch == run_config["epoch_best"]) if criterion == "auc_max" else "",
        })
    return rows


def _mean_sd(values):
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else float("nan"))


def _ranks(values_by_config):
    """Rango 1 = mayor valor (empates: el mismo rango)."""
    return {c: 1 + sum(1 for o in values_by_config.values() if o > v) for c, v in values_by_config.items()}


def config_rows(run_rows_all, runs):
    """Resumen por configuración y familia: media/sd entre semillas, rangos y si el orden cambia."""
    families = {}
    for run in runs:
        for family in run[0]:
            families.setdefault(family, set()).add(run[1])

    out = []
    for family, configs in families.items():
        per = {}  # (configuración, criterio) -> resumen entre semillas
        for config in sorted(configs):
            for criterion in CRITERIA:
                rows = [r for r in run_rows_all if r["configuracion"] == config and r["criterio"] == criterion]
                ap_mean, ap_sd = _mean_sd([r["val_average_precision"] for r in rows])
                auc_mean, auc_sd = _mean_sd([r["val_roc_auc"] for r in rows])
                per[(config, criterion)] = {
                    "n": len(rows), "ap_mean": ap_mean, "ap_sd": ap_sd, "auc_mean": auc_mean, "auc_sd": auc_sd,
                    "epochs": "/".join(str(r["epoca_elegida"]) for r in sorted(rows, key=lambda r: r["semilla"])),
                }
        rank_ap = {c: _ranks({cfg: per[(cfg, c)]["ap_mean"] for cfg in configs}) for c in CRITERIA}
        rank_auc = {c: _ranks({cfg: per[(cfg, c)]["auc_mean"] for cfg in configs}) for c in CRITERIA}
        order = lambda ranks: sorted(ranks, key=lambda cfg: (ranks[cfg], cfg))
        same_order_ap = {c: order(rank_ap[c]) == order(rank_ap[CURRENT_CRITERION]) for c in CRITERIA}
        same_order_auc = {c: order(rank_auc[c]) == order(rank_auc[CURRENT_CRITERION]) for c in CRITERIA}
        for config in sorted(configs):
            row = {"familia": family, "configuracion": config, "n_semillas": per[(config, CURRENT_CRITERION)]["n"]}
            for c in CRITERIA:
                p = per[(config, c)]
                row.update({
                    f"epocas_{c}": p["epochs"],
                    f"ap_media_{c}": round(p["ap_mean"], 4), f"ap_sd_{c}": round(p["ap_sd"], 4),
                    f"auc_media_{c}": round(p["auc_mean"], 4), f"auc_sd_{c}": round(p["auc_sd"], 4),
                    f"rango_ap_{c}": rank_ap[c][config], f"rango_auc_{c}": rank_auc[c][config],
                })
            for c in CRITERIA:
                if c != CURRENT_CRITERION:
                    row[f"orden_ap_igual_a_{CURRENT_CRITERION}_en_{c}"] = same_order_ap[c]
                    row[f"orden_auc_igual_a_{CURRENT_CRITERION}_en_{c}"] = same_order_auc[c]
            out.append(row)
    return out


def write_csv(path, rows):
    if Path(path).exists():
        raise FileExistsError(f"{path} ya existe; no se sobrescribe.")
    columns = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    root = Path.cwd()
    runs = discover_runs(root / RESULTS_REL)
    all_rows, mismatches = [], []
    for run in runs:
        folder = run[-1]
        history = json.loads((folder / "training_history.json").read_text(encoding="utf-8"))
        run_config = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
        rows = run_rows(run, history, run_config)
        all_rows.extend(rows)
        mismatches += [r for r in rows if r["auc_max_coincide_con_run_config"] is False]

    summary = config_rows(all_rows, runs)
    out_dir = root / OUTPUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / PER_RUN_NAME, all_rows)
    write_csv(out_dir / PER_CONFIG_NAME, summary)

    print(f"Corridas leídas: {len(runs)} | filas por corrida: {len(all_rows)} | configuraciones (por familia): {len(summary)}")
    print(f"Época de auc_max distinta de epoch_best en run_config.json: {len(mismatches)} corridas")
    print(f"Escrito: {(OUTPUT_REL / PER_RUN_NAME).as_posix()} y {(OUTPUT_REL / PER_CONFIG_NAME).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
