"""
Análisis de la fase B del brazo del agente de optimización (R7): estudio descriptivo de la atenuación de los pesos de
clase (α), aplicando literalmente results/classification/r7_optimizacion/agente/fase_b_alpha/protocolo_fase_b.json.
Es descriptivo: no hay regla de elección, ni p-valores ni intervalos de confianza. Los umbrales y valores se leen del
protocolo y los puntos que el protocolo no define se informan (ver `open_points`), sin elegir por el autor.

Lee (solo archivos de validation; no entrena, no carga checkpoints y no usa test):
    - las 45 corridas nuevas de fase_b_alpha/corridas/alpha_{000,025,075}/seed_<s>/fold_<k>/;
    - las 30 reutilizadas de B0 (α = 1.0) y B2 (α = 0.5): semilla 42 de robustez_e2/corridas/ y semillas 43 y 44 de
      agente/confirmacion/; si la huella de equivalencia no se cumple, la semilla 42 sale de las 10 corridas condicionales;
    - fase_b_alpha/verificacion_equivalencia/veredicto_huella.txt, que decide qué corridas de semilla 42 entran.
Escribe en fase_b_alpha/analisis/ sin sobrescribir (las figuras fig1 y fig2 las genera el notebook 10):
    metricas_por_corrida.csv, metricas_por_fold.csv, delta_por_fold_y_semilla.csv, resumen_delta.csv, piso_ruido.csv,
    mejor_alpha_observado.json, verificacion_corridas.csv y registro_intentos_fase_b.csv.

Solo usa la biblioteca estándar.

Uso (desde la raíz del repositorio):
    python -m src.classification.agente_fase_b_analysis
"""
from __future__ import annotations

import argparse
import hashlib
import math
import re
import statistics
import sys
from pathlib import Path

from src.classification import agente_confirmacion_analysis as cf
from src.classification import agente_etapa1_analysis as a1
from src.classification import checkpoint_sensitivity as cs
from src.classification import robustez_e2_analysis as e2a
from src.classification.cv_folds_e2 import N_FOLDS, read_csv_rows

AGENT_REL = a1.AGENT_REL
PHASE_B_REL = AGENT_REL / "fase_b_alpha"
PROTOCOL_NAME = "protocolo_fase_b.json"
PROTOCOL_COMMIT = "ed10abd"  # commit del protocolo de la fase B
ANALYSIS_NAME = "analisis"
FINGERPRINT_REL = PHASE_B_REL / "verificacion_equivalencia" / "veredicto_huella.txt"

OUTPUT_FILES = {
    "runs": "metricas_por_corrida.csv",
    "by_fold": "metricas_por_fold.csv",
    "delta": "delta_por_fold_y_semilla.csv",
    "delta_summary": "resumen_delta.csv",
    "noise_floor": "piso_ruido.csv",
    "best_alpha": "mejor_alpha_observado.json",
    "verification": "verificacion_corridas.csv",
    "registro": "registro_intentos_fase_b.csv",
}
FIGURE_FILES = {"fig1": "fig1_ap_auc_vs_alpha.png", "fig2": "fig2_delta_ap_vs_b0.png"}  # las escribe el notebook

B0_ALPHA, B2_ALPHA = 1.0, 0.5
REFERENCE_IDS = {B0_ALPHA: "alpha_100", B2_ALPHA: "alpha_050"}  # ids de las corridas condicionales del protocolo
REFERENCE_NAMES = {B0_ALPHA: "b0", B2_ALPHA: "b2"}
METRIC_KEYS = (("AP", "ap"), ("ROC-AUC", "auc"))
PHASE_A_KEYS = ("lr_schedule", "layer_decay", "drop_path_rate", "label_smoothing", "augmentation", "image_size", "ema_decay")
STATUS_COMPLETED, STATUS_FAILED, STATUS_NOT_RUN = a1.STATUS_COMPLETED, a1.STATUS_FAILED, a1.STATUS_NOT_RUN
LABEL_ABOVE, LABEL_WITHIN = "por_encima_del_piso", "dentro_del_piso_de_ruido"


# ---------------------------------------------------------------------------------------------------------
# Protocolo
# ---------------------------------------------------------------------------------------------------------
def load_protocol(root):
    path = Path(root) / PHASE_B_REL / PROTOCOL_NAME
    return e2a.read_json(path), path


def design(protocol):
    return {"alphas": [float(a) for a in protocol["espacio_de_busqueda"]["alpha"]],
            "seeds": [int(s) for s in protocol["fijos"]["semillas"]], "folds": list(range(N_FOLDS))}


def tie_threshold(protocol):
    """Umbral de empate de mejor_alpha_observado leído del texto ('diferencia con el máximo es < 0.005')."""
    match = re.search(r"<\s*(\d+(?:\.\d+)?)", protocol["mejor_alpha_observado"]["desempate"])
    if not match:
        raise ValueError("No se pudo leer el umbral de empate de mejor_alpha_observado.")
    return float(match.group(1))


def consistency_minimum(protocol):
    """Mínimo de folds de la regla 4/5 leído del texto de la lectura b."""
    match = re.search(r"al menos (\d+) de (\d+)", protocol["analisis_descriptivo"]["2_delta_pareado_frente_a_b0"]["lectura_b_principal"])
    if not match or int(match.group(2)) != N_FOLDS:
        raise ValueError("No se pudo leer la regla 4/5 de la lectura b.")
    return int(match.group(1))


def compared_config_keys(protocol):
    """Claves de run_config.json que deben coincidir con B0 (lista 'iguales_en_las_30' del protocolo)."""
    items = protocol["reutilizacion"]["verificacion_argumento_por_argumento"]["iguales_en_las_30"]
    return [re.split(r"[\s(=]", item.strip(), maxsplit=1)[0] for item in items]


def config_id_of(alpha):
    return REFERENCE_IDS.get(alpha) or f"alpha_{int(round(alpha * 100)):03d}"


def flags_of(alpha):
    return "" if alpha == B0_ALPHA else f"--class-weight-alpha {alpha}"


# ---------------------------------------------------------------------------------------------------------
# Comandos del protocolo y huella de equivalencia
# ---------------------------------------------------------------------------------------------------------
def verify_protocol_commands(protocol):
    """Los 45 comandos nuevos = sustitución de la plantilla; los 10 condicionales = misma plantilla (B0 sin el flag)."""
    commands = protocol["comandos"]
    template, subs = commands["plantilla"], commands["sustituciones"]
    rows = []

    def add(name, ok, detail=""):
        rows.append({"corrida": "comandos_fase_b", "comprobacion": name, "ok": bool(ok), "detalle": detail})

    expected = {}
    for seed in subs["seed"]:
        for item in subs["id_y_alpha"]:
            for k in subs["k"]:
                run_id = f"{item['id']}_seed{seed}_fold{k}"
                expected[run_id] = {"config_id": item["id"], "alpha": float(item["alpha"]), "seed": seed, "fold": k,
                                    "comando": template.format(s=seed, alpha=item["alpha"], k=k, id=item["id"])}
    listed = commands["corridas"]
    add("45 corridas nuevas = 3 α x 3 semillas x 5 folds, ids únicos",
        len(listed) == len(expected) == 45 == len({c["id"] for c in listed}), str(len(listed)))
    mismatches = [c["id"] for c in listed if c["id"] not in expected or
                  {k: c[k] for k in ("config_id", "alpha", "seed", "fold", "comando")} != expected[c["id"]]]
    add("cada comando nuevo = plantilla con su id, α, semilla y fold", not mismatches and set(expected) == {c["id"] for c in listed},
        "; ".join(mismatches[:3]))
    add("orden por semilla (42 -> 43 -> 44)", [c["seed"] for c in listed] == sorted(c["seed"] for c in listed), "")
    add("los α nuevos son los de la rejilla distintos de 1.0 y 0.5",
        sorted({float(i["alpha"]) for i in subs["id_y_alpha"]}) == sorted(a for a in design(protocol)["alphas"] if a not in (B0_ALPHA, B2_ALPHA)), "")

    conditional = commands["condicionales"]["corridas"]
    bad = []
    for c in conditional:
        text = template.format(s=c["seed"], alpha=c["alpha"], k=c["fold"], id=c["config_id"])
        if float(c["alpha"]) == B0_ALPHA:
            text = text.replace(f" --class-weight-alpha {c['alpha']}", "")
        if (c["comando"] != text or c["config_id"] != REFERENCE_IDS.get(float(c["alpha"])) or c["seed"] != 42
                or c["id"] != f"{c['config_id']}_seed{c['seed']}_fold{c['fold']}"):
            bad.append(c["id"])
    add("10 condicionales = B0 y B2, semilla 42, 5 folds, plantilla (B0 sin --class-weight-alpha)",
        not bad and len(conditional) == 10 and {(float(c["alpha"]), c["fold"]) for c in conditional} == {(a, k) for a in (B0_ALPHA, B2_ALPHA) for k in range(N_FOLDS)},
        "; ".join(bad[:3]))
    return rows


def fingerprint_accepted(root, protocol):
    """
    True si veredicto_huella.txt contiene cada línea esperada de comandos.huella_resultado_esperado; False si alguna falta;
    None si el archivo no existe (el protocolo trata 'no se aporta' igual que 'no se cumple').
    """
    path = Path(root) / FINGERPRINT_REL
    expected = re.findall(r"'([^']+)'", protocol["comandos"]["huella_resultado_esperado"])
    if not expected:
        raise ValueError("No se pudieron leer las líneas esperadas de la huella.")
    if not path.exists():
        return None, expected
    lines = {line.strip() for line in path.read_text(encoding="utf-8").splitlines()}
    return all(e in lines for e in expected), expected


# ---------------------------------------------------------------------------------------------------------
# Corridas planeadas: nuevas, reutilizadas y condicionales
# ---------------------------------------------------------------------------------------------------------
def _command_run(item, kind, alpha, origin):
    args = e2a.parse_command(item["comando"])
    return {"id": item["id"], "config_id": item["config_id"], "alpha": alpha, "seed": int(item["seed"]), "fold": int(item["fold"]),
            "command": item["comando"], "args": args, "dir_rel": args["--output-dir"], "flags": flags_of(alpha), "kind": kind,
            "origin": origin}


def planned_new_runs(protocol):
    return [_command_run(c, "nueva", float(c["alpha"]), "nueva") for c in protocol["comandos"]["corridas"]]


def planned_conditional_runs(protocol):
    return [_command_run(c, "condicional", float(c["alpha"]), "nueva") for c in protocol["comandos"]["condicionales"]["corridas"]]


# ---------------------------------------------------------------------------------------------------------
# Verificación propia de la fase B (además de las de la confirmación y de robustez_e2)
# ---------------------------------------------------------------------------------------------------------
def manifest_hash(path):
    """SHA-256 de fold_manifest.csv con finales de línea normalizados a LF (git puede convertirlos al hacer checkout)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def phase_b_checks(run, b0_runs_by_fold, keys):
    """α registrado, claves de configuración iguales a B0, sin claves de la fase A, manifiesto idéntico al de B0, sin NaN y época en rango."""
    cfg = run["config"]
    rows = []

    def add(name, ok, detail=""):
        rows.append({"corrida": run["id"], "comprobacion": name, "ok": bool(ok), "detalle": detail})

    alpha = run["alpha"]
    add("fase B: class_weight_alpha registrado = α esperado (B0 no lo registra: α = 1.0)",
        cfg.get("class_weight_alpha", 1.0) == alpha and (("class_weight_alpha" in cfg) == (alpha != B0_ALPHA or "--class-weight-alpha" in run.get("args", {}))),
        f"registrado={cfg.get('class_weight_alpha', 'no registrado')}, esperado={alpha}")
    expected_weights = e2a.expected_class_weights(
        [int(r["binary_label"]) for r in run["manifest"] if r["experimental_split"] == "train"], alpha)
    add("fase B: pesos de clase = recalculados con las etiquetas de train del fold y el α esperado",
        all(abs(a - b) <= 1e-5 * max(1.0, abs(b)) for a, b in zip(cfg["class_weights"], expected_weights)), str(cfg["class_weights"]))
    add("fase B: sin claves de la fase A", not any(k in cfg for k in PHASE_A_KEYS), str([k for k in PHASE_A_KEYS if k in cfg]))

    references = b0_runs_by_fold[run["fold"]]
    b0_cfg = references[0]["config"]
    differing = {k: (cfg.get(k), b0_cfg.get(k)) for k in keys if cfg.get(k) != b0_cfg.get(k)}
    add("fase B: claves de configuración iguales a B0 del mismo fold (lista 'iguales_en_las_30')", not differing, str(differing) if differing else "")
    own = manifest_hash(run["dir"] / "fold_manifest.csv")
    reference_hashes = {manifest_hash(r["dir"] / "fold_manifest.csv") for r in references}
    add("fase B: SHA-256 de fold_manifest.csv = el de B0 en el mismo fold (todas sus semillas)",
        reference_hashes == {own}, own[:16])

    y_prob = [float(r["probability_malignant"]) for r in run["predictions"]]
    history = run["history"]
    numeric = [v for key in ("train_loss", "val_loss", "val_unweighted_loss", "val_roc_auc", "val_average_precision") for v in history.get(key, [])]
    finite = lambda values: all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)
    add("fase B: sin NaN (métricas, historial, probabilidades)",
        finite([run["metrics"]["roc_auc"], run["metrics"]["average_precision"]]) and finite(numeric) and finite(y_prob), "")
    add("fase B: época del checkpoint dentro del rango", 1 <= cfg["epoch_best"] <= cfg["epochs_run"] <= cfg["epochs_max"],
        f"epoch_best={cfg['epoch_best']}, epochs_run={cfg['epochs_run']}, epochs_max={cfg['epochs_max']}")
    add("fase B: semilla y fold registrados", cfg["seed"] == run["seed"] and cfg.get("fold") == run["fold"], f"seed={cfg['seed']}, fold={cfg.get('fold')}")
    return rows


# ---------------------------------------------------------------------------------------------------------
# Estadísticos (funciones puras)
# ---------------------------------------------------------------------------------------------------------
def describe(deltas, minimum=None):
    """media, DE muestral (n-1, descriptiva), mínimo, máximo y signos; con `minimum`, la regla 4/5 en ambas direcciones."""
    s = e2a.summarize(deltas)
    s["desv_est_muestral"] = statistics.stdev(deltas) if len(deltas) > 1 else None
    if minimum is not None:
        favored = e2a.consistency(s, minimum)["favorece"]
        s["veredicto"] = {"primera": "mejor_consistente", "segunda": "peor_consistente"}.get(favored, "no_consistente")
    return s


def delta_readings(stats, alpha, seeds, minimum, key="ap"):
    """
    Δ frente a B0 (mismo fold y semilla). stats: α -> semilla -> fold -> {'ap', 'auc', ...}.
        a: por semilla (5 folds);  b: promedio de las semillas por fold, regla 4/5;  c: signos de los 15 pares (semilla, fold).
    """
    pairs = {s: [stats[alpha][s][k][key] - stats[B0_ALPHA][s][k][key] for k in range(N_FOLDS)] for s in seeds}
    reading_a = {s: describe(d) for s, d in pairs.items()}
    fold_deltas = [statistics.mean(stats[alpha][s][k][key] for s in seeds) - statistics.mean(stats[B0_ALPHA][s][k][key] for s in seeds)
                   for k in range(N_FOLDS)]
    reading_b = dict(describe(fold_deltas, minimum), deltas=fold_deltas)
    flat = [d for s in seeds for d in pairs[s]]
    reading_c = {"n": len(flat), "positivos": sum(d > 0 for d in flat), "negativos": sum(d < 0 for d in flat), "ceros": sum(d == 0 for d in flat)}
    return {"a": reading_a, "b": reading_b, "c": reading_c, "pairs": pairs}


def noise_floor(stats, seeds, minimum):
    """
    Piso de ruido: para c ∈ {B0, B2} y cada par (s1, s2) de semillas, media sobre los 5 folds de métrica(c, s2, k) - métrica(c, s1, k).
    piso_ap = máximo de |media| de AP sobre los 6 pares. También la DE entre semillas por fold (n-1).
    """
    rows = []
    pairs = [(a, b) for i, a in enumerate(seeds) for b in seeds[i + 1:]]
    for alpha in (B0_ALPHA, B2_ALPHA):
        for s1, s2 in pairs:
            for metric, key in METRIC_KEYS:
                deltas = [stats[alpha][s2][k][key] - stats[alpha][s1][k][key] for k in range(N_FOLDS)]
                d = describe(deltas, minimum)
                rows.append({"tipo": "par_de_semillas", "configuracion": REFERENCE_NAMES[alpha].upper(), "alpha": alpha, "semillas": f"{s1}->{s2}",
                             "metrica": metric, "fold": "", "valor": d["media"], "folds_positivos": d["folds_positivos"],
                             "folds_negativos": d["folds_negativos"], "folds_cero": d["folds_cero"], "veredicto_4_de_5": d["veredicto"] if metric == "AP" else ""})
        for metric, key in METRIC_KEYS:
            sds = [statistics.stdev(stats[alpha][s][k][key] for s in seeds) for k in range(N_FOLDS)]
            for k, sd in enumerate(sds):
                rows.append({"tipo": "de_entre_semillas", "configuracion": REFERENCE_NAMES[alpha].upper(), "alpha": alpha, "semillas": ",".join(map(str, seeds)),
                             "metrica": metric, "fold": k, "valor": sd, "folds_positivos": "", "folds_negativos": "", "folds_cero": "", "veredicto_4_de_5": ""})
            rows.append({"tipo": "de_entre_semillas_media_folds", "configuracion": REFERENCE_NAMES[alpha].upper(), "alpha": alpha,
                         "semillas": ",".join(map(str, seeds)), "metrica": metric, "fold": "", "valor": statistics.mean(sds),
                         "folds_positivos": "", "folds_negativos": "", "folds_cero": "", "veredicto_4_de_5": ""})
    ap_pairs = [r for r in rows if r["tipo"] == "par_de_semillas" and r["metrica"] == "AP"]
    top = max(ap_pairs, key=lambda r: abs(r["valor"]))
    piso = abs(top["valor"])
    rows.append({"tipo": "piso_ap", "configuracion": top["configuracion"], "alpha": top["alpha"], "semillas": top["semillas"], "metrica": "AP",
                 "fold": "", "valor": piso, "folds_positivos": "", "folds_negativos": "", "folds_cero": "", "veredicto_4_de_5": ""})
    consistent_pairs = sum(r["veredicto_4_de_5"] in ("mejor_consistente", "peor_consistente") for r in ap_pairs)
    return rows, piso, top, consistent_pairs


def noise_label(readings, piso):
    """
    Etiqueta de 4_etiqueta_de_lectura (solo AP): por encima del piso si (i) la lectura b es consistente en alguna dirección,
    (ii) el Δ medio de la lectura a tiene el mismo signo estricto en las 3 semillas y (iii) |ΔAP medio de la lectura b| > piso.
    """
    b = readings["b"]
    signs = {e2a.sign_of(r["media"]) for r in readings["a"].values()}
    conditions = {
        "i_lectura_b_consistente": b["veredicto"] in ("mejor_consistente", "peor_consistente"),
        "ii_mismo_signo_en_cada_semilla": len(signs) == 1 and "0" not in signs,
        "iii_abs_delta_medio_b_mayor_que_piso": abs(b["media"]) > piso,
    }
    return {**conditions, "etiqueta": LABEL_ABOVE if all(conditions.values()) else LABEL_WITHIN,
            # descriptivo, no es parte de la regla: si la dirección de la regla 4/5 coincide con el signo de la media
            "direccion_4_de_5_coincide_con_el_signo_medio": (b["veredicto"] == "mejor_consistente" and b["media"] > 0)
            or (b["veredicto"] == "peor_consistente" and b["media"] < 0)}


def curve_shape(values):
    """Forma de una curva ordenada por α creciente: monótona (estricta) creciente / decreciente, o no monótona."""
    diffs = [b - a for a, b in zip(values, values[1:])]
    if all(d > 0 for d in diffs):
        return "monotona_creciente"
    if all(d < 0 for d in diffs):
        return "monotona_decreciente"
    return "no_monotona"


def best_alpha(curve, threshold):
    """
    mejor_alpha_observado: mayor AP medio (3 semillas x 5 folds). Empate (diferencia con el máximo < umbral): mayor ROC-AUC
    medio; después, el α más cercano a 1.0. `curve`: lista de {'alpha', 'ap_medio', 'roc_auc_medio'} de los α completos.
    """
    top = max(c["ap_medio"] for c in curve)
    tied = [c for c in curve if top - c["ap_medio"] < threshold]
    chosen = sorted(tied, key=lambda c: (-c["roc_auc_medio"], abs(c["alpha"] - 1.0)))[0]
    by_ap = max(curve, key=lambda c: c["ap_medio"])
    return {"alpha": chosen["alpha"], "ap_maximo": top, "en_empate": [c["alpha"] for c in tied],
            "el_desempate_cambia_el_resultado": chosen["alpha"] != by_ap["alpha"]}


def multiplicity(n_comparisons, minimum):
    """P(al menos una de n comparaciones cumple la regla de signo por azar), con folds independientes y signos equiprobables."""
    p_one = sum(math.comb(N_FOLDS, j) for j in range(minimum, N_FOLDS + 1)) / 2 ** N_FOLDS
    return {"comparaciones": n_comparisons, "p_una": p_one, "p_al_menos_una": 1 - (1 - p_one) ** n_comparisons,
            "nota": "aproximación gruesa: los folds no son independientes"}


def verify_protocol_noise_values(protocol, floor_rows, piso):
    """Con la reutilización, las cifras del protocolo (pares de semillas, DE por fold y piso, 4 decimales) deben reproducirse."""
    section = protocol["analisis_descriptivo"]["3_piso_de_ruido"]
    rows = []

    def add(name, ok, detail):
        rows.append({"corrida": "piso de ruido", "comprobacion": name, "ok": bool(ok), "detalle": detail})

    for name, expected in section["delta_medio_entre_pares_de_semillas_AP"].items():
        config, s1, _, s2 = name.split("_")
        row = next(r for r in floor_rows if r["tipo"] == "par_de_semillas" and r["metrica"] == "AP"
                   and r["configuracion"] == config and r["semillas"] == f"{s1}->{s2}")
        observed = {"delta_medio": round(row["valor"], 4), "folds_pos": row["folds_positivos"], "folds_neg": row["folds_negativos"]}
        add(f"valor del protocolo: {name}", observed == expected, f"leído={observed}, protocolo={expected}")
    add("valor del protocolo: piso_ap", round(piso, 4) == section["piso_ap"], f"leído={round(piso, 4)}, protocolo={section['piso_ap']}")
    for config in ("B0", "B2"):
        match = re.search(rf"{config} \[([^\]]+)\]", section["de_entre_semillas_por_fold"])
        expected = [float(x) for x in match.group(1).split(",")]
        observed = [round(r["valor"], 4) for r in floor_rows if r["tipo"] == "de_entre_semillas" and r["metrica"] == "AP" and r["configuracion"] == config]
        add(f"valor del protocolo: DE de AP entre semillas por fold ({config})", observed == expected, f"leído={observed}, protocolo={expected}")
    return rows


# ---------------------------------------------------------------------------------------------------------
# Puntos que el protocolo no define
# ---------------------------------------------------------------------------------------------------------
def open_points():
    return [
        {"punto": "config_id de las corridas reutilizadas", "aplicado": "alpha_100 (B0) y alpha_050 (B2), los ids de las condicionales; la nota dice B0/B2",
         "alternativa": "b0/b2 como en la confirmación"},
        {"punto": "claves 'iguales a B0 salvo class_weight_alpha, seed y experiment'", "aplicado": "se comparan solo las claves de 'iguales_en_las_30'; class_weights se verifica por recálculo con α y los campos de resultado no se comparan",
         "alternativa": "comparar todas las claves menos esas tres (fallaría siempre por class_weights y por los campos de resultado)"},
        {"punto": "corrida nueva incompleta (carpeta sin todos los archivos)", "aplicado": "se registra como fallida y su α se informa como incompleto: queda fuera de la curva, de los Δ, de la etiqueta y del mejor α; si es de B0 o B2 (condicional) el análisis se detiene",
         "alternativa": "detener todo el análisis"},
        {"punto": "forma de la curva", "aplicado": "monótona solo si todas las diferencias consecutivas de la media global tienen el mismo signo estricto; un empate exacto es no monótona",
         "alternativa": "monotonía no estricta, o sobre las medias por fold"},
        {"punto": "mínimo y máximo de la curva por α", "aplicado": "se informan dos versiones: sobre las 15 corridas y sobre las 5 medias por fold",
         "alternativa": "una sola"},
        {"punto": "condición (ii) de la etiqueta", "aplicado": "las 3 semillas con el mismo signo estricto entre sí; no se exige que coincida con la dirección de la regla 4/5 (se informa aparte)",
         "alternativa": "exigir además la misma dirección que la lectura b"},
        {"punto": "empate del mejor α", "aplicado": "empatan los α a menos de 0.005 del máximo; entre ellos, mayor ROC-AUC medio y luego el más cercano a 1.0 (como en la etapa 1)",
         "alternativa": "empate por pares encadenados"},
        {"punto": "registro con corridas condicionales", "aplicado": "solo entran las corridas usadas; las originales de semilla 42 no tienen fila",
         "alternativa": "añadir las originales con estado no usada"},
    ]


# ---------------------------------------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------------------------------------
def _print_failures(rows):
    for c in rows:
        if not c["ok"]:
            print(f"  FALLA {c['corrida']}: {c['comprobacion']} -> {c['detalle']}")


def run_analysis(root, output_dir=None, verbose=True):
    """
    Verifica las corridas y escribe las tablas. Devuelve (código de salida, resultados). No escribe nada si falta alguna
    corrida, si alguna comprobación falla o si algún archivo de salida ya existe.
    """
    root = Path(root)
    protocol, protocol_path = load_protocol(root)
    agent_protocol, _ = a1.load_agent_protocol(root)
    e2_protocol, _ = e2a.load_protocol(root)
    plan = design(protocol)
    seeds, alphas = plan["seeds"], plan["alphas"]
    minimum, threshold = consistency_minimum(protocol), tie_threshold(protocol)
    folds_rows = read_csv_rows(root / cf.CV_FOLDS_REL)
    out_dir = Path(output_dir) if output_dir else root / PHASE_B_REL / ANALYSIS_NAME

    checks = verify_protocol_commands(protocol)
    identical = a1.file_matches_commit(root, PHASE_B_REL / PROTOCOL_NAME, PROTOCOL_COMMIT, protocol_path)
    checks.append({"corrida": "protocolo", "comprobacion": f"idéntico al del commit {PROTOCOL_COMMIT}", "ok": identical is not False,
                   "detalle": "no verificable (git no disponible)" if identical is None else str(identical)})
    accepted, expected_lines = fingerprint_accepted(root, protocol)
    reuse_seed42 = accepted is True
    checks.append({"corrida": "huella", "comprobacion": "veredicto_huella.txt contiene las líneas esperadas (decide la reutilización de la semilla 42)",
                   "ok": True, "detalle": {True: "se cumple: se reutilizan las 30 corridas", False: "no se cumple: entran las 10 condicionales",
                                           None: "no se aporta: entran las 10 condicionales"}[accepted]})

    # Semillas 43 y 44 de B0 y B2: comandos y verificación de la confirmación (fase A)
    candidates = a1.candidate_table(agent_protocol)
    _, confirmation_runs, confirmation_checks = cf.load_confirmation_commands(root, agent_protocol, candidates)
    checks += confirmation_checks
    if any(not c["ok"] for c in checks):
        _print_failures(checks)
        return 1, None

    new_runs = [a1.load_agent_run(root, r) for r in planned_new_runs(protocol)]
    conditional_runs = [a1.load_agent_run(root, r) for r in planned_conditional_runs(protocol)]
    reused_43_44 = []
    for r in confirmation_runs:
        if r["config_id"] in ("b0", "b2"):
            alpha = B0_ALPHA if r["config_id"] == "b0" else B2_ALPHA
            reused_43_44.append(a1.load_agent_run(root, dict(r, alpha=alpha, config_id=REFERENCE_IDS[alpha], kind="confirmacion",
                                                             origin=f"reutilizada:{r['dir_rel']}", confirmation_id=r["config_id"])))

    conditional_present = [r["id"] for r in conditional_runs if r["status"] != STATUS_NOT_RUN]
    if reuse_seed42:
        if conditional_present:
            print(f"La huella se cumple pero hay corridas condicionales ({conditional_present[:3]}): el protocolo no las prevé; decide el autor.")
            return 1, None
        seed42 = []
        for ref in a1.load_reference_runs(root, agent_protocol, e2_protocol):
            alpha = B0_ALPHA if ref["config_id"] == "b0" else B2_ALPHA
            seed42.append(dict(ref, alpha=alpha, config_id=REFERENCE_IDS[alpha], kind="robustez_e2", origin=f"reutilizada:{ref['dir_rel']}",
                               id=f"{REFERENCE_IDS[alpha]}_seed42_fold{ref['fold']}", flags=flags_of(alpha)))
    else:
        seed42 = conditional_runs

    runs = new_runs + seed42 + reused_43_44
    not_run = [r["id"] for r in runs if r.get("status") == STATUS_NOT_RUN]
    failed_runs = [r for r in runs if r.get("status") == STATUS_FAILED]
    if verbose:
        print(f"Corridas: {len(runs)} | nuevas: {len(new_runs)} | semilla 42: {'reutilizadas' if reuse_seed42 else 'condicionales'} "
              f"| sin carpeta: {len(not_run)} | incompletas: {len(failed_runs)}")
    if not_run:
        print(f"Faltan corridas por ejecutar (sin carpeta de salida; ¿falta el git pull?): {not_run[:5]}{' ...' if len(not_run) > 5 else ''}")
        return 1, None
    blocking = [r["id"] for r in failed_runs if r["alpha"] in (B0_ALPHA, B2_ALPHA)]
    if blocking:
        print(f"Corridas de B0 o B2 incompletas: sin ellas no hay Δ ni piso; decide el autor: {blocking}")
        return 1, None

    completed = [r for r in runs if r.get("status", STATUS_COMPLETED) == STATUS_COMPLETED]
    b0_by_fold = {k: [r for r in completed if r["alpha"] == B0_ALPHA and r["fold"] == k] for k in range(N_FOLDS)}
    keys = compared_config_keys(protocol)
    design_b = {"seeds": seeds, "folds": plan["folds"]}
    agent_design = cf.read_confirmation_design(agent_protocol)
    for run in completed:
        if run["kind"] in ("nueva", "condicional"):
            checks += cf.verify_confirmation_run(run, folds_rows, design_b, expected_dir=run["dir_rel"])
        elif run["kind"] == "confirmacion":
            checks += cf.verify_confirmation_run(dict(run, config_id=run["confirmation_id"]), folds_rows, agent_design)
        else:
            checks += e2a.verify_run(run, e2_protocol, folds_rows)
        checks += phase_b_checks(run, b0_by_fold, keys)
    for alpha in alphas:
        combos = {(r["seed"], r["fold"]) for r in runs if r["alpha"] == alpha}
        checks.append({"corrida": config_id_of(alpha), "comprobacion": "3 semillas x 5 folds planeados",
                       "ok": combos == {(s, k) for s in seeds for k in range(N_FOLDS)}, "detalle": str(len(combos))})

    stats = {}
    for run in completed:
        stats.setdefault(run["alpha"], {}).setdefault(run["seed"], {})[run["fold"]] = {
            "ap": run["metrics"]["average_precision"], "auc": run["metrics"]["roc_auc"],
            "epoca": run["config"]["epoch_best"], "epocas": run["config"]["epochs_run"]}
    incomplete = sorted({r["alpha"] for r in failed_runs})
    complete_alphas = [a for a in alphas if a not in incomplete]

    floor_rows, piso, floor_top, consistent_pairs = noise_floor(stats, seeds, minimum)
    if reuse_seed42:
        checks += verify_protocol_noise_values(protocol, floor_rows, piso)

    failed = [c for c in checks if not c["ok"]]
    if verbose:
        print(f"Comprobaciones: {len(checks)} | fallidas: {len(failed)}")
    if failed:
        _print_failures(failed)
        return 1, None
    existing = [out_dir / name for name in list(OUTPUT_FILES.values()) + list(FIGURE_FILES.values()) if (out_dir / name).exists()]
    if existing:
        raise FileExistsError("Ya existen archivos de salida; no se sobrescribe ninguno:\n" + "\n".join(str(p) for p in existing))

    # 1. Curva
    run_rows = [{"alpha": r["alpha"], "config_id": r["config_id"], "seed": r["seed"], "fold": r["fold"],
                 "ap": r["metrics"]["average_precision"], "roc_auc": r["metrics"]["roc_auc"], "epoca_elegida": r["config"]["epoch_best"],
                 "epocas_ejecutadas": r["config"]["epochs_run"], "origen": r["origin"]}
                for r in sorted(completed, key=lambda r: (r["alpha"], r["seed"], r["fold"]))]
    fold_rows, curve = [], []
    for alpha in complete_alphas:
        for k in range(N_FOLDS):
            row = {"alpha": alpha, "config_id": config_id_of(alpha), "fold": k}
            for metric, key in METRIC_KEYS:
                values = [stats[alpha][s][k][key] for s in seeds]
                name = "ap" if key == "ap" else "roc_auc"
                row.update({f"{name}_semilla_{s}": v for s, v in zip(seeds, values)})
                row[f"{name}_media_semillas"] = statistics.mean(values)
                row[f"{name}_de_entre_semillas"] = statistics.stdev(values)
            row["epocas_ejecutadas_media"] = statistics.mean(stats[alpha][s][k]["epocas"] for s in seeds)
            row["epoca_elegida_media"] = statistics.mean(stats[alpha][s][k]["epoca"] for s in seeds)
            fold_rows.append(row)
        entry = {"alpha": alpha, "config_id": config_id_of(alpha)}
        for metric, key in METRIC_KEYS:
            name = "ap" if key == "ap" else "roc_auc"
            all_values = [stats[alpha][s][k][key] for s in seeds for k in range(N_FOLDS)]
            fold_means = [statistics.mean(stats[alpha][s][k][key] for s in seeds) for k in range(N_FOLDS)]
            entry.update({f"{name}_medio": statistics.mean(all_values), f"{name}_min_corridas": min(all_values), f"{name}_max_corridas": max(all_values),
                          f"{name}_min_medias_por_fold": min(fold_means), f"{name}_max_medias_por_fold": max(fold_means),
                          f"{name}_de_entre_semillas_media_folds": statistics.mean(statistics.stdev(stats[alpha][s][k][key] for s in seeds) for k in range(N_FOLDS))})
        entry["epocas_ejecutadas_media"] = statistics.mean(stats[alpha][s][k]["epocas"] for s in seeds for k in range(N_FOLDS))
        entry["epoca_elegida_media"] = statistics.mean(stats[alpha][s][k]["epoca"] for s in seeds for k in range(N_FOLDS))
        curve.append(entry)
    shape = {"ap": curve_shape([c["ap_medio"] for c in curve]), "roc_auc": curve_shape([c["roc_auc_medio"] for c in curve]),
             "alphas_en_orden": [c["alpha"] for c in curve], "completa": not incomplete}

    # 2-4. Δ frente a B0, tres lecturas y etiqueta
    delta_rows, summary_rows, labels = [], [], {}
    for alpha in [a for a in complete_alphas if a != B0_ALPHA]:
        for metric, key in METRIC_KEYS:
            readings = delta_readings(stats, alpha, seeds, minimum, key)
            for s in seeds:
                for k, d in enumerate(readings["pairs"][s]):
                    delta_rows.append({"alpha": alpha, "config_id": config_id_of(alpha), "metrica": metric, "seed": s, "fold": k,
                                       "valor_alpha": stats[alpha][s][k][key], "valor_b0": stats[B0_ALPHA][s][k][key], "delta": d, "signo": e2a.sign_of(d)})
            base = {"alpha": alpha, "config_id": config_id_of(alpha), "metrica": metric}
            for s, r in readings["a"].items():
                summary_rows.append(dict(base, lectura="a_por_semilla", semillas=str(s), n=5, media=r["media"], desv_est_muestral=r["desv_est_muestral"],
                                         minimo=r["minimo"], maximo=r["maximo"], positivos=r["folds_positivos"], negativos=r["folds_negativos"],
                                         ceros=r["folds_cero"], veredicto="", etiqueta=""))
            b = readings["b"]
            label = noise_label(readings, piso) if key == "ap" else None
            if label:
                labels[alpha] = label
            summary_rows.append(dict(base, lectura="b_promedio_3_semillas", semillas=",".join(map(str, seeds)), n=5, media=b["media"],
                                     desv_est_muestral=b["desv_est_muestral"], minimo=b["minimo"], maximo=b["maximo"], positivos=b["folds_positivos"],
                                     negativos=b["folds_negativos"], ceros=b["folds_cero"], veredicto=b["veredicto"] if key == "ap" else "",
                                     etiqueta=label["etiqueta"] if label else ""))
            c = readings["c"]
            summary_rows.append(dict(base, lectura="c_15_pares", semillas=",".join(map(str, seeds)), n=c["n"], media="", desv_est_muestral="",
                                     minimo="", maximo="", positivos=c["positivos"], negativos=c["negativos"], ceros=c["ceros"], veredicto="", etiqueta=""))

    best = best_alpha(curve, threshold)
    registry = []
    for r in sorted(runs, key=lambda r: (r["alpha"], r["seed"], r["fold"])):
        done = r.get("status", STATUS_COMPLETED) == STATUS_COMPLETED
        notes = []
        if r["alpha"] in (B0_ALPHA, B2_ALPHA):
            notes.append("B0" if r["alpha"] == B0_ALPHA else "B2")
        if r.get("status") == STATUS_FAILED:
            notes.append("faltan: " + ", ".join(r["missing"]))
        if r["kind"] == "condicional":
            notes.append("condicional: la huella no se cumplió o no se aportó; la corrida original de semilla 42 no entra")
        if r["alpha"] in incomplete and done:
            notes.append("α incompleto (sin imputar valores)")
        registry.append({"etapa": "fase_b", "config_id": r["config_id"], "alpha": r["alpha"], "fold": r["fold"], "seed": r["seed"], "flags": flags_of(r["alpha"]),
                         "ap": r["metrics"]["average_precision"] if done else "", "roc_auc": r["metrics"]["roc_auc"] if done else "",
                         "epoca_elegida": r["config"]["epoch_best"] if done else "", "epocas_ejecutadas": r["config"]["epochs_run"] if done else "",
                         "estado": r.get("status", STATUS_COMPLETED), "origen": r["origin"], "ruta": r["dir_rel"], "nota": "; ".join(notes)})
    registry = [{column: row[column] for column in protocol["registro"]["columnas"]} for row in registry]

    payload = {
        "protocolo": {"archivo": f"{PHASE_B_REL.as_posix()}/{PROTOCOL_NAME}", "commit_de_referencia": PROTOCOL_COMMIT, "idéntico_al_commit": identical},
        "rotulo_obligatorio": protocol["mejor_alpha_observado"]["rotulo_obligatorio"],
        "mejor_alpha_observado": {**best, "config_id": config_id_of(best["alpha"]), "definicion": protocol["mejor_alpha_observado"]["definicion"],
                                  "desempate": protocol["mejor_alpha_observado"]["desempate"], "umbral_de_empate_leido": threshold,
                                  "sesgo": protocol["mejor_alpha_observado"]["sesgo"], "no_es": protocol["mejor_alpha_observado"]["no_es"]},
        "curva": curve, "forma_de_la_curva": shape,
        "alphas_incompletos": incomplete,
        "piso_de_ruido": {"piso_ap": piso, "par_que_lo_define": f"{floor_top['configuracion']} {floor_top['semillas']}",
                          "pares_de_una_misma_configuracion_que_cumplen_4_de_5": consistent_pairs, "pares": 6,
                          "semilla_42": "reutilizada" if reuse_seed42 else "condicional"},
        "etiquetas_frente_a_b0": {str(a): v for a, v in labels.items()},
        "multiplicidad": multiplicity(len([a for a in alphas if a != B0_ALPHA]), minimum),
        "huella": {"veredicto": accepted, "lineas_esperadas": expected_lines},
        "no_se_calcula": protocol["analisis_descriptivo"]["p_valor"],
        "puntos_no_definidos_por_el_protocolo": open_points(),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    cs.write_csv(out_dir / OUTPUT_FILES["runs"], run_rows)
    cs.write_csv(out_dir / OUTPUT_FILES["by_fold"], fold_rows)
    cs.write_csv(out_dir / OUTPUT_FILES["delta"], delta_rows)
    cs.write_csv(out_dir / OUTPUT_FILES["delta_summary"], summary_rows)
    cs.write_csv(out_dir / OUTPUT_FILES["noise_floor"], floor_rows)
    cs.write_csv(out_dir / OUTPUT_FILES["verification"], checks)
    cs.write_csv(out_dir / OUTPUT_FILES["registro"], registry)
    e2a.write_json(out_dir / OUTPUT_FILES["best_alpha"], payload)
    if verbose:
        resumen_etiquetas = ", ".join(f"{a}: {v['etiqueta']}" for a, v in labels.items())
        print(f"Piso de ruido (AP): {piso:.4f} | etiquetas: {{{resumen_etiquetas}}}")
        print(f"Mejor α observado (exploratorio): {best['alpha']} | en empate: {best['en_empate']}")
        print("Escrito en", out_dir)
    return 0, {"checks": checks, "stats": stats, "curve": curve, "summaries": summary_rows, "labels": labels, "floor": floor_rows,
               "piso": piso, "best": best, "payload": payload, "registry": registry}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Análisis descriptivo de la fase B (α) según protocolo_fase_b.json.")
    parser.add_argument("--output-dir", default=None, help=f"Carpeta de salida (por defecto {PHASE_B_REL.as_posix()}/{ANALYSIS_NAME}/).")
    args = parser.parse_args(argv)
    code, _ = run_analysis(Path.cwd(), args.output_dir)
    return code


if __name__ == "__main__":
    sys.exit(main())
