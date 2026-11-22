"""Ejecutor de UN fold de G2 (protocolo §5-§6): receta común, selección de checkpoint y parada, artefactos por intento.

La mayor parte es Python puro y se prueba con un entrenador sintético inyectable (`entrenar`). Lo que requiere torch
(`entrenar_fold_torch`, `precargar_modulos`) usa importación diferida y queda [ND] localmente.

Reglas de §6.2 (misma magnitud, reglas distintas): la pérdida de validación SIN ponderar `val_loss_unweighted` gobierna
(a) el checkpoint: mínimo estricto, sin `min_delta` (un empate conserva la época más temprana) y (b) la parada: mejora mayor
que `min_delta` durante `paciencia` épocas. El código histórico llama `val_weighted_loss` a esa magnitud: aquí se renombra.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import integridad as I
from . import metricas as M
from .common import (GuardError, LEYENDA_PROVISIONAL, MAPEO_ID, PARTICION_ID, PROTOCOLO_ID, directorio_nuevo, escribir_csv_nuevo,
                     escribir_json_nuevo, sha256_archivo)
from .particion import Particion, filas_de_fold, pesos_de_clase

ESTADOS = ("completada", "fallo_tecnico", "detenida_k8", "invalida_integridad")
COLUMNAS_PROB_EPOCA = ["epoch", "image_id", "y_true", "probability_malignant"]   # formato de train.py (_append_val_predictions)


# ---------------------------------------------------------------- selección de épocas (puro)
def epoca_checkpoint(val_loss) -> int:
    """Época (1-based) del mínimo ESTRICTO de la pérdida: la primera época con el menor valor (como train.py L345)."""
    mejor, epoca = float("inf"), None
    for i, v in enumerate(val_loss, start=1):
        if v < mejor:
            mejor, epoca = v, i
    if epoca is None:
        raise GuardError("historial de pérdida vacío")
    return epoca


def simular_parada(val_loss, paciencia: int, min_delta: float):
    """Época en la que la regla de parada (train.py L383) detendría el entrenamiento, o None si no se activa."""
    referencia, contador = float("inf"), 0
    for i, v in enumerate(val_loss, start=1):
        if v < referencia - min_delta:
            referencia, contador = v, 0
        else:
            contador += 1
            if contador >= paciencia:
                return i
    return None


def comprobar_historial(val_loss, receta: dict, epocas_max: int) -> dict:
    """El historial debe seguir la regla: se detuvo en la época simulada, o agotó `epocas_max` sin activar la parada."""
    n = len(val_loss)
    parada = simular_parada(val_loss, receta["early_stopping"]["paciencia"], receta["early_stopping"]["min_delta"])
    if parada is not None and parada != n:
        raise GuardError(f"el historial ({n} épocas) no sigue la regla de parada (se detendría en la época {parada})")
    if parada is None and n != epocas_max:
        raise GuardError(f"el entrenamiento terminó en {n} épocas sin activar la parada (máximo {epocas_max})")
    return {"epocas_entrenadas": n, "parada_anticipada": parada is not None, "epoca_elegida": epoca_checkpoint(val_loss)}


# ---------------------------------------------------------------- probabilidades (sin pérdida de precisión)
def escribir_prob_por_epoca(path, epocas) -> None:
    """`epocas`: {epoca: (image_ids, y, probs float32)}. `repr(float)` de un float32 convertido es exacto."""
    filas = [{"epoch": e, "image_id": i, "y_true": int(y), "probability_malignant": repr(float(p))}
             for e, (ids, ys, ps) in sorted(epocas.items()) for i, y, p in zip(ids, ys, ps)]
    escribir_csv_nuevo(path, filas, COLUMNAS_PROB_EPOCA)


def leer_prob_por_epoca(path) -> dict:
    """{epoca: {"image_id": [...], "y": int array, "p": float32 array}}; float32 exacto desde el texto."""
    por = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = por.setdefault(int(r["epoch"]), {"image_id": [], "y": [], "p": []})
            d["image_id"].append(r["image_id"])
            d["y"].append(int(r["y_true"]))
            d["p"].append(float(r["probability_malignant"]))
    return {e: {"image_id": d["image_id"], "y": np.array(d["y"], dtype=np.int64), "p": np.array(d["p"], dtype=np.float32)}
            for e, d in por.items()}


def verificar_prob_por_epoca(por_epoca: dict, val_rows, n_epocas: int) -> None:
    """Cada época del historial tiene exactamente las imágenes del fold retenido, una vez, con y_true = label_v2."""
    if sorted(por_epoca) != list(range(1, n_epocas + 1)):
        raise GuardError(f"probabilidades por época incompletas: {sorted(por_epoca)} (esperadas 1..{n_epocas})")
    ids = [r["image_id"] for r in val_rows]
    y = np.array([r["label_v2"] for r in val_rows])
    for e, d in por_epoca.items():
        if d["image_id"] != ids or not np.array_equal(d["y"], y):
            raise GuardError(f"K7: las predicciones de la época {e} no corresponden una vez a cada imagen del fold retenido")
        if not np.isfinite(d["p"]).all() or ((d["p"] < 0) | (d["p"] > 1)).any():
            raise GuardError(f"probabilidades no válidas en la época {e}")


def comparar_recarga(guardada, recargada, alerta: float) -> dict:
    """Recarga FP32 del checkpoint frente a las probabilidades guardadas: diferencia máxima y cambios de clase a 0.5."""
    g, r = np.asarray(guardada, dtype=np.float32), np.asarray(recargada, dtype=np.float32)
    if g.shape != r.shape:
        raise GuardError("la recarga no tiene el mismo número de imágenes")
    dif = float(np.max(np.abs(g.astype(np.float64) - r.astype(np.float64)))) if g.size else 0.0
    cambios = int(np.sum((g >= M.UMBRAL_CLASIFICACION) != (r >= M.UMBRAL_CLASIFICACION)))
    return {"diferencia_maxima_absoluta": dif, "cambios_de_clase_a_0_5": cambios, "umbral_de_alerta": alerta,
            "alerta_recarga": dif > alerta}                      # alerta, NO bloqueo (protocolo §6.4)


def sensibilidad_checkpoint(por_epoca: dict, pesos: dict, n_epocas: int) -> dict:
    """§6.3 (descriptiva): épocas que habrían elegido la pérdida PONDERADA mínima y el ROC-AUC máximo, dentro de la trayectoria."""
    perdidas, aucs = [], []
    for e in range(1, n_epocas + 1):
        d = por_epoca[e]
        p = np.clip(d["p"].astype(np.float64), 1e-12, 1 - 1e-12)
        w = np.where(d["y"] == 1, pesos[1], pesos[0])
        nll = -np.where(d["y"] == 1, np.log(p), np.log(1 - p))
        perdidas.append(float(np.sum(w * nll) / np.sum(w)))
        aucs.append(M.roc_auc(d["y"], d["p"]))
    e_pond = epoca_checkpoint(perdidas)
    e_auc = epoca_checkpoint([-a for a in aucs])               # primer máximo
    return {"epoca_perdida_ponderada_minima": e_pond, "epoca_roc_auc_maximo": e_auc}


# ---------------------------------------------------------------- intentos y artefactos
def id_corrida(config, arq: str, semilla: int, fold: int) -> str:
    return config["diseno"]["id_corrida"].format(arch=arq, seed=semilla, fold=fold)


def crear_intento(dir_corrida) -> Path:
    """`intento_NN` nuevo (K5): los intentos fallidos se conservan; nunca se reutiliza un directorio."""
    dir_corrida = Path(dir_corrida)
    previos = sorted(p.name for p in dir_corrida.glob("intento_*")) if dir_corrida.exists() else []
    return directorio_nuevo(dir_corrida / f"intento_{len(previos) + 1:02d}")


def estado_de_intento(dir_intento) -> str:
    """Estado de un intento; sin run_config.json (terminación sin artefactos completos) es `fallo_tecnico`."""
    rc = Path(dir_intento) / "run_config.json"
    if not rc.is_file():
        return "fallo_tecnico"
    return json.loads(rc.read_text(encoding="utf-8"))["estado"]


def intentos_de(dir_corrida) -> list:
    return sorted(Path(dir_corrida).glob("intento_*")) if Path(dir_corrida).exists() else []


def corrida_completada(dir_corrida):
    """Único intento `completada` de la corrida, o None. Más de uno es una inconsistencia."""
    ok = [p for p in intentos_de(dir_corrida) if estado_de_intento(p) == "completada"]
    if len(ok) > 1:
        raise GuardError(f"más de un intento completado en {dir_corrida}")
    return ok[0] if ok else None


def _ahora():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ejecutar_corrida(root, config, particion: Particion, *, modo: str, arq: str, semilla: int, fold: int, raiz_modo,
                     verificar, entrenar, referencia=None, atestaciones=None, epocas_max=None) -> dict:
    """Una corrida (arquitectura, semilla, fold) con K1-K8 ya comprobadas por quien llama.

    `verificar()` devuelve la verificación §4.5; `entrenar(contexto)` es el entrenador (torch en producción, sintético en pruebas).
    Una corrida con K6 bloqueado NO se lanza (no deja directorio). Cualquier otro fallo deja un intento conservado.
    """
    rid = id_corrida(config, arq, semilla, fold)
    receta = config["receta"]
    epocas_max = int(epocas_max or receta["epochs_max"])
    inicio = verificar()
    I.exigir_integridad(inicio)                                                       # K6 antes de entrenar
    train, val = filas_de_fold(particion, fold)
    pesos = pesos_de_clase([r["label_v2"] for r in train], receta["perdida_entrenamiento"]["alpha_pesos_de_clase"])
    dir_intento = crear_intento(Path(raiz_modo) / rid)
    intento = int(dir_intento.name.split("_")[1])
    escribir_json_nuevo(dir_intento / "intento_iniciado.json", {"id_corrida": rid, "intento": intento, "modo": modo,
                                                                 "fecha": _ahora(), "estado": "en_curso"})
    cfg = {
        "esquema": "run_config_g2_baseline_v1/1", "protocolo_id": PROTOCOLO_ID, "mapeo": MAPEO_ID, "particion": PARTICION_ID,
        "condicion": config["estudio"]["condicion"], "leyenda": LEYENDA_PROVISIONAL,
        "conformidad_academica_asesor": config["protocolo"]["conformidad_academica_asesor"], "modo": modo, "id_corrida": rid,
        "arquitectura": arq, "semilla": semilla, "fold": fold, "intento": intento, "fecha_inicio": _ahora(),
        "sha256_bytes_archivo_particion": particion.sha256_bytes_archivo,
        "protocolo_versionado": inicio["protocolo_versionado"], "git_commit_head": inicio["git_commit_head"],
        "rama": inicio["rama"], "entrada": inicio["entrada"], "status_aux": inicio.get("status_aux", []),
        "sha256_contenido_blob_protocolo": (inicio.get("protocolo") or {}).get("sha256_contenido_blob"),
        "referencia_pesos": None if referencia is None else {"git_blob_id": referencia["git_blob_id"],
                                                             "sha256_contenido_blob": referencia["sha256_contenido_blob"]},
        "atestaciones_del_autor": atestaciones or {}, "receta": receta, "epocas_max": epocas_max,
        "n_train": len(train), "n_validacion": len(val),
        "positivas_train": sum(r["label_v2"] for r in train), "positivas_validacion": sum(r["label_v2"] for r in val),
        "pesos_de_clase": {str(k): v for k, v in pesos.items()}, "verificacion_inicio": inicio,
    }
    estado, motivo = "fallo_tecnico", None
    try:
        res = entrenar({"root": root, "config": config, "arq": arq, "semilla": semilla, "fold": fold, "train": train,
                        "val": val, "pesos": pesos, "dir": dir_intento, "epocas_max": epocas_max, "modo": modo,
                        "referencia": referencia})
        hist = res["historial"]
        seleccion = comprobar_historial(hist["val_loss_unweighted"], receta, epocas_max)
        if res.get("epoca_checkpoint_guardado") is not None and res["epoca_checkpoint_guardado"] != seleccion["epoca_elegida"]:
            raise GuardError("el checkpoint guardado no corresponde al mínimo estricto de val_loss_unweighted")
        por_epoca = leer_prob_por_epoca(dir_intento / "probabilidades_por_epoca.csv")
        verificar_prob_por_epoca(por_epoca, val, seleccion["epocas_entrenadas"])
        e = seleccion["epoca_elegida"]
        oof = por_epoca[e]
        escribir_csv_nuevo(dir_intento / "predicciones_oof.csv",
                           [{"image_id": i, "y_true": int(y), "probability_malignant": repr(float(p)), "epoca_elegida": e}
                            for i, y, p in zip(oof["image_id"], oof["y"], oof["p"])],
                           ["image_id", "y_true", "probability_malignant", "epoca_elegida"])
        escribir_json_nuevo(dir_intento / "historial.json", hist)
        recarga = comparar_recarga(oof["p"], res["prob_recarga"], receta["recarga_checkpoint"]["alerta_diferencia_maxima_absoluta"])
        ckpt = Path(res["checkpoint"])
        cfg.update({
            "seleccion": {**seleccion, "val_loss_unweighted_elegida": hist["val_loss_unweighted"][e - 1],
                          "regla_checkpoint": receta["checkpoint"]["regla"], "regla_parada": receta["early_stopping"],
                          "sensibilidad_checkpoint": sensibilidad_checkpoint(por_epoca, pesos, seleccion["epocas_entrenadas"])},
            "recarga": recarga, "alerta_recarga": recarga["alerta_recarga"],
            "checkpoint": {"nombre": ckpt.name, "tamano_bytes": ckpt.stat().st_size, "sha256_bytes_archivo": sha256_archivo(ckpt)},
            "sha256_bytes_archivo_probabilidades_por_epoca": sha256_archivo(dir_intento / "probabilidades_por_epoca.csv"),
            "sha256_bytes_archivo_predicciones_oof": sha256_archivo(dir_intento / "predicciones_oof.csv"),
            "pesos_preentrenados": res.get("pesos_preentrenados"), "version_difiere_referencia": res.get("version_difiere_referencia"),
            "entorno": res.get("entorno")})
        estado = "completada"
    except GuardError as exc:
        estado, motivo = ("detenida_k8" if str(exc).startswith("K8") else "fallo_tecnico"), str(exc)
    except Exception as exc:                                                          # noqa: BLE001 (se conserva el intento)
        estado, motivo = "fallo_tecnico", f"{type(exc).__name__}: {exc}"
    fin = verificar()                                                                 # tras escribir los artefactos
    diferencias = I.comparar_instantaneas(inicio, fin)
    invalida = bool(diferencias) or bool(fin["bloqueos"])
    if invalida and estado == "completada":
        estado = "invalida_integridad"
    cfg.update({"estado": estado, "motivo": motivo, "verificacion_fin": fin, "diferencias_inicio_fin": diferencias,
                "invalida_integridad": invalida, "fecha_fin": _ahora()})
    escribir_json_nuevo(dir_intento / "run_config.json", cfg)
    return {"id_corrida": rid, "intento": intento, "estado": estado, "motivo": motivo, "dir": dir_intento}


# ---------------------------------------------------------------- torch [ND localmente]
def precargar_modulos() -> None:
    """Importa torch y los módulos históricos reutilizados ANTES de la verificación §4.5 (la lista dinámica los ve)."""
    import torch  # noqa: F401
    import torchvision  # noqa: F401
    from src.classification import (dataloader, evaluate, losses, models, train, training_options, transforms, utils)  # noqa: F401
    from . import dataset_lesiones  # noqa: F401  (importa PIL y torch.utils.data)


def entrenar_fold_torch(ctx: dict) -> dict:
    """Entrena un fold con la receta común. [ND] localmente: requiere torch, torchvision, Pillow y las imágenes de desarrollo."""
    import torch
    import torch.nn as nn

    from src.classification import models as modelos
    from src.classification.dataloader import create_dataloader
    from src.classification.evaluate import predict
    from src.classification.train import set_seed, train_model
    from src.classification.transforms import get_train_transforms, get_val_transforms
    from src.classification.utils import compute_class_weights
    from . import pesos as W
    from .dataset_lesiones import DatasetLesiones

    cfg, receta = ctx["config"], ctx["config"]["receta"]
    arq, semilla, dir_intento = ctx["arq"], ctx["semilla"], Path(ctx["dir"])
    crear = {"resnet50": modelos.create_resnet50_classifier, "densenet121": modelos.create_densenet121_classifier,
             "convnext_tiny": modelos.create_convnext_tiny_classifier}[arq]
    size = receta["entrada"]["image_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(semilla)
    ds_train = DatasetLesiones(ctx["root"], ctx["train"], get_train_transforms(receta["aumento"]["nombre"], size))
    ds_val = DatasetLesiones(ctx["root"], ctx["val"], get_val_transforms(size))
    train_loader = create_dataloader(ds_train, batch_size=receta["batch_size"], shuffle=True,
                                     num_workers=receta["num_workers"], seed=semilla)
    val_loader = create_dataloader(ds_val, batch_size=receta["batch_size"], shuffle=False,
                                   num_workers=receta["num_workers"], seed=semilla)
    set_seed(semilla)
    modelo = crear(pretrained=receta["pretrained"], num_classes=receta["num_classes"], freeze_backbone=receta["freeze_backbone"])
    procedencia = W.procedencia_tras_instanciar(arq, modelo, pretrained=receta["pretrained"])        # K8, tras instanciar
    versiones = W.versiones_torch()
    difiere = None
    if ctx["referencia"] is not None:
        difiere = W.comparar_con_referencia(arq, procedencia, ctx["referencia"], versiones)["version_difiere_referencia"]
    modelo = modelo.to(device)

    pesos = torch.tensor([ctx["pesos"][0], ctx["pesos"][1]], dtype=torch.float32)
    esperado = compute_class_weights([int(r["label_v2"]) for r in ctx["train"]], alpha=receta["perdida_entrenamiento"]["alpha_pesos_de_clase"])
    if not torch.allclose(pesos, esperado):
        raise GuardError("los pesos de clase difieren de classification.utils.compute_class_weights")
    criterion = nn.CrossEntropyLoss(weight=pesos.to(device))        # entrenamiento: CE ponderada, α = 1.0
    optimizer = torch.optim.AdamW(modelo.parameters(), lr=receta["learning_rate"], weight_decay=receta["weight_decay"])
    ckpt = dir_intento / "checkpoint_mejor.pth"
    hist = train_model(
        modelo, train_loader, val_loader, criterion, optimizer, device, epochs=ctx["epocas_max"], save_path=str(ckpt),
        patience=receta["early_stopping"]["paciencia"], min_delta=receta["early_stopping"]["min_delta"], seed=semilla,
        use_amp=receta["use_amp"], predictions_path=str(dir_intento / "probabilidades_por_epoca.csv"),
        val_image_ids=[r["image_id"] for r in ctx["val"]], val_expected_labels=np.array([r["label_v2"] for r in ctx["val"]]),
        checkpoint_metric="val_loss", val_criterion=nn.CrossEntropyLoss())   # validación SIN ponderar gobierna checkpoint y parada

    # el historial histórico llama `val_weighted_loss` a la magnitud calculada con `val_criterion` (aquí sin ponderar)
    if list(hist["val_loss"]) != list(hist["val_unweighted_loss"]):
        raise GuardError("val_loss (con val_criterion sin ponderar) difiere de val_unweighted_loss")
    meta = dir_intento / "checkpoint_mejor_meta.json"
    epoca_guardada = json.loads(meta.read_text())["epoch"]
    meta.unlink()                                                    # nombres engañosos (val_weighted_loss): se sustituyen
    historial = {("val_loss_unweighted" if k == "val_loss" else k): v for k, v in hist.items() if k != "val_unweighted_loss"}

    # recarga en FP32 (validate_one_epoch y predict no usan autocast); pretrained=False solo para no volver a descargar pesos
    modelo2 = crear(pretrained=False, num_classes=receta["num_classes"], freeze_backbone=False)
    modelo2.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    _, _, prob_recarga = predict(modelo2.to(device), val_loader, device)
    import sklearn
    entorno = {
        "python": sys.version, "torch": torch.__version__, "torchvision": versiones["torchvision_version"],
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "dispositivo": str(device),
        "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "amp_efectivo": bool(receta["use_amp"] and torch.cuda.is_available()),
    }
    return {"historial": historial, "checkpoint": ckpt, "prob_recarga": prob_recarga, "epoca_checkpoint_guardado": epoca_guardada,
            "pesos_preentrenados": {**procedencia, **versiones}, "version_difiere_referencia": difiere, "entorno": entorno}
