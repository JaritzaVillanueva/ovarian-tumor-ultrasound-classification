"""Entrenador de G4: réplica del de G2 (`r4_v2.entrenamiento.entrenar_fold_torch`) con la pérdida de entrenamiento parametrizada.

Única diferencia con G2: el criterio de ENTRENAMIENTO. `receta.perdida_entrenamiento` (fuente única) decide: `cross_entropy_ponderada` con alpha
(`classification.utils.compute_class_weights`) o `focal_ponderada` con gamma (`classification.losses.WeightedFocalLoss` vía `build_criterion`, pesos
alpha = 1.0). La validación (que gobierna checkpoint y parada) es CE SIN ponderar, igual que G2. Con estrategia base (alpha = 1, gamma = 0) el
criterio, los pesos y la validación coinciden con los de G2 (pruebas); Focal con gamma = 0 coincide con CE ponderada.

`ejecutar_corrida` de G2 calcula los pesos con `particion.pesos_de_clase`, que rechaza alpha != 1. En vez de editar G2, `pesos_efectivos()` sustituye
TEMPORALMENTE `entrenamiento.pesos_de_clase` por una versión que acepta alpha en [0, 1] y lo RESTAURA al salir; con alpha = 1 delega en la de G2. Así el
`run_config.json` de G2 registra los pesos EFECTIVOS. torch se importa de forma diferida; la ejecución completa con modelos reales es [ND] localmente.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from src.r4_v2 import entrenamiento as EN
from src.r4_v2 import particion as P4
from src.r4_v2.common import GuardError, escribir_json_nuevo

TIPOS_PERDIDA = ("cross_entropy_ponderada", "focal_ponderada")


# ---------------------------------------------------------------- pesos efectivos (sin torch)
def pesos_de_clase_g4(etiquetas, alpha: float = 1.0) -> dict:
    """peso_c = total * n_c^(-alpha) / sum_c(n_c * n_c^(-alpha)), como `classification.utils.compute_class_weights`; alpha = 1 delega en la de G2."""
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0.0 <= alpha <= 1.0:
        raise GuardError(f"alpha de los pesos de clase debe estar en [0, 1] (recibido {alpha!r})")
    if alpha == 1.0:
        return P4.pesos_de_clase(etiquetas, 1.0)
    cuentas = Counter(int(e) for e in etiquetas)
    if set(cuentas) != {0, 1}:
        raise GuardError("K3: el entrenamiento del fold debe tener ambas clases para calcular los pesos")
    total = sum(cuentas.values())
    suav = {c: float(n) ** (-alpha) for c, n in sorted(cuentas.items())}
    z = sum(cuentas[c] * suav[c] for c in suav)
    return {c: total * suav[c] / z for c in sorted(suav)}


@contextmanager
def pesos_efectivos():
    """Sustituye `entrenamiento.pesos_de_clase` mientras dura el bloque y lo restaura SIEMPRE; no admite anidamiento ni un original distinto del de G2."""
    original = EN.pesos_de_clase
    if original is not P4.pesos_de_clase:
        raise GuardError("entrenamiento.pesos_de_clase ya está sustituido o difiere del de G2: no se anida ni se continúa")
    EN.pesos_de_clase = pesos_de_clase_g4
    try:
        yield
    finally:
        EN.pesos_de_clase = original


# ---------------------------------------------------------------- criterio (torch diferido)
def validar_perdida(perdida: dict) -> dict:
    if perdida.get("tipo") not in TIPOS_PERDIDA:
        raise GuardError(f"perdida_entrenamiento.tipo desconocido: {perdida.get('tipo')!r} (válidos: {TIPOS_PERDIDA})")
    if perdida["tipo"] == "focal_ponderada" and (perdida.get("gamma") is None or perdida["gamma"] < 0 or perdida["alpha_pesos_de_clase"] != 1.0):
        raise GuardError("focal_ponderada exige gamma >= 0 y pesos de clase alpha = 1.0")
    if perdida["tipo"] == "cross_entropy_ponderada" and perdida.get("gamma") is not None:
        raise GuardError("cross_entropy_ponderada no admite gamma (un solo factor por estrategia)")
    return perdida


def construir_criterio(perdida: dict, pesos, device=None):
    """Criterio de ENTRENAMIENTO. `pesos`: tensor float32 de 2 elementos. CE ponderada o Focal ponderada (`build_criterion`)."""
    import torch.nn as nn
    from src.classification.losses import build_criterion
    validar_perdida(perdida)
    w = pesos if device is None else pesos.to(device)
    if perdida["tipo"] == "cross_entropy_ponderada":
        return nn.CrossEntropyLoss(weight=w)        # igual que G2
    return build_criterion("focal", w, focal_gamma=float(perdida["gamma"]))


def entrenar_fold_g4(ctx: dict) -> dict:
    """Entrena un fold con la receta de G2 y la pérdida de `ctx['config']['receta']['perdida_entrenamiento']`. [ND] localmente (torchvision)."""
    import torch
    import torch.nn as nn

    from src.classification import models as modelos
    from src.classification.dataloader import create_dataloader
    from src.classification.evaluate import predict
    from src.classification.train import set_seed, train_model
    from src.classification.transforms import get_train_transforms, get_val_transforms
    from src.classification.utils import compute_class_weights
    from src.r4_v2 import pesos as W
    from src.r4_v2.dataset_lesiones import DatasetLesiones

    receta = ctx["config"]["receta"]
    perdida = validar_perdida(receta["perdida_entrenamiento"])
    arq, semilla, dir_intento = ctx["arq"], ctx["semilla"], Path(ctx["dir"])
    crear = {"resnet50": modelos.create_resnet50_classifier, "densenet121": modelos.create_densenet121_classifier,
             "convnext_tiny": modelos.create_convnext_tiny_classifier}[arq]
    size = receta["entrada"]["image_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(semilla)
    ds_train = DatasetLesiones(ctx["root"], ctx["train"], get_train_transforms(receta["aumento"]["nombre"], size))
    ds_val = DatasetLesiones(ctx["root"], ctx["val"], get_val_transforms(size))
    train_loader = create_dataloader(ds_train, batch_size=receta["batch_size"], shuffle=True, num_workers=receta["num_workers"], seed=semilla)
    val_loader = create_dataloader(ds_val, batch_size=receta["batch_size"], shuffle=False, num_workers=receta["num_workers"], seed=semilla)
    set_seed(semilla)
    modelo = crear(pretrained=receta["pretrained"], num_classes=receta["num_classes"], freeze_backbone=receta["freeze_backbone"])
    procedencia = W.procedencia_tras_instanciar(arq, modelo, pretrained=receta["pretrained"])        # K8, tras instanciar
    versiones = W.versiones_torch()
    difiere = None
    if ctx["referencia"] is not None:
        difiere = W.comparar_con_referencia(arq, procedencia, ctx["referencia"], versiones)["version_difiere_referencia"]
    modelo = modelo.to(device)

    pesos = torch.tensor([ctx["pesos"][0], ctx["pesos"][1]], dtype=torch.float32)
    esperado = compute_class_weights([int(r["label_v2"]) for r in ctx["train"]], alpha=perdida["alpha_pesos_de_clase"])
    if not torch.allclose(pesos, esperado):
        raise GuardError("los pesos de clase difieren de classification.utils.compute_class_weights")
    criterion = construir_criterio(perdida, pesos, device)
    escribir_json_nuevo(dir_intento / "perdida_efectiva_g4.json", {
        "esquema": "perdida_efectiva_g4/1", "perdida_entrenamiento": perdida, "criterio": type(criterion).__name__,
        "pesos_efectivos": {"0": float(pesos[0]), "1": float(pesos[1])}, "validacion": "CrossEntropyLoss sin ponderar (gobierna checkpoint y parada, como G2)"})
    optimizer = torch.optim.AdamW(modelo.parameters(), lr=receta["learning_rate"], weight_decay=receta["weight_decay"])
    ckpt = dir_intento / "checkpoint_mejor.pth"
    hist = train_model(
        modelo, train_loader, val_loader, criterion, optimizer, device, epochs=ctx["epocas_max"], save_path=str(ckpt),
        patience=receta["early_stopping"]["paciencia"], min_delta=receta["early_stopping"]["min_delta"], seed=semilla,
        use_amp=receta["use_amp"], predictions_path=str(dir_intento / "probabilidades_por_epoca.csv"),
        val_image_ids=[r["image_id"] for r in ctx["val"]], val_expected_labels=np.array([r["label_v2"] for r in ctx["val"]]),
        checkpoint_metric="val_loss", val_criterion=nn.CrossEntropyLoss())   # validación SIN ponderar gobierna checkpoint y parada

    if list(hist["val_loss"]) != list(hist["val_unweighted_loss"]):
        raise GuardError("val_loss (con val_criterion sin ponderar) difiere de val_unweighted_loss")
    meta = dir_intento / "checkpoint_mejor_meta.json"
    epoca_guardada = json.loads(meta.read_text())["epoch"]
    meta.unlink()
    historial = {("val_loss_unweighted" if k == "val_loss" else k): v for k, v in hist.items() if k != "val_unweighted_loss"}

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


def precargar_modulos_g4() -> None:
    """Como G2: importa torch y los módulos históricos ANTES de la verificación de integridad."""
    EN.precargar_modulos()
