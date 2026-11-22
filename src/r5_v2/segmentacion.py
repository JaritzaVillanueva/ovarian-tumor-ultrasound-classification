"""Entrenamiento de UN fold (U-Net `unet__s42__f{k}`), predicción OOF y artefactos por intento.

Receta heredada de `notebooks/R5_input_representation/01_unet_baseline.ipynb` (solo celdas de código) y `src/segmentation/unet.py`
(sin editar). Diferencias deliberadas respecto al notebook (declaradas en el protocolo): datos por filas de la partición con
binarización `> 0`; validación interna = un fold de desarrollo (no `validation/` histórico); el checkpoint se escribe UNA vez al
final (estado de la mejor época en memoria) y solo guarda `model_state_dict` y metadatos (sin estado del optimizador).

Invariante: el fold k NO interviene en entrenamiento ni en la elección del checkpoint, y sus máscaras de referencia no entran al
modelo: la predicción usa un dataset sin máscara y las métricas las leen DESPUÉS de guardar las predicciones. (`verificar_datos`
abre antes las máscaras de las 820, incluidas las del fold k, solo para comprobarlas y bloquear; no alimenta ningún modelo.)
torch/PIL se importan de forma diferida.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

from src.r4_v2.common import sha256_archivo

from . import comun as C
from . import metricas_seg as MS


# ---------------------------------------------------------------- datos (torch diferido)
class DatasetSeg:
    """Dataset por filas de la partición (no carga directorios completos). Imagen: RGB, bilineal, [0,1]; máscara: `> 0` con vecino más próximo."""

    def __init__(self, filas, rutas_imagen, rutas_mascara, tam, con_mascara: bool):
        self.filas, self.rutas_imagen, self.rutas_mascara = list(filas), rutas_imagen, rutas_mascara
        self.tam, self.con_mascara = tuple(tam), con_mascara

    def __len__(self):
        return len(self.filas)

    def __getitem__(self, indice):
        import torch
        from PIL import Image
        from .mascaras import leer_mascara_binaria
        fila = self.filas[indice]
        with Image.open(self.rutas_imagen[fila["image_id"]]) as im:
            arr = np.asarray(im.convert("RGB").resize(self.tam, resample=Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        salida = {"image": torch.from_numpy(np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))), "image_id": fila["image_id"]}
        if self.con_mascara:
            m = leer_mascara_binaria(self.rutas_mascara[fila["image_id"]], self.tam).astype(np.float32)
            salida["mask"] = torch.from_numpy(np.ascontiguousarray(m[None]))
        return salida


def _crear_unet():
    from src.segmentation.unet import UNet
    return UNet(in_channels=3, out_channels=1)


def dice_iou_torch(logits, objetivos, umbral: float, eps: float):
    """Idéntico a `dice_iou_scores` del notebook histórico: por imagen, sigmoide >= umbral."""
    import torch
    pred = (torch.sigmoid(logits) >= umbral).float()
    dims = (1, 2, 3)
    inter, pa, ta = (pred * objetivos).sum(dim=dims), pred.sum(dim=dims), objetivos.sum(dim=dims)
    return (2.0 * inter + eps) / (pa + ta + eps), (inter + eps) / (pa + ta - inter + eps)


def _fijar_semillas(semilla: int) -> None:
    import torch
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)


def _epoca(modelo, cargador, criterio, optimizador, dispositivo, umbral, eps, entrenar: bool) -> dict:
    import torch
    modelo.train(entrenar)
    perdida = dice = iou = 0.0
    n = 0
    with torch.set_grad_enabled(entrenar):
        for lote in cargador:
            x = lote["image"].to(dispositivo, non_blocking=True)
            y = lote["mask"].to(dispositivo, non_blocking=True)
            if entrenar:
                optimizador.zero_grad()
            logits = modelo(x)
            loss = criterio(logits, y)
            if entrenar:
                loss.backward()
                optimizador.step()
            with torch.no_grad():
                d, i = dice_iou_torch(logits, y, umbral, eps)
            b = x.size(0)
            perdida, dice, iou, n = perdida + loss.item() * b, dice + d.sum().item(), iou + i.sum().item(), n + b
    return {"loss": perdida / n, "dice": dice / n, "iou": iou / n}


def entorno_torch(dispositivo) -> dict:
    import torch
    from PIL import Image  # noqa: F401
    import PIL
    try:
        import cv2
        cv = cv2.__version__
    except ImportError:
        cv = None
    return {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "pillow": PIL.__version__, "cv2": cv,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "dispositivo": str(dispositivo), "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark}


def entrenar_fold_torch(ctx: dict) -> dict:
    """Entrena con los folds ∉ {k, v}; el fold interno v solo elige el checkpoint (máximo Dice medio; empate: la época más temprana)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    r, roles = ctx["config"]["receta"], ctx["roles"]
    tam, semilla = tuple(r["image_size"]), r["semilla"]
    dispositivo = ctx.get("device") or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_tr = DatasetSeg(roles["entrenamiento"], ctx["rutas_imagen"], ctx["rutas_mascara"], tam, True)
    ds_va = DatasetSeg(roles["validacion_interna"], ctx["rutas_imagen"], ctx["rutas_mascara"], tam, True)
    gen = torch.Generator()
    gen.manual_seed(semilla)
    pin = dispositivo.type == "cuda"
    cargador_tr = DataLoader(ds_tr, batch_size=r["batch_size"], shuffle=True, num_workers=r["num_workers"], pin_memory=pin, generator=gen)
    cargador_va = DataLoader(ds_va, batch_size=r["batch_size"], shuffle=False, num_workers=r["num_workers"], pin_memory=pin)

    _fijar_semillas(semilla)
    modelo = (ctx.get("crear_modelo") or _crear_unet)().to(dispositivo)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=r["learning_rate"])
    criterio = nn.BCEWithLogitsLoss()
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}
    mejor_dice, mejor_epoca, mejor_estado = -float("inf"), None, None
    for epoca in range(1, ctx["epocas"] + 1):
        tr = _epoca(modelo, cargador_tr, criterio, optimizador, dispositivo, r["umbral"], r["eps_dice"], True)
        va = _epoca(modelo, cargador_va, criterio, None, dispositivo, r["umbral"], r["eps_dice"], False)
        for k, v in (("epoch", epoca), ("train_loss", tr["loss"]), ("val_loss", va["loss"]), ("val_dice", va["dice"]), ("val_iou", va["iou"])):
            hist[k].append(v)
        if va["dice"] > mejor_dice:                                        # estricto: un empate conserva la época más temprana
            mejor_dice, mejor_epoca = va["dice"], epoca
            mejor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
        print(f"fold {ctx['fold']} época {epoca:02d}/{ctx['epocas']} | train_loss {tr['loss']:.4f} | val_loss {va['loss']:.4f} "
              f"val_dice {va['dice']:.4f} val_iou {va['iou']:.4f}", flush=True)
    ckpt = Path(ctx["dir"]) / f"unet_fold{ctx['fold']}_best.pth"
    with open(ckpt, "xb") as f:                                            # una sola escritura; nunca sobrescribe
        torch.save({"epoch": mejor_epoca, "model_state_dict": mejor_estado, "val_dice": mejor_dice, "seed": semilla,
                    "image_size": tam, "batch_size": r["batch_size"], "learning_rate": r["learning_rate"], "threshold": r["umbral"],
                    "fold": ctx["fold"], "segmentador": ctx["segmentador"]}, f)
    return {"historial": hist, "checkpoint": ckpt, "epoca_guardada": mejor_epoca, "entorno": entorno_torch(dispositivo)}


def predecir_fold_torch(ctx: dict, checkpoint, dir_mascaras) -> list:
    """Predice SOLO las imágenes del fold k con el checkpoint del fold k (recargado de disco); guarda `<id>_pred.png` en {0,255}."""
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader

    r = ctx["config"]["receta"]
    tam = tuple(r["image_size"])
    dispositivo = ctx.get("device") or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = (ctx.get("crear_modelo") or _crear_unet)().to(dispositivo)
    guardado = torch.load(checkpoint, map_location=dispositivo, weights_only=True)
    if guardado["fold"] != ctx["fold"]:
        raise C.GuardError(f"el checkpoint {checkpoint} es del fold {guardado['fold']}, no del {ctx['fold']}")
    if guardado.get("segmentador") != ctx["segmentador"]:
        raise C.GuardError(f"el checkpoint {checkpoint} es del segmentador {guardado.get('segmentador')}, no de {ctx['segmentador']}")
    if "epoca_esperada" in ctx and (guardado["epoch"] != ctx["epoca_esperada"] or guardado["val_dice"] != ctx["val_dice_esperado"]):
        raise C.GuardError(f"el checkpoint recargado (época {guardado['epoch']}, val_dice {guardado['val_dice']!r}) no es el de la época "
                           f"elegida por el fold interno (época {ctx['epoca_esperada']}, val_dice {ctx['val_dice_esperado']!r})")
    modelo.load_state_dict(guardado["model_state_dict"])
    modelo.eval()
    ds = DatasetSeg(ctx["roles"]["prediccion"], ctx["rutas_imagen"], None, tam, False)      # sin máscaras de referencia
    salida = []
    with torch.no_grad():
        for lote in DataLoader(ds, batch_size=r["batch_size"], shuffle=False, num_workers=0):
            pred = (torch.sigmoid(modelo(lote["image"].to(dispositivo))) >= r["umbral"]).squeeze(1).cpu().numpy().astype(np.uint8)
            for i, p in zip(lote["image_id"], pred):
                ruta = Path(dir_mascaras) / f"{i}_pred.png"
                with open(ruta, "xb") as f:
                    Image.fromarray(p * 255).save(f, format="PNG")
                salida.append({"image_id": i, "path": ruta})
    return salida


# ---------------------------------------------------------------- un fold con artefactos y run_config
COLUMNAS_MANIFIESTO = ["image_id", "fold", "segmentador", "sha256_bytes_checkpoint", "sha256_mascara", "epoca_elegida", "ruta_mascara"]
COLUMNAS_HISTORIAL = ["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "elegida"]


def ejecutar_fold(root, config, particion, *, modo, fold, raiz_modo, raiz_datos_modo, rutas_imagen, rutas_mascara, verificar,
                  atestaciones, epocas, entrenar=None, predecir=None, calcular_caja=None, crear_modelo=None, device=None) -> dict:
    """Un fold con un intento NUEVO. Con integridad bloqueada NO se crea ningún directorio; otros fallos conservan el intento."""
    entrenar, predecir, calcular_caja = entrenar or entrenar_fold_torch, predecir or predecir_fold_torch, calcular_caja or MS.caja_roi
    inicio = C.exigir_integridad(verificar())
    roles = C.roles_de_fold(particion.filas, fold, config["diseno"]["folds"])
    sid = C.segmentador_id(config, fold)
    dir_intento = C.crear_intento(Path(raiz_modo) / f"fold{fold}")
    intento = int(dir_intento.name.split("_")[1])
    dir_mask = C.directorio_nuevo(Path(raiz_datos_modo) / f"fold{fold}" / dir_intento.name)
    C.escribir_json_nuevo(dir_intento / "intento_iniciado.json", {"fold": fold, "segmentador": sid, "intento": intento, "modo": modo,
                                                                    "fecha": C.ahora(), "estado": "en_curso"})
    claves_rol = ((C.ROL_ENTRENAMIENTO, "entrenamiento"), (C.ROL_VALIDACION_INTERNA, "validacion_interna"), (C.ROL_PREDICCION, "prediccion"))
    por_rol = {rol: roles[clave] for rol, clave in claves_rol}
    C.escribir_csv_nuevo(dir_intento / "ids_por_rol.csv",
                         [{"image_id": r["image_id"], "rol": rol, "fold_de_la_imagen": r["fold"], "label_v2": r["label_v2"]}
                          for rol, v in por_rol.items() for r in v], ["image_id", "rol", "fold_de_la_imagen", "label_v2"])
    cfg = {
        "esquema": "run_config_g3_segmentacion_v1/1", "protocolo_id": C.PROTOCOLO_ID, "mapeo": C.MAPEO_ID, "particion": C.PARTICION_ID,
        "leyenda": C.LEYENDA, "conformidad_academica_asesor": config["protocolo"]["conformidad_academica_asesor"],
        "modo": modo, "fold": fold, "fold_validacion_interna": roles["fold_validacion_interna"], "segmentador": sid, "intento": intento,
        "fecha_inicio": C.ahora(), "git_commit_head": inicio["git_commit_head"], "rama": inicio["rama"],
        "sha256_bytes_archivo_particion": particion.sha256_bytes_archivo, "semilla": config["receta"]["semilla"], "epocas": epocas,
        "receta": config["receta"], "roi": config["roi"], "atestaciones_del_autor": atestaciones or {},
        "n_por_rol": {rol: len(v) for rol, v in por_rol.items()},
        "positivas_por_rol": {rol: sum(r["label_v2"] for r in v) for rol, v in por_rol.items()},
        "sha256_ids_por_rol": {rol: C.hash_de_ids([r["image_id"] for r in v]) for rol, v in por_rol.items()},
        "ids_por_rol_archivo": "ids_por_rol.csv", "verificacion_inicio": inicio,
    }
    estado, motivo = "fallo_tecnico", None
    try:
        ctx = {"root": root, "config": config, "fold": fold, "segmentador": sid, "roles": roles, "dir": dir_intento,
               "rutas_imagen": rutas_imagen, "rutas_mascara": rutas_mascara, "epocas": epocas, "crear_modelo": crear_modelo, "device": device}
        res = entrenar(ctx)
        hist = res["historial"]
        if len(hist["val_dice"]) != epocas:
            raise C.GuardError(f"el historial tiene {len(hist['val_dice'])} épocas (esperadas {epocas})")
        elegida = MS.epoca_maxima_dice(hist["val_dice"])
        if res["epoca_guardada"] != elegida:
            raise C.GuardError("el checkpoint guardado no corresponde al máximo estricto del Dice de validación interna")
        ckpt = Path(res["checkpoint"])
        sha_ckpt = sha256_archivo(ckpt)
        ctx["epoca_esperada"], ctx["val_dice_esperado"] = elegida, hist["val_dice"][elegida - 1]   # el predictor contrasta el archivo recargado
        C.escribir_csv_nuevo(dir_intento / f"historial_fold{fold}.csv",
                             [{"epoch": hist["epoch"][i], "train_loss": repr(float(hist["train_loss"][i])), "val_loss": repr(float(hist["val_loss"][i])),
                               "val_dice": repr(float(hist["val_dice"][i])), "val_iou": repr(float(hist["val_iou"][i])),
                               "elegida": hist["epoch"][i] == elegida} for i in range(epocas)], COLUMNAS_HISTORIAL)

        preds = predecir(ctx, ckpt, dir_mask)
        esperados = [r["image_id"] for r in roles["prediccion"]]
        if sorted(p["image_id"] for p in preds) != sorted(esperados):
            raise C.GuardError(f"fold {fold}: las predicciones no corresponden exactamente (una vez) a las imágenes del fold")
        rutas_pred = {p["image_id"]: Path(p["path"]) for p in preds}
        base = Path(root).resolve()
        manifiesto = [{"image_id": i, "fold": fold, "segmentador": sid, "sha256_bytes_checkpoint": sha_ckpt,
                       "sha256_mascara": sha256_archivo(rutas_pred[i]), "epoca_elegida": elegida,
                       "ruta_mascara": rutas_pred[i].resolve().relative_to(base).as_posix()} for i in esperados]
        C.escribir_csv_nuevo(dir_intento / f"manifiesto_mascaras_fold{fold}.csv", manifiesto, COLUMNAS_MANIFIESTO)
        filas = MS.metricas_de_fold(roles["prediccion"], rutas_mascara, rutas_pred, tuple(config["receta"]["image_size"]), config["roi"],
                                    calcular_caja, config["receta"]["eps_dice"])
        C.escribir_csv_nuevo(dir_intento / f"metricas_seg_fold{fold}.csv", filas, MS.COLUMNAS_METRICAS)
        cfg.update({
            "seleccion": {"regla": config["receta"]["checkpoint"]["regla"], "epoca_elegida": elegida,
                          "val_dice_elegida": hist["val_dice"][elegida - 1], "val_iou_elegida": hist["val_iou"][elegida - 1]},
            "checkpoint": {"nombre": ckpt.name, "tamano_bytes": ckpt.stat().st_size, "sha256_bytes_archivo": sha_ckpt},
            "sha256_bytes_archivo_historial": sha256_archivo(dir_intento / f"historial_fold{fold}.csv"),
            "sha256_bytes_archivo_manifiesto": sha256_archivo(dir_intento / f"manifiesto_mascaras_fold{fold}.csv"),
            "sha256_bytes_archivo_metricas": sha256_archivo(dir_intento / f"metricas_seg_fold{fold}.csv"),
            "directorio_mascaras": dir_mask.resolve().relative_to(base).as_posix(), "n_mascaras_predichas": len(preds),
            "entorno": res.get("entorno")})
        estado = "completada"
    except C.GuardError as exc:
        motivo = str(exc)
    except Exception as exc:                                                          # noqa: BLE001 (se conserva el intento)
        motivo = f"{type(exc).__name__}: {exc}"
    fin = verificar()
    dif = C.comparar_instantaneas(inicio, fin)
    if dif and estado == "completada":
        estado = "invalida_integridad"
    cfg.update({"estado": estado, "motivo": motivo, "verificacion_fin": fin, "diferencias_inicio_fin": dif,
                "invalida_integridad": bool(dif), "fecha_fin": C.ahora()})
    C.escribir_json_nuevo(dir_intento / "run_config.json", cfg)
    return {"fold": fold, "intento": intento, "estado": estado, "motivo": motivo, "dir": dir_intento}
