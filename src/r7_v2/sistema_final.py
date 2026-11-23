"""R7 `sistema_final_v1`: sistema final PROVISIONAL (candidato; NO congelado formalmente; no validado para el ensamble de quince) y sus herramientas.

Sistema: DenseNet121 + representación A, 15 checkpoints de G2 (semillas 42/43/44 × folds 0-4; intento `completada`), promedio aritmético simple de las 15 probabilidades de la
clase maligna (softmax float32 por modelo; promedio en float64; sin ponderaciones ni selección de miembros), regla `p >= t_hat` (empates no se rompen) con t_hat candidato
0.08456226376195748 (S* = 0.90 exploratorio; calibrado con OOF de TRES modelos por imagen: no es validación para el ensamble de quince; NO es el umbral de R8).
Alcance: clasificación binaria de lesión en imagen 2D; no detecta ovarios normales; sin uso clínico validado.

* `construir_manifiesto`: lee los 15 `run_config.json` (vía `r4_v2.agregacion`, con su comparabilidad), recalcula el SHA-256 de cada checkpoint y lo compara con el registrado; bloquea ante
  faltantes, duplicados, otro modelo, hash distinto, intento no completado o comparabilidad rota. Registra también el código de preprocesamiento/entrada y las versiones efectivas.
* `agregar_y_decidir`: promedio y regla `>=`.
* Preprocesamiento: se REUTILIZA el de G2 (`get_val_transforms`, `DatasetLesiones`, `create_dataloader`, `predict`); aquí no se reimplementa nada (`preprocesamiento()`).
* `verificar_fidelidad` (modo `fidelidad`): cada miembro SOLO sobre las imágenes de desarrollo de su PROPIO pliegue reproduce las probabilidades guardadas de su `predicciones_oof.csv`
  (tolerancia de recarga de G2: `receta.recarga_checkpoint.alerta_diferencia_maxima_absoluta`; alerta, no corrección). NO calcula ninguna métrica del ensamble de quince sobre desarrollo.
* `inferir`: función de biblioteca para la futura R8; K4 rechaza rutas con componente `test` y filas de desarrollo con etiquetas. NO hay modo CLI de inferencia.

CLI (solo la autora; D-NNN propia; no se ejecuta ahora): python -B -X pycache_prefix=<dir> -m src.r7_v2.sistema_final --modo manifiesto|fidelidad --autorizacion CONFIRMADO:D-NNN
Partición provisional (phash_le8_v1, exploratoria, post hoc); conformidad académica pendiente.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

from src.r4_v2 import agregacion as AG
from src.r4_v2 import entrenamiento as EN
from src.r4_v2.common import (RAIZ_REPO, GuardError, directorio_nuevo, escribir_json_nuevo, exigir_ruta_no_prohibida, ruta_menciona_test, sha256_archivo)  # noqa: F401
from src.r4_v2.common import CONFIG_REL as G2_CONFIG_REL
from src.r4_v2.common import cargar_config as cargar_config_g2
from src.r4_v2.entrenamiento import id_corrida
from src.r4_v2.particion import cargar_particion, filas_de_fold
from src.r5_v2.comun import comparar_instantaneas, huella_archivo
from src.r7_v2 import umbral_sistema as U
from src.r7_v2.delta_ensamble import exigir_protocolo, validar_comunes, verificar_integridad  # noqa: F401  (misma integridad parametrizada por la ruta del protocolo)

PROTOCOLO_ID = "sistema_final_v1"
ESQUEMA_CONFIG = "sistema_final_v1_config/1"
CONFIG_REL = "configs/r7_v2/sistema_final_v1.yaml"
PROTOCOLO_REL = "src/r7_v2/protocolo_sistema_final_v1.md"
SALIDA_MANIFIESTO_REL = "results/r7_v2/phash_le8_v1/sistema_final_v1/manifiesto"
SALIDA_FIDELIDAD_REL = "results/r7_v2/phash_le8_v1/sistema_final_v1/fidelidad"
MODOS = ("manifiesto", "fidelidad")                                                      # NO hay modo de inferencia
LEYENDA = U.LEYENDA
T_HAT = "0.08456226376195748"
N_MIEMBROS = 15
ESTADO = "candidato; NO congelado formalmente; calibrado con OOF de tres modelos por imagen; no validado para el ensamble de quince"
SISTEMA = {
    "estado": ESTADO, "congelado": False, "arquitectura": "densenet121", "representacion": "A", "semillas": [42, 43, 44], "folds": [0, 1, 2, 3, 4], "n_miembros": 15,
    "agregacion": "promedio aritmetico simple de las 15 probabilidades de la clase maligna (softmax float32 por modelo; promedio en float64); sin ponderaciones ni seleccion de miembros",
    "regla": "p >= t_hat", "empates": "no se rompen",
    "umbral_candidato": {"t_hat": T_HAT, "s_star": "0.90", "s_star_estatus": "objetivo exploratorio, no garantia",
                         "origen": "q_barra de tres modelos por imagen (umbral_sistema_v1); calculado en desarrollo; NO congelado formalmente; no es el umbral de R8"},
    "incertidumbre_umbral_referencia": {"n_remuestreos_validos": 10000, "p2_5": 0.022556618166466553, "mediana": 0.08456226376195748, "p97_5": 0.20131495470801988,
                                        "nota": "solo informativa, no validacion"},
    "alcance": "clasificacion binaria de lesion a nivel de imagen 2D; no detecta ovarios normales; sin uso clinico validado"}
ARCHIVOS_CODIGO = ("src/classification/transforms.py", "src/classification/dataloader.py", "src/classification/models.py", "src/classification/evaluate.py",
                   "src/r4_v2/dataset_lesiones.py", "src/r4_v2/entrenamiento.py")
LEYENDAS = (
    f"sistema final PROVISIONAL: {ESTADO}", "t_hat = 0.08456226376195748 calculado en desarrollo; NO congelado formalmente; no es el umbral de R8; S* = 0.90 es un objetivo exploratorio, no una garantía",
    "la calibración se hizo con OOF de TRES modelos por imagen; el sistema de quince no tiene OOF de desarrollo (12 de los 15 modelos vieron cada imagen): no validado para el ensamble de quince",
    "alcance: clasificación binaria de lesión a nivel de imagen 2D; no detecta ovarios normales; sin uso clínico validado",
    "la fidelidad compara cada miembro con sus propias predicciones guardadas en su propio pliegue; no calcula métricas del ensamble de quince sobre desarrollo",
    "no se afirma independencia absoluta del test histórico")
FRASES_PROTOCOLO = ("candidato", "NO congelado formalmente", "no validado para el ensamble de quince", "0.08456226376195748", "p >= t_hat", "no se rompen",
                    "promedio aritmético simple", "15 checkpoints", "manifiesto_miembros", "recalculado", "fidelidad", "propio pliegue", "alerta_diferencia_maxima_absoluta",
                    "no hay modo de inferencia", "K4", "no detecta ovarios normales", "sin uso clínico validado", "no es el umbral de R8",
                    "no se afirma independencia absoluta del test histórico")
ENCABEZADO = (f"# mapeo v2 | phash_le8_v1 (provisional) | {PROTOCOLO_ID} | sin uso de test | {LEYENDA} | sistema final provisional: {ESTADO} | "
              "t_hat no es el umbral de R8 | sin uso clínico validado")
COLUMNAS_MANIFIESTO = ["arquitectura", "semilla", "fold", "id_corrida", "intento", "ruta_checkpoint", "sha256_bytes_archivo_checkpoint", "sha256_checkpoint_registrado_run_config",
                       "coincide", "epoca_elegida", "estado", "invalida_integridad", "git_commit_head_entrenamiento", "sha256_bytes_archivo_run_config",
                       "sha256_bytes_archivo_predicciones_oof", "sha256_bytes_archivo_particion", "sha256_contenido_blob_config_g2", "sha256_contenido_blob_referencia_pesos"]


# ============================================================================ configuración
def t_hat_exacto(valor) -> float:
    """El umbral se guarda como cadena exacta; `repr(float(cadena))` debe reproducirla (ida y vuelta sin pérdida)."""
    if isinstance(valor, str):
        try:
            x = float(valor)
        except ValueError:
            raise GuardError(f"t_hat {valor!r} no es un número") from None
        if not np.isfinite(x) or repr(x) != valor:
            raise GuardError(f"t_hat {valor!r} no es la cadena exacta de un float (repr(float(t_hat)) = {repr(x)!r})")
        return x
    if isinstance(valor, float) and np.isfinite(valor):
        return float(valor)
    raise GuardError(f"t_hat inválido: {valor!r}")


def validar_config(config) -> None:
    validar_comunes(config, ESQUEMA_CONFIG, PROTOCOLO_ID, PROTOCOLO_REL, CONFIG_REL, "src/r7_v2/sistema_final.py")
    if config.get("sistema") != SISTEMA:
        raise GuardError("sistema del YAML distinto de las constantes del código (miembros, agregación, regla, t_hat, estado, congelado: false, alcance): se bloquea; "
                         "cualquier cambio es una enmienda previa al congelamiento")
    t_hat_exacto(config["sistema"]["umbral_candidato"]["t_hat"])
    if config.get("salidas") != {"manifiesto": SALIDA_MANIFIESTO_REL, "fidelidad": SALIDA_FIDELIDAD_REL}:
        raise GuardError("salidas del YAML distintas de results/r7_v2/phash_le8_v1/sistema_final_v1/{manifiesto,fidelidad}")
    for ruta in config["salidas"].values():
        exigir_ruta_no_prohibida(ruta)
    huella = set(config["integridad"]["archivos_con_huella"])
    faltan = [r for r in ARCHIVOS_CODIGO if r not in huella]
    if faltan or not isinstance(config.get("entorno_esperado"), dict):
        raise GuardError(f"integridad.archivos_con_huella debe incluir {faltan} y debe existir entorno_esperado (solo registro)")


def cargar_config(path) -> dict:
    from src.r7_v2.delta_ensamble import cargar_config as _cargar
    return _cargar(path, validar_config)


def config_g2_sistema(config_g2, config) -> dict:
    s = config["sistema"]
    c = copy.deepcopy(config_g2)
    c["diseno"]["arquitecturas"], c["diseno"]["semillas"], c["diseno"]["folds"] = [s["arquitectura"]], list(s["semillas"]), list(s["folds"])
    c["diseno"]["n_corridas"] = len(s["semillas"]) * len(s["folds"])
    return c


# ============================================================================ (2) promedio y regla
def agregar_y_decidir(probabilidades_por_miembro: dict, t_hat=T_HAT) -> dict:
    """Promedio aritmético SIMPLE (float64) de las probabilidades de los 15 miembros y regla `p >= t_hat` (los empates NO se rompen: p == t_hat es positiva).
    Exige exactamente 15 miembros con el mismo número de imágenes y probabilidades finitas en [0, 1]; sin ponderaciones ni selección de miembros."""
    if len(probabilidades_por_miembro) != N_MIEMBROS:
        raise GuardError(f"se exigen exactamente {N_MIEMBROS} miembros (recibidos {len(probabilidades_por_miembro)}); no se selecciona ni se omite ningún miembro")
    mats = [np.asarray(probabilidades_por_miembro[k], dtype=np.float64) for k in sorted(probabilidades_por_miembro, key=str)]
    if any(m.ndim != 1 or m.shape != mats[0].shape for m in mats):
        raise GuardError("todos los miembros deben aportar un vector de probabilidades del mismo tamaño")
    p = np.stack(mats)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise GuardError("probabilidades no finitas o fuera de [0, 1]")
    t = t_hat_exacto(t_hat)
    puntuacion = np.mean(p, axis=0)                                                      # promedio aritmético simple en float64
    return {"puntuacion": puntuacion, "positivo": puntuacion >= t, "t_hat": t, "n_miembros": N_MIEMBROS}


def metricas_ensamble_sobre_desarrollo(*_, **__):
    """GUARDA explícita: el ensamble de quince NO se evalúa sobre desarrollo como si fuese OOF (12 de los 15 modelos vieron cada imagen). Siempre falla."""
    raise GuardError("prohibido: ninguna métrica (AP, AUC, sensibilidad, especificidad, umbral) del ensamble de quince sobre imágenes de desarrollo; no existe OOF de ese sistema")


# ============================================================================ (1) manifiesto
def _versiones() -> dict:
    v = {"python": sys.version.split()[0], "numpy": np.__version__}
    for nombre in ("torch", "torchvision"):
        try:
            v[nombre] = __import__(nombre).__version__
        except ImportError:
            v[nombre] = None
    return v


def construir_manifiesto(root, config, config_g2, particion) -> dict:
    """Los 15 miembros únicos A|densenet121 con el SHA-256 del checkpoint RECALCULADO y comparado con el registrado. Bloquea ante cualquier inconsistencia."""
    root = Path(root).resolve()
    s = config["sistema"]
    g = config_g2_sistema(config_g2, config)
    raiz = root / config_g2["diseno"]["raiz_resultados"].strip("/") / "completa"
    esperadas = [id_corrida(g, s["arquitectura"], sem, k) for sem in s["semillas"] for k in s["folds"]]
    if len(esperadas) != N_MIEMBROS or len(set(esperadas)) != N_MIEMBROS:
        raise GuardError("el sistema debe ser exactamente 15 miembros únicos (3 semillas × 5 folds)")
    corridas = AG.cargar_corridas(raiz, g, particion)                                    # faltantes, inválidas, intento duplicado, comparabilidad y partición
    if len(corridas) != N_MIEMBROS:
        raise GuardError(f"se esperaban 15 corridas A|densenet121 y hay {len(corridas)}")
    miembros = []
    for (a, sem, k), (d, rc) in sorted(corridas.items()):
        d = Path(d)
        if (a != s["arquitectura"] or rc.get("arquitectura") != s["arquitectura"] or rc.get("condicion") != "A" or rc.get("modo") != "completa"
                or rc.get("estado") != "completada" or rc.get("invalida_integridad") or rc.get("semilla") != sem or rc.get("fold") != k):
            raise GuardError(f"{rc.get('id_corrida')}: no es una corrida completada válida de A|densenet121 con la semilla y el fold esperados")
        ck = d / (rc.get("checkpoint") or {}).get("nombre", "")
        if not (rc.get("checkpoint") or {}).get("nombre") or not ck.is_file():
            raise GuardError(f"{rc['id_corrida']}: checkpoint ausente ({ck.name or 'sin nombre'}); el manifiesto se construye donde residen los checkpoints (Falcon)")
        sha = sha256_archivo(ck)
        registrado = rc["checkpoint"]["sha256_bytes_archivo"]
        if sha != registrado:
            raise GuardError(f"{rc['id_corrida']}: el SHA-256 recalculado del checkpoint difiere del registrado en su run_config (hash distinto)")
        pred = d / "predicciones_oof.csv"
        sha_pred = sha256_archivo(pred)
        if rc.get("sha256_bytes_archivo_predicciones_oof") != sha_pred:
            raise GuardError(f"{rc['id_corrida']}: predicciones_oof.csv no coincide con el SHA-256 registrado")
        ruta_rel = ck.resolve().relative_to(root).as_posix()
        exigir_ruta_no_prohibida(ruta_rel)
        miembros.append({"arquitectura": a, "semilla": sem, "fold": k, "id_corrida": rc["id_corrida"], "intento": d.name, "ruta_checkpoint": ruta_rel,
                         "sha256_bytes_archivo_checkpoint": sha, "sha256_checkpoint_registrado_run_config": registrado, "coincide": True,
                         "epoca_elegida": rc["seleccion"]["epoca_elegida"], "estado": rc["estado"], "invalida_integridad": bool(rc["invalida_integridad"]),
                         "git_commit_head_entrenamiento": rc["git_commit_head"], "sha256_bytes_archivo_run_config": sha256_archivo(d / "run_config.json"),
                         "sha256_bytes_archivo_predicciones_oof": sha_pred, "sha256_bytes_archivo_particion": rc["sha256_bytes_archivo_particion"],
                         "sha256_contenido_blob_config_g2": rc["verificacion_inicio"]["archivos"][G2_CONFIG_REL]["sha256_contenido_blob"],
                         "sha256_contenido_blob_referencia_pesos": (rc.get("referencia_pesos") or {}).get("sha256_contenido_blob")})
    if len({m["id_corrida"] for m in miembros}) != N_MIEMBROS or len({m["sha256_bytes_archivo_checkpoint"] for m in miembros}) != N_MIEMBROS:
        raise GuardError("miembros duplicados (id_corrida o checkpoint repetidos)")
    claves = {(m["sha256_bytes_archivo_particion"], m["sha256_contenido_blob_config_g2"], m["sha256_contenido_blob_referencia_pesos"]) for m in miembros}
    if len(claves) != 1 or next(iter(claves))[0] != particion.sha256_bytes_archivo:
        raise GuardError("los 15 miembros no comparten la clave de comparabilidad (partición, configuración de G2, referencia de pesos) o difieren de la partición cargada")
    codigo = {}
    for rel in ARCHIVOS_CODIGO:
        h = huella_archivo(root, rel)
        if h["sha256_bytes_archivo"] is None:
            raise GuardError(f"{rel} no existe: no se puede registrar el código de preprocesamiento")
        codigo[rel] = h
    advertencias = []
    pin = config["referencia_g2"]["sha256_contenido_blob"]
    if next(iter(claves))[1] != pin:
        advertencias.append("el blob de la configuración de G2 con que se entrenaron los miembros difiere del pin del YAML; se registra, no se corrige")
    return {"miembros": miembros, "codigo": codigo, "receta_entrada": config_g2["receta"]["entrada"], "yaml_g2_pin": pin, "yaml_g2_blob_de_entrenamiento": next(iter(claves))[1],
            "referencia_pesos_blob": next(iter(claves))[2], "advertencias": advertencias, "entorno_efectivo": _versiones(),
            "recarga_g2": {"alerta_diferencia_maxima_absoluta": config_g2["receta"]["recarga_checkpoint"]["alerta_diferencia_maxima_absoluta"]}}


def a_json(obj):
    return U.a_json(obj)


def _preludio(root, config, config_g2, autorizacion, verificar, rel_salida, destino):
    token = U.exigir_autorizacion(autorizacion)                                          # antes de leer o escribir nada
    root = Path(root)
    validar_config(config)
    if config["estudio"]["particion"] != config_g2["estudio"]["particion"]:
        raise GuardError("la partición no coincide con la de G2")
    sha_protocolo = exigir_protocolo(root, config, FRASES_PROTOCOLO)
    verificar = verificar or (lambda: verificar_integridad(root, config))
    inicio = U.exigir_integridad(verificar())
    r6 = U.verificar_r6(root, config)
    destino = U.destino_nuevo(root, rel_salida, destino)
    return token, root, verificar, inicio, r6, sha_protocolo, destino


def ejecutar_manifiesto(root, config, config_g2, *, autorizacion, verificar=None, destino=None) -> dict:
    token, root, verificar, inicio, r6, sha_protocolo, destino = _preludio(root, config, config_g2, autorizacion, verificar, config["salidas"]["manifiesto"], destino)
    particion = cargar_particion(root, config_g2)                                        # K1-K4
    m = construir_manifiesto(root, config, config_g2, particion)
    destino = directorio_nuevo(destino)                                                  # K5: solo tras validar y calcular
    filas = [{c: mm[c] for c in COLUMNAS_MANIFIESTO} for mm in m["miembros"]]
    U.escribir_tabla_con(destino / "manifiesto_miembros.csv", filas, ENCABEZADO, "15 miembros únicos A|densenet121; SHA-256 del checkpoint recalculado y comparado con el run_config")
    fin = verificar()
    dif = comparar_instantaneas(inicio, fin)
    escribir_json_nuevo(destino / "manifiesto_miembros.json", a_json({
        "esquema": "manifiesto_sistema_final_v1/1", "protocolo_id": PROTOCOLO_ID, "leyenda": LEYENDA, "leyendas": list(LEYENDAS), "estado": ESTADO, "congelado": False,
        "sistema": config["sistema"], "n_miembros": N_MIEMBROS, **m, "sha256_bytes_archivo_manifiesto_csv": sha256_archivo(destino / "manifiesto_miembros.csv"),
        "sha256_bytes_archivo_particion": particion.sha256_bytes_archivo, "sha256_bytes_archivo_protocolo": sha_protocolo, "base_r6": r6, "autorizacion": token,
        "git_commit_head": inicio["git_commit_head"], "rama": inicio["rama"], "fecha": U.ahora(), "diferencias_inicio_fin": dif,
        "invalida_integridad": bool(dif) or bool(fin["bloqueos"]), "reporte_aceptado": not (bool(dif) or bool(fin["bloqueos"]))}))
    return {"dir": destino, "manifiesto": m, "reporte_aceptado": not (bool(dif) or bool(fin["bloqueos"])), "diferencias_inicio_fin": dif}


# ============================================================================ (3) preprocesamiento: el de G2, sin reimplementar
def preprocesamiento() -> dict:
    """Devuelve LOS MISMOS objetos que usa el entrenador de G2 (importación diferida: requiere torch y torchvision, [ND] localmente)."""
    from src.classification.dataloader import create_dataloader
    from src.classification.evaluate import predict
    from src.classification.transforms import get_val_transforms
    from src.r4_v2.dataset_lesiones import DatasetLesiones
    return {"get_val_transforms": get_val_transforms, "DatasetLesiones": DatasetLesiones, "create_dataloader": create_dataloader, "predict": predict}


def _cargar_modelo_g2(ruta, device):
    """DenseNet121 como lo construye G2 (`create_densenet121_classifier`), con los pesos del checkpoint (`weights_only=True`, como la recarga de G2)."""
    import torch
    from src.classification import models as modelos
    m = modelos.create_densenet121_classifier(pretrained=False, num_classes=2, freeze_backbone=False)
    m.load_state_dict(torch.load(ruta, map_location=device, weights_only=True))
    return m.to(device)


def _predecir_g2(root, config_g2, semilla):
    """Probabilidad de la clase maligna (softmax float32) con el preprocesamiento y el lector de G2 (`predict` de classification.evaluate)."""
    p = preprocesamiento()
    receta = config_g2["receta"]

    def predecir(modelo, filas, device):
        ds = p["DatasetLesiones"](root, [{**f, "label_v2": f.get("label_v2", 0)} for f in filas], p["get_val_transforms"](receta["entrada"]["image_size"]))
        loader = p["create_dataloader"](ds, batch_size=receta["batch_size"], shuffle=False, num_workers=receta["num_workers"], seed=semilla)
        return np.asarray(p["predict"](modelo, loader, device)[2], dtype=np.float32)
    return predecir


def _dispositivo():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================ (4) fidelidad por miembro (en su propio pliegue; sin métricas del ensamble)
def leer_manifiesto(origen: Path) -> dict:
    j, c = Path(origen) / "manifiesto_miembros.json", Path(origen) / "manifiesto_miembros.csv"
    if not j.is_file() or not c.is_file():
        raise GuardError("falta manifiesto_miembros.json/.csv (se ejecuta primero `--modo manifiesto`)")
    m = json.loads(j.read_text(encoding="utf-8"))
    if sha256_archivo(c) != m.get("sha256_bytes_archivo_manifiesto_csv") or m.get("n_miembros") != N_MIEMBROS or len(m.get("miembros", [])) != N_MIEMBROS:
        raise GuardError("el manifiesto no es coherente (SHA-256 del CSV distinto o no son 15 miembros)")
    if m.get("congelado") is not False or m.get("estado") != ESTADO:
        raise GuardError("el manifiesto declara un estado distinto de «candidato; NO congelado formalmente»")
    return m


def verificar_miembro_en_disco(root, miembro: dict) -> Path:
    ruta = Path(root) / miembro["ruta_checkpoint"]
    exigir_ruta_no_prohibida(miembro["ruta_checkpoint"])
    if not ruta.is_file():
        raise GuardError(f"{miembro['id_corrida']}: checkpoint ausente")
    if sha256_archivo(ruta) != miembro["sha256_bytes_archivo_checkpoint"]:
        raise GuardError(f"{miembro['id_corrida']}: el checkpoint no coincide con el SHA-256 del manifiesto (alterado o distinto); no se carga")
    return ruta


def leer_probabilidades_guardadas(path) -> tuple:
    with open(path, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    return [r["image_id"] for r in filas], np.array([float(r["probability_malignant"]) for r in filas], dtype=np.float32)


def verificar_fidelidad(root, config, config_g2, particion, manifiesto: dict, *, cargar_modelo=None, predecir=None, device=None) -> dict:
    """Cada miembro SOLO sobre las imágenes de desarrollo de su PROPIO pliegue debe reproducir sus probabilidades guardadas. No agrega miembros ni calcula métricas.
    Tolerancia = la de G2 (`receta.recarga_checkpoint.alerta_diferencia_maxima_absoluta`; alerta, no bloqueo)."""
    root = Path(root)
    tol = float(config_g2["receta"]["recarga_checkpoint"]["alerta_diferencia_maxima_absoluta"])
    t_hat = t_hat_exacto(config["sistema"]["umbral_candidato"]["t_hat"])
    if cargar_modelo is None:
        cargar_modelo = _cargar_modelo_g2
    dev = device if device is not None else (_dispositivo() if predecir is None else None)
    filas_out = []
    for m in manifiesto["miembros"]:
        ruta = verificar_miembro_en_disco(root, m)
        _, val = filas_de_fold(particion, m["fold"])                                     # SOLO el pliegue propio
        if any(r["fold"] != m["fold"] for r in val):
            raise GuardError(f"{m['id_corrida']}: la fidelidad se evalúa solo en el pliegue propio del miembro")
        pred_csv = root / config_g2["diseno"]["raiz_resultados"].strip("/") / "completa" / m["id_corrida"] / m["intento"] / "predicciones_oof.csv"
        if sha256_archivo(pred_csv) != m["sha256_bytes_archivo_predicciones_oof"]:
            raise GuardError(f"{m['id_corrida']}: predicciones_oof.csv no coincide con el manifiesto")
        ids, guardada = leer_probabilidades_guardadas(pred_csv)
        if ids != [r["image_id"] for r in val]:
            raise GuardError(f"{m['id_corrida']}: las predicciones guardadas no corresponden a las imágenes de su pliegue")
        modelo = cargar_modelo(ruta, dev)
        pr = (predecir or _predecir_g2(root, config_g2, m["semilla"]))
        recargada = np.asarray(pr(modelo, val, dev), dtype=np.float32)
        cmp = EN.comparar_recarga(guardada, recargada, tol)
        g, r = guardada.astype(np.float64), recargada.astype(np.float64)
        rc = json.loads((pred_csv.parent / "run_config.json").read_text(encoding="utf-8"))
        filas_out.append({"id_corrida": m["id_corrida"], "arquitectura": m["arquitectura"], "semilla": m["semilla"], "fold": m["fold"], "n_imagenes": int(guardada.size),
                          "diferencia_maxima_absoluta": cmp["diferencia_maxima_absoluta"], "cambios_de_clase_a_0_5": cmp["cambios_de_clase_a_0_5"],
                          "cambios_de_clase_a_t_hat": int(np.sum((g >= t_hat) != (r >= t_hat))), "tolerancia": tol, "alerta_recarga": bool(cmp["alerta_recarga"]),
                          "dif_recarga_g2_registrada": (rc.get("recarga") or {}).get("diferencia_maxima_absoluta"), "alerta_recarga_g2_registrada": rc.get("alerta_recarga")})
    return {"miembros": filas_out, "tolerancia": tol, "t_hat": t_hat, "alerta_global": any(f["alerta_recarga"] for f in filas_out),
            "metricas_del_ensamble_de_quince_sobre_desarrollo": False}


def ejecutar_fidelidad(root, config, config_g2, *, autorizacion, verificar=None, destino=None, origen=None, cargar_modelo=None, predecir=None, device=None) -> dict:
    token, root, verificar, inicio, r6, sha_protocolo, destino = _preludio(root, config, config_g2, autorizacion, verificar, config["salidas"]["fidelidad"], destino)
    origen = Path(origen) if origen else root / config["salidas"]["manifiesto"]
    particion = cargar_particion(root, config_g2)                                        # K1-K4
    man = leer_manifiesto(origen)
    if man.get("sha256_bytes_archivo_particion") != particion.sha256_bytes_archivo:
        raise GuardError("el manifiesto se construyó con otra partición")
    res = verificar_fidelidad(root, config, config_g2, particion, man, cargar_modelo=cargar_modelo, predecir=predecir, device=device)
    destino = directorio_nuevo(destino)                                                  # K5
    cols = ["id_corrida", "arquitectura", "semilla", "fold", "n_imagenes", "diferencia_maxima_absoluta", "cambios_de_clase_a_0_5", "cambios_de_clase_a_t_hat", "tolerancia",
            "alerta_recarga", "dif_recarga_g2_registrada", "alerta_recarga_g2_registrada"]
    U.escribir_tabla_con(destino / "fidelidad_por_miembro.csv", [{c: ("" if f[c] is None else (repr(f[c]) if isinstance(f[c], float) else f[c])) for c in cols} for f in res["miembros"]],
                         ENCABEZADO, "cada miembro en su propio pliegue; tolerancia de recarga de G2 (receta.recarga_checkpoint.alerta_diferencia_maxima_absoluta); sin métricas del ensamble")
    fin = verificar()
    dif = comparar_instantaneas(inicio, fin)
    escribir_json_nuevo(destino / "resumen_fidelidad.json", a_json({
        "esquema": "resumen_fidelidad_v1/1", "protocolo_id": PROTOCOLO_ID, "leyenda": LEYENDA, "leyendas": list(LEYENDAS), "estado": ESTADO, "tolerancia": res["tolerancia"],
        "criterio_de_tolerancia": "el de G2: receta.recarga_checkpoint.alerta_diferencia_maxima_absoluta (alerta, no bloqueo; protocolo G2 §6.4)",
        "t_hat": T_HAT, "alerta_global": res["alerta_global"], "n_miembros": len(res["miembros"]), "metricas_del_ensamble_de_quince_sobre_desarrollo": False,
        "sha256_bytes_archivo_manifiesto_csv": man["sha256_bytes_archivo_manifiesto_csv"], "entorno_efectivo": _versiones(), "sha256_bytes_archivo_particion": particion.sha256_bytes_archivo,
        "sha256_bytes_archivo_protocolo": sha_protocolo, "base_r6": r6, "autorizacion": token, "git_commit_head": inicio["git_commit_head"], "fecha": U.ahora(),
        "diferencias_inicio_fin": dif, "invalida_integridad": bool(dif) or bool(fin["bloqueos"]), "reporte_aceptado": not (bool(dif) or bool(fin["bloqueos"]))}))
    return {"dir": destino, "resultados": res, "reporte_aceptado": not (bool(dif) or bool(fin["bloqueos"])), "diferencias_inicio_fin": dif}


# ============================================================================ (5) inferencia como biblioteca para la futura R8 (sin modo CLI)
def exigir_filas_de_inferencia(filas) -> None:
    """K4: ninguna ruta con componente `test` (R8 tiene su propio protocolo y ejecutor autorizado) y ninguna fila de DESARROLLO (con etiquetas/pliegue: el ensamble de quince no se
    evalúa sobre desarrollo)."""
    for f in filas:
        for clave in ("image_path", "image_name"):
            if clave in f and ruta_menciona_test(str(f[clave])):
                raise GuardError(f"K4: ruta con componente 'test' rechazada: {f[clave]}")
        if any(k in f for k in ("label_v2", "fold", "original_class", "group_id")):
            raise GuardError("el ensamble de quince no se evalúa sobre filas de desarrollo (con etiqueta, pliegue o grupo): no existe OOF de ese sistema")


def inferir(root, config, config_g2, manifiesto: dict, filas, *, t_hat=None, cargar_modelo=None, predecir=None, device=None) -> dict:
    """Promedio de los 15 miembros y regla `>=` sobre `filas` (sin etiquetas). Verifica el SHA-256 de cada checkpoint contra el manifiesto. Biblioteca para la futura R8;
    NO hay modo CLI ni se llama desde los modos actuales."""
    exigir_filas_de_inferencia(filas)
    if len(manifiesto["miembros"]) != N_MIEMBROS:
        raise GuardError("el manifiesto debe tener 15 miembros")
    t = t_hat_exacto(t_hat if t_hat is not None else config["sistema"]["umbral_candidato"]["t_hat"])
    cargar_modelo = cargar_modelo or _cargar_modelo_g2
    dev = device if device is not None else (_dispositivo() if predecir is None else None)
    probs = {}
    for m in manifiesto["miembros"]:
        ruta = verificar_miembro_en_disco(root, m)
        pr = predecir or _predecir_g2(Path(root), config_g2, m["semilla"])
        probs[m["id_corrida"]] = np.asarray(pr(cargar_modelo(ruta, dev), filas, dev), dtype=np.float32)
    return {**agregar_y_decidir(probs, t), "miembros": sorted(probs)}


# ============================================================================ CLI (solo manifiesto y fidelidad)
def main(argv=None, *, root=None, config=None, config_g2=None, verificar=None, destino=None, origen=None, cargar_modelo=None, predecir=None, device=None) -> int:
    ap = argparse.ArgumentParser(description="sistema_final_v1 (R7): manifiesto de 15 miembros y fidelidad por miembro; candidato, NO congelado; sin modo de inferencia")
    ap.add_argument("--modo", required=True, choices=MODOS)
    ap.add_argument("--autorizacion", default="", help="CONFIRMADO:D-NNN de ejecución; sin ella no se lee ni se escribe nada")
    args = ap.parse_args(argv)
    root = Path(root or RAIZ_REPO).resolve()
    try:
        U.exigir_autorizacion(args.autorizacion)
        config = config or cargar_config(root / CONFIG_REL)
        config_g2 = config_g2 or cargar_config_g2(root / config["referencia_g2"]["config"])
        if args.modo == "manifiesto":
            r = ejecutar_manifiesto(root, config, config_g2, autorizacion=args.autorizacion, verificar=verificar, destino=destino)
        else:
            r = ejecutar_fidelidad(root, config, config_g2, autorizacion=args.autorizacion, verificar=verificar, destino=destino, origen=origen, cargar_modelo=cargar_modelo,
                                   predecir=predecir, device=device)
        print(f"{LEYENDA}\n{ESTADO}\nsalida: {r['dir']}")
        if not r["reporte_aceptado"]:
            print("SALIDA NO ACEPTADA: la integridad final difiere de la inicial; no debe usarse.", file=sys.stderr)
            return U.CODIGO_SALIDA_INTEGRIDAD_INVALIDA
        if args.modo == "fidelidad" and r["resultados"]["alerta_global"]:
            print("ALERTA de fidelidad: algún miembro excede la tolerancia de recarga de G2 (ver fidelidad_por_miembro.csv).", file=sys.stderr)
            return 2
        return 0
    except GuardError as exc:
        print(f"DETENIDO: {exc}", file=sys.stderr)
        return U.CODIGO_SALIDA_GUARDA


if __name__ == "__main__":
    sys.exit(main())
