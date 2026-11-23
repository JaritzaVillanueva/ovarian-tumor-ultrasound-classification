"""R9: INSTRUMENTO DIAGNÓSTICO D-074 de la discrepancia de la integración con desarrollo (D-072/D-073). No es código de producción de R9: no lo modifica, solo lo invoca.

Solo imágenes de DESARROLLO (K4); sin etiquetas, sin métricas, sin evaluar el ensamble; nunca test. Escribe en directorios NUEVOS bajo `results/r9_gui/diagnostico_d074/` (no toca D-072).

Miembros (fijados antes de ejecutar): los cinco que excedieron la tolerancia en D-072 (s42 f4, s43 f0, s43 f4, s44 f1, s44 f4) y un control que pasó: el primero, en orden (semilla, fold), fuera de ese
conjunto (s42 f0). Imágenes: las dos primeras por `str(image_id)` del pliegue propio de cada miembro (las mismas de D-072). Los identificadores exactos se listan con `plan` (sin modelos ni inferencia).

Etapas (cada una exige `--autorizacion CONFIRMADO:D-074`):
  plan  Sin modelos ni inferencia: miembros, imágenes, posiciones y composición de los lotes de G2, condiciones y reglas. Revisión previa de la autora.
  ab    A) repetibilidad de la ruta de la interfaz con lote 1 (3 repeticiones; comparación con OOF y con las cifras de D-072);
        B) ruta de la interfaz frente a la ruta ORIGINAL de G2 con lote 1: B1) tensores preprocesados (igualdad bit a bit, forma, tipo, pasos de memoria); B2) probabilidades por miembro.
  c     SOLO si A y B no bastan (la autora lo decide y aporta el SHA-256 del resumen de `ab`): reproduce la disposición de lotes de G2 (el lote que contiene cada imagen objetivo, mismo orden y
        composición) frente al OOF; con la ruta de G2 (C1a) y con el preprocesamiento de la interfaz apilado en ese mismo lote (C1b); barrido del tamaño de lote del objetivo (C2: 1, 2, 8, 32;
        el objetivo primero y luego los siguientes del pliegue, cíclico). `--tf32 desactivado` repite con `torch.backends.cudnn.allow_tf32 = False` (variante numérica). Alcance justificado: NO se procesa
        el pliegue completo; basta el lote de G2 que contiene a cada imagen objetivo (≤ 2 lotes de ≤ 32 imágenes por miembro) porque la disposición de un lote la determinan su composición y su tamaño.

  c1    (D-075; alcance MÍNIMO tras `ab`, con el SHA-256 de su resumen aportado por la autora) SOLO C1a y C1b con el lote histórico de G2 (el lote de 32 que contiene cada imagen objetivo). NO ejecuta
        C2 (barrido de tamaños), ni la variante TF32, ni ninguna otra. Token propio `CONFIRMADO:D-075`; salida nueva `D-075_c1`. Las etapas `plan`, `ab` y `c` no se modifican.

Reglas de interpretación (fijadas antes): B1 igualdad bit a bit ⇒ preprocesamiento idéntico (si no, defecto de R9 localizado); B2 |B − A| ≤ 1e-7 ⇒ misma ruta de inferencia con lote 1; si B ≈ A y ambos difieren
del OOF, la discrepancia es propia de la condición «lote 1» y no del código de R9; C1a = 0.0 ⇒ la disposición de lotes de G2 reproduce el OOF; C1b = 0.0 ⇒ equivalencia exacta del preprocesamiento de la
interfaz con G2 en esa disposición; C2/TF32 ⇒ dependencia de la forma del lote (y, si desaparece sin TF32, de TF32). Nada de esto cambia la tolerancia de D-072 ni valida el desempeño.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from src.r4_v2.common import RAIZ_REPO, GuardError, directorio_nuevo, escribir_json_nuevo, exigir_salida_segura, sha256_archivo
from src.r4_v2.common import cargar_config as cargar_config_g2
from src.r4_v2.particion import cargar_particion, filas_de_fold, resolver_imagen
from src.r7_v2 import sistema_final as SF
from src.r7_v2 import umbral_sistema as U
from src.r8_v2 import congelamiento as C
from src.r8_v2 import declaracion as DC
from src.gui.sistema import SHA256_DECLARACION, _predecir_torch
from src.gui.validacion_imagen import validar_y_cargar

TOKEN_DECISION = "D-074"
TOKEN_POR_ETAPA = {"plan": "D-074", "ab": "D-074", "c": "D-074", "c1": "D-075"}                          # cada etapa solo se ejecuta bajo la decisión que la autoriza
SALIDA_REL = "results/r9_gui/diagnostico_d074"
REFERENCIA_D072 = {"densenet121__s42__f4": 0.0010489821434020996, "densenet121__s43__f0": 0.00028283149003982544, "densenet121__s43__f4": 0.0006171464920043945,
                   "densenet121__s44__f1": 0.00015097856521606445, "densenet121__s44__f4": 0.0006714761257171631}        # máximos por miembro reportados por la autora (D-072/D-073)
EXCEDIDOS = tuple(REFERENCIA_D072)
N_IMAGENES, REPETICIONES, LOTE_G2, BARRIDO = 2, 3, 32, (1, 2, 8, 32)


# ============================================================================ selección (pura y determinista)
def _clave(m) -> tuple:
    return int(m["semilla"]), int(m["fold"])


def miembros_prioritarios(filas_miembros) -> list[dict]:
    """Los cinco miembros fuera de tolerancia en D-072 (en orden semilla, fold) y, al final, el control: el primer miembro, en ese orden, que no está en el conjunto."""
    por_id = {m["id_corrida"]: m for m in filas_miembros}
    faltan = [i for i in EXCEDIDOS if i not in por_id]
    if faltan:
        raise GuardError(f"miembros de D-072 ausentes del manifiesto: {faltan}")
    excedidos = sorted((por_id[i] for i in EXCEDIDOS), key=_clave)
    control = next(m for m in sorted(filas_miembros, key=_clave) if m["id_corrida"] not in EXCEDIDOS)
    return excedidos + [control]


def imagenes_objetivo(val, n: int = N_IMAGENES) -> list[dict]:
    """Las `n` primeras imágenes del pliegue propio por `str(image_id)` (la misma regla de `integracion_desarrollo`)."""
    return sorted(val, key=lambda f: str(f["image_id"]))[:n]


def posicion(val, fila) -> int:
    ids = [str(f["image_id"]) for f in val]
    return ids.index(str(fila["image_id"]))


def lote_de_g2(val, p: int, tam: int = LOTE_G2) -> tuple[list[dict], int]:
    """El lote de la disposición de G2 (`shuffle=False`, lotes consecutivos de `tam` en el orden del pliegue) que contiene la posición `p`, y su posición inicial."""
    s = (p // tam) * tam
    return list(val[s:s + tam]), s


def lote_ciclico(val, p: int, k: int) -> list[dict]:
    """Lote de tamaño `k` con el objetivo en primer lugar y a continuación los siguientes del pliegue (cíclico)."""
    return [val[(p + j) % len(val)] for j in range(k)]


# ============================================================================ contexto: llamadas inyectables (real en Falcon; sintético en las pruebas)
@dataclass
class Contexto:
    filas_miembros: list
    val_de_fold: Callable          # fold -> filas del pliegue (orden de G2)
    cargar_modelo: Callable        # fila de miembro -> modelo
    imagen: Callable               # fila -> PIL RGB (ruta de la interfaz)
    tensor_gui: Callable           # PIL -> tensor (1, 3, H, W)  (preprocesamiento de la interfaz)
    tensor_g2: Callable            # fila -> tensor (3, H, W)    (DatasetLesiones de G2)
    prob_gui: Callable             # (modelo, tensor (1,3,H,W)) -> float   (_predecir_torch)
    prob_g2: Callable              # (modelo, filas, tam_lote, semilla) -> ndarray float32   (DatasetLesiones + create_dataloader + predict)
    prob_tensores: Callable        # (modelo, [tensores (1,3,H,W)]) -> ndarray float32      (un lote apilado, mismas operaciones)
    oof: Callable                  # (fila de miembro, image_id) -> float
    entorno: Callable              # () -> dict


def _f(x) -> float:
    return float(np.float32(x))


def _tensor_info(t) -> dict:
    return {"forma": list(t.shape), "tipo": str(t.dtype), "contiguo": bool(t.is_contiguous()), "pasos": list(t.stride())}


def _max_abs(a, b) -> float:
    import torch
    if tuple(a.shape) != tuple(b.shape):
        return float("inf")                                                                              # formas distintas: se informa, no se interrumpe
    return float((a.to(torch.float64) - b.to(torch.float64)).abs().max())


# ============================================================================ plan (sin modelos ni inferencia)
def plan(ctx: Contexto) -> dict:
    miembros = []
    for m in miembros_prioritarios(ctx.filas_miembros):
        val = ctx.val_de_fold(int(m["fold"]))
        objetivos = imagenes_objetivo(val)
        imgs = []
        for f in objetivos:
            p = posicion(val, f)
            lote, s = lote_de_g2(val, p)
            imgs.append({"image_id": str(f["image_id"]), "posicion_en_pliegue": p, "lote_g2": {"inicio": s, "tamano": len(lote)}, "n_pliegue": len(val),
                         "barrido_c2": {str(k): [str(x["image_id"]) for x in lote_ciclico(val, p, k)][:3] + (["…"] if k > 3 else []) for k in BARRIDO}})
        miembros.append({"id_corrida": m["id_corrida"], "semilla": int(m["semilla"]), "fold": int(m["fold"]), "rol": "control" if m["id_corrida"] not in EXCEDIDOS else "excedio_en_D072",
                         "referencia_D072": REFERENCIA_D072.get(m["id_corrida"]), "imagenes": imgs})
    return {"miembros": miembros, "imagenes_por_miembro": N_IMAGENES, "repeticiones_A": REPETICIONES, "lote_g2": LOTE_G2, "barrido_c2": list(BARRIDO),
            "etapas": ["ab", "c"], "salidas": {"plan": "D-074_plan", "ab": "D-074_ab", "c": "D-074_c[_tf32_desactivado]", "directorio_base": SALIDA_REL},
            "condiciones": "dispositivo y precisión por defecto (FP32; TF32 en convoluciones según la bandera por defecto de PyTorch); modo eval; sin autocast; num_workers = 0 en la ruta de G2 (neutro)",
            "inferencia": False, "etiquetas_usadas": False, "metricas_calculadas": False}


# ============================================================================ etapa ab
def etapa_ab(ctx: Contexto) -> dict:
    import torch
    salida = []
    for m in miembros_prioritarios(ctx.filas_miembros):
        val, modelo = ctx.val_de_fold(int(m["fold"])), ctx.cargar_modelo(m)
        imgs = []
        for f in imagenes_objetivo(val):
            iid = str(f["image_id"])
            pil = ctx.imagen(f)
            t_gui = ctx.tensor_gui(pil)
            a = [ctx.prob_gui(modelo, t_gui) for _ in range(REPETICIONES)]                                   # A: lote 1, ruta de la interfaz
            t_g2 = ctx.tensor_g2(f).unsqueeze(0)
            b1_igual = bool(torch.equal(t_gui, t_g2))
            b = _f(ctx.prob_g2(modelo, [f], 1, int(m["semilla"]))[0])                                        # B: ruta de G2, lote 1
            oof = ctx.oof(m, iid)
            imgs.append({"image_id": iid, "p_oof": oof, "A_p": [_f(x) for x in a], "A_repetible_bit_a_bit": len({float(x) for x in a}) == 1, "A_dif_oof": abs(_f(a[0]) - oof),
                         "B1_tensores_iguales": b1_igual, "B1_dif_maxima_tensor": _max_abs(t_gui, t_g2), "B1_gui": _tensor_info(t_gui), "B1_g2": _tensor_info(t_g2),
                         "B2_p_g2_lote1": b, "B2_dif_A_B": abs(_f(a[0]) - b), "B2_dif_B_oof": abs(b - oof)})
        maxA = max(i["A_dif_oof"] for i in imgs)
        ref = REFERENCIA_D072.get(m["id_corrida"])
        salida.append({"id_corrida": m["id_corrida"], "A_dif_maxima_oof": maxA, "reproduce_D072_exactamente": (maxA == ref) if ref is not None else None, "imagenes": imgs})
    todas = [i for m in salida for i in m["imagenes"]]
    return {"etapa": "ab", "miembros": salida,
            "resumen": {"A_repetible_en_todas": all(i["A_repetible_bit_a_bit"] for i in todas), "A_dif_maxima_global": max(i["A_dif_oof"] for i in todas),
                        "D072_reproducido_en_los_cinco": all(m["reproduce_D072_exactamente"] for m in salida if m["reproduce_D072_exactamente"] is not None),
                        "B1_tensores_iguales_en_todas": all(i["B1_tensores_iguales"] for i in todas), "B2_dif_maxima_A_B": max(i["B2_dif_A_B"] for i in todas),
                        "B2_dif_maxima_B_oof": max(i["B2_dif_B_oof"] for i in todas)},
            "interpretacion": _interpretar_ab(todas)}


def _interpretar_ab(imgs) -> list[str]:
    out = []
    out.append("B1: preprocesamiento idéntico bit a bit a G2 en todas las imágenes." if all(i["B1_tensores_iguales"] for i in imgs)
               else "B1: HAY diferencias de tensor entre la interfaz y G2: defecto de preprocesamiento de R9 (ver B1_dif_maxima_tensor).")
    dab = max(i["B2_dif_A_B"] for i in imgs)
    out.append(f"B2: la ruta de la interfaz y la de G2 con lote 1 {'coinciden (≤ 1e-7)' if dab <= 1e-7 else 'DIFIEREN'} (máx. {dab!r}).")
    out.append("A: la inferencia con lote 1 es repetible bit a bit." if all(i["A_repetible_bit_a_bit"] for i in imgs) else "A: la inferencia con lote 1 NO es repetible bit a bit.")
    dbo = max(i["B2_dif_B_oof"] for i in imgs)
    out.append(f"La ruta ORIGINAL de G2 con lote 1 difiere del OOF en {dbo!r} como máximo: " +
               ("la discrepancia es propia de la condición «lote 1», no del código de R9." if (dab <= 1e-7 and dbo > 1e-4) else "ver los valores por imagen."))
    return out


# ============================================================================ etapa c
def etapa_c(ctx: Contexto, tf32_desactivado: bool = False) -> dict:
    salida = []
    for m in miembros_prioritarios(ctx.filas_miembros):
        val, modelo = ctx.val_de_fold(int(m["fold"])), ctx.cargar_modelo(m)
        imgs = []
        for f in imagenes_objetivo(val):
            iid, p = str(f["image_id"]), posicion(val, f)
            lote, s = lote_de_g2(val, p)
            oofs = np.array([ctx.oof(m, str(x["image_id"])) for x in lote], dtype=np.float64)
            c1a = np.asarray(ctx.prob_g2(modelo, lote, LOTE_G2, int(m["semilla"])), dtype=np.float32).astype(np.float64)                  # disposición exacta de G2
            c1b = np.asarray(ctx.prob_tensores(modelo, [ctx.tensor_gui(ctx.imagen(x)) for x in lote]), dtype=np.float32).astype(np.float64)  # preprocesamiento de la interfaz, mismo lote
            barrido = {}
            for k in BARRIDO:
                pk = _f(ctx.prob_g2(modelo, lote_ciclico(val, p, k), k, int(m["semilla"]))[0])
                barrido[str(k)] = {"p_objetivo": pk, "dif_oof": abs(pk - ctx.oof(m, iid))}
            imgs.append({"image_id": iid, "posicion_en_pliegue": p, "lote_g2": {"inicio": s, "tamano": len(lote)}, "C1a_dif_maxima_lote_g2_vs_oof": float(np.abs(c1a - oofs).max()),
                         "C1a_dif_objetivo": float(abs(c1a[p - s] - oofs[p - s])), "C1b_dif_maxima_lote_interfaz_vs_oof": float(np.abs(c1b - oofs).max()),
                         "C1b_dif_objetivo": float(abs(c1b[p - s] - oofs[p - s])), "C2_barrido_tamano_de_lote": barrido})
        salida.append({"id_corrida": m["id_corrida"], "imagenes": imgs})
    todas = [i for m in salida for i in m["imagenes"]]
    res = {"C1a_dif_maxima_global": max(i["C1a_dif_maxima_lote_g2_vs_oof"] for i in todas), "C1b_dif_maxima_global": max(i["C1b_dif_maxima_lote_interfaz_vs_oof"] for i in todas),
           "C2_dif_maxima_por_tamano": {str(k): max(i["C2_barrido_tamano_de_lote"][str(k)]["dif_oof"] for i in todas) for k in BARRIDO}}
    interp = [f"C1a (lote de G2 con la ruta de G2): máx. {res['C1a_dif_maxima_global']!r} " + ("= 0.0: la disposición de lotes reproduce el OOF." if res["C1a_dif_maxima_global"] == 0.0 else "≠ 0.0: la disposición de G2 NO reproduce el OOF; analizar."),
              f"C1b (lote de G2 con el preprocesamiento de la interfaz): máx. {res['C1b_dif_maxima_global']!r} " +
              ("= 0.0: equivalencia exacta del preprocesamiento de la interfaz con G2 en esa disposición." if res["C1b_dif_maxima_global"] == 0.0 else "≠ 0.0: ver detalle."),
              "C2: diferencia máxima con el OOF según el tamaño de lote del objetivo: " + ", ".join(f"k={k}: {v!r}" for k, v in res["C2_dif_maxima_por_tamano"].items()) +
              (" (con TF32 desactivado)." if tf32_desactivado else " (condición por defecto de PyTorch).")]
    return {"etapa": "c", "tf32_desactivado": tf32_desactivado, "miembros": salida, "resumen": res, "interpretacion": interp}


# ============================================================================ etapa c1 (D-075): solo C1a y C1b con el lote histórico de G2
def etapa_c1(ctx: Contexto) -> dict:
    """C1a: ruta de G2 con el lote exacto de G2; C1b: preprocesamiento de la interfaz apilado en ese mismo lote. Nada más: sin barrido de tamaños (C2) ni variante TF32."""
    salida = []
    for m in miembros_prioritarios(ctx.filas_miembros):
        val, modelo = ctx.val_de_fold(int(m["fold"])), ctx.cargar_modelo(m)
        imgs, por_lote = [], {}
        for f in imagenes_objetivo(val):
            iid, p = str(f["image_id"]), posicion(val, f)
            lote, s = lote_de_g2(val, p)
            if s not in por_lote:                                                                           # dos objetivos del mismo lote comparten el cómputo
                oofs = np.array([ctx.oof(m, str(x["image_id"])) for x in lote], dtype=np.float64)
                c1a = np.asarray(ctx.prob_g2(modelo, lote, LOTE_G2, int(m["semilla"])), dtype=np.float32).astype(np.float64)                      # disposición exacta de G2
                c1b = np.asarray(ctx.prob_tensores(modelo, [ctx.tensor_gui(ctx.imagen(x)) for x in lote]), dtype=np.float32).astype(np.float64)   # preprocesamiento de la interfaz
                por_lote[s] = (oofs, c1a, c1b)
            oofs, c1a, c1b = por_lote[s]
            imgs.append({"image_id": iid, "posicion_en_pliegue": p, "lote_g2": {"inicio": s, "tamano": len(lote)},
                         "C1a_dif_maxima_lote_g2_vs_oof": float(np.abs(c1a - oofs).max()), "C1a_dif_objetivo": float(abs(c1a[p - s] - oofs[p - s])),
                         "C1b_dif_maxima_lote_interfaz_vs_oof": float(np.abs(c1b - oofs).max()), "C1b_dif_objetivo": float(abs(c1b[p - s] - oofs[p - s])),
                         "C1b_dif_maxima_vs_C1a": float(np.abs(c1b - c1a).max())})
        salida.append({"id_corrida": m["id_corrida"], "imagenes": imgs})
    todas = [i for m in salida for i in m["imagenes"]]
    res = {"C1a_dif_maxima_global": max(i["C1a_dif_maxima_lote_g2_vs_oof"] for i in todas), "C1b_dif_maxima_global": max(i["C1b_dif_maxima_lote_interfaz_vs_oof"] for i in todas),
           "C1b_dif_maxima_vs_C1a_global": max(i["C1b_dif_maxima_vs_C1a"] for i in todas)}
    interp = [f"C1a (lote de G2 con la ruta de G2): máx. {res['C1a_dif_maxima_global']!r} " + ("= 0.0: la disposición histórica de lotes reproduce el OOF." if res["C1a_dif_maxima_global"] == 0.0 else "≠ 0.0: la disposición de G2 NO reproduce el OOF; analizar."),
              f"C1b (lote de G2 con el preprocesamiento de la interfaz): máx. {res['C1b_dif_maxima_global']!r} " +
              ("= 0.0: equivalencia exacta del preprocesamiento de la interfaz con G2 en esa disposición." if res["C1b_dif_maxima_global"] == 0.0 else "≠ 0.0: ver detalle."),
              f"C1b frente a C1a: máx. {res['C1b_dif_maxima_vs_C1a_global']!r}. No se ejecutó C2 ni la variante TF32 (alcance de D-075)."]
    return {"etapa": "c1", "miembros": salida, "resumen": res, "interpretacion": interp}


# ============================================================================ contexto real (Falcon) y CLI
def contexto_real(root, comprobar_checkpoints: bool = True) -> Contexto:
    import torch
    from src.r4_v2.dataset_lesiones import DatasetLesiones
    root = Path(root)
    DC.exigir_declaracion(root, SHA256_DECLARACION, comprobar_checkpoints=comprobar_checkpoints, comprobar_reporte=False)
    config = C.cargar_config(root / C.CONFIG_REL)
    config_g2 = cargar_config_g2(root / config["referencia_g2"]["config"])
    particion = cargar_particion(root, config_g2)
    dev, pre = SF._dispositivo(), SF.preprocesamiento()
    transformar = pre["get_val_transforms"](config_g2["receta"]["entrada"]["image_size"])
    filas_miembros = C.leer_tabla(root / C.MIEMBROS_REL)
    cache = {}

    def oof(m, iid):
        k = m["id_corrida"]
        if k not in cache:
            ids, probs = SF.leer_probabilidades_guardadas(root / config_g2["diseno"]["raiz_resultados"].strip("/") / "completa" / k / m["intento"] / "predicciones_oof.csv")
            cache[k] = {i: float(p) for i, p in zip(ids, probs)}
        return cache[k][str(iid)]

    def prob_g2(modelo, filas, tam, semilla):
        loader = pre["create_dataloader"](DatasetLesiones(root, [dict(x) for x in filas], transformar), batch_size=tam, shuffle=False, num_workers=0, seed=semilla)
        return np.asarray(pre["predict"](modelo, loader, dev)[2], dtype=np.float32)

    def prob_tensores(modelo, tensores):
        modelo.eval()
        with torch.no_grad():
            return torch.softmax(modelo(torch.cat(tensores, dim=0).to(dev)), dim=1)[:, 1].cpu().numpy().astype(np.float32)

    def entorno():
        return {**SF._versiones(), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "dispositivo": str(dev),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic}

    return Contexto(filas_miembros=filas_miembros, val_de_fold=lambda fold: filas_de_fold(particion, fold)[1],
                    cargar_modelo=lambda m: SF._cargar_modelo_g2(SF.verificar_miembro_en_disco(root, m), dev),
                    imagen=lambda f: validar_y_cargar(resolver_imagen(root, f))[0], tensor_gui=lambda pil: transformar(pil).unsqueeze(0),
                    tensor_g2=lambda f: DatasetLesiones(root, [dict(f)], transformar)[0][0], prob_gui=lambda modelo, t: _predecir_torch(modelo, t, dev),
                    prob_g2=prob_g2, prob_tensores=prob_tensores, oof=oof, entorno=entorno)


def escribir(root, nombre: str, contenido: dict, autorizacion: str) -> Path:
    destino = directorio_nuevo(exigir_salida_segura(root, Path(root) / SALIDA_REL / nombre, "results/r9_gui"))
    escribir_json_nuevo(destino / "resumen_diagnostico.json", U.a_json({**contenido, "autorizacion": autorizacion, "fecha": U.ahora(), "inferencia_de_ensamble": False,
                                                                         "etiquetas_usadas": False, "metricas_de_clasificacion_calculadas": False,
                                                                         "nota": "diagnóstico de D-074; no modifica D-072 ni el sistema congelado; sin etiquetas ni métricas"}))
    return destino


def main(argv=None, *, root=None, ctx: Contexto | None = None) -> int:
    ap = argparse.ArgumentParser(description="R9: diagnóstico numérico D-074 (solo desarrollo). Prototipo académico; sin validez diagnóstica.")
    ap.add_argument("etapa", choices=tuple(TOKEN_POR_ETAPA))
    ap.add_argument("--autorizacion", default="")
    ap.add_argument("--sufijo", default="")
    ap.add_argument("--tf32", choices=("por-defecto", "desactivado"), default="por-defecto")
    ap.add_argument("--resumen-ab", default=None, help="ruta del resumen de la etapa ab (solo etapa c)")
    ap.add_argument("--sha256-resumen-ab", default="", help="SHA-256 del resumen de ab, aportado por la autora (solo etapa c)")
    args = ap.parse_args(argv)
    root = Path(root or RAIZ_REPO).resolve()
    try:
        token = U.exigir_autorizacion(args.autorizacion)
        if token.split(":")[1] != TOKEN_POR_ETAPA[args.etapa]:
            raise GuardError(f"la etapa {args.etapa} solo se ejecuta bajo CONFIRMADO:{TOKEN_POR_ETAPA[args.etapa]}")
        if args.tf32 == "desactivado" and args.etapa != "c":
            raise GuardError("--tf32 desactivado solo se admite en la etapa c (no en c1: D-075 no autoriza la variante TF32)")
        if args.etapa in ("c", "c1"):
            ruta = Path(args.resumen_ab or "")
            if not ruta.is_file() or sha256_archivo(ruta) != args.sha256_resumen_ab or json.loads(ruta.read_text(encoding="utf-8")).get("etapa") != "ab":
                raise GuardError("la etapa c exige el resumen de la etapa ab y su SHA-256 (decisión de la autora tras revisarlo)")
        ctx = ctx or contexto_real(root, comprobar_checkpoints=(args.etapa != "plan"))          # `plan` no toca checkpoints ni modelos
        if args.etapa == "c" and args.tf32 == "desactivado":
            import torch
            torch.backends.cudnn.allow_tf32 = False
        res = {"plan": lambda: plan(ctx), "ab": lambda: etapa_ab(ctx), "c": lambda: etapa_c(ctx, args.tf32 == "desactivado"), "c1": lambda: etapa_c1(ctx)}[args.etapa]()
        res["entorno"] = ctx.entorno() if args.etapa != "plan" else None
        nombre = f"{TOKEN_POR_ETAPA[args.etapa]}_{args.etapa}" +("_tf32_desactivado" if args.tf32 == "desactivado" else "") + args.sufijo
        destino = escribir(root, nombre, res, token)
        print(json.dumps(res.get("resumen", {"plan": "ver resumen_diagnostico.json"}), indent=2, ensure_ascii=False))
        print("\n".join(res.get("interpretacion", [])))
        print(f"salida: {destino}")
        print(f"SHA-256 del resumen: {hashlib.sha256((destino / 'resumen_diagnostico.json').read_bytes()).hexdigest()}")
        return 0
    except GuardError as exc:
        print(f"DETENIDO: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
