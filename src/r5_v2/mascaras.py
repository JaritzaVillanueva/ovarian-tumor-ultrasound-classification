"""Máscaras de referencia: lectura, binarización explícita `> 0`, componentes conexas y `verificar_datos` (solo lectura, sin torch).

Convención de `src/segmentation/dataset.py` (no se importa: arrastra torch y carga el directorio completo, normales incluidas):
imagen `<id>.<ext>` y máscara `<id>_binary.<ext>` en `<imagenes_dir>/<train|validation>/{images,masks}/`. Solo se abren las máscaras
de las imágenes elegibles de la partición; las demás se cuentan por nombre y NO se abren. Nunca se tocan rutas `test`.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import numpy as np

from src.r4_v2.particion import resolver_imagen

from .comun import GuardError, exigir_ruta_no_prohibida, ruta_menciona_test

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}      # como src/segmentation/dataset.py
CARPETAS_DESARROLLO = ("train", "validation")
CODIGOS_BLOQUEANTES = ("mascara_ausente", "mascara_ilegible", "tamano_distinto", "mascara_vacia", "valores_no_binarios",
                       "imagen_ausente", "imagen_ilegible", "mascara_duplicada", "id_distinto_de_nombre")


# ---------------------------------------------------------------- lectura y binarización
def binarizar(arr) -> np.ndarray:
    """Binarización explícita de la máscara de referencia: `> 0` (el dataset histórico no binariza)."""
    return np.asarray(arr) > 0


def clase_de_valores(valores) -> str:
    """'01' si los valores son subconjunto de {0,1}; '0_255' si de {0,255}; 'otros' en cualquier otro caso."""
    s = {int(v) for v in valores}
    if s <= {0, 1}:
        return "01"
    if s <= {0, 255}:
        return "0_255"
    return "otros"


def leer_mascara_cruda(path):
    """(arreglo uint8 en escala de grises 'L' como el dataset, modo PIL original, (ancho, alto))."""
    from PIL import Image
    with Image.open(path) as im:
        modo, tam = im.mode, im.size
        arr = np.asarray(im.convert("L"))
    return arr, modo, tam


def leer_mascara_binaria(path, tam=None) -> np.ndarray:
    """Máscara booleana `> 0`. Con `tam=(ancho, alto)`: reescalado con vecino más próximo ANTES de binarizar (como el dataset)."""
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("L")
        if tam is not None and tuple(tam) != im.size:
            im = im.resize(tuple(tam), resample=Image.Resampling.NEAREST)
        return binarizar(np.asarray(im))


def areas_de_componentes(mask) -> list:
    """Áreas (px) de las componentes conexas con conectividad 8 (como roi.py), de mayor a menor. Sin cv2: etiquetado por tramos."""
    m = np.asarray(mask).astype(bool)
    if m.ndim != 2:
        raise GuardError(f"se esperaba una máscara 2D (shape={m.shape})")
    padre, area = [], []

    def raiz(a):
        while padre[a] != a:
            padre[a] = padre[padre[a]]
            a = padre[a]
        return a

    previos = []
    for fila in m:
        if not fila.any():
            previos = []
            continue
        dx = np.diff(np.concatenate(([0], fila.astype(np.int8), [0])))
        inicios, fines = np.flatnonzero(dx == 1).tolist(), (np.flatnonzero(dx == -1) - 1).tolist()
        actuales = []
        for s, e in zip(inicios, fines):
            idx = len(padre)
            padre.append(idx)
            area.append(e - s + 1)
            actuales.append((s, e, idx))
            for ps, pe, pidx in previos:
                if ps > e + 1:
                    break
                if pe >= s - 1:
                    a, b = raiz(idx), raiz(pidx)
                    if a != b:
                        padre[b] = a
        previos = actuales
    total = Counter()
    for i, a in enumerate(area):
        total[raiz(i)] += a
    return sorted(total.values(), reverse=True)


# ---------------------------------------------------------------- directorios de máscaras (K4)
def dir_mascaras(root, imagenes_dir: str, split: str, subdir: str = "masks") -> Path:
    """Directorio de máscaras de `train` o `validation`; rechaza `test`, enlaces que lo redirijan y cualquier otro split."""
    if split not in CARPETAS_DESARROLLO:
        raise GuardError(f"K4: e1_split {split!r} fuera de {list(CARPETAS_DESARROLLO)}")
    rel = f"{imagenes_dir.strip('/')}/{split}/{subdir}"
    exigir_ruta_no_prohibida(rel)
    raiz = Path(root).resolve()
    nominal = raiz / rel
    resuelta = nominal.resolve()
    if os.path.normcase(str(resuelta)) != os.path.normcase(str(nominal)):
        raise GuardError(f"K4: el directorio de máscaras está redirigido (enlace): {rel}")
    if ruta_menciona_test(resuelta.relative_to(raiz)):
        raise GuardError("K4: ruta de test detectada")
    if not resuelta.is_dir():
        raise GuardError(f"directorio de máscaras de desarrollo no encontrado: {rel}")
    return resuelta


def indexar_mascaras(directorio, sufijo: str = "_binary"):
    """(id -> ruta, ids duplicados, nombres que no siguen la convención). Solo lista nombres; no abre ningún archivo."""
    indice, duplicados, otros = {}, [], []
    for p in sorted(Path(directorio).iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTENSIONS or not p.stem.endswith(sufijo):
            otros.append(p.name)
            continue
        i = p.stem[: -len(sufijo)]
        if i in indice:
            duplicados.append(i)
        else:
            indice[i] = p
    return indice, duplicados, otros


def ruta_mascara_segura(directorio, path) -> Path:
    """La ruta resuelta de la máscara (enlaces incluidos) debe seguir dentro del directorio de desarrollo."""
    resuelta = Path(path).resolve()
    if os.path.normcase(str(resuelta.parent)) != os.path.normcase(str(Path(directorio).resolve())):
        raise GuardError(f"K4: la ruta resuelta de {Path(path).name!r} sale del directorio autorizado")
    return resuelta


def mascaras_de_elegibles(root, imagenes_dir: str, filas, sufijo: str = "_binary") -> dict:
    """image_id -> Path de la máscara, EXIGIENDO que existan todas (entrenamiento). Para el informe tolerante: `verificar_datos`."""
    indices, rutas = {}, {}
    for r in filas:
        d = dir_mascaras(root, imagenes_dir, r["e1_split"])
        if r["e1_split"] not in indices:
            indices[r["e1_split"]] = (d, indexar_mascaras(d, sufijo)[0])
        base, indice = indices[r["e1_split"]]
        p = indice.get(Path(r["image_name"]).stem)
        if p is None:
            raise GuardError(f"máscara de referencia ausente para image_id {r['image_id']!r}")
        rutas[r["image_id"]] = ruta_mascara_segura(base, p)
    return rutas


# ---------------------------------------------------------------- verificar_datos
COLUMNAS_POR_IMAGEN = ["image_id", "fold", "e1_split", "original_class", "label_v2", "mascara", "mascara_presente", "ancho_imagen",
                       "alto_imagen", "ancho_mascara", "alto_mascara", "mismo_tamano", "modo_pil", "valores_distintos",
                       "clase_valores", "area_px", "area_relativa", "vacia", "n_componentes", "multi_componente",
                       "area_relativa_menor_umbral", "bloqueos"]


def verificar_datos(root, config, filas) -> dict:
    """Comprobación de solo lectura de las máscaras de las elegibles. NO corrige nada; reporta todo.

    Bloquean: máscara ausente/ilegible, tamaño distinto de la imagen, máscara vacía, valores que no sean {0,1} ni {0,255}, imagen
    ausente/ilegible, ids duplicados, image_id distinto del nombre de archivo. Solo se reportan: modo/valores, más de una
    componente (conectividad 8, resolución original) y área relativa < `verificacion.area_relativa_minima`.
    """
    from PIL import Image
    d = config["diseno"]
    sufijo, umbral = d["sufijo_mascara"], float(config["verificacion"]["area_relativa_minima"])
    if any(r["original_class"] == 5 for r in filas):
        raise GuardError("K2: categoría 5 (ovario normal) presente entre las elegibles")
    indices, extras, duplicados_dir = {}, {}, {}
    for split in sorted({r["e1_split"] for r in filas}):
        base = dir_mascaras(root, d["imagenes_dir"], split)
        indice, dup, otros = indexar_mascaras(base, sufijo)
        indices[split], duplicados_dir[split] = (base, indice), dup
        extras[split] = {"otros_nombres": len(otros), "total_listadas": len(indice)}
    elegibles = {s: {Path(r["image_name"]).stem for r in filas if r["e1_split"] == s} for s in indices}

    por_imagen, bloqueos = [], []
    valores_globales, modos, clases = set(), Counter(), Counter()
    for r in filas:
        i, split = r["image_id"], r["e1_split"]
        stem = Path(r["image_name"]).stem
        fila = {c: "" for c in COLUMNAS_POR_IMAGEN}
        fila.update(image_id=i, fold=r["fold"], e1_split=split, original_class=r["original_class"], label_v2=r["label_v2"],
                    mascara_presente=False)
        codigos = []

        def bloquear(codigo, detalle):
            codigos.append(codigo)
            bloqueos.append({"codigo": codigo, "image_id": i, "detalle": detalle})

        if stem != i:
            bloquear("id_distinto_de_nombre", f"image_id {i!r} != nombre {stem!r}")
        if stem in duplicados_dir[split]:
            bloquear("mascara_duplicada", stem)
        tam_img = None
        try:
            ruta_img = resolver_imagen(root, r)
            with Image.open(ruta_img) as im:
                tam_img = im.size
            fila["ancho_imagen"], fila["alto_imagen"] = tam_img
        except GuardError as exc:
            if str(exc).startswith("K4"):
                raise
            bloquear("imagen_ausente", str(exc))
        except (OSError, ValueError) as exc:
            bloquear("imagen_ilegible", f"{type(exc).__name__}: {exc}")
        base, indice = indices[split]
        ruta = indice.get(stem)
        if ruta is None:
            bloquear("mascara_ausente", f"{stem}{sufijo}.* en {d['imagenes_dir']}/{split}/masks")
        else:
            fila["mascara"] = f"{d['imagenes_dir'].strip('/')}/{split}/masks/{ruta.name}"
            try:
                arr, modo, tam = leer_mascara_cruda(ruta_mascara_segura(base, ruta))
            except GuardError:
                raise
            except (OSError, ValueError) as exc:
                bloquear("mascara_ilegible", f"{type(exc).__name__}: {exc}")
            else:
                fila["mascara_presente"] = True
                fila["ancho_mascara"], fila["alto_mascara"], fila["modo_pil"] = tam[0], tam[1], modo
                modos[modo] += 1
                valores = np.unique(arr).tolist()
                valores_globales.update(valores)
                fila["valores_distintos"] = "|".join(str(v) for v in valores)
                fila["clase_valores"] = clase_de_valores(valores)
                clases[fila["clase_valores"]] += 1
                if tam_img is not None:
                    fila["mismo_tamano"] = tam == tam_img
                    if tam != tam_img:
                        bloquear("tamano_distinto", f"mascara {tam} != imagen {tam_img}")
                if fila["clase_valores"] == "otros":
                    bloquear("valores_no_binarios", f"valores {valores}")
                m = binarizar(arr)
                area = int(m.sum())
                fila["area_px"], fila["area_relativa"] = area, repr(area / m.size)
                fila["vacia"] = area == 0
                if area == 0:
                    bloquear("mascara_vacia", "sin píxeles > 0")
                else:
                    n = len(areas_de_componentes(m))
                    fila["n_componentes"] = n
                    fila["multi_componente"] = n > 1
                    fila["area_relativa_menor_umbral"] = (area / m.size) < umbral
        fila["bloqueos"] = "|".join(codigos)
        por_imagen.append(fila)

    ids_elegibles_por_split = {s: len(v) for s, v in elegibles.items()}
    no_elegibles = {s: len(set(indices[s][1]) - elegibles[s]) for s in indices}
    por_codigo = Counter(b["codigo"] for b in bloqueos)
    resumen = {
        "n_elegibles": len(filas),
        "n_mascaras_presentes": sum(1 for f in por_imagen if f["mascara_presente"]),
        "n_mascara_ausente": por_codigo["mascara_ausente"], "n_tamano_distinto": por_codigo["tamano_distinto"],
        "n_vacias": por_codigo["mascara_vacia"], "n_valores_no_binarios": por_codigo["valores_no_binarios"],
        "n_multi_componente": sum(1 for f in por_imagen if f["multi_componente"] is True),
        "n_area_relativa_menor_umbral": sum(1 for f in por_imagen if f["area_relativa_menor_umbral"] is True),
        "umbral_area_relativa": umbral, "clase_de_valores": dict(clases), "modos_pil": dict(modos),
        "valores_distintos_globales": sorted(int(v) for v in valores_globales),
        "elegibles_por_split": ids_elegibles_por_split,
        "mascaras_sin_imagen_elegible_no_abiertas": no_elegibles,
        "nombres_fuera_de_convencion_ignorados": {s: e["otros_nombres"] for s, e in extras.items()},
        "bloqueos_por_codigo": dict(por_codigo),
    }
    return {"estado": "bloqueada" if bloqueos else "completada", "resumen": resumen, "bloqueos": bloqueos, "por_imagen": por_imagen}
