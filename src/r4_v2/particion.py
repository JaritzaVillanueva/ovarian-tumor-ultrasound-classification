"""Cargador de la partición `phash_le8_v1` y verificaciones K1-K4 (protocolo §4.1-§4.4).

Solo lee el CSV de la configuración tras comprobar el SHA-256 de sus bytes (K1). `label_v2` se recalcula SIEMPRE desde
`original_class` (K2). Los recuentos 164 / 8-8-8-7-7 / 69 / 51 de `observaciones_reportadas_no_criterio` se registran
pero NO son criterios de aceptación. Nada de aquí abre imágenes ni el test.
"""

from __future__ import annotations

import csv
import io
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .common import GuardError, exigir_ruta_no_prohibida, ruta_menciona_test, sha256_bytes


@dataclass
class Particion:
    filas: list                      # dicts: image_id, image_name, original_class(int), label_v2(int), e1_split, image_path, group_id, fold(int)
    sha256_bytes_archivo: str
    tamano_bytes: int
    observaciones: dict = field(default_factory=dict)   # registradas, NO criterios

    def fold_de(self, image_id):
        return self._indice()[image_id]["fold"]

    def _indice(self):
        if not hasattr(self, "_idx"):
            self._idx = {r["image_id"]: r for r in self.filas}
        return self._idx


def mapa_etiquetas(config) -> dict:
    """original_class -> label_v2, derivado de la configuración congelada (la 5 no figura)."""
    m = config["mapeo_etiquetas"]
    mapa = {int(c): 0 for c in m["clase_0_lesion_no_maligna"]}
    mapa.update({int(c): 1 for c in m["clase_1_lesion_maligna"]})
    if set(mapa) & {int(c) for c in m["excluida_ovario_normal"]}:
        raise GuardError("K2: la configuración mezcla la categoría excluida con las clases del mapeo")
    return mapa


def _nombre_de_imagen_seguro(nombre: str) -> bool:
    norm = nombre.replace("\\", "/")
    return bool(nombre) and norm == Path(norm).name and ".." not in nombre and ":" not in nombre \
        and not nombre.startswith(".") and not ruta_menciona_test(nombre)


def cargar_particion(root, config) -> Particion:
    """K1-K4 sobre el CSV de `config['particion']['csv']`, relativo a `root`. Lanza GuardError ante cualquier fallo."""
    cfg = config["particion"]
    ruta_rel = cfg["csv"]["ruta"]
    exigir_ruta_no_prohibida(ruta_rel)                                                    # K4 sobre la ruta del CSV
    ruta = Path(root) / ruta_rel
    if not ruta.is_file():
        raise GuardError(f"La partición no está disponible en {ruta_rel} (se transfiere por canal privado; protocolo §4.3)")

    datos = ruta.read_bytes()                                                             # un único acceso: hash y análisis
    sha = sha256_bytes(datos)
    if sha != cfg["csv"]["sha256_bytes_archivo"]:                                         # K1
        raise GuardError(f"K1: sha256_bytes_archivo {sha} no coincide con el esperado "
                         f"{cfg['csv']['sha256_bytes_archivo']}; no se usa la partición")

    lector = csv.DictReader(io.StringIO(datos.decode("utf-8"), newline=""))
    columnas = list(lector.fieldnames or [])
    rechazadas = [c for c in cfg["columnas_rechazadas"] if c in columnas]
    if rechazadas:                                                                        # K4: mapeo histórico
        raise GuardError(f"K4: columnas del mapeo histórico presentes: {rechazadas}")
    faltan = [c for c in cfg["columnas_requeridas"] if c not in columnas]
    if faltan:
        raise GuardError(f"K2: faltan columnas requeridas: {faltan}")

    mapa = mapa_etiquetas(config)
    excluidas = {int(c) for c in config["mapeo_etiquetas"]["excluida_ovario_normal"]}
    permitidas = [int(c) for c in cfg["verificaciones_bloqueantes"]["categorias_permitidas"]]
    if set(permitidas) != set(mapa):
        raise GuardError("K2: categorias_permitidas no coincide con el mapeo de la configuración")
    carpetas = ("train", "validation")
    imagenes_dir = config["diseno"]["imagenes_dir"].strip("/")
    exigir_ruta_no_prohibida(imagenes_dir)
    folds_validos = [int(f) for f in cfg["verificaciones_bloqueantes"]["folds"]]

    filas = []
    for n, bruta in enumerate(lector, start=2):
        f = {k: (bruta.get(k) or "").strip() for k in cfg["columnas_requeridas"]}
        quien = f"fila {n} (image_id {f['image_id']!r})"
        if not f["image_id"]:
            raise GuardError(f"K2: {quien}: image_id vacío")
        try:
            f["original_class"] = int(f["original_class"])
            f["label_v2"] = int(f["label_v2"])
            f["fold"] = int(f["fold"])
        except ValueError:
            raise GuardError(f"K2/K3: {quien}: original_class, label_v2 o fold no enteros") from None
        if f["original_class"] in excluidas:
            raise GuardError(f"K2: {quien}: categoría 5 (ovario normal) presente en la partición")
        if f["original_class"] not in mapa:
            raise GuardError(f"K2: {quien}: categoría desconocida {f['original_class']}")
        if f["label_v2"] != mapa[f["original_class"]]:
            raise GuardError(f"K2: {quien}: label_v2 {f['label_v2']} no coincide con el mapeo desde original_class")
        if f["fold"] not in folds_validos:
            raise GuardError(f"K3: {quien}: fold {f['fold']} fuera de {folds_validos}")
        if not f["group_id"]:
            raise GuardError(f"K3: {quien}: group_id vacío")
        # K4: ubicación estricta dentro de train/images o validation/images de desarrollo
        if f["e1_split"] not in carpetas:
            raise GuardError(f"K4: {quien}: e1_split {f['e1_split']!r} fuera de {list(carpetas)}")
        if not _nombre_de_imagen_seguro(f["image_name"]):
            raise GuardError(f"K4: {quien}: image_name no seguro {f['image_name']!r}")
        esperado = f"{imagenes_dir}/{f['e1_split']}/images/{f['image_name']}"
        if f["image_path"].replace("\\", "/") != esperado:
            raise GuardError(f"K4: {quien}: image_path {f['image_path']!r} distinto de {esperado!r}")
        if ruta_menciona_test(f["image_path"]):
            raise GuardError(f"K4: {quien}: image_path con componente 'test'")
        filas.append(f)

    _verificar_conjunto(filas, cfg["verificaciones_bloqueantes"])
    return Particion(filas=filas, sha256_bytes_archivo=sha, tamano_bytes=len(datos),
                     observaciones=observaciones(filas, folds_validos))


def _verificar_conjunto(filas, vb) -> None:
    """K2 (recuentos, unicidad) y K3 (grupos, folds, train/validación) sobre todas las filas.

    M-1 (decisión del autor): para la partición FIJA `phash_le8_v1`, 820 = 782 + 38 son invariantes de integridad BLOQUEANTES
    (se toman de `verificaciones_bloqueantes` de la configuración; no son una exigencia general para otros conjuntos). El protocolo
    §4.4 (L128) los describe como «se registran y reportan sin bloquear», mientras la configuración los lista como bloqueantes:
    discrepancia documentada en Reportes/asesorias/propuesta_resolucion_M1_recuentos_4_4.md; el código sigue la configuración.
    """
    if len(filas) != int(vb["n_filas"]):
        raise GuardError(f"K2: {len(filas)} filas (esperado {vb['n_filas']})")
    ids = [r["image_id"] for r in filas]
    if len(set(ids)) != len(ids):
        raise GuardError("K2: image_id repetidos")
    if len({r["image_path"] for r in filas}) != len(filas):
        raise GuardError("K2: image_path repetidos (una misma imagen aparece más de una vez)")
    n_neg = sum(1 for r in filas if r["label_v2"] == 0)
    n_pos = sum(1 for r in filas if r["label_v2"] == 1)
    if (n_neg, n_pos) != (int(vb["n_lesion_no_maligna"]), int(vb["n_lesion_maligna"])):
        raise GuardError(f"K2: {n_neg} no malignas / {n_pos} malignas (esperado "
                         f"{vb['n_lesion_no_maligna']} / {vb['n_lesion_maligna']})")

    fold_de_grupo, clase_de_grupo = {}, {}
    for r in filas:
        g = r["group_id"]
        if fold_de_grupo.setdefault(g, r["fold"]) != r["fold"]:
            raise GuardError(f"K3: el group_id {g!r} aparece en más de un fold")
        if clase_de_grupo.setdefault(g, r["label_v2"]) != r["label_v2"]:
            raise GuardError(f"K3: el group_id {g!r} mezcla label_v2 0 y 1 (grupo mixto)")

    todos = set(ids)
    minimo = int(vb["minimo_positivas_por_fold"])
    for k in vb["folds"]:
        val = {r["image_id"] for r in filas if r["fold"] == k}
        train = {r["image_id"] for r in filas if r["fold"] != k}
        if not val:
            raise GuardError(f"K3: el fold {k} está vacío")
        if val & train or (val | train) != todos:
            raise GuardError(f"K3: train y validación del fold {k} no son disjuntos o su unión no es el total")
        pos = sum(1 for r in filas if r["fold"] == k and r["label_v2"] == 1)
        if pos < minimo:
            raise GuardError(f"K3: el fold {k} tiene {pos} positivas (< {minimo}); regla operativa heredada")


def observaciones(filas, folds) -> dict:
    """Se REGISTRAN sin bloquear (tamaños de fold, positivas, grupos); no son criterios de aceptación."""
    tam = Counter(r["fold"] for r in filas)
    pos = Counter(r["fold"] for r in filas if r["label_v2"] == 1)
    grupos = defaultdict(list)
    for r in filas:
        grupos[r["group_id"]].append(r["image_id"])
    multi = [g for g, v in grupos.items() if len(v) > 1]
    return {
        "tamano_de_fold": [tam[k] for k in folds],
        "positivas_por_fold": [pos[k] for k in folds],
        "n_grupos": len(grupos),
        "grupos_multiimagen": len(multi),
        "imagenes_en_grupos_multiimagen": sum(len(grupos[g]) for g in multi),
        "grupo_mayor": max(len(v) for v in grupos.values()) if grupos else 0,
    }


def filas_de_fold(particion: Particion, fold: int):
    """(train, validación) del fold: validación = fold retenido; entrenamiento = los otros cuatro."""
    train = [r for r in particion.filas if r["fold"] != fold]
    val = [r for r in particion.filas if r["fold"] == fold]
    if not val or not train:
        raise GuardError(f"K3: el fold {fold} no produce una partición train/validación válida")
    return train, val


def pesos_de_clase(etiquetas, alpha: float = 1.0) -> dict:
    """peso_c = total / (2 * n_c) (protocolo §6.1; α = 1.0, idéntico a classification.utils.compute_class_weights)."""
    if alpha != 1.0:
        raise GuardError("G2 solo usa α = 1.0; las variantes son técnicas de G4")
    cuentas = Counter(int(e) for e in etiquetas)
    if set(cuentas) != {0, 1}:
        raise GuardError("K3: el entrenamiento del fold debe tener ambas clases para calcular los pesos")
    total = sum(cuentas.values())
    return {c: total / (2 * n) for c, n in sorted(cuentas.items())}


def resolver_imagen(root, fila) -> Path:
    """Ruta resuelta de una imagen de desarrollo, comprobada contra su directorio autorizado (K4).

    Sigue enlaces simbólicos: la ruta resuelta debe seguir dentro de `<imagenes_dir>/<e1_split>/images`. No abre el archivo.
    """
    raiz = Path(root).resolve()
    nominal = raiz / Path(fila["image_path"]).parent
    permitido = nominal.resolve()
    if os.path.normcase(str(permitido)) != os.path.normcase(str(nominal)):
        raise GuardError(f"K4: el directorio autorizado está redirigido (enlace): {fila['image_path']}")
    resuelta = (permitido / fila["image_name"]).resolve()
    if os.path.normcase(str(resuelta.parent)) != os.path.normcase(str(permitido)):
        raise GuardError(f"K4: la ruta resuelta de {fila['image_name']!r} sale del directorio autorizado")
    if ruta_menciona_test(resuelta.relative_to(raiz)):
        raise GuardError("K4: ruta de test detectada")
    if not resuelta.is_file():
        raise GuardError(f"Imagen de desarrollo no encontrada: {fila['image_path']}")
    return resuelta
