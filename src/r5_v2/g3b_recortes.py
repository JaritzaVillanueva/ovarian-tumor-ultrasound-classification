"""Recortes ROI (B: máscara predicha OOF de G3a; C: máscara de referencia), índice verificable y resolvedor de entrada del clasificador.

* `roi_de_mascara`: UNA sola función de ROI para ambas fuentes (`src/segmentation/roi.py::extract_roi_from_mask`, sin editar; mismos
  parámetros del YAML). cv2/torch se importan de forma diferida (se pueden inyectar `extraer` y `redimensionar` en pruebas).
* `generar_recortes`: escribe en un directorio de preparación y lo renombra al terminar (nunca sobrescribe); deja un manifiesto con SHA-256 y
  una sentinela. Sin sentinela y manifiesto coincidentes, ningún consumidor usa los recortes.
* `crear_entrenador`: envuelve el entrenador de G2 SIN editarlo: sustituye `image_path/image_name` de las filas de entrenamiento y validación
  por la ruta del recorte (la guarda K4 `resolver_imagen` de G2 sigue aplicando), conservando `image_id`, `label_v2`, `fold` y el orden.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.r4_v2.common import escribir_csv_nuevo, escribir_json_nuevo, directorio_nuevo
from src.r4_v2.particion import resolver_imagen

from . import g3b_comun as G
from .comun import comparar_instantaneas, hash_de_ids
from .mascaras import areas_de_componentes, leer_mascara_binaria, mascaras_de_elegibles

GuardError = G.GuardError


# ---------------------------------------------------------------- ROI (función única para B y C)
def roi_de_mascara(imagen, mascara, roi_cfg: dict, *, extraer=None, redimensionar=None) -> dict:
    """ROI de `imagen` (PIL RGB, tamaño original) con una máscara booleana/{0,1}/{0,255} de CUALQUIER resolución.

    Devuelve `recorte` (PIL), `caja` (izq, arr, der, abj; der/abj exclusivos), `fallback` (máscara vacía o degenerada: imagen completa),
    `n_componentes` (conectividad 8, a resolución original y ANTES de `keep_largest`) y `tam` (ancho, alto). Mismos parámetros para la
    máscara predicha y la de referencia: el YAML es la única fuente.
    """
    if extraer is None or redimensionar is None:
        from src.segmentation import roi as R                                  # requiere cv2 y torch
        extraer = extraer or R.extract_roi_from_mask
        redimensionar = redimensionar or R.resize_mask_to_image
    m = (np.asarray(mascara) > 0).astype(np.uint8)
    ancho, alto = imagen.size
    n_comp = len(areas_de_componentes(np.asarray(redimensionar(m, (ancho, alto))) > 0)) if m.any() else 0
    res = extraer(imagen, m, margin_ratio=roi_cfg["margen"], keep_largest=roi_cfg["keep_largest"],
                  fallback_to_full_image=roi_cfg["fallback_a_imagen_completa"],
                  min_component_area_fraction=roi_cfg["min_component_area_fraction"])
    return {"recorte": res.roi, "caja": tuple(int(v) for v in res.bbox), "fallback": bool(res.used_fallback),
            "n_componentes": int(n_comp), "tam": (int(ancho), int(alto))}


# ---------------------------------------------------------------- generación
def _rel(root, ruta) -> str:
    return Path(ruta).resolve().relative_to(Path(root).resolve()).as_posix()


def generar_recortes(root, config, config_g3a, particion, g3a, *, aceptacion: str, verificar, extraer=None, redimensionar=None) -> dict:
    """Genera los recortes `gt` y `oof_pred` de las 820 elegibles. Exige la aceptación explícita y la consolidación de G3a verificada (`g3a`)."""
    from PIL import Image
    G.exigir_aceptacion_mascaras(aceptacion)
    inicio = G.exigir_integridad(verificar())
    raiz = G.raiz_recortes(root, config)
    en_curso = raiz.parent / (raiz.name + "__en_curso")
    if raiz.exists() or en_curso.exists():
        raise GuardError(f"los recortes ya existen o hay una generación interrumpida ({raiz.name}, {en_curso.name}); no se sobrescribe: la autora mueve o "
                         "archiva el directorio y decide")
    d3 = config_g3a["diseno"]
    rutas_gt = mascaras_de_elegibles(root, d3["imagenes_dir"], particion.filas, d3["sufijo_mascara"])
    roi_cfg = config["roi"]
    base_rel = _rel(root, raiz)
    directorio_nuevo(en_curso)
    for fuente in G.FUENTES:
        (en_curso / fuente).mkdir()
    filas_man = []
    for fila in particion.filas:
        i = fila["image_id"]
        with Image.open(resolver_imagen(root, fila)) as im:                   # K4; igual que A: .convert("RGB")
            img = im.convert("RGB")
            img.load()
        for fuente in G.FUENTES:
            if fuente == "gt":
                ruta_m, segmentador, sha_ck = rutas_gt[i], "", ""
            else:
                e = g3a["manifiesto"][i]
                ruta_m, segmentador, sha_ck = e["ruta_abs"], e["segmentador"], e["sha256_bytes_checkpoint"]
            r = roi_de_mascara(img, leer_mascara_binaria(ruta_m), roi_cfg, extraer=extraer, redimensionar=redimensionar)
            destino = en_curso / fuente / f"{i}.png"
            with open(destino, "xb") as f:                                    # PNG sin pérdida; nunca sobrescribe
                r["recorte"].convert("RGB").save(f, format="PNG")
            izq, arr, der, abj = r["caja"]
            filas_man.append({"image_id": i, "fold": fila["fold"], "fuente": fuente, "ruta_recorte": f"{base_rel}/{fuente}/{i}.png",
                              "sha256_recorte": G.sha256_archivo(destino), "izq": izq, "arr": arr, "der": der, "abj": abj,
                              "ancho_imagen": r["tam"][0], "alto_imagen": r["tam"][1], "fallback": r["fallback"], "n_componentes": r["n_componentes"],
                              "keep_largest": roi_cfg["keep_largest"], "margen": roi_cfg["margen"], "ruta_mascara_origen": _rel(root, ruta_m),
                              "sha256_mascara_origen": G.sha256_archivo(ruta_m), "segmentador": segmentador, "sha256_bytes_checkpoint_unet": sha_ck})
    nombre_man = config["recortes"]["manifiesto"]
    escribir_csv_nuevo(en_curso / nombre_man, filas_man, G.COLUMNAS_RECORTES)
    fin = verificar()
    dif = comparar_instantaneas(inicio, fin)
    if dif:
        raise GuardError(f"la integridad cambió durante la generación ({dif}); recortes en {en_curso.name} NO válidos")
    cent = {"esquema": "recortes_completo_g3b_v1/1", "protocolo_id": G.PROTOCOLO_ID, "leyenda": G.LEYENDA, "fecha": G.ahora(),
            "aceptacion_mascaras_g3a": aceptacion, "git_commit_head": inicio["git_commit_head"], "git_commit_head_g3a": g3a["git_commit_head_g3a"],
            "sha256_bytes_archivo_particion": particion.sha256_bytes_archivo, "roi": roi_cfg,
            "n_por_fuente": {f: sum(1 for r in filas_man if r["fuente"] == f) for f in G.FUENTES},
            "n_fallback_por_fuente": {f: sum(1 for r in filas_man if r["fuente"] == f and r["fallback"]) for f in G.FUENTES},
            "sha256_bytes_manifiesto_recortes": G.sha256_archivo(en_curso / nombre_man),
            "sha256_bytes_manifiesto_oof_g3a": g3a["sha256_manifiesto_oof"], "sha256_bytes_resumen_g3a": g3a["sha256_resumen"],
            "sha256_bytes_metricas_g3a": g3a["sha256_metricas_por_imagen"],
            "integridad": {rel: {c: v[c] for c in ("git_blob_id", "sha256_contenido_blob", "sha256_bytes_archivo")} for rel, v in sorted(inicio["archivos"].items())}}
    escribir_json_nuevo(en_curso / config["recortes"]["sentinela"], cent)
    os.rename(en_curso, raiz)                                                 # publicación atómica: solo ahora existe `recortes/`
    return {"dir": raiz, "n": len(filas_man), "sentinela": cent}


# ---------------------------------------------------------------- índice verificable
@dataclass
class IndiceRecortes:
    filas: dict               # (fuente, image_id) -> fila del manifiesto (valores en texto)
    sha256_manifiesto: str
    centinela: dict
    raiz: Path


def cargar_indice_recortes(root, config, particion, g3a) -> IndiceRecortes:
    """Sentinela + manifiesto coincidentes, cobertura exacta de las elegibles por fuente y coherencia con la consolidación de G3a."""
    raiz = G.raiz_recortes(root, config)
    cp, mp = raiz / config["recortes"]["sentinela"], raiz / config["recortes"]["manifiesto"]
    if not cp.is_file() or not mp.is_file():
        raise GuardError("los recortes no están generados o están incompletos (falta sentinela/manifiesto): ejecute `--modo recortes`")
    cent = json.loads(cp.read_text(encoding="utf-8"))
    sha = G.sha256_archivo(mp)
    if sha != cent["sha256_bytes_manifiesto_recortes"]:
        raise GuardError("el manifiesto de recortes no coincide con la sentinela (SHA-256)")
    if cent["sha256_bytes_manifiesto_oof_g3a"] != g3a["sha256_manifiesto_oof"] or cent["sha256_bytes_archivo_particion"] != particion.sha256_bytes_archivo:
        raise GuardError("los recortes se generaron con otra consolidación de G3a u otra partición")
    if cent["roi"] != config["roi"]:
        raise GuardError("la sección `roi` del YAML difiere de la usada para generar los recortes; no se mezclan configuraciones de ROI")
    por_id = {r["image_id"]: r for r in particion.filas}
    filas = {}
    with open(mp, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            clave = (r["fuente"], r["image_id"])
            if r["fuente"] not in G.FUENTES or clave in filas or r["image_id"] not in por_id or int(r["fold"]) != por_id[r["image_id"]]["fold"]:
                raise GuardError(f"manifiesto de recortes inválido en {clave}")
            if r["fuente"] == "oof_pred":
                e = g3a["manifiesto"][r["image_id"]]
                if r["segmentador"] != e["segmentador"] or r["sha256_bytes_checkpoint_unet"] != e["sha256_bytes_checkpoint"] \
                        or r["sha256_mascara_origen"] != e["sha256_mascara"]:
                    raise GuardError(f"{r['image_id']}: el recorte oof_pred no procede de la máscara/U-Net de su fold según G3a")
            filas[clave] = r
    for fuente in G.FUENTES:
        if {i for (s, i) in filas if s == fuente} != set(por_id):
            raise GuardError(f"el manifiesto de recortes no cubre exactamente las elegibles en la fuente {fuente}")
    return IndiceRecortes(filas=filas, sha256_manifiesto=sha, centinela=cent, raiz=raiz)


def verificar_archivos_de_recortes(root, indice: IndiceRecortes, fuente: str, ids) -> None:
    for i in ids:
        r = indice.filas[(fuente, i)]
        ruta = (Path(root) / r["ruta_recorte"]).resolve()
        if G.ruta_menciona_test(r["ruta_recorte"]) or indice.raiz.resolve() not in ruta.parents or not ruta.is_file():
            raise GuardError(f"{i}: recorte {fuente} ausente o fuera de {indice.raiz.name} (K4)")
        if G.sha256_archivo(ruta) != r["sha256_recorte"]:
            raise GuardError(f"{i}: el SHA-256 del recorte {fuente} no coincide con el manifiesto")


# ---------------------------------------------------------------- resolvedor de entrada y no-fuga
def fuentes_de(spec: dict, rol: str) -> str:
    """Fuente del ROI por rol: `train` -> `fuente_roi_entrenamiento`; `val` -> `fuente_roi_validacion` (validación/predicción del fold k)."""
    if rol not in ("train", "val"):
        raise GuardError(f"rol desconocido {rol!r}")
    return spec["fuente_roi_entrenamiento" if rol == "train" else "fuente_roi_validacion"]


def filas_con_recortes(filas, fuente: str, indice: IndiceRecortes) -> list:
    """Copia de las filas con `image_path/image_name` apuntando al recorte (mismo orden, `image_id`, `label_v2` y `fold`)."""
    salida = []
    for f in filas:
        rel = indice.filas[(fuente, f["image_id"])]["ruta_recorte"]
        salida.append({**f, "image_path": rel, "image_name": Path(rel).name})
    return salida


def verificar_fuentes_sin_fuga(fold: int, train, val, indice: IndiceRecortes, spec: dict, segmentador_semilla: int) -> dict:
    """Ninguna imagen del fold k en entrenamiento; los recortes `oof_pred` de la validación proceden del U-Net de SU fold (el k); los de
    entrenamiento con `oof_pred` (solo B_oof) proceden del U-Net del fold de cada imagen. Devuelve el registro del U-Net del fold k."""
    ti, vi = [r["image_id"] for r in train], [r["image_id"] for r in val]
    if set(ti) & set(vi) or len(set(ti)) != len(ti) or len(set(vi)) != len(vi):
        raise GuardError(f"fold {fold}: solapamiento o repetición entre entrenamiento y validación")
    if any(r["fold"] == fold for r in train) or any(r["fold"] != fold for r in val):
        raise GuardError(f"fold {fold}: fuga de información: imágenes del fold {fold} en entrenamiento o ajenas en validación")
    esperado_k = f"unet__s{segmentador_semilla}__f{fold}"
    f_val, f_tr = fuentes_de(spec, "val"), fuentes_de(spec, "train")
    unets_val = set()
    for r in val:
        e = indice.filas[(f_val, r["image_id"])]
        if f_val == "oof_pred":
            if int(e["fold"]) != fold or e["segmentador"] != esperado_k:
                raise GuardError(f"{r['image_id']}: el recorte de validación no procede del U-Net del fold {fold} ({esperado_k})")
            unets_val.add((e["segmentador"], e["sha256_bytes_checkpoint_unet"]))
    if f_tr == "oof_pred":
        for r in train:
            e = indice.filas[(f_tr, r["image_id"])]
            if int(e["fold"]) != r["fold"] or e["segmentador"] != f"unet__s{segmentador_semilla}__f{r['fold']}":
                raise GuardError(f"{r['image_id']}: el recorte de entrenamiento no procede del U-Net de su propio fold")
    if len(unets_val) > 1:
        raise GuardError(f"fold {fold}: la validación usa más de un U-Net {sorted(unets_val)}")
    seg, sha = next(iter(unets_val)) if unets_val else (None, None)
    return {"segmentador_fold_k": seg, "sha256_bytes_checkpoint_unet_fold_k": sha}


def _resumen_rol(indice, fuente, filas) -> dict:
    ent = [indice.filas[(fuente, f["image_id"])] for f in filas]
    return {"fuente": fuente, "n": len(ent), "n_fallback": sum(1 for e in ent if e["fallback"] == "True"),
            "n_multicomponente": sum(1 for e in ent if int(e["n_componentes"]) > 1),
            "sha256_ids": hash_de_ids([e["image_id"] for e in ent]),
            "sha256_recortes": G.sha256_bytes("\n".join(sorted(f"{e['image_id']}:{e['sha256_recorte']}" for e in ent)).encode("utf-8"))}


def crear_entrenador(root, config, indice: IndiceRecortes, condicion: str, *, entrenar_base=None):
    """Entrenador con la firma de `r4_v2.entrenamiento.ejecutar_corrida`: sustituye la entrada por (condición, rol) y delega en el de G2."""
    spec = config["condiciones"][condicion]
    semilla_unet = config["recortes"]["segmentador_semilla"]

    def entrenar(ctx: dict) -> dict:
        k = ctx["fold"]
        train, val = ctx["train"], ctx["val"]
        u = verificar_fuentes_sin_fuga(k, train, val, indice, spec, semilla_unet)
        f_tr, f_va = fuentes_de(spec, "train"), fuentes_de(spec, "val")
        verificar_archivos_de_recortes(root, indice, f_tr, [r["image_id"] for r in train])
        verificar_archivos_de_recortes(root, indice, f_va, [r["image_id"] for r in val])
        escribir_json_nuevo(Path(ctx["dir"]) / "procedencia_g3b.json", {
            "esquema": "procedencia_g3b_v1/1", "protocolo_id": G.PROTOCOLO_ID, "leyenda": G.LEYENDA, "condicion": condicion,
            "fuente_roi_entrenamiento": f_tr, "fuente_roi_validacion": f_va, "fold": k, "arquitectura": ctx["arq"], "semilla": ctx["semilla"],
            "huella_config_efectiva": G.huella_config_efectiva(config, condicion), "roi": config["roi"],
            "sha256_bytes_manifiesto_recortes": indice.sha256_manifiesto,
            "sha256_bytes_manifiesto_oof_g3a": indice.centinela["sha256_bytes_manifiesto_oof_g3a"],
            "git_commit_head_g3a": indice.centinela["git_commit_head_g3a"], "unet_fold_k": u,
            "entrenamiento": _resumen_rol(indice, f_tr, train), "validacion": _resumen_rol(indice, f_va, val)})
        if entrenar_base is None:
            from src.r4_v2.entrenamiento import entrenar_fold_torch as base            # torch; importación diferida
        else:
            base = entrenar_base
        return base({**ctx, "train": filas_con_recortes(train, f_tr, indice), "val": filas_con_recortes(val, f_va, indice)})

    return entrenar
