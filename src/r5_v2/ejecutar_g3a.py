"""Ejecutor de G3a `g3_segmentacion_v1` (modos `verificar_datos`, `humo`, `completa`).

Uso (solo por la autora, en Falcon; el agente que lo escribió NO lo ejecuta):

    python -B -X pycache_prefix=<directorio nuevo y vacío fuera del repositorio> -m src.r5_v2.ejecutar_g3a --modo verificar_datos
    python -B -X pycache_prefix=<...> -m src.r5_v2.ejecutar_g3a --modo humo --autorizacion-ejecucion CONFIRMADO:D-NNN
    python -B -X pycache_prefix=<...> -m src.r5_v2.ejecutar_g3a --modo completa --autorizacion-ejecucion CONFIRMADO:D-NNN

`verificar_datos` es de solo lectura (CPU, sin entrenar). `humo` y `completa` exigen la autorización explícita, árbol Git limpio y
la partición verificada (K1-K4 de G2); `completa` exige un `humo` completado previo (mismo commit, YAML y partición). No consulta el
test oficial ni las 180 normales. Partición provisional (phash_le8_v1, exploratoria, post hoc); conformidad académica pendiente.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from src.r4_v2.common import cargar_config as cargar_config_g2
from src.r4_v2.common import texto_csv
from src.r4_v2.particion import cargar_particion, resolver_imagen

from . import comun as C
from . import metricas_seg as MS
from .mascaras import COLUMNAS_POR_IMAGEN, mascaras_de_elegibles, verificar_datos
from .segmentacion import COLUMNAS_MANIFIESTO, ejecutar_fold

ARCHIVOS_CONSOLIDADOS = ("manifiesto_mascaras_oof.csv", "metricas_seg_por_imagen.csv", "resumen_metricas_seg.json")


# ---------------------------------------------------------------- humo previo y coherencia de folds ya completados
def exigir_humo_previo(raiz_humo, *, sha256_bytes_archivo_particion, sha256_contenido_blob_config, git_commit_head) -> Path:
    """`completa` exige un `humo` completado, sin invalidez, con la misma partición, YAML y commit."""
    for rc_path in sorted(Path(raiz_humo).glob("fold*/intento_*/run_config.json")):
        rc = json.loads(rc_path.read_text(encoding="utf-8"))
        if (rc.get("modo") == "humo" and rc.get("estado") == "completada" and not rc["invalida_integridad"]
                and rc["git_commit_head"] == git_commit_head and rc["sha256_bytes_archivo_particion"] == sha256_bytes_archivo_particion
                and rc["verificacion_inicio"]["archivos"][C.CONFIG_REL]["sha256_contenido_blob"] == sha256_contenido_blob_config):
            return rc_path.parent
    raise C.GuardError("el modo 'completa' exige un 'humo' completado previo (misma partición, YAML y commit) en la misma raíz de resultados")


def _coherente_con_fold_previo(dir_intento, *, sha_particion, blob_config, head, epocas, fold) -> None:
    rc = json.loads((Path(dir_intento) / "run_config.json").read_text(encoding="utf-8"))
    if (rc["git_commit_head"] != head or rc["sha256_bytes_archivo_particion"] != sha_particion or rc["epocas"] != epocas
            or rc["verificacion_inicio"]["archivos"][C.CONFIG_REL]["sha256_contenido_blob"] != blob_config):
        raise C.GuardError(f"el fold {fold} ya completado se obtuvo con otro commit, YAML, partición o número de épocas; no se mezclan folds "
                           "de configuraciones distintas (la autora decide)")


# ---------------------------------------------------------------- consolidación (solo `completa`)
def _leer_csv(path) -> list:
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def consolidar(root, config, particion, raiz_modo, modo: str):
    """Une los 5 folds completados: verifica que cada imagen se predice una vez, con el segmentador de su fold, y escribe el manifiesto
    OOF, `metricas_seg_por_imagen.csv` y el resumen descriptivo. Devuelve None si aún faltan folds. No sobrescribe: si ya existen, las
    salidas deben coincidir con lo recalculado.
    """
    manifs, metricas, rcs = {}, {}, {}
    for k in config["diseno"]["folds"]:
        d = C.corrida_completada(Path(raiz_modo) / f"fold{k}")
        if d is None:
            return None
        rcs[k] = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
        manifs[k] = _leer_csv(d / f"manifiesto_mascaras_fold{k}.csv")
        metricas[k] = _leer_csv(d / f"metricas_seg_fold{k}.csv")
    base = Path(root).resolve()
    C.verificar_prediccion_unica(particion.filas, manifs)
    referencia = rcs[config["diseno"]["folds"][0]]
    for k, rc in rcs.items():
        if rc["segmentador"] != C.segmentador_id(config, k) or any(
                rc[c] != referencia[c] for c in ("git_commit_head", "sha256_bytes_archivo_particion", "epocas")):
            raise C.GuardError(f"fold {k}: run_config incoherente con los demás folds")
        for m in manifs[k]:
            if C.sha256_archivo(base / m["ruta_mascara"]) != m["sha256_mascara"] or m["sha256_bytes_checkpoint"] != rc["checkpoint"]["sha256_bytes_archivo"] \
                    or int(m["epoca_elegida"]) != rc["seleccion"]["epoca_elegida"]:
                raise C.GuardError(f"{m['image_id']}: el manifiesto no coincide con la máscara/checkpoint/época del fold {k}")
    orden = [r["image_id"] for r in particion.filas]
    por_id_m = {m["image_id"]: m for k in manifs for m in manifs[k]}
    por_id_x = {x["image_id"]: x for k in metricas for x in metricas[k]}
    if set(por_id_m) != set(orden) or set(por_id_x) != set(orden):
        raise C.GuardError("el manifiesto o las métricas consolidadas no cubren exactamente las elegibles")
    filas_m, filas_x = [por_id_m[i] for i in orden], [por_id_x[i] for i in orden]
    texto_m, texto_x = texto_csv(filas_m, COLUMNAS_MANIFIESTO), texto_csv(filas_x, MS.COLUMNAS_METRICAS)
    resumen = {"esquema": "resumen_metricas_seg_g3_v1/1", "protocolo_id": C.PROTOCOLO_ID, "modo": modo, "leyenda": C.LEYENDA,
               "git_commit_head": referencia["git_commit_head"], "sha256_bytes_archivo_particion": referencia["sha256_bytes_archivo_particion"],
               "sha256_texto_manifiesto_oof": C.sha256_bytes(texto_m.encode("utf-8")),
               "sha256_texto_metricas_por_imagen": C.sha256_bytes(texto_x.encode("utf-8")),
               "estadisticos": MS.resumen_metricas_seg(filas_x, config["metricas_seg"]["dice_menor_que"], config["metricas_seg"]["cobertura_menor_que"])}
    destinos = [Path(raiz_modo) / n for n in ARCHIVOS_CONSOLIDADOS]
    existentes = [d.exists() for d in destinos]
    if any(existentes):
        if not all(existentes) or destinos[0].read_text(encoding="utf-8") != texto_m or destinos[1].read_text(encoding="utf-8") != texto_x:
            raise C.GuardError("las salidas consolidadas ya existen y no coinciden con lo recalculado (no se sobrescriben)")
        return {"estado": "ya_consolidado", "dir": Path(raiz_modo)}
    C.escribir_csv_nuevo(destinos[0], filas_m, COLUMNAS_MANIFIESTO)
    C.escribir_csv_nuevo(destinos[1], filas_x, MS.COLUMNAS_METRICAS)
    C.escribir_json_nuevo(destinos[2], resumen)
    return {"estado": "consolidado", "dir": Path(raiz_modo)}


# ---------------------------------------------------------------- modos
def _modo_verificar_datos(root, config, config_g2, verificar) -> int:
    particion = cargar_particion(root, config_g2)                                      # K1-K4 (sin test, sin normales)
    rep = verificar_datos(root, config, particion.filas)
    try:                                                                               # informativo: no bloquea la lectura
        integ = verificar()
        integ = {k: integ[k] for k in ("git_commit_head", "rama", "status_porcelain", "bloqueos")}
    except C.GuardError as exc:
        integ = {"error": str(exc)}
    dir_intento = C.crear_intento(C.raiz_del_modo(root, config, "verificar_datos"))
    C.escribir_csv_nuevo(dir_intento / "verificar_datos_por_imagen.csv", rep["por_imagen"], COLUMNAS_POR_IMAGEN)
    C.escribir_json_nuevo(dir_intento / "verificar_datos.json", {
        "esquema": "verificar_datos_g3_v1/1", "protocolo_id": C.PROTOCOLO_ID, "modo": "verificar_datos", "estado": rep["estado"],
        "leyenda": C.LEYENDA, "fecha": C.ahora(), "sha256_bytes_archivo_particion": particion.sha256_bytes_archivo,
        "binarizacion_referencia": config["receta"]["binarizacion_referencia"], "resumen": rep["resumen"], "bloqueos": rep["bloqueos"],
        "integridad_informativa": integ, "solo_lectura": True, "no_corrige_nada": True})
    r = rep["resumen"]
    print(f"verificar_datos: {rep['estado']} -> {dir_intento.relative_to(Path(root).resolve()).as_posix()} | elegibles {r['n_elegibles']} | "
          f"máscaras presentes {r['n_mascaras_presentes']} | ausentes {r['n_mascara_ausente']} | tamaño distinto {r['n_tamano_distinto']} | "
          f"vacías {r['n_vacias']} | valores no binarios {r['n_valores_no_binarios']} | >1 componente {r['n_multi_componente']} | "
          f"área relativa < {r['umbral_area_relativa']}: {r['n_area_relativa_menor_umbral']} | clases de valores {r['clase_de_valores']}")
    return 0 if rep["estado"] == "completada" else 2


def ejecutar(root, config, config_g2, *, modo, autorizacion="", auditoria="", humo_fold=None, epocas_humo=None, verificar=None,
             entrenar=None, predecir=None, calcular_caja=None, crear_modelo=None, device=None) -> int:
    """Núcleo con inyección de dependencias (las pruebas sustituyen verificación Git, modelo y caja ROI). Códigos: 0 ok; 2 bloqueo/fallo."""
    if modo not in C.MODOS:
        raise C.GuardError(f"modo desconocido: {modo!r} (válidos: {C.MODOS})")
    C.comprobar_coherencia_con_g2(config, config_g2)
    verificar = verificar or (lambda: C.verificar_integridad_g3a(root, config))
    if modo == "verificar_datos":
        return _modo_verificar_datos(root, config, config_g2, verificar)

    atest = {"autorizacion_ejecucion": C.exigir_autorizacion(autorizacion)}           # antes de leer nada
    if C.validar_auditoria_opcional(auditoria):
        atest["auditoria_independiente"] = auditoria
    if calcular_caja is None:                                                          # fallar ANTES de entrenar si roi.py no es importable
        try:
            importlib.import_module("src.segmentation.roi")
        except ImportError as exc:
            raise C.GuardError(f"src/segmentation/roi.py no es importable ({exc}); se requieren torch y cv2 antes de entrenar") from exc
    particion = cargar_particion(root, config_g2)                                      # K1-K4
    rep = verificar_datos(root, config, particion.filas)                               # comprobación previa, sin escribir
    if rep["estado"] != "completada":
        raise C.GuardError(f"verificar_datos con bloqueos {rep['resumen']['bloqueos_por_codigo']}; no se entrena (ejecute --modo verificar_datos)")
    rutas_imagen = {r["image_id"]: resolver_imagen(root, r) for r in particion.filas}
    rutas_mascara = mascaras_de_elegibles(root, config["diseno"]["imagenes_dir"], particion.filas, config["diseno"]["sufijo_mascara"])
    raiz_modo, raiz_datos = C.raiz_del_modo(root, config, modo), C.raiz_datos_del_modo(root, config, modo)
    d = config["diseno"]
    if modo == "humo":
        h = d["modos"]["humo"]
        folds = [h["fold"] if humo_fold is None else int(humo_fold)]
        epocas = int(h["epocas_max"] if epocas_humo is None else epocas_humo)
        if folds[0] not in d["folds"] or not h["epocas_min"] <= epocas <= h["epocas_max"]:
            raise C.GuardError(f"humo: fold en {d['folds']} y {h['epocas_min']}-{h['epocas_max']} épocas (recibido {folds[0]}, {epocas})")
    else:
        folds, epocas = list(d["folds"]), int(config["receta"]["epochs"])
    inicial = C.exigir_integridad(verificar())
    blob_config = inicial["archivos"][C.CONFIG_REL]["sha256_contenido_blob"]
    if modo == "completa":
        exigir_humo_previo(C.raiz_del_modo(root, config, "humo"), sha256_bytes_archivo_particion=particion.sha256_bytes_archivo,
                           sha256_contenido_blob_config=blob_config, git_commit_head=inicial["git_commit_head"])
    for k in folds:
        previo = C.corrida_completada(raiz_modo / f"fold{k}")
        if previo is not None:
            _coherente_con_fold_previo(previo, sha_particion=particion.sha256_bytes_archivo, blob_config=blob_config,
                                       head=inicial["git_commit_head"], epocas=epocas, fold=k)
            print(f"fold {k}: ya completado; se omite (nunca se repite un fold válido)")
            continue
        r = ejecutar_fold(root, config, particion, modo=modo, fold=k, raiz_modo=raiz_modo, raiz_datos_modo=raiz_datos,
                          rutas_imagen=rutas_imagen, rutas_mascara=rutas_mascara, verificar=verificar, atestaciones=atest, epocas=epocas,
                          entrenar=entrenar, predecir=predecir, calcular_caja=calcular_caja, crear_modelo=crear_modelo, device=device)
        print(f"fold {k} intento {r['intento']}: {r['estado']}" + (f" ({r['motivo']})" if r["motivo"] else ""))
        if r["estado"] != "completada":
            return 2                                                                   # el intento queda conservado; decide la autora
    if modo == "completa":
        res = consolidar(root, config, particion, raiz_modo, modo)
        print(f"consolidación: {res['estado'] if res else 'incompleta'} -> {raiz_modo.relative_to(Path(root).resolve()).as_posix()}")
    return 0


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="G3a g3_segmentacion_v1 (mapeo v2, phash_le8_v1; partición provisional; conformidad académica pendiente)")
    p.add_argument("--modo", required=True, choices=C.MODOS)
    p.add_argument("--autorizacion-ejecucion", default="", help="humo y completa: CONFIRMADO:AUT-NNN o CONFIRMADO:D-NNN (mismo formato que G2)")
    p.add_argument("--auditoria-independiente", default="", help="OPCIONAL: CONFIRMADO:AUD-NNN (se registra; no habilita nada)")
    p.add_argument("--humo-fold", type=int, default=None, help="fold del humo (por defecto el de la configuración: 0)")
    p.add_argument("--epocas-humo", type=int, default=None, help="épocas del humo (por defecto el máximo de la configuración: 2)")
    return p


def main(argv=None, *, root=None, config=None, config_g2=None, **inyecciones) -> int:
    args = construir_parser().parse_args(argv)
    root = Path(root or C.RAIZ_REPO).resolve()
    try:
        config = config or C.cargar_config_g3a(root / C.CONFIG_REL)
        config_g2 = config_g2 or cargar_config_g2(root / config["referencia_g2"]["config"])
        return ejecutar(root, config, config_g2, modo=args.modo, autorizacion=args.autorizacion_ejecucion, auditoria=args.auditoria_independiente,
                        humo_fold=args.humo_fold, epocas_humo=args.epocas_humo, **inyecciones)
    except C.GuardError as exc:
        print(f"DETENIDO: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
