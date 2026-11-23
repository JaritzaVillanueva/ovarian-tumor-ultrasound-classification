"""R8: PRECOMPROBACIÓN ESTRUCTURAL ACOTADA del test (solo lectura; la ejecuta la autora en Falcon con una autorización específica).

Comprueba, sobre el manifiesto real, los tres supuestos del ejecutor `evaluacion_final` antes de gastar la ejecución única:
  1. `source_split == "test"` en TODAS las filas con `experimental_split == "test"`;
  2. los recuentos agregados del mapeo v2 coinciden con los documentados (367 negativas elegibles, 15 positivas elegibles, 87 normales excluidas), sin categorías desconocidas;
  3. cada imagen ELEGIBLE existe en `data/processed/MMOTU_OTU_2D_experimental/test/images/` (solo `is_file`: no se abre ni se decodifica ninguna imagen).
Además: columnas requeridas, `image_id`/`image_name` únicos y seguros.

Alcance: NO infiere, NO calcula métricas, NO crea el bloqueo de ejecución única, NO escribe en `evaluacion_final_v1`. Lee el manifiesto una vez; el informe contiene SOLO recuentos agregados, booleanos y
(hasta cinco) nombres de archivos faltantes: ninguna fila, clase por imagen ni imagen. Escribe un resumen en un directorio nuevo bajo `results/r8_v2/phash_le8_v1/precomprobacion_estructural/`.

CLI (solo la autora): python -B -m src.r8_v2.precomprobacion --autorizacion CONFIRMADO:D-NNN --sha256-protocolo <64 hex> [--sufijo _2]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

from src.r4_v2.common import RAIZ_REPO, GuardError, directorio_nuevo, escribir_json_nuevo, exigir_salida_segura, sha256_bytes
from src.r7_v2 import umbral_sistema as U
from src.r8_v2 import evaluacion_final as E

SALIDA_REL = "results/r8_v2/phash_le8_v1/precomprobacion_estructural"
MAX_NOMBRES_FALTANTES = 5


def analizar(root) -> dict:
    """Análisis puro de solo lectura. No lanza ante incumplimientos: los informa (para diagnosticar todo de una vez)."""
    root = Path(root)
    ruta = root / E.MANIFIESTO_TEST_REL
    if not ruta.is_file():
        return {"manifiesto_presente": False, "apto_para_ejecutar": False}
    datos = ruta.read_bytes()                                                              # una única lectura del manifiesto
    lector = csv.DictReader(io.StringIO(datos.decode("utf-8"), newline=""))
    faltan = [c for c in E.COLUMNAS_MANIFIESTO if c not in (lector.fieldnames or [])]
    if faltan:
        return {"manifiesto_presente": True, "sha256_bytes_archivo_manifiesto": sha256_bytes(datos), "columnas_faltantes": faltan, "apto_para_ejecutar": False}
    mapa = {int(c): 0 for c in E.DATOS["mapeo"]["clase_0_lesion_no_maligna"]}
    mapa.update({int(c): 1 for c in E.DATOS["mapeo"]["clase_1_lesion_maligna"]})
    excluidas = {int(c) for c in E.DATOS["mapeo"]["excluida_ovario_normal"]}
    n_test, source, neg, pos, norm, otros, ids, nombres, dup, inseguros, elegibles = 0, Counter(), 0, 0, 0, 0, set(), set(), 0, 0, []
    for bruta in lector:
        if (bruta["experimental_split"] or "").strip().lower() != E.DATOS["subconjunto"]:
            continue
        n_test += 1
        source[(bruta["source_split"] or "").strip().lower()] += 1
        iid, nombre = (bruta["image_id"] or "").strip(), (bruta["image_name"] or "").strip()
        if not iid or iid in ids or nombre in nombres:
            dup += 1
        ids.add(iid)
        nombres.add(nombre)
        if not E._nombre_seguro(nombre):
            inseguros += 1
        try:
            oc = int(str(bruta["original_class"]).strip())
        except ValueError:
            otros += 1
            continue
        if oc in excluidas:
            norm += 1
        elif oc not in mapa:
            otros += 1
        else:
            pos, neg = pos + (mapa[oc] == 1), neg + (mapa[oc] == 0)
            elegibles.append({"image_id": iid, "image_name": nombre, "image_path": f"{E.IMAGENES_DIR}/{E.DATOS['subconjunto']}/images/{nombre}"})
    faltantes = []
    for f in elegibles:
        try:
            E.resolver_imagen_prueba(root, f)                                              # solo is_file y rutas restringidas; no abre la imagen
        except GuardError:
            faltantes.append(f["image_name"])
    recuentos = {"elegibles_clase_0": neg, "elegibles_clase_1": pos, "normales_excluidas": norm}
    s1 = n_test > 0 and set(source) == {E.DATOS["subconjunto"]}
    s2 = recuentos == E.DATOS["recuentos_documentados"] and otros == 0
    s3 = bool(elegibles) and not faltantes and (root / E.IMAGENES_DIR / E.DATOS["subconjunto"] / "images").is_dir()
    basicos = dup == 0 and inseguros == 0
    return {"manifiesto_presente": True, "sha256_bytes_archivo_manifiesto": sha256_bytes(datos), "columnas_requeridas_presentes": True, "filas_con_experimental_split_test": n_test,
            "supuesto_1_source_split_test": {"cumple": s1, "valores_de_source_split_en_filas_test": dict(source)},
            "supuesto_2_recuentos": {"cumple": s2, "observados": recuentos, "documentados": E.DATOS["recuentos_documentados"], "filas_con_categoria_desconocida_o_no_entera": otros},
            "supuesto_3_imagenes": {"cumple": s3, "imagenes_elegibles": len(elegibles), "imagenes_faltantes": len(faltantes), "primeras_faltantes": faltantes[:MAX_NOMBRES_FALTANTES],
                                    "directorio_presente": (root / E.IMAGENES_DIR / E.DATOS["subconjunto"] / "images").is_dir()},
            "ids_o_nombres_repetidos_o_vacios": dup, "nombres_inseguros": inseguros, "apto_para_ejecutar": bool(s1 and s2 and s3 and basicos)}


def precomprobar(root, *, autorizacion, sha256_protocolo, sufijo="", destino=None) -> dict:
    token = U.exigir_autorizacion(autorizacion)
    root = Path(root)
    E.cargar_protocolo(root, sha256_protocolo)                                             # mismo protocolo que usará el ejecutor (hash externo y parámetros)
    destino = exigir_salida_segura(root, Path(destino) if destino else root / SALIDA_REL / (token.split(":")[1] + sufijo), "results/r8_v2")
    if destino.exists():
        raise GuardError(f"K5: {destino} ya existe y no se sobrescribe (use --sufijo para otra precomprobación)")
    unica = (root / E.SALIDA_REL).resolve()
    if destino == unica or unica in destino.parents:
        raise GuardError("la precomprobación no puede escribir dentro del directorio de la evaluación única (no crea el bloqueo)")
    r = analizar(root)
    destino = directorio_nuevo(destino)
    resumen = U.a_json({"esquema": "precomprobacion_estructural_r8/1", "autorizacion": token, "sha256_protocolo": sha256_protocolo, "fecha": U.ahora(),
                        "alcance": "solo lectura del manifiesto y existencia de archivos; sin inferencia, métricas ni bloqueo de ejecución única", "resultado": r})
    escribir_json_nuevo(destino / "resumen_precomprobacion.json", resumen)
    return {"dir": destino, "resultado": r}


def main(argv=None, *, root=None) -> int:
    ap = argparse.ArgumentParser(description="R8: precomprobación estructural acotada del test; solo lectura; solo la autora, en Falcon")
    ap.add_argument("--autorizacion", default="", help="CONFIRMADO:D-NNN (decisión específica de la precomprobación)")
    ap.add_argument("--sha256-protocolo", default="", help="SHA-256 (LF) del protocolo R8 v2")
    ap.add_argument("--sufijo", default="", help="sufijo del directorio de salida para repetir la precomprobación")
    args = ap.parse_args(argv)
    root = Path(root or RAIZ_REPO).resolve()
    try:
        r = precomprobar(root, autorizacion=args.autorizacion, sha256_protocolo=args.sha256_protocolo, sufijo=args.sufijo)
        print(json.dumps(r["resultado"], indent=2, ensure_ascii=False, sort_keys=True))
        print(f"resumen: {r['dir']}")
        print("PRECOMPROBACIÓN: " + ("APTO para ejecutar R8." if r["resultado"]["apto_para_ejecutar"] else "NO APTO: revise los supuestos que no cumplen (no se creó el bloqueo; no se infirió nada)."))
        return 0 if r["resultado"]["apto_para_ejecutar"] else 2
    except GuardError as exc:
        print(f"DETENIDO: {exc}", file=sys.stderr)
        return U.CODIGO_SALIDA_GUARDA


if __name__ == "__main__":
    raise SystemExit(main())
