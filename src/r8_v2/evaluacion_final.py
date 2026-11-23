"""R8 (mapeo v2): ejecutor oficial de la evaluación final ÚNICA del sistema congelado (`sistema_congelado_v1`, declaración D-060).

NO se ejecuta durante la implementación ni en las pruebas sintéticas: solo la autora, en Falcon, con una autorización independiente y específica.

Compuertas (todas ANTES de abrir cualquier archivo del test):
  autorización `CONFIRMADO:D-NNN` y confirmación literal; SHA-256 del protocolo R8 v2 y de la declaración recibidos como REFERENCIAS EXTERNAS y comparados con los archivos y entre sí;
  parámetros del protocolo iguales a las constantes del código (anti-deriva; IOV con su texto literal); verificación completa de la declaración (artefactos, 15 miembros, código de inferencia,
  checkpoints y reporte real); integridad (HEAD, árbol limpio, ejecutor y protocolo versionados); directorio de resultados inexistente; carga y verificación de los 15 checkpoints.
Fases: (1) crear el bloqueo de ejecución única; (2) leer el manifiesto y seleccionar las imágenes elegibles (mapeo v2: positivo 7; negativo {0,1,2,3,4,6}; la categoría 5 se excluye y nunca se
  predice); (3) INFERENCIA SIN ETIQUETAS con los 15 miembros (promedio simple y regla `p >= t_hat`), con predicciones escritas y registradas ANTES de liberar las etiquetas; (4) evaluación
  (`estadistica`): umbral fijo, sin recalibración ni selección. Una segunda ejecución queda bloqueada.

CLI (solo la autora): python -B -m src.r8_v2.evaluacion_final --modo comprobar|ejecutar --autorizacion CONFIRMADO:D-NNN --sha256-protocolo <64 hex> --sha256-declaracion <64 hex>
                      [--confirmar-r8 EJECUTAR_R8_UNA_SOLA_VEZ]   (solo `ejecutar`)
`comprobar` ejecuta las compuertas y la carga de modelos y lee solo el ENCABEZADO del manifiesto (no filas ni imágenes del test); no crea el bloqueo.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np

from src.r4_v2.common import (RAIZ_REPO, GuardError, directorio_nuevo, escribir_json_nuevo, exigir_salida_segura, git_ok, git_texto, sha256_archivo, sha256_bytes)
from src.r4_v2.common import cargar_config as cargar_config_g2
from src.r5_v2.comun import comparar_instantaneas, huella_archivo
from src.r7_v2 import sistema_final as SF
from src.r7_v2 import umbral_sistema as U
from src.r8_v2 import congelamiento as C
from src.r8_v2 import declaracion as DC
from src.r8_v2 import estadistica as ES

PROTOCOLO_REL = "configs/r8_v2/protocolo_r8_v2.json"
PROTOCOLO_HISTORICO_REL = "results/classification/r7_optimizacion/cierre_seleccion/protocolo_r8_congelado.json"
MANIFIESTO_TEST_REL = "configs/splits/mmotu_experimental_split_standard.csv"
IMAGENES_DIR = "data/processed/MMOTU_OTU_2D_experimental"
SALIDA_REL = "results/r8_v2/phash_le8_v1/evaluacion_final_v1"
CONFIRMACION = "EJECUTAR_R8_UNA_SOLA_VEZ"
MODOS = ("comprobar", "ejecutar")
HEX64 = re.compile(r"[0-9a-f]{64}")
COLUMNAS_MANIFIESTO = ("image_id", "image_name", "source_split", "original_class", "experimental_split")
CLAVES_FILA_INFERENCIA = ("image_id", "image_name", "image_path")                    # NINGUNA etiqueta, clase original ni pliegue
RUTAS_VIGILADAS = ("src/r8_v2", "src/r7_v2", "src/r5_v2", "src/r4_v2", "src/classification", "configs/r8_v2", "configs/r4_v2", "configs/r7_v2", "configs/splits")
ARCHIVOS_EJECUTOR = ("src/r8_v2/__init__.py", "src/r8_v2/evaluacion_final.py", "src/r8_v2/estadistica.py", "src/r8_v2/declaracion.py", "src/r8_v2/congelamiento.py", PROTOCOLO_REL,
                     PROTOCOLO_HISTORICO_REL)
IOV = {
    "R8.1": "Las decisiones de selección y optimización se encuentran cerradas antes de realizar la evaluación final.",
    "R8.2": "Se reportan AP, AUC-ROC, sensibilidad y especificidad como métricas de evaluación final.",
    "R8.3": "Las métricas se acompañan de intervalos de confianza del 95 % según el procedimiento estadístico establecido.",
    "R8.4": "Se declara el uso previo del conjunto de prueba en las pruebas preliminares y se reportan las métricas finales una sola vez con la configuración congelada."}
DATOS = {
    "manifiesto": MANIFIESTO_TEST_REL, "imagenes_dir": IMAGENES_DIR, "subconjunto": "test",
    "mapeo": {"clase_0_lesion_no_maligna": [0, 1, 2, 3, 4, 6], "clase_1_lesion_maligna": [7], "excluida_ovario_normal": [5]},
    "recuentos_documentados": {"elegibles_clase_0": 367, "elegibles_clase_1": 15, "normales_excluidas": 87},
    "recuentos_documentados_nota": "documentados históricamente; si difieren se detiene antes de inferir (enmienda y decisión de la autora)",
    "origen_mmotu_de_las_filas_de_test": "validation",
    "origen_mmotu_nota": "source_split es el subconjunto de procedencia en MMOTU (val_cls.txt = validation); experimental_split == test es la asignacion de evaluacion: las 469 filas de test proceden del "
                         "subconjunto de validacion original de MMOTU y las 1000 de desarrollo del de entrenamiento original; ninguna otra fila puede proceder de validation",
    "solapamiento_con_desarrollo": "ninguno por image_id ni image_name (resto del manifiesto y particion v2 de desarrollo); no se afirma independencia por paciente"}
ESTADISTICA = {
    "evaluacion_unica": True, "semilla_bootstrap": ES.SEMILLA_BOOTSTRAP, "n_bootstrap": ES.N_BOOTSTRAP, "nivel_ic": ES.NIVEL_IC,
    "ic_ap_roc_auc": "bootstrap percentil estratificado por imagen", "ic_sensibilidad_especificidad": "binomial exacto de Clopper-Pearson",
    "umbral_primario": SF.T_HAT, "regla": "p >= t_hat", "umbral_secundario_descriptivo": ES.UMBRAL_SECUNDARIO, "analisis_por_numero_de_modelos": False,
    "comparacion_b2_vs_b0": "contraste historico documentado; no ejecutable en v2"}


# ============================================================================ utilidades
def sha256_lf(path) -> str:
    return DC.sha256_lf(path)


def _exigir(cond: bool, mensaje: str) -> None:
    if not cond:
        raise GuardError(mensaje)


def _hex(valor, que: str) -> str:
    _exigir(isinstance(valor, str) and bool(HEX64.fullmatch(valor)), f"{que}: se exige un SHA-256 de 64 hex en minúsculas, recibido por una vía externa")
    return valor


def verificar_integridad_r8(root) -> dict:
    """HEAD, árbol limpio en las rutas vigiladas (incluye el manifiesto), ejecutor y protocolos versionados. No lee el contenido del manifiesto del test."""
    root = Path(root)
    head = git_texto(root, "rev-parse", "HEAD", obligatorio=True)
    status = [x for x in git_ok(root, "status", "--porcelain", "--untracked-files=all", "--", *RUTAS_VIGILADAS).decode("utf-8", "replace").splitlines() if x.strip()]
    bloqueos = [{"codigo": "arbol_sucio", "detalle": x} for x in status]
    archivos = {}
    for rel in ARCHIVOS_EJECUTOR:
        h = huella_archivo(root, rel)
        archivos[rel] = h
        if h["sha256_bytes_archivo"] is None:
            bloqueos.append({"codigo": "archivo_ausente", "detalle": rel})
        elif h["sha256_contenido_blob"] is None:
            bloqueos.append({"codigo": "archivo_no_versionado", "detalle": rel})
    try:
        git_ok(root, "ls-files", "--error-unmatch", MANIFIESTO_TEST_REL)
    except GuardError:
        bloqueos.append({"codigo": "manifiesto_no_versionado", "detalle": MANIFIESTO_TEST_REL})
    return {"git_commit_head": head, "rama": git_texto(root, "rev-parse", "--abbrev-ref", "HEAD"), "status_porcelain": status, "archivos": archivos, "bloqueos": bloqueos}


# ============================================================================ protocolo y compuertas
def validar_protocolo(p: dict) -> None:
    _exigir(p.get("esquema") == "protocolo_r8_v2/1", "esquema del protocolo distinto de protocolo_r8_v2/1")
    _exigir(p.get("estadistica") == ESTADISTICA, "estadistica del protocolo distinta de las constantes del código (semilla, remuestreos, métodos, umbral, regla): se bloquea; exige enmienda")
    _exigir(p.get("datos") == DATOS, "datos del protocolo distintos de las constantes del código (manifiesto, mapeo v2, recuentos): se bloquea; exige enmienda")
    _exigir({k: v.get("texto") for k, v in p.get("iov_originales", {}).items()} == IOV, "los cuatro IOV de R8 deben conservar su texto literal original")
    s = p.get("sistema", {})
    _exigir(s.get("t_hat") == SF.T_HAT and s.get("regla") == "p >= t_hat" and s.get("n_miembros") == SF.N_MIEMBROS and s.get("arquitectura") == "densenet121"
            and s.get("representacion") == "A", "sistema del protocolo distinto del declarado")
    _hex(s.get("declaracion", {}).get("sha256_contenido"), "protocolo.sistema.declaracion.sha256_contenido")
    _exigir(s["declaracion"].get("ruta") == DC.DECLARACION_REL, "ruta de la declaración distinta de la esperada")
    h = p.get("historico", {}).get("protocolo_original", {})
    _exigir(h.get("ruta") == PROTOCOLO_HISTORICO_REL and bool(HEX64.fullmatch(str(h.get("sha256_contenido")))), "el protocolo histórico debe estar referenciado con su SHA-256")
    _exigir("no ejecutable" in str(p.get("historico", {}).get("contraste_b2_vs_b0", {}).get("estado", "")), "B2 − B0 debe figurar como contraste histórico no ejecutable en v2")
    _exigir(p.get("autorizacion_requerida", {}).get("independiente_y_especifica") is True, "R8 exige una autorización independiente y específica de la autora")


def cargar_protocolo(root, sha256_esperado) -> dict:
    _hex(sha256_esperado, "--sha256-protocolo")
    ruta = Path(root) / PROTOCOLO_REL
    _exigir(ruta.is_file(), f"{PROTOCOLO_REL} no existe")
    _exigir(sha256_lf(ruta) == sha256_esperado, "el SHA-256 del protocolo R8 v2 no coincide con la referencia externa (protocolo alterado o distinto)")
    p = json.loads(ruta.read_text(encoding="utf-8"))
    validar_protocolo(p)
    return p


def preparar_contexto(root, *, autorizacion, sha256_protocolo, sha256_declaracion, confirmacion=None, exigir_confirmacion=True, verificar=None, verificar_declaracion=None,
                      cargar_modelo=None, device=None, entorno=None) -> dict:
    """Todas las compuertas y la carga de los 15 modelos. NO abre ningún archivo del test."""
    root = Path(root)
    token = U.exigir_autorizacion(autorizacion)
    if exigir_confirmacion:
        _exigir(confirmacion == CONFIRMACION, f"falta la confirmación explícita (--confirmar-r8 {CONFIRMACION}): R8 se ejecuta una sola vez")
    protocolo = cargar_protocolo(root, sha256_protocolo)
    sha_decl = _hex(sha256_declaracion, "--sha256-declaracion")
    _exigir(protocolo["sistema"]["declaracion"]["sha256_contenido"] == sha_decl, "la declaración recibida difiere de la fijada en el protocolo R8 v2")
    verificar = verificar or (lambda: verificar_integridad_r8(root))
    inicio = U.exigir_integridad(verificar())
    ruta_hist = root / PROTOCOLO_HISTORICO_REL
    _exigir(ruta_hist.is_file() and sha256_lf(ruta_hist) == protocolo["historico"]["protocolo_original"]["sha256_contenido"],
            "el protocolo histórico de R8 cambió o falta (debe conservarse intacto)")
    verif_decl = (verificar_declaracion or (lambda r, s: DC.exigir_declaracion(r, s, comprobar_checkpoints=True, comprobar_reporte=True)))(root, sha_decl)
    config = C.cargar_config(root / C.CONFIG_REL)
    config_g2 = cargar_config_g2(root / config["referencia_g2"]["config"])
    _exigir(sha256_lf(root / config["referencia_g2"]["config"]) == config["referencia_g2"]["sha256_contenido_blob"], "la configuración de G2 difiere del pin del YAML congelado")
    m = config_g2["mapeo_etiquetas"]
    _exigir({"clase_0_lesion_no_maligna": m["clase_0_lesion_no_maligna"], "clase_1_lesion_maligna": m["clase_1_lesion_maligna"], "excluida_ovario_normal": m["excluida_ovario_normal"]}
            == DATOS["mapeo"], "el mapeo binario v2 del protocolo difiere del de G2")
    _exigir(config_g2["diseno"]["imagenes_dir"].strip("/") == IMAGENES_DIR, "imagenes_dir de G2 distinto del del protocolo")
    t_hat = SF.t_hat_exacto(config["sistema"]["umbral"]["t_hat"])
    _exigir(config["sistema"]["umbral"]["t_hat"] == SF.T_HAT == protocolo["estadistica"]["umbral_primario"], "t_hat del YAML congelado distinto del del protocolo (no se modifica el umbral)")
    miembros = C.validar_miembros(C.leer_tabla(root / C.MIEMBROS_REL), config)
    dev_ids = leer_ids_desarrollo(root, config["archivos_de_procedencia"][C.REL_PARTICION])                 # partición de desarrollo (nunca test): ids y nombres para descartar solapamiento
    destino = exigir_salida_segura(root, root / SALIDA_REL, "results/r8_v2")
    _exigir(not destino.exists(), f"K5: {SALIDA_REL} ya existe: R8 se ejecuta una sola vez y no se sobrescribe (si hubo una ejecución previa, requiere una decisión documentada de la autora)")
    dev = device if device is not None else SF._dispositivo()
    cargar = cargar_modelo or SF._cargar_modelo_g2
    modelos = {mm["id_corrida"]: cargar(SF.verificar_miembro_en_disco(root, mm), dev) for mm in miembros}           # SHA-256 verificado justo antes de cargar
    efectivo = entorno if entorno is not None else SF._versiones()
    advertencias = [f"{k}: declarado {config['entorno_registrado'].get(k)!r}, efectivo {efectivo.get(k)!r}" for k in C.ENTORNO_CLAVES if str(config["entorno_registrado"].get(k)) != str(efectivo.get(k))]
    return {"token": token, "protocolo": protocolo, "sha256_protocolo": sha256_protocolo, "sha256_declaracion": sha_decl, "verificacion_declaracion": verif_decl, "config": config, "config_g2": config_g2,
            "t_hat": t_hat, "miembros": miembros, "dev": dev_ids, "modelos": modelos, "device": dev, "destino": destino, "inicio": inicio, "verificar": verificar, "entorno": efectivo, "advertencias_entorno": advertencias}


# ============================================================================ datos del test (solo el ejecutor oficial; tras el bloqueo)
class Reservadas:
    """Clases originales del test, SIN acceso hasta que las predicciones estén escritas y registradas (separación inferencia / evaluación)."""

    def __init__(self, clases: dict, sha_manifiesto: str):
        self._clases, self._sha, self._liberada = dict(clases), sha_manifiesto, False

    def liberar(self, ruta_registro_inferencia, sha_manifiesto_actual: str) -> dict:
        ruta = Path(ruta_registro_inferencia)
        _exigir(ruta.is_file(), "las etiquetas no se liberan antes de escribir y registrar las predicciones")
        _exigir(json.loads(ruta.read_text(encoding="utf-8")).get("estado") == "predicciones_escritas", "el registro de inferencia no está en estado predicciones_escritas")
        _exigir(sha_manifiesto_actual == self._sha, "el manifiesto cambió entre la selección y la evaluación")
        _exigir(not self._liberada, "las etiquetas ya se liberaron")
        self._liberada = True
        mapa = {int(c): 0 for c in DATOS["mapeo"]["clase_0_lesion_no_maligna"]}
        mapa.update({int(c): 1 for c in DATOS["mapeo"]["clase_1_lesion_maligna"]})
        return {i: mapa[c] for i, c in self._clases.items()}, dict(self._clases)


def _nombre_seguro(nombre: str) -> bool:
    norm = nombre.replace("\\", "/")
    return bool(nombre) and norm == Path(norm).name and ".." not in nombre and ":" not in nombre and not nombre.startswith(".") and nombre.lower() != "test"


def leer_encabezado_manifiesto(root) -> list:
    """SOLO la primera línea (nombres de columnas): no lee filas, clases ni imágenes del test."""
    with open(Path(root) / MANIFIESTO_TEST_REL, "rb") as f:                                 # binario: solo se decodifica la primera línea (nada del resto del archivo)
        linea = f.readline().decode("utf-8-sig").rstrip("\r\n")
    columnas = next(csv.reader([linea]))
    faltan = [c for c in COLUMNAS_MANIFIESTO if c not in columnas]
    _exigir(not faltan, f"el manifiesto no tiene las columnas requeridas: {faltan}")
    return columnas


def leer_ids_desarrollo(root, sha256_esperado=None):
    """`image_id` e `image_name` de la partición v2 de DESARROLLO (nunca test). Si se da `sha256_esperado`, el archivo debe coincidir (pin del YAML congelado)."""
    ruta = Path(root) / C.REL_PARTICION
    _exigir(ruta.is_file(), f"falta la partición de desarrollo {C.REL_PARTICION}")
    datos = ruta.read_bytes()
    sha = sha256_bytes(datos)
    _exigir(sha256_esperado is None or sha == sha256_esperado, "la partición de desarrollo difiere del SHA-256 fijado en el YAML congelado")
    lector = csv.DictReader(io.StringIO(datos.decode("utf-8"), newline=""))
    _exigir({"image_id", "image_name"} <= set(lector.fieldnames or []), "la partición de desarrollo no tiene image_id e image_name")
    ids, nombres = set(), set()
    for r in lector:
        ids.add((r["image_id"] or "").strip())
        nombres.add((r["image_name"] or "").strip())
    _exigir(bool(ids), "la partición de desarrollo está vacía")
    return ids, nombres, sha


def leer_manifiesto_prueba(root, dev):
    """Selecciona las imágenes ELEGIBLES del test (mapeo v2). Devuelve filas SIN etiquetas, las clases reservadas y recuentos. Detiene ante cualquier inconsistencia.

    `source_split` es el subconjunto de PROCEDENCIA en MMOTU (las filas de test proceden del subconjunto de validación original: «validation»); `experimental_split == "test"` es la
    asignación de evaluación. `dev` = (ids, nombres) de la partición de desarrollo: no puede haber solapamiento por `image_id` ni `image_name`."""
    datos = (Path(root) / MANIFIESTO_TEST_REL).read_bytes()                                  # un único acceso: hash y análisis
    sha = sha256_bytes(datos)
    lector = csv.DictReader(io.StringIO(datos.decode("utf-8"), newline=""))
    faltan = [c for c in COLUMNAS_MANIFIESTO if c not in (lector.fieldnames or [])]
    _exigir(not faltan, f"el manifiesto no tiene las columnas requeridas: {faltan}")
    mapa = {int(c): 0 for c in DATOS["mapeo"]["clase_0_lesion_no_maligna"]}
    mapa.update({int(c): 1 for c in DATOS["mapeo"]["clase_1_lesion_maligna"]})
    excluidas = {int(c) for c in DATOS["mapeo"]["excluida_ovario_normal"]}
    origen = DATOS["origen_mmotu_de_las_filas_de_test"]
    filas, clases, ids, nombres, n_norm = [], {}, set(), set(), 0
    fuera_ids, fuera_nombres, n_origen_fuera = set(), set(), 0
    for bruta in lector:
        if (bruta["experimental_split"] or "").strip().lower() != DATOS["subconjunto"]:
            fuera_ids.add((bruta["image_id"] or "").strip())
            fuera_nombres.add((bruta["image_name"] or "").strip())
            n_origen_fuera += (bruta["source_split"] or "").strip().lower() == origen
            continue
        _exigir((bruta["source_split"] or "").strip().lower() == origen, f"una fila de test no procede del subconjunto de MMOTU «{origen}» previsto")
        iid, nombre = (bruta["image_id"] or "").strip(), (bruta["image_name"] or "").strip()
        try:
            oc = int(str(bruta["original_class"]).strip())
        except ValueError:
            raise GuardError("original_class no entero en una fila de test") from None
        _exigir(bool(iid) and _nombre_seguro(nombre), "image_id vacío o image_name no seguro en una fila de test")
        _exigir(iid not in ids and nombre not in nombres, "image_id o image_name repetido en el test")
        ids.add(iid)
        nombres.add(nombre)
        if oc in excluidas:
            n_norm += 1                                                                    # categoría 5: se excluye; nunca se predice ni se evalúa
            continue
        _exigir(oc in mapa, f"categoría original desconocida en el test: {oc}")
        filas.append({"image_id": iid, "image_name": nombre, "image_path": f"{IMAGENES_DIR}/{DATOS['subconjunto']}/images/{nombre}"})
        clases[iid] = oc
    _exigir(n_origen_fuera == 0, f"hay filas fuera del test con procedencia MMOTU «{origen}»: la partición de evaluación no es la prevista")
    _exigir(not (ids & fuera_ids) and not (nombres & fuera_nombres), "solapamiento por image_id o image_name entre el test y el resto del manifiesto")
    _exigir(not (ids & set(dev[0])) and not (nombres & set(dev[1])), "solapamiento por image_id o image_name entre el test y la partición de desarrollo")
    filas.sort(key=lambda f: f["image_id"])
    n1 = sum(1 for c in clases.values() if mapa[c] == 1)
    recuentos = {"elegibles_clase_0": len(filas) - n1, "elegibles_clase_1": n1, "normales_excluidas": n_norm}
    _exigir(recuentos == DATOS["recuentos_documentados"], f"los recuentos del test {recuentos} difieren de los documentados {DATOS['recuentos_documentados']}: se detiene antes de inferir")
    return filas, Reservadas(clases, sha), {"sha256_bytes_archivo_manifiesto": sha, "recuentos": recuentos, "n_elegibles": len(filas), "origen_mmotu_filas_test": origen,
                                            "solapamiento_con_desarrollo": 0}


def exigir_filas_sin_etiquetas(filas) -> None:
    for f in filas:
        _exigir(set(f) == set(CLAVES_FILA_INFERENCIA), f"las filas de inferencia solo pueden tener {CLAVES_FILA_INFERENCIA} (sin etiqueta, clase original ni pliegue)")


def resolver_imagen_prueba(root, fila) -> Path:
    """Ruta resuelta de una imagen de TEST, restringida a `<imagenes_dir>/test/images` (sin enlaces ni salidas del directorio). No abre el archivo."""
    raiz = Path(root).resolve()
    _exigir(fila["image_path"].replace("\\", "/") == f"{IMAGENES_DIR}/{DATOS['subconjunto']}/images/{fila['image_name']}" and _nombre_seguro(fila["image_name"]), "ruta de imagen de test no autorizada")
    nominal = raiz / IMAGENES_DIR / DATOS["subconjunto"] / "images"
    permitido = nominal.resolve()
    _exigir(os.path.normcase(str(permitido)) == os.path.normcase(str(nominal)), "el directorio de imágenes del test está redirigido (enlace)")
    resuelta = (permitido / fila["image_name"]).resolve()
    _exigir(os.path.normcase(str(resuelta.parent)) == os.path.normcase(str(permitido)), "la ruta resuelta sale del directorio de imágenes del test")
    _exigir(resuelta.is_file(), "imagen de test no encontrada")
    return resuelta


class DatasetPrueba:
    """Mismo contrato de lectura que `DatasetLesiones` de G2 (RGB + transformación de validación), SIN etiquetas: devuelve el marcador 0. Nivel de módulo: serializable con DataLoader."""

    def __init__(self, root, filas, transform=None):
        exigir_filas_sin_etiquetas(filas)
        self.rutas = [resolver_imagen_prueba(root, f) for f in filas]
        self.transform = transform

    def __len__(self):
        return len(self.rutas)

    def __getitem__(self, i):
        from PIL import Image
        imagen = Image.open(self.rutas[i]).convert("RGB")
        return (self.transform(imagen) if self.transform else imagen), 0


def predictor_prueba(root, config_g2, semilla, preprocesamiento=None):
    """Probabilidad de la clase maligna (softmax float32) con el preprocesamiento de G2 (`get_val_transforms`, `create_dataloader`, `predict`) sobre imágenes de test sin etiquetas."""
    p = preprocesamiento or SF.preprocesamiento()
    receta = config_g2["receta"]

    def predecir(modelo, filas, device):
        ds = DatasetPrueba(root, filas, p["get_val_transforms"](receta["entrada"]["image_size"]))
        loader = p["create_dataloader"](ds, batch_size=receta["batch_size"], shuffle=False, num_workers=receta["num_workers"], seed=semilla)
        return np.asarray(p["predict"](modelo, loader, device)[2], dtype=np.float32)
    return predecir


# ============================================================================ escritura de salidas (siempre archivos nuevos)
def _tabla_nueva(path, columnas, filas) -> None:
    with open(path, "x", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(columnas)
        w.writerows(filas)


def _repr(x) -> str:
    return repr(float(x))


# ============================================================================ ejecución
def comprobar(root, **kw) -> dict:
    """Compuertas + carga de los 15 modelos + encabezado del manifiesto. No crea el bloqueo ni lee filas, clases ni imágenes del test."""
    ctx = preparar_contexto(root, exigir_confirmacion=False, **kw)
    columnas = leer_encabezado_manifiesto(root)
    return {"estado": "listo_para_ejecutar", "n_miembros": len(ctx["miembros"]), "columnas_manifiesto_ok": bool(columnas), "advertencias_entorno": ctx["advertencias_entorno"],
            "git_commit_head": ctx["inicio"]["git_commit_head"], "destino_libre": str(ctx["destino"])}


def ejecutar(root, *, predecir=None, **kw) -> dict:
    """R8: UNA sola ejecución. Tras las compuertas y la carga de modelos crea el bloqueo; cualquier fallo posterior deja el bloqueo y un `estado_final.json` de aborto."""
    root = Path(root)
    ctx = preparar_contexto(root, **kw)
    destino = directorio_nuevo(ctx["destino"])                                           # bloqueo de ejecución única (creación exclusiva)
    base = {"protocolo_id": "protocolo_r8_v2", "autorizacion": ctx["token"], "sha256_protocolo": ctx["sha256_protocolo"], "sha256_declaracion": ctx["sha256_declaracion"],
            "git_commit_head": ctx["inicio"]["git_commit_head"], "rama": ctx["inicio"]["rama"], "t_hat": SF.T_HAT, "regla": "p >= t_hat", "entorno_efectivo": ctx["entorno"],
            "advertencias_entorno": ctx["advertencias_entorno"], "fecha_inicio": U.ahora()}
    escribir_json_nuevo(destino / "ejecucion.lock.json", U.a_json({**base, "estado": "iniciada", "nota": "este archivo bloquea una segunda ejecución; solo una decisión documentada de la autora puede levantarlo"}))
    try:
        return _fases(root, ctx, destino, base, predecir)
    except BaseException as exc:                                                         # incluye GuardError y fallos de hardware; el bloqueo permanece
        try:
            escribir_json_nuevo(destino / "estado_final.json", U.a_json({**base, "estado": "abortada", "motivo": f"{type(exc).__name__}: {str(exc)[:500]}",
                                                                          "fecha_fin": U.ahora(), "traza": traceback.format_exc()[-1500:]}))
        except GuardError:
            pass
        raise


def _fases(root, ctx, destino, base, predecir) -> dict:
    miembros, config_g2, t_hat = ctx["miembros"], ctx["config_g2"], ctx["t_hat"]
    filas, reservadas, info = leer_manifiesto_prueba(root, ctx["dev"])                               # 1) selección sin etiquetas visibles
    for f in filas:
        resolver_imagen_prueba(root, f)                                                  # todas las imágenes existen y están en test/images antes de inferir
    exigir_filas_sin_etiquetas(filas)
    ids = [f["image_id"] for f in filas]
    probs = {}                                                                           # 2) inferencia con los 15 miembros, sin etiquetas
    for m in miembros:
        pr = predecir(m) if callable(predecir) else predictor_prueba(root, config_g2, m["semilla"])
        v = np.asarray(pr(ctx["modelos"][m["id_corrida"]], filas, ctx["device"]), dtype=np.float32)
        _exigir(v.shape == (len(filas),) and np.isfinite(v).all(), f"{m['id_corrida']}: predicciones con forma o valores inválidos")
        probs[m["id_corrida"]] = v
    agg = SF.agregar_y_decidir(probs, SF.T_HAT)                                          # promedio simple (float64) y regla `>=`
    nombres = sorted(probs, key=str)
    _tabla_nueva(destino / "predicciones_por_miembro.csv", ["image_id", *nombres], [[i, *[_repr(probs[n][k]) for n in nombres]] for k, i in enumerate(ids)])
    _tabla_nueva(destino / "predicciones_ensamble.csv", ["image_id", "p_ensamble", "positivo_t_hat"], [[i, _repr(agg["puntuacion"][k]), int(agg["positivo"][k])] for k, i in enumerate(ids)])
    escribir_json_nuevo(destino / "registro_inferencia.json", U.a_json({                 # ANTES de liberar las etiquetas
        **base, "estado": "predicciones_escritas", "n_imagenes": len(ids), "miembros": nombres, "manifiesto": info,
        "sha256_predicciones_por_miembro": sha256_archivo(destino / "predicciones_por_miembro.csv"), "sha256_predicciones_ensamble": sha256_archivo(destino / "predicciones_ensamble.csv"),
        "etiquetas_usadas_en_inferencia": False, "fecha": U.ahora()}))
    sha_actual = sha256_bytes((Path(root) / MANIFIESTO_TEST_REL).read_bytes())
    y_por_id, clases = reservadas.liberar(destino / "registro_inferencia.json", sha_actual)    # 3) evaluación posterior
    y = np.array([y_por_id[i] for i in ids], dtype=np.int64)
    res = ES.calcular_resultados(y, agg["puntuacion"], t_hat)
    fin = ctx["verificar"]()
    dif = comparar_instantaneas(ctx["inicio"], fin)
    _tabla_nueva(destino / "evaluacion_por_imagen.csv", ["image_id", "original_class", "label_v2", "p_ensamble", "positivo_t_hat"],
                 [[i, clases[i], int(y[k]), _repr(agg["puntuacion"][k]), int(agg["positivo"][k])] for k, i in enumerate(ids)])
    p = res["umbral_primario"]
    filas_m = [["ap", _repr(res["puntuales"]["ap"]), _repr(res["ic95_bootstrap_percentil_estratificado"]["ap"]["p2_5"]), _repr(res["ic95_bootstrap_percentil_estratificado"]["ap"]["p97_5"]), "bootstrap percentil estratificado"],
               ["roc_auc", _repr(res["puntuales"]["roc_auc"]), _repr(res["ic95_bootstrap_percentil_estratificado"]["roc_auc"]["p2_5"]),
                _repr(res["ic95_bootstrap_percentil_estratificado"]["roc_auc"]["p97_5"]), "bootstrap percentil estratificado"],
               ["sensibilidad_t_hat", _repr(p["sensibilidad"]), _repr(p["sensibilidad_ic95_clopper_pearson"][0]), _repr(p["sensibilidad_ic95_clopper_pearson"][1]), "Clopper-Pearson"],
               ["especificidad_t_hat", _repr(p["especificidad"]), _repr(p["especificidad_ic95_clopper_pearson"][0]), _repr(p["especificidad_ic95_clopper_pearson"][1]), "Clopper-Pearson"],
               ["vpp_t_hat", "" if p["vpp"] is None else _repr(p["vpp"]), "", "", "sin IC (prevalencia del test)"]]
    _tabla_nueva(destino / "metricas_primarias.csv", ["metrica", "valor", "ic95_inferior", "ic95_superior", "metodo_ic"], filas_m)
    aceptado = not (bool(dif) or bool(fin["bloqueos"]))
    escribir_json_nuevo(destino / "resultados_r8.json", U.a_json({
        **base, "estado": "completada", "resultados": res, "recuentos": info["recuentos"], "n_imagenes": len(ids), "diferencias_inicio_fin": dif, "invalida_integridad": not aceptado,
        "reporte_aceptado": aceptado, "fecha_fin": U.ahora(),
        "limitaciones": ["IC por imagen: sin group_id en test; independencia no demostrada", "pocas positivas: IC de sensibilidad muy amplio", "t_hat calibrado con el OOF de tres modelos; no validado para el ensamble de quince",
                         "un único subtipo maligno; uso exploratorio previo del test; sin uso clínico validado", "no se reajustó nada con el test"]}))
    escribir_json_nuevo(destino / "estado_final.json", U.a_json({**base, "estado": "completada", "fecha_fin": U.ahora(), "reporte_aceptado": aceptado}))
    return {"dir": destino, "resultados": res, "reporte_aceptado": aceptado, "advertencias_entorno": ctx["advertencias_entorno"]}


def main(argv=None, *, root=None, **inyectado) -> int:
    ap = argparse.ArgumentParser(description="R8 v2: evaluación final única del sistema congelado; solo la autora, en Falcon, con autorización independiente")
    ap.add_argument("--modo", required=True, choices=MODOS)
    ap.add_argument("--autorizacion", default="", help="CONFIRMADO:D-NNN (decisión independiente y específica de R8)")
    ap.add_argument("--sha256-protocolo", default="", help="SHA-256 (LF) del protocolo R8 v2, referencia externa")
    ap.add_argument("--sha256-declaracion", default="", help="SHA-256 (LF) de la declaración D-060, referencia externa")
    ap.add_argument("--confirmar-r8", default=None, help=f"solo `ejecutar`: debe ser exactamente {CONFIRMACION}")
    args = ap.parse_args(argv)
    root = Path(root or RAIZ_REPO).resolve()
    try:
        comunes = dict(autorizacion=args.autorizacion, sha256_protocolo=args.sha256_protocolo, sha256_declaracion=args.sha256_declaracion, **inyectado)
        if args.modo == "comprobar":
            r = comprobar(root, **comunes)
            print(json.dumps(r, indent=2, ensure_ascii=False, sort_keys=True))
            print("COMPROBACIÓN SUPERADA: no se abrió ninguna fila ni imagen del test; el bloqueo de ejecución única NO se creó.")
            return 0
        r = ejecutar(root, confirmacion=args.confirmar_r8, **comunes)
        print(f"R8 completada; resultados en {r['dir']}")
        if not r["reporte_aceptado"]:
            print("SALIDA NO ACEPTADA: la integridad final difiere de la inicial.", file=sys.stderr)
            return U.CODIGO_SALIDA_INTEGRIDAD_INVALIDA
        return 0
    except GuardError as exc:
        print(f"DETENIDO: {exc}", file=sys.stderr)
        return U.CODIGO_SALIDA_GUARDA


if __name__ == "__main__":
    raise SystemExit(main())
