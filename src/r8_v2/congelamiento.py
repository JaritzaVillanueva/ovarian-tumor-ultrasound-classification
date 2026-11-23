"""R8 `sistema_congelado_v1`: preparación y verificación del congelamiento formal del sistema final (NO lo declara; NO abre el test).

Sistema: DenseNet121 + representación A, 15 checkpoints de G2 (los del manifiesto verificado, sin sustituir miembros), preprocesamiento original de G2, promedio aritmético simple de las
15 probabilidades de la clase maligna, regla `p >= t_hat` con t_hat = 0.08456226376195748 (empates no se rompen). `t_hat` es una regla experimental previa a R8: NO es un umbral validado
para quince modelos y no garantiza una sensibilidad del 90 %.

Modo único `verificar` (solo lectura de artefactos de DESARROLLO y de los checkpoints; escribe un resumen en un directorio nuevo):
* integridad propia (HEAD, árbol limpio, SHA-256 de archivos con huella, protocolo versionado con el hash registrado en el YAML);
* `seleccion_r6.json` con elegida A|densenet121;
* SHA-256 de cada archivo de procedencia (manifiesto, fidelidad, umbral, partición, configuración de G2) contra los pines del YAML;
* manifiesto de miembros versionado (`configs/r8_v2/manifiesto_miembros_v1.csv`, copia byte a byte): 15 miembros únicos, coherentes con el manifiesto JSON y con la fidelidad;
* `t_hat` del resultado de umbral igual al del YAML (cadena exacta);
* SHA-256 de los 15 checkpoints recalculado en disco (donde residen: Falcon).

CLI (solo la autora, en Falcon; D-NNN propia): python -B -m src.r8_v2.congelamiento --modo verificar --autorizacion CONFIRMADO:D-NNN
No hay modo de inferencia ni de evaluación aquí: R8 tendrá su propio protocolo y ejecutor oficial. Partición provisional (phash_le8_v1, exploratoria, post hoc); conformidad pendiente.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from src.r4_v2.common import RAIZ_REPO, GuardError, directorio_nuevo, escribir_json_nuevo, exigir_ruta_no_prohibida, exigir_salida_segura, sha256_archivo
from src.r5_v2.comun import comparar_instantaneas
from src.r7_v2 import umbral_sistema as U
from src.r7_v2.delta_ensamble import cargar_config as _cargar_yaml
from src.r7_v2.delta_ensamble import exigir_protocolo, validar_comunes, verificar_integridad
from src.r7_v2.sistema_final import N_MIEMBROS, T_HAT, t_hat_exacto

PROTOCOLO_ID = "sistema_congelado_v1"
ESQUEMA_CONFIG = "sistema_congelado_v1_config/1"
CONFIG_REL = "configs/r8_v2/sistema_congelado_v1.yaml"
PROTOCOLO_REL = "src/r8_v2/protocolo_sistema_congelado_v1.md"
MIEMBROS_REL = "configs/r8_v2/manifiesto_miembros_v1.csv"
SALIDA_REL = "results/r8_v2/phash_le8_v1/sistema_congelado_v1/verificacion"
LEYENDA = U.LEYENDA
ESTADO = "preparado para congelamiento; no declarado congelado (la declaración formal es una decisión posterior); no validado para el ensamble de quince"
SISTEMA = {
    "estado": ESTADO, "congelado": False, "arquitectura": "densenet121", "representacion": "A", "semillas": [42, 43, 44], "folds": [0, 1, 2, 3, 4], "n_miembros": 15,
    "preprocesamiento": "el original de G2, sin modificaciones (get_val_transforms, DatasetLesiones, create_dataloader y predict de src/classification y src/r4_v2); sin augmentation",
    "agregacion": "promedio aritmetico simple de las 15 probabilidades de la clase maligna (softmax float32 por modelo; promedio en float64); sin ponderaciones ni seleccion de miembros",
    "regla": "p >= t_hat", "empates": "no se rompen",
    "umbral": {"t_hat": T_HAT, "s_star": "0.90",
               "estatus": "regla experimental previa a R8; no es un umbral validado para quince modelos; no garantiza una sensibilidad del 90 %",
               "origen": "q_barra de tres modelos por imagen (umbral_sistema_v1); calculado en desarrollo"},
    "alcance": "clasificacion binaria de lesion a nivel de imagen 2D; no detecta ovarios normales; sin uso clinico validado"}
PINES_REQUERIDOS = (
    "results/r7_v2/phash_le8_v1/sistema_final_v1/manifiesto/manifiesto_miembros.csv", "results/r7_v2/phash_le8_v1/sistema_final_v1/manifiesto/manifiesto_miembros.json",
    "results/r7_v2/phash_le8_v1/sistema_final_v1/fidelidad/fidelidad_por_miembro.csv", "results/r7_v2/phash_le8_v1/sistema_final_v1/fidelidad/resumen_fidelidad.json",
    "results/r7_v2/phash_le8_v1/umbral_sistema_v1/umbral_primario.json", "results/r7_v2/phash_le8_v1/umbral_sistema_v1/resumen_umbral.json",
    "results/r7_v2/phash_le8_v1/umbral_sistema_v1/trazabilidad.json", "configs/splits/v2_auto/cv_folds_phash_le8_v1.csv")
REL_MANIFIESTO_CSV, REL_MANIFIESTO_JSON, REL_FIDELIDAD_CSV, REL_FIDELIDAD_JSON, REL_UMBRAL = (PINES_REQUERIDOS[0], PINES_REQUERIDOS[1], PINES_REQUERIDOS[2], PINES_REQUERIDOS[3],
                                                                                            PINES_REQUERIDOS[4])
REL_PARTICION = PINES_REQUERIDOS[7]
ARCHIVOS_CODIGO = ("src/r8_v2/__init__.py", "src/r8_v2/congelamiento.py", "src/r7_v2/sistema_final.py", "src/r7_v2/umbral_sistema.py", "src/r7_v2/delta_ensamble.py",
                   "src/classification/transforms.py", "src/classification/dataloader.py", "src/classification/models.py", "src/classification/evaluate.py",
                   "src/r4_v2/dataset_lesiones.py", "src/r4_v2/entrenamiento.py", MIEMBROS_REL)
FRASES_PROTOCOLO = ("Preparado para congelamiento", "no declarado congelado", "no validado para el ensamble de quince", "0.08456226376195748", "p >= t_hat", "no se rompen",
                    "promedio aritmético simple", "15 checkpoints", "manifiesto_miembros_v1", "No garantiza sensibilidad del 90 %", "Preprocesamiento original de G2",
                    "declaración formal de congelamiento", "regla experimental previa a R8", "No declara el congelamiento", "No se afirma independencia absoluta del test histórico",
                    "sin sustituir miembros", "No detecta ovarios normales", "sin uso clínico validado")
COLUMNAS_MIEMBROS = ("arquitectura", "semilla", "fold", "id_corrida", "intento", "ruta_checkpoint", "sha256_bytes_archivo_checkpoint", "sha256_checkpoint_registrado_run_config",
                     "coincide", "estado", "invalida_integridad", "sha256_bytes_archivo_particion", "sha256_contenido_blob_config_g2")
HEX64 = re.compile(r"[0-9a-f]{64}")
ENTORNO_CLAVES = ("python", "torch", "torchvision", "numpy")


# ============================================================================ configuración
def validar_config(config) -> None:
    validar_comunes(config, ESQUEMA_CONFIG, PROTOCOLO_ID, PROTOCOLO_REL, CONFIG_REL, "src/r8_v2/congelamiento.py")
    if config.get("sistema") != SISTEMA:
        raise GuardError("sistema del YAML distinto de las constantes del código (miembros, preprocesamiento, agregación, regla, t_hat, estado, congelado: false, alcance): "
                         "se bloquea; cualquier cambio es una enmienda documentada previa al congelamiento")
    t_hat_exacto(config["sistema"]["umbral"]["t_hat"])
    if config.get("salidas") != {"verificacion": SALIDA_REL}:
        raise GuardError(f"salidas del YAML distintas de {SALIDA_REL}")
    exigir_ruta_no_prohibida(SALIDA_REL)
    pines = config.get("archivos_de_procedencia")
    if not isinstance(pines, dict) or set(pines) != set(PINES_REQUERIDOS) or not all(isinstance(v, str) and HEX64.fullmatch(v) for v in pines.values()):
        raise GuardError("archivos_de_procedencia debe fijar exactamente los archivos requeridos, cada uno con un SHA-256 de 64 hex en minúsculas")
    m = config.get("miembros")
    if not isinstance(m, dict) or m.get("archivo") != MIEMBROS_REL or not HEX64.fullmatch(str(m.get("sha256_bytes_archivo"))):
        raise GuardError(f"miembros.archivo debe ser {MIEMBROS_REL} con su SHA-256 de 64 hex")
    if m["sha256_bytes_archivo"] != pines[REL_MANIFIESTO_CSV]:
        raise GuardError("el manifiesto de miembros versionado debe ser copia byte a byte del manifiesto de D-055 (mismo SHA-256 que su archivo de procedencia)")
    huella = set(config["integridad"]["archivos_con_huella"])
    faltan = [r for r in ARCHIVOS_CODIGO if r not in huella]
    if faltan or not isinstance(config.get("entorno_registrado"), dict) or not all(k in config["entorno_registrado"] for k in ENTORNO_CLAVES):
        raise GuardError(f"integridad.archivos_con_huella debe incluir {faltan} y entorno_registrado debe fijar {ENTORNO_CLAVES}")
    if not config.get("trazabilidad") or not all(f"D-0{n}" in " ".join(map(str, config["trazabilidad"])) for n in (50, 53, 55)):
        raise GuardError("trazabilidad debe referenciar al menos D-050, D-053 y D-055 (procedencia experimental)")


def cargar_config(path) -> dict:
    return _cargar_yaml(path, validar_config)


# ============================================================================ lectura de tablas con encabezado `# ...`
def leer_tabla(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        lineas = [ln for ln in fh.read().splitlines() if ln.strip() and not ln.startswith("#")]
    return list(csv.DictReader(lineas))


def _verdadero(x, valor: bool) -> bool:
    return str(x).strip().lower() == ("true" if valor else "false")


# ============================================================================ comprobaciones
def verificar_pines(root, config) -> dict:
    """Cada archivo de procedencia debe existir y coincidir con el SHA-256 fijado en el YAML (la ausencia bloquea: se verifica donde residen, Falcon)."""
    observados = {}
    for rel, esperado in config["archivos_de_procedencia"].items():
        exigir_ruta_no_prohibida(rel)
        ruta = Path(root) / rel
        if not ruta.is_file():
            raise GuardError(f"{rel} no existe: la verificación se ejecuta donde residen los resultados de desarrollo (Falcon)")
        obtenido = sha256_archivo(ruta)
        if obtenido != esperado:
            raise GuardError(f"{rel}: SHA-256 distinto del fijado en el YAML (alterado o regenerado); se bloquea")
        observados[rel] = obtenido
    ref = config["referencia_g2"]
    return {"archivos": observados, "yaml_g2_blob_pin": ref["sha256_contenido_blob"]}


def leer_miembros(root, config) -> list[dict]:
    m = config["miembros"]
    ruta = Path(root) / m["archivo"]
    if not ruta.is_file():
        raise GuardError(f"{m['archivo']} no existe: falta la copia versionada del manifiesto de miembros")
    if sha256_archivo(ruta) != m["sha256_bytes_archivo"]:
        raise GuardError(f"{m['archivo']}: SHA-256 distinto del fijado (¿copia no byte a byte o fin de línea convertido?); se bloquea")
    filas = leer_tabla(ruta)
    if filas and any(c not in filas[0] for c in COLUMNAS_MIEMBROS):
        raise GuardError(f"{m['archivo']}: faltan columnas {[c for c in COLUMNAS_MIEMBROS if c not in filas[0]]}")
    return filas


def validar_miembros(filas: list[dict], config) -> list[dict]:
    """15 miembros únicos A|densenet121 = semillas × folds exactos, intento completado, sin invalidez, hashes bien formados, misma partición y misma configuración de G2."""
    s = config["sistema"]
    if len(filas) != N_MIEMBROS:
        raise GuardError(f"el manifiesto de miembros debe tener exactamente {N_MIEMBROS} filas (tiene {len(filas)}); no se sustituyen ni se omiten miembros")
    esperados = {(sem, k) for sem in s["semillas"] for k in s["folds"]}
    vistos, miembros = set(), []
    for f in filas:
        clave = (int(f["semilla"]), int(f["fold"]))
        if f["arquitectura"] != s["arquitectura"] or f["estado"] != "completada" or not _verdadero(f["coincide"], True) or not _verdadero(f["invalida_integridad"], False):
            raise GuardError(f"{f.get('id_corrida')}: no es un miembro completado y válido de {s['arquitectura']} (coincide=True, invalida_integridad=False)")
        if clave in vistos or clave not in esperados:
            raise GuardError(f"{f.get('id_corrida')}: miembro duplicado o fuera de semillas × folds esperados {clave}")
        vistos.add(clave)
        for col in ("sha256_bytes_archivo_checkpoint", "sha256_checkpoint_registrado_run_config", "sha256_bytes_archivo_particion", "sha256_contenido_blob_config_g2"):
            if not HEX64.fullmatch(f[col]):
                raise GuardError(f"{f['id_corrida']}: {col} no es un SHA-256 válido")
        if f["sha256_bytes_archivo_checkpoint"] != f["sha256_checkpoint_registrado_run_config"]:
            raise GuardError(f"{f['id_corrida']}: el SHA-256 del checkpoint difiere del registrado en su run_config")
        exigir_ruta_no_prohibida(f["ruta_checkpoint"])
        miembros.append({"id_corrida": f["id_corrida"], "semilla": clave[0], "fold": clave[1], "ruta_checkpoint": f["ruta_checkpoint"],
                         "sha256_bytes_archivo_checkpoint": f["sha256_bytes_archivo_checkpoint"]})
    if len({m["id_corrida"] for m in miembros}) != N_MIEMBROS or len({m["sha256_bytes_archivo_checkpoint"] for m in miembros}) != N_MIEMBROS \
            or len({m["ruta_checkpoint"] for m in miembros}) != N_MIEMBROS:
        raise GuardError("miembros duplicados (id_corrida, ruta o checkpoint repetidos)")
    if {f["sha256_bytes_archivo_particion"] for f in filas} != {config["archivos_de_procedencia"][REL_PARTICION]}:
        raise GuardError("los 15 miembros no comparten la partición fijada en el YAML")
    if {f["sha256_contenido_blob_config_g2"] for f in filas} != {config["referencia_g2"]["sha256_contenido_blob"]}:
        raise GuardError("los 15 miembros no fueron entrenados con la configuración de G2 fijada en el YAML (el pin no coincide)")
    return sorted(miembros, key=lambda m: (m["semilla"], m["fold"]))


def verificar_coherencia_resultados(root, config, miembros: list[dict]) -> dict:
    """Manifiesto JSON, fidelidad y umbral deben ser coherentes con el manifiesto versionado y con el YAML."""
    root = Path(root)
    mj = json.loads((root / REL_MANIFIESTO_JSON).read_text(encoding="utf-8"))
    if mj.get("n_miembros") != N_MIEMBROS or mj.get("advertencias") != [] or mj.get("reporte_aceptado") is not True or mj.get("invalida_integridad"):
        raise GuardError("manifiesto_miembros.json: se exigen 15 miembros, advertencias vacías, reporte aceptado y sin invalidez de integridad")
    if mj.get("sha256_bytes_archivo_manifiesto_csv") != config["archivos_de_procedencia"][REL_MANIFIESTO_CSV]:
        raise GuardError("manifiesto_miembros.json no corresponde al CSV fijado (SHA-256 del CSV distinto)")
    por_id = {m["id_corrida"]: m for m in mj.get("miembros", [])}
    for m in miembros:
        o = por_id.get(m["id_corrida"])
        if o is None or o["sha256_bytes_archivo_checkpoint"] != m["sha256_bytes_archivo_checkpoint"] or o["ruta_checkpoint"] != m["ruta_checkpoint"]:
            raise GuardError(f"{m['id_corrida']}: el manifiesto JSON y el CSV versionado no coinciden")
    fj = json.loads((root / REL_FIDELIDAD_JSON).read_text(encoding="utf-8"))
    if (fj.get("n_miembros") != N_MIEMBROS or fj.get("alerta_global") is not False or fj.get("reporte_aceptado") is not True or fj.get("invalida_integridad")
            or fj.get("t_hat") != T_HAT or fj.get("sha256_bytes_archivo_manifiesto_csv") != config["archivos_de_procedencia"][REL_MANIFIESTO_CSV]
            or fj.get("metricas_del_ensamble_de_quince_sobre_desarrollo") is not False):
        raise GuardError("resumen_fidelidad.json: se exigen 15 miembros, sin alerta de recarga, reporte aceptado, mismo manifiesto y mismo t_hat, sin métricas del ensamble")
    tabla = leer_tabla(root / REL_FIDELIDAD_CSV)
    if sorted(r["id_corrida"] for r in tabla) != sorted(m["id_corrida"] for m in miembros):
        raise GuardError("fidelidad_por_miembro.csv no cubre exactamente los 15 miembros del manifiesto")
    dif = max(float(r["diferencia_maxima_absoluta"]) for r in tabla)
    tol = float(fj["tolerancia"])
    if dif > tol:
        raise GuardError(f"diferencia máxima de recarga {dif!r} mayor que la tolerancia {tol!r}")
    uj = json.loads((root / REL_UMBRAL).read_text(encoding="utf-8"))
    c = uj["umbral_primario"]["calibracion"]
    if c.get("umbral_repr") != config["sistema"]["umbral"]["t_hat"] or uj.get("umbral_congelado") is not False:
        raise GuardError(f"t_hat del resultado de umbral ({c.get('umbral_repr')!r}) distinto del YAML ({config['sistema']['umbral']['t_hat']!r})")
    if c.get("k") != 35 or c.get("n_pos") != 38:
        raise GuardError(f"umbral_primario.json: se esperaba k = 35 de 38 positivas (k={c.get('k')!r}, n_pos={c.get('n_pos')!r})")
    return {"fidelidad_diferencia_maxima_absoluta": dif, "fidelidad_tolerancia": tol, "t_hat_resultado_umbral": c["umbral_repr"], "k": c["k"], "n_pos": c["n_pos"],
            "advertencias_manifiesto": mj["advertencias"]}


def verificar_checkpoints(root, miembros: list[dict]) -> list[dict]:
    """SHA-256 recalculado de cada checkpoint contra el del manifiesto versionado (donde residen los checkpoints)."""
    out = []
    for m in miembros:
        ruta = Path(root) / m["ruta_checkpoint"]
        if not ruta.is_file():
            raise GuardError(f"{m['id_corrida']}: checkpoint ausente ({m['ruta_checkpoint']}); la verificación se ejecuta donde residen los checkpoints (Falcon)")
        sha = sha256_archivo(ruta)
        if sha != m["sha256_bytes_archivo_checkpoint"]:
            raise GuardError(f"{m['id_corrida']}: el checkpoint no coincide con el SHA-256 del manifiesto (alterado o distinto)")
        out.append({"id_corrida": m["id_corrida"], "sha256_recalculado": sha, "tamano_bytes": ruta.stat().st_size})
    return out


def entorno_efectivo() -> dict:
    import numpy as np
    v = {"python": sys.version.split()[0], "numpy": np.__version__, "torch": None, "torchvision": None}
    for nombre in ("torch", "torchvision"):
        try:
            v[nombre] = __import__(nombre).__version__
        except ImportError:
            pass
    return v


def comparar_entorno(config, efectivo: dict) -> list[str]:
    reg = config["entorno_registrado"]
    return [f"{k}: registrado {reg.get(k)!r}, efectivo {efectivo.get(k)!r}" for k in ENTORNO_CLAVES if str(reg.get(k)) != str(efectivo.get(k))]


# ============================================================================ ejecución
def ejecutar_verificar(root, config, *, autorizacion, verificar=None, destino=None, entorno=None) -> dict:
    token = U.exigir_autorizacion(autorizacion)                                           # antes de leer o escribir nada
    root = Path(root)
    validar_config(config)
    sha_protocolo = exigir_protocolo(root, config, FRASES_PROTOCOLO)
    verificar = verificar or (lambda: verificar_integridad(root, config))
    inicio = U.exigir_integridad(verificar())
    r6 = U.verificar_r6(root, config)
    destino = exigir_salida_segura(root, Path(destino) if destino else root / config["salidas"]["verificacion"] / token.split(":")[1], "results/r8_v2")
    if destino.exists():
        raise GuardError(f"K5: el directorio de salida ya existe y no se sobrescribe: {destino}")
    pines = verificar_pines(root, config)
    miembros = validar_miembros(leer_miembros(root, config), config)
    coherencia = verificar_coherencia_resultados(root, config, miembros)
    checkpoints = verificar_checkpoints(root, miembros)
    efectivo = entorno if entorno is not None else entorno_efectivo()
    advertencias = comparar_entorno(config, efectivo)
    fin = verificar()
    dif = comparar_instantaneas(inicio, fin)
    destino = directorio_nuevo(destino)                                                   # K5: solo tras validar y calcular
    aceptado = not (bool(dif) or bool(fin["bloqueos"]))
    resumen = U.a_json({
        "esquema": "verificacion_sistema_congelado_v1/1", "protocolo_id": PROTOCOLO_ID, "leyenda": LEYENDA, "estado": ESTADO, "congelado": False,
        "nota": "esta verificación NO declara el congelamiento; la declaración formal es una decisión posterior y separada",
        "sistema": config["sistema"], "miembros": [{**m, **{"sha256_recalculado": c["sha256_recalculado"]}} for m, c in zip(miembros, checkpoints)],
        "archivos_de_procedencia_verificados": pines["archivos"], "coherencia": coherencia, "base_r6": r6, "entorno_efectivo": efectivo, "entorno_registrado": config["entorno_registrado"],
        "advertencias_entorno": advertencias, "sha256_bytes_archivo_protocolo": sha_protocolo, "protocolo_commit": config["protocolo"]["git_commit"],
        "protocolo_sha256_contenido_blob": config["protocolo"]["sha256_contenido_blob"], "sha256_bytes_archivo_config": inicio["archivos"][CONFIG_REL]["sha256_bytes_archivo"],
        "sha256_bytes_archivo_miembros_versionados": config["miembros"]["sha256_bytes_archivo"], "autorizacion": token, "git_commit_head": inicio["git_commit_head"],
        "rama": inicio["rama"], "fecha": U.ahora(), "diferencias_inicio_fin": dif, "invalida_integridad": not aceptado, "reporte_aceptado": aceptado})
    escribir_json_nuevo(destino / "resumen_verificacion.json", resumen)
    return {"dir": destino, "resumen": resumen, "reporte_aceptado": aceptado, "advertencias_entorno": advertencias}


def main(argv=None, *, root=None, config=None, verificar=None, destino=None, entorno=None) -> int:
    ap = argparse.ArgumentParser(description="sistema_congelado_v1 (R8): verificación previa a la declaración de congelamiento; solo lectura; no abre el test")
    ap.add_argument("--modo", required=True, choices=("verificar",))
    ap.add_argument("--autorizacion", default="", help="CONFIRMADO:D-NNN de ejecución; sin ella no se lee ni se escribe nada")
    args = ap.parse_args(argv)
    root = Path(root or RAIZ_REPO).resolve()
    try:
        U.exigir_autorizacion(args.autorizacion)
        config = config or cargar_config(root / CONFIG_REL)
        r = ejecutar_verificar(root, config, autorizacion=args.autorizacion, verificar=verificar, destino=destino, entorno=entorno)
        print(f"{LEYENDA}\n{ESTADO}\nsalida: {r['dir']}\nESTA VERIFICACIÓN NO DECLARA EL CONGELAMIENTO.")
        for a in r["advertencias_entorno"]:
            print(f"ADVERTENCIA de entorno: {a}", file=sys.stderr)
        if not r["reporte_aceptado"]:
            print("SALIDA NO ACEPTADA: la integridad final difiere de la inicial; no debe usarse.", file=sys.stderr)
            return U.CODIGO_SALIDA_INTEGRIDAD_INVALIDA
        return 0
    except GuardError as exc:
        print(f"DETENIDO: {exc}", file=sys.stderr)
        return U.CODIGO_SALIDA_GUARDA


if __name__ == "__main__":
    raise SystemExit(main())
