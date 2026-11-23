"""R7 `delta_ensamble_v1`: contrastes PAREADOS del ensamble de tres semillas (q_barra) frente a cada semilla individual (solo desarrollo; exploratorio).

* Entrada ÚNICA: `umbral_sistema_v1/q_barra_oof.csv` (image_id, fold, label_v2, p_s42, p_s43, p_s44, q_barra) verificado por SHA-256 contra el `trazabilidad.json` de esa carpeta,
  más `group_id` de la partición (K1-K4). Los índices `b` son los de G2 (regenerados y verificados con `verificar_indices_g2`; B = 10 000). Mismos `b` para TODAS las columnas.
* Contrastes: ΔAP(q_barra − p_s) para s = 42, 43, 44 (tres individuales; IC95 % percentil bootstrap agrupado estratificado y no estratificado; IC ajustado por multiplicidad de nivel
  1 − 0.05/3) y, solo como resumen descriptivo, ΔAP(q_barra − media de las AP individuales) con la media calculada DENTRO de cada remuestreo. Igual para ROC-AUC (secundaria).
* OOF de un ensamble de TRES modelos del mismo pliegue: no es evidencia sobre el ensamble de quince ni sobre test; no reabre R6 ni cambia la arquitectura. Sin lenguaje de contraste de
  hipótesis ni afirmación de utilidad clínica. Partición provisional (phash_le8_v1, exploratoria, post hoc); conformidad académica pendiente.

CLI (solo la autora, con commits de congelamiento y D-NNN de ejecución): python -B -X pycache_prefix=<dir> -m src.r7_v2.delta_ensamble --autorizacion CONFIRMADO:D-NNN
Este módulo aloja además la comprobación de integridad y de protocolo parametrizadas por la ruta del protocolo, que reutiliza `sistema_final`.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

from src.r4_v2 import bootstrap as BS
from src.r4_v2 import metricas as M
from src.r4_v2.common import (RAIZ_REPO, GuardError, directorio_nuevo, escribir_json_nuevo, exigir_ruta_no_prohibida, git_ok, git_texto, sha256_archivo,  # noqa: F401
                              sha256_bytes)
from src.r4_v2.common import cargar_config as cargar_config_g2
from src.r4_v2.particion import cargar_particion
from src.r5_v2.comun import comparar_instantaneas, huella_archivo
from src.r7_v2 import g4_reporte as R
from src.r7_v2 import umbral_sistema as U

PROTOCOLO_ID = "delta_ensamble_v1"
ESQUEMA_CONFIG = "delta_ensamble_v1_config/1"
CONFIG_REL = "configs/r7_v2/delta_ensamble_v1.yaml"
PROTOCOLO_REL = "src/r7_v2/protocolo_delta_ensamble_v1.md"
SALIDA_REL = "results/r7_v2/phash_le8_v1/delta_ensamble_v1"
ENTRADA_REL = "results/r7_v2/phash_le8_v1/umbral_sistema_v1"
LEYENDA = U.LEYENDA
SEMILLAS = (42, 43, 44)
ALFA, K_CONTRASTES = 0.05, 3
NIVEL_AJUSTADO = 1.0 - ALFA / K_CONTRASTES
COLUMNAS_ENTRADA = ["image_id", "fold", "label_v2", "p_s42", "p_s43", "p_s44", "q_barra"]
ROTULO_AJUSTE = "ajuste por multiplicidad; no corrige el sesgo de selección exploratoria"
ROTULO_DESCRIPTIVO = "resumen descriptivo; no es un comparador que exista como sistema"
LEYENDAS = (
    "análisis exploratorio, no confirmatorio",
    "el ensamble de tres semillas es una elección de diseño fijada antes de ver estos números (D-048); los contrastes son posteriores a esa elección",
    "la mejor semilla individual identificada mirando resultados NO es el único comparador: se reportan las tres",
    "dependencia entre modelos (mismas imágenes, mismos pliegues): los IC no asumen independencia; agrupamiento pHash ≤ 8 exploratorio, sin independencia por paciente",
    "38 positivas (≈ 7–8 por fold): incertidumbre alta",
    "es OOF de un ensamble de TRES modelos del mismo pliegue: no es evidencia sobre el ensamble de quince ni sobre test",
    "no reabre R6 ni cambia la arquitectura seleccionada; sin afirmar utilidad clínica",
    ROTULO_AJUSTE)
LIMITES = ("38 positivas (≈ 7–8 por fold)", "partición provisional (pHash ≤ 8 exploratoria y post hoc)", "desarrollo reutilizado en toda la cadena de decisiones",
           "accesos históricos al test declarados (P335); este análisis no abre el test")
PROCEDIMIENTO = {
    "entrada": "umbral_sistema_v1/q_barra_oof.csv verificado por SHA-256 contra trazabilidad.json",
    "columnas": list(COLUMNAS_ENTRADA), "metrica_primaria": "ap", "metrica_secundaria": "roc_auc",
    "contrastes_individuales": {"comparador": "p_s", "semillas": [42, 43, 44]},
    "resumen_descriptivo": "media de las metricas individuales calculada dentro de cada remuestreo; solo descriptivo",
    "bootstrap": {"n_remuestreos": 10000, "nivel_ic": 0.95, "estratificado": True, "no_estratificado": True, "indices_b": "los de G2 (verificar_indices_g2)"},
    "multiplicidad": {"k_contrastes": 3, "nivel_ajustado": "1 - 0.05/3", "percentiles": "alfa/6 y 1 - alfa/6", "rotulo": ROTULO_AJUSTE},
    "exploratorio": True, "reabre_r6": False, "cambia_arquitectura": False, "evidencia_sobre_ensamble_de_quince": False}
FRASES_PROTOCOLO = ("q_barra_oof.csv", "SHA-256", "1 − 0.05/3", "resumen descriptivo; no es un comparador que exista como sistema", "ajuste por multiplicidad",
                    "no corrige el sesgo de selección exploratoria", "tres semillas", "D-048", "OOF de un ensamble de TRES modelos del mismo pliegue", "no reabre R6",
                    "mismos índices `b`", "exploratorio", "no se afirma independencia absoluta del test histórico")
ENCABEZADO = (f"# mapeo v2 | phash_le8_v1 (provisional) | {PROTOCOLO_ID} | sin uso de test | {LEYENDA} | análisis exploratorio, no confirmatorio | "
              "OOF de ensamble de TRES modelos del mismo pliegue: no es evidencia sobre el ensamble de quince ni sobre test | no reabre R6 | sin afirmar utilidad clínica")


# ============================================================================ validación e integridad (parametrizadas por la ruta del protocolo)
def cargar_config(path, validar=None) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise GuardError("Se requiere PyYAML") from exc
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    (validar or validar_config)(config)
    return config


def validar_comunes(config, esquema, protocolo_id, protocolo_rel, config_rel, modulo_rel) -> None:
    if not isinstance(config, dict) or config.get("esquema") != esquema:
        raise GuardError(f"Configuración con esquema distinto de {esquema!r}")
    if config["protocolo"]["id"] != protocolo_id or config["protocolo"]["ruta"] != protocolo_rel:
        raise GuardError(f"La configuración no corresponde a {protocolo_id}")
    if config["estudio"]["particion"] != "phash_le8_v1" or config["estudio"]["mapeo"] != "v2":
        raise GuardError("La configuración no corresponde a phash_le8_v1 / mapeo v2")
    if (config.get("acceso_test") or {}).get("estado") != "cerrado":
        raise GuardError("acceso_test debe ser 'cerrado'")
    if config["referencia_r6"]["elegida"] != {"condicion": "A", "arquitectura": "densenet121"}:
        raise GuardError("referencia_r6.elegida debe ser A|densenet121")
    huella = set(config["integridad"]["archivos_con_huella"])
    for rel in (config_rel, protocolo_rel, modulo_rel, config["referencia_g2"]["config"]):
        if rel not in huella:
            raise GuardError(f"integridad.archivos_con_huella debe incluir {rel}")


def validar_config(config) -> None:
    validar_comunes(config, ESQUEMA_CONFIG, PROTOCOLO_ID, PROTOCOLO_REL, CONFIG_REL, "src/r7_v2/delta_ensamble.py")
    if config.get("procedimiento") != PROCEDIMIENTO:
        raise GuardError("procedimiento del YAML distinto de las constantes del código (contrastes, bootstrap, multiplicidad, exploratorio, R6): se bloquea; "
                         "cualquier cambio es una enmienda previa al congelamiento")
    if config.get("salidas") != {"delta": SALIDA_REL} or config.get("entrada_dir") != ENTRADA_REL:
        raise GuardError("salidas/entrada del YAML distintas de results/r7_v2/phash_le8_v1/{umbral_sistema_v1 -> delta_ensamble_v1}")
    for ruta in (SALIDA_REL, ENTRADA_REL):
        exigir_ruta_no_prohibida(ruta)


def exigir_protocolo(root, config, frases) -> str:
    """El protocolo versionado debe contener las frases clave (anti-deriva) y NO su propio hash (sin circularidad). Devuelve su SHA-256 de bytes."""
    ruta = Path(root) / config["protocolo"]["ruta"]
    if not ruta.is_file():
        raise GuardError(f"{config['protocolo']['ruta']} no existe: no se puede validar la coherencia del procedimiento")
    texto = ruta.read_text(encoding="utf-8")
    faltan = [x for x in frases if x not in texto]
    if faltan or re.search(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", texto):
        raise GuardError(f"el protocolo no es coherente con el procedimiento del código (faltan frases {faltan}) o contiene un hash de 64 hex (no debe tener hash propio)")
    return sha256_archivo(ruta)


def verificar_integridad(root, config) -> dict:
    """HEAD, árbol limpio en `rutas_sin_cambios`, SHA-256 de `archivos_con_huella`, pin del YAML de G2 y PROTOCOLO CONGELADO (commit y SHA-256 registrados en el YAML, iguales
    al blob de HEAD y al del commit registrado). Igual que `umbral_sistema.verificar_integridad` pero con la ruta del protocolo de ESTA configuración. NO es K6."""
    ig, p = config["integridad"], config["protocolo"]
    head = git_texto(root, "rev-parse", "HEAD", obligatorio=True)
    estado = git_ok(root, "status", "--porcelain", "--untracked-files=all", "--", *ig["rutas_sin_cambios"]).decode("utf-8", "replace")
    status = [l for l in estado.splitlines() if l.strip()]
    bloqueos = [{"codigo": "arbol_sucio", "detalle": l} for l in status]
    archivos = {}
    for rel in ig["archivos_con_huella"]:
        reg = huella_archivo(root, rel)
        archivos[rel] = reg
        if reg["sha256_bytes_archivo"] is None:
            bloqueos.append({"codigo": "archivo_ausente", "detalle": rel})
        elif reg["sha256_contenido_blob"] is None:
            bloqueos.append({"codigo": "archivo_no_versionado", "detalle": rel})
    ref = config["referencia_g2"]
    if archivos[ref["config"]]["sha256_contenido_blob"] != ref["sha256_contenido_blob"]:
        bloqueos.append({"codigo": "yaml_referencia_g2_distinto_del_fijado", "detalle": ref["config"]})
    blob_head = archivos[p["ruta"]]["sha256_contenido_blob"]
    if not p.get("git_commit") or not p.get("sha256_contenido_blob"):
        bloqueos.append({"codigo": "protocolo_no_congelado", "detalle": "el YAML no registra commit y SHA-256 del protocolo (commit de congelamiento pendiente)"})
    else:
        if blob_head != p["sha256_contenido_blob"]:
            bloqueos.append({"codigo": "protocolo_distinto_del_registrado", "detalle": f"blob de HEAD {blob_head} != YAML"})
        try:
            en_commit = sha256_bytes(git_ok(root, "cat-file", "blob", f"{p['git_commit']}:{p['ruta']}"))
        except GuardError:
            en_commit = None
        if en_commit != p["sha256_contenido_blob"]:
            bloqueos.append({"codigo": "protocolo_del_commit_registrado_distinto", "detalle": f"commit {p['git_commit']}"})
    return {"git_commit_head": head, "rama": git_texto(root, "rev-parse", "--abbrev-ref", "HEAD"), "rutas_sin_cambios": list(ig["rutas_sin_cambios"]),
            "status_porcelain": status, "archivos": archivos, "bloqueos": bloqueos}


# ============================================================================ entrada verificada
def leer_entrada(origen: Path, filas) -> dict:
    """q_barra_oof.csv verificado por SHA-256 contra `trazabilidad.json`; columnas exactas, una fila por imagen, mismas imágenes/orden/fold/label_v2 que la partición."""
    import json
    tz, qc = Path(origen) / "trazabilidad.json", Path(origen) / "q_barra_oof.csv"
    if not tz.is_file() or not qc.is_file():
        raise GuardError("falta q_barra_oof.csv o trazabilidad.json de umbral_sistema_v1 (se ejecuta primero `umbral_sistema`)")
    traz = json.loads(tz.read_text(encoding="utf-8"))
    sha = sha256_archivo(qc)
    if sha != (traz.get("sha256_bytes_archivo") or {}).get("q_barra_oof.csv"):
        raise GuardError("q_barra_oof.csv no coincide con el SHA-256 de trazabilidad.json; no se usa")
    lineas = qc.read_text(encoding="utf-8").splitlines()
    lector = csv.DictReader(lineas[1:] if lineas and lineas[0].startswith("#") else lineas)
    if list(lector.fieldnames or []) != COLUMNAS_ENTRADA:
        raise GuardError(f"q_barra_oof.csv: columnas {list(lector.fieldnames or [])} distintas de {COLUMNAS_ENTRADA}")
    datos = list(lector)
    ids = [d["image_id"] for d in datos]
    if len(set(ids)) != len(ids):
        raise GuardError("q_barra_oof.csv: image_id duplicados (un registro por imagen)")
    if ids != [r["image_id"] for r in filas] or [int(d["fold"]) for d in datos] != [r["fold"] for r in filas] or [int(d["label_v2"]) for d in datos] != [r["label_v2"] for r in filas]:
        raise GuardError("q_barra_oof.csv no corresponde a la partición (imágenes, orden, fold o label_v2 distintos; deben ser las mismas imágenes)")
    try:
        cols = {k: np.array([float(d[k]) for d in datos], dtype=np.float64) for k in ("p_s42", "p_s43", "p_s44", "q_barra")}
    except ValueError:
        raise GuardError("q_barra_oof.csv: valores no numéricos") from None
    if any(not np.isfinite(v).all() or ((v < 0) | (v > 1)).any() for v in cols.values()):
        raise GuardError("q_barra_oof.csv: probabilidades no finitas o fuera de [0, 1]")
    return {"columnas": cols, "sha256_bytes_archivo": sha, "trazabilidad": traz, "sha256_trazabilidad": sha256_archivo(tz)}


# ============================================================================ cálculo puro (mismos b para todas las columnas)
def ic_pareado(d, nivel: float = 0.95) -> dict:
    r = R.ic(d, nivel)
    r["ic_excluye_cero"] = bool(r["n_remuestreos_validos"] > 0 and (r["ic_inf"] > 0 or r["ic_sup"] < 0))
    return r


def contrastes_metrica(dist_e: dict, dist_n: dict, puntual: dict, metrica: str) -> list:
    """Tres contrastes individuales (q_barra − p_s) y el resumen descriptivo (q_barra − media dentro de cada remuestreo de las métricas individuales).
    `dist_e`/`dist_n`: distribuciones con los MISMOS `b` (estratificado / no estratificado); claves "q" y "p_s42", "p_s43", "p_s44"."""
    filas = []
    claves = [f"p_s{s}" for s in SEMILLAS]
    for s, k in zip(SEMILLAS, claves):
        de, dn = dist_e["q"][metrica] - dist_e[k][metrica], dist_n["q"][metrica] - dist_n[k][metrica]
        filas.append({"contraste": f"q_barra - p_s{s}", "tipo": "individual", "semilla": s, "valor_q_barra": float(puntual["q"][metrica]), "valor_comparador": float(puntual[k][metrica]),
                      "estimacion": float(puntual["q"][metrica] - puntual[k][metrica]), "estratificado": ic_pareado(de), "no_estratificado": ic_pareado(dn),
                      "ajustado_estratificado": {"nivel": NIVEL_AJUSTADO, **ic_pareado(de, NIVEL_AJUSTADO)}, "ajustado_no_estratificado": {"nivel": NIVEL_AJUSTADO, **ic_pareado(dn, NIVEL_AJUSTADO)},
                      "rotulo": ROTULO_AJUSTE})
    me = np.mean([dist_e[k][metrica] for k in claves], axis=0)                           # la media se calcula DENTRO de cada remuestreo
    mn = np.mean([dist_n[k][metrica] for k in claves], axis=0)
    media_puntual = float(np.mean([puntual[k][metrica] for k in claves]))
    filas.append({"contraste": "q_barra - media de las individuales", "tipo": "resumen descriptivo", "semilla": None, "valor_q_barra": float(puntual["q"][metrica]),
                  "valor_comparador": media_puntual, "estimacion": float(puntual["q"][metrica] - media_puntual), "estratificado": ic_pareado(dist_e["q"][metrica] - me),
                  "no_estratificado": ic_pareado(dist_n["q"][metrica] - mn), "ajustado_estratificado": None, "ajustado_no_estratificado": None, "rotulo": ROTULO_DESCRIPTIVO})
    return filas


def analizar(y, columnas: dict, gpi, cont_e, cont_n) -> dict:
    y = np.asarray(y)
    vec = {"q": columnas["q_barra"], **{f"p_s{s}": columnas[f"p_s{s}"] for s in SEMILLAS}}
    preps = {k: M.preparar(y, v) for k, v in vec.items()}
    dist_e, dist_n = BS.distribuciones(preps, cont_e, gpi), BS.distribuciones(preps, cont_n, gpi)
    puntual = {k: M.calcular(p) for k, p in preps.items()}
    return {m: contrastes_metrica(dist_e, dist_n, puntual, m) for m in ("ap", "roc_auc")}


# ============================================================================ salidas
def _fila(c) -> dict:
    out = {"contraste": c["contraste"], "tipo": c["tipo"], "semilla": "" if c["semilla"] is None else c["semilla"], "valor_q_barra": U.f(c["valor_q_barra"]),
           "valor_comparador": U.f(c["valor_comparador"]), "estimacion": U.f(c["estimacion"])}
    for nom in ("estratificado", "no_estratificado"):
        d = c[nom]
        out.update({f"{nom}_ic95_inf": U.f(d["ic_inf"]), f"{nom}_ic95_sup": U.f(d["ic_sup"]), f"{nom}_n_remuestreos_validos": d["n_remuestreos_validos"],
                    f"{nom}_ic_excluye_cero": d["ic_excluye_cero"]})
    for nom in ("ajustado_estratificado", "ajustado_no_estratificado"):
        d = c[nom]
        out.update({f"{nom}_nivel": U.f(d["nivel"]) if d else "", f"{nom}_ic_inf": U.f(d["ic_inf"]) if d else "", f"{nom}_ic_sup": U.f(d["ic_sup"]) if d else "",
                    f"{nom}_ic_excluye_cero": d["ic_excluye_cero"] if d else ""})
    out["rotulo"] = c["rotulo"]
    return out


def ejecutar(root, config, config_g2, *, autorizacion, verificar=None, n_remuestreos=None, destino=None, origen=None) -> dict:
    token = U.exigir_autorizacion(autorizacion)                                          # antes de leer o escribir nada
    root = Path(root)
    validar_config(config)
    if config["estudio"]["particion"] != config_g2["estudio"]["particion"]:
        raise GuardError("la partición no coincide con la de G2")
    sha_protocolo = exigir_protocolo(root, config, FRASES_PROTOCOLO)
    verificar = verificar or (lambda: verificar_integridad(root, config))
    inicio = U.exigir_integridad(verificar())
    r6 = U.verificar_r6(root, config)
    destino = U.destino_nuevo(root, config["salidas"]["delta"], destino)
    origen = Path(origen) if origen else root / config["entrada_dir"]
    particion = cargar_particion(root, config_g2)                                        # K1-K4
    ent = leer_entrada(origen, particion.filas)
    y = np.array([r["label_v2"] for r in particion.filas])
    gpi, cont_e, indices, n_b = U.indices_b(root, config, config_g2, [r["group_id"] for r in particion.filas], y, n_remuestreos)
    _, clase, _ = BS.indexar_grupos([r["group_id"] for r in particion.filas], y)
    cont_n = BS.generar_conteos(clase, n_b, config_g2["bootstrap"]["semilla"], estratificado=False)
    res = analizar(y, ent["columnas"], gpi, cont_e, cont_n)
    destino = directorio_nuevo(destino)                                                  # K5: solo tras validar y calcular
    for nombre, m in (("delta_ap.csv", "ap"), ("delta_roc_auc.csv", "roc_auc")):
        filas = [_fila(c) for c in res[m]]
        U.escribir_tabla_con(destino / nombre, filas, ENCABEZADO, f"{'AP (primaria)' if m == 'ap' else 'ROC-AUC (secundaria)'}; mismos b de G2; IC95 % percentil; {ROTULO_AJUSTE}")
    escribir_json_nuevo(destino / "trazabilidad.json", U.a_json({
        "esquema": "trazabilidad_delta_ensamble_v1/1", "protocolo_id": PROTOCOLO_ID, "leyenda": LEYENDA, "fecha": U.ahora(), "autorizacion": token,
        "git_commit_head": inicio["git_commit_head"], "rama": inicio["rama"], "sha256_bytes_archivo_particion": particion.sha256_bytes_archivo,
        "sha256_bytes_archivo_protocolo": sha_protocolo, "base_r6": r6, "numpy_version": np.__version__,
        "entradas": {"q_barra_oof.csv": ent["sha256_bytes_archivo"], "trazabilidad.json (umbral_sistema_v1)": ent["sha256_trazabilidad"]},
        "git_commit_head_umbral_sistema": ent["trazabilidad"].get("git_commit_head")}))
    fin = verificar()
    dif = comparar_instantaneas(inicio, fin)
    escribir_json_nuevo(destino / "resumen_delta_ensamble.json", U.a_json({
        "esquema": "resumen_delta_ensamble_v1/1", "protocolo_id": PROTOCOLO_ID, "leyenda": LEYENDA, "leyendas": list(LEYENDAS), "limites": list(LIMITES),
        "banderas": {"exploratorio": True, "reabre_r6": False, "cambia_arquitectura": False, "evidencia_sobre_ensamble_de_quince": False, "evidencia_sobre_test": False,
                     "utilidad_clinica_afirmada": False, "mejor_semilla_individual_es_el_unico_comparador": False},
        "multiplicidad": {"k_contrastes": K_CONTRASTES, "nivel_ajustado": NIVEL_AJUSTADO, "percentiles": [ALFA / (2 * K_CONTRASTES) * 100, (1 - ALFA / (2 * K_CONTRASTES)) * 100],
                          "rotulo": ROTULO_AJUSTE}, "n_remuestreos": n_b, "indices_b_verificacion": indices, "resultados": res, "numpy_version": np.__version__,
        "diferencias_inicio_fin": dif, "invalida_integridad": bool(dif) or bool(fin["bloqueos"]), "reporte_aceptado": not (bool(dif) or bool(fin["bloqueos"])),
        "sha256_bytes_archivo": {p.name: sha256_archivo(p) for p in sorted(destino.iterdir()) if p.suffix in (".csv", ".json") and p.name != "trazabilidad.json"}}))
    return {"dir": destino, "resultados": res, "reporte_aceptado": not (bool(dif) or bool(fin["bloqueos"])), "diferencias_inicio_fin": dif}


def main(argv=None, *, root=None, config=None, config_g2=None, verificar=None, n_remuestreos=None, destino=None, origen=None) -> int:
    ap = argparse.ArgumentParser(description="delta_ensamble_v1 (R7): contrastes pareados q_barra vs semillas; exploratorio; partición provisional")
    ap.add_argument("--autorizacion", default="", help="CONFIRMADO:D-NNN de ejecución; sin ella no se lee ni se escribe nada")
    args = ap.parse_args(argv)
    root = Path(root or RAIZ_REPO).resolve()
    try:
        U.exigir_autorizacion(args.autorizacion)
        config = config or cargar_config(root / CONFIG_REL)
        config_g2 = config_g2 or cargar_config_g2(root / config["referencia_g2"]["config"])
        r = ejecutar(root, config, config_g2, autorizacion=args.autorizacion, verificar=verificar, n_remuestreos=n_remuestreos, destino=destino, origen=origen)
        print(f"{LEYENDA}\nanálisis exploratorio; no reabre R6\nsalida: {r['dir']}")
        if not r["reporte_aceptado"]:
            print("SALIDA NO ACEPTADA: la integridad final difiere de la inicial; no debe usarse.", file=sys.stderr)
            return U.CODIGO_SALIDA_INTEGRIDAD_INVALIDA
        return 0
    except GuardError as exc:
        print(f"DETENIDO: {exc}", file=sys.stderr)
        return U.CODIGO_SALIDA_GUARDA


if __name__ == "__main__":
    sys.exit(main())
