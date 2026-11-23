"""Utilidades comunes de G4 `g4_optimizacion_v1`: configuración, estrategias, integridad propia, base de G2 y promoción (fase A -> B).

Solo IMPORTA de G2 (`src.r4_v2`) y G3b (`src.r5_v2.g3b_comun`); no los modifica. La receta de clasificación es la de G2 (YAML de G2 en solo lectura, fijado
por blob): la única variable declarada entre la base y cada estrategia es la pérdida de entrenamiento. Sin torch en este módulo. Test cerrado (K4).
Partición provisional (phash_le8_v1, exploratoria, post hoc); conformidad académica pendiente.
"""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

from src.r4_v2 import agregacion as AG
from src.r4_v2 import metricas as M
from src.r4_v2.common import (RAIZ_REPO, GuardError, exigir_ruta_no_prohibida, exigir_salida_segura, git_ok, git_texto,  # noqa: F401
                              sha256_archivo, sha256_bytes)
from src.r4_v2.entrenamiento import corrida_completada, id_corrida, intentos_de  # noqa: F401
from src.r5_v2.comun import huella_archivo
from src.r5_v2.g3b_comun import _canon, comparar_con_A

PROTOCOLO_ID = "g4_optimizacion_v1"
ESQUEMA_CONFIG = "g4_optimizacion_v1_config/1"
CONFIG_REL = "configs/r7_v2/g4_optimizacion_v1.yaml"
PROTOCOLO_REL = "src/r7_v2/protocolo_g4_v1.md"
LEYENDA = "partición provisional (phash_le8_v1, exploratoria, post hoc); conformidad académica pendiente"
MODOS = ("humo", "fase_a", "promocion", "fase_b", "fase_c", "fase_c_confirmacion")
AUTORIZACION = re.compile(r"CONFIRMADO:D-[0-9]{3}")
ID_CONFIG = re.compile(r"S[12]_[ag][0-9]{3}")
# Estrategias de la fase A en ORDEN FIJO (el YAML debe coincidir exactamente; no se salta ni se reordena)
ESTRATEGIAS_FASE_A = (("S1_a050", "S1", 0.5), ("S2_g200", "S2", 2.0), ("S1_a025", "S1", 0.25), ("S2_g100", "S2", 1.0),
                      ("S1_a075", "S1", 0.75), ("S2_g050", "S2", 0.5))
LIMITES_FAMILIA = {"S1": (0.0, 1.0), "S2": (0.0, 3.0)}
VALOR_DE_LA_BASE = {"S1": 1.0, "S2": 0.0}          # alpha = 1 (pesos de G2) y gamma = 0 (CE ponderada) equivalen a la base


# ---------------------------------------------------------------- autorización y estrategias
def exigir_autorizacion(valor, flag="--autorizacion-ejecucion") -> str:
    if not isinstance(valor, str) or not AUTORIZACION.fullmatch(valor):
        raise GuardError(f"cada modo exige {flag} con formato {AUTORIZACION.pattern!r}; no se otorga ni se infiere")
    return valor


def id_de(familia: str, valor: float) -> str:
    return f"{familia}_{'a' if familia == 'S1' else 'g'}{round(valor * 100):03d}"


def estrategia_de(familia: str, valor: float, orden=None) -> dict:
    """S1: CE ponderada con alpha=valor. S2: Focal ponderada con gamma=valor y pesos alpha=1.0 (los de G2)."""
    if familia not in LIMITES_FAMILIA:
        raise GuardError(f"familia desconocida: {familia!r}")
    lo, hi = LIMITES_FAMILIA[familia]
    if isinstance(valor, bool) or not isinstance(valor, (int, float)) or not lo <= valor <= hi or round(valor, 2) != valor:
        raise GuardError(f"valor {valor!r} fuera de [{lo}, {hi}] o con más de 2 decimales para la familia {familia}")
    return {"id": id_de(familia, valor), "familia": familia, "orden": orden, "alpha": float(valor) if familia == "S1" else 1.0,
            "gamma": None if familia == "S1" else float(valor)}


def perdida_de(estrategia: dict) -> dict:
    """Bloque `receta.perdida_entrenamiento` EFECTIVO de la estrategia (queda en el run_config de G2)."""
    if estrategia["familia"] == "S1":
        return {"tipo": "cross_entropy_ponderada", "alpha_pesos_de_clase": estrategia["alpha"]}
    return {"tipo": "focal_ponderada", "alpha_pesos_de_clase": 1.0, "gamma": estrategia["gamma"]}


def estrategias_fase_a(config) -> list:
    return [dict(estrategia_de(f, v, orden=i), id=cid) for i, (cid, f, v) in enumerate(ESTRATEGIAS_FASE_A, start=1)]


def seleccionar_fase_a(config, hasta=None) -> list:
    """Primeras `hasta` estrategias en el orden fijo (por defecto las 6)."""
    todas = estrategias_fase_a(config)
    n = len(todas) if hasta is None else hasta
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= len(todas):
        raise GuardError(f"--hasta debe ser un entero entre 1 y {len(todas)} (recibido {hasta!r})")
    return todas[:n]


# ---------------------------------------------------------------- reglas de decisión (leídas del YAML y validadas contra constantes y protocolo)
S_STAR = "0.90"
NIVEL_IC, ALFA_MULTIPLICIDAD, SEMILLAS_VETO_FORMAL = 0.95, 0.05, 3
CLASES_PUNTO_OPERACION = ("deterioro respaldado", "mejora respaldada", "no distinguible")
ETIQUETA_MULTIPLICIDAD = ("respaldo exploratorio en desarrollo; el ajuste no corrige el sesgo de la búsqueda adaptativa ni es confirmación independiente")
LEYENDAS_FIJAS = (
    "(1) mejora de discriminación frente a la base, (2) comportamiento en el punto de operación exploratorio y (3) aptitud clínica son cosas distintas; "
    "la aptitud clínica NO puede establecerse con estos experimentos (viabilidad clínica: no determinada)",
    "la ausencia de veto NO se interpreta como equivalencia, no inferioridad ni aptitud clínica",
    "el IC95 % del veto no está ajustado: tiene la limitación de multiplicidad",
    "S* = 0.90 es un objetivo exploratorio, no una garantía; no se fija ningún mínimo absoluto de especificidad ni máximo de FP",
    "la transferencia entre folds es una estimación en desarrollo, no validación independiente; no es aptitud clínica",
    "no se afirma independencia absoluta del test histórico")
ORDEN_PREDEFINIDO = tuple(e[0] for e in ESTRATEGIAS_FASE_A)
REGLA_SELECCION = ("entre las configuraciones con 3 semillas completas que cumplen (i) mejora mejor respaldada en AP y (ii) no tienen deterioro respaldado, "
                   "se selecciona la de mayor delta AP medio frente a la base (3 semillas, AP sin redondear); empate exacto: orden predefinido "
                   "(S1_a050, S2_g200, S1_a025, S2_g100, S1_a075, S2_g050; luego las propuestas del agente por número de propuesta); "
                   "sin elegibles: se conserva la base")
REGLA_CONFIRMACION = ("promoción exploratoria, no evidencia confirmatoria: para confirmar una propuesta del agente con las semillas 43 y 44 su delta AP en la "
                      "semilla 42 debe ser positivo frente a la base Y frente a la candidata convencional que refina; si varias cumplen se confirma UNA "
                      "(presupuesto de 10 intentos): la de mayor delta AP frente a la candidata convencional en la semilla 42; empate exacto: menor número de propuesta")
FRASES_PROTOCOLO = ("S* = 0.90", "deterioro respaldado", "1 − 0.05/K", "sin veto formal", "5n + 10", "tope duro de 75 intentos", "viabilidad clínica: no determinada",
                    "atribuible_al_agente", "hasta 75 corridas nuevas", "configuracion_final_seleccionada", "promoción exploratoria",
                    "mayor delta AP medio frente a la base", "menor número de propuesta", "mejora mejor respaldada", "no distinguible", "no se afirma independencia absoluta del test histórico")


def validar_reglas(config) -> dict:
    """El bloque `reglas_de_decision` del YAML debe coincidir con las constantes del código (anti-deriva); se devuelve para su uso."""
    r = config.get("reglas_de_decision")
    try:
        ok = (r["s_star"] == S_STAR and r["punto_de_operacion"]["nivel_ic"] == NIVEL_IC and r["veto"]["formal_solo_con_semillas"] == SEMILLAS_VETO_FORMAL
              and r["veto"]["prevalece_sobre_mejora"] is True and r["veto"]["ic_sin_ajustar"] is True and r["veto"]["clasificacion"] == list(CLASES_PUNTO_OPERACION)
              and r["multiplicidad"]["alfa"] == ALFA_MULTIPLICIDAD and r["multiplicidad"]["nivel_ajustado"] == "1 - 0.05/K"
              and r["multiplicidad"]["etiqueta"] == ETIQUETA_MULTIPLICIDAD
              and r["reemplazo_base"]["requiere"] == ["mejor_respaldada_ap", "sin_deterioro_respaldado"]
              and r["agente"]["compara_con"] == ["base", "candidata_convencional_de_la_familia"]
              and r["seleccion_final"] == {"regla": REGLA_SELECCION, "orden_predefinido": list(ORDEN_PREDEFINIDO), "campo": "configuracion_final_seleccionada"}
              and r["agente"]["promocion_a_confirmacion"] == {"regla": REGLA_CONFIRMACION, "naturaleza": "exploratoria", "semilla": 42, "una_sola": True}
              and r["criterios_absolutos"] == {"especificidad_minima": None, "fp_maximo": None, "viabilidad_clinica": "no_determinada"})
    except (KeyError, TypeError):
        ok = False
    if not ok:
        raise GuardError("reglas_de_decision del YAML ausentes o distintas de las constantes del código (S*, IC, veto, multiplicidad, reemplazo, agente, "
                         "criterios absolutos): se bloquea; cualquier cambio es una enmienda previa al congelamiento")
    return r


def exigir_protocolo_coherente(root) -> str:
    """El protocolo versionado debe contener las frases clave de las reglas (anti-deriva) y NO su propio hash (sin circularidad)."""
    ruta = Path(root) / PROTOCOLO_REL
    if not ruta.is_file():
        raise GuardError(f"{PROTOCOLO_REL} no existe: no se puede validar la coherencia de las reglas")
    texto = ruta.read_text(encoding="utf-8")
    faltan = [f for f in FRASES_PROTOCOLO if f not in texto]
    if faltan or re.search(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", texto):
        raise GuardError(f"el protocolo no es coherente con las reglas del código (faltan frases {faltan}) o contiene un hash de 64 hex (no debe tener hash propio)")
    return sha256_archivo(ruta)


def orden_de_configuraciones(propuestas=()) -> dict:
    """Orden predefinido: las seis de la fase A y después las propuestas del agente por número de propuesta (`propuesta_NN.json`)."""
    orden = {cid: i for i, cid in enumerate(ORDEN_PREDEFINIDO)}
    for p in propuestas:
        orden[p["estrategia"]["id"]] = len(ORDEN_PREDEFINIDO) + int(p["archivo"][len("propuesta_"):-len(".json")])
    return orden


def seleccionar_configuracion_final(comparaciones, orden: dict) -> dict:
    """Selección determinista (no altera (i), (ii) ni el veto; solo ordena a las que ya cumplen ambos): entre las comparaciones frente a la base con 3 semillas
    cuya decisión es «puede reemplazar a la base», la de mayor delta AP medio (sin redondear); empate exacto: orden predefinido; sin elegibles: la base."""
    cumplen = [{"config_id": c["x"], "delta_ap_medio": float(c["delta_ap"]["estimacion"]), "orden": orden[c["x"]]} for c in comparaciones
               if c["y"] == "BASE" and sorted(c["semillas"]) == [42, 43, 44] and c["decision_reemplazo"]["decision"] == "puede reemplazar a la base"]
    cumplen.sort(key=lambda t: (-t["delta_ap_medio"], t["orden"]))
    empate = len(cumplen) > 1 and cumplen[0]["delta_ap_medio"] == cumplen[1]["delta_ap_medio"]
    return {"configuracion_final_seleccionada": cumplen[0]["config_id"] if cumplen else "BASE (DenseNet121 + A)", "conserva_la_base": not cumplen,
            "regla": REGLA_SELECCION, "cumplen_ordenadas": cumplen,
            "criterio_desempate_usado": "orden predefinido (empate exacto en delta AP medio)" if empate else None}


def elegir_confirmacion(delta_base: dict, delta_conv: dict, orden_propuestas: list) -> dict:
    """Promoción EXPLORATORIA de una propuesta a la confirmación (semillas 43 y 44): delta AP de la semilla 42 > 0 frente a la base Y frente a la candidata
    convencional. Con varias, UNA: la de mayor delta frente a la candidata; empate exacto: menor número de propuesta (posición en `orden_propuestas`)."""
    aut = [i for i in orden_propuestas if delta_base[i] > 0 and delta_conv[i] > 0]
    aut_ord = sorted(aut, key=lambda i: (-delta_conv[i], orden_propuestas.index(i)))
    empate = len(aut_ord) > 1 and delta_conv[aut_ord[0]] == delta_conv[aut_ord[1]]
    return {"autorizadas": aut_ord, "elegida": aut_ord[0] if aut_ord else None, "desempate_por_numero_de_propuesta": bool(empate),
            "regla": REGLA_CONFIRMACION, "naturaleza": "exploratoria; no es evidencia confirmatoria"}


# ---------------------------------------------------------------- configuración
def cargar_config_g4(path) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise GuardError("Se requiere PyYAML para leer configs/r7_v2/g4_optimizacion_v1.yaml") from exc
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    validar_config(config)
    return config


def validar_config(config) -> None:
    """Invariantes que el código no deja cambiar por YAML: estrategias y orden, tope, reparto, base, K4 y acceso al test cerrado."""
    from .g4_presupuesto import PLANIFICADAS_POR_FASE, TOPE_INTENTOS
    if not isinstance(config, dict) or config.get("esquema") != ESQUEMA_CONFIG:
        raise GuardError(f"Configuración con esquema distinto de {ESQUEMA_CONFIG!r}")
    if config["protocolo"]["id"] != PROTOCOLO_ID or config["protocolo"]["ruta"] != PROTOCOLO_REL:
        raise GuardError("La configuración no corresponde a g4_optimizacion_v1")
    if config["estudio"]["particion"] != "phash_le8_v1" or config["estudio"]["mapeo"] != "v2":
        raise GuardError("La configuración no corresponde a phash_le8_v1 / mapeo v2")
    if (config.get("acceso_test") or {}).get("estado") != "cerrado":
        raise GuardError("acceso_test debe ser 'cerrado'")
    b = config["base"]
    if (b["condicion"], b["arquitectura"], b["semillas"], b["folds"]) != ("A", "densenet121", [42, 43, 44], [0, 1, 2, 3, 4]) \
            or config["diseno"]["folds"] != b["folds"]:
        raise GuardError("base: debe ser A|densenet121, semillas [42, 43, 44] y folds [0..4] (15 corridas de G2)")
    esperado = [{"orden": i, "id": cid, "familia": f, "alpha": v if f == "S1" else 1.0, "gamma": None if f == "S1" else v}
                for i, (cid, f, v) in enumerate(ESTRATEGIAS_FASE_A, start=1)]
    if [{k: e.get(k) for k in ("orden", "id", "familia", "alpha", "gamma")} for e in config["estrategias"]] != esperado:
        raise GuardError("estrategias: el YAML debe contener exactamente las seis del protocolo y en su orden fijo")
    p = config["presupuesto"]
    if p["tope_intentos"] != TOPE_INTENTOS or p["planificadas"] != PLANIFICADAS_POR_FASE or sum(p["planificadas"].values()) != TOPE_INTENTOS:
        raise GuardError(f"presupuesto: tope {TOPE_INTENTOS} y reparto A=30/B=20/C=25 son fijos (se cambian solo por enmienda)")
    d = config["diseno"]
    exigir_ruta_no_prohibida(d["raiz_resultados"])
    if not str(d["raiz_resultados"]).startswith("results/r7_v2/"):
        raise GuardError("raiz_resultados de G4 debe estar bajo results/r7_v2/ (ruta nueva; sin reutilizar históricos)")
    if d["fase_a_semilla"] != 42 or d["semillas_confirmacion"] != [43, 44]:
        raise GuardError("fase A solo con la semilla 42; confirmación con 43 y 44")
    if d["humo"]["config_por_defecto"] not in {e[0] for e in ESTRATEGIAS_FASE_A}:
        raise GuardError("humo.config_por_defecto debe ser una de las seis estrategias")
    if config["promocion"]["max_promovidas"] != 2:
        raise GuardError("promocion.max_promovidas debe ser 2")
    validar_reglas(config)
    huella = set(config["integridad"]["archivos_con_huella"])
    for rel in (CONFIG_REL, PROTOCOLO_REL, config["referencia_g2"]["config"]):
        if rel not in huella:
            raise GuardError(f"integridad.archivos_con_huella debe incluir {rel}")


def comprobar_coherencia(config, config_g2) -> None:
    if [int(f) for f in config_g2["particion"]["verificaciones_bloqueantes"]["folds"]] != config["diseno"]["folds"]:
        raise GuardError("los folds de G4 no coinciden con los de G2")
    if config["estudio"]["particion"] != config_g2["estudio"]["particion"]:
        raise GuardError("la partición de G4 no coincide con la de G2")
    if config["base"]["arquitectura"] not in config_g2["diseno"]["arquitecturas"] or not set(config["base"]["semillas"]) <= set(config_g2["diseno"]["semillas"]):
        raise GuardError("la base de G4 no está dentro del diseño de G2")


def config_g2_g4(config_g2, config, estrategia, semillas) -> dict:
    """Copia de la configuración de G2 con la pérdida de la estrategia; todo lo demás (receta, partición, validación sin ponderar) es EXACTAMENTE G2."""
    c = copy.deepcopy(config_g2)
    c["estudio"]["condicion"] = f"A+{estrategia['id']}"
    c["diseno"]["arquitecturas"], c["diseno"]["semillas"] = [config["base"]["arquitectura"]], list(semillas)
    c["diseno"]["folds"] = list(config["diseno"]["folds"])
    c["diseno"]["n_corridas"] = len(semillas) * len(c["diseno"]["folds"])
    c["receta"]["perdida_entrenamiento"] = perdida_de(estrategia)
    return c


def config_g2_base(config_g2, config, semillas) -> dict:
    c = copy.deepcopy(config_g2)
    c["diseno"]["arquitecturas"], c["diseno"]["semillas"] = [config["base"]["arquitectura"]], list(semillas)
    c["diseno"]["folds"] = list(config["diseno"]["folds"])
    c["diseno"]["n_corridas"] = len(semillas) * len(c["diseno"]["folds"])
    return c


def plan_de_corridas(config_g2d, config_id: str) -> list:
    d = config_g2d["diseno"]
    return [{"config_id": config_id, "arq": a, "semilla": s, "fold": k, "id": id_corrida(config_g2d, a, s, k)}
            for a in d["arquitecturas"] for s in d["semillas"] for k in d["folds"]]


# ---------------------------------------------------------------- rutas (todas bajo results/r7_v2/; K4)
def raiz_g4(root, config) -> Path:
    base = config["diseno"]["raiz_resultados"].strip("/")
    return exigir_salida_segura(root, Path(root) / base, base)


def raiz_corridas(root, config, config_id: str, modo: str) -> Path:
    if not ID_CONFIG.fullmatch(config_id) or modo not in ("humo", "completa"):
        raise GuardError(f"config_id o modo inválidos: {config_id!r}/{modo!r}")
    base = config["diseno"]["raiz_resultados"].strip("/")
    return exigir_salida_segura(root, Path(root) / base / config_id / modo, base)


def ruta_ledger(root, config) -> Path:
    return raiz_g4(root, config) / config["presupuesto"]["ledger"]


def siguiente_archivo_nuevo(carpeta: Path, prefijo: str, sufijo: str = ".json") -> Path:
    """`promocion.json`, luego `promocion_02.json`, ...: nunca sobrescribe."""
    p = carpeta / f"{prefijo}{sufijo}"
    n = 2
    while p.exists():
        p = carpeta / f"{prefijo}_{n:02d}{sufijo}"
        n += 1
    return p


def ultimo_archivo(carpeta: Path, prefijo: str, sufijo: str = ".json"):
    cand = [p for p in Path(carpeta).glob(f"{prefijo}*{sufijo}") if re.fullmatch(rf"{prefijo}(_[0-9]{{2}})?{re.escape(sufijo)}", p.name)]
    return sorted(cand, key=lambda p: (len(p.name), p.name))[-1] if cand else None


# ---------------------------------------------------------------- integridad propia (estilo G3a/G3b; no es K6)
def verificar_integridad_g4(root, config) -> dict:
    """HEAD, árbol limpio en `integridad.rutas_sin_cambios`, SHA-256 de `archivos_con_huella` y pin del blob del YAML de G2.

    Compatible con `r4_v2.entrenamiento.ejecutar_corrida` (campos `bloqueos`, `archivos[*]` con los tres hashes, `protocolo_versionado`, `entrada`,
    `rama`). NO es K6. Un error de Git lanza GuardError.
    """
    ig = config["integridad"]
    head = git_texto(root, "rev-parse", "HEAD", obligatorio=True)
    rama = git_texto(root, "rev-parse", "--abbrev-ref", "HEAD")
    estado = git_ok(root, "status", "--porcelain", "--untracked-files=all", "--", *ig["rutas_sin_cambios"]).decode("utf-8", "replace")
    status = [l for l in estado.splitlines() if l.strip()]
    bloqueos = [{"codigo": "arbol_sucio", "ruta": l[3:], "detalle": l} for l in status]
    archivos = {}
    for rel in ig["archivos_con_huella"]:
        reg = huella_archivo(root, rel)
        reg["git_blob_id"] = git_texto(root, "rev-parse", f"HEAD:{rel}")
        archivos[rel] = reg
        if reg["sha256_bytes_archivo"] is None:
            bloqueos.append({"codigo": "archivo_ausente", "ruta": rel, "detalle": rel})
        elif reg["sha256_contenido_blob"] is None:
            bloqueos.append({"codigo": "archivo_no_versionado", "ruta": rel, "detalle": rel})
    ref = config["referencia_g2"]
    if archivos[ref["config"]]["sha256_contenido_blob"] != ref["sha256_contenido_blob"]:
        bloqueos.append({"codigo": "yaml_referencia_g2_distinto_del_fijado", "ruta": ref["config"],
                         "detalle": f"{ref['config']}: {archivos[ref['config']]['sha256_contenido_blob']}"})
    principal = sys.modules.get("__main__")
    entrada = None if principal is None or not getattr(principal, "__file__", None) else ("modulo" if getattr(principal, "__spec__", None) else "script")
    return {"git_commit_head": head, "rama": rama, "rutas_sin_cambios": list(ig["rutas_sin_cambios"]), "status_porcelain": status, "status_aux": status,
            "archivos": archivos, "bloqueos": bloqueos, "protocolo_versionado": archivos[PROTOCOLO_REL]["sha256_contenido_blob"] is not None,
            "entrada": entrada, "modulos_en_repo": [], "modulos_virtuales_tolerados": [],
            "python": {"version": sys.version.split()[0], "dont_write_bytecode": bool(sys.flags.dont_write_bytecode),
                       "pycache_prefix": getattr(sys, "pycache_prefix", None)}}


def exigir_integridad(res: dict) -> dict:
    if res["bloqueos"]:
        raise GuardError("integridad: " + "; ".join(f"{b['codigo']}: {b.get('detalle')}" for b in res["bloqueos"][:6])
                         + (" ..." if len(res["bloqueos"]) > 6 else "") + ". No se entrena")
    return res


# ---------------------------------------------------------------- base (R6) y comparabilidad con la base de G2
def verificar_base_r6(root, config) -> dict:
    """GUARDA de R6 (toda ejecución real de G4 y el reporte): `seleccion_r6.json` DEBE existir y su `elegida` ser A|densenet121; la ausencia BLOQUEA.
    No hay bandera ni parámetro de CLI que la debilite (las pruebas usan un archivo sintético explícito en un directorio temporal)."""
    rel = config["referencia_r6"]["salida"]
    exigir_ruta_no_prohibida(rel)
    ruta = Path(root) / rel
    if not ruta.is_file():
        raise GuardError(f"{rel} no existe: G4 exige la selección de R6 con elegida A|densenet121 (la ausencia bloquea)")
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        elegida = datos["elegida"]
    except (OSError, ValueError, KeyError):
        raise GuardError(f"{rel}: ilegible o sin 'elegida'; no se verifica la base de R6") from None
    esperada = config["referencia_r6"]["elegida"]
    if elegida != esperada:
        raise GuardError(f"R6 eligió {elegida['condicion']}|{elegida['arquitectura']} y la base de G4 es "
                         f"{esperada['condicion']}|{esperada['arquitectura']}: se bloquea (decisión de la autora)")
    return {"estado": "verificada", "ruta": rel, "elegida": elegida, "sha256_bytes_archivo": sha256_archivo(ruta)}


def raiz_base(root, config_g2) -> Path:
    return Path(root) / config_g2["diseno"]["raiz_resultados"].strip("/") / "completa"


def comparar_con_base(root, config, config_g2, particion, referencia, semillas, *, exigir: bool = True) -> dict:
    """Las corridas A|densenet121 de G2 de esas semillas deben existir, ser `completa` válidas y comparables (partición, receta numérica, pesos)."""
    cb = config_g2_base(config_g2, config, semillas)
    plan = [{"arq": p["arq"], "semilla": p["semilla"], "fold": p["fold"], "id": p["id"]} for p in plan_de_corridas(cb, "A")]
    return comparar_con_A(root, config_g2, particion, referencia, plan, raiz_base(root, config_g2), exigir=exigir)


def exigir_corrida_comparable(rc: dict, config_g2, estrategia: dict, particion) -> None:
    """Una corrida de G4 solo se compara con la base si difiere ÚNICAMENTE en la pérdida de entrenamiento."""
    def sin_perdida(r):
        return _canon({k: v for k, v in r.items() if k != "perdida_entrenamiento"})
    if (rc.get("modo") != "completa" or rc.get("estado") != "completada" or rc.get("invalida_integridad")
            or rc.get("sha256_bytes_archivo_particion") != particion.sha256_bytes_archivo
            or sin_perdida(rc["receta"]) != sin_perdida(config_g2["receta"]) or rc.get("epocas_max") != config_g2["receta"]["epochs_max"]):
        raise GuardError(f"{rc.get('id_corrida')}: no es una corrida `completa` válida y comparable con la base (partición/receta distintas de G2)")
    if rc["receta"]["perdida_entrenamiento"] != perdida_de(estrategia) or rc.get("condicion") != f"A+{estrategia['id']}":
        raise GuardError(f"{rc.get('id_corrida')}: la pérdida registrada no corresponde a la estrategia {estrategia['id']}")


# ---------------------------------------------------------------- lectura de AP OOF (solo predicciones guardadas)
def _ap(oof, arq, semilla) -> float:
    return float(M.metricas(oof.y, oof.prob[(arq, semilla)])["ap"])


def ap_base(root, config, config_g2, particion, semilla) -> float:
    cb = config_g2_base(config_g2, config, [semilla])
    corr = AG.cargar_corridas(raiz_base(root, config_g2), cb, particion)
    return _ap(AG.ensamblar_oof(corr, particion, cb), config["base"]["arquitectura"], semilla)


def ap_estrategia(root, config, config_g2, particion, estrategia, semilla):
    """AP OOF sin redondear de (estrategia, semilla) o None si sus 5 folds no están completos."""
    g = config_g2_g4(config_g2, config, estrategia, [semilla])
    raiz = raiz_corridas(root, config, estrategia["id"], "completa")
    if any(corrida_completada(raiz / p["id"]) is None for p in plan_de_corridas(g, estrategia["id"])):
        return None
    corr = AG.cargar_corridas(raiz, g, particion)
    for (_, _, _), (_, rc) in corr.items():
        exigir_corrida_comparable(rc, config_g2, estrategia, particion)
    return _ap(AG.ensamblar_oof(corr, particion, g), config["base"]["arquitectura"], semilla)


# ---------------------------------------------------------------- promoción A -> B (pura; AP sin redondear)
def decidir_promocion(delta_por_id: dict, orden: list, max_promovidas: int = 2, n_total: int = 6) -> dict:
    """`delta_por_id`: {config_id: ΔAP o None (incompleta)}. Promueve hasta 2 con ΔAP > 0 por mayor ΔAP; empate exacto por el orden predefinido."""
    evaluadas = [i for i in orden if delta_por_id.get(i) is not None]
    candidatas = [i for i in evaluadas if delta_por_id[i] > 0]
    promovidas = sorted(candidatas, key=lambda i: (-delta_por_id[i], orden.index(i)))[:max_promovidas]
    completa = len(evaluadas) == n_total
    return {"evaluadas": evaluadas, "incompletas": [i for i in orden if i not in evaluadas], "delta_ap": {i: delta_por_id[i] for i in evaluadas},
            "promovidas": promovidas, "fase_a_completa": completa, "promocion_provisional": not completa,
            "no_se_declara_mejor_de_las_seis": not completa}


def calcular_promocion(root, config, config_g2, particion) -> dict:
    semilla = config["diseno"]["fase_a_semilla"]
    base = ap_base(root, config, config_g2, particion, semilla)
    estr = estrategias_fase_a(config)
    aps = {e["id"]: ap_estrategia(root, config, config_g2, particion, e, semilla) for e in estr}
    deltas = {i: (None if ap is None else ap - base) for i, ap in aps.items()}
    res = decidir_promocion(deltas, [e["id"] for e in estr], config["promocion"]["max_promovidas"], len(estr))
    res["ap_base_semilla_42"], res["ap_por_config"] = base, aps                      # sin redondear
    return res
