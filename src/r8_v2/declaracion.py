"""R8 `declaracion_congelamiento_v1`: evidencia VERSIONADA y VERIFICABLE POR MÁQUINA de la declaración formal de congelamiento del sistema final.

`configs/r8_v2/declaracion_congelamiento_v1.json` identifica el sistema declarado y lo vincula, por SHA-256, a: la configuración `sistema_congelado_v1.yaml` (que se conserva intacta, con su
`congelado: false` histórico), el protocolo (commit y hash), el manifiesto versionado de los 15 miembros, el código de inferencia (preprocesamiento y agregación) y el reporte real de la
verificación de D-058. NO contiene su propio hash (sin circularidad): el futuro ejecutor de R8 debe recibir el SHA-256 de este archivo por una vía externa a él (protocolo R8 v2 o argumento) y
llamar a `exigir_declaracion`, que recalcula y compara todo y se detiene ante cualquier discrepancia (el registro local de decisiones no es evidencia suficiente).

`t_hat` es una regla experimental previa a R8: calibrada con el OOF de TRES modelos por imagen, NO validada para el ensamble de quince y sin garantía de sensibilidad del 90 %.
Solo lectura de artefactos de desarrollo y checkpoints (SHA-256). No abre el test, no infiere, no entrena. Los textos versionados se comparan con fin de línea normalizado a LF (el blob de
Git es LF; `core.autocrlf` puede alterar la copia de trabajo); checkpoints y reporte real, byte a byte.

CLI (solo la autora; D-NNN propia): python -B -m src.r8_v2.declaracion --modo verificar --autorizacion CONFIRMADO:D-NNN [--sha256-declaracion <64 hex>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from src.r4_v2.common import RAIZ_REPO, GuardError, exigir_ruta_no_prohibida, sha256_archivo
from src.r7_v2 import umbral_sistema as U
from src.r8_v2 import congelamiento as C

ESQUEMA = "declaracion_congelamiento_v1/1"
DECLARACION_REL = "configs/r8_v2/declaracion_congelamiento_v1.json"
CODIGO_INFERENCIA = ("src/classification/transforms.py", "src/classification/dataloader.py", "src/classification/models.py", "src/classification/evaluate.py",
                     "src/r4_v2/dataset_lesiones.py", "src/r4_v2/entrenamiento.py", "src/r7_v2/sistema_final.py")
CLAVES_SISTEMA_EXCLUIDAS = ("estado", "congelado")                                           # el YAML conserva su estado histórico; la declaración es otra evidencia
CLAVES_MIEMBRO = ("id_corrida", "semilla", "fold", "ruta_checkpoint", "sha256_bytes_archivo_checkpoint")
DECISION = re.compile(r"D-[0-9]{3}")
LIMITACIONES = (
    "t_hat = 0.08456226376195748 se calibró en desarrollo con el OOF de TRES modelos por imagen (S* = 0.90, objetivo exploratorio) y NO está validado para el ensamble de quince: es una regla "
    "experimental previa a R8; no garantiza una sensibilidad del 90 %",
    "el sistema de quince no tiene OOF de desarrollo (12 de los 15 modelos vieron cada imagen); la fidelidad por miembro (diferencia máxima 0.0) verifica integridad y equivalencia de "
    "preprocesamiento, no desempeño",
    "partición provisional phash_le8_v1 (exploratoria, post hoc; sin independencia por paciente demostrada); conformidad académica del asesor: PENDIENTE",
    "clasificación binaria de lesión a nivel de imagen 2D; no detecta ovarios normales; un único subtipo maligno en el dataset; sin uso clínico validado",
    "acceso histórico previo al test documentado; no se afirma independencia absoluta del test histórico; el congelamiento no abre el test")
EXIGENCIAS_EJECUTOR = (
    "recibir el SHA-256 de esta declaración por una vía externa a ella y compararlo con el del archivo (LF normalizado)",
    "llamar a `exigir_declaracion` con comprobación de checkpoints y del reporte real; detenerse ante cualquier discrepancia",
    "usar exactamente los 15 miembros, el preprocesamiento, el promedio simple y la regla `p >= t_hat` declarados; no seleccionar, sustituir ni reordenar miembros",
    "no recalibrar ni ajustar nada con el test; una sola ejecución; autorización específica de la autora como ejecutora oficial")


def sha256_lf(path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _exigir(cond: bool, mensaje: str) -> None:
    if not cond:
        raise GuardError(mensaje)


def ruta_reporte(decision: str) -> str:
    return f"{C.SALIDA_REL}/{decision}/resumen_verificacion.json"


# ============================================================================ construcción (genera el contenido; la autora lo versiona)
def construir_declaracion(root, *, decision: str, fecha: str, verificacion: dict) -> dict:
    """Contenido de la declaración a partir de los artefactos versionados. `verificacion` aporta el reporte REAL de Falcon: decision, sha256_bytes_archivo, git_commit_head
    (este generador NO lee ese reporte; lo comprueba `verificar_declaracion` donde reside)."""
    root = Path(root)
    _exigir(bool(DECISION.fullmatch(decision)) and bool(DECISION.fullmatch(verificacion.get("decision", ""))), "decisiones con formato D-NNN")
    _exigir(bool(C.HEX64.fullmatch(verificacion.get("sha256_bytes_archivo", ""))) and bool(re.fullmatch(r"[0-9a-f]{40}", verificacion.get("git_commit_head", ""))),
            "el reporte de verificación exige SHA-256 (64 hex) y commit (40 hex)")
    config = C.cargar_config(root / C.CONFIG_REL)
    miembros = C.validar_miembros(C.leer_tabla(root / C.MIEMBROS_REL), config)
    p = config["protocolo"]
    return {
        "esquema": ESQUEMA,
        "declaracion": {"decision": decision, "fecha": fecha, "estado": "sistema declarado congelado para R8 (decision de la autora)", "autoriza": "la autora",
                        "conformidad_academica_asesor": "PENDIENTE", "yaml_conserva_congelado_false_historico": True},
        "sistema": {k: v for k, v in config["sistema"].items() if k not in CLAVES_SISTEMA_EXCLUIDAS},
        "artefactos": {
            "config": {"ruta": C.CONFIG_REL, "sha256_contenido": sha256_lf(root / C.CONFIG_REL)},
            "protocolo": {"ruta": C.PROTOCOLO_REL, "git_commit": p["git_commit"], "sha256_contenido": p["sha256_contenido_blob"]},
            "miembros": {"ruta": C.MIEMBROS_REL, "sha256_contenido": sha256_lf(root / C.MIEMBROS_REL)}},
        "miembros": [{k: m[k] for k in CLAVES_MIEMBRO} for m in miembros],
        "codigo_inferencia": {rel: sha256_lf(root / rel) for rel in CODIGO_INFERENCIA},
        "verificacion": {"decision": verificacion["decision"], "ruta": ruta_reporte(verificacion["decision"]), "sha256_bytes_archivo": verificacion["sha256_bytes_archivo"],
                         "git_commit_head": verificacion["git_commit_head"], "exit": 0},
        "entorno_verificacion": config["entorno_registrado"],
        "limitaciones": list(LIMITACIONES),
        "exigencias_para_el_ejecutor_r8": list(EXIGENCIAS_EJECUTOR),
        "trazabilidad": list(config["trazabilidad"]) + [verificacion["decision"], decision]}


# ============================================================================ verificación
def verificar_declaracion(root, ruta=None, *, comprobar_checkpoints: bool = True, comprobar_reporte: bool = True) -> dict:
    """Recalcula y compara todo lo que la declaración fija. Lanza GuardError ante cualquier discrepancia."""
    root = Path(root)
    ruta = Path(ruta) if ruta else root / DECLARACION_REL
    _exigir(ruta.is_file(), f"{ruta.name} no existe: sin declaración versionada no hay congelamiento formal verificable")
    d = json.loads(ruta.read_text(encoding="utf-8"))
    _exigir(d.get("esquema") == ESQUEMA, f"esquema distinto de {ESQUEMA!r}")
    _exigir(not any(k.startswith("sha256_declaracion") or k == "hash_propio" for k in d), "la declaración no debe contener su propio hash (sin circularidad)")
    dec = d["declaracion"]
    _exigir(bool(DECISION.fullmatch(str(dec.get("decision")))) and dec.get("yaml_conserva_congelado_false_historico") is True, "decisión de declaración inválida")
    config = C.cargar_config(root / C.CONFIG_REL)                                          # valida sistema == constantes del código; `congelado` debe seguir en false
    _exigir(config["sistema"]["congelado"] is False, "el YAML debe conservar `congelado: false` (estado histórico intacto)")
    a = d["artefactos"]
    for clave, rel in (("config", C.CONFIG_REL), ("protocolo", C.PROTOCOLO_REL), ("miembros", C.MIEMBROS_REL)):
        _exigir(a[clave]["ruta"] == rel, f"artefactos.{clave}.ruta debe ser {rel}")
        _exigir(sha256_lf(root / rel) == a[clave]["sha256_contenido"], f"{rel}: SHA-256 distinto del declarado (el artefacto cambió después de la declaración)")
    p = config["protocolo"]
    _exigir(a["protocolo"]["git_commit"] == p["git_commit"] and a["protocolo"]["sha256_contenido"] == p["sha256_contenido_blob"], "protocolo: commit/hash distintos de los del YAML")
    _exigir(a["miembros"]["sha256_contenido"] == config["miembros"]["sha256_bytes_archivo"], "miembros: SHA-256 distinto del fijado en el YAML")
    esperado = {k: v for k, v in config["sistema"].items() if k not in CLAVES_SISTEMA_EXCLUIDAS}
    _exigir(d["sistema"] == esperado, "el sistema declarado difiere del YAML (arquitectura, miembros, preprocesamiento, agregación, regla o umbral)")
    miembros = C.validar_miembros(C.leer_tabla(root / C.MIEMBROS_REL), config)
    _exigir(d["miembros"] == [{k: m[k] for k in CLAVES_MIEMBRO} for m in miembros], "los 15 miembros declarados difieren del manifiesto versionado")
    _exigir(set(d["codigo_inferencia"]) == set(CODIGO_INFERENCIA), "codigo_inferencia debe fijar exactamente los archivos de preprocesamiento y agregación")
    for rel, h in d["codigo_inferencia"].items():
        _exigir(sha256_lf(root / rel) == h, f"{rel}: el código de inferencia cambió después de la declaración")
    if comprobar_checkpoints:
        C.verificar_checkpoints(root, miembros)                                            # SHA-256 recalculado en disco (Falcon)
    reporte = None
    if comprobar_reporte:
        v = d["verificacion"]
        _exigir(v["ruta"] == ruta_reporte(v["decision"]), "ruta del reporte distinta de la esperada para su decisión")
        exigir_ruta_no_prohibida(v["ruta"])
        rp = root / v["ruta"]
        _exigir(rp.is_file(), f"{v['ruta']} no existe: se comprueba donde reside el reporte real (Falcon)")
        _exigir(sha256_archivo(rp) == v["sha256_bytes_archivo"], "el reporte real de verificación difiere del SHA-256 declarado")
        r = json.loads(rp.read_text(encoding="utf-8"))
        _exigir(r.get("reporte_aceptado") is True and r.get("invalida_integridad") is False and r.get("congelado") is False and r.get("advertencias_entorno") == [],
                "el reporte de verificación no es aceptable (reporte_aceptado, invalida_integridad, advertencias_entorno)")
        _exigir(r.get("autorizacion") == f"CONFIRMADO:{v['decision']}" and r.get("git_commit_head") == v["git_commit_head"], "el reporte corresponde a otra autorización o a otro commit")
        _exigir(r.get("sha256_bytes_archivo_config") == a["config"]["sha256_contenido"] and r.get("protocolo_commit") == p["git_commit"]
                and r.get("protocolo_sha256_contenido_blob") == p["sha256_contenido_blob"]
                and r.get("sha256_bytes_archivo_miembros_versionados") == a["miembros"]["sha256_contenido"], "el reporte se emitió sobre otros artefactos que los declarados")
        _exigir([(m["id_corrida"], m["sha256_recalculado"]) for m in r.get("miembros", [])] == [(m["id_corrida"], m["sha256_bytes_archivo_checkpoint"]) for m in miembros],
                "los 15 SHA-256 del reporte no coinciden con los miembros declarados")
        reporte = {"ruta": v["ruta"], "sha256_bytes_archivo": v["sha256_bytes_archivo"]}
    return {"declaracion": dec["decision"], "n_miembros": len(miembros), "t_hat": config["sistema"]["umbral"]["t_hat"], "checkpoints_comprobados": comprobar_checkpoints,
            "reporte_comprobado": reporte, "sha256_declaracion": sha256_lf(ruta)}


def exigir_declaracion(root, sha256_esperado: str, ruta=None, **kw) -> dict:
    """Punto de entrada para el ejecutor de R8: el SHA-256 de la declaración llega por una vía externa a ella y debe coincidir; luego se verifica todo."""
    _exigir(bool(C.HEX64.fullmatch(str(sha256_esperado))), "se exige el SHA-256 (64 hex) de la declaración, recibido por una vía externa a ella")
    ruta = Path(ruta) if ruta else Path(root) / DECLARACION_REL
    _exigir(ruta.is_file(), "falta la declaración versionada")
    _exigir(sha256_lf(ruta) == sha256_esperado, "el SHA-256 de la declaración no coincide con el esperado (declaración alterada o distinta)")
    return verificar_declaracion(root, ruta, **kw)


def main(argv=None, *, root=None) -> int:
    ap = argparse.ArgumentParser(description="declaracion_congelamiento_v1 (R8): verificación de la declaración formal; solo lectura; no abre el test")
    ap.add_argument("--modo", required=True, choices=("verificar",))
    ap.add_argument("--autorizacion", default="", help="CONFIRMADO:D-NNN de ejecución; sin ella no se lee nada")
    ap.add_argument("--sha256-declaracion", default=None, help="SHA-256 esperado de la declaración (vía externa); si se da, se exige que coincida")
    args = ap.parse_args(argv)
    root = Path(root or RAIZ_REPO).resolve()
    try:
        U.exigir_autorizacion(args.autorizacion)
        r = exigir_declaracion(root, args.sha256_declaracion) if args.sha256_declaracion else verificar_declaracion(root)
        print(json.dumps(r, indent=2, ensure_ascii=False, sort_keys=True))
        print("DECLARACIÓN VERIFICADA: artefactos, miembros, código de inferencia, checkpoints y reporte coinciden con lo declarado.")
        return 0
    except GuardError as exc:
        print(f"DETENIDO: {exc}", file=sys.stderr)
        return U.CODIGO_SALIDA_GUARDA


if __name__ == "__main__":
    raise SystemExit(main())
