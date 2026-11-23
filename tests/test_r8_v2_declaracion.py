"""Pruebas sintéticas (CPU, sin torch, sin Git real) de `r8_v2.declaracion`: la declaración formal de congelamiento como evidencia versionada y verificable por máquina.

Cubren: construcción y verificación, SHA-256 externo de la declaración, fin de línea LF normalizado, alteración de cada artefacto vinculado (YAML, protocolo, manifiesto, código de inferencia,
checkpoints, reporte real), alteración del propio contenido declarado (sistema, umbral, miembros), `congelado: false` histórico intacto, ausencia de hash propio, CLI y la declaración real
versionada. No abren test ni datos reales. Comando: py -3.13 -B -m pytest tests/test_r8_v2_declaracion.py -q -p no:cacheprovider
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from src.r4_v2.common import GuardError  # noqa: E402
from src.r7_v2 import umbral_sistema as U  # noqa: E402
from src.r8_v2 import congelamiento as C  # noqa: E402
from src.r8_v2 import declaracion as D  # noqa: E402
from tests.test_r8_v2_congelamiento import AUT, ENTORNO, REPO, armar, correr, sha  # noqa: E402

DECISION_DECL = "D-060"
HEAD = "a" * 40


def verificar_con_yaml(root):
    def f():
        h = {"sha256_bytes_archivo": sha((Path(root) / C.CONFIG_REL).read_bytes()), "sha256_contenido_blob": "c" * 64}
        return {"git_commit_head": HEAD, "rama": "sintetica", "status_porcelain": [], "bloqueos": [], "archivos": {C.CONFIG_REL: h}}
    return f


def armar_decl(tmp_path):
    """Raíz sintética con YAML escrito (protocolo ya registrado), código de inferencia, 15 checkpoints, reporte real de verificación y declaración construida."""
    root, config = armar(tmp_path)
    config["protocolo"]["git_commit"] = "b" * 40
    config["protocolo"]["sha256_contenido_blob"] = D.sha256_lf(root / C.PROTOCOLO_REL)
    (root / C.CONFIG_REL).parent.mkdir(parents=True, exist_ok=True)
    (root / C.CONFIG_REL).write_bytes(yaml.safe_dump(config, allow_unicode=True, sort_keys=False).encode("utf-8"))
    for rel in D.CODIGO_INFERENCIA:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(f"# codigo sintetico {rel}\n", encoding="utf-8", newline="\n")
    r = correr(root, config, verificar=verificar_con_yaml(root))
    assert r["reporte_aceptado"]
    reporte = root / D.ruta_reporte("D-058")
    d = D.construir_declaracion(root, decision=DECISION_DECL, fecha="2026-11-23",
                                verificacion={"decision": "D-058", "sha256_bytes_archivo": sha(reporte.read_bytes()), "git_commit_head": HEAD})
    ruta = root / D.DECLARACION_REL
    escribir(ruta, d)
    return root, config, ruta, d


def escribir(ruta: Path, d: dict):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_bytes((json.dumps(d, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def reescribir_reporte(root, d, ruta_decl, mutar):
    """Altera el contenido del reporte real y re-firma su SHA-256 en la declaración (para probar la verificación de CONTENIDO, no solo el hash)."""
    rp = root / d["verificacion"]["ruta"]
    r = json.loads(rp.read_text(encoding="utf-8"))
    mutar(r)
    rp.write_bytes(json.dumps(r, sort_keys=True).encode("utf-8"))
    d = copy.deepcopy(d)
    d["verificacion"]["sha256_bytes_archivo"] = sha(rp.read_bytes())
    escribir(ruta_decl, d)


# ============================================================================ caso feliz y vía externa del hash
def test_declaracion_construida_se_verifica_completa(tmp_path):
    root, config, ruta, d = armar_decl(tmp_path)
    r = D.verificar_declaracion(root)
    assert r["n_miembros"] == 15 and r["t_hat"] == C.T_HAT == "0.08456226376195748" and r["checkpoints_comprobados"] and r["reporte_comprobado"]["sha256_bytes_archivo"]
    assert d["declaracion"]["decision"] == DECISION_DECL and d["declaracion"]["yaml_conserva_congelado_false_historico"] is True
    assert yaml.safe_load((root / C.CONFIG_REL).read_text(encoding="utf-8"))["sistema"]["congelado"] is False
    assert not re.search(r"(?<![0-9a-f])" + re.escape(r["sha256_declaracion"]) + r"(?![0-9a-f])", ruta.read_text(encoding="utf-8"))          # sin hash propio


def test_exigir_declaracion_requiere_el_sha256_externo(tmp_path):
    root, _, ruta, _ = armar_decl(tmp_path)
    esperado = D.sha256_lf(ruta)
    assert D.exigir_declaracion(root, esperado)["sha256_declaracion"] == esperado
    for malo in (None, "", "abc", "g" * 64, "0" * 64, esperado.upper()):
        with pytest.raises(GuardError):
            D.exigir_declaracion(root, malo)


def test_declaracion_alterada_no_coincide_con_su_sha256_externo(tmp_path):
    root, _, ruta, d = armar_decl(tmp_path)
    esperado = D.sha256_lf(ruta)
    d["declaracion"]["fecha"] = "2026-12-01"
    escribir(ruta, d)
    with pytest.raises(GuardError, match="no coincide"):
        D.exigir_declaracion(root, esperado)


def test_falta_la_declaracion_o_tiene_hash_propio(tmp_path):
    root, _, ruta, d = armar_decl(tmp_path)
    d["sha256_declaracion"] = "1" * 64
    escribir(ruta, d)
    with pytest.raises(GuardError, match="hash"):
        D.verificar_declaracion(root)
    ruta.unlink()
    with pytest.raises(GuardError, match="no existe"):
        D.verificar_declaracion(root)


def test_fin_de_linea_crlf_en_textos_versionados_es_equivalente(tmp_path):
    root, _, _, _ = armar_decl(tmp_path)
    for rel in (C.CONFIG_REL, C.PROTOCOLO_REL, C.MIEMBROS_REL, D.CODIGO_INFERENCIA[0]):
        p = root / rel
        p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))                  # copia de trabajo con autocrlf
    assert D.verificar_declaracion(root, comprobar_checkpoints=False, comprobar_reporte=False)["n_miembros"] == 15


# ============================================================================ alteración de artefactos vinculados
@pytest.mark.parametrize("rel,patron", [(C.CONFIG_REL, "cambió"), (C.PROTOCOLO_REL, "cambió"), (C.MIEMBROS_REL, "cambió"), (D.CODIGO_INFERENCIA[0], "código de inferencia"),
                                        (D.CODIGO_INFERENCIA[-1], "código de inferencia")])
def test_artefacto_vinculado_alterado_bloquea(tmp_path, rel, patron):
    root, _, _, _ = armar_decl(tmp_path)
    p = root / rel
    p.write_bytes(p.read_bytes() + b"\n# alterado\n")
    with pytest.raises(GuardError, match=patron):
        D.verificar_declaracion(root)


def test_yaml_con_congelado_true_bloquea(tmp_path):
    root, config, _, _ = armar_decl(tmp_path)
    c = copy.deepcopy(config)
    c["sistema"]["congelado"] = True
    (root / C.CONFIG_REL).write_bytes(yaml.safe_dump(c, allow_unicode=True, sort_keys=False).encode("utf-8"))
    with pytest.raises(GuardError):
        D.verificar_declaracion(root)


def test_checkpoint_alterado_o_ausente_bloquea_solo_si_se_comprueban(tmp_path):
    root, _, _, d = armar_decl(tmp_path)
    ck = root / d["miembros"][3]["ruta_checkpoint"]
    ck.write_bytes(ck.read_bytes() + b"X")
    with pytest.raises(GuardError, match="SHA-256"):
        D.verificar_declaracion(root)
    assert D.verificar_declaracion(root, comprobar_checkpoints=False)["checkpoints_comprobados"] is False
    ck.unlink()
    with pytest.raises(GuardError, match="ausente"):
        D.verificar_declaracion(root)


# ============================================================================ alteración del contenido declarado
@pytest.mark.parametrize("mut", [
    lambda d: d["sistema"]["umbral"].update(t_hat="0.0846"), lambda d: d["sistema"].update(arquitectura="convnext_tiny"), lambda d: d["sistema"].update(representacion="B"),
    lambda d: d["sistema"].update(agregacion="mediana"), lambda d: d["sistema"].update(regla="p > t_hat"), lambda d: d["sistema"].update(preprocesamiento="otro"),
    lambda d: d["sistema"].update(n_miembros=14), lambda d: d["miembros"].pop(), lambda d: d["miembros"].reverse(),
    lambda d: d["miembros"][0].update(sha256_bytes_archivo_checkpoint="e" * 64), lambda d: d["artefactos"]["protocolo"].update(git_commit="c" * 40),
    lambda d: d["artefactos"]["config"].update(sha256_contenido="0" * 64), lambda d: d["codigo_inferencia"].pop(D.CODIGO_INFERENCIA[0]),
    lambda d: d["codigo_inferencia"].update({D.CODIGO_INFERENCIA[1]: "0" * 64}), lambda d: d["declaracion"].update(yaml_conserva_congelado_false_historico=False),
    lambda d: d.update(esquema="otro/1")])
def test_contenido_declarado_distinto_de_lo_vinculado_bloquea(tmp_path, mut):
    root, _, ruta, d = armar_decl(tmp_path)
    mut(d)
    escribir(ruta, d)
    with pytest.raises(GuardError):
        D.verificar_declaracion(root)


# ============================================================================ reporte real de verificación
def test_reporte_ausente_o_alterado_bloquea(tmp_path):
    root, _, _, d = armar_decl(tmp_path)
    rp = root / d["verificacion"]["ruta"]
    rp.write_bytes(rp.read_bytes() + b" ")
    with pytest.raises(GuardError, match="SHA-256"):
        D.verificar_declaracion(root)
    assert D.verificar_declaracion(root, comprobar_reporte=False)["reporte_comprobado"] is None
    rp.unlink()
    with pytest.raises(GuardError, match="no existe"):
        D.verificar_declaracion(root)


@pytest.mark.parametrize("mut", [
    lambda r: r.update(reporte_aceptado=False), lambda r: r.update(invalida_integridad=True), lambda r: r.update(congelado=True),
    lambda r: r.update(advertencias_entorno=["torch distinto"]), lambda r: r.update(autorizacion="CONFIRMADO:D-099"), lambda r: r.update(git_commit_head="d" * 40),
    lambda r: r.update(sha256_bytes_archivo_config="0" * 64), lambda r: r.update(protocolo_commit="d" * 40), lambda r: r.update(sha256_bytes_archivo_miembros_versionados="0" * 64),
    lambda r: r["miembros"][0].update(sha256_recalculado="0" * 64), lambda r: r["miembros"].pop()])
def test_reporte_con_contenido_inaceptable_o_de_otros_artefactos_bloquea(tmp_path, mut):
    root, _, ruta, d = armar_decl(tmp_path)
    reescribir_reporte(root, d, ruta, mut)
    with pytest.raises(GuardError):
        D.verificar_declaracion(root)


def test_ruta_de_reporte_fuera_de_lo_esperado_bloquea(tmp_path):
    root, _, ruta, d = armar_decl(tmp_path)
    d["verificacion"]["ruta"] = "results/r8_v2/test/resumen_verificacion.json"
    escribir(ruta, d)
    with pytest.raises(GuardError, match="ruta del reporte"):
        D.verificar_declaracion(root)


@pytest.mark.parametrize("kw", [{"decision": "D-60"}, {"fecha": "x", "verificacion": {"decision": "D-058", "sha256_bytes_archivo": "xyz", "git_commit_head": HEAD}},
                                {"verificacion": {"decision": "D-058", "sha256_bytes_archivo": "1" * 64, "git_commit_head": "abc"}}])
def test_construccion_rechaza_datos_mal_formados(tmp_path, kw):
    root, _, _, _ = armar_decl(tmp_path)
    args = {"decision": DECISION_DECL, "fecha": "2026-11-23", "verificacion": {"decision": "D-058", "sha256_bytes_archivo": "1" * 64, "git_commit_head": HEAD}, **kw}
    with pytest.raises(GuardError):
        D.construir_declaracion(root, **args)


# ============================================================================ CLI y alcance
def test_cli_codigos_de_salida(tmp_path):
    root, _, ruta, _ = armar_decl(tmp_path)
    assert D.main(["--modo", "verificar"], root=root) == U.CODIGO_SALIDA_GUARDA
    assert D.main(["--modo", "verificar", "--autorizacion", "CONFIRMADO:D-060"], root=root) == 0
    assert D.main(["--modo", "verificar", "--autorizacion", "CONFIRMADO:D-060", "--sha256-declaracion", D.sha256_lf(ruta)], root=root) == 0
    assert D.main(["--modo", "verificar", "--autorizacion", "CONFIRMADO:D-060", "--sha256-declaracion", "0" * 64], root=root) == U.CODIGO_SALIDA_GUARDA


def test_modulo_solo_lectura_sin_inferencia_ni_torch():
    fuente = Path(D.__file__).read_text(encoding="utf-8")
    assert not any(hasattr(D, n) for n in ("inferir", "predecir", "evaluar", "calcular_metricas"))
    assert not re.search(r"^\s*(import|from)\s+(torch|torchvision)\b", fuente, re.M)
    assert not re.search(r"write_text|write_bytes|open\(.*[\"']w", fuente)                                 # solo lee; la autora versiona la declaración


# ============================================================================ declaración real versionada
def test_declaracion_real_coincide_con_los_artefactos_versionados():
    """Contra los archivos reales del repositorio (sin checkpoints ni reporte: residen en Falcon; allí se comprueban con el CLI)."""
    ruta = REPO / D.DECLARACION_REL
    if not ruta.is_file():
        pytest.skip("la declaración real aún no está en el árbol de trabajo")
    r = D.verificar_declaracion(REPO, comprobar_checkpoints=False, comprobar_reporte=False)
    d = json.loads(ruta.read_text(encoding="utf-8"))
    assert r["n_miembros"] == 15 and r["t_hat"] == "0.08456226376195748"
    assert d["declaracion"]["decision"] == "D-060" and d["verificacion"]["decision"] == "D-058"
    assert d["verificacion"]["sha256_bytes_archivo"] == "cd82122e6e661e0948526dda3fd9fe9e98cb4f5693afb00780f7b99bfe18683e"
    assert any("TRES modelos" in x and "NO está validado para el ensamble de quince" in x for x in d["limitaciones"])
    assert len(d["miembros"]) == 15 and ENTORNO["python"] == d["entorno_verificacion"]["python"] and AUT == "CONFIRMADO:D-058"
