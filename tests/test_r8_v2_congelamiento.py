"""Pruebas sintéticas (CPU, sin torch, sin Git real) de `r8_v2.congelamiento` (preparación del congelamiento; NO declara congelamiento ni abre el test).

Cubren: configuración y protocolo reales, caso feliz, ausencia de autorización, integridad, pines de procedencia, manifiesto de miembros (15 únicos, estructura, hashes), coherencia con
manifiesto JSON / fidelidad / umbral, `t_hat` exacto, checkpoints (ausentes o alterados), salida nueva sin sobrescritura, advertencia de entorno y ausencia de inferencia/métricas.
Comando: py -3.13 -B -m pytest tests/test_r8_v2_congelamiento.py -q -p no:cacheprovider
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
from pathlib import Path

import pytest

pytest.importorskip("yaml")

from src.r4_v2.common import GuardError  # noqa: E402
from src.r7_v2 import umbral_sistema as U  # noqa: E402
from src.r7_v2.sistema_final import T_HAT  # noqa: E402
from src.r8_v2 import congelamiento as C  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
AUT = "CONFIRMADO:D-058"
SEMILLAS, FOLDS = (42, 43, 44), range(5)
G2_BLOB = "a" * 64
ENTORNO = {"python": "3.10.12", "torch": "2.5.1+cu121", "torchvision": "0.20.1+cu121", "numpy": "2.2.6"}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def verificar_falso(head="a" * 40, bloqueos=None):
    def f():
        h = {"sha256_bytes_archivo": "1" * 64, "sha256_contenido_blob": "c" * 64}
        return {"git_commit_head": head, "rama": "sintetica", "status_porcelain": [], "bloqueos": list(bloqueos or []), "archivos": {C.CONFIG_REL: h}}
    return f


def escribir_tabla(path: Path, filas: list[dict], comentario="# sintetico"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(comentario + "\n")
        w = csv.DictWriter(fh, fieldnames=list(filas[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(filas)


def armar(root: Path, *, mutar_filas=None, mutar_manifiesto=None, mutar_fidelidad=None, mutar_umbral=None, sin_ckpt=None, alterar_ckpt=None):
    """Raíz sintética con 15 checkpoints, manifiesto/fidelidad/umbral sintéticos y una copia de la configuración real con los pines recalculados."""
    root = Path(root)
    config = C.cargar_config(REPO / C.CONFIG_REL)
    config = copy.deepcopy(config)
    (root / C.PROTOCOLO_REL).parent.mkdir(parents=True, exist_ok=True)
    (root / C.PROTOCOLO_REL).write_bytes((REPO / C.PROTOCOLO_REL).read_bytes())
    r6 = root / config["referencia_r6"]["salida"]
    r6.parent.mkdir(parents=True, exist_ok=True)
    r6.write_text(json.dumps({"elegida": {"condicion": "A", "arquitectura": "densenet121"}}), encoding="utf-8")
    part = root / C.REL_PARTICION
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_bytes(b"image_id,image_name,fold,group_id,label_v2\nDEV_0,DEV_0.png,0,g1,0\nDEV_1,DEV_1.png,1,g2,1\n")
    sha_part = sha(part.read_bytes())
    filas = []
    for sem in SEMILLAS:
        for k in FOLDS:
            idc = f"A|densenet121|s{sem}|f{k}"
            rel = f"results/r4_v2/phash_le8_v1/g2_baseline_v1/completa/{idc.replace('|', '_')}/intento_1/checkpoint_mejor.pth"
            contenido = f"ckpt s{sem} f{k}".encode()
            ruta = root / rel
            ruta.parent.mkdir(parents=True, exist_ok=True)
            if (sem, k) != sin_ckpt:
                ruta.write_bytes(contenido + (b"X" if (sem, k) == alterar_ckpt else b""))
            h = sha(contenido)
            filas.append({"arquitectura": "densenet121", "semilla": sem, "fold": k, "id_corrida": idc, "intento": "intento_1", "ruta_checkpoint": rel,
                          "sha256_bytes_archivo_checkpoint": h, "sha256_checkpoint_registrado_run_config": h, "coincide": "True", "estado": "completada",
                          "invalida_integridad": "False", "sha256_bytes_archivo_particion": sha_part, "sha256_contenido_blob_config_g2": G2_BLOB})
    if mutar_filas:
        filas = mutar_filas(filas)
    base = "results/r7_v2/phash_le8_v1/sistema_final_v1"
    man_csv = root / C.REL_MANIFIESTO_CSV
    escribir_tabla(man_csv, filas, "# manifiesto sintetico")
    sha_man = sha(man_csv.read_bytes())
    versionado = root / C.MIEMBROS_REL
    versionado.parent.mkdir(parents=True, exist_ok=True)
    versionado.write_bytes(man_csv.read_bytes())
    mj = {"n_miembros": 15, "advertencias": [], "reporte_aceptado": True, "invalida_integridad": False, "sha256_bytes_archivo_manifiesto_csv": sha_man,
          "miembros": [{"id_corrida": f["id_corrida"], "ruta_checkpoint": f["ruta_checkpoint"], "sha256_bytes_archivo_checkpoint": f["sha256_bytes_archivo_checkpoint"]} for f in filas]}
    if mutar_manifiesto:
        mutar_manifiesto(mj)
    for rel in (C.REL_FIDELIDAD_JSON, C.REL_UMBRAL, C.PINES_REQUERIDOS[5], C.PINES_REQUERIDOS[6]):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / C.REL_MANIFIESTO_JSON).write_text(json.dumps(mj), encoding="utf-8")
    fj = {"n_miembros": 15, "alerta_global": False, "reporte_aceptado": True, "invalida_integridad": False, "t_hat": T_HAT, "sha256_bytes_archivo_manifiesto_csv": sha_man,
          "metricas_del_ensamble_de_quince_sobre_desarrollo": False, "tolerancia": 0.01}
    if mutar_fidelidad:
        mutar_fidelidad(fj)
    (root / C.REL_FIDELIDAD_JSON).write_text(json.dumps(fj), encoding="utf-8")
    escribir_tabla(root / C.REL_FIDELIDAD_CSV, [{"id_corrida": f["id_corrida"], "diferencia_maxima_absoluta": "0.0"} for f in filas])
    uj = {"umbral_primario": {"calibracion": {"umbral_repr": T_HAT, "k": 35, "n_pos": 38}}, "umbral_congelado": False}
    if mutar_umbral:
        mutar_umbral(uj)
    (root / C.REL_UMBRAL).write_text(json.dumps(uj), encoding="utf-8")
    for rel in (C.PINES_REQUERIDOS[5], C.PINES_REQUERIDOS[6]):
        (root / rel).write_text("{}", encoding="utf-8")
    config["referencia_g2"]["sha256_contenido_blob"] = G2_BLOB
    config["archivos_de_procedencia"] = {rel: sha((root / rel).read_bytes()) for rel in C.PINES_REQUERIDOS}
    config["miembros"]["sha256_bytes_archivo"] = config["archivos_de_procedencia"][C.REL_MANIFIESTO_CSV]
    assert base in C.REL_MANIFIESTO_CSV
    return root, config


def correr(root, config, **kw):
    kw.setdefault("verificar", verificar_falso())
    kw.setdefault("entorno", dict(ENTORNO))
    return C.ejecutar_verificar(root, config, autorizacion=AUT, **kw)


def sin_salida(root):
    return not (Path(root) / "results" / "r8_v2").exists()


# ============================================================================ configuración y protocolo reales
def problemas_protocolo_registrado(protocolo: dict, contenido: bytes) -> list[str]:
    """Esquema de dos commits ya completado: commit (40 hex) y SHA-256 del contenido del protocolo (64 hex) deben estar registrados, y el SHA-256 debe ser el del contenido real
    (fin de línea normalizado a LF: el blob de Git se calcula sobre el texto versionado, y `core.autocrlf` puede alterar la copia de trabajo). Sin Git."""
    p = []
    commit, digest = protocolo.get("git_commit"), protocolo.get("sha256_contenido_blob")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit) or set(commit) == {"0"}:
        p.append("git_commit ausente o con formato inválido (se esperan 40 hex en minúsculas)")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        p.append("sha256_contenido_blob ausente o con formato inválido (se esperan 64 hex en minúsculas)")
    elif digest != sha(contenido.replace(b"\r\n", b"\n")):
        p.append("sha256_contenido_blob no coincide con el contenido real del protocolo")
    return p


def test_config_real_valida_y_no_declara_congelamiento():
    c = C.cargar_config(REPO / C.CONFIG_REL)
    assert c["sistema"]["congelado"] is False and c["sistema"]["n_miembros"] == 15           # sigue falso: no hay declaración formal de congelamiento
    assert c["sistema"]["umbral"]["t_hat"] == "0.08456226376195748" == T_HAT
    assert repr(float(c["sistema"]["umbral"]["t_hat"])) == T_HAT
    assert problemas_protocolo_registrado(c["protocolo"], (REPO / C.PROTOCOLO_REL).read_bytes()) == []   # esquema de dos commits completado (commit 3)
    assert c["miembros"]["sha256_bytes_archivo"] == c["archivos_de_procedencia"][C.REL_MANIFIESTO_CSV]
    assert c["acceso_test"]["estado"] == "cerrado"


def test_comprobacion_del_protocolo_registrado_discrimina():
    contenido = b"protocolo sintetico\nlinea 2\n"
    ok = {"git_commit": "0123456789abcdef0123456789abcdef01234567", "sha256_contenido_blob": sha(contenido)}
    assert problemas_protocolo_registrado(ok, contenido) == []
    assert problemas_protocolo_registrado(ok, contenido.replace(b"\n", b"\r\n")) == []           # fin de línea convertido por Git: mismo contenido versionado
    for malo in ({"git_commit": None, "sha256_contenido_blob": None}, {**ok, "git_commit": None}, {**ok, "sha256_contenido_blob": None}, {**ok, "git_commit": "009f1bb"},
                 {**ok, "git_commit": "0" * 40}, {**ok, "git_commit": ok["git_commit"].upper()}, {**ok, "sha256_contenido_blob": "g" * 64}, {**ok, "sha256_contenido_blob": "1" * 64}):
        assert problemas_protocolo_registrado(malo, contenido), malo
    assert problemas_protocolo_registrado(ok, contenido + b"x")                                  # contenido alterado


def test_commit_registrado_corresponde_al_blob_del_protocolo_si_git_lo_tiene():
    """Correspondencia con Git real SOLO si el commit está disponible en el clon (si no, se omite: las pruebas sintéticas no dependen de Git)."""
    import shutil
    import subprocess
    p = C.cargar_config(REPO / C.CONFIG_REL)["protocolo"]
    if shutil.which("git") is None or not p.get("git_commit"):
        pytest.skip("git no disponible o commit no registrado")
    try:
        r = subprocess.run(["git", "-C", str(REPO), "cat-file", "blob", f"{p['git_commit']}:{p['ruta']}"], capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("git no ejecutable")
    if r.returncode != 0:
        pytest.skip("el commit registrado no está disponible en este clon")
    assert sha(r.stdout) == p["sha256_contenido_blob"]


def test_protocolo_real_coherente_y_sin_hash_propio():
    texto = (REPO / C.PROTOCOLO_REL).read_text(encoding="utf-8")
    assert all(f in texto for f in C.FRASES_PROTOCOLO)
    assert not re.search(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", texto)


def test_config_rechaza_deriva_del_sistema():
    c = C.cargar_config(REPO / C.CONFIG_REL)
    for ruta, valor in (("umbral.t_hat", "0.0846"), ("congelado", True), ("regla", "p > t_hat"), ("n_miembros", 14), ("arquitectura", "convnext_tiny"), ("representacion", "B")):
        d = copy.deepcopy(c)
        if ruta.startswith("umbral."):
            d["sistema"]["umbral"]["t_hat"] = valor
        else:
            d["sistema"][ruta] = valor
        with pytest.raises(GuardError):
            C.validar_config(d)


def test_config_rechaza_pin_mal_formado_o_miembros_distintos_del_manifiesto():
    c = C.cargar_config(REPO / C.CONFIG_REL)
    d = copy.deepcopy(c)
    d["archivos_de_procedencia"][C.REL_UMBRAL] = "ABC"
    with pytest.raises(GuardError):
        C.validar_config(d)
    d = copy.deepcopy(c)
    d["miembros"]["sha256_bytes_archivo"] = "0" * 64
    with pytest.raises(GuardError):
        C.validar_config(d)


# ============================================================================ caso feliz y guardas generales
def test_caso_feliz_escribe_resumen_y_no_declara_congelamiento(tmp_path):
    root, config = armar(tmp_path)
    r = correr(root, config)
    res = json.loads((r["dir"] / "resumen_verificacion.json").read_text(encoding="utf-8"))
    assert r["reporte_aceptado"] and r["advertencias_entorno"] == []
    assert res["congelado"] is False and "NO declara el congelamiento" in res["nota"]
    assert len(res["miembros"]) == 15 and all(m["sha256_recalculado"] == m["sha256_bytes_archivo_checkpoint"] for m in res["miembros"])
    assert res["coherencia"]["t_hat_resultado_umbral"] == T_HAT and res["coherencia"]["k"] == 35 and res["autorizacion"] == AUT
    assert res["coherencia"]["fidelidad_diferencia_maxima_absoluta"] == 0.0
    assert r["dir"].parent.as_posix().endswith("results/r8_v2/phash_le8_v1/sistema_congelado_v1/verificacion")


@pytest.mark.parametrize("aut", ["", "CONFIRMADO", "D-058", "confirmado:D-058", "CONFIRMADO:D-58"])
def test_sin_autorizacion_valida_no_se_lee_ni_se_escribe(tmp_path, aut):
    root, config = armar(tmp_path)
    with pytest.raises(GuardError):
        C.ejecutar_verificar(root, config, autorizacion=aut, verificar=verificar_falso(), entorno=dict(ENTORNO))
    assert sin_salida(root)


def test_integridad_con_bloqueos_detiene_y_no_escribe(tmp_path):
    root, config = armar(tmp_path)
    with pytest.raises(GuardError, match="protocolo_no_congelado"):
        correr(root, config, verificar=verificar_falso(bloqueos=[{"codigo": "protocolo_no_congelado", "detalle": "x"}]))
    assert sin_salida(root)


def test_integridad_que_cambia_durante_la_verificacion_invalida_la_salida(tmp_path):
    root, config = armar(tmp_path)
    estados = iter([verificar_falso("a" * 40)(), verificar_falso("b" * 40)()])
    r = correr(root, config, verificar=lambda: next(estados))
    assert r["reporte_aceptado"] is False
    assert "git_commit_head" in r["resumen"]["diferencias_inicio_fin"] and r["resumen"]["invalida_integridad"] is True


def test_no_sobrescribe_directorio_existente(tmp_path):
    root, config = armar(tmp_path)
    correr(root, config)
    with pytest.raises(GuardError, match="K5"):
        correr(root, config)


def test_salida_fuera_de_results_r8_v2_se_rechaza(tmp_path):
    root, config = armar(tmp_path)
    for malo in (root / "results" / "r7_v2" / "x", root / "results" / "r8_v2" / "test" / "x", root / "otra"):
        with pytest.raises(GuardError):
            correr(root, config, destino=malo)


def test_falta_r6_o_elegida_distinta_bloquea(tmp_path):
    root, config = armar(tmp_path)
    (root / config["referencia_r6"]["salida"]).write_text(json.dumps({"elegida": {"condicion": "B", "arquitectura": "densenet121"}}), encoding="utf-8")
    with pytest.raises(GuardError):
        correr(root, config)
    (root / config["referencia_r6"]["salida"]).unlink()
    with pytest.raises(GuardError):
        correr(root, config)
    assert sin_salida(root)


# ============================================================================ pines, miembros y coherencia
@pytest.mark.parametrize("rel", C.PINES_REQUERIDOS)
def test_archivo_de_procedencia_alterado_o_ausente_bloquea(tmp_path, rel):
    root, config = armar(tmp_path)
    (root / rel).write_bytes((root / rel).read_bytes() + b" ")
    with pytest.raises(GuardError):
        correr(root, config)
    (root / rel).unlink()
    with pytest.raises(GuardError, match="no existe"):
        correr(root, config)
    assert sin_salida(root)


def test_miembros_versionados_distintos_del_manifiesto_bloquean(tmp_path):
    root, config = armar(tmp_path)
    p = root / C.MIEMBROS_REL
    p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))                                  # p. ej. fin de línea convertido por Git
    with pytest.raises(GuardError, match="SHA-256"):
        correr(root, config)
    p.unlink()
    with pytest.raises(GuardError, match="no existe"):
        correr(root, config)


def test_menos_de_quince_miembros_bloquea(tmp_path):
    root, config = armar(tmp_path, mutar_filas=lambda f: f[:-1])
    with pytest.raises(GuardError, match="15"):
        correr(root, config)


@pytest.mark.parametrize("mut,patron", [
    (lambda f: f[:-1] + [{**f[0]}], "duplicado"),                                         # miembro repetido
    (lambda f: [{**f[0], "semilla": 99}] + f[1:], "fuera"),                               # semilla fuera de la rejilla
    (lambda f: [{**f[0], "arquitectura": "convnext_tiny"}] + f[1:], "densenet121"),
    (lambda f: [{**f[0], "estado": "fallida"}] + f[1:], "completado"),
    (lambda f: [{**f[0], "invalida_integridad": "True"}] + f[1:], "completado"),
    (lambda f: [{**f[0], "coincide": "False"}] + f[1:], "completado"),
    (lambda f: [{**f[0], "sha256_checkpoint_registrado_run_config": "b" * 64}] + f[1:], "run_config"),
    (lambda f: [{**f[0], "sha256_bytes_archivo_checkpoint": "xyz"}] + f[1:], "SHA-256"),
    (lambda f: [{**f[0], "sha256_contenido_blob_config_g2": "b" * 64}] + f[1:], "configuración de G2"),
    (lambda f: [{**f[0], "sha256_bytes_archivo_particion": "b" * 64}] + f[1:], "partición"),
    (lambda f: [{**f[0], "ruta_checkpoint": "data/test/x.pth"}] + f[1:], "test"),             # componente `test`
])
def test_estructura_de_miembros_se_bloquea(tmp_path, mut, patron):
    root, config = armar(tmp_path, mutar_filas=mut)
    with pytest.raises(GuardError, match=patron):
        correr(root, config)
    assert sin_salida(root)


def test_checkpoint_repetido_entre_miembros_bloquea(tmp_path):
    def mut(f):
        f[1] = {**f[1], "sha256_bytes_archivo_checkpoint": f[0]["sha256_bytes_archivo_checkpoint"], "sha256_checkpoint_registrado_run_config": f[0]["sha256_bytes_archivo_checkpoint"]}
        return f
    root, config = armar(tmp_path, mutar_filas=mut)
    with pytest.raises(GuardError, match="duplicados"):
        correr(root, config)


@pytest.mark.parametrize("mut", [
    lambda m: m.update(advertencias=["blob distinto"]), lambda m: m.update(reporte_aceptado=False), lambda m: m.update(invalida_integridad=True),
    lambda m: m.update(n_miembros=14), lambda m: m["miembros"][0].update(sha256_bytes_archivo_checkpoint="d" * 64), lambda m: m["miembros"].pop()])
def test_manifiesto_json_incoherente_bloquea(tmp_path, mut):
    root, config = armar(tmp_path, mutar_manifiesto=mut)
    with pytest.raises(GuardError):
        correr(root, config)
    assert sin_salida(root)


@pytest.mark.parametrize("mut", [
    lambda f: f.update(alerta_global=True), lambda f: f.update(t_hat="0.1"), lambda f: f.update(reporte_aceptado=False), lambda f: f.update(n_miembros=14),
    lambda f: f.update(metricas_del_ensamble_de_quince_sobre_desarrollo=True), lambda f: f.update(sha256_bytes_archivo_manifiesto_csv="e" * 64)])
def test_fidelidad_incoherente_bloquea(tmp_path, mut):
    root, config = armar(tmp_path, mutar_fidelidad=mut)
    with pytest.raises(GuardError):
        correr(root, config)


def test_fidelidad_por_encima_de_la_tolerancia_bloquea(tmp_path):
    root, config = armar(tmp_path)
    p = root / C.REL_FIDELIDAD_CSV
    p.write_text(p.read_text(encoding="utf-8").replace("0.0\n", "0.5\n", 1), encoding="utf-8")
    config["archivos_de_procedencia"][C.REL_FIDELIDAD_CSV] = sha(p.read_bytes())
    with pytest.raises(GuardError, match="tolerancia"):
        correr(root, config)


@pytest.mark.parametrize("mut", [
    lambda u: u["umbral_primario"]["calibracion"].update(umbral_repr="0.0845622637619575"), lambda u: u["umbral_primario"]["calibracion"].update(k=36),
    lambda u: u["umbral_primario"]["calibracion"].update(n_pos=37), lambda u: u.update(umbral_congelado=True)])
def test_t_hat_del_resultado_de_umbral_debe_ser_el_del_yaml(tmp_path, mut):
    root, config = armar(tmp_path, mutar_umbral=mut)
    with pytest.raises(GuardError):
        correr(root, config)
    assert sin_salida(root)


# ============================================================================ checkpoints y entorno
def test_checkpoint_ausente_bloquea(tmp_path):
    root, config = armar(tmp_path, sin_ckpt=(43, 2))
    with pytest.raises(GuardError, match="ausente"):
        correr(root, config)
    assert sin_salida(root)


def test_checkpoint_alterado_bloquea(tmp_path):
    root, config = armar(tmp_path, alterar_ckpt=(44, 4))
    with pytest.raises(GuardError, match="SHA-256"):
        correr(root, config)
    assert sin_salida(root)


def test_entorno_distinto_es_advertencia_no_bloqueo(tmp_path):
    root, config = armar(tmp_path)
    r = correr(root, config, entorno={**ENTORNO, "torch": "2.6.0+cu124"})
    assert r["reporte_aceptado"] and len(r["advertencias_entorno"]) == 1 and "torch" in r["advertencias_entorno"][0]


# ============================================================================ alcance del módulo
def test_modulo_no_tiene_inferencia_ni_metricas_ni_importa_torch():
    fuente = Path(C.__file__).read_text(encoding="utf-8")
    assert not any(hasattr(C, n) for n in ("inferir", "predecir", "calcular_metricas", "evaluar"))
    assert not re.search(r"^\s*(import|from)\s+(torch|torchvision)\b", fuente, re.M)               # el entorno se consulta por importación diferida y opcional


def test_cli_codigos_de_salida(tmp_path):
    root, config = armar(tmp_path)
    assert C.main(["--modo", "verificar"], root=root, config=config, verificar=verificar_falso(), entorno=dict(ENTORNO)) == U.CODIGO_SALIDA_GUARDA
    assert sin_salida(root)
    assert C.main(["--modo", "verificar", "--autorizacion", AUT], root=root, config=config, verificar=verificar_falso(), entorno=dict(ENTORNO)) == 0
    assert C.main(["--modo", "verificar", "--autorizacion", AUT], root=root, config=config, verificar=verificar_falso(), entorno=dict(ENTORNO)) == U.CODIGO_SALIDA_GUARDA
