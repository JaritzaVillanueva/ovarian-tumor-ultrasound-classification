"""Pruebas sintéticas (CPU) de `delta_ensamble` (R7, D-051): pareo con los mismos `b`, IC estratificado vs no estratificado, ajuste 1 − 0.05/3, media dentro del remuestreo, ROC-AUC,
verificación de la entrada por SHA-256, mismos `b` de G2, leyendas y guardas. No abren test ni datos reales; no entrenan modelos.
Comando: py -3.13 -B -m pytest tests/test_r7_v2_delta_ensamble.py -q -p no:cacheprovider
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("yaml")
pytest.importorskip("PIL")

from src.r4_v2 import bootstrap as BS  # noqa: E402
from src.r4_v2 import common as C4  # noqa: E402
from src.r4_v2 import metricas as M  # noqa: E402
from src.r4_v2.common import GuardError  # noqa: E402
from src.r7_v2 import delta_ensamble as D  # noqa: E402
from src.r7_v2 import umbral_sistema as U  # noqa: E402
from tests.test_r7_v2_umbral import AUT, N_B, REPO, armar, escribir_r6, leer_csv, verificar_u  # noqa: E402,F401
from tests.test_r7_v2_umbral import _git_aislado  # noqa: E402,F401  (fixture autouse)
from tests.test_r7_v2_umbral import correr as correr_umbral  # noqa: E402


def correr(e, destino=None, **kw):
    args = dict(autorizacion=AUT, verificar=verificar_u(), n_remuestreos=N_B) | kw
    return D.ejecutar(e.root, e.config_d, e.config_g2, destino=destino, **args)


def preparar(tmp_path_factory, nombre):
    e = armar(tmp_path_factory.mktemp(nombre), por_fold=20, pos_por_fold=4)                  # 100 imágenes, 20 positivas
    correr_umbral(e)                                                                         # genera umbral_sistema_v1 sintético (q_barra y trazabilidad)
    e.config_d = D.cargar_config(REPO / D.CONFIG_REL)
    (e.root / D.PROTOCOLO_REL).write_bytes((REPO / D.PROTOCOLO_REL).read_bytes())
    e.entrada = e.root / D.ENTRADA_REL
    return e


@pytest.fixture(scope="module")
def banco(tmp_path_factory):
    e = preparar(tmp_path_factory, "banco_delta")
    e.res = correr(e)
    e.dir = e.res["dir"]
    return e


# ============================================================================ pareo (mismos b) con configuración de ΔAP conocido
def _escenario(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = (np.arange(n) % 6 == 0).astype(np.int64)
    gid = [f"g{i // 2}" for i in range(n)]                                                   # grupos de 2 imágenes (misma clase por construcción)
    y = y.reshape(-1, 2)[:, :1].repeat(2, axis=1).ravel()
    bueno = np.clip(0.5 + 0.30 * (2 * y - 1) + rng.normal(0, 0.25, n), 0.01, 0.99)
    malo = np.clip(0.5 + 0.05 * (2 * y - 1) + rng.normal(0, 0.30, n), 0.01, 0.99)
    cols = {"q_barra": bueno.copy(), "p_s42": malo, "p_s43": bueno.copy(), "p_s44": np.clip(0.5 + 0.15 * (2 * y - 1) + rng.normal(0, 0.28, n), 0.01, 0.99)}
    gpi, clase, _ = BS.indexar_grupos(gid, y)
    ce = BS.generar_conteos(clase, 400, 20261121, True)
    cn = BS.generar_conteos(clase, 400, 20261121, False)
    return y, cols, gpi, ce, cn


def test_pareo_con_los_mismos_b_y_delta_conocido():
    y, cols, gpi, ce, cn = _escenario()
    res = D.analizar(y, cols, gpi, ce, cn)
    por = {c["contraste"]: c for c in res["ap"]}
    assert list(por) == ["q_barra - p_s42", "q_barra - p_s43", "q_barra - p_s44", "q_barra - media de las individuales"]
    ident = por["q_barra - p_s43"]                                                           # q_barra idéntico a p_s43: Δ = 0 en CADA remuestreo (solo si los b son los mismos)
    assert ident["estimacion"] == 0.0 and (ident["estratificado"]["ic_inf"], ident["estratificado"]["ic_sup"]) == (0.0, 0.0) and not ident["estratificado"]["ic_excluye_cero"]
    assert (ident["no_estratificado"]["ic_inf"], ident["no_estratificado"]["ic_sup"]) == (0.0, 0.0) and ident["estratificado"]["n_remuestreos_validos"] == 400
    p42 = por["q_barra - p_s42"]
    assert p42["estimacion"] == pytest.approx(M.ap(y, cols["q_barra"]) - M.ap(y, cols["p_s42"])) and p42["estimacion"] > 0 and p42["estratificado"]["ic_excluye_cero"]
    prep = {k: M.preparar(y, v) for k, v in (("q", cols["q_barra"]), ("s", cols["p_s42"]))}                           # recomputo manual con los MISMOS b
    d = BS.distribuciones(prep, ce, gpi)
    inf, sup, n_ok = BS.percentiles(d["q"]["ap"] - d["s"]["ap"])
    assert (p42["estratificado"]["ic_inf"], p42["estratificado"]["ic_sup"], p42["estratificado"]["n_remuestreos_validos"]) == (inf, sup, n_ok)
    assert por["q_barra - p_s42"]["tipo"] == "individual" and por["q_barra - media de las individuales"]["tipo"] == "resumen descriptivo"


def test_ic_estratificado_distinto_del_no_estratificado_y_ajuste_mas_ancho():
    y, cols, gpi, ce, cn = _escenario()
    res = D.analizar(y, cols, gpi, ce, cn)
    for c in res["ap"][:3]:
        if c["contraste"] == "q_barra - p_s43":
            continue                                                                         # degenerado (Δ = 0)
        e, n = c["estratificado"], c["no_estratificado"]
        assert (e["ic_inf"], e["ic_sup"]) != (n["ic_inf"], n["ic_sup"])
        for nom, conv in (("ajustado_estratificado", e), ("ajustado_no_estratificado", n)):
            a = c[nom]
            assert a["nivel"] == pytest.approx(1 - 0.05 / 3) == D.NIVEL_AJUSTADO
            assert a["ic_sup"] - a["ic_inf"] > conv["ic_sup"] - conv["ic_inf"] and a["ic_inf"] <= conv["ic_inf"] and a["ic_sup"] >= conv["ic_sup"]
    c = res["ap"][0]
    assert (c["ajustado_estratificado"]["ic_inf"], c["ajustado_estratificado"]["ic_sup"]) == BS.percentiles(
        BS.distribuciones({"q": M.preparar(y, cols["q_barra"]), "s": M.preparar(y, cols["p_s42"])}, ce, gpi)["q"]["ap"] - BS.distribuciones(
            {"q": M.preparar(y, cols["q_barra"]), "s": M.preparar(y, cols["p_s42"])}, ce, gpi)["s"]["ap"], 1 - 0.05 / 3)[:2]       # percentiles α/6 y 1 − α/6
    assert res["ap"][3]["ajustado_estratificado"] is None and res["ap"][3]["rotulo"] == D.ROTULO_DESCRIPTIVO            # el resumen descriptivo no lleva ajuste ni es contraste


def test_media_dentro_del_remuestreo_no_es_el_delta_contra_la_media_puntual():
    y, cols, gpi, ce, cn = _escenario()
    res = D.analizar(y, cols, gpi, ce, cn)
    desc = res["ap"][3]
    preps = {k: M.preparar(y, v) for k, v in (("q", cols["q_barra"]), ("a", cols["p_s42"]), ("b", cols["p_s43"]), ("c", cols["p_s44"]))}
    d = BS.distribuciones(preps, ce, gpi)
    dentro = d["q"]["ap"] - np.mean([d["a"]["ap"], d["b"]["ap"], d["c"]["ap"]], axis=0)                             # media DENTRO de cada remuestreo
    media_puntual = np.mean([M.ap(y, cols[k]) for k in ("p_s42", "p_s43", "p_s44")])
    contra_puntual = d["q"]["ap"] - media_puntual                                                                    # lo que NO se calcula
    assert (desc["estratificado"]["ic_inf"], desc["estratificado"]["ic_sup"]) == BS.percentiles(dentro)[:2]
    assert not np.allclose(dentro, contra_puntual) and BS.percentiles(dentro)[:2] != BS.percentiles(contra_puntual)[:2]
    assert desc["estimacion"] == pytest.approx(M.ap(y, cols["q_barra"]) - media_puntual) and desc["valor_comparador"] == pytest.approx(media_puntual)


def test_roc_auc_como_metrica_secundaria():
    y, cols, gpi, ce, cn = _escenario()
    res = D.analizar(y, cols, gpi, ce, cn)
    assert [c["contraste"] for c in res["roc_auc"]] == [c["contraste"] for c in res["ap"]]
    c = res["roc_auc"][0]
    assert c["estimacion"] == pytest.approx(M.roc_auc(y, cols["q_barra"]) - M.roc_auc(y, cols["p_s42"])) and c["estratificado"]["n_remuestreos_validos"] == 400
    assert res["roc_auc"][1]["estimacion"] == 0.0 and res["roc_auc"][3]["rotulo"] == D.ROTULO_DESCRIPTIVO


# ============================================================================ salidas del flujo completo
def test_salidas_csv_json_y_trazabilidad(banco):
    esperados = {"delta_ap.csv", "delta_roc_auc.csv", "trazabilidad.json", "resumen_delta_ensamble.json"}
    assert {p.name for p in banco.dir.iterdir()} == esperados
    enc, filas = leer_csv(banco.dir / "delta_ap.csv")
    assert enc.startswith("#") and U.LEYENDA in enc and "sin uso de test" in enc and "no es evidencia sobre el ensamble de quince ni sobre test" in enc and len(filas) == 4
    assert [r["tipo"] for r in filas] == ["individual"] * 3 + ["resumen descriptivo"] and filas[3]["rotulo"] == D.ROTULO_DESCRIPTIVO and filas[0]["rotulo"] == D.ROTULO_AJUSTE
    assert all(r["estratificado_ic_excluye_cero"] in ("True", "False") and int(r["estratificado_n_remuestreos_validos"]) > 0 for r in filas)
    assert filas[3]["ajustado_estratificado_ic_inf"] == "" and filas[0]["ajustado_estratificado_ic_inf"] != ""
    assert len(leer_csv(banco.dir / "delta_roc_auc.csv")[1]) == 4
    q = json.loads((banco.entrada / "trazabilidad.json").read_text(encoding="utf-8"))["sha256_bytes_archivo"]["q_barra_oof.csv"]
    t = json.loads((banco.dir / "trazabilidad.json").read_text(encoding="utf-8"))
    assert t["entradas"]["q_barra_oof.csv"] == q == C4.sha256_archivo(banco.entrada / "q_barra_oof.csv") and t["git_commit_head"] == "a" * 40
    assert t["sha256_bytes_archivo_particion"] == banco.particion.sha256_bytes_archivo and t["numpy_version"] == np.__version__
    # la estimación del flujo coincide con el cálculo directo sobre las columnas del archivo
    y = np.array([r["label_v2"] for r in banco.particion.filas])
    cols = D.leer_entrada(banco.entrada, banco.particion.filas)["columnas"]
    assert float(filas[0]["estimacion"]) == pytest.approx(M.ap(y, cols["q_barra"]) - M.ap(y, cols["p_s42"]))


def test_leyendas_banderas_y_ausencia_de_afirmaciones_de_validez(banco):
    r = json.loads((banco.dir / "resumen_delta_ensamble.json").read_text(encoding="utf-8"))
    txt = " ".join(r["leyendas"])
    for frase in ("exploratorio, no confirmatorio", "(D-048)", "NO es el único comparador", "no asumen independencia", "pHash ≤ 8 exploratorio", "38 positivas",
                  "OOF de un ensamble de TRES modelos del mismo pliegue",
                  "no reabre R6", "sin afirmar utilidad clínica", D.ROTULO_AJUSTE):
        assert frase in txt, frase
    assert r["banderas"] == {"exploratorio": True, "reabre_r6": False, "cambia_arquitectura": False, "evidencia_sobre_ensamble_de_quince": False, "evidencia_sobre_test": False,
                             "utilidad_clinica_afirmada": False, "mejor_semilla_individual_es_el_unico_comparador": False}
    assert r["multiplicidad"]["k_contrastes"] == 3 and r["multiplicidad"]["nivel_ajustado"] == pytest.approx(1 - 0.05 / 3) and r["multiplicidad"]["rotulo"] == D.ROTULO_AJUSTE
    assert r["multiplicidad"]["percentiles"][0] == pytest.approx(0.05 / 6 * 100) and r["multiplicidad"]["percentiles"][1] == pytest.approx(100 - 0.05 / 6 * 100)
    assert r["reporte_aceptado"] is True and any("38 positivas" in l for l in r["limites"]) and any("test" in l for l in r["limites"])
    prohibidas = ("clínicamente útil", "sistema validado", "validado", "validez", "significativ", "p-valor", "hipótesis nula", "rechaza", "equivalen", "no inferior")
    for p in banco.dir.iterdir():
        t = p.read_text(encoding="utf-8")
        assert not [x for x in prohibidas if x in t], (p.name, [x for x in prohibidas if x in t])
        for m in re.finditer("reabre R6", t):
            assert t[max(0, m.start() - 3):m.start()] == "no ", (p.name, t[max(0, m.start() - 20):m.end()])           # solo en la forma «no reabre R6»
    for p in banco.dir.glob("*.csv"):
        assert U.LEYENDA in p.read_text(encoding="utf-8").splitlines()[0]


# ============================================================================ entrada verificada: SHA-256, columnas, imágenes
def _copia(e, tmp_path, nombre):
    destino = tmp_path / nombre
    shutil.copytree(e.entrada, destino)
    return destino


def _recalcular_sha(origen):
    t = json.loads((origen / "trazabilidad.json").read_text(encoding="utf-8"))
    t["sha256_bytes_archivo"]["q_barra_oof.csv"] = C4.sha256_archivo(origen / "q_barra_oof.csv")
    (origen / "trazabilidad.json").write_text(json.dumps(t), encoding="utf-8")


def test_sha256_de_q_barra_y_estructura_de_la_entrada(banco, tmp_path):
    ok = _copia(banco, tmp_path, "ok")
    assert D.leer_entrada(ok, banco.particion.filas)["columnas"]["q_barra"].size == len(banco.particion.filas)
    q = ok / "q_barra_oof.csv"
    lineas = q.read_text(encoding="utf-8").splitlines()
    q.write_bytes(q.read_bytes().replace(b"0.", b"0.1", 1))
    with pytest.raises(GuardError, match="no coincide con el SHA-256"):                      # cualquier cambio de bytes bloquea
        D.leer_entrada(ok, banco.particion.filas)
    for nombre, mutar, patron in (
            ("sin_columna", lambda L: [l.replace(",p_s44", "") if i == 1 else ",".join(l.split(",")[:3] + l.split(",")[4:]) if i > 1 else l for i, l in enumerate(L)], "columnas"),
            ("falta_imagen", lambda L: L[:-1], "no corresponde a la partición"),
            ("duplicada", lambda L: L[:-1] + [L[-2]], "duplicados"),
            ("no_numerico", lambda L: L[:2] + [L[2].rsplit(",", 1)[0] + ",abc"] + L[3:], "no numéricos")):
        o = _copia(banco, tmp_path, nombre)
        (o / "q_barra_oof.csv").write_text("\n".join(mutar(lineas)) + "\n", encoding="utf-8")
        _recalcular_sha(o)                                                                   # SHA coherente: el fallo es de estructura, no de hash
        with pytest.raises(GuardError, match=patron):
            D.leer_entrada(o, banco.particion.filas)
    sin = _copia(banco, tmp_path, "sin_traz")
    (sin / "trazabilidad.json").unlink()
    with pytest.raises(GuardError, match="falta q_barra_oof.csv o trazabilidad.json"):
        D.leer_entrada(sin, banco.particion.filas)
    fuera = _copia(banco, tmp_path, "fuera_de_rango")
    t = (fuera / "q_barra_oof.csv").read_text(encoding="utf-8").splitlines()
    t[2] = t[2].rsplit(",", 1)[0] + ",1.5"
    (fuera / "q_barra_oof.csv").write_text("\n".join(t) + "\n", encoding="utf-8")
    _recalcular_sha(fuera)
    with pytest.raises(GuardError, match="fuera de"):
        D.leer_entrada(fuera, banco.particion.filas)


def test_ejecutar_bloquea_con_entrada_alterada_sin_crear_nada(banco, tmp_path):
    o = _copia(banco, tmp_path, "alterada")
    (o / "q_barra_oof.csv").write_bytes((o / "q_barra_oof.csv").read_bytes() + b" ")
    with pytest.raises(GuardError, match="no coincide con el SHA-256"):
        correr(banco, destino=banco.root / "results/r7_v2/delta_alt", origen=o)
    assert not (banco.root / "results/r7_v2/delta_alt").exists()


# ============================================================================ mismos b de G2
def test_mismos_b_de_g2_verificados_con_npz_y_bloqueo_si_difieren(tmp_path_factory):
    e = preparar(tmp_path_factory, "delta_b")
    cfg = e.config_g2["bootstrap"]
    y = np.array([r["label_v2"] for r in e.particion.filas])
    _, clase, nombres = BS.indexar_grupos([r["group_id"] for r in e.particion.filas], y)
    d = e.root / e.config_d["referencia_g2"]["indices_b_dir"]
    d.mkdir(parents=True)
    for et, s in (("estratificado", cfg["semilla"]), ("estratificado_semilla_comprobacion", cfg["semilla_de_comprobacion"])):
        BS.guardar_indices(d / f"indices_b_{et}_s{s}.npz", BS.generar_conteos(clase, cfg["n_remuestreos"], s, True), semilla=s, estratificado=True, nombres_de_grupo=nombres)
    r = correr(e, destino=e.root / "results/r7_v2/delta_b_ok", n_remuestreos=None)
    res = json.loads((r["dir"] / "resumen_delta_ensamble.json").read_text(encoding="utf-8"))
    assert res["n_remuestreos"] == 10000 and all(x["estado"].startswith("verificado") for x in res["indices_b_verificacion"])
    f = d / f"indices_b_estratificado_s{cfg['semilla']}.npz"
    f.unlink()
    c = BS.generar_conteos(clase, 10000, cfg["semilla"], True).copy()
    c[0, 0] += 1
    BS.guardar_indices(f, c, semilla=cfg["semilla"], estratificado=True, nombres_de_grupo=nombres)
    with pytest.raises(GuardError, match="NO coinciden"):
        correr(e, destino=e.root / "results/r7_v2/delta_b_mal", n_remuestreos=None)
    assert not (e.root / "results/r7_v2/delta_b_mal").exists()


# ============================================================================ guardas
def test_guarda_r6_autorizacion_k4_no_sobrescribe_y_determinismo(tmp_path_factory):
    e = preparar(tmp_path_factory, "delta_guardas")
    ruta = e.root / e.config_d["referencia_r6"]["salida"]
    original = ruta.read_bytes()
    ruta.unlink()
    with pytest.raises(GuardError, match="no existe"):
        correr(e, destino=e.root / "results/r7_v2/d_sin_r6")
    for elegida in ({"condicion": "B_hibrida", "arquitectura": "densenet121"}, {"condicion": "A", "arquitectura": "resnet50"}):
        escribir_r6(e.root, e.config_d, elegida)
        with pytest.raises(GuardError, match="R6 eligió"):
            correr(e, destino=e.root / "results/r7_v2/d_otra_r6")
        ruta.unlink()
    ruta.write_bytes(original)                                                               # el fixture explícito permite
    for valor in ("", None, "CONFIRMADO", "CONFIRMADO:D-51", "confirmado:D-051", "CONFIRMADO:AUT-051", " CONFIRMADO:D-051", "PENDIENTE", 51):
        with pytest.raises(GuardError, match="autorizacion"):
            correr(e, destino=e.root / "results/r7_v2/d_sin_aut", autorizacion=valor)
    vacio = e.root / "vacio"
    vacio.mkdir()
    assert D.main([], root=vacio) == U.CODIGO_SALIDA_GUARDA == 3 and list(vacio.iterdir()) == []
    for malo in (e.root / "results/r7_v2/test/x", e.root / "results/r4_v2/x", e.root / "fuera"):
        with pytest.raises(GuardError):
            correr(e, destino=malo)
    cfg = copy.deepcopy(e.config_d)
    cfg["salidas"]["delta"] = "results/r7_v2/phash_le8_v1/test/delta"
    with pytest.raises(GuardError):
        D.validar_config(cfg)
    assert not [p for p in (e.root / "results/r7_v2").glob("d_*")]
    a = correr(e, destino=e.root / "results/r7_v2/det_a")["dir"]
    b = correr(e, destino=e.root / "results/r7_v2/det_b")["dir"]
    for p in a.iterdir():
        if p.suffix == ".csv":
            assert C4.sha256_archivo(p) == C4.sha256_archivo(b / p.name), p.name
        else:
            ja, jb = json.loads(p.read_text(encoding="utf-8")), json.loads((b / p.name).read_text(encoding="utf-8"))
            ja.pop("fecha", None), jb.pop("fecha", None)
            assert ja == jb, p.name
    with pytest.raises(GuardError, match="ya existe"):
        correr(e, destino=a)
    antes = {p.name: C4.sha256_archivo(p) for p in a.iterdir()}
    with pytest.raises(GuardError):
        correr(e, destino=a)
    assert {p.name: C4.sha256_archivo(p) for p in a.iterdir()} == antes
    n = {"i": 0}

    def cambia():
        n["i"] += 1
        r = verificar_u()()
        if n["i"] > 1:
            r["git_commit_head"] = "z" * 40
        return r
    r = correr(e, destino=e.root / "results/r7_v2/d_integr", verificar=cambia)
    assert r["reporte_aceptado"] is False and "git_commit_head" in r["diferencias_inicio_fin"]
    n["i"] = 0
    assert D.main(["--autorizacion", AUT], root=e.root, config=e.config_d, config_g2=e.config_g2, verificar=cambia, n_remuestreos=N_B,
                  destino=e.root / "results/r7_v2/d_integr2") == U.CODIGO_SALIDA_INTEGRIDAD_INVALIDA == 4


def test_procedimiento_anti_deriva_y_yaml_protocolo_coherentes(tmp_path):
    config = D.cargar_config(REPO / D.CONFIG_REL)
    gc, sh = config["protocolo"]["git_commit"], config["protocolo"]["sha256_contenido_blob"]
    assert config["protocolo"]["trazabilidad"] == "D-051"
    assert (gc is None and sh is None) or (re.fullmatch("[0-9a-f]{40}", gc) and re.fullmatch("[0-9a-f]{64}", sh))
    if gc is None:
        lineas = (REPO / D.CONFIG_REL).read_text(encoding="utf-8").splitlines()
        assert lineas.count("  git_commit: null") == 1 and lineas.count("  sha256_contenido_blob: null") == 1
    cambios = [lambda c: c["procedimiento"]["multiplicidad"].update(k_contrastes=2), lambda c: c["procedimiento"].update(reabre_r6=True),
               lambda c: c["procedimiento"]["bootstrap"].update(n_remuestreos=1000), lambda c: c["procedimiento"]["contrastes_individuales"].update(semillas=[42]),
               lambda c: c["procedimiento"].update(exploratorio=False), lambda c: c["procedimiento"].update(evidencia_sobre_ensamble_de_quince=True),
               lambda c: c["acceso_test"].update(estado="abierto"), lambda c: c["referencia_r6"]["elegida"].update(arquitectura="resnet50"),
               lambda c: c["integridad"]["archivos_con_huella"].remove(D.PROTOCOLO_REL), lambda c: c["salidas"].update(delta="results/r7_v2/x"),
               lambda c: c.update(entrada_dir="results/r7_v2/phash_le8_v1/test/umbral")]
    for cambio in cambios:
        mala = copy.deepcopy(config)
        cambio(mala)
        with pytest.raises(GuardError):
            D.validar_config(mala)
    protocolo = (REPO / D.PROTOCOLO_REL).read_text(encoding="utf-8")
    assert len(protocolo.splitlines()) <= 60 and D.exigir_protocolo(REPO, config, D.FRASES_PROTOCOLO)
    for i, frase in enumerate(D.FRASES_PROTOCOLO):
        raiz = tmp_path / f"p{i}"
        (raiz / D.PROTOCOLO_REL).parent.mkdir(parents=True)
        (raiz / D.PROTOCOLO_REL).write_text(protocolo.replace(frase, "X"), encoding="utf-8")
        with pytest.raises(GuardError, match="no es coherente"):
            D.exigir_protocolo(raiz, config, D.FRASES_PROTOCOLO)
    (tmp_path / "h" / D.PROTOCOLO_REL).parent.mkdir(parents=True)
    (tmp_path / "h" / D.PROTOCOLO_REL).write_text(protocolo + "\n" + "b" * 64, encoding="utf-8")
    with pytest.raises(GuardError, match="no es coherente"):
        D.exigir_protocolo(tmp_path / "h", config, D.FRASES_PROTOCOLO)
    with pytest.raises(GuardError, match="no existe"):
        D.exigir_protocolo(tmp_path / "nada", config, D.FRASES_PROTOCOLO)


def _git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), "-c", "user.email=t@t", "-c", "user.name=t", *args], capture_output=True, text=True, check=True).stdout.strip()


def test_integridad_con_git_real_usa_la_ruta_del_protocolo_de_esta_configuracion(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "src/r7_v2").mkdir(parents=True)
    archivos = {"src/r7_v2/a.py": b"x = 1\n", D.PROTOCOLO_REL: b"protocolo delta\n", "g2.yaml": b"g2: 1\n"}
    for rel, b in archivos.items():
        (tmp_path / rel).write_bytes(b)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "c1")
    c1 = _git(tmp_path, "rev-parse", "HEAD")
    blob = hashlib.sha256(archivos[D.PROTOCOLO_REL]).hexdigest()
    cfg = {"integridad": {"rutas_sin_cambios": ["src/r7_v2"], "archivos_con_huella": list(archivos)}, "referencia_g2": {"config": "g2.yaml", "sha256_contenido_blob": hashlib.sha256(b"g2: 1\n").hexdigest()},
           "protocolo": {"ruta": D.PROTOCOLO_REL, "git_commit": None, "sha256_contenido_blob": None}}
    assert [b["codigo"] for b in D.verificar_integridad(tmp_path, cfg)["bloqueos"]] == ["protocolo_no_congelado"]
    cfg["protocolo"].update(git_commit=c1, sha256_contenido_blob=blob)
    assert not D.verificar_integridad(tmp_path, cfg)["bloqueos"]
    cfg["protocolo"]["sha256_contenido_blob"] = "0" * 64
    assert {b["codigo"] for b in D.verificar_integridad(tmp_path, cfg)["bloqueos"]} == {"protocolo_distinto_del_registrado", "protocolo_del_commit_registrado_distinto"}
    cfg["protocolo"]["sha256_contenido_blob"] = blob
    (tmp_path / "src/r7_v2/a.py").write_bytes(b"x = 2\n")
    assert [b["codigo"] for b in D.verificar_integridad(tmp_path, cfg)["bloqueos"]] == ["arbol_sucio"]
