"""Pruebas sintéticas (CPU) de `umbral_sistema` (R7): calibración, agregaciones m = 1/2/3, VPP, transferencia, t̂ recalculado por remuestreo, mismos `b` de G2,
leyendas, guardas (autorización, R6, integridad con Git real, K4, no sobrescribe) y determinismo. No abren test, ni datos reales; no entrenan modelos.
Comando: py -3.13 -B -m pytest tests/test_r7_v2_umbral.py -q -p no:cacheprovider
"""

from __future__ import annotations

import copy
import hashlib
import re
import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("yaml")
pytest.importorskip("PIL")

from src.r4_v2 import bootstrap as BS  # noqa: E402
from src.r4_v2 import calibracion as CAL  # noqa: E402
from src.r4_v2 import common as C4  # noqa: E402
from src.r4_v2 import entrenamiento as EN  # noqa: E402
from src.r4_v2 import metricas as M  # noqa: E402
from src.r4_v2 import particion as P4  # noqa: E402
from src.r4_v2.common import GuardError  # noqa: E402
from src.r7_v2 import g4_reporte as R  # noqa: E402
from src.r7_v2 import umbral_sistema as U  # noqa: E402
from tests.test_r5_v2 import construir_datos  # noqa: E402
from tests.test_r7_v2_g4 import REF, _git_aislado, sintetico, verificar_falso  # noqa: E402,F401

REPO = Path(__file__).resolve().parents[1]
AUT = "CONFIRMADO:D-049"
N_B = 200


def verificar_u(head="a" * 40):
    def f():
        h = {"sha256_bytes_archivo": "1" * 64, "sha256_contenido_blob": "c" * 64}
        return {"git_commit_head": head, "rama": "sintetica", "status_porcelain": [], "bloqueos": [], "archivos": {U.CONFIG_REL: h}}
    return f


def escribir_r6(root, config, elegida=None):
    ruta = Path(root) / config["referencia_r6"]["salida"]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps({"elegida": elegida or {"condicion": "A", "arquitectura": "densenet121"}}), encoding="utf-8")
    return ruta


def armar(root, por_fold=8, pos_por_fold=2):
    """Partición sintética, 15 corridas A|densenet121 sintéticas de G2, `seleccion_r6.json` sintético (fixture explícito) y copia del protocolo."""
    root = Path(root)
    _, config_g2 = construir_datos(root, por_fold=por_fold, pos_por_fold=pos_por_fold, escribir_mascaras=False)
    config = U.cargar_config(REPO / U.CONFIG_REL)
    particion = P4.cargar_particion(root, config_g2)
    escribir_r6(root, config)
    (root / U.PROTOCOLO_REL).parent.mkdir(parents=True, exist_ok=True)
    (root / U.PROTOCOLO_REL).write_bytes((REPO / U.PROTOCOLO_REL).read_bytes())
    cb = U.config_g2_sistema(config_g2, config)
    raiz_a = root / config_g2["diseno"]["raiz_resultados"].strip("/") / "completa"
    for s in cb["diseno"]["semillas"]:
        for k in cb["diseno"]["folds"]:
            r = EN.ejecutar_corrida(root, cb, particion, modo="completa", arq="densenet121", semilla=s, fold=k, raiz_modo=raiz_a, verificar=verificar_falso(),
                                    entrenar=sintetico(), referencia=REF, atestaciones={"autorizacion_ejecucion": "SINTETICA"})
            assert r["estado"] == "completada", r
    class Env:
        pass
    e = Env()
    e.root, e.config, e.config_g2, e.particion, e.raiz_a = root, config, config_g2, particion, raiz_a
    return e


def correr(e, destino=None, **kw):
    args = dict(autorizacion=AUT, verificar=verificar_u(), n_remuestreos=N_B) | kw
    return U.ejecutar(e.root, e.config, e.config_g2, destino=destino, **args)


def leer_csv(path):
    lineas = Path(path).read_text(encoding="utf-8").splitlines()
    import csv
    return lineas[0], list(csv.DictReader(lineas[1:]))


@pytest.fixture(scope="module")
def banco(tmp_path_factory):
    e = armar(tmp_path_factory.mktemp("banco_umbral"))
    e.res = correr(e)
    e.dir = e.res["dir"]
    return e


def datos_38(n=300, seed=0, redondeo=2):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.int64)
    y[rng.choice(n, 38, replace=False)] = 1
    q = np.round(np.clip(0.35 + 0.35 * y + rng.normal(0, 0.25, n), 0, 1), redondeo)
    return y, q


# ============================================================================ calibración (S*, k, empates, regla >=)
def test_calibracion_k35_con_38_positivas_y_empates_sin_romper():
    y, q = datos_38()
    q = q.copy()
    pos = np.flatnonzero(y == 1)
    orden = pos[np.argsort(-q[pos], kind="stable")]
    v = float(q[orden[34]])                                                                # valor de la 35.ª positiva
    q[orden[33:37]] = v                                                                    # las positivas 34.ª a 37.ª empatan en el valor de t̂
    q[np.flatnonzero(y == 0)[:5]] = v                                                      # y 5 negativas también
    c = U.calibracion_primaria(y, q, "0.90")
    assert c["n_pos"] == 38 and c["k"] == 35 == -(-9 * 38 // 10) and c["umbral"] == v and c["umbral_repr"] == repr(v)
    assert c["m_tp"] == c["tp"] == int(((q >= v) & (y == 1)).sum()) and c["tp"] >= 37
    assert c["n_pos_empatadas"] >= 4 and c["n_neg_empatadas"] >= 5                         # los empates NO se rompen: todas las empatadas son positivas (p >= umbral)
    assert c["fp"] == int(((q >= v) & (y == 0)).sum()) and c["tp"] + c["fn"] == 38 and c["fp"] + c["tn"] == int((y == 0).sum())
    assert c["sens_real"] == pytest.approx(c["tp"] / 38) and c["sens_real"] > 0.90
    ref = CAL.calibrar_umbral(y.tolist(), q.tolist(), "0.90")                              # misma función sin modificar
    assert ref["umbral"] == c["umbral"] and ref["k"] == c["k"]


def test_regla_mayor_o_igual_y_s_star_como_cadena_o_fraction_nunca_float():
    y = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    q = [0.1, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99]
    ev = CAL.evaluar_en_umbral(y, q, 0.5)
    assert ev["tp"] == 9 and ev["fp"] == 1                                                 # p == umbral es positiva
    assert U.calibracion_primaria(np.array(y), np.array(q), Fraction(9, 10))["k"] == 9
    with pytest.raises(GuardError, match="fracción exacta"):
        U.calibracion_primaria(np.array(y), np.array(q), 0.9)
    with pytest.raises(GuardError):
        U.calibracion_primaria(np.array(y), np.array(q), True)


def test_s_star_descriptivos_no_se_seleccionan_ni_se_destacan(banco):
    res = banco.res["resultados"]
    assert [d["s_star"] for d in res["s_star_descriptivos"]] == ["0.85", "0.95"] == list(U.S_DESCRIPTIVOS) and U.S_PRIMARIO == "0.90"
    enc, filas = leer_csv(banco.dir / "sensibilidad_S_star.csv")
    assert [r["s_star"] for r in filas] == ["0.85", "0.95"] and "SOLO descriptivos" in enc and "sin ranking" in enc
    assert not [c for c in filas[0] if any(w in c for w in ("mejor", "elegid", "selecc", "rank", "favorable"))]
    resumen = json.loads((banco.dir / "resumen_umbral.json").read_text(encoding="utf-8"))
    assert resumen["s_star"]["primario"] == "0.90" and resumen["s_star"]["descriptivos_solo"] == ["0.85", "0.95"] and "ninguna" in resumen["s_star"]["seleccion_entre_ellos"]
    assert resumen["t_hat_primario"] == res["umbral_primario"]["calibracion"]["umbral_repr"]


# ============================================================================ agregaciones m = 1, 2, 3 y AP/AUC
def test_agregaciones_m_son_promedios_aritmeticos_de_probabilidades():
    rng = np.random.default_rng(1)
    probs = {s: rng.random(50) for s in (42, 43, 44)}
    v = U.vectores_m(probs)
    assert len(v) == 7 and sorted(k for k in v if k.startswith("m1")) == ["m1_s42", "m1_s43", "m1_s44"]
    assert np.array_equal(v["m1_s43"][2], probs[43])
    assert np.allclose(v["m2_s42+s44"][2], (probs[42] + probs[44]) / 2) and np.allclose(v["m3_s42+s43+s44"][2], (probs[42] + probs[43] + probs[44]) / 3)
    assert v["m2_s43+s44"][1] == [43, 44] and v["m3_s42+s43+s44"][1] == [42, 43, 44] and [m for m, _, _ in v.values()].count(2) == 3


def test_ap_y_auc_de_las_agregaciones_con_ic_y_sin_afirmar_mejora(banco):
    ag = banco.res["resultados"]["agregaciones"]
    assert [(a["m"], a["etiqueta"]) for a in ag] == [(1, "m1_s42"), (1, "m1_s43"), (1, "m1_s44"), (1, "m1_media"), (2, "m2_s42+s43"), (2, "m2_s42+s44"), (2, "m2_s43+s44"),
                                                      (2, "m2_media"), (3, "m3_s42+s43+s44")]
    oof = banco.res["resultados"]["q_barra"]
    y = np.array([r["label_v2"] for r in banco.particion.filas])
    m3 = ag[-1]
    assert m3["ap"] == pytest.approx(M.metricas(y, oof)["ap"]) and m3["roc_auc"] == pytest.approx(M.metricas(y, oof)["roc_auc"])
    assert ag[3]["ap"] == pytest.approx(np.mean([a["ap"] for a in ag[:3]])) and ag[7]["ap"] == pytest.approx(np.mean([a["ap"] for a in ag[4:7]]))
    enc, filas = leer_csv(banco.dir / "agregaciones_ap_auc.csv")
    assert "sin afirmar mejora" in enc and len(filas) == 9 and all(r["ap_ic95_inf"] and r["roc_auc_ic95_sup"] for r in filas)


def test_q_barra_csv_es_la_media_de_las_tres_oof_y_conserva_la_precision(banco):
    enc, filas = leer_csv(banco.dir / "q_barra_oof.csv")
    assert enc.startswith("#") and list(filas[0]) == ["image_id", "fold", "label_v2", "p_s42", "p_s43", "p_s44", "q_barra"] and len(filas) == len(banco.particion.filas)
    for r, fila in zip(filas, banco.particion.filas):
        assert r["image_id"] == fila["image_id"] and int(r["fold"]) == fila["fold"] and int(r["label_v2"]) == fila["label_v2"]
        p = [float(r[c]) for c in ("p_s42", "p_s43", "p_s44")]
        assert float(r["q_barra"]) == pytest.approx(sum(p) / 3, abs=1e-12) and repr(float(r["q_barra"])) == r["q_barra"]


# ============================================================================ tabla frente a 0.5, VPP y diferencias
def test_vpp_con_la_prevalencia_calculada_y_diferencias_absolutas(banco):
    t = banco.res["resultados"]["tabla_vs_0_5"]
    n, npos = len(banco.particion.filas), sum(r["label_v2"] for r in banco.particion.filas)
    assert (t["n"], t["n_pos"]) == (n, npos) and t["prevalencia"] == pytest.approx(npos / n)
    a, b = t["q_barra_a_0_5"], t["q_barra_a_t_hat"]
    for d in (a, b):
        assert d["vpp"] == pytest.approx(d["tp"] / (d["tp"] + d["fp"])) and d["sens"] == pytest.approx(d["tp"] / (d["tp"] + d["fn"])) and d["esp"] == pytest.approx(d["tn"] / (d["tn"] + d["fp"]))
    dd = t["diferencias_t_hat_menos_0_5"]
    assert dd["sens"] == pytest.approx(b["sens"] - a["sens"]) and dd["esp"] == pytest.approx(b["esp"] - a["esp"])
    assert (dd["fp"], dd["fn"]) == (b["fp"] - a["fp"], b["fn"] - a["fn"]) and dd["vpp"] == pytest.approx(b["vpp"] - a["vpp"])
    assert set(t["por_semilla_a_0_5"]) == {42, 43, 44} and t["control_coherencia"]["valor_reportado_g2"] == 0.447368
    enc, filas = leer_csv(banco.dir / "tabla_vs_0_5.csv")
    assert "solo válido para la prevalencia del desarrollo; no transferible" in enc and [r["fila"] for r in filas][:2] == ["q_barra a 0.5", "q_barra a t_hat"]
    assert filas[-1]["fila"] == "diferencia (t_hat - 0.5)" and len(filas) == 6
    assert float(filas[1]["umbral"]) == banco.res["resultados"]["umbral_primario"]["calibracion"]["umbral"]


def test_vpp_y_sin_positivos_predichos_no_falla():
    y = np.array([0, 0, 1, 1])
    assert np.isnan(U.metricas_en_umbral(y, np.array([0.1, 0.2, 0.3, 0.4]), 0.9)["vpp"])


# ============================================================================ transferencia y t̂ recalculado en cada remuestreo
def test_transferencia_coincide_con_comprobar_transferencia(banco):
    tr = banco.res["resultados"]["transferencia"]
    y = [r["label_v2"] for r in banco.particion.filas]
    fold = [r["fold"] for r in banco.particion.filas]
    q = banco.res["resultados"]["q_barra"].tolist()
    ref = CAL.comprobar_transferencia(y, q, fold, "0.90")
    for k, v in ref["por_fold"].items():
        assert tr["puntual"]["por_fold"][k]["umbral"] == repr(float(v["umbral"])) and tr["puntual"]["por_fold"][k]["tp"] == v["tp"]
    assert tr["puntual"]["sens"] == pytest.approx(float(ref["sensibilidad_agrupada"])) and tr["puntual"]["tp"] == ref["tp"]
    d = tr["dispersion_t_hat_entre_folds"]
    assert d["min"] <= d["media"] <= d["max"] and d["rango"] == pytest.approx(d["max"] - d["min"])
    enc, filas = leer_csv(banco.dir / "transferencia.csv")
    assert "estimación de transferencia en desarrollo, no validación independiente" in enc and [r["fila"] for r in filas][-1] == "agrupada" and len(filas) == 6
    assert filas[-1]["sens_ic95_inf"] and filas[-1]["esp_ic95_sup"]
    assert json.loads((banco.dir / "transferencia.json").read_text(encoding="utf-8"))["rotulo"] == U.TRANSFERENCIA


def test_t_hat_recalculado_en_cada_remuestreo_coincide_con_replicar_filas():
    y, q = datos_38(n=240, seed=2)
    gpi = np.arange(y.size)
    rng = np.random.default_rng(5)
    conteos = rng.integers(0, 4, size=(12, y.size)).astype(np.uint16)
    d = U.umbral_ponderado(y, q, conteos, gpi, "0.90")
    for b in range(conteos.shape[0]):
        idx = np.repeat(np.arange(y.size), conteos[b])
        ref = CAL.calibrar_umbral(y[idx].tolist(), q[idx].tolist(), "0.90")
        assert d["t"][b] == ref["umbral"] and d["k"][b] == ref["k"] and d["n_pos"][b] == ref["n_pos"] and d["tp"][b] == ref["tp"] and d["fp"][b] == ref["fp"]
        assert d["esp"][b] == pytest.approx(float(ref["especificidad"])) and d["sens"][b] == pytest.approx(float(ref["sensibilidad_real"]))
    sin = np.ones((1, y.size), dtype=np.uint16)
    sin[0, y == 1] = 0
    assert np.isnan(U.umbral_ponderado(y, q, sin, gpi)["t"][0])                            # sin positivas no hay umbral


def test_bootstrap_del_umbral_primario_tiene_percentiles_y_rotulo(banco):
    u = banco.res["resultados"]["umbral_primario"]
    assert u["rotulo"] == U.NO_CONGELADO and u["bootstrap_t_hat"]["t_hat"]["p2_5"] <= u["bootstrap_t_hat"]["t_hat"]["p50"] <= u["bootstrap_t_hat"]["t_hat"]["p97_5"]
    enc, filas = leer_csv(banco.dir / "umbral_primario.csv")
    assert filas[0]["rotulo"] == U.NO_CONGELADO and filas[0]["s_star"] == "9/10" and filas[0]["k"] == str(u["calibracion"]["k"]) and filas[0]["t_hat"] == u["calibracion"]["umbral_repr"]
    assert {"n_pos_empatadas", "n_neg_empatadas", "boot_t_hat_p2_5", "boot_t_hat_p97_5", "boot_especificidad_p50", "boot_fp_p50"} <= set(filas[0])


def test_m_agregaciones_describe_sin_criterio_de_decision(banco):
    enc, filas = leer_csv(banco.dir / "m_agregaciones.csv")
    assert len(filas) == 7 and [r["m"] for r in filas] == ["1", "1", "1", "2", "2", "2", "3"] and "sin criterio de decisión numérico" in enc
    assert {"t_hat_s090", "esp_en_t_hat", "fp_en_t_hat", "sens_transferencia_agrupada", "iqr_positivas", "iqr_negativas"} <= set(filas[0])
    assert not any(("relevante" in c or "0.05" in c or "discrepancia" in c) for c in filas[0])        # ningún criterio numérico de «discrepancia relevante»
    r3 = banco.res["resultados"]["m_agregaciones"][-1]
    assert float(filas[-1]["t_hat_s090"]) == r3["t_hat"] and r3["iqr_scores"]["positivas"]["iqr"] == pytest.approx(r3["iqr_scores"]["positivas"]["q3"] - r3["iqr_scores"]["positivas"]["q1"])


# ============================================================================ mismos b de G2
def test_mismos_b_de_g2_se_verifican_y_si_difieren_bloquean(tmp_path):
    e = armar(tmp_path)
    cfg_b = e.config_g2["bootstrap"]
    y = np.array([r["label_v2"] for r in e.particion.filas])
    gids = [r["group_id"] for r in e.particion.filas]
    _, clase, nombres = BS.indexar_grupos(gids, y)
    d = tmp_path / e.config["referencia_g2"]["indices_b_dir"]
    d.mkdir(parents=True)
    for et, s in (("estratificado", cfg_b["semilla"]), ("estratificado_semilla_comprobacion", cfg_b["semilla_de_comprobacion"])):
        BS.guardar_indices(d / f"indices_b_{et}_s{s}.npz", BS.generar_conteos(clase, cfg_b["n_remuestreos"], s, True), semilla=s, estratificado=True, nombres_de_grupo=nombres)
    _, cont, ver, n_b = U.indices_b(tmp_path, e.config, e.config_g2, gids, y)
    assert n_b == 10000 and all(x["estado"].startswith("verificado") for x in ver) and np.array_equal(cont, BS.generar_conteos(clase, 10000, cfg_b["semilla"], True))
    (d / f"indices_b_estratificado_s{cfg_b['semilla']}.npz").unlink()
    c = BS.generar_conteos(clase, 10000, cfg_b["semilla"], True).copy()
    c[0, 0] += 1
    BS.guardar_indices(d / f"indices_b_estratificado_s{cfg_b['semilla']}.npz", c, semilla=cfg_b["semilla"], estratificado=True, nombres_de_grupo=nombres)
    with pytest.raises(GuardError, match="NO coinciden"):
        U.indices_b(tmp_path, e.config, e.config_g2, gids, y)
    with pytest.raises(GuardError, match="NO coinciden"):                                  # y la ejecución completa se bloquea sin crear nada
        correr(e, n_remuestreos=None)
    assert not (tmp_path / U.SALIDA_UMBRAL_REL).exists()


# ============================================================================ leyendas, discrepancia A vs B, afirmaciones
def test_leyendas_discrepancia_listas_y_no_congelado(banco):
    r = json.loads((banco.dir / "resumen_umbral.json").read_text(encoding="utf-8"))
    d = r["discrepancia_A_vs_B"]
    assert "12 de los 15" in d["B"] and "no existe OOF de B" in d["B"] and "TRES modelos del mismo fold" in d["A"] and "no es cuantificable" in d["consecuencia"] and d["criterio_numerico_de_discrepancia"] is None
    assert len(r["afirmaciones_posibles"]) == 3 and len(r["afirmaciones_no_permitidas"]) == 6 and r["umbral_congelado"] is False and r["viabilidad_clinica"] == "no determinada"
    txt = " ".join(r["afirmaciones_no_permitidas"])
    for frase in ("garantice sensibilidad", "esté validado", "extrapolar numéricamente de m <= 3 a 15", "utilidad clínica", "mover el umbral", "independencia absoluta del test histórico"):
        assert frase in txt
    assert U.NO_CONGELADO in r["leyendas"] and U.TRANSFERENCIA in r["leyendas"] and any("objetivo exploratorio" in l and "no una garantía" in l for l in r["leyendas"])
    assert any("38 positivas" in l for l in r["limites"]) and any("pHash" in l for l in r["limites"]) and any("test" in l for l in r["limites"])
    assert r["reporte_aceptado"] is True and "PROVISIONAL" in " ".join(r["leyendas"]) and "no validado" in " ".join(r["leyendas"])
    for p in banco.dir.glob("*.csv"):
        enc = p.read_text(encoding="utf-8").splitlines()[0]
        assert enc.startswith("#") and U.LEYENDA in enc and U.NO_CONGELADO in enc and "sin uso de test" in enc


def test_sin_frases_de_utilidad_clinica_ni_validacion_afirmativa(banco):
    prohibidas = ("clínicamente útil", "apto para", "sistema validado", "sensibilidad garantizada", "garantiza la sensibilidad", "mejora el AP", "mejora la ROC-AUC", "umbral final", "umbral congelado:")
    for p in banco.dir.iterdir():
        t = p.read_text(encoding="utf-8")
        for s_ in U.AFIRMACIONES_NO_PERMITIDAS:                                            # la lista de lo NO permitido se declara explícitamente y se excluye del barrido
            t = t.replace(s_, "")
        assert not [x for x in prohibidas if x in t], (p.name, [x for x in prohibidas if x in t])
        for m in re.finditer("validad[oa]", t):
            assert "no " in t[max(0, m.start() - 40):m.start()].lower() or "NO " in t[max(0, m.start() - 40):m.start()], (p.name, t[max(0, m.start() - 40):m.start() + 20])


# ============================================================================ entradas: solo A|densenet121 y 15 corridas
def test_trazabilidad_sha256_de_las_15_corridas_y_un_registro_por_imagen(banco):
    t = json.loads((banco.dir / "trazabilidad.json").read_text(encoding="utf-8"))
    assert len(t["corridas"]) == 15 and all(k.startswith("densenet121__s") for k in t["corridas"]) and t["sha256_bytes_archivo"]["q_barra_oof.csv"] == C4.sha256_archivo(banco.dir / "q_barra_oof.csv")
    c = t["corridas"]["densenet121__s43__f2"]
    d = banco.raiz_a / "densenet121__s43__f2" / c["intento"]
    assert c["sha256_bytes_archivo_predicciones_oof"] == C4.sha256_archivo(d / "predicciones_oof.csv") and c["sha256_bytes_archivo_run_config"] == C4.sha256_archivo(d / "run_config.json")
    assert c["sha256_bytes_archivo_checkpoint_registrado"] == json.loads((d / "run_config.json").read_text(encoding="utf-8"))["checkpoint"]["sha256_bytes_archivo"]
    assert t["git_commit_head"] == "a" * 40 and t["sha256_bytes_archivo_particion"] == banco.particion.sha256_bytes_archivo and t["numpy_version"] == np.__version__ and t["un_registro_por_imagen_y_semilla"]


def test_otras_arquitecturas_no_entran_y_falta_o_duplicado_bloquea(tmp_path):
    e = armar(tmp_path)
    basura = e.raiz_a / "resnet50__s42__f0" / "intento_01"
    basura.mkdir(parents=True)
    (basura / "run_config.json").write_text("{no es json", encoding="utf-8")               # una corrida de otra arquitectura NO se lee
    assert correr(e)["reporte_aceptado"] is True
    d = e.raiz_a / "densenet121__s44__f3"
    d.rename(d.with_name("oculta"))
    with pytest.raises(GuardError, match="K7: faltan 1 corridas"):
        correr(e, destino=tmp_path / "results/r7_v2/otro1")
    d.with_name("oculta").rename(d)
    shutil.copytree(d / "intento_01", d / "intento_02")                                    # duplicado: dos intentos completados
    with pytest.raises(GuardError, match="más de un intento completado"):
        correr(e, destino=tmp_path / "results/r7_v2/otro2")
    assert not (tmp_path / "results/r7_v2/otro1").exists() and not (tmp_path / "results/r7_v2/otro2").exists()


def test_prediccion_alterada_respecto_a_su_run_config_bloquea(tmp_path):
    e = armar(tmp_path)
    p = e.raiz_a / "densenet121__s42__f1" / "intento_01" / "predicciones_oof.csv"
    p.write_bytes(p.read_bytes() + b" ")
    with pytest.raises(GuardError, match="no coincide con el SHA-256 registrado"):
        correr(e)


# ============================================================================ guardas
def test_guarda_r6_ausente_o_otra_elegida_bloquea_y_el_fixture_permite(tmp_path):
    e = armar(tmp_path)
    ruta = tmp_path / e.config["referencia_r6"]["salida"]
    ruta.unlink()
    with pytest.raises(GuardError, match="no existe"):
        correr(e)
    for elegida in ({"condicion": "B_hibrida", "arquitectura": "densenet121"}, {"condicion": "A", "arquitectura": "resnet50"}):
        escribir_r6(tmp_path, e.config, elegida)
        with pytest.raises(GuardError, match="R6 eligió"):
            correr(e)
    ruta.write_text("{", encoding="utf-8")
    with pytest.raises(GuardError, match="ilegible"):
        correr(e)
    assert not (tmp_path / U.SALIDA_UMBRAL_REL).exists()
    escribir_r6(tmp_path, e.config)
    ruta.unlink()
    escribir_r6(tmp_path, e.config)
    assert correr(e)["reporte_aceptado"] is True


@pytest.mark.parametrize("valor", ["", None, "CONFIRMADO", "CONFIRMADO:D-48", "confirmado:D-048", "CONFIRMADO:AUT-048", " CONFIRMADO:D-048", "PENDIENTE", 48])
def test_autorizacion_exigida_formato_y_no_se_lee_ni_se_escribe(tmp_path, valor):
    e = armar(tmp_path)
    with pytest.raises(GuardError, match="autorizacion"):
        correr(e, autorizacion=valor)
    assert not (tmp_path / U.SALIDA_UMBRAL_REL).exists()
    vacio = tmp_path / "vacio"
    vacio.mkdir()
    assert U.main([], root=vacio) == U.CODIGO_SALIDA_GUARDA == 3 and list(vacio.iterdir()) == []      # sin autorización ni siquiera se busca el YAML


def test_k4_rutas_con_test_y_fuera_de_results_r7_se_rechazan_y_no_sobrescribe(tmp_path):
    e = armar(tmp_path)
    for malo in (tmp_path / "results/r7_v2/test/x", tmp_path / "results/r4_v2/x", tmp_path / "data/x"):
        with pytest.raises(GuardError):
            U.destino_nuevo(tmp_path, U.SALIDA_UMBRAL_REL, malo)
    cfg = copy.deepcopy(e.config)
    cfg["salidas"]["umbral"] = "results/r7_v2/phash_le8_v1/test/umbral"
    with pytest.raises(GuardError):
        U.validar_config(cfg)
    r = correr(e)
    assert r["dir"] == (tmp_path / U.SALIDA_UMBRAL_REL).resolve() or r["dir"] == tmp_path / U.SALIDA_UMBRAL_REL
    antes = {p.name: C4.sha256_archivo(p) for p in r["dir"].iterdir()}
    with pytest.raises(GuardError, match="ya existe"):
        correr(e)
    assert {p.name: C4.sha256_archivo(p) for p in r["dir"].iterdir()} == antes             # nada se sobrescribió


def test_determinismo_dos_ejecuciones_producen_los_mismos_csv_y_json_salvo_fecha(tmp_path):
    e = armar(tmp_path)
    a, b = correr(e, destino=tmp_path / "results/r7_v2/det_a")["dir"], correr(e, destino=tmp_path / "results/r7_v2/det_b")["dir"]
    assert sorted(p.name for p in a.iterdir()) == sorted(p.name for p in b.iterdir())
    for p in a.iterdir():
        if p.suffix == ".csv":
            assert C4.sha256_archivo(p) == C4.sha256_archivo(b / p.name), p.name
        else:
            ja, jb = json.loads(p.read_text(encoding="utf-8")), json.loads((b / p.name).read_text(encoding="utf-8"))
            ja.pop("fecha", None), jb.pop("fecha", None)
            assert ja == jb, p.name


def test_procedimiento_y_protocolo_anti_deriva(tmp_path):
    config = U.cargar_config(REPO / U.CONFIG_REL)
    gc, sh = config["protocolo"]["git_commit"], config["protocolo"]["sha256_contenido_blob"]
    assert config["protocolo"]["trazabilidad"] == "D-048"
    assert (gc is None and sh is None) or (re.fullmatch("[0-9a-f]{40}", gc) and re.fullmatch("[0-9a-f]{64}", sh))      # antes del commit 2: null; después: commit y hash registrados
    texto = (REPO / U.CONFIG_REL).read_text(encoding="utf-8")
    if gc is None:
        lineas = texto.splitlines()
        assert lineas.count("  git_commit: null") == 1 and lineas.count("  sha256_contenido_blob: null") == 1          # una línea cada una, sin comentario, para el `sed`
    cambios = [lambda c: c["procedimiento"].update(s_star_primario="0.95"), lambda c: c["procedimiento"].update(s_star_descriptivos=["0.85"]),
               lambda c: c["procedimiento"].update(umbral_congelado=True), lambda c: c["procedimiento"].update(criterio_numerico_de_discrepancia=0.05),
               lambda c: c["procedimiento"].update(categoria_6_reevaluada_como_positiva=True), lambda c: c["procedimiento"].update(r7_3_declarado_cumplido=True),
               lambda c: c["procedimiento"]["bootstrap"].update(n_remuestreos=1000), lambda c: c["procedimiento"]["sistema_provisional"].update(arquitectura="resnet50"),
               lambda c: c["procedimiento"].update(regla_positiva="p > umbral"), lambda c: c["acceso_test"].update(estado="abierto"),
               lambda c: c["referencia_r6"]["elegida"].update(arquitectura="resnet50"), lambda c: c["integridad"]["archivos_con_huella"].remove(U.PROTOCOLO_REL),
               lambda c: c["salidas"].update(umbral="results/r7_v2/x")]
    for cambio in cambios:
        mala = copy.deepcopy(config)
        cambio(mala)
        with pytest.raises(GuardError):
            U.validar_config(mala)
    protocolo = (REPO / U.PROTOCOLO_REL).read_text(encoding="utf-8")
    assert len(protocolo.splitlines()) <= 70 and U.exigir_protocolo_coherente(REPO)
    for i, frase in enumerate(U.FRASES_PROTOCOLO):
        raiz = tmp_path / f"p{i}"
        (raiz / U.PROTOCOLO_REL).parent.mkdir(parents=True)
        (raiz / U.PROTOCOLO_REL).write_text(protocolo.replace(frase, "X"), encoding="utf-8")
        with pytest.raises(GuardError, match="no es coherente"):
            U.exigir_protocolo_coherente(raiz)
    (tmp_path / "h" / U.PROTOCOLO_REL).parent.mkdir(parents=True)
    (tmp_path / "h" / U.PROTOCOLO_REL).write_text(protocolo + "\n" + "b" * 64, encoding="utf-8")
    with pytest.raises(GuardError, match="no es coherente"):
        U.exigir_protocolo_coherente(tmp_path / "h")
    with pytest.raises(GuardError, match="no existe"):
        U.exigir_protocolo_coherente(tmp_path / "nada")


def _git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), "-c", "user.email=t@t", "-c", "user.name=t", *args], capture_output=True, text=True, check=True).stdout.strip()


def test_integridad_con_git_real_protocolo_congelado_y_arbol_limpio(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "src/r7_v2").mkdir(parents=True)
    archivos = {"src/r7_v2/a.py": b"x = 1\n", U.PROTOCOLO_REL: b"protocolo\n", "g2.yaml": b"g2: 1\n"}
    for rel, b in archivos.items():
        (tmp_path / rel).write_bytes(b)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "c1")
    c1 = _git(tmp_path, "rev-parse", "HEAD")
    blob = hashlib.sha256(archivos[U.PROTOCOLO_REL]).hexdigest()
    cfg = {"integridad": {"rutas_sin_cambios": ["src/r7_v2"], "archivos_con_huella": list(archivos)},
           "referencia_g2": {"config": "g2.yaml", "sha256_contenido_blob": hashlib.sha256(b"g2: 1\n").hexdigest()},
           "protocolo": {"git_commit": None, "sha256_contenido_blob": None}}
    assert [b["codigo"] for b in U.verificar_integridad(tmp_path, cfg)["bloqueos"]] == ["protocolo_no_congelado"]        # YAML sin commit/hash: no se ejecuta
    with pytest.raises(GuardError, match="protocolo_no_congelado"):
        U.exigir_integridad(U.verificar_integridad(tmp_path, cfg))
    cfg["protocolo"] = {"git_commit": c1, "sha256_contenido_blob": blob}
    ok = U.verificar_integridad(tmp_path, cfg)
    assert not ok["bloqueos"] and ok["git_commit_head"] == _git(tmp_path, "rev-parse", "HEAD") and set(ok["archivos"]["g2.yaml"]) >= {"sha256_contenido_blob", "sha256_bytes_archivo"}
    cfg["protocolo"]["sha256_contenido_blob"] = "0" * 64
    assert {b["codigo"] for b in U.verificar_integridad(tmp_path, cfg)["bloqueos"]} == {"protocolo_distinto_del_registrado", "protocolo_del_commit_registrado_distinto"}
    cfg["protocolo"] = {"git_commit": c1, "sha256_contenido_blob": blob}
    (tmp_path / "src/r7_v2/a.py").write_bytes(b"x = 2\n")
    sucio = U.verificar_integridad(tmp_path, cfg)
    assert [b["codigo"] for b in sucio["bloqueos"]] == ["arbol_sucio"]
    _git(tmp_path, "checkout", "--", "src/r7_v2/a.py")
    cfg["referencia_g2"]["sha256_contenido_blob"] = "0" * 64
    assert [b["codigo"] for b in U.verificar_integridad(tmp_path, cfg)["bloqueos"]] == ["yaml_referencia_g2_distinto_del_fijado"]
    cfg["referencia_g2"]["sha256_contenido_blob"] = hashlib.sha256(b"g2: 1\n").hexdigest()
    cfg["protocolo"]["git_commit"] = "f" * 40                                                  # un commit que no existe
    assert "protocolo_del_commit_registrado_distinto" in [b["codigo"] for b in U.verificar_integridad(tmp_path, cfg)["bloqueos"]]


def test_la_salida_no_se_acepta_si_la_integridad_cambia_al_final(tmp_path):
    e = armar(tmp_path)
    n = {"i": 0}

    def cambia():
        n["i"] += 1
        r = verificar_u()()
        if n["i"] > 1:
            r["git_commit_head"] = "z" * 40
        return r
    r = correr(e, verificar=cambia)
    assert r["reporte_aceptado"] is False and "git_commit_head" in r["diferencias_inicio_fin"]
    assert json.loads((r["dir"] / "resumen_umbral.json").read_text(encoding="utf-8"))["invalida_integridad"] is True
    n["i"] = 0
    assert U.main(["--autorizacion", AUT], root=tmp_path, config=e.config, config_g2=e.config_g2, verificar=cambia, n_remuestreos=N_B,
                  destino=tmp_path / "results/r7_v2/cli4") == U.CODIGO_SALIDA_INTEGRIDAD_INVALIDA == 4
