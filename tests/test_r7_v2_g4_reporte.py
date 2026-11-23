"""Pruebas sintéticas (CPU) del reporte de G4: segundo eje (punto de operación con umbral recalculado), veto, niveles de evidencia, reemplazo de la base,
atribución del agente, evidencia de la fase B, guarda de R6, presupuesto (30 + 20 + 1 humo -> 24 -> máx. 2 propuestas), K4, no sobrescritura.
No abren test, ni datos reales ni entrenan modelos reales. Comando: py -3.13 -B -m pytest tests/test_r7_v2_g4_reporte.py -q -p no:cacheprovider
"""

from __future__ import annotations

import copy
import csv
import json
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("yaml")
pytest.importorskip("PIL")

from src.r4_v2 import bootstrap as BS  # noqa: E402
from src.r4_v2 import calibracion as CAL  # noqa: E402
from src.r4_v2 import common as C4  # noqa: E402
from src.r4_v2.common import GuardError  # noqa: E402
from src.r7_v2 import g4_agente as AGT  # noqa: E402
from src.r7_v2 import g4_comun as C  # noqa: E402
from src.r7_v2 import g4_presupuesto as P  # noqa: E402
from src.r7_v2 import g4_reporte as R  # noqa: E402
from tests.test_r7_v2_g4 import (AUT, REF, REPO, _evidencia, _propuesta, _relleno, armar, correr, escribir_r6, verificar_falso)  # noqa: E402,F401
from tests.test_r7_v2_g4 import _git_aislado  # noqa: E402,F401  (fixture autouse)

N_B = 300


def mk(inf, sup, n=100):
    return {"ic_inf": inf, "ic_sup": sup, "n_remuestreos_validos": n}


# ============================================================================ clasificación del punto de operación y veto (puras)
def test_veto_con_ic_completamente_negativo_en_sensibilidad_especificidad_o_ambas():
    r = R.clasificar_punto_operacion(mk(-0.2, -0.01), mk(-0.1, 0.1), 3)
    assert r["clasificacion"] == "deterioro respaldado" and r["deterioro_en"] == ["sensibilidad"] and r["veto_aplicado"] and r["veto_formal"] and r["rotulo"] is None
    r = R.clasificar_punto_operacion(mk(-0.1, 0.1), mk(-0.3, -0.05), 3)
    assert r["clasificacion"] == "deterioro respaldado" and r["deterioro_en"] == ["especificidad"] and r["veto_aplicado"]
    r = R.clasificar_punto_operacion(mk(-0.3, -0.01), mk(-0.3, -0.05), 3)
    assert r["deterioro_en"] == ["sensibilidad", "especificidad"] and r["clasificacion"] == "deterioro respaldado"


def test_el_veto_prevalece_sobre_la_mejora_en_la_otra_metrica():
    r = R.clasificar_punto_operacion(mk(-0.2, -0.01), mk(0.2, 0.5), 3)
    assert r["clasificacion"] == "deterioro respaldado" and r["mejora_en"] == ["especificidad"] and r["veto_prevalece_sobre_mejora"] and r["veto_aplicado"]
    r = R.clasificar_punto_operacion(mk(0.05, 0.3), mk(-0.5, -0.1), 3)
    assert r["clasificacion"] == "deterioro respaldado" and r["veto_prevalece_sobre_mejora"]


def test_mejora_respaldada_no_distinguible_y_limites_estrictos():
    assert R.clasificar_punto_operacion(mk(0.01, 0.2), mk(-0.1, 0.1), 3)["clasificacion"] == "mejora respaldada"
    assert R.clasificar_punto_operacion(mk(-0.1, 0.1), mk(0.02, 0.4), 3)["clasificacion"] == "mejora respaldada"
    assert R.clasificar_punto_operacion(mk(-0.1, 0.1), mk(-0.2, 0.3), 3)["clasificacion"] == "no distinguible"
    assert R.clasificar_punto_operacion(mk(-0.2, 0.0), mk(0.0, 0.2), 3)["clasificacion"] == "no distinguible"     # límite superior = 0 no es deterioro; inferior = 0 no es mejora
    assert R.clasificar_punto_operacion(mk(float("nan"), float("nan"), 0), mk(0.1, 0.2), 3)["clasificacion"] == "mejora respaldada"
    assert R.clasificar_punto_operacion(mk(float("nan"), float("nan"), 0), mk(float("nan"), float("nan"), 0), 3)["clasificacion"] == "no distinguible"


def test_una_semilla_se_clasifica_pero_sin_veto_formal():
    r = R.clasificar_punto_operacion(mk(-0.2, -0.01), mk(-0.1, 0.1), 1)
    assert r["clasificacion"] == "deterioro respaldado" and r["veto_formal"] is False and r["veto_aplicado"] is False and r["rotulo"] == "exploratoria; sin veto formal"
    assert R.clasificar_punto_operacion(mk(-0.2, -0.01), mk(-0.1, 0.1), 2)["veto_aplicado"] is False


# ============================================================================ niveles de evidencia, reemplazo y agente (puras)
def test_tres_niveles_de_evidencia_calculados_por_codigo():
    ok = R.nivel_evidencia({42: 0.02, 43: 0.01, 44: 0.03}, mk(0.003, 0.05), mk(0.001, 0.06), [42, 43, 44])
    assert ok["nivel"] == R.NIVELES[1] and ok["cumple_a"] and ok["cumple_b"] and ok["candidata_prometedora"]
    a_falla = R.nivel_evidencia({42: 0.05, 43: -0.01, 44: 0.04}, mk(0.003, 0.05), mk(0.001, 0.06), [42, 43, 44])      # media > 0 pero una semilla < 0
    assert a_falla["nivel"] == R.NIVELES[0] and not a_falla["cumple_a"]
    b_falla = R.nivel_evidencia({42: 0.02, 43: 0.01, 44: 0.03}, mk(0.003, 0.05), mk(-0.001, 0.06), [42, 43, 44])     # no estratificado incluye 0
    assert b_falla["nivel"] == R.NIVELES[0] and b_falla["cumple_a"] and not b_falla["cumple_b"]
    assert R.nivel_evidencia({42: 0.02, 43: 0.01, 44: 0.03}, mk(-0.001, 0.05), mk(0.001, 0.06), [42, 43, 44])["nivel"] == R.NIVELES[0]
    una = R.nivel_evidencia({42: 0.02}, mk(0.01, 0.05), mk(0.01, 0.06), [42])
    assert una["nivel"] == R.NIVELES[0] and una["candidata_prometedora"] and not una["nivel_ii_evaluable"] and not una["cumple_a"]
    assert R.nivel_evidencia({42: -0.02}, mk(0.01, 0.05), mk(0.01, 0.06), [42])["nivel"] == R.NIVELES[2]
    assert R.nivel_evidencia({42: 0.0}, mk(0.01, 0.05), mk(0.01, 0.06), [42])["nivel"] == R.NIVELES[2]                # delta = 0 no es candidata
    assert R.nivel_evidencia({42: -0.01, 43: 0.05, 44: 0.05}, mk(0.01, 0.05), mk(0.01, 0.06), [42, 43, 44])["nivel"] == R.NIVELES[2]


def test_reemplazo_de_la_base_solo_con_i_y_ii_y_etiqueta_de_deterioro():
    mejor, cand = {"nivel": R.NIVELES[1]}, {"nivel": R.NIVELES[0]}
    ok, det = {"clasificacion": "no distinguible"}, {"clasificacion": "deterioro respaldado"}
    assert R.decidir_reemplazo(mejor, ok)["decision"] == "puede reemplazar a la base"
    assert R.decidir_reemplazo(mejor, {"clasificacion": "mejora respaldada"})["decision"] == "puede reemplazar a la base"
    d = R.decidir_reemplazo(mejor, det)
    assert d["decision"] == "conserva la base" and d["nota"] == "mejora de discriminación con deterioro del punto de operación" and d["cumple_i"] and not d["cumple_ii"]
    assert R.decidir_reemplazo(cand, ok)["decision"] == "conserva la base" and R.decidir_reemplazo(cand, ok)["nota"] is None
    assert R.decidir_reemplazo({"nivel": R.NIVELES[2]}, det)["decision"] == "conserva la base"


def test_atribucion_al_agente_solo_si_supera_a_la_candidata_convencional():
    mejor = {"nivel": R.NIVELES[1], "nivel_ii_evaluable": True}
    cand = {"nivel": R.NIVELES[0], "nivel_ii_evaluable": True}
    ok, det = {"clasificacion": "no distinguible"}, {"clasificacion": "deterioro respaldado"}
    assert R.atribucion_agente(mejor, ok, mejor) == {"atribuible_al_agente": True, "nota": None}
    r = R.atribucion_agente(cand, ok, mejor)                                                # supera la base pero no a la convencional
    assert r["atribuible_al_agente"] is False and "superar solo la base NO basta" in r["nota"]
    assert R.atribucion_agente(mejor, det, mejor)["atribuible_al_agente"] is False         # deterioro respaldado frente a la convencional
    una = R.atribucion_agente({"nivel": R.NIVELES[0], "nivel_ii_evaluable": False}, ok, {"nivel": R.NIVELES[0]})
    assert una["atribuible_al_agente"] is False and "semillas insuficientes" in una["nota"]


# ============================================================================ transferencia con umbral recalculado (vectorizada) vs referencia
def _datos(n=200, seed=0, redondeo=2):
    rng = np.random.default_rng(seed)
    y = (np.arange(n) % 6 == 0).astype(np.int64)
    q = np.round(np.clip(0.35 + 0.35 * y + rng.normal(0, 0.25, n), 0, 1), redondeo)       # con empates a propósito
    return y, q, (np.arange(n) // 6) % 5


def test_vectorizada_con_pesos_unitarios_igual_a_comprobar_transferencia():
    y, q, folds = _datos()
    pun = R.punto_operacion_puntual(y, q, folds)                                          # internamente compara contra calibracion.comprobar_transferencia
    ref = CAL.comprobar_transferencia(y.tolist(), q.tolist(), folds.tolist(), "0.90")
    assert pun["tp"] + pun["fn"] == int(y.sum()) == pun["n_malignas"] and pun["tn"] + pun["fp"] == int((1 - y).sum()) == pun["n_no_malignas"]
    assert pun["sens"] == pytest.approx(float(ref["sensibilidad_agrupada"])) and pun["s_star"] == "9/10"
    assert pun["vpp"] == pytest.approx(pun["tp"] / (pun["tp"] + pun["fp"])) and pun["prevalencia"] == pytest.approx(y.mean())
    assert set(pun["por_fold"]) == set(range(5)) and all(isinstance(v["umbral"], str) for v in pun["por_fold"].values())


def test_vectorizada_con_multiplicidades_igual_a_replicar_filas_y_recalcular_el_umbral():
    y, q, folds = _datos()
    n = y.size
    rng = np.random.default_rng(3)
    gpi = np.arange(n)
    conteos = rng.integers(0, 4, size=(15, n)).astype(np.uint16)                          # 15 remuestreos con multiplicidades 0..3
    v = R.transferencia_ponderada(y, q, folds, conteos, gpi)
    for b in range(conteos.shape[0]):
        idx = np.repeat(np.arange(n), conteos[b])
        ref = CAL.comprobar_transferencia(y[idx].tolist(), q[idx].tolist(), folds[idx].tolist(), "0.90")      # umbral RECALCULADO con las filas remuestreadas
        tp = sum(f["tp"] for f in ref["por_fold"].values())
        fn = sum(f["fn"] for f in ref["por_fold"].values())
        fp = sum(f["fp"] for f in ref["por_fold"].values())
        tn = sum(f["tn"] for f in ref["por_fold"].values())
        assert (v["tp"][b], v["fn"][b], v["fp"][b], v["tn"][b]) == (tp, fn, fp, tn)
        assert v["sens"][b] == pytest.approx(tp / (tp + fn)) and v["esp"][b] == pytest.approx(tn / (tn + fp))


def test_vectorizada_remuestreo_sin_positivas_en_el_entrenamiento_es_nan():
    y, q, folds = _datos()
    conteos = np.ones((2, y.size), dtype=np.uint16)
    conteos[1, y == 1] = 0                                                                # el segundo remuestreo no tiene positivas
    v = R.transferencia_ponderada(y, q, folds, conteos, np.arange(y.size))
    assert np.isfinite(v["sens"][0]) and np.isnan(v["sens"][1]) and np.isnan(v["tp"][1])
    assert R.ic(v["sens"])["n_remuestreos_validos"] == 1


# ============================================================================ eje 2 y veto con datos sintéticos (analizar completo)
ARQ = "densenet121"


def _escenario(pos_f, neg_f, semillas, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    y = (i % 5 == 0).astype(np.int64)
    fold = (i // 5) % 5
    probs = {}
    for s in semillas:
        r = np.random.default_rng(seed + s)
        probs[s] = np.where(y == 1, pos_f(r, n), neg_f(r, n))
    return types.SimpleNamespace(image_ids=[str(k) for k in i], y=y, fold=fold, group_ids=[f"g{k}" for k in i], prob={(ARQ, s): p for s, p in probs.items()})


def _analizar(base, conf, semillas, K=2, B=300):
    oofs = {R.BASE: (base, semillas), "S1_a050": (conf, semillas)}
    gpi, clase, _ = BS.indexar_grupos(base.group_ids, base.y)
    ce = BS.generar_conteos(clase, B, 20261121, estratificado=True)
    cn = BS.generar_conteos(clase, B, 20261121, estratificado=False)
    return R.analizar(oofs, [("S1_a050", R.BASE, semillas)], ARQ, K, B, ce, cn, gpi, {})


PERFECTA = (lambda r, n: np.ones(n), lambda r, n: np.zeros(n))


def test_deterioro_en_sensibilidad_solo_con_datos_y_umbral_recalculado():
    base = _escenario(*PERFECTA, [42, 43, 44])                                            # sens = 1 y esp = 1 en cada remuestreo
    conf = _escenario(lambda r, n: r.uniform(0.5, 1.0, n), lambda r, n: np.zeros(n), [42, 43, 44])    # esp = 1, sens ~ 0.9 por la calibración
    c = _analizar(base, conf, [42, 43, 44])["comparaciones"][0]
    po = c["punto_operacion_pareado"]
    assert po["sens"]["ic_sup"] < 0 and po["esp"]["ic_inf"] == po["esp"]["ic_sup"] == 0.0
    assert po["clasificacion"]["clasificacion"] == "deterioro respaldado" and po["clasificacion"]["deterioro_en"] == ["sensibilidad"] and po["clasificacion"]["veto_aplicado"]
    assert c["clasificacion_punto_operacion"] == po["clasificacion"]


def test_deterioro_en_especificidad_solo_y_en_ambas_con_datos():
    base = _escenario(*PERFECTA, [42, 43, 44])
    conf_esp = _escenario(lambda r, n: np.full(n, 0.5), lambda r, n: np.where(r.random(n) < 0.5, 0.9, 0.0), [42, 43, 44])     # sens = 1; la mitad de las negativas supera a las positivas
    po = _analizar(base, conf_esp, [42, 43, 44])["comparaciones"][0]["punto_operacion_pareado"]
    assert po["esp"]["ic_sup"] < 0 and po["sens"]["ic_sup"] == po["sens"]["ic_inf"] == 0.0
    assert po["clasificacion"]["deterioro_en"] == ["especificidad"] and po["clasificacion"]["veto_aplicado"]
    conf_ambas = _escenario(lambda r, n: r.uniform(0.5, 1.0, n), lambda r, n: np.where(r.random(n) < 0.5, 0.9, 0.0), [42, 43, 44])
    po = _analizar(base, conf_ambas, [42, 43, 44])["comparaciones"][0]["punto_operacion_pareado"]
    assert po["clasificacion"]["deterioro_en"] == ["sensibilidad", "especificidad"] and po["clasificacion"]["clasificacion"] == "deterioro respaldado"


def test_mejora_respaldada_y_una_semilla_exploratoria_sin_veto_formal_con_datos():
    malo = _escenario(lambda r, n: np.full(n, 0.5), lambda r, n: np.where(r.random(n) < 0.5, 0.9, 0.0), [42, 43, 44])
    bueno = _escenario(lambda r, n: np.full(n, 1.0), lambda r, n: np.zeros(n), [42, 43, 44])
    c = _analizar(malo, bueno, [42, 43, 44])["comparaciones"][0]
    assert c["punto_operacion_pareado"]["esp"]["ic_inf"] > 0 and c["clasificacion_punto_operacion"]["clasificacion"] == "mejora respaldada"
    # la misma situación inversa con UNA semilla: se reporta y se clasifica, pero rotulada y sin veto
    base = _escenario(*PERFECTA, [42])
    conf = _escenario(lambda r, n: np.full(n, 0.5), lambda r, n: np.where(r.random(n) < 0.5, 0.9, 0.0), [42])
    c1 = _analizar(base, conf, [42])["comparaciones"][0]
    cl = c1["punto_operacion_pareado"]["clasificacion"]
    assert cl["clasificacion"] == "deterioro respaldado" and cl["veto_formal"] is False and cl["veto_aplicado"] is False and cl["rotulo"] == "exploratoria; sin veto formal"
    assert c1["nivel_evidencia"]["nivel_ii_evaluable"] is False


def test_ap_mejor_con_deterioro_respaldado_conserva_la_base_y_el_veto_prevalece_sobre_la_mejora():
    base = _escenario(lambda r, n: np.full(n, 0.8), lambda r, n: np.where(r.random(n) < 0.5, 0.9, r.uniform(0, 0.5, n)), [42, 43, 44])   # sens = 1; AP bajo; esp ~ 0.5
    conf = _escenario(lambda r, n: r.uniform(0.5, 1.0, n), lambda r, n: r.uniform(0.0, 0.6, n) * 0.9, [42, 43, 44])     # AP alto; sens ~ 0.9; esp alta
    r = _analizar(base, conf, [42, 43, 44], K=7)
    c = r["comparaciones"][0]
    assert c["nivel_evidencia"]["nivel"] == R.NIVELES[1] and all(v > 0 for v in c["delta_ap"]["por_semilla"].values())
    po = c["punto_operacion_pareado"]
    assert po["sens"]["ic_sup"] < 0 and po["esp"]["ic_inf"] > 0 and po["clasificacion"]["veto_prevalece_sobre_mejora"]
    d = c["decision_reemplazo"]
    assert d["decision"] == "conserva la base" and d["nota"] == "mejora de discriminación con deterioro del punto de operación"
    m = c["multiplicidad"]
    assert m["k"] == 7 and m["nivel"] == pytest.approx(1 - 0.05 / 7) and m["etiqueta"] == C.ETIQUETA_MULTIPLICIDAD and m["respaldada_tras_ajuste_por_multiplicidad"] in (True, False)
    assert c["delta_ap_semillas_43_44"]["informativo"] is True and set(c["delta_ap"]["por_semilla"]) == {"42", "43", "44"}


def test_vpp_conteos_absolutos_y_prevalencia_rotulada_en_las_tablas():
    base = _escenario(*PERFECTA, [42])
    r = _analizar(base, base, [42])
    ops = r["puntos_de_operacion"]
    assert {v["config_id"] for v in ops.values()} == {R.BASE, "S1_a050"}
    p = next(iter(ops.values()))["puntual"]
    assert p["n_malignas"] == 200 and p["n_no_malignas"] == 800 and p["fp"] == 0 and p["fn"] == 0 and p["vpp"] == 1.0 and p["prevalencia"] == pytest.approx(0.2)
    res = {"ap_por_semilla": r["ap_por_semilla"], "comparaciones": r["comparaciones"], "puntos_de_operacion": ops}
    rl = {**P.resumen([]), "_presupuesto_c": 25}
    t = R.tablas(res, rl, 3, {})
    fila = t["T-G4-3_punto_operacion"][0][0]
    assert fila["fp_sobre_no_malignas"] == "0/800" and fila["fn_sobre_malignas"] == "0/200" and fila["rotulo"] == "exploratoria; sin veto formal" and fila["s_star"] == "0.90"
    assert "solo válido para esa prevalencia; no transferible a otras poblaciones" in R.NOTA_VPP and "{pos}/{n}" in R.NOTA_VPP


# ============================================================================ banco sintético: humo + fase A completa + promoción + fase B (+ reporte fase B)
def _construir(tmp_path_factory, nombre):
    e = armar(tmp_path_factory.mktemp(nombre))
    assert correr(e, "humo") == 0 and correr(e, "fase_a") == 0 and correr(e, "promocion") == 0 and correr(e, "fase_b") == 0
    return e


def _reporte(e, modo, **kw):
    args = dict(modo=modo, verificar=e.kw["verificar"], referencia=REF, n_remuestreos=N_B) | kw
    return R.generar_reporte(e.root, e.config, e.config_g2, e.particion, **args)


def _tabla(path):
    lineas = Path(path).read_text(encoding="utf-8").splitlines()
    return lineas[0], list(csv.DictReader(lineas[1:]))


@pytest.fixture(scope="module")
def banco(tmp_path_factory):
    e = _construir(tmp_path_factory, "banco_g4_rep")
    e.rep_b = _reporte(e, "fase_b")                                                         # emite evidencia_fase_b.json
    return e


def test_caso_de_presupuesto_30_mas_20_mas_humo_deja_24_y_maximo_2_propuestas(banco):
    ev = P.leer_ledger(banco.ledger)
    r = P.resumen(ev)
    assert r["planificadas_por_fase"] == {"A": 30, "B": 20, "C": 0} and r["humo_consumidos"] == 1 and r["completadas"] == 50 and r["intentos_consumidos"] == 51
    assert r["restante"] == 24 and P.presupuesto_fase_c(ev) == 24 and P.maximo_propuestas(ev) == 2
    assert P.maximo_propuestas(ev[:0]) == 3 and 5 * 2 + 10 <= 24 < 5 * 3 + 10
    _, t5 = _tabla(banco.rep_b["dir"] / "T-G4-5_presupuesto_ledger.csv")
    val = {f["concepto"]: f["valor"] for f in t5}
    assert val["intentos_consumidos"] == "51" and val["restante"] == "24" and val["humo_consumidos"] == "1" and val["completadas"] == "50"
    assert val["maximo_propuestas_del_agente (5n + 10 <= presupuesto de C; n <= 3)"] == "2" and val["presupuesto_de_C (75 - consumidos, acotado a 25)"] == "24"
    assert {"planificadas", "intentos_fallidos", "reintentos", "consumo_segundos", "gpus"} <= set(val)


def test_tres_propuestas_no_caben_con_24_restantes_y_no_se_crea_nada(banco):
    ruta = banco.raiz / banco.config["agente"]["evidencia_fase_b"]
    ev = AGT.leer_evidencia_fase_b(ruta)                                                     # el real emitido por el reporte
    assert AGT.familia_seleccionada(ev)["config_id"] == "S1_a050"
    carpeta = None
    for n, v in enumerate((0.35, 0.65, 0.15), start=1):
        carpeta = _propuesta(banco, ruta, n, valor=v)
    antes = (banco.ledger.read_text(encoding="utf-8"), sorted(p.name for p in banco.raiz.iterdir()))
    with pytest.raises(P.PresupuestoAgotado, match="exceden"):
        correr(banco, "fase_c")                                                              # 5*3 + 10 = 25 > 24
    assert (banco.ledger.read_text(encoding="utf-8"), sorted(p.name for p in banco.raiz.iterdir())) == antes and AGT.leer_registro(carpeta) == []
    (carpeta / "propuesta_03.json").unlink()
    (carpeta / "propuesta_02.json").unlink()                                                 # restablece el estado del banco
    (carpeta / "propuesta_01.json").unlink()
    carpeta.rmdir()


def test_evidencia_fase_b_compatible_con_el_agente_y_niveles_calculados(banco):
    ruta = banco.rep_b["dir"] / "evidencia_fase_b.json"
    ev = AGT.leer_evidencia_fase_b(ruta)
    assert ev["esquema"] == "evidencia_fase_b_g4/1" and [c["config_id"] for c in ev["configuraciones"]] == ["S1_a050", "S1_a025"]
    for c in ev["configuraciones"]:
        assert isinstance(c["cumple_a"], bool) and c["semillas"] == [42, 43, 44] and c["familia"] == "S1"
    assert ev["configuraciones"][0]["cumple_a"] is True and AGT.familia_seleccionada(ev)["config_id"] == "S1_a050"
    res = banco.rep_b["resultados"]["resultados"]
    cs = {c["x"]: c for c in res["comparaciones"] if c["y"] == R.BASE}
    assert set(cs) == {"S1_a050", "S1_a025"}
    for i, c in cs.items():
        assert c["nivel_evidencia"]["cumple_a"] == next(x for x in ev["configuraciones"] if x["config_id"] == i)["cumple_a"]
        assert c["decision_reemplazo"] == R.decidir_reemplazo(c["nivel_evidencia"], c["punto_operacion_pareado"]["clasificacion"])
        assert c["punto_operacion_pareado"]["clasificacion"]["veto_formal"] is True and c["delta_ap_semillas_43_44"]["informativo"] is True
        assert c["multiplicidad"]["k"] == 6 and c["multiplicidad"]["nivel"] == pytest.approx(1 - 0.05 / 6)
    meta = banco.rep_b
    assert meta["reporte_aceptado"] and not meta["invalida_integridad"] and meta["k"] == 6 and meta["modo"] == "fase_b"
    assert meta["sha256_bytes_archivo"]["resultados_g4.json"] == C4.sha256_archivo(meta["dir"] / "resultados_g4.json")
    assert meta["resultados"]["maximo_propuestas_del_agente"] == 2 and meta["resultados"]["viabilidad_clinica"] == "no determinada"


def test_delta_ap_sin_redondear_y_mismos_b_de_g2(banco):
    res = banco.rep_b["resultados"]["resultados"]
    c = next(x for x in res["comparaciones"] if x["x"] == "S1_a050" and x["y"] == R.BASE)
    est = next(e for e in C.estrategias_fase_a(banco.config) if e["id"] == "S1_a050")
    esperado = float(np.mean([C.ap_estrategia(banco.root, banco.config, banco.config_g2, banco.particion, est, s) - C.ap_base(banco.root, banco.config, banco.config_g2, banco.particion, s)
                              for s in (42, 43, 44)]))
    assert c["delta_ap"]["estimacion"] == pytest.approx(esperado, abs=1e-12) and c["delta_ap"]["estimacion"] != round(c["delta_ap"]["estimacion"], 4)
    cfg_b = banco.config_g2["bootstrap"]
    y = np.array([r["label_v2"] for r in banco.particion.filas])
    _, clase, _ = BS.indexar_grupos([r["group_id"] for r in banco.particion.filas], y)
    guardado = BS.cargar_indices(banco.rep_b["dir"] / f"indices_b_estratificado_s{cfg_b['semilla']}.npz")
    assert np.array_equal(guardado["conteos"], BS.generar_conteos(clase, N_B, cfg_b["semilla"], estratificado=True))     # los `b` de G2 (misma semilla y generador)


def test_indices_b_se_verifican_contra_los_npz_de_g2_con_b_completo(banco):
    cfg_b = banco.config_g2["bootstrap"]
    y = np.array([r["label_v2"] for r in banco.particion.filas])
    _, clase, nombres = BS.indexar_grupos([r["group_id"] for r in banco.particion.filas], y)
    d = banco.root / banco.config["referencia_g2"]["indices_b_dir"]
    d.mkdir(parents=True)
    ruta = {}
    for et, s in (("estratificado", cfg_b["semilla"]), ("estratificado_semilla_comprobacion", cfg_b["semilla_de_comprobacion"])):
        ruta[s] = d / f"indices_b_{et}_s{s}.npz"
        BS.guardar_indices(ruta[s], BS.generar_conteos(clase, cfg_b["n_remuestreos"], s, True), semilla=s, estratificado=True, nombres_de_grupo=nombres)
    try:
        meta = _reporte(banco, "fase_a", hasta=1, etiqueta="b_completo", n_remuestreos=None)
        assert meta["n_remuestreos"] == 10000 and all(x["estado"].startswith("verificado") for x in meta["indices_b_verificacion"])
        ruta[cfg_b["semilla"]].unlink()
        c = BS.generar_conteos(clase, cfg_b["n_remuestreos"], cfg_b["semilla"], True).copy()
        c[0, 0] += 1
        BS.guardar_indices(ruta[cfg_b["semilla"]], c, semilla=cfg_b["semilla"], estratificado=True, nombres_de_grupo=nombres)
        with pytest.raises(GuardError, match="NO coinciden"):
            _reporte(banco, "fase_a", hasta=1, etiqueta="b_alterado", n_remuestreos=None)
        assert not (banco.raiz / "reporte_fase_a_b_alterado").exists()
    finally:
        for p in d.glob("*.npz"):
            p.unlink()


def test_reporte_fase_a_completa_una_semilla_sin_veto_formal_y_leyendas(banco):
    meta = _reporte(banco, "fase_a", etiqueta="completa")
    res = meta["resultados"]
    assert res["fase_a_completa"] is True and res["rotulo_fase_a"] is None and res["no_se_declara_mejor_de_las_seis"] is None and res["semillas"] == [42]
    comps = res["resultados"]["comparaciones"]
    assert [c["x"] for c in comps] == ["S1_a050", "S2_g200", "S1_a025", "S2_g100", "S1_a075", "S2_g050"] and {c["y"] for c in comps} == {R.BASE}
    for c in comps:
        cl = c["punto_operacion_pareado"]["clasificacion"]
        assert cl["veto_formal"] is False and cl["veto_aplicado"] is False and cl["rotulo"] == "exploratoria; sin veto formal" and c["nivel_evidencia"]["nivel_ii_evaluable"] is False
        assert c["nivel_evidencia"]["nivel"] in (R.NIVELES[0], R.NIVELES[2]) and c["delta_ap_semillas_43_44"] is None
        assert c["decision_reemplazo"]["decision"] == "conserva la base"
    for t in meta["dir"].glob("T-G4-*.csv"):
        enc = t.read_text(encoding="utf-8").splitlines()[0]
        assert enc.startswith("#") and C.LEYENDA in enc and all(l in enc for l in C.LEYENDAS_FIJAS) and "sin uso de test" in enc
    enc3, t3 = _tabla(meta["dir"] / "T-G4-3_punto_operacion.csv")
    assert "solo válido para esa prevalencia; no transferible a otras poblaciones" in enc3 and all(r["rotulo"] == "exploratoria; sin veto formal" for r in t3)
    assert all("/" in r["fp_sobre_no_malignas"] and r["fn_sobre_malignas"].endswith("/10") for r in t3) and all(r["s_star"] == "0.90" for r in t3)
    assert "solo válido para esa prevalencia" in _tabla(meta["dir"] / "T-G4-4_delta_punto_operacion.csv")[0]
    assert not (meta["dir"] / "evidencia_fase_b.json").exists()                              # solo la fase B la emite


def test_fase_a_incompleta_rotula_parciales_y_no_declara_la_mejor_de_las_seis(banco):
    meta = _reporte(banco, "fase_a", hasta=2, etiqueta="parcial")
    res = meta["resultados"]
    assert res["fase_a_completa"] is False and res["no_se_declara_mejor_de_las_seis"] is True and res["rotulo_fase_a"] == R.NOTA_PARCIAL
    assert [c["x"] for c in res["resultados"]["comparaciones"]] == ["S1_a050", "S2_g200"]
    for t in meta["dir"].glob("T-G4-*.csv"):
        assert "parciales exploratorios provisionales" in t.read_text(encoding="utf-8").splitlines()[0]


def test_reporte_rechaza_corridas_faltantes_y_no_escribe_nada(banco):
    d = banco.raiz / "S2_g050/completa/densenet121__s42__f4"
    d.rename(d.with_name("oculta"))
    try:
        with pytest.raises(GuardError, match="faltan 1 de 5 corridas de S2_g050"):
            _reporte(banco, "fase_a", etiqueta="falta")
        assert not (banco.raiz / "reporte_fase_a_falta").exists()
        assert _reporte(banco, "fase_a", hasta=5, etiqueta="sin_la_sexta")["resultados"]["fase_a_completa"] is False         # solo las pedidas deben estar completas
    finally:
        d.with_name("oculta").rename(d)
    e = banco.raiz / "S1_a025/completa/densenet121__s44__f2"
    e.rename(e.with_name("oculta"))
    try:
        with pytest.raises(GuardError, match="faltan 1 de 15 corridas de S1_a025"):
            _reporte(banco, "fase_b", etiqueta="falta")
    finally:
        e.with_name("oculta").rename(e)
    assert not (banco.raiz / "reporte_fase_b_falta").exists()


def test_reporte_no_sobrescribe_integridad_k4_y_codigos_de_salida(banco, capsys):
    with pytest.raises(GuardError, match="ya existe"):
        _reporte(banco, "fase_b")                                                            # el directorio reporte_fase_b ya existe (K5)
    estado = {"n": 0}

    def cambia():
        estado["n"] += 1
        r = banco.kw["verificar"]()
        if estado["n"] > 1:
            r["git_commit_head"] = "z" * 40
        return r
    meta = _reporte(banco, "fase_a", hasta=1, etiqueta="integridad", verificar=cambia)
    assert meta["reporte_aceptado"] is False and meta["invalida_integridad"] and "git_commit_head" in meta["diferencias_inicio_fin"]
    sucia = banco.kw["verificar"]()
    sucia["bloqueos"] = [{"codigo": "arbol_sucio", "detalle": "x"}]
    with pytest.raises(GuardError, match="arbol_sucio"):
        _reporte(banco, "fase_a", hasta=1, etiqueta="sucio", verificar=lambda: sucia)
    malo = copy.deepcopy(banco.config)
    malo["diseno"]["raiz_resultados"] = "results/r7_v2/phash_le8_v1/test/"
    with pytest.raises(GuardError):                                                          # K4: ninguna ruta con componente test
        R.generar_reporte(banco.root, malo, banco.config_g2, banco.particion, modo="fase_a", verificar=banco.kw["verificar"], referencia=REF, n_remuestreos=N_B)
    kw = dict(root=banco.root, config=banco.config, config_g2=banco.config_g2, referencia=REF, n_remuestreos=N_B)
    assert R.main(["--modo", "fase_a", "--hasta", "1", "--etiqueta", "cli_ok"], verificar=banco.kw["verificar"], **kw) == 0
    assert R.main(["--modo", "fase_a", "--etiqueta", "cli_ok"], verificar=banco.kw["verificar"], **kw) == R.CODIGO_SALIDA_GUARDA == 3        # ya existe
    estado["n"] = 0
    assert R.main(["--modo", "fase_a", "--hasta", "1", "--etiqueta", "cli_4"], verificar=cambia, **kw) == R.CODIGO_SALIDA_INTEGRIDAD_INVALIDA == 4
    capsys.readouterr()


def test_guarda_de_r6_en_el_reporte_ausencia_u_otra_elegida_bloquea(banco):
    ruta = banco.root / banco.config["referencia_r6"]["salida"]
    original = ruta.read_bytes()
    try:
        ruta.unlink()
        with pytest.raises(GuardError, match="no existe"):
            _reporte(banco, "fase_a", hasta=1, etiqueta="r6")
        escribir_r6(banco.root, banco.config, {"condicion": "A", "arquitectura": "resnet50"})
        with pytest.raises(GuardError, match="R6 eligió"):
            _reporte(banco, "fase_a", hasta=1, etiqueta="r6")
        assert R.main(["--modo", "fase_a", "--etiqueta", "r6"], root=banco.root, config=banco.config, config_g2=banco.config_g2, verificar=banco.kw["verificar"],
                      referencia=REF, n_remuestreos=N_B) == 3
    finally:
        ruta.write_bytes(original)
    assert not (banco.raiz / "reporte_fase_a_r6").exists()


def test_reglas_y_protocolo_anti_deriva(banco, tmp_path):
    reglas = C.validar_reglas(banco.config)
    assert reglas["s_star"] == "0.90" and reglas["criterios_absolutos"] == {"especificidad_minima": None, "fp_maximo": None, "viabilidad_clinica": "no_determinada"}
    texto = (REPO / C.PROTOCOLO_REL).read_text(encoding="utf-8")
    assert len(texto.splitlines()) <= 90 and "git_commit" not in texto and C.exigir_protocolo_coherente(REPO)
    for frase in C.FRASES_PROTOCOLO:
        raiz = tmp_path / f"p{abs(hash(frase))}"
        (raiz / C.PROTOCOLO_REL).parent.mkdir(parents=True)
        (raiz / C.PROTOCOLO_REL).write_text(texto.replace(frase, "X"), encoding="utf-8")
        with pytest.raises(GuardError, match="no es coherente"):
            C.exigir_protocolo_coherente(raiz)
    (tmp_path / "h").mkdir()
    (tmp_path / "h" / C.PROTOCOLO_REL).parent.mkdir(parents=True)
    (tmp_path / "h" / C.PROTOCOLO_REL).write_text(texto + "\n" + "a" * 64 + "\n", encoding="utf-8")
    with pytest.raises(GuardError, match="no es coherente"):
        C.exigir_protocolo_coherente(tmp_path / "h")
    with pytest.raises(GuardError, match="no existe"):
        C.exigir_protocolo_coherente(tmp_path / "nada")
    cfg = copy.deepcopy(banco.config)
    cfg["reglas_de_decision"]["s_star"] = "0.95"
    with pytest.raises(GuardError, match="reglas_de_decision"):
        _reporte_con_config(banco, cfg)


def _reporte_con_config(e, cfg):
    return R.generar_reporte(e.root, cfg, e.config_g2, e.particion, modo="fase_a", hasta=1, etiqueta="deriva", verificar=e.kw["verificar"], referencia=REF, n_remuestreos=N_B)


# ============================================================================ fase C: agente frente a la base y a la candidata convencional
@pytest.fixture(scope="module")
def banco_c(tmp_path_factory):
    e = _construir(tmp_path_factory, "banco_g4_c")
    e.rep_b = _reporte(e, "fase_b")
    ref = e.raiz / e.config["agente"]["evidencia_fase_b"]
    e.carpeta = None
    for n, v in enumerate((0.35, 0.65), start=1):                                           # 5*2 + 10 = 20 <= 24
        e.carpeta = _propuesta(e, ref, n, valor=v)
    assert correr(e, "fase_c") == 0 and correr(e, "fase_c_confirmacion") == 0
    e.rep_c = _reporte(e, "fase_c")
    return e


def test_reporte_fase_c_compara_con_la_base_y_con_la_candidata_convencional(banco_c):
    res = banco_c.rep_c["resultados"]
    assert res["candidata_convencional"]["config_id"] == "S1_a050" and res["modo"] == "fase_c"
    comps = {(c["x"], c["y"]): c for c in res["resultados"]["comparaciones"]}
    assert {("S1_a050", R.BASE), ("S1_a035", R.BASE), ("S1_a035", "S1_a050"), ("S1_a065", R.BASE), ("S1_a065", "S1_a050")} == set(comps)
    assert comps[("S1_a035", R.BASE)]["semillas"] == [42, 43, 44] and comps[("S1_a065", R.BASE)]["semillas"] == [42]            # solo la mejor se confirmó
    assert comps[("S1_a065", "S1_a050")]["punto_operacion_pareado"]["clasificacion"]["rotulo"] == "exploratoria; sin veto formal"
    promo = res["agente"]["promocion_a_confirmacion"]
    assert promo["elegida"] == "S1_a035" and "exploratoria" in promo["naturaleza"] and promo["regla"] == C.REGLA_CONFIRMACION
    ag = {a["config_id"]: a for a in res["agente"]["propuestas"]}
    assert ag["S1_a035"]["seleccionada_para_confirmacion"] is True and ag["S1_a065"]["seleccionada_para_confirmacion"] is False
    assert ag["S1_a035"]["delta_ap_semilla_42_vs_base"] > 0 and ag["S1_a035"]["delta_ap_semilla_42_vs_candidata_convencional"] > 0
    assert ag["S1_a065"]["promocion_exploratoria_a_confirmacion"] is False and ag["S1_a065"]["delta_ap_semilla_42_vs_candidata_convencional"] <= 0
    assert "promoción del agente a confirmación: regla exploratoria, no evidencia confirmatoria" in (banco_c.rep_c["dir"] / "T-G4-6_decision.csv").read_text(encoding="utf-8").splitlines()[0]
    assert set(ag) == {"S1_a035", "S1_a065"} and all(isinstance(a["atribuible_al_agente"], bool) for a in ag.values())
    for i, a in ag.items():
        vc, vb = comps[(i, "S1_a050")], comps[(i, R.BASE)]
        assert a["atribuible_al_agente"] == R.atribucion_agente(vc["nivel_evidencia"], vc["clasificacion_punto_operacion"], vb["nivel_evidencia"])["atribuible_al_agente"]
        assert a["sha256_bytes_archivo"] == C4.sha256_archivo(banco_c.carpeta / a["propuesta"])
        if not a["atribuible_al_agente"]:
            assert a["nota"]
    assert ag["S1_a065"]["atribuible_al_agente"] is False                                   # una semilla: no evaluable para atribución
    k = banco_c.rep_c["k"]
    assert k == 8 and comps[("S1_a035", R.BASE)]["multiplicidad"]["k"] == 8 and comps[("S1_a035", R.BASE)]["multiplicidad"]["nivel"] == pytest.approx(1 - 0.05 / 8)   # K incluye las del agente
    ev = P.leer_ledger(banco_c.ledger)
    assert P.resumen(ev)["intentos_consumidos"] == 71 and P.resumen(ev)["planificadas_por_fase"]["C"] == 20 and P.maximo_propuestas(ev) == 0


def test_reporte_fase_c_sin_propuestas_o_con_alteradas_se_rechaza(banco_c):
    p = banco_c.carpeta / "propuesta_02.json"
    original = p.read_bytes()
    try:
        _propuesta(banco_c, banco_c.raiz / banco_c.config["agente"]["evidencia_fase_b"], 2, valor=0.65, justificacion="Justificación reescrita después de registrar la propuesta original.")
        with pytest.raises(GuardError, match="cambió tras su registro"):
            _reporte(banco_c, "fase_c", etiqueta="alterada")
    finally:
        p.write_bytes(original)
    assert not (banco_c.raiz / "reporte_fase_c_alterada").exists()


# ============================================================================ AJUSTE 1: selección determinista de la configuración final
def _cmp(x, delta, nivel, clasif, semillas=(42, 43, 44), y=R.BASE):
    n = {"nivel": nivel, "nivel_ii_evaluable": True}
    return {"x": x, "y": y, "semillas": list(semillas), "delta_ap": {"estimacion": delta}, "nivel_evidencia": n, "clasificacion_punto_operacion": {"clasificacion": clasif},
            "decision_reemplazo": R.decidir_reemplazo(n, {"clasificacion": clasif})}


MEJOR, CAND = R.NIVELES[1], R.NIVELES[0]
ORDEN = C.orden_de_configuraciones()


def test_seleccion_una_sola_elegible_y_sin_elegibles_conserva_la_base():
    r = C.seleccionar_configuracion_final([_cmp("S2_g100", 0.03, MEJOR, "no distinguible"), _cmp("S1_a050", 0.05, CAND, "no distinguible")], ORDEN)
    assert r["configuracion_final_seleccionada"] == "S2_g100" and r["conserva_la_base"] is False and r["criterio_desempate_usado"] is None
    assert [c["config_id"] for c in r["cumplen_ordenadas"]] == ["S2_g100"] and r["regla"] == C.REGLA_SELECCION
    nada = C.seleccionar_configuracion_final([_cmp("S1_a050", 0.05, CAND, "no distinguible"), _cmp("S2_g200", 0.04, R.NIVELES[2], "mejora respaldada")], ORDEN)
    assert nada["conserva_la_base"] is True and nada["cumplen_ordenadas"] == [] and nada["configuracion_final_seleccionada"].startswith("BASE")
    assert C.seleccionar_configuracion_final([], ORDEN)["conserva_la_base"] is True


def test_seleccion_varias_elegibles_mayor_delta_ap_medio_y_empate_exacto_por_orden_predefinido():
    cs = [_cmp("S2_g200", 0.030, MEJOR, "no distinguible"), _cmp("S1_a025", 0.045, MEJOR, "mejora respaldada"), _cmp("S1_a050", 0.020, MEJOR, "no distinguible")]
    r = C.seleccionar_configuracion_final(cs, ORDEN)
    assert r["configuracion_final_seleccionada"] == "S1_a025" and [c["config_id"] for c in r["cumplen_ordenadas"]] == ["S1_a025", "S2_g200", "S1_a050"]
    assert r["criterio_desempate_usado"] is None
    emp = C.seleccionar_configuracion_final([_cmp("S1_a025", 0.04, MEJOR, "no distinguible"), _cmp("S2_g200", 0.04, MEJOR, "no distinguible"),
                                             _cmp("S2_g050", 0.04, MEJOR, "no distinguible")], ORDEN)
    assert [c["config_id"] for c in emp["cumplen_ordenadas"]] == ["S2_g200", "S1_a025", "S2_g050"] and emp["configuracion_final_seleccionada"] == "S2_g200"
    assert emp["criterio_desempate_usado"].startswith("orden predefinido")
    props = [{"estrategia": {"id": "S1_a035"}, "archivo": "propuesta_01.json"}, {"estrategia": {"id": "S1_a065"}, "archivo": "propuesta_02.json"}]
    orden = C.orden_de_configuraciones(props)
    assert orden["S1_a050"] < orden["S2_g050"] < orden["S1_a035"] < orden["S1_a065"]                  # las propuestas van después, por número
    e2 = C.seleccionar_configuracion_final([_cmp("S1_a065", 0.04, MEJOR, "no distinguible"), _cmp("S1_a035", 0.04, MEJOR, "no distinguible")], orden)
    assert e2["configuracion_final_seleccionada"] == "S1_a035"
    e3 = C.seleccionar_configuracion_final([_cmp("S1_a035", 0.04, MEJOR, "no distinguible"), _cmp("S2_g050", 0.04, MEJOR, "no distinguible")], orden)
    assert e3["configuracion_final_seleccionada"] == "S2_g050"


def test_seleccion_no_altera_i_ii_ni_el_veto():
    cs = [_cmp("S1_a050", 0.09, MEJOR, "deterioro respaldado"),                                    # (i) sin (ii): el veto la excluye aunque tenga el mayor delta
          _cmp("S2_g200", 0.05, CAND, "no distinguible"),                                          # sin (i)
          _cmp("S1_a025", 0.03, MEJOR, "mejora respaldada"),
          _cmp("S1_a075", 0.08, MEJOR, "no distinguible", semillas=(42,)),                         # una semilla: nunca entra
          _cmp("S2_g100", 0.07, MEJOR, "no distinguible", y="S1_a050")]                            # comparación frente a otra candidata, no frente a la base
    antes = copy.deepcopy(cs)
    r = C.seleccionar_configuracion_final(cs, ORDEN)
    assert cs == antes and r["configuracion_final_seleccionada"] == "S1_a025" and [c["config_id"] for c in r["cumplen_ordenadas"]] == ["S1_a025"]
    assert cs[0]["decision_reemplazo"]["nota"] == "mejora de discriminación con deterioro del punto de operación"      # (i), (ii) y el veto intactos
    assert C.seleccionar_configuracion_final(cs[:1], ORDEN)["conserva_la_base"] is True


def test_el_reporte_registra_la_seleccion_final(banco):
    res = banco.rep_b["resultados"]
    esperado = C.seleccionar_configuracion_final(res["resultados"]["comparaciones"], C.orden_de_configuraciones())
    assert res["decision_base"] == esperado and res["configuracion_final_seleccionada"] == esperado["configuracion_final_seleccionada"]
    assert "eleccion_entre_varias" not in res["decision_base"] and "decide la autora" not in json.dumps(res["decision_base"])
    enc, filas = _tabla(banco.rep_b["dir"] / "T-G4-7_seleccion_final.csv")
    assert C.REGLA_SELECCION in enc and len(filas) == max(1, len(esperado["cumplen_ordenadas"]))
    if esperado["cumplen_ordenadas"]:
        assert filas[0]["config_id"] == esperado["configuracion_final_seleccionada"] and filas[0]["seleccionada"] == "True"
    else:
        assert filas[0]["criterio_desempate"] == "sin elegibles: se conserva la base"
