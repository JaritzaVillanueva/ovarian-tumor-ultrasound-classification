"""Pruebas sintéticas del procedimiento estadístico de R8 (`r8_v2.estadistica`): métricas puntuales contra sklearn, Clopper–Pearson contra valores conocidos, bootstrap estratificado
reproducible y contrastado con una réplica explícita, regla `>=` con empates, VPP indefinido y validaciones. Sin datos reales ni test.
Comando: py -3.13 -B -m pytest tests/test_r8_v2_estadistica.py -q -p no:cacheprovider
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")
sk = pytest.importorskip("sklearn.metrics")

from src.r8_v2 import estadistica as ES  # noqa: E402


def datos(n_pos=15, n_neg=120, semilla=7, empates=False):
    rng = np.random.default_rng(semilla)
    y = np.array([1] * n_pos + [0] * n_neg)
    s = np.clip(np.where(y == 1, rng.normal(0.35, 0.2, y.size), rng.normal(0.12, 0.1, y.size)), 0.0, 1.0)
    if empates:
        s = np.round(s, 2)
    perm = rng.permutation(y.size)
    return y[perm], s[perm]


def test_constantes_aprobadas():
    assert (ES.SEMILLA_BOOTSTRAP, ES.N_BOOTSTRAP, ES.NIVEL_IC, ES.UMBRAL_SECUNDARIO) == (20261123, 10000, 0.95, 0.5)


@pytest.mark.parametrize("k,n,esperado", [(7, 10, (0.3475, 0.9333)), (0, 15, (0.0, 0.2181)), (15, 15, (0.7819, 1.0)), (1, 20, (0.0013, 0.2487))])
def test_clopper_pearson_valores_conocidos(k, n, esperado):
    bajo, alto = ES.clopper_pearson(k, n)
    assert bajo == pytest.approx(esperado[0], abs=2e-4) and alto == pytest.approx(esperado[1], abs=2e-4)


def test_clopper_pearson_contiene_la_proporcion_y_valida_argumentos():
    for k, n in ((3, 15), (14, 15), (110, 120)):
        bajo, alto = ES.clopper_pearson(k, n)
        assert 0.0 <= bajo <= k / n <= alto <= 1.0
    for k, n in ((-1, 5), (6, 5), (0, 0)):
        with pytest.raises(ValueError):
            ES.clopper_pearson(k, n)


@pytest.mark.parametrize("empates", [False, True])
def test_metricas_puntuales_coinciden_con_sklearn(empates):
    y, s = datos(empates=empates)
    r = ES.calcular_resultados(y, s, 0.2, B=50)
    assert r["puntuales"]["ap"] == pytest.approx(sk.average_precision_score(y, s), abs=1e-12)
    assert r["puntuales"]["roc_auc"] == pytest.approx(sk.roc_auc_score(y, s), abs=1e-12)


def test_matriz_umbral_fijo_y_regla_mayor_o_igual_con_empates():
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.9, 0.2, 0.1, 0.3, 0.05])
    m = ES.matriz(y, s, 0.2)                                                    # 0.2 == umbral: positivo (regla >=)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 2, 0)
    assert m["sensibilidad"] == 1.0 and m["especificidad"] == pytest.approx(2 / 3) and m["vpp"] == pytest.approx(2 / 3)
    m2 = ES.matriz(y, s, np.nextafter(0.2, 1.0))
    assert (m2["tp"], m2["fn"]) == (1, 1)


def test_vpp_indefinido_si_no_hay_positivos_predichos():
    y, s = datos()
    r = ES.calcular_resultados(y, s * 0.0, 0.5, B=20)
    assert r["umbral_primario"]["vpp"] is None and r["umbral_primario"]["fp"] == 0 and r["umbral_primario"]["fn"] == 15


def test_resultados_estructura_ic_y_umbral_secundario_descriptivo():
    y, s = datos()
    r = ES.calcular_resultados(y, s, 0.08456226376195748, B=200)
    p = r["umbral_primario"]
    assert r["n"] == 135 and r["n_positivas"] == 15 and r["n_negativas"] == 120 and r["prevalencia_test"] == pytest.approx(15 / 135)
    assert p["tp"] + p["fn"] == 15 and p["tn"] + p["fp"] == 120 and p["umbral"] == 0.08456226376195748
    lo, hi = p["sensibilidad_ic95_clopper_pearson"]
    assert lo <= p["sensibilidad"] <= hi
    assert r["umbral_secundario_0_5_descriptivo"]["umbral"] == 0.5 and "sensibilidad_ic95_clopper_pearson" not in r["umbral_secundario_0_5_descriptivo"]
    ic = r["ic95_bootstrap_percentil_estratificado"]
    for m in ("ap", "roc_auc"):
        assert ic[m]["n_validos"] == 200 and 0.0 <= ic[m]["p2_5"] <= ic[m]["p97_5"] <= 1.0
    assert ic["B"] == 200 and ic["semilla"] == ES.SEMILLA_BOOTSTRAP


def test_bootstrap_reproducible_y_depende_de_la_semilla():
    y, s = datos()
    a = ES.bootstrap_estratificado(y, s, 100, 123)
    b = ES.bootstrap_estratificado(y, s, 100, 123)
    c = ES.bootstrap_estratificado(y, s, 100, 124)
    assert np.array_equal(a["ap"], b["ap"]) and np.array_equal(a["roc_auc"], b["roc_auc"])
    assert not np.array_equal(a["ap"], c["ap"])
    r1, r2 = ES.calcular_resultados(y, s, 0.2, B=100), ES.calcular_resultados(y, s, 0.2, B=100)
    assert r1 == r2                                                              # incluye los SHA-256 de las réplicas


def test_bootstrap_coincide_con_una_replica_explicita_con_sklearn():
    """El procedimiento documentado (imágenes ordenadas, posiciones positivas (B, P) y luego negativas (B, N)) reproduce cada réplica con sklearn."""
    y, s = datos(empates=True)
    B, semilla = 12, 20261123
    boot = ES.bootstrap_estratificado(y, s, B, semilla)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(semilla)
    sel_pos = pos[rng.integers(0, pos.size, size=(B, pos.size))]
    sel_neg = neg[rng.integers(0, neg.size, size=(B, neg.size))]
    for k in range(B):
        idx = np.concatenate([sel_pos[k], sel_neg[k]])
        assert sel_pos[k].size == pos.size and sel_neg[k].size == neg.size              # estratificado: composición fija en cada réplica
        assert boot["ap"][k] == pytest.approx(sk.average_precision_score(y[idx], s[idx]), abs=1e-12)
        assert boot["roc_auc"][k] == pytest.approx(sk.roc_auc_score(y[idx], s[idx]), abs=1e-12)


@pytest.mark.parametrize("y,s", [(np.zeros(5), np.linspace(0, 1, 5)), (np.ones(5), np.linspace(0, 1, 5)), (np.array([0, 1, 1]), np.array([0.1, np.nan, 0.5])),
                                 (np.array([0, 1, 1]), np.array([0.1, 1.2, 0.5])), (np.array([0, 2, 1]), np.array([0.1, 0.2, 0.5])), (np.array([]), np.array([]))])
def test_entradas_invalidas_se_rechazan(y, s):
    with pytest.raises(ValueError):
        ES.calcular_resultados(y, s, 0.2, B=5)


def test_modulo_sin_calibracion_ni_acceso_a_datos():
    import inspect
    fuente = inspect.getsource(ES)
    assert "open(" not in fuente and "read_" not in fuente and "calibrar" not in fuente.lower().replace("recalibr", "")
    assert not hasattr(ES, "calibrar_umbral")
