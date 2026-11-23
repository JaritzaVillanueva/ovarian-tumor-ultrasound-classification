"""Pruebas focalizadas de G4 (`g4_optimizacion_v1`) con datos SINTÉTICOS en directorios temporales. No abren test, ni datos reales, ni entrenan modelos reales.

Cubre: ledger/presupuesto (tope duro de intentos), orden fijo y `--hasta`, promoción, autorización, humo, pesos efectivos, entrenador (con torch: skip claro
si falta), integridad, base de R6, comparabilidad con la base y mecanismo del agente. El REPORTE (niveles de evidencia, multiplicidad, punto de operación) y el
protocolo están PENDIENTES por la regla de punto de operación: sus pruebas no existen todavía.
Comando: py -3.13 -B -m pytest tests/test_r7_v2_g4.py -q -p no:cacheprovider
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import zlib
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("yaml")
pytest.importorskip("PIL")

from src.r4_v2 import common as C4  # noqa: E402
from src.r4_v2 import entrenamiento as EN  # noqa: E402
from src.r4_v2 import particion as P4  # noqa: E402
from src.r4_v2.common import GuardError  # noqa: E402
from src.r7_v2 import g4_agente as AGT  # noqa: E402
from src.r7_v2 import g4_comun as C  # noqa: E402
from src.r7_v2 import g4_ejecutar as E  # noqa: E402
from src.r7_v2 import g4_entrenador as T  # noqa: E402
from src.r7_v2 import g4_presupuesto as P  # noqa: E402
from tests.test_r5_v2 import construir_datos  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
HAY_TORCH = importlib.util.find_spec("torch") is not None
requiere_torch = pytest.mark.skipif(not HAY_TORCH, reason="torch no instalado: ejecutar en Falcon")
AUT = "CONFIRMADO:D-040"
REF = {"git_blob_id": "r" * 40, "sha256_contenido_blob": "r" * 64}
RAIZ_G4 = "results/r7_v2/phash_le8_v1/g4_optimizacion_v1"


@pytest.fixture(autouse=True)
def _git_aislado(monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")


# ============================================================================ sustitutos y datos sintéticos
def verificar_falso(head="a" * 40):
    def f():
        h = {"git_blob_id": "c" * 40, "sha256_contenido_blob": "c" * 64, "sha256_bytes_archivo": "1" * 64}
        return {"git_commit_head": head, "rama": "rama-sintetica", "status_porcelain": [], "status_aux": [], "bloqueos": [], "protocolo_versionado": True,
                "entrada": "modulo", "modulos_en_repo": [], "modulos_virtuales_tolerados": [], "archivos": {C.CONFIG_REL: h, C4.CONFIG_REL: h}}
    return f


def calidad_por_perdida(p):
    """Efecto sintético de la estrategia sobre la calidad del clasificador (solo para comprobar el flujo, sin significado)."""
    if p["tipo"] == "cross_entropy_ponderada":
        return {1.0: 0.0, 0.5: 0.9, 0.25: 0.7, 0.35: 6.0, 0.65: 0.3, 0.8: 4.0}.get(p["alpha_pesos_de_clase"], 0.0)
    return {2.0: -0.5}.get(p["gamma"], 0.0)


def sintetico(llamadas=None, falla=None):
    """Entrenador sintético con la firma del de G2/G4; `falla(ctx)` (opcional) lanza para simular un fallo técnico."""
    def entrenar(ctx):
        if falla is not None:
            falla(ctx)
        perd = ctx["config"]["receta"]["perdida_entrenamiento"]
        if llamadas is not None:
            llamadas.append({"arq": ctx["arq"], "semilla": ctx["semilla"], "fold": ctx["fold"], "perdida": perd, "pesos": dict(ctx["pesos"])})
        rng = np.random.default_rng(zlib.crc32(f"{ctx['arq']}{ctx['semilla']}{ctx['fold']}".encode()))
        ids, y = [r["image_id"] for r in ctx["val"]], np.array([r["label_v2"] for r in ctx["val"]])
        n_ep = min(9, ctx["epocas_max"])
        perdida = [0.9, 0.7, 0.5, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5][:n_ep] if n_ep == 9 else [0.9 - 0.1 * i for i in range(n_ep)]
        calidad = 0.8 + calidad_por_perdida(perd)
        por = {e: (ids, y, (1 / (1 + np.exp(-(calidad * (2 * y - 1) + rng.normal(0, 1.2, y.size) - 1.5)))).astype(np.float32)) for e in range(1, n_ep + 1)}
        EN.escribir_prob_por_epoca(ctx["dir"] / "probabilidades_por_epoca.csv", por)
        (ctx["dir"] / "checkpoint_mejor.pth").write_bytes(b"ckpt " + bytes(str(ctx["fold"]), "ascii"))
        el = EN.epoca_checkpoint(perdida)
        return {"historial": {"train_loss": [1.0] * n_ep, "val_loss_unweighted": perdida}, "checkpoint": ctx["dir"] / "checkpoint_mejor.pth",
                "prob_recarga": por[el][2] + np.float32(1e-4), "epoca_checkpoint_guardado": el, "pesos_preentrenados": {}, "version_difiere_referencia": False,
                "entorno": {"python": "sintetico", "dispositivo": "cpu", "gpu": "GPU-sintetica"}}
    return entrenar


def escribir_r6(root, config, elegida=None):
    """Fixture EXPLÍCITO de la guarda de R6: archivo sintético en el directorio temporal (la guarda real no tiene bandera que la debilite)."""
    ruta = Path(root) / config["referencia_r6"]["salida"]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps({"elegida": elegida or {"condicion": "A", "arquitectura": "densenet121"}}), encoding="utf-8")
    return ruta


def armar(root):
    _, config_g2 = construir_datos(root)
    config = C.cargar_config_g4(REPO / C.CONFIG_REL)
    particion = P4.cargar_particion(root, config_g2)
    escribir_r6(root, config)
    (Path(root) / C.PROTOCOLO_REL).parent.mkdir(parents=True, exist_ok=True)
    (Path(root) / C.PROTOCOLO_REL).write_bytes((REPO / C.PROTOCOLO_REL).read_bytes())            # el reporte valida las reglas contra el protocolo
    cb = C.config_g2_base(config_g2, config, config["base"]["semillas"])
    raiz_a = Path(root) / config_g2["diseno"]["raiz_resultados"].strip("/") / "completa"
    for s in cb["diseno"]["semillas"]:                                                     # base A|densenet121 de G2 (sintética): 3 semillas x 5 folds
        for k in cb["diseno"]["folds"]:
            r = EN.ejecutar_corrida(root, cb, particion, modo="completa", arq="densenet121", semilla=s, fold=k, raiz_modo=raiz_a, verificar=verificar_falso(),
                                    entrenar=sintetico(), referencia=REF, atestaciones={"autorizacion_ejecucion": "SINTETICA"})
            assert r["estado"] == "completada", r
    class Env:
        pass
    e = Env()
    e.root, e.config, e.config_g2, e.particion = Path(root), config, config_g2, particion
    e.llamadas = []
    e.kw = dict(verificar=verificar_falso(), entrenar=sintetico(e.llamadas), referencia=REF)
    e.raiz = Path(root) / RAIZ_G4
    e.ledger = e.raiz / "presupuesto_ledger.jsonl"
    return e


@pytest.fixture
def ent(tmp_path):
    return armar(tmp_path)


def correr(e, modo, **kw):
    args = dict(autorizacion=AUT) | e.kw | kw
    return E.ejecutar(e.root, e.config, e.config_g2, modo=modo, **args)


def resumen(e):
    return P.resumen(P.leer_ledger(e.ledger))


def rc_de(e, cid, semilla, fold, modo="completa", intento="intento_01"):
    return json.loads((e.raiz / cid / modo / f"densenet121__s{semilla}__f{fold}" / intento / "run_config.json").read_text(encoding="utf-8"))


# ============================================================================ ledger y presupuesto (tope duro de INTENTOS)
def _ev(path, evento, fase="A", cid="S1_a050", rid="r", intento=1, **kw):
    return P.registrar(path, fase=fase, config_id=cid, id_corrida=rid, intento=intento, evento=evento, head="h" * 40, autorizacion=AUT, **kw)


def test_ledger_append_only_y_resumen_derivado(tmp_path):
    lg = tmp_path / "x" / "presupuesto_ledger.jsonl"
    _ev(lg, "planificada", rid="r1")
    n1 = lg.read_text(encoding="utf-8")
    _ev(lg, "planificada", rid="r2")
    _ev(lg, "iniciada", rid="r1", segundos=None)
    _ev(lg, "fallida", rid="r1", segundos=10.5, gpu="RTX", dispositivo="cuda")
    _ev(lg, "reintento", rid="r1", intento=2)
    _ev(lg, "iniciada", rid="r1", intento=2)
    _ev(lg, "completada", rid="r1", intento=2, segundos=20.0, gpu="RTX", dispositivo="cuda")
    _ev(lg, "iniciada", fase="humo", cid="S2_g200", rid="h", nota="humo")
    _ev(lg, "completada", fase="humo", cid="S2_g200", rid="h", segundos=1.0, nota="humo")
    assert lg.read_text(encoding="utf-8").startswith(n1) and len(lg.read_text(encoding="utf-8").splitlines()) == 9    # solo se agregan líneas
    ev = P.leer_ledger(lg)
    r = P.resumen(ev)
    assert r["planificadas"] == 2 and r["completadas"] == 1 and r["intentos_fallidos"] == 1 and r["reintentos"] == 1
    assert r["intentos_consumidos"] == 3 == P.consumidos(ev) and r["humo_consumidos"] == 1 and r["restante"] == 72 and r["iniciadas_sin_cierre"] == 0
    assert r["consumo_segundos"] == pytest.approx(31.5) and r["gpus"] == ["RTX"] and r["dispositivos"] == ["cuda"]
    assert set(json.loads(lg.read_text(encoding="utf-8").splitlines()[0])) >= {"fecha", "fase", "config_id", "id_corrida", "intento", "evento", "segundos",
                                                                             "dispositivo", "gpu", "head", "autorizacion"}
    with pytest.raises(GuardError):
        _ev(lg, "otra")
    lg.write_text(lg.read_text(encoding="utf-8") + "no json\n", encoding="utf-8")
    with pytest.raises(GuardError, match="corrupto"):
        P.leer_ledger(lg)


def _relleno(path, n, fase="A"):
    for i in range(n):
        _ev(path, "iniciada", fase=fase, rid=f"x{i}")


def test_tope_duro_cuenta_intentos_fallidos_y_reintentos_no_corridas(tmp_path):
    lg = tmp_path / "l.jsonl"
    _relleno(lg, 73)
    _ev(lg, "fallida", rid="x0")
    _ev(lg, "reintento", rid="x0", intento=2)
    _ev(lg, "iniciada", rid="x0", intento=2)                                               # 74 consumidos (con fallidos y reintento), 0 completadas
    ev = P.leer_ledger(lg)
    assert P.consumidos(ev) == 74 and P.resumen(ev)["completadas"] == 0 and P.resumen(ev)["restante"] == 1
    P.guardia(ev, 1)
    with pytest.raises(P.PresupuestoAgotado, match="exceden"):
        P.guardia(ev, 2)                                                                   # consumidos + pendientes > 75
    _ev(lg, "iniciada", rid="y")
    with pytest.raises(P.PresupuestoAgotado, match="tope duro"):
        P.guardia(P.leer_ledger(lg), 1)
    with pytest.raises(P.PresupuestoAgotado):
        P.guardia(P.leer_ledger(lg), 0)
    assert issubclass(P.PresupuestoAgotado, GuardError) and P.CODIGO_SALIDA_PRESUPUESTO == 5
    P.guardia(ev[:10], 50, reserva=15)                                                     # 9 + 50 + 15 <= 75
    with pytest.raises(P.PresupuestoAgotado):
        P.guardia(ev[:10], 50, reserva=17)


def test_reparto_de_c_es_75_menos_consumidos_tras_b_y_nunca_se_amplia(tmp_path):
    lg = tmp_path / "l.jsonl"
    _relleno(lg, 31, "A")                                                                  # A completa (30) + el humo
    _relleno(lg, 20, "B")
    assert P.presupuesto_fase_c(P.leer_ledger(lg)) == 24                                   # 75 - 51: el humo ya consume margen
    _ev(lg, "reintento", fase="B", rid="b", intento=2)
    _relleno(lg, 3, "B")                                                                   # los reintentos de B reducen C
    assert P.presupuesto_fase_c(P.leer_ledger(lg)) == 21
    vacio = tmp_path / "v.jsonl"
    _relleno(vacio, 10)
    assert P.presupuesto_fase_c(P.leer_ledger(vacio)) == 25 and P.PLANIFICADAS_POR_FASE == {"A": 30, "B": 20, "C": 25}   # nunca > 25 planificadas
    assert P.presupuesto_fase_c(P.leer_ledger(lg), tope=75) <= P.TOPE_INTENTOS


def test_planificadas_sin_duplicar_y_con_tope_por_fase(tmp_path):
    lg = tmp_path / "l.jsonl"
    items = [{"config_id": "S1_a050", "id": f"r{i}"} for i in range(30)]
    assert len(P.planificar(lg, [], "A", items, head="h", autorizacion=AUT)) == 30
    assert P.planificar(lg, P.leer_ledger(lg), "A", items, head="h", autorizacion=AUT) == []         # reanudar no duplica
    assert P.resumen(P.leer_ledger(lg))["planificadas"] == 30 and P.resumen(P.leer_ledger(lg))["planificadas_por_fase"]["A"] == 30
    with pytest.raises(P.PresupuestoAgotado, match="planificadas"):
        P.planificar(lg, P.leer_ledger(lg), "A", [{"config_id": "S2_g200", "id": "extra"}], head="h", autorizacion=AUT)


# ============================================================================ configuración, estrategias, orden fijo
def test_configuracion_real_valida_y_es_invariante():
    config = C.cargar_config_g4(REPO / C.CONFIG_REL)
    assert config["protocolo"]["git_commit"] is None and config["protocolo"]["sha256_contenido_blob"] is None and config["protocolo"]["conformidad_academica_asesor"] == "PENDIENTE"
    assert config["referencia_g2"]["sha256_contenido_blob"] == "6626f34e654db0ca119e2e1fab11bd12f68d3791481c55cc49c2e1e557fb17ff"
    assert config["acceso_test"]["estado"] == "cerrado" and config["presupuesto"]["tope_intentos"] == 75
    assert [e["id"] for e in C.estrategias_fase_a(config)] == ["S1_a050", "S2_g200", "S1_a025", "S2_g100", "S1_a075", "S2_g050"]
    texto = (REPO / C.CONFIG_REL).read_text(encoding="utf-8")
    assert texto.count("git_commit: null") == 1 and texto.count("sha256_contenido_blob: null") == 1                 # una línea cada una para `sed`
    cambios = [lambda c: c["estrategias"].reverse(), lambda c: c["estrategias"][0].update(alpha=0.6), lambda c: c["presupuesto"].update(tope_intentos=100),
               lambda c: c["presupuesto"]["planificadas"].update(C=30), lambda c: c["acceso_test"].update(estado="abierto"),
               lambda c: c["base"].update(arquitectura="resnet50"), lambda c: c["diseno"].update(raiz_resultados="results/r7_v2/test/x/"),
               lambda c: c["diseno"].update(raiz_resultados="results/r4_v2/x/"), lambda c: c["diseno"].update(fase_a_semilla=43),
               lambda c: c["promocion"].update(max_promovidas=3), lambda c: c["integridad"]["archivos_con_huella"].remove(C.PROTOCOLO_REL),
               lambda c: c["reglas_de_decision"].update(s_star="0.85"), lambda c: c["reglas_de_decision"]["veto"].update(formal_solo_con_semillas=1),
               lambda c: c["reglas_de_decision"]["criterios_absolutos"].update(especificidad_minima=0.5), lambda c: c.pop("reglas_de_decision"),
               lambda c: c["reglas_de_decision"]["multiplicidad"].update(alfa=0.1), lambda c: c["reglas_de_decision"]["veto"].update(prevalece_sobre_mejora=False),
               lambda c: c["reglas_de_decision"]["seleccion_final"].update(regla="elige la autora"), lambda c: c["reglas_de_decision"]["seleccion_final"]["orden_predefinido"].reverse(),
               lambda c: c["reglas_de_decision"]["agente"]["promocion_a_confirmacion"].update(una_sola=False),
               lambda c: c["reglas_de_decision"]["agente"]["promocion_a_confirmacion"].update(regla="solo frente a la base")]
    for cambio in cambios:
        mala = copy.deepcopy(config)
        cambio(mala)
        with pytest.raises(GuardError):
            C.validar_config(mala)
    todas = {p.name for p in (REPO / "src/r7_v2").glob("g4_*.py")}                        # la integridad cubre todo el código G4 existente
    assert {Path(r).name for r in config["integridad"]["archivos_con_huella"] if r.startswith("src/r7_v2/g4_")} == todas and "g4_reporte.py" in todas


def test_estrategias_un_solo_factor_y_perdida_efectiva():
    s1, s2 = C.estrategia_de("S1", 0.5), C.estrategia_de("S2", 2.0)
    assert (s1["id"], s1["alpha"], s1["gamma"]) == ("S1_a050", 0.5, None) and (s2["id"], s2["alpha"], s2["gamma"]) == ("S2_g200", 1.0, 2.0)
    assert C.perdida_de(s1) == {"tipo": "cross_entropy_ponderada", "alpha_pesos_de_clase": 0.5}
    assert C.perdida_de(s2) == {"tipo": "focal_ponderada", "alpha_pesos_de_clase": 1.0, "gamma": 2.0}           # S2 mantiene pesos de clase alpha = 1.0
    for fam, v in (("S1", 1.2), ("S1", -0.1), ("S2", 3.5), ("S2", 0.255), ("S3", 0.5), ("S1", True), ("S1", "0.5")):
        with pytest.raises(GuardError):
            C.estrategia_de(fam, v)
    assert C.id_de("S1", 0.35) == "S1_a035" and C.id_de("S2", 3.0) == "S2_g300"


def test_orden_fijo_y_hasta(ent):
    todas = [e["id"] for e in C.estrategias_fase_a(ent.config)]
    assert [e["id"] for e in C.seleccionar_fase_a(ent.config)] == todas
    assert [e["id"] for e in C.seleccionar_fase_a(ent.config, 2)] == todas[:2] and [e["id"] for e in C.seleccionar_fase_a(ent.config, 6)] == todas
    for malo in (0, 7, -1, "2", True, 2.5):
        with pytest.raises(GuardError, match="--hasta"):
            C.seleccionar_fase_a(ent.config, malo)
    # el orden no se salta: una configuración posterior con intentos mientras la anterior está incompleta bloquea
    (ent.raiz / "S2_g100/completa/densenet121__s42__f0/intento_01").mkdir(parents=True)
    ctx = E.Ctx(ent.root, ent.config, ent.config_g2, ent.particion, REF, verificar_falso(), None, AUT, "h", ent.ledger, None, None)
    with pytest.raises(GuardError, match="orden fijo"):
        E.exigir_orden_fase_a(ctx, C.seleccionar_fase_a(ent.config, 4))
    with pytest.raises(GuardError, match="--hasta"):
        correr(ent, "humo", hasta=2)
    with pytest.raises(GuardError, match="humo-config"):
        correr(ent, "humo", humo_config="S3_x")


# ============================================================================ promoción (pura: AP sin redondear, empates, máximo, banderas)
ORDEN = ["S1_a050", "S2_g200", "S1_a025", "S2_g100", "S1_a075", "S2_g050"]


def test_promocion_ap_sin_redondear_empates_por_orden_y_maximo_dos():
    d = {"S1_a050": 0.0200000001, "S2_g200": 0.0200000002, "S1_a025": 0.02, "S2_g100": 0.05, "S1_a075": 0.0, "S2_g050": -0.01}
    r = C.decidir_promocion(d, ORDEN)
    assert r["promovidas"] == ["S2_g100", "S2_g200"]                                       # 4 decimales los igualarían: el AP sin redondear decide; máx. 2
    assert r["fase_a_completa"] is True and r["promocion_provisional"] is False and r["no_se_declara_mejor_de_las_seis"] is False
    empate = C.decidir_promocion({**d, "S2_g200": 0.05, "S1_a050": 0.05, "S1_a075": 0.05}, ORDEN)
    assert empate["promovidas"] == ["S1_a050", "S2_g200"]                                  # empate exacto: orden predefinido
    assert C.decidir_promocion({i: (0.0 if n % 2 else -0.3) for n, i in enumerate(ORDEN)}, ORDEN)["promovidas"] == []      # ΔAP <= 0 no promueve (0 incluido)
    assert C.decidir_promocion({**d, "S2_g100": 0.07}, ORDEN)["delta_ap"]["S2_g100"] == 0.07


def test_promocion_con_fase_a_incompleta_marca_banderas():
    r = C.decidir_promocion({"S1_a050": -0.01, "S2_g200": 0.03, **{i: None for i in ORDEN[2:]}}, ORDEN)
    assert r["promovidas"] == ["S2_g200"] and r["evaluadas"] == ORDEN[:2] and r["incompletas"] == ORDEN[2:]
    assert r["fase_a_completa"] is False and r["promocion_provisional"] is True and r["no_se_declara_mejor_de_las_seis"] is True


# ============================================================================ pesos efectivos y entrenador
def test_pesos_efectivos_alpha_exacto_y_restauracion():
    y = [0] * 78 + [1] * 2
    assert T.pesos_de_clase_g4(y, 1.0) == P4.pesos_de_clase(y, 1.0)                       # base: idéntico a G2
    w = T.pesos_de_clase_g4(y, 0.5)
    esperado0 = 80 * 78 ** -0.5 / (78 * 78 ** -0.5 + 2 * 2 ** -0.5)
    assert w[0] == pytest.approx(esperado0) and w[0] * 78 + w[1] * 2 == pytest.approx(80)                  # peso medio por muestra = 1
    assert T.pesos_de_clase_g4(y, 0.0) == {0: pytest.approx(1.0), 1: pytest.approx(1.0)}
    for malo in (-0.1, 1.5, True, None):
        with pytest.raises(GuardError):
            T.pesos_de_clase_g4(y, malo)
    original = EN.pesos_de_clase
    with pytest.raises(GuardError, match="solo usa"):
        EN.pesos_de_clase(y, 0.5)                                                          # G2 sin editar: rechaza alpha != 1
    with T.pesos_efectivos():
        assert EN.pesos_de_clase is T.pesos_de_clase_g4 and EN.pesos_de_clase(y, 0.5)[0] == w[0]
        with pytest.raises(GuardError, match="sustituido"):
            with T.pesos_efectivos():
                pass
    assert EN.pesos_de_clase is original is P4.pesos_de_clase
    with pytest.raises(RuntimeError):
        with T.pesos_efectivos():
            raise RuntimeError("fallo dentro del bloque")
    assert EN.pesos_de_clase is original                                                  # se restaura SIEMPRE


def test_validar_perdida():
    ok = {"tipo": "focal_ponderada", "alpha_pesos_de_clase": 1.0, "gamma": 2.0}
    assert T.validar_perdida(ok) is ok
    for mala in ({"tipo": "otra"}, {"tipo": "focal_ponderada", "alpha_pesos_de_clase": 0.5, "gamma": 2.0}, {"tipo": "focal_ponderada", "alpha_pesos_de_clase": 1.0},
                 {"tipo": "cross_entropy_ponderada", "alpha_pesos_de_clase": 0.5, "gamma": 2.0}):       # no se combinan alpha y gamma
        with pytest.raises(GuardError):
            T.validar_perdida(mala)


@requiere_torch
def test_criterio_base_igual_a_g2_y_focal_gamma_cero_igual_a_ce_ponderada():
    import torch
    import torch.nn as nn
    from src.classification.utils import compute_class_weights
    torch.manual_seed(0)
    y = [0] * 78 + [1] * 2
    logits, tgt = torch.randn(32, 2), torch.tensor([0] * 24 + [1] * 8)
    w1 = compute_class_weights(y, alpha=1.0)
    g2 = nn.CrossEntropyLoss(weight=w1)(logits, tgt)                                       # lo que construye `entrenar_fold_torch` de G2
    base = T.construir_criterio({"tipo": "cross_entropy_ponderada", "alpha_pesos_de_clase": 1.0}, w1)
    assert type(base) is nn.CrossEntropyLoss and torch.equal(base(logits, tgt), g2)
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):                                              # pesos del parche == compute_class_weights
        pw = T.pesos_de_clase_g4(y, alpha)
        assert torch.allclose(torch.tensor([pw[0], pw[1]], dtype=torch.float32), compute_class_weights(y, alpha=alpha))
    w5 = compute_class_weights(y, alpha=0.5)
    s1 = T.construir_criterio({"tipo": "cross_entropy_ponderada", "alpha_pesos_de_clase": 0.5}, w5)
    assert torch.equal(s1(logits, tgt), nn.CrossEntropyLoss(weight=w5)(logits, tgt))
    f0 = T.construir_criterio({"tipo": "focal_ponderada", "alpha_pesos_de_clase": 1.0, "gamma": 0.0}, w1)
    assert type(f0).__name__ == "WeightedFocalLoss" and torch.allclose(f0(logits, tgt), g2, atol=1e-6)    # focal gamma = 0 == CE ponderada
    f2 = T.construir_criterio({"tipo": "focal_ponderada", "alpha_pesos_de_clase": 1.0, "gamma": 2.0}, w1)
    assert torch.isfinite(f2(logits, tgt)) and not torch.allclose(f2(logits, tgt), g2)
    val = nn.CrossEntropyLoss()                                                            # validación sin ponderar, igual que G2 (no depende de la estrategia)
    assert torch.equal(val(logits, tgt), nn.CrossEntropyLoss()(logits, tgt))


# ============================================================================ autorización, humo, fase A, orden, promoción y fase B (flujo sintético)
@pytest.mark.parametrize("modo", list(C.MODOS))
@pytest.mark.parametrize("valor", ["", None, "CONFIRMADO", "CONFIRMADO:D-4", "confirmado:D-040", "CONFIRMADO:AUT-040", " CONFIRMADO:D-040", "PENDIENTE"])
def test_autorizacion_exigida_y_no_se_lee_ni_se_escribe(ent, modo, valor):
    with pytest.raises(GuardError, match="autorizacion-ejecucion"):
        correr(ent, modo, autorizacion=valor)
    assert not ent.raiz.exists() and not ent.llamadas
    assert E.construir_parser().parse_args(["--modo", "humo"]).autorizacion_ejecucion == ""


def test_main_sin_autorizacion_no_toca_nada(tmp_path, capsys):
    assert E.main(["--modo", "fase_a"], root=tmp_path) == 3 and "autorizacion-ejecucion" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_humo_consume_un_intento_pero_no_cuenta_como_resultado(ent):
    assert correr(ent, "humo") == 0
    d = ent.raiz / "S2_g200/humo/densenet121__s42__f0/intento_01"
    rc = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
    assert rc["modo"] == "humo" and rc["epocas_max"] == 1 and rc["estado"] == "completada" and rc["condicion"] == "A+S2_g200"
    sello = json.loads((d / "sello_g4.json").read_text(encoding="utf-8"))
    assert sello["humo"] is True and sello["fase"] == "humo" and sello["autorizacion_ejecucion"] == AUT and sello["gamma"] == 2.0 and sello["alpha"] == 1.0
    assert "partición provisional" in sello["leyenda"] and sello["sha256_bytes_run_config"] == C4.sha256_archivo(d / "run_config.json")
    r = resumen(ent)
    assert r["intentos_consumidos"] == 1 and r["humo_consumidos"] == 1 and r["completadas"] == 0 and r["planificadas"] == 0 and r["restante"] == 74
    ev = P.leer_ledger(ent.ledger)
    assert [e["evento"] for e in ev] == ["iniciada", "completada"] and all(e["nota"] == "humo" and e["fase"] == "humo" and e["autorizacion"] == AUT for e in ev)
    assert correr(ent, "humo") == 0 and resumen(ent)["intentos_consumidos"] == 1           # ya completado: se omite, no consume otro intento
    assert len(ent.llamadas) == 1 and not (ent.raiz / "S1_a050").exists()


def test_fase_a_exige_humo_previo_del_mismo_commit_yaml_y_particion(ent):
    with pytest.raises(GuardError, match="humo"):
        correr(ent, "fase_a", hasta=1)
    assert not (ent.raiz / "S1_a050").exists() and not ent.ledger.exists()
    correr(ent, "humo")
    ent.kw["verificar"] = verificar_falso(head="b" * 40)                                   # otro commit
    with pytest.raises(GuardError, match="humo"):
        correr(ent, "fase_a", hasta=1)
    assert not (ent.raiz / "S1_a050").exists()


def test_fase_a_hasta_pesos_efectivos_ledger_reanudacion_y_promocion_provisional(ent):
    correr(ent, "humo")
    assert correr(ent, "fase_a", hasta=2) == 0
    r = resumen(ent)
    assert r["planificadas"] == 30 and r["planificadas_por_fase"]["A"] == 30 and r["completadas"] == 10 and r["intentos_consumidos"] == 11 and r["restante"] == 64
    assert not (ent.raiz / "S1_a025").exists() and sorted(p.name for p in (ent.raiz / "S1_a050/completa").iterdir()) == [f"densenet121__s42__f{k}" for k in range(5)]
    assert all("__s42__" in p.name for p in ent.raiz.glob("*/completa/*"))                 # fase A: solo la semilla 42
    # pesos EFECTIVOS y estrategia en el run_config (no los de alpha = 1 de G2)
    rc = rc_de(ent, "S1_a050", 42, 1)
    n1, n = rc["positivas_train"], rc["n_train"]
    n0 = n - n1
    z = n0 * n0 ** -0.5 + n1 * n1 ** -0.5
    assert rc["pesos_de_clase"]["0"] == pytest.approx(n * n0 ** -0.5 / z) and rc["pesos_de_clase"]["1"] == pytest.approx(n * n1 ** -0.5 / z)
    assert rc["pesos_de_clase"]["1"] != pytest.approx(n / (2 * n1)) and rc["receta"]["perdida_entrenamiento"] == {"tipo": "cross_entropy_ponderada", "alpha_pesos_de_clase": 0.5}
    rc2 = rc_de(ent, "S2_g200", 42, 1)
    assert rc2["pesos_de_clase"]["1"] == pytest.approx(n / (2 * n1)) and rc2["receta"]["perdida_entrenamiento"]["gamma"] == 2.0 and rc2["condicion"] == "A+S2_g200"
    base = rc_de(ent, "S1_a050", 42, 1)["receta"]
    assert {k: v for k, v in base.items() if k != "perdida_entrenamiento"} == {k: v for k, v in ent.config_g2["receta"].items() if k != "perdida_entrenamiento"}
    s = json.loads((ent.raiz / "S1_a050/completa/densenet121__s42__f1/intento_01/sello_g4.json").read_text(encoding="utf-8"))
    assert s["familia"] == "S1" and s["alpha"] == 0.5 and s["gamma"] is None and s["git_commit_head"] == "a" * 40 and s["comparabilidad_con_base"]["estado"] == "ok"
    assert EN.pesos_de_clase is P4.pesos_de_clase                                          # G2 restaurado tras las corridas
    n_llamadas = len(ent.llamadas)
    assert correr(ent, "fase_a", hasta=2) == 0 and len(ent.llamadas) == n_llamadas and resumen(ent)["intentos_consumidos"] == 11       # reanuda: omite completadas
    # promoción provisional (fase A incompleta)
    assert correr(ent, "promocion") == 0
    prom = json.loads((ent.raiz / "promocion.json").read_text(encoding="utf-8"))
    assert prom["promovidas"] == ["S1_a050"] and prom["promocion_provisional"] is True and prom["fase_a_completa"] is False
    assert prom["no_se_declara_mejor_de_las_seis"] is True and prom["incompletas"] == ORDEN[2:] and prom["autorizacion_ejecucion"] == AUT
    assert prom["delta_ap"]["S1_a050"] > 0 > prom["delta_ap"]["S2_g200"] and isinstance(prom["ap_base_semilla_42"], float)
    assert correr(ent, "promocion") == 0 and (ent.raiz / "promocion_02.json").is_file()    # nunca sobrescribe
    assert resumen(ent)["intentos_consumidos"] == 11                                       # promocion no entrena ni consume
    # fase B: provisional exige la aceptación específica
    with pytest.raises(GuardError, match="aceptar-fase-a-incompleta"):
        correr(ent, "fase_b")
    with pytest.raises(GuardError, match="aceptar-fase-a-incompleta"):
        correr(ent, "fase_b", aceptar_incompleta="CONFIRMADO:AUT-041")
    assert not (ent.raiz / "S1_a050/completa/densenet121__s43__f0").exists()
    assert correr(ent, "fase_b", aceptar_incompleta="CONFIRMADO:D-041") == 0
    r = resumen(ent)
    assert r["planificadas_por_fase"] == {"A": 30, "B": 10, "C": 0} and r["completadas"] == 20 and r["intentos_consumidos"] == 21
    assert sorted(p.name for p in (ent.raiz / "S1_a050/completa").iterdir()) == sorted(f"densenet121__s{s}__f{k}" for s in (42, 43, 44) for k in range(5))
    sb = json.loads((ent.raiz / "S1_a050/completa/densenet121__s44__f4/intento_01/sello_g4.json").read_text(encoding="utf-8"))
    assert sb["fase"] == "B" and sb["promocion"]["provisional"] is True and sb["promocion"]["aceptacion_fase_a_incompleta"] == "CONFIRMADO:D-041"
    assert not (ent.raiz / "S2_g200/completa/densenet121__s43__f0").exists()               # solo las promovidas


def test_orden_fijo_se_respeta_al_reanudar_con_otra_configuracion_posterior(ent):
    correr(ent, "humo")
    (ent.raiz / "S1_a025/completa/densenet121__s42__f0/intento_01").mkdir(parents=True)     # intento de la 3.ª sin completar la 1.ª y la 2.ª
    with pytest.raises(GuardError, match="orden fijo"):
        correr(ent, "fase_a", hasta=3)
    assert resumen(ent)["intentos_consumidos"] == 1 and not (ent.raiz / "S1_a050").exists()


def test_fallo_tecnico_y_reintento_cuentan_y_se_conservan(ent):
    correr(ent, "humo")
    estado = {"fallar": True}

    def falla(ctx):
        if estado["fallar"] and ctx["fold"] == 0:
            raise RuntimeError("fallo simulado")
    ent.kw["entrenar"] = sintetico(falla=falla)
    assert correr(ent, "fase_a", hasta=1) == 2                                             # se detiene; el intento queda conservado
    d = ent.raiz / "S1_a050/completa/densenet121__s42__f0"
    assert rc_de(ent, "S1_a050", 42, 0)["estado"] == "fallo_tecnico" and (d / "intento_01/sello_g4.json").is_file()
    r = resumen(ent)
    assert r["intentos_fallidos"] == 1 and r["completadas"] == 0 and r["intentos_consumidos"] == 2 and r["reintentos"] == 0
    estado["fallar"] = False
    assert correr(ent, "fase_a", hasta=1) == 0
    r = resumen(ent)
    assert sorted(p.name for p in d.iterdir()) == ["intento_01", "intento_02"] and r["reintentos"] == 1 and r["intentos_consumidos"] == 2 + 5 and r["completadas"] == 5
    ev0 = [e["evento"] for e in P.leer_ledger(ent.ledger) if e["id_corrida"] == "densenet121__s42__f0" and e["config_id"] == "S1_a050"]
    assert ev0 == ["planificada", "iniciada", "fallida", "reintento", "iniciada", "completada"]
    fall = next(e for e in P.leer_ledger(ent.ledger) if e["evento"] == "fallida")
    assert "fallo_tecnico" in fall["nota"] and fall["segundos"] is not None


def test_tope_duro_rechaza_sin_crear_directorio_y_sale_con_codigo_especifico(ent, capsys):
    _relleno(ent.ledger, 74)                                                               # 74 intentos ya consumidos (p. ej. con fallidos y reintentos)
    assert correr(ent, "humo") == 0                                                        # el 75.º intento
    assert resumen(ent)["intentos_consumidos"] == 75 and resumen(ent)["restante"] == 0
    with pytest.raises(P.PresupuestoAgotado):
        correr(ent, "fase_a", hasta=1)
    assert not (ent.raiz / "S1_a050").exists()                                             # rechazado SIN crear directorio
    rc = E.main(["--modo", "fase_a", "--hasta", "1", "--autorizacion-ejecucion", AUT], root=ent.root, config=ent.config, config_g2=ent.config_g2,
                precargar=lambda: None, **ent.kw)
    assert rc == P.CODIGO_SALIDA_PRESUPUESTO == 5 and "PRESUPUESTO" in capsys.readouterr().err and resumen(ent)["intentos_consumidos"] == 75
    with pytest.raises(GuardError, match="no puede superar"):                              # el tope no se amplía ni por código
        correr(ent, "humo", tope=100)


def test_presupuesto_insuficiente_para_la_fase_se_rechaza_antes_de_empezar(ent):
    _relleno(ent.ledger, 60)
    correr(ent, "humo")                                                                    # 61
    with pytest.raises(P.PresupuestoAgotado, match="exceden"):
        correr(ent, "fase_a", hasta=3)                                                     # 15 pendientes + 61 > 75: no se inicia ninguna
    assert not (ent.raiz / "S1_a050").exists() and resumen(ent)["planificadas"] == 0
    assert correr(ent, "fase_a", hasta=2) == 0                                             # 10 + 61 <= 75


def test_guarda_de_r6_ausencia_o_otra_elegida_bloquea_y_el_fixture_permite(ent):
    ruta = ent.root / ent.config["referencia_r6"]["salida"]
    assert C.verificar_base_r6(ent.root, ent.config)["estado"] == "verificada"             # fixture explícito del directorio temporal
    ruta.unlink()
    with pytest.raises(GuardError, match="no existe"):                                     # la ausencia BLOQUEA (antes era `no_disponible`)
        C.verificar_base_r6(ent.root, ent.config)
    for modo in ("humo", "fase_a", "promocion", "fase_b", "fase_c", "fase_c_confirmacion"):
        with pytest.raises(GuardError, match="no existe"):
            correr(ent, modo)
    ruta.write_text(json.dumps({"elegida": {"condicion": "B_hibrida", "arquitectura": "densenet121"}}), encoding="utf-8")
    with pytest.raises(GuardError, match="R6 eligió"):
        correr(ent, "humo")
    ruta.write_text(json.dumps({"elegida": {"condicion": "A", "arquitectura": "resnet50"}}), encoding="utf-8")
    with pytest.raises(GuardError, match="R6 eligió"):
        correr(ent, "humo")
    ruta.write_text("{", encoding="utf-8")
    with pytest.raises(GuardError, match="ilegible"):
        correr(ent, "humo")
    assert not ent.raiz.exists() and not ent.llamadas
    assert not [o for ac in E.construir_parser()._actions for o in ac.option_strings if "r6" in o.lower()]            # ninguna bandera del CLI debilita la guarda
    escribir_r6(ent.root, ent.config)
    assert correr(ent, "humo") == 0


def test_base_no_comparable_con_g4_bloquea_antes_de_cualquier_intento(ent):
    d = ent.root / ent.config_g2["diseno"]["raiz_resultados"].strip("/") / "completa/densenet121__s42__f2/intento_01/run_config.json"
    rc = json.loads(d.read_text(encoding="utf-8"))
    rc["receta"]["learning_rate"] = 0.5
    d.write_text(json.dumps(rc), encoding="utf-8")
    with pytest.raises(GuardError, match="no es comparable"):                              # base alterada: ni el humo ni la fase A inician intentos
        correr(ent, "humo")
    with pytest.raises(GuardError, match="humo|no es comparable"):
        correr(ent, "fase_a", hasta=1)
    assert not ent.llamadas and not ent.raiz.exists()


def test_corrida_previa_de_otra_estrategia_no_se_mezcla(ent):
    correr(ent, "humo")
    correr(ent, "fase_a", hasta=1)
    p = ent.raiz / "S1_a050/completa/densenet121__s42__f3/intento_01/run_config.json"
    rc = json.loads(p.read_text(encoding="utf-8"))
    rc["receta"]["perdida_entrenamiento"]["alpha_pesos_de_clase"] = 0.9
    p.write_text(json.dumps(rc), encoding="utf-8")
    with pytest.raises(GuardError, match="otra partición, receta o estrategia"):
        correr(ent, "fase_a", hasta=1)


def test_k4_rutas_con_test_y_salida_dentro_de_results_r7(ent):
    for malo in ("S1_a050/../test", "S1_a05", "x", ""):
        with pytest.raises(GuardError):
            C.raiz_corridas(ent.root, ent.config, malo, "completa")
    with pytest.raises(GuardError):
        C.raiz_corridas(ent.root, ent.config, "S1_a050", "test")
    cfg = copy.deepcopy(ent.config)
    cfg["diseno"]["raiz_resultados"] = "results/r7_v2/phash_le8_v1/test/"
    with pytest.raises(GuardError):
        C.validar_config(cfg)
    nombres = [p.name for p in Path(REPO / "src/r7_v2").glob("g4_*.py")]
    assert nombres and not any("test" == part.lower() for n in nombres for part in Path(n).parts)
    assert C.LEYENDA.startswith("partición provisional (phash_le8_v1, exploratoria, post hoc)")


# ============================================================================ integridad propia con Git real
def _git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), "-c", "user.email=t@t", "-c", "user.name=t", *args], capture_output=True, text=True, check=True).stdout.strip()


def test_integridad_g4_compatible_con_g2_arbol_limpio_y_pin_del_yaml_de_g2(tmp_path):
    from src.r4_v2 import integridad as I4
    _git(tmp_path, "init", "-q")
    (tmp_path / "src/r7_v2").mkdir(parents=True)
    archivos = {"src/r7_v2/a.py": b"x = 1\n", C.PROTOCOLO_REL: b"p\n", "g2.yaml": b"g2: 1\n"}
    for rel, b in archivos.items():
        (tmp_path / rel).write_bytes(b)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "c")
    cfg = {"integridad": {"rutas_sin_cambios": ["src/r7_v2"], "archivos_con_huella": list(archivos)},
           "referencia_g2": {"config": "g2.yaml", "sha256_contenido_blob": hashlib.sha256(b"g2: 1\n").hexdigest()}}
    ok = C.verificar_integridad_g4(tmp_path, cfg)
    assert ok["git_commit_head"] == _git(tmp_path, "rev-parse", "HEAD") and not ok["bloqueos"] and ok["protocolo_versionado"] is True
    assert set(ok["archivos"]["g2.yaml"]) >= {"git_blob_id", "sha256_contenido_blob", "sha256_bytes_archivo"}
    assert I4.exigir_integridad(ok) is ok and I4.comparar_instantaneas(ok, C.verificar_integridad_g4(tmp_path, cfg)) == []
    (tmp_path / "src/r7_v2/a.py").write_bytes(b"x = 2\n")
    sucio = C.verificar_integridad_g4(tmp_path, cfg)
    assert [b["codigo"] for b in sucio["bloqueos"]] == ["arbol_sucio"] and I4.comparar_instantaneas(ok, sucio)
    with pytest.raises(GuardError, match="arbol_sucio"):
        C.exigir_integridad(sucio)
    _git(tmp_path, "checkout", "--", "src/r7_v2/a.py")
    (tmp_path / "src/r7_v2/nuevo.py").write_bytes(b"y = 1\n")                              # archivo sin seguimiento dentro de las rutas vigiladas
    assert [b["codigo"] for b in C.verificar_integridad_g4(tmp_path, cfg)["bloqueos"]] == ["arbol_sucio"]
    (tmp_path / "src/r7_v2/nuevo.py").unlink()
    cfg["referencia_g2"]["sha256_contenido_blob"] = "0" * 64
    assert [b["codigo"] for b in C.verificar_integridad_g4(tmp_path, cfg)["bloqueos"]] == ["yaml_referencia_g2_distinto_del_fijado"]


# ============================================================================ mecanismo del agente (propuestas, límites, duplicados, alteración, conclusión)
def _evidencia(e, configuraciones):
    ruta = e.raiz / e.config["agente"]["evidencia_fase_b"]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps({"esquema": AGT.ESQUEMA_EVIDENCIA, "configuraciones": configuraciones}), encoding="utf-8")
    return ruta


def _cfg_ev(cid="S1_a050", familia="S1", cumple=True, delta=0.03, semillas=(42, 43, 44)):
    return {"config_id": cid, "familia": familia, "semillas": list(semillas), "cumple_a": cumple, "delta_ap_medio": delta}


def _propuesta(e, ruta_ref, nn, familia="S1", valor=0.35, recalcular=False, **cambios):
    carpeta = e.raiz / e.config["agente"]["carpeta_propuestas"]
    carpeta.mkdir(parents=True, exist_ok=True)
    d = {"familia": familia, "valor": valor, "justificacion": "Refina alpha entre 0.25 y 0.5, donde el resultado previo fue favorable.", "fecha": "2026-11-22",
         "referencia_resultado_previo": {"archivo": ruta_ref.relative_to(e.root).as_posix(), "sha256_bytes_archivo": C4.sha256_archivo(ruta_ref)}}
    d["sha256_propuesta"] = AGT.hash_de_propuesta(d)
    d.update(cambios)
    if recalcular:
        d["sha256_propuesta"] = AGT.hash_de_propuesta(d)
    (carpeta / f"propuesta_{nn:02d}.json").write_text(json.dumps(d), encoding="utf-8")
    return carpeta


def test_activacion_solo_con_tecnica_que_cumple_a_con_tres_semillas(ent):
    ruta = _evidencia(ent, [_cfg_ev("S1_a050", "S1", True, 0.03), _cfg_ev("S2_g200", "S2", True, 0.05), _cfg_ev("S1_a025", "S1", False, 0.09)])
    f = AGT.familia_seleccionada(AGT.leer_evidencia_fase_b(ruta))
    assert (f["config_id"], f["familia"]) == ("S2_g200", "S2")                             # mayor ΔAP medio entre las que cumplen (a)
    ruta = _evidencia(ent, [_cfg_ev("S1_a050", "S1", True, 0.03), _cfg_ev("S2_g200", "S2", True, 0.03)])
    assert AGT.familia_seleccionada(AGT.leer_evidencia_fase_b(ruta))["config_id"] == "S1_a050"           # empate: primero de la lista
    for malo in ([_cfg_ev(cumple=False)], [_cfg_ev(semillas=(42, 43))], []):
        ruta = _evidencia(ent, malo)
        with pytest.raises(GuardError, match="agente no se activa"):
            AGT.familia_seleccionada(AGT.leer_evidencia_fase_b(ruta))
    with pytest.raises(GuardError, match="no se activa"):
        AGT.leer_evidencia_fase_b(ent.raiz / "no_existe.json")
    ruta.write_text(json.dumps({"esquema": "otro"}), encoding="utf-8")
    with pytest.raises(GuardError, match="contrato"):
        AGT.leer_evidencia_fase_b(ruta)


def test_validacion_de_propuestas_familia_limites_duplicados_y_hash(ent):
    ref = _evidencia(ent, [_cfg_ev()])
    carpeta = _propuesta(ent, ref, 1)
    ok = AGT.validar_propuestas(ent.root, carpeta, "S1")
    assert [v["estrategia"]["id"] for v in ok] == ["S1_a035"] and ok[0]["estrategia"]["alpha"] == 0.35 and ok[0]["estrategia"]["gamma"] is None
    casos = [({"familia": "S2", "valor": 1.5}, "no es la seleccionada"), ({"valor": 1.2}, "fuera de"), ({"valor": -0.1}, "fuera de"), ({"valor": 0.355}, "fuera de"),
             ({"valor": 0.5}, "duplicado"), ({"valor": 1.0}, "base"), ({"valor": 0.25}, "duplicado"), ({"justificacion": "corta"}, "justificación"),
             ({"fecha": "ayer"}, "fecha"), ({"sha256_propuesta": "0" * 64}, "sha256_propuesta"), ({"gamma": 2.0}, "no permitidos")]
    for cambios, patron in casos:
        c = _propuesta(ent, ref, 1, **({"familia": "S1", "valor": 0.35} | cambios))
        with pytest.raises(GuardError, match=patron):
            AGT.validar_propuestas(ent.root, c, "S1")
    c = _propuesta(ent, ref, 1, valor=0.35, referencia_resultado_previo={"archivo": ref.relative_to(ent.root).as_posix(), "sha256_bytes_archivo": "0" * 64}, recalcular=True)
    with pytest.raises(GuardError, match="referencia_resultado_previo"):
        AGT.validar_propuestas(ent.root, c, "S1")
    _propuesta(ent, ref, 1, valor=0.35)
    _propuesta(ent, ref, 2, valor=0.35)                                                    # duplicada entre propuestas
    with pytest.raises(GuardError, match="duplicado"):
        AGT.validar_propuestas(ent.root, carpeta, "S1")
    _propuesta(ent, ref, 2, valor=0.65)
    _propuesta(ent, ref, 3, valor=0.15)
    assert len(AGT.validar_propuestas(ent.root, carpeta, "S1")) == 3
    _propuesta(ent, ref, 4, valor=0.9)                                                     # más de 3
    with pytest.raises(GuardError, match="máximo 3"):
        AGT.validar_propuestas(ent.root, carpeta, "S1")
    (carpeta / "propuesta_04.json").unlink()
    (carpeta / "propuesta_02.json").unlink()                                               # numeración no contigua
    with pytest.raises(GuardError, match="contigua"):
        AGT.validar_propuestas(ent.root, carpeta, "S1")


def test_propuesta_alterada_o_borrada_tras_su_registro_bloquea(ent):
    ref = _evidencia(ent, [_cfg_ev()])
    carpeta = _propuesta(ent, ref, 1, valor=0.35)
    v = AGT.validar_propuestas(ent.root, carpeta, "S1")
    assert AGT.registrar_propuestas(carpeta, v, head="h", autorizacion=AUT) == 1
    assert AGT.registrar_propuestas(carpeta, AGT.validar_propuestas(ent.root, carpeta, "S1"), head="h", autorizacion=AUT) == 0     # idempotente
    reg = AGT.leer_registro(carpeta)
    assert len(reg) == 1 and reg[0]["sha256_bytes_archivo"] == C4.sha256_archivo(carpeta / "propuesta_01.json") and reg[0]["config_id"] == "S1_a035"
    _propuesta(ent, ref, 1, valor=0.35, justificacion="Justificación reescrita después de registrar la propuesta original.")      # contenido válido pero distinto
    with pytest.raises(GuardError, match="cambió tras su registro"):
        AGT.validar_propuestas(ent.root, carpeta, "S1")
    _propuesta(ent, ref, 1, valor=0.35)
    (carpeta / "propuesta_01.json").unlink()
    with pytest.raises(GuardError, match="ya no están"):
        AGT.validar_propuestas(ent.root, carpeta, "S1")


def test_conclusion_sin_mejora_se_anota_y_no_es_error(ent):
    ref = _evidencia(ent, [_cfg_ev()])
    carpeta = ent.raiz / ent.config["agente"]["carpeta_propuestas"]
    carpeta.mkdir(parents=True)
    assert AGT.leer_conclusion(ent.root, carpeta) is None
    ref_json = {"archivo": ref.relative_to(ent.root).as_posix(), "sha256_bytes_archivo": C4.sha256_archivo(ref)}
    (carpeta / AGT.CONCLUSION).write_text(json.dumps({"fecha": "2026-11-23", "justificacion": "Sin región prometedora para refinar la familia elegida.",
                                                      "referencia_resultado_previo": ref_json}), encoding="utf-8")
    c = AGT.leer_conclusion(ent.root, carpeta)
    assert AGT.anotar_conclusion(carpeta, c, head="h", autorizacion=AUT) is True and AGT.anotar_conclusion(carpeta, c, head="h", autorizacion=AUT) is False
    (carpeta / AGT.CONCLUSION).write_text(json.dumps({"fecha": "2026-11-23", "justificacion": "x", "referencia_resultado_previo": ref_json}), encoding="utf-8")
    with pytest.raises(GuardError, match="justificación"):
        AGT.leer_conclusion(ent.root, carpeta)


def _preparar_fase_c(ent, valores=(0.35, 0.65), con_conv=True):
    if con_conv:                                                                           # la candidata convencional S1_a050 (semilla 42) ya evaluada en la fase A
        assert correr(ent, "humo") == 0 and correr(ent, "fase_a", hasta=1) == 0
    ref = _evidencia(ent, [_cfg_ev()])
    carpeta = None
    for n, v in enumerate(valores, start=1):
        carpeta = _propuesta(ent, ref, n, valor=v)
    return ref, carpeta


def test_fase_c_conclusion_sin_mejora_no_entrena_ni_consume(ent):
    ref, _ = _preparar_fase_c(ent, valores=(), con_conv=False)
    carpeta = ent.raiz / ent.config["agente"]["carpeta_propuestas"]
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / AGT.CONCLUSION).write_text(json.dumps({"fecha": "2026-11-23", "justificacion": "Sin región prometedora para refinar la familia elegida.",
                                                      "referencia_resultado_previo": {"archivo": ref.relative_to(ent.root).as_posix(),
                                                                                      "sha256_bytes_archivo": C4.sha256_archivo(ref)}}), encoding="utf-8")
    assert correr(ent, "fase_c") == 0 and not ent.llamadas and not ent.ledger.exists()
    assert [r["evento"] for r in AGT.leer_registro(carpeta)] == ["conclusion_sin_mejora"]
    _propuesta(ent, ref, 1, valor=0.35)
    with pytest.raises(GuardError, match="a la vez"):
        correr(ent, "fase_c")


def test_fase_c_sin_activacion_o_sin_propuestas_no_corre(ent):
    with pytest.raises(GuardError, match="no se activa"):
        correr(ent, "fase_c")                                                              # sin evidencia de la fase B
    _evidencia(ent, [_cfg_ev(cumple=False)])
    with pytest.raises(GuardError, match="no se activa"):
        correr(ent, "fase_c")
    _evidencia(ent, [_cfg_ev()])
    with pytest.raises(GuardError, match="ni conclusion"):
        correr(ent, "fase_c")
    assert not ent.ledger.exists() and not (ent.raiz / "S1_a035").exists()


def test_fase_c_y_confirmacion_flujo_presupuesto_y_alteracion(ent):
    ref, carpeta = _preparar_fase_c(ent)
    previo = ent.ledger.read_bytes()                                                       # humo + fase A de la candidata convencional: 6 intentos
    _relleno(ent.ledger, 60)                                                               # 66 consumidos: restan 9; 5*2 + 10 reservados no caben
    with pytest.raises(P.PresupuestoAgotado, match="exceden"):
        correr(ent, "fase_c")
    assert not (ent.raiz / "S1_a035").exists() and AGT.leer_registro(carpeta) == []        # rechazado antes de registrar o crear nada
    ent.ledger.write_bytes(previo)
    assert correr(ent, "fase_c") == 0                                                      # 2 propuestas x semilla 42 x 5 folds = 10
    r = resumen(ent)
    assert r["planificadas_por_fase"]["C"] == 10 and r["completadas"] == 15 and r["intentos_consumidos"] == 16
    assert sorted(p.name for p in ent.raiz.glob("S1_a0[36]*")) == ["S1_a035", "S1_a065"] and all("__s42__" in p.name for p in ent.raiz.glob("S1_a0[36]*/completa/*"))
    s = json.loads((ent.raiz / "S1_a035/completa/densenet121__s42__f0/intento_01/sello_g4.json").read_text(encoding="utf-8"))
    assert s["fase"] == "C" and s["id_propuesta"] == "propuesta_01.json" and s["alpha"] == 0.35 and s["propuesta"]["sha256_bytes_archivo"] == C4.sha256_archivo(carpeta / "propuesta_01.json")
    assert [x["config_id"] for x in AGT.leer_registro(carpeta)] == ["S1_a035", "S1_a065"]
    # confirmación: promoción exploratoria (ΔAP semilla 42 > 0 frente a la base Y a la candidata convencional) con semillas 43 y 44
    assert correr(ent, "fase_c_confirmacion") == 0
    r = resumen(ent)
    assert r["planificadas_por_fase"]["C"] == 20 and r["completadas"] == 25 and sorted(p.name for p in (ent.raiz / "S1_a035/completa").iterdir()) == \
        sorted(f"densenet121__s{x}__f{k}" for x in (42, 43, 44) for k in range(5))
    assert not (ent.raiz / "S1_a065/completa/densenet121__s43__f0").exists()
    # un archivo alterado tras su registro bloquea
    _propuesta(ent, ref, 2, valor=0.65, justificacion="Justificación reescrita después de registrar la propuesta original.")
    with pytest.raises(GuardError, match="cambió tras su registro"):
        correr(ent, "fase_c")
    with pytest.raises(GuardError, match="cambió tras su registro"):
        correr(ent, "fase_c_confirmacion")
    assert resumen(ent)["intentos_consumidos"] == 26


def test_confirmacion_exige_fase_c_completa_y_delta_positivo(ent):
    _, carpeta = _preparar_fase_c(ent, valores=(0.15,))                                    # alpha = 0.15: sin efecto sintético (ΔAP = 0)
    assert correr(ent, "fase_c") == 0
    n = resumen(ent)["intentos_consumidos"]
    assert correr(ent, "fase_c_confirmacion") == 0 and resumen(ent)["intentos_consumidos"] == n and not (ent.raiz / "S1_a015/completa/densenet121__s43__f0").exists()


# ============================================================================ AJUSTE 2: promoción EXPLORATORIA del agente a la confirmación (43 y 44)
def test_elegir_confirmacion_pura_ambas_positivas_varias_empate_y_ninguna():
    ok = C.elegir_confirmacion({"a": 0.05, "b": 0.01}, {"a": -0.01, "b": 0.02}, ["a", "b"])                 # a: > 0 frente a la base pero <= 0 frente a la candidata
    assert ok["autorizadas"] == ["b"] and ok["elegida"] == "b" and "exploratoria" in ok["naturaleza"] and ok["regla"] == C.REGLA_CONFIRMACION
    nada = C.elegir_confirmacion({"a": 0.05, "b": -0.01}, {"a": 0.0, "b": 0.3}, ["a", "b"])                 # 0 no es positivo; b no supera a la base
    assert nada["elegida"] is None and nada["autorizadas"] == []
    varias = C.elegir_confirmacion({"a": 0.02, "b": 0.02, "c": 0.02}, {"a": 0.01, "b": 0.05, "c": 0.03}, ["a", "b", "c"])
    assert varias["autorizadas"] == ["b", "c", "a"] and varias["elegida"] == "b" and varias["desempate_por_numero_de_propuesta"] is False
    empate = C.elegir_confirmacion({"a": 0.02, "b": 0.02, "c": 0.02}, {"a": 0.04, "b": 0.04, "c": 0.01}, ["a", "b", "c"])
    assert empate["elegida"] == "a" and empate["desempate_por_numero_de_propuesta"] is True            # empate exacto: menor número de propuesta
    assert C.elegir_confirmacion({"a": 0.02, "b": 0.02}, {"a": 0.04, "b": 0.04}, ["b", "a"])["elegida"] == "b"


def _ap_vs(ent, cid, semilla=42):
    """(delta AP frente a la base, delta AP frente a la candidata convencional S1_a050) de una propuesta, calculados de forma independiente."""
    est = C.estrategia_de("S1", int(cid[4:]) / 100)
    conv = C.estrategias_fase_a(ent.config)[0]
    ap = C.ap_estrategia(ent.root, ent.config, ent.config_g2, ent.particion, est, semilla)
    return (ap - C.ap_base(ent.root, ent.config, ent.config_g2, ent.particion, semilla),
            ap - C.ap_estrategia(ent.root, ent.config, ent.config_g2, ent.particion, conv, semilla))


def test_confirmacion_no_se_autoriza_si_no_supera_a_la_candidata_convencional(ent):
    _preparar_fase_c(ent, valores=(0.65,))
    assert correr(ent, "fase_c") == 0
    d_base, d_conv = _ap_vs(ent, "S1_a065")
    assert d_base > 0 and d_conv <= 0                                                      # mejora a la base pero no a la candidata que refina
    n = resumen(ent)["intentos_consumidos"]
    assert correr(ent, "fase_c_confirmacion") == 0 and resumen(ent)["intentos_consumidos"] == n
    assert not (ent.raiz / "S1_a065/completa/densenet121__s43__f0").exists() and resumen(ent)["planificadas_por_fase"]["C"] == 5


def test_confirmacion_varias_cumplen_se_confirma_una_sola_y_presupuesto_insuficiente_rechaza(ent):
    _preparar_fase_c(ent, valores=(0.35, 0.8))
    assert correr(ent, "fase_c") == 0
    dc = {c: _ap_vs(ent, c) for c in ("S1_a035", "S1_a080")}
    assert all(b > 0 and v > 0 for b, v in dc.values())                                    # ambas cumplen la condición de promoción
    esperada = max(dc, key=lambda c: (dc[c][1], c == "S1_a035"))                           # mayor delta frente a la candidata; empate -> propuesta 01
    previo = ent.ledger.read_bytes()
    _relleno(ent.ledger, 50)                                                               # 66 consumidos: restan 9 < 10, la confirmación no cabe
    with pytest.raises(P.PresupuestoAgotado, match="exceden"):
        correr(ent, "fase_c_confirmacion")
    assert not list(ent.raiz.glob("S1_a0[38]*/completa/densenet121__s43__f*"))               # rechazado SIN crear nada
    ent.ledger.write_bytes(previo)
    assert correr(ent, "fase_c_confirmacion") == 0
    confirmadas = sorted(p.parent.parent.name for p in ent.raiz.glob("S1_a0[38]*/completa/densenet121__s43__f0"))
    assert confirmadas == [esperada] and resumen(ent)["planificadas_por_fase"]["C"] == 20                    # UNA sola (10 intentos)
    s = json.loads((ent.raiz / esperada / "completa/densenet121__s43__f0/intento_01/sello_g4.json").read_text(encoding="utf-8"))
    pe = s["promocion_exploratoria"]
    assert pe["elegida"] == esperada and pe["candidata_convencional"] == "S1_a050" and "exploratoria" in pe["naturaleza"]
    assert pe["delta_ap_semilla_42_vs_base"] > 0 and pe["delta_ap_semilla_42_vs_candidata_convencional"] > 0


def test_confirmacion_exige_la_candidata_convencional_evaluada(ent):
    _preparar_fase_c(ent, valores=(0.35,), con_conv=False)
    assert correr(ent, "fase_c") == 0
    with pytest.raises(GuardError, match="candidata convencional"):
        correr(ent, "fase_c_confirmacion")
