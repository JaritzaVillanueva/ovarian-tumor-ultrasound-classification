"""Pruebas sintéticas (CPU) de `analisis_categorias` (R7): n por categoría y fold, mediana/IQR, proporción sobre el umbral de transferencia por fold, AUC contra la 7 (IC solo con
n_c >= 10), foco en la categoría 6 sin p-valores, bloqueo de categorías no permitidas, etiquetas sin alterar, q_barra verificado por SHA-256 y guardas. No abren test ni datos reales.
Comando: py -3.13 -B -m pytest tests/test_r7_v2_categorias.py -q -p no:cacheprovider
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("yaml")
pytest.importorskip("PIL")

from src.r4_v2 import calibracion as CAL  # noqa: E402
from src.r4_v2 import common as C4  # noqa: E402
from src.r4_v2 import metricas as M  # noqa: E402
from src.r4_v2.common import GuardError  # noqa: E402
from src.r7_v2 import analisis_categorias as AC  # noqa: E402
from src.r7_v2 import g4_reporte as R  # noqa: E402
from src.r7_v2 import umbral_sistema as U  # noqa: E402
from tests.test_r7_v2_umbral import AUT, N_B, REPO, armar, escribir_r6, leer_csv, verificar_u  # noqa: E402,F401
from tests.test_r7_v2_umbral import _git_aislado  # noqa: E402,F401  (fixture autouse)
from tests.test_r7_v2_umbral import correr as correr_umbral  # noqa: E402


def correr(e, destino=None, **kw):
    args = dict(autorizacion=AUT, verificar=verificar_u(), n_remuestreos=N_B) | kw
    return AC.ejecutar(e.root, e.config, e.config_g2, destino=destino, **args)


@pytest.fixture(scope="module")
def banco(tmp_path_factory):
    e = armar(tmp_path_factory.mktemp("banco_cat"), por_fold=40, pos_por_fold=8)             # 200 imágenes, 40 positivas, ≈ 27 por categoría no maligna
    e.umbral = correr_umbral(e)
    e.cat = correr(e)
    e.filas = e.particion.filas
    e.q = e.umbral["resultados"]["q_barra"]
    return e


def copia_origen(e, tmp_path, nombre):
    destino = tmp_path / nombre
    shutil.copytree(e.umbral["dir"], destino)
    return destino


# ============================================================================ n, mediana/IQR, proporciones con el umbral de transferencia por fold
def test_n_por_categoria_y_por_fold(banco):
    cats = {c["categoria"]: c for c in banco.cat["resultados"]["categorias"]}
    assert list(cats) == [0, 1, 2, 3, 4, 6, 7] and sum(c["n"] for c in cats.values()) == len(banco.filas) == 200
    for c, v in cats.items():
        manual = [r for r in banco.filas if r["original_class"] == c]
        assert v["n"] == len(manual) and v["n_por_fold"] == {k: sum(1 for r in manual if r["fold"] == k) for k in range(5)} and sum(v["n_por_fold"].values()) == v["n"]
    assert cats[7]["n"] == 40 and all(cats[c]["n"] >= 10 for c in (0, 1, 2, 3, 4, 6))
    enc, filas = leer_csv(banco.cat["dir"] / "categorias.csv")
    assert [int(r["categoria"]) for r in filas] == [0, 1, 2, 3, 4, 6, 7] and filas[-1]["n_por_fold"] == " ".join(f"f{k}:8" for k in range(5))


def test_mediana_e_iqr_de_q_barra(banco):
    for c in banco.cat["resultados"]["categorias"]:
        q = np.array([banco.q[i] for i, r in enumerate(banco.filas) if r["original_class"] == c["categoria"]])
        q1, med, q3 = np.percentile(q, [25, 50, 75])
        assert (c["q1"], c["mediana"], c["q3"]) == (float(q1), float(med), float(q3)) and c["iqr"] == pytest.approx(float(q3 - q1))
    _, filas = leer_csv(banco.cat["dir"] / "categorias.csv")
    assert float(filas[-1]["mediana_q_barra"]) == banco.cat["resultados"]["categorias"][-1]["mediana"]


def test_proporcion_sobre_el_umbral_de_transferencia_de_cada_fold(banco):
    y = [r["label_v2"] for r in banco.filas]
    fold = [r["fold"] for r in banco.filas]
    ref = CAL.comprobar_transferencia(y, banco.q.tolist(), fold, "0.90")
    umbrales = {k: float(v["umbral"]) for k, v in ref["por_fold"].items()}
    assert banco.cat["resultados"]["umbral_transferencia_por_fold"] == umbrales               # calculado con los otros cuatro folds
    cats = {c["categoria"]: c for c in banco.cat["resultados"]["categorias"]}
    for c, v in cats.items():
        x = sum(1 for i, r in enumerate(banco.filas) if r["original_class"] == c and banco.q[i] >= umbrales[r["fold"]])
        assert v["sobre_umbral"] == x and v["proporcion"] == pytest.approx(x / v["n"])
    assert cats[7]["lectura"] == "sensibilidad (TP/n)" and cats[7]["sobre_umbral"] == ref["tp"]                     # para la 7 es la sensibilidad
    assert all(cats[c]["lectura"] == "tasa de falsos positivos (FP/n)" for c in (0, 1, 2, 3, 4, 6))
    assert sum(cats[c]["sobre_umbral"] for c in (0, 1, 2, 3, 4, 6)) == sum(v["fp"] for v in ref["por_fold"].values())
    _, filas = leer_csv(banco.cat["dir"] / "categorias.csv")
    assert filas[-1]["sobre_umbral_x_sobre_n"] == f"{cats[7]['sobre_umbral']}/40" and filas[0]["sobre_umbral_x_sobre_n"].endswith(f"/{cats[0]['n']}")


# ============================================================================ AUC contra la 7 (IC solo con n_c >= 10)
def test_auc_vs_7_con_ic_si_n_es_suficiente(banco):
    aucs = {a["categoria"]: a for a in banco.cat["resultados"]["auc_vs_7"]}
    assert list(aucs) == [0, 1, 2, 3, 4, 6]
    for c, a in aucs.items():
        idx = [i for i, r in enumerate(banco.filas) if r["original_class"] in (c, 7)]
        y_sub = np.array([1 if banco.filas[i]["original_class"] == 7 else 0 for i in idx])
        assert a["n_categoria"] >= 10 and a["auc"] == pytest.approx(M.metricas(y_sub, banco.q[idx])["roc_auc"]) and a["ic"]["n_remuestreos_validos"] > 0
        assert a["ic"]["ic_inf"] <= a["ic"]["ic_sup"] and a["estado"].startswith("con IC95 %")
    enc, filas = leer_csv(banco.cat["dir"] / "auc_vs_7.csv")
    assert all(r["ic95_inf"] and r["ic95_sup"] for r in filas) and "IC solo si n_c >= 10" in enc and "mismos b de G2" in enc


def test_auc_vs_7_sin_ic_si_n_menor_que_10():
    y = np.array([1] * 20 + [0] * 4 + [0] * 12)
    orig = np.array([7] * 20 + [6] * 4 + [3] * 12)
    q = np.linspace(0, 1, y.size)
    gpi = np.arange(y.size)
    conteos = np.ones((5, y.size), dtype=np.uint16)
    a6 = AC.auc_vs_7(orig, q, conteos, gpi, 6)
    assert a6["estado"] == "n insuficiente" and a6["ic"] is None and a6["auc"] is None and a6["n_categoria"] == 4
    a3 = AC.auc_vs_7(orig, q, conteos, gpi, 3)
    assert a3["ic"] is not None and a3["auc"] is not None and a3["n_categoria"] == 12
    assert AC.auc_vs_7(orig, q, conteos, gpi, 3, n_min=13)["ic"] is None                    # justo bajo el mínimo
    assert AC.auc_vs_7(orig, q, conteos, gpi, 3, n_min=12)["ic"] is not None                # n_c == mínimo: se calcula


def test_auc_usa_los_mismos_b_de_g2_ponderando_por_grupo(banco):
    y = np.array([r["label_v2"] for r in banco.filas])
    gpi, conteos, _, _ = U.indices_b(banco.root, banco.config, banco.config_g2, [r["group_id"] for r in banco.filas], y, N_B)
    orig = np.array([r["original_class"] for r in banco.filas])
    a = AC.auc_vs_7(orig, banco.q, conteos, gpi, 2)
    idx = np.flatnonzero((orig == 2) | (orig == 7))
    prep = M.preparar((orig[idx] == 7).astype(np.int64), banco.q[idx])
    assert a["ic"] == R.ic(M.calcular(prep, conteos[:, gpi[idx]])["roc_auc"])
    assert banco.cat["resultados"]["auc_vs_7"][2]["ic"] == a["ic"]
    res = json.loads((banco.cat["dir"] / "resumen_categorias.json").read_text(encoding="utf-8"))
    assert res["n_remuestreos"] == N_B and res["n_minimo_auc"] == 10


# ============================================================================ foco categoría 6, sin p-valores
def test_foco_categoria_6_sin_p_valores(banco):
    f = banco.cat["resultados"]["foco_categoria_6"]
    cats = {c["categoria"]: c for c in banco.cat["resultados"]["categorias"]}
    resto = sum(cats[c]["n"] for c in (0, 1, 2, 3, 4))
    fp_resto = sum(cats[c]["sobre_umbral"] for c in (0, 1, 2, 3, 4))
    assert f["n_6"] == cats[6]["n"] and f["fp_6"] == cats[6]["sobre_umbral"] and f["fp_6_sobre_n_6"] == f"{f['fp_6']}/{f['n_6']}"
    assert f["n_resto_no_malignas"] == resto and f["fp_resto"] == fp_resto and f["proporcion_resto"] == pytest.approx(fp_resto / resto)
    assert f["diferencia_de_proporciones_6_menos_resto"] == pytest.approx(f["fp_6"] / f["n_6"] - fp_resto / resto)
    assert f["p_valores"] == "no se calculan" and f["reevaluada_como_positiva"] is False
    txt = (banco.cat["dir"] / "foco_categoria_6.json").read_text(encoding="utf-8")
    assert "p_valor" in txt and not any(w in txt.lower() for w in ("p-value", "pvalue", "p_value", "significativ", "fisher"))
    assert json.loads(txt)["foco_categoria_6"]["n_6"] == f["n_6"]


# ============================================================================ categorías no permitidas, etiquetas sin alterar
def test_categoria_5_o_fuera_de_rango_bloquea_y_label_incoherente_tambien():
    ok = [{"image_id": str(i), "original_class": c, "label_v2": 1 if c == 7 else 0} for i, c in enumerate((0, 1, 2, 3, 4, 6, 7))]
    AC.exigir_categorias(ok)
    for malo, patron in (({"image_id": "x", "original_class": 5, "label_v2": 0}, "categoría 5"), ({"image_id": "x", "original_class": 8, "label_v2": 0}, "fuera de"),
                         ({"image_id": "x", "original_class": 7, "label_v2": 0}, "label_v2"), ({"image_id": "x", "original_class": 3, "label_v2": 1}, "label_v2")):
        with pytest.raises(GuardError, match=patron):
            AC.exigir_categorias(ok + [malo])


def test_etiquetas_sin_alterar_y_banderas_de_alcance(banco):
    antes = copy.deepcopy(banco.filas)
    r = json.loads((banco.cat["dir"] / "resumen_categorias.json").read_text(encoding="utf-8"))
    assert r["etiquetas_sin_alterar"]["iguales"] is True and r["etiquetas_sin_alterar"]["sha256_antes"] == r["etiquetas_sin_alterar"]["sha256_despues"] == AC.digest_etiquetas(banco.filas)
    assert banco.filas == antes and AC.digest_etiquetas(antes) == AC.digest_etiquetas(banco.filas)
    assert AC.digest_etiquetas([{**banco.filas[0], "label_v2": 1 - banco.filas[0]["label_v2"]}] + banco.filas[1:]) != r["etiquetas_sin_alterar"]["sha256_antes"]
    assert r["categoria_6_reevaluada_como_positiva"] is False and r["r7_3_declarado_cumplido"] is False and r["viabilidad_clinica"] == "no determinada"
    texto = " ".join(r["leyendas"])
    for frase in ("DESCRIPTIVO", "no altera etiquetas, entrenamiento ni selección", "NO se reevalúa como positiva", "R7.3 no se declara cumplido", "sin p-valores",
                  "sin independencia por paciente", "no se afirma independencia absoluta del test histórico", "no es el umbral de R8"):
        assert frase in texto, frase
    for p in banco.cat["dir"].glob("*.csv"):
        enc = p.read_text(encoding="utf-8").splitlines()[0]
        assert enc.startswith("#") and U.LEYENDA in enc and "R7.3 no se declara cumplido" in enc and "sin uso de test" in enc


# ============================================================================ q_barra verificado por SHA-256 y coherencia con la partición
def test_q_barra_verificado_por_sha256_contra_trazabilidad(banco, tmp_path):
    ok = copia_origen(banco, tmp_path, "origen_ok")
    assert correr(banco, destino=banco.root / "results/r7_v2/cat_ok", origen=ok)["reporte_aceptado"] is True
    res = json.loads((banco.cat["dir"] / "resumen_categorias.json").read_text(encoding="utf-8"))
    assert res["entrada"]["q_barra_oof.csv"] == C4.sha256_archivo(banco.umbral["dir"] / "q_barra_oof.csv") and res["entrada"]["verificado_contra"].startswith("trazabilidad.json")
    alt = copia_origen(banco, tmp_path, "origen_alterado")
    q = alt / "q_barra_oof.csv"
    q.write_bytes(q.read_bytes().replace(b"0.", b"0.1", 1))
    with pytest.raises(GuardError, match="no coincide con el SHA-256"):
        correr(banco, destino=banco.root / "results/r7_v2/cat_alt", origen=alt)
    sin = copia_origen(banco, tmp_path, "origen_sin_traz")
    (sin / "trazabilidad.json").unlink()
    with pytest.raises(GuardError, match="falta q_barra_oof.csv o trazabilidad.json"):
        correr(banco, destino=banco.root / "results/r7_v2/cat_sin", origen=sin)
    otra = copia_origen(banco, tmp_path, "origen_otra_particion")
    t = json.loads((otra / "trazabilidad.json").read_text(encoding="utf-8"))
    t["sha256_bytes_archivo_particion"] = "0" * 64
    (otra / "trazabilidad.json").write_text(json.dumps(t), encoding="utf-8")
    with pytest.raises(GuardError, match="otra partición"):
        correr(banco, destino=banco.root / "results/r7_v2/cat_part", origen=otra)
    desfasada = copia_origen(banco, tmp_path, "origen_fold")                                # SHA coherente pero el contenido no corresponde a la partición
    lineas = (desfasada / "q_barra_oof.csv").read_text(encoding="utf-8").splitlines()
    fila = lineas[2].split(",")
    fila[1] = str((int(fila[1]) + 1) % 5)
    lineas[2] = ",".join(fila)
    (desfasada / "q_barra_oof.csv").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    t = json.loads((desfasada / "trazabilidad.json").read_text(encoding="utf-8"))
    t["sha256_bytes_archivo"]["q_barra_oof.csv"] = C4.sha256_archivo(desfasada / "q_barra_oof.csv")
    (desfasada / "trazabilidad.json").write_text(json.dumps(t), encoding="utf-8")
    with pytest.raises(GuardError, match="no corresponde a la partición"):
        correr(banco, destino=banco.root / "results/r7_v2/cat_fold", origen=desfasada)
    assert not [p for p in (banco.root / "results/r7_v2").glob("cat_*") if p.name != "cat_ok"]


# ============================================================================ guardas
def test_autorizacion_r6_k4_y_no_sobrescribe(banco, tmp_path):
    for valor in ("", None, "CONFIRMADO", "CONFIRMADO:D-48", "confirmado:D-048", "CONFIRMADO:AUT-048", "PENDIENTE", 48):
        with pytest.raises(GuardError, match="autorizacion"):
            correr(banco, destino=banco.root / "results/r7_v2/sin_aut", autorizacion=valor)
    assert not (banco.root / "results/r7_v2/sin_aut").exists()
    vacio = tmp_path / "vacio"
    vacio.mkdir()
    assert AC.main([], root=vacio) == U.CODIGO_SALIDA_GUARDA == 3 and list(vacio.iterdir()) == []
    ruta = banco.root / banco.config["referencia_r6"]["salida"]
    original = ruta.read_bytes()
    try:
        ruta.unlink()
        with pytest.raises(GuardError, match="no existe"):
            correr(banco, destino=banco.root / "results/r7_v2/sin_r6")
        escribir_r6(banco.root, banco.config, {"condicion": "A", "arquitectura": "resnet50"})
        with pytest.raises(GuardError, match="R6 eligió"):
            correr(banco, destino=banco.root / "results/r7_v2/otra_r6")
    finally:
        ruta.write_bytes(original)
    for malo in (banco.root / "results/r7_v2/test/x", banco.root / "results/r4_v2/x", tmp_path / "fuera"):
        with pytest.raises(GuardError):
            correr(banco, destino=malo)
    with pytest.raises(GuardError, match="ya existe"):                                      # nunca sobrescribe
        correr(banco)
    assert U.SALIDA_CATEGORIAS_REL.endswith("analisis_categorias_v1") and (banco.root / U.SALIDA_CATEGORIAS_REL).exists()
    falta = tmp_path / "sin_origen"
    with pytest.raises(GuardError, match="falta q_barra_oof.csv"):
        correr(banco, destino=banco.root / "results/r7_v2/c_sin_origen", origen=falta)


def test_la_integridad_final_distinta_marca_la_salida_como_no_aceptada(banco, tmp_path):
    n = {"i": 0}

    def cambia():
        n["i"] += 1
        r = verificar_u()()
        if n["i"] > 1:
            r["git_commit_head"] = "z" * 40
        return r
    r = correr(banco, destino=banco.root / "results/r7_v2/c_integr", verificar=cambia)
    assert r["reporte_aceptado"] is False and json.loads((r["dir"] / "resumen_categorias.json").read_text(encoding="utf-8"))["invalida_integridad"] is True
    n["i"] = 0
    assert AC.main(["--autorizacion", AUT], root=banco.root, config=banco.config, config_g2=banco.config_g2, verificar=cambia, n_remuestreos=N_B,
                   destino=banco.root / "results/r7_v2/c_integr2") == U.CODIGO_SALIDA_INTEGRIDAD_INVALIDA == 4


def test_determinismo_y_sin_afirmaciones_de_utilidad(banco, tmp_path):
    a = correr(banco, destino=banco.root / "results/r7_v2/det_a")["dir"]
    b = correr(banco, destino=banco.root / "results/r7_v2/det_b")["dir"]
    for p in a.iterdir():
        if p.suffix == ".csv":
            assert C4.sha256_archivo(p) == C4.sha256_archivo(b / p.name)
        else:
            ja, jb = json.loads(p.read_text(encoding="utf-8")), json.loads((b / p.name).read_text(encoding="utf-8"))
            ja.pop("fecha", None), jb.pop("fecha", None)
            assert ja == jb
    for p in a.iterdir():
        t = p.read_text(encoding="utf-8")
        assert not [x for x in ("clínicamente útil", "apto para", "sistema validado", "umbral final", "mejora el AP") if x in t]
