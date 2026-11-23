"""R9 (D-074): pruebas SINTÉTICAS del instrumento diagnóstico `src.gui.diagnostico_numerico` (selección de miembros e imágenes, plan sin inferencia, etapas ab y c, reglas de interpretación,
ausencia de etiquetas, salidas nuevas, compuertas del CLI). Modelos de prueba en CPU; nunca checkpoints reales ni test. La ejecución real (Falcon) es otra cosa: no se hace aquí.
Comando: py -3.13 -B -m pytest tests/gui -q -p no:cacheprovider
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")
pytest.importorskip("yaml")
from PIL import Image  # noqa: E402

from src.r4_v2.common import GuardError  # noqa: E402
from src.gui import diagnostico_numerico as D  # noqa: E402

SEMILLAS, FOLDS, N_VAL = (42, 43, 44), range(5), 64


def filas_miembros():
    return [{"id_corrida": f"densenet121__s{s}__f{k}", "semilla": str(s), "fold": str(k), "intento": "intento_01"} for s in SEMILLAS for k in FOLDS]


def pliegue(fold):
    """64 imágenes de un pliegue sintético, en un orden fijo no ordenado por id; con una columna `label_v2` que el instrumento NO debe usar ni mostrar."""
    ids = [str((i * 37 + fold * 11) % 997 + 1) for i in range(N_VAL)]
    return [{"image_id": i, "image_name": f"{i}.JPG", "fold": fold, "label_v2": (j % 2), "group_id": f"g{i}"} for j, i in enumerate(ids)]


class ModeloLote(torch.nn.Module):
    """Modelo de prueba cuya salida depende del tamaño del lote (para simular el efecto que se quiere detectar) y, por lo demás, es independiente entre filas."""

    def __init__(self, semilla, con_ruido=True):
        super().__init__()
        g = torch.Generator().manual_seed(semilla)
        self.w = torch.randn(3 * 8 * 8, 2, generator=g) * 0.05
        self.con_ruido = con_ruido

    def forward(self, x):
        logits = x.flatten(1) @ self.w
        if self.con_ruido:
            logits = logits + torch.tensor([0.0, 1e-3 * math.log1p(x.shape[0])])
        return logits


def imagen_pil(fila):
    rng = np.random.RandomState(int(fila["image_id"]))
    return Image.fromarray(rng.randint(0, 255, (8, 8, 3), dtype=np.uint8))


def tensor3(pil):
    return torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()


def probs_modelo(modelo, tensor_lote):
    modelo.eval()
    with torch.no_grad():
        return torch.softmax(modelo(tensor_lote), dim=1)[:, 1].numpy().astype(np.float32)


def contexto(*, con_ruido=True, tensor_g2=None, llamadas=None):
    ms = filas_miembros()
    modelos = {m["id_corrida"]: ModeloLote(int(m["semilla"]) * 10 + int(m["fold"]), con_ruido) for m in ms}

    def prob_g2(modelo, filas, tam, semilla):
        out = []
        for s in range(0, len(filas), tam):
            out.append(probs_modelo(modelo, torch.stack([tensor3(imagen_pil(f)) for f in filas[s:s + tam]])))
        return np.concatenate(out).astype(np.float32)

    oof = {}
    for m in ms:                                                                                      # OOF = disposición de G2: lotes de 32 en el orden del pliegue
        val = pliegue(int(m["fold"]))
        oof[m["id_corrida"]] = dict(zip([str(f["image_id"]) for f in val], prob_g2(modelos[m["id_corrida"]], val, D.LOTE_G2, 0)))

    def cargar(m):
        if llamadas is not None:
            llamadas.append(m["id_corrida"])
        return modelos[m["id_corrida"]]
    return D.Contexto(filas_miembros=ms, val_de_fold=pliegue, cargar_modelo=cargar, imagen=imagen_pil, tensor_gui=lambda pil: tensor3(pil).unsqueeze(0),
                      tensor_g2=tensor_g2 or (lambda f: tensor3(imagen_pil(f))), prob_gui=lambda modelo, t: float(probs_modelo(modelo, t)[0]), prob_g2=prob_g2,
                      prob_tensores=lambda modelo, ts: probs_modelo(modelo, torch.cat(ts, dim=0)), oof=lambda m, iid: float(oof[m["id_corrida"]][str(iid)]),
                      entorno=lambda: {"dispositivo": "cpu", "cudnn_allow_tf32": None})


# ============================================================================ selección determinista
def test_miembros_prioritarios_son_los_cinco_de_d072_y_un_control():
    ms = D.miembros_prioritarios(filas_miembros())
    assert [m["id_corrida"] for m in ms] == ["densenet121__s42__f4", "densenet121__s43__f0", "densenet121__s43__f4", "densenet121__s44__f1", "densenet121__s44__f4", "densenet121__s42__f0"]
    assert set(D.EXCEDIDOS) == set(D.REFERENCIA_D072) and len(D.EXCEDIDOS) == 5


def test_miembro_de_d072_ausente_detiene():
    with pytest.raises(GuardError, match="ausentes"):
        D.miembros_prioritarios([m for m in filas_miembros() if m["id_corrida"] != "densenet121__s43__f4"])


def test_imagenes_objetivo_ordenan_por_texto_no_por_numero():
    val = [{"image_id": i} for i in ("2", "10", "1", "9", "100")]
    assert [f["image_id"] for f in D.imagenes_objetivo(val)] == ["1", "10"]                           # igual que D-072 (`str(image_id)`)


def test_lotes_de_g2_y_ciclico():
    val = [{"image_id": str(i)} for i in range(70)]
    lote, s = D.lote_de_g2(val, 33)
    assert s == 32 and len(lote) == 32 and lote[0]["image_id"] == "32" and lote[1]["image_id"] == "33"
    lote2, s2 = D.lote_de_g2(val, 69)
    assert s2 == 64 and len(lote2) == 6                                                                # último lote, más pequeño
    c = D.lote_ciclico(val, 68, 5)
    assert [f["image_id"] for f in c] == ["68", "69", "0", "1", "2"] and D.lote_ciclico(val, 5, 1) == [val[5]]
    assert D.posicion(val, {"image_id": "41"}) == 41


# ============================================================================ plan: sin modelos ni inferencia, sin etiquetas
def test_plan_no_carga_modelos_ni_usa_etiquetas():
    llamadas = []
    p = D.plan(contexto(llamadas=llamadas))
    assert llamadas == [] and p["inferencia"] is False and p["etiquetas_usadas"] is False and p["metricas_calculadas"] is False
    assert len(p["miembros"]) == 6 and all(len(m["imagenes"]) == 2 for m in p["miembros"]) and [m["rol"] for m in p["miembros"]].count("control") == 1
    assert {m["id_corrida"]: m["referencia_D072"] for m in p["miembros"] if m["referencia_D072"]} == D.REFERENCIA_D072
    texto = json.dumps(p).lower()
    assert "label" not in texto and "etiqueta_v2" not in texto and "y_true" not in texto and "group_id" not in texto


# ============================================================================ etapa ab
def test_etapa_ab_con_modelo_que_depende_del_lote():
    r = D.etapa_ab(contexto())
    res = r["resumen"]
    assert res["A_repetible_en_todas"] is True and res["B1_tensores_iguales_en_todas"] is True and res["B2_dif_maxima_A_B"] <= 1e-7
    assert res["B2_dif_maxima_B_oof"] > 1e-5                                                         # la ruta de G2 con lote 1 también difiere del OOF (lote 32)
    assert any("propia de la condición «lote 1»" in x for x in r["interpretacion"])
    assert all(m["reproduce_D072_exactamente"] is False for m in r["miembros"][:5]) and r["miembros"][5]["reproduce_D072_exactamente"] is None
    assert all(len(i["A_p"]) == D.REPETICIONES for m in r["miembros"] for i in m["imagenes"])


def test_etapa_ab_sin_dependencia_del_lote_no_diferencia():
    res = D.etapa_ab(contexto(con_ruido=False))["resumen"]
    assert res["A_dif_maxima_global"] <= 1e-6 and res["B2_dif_maxima_A_B"] <= 1e-7


def test_etapa_ab_detecta_una_diferencia_de_preprocesamiento():
    r = D.etapa_ab(contexto(tensor_g2=lambda f: tensor3(imagen_pil(f)) + 1e-3))
    assert r["resumen"]["B1_tensores_iguales_en_todas"] is False
    assert any(x.startswith("B1: HAY diferencias") for x in r["interpretacion"])
    assert max(i["B1_dif_maxima_tensor"] for m in r["miembros"] for i in m["imagenes"]) == pytest.approx(1e-3, rel=1e-2)


def test_etapa_ab_informa_formas_distintas_sin_interrumpirse():
    r = D.etapa_ab(contexto(tensor_g2=lambda f: tensor3(imagen_pil(f))[:, :4, :]))
    todas = [i for m in r["miembros"] for i in m["imagenes"]]
    assert all(i["B1_tensores_iguales"] is False and math.isinf(i["B1_dif_maxima_tensor"]) for i in todas)


# ============================================================================ etapa c
def test_etapa_c_reproduce_el_oof_con_la_disposicion_de_g2_y_muestra_el_efecto_del_lote():
    r = D.etapa_c(contexto())
    res = r["resumen"]
    assert res["C1a_dif_maxima_global"] == 0.0                                                       # lote exacto de G2, ruta de G2
    assert res["C1b_dif_maxima_global"] <= 1e-7                                                      # preprocesamiento de la interfaz apilado en ese lote
    d = res["C2_dif_maxima_por_tamano"]
    assert d["1"] > 1e-5 and d["2"] > 1e-5 and d["8"] > 1e-6 and d["32"] <= 1e-7                      # el barrido revela la dependencia del tamaño de lote
    assert r["tf32_desactivado"] is False and any("= 0.0" in x for x in r["interpretacion"])


def test_etapa_c_con_tf32_desactivado_solo_marca_la_variante():
    assert D.etapa_c(contexto(), True)["tf32_desactivado"] is True


def test_etapa_c_nunca_procesa_el_pliegue_completo():
    ctx = contexto()
    vistos = []
    original = ctx.prob_g2
    ctx.prob_g2 = lambda modelo, filas, tam, sem: (vistos.append(len(filas)), original(modelo, filas, tam, sem))[1]
    D.etapa_c(ctx)
    assert vistos and max(vistos) == D.LOTE_G2 and N_VAL > D.LOTE_G2                                   # solo lotes de ≤ 32 imágenes; el pliegue sintético tiene 64


def test_las_salidas_no_contienen_etiquetas_ni_metricas():
    for r in (D.plan(contexto()), D.etapa_ab(contexto()), D.etapa_c(contexto())):
        texto = json.dumps(r).lower()
        assert not any(x in texto for x in ("label", "y_true", "group_id", "average_precision", "roc_auc", "sensibilidad", "especificidad", "auc"))


# ============================================================================ CLI: compuertas, salidas nuevas y orden de etapas
AUT = "CONFIRMADO:D-074"


def correr(tmp_path, *args, ctx=None):
    return D.main([*args], root=tmp_path, ctx=ctx or contexto())


def test_cli_exige_la_autorizacion_de_d074_y_nada_mas(tmp_path):
    for malo in ("", "CONFIRMADO", "CONFIRMADO:D-072", "CONFIRMADO:D-073", "CONFIRMADO:D-075"):
        assert correr(tmp_path, "plan", "--autorizacion", malo) == 3
    assert not (tmp_path / "results").exists()


def test_cli_plan_y_ab_escriben_directorios_nuevos_sin_tocar_d072(tmp_path):
    d072 = tmp_path / "results" / "r9_gui" / "integracion_desarrollo" / "D-072"
    d072.mkdir(parents=True)
    (d072 / "resumen_integracion.json").write_text('{"ok": false}', encoding="utf-8")
    antes = (d072 / "resumen_integracion.json").read_bytes()
    assert correr(tmp_path, "plan", "--autorizacion", AUT) == 0 and correr(tmp_path, "ab", "--autorizacion", AUT) == 0
    base = tmp_path / D.SALIDA_REL
    assert (base / "D-074_plan" / "resumen_diagnostico.json").is_file() and (base / "D-074_ab" / "resumen_diagnostico.json").is_file()
    assert (d072 / "resumen_integracion.json").read_bytes() == antes and "integracion_desarrollo" not in str(base)
    assert correr(tmp_path, "ab", "--autorizacion", AUT) == 3                                         # no sobrescribe
    assert correr(tmp_path, "ab", "--autorizacion", AUT, "--sufijo", "_2") == 0


def test_cli_la_etapa_c_exige_el_resumen_de_ab_con_su_sha256(tmp_path):
    assert correr(tmp_path, "c", "--autorizacion", AUT) == 3
    assert correr(tmp_path, "ab", "--autorizacion", AUT) == 0
    resumen = tmp_path / D.SALIDA_REL / "D-074_ab" / "resumen_diagnostico.json"
    sha = hashlib.sha256(resumen.read_bytes()).hexdigest()
    assert correr(tmp_path, "c", "--autorizacion", AUT, "--resumen-ab", str(resumen), "--sha256-resumen-ab", "0" * 64) == 3
    assert correr(tmp_path, "c", "--autorizacion", AUT, "--resumen-ab", str(tmp_path / "no.json"), "--sha256-resumen-ab", sha) == 3
    plan_json = tmp_path / D.SALIDA_REL / "D-074_plan" / "resumen_diagnostico.json"
    assert correr(tmp_path, "plan", "--autorizacion", AUT) == 0
    assert correr(tmp_path, "c", "--autorizacion", AUT, "--resumen-ab", str(plan_json), "--sha256-resumen-ab", hashlib.sha256(plan_json.read_bytes()).hexdigest()) == 3   # no es de la etapa ab
    assert correr(tmp_path, "c", "--autorizacion", AUT, "--resumen-ab", str(resumen), "--sha256-resumen-ab", sha) == 0
    assert (tmp_path / D.SALIDA_REL / "D-074_c" / "resumen_diagnostico.json").is_file()


def test_cli_tf32_desactivado_solo_en_la_etapa_c_y_restaura_la_bandera(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    assert correr(tmp_path, "ab", "--autorizacion", AUT, "--tf32", "desactivado") == 3
    assert correr(tmp_path, "ab", "--autorizacion", AUT) == 0
    resumen = tmp_path / D.SALIDA_REL / "D-074_ab" / "resumen_diagnostico.json"
    sha = hashlib.sha256(resumen.read_bytes()).hexdigest()
    assert correr(tmp_path, "c", "--autorizacion", AUT, "--tf32", "desactivado", "--resumen-ab", str(resumen), "--sha256-resumen-ab", sha) == 0
    assert torch.backends.cudnn.allow_tf32 is False                                                   # lo fija el instrumento solo en esa ejecución (monkeypatch lo restaura)
    assert (tmp_path / D.SALIDA_REL / "D-074_c_tf32_desactivado" / "resumen_diagnostico.json").is_file()


def test_las_salidas_registran_autorizacion_y_que_no_hay_etiquetas_ni_metricas(tmp_path):
    assert correr(tmp_path, "ab", "--autorizacion", AUT) == 0
    j = json.loads((tmp_path / D.SALIDA_REL / "D-074_ab" / "resumen_diagnostico.json").read_text(encoding="utf-8"))
    assert j["autorizacion"] == AUT and j["etiquetas_usadas"] is False and j["metricas_de_clasificacion_calculadas"] is False and j["inferencia_de_ensamble"] is False
    assert j["entorno"] == {"dispositivo": "cpu", "cudnn_allow_tf32": None}


# ============================================================================ etapa c1 (D-075): solo C1a y C1b con el lote histórico de G2
AUT75 = "CONFIRMADO:D-075"


def test_etapa_c1_reproduce_el_oof_con_el_lote_historico_aunque_el_lote_1_difiera():
    ctx = contexto()
    assert D.etapa_ab(ctx)["resumen"]["B2_dif_maxima_B_oof"] > 1e-5                                    # el modelo sintético depende del tamaño de lote (como en «lote 1»)
    r = D.etapa_c1(ctx)
    res = r["resumen"]
    assert r["etapa"] == "c1" and res["C1a_dif_maxima_global"] == 0.0 and res["C1b_dif_maxima_global"] <= 1e-7 and res["C1b_dif_maxima_vs_C1a_global"] <= 1e-7
    assert any("= 0.0" in x for x in r["interpretacion"])
    assert [m["id_corrida"] for m in r["miembros"]] == [m["id_corrida"] for m in D.miembros_prioritarios(filas_miembros())]
    assert all(len(m["imagenes"]) == D.N_IMAGENES for m in r["miembros"])


def test_etapa_c1_no_ejecuta_c2_ni_el_pliegue_completo_ni_otros_tamanos():
    ctx = contexto()
    g2, apilados = [], []
    original_g2, original_t = ctx.prob_g2, ctx.prob_tensores
    ctx.prob_g2 = lambda modelo, filas, tam, sem: (g2.append((len(filas), tam)), original_g2(modelo, filas, tam, sem))[1]
    ctx.prob_tensores = lambda modelo, ts: (apilados.append(len(ts)), original_t(modelo, ts))[1]
    r = D.etapa_c1(ctx)
    assert g2 and all(tam == D.LOTE_G2 and n <= D.LOTE_G2 for n, tam in g2) and apilados and max(apilados) <= D.LOTE_G2
    assert len(g2) <= 2 * 6 and len(g2) == len(apilados)                                              # ≤ 2 lotes por miembro, un cómputo por lote distinto
    assert "C2_dif_maxima_por_tamano" not in r["resumen"] and "tf32_desactivado" not in r and "C2" not in json.dumps(r["miembros"])


def test_etapa_c1_detecta_una_diferencia_de_preprocesamiento_de_la_interfaz():
    ctx = contexto()
    ctx.tensor_gui = lambda pil: tensor3(pil).unsqueeze(0) + 1e-3
    res = D.etapa_c1(ctx)["resumen"]
    assert res["C1a_dif_maxima_global"] == 0.0 and res["C1b_dif_maxima_global"] > 1e-6 and res["C1b_dif_maxima_vs_C1a_global"] > 1e-6


def test_etapa_c1_detecta_una_disposicion_de_lotes_distinta_a_la_de_g2():
    ctx = contexto()
    original = ctx.prob_g2
    ctx.prob_g2 = lambda modelo, filas, tam, sem: original(modelo, filas, 8, sem)                       # lotes de 8 en lugar de los 32 históricos
    r = D.etapa_c1(ctx)
    assert r["resumen"]["C1a_dif_maxima_global"] > 1e-6 and any("NO reproduce el OOF" in x for x in r["interpretacion"])


def test_etapa_c1_las_salidas_no_contienen_etiquetas_ni_metricas():
    texto = json.dumps(D.etapa_c1(contexto())).lower()
    assert not any(x in texto for x in ("label", "y_true", "group_id", "average_precision", "roc_auc", "sensibilidad", "especificidad", "auc"))


def preparar_ab(tmp_path):
    assert correr(tmp_path, "ab", "--autorizacion", AUT) == 0
    resumen = tmp_path / D.SALIDA_REL / "D-074_ab" / "resumen_diagnostico.json"
    return resumen, hashlib.sha256(resumen.read_bytes()).hexdigest()


def test_cli_c1_exige_el_token_d075_y_las_demas_etapas_no_lo_aceptan(tmp_path):
    resumen, sha = preparar_ab(tmp_path)
    args = ("--resumen-ab", str(resumen), "--sha256-resumen-ab", sha)
    for malo in ("", "CONFIRMADO", "CONFIRMADO:D-074", "CONFIRMADO:D-072", "CONFIRMADO:D-076"):
        assert correr(tmp_path, "c1", "--autorizacion", malo, *args) == 3
    assert not (tmp_path / D.SALIDA_REL / "D-075_c1").exists()
    for etapa in ("plan", "ab", "c"):
        assert correr(tmp_path, etapa, "--autorizacion", AUT75, *args, "--sufijo", "_x") == 3
    assert correr(tmp_path, "c1", "--autorizacion", AUT75, *args) == 0


def test_cli_c1_exige_el_resumen_de_ab_con_su_sha256(tmp_path):
    assert correr(tmp_path, "c1", "--autorizacion", AUT75) == 3
    resumen, sha = preparar_ab(tmp_path)
    assert correr(tmp_path, "c1", "--autorizacion", AUT75, "--resumen-ab", str(resumen), "--sha256-resumen-ab", "0" * 64) == 3
    assert correr(tmp_path, "c1", "--autorizacion", AUT75, "--resumen-ab", str(tmp_path / "no.json"), "--sha256-resumen-ab", sha) == 3
    assert correr(tmp_path, "plan", "--autorizacion", AUT) == 0
    plan_json = tmp_path / D.SALIDA_REL / "D-074_plan" / "resumen_diagnostico.json"
    assert correr(tmp_path, "c1", "--autorizacion", AUT75, "--resumen-ab", str(plan_json), "--sha256-resumen-ab", hashlib.sha256(plan_json.read_bytes()).hexdigest()) == 3
    assert not (tmp_path / D.SALIDA_REL / "D-075_c1").exists()


def test_cli_c1_rechaza_tf32_escribe_directorio_nuevo_y_no_toca_ab_ni_d072(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    d072 = tmp_path / "results" / "r9_gui" / "integracion_desarrollo" / "D-072"
    d072.mkdir(parents=True)
    (d072 / "resumen_integracion.json").write_text('{"ok": false}', encoding="utf-8")
    resumen, sha = preparar_ab(tmp_path)
    antes_ab, antes_d072 = resumen.read_bytes(), (d072 / "resumen_integracion.json").read_bytes()
    args = ("--resumen-ab", str(resumen), "--sha256-resumen-ab", sha)
    assert correr(tmp_path, "c1", "--autorizacion", AUT75, *args, "--tf32", "desactivado") == 3
    assert correr(tmp_path, "c1", "--autorizacion", AUT75, *args) == 0
    salida = tmp_path / D.SALIDA_REL / "D-075_c1" / "resumen_diagnostico.json"
    j = json.loads(salida.read_text(encoding="utf-8"))
    assert j["etapa"] == "c1" and j["autorizacion"] == AUT75 and j["etiquetas_usadas"] is False and j["metricas_de_clasificacion_calculadas"] is False and j["inferencia_de_ensamble"] is False
    assert torch.backends.cudnn.allow_tf32 is True                                                     # c1 no toca la bandera TF32
    assert resumen.read_bytes() == antes_ab and (d072 / "resumen_integracion.json").read_bytes() == antes_d072
    assert not (tmp_path / D.SALIDA_REL / "D-074_c").exists() and not (tmp_path / D.SALIDA_REL / "D-074_c_tf32_desactivado").exists()
    assert correr(tmp_path, "c1", "--autorizacion", AUT75, *args) == 3                                  # no sobrescribe


# ============================================================================ alcance del código del instrumento
def test_el_instrumento_no_usa_etiquetas_ni_calcula_metricas_ni_toca_produccion():
    fuente = (Path(__file__).resolve().parents[2] / "src" / "gui" / "diagnostico_numerico.py").read_text(encoding="utf-8")
    for prohibido in ("label_v2", "y_true", "average_precision", "roc_auc", "calibrar", "optimizer", ".fit(", "agregar_y_decidir", "write_text(", "write_bytes(", "allow_tf32 = True"):
        assert prohibido not in fuente, prohibido
    assert "results/r9_gui/diagnostico_d074" in fuente and "integracion_desarrollo/D-072" not in fuente
