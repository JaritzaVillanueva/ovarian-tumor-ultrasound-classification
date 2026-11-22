"""Pruebas sintéticas de la regla R6 (`src/r7_v2/seleccion_r6.py`): sin torch, sin datos reales, sin test.

Comando: py -3.13 -B -m pytest tests/test_r7_v2_r6.py -q -p no:cacheprovider
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from src.r4_v2.common import GuardError
from src.r7_v2 import seleccion_r6 as S

REPO = Path(__file__).resolve().parents[1]
AUT = "CONFIRMADO:D-040"
ARQS = list(S.ARQUITECTURAS)


@pytest.fixture(autouse=True)
def _git_aislado(monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")


def filas(ap=None, roc=None, semillas_b=(42,), semillas_a=(42, 43, 44), extra=()):
    """6 combinaciones sintéticas; `ap`/`roc` = {(cond, arq): valor} sobre un valor base bajo."""
    ap, roc = ap or {}, roc or {}
    out = []
    for c in S.CONDICIONES:
        for a in ARQS:
            for s in (semillas_a if c == "A" else semillas_b):
                out.append({"condicion": c, "arquitectura": a, "semilla": s, "ap": ap.get((c, a), 0.30), "roc_auc": roc.get((c, a), 0.80),
                            "sensibilidad_0.5": 0.5, "especificidad_0.5": 0.9})
    return out + list(extra)


def elegida(r):
    return (r["elegida"]["condicion"], r["elegida"]["arquitectura"])


# ---------------------------------------------------------------- regla pura
def test_maximo_ap_gana():
    r = S.aplicar_regla(filas({("B_hibrida", "densenet121"): 0.45, ("A", "resnet50"): 0.40}, semillas_b=(42, 43, 44)))
    assert elegida(r) == ("B_hibrida", "densenet121") and not r["desempate_usado"] and r["paso_desempate"] is None
    assert r["ranking"][:2] == ["B_hibrida|densenet121", "A|resnet50"] and r["margen_1_2_ap_media"] == pytest.approx(0.05)


def test_ap_media_es_media_entre_semillas():
    f = filas({("B_hibrida", "resnet50"): 0.40}, semillas_b=(42, 43, 44))
    for x, s in zip((0.2, 0.3, 0.7), (42, 43, 44)):                      # A resnet50: media 0.4 -> empata con B 0.40 (A gana por parsimonia)
        next(g for g in f if (g["condicion"], g["arquitectura"], g["semilla"]) == ("A", "resnet50", s))["ap"] = x
    r = S.aplicar_regla(f)
    fila = next(t for t in r["tabla"] if (t["condicion"], t["arquitectura"]) == ("A", "resnet50"))
    assert fila["ap_media"] == pytest.approx(0.4) and fila["ap_rango"] == pytest.approx(0.5) and fila["n_semillas"] == 3
    assert elegida(r) == ("A", "resnet50") and r["paso_desempate"] == "i"


def test_redondeo_a_4_decimales_solo_detecta_empates():
    sin_empate = S.aplicar_regla(filas({("A", "resnet50"): 0.40001, ("B_hibrida", "resnet50"): 0.40009}, semillas_b=(42, 43, 44)))   # 0.4000 vs 0.4001: no hay empate
    assert elegida(sin_empate) == ("B_hibrida", "resnet50") and not sin_empate["desempate_usado"]
    con_empate = S.aplicar_regla(filas({("A", "resnet50"): 0.400010, ("B_hibrida", "resnet50"): 0.400040}, semillas_b=(42, 43, 44)))   # ambas 0.4000: empate -> A (i)
    assert elegida(con_empate) == ("A", "resnet50") and con_empate["paso_desempate"] == "i"
    assert con_empate["tabla"][0]["ap_media"] == pytest.approx(0.40001)      # la media NO se redondea


def test_desempate_i_a_sobre_b():
    r = S.aplicar_regla(filas({("A", "convnext_tiny"): 0.40, ("B_hibrida", "resnet50"): 0.40}, roc={("B_hibrida", "resnet50"): 0.99}, semillas_b=(42, 43, 44)))
    assert elegida(r) == ("A", "convnext_tiny") and r["desempate_usado"] and r["paso_desempate"] == "i"
    assert set(r["empatados_en_el_maximo"]) == {"A|convnext_tiny", "B_hibrida|resnet50"}


def test_desempate_ii_roc_auc_media():
    r = S.aplicar_regla(filas({("A", "resnet50"): 0.40, ("A", "convnext_tiny"): 0.40}, roc={("A", "convnext_tiny"): 0.90, ("A", "resnet50"): 0.85}))
    assert elegida(r) == ("A", "convnext_tiny") and r["paso_desempate"] == "ii"
    r = S.aplicar_regla(filas({("B_hibrida", "resnet50"): 0.40, ("B_hibrida", "densenet121"): 0.40}, roc={("B_hibrida", "densenet121"): 0.9}, semillas_b=(42, 43, 44)))
    assert elegida(r) == ("B_hibrida", "densenet121") and r["paso_desempate"] == "ii"


def test_desempate_iii_orden_fijo():
    ap = {("A", a): 0.40 for a in ARQS}
    r = S.aplicar_regla(filas(ap))                                             # ROC-AUC iguales
    assert elegida(r) == ("A", "resnet50") and r["paso_desempate"] == "iii"
    r = S.aplicar_regla(filas({("A", "densenet121"): 0.40, ("A", "convnext_tiny"): 0.40}))
    assert elegida(r) == ("A", "densenet121") and r["paso_desempate"] == "iii"
    r = S.aplicar_regla(filas({("B_hibrida", "densenet121"): 0.40, ("B_hibrida", "convnext_tiny"): 0.40}, semillas_b=(42, 43, 44)))
    assert elegida(r) == ("B_hibrida", "densenet121") and r["paso_desempate"] == "iii"


def test_condicion_c_y_b_oof_se_ignoran_y_se_registran():
    extra = [{"condicion": "C", "arquitectura": "resnet50", "semilla": 42, "ap": 0.99, "roc_auc": 0.99},
             {"condicion": "B_oof", "arquitectura": "resnet50", "semilla": 42, "ap": 0.98, "roc_auc": 0.99}]
    r = S.aplicar_regla(filas({("A", "resnet50"): 0.35}, extra=extra))
    assert elegida(r) == ("A", "resnet50") and r["condiciones_ignoradas"] == {"C": 1, "B_oof": 1} and len(r["tabla"]) == 6
    with pytest.raises(GuardError, match="desconocida"):
        S.aplicar_regla(filas(extra=[{"condicion": "X", "arquitectura": "resnet50", "semilla": 42, "ap": 0.1, "roc_auc": 0.1}]))


def test_bandera_r6_2_con_b_de_una_y_tres_semillas():
    r1 = S.aplicar_regla(filas())
    assert r1["b_excluida_de_la_seleccion_por_semillas_insuficientes"] is True and r1["seleccion_formal_solo_entre"] == r1["ranking"]
    assert all(i.startswith("A|") for i in r1["ranking"]) and len(r1["ranking"]) == 3
    assert r1["r6_2_media_entre_semillas_completa"] is False and r1["combinaciones_con_menos_de_3_semillas"] == [f"B_hibrida|{a}" for a in ARQS]
    assert r1["semillas_comunes"] == [42] and r1["semillas_homogeneas_entre_combinaciones"] is False and "n_semillas" in r1["nota_limitacion"]
    r3 = S.aplicar_regla(filas(semillas_b=(42, 43, 44)))
    assert r3["r6_2_media_entre_semillas_completa"] is True and r3["combinaciones_con_menos_de_3_semillas"] == []
    assert r3["semillas_homogeneas_entre_combinaciones"] is True and r3["b_excluida_de_la_seleccion_por_semillas_insuficientes"] is False
    assert len(r3["seleccion_formal_solo_entre"]) == 6 and len(r3["ranking"]) == 6


def test_b_de_una_semilla_no_es_elegida_ni_con_ap_mayor():
    """Patrón real: A con 3 semillas por arquitectura; B solo semilla 42. La elegida sale de A con B menor y con B mayor."""
    for ap_b in (0.10, 0.90):
        ap = {("A", "resnet50"): 0.30, ("A", "densenet121"): 0.35, ("A", "convnext_tiny"): 0.32, **{("B_hibrida", a): ap_b for a in ARQS}}
        r = S.aplicar_regla(filas(ap))
        assert elegida(r) == ("A", "densenet121") and not r["desempate_usado"]          # sale de los datos, no está codificada
        assert [t["rango"] for t in r["tabla"][:3]] == [1, 2, 3] and len(r["ranking"]) == 3
        b = [t for t in r["tabla"] if t["condicion"] == "B_hibrida"]
        assert len(b) == 3 and all(t["elegible"] is False and t["elegida"] is False and t["rango"] is None and t["n_semillas"] == 1 for t in b)
        assert r["margen_1_2_ap_media"] == pytest.approx(0.03) and not any(i.startswith("B") for i in r["empatados_en_el_maximo"])
    ap = {("A", "resnet50"): 0.30, ("A", "densenet121"): 0.35, ("A", "convnext_tiny"): 0.32, **{("B_hibrida", a): 0.90 for a in ARQS}}
    r = S.aplicar_regla(filas(ap, semillas_b=(42, 43, 44)))                         # con 3 semillas B sí es elegible y puede ser elegida
    assert elegida(r)[0] == "B_hibrida" and len(r["ranking"]) == 6


def test_comparacion_emparejada_semilla_comun():
    ap = {("A", "resnet50"): 0.30, ("B_hibrida", "resnet50"): 0.36}
    f = filas(ap)
    next(g for g in f if (g["condicion"], g["arquitectura"], g["semilla"]) == ("A", "resnet50", 42))["ap"] = 0.32
    r = S.aplicar_regla(f)["comparacion_emparejada_semilla_comun"]
    assert [e["arquitectura"] for e in r] == ARQS
    e = r[0]
    assert e["semillas_comunes"] == [42] and e["ap_A"] == {"42": 0.32} and e["ap_B_hibrida"] == {"42": 0.36}
    assert e["diferencia_B_menos_A"] == pytest.approx(0.04) and "condicional" in e["etiqueta"] and "IC" in e["etiqueta"]
    sin = S.aplicar_regla(filas(semillas_b=(50,)))["comparacion_emparejada_semilla_comun"]       # sin semillas comunes: vacío
    assert sin == []


def test_a_debe_tener_tres_semillas_en_las_tres_arquitecturas():
    with pytest.raises(GuardError, match="A debe tener 3 semillas"):
        S.aplicar_regla(filas(semillas_a=(42, 43)))
    f = [g for g in filas() if not (g["condicion"] == "A" and g["arquitectura"] == "convnext_tiny" and g["semilla"] == 44)]
    with pytest.raises(GuardError, match="A debe tener 3 semillas"):
        S.aplicar_regla(f)


def test_faltan_arquitecturas_o_filas_duplicadas_fallan():
    with pytest.raises(GuardError, match="faltan combinaciones"):
        S.aplicar_regla([f for f in filas() if f["arquitectura"] != "convnext_tiny"])
    with pytest.raises(GuardError, match="faltan combinaciones"):
        S.aplicar_regla([f for f in filas() if f["condicion"] != "B_hibrida"])
    with pytest.raises(GuardError, match="duplicada"):
        S.aplicar_regla(filas() + [dict(filas()[0])])
    malo = filas()
    malo[0]["ap"] = "nan"
    with pytest.raises(GuardError, match="rango|finito"):
        S.aplicar_regla(malo)


def test_descriptivas_no_alteran_la_seleccion():
    f = filas({("A", "resnet50"): 0.40, ("A", "densenet121"): 0.39})
    for g in f:
        g["sensibilidad_0.5"] = 0.0 if (g["condicion"], g["arquitectura"]) == ("A", "resnet50") else 1.0
    r = S.aplicar_regla(f)
    assert elegida(r) == ("A", "resnet50") and r["tabla"][0]["desc_sensibilidad_0.5_media"] == 0.0


def test_determinismo_de_la_regla():
    f = filas({("A", "densenet121"): 0.41})
    assert json.dumps(S.aplicar_regla(f), sort_keys=True) == json.dumps(S.aplicar_regla(list(reversed(f))), sort_keys=True)


# ---------------------------------------------------------------- CLI / ejecución sobre un repositorio sintético
def escribir_t(path, cols, rows, comentario=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        if comentario:
            fh.write("# mapeo v2 | phash_le8_v1 | sintetico\n")
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def montar_repo(tmp_path, ap=None):
    root = tmp_path / "repo"
    (root / "src" / "r7_v2").mkdir(parents=True)
    for n in ("protocolo_r6_v1.md", "seleccion_r6.py"):
        shutil.copyfile(REPO / "src" / "r7_v2" / n, root / "src" / "r7_v2" / n)
    f = filas(ap)
    cols3 = ["arquitectura", "semilla", "ap", "roc_auc", "sensibilidad_0.5", "especificidad_0.5", "fp", "fn"]
    a = [{**g, "fp": 1, "fn": 1} for g in f if g["condicion"] == "A"]
    escribir_t(root / S.T3_REL, cols3, [{k: g[k] for k in cols3} for g in a])
    colsb = ["condicion", *cols3]
    b = [{**g, "fp": 1, "fn": 1} for g in f if g["condicion"] == "B_hibrida" or g["semilla"] == 42]
    b.append({"condicion": "C", "arquitectura": "resnet50", "semilla": 42, "ap": 0.99, "roc_auc": 0.99, "fp": 0, "fn": 0})
    escribir_t(root / S.TB1_REL, colsb, [{k: g.get(k, "") for k in colsb} for g in b])
    for cmd in (["init", "-q"], ["add", "src"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x"]):
        subprocess.run(["git", "-C", str(root), *cmd], check=True, capture_output=True)
    return root


def sin_fecha(path):
    d = json.loads((path / "seleccion_r6.json").read_text(encoding="utf-8"))
    d.pop("fecha_utc")
    return d


def test_ejecutar_escribe_json_y_csv_y_no_sobrescribe(tmp_path):
    root = montar_repo(tmp_path, {("A", "densenet121"): 0.42})
    destino = S.ejecutar(root, AUT)
    j = json.loads((destino / "seleccion_r6.json").read_text(encoding="utf-8"))
    assert j["elegida"] == {"condicion": "A", "arquitectura": "densenet121"} and j["r6_2_media_entre_semillas_completa"] is False
    assert j["condiciones_ignoradas"] == {"C": 1} and j["comprobacion_A_t3_vs_tb1"]["n_comparaciones_A"] == 3 and j["autorizacion"] == AUT
    assert len(j["integridad"]["git_commit_head"]) == 40 and len(j["tabla"]) == 6 and "nota_limitacion" in j and j["leyenda"] == S.LEYENDA
    assert len(j["comparacion_emparejada_semilla_comun"]) == 3 and j["seleccion_formal_solo_entre"] == j["ranking"] and len(j["ranking"]) == 3
    assert all(len(v["sha256_bytes_archivo"]) == 64 for v in j["insumos_sha256"].values())
    assert j["integridad"]["archivos"][S.PROTOCOLO_REL]["sha256_contenido_blob"] is not None
    lineas = (destino / "tabla_r6.csv").read_text(encoding="utf-8").splitlines()
    assert lineas[0].startswith("# ") and S.LEYENDA in lineas[0] and len(lineas) == 8 and lineas[1].split(",")[0] == "rango"
    with pytest.raises(GuardError, match="no se sobrescribe"):
        S.ejecutar(root, AUT)


def test_determinismo_de_la_ejecucion(tmp_path):
    root = montar_repo(tmp_path, {("A", "densenet121"): 0.42})
    d1 = S.ejecutar(root, AUT)
    shutil.move(str(d1), str(tmp_path / "primera"))
    d2 = S.ejecutar(root, AUT)
    assert sin_fecha(tmp_path / "primera") == sin_fecha(d2)
    assert (tmp_path / "primera" / "tabla_r6.csv").read_bytes() == (d2 / "tabla_r6.csv").read_bytes()


def test_exige_autorizacion_y_rechaza_rutas_test(tmp_path):
    root = montar_repo(tmp_path)
    for mala in (None, "", "CONFIRMADO:AUT-001", "D-001", "CONFIRMADO:D-1"):
        with pytest.raises(GuardError, match="autorizacion"):
            S.ejecutar(root, mala)
    copia = root / "results" / "test" / "T3.csv"
    copia.parent.mkdir(parents=True)
    shutil.copyfile(root / S.T3_REL, copia)
    with pytest.raises(GuardError, match="test"):
        S.ejecutar(root, AUT, g2_t3="results/test/T3.csv")
    with pytest.raises(GuardError, match="test"):
        S.ejecutar(root, AUT, g2_t4="results/Test/T4.csv")
    assert not (root / S.SALIDA_REL).exists()


def test_arbol_sucio_en_r7_v2_bloquea(tmp_path):
    root = montar_repo(tmp_path)
    (root / "src" / "r7_v2" / "extra.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(GuardError, match="arbol_sucio"):
        S.ejecutar(root, AUT)
    assert not (root / S.SALIDA_REL).exists()


def test_a_de_g2_y_g3b_deben_coincidir_y_t4_es_solo_descriptivo(tmp_path):
    root = montar_repo(tmp_path)
    t4 = root / "results" / "r4_v2" / "T4_delta_ap.csv"
    escribir_t(t4, ["comparacion", "metrica", "delta"], [{"comparacion": "densenet121-resnet50", "metrica": "ap", "delta": "0.010000"},
                                                          {"comparacion": "densenet121-resnet50", "metrica": "roc_auc", "delta": "0.5"}])
    destino = S.ejecutar(root, AUT, g2_t4="results/r4_v2/T4_delta_ap.csv")
    j = json.loads((destino / "seleccion_r6.json").read_text(encoding="utf-8"))
    assert j["t4_descriptivo_no_participa"] == [{"comparacion": "densenet121-resnet50", "delta_ap": "0.010000"}]
    shutil.rmtree(destino)
    tb1 = root / S.TB1_REL
    lineas = tb1.read_text(encoding="utf-8").splitlines()
    i = next(n for n, l in enumerate(lineas) if l.startswith("A,resnet50,42"))
    partes = lineas[i].split(",")
    partes[3] = "0.310000"
    lineas[i] = ",".join(partes)
    tb1.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    with pytest.raises(GuardError, match="difiere entre T3"):
        S.ejecutar(root, AUT)


def test_csv_sin_comentario_o_sin_columnas(tmp_path):
    p = tmp_path / "t.csv"
    escribir_t(p, ["arquitectura", "semilla", "ap", "roc_auc"], [{"arquitectura": "resnet50", "semilla": 42, "ap": 0.3, "roc_auc": 0.8}], comentario=False)
    assert S.leer_csv_comentado(p, ("arquitectura", "ap"))[0]["ap"] == "0.3"
    with pytest.raises(GuardError, match="sin columnas"):
        S.leer_csv_comentado(p, ("condicion",))
