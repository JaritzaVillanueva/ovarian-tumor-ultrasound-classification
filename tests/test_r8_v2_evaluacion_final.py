"""Pruebas sintéticas del ejecutor de R8 (`r8_v2.evaluacion_final`): compuertas, ejecución única, mapeo binario v2, separación inferencia/evaluación, integridad de checkpoints,
reproducibilidad y umbral intacto. TODO es sintético: raíz temporal, manifiesto y «imágenes» inventados, modelos y predictor de prueba. NO abren el test real ni ejecutan R8.
Comando: py -3.13 -B -m pytest tests/test_r8_v2_evaluacion_final.py -q -p no:cacheprovider
"""

from __future__ import annotations

import copy
import csv
import hashlib
import inspect
import json
import os
import subprocess
import zlib
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("yaml")
pytest.importorskip("scipy")
sk = pytest.importorskip("sklearn.metrics")

from src.r4_v2.common import GuardError  # noqa: E402
from src.r7_v2 import sistema_final as SF  # noqa: E402
from src.r7_v2 import umbral_sistema as U  # noqa: E402
from src.r8_v2 import declaracion as D  # noqa: E402
from src.r8_v2 import estadistica as ES  # noqa: E402
from src.r8_v2 import evaluacion_final as E  # noqa: E402
from tests import test_r8_v2_congelamiento as TC  # noqa: E402
from tests import test_r8_v2_declaracion as TD  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
AUT = "CONFIRMADO:D-061"
G2_REL = "configs/r4_v2/g2_baseline_v1.yaml"
NEG_CLASES = (0, 1, 2, 3, 4, 6)


# ============================================================================ banco sintético
def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def filas_manifiesto(n0=367, n1=15, nn=87, extra=(("train", 12), ("validation", 8))):
    """Manifiesto mixto sintético. `binary_label` histórico marca 6 y 7 como positivos (mapeo anterior) para comprobar que NO se usa. Los ids de positivas empiezan por `T1_`."""
    filas = []
    for k in range(n0):
        c = NEG_CLASES[k % len(NEG_CLASES)]
        filas.append({"image_id": f"T0_{k:04d}", "image_name": f"T0_{k:04d}.png", "source_split": "test", "original_class": c, "binary_label": 1 if c == 6 else 0, "binary_class": "x",
                      "experimental_split": "test"})
    for k in range(n1):
        filas.append({"image_id": f"T1_{k:04d}", "image_name": f"T1_{k:04d}.png", "source_split": "test", "original_class": 7, "binary_label": 1, "binary_class": "x", "experimental_split": "test"})
    for k in range(nn):
        filas.append({"image_id": f"TN_{k:04d}", "image_name": f"TN_{k:04d}.png", "source_split": "test", "original_class": 5, "binary_label": 0, "binary_class": "x", "experimental_split": "test"})
    for split, n in extra:
        for k in range(n):
            filas.append({"image_id": f"D_{split}_{k:03d}", "image_name": f"D_{split}_{k:03d}.png", "source_split": "train", "original_class": 7 if k % 5 == 0 else 1,
                          "binary_label": 0, "binary_class": "x", "experimental_split": split})
    return filas


def escribir_manifiesto(root: Path, filas):
    p = root / E.MANIFIESTO_TEST_REL
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(filas)
    return p


def crear_imagenes(root: Path, filas):
    d = root / E.IMAGENES_DIR / "test" / "images"
    d.mkdir(parents=True, exist_ok=True)
    for f in filas:
        if f["experimental_split"] == "test" and str(f["original_class"]) != "5":                  # clases inválidas de prueba (p. ej. «x») también tienen archivo
            (d / f["image_name"]).parent.mkdir(parents=True, exist_ok=True)
            (d / f["image_name"]).write_bytes(b"\x89PNG sintetica")


def reescribir_protocolo(root: Path, mutar=None) -> str:
    p = json.loads((root / E.PROTOCOLO_REL).read_text(encoding="utf-8"))
    if mutar:
        mutar(p)
    (root / E.PROTOCOLO_REL).write_bytes((json.dumps(p, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return D.sha256_lf(root / E.PROTOCOLO_REL)


class Espia:
    def __init__(self):
        self.llamadas, self.modelos_cargados = [], []

    def cargar_modelo(self, ruta, device):
        self.modelos_cargados.append(str(ruta))
        return {"ruta": str(ruta)}

    def predecir(self, m):
        def pr(modelo, filas, device):
            self.llamadas.append((m["id_corrida"], tuple(sorted(filas[0])), len(filas), modelo["ruta"]))
            base = np.array([(zlib.crc32(f"{f['image_id']}|{m['id_corrida']}".encode()) % 1000) / 1000.0 * 0.3 for f in filas])
            sube = np.array([0.4 if f["image_id"].startswith("T1_") else 0.0 for f in filas])          # doble de prueba «informativo»: usa el id, no etiquetas reales
            return np.clip(base + sube, 0.0, 1.0).astype(np.float32)
        return pr


def snapshot_falso(head="a" * 40, bloqueos=None):
    def f():
        return {"git_commit_head": head, "rama": "sintetica", "status_porcelain": [], "bloqueos": list(bloqueos or []), "archivos": {E.PROTOCOLO_REL: {"sha256_bytes_archivo": "1" * 64}}}
    return f


class Banco:
    def __init__(self, tmp_path, monkeypatch, filas=None):
        monkeypatch.setattr(TC, "G2_BLOB", D.sha256_lf(REPO / G2_REL))                               # el YAML sintético fija el SHA-256 real de la configuración de G2
        self.root, self.config, self.ruta_decl, self.decl = TD.armar_decl(tmp_path)
        (self.root / G2_REL).parent.mkdir(parents=True, exist_ok=True)
        (self.root / G2_REL).write_bytes((REPO / G2_REL).read_bytes())
        hist = self.root / E.PROTOCOLO_HISTORICO_REL
        hist.parent.mkdir(parents=True, exist_ok=True)
        hist.write_bytes(b'{"historico": "sintetico"}\n')
        p = json.loads((REPO / E.PROTOCOLO_REL).read_text(encoding="utf-8"))
        p["sistema"]["declaracion"]["sha256_contenido"] = D.sha256_lf(self.ruta_decl)
        p["historico"]["protocolo_original"]["sha256_contenido"] = D.sha256_lf(hist)
        (self.root / E.PROTOCOLO_REL).parent.mkdir(parents=True, exist_ok=True)
        (self.root / E.PROTOCOLO_REL).write_bytes((json.dumps(p, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        self.filas = filas if filas is not None else filas_manifiesto()
        escribir_manifiesto(self.root, self.filas)
        crear_imagenes(self.root, self.filas)
        self.espia = Espia()
        self.sha_decl = D.sha256_lf(self.ruta_decl)

    @property
    def sha_proto(self):
        return D.sha256_lf(self.root / E.PROTOCOLO_REL)

    @property
    def salida(self):
        return self.root / E.SALIDA_REL

    def kw(self, **extra):
        base = dict(autorizacion=AUT, sha256_protocolo=self.sha_proto, sha256_declaracion=self.sha_decl, verificar=snapshot_falso(), cargar_modelo=self.espia.cargar_modelo, device="cpu",
                    entorno=dict(TC.ENTORNO))
        base.update(extra)
        return base

    def ejecutar(self, **extra):
        kw = self.kw(confirmacion=E.CONFIRMACION, predecir=self.espia.predecir)
        kw.update(extra)
        return E.ejecutar(self.root, **kw)


@pytest.fixture
def banco(tmp_path, monkeypatch):
    return Banco(tmp_path, monkeypatch)


@pytest.fixture
def sin_test(monkeypatch):
    """Detecta cualquier lectura del manifiesto del test o inferencia: debe NO ocurrir cuando una compuerta falla."""
    def prohibido(*a, **k):
        raise AssertionError("se abrió el test antes de superar las compuertas")
    monkeypatch.setattr(E, "leer_manifiesto_prueba", prohibido)
    monkeypatch.setattr(E, "predictor_prueba", prohibido)


def sin_salida(b):
    return not b.salida.exists()


# ============================================================================ flujo completo, mapeo v2 y reproducibilidad
def test_flujo_completo_mapeo_v2_y_resultados_consistentes(banco):
    r = banco.ejecutar()
    d = banco.salida
    assert r["reporte_aceptado"] and {p.name for p in d.iterdir()} == set(json.loads((REPO / E.PROTOCOLO_REL).read_text(encoding="utf-8"))["salidas"]["archivos"])
    with open(d / "evaluacion_por_imagen.csv", encoding="utf-8", newline="") as fh:
        ev = list(csv.DictReader(fh))
    ids = {x["image_id"] for x in ev}
    assert len(ev) == 382 and not any(i.startswith("TN_") or i.startswith("D_") for i in ids)               # normales (5) y desarrollo fuera
    assert {int(x["original_class"]) for x in ev} == {0, 1, 2, 3, 4, 6, 7}
    assert all(int(x["label_v2"]) == (1 if x["original_class"] == "7" else 0) for x in ev)                    # positivo solo 7; el 6 es negativo aunque binary_label histórico sea 1
    res = json.loads((d / "resultados_r8.json").read_text(encoding="utf-8"))
    y = np.array([int(x["label_v2"]) for x in ev])
    p = np.array([float(x["p_ensamble"]) for x in ev])
    t = float(SF.T_HAT)
    pr = res["resultados"]
    assert res["recuentos"] == {"elegibles_clase_0": 367, "elegibles_clase_1": 15, "normales_excluidas": 87} and pr["n"] == 382 and pr["n_positivas"] == 15
    assert pr["umbral_primario"]["umbral"] == t and (pr["umbral_primario"]["tp"], pr["umbral_primario"]["fn"]) == (int(np.sum((p >= t) & (y == 1))), int(np.sum((p < t) & (y == 1))))
    assert pr["umbral_primario"]["fp"] == int(np.sum((p >= t) & (y == 0)))
    assert pr["puntuales"]["ap"] == pytest.approx(sk.average_precision_score(y, p), abs=1e-12) and pr["puntuales"]["roc_auc"] == pytest.approx(sk.roc_auc_score(y, p), abs=1e-12)
    assert pr["ic95_bootstrap_percentil_estratificado"]["B"] == 10000 and pr["ic95_bootstrap_percentil_estratificado"]["semilla"] == 20261123
    assert [int(x["positivo_t_hat"]) for x in ev] == [int(v >= t) for v in p]
    estado = json.loads((d / "estado_final.json").read_text(encoding="utf-8"))
    assert estado["estado"] == "completada" and json.loads((d / "ejecucion.lock.json").read_text(encoding="utf-8"))["estado"] == "iniciada"


def test_el_promedio_es_aritmetico_simple_de_los_quince_miembros(banco):
    banco.ejecutar()
    with open(banco.salida / "predicciones_por_miembro.csv", encoding="utf-8", newline="") as fh:
        pm = list(csv.DictReader(fh))
    with open(banco.salida / "predicciones_ensamble.csv", encoding="utf-8", newline="") as fh:
        en = list(csv.DictReader(fh))
    assert len(pm[0]) == 16 and len(banco.espia.modelos_cargados) == 15 and len({c[0] for c in banco.espia.llamadas}) == 15
    for a, b in zip(pm[:50], en[:50]):
        m = np.mean([float(v) for k, v in a.items() if k != "image_id"], dtype=np.float64)
        assert float(b["p_ensamble"]) == pytest.approx(m, abs=1e-12) and a["image_id"] == b["image_id"]


def test_resultados_reproducibles_entre_ejecuciones_independientes(tmp_path, monkeypatch):
    a = Banco(tmp_path / "a", monkeypatch)
    b = Banco(tmp_path / "b", monkeypatch)
    a.ejecutar()
    b.ejecutar()
    ra = json.loads((a.salida / "resultados_r8.json").read_text(encoding="utf-8"))["resultados"]
    rb = json.loads((b.salida / "resultados_r8.json").read_text(encoding="utf-8"))["resultados"]
    assert ra == rb
    assert (a.salida / "predicciones_ensamble.csv").read_bytes() == (b.salida / "predicciones_ensamble.csv").read_bytes()
    assert (a.salida / "metricas_primarias.csv").read_bytes() == (b.salida / "metricas_primarias.csv").read_bytes()


# ============================================================================ compuertas: ninguna se supera => el test no se abre
@pytest.mark.parametrize("extra", [
    {"confirmacion": None}, {"confirmacion": "EJECUTAR"}, {"autorizacion": ""}, {"autorizacion": "CONFIRMADO:D-61"}, {"sha256_protocolo": "0" * 64}, {"sha256_protocolo": "abc"},
    {"sha256_declaracion": "0" * 64}, {"sha256_declaracion": ""}, {"sha256_declaracion": "A" * 64}, {"verificar": snapshot_falso(bloqueos=[{"codigo": "arbol_sucio", "detalle": "x"}])}])
def test_compuerta_fallida_no_abre_el_test_ni_crea_salida(banco, sin_test, extra):
    with pytest.raises(GuardError):
        banco.ejecutar(**extra)
    assert sin_salida(banco) and banco.espia.llamadas == []


def test_protocolo_alterado_o_con_parametros_distintos_bloquea(banco, sin_test):
    cambios = [lambda p: p["estadistica"].update(n_bootstrap=2000), lambda p: p["estadistica"].update(semilla_bootstrap=20260101), lambda p: p["estadistica"].update(umbral_primario="0.5"),
               lambda p: p["estadistica"].update(analisis_por_numero_de_modelos=True), lambda p: p["estadistica"].update(ic_sensibilidad_especificidad="Wald"),
               lambda p: p["datos"]["mapeo"].update(clase_1_lesion_maligna=[6, 7]), lambda p: p["datos"]["mapeo"].update(excluida_ovario_normal=[]),
               lambda p: p["datos"]["recuentos_documentados"].update(elegibles_clase_1=16), lambda p: p["iov_originales"]["R8.2"].update(texto="Se reporta AP."),
               lambda p: p["iov_originales"].pop("R8.4"), lambda p: p["sistema"].update(t_hat="0.1"), lambda p: p["sistema"].update(n_miembros=3),
               lambda p: p["historico"].pop("contraste_b2_vs_b0"), lambda p: p["historico"]["contraste_b2_vs_b0"].update(estado="ejecutado y cumplido"),
               lambda p: p["historico"]["protocolo_original"].update(sha256_contenido="0" * 64), lambda p: p["autorizacion_requerida"].update(independiente_y_especifica=False),
               lambda p: p.update(esquema="otro/1")]
    original = (banco.root / E.PROTOCOLO_REL).read_bytes()
    for mut in cambios:
        (banco.root / E.PROTOCOLO_REL).write_bytes(original)
        nuevo = reescribir_protocolo(banco.root, mut)
        with pytest.raises(GuardError):
            banco.ejecutar(sha256_protocolo=nuevo)
        assert sin_salida(banco)


def test_protocolo_que_apunta_a_otra_declaracion_bloquea(banco, sin_test):
    nuevo = reescribir_protocolo(banco.root, lambda p: p["sistema"]["declaracion"].update(sha256_contenido="2" * 64))
    with pytest.raises(GuardError, match="declaración"):
        banco.ejecutar(sha256_protocolo=nuevo)
    assert sin_salida(banco)


@pytest.mark.parametrize("que", ["yaml", "protocolo_de_congelamiento", "manifiesto_miembros", "codigo_inferencia", "checkpoint", "reporte", "protocolo_historico", "config_g2"])
def test_artefacto_congelado_alterado_bloquea_antes_de_abrir_el_test(banco, sin_test, que):
    rutas = {"yaml": TC.C.CONFIG_REL, "protocolo_de_congelamiento": TC.C.PROTOCOLO_REL, "manifiesto_miembros": TC.C.MIEMBROS_REL, "codigo_inferencia": D.CODIGO_INFERENCIA[0],
             "checkpoint": banco.decl["miembros"][4]["ruta_checkpoint"], "reporte": banco.decl["verificacion"]["ruta"], "protocolo_historico": E.PROTOCOLO_HISTORICO_REL, "config_g2": G2_REL}
    p = banco.root / rutas[que]
    p.write_bytes(p.read_bytes() + b"\n# alterado\n")
    with pytest.raises(GuardError):
        banco.ejecutar()
    assert sin_salida(banco) and banco.espia.llamadas == []


def test_declaracion_ausente_bloquea(banco, sin_test):
    banco.ruta_decl.unlink()
    with pytest.raises(GuardError):
        banco.ejecutar()


def test_el_yaml_con_congelado_true_no_es_aceptado(banco, sin_test):
    import yaml
    c = yaml.safe_load((banco.root / TC.C.CONFIG_REL).read_text(encoding="utf-8"))
    c["sistema"]["congelado"] = True
    (banco.root / TC.C.CONFIG_REL).write_bytes(yaml.safe_dump(c, allow_unicode=True, sort_keys=False).encode("utf-8"))
    with pytest.raises(GuardError):
        banco.ejecutar()


# ============================================================================ ejecución única
def test_segunda_ejecucion_bloqueada_y_resultados_intactos(banco):
    banco.ejecutar()
    antes = {p.name: p.read_bytes() for p in banco.salida.iterdir()}
    n = len(banco.espia.llamadas)
    with pytest.raises(GuardError, match="ya existe"):
        banco.ejecutar()
    assert {p.name: p.read_bytes() for p in banco.salida.iterdir()} == antes and len(banco.espia.llamadas) == n


def test_directorio_de_salida_preexistente_bloquea_sin_abrir_el_test(banco, sin_test):
    banco.salida.mkdir(parents=True)
    with pytest.raises(GuardError, match="ya existe"):
        banco.ejecutar()


def test_fallo_tras_el_bloqueo_deja_bloqueo_y_estado_abortada(banco, monkeypatch):
    liberadas = []
    original = E.Reservadas.liberar
    monkeypatch.setattr(E.Reservadas, "liberar", lambda self, *a: liberadas.append(1) or original(self, *a))

    def predecir_roto(m):
        def pr(modelo, filas, device):
            raise RuntimeError("fallo de GPU simulado")
        return pr
    with pytest.raises(RuntimeError):
        banco.ejecutar(predecir=predecir_roto)
    assert (banco.salida / "ejecucion.lock.json").is_file() and not (banco.salida / "predicciones_ensamble.csv").exists() and liberadas == []
    assert json.loads((banco.salida / "estado_final.json").read_text(encoding="utf-8"))["estado"] == "abortada"
    with pytest.raises(GuardError, match="ya existe"):
        banco.ejecutar()


# ============================================================================ mapeo v2, selección y consistencia del manifiesto (aborta ANTES de inferir)
def _aborta_antes_de_inferir(banco, patron):
    with pytest.raises(GuardError, match=patron):
        banco.ejecutar()
    assert banco.espia.llamadas == [] and not (banco.salida / "predicciones_ensamble.csv").exists()
    assert json.loads((banco.salida / "estado_final.json").read_text(encoding="utf-8"))["estado"] == "abortada"


def test_recuentos_distintos_de_los_documentados_detienen_antes_de_inferir(tmp_path, monkeypatch):
    _aborta_antes_de_inferir(Banco(tmp_path, monkeypatch, filas_manifiesto(n0=366)), "documentados")


def test_categoria_desconocida_detiene_antes_de_inferir(tmp_path, monkeypatch):
    f = filas_manifiesto()
    f[0]["original_class"] = 8
    _aborta_antes_de_inferir(Banco(tmp_path, monkeypatch, f), "desconocida")


def test_fila_de_test_que_no_procede_de_test_detiene(tmp_path, monkeypatch):
    f = filas_manifiesto()
    f[3]["source_split"] = "train"
    _aborta_antes_de_inferir(Banco(tmp_path, monkeypatch, f), "partición oficial")


@pytest.mark.parametrize("cambio,patron", [({"image_id": "T0_0000"}, "repetido"), ({"image_name": "../x.png"}, "no seguro"), ({"image_name": "a/b.png"}, "no seguro"), ({"original_class": "x"}, "no entero")])
def test_ids_o_nombres_invalidos_detienen(tmp_path, monkeypatch, cambio, patron):
    f = filas_manifiesto()
    f[1].update(cambio)
    b = Banco(tmp_path, monkeypatch, f)
    _aborta_antes_de_inferir(b, patron)


def test_imagen_faltante_detiene_antes_de_inferir(tmp_path, monkeypatch):
    b = Banco(tmp_path, monkeypatch)
    (b.root / E.IMAGENES_DIR / "test" / "images" / "T0_0005.png").unlink()
    _aborta_antes_de_inferir(b, "no encontrada")


def test_las_normales_no_se_predicen_aunque_existan_sus_archivos(banco):
    d = banco.root / E.IMAGENES_DIR / "test" / "images"
    for k in range(87):
        (d / f"TN_{k:04d}.png").write_bytes(b"x")
    banco.ejecutar()
    ids = [ln.split(",")[0] for ln in (banco.salida / "predicciones_ensamble.csv").read_text(encoding="utf-8").splitlines()[1:]]
    assert len(ids) == 382 and not any(i.startswith(("TN_", "D_")) for i in ids)


def test_el_binary_label_historico_no_interviene(banco):
    banco.ejecutar()
    with open(banco.salida / "evaluacion_por_imagen.csv", encoding="utf-8", newline="") as fh:
        ev = list(csv.DictReader(fh))
    seis = [x for x in ev if x["original_class"] == "6"]
    assert seis and all(x["label_v2"] == "0" for x in seis)


# ============================================================================ separación estricta inferencia / evaluación
def test_las_filas_de_inferencia_no_llevan_etiquetas_ni_clases(banco):
    banco.ejecutar()
    assert {c[1] for c in banco.espia.llamadas} == {tuple(sorted(E.CLAVES_FILA_INFERENCIA))}
    assert all(c[2] == 382 for c in banco.espia.llamadas)
    for extra in ("label_v2", "original_class", "fold", "group_id", "binary_label"):
        with pytest.raises(GuardError):
            E.exigir_filas_sin_etiquetas([{"image_id": "a", "image_name": "a.png", "image_path": "x", extra: 1}])


def test_las_etiquetas_se_liberan_solo_con_las_predicciones_ya_escritas(banco, monkeypatch):
    visto = []
    original = E.Reservadas.liberar

    def espiar(self, ruta, sha_manifiesto):
        visto.append({n: (banco.salida / n).is_file() for n in ("predicciones_por_miembro.csv", "predicciones_ensamble.csv", "registro_inferencia.json", "resultados_r8.json", "metricas_primarias.csv")})
        return original(self, ruta, sha_manifiesto)
    monkeypatch.setattr(E.Reservadas, "liberar", espiar)
    banco.ejecutar()
    assert len(visto) == 1 and visto[0] == {"predicciones_por_miembro.csv": True, "predicciones_ensamble.csv": True, "registro_inferencia.json": True, "resultados_r8.json": False,
                                            "metricas_primarias.csv": False}
    reg = json.loads((banco.salida / "registro_inferencia.json").read_text(encoding="utf-8"))
    assert reg["etiquetas_usadas_en_inferencia"] is False and reg["estado"] == "predicciones_escritas"
    assert reg["sha256_predicciones_ensamble"] == sha((banco.salida / "predicciones_ensamble.csv").read_bytes())


def test_reservadas_no_se_liberan_sin_registro_ni_dos_veces(tmp_path):
    r = E.Reservadas({"a": 7}, "s")
    with pytest.raises(GuardError, match="antes de escribir"):
        r.liberar(tmp_path / "no.json", "s")
    reg = tmp_path / "registro.json"
    reg.write_text(json.dumps({"estado": "iniciada"}), encoding="utf-8")
    with pytest.raises(GuardError, match="predicciones_escritas"):
        r.liberar(reg, "s")
    reg.write_text(json.dumps({"estado": "predicciones_escritas"}), encoding="utf-8")
    with pytest.raises(GuardError, match="cambió"):
        r.liberar(reg, "otro")
    assert r.liberar(reg, "s")[0] == {"a": 1}
    with pytest.raises(GuardError, match="ya se liberaron"):
        r.liberar(reg, "s")


def test_las_predicciones_no_dependen_de_las_etiquetas(tmp_path, monkeypatch):
    """Mismos ids e imágenes, etiquetas permutadas (mismos recuentos): las predicciones son idénticas byte a byte; solo cambian las métricas."""
    a = Banco(tmp_path / "a", monkeypatch)
    f2 = filas_manifiesto()
    for f in f2:
        if f["image_id"].startswith("T1_"):
            f["original_class"] = 0
        elif f["image_id"] in {f"T0_{k:04d}" for k in range(15)}:
            f["original_class"] = 7
    b = Banco(tmp_path / "b", monkeypatch, f2)
    a.ejecutar()
    b.ejecutar()
    assert (a.salida / "predicciones_ensamble.csv").read_bytes() == (b.salida / "predicciones_ensamble.csv").read_bytes()
    assert (a.salida / "predicciones_por_miembro.csv").read_bytes() == (b.salida / "predicciones_por_miembro.csv").read_bytes()
    ra = json.loads((a.salida / "resultados_r8.json").read_text(encoding="utf-8"))["resultados"]["puntuales"]
    rb = json.loads((b.salida / "resultados_r8.json").read_text(encoding="utf-8"))["resultados"]["puntuales"]
    assert ra["ap"] != rb["ap"]


# ============================================================================ umbral y configuración congelada intactos
def test_umbral_fijo_y_sin_codigo_de_calibracion(banco):
    banco.ejecutar()
    res = json.loads((banco.salida / "resultados_r8.json").read_text(encoding="utf-8"))
    assert res["t_hat"] == "0.08456226376195748" and res["resultados"]["umbral_primario"]["umbral"] == 0.08456226376195748 and res["regla"] == "p >= t_hat"
    fuente = inspect.getsource(E).lower() + inspect.getsource(ES).lower()
    assert "calibrar_umbral" not in fuente and "comprobar_transferencia" not in fuente and "from src.r4_v2 import calibracion" not in fuente


def test_configuracion_congelada_y_declaracion_no_se_modifican(banco):
    antes = {rel: (banco.root / rel).read_bytes() for rel in (TC.C.CONFIG_REL, TC.C.PROTOCOLO_REL, TC.C.MIEMBROS_REL, D.DECLARACION_REL, E.PROTOCOLO_REL, E.PROTOCOLO_HISTORICO_REL, G2_REL)}
    banco.ejecutar()
    assert antes == {rel: (banco.root / rel).read_bytes() for rel in antes}


def test_checkpoints_se_verifican_y_cargan_los_quince(banco):
    banco.ejecutar()
    esperados = {str(banco.root / m["ruta_checkpoint"]) for m in banco.decl["miembros"]}
    assert {str(Path(c)) for c in banco.espia.modelos_cargados} == {str(Path(x)) for x in esperados} and len(banco.espia.modelos_cargados) == 15


# ============================================================================ modo comprobar, CLI y recursos reales del repositorio
def test_comprobar_no_crea_bloqueo_ni_lee_filas_del_test(banco, sin_test):
    p = banco.root / E.MANIFIESTO_TEST_REL
    cuerpo = p.read_bytes().split(b"\n", 1)
    p.write_bytes(cuerpo[0] + b"\n" + b"\xff\xfe filas que no deben leerse\n")
    r = E.comprobar(banco.root, **banco.kw())
    assert r["estado"] == "listo_para_ejecutar" and r["n_miembros"] == 15 and sin_salida(banco)


def test_comprobar_exige_las_mismas_compuertas(banco, sin_test):
    for extra in ({"sha256_protocolo": "0" * 64}, {"sha256_declaracion": "0" * 64}, {"autorizacion": ""}):
        with pytest.raises(GuardError):
            E.comprobar(banco.root, **banco.kw(**extra))
    p = banco.root / E.MANIFIESTO_TEST_REL
    p.write_text("otra,cabecera\n", encoding="utf-8")
    with pytest.raises(GuardError, match="columnas"):
        E.comprobar(banco.root, **banco.kw())


def test_cli_codigos_de_salida(banco):
    comunes = ["--autorizacion", AUT, "--sha256-protocolo", banco.sha_proto, "--sha256-declaracion", banco.sha_decl]
    inyectado = dict(verificar=snapshot_falso(), cargar_modelo=banco.espia.cargar_modelo, device="cpu", entorno=dict(TC.ENTORNO))
    assert E.main(["--modo", "comprobar", *comunes], root=banco.root, **inyectado) == 0
    assert E.main(["--modo", "ejecutar", *comunes], root=banco.root, **inyectado) == U.CODIGO_SALIDA_GUARDA               # sin confirmación literal
    assert sin_salida(banco)
    assert E.main(["--modo", "comprobar"], root=banco.root, **inyectado) == U.CODIGO_SALIDA_GUARDA
    assert E.main(["--modo", "ejecutar", *comunes, "--confirmar-r8", E.CONFIRMACION], root=banco.root, predecir=banco.espia.predecir, **inyectado) == 0
    assert E.main(["--modo", "ejecutar", *comunes, "--confirmar-r8", E.CONFIRMACION], root=banco.root, predecir=banco.espia.predecir, **inyectado) == U.CODIGO_SALIDA_GUARDA


def test_protocolo_real_valido_y_vinculado_a_la_declaracion_real():
    p = json.loads((REPO / E.PROTOCOLO_REL).read_text(encoding="utf-8"))
    E.validar_protocolo(p)
    assert p["sistema"]["declaracion"]["sha256_contenido"] == D.sha256_lf(REPO / D.DECLARACION_REL) == "2dd79df4815b676e1bfe5c880ebb1040548bf57d21de3bbf91652842a8598cf1"
    assert p["historico"]["protocolo_original"]["sha256_contenido"] == D.sha256_lf(REPO / E.PROTOCOLO_HISTORICO_REL)
    assert p["estadistica"]["semilla_bootstrap"] == 20261123 and p["estadistica"]["n_bootstrap"] == 10000 and p["estadistica"]["analisis_por_numero_de_modelos"] is False
    assert "no ejecutable" in p["historico"]["contraste_b2_vs_b0"]["estado"]
    assert "R8.1" in p["iov_originales"] and all(p["iov_originales"][k]["texto"] == E.IOV[k] for k in E.IOV)
    assert "protocolo_r8_congelado.json" not in str(E.PROTOCOLO_REL)                                  # el protocolo histórico se conserva en su ruta; este es un archivo nuevo


def test_el_ejecutor_no_se_ejecuta_solo_al_importar_ni_toca_g2_a_r7():
    fuente = inspect.getsource(E)
    assert "__main__" in fuente and "def main" in fuente
    for escribir in ("write_text(", "write_bytes(", "shutil", "os.remove", "unlink("):
        assert escribir not in fuente.replace("escribir_json_nuevo", "")


# ============================================================================ integridad real con Git en un repositorio temporal
def _git(cwd, *args):
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0"}
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(cwd), *args], capture_output=True, env=env, check=True)


def test_verificar_integridad_real_en_repositorio_temporal(tmp_path):
    if subprocess.run(["git", "--version"], capture_output=True).returncode != 0:
        pytest.skip("git no disponible")
    _git(tmp_path, "init", "-q")
    for rel in E.ARCHIVOS_EJECUTOR + (E.MANIFIESTO_TEST_REL,):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x\n", encoding="utf-8", newline="\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    assert E.verificar_integridad_r8(tmp_path)["bloqueos"] == []
    (tmp_path / "src/r8_v2/estadistica.py").write_text("cambio\n", encoding="utf-8", newline="\n")
    assert [b["codigo"] for b in E.verificar_integridad_r8(tmp_path)["bloqueos"]] == ["arbol_sucio"]
    _git(tmp_path, "checkout", "--", "src/r8_v2/estadistica.py")
    (tmp_path / "src/r8_v2/nuevo.py").write_text("x\n", encoding="utf-8", newline="\n")
    assert "arbol_sucio" in [b["codigo"] for b in E.verificar_integridad_r8(tmp_path)["bloqueos"]]
    (tmp_path / "src/r8_v2/nuevo.py").unlink()
    _git(tmp_path, "rm", "-q", "--cached", E.MANIFIESTO_TEST_REL)
    codigos = [b["codigo"] for b in E.verificar_integridad_r8(tmp_path)["bloqueos"]]
    assert "manifiesto_no_versionado" in codigos


# ============================================================================ lectura de imágenes de test: equivalente a G2, sin etiquetas, restringida a test/images
def _imagen(path: Path, semilla=3):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.default_rng(semilla).integers(0, 255, (10, 12, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_dataset_prueba_equivale_al_de_g2_y_no_lleva_etiquetas(tmp_path):
    pytest.importorskip("torch")
    from src.r4_v2.dataset_lesiones import DatasetLesiones
    for split in ("train", "test"):
        _imagen(tmp_path / E.IMAGENES_DIR / split / "images" / "a.png")
    tr = lambda im: np.asarray(im, dtype=np.float32)  # noqa: E731
    g2 = DatasetLesiones(tmp_path, [{"image_path": f"{E.IMAGENES_DIR}/train/images/a.png", "image_name": "a.png", "label_v2": 0}], tr)
    t = E.DatasetPrueba(tmp_path, [{"image_id": "a", "image_name": "a.png", "image_path": f"{E.IMAGENES_DIR}/test/images/a.png"}], tr)
    a, b = g2[0][0], t[0][0]
    assert len(t) == 1 and a.shape == b.shape == (10, 12, 3) and np.array_equal(a, b) and t[0][1] == 0
    with pytest.raises(GuardError):
        E.DatasetPrueba(tmp_path, [{"image_id": "a", "image_name": "a.png", "image_path": f"{E.IMAGENES_DIR}/test/images/a.png", "label_v2": 1}], tr)


@pytest.mark.parametrize("fila", [{"image_name": "../a.png", "image_path": f"{E.IMAGENES_DIR}/test/images/../a.png"}, {"image_name": "a.png", "image_path": f"{E.IMAGENES_DIR}/train/images/a.png"},
                                  {"image_name": "a.png", "image_path": "data/otro/test/images/a.png"}, {"image_name": "sub/a.png", "image_path": f"{E.IMAGENES_DIR}/test/images/sub/a.png"}])
def test_rutas_de_test_fuera_de_su_directorio_se_rechazan(tmp_path, fila):
    _imagen(tmp_path / E.IMAGENES_DIR / "test" / "images" / "a.png")
    with pytest.raises(GuardError):
        E.resolver_imagen_prueba(tmp_path, {"image_id": "a", **fila})


def test_predictor_usa_las_piezas_de_g2_y_devuelve_la_probabilidad_maligna(tmp_path):
    torch = pytest.importorskip("torch")
    from src.classification.dataloader import create_dataloader
    from src.classification.evaluate import predict
    for k in range(5):
        _imagen(tmp_path / E.IMAGENES_DIR / "test" / "images" / f"i{k}.png", semilla=k)
    filas = [{"image_id": f"i{k}", "image_name": f"i{k}.png", "image_path": f"{E.IMAGENES_DIR}/test/images/i{k}.png"} for k in range(5)]
    tensor = lambda size: (lambda im: torch.from_numpy(np.asarray(im.resize((size, size)), dtype=np.float32)).permute(2, 0, 1) / 255.0)  # noqa: E731
    pre = {"get_val_transforms": tensor, "create_dataloader": create_dataloader, "predict": predict}
    torch.manual_seed(0)
    modelo = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 2))
    cfg = {"receta": {"entrada": {"image_size": 8}, "batch_size": 2, "num_workers": 0}}
    prob = E.predictor_prueba(tmp_path, cfg, 42, pre)(modelo, filas, "cpu")
    esperado = torch.stack([tensor(8)(__import__("PIL.Image", fromlist=["Image"]).open(tmp_path / E.IMAGENES_DIR / "test" / "images" / f"i{k}.png").convert("RGB")) for k in range(5)])
    with torch.no_grad():
        ref = torch.softmax(modelo(esperado), dim=1)[:, 1].numpy().astype(np.float32)
    assert prob.dtype == np.float32 and prob.shape == (5,) and np.allclose(prob, ref, atol=1e-6)
