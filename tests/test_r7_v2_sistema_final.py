"""Pruebas sintéticas (CPU) de `sistema_final` (R7, D-052): manifiesto de 15 miembros, promedio y regla `>=`, umbral exacto, guardas anti-test y anti-desarrollo, preprocesamiento reutilizado
de G2, fidelidad con un modelo diminuto (requiere torch; skip claro si falta), ausencia de métricas del ensamble y de modo de inferencia, leyendas, guardas y determinismo.
No abren test ni datos reales; no entrenan modelos reales ni usan checkpoints reales. Comando: py -3.13 -B -m pytest tests/test_r7_v2_sistema_final.py -q -p no:cacheprovider
"""

from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import re
import shutil
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
from src.r7_v2 import delta_ensamble as D  # noqa: E402
from src.r7_v2 import sistema_final as S  # noqa: E402
from src.r7_v2 import umbral_sistema as U  # noqa: E402
from tests.test_r5_v2 import construir_datos  # noqa: E402
from tests.test_r7_v2_g4 import REF, sintetico, verificar_falso  # noqa: E402
from tests.test_r7_v2_g4 import _git_aislado  # noqa: E402,F401  (fixture autouse)
from tests.test_r7_v2_umbral import AUT, REPO, escribir_r6, leer_csv, verificar_u  # noqa: E402

HAY_TORCH = importlib.util.find_spec("torch") is not None
requiere_torch = pytest.mark.skipif(not HAY_TORCH, reason="torch no instalado: ejecutar en Falcon")
SEMILLAS, FOLDS = (42, 43, 44), range(5)


def sintetico_sf():
    """Entrenador sintético de G2 con un checkpoint DISTINTO por (semilla, fold) (el de G4 solo depende del fold)."""
    base = sintetico()

    def entrenar(ctx):
        r = base(ctx)
        (ctx["dir"] / "checkpoint_mejor.pth").write_bytes(f"ckpt s{ctx['semilla']} f{ctx['fold']}".encode())
        return r
    return entrenar


def armar_sf(root, entrenador=None):
    root = Path(root)
    _, config_g2 = construir_datos(root, por_fold=8, pos_por_fold=2, escribir_mascaras=False)
    config = S.cargar_config(REPO / S.CONFIG_REL)
    particion = P4.cargar_particion(root, config_g2)
    escribir_r6(root, config)
    for rel in (S.PROTOCOLO_REL, *S.ARCHIVOS_CODIGO):                                         # el manifiesto registra su SHA-256: copias en la raíz temporal
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes((REPO / rel).read_bytes())
    cb = S.config_g2_sistema(config_g2, config)
    raiz_a = root / config_g2["diseno"]["raiz_resultados"].strip("/") / "completa"
    for s in SEMILLAS:
        for k in FOLDS:
            r = EN.ejecutar_corrida(root, cb, particion, modo="completa", arq="densenet121", semilla=s, fold=k, raiz_modo=raiz_a, verificar=verificar_falso(),
                                    entrenar=entrenador or sintetico_sf(), referencia=REF, atestaciones={"autorizacion_ejecucion": "SINTETICA"})
            assert r["estado"] == "completada", r
    class Env:
        pass
    e = Env()
    e.root, e.config, e.config_g2, e.particion, e.raiz_a = root, config, config_g2, particion, raiz_a
    return e


def manifiesto(e, destino=None, **kw):
    args = dict(autorizacion=AUT, verificar=verificar_u()) | kw
    return S.ejecutar_manifiesto(e.root, e.config, e.config_g2, destino=destino, **args)


@pytest.fixture(scope="module")
def banco(tmp_path_factory):
    e = armar_sf(tmp_path_factory.mktemp("banco_sf"))
    e.res = manifiesto(e)
    e.dir = e.res["dir"]
    return e


@pytest.fixture
def sf(tmp_path):
    return armar_sf(tmp_path)


def _rc(e, nombre, intento="intento_01"):
    return e.raiz_a / nombre / intento / "run_config.json"


def _mutar_rc(path, **cambios):
    rc = json.loads(Path(path).read_text(encoding="utf-8"))
    rc.update(cambios)
    Path(path).write_text(json.dumps(rc), encoding="utf-8")


# ============================================================================ (1) manifiesto
def test_manifiesto_15_miembros_unicos_con_checkpoint_recalculado(banco):
    enc, filas = leer_csv(banco.dir / "manifiesto_miembros.csv")
    assert enc.startswith("#") and U.LEYENDA in enc and "sin uso de test" in enc and list(filas[0]) == S.COLUMNAS_MANIFIESTO and len(filas) == 15
    assert len({r["id_corrida"] for r in filas}) == 15 and len({r["sha256_bytes_archivo_checkpoint"] for r in filas}) == 15
    assert {(r["arquitectura"], int(r["semilla"]), int(r["fold"])) for r in filas} == {("densenet121", s, k) for s in SEMILLAS for k in FOLDS}
    for r in filas:
        d = banco.raiz_a / r["id_corrida"] / r["intento"]
        rc = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
        assert (banco.root / r["ruta_checkpoint"]).is_file() and r["ruta_checkpoint"].endswith("checkpoint_mejor.pth") and "test" not in Path(r["ruta_checkpoint"]).parts
        assert r["sha256_bytes_archivo_checkpoint"] == C4.sha256_archivo(banco.root / r["ruta_checkpoint"]) == r["sha256_checkpoint_registrado_run_config"] == rc["checkpoint"]["sha256_bytes_archivo"]
        assert r["coincide"] == "True" and r["estado"] == "completada" and r["invalida_integridad"] == "False" and int(r["epoca_elegida"]) == rc["seleccion"]["epoca_elegida"]
        assert r["sha256_bytes_archivo_run_config"] == C4.sha256_archivo(d / "run_config.json") and r["sha256_bytes_archivo_predicciones_oof"] == C4.sha256_archivo(d / "predicciones_oof.csv")
        assert r["sha256_bytes_archivo_particion"] == banco.particion.sha256_bytes_archivo and r["git_commit_head_entrenamiento"] == rc["git_commit_head"]
    m = json.loads((banco.dir / "manifiesto_miembros.json").read_text(encoding="utf-8"))
    assert m["estado"] == S.ESTADO and m["congelado"] is False and m["n_miembros"] == 15 and m["sistema"]["umbral_candidato"]["t_hat"] == S.T_HAT
    assert m["sha256_bytes_archivo_manifiesto_csv"] == C4.sha256_archivo(banco.dir / "manifiesto_miembros.csv")
    assert set(m["codigo"]) == set(S.ARCHIVOS_CODIGO) and all(m["codigo"][r]["sha256_bytes_archivo"] == C4.sha256_archivo(banco.root / r) for r in S.ARCHIVOS_CODIGO)
    assert m["receta_entrada"]["image_size"] == 224 and m["receta_entrada"]["normalizacion"]["mean"] == [0.485, 0.456, 0.406] and m["yaml_g2_pin"] == banco.config["referencia_g2"]["sha256_contenido_blob"]
    assert {"python", "numpy", "torch", "torchvision"} <= set(m["entorno_efectivo"]) and m["recarga_g2"]["alerta_diferencia_maxima_absoluta"] == 0.01
    assert m["sha256_bytes_archivo_particion"] == banco.particion.sha256_bytes_archivo and m["reporte_aceptado"] is True and "entorno_esperado" not in m


def test_manifiesto_bloquea_ante_cualquier_inconsistencia(sf, tmp_path):
    ok = S.construir_manifiesto(sf.root, sf.config, sf.config_g2, sf.particion)
    assert len(ok["miembros"]) == 15
    ck = sf.raiz_a / "densenet121__s43__f2" / "intento_01" / "checkpoint_mejor.pth"
    original = ck.read_bytes()
    ck.write_bytes(original + b"x")                                                         # checkpoint alterado: hash distinto
    with pytest.raises(GuardError, match="hash distinto"):
        S.construir_manifiesto(sf.root, sf.config, sf.config_g2, sf.particion)
    ck.unlink()
    with pytest.raises(GuardError, match="checkpoint ausente"):
        S.construir_manifiesto(sf.root, sf.config, sf.config_g2, sf.particion)
    ck.write_bytes(original)
    p = sf.raiz_a / "densenet121__s42__f1" / "intento_01" / "predicciones_oof.csv"
    p.write_bytes(p.read_bytes() + b" ")
    with pytest.raises(GuardError, match="predicciones_oof.csv no coincide"):
        S.construir_manifiesto(sf.root, sf.config, sf.config_g2, sf.particion)
    p.write_bytes(p.read_bytes()[:-1])
    assert len(S.construir_manifiesto(sf.root, sf.config, sf.config_g2, sf.particion)["miembros"]) == 15          # restaurado: vuelve a pasar
    d = sf.raiz_a / "densenet121__s44__f3"
    d.rename(d.with_name("oculta"))                                                         # miembro faltante
    with pytest.raises(GuardError, match="K7: faltan 1 corridas"):
        S.construir_manifiesto(sf.root, sf.config, sf.config_g2, sf.particion)
    d.with_name("oculta").rename(d)
    shutil.copytree(d / "intento_01", d / "intento_02")                                    # miembro duplicado (dos intentos completados)
    with pytest.raises(GuardError, match="más de un intento completado"):
        S.construir_manifiesto(sf.root, sf.config, sf.config_g2, sf.particion)
    shutil.rmtree(d / "intento_02")
    rc = _rc(sf, "densenet121__s42__f0")
    guardado = rc.read_text(encoding="utf-8")
    for cambio, patron in (({"arquitectura": "resnet50"}, "no es una corrida completada válida de A\\|densenet121"), ({"condicion": "B_hibrida"}, "no es una corrida completada válida"),
                           ({"estado": "fallo_tecnico"}, "K7: faltan 1 corridas"), ({"invalida_integridad": True}, "no es una corrida completa válida"),
                           ({"sha256_bytes_archivo_particion": "0" * 64}, "no son comparables"), ({"modo": "humo"}, "no es una corrida completa válida")):
        _mutar_rc(rc, **cambio)
        with pytest.raises(GuardError, match=patron):
            S.construir_manifiesto(sf.root, sf.config, sf.config_g2, sf.particion)
        rc.write_text(guardado, encoding="utf-8")
    assert len(S.construir_manifiesto(sf.root, sf.config, sf.config_g2, sf.particion)["miembros"]) == 15
    cfg_rota = copy.deepcopy(sf.config_g2)
    cfg_rota["particion"]["csv"]["sha256_bytes_archivo"] = "0" * 64                         # partición distinta de la usada en entrenamiento (K1 de G2)
    with pytest.raises(GuardError):
        P4.cargar_particion(sf.root, cfg_rota)


def test_manifiesto_comparabilidad_rota_entre_corridas_por_configuracion_o_pesos(sf):
    rc = _rc(sf, "densenet121__s43__f4")
    guardado = rc.read_text(encoding="utf-8")
    d = json.loads(guardado)
    d["verificacion_inicio"]["archivos"][C4.CONFIG_REL]["sha256_contenido_blob"] = "9" * 64                     # otra configuración de G2
    rc.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(GuardError, match="no son comparables"):
        S.construir_manifiesto(sf.root, sf.config, sf.config_g2, sf.particion)
    d = json.loads(guardado)
    d["referencia_pesos"] = {"git_blob_id": "x" * 40, "sha256_contenido_blob": "8" * 64}                       # otra referencia de pesos
    rc.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(GuardError, match="no son comparables"):
        S.construir_manifiesto(sf.root, sf.config, sf.config_g2, sf.particion)


# ============================================================================ (2) promedio y regla >=, umbral exacto
def _probs(n=6, seed=0):
    rng = np.random.default_rng(seed)
    return {f"m{i:02d}": rng.random(n).astype(np.float32) for i in range(15)}


def test_promedio_aritmetico_simple_y_regla_mayor_o_igual():
    p = _probs()
    r = S.agregar_y_decidir(p)
    manual = np.mean([np.asarray(v, dtype=np.float64) for v in p.values()], axis=0)
    assert np.array_equal(r["puntuacion"], manual) and r["t_hat"] == float(S.T_HAT) and r["n_miembros"] == 15
    assert np.array_equal(r["positivo"], manual >= float(S.T_HAT))
    otro_orden = dict(reversed(list(p.items())))
    assert np.array_equal(S.agregar_y_decidir(otro_orden)["puntuacion"], r["puntuacion"])                 # sin selección ni ponderación por posición
    sim = {k: np.full(4, 0.25, dtype=np.float32) for k in p}                                               # empate exacto en binario: 15 × 0.25 / 15
    assert S.agregar_y_decidir(sim, t_hat=0.25)["positivo"].all()                                          # p == t_hat es positiva (los empates no se rompen)
    m = float(manual[0])
    assert S.agregar_y_decidir(p, t_hat=repr(m))["positivo"][0]                                           # umbral exactamente igual a la puntuación: positiva
    assert not S.agregar_y_decidir(p, t_hat=repr(float(np.nextafter(m, 1.0))))["positivo"][0]             # una unidad de coma flotante por encima: negativa
    assert S.agregar_y_decidir(p, t_hat=repr(float(np.nextafter(m, 0.0))))["positivo"][0]


def test_agregacion_exige_exactamente_15_miembros_y_probabilidades_validas():
    p = _probs()
    k0 = next(iter(p))
    with pytest.raises(GuardError, match="exactamente 15"):
        S.agregar_y_decidir({k: v for k, v in p.items() if k != k0})
    with pytest.raises(GuardError, match="exactamente 15"):
        S.agregar_y_decidir({**p, "extra": p[k0]})
    for malo in (np.array([0.1, np.nan, 0.3, 0.2, 0.1, 0.2]), np.array([0.1, 1.2, 0.3, 0.2, 0.1, 0.2]), np.array([-0.1, 0.1, 0.3, 0.2, 0.1, 0.2])):
        with pytest.raises(GuardError, match="no finitas|fuera de"):
            S.agregar_y_decidir({**p, k0: malo})
    with pytest.raises(GuardError, match="mismo tamaño"):
        S.agregar_y_decidir({**p, k0: p[k0][:3]})


def test_umbral_candidato_exacto_en_repr_y_en_el_yaml():
    assert repr(float(S.T_HAT)) == S.T_HAT == "0.08456226376195748" and S.t_hat_exacto(S.T_HAT) == 0.08456226376195748
    config = S.cargar_config(REPO / S.CONFIG_REL)
    assert config["sistema"]["umbral_candidato"]["t_hat"] == S.T_HAT and isinstance(config["sistema"]["umbral_candidato"]["t_hat"], str)
    for malo in ("0.080", "0.0845622637619575000", "8.456226376195748e-02", "abc", "nan"):
        with pytest.raises(GuardError):
            S.t_hat_exacto(malo)
    r = config["sistema"]["incertidumbre_umbral_referencia"]
    assert (r["p2_5"], r["mediana"], r["p97_5"], r["n_remuestreos_validos"]) == (0.022556618166466553, 0.08456226376195748, 0.20131495470801988, 10000)
    for cambio in (lambda c: c["sistema"]["umbral_candidato"].update(t_hat="0.0845622637619575"), lambda c: c["sistema"].update(congelado=True),
                   lambda c: c["sistema"].update(n_miembros=14), lambda c: c["sistema"].update(estado="congelado"), lambda c: c["sistema"].update(regla="p > t_hat"),
                   lambda c: c["sistema"].update(arquitectura="resnet50"), lambda c: c["sistema"]["incertidumbre_umbral_referencia"].update(p2_5=0.0)):
        mala = copy.deepcopy(config)
        cambio(mala)
        with pytest.raises(GuardError):
            S.validar_config(mala)


# ============================================================================ K4 y guarda anti-desarrollo; sin CLI de inferencia
def test_guarda_anti_test_y_anti_desarrollo_de_la_biblioteca_de_inferencia(banco):
    S.exigir_filas_de_inferencia([{"image_path": "data/nuevas/imagenes/a.png", "image_name": "a.png"}])
    for malo in ({"image_path": "data/processed/MMOTU/test/images/a.png", "image_name": "a.png"}, {"image_path": "data/x/a.png", "image_name": "TEST"},
                 {"image_path": "Test/a.png"}):
        with pytest.raises(GuardError, match="K4"):
            S.exigir_filas_de_inferencia([malo])
    for k in ("label_v2", "fold", "original_class", "group_id"):
        with pytest.raises(GuardError, match="desarrollo"):
            S.exigir_filas_de_inferencia([{"image_path": "data/x/a.png", "image_name": "a.png", k: 0}])
    m = json.loads((banco.dir / "manifiesto_miembros.json").read_text(encoding="utf-8"))
    with pytest.raises(GuardError, match="K4"):
        S.inferir(banco.root, banco.config, banco.config_g2, m, [{"image_path": "data/test/images/a.png", "image_name": "a.png"}])
    with pytest.raises(GuardError, match="desarrollo"):
        S.inferir(banco.root, banco.config, banco.config_g2, m, [dict(banco.particion.filas[0])])
    with pytest.raises(GuardError, match="prohibido"):
        S.metricas_ensamble_sobre_desarrollo(np.zeros(3), np.zeros(3))                   # guarda explícita: ninguna métrica del ensamble de quince sobre desarrollo
    filas = [{"image_path": "data/x/a.png", "image_name": "a.png"}, {"image_path": "data/x/b.png", "image_name": "b.png"}]
    esperado = {}

    def cargar(ruta, dev):
        return ruta

    def predecir(modelo, f, dev):
        v = np.full(len(f), (zlib.crc32(modelo.parent.parent.name.encode()) % 100) / 100, dtype=np.float32)
        esperado[modelo.parent.parent.name] = v
        return v
    r = S.inferir(banco.root, banco.config, banco.config_g2, m, filas, cargar_modelo=cargar, predecir=predecir)
    assert r["miembros"] == sorted(esperado) and len(esperado) == 15 and np.array_equal(r["puntuacion"], S.agregar_y_decidir(esperado)["puntuacion"])
    ck = banco.root / m["miembros"][0]["ruta_checkpoint"]
    original = ck.read_bytes()
    ck.write_bytes(original + b"x")
    try:
        with pytest.raises(GuardError, match="no coincide con el SHA-256 del manifiesto"):
            S.inferir(banco.root, banco.config, banco.config_g2, m, filas, cargar_modelo=cargar, predecir=predecir)
    finally:
        ck.write_bytes(original)


def test_no_hay_modo_cli_de_inferencia_ni_funcion_expuesta_como_cli():
    assert S.MODOS == ("manifiesto", "fidelidad")
    for modo in ("inferencia", "inferir", "predecir"):
        with pytest.raises(SystemExit):
            S.main(["--modo", modo, "--autorizacion", AUT])
    fuente = inspect.getsource(S.main)
    assert "inferir(" not in fuente and "inferencia" not in fuente.replace("sin modo de inferencia", "")
    assert not [n for n in dir(S) if n.startswith("main_") or n == "cli_inferencia"]


def test_preprocesamiento_es_el_de_g2_y_no_se_reimplementa():
    fuente = inspect.getsource(S)
    pre = inspect.getsource(S.preprocesamiento)
    for nombre in ("from src.classification.transforms import get_val_transforms", "from src.r4_v2.dataset_lesiones import DatasetLesiones",
                   "from src.classification.dataloader import create_dataloader", "from src.classification.evaluate import predict"):
        assert nombre in pre, nombre
    assert not [w for w in ("transforms.Compose", "ToTensor", "Normalize(", "ResizeWithPadding", "def get_val_transforms", "class DatasetLesiones") if w in fuente]
    ent = json.loads(json.dumps(S.config_g2_sistema(construir_g2(), S.cargar_config(REPO / S.CONFIG_REL))["receta"]["entrada"]))
    assert ent["image_size"] == 224 and "relleno" in ent["redimensionado"]                                  # la receta de entrada de G2 (224 con relleno) se registra en el manifiesto


def construir_g2():
    return C4.cargar_config(REPO / "configs" / "r4_v2" / "g2_baseline_v1.yaml")


@pytest.mark.skipif(importlib.util.find_spec("torchvision") is None, reason="torchvision no instalado: la identidad de objetos se comprueba en Falcon [ND]")
def test_preprocesamiento_mismos_objetos_que_g2():
    from src.classification.transforms import get_val_transforms
    from src.r4_v2.dataset_lesiones import DatasetLesiones
    p = S.preprocesamiento()
    assert p["get_val_transforms"] is get_val_transforms and p["DatasetLesiones"] is DatasetLesiones


# ============================================================================ (4) fidelidad con un modelo diminuto (torch)
def _tiny_modelo(semilla, fold):
    import torch
    import torch.nn as nn
    g = torch.Generator().manual_seed(zlib.crc32(f"{semilla}{fold}".encode()))
    m = nn.Linear(3, 2)
    with torch.no_grad():
        m.weight.copy_(torch.randn(2, 3, generator=g) * 5)
        m.bias.copy_(torch.randn(2, generator=g))
    return m


def _tiny_predecir(root):
    def predecir(modelo, filas, device):
        import torch
        from PIL import Image
        out = []
        for f in filas:
            with Image.open(P4.resolver_imagen(root, f)) as im:
                x = torch.tensor(np.asarray(im.convert("RGB"), dtype=np.float32).mean(axis=(0, 1)) / 255.0)
            with torch.no_grad():
                out.append(torch.softmax(modelo(x.unsqueeze(0)), dim=1)[0, 1].item())
        return np.array(out, dtype=np.float32)
    return predecir


def _tiny_cargar(ruta, device):
    import torch
    import torch.nn as nn
    m = nn.Linear(3, 2)
    m.load_state_dict(torch.load(ruta, map_location="cpu", weights_only=True))
    return m


def _tiny_entrenador():
    def entrenar(ctx):
        import torch
        m = _tiny_modelo(ctx["semilla"], ctx["fold"])
        ck = ctx["dir"] / "checkpoint_mejor.pth"
        torch.save(m.state_dict(), ck)
        probs = _tiny_predecir(ctx["root"])(m, ctx["val"], None)
        ids, y = [r["image_id"] for r in ctx["val"]], np.array([r["label_v2"] for r in ctx["val"]])
        perdida = [0.9, 0.7, 0.5, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5]
        EN.escribir_prob_por_epoca(ctx["dir"] / "probabilidades_por_epoca.csv", {e: (ids, y, probs) for e in range(1, 10)})
        return {"historial": {"train_loss": [1.0] * 9, "val_loss_unweighted": perdida}, "checkpoint": ck, "prob_recarga": probs, "epoca_checkpoint_guardado": EN.epoca_checkpoint(perdida),
                "pesos_preentrenados": {}, "version_difiere_referencia": False, "entorno": {"python": "sintetico"}}
    return entrenar


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    pytest.importorskip("torch")
    e = armar_sf(tmp_path_factory.mktemp("tiny_sf"), entrenador=_tiny_entrenador())
    e.man = manifiesto(e)
    e.pred = _tiny_predecir(e.root)
    return e


def fidelidad(e, destino=None, **kw):
    args = dict(autorizacion=AUT, verificar=verificar_u(), cargar_modelo=_tiny_cargar, predecir=e.pred) | kw
    return S.ejecutar_fidelidad(e.root, e.config, e.config_g2, destino=destino, origen=e.man["dir"], **args)


@requiere_torch
def test_fidelidad_reproduce_las_probabilidades_guardadas_de_cada_miembro(tiny):
    r = fidelidad(tiny)
    res = r["resultados"]
    assert r["reporte_aceptado"] and len(res["miembros"]) == 15 and res["alerta_global"] is False and res["tolerancia"] == tiny.config_g2["receta"]["recarga_checkpoint"]["alerta_diferencia_maxima_absoluta"] == 0.01
    for f in res["miembros"]:
        assert f["diferencia_maxima_absoluta"] == 0.0 and f["cambios_de_clase_a_0_5"] == 0 and f["cambios_de_clase_a_t_hat"] == 0 and f["alerta_recarga"] is False
        _, val = P4.filas_de_fold(tiny.particion, f["fold"])
        assert f["n_imagenes"] == len(val)                                                # solo las imágenes de SU propio pliegue
    enc, filas = leer_csv(r["dir"] / "fidelidad_por_miembro.csv")
    assert len(filas) == 15 and U.LEYENDA in enc and "sin métricas del ensamble" in enc and all(x["alerta_recarga"] == "False" for x in filas)
    j = json.loads((r["dir"] / "resumen_fidelidad.json").read_text(encoding="utf-8"))
    assert j["metricas_del_ensamble_de_quince_sobre_desarrollo"] is False and "alerta_diferencia_maxima_absoluta" in j["criterio_de_tolerancia"] and j["t_hat"] == S.T_HAT


@requiere_torch
def test_fidelidad_no_calcula_metricas_ni_agrega_el_ensamble(tiny, monkeypatch):
    def prohibido(*a, **k):
        raise AssertionError("la fidelidad no debe agregar ni evaluar el ensamble")
    monkeypatch.setattr(S, "agregar_y_decidir", prohibido)
    monkeypatch.setattr(S, "metricas_ensamble_sobre_desarrollo", prohibido)
    r = fidelidad(tiny, destino=tiny.root / "results/r7_v2/fid_sin_metricas")
    assert r["reporte_aceptado"]
    texto = " ".join(p.read_text(encoding="utf-8") for p in r["dir"].iterdir()).lower()
    assert not [w for w in ("roc_auc", "\"ap\"", "average_precision", "sensibilidad", "especificidad", "auc") if w in texto.replace("metricas_del_ensamble", "")]
    fuente = inspect.getsource(S.verificar_fidelidad) + inspect.getsource(S.ejecutar_fidelidad)
    assert "agregar_y_decidir" not in fuente and "calibr" not in fuente and "metricas(" not in fuente


@requiere_torch
def test_fidelidad_detecta_checkpoint_corrupto_y_deriva_de_preprocesamiento(tiny, monkeypatch):
    m = json.loads((tiny.man["dir"] / "manifiesto_miembros.json").read_text(encoding="utf-8"))
    ck = tiny.root / m["miembros"][3]["ruta_checkpoint"]
    original = ck.read_bytes()
    corrupto = bytearray(original)
    corrupto[len(corrupto) // 2] ^= 0xFF                                                   # checkpoint corrupto (un byte alterado)
    ck.write_bytes(bytes(corrupto))
    try:
        with pytest.raises(GuardError, match="no coincide con el SHA-256 del manifiesto"):
            fidelidad(tiny, destino=tiny.root / "results/r7_v2/fid_corrupto")
    finally:
        ck.write_bytes(original)
    assert not (tiny.root / "results/r7_v2/fid_corrupto").exists()
    base = tiny.pred

    def deriva(modelo, filas, device):                                                      # p. ej. un preprocesamiento distinto: las probabilidades ya no se reproducen
        return (np.float32(1.0) - base(modelo, filas, device)).astype(np.float32)
    r = fidelidad(tiny, destino=tiny.root / "results/r7_v2/fid_deriva", predecir=deriva)
    assert r["resultados"]["alerta_global"] is True and all(f["alerta_recarga"] and f["diferencia_maxima_absoluta"] > 0.01 for f in r["resultados"]["miembros"])
    assert any(f["cambios_de_clase_a_t_hat"] > 0 for f in r["resultados"]["miembros"])
    assert S.main(["--modo", "fidelidad", "--autorizacion", AUT], root=tiny.root, config=tiny.config, config_g2=tiny.config_g2, verificar=verificar_u(),
                  destino=tiny.root / "results/r7_v2/fid_deriva2", origen=tiny.man["dir"], cargar_modelo=_tiny_cargar, predecir=deriva) == 2
    monkeypatch.setattr(S, "filas_de_fold", lambda part, k: P4.filas_de_fold(part, (k + 1) % 5))                          # un miembro sobre otro pliegue: bloquea
    with pytest.raises(GuardError, match="pliegue propio"):
        fidelidad(tiny, destino=tiny.root / "results/r7_v2/fid_otro_pliegue")


@requiere_torch
def test_fidelidad_exige_manifiesto_coherente(tiny, tmp_path):
    for nombre, mutar, patron in (("sin_json", lambda d: (d / "manifiesto_miembros.json").unlink(), "falta manifiesto"),
                                  ("csv_alterado", lambda d: (d / "manifiesto_miembros.csv").write_bytes((d / "manifiesto_miembros.csv").read_bytes() + b" "), "no es coherente")):
        o = tmp_path / nombre
        shutil.copytree(tiny.man["dir"], o)
        mutar(o)
        with pytest.raises(GuardError, match=patron):
            S.ejecutar_fidelidad(tiny.root, tiny.config, tiny.config_g2, autorizacion=AUT, verificar=verificar_u(), destino=tiny.root / f"results/r7_v2/fid_{nombre}", origen=o,
                                 cargar_modelo=_tiny_cargar, predecir=tiny.pred)
    o = tmp_path / "congelado"
    shutil.copytree(tiny.man["dir"], o)
    j = json.loads((o / "manifiesto_miembros.json").read_text(encoding="utf-8"))
    j["congelado"] = True
    (o / "manifiesto_miembros.json").write_text(json.dumps(j), encoding="utf-8")
    with pytest.raises(GuardError, match="estado distinto"):
        S.ejecutar_fidelidad(tiny.root, tiny.config, tiny.config_g2, autorizacion=AUT, verificar=verificar_u(), destino=tiny.root / "results/r7_v2/fid_congelado", origen=o,
                             cargar_modelo=_tiny_cargar, predecir=tiny.pred)


# ============================================================================ leyendas y coherencia
def test_leyendas_candidato_no_congelado_no_validado_y_sin_validado_afirmativo(banco):
    m = json.loads((banco.dir / "manifiesto_miembros.json").read_text(encoding="utf-8"))
    txt = " ".join(m["leyendas"]) + " " + m["estado"]
    for frase in ("candidato", "NO congelado formalmente", "no validado para el ensamble de quince", "no es el umbral de R8", "no detecta ovarios normales", "sin uso clínico validado",
                  "no se afirma independencia absoluta del test histórico", "S* = 0.90 es un objetivo exploratorio"):
        assert frase in txt, frase
    enc = (banco.dir / "manifiesto_miembros.csv").read_text(encoding="utf-8").splitlines()[0]
    assert U.LEYENDA in enc and "NO congelado formalmente" in enc and "no validado para el ensamble de quince" in enc and "sin uso clínico validado" in enc
    for p in banco.dir.iterdir():
        t = p.read_text(encoding="utf-8")
        for mm in re.finditer("validad[oa]", t):
            ventana = t[max(0, mm.start() - 45):mm.start()].lower()
            assert "no " in ventana or "sin " in ventana, (p.name, ventana)
        assert not [x for x in ("clínicamente útil", "sistema validado", "umbral congelado de", "listo para uso") if x in t]


def test_yaml_y_protocolo_coherentes_y_sin_hash_propio(tmp_path):
    config = S.cargar_config(REPO / S.CONFIG_REL)
    gc, sh = config["protocolo"]["git_commit"], config["protocolo"]["sha256_contenido_blob"]
    assert config["protocolo"]["trazabilidad"] == "D-052" and config["sistema"]["congelado"] is False
    assert (gc is None and sh is None) or (re.fullmatch("[0-9a-f]{40}", gc) and re.fullmatch("[0-9a-f]{64}", sh))
    if gc is None:
        lineas = (REPO / S.CONFIG_REL).read_text(encoding="utf-8").splitlines()
        assert lineas.count("  git_commit: null") == 1 and lineas.count("  sha256_contenido_blob: null") == 1
    protocolo = (REPO / S.PROTOCOLO_REL).read_text(encoding="utf-8")
    assert len(protocolo.splitlines()) <= 70 and S.exigir_protocolo(REPO, config, S.FRASES_PROTOCOLO)
    assert S.T_HAT in protocolo and "0.022556618166466553" in protocolo and "0.20131495470801988" in protocolo
    for i, frase in enumerate(S.FRASES_PROTOCOLO):
        raiz = tmp_path / f"p{i}"
        (raiz / S.PROTOCOLO_REL).parent.mkdir(parents=True)
        (raiz / S.PROTOCOLO_REL).write_text(protocolo.replace(frase, "X"), encoding="utf-8")
        with pytest.raises(GuardError, match="no es coherente"):
            S.exigir_protocolo(raiz, config, S.FRASES_PROTOCOLO)
    for cambio in (lambda c: c["acceso_test"].update(estado="abierto"), lambda c: c["referencia_r6"]["elegida"].update(arquitectura="resnet50"),
                   lambda c: c["integridad"]["archivos_con_huella"].remove("src/classification/transforms.py"), lambda c: c["salidas"].update(fidelidad="results/r7_v2/x"),
                   lambda c: c["integridad"]["archivos_con_huella"].remove(S.PROTOCOLO_REL), lambda c: c.pop("entorno_esperado")):
        mala = copy.deepcopy(config)
        cambio(mala)
        with pytest.raises(GuardError):
            S.validar_config(mala)
    assert S.verificar_integridad is D.verificar_integridad                                  # la misma integridad parametrizada por la ruta del protocolo
    huella = set(config["integridad"]["archivos_con_huella"])
    assert {"src/r7_v2/sistema_final.py", "src/r7_v2/delta_ensamble.py", "src/r7_v2/umbral_sistema.py", *S.ARCHIVOS_CODIGO} <= huella


# ============================================================================ guardas
def test_guarda_r6_autorizacion_k4_no_sobrescribe_y_determinismo(sf):
    ruta = sf.root / sf.config["referencia_r6"]["salida"]
    original = ruta.read_bytes()
    ruta.unlink()
    with pytest.raises(GuardError, match="no existe"):
        manifiesto(sf, destino=sf.root / "results/r7_v2/m_sin_r6")
    for elegida in ({"condicion": "B_hibrida", "arquitectura": "densenet121"}, {"condicion": "A", "arquitectura": "resnet50"}):
        escribir_r6(sf.root, sf.config, elegida)
        with pytest.raises(GuardError, match="R6 eligió"):
            manifiesto(sf, destino=sf.root / "results/r7_v2/m_otra_r6")
        ruta.unlink()
    ruta.write_bytes(original)
    for valor in ("", None, "CONFIRMADO", "CONFIRMADO:D-52", "confirmado:D-052", "CONFIRMADO:AUT-052", " CONFIRMADO:D-052", "PENDIENTE", 52):
        with pytest.raises(GuardError, match="autorizacion"):
            manifiesto(sf, destino=sf.root / "results/r7_v2/m_sin_aut", autorizacion=valor)
    vacio = sf.root / "vacio"
    vacio.mkdir()
    for modo in S.MODOS:
        assert S.main(["--modo", modo], root=vacio) == U.CODIGO_SALIDA_GUARDA == 3 and list(vacio.iterdir()) == []     # sin autorización no se lee ni se escribe
    for malo in (sf.root / "results/r7_v2/test/x", sf.root / "results/r4_v2/x", sf.root / "fuera"):
        with pytest.raises(GuardError):
            manifiesto(sf, destino=malo)
    cfg = copy.deepcopy(sf.config)
    cfg["salidas"]["manifiesto"] = "results/r7_v2/phash_le8_v1/test/sistema"
    with pytest.raises(GuardError):
        S.validar_config(cfg)
    assert not list((sf.root / "results/r7_v2").glob("m_*"))
    a = manifiesto(sf, destino=sf.root / "results/r7_v2/det_a")["dir"]
    b = manifiesto(sf, destino=sf.root / "results/r7_v2/det_b")["dir"]
    assert C4.sha256_archivo(a / "manifiesto_miembros.csv") == C4.sha256_archivo(b / "manifiesto_miembros.csv")
    ja, jb = (json.loads((x / "manifiesto_miembros.json").read_text(encoding="utf-8")) for x in (a, b))
    ja.pop("fecha"), jb.pop("fecha")
    assert ja == jb
    antes = {p.name: C4.sha256_archivo(p) for p in a.iterdir()}
    with pytest.raises(GuardError, match="ya existe"):
        manifiesto(sf, destino=a)
    assert {p.name: C4.sha256_archivo(p) for p in a.iterdir()} == antes                      # nunca sobrescribe
    n = {"i": 0}

    def cambia():
        n["i"] += 1
        r = verificar_u()()
        if n["i"] > 1:
            r["git_commit_head"] = "z" * 40
        return r
    r = manifiesto(sf, destino=sf.root / "results/r7_v2/m_integr", verificar=cambia)
    assert r["reporte_aceptado"] is False and "git_commit_head" in r["diferencias_inicio_fin"]
    n["i"] = 0
    assert S.main(["--modo", "manifiesto", "--autorizacion", AUT], root=sf.root, config=sf.config, config_g2=sf.config_g2, verificar=cambia,
                  destino=sf.root / "results/r7_v2/m_integr2") == U.CODIGO_SALIDA_INTEGRIDAD_INVALIDA == 4
