"""Pruebas focalizadas de G3b (`g3_condiciones_v1`) con datos SINTÉTICOS en directorios temporales. No abren test ni datos reales; sin torch.

ROI real (cv2/torch) y entrenador torch quedan [ND] localmente: se prueban con sustitutos (la prueba de ROI real se omite con skip claro sin cv2).
Comando: py -3.13 -B -m pytest tests/test_r5_v2_g3b.py -q -p no:cacheprovider
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import types
import zlib
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("yaml")
pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from src.r4_v2 import agregacion as AG  # noqa: E402
from src.r4_v2 import bootstrap as BS  # noqa: E402
from src.r4_v2 import common as C4  # noqa: E402
from src.r4_v2 import entrenamiento as EN  # noqa: E402
from src.r4_v2 import integridad as I4  # noqa: E402
from src.r4_v2 import metricas as M  # noqa: E402
from src.r4_v2 import particion as P4  # noqa: E402
from src.r4_v2.common import GuardError  # noqa: E402
from src.r5_v2 import g3b_comun as G  # noqa: E402
from src.r5_v2 import g3b_ejecutar as E  # noqa: E402
from src.r5_v2 import g3b_recortes as R  # noqa: E402
from src.r5_v2 import g3b_reporte_condiciones as RP  # noqa: E402
from src.r5_v2 import metricas_seg as MS  # noqa: E402
from src.r5_v2.mascaras import leer_mascara_binaria  # noqa: E402
from src.r5_v2.segmentacion import COLUMNAS_MANIFIESTO  # noqa: E402
from tests.test_r5_v2 import IMAGENES_DIR, construir_datos  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
HAY_TORCH = importlib.util.find_spec("torch") is not None
HAY_CV2 = importlib.util.find_spec("cv2") is not None and HAY_TORCH
HAY_MPL = importlib.util.find_spec("matplotlib") is not None
requiere_roi = pytest.mark.skipif(not HAY_CV2, reason="cv2/torch no instalados: src/segmentation/roi.py no es importable aquí; ejecutar en Falcon")
AUT, ACEPTA = "CONFIRMADO:D-034", "CONFIRMADO:D-035"
REFERENCIA_FALSA = {"git_blob_id": "r" * 40, "sha256_contenido_blob": "r" * 64}
ARQS = ["resnet50", "densenet121"]


@pytest.fixture(autouse=True)
def _git_aislado(monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")


# ============================================================================ sustitutos y datos sintéticos
def redim_falso(m, tam):
    return np.asarray(Image.fromarray((np.asarray(m) > 0).astype(np.uint8)).resize(tuple(tam), Image.Resampling.NEAREST))


def extraer_falso(registro=None):
    """Imita `roi.extract_roi_from_mask` (sin cv2; sin keep_largest: las máscaras sintéticas tienen una componente) y registra sus argumentos."""
    def f(imagen, mascara, margin_ratio, keep_largest, fallback_to_full_image, min_component_area_fraction):
        if registro is not None:
            registro.append({"margin_ratio": margin_ratio, "keep_largest": keep_largest, "fallback_to_full_image": fallback_to_full_image,
                             "min_component_area_fraction": min_component_area_fraction})
        ancho, alto = imagen.size
        m = redim_falso(mascara, (ancho, alto)) > 0
        if not m.any() or m.sum() / m.size < min_component_area_fraction:
            return types.SimpleNamespace(roi=imagen.copy(), bbox=(0, 0, ancho, alto), used_fallback=True)
        ys, xs = np.where(m)
        l, t, r, b = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        mx, my = int(round(max(1, r - l) * margin_ratio)), int(round(max(1, b - t) * margin_ratio))
        caja = (max(0, l - mx), max(0, t - my), min(ancho, r + mx), min(alto, b + my))
        return types.SimpleNamespace(roi=imagen.crop(caja), bbox=caja, used_fallback=False)
    return f


def verificar_falso(head="a" * 40):
    def f():
        h = {"git_blob_id": "c" * 40, "sha256_contenido_blob": "c" * 64, "sha256_bytes_archivo": "1" * 64}
        return {"git_commit_head": head, "rama": "rama-sintetica", "status_porcelain": [], "status_aux": [], "bloqueos": [], "protocolo_versionado": True,
                "entrada": "modulo", "modulos_en_repo": [], "modulos_virtuales_tolerados": [], "archivos": {G.CONFIG_REL: h, C4.CONFIG_REL: h}}
    return f


def _reescribir_csv(path, fn):
    with open(path, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
        cols = list(filas[0])
    filas = fn(filas)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(filas)


def sintetizar_g3a(root, config_g3a, particion, *, mascaras=None, head="g" * 40):
    """Consolidación de G3a `completa` ya hecha (sin entrenar): 5 folds, máscaras OOF {0,255} = la de referencia salvo `mascaras[id]`."""
    raiz = Path(root) / config_g3a["diseno"]["raiz_resultados"].strip("/") / "completa"
    base_m = Path(root) / config_g3a["diseno"]["raiz_datos_derivados"].strip("/") / "completa"
    todas = []
    for k in config_g3a["diseno"]["folds"]:
        d, md = raiz / f"fold{k}" / "intento_01", base_m / f"fold{k}" / "intento_01"
        d.mkdir(parents=True)
        md.mkdir(parents=True)
        sha_ck, sid = hashlib.sha256(f"ckpt{k}".encode()).hexdigest(), f"unet__s42__f{k}"
        filas = []
        for r in (r for r in particion.filas if r["fold"] == k):
            gt = leer_mascara_binaria(Path(root) / IMAGENES_DIR / r["e1_split"] / "masks" / f"{r['image_id']}_binary.PNG")
            pred = (gt.astype(np.uint8) * 255) if not mascaras or r["image_id"] not in mascaras else mascaras[r["image_id"]]
            ruta = md / f"{r['image_id']}_pred.png"
            Image.fromarray(pred).save(ruta)
            filas.append({"image_id": r["image_id"], "fold": k, "segmentador": sid, "sha256_bytes_checkpoint": sha_ck,
                          "sha256_mascara": C4.sha256_archivo(ruta), "epoca_elegida": 2, "ruta_mascara": ruta.relative_to(root).as_posix()})
        C4.escribir_csv_nuevo(d / f"manifiesto_mascaras_fold{k}.csv", filas, COLUMNAS_MANIFIESTO)
        C4.escribir_json_nuevo(d / "run_config.json", {"modo": "completa", "estado": "completada", "invalida_integridad": False, "segmentador": sid,
                                                       "fold": k, "git_commit_head": head, "sha256_bytes_archivo_particion": particion.sha256_bytes_archivo,
                                                       "epocas": 2, "checkpoint": {"sha256_bytes_archivo": sha_ck}})
        todas += filas
    orden = {r["image_id"]: n for n, r in enumerate(particion.filas)}
    todas.sort(key=lambda m: orden[m["image_id"]])
    C4.escribir_csv_nuevo(raiz / "manifiesto_mascaras_oof.csv", todas, COLUMNAS_MANIFIESTO)
    met = [{c: "" for c in MS.COLUMNAS_METRICAS} | {"image_id": m["image_id"], "fold": m["fold"], "dice": "0.8", "iou": "0.7", "cobertura": "0.9",
                                                     "fallback_pred": "False", "fallback_gt": "False", "pred_vacia": "False"} for m in todas]
    C4.escribir_csv_nuevo(raiz / "metricas_seg_por_imagen.csv", met, MS.COLUMNAS_METRICAS)
    C4.escribir_json_nuevo(raiz / "resumen_metricas_seg.json", {"sha256_texto_manifiesto_oof": C4.sha256_archivo(raiz / "manifiesto_mascaras_oof.csv")})
    return raiz


def armar(root, *, generar=True, mascaras=None):
    config_g3a, config_g2 = construir_datos(root)
    config_g2["diseno"]["arquitecturas"] = list(ARQS)
    config = G.cargar_config_g3b(REPO / G.CONFIG_REL)
    config["diseno"]["arquitecturas"] = list(ARQS)
    particion = P4.cargar_particion(root, config_g2)
    sintetizar_g3a(root, config_g3a, particion, mascaras=mascaras)
    g3a = G.verificar_consolidacion_g3a(root, config_g3a, particion)
    registro = []
    e = types.SimpleNamespace(root=root, config=config, config_g2=config_g2, config_g3a=config_g3a, particion=particion, g3a=g3a, registro=registro,
                              verificar=verificar_falso(), extraer=extraer_falso(registro))
    if generar:
        generar_recortes(e)
        e.indice = R.cargar_indice_recortes(root, config, particion, g3a)
    return e


def generar_recortes(e, **kw):
    args = dict(aceptacion=ACEPTA, verificar=e.verificar, extraer=e.extraer, redimensionar=redim_falso) | kw
    return R.generar_recortes(e.root, e.config, e.config_g3a, e.particion, e.g3a, **args)


@pytest.fixture
def ent(tmp_path):
    return armar(tmp_path)


@pytest.fixture
def ent_sin_recortes(tmp_path):
    return armar(tmp_path, generar=False)


def sintetico(llamadas=None, bump=None):
    """Entrenador sintético con la firma del de G2 (`entrenar_base`): sin torch. `bump(val_path)` desplaza la calidad según el origen de la entrada."""
    def entrenar(ctx):
        if llamadas is not None:
            llamadas.append({"arq": ctx["arq"], "semilla": ctx["semilla"], "fold": ctx["fold"], "train": [r["image_path"] for r in ctx["train"]],
                             "val": [r["image_path"] for r in ctx["val"]], "val_ids": [r["image_id"] for r in ctx["val"]]})
        rng = np.random.default_rng(zlib.crc32(f"{ctx['arq']}{ctx['semilla']}{ctx['fold']}".encode()))
        ids, y = [r["image_id"] for r in ctx["val"]], np.array([r["label_v2"] for r in ctx["val"]])
        n_ep = min(9, ctx["epocas_max"])
        perdida = [0.9, 0.7, 0.5, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5][:n_ep] if n_ep == 9 else [0.9 - 0.1 * i for i in range(n_ep)]
        calidad = (1.4 if ctx["arq"] == "resnet50" else 0.8) + (bump(ctx["val"][0]["image_path"]) if bump else 0.0)
        por = {e: (ids, y, (1 / (1 + np.exp(-(calidad * (2 * y - 1) + rng.normal(0, 1.2, y.size) - 1.5)))).astype(np.float32)) for e in range(1, n_ep + 1)}
        EN.escribir_prob_por_epoca(ctx["dir"] / "probabilidades_por_epoca.csv", por)
        (ctx["dir"] / "checkpoint_mejor.pth").write_bytes(b"ckpt " + bytes(str(ctx["fold"]), "ascii"))
        el = EN.epoca_checkpoint(perdida)
        return {"historial": {"train_loss": [1.0] * n_ep, "val_loss_unweighted": perdida}, "checkpoint": ctx["dir"] / "checkpoint_mejor.pth",
                "prob_recarga": por[el][2] + np.float32(1e-4), "epoca_checkpoint_guardado": el, "pesos_preentrenados": {}, "version_difiere_referencia": False,
                "entorno": {"python": "sintetico"}}
    return entrenar


def bump_condicion(path):
    return 0.3 if "/oof_pred/" in path else (0.6 if "/gt/" in path else 0.0)


# ============================================================================ ROI idéntico, recortes y manifiestos
def test_roi_identico_para_b_y_c_misma_funcion_y_parametros(tmp_path):
    e = armar(tmp_path, generar=False)
    generar_recortes(e)
    cfg = e.config["roi"]
    esperado = {"margin_ratio": cfg["margen"], "keep_largest": cfg["keep_largest"], "fallback_to_full_image": cfg["fallback_a_imagen_completa"],
                "min_component_area_fraction": cfg["min_component_area_fraction"]}
    assert len(e.registro) == 2 * 40 and all(r == esperado for r in e.registro)             # un único conjunto de parámetros, para ambas fuentes
    man = {(m["fuente"], m["image_id"]): m for m in csv.DictReader(open(tmp_path / "data/processed/r5_v2/phash_le8_v1/recortes/manifiesto_recortes.csv", encoding="utf-8"))}
    base = tmp_path / "data/processed/r5_v2/phash_le8_v1/recortes"
    for r in e.particion.filas:
        g, o = man[("gt", r["image_id"])], man[("oof_pred", r["image_id"])]
        assert [g[c] for c in ("izq", "arr", "der", "abj", "fallback", "n_componentes")] == [o[c] for c in ("izq", "arr", "der", "abj", "fallback", "n_componentes")]
        a, b = (np.asarray(Image.open(base / f / f"{r['image_id']}.png")) for f in ("gt", "oof_pred"))
        assert a.shape == b.shape and np.array_equal(a, b)                                    # misma máscara -> mismo recorte, bit a bit


def test_recorte_es_el_de_la_caja_y_manifiesto_con_hashes_y_fallback(tmp_path):
    vacia = np.zeros((20, 24), dtype=np.uint8)
    e = armar(tmp_path, generar=False, mascaras={"3": vacia})
    r = generar_recortes(e)
    base = tmp_path / "data/processed/r5_v2/phash_le8_v1/recortes"
    assert r["dir"] == base.resolve() and not (base.parent / "recortes__en_curso").exists()
    cols = list(csv.DictReader(open(base / "manifiesto_recortes.csv", encoding="utf-8")))
    assert len(cols) == 80 and list(cols[0]) == G.COLUMNAS_RECORTES
    por = {(m["fuente"], m["image_id"]): m for m in cols}
    for m in cols:
        assert hashlib.sha256((tmp_path / m["ruta_recorte"]).read_bytes()).hexdigest() == m["sha256_recorte"]
        assert hashlib.sha256((tmp_path / m["ruta_mascara_origen"]).read_bytes()).hexdigest() == m["sha256_mascara_origen"]
        if m["fuente"] == "oof_pred":
            assert m["segmentador"] == f"unet__s42__f{m['fold']}" and m["sha256_bytes_checkpoint_unet"] == e.g3a["checkpoints"][int(m["fold"])]
        else:
            assert m["segmentador"] == "" and m["sha256_bytes_checkpoint_unet"] == ""
    assert por[("oof_pred", "3")]["fallback"] == "True" and (por[("oof_pred", "3")]["izq"], por[("oof_pred", "3")]["der"]) == ("0", "24")
    assert por[("oof_pred", "3")]["n_componentes"] == "0" and por[("gt", "3")]["fallback"] == "False" and por[("gt", "3")]["n_componentes"] == "1"
    c = por[("gt", "4")]                                                                      # recorte = imagen original recortada con la caja registrada
    with Image.open(tmp_path / IMAGENES_DIR / "validation" / "images" / "4.JPG") as im:
        esperado = np.asarray(im.convert("RGB").crop((int(c["izq"]), int(c["arr"]), int(c["der"]), int(c["abj"]))))
    assert np.array_equal(np.asarray(Image.open(tmp_path / c["ruta_recorte"])), esperado)
    cent = json.loads((base / "recortes_completo.json").read_text(encoding="utf-8"))
    assert cent["sha256_bytes_manifiesto_recortes"] == C4.sha256_archivo(base / "manifiesto_recortes.csv") and cent["n_fallback_por_fuente"] == {"gt": 0, "oof_pred": 1}
    assert cent["aceptacion_mascaras_g3a"] == ACEPTA and "partición provisional" in cent["leyenda"] and cent["roi"] == e.config["roi"]


def test_recortes_no_sobrescriben_ni_reanudan(ent):
    with pytest.raises(GuardError, match="ya existen"):
        generar_recortes(ent)
    (ent.root / "data/processed/r5_v2/phash_le8_v1/recortes").rename(ent.root / "data/processed/r5_v2/phash_le8_v1/recortes__en_curso")
    with pytest.raises(GuardError, match="interrumpida"):
        generar_recortes(ent)


@pytest.mark.parametrize("valor", ["", None, "CONFIRMADO", "CONFIRMADO:D-35", "confirmado:D-035", "CONFIRMADO:AUT-001", " CONFIRMADO:D-035", "PENDIENTE"])
def test_recortes_exigen_la_bandera_y_no_crean_nada(ent_sin_recortes, valor):
    e = ent_sin_recortes
    with pytest.raises(GuardError, match="aceptar-mascaras-g3a"):
        generar_recortes(e, aceptacion=valor)
    with pytest.raises(GuardError, match="aceptar-mascaras-g3a"):
        E.ejecutar(e.root, e.config, e.config_g2, e.config_g3a, modo="recortes", aceptacion=valor, verificar=e.verificar, extraer=e.extraer)
    assert not (e.root / "data/processed/r5_v2/phash_le8_v1/recortes").exists()
    assert not (e.root / "data/processed/r5_v2/phash_le8_v1/recortes__en_curso").exists()


def test_modo_recortes_por_ejecutar_genera_con_la_bandera(ent_sin_recortes, capsys):
    e = ent_sin_recortes
    assert E.ejecutar(e.root, e.config, e.config_g2, e.config_g3a, modo="recortes", aceptacion=ACEPTA, verificar=e.verificar, extraer=e.extraer,
                      redimensionar=redim_falso) == 0
    assert "recortes generados: 80" in capsys.readouterr().out


def _manifiestos(e):
    return e.root / e.config_g3a["diseno"]["raiz_resultados"].strip("/") / "completa"


def test_consolidacion_g3a_incompleta_o_con_hash_distinto_impide_generar(ent_sin_recortes):
    e = ent_sin_recortes
    raiz = _manifiestos(e)
    primera = next((e.root / e.g3a["manifiesto"]["1"]["ruta_mascara"]).parent.iterdir())

    def verificar(): return G.verificar_consolidacion_g3a(e.root, e.config_g3a, e.particion)
    assert verificar()["sha256_manifiesto_oof"]
    casos = []

    def caso(nombre, patron, mutar):
        casos.append((nombre, patron, mutar))

    caso("máscara con otro contenido", "SHA-256 de la máscara", lambda: primera.write_bytes(primera.read_bytes() + b"x"))
    caso("falta una fila (820 -> 819)", "no cubre exactamente", lambda: _reescribir_csv(raiz / "manifiesto_mascaras_oof.csv", lambda f: f[:-1]))
    caso("checkpoint distinto del registrado", "difiere", lambda: _reescribir_csv(raiz / "manifiesto_mascaras_oof.csv", lambda f: [{**f[0], "sha256_bytes_checkpoint": "0" * 64}] + f[1:]))
    caso("resumen de otro manifiesto", "resumen consolidado", lambda: (raiz / "resumen_metricas_seg.json").write_text(json.dumps({"sha256_texto_manifiesto_oof": "0" * 64}), encoding="utf-8"))
    caso("fold 2 con integridad invalidada", "no es una corrida", lambda: _mutar_rc(raiz / "fold2/intento_01/run_config.json", invalida_integridad=True))
    caso("fold 3 sin intento completado", "fold 3", lambda: _mutar_rc(raiz / "fold3/intento_01/run_config.json", estado="fallo_tecnico"))
    caso("otra partición", "no es una corrida", lambda: _mutar_rc(raiz / "fold1/intento_01/run_config.json", sha256_bytes_archivo_particion="0" * 64))
    caso("commits distintos entre folds", "commits o números de época", lambda: _mutar_rc(raiz / "fold4/intento_01/run_config.json", git_commit_head="h" * 40))
    caso("sin manifiesto consolidado", "no existe o está incompleta", lambda: (raiz / "manifiesto_mascaras_oof.csv").unlink())
    for nombre, patron, mutar in casos:
        copia = {p: p.read_bytes() for p in raiz.rglob("*") if p.is_file()} | {primera: primera.read_bytes()}
        mutar()
        with pytest.raises(GuardError, match=patron):
            verificar()
        with pytest.raises(GuardError):                                                       # y el modo recortes no genera nada
            E.ejecutar(e.root, e.config, e.config_g2, e.config_g3a, modo="recortes", aceptacion=ACEPTA, verificar=e.verificar, extraer=e.extraer,
                       redimensionar=redim_falso)
        assert not (e.root / "data/processed/r5_v2/phash_le8_v1/recortes").exists(), nombre
        for p, b in copia.items():
            p.write_bytes(b)
    assert verificar()["sha256_manifiesto_oof"]                                               # restaurado: vuelve a pasar


def _mutar_rc(path, **cambios):
    rc = json.loads(Path(path).read_text(encoding="utf-8"))
    rc.update(cambios)
    Path(path).write_text(json.dumps(rc), encoding="utf-8")


def test_indice_rechaza_manifiesto_o_configuracion_alterados(ent):
    base = ent.root / "data/processed/r5_v2/phash_le8_v1/recortes"
    assert len(ent.indice.filas) == 80 and ent.indice.sha256_manifiesto == C4.sha256_archivo(base / "manifiesto_recortes.csv")
    R.verificar_archivos_de_recortes(ent.root, ent.indice, "gt", ["1", "2"])
    cfg = copy.deepcopy(ent.config)
    cfg["roi"]["margen"] = 0.2
    with pytest.raises(GuardError, match="difiere de la usada"):
        R.cargar_indice_recortes(ent.root, cfg, ent.particion, ent.g3a)
    with pytest.raises(GuardError, match="otra consolidación"):
        R.cargar_indice_recortes(ent.root, ent.config, ent.particion, {**ent.g3a, "sha256_manifiesto_oof": "0" * 64})
    destino = ent.root / ent.indice.filas[("gt", "1")]["ruta_recorte"]
    destino.write_bytes(destino.read_bytes() + b"x")
    with pytest.raises(GuardError, match="SHA-256 del recorte"):
        R.verificar_archivos_de_recortes(ent.root, ent.indice, "gt", ["1"])
    destino.unlink()
    with pytest.raises(GuardError, match="ausente"):
        R.verificar_archivos_de_recortes(ent.root, ent.indice, "gt", ["1"])
    with open(base / "manifiesto_recortes.csv", "a", encoding="utf-8") as f:
        f.write("x\n")
    with pytest.raises(GuardError, match="no coincide con la sentinela"):
        R.cargar_indice_recortes(ent.root, ent.config, ent.particion, ent.g3a)
    (base / "recortes_completo.json").unlink()
    with pytest.raises(GuardError, match="no están generados"):
        R.cargar_indice_recortes(ent.root, ent.config, ent.particion, ent.g3a)


# ============================================================================ resolvedor de entrada por (condición, rol) y no-fuga
def _llamar(e, cond, fold, tmp_path):
    capturado = []
    entrenar = R.crear_entrenador(e.root, e.config, e.indice, cond, entrenar_base=lambda ctx: capturado.append(ctx) or {"ok": True})
    train, val = P4.filas_de_fold(e.particion, fold)
    d = tmp_path / f"d_{cond}_{fold}"
    d.mkdir()
    entrenar({"root": e.root, "fold": fold, "train": train, "val": val, "dir": d, "arq": "resnet50", "semilla": 42})
    return capturado[0], train, val, json.loads((d / "procedencia_g3b.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("cond,tr,va", [("B_hibrida", "gt", "oof_pred"), ("C", "gt", "gt"), ("B_oof", "oof_pred", "oof_pred")])
def test_resolvedor_usa_la_fuente_correcta_por_condicion_y_rol(ent, tmp_path, cond, tr, va):
    ctx, train, val, pr = _llamar(ent, cond, 2, tmp_path)
    assert all(f"/recortes/{tr}/" in r["image_path"] and r["image_name"] == f"{r['image_id']}.png" for r in ctx["train"])
    assert all(f"/recortes/{va}/" in r["image_path"] for r in ctx["val"])
    for nuevo, viejo in zip(ctx["train"] + ctx["val"], train + val):                          # solo cambia el origen de la imagen
        assert {k: v for k, v in nuevo.items() if k not in ("image_path", "image_name")} == {k: v for k, v in viejo.items() if k not in ("image_path", "image_name")}
    assert [r["image_id"] for r in ctx["val"]] == [r["image_id"] for r in val] and ctx["root"] == ent.root
    assert pr["condicion"] == cond and (pr["fuente_roi_entrenamiento"], pr["fuente_roi_validacion"]) == (tr, va)
    assert pr["unet_fold_k"]["segmentador_fold_k"] == ("unet__s42__f2" if va == "oof_pred" else None)
    for r in ctx["train"] + ctx["val"]:                                                       # la guarda K4 de G2 acepta la ruta del recorte
        assert P4.resolver_imagen(ent.root, r).is_file()


def test_sin_fuga_nada_del_fold_k_en_entrenamiento_y_unet_del_propio_fold(ent):
    spec_h, spec_o = ent.config["condiciones"]["B_hibrida"], ent.config["condiciones"]["B_oof"]
    train, val = P4.filas_de_fold(ent.particion, 1)
    info = R.verificar_fuentes_sin_fuga(1, train, val, ent.indice, spec_h, 42)
    assert info["segmentador_fold_k"] == "unet__s42__f1" and info["sha256_bytes_checkpoint_unet_fold_k"] == ent.g3a["checkpoints"][1]
    with pytest.raises(GuardError, match="fuga"):
        R.verificar_fuentes_sin_fuga(1, train + val[:1], val[1:], ent.indice, spec_h, 42)       # imagen del fold k en entrenamiento
    with pytest.raises(GuardError, match="solapamiento"):
        R.verificar_fuentes_sin_fuga(1, train, val + train[:1], ent.indice, spec_h, 42)
    ajeno = next(r for r in ent.particion.filas if r["fold"] == 3)
    with pytest.raises(GuardError, match="fuga"):
        R.verificar_fuentes_sin_fuga(1, [r for r in train if r is not ajeno], val + [ajeno], ent.indice, spec_h, 42)   # validación con imagen de otro fold
    indice = copy.deepcopy(ent.indice)
    indice.filas[("oof_pred", val[0]["image_id"])]["segmentador"] = "unet__s42__f3"            # recorte de validación de otro U-Net
    with pytest.raises(GuardError, match="no procede del U-Net del fold 1"):
        R.verificar_fuentes_sin_fuga(1, train, val, indice, spec_h, 42)
    indice = copy.deepcopy(ent.indice)
    indice.filas[("oof_pred", train[0]["image_id"])]["segmentador"] = "unet__s42__f1"          # B_oof: la máscara de entrenamiento debe ser de SU fold
    with pytest.raises(GuardError, match="propio fold"):
        R.verificar_fuentes_sin_fuga(1, train, val, indice, spec_o, 42)


def test_entrenador_rechaza_recorte_alterado_sin_entrenar(ent, tmp_path):
    train, val = P4.filas_de_fold(ent.particion, 0)
    p = ent.root / ent.indice.filas[("oof_pred", val[0]["image_id"])]["ruta_recorte"]
    p.write_bytes(p.read_bytes() + b"x")
    llamadas = []
    entrenar = R.crear_entrenador(ent.root, ent.config, ent.indice, "B_hibrida", entrenar_base=lambda ctx: llamadas.append(1))
    d = tmp_path / "d"
    d.mkdir()
    with pytest.raises(GuardError, match="SHA-256 del recorte"):
        entrenar({"root": ent.root, "fold": 0, "train": train, "val": val, "dir": d, "arq": "resnet50", "semilla": 42})
    assert not llamadas


# ============================================================================ autorización, humo previo, configuración
@pytest.mark.parametrize("modo", ["humo", "completa"])
@pytest.mark.parametrize("valor", ["", None, "CONFIRMADO", "CONFIRMADO:D-28", "confirmado:D-034", "CONFIRMADO:ASE-001", "PENDIENTE"])
def test_autorizacion_exigida_y_no_se_crea_nada(ent, modo, valor):
    with pytest.raises(GuardError, match="autorizacion-ejecucion"):
        E.ejecutar(ent.root, ent.config, ent.config_g2, ent.config_g3a, modo=modo, condicion="B_hibrida", autorizacion=valor, verificar=ent.verificar,
                   entrenar_base=sintetico(), referencia=REFERENCIA_FALSA)
    assert not (ent.root / "results/r5_v2/phash_le8_v1/g3b_condiciones_v1").exists()
    assert E.construir_parser().parse_args(["--modo", "humo"]).autorizacion_ejecucion == ""


def test_humo_previo_exige_misma_condicion_commit_yaml_y_particion(tmp_path):
    rc = {"modo": "humo", "estado": "completada", "invalida_integridad": False, "condicion": "B_hibrida", "git_commit_head": "a" * 40,
          "sha256_bytes_archivo_particion": "p" * 64, "verificacion_inicio": {"archivos": {G.CONFIG_REL: {"sha256_contenido_blob": "y" * 64}}}}
    d = tmp_path / "x__s42__f0" / "intento_01"
    d.mkdir(parents=True)
    (d / "run_config.json").write_text(json.dumps(rc), encoding="utf-8")
    ok = dict(condicion="B_hibrida", sha256_bytes_archivo_particion="p" * 64, sha256_contenido_blob_config="y" * 64, git_commit_head="a" * 40)
    assert G.exigir_humo_previo(tmp_path, **ok) == d
    for campo, valor in (("condicion", "C"), ("sha256_bytes_archivo_particion", "q" * 64), ("sha256_contenido_blob_config", "z" * 64), ("git_commit_head", "b" * 40)):
        with pytest.raises(GuardError, match="exige un 'humo'"):
            G.exigir_humo_previo(tmp_path, **{**ok, campo: valor})


def test_configuracion_real_valida_y_semantica_fija():
    config = G.cargar_config_g3b(REPO / G.CONFIG_REL)
    assert config["protocolo"]["conformidad_academica_asesor"] == "PENDIENTE" and "provisional" in config["estudio"]["leyenda"]
    assert [config["condiciones"][c]["semillas"] for c in ("B_hibrida", "C")] == [[42], [42]] and config["condiciones"]["B_oof"]["habilitada"] is False
    assert config["roi"]["margen"] == 0.10 and config["roi"]["keep_largest"] is True and config["roi"]["fallback_a_imagen_completa"] is True
    from src.r5_v2 import comun as C3
    G.comprobar_coherencia(config, C4.cargar_config(REPO / config["referencia_g2"]["config"]), C3.cargar_config_g3a(REPO / config["referencia_g3a"]["config"]))
    mala = copy.deepcopy(config)
    mala["condiciones"]["B_hibrida"]["fuente_roi_entrenamiento"] = "oof_pred"                # otra semántica con el mismo nombre
    with pytest.raises(GuardError, match="nombre fija la semántica"):
        G.validar_config(mala)
    for ruta in ("results/r5_v2/test/x/", "results/r4_v2/x/"):
        mala = copy.deepcopy(config)
        mala["diseno"]["raiz_resultados"] = ruta
        with pytest.raises(GuardError, match="K4|bajo results"):
            G.validar_config(mala)
    mala = copy.deepcopy(config)
    mala["diseno"]["raiz_datos_derivados"] = "data/processed/r5_v2/test/recortes/"
    with pytest.raises(GuardError, match="K4"):
        G.validar_config(mala)
    mala = copy.deepcopy(config)
    mala["roi"]["fallback_a_imagen_completa"] = False
    with pytest.raises(GuardError, match="fallback"):
        G.validar_config(mala)
    with pytest.raises(GuardError, match="deshabilitada"):
        G.seleccionar_diseno(config, C4.cargar_config(REPO / config["referencia_g2"]["config"]), "B_oof")


def test_integridad_cubre_todo_el_codigo_g3b_y_la_huella_efectiva_es_invariante_del_plan():
    config = G.cargar_config_g3b(REPO / G.CONFIG_REL)
    cubiertos = set(config["integridad"]["archivos_con_huella"])
    modulos = {f"src/r5_v2/{p.name}" for p in (REPO / "src/r5_v2").glob("g3b_*.py")}
    assert modulos and modulos <= cubiertos and {"src/r5_v2/comun.py", "src/r5_v2/mascaras.py", "src/r4_v2/entrenamiento.py"} <= cubiertos
    h = G.huella_config_efectiva(config, "B_hibrida")
    plan = copy.deepcopy(config)
    plan["diseno"]["arquitecturas"], plan["condiciones"]["B_hibrida"]["semillas"] = ["resnet50"], [42, 43, 44]
    assert G.huella_config_efectiva(plan, "B_hibrida") == h                                    # arquitecturas y semillas son variables del plan
    for cambio in (lambda c: c["roi"].update(margen=0.2), lambda c: c["roi"].update(keep_largest=False),
                   lambda c: c["referencia_g2"].update(sha256_contenido_blob="0" * 64)):
        otra = copy.deepcopy(config)
        cambio(otra)
        assert G.huella_config_efectiva(otra, "B_hibrida") != h
    assert G.huella_config_efectiva(config, "C") != h


def _git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), "-c", "user.email=t@t", "-c", "user.name=t", *args], capture_output=True, text=True, check=True).stdout.strip()


def test_integridad_propia_compatible_con_g2_arbol_limpio_y_pines(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "src/r5_v2").mkdir(parents=True)
    archivos = {"src/r5_v2/a.py": b"x = 1\n", G.PROTOCOLO_REL: b"p\n", "g2.yaml": b"g2: 1\n", "g3a.yaml": b"g3a: 1\n"}
    for rel, b in archivos.items():
        (tmp_path / rel).write_bytes(b)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "c")
    cfg = {"integridad": {"rutas_sin_cambios": ["src/r5_v2"], "archivos_con_huella": list(archivos)},
           "referencia_g2": {"config": "g2.yaml", "sha256_contenido_blob": hashlib.sha256(b"g2: 1\n").hexdigest()},
           "referencia_g3a": {"config": "g3a.yaml", "sha256_contenido_blob": hashlib.sha256(b"g3a: 1\n").hexdigest()}}
    ok = G.verificar_integridad_g3b(tmp_path, cfg)
    assert ok["git_commit_head"] == _git(tmp_path, "rev-parse", "HEAD") and not ok["bloqueos"] and ok["protocolo_versionado"] is True
    assert set(ok["archivos"]["g2.yaml"]) >= {"git_blob_id", "sha256_contenido_blob", "sha256_bytes_archivo"}
    assert I4.exigir_integridad(ok) is ok and I4.comparar_instantaneas(ok, G.verificar_integridad_g3b(tmp_path, cfg)) == []   # contrato con r4_v2.entrenamiento
    (tmp_path / "src/r5_v2/a.py").write_bytes(b"x = 2\n")
    sucio = G.verificar_integridad_g3b(tmp_path, cfg)
    assert [b["codigo"] for b in sucio["bloqueos"]] == ["arbol_sucio"] and I4.comparar_instantaneas(ok, sucio)
    with pytest.raises(GuardError, match="arbol_sucio"):
        G.exigir_integridad(sucio)
    _git(tmp_path, "checkout", "--", "src/r5_v2/a.py")
    cfg["referencia_g3a"]["sha256_contenido_blob"] = "0" * 64
    assert [b["codigo"] for b in G.verificar_integridad_g3b(tmp_path, cfg)["bloqueos"]] == ["yaml_referencia_g3a_distinto_del_fijado"]


# ============================================================================ comparabilidad con A
def _escribir_a(raiz, config_g2, particion, plan, **cambios):
    for p in plan:
        d = raiz / p["id"] / "intento_01"
        d.mkdir(parents=True)
        rc = {"modo": "completa", "condicion": "A", "estado": "completada", "invalida_integridad": False, "sha256_bytes_archivo_particion": particion.sha256_bytes_archivo,
              "referencia_pesos": {"sha256_contenido_blob": REFERENCIA_FALSA["sha256_contenido_blob"]}, "receta": config_g2["receta"],
              "epocas_max": config_g2["receta"]["epochs_max"], "mapeo": "mapeo v2", "particion": "phash_le8_v1",
              "verificacion_inicio": {"archivos": {C4.CONFIG_REL: {"sha256_contenido_blob": "c" * 64}}}} | cambios
        (d / "run_config.json").write_text(json.dumps(rc), encoding="utf-8")


def test_comparabilidad_con_A_misma_particion_receta_y_pesos(ent_sin_recortes, tmp_path):
    e = ent_sin_recortes
    g2d = G.config_g2_derivada(e.config_g2, e.config, "C", ["resnet50"], [42])
    plan = G.plan_de_corridas(g2d)[:2]
    raiz = tmp_path / "A"
    _escribir_a(raiz, e.config_g2, e.particion, plan)
    ok = G.comparar_con_A(e.root, e.config_g2, e.particion, REFERENCIA_FALSA, plan, raiz, exigir=True)
    assert ok["estado"] == "ok" and ok["n_comparadas"] == 2
    receta = copy.deepcopy(e.config_g2["receta"])
    receta["learning_rate"] = 0.001
    for cambios, parte in (({"sha256_bytes_archivo_particion": "0" * 64}, "particion"), ({"referencia_pesos": {"sha256_contenido_blob": "0" * 64}}, "referencia_pesos"),
                           ({"receta": receta}, "receta"), ({"condicion": "B"}, "condición A")):
        mala = tmp_path / f"A_{parte.replace(' ', '_')}"
        _escribir_a(mala, e.config_g2, e.particion, plan, **cambios)
        with pytest.raises(GuardError, match=parte):
            G.comparar_con_A(e.root, e.config_g2, e.particion, REFERENCIA_FALSA, plan, mala, exigir=True)
    vacio = tmp_path / "sin_A"
    with pytest.raises(GuardError, match="faltan 2 corridas A"):
        G.comparar_con_A(e.root, e.config_g2, e.particion, REFERENCIA_FALSA, plan, vacio, exigir=True)
    assert G.comparar_con_A(e.root, e.config_g2, e.particion, REFERENCIA_FALSA, plan, vacio, exigir=False)["estado"] == "no_disponible"   # humo sin A a mano


# ============================================================================ flujo integrado sintético (humo -> completa) y reporte
@pytest.fixture(scope="module")
def banco(tmp_path_factory):
    root = tmp_path_factory.mktemp("banco_g3b")
    e = armar(root)
    g2 = e.config_g2
    raiz_a = root / g2["diseno"]["raiz_resultados"].strip("/") / "completa"
    for a in ARQS:                                                                            # A de G2 sintético (2 arquitecturas x 3 semillas x 5 folds)
        for s in g2["diseno"]["semillas"]:
            for k in g2["diseno"]["folds"]:
                r = EN.ejecutar_corrida(root, g2, e.particion, modo="completa", arq=a, semilla=s, fold=k, raiz_modo=raiz_a, verificar=verificar_falso(),
                                        entrenar=sintetico(), referencia=REFERENCIA_FALSA, atestaciones={"autorizacion_ejecucion": "SINTETICA"})
                assert r["estado"] == "completada", r
    e.llamadas = []
    kw = dict(autorizacion=AUT, verificar=e.verificar, entrenar_base=sintetico(e.llamadas, bump_condicion), referencia=REFERENCIA_FALSA)
    e.kw = kw
    with pytest.raises(GuardError, match="exige un 'humo'"):                                  # completa sin humo previo: no se crea nada
        E.ejecutar(root, e.config, g2, e.config_g3a, modo="completa", condicion="B_hibrida", **kw)
    assert not (root / "results/r5_v2/phash_le8_v1/g3b_condiciones_v1/B_hibrida/completa").exists() and not e.llamadas
    for cond in ("B_hibrida", "C"):
        assert E.ejecutar(root, e.config, g2, e.config_g3a, modo="humo", condicion=cond, **kw) == 0
    assert E.ejecutar(root, e.config, g2, e.config_g3a, modo="completa", condicion="B_hibrida", semillas=[42, 43], **kw) == 0
    assert E.ejecutar(root, e.config, g2, e.config_g3a, modo="completa", condicion="C", **kw) == 0
    e.base = root / "results/r5_v2/phash_le8_v1/g3b_condiciones_v1"
    return e


def test_humo_no_cuenta_y_deja_artefactos_de_g2_mas_procedencia(banco):
    d = banco.base / "B_hibrida/humo/resnet50__s42__f0/intento_01"
    assert {p.name for p in d.iterdir()} >= {"run_config.json", "predicciones_oof.csv", "probabilidades_por_epoca.csv", "checkpoint_mejor.pth",
                                              "procedencia_g3b.json", "sello_g3b.json", "historial.json"}
    rc = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
    assert rc["modo"] == "humo" and rc["condicion"] == "B_hibrida" and rc["estado"] == "completada" and rc["epocas_max"] == 1 and not rc["invalida_integridad"]
    assert rc["sha256_bytes_archivo_particion"] == banco.particion.sha256_bytes_archivo and rc["referencia_pesos"]["sha256_contenido_blob"] == "r" * 64
    sello = json.loads((d / "sello_g3b.json").read_text(encoding="utf-8"))
    assert sello["sha256_bytes_run_config"] == C4.sha256_archivo(d / "run_config.json") and sello["comparabilidad_con_A"]["estado"] == "ok"
    assert "partición provisional" in sello["leyenda"]


def test_completa_estructura_resumible_y_receta_de_g2(banco):
    llamadas_antes = len(banco.llamadas)
    completa = banco.base / "B_hibrida/completa"
    assert sorted(p.name for p in completa.iterdir()) == sorted(f"{a}__s{s}__f{k}" for a in ARQS for s in (42, 43) for k in range(5))
    assert sorted(p.name for p in (banco.base / "C/completa").iterdir()) == sorted(f"{a}__s42__f{k}" for a in ARQS for k in range(5))
    rc = json.loads((completa / "resnet50__s42__f1/intento_01/run_config.json").read_text(encoding="utf-8"))
    assert rc["receta"] == banco.config_g2["receta"] and rc["mapeo"] == "mapeo v2" and rc["condicion"] == "B_hibrida"           # misma receta numérica que A
    assert E.ejecutar(banco.root, banco.config, banco.config_g2, banco.config_g3a, modo="completa", condicion="B_hibrida", semillas=[42, 43], **banco.kw) == 0
    assert len(banco.llamadas) == llamadas_antes                                              # corridas completadas: se omiten, no se reentrena
    assert not list(completa.glob("*/intento_02"))


def test_entradas_del_clasificador_por_condicion_en_el_flujo(banco):
    por = {}
    for ll in banco.llamadas:
        k = "B" if all("/gt/" in p for p in ll["train"]) and all("/oof_pred/" in p for p in ll["val"]) else "C" if all("/gt/" in p for p in ll["train"] + ll["val"]) else "?"
        por.setdefault(k, []).append(ll)
    assert set(por) == {"B", "C"} and len(por["B"]) == 1 + 20 and len(por["C"]) == 1 + 10     # humo + completa de cada condición
    pf = {r["image_id"]: r["fold"] for r in banco.particion.filas}
    for ll in banco.llamadas:
        assert {pf[i] for i in ll["val_ids"]} == {ll["fold"]}                                 # validación = fold k completo; nada del fold k entrena
        ent_ids = {Path(p).stem for p in ll["train"]}
        assert not ({pf[i] for i in ent_ids} & {ll["fold"]}) and not (ent_ids & set(ll["val_ids"]))


def test_omitir_exige_la_misma_configuracion_efectiva(banco):
    d = banco.base / "C/completa/resnet50__s42__f0/intento_01"
    pr = json.loads((d / "procedencia_g3b.json").read_text(encoding="utf-8"))
    with pytest.raises(GuardError, match="otra partición, otros recortes u otra configuración"):
        E._coherente_con_previo(d, sha_particion="0" * 64, sha_manifiesto=pr["sha256_bytes_manifiesto_recortes"], huella_efectiva=pr["huella_config_efectiva"], id_corrida="x")
    with pytest.raises(GuardError, match="otra partición"):
        E._coherente_con_previo(d, sha_particion=banco.particion.sha256_bytes_archivo, sha_manifiesto=pr["sha256_bytes_manifiesto_recortes"], huella_efectiva="0" * 64, id_corrida="x")
    E._coherente_con_previo(d, sha_particion=banco.particion.sha256_bytes_archivo, sha_manifiesto=pr["sha256_bytes_manifiesto_recortes"],
                            huella_efectiva=pr["huella_config_efectiva"], id_corrida="x")


def _reporte(banco, salida, **kw):
    args = dict(condiciones=["B_hibrida", "C"], semillas_por_condicion={"B_hibrida": [42, 43], "C": [42]}, verificar=banco.verificar, salida=salida,
                n_remuestreos=300, con_figuras=False, g3a_info=banco.g3a) | kw
    return RP.generar_reporte(banco.root, banco.config, banco.config_g2, banco.config_g3a, banco.particion, **args)


def test_reporte_rechaza_corridas_faltantes_y_no_escribe_nada(banco, tmp_path):
    d = banco.base / "C/completa/densenet121__s42__f4"
    d.rename(d.with_name("oculta"))
    try:
        with pytest.raises(GuardError, match="faltan 1 de 10 corridas de C"):
            _reporte(banco, tmp_path / "rep_incompleto")
        with pytest.raises(GuardError, match="faltan"):                                       # semilla pedida que no se corrió
            _reporte(banco, tmp_path / "rep_incompleto2", condiciones=["B_hibrida"], semillas_por_condicion={"B_hibrida": [42, 43, 44]})
    finally:
        d.with_name("oculta").rename(d)
    assert not (tmp_path / "rep_incompleto").exists() and not (tmp_path / "rep_incompleto2").exists()


def test_reporte_tablas_deltas_emparejados_e_identidad(banco, tmp_path):
    meta = _reporte(banco, tmp_path / "rep")
    assert meta["reporte_aceptado"] and not meta["invalida_integridad"] and meta["semillas"] == {"B_hibrida": [42, 43], "C": [42]}
    sal = tmp_path / "rep"
    esperados = {"T-B1_ap_por_condicion_arquitectura.csv", "T-B1_por_semilla.csv", "T-B2_delta_ap.csv", "T-B2_identidad_C-A.csv", "T-B3_segmentacion_por_fold.csv",
                 "T-B4_trazabilidad.csv", "T-B4_unet_por_fold.csv", "bootstrap_resumen.json", "reporte_condiciones_metadatos.json",
                 "indices_b_estratificado_s20261121.npz", "indices_b_estratificado_s20261122.npz"}
    assert esperados <= {p.name for p in sal.iterdir()}
    for t in sal.glob("T-B*.csv"):
        assert t.read_text(encoding="utf-8").splitlines()[0].endswith(G.LEYENDA)              # encabezado + leyenda obligatoria
    t2 = list(csv.DictReader(open(sal / "T-B2_delta_ap.csv", encoding="utf-8").readlines()[1:]))
    por = {(r["comparacion"], r["arquitectura"]): r for r in t2}
    assert set(c for c, _ in por) == {"B_hibrida-A", "C-A", "C-B_hibrida"} and len(t2) == 3 * len(ARQS)
    assert por[("C-A", "resnet50")]["etiqueta_semillas"] == "condicional a la semilla 42" and por[("C-A", "resnet50")]["semillas"] == "42"
    assert por[("B_hibrida-A", "resnet50")]["semillas"] == "42 43" and "condicional a las semillas [42, 43]" == por[("B_hibrida-A", "resnet50")]["etiqueta_semillas"]
    assert por[("C-B_hibrida", "resnet50")]["descriptivo"] == "True" and por[("C-B_hibrida", "resnet50")]["semillas"] == "42"
    for r in t2:
        assert float(r["ic95_inf"]) <= float(r["ic95_sup"])
    ident = list(csv.DictReader(open(sal / "T-B2_identidad_C-A.csv", encoding="utf-8").readlines()[1:]))
    assert len(ident) == len(ARQS) and all(float(r["residuo_maximo_por_remuestreo"]) < 1e-12 and float(r["residuo_puntual"]) < 1e-12 for r in ident)
    for r in ident:
        assert float(r["C-A"]) == pytest.approx(float(r["C-B"]) + float(r["B-A"]), abs=2e-6)
    t3 = list(csv.DictReader(open(sal / "T-B3_segmentacion_por_fold.csv", encoding="utf-8").readlines()[1:]))
    assert [r["fold"] for r in t3] == ["0", "1", "2", "3", "4", "total"] and t3[-1]["n"] == "40" and t3[-1]["dice_media"] == "0.800000"
    t4 = list(csv.DictReader(open(sal / "T-B4_trazabilidad.csv", encoding="utf-8").readlines()[1:]))
    assert len(t4) == 2 + 20 + 10 and {r["comparabilidad_con_A"] for r in t4} == {"ok"} and all(r["estado"] == "completada" for r in t4)
    assert {r["segmentador_fold_k"] for r in t4 if r["condicion"] == "B_hibrida"} == {f"unet__s42__f{k}" for k in range(5)}
    assert {r["segmentador_fold_k"] for r in t4 if r["condicion"] == "C"} == {""}
    assert meta["sha256_bytes_archivo"]["T-B2_delta_ap.csv"] == C4.sha256_archivo(sal / "T-B2_delta_ap.csv")
    with pytest.raises(GuardError, match="ya existe"):                                         # no sobrescribe
        _reporte(banco, tmp_path / "rep")


@pytest.mark.skipif(not HAY_MPL, reason="matplotlib no instalado")
def test_reporte_con_figura_delta_ap(banco, tmp_path):
    meta = _reporte(banco, tmp_path / "rep_fig", condiciones=["C"], semillas_por_condicion={"C": [42]}, con_figuras=True)
    assert meta["figura"] == "F-B1_delta_ap.png" and (tmp_path / "rep_fig/F-B1_delta_ap.png").stat().st_size > 1000


def test_reporte_no_aceptado_si_la_integridad_cambia_al_final(banco, tmp_path):
    estado = {"n": 0}

    def cambia():
        estado["n"] += 1
        r = banco.verificar()
        if estado["n"] > 1:
            r["git_commit_head"] = "z" * 40
        return r
    meta = _reporte(banco, tmp_path / "rep_dif", condiciones=["C"], semillas_por_condicion={"C": [42]}, verificar=cambia)
    assert meta["reporte_aceptado"] is False and meta["invalida_integridad"] and "git_commit_head" in meta["diferencias_inicio_fin"]


def test_reporte_condiciones_no_comparables_con_A_se_rechazan(banco):
    corr_a = AG.cargar_corridas(banco.root / banco.config_g2["diseno"]["raiz_resultados"].strip("/") / "completa", banco.config_g2, banco.particion)
    _, corr_c, _ = RP.cargar_condicion(banco.root, banco.config, banco.config_g2, banco.particion, "C", ARQS, [42])
    RP.exigir_comparables(corr_a, corr_c, "C")
    k = next(iter(corr_c))
    mala = {**corr_c, k: (corr_c[k][0], {**corr_c[k][1], "receta": {**corr_c[k][1]["receta"], "learning_rate": 1.0}})}
    with pytest.raises(GuardError, match="no es comparable con A"):
        RP.exigir_comparables(corr_a, mala, "C")


# ============================================================================ bootstrap emparejado con índices fijos (sin corridas)
def _oof_falso(rng, y, ids, grupos, sesgo, archs=("a",), semillas=(42, 43)):
    prob = {(a, s): np.clip(1 / (1 + np.exp(-(sesgo * (2 * y - 1) + rng.normal(0, 1, y.size)))), 1e-6, 1 - 1e-6).astype(np.float32) for a in archs for s in semillas}
    return types.SimpleNamespace(y=y, image_ids=ids, group_ids=grupos, prob=prob)


def test_bootstrap_emparejado_indices_fijos_e_identidad_c_a():
    rng = np.random.default_rng(1)
    n = 60
    y = (np.arange(n) % 6 == 0).astype(np.int64)
    ids, grupos = [str(i) for i in range(n)], [f"g{i // 2}" for i in range(n)]                # grupos de 2 imágenes (misma clase por construcción)
    y = y.reshape(-1, 2)[:, :1].repeat(2, axis=1).ravel()
    oofs = {"A": _oof_falso(rng, y, ids, grupos, 0.5), "B_hibrida": _oof_falso(rng, y, ids, grupos, 0.9), "C": _oof_falso(rng, y, ids, grupos, 1.3)}
    gpi, clase, nombres = BS.indexar_grupos(grupos, y)
    conteos = BS.generar_conteos(clase, 400, 20261121, estratificado=True)
    assert np.array_equal(conteos, BS.generar_conteos(clase, 400, 20261121, estratificado=True))          # índices b deterministas por semilla
    semillas_c = {"B_hibrida": [42, 43], "C": [42]}
    res = RP.analizar_condiciones(oofs, semillas_c, ["a"], [42, 43, 44], conteos, gpi)
    d = {(x["comparacion"], x["arquitectura"]): x for x in res["deltas"]}
    # recomputo manual con los MISMOS b: ΔAP(B-A) = media_s AP_B(b,s) - media_s AP_A(b,s)
    w = conteos[:, gpi]
    ap = {(c, s): M.calcular(M.preparar(y, oofs[c].prob[("a", s)]), w)["ap"] for c in oofs for s in (42, 43)}
    manual = (ap[("B_hibrida", 42)] + ap[("B_hibrida", 43)]) / 2 - (ap[("A", 42)] + ap[("A", 43)]) / 2
    inf, sup, n_ok = BS.percentiles(manual)
    assert (d[("B_hibrida-A", "a")]["ic_inf"], d[("B_hibrida-A", "a")]["ic_sup"], d[("B_hibrida-A", "a")]["n_remuestreos_validos"]) == (inf, sup, n_ok)
    assert d[("B_hibrida-A", "a")]["semillas"] == [42, 43] and d[("C-A", "a")]["semillas"] == [42] and d[("C-B_hibrida", "a")]["semillas"] == [42]
    assert d[("C-B_hibrida", "a")]["descriptivo"] is True and d[("C-A", "a")]["descriptivo"] is False
    manual_ca = ap[("C", 42)] - ap[("A", 42)]                                                  # C-A condicional a la semilla 42
    assert (d[("C-A", "a")]["ic_inf"], d[("C-A", "a")]["ic_sup"]) == BS.percentiles(manual_ca)[:2]
    (ident,) = res["identidad"]
    assert ident["semillas"] == [42] and ident["residuo_maximo_por_remuestreo"] < 1e-12 and ident["residuo_puntual"] < 1e-12
    assert ident["C-A"] == pytest.approx(ident["C-B"] + ident["B-A"], abs=1e-12)
    assert RP.etiqueta_semillas([42], [42, 43, 44]) == "condicional a la semilla 42" and RP.etiqueta_semillas([42, 43, 44], [42, 43, 44]).startswith("media de las semillas")
    with pytest.raises(GuardError, match="sin su homóloga A"):
        RP.analizar_condiciones(oofs, {"B_hibrida": [42, 99]}, ["a"], [42, 43, 44], conteos, gpi)
    with pytest.raises(GuardError, match="orden de imágenes"):
        RP.analizar_condiciones({**oofs, "C": _oof_falso(rng, 1 - y, ids, grupos, 1.0)}, semillas_c, ["a"], [42, 43, 44], conteos, gpi)


def test_indices_b_se_verifican_contra_el_npz_de_g2_si_existe(tmp_path):
    config = G.cargar_config_g3b(REPO / G.CONFIG_REL)
    g2 = C4.cargar_config(REPO / config["referencia_g2"]["config"])
    clase = np.array([0] * 8 + [1] * 2)
    nombres = [f"g{i}" for i in range(10)]
    cs = {s: BS.generar_conteos(clase, 50, s, True) for s in (g2["bootstrap"]["semilla"], g2["bootstrap"]["semilla_de_comprobacion"])}
    r = RP.verificar_indices_g2(tmp_path, config, g2, cs, nombres)
    assert all(x["estado"].startswith("[ND]") for x in r)                                       # sin .npz de G2: [ND], sigue
    d = tmp_path / config["referencia_g2"]["indices_b_dir"]
    d.mkdir(parents=True)
    for s, c in cs.items():
        et = "estratificado" if s == g2["bootstrap"]["semilla"] else "estratificado_semilla_comprobacion"
        BS.guardar_indices(d / f"indices_b_{et}_s{s}.npz", c, semilla=s, estratificado=True, nombres_de_grupo=nombres)
    assert all(x["estado"].startswith("verificado") for x in RP.verificar_indices_g2(tmp_path, config, g2, cs, nombres))
    s0 = g2["bootstrap"]["semilla"]
    cs[s0] = cs[s0].copy()
    cs[s0][0, 0] += 1
    with pytest.raises(GuardError, match="NO coinciden"):
        RP.verificar_indices_g2(tmp_path, config, g2, cs, nombres)


# ============================================================================ ROI real (cv2/torch)
@requiere_roi
def test_roi_real_equivale_al_sustituto_y_mismo_resultado_para_b_y_c():
    img = Image.fromarray(np.random.RandomState(0).randint(0, 255, (60, 80, 3), dtype=np.uint8))
    m = np.zeros((60, 80), dtype=np.uint8)
    m[10:30, 20:50] = 1
    m[50:52, 70:72] = 1                                                                       # componente menor: keep_largest la descarta
    cfg = G.cargar_config_g3b(REPO / G.CONFIG_REL)["roi"]
    real = R.roi_de_mascara(img, m, cfg)
    pred = R.roi_de_mascara(img, (m * 255).astype(np.uint8), cfg)                              # {0,255} y {0,1} dan el mismo ROI
    assert real["caja"] == pred["caja"] == (17, 8, 53, 32) and real["n_componentes"] == 2 and not real["fallback"]
    vacio = R.roi_de_mascara(img, np.zeros_like(m), cfg)
    assert vacio["fallback"] and vacio["caja"] == (0, 0, 80, 60) and vacio["n_componentes"] == 0
