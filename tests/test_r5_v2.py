"""Pruebas focalizadas de G3a (`g3_segmentacion_v1`) con datos SINTÉTICOS en directorios temporales. No abren test, ni datos reales.

Las pruebas que necesitan torch / cv2 se marcan con skip explícito si no están instaladas (en Falcon corren completas).
Orden de las secciones = (a)...(j) de la especificación + configuración, integridad y resumen.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import io
import json
import os
import subprocess
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("yaml")
PILmod = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from src.r4_v2 import common as C4  # noqa: E402
from src.r4_v2 import particion as P4  # noqa: E402
from src.r4_v2.common import GuardError  # noqa: E402
from src.r5_v2 import comun as C  # noqa: E402
from src.r5_v2 import ejecutar_g3a as E  # noqa: E402
from src.r5_v2 import mascaras as M  # noqa: E402
from src.r5_v2 import metricas_seg as MS  # noqa: E402
from src.r5_v2 import segmentacion as S  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
HAY_TORCH = importlib.util.find_spec("torch") is not None
HAY_CV2 = importlib.util.find_spec("cv2") is not None and HAY_TORCH          # roi.py importa torch y cv2
requiere_torch = pytest.mark.skipif(not HAY_TORCH, reason="torch no está instalado en este entorno: ejecutar en Falcon")
requiere_roi = pytest.mark.skipif(not HAY_CV2, reason="cv2/torch no están instalados: src/segmentation/roi.py no es importable aquí; ejecutar en Falcon")
AUT = "CONFIRMADO:D-028"
IMAGENES_DIR = "data/processed/MMOTU_OTU_2D_experimental"
COLUMNAS = ["image_id", "image_name", "original_class", "label_v2", "class_v2", "e1_split", "image_path", "group_id", "group_size", "fold"]
CLASES_NEG = [0, 1, 2, 3, 4, 6]


@pytest.fixture(autouse=True)
def _git_aislado(monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")


# ============================================================================ datos sintéticos
def configs_reales():
    return C.cargar_config_g3a(REPO / C.CONFIG_REL), C4.cargar_config(REPO / "configs" / "r4_v2" / "g2_baseline_v1.yaml")


def construir_datos(root, *, por_fold=8, pos_por_fold=2, tam=(24, 20), epocas=2, escribir_mascaras=True):
    """Partición sintética de 40 imágenes (5 folds x 8), imágenes y máscaras diminutas, y configuraciones ajustadas (sin tocar los YAML)."""
    root = Path(root)
    config, config_g2 = configs_reales()
    config["receta"]["image_size"], config["receta"]["epochs"] = [16, 16], epocas
    filas, i = [], 0
    for k in range(5):
        for j in range(por_fold):
            i += 1
            pos = j < pos_por_fold
            split = "train" if i % 2 else "validation"
            filas.append({"image_id": str(i), "image_name": f"{i}.JPG", "original_class": "7" if pos else str(CLASES_NEG[i % 6]),
                          "label_v2": "1" if pos else "0", "class_v2": "lesion_maligna" if pos else "lesion_no_maligna",
                          "e1_split": split, "image_path": f"{IMAGENES_DIR}/{split}/images/{i}.JPG", "group_id": f"g{i}",
                          "group_size": "1", "fold": str(k)})
    rng = np.random.RandomState(0)
    for split in ("train", "validation", "test"):                                # `test` solo para comprobar que NUNCA se abre
        (root / IMAGENES_DIR / split / "images").mkdir(parents=True)
        (root / IMAGENES_DIR / split / "masks").mkdir(parents=True)
    for f in filas:
        arr = rng.randint(0, 255, (tam[1], tam[0], 3), dtype=np.uint8)
        Image.fromarray(arr).save(root / f["image_path"])
        if escribir_mascaras:
            m = np.zeros((tam[1], tam[0]), dtype=np.uint8)
            m[5:15, 6:18] = 255 if int(f["image_id"]) % 2 else 1                 # mezcla {0,255} y {0,1}
            Image.fromarray(m).save(root / IMAGENES_DIR / f["e1_split"] / "masks" / f"{f['image_id']}_binary.PNG")
    (root / IMAGENES_DIR / "train" / "masks" / "999_binary.PNG").write_bytes(b"no es una imagen")   # máscara SIN imagen elegible: no se abre
    (root / IMAGENES_DIR / "test" / "masks" / "1_binary.PNG").write_bytes(b"test: no debe abrirse")
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNAS, lineterminator="\n")
    w.writeheader()
    w.writerows(filas)
    datos = buf.getvalue().encode("utf-8")
    ruta = root / config_g2["particion"]["csv"]["ruta"]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_bytes(datos)
    vb = config_g2["particion"]["verificaciones_bloqueantes"]
    vb.update(n_filas=5 * por_fold, n_lesion_no_maligna=5 * (por_fold - pos_por_fold), n_lesion_maligna=5 * pos_por_fold, minimo_positivas_por_fold=1)
    config_g2["particion"]["csv"]["sha256_bytes_archivo"] = hashlib.sha256(datos).hexdigest()
    return config, config_g2


def verificar_falso(head="a" * 40, blob="b" * 64):
    def f():
        return {"git_commit_head": head, "rama": "rama-sintetica", "rutas_sin_cambios": [], "status_porcelain": [], "bloqueos": [],
                "archivos": {C.CONFIG_REL: {"sha256_bytes_archivo": "1" * 64, "sha256_contenido_blob": blob}}, "python": {}}
    return f


def modelo_diminuto():
    import torch.nn as nn
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 1, 1))


def caja_de_prueba(mascara, tam, cfg):
    """roi.py real si hay cv2; si no, una caja mínima equivalente (solo para probar el flujo; la prueba de ROI se omite sin cv2)."""
    if HAY_CV2:
        return MS.caja_roi(mascara, tam, cfg)
    m = np.asarray(Image.fromarray((np.asarray(mascara) > 0).astype(np.uint8)).resize(tuple(tam), Image.Resampling.NEAREST)) > 0
    if not m.any():
        return (0, 0, tam[0], tam[1]), True
    ys, xs = np.where(m)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), False


@pytest.fixture
def datos(tmp_path):
    config, config_g2 = construir_datos(tmp_path)
    return tmp_path, config, config_g2


def filas_820():
    return [{"image_id": str(i), "fold": i % 5, "label_v2": int(i % 22 == 0), "e1_split": "train"} for i in range(820)]


# ============================================================================ (a) roles por fold, sin solapamiento
def test_a_roles_sin_solapamiento_y_cobertura_820():
    filas = filas_820()
    predichas = []
    for k in range(5):
        r = C.roles_de_fold(filas, k, list(range(5)))
        v = (k + 1) % 5
        assert r["fold_validacion_interna"] == v
        ent, val, pred = ({x["image_id"] for x in r[n]} for n in ("entrenamiento", "validacion_interna", "prediccion"))
        assert not (ent & val) and not ((ent | val) & pred)
        assert ent | val | pred == {x["image_id"] for x in filas}
        assert {x["fold"] for x in r["entrenamiento"]} == set(range(5)) - {k, v}
        assert {x["fold"] for x in r["validacion_interna"]} == {v} and {x["fold"] for x in r["prediccion"]} == {k}
        predichas += [x["image_id"] for x in r["prediccion"]]
    assert sorted(predichas) == sorted(x["image_id"] for x in filas) and len(predichas) == 820


def test_a_fuga_detectada():
    filas = filas_820()
    r = C.roles_de_fold(filas, 2, list(range(5)))
    r["entrenamiento"].append(r["prediccion"][0])                      # una imagen del fold k entra al entrenamiento
    with pytest.raises(GuardError, match="fuga"):
        C.verificar_roles(r, filas)
    r = C.roles_de_fold(filas, 2, list(range(5)))
    r["validacion_interna"].append(r["prediccion"][0])
    with pytest.raises(GuardError, match="fuga"):
        C.verificar_roles(r, filas)
    with pytest.raises(GuardError):
        C.fold_validacion_interna(list(range(5)), 7)


def test_a_prediccion_unica_del_segmentador_de_su_fold():
    filas = [{"image_id": str(i), "fold": i % 5} for i in range(20)]
    man = {k: [{"image_id": x["image_id"], "fold": k} for x in filas if x["fold"] == k] for k in range(5)}
    C.verificar_prediccion_unica(filas, man)
    malo = copy.deepcopy(man)
    malo[0].append({"image_id": "1", "fold": 0})                       # imagen del fold 1 predicha por el segmentador del fold 0
    with pytest.raises(GuardError, match="pertenece al fold"):
        C.verificar_prediccion_unica(filas, malo)
    dup = copy.deepcopy(man)
    dup[1].append(dup[1][0])
    with pytest.raises(GuardError, match="más de una vez"):
        C.verificar_prediccion_unica(filas, dup)
    falta = copy.deepcopy(man)
    falta[3].pop()
    with pytest.raises(GuardError, match="faltan"):
        C.verificar_prediccion_unica(filas, falta)


# ============================================================================ (b) rutas de test rechazadas
def test_b_rutas_test_rechazadas(datos, tmp_path):
    root, config, config_g2 = datos
    for ruta in (f"{IMAGENES_DIR}/test/masks", "x/Test/y", "test"):
        with pytest.raises(GuardError, match="K4"):
            C4.exigir_ruta_no_prohibida(ruta)
    with pytest.raises(GuardError, match="K4"):
        M.dir_mascaras(root, IMAGENES_DIR, "test")                     # el directorio `test` existe en el temporal, aun así se rechaza
    with pytest.raises(GuardError, match="K4"):
        M.dir_mascaras(root, "data/processed/test_X/test", "train")
    c = copy.deepcopy(config)
    c["diseno"]["raiz_resultados"] = "results/r5_v2/test/x/"
    with pytest.raises(GuardError, match="K4"):
        C.validar_config(c)
    c = copy.deepcopy(config)
    c["diseno"]["raiz_datos_derivados"] = "data/processed/MMOTU_OTU_2D_roi_pred/"        # artefacto histórico: fuera de r5_v2
    with pytest.raises(GuardError, match="r5_v2"):
        C.validar_config(c)
    # una fila con e1_split=test en la partición la rechaza el cargador de G2 (K4) antes de cualquier lectura
    ruta = root / config_g2["particion"]["csv"]["ruta"]
    texto = ruta.read_text(encoding="utf-8").replace(",train,", ",test,", 1)
    ruta.write_bytes(texto.encode("utf-8"))
    config_g2["particion"]["csv"]["sha256_bytes_archivo"] = hashlib.sha256(texto.encode("utf-8")).hexdigest()
    with pytest.raises(GuardError, match="K4"):
        P4.cargar_particion(root, config_g2)


# ============================================================================ (c) Dice / IoU
def test_c_dice_iou_valores_conocidos_y_vacias():
    gt = np.zeros((4, 4), dtype=np.uint8)
    gt[0, :] = 1                                                        # 4 px
    pred = np.zeros((4, 4), dtype=np.uint8)
    pred[0, :2], pred[1, :2] = 1, 1                                     # 4 px, 2 en común
    d, i = MS.dice_iou(pred, gt)
    assert d == pytest.approx(0.5, abs=1e-6) and i == pytest.approx(2 / 6, abs=1e-6)
    d, i = MS.dice_iou(gt, gt)
    assert d == pytest.approx(1.0) and i == pytest.approx(1.0)
    d, i = MS.dice_iou(np.zeros((4, 4)), np.zeros((4, 4)))              # ambas vacías: 1.0 (convención heredada del notebook)
    assert d == pytest.approx(1.0) and i == pytest.approx(1.0)
    d, i = MS.dice_iou(pred, np.zeros((4, 4)))                          # GT vacía y predicción no vacía: ~0
    assert d < 1e-6 and i < 1e-6
    d, i = MS.dice_iou(np.zeros((4, 4)), gt)                            # predicción vacía y GT no vacía: ~0
    assert d < 1e-6 and i < 1e-6
    d, _ = MS.dice_iou(pred * 255, gt * 255)                            # {0,255} y {0,1} equivalen (> 0)
    assert d == pytest.approx(0.5, abs=1e-6)
    with pytest.raises(GuardError):
        MS.dice_iou(np.zeros((4, 4)), np.zeros((3, 3)))


@requiere_torch
def test_c_dice_torch_igual_al_numpy():
    import torch
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(5, 1, 8, 8, generator=g)
    y = (torch.rand(5, 1, 8, 8, generator=g) > 0.5).float()
    dt, it = S.dice_iou_torch(logits, y, 0.5, 1e-7)
    pred = (torch.sigmoid(logits) >= 0.5).numpy()
    for n in range(5):
        d, i = MS.dice_iou(pred[n, 0], y[n, 0].numpy())
        assert dt[n].item() == pytest.approx(d, abs=1e-5) and it[n].item() == pytest.approx(i, abs=1e-5)


# ============================================================================ (d) empates de épocas
def test_d_empate_conserva_la_epoca_mas_temprana():
    assert MS.epoca_maxima_dice([0.2, 0.7, 0.7, 0.5]) == 2
    assert MS.epoca_maxima_dice([0.5, 0.5, 0.5]) == 1
    assert MS.epoca_maxima_dice([0.1, 0.2, 0.9]) == 3
    with pytest.raises(GuardError):
        MS.epoca_maxima_dice([0.1, float("nan")])
    with pytest.raises(GuardError):
        MS.epoca_maxima_dice([])


# ============================================================================ (e) cobertura y ROI
def test_e_cobertura_con_cajas_explicitas():
    gt = np.zeros((20, 20), dtype=np.uint8)
    gt[1:3, 1:3] = 1                                                    # componente pequeña (4 px)
    gt[10:14, 10:13] = 1                                                # componente grande (12 px)
    assert MS.cobertura(gt, (10, 10, 13, 14)) == pytest.approx(12 / 16)       # solo la mayor componente
    assert MS.cobertura(gt, (0, 0, 20, 20)) == 1.0                            # imagen completa (fallback)
    assert MS.cobertura(gt, (5, 5, 8, 8)) == 0.0
    assert MS.cobertura(np.zeros((5, 5)), (0, 0, 5, 5)) is None               # GT vacía: indefinida
    assert M.areas_de_componentes(gt) == [12, 4]


@requiere_roi
def test_e_roi_real_keep_largest_y_fallback():
    cfg = copy.deepcopy(configs_reales()[0]["roi"])
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[5:10, 5:10] = 1
    gt[50:80, 50:80] = 1
    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[50:80, 50:80] = 1
    pred[5:10, 5:10] = 1
    caja, fb = MS.caja_roi(pred, (100, 100), cfg)                       # keep_largest=True: solo la mayor componente
    assert not fb and caja[0] > 40 and MS.cobertura(gt, caja) == pytest.approx(900 / 925)
    cfg["keep_largest"] = False                                         # el parámetro sale de la configuración
    caja, fb = MS.caja_roi(pred, (100, 100), cfg)
    assert not fb and MS.cobertura(gt, caja) == 1.0
    cfg["keep_largest"] = True
    caja, fb = MS.caja_roi(np.zeros((16, 16), dtype=np.uint8), (100, 80), cfg)      # vacía: fallback a imagen completa
    assert fb and caja == (0, 0, 100, 80) and MS.cobertura(np.ones((80, 100)), caja) == 1.0
    minima = np.zeros((256, 256), dtype=np.uint8)
    minima[3, 3] = 1                                                    # área 1/65536 < 0.001: fallback
    _, fb = MS.caja_roi(minima, (256, 256), cfg)
    assert fb
    caja_ref, fb_ref = MS.caja_roi(gt, (100, 100), cfg)                 # misma función para la referencia (a resolución original)
    assert not fb_ref and MS.cobertura(gt, caja_ref) == pytest.approx(900 / 925)


@requiere_roi
def test_e_componentes_coinciden_con_cv2():
    import cv2
    rng = np.random.RandomState(3)
    for _ in range(20):
        m = (rng.rand(30, 40) > 0.65).astype(np.uint8)
        n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        assert M.areas_de_componentes(m) == sorted((int(a) for a in st[1:, cv2.CC_STAT_AREA]), reverse=True)


# ============================================================================ (f) valores {0,1} / {0,255} y binarización
def test_f_binarizacion_explicita_mayor_que_cero(tmp_path):
    for nombre, valor in (("a", 1), ("b", 255)):
        m = np.zeros((6, 8), dtype=np.uint8)
        m[2:4, 3:6] = valor
        Image.fromarray(m).save(tmp_path / f"{nombre}.png")
    a, b = M.leer_mascara_binaria(tmp_path / "a.png"), M.leer_mascara_binaria(tmp_path / "b.png")
    assert a.dtype == bool and np.array_equal(a, b) and a.sum() == 6
    assert M.clase_de_valores([0, 1]) == "01" and M.clase_de_valores([0, 255]) == "0_255" and M.clase_de_valores([0, 2]) == "otros"
    assert M.clase_de_valores([0]) == "01" and M.clase_de_valores([1, 255]) == "otros"
    chica = M.leer_mascara_binaria(tmp_path / "b.png", (4, 3))           # reescalado con vecino más próximo, luego > 0
    assert chica.shape == (3, 4) and set(np.unique(chica).tolist()) <= {False, True}


@requiere_torch
def test_f_dataset_entrega_mascara_binaria_aunque_el_archivo_sea_0_255(datos):
    root, config, config_g2 = datos
    part = P4.cargar_particion(root, config_g2)
    rutas_i = {r["image_id"]: P4.resolver_imagen(root, r) for r in part.filas}
    rutas_m = M.mascaras_de_elegibles(root, IMAGENES_DIR, part.filas)
    ds = S.DatasetSeg(part.filas[:4], rutas_i, rutas_m, (16, 16), True)
    for n in range(4):
        x = ds[n]
        assert tuple(x["image"].shape) == (3, 16, 16) and tuple(x["mask"].shape) == (1, 16, 16)
        assert set(np.unique(x["mask"].numpy()).tolist()) <= {0.0, 1.0} and x["mask"].sum() > 0
    assert "mask" not in S.DatasetSeg(part.filas[:1], rutas_i, None, (16, 16), False)[0]


# ============================================================================ (g) verificar_datos
def test_g_verificar_datos_ok_y_no_abre_test_ni_no_elegibles(datos, monkeypatch):
    root, config, config_g2 = datos
    part = P4.cargar_particion(root, config_g2)
    abiertas = []
    original = M.leer_mascara_cruda
    monkeypatch.setattr(M, "leer_mascara_cruda", lambda p: (abiertas.append(str(p)), original(p))[1])
    rep = M.verificar_datos(root, config, part.filas)
    assert rep["estado"] == "completada" and not rep["bloqueos"]
    r = rep["resumen"]
    assert r["n_elegibles"] == 40 and r["n_mascaras_presentes"] == 40 and r["n_vacias"] == 0
    assert r["clase_de_valores"] == {"01": 20, "0_255": 20} and r["valores_distintos_globales"] == [0, 1, 255]
    assert r["mascaras_sin_imagen_elegible_no_abiertas"] == {"train": 1, "validation": 0}      # 999_binary.PNG: contada y NO abierta
    assert len(abiertas) == 40 and not any(M.ruta_menciona_test(Path(p).relative_to(Path(root).resolve())) for p in abiertas)
    assert not any(Path(p).name.startswith("999_") for p in abiertas)


def test_g_verificar_datos_detecta_ausente_vacia_tamano_y_valores(datos):
    root, config, config_g2 = datos
    part = P4.cargar_particion(root, config_g2)
    por_id = {r["image_id"]: r for r in part.filas}

    def mascara(i):
        return root / IMAGENES_DIR / por_id[i]["e1_split"] / "masks" / f"{i}_binary.PNG"

    mascara("1").unlink()                                                                         # ausente
    Image.fromarray(np.zeros((20, 24), dtype=np.uint8)).save(mascara("2"))                        # vacía
    Image.fromarray(np.full((10, 10), 255, dtype=np.uint8)).save(mascara("3"))                    # tamaño distinto
    m = np.zeros((20, 24), dtype=np.uint8)
    m[5:10, 5:10] = 2                                                                              # valores no binarios
    Image.fromarray(m).save(mascara("4"))
    m = np.zeros((20, 24), dtype=np.uint8)
    m[0, 0] = 255
    m[10:15, 10:15] = 255                                                                          # 2 componentes (1 px y 25 px)
    Image.fromarray(m).save(mascara("5"))                                                          # válida: solo se reporta
    antes = {i: mascara(i).read_bytes() for i in ("2", "3", "4", "5")}
    rep = M.verificar_datos(root, config, part.filas)
    cods = {(b["codigo"], b["image_id"]) for b in rep["bloqueos"]}
    assert cods == {("mascara_ausente", "1"), ("mascara_vacia", "2"), ("tamano_distinto", "3"), ("valores_no_binarios", "4")}   # reporta TODO
    assert rep["estado"] == "bloqueada" and rep["resumen"]["n_multi_componente"] == 1
    assert {i: mascara(i).read_bytes() for i in antes} == antes                                    # no corrige nada
    with pytest.raises(GuardError, match="ausente"):
        M.mascaras_de_elegibles(root, IMAGENES_DIR, part.filas)


def test_g_modo_verificar_datos_escribe_resultados_nuevos_y_falla_si_bloquea(datos, capsys):
    root, config, config_g2 = datos
    assert E.ejecutar(root, config, config_g2, modo="verificar_datos") == 0                        # sin autorización de entrenamiento
    d1 = root / "results/r5_v2/phash_le8_v1/g3_segmentacion_v1/verificar_datos/intento_01"
    rep = json.loads((d1 / "verificar_datos.json").read_text(encoding="utf-8"))
    assert rep["estado"] == "completada" and rep["resumen"]["n_elegibles"] == 40 and (d1 / "verificar_datos_por_imagen.csv").is_file()
    (root / IMAGENES_DIR / "train" / "masks" / "1_binary.PNG").unlink()
    assert E.ejecutar(root, config, config_g2, modo="verificar_datos") == 2                        # bloquea, pero informa y no pisa lo anterior
    d2 = root / "results/r5_v2/phash_le8_v1/g3_segmentacion_v1/verificar_datos/intento_02"
    assert json.loads((d2 / "verificar_datos.json").read_text(encoding="utf-8"))["estado"] == "bloqueada" and (d1 / "verificar_datos.json").is_file()
    with pytest.raises(GuardError, match="verificar_datos con bloqueos"):                          # humo/completa no entrenan con datos bloqueados
        E.ejecutar(root, config, config_g2, modo="humo", autorizacion=AUT, verificar=verificar_falso(), calcular_caja=caja_de_prueba)
    assert not (root / "results/r5_v2/phash_le8_v1/g3_segmentacion_v1/humo").exists()


# ============================================================================ (h) no sobrescritura y reanudación
def test_h_no_sobrescribe(tmp_path):
    C.directorio_nuevo(tmp_path / "a")
    with pytest.raises(GuardError, match="K5"):
        C.directorio_nuevo(tmp_path / "a")
    C.escribir_json_nuevo(tmp_path / "x.json", {"a": 1})
    with pytest.raises(GuardError, match="K5"):
        C.escribir_json_nuevo(tmp_path / "x.json", {"a": 2})
    assert json.loads((tmp_path / "x.json").read_text())["a"] == 1
    assert C.crear_intento(tmp_path / "f").name == "intento_01" and C.crear_intento(tmp_path / "f").name == "intento_02"
    with pytest.raises(GuardError, match="K5"):
        C.raiz_del_modo(tmp_path, {"diseno": {"raiz_resultados": "results/r5_v2/x/"}}, "../../fuera")


@requiere_torch
def test_h_fallo_crea_intento_nuevo_y_los_folds_completados_se_omiten(datos, capsys):
    root, config, config_g2 = datos
    raiz = root / "results/r5_v2/phash_le8_v1/g3_segmentacion_v1/humo/fold0"
    llamadas = []

    def falla(ctx):
        llamadas.append(ctx["fold"])
        raise RuntimeError("fallo técnico sintético")

    kw = dict(modo="humo", autorizacion=AUT, verificar=verificar_falso(), calcular_caja=caja_de_prueba, crear_modelo=modelo_diminuto)
    assert E.ejecutar(root, config, config_g2, entrenar=falla, **kw) == 2
    rc1 = json.loads((raiz / "intento_01/run_config.json").read_text(encoding="utf-8"))
    assert rc1["estado"] == "fallo_tecnico" and "RuntimeError" in rc1["motivo"]
    assert E.ejecutar(root, config, config_g2, **kw) == 0                                           # intento nuevo, no pisa el anterior
    assert json.loads((raiz / "intento_02/run_config.json").read_text(encoding="utf-8"))["estado"] == "completada"
    assert (raiz / "intento_01/run_config.json").is_file()
    assert E.ejecutar(root, config, config_g2, entrenar=falla, **kw) == 0                           # completado: se omite y no entrena
    assert llamadas == [0] and not (raiz / "intento_03").exists()
    assert "se omite" in capsys.readouterr().out
    # un fold completado con otro commit no se mezcla
    with pytest.raises(GuardError, match="otro commit"):
        E.ejecutar(root, config, config_g2, **{**kw, "verificar": verificar_falso(head="c" * 40)})


# ============================================================================ (i) flujo mínimo sintético y no-fuga
@pytest.fixture(scope="module")
def flujo(tmp_path_factory):
    if not HAY_TORCH:
        pytest.skip("torch no está instalado en este entorno: ejecutar en Falcon")
    root = tmp_path_factory.mktemp("flujo")
    config, config_g2 = construir_datos(root, epocas=2)
    accesos, estado = [], {"fold": None, "fase": None}
    mp = pytest.MonkeyPatch()
    original = S.DatasetSeg.__getitem__

    def espia(self, i):
        accesos.append((estado["fold"], estado["fase"], self.filas[i]["image_id"], self.con_mascara))
        return original(self, i)

    mp.setattr(S.DatasetSeg, "__getitem__", espia)

    def entrenar(ctx):
        estado.update(fold=ctx["fold"], fase="entrenamiento")
        return S.entrenar_fold_torch(ctx)

    def predecir(ctx, ckpt, d):
        estado.update(fold=ctx["fold"], fase="prediccion")
        return S.predecir_fold_torch(ctx, ckpt, d)

    kw = dict(autorizacion=AUT, verificar=verificar_falso(), entrenar=entrenar, predecir=predecir, calcular_caja=caja_de_prueba,
              crear_modelo=modelo_diminuto)
    try:
        with pytest.raises(GuardError, match="exige un 'humo'"):                                   # completa sin humo previo
            E.ejecutar(root, config, config_g2, modo="completa", **kw)
        assert not (root / "results/r5_v2/phash_le8_v1/g3_segmentacion_v1/completa").exists()
        assert E.ejecutar(root, config, config_g2, modo="humo", **kw) == 0
        assert E.ejecutar(root, config, config_g2, modo="completa", **kw) == 0
    finally:
        mp.undo()
    yield {"root": root, "config": config, "config_g2": config_g2, "accesos": accesos,
           "base": root / "results/r5_v2/phash_le8_v1/g3_segmentacion_v1", "datos": root / "data/processed/r5_v2/phash_le8_v1/mascaras_pred_oof",
           "particion": P4.cargar_particion(root, config_g2)}


@requiere_torch
def test_i_no_fuga_en_accesos_a_datos(flujo):
    por_fold = {}
    for r in flujo["particion"].filas:
        por_fold.setdefault(int(r["fold"]), set()).add(r["image_id"])
    acc = [a for a in flujo["accesos"]]
    vistos_pred = []
    for k in range(5):
        v = (k + 1) % 5
        entr = {a[2] for a in acc if a[0] == k and a[1] == "entrenamiento"}
        pred = [a for a in acc if a[0] == k and a[1] == "prediccion"]
        assert entr and not (entr & por_fold[k])                                                  # el fold k no interviene en entrenamiento/selección
        assert entr == set().union(*(por_fold[j] for j in range(5) if j != k))                   # entrenamiento + validación interna = los otros cuatro
        assert pred and {a[2] for a in pred} <= por_fold[k] and not any(a[3] for a in pred)      # se predice SOLO fold k y sin leer su máscara
        vistos_pred += [a[2] for a in pred]
    # humo (fold 0) + completa (fold 0..4): el fold 0 se predice dos veces en total, una por modo; en completa, cada imagen una vez
    assert len(vistos_pred) == 40 + 8


@requiere_torch
def test_i_prediccion_rechaza_checkpoint_de_otra_epoca_o_segmentador(flujo):
    import torch
    ckpt = next((flujo["base"] / "completa").rglob("unet_fold0_best.pth"))
    g = torch.load(ckpt, map_location="cpu", weights_only=True)
    ctx = {"config": flujo["config"], "fold": 0, "segmentador": "unet__s42__f0", "crear_modelo": modelo_diminuto, "roles": {},
           "rutas_imagen": {}, "epoca_esperada": g["epoch"], "val_dice_esperado": g["val_dice"]}
    for cambio, patron in (({"epoca_esperada": g["epoch"] + 1}, "no es el de la época elegida"),
                           ({"val_dice_esperado": g["val_dice"] + 0.1}, "no es el de la época elegida"),
                           ({"segmentador": "unet__s42__f1"}, "es del segmentador")):
        with pytest.raises(GuardError, match=patron):
            S.predecir_fold_torch({**ctx, **cambio}, ckpt, flujo["base"])


@requiere_torch
def test_i_artefactos_por_fold_y_prediccion_con_el_modelo_de_su_fold(flujo):
    import torch
    base, datos_dir, part = flujo["base"] / "completa", flujo["datos"] / "completa", flujo["particion"]
    ids_todos = {r["image_id"] for r in part.filas}
    for k in range(5):
        d = base / f"fold{k}" / "intento_01"
        rc = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
        assert rc["estado"] == "completada" and rc["segmentador"] == f"unet__s42__f{k}" and rc["fold_validacion_interna"] == (k + 1) % 5
        assert rc["git_commit_head"] == "a" * 40 and rc["sha256_bytes_archivo_particion"] == part.sha256_bytes_archivo and rc["semilla"] == 42
        ck = d / f"unet_fold{k}_best.pth"
        assert rc["checkpoint"]["sha256_bytes_archivo"] == hashlib.sha256(ck.read_bytes()).hexdigest()
        roles = list(csv.DictReader(open(d / "ids_por_rol.csv", encoding="utf-8")))
        por = {n: {r["image_id"] for r in roles if r["rol"] == n} for n in ("entrenamiento", "validacion_interna", "prediccion_fold_k")}
        assert not ((por["entrenamiento"] | por["validacion_interna"]) & por["prediccion_fold_k"])
        assert por["entrenamiento"] | por["validacion_interna"] | por["prediccion_fold_k"] == ids_todos
        assert rc["sha256_ids_por_rol"]["prediccion_fold_k"] == C.hash_de_ids(por["prediccion_fold_k"])
        hist = list(csv.DictReader(open(d / f"historial_fold{k}.csv", encoding="utf-8")))
        elegidas = [h for h in hist if h["elegida"] == "True"]
        assert len(hist) == 2 and len(elegidas) == 1 and int(elegidas[0]["epoch"]) == rc["seleccion"]["epoca_elegida"]
        assert int(elegidas[0]["epoch"]) == MS.epoca_maxima_dice([float(h["val_dice"]) for h in hist])
        man = list(csv.DictReader(open(d / f"manifiesto_mascaras_fold{k}.csv", encoding="utf-8")))
        assert {m["image_id"] for m in man} == por["prediccion_fold_k"]
        # la máscara guardada es la que produce el checkpoint del propio fold (recarga de disco)
        modelo = modelo_diminuto()
        modelo.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True)["model_state_dict"])
        modelo.eval()
        m0 = man[0]
        fila = next(r for r in part.filas if r["image_id"] == m0["image_id"])
        with Image.open(flujo["root"] / fila["image_path"]) as im:
            x = np.asarray(im.convert("RGB").resize((16, 16), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        with torch.no_grad():
            esperado = (torch.sigmoid(modelo(torch.from_numpy(x.transpose(2, 0, 1)[None]))) >= 0.5).squeeze().numpy().astype(np.uint8) * 255
        guardada = np.asarray(Image.open(flujo["root"] / m0["ruta_mascara"]))
        assert guardada.shape == (16, 16) and set(np.unique(guardada).tolist()) <= {0, 255} and np.array_equal(guardada, esperado)
        assert hashlib.sha256((flujo["root"] / m0["ruta_mascara"]).read_bytes()).hexdigest() == m0["sha256_mascara"]
        assert f"mascaras_pred_oof/completa/fold{k}/intento_01/" in m0["ruta_mascara"] and datos_dir.is_dir()


@requiere_torch
def test_i_consolidacion_820_como_40_una_vez_cada_imagen(flujo):
    base = flujo["base"] / "completa"
    man = list(csv.DictReader(open(base / "manifiesto_mascaras_oof.csv", encoding="utf-8")))
    assert list(man[0]) == S.COLUMNAS_MANIFIESTO and len(man) == 40 and len({m["image_id"] for m in man}) == 40
    pf = {r["image_id"]: int(r["fold"]) for r in flujo["particion"].filas}
    assert all(m["segmentador"] == f"unet__s42__f{pf[m['image_id']]}" and int(m["fold"]) == pf[m["image_id"]] for m in man)
    met = list(csv.DictReader(open(base / "metricas_seg_por_imagen.csv", encoding="utf-8")))
    assert list(met[0]) == MS.COLUMNAS_METRICAS and len(met) == 40
    assert all(0.0 <= float(m["dice"]) <= 1.0 and m["fallback_gt"] in ("True", "False") and m["gt_vacia"] == "False" for m in met)
    resumen = json.loads((base / "resumen_metricas_seg.json").read_text(encoding="utf-8"))
    est = resumen["estadisticos"]
    assert est["n"] == 40 and est["sin_intervalos_de_confianza"] and set(est["por_fold"]) == {"0", "1", "2", "3", "4"}
    assert "partición provisional" in resumen["leyenda"]
    assert E.consolidar(flujo["root"], flujo["config"], flujo["particion"], base, "completa")["estado"] == "ya_consolidado"   # idempotente, no sobrescribe
    (base / "metricas_seg_por_imagen.csv").write_text("x", encoding="utf-8")
    with pytest.raises(GuardError, match="no coinciden"):
        E.consolidar(flujo["root"], flujo["config"], flujo["particion"], base, "completa")


@requiere_torch
def test_i_unet_real_un_fold_una_epoca(datos):
    """El U-Net real de src/segmentation/unet.py con imágenes de 16x16 (CPU): checkpoint, historial y máscaras del fold."""
    root, config, config_g2 = datos
    rc = E.ejecutar(root, config, config_g2, modo="humo", autorizacion=AUT, epocas_humo=1, humo_fold=3, verificar=verificar_falso(),
                    calcular_caja=caja_de_prueba)
    assert rc == 0
    d = root / "results/r5_v2/phash_le8_v1/g3_segmentacion_v1/humo/fold3/intento_01"
    run = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
    assert run["estado"] == "completada" and run["segmentador"] == "unet__s42__f3" and run["fold_validacion_interna"] == 4
    assert run["checkpoint"]["tamano_bytes"] > 10_000_000
    assert len(list((root / run["directorio_mascaras"]).glob("*_pred.png"))) == 8


def test_i_humo_previo_exige_mismo_commit_yaml_y_particion(tmp_path):
    rc = {"modo": "humo", "estado": "completada", "invalida_integridad": False, "git_commit_head": "a" * 40,
          "sha256_bytes_archivo_particion": "p" * 64, "verificacion_inicio": {"archivos": {C.CONFIG_REL: {"sha256_contenido_blob": "y" * 64}}}}
    d = tmp_path / "fold0" / "intento_01"
    d.mkdir(parents=True)
    (d / "run_config.json").write_text(json.dumps(rc), encoding="utf-8")
    ok = dict(sha256_bytes_archivo_particion="p" * 64, sha256_contenido_blob_config="y" * 64, git_commit_head="a" * 40)
    assert E.exigir_humo_previo(tmp_path, **ok) == d
    for campo, valor in (("sha256_bytes_archivo_particion", "q" * 64), ("sha256_contenido_blob_config", "z" * 64), ("git_commit_head", "b" * 40)):
        with pytest.raises(GuardError, match="exige un 'humo'"):
            E.exigir_humo_previo(tmp_path, **{**ok, campo: valor})


# ============================================================================ (j) autorización exigida
@pytest.mark.parametrize("valor", ["", None, "CONFIRMADO", "confirmado:D-028", "CONFIRMADO:D-28", " CONFIRMADO:D-028", "PENDIENTE",
                                   "CONFIRMADO:ASE-001", "CONFIRMADO:D-0281"])
def test_j_autorizacion_invalida(valor):
    with pytest.raises(GuardError, match="autorizacion-ejecucion"):
        C.exigir_autorizacion(valor)


def test_j_autorizacion_valida_y_auditoria_opcional():
    assert C.exigir_autorizacion("CONFIRMADO:D-028") == "CONFIRMADO:D-028" and C.exigir_autorizacion("CONFIRMADO:AUT-001")
    assert C.validar_auditoria_opcional("") is None and C.validar_auditoria_opcional("CONFIRMADO:AUD-002") == "CONFIRMADO:AUD-002"
    with pytest.raises(GuardError):
        C.validar_auditoria_opcional("sí")


def test_j_sin_roi_importable_se_detiene_antes_de_entrenar(datos, monkeypatch):
    import sys
    root, config, config_g2 = datos
    monkeypatch.setitem(sys.modules, "src.segmentation.roi", None)                      # simula cv2/torch ausentes
    with pytest.raises(GuardError, match="roi.py no es importable"):
        E.ejecutar(root, config, config_g2, modo="humo", autorizacion=AUT, verificar=verificar_falso())
    assert not (root / "results").exists() and not (root / "data/processed/r5_v2").exists()


@pytest.mark.parametrize("modo", ["humo", "completa"])
def test_j_main_sin_autorizacion_se_detiene_sin_crear_nada(datos, modo, capsys):
    root, config, config_g2 = datos
    assert E.main(["--modo", modo], root=root, config=config, config_g2=config_g2, verificar=verificar_falso()) == 3
    assert "autorizacion-ejecucion" in capsys.readouterr().err and not (root / "results").exists() and not (root / "data/processed/r5_v2").exists()


def test_j_main_verificar_datos_no_exige_autorizacion(datos):
    root, config, config_g2 = datos
    assert E.main(["--modo", "verificar_datos"], root=root, config=config, config_g2=config_g2) == 0     # sin Git en el temporal: integridad solo informativa


# ============================================================================ configuración, integridad y resumen
def test_configuracion_real_coherente_con_la_especificacion():
    c, g2 = configs_reales()
    r = c["receta"]
    assert (r["optimizador"], r["learning_rate"], r["perdida"], r["batch_size"], r["image_size"], r["umbral"], r["epochs"], r["semilla"]) == \
        ("Adam", 1e-4, "BCEWithLogitsLoss", 4, [256, 256], 0.5, 50, 42)
    assert r["aumento_de_datos"] is None and r["parada_anticipada"] is None and r["binarizacion_referencia"] == "> 0" and r["eps_dice"] == 1e-7
    assert c["roi"]["margen"] == 0.10 and c["roi"]["min_component_area_fraction"] == 0.001 and "provisional" in c["roi"]["estado"]
    assert c["diseno"]["modos"]["humo"] == {"fold": 0, "epocas_min": 1, "epocas_max": 2, "cuenta_como_resultado": False}
    C.comprobar_coherencia_con_g2(c, g2)
    assert C.segmentador_id(c, 3) == "unet__s42__f3" and C.fold_validacion_interna(c["diseno"]["folds"], 4) == 0
    bruto = (REPO / c["referencia_g2"]["config"]).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(bruto).hexdigest() == c["referencia_g2"]["sha256_contenido_blob"]            # el YAML de G2 no cambió
    for rel in c["integridad"]["archivos_con_huella"]:
        assert (REPO / rel).is_file(), rel
    protocolo = (REPO / c["protocolo"]["ruta"]).read_text(encoding="utf-8")
    assert len(protocolo.splitlines()) <= 60 and "candidato a congelamiento" in protocolo.lower() and "phash_le8_v1" in protocolo
    assert c["protocolo"]["conformidad_academica_asesor"] == "PENDIENTE" and c["acceso_test"]["estado"] == "cerrado"


def test_configuracion_invalida_se_rechaza():
    c, _ = configs_reales()
    for mutar, patron in ((lambda x: x["roi"].update(umbral=0.4), "roi.umbral"), (lambda x: x["receta"].update(binarizacion_referencia=">= 0.5"), "binarización"),
                          (lambda x: x["diseno"].update(folds=[0, 1]), "folds"), (lambda x: x.update(esquema="otro"), "esquema")):
        m = copy.deepcopy(c)
        mutar(m)
        with pytest.raises(GuardError, match=patron):
            C.validar_config(m)
    _, g2 = configs_reales()
    g2["particion"]["verificaciones_bloqueantes"]["folds"] = [0, 1, 2, 3]
    with pytest.raises(GuardError, match="folds"):
        C.comprobar_coherencia_con_g2(c, g2)


def _git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), "-c", "user.email=t@t", "-c", "user.name=t", *args], capture_output=True, text=True, check=True).stdout.strip()


def test_integridad_sencilla_head_arbol_limpio_y_pin(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "src/r5_v2").mkdir(parents=True)
    (tmp_path / "src/r5_v2/a.py").write_bytes(b"x = 1\n")
    (tmp_path / "g2.yaml").write_bytes(b"g2: 1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "c")
    cfg = {"integridad": {"rutas_sin_cambios": ["src/r5_v2", "configs/r5_v2"], "archivos_con_huella": ["src/r5_v2/a.py", "g2.yaml"]},
           "referencia_g2": {"config": "g2.yaml", "sha256_contenido_blob": hashlib.sha256(b"g2: 1\n").hexdigest()}}
    ok = C.verificar_integridad_g3a(tmp_path, cfg)
    assert ok["git_commit_head"] == _git(tmp_path, "rev-parse", "HEAD") and not ok["bloqueos"] and ok["archivos"]["src/r5_v2/a.py"]["sha256_contenido_blob"]
    assert C.exigir_integridad(ok) is ok
    (tmp_path / "src/r5_v2/a.py").write_bytes(b"x = 2\n")                                            # modificado
    sucio = C.verificar_integridad_g3a(tmp_path, cfg)
    assert [b["codigo"] for b in sucio["bloqueos"]] == ["arbol_sucio"] and C.comparar_instantaneas(ok, sucio)
    with pytest.raises(GuardError, match="arbol_sucio"):
        C.exigir_integridad(sucio)
    _git(tmp_path, "checkout", "--", "src/r5_v2/a.py")
    (tmp_path / "src/r5_v2/nuevo.py").write_text("y = 1\n")                                           # sin rastrear en una ruta vigilada
    assert [b["codigo"] for b in C.verificar_integridad_g3a(tmp_path, cfg)["bloqueos"]] == ["arbol_sucio"]
    (tmp_path / "src/r5_v2/nuevo.py").unlink()
    cfg["referencia_g2"]["sha256_contenido_blob"] = "0" * 64                                          # YAML de G2 distinto del fijado
    assert [b["codigo"] for b in C.verificar_integridad_g3a(tmp_path, cfg)["bloqueos"]] == ["yaml_g2_distinto_del_fijado"]


def test_resumen_metricas_seg_descriptivo_sin_ic():
    def f(i, fold, dice, iou, cob, fb, vacia=False):
        return {"image_id": str(i), "fold": str(fold), "dice": repr(dice), "iou": repr(iou), "cobertura": "" if cob is None else repr(cob),
                "fallback_pred": str(fb), "fallback_gt": "False", "pred_vacia": str(vacia), "gt_vacia": "False"}
    filas = [f(1, 0, 0.9, 0.8, 1.0, False), f(2, 0, 0.05, 0.02, 0.4, False), f(3, 1, 0.5, 0.3, 0.6, False),
             f(4, 1, 0.0, 0.0, 1.0, True, vacia=True), f(5, 1, 0.7, 0.6, 0.2, False)]
    r = MS.resumen_metricas_seg(filas)
    assert r["n"] == 5 and r["dice"]["media"] == pytest.approx(np.mean([0.9, 0.05, 0.5, 0.0, 0.7])) and r["dice"]["mediana"] == 0.5
    assert r["iou"]["mediana"] == 0.3 and r["n_dice_menor_umbral"] == 2
    assert r["por_fold"]["0"]["dice"]["n"] == 2 and r["por_fold"]["1"]["n_dice_menor_umbral"] == 1 and r["por_fold"]["1"]["n_fallback_pred"] == 1
    cs = r["cobertura_sin_fallback_pred"]                                                             # la fila con fallback queda aparte
    assert cs["n"] == 4 and cs["minimo"] == 0.2 and cs["n_menor_umbral"] == 2 and cs["media"] == pytest.approx(np.mean([1.0, 0.4, 0.6, 0.2]))
    assert r["cobertura_todas"]["n"] == 5 and r["fallback_pred"] == {"n": 1, "tasa": 0.2} and r["n_pred_vacia"] == 1
    assert r["sin_intervalos_de_confianza"] is True and not any("interpret" in k for k in r)
    with pytest.raises(GuardError):
        MS.resumen_metricas_seg([])
