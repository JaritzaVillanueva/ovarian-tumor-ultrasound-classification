"""R9 (D-077): selección objetiva de los dos ejemplos reales de desarrollo (regla fijada antes de consultar predicciones). Sintético: partición y archivos de prueba; nunca test ni modelos."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("yaml")
pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from src.r4_v2.common import GuardError  # noqa: E402
from src.r4_v2.particion import Particion, resolver_imagen  # noqa: E402
from src.gui import ejemplos_demostracion as E  # noqa: E402

BASE = "data/processed/MMOTU_OTU_2D_experimental"


def fila(i, categoria, split="train", nombre=None):
    nombre = nombre or f"{i}.JPG"
    return {"image_id": str(i), "image_name": nombre, "original_class": categoria, "label_v2": int(categoria == 7), "class_v2": "x", "e1_split": split,
            "image_path": f"{BASE}/{split}/images/{nombre}", "group_id": f"g{i}", "group_size": "1", "fold": 0}


def crear(root, filas, invalidas=()):
    for f in filas:
        p = Path(root) / f["image_path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        if f["image_id"] in invalidas:
            p.write_bytes(b"no es imagen")
        else:
            Image.new("RGB", (64, 48), (int(f["image_id"]) % 255, 80, 80)).save(p, format="JPEG")


def test_orden_de_texto_y_categorias_de_cada_caso(tmp_path):
    filas = [fila(2, 7), fila(10, 7), fila(9, 6), fila(100, 3), fila(11, 0), fila(1, 5)]            # la 5 no debería existir en la partición v2; aquí se comprueba que no se elige
    crear(tmp_path, filas)
    sel = E.seleccionar(filas, lambda f: resolver_imagen(tmp_path, f))
    assert sel["A"]["image_id"] == "100" and sel["B"]["image_id"] == "10"                           # texto: «10» < «100» < «11» < «2» < «9»; la categoría 6 («9») no es del caso A


def test_una_imagen_no_valida_se_salta_por_orden(tmp_path):
    filas = [fila(1, 2), fila(10, 2), fila(3, 7)]
    crear(tmp_path, filas, invalidas={"1"})
    assert E.seleccionar(filas, lambda f: resolver_imagen(tmp_path, f))["A"]["image_id"] == "10"


def test_sin_imagenes_validas_de_un_caso_detiene(tmp_path):
    filas = [fila(1, 2)]
    crear(tmp_path, filas)
    with pytest.raises(GuardError, match="caso B"):
        E.seleccionar(filas, lambda f: resolver_imagen(tmp_path, f))


def test_la_seleccion_no_depende_del_orden_de_entrada(tmp_path):
    filas = [fila(5, 1), fila(4, 1), fila(8, 7), fila(7, 7)]
    crear(tmp_path, filas)
    r = lambda f: resolver_imagen(tmp_path, f)  # noqa: E731
    assert E.seleccionar(filas, r) == E.seleccionar(list(reversed(filas)), r)


def test_verificar_acepta_lo_fijado_y_se_detiene_si_difiere(tmp_path, monkeypatch):
    filas = [fila(1, 4, nombre="1.JPG"), fila(1417, 7), fila(20, 7)]
    crear(tmp_path, filas)
    part = Particion(filas=filas, sha256_bytes_archivo="0" * 64, tamano_bytes=0)
    from src.r4_v2.common import sha256_archivo
    fijos = {"A": {"image_id": "1", "image_name": "1.JPG", "original_class": 4, "e1_split": "train", "sha256": sha256_archivo(tmp_path / filas[0]["image_path"])},
             "B": {"image_id": "1417", "image_name": "1417.JPG", "original_class": 7, "e1_split": "train", "sha256": sha256_archivo(tmp_path / filas[1]["image_path"])}}
    monkeypatch.setattr(E, "CASOS", fijos)
    # la regla sobre estos datos elige 1417 como caso B? el orden de texto da «1417» < «20»: sí
    salida = E.verificar(tmp_path, part)
    assert [s["caso"] for s in salida] == ["A", "B"] and salida[0]["label_v2"] == 0 and salida[1]["label_v2"] == 1 and salida[0]["fold"] == 0
    monkeypatch.setattr(E, "CASOS", {**fijos, "B": {**fijos["B"], "sha256": "0" * 64}})            # el archivo cambió
    with pytest.raises(GuardError, match="difiere de la fijada"):
        E.verificar(tmp_path, part)
    monkeypatch.setattr(E, "CASOS", {**fijos, "A": {**fijos["A"], "image_id": "2"}})               # otro identificador
    with pytest.raises(GuardError, match="difiere de la fijada"):
        E.verificar(tmp_path, part)


def test_cli_exige_el_token_de_d077(tmp_path, capsys):
    for malo in ("", "CONFIRMADO", "CONFIRMADO:D-076", "CONFIRMADO:D-078"):
        assert E.main(["--autorizacion", malo], root=tmp_path) == 3


def test_los_identificadores_fijados_son_coherentes_con_el_mapeo_v2():
    assert E.CASOS["A"]["original_class"] in (0, 1, 2, 3, 4) and E.CASOS["B"]["original_class"] == 7
    assert E.CASOS["A"]["e1_split"] in ("train", "validation") and E.CASOS["B"]["e1_split"] in ("train", "validation")
    assert {k: v["image_id"] for k, v in E.CASOS.items()} == {"A": "1", "B": "1417"}


def test_el_modulo_no_infiere_ni_calcula_metricas():
    fuente = (Path(__file__).resolve().parents[2] / "src" / "gui" / "ejemplos_demostracion.py").read_text(encoding="utf-8").lower()
    for prohibido in ("import torch", "sistema_final", "cargar_sistema", "predecir", "average_precision", "roc_auc", "probabilidades_guardadas", "predicciones_oof"):
        assert prohibido not in fuente, prohibido
