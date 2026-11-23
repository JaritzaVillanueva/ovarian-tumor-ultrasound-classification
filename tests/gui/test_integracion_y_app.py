"""R9: (1) integración con imágenes de DESARROLLO sintéticas (misma lógica que `src.gui.cli integracion-desarrollo`, que en Falcon se ejecuta con los checkpoints reales) y (2) humo de la
ventana Tkinter (se omite sin pantalla). Sin etiquetas, sin métricas, sin test.
Comando: py -3.13 -B -m pytest tests/gui -q -p no:cacheprovider
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("yaml")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")
pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from src.r4_v2.particion import filas_de_fold, resolver_imagen  # noqa: E402
from src.r7_v2 import sistema_final as SF  # noqa: E402
from src.gui import cli, textos as T  # noqa: E402
from src.gui.controlador import Controlador  # noqa: E402
from src.gui.sistema import SistemaNoDisponible  # noqa: E402
from src.gui import validacion_imagen as V  # noqa: E402
from src.gui.validacion_imagen import validar_y_cargar  # noqa: E402
from tests.gui.test_sistema_controlador_cli import imagen, sistema_falso  # noqa: E402
from tests.test_r7_v2_sistema_final import armar_sf  # noqa: E402


@pytest.fixture(scope="module")
def sf(tmp_path_factory):
    return armar_sf(tmp_path_factory.mktemp("banco_gui_sf"))


@pytest.fixture(autouse=True)
def lado_minimo_sintetico(monkeypatch):
    """Las imágenes sintéticas de G2 son diminutas (24×20); las reales de MMOTU superan con holgura el mínimo de la validación (32 px)."""
    monkeypatch.setattr(V, "LADO_MIN", 8)


def huella(img) -> str:
    return hashlib.sha1(img.tobytes()).hexdigest()


def tabla_oof(sf, miembros):
    """(id_corrida, huella de imagen) -> probabilidad guardada en `predicciones_oof.csv`, para TODAS las imágenes del pliegue propio de cada miembro (preparación del doble de prueba)."""
    tabla = {}
    for m in miembros:
        _, val = filas_de_fold(sf.particion, m["fold"])
        ids, probs = SF.leer_probabilidades_guardadas(sf.raiz_a / m["id_corrida"] / m["intento"] / "predicciones_oof.csv")
        por_id = dict(zip(ids, probs))
        for f in val:
            tabla[(m["id_corrida"], huella(validar_y_cargar(resolver_imagen(sf.root, f))[0]))] = float(por_id[str(f["image_id"])])
    return tabla


@pytest.fixture(scope="module")
def miembros(sf):
    return SF.construir_manifiesto(sf.root, sf.config, sf.config_g2, sf.particion)["miembros"]


def test_integracion_con_desarrollo_reproduce_las_probabilidades_guardadas(sf, miembros):
    tabla = tabla_oof(sf, miembros)
    res = cli.integracion_desarrollo(sf.root, sf.config_g2, sf.particion, miembros, lambda idc, img: tabla[(idc, huella(img))], n_por_fold=2)
    assert res["ok"] and res["diferencia_maxima_global"] == 0.0 and len(res["miembros"]) == 15 and all(x["n_imagenes"] == 2 for x in res["miembros"])
    assert res["metricas_calculadas"] is False and res["etiquetas_usadas"] is False and res["ensamble_evaluado_sobre_desarrollo"] is False


def test_la_integracion_no_incluye_etiquetas_ni_metricas_en_su_salida(sf, miembros):
    tabla = tabla_oof(sf, miembros)
    res = cli.integracion_desarrollo(sf.root, sf.config_g2, sf.particion, miembros[:3], lambda idc, img: tabla[(idc, huella(img))], n_por_fold=1)
    texto = json.dumps(res).lower()
    assert not any(x in texto for x in ("label", "etiqueta_v2", "original_class", "average_precision", "roc_auc", "sensibilidad", "especificidad"))
    assert set(res) == {"miembros", "n_por_fold", "diferencia_maxima_global", "tolerancia", "ok", "metricas_calculadas", "etiquetas_usadas", "ensamble_evaluado_sobre_desarrollo"}


def test_la_integracion_detecta_una_ruta_de_inferencia_distinta(sf, miembros):
    res = cli.integracion_desarrollo(sf.root, sf.config_g2, sf.particion, miembros[:2], lambda idc, img: 0.123456, n_por_fold=1)
    assert res["ok"] is False and res["diferencia_maxima_global"] > cli.TOLERANCIA_INTEGRACION


def test_la_integracion_solo_usa_imagenes_de_su_propio_pliegue_y_de_desarrollo(sf, miembros):
    vistas = []
    tabla = tabla_oof(sf, miembros[:5])

    def sonda(idc, img):
        vistas.append(idc)
        return tabla[(idc, huella(img))]
    cli.integracion_desarrollo(sf.root, sf.config_g2, sf.particion, miembros[:5], sonda, n_por_fold=3)
    assert len(vistas) == 5 * 3
    for m in miembros[:5]:
        _, val = filas_de_fold(sf.particion, m["fold"])
        assert all(f["e1_split"] in ("train", "validation") for f in val)                  # K4: nunca test


def test_el_sistema_completo_clasifica_una_imagen_de_desarrollo_sin_producir_metricas(sf, tmp_path):
    f = filas_de_fold(sf.particion, 0)[1][0]
    img, _ = validar_y_cargar(resolver_imagen(sf.root, f))
    r = sistema_falso(0.2).clasificar(img)
    assert r.clase in ("BENIGNO", "MALIGNO") and not hasattr(r, "auc") and not hasattr(r, "sensibilidad")


# ============================================================================ ventana Tkinter (humo; se omite sin pantalla)
@pytest.fixture
def raiz_tk():
    tk = pytest.importorskip("tkinter")
    try:
        raiz = tk.Tk()
    except tk.TclError:
        pytest.skip("sin pantalla disponible para Tkinter")
    raiz.withdraw()
    yield raiz
    raiz.destroy()


def test_ventana_flujo_cargar_visualizar_clasificar(raiz_tk, tmp_path):
    pytest.importorskip("PIL.ImageTk")
    from src.gui.app_tk import AppTk
    c = Controlador(cargar=lambda: sistema_falso(0.3))
    c.iniciar()
    app = AppTk(raiz_tk, c, usar_hilo=False)
    assert str(app.b_clasificar.cget("state")) == "disabled"                                  # sin imagen no se puede clasificar
    assert app.cargar_imagen(imagen(tmp_path)) and str(app.b_clasificar.cget("state")) == "normal"
    assert "64×48" in app.info.cget("text")
    assert app._clasificar_sync() and app.clase.cget("text") == "MALIGNO"
    assert "0.08456226376195748" in app.detalle.cget("text")
    texto = app.avisos.get("1.0", "end")
    assert all(a[:40] in texto for a in T.advertencias())                                      # los avisos están siempre visibles


def test_ventana_imagen_invalida_muestra_un_error_claro(raiz_tk, tmp_path):
    pytest.importorskip("PIL.ImageTk")
    from src.gui.app_tk import AppTk
    c = Controlador(cargar=lambda: sistema_falso(0.3))
    c.iniciar()
    app = AppTk(raiz_tk, c, usar_hilo=False)
    (tmp_path / "x.png").write_bytes(b"no es imagen")
    assert app.cargar_imagen(tmp_path / "x.png") is False and "dañado" in app._ultimo_error
    assert str(app.b_clasificar.cget("state")) == "disabled"


def test_ventana_en_modo_demostracion_no_infiere(raiz_tk, tmp_path):
    pytest.importorskip("PIL.ImageTk")
    from src.gui.app_tk import AppTk

    def falla():
        raise SistemaNoDisponible("sin checkpoints")
    c = Controlador(cargar=falla)
    c.iniciar()
    app = AppTk(raiz_tk, c, usar_hilo=False)
    assert app.clase.cget("text") == T.MODO_MAQUETA and T.MODO_MAQUETA in app.estado.cget("text")
    assert app.cargar_imagen(imagen(tmp_path)) and str(app.b_clasificar.cget("state")) == "disabled"      # se puede ver la imagen, no clasificar
    assert app._clasificar_sync() is False and app.clase.cget("text") == T.MODO_MAQUETA and c.resultado is None
