"""R9 (sin torch): textos de alcance/advertencias, ausencia de código de métricas o entrenamiento, y validación de la imagen de entrada (F1: cargar). Todo sintético; nunca test.
Comando: py -3.13 -B -m pytest tests/gui -q -p no:cacheprovider
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from src.gui import app_tk, cli, controlador, sistema, textos as T, validacion_imagen as V  # noqa: E402


# ============================================================================ textos: alcance, etiquetas y rendimiento (decisiones aprobadas)
def test_r9_3_opcion_a_conserva_el_indicador_literal_con_los_calificativos():
    p = T.PRESENTACION.lower()
    assert "prototipo básico de sistema de apoyo diagnóstico" in p
    for q in ("académico", "demostrativo", "sin validez diagnóstica", "no destinado al uso clínico", "no es un dispositivo médico"):
        assert q in p


def test_aviso_principal_y_alcance():
    a = T.AVISO_PRINCIPAL.lower()
    assert all(x in a for x in ("no es un dispositivo médico", "no tiene validez diagnóstica", "no está destinado al uso clínico", "no sustituye a especialistas"))
    s = T.ALCANCE.lower()
    assert "no detecta ovarios normales" in s and "previamente identificadas" in s and "no está validado" in s


def test_etiquetas_con_el_mapeo_v2_explicado():
    assert (T.ETIQUETA_BENIGNO, T.ETIQUETA_MALIGNO) == ("BENIGNO", "MALIGNO")
    n = T.NOTA_ETIQUETAS
    assert "mapeo v2" in n and "0, 1, 2, 3, 4 y 6" in n and "categoría 6" in n and "categoría original 7" in n and "no equivalen a un diagnóstico histopatológico" in n


def test_puntuacion_no_calibrada_y_umbral_exacto():
    n = T.NOTA_PUNTUACION
    assert "NO es una probabilidad calibrada" in n and "0.08456226376195748" in n and "no validada para el ensamble de quince" in n and "tres modelos" in n
    assert T.T_HAT_TEXTO == "0.08456226376195748"


def test_rendimiento_de_r8_con_las_cifras_aprobadas():
    r = T.RENDIMIENTO
    for x in ("0.60", "0.32–0.84", "0.921", "0.237", "15 positivas", "367 negativas", "no se alcanzó", "Sin validación clínica", "único subtipo maligno", "0.882"):
        assert x in r, x


def test_orden_de_las_advertencias_y_modo_maqueta():
    assert T.advertencias() == [T.AVISO_PRINCIPAL, T.ALCANCE, T.NOTA_ETIQUETAS, T.NOTA_PUNTUACION, T.RENDIMIENTO]
    assert T.MODO_MAQUETA == "Modo demostración: modelo no disponible" and "no se muestran resultados simulados" in T.MAQUETA_DETALLE


def test_la_interfaz_no_contiene_codigo_de_metricas_ni_entrenamiento():
    prohibidos = ("average_precision", "roc_auc", "roc_curve", "calibrar", "optimizer", ".fit(", "backward(", "train(", "calcular_resultados", "bootstrap")
    for mod in (V, sistema, controlador, cli, app_tk):
        fuente = inspect.getsource(mod).lower()
        assert not any(p.lower() in fuente for p in prohibidos), (mod.__name__, [p for p in prohibidos if p.lower() in fuente])


def test_sin_umbral_de_abstencion_ni_umbral_alternativo():
    for mod in (sistema, controlador, cli, app_tk):
        fuente = inspect.getsource(mod)
        assert "abst" not in fuente.lower() and not re.search(r"0\.5\b", fuente.replace("0.5.", ""))      # sin umbral 0.5 ni de abstención


# ============================================================================ F1: validación y carga de la imagen
def guardar(ruta: Path, formato="PNG", modo="RGB", tam=(64, 48), color=120):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    Image.new(modo, tam, color if modo != "RGB" else (color, color, color)).save(ruta, format=formato)
    return ruta


@pytest.mark.parametrize("nombre,formato", [("a.png", "PNG"), ("a.jpg", "JPEG"), ("a.JPG", "JPEG"), ("a.jpeg", "JPEG")])
def test_acepta_png_y_jpg(tmp_path, nombre, formato):
    img, info = V.validar_y_cargar(guardar(tmp_path / nombre, formato))
    assert img.mode == "RGB" and img.size == (64, 48) and info["formato"] == formato and len(info["sha256_bytes_archivo"]) == 64


@pytest.mark.parametrize("modo", ["L", "RGBA", "P"])
def test_convierte_a_rgb_como_el_lector_de_g2(tmp_path, modo):
    guardar(tmp_path / "m.png", "PNG", modo)
    img, info = V.validar_y_cargar(tmp_path / "m.png")
    assert img.mode == "RGB" and info["modo_original"] == modo


def test_rechaza_inexistente_vacio_extension_y_contenido(tmp_path):
    with pytest.raises(V.ImagenInvalida, match="no existe"):
        V.validar_y_cargar(tmp_path / "no.png")
    (tmp_path / "v.png").write_bytes(b"")
    with pytest.raises(V.ImagenInvalida, match="vacío"):
        V.validar_y_cargar(tmp_path / "v.png")
    (tmp_path / "t.txt").write_text("x")
    with pytest.raises(V.ImagenInvalida, match="PNG o JPG"):
        V.validar_y_cargar(tmp_path / "t.txt")
    (tmp_path / "falsa.png").write_bytes(b"no es una imagen")
    with pytest.raises(V.ImagenInvalida, match="dañado"):
        V.validar_y_cargar(tmp_path / "falsa.png")
    guardar(tmp_path / "bmp_como_png.png", "BMP")
    with pytest.raises(V.ImagenInvalida, match="PNG ni JPG"):
        V.validar_y_cargar(tmp_path / "bmp_como_png.png")
    (tmp_path / "dir.png").mkdir()
    with pytest.raises(V.ImagenInvalida):
        V.validar_y_cargar(tmp_path / "dir.png")


def test_rechaza_archivo_truncado_y_dimensiones_fuera_de_rango(tmp_path):
    p = guardar(tmp_path / "ok.png", tam=(200, 200))
    (tmp_path / "trunc.png").write_bytes(p.read_bytes()[:60])
    with pytest.raises(V.ImagenInvalida):
        V.validar_y_cargar(tmp_path / "trunc.png")
    with pytest.raises(V.ImagenInvalida, match="Dimensiones"):
        V.validar_y_cargar(guardar(tmp_path / "chica.png", tam=(V.LADO_MIN - 1, 100)))


def test_rechaza_tamano_excesivo(tmp_path, monkeypatch):
    p = guardar(tmp_path / "g.png", tam=(100, 100))
    monkeypatch.setattr(V, "BYTES_MAX", 10)
    with pytest.raises(V.ImagenInvalida, match="tamaño máximo"):
        V.validar_y_cargar(p)


@pytest.mark.parametrize("carpeta", ["test", "TEST", "Test"])
def test_nunca_abre_rutas_con_componente_test(tmp_path, carpeta):
    p = guardar(tmp_path / carpeta / "images" / "x.png")
    with pytest.raises(V.ImagenInvalida, match="test"):
        V.validar_y_cargar(p)


def test_un_nombre_que_solo_contiene_la_palabra_test_no_se_rechaza(tmp_path):
    V.validar_y_cargar(guardar(tmp_path / "test_imagen_sintetica.png"))                  # el componente es «test_imagen_sintetica.png», no «test»
