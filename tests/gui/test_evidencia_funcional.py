"""R9 (D-076): pruebas SINTÉTICAS del ejecutor de pruebas funcionales con la ventana Tkinter (`src.gui.evidencia_funcional`): casos F1–F3 y de errores, comprobaciones de F3 (una imagen, 15 modelos,
promedio, umbral), detección de defectos, compuertas, directorios nuevos y ausencia de etiquetas/métricas. Sistema falso y sin capturas de pantalla; nunca checkpoints reales ni test.
Las capturas auténticas y la ejecución con los 15 checkpoints reales las hace la autora en Windows. Se omiten sin tkinter o sin pantalla.
Comando: py -3.13 -B -m pytest tests/gui -q -p no:cacheprovider
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("yaml")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")
pytest.importorskip("PIL")
torch = pytest.importorskip("torch")
tk = pytest.importorskip("tkinter")
from PIL import Image  # noqa: E402

from src.r4_v2.common import GuardError  # noqa: E402
from src.r7_v2 import sistema_final as SF  # noqa: E402
from src.gui import evidencia_funcional as E, textos as T  # noqa: E402
from src.gui.controlador import Controlador  # noqa: E402
from src.gui.sistema import SistemaCongelado  # noqa: E402
from tests.gui.test_sistema_controlador_cli import miembros_falsos  # noqa: E402

AUT = "CONFIRMADO:D-076"
DEV = Path(*E.BASE_DESARROLLO)


@pytest.fixture
def raiz_tk():
    try:
        raiz = tk.Tk()
    except tk.TclError:
        pytest.skip("sin pantalla disponible para Tkinter")
    raiz.withdraw()
    yield raiz
    try:
        raiz.destroy()
    except tk.TclError:
        pass


def a_tensor(im):
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous()


def sistema(prob=0.3, *, preprocesar=None, predecir=None):
    ms = miembros_falsos()
    return SistemaCongelado(miembros=ms, modelos={m["id_corrida"]: {"p": prob} for m in ms}, preprocesar=preprocesar or a_tensor,
                            predecir=predecir or (lambda modelo, tensor, device: modelo["p"]), device="cpu", t_hat_texto=SF.T_HAT)


def controlador(sis=None):
    c = Controlador(cargar=lambda: sis or sistema())
    c.iniciar()
    return c


def crear_desarrollo(root: Path, n=3):
    carpeta = root / DEV / "train" / "images"
    carpeta.mkdir(parents=True)
    rutas = []
    for i, nombre in enumerate(("1.JPG", "10.JPG", "2.JPG")[:n]):
        Image.new("RGB", (64 + i, 48), (60 + 20 * i, 90, 120)).save(carpeta / nombre, format="JPEG")
        rutas.append(carpeta / nombre)
    return rutas


def referencia_g2(c):
    """Doble de la ruta de lectura de G2: abre el archivo y aplica la misma transformación (en Windows se usa `DatasetLesiones`)."""
    return lambda ruta: c.sistema.preprocesar(Image.open(ruta).convert("RGB"))[0]


def correr(raiz_tk, tmp_path, c=None, *, usar_hilo=True, desarrollo=None, **kw):
    c = c or controlador()
    salida = tmp_path / "salida"
    salida.mkdir(exist_ok=True)
    desarrollo = desarrollo if desarrollo is not None else crear_desarrollo(tmp_path / "repo", 2)
    return E.ejecutar(tmp_path / "repo", salida, raiz_tk, c, imagenes_desarrollo=desarrollo, capturas=False, tensor_g2=referencia_g2(c), usar_hilo=usar_hilo, **kw)


def por_id(reg):
    return {c["id"]: c for c in reg["casos"]}


# ============================================================================ casos con un sistema correcto
def test_todos_los_casos_se_superan_con_un_sistema_correcto_y_la_ruta_real_del_boton(raiz_tk, tmp_path):
    reg = correr(raiz_tk, tmp_path)
    c = por_id(reg)
    assert reg["todos_superados"], [(k, v["observado"]) for k, v in c.items() if v["resultado"] != "superada"]
    assert list(c) == ["P00", "E01", "F1-S", "F2-S", "F3-S", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "F1-D1", "F2-D1", "F3-D1", "F1-D2", "F2-D2", "F3-D2", "E10", "P99"]
    assert reg["capturas_reales"] is False and all(x.get("captura") is None for x in reg["casos"])


def test_las_comprobaciones_de_f3_cubren_una_imagen_quince_modelos_promedio_y_umbral(raiz_tk, tmp_path):
    c = por_id(correr(raiz_tk, tmp_path))
    for id_ in ("F3-S", "F3-D1", "F3-D2"):
        comp = c[id_]["comprobaciones"]
        assert all(v for k, v in comp.items() if isinstance(v, bool)), comp
        assert {"un_preprocesamiento", "quince_inferencias", "lote_de_una_imagen", "mismo_tensor_en_los_quince", "quince_modelos_distintos", "promedio_independiente_ok", "umbral_exacto",
                "regla_mayor_o_igual", "repetible_bit_a_bit", "resultado_mostrado_en_la_ventana"} <= set(comp)
        assert comp["forma_del_tensor"][0] == 1 and comp["n_miembros"] == 15 and c[id_]["resultado"] == "superada"
        assert ("tensor_igual_a_la_ruta_de_g2_bit_a_bit" in comp) == (id_ != "F3-S")
    # la sintética no está en el árbol de desarrollo: no tiene ruta de G2 con la que comparar
    assert c["F3-D1"]["resultado"] == "superada" and c["F1-D1"]["imagen_sha256"] != c["F1-D2"]["imagen_sha256"]


def test_los_errores_muestran_mensajes_claros_y_conservan_el_estado(raiz_tk, tmp_path):
    c = por_id(correr(raiz_tk, tmp_path))
    for id_, texto in (("E02", "no existe"), ("E03", "PNG o JPG"), ("E04", "vacío"), ("E05", "dañado"), ("E06", "PNG ni JPG"), ("E07", "Dimensiones"), ("E08", "test")):
        assert c[id_]["resultado"] == "superada" and texto in c[id_]["observado"], (id_, c[id_]["observado"])
    assert "Primero cargue" in c["E01"]["observado"]
    assert "Modo demostración: modelo no disponible" in c["E10"]["observado"] and "sin resultados simulados" in c["E10"]["observado"]


def test_funciona_tambien_con_la_clasificacion_sincrona(raiz_tk, tmp_path):
    assert correr(raiz_tk, tmp_path, usar_hilo=False)["todos_superados"]


@pytest.mark.parametrize("prob,clase", [(0.0846, "MALIGNO"), (0.0845, "BENIGNO")])
def test_por_encima_y_por_debajo_del_umbral_se_clasifican_con_la_regla_mayor_o_igual(raiz_tk, tmp_path, prob, clase):
    reg = correr(raiz_tk, tmp_path, controlador(sistema(prob)))
    assert reg["todos_superados"] and por_id(reg)["F3-S"]["observado"].startswith(f"clase={clase}")


# ============================================================================ el ejecutor DETECTA defectos
def clase_de_caso(reg, id_):
    return por_id(reg)[id_]


def test_detecta_una_inferencia_con_lote_de_mas_de_una_imagen(raiz_tk, tmp_path):
    s = sistema(preprocesar=lambda im: torch.cat([a_tensor(im), a_tensor(im)], dim=0))
    reg = correr(raiz_tk, tmp_path, controlador(s))
    comp = clase_de_caso(reg, "F3-S")["comprobaciones"]
    assert clase_de_caso(reg, "F3-S")["resultado"] == "NO superada" and comp["lote_de_una_imagen"] is False


def test_detecta_un_preprocesamiento_distinto_por_modelo(raiz_tk, tmp_path):
    s = sistema()
    original = s.probabilidad_miembro
    s.probabilidad_miembro = lambda idc, imagen, tensor=None: original(idc, imagen, None)                     # re-preprocesa para cada miembro
    comp = clase_de_caso(correr(raiz_tk, tmp_path, controlador(s)), "F3-S")["comprobaciones"]
    assert comp["un_preprocesamiento"] is False and comp["mismo_tensor_en_los_quince"] is False


def test_detecta_un_umbral_distinto_del_congelado(raiz_tk, tmp_path, monkeypatch):
    original = SF.agregar_y_decidir
    monkeypatch.setattr(SF, "agregar_y_decidir", lambda probs, t_hat=SF.T_HAT: original(probs, "0.5"))        # umbral ajeno: 0.3 sería BENIGNO
    caso = clase_de_caso(correr(raiz_tk, tmp_path, controlador(sistema(0.3))), "F3-S")
    assert caso["resultado"] == "NO superada" and caso["comprobaciones"]["regla_mayor_o_igual"] is False


def test_detecta_una_clasificacion_no_repetible(raiz_tk, tmp_path):
    n = {"i": 0}

    def ruidosa(modelo, tensor, device):
        n["i"] += 1
        return min(1.0, modelo["p"] + 1e-6 * (n["i"] > 15))
    caso = clase_de_caso(correr(raiz_tk, tmp_path, controlador(sistema(predecir=ruidosa))), "F3-S")
    assert caso["comprobaciones"]["repetible_bit_a_bit"] is False and caso["resultado"] == "NO superada"


def test_detecta_un_tensor_de_la_interfaz_distinto_del_de_g2(raiz_tk, tmp_path):
    c2 = controlador()
    salida = tmp_path / "s2"
    salida.mkdir()
    reg = E.ejecutar(tmp_path / "repo2", salida, raiz_tk, c2, imagenes_desarrollo=crear_desarrollo(tmp_path / "repo2", 2), capturas=False,
                     tensor_g2=lambda ruta: referencia_g2(c2)(ruta) + 1e-3, usar_hilo=False)
    assert por_id(reg)["F3-D1"]["comprobaciones"]["tensor_igual_a_la_ruta_de_g2_bit_a_bit"] is False and not reg["todos_superados"]


def test_un_error_de_clasificacion_se_registra_como_caso_no_superado_sin_interrumpir(raiz_tk, tmp_path):
    def falla(modelo, tensor, device):
        raise RuntimeError("fallo sintético de inferencia")
    reg = correr(raiz_tk, tmp_path, controlador(sistema(predecir=falla)), usar_hilo=False)
    assert por_id(reg)["F3-S"]["resultado"] == "NO superada" and por_id(reg)["P99"]["resultado"] == "superada" and reg["total"] == 20


# ============================================================================ imágenes y compuertas
def test_la_imagen_sintetica_es_determinista_y_valida(tmp_path):
    a, b = E.imagen_sintetica(tmp_path / "a.png"), E.imagen_sintetica(tmp_path / "b.png")
    assert a.read_bytes() == b.read_bytes() and Image.open(a).size == (512, 384) and Image.open(a).format == "PNG"


def test_ruta_de_desarrollo_solo_acepta_train_o_validation_del_arbol_de_desarrollo(tmp_path):
    ok = crear_desarrollo(tmp_path, 1)[0]
    assert E.ruta_desarrollo(tmp_path, ok) == ok.resolve()
    for malo in (tmp_path / DEV / "test" / "images" / "x.JPG", tmp_path / DEV / "train" / "masks" / "x.JPG", tmp_path / "otra" / "x.JPG", tmp_path.parent / "fuera.JPG"):
        malo.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 40)).save(malo, format="JPEG")
        with pytest.raises(GuardError):
            E.ruta_desarrollo(tmp_path, malo)
    with pytest.raises(GuardError, match="no existe"):
        E.ruta_desarrollo(tmp_path, tmp_path / DEV / "train" / "images" / "no.JPG")


def test_imagenes_de_desarrollo_por_defecto_son_las_primeras_por_nombre_sin_etiquetas(tmp_path):
    crear_desarrollo(tmp_path, 3)
    assert [p.name for p in E.imagenes_desarrollo_por_defecto(tmp_path)] == ["1.JPG", "10.JPG"]               # orden por texto, como las de D-072/D-074
    with pytest.raises(GuardError):
        E.imagenes_desarrollo_por_defecto(tmp_path / "vacio")


def test_cli_exige_el_token_de_d076_y_no_crea_nada_sin_el(tmp_path, raiz_tk):
    crear_desarrollo(tmp_path, 2)
    for malo in ("", "CONFIRMADO", "CONFIRMADO:D-075", "CONFIRMADO:D-074", "CONFIRMADO:D-077"):
        assert E.main(["--autorizacion", malo, "--sin-capturas"], root=tmp_path, controlador=controlador(), raiz_tk=raiz_tk) == 3
    assert not (tmp_path / "results").exists()


def test_cli_rechaza_una_imagen_de_test_y_un_sistema_no_disponible(tmp_path, raiz_tk):
    crear_desarrollo(tmp_path, 2)
    prohibida = tmp_path / DEV / "test" / "images" / "x.JPG"
    prohibida.parent.mkdir(parents=True)
    Image.new("RGB", (40, 40)).save(prohibida, format="JPEG")
    assert E.main(["--autorizacion", AUT, "--sin-capturas", "--imagen-desarrollo", str(prohibida)], root=tmp_path, controlador=controlador(), raiz_tk=raiz_tk) == 3
    from src.gui.sistema import SistemaNoDisponible

    def falla():
        raise SistemaNoDisponible("sin checkpoints")
    c = Controlador(cargar=falla)
    c.iniciar()
    assert E.main(["--autorizacion", AUT, "--sin-capturas"], root=tmp_path, controlador=c, raiz_tk=raiz_tk) == 3
    assert not (tmp_path / "results").exists()


def test_cli_escribe_un_directorio_nuevo_con_registro_json_y_md_y_no_sobrescribe(tmp_path, raiz_tk):
    crear_desarrollo(tmp_path, 3)
    def kw():
        c = controlador()                                                                                      # un controlador y una ventana nuevos por ejecución
        return dict(root=tmp_path, controlador=c, tensor_g2=referencia_g2(c), raiz_tk=tk.Toplevel(raiz_tk))
    assert E.main(["--autorizacion", AUT, "--sin-capturas"], **kw()) == 0
    d = tmp_path / E.SALIDA_REL / "D-076"
    j = json.loads((d / "registro_pruebas_funcionales.json").read_text(encoding="utf-8"))
    assert (d / "registro_pruebas_funcionales.md").is_file() and (d / "sintetica_01.png").is_file()
    assert j["autorizacion"] == AUT and j["todos_superados"] is True and j["total"] == 20 and j["n_miembros"] == 15 and j["umbral"] == "0.08456226376195748"
    assert j["imagenes_desarrollo"] == [f"{DEV.as_posix()}/train/images/1.JPG", f"{DEV.as_posix()}/train/images/10.JPG"]
    assert E.main(["--autorizacion", AUT, "--sin-capturas"], **kw()) == 3                                     # no sobrescribe
    assert E.main(["--autorizacion", AUT, "--sin-capturas", "--sufijo", "_2"], **kw()) == 0


def test_el_registro_no_contiene_etiquetas_ni_metricas_ni_comparaciones_con_el_oof(tmp_path, raiz_tk):
    crear_desarrollo(tmp_path, 3)
    c = controlador()
    assert E.main(["--autorizacion", AUT, "--sin-capturas"], root=tmp_path, controlador=c, tensor_g2=referencia_g2(c), raiz_tk=raiz_tk) == 0
    texto = ((tmp_path / E.SALIDA_REL / "D-076" / "registro_pruebas_funcionales.json").read_text(encoding="utf-8")
             + (tmp_path / E.SALIDA_REL / "D-076" / "registro_pruebas_funcionales.md").read_text(encoding="utf-8")).lower()
    assert not any(x in texto for x in ("label_v2", "y_true", "average_precision", "roc_auc", "predicciones_oof", "dif_oof"))
    assert "la clase mostrada no es evidencia de desempeño" in texto and "12 de los 15" in texto


def test_el_ejecutor_no_contiene_codigo_de_metricas_calibracion_ni_entrenamiento():
    fuente = (Path(__file__).resolve().parents[2] / "src" / "gui" / "evidencia_funcional.py").read_text(encoding="utf-8").lower()
    for prohibido in ("average_precision", "roc_auc", "roc_curve", "calibrar", "optimizer", ".fit(", "backward(", "bootstrap", "predicciones_oof", "leer_probabilidades_guardadas"):
        assert prohibido not in fuente, prohibido
    assert "label_v2\": 0" in fuente                                                                           # único uso: marcador exigido por DatasetLesiones, documentado


def test_el_protocolo_versionado_es_coherente_con_el_ejecutor_y_no_contiene_resultados():
    doc = (Path(__file__).resolve().parents[2] / "src" / "gui" / "protocolo_validacion_funcional_r9_v1.md").read_text(encoding="utf-8")
    assert "Total: 20 casos" in doc and "0.08456226376195748" in doc and "CONFIRMADO:D-076" in doc and "verificar-checkpoints" in doc and "evidencia_funcional" in doc
    for cf in (f"CF-{i}" for i in range(1, 9)):
        assert cf in doc
    assert "**No disponible**" in doc and doc.count("pendiente") >= 6                                          # sin resultados inventados
    assert "no sustituye la tolerancia de 1e-4" in doc.lower() and "no se afirma que tales diferencias no puedan modificar la clase" in doc.lower()
