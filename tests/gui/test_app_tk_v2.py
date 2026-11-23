"""R9, presentación v2 (rediseño visual): pruebas SINTÉTICAS de regresión funcional (las mismas pruebas sobre la presentación v1 y la v2), de estructura y de distribución de la ventana.
Sistema falso; nunca checkpoints reales ni test. Se omiten sin tkinter o sin pantalla.
Comando: py -3.13 -B -m pytest tests/gui -q -p no:cacheprovider
"""

from __future__ import annotations

from pathlib import Path
from tkinter import ttk

import pytest

pytest.importorskip("yaml")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")
pytest.importorskip("PIL")
tk = pytest.importorskip("tkinter")
pytest.importorskip("PIL.ImageTk")

from src.gui import textos as T  # noqa: E402
from src.gui.app_tk import AppTk  # noqa: E402
from src.gui.app_tk_v2 import AppTkV2  # noqa: E402
from src.gui.controlador import Controlador  # noqa: E402
from src.gui.sistema import SistemaNoDisponible  # noqa: E402
from tests.gui.test_sistema_controlador_cli import imagen, sistema_falso  # noqa: E402


@pytest.fixture
def raiz_tk():
    try:
        raiz = tk.Tk()
    except tk.TclError:
        pytest.skip("sin pantalla disponible para Tkinter")
    raiz.withdraw()
    yield raiz
    raiz.destroy()


def construir(raiz, clase, prob=0.3, **kw):
    c = Controlador(cargar=lambda: sistema_falso(prob, **kw))
    c.iniciar()
    return c, clase(raiz, c, usar_hilo=False)


# ============================================================================ regresión funcional: MISMAS comprobaciones sobre v1 y v2
@pytest.mark.parametrize("clase", [AppTk, AppTkV2])
def test_f1_f2_f3_flujo_completo(raiz_tk, tmp_path, clase):
    c, app = construir(raiz_tk, clase)
    assert str(app.b_clasificar.cget("state")) == "disabled"
    assert app.cargar_imagen(imagen(tmp_path)) and str(app.b_clasificar.cget("state")) == "normal"                 # F1
    assert "64×48" in app.info.cget("text") and "a.png" in app.info.cget("text") and str(app.lienzo.cget("image")) != ""   # F2
    assert app._clasificar_sync() and app.clase.cget("text") == "MALIGNO"                                            # F3
    d = app.detalle.cget("text")
    assert "0.08456226376195748" in d and "no es una probabilidad calibrada" in d and "Puntuación del sistema (promedio de 15 modelos)" in d
    texto = app.avisos.get("1.0", "end")
    assert all(a[:40] in texto for a in T.advertencias())


@pytest.mark.parametrize("clase", [AppTk, AppTkV2])
@pytest.mark.parametrize("prob,esperada", [(0.0846, "MALIGNO"), (0.0845, "BENIGNO")])
def test_la_clase_mostrada_sigue_la_regla_del_umbral(raiz_tk, tmp_path, clase, prob, esperada):
    c, app = construir(raiz_tk, clase, prob)
    app.cargar_imagen(imagen(tmp_path))
    app._clasificar_sync()
    assert app.clase.cget("text") == esperada and c.resultado.umbral_texto == "0.08456226376195748"


@pytest.mark.parametrize("clase", [AppTk, AppTkV2])
def test_error_de_carga_mensaje_claro_estado_conservado_y_sin_clase_residual(raiz_tk, tmp_path, clase):
    c, app = construir(raiz_tk, clase)
    assert app.cargar_imagen(imagen(tmp_path)) and app._clasificar_sync()
    anterior = c.imagen
    (tmp_path / "x.png").write_bytes(b"no es imagen")
    assert app.cargar_imagen(tmp_path / "x.png") is False
    assert "dañado" in app._ultimo_error and c.imagen is anterior and str(app.b_clasificar.cget("state")) == "disabled" and app.clase.cget("text") == "—"
    assert app.cargar_imagen(imagen(tmp_path, "otra.png")) and app._ultimo_error == "" and str(app.b_clasificar.cget("state")) == "normal"


@pytest.mark.parametrize("clase", [AppTk, AppTkV2])
def test_modo_demostracion_sin_resultados_simulados(raiz_tk, tmp_path, clase):
    def falla():
        raise SistemaNoDisponible("sin checkpoints")
    c = Controlador(cargar=falla)
    c.iniciar()
    app = clase(raiz_tk, c, usar_hilo=False)
    assert app.clase.cget("text") == T.MODO_MAQUETA and T.MODO_MAQUETA in app.estado.cget("text") and "no se muestran resultados simulados" in app.estado.cget("text")
    assert app.cargar_imagen(imagen(tmp_path)) and str(app.b_clasificar.cget("state")) == "disabled"
    assert app._clasificar_sync() is False and c.resultado is None and app.clase.cget("text") == T.MODO_MAQUETA


@pytest.mark.parametrize("clase", [AppTk, AppTkV2])
def test_la_clasificacion_con_hilo_de_trabajo_tambien_funciona(raiz_tk, tmp_path, clase):
    c = Controlador(cargar=lambda: sistema_falso(0.3))
    c.iniciar()
    app = clase(raiz_tk, c, usar_hilo=True)
    app.cargar_imagen(imagen(tmp_path))
    app.clasificar()
    for _ in range(200):
        raiz_tk.update()
        if app.clase.cget("text") == "MALIGNO":
            break
        raiz_tk.after(20)
    assert app.clase.cget("text") == "MALIGNO" and c.resultado is not None


@pytest.mark.parametrize("clase", [AppTk, AppTkV2])
def test_las_dos_presentaciones_dan_exactamente_el_mismo_resultado(raiz_tk, tmp_path, clase):
    c, app = construir(raiz_tk, clase, 0.2)
    app.cargar_imagen(imagen(tmp_path))
    app._clasificar_sync()
    r = c.resultado
    assert (r.clase, r.puntuacion, r.umbral, r.n_miembros, r.version_interfaz) == ("MALIGNO", pytest.approx(0.2, abs=1e-7), float(T.T_HAT_TEXTO), 15, "r9_gui_v1")


# ============================================================================ estructura de la presentación v2
def test_v2_encabezado_estado_de_verificacion_aviso_y_presentacion(raiz_tk):
    c, app = construir(raiz_tk, AppTkV2)
    assert "15/15 modelos verificados (SHA-256)" in app.chip.cget("text")
    titulo, subtitulo = T.TITULO.split(" — ", 1)
    assert app.version.cget("text") == "r9_gui_v1 · presentación v2" and app.aviso.cget("text") == T.AVISO_PRINCIPAL and app.presentacion.cget("text") == T.PRESENTACION
    assert "Sistema verificado" in app.estado.cget("text") and titulo in raiz_tk.title()


def test_v2_el_chip_distingue_entorno_distinto_y_modo_demostracion(raiz_tk):
    _, app = construir(raiz_tk, AppTkV2, advertencias_entorno=("torch: verificado con '2.5.1+cu121', este equipo '2.10.0'",))
    assert "entorno distinto" in app.chip.cget("text") and "ATENCIÓN, entorno distinto" in app.estado.cget("text")

    def falla():
        raise SistemaNoDisponible("x")
    c = Controlador(cargar=falla)
    c.iniciar()
    app2 = AppTkV2(tk.Toplevel(raiz_tk), c, usar_hilo=False)
    assert T.MODO_MAQUETA in app2.chip.cget("text")


def test_v2_la_informacion_metodologica_esta_en_su_pestana_con_todos_los_avisos(raiz_tk):
    _, app = construir(raiz_tk, AppTkV2)
    assert [app.notebook.tab(t, "text") for t in app.notebook.tabs()] == ["Clasificación", "Información metodológica"]
    info = app.avisos.get("1.0", "end")
    for seccion, texto in zip(("Aviso principal", "Alcance", "Etiquetas y mapeo v2", "Puntuación y umbral", "Nota numérica", "Rendimiento (evaluación final R8)"), T.advertencias()):
        assert seccion in info and texto in info
    assert T.ACERCA_DE in info and "No se garantiza" in info and "0.60" in info and "0.08456226376195748" in info
    app.mostrar_informacion()
    assert app.notebook.index(app.notebook.select()) == 1
    app.mostrar_clasificacion()
    assert app.notebook.index(app.notebook.select()) == 0


def test_v2_la_zona_principal_no_repite_los_textos_extensos_de_la_informacion_metodologica(raiz_tk, tmp_path):
    c, app = construir(raiz_tk, AppTkV2)
    app.cargar_imagen(imagen(tmp_path))
    app._clasificar_sync()
    principal = " ".join([app.aviso.cget("text"), app.presentacion.cget("text"), app.detalle.cget("text"), app.estado.cget("text"), app.chip.cget("text"), app.info.cget("text")])
    for largo in (T.ALCANCE, T.NOTA_ETIQUETAS, T.NOTA_PUNTUACION, T.NOTA_NUMERICA, T.RENDIMIENTO):
        assert largo not in principal
    assert T.AVISO_PRINCIPAL in principal and "prototipo básico de sistema de apoyo diagnóstico" in principal.lower()      # el aviso de no validez diagnóstica sigue visible


def test_v2_botones_consistentes_y_estilos_ttk(raiz_tk):
    _, app = construir(raiz_tk, AppTkV2)
    assert isinstance(app.b_cargar, ttk.Button) and isinstance(app.b_clasificar, ttk.Button)
    assert str(app.b_cargar.cget("style")) == "Secundario.TButton" and str(app.b_clasificar.cget("style")) == "Primario.TButton"
    assert app.b_cargar.cget("text") == "Cargar imagen…" and app.b_clasificar.cget("text") == "Clasificar"                  # textos originales


def test_v2_color_de_la_clase_y_neutro_antes_de_clasificar(raiz_tk, tmp_path):
    c, app = construir(raiz_tk, AppTkV2, 0.3)
    neutro = app.clase.cget("bg")
    app.cargar_imagen(imagen(tmp_path))
    assert app.clase.cget("text") == "—" and app.clase.cget("bg") == neutro
    app._clasificar_sync()
    assert app.clase.cget("bg") == "#fbe4e4" and app.clase.cget("fg") == "#8b0000"
    c2, app2 = construir(tk.Toplevel(raiz_tk), AppTkV2, 0.01)
    app2.cargar_imagen(imagen(tmp_path))
    app2._clasificar_sync()
    assert app2.clase.cget("text") == "BENIGNO" and app2.clase.cget("bg") == "#e3f3ee"


# ============================================================================ distribución adaptable (ventana visible pero transparente)
def mostrar(raiz, ancho, alto):
    pantalla = (raiz.winfo_screenwidth(), raiz.winfo_screenheight())
    if ancho > pantalla[0] or alto > pantalla[1]:
        pytest.skip("pantalla más pequeña que el tamaño de prueba")
    raiz.attributes("-alpha", 0.0)
    raiz.deiconify()
    raiz.geometry(f"{ancho}x{alto}+0+0")
    for _ in range(4):
        raiz.update()


def dentro(raiz, w):
    x, y = w.winfo_rootx() - raiz.winfo_rootx(), w.winfo_rooty() - raiz.winfo_rooty()
    return x >= 0 and y >= 0 and x + w.winfo_width() <= raiz.winfo_width() + 1 and y + w.winfo_height() <= raiz.winfo_height() + 1


@pytest.mark.parametrize("ancho,alto", [(900, 660), (1180, 780), (1500, 900)])
def test_v2_los_componentes_clave_caben_sin_solaparse_a_varios_tamanos(raiz_tk, tmp_path, ancho, alto):
    c, app = construir(raiz_tk, AppTkV2)
    mostrar(raiz_tk, ancho, alto)
    app.cargar_imagen(imagen(tmp_path, tam=(800, 600)))
    app._clasificar_sync()
    for _ in range(4):
        raiz_tk.update()
    claves = (app.chip, app.aviso, app.presentacion, app.b_cargar, app.b_clasificar, app.marco_imagen, app.clase, app.detalle, app.estado, app.info)
    for w in claves:
        assert w.winfo_viewable() and dentro(raiz_tk, w), (w, w.winfo_rootx(), w.winfo_rooty(), w.winfo_width(), w.winfo_height())
    for w in (app.aviso, app.presentacion, app.estado, app.detalle, app.clase):
        assert w.winfo_height() >= w.winfo_reqheight() - 1, (w.cget("text")[:30], w.winfo_height(), w.winfo_reqheight())         # texto sin recortar
    assert app.marco_imagen.winfo_rootx() + app.marco_imagen.winfo_width() <= app.clase.winfo_rootx()                              # paneles izquierdo y derecho sin solaparse
    assert app.b_cargar.winfo_rootx() + app.b_cargar.winfo_width() <= app.b_clasificar.winfo_rootx() + 1


def test_v2_la_imagen_se_adapta_al_tamano_de_la_ventana(raiz_tk, tmp_path):
    c, app = construir(raiz_tk, AppTkV2)
    mostrar(raiz_tk, 900, 660)
    app.cargar_imagen(imagen(tmp_path, tam=(1600, 1200)))
    for _ in range(6):
        raiz_tk.update()
        raiz_tk.after(80)
    pequena = app._foto.width()
    mostrar(raiz_tk, 1500, 900)
    for _ in range(8):
        raiz_tk.update()
        raiz_tk.after(80)
    raiz_tk.update()
    assert app._foto.width() > pequena and app._foto.width() <= app.marco_imagen.winfo_width()
    assert app._foto.width() / app._foto.height() == pytest.approx(4 / 3, abs=0.02)                                              # conserva la relación de aspecto


# ============================================================================ alcance del código de la presentación v2
def test_v2_no_contiene_inferencia_metricas_ni_dependencias_nuevas():
    fuente = (Path(__file__).resolve().parents[2] / "src" / "gui" / "app_tk_v2.py").read_text(encoding="utf-8")
    bajo = fuente.lower()
    for prohibido in ("import torch", "torchvision", "cargar_sistema", "sistema_final", "average_precision", "roc_auc", "calibrar", "bootstrap", ".fit(", "umbral_alt", "abst", "open(", "read_bytes", "write_"):
        assert prohibido not in bajo, prohibido
    importados = {linea.split()[1].split(".")[0] for linea in fuente.splitlines() if linea.startswith(("import ", "from "))}
    assert importados <= {"__future__", "argparse", "tkinter", "pathlib", "src", "queue"}, importados


# ============================================================================ el ejecutor de evidencias (D-076) también valida la presentación v2
def test_el_ejecutor_de_pruebas_funcionales_supera_los_20_casos_con_la_presentacion_v2(raiz_tk, tmp_path):
    from src.gui import evidencia_funcional as E
    from tests.gui.test_evidencia_funcional import controlador as ctl, crear_desarrollo, referencia_g2
    c = ctl()
    salida = tmp_path / "salida"
    salida.mkdir()
    reg = E.ejecutar(tmp_path / "repo", salida, raiz_tk, c, imagenes_desarrollo=crear_desarrollo(tmp_path / "repo", 2), capturas=False, tensor_g2=referencia_g2(c), clase_app=AppTkV2)
    fallos = [(x["id"], x["observado"]) for x in reg["casos"] if x["resultado"] != "superada"]
    assert not fallos and reg["total"] == 20


def test_la_cli_del_ejecutor_acepta_la_presentacion_v2_y_la_registra(raiz_tk, tmp_path):
    import json
    from src.gui import evidencia_funcional as E
    from tests.gui.test_evidencia_funcional import controlador as ctl, crear_desarrollo, referencia_g2
    crear_desarrollo(tmp_path, 3)
    c = ctl()
    kw = dict(root=tmp_path, controlador=c, tensor_g2=referencia_g2(c), raiz_tk=tk.Toplevel(raiz_tk))
    assert E.main(["--autorizacion", "CONFIRMADO:D-076", "--sin-capturas", "--interfaz", "v2"], **kw) == 3      # la presentación v2 exige su propia decisión (D-078)
    assert E.main(["--autorizacion", "CONFIRMADO:D-078", "--sin-capturas"], **kw) == 3                          # y la v1 sigue exigiendo D-076
    assert not (tmp_path / "results").exists()
    assert E.main(["--autorizacion", "CONFIRMADO:D-078", "--sin-capturas", "--interfaz", "v2"], **kw) == 0
    j = json.loads((tmp_path / E.SALIDA_REL / "D-078" / "registro_pruebas_funcionales.json").read_text(encoding="utf-8"))
    assert j["presentacion"] == "v2" and j["todos_superados"] is True and j["autorizacion"] == "CONFIRMADO:D-078"
