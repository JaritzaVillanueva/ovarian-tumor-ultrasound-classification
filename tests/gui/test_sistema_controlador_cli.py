"""R9 (F2, F3 y modo demostración): sistema congelado (carga verificada y clasificación con los 15 miembros), controlador y CLI. Sintético: modelos y preprocesamiento de prueba; los checkpoints reales
solo se usan en Falcon (`src.gui.cli verificar` / `integracion-desarrollo`). Nunca test.
Comando: py -3.13 -B -m pytest tests/gui -q -p no:cacheprovider
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("yaml")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")
pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from src.r7_v2 import sistema_final as SF  # noqa: E402
from src.gui import cli, textos as T  # noqa: E402
from src.gui.controlador import Controlador  # noqa: E402
from src.gui.sistema import SHA256_DECLARACION, SistemaCongelado, SistemaNoDisponible, cargar_sistema  # noqa: E402
from src.gui.validacion_imagen import ImagenInvalida  # noqa: E402
from tests.test_r8_v2_evaluacion_final import Banco  # noqa: E402

T_HAT = float(SF.T_HAT)


def miembros_falsos():
    return [{"id_corrida": f"densenet121__s{s}__f{k}", "semilla": s, "fold": k, "intento": "intento_01"} for s in (42, 43, 44) for k in range(5)]


def sistema_falso(probabilidades, **kw):
    """Sistema con 15 miembros de prueba: `probabilidades` es un float (todos iguales) o una lista de 15."""
    ms = miembros_falsos()
    ps = [probabilidades] * 15 if isinstance(probabilidades, float) else list(probabilidades)
    modelos = {m["id_corrida"]: {"p": p} for m, p in zip(ms, ps)}
    return SistemaCongelado(miembros=ms, modelos=modelos, preprocesar=lambda im: np.asarray(im, dtype=np.float32), predecir=lambda modelo, tensor, device: modelo["p"], device="cpu",
                            t_hat_texto=kw.pop("t_hat_texto", SF.T_HAT), **kw)


def imagen(tmp_path, nombre="a.png", tam=(64, 48)):
    p = tmp_path / nombre
    Image.new("RGB", tam, (90, 90, 90)).save(p)
    return p


# ============================================================================ F3: clasificación con el sistema congelado
def test_arriba_del_umbral_es_maligno_y_debajo_benigno(tmp_path):
    img = Image.new("RGB", (40, 40))
    alto, bajo = sistema_falso(0.0846).clasificar(img), sistema_falso(0.0845).clasificar(img)
    assert (alto.clase, alto.positivo) == ("MALIGNO", True) and (bajo.clase, bajo.positivo) == ("BENIGNO", False)
    assert alto.umbral == T_HAT and alto.umbral_texto == "0.08456226376195748" and alto.regla == "p >= t_hat" and alto.n_miembros == 15


def test_la_puntuacion_es_el_promedio_aritmetico_simple_float64_de_los_float32():
    rng = np.random.default_rng(3)
    ps = [float(np.float32(x)) for x in rng.uniform(0, 0.4, 15)]
    r = sistema_falso(ps).clasificar(Image.new("RGB", (40, 40)))
    esperado = float(np.mean(np.asarray(ps, dtype=np.float32).astype(np.float64)))
    assert r.puntuacion == pytest.approx(esperado, abs=1e-15) and r.positivo == (esperado >= T_HAT)
    assert set(r.probabilidades_por_miembro) == {m["id_corrida"] for m in miembros_falsos()} and len(r.probabilidades_por_miembro) == 15


def test_delega_en_la_funcion_congelada_con_el_umbral_exacto(monkeypatch):
    visto = {}
    original = SF.agregar_y_decidir

    def espia(probs, t_hat=SF.T_HAT):
        visto["t_hat"], visto["n"] = t_hat, len(probs)
        return original(probs, t_hat)
    monkeypatch.setattr(SF, "agregar_y_decidir", espia)
    sistema_falso(0.2).clasificar(Image.new("RGB", (40, 40)))
    assert visto == {"t_hat": "0.08456226376195748", "n": 15}


def test_el_preprocesamiento_se_aplica_una_vez_y_el_mismo_tensor_va_a_los_15_miembros():
    llamadas, tensores = [], []
    ms = miembros_falsos()
    s = SistemaCongelado(miembros=ms, modelos={m["id_corrida"]: {} for m in ms}, preprocesar=lambda im: llamadas.append(1) or np.ones((1, 3, 4, 4), dtype=np.float32),
                         predecir=lambda modelo, tensor, device: tensores.append(id(tensor)) or 0.1, device="cpu", t_hat_texto=SF.T_HAT)
    s.clasificar(Image.new("RGB", (40, 40)))
    assert len(llamadas) == 1 and len(tensores) == 15 and len(set(tensores)) == 1


def test_el_sistema_rechaza_umbral_o_miembros_distintos_de_los_congelados():
    ms = miembros_falsos()
    with pytest.raises(SistemaNoDisponible):
        sistema_falso(0.1, t_hat_texto="0.1")
    with pytest.raises(SistemaNoDisponible):
        SistemaCongelado(miembros=ms[:14], modelos={m["id_corrida"]: {} for m in ms[:14]}, preprocesar=lambda x: x, predecir=lambda *a: 0.1, device="cpu", t_hat_texto=SF.T_HAT)
    with pytest.raises(SistemaNoDisponible):
        SistemaCongelado(miembros=ms, modelos={m["id_corrida"]: {} for m in ms[:14]}, preprocesar=lambda x: x, predecir=lambda *a: 0.1, device="cpu", t_hat_texto=SF.T_HAT)


def test_equivalencia_del_preprocesamiento_con_el_lector_de_g2(tmp_path):
    """La ruta de la interfaz (validar_y_cargar + transformación) produce el mismo tensor que `DatasetLesiones` de G2 para la misma imagen."""
    pytest.importorskip("torch")
    from src.r4_v2.dataset_lesiones import DatasetLesiones
    from src.gui.validacion_imagen import validar_y_cargar
    p = imagen(tmp_path, "dev.png", (48, 40))
    carpeta = tmp_path / "data" / "processed" / "MMOTU_OTU_2D_experimental" / "train" / "images"
    carpeta.mkdir(parents=True)
    (carpeta / "dev.png").write_bytes(p.read_bytes())
    tr = lambda im: np.asarray(im, dtype=np.float32)  # noqa: E731
    g2 = DatasetLesiones(tmp_path, [{"image_path": "data/processed/MMOTU_OTU_2D_experimental/train/images/dev.png", "image_name": "dev.png", "label_v2": 0}], tr)[0][0]
    assert np.array_equal(g2, tr(validar_y_cargar(p)[0]))


def test_prediccion_de_un_miembro_con_torch_es_softmax_float32_de_la_clase_maligna(tmp_path):
    torch = pytest.importorskip("torch")
    from src.gui.sistema import _predecir_torch
    torch.manual_seed(1)
    modelo = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 2))
    x = torch.rand(1, 3, 4, 4)
    esperado = float(torch.softmax(modelo(x), dim=1)[0, 1].detach().numpy().astype(np.float32))
    assert _predecir_torch(modelo, x, "cpu") == pytest.approx(esperado, abs=1e-7) and 0.0 <= esperado <= 1.0


# ============================================================================ carga verificada del sistema congelado (raíz sintética con la declaración, el manifiesto y checkpoints)
@pytest.fixture
def banco(tmp_path, monkeypatch):
    return Banco(tmp_path, monkeypatch)


def cargar(banco, **kw):
    args = dict(sha256_declaracion=banco.sha_decl, cargar_modelo=lambda ruta, dev: {"ruta": str(ruta)}, device="cpu", preprocesar=lambda im: im, predecir=lambda m, t, d: 0.1)
    args.update(kw)
    return cargar_sistema(banco.root, **args)


def test_carga_verificada_del_sistema_congelado(banco):
    s = cargar(banco)
    assert len(s.miembros) == 15 and len(s.modelos) == 15 and s.t_hat_texto == "0.08456226376195748" and isinstance(s.advertencias_entorno, tuple)
    assert s.clasificar(Image.new("RGB", (40, 40))).n_miembros == 15


def test_el_sha256_de_la_declaracion_por_defecto_es_el_de_d060():
    assert SHA256_DECLARACION == "2dd79df4815b676e1bfe5c880ebb1040548bf57d21de3bbf91652842a8598cf1"


@pytest.mark.parametrize("que", ["sha_declaracion", "checkpoint_alterado", "checkpoint_ausente", "yaml", "manifiesto", "codigo_inferencia", "declaracion_ausente", "config_g2"])
def test_cualquier_fallo_de_verificacion_deja_el_sistema_no_disponible(banco, que):
    d = banco.decl
    if que == "sha_declaracion":
        kw = {"sha256_declaracion": "0" * 64}
    else:
        kw = {}
        ruta = {"checkpoint_alterado": d["miembros"][2]["ruta_checkpoint"], "checkpoint_ausente": d["miembros"][5]["ruta_checkpoint"], "yaml": "configs/r8_v2/sistema_congelado_v1.yaml",
                "manifiesto": "configs/r8_v2/manifiesto_miembros_v1.csv", "codigo_inferencia": "src/classification/transforms.py",
                "declaracion_ausente": "configs/r8_v2/declaracion_congelamiento_v1.json", "config_g2": "configs/r4_v2/g2_baseline_v1.yaml"}[que]
        p = banco.root / ruta
        if que in ("checkpoint_ausente", "declaracion_ausente"):
            p.unlink()
        else:
            p.write_bytes(p.read_bytes() + b"\n# alterado\n")
    with pytest.raises(SistemaNoDisponible):
        cargar(banco, **kw)


def test_exige_la_verificacion_de_la_declaracion_aunque_se_inyecte_un_cargador(banco):
    def rota(root, sha):
        raise SistemaNoDisponible("verificación inyectada")
    with pytest.raises(SistemaNoDisponible, match="verificación inyectada"):
        cargar(banco, verificar_declaracion=rota)


# ============================================================================ controlador: modo demostración sin inferencia
def test_modo_demostracion_sin_resultados_simulados(tmp_path):
    def falla():
        raise SistemaNoDisponible("faltan los checkpoints")
    c = Controlador(cargar=falla)
    c.iniciar()
    assert not c.listo and c.estado_texto().startswith(T.MODO_MAQUETA) and "faltan los checkpoints" in c.estado_texto() and "no se muestran resultados simulados" in c.estado_texto()
    c.abrir(imagen(tmp_path))                                                                  # se puede visualizar la imagen
    with pytest.raises(SistemaNoDisponible, match="Modo demostración"):
        c.clasificar()
    assert c.resultado is None


def test_flujo_cargar_visualizar_clasificar(tmp_path):
    c = Controlador(cargar=lambda: sistema_falso(0.3))
    c.iniciar()
    assert c.listo and "Sistema verificado" in c.estado_texto()
    with pytest.raises(ImagenInvalida, match="Primero cargue"):
        c.clasificar()
    info = c.abrir(imagen(tmp_path))
    assert info["ancho"] == 64 and c.imagen.mode == "RGB"
    r = c.clasificar()
    lineas = "\n".join(Controlador.lineas_resultado(r))
    assert r.clase == "MALIGNO" and "Clase asignada: MALIGNO" in lineas and "0.08456226376195748" in lineas and "no es una probabilidad calibrada" in lineas and SHA256_DECLARACION in lineas
    assert Controlador.avisos() == T.advertencias()


def test_una_imagen_invalida_conserva_la_anterior(tmp_path):
    c = Controlador(cargar=lambda: sistema_falso(0.3))
    c.iniciar()
    c.abrir(imagen(tmp_path, "ok.png"))
    anterior = c.imagen
    (tmp_path / "mala.png").write_bytes(b"x")
    with pytest.raises(ImagenInvalida):
        c.abrir(tmp_path / "mala.png")
    assert c.imagen is anterior


def test_el_entorno_distinto_se_muestra_como_atencion(tmp_path):
    c = Controlador(cargar=lambda: sistema_falso(0.3, advertencias_entorno=("torch: verificado con '2.5.1+cu121', este equipo '2.10.0'",)))
    c.iniciar()
    assert "ATENCIÓN, entorno distinto" in c.estado_texto()
    c.abrir(imagen(tmp_path))
    assert "Atención, entorno distinto" in "\n".join(Controlador.lineas_resultado(c.clasificar()))


# ============================================================================ CLI sin pantalla
def test_cli_clasificar_json(tmp_path, capsys):
    assert cli.main(["clasificar", "--imagen", str(imagen(tmp_path)), "--json"], root=tmp_path, sistema=sistema_falso(0.3)) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["clase"] == "MALIGNO" and out["umbral"] == "0.08456226376195748" and out["n_miembros"] == 15 and len(out["probabilidades_por_miembro"]) == 15 and out["advertencias"] == T.advertencias()


def test_cli_clasificar_texto_incluye_los_avisos(tmp_path, capsys):
    assert cli.main(["clasificar", "--imagen", str(imagen(tmp_path))], root=tmp_path, sistema=sistema_falso(0.01)) == 0
    out = capsys.readouterr().out
    assert "Clase asignada: BENIGNO" in out and "AVISOS" in out and "no es un dispositivo médico" in out.lower() and "sensibilidad 0.60" in out


def test_cli_codigos_de_salida(tmp_path, capsys):
    ok = sistema_falso(0.3)
    assert cli.main(["clasificar", "--imagen", str(tmp_path / "no.png")], root=tmp_path, sistema=ok) == 3
    prohibida = tmp_path / "test" / "images"
    prohibida.mkdir(parents=True)
    assert cli.main(["clasificar", "--imagen", str(imagen(prohibida))], root=tmp_path, sistema=ok) == 3
    assert "test" in capsys.readouterr().err
    assert cli.main(["clasificar", "--imagen", str(imagen(tmp_path))], root=tmp_path) == 2               # sin checkpoints: modo demostración
    assert "Modo demostración: modelo no disponible" in capsys.readouterr().err
    assert cli.main(["verificar"], root=tmp_path, sistema=ok) == 3                                       # sin autorización
    assert cli.main(["integracion-desarrollo", "--autorizacion", "CONFIRMADO"], root=tmp_path, sistema=ok) == 3


def test_la_logica_y_la_cli_se_importan_y_funcionan_sin_tkinter():
    """Portabilidad Windows/Falcon: ni la CLI, ni el controlador, ni el sistema, ni la validación importan tkinter (solo `app_tk`)."""
    repo = Path(__file__).resolve().parents[2]
    codigo = ("import sys; sys.modules['tkinter'] = None; sys.modules['PIL.ImageTk'] = None\n"
              "import src.gui.cli, src.gui.controlador, src.gui.sistema, src.gui.validacion_imagen, src.gui.textos\n"
              "assert 'src.gui.app_tk' not in sys.modules\n"
              "from src.gui.controlador import Controlador\n"
              "print('ok')")
    r = subprocess.run([sys.executable, "-B", "-c", codigo], cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-400:]
    r2 = subprocess.run([sys.executable, "-B", "-c", "import sys; sys.modules['tkinter'] = None\nimport src.gui.app_tk"], cwd=repo, capture_output=True, text=True)
    assert r2.returncode != 0                                                                        # solo la ventana exige tkinter
