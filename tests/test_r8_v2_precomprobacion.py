"""Pruebas sintéticas de la precomprobación estructural de R8 (`r8_v2.precomprobacion`): los tres supuestos del manifiesto, su equivalencia con las compuertas del ejecutor, ausencia de bloqueo
de ejecución única, de lectura de imágenes y de datos por imagen en el informe. Todo sintético; no abren el test real.
Comando: py -3.13 -B -m pytest tests/test_r8_v2_precomprobacion.py -q -p no:cacheprovider
"""

from __future__ import annotations

import inspect
import json
import shutil

import pytest

pytest.importorskip("yaml")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from src.r4_v2.common import GuardError  # noqa: E402
from src.r7_v2 import umbral_sistema as U  # noqa: E402
from src.r8_v2 import evaluacion_final as E  # noqa: E402
from src.r8_v2 import precomprobacion as P  # noqa: E402
from tests.test_r8_v2_evaluacion_final import Banco, escribir_manifiesto, filas_manifiesto  # noqa: E402

AUT = "CONFIRMADO:D-063"


def correr(b, **kw):
    return P.precomprobar(b.root, autorizacion=kw.pop("autorizacion", AUT), sha256_protocolo=kw.pop("sha256_protocolo", b.sha_proto), **kw)


def ejecutor_acepta(root) -> bool:
    """¿Superaría el ejecutor la lectura del manifiesto y la resolución de todas las imágenes? (la misma lógica que usa tras el bloqueo)"""
    try:
        filas, _, _ = E.leer_manifiesto_prueba(root, E.leer_ids_desarrollo(root)[:2])
        for f in filas:
            E.resolver_imagen_prueba(root, f)
        return True
    except GuardError:
        return False


@pytest.fixture
def banco(tmp_path, monkeypatch):
    return Banco(tmp_path, monkeypatch)


def test_manifiesto_conforme_es_apto_y_no_crea_bloqueo(banco):
    r = correr(banco)
    x = r["resultado"]
    assert x["apto_para_ejecutar"] and ejecutor_acepta(banco.root)
    assert x["supuesto_1_origen_mmotu"] == {"cumple": True, "esperado": "validation", "valores_de_source_split_en_filas_test": {"validation": 469}, "filas_fuera_del_test_con_ese_origen": 0}
    assert x["supuesto_2_recuentos"]["observados"] == {"elegibles_clase_0": 367, "elegibles_clase_1": 15, "normales_excluidas": 87}
    assert x["supuesto_3_imagenes"]["imagenes_elegibles"] == 382 and x["supuesto_3_imagenes"]["imagenes_faltantes"] == 0
    assert not (banco.root / E.SALIDA_REL).exists()                                            # NO crea el bloqueo de evaluación única
    assert (r["dir"] / "resumen_precomprobacion.json").is_file() and "precomprobacion_estructural" in str(r["dir"])


def test_el_informe_no_contiene_datos_por_imagen(banco):
    texto = (correr(banco)["dir"] / "resumen_precomprobacion.json").read_text(encoding="utf-8")
    assert "T0_" not in texto and "T1_" not in texto and "TN_" not in texto and "original_class" not in texto
    assert set(json.loads(texto)["resultado"]) == {"manifiesto_presente", "sha256_bytes_archivo_manifiesto", "columnas_requeridas_presentes", "filas_con_experimental_split_test",
                                                   "supuesto_1_origen_mmotu", "supuesto_2_recuentos", "supuesto_3_imagenes", "supuesto_4_sin_solapamiento_con_desarrollo", "ids_o_nombres_repetidos_o_vacios",
                                                   "nombres_inseguros", "apto_para_ejecutar"}


def test_cada_supuesto_incumplido_se_informa_y_coincide_con_el_ejecutor(tmp_path, monkeypatch):
    def source(f):
        f[3]["source_split"] = "train"

    def recuentos(f):
        f[0]["original_class"] = 5

    def categoria(f):
        f[0]["original_class"] = 8

    def repetido(f):
        f[1]["image_id"] = f[0]["image_id"]

    def inseguro(f):
        f[1]["image_name"] = "../x.png"

    def source_test(f):
        for x in f:
            if x["experimental_split"] == "test":
                x["source_split"] = "test"                                                        # el valor erróneo supuesto en la versión inicial

    def desarrollo_validation(f):
        f[-1]["source_split"] = "validation"

    def solapa_manifiesto(f):
        f[-1]["image_id"] = "T0_0000"

    def solapa_particion(f):
        f[0]["image_id"] = "DEV_0"

    casos = [("source", source, "supuesto_1_origen_mmotu"), ("source_test", source_test, "supuesto_1_origen_mmotu"), ("dev_validation", desarrollo_validation, "supuesto_1_origen_mmotu"),
             ("recuentos", recuentos, "supuesto_2_recuentos"), ("categoria", categoria, "supuesto_2_recuentos"), ("repetido", repetido, None), ("inseguro", inseguro, None),
             ("solapa_manifiesto", solapa_manifiesto, "supuesto_4_sin_solapamiento_con_desarrollo"), ("solapa_particion", solapa_particion, "supuesto_4_sin_solapamiento_con_desarrollo")]
    for nombre, mut, clave in casos:
        filas = filas_manifiesto()
        mut(filas)
        b = Banco(tmp_path / nombre, monkeypatch, filas)
        x = correr(b)["resultado"]
        assert x["apto_para_ejecutar"] is False and ejecutor_acepta(b.root) is False, nombre
        if clave:
            assert x[clave]["cumple"] is False, nombre


def test_imagenes_faltantes_se_cuentan_sin_abrirlas(banco):
    d = banco.root / E.IMAGENES_DIR / "test" / "images"
    for k in (1, 2, 3, 4, 5, 6, 7):
        (d / f"T0_{k:04d}.png").unlink()
    x = correr(banco)["resultado"]["supuesto_3_imagenes"]
    assert x["cumple"] is False and x["imagenes_faltantes"] == 7 and len(x["primeras_faltantes"]) == P.MAX_NOMBRES_FALTANTES
    assert ejecutor_acepta(banco.root) is False


def test_directorio_de_imagenes_ausente(banco):
    shutil.rmtree(banco.root / E.IMAGENES_DIR / "test")
    x = correr(banco)["resultado"]["supuesto_3_imagenes"]
    assert x["cumple"] is False and x["directorio_presente"] is False


def test_manifiesto_sin_columnas_o_ausente(banco):
    (banco.root / E.MANIFIESTO_TEST_REL).write_text("a,b\n1,2\n", encoding="utf-8")
    x = correr(banco)["resultado"]
    assert x["apto_para_ejecutar"] is False and set(x["columnas_faltantes"]) == set(E.COLUMNAS_MANIFIESTO)
    (banco.root / E.MANIFIESTO_TEST_REL).unlink()
    assert correr(banco, sufijo="_2")["resultado"] == {"manifiesto_presente": False, "apto_para_ejecutar": False}


@pytest.mark.parametrize("kw", [{"autorizacion": ""}, {"autorizacion": "CONFIRMADO:D-63"}, {"sha256_protocolo": "0" * 64}, {"sha256_protocolo": "abc"}])
def test_compuertas_fallidas_no_leen_ni_escriben(banco, monkeypatch, kw):
    def prohibido(*a, **k):
        raise AssertionError("se leyó el manifiesto sin superar las compuertas")
    monkeypatch.setattr(P, "analizar", prohibido)
    with pytest.raises(GuardError):
        correr(banco, **kw)
    assert not (banco.root / "results" / "r8_v2" / "phash_le8_v1" / "precomprobacion_estructural").exists()


def test_no_sobrescribe_y_no_escribe_en_el_directorio_de_la_evaluacion_unica(banco):
    correr(banco)
    with pytest.raises(GuardError, match="ya existe"):
        correr(banco)
    assert correr(banco, sufijo="_2")["dir"].name == "D-063_2"
    for malo in (banco.root / E.SALIDA_REL, banco.root / E.SALIDA_REL / "x", banco.root / "results" / "r7_v2" / "x"):
        with pytest.raises(GuardError):
            correr(banco, destino=malo)
    assert not (banco.root / E.SALIDA_REL).exists()


def test_la_evaluacion_unica_sigue_disponible_tras_la_precomprobacion(banco):
    correr(banco)
    r = banco.ejecutar()
    assert r["reporte_aceptado"] and (banco.root / E.SALIDA_REL / "ejecucion.lock.json").is_file()


def test_no_abre_imagenes_ni_infiere():
    fuente = inspect.getsource(P)
    assert "PIL" not in fuente and "torch" not in fuente and "Image.open" not in fuente
    assert not any(hasattr(P, n) for n in ("inferir", "predecir", "evaluar", "calcular_resultados"))


def test_cli_codigos_de_salida(banco):
    comunes = ["--autorizacion", AUT, "--sha256-protocolo", banco.sha_proto]
    assert P.main(comunes, root=banco.root) == 0
    assert P.main([*comunes, "--sufijo", "_2"], root=banco.root) == 0
    assert P.main(comunes, root=banco.root) == U.CODIGO_SALIDA_GUARDA                      # ya existe
    assert P.main(["--autorizacion", AUT, "--sha256-protocolo", "0" * 64], root=banco.root) == U.CODIGO_SALIDA_GUARDA
    escribir_manifiesto(banco.root, filas_manifiesto(n0=366))
    assert P.main([*comunes, "--sufijo", "_3"], root=banco.root) == 2                       # no apto


def test_particion_de_desarrollo_ausente_hace_no_apto_el_supuesto_4(banco):
    (banco.root / E.C.REL_PARTICION).unlink()
    x = correr(banco)["resultado"]
    assert x["supuesto_4_sin_solapamiento_con_desarrollo"]["particion_v2_presente"] is False and x["supuesto_4_sin_solapamiento_con_desarrollo"]["cumple"] is False
    assert x["apto_para_ejecutar"] is False


def test_el_caso_real_observado_por_la_autora_se_reproduce_sintetico_y_ahora_es_apto(banco):
    """Caso D-063 (exit=2): 469 filas con source_split = validation y recuentos 367/15/87 correctos. Con la Enmienda 1 es apto."""
    x = correr(banco)["resultado"]
    assert x["filas_con_experimental_split_test"] == 469 and x["supuesto_1_origen_mmotu"]["valores_de_source_split_en_filas_test"] == {"validation": 469}
    assert x["supuesto_4_sin_solapamiento_con_desarrollo"] == {"cumple": True, "solapamiento_con_resto_del_manifiesto": 0, "particion_v2_presente": True,
                                                               "solapamiento_con_particion_v2": 0, "sha256_bytes_archivo_particion_v2": x["supuesto_4_sin_solapamiento_con_desarrollo"]["sha256_bytes_archivo_particion_v2"]}
    assert x["apto_para_ejecutar"] is True
