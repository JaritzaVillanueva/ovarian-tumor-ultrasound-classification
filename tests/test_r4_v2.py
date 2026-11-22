"""Pruebas de G2 `g2_baseline_v1` (protocolo §18) con datos SINTÉTICOS.

No abren imágenes reales, particiones reales (`configs/splits/v2_auto/` no existe en el clon), el test oficial ni el
árbol `data/`. Los repositorios Git y los CSV se crean en directorios temporales fuera del proyecto. Lo que requiere
torch se salta localmente (`pytest.importorskip("torch")`) y queda [ND] hasta Falcon.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from src.r4_v2 import common as C
from src.r4_v2 import particion as P

REPO = Path(__file__).resolve().parents[1]
CONFIG_REAL = REPO / "configs" / "r4_v2" / "g2_baseline_v1.yaml"
yaml = pytest.importorskip("yaml")


@pytest.fixture(autouse=True)
def _git_aislado(monkeypatch):
    """Los repositorios Git temporales no deben depender de la configuración global/sistema del autor."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")


@pytest.fixture(scope="module")
def config_base():
    return C.cargar_config(CONFIG_REAL)


@pytest.fixture
def config(config_base):
    return copy.deepcopy(config_base)


# ============================================================================ 1. partición K1-K4
COLUMNAS = ["image_id", "image_name", "original_class", "label_v2", "class_v2", "e1_split", "image_path",
            "group_id", "group_size", "fold"]
CLASES_NEG = [0, 1, 2, 3, 4, 6]
IMAGENES_DIR = "data/processed/MMOTU_OTU_2D_experimental"


def filas_sinteticas():
    """820 filas: 782 no malignas (categorías 0,1,2,3,4,6) y 38 malignas (7); 5 folds con 8/8/8/7/7 positivas."""
    filas = []
    pos_por_fold = [8, 8, 8, 7, 7]
    n = 0
    for k, npos in enumerate(pos_por_fold):
        for j in range(npos):
            filas.append(dict(original_class=7, label_v2=1, fold=k, grupo=f"gp{k}_{j // 2}"))   # grupos de hasta 2
    n_neg_por_fold = [157, 157, 156, 156, 156]
    for k, nneg in enumerate(n_neg_por_fold):
        for j in range(nneg):
            c = CLASES_NEG[(n + j) % 6]
            filas.append(dict(original_class=c, label_v2=0, fold=k, grupo=f"gn{k}_{j // 3}"))
    out = []
    for i, f in enumerate(filas):
        split = "train" if i % 5 else "validation"
        name = f"{i + 1}.JPG"
        out.append({"image_id": str(i + 1), "image_name": name, "original_class": str(f["original_class"]),
                    "label_v2": str(f["label_v2"]),
                    "class_v2": "lesion_maligna" if f["label_v2"] else "lesion_no_maligna",
                    "e1_split": split, "image_path": f"{IMAGENES_DIR}/{split}/images/{name}",
                    "group_id": f["grupo"], "group_size": "1", "fold": str(f["fold"])})
    return out


def escribir_particion(root, config, filas=None, columnas=COLUMNAS, eol="\n", mutar_bytes=None, registrar_hash=True):
    """Escribe el CSV sintético en la ruta de la configuración y fija en la copia de `config` el hash de sus bytes."""
    filas = filas_sinteticas() if filas is None else filas
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columnas, lineterminator=eol, extrasaction="ignore")
    w.writeheader()
    w.writerows(filas)
    datos = buf.getvalue().encode("utf-8")
    ruta = Path(root) / config["particion"]["csv"]["ruta"]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if registrar_hash:
        config["particion"]["csv"]["sha256_bytes_archivo"] = hashlib.sha256(datos).hexdigest()
    ruta.write_bytes(mutar_bytes(datos) if mutar_bytes else datos)
    return ruta


def cargar(tmp_path, config, **kw):
    escribir_particion(tmp_path, config, **kw)
    return P.cargar_particion(tmp_path, config)


def test_particion_valida_y_observaciones_no_son_criterio(tmp_path, config):
    part = cargar(tmp_path, config)
    assert len(part.filas) == 820
    assert part.observaciones["positivas_por_fold"] == [8, 8, 8, 7, 7]
    # las observaciones (164, 69, 51...) NO son criterios: un reparto distinto con mínimo >= 4 también se acepta
    filas = filas_sinteticas()
    filas[0]["fold"] = "4"  # un positivo del grupo gp0_0 cambia de fold: ahora el grupo cruza folds -> K3 (no por tamaños)
    with pytest.raises(C.GuardError, match="K3.*más de un fold"):
        cargar(tmp_path / "x", copy.deepcopy(config), filas=filas)


def test_k1_hash_de_bytes(tmp_path, config):
    escribir_particion(tmp_path, config)
    ruta = tmp_path / config["particion"]["csv"]["ruta"]
    ruta.write_bytes(ruta.read_bytes() + b"\n")     # un byte más: el hash registrado ya no coincide
    with pytest.raises(C.GuardError, match="K1"):
        P.cargar_particion(tmp_path, config)


def test_k1_no_normaliza_finales_de_linea(tmp_path, config):
    escribir_particion(tmp_path, config, eol="\r\n")                    # hash registrado de los bytes CRLF
    assert len(P.cargar_particion(tmp_path, config).filas) == 820
    ruta = tmp_path / config["particion"]["csv"]["ruta"]
    ruta.write_bytes(ruta.read_bytes().replace(b"\r\n", b"\n"))         # misma información, otros bytes
    with pytest.raises(C.GuardError, match="K1"):
        P.cargar_particion(tmp_path, config)


def test_particion_ausente_es_no_disponible(tmp_path, config):
    with pytest.raises(C.GuardError, match="no está disponible"):
        P.cargar_particion(tmp_path, config)


@pytest.mark.parametrize("mutacion, patron", [
    (lambda f: f[0].update(original_class="5"), "K2.*categoría 5"),
    (lambda f: f[0].update(original_class="8"), "K2.*desconocida"),
    (lambda f: f[0].update(original_class="x"), "K2/K3.*enteros"),
    (lambda f: f.pop(), "K2.*819 filas"),
    (lambda f: f[1].update(image_id=f[0]["image_id"]), "K2.*image_id repetidos"),
    (lambda f: f[1].update(image_path=f[0]["image_path"], image_name=f[0]["image_name"],
                           e1_split=f[0]["e1_split"]), "K2.*image_path repetidos"),
])
def test_k2_mapeo_y_unicidad(tmp_path, config, mutacion, patron):
    filas = filas_sinteticas()
    mutacion(filas)
    with pytest.raises(C.GuardError, match=patron):
        cargar(tmp_path, config, filas=filas)


def test_k2_label_v2_se_recalcula_desde_original_class(tmp_path, config):
    filas = filas_sinteticas()
    neg6 = next(r for r in filas if r["original_class"] == "6")
    assert neg6["label_v2"] == "0"                           # la categoría 6 es lesión no maligna (mapeo v2)
    assert P.mapa_etiquetas(config) == {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 6: 0, 7: 1}
    neg6["label_v2"] = "1"                                   # etiqueta del mapeo histórico (0-5 frente a 6-7): se rechaza
    with pytest.raises(C.GuardError, match="K2.*label_v2"):
        cargar(tmp_path, config, filas=filas)


def test_k2_recuentos_782_38(tmp_path, config):
    filas = filas_sinteticas()
    neg = next(r for r in filas if r["label_v2"] == "0")
    neg.update(original_class="7", label_v2="1")             # 781 / 39
    with pytest.raises(C.GuardError, match="K2.*781"):
        cargar(tmp_path, config, filas=filas)


def test_k2_columnas_requeridas(tmp_path, config):
    with pytest.raises(C.GuardError, match="K2.*faltan columnas"):
        cargar(tmp_path, config, columnas=[c for c in COLUMNAS if c != "group_id"])


@pytest.mark.parametrize("extra", ["binary_label", "binary_class"])
def test_k4_rechaza_columnas_del_mapeo_historico(tmp_path, config, extra):
    filas = filas_sinteticas()
    for r in filas:
        r[extra] = "0"
    with pytest.raises(C.GuardError, match="K4.*histórico"):
        cargar(tmp_path, config, filas=filas, columnas=COLUMNAS + [extra])


def test_k3_grupo_en_dos_folds(tmp_path, config):
    filas = filas_sinteticas()
    a = next(r for r in filas if r["group_id"] == "gn0_0")
    b = [r for r in filas if r["group_id"] == "gn0_0"][-1]
    b["fold"] = "3"
    assert a is not b
    with pytest.raises(C.GuardError, match="K3.*más de un fold"):
        cargar(tmp_path, config, filas=filas)


def test_k3_grupo_mixto(tmp_path, config):
    filas = filas_sinteticas()
    pos = next(r for r in filas if r["label_v2"] == "1")
    neg = next(r for r in filas if r["label_v2"] == "0" and r["fold"] == pos["fold"])
    neg["group_id"] = pos["group_id"]
    with pytest.raises(C.GuardError, match="K3.*mixto"):
        cargar(tmp_path, config, filas=filas)


def test_k3_fold_fuera_de_rango(tmp_path, config):
    filas = filas_sinteticas()
    filas[0]["fold"] = "5"
    with pytest.raises(C.GuardError, match="K3.*fuera de"):
        cargar(tmp_path, config, filas=filas)


def test_k3_minimo_de_cuatro_positivas_bloquea(tmp_path, config):
    filas = filas_sinteticas()
    pos4 = [r for r in filas if r["label_v2"] == "1" and r["fold"] == "4"]
    for r in pos4[:4]:                                       # el fold 4 se queda con 3 positivas
        r.update(fold="3", group_id=f"mov_{r['image_id']}")
    with pytest.raises(C.GuardError, match="K3.*fold 4 tiene 3 positivas"):
        cargar(tmp_path, config, filas=filas)


def test_k3_train_y_validacion_disjuntos_y_cubren_el_total(tmp_path, config):
    part = cargar(tmp_path, config)
    for k in range(5):
        train, val = P.filas_de_fold(part, k)
        ids_t, ids_v = {r["image_id"] for r in train}, {r["image_id"] for r in val}
        assert not ids_t & ids_v and len(ids_t | ids_v) == 820
        assert not {r["group_id"] for r in train} & {r["group_id"] for r in val}   # ningún grupo cruza train/validación
    with pytest.raises(C.GuardError, match="K3"):
        P.filas_de_fold(part, 9)


@pytest.mark.parametrize("campo, valor", [
    ("e1_split", "test"), ("e1_split", "Test"),
    ("image_path", f"{IMAGENES_DIR}/test/images/1.JPG"),
    ("image_path", f"{IMAGENES_DIR}/TEST/images/1.JPG"),
    ("image_path", f"{IMAGENES_DIR}/train/images/../../test/images/1.JPG"),
    ("image_name", "../x.JPG"), ("image_name", "a/b.JPG"), ("image_name", "C:x.JPG"),
])
def test_k4_rutas_de_test_o_fuera_de_desarrollo(tmp_path, config, campo, valor):
    filas = filas_sinteticas()
    filas[0][campo] = valor
    with pytest.raises(C.GuardError, match="K4"):
        cargar(tmp_path, config, filas=filas)


@pytest.mark.parametrize("ruta", [
    "configs/splits/mmotu_experimental_split.csv", "configs/splits/mmotu_experimental_split_v2.csv",
    "configs/splits/cv_folds_e2.csv", "configs/splits/test/cv_folds.csv", "configs/Test/x.csv",
])
def test_k4_rutas_prohibidas_del_csv(tmp_path, config, ruta):
    config["particion"]["csv"]["ruta"] = ruta
    with pytest.raises(C.GuardError, match="K4"):
        P.cargar_particion(tmp_path, config)


def test_k4_resolver_imagen_dentro_de_desarrollo(tmp_path):
    fila = {"image_path": f"{IMAGENES_DIR}/train/images/1.JPG", "image_name": "1.JPG"}
    d = tmp_path / IMAGENES_DIR / "train" / "images"
    d.mkdir(parents=True)
    (d / "1.JPG").write_bytes(b"x")
    assert P.resolver_imagen(tmp_path, fila).name == "1.JPG"                 # no abre el archivo, solo resuelve
    with pytest.raises(C.GuardError, match="no encontrada"):
        P.resolver_imagen(tmp_path, {**fila, "image_name": "2.JPG"})
    with pytest.raises(C.GuardError, match="K4"):
        P.resolver_imagen(tmp_path, {**fila, "image_name": "../../test/images/1.JPG"})


def test_pesos_de_clase_total_sobre_2n(tmp_path, config):
    part = cargar(tmp_path, config)
    train, _ = P.filas_de_fold(part, 0)
    w = P.pesos_de_clase([r["label_v2"] for r in train])
    n1 = sum(1 for r in train if r["label_v2"] == 1)
    assert w[1] == pytest.approx(len(train) / (2 * n1)) and w[0] == pytest.approx(len(train) / (2 * (len(train) - n1)))
    assert w[1] / w[0] > 15                                                  # razón del orden de 21:1 del protocolo §6.1
    with pytest.raises(C.GuardError):
        P.pesos_de_clase([0, 0, 0])
    with pytest.raises(C.GuardError):
        P.pesos_de_clase([0, 1], alpha=0.5)


def test_config_real_es_coherente_con_el_protocolo(config_base):
    assert config_base["particion"]["verificaciones_bloqueantes"]["n_filas"] == 820
    assert config_base["calibracion_umbral"]["sensibilidad_objetivo_S_star"] is None      # S* NO congelado
    assert config_base["calibracion_umbral"]["techo_corridas_g4"] is None                  # techo de 75 NO congelado
    assert config_base["integridad"]["referencia_pesos"]["hash_esperado"] is None          # sin hash inventado


# ============================================================================ 2. integridad §4.5 (repositorio Git temporal)
import shutil
import types

from src.r4_v2 import integridad as I

PROTO = C.PROTOCOLO_REL
LISTA_REAL = I.LISTA_FIJA_CERRADA                                               # la lista cerrada REAL del código
LISTA_PEQUENA = (PROTO, "configs/r4_v2/", "src/r4_v2/", ".gitattributes")       # la de los repositorios sintéticos


@pytest.fixture(autouse=True)
def _lista_fija_de_los_repositorios_sinteticos(monkeypatch):
    monkeypatch.setattr(I, "LISTA_FIJA_CERRADA", LISTA_PEQUENA)


def run_git(cwd, *args, check=True):
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, check=check)


def commit_all(repo, msg="c"):
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", msg)


def escribir(repo, rel, contenido):
    p = Path(repo) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    data = contenido if isinstance(contenido, bytes) else contenido.encode("utf-8")
    p.write_bytes(data)
    return p


def crear_repo(tmp_path, nombre="repo", autocrlf="false", archivos=None):
    repo = tmp_path / nombre
    repo.mkdir()
    run_git(repo, "init", "-q")
    for k, v in (("user.name", "t"), ("user.email", "t@example.org"), ("core.autocrlf", autocrlf),
                 ("commit.gpgsign", "false")):
        run_git(repo, "config", k, v)
    base = {
        ".gitattributes": "configs/splits/v2_auto/*.csv -text\nconfigs/splits/v2_auto/*.json -text\n",
        ".gitignore": "__pycache__/\n*.log\nignorada/\n",
        PROTO: "# protocolo sintético\nlinea 2\n",
        "configs/r4_v2/g2_baseline_v1.yaml": "esquema: x\n",
        "src/r4_v2/modulo_a.py": "VALOR = 1\n",
        "tests/auxiliar.py": "T = 1\n",
        "raiz_mod.py": "R = 1\n",
        "notebooks/orq.ipynb": '{"cells": []}\n',
    }
    base.update(archivos or {})
    for rel, cont in base.items():
        escribir(repo, rel, cont)
    commit_all(repo, "base")
    return repo


def config_integridad(repo, extra_fijas=()):
    """Configuración mínima para el repositorio sintético; los hashes del protocolo se leen del commit."""
    return {"integridad": {"rutas_verificadas_lista_fija": [PROTO, "configs/r4_v2/", "src/r4_v2/", ".gitattributes",
                                                            *extra_fijas]},
            "protocolo": {"git_blob_id": run_git(repo, "rev-parse", f"HEAD:{PROTO}").stdout.decode().strip(),
                          "sha256_contenido_blob": hashlib.sha256(
                              run_git(repo, "cat-file", "blob", f"HEAD:{PROTO}").stdout).hexdigest()}}


class _SourceFileLoader:
    pass


_SourceFileLoader.__name__ = "SourceFileLoader"


class _SourcelessFileLoader:
    pass


_SourcelessFileLoader.__name__ = "SourcelessFileLoader"


def modulo_falso(nombre, ruta, cached=None, loader=_SourceFileLoader):
    m = types.ModuleType(nombre)
    m.__file__ = str(ruta)
    m.__spec__ = types.SimpleNamespace(origin=str(ruta), loader=loader(), cached=cached)
    m.__cached__ = cached
    return m


def entorno_ok(repo, tmp_path, modulos=(), **kw):
    """Entorno que cumple los controles de bytecode: -B, prefijo vacío fuera del repositorio, entrada = script del repo."""
    prefijo = tmp_path / "prefijo_pyc"
    prefijo.mkdir(exist_ok=True)
    principal = types.ModuleType("__main__")
    principal.__file__ = str(Path(repo) / "raiz_mod.py")
    principal.__spec__ = None
    mods = {m.__name__: m for m in modulos}
    datos = dict(modules=mods, version_info=(3, 10, 12), version="3.10.12", flags_dont_write_bytecode=1,
                 dont_write_bytecode=True, pycache_prefix=str(prefijo), main=principal, sys_path=[])
    datos.update(kw)
    return I.EntornoPython(**datos)


def codigos(res):
    return sorted({b["codigo"] for b in res["bloqueos"]})


def verificar(repo, tmp_path, modo="completa", modulos=(), cfg=None, **kw):
    return I.verificar_integridad(repo, cfg or config_integridad(repo), modo=modo,
                                  entorno=kw.pop("entorno", None) or entorno_ok(repo, tmp_path, modulos), **kw)


def test_integridad_repositorio_limpio_pasa_y_registra_campos(tmp_path):
    repo = crear_repo(tmp_path)
    res = verificar(repo, tmp_path)
    assert res["ok"], res["bloqueos"]
    reg = res["archivos"][PROTO]
    assert set(reg) >= {"git_blob_id", "sha256_contenido_blob", "sha256_bytes_archivo", "eol_trabajo_difiere",
                        "hash_object_coincide", "git_ultimo_commit", "atributo_text"}
    assert reg["eol_trabajo_difiere"] is False and reg["hash_object_coincide"] is True
    assert res["protocolo_versionado"] is True
    assert res["git_commit_head"] == run_git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    assert I.exigir_integridad(res) is res


def test_vocabulario_de_hashes_campos_no_intercambiables(tmp_path):
    repo = crear_repo(tmp_path)
    reg = verificar(repo, tmp_path)["archivos"][PROTO]
    assert len(reg["git_blob_id"]) == 40 and len(reg["sha256_contenido_blob"]) == 64
    assert reg["git_blob_id"] == run_git(repo, "rev-parse", f"HEAD:{PROTO}").stdout.decode().strip()
    assert reg["sha256_contenido_blob"] == hashlib.sha256(
        run_git(repo, "cat-file", "blob", f"HEAD:{PROTO}").stdout).hexdigest()
    assert reg["sha256_bytes_archivo"] == C.sha256_archivo(repo / PROTO)


def test_archivo_modificado_bloquea(tmp_path):
    repo = crear_repo(tmp_path)
    escribir(repo, "src/r4_v2/modulo_a.py", "VALOR = 2\n")
    assert "B_contenido_difiere" in codigos(verificar(repo, tmp_path))
    with pytest.raises(C.GuardError, match="K6"):
        I.exigir_integridad(verificar(repo, tmp_path))


def test_cambio_preparado_bloquea_aunque_el_arbol_vuelva_a_head(tmp_path):
    repo = crear_repo(tmp_path)
    escribir(repo, "src/r4_v2/modulo_a.py", "VALOR = 2\n")
    run_git(repo, "add", "src/r4_v2/modulo_a.py")
    escribir(repo, "src/r4_v2/modulo_a.py", "VALOR = 1\n")                    # el árbol vuelve al contenido de HEAD
    assert codigos(verificar(repo, tmp_path)) == ["preparado"]               # [B] pasa; el índice sigue distinto de HEAD


def test_archivo_nuevo_sin_seguimiento_bloquea(tmp_path):
    repo = crear_repo(tmp_path)
    escribir(repo, "src/r4_v2/suelto.py", "X = 1\n")
    assert "sin_seguimiento" in codigos(verificar(repo, tmp_path))


def test_archivo_ignorado_bloquea_pero_pycache_no(tmp_path):
    repo = crear_repo(tmp_path)
    escribir(repo, "src/r4_v2/__pycache__/modulo_a.cpython-310.pyc", b"x")
    assert verificar(repo, tmp_path)["ok"]
    escribir(repo, "src/r4_v2/salida.log", "x")
    assert codigos(verificar(repo, tmp_path)) == ["ignorado"]


def test_modulo_del_repo_fuera_de_la_lista_en_tests_y_raiz(tmp_path):
    repo = crear_repo(tmp_path)
    en_tests = modulo_falso("tests.auxiliar", repo / "tests" / "auxiliar.py")
    en_raiz = modulo_falso("raiz_mod", repo / "raiz_mod.py")
    res = verificar(repo, tmp_path, modulos=[en_tests, en_raiz])
    assert res["ok"] and {"tests/auxiliar.py", "raiz_mod.py"} <= set(res["archivos"])        # entran por la lista dinámica
    assert {"tests.auxiliar", "raiz_mod"} <= set(res["modulos_en_repo"])
    escribir(repo, "tests/auxiliar.py", "T = 2\n")                                            # modificado: bloquea
    assert "B_contenido_difiere" in codigos(verificar(repo, tmp_path, modulos=[en_tests]))
    suelto = escribir(repo, "tests/suelto.py", "S = 1\n")                                     # nunca versionado: bloquea
    res = verificar(repo, tmp_path, modulos=[modulo_falso("tests.suelto", suelto)])
    assert {"no_rastreado", "sin_seguimiento"} <= set(codigos(res))


def test_modulos_del_entorno_no_se_verifican_por_hash(tmp_path):
    repo = crear_repo(tmp_path)
    ajeno = tmp_path / "otro" / "libreria.py"
    ajeno.parent.mkdir()
    ajeno.write_text("A = 1\n")
    res = verificar(repo, tmp_path, modulos=[modulo_falso("libreria", ajeno)])
    assert res["ok"] and "libreria" not in res["modulos_en_repo"]


def test_paquete_de_espacio_de_nombres_sin_init(tmp_path):
    repo = crear_repo(tmp_path)
    ns = types.ModuleType("src")
    ns.__path__ = [str(repo / "src")]
    ns.__file__ = None
    ns.__spec__ = types.SimpleNamespace(origin=None, loader=None, cached=None)
    hijo = modulo_falso("src.r4_v2.modulo_a", repo / "src" / "r4_v2" / "modulo_a.py")
    res = verificar(repo, tmp_path, modulos=[ns, hijo])
    assert res["ok"], res["bloqueos"]
    assert res["paquetes_espacio_de_nombres"] == {"src": ["src"]}                           # se registra por __path__ relativo
    assert "src/r4_v2/modulo_a.py" in res["archivos"]                                        # el submódulo se verifica por archivo
    ns.__path__ = [str(repo / "src"), str(tmp_path / "otro_src")]
    assert "namespace_fuera_del_repo" in codigos(verificar(repo, tmp_path, modulos=[ns]))


def test_modulo_solo_pyc_bloquea(tmp_path):
    repo = crear_repo(tmp_path)
    pyc = escribir(repo, "src/r4_v2/solo.pyc", b"x")
    m = modulo_falso("src.r4_v2.solo", pyc, loader=_SourcelessFileLoader)
    assert "modulo_sin_fuente_rastreable" in codigos(verificar(repo, tmp_path, modulos=[m]))


def test_notebook_orquestador_rastreado_y_modificado(tmp_path):
    repo = crear_repo(tmp_path)
    res = verificar(repo, tmp_path, notebook="notebooks/orq.ipynb")
    assert res["ok"] and res["notebook"]["git_blob_id"] and res["notebook"]["git_ultimo_commit"]
    escribir(repo, "notebooks/orq.ipynb", '{"cells": [1]}\n')
    assert "B_contenido_difiere" in codigos(verificar(repo, tmp_path, notebook="notebooks/orq.ipynb"))
    escribir(repo, "notebooks/nuevo.ipynb", "{}\n")
    assert "no_rastreado" in codigos(verificar(repo, tmp_path, notebook="notebooks/nuevo.ipynb"))
    assert "notebook_fuera_del_repo" in codigos(verificar(repo, tmp_path, notebook=tmp_path / "x.ipynb"))


def test_cambio_entre_verificacion_inicio_y_fin(tmp_path):
    repo = crear_repo(tmp_path)
    ini = verificar(repo, tmp_path)
    assert I.comparar_instantaneas(ini, verificar(repo, tmp_path)) == []
    escribir(repo, "src/r4_v2/modulo_a.py", "VALOR = 3\n")
    fin = verificar(repo, tmp_path)
    assert I.comparar_instantaneas(ini, fin) == ["archivo:src/r4_v2/modulo_a.py"]


def test_protocolo_hash_distinto_y_no_versionado(tmp_path):
    repo = crear_repo(tmp_path)
    cfg = config_integridad(repo)
    cfg["protocolo"]["sha256_contenido_blob"] = "0" * 64
    assert "protocolo_hash_distinto" in codigos(verificar(repo, tmp_path, cfg=cfg))
    cfg = config_integridad(repo)
    cfg["protocolo"]["git_blob_id"] = "0" * 40
    assert "protocolo_blob_id_distinto" in codigos(verificar(repo, tmp_path, cfg=cfg))
    # protocolo no versionado: humo y completa bloquean; referencia solo con la excepción documentada
    repo2 = crear_repo(tmp_path, "repo2")
    cfg2 = config_integridad(repo2)
    run_git(repo2, "rm", "-q", "--cached", PROTO)
    run_git(repo2, "commit", "-q", "-m", "sin protocolo")
    for modo in ("humo", "completa", "referencia"):
        assert "protocolo_no_versionado" in codigos(verificar(repo2, tmp_path, modo=modo, cfg=cfg2))
    exc = {"motivo": "protocolo aún sin commit", "sha256_bytes_archivo_borrador": "a" * 64}
    res = verificar(repo2, tmp_path, modo="referencia", cfg=cfg2, excepcion_protocolo=exc)
    assert res["protocolo_versionado"] is False and res["excepcion_protocolo"] == exc
    assert "protocolo_no_versionado" not in codigos(res) and "sin_seguimiento" not in codigos(res)
    res = verificar(repo2, tmp_path, modo="humo", cfg=cfg2, excepcion_protocolo=exc)           # la excepción no vale en humo
    assert "protocolo_no_versionado" in codigos(res)


# ---- CRLF/LF (M1)
def test_crlf_checkout_real_con_autocrlf_true_pasa(tmp_path):
    origen = crear_repo(tmp_path, "origen")                                                   # blobs en LF
    clon = tmp_path / "clon"
    subprocess.run(["git", "clone", "-q", "-c", "core.autocrlf=true", str(origen), str(clon)], check=True,
                   capture_output=True)
    assert b"\r\n" in (clon / PROTO).read_bytes()                                             # archivo de trabajo en CRLF
    res = verificar(clon, tmp_path, cfg=config_integridad(clon))
    assert res["ok"], res["bloqueos"]
    assert res["archivos"][PROTO]["eol_trabajo_difiere"] is True and res["archivos"][PROTO]["hash_object_coincide"] is True


def test_crlf_mismos_bytes_sobre_checkout_lf_pasa_y_se_registra(tmp_path):
    repo = crear_repo(tmp_path)
    p = repo / "src" / "r4_v2" / "modulo_a.py"
    p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))                                     # mismo contenido, CRLF
    res = verificar(repo, tmp_path)
    assert res["ok"], res["bloqueos"]
    assert res["archivos"]["src/r4_v2/modulo_a.py"]["eol_trabajo_difiere"] is True
    assert "src/r4_v2/modulo_a.py" in res["status_m_sin_cambio_de_contenido"]


def test_crlf_con_cambio_real_de_contenido_bloquea(tmp_path):
    repo = crear_repo(tmp_path)
    escribir(repo, "src/r4_v2/modulo_a.py", b"VALOR = 9\r\n")
    assert "B_contenido_difiere" in codigos(verificar(repo, tmp_path))


@pytest.mark.parametrize("ocultar", ["skip-worktree", "assume-unchanged"])
def test_cambio_real_oculto_a_git_status_bloquea(tmp_path, ocultar):
    repo = crear_repo(tmp_path)
    run_git(repo, "update-index", f"--{ocultar}", "src/r4_v2/modulo_a.py")
    escribir(repo, "src/r4_v2/modulo_a.py", "VALOR = 7\n")
    assert b"modulo_a" not in run_git(repo, "status", "--porcelain").stdout                  # git status no lo ve
    assert "B_contenido_difiere" in codigos(verificar(repo, tmp_path))


def _filtro_clean(repo, nombre, salida):
    """Filtro `clean` local (config del repositorio + .git/info/attributes) que siempre emite `salida`."""
    script = repo.parent / f"{nombre}.py"
    script.write_text(f"import sys\nsys.stdin.buffer.read()\nsys.stdout.buffer.write({salida!r}.encode())\n")
    run_git(repo, "config", f"filter.{nombre}.clean", f'"{sys.executable}" "{script.as_posix()}"')
    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "attributes").write_text(f"src/r4_v2/modulo_a.py filter={nombre}\n")


def test_filtro_clean_local_oculta_el_cambio_a_git_pero_lo_detecta_b(tmp_path):
    repo = crear_repo(tmp_path)
    _filtro_clean(repo, "ocultar", "VALOR = 1\n")
    escribir(repo, "src/r4_v2/modulo_a.py", "VALOR = 7\n")
    assert run_git(repo, "hash-object", "src/r4_v2/modulo_a.py").stdout.decode().strip() == \
        run_git(repo, "rev-parse", "HEAD:src/r4_v2/modulo_a.py").stdout.decode().strip()      # [A] es engañado por el filtro
    res = verificar(repo, tmp_path)
    assert "B_contenido_difiere" in codigos(res) and "A_hash_object_distinto" not in codigos(res)


def test_bytes_identicos_al_blob_con_hash_object_distinto_bloquea_por_a(tmp_path):
    repo = crear_repo(tmp_path)
    _filtro_clean(repo, "raro", "otra cosa\n")
    res = verificar(repo, tmp_path)                                                           # bytes de trabajo == blob
    reg = res["archivos"]["src/r4_v2/modulo_a.py"]
    assert reg["eol_trabajo_difiere"] is False and reg["hash_object_coincide"] is False
    assert "A_hash_object_distinto" in codigos(res)


def test_check_attr_unset_en_particion_y_unspecified_en_src(tmp_path):
    repo = crear_repo(tmp_path, archivos={"configs/splits/v2_auto/cv.csv": "a,b\n"})
    atr = lambda rel: run_git(repo, "check-attr", "text", "--", rel).stdout.decode().strip().rsplit(": ", 1)[1]
    assert atr("configs/splits/v2_auto/cv.csv") == "unset" and atr("src/r4_v2/modulo_a.py") == "unspecified"
    extra = ["configs/splits/v2_auto/cv.csv"]
    res = verificar(repo, tmp_path, cfg=config_integridad(repo, extra_fijas=extra), lista_fija=(*LISTA_PEQUENA, *extra))
    assert res["archivos"]["configs/splits/v2_auto/cv.csv"]["atributo_text"] == "unset"
    assert res["archivos"]["src/r4_v2/modulo_a.py"]["atributo_text"] == "unspecified"      # unspecified NO es -text
    # el archivo -text se conserva byte a byte aunque core.autocrlf=true (clon real)
    clon = tmp_path / "clon_text"
    subprocess.run(["git", "clone", "-q", "-c", "core.autocrlf=true", str(repo), str(clon)], check=True, capture_output=True)
    assert (clon / "configs/splits/v2_auto/cv.csv").read_bytes() == b"a,b\n"
    assert b"\r\n" in (clon / "src/r4_v2/modulo_a.py").read_bytes()


# ---- bytecode (M4), controles ejecutables en proceso
def test_bytecode_controles_efectivos_en_proceso(tmp_path):
    repo = crear_repo(tmp_path)
    assert verificar(repo, tmp_path)["ok"]

    def con(**kw):
        return codigos(verificar(repo, tmp_path, entorno=entorno_ok(repo, tmp_path, **kw)))

    assert "sin_pycache_prefix" in con(pycache_prefix=None)
    dentro = repo / "pyc"
    dentro.mkdir()
    assert "pycache_prefix_en_repo" in con(pycache_prefix=str(dentro))
    lleno = tmp_path / "lleno"
    lleno.mkdir()
    (lleno / "x.pyc").write_bytes(b"x")
    assert "pycache_prefix_no_vacio" in con(pycache_prefix=str(lleno))
    assert "pycache_prefix_inexistente" in con(pycache_prefix=str(tmp_path / "no_existe"))
    assert "bytecode_se_escribe" in con(flags_dont_write_bytecode=0)
    assert "bytecode_se_escribe" in con(dont_write_bytecode=False)
    assert "python_antiguo" in con(version_info=(3, 7, 9))
    assert "python_antiguo" not in con(version_info=(3, 8, 0))


def test_bytecode_cached_en_el_arbol_bloquea_y_bajo_el_prefijo_pasa(tmp_path):
    repo = crear_repo(tmp_path)
    en_arbol = modulo_falso("raiz_mod", repo / "raiz_mod.py", cached=str(repo / "__pycache__" / "raiz_mod.cpython-310.pyc"))
    assert "bytecode_en_arbol" in codigos(verificar(repo, tmp_path, modulos=[en_arbol]))
    bajo = tmp_path / "prefijo_pyc" / "x" / "raiz_mod.cpython-310.pyc"
    assert verificar(repo, tmp_path, modulos=[modulo_falso("raiz_mod", repo / "raiz_mod.py", cached=str(bajo))])["ok"]


@pytest.mark.parametrize("principal", [None, types.ModuleType("__main__")], ids=["sin_main", "sin_archivo"])
def test_entrada_sin_archivo_se_rechaza(tmp_path, principal):
    """python -c, REPL, stdin (sin __main__ con archivo) y kernel de notebook (__main__ sin __file__)."""
    repo = crear_repo(tmp_path)
    res = verificar(repo, tmp_path, entorno=entorno_ok(repo, tmp_path, main=principal))
    assert "entrada_sin_archivo" in codigos(res)


def test_entrada_fuera_del_repo_se_rechaza(tmp_path):
    repo = crear_repo(tmp_path)
    ajeno = types.ModuleType("__main__")
    ajeno.__file__ = str(tmp_path / "suelto.py")
    ajeno.__spec__ = None
    assert "entrada_fuera_del_repo" in codigos(verificar(repo, tmp_path, entorno=entorno_ok(repo, tmp_path, main=ajeno)))


def test_lista_fija_cerrada_del_codigo_coincide_con_el_protocolo_y_con_la_configuracion(config_base):
    esperada = {PROTO, "configs/r4_v2/", "src/r4_v2/", ".gitattributes",
                *("src/classification/" + n + ".py" for n in ("__init__", "models", "transforms", "dataloader", "evaluate",
                                                                "losses", "utils", "train", "training_options"))}
    assert set(LISTA_REAL) == esperada and len(LISTA_REAL) == len(esperada) == 13
    assert set(config_base["integridad"]["rutas_verificadas_lista_fija"]) == set(LISTA_REAL)     # YAML real = código
    for prohibido in ("run_experiment", "final_evaluation", "cv_folds_e2", "dataset.py"):
        assert not any(prohibido in f for f in LISTA_REAL)


# ---- bytecode (M4): pruebas con SUBPROCESOS reales; deben repetirse en el Python de Falcon (3.10) [ND]
import py_compile

SRC_R4 = REPO / "src" / "r4_v2"

ENTRADA = '''\
import json, os, sys
herr, raiz, cfg = sys.argv[1:4]
if os.environ.get("R4_PREFIJO_EN_CODIGO"):
    sys.pycache_prefix = os.environ["R4_PREFIJO_EN_CODIGO"]          # antes de importar cualquier módulo del repositorio
sys.path.insert(0, herr)
sys.path.insert(0, os.path.join(raiz, "app"))
from r4tools import integridad as I
modulos = [m for m in os.environ.get("R4_IMPORTAR", "modx").split(",") if m]
valores = {m: getattr(__import__(m), "VALUE", None) for m in modulos}
cfg_json = json.load(open(cfg))
res = I.verificar_integridad(raiz, cfg_json, modo="completa", lista_fija=cfg_json["integridad"]["rutas_verificadas_lista_fija"])
print(json.dumps({"valores": valores, "codigos": sorted({b["codigo"] for b in res["bloqueos"]})}))
'''


@pytest.fixture
def banco_bytecode(tmp_path):
    """Repositorio temporal con `app/entrada.py` y `app/modx.py` (con un .pyc obsoleto), y las herramientas copiadas fuera."""
    herr = tmp_path / "herramienta" / "r4tools"
    herr.mkdir(parents=True)
    (herr / "__init__.py").write_text("")
    for nombre in ("common.py", "integridad.py"):
        shutil.copy(SRC_R4 / nombre, herr / nombre)
    repo = crear_repo(tmp_path, archivos={"app/entrada.py": ENTRADA, "app/modx.py": "VALUE = 1\n"})
    modx = repo / "app" / "modx.py"
    py_compile.compile(str(modx), doraise=True)                      # .pyc con el código VALUE = 1
    st = modx.stat()
    modx.write_bytes(b"VALUE = 2\n")                                 # mismo tamaño: el código cambia, mtime y tamaño no
    os.utime(modx, ns=(st.st_atime_ns, st.st_mtime_ns))
    # Git confía en (mtime, tamaño) y no vería el cambio: se registra el contenido nuevo directamente en el índice
    blob = run_git(repo, "hash-object", "-w", "app/modx.py").stdout.decode().strip()
    run_git(repo, "update-index", "--cacheinfo", f"100644,{blob},app/modx.py")
    run_git(repo, "commit", "-q", "-m", "modx 2")
    assert run_git(repo, "cat-file", "blob", "HEAD:app/modx.py").stdout == b"VALUE = 2\n"
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(config_integridad(repo)))
    return types.SimpleNamespace(repo=repo, herr=tmp_path / "herramienta", cfg=cfg, tmp=tmp_path)


def lanzar(banco, flags, extra_env=None, args=None, stdin=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update(extra_env or {})
    cmd = [sys.executable, *flags]
    cmd += args if args is not None else [str(banco.repo / "app" / "entrada.py"), str(banco.herr), str(banco.repo),
                                          str(banco.cfg)]
    r = subprocess.run(cmd, capture_output=True, env=env, input=stdin, cwd=str(banco.tmp))
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    return json.loads(r.stdout.decode().strip().splitlines()[-1])


def prefijo_vacio(banco, nombre="pfx"):
    p = banco.tmp / nombre
    p.mkdir()
    return p


def test_bytecode_pyc_obsoleto_con_I_y_variable_de_entorno_bloquea(banco_bytecode):
    b = banco_bytecode
    pfx = prefijo_vacio(b)
    salida = lanzar(b, ["-I"], {"PYTHONPYCACHEPREFIX": str(pfx)})        # -I ignora la variable: prefijo None
    assert salida["valores"]["modx"] == 1                                # se cargó el .pyc obsoleto del árbol
    assert {"sin_pycache_prefix", "bytecode_en_arbol", "bytecode_se_escribe"} <= set(salida["codigos"])


def test_bytecode_con_B_y_X_pycache_prefix_pasa_y_carga_el_codigo_actual(banco_bytecode):
    b = banco_bytecode
    pfx = prefijo_vacio(b)
    salida = lanzar(b, ["-B", "-X", f"pycache_prefix={pfx}"])
    assert salida["valores"]["modx"] == 2 and salida["codigos"] == []
    assert not any(pfx.iterdir())                                        # -B: el prefijo sigue vacío
    pfx2 = prefijo_vacio(b, "pfx2")
    assert lanzar(b, ["-I", "-B", "-X", f"pycache_prefix={pfx2}"])["codigos"] == []      # también con -I


def test_bytecode_prefijo_asignado_en_codigo_antes_de_importar_pasa(banco_bytecode):
    b = banco_bytecode
    pfx = prefijo_vacio(b)
    salida = lanzar(b, ["-B"], {"R4_PREFIJO_EN_CODIGO": str(pfx)})
    assert salida["valores"]["modx"] == 2 and salida["codigos"] == []


def test_bytecode_sin_B_con_prefijo_bloquea_por_escritura(banco_bytecode):
    b = banco_bytecode
    pfx = prefijo_vacio(b)
    salida = lanzar(b, ["-X", f"pycache_prefix={pfx}"])
    assert "bytecode_se_escribe" in salida["codigos"] and "pycache_prefix_no_vacio" in salida["codigos"]


def test_bytecode_modulo_solo_pyc_bloquea(banco_bytecode):
    b = banco_bytecode
    fuente = b.repo / "app" / "solo.py"
    fuente.write_bytes(b"VALUE = 5\n")
    py_compile.compile(str(fuente), cfile=str(b.repo / "app" / "solo.pyc"), doraise=True)
    fuente.unlink()
    pfx = prefijo_vacio(b)
    salida = lanzar(b, ["-B", "-X", f"pycache_prefix={pfx}"], {"R4_IMPORTAR": "solo"})
    assert salida["valores"]["solo"] == 5 and "modulo_sin_fuente_rastreable" in salida["codigos"]


CODIGO_C = ("import sys, json; sys.path.insert(0, sys.argv[1]); from r4tools import integridad as I; "
            "c = json.load(open(sys.argv[3])); res = I.verificar_integridad(sys.argv[2], c, modo='completa', "
            "lista_fija=c['integridad']['rutas_verificadas_lista_fija']); "
            "print(json.dumps({'valores': {}, 'codigos': sorted({b['codigo'] for b in res['bloqueos']})}))")


def test_bytecode_python_c_y_stdin_se_rechazan(banco_bytecode):
    b = banco_bytecode
    pfx = prefijo_vacio(b)
    comunes = ["-B", "-X", f"pycache_prefix={pfx}"]
    pos = [str(b.herr), str(b.repo), str(b.cfg)]
    assert "entrada_sin_archivo" in lanzar(b, comunes, args=["-c", CODIGO_C, *pos])["codigos"]
    script = CODIGO_C.replace("sys.argv", "ARGV")
    stdin = f"import sys; ARGV = {['-'] + pos!r}\n{script}\n".encode()
    assert "entrada_sin_archivo" in lanzar(b, comunes, args=["-"], stdin=stdin)["codigos"]


# ============================================================================ 3. métricas, bootstrap y calibración (numpy; sin torch)
np = pytest.importorskip("numpy")
sk_metrics = pytest.importorskip("sklearn.metrics")

from src.r4_v2 import bootstrap as BS
from src.r4_v2 import calibracion as CAL
from src.r4_v2 import metricas as M


def datos_con_empates(rng, n=300, prevalencia=0.1, decimales=2):
    y = (rng.random(n) < prevalencia).astype(int)
    y[:3] = 1
    y[3:6] = 0
    score = np.round(np.clip(rng.normal(0.3 + 0.35 * y, 0.25), 0, 1), decimales)          # muchos empates
    return y, score


@pytest.mark.parametrize("semilla", [0, 1, 2, 3])
def test_ap_y_roc_auc_coinciden_con_sklearn_y_con_la_implementacion_ingenua(semilla):
    y, s = datos_con_empates(np.random.default_rng(semilla))
    m = M.metricas(y, s)
    assert m["ap"] == pytest.approx(sk_metrics.average_precision_score(y, s), abs=1e-12)
    assert m["ap"] == pytest.approx(M.ap_ingenuo(y, s), abs=1e-12)
    assert m["roc_auc"] == pytest.approx(sk_metrics.roc_auc_score(y, s), abs=1e-12)
    tn, fp, fn, tp = sk_metrics.confusion_matrix(y, (s >= 0.5).astype(int)).ravel()
    assert (m["tp_05"], m["fp_05"], m["fn_05"], m["tn_05"]) == (tp, fp, fn, tn)
    assert m["sensibilidad_05"] == pytest.approx(tp / (tp + fn)) and m["especificidad_05"] == pytest.approx(tn / (tn + fp))


def test_ap_sin_empates_y_casos_extremos():
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    assert M.ap(y, s) == pytest.approx(sk_metrics.average_precision_score(y, s))
    assert M.ap(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert M.roc_auc(np.array([0, 0, 1, 1]), np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)      # todo empatado
    with pytest.raises(ValueError):
        M.preparar([0, 1], [0.1, float("nan")])
    with pytest.raises(ValueError):
        M.preparar([0, 2], [0.1, 0.2])


@pytest.mark.parametrize("semilla", [0, 1])
def test_pesos_enteros_equivalen_a_repetir_imagenes(semilla):
    rng = np.random.default_rng(semilla)
    y, s = datos_con_empates(rng, n=120)
    w = rng.integers(0, 4, size=y.size)
    w[:6] = 1                                                                       # ambas clases presentes
    pesos = M.calcular(M.preparar(y, s), w)
    yy, ss = np.repeat(y, w), np.repeat(s, w)
    ref = M.metricas(yy, ss)
    for k in ("ap", "roc_auc", "sensibilidad_05", "especificidad_05", "sensibilidad_esp_0.90", "sensibilidad_esp_0.95"):
        assert pesos[k] == pytest.approx(ref[k], abs=1e-12), k
    assert pesos["ap"] == pytest.approx(sk_metrics.average_precision_score(y, s, sample_weight=w), abs=1e-12)
    # matriz de pesos (B, n): cada fila coincide con su cálculo individual
    W2 = np.stack([w, np.ones_like(w)])
    m2 = M.calcular(M.preparar(y, s), W2)
    assert m2["ap"][1] == pytest.approx(M.ap(y, s)) and m2["ap"][0] == pytest.approx(pesos["ap"])


def test_sensibilidad_a_especificidad_fija_es_un_vertice_de_la_roc():
    # 10 negativas y 4 positivas sin empates: especificidad >= 0.90 admite 1 FP; >= 0.95 (1 - 0.05 * 10 = 0.5) admite 0 FP
    y = np.array([1] * 4 + [0] * 10)
    s = np.array([0.95, 0.85, 0.65, 0.35] + [0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05])
    m = M.metricas(y, s)
    assert m["sensibilidad_esp_0.90"] == pytest.approx(0.5)          # umbral 0.85: TP=2 (0.95, 0.85), FP=1 (0.9)
    assert m["sensibilidad_esp_0.95"] == pytest.approx(0.25)         # umbral 0.95: TP=1, FP=0
    fpr, tpr, _ = sk_metrics.roc_curve(y, s)
    assert m["sensibilidad_esp_0.90"] == pytest.approx(tpr[fpr <= 0.10 + 1e-12].max())
    assert m["sensibilidad_esp_0.95"] == pytest.approx(tpr[fpr <= 0.05 + 1e-12].max())


def test_media_de_metricas_distinta_de_metrica_de_la_media():
    y = np.array([1, 1, 0, 0, 0, 0])
    q1 = np.array([0.9, 0.2, 0.8, 0.1, 0.1, 0.1])
    q2 = np.array([0.2, 0.9, 0.1, 0.8, 0.1, 0.1])
    media_de_ap = (M.ap(y, q1) + M.ap(y, q2)) / 2
    ap_de_media = M.ap(y, (q1 + q2) / 2)
    assert media_de_ap != pytest.approx(ap_de_media)                 # la primaria es la media de AP por semilla, no el AP del promedio


def test_ap_ingenuo_no_depende_de_sklearn_para_la_prevalencia():
    y = np.array([1, 0, 0, 0])
    assert M.ap(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.25)     # todo empatado: AP = prevalencia


# ---- bootstrap agrupado
def banco_bootstrap(n_grupos_neg=60, n_grupos_pos=6, multi=3, semilla=0):
    """Imágenes agrupadas: algunos grupos con varias imágenes; predicciones para dos arquitecturas x tres semillas."""
    rng = np.random.default_rng(semilla)
    y, grupos = [], []
    for g in range(n_grupos_neg):
        for _ in range(1 + (g % multi == 0) * 2):
            y.append(0)
            grupos.append(f"n{g}")
    for g in range(n_grupos_pos):
        for _ in range(1 + (g % 2)):
            y.append(1)
            grupos.append(f"p{g}")
    y = np.array(y)
    preds = {}
    for a, calidad in (("A", 0.9), ("B", 0.5)):
        for s in (42, 43, 44):
            preds[(a, s)] = np.clip(rng.normal(0.3 + calidad * 0.4 * y, 0.25), 0, 1).astype(np.float32).astype(np.float64)
    return y, grupos, preds


def test_bootstrap_remuestrea_grupos_completos_con_conteos_por_clase():
    y, grupos, preds = banco_bootstrap()
    gpi, clase, nombres = BS.indexar_grupos(grupos, y)
    n_neg, n_pos = int((clase == 0).sum()), int((clase == 1).sum())
    c = BS.generar_conteos(clase, 200, 123, estratificado=True)
    assert c.shape == (200, len(nombres))
    assert (c[:, clase == 0].sum(axis=1) == n_neg).all() and (c[:, clase == 1].sum(axis=1) == n_pos).all()   # mismos nº de grupos por clase
    pesos = c[:, gpi]
    for b in range(5):                                              # cada imagen entra tantas veces como su grupo (grupos completos)
        for g in range(len(nombres)):
            assert len(set(pesos[b][gpi == g])) == 1
    assert (BS.generar_conteos(clase, 200, 123) == c).all()         # reproducible por semilla
    assert not (BS.generar_conteos(clase, 200, 124) == c).all()
    nes = BS.generar_conteos(clase, 200, 123, estratificado=False)
    assert (nes.sum(axis=1) == len(nombres)).all()
    assert (nes[:, clase == 1].sum(axis=1) != n_pos).any()          # sin estratificar, el nº de positivos varía


def test_bootstrap_grupo_mixto_se_rechaza():
    with pytest.raises(ValueError, match="mezcla clases"):
        BS.indexar_grupos(["a", "a"], [0, 1])


def test_bootstrap_remuestreos_sin_una_clase_se_cuentan():
    y = np.array([0] * 20 + [1] * 2)
    grupos = [f"n{i}" for i in range(20)] + ["p0", "p1"]
    gpi, clase, _ = BS.indexar_grupos(grupos, y)
    preds = {("A", 42): np.linspace(0, 1, 22)}
    c = BS.generar_conteos(clase, 4000, 7, estratificado=False)
    res = BS.analizar({k: M.preparar(y, p) for k, p in preds.items()}, ["A"], [42], c, gpi)
    sin_positivos = int(((c[:, clase == 1].sum(axis=1)) == 0).sum())
    assert sin_positivos > 0 and res["remuestreos_sin_una_clase"] == sin_positivos
    assert res["por_arquitectura"]["A"]["ap"]["n_remuestreos_validos"] == 4000 - sin_positivos
    c2 = BS.generar_conteos(clase, 200, 7, estratificado=True)       # estratificado: nunca falta una clase
    assert BS.analizar({k: M.preparar(y, p) for k, p in preds.items()}, ["A"], [42], c2, gpi)["remuestreos_sin_una_clase"] == 0


def test_bootstrap_ic_y_delta_emparejada_con_el_mismo_b():
    y, grupos, preds = banco_bootstrap()
    gpi, clase, _ = BS.indexar_grupos(grupos, y)
    preps = {k: M.preparar(y, p) for k, p in preds.items()}
    c = BS.generar_conteos(clase, 400, 20261121)
    res = BS.analizar(preps, ["A", "B"], [42, 43, 44], c, gpi)
    dist = BS.distribuciones(preps, c, gpi)
    # media entre semillas por remuestreo; Δ = M_A - M_B con el mismo b (no la diferencia de percentiles)
    mA = BS.media_entre_semillas(dist, "A", [42, 43, 44], "ap")
    mB = BS.media_entre_semillas(dist, "B", [42, 43, 44], "ap")
    lo, hi = np.percentile(mA - mB, [2.5, 97.5])
    d = res["deltas"]["A__menos__B"]["ap"]
    assert (d["ic_inf"], d["ic_sup"]) == pytest.approx((lo, hi)) and d["ic_excluye_cero"] == (lo > 0 or hi < 0)
    assert list(res["deltas"]) == ["A__menos__B"]                    # orden de las arquitecturas = orden de la configuración
    inv = BS.analizar(preps, ["B", "A"], [42, 43, 44], c, gpi)["deltas"]["B__menos__A"]["ap"]
    assert inv["ic_inf"] == pytest.approx(-d["ic_sup"]) and inv["ic_sup"] == pytest.approx(-d["ic_inf"])
    assert res["por_arquitectura"]["A"]["ap"]["estimacion"] == pytest.approx(
        np.mean([M.ap(y, preds[("A", s)]) for s in (42, 43, 44)]))
    # un único b para todo: la distribución de cada (a, s) es del mismo remuestreo -> la correlación de ΔAP no es una concatenación
    assert dist[("A", 42)]["ap"].shape == (400,)


def test_bootstrap_tres_semillas_identicas_no_cambian_el_ic():
    y, grupos, preds = banco_bootstrap()
    gpi, clase, _ = BS.indexar_grupos(grupos, y)
    una = preds[("A", 42)]
    tres = {("A", s): una for s in (42, 43, 44)}
    c = BS.generar_conteos(clase, 300, 5)
    r3 = BS.analizar({k: M.preparar(y, p) for k, p in tres.items()}, ["A"], [42, 43, 44], c, gpi)
    r1 = BS.analizar({("A", 42): M.preparar(y, una)}, ["A"], [42], c, gpi)
    for m in ("ap", "roc_auc"):
        assert r3["por_arquitectura"]["A"][m]["ic_inf"] == pytest.approx(r1["por_arquitectura"]["A"][m]["ic_inf"])
        assert r3["por_arquitectura"]["A"][m]["ic_sup"] == pytest.approx(r1["por_arquitectura"]["A"][m]["ic_sup"])
    # no se concatenan semillas como imágenes: el IC con 3 réplicas idénticas no se estrecha


def test_bootstrap_segunda_semilla_y_diferencia_maxima_y_guardado_de_indices(tmp_path):
    y, grupos, preds = banco_bootstrap()
    gpi, clase, nombres = BS.indexar_grupos(grupos, y)
    preps = {k: M.preparar(y, p) for k, p in preds.items()}
    a = BS.analizar(preps, ["A", "B"], [42, 43, 44], BS.generar_conteos(clase, 300, 20261121), gpi)
    b = BS.analizar(preps, ["A", "B"], [42, 43, 44], BS.generar_conteos(clase, 300, 20261122), gpi)
    dif = BS.diferencia_maxima_extremos(a, b)
    assert dif > 0 and BS.diferencia_maxima_extremos(a, a) == 0
    c = BS.generar_conteos(clase, 50, 1)
    BS.guardar_indices(tmp_path / "b.npz", c, semilla=1, estratificado=True, nombres_de_grupo=nombres)
    z = BS.cargar_indices(tmp_path / "b.npz")
    assert (z["conteos"] == c).all() and int(z["semilla"]) == 1
    with pytest.raises(FileExistsError):
        BS.guardar_indices(tmp_path / "b.npz", c, semilla=1, estratificado=True, nombres_de_grupo=nombres)   # no sobrescribe


# ---- calibración del umbral (§10.1), solo con datos sintéticos
@pytest.mark.parametrize("s_star, n_pos, k", [("0.90", 38, 35), ("0.7", 38, 27), ("0.8", 38, 31), ("0.95", 38, 37),
                                               ("1", 38, 38), ("0.90", 10, 9), ("0.9", 1, 1), (Fraction(9, 10), 38, 35)])
def test_k_en_aritmetica_entera(s_star, n_pos, k):
    assert CAL.k_objetivo(s_star, n_pos) == k


def test_k_es_exacto_donde_el_float_falla():
    import math
    assert math.ceil(0.07 * 100) == 8 and CAL.k_objetivo("0.07", 100) == 7          # 0.07*100 = 7.000000000000001 en float
    assert math.ceil(0.14 * 100) == 15 and CAL.k_objetivo("0.14", 100) == 14
    assert CAL.k_objetivo("0.57", 100) == 57 and CAL.k_objetivo("0.7", 10) == 7


def test_validaciones_de_calibracion():
    with pytest.raises(C.GuardError, match="fracción exacta"):
        CAL.como_fraccion(0.9)
    for malo in ("0", "-0.1", "1.1", "abc", True):
        with pytest.raises(C.GuardError):
            CAL.como_fraccion(malo)
    with pytest.raises(C.GuardError, match="n⁺"):
        CAL.k_objetivo("0.9", 0)
    with pytest.raises(C.GuardError, match="no tiene positivas"):
        CAL.calibrar_umbral([0, 0], [0.1, 0.2], "0.9")
    for malo in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(C.GuardError, match="finitas"):
            CAL.calibrar_umbral([1, 0], [0.9, malo], "0.9")
    with pytest.raises(C.GuardError, match="0 o 1"):
        CAL.calibrar_umbral([1, 2], [0.9, 0.1], "0.9")


def test_umbral_es_la_k_esima_mayor_puntuacion_positiva_y_el_mayor_compatible():
    pos = [0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55]            # n+ = 10
    neg = [0.97, 0.80, 0.50, 0.40, 0.30]
    y = [1] * 10 + [0] * 5
    r = CAL.calibrar_umbral(y, pos + neg, "0.80")
    assert (r["k"], r["umbral"], r["m"], r["n_pos"]) == (8, 0.65, 8, 10)
    assert r["sensibilidad_real"] == Fraction(8, 10) and r["fp"] == 2 and r["tn"] == 3 and r["fn"] == 2
    # el mayor: cualquier t > t_hat deja menos positivas que k; con t_hat se cumple S >= S*
    for t in (0.65, 0.5, 0.0):
        assert CAL.evaluar_en_umbral(y, pos + neg, t)["sensibilidad"] >= Fraction(8, 10)
    for t in (0.6500001, 0.7, 0.99):
        assert CAL.evaluar_en_umbral(y, pos + neg, t)["sensibilidad"] < Fraction(8, 10)


def test_empates_no_se_rompen_y_la_sensibilidad_realizada_puede_superar_s_star():
    # k = ceil(0.9 * 10) = 9 -> t_hat = 9.ª mayor positiva; las dos positivas más bajas están empatadas en 0.5
    pos = [0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.50, 0.50]
    neg = [0.50, 0.50, 0.20]                                                      # dos negativas empatadas en t_hat
    r = CAL.calibrar_umbral([1] * 10 + [0] * 3, pos + neg, "0.90")
    assert r["umbral"] == 0.50 and r["k"] == 9 and r["m"] == 10                     # m > k: entran TODAS las empatadas
    assert r["sensibilidad_real"] == 1 and r["sensibilidad_real"] > Fraction(9, 10)
    assert r["n_pos_empatadas"] == 2 and r["n_neg_empatadas"] == 2 and r["fp"] == 2  # las negativas empatadas cuentan como FP


def test_casos_limite_n_pos_1_y_s_star_1():
    r = CAL.calibrar_umbral([1, 0, 0], [0.4, 0.9, 0.1], "0.9")
    assert r["k"] == 1 and r["umbral"] == 0.4 and r["fp"] == 1
    r = CAL.calibrar_umbral([1, 1, 1, 0], [0.9, 0.5, 0.2, 0.1], 1)
    assert r["k"] == 3 and r["umbral"] == 0.2 and r["sensibilidad_real"] == 1


def test_convencion_mayor_o_igual_y_puntuaciones_float32_sin_perdida(tmp_path):
    rng = np.random.default_rng(3)
    s32 = rng.random(50).astype(np.float32)
    y = (rng.random(50) < 0.3).astype(int)
    y[:2] = 1
    CAL.guardar_puntuaciones(tmp_path / "p.csv", [f"i{k}" for k in range(50)], y, s32)
    ids, y2, s2 = CAL.cargar_puntuaciones(tmp_path / "p.csv")
    assert y2 == list(y) and [np.float32(v) for v in s2] == list(s32)                # ida y vuelta exacta
    a = CAL.calibrar_umbral(y, s32, "0.9")
    b = CAL.calibrar_umbral(y2, s2, "0.9")
    assert a["umbral"] == b["umbral"] and a["m"] == b["m"] and a["fp"] == b["fp"]
    redondeado = [round(v, 3) for v in s2]                                             # valores redondeados NO son equivalentes
    assert any(r != v for r, v in zip(redondeado, s2))
    with pytest.raises(C.GuardError):
        CAL.guardar_puntuaciones(tmp_path / "p.csv", ids, y2, s2)                       # no sobrescribe


def test_comprobacion_de_transferencia_un_fold_fuera():
    rng = np.random.default_rng(1)
    n = 200
    folds = np.arange(n) % 5
    y = (rng.random(n) < 0.15).astype(int)
    for k in range(5):                                                                  # al menos 3 positivas por fold
        y[np.flatnonzero(folds == k)[:3]] = 1
    s = np.clip(rng.normal(0.3 + 0.4 * y, 0.2), 0, 1)
    t = CAL.comprobar_transferencia(y, s, folds, "0.90")
    assert set(t["por_fold"]) == {0, 1, 2, 3, 4} and t["n_pos"] == int(y.sum())
    # t_hat de cada fold se calcula SIN ese fold
    k = 2
    ref = CAL.calibrar_umbral(y[folds != k], s[folds != k], "0.90")
    assert t["por_fold"][k]["umbral"] == ref["umbral"]
    manual = CAL.evaluar_en_umbral(y[folds == k], s[folds == k], ref["umbral"])
    assert (t["por_fold"][k]["tp"], t["por_fold"][k]["fn"]) == (manual["tp"], manual["fn"])
    assert t["sensibilidad_agrupada"] == Fraction(t["tp"], t["n_pos"])


def test_el_ejecutor_de_g2_no_invoca_la_calibracion_ni_fija_s_star():
    for nombre in ("ejecutar_g2.py", "entrenamiento.py", "agregacion.py", "modos.py", "bootstrap.py", "reporte.py"):
        texto = (SRC_R4 / nombre).read_text(encoding="utf-8")
        assert "calibracion" not in texto and "calibrar_umbral" not in texto, nombre


# ============================================================================ 4. ejecutor de un fold (pieza pura), agregación y K7
import zlib

from src.r4_v2 import agregacion as AG
from src.r4_v2 import entrenamiento as EN
from src.r4_v2 import modos as MO
from src.r4_v2 import pesos as W

RECETA_PARADA = {"early_stopping": {"paciencia": 5, "min_delta": 0.001}}


def test_checkpoint_minimo_estricto_y_empate_conserva_la_epoca_mas_temprana():
    assert EN.epoca_checkpoint([0.9, 0.7, 0.5, 0.45, 0.46]) == 4
    assert EN.epoca_checkpoint([0.9, 0.5, 0.7, 0.5, 0.5]) == 2             # empate exacto: la más temprana
    assert EN.epoca_checkpoint([0.5, 0.4999999]) == 2                       # sin min_delta: cualquier mínimo estricto guarda
    with pytest.raises(C.GuardError):
        EN.epoca_checkpoint([])


def test_parada_exige_mejora_mayor_que_min_delta_durante_la_paciencia():
    # mejoras menores que min_delta (0.001) respecto de la referencia 0.8 NO reinician la paciencia; el checkpoint SÍ guarda el mínimo
    perdida = [0.9, 0.8, 0.7995, 0.7993, 0.7991, 0.7992, 0.7994]
    assert EN.simular_parada(perdida, 5, 0.001) == 7                        # 5 épocas seguidas sin mejora mayor que min_delta
    assert EN.epoca_checkpoint(perdida) == 5                                # el checkpoint sigue el mínimo estricto (sin min_delta)
    assert EN.simular_parada([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], 5, 0.001) is None
    # la paciencia se reinicia con una mejora suficiente
    assert EN.simular_parada([0.9] * 4 + [0.8] * 6, 5, 0.001) == 10
    assert EN.simular_parada([0.9] * 4 + [0.8] * 5, 5, 0.001) is None


def test_historial_debe_seguir_la_regla():
    ok = [0.9, 0.7, 0.5, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5]
    assert EN.comprobar_historial(ok, RECETA_PARADA, 30) == {"epocas_entrenadas": 9, "parada_anticipada": True, "epoca_elegida": 4}
    with pytest.raises(C.GuardError, match="no sigue la regla de parada"):
        EN.comprobar_historial(ok + [0.51], RECETA_PARADA, 30)
    with pytest.raises(C.GuardError, match="sin activar la parada"):
        EN.comprobar_historial(ok[:5], RECETA_PARADA, 30)
    assert EN.comprobar_historial(ok[:2], RECETA_PARADA, 2)["parada_anticipada"] is False


def test_probabilidades_por_epoca_float32_exacto_y_formato(tmp_path):
    rng = np.random.default_rng(0)
    ids, y = ["a", "b", "c"], [0, 1, 0]
    ep = {e: (ids, y, rng.random(3).astype(np.float32)) for e in (1, 2)}
    EN.escribir_prob_por_epoca(tmp_path / "p.csv", ep)
    assert (tmp_path / "p.csv").read_text(encoding="utf-8").splitlines()[0] == "epoch,image_id,y_true,probability_malignant"   # formato de train.py
    leidas = EN.leer_prob_por_epoca(tmp_path / "p.csv")
    for e in (1, 2):
        assert leidas[e]["p"].dtype == np.float32 and (leidas[e]["p"] == ep[e][2]).all()                       # sin pérdida
    with pytest.raises(C.GuardError):
        EN.escribir_prob_por_epoca(tmp_path / "p.csv", ep)                                                      # no sobrescribe


def test_recarga_diferencia_cambios_de_clase_y_alerta_no_bloqueante():
    g = np.array([0.2, 0.5, 0.7], dtype=np.float32)
    assert EN.comparar_recarga(g, g.copy(), 0.01) == {"diferencia_maxima_absoluta": 0.0, "cambios_de_clase_a_0_5": 0,
                                                       "umbral_de_alerta": 0.01, "alerta_recarga": False}
    r = EN.comparar_recarga(g, np.array([0.2, 0.4999, 0.7005], dtype=np.float32), 0.01)
    assert r["cambios_de_clase_a_0_5"] == 1 and r["alerta_recarga"] is False                 # < 0.01: sin alerta
    r = EN.comparar_recarga(g, np.array([0.2, 0.5, 0.75], dtype=np.float32), 0.01)
    assert r["alerta_recarga"] is True and r["diferencia_maxima_absoluta"] == pytest.approx(0.05, abs=1e-6)
    with pytest.raises(C.GuardError):
        EN.comparar_recarga(g, g[:2], 0.01)


def test_sensibilidad_del_checkpoint_usa_perdida_ponderada_y_auc():
    y = np.array([1, 1, 0, 0, 0, 0])
    ids = [str(i) for i in range(6)]
    # época 1: separa bien (AUC 1) con probabilidades poco confiadas; época 2: peor AUC pero más confiada
    por = {1: {"image_id": ids, "y": y, "p": np.array([0.6, 0.6, 0.4, 0.4, 0.4, 0.4], dtype=np.float32)},
           2: {"image_id": ids, "y": y, "p": np.array([0.99, 0.01, 0.01, 0.01, 0.01, 0.6], dtype=np.float32)}}
    r = EN.sensibilidad_checkpoint(por, {0: 0.6, 1: 3.0}, 2)
    assert r["epoca_roc_auc_maximo"] == 1 and r["epoca_perdida_ponderada_minima"] == 1


def test_intentos_no_sobrescriben_y_estado_sin_run_config_es_fallo_tecnico(tmp_path):
    d1 = EN.crear_intento(tmp_path / "r50__s42__f0")
    d2 = EN.crear_intento(tmp_path / "r50__s42__f0")
    assert (d1.name, d2.name) == ("intento_01", "intento_02")
    assert EN.estado_de_intento(d1) == "fallo_tecnico"
    with pytest.raises(C.GuardError, match="K5"):
        C.directorio_nuevo(d1)
    assert EN.corrida_completada(tmp_path / "r50__s42__f0") is None


# ---- banco sintético: 45 corridas con entrenador y verificación simulados
CALIDAD = {"resnet50": 1.4, "densenet121": 1.0, "convnext_tiny": 0.6}
PERDIDA_9 = [0.9, 0.7, 0.5, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5]


def entrenador_sintetico(*, recarga_ruido=1e-4, fallar_en=None, epoca_guardada=None, calidad=None):
    calidad = calidad or CALIDAD

    def entrenar(ctx):
        if fallar_en == (ctx["arq"], ctx["semilla"], ctx["fold"]):
            raise RuntimeError("fallo sintético")
        rng = np.random.default_rng(zlib.crc32(f"{ctx['arq']}{ctx['semilla']}{ctx['fold']}".encode()))
        val = ctx["val"]
        ids = [r["image_id"] for r in val]
        y = np.array([r["label_v2"] for r in val])
        n_ep = min(9, ctx["epocas_max"])
        perdida = PERDIDA_9[:n_ep] if n_ep == 9 else [0.9 - 0.1 * i for i in range(n_ep)]
        por = {}
        for e in range(1, n_ep + 1):
            z = calidad[ctx["arq"]] * (2 * y - 1) + rng.normal(0, 1.2, size=y.size) - 1.5 * (1 + 0.0 * e)
            por[e] = (ids, y, (1 / (1 + np.exp(-z))).astype(np.float32))
        EN.escribir_prob_por_epoca(ctx["dir"] / "probabilidades_por_epoca.csv", por)
        (ctx["dir"] / "checkpoint_mejor.pth").write_bytes(b"checkpoint sintetico " + bytes(str(ctx["fold"]), "ascii"))
        elegida = EN.epoca_checkpoint(perdida)
        recarga = por[elegida][2] + np.float32(recarga_ruido)
        return {"historial": {"train_loss": [1.0] * n_ep, "val_loss_unweighted": perdida}, "checkpoint": ctx["dir"] / "checkpoint_mejor.pth",
                "prob_recarga": recarga, "epoca_checkpoint_guardado": epoca_guardada or elegida,
                "pesos_preentrenados": {"enum_pesos": "X_Weights.V1"}, "version_difiere_referencia": False,
                "entorno": {"python": "sintetico"}}
    return entrenar


def verificacion_falsa(cambia_al_final=False, bloqueo_al_final=False):
    llamadas = {"n": 0}

    def verificar():
        llamadas["n"] += 1
        archivo = "z" * 64 if (cambia_al_final and llamadas["n"] > 1) else "c" * 64
        bloqueos = ([{"codigo": "B_contenido_difiere", "ruta": "src/r4_v2/x.py", "detalle": "simulado"}]
                    if (bloqueo_al_final and llamadas["n"] > 1) else [])
        return {"bloqueos": bloqueos, "ok": not bloqueos, "git_commit_head": "h" * 40, "rama": "rama", "protocolo_versionado": True,
                "entrada": "modulo", "status_aux": [], "excepcion_protocolo": None, "modulos_en_repo": ["m"],
                "protocolo": {"sha256_contenido_blob": "p" * 64},
                "archivos": {C.CONFIG_REL: {"git_blob_id": "c" * 40, "sha256_contenido_blob": "c" * 64, "sha256_bytes_archivo": archivo}}}
    return verificar


REFERENCIA_FALSA = {"git_blob_id": "r" * 40, "sha256_contenido_blob": "r" * 64}


def lanzar_corrida(root, config, part, raiz_modo, arq, semilla, fold, *, modo="completa", entrenar=None, verificar=None, epocas_max=None):
    return EN.ejecutar_corrida(root, config, part, modo=modo, arq=arq, semilla=semilla, fold=fold, raiz_modo=raiz_modo,
                               verificar=verificar or verificacion_falsa(), entrenar=entrenar or entrenador_sintetico(),
                               referencia=REFERENCIA_FALSA, atestaciones={"autorizacion_ejecucion": "SINTETICA"}, epocas_max=epocas_max)


@pytest.fixture(scope="module")
def banco(tmp_path_factory, config_base):
    root = tmp_path_factory.mktemp("banco_g2")
    config = copy.deepcopy(config_base)
    escribir_particion(root, config)
    part = P.cargar_particion(root, config)
    raiz_modo = root / "results" / "completa"
    for a in config["diseno"]["arquitecturas"]:
        for s in config["diseno"]["semillas"]:
            for k in config["diseno"]["folds"]:
                r = lanzar_corrida(root, config, part, raiz_modo, a, s, k)
                assert r["estado"] == "completada", r
    return types.SimpleNamespace(root=root, config=config, part=part, raiz=raiz_modo)


def test_corrida_completada_artefactos_y_run_config(banco):
    d = banco.raiz / "resnet50__s42__f0" / "intento_01"
    assert {p.name for p in d.iterdir()} == {"intento_iniciado.json", "probabilidades_por_epoca.csv", "predicciones_oof.csv",
                                             "historial.json", "checkpoint_mejor.pth", "run_config.json"}
    rc = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
    assert rc["estado"] == "completada" and rc["invalida_integridad"] is False and rc["modo"] == "completa"
    assert rc["mapeo"] == "mapeo v2" and rc["particion"] == "phash_le8_v1" and rc["protocolo_id"] == "g2_baseline_v1"
    assert rc["leyenda"] == "partición provisional; conformidad académica pendiente"
    assert rc["sha256_bytes_archivo_particion"] == banco.part.sha256_bytes_archivo
    assert rc["checkpoint"]["sha256_bytes_archivo"] == C.sha256_archivo(d / "checkpoint_mejor.pth")
    assert rc["seleccion"]["epoca_elegida"] == 4 and rc["seleccion"]["epocas_entrenadas"] == 9
    assert "val_loss_unweighted" in json.loads((d / "historial.json").read_text(encoding="utf-8"))
    assert rc["referencia_pesos"] == REFERENCIA_FALSA and rc["alerta_recarga"] is False
    assert "verificacion_inicio" in rc and "verificacion_fin" in rc and rc["diferencias_inicio_fin"] == []
    assert abs(sum(rc["pesos_de_clase"].values()) - rc["n_train"] * (1 / (2 * (rc["n_train"] - rc["positivas_train"]))
                                                                    + 1 / (2 * rc["positivas_train"]))) < 1e-9
    texto = json.dumps(rc)
    for prohibido in ("sha256_trabajo", "sha256_archivo_pesos"):
        assert prohibido not in texto


def test_oof_de_la_corrida_es_el_checkpoint_elegido_y_solo_el_fold_retenido(banco):
    d = banco.raiz / "densenet121__s43__f2" / "intento_01"
    oof = EN.leer_prob_por_epoca(d / "probabilidades_por_epoca.csv")[4]
    filas = list(csv.DictReader(open(d / "predicciones_oof.csv", newline="")))
    assert [r["image_id"] for r in filas] == oof["image_id"] and {r["epoca_elegida"] for r in filas} == {"4"}
    ids_fold = {r["image_id"] for r in banco.part.filas if r["fold"] == 2}
    assert set(oof["image_id"]) == ids_fold                                           # solo el fold retenido
    assert [np.float32(float(r["probability_malignant"])) for r in filas] == list(oof["p"])


def test_k7_cada_imagen_una_vez_por_arquitectura_y_semilla(banco):
    corridas = AG.cargar_corridas(banco.raiz, banco.config, banco.part)
    assert len(corridas) == 45
    oof = AG.ensamblar_oof(corridas, banco.part, banco.config)
    assert len(oof.prob) == 9 and all(v.shape == (820,) and np.isfinite(v).all() for v in oof.prob.values())
    assert int(oof.y.sum()) == 38 and (oof.y == np.array([r["label_v2"] for r in banco.part.filas])).all()


def copiar_banco(banco, tmp_path):
    destino = tmp_path / "copia"
    shutil.copytree(banco.raiz, destino)
    return destino


def test_k7_falta_una_corrida_o_hay_duplicados_o_etiqueta_distinta(banco, tmp_path):
    raiz = copiar_banco(banco, tmp_path)
    shutil.rmtree(raiz / "resnet50__s42__f3")
    with pytest.raises(C.GuardError, match="K7.*faltan 1"):
        AG.cargar_corridas(raiz, banco.config, banco.part)
    raiz = copiar_banco(banco, tmp_path / "b")
    oof_csv = raiz / "resnet50__s42__f1" / "intento_01" / "predicciones_oof.csv"
    filas = oof_csv.read_text(encoding="utf-8").splitlines()
    oof_csv.write_text("\n".join(filas + [filas[1]]) + "\n")                         # una imagen dos veces
    corridas = AG.cargar_corridas(raiz, banco.config, banco.part)
    with pytest.raises(C.GuardError, match="K7"):
        AG.ensamblar_oof(corridas, banco.part, banco.config)
    raiz = copiar_banco(banco, tmp_path / "c")
    oof_csv = raiz / "resnet50__s42__f1" / "intento_01" / "predicciones_oof.csv"
    lineas = oof_csv.read_text(encoding="utf-8").splitlines()
    cab, primera = lineas[0], lineas[1].split(",")
    primera[1] = "1" if primera[1] == "0" else "0"                                    # y_true distinto de label_v2
    oof_csv.write_text("\n".join([cab, ",".join(primera)] + lineas[2:]) + "\n")
    with pytest.raises(C.GuardError, match="K7.*y_true"):
        AG.ensamblar_oof(AG.cargar_corridas(raiz, banco.config, banco.part), banco.part, banco.config)


def test_corridas_de_particion_o_referencia_distinta_no_son_comparables(banco, tmp_path):
    raiz = copiar_banco(banco, tmp_path)
    rc_path = raiz / "convnext_tiny__s44__f4" / "intento_01" / "run_config.json"
    rc = json.loads(rc_path.read_text(encoding="utf-8"))
    rc["referencia_pesos"]["sha256_contenido_blob"] = "9" * 64
    rc_path.write_text(json.dumps(rc))
    with pytest.raises(C.GuardError, match="no son comparables"):
        AG.cargar_corridas(raiz, banco.config, banco.part)
    otra = copy.deepcopy(banco.part)
    otra.sha256_bytes_archivo = "0" * 64
    with pytest.raises(C.GuardError, match="partición distinta"):
        AG.cargar_corridas(banco.raiz, banco.config, otra)


def test_las_corridas_humo_no_entran_en_la_agregacion(banco, tmp_path):
    raiz = copiar_banco(banco, tmp_path)
    rc_path = raiz / "resnet50__s42__f0" / "intento_01" / "run_config.json"
    rc = json.loads(rc_path.read_text(encoding="utf-8"))
    rc["modo"] = "humo"
    rc_path.write_text(json.dumps(rc))
    with pytest.raises(C.GuardError, match="no es una corrida completa válida"):
        AG.cargar_corridas(raiz, banco.config, banco.part)


def test_agregacion_ap_media_entre_semillas_contra_referencias_independientes(banco):
    oof = AG.ensamblar_oof(AG.cargar_corridas(banco.raiz, banco.config, banco.part), banco.part, banco.config)
    por = AG.metricas_por_semilla(oof, banco.config)
    for (a, s), p in oof.prob.items():
        assert por[(a, s)]["ap"] == pytest.approx(sk_metrics.average_precision_score(oof.y, p), abs=1e-12)
        assert por[(a, s)]["ap"] == pytest.approx(M.ap_ingenuo(oof.y, p), abs=1e-12)
        assert por[(a, s)]["roc_auc"] == pytest.approx(sk_metrics.roc_auc_score(oof.y, p), abs=1e-12)
    media = AG.media_entre_semillas(por, banco.config, "ap")
    for a in banco.config["diseno"]["arquitecturas"]:
        assert media[a] == pytest.approx(np.mean([sk_metrics.average_precision_score(oof.y, oof.prob[(a, s)])
                                                   for s in (42, 43, 44)]))
    assert media["resnet50"] > media["densenet121"] > media["convnext_tiny"] > 38 / 820 * 0.9     # orden de la calidad sintética


def test_ap_por_fold_y_ensamble_de_semillas_por_fold(banco):
    oof = AG.ensamblar_oof(AG.cargar_corridas(banco.raiz, banco.config, banco.part), banco.part, banco.config)
    pf = AG.ap_por_fold(oof, banco.config)
    assert len(pf) == 45 and all(0 < ap <= 1 and npos in (7, 8) for ap, npos in pf.values())
    k = 1
    m = oof.fold == k
    assert pf[("resnet50", 42, k)][0] == pytest.approx(sk_metrics.average_precision_score(oof.y[m], oof.prob[("resnet50", 42)][m]))
    ens = AG.ap_ensamble_por_fold(oof, banco.config)
    q = np.mean([oof.prob[("resnet50", s)] for s in (42, 43, 44)], axis=0)
    assert ens["resnet50"]["agregado"] == pytest.approx(sk_metrics.average_precision_score(oof.y, q))
    assert ens["resnet50"]["por_fold"][k] == pytest.approx(sk_metrics.average_precision_score(oof.y[m], q[m]))


def test_sensibilidad_del_checkpoint_en_la_agregacion(banco):
    corridas = AG.cargar_corridas(banco.raiz, banco.config, banco.part)
    alt = AG.ap_sensibilidad_checkpoint(corridas, banco.part, banco.config)
    assert set(alt) == {"epoca_perdida_ponderada_minima", "epoca_roc_auc_maximo"}
    assert all(set(v) == {"resnet50", "densenet121", "convnext_tiny"} for v in alt.values())


def test_bootstrap_primario_en_la_agregacion_con_pocos_remuestreos(banco):
    oof = AG.ensamblar_oof(AG.cargar_corridas(banco.raiz, banco.config, banco.part), banco.part, banco.config)
    res = AG.ejecutar_bootstrap(oof, banco.config, n_remuestreos=200)
    assert res["estratificado"]["semilla"] == 20261121 and res["estratificado_semilla_comprobacion"]["semilla"] == 20261122
    assert res["estratificado"]["remuestreos_sin_una_clase"] == 0 and res["diferencia_maxima_extremos_ic_entre_semillas"] >= 0
    assert set(res["estratificado"]["deltas"]) == {"resnet50__menos__densenet121", "resnet50__menos__convnext_tiny",
                                                    "densenet121__menos__convnext_tiny"}                       # 3 comparaciones
    ic = res["estratificado"]["por_arquitectura"]["resnet50"]["ap"]
    assert ic["ic_inf"] <= ic["estimacion"] <= ic["ic_sup"] or ic["ic_inf"] <= ic["ic_sup"]
    assert res["n_grupos"] == len({r["group_id"] for r in banco.part.filas})


# ---- fallos, intentos y K6/K8 en la corrida
def test_fallo_tecnico_conserva_el_intento_y_el_siguiente_es_nuevo(banco, tmp_path):
    raiz = tmp_path / "res"
    r1 = lanzar_corrida(banco.root, banco.config, banco.part, raiz, "resnet50", 42, 0,
                        entrenar=entrenador_sintetico(fallar_en=("resnet50", 42, 0)))
    assert r1["estado"] == "fallo_tecnico" and "fallo sintético" in r1["motivo"] and r1["intento"] == 1
    r2 = lanzar_corrida(banco.root, banco.config, banco.part, raiz, "resnet50", 42, 0)             # reintento, misma semilla
    assert r2["estado"] == "completada" and r2["intento"] == 2
    assert (raiz / "resnet50__s42__f0" / "intento_01" / "run_config.json").is_file()               # el fallido se conserva
    assert EN.estado_de_intento(raiz / "resnet50__s42__f0" / "intento_01") == "fallo_tecnico"
    assert EN.corrida_completada(raiz / "resnet50__s42__f0").name == "intento_02"


def test_detenida_por_k8_se_registra_como_intento(banco, tmp_path):
    def k8(ctx):
        raise C.GuardError("K8: los pesos efectivos de resnet50 difieren de la referencia")
    r = lanzar_corrida(banco.root, banco.config, banco.part, tmp_path / "res", "resnet50", 42, 0, entrenar=k8)
    assert r["estado"] == "detenida_k8"
    assert json.loads((r["dir"] / "run_config.json").read_text(encoding="utf-8"))["estado"] == "detenida_k8"


def test_integridad_bloqueada_antes_de_entrenar_no_lanza_ni_deja_directorio(banco, tmp_path):
    def bloqueada():
        res = verificacion_falsa()()
        res["bloqueos"] = [{"codigo": "B_contenido_difiere", "ruta": "src/r4_v2/x.py", "detalle": ""}]
        return res
    llamado = []
    with pytest.raises(C.GuardError, match="K6"):
        lanzar_corrida(banco.root, banco.config, banco.part, tmp_path / "res", "resnet50", 42, 0, verificar=bloqueada,
                       entrenar=lambda ctx: llamado.append(1))
    assert not llamado and not (tmp_path / "res").exists()


def test_cambio_entre_inicio_y_fin_marca_invalida_integridad(banco, tmp_path):
    r = lanzar_corrida(banco.root, banco.config, banco.part, tmp_path / "res", "resnet50", 42, 0,
                       verificar=verificacion_falsa(cambia_al_final=True))
    assert r["estado"] == "invalida_integridad"
    rc = json.loads((r["dir"] / "run_config.json").read_text(encoding="utf-8"))
    assert rc["invalida_integridad"] is True and rc["diferencias_inicio_fin"] == [f"archivo:{C.CONFIG_REL}"]
    assert EN.corrida_completada(tmp_path / "res" / "resnet50__s42__f0") is None                   # no entra en los análisis


def test_checkpoint_guardado_en_otra_epoca_es_fallo_tecnico(banco, tmp_path):
    r = lanzar_corrida(banco.root, banco.config, banco.part, tmp_path / "res", "resnet50", 42, 0,
                       entrenar=entrenador_sintetico(epoca_guardada=3))
    assert r["estado"] == "fallo_tecnico" and "mínimo estricto" in r["motivo"]


def test_alerta_de_recarga_no_bloquea_ni_cambia_el_checkpoint(banco, tmp_path):
    r = lanzar_corrida(banco.root, banco.config, banco.part, tmp_path / "res", "resnet50", 42, 0,
                       entrenar=entrenador_sintetico(recarga_ruido=0.05))
    rc = json.loads((r["dir"] / "run_config.json").read_text(encoding="utf-8"))
    assert r["estado"] == "completada" and rc["alerta_recarga"] is True and rc["seleccion"]["epoca_elegida"] == 4
    assert rc["recarga"]["diferencia_maxima_absoluta"] == pytest.approx(0.05, abs=1e-4)


def test_humo_limita_las_epocas_y_no_cuenta_entre_las_45(banco, tmp_path):
    r = lanzar_corrida(banco.root, banco.config, banco.part, tmp_path / "humo", "resnet50", 42, 0, modo="humo", epocas_max=2)
    rc = json.loads((r["dir"] / "run_config.json").read_text(encoding="utf-8"))
    assert r["estado"] == "completada" and rc["modo"] == "humo" and rc["seleccion"]["epocas_entrenadas"] == 2
    assert (tmp_path / "humo" / "resnet50__s42__f0" / "intento_01").is_dir() and not (tmp_path / "completa").exists()


def test_la_salida_de_resultados_es_segura(tmp_path, config_base):
    assert MO.raiz_del_modo(tmp_path, config_base, "humo").as_posix().endswith("results/r4_v2/phash_le8_v1/g2_baseline_v1/humo")
    with pytest.raises(C.GuardError):
        C.exigir_salida_segura(tmp_path, tmp_path / "results" / "classification" / "x", "results/r4_v2/phash_le8_v1/g2_baseline_v1")
    with pytest.raises(C.GuardError, match="agentes o reportes"):
        C.exigir_salida_segura(tmp_path, tmp_path / "results" / "r4_v2" / "phash_le8_v1" / "g2_baseline_v1" / ".claude",
                               "results/r4_v2/phash_le8_v1/g2_baseline_v1")
    with pytest.raises(C.GuardError, match="test"):
        C.exigir_salida_segura(tmp_path, tmp_path / "results" / "r4_v2" / "phash_le8_v1" / "g2_baseline_v1" / "Test",
                               "results/r4_v2/phash_le8_v1/g2_baseline_v1")


# ============================================================================ 5. reporte determinista
from src.r4_v2 import reporte as RP


def test_reporte_tablas_deterministas_con_leyenda_y_sin_sobrescribir(banco, tmp_path):
    kw = dict(n_remuestreos=200, con_figuras=False)
    m1 = RP.generar_reporte(banco.root, banco.config, banco.part, banco.root / "results", tmp_path / "r1", verificar=verificacion_falsa(), **kw)
    m2 = RP.generar_reporte(banco.root, banco.config, banco.part, banco.root / "results", tmp_path / "r2", verificar=verificacion_falsa(), **kw)
    nombres = ["T1_particion.csv", "T2_arquitecturas.csv", "T3_por_semilla.csv", "T3_ap_por_fold.csv", "T4_delta_ap.csv",
               "T5_sensibilidades.csv", "T6_trazabilidad.csv", "bootstrap_resumen.json"]
    for n in nombres:                                                                  # mismas entradas -> mismos bytes
        assert (tmp_path / "r1" / n).read_bytes() == (tmp_path / "r2" / n).read_bytes(), n
        assert m1["sha256_bytes_archivo"][n] == m2["sha256_bytes_archivo"][n]
    for n in nombres[:-1]:
        primera = (tmp_path / "r1" / n).read_text(encoding="utf-8").splitlines()[0]
        assert "mapeo v2" in primera and "phash_le8_v1 (provisional)" in primera and "g2_baseline_v1" in primera
        assert "45 corridas" in primera and "sin uso de test" in primera and "conformidad académica pendiente" in primera
    assert m1["n_corridas"] == 45 and len(list((tmp_path / "r1").glob("indices_b_*.npz"))) == 3   # índices b guardados
    with pytest.raises(C.GuardError, match="K5"):
        RP.generar_reporte(banco.root, banco.config, banco.part, banco.root / "results", tmp_path / "r1", verificar=verificacion_falsa(), **kw)


def leer_tabla(path):
    lineas = path.read_text(encoding="utf-8").splitlines()[1:]
    return list(csv.DictReader(lineas))


def test_reporte_contenido_t1_t2_t4_t6(banco, tmp_path):
    RP.generar_reporte(banco.root, banco.config, banco.part, banco.root / "results", tmp_path / "r", verificar=verificacion_falsa(), n_remuestreos=200, con_figuras=False)
    t1 = leer_tabla(tmp_path / "r" / "T1_particion.csv")
    assert t1[-1]["fold"] == "total" and (t1[-1]["n"], t1[-1]["positivas"]) == ("820", "38")
    assert [r["positivas"] for r in t1[:5]] == ["8", "8", "8", "7", "7"]
    t2 = leer_tabla(tmp_path / "r" / "T2_arquitecturas.csv")
    assert [r["arquitectura"] for r in t2] == ["resnet50", "densenet121", "convnext_tiny"]
    assert {"sensibilidad_en_especificidad_0.90", "sensibilidad_en_especificidad_0.95"} <= set(t2[0])
    assert not any("umbral" in c or "s_star" in c.lower() for c in t2[0])                  # puntos de la ROC, no un umbral ni S*
    assert float(t2[0]["ap_ic95_inf"]) <= float(t2[0]["ap_ic95_sup"])
    t4 = leer_tabla(tmp_path / "r" / "T4_delta_ap.csv")
    assert len(t4) == 6 and {r["lectura"] for r in t4} <= {"diferencia observada en AP", "no concluyente"}
    t6 = leer_tabla(tmp_path / "r" / "T6_trazabilidad.csv")
    assert len(t6) == 45 and {r["modo"] for r in t6} == {"completa"} and {r["estado"] for r in t6} == {"completada"}
    assert all(len(r["sha256_bytes_archivo_checkpoint"]) == 64 for r in t6)


def test_t6_incluye_intentos_fallidos_humo_y_referencia_solo_como_intentos(banco, tmp_path):
    res = tmp_path / "res"
    lanzar_corrida(banco.root, banco.config, banco.part, res / "completa", "resnet50", 42, 0, entrenar=entrenador_sintetico(fallar_en=("resnet50", 42, 0)))
    lanzar_corrida(banco.root, banco.config, banco.part, res / "completa", "resnet50", 42, 0)
    lanzar_corrida(banco.root, banco.config, banco.part, res / "humo", "resnet50", 42, 0, modo="humo", epocas_max=1)
    (res / "referencia" / "intento_01").mkdir(parents=True)
    t6 = RP.tabla_t6(res)
    assert [(r["modo"], r["intento"], r["estado"]) for r in t6] == [
        ("referencia", "intento_01", "fallo_tecnico"), ("humo", "intento_01", "completada"),
        ("completa", "intento_01", "fallo_tecnico"), ("completa", "intento_02", "completada")]


def test_reporte_figuras_una_por_proposito(banco, tmp_path):
    pytest.importorskip("matplotlib")
    meta = RP.generar_reporte(banco.root, banco.config, banco.part, banco.root / "results", tmp_path / "r", verificar=verificacion_falsa(), n_remuestreos=100)
    assert meta["figuras"] == ["F1_ap_por_arquitectura.png", "F2_delta_ap.png", "F3_ap_por_fold.png", "F4_curva_pr_agregada.png"]
    assert all((tmp_path / "r" / n).stat().st_size > 1000 for n in meta["figuras"])


def test_reporte_sin_las_45_corridas_no_escribe_nada(banco, tmp_path):
    raiz = copiar_banco(banco, tmp_path)
    shutil.rmtree(raiz / "resnet50__s42__f0")
    (tmp_path / "res").mkdir()
    shutil.move(str(raiz), str(tmp_path / "res" / "completa"))
    with pytest.raises(C.GuardError, match="K7"):
        RP.generar_reporte(banco.root, banco.config, banco.part, tmp_path / "res", tmp_path / "salida", verificar=verificacion_falsa(), n_remuestreos=50, con_figuras=False)
    assert not (tmp_path / "salida").exists()


# ============================================================================ 6. pesos preentrenados (K8) y referencia
def archivo_de_pesos(tmp_path, nombre="resnet50-0676ba61.pth", datos=b"pesos sinteticos"):
    p = tmp_path / "cache" / "checkpoints" / nombre
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(datos)
    return p


def resolucion(ruta, enum="ResNet50_Weights.IMAGENET1K_V2", url="https://example.org/w/resnet50-0676ba61.pth"):
    return {"enum_pesos": enum, "url": url, "archivo_cache": ruta}


def test_k8_procedencia_registra_archivo_real_con_ruta_relativa(tmp_path):
    ruta = archivo_de_pesos(tmp_path)
    reg = W.registrar_procedencia("resnet50", resolucion(ruta))
    assert reg["sha256_bytes_archivo"] == hashlib.sha256(b"pesos sinteticos").hexdigest() and reg["tamano_bytes"] == 16
    assert reg["ruta_cache_relativa"] == "checkpoints/resnet50-0676ba61.pth" and reg["nombre_archivo_cache"] == ruta.name
    assert str(tmp_path) not in json.dumps(reg)                                                  # nunca rutas absolutas


@pytest.mark.parametrize("caso", ["ausente", "vacio", "sin_url", "sin_enum", "arquitectura_desconocida"])
def test_k8_se_detiene_si_la_procedencia_no_es_fiable(tmp_path, caso):
    ruta = archivo_de_pesos(tmp_path)
    res = resolucion(ruta)
    arq = "resnet50"
    if caso == "ausente":
        ruta.unlink()
    elif caso == "vacio":
        ruta.write_bytes(b"")
    elif caso == "sin_url":
        res["url"] = ""
    elif caso == "sin_enum":
        res["enum_pesos"] = None
    else:
        arq = "vgg16"
    with pytest.raises(C.GuardError, match="K8"):
        W.registrar_procedencia(arq, res)


def test_k8_pretrained_falso_y_arquitectura_efectiva_distinta(tmp_path):
    ruta = archivo_de_pesos(tmp_path)
    resnet = type("ResNet", (), {})()
    res = lambda a: resolucion(ruta)
    assert W.procedencia_tras_instanciar("resnet50", resnet, resolver=res)["tamano_bytes"] == 16
    with pytest.raises(C.GuardError, match="K8.*pretrained=False"):
        W.procedencia_tras_instanciar("resnet50", resnet, pretrained=False, resolver=res)
    with pytest.raises(C.GuardError, match="K8.*arquitectura efectiva"):
        W.procedencia_tras_instanciar("densenet121", resnet, resolver=res)


def registros_falsos(tmp_path):
    out = {}
    for a, n in (("resnet50", "r"), ("densenet121", "d"), ("convnext_tiny", "c")):
        ruta = archivo_de_pesos(tmp_path, f"{a}.pth", n.encode() * 10)
        out[a] = W.registrar_procedencia(a, resolucion(ruta, enum=f"{a}_Weights.V1", url=f"https://example.org/{a}.pth"))
    return out


VERSIONES = {"torch_version": "2.5.1+cu121", "torchvision_version": "0.20.1+cu121"}


def test_candidato_de_referencia_esquema_y_validacion(tmp_path):
    cand = W.construir_candidato(registros_falsos(tmp_path), VERSIONES, "2026-11-22")
    assert W.problemas_de_referencia(cand) == [] and cand["modo_origen"] == "referencia"
    assert set(cand["arquitecturas"]["resnet50"]) == set(W.CAMPOS_POR_ARQUITECTURA)
    assert "ruta_cache_relativa" not in json.dumps(cand) and "sha256_trabajo" not in json.dumps(cand)
    malo = copy.deepcopy(cand)
    del malo["arquitecturas"]["densenet121"]
    assert any("densenet121" in p for p in W.problemas_de_referencia(malo))
    malo = copy.deepcopy(cand)
    malo["arquitecturas"]["resnet50"]["sha256_bytes_archivo"] = "xyz"
    assert any("sha256" in p for p in W.problemas_de_referencia(malo))
    malo = copy.deepcopy(cand)
    malo["arquitecturas"]["resnet50"]["nombre_archivo_cache"] = "C:\\Users\\alguien\\x.pth"
    assert "contiene una ruta absoluta" in W.problemas_de_referencia(malo)
    malo = copy.deepcopy(cand)
    malo["modo_origen"] = "humo"
    assert any("modo_origen" in p for p in W.problemas_de_referencia(malo))
    with pytest.raises(C.GuardError):
        W.construir_candidato({"resnet50": cand["arquitecturas"]["resnet50"]}, VERSIONES, "2026-11-22")


def test_k8_referencia_ausente_no_rastreada_modificada_o_invalida(tmp_path):
    cand = W.construir_candidato(registros_falsos(tmp_path), VERSIONES, "2026-11-22")
    texto = json.dumps(cand, indent=2, sort_keys=True) + chr(10)
    with pytest.raises(C.GuardError, match="K8.*no existe"):                                   # humo/completa se detienen
        W.cargar_referencia(crear_repo(tmp_path, "r_ausente"))
    repo = crear_repo(tmp_path, "r_valida")
    escribir(repo, W.REFERENCIA_REL, texto)                                                    # sin versionar
    with pytest.raises(C.GuardError, match="K8.*rastreada o está modificada"):
        W.cargar_referencia(repo)
    commit_all(repo, "referencia")
    ref = W.cargar_referencia(repo)                                                            # rastreada y sin cambios
    assert len(ref["git_blob_id"]) == 40 and len(ref["sha256_contenido_blob"]) == 64 and ref["datos"]["esquema"] == W.ESQUEMA_REFERENCIA
    escribir(repo, W.REFERENCIA_REL, texto.replace("2026-11-22", "2026-11-23"))                # modificada en el árbol
    with pytest.raises(C.GuardError, match="K8.*modificada"):
        W.cargar_referencia(repo)
    run_git(repo, "add", W.REFERENCIA_REL)                                                     # preparada y árbol de vuelta a HEAD
    escribir(repo, W.REFERENCIA_REL, texto)
    with pytest.raises(C.GuardError, match="K8.*preparado"):
        W.cargar_referencia(repo)
    incompleto = copy.deepcopy(cand)
    del incompleto["arquitecturas"]["convnext_tiny"]
    repo3 = crear_repo(tmp_path, "r_incompleta")
    escribir(repo3, W.REFERENCIA_REL, json.dumps(incompleto))
    commit_all(repo3, "referencia incompleta")
    with pytest.raises(C.GuardError, match="K8.*incompleto"):
        W.cargar_referencia(repo3)
    escribir(repo3, W.REFERENCIA_REL, "no es json")
    commit_all(repo3, "ilegible")
    with pytest.raises(C.GuardError, match="K8"):
        W.cargar_referencia(repo3)


def test_k8_comparacion_con_la_referencia(tmp_path):
    regs = registros_falsos(tmp_path)
    ref = {"datos": W.construir_candidato(regs, VERSIONES, "2026-11-22")}
    assert W.comparar_con_referencia("resnet50", regs["resnet50"], ref, VERSIONES) == {"version_difiere_referencia": False}
    # solo difieren las versiones de torch: se registra sin bloquear
    assert W.comparar_con_referencia("resnet50", regs["resnet50"], ref, {**VERSIONES, "torch_version": "2.6.0"}) == {"version_difiere_referencia": True}
    for campo, valor in (("sha256_bytes_archivo", "0" * 64), ("url", "https://otro"), ("enum_pesos", "X.Y"), ("tamano_bytes", 1)):
        distinto = {**regs["resnet50"], campo: valor}
        with pytest.raises(C.GuardError, match="K8.*difieren"):
            W.comparar_con_referencia("resnet50", distinto, ref, VERSIONES)


# ============================================================================ 7. modos §4.7
def test_atestaciones_por_modo_y_sin_otorgar_conformidad():
    ok = {"auditoria_independiente": "CONFIRMADO:AUD-001", "autorizacion_ejecucion": "CONFIRMADO:AUT-001"}
    assert set(MO.exigir_atestaciones("referencia", ok)) == set(ok)                           # referencia no exige conformidad ni partición [P; X10]
    for modo in ("humo", "completa"):                                                          # enmienda 2: la conformidad ya no se exige
        assert set(MO.exigir_atestaciones(modo, ok)) == set(ok)
        assert "conformidad_asesor" in MO.exigir_atestaciones(modo, {**ok, "conformidad_asesor": "CONFIRMADO:ASE-001"})
    for falta in ok:
        with pytest.raises(C.GuardError):
            MO.exigir_atestaciones("referencia", {**ok, falta: "  "})
    with pytest.raises(C.GuardError, match="modo desconocido"):
        MO.exigir_atestaciones("primera_corrida_de_humo", ok)


def test_plan_de_corridas_por_modo(config_base):
    assert MO.plan_de_corridas("referencia", config_base) == []
    h = MO.plan_de_corridas("humo", config_base)
    assert len(h) == 1 and h[0]["id"] == "resnet50__s42__f0"
    c = MO.plan_de_corridas("completa", config_base)
    assert len(c) == 45 and len({x["id"] for x in c}) == 45 and c[0]["id"] == "resnet50__s42__f0" and c[-1]["id"] == "convnext_tiny__s44__f4"
    assert MO.plan_de_corridas("humo", config_base, humo_arq="densenet121", humo_semilla=43, humo_fold=2)[0]["id"] == "densenet121__s43__f2"
    with pytest.raises(C.GuardError):
        MO.plan_de_corridas("humo", config_base, humo_arq="vgg16")
    assert MO.epocas_humo(config_base, 1) == 1 and MO.epocas_humo(config_base, 2) == 2
    for malo in (0, 3):
        with pytest.raises(C.GuardError):
            MO.epocas_humo(config_base, malo)


def test_completa_exige_un_humo_previo_sin_fallo_con_la_misma_tupla(banco, tmp_path):
    kw = dict(sha256_bytes_archivo_particion=banco.part.sha256_bytes_archivo, sha256_contenido_blob_config="c" * 64,
              sha256_contenido_blob_referencia="r" * 64)
    humo = tmp_path / "humo"
    with pytest.raises(C.GuardError, match="humo"):
        MO.exigir_humo_previo(humo, **kw)
    lanzar_corrida(banco.root, banco.config, banco.part, humo, "resnet50", 42, 0, modo="humo", epocas_max=1,
                   entrenar=entrenador_sintetico(fallar_en=("resnet50", 42, 0)))              # solo un fallo técnico
    with pytest.raises(C.GuardError, match="humo"):
        MO.exigir_humo_previo(humo, **kw)
    lanzar_corrida(banco.root, banco.config, banco.part, humo, "resnet50", 42, 0, modo="humo", epocas_max=1)
    assert MO.exigir_humo_previo(humo, **kw).name == "intento_02"
    for campo, valor in (("sha256_bytes_archivo_particion", "0" * 64), ("sha256_contenido_blob_config", "0" * 64),
                         ("sha256_contenido_blob_referencia", "0" * 64)):
        with pytest.raises(C.GuardError, match="humo"):
            MO.exigir_humo_previo(humo, **{**kw, campo: valor})


# ---- modo referencia y ejecutor de punta a punta en un repositorio Git temporal sintético
def config_para_repo(repo, config_base, hash_csv=None):
    cfg = copy.deepcopy(config_base)
    cfg["integridad"]["rutas_verificadas_lista_fija"] = [PROTO, "configs/r4_v2/", "src/r4_v2/", ".gitattributes"]
    cfg["protocolo"]["git_blob_id"] = run_git(repo, "rev-parse", f"HEAD:{PROTO}").stdout.decode().strip()
    cfg["protocolo"]["sha256_contenido_blob"] = hashlib.sha256(run_git(repo, "cat-file", "blob", f"HEAD:{PROTO}").stdout).hexdigest()
    return cfg


def instanciar_falso(arq):
    return type(W.CLASE_ESPERADA[arq], (), {})()


def resolver_falso(tmp_path):
    def resolver(arq):
        return resolucion(archivo_de_pesos(tmp_path, f"{arq}.pth", arq.encode()), enum=f"{arq}_Weights.V1", url=f"https://example.org/{arq}.pth")
    return resolver


def test_modo_referencia_no_lee_datos_no_entrena_ni_escribe_en_configs(tmp_path, config_base, monkeypatch):
    repo = crear_repo(tmp_path)
    cfg = config_para_repo(repo, config_base)
    antes_configs = {p.name: C.sha256_archivo(p) for p in (repo / "configs").rglob("*") if p.is_file()}
    monkeypatch.setattr(P, "cargar_particion", lambda *a, **k: (_ for _ in ()).throw(AssertionError("referencia leyó la partición")))
    raiz_modo = MO.raiz_del_modo(repo, cfg, "referencia")
    r = MO.ejecutar_referencia(repo, cfg, verificar=lambda: I.verificar_integridad(repo, cfg, modo="referencia", entorno=entorno_ok(repo, tmp_path)),
                               instanciar=instanciar_falso, resolver=resolver_falso(tmp_path), versiones=lambda: VERSIONES,
                               raiz_modo=raiz_modo, atestaciones={"autorizacion_ejecucion": "E"})
    assert r["estado"] == "completada"
    archivos = sorted(p.name for p in r["dir"].iterdir())
    assert archivos == ["referencia_pesos_preentrenados_v1.candidato.json", "run_config.json"]      # sin probabilidades, checkpoints ni métricas
    rc = json.loads((r["dir"] / "run_config.json").read_text(encoding="utf-8"))
    assert rc["modo"] == "referencia" and rc["sin_metricas"] is True and rc["protocolo_versionado"] is True
    cand = json.loads((r["dir"] / "referencia_pesos_preentrenados_v1.candidato.json").read_text(encoding="utf-8"))
    assert W.problemas_de_referencia(cand) == [] and cand["modo_origen"] == "referencia"
    assert rc["pesos_preentrenados"]["resnet50"]["sha256_bytes_archivo"] == cand["arquitecturas"]["resnet50"]["sha256_bytes_archivo"]
    assert {p.name: C.sha256_archivo(p) for p in (repo / "configs").rglob("*") if p.is_file()} == antes_configs   # no escribe en configs/
    assert not (repo / "configs" / "r4_v2" / "referencia_pesos_preentrenados_v1.json").exists()
    assert r["dir"].relative_to(repo).as_posix().startswith("results/r4_v2/phash_le8_v1/g2_baseline_v1/referencia/intento_")


def test_modo_referencia_se_detiene_por_k8_o_por_integridad(tmp_path, config_base):
    repo = crear_repo(tmp_path)
    cfg = config_para_repo(repo, config_base)
    verif = lambda: I.verificar_integridad(repo, cfg, modo="referencia", entorno=entorno_ok(repo, tmp_path))
    raiz = MO.raiz_del_modo(repo, cfg, "referencia")
    def resolver_roto(arq):
        return resolucion(tmp_path / "no_existe.pth")
    r = MO.ejecutar_referencia(repo, cfg, verificar=verif, instanciar=instanciar_falso, resolver=resolver_roto,
                               versiones=lambda: VERSIONES, raiz_modo=raiz)
    assert r["estado"] == "detenida_k8" and not (r["dir"] / "referencia_pesos_preentrenados_v1.candidato.json").exists()
    escribir(repo, "src/r4_v2/modulo_a.py", "VALOR = 99\n")                                    # K6: código modificado
    with pytest.raises(C.GuardError, match="K6"):
        MO.ejecutar_referencia(repo, cfg, verificar=verif, instanciar=instanciar_falso, resolver=resolver_falso(tmp_path),
                               versiones=lambda: VERSIONES, raiz_modo=raiz)


ATEST = ["--autorizacion-ejecucion", "CONFIRMADO:D-901", "--auditoria-independiente", "CONFIRMADO:AUD-901"]
ATEST_HUMO = ATEST + ["--conformidad-asesor", "CONFIRMADO:ASE-901"]


def banco_ejecutor(tmp_path, config_base):
    """Repositorio sintético con config YAML, protocolo y código mínimos, y una partición sintética no versionada."""
    repo = crear_repo(tmp_path)
    cfg = config_para_repo(repo, config_base)
    escribir_particion(repo, cfg)                                                              # CSV sin versionar (fuera de las rutas verificadas)
    (repo / "configs" / "r4_v2" / "g2_baseline_v1.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    commit_all(repo, "config")
    return repo, cfg


def inyecciones(repo, tmp_path, **extra):
    return dict(root=repo, precargar=lambda: None, instanciar=instanciar_falso, resolver=resolver_falso(tmp_path),
                versiones=lambda: VERSIONES, entorno=lambda: entorno_ok(repo, tmp_path), **extra)


def test_ejecutor_sin_atestaciones_no_hace_nada(tmp_path, config_base, capsys):
    repo, cfg = banco_ejecutor(tmp_path, config_base)
    from src.r4_v2 import ejecutar_g2 as EJ
    assert EJ.main(["--modo", "referencia"], **inyecciones(repo, tmp_path)) == 3
    assert "autorizacion_ejecucion" in capsys.readouterr().err and not (repo / "results").exists()


def test_flujo_referencia_humo_completa_con_entrenador_sintetico(tmp_path, config_base, monkeypatch, capsys):
    from src.r4_v2 import ejecutar_g2 as EJ
    repo, cfg = banco_ejecutor(tmp_path, config_base)
    inj = inyecciones(repo, tmp_path, entrenar=entrenador_sintetico())
    # humo y completa sin referencia de pesos: se detienen (K8) y no crean resultados
    assert EJ.main(["--modo", "humo", *ATEST_HUMO], **inj) == 3
    assert "K8" in capsys.readouterr().err and not (repo / "results").exists()
    # 1. referencia: genera el candidato; el autor lo verifica y lo versiona (aquí, simulado byte a byte)
    assert EJ.main(["--modo", "referencia", *ATEST], **inj) == 0
    candidato = next((repo / "results").rglob("referencia_pesos_preentrenados_v1.candidato.json"))
    escribir(repo, W.REFERENCIA_REL, candidato.read_bytes())
    # referencia sin versionar: humo sigue detenido
    assert EJ.main(["--modo", "humo", *ATEST_HUMO], **inj) == 3
    assert "K8" in capsys.readouterr().err
    commit_all(repo, "referencia verificada por el autor")
    # completa sin humo previo: se detiene antes de entrenar
    assert EJ.main(["--modo", "completa", *ATEST_HUMO], **inj) == 3
    assert "humo" in capsys.readouterr().err and not list((repo / "results").glob("**/completa/*/*"))
    # 2. humo: una arquitectura, un fold, 1 época
    assert EJ.main(["--modo", "humo", "--epocas-humo", "1", *ATEST_HUMO], **inj) == 0
    raiz = repo / "results/r4_v2/phash_le8_v1/g2_baseline_v1"
    rc = json.loads(next(raiz.glob("humo/*/intento_01/run_config.json")).read_text(encoding="utf-8"))
    assert rc["modo"] == "humo" and rc["referencia_pesos"]["git_blob_id"] == run_git(repo, "rev-parse", f"HEAD:{W.REFERENCIA_REL}").stdout.decode().strip()
    assert rc["verificacion_inicio"]["ok"] and rc["verificacion_fin"]["ok"] and rc["invalida_integridad"] is False
    assert rc["atestaciones_del_autor"]["conformidad_asesor"] == "CONFIRMADO:ASE-901"
    # 3. completa (plan reducido a 2 corridas para la prueba): reanuda sin repetir una corrida válida
    plan = MO.plan_de_corridas("completa", cfg)[:2]
    monkeypatch.setattr(MO, "plan_de_corridas", lambda modo, config, **kw: plan)
    assert EJ.main(["--modo", "completa", *ATEST_HUMO], **inj) == 0
    assert len(list(raiz.glob("completa/*/intento_01/run_config.json"))) == 2
    capsys.readouterr()
    assert EJ.main(["--modo", "completa", *ATEST_HUMO], **inj) == 0
    assert "ya completada" in capsys.readouterr().out and len(list(raiz.glob("completa/*/intento_*"))) == 2   # no se repite ni sobrescribe


def test_ejecutor_se_detiene_ante_k1_hash_de_particion(tmp_path, config_base, capsys):
    from src.r4_v2 import ejecutar_g2 as EJ
    repo, cfg = banco_ejecutor(tmp_path, config_base)
    ruta = repo / cfg["particion"]["csv"]["ruta"]
    ruta.write_bytes(ruta.read_bytes() + b" ")
    # una referencia válida y versionada para que el primer bloqueo sea K1 y no K8
    inj = inyecciones(repo, tmp_path, entrenar=entrenador_sintetico())
    assert EJ.main(["--modo", "referencia", *ATEST], **inj) == 0
    escribir(repo, W.REFERENCIA_REL, next((repo / "results").rglob("*.candidato.json")).read_bytes())
    commit_all(repo, "ref")
    assert EJ.main(["--modo", "humo", *ATEST_HUMO], **inj) == 3
    assert "K1" in capsys.readouterr().err
    assert not list((repo / "results").glob("**/humo/*"))


def test_fallo_tecnico_detiene_la_secuencia_y_conserva_el_intento(tmp_path, config_base, capsys):
    from src.r4_v2 import ejecutar_g2 as EJ
    repo, cfg = banco_ejecutor(tmp_path, config_base)
    inj = inyecciones(repo, tmp_path, entrenar=entrenador_sintetico(fallar_en=("resnet50", 42, 0)))
    assert EJ.main(["--modo", "referencia", *ATEST], **inj) == 0
    escribir(repo, W.REFERENCIA_REL, next((repo / "results").rglob("*.candidato.json")).read_bytes())
    commit_all(repo, "ref")
    assert EJ.main(["--modo", "humo", *ATEST_HUMO], **inj) == 2
    assert "fallo_tecnico" in capsys.readouterr().out
    assert len(list((repo / "results").glob("**/humo/*/intento_01"))) == 1


# ============================================================================ 8. estáticas: alcance, vocabulario y compilación
MODULOS_R4 = sorted(SRC_R4.glob("*.py"))


def test_no_existe_src_init_y_no_se_reutilizan_modulos_prohibidos():
    import ast
    assert not (REPO / "src" / "__init__.py").exists()                                       # src es paquete de espacio de nombres
    modulos_prohibidos = ("cv_folds_e2", "run_experiment", "final_evaluation", "classification.dataset", "r2_v2", "robustez_e2")
    for mod in MODULOS_R4:
        texto = mod.read_text(encoding="utf-8")
        importados = []
        for nodo in ast.walk(ast.parse(texto)):
            if isinstance(nodo, ast.Import):
                importados += [a.name for a in nodo.names]
            elif isinstance(nodo, ast.ImportFrom):
                importados += [(nodo.module or "") + "." + a.name for a in nodo.names] + [nodo.module or ""]
        for p in modulos_prohibidos:
            assert not any(p in i for i in importados), f"{mod.name} importa {p}"
        for ruta in ("r8_evaluacion_final", "protocolo_r8_congelado"):                          # nunca se nombran rutas de R8/test
            assert ruta not in texto, f"{mod.name} menciona {ruta}"
        if mod.name != "common.py":                                                              # solo common.py nombra los archivos que RECHAZA
            assert "cv_folds_e2" not in texto.replace("cv_folds_e2.py", "") or mod.name == "particion.py", mod.name


def test_imports_de_classification_solo_los_reutilizables():
    import ast
    permitidos = {"models", "transforms", "dataloader", "evaluate", "losses", "utils", "train", "training_options"}
    vistos = set()
    for mod in MODULOS_R4:
        for nodo in ast.walk(ast.parse(mod.read_text(encoding="utf-8"))):
            if isinstance(nodo, ast.ImportFrom) and nodo.module and nodo.module.startswith("src.classification"):
                if nodo.module == "src.classification":
                    nombres = {a.name for a in nodo.names}
                else:
                    nombres = {nodo.module.split(".")[2]}
                assert nombres <= permitidos, (mod.name, nodo.module, nombres)
                vistos |= nombres
            elif isinstance(nodo, ast.Import):
                assert not any(a.name.startswith("src.classification") for a in nodo.names), mod.name
    assert vistos and vistos <= permitidos


def test_torch_solo_con_importacion_diferida():
    import ast
    for mod in MODULOS_R4:
        arbol = ast.parse(mod.read_text(encoding="utf-8"))
        for nodo in arbol.body:                                                                # solo el nivel del módulo
            nombres = []
            if isinstance(nodo, ast.Import):
                nombres = [a.name.split(".")[0] for a in nodo.names]
            elif isinstance(nodo, ast.ImportFrom) and nodo.module and nodo.level == 0:
                nombres = [nodo.module.split(".")[0]]
            elif isinstance(nodo, ast.ImportFrom) and nodo.level == 1:
                assert nodo.module != "dataset_lesiones", f"{mod.name} importa dataset_lesiones de forma no diferida"
            if mod.name == "dataset_lesiones.py":                                              # módulo dedicado: importa torch/PIL al cargarse
                continue
            assert not {"torch", "torchvision", "PIL", "sklearn", "src"} & set(nombres), (mod.name, nombres)


def test_vocabulario_de_hashes_sin_sinonimos_en_codigo_notebook_y_configuracion():
    candidatos = [*MODULOS_R4, REPO / "notebooks" / "R4_v2" / "01_g2_baseline_v1.ipynb", CONFIG_REAL]
    for archivo in candidatos:
        texto = archivo.read_text(encoding="utf-8")
        for sinonimo in ("sha256_trabajo", "sha256_archivo_pesos"):
            assert sinonimo not in texto, (archivo.name, sinonimo)
    assert C.CAMPOS_HASH == ("git_blob_id", "sha256_contenido_blob", "sha256_bytes_archivo")


def test_py_compile_de_todo_src_r4_v2_y_de_las_pruebas():
    for archivo in [*MODULOS_R4, Path(__file__)]:
        compile(archivo.read_text(encoding="utf-8"), str(archivo), "exec")                  # sin escribir .pyc en el árbol


def test_protocolo_y_configuracion_congelados_coinciden_con_el_blob_de_head():
    r = subprocess.run(["git", "-C", str(REPO), "cat-file", "blob", f"HEAD:{PROTO}"], capture_output=True)
    if r.returncode != 0:
        pytest.skip("el protocolo no está en HEAD de este clon")
    cfg = C.cargar_config(CONFIG_REAL)
    assert hashlib.sha256(r.stdout).hexdigest() == cfg["protocolo"]["sha256_contenido_blob"]
    assert subprocess.run(["git", "-C", str(REPO), "rev-parse", f"HEAD:{PROTO}"], capture_output=True).stdout.decode().strip() == \
        cfg["protocolo"]["git_blob_id"]


def test_notebook_orquestador_minimo_sin_salidas_y_por_subproceso():
    nb = json.loads((REPO / "notebooks" / "R4_v2" / "01_g2_baseline_v1.ipynb").read_text(encoding="utf-8"))
    codigo = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert nb["cells"][0]["source"][0].startswith("# R4-G2 - ")
    assert "RUN_TRAINING = False" in codigo and "subprocess.run" in codigo and "pycache_prefix=" in codigo and '"-B"' in codigo
    assert "-m\", \"src.r4_v2.ejecutar_g2" in codigo.replace("'", '"') or '"-m", "src.r4_v2.ejecutar_g2"' in codigo
    assert all(c.get("outputs") == [] and c.get("execution_count") is None for c in nb["cells"] if c["cell_type"] == "code")
    texto = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    for seccion in ("Objetivo", "Configuración y fuente del protocolo", "Método", "Verificación", "Limitaciones", "Interpretación"):
        assert seccion in texto
    assert "Pendiente" in texto and "partición provisional" in texto.lower() and "conformidad académica pendiente" in texto.lower()
    assert "import torch" not in codigo and "train_model" not in codigo                      # sin lógica de entrenamiento en el notebook


# ============================================================================ 9. torch [ND localmente]: se saltan sin torch
def test_torch_pesos_de_clase_iguales_a_compute_class_weights_historico():
    torch = pytest.importorskip("torch", reason="[ND] requiere torch (Falcon)")
    from src.classification.utils import compute_class_weights
    etiquetas = [0] * 625 + [1] * 31
    w = P.pesos_de_clase(etiquetas)
    assert torch.allclose(torch.tensor([w[0], w[1]], dtype=torch.float32), compute_class_weights(etiquetas, alpha=1.0))


def test_torch_train_model_checkpoint_y_parada_coinciden_con_la_regla_simulada(tmp_path):
    torch = pytest.importorskip("torch", reason="[ND] requiere torch (Falcon)")
    pytest.importorskip("sklearn")
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from src.classification.train import train_model
    g = torch.Generator().manual_seed(0)
    x = torch.randn(80, 4, generator=g)
    y = (x[:, 0] + 0.5 * torch.randn(80, generator=g) > 0.8).long()
    y[:4] = 1
    y[4:8] = 0
    xv = torch.randn(40, 4, generator=g)
    yv = (xv[:, 0] + 0.5 * torch.randn(40, generator=g) > 0.8).long()
    yv[:3] = 1
    yv[3:6] = 0
    modelo = nn.Linear(4, 2)
    opt = torch.optim.AdamW(modelo.parameters(), lr=0.05)
    pesos = torch.tensor([0.6, 3.0])
    ids = [f"v{i}" for i in range(40)]
    receta = {"early_stopping": {"paciencia": 3, "min_delta": 0.001}}
    hist = train_model(modelo, DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True),
                       DataLoader(TensorDataset(xv, yv), batch_size=16, shuffle=False), nn.CrossEntropyLoss(weight=pesos), opt,
                       torch.device("cpu"), epochs=25, save_path=str(tmp_path / "m.pth"), patience=3, min_delta=0.001, seed=1,
                       use_amp=False, predictions_path=str(tmp_path / "p.csv"), val_image_ids=ids,
                       val_expected_labels=yv.numpy(), checkpoint_metric="val_loss", val_criterion=nn.CrossEntropyLoss())
    assert hist["val_loss"] == hist["val_unweighted_loss"]                                  # la magnitud histórica «ponderada» es la sin ponderar
    sel = EN.comprobar_historial(hist["val_loss"], receta, 25)
    meta = json.loads((tmp_path / "m_meta.json").read_text(encoding="utf-8"))
    assert meta["epoch"] == sel["epoca_elegida"]
    por_epoca = EN.leer_prob_por_epoca(tmp_path / "p.csv")
    EN.verificar_prob_por_epoca(por_epoca, [{"image_id": i, "label_v2": int(t)} for i, t in zip(ids, yv)], sel["epocas_entrenadas"])
    modelo.load_state_dict(torch.load(tmp_path / "m.pth", weights_only=True))
    with torch.no_grad():
        p = torch.softmax(modelo(xv), dim=1)[:, 1].numpy()
    assert np.max(np.abs(p - por_epoca[sel["epoca_elegida"]]["p"])) < 1e-6                  # recarga coherente con la probabilidad guardada


# ============================================================================ 10. refuerzos tras la revisión de mutaciones
def test_k4_e1_split_distinto_de_train_y_validation_aunque_la_ruta_sea_coherente(tmp_path, config):
    filas = filas_sinteticas()
    filas[0].update(e1_split="otro", image_path=f"{IMAGENES_DIR}/otro/images/{filas[0]['image_name']}")
    with pytest.raises(C.GuardError, match="K4.*e1_split"):
        cargar(tmp_path, config, filas=filas)


def test_cambio_del_conjunto_de_modulos_entre_inicio_y_fin(tmp_path):
    repo = crear_repo(tmp_path)
    ini = verificar(repo, tmp_path)
    fin = verificar(repo, tmp_path, modulos=[modulo_falso("alias_raiz", repo / "raiz_mod.py")])    # mismo archivo, un módulo más
    assert I.comparar_instantaneas(ini, fin) == ["modulos_en_repo"]


def test_k7_por_epoca_imagenes_duplicadas_faltantes_o_con_otra_etiqueta():
    val = [{"image_id": str(i), "label_v2": i % 2} for i in range(4)]
    ids = [r["image_id"] for r in val]
    y = np.array([r["label_v2"] for r in val])
    bien = {1: {"image_id": ids, "y": y, "p": np.full(4, 0.5, dtype=np.float32)}}
    EN.verificar_prob_por_epoca(bien, val, 1)
    with pytest.raises(C.GuardError, match="incompletas"):
        EN.verificar_prob_por_epoca(bien, val, 2)
    dup = {1: {**bien[1], "image_id": ids[:3] + [ids[0]]}}
    with pytest.raises(C.GuardError, match="K7"):
        EN.verificar_prob_por_epoca(dup, val, 1)
    otra = {1: {**bien[1], "y": 1 - y}}
    with pytest.raises(C.GuardError, match="K7"):
        EN.verificar_prob_por_epoca(otra, val, 1)
    with pytest.raises(C.GuardError, match="no válidas"):
        EN.verificar_prob_por_epoca({1: {**bien[1], "p": np.array([0.1, 0.2, 1.5, 0.3], dtype=np.float32)}}, val, 1)


def test_k7_en_la_corrida_una_imagen_repetida_en_las_probabilidades_es_fallo_tecnico(banco, tmp_path):
    base = entrenador_sintetico()

    def corrupto(ctx):
        res = base(ctx)
        ruta = ctx["dir"] / "probabilidades_por_epoca.csv"
        lineas = ruta.read_text(encoding="utf-8").splitlines()
        ruta.write_text(chr(10).join(lineas + [lineas[1]]) + chr(10), encoding="utf-8")        # imagen repetida en la época 1
        return res
    r = lanzar_corrida(banco.root, banco.config, banco.part, tmp_path / "res", "resnet50", 42, 0, entrenar=corrupto)
    assert r["estado"] == "fallo_tecnico" and "K7" in r["motivo"]


def test_plan_completa_exige_que_las_corridas_sumen_n_corridas_de_la_configuracion(config_base):
    incoherente = copy.deepcopy(config_base)
    incoherente["diseno"]["n_corridas"] = 44
    with pytest.raises(C.GuardError, match="45 corridas|44"):
        MO.plan_de_corridas("completa", incoherente)


# ============================================================================ 11. correcciones de la auditoría independiente (I-1..I-4, M-1, M-5, M-6, M-12)
# ---- I-1: lista fija cerrada en el CÓDIGO; la del YAML no puede estrechar ni ampliar K6
def test_i1_lista_del_yaml_recortada_o_ampliada_bloquea_y_el_alcance_sigue_siendo_el_del_codigo(tmp_path):
    repo = crear_repo(tmp_path)
    recortada = config_integridad(repo)
    recortada["integridad"]["rutas_verificadas_lista_fija"] = [PROTO, ".gitattributes"]      # sin configs/r4_v2/ ni src/r4_v2/
    res = verificar(repo, tmp_path, cfg=recortada)
    assert "lista_fija_discrepa" in codigos(res)
    detalle = next(b["detalle"] for b in res["bloqueos"] if b["codigo"] == "lista_fija_discrepa")
    assert "src/r4_v2/" in detalle and "configs/r4_v2/" in detalle                          # dice qué falta en el YAML
    escribir(repo, "src/r4_v2/modulo_a.py", "VALOR = 2\n")                                  # el alcance real sigue siendo el del código
    assert "B_contenido_difiere" in codigos(verificar(repo, tmp_path, cfg=recortada))
    ampliada = config_integridad(repo, extra_fijas=["notebooks/orq.ipynb"])
    res = verificar(repo, tmp_path, cfg=ampliada)
    assert "lista_fija_discrepa" in codigos(res) and "notebooks/orq.ipynb" in next(
        b["detalle"] for b in res["bloqueos"] if b["codigo"] == "lista_fija_discrepa")
    assert "notebooks/orq.ipynb" not in res["archivos"]                                             # una ampliación del YAML no cambia el alcance


def test_i1_lista_igual_en_otro_orden_o_con_barras_distintas_no_bloquea(tmp_path):
    repo = crear_repo(tmp_path)
    cfg = config_integridad(repo)
    cfg["integridad"]["rutas_verificadas_lista_fija"] = list(reversed(cfg["integridad"]["rutas_verificadas_lista_fija"]))
    assert verificar(repo, tmp_path, cfg=cfg)["ok"]


def test_i1_el_valor_por_defecto_es_la_constante_del_codigo_y_la_de_las_pruebas_no_se_filtra(tmp_path):
    repo = crear_repo(tmp_path)
    assert I.LISTA_FIJA_CERRADA == LISTA_PEQUENA                                              # parcheada solo por el fixture autouse
    cfg = config_integridad(repo)
    assert "lista_fija_discrepa" in codigos(verificar(repo, tmp_path, cfg=cfg, lista_fija=LISTA_REAL))
    assert isinstance(LISTA_REAL, tuple) and LISTA_REAL[0] == PROTO


# ---- I-2: el reporte aplica §4.5 antes de generar y registra el commit del código
def test_i2_el_reporte_no_se_genera_si_la_integridad_esta_bloqueada(banco, tmp_path):
    def bloqueada():
        res = verificacion_falsa()()
        res["bloqueos"] = [{"codigo": "B_contenido_difiere", "ruta": "src/r4_v2/x.py", "detalle": ""}]
        return res
    with pytest.raises(C.GuardError, match="K6"):
        RP.generar_reporte(banco.root, banco.config, banco.part, banco.root / "results", tmp_path / "r", verificar=bloqueada,
                           n_remuestreos=20, con_figuras=False)
    assert not (tmp_path / "r").exists()
    with pytest.raises(TypeError):                                                            # `verificar` es obligatorio: no se puede omitir
        RP.generar_reporte(banco.root, banco.config, banco.part, banco.root / "results", tmp_path / "r2", n_remuestreos=20)


def test_i2_el_reporte_registra_commit_integridad_y_commits_de_las_corridas(banco, tmp_path):
    kw = dict(n_remuestreos=20, con_figuras=False)
    meta = RP.generar_reporte(banco.root, banco.config, banco.part, banco.root / "results", tmp_path / "a",
                              verificar=verificacion_falsa(), **kw)
    en_disco = json.loads((tmp_path / "a" / "reporte_metadatos.json").read_text(encoding="utf-8"))
    assert en_disco["git_commit_head"] == "h" * 40 and en_disco["rama"] == "rama" and en_disco["protocolo_versionado"] is True
    assert set(en_disco["integridad"][C.CONFIG_REL]) == set(C.CAMPOS_HASH)                       # vocabulario único
    assert en_disco["invalida_integridad"] is False and en_disco["diferencias_inicio_fin"] == []
    assert en_disco["git_commit_head_de_las_corridas"] == ["h" * 40]
    assert en_disco["commits_de_las_corridas_difieren_del_reporte"] is False and meta["git_commit_head"] == "h" * 40

    def otro_commit():
        return {**verificacion_falsa()(), "git_commit_head": "x" * 40}
    meta = RP.generar_reporte(banco.root, banco.config, banco.part, banco.root / "results", tmp_path / "b", verificar=otro_commit, **kw)
    assert meta["git_commit_head"] == "x" * 40 and meta["commits_de_las_corridas_difieren_del_reporte"] is True
    meta = RP.generar_reporte(banco.root, banco.config, banco.part, banco.root / "results", tmp_path / "c",
                              verificar=verificacion_falsa(cambia_al_final=True), **kw)
    assert meta["invalida_integridad"] is True and meta["diferencias_inicio_fin"] == [f"archivo:{C.CONFIG_REL}"]


# ---- I-3: solo confirmaciones positivas inequívocas habilitan humo/completa
FORMATO_OK = {"conformidad_asesor": "CONFIRMADO:ASE-001", "auditoria_independiente": "CONFIRMADO:AUD-001",
              "autorizacion_ejecucion": "CONFIRMADO:AUT-001"}
MARCADORES_INVALIDOS = ["PENDIENTE", "pendiente", "Pendiente", "no", "No", "NO", "si", "sí", "SI", "ok", "true", "n/a", "ninguna",
                        "", " ", "   ", "\t", "\n", "CONFIRMADO", "CONFIRMADO:", "CONFIRMADO:PENDIENTE", "CONFIRMADO:NO",
                        "confirmado:ASE-001", "Confirmado:ASE-001", "CONFIRMADO:ase-001", "CONFIRMADO ASE-001", "CONFIRMADO:ASE-1",
                        "CONFIRMADO:ASE-0001", "CONFIRMADO:ASE-00A", " CONFIRMADO:ASE-001", "CONFIRMADO:ASE-001 ",
                        "CONFIRMADO:ASE-001\n", "CONFIRMADO:ASE-001;PENDIENTE", "registro real", "cualquier cosa", None, True, 1, 0]


@pytest.mark.parametrize("modo", ["humo", "completa"])
@pytest.mark.parametrize("campo", ["conformidad_asesor", "auditoria_independiente", "autorizacion_ejecucion"])
def test_i3_marcadores_y_cadenas_arbitrarias_no_habilitan_humo_ni_completa(modo, campo):
    assert set(MO.exigir_atestaciones(modo, FORMATO_OK)) == set(FORMATO_OK)                   # el formato válido sí pasa
    opcional = campo == "conformidad_asesor"                                                  # enmienda 2: vacía/ausente se ignora
    for malo in MARCADORES_INVALIDOS:
        if opcional and (malo is None or malo == ""):
            continue
        with pytest.raises(C.GuardError, match=campo):
            MO.exigir_atestaciones(modo, {**FORMATO_OK, campo: malo})
    if not opcional:
        with pytest.raises(C.GuardError, match=campo):                                        # clave ausente
            MO.exigir_atestaciones(modo, {k: v for k, v in FORMATO_OK.items() if k != campo})


def test_i3_cada_atestacion_exige_su_propio_prefijo_y_el_caso_del_informe_de_auditoria():
    cruzada = {**FORMATO_OK, "conformidad_asesor": "CONFIRMADO:AUD-001"}                     # una auditoría no es conformidad del asesor
    with pytest.raises(C.GuardError, match="conformidad_asesor"):
        MO.exigir_atestaciones("completa", cruzada)
    with pytest.raises(C.GuardError):                                                         # el caso comprobado por el auditor
        MO.exigir_atestaciones("completa", {"conformidad_asesor": "PENDIENTE", "auditoria_independiente": "pendiente",
                                            "autorizacion_ejecucion": "no"})
    assert MO.exigir_atestaciones("referencia", {"auditoria_independiente": "CONFIRMADO:AUD-002",
                                                 "autorizacion_ejecucion": "CONFIRMADO:D-029"})
    for valida in ("CONFIRMADO:D-029", "CONFIRMADO:AUT-123"):
        assert MO.exigir_atestaciones("completa", {**FORMATO_OK, "autorizacion_ejecucion": valida})
    with pytest.raises(C.GuardError):                                                         # `referencia` tampoco acepta marcadores
        MO.exigir_atestaciones("referencia", {"auditoria_independiente": "pendiente", "autorizacion_ejecucion": "CONFIRMADO:D-029"})
    with pytest.raises(C.GuardError, match="No se otorgan ni se infieren"):
        MO.exigir_atestaciones("humo", {})                                                    # el mensaje no otorga nada


def test_i3_el_ejecutor_se_detiene_con_atestaciones_pendientes_y_no_crea_nada(tmp_path, config_base, capsys):
    from src.r4_v2 import ejecutar_g2 as EJ
    repo, cfg = banco_ejecutor(tmp_path, config_base)
    inj = inyecciones(repo, tmp_path, entrenar=entrenador_sintetico())
    for modo in ("humo", "completa"):
        for valor in ("PENDIENTE", "pendiente", "no", " ", "x", "CONFIRMADO:ASE-1"):         # "" se ignora (enmienda 2)
            args = ["--modo", modo, "--autorizacion-ejecucion", "CONFIRMADO:D-901", "--auditoria-independiente", "CONFIRMADO:AUD-901",
                    "--conformidad-asesor", valor]
            assert EJ.main(args, **inj) == 3
            assert "conformidad_asesor" in capsys.readouterr().err
    assert not (repo / "results").exists()


# ---- I-4: IC bootstrap con tres semillas DISTINTAS frente a una implementación independiente
def test_i4_ic_con_tres_semillas_distintas_coincide_con_el_calculo_independiente_y_difiere_de_una_sola():
    y, grupos, preds = banco_bootstrap()
    semillas = [42, 43, 44]
    assert not np.array_equal(preds[("A", 42)], preds[("A", 43)])                      # semillas realmente distintas
    gpi, clase, _ = BS.indexar_grupos(grupos, y)
    conteos = BS.generar_conteos(clase, 80, 20261121)
    res = BS.analizar({k: M.preparar(y, p) for k, p in preds.items()}, ["A", "B"], semillas, conteos, gpi)
    pesos = conteos[:, gpi].astype(int)
    media = {a: [] for a in "AB"}
    primera = {a: [] for a in "AB"}
    for b in range(conteos.shape[0]):                                                   # referencia independiente: muestra expandida + sklearn
        idx = np.repeat(np.arange(y.size), pesos[b])
        for a in "AB":
            por_semilla = [sk_metrics.average_precision_score(y[idx], preds[(a, s)][idx]) for s in semillas]
            media[a].append(np.mean(por_semilla))
            primera[a].append(por_semilla[0])
    for a in "AB":
        lo, hi = np.percentile(media[a], [2.5, 97.5])
        ic = res["por_arquitectura"][a]["ap"]
        assert (ic["ic_inf"], ic["ic_sup"]) == pytest.approx((lo, hi), abs=1e-9)
        lo1, hi1 = np.percentile(primera[a], [2.5, 97.5])
        assert abs(lo - lo1) + abs(hi - hi1) > 1e-4                                       # una sola semilla daría OTRO intervalo
        assert ic["estimacion"] == pytest.approx(np.mean([M.ap(y, preds[(a, s)]) for s in semillas]), abs=1e-12)
    d = np.array(media["A"]) - np.array(media["B"])
    lo, hi = np.percentile(d, [2.5, 97.5])
    delta = res["deltas"]["A__menos__B"]["ap"]
    assert (delta["ic_inf"], delta["ic_sup"]) == pytest.approx((lo, hi), abs=1e-9)
    d1 = np.array(primera["A"]) - np.array(primera["B"])
    assert abs(np.percentile(d1, 2.5) - lo) + abs(np.percentile(d1, 97.5) - hi) > 1e-4


def test_i4_la_distribucion_por_remuestreo_usa_la_media_de_las_tres_semillas():
    y, grupos, preds = banco_bootstrap()
    gpi, clase, _ = BS.indexar_grupos(grupos, y)
    c = BS.generar_conteos(clase, 12, 3)
    preps = {k: M.preparar(y, p) for k, p in preds.items()}
    dist = BS.distribuciones(preps, c, gpi)
    esperado = [np.mean([sk_metrics.average_precision_score(np.repeat(y, c[b][gpi]), np.repeat(preds[("A", s)], c[b][gpi]))
                         for s in (42, 43, 44)]) for b in range(12)]
    assert BS.media_entre_semillas(dist, "A", [42, 43, 44], "ap") == pytest.approx(esperado, abs=1e-12)
    assert BS.media_entre_semillas(dist, "A", [42], "ap") != pytest.approx(esperado, abs=1e-6)


# ---- M-5: un error de Git bloquea, nunca pasa en silencio
def test_m5_git_lista_z_y_git_texto_obligatorio_lanzan_en_un_directorio_que_no_es_repositorio(tmp_path):
    fuera = tmp_path / "no_repo"
    fuera.mkdir()
    with pytest.raises(C.GuardError, match="K6.*Git"):
        C.git_lista_z(fuera, "ls-files", "-z")
    with pytest.raises(C.GuardError, match="K6.*Git"):
        C.git_texto(fuera, "rev-parse", "HEAD", obligatorio=True)
    assert C.git_texto(fuera, "rev-parse", "HEAD") is None                                    # el modo no obligatorio se conserva


@pytest.mark.parametrize("subcomando", ["ls-tree", "ls-files", "diff", "status", "check-attr", "log", "hash-object"])
def test_m5_un_fallo_de_git_en_cualquier_control_lanza_excepcion(tmp_path, monkeypatch, subcomando):
    repo = crear_repo(tmp_path)
    assert verificar(repo, tmp_path)["ok"]
    real = C.git

    def git_roto(root, *args, **kw):
        if subcomando in args:
            return subprocess.CompletedProcess(args, 128, b"", b"fatal: fallo simulado")
        return real(root, *args, **kw)
    monkeypatch.setattr(C, "git", git_roto)
    with pytest.raises(C.GuardError, match="K6.*Git"):
        verificar(repo, tmp_path)


def test_m5_rev_parse_de_un_blob_que_falla_lanza_excepcion(tmp_path, monkeypatch):
    repo = crear_repo(tmp_path)
    real = C.git

    def git_roto(root, *args, **kw):
        if "rev-parse" in args and any(str(a).startswith("HEAD:") for a in args):
            return subprocess.CompletedProcess(args, 128, b"", b"fatal")
        return real(root, *args, **kw)
    monkeypatch.setattr(C, "git", git_roto)
    with pytest.raises(C.GuardError, match="K6.*Git"):
        verificar(repo, tmp_path)


# ---- M-12: formato estricto de los hashes recibidos por CLI
@pytest.mark.parametrize("valor", ["a" * 64, "0123456789abcdef" * 4, hashlib.sha256(b"x").hexdigest()])
def test_m12_sha256_valido(valor):
    assert C.validar_sha256(valor) == valor


@pytest.mark.parametrize("valor", ["A" * 64, "a" * 63, "a" * 65, "g" + "a" * 63, " " + "a" * 63, "a" * 64 + "\n", "a" * 32, "", None, 7,
                                   "sha256:" + "a" * 64, ("a" * 64).upper()])
def test_m12_sha256_invalido_se_rechaza_sin_normalizar(valor):
    with pytest.raises(C.GuardError, match="SHA-256"):
        C.validar_sha256(valor, "--sha256-borrador-aprobado")


def test_m12_el_ejecutor_rechaza_un_hash_mal_formado_o_sin_motivo(tmp_path, config_base, capsys):
    from src.r4_v2 import ejecutar_g2 as EJ
    repo, cfg = banco_ejecutor(tmp_path, config_base)
    inj = inyecciones(repo, tmp_path)
    for sha in ("A" * 64, "abc", "a" * 63, " " + "a" * 64):
        assert EJ.main(["--modo", "referencia", *ATEST, "--excepcion-protocolo-no-versionado", "motivo", "--sha256-borrador-aprobado", sha],
                       **inj) == 3
        assert "SHA-256" in capsys.readouterr().err
    assert EJ.main(["--modo", "referencia", *ATEST, "--excepcion-protocolo-no-versionado", "motivo"], **inj) == 3     # falta el hash
    assert EJ.main(["--modo", "referencia", *ATEST, "--sha256-borrador-aprobado", "a" * 64], **inj) == 3                # falta el motivo
    assert "motivo" in capsys.readouterr().err
    assert not (repo / "results").exists()


def test_m12_la_excepcion_con_hash_mal_formado_no_habilita_el_protocolo_no_versionado(tmp_path):
    repo = crear_repo(tmp_path)
    cfg = config_integridad(repo)
    run_git(repo, "rm", "-q", "--cached", PROTO)
    run_git(repo, "commit", "-q", "-m", "sin protocolo")
    malo = {"motivo": "m", "sha256_bytes_archivo_borrador": "A" * 64}
    assert "protocolo_no_versionado" in codigos(verificar(repo, tmp_path, modo="referencia", cfg=cfg, excepcion_protocolo=malo))
    bueno = {"motivo": "m", "sha256_bytes_archivo_borrador": "a" * 64}
    assert "protocolo_no_versionado" not in codigos(verificar(repo, tmp_path, modo="referencia", cfg=cfg, excepcion_protocolo=bueno))


# ---- M-6: los JSON que deben conservarse no se sobrescriben
def test_m6_escribir_json_nuevo_y_csv_no_sobrescriben_y_conservan_el_contenido(tmp_path):
    ruta = tmp_path / "run_config.json"
    C.escribir_json_nuevo(ruta, {"a": 1})
    antes = ruta.read_bytes()
    with pytest.raises(C.GuardError, match="K5"):
        C.escribir_json_nuevo(ruta, {"a": 2})
    assert ruta.read_bytes() == antes
    csv_ruta = tmp_path / "t.csv"
    C.escribir_csv_nuevo(csv_ruta, [{"x": 1}], ["x"])
    antes_csv = csv_ruta.read_bytes()
    with pytest.raises(C.GuardError, match="K5"):
        C.escribir_csv_nuevo(csv_ruta, [{"x": 2}], ["x"])
    assert csv_ruta.read_bytes() == antes_csv


def test_m6_los_artefactos_json_de_una_corrida_no_se_pueden_reescribir(banco):
    d = banco.raiz / "resnet50__s42__f0" / "intento_01"
    for nombre in ("run_config.json", "historial.json", "intento_iniciado.json"):
        antes = (d / nombre).read_bytes()
        with pytest.raises(C.GuardError, match="K5"):
            C.escribir_json_nuevo(d / nombre, {"sobrescrito": True})
        assert (d / nombre).read_bytes() == antes


def test_m6_el_reporte_no_sobrescribe_sus_json_ni_sus_tablas(banco, tmp_path):
    RP.generar_reporte(banco.root, banco.config, banco.part, banco.root / "results", tmp_path / "r", verificar=verificacion_falsa(),
                       n_remuestreos=20, con_figuras=False)
    for nombre in ("reporte_metadatos.json", "bootstrap_resumen.json", "T2_arquitecturas.csv"):
        antes = (tmp_path / "r" / nombre).read_bytes()
        with pytest.raises(C.GuardError):
            RP.generar_reporte(banco.root, banco.config, banco.part, banco.root / "results", tmp_path / "r",
                               verificar=verificacion_falsa(), n_remuestreos=20, con_figuras=False)
        assert (tmp_path / "r" / nombre).read_bytes() == antes


# ---- M-1: 820 = 782 + 38 son invariantes BLOQUEANTES de la partición fija phash_le8_v1 (decisión del autor)
def test_m1_los_recuentos_782_38_bloquean_para_la_particion_fija(tmp_path, config):
    assert config["particion"]["verificaciones_bloqueantes"]["n_lesion_no_maligna"] == 782
    assert config["particion"]["verificaciones_bloqueantes"]["n_lesion_maligna"] == 38
    filas = filas_sinteticas()
    pos = next(r for r in filas if r["label_v2"] == "1")
    pos.update(original_class="0", label_v2="0")                                               # 820 filas, pero 783 / 37
    with pytest.raises(C.GuardError, match="K2.*783.*37"):
        cargar(tmp_path / "a", copy.deepcopy(config), filas=filas)
    filas = filas_sinteticas()
    neg = next(r for r in filas if r["label_v2"] == "0")
    neg.update(original_class="7", label_v2="1")                                               # 781 / 39
    with pytest.raises(C.GuardError, match="K2.*781.*39"):
        cargar(tmp_path / "b", copy.deepcopy(config), filas=filas)
    assert len(P.cargar_particion(tmp_path / "c", _con_csv(tmp_path / "c", config)).filas) == 820                   # los valores correctos pasan


def _con_csv(root, config):
    cfg = copy.deepcopy(config)
    escribir_particion(root, cfg)
    return cfg


@pytest.mark.parametrize("campo, ajenos", [
    ("conformidad_asesor", ["CONFIRMADO:AUD-001", "CONFIRMADO:AUT-001", "CONFIRMADO:D-001"]),
    ("auditoria_independiente", ["CONFIRMADO:ASE-001", "CONFIRMADO:AUT-001", "CONFIRMADO:D-001"]),
    ("autorizacion_ejecucion", ["CONFIRMADO:ASE-001", "CONFIRMADO:AUD-001", "CONFIRMADO:DD-001"]),
])
def test_i3_ninguna_atestacion_acepta_el_prefijo_de_otra(campo, ajenos):
    for ajeno in ajenos:
        with pytest.raises(C.GuardError, match=campo):
            MO.exigir_atestaciones("completa", {**FORMATO_OK, campo: ajeno})


# ============================================================================ 12. N-1 y N-2 de la reauditoría limitada
# ---- N-2: los bloqueos de la verificación FINAL no pueden ignorarse (mismos hashes, solo bloqueos)
def test_n2_bloqueos_en_la_verificacion_final_invalidan_el_reporte_aunque_los_hashes_coincidan(banco, tmp_path):
    kw = dict(n_remuestreos=20, con_figuras=False)
    bueno = RP.generar_reporte(banco.root, banco.config, banco.part, banco.root / "results", tmp_path / "ok",
                               verificar=verificacion_falsa(), **kw)
    assert bueno["invalida_integridad"] is False and bueno["reporte_aceptado"] is True
    meta = RP.generar_reporte(banco.root, banco.config, banco.part, banco.root / "results", tmp_path / "bloq",
                              verificar=verificacion_falsa(bloqueo_al_final=True), **kw)
    assert meta["diferencias_inicio_fin"] == []                                     # las instantáneas son iguales: solo hay bloqueos
    assert meta["invalida_integridad"] is True and meta["reporte_aceptado"] is False
    en_disco = json.loads((tmp_path / "bloq" / "reporte_metadatos.json").read_text(encoding="utf-8"))
    assert en_disco["invalida_integridad"] is True and en_disco["reporte_aceptado"] is False
    meta = RP.generar_reporte(banco.root, banco.config, banco.part, banco.root / "results", tmp_path / "dif",
                              verificar=verificacion_falsa(cambia_al_final=True), **kw)
    assert meta["invalida_integridad"] is True and meta["reporte_aceptado"] is False     # y también cuando solo cambian los hashes


def test_n1_main_devuelve_codigo_distinto_de_cero_y_avisa_si_la_integridad_final_es_invalida(banco, tmp_path, monkeypatch, capsys):
    def falso_generar(*a, **k):
        return {"figuras": [], "nota_figuras": None, "reporte_aceptado": False, "invalida_integridad": True,
                "diferencias_inicio_fin": [], "git_commit_head": "h" * 40}
    monkeypatch.setattr(RP, "generar_reporte", falso_generar)
    monkeypatch.setattr(RP, "cargar_particion", lambda *a, **k: banco.part)
    monkeypatch.setattr(RP, "cargar_config", lambda *a, **k: banco.config)
    rc = RP.main(["--salida", "results/r4_v2/phash_le8_v1/g2_baseline_v1/reporte_x", "--sin-figuras"])
    assert rc == RP.CODIGO_SALIDA_INTEGRIDAD_INVALIDA and rc not in (0, RP.CODIGO_SALIDA_GUARDA)
    assert "REPORTE NO ACEPTADO" in capsys.readouterr().err
    monkeypatch.setattr(RP, "generar_reporte", lambda *a, **k: {**falso_generar(), "reporte_aceptado": True, "invalida_integridad": False})
    assert RP.main(["--salida", "results/r4_v2/phash_le8_v1/g2_baseline_v1/reporte_y", "--sin-figuras"]) == 0


def test_n1_otros_main_ya_devuelven_codigo_distinto_de_cero_ante_integridad_invalida(banco, tmp_path):
    """ejecutar_g2.main: una corrida `invalida_integridad` o un `referencia` inválido NO devuelven 0 (revisado; sin cambio de código)."""
    texto = (SRC_R4 / "ejecutar_g2.py").read_text(encoding="utf-8")
    assert 'return 0 if r["estado"] == "completada" else 2' in texto                    # referencia
    assert 'if r["estado"] != "completada":' in texto and "return 2" in texto            # humo/completa (incluye invalida_integridad)
    r = lanzar_corrida(banco.root, banco.config, banco.part, tmp_path / "res", "resnet50", 42, 0,
                       verificar=verificacion_falsa(bloqueo_al_final=True))
    assert r["estado"] == "invalida_integridad" and r["estado"] != "completada"


# ---- N-1 con SUBPROCESOS reales sobre un repositorio Git temporal que contiene una copia del código verificado
WRAPPER = '''\
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.r4_v2 import integridad as I
from src.r4_v2 import reporte as RP

modo = os.environ.get("R4_FALLO_FINAL", "")
real = I.verificar_integridad
llamadas = {"n": 0}


def con_fallo_final(*a, **k):
    res = real(*a, **k)
    llamadas["n"] += 1
    if llamadas["n"] >= 2 and modo == "bloqueo":
        res["bloqueos"] = res["bloqueos"] + [{"codigo": "B_contenido_difiere", "ruta": "src/r4_v2/x.py", "detalle": "simulado"}]
    if llamadas["n"] >= 2 and modo == "hash":
        res["archivos"][sorted(res["archivos"])[0]]["sha256_contenido_blob"] = "0" * 64
    return res


I.verificar_integridad = con_fallo_final
sys.exit(RP.main(sys.argv[1:]))
'''


@pytest.fixture(scope="module")
def repo_de_reporte(tmp_path_factory, config_base):
    """Copia del código real en un repositorio temporal con las 45 corridas sintéticas ya hechas (mismo árbol que K6 verifica)."""
    tmp = tmp_path_factory.mktemp("rr")
    repo = tmp / "r"
    repo.mkdir()
    run_git(repo, "init", "-q")
    for k, v in (("user.name", "t"), ("user.email", "t@example.org"), ("core.autocrlf", "false"), ("commit.gpgsign", "false")):
        run_git(repo, "config", k, v)
    for rel in LISTA_REAL:
        if rel.endswith("/"):
            continue
        blob = subprocess.run(["git", "-C", str(REPO), "cat-file", "blob", f"HEAD:{rel}"], capture_output=True)
        if blob.returncode != 0:
            pytest.skip("el clon local no tiene el protocolo/los módulos históricos en HEAD")
        escribir(repo, rel, blob.stdout)
    yaml_head = subprocess.run(["git", "-C", str(REPO), "cat-file", "blob", f"HEAD:{C.CONFIG_REL}"], capture_output=True)
    cfg = yaml.safe_load(yaml_head.stdout.decode("utf-8"))
    for fuente in sorted(SRC_R4.glob("*.py")):
        escribir(repo, f"src/r4_v2/{fuente.name}", fuente.read_bytes().replace(b"\r\n", b"\n"))
    escribir_particion(repo, cfg)                                                           # CSV sintético; el YAML copia su hash
    escribir(repo, C.CONFIG_REL, yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    escribir(repo, "app/lanzar_reporte.py", WRAPPER)
    commit_all(repo, "copia verificada")
    config = C.cargar_config(repo / C.CONFIG_REL)
    part = P.cargar_particion(repo, config)
    raiz = repo / "results" / "r4_v2" / "phash_le8_v1" / "g2_baseline_v1" / "completa"
    for a in config["diseno"]["arquitecturas"]:
        for s in config["diseno"]["semillas"]:
            for k in config["diseno"]["folds"]:
                assert lanzar_corrida(repo, config, part, raiz, a, s, k)["estado"] == "completada"
    return types.SimpleNamespace(repo=repo, tmp=tmp, config=config)


def lanzar_reporte(rr, nombre, fallo=""):
    import tempfile
    prefijo = tempfile.mkdtemp(prefix="pfx_", dir=rr.tmp)                                       # nuevo y vacío en cada lanzamiento
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON") or k == "PYTHONPATH"}
    env["R4_FALLO_FINAL"] = fallo
    destino = f"results/r4_v2/phash_le8_v1/g2_baseline_v1/reporte_{nombre}"
    r = subprocess.run([sys.executable, "-B", "-X", f"pycache_prefix={prefijo}", str(rr.repo / "app" / "lanzar_reporte.py"),
                        "--salida", destino, "--sin-figuras"], cwd=str(rr.repo), capture_output=True, env=env)
    return r.returncode, r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace"), rr.repo / destino


def test_n1_subproceso_real_integridad_valida_devuelve_cero_y_reporte_aceptado(repo_de_reporte):
    rc, out, err, destino = lanzar_reporte(repo_de_reporte, "ok")
    assert rc == 0, err
    meta = json.loads((destino / "reporte_metadatos.json").read_text(encoding="utf-8"))
    assert meta["reporte_aceptado"] is True and meta["invalida_integridad"] is False and len(meta["git_commit_head"]) == 40
    assert meta["integridad"]["src/r4_v2/reporte.py"]["sha256_contenido_blob"]                     # vocabulario único, código verificado


@pytest.mark.parametrize("fallo", ["bloqueo", "hash"])
def test_n1_subproceso_real_integridad_final_invalida_devuelve_distinto_de_cero_y_no_acepta_el_reporte(repo_de_reporte, fallo):
    rc, out, err, destino = lanzar_reporte(repo_de_reporte, fallo, fallo)
    assert rc == RP.CODIGO_SALIDA_INTEGRIDAD_INVALIDA == 4 and rc != 0
    assert "REPORTE NO ACEPTADO" in err
    meta = json.loads((destino / "reporte_metadatos.json").read_text(encoding="utf-8"))
    assert meta["reporte_aceptado"] is False and meta["invalida_integridad"] is True
    assert json.loads(out.strip().splitlines()[-1])["reporte_aceptado"] is False


def test_n1_subproceso_real_guarda_distinta_usa_otro_codigo_y_no_genera_nada(repo_de_reporte):
    rc, _, err, destino = lanzar_reporte(repo_de_reporte, "k5")
    assert rc == 0, err
    rc, out, err, _ = lanzar_reporte(repo_de_reporte, "k5")                                      # el destino ya existe: K5
    assert rc == RP.CODIGO_SALIDA_GUARDA == 3 and "K5" in err
    assert rc != RP.CODIGO_SALIDA_INTEGRIDAD_INVALIDA


# ============================================================================ 13. módulos virtuales de PyTorch (falso positivo de K6 en Falcon)
# Diagnóstico real (Falcon, Python 3.10.12, torch 2.5.1+cu121): torch.classes.__file__ == "_classes.py" y torch.ops.__file__ == "_ops.py",
# ambos con __spec__.origin None. Causa: `origen_de_modulo` cae en `__file__`; como es relativo, `verificar_integridad` lo resolvía contra
# el directorio actual (la raíz del repositorio) y lo trataba como un archivo del repo: `ausente` y `no_rastreado`.
def modulo_virtual(nombre, archivo, spec="ninguno"):
    m = types.ModuleType(nombre)
    m.__file__ = archivo
    m.__spec__ = None if spec == "ninguno" else types.SimpleNamespace(origin=spec, loader=None, cached=None)
    return m


def entorno_virtuales(repo, tmp_path, monkeypatch, *modulos, **kw):
    """Entorno cuyos módulos se insertan de verdad en sys.modules (con limpieza segura) y se leen con EntornoPython.actual()."""
    for m in modulos:
        monkeypatch.setitem(sys.modules, m.__name__, m)
    actual = I.EntornoPython.actual()
    assert all(actual.modules[m.__name__] is m for m in modulos)
    return entorno_ok(repo, tmp_path, modulos=[actual.modules[m.__name__] for m in modulos], **kw)


def torch_virtuales():
    return modulo_virtual("torch.classes", "_classes.py"), modulo_virtual("torch.ops", "_ops.py", spec=None)


def test_v1_torch_classes_y_ops_no_son_archivos_del_repo_y_quedan_registrados(tmp_path, monkeypatch):
    repo = crear_repo(tmp_path)
    monkeypatch.chdir(repo)                                                                   # como en Falcon: cwd = raíz del repositorio
    ent = entorno_virtuales(repo, tmp_path, monkeypatch, *torch_virtuales())
    res = verificar(repo, tmp_path, entorno=ent)
    assert res["ok"], res["bloqueos"]
    assert not {"ausente", "no_rastreado"} & set(codigos(res))
    assert res["modulos_virtuales_tolerados"] == [
        {"modulo": "torch.classes", "__file__": "_classes.py", "motivo": I.MODULOS_VIRTUALES_TOLERADOS["torch.classes"]},
        {"modulo": "torch.ops", "__file__": "_ops.py", "motivo": I.MODULOS_VIRTUALES_TOLERADOS["torch.ops"]}]
    assert "_classes.py" not in res["archivos"] and "_ops.py" not in res["archivos"]         # no se hashean como archivos del repo
    assert not {"torch.classes", "torch.ops"} & set(res["modulos_en_repo"])
    for reg in res["modulos_virtuales_tolerados"]:
        assert set(reg) == {"modulo", "__file__", "motivo"} and "sha256" not in json.dumps(reg)


def test_v1_la_causa_raiz_era_resolver_el_file_relativo_contra_el_cwd(tmp_path, monkeypatch):
    repo = crear_repo(tmp_path)
    monkeypatch.chdir(repo)
    m = modulo_virtual("torch.foo", "_classes.py")                                          # mismo __file__, nombre fuera de la lista
    assert I.origen_de_modulo(m) == Path("_classes.py")                                      # origen relativo sin directorio
    res = verificar(repo, tmp_path, entorno=entorno_virtuales(repo, tmp_path, monkeypatch, m))
    assert {"ausente", "no_rastreado"} <= set(codigos(res))                                  # el falso positivo original, ahora solo fuera de la lista
    assert res["modulos_virtuales_tolerados"] == []


def test_v1_sin_spec_y_con_spec_de_origen_nulo_se_toleran_igual(tmp_path, monkeypatch):
    repo = crear_repo(tmp_path)
    monkeypatch.chdir(repo)
    for spec in ("ninguno", None):
        m = modulo_virtual("torch.ops", "_ops.py", spec=spec)
        res = verificar(repo, tmp_path, entorno=entorno_virtuales(repo, tmp_path, monkeypatch, m))
        assert res["ok"] and [v["modulo"] for v in res["modulos_virtuales_tolerados"]] == ["torch.ops"]


@pytest.mark.parametrize("nombre, archivo", [
    ("src.x", "x.py"), ("src", "x.py"), ("torch.foo", "foo.py"), ("numpy.bar", "bar.py"), ("torch", "torch.py"),
    ("torch.classes.sub", "_classes.py"), ("torch._ops", "_ops.py"), ("yaml.x", "_ops.py"), ("sklearn.y", "_classes.py")])
def test_v2_modulos_fuera_de_la_lista_con_file_relativo_sin_origen_siguen_bloqueando(tmp_path, monkeypatch, nombre, archivo):
    repo = crear_repo(tmp_path)
    monkeypatch.chdir(repo)
    res = verificar(repo, tmp_path, entorno=entorno_virtuales(repo, tmp_path, monkeypatch, modulo_virtual(nombre, archivo)))
    assert {"ausente", "no_rastreado"} <= set(codigos(res)) and res["modulos_virtuales_tolerados"] == []


@pytest.mark.parametrize("nombre, archivo, spec", [
    ("torch.classes", "sub/_classes.py", "ninguno"),             # __file__ con directorio
    ("torch.classes", "src/_classes.py", "ninguno"),
    ("torch.ops", "pkg\\_ops.py", "ninguno"),                    # separador de Windows
    ("torch.ops", "C:_ops.py", "ninguno"),                       # unidad
    ("torch.classes", "_classes.py", "_classes.py"),             # __spec__.origin no nulo (relativo) y archivo ausente
    ("torch.ops", "_ops.py", "src/r4_v2/inexistente.py"),        # __spec__.origin no nulo y archivo ausente
    ("torch.ops", "", "ninguno"), ("torch.ops", None, "ninguno")])
def test_v3_torch_classes_ops_con_otra_estructura_no_se_toleran(tmp_path, monkeypatch, nombre, archivo, spec):
    repo = crear_repo(tmp_path)
    monkeypatch.chdir(repo)
    m = modulo_virtual(nombre, archivo, spec=spec)
    assert I.modulo_virtual_tolerado(nombre, m, repo) is None
    res = verificar(repo, tmp_path, entorno=entorno_virtuales(repo, tmp_path, monkeypatch, m))
    assert res["modulos_virtuales_tolerados"] == []
    if archivo:                                                                              # dentro del repo y ausente: bloquea
        assert "ausente" in codigos(res) or "no_rastreado" in codigos(res) or archivo in ("C:_ops.py", "pkg\\_ops.py")


def test_v4_si_el_repo_tiene_un_archivo_homonimo_no_se_tolera_y_se_verifica_normalmente(tmp_path, monkeypatch):
    repo = crear_repo(tmp_path, archivos={"_ops.py": "X = 1\n"})                              # homónimo en la raíz, rastreado
    monkeypatch.chdir(repo)
    ops = modulo_virtual("torch.ops", "_ops.py")
    limpio = verificar(repo, tmp_path, entorno=entorno_virtuales(repo, tmp_path, monkeypatch, ops))
    assert limpio["modulos_virtuales_tolerados"] == [], limpio["modulos_virtuales_tolerados"]
    assert limpio["ok"] and "_ops.py" in limpio["archivos"], limpio["bloqueos"]    # verificado como archivo del repo
    escribir(repo, "_ops.py", "X = 2\n")                                                      # no coincide con HEAD
    res = verificar(repo, tmp_path, entorno=entorno_virtuales(repo, tmp_path, monkeypatch, ops))
    assert "B_contenido_difiere" in codigos(res) and res["modulos_virtuales_tolerados"] == []


def test_v4_homonimo_en_el_arbol_verificado_o_sin_seguimiento_tampoco_se_tolera(tmp_path, monkeypatch):
    repo = crear_repo(tmp_path, archivos={"src/r4_v2/_classes.py": "Y = 1\n"})                # homónimo en una subcarpeta rastreada
    monkeypatch.chdir(repo)
    clases = modulo_virtual("torch.classes", "_classes.py")
    assert I.modulo_virtual_tolerado("torch.classes", clases, repo) is None
    res = verificar(repo, tmp_path, entorno=entorno_virtuales(repo, tmp_path, monkeypatch, clases))
    assert "ausente" in codigos(res)                                                         # se resuelve contra el cwd y falta: bloquea
    repo2 = crear_repo(tmp_path, "repo2")
    escribir(repo2, "tests/_ops.py", "Z = 1\n")                                               # homónimo sin seguimiento
    ops = modulo_virtual("torch.ops", "_ops.py", spec=None)
    assert I.modulo_virtual_tolerado("torch.ops", ops, repo2) is None


def test_v5_modulos_del_repo_con_ruta_relativa_legitima_siguen_protegidos(tmp_path, monkeypatch):
    repo = crear_repo(tmp_path)
    monkeypatch.chdir(repo)

    def con_origen_relativo(nombre, archivo):
        m = types.ModuleType(nombre)
        m.__file__ = archivo
        m.__spec__ = types.SimpleNamespace(origin=archivo, loader=_SourceFileLoader(), cached=None)
        return m
    legitimo = con_origen_relativo("raiz_mod", "raiz_mod.py")
    res = verificar(repo, tmp_path, entorno=entorno_virtuales(repo, tmp_path, monkeypatch, legitimo))
    assert res["ok"] and "raiz_mod.py" in res["archivos"] and "raiz_mod" in res["modulos_en_repo"]       # verificado por contenido
    escribir(repo, "raiz_mod.py", "R = 2\n")                                                  # modificado: bloquea
    assert "B_contenido_difiere" in codigos(verificar(repo, tmp_path, entorno=entorno_virtuales(repo, tmp_path, monkeypatch, legitimo)))
    escribir(repo, "suelto.py", "S = 1\n")                                                    # sin seguimiento: bloquea
    nuevo = con_origen_relativo("suelto", "suelto.py")
    assert {"no_rastreado", "sin_seguimiento"} <= set(codigos(verificar(repo, tmp_path, entorno=entorno_virtuales(repo, tmp_path, monkeypatch, nuevo))))
    # el punto de entrada con __file__ relativo (Python < 3.9 con scripts) sigue verificándose
    principal = types.ModuleType("__main__")
    principal.__file__ = "raiz_mod.py"
    principal.__spec__ = None
    res = verificar(repo, tmp_path, entorno=entorno_ok(repo, tmp_path, main=principal))
    assert "B_contenido_difiere" in codigos(res)


def test_v6_los_virtuales_tolerados_entran_en_la_comparacion_inicio_fin(tmp_path, monkeypatch):
    repo = crear_repo(tmp_path)
    monkeypatch.chdir(repo)
    ini = verificar(repo, tmp_path)
    fin = verificar(repo, tmp_path, entorno=entorno_virtuales(repo, tmp_path, monkeypatch, *torch_virtuales()))
    assert I.comparar_instantaneas(ini, fin) == ["modulos_virtuales_tolerados"]
    assert I.comparar_instantaneas(fin, fin) == []


def test_v7_run_config_y_reporte_registran_los_modulos_virtuales(tmp_path, config_base, monkeypatch, banco):
    repo = crear_repo(tmp_path)
    monkeypatch.chdir(repo)
    cfg = config_para_repo(repo, config_base)
    ent = lambda: entorno_virtuales(repo, tmp_path, monkeypatch, *torch_virtuales())
    r = MO.ejecutar_referencia(repo, cfg, verificar=lambda: I.verificar_integridad(repo, cfg, modo="referencia", entorno=ent()),
                               instanciar=instanciar_falso, resolver=resolver_falso(tmp_path), versiones=lambda: VERSIONES,
                               raiz_modo=MO.raiz_del_modo(repo, cfg, "referencia"), atestaciones={"autorizacion_ejecucion": "E"})
    assert r["estado"] == "completada"
    rc = json.loads((r["dir"] / "run_config.json").read_text(encoding="utf-8"))
    for clave in ("verificacion_inicio", "verificacion_fin"):
        assert [v["modulo"] for v in rc[clave]["modulos_virtuales_tolerados"]] == ["torch.classes", "torch.ops"]
    assert rc["invalida_integridad"] is False
    # reporte
    base = verificacion_falsa()

    def con_virtuales():
        return {**base(), "modulos_virtuales_tolerados": [{"modulo": "torch.ops", "__file__": "_ops.py", "motivo": "m"}]}
    meta = RP.generar_reporte(banco.root, banco.config, banco.part, banco.root / "results", tmp_path / "rep", verificar=con_virtuales,
                              n_remuestreos=20, con_figuras=False)
    assert meta["modulos_virtuales_tolerados"] == [{"modulo": "torch.ops", "__file__": "_ops.py", "motivo": "m"}]


def test_v8_la_lista_cerrada_solo_contiene_torch_classes_y_torch_ops():
    assert set(I.MODULOS_VIRTUALES_TOLERADOS) == {"torch.classes", "torch.ops"}               # ampliarla exige evidencia real y una prueba nueva
    assert not any(n == "src" or n.startswith("src.") for n in I.MODULOS_VIRTUALES_TOLERADOS)
    assert all(isinstance(motivo, str) and motivo for motivo in I.MODULOS_VIRTUALES_TOLERADOS.values())


def test_v4_homonimo_ignorado_por_git_en_la_raiz_o_en_el_cwd_tampoco_se_tolera(tmp_path, monkeypatch):
    repo = crear_repo(tmp_path)
    escribir(repo, ".gitignore", "__pycache__/" + chr(10) + "_ops.py" + chr(10))
    commit_all(repo, "ignora _ops.py")
    escribir(repo, "_ops.py", "X = 1" + chr(10))                                        # existe en la raíz, pero git lo ignora (no sale en ls-files)
    assert C.git_lista_z(repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard").count("_ops.py") == 0
    monkeypatch.chdir(repo)
    ops = modulo_virtual("torch.ops", "_ops.py")
    assert I.modulo_virtual_tolerado("torch.ops", ops, repo) is None
    res = verificar(repo, tmp_path, entorno=entorno_virtuales(repo, tmp_path, monkeypatch, ops))
    assert res["modulos_virtuales_tolerados"] == [] and not res["ok"]                  # se trata como archivo del repo y bloquea
    otro = tmp_path / "otro_cwd"
    otro.mkdir()
    escribir(otro, "_classes.py", "Y = 1" + chr(10))                                    # homónimo solo en el directorio actual (fuera del repo)
    monkeypatch.chdir(otro)
    assert I.modulo_virtual_tolerado("torch.classes", modulo_virtual("torch.classes", "_classes.py"), crear_repo(tmp_path, "r3")) is None


# ============================================================================ 14. enmienda 2 (ASE): la conformidad del asesor ya no bloquea humo/completa
ATEST_SIN_ASE = ["--autorizacion-ejecucion", "CONFIRMADO:D-901", "--auditoria-independiente", "CONFIRMADO:AUD-901"]


@pytest.mark.parametrize("modo", ["humo", "completa"])
def test_e2_humo_y_completa_pasan_con_aud_y_aut_validos_y_sin_ase(modo):
    ok = {"auditoria_independiente": "CONFIRMADO:AUD-001", "autorizacion_ejecucion": "CONFIRMADO:AUT-001"}
    assert MO.ATESTACIONES[modo] == ("auditoria_independiente", "autorizacion_ejecucion")
    assert MO.exigir_atestaciones(modo, ok) == ok
    assert MO.exigir_atestaciones(modo, {**ok, "conformidad_asesor": ""}) == ok               # vacía: se ignora y no se registra
    assert MO.exigir_atestaciones(modo, {**ok, "conformidad_asesor": None}) == ok
    for campo in ok:                                                                          # AUD y AUT siguen siendo obligatorias y válidas
        for malo in (None, "", "PENDIENTE", "no", "CONFIRMADO:ASE-001"):
            with pytest.raises(C.GuardError, match=campo):
                MO.exigir_atestaciones(modo, {**ok, campo: malo})
        with pytest.raises(C.GuardError, match=campo):
            MO.exigir_atestaciones(modo, {k: v for k, v in ok.items() if k != campo})


@pytest.mark.parametrize("modo", ["referencia", "humo", "completa"])
def test_e2_ase_opcional_invalida_se_rechaza_y_valida_se_registra_sin_habilitar_nada(modo):
    ok = {"auditoria_independiente": "CONFIRMADO:AUD-001", "autorizacion_ejecucion": "CONFIRMADO:AUT-001"}
    for malo in ("PENDIENTE", "pendiente", "no", " ", "x", "CONFIRMADO:ASE-1", "CONFIRMADO:AUD-001", "confirmado:ASE-001", 1, True):
        with pytest.raises(C.GuardError, match="conformidad_asesor"):
            MO.exigir_atestaciones(modo, {**ok, "conformidad_asesor": malo})
    assert MO.exigir_atestaciones(modo, {**ok, "conformidad_asesor": "CONFIRMADO:ASE-007"})["conformidad_asesor"] == "CONFIRMADO:ASE-007"
    for falta in ok:                                                                          # una ASE válida NO sustituye a AUD ni a AUT
        with pytest.raises(C.GuardError, match=falta):
            MO.exigir_atestaciones(modo, {"conformidad_asesor": "CONFIRMADO:ASE-007", **{k: v for k, v in ok.items() if k != falta}})


def test_e2_referencia_conserva_sus_atestaciones():
    assert MO.ATESTACIONES["referencia"] == ("auditoria_independiente", "autorizacion_ejecucion")
    assert MO.ATESTACIONES_OPCIONALES == ("conformidad_asesor",)


def test_e2_el_ejecutor_lanza_humo_sin_conformidad_y_run_config_dice_pendiente(tmp_path, config_base, capsys):
    from src.r4_v2 import ejecutar_g2 as EJ
    repo, cfg = banco_ejecutor(tmp_path, config_base)
    inj = inyecciones(repo, tmp_path, entrenar=entrenador_sintetico())
    assert EJ.main(["--modo", "referencia", *ATEST], **inj) == 0
    escribir(repo, W.REFERENCIA_REL, next((repo / "results").rglob("*.candidato.json")).read_bytes())
    commit_all(repo, "referencia verificada por el autor")
    raiz = repo / "results/r4_v2/phash_le8_v1/g2_baseline_v1"
    rc_ref = json.loads(next(raiz.glob("referencia/intento_*/run_config.json")).read_text(encoding="utf-8"))
    assert rc_ref["conformidad_academica_asesor"] == "PENDIENTE" and rc_ref["leyenda"] == "partición provisional; conformidad académica pendiente"
    assert EJ.main(["--modo", "humo", "--epocas-humo", "1", *ATEST_SIN_ASE], **inj) == 0                       # sin --conformidad-asesor
    rc = json.loads(next(raiz.glob("humo/*/intento_01/run_config.json")).read_text(encoding="utf-8"))
    assert rc["estado"] == "completada" and rc["conformidad_academica_asesor"] == "PENDIENTE"
    assert rc["conformidad_academica_asesor"] == cfg["protocolo"]["conformidad_academica_asesor"]             # tomado del YAML, no inventado
    assert "conformidad_asesor" not in rc["atestaciones_del_autor"] and rc["leyenda"] == "partición provisional; conformidad académica pendiente"
    capsys.readouterr()
    assert EJ.main(["--modo", "humo", "--epocas-humo", "1", "--humo-fold", "1", *ATEST_SIN_ASE, "--conformidad-asesor", "PENDIENTE"], **inj) == 3
    assert "conformidad_asesor" in capsys.readouterr().err                                                    # una ASE inválida se rechaza


def test_e2_los_consumidores_de_run_config_no_se_rompen(banco, tmp_path):
    rc = json.loads((banco.raiz / "resnet50__s42__f0" / "intento_01" / "run_config.json").read_text(encoding="utf-8"))
    assert rc["conformidad_academica_asesor"] == "PENDIENTE"
    assert len(AG.cargar_corridas(banco.raiz, banco.config, banco.part)) == 45                                # agregacion lee run_config
    humo = tmp_path / "humo"
    lanzar_corrida(banco.root, banco.config, banco.part, humo, "resnet50", 42, 0, modo="humo", epocas_max=1)
    assert MO.exigir_humo_previo(humo, sha256_bytes_archivo_particion=banco.part.sha256_bytes_archivo,
                                 sha256_contenido_blob_config="c" * 64, sha256_contenido_blob_referencia="r" * 64).name == "intento_01"
