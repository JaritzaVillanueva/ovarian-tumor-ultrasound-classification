"""Integridad de protocolo, configuración y código REALMENTE ejecutado (K6; protocolo §4.5).

Decide por CONTENIDO frente al blob de `HEAD` ([B]); `git hash-object` ([A]) solo bloquea con bytes idénticos al blob;
`git status` es un indicador auxiliar. Cubre preparados, sin seguimiento e ignorados, la lista fija ∪ la dinámica
(`sys.modules` dentro del repositorio, punto de entrada, notebook), paquetes de espacio de nombres y los controles
efectivos de bytecode (`-B` y `-X pycache_prefix`). No lee datos ni imágenes. Solo ejecuta órdenes Git de lectura.

`verificar_integridad` devuelve un diccionario con `bloqueos` (lista; vacía = pasa). `exigir_integridad` lanza GuardError.
"""

from __future__ import annotations

import site
import sys
import sysconfig
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Mapping

from .common import (GuardError, PROTOCOLO_REL, git, git_lista_z, git_ok, git_texto, sha256_bytes, validar_sha256)

PYTHON_MINIMO = (3, 8)

# Módulos VIRTUALES de PyTorch tolerados en la lista dinámica (lista cerrada, en código; evidencia: diagnóstico real en Falcon con
# Python 3.10.12 / torch 2.5.1+cu121: `torch.classes.__file__ == "_classes.py"` y `torch.ops.__file__ == "_ops.py"`, ambos con
# `__spec__` None u `origin` None). No son archivos: son objetos `ModuleType` con un `__file__` simbólico sin directorio. Un módulo solo
# se tolera si cumple TODAS las condiciones de `modulo_virtual_tolerado`; cualquier otro nombre sigue el tratamiento normal.
MODULOS_VIRTUALES_TOLERADOS = {
    "torch.classes": "módulo virtual de PyTorch (torch._classes._Classes): __file__ simbólico '_classes.py' y sin __spec__.origin",
    "torch.ops": "módulo virtual de PyTorch (torch._ops._Ops): __file__ simbólico '_ops.py' y sin __spec__.origin",
}

# Lista fija CERRADA de §4.5.1 (i)-(v), definida EN CÓDIGO: el alcance de K6 no depende de lo que diga el YAML del árbol de trabajo.
# La lista de la configuración debe ser exactamente igual (como conjunto): cualquier diferencia, en más o en menos, bloquea.
LISTA_FIJA_CERRADA = (
    "src/r4_v2/protocolo_g2_baseline_v1.md",
    "configs/r4_v2/",
    "src/r4_v2/",
    "src/classification/__init__.py",
    "src/classification/models.py",
    "src/classification/transforms.py",
    "src/classification/dataloader.py",
    "src/classification/evaluate.py",
    "src/classification/losses.py",
    "src/classification/utils.py",
    "src/classification/train.py",
    "src/classification/training_options.py",
    ".gitattributes",
)
EXTENSIONES_COMPILADAS = {".pyc", ".pyd", ".so", ".dll", ".dylib"}


# ---------------------------------------------------------------- entorno de Python (inyectable en pruebas)
@dataclass
class EntornoPython:
    modules: Mapping = field(default_factory=dict)
    version_info: tuple = (3, 12, 0)
    version: str = ""
    flags_dont_write_bytecode: int = 0
    dont_write_bytecode: bool = False
    pycache_prefix: str | None = None
    main: ModuleType | None = None
    sys_path: list = field(default_factory=list)

    @classmethod
    def actual(cls) -> "EntornoPython":
        return cls(modules=dict(sys.modules), version_info=tuple(sys.version_info[:3]), version=sys.version,
                   flags_dont_write_bytecode=sys.flags.dont_write_bytecode,
                   dont_write_bytecode=sys.dont_write_bytecode, pycache_prefix=sys.pycache_prefix,
                   main=sys.modules.get("__main__"), sys_path=list(sys.path))


# ---------------------------------------------------------------- utilidades de rutas y módulos
def _resolver(p) -> Path:
    return Path(p).resolve()


def _dentro(base: Path, p: Path) -> bool:
    return p == base or base in p.parents


def _relativa(raiz: Path, p) -> str | None:
    try:
        return _resolver(p).relative_to(raiz).as_posix()
    except (ValueError, OSError):
        return None


def _dirs_del_entorno() -> list:
    """Biblioteca estándar y site-packages: no se verifican por hash aunque estén dentro del repositorio (p. ej. .venv)."""
    dirs = set()
    for clave in ("stdlib", "platstdlib", "purelib", "platlib"):
        v = sysconfig.get_paths().get(clave)
        if v:
            dirs.add(_resolver(v))
    try:
        for v in [*site.getsitepackages(), site.getusersitepackages()]:
            dirs.add(_resolver(v))
    except Exception:  # pragma: no cover  (algunos entornos virtuales no exponen getsitepackages)
        pass
    return list(dirs)


def origen_de_modulo(mod) -> Path | None:
    spec = getattr(mod, "__spec__", None)
    origen = getattr(spec, "origin", None) if spec is not None else None
    if origen in (None, "built-in", "frozen", "namespace"):
        origen = getattr(mod, "__file__", None)
    if not origen or origen in ("built-in", "frozen") or (str(origen).startswith("<") and str(origen).endswith(">")):
        return None                                                  # p. ej. '<stdin>': no es un archivo
    return Path(origen)


def _hay_archivo_homonimo(root, nombre_archivo: str) -> bool:
    """True si existe (en el árbol de trabajo, en el directorio actual, en el índice o sin seguimiento) un archivo propio con ese nombre,
    en cualquier carpeta del repositorio: entonces pudo ser el realmente cargado y NO se tolera el módulo virtual."""
    if (Path(root) / nombre_archivo).exists() or (Path.cwd() / nombre_archivo).exists():
        return True
    for ruta in git_lista_z(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard"):
        if ruta.replace("\\", "/").split("/")[-1] == nombre_archivo:
            return True
    return False


def modulo_virtual_tolerado(nombre: str, mod, root) -> dict | None:
    """Criterio estructural estricto (todas las condiciones a la vez). Devuelve el registro del módulo tolerado o None.

    1. el nombre está en la lista cerrada del código (`MODULOS_VIRTUALES_TOLERADOS`) y no es `src` ni `src.*`;
    2. `__spec__` es None o su `origin` es None/ausente (no hay archivo de origen declarado por el importador);
    3. `__file__` es una cadena no vacía SIN directorio (sin separadores ni unidad: un simple nombre de archivo);
    4. no existe en el repositorio ningún archivo con ese nombre que pudiera haber sido el cargado de verdad.
    """
    if nombre not in MODULOS_VIRTUALES_TOLERADOS or nombre == "src" or nombre.startswith("src."):
        return None
    spec = getattr(mod, "__spec__", None)
    if spec is not None and getattr(spec, "origin", None) is not None:
        return None
    archivo = getattr(mod, "__file__", None)
    if not isinstance(archivo, str) or not archivo or any(c in archivo for c in ("/", "\\", ":")) or Path(archivo).name != archivo:
        return None
    if _hay_archivo_homonimo(root, archivo):
        return None
    return {"modulo": nombre, "__file__": archivo, "motivo": MODULOS_VIRTUALES_TOLERADOS[nombre]}


def _nombre_del_cargador(mod) -> str:
    spec = getattr(mod, "__spec__", None)
    loader = getattr(spec, "loader", None) if spec is not None else None
    return type(loader).__name__ if loader is not None else ""


# ---------------------------------------------------------------- control por archivo (§4.5.3)
def _normalizar_eol(b: bytes) -> bytes:
    return b.replace(b"\r\n", b"\n")


def controlar_archivo(root, rel: str, bloqueos: list) -> dict:
    """Registro de campos de §4.5 y bloqueos [B] y [A] de un archivo rastreado. `rel` en formato POSIX."""
    ruta = Path(root) / rel
    reg = {"git_blob_id": None, "sha256_contenido_blob": None, "sha256_bytes_archivo": None,
           "eol_trabajo_difiere": None, "hash_object_coincide": None, "git_ultimo_commit": None}
    if not ruta.is_file():
        bloqueos.append({"codigo": "ausente", "ruta": rel, "detalle": "el archivo no existe en el árbol de trabajo"})
        return reg
    trabajo = ruta.read_bytes()
    r = git(root, "cat-file", "blob", f"HEAD:{rel}")
    if r.returncode != 0:
        bloqueos.append({"codigo": "no_rastreado", "ruta": rel, "detalle": "no existe en HEAD (no versionado)"})
        reg["sha256_bytes_archivo"] = sha256_bytes(trabajo)
        return reg
    blob = r.stdout
    blob_id = git_texto(root, "rev-parse", f"HEAD:{rel}", obligatorio=True)
    hash_obj = git_texto(root, "hash-object", "--", rel, obligatorio=True)
    reg["git_blob_id"] = blob_id
    reg["sha256_contenido_blob"] = sha256_bytes(blob)
    reg["sha256_bytes_archivo"] = sha256_bytes(trabajo)
    reg["eol_trabajo_difiere"] = reg["sha256_bytes_archivo"] != reg["sha256_contenido_blob"]
    reg["hash_object_coincide"] = hash_obj == blob_id
    reg["git_ultimo_commit"] = git_texto(root, "log", "-n1", "--format=%H", "--", rel, obligatorio=True)
    # [B] decide: única diferencia tolerada, CRLF <-> LF
    if sha256_bytes(_normalizar_eol(trabajo)) != sha256_bytes(_normalizar_eol(blob)):
        bloqueos.append({"codigo": "B_contenido_difiere", "ruta": rel,
                         "detalle": "el contenido (con CRLF->LF) difiere del blob de HEAD"})
    # [A] solo bloquea con bytes idénticos al blob que Git ve distintos (filtros o atributos anómalos)
    if not reg["hash_object_coincide"] and not reg["eol_trabajo_difiere"]:
        bloqueos.append({"codigo": "A_hash_object_distinto", "ruta": rel,
                         "detalle": "bytes idénticos al blob pero git hash-object difiere (filtros/atributos anómalos)"})
    return reg


# ---------------------------------------------------------------- verificación principal
def _expandir_fijas(root, entradas) -> tuple:
    """(archivos, alcance): archivos rastreados de la lista fija (HEAD ∪ índice) y rutas para los controles masivos."""
    archivos, alcance = set(), []
    for e in entradas:
        es_directorio = e.endswith("/") or (Path(root) / e).is_dir()
        e = e.rstrip("/")
        alcance.append(e)
        if es_directorio:
            archivos.update(git_lista_z(root, "ls-tree", "-r", "--name-only", "-z", "HEAD", "--", e))
            archivos.update(git_lista_z(root, "ls-files", "-z", "--", e))
            # los archivos del árbol de trabajo que no estén en HEAD/índice los detectan los controles (d) y (e)
        else:
            archivos.add(e)
    return archivos, alcance


def _normalizar_ruta(e: str) -> str:
    return e.replace("\\", "/")


def verificar_integridad(root, config, *, modo: str, entorno: EntornoPython | None = None, notebook=None,
                         excepcion_protocolo: dict | None = None, lista_fija=None) -> dict:
    """Aplica §4.5 y devuelve el registro (`run_config`) con la lista de `bloqueos`.

    `lista_fija` es la lista cerrada del CÓDIGO (None = `LISTA_FIJA_CERRADA`; solo las pruebas la sustituyen). La de la
    configuración debe coincidir exactamente con ella; si no, se bloquea (`lista_fija_discrepa`). Un error de Git lanza GuardError.
    """
    entorno = entorno or EntornoPython.actual()
    root = Path(root).resolve()
    bloqueos: list = []

    def bloquear(codigo, ruta=None, detalle=""):
        bloqueos.append({"codigo": codigo, "ruta": ruta, "detalle": detalle})

    res = {"modo": modo, "fecha": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "bloqueos": bloqueos}
    head = git_texto(root, "rev-parse", "HEAD")
    top = git_texto(root, "rev-parse", "--show-toplevel")
    if head is None or top is None:
        bloquear("sin_git", None, "no es un repositorio Git con HEAD")
        res["ok"] = False
        return res
    if _resolver(top) != root:
        bloquear("raiz_git_distinta", None, "la raíz indicada no es la raíz del repositorio Git")
    res["git_commit_head"] = head
    res["rama"] = git_texto(root, "rev-parse", "--abbrev-ref", "HEAD")

    # ---- protocolo versionado y relación con la configuración (§4.5.4)
    en_head = git(root, "cat-file", "-e", f"HEAD:{PROTOCOLO_REL}").returncode == 0
    res["protocolo_versionado"] = en_head
    permitida = (not en_head) and modo == "referencia" and _excepcion_valida(excepcion_protocolo)
    if not en_head and not permitida:
        bloquear("protocolo_no_versionado", PROTOCOLO_REL,
                 "humo/completa exigen el protocolo versionado; referencia solo con la excepción documentada del autor")
    res["excepcion_protocolo"] = excepcion_protocolo if permitida else None

    # ---- conjunto verificado: lista fija ∪ dinámica
    lista_fija = LISTA_FIJA_CERRADA if lista_fija is None else lista_fija
    en_config = {_normalizar_ruta(e) for e in config["integridad"]["rutas_verificadas_lista_fija"]}
    en_codigo = {_normalizar_ruta(e) for e in lista_fija}
    if en_config != en_codigo:
        bloquear("lista_fija_discrepa", None,
                 f"la lista del YAML difiere de la lista cerrada del código: faltan en el YAML {sorted(en_codigo - en_config)}; "
                 f"sobran en el YAML {sorted(en_config - en_codigo)}")
    fijas, alcance = _expandir_fijas(root, list(lista_fija))
    if permitida:
        fijas.discard(PROTOCOLO_REL)
        alcance = [a for a in alcance if a != PROTOCOLO_REL]
    dinamicas, paquetes, modulos_repo, virtuales = set(), {}, [], []
    entorno_dirs = _dirs_del_entorno()
    modulos = dict(entorno.modules)
    if entorno.main is not None:
        modulos.setdefault("__main__", entorno.main)
    for nombre, mod in sorted(modulos.items()):
        if mod is None:
            continue
        virtual = modulo_virtual_tolerado(nombre, mod, root)
        if virtual is not None:                                                      # no es un archivo del repositorio: no se hashea
            virtuales.append(virtual)
            continue
        origen = origen_de_modulo(mod)
        if origen is None:
            ruta_ns = getattr(mod, "__path__", None)
            if ruta_ns is not None and getattr(mod, "__file__", None) is None:        # paquete de espacio de nombres
                entradas = [str(p) for p in list(ruta_ns)]
                dentro = [e for e in entradas if _relativa(root, e) is not None]
                if dentro:
                    if len(dentro) != len(entradas):
                        bloquear("namespace_fuera_del_repo", nombre, "una entrada de __path__ está fuera del repositorio")
                    paquetes[nombre] = sorted(_relativa(root, e) for e in dentro)
            continue
        if not origen.is_absolute():
            origen = Path.cwd() / origen
        rel = _relativa(root, origen)
        if rel is None or any(_dentro(d, _resolver(origen)) for d in entorno_dirs):
            continue                                                                 # fuera del repositorio o del entorno
        if origen.suffix.lower() in EXTENSIONES_COMPILADAS or _nombre_del_cargador(mod) == "SourcelessFileLoader":
            bloquear("modulo_sin_fuente_rastreable", rel, f"módulo {nombre} sin archivo .py (solo .pyc o extensión compilada)")
            continue
        dinamicas.add(rel)
        modulos_repo.append(nombre)
    nb_rel = None
    if notebook is not None:
        nb_rel = _relativa(root, Path(notebook) if Path(notebook).is_absolute() else root / notebook)
        if nb_rel is None:
            bloquear("notebook_fuera_del_repo", str(notebook), "el notebook orquestador debe estar dentro del repositorio")
        else:
            dinamicas.add(nb_rel)
    union = sorted(fijas | dinamicas)
    res["modulos_en_repo"] = sorted(modulos_repo)
    res["modulos_virtuales_tolerados"] = sorted(virtuales, key=lambda v: v["modulo"])
    res["paquetes_espacio_de_nombres"] = paquetes
    res["sys_path_en_repo"] = sorted(r for r in (_relativa(root, p) for p in entorno.sys_path if p) if r is not None)

    # ---- controles por archivo
    archivos = {rel: controlar_archivo(root, rel, bloqueos) for rel in union}
    res["archivos"] = archivos
    atributos = {}
    salida_attr = git_ok(root, "check-attr", "text", "--", *union) if union else b""
    for linea in salida_attr.decode("utf-8", "replace").splitlines():
        partes = linea.rsplit(": ", 2)
        if len(partes) == 3:
            atributos[partes[0]] = partes[2].strip()          # 'unset' (= -text), 'unspecified', 'set'...
    for rel, reg in archivos.items():
        reg["atributo_text"] = atributos.get(rel)

    # ---- controles masivos: índice, rastreo, sin seguimiento, ignorados
    ambito = sorted(set(alcance) | dinamicas)
    if ambito:
        indice = set(git_lista_z(root, "ls-files", "-z", "--", *ambito))
        ya = {b["ruta"] for b in bloqueos if b["codigo"] == "no_rastreado"}
        for rel in union:
            if rel not in indice and rel not in ya:
                bloquear("no_rastreado", rel, "el archivo no está en el índice (git ls-files --error-unmatch)")
        for rel in git_lista_z(root, "diff", "--cached", "--name-only", "-z", "HEAD", "--", *ambito):
            bloquear("preparado", rel, "hay cambios preparados (índice distinto de HEAD)")
        for rel in git_lista_z(root, "ls-files", "--others", "--exclude-standard", "-z", "--", *ambito):
            if permitida and rel == PROTOCOLO_REL:
                continue                                              # excepción documentada del autor (solo modo referencia)
            bloquear("sin_seguimiento", rel, "archivo nuevo sin seguimiento dentro de las rutas verificadas")
        for rel in git_lista_z(root, "ls-files", "--others", "--ignored", "--exclude-standard", "-z", "--", *ambito):
            if "__pycache__" not in rel.replace("\\", "/").split("/"):
                bloquear("ignorado", rel, "archivo ignorado dentro de las rutas verificadas")
        # indicador auxiliar (no decide): ' M' con contenido sin cambio real se registra sin bloquear
        estado = git_ok(root, "status", "--porcelain", "--untracked-files=all", "-z", "--", *ambito)
        lineas = [s.decode("utf-8", "replace") for s in estado.split(bytes([0])) if s]
        res["status_aux"] = lineas
        con_cambio = {b["ruta"] for b in bloqueos if b["codigo"] == "B_contenido_difiere"}
        res["status_m_sin_cambio_de_contenido"] = sorted(
            s[3:] for s in lineas if s[:2].strip() == "M" and s[3:] in archivos and s[3:] not in con_cambio)

    # ---- protocolo frente a la configuración registrada (§4.5.4)
    if en_head:
        reg_p = archivos.get(PROTOCOLO_REL) or controlar_archivo(root, PROTOCOLO_REL, bloqueos)
        res["protocolo"] = {"git_blob_id": reg_p["git_blob_id"], "sha256_contenido_blob": reg_p["sha256_contenido_blob"]}
        esperado = config["protocolo"]["sha256_contenido_blob"]
        if reg_p["sha256_contenido_blob"] != esperado:
            bloquear("protocolo_hash_distinto", PROTOCOLO_REL, "sha256_contenido_blob del protocolo difiere del de la configuración")
        if reg_p["git_blob_id"] != config["protocolo"]["git_blob_id"]:
            bloquear("protocolo_blob_id_distinto", PROTOCOLO_REL, "git_blob_id del protocolo difiere del de la configuración")
    res["notebook"] = archivos.get(nb_rel) if nb_rel else None

    # ---- bytecode y punto de entrada (§4.5.6)
    _controles_bytecode(root, entorno, modulos, modulos_repo, bloquear, res)
    res["ok"] = not bloqueos
    return res


def _controles_bytecode(root, entorno, modulos, modulos_repo, bloquear, res) -> None:
    if tuple(entorno.version_info) < PYTHON_MINIMO:
        bloquear("python_antiguo", None, f"Python {tuple(entorno.version_info)} < {PYTHON_MINIMO}: sin -X pycache_prefix")
    if entorno.flags_dont_write_bytecode != 1 or entorno.dont_write_bytecode is not True:
        bloquear("bytecode_se_escribe", None, "se exige -B (sys.flags.dont_write_bytecode == 1)")
    prefijo = entorno.pycache_prefix
    prefijo_ruta = None
    if prefijo is None:
        bloquear("sin_pycache_prefix", None, "se exige -X pycache_prefix=<directorio nuevo y vacío fuera del repositorio>")
    else:
        prefijo_ruta = _resolver(prefijo)
        if not prefijo_ruta.is_dir():
            bloquear("pycache_prefix_inexistente", None, "el directorio del prefijo no existe")
        elif _dentro(root, prefijo_ruta) or _dentro(prefijo_ruta, root):
            bloquear("pycache_prefix_en_repo", None, "el prefijo está dentro del repositorio (o lo contiene)")
        elif any(prefijo_ruta.iterdir()):
            bloquear("pycache_prefix_no_vacio", None, "el prefijo no está vacío tras importar (se consumió o escribió bytecode)")
    for nombre in modulos_repo:
        mod = modulos[nombre]
        spec = getattr(mod, "__spec__", None)
        if spec is None:
            continue                                                  # el script de entrada se ejecuta desde su fuente
        if _nombre_del_cargador(mod) != "SourceFileLoader":
            bloquear("cargador_no_fuente", nombre, f"cargador {_nombre_del_cargador(mod)!r}: se exige SourceFileLoader")
        for atributo in (getattr(spec, "cached", None), getattr(mod, "__cached__", None)):
            if atributo is None:
                continue
            c = _resolver(atributo)
            if _dentro(root, c) or prefijo_ruta is None or not _dentro(prefijo_ruta, c):
                bloquear("bytecode_en_arbol", nombre, "__spec__.cached / __cached__ no está bajo sys.pycache_prefix")
                break
    main = entorno.main
    entrada = None
    origen_main = origen_de_modulo(main) if main is not None else None
    if origen_main is None:
        bloquear("entrada_sin_archivo", "__main__", "python -c, REPL, stdin o kernel de notebook: la entrada debe ser un archivo")
    else:
        entrada = "modulo" if getattr(main, "__spec__", None) is not None else "script"
        if _relativa(root, origen_main if origen_main.is_absolute() else Path.cwd() / origen_main) is None:
            bloquear("entrada_fuera_del_repo", "__main__", "el punto de entrada debe estar dentro del repositorio")
    res["entrada"] = entrada
    res["bytecode"] = {"sys_version": entorno.version, "flags_dont_write_bytecode": entorno.flags_dont_write_bytecode,
                       "dont_write_bytecode": entorno.dont_write_bytecode,
                       "pycache_prefix": None if prefijo_ruta is None else
                       ("dentro_del_repo" if _dentro(root, prefijo_ruta) else "fuera_del_repo")}


def _excepcion_valida(exc) -> bool:
    """Excepción documentada del autor (solo `referencia`): motivo no vacío y SHA-256 del borrador bien formado (64 hex minúsculas)."""
    if not exc or not str(exc.get("motivo", "")).strip():
        return False
    try:
        validar_sha256(exc.get("sha256_bytes_archivo_borrador"), "sha256_bytes_archivo_borrador")
    except GuardError:
        return False
    return True


def exigir_integridad(res: dict) -> dict:
    if res["bloqueos"]:
        lista = "; ".join(f"{b['codigo']}({b.get('ruta')})" for b in res["bloqueos"])
        raise GuardError(f"K6: integridad no verificada: {lista}")
    return res


# ---------------------------------------------------------------- instantáneas inicio / fin (§4.5.5)
def huella(res: dict) -> dict:
    """Parte comparable de la verificación: HEAD, módulos y hashes de cada archivo."""
    return {"git_commit_head": res.get("git_commit_head"), "modulos_en_repo": res.get("modulos_en_repo"),
            "modulos_virtuales_tolerados": res.get("modulos_virtuales_tolerados", []),
            "archivos": {rel: [r["git_blob_id"], r["sha256_contenido_blob"], r["sha256_bytes_archivo"]]
                         for rel, r in res.get("archivos", {}).items()}}


def comparar_instantaneas(inicio: dict, fin: dict) -> list:
    """Diferencias entre `verificacion_inicio` y `verificacion_fin` (lista vacía = iguales)."""
    a, b = huella(inicio), huella(fin)
    dif = []
    if a["git_commit_head"] != b["git_commit_head"]:
        dif.append("git_commit_head")
    if a["modulos_en_repo"] != b["modulos_en_repo"]:
        dif.append("modulos_en_repo")
    if a["modulos_virtuales_tolerados"] != b["modulos_virtuales_tolerados"]:
        dif.append("modulos_virtuales_tolerados")
    for rel in sorted(set(a["archivos"]) | set(b["archivos"])):
        if a["archivos"].get(rel) != b["archivos"].get(rel):
            dif.append(f"archivo:{rel}")
    return dif
