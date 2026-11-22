"""Procedencia de los pesos preentrenados (K8; protocolo §4.6).

`src/classification/models.py` resuelve los pesos con el alias `DEFAULT` de torchvision, que puede cambiar entre versiones.
Aquí se registra, por arquitectura, qué enum se resolvió, su `url`, el archivo de la caché (ruta RELATIVA), su tamaño y su
`sha256_bytes_archivo`, y se compara con la referencia versionada `configs/r4_v2/referencia_pesos_preentrenados_v1.json`.
No se inventa ningún hash esperado: el primer registro verificado por el autor fija la referencia.

Lo que depende de torch/torchvision (resolución del enum, `.url`, `torch.hub.get_dir()`) se importa de forma diferida y es
inyectable (`resolver`); localmente queda [ND] hasta Falcon.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

from .common import GuardError, git, sha256_archivo
from .integridad import controlar_archivo

ESQUEMA_REFERENCIA = "referencia_pesos_preentrenados_v1"
REFERENCIA_REL = "configs/r4_v2/referencia_pesos_preentrenados_v1.json"
ARQUITECTURAS = ("resnet50", "densenet121", "convnext_tiny")
CLASE_ESPERADA = {"resnet50": "ResNet", "densenet121": "DenseNet", "convnext_tiny": "ConvNeXt"}   # type(modelo).__name__
CAMPOS_POR_ARQUITECTURA = ("enum_pesos", "url", "nombre_archivo_cache", "tamano_bytes", "sha256_bytes_archivo")
_RUTA_ABSOLUTA = re.compile(r"^([A-Za-z]:[\\/]|/|\\\\)")


def versiones_torch() -> dict:
    import torch
    import torchvision
    return {"torch_version": torch.__version__, "torchvision_version": torchvision.__version__}


def resolver_pesos_torchvision(arquitectura: str) -> dict:
    """Misma regla que el constructor (`get_model_weights(nombre).DEFAULT`); el enum se LEE en ejecución, no se asume.

    [CITA POR VERIFICAR] el atributo `.url` del enum de pesos (no figura en la página de documentación consultada por el protocolo).
    """
    import torch
    from torchvision import models
    miembro = models.get_model_weights(arquitectura).DEFAULT
    url = miembro.url
    nombre = Path(urlparse(url).path).name
    return {"enum_pesos": f"{type(miembro).__name__}.{miembro.name}", "url": url,
            "archivo_cache": Path(torch.hub.get_dir()) / "checkpoints" / nombre}


def registrar_procedencia(arquitectura: str, resolucion: dict) -> dict:
    """Registro de §4.6 a partir de la resolución. Se invoca DESPUÉS de instanciar el modelo (el constructor descarga/lee el archivo)."""
    if arquitectura not in ARQUITECTURAS:
        raise GuardError(f"K8: arquitectura desconocida {arquitectura!r}")
    for campo in ("enum_pesos", "url", "archivo_cache"):
        if not resolucion.get(campo):
            raise GuardError(f"K8: no se pudo resolver {campo} de los pesos de {arquitectura}")
    archivo = Path(resolucion["archivo_cache"])
    if not archivo.is_file():
        raise GuardError(f"K8: el archivo de pesos de {arquitectura} no existe: {archivo.name}")
    tam = archivo.stat().st_size
    if tam == 0:
        raise GuardError(f"K8: el archivo de pesos de {arquitectura} está vacío: {archivo.name}")
    try:
        sha = sha256_archivo(archivo)
    except OSError as exc:
        raise GuardError(f"K8: no se pudo leer/hashear el archivo de pesos de {arquitectura}: {exc}") from exc
    return {"enum_pesos": resolucion["enum_pesos"], "url": resolucion["url"], "nombre_archivo_cache": archivo.name,
            "ruta_cache_relativa": f"checkpoints/{archivo.name}", "tamano_bytes": tam, "sha256_bytes_archivo": sha}


def procedencia_tras_instanciar(arquitectura: str, modelo, *, pretrained: bool = True, resolver=None) -> dict:
    """K8: comprueba pretrained y arquitectura efectiva y registra la procedencia del archivo realmente cargado."""
    if not pretrained:
        raise GuardError("K8: se pidió pretrained=False; G2 exige pesos preentrenados")
    clase = type(modelo).__name__
    if clase != CLASE_ESPERADA[arquitectura]:
        raise GuardError(f"K8: la arquitectura efectiva es {clase}, no la solicitada ({arquitectura})")
    return registrar_procedencia(arquitectura, (resolver or resolver_pesos_torchvision)(arquitectura))


# ---------------------------------------------------------------- referencia (artefacto de datos)
def construir_candidato(registros: dict, versiones: dict, fecha: str) -> dict:
    """Archivo CANDIDATO (esquema v1): sin rutas absolutas ni datos personales; el autor lo verifica y versiona."""
    if set(registros) != set(ARQUITECTURAS):
        raise GuardError("K8: faltan arquitecturas en el candidato de referencia")
    return {"esquema": ESQUEMA_REFERENCIA, "modo_origen": "referencia", "fecha": fecha,
            "torch_version": versiones["torch_version"], "torchvision_version": versiones["torchvision_version"],
            "arquitecturas": {a: {c: registros[a][c] for c in CAMPOS_POR_ARQUITECTURA} for a in ARQUITECTURAS}}


def problemas_de_referencia(ref) -> list:
    """Esquema incompleto o inválido (lista de problemas; vacía = válida)."""
    if not isinstance(ref, dict):
        return ["no es un objeto JSON"]
    p = []
    if ref.get("esquema") != ESQUEMA_REFERENCIA:
        p.append("esquema distinto")
    for campo in ("torch_version", "torchvision_version", "fecha", "modo_origen"):
        if not ref.get(campo):
            p.append(f"falta {campo}")
    if ref.get("modo_origen") != "referencia":
        p.append("modo_origen distinto de 'referencia' (nunca se genera desde humo/completa)")
    arqs = ref.get("arquitecturas")
    if not isinstance(arqs, dict):
        return p + ["faltan arquitecturas"]
    for a in ARQUITECTURAS:
        reg = arqs.get(a)
        if not isinstance(reg, dict):
            p.append(f"falta la arquitectura {a}")
            continue
        for c in CAMPOS_POR_ARQUITECTURA:
            if reg.get(c) in (None, "", 0):
                p.append(f"{a}: falta {c}")
        if isinstance(reg.get("sha256_bytes_archivo"), str) and not re.fullmatch(r"[0-9a-f]{64}", reg["sha256_bytes_archivo"]):
            p.append(f"{a}: sha256_bytes_archivo inválido")
    for texto in _cadenas(ref):
        if _RUTA_ABSOLUTA.match(texto) and not texto.startswith(("http://", "https://")):
            p.append("contiene una ruta absoluta")
            break
    return p


def _cadenas(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _cadenas(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _cadenas(v)


def cargar_referencia(root) -> dict:
    """K8: la referencia debe existir, ser válida, estar rastreada y sin modificar (§4.5.3). Devuelve datos y sus hashes."""
    ruta = Path(root) / REFERENCIA_REL
    if not ruta.is_file():
        raise GuardError(f"K8: la referencia de pesos {REFERENCIA_REL} no existe; ejecuta primero el modo 'referencia' "
                         "y versiona el candidato verificado")
    bloqueos: list = []
    reg = controlar_archivo(root, REFERENCIA_REL, bloqueos)
    if git(root, "ls-files", "--error-unmatch", "--", REFERENCIA_REL).returncode != 0:
        bloqueos.append({"codigo": "no_rastreado", "ruta": REFERENCIA_REL})
    if git(root, "diff", "--cached", "--quiet", "HEAD", "--", REFERENCIA_REL).returncode != 0:
        bloqueos.append({"codigo": "preparado", "ruta": REFERENCIA_REL})
    if bloqueos:
        raise GuardError("K8: la referencia de pesos no está rastreada o está modificada: "
                         + ", ".join(sorted({b["codigo"] for b in bloqueos})))
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GuardError(f"K8: la referencia de pesos no se puede leer: {exc}") from exc
    problemas = problemas_de_referencia(datos)
    if problemas:
        raise GuardError("K8: referencia de pesos con esquema inválido o incompleto: " + "; ".join(problemas))
    return {"datos": datos, "git_blob_id": reg["git_blob_id"], "sha256_contenido_blob": reg["sha256_contenido_blob"]}


def comparar_con_referencia(arquitectura: str, registro: dict, ref: dict, versiones: dict) -> dict:
    """K8 (humo/completa): enum, url, tamaño y sha256_bytes_archivo deben coincidir. Versiones distintas solo se registran."""
    esperado = ref["datos"]["arquitecturas"][arquitectura]
    difieren = [c for c in ("enum_pesos", "url", "tamano_bytes", "sha256_bytes_archivo") if registro[c] != esperado[c]]
    if difieren:
        raise GuardError(f"K8: los pesos efectivos de {arquitectura} difieren de la referencia en {difieren}; se detiene")
    return {"version_difiere_referencia": any(
        ref["datos"][k] != versiones[k] for k in ("torch_version", "torchvision_version"))}
