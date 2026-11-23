"""Carga y uso del sistema final CONGELADO (declaración D-060) para la interfaz de R9. Solo lectura de los artefactos congelados; no modifica nada.

`cargar_sistema` verifica, antes de habilitar cualquier inferencia: el SHA-256 de la declaración (referencia externa fijada aquí), los artefactos vinculados (YAML, protocolo, manifiesto, código de
inferencia), el SHA-256 de los 15 checkpoints (y, si se pide, el reporte real D-058), la configuración de G2 y el umbral exacto. Si algo falla lanza `SistemaNoDisponible`: la interfaz pasa al modo
demostración SIN inferencia (nunca resultados simulados).

`SistemaCongelado.clasificar` aplica el preprocesamiento de G2 a UNA imagen, obtiene la probabilidad de la clase maligna de cada uno de los 15 miembros (softmax float32), las promedia (float64) y aplica
`p >= t_hat` con `SF.agregar_y_decidir` (la función congelada). No hay ninguna función de métricas ni de entrenamiento aquí.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.r4_v2.common import RAIZ_REPO, GuardError
from src.r4_v2.common import cargar_config as cargar_config_g2
from src.r7_v2 import sistema_final as SF
from src.r8_v2 import congelamiento as C
from src.r8_v2 import declaracion as DC
from src.gui.textos import ETIQUETA_BENIGNO, ETIQUETA_MALIGNO, VERSION_PROTOTIPO

SHA256_DECLARACION = "2dd79df4815b676e1bfe5c880ebb1040548bf57d21de3bbf91652842a8598cf1"       # declaración D-060, referencia externa a ella
SISTEMA_ID = "sistema_congelado_v1 (declaración D-060): DenseNet121 + A, 15 modelos, promedio simple"


class SistemaNoDisponible(RuntimeError):
    """El sistema congelado no se pudo cargar o verificar: no hay inferencia (modo demostración)."""


@dataclass(frozen=True)
class Resultado:
    puntuacion: float
    clase: str
    positivo: bool
    umbral: float
    umbral_texto: str
    regla: str
    n_miembros: int
    probabilidades_por_miembro: dict = field(default_factory=dict)
    sha256_declaracion: str = SHA256_DECLARACION
    sistema: str = SISTEMA_ID
    version_interfaz: str = VERSION_PROTOTIPO
    advertencias_entorno: tuple = ()


def _predecir_torch(modelo, tensor, device) -> float:
    """Probabilidad de la clase maligna (softmax float32) de un miembro: igual que `classification.evaluate.predict` para un lote de una imagen."""
    import torch
    modelo.eval()
    with torch.no_grad():
        salida = modelo(tensor.to(device))
        return float(torch.softmax(salida, dim=1)[0, 1].cpu().numpy().astype(np.float32))


class SistemaCongelado:
    def __init__(self, *, miembros, modelos, preprocesar, predecir, device, t_hat_texto, advertencias_entorno=()):
        self.miembros, self.modelos, self.preprocesar, self.predecir, self.device = list(miembros), dict(modelos), preprocesar, predecir, device
        self.t_hat_texto, self.t_hat = t_hat_texto, SF.t_hat_exacto(t_hat_texto)
        self.advertencias_entorno = tuple(advertencias_entorno)
        if t_hat_texto != SF.T_HAT or len(self.miembros) != SF.N_MIEMBROS or set(self.modelos) != {m["id_corrida"] for m in self.miembros}:
            raise SistemaNoDisponible("el sistema no coincide con el congelado (umbral o miembros)")

    def probabilidad_miembro(self, id_corrida: str, imagen, tensor=None) -> float:
        tensor = self.preprocesar(imagen) if tensor is None else tensor
        return float(np.float32(self.predecir(self.modelos[id_corrida], tensor, self.device)))

    def clasificar(self, imagen) -> Resultado:
        tensor = self.preprocesar(imagen)                                                   # un único preprocesamiento (el de G2) para los 15 miembros
        probs = {m["id_corrida"]: np.asarray([self.probabilidad_miembro(m["id_corrida"], imagen, tensor)], dtype=np.float32) for m in self.miembros}
        agg = SF.agregar_y_decidir(probs, SF.T_HAT)                                         # promedio simple (float64) y regla `>=`
        positivo = bool(agg["positivo"][0])
        return Resultado(puntuacion=float(agg["puntuacion"][0]), clase=ETIQUETA_MALIGNO if positivo else ETIQUETA_BENIGNO, positivo=positivo, umbral=float(agg["t_hat"]),
                         umbral_texto=self.t_hat_texto, regla="p >= t_hat", n_miembros=int(agg["n_miembros"]),
                         probabilidades_por_miembro={k: float(v[0]) for k, v in sorted(probs.items())}, advertencias_entorno=self.advertencias_entorno)


def cargar_sistema(root=None, *, sha256_declaracion=SHA256_DECLARACION, exigir_reporte=False, verificar_declaracion=None, cargar_modelo=None, device=None, preprocesar=None, predecir=None):
    """Verifica y carga el sistema congelado. Lanza `SistemaNoDisponible` (con el motivo) ante cualquier fallo; nunca devuelve un sistema parcial."""
    root = Path(root or RAIZ_REPO)
    try:
        verif = verificar_declaracion or (lambda r, s: DC.exigir_declaracion(r, s, comprobar_checkpoints=True, comprobar_reporte=exigir_reporte))
        verif(root, sha256_declaracion)
        config = C.cargar_config(root / C.CONFIG_REL)
        config_g2 = cargar_config_g2(root / config["referencia_g2"]["config"])
        if DC.sha256_lf(root / config["referencia_g2"]["config"]) != config["referencia_g2"]["sha256_contenido_blob"]:
            raise GuardError("la configuración de G2 difiere del pin del sistema congelado")
        t_hat_texto = config["sistema"]["umbral"]["t_hat"]
        miembros = C.validar_miembros(C.leer_tabla(root / C.MIEMBROS_REL), config)
        dev = device if device is not None else SF._dispositivo()
        cargar = cargar_modelo or SF._cargar_modelo_g2
        modelos = {m["id_corrida"]: cargar(SF.verificar_miembro_en_disco(root, m), dev) for m in miembros}      # SHA-256 verificado justo antes de cargar
        if preprocesar is None:
            transformar = SF.preprocesamiento()["get_val_transforms"](config_g2["receta"]["entrada"]["image_size"])
            preprocesar = lambda imagen: transformar(imagen).unsqueeze(0)  # noqa: E731
        efectivo = SF._versiones()
        avisos = tuple(f"{k}: verificado con {config['entorno_registrado'].get(k)!r}, este equipo {efectivo.get(k)!r}" for k in C.ENTORNO_CLAVES
                       if str(config["entorno_registrado"].get(k)) != str(efectivo.get(k)))
        return SistemaCongelado(miembros=miembros, modelos=modelos, preprocesar=preprocesar, predecir=predecir or _predecir_torch, device=dev, t_hat_texto=t_hat_texto,
                                advertencias_entorno=avisos)
    except SistemaNoDisponible:
        raise
    except (GuardError, OSError, ImportError, KeyError, ValueError, RuntimeError) as exc:
        raise SistemaNoDisponible(f"{type(exc).__name__}: {str(exc)[:300]}") from exc
