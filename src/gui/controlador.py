"""Lógica de la interfaz de R9 sin widgets (se prueba sin pantalla): estado del sistema, imagen cargada, clasificación y textos del resultado.

Sin sistema verificado no hay inferencia: `clasificar` lanza `SistemaNoDisponible` y el estado es «Modo demostración: modelo no disponible». Jamás se construye un resultado simulado.
"""

from __future__ import annotations

from src.gui import textos as T
from src.gui.sistema import SHA256_DECLARACION, SISTEMA_ID, Resultado, SistemaNoDisponible, cargar_sistema
from src.gui.validacion_imagen import ImagenInvalida, validar_y_cargar


class Controlador:
    def __init__(self, cargar=None):
        self._cargar = cargar or cargar_sistema
        self.sistema, self.motivo = None, None
        self.imagen, self.info, self.resultado = None, None, None

    def iniciar(self) -> None:
        try:
            self.sistema, self.motivo = self._cargar(), None
        except SistemaNoDisponible as exc:
            self.sistema, self.motivo = None, str(exc)

    @property
    def listo(self) -> bool:
        return self.sistema is not None

    def estado_texto(self) -> str:
        if not self.listo:
            return f"{T.MODO_MAQUETA}. " + T.MAQUETA_DETALLE.format(motivo=self.motivo or "el sistema aún no se ha iniciado")
        s = self.sistema
        linea = f"Sistema verificado: {SISTEMA_ID}. {len(s.miembros)}/{len(s.miembros)} checkpoints con SHA-256 comprobado. Declaración {SHA256_DECLARACION[:12]}…"
        return linea + (" ATENCIÓN, entorno distinto del verificado: " + "; ".join(s.advertencias_entorno) if s.advertencias_entorno else "")

    def abrir(self, ruta) -> dict:
        """Valida y carga la imagen; si falla lanza `ImagenInvalida` y conserva la imagen anterior."""
        imagen, info = validar_y_cargar(ruta)
        self.imagen, self.info, self.resultado = imagen, info, None
        return info

    def clasificar(self) -> Resultado:
        if not self.listo:
            raise SistemaNoDisponible(T.MODO_MAQUETA)
        if self.imagen is None:
            raise ImagenInvalida("Primero cargue una imagen ecográfica.")
        self.resultado = self.sistema.clasificar(self.imagen)
        return self.resultado

    @staticmethod
    def lineas_resultado(r: Resultado) -> list[str]:
        lineas = [f"Clase asignada: {r.clase}", f"Puntuación del sistema (promedio de {r.n_miembros} modelos): {r.puntuacion:.6f}  — no es una probabilidad calibrada",
                  f"Umbral aplicado: {r.umbral_texto}   Regla: puntuación ≥ umbral → {T.ETIQUETA_MALIGNO}; en otro caso → {T.ETIQUETA_BENIGNO}",
                  f"Sistema: {r.sistema}", f"Declaración (SHA-256): {r.sha256_declaracion}", f"Interfaz: {r.version_interfaz}"]
        if r.advertencias_entorno:
            lineas.append("Atención, entorno distinto del verificado: " + "; ".join(r.advertencias_entorno))
        return lineas

    @staticmethod
    def avisos() -> list[str]:
        return T.advertencias()
