"""Interfaz gráfica R9, PRESENTACIÓN v2 (Tkinter + ttk, sin dependencias nuevas): solo cambia el aspecto y la organización de la ventana.

  python -B -m src.gui.app_tk_v2                                  # interfaz interactiva
  python -B -m src.gui.app_tk_v2 --imagen RUTA --clasificar --captura SALIDA.png --salir

`AppTkV2` HEREDA de `AppTk` (src/gui/app_tk.py, presentación v1, que se conserva intacta) toda la lógica de cargar (F1), visualizar (F2) y clasificar (F3): mismos métodos, mismos hilos, mismos textos
(`src/gui/textos.py`), mismo controlador y mismo sistema congelado. Aquí solo se redefinen la construcción de los widgets y el color/tamaño de lo que se muestra. No hay código de inferencia,
métricas, calibración ni acceso a archivos: la clasificación sigue siendo `Controlador.clasificar` → `SistemaCongelado.clasificar` (quince modelos, promedio simple y umbral 0.08456226376195748).

Organización: encabezado (nombre, versión y estado de verificación de los quince modelos); aviso permanente de que no tiene validez diagnóstica ni uso clínico; pestaña «Clasificación» (panel izquierdo:
cargar y visualizar la ecografía; panel derecho: clase, puntuación del ensamble, umbral e identificación del sistema) y pestaña «Información metodológica» (avisos, alcance, mapeo v2, nota numérica y
rendimiento de R8); barra de estado con la verificación completa del sistema. El tamaño de la imagen y el ajuste de los textos se adaptan a la ventana.
"""

from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from src.gui import textos as T
from src.gui.app_tk import ANCHO_IMAGEN, ALTO_IMAGEN, COLOR_AVISO, COLOR_BENIGNO, COLOR_MALIGNO, COLOR_MAQUETA, AppTk
from src.gui.controlador import Controlador

VERSION_PRESENTACION = "presentación v2"
FUENTE = "Segoe UI"
AZUL, FONDO, TARJETA, BORDE, TEXTO_SUAVE = "#1f3a5f", "#eef1f5", "#ffffff", "#d3dae3", "#4a5568"
NEUTRO = {"bg": "#eef1f4", "fg": "#222222"}
COLORES_CLASE = {T.ETIQUETA_BENIGNO: {"bg": "#e3f3ee", "fg": COLOR_BENIGNO}, T.ETIQUETA_MALIGNO: {"bg": "#fbe4e4", "fg": COLOR_MALIGNO}, T.MODO_MAQUETA: {"bg": "#fff1d6", "fg": COLOR_MAQUETA}}
SECCIONES_INFO = ("Aviso principal", "Alcance", "Etiquetas y mapeo v2", "Puntuación y umbral", "Nota numérica", "Rendimiento (evaluación final R8)")
MIN_ANCHO, MIN_ALTO = 880, 640


def _estilos(raiz) -> None:
    s = ttk.Style(raiz)
    try:
        s.theme_use("clam")                                                            # tema neutro y consistente entre equipos
    except tk.TclError:
        pass
    base = {"font": (FUENTE, 10, "bold"), "padding": (16, 8), "borderwidth": 0, "focusthickness": 2, "relief": "flat"}
    s.configure("Primario.TButton", background=AZUL, foreground="white", **base)
    s.map("Primario.TButton", background=[("disabled", "#c9d1db"), ("pressed", "#16304f"), ("active", "#2a4d7c")], foreground=[("disabled", "#7b8794")])
    s.configure("Secundario.TButton", background="#e1e7ee", foreground=AZUL, **base)
    s.map("Secundario.TButton", background=[("disabled", "#eceff3"), ("pressed", "#c9d3de"), ("active", "#d3dce6")], foreground=[("disabled", "#9aa5b1")])
    s.configure("Tarjeta.TFrame", background=TARJETA, relief="solid", borderwidth=1, bordercolor=BORDE)
    s.configure("Fondo.TFrame", background=FONDO)
    s.configure("TNotebook", background=FONDO, borderwidth=0, tabmargins=(8, 6, 8, 0))
    s.configure("TNotebook.Tab", font=(FUENTE, 10, "bold"), padding=(18, 7), background="#dfe5ec", foreground=TEXTO_SUAVE)
    s.map("TNotebook.Tab", background=[("selected", TARJETA)], foreground=[("selected", AZUL)])


def _titulo_y_subtitulo() -> tuple[str, str]:
    partes = T.TITULO.split(" — ", 1)                                                   # «Prototipo demostrativo» / «clasificación binaria de lesiones ováricas en ecografía 2D»
    return partes[0], (partes[1][:1].upper() + partes[1][1:] if len(partes) > 1 else "")


class AppTkV2(AppTk):
    def __init__(self, raiz: tk.Tk, controlador: Controlador, *, usar_hilo: bool = True):
        self.raiz, self.c, self.usar_hilo = raiz, controlador, usar_hilo
        self._foto, self._ultimo_error, self._ajuste_pendiente = None, "", None
        import queue
        self._cola = queue.Queue()
        _estilos(raiz)
        raiz.title(f"{T.TITULO} — {T.VERSION_PROTOTIPO}")
        ancho, alto = min(1180, raiz.winfo_screenwidth() - 40), min(780, raiz.winfo_screenheight() - 80)
        raiz.geometry(f"{ancho}x{alto}")
        raiz.minsize(min(MIN_ANCHO, ancho), min(MIN_ALTO, alto))
        raiz.configure(bg=FONDO)
        raiz.columnconfigure(0, weight=1)
        raiz.rowconfigure(3, weight=1)
        self._encabezado(raiz)
        self._aviso(raiz)
        self.notebook = ttk.Notebook(raiz)
        self.notebook.grid(row=3, column=0, sticky="nsew", padx=8, pady=(0, 4))
        self.pestana_clasificacion = ttk.Frame(self.notebook, style="Fondo.TFrame")
        self.pestana_info = ttk.Frame(self.notebook, style="Fondo.TFrame")
        self.notebook.add(self.pestana_clasificacion, text="Clasificación")
        self.notebook.add(self.pestana_info, text="Información metodológica")
        self._panel_izquierdo(self.pestana_clasificacion)
        self._panel_derecho(self.pestana_clasificacion)
        self._informacion(self.pestana_info)
        self.estado = tk.Label(raiz, text="", bg="#e3e8ee", fg="#222222", anchor="w", justify="left", font=(FUENTE, 9), padx=10, pady=5, wraplength=1000)
        self.estado.grid(row=4, column=0, sticky="ew")
        raiz.bind("<Configure>", self._al_redimensionar, add="+")
        self.refrescar_estado()

    # ---------------------------------------------------------------- construcción
    def _encabezado(self, raiz) -> None:
        cab = tk.Frame(raiz, bg=AZUL, padx=14, pady=8)
        cab.grid(row=0, column=0, sticky="ew")
        cab.columnconfigure(0, weight=1)
        titulo, subtitulo = _titulo_y_subtitulo()
        tk.Label(cab, text=titulo, bg=AZUL, fg="white", font=(FUENTE, 16, "bold"), anchor="w").grid(row=0, column=0, sticky="w")
        tk.Label(cab, text=subtitulo, bg=AZUL, fg="#c9d6e6", font=(FUENTE, 10), anchor="w").grid(row=1, column=0, sticky="w")
        self.chip = tk.Label(cab, text="", font=(FUENTE, 10, "bold"), padx=12, pady=5, fg="white")
        self.chip.grid(row=0, column=1, rowspan=1, sticky="e", padx=(12, 0))
        self.version = tk.Label(cab, text=f"{T.VERSION_PROTOTIPO} · {VERSION_PRESENTACION}", bg=AZUL, fg="#c9d6e6", font=(FUENTE, 9), anchor="e")
        self.version.grid(row=1, column=1, sticky="e", padx=(12, 0))

    def _aviso(self, raiz) -> None:
        self.aviso = tk.Label(raiz, text=T.AVISO_PRINCIPAL, bg="#fde8e8", fg=COLOR_AVISO, font=(FUENTE, 10, "bold"), justify="left", anchor="w", padx=14, pady=6, wraplength=1100)
        self.aviso.grid(row=1, column=0, sticky="ew")
        self.presentacion = tk.Label(raiz, text=T.PRESENTACION, bg=FONDO, fg=TEXTO_SUAVE, font=(FUENTE, 9, "italic"), justify="left", anchor="w", padx=14, pady=4, wraplength=1100)
        self.presentacion.grid(row=2, column=0, sticky="ew")

    def _panel_izquierdo(self, madre) -> None:
        madre.columnconfigure(0, weight=3, minsize=380)
        madre.columnconfigure(1, weight=2, minsize=320)
        madre.rowconfigure(0, weight=1)
        izq = ttk.Frame(madre, style="Tarjeta.TFrame", padding=12)
        izq.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        izq.columnconfigure(0, weight=1)
        izq.rowconfigure(1, weight=1)
        tk.Label(izq, text="Ecografía", bg=TARJETA, fg=AZUL, font=(FUENTE, 12, "bold"), anchor="w").grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.marco_imagen = tk.Frame(izq, bg="#222222", width=ANCHO_IMAGEN, height=ALTO_IMAGEN)
        self.marco_imagen.grid(row=1, column=0, sticky="nsew")
        self.marco_imagen.grid_propagate(False)
        self.marco_imagen.pack_propagate(False)
        self.lienzo = tk.Label(self.marco_imagen, text="(ninguna imagen cargada)", bg="#222222", fg="#cccccc", font=(FUENTE, 10))
        self.lienzo.place(relx=0.5, rely=0.5, anchor="center")
        self.info = tk.Label(izq, text="", bg=TARJETA, fg="black", justify="left", anchor="w", font=(FUENTE, 9))
        self.info.grid(row=2, column=0, sticky="ew", pady=(6, 8))
        botones = tk.Frame(izq, bg=TARJETA)
        botones.grid(row=3, column=0, sticky="ew")
        botones.columnconfigure((0, 1), weight=1, uniform="botones")
        self.b_cargar = ttk.Button(botones, text="Cargar imagen…", command=self.cargar_imagen, style="Secundario.TButton")
        self.b_clasificar = ttk.Button(botones, text="Clasificar", command=self.clasificar, style="Primario.TButton", state="disabled")
        self.b_cargar.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.b_clasificar.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.marco_imagen.bind("<Configure>", lambda _e: self._programar_ajuste(), add="+")

    def _panel_derecho(self, madre) -> None:
        der = ttk.Frame(madre, style="Tarjeta.TFrame", padding=12)
        der.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)
        der.columnconfigure(0, weight=1)
        der.rowconfigure(2, weight=1)
        self.tarjeta_resultado = der
        tk.Label(der, text="Clasificación", bg=TARJETA, fg=AZUL, font=(FUENTE, 12, "bold"), anchor="w").grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.clase = tk.Label(der, text="—", font=(FUENTE, 30, "bold"), pady=10, **NEUTRO)
        self.clase.grid(row=1, column=0, sticky="ew")
        self.detalle = tk.Label(der, text="", bg=TARJETA, fg="black", justify="left", anchor="nw", font=(FUENTE, 9), wraplength=360)
        self.detalle.grid(row=2, column=0, sticky="nsew", pady=(10, 8))
        ttk.Button(der, text="Información metodológica y limitaciones", style="Secundario.TButton", command=self.mostrar_informacion).grid(row=3, column=0, sticky="ew")
        der.bind("<Configure>", lambda e: self.detalle.configure(wraplength=max(220, e.width - 40)), add="+")

    def _informacion(self, madre) -> None:
        madre.columnconfigure(0, weight=1)
        madre.rowconfigure(0, weight=1)
        self.avisos = ScrolledText(madre, wrap="word", font=(FUENTE, 10), bg=TARJETA, relief="flat", padx=14, pady=10, spacing3=4)
        self.avisos.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.avisos.tag_configure("titulo", font=(FUENTE, 10, "bold"), foreground=AZUL, spacing1=8)
        for seccion, texto in zip(SECCIONES_INFO, T.advertencias()):                     # mismos textos y mismo orden que T.advertencias()
            self.avisos.insert("end", seccion + "\n", "titulo")
            self.avisos.insert("end", texto + "\n\n")
        self.avisos.insert("end", "Acerca de\n", "titulo")
        self.avisos.insert("end", T.ACERCA_DE + "\n\n" + T.PRESENTACION + "\n")
        self.avisos.configure(state="disabled")

    # ---------------------------------------------------------------- navegación y adaptación
    def mostrar_informacion(self) -> None:
        self.notebook.select(self.pestana_info)

    def mostrar_clasificacion(self) -> None:
        self.notebook.select(self.pestana_clasificacion)

    def _al_redimensionar(self, evento) -> None:
        if evento.widget is self.raiz:
            ancho = max(300, evento.width - 40)
            for etiqueta in (self.aviso, self.presentacion, self.estado):
                etiqueta.configure(wraplength=ancho)

    def _programar_ajuste(self) -> None:
        if self._ajuste_pendiente is not None:
            self.raiz.after_cancel(self._ajuste_pendiente)
        self._ajuste_pendiente = self.raiz.after(60, self._ajustar_imagen)

    def _ajustar_imagen(self) -> None:
        """Vuelve a dibujar la ecografía cargada para que quepa en el marco (no amplía imágenes pequeñas)."""
        self._ajuste_pendiente = None
        if self.c.imagen is None:
            return
        from PIL import ImageTk
        ancho, alto = self.marco_imagen.winfo_width() - 12, self.marco_imagen.winfo_height() - 12
        if ancho < 120 or alto < 90:                                                     # ventana aún sin dibujar (p. ej. oculta): tamaño nominal
            ancho, alto = ANCHO_IMAGEN, ALTO_IMAGEN
        vista = self.c.imagen.copy()
        vista.thumbnail((ancho, alto))
        self._foto = ImageTk.PhotoImage(vista)
        self.lienzo.configure(image=self._foto, text="", width=vista.width, height=vista.height)

    # ---------------------------------------------------------------- estado visible (la lógica es la de AppTk)
    def _colorear_clase(self) -> None:
        self.clase.configure(**COLORES_CLASE.get(self.clase.cget("text"), NEUTRO))

    def _actualizar_chip(self) -> None:
        if self.c.listo:
            s = self.c.sistema
            extra = " · entorno distinto" if s.advertencias_entorno else ""
            self.chip.configure(text=f"● {len(s.miembros)}/{len(s.miembros)} modelos verificados (SHA-256){extra}", bg="#b26a00" if extra else "#1e7a5f")
        else:
            self.chip.configure(text=f"● {T.MODO_MAQUETA}", bg="#b26a00")

    def refrescar_estado(self) -> None:
        super().refrescar_estado()
        self._actualizar_chip()
        self._colorear_clase()

    def cargar_imagen(self, ruta=None) -> bool:
        ok = super().cargar_imagen(ruta)
        if ok:
            self._ajustar_imagen()
        self._colorear_clase()
        return ok

    def _mostrar_resultado(self, r) -> None:
        super()._mostrar_resultado(r)
        self._colorear_clase()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="R9: interfaz demostrativa (Tkinter, presentación v2). Prototipo académico; sin validez diagnóstica ni uso clínico.")
    ap.add_argument("--imagen", default=None)
    ap.add_argument("--clasificar", action="store_true")
    ap.add_argument("--captura", default=None, help="guarda una captura PNG de la ventana")
    ap.add_argument("--salir", action="store_true", help="cierra tras cargar/clasificar/capturar")
    args = ap.parse_args(argv)
    controlador = Controlador()
    controlador.iniciar()
    raiz = tk.Tk()
    app = AppTkV2(raiz, controlador, usar_hilo=not (args.clasificar and args.salir))
    if args.imagen:
        app.cargar_imagen(args.imagen)
        if args.clasificar:
            app._clasificar_sync()
    if args.captura:
        app.capturar(Path(args.captura))
    if args.salir:
        raiz.destroy()
        return 0
    raiz.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
