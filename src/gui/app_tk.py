"""Interfaz gráfica (Tkinter) del prototipo demostrativo de R9: cargar una ecografía 2D, visualizarla y obtener la clasificación BENIGNO/MALIGNO del sistema congelado D-060.

  python -B -m src.gui.app_tk                                  # interfaz interactiva
  python -B -m src.gui.app_tk --imagen RUTA --clasificar --captura SALIDA.png --salir     # captura reproducible de la ventana real (Windows/macOS)

Sin sistema verificado la ventana muestra «Modo demostración: modelo no disponible» y el botón Clasificar queda desactivado (nunca resultados simulados). No abre imágenes de test.
"""

from __future__ import annotations

import argparse
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext

from src.gui import textos as T
from src.gui.controlador import Controlador
from src.gui.sistema import SistemaNoDisponible
from src.gui.validacion_imagen import ImagenInvalida

COLOR_AVISO, COLOR_MALIGNO, COLOR_BENIGNO, COLOR_MAQUETA = "#8b0000", "#8b0000", "#0b5c4a", "#7a4b00"
ANCHO_IMAGEN, ALTO_IMAGEN = 460, 360


class AppTk:
    def __init__(self, raiz: tk.Tk, controlador: Controlador, *, usar_hilo: bool = True):
        self.raiz, self.c, self.usar_hilo = raiz, controlador, usar_hilo
        self._foto, self._cola, self._ultimo_error = None, queue.Queue(), ""
        raiz.title(f"{T.TITULO} — {T.VERSION_PROTOTIPO}")
        raiz.geometry("1040x940")
        tk.Label(raiz, text=T.AVISO_PRINCIPAL, bg="#fde8e8", fg=COLOR_AVISO, wraplength=1000, justify="left", font=("Segoe UI", 10, "bold"), padx=8, pady=6).pack(fill="x")
        tk.Label(raiz, text=T.PRESENTACION, wraplength=1000, justify="left", font=("Segoe UI", 9), padx=8).pack(fill="x")
        self.estado = tk.Label(raiz, text="", wraplength=1000, justify="left", font=("Segoe UI", 9), padx=8, pady=4)
        self.estado.pack(fill="x")
        cuerpo = tk.Frame(raiz)
        cuerpo.pack(fill="both", expand=True, padx=8)
        izq, der = tk.Frame(cuerpo), tk.Frame(cuerpo)
        izq.pack(side="left", padx=4, pady=4)
        der.pack(side="left", fill="both", expand=True, padx=8, pady=4)
        self.lienzo = tk.Label(izq, text="(ninguna imagen cargada)", width=ANCHO_IMAGEN // 8, height=ALTO_IMAGEN // 16, relief="sunken", bg="#222222", fg="#cccccc")
        self.lienzo.pack()
        self.info = tk.Label(izq, text="", justify="left", font=("Segoe UI", 8))
        self.info.pack(anchor="w")
        botones = tk.Frame(der)
        botones.pack(anchor="w")
        self.b_cargar = tk.Button(botones, text="Cargar imagen…", command=self.cargar_imagen, width=18)
        self.b_clasificar = tk.Button(botones, text="Clasificar", command=self.clasificar, width=18, state="disabled")
        tk.Button(botones, text="Acerca de", command=lambda: messagebox.showinfo("Acerca de", T.ACERCA_DE + "\n\n" + T.PRESENTACION), width=12).pack(side="right", padx=4)
        self.b_cargar.pack(side="left", padx=4)
        self.b_clasificar.pack(side="left", padx=4)
        self.clase = tk.Label(der, text="—", font=("Segoe UI", 30, "bold"))
        self.clase.pack(anchor="w", pady=(10, 0))
        self.detalle = tk.Label(der, text="", justify="left", wraplength=480, font=("Segoe UI", 9))
        self.detalle.pack(anchor="w")
        self.avisos = scrolledtext.ScrolledText(raiz, height=15, wrap="word", font=("Segoe UI", 9))
        self.avisos.pack(fill="both", padx=8, pady=6)
        self.avisos.insert("1.0", "\n\n".join(T.advertencias()))
        self.avisos.configure(state="disabled")
        self.refrescar_estado()

    # ---------------------------------------------------------------- estado y errores
    def refrescar_estado(self) -> None:
        listo = self.c.listo
        self.estado.configure(text=self.c.estado_texto(), fg=("#0b5c4a" if listo else COLOR_MAQUETA))
        if not listo:
            self.clase.configure(text=T.MODO_MAQUETA, fg=COLOR_MAQUETA, font=("Segoe UI", 16, "bold"))
        self.b_clasificar.configure(state="normal" if (listo and self.c.imagen is not None) else "disabled")

    def _error(self, mensaje: str) -> None:
        self.detalle.configure(text=mensaje, fg=COLOR_AVISO)
        self.info.configure(text="")
        self._ultimo_error = mensaje

    # ---------------------------------------------------------------- cargar y visualizar (F1, F2)
    def cargar_imagen(self, ruta=None) -> bool:
        ruta = ruta or filedialog.askopenfilename(title="Seleccione una ecografía 2D (PNG o JPG)", filetypes=[("Imágenes PNG/JPG", "*.png *.jpg *.jpeg")])
        if not ruta:
            return False
        try:
            info = self.c.abrir(ruta)
        except ImagenInvalida as exc:
            self._error(str(exc))
            self.b_clasificar.configure(state="disabled")
            if self.c.listo:
                self.clase.configure(text="—", fg="black", font=("Segoe UI", 30, "bold"))       # no dejar visible la clase de una imagen anterior junto al error de otro archivo
            return False
        self._ultimo_error = ""
        from PIL import ImageTk
        vista = self.c.imagen.copy()
        vista.thumbnail((ANCHO_IMAGEN, ALTO_IMAGEN))
        self._foto = ImageTk.PhotoImage(vista)
        self.lienzo.configure(image=self._foto, text="", width=vista.width, height=vista.height)
        self.info.configure(text=f"{info['archivo']} · {info['formato']} · {info['ancho']}×{info['alto']} px", fg="black")
        self.detalle.configure(text="Imagen cargada. Se presupone una lesión ovárica previamente identificada.", fg="black")
        if self.c.listo:
            self.clase.configure(text="—", fg="black", font=("Segoe UI", 30, "bold"))
        self.refrescar_estado()
        return True

    # ---------------------------------------------------------------- clasificar (F3)
    def _mostrar_resultado(self, r) -> None:
        color = COLOR_MALIGNO if r.positivo else COLOR_BENIGNO
        self.clase.configure(text=r.clase, fg=color, font=("Segoe UI", 30, "bold"))
        self.detalle.configure(text="\n".join(Controlador.lineas_resultado(r)[1:]), fg="black")
        self.b_cargar.configure(state="normal")
        self.refrescar_estado()

    def _clasificar_sync(self) -> bool:
        try:
            self._mostrar_resultado(self.c.clasificar())
            return True
        except (SistemaNoDisponible, ImagenInvalida) as exc:
            self._error(str(exc))
            return False

    def clasificar(self) -> None:
        if not self.usar_hilo:
            self._clasificar_sync()
            return
        self.b_clasificar.configure(state="disabled")
        self.b_cargar.configure(state="disabled")
        self.detalle.configure(text="Clasificando con los 15 modelos…", fg="black")

        def trabajo():
            try:
                self._cola.put(("ok", self.c.clasificar()))
            except (SistemaNoDisponible, ImagenInvalida) as exc:
                self._cola.put(("error", str(exc)))
            except Exception as exc:                                                          # noqa: BLE001 - nunca dejar la interfaz bloqueada
                self._cola.put(("error", f"Error inesperado durante la clasificación: {type(exc).__name__}"))
        threading.Thread(target=trabajo, daemon=True).start()
        self.raiz.after(150, self._esperar)

    def _esperar(self) -> None:
        try:
            tipo, valor = self._cola.get_nowait()
        except queue.Empty:
            self.raiz.after(150, self._esperar)
            return
        if tipo == "ok":
            self._mostrar_resultado(valor)
        else:
            self._error(valor)
            self.b_cargar.configure(state="normal")
            self.refrescar_estado()

    def capturar(self, ruta_png) -> None:
        """Captura REAL de la ventana (PIL.ImageGrab; Windows/macOS)."""
        import time
        from PIL import ImageGrab
        self.raiz.attributes("-topmost", True)                                                # ventana al frente y completamente dibujada antes de capturar
        self.raiz.lift()
        for _ in range(3):
            self.raiz.update()
            time.sleep(0.4)
        x, y, w, h = self.raiz.winfo_rootx(), self.raiz.winfo_rooty(), self.raiz.winfo_width(), self.raiz.winfo_height()
        ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(ruta_png)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="R9: interfaz demostrativa (Tkinter). Prototipo académico; sin validez diagnóstica ni uso clínico.")
    ap.add_argument("--imagen", default=None)
    ap.add_argument("--clasificar", action="store_true")
    ap.add_argument("--captura", default=None, help="guarda una captura PNG de la ventana")
    ap.add_argument("--salir", action="store_true", help="cierra tras cargar/clasificar/capturar")
    args = ap.parse_args(argv)
    controlador = Controlador()
    controlador.iniciar()
    raiz = tk.Tk()
    app = AppTk(raiz, controlador, usar_hilo=not (args.clasificar and args.salir))
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
