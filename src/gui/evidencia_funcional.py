"""R9 (D-076): pruebas funcionales F1–F3 y de gestión de errores sobre la ventana Tkinter REAL, con capturas auténticas de pantalla y un registro reproducible.

  F1 cargar una imagen PNG/JPG (con validación y mensajes claros)   F2 visualizar la imagen cargada y sus datos básicos   F3 clasificar con el sistema congelado y mostrar clase, puntuación, umbral y avisos

  python -B -m src.gui.evidencia_funcional --autorizacion CONFIRMADO:D-076 [--imagen-desarrollo RUTA ...] [--sin-capturas] [--sufijo _2]

Qué hace: abre la ventana real (sistema congelado verificado: declaración D-060 y SHA-256 de los 15 checkpoints), ejecuta los casos de `CASOS` en orden, captura la pantalla (PIL.ImageGrab) en cada estado
relevante y escribe, en un directorio NUEVO de `results/r9_gui/evidencia_funcional/`, las capturas, `registro_pruebas_funcionales.json` y `registro_pruebas_funcionales.md`.
Imágenes: una sintética generada aquí (no clínica) y, por defecto, las dos primeras por nombre de `train/images` de desarrollo (ninguna regla usa etiquetas); jamás test (la validación rechaza rutas `test`).

Qué NO hace: no calcula métricas de clasificación, no lee etiquetas, no compara con el OOF ni con la tolerancia de D-072, no repite D-072/D-074/D-075, no evalúa sobre desarrollo, no entrena ni recalibra.
Comprobaciones de F3 sobre cada clasificación (criterio funcional propuesto, ver docs/09): una imagen por inferencia (lote de tamaño 1) para cada uno de los 15 modelos, un único preprocesamiento, el mismo tensor
en los 15, promedio aritmético simple recomputado de forma independiente, umbral exacto 0.08456226376195748 con `>=`, repetibilidad bit a bit y, para imágenes de desarrollo, tensor idéntico al de la ruta de lectura de G2.
Las capturas con imágenes de desarrollo se nombran `dev_*` (el `.gitignore` las excluye: revisar antes de versionar o publicar).
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import tempfile
import time
import tkinter as tk
from pathlib import Path

import numpy as np

from src.r4_v2.common import RAIZ_REPO, GuardError, directorio_nuevo, escribir_json_nuevo, exigir_salida_segura, ruta_menciona_test, sha256_archivo
from src.r4_v2.common import cargar_config as cargar_config_g2
from src.r7_v2 import sistema_final as SF
from src.r7_v2 import umbral_sistema as U
from src.r8_v2 import congelamiento as C
from src.gui import textos as T
from src.gui.app_tk import AppTk
from src.gui.controlador import Controlador
from src.gui.sistema import SHA256_DECLARACION, SISTEMA_ID, SistemaNoDisponible, cargar_sistema
from src.gui.validacion_imagen import ImagenInvalida

TOKEN_DECISION = "D-076"                                                            # presentación v1 (protocolo de validación funcional)
TOKEN_POR_INTERFAZ = {"v1": "D-076", "v2": "D-078"}                                 # la presentación v2 (rediseño visual) se valida bajo su propia decisión
SALIDA_REL = "results/r9_gui/evidencia_funcional"
BASE_DESARROLLO = ("data", "processed", "MMOTU_OTU_2D_experimental")
SPLITS_DESARROLLO = ("train", "validation")
N_DESARROLLO = 2
TOLERANCIA_PROMEDIO = 1e-12                                                         # solo el orden de suma en float64; no es una tolerancia de probabilidades
CODIGO_NO_SUPERADA = 6


# ============================================================================ imágenes de entrada
def imagen_sintetica(ruta: Path) -> Path:
    """Imagen SINTÉTICA (ruido gaussiano suavizado y una elipse hipoecoica): no es una ecografía real ni contiene datos de pacientes. Determinista."""
    from PIL import Image, ImageFilter
    rng = np.random.RandomState(76)
    base = np.clip(rng.normal(95, 28, (384, 512)), 0, 255)
    yy, xx = np.mgrid[0:384, 0:512]
    elipse = ((xx - 270) / 120.0) ** 2 + ((yy - 190) / 80.0) ** 2 <= 1.0
    base[elipse] = np.clip(base[elipse] * 0.35, 0, 255)
    Image.fromarray(base.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.4)).convert("RGB").save(ruta, format="PNG")
    return ruta


def ruta_desarrollo(root, ruta) -> Path:
    """Valida que la ruta sea una imagen de DESARROLLO (`train/images` o `validation/images` bajo `data/processed/MMOTU_OTU_2D_experimental`) y la devuelve resuelta."""
    raiz, p = Path(root).resolve(), Path(ruta).resolve()
    try:
        partes = p.relative_to(raiz).parts
    except ValueError:
        raise GuardError(f"la imagen de desarrollo debe estar dentro del repositorio: {ruta}") from None
    if len(partes) != 6 or partes[:3] != BASE_DESARROLLO or partes[3] not in SPLITS_DESARROLLO or partes[4] != "images" or ruta_menciona_test(partes):
        raise GuardError(f"solo se admiten imágenes de {'/'.join(BASE_DESARROLLO)}/(train|validation)/images; recibido: {ruta}")
    if not p.is_file():
        raise GuardError(f"la imagen de desarrollo no existe: {ruta}")
    return p


def imagenes_desarrollo_por_defecto(root, n: int = N_DESARROLLO) -> list[Path]:
    """Las `n` primeras de `train/images` por nombre (texto): regla sin etiquetas ni resultados."""
    carpeta = Path(root) / Path(*BASE_DESARROLLO) / "train" / "images"
    nombres = sorted((p.name for p in carpeta.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")), key=str) if carpeta.is_dir() else []
    if len(nombres) < n:
        raise GuardError(f"no se encontraron {n} imágenes de desarrollo en {carpeta}")
    return [carpeta / x for x in nombres[:n]]


def tensor_de_g2(root, config_g2, ruta: Path):
    """Tensor de la imagen por la ruta de lectura de G2 (`DatasetLesiones`: Image.open + convert RGB + las transformaciones de validación). `label_v2: 0` es un marcador exigido por el
    constructor; no es una etiqueta real y no se lee ninguna."""
    from src.r4_v2.dataset_lesiones import DatasetLesiones
    transformar = SF.preprocesamiento()["get_val_transforms"](config_g2["receta"]["entrada"]["image_size"])
    rel = Path(ruta).resolve().relative_to(Path(root).resolve())
    return DatasetLesiones(root, [{"image_path": rel.as_posix(), "image_name": rel.name, "label_v2": 0}], transformar)[0][0]


# ============================================================================ instrumentación (observa; no altera el resultado)
class Sonda:
    """Envuelve `preprocesar` y `predecir` del sistema para registrar cuántas veces se llaman y con qué tensores; devuelve exactamente lo mismo que el original."""

    def __init__(self, sistema):
        self.sistema, self.pre, self.pred = sistema, [], []

    def __enter__(self):
        s = self.sistema
        self._pre, self._pred = s.preprocesar, s.predecir

        def pre(imagen):
            t = self._pre(imagen)
            self.pre.append({"forma": list(t.shape), "id": id(t)})
            return t

        def pred(modelo, tensor, device):
            self.pred.append({"forma": list(tensor.shape), "tipo": str(tensor.dtype), "id": id(tensor), "modelo": id(modelo)})
            return self._pred(modelo, tensor, device)
        s.preprocesar, s.predecir = pre, pred
        return self

    def __exit__(self, *_):
        self.sistema.preprocesar, self.sistema.predecir = self._pre, self._pred


def comprobaciones_f3(resultado, sonda: Sonda, sistema) -> dict:
    """Criterio funcional de F3 sobre UNA clasificación (booleanos + valores observados). No usa etiquetas ni métricas."""
    probs = resultado.probabilidades_por_miembro
    claves = sorted(probs, key=str)
    media = float(np.mean(np.asarray([probs[k] for k in claves], dtype=np.float64)))
    t = float(T.T_HAT_TEXTO)
    return {"un_preprocesamiento": len(sonda.pre) == 1,
            "quince_inferencias": len(sonda.pred) == SF.N_MIEMBROS,
            "lote_de_una_imagen": bool(sonda.pred) and all(len(x["forma"]) == 4 and x["forma"][0] == 1 and x["forma"][1] == 3 for x in sonda.pred),
            "forma_del_tensor": sonda.pred[0]["forma"] if sonda.pred else None, "tipo_del_tensor": sonda.pred[0]["tipo"] if sonda.pred else None,
            "mismo_tensor_en_los_quince": len({x["id"] for x in sonda.pred}) == 1 and bool(sonda.pre) and sonda.pre[0]["id"] == sonda.pred[0]["id"],
            "quince_modelos_distintos": len({x["modelo"] for x in sonda.pred}) == SF.N_MIEMBROS and set(probs) == {m["id_corrida"] for m in sistema.miembros},
            "probabilidades_en_rango": all(0.0 <= v <= 1.0 and math.isfinite(v) for v in probs.values()),
            "promedio_independiente_dif": abs(media - resultado.puntuacion), "promedio_independiente_ok": abs(media - resultado.puntuacion) <= TOLERANCIA_PROMEDIO,
            "umbral_exacto": resultado.umbral_texto == T.T_HAT_TEXTO and resultado.umbral == t and resultado.regla == "p >= t_hat",
            "regla_mayor_o_igual": resultado.positivo == (resultado.puntuacion >= t) and resultado.clase == (T.ETIQUETA_MALIGNO if resultado.positivo else T.ETIQUETA_BENIGNO),
            "n_miembros": resultado.n_miembros}


def _resumen_resultado(r) -> dict:
    return {"clase": r.clase, "puntuacion": repr(r.puntuacion), "umbral": r.umbral_texto, "regla": r.regla, "n_miembros": r.n_miembros,
            "probabilidades_por_miembro": {k: repr(v) for k, v in r.probabilidades_por_miembro.items()}, "advertencias_entorno": list(r.advertencias_entorno)}


# ============================================================================ ejecución de los casos
class Ejecucion:
    def __init__(self, salida: Path, app: AppTk, controlador: Controlador, *, capturas: bool, root, tensor_g2=None, clase_app=AppTk):
        self.salida, self.app, self.c, self.capturas, self.root, self.tensor_g2, self.clase_app = Path(salida), app, controlador, capturas, root, tensor_g2, clase_app
        self.casos: list[dict] = []

    # ---- utilidades
    def capturar(self, nombre: str, app: AppTk | None = None):
        if not self.capturas:
            return None
        archivo = f"{nombre}.png"
        (app or self.app).capturar(self.salida / archivo)
        return archivo

    def caso(self, id_, funcionalidad, descripcion, esperado, fn) -> dict:
        """Ejecuta `fn() -> (observado, ok, datos)`; una excepción se registra como caso NO superado sin interrumpir la ejecución."""
        try:
            observado, ok, datos = fn()
        except Exception as exc:                                                    # noqa: BLE001 - el registro debe completarse siempre
            observado, ok, datos = f"excepción {type(exc).__name__}: {str(exc)[:300]}", False, {}
        fila = {"id": id_, "funcionalidad": funcionalidad, "descripcion": descripcion, "esperado": esperado, "observado": observado, "resultado": "superada" if ok else "NO superada", **datos}
        self.casos.append(fila)
        print(f"[{fila['resultado']}] {id_}: {observado[:160]}")
        return fila

    def esperar_clasificacion(self, plazo: float = 600.0) -> None:
        """Bombea los eventos de la ventana hasta que el hilo de clasificación termine (clase mostrada o error)."""
        t0 = time.time()
        while time.time() - t0 < plazo:
            self.app.raiz.update()
            if self.app.clase.cget("text") in (T.ETIQUETA_BENIGNO, T.ETIQUETA_MALIGNO) or self.app._ultimo_error:
                self.app.raiz.update()
                return
            time.sleep(0.05)
        raise TimeoutError("la clasificación no terminó en el plazo")

    # ---- casos
    def inicio(self):
        s = self.c.sistema
        cap = self.capturar("p00_inicio")
        ok = bool(self.c.listo) and len(s.miembros) == 15 and s.t_hat_texto == T.T_HAT_TEXTO and str(self.app.b_clasificar.cget("state")) == "disabled"
        return (f"estado: {self.c.estado_texto()[:140]}…; miembros={len(s.miembros)}; umbral={s.t_hat_texto}; Clasificar desactivado sin imagen", ok,
                {"captura": cap, "advertencias_entorno": list(s.advertencias_entorno)})

    def clasificar_sin_imagen(self):
        try:
            self.c.clasificar()
        except ImagenInvalida as exc:
            return f"mensaje: «{exc}»; sin resultado", self.c.resultado is None and "Primero cargue" in str(exc), {}
        return "no se produjo el error esperado", False, {}

    def cargar(self, ruta: Path, captura: str):
        ok_carga = self.app.cargar_imagen(ruta)
        cap = self.capturar(captura)
        return (f"cargada={ok_carga}; {self.app.info.cget('text')}", bool(ok_carga) and str(self.app.b_clasificar.cget("state")) == "normal", {"captura": cap, "imagen_sha256": sha256_archivo(ruta)})

    def visualizar(self, ruta: Path):
        info = self.app.info.cget("text")
        mostrada = self.app._foto is not None and str(self.app.lienzo.cget("image")) != ""
        ok = mostrada and ruta.name in info and f"{self.c.info['ancho']}×{self.c.info['alto']}" in info and self.c.imagen.mode == "RGB"
        return f"imagen en el lienzo={mostrada}; datos: {info}; modo interno RGB", ok, {}

    def clasificar(self, ruta: Path, captura: str, *, desarrollo: bool):
        sistema = self.c.sistema
        self.app._ultimo_error = ""
        with Sonda(sistema) as sonda:
            self.app.clasificar()                                                   # ruta real del botón (hilo de trabajo + cola)
            self.esperar_clasificacion()
        r = self.c.resultado
        if r is None:
            return f"sin resultado; error: {self.app._ultimo_error}", False, {"captura": self.capturar(captura)}
        comp = comprobaciones_f3(r, sonda, sistema)
        r2 = sistema.clasificar(self.c.imagen)                                      # segunda clasificación de la misma imagen: repetibilidad bit a bit (misma sesión y condiciones)
        comp["repetible_bit_a_bit"] = r2.probabilidades_por_miembro == r.probabilidades_por_miembro and r2.puntuacion == r.puntuacion and r2.clase == r.clase
        if desarrollo and self.tensor_g2 is not None:
            import torch
            comp["tensor_igual_a_la_ruta_de_g2_bit_a_bit"] = bool(torch.equal(sistema.preprocesar(self.c.imagen)[0], self.tensor_g2(ruta)))
        mostrado = self.app.clase.cget("text") == r.clase and T.T_HAT_TEXTO in self.app.detalle.cget("text") and "no es una probabilidad calibrada" in self.app.detalle.cget("text")
        comp["resultado_mostrado_en_la_ventana"] = bool(mostrado)
        cap = self.capturar(captura)
        ok = all(v for k, v in comp.items() if isinstance(v, bool))
        return (f"clase={r.clase}; puntuación={r.puntuacion:.6f}; umbral={r.umbral_texto}; n_miembros={r.n_miembros}; comprobaciones de F3 {'todas superadas' if ok else 'CON FALLOS'}", ok,
                {"captura": cap, "clasificacion": _resumen_resultado(r), "comprobaciones": comp})

    def error_de_carga(self, ruta: Path, fragmento: str, captura: str):
        anterior, resultado_antes = self.c.imagen, self.c.resultado
        ok_carga = self.app.cargar_imagen(ruta)
        msg = self.app._ultimo_error
        cap = self.capturar(captura)
        ok = (ok_carga is False and fragmento in msg and self.c.imagen is anterior and self.c.resultado is resultado_antes and str(self.app.b_clasificar.cget("state")) == "disabled"
              and self.app.clase.cget("text") == ("—" if self.c.listo else T.MODO_MAQUETA))
        return f"mensaje: «{msg}»; la imagen anterior se conserva; Clasificar desactivado; sin resultado asociado al archivo inválido", ok, {"captura": cap}

    def modo_demostracion(self, raiz_tk, ruta_valida: Path):
        ventana = tk.Toplevel(raiz_tk)
        with tempfile.TemporaryDirectory() as vacio:
            c = Controlador(cargar=lambda: cargar_sistema(vacio))                  # fallo REAL de verificación: no hay declaración ni checkpoints en esa raíz
            c.iniciar()
        app = self.clase_app(ventana, c, usar_hilo=False)
        ventana.geometry(f"1040x{min(940, raiz_tk.winfo_screenheight() - 70)}+10+10" if self.clase_app is AppTk else "+10+10")
        try:
            carga = app.cargar_imagen(ruta_valida)
            clasifica = app._clasificar_sync()
            cap = self.capturar("e10_modo_demostracion", app)
            ok = (not c.listo and carga is True and clasifica is False and c.resultado is None and app.clase.cget("text") == T.MODO_MAQUETA
                  and str(app.b_clasificar.cget("state")) == "disabled" and "no se muestran resultados simulados" in app.estado.cget("text"))
            return (f"{T.MODO_MAQUETA}; la imagen se visualiza (cargada={carga}); Clasificar desactivado; sin resultados simulados; motivo: {str(c.motivo)[:120]}", ok, {"captura": cap})
        finally:
            ventana.destroy()

    def avisos(self):
        if hasattr(self.app, "mostrar_informacion"):                                 # presentación v2: los avisos están en la pestaña «Información metodológica»
            self.app.mostrar_informacion()
            self.app.raiz.update()
        self.app.avisos.yview_moveto(1.0)
        self.app.raiz.update()
        texto = self.app.avisos.get("1.0", "end")
        cap = self.capturar("p99_avisos_final")
        todos = all(a[:40] in texto for a in T.advertencias())
        return f"{len(T.advertencias())} avisos en la ventana (incluye la nota numérica y el aviso de no uso clínico): {todos}", todos and T.NOTA_NUMERICA[:40] in texto, {"captura": cap}


def ejecutar(root, salida: Path, raiz_tk, controlador: Controlador, *, imagenes_desarrollo, capturas: bool = True, tensor_g2=None, usar_hilo: bool = True, clase_app=AppTk) -> dict:
    root, salida = Path(root), Path(salida)
    if not controlador.listo:
        raise SistemaNoDisponible(f"el sistema congelado no está disponible: {controlador.motivo}")
    app = clase_app(raiz_tk, controlador, usar_hilo=usar_hilo)
    pantalla = (raiz_tk.winfo_screenwidth(), raiz_tk.winfo_screenheight())
    raiz_tk.geometry(f"1040x{min(940, pantalla[1] - 70)}+10+10" if clase_app is AppTk else "+10+10")      # la presentación v2 fija su propio tamaño; solo se coloca arriba a la izquierda
    raiz_tk.update()
    ej = Ejecucion(salida, app, controlador, capturas=capturas, root=root, tensor_g2=tensor_g2, clase_app=clase_app)
    sintetica = imagen_sintetica(salida / "sintetica_01.png")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        from PIL import Image
        (tmp / "notas.txt").write_text("no es una imagen", encoding="utf-8")
        (tmp / "vacio.png").write_bytes(b"")
        (tmp / "corrupta.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"basura" * 20)
        Image.new("RGB", (64, 48), (80, 80, 80)).save(tmp / "bmp_como_png.png", format="BMP")
        Image.new("RGB", (16, 16), (80, 80, 80)).save(tmp / "diminuta.png", format="PNG")
        (tmp / "test" / "images").mkdir(parents=True)
        Image.new("RGB", (64, 48), (80, 80, 80)).save(tmp / "test" / "images" / "sintetica_en_carpeta_test.png", format="PNG")
        ej.caso("P00", "estado", "Inicio de la ventana con el sistema congelado verificado (15 checkpoints SHA-256 y declaración D-060)", "sistema verificado; Clasificar desactivado sin imagen", ej.inicio)
        ej.caso("E01", "gestión de errores", "Clasificar sin imagen cargada", "mensaje claro y ningún resultado", ej.clasificar_sin_imagen)
        ej.caso("F1-S", "F1 cargar", "Cargar una imagen PNG sintética válida", "imagen cargada; Clasificar habilitado", lambda: ej.cargar(sintetica, "f1_f2_sintetica"))
        ej.caso("F2-S", "F2 visualizar", "Visualizar la imagen sintética con sus datos (archivo, formato, dimensiones)", "imagen en el lienzo y datos coherentes", lambda: ej.visualizar(sintetica))
        ej.caso("F3-S", "F3 clasificar", "Clasificar la imagen sintética con el sistema congelado (botón real, hilo de trabajo)", "clase, puntuación, umbral exacto y avisos; comprobaciones de F3", lambda: ej.clasificar(sintetica, "f3_sintetica", desarrollo=False))
        for id_, ruta, fragmento in (("E02", tmp / "no_existe.png", "no existe"), ("E03", tmp / "notas.txt", "PNG o JPG"), ("E04", tmp / "vacio.png", "vacío"), ("E05", tmp / "corrupta.png", "dañado"),
                                     ("E06", tmp / "bmp_como_png.png", "PNG ni JPG"), ("E07", tmp / "diminuta.png", "Dimensiones"),
                                     ("E08", tmp / "test" / "images" / "sintetica_en_carpeta_test.png", "«test»")):
            ej.caso(id_, "gestión de errores", f"Cargar {ruta.name} (entrada inválida)", f"mensaje con «{fragmento}»; estado previo conservado; sin clasificación",
                    lambda ruta=ruta, fragmento=fragmento, id_=id_: ej.error_de_carga(ruta, fragmento, f"{id_.lower()}_error"))
        for i, ruta in enumerate(imagenes_desarrollo, 1):
            ruta = Path(ruta)
            ej.caso(f"F1-D{i}", "F1 cargar", f"Cargar una imagen de desarrollo ({ruta.name}) tras los errores (recuperación)", "imagen cargada; Clasificar habilitado", lambda ruta=ruta, i=i: ej.cargar(ruta, f"dev_f1_f2_{i:02d}"))
            ej.caso(f"F2-D{i}", "F2 visualizar", f"Visualizar la imagen de desarrollo {ruta.name}", "imagen en el lienzo y datos coherentes", lambda ruta=ruta: ej.visualizar(ruta))
            ej.caso(f"F3-D{i}", "F3 clasificar", f"Clasificar la imagen de desarrollo {ruta.name} (sin leer etiquetas ni calcular métricas)", "clase, puntuación, umbral exacto y avisos; comprobaciones de F3 y tensor igual al de G2",
                    lambda ruta=ruta, i=i: ej.clasificar(ruta, f"dev_f3_{i:02d}", desarrollo=True))
        ej.caso("E10", "modo demostración", "Sistema no disponible (verificación fallida real): visualizar sí, clasificar no, sin resultados simulados", "«Modo demostración: modelo no disponible»",
                lambda: ej.modo_demostracion(raiz_tk, sintetica))
        ej.caso("P99", "avisos", "Avisos de alcance, puntuación, nota numérica y rendimiento visibles en la ventana", "todos los avisos presentes", ej.avisos)
    superados = sum(c["resultado"] == "superada" for c in ej.casos)
    return {"casos": ej.casos, "superados": superados, "total": len(ej.casos), "todos_superados": superados == len(ej.casos),
            "pantalla": {"ancho": pantalla[0], "alto": pantalla[1]}, "capturas_reales": bool(capturas)}


# ============================================================================ registro
def markdown(reg: dict) -> str:
    filas = ["| ID | Funcionalidad | Descripción | Esperado | Observado | Resultado | Captura |", "|---|---|---|---|---|---|---|"]
    esc = lambda s: str(s).replace("|", "\\|").replace("\n", " ")  # noqa: E731
    for c in reg["casos"]:
        filas.append(f"| {c['id']} | {esc(c['funcionalidad'])} | {esc(c['descripcion'])} | {esc(c['esperado'])} | {esc(c['observado'])} | {c['resultado']} | {c.get('captura') or '—'} |")
    return (f"# Registro de pruebas funcionales de R9 ({TOKEN_DECISION})\n\nSuperados: {reg['superados']}/{reg['total']}. Sistema: {reg['sistema']}. Declaración SHA-256: {reg['sha256_declaracion']}.\n"
            f"Entorno: {json.dumps(reg['entorno'], ensure_ascii=False)}\n\nSin métricas de clasificación, sin etiquetas, sin comparación con el OOF ni con la tolerancia de D-072; sin test.\n\n" + "\n".join(filas) + "\n")


def main(argv=None, *, root=None, controlador: Controlador | None = None, tensor_g2=None, raiz_tk=None) -> int:
    ap = argparse.ArgumentParser(description="R9 (D-076): pruebas funcionales con la ventana real y capturas auténticas. Prototipo académico; sin validez diagnóstica.")
    ap.add_argument("--autorizacion", default="")
    ap.add_argument("--imagen-desarrollo", action="append", default=None, help="imagen de desarrollo (repetible); por defecto, las dos primeras de train/images por nombre")
    ap.add_argument("--sin-capturas", action="store_true", help="no captura la pantalla (solo pruebas); NO produce evidencia visual")
    ap.add_argument("--sufijo", default="")
    ap.add_argument("--interfaz", choices=("v1", "v2"), default="v1", help="presentación a probar: v1 (app_tk, conservada) o v2 (app_tk_v2, rediseño visual); use --sufijo distinto para no mezclar evidencias")
    args = ap.parse_args(argv)
    if callable(getattr(sys.stdout, "reconfigure", None)):                          # consolas de Windows en cp1252: los mensajes llevan tildes y símbolos
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    root = Path(root or RAIZ_REPO).resolve()
    propia = False
    try:
        token = U.exigir_autorizacion(args.autorizacion)
        decision = TOKEN_POR_INTERFAZ[args.interfaz]
        if token.split(":")[1] != decision:
            raise GuardError(f"con --interfaz {args.interfaz} esta herramienta solo se ejecuta bajo CONFIRMADO:{decision}")
        devs = [ruta_desarrollo(root, r) for r in (args.imagen_desarrollo or imagenes_desarrollo_por_defecto(root))]
        if controlador is None:
            controlador = Controlador()
            controlador.iniciar()
        if not controlador.listo:
            raise SistemaNoDisponible(controlador.motivo or "sistema no disponible")
        config_g2 = None
        if tensor_g2 is None:
            config_g2 = cargar_config_g2(root / C.cargar_config(root / C.CONFIG_REL)["referencia_g2"]["config"])
            tensor_g2 = lambda ruta: tensor_de_g2(root, config_g2, ruta)  # noqa: E731
        destino = directorio_nuevo(exigir_salida_segura(root, root / SALIDA_REL / f"{decision}{args.sufijo}", "results/r9_gui"))
        propia = raiz_tk is None
        if propia:
            try:                                                                    # coordenadas físicas coherentes con la captura en pantallas con escala > 100 %
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:                                                       # noqa: BLE001 - no es Windows o ya está establecido
                pass
            raiz_tk = tk.Tk()
        if args.interfaz == "v2":
            from src.gui.app_tk_v2 import AppTkV2
            clase_app = AppTkV2
        else:
            clase_app = AppTk
        reg = ejecutar(root, destino, raiz_tk, controlador, imagenes_desarrollo=devs, capturas=not args.sin_capturas, tensor_g2=tensor_g2, clase_app=clase_app)
        s = controlador.sistema
        reg.update({"autorizacion": token, "fecha": U.ahora(), "version_interfaz": T.VERSION_PROTOTIPO, "presentacion": args.interfaz, "sistema": SISTEMA_ID,
                    "sha256_declaracion": SHA256_DECLARACION, "umbral": s.t_hat_texto, "n_miembros": len(s.miembros),
                    "imagenes_desarrollo": [str(Path(d).relative_to(root).as_posix()) for d in devs], "imagen_sintetica": "sintetica_01.png (generada; no clínica)",
                    "entorno": {**SF._versiones(), "sistema_operativo": platform.platform(), "dispositivo": str(s.device), "advertencias_entorno": list(s.advertencias_entorno)},
                    "nota": "sin etiquetas, sin métricas de clasificación, sin comparación con el OOF ni con la tolerancia de D-072, sin test; las imágenes de desarrollo fueron vistas en entrenamiento por 12 de los 15 modelos: "
                            "la clase mostrada no es evidencia de desempeño"})
        escribir_json_nuevo(destino / "registro_pruebas_funcionales.json", U.a_json(reg))
        with open(destino / "registro_pruebas_funcionales.md", "x", encoding="utf-8", newline="\n") as f:
            f.write(markdown(reg))
        print(f"\nsalida: {destino}\nSuperados: {reg['superados']}/{reg['total']}")
        return 0 if reg["todos_superados"] else CODIGO_NO_SUPERADA
    except (GuardError, SistemaNoDisponible) as exc:
        print(f"DETENIDO: {exc}", file=sys.stderr)
        return 3
    finally:
        if propia and raiz_tk is not None:
            try:
                raiz_tk.destroy()
            except tk.TclError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
