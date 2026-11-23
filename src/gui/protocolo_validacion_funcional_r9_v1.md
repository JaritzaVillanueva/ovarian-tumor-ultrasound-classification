# R9 — Protocolo de validación funcional de la interfaz (v1) — PROPUESTO

> **Estado:** propuesto el 2026-11-23 (decisión D-076). Queda **congelado** cuando la autora lo versione **antes** de la primera ejecución real en Windows; cualquier enmienda posterior se documenta con fecha, razón e impacto. Conformidad del asesor académico: **PENDIENTE**.
> **Naturaleza:** prototipo académico demostrativo; sin validez diagnóstica ni uso clínico; no es un dispositivo médico. Este documento **no contiene resultados**: la sección 8 está vacía hasta que la autora ejecute el procedimiento.

## 1. Alcance y límites

- Verifica **funcionalidad** (cargar, visualizar, clasificar, gestionar errores) de la interfaz Tkinter con el **sistema congelado** (declaración D-060: DenseNet121 + A, 15 modelos, preprocesamiento de G2, promedio aritmético simple, umbral `0.08456226376195748` con `puntuación >= umbral`).
- **No** repite D-072, D-074, D-075 ni R8; **no** abre el test; **no** calcula métricas de clasificación; **no** lee etiquetas; **no** compara con el OOF ni con la tolerancia de D-072; **no** entrena, recalibra ni cambia el sistema, G2–R8 o el informe.
- Imágenes: una **sintética** generada por el script (no clínica) y dos de **desarrollo** (`train/images`, las dos primeras por nombre; regla sin etiquetas). Las imágenes de desarrollo fueron vistas en entrenamiento por 12 de los 15 modelos: **la clase mostrada no es evidencia de desempeño**.

## 2. Base de evidencia numérica (resumen; detalle en `Reportes/trazabilidad/registro_decisiones.md`)

| Decisión | Evidencia (reportada por la autora, Falcon) | Lo que sostiene | Lo que NO sostiene |
|---|---|---|---|
| D-072 | `integracion-desarrollo`: `exit=5`, diferencia máxima 1.0489821434020996e-3 frente a la tolerancia predefinida 1e-4 | Resultado **negativo conservado**: la integración no superó esa tolerancia | No se reinterpreta ni se supera retrospectivamente |
| D-074 (A–B) | tensores iguales bit a bit (R9 vs G2) en las imágenes examinadas; `B2_dif_maxima_A_B = 0.0`; repetible; D-072 reproducido en los cinco miembros | El preprocesamiento y la inferencia de R9 y de G2 con lote 1 coinciden exactamente | No identifica una operación numérica como causa |
| D-075 (C1) | `C1a = 0.0`, `C1b = 0.0`, `C1b − C1a = 0.0` con el lote histórico de G2 | La disposición histórica de lotes reproduce el OOF con la ruta de G2 **y** con el preprocesamiento de R9 | No amplía a otras imágenes ni a otros tamaños de lote (C2 y TF32 no se ejecutaron) |

**Lectura (hipótesis consistente, no demostración causal):** la diferencia de D-072 depende de las condiciones de procesamiento por lotes; no se identificó una operación concreta. Nada se modificó en checkpoints, preprocesamiento, umbral ni sistema congelado.

## 3. Funcionalidades

- **F1 Cargar:** seleccionar un archivo PNG/JPG; se valida (existencia, extensión, contenido, tamaño, dimensiones; rechazo de rutas con componente `test`) y se convierte a RGB como el lector de G2.
- **F2 Visualizar:** mostrar la imagen cargada y sus datos básicos (archivo, formato, ancho × alto).
- **F3 Clasificar:** obtener la probabilidad de la clase maligna de cada uno de los 15 modelos para **una sola imagen**, promediarlas y aplicar **únicamente** el umbral `0.08456226376195748`; mostrar clase (BENIGNO/MALIGNO), puntuación, umbral, regla, identificación del sistema y avisos.

*(F1–F3 se derivan del alcance del informe «cargar una imagen ecográfica, visualizarla y obtener la predicción» y de los rótulos de `src/gui/app_tk.py` y `tests/gui`. Si la autora las definió de otro modo en el informe, se ajusta esta sección antes de congelar.)*

## 4. Criterio de aceptación funcional propuesto (A)

El criterio evalúa **equivalencia funcional bajo condiciones de inferencia iguales** (misma imagen, mismo preprocesamiento, mismo lote de una imagen, mismo equipo y sesión). **No sustituye la tolerancia de 1e-4 frente al OOF ni modifica D-072.**

| ID | Criterio | Cómo se comprueba | Respaldo |
|---|---|---|---|
| CF-1 | Identidad del sistema: declaración D-060 (SHA-256 `2dd79df4…8cf1`) y **15/15** checkpoints con SHA-256 igual al del manifiesto versionado; umbral exacto; 15 miembros | `src.gui.cli verificar-checkpoints`; comprobación independiente con `Get-FileHash`; caso P00 | D-060, D-072 (`verificar`) |
| CF-2 | **Una imagen por inferencia:** 1 preprocesamiento; 15 inferencias, cada una con tensor de lote 1 (`[1, 3, H, W]`); el **mismo** tensor para los 15; 15 modelos distintos | Sonda en el caso F3 (envuelve `preprocesar` y `predecir`, sin alterar el resultado) | D-074 (lote 1) |
| CF-3 | **Agregación:** puntuación = promedio aritmético simple (float64) de las 15 probabilidades, recomputado de forma independiente: `|Δ| ≤ 1e-12` (solo orden de suma) | Casos F3 | `SF.agregar_y_decidir` |
| CF-4 | **Umbral:** `umbral == 0.08456226376195748` (texto y valor), regla `>=`; la clase mostrada coincide con la regla; ningún otro umbral | Casos F3; pruebas que detectan un umbral ajeno | D-060 |
| CF-5 | **Preprocesamiento equivalente a G2:** el tensor de la interfaz es **igual bit a bit** al de la ruta de lectura de G2 (`DatasetLesiones`) en este equipo, para las imágenes de desarrollo | Casos F3-D1/F3-D2 (`torch.equal`) | D-074 B1 (Falcon), D-075 C1b |
| CF-6 | **Repetibilidad:** dos clasificaciones consecutivas de la misma imagen dan exactamente las mismas 15 probabilidades, puntuación y clase | Casos F3 | D-074 A |
| CF-7 | **Gestión de errores:** entrada inválida → mensaje claro en español, sin inferencia, estado previo conservado, sin clase de otra imagen visible, aplicación operativa | Casos E01–E08 | D-071 |
| CF-8 | **Avisos y modo demostración:** los 6 avisos visibles; con sistema no disponible, «Modo demostración: modelo no disponible», visualización sí, clasificación no, **sin resultados simulados** | Casos E10 y P99 | D-071 (R9.3, opción A) |

**Aceptación:** todos los casos del registro «superada» **y** las capturas de cada caso presentes, no vacías y revisadas visualmente por la autora. **Si un criterio falla:** detener, no ajustar nada del sistema, registrar el hallazgo y decidir (nueva decisión) — un defecto funcional nuevo habilita un análisis acotado; no se amplía el diagnóstico numérico.

**Fuera del criterio (declarado):** la concordancia de las probabilidades con el OOF guardado no es criterio funcional; la equivalencia con la ruta por lotes de G2 se acreditó en Falcon (D-075) y **no se reproduce en Windows**.

## 5. Casos de prueba (los ejecuta `src.gui.evidencia_funcional`)

| ID | Func. | Descripción | Esperado |
|---|---|---|---|
| P00 | estado | Inicio con sistema verificado | 15 miembros, umbral exacto, Clasificar desactivado |
| E01 | error | Clasificar sin imagen | «Primero cargue una imagen ecográfica.» |
| F1-S / F2-S / F3-S | F1, F2, F3 | Imagen sintética PNG: cargar, visualizar, clasificar (botón real, hilo de trabajo) | CF-2…CF-6 |
| E02 | error | Archivo inexistente | «no existe» |
| E03 | error | Archivo `.txt` | «PNG o JPG» |
| E04 | error | PNG vacío | «vacío» |
| E05 | error | PNG corrupto | «dañado» |
| E06 | error | BMP con extensión `.png` | «PNG ni JPG» |
| E07 | error | Imagen de 16×16 px | «Dimensiones no admitidas» |
| E08 | error | Imagen sintética dentro de una carpeta llamada `test` | «componente «test»» (no se abre) |
| F1-D1…F3-D2 | F1, F2, F3 | Dos imágenes de desarrollo tras los errores (recuperación): cargar, visualizar, clasificar | CF-2…CF-6 y CF-5 |
| E10 | demostración | Sistema no disponible (verificación fallida real) | «Modo demostración: modelo no disponible» |
| P99 | avisos | Avisos visibles (desplazados al final) | 6 avisos |

Total: 20 casos. Capturas: `p00_inicio`, `f1_f2_sintetica`, `f3_sintetica`, `e02…e08_error`, `dev_f1_f2_01/02`, `dev_f3_01/02`, `e10_modo_demostracion`, `p99_avisos_final` (las `dev_*` no se versionan por defecto).

## 6. Entorno y advertencias de ejecución

- **Referencia verificada del sistema (Falcon, `entorno_registrado`):** Python 3.10.12, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, numpy 2.2.6, GPU RTX 3080.
- **Windows (previsto):** Python 3.13, PyTorch 2.10.0+cpu, torchvision a instalar, **CPU**. La interfaz mostrará «ATENCIÓN, entorno distinto del verificado» y el registro lo guarda. Las probabilidades individuales pueden diferir numéricamente de las de Falcon; **no se afirma equivalencia numérica con Falcon** ni se compara.
- **Aviso numérico (D):** el procesamiento por imagen individual puede diferir en pequeña magnitud del OOF generado por lotes (en D-072 se observaron diferencias de hasta 1.05e-3 en la probabilidad de un modelo individual; no se evaluó su efecto sobre el promedio de los quince). **No se afirma que tales diferencias no puedan modificar la clase de una imagen cuya puntuación esté próxima al umbral.** La puntuación **no** es una probabilidad calibrada. El umbral se calibró con tres modelos y no está validado para el ensamble de quince. Los 6 avisos de la ventana (incluida esta nota) se muestran siempre.
- Rendimiento citado en la interfaz: evaluación final única de R8 (sensibilidad 0.60 [0.32, 0.84]; 15 positivas; un único subtipo maligno); se reproduce como texto, **no se recalcula**.
- Un error de carga desactiva «Clasificar» y conserva visible la imagen anterior hasta cargar otra (comportamiento intencional; se reinicia la clase mostrada).

## 7. Procedimiento en Windows (PowerShell; lo ejecuta la autora)

Sustituir `<USUARIO@HOST_FALCON>` y `<RUTA_REPO_FALCON>`. Los `.pth` **no se versionan** (`.gitignore` los excluye); si se copian por otra vía, deben quedar en las mismas rutas relativas.

```powershell
# 0. Repositorio y entorno
cd "C:\Users\PC\Desktop\JARITZA - PUCP\26-2\Tesis\Proyecto\ovarian-tumor-ultrasound-classification"
$env:PYTHONUTF8 = "1"
git rev-parse HEAD; git status --short

# 1. Dependencias (PyTorch 2.10.0+cpu ya instalado; solo falta torchvision). --no-deps evita que pip cambie torch.
py -3.13 -m pip install torchvision==0.25.0 --no-deps --index-url https://download.pytorch.org/whl/cpu
py -3.13 -c "import sys,torch,torchvision,PIL,numpy,yaml,scipy,sklearn,tkinter; print(sys.version.split()[0], torch.__version__, torchvision.__version__, numpy.__version__)"
# Si torchvision no importa o no es compatible con torch 2.10.0, DETENERSE e informar (no cambiar torch por cuenta propia).

# 2. Copiar los 15 checkpoints originales desde Falcon (sin versionarlos)
$FALCON = "<USUARIO@HOST_FALCON>"; $REMOTO = "<RUTA_REPO_FALCON>"
foreach ($s in 42,43,44) { foreach ($f in 0..4) {
  $rel = "results/r4_v2/phash_le8_v1/g2_baseline_v1/completa/densenet121__s${s}__f${f}/intento_01"
  New-Item -ItemType Directory -Force -Path $rel | Out-Null
  scp "${FALCON}:${REMOTO}/$rel/checkpoint_mejor.pth" "$rel/checkpoint_mejor.pth"
} }
git status --short            # no debe listar ningún .pth
git check-ignore -v results/r4_v2/phash_le8_v1/g2_baseline_v1/completa/densenet121__s42__f0/intento_01/checkpoint_mejor.pth

# 3. Verificación previa de los 15 SHA-256 (CF-1): con el verificador del proyecto y, de forma independiente, con Get-FileHash
py -3.13 -B -m src.gui.cli verificar-checkpoints --autorizacion CONFIRMADO:D-076
Get-Content configs/r8_v2/manifiesto_miembros_v1.csv | Select-Object -Skip 1 | ConvertFrom-Csv | ForEach-Object {
  $h = (Get-FileHash -Algorithm SHA256 $_.ruta_checkpoint).Hash.ToLower()
  "{0} {1}" -f ($(if ($h -eq $_.sha256_bytes_archivo_checkpoint) {"OK  "} else {"DIST"})), $_.id_corrida }

# 4. Pruebas sintéticas (sin checkpoints reales)
py -3.13 -B -m pytest tests/gui -q -p no:cacheprovider

# 5. Pruebas funcionales con la ventana real y capturas auténticas (no usar el ratón ni tapar la ventana durante ~2–5 min)
New-Item -ItemType Directory -Force -Path results/r9_gui/evidencia_funcional | Out-Null
py -3.13 -B -m src.gui.evidencia_funcional --autorizacion CONFIRMADO:D-076 2>&1 | Tee-Object -FilePath results/r9_gui/evidencia_funcional/D-076.log
"exit=$LASTEXITCODE" | Tee-Object -FilePath results/r9_gui/evidencia_funcional/D-076.exit

# 6. Huellas de los registros generados
Get-FileHash -Algorithm SHA256 results/r9_gui/evidencia_funcional/D-076/registro_pruebas_funcionales.json, results/r9_gui/evidencia_funcional/D-076/registro_pruebas_funcionales.md
Get-ChildItem results/r9_gui/evidencia_funcional/D-076 | Select-Object Name, Length

# 7. Uso interactivo (cargar una imagen, visualizarla, clasificar)
py -3.13 -B -m src.gui.app_tk
```

Códigos de salida del paso 5: `0` todos los casos superados; `6` algún caso no superado; `3` compuerta o sistema no disponible. Para repetir, usar `--sufijo _2` (no se sobrescribe).

## 8. Resultados observados

**No disponible** — se completan únicamente con `registro_pruebas_funcionales.json/.md`, las capturas, el log, el `.exit` y las huellas aportadas por la autora tras ejecutar la sección 7. Hasta entonces, R9.1 y R9.2 **no** se declaran cumplidos.

| Dato | Valor |
|---|---|
| Commit (`git rev-parse HEAD`) | pendiente |
| Entorno efectivo (Python / torch / torchvision / numpy / dispositivo) | pendiente |
| `verificar-checkpoints` | pendiente |
| Casos superados / total | pendiente |
| SHA-256 `registro_pruebas_funcionales.json` | pendiente |
| Capturas revisadas por la autora | pendiente |

## 9. Correspondencia con los indicadores de R9

| Indicador | Evidencia prevista | Estado |
|---|---|---|
| R9.1 La interfaz permite cargar imágenes ecográficas 2D y visualizar los resultados de clasificación | F1–F3 con capturas auténticas; `registro_pruebas_funcionales.*` | pendiente de ejecución |
| R9.2 Las funcionalidades básicas se verifican con pruebas funcionales documentadas | Este protocolo (criterio CF-1…CF-8), 20 casos, pruebas automatizadas (`tests/gui`), registro y capturas | pendiente de ejecución |
| R9.3 El modelo final se presenta como prototipo básico de sistema de apoyo diagnóstico | Opción A (D-071): texto literal del indicador con los calificativos «académico», «demostrativo», «sin validez diagnóstica», «no destinado al uso clínico», «no es un dispositivo médico»; avisos visibles (P99) | implementado; pendiente de captura (P99/P00) |

La terminología «benigno/maligno» se conserva por decisión de la autora (2026-11-21); la interfaz explica siempre el mapeo v2 subyacente (negativa = lesión no maligna, categorías originales 0, 1, 2, 3, 4 y 6; positiva = categoría 7).
