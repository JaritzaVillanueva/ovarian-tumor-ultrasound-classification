# R9 — Propuesta técnica: visualización de atribución (Grad-CAM) exploratoria (v1) — PROPUESTA, NO IMPLEMENTADA

> **Estado:** propuesta del 2026-11-23 (decisión D-078) para revisión de la autora. **No hay código de Grad-CAM** ni ejecución alguna. Se implementará solo tras su aprobación expresa (nueva decisión) y, antes de cualquier uso con modelos reales, con una autorización específica de ejecución. Conformidad del asesor académico: **PENDIENTE**.
> **Naturaleza:** funcionalidad **exploratoria adicional**. **No** es un criterio de R9.1, R9.2 ni R9.3 y no se cuenta como evidencia de ellos.
> **Límites que se mantienen:** sin test; sin entrenar ni recalibrar; sin cambiar checkpoints, arquitectura, preprocesamiento, promedio de quince ni umbral `0.08456226376195748`; sin métricas de clasificación nuevas; Claude Code no ejecuta modelos reales.

## 0. Resumen de la recomendación

1. **Capa:** salida de `features` de DenseNet121 (tras `norm5`) con ReLU, es decir, el tensor `A` de 1024 × 7 × 7 que entra al promedio global y al clasificador lineal.
2. **Puntuación explicada:** el **log-odds de la clase maligna** de cada miembro, `Δ = z₁ − z₀`, porque `p = σ(Δ)` es exactamente la probabilidad que se promedia.
3. **Ensamble:** **atribución del ensamble matemáticamente exacta** (no el mapa de un miembro): descomposición aditiva de la puntuación `S = (1/15) Σ pₘ` en las 7 × 7 celdas, mediante gradientes integrados a lo largo del segmento que escala la contribución de cada miembro. Suma exacta (completitud), sin retropropagación.
4. **Sin alterar la clasificación:** módulo separado, solo lectura, sin ganchos, sin `backward`, sin cambiar el modo ni los parámetros; el resultado mostrado nunca se recalcula.
5. **Verificación:** igualdad de las probabilidades por miembro del explicador con las ya mostradas, identidad de puntuación y clase antes/después, y comprobación de completitud; si algo no cuadra, **no se muestra el mapa** (falla cerrada).
6. **Coste:** 15 pasadas hacia delante (sin gradientes), del mismo orden que «Clasificar»; hilo de trabajo, progreso k/15 y caché por imagen.
7. **Estatus:** visualización **exploratoria**, con la leyenda literal «Mapa de atribución experimental. No demuestra localización anatómica ni validez diagnóstica».

## 1. Capa convolucional (pregunta 1)

DenseNet121 de torchvision (`src/classification/models.py`: `densenet121` con `classifier` sustituido por `nn.Linear(1024, 2)`, sin *dropout*): `features` → `ReLU` → promedio global (`adaptive_avg_pool2d`) → `Linear`. Con entrada de 224 × 224, `features` produce 1024 × 7 × 7 (cada celda cubre 32 × 32 píxeles de la entrada).

- **Propuesta:** `A = ReLU(features(x))`, **sin ReLU en el sitio** (la implementación `inplace` de torchvision modifica el tensor que se quiere observar). El explicador **reproduce** el `forward` (`features` → `relu` → promedio global → `classifier`) y comprueba que sus logits coinciden con los de `modelo(x)`.
- **Por qué esta capa:** es la última antes del clasificador, está directamente ligada a la decisión y permite una descomposición exacta (sección 3). Es la capa estándar de Grad-CAM para DenseNet.
- **Alternativas descartadas:** `denseblock3` (14 × 14, mayor resolución) exige retropropagación a través de `transition3` y `denseblock4`, es más costosa en CPU y su atribución no suma exactamente a la puntuación; se deja como extensión futura, no recomendada para la sustentación.
- **Limitación declarada:** 7 × 7 es una resolución gruesa; los campos receptivos de la última capa cubren gran parte de la imagen.

## 2. Puntuación de clase explicada (pregunta 2)

Para el miembro `m`: `Δₘ = (w₁ − w₀)ₘ · GAP(Aₘ) + (b₁ − b₀)ₘ`, y con dos clases `pₘ = softmax(z)₁ = σ(Δₘ)`. Se explica `Δₘ` (log-odds de MALIGNO) y, en el ensamble, la propia puntuación `S`.

- **Por qué no el logit de una sola clase:** ignoraría la contribución de la clase no maligna; el log-odds es exactamente la cantidad cuya sigmoide se promedia.
- **Por qué `S` y no la clase:** la decisión es `S ≥ t̂`; `S` es continua y es lo que muestra la interfaz.
- **Signo:** el mapa es **con signo**: positivo = la celda aumenta la puntuación del ensamble (hacia MALIGNO); negativo = la disminuye (hacia BENIGNO). Grad-CAM clásico aplica ReLU y oculta la evidencia en contra; se ofrecería la versión «solo positivos» como opción (decisión D1 de la sección 9).

## 3. Cómo representar el ensamble de quince modelos (pregunta 3)

**No se presenta el mapa de un miembro como explicación del ensamble.** Propuesta: una atribución del ensamble con garantía matemática.

Definiciones (todas verificables por código): para el miembro `m` y la celda `(i, j)`,

- `cₘ(i,j) = (w₁ − w₀)ₘ · Aₘ(:, i, j) / (H·W)` ⇒ `Δₘ = bₘ + Σᵢⱼ cₘ(i,j)`, con `bₘ = (b₁ − b₀)ₘ`.
- `pₘ = σ(bₘ + Σ c)`; `S = (1/15) Σₘ pₘ`.
- Gradientes integrados sobre la trayectoria `t ∈ [0,1]` que escala todas las contribuciones (`t = 0`: mapa de características nulo; `t = 1`: el real):
  `pₘ − σ(bₘ) = Σᵢⱼ cₘ(i,j) · κₘ`, con `κₘ = (σ(Δₘ) − σ(bₘ)) / (Δₘ − bₘ)` (si `Δₘ = bₘ`, `κₘ = σ′(bₘ)`).
- **Mapa del ensamble:** `E(i,j) = (1/15) Σₘ κₘ · cₘ(i,j)`.
- **Completitud exacta:** `Σᵢⱼ E(i,j) = S − S₀`, con `S₀ = (1/15) Σₘ σ(bₘ)` (puntuación con características nulas). Se comprueba en cada ejecución (tolerancia 1e-6 por aritmética de punto flotante).

Propiedades y límites que se declaran:

- Los quince modelos tienen la misma arquitectura y reciben **el mismo tensor** de entrada, de modo que sus rejillas 7 × 7 están alineadas espacialmente y sus contribuciones son sumables.
- La atribución es **respecto de las celdas de la última capa**, no de píxeles, y **respecto de una línea base** (características nulas), no respecto del umbral `t̂` ni de «ausencia de lesión».
- Para imágenes con puntuación baja, `S − S₀` suele ser negativo y el mapa será mayoritariamente de signo negativo; por eso la interfaz mostrará `S`, `S₀` y `ΣE` junto al mapa.
- Para un miembro individual, el Grad-CAM clásico equivale a `ReLU(cₘ)` (porque `α_k = ∂Δ/∂Aₖ promediado = (w₁ − w₀)ₖ/(H·W)` exactamente): **no hace falta `backward`**; se contrasta con autograd solo en las pruebas.
- **Alternativas descartadas:** (a) promediar los mapas de los miembros — heurística sin relación con `S`; (b) mapa de un miembro «representativo» — prohibido por la autora y engañoso; (c) promedio de los log-odds — es la atribución de otra cantidad (media de log-odds), no la de `S`. El **mapa de un miembro identificado** podría ofrecerse más adelante, con la leyenda «explica solo a este modelo, no al ensamble»; **no se recomienda** incluirlo ahora.

## 4. Cómo evitar que los gradientes alteren la clasificación (pregunta 4)

- **No se usan gradientes de autograd:** la atribución es analítica (sección 3). No hay `backward`, ni `register_*_hook`, ni `requires_grad_`, ni `train()`; el cómputo va en `torch.inference_mode()`.
- Módulo propio (`src/gui/atribucion.py`), **solo lectura**: lee `sistema.modelos`, `sistema.miembros` y `sistema.preprocesar`; **no** llama a `SistemaCongelado.clasificar`, no modifica `sistema.py`, `controlador.py` ni la ruta de F3.
- `Resultado` es un `dataclass` congelado: la interfaz conserva el objeto mostrado y el explicador recibe una **copia de solo lectura**; el resultado mostrado **no se recalcula** a partir del explicador.
- Comprobaciones: `modelo.training` sigue en `False`, no hay ganchos registrados (`_forward_hooks` vacío), y —en la verificación real— el SHA-256 del `state_dict` de los 15 modelos en memoria es idéntico antes y después.
- Cada clasificación se hace con una sola imagen (CF-2 de D-076); el explicador usa el **mismo** preprocesamiento (`sistema.preprocesar`, reproducible bit a bit, D-074).

## 5. Verificación de que puntuación y clase no cambian (pregunta 5)

1. **En tiempo de ejecución (falla cerrada):** el explicador recalcula `pₘ′ = σ(Δₘ)` y exige `|pₘ′ − pₘ mostrada| ≤ 1e-6` para los 15; exige además `|ΣE − (S′ − S₀)| ≤ 1e-6`. Si no se cumple, **no se muestra el mapa**, se informa del motivo y el resultado mostrado queda intacto.
2. **Interfaz:** se toma una instantánea (clase, puntuación, umbral, 15 probabilidades) antes de generar el mapa y se exige igualdad exacta después; las etiquetas de la ventana no cambian.
3. **Verificación real (a cargo de la autora, con autorización nueva, solo Casos A y B de D-077):** (i) logits del explicador = `modelo(x)` por miembro (igualdad bit a bit esperada; se informa la diferencia máxima); (ii) un miembro: mapa analítico frente a Grad-CAM con autograd (diferencia máxima); (iii) completitud; (iv) `Clasificar` → mapa → `Clasificar`: probabilidades, puntuación y clase idénticas; (v) SHA-256 del `state_dict` antes/después; (vi) tiempos.
4. **Pruebas sintéticas previas (sección 8).**

## 6. Coste computacional en Windows CPU (pregunta 6)

- Solo **pasadas hacia delante** (15, sin gradientes ni almacenamiento de activaciones intermedias): del mismo orden que pulsar «Clasificar» (no ≈3× como con retropropagación). **No hay medición todavía**: se registrará en la verificación real (tiempo de «Clasificar» y de «Visualizar Grad-CAM»); criterio orientativo no vinculante: ≤ 2× el de «Clasificar».
- **Bajo demanda** (solo tras una clasificación), en **hilo de trabajo** con progreso «k/15» y botones desactivados mientras corre; la ventana no se bloquea.
- **Caché** por SHA-256 de la imagen cargada: repetir la visualización no recalcula; se invalida al cargar otra imagen.
- Memoria: por miembro, 1024 × 7 × 7 × 4 B ≈ 200 KB; los 15 modelos ya están cargados por la clasificación.
- Sin dependencias nuevas: NumPy, PIL y torch (ya presentes); el mapa de color se implementa como tabla propia (sin matplotlib).

## 7. Exploratorio frente a localización validada (pregunta 7)

- **Leyenda obligatoria, literal y siempre visible junto al mapa y en el PNG exportado:** «Mapa de atribución experimental. No demuestra localización anatómica ni validez diagnóstica».
- Ubicación: pestaña propia **«Atribución (exploratoria)»**, con rótulo «Experimental»; **no** en la zona principal del resultado ni en la clasificación.
- La interfaz muestra tres paneles: ecografía original, mapa de atribución (escala relativa por imagen, normalizada por `max|E|`, con barra de escala y valores numéricos) y superposición (opacidad 45 %).
- **Qué no se afirma:** que el mapa señale el tumor; que cumpla pruebas de cordura o fidelidad; que sea estable entre imágenes; que los quince modelos coincidan. El texto de la pestaña lista estas limitaciones.
- **Qué haría falta para hablar de localización validada (fuera de alcance):** protocolo previo y congelado; comparación con máscaras de desarrollo (pointing game o masa dentro de la máscara); pruebas de cordura (aleatorización de pesos y de etiquetas); pruebas de fidelidad (eliminación/inserción); análisis por fold. **Se recomienda no hacerlo** para la sustentación: ampliaría la investigación experimental.
- **Geometría:** el preprocesamiento (`ResizeWithPadding`) conserva la relación de aspecto y rellena a 224 × 224; el mapa 7 × 7 se interpola a 224 × 224 (bilineal), se **recorta el relleno** (`delta // 2`) y se escala al tamaño original para la superposición. Probado con imágenes no cuadradas.

## 8. Plan de pruebas (sintéticas, antes de cualquier uso real)

| ID | Prueba | Objetivo |
|---|---|---|
| G1 | Mini-DenseNet de prueba (`features` + `Linear`) con pesos aleatorios: logits del explicador = `modelo(x)` | reproduce el `forward` |
| G2 | Mapa analítico de un miembro = Grad-CAM con autograd (`allclose` 1e-6) | equivalencia de la fórmula |
| G3 | Completitud `ΣE = S − S₀` con 15 miembros de prueba (y caso `Δ = b`) | exactitud de la atribución |
| G4 | Invariancia: `training`, ganchos, `requires_grad`, SHA-256 del `state_dict` y modo de gradiente antes/después | no altera los modelos |
| G5 | `Resultado` y etiquetas de la ventana idénticos antes/después del mapa; nueva clasificación idéntica | no altera la clasificación |
| G6 | Inconsistencia inyectada (probabilidad de un miembro distinta) → no se muestra el mapa | falla cerrada |
| G7 | Imagen no cuadrada con una celda «caliente»: el mapa recortado y reescalado cae en la zona correcta | geometría |
| G8 | Leyenda literal en la ventana y en el PNG; botón desactivado sin clasificación y en modo demostración | presentación |
| G9 | Hilo de trabajo, progreso y caché; sin dependencias nuevas; sin código de métricas ni de entrenamiento | operación y alcance |

## 9. Decisiones que necesito de la autora

- **D1.** Mapa **con signo** (recomendado) o clásico «solo positivos».
- **D2.** Normalización por imagen con `max|E|` (recomendado) o escala fija.
- **D3.** Mostrar solo el mapa del **ensamble** (recomendado) o añadir, más adelante, un selector de miembro con leyenda propia.
- **D4.** ¿Se implementa tras la verificación real de la presentación v2, o antes?
- **D5.** Aprobación de las pruebas reales (A y B de D-077) con una autorización específica.

## 10. Alcance de implementación si se aprueba

`src/gui/atribucion.py` (cálculo puro), pestaña y botón en `app_tk_v2.py`, `tests/gui/test_atribucion.py`, verificador real separado con token propio y directorio de salida nuevo (`dev_*` no versionados), y entrada de trazabilidad. Sin tocar `sistema.py`, `controlador.py`, G2–R8, el informe ni los checkpoints.
