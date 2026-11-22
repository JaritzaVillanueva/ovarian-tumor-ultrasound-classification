# G1 / R2 v2: manifiesto de desarrollo, auditoría de duplicados y propuesta de folds

**Estado:** la auditoría de desarrollo ya se ejecutó en Falcon (D-024; resultados reportados por el autor). **Ruta vigente provisional para los folds: agrupamiento automático por pHash ≤ 8 (`phash_le8_v1`, sección «Agrupamiento automático»)**, autorizada de forma técnica y provisional por el autor (D-025/D-026), con la conformidad del asesor académico **pendiente**. La ruta con revisión humana de 368 parejas (`make_folds_v2.py`, `revision_ciega.csv`) queda **suspendida**, no eliminada. Los folds son solo una **propuesta**: G1 no se cierra hasta ejecutar y verificar el procedimiento y contar con el visto bueno del autor y del asesor.

## Fuente única y mapeo

- **Única fuente de desarrollo:** `configs/splits/cv_folds_e2.csv` (1000 imágenes del train oficial, todas con `source_split = train`). El código **no abre los manifiestos mixtos** (`mmotu_experimental_split*.csv`), que contienen filas del test oficial.
- **Columnas usadas:** `image_id`, `image_name`, `source_split`, `original_class` y `e1_split`.
- **Columnas ignoradas:** `binary_label`, `binary_class` y `fold` son del mapeo histórico. No intervienen en ninguna decisión ni se copian a las salidas.
- **Etiquetas v2, solo desde `original_class`:** 0, 1, 2, 3, 4 y 6 → 0 (lesión no maligna); 7 → 1 (lesión maligna); 5 (ovario normal) excluida.
- **Rutas:** `data/processed/MMOTU_OTU_2D_experimental/<e1_split>/images/<image_name>`. Cada ruta se **resuelve** y debe seguir dentro de `train/images` o `validation/images`. Un enlace simbólico que la redirija, un nombre con separadores o `..`, o un componente `test` abortan el proceso.
- **Recuentos esperados** (se verifican, no se asumen): 1000 en total; 820 elegibles (782 no malignas y 38 malignas) y 180 normales.
- Las salidas ya no incluyen rutas de máscaras. G3 las necesitará desde la fuente correspondiente.

## Requisitos

- Python ≥ 3.10 (`int.bit_count`), Pillow ≥ 9.1 (`Image.Resampling`), numpy, scipy (se importa directamente) y scikit-learn ≥ 1.0.
- Se ejecuta **desde la raíz del repositorio** y **solo en Falcon**, donde están las imágenes.
- No hay archivo de dependencias en el repositorio. Las versiones efectivas de Falcon no están verificadas; el resumen JSON las registra.

## Archivos

- `common.py`: fuente única, guardas contra el test, publicación segura de artefactos y procedencia.
- `build_development_audit.py`: auditoría.
- `make_folds_v2.py`: propuesta de folds con revisión humana (ruta suspendida).
- `make_folds_phash_v1.py` y `protocolo_agrupamiento_phash_v1.md`: agrupamiento automático por pHash ≤ 8 y asignador determinista (ruta provisional vigente).
- `tests/test_r2_v2.py`: pruebas sintéticas de la auditoría y de la ruta con revisión humana.
- `tests/test_r2_phash_v1.py`: pruebas sintéticas del agrupamiento automático.
- `tests/mutation_check_r2_v2.py`: control de mutaciones opcional (no lo recoge pytest). Rompe una guarda por vez en una copia y comprueba que la suite falla: `python tests/mutation_check_r2_v2.py <raiz_del_repo> <directorio_nuevo_fuera_del_repo>`.
- **Dónde escriben los scripts:** solo dentro de `results/r2_v2/` (auditoría), `configs/splits/v2/` (folds con revisión humana) y `configs/splits/v2_auto/` (agrupamiento automático). `--out` fuera de esos directorios, igual a su raíz, o con nombres `.claude`, `agent-memory`, `Reportes` o `Version-Informes` se rechaza, también con `--force`. Esos directorios de agentes y reportes pertenecen a `Proyecto/`, no al repositorio.
- Este README está en `src/r2_v2/` porque `docs/` está en `.gitignore` (ningún archivo de `docs/` está versionado). Las copias previas en `docs/r2_v2/` no se tocaron.

## Pasos

1. **Pruebas:** `python -m pytest -q tests/test_r2_v2.py`
2. **Commit del código y de la fuente.** La auditoría **se niega a ejecutarse** si algún archivo de `src/r2_v2/*.py` o `configs/splits/cv_folds_e2.csv` no está en `HEAD` o difiere de él. `--permitir-codigo-sin-commit` existe solo para pruebas y marca el resumen como no válido para G1.
3. **Auditoría:** `python -m src.r2_v2.build_development_audit`
   - Calcula todo en memoria, lo prepara en un directorio temporal, lo **valida** (hashes, recuentos, mapeo, hojas) y solo entonces lo publica con un renombrado. Si algo falla, no queda nada escrito.
   - Si `--out` ya tiene contenido, se niega salvo con `--force`. Con `--force` el contenido anterior se conserva renombrado (`*.reemplazado-<fecha>`); no lo subas al repositorio.
   - **La planilla `revision_ciega.csv` (con anotaciones) no se sobrescribe nunca, ni con `--force`.** Archívala tú mismo si quieres rehacer la auditoría.
   - **Revisión ciega.** El revisor trabaja solo con `revision_ciega.csv`, que tiene exactamente las columnas `hoja`, `revision`, `revisor` y `nota`, y con las hojas `par_XXXX.png` (orden aleatorio; la imagen no muestra distancia ni origen). La planilla **no revela** distancia pHash, categoría original ni etiqueta binaria. La correspondencia hoja → pareja está **aparte**, en `candidate_pairs.csv` y `calibration_pairs.csv` (con la distancia, las categorías y las etiquetas): el revisor no debe abrirlos. Son un archivo de correspondencia por origen (candidatos y calibración), no se editan y su **hash exacto** queda en el resumen.
   - La carpeta `review_sheets/` lleva su propio `.gitignore` (se escribe antes que las hojas), porque contiene ecografías.
   - `resumen_auditoria.json` registra `codigo_en_commit`, el commit, las versiones, el comando, la huella de la fuente, la huella del conjunto de hojas de la planilla y los hashes de las salidas.
4. **Revisión humana.** En `revision_ciega.csv`, abre la hoja indicada en `hoja` y llena `revision` con `misma`, `distinta` o `duda`. Llena también `revisor` y `nota`. **Ninguna hoja puede quedar vacía ni en `duda`.** No agregues, quites, repitas ni renombres filas.
   - Los indicadores `categorias_originales_distintas` y `etiquetas_contradictorias` (en los CSV de correspondencia, no en la planilla) son **independientes**. El primero es informativo (p. ej. 0 y 6 son categorías distintas pero la misma etiqueta binaria); el segundo compara `label_v2`.
5. **Parámetros.** Aprobados por el autor para la **auditoría inicial** (cribado, no prueba de identidad): corte pHash ≤ 12 y banda adicional de calibración 13–18. **Siguen siendo propuesta, sin congelar la partición:**
   - 5 folds con semilla 42;
   - mínimo de 4 positivas por fold (por debajo, error);
   - rango de 6 a 9 positivas y desviación de tamaño del 15 % respecto de 164 (fuera de ellos, solo advertencia).
   Regístralos en la trazabilidad antes de generar folds (`--aprobacion` lo exige).
6. **Commit** de la auditoría y de la revisión (sin `review_sheets/`).
7. **Propuesta de folds:** `python -m src.r2_v2.make_folds_v2 --aprobacion "D-022 2026-11-21"` (D-022 es la entrada de la trazabilidad con los parámetros; recuerda que solo aprueba corte y banda del pHash, y que los parámetros de folds siguen como propuesta)
   - Se detiene, sin escribir nada, si falta un archivo de la auditoría o la columna `revision`; si queda un par sin revisar, en `duda` o sin `revisor`; si algún artefacto, **incluida la correspondencia hoja → pareja, fue alterado (hash exacto)**; si la planilla no corresponde uno a uno con esa correspondencia (hojas añadidas, quitadas, repetidas o renombradas); si un par tiene revisiones distintas en los dos orígenes; o si falta `--aprobacion`. Solo entonces traduce cada revisión a su pareja.
   - **El resumen de la auditoría no se toma como verdad por sí mismo:** exige que declare el hash de **cada** artefacto esperado (no solo de los que él lista), que `cv_folds_e2.csv` actual coincida con la fuente auditada y que su propia procedencia registre código y fuente en commit (no basta el booleano `codigo_en_commit` autodeclarado; pero las banderas `en_commit` por archivo dentro de la procedencia siguen siendo autodeclaradas, porque el resumen no está anclado a git). Si el código de la auditoría cambió después de ejecutarla, solo **advierte** (comparando ids de blob de git) y lo registra. Registra el SHA-256 del resumen: anótalo en la trazabilidad.
   - Une automáticamente los duplicados exactos por SHA-256. Se detiene si un duplicado o un par `misma` une etiquetas v2 distintas.
   - Nunca separa un grupo verificado. Se detiene si un fold queda con menos de 4 positivas.
   - Se niega a sobrescribir una propuesta existente salvo con `--force` (que la conserva renombrada).
8. **Revisión de la propuesta y visto bueno del autor y del asesor.** Solo entonces se versiona como congelada y se cierra G1. El código no congela nada por sí mismo.

## Agrupamiento automático por pHash ≤ 8 (`phash_le8_v1`) — ruta provisional vigente

Protocolo completo: [`protocolo_agrupamiento_phash_v1.md`](protocolo_agrupamiento_phash_v1.md). **Debe estar en un commit antes de la primera ejecución.**

**Qué es.** Una medida de mitigación de fuga de información: las imágenes conectadas por distancia pHash ≤ 8 (más duplicados exactos) forman grupos que **nunca** se separan, y un asignador **determinista, sin aleatoriedad (no invoca scikit-learn; importarlo requiere tenerlo instalado)**, reparte los grupos en cinco folds procurando equilibrar las positivas.

**Qué NO es.**
- **No demuestra identidad ni independencia por paciente.** No hay identificadores de paciente: imágenes de la misma paciente con pHash distinto pueden caer en folds distintos.
- **El umbral ≤ 8 es exploratorio y post hoc:** se eligió después de observar datos de desarrollo y su criterio usó las etiquetas. No está validado externamente.
- **No hay validación humana** de los grupos ni de las 69 parejas ≤ 8 (decisión del autor, D-026); la precisión de pHash ≤ 8 como detector de casi duplicados **no se ha medido**.
- **Riesgo residual:** parejas con distancia mayor que 8 (p. ej. 9–12 y las de calibración 13–18) **pueden quedar en folds distintos**. Las estimaciones pueden ser optimistas en una magnitud desconocida. El resumen cuenta cuántas quedan separadas (transparencia; **no se usa para ajustar nada**).
- Los valores observados en el diagnóstico exploratorio (69 pares, grupo máximo de cuatro, cero grupos mixtos) **no son criterios de aceptación**.

**Qué lee y qué verifica.** Solo CSV y JSON de la auditoría: no abre imágenes, no toca el test, no lee ni modifica `revision_ciega.csv` y no escribe en el directorio de la auditoría. Bloquea, sin publicar nada, si algún artefacto no coincide con el hash declarado en el resumen; si la fuente cambió; si la auditoría se ejecutó con código sin commit; o si las distancias **recalculadas** desde `image_hashes.csv` no reproducen exactamente `candidate_pairs.csv` y `calibration_pairs.csv`. Lee los CSV de correspondencia (con distancias): **no lo mezcles con una revisión humana ciega**.

**Bloqueos del propio procedimiento.** Un **grupo mixto** (lesiones no malignas y malignas) detiene la generación y se reporta: no se divide ni se corrige ninguna etiqueta; la decisión se documenta por escrito. Un fold con **menos de 4 positivas** es un error («Partición no viable»); fuera de 6–9 positivas o con una desviación de tamaño > 15 % de 164 solo se advierte. Antes de publicar se relee la salida y se revalida (grupos, aristas, etiquetas, hashes).

**Ejecución (solo el autor, en Falcon, tras el commit; no se ha ejecutado):**

```bash
# desde la raíz del clon, con el código y el protocolo ya en HEAD
python -m pytest -q tests/test_r2_v2.py tests/test_r2_phash_v1.py
# --resumen-sha256: el SHA-256 de results/r2_v2/auditoria_desarrollo/resumen_auditoria.json registrado en la trazabilidad (D-024)
python -m src.r2_v2.make_folds_phash_v1 --aprobacion "D-026 2026-11-21" --resumen-sha256 <SHA-256 registrado>
```

Salidas en `configs/splits/v2_auto/`: `cv_folds_phash_le8_v1.csv`, `aristas_phash_le8_v1.csv` y `resumen_phash_le8_v1.json`. Se niega a sobrescribir salvo `--force` (que conserva lo anterior renombrado). **Raíz de confianza:** `--resumen-sha256` ancla el resumen de la auditoría; sin él se rechaza el comando. `image_hashes.csv` no es verificable sin imágenes (solo su coherencia con los demás artefactos). Las cifras de la auditoría real (368 hojas, 308 + 60 parejas, 69 parejas ≤ 8) son **reportadas por el autor** y no se pueden verificar desde el repositorio, porque `results/r2_v2/` no está versionado. Tras la ejecución: revisar el resumen, anotar su SHA-256 en la trazabilidad y obtener el visto bueno del autor y del asesor **antes** de congelar nada.

## Qué se verificó (evidencia disponible)

- **Pruebas sintéticas (`tests/test_r2_v2.py`):** en entornos aislados locales con Windows, Python 3.12.10, numpy 2.5.5, scipy 1.18.1, Pillow 13.0.0 y pytest 9.2.0, pasan **las 105, 0 saltadas, con scikit-learn 1.9.1 y con 1.7.2** (la versión de Falcon). Las pruebas no dependen del estado de Git (un `_git` simulado se fija por defecto en cada prueba) ni de la distribución exacta que produce `StratifiedGroupKFold` (las pruebas de extremo a extremo usan un asignador determinista de prueba; el `assign_folds` real se prueba por sus invariantes).
- **Historial de compatibilidad:** la versión anterior de las pruebas dio **92 aprobadas y 8 fallidas en Falcon** (Python 3.10.12, scikit-learn 1.7.2; resultado informado por el autor). Se reprodujo localmente con scikit-learn 1.7.2 (los mismos 8) y se aisló la causa: **4 fallos dependían del Git real** (el código ya estaba en commit; fallan también con scikit-learn 1.9.1) y **4 de la distribución que produce scikit-learn** (siguen fallando con el Git aislado); ninguno dependía de ambas cosas. La versión corregida **no se ha ejecutado aún en Falcon**.
- **Control de mutaciones (`tests/mutation_check_r2_v2.py`).** Cada mutación rompe una guarda en una copia y la suite debe fallar. El script da por detectada toda mutación que hace fallar la suite; **no comprueba qué prueba falló**. En la copia, una prueba (`test_real_git_runner_on_real_repository`) se salta porque la copia no es un clon de git; en el repositorio no se salta.
  - **Pruebas corregidas por compatibilidad, scikit-learn 1.7.2: 30 de 30 detectadas**, 0 sobrevivientes, 0 errores (corrida completa de las 30 mutaciones definidas en ese momento). La mutación 31, `[no viable]` (quitar el bloqueo "Partición no viable"), se añadió después junto con su prueba determinista y se ejecutó sola: detectada. Total definido hasta ese momento: **31**; la corrida conjunta de las 31 no se hizo.
  - **Versiones anteriores:** 25 de 25 antes de la planilla ciega y 10 de 10 de las afectadas por ella, sobre las pruebas previas al parche de compatibilidad. Quedan como historial; no se citan como estado actual.
  - Una mutación (`PROTECTED_FILES = ()`) muestra «2 saltadas»: vacía el parámetro de una prueba parametrizada y pytest la salta.
- **Fuente real, solo CSV:** una prueba lee `cv_folds_e2.csv` y obtiene 1000, 820, 782, 38 y 180, con la categoría 5 como único normal y la 6 como no maligna. No abre imágenes.
- **Estado de versionado hoy:** `fuente_en_commit` es verdadero. `codigo_en_commit` es falso porque `src/r2_v2/*.py` aún no está en un commit.

- **Agrupamiento automático (`tests/test_r2_phash_v1.py`): 50 pruebas sintéticas pasan, 0 saltadas**, con scikit-learn 1.7.2; junto con las de `tests/test_r2_v2.py` son **155** en total. No dependen de Git externo y no invocan scikit-learn. Cubren: asignador determinista, independiente del orden de entrada y que nunca separa grupos ni cadenas; bloqueo de menos de 4 positivas; detención ante grupo mixto y ante duplicado exacto contradictorio; hashes de las entradas, ancla del resumen, recuentos recalculados, cruce con la fuente y recomputación de distancias (incluidas falsificaciones coherentes de un archivo); ninguna lectura de imágenes, de `revision_ciega.csv` ni de test; la auditoría queda byte a byte intacta; `--out` restringido y sin sobrescritura; resultado fijo de regresión.
- **Mutaciones del agrupamiento automático (16 definidas, scikit-learn 1.7.2, código final):** en la corrida conjunta se detectaron **15 de 16**. La 16.ª (`--resumen-sha256` obligatorio) **sobrevivió**: sin el control de formato, un ancla vacía igualmente se bloqueaba por no coincidir con el hash real, y solo cambiaba el mensaje. Se afinó la prueba para exigir el mensaje «Falta --resumen-sha256» y esa mutación se repitió **sola**: detectada. La corrida conjunta de las 16 con la prueba final no se repitió. Las otras 31 mutaciones definidas (auditoría y ruta con revisión humana) no se repitieron en esta tanda porque sus guardas y sus pruebas no cambiaron (total en el script: 47).

## No verificado

- No se ejecutó con las imágenes reales, ni en Falcon, ni con Python 3.10 (no hay intérprete 3.10 local; la gramática sí se comprobó con `ast`).
- No se sabe cuántos pares candidatos habrá con d ≤ 12 ni cuántas hojas habrá que revisar.
- Las versiones de las librerías en Falcon.

## Límites

- El pHash controla la **dependencia visual detectable**, no la dependencia por paciente: dos imágenes de la misma paciente pueden tener hashes muy distintos. `calibration_pairs.csv` es la muestra de control fuera del corte.
- El umbral del pHash usa la mediana sin el coeficiente DC. Es una variante de Zauner (2010) [CITA POR VERIFICAR] y sus valores no son intercambiables con `imagehash.phash`.
- **Guarda de rutas.** Cada imagen debe resolverse dentro de `train/images` o `validation/images`: se rechazan nombres con separadores o `..`, un componente `test` (sin distinguir mayúsculas), un enlace simbólico o unión que redirija esos directorios y un archivo-enlace que resuelva fuera de ellos. **Límite:** si `data/processed/MMOTU_OTU_2D_experimental` completo fuera un enlace, se resuelve y se acepta (no se compara su destino), y no se puede detectar un enlace a datos de test bajo un nombre sin `test` sin resolver rutas del test.
- La suite no ejercita un enlace a archivo ni un enlace en el padre `train/`; la lógica los cubre, pero no hay prueba que los ejecute.
- `archivos_sin_commit` del resumen puede listar hasta 50 rutas del árbol de trabajo: revísalo antes de versionar el resumen.
- Pendientes menores del segundo informe del auditor, sin corregir: constantes `manifiestos_mixtos_abiertos` y `rutas_resueltas_verificadas` fijadas en el código (no medidas); `--aprobacion` es texto libre.
- **Límite de la revisión ciega:** la planilla no revela nada de los pares, pero la ceguera depende de que el revisor no abra los CSV de correspondencia. El código no puede impedirlo. Las hojas son imágenes: dos hojas pueden parecer más o menos similares y eso no se oculta.
- Corte y banda del pHash están aprobados solo para la auditoría inicial. Folds, semilla, mínimos y rangos son **propuestos**, no validados estadísticamente.
- Con 38 positivas, 5 folds dejan unas 7–8 por fold. Es una limitación que hay que declarar.
- **El reparto de folds depende de la versión de scikit-learn.** `StratifiedGroupKFold` es una heurística: nunca separa un grupo (comprobado en ambas versiones), pero **no garantiza el equilibrio de positivas**, y ese equilibrio cambia entre versiones. Medición **sintética** (no es un resultado sobre los datos reales): en 200 escenarios con 820 imágenes y 38 positivas, positivas y parejas agrupadas al azar y semilla 42, scikit-learn 1.9.1 no dejó ningún fold con menos de 4 positivas (mínimo medio 7,00; máximo medio 8,00), mientras que 1.7.2 dejó algún fold por debajo de 4 en el **16 %** de los escenarios (mínimo medio 4,64; máximo medio 10,70; peor caso 15 positivas en un fold). Consecuencias: con 1.7.2 la generación de folds puede **bloquearse de forma segura** ("Partición no viable") o producir folds muy desiguales (p. ej. 4 y 13 positivas). El mínimo de 4 se mantiene y no se cambió ninguna semilla ni regla para evitarlo. Esto afecta a la ruta con revisión humana (`make_folds_v2.py`, suspendida), que sigue usando `StratifiedGroupKFold`; la versión usada queda registrada en el resumen. **La ruta provisional vigente (`make_folds_phash_v1.py`) ya no depende de ese reparto**: usa un asignador propio, determinista y sin aleatoriedad, autorizado de forma técnica y provisional (D-026).
- La clase maligna es un solo subtipo (categoría 7). Los resultados no demuestran generalización a todos los cánceres ováricos.
