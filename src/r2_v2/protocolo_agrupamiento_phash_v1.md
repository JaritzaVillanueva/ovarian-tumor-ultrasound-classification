# Protocolo `phash_le8_v1`: agrupamiento automático por similitud visual y asignación determinista de 5 folds (G1 / R2 v2)

**Estado:** aprobación técnica **provisional** del autor (2026-11-21; D-025 y D-026 en la trazabilidad). **Conformidad del asesor académico: pendiente.** Los folds que genere este procedimiento son una **propuesta**: no se congelan sin ejecutarlo, verificarlo y contar con el visto bueno del autor.

Este archivo debe estar en un commit **antes** de la primera ejecución. El script se niega a ejecutarse si no lo está (o si difiere del commit), salvo `--permitir-codigo-sin-commit`, que solo es para pruebas.

## 1. Propósito y límites

- **Propósito:** reducir la fuga de información entre folds por imágenes visualmente casi idénticas. Es **solo una medida de mitigación**.
- **No demuestra identidad ni independencia por paciente.** No hay identificadores de paciente. Dos imágenes de la misma paciente pueden tener pHash muy distinto y caer en folds distintos.
- **Sustituye provisionalmente** la revisión manual de pares (antes obligatoria en `CLAUDE.md` §5). **No hay validación humana de los grupos.** La precisión de pHash ≤ 8 como detector de casi duplicados **no se ha medido**.
- **El umbral es exploratorio y post hoc.** Se eligió después de observar datos de desarrollo, y el criterio de elección (ausencia de grupos con etiquetas mixtas y tamaño máximo de grupo) **usó las etiquetas de desarrollo**. No está validado externamente.
- **Riesgo residual:** las parejas con distancia pHash mayor que 8 (p. ej. 9–12 y las de calibración 13–18) **pueden quedar en folds distintos**. Las estimaciones de rendimiento pueden ser optimistas en una magnitud desconocida. El resumen cuenta cuántas quedan separadas, solo como transparencia: **ese recuento no se usa para ajustar nada**.
- Con 38 positivas y 5 folds hay unas 7–8 por fold: incertidumbre alta por fold.

## 2. Entradas (solo lectura)

Archivos de la auditoría de D-024, en `results/r2_v2/auditoria_desarrollo/`: `development_eligible_v2.csv`, `image_hashes.csv`, `exact_duplicates.csv`, `candidate_pairs.csv`, `calibration_pairs.csv` y `resumen_auditoria.json`.

- **No se abren imágenes, máscaras ni rutas de test.** No se lee ni se modifica `revision_ciega.csv`, ni se tocan las hojas.
- Este procedimiento **lee los CSV de correspondencia** (con distancias), por lo que **no debe mezclarse con una revisión humana ciega**.

## 3. Verificaciones de integridad y reproducibilidad (bloquean)

Cualquier discrepancia **detiene el proceso, sin publicar nada**, y debe explicarse antes de continuar:

1. El SHA-256 de `resumen_auditoria.json` coincide con el **anclado** con `--resumen-sha256` (el valor registrado en la trazabilidad, D-024): sin ese ancla, una falsificación coherente de resumen y artefactos no se detectaría. El resumen declara el SHA-256 de cada entrada y coincide con el archivo actual. `image_hashes.csv` no es verificable sin imágenes: solo se comprueba su coherencia con los demás artefactos.
2. Los recuentos de la auditoría coinciden con los esperados de G1 (1000 / 820 / 782 / 38 / 180), se **recalculan** desde las filas de `development_eligible_v2.csv` y los elegibles y los normales se cruzan con la fuente de desarrollo `configs/splits/cv_folds_e2.csv` (solo CSV; ni identificadores ni categorías pueden diferir).
3. `cv_folds_e2.csv` actual coincide con la fuente auditada; la procedencia de la auditoría registra código y fuente en commit.
4. Las distancias de Hamming se **recalculan** desde `phash_hex` de `image_hashes.csv` para todas las parejas de lesiones elegibles. El conjunto de parejas con distancia ≤ el corte de la auditoría y sus distancias **deben coincidir exactamente** con `candidate_pairs.csv`, y las parejas de calibración deben coincidir en distancia y estar por encima del corte.
5. Las categorías de `image_hashes.csv` coinciden con las de los elegibles; las etiquetas v2 se recalculan **solo** desde `original_class`.

Los valores observados en el diagnóstico exploratorio (**69 pares, grupo máximo de cuatro imágenes, cero grupos mixtos con pHash ≤ 8**) son **informativos**: **no** son criterios de aceptación. El resumen los registra como observados.

## 4. Procedimiento

1. **Aristas:** cada pareja de lesiones elegibles con distancia pHash **≤ 8** y los duplicados exactos (SHA-256 idéntico).
2. **Grupos:** componentes conexas (cierre transitivo). `group_id` es el menor `image_id` del grupo.
3. **Grupo mixto** (con lesiones no malignas y malignas): **se detiene la generación** y se reporta para decisión documentada. **No se dividen grupos ni se corrigen etiquetas.**
4. **Asignador determinista, sin aleatoriedad; no invoca scikit-learn** (importar `make_folds_v2.py` requiere tenerlo instalado, pero no se llama a ninguna de sus funciones):
   - Orden de grupos con positivas: (positivas ↓, tamaño ↓, menor `image_id` ↑). Cada uno va al fold con **menos positivas** (desempate: menos imágenes, luego menor índice de fold).
   - Después, los grupos sin positivas, por (tamaño ↓, menor `image_id` ↑), al fold con **menos imágenes** (desempate: menor índice).
   - No hay semilla: el resultado depende solo de los grupos y las etiquetas. **No se elige la partición según el rendimiento de ningún modelo.**
5. **Validación de la salida:** ningún grupo ni arista queda separada; todas las imágenes tienen fold; **error** si algún fold queda con menos de **4 positivas** («Partición no viable»); advertencia si las positivas por fold salen de **6–9** o el tamaño se desvía más del **15 %** de 164.
6. **Publicación segura:** se preparan los archivos en un directorio temporal, se **revalidan** leyéndolos de nuevo y solo entonces se publican con un renombrado. Se niega a sobrescribir salvo `--force` (conserva lo anterior renombrado).

## 5. Parámetros (fijos en el código; no se exponen por línea de comandos)

| Parámetro | Valor | Estado |
|---|---|---|
| Umbral pHash | ≤ 8 | exploratorio, post hoc, no validado |
| Folds | 5 | propuesto |
| Mínimo de positivas por fold | 4 (error) | propuesto, no prueba de validez estadística |
| Intervalo de positivas (advertencia) | 6–9 | propuesto |
| Desviación de tamaño (advertencia) | 15 % de 164 | propuesto |

## 6. Salidas (en `configs/splits/v2_auto/`, distinto de `configs/splits/v2/` de la ruta con revisión humana)

- `cv_folds_phash_le8_v1.csv`: lesiones elegibles con `group_id`, `group_size` y `fold`.
- `aristas_phash_le8_v1.csv`: aristas con su distancia y origen (`phash<=8` o `sha256_identico`).
- `resumen_phash_le8_v1.json`: parámetros, hashes de entradas y salidas, procedencia (commit, versiones), valores observados, advertencias, diagnóstico de riesgo residual y las limitaciones de este protocolo.

Los grupos son **automáticos**: se identifican por el nombre de los archivos, por la columna `origen` de las aristas y por el resumen; nunca se escriben como revisiones humanas ni en `revision_ciega.csv`. El diagnóstico de calibración del resumen es una **muestra** (hasta 60 parejas), no la población.

## 7. Qué no hace

No abre imágenes ni test, no entrena, no modifica la auditoría, no invoca scikit-learn y no genera folds definitivos. No declara cerrada G1.
