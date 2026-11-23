# Protocolo R7 `sistema_final_v1`: sistema final provisional (candidato) — manifiesto de miembros y fidelidad

| Campo | Estado |
|---|---|
| Estado | **Candidato; NO congelado formalmente; calibrado con OOF de tres modelos por imagen; no validado para el ensamble de quince (D-052).** Su commit y su `sha256_contenido_blob` se registran en `configs/r7_v2/sistema_final_v1.yaml` (`protocolo:`) en un commit posterior; este archivo no contiene su propio hash. No autoriza ejecutar sobre checkpoints reales |
| Leyenda | partición provisional (`phash_le8_v1`, exploratoria, post hoc); conformidad académica pendiente. Mapeo v2. Solo desarrollo; test cerrado (K4); 180 normales fuera |
| Código | `src/r7_v2/sistema_final.py`; configuración `configs/r7_v2/sistema_final_v1.yaml` (bloque `sistema`, validado por código contra constantes y contra este texto) |

## Diseño provisional del sistema
DenseNet121, representación A, 15 checkpoints (semillas 42/43/44 × folds 0–4) de G2, intento `completada` de `results/r4_v2/phash_le8_v1/g2_baseline_v1/completa/`. Agregación: promedio aritmético simple de las 15 probabilidades de la clase maligna (softmax float32 por modelo; promedio en float64), sin ponderaciones aprendidas ni selección de miembros. Regla: `p >= t_hat` con `t_hat = 0.08456226376195748` (cadena exacta en el YAML: `repr(float(cadena))` la reproduce); los empates no se rompen. S\* = 0.90 es el objetivo primario, exploratorio, no una garantía. Incertidumbre de referencia del umbral (bootstrap, 10 000 remuestreos válidos): p2.5 = 0.022556618166466553, mediana = 0.08456226376195748, p97.5 = 0.20131495470801988; solo informativa, no es validación.
Alcance de decisión: clasificación binaria de lesión a nivel de imagen 2D; no detecta ovarios normales; sin uso clínico validado.

## Lo que NO es
t\_hat se calculó en desarrollo con OOF de tres modelos por imagen: es candidato, NO congelado formalmente y no es el umbral de R8. El sistema de quince no tiene OOF de desarrollo (12 de los 15 modelos vieron cada imagen): no validado para el ensamble de quince, y por eso NINGUNA métrica (AP, AUC, umbral, sensibilidad, especificidad) del ensamble de quince se calcula sobre imágenes de desarrollo; el código lo impide con una guarda.

## Modos (solo `manifiesto` y `fidelidad`; no hay modo de inferencia)
Ambos exigen `--autorizacion CONFIRMADO:D-NNN` (sin ella no se lee ni se escribe), `seleccion_r6.json` con elegida A|densenet121 (ausencia o distinta bloquea), integridad propia (HEAD, árbol limpio, SHA-256, protocolo versionado con el hash registrado en el YAML), K4 y directorio nuevo (nunca sobrescribe). Salidas en `results/r7_v2/phash_le8_v1/sistema_final_v1/{manifiesto,fidelidad}/`.
**`manifiesto`**: `manifiesto_miembros.csv/json` con los 15 miembros únicos (arquitectura, semilla, fold, id_corrida, intento, ruta relativa del checkpoint, SHA-256 del checkpoint recalculado y comparado con el registrado en el `run_config`, época elegida, estado, `invalida_integridad`, commit de entrenamiento, SHA-256 del `run_config` y de `predicciones_oof.csv`, de la partición, blob de la configuración de G2 y referencia de pesos), los SHA-256 del código de preprocesamiento (`transforms`, `dataloader`, `models`, `evaluate`, `dataset_lesiones`, `entrenamiento`), los parámetros de entrada de la receta de G2 (224 con relleno, normalización; pin del blob de su YAML) y las versiones efectivas. Bloquea ante faltantes, duplicados, otro modelo, hash distinto, intento no completado o comparabilidad rota.
**`fidelidad`**: cada miembro, solo sobre las imágenes de desarrollo de su propio pliegue, debe reproducir las probabilidades guardadas de su `predicciones_oof.csv` con el preprocesamiento de G2 (se reutiliza `get_val_transforms` y el dataset de G2, sin reimplementar). Reporta por miembro la diferencia máxima absoluta y los cambios de clase a 0.5 y a `t_hat`, con la tolerancia de recarga de G2 (`receta.recarga_checkpoint.alerta_diferencia_maxima_absoluta`; alerta, no corrección). No agrega miembros ni calcula métricas.

## Biblioteca para R8
`inferir` (promedio de 15 y regla `>=`) existe solo como función de biblioteca: no hay CLI; K4 rechaza rutas con componente `test` y filas de desarrollo con etiquetas. R8 es un protocolo aparte, con su propio ejecutor autorizado: este protocolo no lo actualiza ni abre el test; no se afirma independencia absoluta del test histórico.
