# Protocolo G3a `g3_segmentacion_v1`: segmentador U-Net por fold y máscaras OOF

| Campo | Estado |
|---|---|
| Estado | **Candidato a congelamiento por la autora.** Su commit y `sha256_contenido_blob` se registran en `configs/r5_v2/g3_segmentacion_v1.yaml` (`protocolo:`) en un commit posterior; este archivo no contiene su propio hash. No autoriza entrenar: `humo` y `completa` exigen autorización expresa |
| Leyenda | partición provisional (`phash_le8_v1`, exploratoria, post hoc); conformidad académica pendiente. No se afirma independencia por paciente |
| Configuración | `configs/r5_v2/g3_segmentacion_v1.yaml` (todos los parámetros) · código `src/r5_v2/` · fuente: `Reportes/asesorias/borrador_protocolo_G3_G4_v0.md` §3, H3, H4, §13 |
| Datos | Las 820 elegibles de `phash_le8_v1` (K1–K4 de G2, YAML de G2 en solo lectura). Solo `train/` y `validation/` de desarrollo. Test cerrado; 180 normales (categoría 5) fuera |

## Alcance
Entrenar 5 U-Net (uno por fold) y producir una máscara fuera de fold por imagen (820). Calcula Dice, IoU, cobertura de la caja ROI y fallback por imagen y un resumen descriptivo sin IC ni interpretación. **No genera recortes** (G3b), no clasifica, no usa test.

## Decisiones
1. U-Net `unet__s42__f{k}` por fold k∈{0..4}, entrenado solo con folds ∉ {k, v}; v=(k+1) mod 5 es el fold interno (solo elige el checkpoint). El fold k no interviene en entrenamiento ni selección; sus máscaras de referencia no entran al modelo: la predicción no las usa y las métricas las leen después de guardar las predicciones (`verificar_datos` las abre antes solo para comprobarlas y bloquear). El predictor recarga el checkpoint y comprueba segmentador, fold, época y `val_dice` contra la época elegida.
2. Receta heredada del notebook `01_unet_baseline` (solo celdas de código) y `src/segmentation/unet.py`: Adam lr 1e-4, `BCEWithLogitsLoss`, lote 4, 256×256 (imagen bilineal, máscara vecino más próximo), umbral 0.5, 50 épocas, sin aumento, semilla 42 en los cinco. Checkpoint = época de **máximo estricto** del Dice medio por imagen (eps 1e-7) en el fold v (empate: la más temprana); sin parada anticipada. El notebook coincide con la especificación; difiere en la validación (fold v) y en que el checkpoint se escribe una vez y sin estado del optimizador.
3. Máscara de referencia binarizada con `> 0` (el dataset histórico no binariza).
4. Máscaras OOF `<image_id>_pred.png` 256×256 en {0,255} bajo `data/processed/r5_v2/phash_le8_v1/mascaras_pred_oof/<modo>/fold{k}/intento_NN/` (derivado, no versionado); manifiesto (`image_id, fold, segmentador, sha256_bytes_checkpoint, sha256_mascara, epoca_elegida, ruta_mascara`) bajo `results/r5_v2/phash_le8_v1/g3_segmentacion_v1/`.
5. Por fold: `unet_fold{k}_best.pth` (SHA-256 en `run_config.json`), historial por época, `run_config.json` (commit HEAD, SHA-256 de partición/YAML/código, versiones, semilla, IDs por rol con hash y conteos). Intentos nuevos; un fold completado no se repite.
6. ROI (solo caja): `src/segmentation/roi.py` sin editar; umbral 0.5, máscara predicha reescalada al tamaño original (vecino más próximo), mayor componente conexa, margen 0.10, fallback a imagen completa si el área < 0.001. Misma función para la máscara de referencia. Parámetros en el YAML, **provisionales**.
7. `metricas_seg_por_imagen.csv` y `resumen_metricas_seg.json` (media/mediana de Dice e IoU, por fold, n con Dice<0.1, cobertura media/mínima, n<0.5, tasa de fallback con las filas de fallback aparte). Dice/IoU y áreas a 256×256; cobertura a resolución original.

## Modos
`verificar_datos` (solo lectura, bloquea por máscara ausente, vacía, de tamaño distinto o con valores que no sean {0,1}/{0,255}) → `humo` (1 fold, 2 épocas, no cuenta) → `completa` (5×50; exige `humo` completado con el mismo commit, YAML y partición). `humo` y `completa` exigen `--autorizacion-ejecucion CONFIRMADO:AUT-NNN|D-NNN` y árbol Git limpio.

## Invariantes de no-fuga (verificados por código y pruebas)
(entrenamiento ∪ validación interna) ∩ fold k = ∅; roles disjuntos con cobertura exacta; cada imagen se predice una vez, con el segmentador de su fold; rutas con componente `test` rechazadas; ninguna categoría 5.

## Lo que este protocolo NO resuelve
B-híbrida frente a B-original · `keep_largest` de la condición C (la C histórica usaba `false` y sin fallback) · IC de Dice/IoU (bootstrap agrupado con los `b` de G2, posterior) · generación de recortes (G3b) · variabilidad entre semillas del segmentador (un solo U-Net por fold).

## Limitaciones
≈492 imágenes de entrenamiento y ≈6 positivas en el fold interno; pHash ≤ 8 exploratorio con riesgo residual de fuga; sin validación humana de grupos. Comprobación de integridad propia (no K6): HEAD, árbol limpio en `src/r5_v2`, `src/segmentation`, `src/r4_v2`, `configs/r5_v2`, `configs/r4_v2`, y SHA-256 de archivos clave.
