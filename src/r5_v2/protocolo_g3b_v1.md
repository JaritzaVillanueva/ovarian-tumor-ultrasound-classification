# Protocolo G3b `g3_condiciones_v1`: condiciones de entrada B y C frente a A

| Campo | Estado |
|---|---|
| Estado | **Candidato a congelamiento por la autora.** Su commit y `sha256_contenido_blob` se registran en `configs/r5_v2/g3_condiciones_v1.yaml` (`protocolo:`) en un commit posterior; este archivo no contiene su propio hash. No autoriza generar recortes ni entrenar |
| Leyenda | partición provisional (`phash_le8_v1`, exploratoria, post hoc); conformidad académica pendiente. Sin afirmar superioridad de ninguna condición |
| Fuente | `Reportes/asesorias/borrador_protocolo_G3_G4_v0.md` §3–§5, §13; IOV R5.1 y R5.4 · configuración `configs/r5_v2/g3_condiciones_v1.yaml` · código `src/r5_v2/g3b_*.py` |
| Datos | Las 820 elegibles de `phash_le8_v1` (K1–K4 de G2). Solo desarrollo; test cerrado; 180 normales fuera |

## Alcance
Clasificar con la receta de G2 (sin cambios) tres condiciones de entrada: **A** imagen completa (resultados de G2, solo lectura), **B** ROI de la máscara predicha OOF de G3a y **C** ROI de la máscara de referencia. La única variable declarada es el origen de la imagen.

## Decisiones de la autora (2026-11-22)
1. **B = B-híbrida (`B_hibrida`)**: entrenamiento con ROI de la máscara de referencia; validación/predicción del fold k con el ROI de la máscara OOF del U-Net `unet__s42__f{k}` (no entrenado con esas imágenes). **C**: referencia en entrenamiento y validación. **`B_oof`** (entrenamiento con la máscara OOF de cada imagen) queda **deshabilitada**, como análisis secundario opcional; riesgo: los U-Net de otros folds vieron imágenes del fold k (fuga indirecta débil). B-híbrida **difiere de la B histórica** y debe nombrarse así en el informe.
2. ROI idéntico para B y C: `src/segmentation/roi.py::extract_roi_from_mask` sin editar; `keep_largest=true`, margen 0.10, área mínima 0.001, fallback a imagen completa; se registra por imagen fallback, nº de componentes y caja.
3. El recorte (PNG sin pérdida) entra al clasificador como la imagen completa de A: misma transformación, normalización y receta de G2; se reutiliza `r4_v2.entrenamiento.ejecutar_corrida` y su entrenador torch, sustituyendo solo la ruta de la imagen (guarda K4 de G2 intacta).
4. Alcance inicial: 3 arquitecturas × folds 0–4 × **semilla 42** para B-híbrida (15) y C (15): **30 corridas**. El contraste usa A de G2 con la **misma semilla 42 y los mismos folds**; no se compara contra A promediada entre semillas. Las semillas 43 y 44 de B quedan **pendientes de decisión posterior**; con una sola semilla no se afirma robustez multisemilla ni se aplica R6 como media entre semillas.

## Distinción comprobada entre B y C (`g3b_recortes.py::crear_entrenador`, `fuentes_de`)
Entrenamiento: **idéntico** (ambas leen `recortes/gt/`; los SHA-256 de entrenamiento coinciden). Validación/predicción del fold k: **B** lee `recortes/oof_pred/` (ROI de la máscara OOF) y **C** lee `recortes/gt/`. Como la pérdida de validación gobierna checkpoint y parada (G2), la entrada de validación también puede alterar la época elegida. Por tanto las entradas evaluadas difieren salvo en las imágenes cuya caja predicha coincide con la de referencia (recorte con el mismo SHA-256); C−B mide el coste de la máscara automática en la evaluación (más el posible efecto de selección de época). `procedencia_g3b.json` registra `sha256_recortes` por rol; tras `recortes` se comprueba el nº de imágenes con recorte `oof_pred` igual al `gt`. Si ese nº fuera ≈ 820, B y C no serían distinguibles y habría que detenerse.

## Modos y precondiciones
`recortes` (CPU, sin entrenar): exige `--aceptar-mascaras-g3a CONFIRMADO:D-NNN` (la autora decide la calidad de las máscaras tras ver Dice/IoU/cobertura; **sin umbrales numéricos de calidad**) y la consolidación de G3a `completa` con 5 folds `completada`, 820 máscaras con SHA-256 coincidente y cada imagen predicha una vez por el U-Net de su fold. Los recortes se escriben en un directorio de preparación y se publican por renombrado (nunca se sobrescribe); manifiesto con SHA-256 por recorte y por máscara de origen, y sentinela.
`humo` (1 arquitectura, 1 fold, 1 época; no cuenta) → `completa` (exige `humo` completado de la **misma condición**, mismo commit, YAML y partición). Ambos: `--autorizacion-ejecucion CONFIRMADO:AUT|D-NNN`, integridad propia (HEAD, árbol limpio, SHA-256 de archivos clave y pines de YAML de G2/G3a), K1–K4 y K8 de G2, recortes verificados y **comparabilidad con A** (misma partición `sha256_bytes_archivo`, misma receta numérica y misma referencia de pesos).

## Invariantes de no-fuga (código y pruebas)
Ninguna imagen del fold k en entrenamiento; los recortes `oof_pred` de la validación proceden del U-Net de su propio fold; SHA-256 de cada recorte verificado antes de entrenar; rutas con componente `test` rechazadas; ninguna categoría 5.

## Reporte (`g3b_reporte_condiciones.py`)
Solo con **0 corridas faltantes** de lo solicitado. AP, ROC-AUC, sensibilidad y especificidad a 0.5 por arquitectura/semilla; media entre semillas; ΔAP(B−A), ΔAP(C−A) y, descriptivo, C−B, con IC95 por bootstrap agrupado por `group_id`, estratificado por clase, B=10 000, **con los índices `b` de G2** (se regeneran con sus semillas y se verifican contra los `.npz` de G2 si existen). Con una sola semilla el contraste es «condicional a la semilla 42». Identidad descriptiva C−A = (C−B) + (B−A). Tablas T-B1…T-B4 y figura ΔAP; sin texto interpretativo.

## Limitaciones
Recortes de G3a provisionales (ROI y calidad de máscaras sin validar); B-híbrida mide el efecto de segmentación automática solo en validación; una semilla del U-Net por fold; ΔAP con ≈7–8 positivas por fold e IC amplios; pHash ≤ 8 exploratorio con riesgo residual de fuga; comprobación de integridad propia (no K6).
