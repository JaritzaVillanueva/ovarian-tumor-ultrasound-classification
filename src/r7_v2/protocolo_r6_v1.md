# Protocolo R6 `seleccion_r6_v1`: regla de selección de configuración

Mapeo v2; partición provisional (`phash_le8_v1`, exploratoria, post hoc); conformidad académica pendiente. Solo desarrollo; test cerrado; normales fuera. Estado: borrador para aprobación de la autora (no congelado; sin hash propio, se registra su SHA-256 en HEAD en `seleccion_r6.json`).

## Regla (IOV R6.1 y R6.2)
- Conjunto: 3 arquitecturas (resnet50, densenet121, convnext_tiny) x {A (imagen completa), B_hibrida (ROI de la máscara predicha OOF)} = 6 combinaciones. C no participa: usa la máscara de referencia en evaluación y no es desplegable (B_oof tampoco; se ignoran y se registran).
- `AP_media(c)` = media entre semillas del AP OOF por semilla (tablas ya aceptadas: `T3_por_semilla.csv` de G2 para A y `T-B1_por_semilla.csv` de G3b para B_hibrida; valores a 6 decimales).
- Elegibilidad: solo participan en el ranking formal las combinaciones con exactamente 3 semillas (`n_semillas == 3`). Las demás se conservan en la tabla con `elegible: false`, sin rango, y no entran en el ranking, el margen ni el empate. Falla si alguna arquitectura de A no tiene 3 semillas. Con B de 3 semillas, las 6 son elegibles y la regla completa aplica sin cambios.
- Se redondea a 4 decimales SOLO para detectar empates exactos. Elegida = máxima `AP_media`.
- Desempate determinista, solo con empate exacto a 4 decimales: (i) A sobre B_hibrida (parsimonia); (ii) mayor ROC-AUC media entre semillas; (iii) orden fijo resnet50 < densenet121 < convnext_tiny. Sin tolerancias.
- Se aplica una vez mediante `python -B -m src.r7_v2.seleccion_r6 --autorizacion CONFIRMADO:D-NNN`; salida nueva, nunca sobrescrita.

## Qué NO es criterio de selección
IC95 o bootstrap; sensibilidad/especificidad a 0.5; estabilidad entre semillas (rango, desviación estándar); umbral de decisión; ΔAP o T4. Las columnas `desc_*`, `ap_semillas_comunes`, rango y desviación se escriben solo como descripción. Selección no es inferencia: no se afirma diferencia entre configuraciones.

## Limitación declarada
B_hibrida tiene solo la semilla 42; A tiene 42/43/44. R6.2 («media entre semillas») queda PARCIAL mientras alguna combinación tenga < 3 semillas (`r6_2_media_entre_semillas_completa = false`). La selección formal se fundamenta en A con tres semillas; el AP de B de una semilla NO se compara con la media de A como estimaciones equivalentes (`b_excluida_de_la_seleccion_por_semillas_insuficientes`, `seleccion_formal_solo_entre`).
Complemento descriptivo (no participa): `comparacion_emparejada_semilla_comun` da, por arquitectura, el AP de A y de B en las semillas comunes y la diferencia B-A, condicional a esas semillas y sin IC (los IC emparejados están en T-B2 de G3b). Con 38 positivas y grupos pHash exploratorios, la incertidumbre es alta.

## Procedencia honesta de la regla
Redactada en el borrador G3–G4 v0 (22-nov-2026) antes de conocer los AP de A y B; formalizada y aplicada después de observarlos. El desempate no interviene salvo empate exacto.

## Verificaciones del script
Falla si faltan arquitecturas o combinaciones, hay filas duplicadas, valores no válidos, rutas con componente `test`, salida existente o autorización ausente. Comprueba que A de G2 coincide con A de G3b en los pares comunes. Integridad sencilla (HEAD, árbol limpio en `src/r7_v2`, SHA-256); no es K6.
