# Protocolo R7 `umbral_sistema_v1`: calibración exploratoria del umbral del sistema provisional y análisis descriptivo por categoría

| Campo | Estado |
|---|---|
| Estado | **Candidato a congelamiento por la autora (D-048).** Su commit y su `sha256_contenido_blob` se registran en `configs/r7_v2/umbral_sistema_v1.yaml` (`protocolo:`) en un commit posterior; este archivo no contiene su propio hash. No autoriza ejecutar sobre predicciones reales |
| Leyenda | partición provisional (`phash_le8_v1`, exploratoria, post hoc); conformidad académica pendiente. Mapeo v2. Solo desarrollo; test cerrado (K4); 180 normales fuera |
| Código | `src/r7_v2/umbral_sistema.py`, `src/r7_v2/analisis_categorias.py`; configuración `configs/r7_v2/umbral_sistema_v1.yaml` (bloque `procedimiento`, validado por código contra constantes y contra este texto) |

## Entradas y guardas
Solo las 15 corridas A|densenet121 de G2 (3 semillas × 5 folds; solo lectura, `r4_v2.agregacion` con su comparabilidad) y la partición (K1–K4). Falta o duplicado de una corrida bloquea. Guardas: `--autorizacion CONFIRMADO:D-NNN` (sin ella no se lee ni se escribe), `seleccion_r6.json` con elegida A|densenet121 (ausencia o distinta bloquea), integridad propia (HEAD, árbol limpio, SHA-256 de archivos clave, protocolo versionado con el hash registrado en el YAML), directorio nuevo (nunca sobrescribe).

## Sistema final provisional y fuente de calibración
Sistema PROVISIONAL (no validado; su umbral numérico no se congela en este hito): DenseNet121, representación A, 15 checkpoints, promedio aritmético simple de probabilidades, sin ponderaciones aprendidas ni selección retrospectiva de miembros. Fuente de desarrollo: A = `q_barra`, promedio de las tres predicciones OOF por imagen (semillas 42, 43, 44): OOF exacta de un ensamble de TRES modelos del mismo fold.
**`discrepancia_A_vs_B`**: el sistema B es el promedio de 15 modelos; para una imagen de desarrollo 12 de los 15 modelos la vieron en entrenamiento, por lo que no existe OOF de B. La calibración sobre A no se presenta como equivalente ni validada para B y el desajuste no es cuantificable con este diseño. No se usa ningún criterio numérico de «discrepancia relevante».

## Calibración (S*)
S* = 0.90 es el objetivo PRIMARIO: objetivo exploratorio ya usado en G4, no garantía de sensibilidad ni validación clínica. S* = 0.85 y 0.95 son solo descriptivos (sensibilidad al criterio): no se selecciona entre ellos ni se destaca el más favorable. S* se pasa como cadena/`Fraction`, nunca como float. Regla `p >= umbral`; los empates no se rompen (se reportan `n_pos_empatadas` y `n_neg_empatadas`); k = ceil(S*·n⁺) en aritmética entera (38 positivas y 0.90: k = 35). Se reutilizan `calibrar_umbral`, `comprobar_transferencia` y `evaluar_en_umbral` sin modificarlos. El valor de t̂ se rotula «calculado en desarrollo; NO congelado; no es el umbral de R8».

## Salidas de `umbral_sistema`
`q_barra_oof.csv` y `trazabilidad.json` (SHA-256 de cada predicción, run_config y checkpoint, HEAD, partición, numpy); `agregaciones_ap_auc.csv` (m = 1, 2, 3, IC95 % bootstrap agrupado estratificado con los índices `b` de G2 verificados, B = 10 000; sin afirmar mejora); `umbral_primario` (t̂, k, n⁺, TP/FN/FP/TN, empates y distribución de t̂ recalculado en cada remuestreo); `tabla_vs_0_5.csv` (sensibilidad, especificidad, FP, FN, VPP y diferencias; VPP solo válido para la prevalencia del desarrollo); `transferencia` (leave-one-fold-out con t̂ recalculado en cada remuestreo: estimación de transferencia en desarrollo, no validación independiente); `sensibilidad_S_star.csv`; `m_agregaciones.csv`; `resumen_umbral.json`. Mover el umbral desplaza el punto sobre la misma curva: no mejora AP ni ROC-AUC.

## Afirmaciones
Posibles (sin reentrenar ni test): lo calculado sobre `q_barra` de tres semillas; cómo cambian AP, ROC-AUC, t̂ y el punto de operación con m = 1, 2, 3; la transferencia entre folds como estimación en desarrollo. NO permitidas: que t̂ garantice sensibilidad en el ensamble de 15 o en test; que el sistema de 15 esté validado; extrapolar numéricamente de m ≤ 3 a 15; utilidad clínica; mejora de AP o ROC-AUC por mover el umbral; independencia absoluta del test histórico. Sin mínimos absolutos de especificidad ni máximos de FP (viabilidad clínica: no determinada).

## Análisis por categoría original (`analisis_categorias`)
Descriptivo; no altera etiquetas, entrenamiento ni selección; el mapeo v2 no cambia y la categoría 6 NO se reevalúa como positiva. R7.3 no se declara cumplido (discrepancia de alcance: el IOV se definió para el mapeo anterior). Lee `q_barra_oof.csv` verificando su SHA-256 contra `trazabilidad.json`; la categoría 5 y cualquier clase fuera de {0, 1, 2, 3, 4, 6, 7} bloquean. Por categoría: n total y por fold; mediana e IQR de `q_barra`; proporción por encima del t̂ de transferencia del fold (S* = 0.90, otros cuatro folds): sensibilidad para la 7, tasa de falsos positivos (x/n) para las demás; AUC de cada categoría no maligna contra la 7 con IC95 % bootstrap agrupado (mismos `b`) solo si n ≥ 10. Foco categoría 6: n, FP/n y diferencia de proporción frente al resto de no malignas, sin p-valores; sin independencia por paciente.

## Límites
38 positivas (≈ 7–8 por fold); pHash ≤ 8 exploratorio con riesgo residual; desarrollo reutilizado; accesos históricos al test declarados; no se afirma independencia absoluta del test histórico. R8 es un protocolo aparte: este protocolo no lo actualiza.
