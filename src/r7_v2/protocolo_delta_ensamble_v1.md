# Protocolo R7 `delta_ensamble_v1`: contrastes pareados del ensamble de tres semillas frente a cada semilla individual

| Campo | Estado |
|---|---|
| Estado | **Candidato a congelamiento por la autora (D-051).** Su commit y su `sha256_contenido_blob` se registran en `configs/r7_v2/delta_ensamble_v1.yaml` (`protocolo:`) en un commit posterior; este archivo no contiene su propio hash. No autoriza ejecutar sobre predicciones reales |
| Leyenda | partición provisional (`phash_le8_v1`, exploratoria, post hoc); conformidad académica pendiente. Mapeo v2. Solo desarrollo; test cerrado (K4); 180 normales fuera |
| Código | `src/r7_v2/delta_ensamble.py`; configuración `configs/r7_v2/delta_ensamble_v1.yaml` (bloque `procedimiento`, validado por código contra constantes y contra este texto) |

## Entrada y guardas
Una sola entrada: `umbral_sistema_v1/q_barra_oof.csv` (image_id, fold, label_v2, p_s42, p_s43, p_s44, q_barra), verificado por SHA-256 contra el `trazabilidad.json` de esa carpeta; fallan columnas ausentes, imágenes faltantes o duplicadas. `group_id` viene de la partición (K1–K4). Guardas: `--autorizacion CONFIRMADO:D-NNN` (sin ella no se lee ni se escribe), `seleccion_r6.json` con elegida A|densenet121 (ausencia o distinta bloquea), integridad propia (HEAD, árbol limpio, SHA-256, protocolo versionado con el hash registrado en el YAML), directorio nuevo (nunca sobrescribe).

## Contrastes (AP primaria; ROC-AUC secundaria)
Los mismos índices `b` de G2 (regenerados con sus semillas, verificados con `verificar_indices_g2`, B = 10 000) se usan para todas las columnas y las 820 imágenes: bootstrap agrupado por `group_id`, estratificado por clase del grupo y, como sensibilidad, no estratificado. Tres contrastes individuales: ΔAP(q_barra − p_s) para s = 42, 43, 44, con estimación puntual, IC95 % percentil, número de remuestreos válidos y bandera «IC excluye 0» (sin lenguaje de contraste de hipótesis). Se reporta además el IC ajustado de nivel 1 − 0.05/3 (percentiles α/6 y 1 − α/6) para los tres, rotulado «ajuste por multiplicidad; no corrige el sesgo de selección exploratoria». Aparte, ΔAP(q_barra − media de las AP individuales), con la media calculada DENTRO de cada remuestreo: resumen descriptivo; no es un comparador que exista como sistema. Lo mismo para ROC-AUC.

## Lectura permitida y límites
Es un análisis exploratorio. El ensamble de tres semillas es una elección de diseño fijada antes de ver estos números (D-048); los contrastes son posteriores. La mejor semilla individual identificada mirando resultados no es el único comparador: se reportan las tres. Hay dependencia entre modelos (mismas imágenes, mismos pliegues): los IC no asumen independencia; el agrupamiento pHash ≤ 8 es exploratorio, sin independencia por paciente. Hay 38 positivas (≈ 7–8 por fold). Es OOF de un ensamble de TRES modelos del mismo pliegue: no es evidencia sobre el ensamble de quince ni sobre test. El análisis no reabre R6 ni cambia la arquitectura seleccionada, y no afirma utilidad clínica. Por tanto no se afirma independencia absoluta del test histórico.

## Salidas (`results/r7_v2/phash_le8_v1/delta_ensamble_v1/`)
`delta_ap.csv`, `delta_roc_auc.csv`, `trazabilidad.json` (SHA-256 de las entradas, commit, partición, numpy) y `resumen_delta_ensamble.json` (leyendas, banderas, límites). Cada CSV lleva el encabezado `# ...` con la leyenda fija. R8 es un protocolo aparte: este protocolo no lo actualiza.
