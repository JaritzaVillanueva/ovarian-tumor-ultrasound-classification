# Protocolo R8 `sistema_congelado_v1`: preparación del congelamiento formal del sistema final

| Campo | Estado |
|---|---|
| Estado | **Preparado para congelamiento; no declarado congelado.** La declaración formal de congelamiento es una decisión posterior, separada, que solo se toma después de verificar los commits y los artefactos (modo `verificar`). Este texto no contiene su propio hash: el commit y el `sha256_contenido_blob` del protocolo se registran en `configs/r8_v2/sistema_congelado_v1.yaml` (`protocolo:`) en un commit posterior |
| Leyenda | partición provisional (`phash_le8_v1`, exploratoria, post hoc); conformidad académica del asesor: PENDIENTE. Mapeo v2. Solo desarrollo; test cerrado; las 180 normales fuera |
| Código | `src/r8_v2/congelamiento.py`; configuración `configs/r8_v2/sistema_congelado_v1.yaml` (bloque `sistema`, comparado por el código con sus constantes y con este texto) |
| Autorización | decisión autoral de preparar el congelamiento (D-057); la autora no delega la declaración final |

## Sistema que se prepara para congelar
DenseNet121, representación A, 15 checkpoints de G2 (semillas 42/43/44 × folds 0–4; intento `completada`), los registrados en el manifiesto verificado (`manifiesto_miembros_v1`, copia byte a byte del manifiesto de D-055), sin sustituir miembros. Preprocesamiento original de G2, sin modificaciones (`get_val_transforms`, `DatasetLesiones`, `create_dataloader`, `predict`; sin augmentation). Agregación: promedio aritmético simple de las 15 probabilidades de la clase maligna (softmax float32 por modelo; promedio en float64), sin ponderaciones ni selección de miembros. Regla: `p >= t_hat` con `t_hat = 0.08456226376195748` (cadena exacta); los empates no se rompen.

## Lo que NO es
`t_hat` es una **regla experimental previa a R8**, no un umbral validado para quince modelos (sistema no validado para el ensamble de quince): se calculó en desarrollo con el OOF de tres modelos por imagen (S* = 0.90, objetivo exploratorio) y el sistema de quince no tiene OOF de desarrollo (12 de los 15 modelos vieron cada imagen). No garantiza sensibilidad del 90 %. No es una validación clínica. No detecta ovarios normales; sin uso clínico validado. El congelamiento no valida ni mejora el sistema: lo fija antes de R8.

## Qué fija el artefacto
El YAML registra: los SHA-256 de los archivos de procedencia (manifiesto, fidelidad, umbral, partición, configuración de G2), el SHA-256 del manifiesto de miembros versionado, las versiones del entorno de Falcon donde se verificó, y las referencias de trazabilidad (D-050, D-051, D-053, D-054, D-055, D-056 y la decisión de preparación). Cualquier cambio posterior exige una enmienda documentada con fecha, razón e impacto.

## Modo `verificar` (solo lectura; no abre el test)
Exige `--autorizacion CONFIRMADO:D-NNN`. Comprueba, en este orden: integridad propia (HEAD, árbol limpio en las rutas vigiladas, SHA-256 de los archivos con huella, protocolo versionado con el hash registrado en el YAML); `seleccion_r6.json` con elegida A|densenet121; SHA-256 de cada archivo de procedencia; coherencia entre el manifiesto de miembros versionado, el manifiesto JSON y el resumen de fidelidad (15 miembros únicos, sin advertencias, sin alerta de recarga); `t_hat` del resultado de umbral igual al del YAML; SHA-256 de los 15 checkpoints recalculado en disco. Las diferencias de versiones del entorno se registran como advertencia. Escribe un resumen en un directorio nuevo (nunca sobrescribe). **No declara el congelamiento.**

## Límites
R8 es un protocolo aparte, con su propio ejecutor oficial autorizado: este protocolo no lo actualiza ni abre el test. No se afirma independencia absoluta del test histórico. Los resultados históricos (B0/B2, E1/E2, ConvNeXt) pertenecen al mapeo anterior y no se reutilizan.
