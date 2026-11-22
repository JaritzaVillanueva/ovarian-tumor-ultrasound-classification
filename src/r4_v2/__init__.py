"""G2 línea base v1 (`g2_baseline_v1`): implementación del protocolo congelado `protocolo_g2_baseline_v1.md`.

Mapeo v2 (lesión no maligna 0,1,2,3,4,6 / lesión maligna 7; la categoría 5 queda excluida), partición provisional
`phash_le8_v1`; conformidad académica pendiente. Este paquete NO abre el test oficial ni las 180 normales.

Los módulos que no dependen de torch (partición, integridad, métricas, bootstrap, calibración, reporte) se prueban
con datos sintéticos; torch y torchvision solo se importan de forma diferida dentro de las funciones que los usan.
"""
