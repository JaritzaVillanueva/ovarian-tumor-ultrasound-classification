"""R9: prototipo académico DEMOSTRATIVO de interfaz para el sistema final congelado (declaración D-060).

Carga una ecografía 2D, la muestra y clasifica la lesión como BENIGNO/MALIGNO con exactamente el sistema congelado: DenseNet121 + representación A, 15 checkpoints de G2 verificados por SHA-256,
preprocesamiento de G2, promedio aritmético simple y umbral exacto 0.08456226376195748 (regla `p >= t_hat`). No entrena, no recalibra, no calcula métricas y no abre el test.
NO es un dispositivo médico: sin validez diagnóstica y no destinado al uso clínico. Partición provisional (phash_le8_v1, exploratoria, post hoc); conformidad académica pendiente.
"""
