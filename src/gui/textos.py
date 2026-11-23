"""Textos fijos de la interfaz de R9 (únicos; la interfaz, la CLI y las pruebas los comparten).

Etiquetas BENIGNO/MALIGNO por correspondencia con el indicador R9.1 (decisión de la autora); se explican siempre con el mapeo v2. Cifras de rendimiento: evaluación final única de R8
(D-070; reportadas por la autora), sin recalcular. La puntuación NO se presenta como probabilidad calibrada.
"""

from __future__ import annotations

VERSION_PROTOTIPO = "r9_gui_v1"
T_HAT_TEXTO = "0.08456226376195748"
ETIQUETA_MALIGNO = "MALIGNO"
ETIQUETA_BENIGNO = "BENIGNO"
TITULO = "Prototipo demostrativo — clasificación binaria de lesiones ováricas en ecografía 2D"
PRESENTACION = ("Prototipo básico de sistema de apoyo diagnóstico, de carácter académico y demostrativo: sin validez diagnóstica, no destinado al uso clínico y no es un dispositivo médico.")
AVISO_PRINCIPAL = ("PROTOTIPO ACADÉMICO DEMOSTRATIVO. No es un dispositivo médico ni de diagnóstico, no tiene validez diagnóstica, no sustituye a especialistas y no está destinado al uso clínico.")
ALCANCE = ("Clasifica ecografías 2D de lesiones ováricas previamente identificadas. No detecta ovarios normales ni determina la presencia o ausencia de una lesión; "
           "no está validado para ovarios normales ni para población general.")
NOTA_ETIQUETAS = ("BENIGNO = clase negativa del mapeo v2 (lesión no maligna: categorías originales 0, 1, 2, 3, 4 y 6; incluye la categoría 6). "
                  "MALIGNO = clase positiva del mapeo v2 (categoría original 7: carcinoma seroso de alto grado en este conjunto de datos). Estas salidas no equivalen a un diagnóstico histopatológico.")
NOTA_PUNTUACION = (f"La puntuación es el promedio aritmético simple de las 15 probabilidades de los modelos; NO es una probabilidad calibrada. Regla: MALIGNO si puntuación ≥ umbral ({T_HAT_TEXTO}); "
                   "BENIGNO en otro caso. El umbral es una regla experimental fijada antes de la evaluación final, calibrada con un ensamble de tres modelos y no validada para el ensamble de quince.")
RENDIMIENTO = ("Evaluación final única (R8; 382 imágenes elegibles: 367 negativas y 15 positivas), con este mismo umbral: sensibilidad 0.60 (IC 95 % 0.32–0.84), especificidad 0.921 (IC 95 % 0.888–0.946), "
               "valor predictivo positivo 0.237 (prevalencia de la muestra 3.9 %), ROC-AUC 0.882. El objetivo exploratorio de sensibilidad de 0.90 no se alcanzó. Con solo 15 positivas la incertidumbre es amplia; "
               "no hay identificadores de paciente y el conjunto contiene un único subtipo maligno. Sin validación clínica.")
MODO_MAQUETA = "Modo demostración: modelo no disponible"
MAQUETA_DETALLE = "No se ejecuta ninguna predicción y no se muestran resultados simulados. Motivo: {motivo}"
ACERCA_DE = (f"Interfaz {VERSION_PROTOTIPO}. Sistema: congelado según la declaración D-060 (DenseNet121 + representación A; 15 modelos de G2; preprocesamiento de G2; "
             "promedio aritmético simple). La interfaz no entrena, no recalibra y no calcula métricas de rendimiento.")


def advertencias() -> list[str]:
    """Bloque de avisos que debe mostrarse siempre junto a cualquier resultado, en este orden."""
    return [AVISO_PRINCIPAL, ALCANCE, NOTA_ETIQUETAS, NOTA_PUNTUACION, RENDIMIENTO]
