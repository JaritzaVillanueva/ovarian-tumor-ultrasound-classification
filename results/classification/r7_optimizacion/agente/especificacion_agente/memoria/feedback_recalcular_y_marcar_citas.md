---
name: feedback-recalcular-y-marcar-citas
description: Recalcular toda cifra derivada cuando cambia el diseño y marcar las citas leídas solo por resumen automático
metadata:
  type: feedback
---

1. Toda cifra derivada se recalcula cuando cambia el diseño. En la Fase 1 di 0.94 como probabilidad de que alguna candidata cumpla la regla de signo por azar (14 candidatas: 1 − 0.8125^14 ≈ 0.945); al quedar 9 candidatas el valor correcto es 1 − 0.8125^9 ≈ 0.846. El autor pidió «incluir el 0.94»; se incluyó el cálculo correcto y se explicó el origen del 0.94 (`protocolo_agente.json`, `sesgo_de_ganador`).
2. Los valores sin fuente se rotulan por quién los eligió («elección del autor, sin fuente» vs «decisión del agente, sin fuente»), y las citas o cifras leídas solo mediante WebFetch/WebSearch con resumen automático llevan [CITA POR VERIFICAR] (Tablas 5-6 de ConvNeXt, Tabla 1 de Tupper y Gagné, DOI de Tirindelli).

**Why:** el jurado debe poder distinguir lo respaldado de lo no verificado; el autor lo pidió explícitamente el 2026-11-11.
**How to apply:** al cerrar cualquier protocolo o informe, revisar las cifras derivadas contra el diseño vigente y rotular cada fuente por su estado de verificación.
