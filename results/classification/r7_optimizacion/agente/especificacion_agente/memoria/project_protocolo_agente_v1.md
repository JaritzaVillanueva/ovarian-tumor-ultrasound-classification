---
name: project-protocolo-agente-v1
description: Decisiones aprobadas por el autor para el protocolo v1 del brazo del agente de optimización R7 (base B0, 9 candidatas, confirmación con semillas 43/44, 75 corridas)
metadata:
  type: project
---

Protocolo v1 redactado el 2026-11-11 en `ovarian-tumor-ultrasound-classification/results/classification/r7_optimizacion/agente/protocolo_agente.json` (pendiente de commit del autor).

- Base de la búsqueda: B0 (α = 1.0, fijo). B0 y B2 son referencias a la vez porque en E2 B2 no supera a B0 (ΔAP medio −0.0431, 2/5 folds; `robustez_e2/analisis/resumen_delta.csv`).
- Etapa 1: 9 candidatas (a01, a02, a03, a04, a05, a06, a08, a09, a10) × 5 folds, semilla 42 = 45 corridas. Sin a07 (weight decay) y sin etapa 2 (combinaciones).
- Confirmación: el finalista (mayor AP medio entre los que no caen más de 0.02 de ROC-AUC frente a B0) + B0 + B2 con semillas 43 y 44 = 30 corridas. La decisión usa solo esas semillas. Total 75 corridas.
- Sin ganadora: R8 se queda con B0 y B2 (`cierre_seleccion/protocolo_r8_congelado.json`, 6 modelos); no se sustituye por B2.
- Marcado «elección del autor, sin fuente»: a02 (lr 2e-4) y la rampa coseno de 15 épocas (a03).

**Why:** el autor aprobó con cambios la propuesta de la Fase 1 para ajustar el presupuesto y reducir el sesgo de ganador.
**How to apply:** si se pide una enmienda, respetar que nada nuevo entra sin enmienda commiteada. Los argumentos nuevos de `run_experiment.py` (--lr-schedule, --layer-decay, --drop-path-rate, --label-smoothing, --augmentation, --image-size, --ema-decay) aún no existen: se implementan en la Fase 4 con valor por defecto neutro. Verificar el estado actual del repositorio antes de asumirlo.
