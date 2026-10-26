# Historical Notebooks

This directory preserves preliminary experiments and earlier analyses that preceded the formal R4-R6 workflow. These notebooks are retained for methodological traceability and are **not part of the current experimental protocol**.

## Contents

| Notebooks                                                     | Purpose                                                         |
| ------------------------------------------------------------- | --------------------------------------------------------------- |
| `baselines_preliminares/<arch>/03_entrenamiento_<arch>.ipynb` | Preliminary training of ResNet50, DenseNet121 and ConvNeXt-Tiny |
| `baselines_preliminares/<arch>/04_evaluacion_<arch>.ipynb`    | Evaluation of the preliminary models on test                    |
| `00_seleccion_modelo.ipynb`                                   | Previous analysis with ROC-AUC criterion                        |
| `04_analisis_resultados_formales_R5_R6.ipynb`                 | Earlier single-seed analysis and R6 selection rule              |

Preliminary results are stored in `results/classification/<arch>_baseline/`, with local checkpoints in `models/classification/`.

## Preliminary Classification Baselines

Each architecture was trained once for 30 epochs, without a fixed seed, data augmentation or early stopping. Checkpoints were selected using minimum validation loss and subsequently evaluated on **test (469 images)**.

| Architecture  | Balanced accuracy | ROC-AUC | Malignant recall |
| ------------- | ----------------: | ------: | ---------------: |
| ResNet50      |             0.568 |   0.753 |             0.15 |
| DenseNet121   |             0.626 |   0.755 |             0.29 |
| ConvNeXt-Tiny |             0.713 |   0.816 |             0.77 |

These evaluations preceded the formal R4/R5 experiments. Their results did not enter the formal training data, although indirect influence on subsequent experimental decisions cannot be fully excluded.

The formal protocol instead uses three seeds (42-44), data augmentation, early stopping and checkpoint selection by validation ROC-AUC. Its 27 classification runs use **train and validation only**.

The `*_baseline.yaml` files in `configs/classification/` belong to this later formal protocol, not to the preliminary baselines.

## Earlier Model Selection

`04_analisis_resultados_formales_R5_R6.ipynb` used a single seed (42), condition B and Average Precision to select ConvNeXt-Tiny.

`00_seleccion_modelo.ipynb` evaluated mean validation ROC-AUC across three seeds, with which DenseNet121-A ranks first. Its outputs are stored in `results/classification/r6_model_selection/historico/`.

The current R6 selection follows the approved IOV: highest mean validation AP across three seeds (`notebooks/R6_model_selection/01_seleccion_modelo_IOV_AP.ipynb`), which selects **ConvNeXt-Tiny-A**. ROC-AUC is kept there as an alternative sensitivity analysis.

## Reproducibility Notes

- Do not re-execute the preliminary notebooks without reviewing their dependencies: training may overwrite checkpoints, and evaluation reads test.
- The ResNet50 notebooks contain relative paths that no longer resolve correctly after relocation.
- The single-seed analysis reads `results/classification/formal/` and must remain at its current directory depth.
- The U-Net's historical test evaluation is documented separately in `notebooks/R5_input_representation/README.md`.

Current training and model selection are implemented in `src/classification/run_experiment.py` and `notebooks/R6_model_selection/01_seleccion_modelo_IOV_AP.ipynb`, respectively.
