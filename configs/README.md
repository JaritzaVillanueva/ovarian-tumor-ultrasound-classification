# Configurations

Configuration files used in the thesis experiments.

## Classification

| File | Purpose |
|---|---|
| `classification/resnet50_baseline.yaml` | ResNet50 training configuration |
| `classification/densenet121_baseline.yaml` | DenseNet121 training configuration |
| `classification/convnext_tiny_baseline.yaml` | ConvNeXt-Tiny training configuration |
| `unet_baseline.yaml` | Empty placeholder; U-Net parameters are defined in its training notebook |

The three classification YAML files define the shared training protocol used from R4 onward. Despite the `_baseline` suffix, they do not reproduce the historical preliminary baselines.

`run_experiment.py` reads the training parameters, early stopping settings, pretrained initialization and number of classes. Architecture selection (`model.name`), optimizer and weighted loss are controlled by the code rather than their corresponding YAML entries.

## Experimental datasets

The dataset uses a frozen partition of **800 train, 200 validation and 469 test images**.

The three manifests are documented in [splits/README.md](splits/README.md):

- `mmotu_experimental_split_standard.csv`: operational manifest used by the training pipeline.
- `mmotu_experimental_split.csv`: canonical reference with full traceability.
- `legacy/mmotu_experimental_split_legacy.csv`: original project manifest used by the R2 split-documentation notebook.

Do not modify or regenerate these manifests.

## Experiment configurations

| Experiment | Conditions | Seeds | Checkpoint selection |
|---|---|---|---|
| `formal/` (preliminary) | A, B, C | 42 | Minimum validation loss |
| `pilot_lr3e-5/` | A | 42 | Minimum validation loss |
| `diagnostic_checkpoint/` | A | 42 | Both loss- and AUC-based checkpoints |
| `r4_benchmark/` | A | 42 - 44 | Maximum validation ROC-AUC |
| `r5_input_representation/` | B, C | 42 - 44 | Maximum validation ROC-AUC |

All classification runs use the same YAML configurations, with experiment-specific command-line arguments. Early stopping remains based on validation weighted loss.

Execution parameters and results are preserved in each run's `run_config.json`. R6 selects its classifier using results from R4 and R5.

## Reproducibility

- Keep the frozen manifests and existing classification YAML values unchanged.
- Use separate experiment configurations for subsequent changes.
- Consult the source code and `run_config.json` for effective runtime parameters.
- U-Net configuration is recorded in `notebooks/R5_input_representation/01_unet_baseline.ipynb`.