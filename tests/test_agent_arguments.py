"""
Pruebas de los argumentos nuevos de run_experiment.py del brazo del agente de optimización (R7):
--lr-schedule, --warmup-epochs, --cosine-epochs, --lr-min, --layer-decay, --drop-path-rate, --label-smoothing,
--augmentation, --image-size y --ema-decay (protocolo: results/classification/r7_optimizacion/agente/protocolo_agente.json).

Cada bloque prueba (1) que el valor por defecto no cambia el comportamiento y (2) el comportamiento nuevo.
Requieren torch y torchvision: se ejecutan en Falcon.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.classification.train import set_seed, train_model, validate_one_epoch
from src.classification.training_options import ModelEMA, build_layer_decay_param_groups, lr_multiplier_fn
from tests import test_classification_protocol as base

CPU = torch.device("cpu")
WEIGHTS_ALPHA1 = (0.5610098242759705, 4.597701072692871)  # pesos de B0 en E1 (713 / 87)


# ---------------------------------------------------------------------------------------------------------
# Entrenamiento diminuto determinista (mismo esquema que TestCheckpointDiagnostic)
# ---------------------------------------------------------------------------------------------------------
def _loaders():
    from torch.utils.data import DataLoader

    train_ds = base.TestCheckpointDiagnostic._make_dataset(64, seed=1)
    val_ds = base.TestCheckpointDiagnostic._make_dataset(40, seed=2)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(7))  # 4 pasos por época
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    return train_loader, val_loader


def _train(directory, epochs=4, lr=0.5, model_factory=None, optimizer_factory=None, criterion=None, **kwargs):
    train_loader, val_loader = _loaders()
    set_seed(99)
    model = model_factory() if model_factory else nn.Linear(4, 2)
    optimizer = optimizer_factory(model) if optimizer_factory else torch.optim.SGD(model.parameters(), lr=lr)
    criterion = criterion if criterion is not None else nn.CrossEntropyLoss(weight=torch.tensor([0.6, 1.4]))
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer, device=CPU, epochs=epochs,
        save_path=str(directory / "best_auc.pth"), patience=100, min_delta=0.0, seed=42, use_amp=False,
        checkpoint_metric="val_roc_auc", **kwargs,
    )
    return history, model, optimizer, val_loader, criterion


def _spy_learning_rates(optimizer):
    """Registra el learning rate de todos los grupos en cada llamada a optimizer.step()."""
    seen = []
    original = optimizer.step

    def spy(*args, **kwargs):
        seen.append([group["lr"] for group in optimizer.param_groups])
        return original(*args, **kwargs)

    optimizer.step = spy
    return seen


class _TwoLevelNet(nn.Module):
    """Modelo diminuto con la misma estructura (features + classifier) que ConvNeXt."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        self.classifier = nn.Linear(4, 2)

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------------------------------------
# --lr-schedule, --warmup-epochs, --cosine-epochs, --lr-min
# ---------------------------------------------------------------------------------------------------------
class TestLrSchedule:
    def test_multiplier_warmup_cosine_and_floor(self):
        peak, steps, floor = 1e-4, 25, 1e-6 / 1e-4
        m = lr_multiplier_fn(peak, steps, warmup_epochs=1, cosine_epochs=16, lr_min=1e-6)

        assert m(0) == pytest.approx(1 / 25)  # calentamiento lineal desde pico/25
        assert m(24) == pytest.approx(1.0)  # al final del calentamiento se alcanza el pico
        assert m(25) == pytest.approx(1.0)  # empieza el coseno en el pico
        assert m(25 + 200) == pytest.approx(floor + (1 - floor) * 0.5)  # mitad del coseno (400 pasos)
        assert m(25 + 400) == pytest.approx(floor, abs=1e-12)  # fin del coseno
        assert m(10_000) == pytest.approx(floor, abs=1e-12)  # después se mantiene en lr_min
        assert all(m(s) <= m(s + 1) for s in range(24))  # el calentamiento no decrece
        assert all(m(s) >= m(s + 1) for s in range(25, 424))  # el coseno no crece

    def test_invalid_parameters_are_rejected(self):
        with pytest.raises(ValueError):
            lr_multiplier_fn(1e-4, 25, 1, 0, 1e-6)  # cosine_epochs = 0
        with pytest.raises(ValueError):
            lr_multiplier_fn(1e-4, 25, -1, 15, 1e-6)  # warmup negativo
        with pytest.raises(ValueError):
            lr_multiplier_fn(1e-4, 25, 1, 15, 2e-4)  # lr_min >= pico

    def test_default_does_not_touch_learning_rate(self, tmp_path):
        history, _, optimizer, _, _ = _train(tmp_path, epochs=3)
        assert [g["lr"] for g in optimizer.param_groups] == [0.5]
        assert "base_lr" not in optimizer.param_groups[0]
        assert "lr" not in history

    def test_learning_rate_is_set_before_every_iteration(self, tmp_path):
        fn = lr_multiplier_fn(0.5, 4, warmup_epochs=1, cosine_epochs=2, lr_min=0.05)  # 4 pasos por época
        seen = []

        def factory(model):
            optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
            seen.append(_spy_learning_rates(optimizer))
            return optimizer

        history, _, _, _, _ = _train(tmp_path, epochs=5, optimizer_factory=factory, lr_fn=fn)
        steps = seen[0]
        expected = [0.5 * fn(s) for s in range(5 * 4)]

        assert len(steps) == 20
        assert [s[0] for s in steps] == pytest.approx(expected)
        assert history["lr"] == pytest.approx([expected[(e + 1) * 4 - 1] for e in range(5)])
        assert expected[3] == pytest.approx(0.5)  # pico al terminar la época de calentamiento
        assert expected[-1] == pytest.approx(0.05)  # piso tras el coseno

    def test_schedule_keeps_checkpoint_criterion(self, tmp_path):
        fn = lr_multiplier_fn(0.5, 4, warmup_epochs=1, cosine_epochs=2, lr_min=0.05)
        history, _, _, _, _ = _train(tmp_path, epochs=6, lr_fn=fn)
        meta = json.loads((tmp_path / "best_auc_meta.json").read_text())
        assert meta["criterion"] == "max val_roc_auc"
        assert meta["epoch"] == int(np.argmax(history["val_roc_auc"])) + 1


# ---------------------------------------------------------------------------------------------------------
# --layer-decay
# ---------------------------------------------------------------------------------------------------------
class TestLayerDecay:
    def test_convnext_tiny_groups_cover_every_parameter_once(self):
        from src.classification.models import create_convnext_tiny_classifier

        model = create_convnext_tiny_classifier(pretrained=False, num_classes=2)
        groups = build_layer_decay_param_groups(model, lr=1e-4, weight_decay=1e-4, layer_decay=0.8)

        assert [g["name"] for g in groups] == [f"features.{i}" for i in range(8)] + ["classifier"]
        ids = [id(p) for g in groups for p in g["params"]]
        assert len(ids) == len(set(ids)) == len(list(model.parameters()))
        assert set(ids) == {id(p) for p in model.parameters()}
        for index in range(8):
            assert groups[index]["lr"] == pytest.approx(1e-4 * 0.8 ** (8 - index))  # el módulo 7 recibe 0.8, el 0 recibe 0.8^8
        assert groups[7]["lr_scale"] == pytest.approx(0.8)
        assert groups[0]["lr_scale"] == pytest.approx(0.8 ** 8)
        assert groups[-1]["lr"] == 1e-4  # la cabeza recibe el lr completo
        assert {g["weight_decay"] for g in groups} == {1e-4}

        optimizer = torch.optim.AdamW(groups, lr=1e-4, weight_decay=1e-4)
        assert [g["lr"] for g in optimizer.param_groups] == pytest.approx([g["lr"] for g in groups])

    def test_decay_one_gives_the_same_lr_everywhere_and_invalid_values_fail(self):
        model = _TwoLevelNet()
        assert {g["lr"] for g in build_layer_decay_param_groups(model, 0.1, 0.0, 1.0)} == {0.1}
        for bad in (0.0, 1.5, -0.2):
            with pytest.raises(ValueError):
                build_layer_decay_param_groups(model, 0.1, 0.0, bad)
        with pytest.raises(ValueError):
            build_layer_decay_param_groups(nn.Linear(4, 2), 0.1, 0.0, 0.8)  # sin features/classifier

    def test_schedule_scales_every_group_from_its_own_base_lr(self, tmp_path):
        fn = lr_multiplier_fn(0.5, 4, warmup_epochs=1, cosine_epochs=2, lr_min=0.05)
        seen = []

        def factory(model):
            groups = build_layer_decay_param_groups(model, lr=0.5, weight_decay=0.0, layer_decay=0.8)
            optimizer = torch.optim.SGD(groups, lr=0.5)
            seen.append(_spy_learning_rates(optimizer))
            return optimizer

        _train(tmp_path, epochs=3, model_factory=_TwoLevelNet, optimizer_factory=factory, lr_fn=fn)
        scales = [0.8 ** 2, 0.8, 1.0]
        for step, lrs in enumerate(seen[0]):
            assert lrs == pytest.approx([0.5 * scale * fn(step) for scale in scales])


# ---------------------------------------------------------------------------------------------------------
# --drop-path-rate
# ---------------------------------------------------------------------------------------------------------
def _drop_path_probabilities(model):
    from torchvision.ops import StochasticDepth

    return [module.p for module in model.modules() if isinstance(module, StochasticDepth)]


class TestDropPathRate:
    def test_default_matches_torchvision_default(self):
        from torchvision import models
        from src.classification.models import create_convnext_tiny_classifier

        mine = _drop_path_probabilities(create_convnext_tiny_classifier(pretrained=False))
        reference = _drop_path_probabilities(models.convnext_tiny(weights=None))
        assert mine == reference
        assert len(mine) == 18 and mine[0] == 0.0 and mine[-1] == pytest.approx(0.1)

    def test_rate_sets_the_last_block_and_ramps_linearly(self):
        from src.classification.models import create_convnext_tiny_classifier

        probabilities = _drop_path_probabilities(create_convnext_tiny_classifier(pretrained=False, drop_path_rate=0.3))
        assert probabilities == pytest.approx([0.3 * i / 17 for i in range(18)])

    def test_same_parameters_and_same_eval_output(self):
        """El stochastic depth no tiene parámetros y solo actúa en entrenamiento: los pesos preentrenados se cargan igual."""
        from src.classification.models import create_convnext_tiny_classifier

        torch.manual_seed(0)
        default = create_convnext_tiny_classifier(pretrained=False)
        torch.manual_seed(0)
        other = create_convnext_tiny_classifier(pretrained=False, drop_path_rate=0.3)
        assert {k: v.shape for k, v in default.state_dict().items()} == {k: v.shape for k, v in other.state_dict().items()}

        other.load_state_dict(default.state_dict())
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            assert torch.allclose(default.eval()(x), other.eval()(x))

    def test_invalid_rate_is_rejected(self):
        from src.classification.models import create_convnext_tiny_classifier

        for bad in (-0.1, 1.0):
            with pytest.raises(ValueError):
                create_convnext_tiny_classifier(pretrained=False, drop_path_rate=bad)


# ---------------------------------------------------------------------------------------------------------
# --label-smoothing
# ---------------------------------------------------------------------------------------------------------
WEIGHTS_FOLD4 = (800 / (2 * 712), 800 / (2 * 88))  # pesos α = 1.0 del fold 4 de E2 (712 benignas / 88 malignas en train)


def _expected_uniform_smoothed_weighted_ce(logits, target, weight, eps):
    """L = sum_n w[y_n] * (-sum_c q_c log p_c) / sum_n w[y_n], con q = (1-e)*onehot + e/C (enmienda E1)."""
    log_p = F.log_softmax(logits, dim=1)
    n_classes = logits.shape[1]
    q = torch.full_like(log_p, eps / n_classes)
    for n, label in enumerate(target.tolist()):
        q[n, label] += 1.0 - eps
    sample_loss = -(q * log_p).sum(dim=1)
    return (weight[target] * sample_loss).sum() / weight[target].sum()


class TestLabelSmoothing:
    """Enmienda E1 del protocolo del agente (a06): suavizado uniforme con los pesos de clase como peso por muestra."""

    @staticmethod
    def _batch(n=64, seed=0):
        generator = torch.Generator().manual_seed(seed)
        logits = 2.0 * torch.randn(n, 2, generator=generator, dtype=torch.float64)
        target = torch.randint(0, 2, (n,), generator=generator)
        target[0], target[1] = 0, 1
        return logits, target

    def test_epsilon_zero_is_the_current_weighted_cross_entropy(self):
        from src.classification.losses import SmoothedWeightedCrossEntropy

        logits, target = self._batch()
        weight = torch.tensor(WEIGHTS_ALPHA1, dtype=torch.float64)

        logits_a = logits.clone().requires_grad_(True)
        logits_b = logits.clone().requires_grad_(True)
        loss_new = SmoothedWeightedCrossEntropy(weight, label_smoothing=0.0)(logits_a, target)
        loss_old = nn.CrossEntropyLoss(weight=weight)(logits_b, target)
        loss_new.backward()
        loss_old.backward()

        torch.testing.assert_close(loss_new, loss_old, rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(logits_a.grad, logits_b.grad, rtol=1e-10, atol=1e-14)

    @pytest.mark.parametrize("weights", [WEIGHTS_ALPHA1, WEIGHTS_FOLD4], ids=["pesos_B0", "pesos_fold4"])
    def test_matches_the_amended_formula(self, weights):
        from src.classification.losses import SmoothedWeightedCrossEntropy

        logits, target = self._batch(seed=3)
        weight = torch.tensor(weights, dtype=torch.float64)
        for eps in (0.05, 0.1, 0.3):
            loss = SmoothedWeightedCrossEntropy(weight, label_smoothing=eps)(logits, target)
            torch.testing.assert_close(loss, _expected_uniform_smoothed_weighted_ce(logits, target, weight, eps), rtol=1e-10, atol=1e-14)

    @pytest.mark.parametrize("weights", [WEIGHTS_ALPHA1, WEIGHTS_FOLD4], ids=["pesos_B0", "pesos_fold4"])
    def test_optimum_is_the_uniformly_smoothed_target_whatever_the_weights(self, weights):
        """Una benigna tiene su óptimo en P(maligna) = 0.05 y una maligna en 0.95, con gradiente cero ahí."""
        from src.classification.losses import SmoothedWeightedCrossEntropy

        eps = 0.1
        criterion = SmoothedWeightedCrossEntropy(torch.tensor(weights, dtype=torch.float64), label_smoothing=eps)
        for label, p_malignant in ((0, 0.05), (1, 0.95)):
            q = torch.tensor([1.0 - p_malignant, p_malignant], dtype=torch.float64)
            logits = torch.log(q)[None, :].clone().requires_grad_(True)  # softmax(logits) = q
            criterion(logits, torch.tensor([label])).backward()
            assert torch.allclose(logits.grad, torch.zeros_like(logits.grad), atol=1e-12)

        # lote mixto: cada muestra tiene su óptimo en su propio objetivo suavizado (el peso solo escala su gradiente)
        target = torch.tensor([0, 1, 1, 0, 0])
        q = torch.tensor([[0.95, 0.05] if label == 0 else [0.05, 0.95] for label in target.tolist()], dtype=torch.float64)
        logits = torch.log(q).clone().requires_grad_(True)
        criterion(logits, target).backward()
        assert torch.allclose(logits.grad, torch.zeros_like(logits.grad), atol=1e-12)
        assert q[0].tolist() == pytest.approx([0.95, 0.05]) and q[1].tolist() == pytest.approx([0.05, 0.95])

    def test_a_sample_is_weighted_by_the_weight_of_its_true_class(self):
        """Dos muestras con el mismo suavizado: la pérdida total es la media de sus pérdidas ponderada por w[y]."""
        from src.classification.losses import SmoothedWeightedCrossEntropy

        weight = torch.tensor([1.0, 4.0], dtype=torch.float64)
        criterion = SmoothedWeightedCrossEntropy(weight, label_smoothing=0.1)
        logits = torch.tensor([[0.3, -0.2], [0.1, 0.9]], dtype=torch.float64)
        loss_benign = criterion(logits[:1], torch.tensor([0]))
        loss_malignant = criterion(logits[1:], torch.tensor([1]))
        both = criterion(logits, torch.tensor([0, 1]))
        assert both.item() == pytest.approx((1.0 * loss_benign.item() + 4.0 * loss_malignant.item()) / 5.0)

    def test_build_criterion_defaults_and_options(self):
        from src.classification.losses import SmoothedWeightedCrossEntropy, build_criterion, build_validation_criterion

        weight = torch.tensor(WEIGHTS_ALPHA1)
        generator = torch.Generator().manual_seed(1)
        logits = torch.randn(32, 2, generator=generator)
        target = torch.randint(0, 2, (32,), generator=generator)

        default = build_criterion("cross_entropy", weight, 2.0)
        assert type(default) is nn.CrossEntropyLoss and default.label_smoothing == 0.0
        assert default(logits, target).item() == nn.CrossEntropyLoss(weight=weight)(logits, target).item()

        smoothed = build_criterion("cross_entropy", weight, 2.0, label_smoothing=0.1)
        assert isinstance(smoothed, SmoothedWeightedCrossEntropy)
        assert smoothed.label_smoothing == 0.1 and torch.equal(smoothed.weight, weight)

        validation = build_validation_criterion(weight)
        assert type(validation) is nn.CrossEntropyLoss and validation.label_smoothing == 0.0
        assert validation(logits, target).item() == default(logits, target).item()

        with pytest.raises(ValueError):
            build_criterion("focal", weight, 2.0, label_smoothing=0.1)
        for bad in (-0.1, 1.0):
            with pytest.raises(ValueError):
                build_criterion("cross_entropy", weight, 2.0, label_smoothing=bad)
            with pytest.raises(ValueError):
                SmoothedWeightedCrossEntropy(weight, label_smoothing=bad)

    def test_validation_loss_is_computed_without_smoothing(self, tmp_path):
        from src.classification.losses import build_criterion

        weight = torch.tensor([0.6, 1.4])
        smoothed = build_criterion("cross_entropy", weight, 2.0, label_smoothing=0.2)  # enmienda E1
        plain = nn.CrossEntropyLoss(weight=weight)

        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        with_val, model, _, val_loader, _ = _train(tmp_path / "a", epochs=1, criterion=smoothed, val_criterion=plain)
        without_val, _, _, _, _ = _train(tmp_path / "b", epochs=1, criterion=smoothed)

        # val_loss recalculado a mano con el modelo final (1 época): pérdida ponderada SIN suavizado
        expected_plain, _, _ = validate_one_epoch(model, val_loader, plain, CPU)
        assert with_val["val_loss"][0] == pytest.approx(expected_plain, abs=1e-7)
        # sin val_criterion, val_loss sería la suavizada (distinta), y el resto de la trayectoria es idéntica
        assert without_val["val_loss"][0] != pytest.approx(with_val["val_loss"][0], abs=1e-4)
        assert without_val["train_loss"] == with_val["train_loss"]
        assert without_val["val_unweighted_loss"] == with_val["val_unweighted_loss"]

    def test_val_criterion_none_is_the_training_criterion(self, tmp_path):
        """Por defecto la pérdida de validation es la misma de entrenamiento, como antes."""
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.6, 1.4]))
        default, _, _, _, _ = _train(tmp_path / "a", epochs=3, criterion=criterion)
        explicit, _, _, _, _ = _train(tmp_path / "b", epochs=3, criterion=criterion, val_criterion=criterion)
        assert default == explicit


# ---------------------------------------------------------------------------------------------------------
# --augmentation y --image-size
# ---------------------------------------------------------------------------------------------------------
def _find(compose, cls_name):
    return next(t for t in compose.transforms if type(t).__name__ == cls_name)


def _describe(transform):
    """Descripción comparable (algunas transformaciones no definen __repr__ y su repr incluye la dirección de memoria)."""
    name = type(transform).__name__
    if name == "ResizeWithPadding":
        return (name, transform.target_size, transform.fill)
    if name == "Lambda":
        return (name,)
    return (name, repr(transform))


class TestAugmentationAndImageSize:
    def test_standard_is_the_original_augmentation(self):
        from src.classification.transforms import get_train_transforms

        t = get_train_transforms()
        affine, jitter = _find(t, "RandomAffine"), _find(t, "ColorJitter")
        assert list(affine.degrees) == pytest.approx([-10, 10])
        assert list(affine.translate) == pytest.approx([0.05, 0.05])
        assert list(affine.scale) == pytest.approx([0.9, 1.1])
        assert list(jitter.brightness) == pytest.approx([0.9, 1.1]) and list(jitter.contrast) == pytest.approx([0.9, 1.1])
        assert jitter.saturation is None and jitter.hue is None
        assert _find(t, "RandomHorizontalFlip").p == 0.5
        assert _find(t, "ResizeWithPadding").target_size == 224
        assert [_describe(x) for x in get_train_transforms("standard", 224).transforms] == [_describe(x) for x in t.transforms]

    def test_moderate_changes_only_rotation_and_brightness_contrast(self):
        from src.classification.transforms import get_train_transforms

        standard, moderate = get_train_transforms(), get_train_transforms("moderate")
        assert [type(t) for t in standard.transforms] == [type(t) for t in moderate.transforms]
        a_std, a_mod = _find(standard, "RandomAffine"), _find(moderate, "RandomAffine")
        j_std, j_mod = _find(standard, "ColorJitter"), _find(moderate, "ColorJitter")

        assert list(a_mod.degrees) == pytest.approx([-30, 30])
        assert list(j_mod.brightness) == pytest.approx([0.8, 1.2]) and list(j_mod.contrast) == pytest.approx([0.8, 1.2])
        # el resto del aumento es idéntico
        for attribute in ("translate", "scale", "shear", "interpolation", "fill", "center"):
            assert getattr(a_std, attribute) == getattr(a_mod, attribute)
        assert (j_std.saturation, j_std.hue) == (j_mod.saturation, j_mod.hue) == (None, None)
        for t_std, t_mod in zip(standard.transforms, moderate.transforms):
            if type(t_std).__name__ not in ("RandomAffine", "ColorJitter"):
                assert _describe(t_std) == _describe(t_mod)

    def test_validation_transforms_do_not_depend_on_augmentation(self):
        from src.classification.transforms import get_val_transforms

        assert list(inspect.signature(get_val_transforms).parameters) == ["image_size"]

    def test_unknown_augmentation_is_rejected(self):
        from src.classification.transforms import get_train_transforms

        with pytest.raises(ValueError):
            get_train_transforms("heavy")

    def test_image_size_applies_to_train_and_validation(self):
        from src.classification.transforms import get_train_transforms, get_val_transforms

        image = Image.fromarray(np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8), mode="RGB")
        assert get_val_transforms()(image).shape == (3, 224, 224)
        assert get_train_transforms()(image).shape == (3, 224, 224)
        assert get_val_transforms(288)(image).shape == (3, 288, 288)
        assert get_train_transforms("moderate", 288)(image).shape == (3, 288, 288)

    def test_convnext_tiny_accepts_288(self):
        from src.classification.models import create_convnext_tiny_classifier

        model = create_convnext_tiny_classifier(pretrained=False).eval()
        with torch.no_grad():
            assert model(torch.randn(1, 3, 288, 288)).shape == (1, 2)


# ---------------------------------------------------------------------------------------------------------
# --ema-decay
# ---------------------------------------------------------------------------------------------------------
class TestEma:
    def test_update_formula_and_initialisation(self):
        model = nn.Linear(4, 2)
        ema = ModelEMA(model, decay=0.9)
        initial = {k: v.clone() for k, v in model.state_dict().items()}

        for key, value in ema.module.state_dict().items():
            assert torch.equal(value, initial[key])  # inicializada con los pesos de partida
        assert not ema.module.training and not any(p.requires_grad for p in ema.module.parameters())

        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        ema.update(model)
        for key, value in ema.module.state_dict().items():
            assert torch.allclose(value, 0.9 * initial[key] + 0.1 * (initial[key] + 1.0))
        assert all(a.data_ptr() != b.data_ptr() for a, b in zip(ema.module.parameters(), model.parameters()))

    def test_non_float_buffers_are_copied_and_invalid_decay_fails(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        ema = ModelEMA(model, 0.5)
        model.train()(torch.randn(8, 2))
        ema.update(model)
        assert ema.module[1].num_batches_tracked.item() == model[1].num_batches_tracked.item() == 1
        for bad in (0.0, 1.0, 1.5):
            with pytest.raises(ValueError):
                ModelEMA(model, bad)

    def test_ema_does_not_change_the_training_trajectory(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        plain, model_plain, _, _, _ = _train(tmp_path / "a", epochs=3)
        with_ema, model_ema, _, _, _ = _train(tmp_path / "b", epochs=3, ema_decay=0.9)

        assert with_ema["train_loss"] == plain["train_loss"] and with_ema["train_acc"] == plain["train_acc"]
        assert all(torch.equal(a, b) for a, b in zip(model_plain.parameters(), model_ema.parameters()))
        assert with_ema["val_loss"] != plain["val_loss"]  # la validación sí usa otros pesos

    def test_validation_and_checkpoint_use_ema_weights(self, tmp_path):
        history, model, _, val_loader, criterion = _train(tmp_path, epochs=1, ema_decay=0.5)  # el checkpoint es el de la época 1

        checkpoint = torch.load(tmp_path / "best_auc.pth", map_location="cpu", weights_only=True)
        loaded = nn.Linear(4, 2)
        loaded.load_state_dict(checkpoint)

        loaded_loss, _, _ = validate_one_epoch(loaded, val_loader, criterion, CPU)
        live_loss, _, _ = validate_one_epoch(model, val_loader, criterion, CPU)
        assert history["val_loss"][0] == pytest.approx(loaded_loss, abs=1e-7)  # se validó con los pesos guardados (EMA)
        assert abs(history["val_loss"][0] - live_loss) > 1e-4  # y no con los pesos del modelo en entrenamiento
        assert not torch.allclose(checkpoint["weight"], model.weight.detach())

    def test_default_saves_the_live_model(self, tmp_path):
        _, model, _, _, _ = _train(tmp_path, epochs=1)
        checkpoint = torch.load(tmp_path / "best_auc.pth", map_location="cpu", weights_only=True)
        assert all(torch.equal(checkpoint[k], v) for k, v in model.state_dict().items())

    def test_checkpoint_criterion_is_unchanged_with_ema(self, tmp_path):
        history, _, _, _, _ = _train(tmp_path, epochs=6, ema_decay=0.9)
        meta = json.loads((tmp_path / "best_auc_meta.json").read_text())
        assert meta["criterion"] == "max val_roc_auc"
        assert meta["epoch"] == int(np.argmax(history["val_roc_auc"])) + 1


# ---------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------
NEW_ARGUMENTS = (
    "lr_schedule", "warmup_epochs", "cosine_epochs", "lr_min", "layer_decay", "drop_path_rate", "label_smoothing",
    "augmentation", "image_size", "ema_decay",
)


class TestRunExperimentCli:
    def test_new_arguments_default_to_none(self, monkeypatch):
        """Sin ningún argumento nuevo, todos valen None (y entonces no se ejecuta ninguna rama nueva)."""
        pytest.importorskip("torchvision")
        from src.classification import run_experiment

        captured = {}
        original = argparse.ArgumentParser.parse_args

        def spy(self, *args, **kwargs):
            captured["namespace"] = original(self, *args, **kwargs)
            raise RuntimeError("stop")

        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", spy)
        monkeypatch.setattr(sys, "argv", ["run_experiment", "--arch", "convnext_tiny", "--condition", "A"])
        with pytest.raises(RuntimeError):
            run_experiment.main()

        assert {name: getattr(captured["namespace"], name) for name in NEW_ARGUMENTS} == {name: None for name in NEW_ARGUMENTS}

    @pytest.mark.parametrize(
        "arch, extra_args, message",
        [
            ("convnext_tiny", ["--drop-path-rate", "0.3"], "formal"),
            ("convnext_tiny", ["--experiment", "x", "--warmup-epochs", "1"], "solo se usan con --lr-schedule warmup_cosine"),
            ("convnext_tiny", ["--experiment", "x", "--lr-schedule", "warmup_cosine"], "requiere --cosine-epochs"),
            ("convnext_tiny", ["--experiment", "x", "--lr-schedule", "warmup_cosine", "--cosine-epochs", "15", "--lr-min", "1e-3"],
             "debe ser menor que el learning rate pico"),
            ("convnext_tiny", ["--experiment", "x", "--lr-schedule", "cosine"], "invalid choice"),
            ("convnext_tiny", ["--experiment", "x", "--layer-decay", "0"], "(0, 1]"),
            ("resnet50", ["--experiment", "x", "--layer-decay", "0.8"], "solo está implementado"),
            ("convnext_tiny", ["--experiment", "x", "--drop-path-rate", "1.0"], "[0, 1)"),
            ("densenet121", ["--experiment", "x", "--drop-path-rate", "0.3"], "solo está implementado"),
            ("convnext_tiny", ["--experiment", "x", "--label-smoothing", "1.0"], "[0, 1)"),
            ("convnext_tiny", ["--experiment", "x", "--label-smoothing", "0.1", "--loss-type", "focal"], "no se combina"),
            ("convnext_tiny", ["--experiment", "x", "--augmentation", "heavy"], "invalid choice"),
            ("convnext_tiny", ["--experiment", "x", "--image-size", "0"], "positivo"),
            ("convnext_tiny", ["--experiment", "x", "--ema-decay", "1.0"], "(0, 1)"),
        ],
    )
    def test_cli_validates_new_arguments(self, monkeypatch, capsys, arch, extra_args, message):
        pytest.importorskip("torchvision")
        from src.classification import run_experiment

        monkeypatch.setattr(sys, "argv", ["run_experiment", "--arch", arch, "--condition", "A", *extra_args])
        with pytest.raises(SystemExit) as exit_info:
            run_experiment.main()

        assert exit_info.value.code == 2
        assert message in capsys.readouterr().err
