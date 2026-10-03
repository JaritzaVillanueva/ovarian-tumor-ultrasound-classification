"""
Tests unitarios para el protocolo experimental de clasificación R5.
Cubre: transforms, métricas, early stopping/checkpoint y reproducibilidad.
"""
from __future__ import annotations

import json
import os
import random
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

# Transforms
class TestTransforms:
    # Función auxiliar para crear imágenes RGB aleatorias
    def _make_rgb_image(self, width=300, height=200):
        arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        return Image.fromarray(arr, mode="RGB")

    # Función auxiliar para verificar la forma de un tensor
    def _check_tensor_shape(self, tensor, expected_shape):
        assert tensor.shape == expected_shape

    # Pruebas de transforms
    def test_val_transforms_output_shape(self):
        from src.classification.transforms import get_val_transforms
        img = self._make_rgb_image(300, 200)
        t = get_val_transforms()
        tensor = t(img)
        assert tensor.shape == (3, 224, 224)

    # Pruebas de transforms
    def test_train_transforms_output_shape(self):
        from src.classification.transforms import get_train_transforms
        img = self._make_rgb_image(300, 200)
        t = get_train_transforms()
        tensor = t(img)
        assert tensor.shape == (3, 224, 224)

    # Prueba que get_classification_transforms es un alias de get_val_transforms
    def test_get_classification_transforms_is_alias_for_val(self):
        from src.classification.transforms import ( get_classification_transforms, get_val_transforms, )
        img = self._make_rgb_image(150, 150)
        torch.manual_seed(0)
        t_alias = get_classification_transforms()(img)
        torch.manual_seed(0)
        t_val = get_val_transforms()(img)
        assert torch.allclose(t_alias, t_val)

    # Prueba para verificar que los transforms de validación son determinísticos
    def test_val_transforms_are_deterministic(self):
        from src.classification.transforms import get_val_transforms
        img = self._make_rgb_image(400, 300)
        t = get_val_transforms()
        out1 = t(img)
        out2 = t(img)
        assert torch.allclose(out1, out2)

    def test_normalization_uses_imagenet_stats(self):
        """Un tensor de imagen toda blanca debe dar valores cercanos a ImageNet normalizados."""
        from src.classification.transforms import get_val_transforms
        white = Image.fromarray(np.full((224, 224, 3), 255, dtype=np.uint8), mode="RGB")
        t = get_val_transforms()
        tensor = t(white)
        # Pixel blanco normalizado con mean=0.485 y std=0.229
        assert tensor[0, 0, 0] == pytest.approx(2.249, abs=0.01)

    # Prueba que ResizeWithPadding produce una imagen cuadrada
    def test_resize_with_padding_output_square(self):
        from src.classification.transforms import ResizeWithPadding
        img = Image.fromarray(np.zeros((100, 400, 3), dtype=np.uint8), mode="RGB")
        out = ResizeWithPadding(target_size=224)(img)
        assert out.size == (224, 224)

    # Prueba para verificar que los transforms de entrenamiento aceptan imágenes en escala de grises
    def test_train_transforms_accepts_grayscale(self):
        from src.classification.transforms import get_train_transforms
        gray = Image.fromarray(np.zeros((100, 100), dtype=np.uint8), mode="L")
        t = get_train_transforms()
        tensor = t(gray)
        assert tensor.shape == (3, 224, 224)


# Métricas
class TestMetrics:

    # Funcion para crear un caso binario perfecto
    def _perfect_binary_case(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.1, 0.1, 0.9, 0.9])
        return y_true, y_pred, y_prob

    # Funcion para crear un caso binario mixto
    def _mixed_binary_case(self):
        # 3 TN, 1 FP, 1 FN, 1 TP
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.1, 0.1, 0.6, 0.4, 0.9])
        return y_true, y_pred, y_prob

    # Prueba que la especificidad es 1.0 en un caso perfecto
    def test_specificity_perfect(self):
        from src.classification.evaluate import evaluate_predictions
        y_true, y_pred, y_prob = self._perfect_binary_case()
        results = evaluate_predictions(y_true, y_pred, y_prob)
        assert results["specificity"] == pytest.approx(1.0)

    # Prueba que la especificidad es 0.75 en un caso mixto
    def test_specificity_mixed(self):
        from src.classification.evaluate import evaluate_predictions
        y_true, y_pred, y_prob = self._mixed_binary_case()
        results = evaluate_predictions(y_true, y_pred, y_prob)
        # TN=3, FP=1 -> specificity = 3/4 = 0.75
        assert results["specificity"] == pytest.approx(0.75)

    # Prueba que el recall de la clase maligna es 0.5 en un caso mixto
    def test_recall_malignant_mixed(self):
        from src.classification.evaluate import evaluate_predictions
        y_true, y_pred, y_prob = self._mixed_binary_case()
        results = evaluate_predictions(y_true, y_pred, y_prob)
        # TP=1, FN=1 -> recall = 0.5
        assert results["recall_malignant"] == pytest.approx(0.5)

    # Prueba que el promedio de precisión es 1.0 en un caso perfecto
    def test_average_precision_perfect(self):
        from src.classification.evaluate import evaluate_predictions
        y_true, y_pred, y_prob = self._perfect_binary_case()
        results = evaluate_predictions(y_true, y_pred, y_prob)
        assert results["average_precision"] == pytest.approx(1.0)

    # Prueba que la matriz de confusión tiene forma 2x2 incluso si falta una clase
    def test_confusion_matrix_uses_explicit_labels(self):
        from src.classification.evaluate import evaluate_predictions
        # Solo clase 0 presente -> labels=[0,1] produce matriz 2×2 con segunda fila/col ceros
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        y_prob = np.array([0.1, 0.1, 0.2])
        results = evaluate_predictions(y_true, y_pred, y_prob)
        cm = results["confusion_matrix"]
        assert cm.shape == (2, 2)

    # Prueba que todas las claves esperadas están presentes en los resultados de evaluación
    def test_all_keys_present(self):
        from src.classification.evaluate import evaluate_predictions
        y_true, y_pred, y_prob = self._mixed_binary_case()
        results = evaluate_predictions(y_true, y_pred, y_prob)
        for key in [
            "balanced_accuracy", "roc_auc", "average_precision",
            "precision_malignant", "recall_malignant", "f1_malignant",
            "specificity", "accuracy", "confusion_matrix", "classification_report",
        ]:
            assert key in results, f"Falta clave: {key}"

    def test_predict_uses_threshold_05(self):
        """y_pred debe ser 1 cuando y_prob >= 0.5, 0 en caso contrario."""
        class FixedModel(nn.Module):
            def forward(self, x):
                # logits: [0.3, 0.7] -> softmax -> prob_class1 
                return torch.tensor([[0.3, 0.7]]).expand(x.size(0), -1)

        from torch.utils.data import DataLoader, TensorDataset
        from src.classification.evaluate import predict

        images = torch.zeros(4, 3, 224, 224)
        labels = torch.zeros(4, dtype=torch.long)
        ds = TensorDataset(images, labels)
        dl = DataLoader(ds, batch_size=4)

        model = FixedModel()
        y_true, y_pred, y_prob = predict(model, dl, device=torch.device("cpu"))
        assert all(p == 1 for p in y_pred), "prob > 0.5 debe dar y_pred=1"


# Early stopping y checkpoint
class TestEarlyStopping:

    def _make_tiny_model_and_data(self):
        model = nn.Linear(4, 2)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        device = torch.device("cpu")
        x = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))
        from torch.utils.data import DataLoader, TensorDataset
        ds = TensorDataset(x, y)
        loader = DataLoader(ds, batch_size=8)
        return model, criterion, optimizer, device, loader

    def test_early_stopping_triggers_before_max_epochs(self):
        from src.classification.train import train_model
        model, criterion, optimizer, device, loader = self._make_tiny_model_and_data()

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            save_path = f.name
        try:
            # min_delta=999 hace que la mejora nunca supere el umbral → patience se incrementa
            # cada época después de la primera → con patience=2 para en época 3 de 30.
            history = train_model(
                model, loader, loader, criterion, optimizer, device,
                epochs=30, save_path=save_path,
                patience=2, min_delta=999.0, seed=42, use_amp=False,
            )
            assert len(history["val_loss"]) < 30, "Early stopping no disparó"
        finally:
            os.unlink(save_path)
            meta_path = os.path.splitext(save_path)[0] + "_meta.json"
            if os.path.exists(meta_path):
                os.unlink(meta_path)

    def test_checkpoint_saved_on_absolute_minimum(self):
        """El checkpoint se guarda cuando hay mejora < min_delta pero es mínimo absoluto."""
        from src.classification.train import train_model
        model, criterion, optimizer, device, loader = self._make_tiny_model_and_data()

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            save_path = f.name
        try:
            history = train_model(
                model, loader, loader, criterion, optimizer, device,
                epochs=5, save_path=save_path,
                patience=5, min_delta=999.0,  # min_delta enorme → patience nunca resetea
                seed=42, use_amp=False,
            )
            assert os.path.exists(save_path), "No se guardó ningún checkpoint"
        finally:
            os.unlink(save_path)
            meta_path = os.path.splitext(save_path)[0] + "_meta.json"
            if os.path.exists(meta_path):
                os.unlink(meta_path)

    def test_meta_json_created_with_correct_keys(self):
        from src.classification.train import train_model
        model, criterion, optimizer, device, loader = self._make_tiny_model_and_data()

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            save_path = f.name
        try:
            train_model(
                model, loader, loader, criterion, optimizer, device,
                epochs=2, save_path=save_path,
                patience=5, min_delta=0.001, seed=42, use_amp=False,
            )
            meta_path = os.path.splitext(save_path)[0] + "_meta.json"
            assert os.path.exists(meta_path), "No se creó el meta JSON"
            with open(meta_path) as f:
                meta = json.load(f)
            for key in ["epoch", "val_weighted_loss", "val_unweighted_loss", "val_acc"]:
                assert key in meta, f"Falta clave en meta: {key}"
        finally:
            os.unlink(save_path)
            meta_path = os.path.splitext(save_path)[0] + "_meta.json"
            if os.path.exists(meta_path):
                os.unlink(meta_path)

    def test_pth_is_state_dict_compatible(self):
        """El .pth debe poder cargarse directamente con model.load_state_dict."""
        from src.classification.train import train_model
        model, criterion, optimizer, device, loader = self._make_tiny_model_and_data()

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            save_path = f.name
        try:
            train_model(
                model, loader, loader, criterion, optimizer, device,
                epochs=2, save_path=save_path,
                patience=5, min_delta=0.001, seed=42, use_amp=False,
            )
            loaded = torch.load(save_path, map_location="cpu")
            new_model = nn.Linear(4, 2)
            new_model.load_state_dict(loaded)  # debe funcionar sin KeyError
        finally:
            os.unlink(save_path)
            meta_path = os.path.splitext(save_path)[0] + "_meta.json"
            if os.path.exists(meta_path):
                os.unlink(meta_path)

    def test_history_contains_val_unweighted_loss(self):
        from src.classification.train import train_model
        model, criterion, optimizer, device, loader = self._make_tiny_model_and_data()

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            save_path = f.name
        try:
            history = train_model(
                model, loader, loader, criterion, optimizer, device,
                epochs=2, save_path=save_path,
                patience=5, min_delta=0.001, seed=42, use_amp=False,
            )
            assert "val_unweighted_loss" in history
            assert len(history["val_unweighted_loss"]) == len(history["val_loss"])
        finally:
            os.unlink(save_path)
            meta_path = os.path.splitext(save_path)[0] + "_meta.json"
            if os.path.exists(meta_path):
                os.unlink(meta_path)


# Reproducibilidad
class TestReproducibility:

    def test_set_seed_produces_same_torch_random(self):
        from src.classification.train import set_seed
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.allclose(a, b)

    def test_set_seed_produces_same_numpy_random(self):
        from src.classification.train import set_seed
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_set_seed_produces_same_python_random(self):
        from src.classification.train import set_seed
        set_seed(42)
        a = [random.random() for _ in range(5)]
        set_seed(42)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_dataloader_shuffle_reproducible_with_seed(self):
        from src.classification.dataloader import create_dataloader
        from torch.utils.data import TensorDataset

        data = torch.arange(100, dtype=torch.float32).unsqueeze(1)
        labels = torch.zeros(100, dtype=torch.long)
        ds = TensorDataset(data, labels)

        dl1 = create_dataloader(ds, batch_size=10, shuffle=True, num_workers=0, seed=42)
        dl2 = create_dataloader(ds, batch_size=10, shuffle=True, num_workers=0, seed=42)

        batches1 = [batch[0] for batch in dl1]
        batches2 = [batch[0] for batch in dl2]

        for b1, b2 in zip(batches1, batches2):
            assert torch.equal(b1, b2), "Mismo seed debe producir mismo orden de batches"

    def test_amp_fallback_on_cpu(self):
        """use_amp=True no debe fallar si CUDA no está disponible."""
        from src.classification.train import train_model
        model = nn.Linear(4, 2)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        from torch.utils.data import DataLoader, TensorDataset
        ds = TensorDataset(torch.randn(8, 4), torch.randint(0, 2, (8,)))
        loader = DataLoader(ds, batch_size=8)
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            save_path = f.name
        try:
            history = train_model(
                model, loader, loader, criterion, optimizer,
                device=torch.device("cpu"),
                epochs=2, save_path=save_path,
                patience=5, min_delta=0.001, seed=42, use_amp=True,
            )
            assert len(history["train_loss"]) == 2
        finally:
            os.unlink(save_path)
            meta_path = os.path.splitext(save_path)[0] + "_meta.json"
            if os.path.exists(meta_path):
                os.unlink(meta_path)

    def test_two_training_runs_same_seed_same_history(self):
        """Dos runs con la misma seed y mismos pesos iniciales producen historial idéntico."""
        from src.classification.train import train_model, set_seed
        from torch.utils.data import DataLoader, TensorDataset

        set_seed(0)
        x = torch.randn(16, 4)
        y = torch.randint(0, 2, (16,))
        ds = TensorDataset(x, y)
        loader = DataLoader(ds, batch_size=8)

        def run():
            # Seed fija también la inicialización del modelo para garantizar mismas condiciones.
            set_seed(99)
            model = nn.Linear(4, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()
            with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
                save_path = f.name
            try:
                history = train_model(
                    model, loader, loader, criterion, optimizer,
                    device=torch.device("cpu"),
                    epochs=3, save_path=save_path,
                    patience=5, min_delta=0.0, seed=42, use_amp=False,
                )
            finally:
                os.unlink(save_path)
                meta_path = os.path.splitext(save_path)[0] + "_meta.json"
                if os.path.exists(meta_path):
                    os.unlink(meta_path)
            return history

        h1 = run()
        h2 = run()
        assert h1["train_loss"] == pytest.approx(h2["train_loss"], abs=1e-5)
        assert h1["val_loss"] == pytest.approx(h2["val_loss"], abs=1e-5)
