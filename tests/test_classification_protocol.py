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


# Diagnóstico de checkpoints: best_loss (oficial) y best_auc, con probabilidades de validation por época
class TestCheckpointDiagnostic:
    """
    Cubre el modo opt-in de train_model con auc_save_path y predictions_path:
    1. compatibilidad sin las opciones nuevas;
    2. misma trayectoria de entrenamiento con y sin el diagnóstico;
    3. best_auc coincide con el argmax de val_roc_auc;
    4. integridad del CSV de probabilidades por época.
    """
    EPOCHS = 8
    N_VAL = 40

    @staticmethod
    def _make_dataset(n, seed):
        """Dataset sintético con señal débil, ambas clases y atributo df (image_id, binary_label)."""
        import pandas as pd
        from torch.utils.data import Dataset

        generator = torch.Generator().manual_seed(seed)
        x = torch.randn(n, 4, generator=generator)
        noise = 0.8 * torch.randn(n, generator=generator)
        y = ((x[:, 0] + noise) > 0).long()
        y[0], y[1] = 0, 1  # garantiza ambas clases

        class _TinyDataset(Dataset):
            def __init__(self):
                self.x = x
                self.y = y
                self.df = pd.DataFrame({
                    "image_id": [1000 + 7 * i for i in range(n)],
                    "binary_label": y.numpy(),
                })

            def __len__(self):
                return len(self.y)

            def __getitem__(self, idx):
                return self.x[idx], int(self.y[idx])

        return _TinyDataset()

    def _run(
        self, directory, diagnostic, val_shuffle=False, val_image_ids=None, val_expected_labels=None,
        checkpoint_metric="val_loss", record_predictions=False,
        lr=0.5, epochs=None, patience=100, min_delta=0.0, early_stopping=True,
        class_weights=(0.6, 1.4), loss=None,
    ):
        """Entrena un modelo diminuto con seeds fijas. Devuelve (history, val_dataset)."""
        from torch.utils.data import DataLoader
        from src.classification.train import train_model, set_seed

        train_ds = self._make_dataset(64, seed=1)
        val_ds = self._make_dataset(self.N_VAL, seed=2)

        set_seed(99)
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = loss if loss is not None else nn.CrossEntropyLoss(weight=torch.tensor(list(class_weights)))

        train_loader = DataLoader(
            train_ds, batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(7),
        )
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=val_shuffle)

        kwargs = {}
        if diagnostic or record_predictions:
            kwargs = dict(
                predictions_path=str(directory / "val_predictions.csv"),
                val_image_ids=val_image_ids if val_image_ids is not None else val_ds.df["image_id"].tolist(),
                val_expected_labels=(
                    val_expected_labels if val_expected_labels is not None
                    else val_ds.df["binary_label"].to_numpy()
                ),
            )
        if diagnostic:
            kwargs["auc_save_path"] = str(directory / "best_auc.pth")

        if diagnostic:
            save_name = "best_loss.pth"
        elif checkpoint_metric == "val_roc_auc":
            save_name = "best_auc.pth"
        else:
            save_name = "best.pth"

        history = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            device=torch.device("cpu"),
            epochs=epochs if epochs is not None else self.EPOCHS, save_path=str(directory / save_name),
            patience=patience, min_delta=min_delta, seed=42, use_amp=False,
            checkpoint_metric=checkpoint_metric,
            early_stopping=early_stopping,
            **kwargs,
        )
        return history, val_ds

    # 1. Compatibilidad sin las opciones nuevas
    def test_default_behavior_is_unchanged(self, tmp_path):
        from src.classification.train import EPOCH_VALIDATION_METRICS

        history, _ = self._run(tmp_path, diagnostic=False)

        assert sorted(p.name for p in tmp_path.iterdir()) == ["best.pth", "best_meta.json"]

        with open(tmp_path / "best_meta.json") as f:
            meta = json.load(f)
        assert set(meta) == {"epoch", "val_weighted_loss", "val_unweighted_loss", "val_acc"}

        expected_keys = {"train_loss", "train_acc", "val_loss", "val_unweighted_loss", "val_acc"}
        assert set(history) == expected_keys | set(EPOCH_VALIDATION_METRICS)

    def test_validate_one_epoch_default_returns_three_values(self):
        from torch.utils.data import DataLoader
        from src.classification.train import validate_one_epoch

        val_ds = self._make_dataset(self.N_VAL, seed=2)
        result = validate_one_epoch(
            nn.Linear(4, 2), DataLoader(val_ds, batch_size=16), nn.CrossEntropyLoss(), torch.device("cpu"),
        )
        assert len(result) == 3

    # 2. Misma trayectoria con y sin diagnóstico
    def test_same_training_trajectory_with_and_without_diagnostic(self, tmp_path):
        plain_dir = tmp_path / "plain"
        diag_dir = tmp_path / "diag"
        plain_dir.mkdir()
        diag_dir.mkdir()

        history_plain, _ = self._run(plain_dir, diagnostic=False)
        history_diag, _ = self._run(diag_dir, diagnostic=True)

        assert set(history_plain) == set(history_diag)
        for key in history_plain:
            assert history_diag[key] == pytest.approx(history_plain[key], abs=1e-9), key

        # El checkpoint oficial y su meta son los mismos que sin diagnóstico
        state_plain = torch.load(plain_dir / "best.pth", weights_only=True)
        state_diag = torch.load(diag_dir / "best_loss.pth", weights_only=True)
        assert state_plain.keys() == state_diag.keys()
        for name in state_plain:
            assert torch.equal(state_plain[name], state_diag[name]), name

        with open(plain_dir / "best_meta.json") as f:
            meta_plain = json.load(f)
        with open(diag_dir / "best_loss_meta.json") as f:
            meta_diag = json.load(f)
        assert meta_diag["epoch"] == meta_plain["epoch"]
        assert meta_diag["val_weighted_loss"] == pytest.approx(meta_plain["val_weighted_loss"], abs=1e-9)
        assert meta_diag["criterion"] == "min val_weighted_loss"

    # 3. best_auc coincide con el argmax de val_roc_auc
    def test_best_auc_matches_argmax_of_val_roc_auc(self, tmp_path):
        from torch.utils.data import DataLoader
        from sklearn.metrics import roc_auc_score
        from src.classification.evaluate import predict

        history, val_ds = self._run(tmp_path, diagnostic=True)

        with open(tmp_path / "best_auc_meta.json") as f:
            auc_meta = json.load(f)
        with open(tmp_path / "best_loss_meta.json") as f:
            loss_meta = json.load(f)

        aucs = history["val_roc_auc"]
        assert auc_meta["criterion"] == "max val_roc_auc"
        assert auc_meta["epoch"] == int(np.argmax(aucs)) + 1  # primera época en caso de empate
        assert auc_meta["val_roc_auc"] == pytest.approx(max(aucs), abs=1e-12)
        assert loss_meta["epoch"] == int(np.argmin(history["val_loss"])) + 1

        # El checkpoint guardado reproduce la AUC de su época declarada
        model = nn.Linear(4, 2)
        model.load_state_dict(torch.load(tmp_path / "best_auc.pth", weights_only=True))
        y_true, _, y_prob = predict(model, DataLoader(val_ds, batch_size=16), torch.device("cpu"))
        assert roc_auc_score(y_true, y_prob) == pytest.approx(auc_meta["val_roc_auc"], abs=1e-6)

    # 4. Integridad del CSV de probabilidades
    def test_val_predictions_csv_integrity(self, tmp_path):
        import pandas as pd
        from sklearn.metrics import roc_auc_score

        history, val_ds = self._run(tmp_path, diagnostic=True)
        df = pd.read_csv(tmp_path / "val_predictions.csv")

        epochs_run = len(history["val_loss"])
        expected_ids = val_ds.df["image_id"].tolist()
        expected_labels = val_ds.df["binary_label"].tolist()

        assert list(df.columns) == ["epoch", "image_id", "y_true", "probability_malignant"]
        assert len(df) == epochs_run * self.N_VAL
        assert sorted(df["epoch"].unique()) == list(range(1, epochs_run + 1))
        assert df["probability_malignant"].between(0.0, 1.0).all()
        assert not df.isna().any().any()

        for epoch in range(1, epochs_run + 1):
            rows = df[df["epoch"] == epoch]
            assert rows["image_id"].tolist() == expected_ids
            assert rows["y_true"].tolist() == expected_labels
            # Las probabilidades del CSV reproducen la AUC registrada para esa época
            assert roc_auc_score(rows["y_true"], rows["probability_malignant"]) == pytest.approx(
                history["val_roc_auc"][epoch - 1], abs=1e-6
            )

    # Verificaciones de orden de validation
    def test_shuffled_val_loader_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="shuffle=False"):
            self._run(tmp_path, diagnostic=True, val_shuffle=True)

    def test_mismatched_number_of_image_ids_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="mismo número de elementos"):
            self._run(tmp_path, diagnostic=True, val_image_ids=list(range(self.N_VAL - 1)))

    def test_mismatched_label_order_is_rejected(self, tmp_path):
        wrong_labels = np.array([1 - v for v in self._make_dataset(self.N_VAL, seed=2).df["binary_label"]])
        with pytest.raises(ValueError, match="no coinciden"):
            self._run(tmp_path, diagnostic=True, val_expected_labels=wrong_labels)

    # ── Checkpoint oficial por max val_roc_auc (R4) ──────────────────────────────────────────
    # 1. El checkpoint corresponde al primer argmax de val_roc_auc
    def test_auc_checkpoint_is_first_argmax_of_val_roc_auc(self, tmp_path):
        from torch.utils.data import DataLoader
        from sklearn.metrics import roc_auc_score
        from src.classification.evaluate import predict

        history, val_ds = self._run(tmp_path, diagnostic=False, checkpoint_metric="val_roc_auc")

        # Solo se guarda el checkpoint por AUC (sin best.pth ni checkpoint por pérdida)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["best_auc.pth", "best_auc_meta.json"]

        with open(tmp_path / "best_auc_meta.json") as f:
            meta = json.load(f)

        aucs = history["val_roc_auc"]
        assert meta["criterion"] == "max val_roc_auc"
        assert meta["epoch"] == int(np.argmax(aucs)) + 1  # primera época en caso de empate
        assert meta["val_roc_auc"] == pytest.approx(max(aucs), abs=1e-12)
        # Los demás campos del meta son los valores en la época seleccionada
        assert meta["val_weighted_loss"] == pytest.approx(history["val_loss"][meta["epoch"] - 1], abs=1e-12)
        assert meta["val_acc"] == pytest.approx(history["val_acc"][meta["epoch"] - 1], abs=1e-12)

        # Los pesos guardados reproducen la AUC de esa época
        model = nn.Linear(4, 2)
        model.load_state_dict(torch.load(tmp_path / "best_auc.pth", weights_only=True))
        y_true, _, y_prob = predict(model, DataLoader(val_ds, batch_size=16), torch.device("cpu"))
        assert roc_auc_score(y_true, y_prob) == pytest.approx(meta["val_roc_auc"], abs=1e-6)

    # 2. Seleccionar por AUC no cambia la trayectoria de entrenamiento
    def test_auc_checkpoint_does_not_change_training_trajectory(self, tmp_path):
        loss_dir = tmp_path / "by_loss"
        auc_dir = tmp_path / "by_auc"
        loss_dir.mkdir()
        auc_dir.mkdir()

        history_loss, _ = self._run(loss_dir, diagnostic=False, checkpoint_metric="val_loss")
        history_auc, _ = self._run(auc_dir, diagnostic=False, checkpoint_metric="val_roc_auc")

        assert set(history_loss) == set(history_auc)
        for key in history_loss:
            assert history_auc[key] == pytest.approx(history_loss[key], abs=1e-9), key

    # 3. El early stopping depende solo de val_weighted_loss
    def test_early_stopping_depends_only_on_val_weighted_loss(self, tmp_path):
        def expected_stop_epoch(val_loss, patience, min_delta):
            """Regla de early stopping calculada solo a partir de la val loss ponderada."""
            reference, counter = float("inf"), 0
            for epoch, loss in enumerate(val_loss, start=1):
                if loss < reference - min_delta:
                    reference, counter = loss, 0
                else:
                    counter += 1
                    if counter >= patience:
                        return epoch
            return len(val_loss)

        kwargs = dict(lr=20.0, epochs=15, patience=1, min_delta=0.0)  # lr alto: la val loss oscila
        loss_dir = tmp_path / "by_loss"
        auc_dir = tmp_path / "by_auc"
        loss_dir.mkdir()
        auc_dir.mkdir()

        history_loss, _ = self._run(loss_dir, diagnostic=False, checkpoint_metric="val_loss", **kwargs)
        history_auc, _ = self._run(auc_dir, diagnostic=False, checkpoint_metric="val_roc_auc", **kwargs)

        n_loss = len(history_loss["val_loss"])
        n_auc = len(history_auc["val_loss"])

        assert n_loss < kwargs["epochs"], "El early stopping debía dispararse en este escenario"
        assert n_auc == n_loss
        assert expected_stop_epoch(history_auc["val_loss"], kwargs["patience"], kwargs["min_delta"]) == n_auc

    # 3b. R7: desactivar el early stopping (opt-in) solo elimina la parada anticipada
    def test_early_stopping_is_enabled_by_default(self):
        import inspect
        from src.classification.train import train_model

        assert inspect.signature(train_model).parameters["early_stopping"].default is True

    def test_disabled_early_stopping_runs_all_epochs(self, tmp_path):
        kwargs = dict(lr=20.0, epochs=15, patience=1, min_delta=0.0)  # escenario en el que sí se detiene
        on_dir, off_dir = tmp_path / "on", tmp_path / "off"
        on_dir.mkdir()
        off_dir.mkdir()

        history_on, _ = self._run(on_dir, diagnostic=False, checkpoint_metric="val_roc_auc", **kwargs)
        history_off, _ = self._run(
            off_dir, diagnostic=False, checkpoint_metric="val_roc_auc", early_stopping=False, **kwargs,
        )

        assert len(history_on["val_loss"]) < kwargs["epochs"]
        assert all(len(values) == kwargs["epochs"] for values in history_off.values())

    def test_disabled_early_stopping_keeps_the_same_trajectory_until_the_original_stop(self, tmp_path):
        kwargs = dict(lr=20.0, epochs=15, patience=1, min_delta=0.0)
        on_dir, off_dir = tmp_path / "on", tmp_path / "off"
        on_dir.mkdir()
        off_dir.mkdir()

        history_on, _ = self._run(on_dir, diagnostic=False, checkpoint_metric="val_roc_auc", **kwargs)
        history_off, _ = self._run(
            off_dir, diagnostic=False, checkpoint_metric="val_roc_auc", early_stopping=False, **kwargs,
        )

        n_stop = len(history_on["val_loss"])
        assert set(history_on) == set(history_off)
        for key in history_on:
            assert history_off[key][:n_stop] == pytest.approx(history_on[key], abs=1e-9), key

    def test_disabled_early_stopping_auc_checkpoint_is_first_argmax_over_all_epochs(self, tmp_path):
        history, _ = self._run(
            tmp_path, diagnostic=False, checkpoint_metric="val_roc_auc", early_stopping=False,
            lr=20.0, epochs=15, patience=1, min_delta=0.0,
        )
        with open(tmp_path / "best_auc_meta.json") as f:
            meta = json.load(f)

        assert len(history["val_roc_auc"]) == 15
        assert meta["epoch"] == int(np.argmax(history["val_roc_auc"])) + 1
        assert meta["val_roc_auc"] == max(history["val_roc_auc"])

    def test_resolve_output_dir_only_accepts_r7_subfolders(self, tmp_path):
        pytest.importorskip("torchvision")
        from src.classification.run_experiment import resolve_output_dir

        accepted = resolve_output_dir(tmp_path, "results/classification/r7_optimizacion/early_stopping/seed_42")
        assert accepted == (tmp_path / "results/classification/r7_optimizacion/early_stopping/seed_42").resolve()

        for rejected in (
            "results/classification/r4_benchmark/convnext_tiny/A/seed_42",
            "results/classification/r6_model_selection",
            "results/classification/r7_optimizacion",
            "results/classification/r7_optimizacion/../r4_benchmark",
            str(tmp_path.parent / "fuera"),
        ):
            with pytest.raises(ValueError, match="r7_optimizacion"):
                resolve_output_dir(tmp_path, rejected)

    # 4. Override de --seed (lógica que resuelve la seed efectiva)
    def test_resolve_seed_defaults_to_yaml(self):
        from src.classification.run_experiment import resolve_seed

        assert resolve_seed(42, None) == (42, "yaml")

    def test_resolve_seed_cli_override_changes_only_effective_seed(self):
        from src.classification.run_experiment import resolve_seed

        assert resolve_seed(42, 43) == (43, "cli_override")
        assert resolve_seed(42, 44) == (44, "cli_override")
        # --seed 42 explícito también se registra como override, aunque coincida con el YAML
        assert resolve_seed(42, 42) == (42, "cli_override")

    # 5. Probabilidades del checkpoint seleccionado, por image_id y en el orden correcto
    def test_selected_predictions_match_selected_checkpoint_and_id_order(self, tmp_path):
        import pandas as pd
        from torch.utils.data import DataLoader
        from sklearn.metrics import roc_auc_score
        from src.classification.evaluate import predict
        from src.classification.run_experiment import save_selected_predictions

        history, val_ds = self._run(
            tmp_path, diagnostic=False, checkpoint_metric="val_roc_auc", record_predictions=True,
        )
        with open(tmp_path / "best_auc_meta.json") as f:
            meta = json.load(f)

        # Se recarga el checkpoint seleccionado y se predice como en run_experiment
        model = nn.Linear(4, 2)
        model.load_state_dict(torch.load(tmp_path / "best_auc.pth", weights_only=True))
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
        y_true, _, y_prob = predict(model, val_loader, torch.device("cpu"))

        output_path = tmp_path / "val_predictions_best_auc.csv"
        save_selected_predictions(output_path, meta["epoch"], val_loader, y_true, y_prob)

        selected = pd.read_csv(output_path)
        per_epoch = pd.read_csv(tmp_path / "val_predictions.csv")
        epoch_rows = per_epoch[per_epoch["epoch"] == meta["epoch"]].reset_index(drop=True)

        assert list(selected.columns) == ["epoch", "image_id", "y_true", "probability_malignant"]
        assert len(selected) == len(val_ds)
        assert (selected["epoch"] == meta["epoch"]).all()
        # Todos los image_id, en el orden del dataset de validation
        assert selected["image_id"].tolist() == val_ds.df["image_id"].tolist()
        assert selected["y_true"].tolist() == val_ds.df["binary_label"].tolist()
        # Las probabilidades son las de la época seleccionada de la misma trayectoria
        assert selected["probability_malignant"].to_numpy() == pytest.approx(
            epoch_rows["probability_malignant"].to_numpy(), abs=1e-6
        )
        assert roc_auc_score(selected["y_true"], selected["probability_malignant"]) == pytest.approx(
            meta["val_roc_auc"], abs=1e-6
        )

    def test_selected_predictions_reject_shuffled_loader(self, tmp_path):
        from torch.utils.data import DataLoader
        from src.classification.run_experiment import save_selected_predictions

        val_ds = self._make_dataset(self.N_VAL, seed=2)
        shuffled_loader = DataLoader(val_ds, batch_size=16, shuffle=True)
        labels = val_ds.df["binary_label"].to_numpy()

        with pytest.raises(ValueError, match="shuffle=False"):
            save_selected_predictions(tmp_path / "x.csv", 1, shuffled_loader, labels, np.zeros(len(labels)))


class TestClassWeights:
    """
    R7-H2: ponderación de clases con exponente alpha (--class-weight-alpha).
    Con alpha = 1.0 se conservan los pesos de R4-R6 y B1.
    """
    TRAIN_COUNTS = (713, 87)  # entrenamiento: benignas (0) y malignas (1)

    @staticmethod
    def _labels(n_benign, n_malignant):
        return np.array([0] * n_benign + [1] * n_malignant)

    @staticmethod
    def _original_weights(labels):
        # Fórmula de R4-R6 escrita aparte, para detectar cualquier cambio del comportamiento por defecto
        _, counts = np.unique(labels, return_counts=True)
        return torch.tensor([counts.sum() / (len(counts) * c) for c in counts], dtype=torch.float32)

    def test_default_and_alpha_one_reproduce_original_weights(self):
        from src.classification.utils import compute_class_weights

        labels = self._labels(*self.TRAIN_COUNTS)
        expected = self._original_weights(labels)

        assert torch.equal(compute_class_weights(labels), expected)
        assert torch.equal(compute_class_weights(labels, alpha=1.0), expected)
        assert expected.tolist() == pytest.approx([0.5610, 4.5977], abs=1e-4)

    def test_alpha_half_gives_expected_weights(self):
        from src.classification.utils import compute_class_weights

        weights = compute_class_weights(self._labels(*self.TRAIN_COUNTS), alpha=0.5)

        assert weights.tolist() == pytest.approx([0.8315, 2.3805], abs=1e-3)
        assert (weights[1] / weights[0]).item() == pytest.approx((713 / 87) ** 0.5, rel=1e-5)  # ≈ 2.86 (con alpha = 1: ≈ 8.20)

    def test_weights_follow_label_order(self):
        from src.classification.utils import compute_class_weights

        # Clase 0 mayoritaria: su peso es el menor; con la clase 1 mayoritaria el orden se invierte
        assert compute_class_weights(self._labels(70, 30), alpha=0.5).argmin().item() == 0
        assert compute_class_weights(self._labels(30, 70), alpha=0.5).argmin().item() == 1

    @pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 1.0])
    def test_alpha_scales_ratio_and_keeps_mean_weight_per_sample_equal_to_one(self, alpha):
        from src.classification.utils import compute_class_weights

        n_benign, n_malignant = self.TRAIN_COUNTS
        weights = compute_class_weights(self._labels(n_benign, n_malignant), alpha=alpha).numpy()

        assert weights[1] / weights[0] == pytest.approx((n_benign / n_malignant) ** alpha, rel=1e-5)
        assert (n_benign * weights[0] + n_malignant * weights[1]) / (n_benign + n_malignant) == pytest.approx(1.0, abs=1e-5)

    def test_alpha_zero_is_unweighted(self):
        from src.classification.utils import compute_class_weights

        assert compute_class_weights(self._labels(*self.TRAIN_COUNTS), alpha=0.0).tolist() == pytest.approx([1.0, 1.0])

    @pytest.mark.parametrize("alpha", [-0.1, 1.5])
    def test_invalid_alpha_is_rejected(self, alpha):
        from src.classification.utils import compute_class_weights

        with pytest.raises(ValueError, match="alpha"):
            compute_class_weights(self._labels(*self.TRAIN_COUNTS), alpha=alpha)

    # Compatibilidad con R4-R6 y B1: los pesos por defecto coinciden con los registrados en sus corridas
    @pytest.mark.parametrize(
        "run_dir",
        [
            "r4_benchmark/convnext_tiny/A/seed_42",
            "r4_benchmark/convnext_tiny/A/seed_43",
            "r4_benchmark/convnext_tiny/A/seed_44",
            "r7_optimizacion/early_stopping/seed_42",
        ],
    )
    def test_default_weights_match_recorded_historical_runs(self, run_dir):
        import pandas as pd
        from pathlib import Path
        from src.classification.utils import compute_class_weights

        root = Path(__file__).resolve().parents[1]
        run_config_path = root / "results" / "classification" / run_dir / "run_config.json"
        if not run_config_path.exists():
            pytest.skip(f"No existe {run_config_path}")

        manifest = pd.read_csv(root / "configs" / "splits" / "mmotu_experimental_split_standard.csv")
        train_labels = manifest.loc[manifest["experimental_split"] == "train", "binary_label"].to_numpy()
        with open(run_config_path) as f:
            recorded = json.load(f)["class_weights"]

        assert compute_class_weights(train_labels).tolist() == recorded

    # Particiones y etiquetas sin cambios (frecuencias documentadas en configs/splits/README.md)
    def test_partitions_and_binary_labels_are_unchanged(self):
        import pandas as pd
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        manifest = pd.read_csv(root / "configs" / "splits" / "mmotu_experimental_split_standard.csv")

        counts = manifest.groupby(["experimental_split", "binary_label"]).size().to_dict()
        assert counts == {
            ("train", 0): 713, ("train", 1): 87,
            ("validation", 0): 178, ("validation", 1): 22,
            ("test", 0): 421, ("test", 1): 48,
        }
        assert ((manifest["binary_class"] == "malignant") == (manifest["binary_label"] == 1)).all()

    # B2 mantiene el early stopping activo y el checkpoint por máximo val_roc_auc con los pesos atenuados
    def test_attenuated_weights_keep_early_stopping_and_auc_checkpoint(self, tmp_path):
        from src.classification.utils import compute_class_weights

        weights = tuple(compute_class_weights(self._labels(*self.TRAIN_COUNTS), alpha=0.5).tolist())
        history, _ = TestCheckpointDiagnostic()._run(
            tmp_path, diagnostic=False, checkpoint_metric="val_roc_auc",
            lr=20.0, epochs=15, patience=1, min_delta=0.0, class_weights=weights,  # escenario en el que sí se detiene
        )
        with open(tmp_path / "best_auc_meta.json") as f:
            meta = json.load(f)

        assert len(history["val_loss"]) < 15, "El early stopping debía seguir activo"
        assert meta["criterion"] == "max val_roc_auc"
        assert meta["epoch"] == int(np.argmax(history["val_roc_auc"])) + 1

    @pytest.mark.parametrize(
        "extra_args, message",
        [
            (["--experiment", "x", "--class-weight-alpha", "1.5"], "entre 0 y 1"),
            (["--experiment", "x", "--class-weight-alpha", "-0.1"], "entre 0 y 1"),
            (["--class-weight-alpha", "0.5"], "formal"),
        ],
    )
    def test_cli_rejects_invalid_class_weight_alpha(self, monkeypatch, capsys, extra_args, message):
        import sys

        pytest.importorskip("torchvision")
        from src.classification import run_experiment

        monkeypatch.setattr(sys, "argv", ["run_experiment", "--arch", "convnext_tiny", "--condition", "A", *extra_args])
        with pytest.raises(SystemExit) as exit_info:
            run_experiment.main()

        assert exit_info.value.code == 2
        assert message in capsys.readouterr().err


class TestFocalLoss:
    """
    R7-H3: Focal Loss ponderada (src/classification/losses.py).
    Con gamma = 0 debe reproducir la entropía cruzada ponderada del protocolo R4-R6.
    """
    WEIGHTS = (0.5610, 4.5977)  # pesos originales (alpha = 1.0)

    @staticmethod
    def _batch(n=64, seed=0, scale=3.0):
        generator = torch.Generator().manual_seed(seed)
        logits = scale * torch.randn(n, 2, generator=generator)
        target = torch.randint(0, 2, (n,), generator=generator)
        target[0], target[1] = 0, 1  # ambas clases presentes
        return logits, target

    def test_gamma_zero_reproduces_weighted_cross_entropy_value_and_gradient(self):
        from src.classification.losses import WeightedFocalLoss

        weights = torch.tensor(self.WEIGHTS)
        logits, target = self._batch()

        logits_ce = logits.clone().requires_grad_(True)
        logits_focal = logits.clone().requires_grad_(True)
        loss_ce = nn.CrossEntropyLoss(weight=weights, reduction="mean")(logits_ce, target)
        loss_focal = WeightedFocalLoss(weight=weights, gamma=0.0)(logits_focal, target)
        loss_ce.backward()
        loss_focal.backward()

        assert loss_focal.item() == pytest.approx(loss_ce.item(), rel=1e-6)
        assert torch.allclose(logits_focal.grad, logits_ce.grad, rtol=1e-5, atol=1e-7)

    def test_gamma_two_is_finite_and_backpropagates_also_with_extreme_logits(self):
        from src.classification.losses import WeightedFocalLoss

        weights = torch.tensor(self.WEIGHTS)
        for scale in (3.0, 80.0):  # el segundo caso lleva p_t hacia 0 y hacia 1
            logits, target = self._batch(scale=scale)
            logits.requires_grad_(True)
            loss = WeightedFocalLoss(weight=weights, gamma=2.0)(logits, target)
            loss.backward()

            assert torch.isfinite(loss)
            assert torch.isfinite(logits.grad).all()
            assert logits.grad.abs().sum() > 0

    def test_matches_the_formula_with_class_weights(self):
        from src.classification.losses import WeightedFocalLoss

        weights = torch.tensor(self.WEIGHTS)
        logits, target = self._batch(n=32, seed=1)
        gamma = 2.0

        probs = torch.softmax(logits.double(), dim=1)
        p_t = probs[torch.arange(len(target)), target]
        w = weights.double()[target]
        expected = (-(w * (1 - p_t) ** gamma * torch.log(p_t))).sum() / w.sum()

        assert WeightedFocalLoss(weight=weights, gamma=gamma)(logits, target).item() == pytest.approx(expected.item(), rel=1e-5)

    def test_class_weights_change_the_loss_and_gamma_reduces_it(self):
        from src.classification.losses import WeightedFocalLoss

        logits, target = self._batch(seed=2)
        weighted = WeightedFocalLoss(weight=torch.tensor(self.WEIGHTS), gamma=2.0)(logits, target)
        unweighted = WeightedFocalLoss(weight=None, gamma=2.0)(logits, target)
        cross_entropy = WeightedFocalLoss(weight=torch.tensor(self.WEIGHTS), gamma=0.0)(logits, target)

        assert weighted.item() != pytest.approx(unweighted.item(), rel=1e-3)
        assert weighted.item() < cross_entropy.item()  # (1 - p_t)^gamma <= 1

    def test_negative_gamma_is_rejected(self):
        from src.classification.losses import WeightedFocalLoss

        with pytest.raises(ValueError, match="gamma"):
            WeightedFocalLoss(gamma=-1.0)

    def test_default_criterion_is_the_original_weighted_cross_entropy(self):
        from src.classification.losses import build_criterion
        from src.classification.utils import compute_class_weights

        weights = compute_class_weights(np.array([0] * 713 + [1] * 87))
        criterion = build_criterion("cross_entropy", weights)

        assert type(criterion) is nn.CrossEntropyLoss
        assert criterion.reduction == "mean"
        assert torch.equal(criterion.weight, weights)

        logits, target = self._batch()
        assert torch.equal(criterion(logits, target), nn.CrossEntropyLoss(weight=weights)(logits, target))

    def test_unknown_loss_type_is_rejected(self):
        from src.classification.losses import build_criterion

        with pytest.raises(ValueError, match="loss_type"):
            build_criterion("hinge", torch.tensor(self.WEIGHTS))

    # B3 se entrena con los pesos originales (alpha = 1.0), no con los de B2
    def test_focal_uses_the_original_class_weights(self):
        from src.classification.losses import build_criterion
        from src.classification.utils import compute_class_weights

        weights = compute_class_weights(np.array([0] * 713 + [1] * 87))
        criterion = build_criterion("focal", weights, 2.0)

        assert criterion.weight.tolist() == pytest.approx([0.5610, 4.5977], abs=1e-4)
        assert criterion.gamma == 2.0

    # Integración con el bucle de entrenamiento existente
    def test_focal_gamma_zero_gives_the_same_training_trajectory_as_cross_entropy(self, tmp_path):
        from src.classification.losses import WeightedFocalLoss

        ce_dir, focal_dir = tmp_path / "ce", tmp_path / "focal"
        ce_dir.mkdir()
        focal_dir.mkdir()
        helper = TestCheckpointDiagnostic()

        history_ce, _ = helper._run(ce_dir, diagnostic=False, checkpoint_metric="val_roc_auc", lr=1.0, epochs=6, patience=100)
        history_focal, _ = helper._run(
            focal_dir, diagnostic=False, checkpoint_metric="val_roc_auc", lr=1.0, epochs=6, patience=100,
            loss=WeightedFocalLoss(weight=torch.tensor([0.6, 1.4]), gamma=0.0),
        )

        assert set(history_ce) == set(history_focal)
        for key in history_ce:
            assert history_focal[key] == pytest.approx(history_ce[key], abs=1e-5), key

    def test_focal_keeps_early_stopping_auc_checkpoint_and_unweighted_validation_loss(self, tmp_path):
        from src.classification.losses import WeightedFocalLoss

        history, _ = TestCheckpointDiagnostic()._run(
            tmp_path, diagnostic=False, checkpoint_metric="val_roc_auc",
            lr=20.0, epochs=15, patience=1, min_delta=0.0,  # escenario en el que sí se detiene
            loss=WeightedFocalLoss(weight=torch.tensor(self.WEIGHTS), gamma=2.0),
        )
        with open(tmp_path / "best_auc_meta.json") as f:
            meta = json.load(f)

        assert len(history["val_loss"]) < 15, "El early stopping debía seguir activo"
        assert meta["criterion"] == "max val_roc_auc"
        assert meta["epoch"] == int(np.argmax(history["val_roc_auc"])) + 1
        # val_unweighted_loss sigue siendo la entropía cruzada sin pesos, distinta de la pérdida focal ponderada
        assert all(np.isfinite(history["val_unweighted_loss"])) and history["val_unweighted_loss"] != history["val_loss"]

    @pytest.mark.parametrize(
        "extra_args, message",
        [
            (["--loss-type", "focal"], "formal"),
            (["--experiment", "x", "--loss-type", "hinge"], "invalid choice"),
            (["--experiment", "x", "--focal-gamma", "2.0"], "requiere --loss-type focal"),
            (["--experiment", "x", "--loss-type", "cross_entropy", "--focal-gamma", "2.0"], "requiere --loss-type focal"),
            (["--experiment", "x", "--loss-type", "focal", "--focal-gamma", "-1"], "mayor o igual que 0"),
            (["--experiment", "x", "--loss-type", "focal", "--class-weight-alpha", "0.5"], "pesos originales"),
        ],
    )
    def test_cli_validates_loss_type_and_gamma(self, monkeypatch, capsys, extra_args, message):
        import sys

        pytest.importorskip("torchvision")
        from src.classification import run_experiment

        monkeypatch.setattr(sys, "argv", ["run_experiment", "--arch", "convnext_tiny", "--condition", "A", *extra_args])
        with pytest.raises(SystemExit) as exit_info:
            run_experiment.main()

        assert exit_info.value.code == 2
        assert message in capsys.readouterr().err
