"""
Pruebas del esquema E2 (validación cruzada de 5 folds): partición, manifiesto por fold, dataset y CLI.
Las de la partición usan solo la biblioteca estándar; las de dataset y CLI requieren torch (Falcon).
"""
from __future__ import annotations
from pathlib import Path
import pytest
from src.classification import cv_folds_e2 as e2

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _manifest(n_benign_train=70, n_malignant_train=9, n_benign_val=20, n_malignant_val=2, n_test=15):
    rows, image_id = [], 1

    def add(label, e1_split, source_split):
        nonlocal image_id
        rows.append({
            "image_id": str(image_id), "image_name": f"{image_id}.JPG", "source_split": source_split,
            "original_class": "5" if label else "1", "binary_label": str(label),
            "binary_class": "malignant" if label else "benign", "experimental_split": e1_split,
        })
        image_id += 1

    for _ in range(n_benign_train): add(0, "train", "train")
    for _ in range(n_malignant_train): add(1, "train", "train")
    for _ in range(n_benign_val): add(0, "validation", "train")
    for _ in range(n_malignant_val): add(1, "validation", "train")
    for i in range(n_test): add(i % 2, "test", "validation")
    return rows


class TestFoldAssignment:
    def test_folds_are_stratified_balanced_and_exclude_test(self):
        manifest = _manifest()
        official = e2.select_official_train(manifest)
        folds = e2.assign_folds(official)

        stats = e2.verify_folds(folds, manifest, expected_n=len(official), verbose=False)
        assert sum(n for n, _ in stats.values()) == 101
        assert max(n for n, _ in stats.values()) - min(n for n, _ in stats.values()) <= 1
        assert max(p for _, p in stats.values()) - min(p for _, p in stats.values()) <= 1
        assert not {r["image_id"] for r in folds} & {r["image_id"] for r in manifest if r["experimental_split"] == "test"}

    def test_assignment_is_deterministic_and_independent_of_input_order(self):
        official = e2.select_official_train(_manifest())
        first = e2.assign_folds(official)
        assert first == e2.assign_folds(list(reversed(official)))
        assert first == e2.assign_folds(official)
        assert [r["fold"] for r in first] != [r["fold"] for r in e2.assign_folds(official, seed=e2.SEED_E2 + 1)]

    def test_select_official_train_rejects_rows_outside_mmotu_train(self):
        manifest = _manifest()
        manifest[0]["source_split"] = "validation"
        with pytest.raises(ValueError, match="train oficial"):
            e2.select_official_train(manifest)

    def test_verify_detects_repeated_and_test_images(self):
        manifest = _manifest()
        official = e2.select_official_train(manifest)
        folds = e2.assign_folds(official)

        with pytest.raises(ValueError, match="repetidas"):
            e2.verify_folds(folds + [dict(folds[0])], manifest, expected_n=len(folds) + 1, verbose=False)

        test_row = next(r for r in manifest if r["experimental_split"] == "test")
        leaked = folds[:-1] + [{**folds[-1], "image_id": test_row["image_id"]}]
        with pytest.raises(ValueError, match="test"):
            e2.verify_folds(leaked, manifest, expected_n=len(folds), verbose=False)


class TestFoldManifest:
    def test_derive_fold_rows_uses_fold_as_validation_and_keeps_physical_folder(self):
        folds = e2.assign_folds(e2.select_official_train(_manifest()))
        for fold in range(e2.N_FOLDS):
            derived = e2.derive_fold_rows(folds, fold)
            validation = [r for r in derived if r["experimental_split"] == "validation"]
            train = [r for r in derived if r["experimental_split"] == "train"]
            assert {r["image_id"] for r in validation} == {r["image_id"] for r in folds if int(r["fold"]) == fold}
            e2.check_train_validation_disjoint([r["image_id"] for r in train], [r["image_id"] for r in validation], expected_n=len(folds))
            assert [r["e1_split"] for r in derived] == [r["e1_split"] for r in folds]

    def test_each_image_is_validation_in_exactly_one_fold(self):
        folds = e2.assign_folds(e2.select_official_train(_manifest()))
        times_validation = {}
        for fold in range(e2.N_FOLDS):
            for row in e2.derive_fold_rows(folds, fold):
                times_validation[row["image_id"]] = times_validation.get(row["image_id"], 0) + (row["experimental_split"] == "validation")
        assert set(times_validation.values()) == {1}

    def test_invalid_fold_and_overlap_are_rejected(self):
        folds = e2.assign_folds(e2.select_official_train(_manifest()))
        with pytest.raises(ValueError, match="fold debe estar"):
            e2.derive_fold_rows(folds, e2.N_FOLDS)
        with pytest.raises(ValueError, match="a la vez"):
            e2.check_train_validation_disjoint(["1", "2"], ["2", "3"], expected_n=4)


class TestCommittedPartition:
    """Verifica el CSV versionado contra el manifiesto operativo (si existen)."""

    def test_committed_csv_matches_operational_manifest_and_regeneration(self):
        folds_path = PROJECT_ROOT / e2.OUTPUT_REL
        manifest_path = PROJECT_ROOT / e2.MANIFEST_REL
        if not folds_path.exists() or not manifest_path.exists():
            pytest.skip("cv_folds_e2.csv o el manifiesto no están en este entorno.")

        manifest = e2.read_csv_rows(manifest_path)
        committed = e2.read_csv_rows(folds_path)
        stats = e2.verify_folds(committed, manifest, verbose=False)

        assert list(committed[0].keys()) == e2.FOLD_COLUMNS
        assert [n for n, _ in stats.values()] == [200] * 5
        assert sum(p for _, p in stats.values()) == 109
        regenerated = e2.assign_folds(e2.select_official_train(manifest))
        assert [(r["image_id"], str(r["fold"])) for r in committed] == [(r["image_id"], str(r["fold"])) for r in regenerated]


class TestDatasetFolderColumn:
    @staticmethod
    def _make_images(root):
        from PIL import Image

        for folder in ("train", "validation"):
            (root / folder / "images").mkdir(parents=True)
        for name, folder in (("1.JPG", "train"), ("2.JPG", "validation")):
            Image.new("RGB", (8, 8)).save(root / folder / "images" / name)

    def test_default_uses_split_folder_and_folder_column_uses_physical_folder(self, tmp_path):
        pytest.importorskip("torch")
        from src.classification.dataset import MMOTUClassificationDataset

        self._make_images(tmp_path)
        csv_path = tmp_path / "split.csv"
        csv_path.write_text(
            "image_id,image_name,binary_label,experimental_split,e1_split\n"
            "1,1.JPG,0,train,train\n"
            "2,2.JPG,1,train,validation\n",  # E2: imagen físicamente en validation/ pero asignada a train
            encoding="utf-8",
        )

        # Con folder_column la imagen 2 se encuentra en su carpeta física.
        ds = MMOTUClassificationDataset(csv_path, tmp_path, split="train", folder_column="e1_split")
        assert [ds[i][1] for i in range(len(ds))] == [0, 1]

        # Por defecto (None) se busca en la carpeta del split, como antes: la imagen 2 no está en train/.
        ds_default = MMOTUClassificationDataset(csv_path, tmp_path, split="train")
        assert ds_default.folder_column is None
        assert ds_default[0][1] == 0
        with pytest.raises(FileNotFoundError):
            ds_default[1]


class TestRunExperimentE2Cli:
    @pytest.mark.parametrize(
        "extra_args, message",
        [
            (["--experiment", "x", "--fold", "0"], "se usan juntos"),
            (["--experiment", "x", "--split-csv", "configs/splits/cv_folds_e2.csv"], "se usan juntos"),
            (["--split-csv", "configs/splits/cv_folds_e2.csv", "--fold", "0"], "formal"),
            (["--experiment", "x", "--split-csv", "configs/splits/cv_folds_e2.csv", "--fold", "0"], "--output-dir"),
            (["--experiment", "x", "--split-csv", "configs/splits/cv_folds_e2.csv", "--fold", "5",
              "--output-dir", "results/classification/r7_optimizacion/robustez_e2/x"], "entre 0 y 4"),
        ],
    )
    def test_cli_validates_split_csv_and_fold(self, monkeypatch, capsys, extra_args, message):
        import sys

        pytest.importorskip("torchvision")
        from src.classification import run_experiment

        monkeypatch.setattr(sys, "argv", ["run_experiment", "--arch", "convnext_tiny", "--condition", "A", *extra_args])
        with pytest.raises(SystemExit) as exit_info:
            run_experiment.main()

        assert exit_info.value.code == 2
        assert message in capsys.readouterr().err
