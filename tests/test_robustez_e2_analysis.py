"""
Pruebas del análisis E2 (src/classification/robustez_e2_analysis.py). Solo biblioteca estándar: se ejecutan
también sin torch. Las que leen las 15 corridas reales se omiten si no están en este entorno.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.classification import robustez_e2_analysis as an

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / an.E2_REL / an.PROTOCOL_NAME


def _summary(deltas):
    return an.summarize(deltas)


class TestRecomputedMetrics:
    def test_roc_auc_and_average_precision_known_values(self):
        y, s = [1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]
        assert an.roc_auc(y, s) == pytest.approx(0.75)
        assert an.average_precision(y, s) == pytest.approx(0.5 + (2 / 3) * 0.5)

    def test_ties_count_as_half_in_auc_and_one_threshold_in_ap(self):
        assert an.roc_auc([1, 0], [0.5, 0.5]) == pytest.approx(0.5)
        assert an.average_precision([1, 0], [0.5, 0.5]) == pytest.approx(0.5)

    def test_perfect_ranking(self):
        y, s = [1, 1, 0, 0, 0], [0.9, 0.8, 0.3, 0.2, 0.1]
        assert an.roc_auc(y, s) == pytest.approx(1.0)
        assert an.average_precision(y, s) == pytest.approx(1.0)


class TestReadingRules:
    def test_summary_counts_signs_and_range(self):
        s = _summary([0.1, -0.2, 0.0, 0.3, 0.05])
        assert (s["folds_positivos"], s["folds_negativos"], s["folds_cero"]) == (3, 1, 1)
        assert s["minimo"] == -0.2 and s["maximo"] == 0.3
        assert s["media"] == pytest.approx(0.05)

    @pytest.mark.parametrize(
        "deltas, consistent, favored",
        [
            ([0.1, 0.1, 0.1, 0.1, 0.1], True, "primera"),
            ([0.1, 0.1, 0.1, 0.1, -0.1], True, "primera"),  # 4 de 5
            ([0.1, 0.1, 0.1, -0.1, -0.1], False, None),  # 3 de 5: no consistente
            ([-0.1, -0.1, -0.1, -0.1, 0.1], True, "segunda"),
            ([0.1, 0.1, 0.1, 0.0, 0.0], False, None),  # el 0 no cuenta en ningún signo
            ([0.1, 0.1, 0.1, 0.1, 0.0], True, "primera"),
        ],
    )
    def test_consistency_needs_four_of_five_with_same_strict_sign(self, deltas, consistent, favored):
        verdict = an.consistency(_summary(deltas))
        assert verdict["consistente"] is consistent and verdict["favorece"] == favored

    @staticmethod
    def _protocol():
        return {
            "estado": "x",
            "reglas_de_lectura": {
                "consistencia": "c", "declarar_mejor": "d", "regla_robustez_de_b0": "b0", "p_valor": "p",
                "metrica_secundaria": "m", "no_se_hace": [],
            },
        }

    @staticmethod
    def _summaries(c1_deltas, c2_deltas):
        rows = []
        for cid, deltas in (("C1", c1_deltas), ("C2", c2_deltas)):
            rows.append({"comparacion": cid, "metrica": "AP", **an.summarize(deltas)})
            rows.append({"comparacion": cid, "metrica": "ROC-AUC", **an.summarize([0.01] * 5)})
        return rows

    COMPARISONS = [
        {"id": "C1", "nombre": "n1", "first": "convnext_a_alpha1", "second": "densenet_a_alpha1"},
        {"id": "C2", "nombre": "n2", "first": "convnext_a_alpha05", "second": "convnext_a_alpha1"},
    ]

    @pytest.mark.parametrize("negatives, applies", [(5, True), (4, True), (3, False), (0, False)])
    def test_b0_rule_applies_with_four_or_more_negative_c1_folds(self, negatives, applies):
        c1 = [-0.05] * negatives + [0.05] * (5 - negatives)
        rules = an.apply_rules(self._summaries(c1, [0.1] * 5), self.COMPARISONS, self._protocol(), True)
        assert rules["regla_robustez_de_b0"]["se_cumple"] is applies
        phrase = rules["regla_robustez_de_b0"]["frase_del_veredicto"]
        assert ("no es robusta a esta partición" in phrase) is applies
        assert ("Esto no modifica R8" in phrase) is applies
        assert ("no concluye que B0 sea robusto" in phrase) is (not applies)

    def test_c2_has_no_specific_rule_and_only_ap_decides_the_verdict(self):
        rules = an.apply_rules(self._summaries([0.1] * 5, [0.1, -0.1, 0.1, -0.1, 0.1]), self.COMPARISONS, self._protocol(), True)
        assert rules["regla_c2"]["existe_en_el_protocolo"] is False
        assert rules["comparaciones"]["C2"]["consistente"] is False
        assert "no se declara" in rules["comparaciones"]["C2"]["frase_del_veredicto"]
        assert rules["comparaciones"]["C1"]["consistente"] is True
        assert rules["metrica_del_veredicto"] == "AP"


class TestSensitivitySummary:
    def test_detects_changes_with_respect_to_current_criterion(self):
        comparison = [{"id": "C2", "nombre": "n", "first": "a", "second": "b"}]
        rows = []
        for criterion, deltas in (("auc_max", [0.1, -0.1, 0.1, -0.1, 0.1]), ("ap_max", [-0.1] * 5), ("loss_min", [0.1, -0.1, 0.1, -0.1, 0.1])):
            for fold, d in enumerate(deltas):
                rows.append({"comparacion": "C2", "criterio": criterion, "fold": fold, "ap_primera": d, "ap_segunda": 0.0, "delta_ap": d, "signo": an.sign_of(d)})
        summary = {(r["comparacion"], r["criterio"]): r for r in an.sensitivity_summary(rows, comparison)}

        assert summary[("C2", "auc_max")]["folds_con_signo_distinto_a_auc_max"] == 0
        assert summary[("C2", "loss_min")]["veredicto_de_consistencia_distinto_a_auc_max"] is False
        ap = summary[("C2", "ap_max")]
        assert ap["folds_con_signo_distinto_a_auc_max"] == 5 - 2  # coincide el signo solo en los 2 folds con delta negativo
        assert ap["signo_de_la_media_distinto_a_auc_max"] is True
        assert ap["consistente_4_de_5"] is True and ap["veredicto_de_consistencia_distinto_a_auc_max"] is True


@pytest.mark.skipif(not PROTOCOL_PATH.exists(), reason="el protocolo E2 no está en este entorno")
class TestWithProtocol:
    def test_comparisons_are_read_from_the_protocol(self):
        protocol, _ = an.load_protocol(PROJECT_ROOT)
        comparisons = {c["id"]: c for c in an.parse_comparisons(protocol)}
        assert (comparisons["C1"]["first"], comparisons["C1"]["second"]) == ("convnext_a_alpha1", "densenet_a_alpha1")
        assert (comparisons["C2"]["first"], comparisons["C2"]["second"]) == ("convnext_a_alpha05", "convnext_a_alpha1")
        assert "regla_c2" not in protocol["reglas_de_lectura"]
        assert len(protocol["comandos_para_falcon"]["corridas"]) == 15

    def test_analysis_runs_on_real_runs_and_does_not_overwrite(self, tmp_path):
        protocol, _ = an.load_protocol(PROJECT_ROOT)
        missing = [r["salida"] for r in protocol["comandos_para_falcon"]["corridas"]
                   if not (PROJECT_ROOT / r["salida"] / "validation_metrics.json").exists()]
        if missing:
            pytest.skip("faltan corridas E2 en este entorno (git pull)")

        code, results = an.run_analysis(PROJECT_ROOT, tmp_path, verbose=False)
        assert code == 0 and all(c["ok"] for c in results["checks"])
        assert {p.name for p in tmp_path.iterdir()} == set(an.OUTPUT_FILES.values())
        assert len(results["metrics"]) == 15 and len(results["deltas"]) == 2 * 2 * 5

        with pytest.raises(FileExistsError):
            an.run_analysis(PROJECT_ROOT, tmp_path, verbose=False)
