"""Evaluation filters must use the same normalized paper identity as matching."""

import importlib.util
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "evaluate_filters_mod", Path(__file__).resolve().parents[1] / "evaluate.py"
)
evaluate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate)


def test_frequency_filter_uses_normalized_paper_key():
    gold = pd.DataFrame({"paper_id": ["sample source"], "frequency_mhz": [900.0]})
    pred = pd.DataFrame({
        "paper_id": ["1. sample source", "1. sample source"],
        "frequency_mhz": [900.0, 2400.0],
    })

    result = evaluate.filter_pred_by_gold_frequencies(gold, pred)

    assert result["frequency_mhz"].tolist() == [900.0]


def test_frequency_filter_uses_requested_tolerance_inclusively():
    gold = pd.DataFrame({"paper_id": ["source"], "frequency_mhz": [100.0]})
    pred = pd.DataFrame({
        "paper_id": ["source", "source"],
        "frequency_mhz": [105.0, 108.0],
    })

    default_result = evaluate.filter_pred_by_gold_frequencies(gold, pred)
    wider_result = evaluate.filter_pred_by_gold_frequencies(
        gold, pred, freq_tol=0.10
    )

    assert default_result["frequency_mhz"].tolist() == [105.0]
    assert wider_result["frequency_mhz"].tolist() == [105.0, 108.0]


def test_legacy_conductivity_is_not_treated_as_salt(tmp_path):
    path = tmp_path / "pred.csv"
    pd.DataFrame({
        "paper_id": ["synthetic source"],
        "salt_content": ["0.5 S/m"],
    }).to_csv(path, index=False)

    result = evaluate.load_pred(path)

    assert result.loc[0, "electrical_conductivity_s_m"] == 0.5
    assert pd.isna(result.loc[0, "salt_content"])
