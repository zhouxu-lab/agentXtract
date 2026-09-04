"""Record identity is determined before dielectric values are scored."""

import importlib.util
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "evaluate_matching_mod", Path(__file__).resolve().parents[1] / "evaluate.py"
)
evaluate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate)


def _frame(value: float, moisture=10.0):
    return pd.DataFrame([{
        "paper_id": "synthetic source",
        "paper_key": "synthetic source",
        "material_name": "sample powder",
        "frequency_mhz": 900.0,
        "temperature_c": 25.0,
        "moisture_content_pct": moisture,
        "moisture_basis": "wet",
        "salt_content": None,
        "dielectric_constant": value,
        "loss_factor": 2.0,
    }])


def test_wrong_value_still_matches_then_counts_as_value_error():
    gold = _frame(10.0)
    pred = _frame(100.0)

    pairs, fp, fn = evaluate.match_records(gold, pred)
    metrics = evaluate.compute_metrics(gold, pred, pairs, fp, fn)

    assert pairs == [(0, 0)]
    assert metrics["value_accuracy"] == 0.0


def test_missing_predicted_value_counts_as_value_error():
    gold = _frame(10.0)
    pred = _frame(10.0)
    pred.loc[0, ["dielectric_constant", "loss_factor"]] = float("nan")

    pairs, fp, fn = evaluate.match_records(gold, pred)
    metrics = evaluate.compute_metrics(gold, pred, pairs, fp, fn)

    assert pairs == [(0, 0)]
    assert metrics["value_accuracy"] == 0.0
    assert metrics["value_errors"] == 1


def test_condition_aware_mode_requires_reported_moisture():
    gold = _frame(10.0, moisture=10.0)
    pred = _frame(10.0, moisture=None)

    loose, _, _ = evaluate.match_records(gold, pred)
    strict, _, _ = evaluate.match_records(
        gold, pred, require_reported_moisture=True
    )

    assert loose == [(0, 0)]
    assert strict == []


def test_condition_aware_mode_can_require_matching_moisture_basis():
    gold = _frame(10.0)
    pred = _frame(10.0)
    pred.loc[0, "moisture_basis"] = "dry"

    baseline, _, _ = evaluate.match_records(gold, pred)
    strict, _, _ = evaluate.match_records(
        gold, pred, require_matching_moisture_basis=True
    )

    assert baseline == [(0, 0)]
    assert strict == []


def test_matching_maximizes_pairs_independent_of_prediction_order():
    gold = pd.DataFrame([
        {**_frame(10.0).iloc[0].to_dict(), "temperature_c": 20.0},
        {**_frame(20.0).iloc[0].to_dict(), "temperature_c": 22.0},
    ])
    pred = pd.DataFrame([
        {**_frame(99.0).iloc[0].to_dict(), "temperature_c": 21.0},
        {**_frame(88.0).iloc[0].to_dict(), "temperature_c": 20.0},
    ])

    pairs, fp, fn = evaluate.match_records(gold, pred, temp_tol=1.1)

    assert len(pairs) == 2
    assert fp == []
    assert fn == []
