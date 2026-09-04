"""Tests for evaluator parsing, identity matching, and resume filenames."""

import numpy as np
import pandas as pd

from benchmark import extracted_filename
from evaluate import _extract_mean, match_records

# ---------------------------------------------------------------------------
# _extract_mean
# ---------------------------------------------------------------------------

def test_extract_mean_plain_numbers():
    assert _extract_mean("87.5") == 87.5
    assert _extract_mean(87.5) == 87.5
    assert _extract_mean(" 42 ") == 42.0
    assert _extract_mean(0) == 0.0


def test_extract_mean_negative():
    # Old implementation split on any '-' and returned None for negatives.
    assert _extract_mean("-0.5") == -0.5
    assert _extract_mean("-12.3") == -12.3
    # Unicode minus
    assert _extract_mean("−0.5") == -0.5


def test_extract_mean_scientific_notation():
    # Old implementation split on the '-' in the exponent and returned None.
    assert _extract_mean("1.2e-3") == 1.2e-3
    assert _extract_mean("1.2E-3") == 1.2e-3
    assert _extract_mean("3.4e+2") == 340.0
    assert _extract_mean("-1.2e-3") == -1.2e-3


def test_extract_mean_uncertainty_forms():
    assert _extract_mean("87.5 ± 1.0") == 87.5
    assert _extract_mean("87.5±1.0") == 87.5
    assert _extract_mean("87.5 +/- 1.0") == 87.5
    assert _extract_mean("87.5+-1.0") == 87.5
    # Negative mean with uncertainty
    assert _extract_mean("-0.5 ± 0.1") == -0.5


def test_extract_mean_invalid():
    assert _extract_mean(None) is None
    assert _extract_mean(float("nan")) is None
    assert _extract_mean("") is None
    assert _extract_mean("   ") is None
    assert _extract_mean("abc") is None


# ---------------------------------------------------------------------------
# match_records separates record identity from value accuracy
# ---------------------------------------------------------------------------

def _make_frames():
    """Small synthetic gold/pred pair.

    gold rows: 3 records at 915 MHz.
    pred rows:
      0 -> matches gold 0 exactly
      1 -> matches gold 1 with 15% value error
      2 -> matches gold 2 with 30% value error

    Dielectric values are outcomes under evaluation, not identity fields.
    """
    gold = pd.DataFrame([
        {"paper_id": "p1", "material_name": "sample powder", "frequency_mhz": 900.0,
         "temperature_c": 20.0, "dielectric_constant": 12.0, "loss_factor": 3.0},
        {"paper_id": "p1", "material_name": "sample powder", "frequency_mhz": 900.0,
         "temperature_c": 40.0, "dielectric_constant": 10.0, "loss_factor": 2.0},
        {"paper_id": "p1", "material_name": "sample powder", "frequency_mhz": 900.0,
         "temperature_c": 60.0, "dielectric_constant": 8.0, "loss_factor": 1.0},
    ])
    pred = pd.DataFrame([
        {"paper_id": "p1", "material_name": "sample powder", "frequency_mhz": 900.0,
         "temperature_c": 20.0, "dielectric_constant": 12.0, "loss_factor": 3.0},
        {"paper_id": "p1", "material_name": "sample powder", "frequency_mhz": 900.0,
         "temperature_c": 40.0, "dielectric_constant": 10.0 * 1.15, "loss_factor": 2.0},
        {"paper_id": "p1", "material_name": "sample powder", "frequency_mhz": 900.0,
         "temperature_c": 60.0, "dielectric_constant": 8.0 * 1.30, "loss_factor": 1.0},
    ])
    return gold, pred


def test_values_do_not_gate_record_identity():
    gold, pred = _make_frames()
    tp, fp, fn = match_records(gold, pred)
    assert sorted(tp) == [(0, 0), (1, 1), (2, 2)]
    assert fp == []
    assert fn == []


def test_match_records_signature_has_no_value_gate():
    import inspect
    params = inspect.signature(match_records).parameters
    assert "value_tol" not in params
    assert "match_gate_tol" not in params


def test_match_records_nan_rows_are_unmatchable():
    gold, pred = _make_frames()
    pred.loc[1, "frequency_mhz"] = np.nan
    tp, fp, _fn = match_records(gold, pred)
    assert (1 in fp) and all(pi != 1 for _, pi in tp)


# ---------------------------------------------------------------------------
# Per-paper resume filename logic
# ---------------------------------------------------------------------------

def test_extracted_filename_slash_replacement():
    assert extracted_filename("synthetic/source-a") == \
        "synthetic_source-a_table.json"
    assert extracted_filename("local_4  Sample") == "local_4  Sample_table.json"


def test_per_paper_resume_done_detection():
    """Papers whose extraction JSON already exists must be skipped on resume."""
    dois = ["10.1000/a", "10.1000/b", "local_c"]
    done_names = {extracted_filename("10.1000/a"), extracted_filename("local_c")}
    missing = [d for d in dois if extracted_filename(d) not in done_names]
    assert missing == ["10.1000/b"]
