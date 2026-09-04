"""Validation thresholds must respect their physical domain."""

import pandas as pd

from src.assembler import _validate_records


def _record(frequency, dielectric_constant, loss_factor):
    return {
        "paper_id": "synthetic conductive liquid",
        "frequency_mhz": frequency,
        "temperature_c": 25.0,
        "dielectric_constant": dielectric_constant,
        "loss_factor": loss_factor,
        "loss_tangent": loss_factor / dielectric_constant,
        "moisture_content_pct": None,
    }


def test_high_loss_tangent_is_valid_in_rf_but_screened_in_microwave():
    frame = pd.DataFrame([
        _record(50.0, 10.0, 120.0),
        _record(500.0, 10.0, 120.0),
    ])

    result = _validate_records(frame)

    assert result["frequency_mhz"].tolist() == [50.0]
