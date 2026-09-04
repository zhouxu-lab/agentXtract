"""Regression tests for extractor bug fixes.

Covers:
1. filter_cited_values — frequency filter genuinely disabled when the
   screener returned no measurement frequencies.
2. _evaluate_equations — temperature_range_c with None endpoints must not crash.
3. _evaluate_equations — malformed/missing coefficients are skipped, numeric
   strings (incl. Unicode minus) are coerced.
6. _parse_records — loss_tangent is parsed (with float coercion) and survives
   the source filter when ε'/ε'' are present.

No network: these functions are pure (no LLM calls).
"""

import pytest

from src.schema import DielectricRecord, PaperMetadata
from src.table_extractor import (
    _DEFAULT_TEMPS,
    _evaluate_equations,
    _parse_records,
    filter_cited_values,
)


def _make_meta(**kwargs) -> PaperMetadata:
    defaults = {
        "doi": "10.1234/test",
        "title": "Test paper",
        "primary_materials": ["sample powder"],
        "measurement_frequencies_mhz": [915.0, 2450.0],
    }
    defaults.update(kwargs)
    return PaperMetadata(**defaults)


def _make_record(**kwargs) -> DielectricRecord:
    defaults = {
        "material_name": "sample powder",
        "dielectric_constant": 12.0,
        "loss_factor": 3.0,
        "frequency_mhz": 2450.0,
        "temperature_c": 25.0,
        "doi": "10.1234/test",
        "source_table": "Table 1",
    }
    defaults.update(kwargs)
    return DielectricRecord(**defaults)


# -- Fix 1: frequency filter actually disabled when screener gave no freqs ----

def test_freq_filter_disabled_when_screener_empty():
    """With no screener frequencies, records at ANY frequency must be kept
    (previously the always-non-empty ISM set silently kept the filter on)."""
    meta = _make_meta(measurement_frequencies_mhz=[])
    records = [_make_record(frequency_mhz=1800.0)]  # not an ISM frequency
    filtered = filter_cited_values(records, meta)
    assert len(filtered) == 1


def test_freq_filter_still_applies_when_screener_nonempty():
    meta = _make_meta(measurement_frequencies_mhz=[2450.0])
    records = [
        _make_record(frequency_mhz=1800.0),  # wrong freq -> filtered
        _make_record(frequency_mhz=2450.0),  # right freq -> kept
        _make_record(frequency_mhz=915.0),   # common ISM -> kept
    ]
    filtered = filter_cited_values(records, meta)
    assert len(filtered) == 2
    assert all(r.frequency_mhz in (2450.0, 915.0) for r in filtered)


# -- Fix 2: temperature_range_c with None endpoints ---------------------------

@pytest.mark.parametrize("t_range", [(None, 60.0), (20.0, None), (None, None)])
def test_equations_none_temperature_endpoint_falls_back(t_range):
    """None endpoints in temperature_range_c must not raise TypeError;
    the default temperature grid is used instead."""
    meta = _make_meta(temperature_range_c=t_range)
    data = {
        "equations": [
            {
                "material_name": "sample liquid",
                "property": "dielectric_constant",
                "coefficients": [20.0, -0.05],
                "frequency_mhz": 915.0,
                "source_table": "Table 2",
            }
        ]
    }
    records = _evaluate_equations(data, "10.1234/test", meta)
    assert len(records) == len(_DEFAULT_TEMPS)
    assert {r.temperature_c for r in records} == set(_DEFAULT_TEMPS)


def test_equations_valid_temperature_range_still_used():
    meta = _make_meta(temperature_range_c=(20.0, 40.0))
    data = {
        "equations": [
            {
                "material_name": "sample liquid",
                "property": "dielectric_constant",
                "coefficients": [20.0, -0.05],
                "frequency_mhz": 915.0,
                "source_table": "Table 2",
            }
        ]
    }
    records = _evaluate_equations(data, "10.1234/test", meta)
    assert {r.temperature_c for r in records} == {20.0, 30.0, 40.0}


# -- Fix 3: malformed coefficients -------------------------------------------

def test_equation_missing_coefficients_key_skipped():
    """An equation without a 'coefficients' key must be skipped, not KeyError."""
    meta = _make_meta()
    data = {
        "equations": [
            {
                "material_name": "sample liquid",
                "property": "dielectric_constant",
                # no "coefficients" key
                "frequency_mhz": 915.0,
                "source_table": "Table 2",
            }
        ]
    }
    records = _evaluate_equations(data, "10.1234/test", meta)
    assert all(r.dielectric_constant is None for r in records)


def test_equation_string_coefficients_coerced():
    """Numeric-string coefficients (incl. Unicode minus) are coerced to float."""
    meta = _make_meta()
    data = {
        "equations": [
            {
                "material_name": "sample liquid",
                "property": "dielectric_constant",
                "coefficients": ["20.0", "−0.05"],  # Unicode minus U+2212
                "frequency_mhz": 915.0,
                "source_table": "Table 2",
            }
        ]
    }
    records = _evaluate_equations(data, "10.1234/test", meta)
    assert len(records) == len(_DEFAULT_TEMPS)
    r10 = next(r for r in records if r.temperature_c == 10.0)
    assert r10.dielectric_constant == pytest.approx(19.5)


def test_equation_garbage_coefficients_skipped_not_crash(caplog):
    """Non-numeric coefficients must be skipped with a warning, not TypeError."""
    import logging

    meta = _make_meta()
    data = {
        "equations": [
            {
                "material_name": "sample liquid",
                "property": "dielectric_constant",
                "coefficients": ["not a number", 0.5],
                "frequency_mhz": 915.0,
                "source_table": "Table 2",
            },
            {
                "material_name": "sample liquid",
                "property": "loss_factor",
                "coefficients": [2.0, 0.01],
                "frequency_mhz": 915.0,
                "source_table": "Table 2",
            },
        ]
    }
    with caplog.at_level(logging.WARNING, logger="src.table_extractor"):
        records = _evaluate_equations(data, "10.1234/test", meta)
    # Good loss_factor equation still evaluated; bad ε' equation dropped
    assert len(records) == len(_DEFAULT_TEMPS)
    assert all(r.dielectric_constant is None for r in records)
    assert all(r.loss_factor is not None for r in records)
    assert any("malformed equation" in m for m in caplog.messages)


# -- Fix 6: loss_tangent parsed from table records ----------------------------

def test_parse_records_includes_loss_tangent():
    data = {
        "records": [
            {
                "material_name": "sample powder",
                "dielectric_constant": 12.0,
                "loss_factor": 3.0,
                "loss_tangent": 0.25,
                "frequency_mhz": 2450.0,
                "source_table": "Table 1",
            }
        ]
    }
    records = _parse_records(data, "10.1234/test", "test-model")
    assert len(records) == 1
    assert records[0].loss_tangent == pytest.approx(0.25)


def test_parse_records_loss_tangent_string_coerced():
    data = {
        "records": [
            {"material_name": "a", "dielectric_constant": 12.0,
             "loss_tangent": "0.25", "source_table": "Table 1"},
            {"material_name": "b", "dielectric_constant": 12.0,
             "loss_tangent": "n.d.", "source_table": "Table 1"},
            {"material_name": "c", "dielectric_constant": 12.0,
             "source_table": "Table 1"},
        ]
    }
    records = _parse_records(data, "10.1234/test", "test-model")
    assert len(records) == 3
    assert records[0].loss_tangent == pytest.approx(0.25)
    assert records[1].loss_tangent is None  # non-numeric string -> None
    assert records[2].loss_tangent is None  # absent -> None


def test_loss_tangent_kept_through_source_filter():
    """Records with ε' + tan δ keep their loss_tangent; records with ONLY
    loss_tangent are still removed (existing 'no values' semantics kept)."""
    meta = _make_meta()
    keep = _make_record(loss_tangent=0.23)
    drop = _make_record(dielectric_constant=None, loss_factor=None, loss_tangent=0.23)
    filtered = filter_cited_values([keep, drop], meta)
    assert len(filtered) == 1
    assert filtered[0].loss_tangent == pytest.approx(0.23)
