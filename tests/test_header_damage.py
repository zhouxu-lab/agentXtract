"""Detect structurally damaged headers using synthetic table fixtures."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.schema import DielectricRecord, ParsedTable
from src.table_extractor import (
    _conditions_missing,
    _header_has_gaps,
)


def _rec(temp=30.0, moisture=42.0, freq=33.0):
    return DielectricRecord(
        material_name="sample matrix", dielectric_constant=12.0,
        loss_factor=3.0, frequency_mhz=freq, temperature_c=temp,
        moisture_content_pct=moisture, source_table="Synthetic table",
    )


# ---------------------------------------------------------------------------
# Damaged headers
# ---------------------------------------------------------------------------

def test_flattened_multilevel_header_is_detected():
    t = ParsedTable(
        table_id="synthetic_table",
        headers=[["Moisture (%)", "", "Frequency (MHz)", "", "Temperature (C)", ""]],
    )
    assert _header_has_gaps(t)


def test_intact_header_is_not_flagged():
    t = ParsedTable(
        table_id="table_1",
        headers=[["Temperature (C)", "27 MHz", "915 MHz", "2450 MHz"]],
    )
    assert not _header_has_gaps(t)


def test_single_blank_cell_is_tolerated():
    """One empty cell is common in real tables (a row-label column) and is
    not evidence of a collapsed header."""
    t = ParsedTable(
        table_id="table_2",
        headers=[["", "27 MHz", "915 MHz", "2450 MHz"]],
    )
    assert not _header_has_gaps(t)


def test_short_header_is_not_flagged():
    t = ParsedTable(table_id="table_3", headers=[["A", "", ""]])
    assert not _header_has_gaps(t)


def test_missing_header_is_safe():
    assert not _header_has_gaps(ParsedTable(table_id="t", headers=[]))


# ---------------------------------------------------------------------------
# Conditions the header promises but the records lack
# ---------------------------------------------------------------------------

def test_temperature_advertised_but_absent_is_flagged():
    t = ParsedTable(
        table_id="synthetic_table",
        caption="Synthetic measurements at T ( ° C) 10-70.",
        headers=[["M (%)", "f (MHz)", "T ( ° C)"]],
    )
    records = [_rec(temp=None) for _ in range(10)]
    assert _conditions_missing(t, records)


def test_temperature_present_is_not_flagged():
    t = ParsedTable(
        table_id="synthetic_table",
        caption="Synthetic measurements at T ( ° C) 10-70.",
        headers=[["M (%)", "f (MHz)", "T ( ° C)"]],
    )
    records = [_rec(temp=10.0 + 15 * i) for i in range(5)]
    assert not _conditions_missing(t, records)


def test_minority_missing_is_tolerated():
    """Some rows legitimately omit a condition; only a majority is a signal."""
    t = ParsedTable(
        table_id="synthetic_table",
        caption="Dielectric properties versus T ( ° C)",
        headers=[["T ( ° C)", "eps'"]],
    )
    records = [_rec(temp=30.0) for _ in range(8)] + [_rec(temp=None) for _ in range(2)]
    assert not _conditions_missing(t, records)


def test_condition_not_advertised_is_not_required():
    """A table with no temperature column must not be faulted for records
    that carry no temperature."""
    t = ParsedTable(
        table_id="table_2",
        caption="Table 2. Composition of the samples.",
        headers=[["Product", "Protein (%)", "Ash (%)"]],
    )
    records = [_rec(temp=None) for _ in range(10)]
    assert not _conditions_missing(t, records)


def test_no_records_is_safe():
    t = ParsedTable(table_id="t", caption="T ( ° C)", headers=[["T ( ° C)"]])
    assert not _conditions_missing(t, [])
