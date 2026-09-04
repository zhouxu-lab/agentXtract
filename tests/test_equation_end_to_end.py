"""Synthetic end-to-end checks for equation-derived records."""

from src.schema import PaperMetadata, ParsedPaper, ParsedTable
from src.table_extractor import (
    _enrich_condition_metadata_from_source,
    evaluate_equations_with_report,
)

FREQUENCIES = [90.0, 900.0]


def _payload() -> dict:
    equations = []
    for frequency in FREQUENCIES:
        equations.extend([
            {
                "material_name": "synthetic powder",
                "property": "dielectric_constant",
                "variables": {
                    "1": "moisture_content_pct",
                    "2": "temperature_c",
                },
                "subscripts": {
                    "alpha 0": "5.0",
                    "alpha 1": "3.0 × 10 - 1",
                    "alpha 2": "- 1.0 × 10 - 2",
                    "alpha 12": "5.0 × 10 - 4",
                },
                "domain": {
                    "moisture_content_pct": [10, 30],
                    "temperature_c": [20, 80],
                },
                "frequency_mhz": frequency,
                "source_table": "Synthetic Table A",
            },
            {
                "material_name": "synthetic powder",
                "property": "loss_factor",
                "variables": {
                    "1": "moisture_content_pct",
                    "2": "temperature_c",
                },
                "subscripts": {
                    "alpha 0": "0.5",
                    "alpha 1": "5.0 × 10 - 2",
                    "alpha 2": "2.0 × 10 - 3",
                    "alpha 12": "1.0 × 10 - 4",
                },
                "domain": {
                    "moisture_content_pct": [10, 30],
                    "temperature_c": [20, 80],
                },
                "frequency_mhz": frequency,
                "source_table": "Synthetic Table B",
            },
        ])
    return {"equations": equations}


def _metadata() -> PaperMetadata:
    return PaperMetadata(
        doi="synthetic:equation-source",
        title="Synthetic multivariate response surface",
        measurement_frequencies_mhz=FREQUENCIES,
        temperature_range_c=(20.0, 80.0),
        moisture_range_pct=(10.0, 30.0),
        moisture_levels_pct=[10.0, 20.0, 30.0],
        moisture_basis="wet",
    )


def test_multivariate_payload_yields_paired_records():
    records, report = evaluate_equations_with_report(
        _payload(),
        "synthetic:equation-source",
        _metadata(),
        paper_id="synthetic-equation-source",
    )
    assert report.models_built == 4
    assert report.models_unparsed == 0
    assert records
    assert {record.frequency_mhz for record in records} == set(FREQUENCIES)
    assert all(record.data_provenance == "equation_derived" for record in records)
    assert all(record.extraction_source == "equation" for record in records)
    assert all(record.model_expression for record in records)
    assert all(
        record.dielectric_constant is not None and record.loss_factor is not None
        for record in records
    )


def test_implausible_output_is_reported_and_removed():
    payload = {"equations": [{
        "material_name": "synthetic powder",
        "property": "loss_factor",
        "expression": "-10 - M",
        "variables": ["moisture_content_pct"],
        "domain": {"moisture_content_pct": [10, 30]},
        "frequency_mhz": 900.0,
        "source_table": "Synthetic Table C",
    }]}
    records, report = evaluate_equations_with_report(
        payload,
        "synthetic:equation-source",
        _metadata(),
        "synthetic-equation-source",
    )
    assert records == []
    assert report.points_implausible > 0
    assert any("implausible" in message for message in report.messages)


def test_reported_condition_levels_are_used():
    records, _ = evaluate_equations_with_report(
        _payload(),
        "synthetic:equation-source",
        _metadata(),
        "synthetic-equation-source",
    )
    assert sorted({record.moisture_content_pct for record in records}) == [10.0, 20.0, 30.0]
    assert {record.moisture_basis for record in records} == {"wet"}


def test_model_without_a_grounded_grid_reports_reason():
    payload = {"equations": [{
        "material_name": "synthetic granules",
        "property": "dielectric_constant",
        "expression": "1.5 + 0.4 * bulk_density",
        "variables": ["bulk_density"],
        "frequency_mhz": 2400.0,
        "source_table": "Synthetic Table D",
    }]}
    records, report = evaluate_equations_with_report(
        payload, "synthetic:unbounded", None, "synthetic-unbounded"
    )
    assert records == []
    assert report.models_built == 1
    assert any("no evaluable grid" in message for message in report.messages)


def test_frequency_axis_converts_ghz_model_units_to_mhz_storage():
    payload = {"equations": [
        {
            "material_name": "synthetic granules",
            "property": prop,
            "expression": expression,
            "variables": ["frequency"],
            "domain": {"frequency": [5, 15]},
            "source_table": "Synthetic Table E",
        }
        for prop, expression in (
            ("dielectric_constant", "2.5 + 0.02 * F"),
            ("loss_factor", "0.1 + 0.01 * F"),
        )
    ]}
    metadata = PaperMetadata(
        doi="synthetic:frequency-source",
        measurement_frequencies_mhz=[5000, 10000, 15000],
    )
    records, _ = evaluate_equations_with_report(
        payload, "synthetic:frequency-source", metadata, "synthetic-frequency-source"
    )
    assert {record.frequency_mhz for record in records} == {5000, 10000, 15000}


def test_condition_metadata_is_grounded_in_source_content():
    from_table = ParsedPaper(
        doi="synthetic:condition-table",
        metadata=PaperMetadata(doi="synthetic:condition-table"),
        tables=[ParsedTable(
            table_id="synthetic-table",
            headers=[["Moisture Content (w.b.)", "Density"]],
            rows=[["15.0 ± 1.0%", "1.1"], ["75.0 ± 1.0%", "0.9"]],
        )],
    )
    _enrich_condition_metadata_from_source(from_table)
    assert from_table.metadata.moisture_levels_pct == [15.0, 75.0]
    assert from_table.metadata.moisture_basis == "wet"

    from_text = ParsedPaper(
        doi="synthetic:condition-text",
        metadata=PaperMetadata(doi="synthetic:condition-text"),
        full_text="sample moisture content ranged from 5 to 30% dry basis",
    )
    _enrich_condition_metadata_from_source(from_text)
    assert from_text.metadata.moisture_range_pct == (5.0, 30.0)
    assert from_text.metadata.moisture_basis == "dry"


def test_empty_source_model_intersection_is_reported():
    payload = {"equations": [{
        "material_name": "synthetic gel",
        "property": "dielectric_constant",
        "expression": "2 + M",
        "variables": ["moisture_content_pct"],
        "domain": {"moisture_content_pct": [10, 20]},
        "frequency_mhz": 900.0,
    }]}
    metadata = PaperMetadata(
        doi="synthetic:empty-intersection",
        moisture_range_pct=(0, 5),
        moisture_levels_pct=[0, 5],
    )
    records, report = evaluate_equations_with_report(
        payload,
        "synthetic:empty-intersection",
        metadata,
        "synthetic-empty-intersection",
    )
    assert records == []
    assert any(
        "no evaluable grid" in message and "reported ranges" in message
        for message in report.messages
    )
