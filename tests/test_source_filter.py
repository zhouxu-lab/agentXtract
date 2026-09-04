"""Tests for source discrimination filter."""

from src.schema import DielectricRecord, PaperMetadata
from src.table_extractor import filter_cited_values


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
        "source_table": "Table 1",  # required by filter_cited_values
    }
    defaults.update(kwargs)
    return DielectricRecord(**defaults)


def test_keeps_valid_records():
    meta = _make_meta()
    records = [_make_record()]
    filtered = filter_cited_values(records, meta)
    assert len(filtered) == 1


def test_removes_no_values():
    meta = _make_meta()
    records = [_make_record(dielectric_constant=None, loss_factor=None)]
    filtered = filter_cited_values(records, meta)
    assert len(filtered) == 0


def test_removes_wrong_frequency():
    meta = _make_meta(measurement_frequencies_mhz=[2450.0])
    records = [_make_record(frequency_mhz=1800.0)]
    filtered = filter_cited_values(records, meta)
    assert len(filtered) == 0


def test_keeps_close_frequency():
    meta = _make_meta(measurement_frequencies_mhz=[2450.0])
    records = [_make_record(frequency_mhz=2450.0)]
    filtered = filter_cited_values(records, meta)
    assert len(filtered) == 1


def test_removes_wrong_material():
    # With >2 primary materials, wrong material is filtered out
    meta = _make_meta(primary_materials=["sample powder", "sample gel", "sample liquid"])
    records = [_make_record(material_name="unrelated matrix")]
    filtered = filter_cited_values(records, meta)
    assert len(filtered) == 0


def test_renames_wrong_material_single_primary():
    # With ≤2 primary materials, wrong-named records are RENAMED (not filtered)
    meta = _make_meta(primary_materials=["sample powder"])
    records = [_make_record(material_name="unrelated matrix")]
    filtered = filter_cited_values(records, meta)
    assert len(filtered) == 1
    assert filtered[0].material_name == "sample powder"


def test_keeps_fuzzy_material_match():
    meta = _make_meta(primary_materials=["sample powder"])
    records = [_make_record(material_name="sample powder (1.3% salt)")]
    filtered = filter_cited_values(records, meta)
    assert len(filtered) == 1


def test_no_filter_when_no_metadata():
    """If paper metadata has no materials, material filter is disabled.
    Common ISM frequencies (915, 2450 MHz) are always accepted even without screener data."""
    meta = _make_meta(primary_materials=[], measurement_frequencies_mhz=[])
    # 915 MHz is in the always-accepted ISM set
    records = [_make_record(material_name="anything", frequency_mhz=915.0)]
    filtered = filter_cited_values(records, meta)
    assert len(filtered) == 1


def test_empty_input():
    meta = _make_meta()
    assert filter_cited_values([], meta) == []
