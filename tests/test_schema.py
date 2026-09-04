"""Tests for Pydantic schema models."""

from src.schema import (
    Complexity,
    CostEntry,
    DielectricRecord,
    ExtractionPriority,
    ExtractionResult,
    PaperCheckpoint,
    PaperMetadata,
    PaperValidationSummary,
    ParsedFigure,
    ParsedPaper,
    ParsedSection,
    ParsedTable,
    PipelineStatus,
    ScreenerResult,
    TableChunk,
    ValidatedRecord,
    ValidationVerdict,
)


def test_pipeline_status_values():
    assert PipelineStatus.PARSED == "parsed"
    assert PipelineStatus.ASSEMBLED == "assembled"


def test_paper_metadata_minimal():
    m = PaperMetadata(doi="10.1234/test")
    assert m.doi == "10.1234/test"
    assert m.title == ""
    assert m.authors == []
    assert m.discovery_source == "local_pdf"


def test_paper_metadata_full():
    m = PaperMetadata(
        doi="synthetic:metadata-source",
        title="Synthetic dielectric measurements",
        authors=["Alice", "Bob"],
        year=2024,
        journal="Example Journal",
        abstract="We measured dielectric properties...",
        primary_materials=["sample powder"],
        measurement_frequencies_mhz=[915.0, 2450.0],
        temperature_range_c=(20.0, 121.0),
        data_tables=["Table 1", "Table 2"],
        skip_tables=["Table 3"],
        estimated_total_records=120,
        measurement_method="open-ended coaxial probe",
    )
    assert m.year == 2024
    assert len(m.authors) == 2
    assert m.primary_materials == ["sample powder"]
    assert m.measurement_frequencies_mhz == [915.0, 2450.0]


def test_parsed_table():
    t = ParsedTable(
        table_id="table_1",
        caption="Dielectric constants at 2450 MHz",
        headers=[["Material", "ε'", "ε''"]],
        rows=[["sample powder", "12.0", "3.0"]],
    )
    assert len(t.rows) == 1
    assert t.headers[0][1] == "ε'"


def test_parsed_table_with_data_rows():
    t = ParsedTable(
        table_id="table_1",
        caption="Test",
        headers=[["Material", "ε'"]],
        data_rows=[{"Material": "sample powder", "ε'": "12.0"}],
    )
    assert len(t.data_rows) == 1
    assert t.data_rows[0]["Material"] == "sample powder"


def test_table_chunk():
    tc = TableChunk(
        table_id="table_1",
        caption="Test",
        headers=[["A", "B"]],
        data_rows=[{"A": "1", "B": "2"}],
        chunk_index=0,
        total_chunks=3,
        condition_label="Moisture = 6%",
    )
    assert tc.chunk_index == 0
    assert tc.total_chunks == 3


def test_parsed_paper():
    p = ParsedPaper(
        doi="10.1234/test",
        metadata=PaperMetadata(doi="10.1234/test", title="Test Paper"),
        sections=[ParsedSection(heading="Introduction", text="Some text", level=1)],
        tables=[ParsedTable(table_id="t1", caption="Table 1")],
        figures=[ParsedFigure(figure_id="f1", caption="Figure 1")],
        full_text="Full text content...",
    )
    assert len(p.sections) == 1
    assert len(p.tables) == 1
    assert len(p.figures) == 1


def test_screener_result():
    sr = ScreenerResult(
        doi="10.1234/test",
        estimated_records=5,
        data_sources=["table", "text"],
        extraction_priority=ExtractionPriority.HIGH,
        complexity=Complexity.MODERATE,
    )
    assert sr.extraction_priority == ExtractionPriority.HIGH
    assert sr.estimated_records == 5


def test_dielectric_record():
    r = DielectricRecord(
        material_name="sample powder",
        dielectric_constant=12.0,
        loss_factor=3.0,
        frequency_mhz=2450.0,
        temperature_c=25.0,
        moisture_content_pct=78.0,
        moisture_basis="wet",
        measurement_method="open-ended coaxial probe",
        source_table="Table 2",
        source_location="Table 2, row 3",
        extraction_source="table",
        extraction_model="claude-sonnet-4-5-20250929",
        doi="10.1234/test",
    )
    assert r.material_name == "sample powder"
    assert r.dielectric_constant == 12.0
    assert r.moisture_basis == "wet"
    assert r.source_table == "Table 2"
    assert r.extraction_model == "claude-sonnet-4-5-20250929"


def test_extraction_result():
    er = ExtractionResult(
        doi="10.1234/test",
        records=[
            DielectricRecord(material_name="sample powder", dielectric_constant=12.0),
            DielectricRecord(material_name="sample gel", dielectric_constant=10.0),
        ],
        tables_processed=["Table 1"],
        tables_skipped=["Table 3"],
    )
    assert len(er.records) == 2
    assert er.tables_processed == ["Table 1"]


def test_validation():
    vr = ValidatedRecord(
        record=DielectricRecord(material_name="sample powder", dielectric_constant=12.0),
        verdict=ValidationVerdict.ACCEPTED,
    )
    assert vr.verdict == ValidationVerdict.ACCEPTED

    summary = PaperValidationSummary(
        doi="10.1234/test",
        validated_records=[vr],
        accepted_records=1,
        flagged_records=0,
        rejected_records=0,
    )
    assert summary.accepted_records == 1


def test_paper_checkpoint():
    cp = PaperCheckpoint(
        doi="10.1234/test",
        status=PipelineStatus.PARSED,
        pdf_path="/path/to/paper.pdf",
    )
    assert cp.status == PipelineStatus.PARSED
    assert cp.api_cost_usd == 0.0


def test_cost_entry():
    ce = CostEntry(
        stage="screen",
        model="claude-haiku-4-5-20251001",
        doi="10.1234/test",
        input_tokens=500,
        output_tokens=200,
        cost_usd=0.005,
    )
    assert ce.cost_usd == 0.005


def test_model_serialization_roundtrip():
    record = DielectricRecord(
        material_name="sample matrix",
        dielectric_constant=12.0,
        loss_factor=3.0,
        frequency_mhz=915.0,
        temperature_c=20.0,
        extraction_source="table",
    )
    json_str = record.model_dump_json()
    restored = DielectricRecord.model_validate_json(json_str)
    assert restored.material_name == "sample matrix"
    assert restored.frequency_mhz == 915.0
    assert restored.extraction_source == "table"
