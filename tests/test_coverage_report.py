import json

import pandas as pd
import pytest

from src.assembler import _flatten_records, _resolve_metadata, _write_paper_coverage
from src.schema import (
    DATABASE_COLUMN_NAMES,
    DielectricRecord,
    ExtractionResult,
    PaperMetadata,
    ParsedPaper,
)


def _paper(uid: str, title: str, doi: str = "10.1/shared") -> ParsedPaper:
    return ParsedPaper(
        paper_uid=uid,
        doi=doi,
        pdf_path=f"data/corpus/{title}.pdf",
        metadata=PaperMetadata(doi=doi, title=title),
    )


def test_export_schema_preserves_stable_identity_and_extraction_provenance():
    assert "paper_uid" in DATABASE_COLUMN_NAMES
    assert "doi" in DATABASE_COLUMN_NAMES
    assert "source_location" in DATABASE_COLUMN_NAMES
    assert "extraction_source" in DATABASE_COLUMN_NAMES
    assert "extraction_model" in DATABASE_COLUMN_NAMES


def test_uid_join_survives_duplicate_doi():
    paper_a = _paper("a" * 12, "Paper A")
    paper_b = _paper("b" * 12, "Paper B")
    extracted = ExtractionResult(
        paper_uid=paper_b.paper_uid,
        doi=paper_b.doi,
        records=[DielectricRecord(
            material_name="sample",
            paper_uid=paper_b.paper_uid,
            dielectric_constant=2.0,
            loss_factor=0.1,
        )],
    )
    rows = _flatten_records(
        [extracted],
        {paper_a.paper_uid: paper_a, paper_b.paper_uid: paper_b},
    )
    assert rows[0]["title"] == "Paper B"


def test_metadata_resolution_restores_uid_after_legacy_postprocessing():
    paper = _paper("a" * 12, "Synthetic Source", "synthetic:source-a")
    legacy_row = pd.DataFrame([{
        "paper_id": "synthetic source",
        "paper_uid": "",
        "doi": "synthetic:source-a",
        "title": "",
        "authors": "",
        "year": None,
        "journal": "",
        "measurement_method": "",
    }])

    resolved = _resolve_metadata(legacy_row, {paper.paper_uid: paper})
    assert resolved.loc[0, "paper_uid"] == paper.paper_uid
    assert resolved.loc[0, "title"] == "Synthetic Source"


def test_coverage_report_exposes_zero_yield_and_strict_mode(tmp_path):
    contributor = _paper("a" * 12, "Contributor", "10.1/a")
    missing = _paper("b" * 12, "Missing", "10.1/b")
    extracted = ExtractionResult(
        paper_uid=contributor.paper_uid,
        doi=contributor.doi,
        records=[DielectricRecord(
            material_name="sample",
            paper_uid=contributor.paper_uid,
            dielectric_constant=2.0,
            loss_factor=0.1,
        )],
    )
    assembled = pd.DataFrame([{"paper_uid": contributor.paper_uid}])
    papers = {contributor.paper_uid: contributor, missing.paper_uid: missing}

    path = _write_paper_coverage(assembled, [extracted], papers, tmp_path)
    ledger = pd.read_csv(path)
    outcomes = dict(zip(ledger["title"], ledger["outcome"]))
    assert outcomes == {
        "Contributor": "contributed",
        "Missing": "no_extraction_artifact",
    }
    summary = json.loads((tmp_path / "paper_coverage_summary.json").read_text())
    assert summary["selected_papers"] == 2
    assert summary["zero_yield_papers"] == 1

    with pytest.raises(RuntimeError, match="Strict coverage failed"):
        _write_paper_coverage(
            assembled, [extracted], papers, tmp_path, strict=True
        )


def test_partial_rows_do_not_hide_incomplete_extraction(tmp_path):
    paper = _paper("a" * 12, "Partial", "10.1/a")
    extracted = ExtractionResult(
        paper_uid=paper.paper_uid,
        doi=paper.doi,
        complete=False,
        incomplete_reason="one table failed",
        records=[DielectricRecord(
            material_name="sample",
            paper_uid=paper.paper_uid,
            dielectric_constant=2.0,
            loss_factor=0.1,
        )],
    )
    assembled = pd.DataFrame([{"paper_uid": paper.paper_uid}])

    path = _write_paper_coverage(
        assembled, [extracted], {paper.paper_uid: paper}, tmp_path
    )
    assert pd.read_csv(path).loc[0, "outcome"] == "extraction_incomplete"
    summary = json.loads((tmp_path / "paper_coverage_summary.json").read_text())
    assert summary["contributing_papers"] == 0
    assert summary["incomplete_papers"] == 1
    with pytest.raises(RuntimeError, match="Strict coverage failed"):
        _write_paper_coverage(
            assembled, [extracted], {paper.paper_uid: paper}, tmp_path,
            strict=True,
        )
