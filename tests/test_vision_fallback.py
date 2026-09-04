"""Tests for the vision fallback.

The helper _render_table_page() previously had no call site in the codebase,
so the fallback could not be exercised in production.
These tests pin the wiring: when text extraction of a data table comes back
poor, the page is rendered and re-read, and the better of the two results is
kept.
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import table_extractor as te
from src.schema import (
    CostEntry,
    DielectricRecord,
    PaperMetadata,
    ParsedPaper,
    ParsedTable,
)
from src.utils import call_llm


def _paper() -> ParsedPaper:
    return ParsedPaper(
        paper_uid="a1b2c3d4e5f6",
        doi="synthetic:vision-source",
        pdf_path="/fixtures/Synthetic Tables.pdf",
        metadata=PaperMetadata(
            doi="synthetic:vision-source",
            title="Synthetic source with a complex table",
            primary_materials=["sample powder"],
            measurement_frequencies_mhz=[27.0, 915.0],
            data_tables=["Table 1"],
        ),
        tables=[],
    )


def _table(rows=None) -> ParsedTable:
    return ParsedTable(
        table_id="table_1",
        caption="Table 1. Synthetic dielectric measurements.",
        headers=[["Temperature", "27 MHz", "915 MHz"]],
        rows=rows if rows is not None else [
            ["20", "12.0", "10.0"], ["30", "13.0", "11.0"],
            ["40", "14.0", "12.0"], ["50", "15.0", "13.0"],
            ["60", "16.0", "14.0"], ["70", "17.0", "15.0"],
        ],
    )


def _record(dc=12.0, lf=3.0, prov="vision_table", temp=20.0) -> DielectricRecord:
    return DielectricRecord(
        material_name="sample powder", dielectric_constant=dc, loss_factor=lf,
        frequency_mhz=915.0, temperature_c=temp, data_provenance=prov,
        source_table="Table 1",
    )


def _cfg(**overrides) -> dict:
    cfg = {
        "table_extractor": {"model": "claude-haiku-4-5-20251001", "max_tokens": 4096},
        "vision_fallback": {"enabled": True, "model": "claude-sonnet-4-6",
                            "coverage_threshold": 0.5},
        "default_concurrency": 2,
    }
    cfg.update(overrides)
    return cfg


def _equation_table() -> ParsedTable:
    return ParsedTable(
        table_id="table_2",
        caption="Table 2. Regression equations for dielectric properties.",
        headers=[["Frequency", "Dielectric property correlation"]],
        rows=[["915 MHz", "eps' = 60 - 0.1 T; eps'' = 10 - 0.02 T"]],
    )


# ---------------------------------------------------------------------------
# The wiring exists
# ---------------------------------------------------------------------------

def test_vision_helper_now_has_a_call_site():
    src = Path(te.__file__).read_text(encoding="utf-8")
    calls = src.count("_render_table_page(")
    assert calls >= 2, "render helper is defined but still never called"


def test_extract_table_via_vision_is_reachable():
    assert callable(getattr(te, "_extract_table_via_vision", None))


def test_equation_tables_use_stronger_configured_model(monkeypatch):
    called = {}

    async def fake_call_llm(**kwargs):
        called.update(kwargs)
        return '{"equations": []}', CostEntry(
            stage="extract_table", model=kwargs["model"], doi="synthetic:vision-source"
        )

    monkeypatch.setattr(te, "call_llm", fake_call_llm)
    monkeypatch.setattr(te, "load_skill", lambda name: "system prompt")
    cfg = _cfg(
        equation_extractor={"model": "strong-equation-model", "max_tokens": 8192},
        vision_fallback={"enabled": False},
    )

    asyncio.run(te.extract_table(_equation_table(), _paper(), cfg))

    assert called["model"] == "strong-equation-model"
    assert called["max_tokens"] == 8192


def test_coefficient_matrix_is_classified_as_equation_table():
    table = ParsedTable(
        table_id="table_3",
        caption="",
        headers=[["", "", "a", "b", "c", "d", "e", "f", "R2"]],
        rows=[
            ["Sample A", "eps'", "4.0", "0.3", "-0.01", "0.0005"],
            ["", "eps''", "0.5", "0.05", "0.002", "0.0001"],
        ],
    )

    assert te._is_equation_table(table)


def test_measurement_table_does_not_suppress_equation_table(monkeypatch):
    paper = _paper()
    paper.tables = [_table(), _equation_table()]
    processed = []

    async def fake_extract(table, paper, config, **kwargs):
        processed.append(table.table_id)
        return [], CostEntry(stage="extract_table", model="test", doi=paper.doi)

    monkeypatch.setattr(te, "extract_table", fake_extract)
    monkeypatch.setattr(te, "_is_dielectric_table", lambda table, metadata: True)

    asyncio.run(te.extract_all_tables(paper, _cfg()))

    assert processed == ["table_1", "table_2"]


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------

def test_vision_used_when_text_extraction_recovers_nothing(monkeypatch):
    """A data table that yields no records is exactly the case the fallback
    exists for."""
    rendered = {"called": False}

    def fake_render(pdf_path, table, image_dir=None):
        rendered["called"] = True
        return "/tmp/page.png"

    async def fake_call_llm(**kwargs):
        if kwargs.get("images"):
            return (
                (
                    '{"records": [{"material_name": "sample powder", '
                    '"dielectric_constant": 12, "loss_factor": 3, '
                    '"frequency_mhz": 915, "temperature_c": 20}]}'
                ),
                CostEntry(
                    stage="extract_table",
                    model="claude-sonnet-4-6",
                    doi="synthetic:vision-source",
                ),
            )
        return '{"records": []}', CostEntry(stage="extract_table", model="claude-haiku-4-5-20251001", doi="synthetic:vision-source")

    monkeypatch.setattr(te, "_render_table_page", fake_render)
    monkeypatch.setattr(te, "call_llm", fake_call_llm)
    monkeypatch.setattr(te, "load_skill", lambda name: "system prompt")

    records, _cost = asyncio.run(
        te.extract_table(_table(), _paper(), _cfg())
    )

    assert rendered["called"], "vision fallback never fired"
    assert records, "vision result was discarded"
    assert all(r.data_provenance == "vision_table" for r in records)


def test_text_result_kept_when_vision_does_not_improve(monkeypatch):
    async def fake_call_llm(**kwargs):
        if kwargs.get("images"):
            return '{"records": []}', CostEntry(stage="extract_table", model="claude-sonnet-4-6", doi="synthetic:vision-source")
        return (
            (
                '{"records": [{"material_name": "sample powder", '
                '"dielectric_constant": 12, "loss_factor": 3, '
                '"frequency_mhz": 915, "temperature_c": 20}]}'
            ),
            CostEntry(stage="extract_table", model="haiku", doi="synthetic:vision-source"),
        )

    monkeypatch.setattr(
        te, "_render_table_page", lambda p, t, image_dir=None: "/tmp/page.png"
    )
    monkeypatch.setattr(te, "call_llm", fake_call_llm)
    monkeypatch.setattr(te, "load_skill", lambda name: "system prompt")

    records, _ = asyncio.run(te.extract_table(_table(), _paper(), _cfg()))
    assert len(records) == 1
    assert records[0].data_provenance == "measured_table"


def test_vision_disabled_by_config(monkeypatch):
    rendered = {"called": False}
    monkeypatch.setattr(
        te, "_render_table_page",
        lambda p, t, image_dir=None: (
            rendered.__setitem__("called", True) or "/tmp/page.png"
        ),
    )

    async def fake_call_llm(**kwargs):
        return '{"records": []}', CostEntry(stage="extract_table", model="haiku", doi="synthetic:vision-source")

    monkeypatch.setattr(te, "call_llm", fake_call_llm)
    monkeypatch.setattr(te, "load_skill", lambda name: "system prompt")

    cfg = _cfg(vision_fallback={"enabled": False})
    asyncio.run(te.extract_table(_table(), _paper(), cfg))
    assert not rendered["called"]


def test_unrenderable_page_degrades_quietly(monkeypatch):
    monkeypatch.setattr(te, "_render_table_page", lambda p, t, image_dir=None: None)

    async def fake_call_llm(**kwargs):
        return '{"records": []}', CostEntry(stage="extract_table", model="haiku", doi="synthetic:vision-source")

    monkeypatch.setattr(te, "call_llm", fake_call_llm)
    monkeypatch.setattr(te, "load_skill", lambda name: "system prompt")

    records, cost = asyncio.run(te.extract_table(_table(), _paper(), _cfg()))
    assert records == []
    assert cost is not None


def test_vision_api_failure_does_not_kill_the_table(monkeypatch):
    async def fake_call_llm(**kwargs):
        if kwargs.get("images"):
            raise RuntimeError("overloaded_error")
        return (
            (
                '{"records": [{"material_name": "sample powder", '
                '"dielectric_constant": 12, "loss_factor": 3, '
                '"frequency_mhz": 915, "temperature_c": 20}]}'
            ),
            CostEntry(stage="extract_table", model="haiku", doi="synthetic:vision-source"),
        )

    monkeypatch.setattr(
        te, "_render_table_page", lambda p, t, image_dir=None: "/tmp/page.png"
    )
    monkeypatch.setattr(te, "call_llm", fake_call_llm)
    monkeypatch.setattr(te, "load_skill", lambda name: "system prompt")

    records, _ = asyncio.run(te.extract_table(_table(), _paper(), _cfg()))
    assert len(records) == 1


# ---------------------------------------------------------------------------
# Images must never be silently dropped again
# ---------------------------------------------------------------------------

def test_images_to_a_non_anthropic_model_raise():
    with pytest.raises(ValueError, match="only the Anthropic wrapper"):
        asyncio.run(call_llm(
            prompt="read this table", model="gpt-4.1",
            images=[{"path": "/tmp/page.png"}], config={},
        ))


# ---------------------------------------------------------------------------
# A per-paper timeout must not look like a paper with no data
# ---------------------------------------------------------------------------

def test_timeout_keeps_records_extracted_before_the_deadline(monkeypatch):
    """The original code discarded every record when the per-paper timeout
    fired, which is indistinguishable from a paper that genuinely had none."""
    import src.table_extractor as tex

    async def slow_extract_all(paper, config, status_fn=None, sink=None):
        if sink is not None:
            sink.setdefault("records", []).append(_record(prov="measured_table"))
            sink["records"].append(
                _record(dc=13.0, lf=2.0, prov="measured_table", temp=40.0)
            )
        await asyncio.sleep(5)          # never completes within the timeout
        return [], None

    monkeypatch.setattr(tex, "extract_all_tables", slow_extract_all)

    paper = _paper()
    paper.tables = [_table()]
    cfg = _cfg(paper_timeout_seconds=0.05)

    results = asyncio.run(tex.run([(paper, None)], cfg, None, progress=None))
    assert len(results) == 1
    er = results[0]
    assert er.timed_out is True
    assert er.complete is False
    assert len(er.records) == 2, "partial results were discarded again"
    assert "INCOMPLETE" in er.notes


def test_table_failure_is_persisted_as_incomplete(monkeypatch, tmp_path):
    async def fail_table(*args, **kwargs):
        raise RuntimeError("provider failed")

    monkeypatch.setattr(te, "extract_table", fail_table)
    paper = _paper()
    paper.tables = [_table()]
    results = asyncio.run(te.run(
        [(paper, None)], _cfg(output_dir=str(tmp_path)), None, progress=None
    ))

    assert results[0].complete is False
    assert "provider failed" in results[0].incomplete_reason
    artifact = next((tmp_path / "extracted").glob("*_table.json"))
    assert json.loads(artifact.read_text())["complete"] is False


def test_record_merge_respects_basis_conductivity_and_provenance():
    dc = _record(dc=3.0, lf=None, prov="measured_table")
    dc.moisture_content_pct = 20.0
    dc.moisture_basis = "wet"
    dc.electrical_conductivity_s_m = 1.0
    lf = _record(dc=None, lf=1.0, prov="equation_derived")
    lf.moisture_content_pct = 20.0
    lf.moisture_basis = "dry"
    lf.electrical_conductivity_s_m = 2.0

    assert len(te.merge_split_records([dc, lf])) == 2


# ---------------------------------------------------------------------------
# Tables that print e' and e'' on separate rows
# ---------------------------------------------------------------------------

def _split_value_json(n: int, first_value: float = 3.0) -> str:
    """n records, each carrying only a dielectric constant — the shape produced
    by a table whose e' and e'' live on different rows."""
    recs = ", ".join(
        f'{{"material_name": "sample powder", "dielectric_constant": {first_value + i:.1f}, '
        f'"frequency_mhz": 915, "temperature_c": {25 + i * 10}}}'
        for i in range(n)
    )
    return f'{{"records": [{recs}]}}'


def test_split_value_vision_result_is_not_discarded(monkeypatch):
    """Scoring both passes by records holding BOTH values calls a split-value
    table zero however well it was read, which silently threw away the better
    vision result."""
    async def fake_call_llm(**kwargs):
        if kwargs.get("images"):
            return (_split_value_json(6),
                    CostEntry(stage="extract_table", model="claude-sonnet-4-6", doi="synthetic:vision-source"))
        return (_split_value_json(2),
                CostEntry(stage="extract_table", model="haiku", doi="synthetic:vision-source"))

    monkeypatch.setattr(
        te, "_render_table_page", lambda p, t, image_dir=None: "/tmp/page.png"
    )
    monkeypatch.setattr(te, "call_llm", fake_call_llm)
    monkeypatch.setattr(te, "load_skill", lambda name: "system prompt")

    records, _ = asyncio.run(te.extract_table(_table(), _paper(), _cfg()))

    assert len(records) == 6, "the richer vision read was thrown away"
    assert all(r.data_provenance == "vision_table" for r in records)


def test_split_value_text_result_wins_when_it_reads_more(monkeypatch):
    """The comparison still has to point the right way round."""
    async def fake_call_llm(**kwargs):
        if kwargs.get("images"):
            return (_split_value_json(3),
                    CostEntry(stage="extract_table", model="claude-sonnet-4-6", doi="synthetic:vision-source"))
        return (_split_value_json(6),
                CostEntry(stage="extract_table", model="haiku", doi="synthetic:vision-source"))

    monkeypatch.setattr(
        te, "_render_table_page", lambda p, t, image_dir=None: "/tmp/page.png"
    )
    monkeypatch.setattr(te, "call_llm", fake_call_llm)
    monkeypatch.setattr(te, "load_skill", lambda name: "system prompt")

    records, _ = asyncio.run(te.extract_table(_table(), _paper(), _cfg()))

    assert len(records) == 6
    assert all(r.data_provenance == "measured_table" for r in records)


def test_batch_mode_uses_the_same_vision_fallback(monkeypatch, tmp_path):
    """Batch extraction must not bypass the recovery path used in realtime."""
    paper = _paper()
    paper.tables = [_table()]
    called = {"vision": False}

    async def fake_batch(requests, config, poll_interval):
        return {
            request["custom_id"]: (
                '{"records": []}',
                CostEntry(
                    stage="extract_table",
                    model=request["model"],
                    doi=paper.doi,
                ),
            )
            for request in requests
        }

    async def fake_vision(*args, **kwargs):
        called["vision"] = True
        return [
            _record(dc=14.0, lf=2.5, prov="vision_table")
        ], CostEntry(
            stage="extract_table", model="vision", doi=paper.doi
        )

    monkeypatch.setattr(te, "run_batch", fake_batch)
    monkeypatch.setattr(te, "_extract_table_via_vision", fake_vision)
    monkeypatch.setattr(te, "load_skill", lambda name: "system prompt")
    cfg = _cfg(output_dir=str(tmp_path), use_batch_api=True)

    results = asyncio.run(te.run([(paper, None)], cfg))

    assert called["vision"]
    assert len(results) == 1
    assert results[0].records[0].data_provenance == "vision_table"


def test_batch_mode_persists_empty_selected_tables(monkeypatch, tmp_path):
    paper = _paper()
    paper.tables = [_table(rows=[])]
    cfg = _cfg(output_dir=str(tmp_path), use_batch_api=True)

    results = asyncio.run(te.run([(paper, None)], cfg))

    assert len(results) == 1
    assert results[0].records == []
    artifacts = list((tmp_path / "extracted").glob("*_table.json"))
    assert len(artifacts) == 1


def test_missing_batch_response_is_persisted_as_incomplete(monkeypatch, tmp_path):
    paper = _paper()
    paper.tables = [_table()]

    async def missing_batch(*args, **kwargs):
        return {}

    monkeypatch.setattr(te, "run_batch", missing_batch)
    monkeypatch.setattr(te, "load_skill", lambda name: "system prompt")
    cfg = _cfg(output_dir=str(tmp_path), use_batch_api=True)

    results = asyncio.run(te.run([(paper, None)], cfg))

    assert len(results) == 1
    assert results[0].complete is False
    assert "missing result" in results[0].incomplete_reason


def test_realtime_mode_persists_paper_with_no_tables(tmp_path):
    paper = _paper()
    paper.tables = []
    cfg = _cfg(output_dir=str(tmp_path), use_batch_api=False)

    results = asyncio.run(te.run([(paper, None)], cfg))

    assert len(results) == 1
    assert results[0].records == []
    assert len(list((tmp_path / "extracted").glob("*_table.json"))) == 1
