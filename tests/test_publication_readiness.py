"""Regression tests for publication-readiness and reproducibility invariants."""

import asyncio
import json
import re
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pandas as pd
import pytest

import benchmark as benchmark_module
from benchmark import (
    MODELS,
    _benchmark_artifact_complete,
    _load_base_config,
    _patch_config_for_model,
    _requires_text_extraction,
)
from src import screener, text_extractor, utils
from src.assembler import _deduplicate, _merge_split_records
from src.assembler import run as assemble
from src.audit import run_audit
from src.pipeline import (
    _already_extracted,
    _deduplicate_pdf_paths,
    _discover_corpus_pdfs,
    load_parsed_papers,
)
from src.schema import (
    CostEntry,
    DielectricRecord,
    ExtractionResult,
    PaperMetadata,
    ParsedPaper,
    ScreenerResult,
)
from src.table_extractor import _parse_records, _parse_response

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _tracked_paths() -> list[Path]:
    completed = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [
        path
        for entry in completed.stdout.split(b"\0")
        if entry and (path := REPOSITORY_ROOT / entry.decode("utf-8")).exists()
    ]


def test_tracked_tree_excludes_private_artifact_types_and_data_outputs():
    forbidden_suffixes = {
        ".7z", ".db", ".doc", ".docx", ".duckdb", ".gz", ".parquet",
        ".pdf", ".ppt", ".pptx", ".sqlite", ".sqlite3", ".tar", ".tif",
        ".tiff", ".xls", ".xlsx", ".zip",
    }
    forbidden_names = {
        ".env", "dielectric_properties.csv", "manifest.json",
        "paper_coverage.csv", "paper_coverage_summary.json",
        "run_provenance.json",
    }
    violations = []
    for path in _tracked_paths():
        relative = path.relative_to(REPOSITORY_ROOT)
        if path.suffix.lower() in forbidden_suffixes or path.name.lower() in forbidden_names:
            violations.append(str(relative))
        if relative.parts and relative.parts[0] == "data" and relative.as_posix() != "data/README.md":
            violations.append(str(relative))
    assert not violations, f"Private/generated artifacts are tracked: {sorted(set(violations))}"


def test_tracked_text_excludes_credentials_local_paths_and_project_notes():
    publication_workflow_phrases = [
        "manu" + "script",
        "review" + "er comment",
        "response" + " letter",
        "revision" + "-1",
        "revision" + "-2",
    ]
    secret_patterns = [
        re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
        re.compile(r"(?:sk|rk)-[A-Za-z0-9_-]{20,}"),
        re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        re.compile(r"10\.(?!1000/|1234/)[0-9]{4,9}/[^\s\"']+"),
        re.compile(
            r"(?im)^[ \t]*(?:ANTHROPIC|OPENAI|GEMINI|GOOGLE)_API_KEY[ \t]*="
            r"[ \t]*[^\s#]+"
        ),
    ]
    absolute_path_patterns = [
        re.compile(r"(?i)[A-Z]:[\\/]Users[\\/][^\\/\s]+"),
        re.compile(r"/(?:Users|home)/[^/\s]+"),
    ]
    violations = []
    for path in _tracked_paths():
        if path.suffix.lower() in {".lock"} or "lock" in path.name.lower():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        lower = text.lower()
        for phrase in publication_workflow_phrases:
            if phrase in lower:
                violations.append(f"{path.relative_to(REPOSITORY_ROOT)}: private phrase")
        for pattern in [*secret_patterns, *absolute_path_patterns]:
            if pattern.search(text):
                violations.append(f"{path.relative_to(REPOSITORY_ROOT)}: sensitive pattern")
    assert not violations, "\n".join(violations)


def _record(uid: str, value: float, **kwargs) -> DielectricRecord:
    fields = {
        "material_name": "sample",
        "paper_uid": uid,
        "paper_id": "same filename",
        "frequency_mhz": 915.0,
        "temperature_c": 25.0,
        "dielectric_constant": value,
        "loss_factor": 1.0,
        "source_table": "Table 1",
    }
    fields.update(kwargs)
    return DielectricRecord(**fields)


def _result(uid: str, value: float, **kwargs) -> ExtractionResult:
    return ExtractionResult(
        paper_uid=uid,
        doi=f"10.1/{uid}",
        records=[_record(uid, value, **kwargs)],
    )


def test_table_parser_retains_all_supported_provenance_and_property_fields():
    records = _parse_records(
        {
            "records": [{
                "material_name": "sample",
                "dielectric_constant": 3.0,
                "loss_factor": 0.2,
                "loss_tangent": 0.067,
                "electrical_conductivity_s_m": 1.25,
                "source_table": "Table 4",
                "source_location": "page 7, row 2",
            }]
        },
        "10.1/test",
        "gpt-test",
    )

    assert len(records) == 1
    record = records[0]
    assert record.electrical_conductivity_s_m == 1.25
    assert record.source_location == "page 7, row 2"
    assert record.extraction_model == "gpt-test"


def test_table_response_uses_known_table_id_when_model_omits_it():
    records = _parse_response(
        {"records": [{"material_name": "sample", "dielectric_constant": 3.0}]},
        "10.1/test",
        "gpt-test",
        fallback_source_table="table_4",
    )
    assert records[0].source_table == "Table 4"


def test_distinct_pdfs_with_same_human_id_are_not_merged_or_deduplicated():
    rows = []
    for uid, dc in (("a" * 12, 3.0), ("b" * 12, 3.0)):
        rows.append({
            **_record(uid, dc).model_dump(),
            "moisture_content_pct": None,
            "electrical_conductivity_s_m": None,
        })
    frame = pd.DataFrame(rows)

    merged = _merge_split_records(frame)
    deduplicated = _deduplicate(merged)

    assert set(deduplicated["paper_uid"]) == {"a" * 12, "b" * 12}


def test_ambiguous_replicates_are_preserved():
    base = {
        **_record("a" * 12, 3.0).model_dump(),
        "loss_factor": None,
        "moisture_content_pct": 0.0,
        "electrical_conductivity_s_m": None,
    }
    frame = pd.DataFrame([
        base,
        {**base, "dielectric_constant": 3.5},
        {**base, "dielectric_constant": None, "loss_factor": 0.4},
    ])

    merged = _merge_split_records(frame)

    assert len(merged) == 3
    assert sorted(merged["dielectric_constant"].dropna()) == [3.0, 3.5]


def test_moisture_basis_is_part_of_merge_and_dedup_conditions():
    base = {
        **_record("a" * 12, 3.0).model_dump(),
        "moisture_content_pct": 20.0,
        "electrical_conductivity_s_m": None,
    }
    split = pd.DataFrame([
        {**base, "moisture_basis": "wet", "loss_factor": None},
        {
            **base, "moisture_basis": "dry", "dielectric_constant": None,
            "loss_factor": 1.0,
        },
    ])
    assert len(_merge_split_records(split)) == 2

    complete = pd.DataFrame([
        {**base, "moisture_basis": "wet"},
        {**base, "moisture_basis": "dry"},
    ])
    assert len(_deduplicate(complete)) == 2


def test_dedup_prefers_measured_table_and_is_metadata_deterministic():
    base = {
        **_record("a" * 12, 3.0).model_dump(),
        "moisture_content_pct": None,
        "electrical_conductivity_s_m": None,
    }
    equation = {
        **base,
        "data_provenance": "equation_derived",
        "extraction_source": "equation",
        "measurement_method": "method Z",
    }
    measured = {
        **base,
        "data_provenance": "measured_table",
        "extraction_source": "table",
        "measurement_method": "method A",
    }
    first = _deduplicate(pd.DataFrame([equation, measured]))
    second = _deduplicate(pd.DataFrame([measured, equation]))
    assert first.iloc[0]["data_provenance"] == "measured_table"
    pd.testing.assert_frame_equal(first, second)


def test_assembly_is_stable_and_backfills_narrative_source(tmp_path):
    uid_a, uid_b = "a" * 12, "b" * 12
    narrative = _result(
        uid_a,
        4.0,
        source_table="",
        extraction_source="text",
        data_provenance="measured_text",
    )
    table = _result(uid_b, 3.0)
    config = {"validation": {"enabled": False}, "output_dir": str(tmp_path)}
    first = tmp_path / "first"
    second = tmp_path / "second"

    assemble([narrative, table], {}, first, config)
    assemble([table, narrative], {}, second, config)

    assert (first / "dielectric_properties.csv").read_bytes() == (
        second / "dielectric_properties.csv"
    ).read_bytes()
    exported = pd.read_csv(first / "dielectric_properties.csv")
    text_row = exported[exported["data_provenance"] == "measured_text"].iloc[0]
    assert text_row["source_table"] == "Narrative text"
    provenance = json.loads((first / "run_provenance.json").read_text())
    assert provenance["record_count"] == 2
    assert len(provenance["extraction_result_sha256"]) == 2


def test_empty_assembly_refreshes_every_primary_artifact(tmp_path):
    output = tmp_path / "database"
    output.mkdir()
    for name in ("dielectric_properties.csv", "dielectric_properties.parquet"):
        (output / name).write_text("stale", encoding="utf-8")
    stale_db = duckdb.connect(str(output / "cowork.duckdb"))
    stale_db.execute("CREATE TABLE dielectric_properties(value INTEGER)")
    stale_db.execute("INSERT INTO dielectric_properties VALUES (1)")
    stale_db.close()

    db_path = assemble(
        [],
        {},
        output,
        {"validation": {"enabled": False}, "output_dir": str(tmp_path)},
    )

    assert list(pd.read_csv(output / "dielectric_properties.csv").columns)
    assert pd.read_parquet(output / "dielectric_properties.parquet").empty
    connection = duckdb.connect(str(db_path), read_only=True)
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM dielectric_properties"
        ).fetchone()[0] == 0
    finally:
        connection.close()
    assert json.loads((output / "paper_coverage_summary.json").read_text())[
        "assembled_records"
    ] == 0


def test_audit_refreshes_a_stale_report_even_when_there_are_no_flags(tmp_path):
    source = tmp_path / "database.csv"
    report = tmp_path / "audit.csv"
    pd.DataFrame([{
        "title": "paper",
        "authors": "author",
        "material_name": "sample",
        "frequency_mhz": 915.0,
        "temperature_c": 25.0,
        "moisture_content_pct": 20.0,
        "salt_content": None,
        "dielectric_constant": 3.0,
        "loss_factor": 0.2,
        "loss_tangent": 0.067,
    }]).to_csv(source, index=False)
    report.write_text("stale", encoding="utf-8")

    result = run_audit(source, report)

    assert result.empty
    assert "severity" in pd.read_csv(report).columns


def test_audit_missing_input_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Input CSV not found"):
        run_audit(tmp_path / "missing.csv", tmp_path / "report.csv")


def test_duplicate_provenance_selection_is_independent_of_input_order(tmp_path):
    uid = "a" * 12
    first_result = _result(uid, 3.0, source_table="Table 2")
    second_result = _result(uid, 3.0, source_table="Table 1")
    config = {"validation": {"enabled": False}, "output_dir": str(tmp_path)}
    output_a = tmp_path / "a"
    output_b = tmp_path / "b"

    assemble([first_result, second_result], {}, output_a, config)
    assemble([second_result, first_result], {}, output_b, config)

    assert (output_a / "dielectric_properties.csv").read_bytes() == (
        output_b / "dielectric_properties.csv"
    ).read_bytes()
    exported = pd.read_csv(output_a / "dielectric_properties.csv")
    assert exported.loc[0, "source_table"] == "Table 1"


def test_manifest_empty_and_corrupt_are_not_treated_as_load_everything(tmp_path):
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    paper = ParsedPaper(
        doi="10.1/stale",
        metadata=PaperMetadata(doi="10.1/stale"),
    )
    (parsed / "stale.json").write_text(
        json.dumps(paper.model_dump(mode="json")), encoding="utf-8"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    assert load_parsed_papers(parsed) == []

    manifest.write_text("{broken", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Manifest is unreadable"):
        load_parsed_papers(parsed)


def test_recursive_case_insensitive_discovery_and_content_deduplication(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    first = nested / "A.PDF"
    duplicate = tmp_path / "copy.pdf"
    other = tmp_path / "other.Pdf"
    first.write_bytes(b"same")
    duplicate.write_bytes(b"same")
    other.write_bytes(b"different")

    discovered = _discover_corpus_pdfs(tmp_path)
    unique = _deduplicate_pdf_paths(discovered)

    assert len(discovered) == 3
    assert len(unique) == 2


def test_resume_requires_text_artifact_for_prose_only_paper(tmp_path):
    uid = "a" * 12
    payload = json.dumps(_result(uid, 3.0).model_dump(mode="json"))
    (tmp_path / f"{uid}__paper_table.json").write_text(payload, encoding="utf-8")
    assert _already_extracted(uid, tmp_path)
    assert not _already_extracted(uid, tmp_path, require_text=True)
    (tmp_path / f"{uid}__paper_text.json").write_text(payload, encoding="utf-8")
    assert _already_extracted(uid, tmp_path, require_text=True)


def test_corrupt_extraction_artifact_does_not_block_resume(tmp_path):
    uid = "a" * 12
    (tmp_path / f"{uid}__paper_table.json").write_text("{broken", encoding="utf-8")
    assert not _already_extracted(uid, tmp_path)

    incomplete = _result(uid, 3.0)
    incomplete.complete = False
    incomplete.incomplete_reason = "provider response missing"
    (tmp_path / f"{uid}__paper_table.json").write_text(
        json.dumps(incomplete.model_dump(mode="json")), encoding="utf-8"
    )
    assert not _already_extracted(uid, tmp_path)


def test_text_extraction_assigns_narrative_locator(monkeypatch, tmp_path):
    paper = ParsedPaper(
        paper_uid="a" * 12,
        doi="10.1/text",
        pdf_path="paper.pdf",
        full_text="At 915 MHz the sample had dielectric properties.",
        metadata=PaperMetadata(doi="10.1/text"),
    )

    async def fake_call_llm(**kwargs):
        return json.dumps({
            "records": [{
                "material_name": "sample",
                "frequency_mhz": 915,
                "dielectric_constant": 3.0,
                "loss_factor": 0.2,
                "source_table": "",
            }]
        }), CostEntry(stage="", model=kwargs["model"], doi=paper.doi)

    monkeypatch.setattr(text_extractor, "call_llm", fake_call_llm)
    results = asyncio.run(text_extractor.run(
        [(paper, None)],
        {
            "output_dir": str(tmp_path),
            "use_batch_api": False,
            "text_extractor": {"model": "claude-test", "max_tokens": 100},
        },
    ))
    assert results[0].records[0].source_table == "Narrative text"


def test_text_extraction_marks_partial_window_failure_incomplete(monkeypatch, tmp_path):
    paper = ParsedPaper(
        paper_uid="a" * 12,
        doi="10.1/text",
        pdf_path="paper.pdf",
        full_text="results",
        metadata=PaperMetadata(
            doi="10.1/text", measurement_frequencies_mhz=[915.0]
        ),
    )
    monkeypatch.setattr(
        text_extractor, "_window_results_text", lambda text: ["good", "bad"]
    )

    async def fake_call_llm(**kwargs):
        response = (
            json.dumps({"records": [{
                "material_name": "sample", "frequency_mhz": 915,
                "dielectric_constant": 3.0, "loss_factor": 0.2,
            }]})
            if "good" in kwargs["prompt"] else "not JSON"
        )
        return response, CostEntry(stage="", model=kwargs["model"], doi=paper.doi)

    monkeypatch.setattr(text_extractor, "call_llm", fake_call_llm)
    result = asyncio.run(text_extractor.run(
        [(paper, None)],
        {
            "output_dir": str(tmp_path), "use_batch_api": False,
            "text_extractor": {"model": "claude-test", "max_tokens": 100},
        },
    ))[0]
    assert len(result.records) == 1
    assert result.complete is False
    assert "window 2" in result.incomplete_reason


def test_batch_text_uses_all_windows(monkeypatch, tmp_path):
    paper = ParsedPaper(
        paper_uid="a" * 12,
        doi="10.1/text",
        pdf_path="paper.pdf",
        full_text="results",
        metadata=PaperMetadata(doi="10.1/text"),
    )
    monkeypatch.setattr(
        text_extractor, "_window_results_text", lambda text: ["one", "two"]
    )
    captured = {}

    async def fake_batch(requests, config, poll_interval):
        captured["requests"] = requests
        return {
            request["custom_id"]: (
                '{"records": []}',
                CostEntry(stage="", model=request["model"], doi=paper.doi),
            )
            for request in requests
        }

    monkeypatch.setattr(text_extractor, "run_batch", fake_batch)
    result = asyncio.run(text_extractor.run(
        [(paper, None)],
        {
            "output_dir": str(tmp_path), "use_batch_api": True,
            "text_extractor": {"model": "claude-test", "max_tokens": 100},
        },
    ))[0]
    assert len(captured["requests"]) == 2
    assert result.complete is True


def test_screener_cache_replays_metadata_onto_a_fresh_parse(monkeypatch, tmp_path):
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    uid = "a" * 12
    cached = ScreenerResult(
        paper_uid=uid,
        doi="10.1/real",
        estimated_records=12,
        has_equations=True,
        metadata={
            "doi": "10.1/real",
            "primary_materials": ["sample"],
            "measurement_frequencies_mhz": [915.0],
            "moisture_basis": "wet",
            "data_tables": ["table_1"],
        },
    )
    (parsed / "screener_results.json").write_text(
        json.dumps([cached.model_dump(mode="json")]), encoding="utf-8"
    )
    paper = ParsedPaper(
        paper_uid=uid,
        doi="local/paper",
        metadata=PaperMetadata(doi="local/paper", title="paper"),
    )

    async def unexpected_call(**kwargs):
        raise AssertionError("cache replay should not call a model")

    monkeypatch.setattr(screener, "call_llm", unexpected_call)
    results = asyncio.run(screener.run(
        [paper],
        {
            "output_dir": str(tmp_path),
            "use_batch_api": False,
            "screener": {"model": "claude-test", "max_tokens": 100},
        },
    ))

    assert results[0].has_equations
    assert paper.doi == "10.1/real"
    assert paper.metadata.data_tables == ["table_1"]
    assert paper.metadata.moisture_basis == "wet"


def test_screener_cache_cancellation_finishes_atomic_replace(monkeypatch, tmp_path):
    write_started = threading.Event()
    release_write = threading.Event()
    original_write_text = Path.write_text

    def blocking_write_text(path, data, *args, **kwargs):
        written = original_write_text(path, data, *args, **kwargs)
        if path.name == "screener_results.json.tmp":
            write_started.set()
            if not release_write.wait(timeout=5):
                raise TimeoutError("synthetic cache write was never released")
        return written

    monkeypatch.setattr(Path, "write_text", blocking_write_text)
    monkeypatch.setattr(screener, "load_skill", lambda _name: "synthetic prompt")

    async def exercise_cancellation():
        task = asyncio.create_task(
            screener.run([], {"output_dir": str(tmp_path), "use_batch_api": False})
        )
        assert await asyncio.to_thread(write_started.wait, 5)
        try:
            task.cancel()
            await asyncio.sleep(0)
            assert (tmp_path / "parsed" / "screener_results.json.tmp").exists()
            assert not (tmp_path / "parsed" / "screener_results.json").exists()
        finally:
            release_write.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise_cancellation())
    final_cache = tmp_path / "parsed" / "screener_results.json"
    assert json.loads(final_cache.read_text(encoding="utf-8")) == []
    assert not final_cache.with_suffix(".json.tmp").exists()


def test_batch_routes_mixed_providers_and_prices_each_model(monkeypatch):
    calls = []

    async def submit_anthropic(requests, config):
        calls.append(("anthropic", [request["model"] for request in requests]))
        return "anthropic-batch"

    async def poll_anthropic(batch_id, poll_interval, config):
        return {
            utils._sanitize_custom_id("claude/id"): {
                "text": "a", "input_tokens": 10, "output_tokens": 5
            }
        }

    async def submit_openai(requests, config):
        calls.append(("openai", [request["model"] for request in requests]))
        return "openai-batch"

    async def poll_openai(batch_id, poll_interval, config):
        return {
            "gpt_id": {"text": "b", "input_tokens": 20, "output_tokens": 4}
        }

    monkeypatch.setattr(utils, "submit_anthropic_batch", submit_anthropic)
    monkeypatch.setattr(utils, "poll_anthropic_batch", poll_anthropic)
    monkeypatch.setattr(utils, "submit_openai_batch", submit_openai)
    monkeypatch.setattr(utils, "poll_openai_batch", poll_openai)
    requests = [
        {
            "custom_id": "claude/id", "model": "claude-a", "max_tokens": 10,
            "messages": [],
        },
        {
            "custom_id": "gpt_id", "model": "gpt-b", "max_tokens": 10,
            "messages": [],
        },
    ]
    config = {
        "batch_discount": 0.5,
        "pricing": {
            "claude-a": {"input_per_mtok": 2.0, "output_per_mtok": 4.0},
            "gpt-b": {"input_per_mtok": 10.0, "output_per_mtok": 20.0},
        },
    }

    results = asyncio.run(utils.run_batch(requests, config, poll_interval=0))

    assert sorted(calls) == [
        ("anthropic", ["claude-a"]), ("openai", ["gpt-b"])
    ]
    assert results["claude/id"][1].model == "claude-a"
    assert results["gpt_id"][1].model == "gpt-b"
    assert results["claude/id"][1].cost_usd != results["gpt_id"][1].cost_usd


def test_custom_id_sanitization_cannot_collapse_distinct_inputs():
    assert utils._sanitize_custom_id("a/b") != utils._sanitize_custom_id("a b")


def test_openai_batch_preserves_system_prompt_and_sanitized_id(monkeypatch):
    captured = {}

    class Files:
        def create(self, file, purpose):
            captured["jsonl"] = file.read().decode("utf-8")
            return SimpleNamespace(id="file-id")

    class Batches:
        def create(self, **kwargs):
            return SimpleNamespace(id="batch-id", status="validating")

    fake_client = SimpleNamespace(files=Files(), batches=Batches())
    monkeypatch.setattr(utils.openai, "OpenAI", lambda: fake_client)
    request = {
        "custom_id": "paper/id",
        "model": "gpt-4.1-mini",
        "max_tokens": 100,
        "system": "follow the extraction schema",
        "messages": [{"role": "user", "content": "extract"}],
    }

    assert asyncio.run(utils.submit_openai_batch([request])) == "batch-id"
    line = json.loads(captured["jsonl"])
    assert line["custom_id"] == utils._sanitize_custom_id("paper/id")
    assert line["body"]["messages"][0] == {
        "role": "system", "content": "follow the extraction schema"
    }


def test_openai_batch_cancellation_does_not_delete_active_upload(monkeypatch):
    upload_started = threading.Event()
    release_upload = threading.Event()
    observed = {}

    class Files:
        def create(self, file, purpose):
            observed["temp_path"] = Path(file.name)
            upload_started.set()
            if not release_upload.wait(timeout=5):
                raise TimeoutError("synthetic upload was never released")
            observed["exists_at_release"] = Path(file.name).exists()
            return SimpleNamespace(id="synthetic-file")

    class Batches:
        def create(self, **kwargs):
            observed["batch_created"] = True
            return SimpleNamespace(id="synthetic-batch", status="validating")

    fake_client = SimpleNamespace(files=Files(), batches=Batches())
    monkeypatch.setattr(utils.openai, "OpenAI", lambda: fake_client)
    request = {
        "custom_id": "synthetic/request",
        "model": "gpt-4.1-mini",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "synthetic input"}],
    }

    async def exercise_cancellation():
        task = asyncio.create_task(utils.submit_openai_batch([request]))
        assert await asyncio.to_thread(upload_started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert observed["temp_path"].exists()
        release_upload.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise_cancellation())
    assert observed["exists_at_release"] is True
    assert observed["batch_created"] is True
    assert not observed["temp_path"].exists()


def test_unknown_model_provider_fails_closed():
    with pytest.raises(ValueError, match="Cannot infer a provider"):
        utils._detect_provider("vendor-model-without-a-known-prefix")


def test_batch_poll_has_a_finite_timeout(monkeypatch):
    counts = SimpleNamespace(succeeded=0, errored=0, processing=1, canceled=0)
    batch = SimpleNamespace(processing_status="in_progress", request_counts=counts)
    fake_client = SimpleNamespace(
        messages=SimpleNamespace(
            batches=SimpleNamespace(retrieve=lambda batch_id: batch)
        )
    )
    monkeypatch.setattr(utils.anthropic, "Anthropic", lambda: fake_client)

    with pytest.raises(TimeoutError, match="did not finish"):
        asyncio.run(utils.poll_anthropic_batch(
            "batch-id", poll_interval=0, config={"batch_timeout_seconds": 0}
        ))


def test_benchmark_candidate_controls_every_extraction_model():
    assert len(MODELS) == 10
    config = {
        "equation_extractor": {"model": "old-equation"},
        "vision_fallback": {"enabled": True, "model": "old-vision"},
    }
    patched = _patch_config_for_model(config, "gpt-4.1-mini")
    expected = MODELS["gpt-4.1-mini"]["model"]
    assert patched["table_extractor"]["model"] == expected
    assert patched["text_extractor"]["model"] == expected
    assert patched["equation_extractor"]["model"] == expected
    assert patched["vision_fallback"]["enabled"] is False


def test_benchmark_pricing_matches_the_runtime_configuration():
    configured_prices = _load_base_config()["pricing"]
    for candidate in MODELS.values():
        assert configured_prices[candidate["model"]] == candidate["pricing"]


def test_ci_lock_excludes_the_unused_gpu_parser_stack():
    lock_text = (Path(__file__).parents[1] / "requirements-ci-lock.txt").read_text(
        encoding="utf-8"
    ).lower()
    forbidden = ("docling==", "torch==", "nvidia-", "cuda-", "accelerate==")
    assert not any(package in lock_text for package in forbidden)


def test_benchmark_uses_production_prose_and_table_routes():
    prose_paper = SimpleNamespace(metadata=SimpleNamespace(data_tables=[]))
    table_paper = SimpleNamespace(
        metadata=SimpleNamespace(data_tables=["Table 1"])
    )

    assert _requires_text_extraction(prose_paper) is True
    assert _requires_text_extraction(table_paper) is False


def test_benchmark_resume_requires_the_route_specific_artifact(tmp_path):
    prose_paper = SimpleNamespace(
        paper_uid="prose-uid",
        metadata=SimpleNamespace(data_tables=[]),
    )
    table_paper = SimpleNamespace(
        paper_uid="table-uid",
        metadata=SimpleNamespace(data_tables=["Table 1"]),
    )

    prose_result = _result("prose-uid", 3.0)
    table_result = _result("table-uid", 4.0)
    (tmp_path / "prose-uid__paper_table.json").write_text(
        json.dumps(prose_result.model_dump(mode="json")), encoding="utf-8"
    )
    (tmp_path / "table-uid__paper_table.json").write_text(
        json.dumps(table_result.model_dump(mode="json")), encoding="utf-8"
    )

    # A zero-row or complete table pass cannot mask a missing text pass for a
    # prose-only paper, while table-classified papers require the table result.
    assert _benchmark_artifact_complete(prose_paper, tmp_path) is False
    assert _benchmark_artifact_complete(table_paper, tmp_path) is True

    (tmp_path / "prose-uid__paper_text.json").write_text(
        json.dumps(prose_result.model_dump(mode="json")), encoding="utf-8"
    )
    assert _benchmark_artifact_complete(prose_paper, tmp_path) is True


def test_benchmark_calls_text_for_prose_only_and_table_for_every_paper(
    monkeypatch, tmp_path
):
    import src.assembler as assembler_module
    import src.pipeline as pipeline_module
    import src.table_extractor as table_extractor_module

    prose = ParsedPaper(
        paper_uid="prose-uid",
        doi="10.1/prose",
        pdf_path="prose.pdf",
        metadata=PaperMetadata(doi="10.1/prose", data_tables=[]),
    )
    tabular = ParsedPaper(
        paper_uid="table-uid",
        doi="10.1/table",
        pdf_path="table.pdf",
        metadata=PaperMetadata(doi="10.1/table", data_tables=["Table 1"]),
    )
    papers = [prose, tabular]
    screener_result = SimpleNamespace()
    captured = {}

    monkeypatch.setattr(
        pipeline_module, "load_parsed_papers", lambda _path: papers
    )
    monkeypatch.setattr(
        pipeline_module, "load_screener_results", lambda _path: []
    )
    monkeypatch.setattr(
        pipeline_module,
        "pair_papers_with_screener",
        lambda loaded, _screened: [(paper, screener_result) for paper in loaded],
    )

    async def fake_text_run(pairs, *_args, **_kwargs):
        captured["text"] = [paper.paper_uid for paper, _sr in pairs]
        return [_result("prose-uid", 3.0)]

    async def fake_table_run(pairs, *_args, **_kwargs):
        captured["table"] = [paper.paper_uid for paper, _sr in pairs]
        return [_result(paper.paper_uid, 4.0) for paper, _sr in pairs]

    monkeypatch.setattr(text_extractor, "run", fake_text_run)
    monkeypatch.setattr(table_extractor_module, "run", fake_table_run)
    monkeypatch.setattr(
        pipeline_module,
        "load_extraction_results",
        lambda *_args, **_kwargs: [
            _result("prose-uid", 3.0),
            _result("table-uid", 4.0),
        ],
    )
    monkeypatch.setattr(assembler_module, "run", lambda **_kwargs: None)

    class FakeConnection:
        def execute(self, _query):
            return self

        def fetchall(self):
            return []

    checkpoint = SimpleNamespace(
        total_cost=lambda: 0.0,
        _conn=FakeConnection(),
    )
    config = {"output_dir": str(tmp_path / "base")}

    asyncio.run(benchmark_module._run_extract_and_assemble(
        config,
        "haiku",
        tmp_path / "run",
        checkpoint,
    ))

    assert captured["text"] == ["prose-uid"]
    assert captured["table"] == ["prose-uid", "table-uid"]
