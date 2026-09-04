"""Regression tests for infrastructure fixes in utils.py, audit.py (batch ids,
Gemini rate-limit detection, bare-array JSON stripping, audit flag parsing,
checkpoint upsert)."""

import asyncio
import json
import re

import pandas as pd
import pytest

import src.utils
from src.pipeline import _clear_forced_extractions, _resolve_requested_pdfs
from src.schema import PaperCheckpoint, PipelineStatus
from src.utils import (
    CheckpointDB,
    _is_gemini_rate_limit,
    _sanitize_custom_id,
    load_config,
    load_skill,
    parse_json_safe,
    strip_code_fences,
)

# ---------------------------------------------------------------------------
# Fix 2: _sanitize_custom_id — long ids must stay unique (hash suffix)
# ---------------------------------------------------------------------------

class TestSanitizeCustomId:
    def test_short_id_unchanged(self):
        assert _sanitize_custom_id("screen_abc_1") == "screen_abc_1"

    def test_special_chars_replaced(self):
        out = _sanitize_custom_id("table_synthetic/source:a_t1")
        assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", out)

    def test_long_ids_do_not_collide(self):
        base = "table_synthetic/source.identifier.with.an.extremely.long.suffix"
        a = _sanitize_custom_id(base + ".chunk1")
        b = _sanitize_custom_id(base + ".chunk2")
        assert a != b
        assert len(a) <= 64 and len(b) <= 64
        assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", a)
        assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", b)

    def test_long_id_deterministic(self):
        cid = "x" * 100
        assert _sanitize_custom_id(cid) == _sanitize_custom_id(cid)
        assert len(_sanitize_custom_id(cid)) == 64


# ---------------------------------------------------------------------------
# Fix 6: Gemini rate-limit detection must not match "generate"
# ---------------------------------------------------------------------------

class TestGeminiRateLimitDetection:
    def test_generate_is_not_rate_limit(self):
        assert not _is_gemini_rate_limit("failed to generate content: internal error")

    def test_500_is_not_rate_limit(self):
        assert not _is_gemini_rate_limit("500 internal server error")

    @pytest.mark.parametrize("msg", [
        "429 Too Many Requests",
        "Error: RESOURCE_EXHAUSTED",
        "Rate limit exceeded, retry later",
        "rate-limit hit",
        "ratelimit reached for project",
        "Quota exceeded for model",
    ])
    def test_rate_limit_variants_detected(self, msg):
        assert _is_gemini_rate_limit(msg)


# ---------------------------------------------------------------------------
# Fix 9: strip_code_fences must preserve bare-array JSON responses
# ---------------------------------------------------------------------------

class TestStripCodeFencesBareArray:
    def test_bare_array_with_prose(self):
        text = 'Here are the records:\n\n[{"material": "apple"}, {"material": "pear"}]\n\nDone.'
        out = strip_code_fences(text)
        assert json.loads(out) == [{"material": "apple"}, {"material": "pear"}]

    def test_bare_array_in_fences(self):
        text = '```json\n[{"a": 1}, {"b": 2}]\n```'
        assert json.loads(strip_code_fences(text)) == [{"a": 1}, {"b": 2}]

    def test_array_of_scalars_no_braces(self):
        text = "The frequencies are: [915, 2450]"
        assert json.loads(strip_code_fences(text)) == [915, 2450]

    def test_object_still_extracted(self):
        text = 'Sure, here it is:\n{"records": [], "notes": "n"}\nThanks!'
        assert json.loads(strip_code_fences(text)) == {"records": [], "notes": "n"}


def test_truncated_equation_response_salvages_complete_models():
    text = (
        '{"equations":['
        '{"material_name":"fish","property":"dielectric_constant",'
        '"coefficients":[77,-0.03],"variable":"frequency_mhz"},'
        '{"material_name":"fish","property":"loss_factor",'
        '"coefficients":[33,-0.01],"variable":"frequency_mhz"}'
        '],"notes":"long explanation that was truncated'
    )
    parsed = parse_json_safe(text)
    assert len(parsed["equations"]) == 2
    assert parsed["equations"][1]["property"] == "loss_factor"


# ---------------------------------------------------------------------------
# Fix 15: audit flag parsing — "Flag 1" must not match "Flag 10"
# ---------------------------------------------------------------------------

def test_audit_flag_parsing_no_prefix_collision(monkeypatch):
    from src.audit import _llm_review

    n_flags = 10
    flags_df = pd.DataFrame([
        {
            "title": "paper-a",
            "material_name": f"mat{i}",
            "frequency_mhz": 915.0,
            "temperature_c": 20.0,
            "severity": "CRITICAL",
            "rule_id": f"RULE_{i}",
            "description": f"issue {i}",
        }
        for i in range(n_flags)
    ])
    source_df = pd.DataFrame([{
        "title": "paper-a",
        "material_name": "mat0",
        "frequency_mhz": 915.0,
        "temperature_c": 20.0,
        "moisture_content_pct": 50.0,
        "salt_content": None,
        "dielectric_constant": 60.0,
        "loss_factor": 15.0,
    }])

    # Flag 10 listed FIRST — the old substring match assigned its verdict to Flag 1
    lines = ["Flag 10: LIKELY_ERROR — value out of range", "Flag 1: FALSE_ALARM — value is fine"]
    lines += [f"Flag {i}: POSSIBLY_OK — plausible" for i in range(2, 10)]
    response = "\n".join(lines)

    async def fake_call_llm(prompt, **kwargs):
        assert kwargs["model"] == "gpt-4.1-mini"
        assert kwargs["config"]["validator"]["provider"] == "openai"
        return response, None

    monkeypatch.setattr(src.utils, "call_llm", fake_call_llm)

    result = asyncio.run(_llm_review(flags_df, source_df))
    assert result.loc[0, "llm_verdict"] == "FALSE_ALARM"
    assert result.loc[9, "llm_verdict"] == "LIKELY_ERROR"
    assert (result.loc[1:8, "llm_verdict"] == "POSSIBLY_OK").all()


# ---------------------------------------------------------------------------
# Fix 8: checkpoint upsert must not wipe pdf_path with '' or errors with NULL
# ---------------------------------------------------------------------------

class TestCheckpointUpsertRoundTrip:
    @pytest.fixture
    def db(self, tmp_path):
        db = CheckpointDB(tmp_path / "cp.db")
        yield db
        db.close()

    def test_pdf_path_survives_empty_upsert(self, db):
        db.upsert(PaperCheckpoint(
            doi="10.1234/x",
            status=PipelineStatus.PARSED,
            pdf_path="/data/corpus/x.pdf",
            error_message="parse warning",
        ))
        # Second upsert (e.g. from screen/extract stage) with no pdf_path/error
        db.upsert(PaperCheckpoint(
            doi="10.1234/x",
            status=PipelineStatus.EXTRACTED,
            pdf_path="",
            error_message=None,
        ))
        cp = db.get("10.1234/x")
        assert cp.status == PipelineStatus.EXTRACTED
        assert cp.pdf_path == "/data/corpus/x.pdf"
        assert cp.error_message == "parse warning"

    def test_new_pdf_path_overrides(self, db):
        db.upsert(PaperCheckpoint(doi="10.1234/y", pdf_path="/old.pdf"))
        db.upsert(PaperCheckpoint(doi="10.1234/y", pdf_path="/new.pdf"))
        assert db.get("10.1234/y").pdf_path == "/new.pdf"


def test_repo_config_and_skills_load_outside_project_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config("configs/test_mode.yaml")
    assert config["input_dir"] == "data/corpus"
    assert "dielectric" in load_skill("extractor_table").lower()


def test_forced_single_file_cleanup_is_scoped(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"selected paper")
    other_pdf = tmp_path / "other.pdf"
    other_pdf.write_bytes(b"other paper")
    output = tmp_path / "data"
    parsed = output / "parsed"
    extracted = output / "extracted"
    parsed.mkdir(parents=True)
    extracted.mkdir()

    from src.paper_id import compute_uid

    selected_uid = compute_uid(pdf)
    other_uid = compute_uid(other_pdf)
    selected_artifact = extracted / f"{selected_uid}__paper_table.json"
    other_artifact = extracted / f"{other_uid}__other_table.json"
    selected_artifact.write_text("{}", encoding="utf-8")
    other_artifact.write_text("{}", encoding="utf-8")

    removed = _clear_forced_extractions([pdf], {"output_dir": str(output)})

    assert removed == [selected_artifact]
    assert not selected_artifact.exists()
    assert other_artifact.exists()


def test_repeatable_file_selection_resolves_a_batch(tmp_path):
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    assert _resolve_requested_pdfs([str(first), str(second)]) == [
        first.resolve(), second.resolve()
    ]
