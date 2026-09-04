"""Synthetic regression tests for the corpus-agnostic assembler."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from src.assembler import (
    _apply_configured_overrides,
    _apply_extension_hooks,
    _flatten_records,
    _merge_split_records,
    _normalize_material_names,
)
from src.assembler import (
    run as assemble,
)
from src.schema import DielectricRecord, ExtractionResult, PaperMetadata, ParsedPaper


def _base_df(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "paper_id": "synthetic-source",
        "paper_uid": "a" * 12,
        "material_name": "sample powder",
        "frequency_mhz": None,
        "temperature_c": None,
        "dielectric_constant": None,
        "loss_factor": None,
        "loss_tangent": None,
        "moisture_content_pct": None,
        "moisture_basis": None,
        "salt_content": None,
        "electrical_conductivity_s_m": None,
        "source_table": "Table A",
        "source_location": None,
        "data_provenance": "measured_table",
        "model_expression": None,
        "model_r_squared": None,
        "extraction_source": "table",
        "extraction_model": "test-model",
        "doi": "synthetic:source-a",
        "title": "Synthetic source",
        "authors": "",
        "year": 2020,
        "journal": "",
        "measurement_method": "",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_material_normalization_is_casefold_unique_and_order_invariant():
    rows = [
        {"material_name": " Sample   Powder "},
        {"material_name": "sample powder"},
        {"material_name": "Reference Gel"},
        {"material_name": "reference gel"},
        {"material_name": "reference gel"},
    ]
    forward = _normalize_material_names(_base_df(rows))
    reverse = _normalize_material_names(_base_df(list(reversed(rows))))
    assert set(forward["material_name"]) == {"Sample Powder", "reference gel"}
    assert set(reverse["material_name"]) == {"Sample Powder", "reference gel"}


def test_material_normalization_supports_local_aliases():
    frame = _base_df([{"material_name": "powder-a"}])
    normalized = _normalize_material_names(
        frame, aliases={"powder-a": "Sample powder"}
    )
    assert normalized.loc[0, "material_name"] == "Sample powder"


def test_flatten_records_inherits_source_moisture_basis():
    paper = ParsedPaper(
        paper_uid="paper-uid",
        doi="synthetic:source-a",
        metadata=PaperMetadata(doi="synthetic:source-a", moisture_basis="wet"),
    )
    result = ExtractionResult(
        paper_uid="paper-uid",
        doi="synthetic:source-a",
        records=[
            DielectricRecord(
                paper_uid="paper-uid",
                paper_id="synthetic-source",
                material_name="sample powder",
                moisture_pct=12.0,
                moisture_basis="unknown",
            ),
            DielectricRecord(
                paper_uid="paper-uid",
                paper_id="synthetic-source",
                material_name="sample powder",
            ),
        ],
    )
    rows = _flatten_records([result], {"synthetic-source": paper})
    assert rows[0]["moisture_basis"] == "wet"
    assert rows[1]["moisture_basis"] is None


def test_merge_split_records_keeps_unknown_condition_rows():
    frame = _base_df([
        {
            "frequency_mhz": 900.0,
            "temperature_c": 20.0,
            "dielectric_constant": 50.0,
            "loss_factor": 10.0,
        },
        {
            "frequency_mhz": None,
            "temperature_c": 20.0,
            "dielectric_constant": 60.0,
        },
        {
            "frequency_mhz": None,
            "temperature_c": 20.0,
            "loss_factor": 12.0,
        },
    ])
    merged = _merge_split_records(frame)
    assert len(merged) == 2
    unknown = merged[merged["frequency_mhz"].isna()].iloc[0]
    assert unknown["dielectric_constant"] == 60.0
    assert unknown["loss_factor"] == 12.0


def test_merge_split_records_preserves_both_equation_sources():
    frame = _base_df([
        {
            "frequency_mhz": 900.0,
            "temperature_c": 40.0,
            "dielectric_constant": 50.0,
            "source_table": "Table A",
            "model_expression": "dielectric_constant: 60 - 0.25*T",
            "model_r_squared": 0.99,
        },
        {
            "frequency_mhz": 900.0,
            "temperature_c": 40.0,
            "loss_factor": 8.0,
            "source_table": "Table B",
            "model_expression": "loss_factor: 10 - 0.05*T",
            "model_r_squared": 0.98,
        },
    ])
    merged = _merge_split_records(frame)
    assert len(merged) == 1
    assert merged.loc[0, "source_table"] == "Table A; Table B"
    assert merged.loc[0, "model_expression"] == (
        "dielectric_constant: 60 - 0.25*T; loss_factor: 10 - 0.05*T"
    )


def test_local_override_rules_are_opt_in_and_declarative(tmp_path):
    rules = tmp_path / "assembly.local.yaml"
    rules.write_text(
        """
doi_overrides:
  synthetic:old: synthetic:new
material_aliases:
  sample powder: Sample Powder
row_rules:
  - where:
      paper_uid: aaaaaaaaaaaa
      temperature_c: 40.0
    set:
      source_location: synthetic fixture
  - where:
      paper_uid: bbbbbbbbbbbb
    drop: true
""".strip(),
        encoding="utf-8",
    )
    frame = _base_df([
        {"paper_uid": "a" * 12, "doi": "synthetic:old", "temperature_c": 40.0},
        {"paper_uid": "b" * 12, "doi": "synthetic:old", "temperature_c": 50.0},
    ])
    result, aliases = _apply_configured_overrides(
        frame,
        {"assembly": {"overrides_file": str(rules)}},
    )
    assert len(result) == 1
    assert result.loc[0, "doi"] == "synthetic:new"
    assert result.loc[0, "source_location"] == "synthetic fixture"
    assert aliases == {"sample powder": "Sample Powder"}


def test_configured_override_file_must_exist(tmp_path):
    frame = _base_df([{}])

    with pytest.raises(FileNotFoundError, match="override file does not exist"):
        _apply_configured_overrides(
            frame,
            {"assembly": {"overrides_file": str(tmp_path / "missing.local.yaml")}},
        )


def test_override_rule_rejects_unknown_filter_columns(tmp_path):
    rules = tmp_path / "assembly.local.yaml"
    rules.write_text(
        """
row_rules:
  - where:
      paper_udi: aaaaaaaaaaaa
    set:
      source_location: corrected
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown filter columns.*paper_udi"):
        _apply_configured_overrides(
            _base_df([{}]),
            {"assembly": {"overrides_file": str(rules)}},
        )


def test_computed_loss_tangent_is_validated_before_export(tmp_path):
    result = ExtractionResult(
        doi="synthetic:source-a",
        records=[DielectricRecord(
            paper_id="synthetic-source",
            material_name="sample powder",
            frequency_mhz=900.0,
            temperature_c=20.0,
            dielectric_constant=1.0,
            loss_factor=200.0,
        )],
    )

    assemble(
        [result],
        {},
        tmp_path,
        {"validation": {"enabled": True}},
    )

    assert pd.read_csv(tmp_path / "dielectric_properties.csv").empty


def test_local_extension_hook_receives_copy(monkeypatch):
    def mark(frame, *, parsed_papers, config):
        assert parsed_papers == {}
        frame["source_location"] = "extension output"
        return frame

    monkeypatch.setitem(sys.modules, "synthetic_extension", SimpleNamespace(mark=mark))
    source = _base_df([{}])
    result = _apply_extension_hooks(
        source,
        {},
        {"assembly": {"postprocessors": ["synthetic_extension:mark"]}},
    )
    assert result.loc[0, "source_location"] == "extension output"
    assert pd.isna(source.loc[0, "source_location"])


def test_override_content_is_hashed_without_recording_local_path(tmp_path):
    rules = tmp_path / "private-name.local.yaml"
    raw = b"material_aliases:\n  sample powder: Synthetic powder\n"
    rules.write_bytes(raw)
    result = ExtractionResult(
        doi="synthetic:source-a",
        records=[DielectricRecord(
            paper_id="synthetic-source",
            material_name="sample powder",
            frequency_mhz=900.0,
            temperature_c=20.0,
            dielectric_constant=10.0,
            loss_factor=2.0,
        )],
    )

    assemble(
        [result],
        {},
        tmp_path / "output",
        {
            "validation": {"enabled": False},
            "assembly": {"overrides_file": str(rules)},
        },
    )

    provenance_path = tmp_path / "output" / "run_provenance.json"
    provenance_text = provenance_path.read_text(encoding="utf-8")
    provenance = json.loads(provenance_text)
    assert provenance["assembly_extensions"]["override"] == {
        "selected_via": "configuration",
        "content_sha256": hashlib.sha256(raw).hexdigest(),
    }
    assert str(tmp_path) not in provenance_text
    assert provenance["runtime_config"]["assembly"]["overrides_file"] == (
        "<local-assembly-override>"
    )


def test_environment_override_is_recorded_in_provenance(tmp_path, monkeypatch):
    rules = tmp_path / "assembly.local.yaml"
    raw = b"material_aliases: {}\n"
    rules.write_bytes(raw)
    monkeypatch.setenv("AGENTXTRACT_ASSEMBLY_OVERRIDES", str(rules))

    assemble(
        [],
        {},
        tmp_path / "output",
        {"validation": {"enabled": False}},
    )

    provenance = json.loads(
        (tmp_path / "output" / "run_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["assembly_extensions"]["override"] == {
        "selected_via": "environment",
        "content_sha256": hashlib.sha256(raw).hexdigest(),
    }


def test_extension_hook_source_is_hashed_without_recording_source_path(
    tmp_path, monkeypatch
):
    module_path = tmp_path / "synthetic_provenance_hook.py"
    module_path.write_text(
        """
def mark(frame, *, parsed_papers, config):
    frame["source_location"] = "synthetic hook"
    return frame
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    result = ExtractionResult(
        doi="synthetic:source-a",
        records=[DielectricRecord(
            paper_id="synthetic-source",
            material_name="sample powder",
        )],
    )

    assemble(
        [result],
        {},
        tmp_path / "output",
        {
            "validation": {"enabled": False},
            "assembly": {
                "postprocessors": ["synthetic_provenance_hook:mark"],
            },
        },
    )

    provenance_path = tmp_path / "output" / "run_provenance.json"
    provenance_text = provenance_path.read_text(encoding="utf-8")
    provenance = json.loads(provenance_text)
    hook = provenance["assembly_extensions"]["postprocessors"][0]
    assert hook["identifier"] == "synthetic_provenance_hook:mark"
    assert hook["source_sha256"] == [
        hashlib.sha256(module_path.read_bytes()).hexdigest()
    ]
    assert str(tmp_path) not in provenance_text
