"""The evaluator accepts user-owned tidy reference workbooks."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("evaluate_reference_mod", ROOT / "evaluate.py")
evaluate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate)


def test_reference_loader_concatenates_tidy_excel_sheets(tmp_path):
    workbook = tmp_path / "synthetic_reference.xlsx"
    first = pd.DataFrame([{
        "paper_id": "source-a",
        "material_name": "sample powder",
        "frequency_mhz": 900.0,
        "temperature_c": 20.0,
        "dielectric_constant": 8.0,
        "loss_factor": 1.0,
    }])
    second = pd.DataFrame([{
        "paper_id": "source-b",
        "material_name": "reference gel",
        "frequency_mhz": 2400.0,
        "temperature_c": 30.0,
        "dielectric_constant": 16.0,
        "loss_factor": 2.0,
    }])
    with pd.ExcelWriter(workbook) as writer:
        first.to_excel(writer, sheet_name="set_a", index=False)
        second.to_excel(writer, sheet_name="set_b", index=False)

    loaded = evaluate.load_gold(workbook)

    assert loaded["paper_id"].tolist() == ["source-a", "source-b"]
    assert loaded["reference_sheet"].tolist() == ["set_a", "set_b"]


def test_reference_loader_ignores_non_record_excel_sheets(tmp_path):
    workbook = tmp_path / "synthetic_reference.xlsx"
    records = pd.DataFrame([{
        "paper_id": "source-a",
        "frequency_mhz": 900.0,
        "temperature_c": 20.0,
        "dielectric_constant": 8.0,
        "loss_factor": 1.0,
    }])
    notes = pd.DataFrame({"notes": ["Synthetic workbook instructions"]})
    with pd.ExcelWriter(workbook) as writer:
        records.to_excel(writer, sheet_name="records", index=False)
        notes.to_excel(writer, sheet_name="notes", index=False)

    loaded = evaluate.load_gold(workbook)

    assert len(loaded) == 1
    assert loaded.loc[0, "paper_id"] == "source-a"
    assert loaded.loc[0, "reference_sheet"] == "records"


def test_reference_loader_rejects_non_tidy_workbook(tmp_path):
    workbook = tmp_path / "invalid_reference.xlsx"
    pd.DataFrame({"summary": ["not record-level data"]}).to_excel(
        workbook, index=False
    )
    with pytest.raises(ValueError, match="missing columns"):
        evaluate.load_gold(workbook)
