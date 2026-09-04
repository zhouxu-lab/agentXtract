"""Material names must match on words, not on string order."""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

_spec = importlib.util.spec_from_file_location(
    "evaluate_mod", Path(__file__).resolve().parents[1] / "evaluate.py"
)
evaluate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate)


@pytest.mark.parametrize("pred,gold", [
    ("treated sample", "sample (treated)"),
    ("granule kernels", "granule kernel"),
    ("sample powder (1.3% salt)", "sample powder"),
    ("tomatoes", "tomato"),
    ("dried berry slices", "berry"),
    ("Berries", "berry"),
    ("reference gel", "Reference Gel"),
])
def test_same_material_matches(pred, gold):
    assert evaluate._material_match(pred, gold)


@pytest.mark.parametrize("pred,gold", [
    ("treated matrix", "untreated matrix"),
    ("alpha gel", "beta gel"),
    ("powder alpha", "powder beta"),
])
def test_different_materials_stay_apart(pred, gold):
    assert not evaluate._material_match(pred, gold)


def test_missing_name_is_not_disagreement():
    # Several gold sheets leave the material column blank; a blank cannot
    # contradict the pipeline's name, so the pair is decided by the other gates.
    assert evaluate._material_match("sample powder", "")
    assert evaluate._material_match("", "reference gel")


def test_match_is_symmetric():
    for a, b in [("treated sample", "sample (treated)"), ("alpha gel", "beta gel")]:
        assert evaluate._material_match(a, b) == evaluate._material_match(b, a)


def test_numbers_and_units_do_not_carry_identity():
    # "36%" and "1.3" describe a condition, not a material, so they neither
    # create nor block a match.
    assert evaluate._material_tokens("sample powder 1.3% 36%") == {"powder"}
