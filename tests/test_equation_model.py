"""Synthetic tests for safe multivariate equation evaluation."""

import math

from src.equation_model import (
    EvalReport,
    build_grid,
    build_model,
    canonical_var,
    parse_number,
    parse_subscript_key,
    plausible,
)


def test_parse_number_handles_common_pdf_damage():
    cases = {
        "- 31.27": -31.27,
        "6.42 × 10 - 3 *": 0.00642,
        "3.17e-4": 0.000317,
        "5.83 * 1": 5.83,
        "−0.074": -0.074,
        "3.2 × 10²": 320.0,
    }
    for raw, expected in cases.items():
        assert math.isclose(parse_number(raw), expected, rel_tol=1e-9)


def test_canonical_var_accepts_supported_names():
    assert canonical_var("moisture_content_pct") == "M"
    assert canonical_var("temperature_c") == "T"
    assert canonical_var("bulk density") == "RHO"
    assert canonical_var("frequency_mhz") == "F"
    assert canonical_var("NaCl") == "S"


def test_canonical_var_does_not_prefix_match_unrelated_names():
    assert canonical_var("time") is None
    assert canonical_var("mass") is None
    assert canonical_var("fat_content") is None


def test_model_rejects_an_unrecognized_declared_variable():
    assert build_model({
        "material_name": "synthetic powder",
        "property": "dielectric_constant",
        "expression": "2 + unknown_axis",
        "variables": ["unknown_axis"],
    }) is None


def test_parse_subscript_key():
    order = ["M", "T"]
    assert parse_subscript_key("alpha 0", order) == {}
    assert parse_subscript_key("a1", order) == {"M": 1}
    assert parse_subscript_key("alpha_12", order) == {"M": 1, "T": 1}
    assert parse_subscript_key("alpha 112", order) == {"M": 2, "T": 1}
    assert parse_subscript_key("a222", order) == {"T": 3}


SYNTHETIC_RESPONSE = {
    "material_name": "synthetic powder",
    "property": "dielectric_constant",
    "variables": {"1": "moisture_content_pct", "2": "temperature_c"},
    "subscripts": {
        "alpha 0": "4.0",
        "alpha 1": "3.0 × 10 - 1",
        "alpha 2": "- 1.0 × 10 - 2",
        "alpha 12": "2.0 × 10 - 3",
    },
    "domain": {
        "moisture_content_pct": [10, 30],
        "temperature_c": [20, 80],
    },
    "frequency_mhz": 900.0,
    "source_table": "Synthetic Table A",
}


def test_subscript_model_builds_and_evaluates():
    model = build_model(SYNTHETIC_RESPONSE)
    assert model is not None
    assert set(model.variables) == {"M", "T"}
    observed = model.evaluate({"M": 20.0, "T": 40.0})
    expected = 4.0 + 0.3 * 20.0 - 0.01 * 40.0 + 0.002 * 20.0 * 40.0
    assert math.isclose(observed, expected, rel_tol=1e-9)
    assert plausible("dielectric_constant", observed, 900.0)


def test_grid_respects_model_domain():
    model = build_model(SYNTHETIC_RESPONSE)
    grid = build_grid(model)
    assert grid
    assert all(10.0 <= point["M"] <= 30.0 for point in grid)
    assert all(20.0 <= point["T"] <= 80.0 for point in grid)


def test_grid_prefers_reported_levels():
    model = build_model(SYNTHETIC_RESPONSE)
    grid = build_grid(model, paper_levels={"M": [10, 17.5, 30]})
    assert sorted({point["M"] for point in grid}) == [10, 17.5, 30]


def test_grid_converts_percent_levels_for_fraction_domain():
    model = build_model({
        "material_name": "synthetic gel",
        "property": "dielectric_constant",
        "expression": "2 + 40 * M",
        "variables": ["moisture_content_pct"],
        "domain": {"moisture_content_pct": [0.10, 0.30]},
        "frequency_mhz": 2400.0,
    })
    grid = build_grid(model, paper_levels={"M": [10, 20, 30]})
    assert [point["M"] for point in grid] == [0.1, 0.2, 0.3]


def test_grid_intersects_model_and_reported_domains():
    model = build_model({
        "material_name": "synthetic gel",
        "property": "dielectric_constant",
        "expression": "2 + M",
        "variables": ["moisture_content_pct"],
        "domain": {"moisture_content_pct": [10, 20]},
        "frequency_mhz": 900.0,
    })
    grid = build_grid(
        model,
        paper_ranges={"M": (5, 15)},
        paper_levels={"M": [5, 10, 15, 20]},
    )
    assert [point["M"] for point in grid] == [10, 15]


def test_grid_is_empty_when_domains_do_not_overlap():
    model = build_model({
        "material_name": "synthetic gel",
        "property": "dielectric_constant",
        "expression": "2 + M",
        "variables": ["moisture_content_pct"],
        "domain": {"moisture_content_pct": [10, 20]},
        "frequency_mhz": 900.0,
    })
    assert build_grid(
        model,
        paper_ranges={"M": (0, 5)},
        paper_levels={"M": [0, 5]},
    ) == []


def test_grid_empty_when_variable_is_unconstrained():
    model = build_model({
        "material_name": "synthetic granules",
        "property": "dielectric_constant",
        "expression": "1 + 0.5 * RHO",
        "variables": ["bulk_density"],
        "frequency_mhz": 2400.0,
    })
    assert build_grid(model) == []


def test_legacy_univariate_coefficient_list():
    model = build_model({
        "material_name": "synthetic liquid",
        "property": "dielectric_constant",
        "coefficients": [20.0, -0.1, -0.001],
        "variable": "temperature_c",
        "frequency_mhz": 900.0,
    })
    assert model is not None
    assert math.isclose(model.evaluate({"T": 20.0}), 17.6, rel_tol=1e-9)


def test_expression_model():
    model = build_model({
        "material_name": "synthetic gel",
        "property": "dielectric_constant",
        "expression": "eps = 12 * exp(-0.01 * T) + 0.25 * M",
        "variables": ["temperature_c", "moisture_content_pct"],
        "domain": {"temperature_c": [20, 80], "moisture_content_pct": [5, 60]},
        "frequency_mhz": 2400.0,
    })
    assert model is not None
    assert math.isclose(
        model.evaluate({"T": 20.0, "M": 40.0}),
        12 * math.exp(-0.2) + 10,
        rel_tol=1e-9,
    )


def test_expression_normalizes_alias_and_caret_power():
    model = build_model({
        "material_name": "synthetic liquid",
        "property": "dielectric_constant",
        "expression": "30 - 0.02*f + 0.00001*f^2",
        "variables": ["frequency_mhz"],
        "domain": {"frequency_mhz": [1, 2500]},
    })
    assert math.isclose(model.evaluate({"F": 100}), 28.1, abs_tol=1e-6)


def test_expression_rejects_unsafe_input():
    model = build_model({
        "material_name": "synthetic material",
        "property": "loss_factor",
        "expression": "__import__('os').system('echo unsafe')",
        "variables": ["temperature_c"],
    })
    assert model is None or model.evaluate({"T": 25.0}) is None


def test_eval_report_summary():
    report = EvalReport(
        models_built=4,
        points_evaluated=160,
        points_kept=152,
        points_implausible=8,
    )
    assert "152/160" in report.summary()
