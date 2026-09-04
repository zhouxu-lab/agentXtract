"""Multivariate regression-model evaluation for dielectric property equations.

Background
----------
Many dielectric-property papers do not tabulate individual measurements. They
report empirical models fitted to the measured surface, most often

    eps' or eps'' = f(moisture content, temperature)      at a fixed frequency

The original pipeline could only evaluate a single-variable polynomial
(sum c_i * x**i). When it was handed the coefficient vector of a bivariate
model it produced meaningless numbers, the physical-range check discarded
them, and the paper silently yielded zero records.

This module supports common model forms:

1. ``terms``       - explicit multivariate polynomial terms with per-variable
                     exponents. The general case.
2. ``subscripts``  - response-surface notation (alpha_0, alpha_1, alpha_12,
                     alpha_112 ...) where the subscript digits index the
                     variables multiplied together. Common in the
                     response-surface reports.
3. ``coefficients``- legacy univariate list [a0, a1, a2, ...]. Kept for
                     backward compatibility.
4. ``expression``  - a restricted arithmetic expression in the declared
                     variables, for non-polynomial forms (Arrhenius,
                     power law, empirical density models).

Evaluation happens on a grid built from the model's declared domain
intersected with the paper's reported ranges, so generated points stay inside
the range over which the model was fitted. Points outside the domain, and
points that fail the physical-range check, are counted and reported rather
than dropped silently.
"""

from __future__ import annotations

import ast
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical variable names
# ---------------------------------------------------------------------------

# Everything the LLM might call a variable, mapped onto a canonical symbol.
_VAR_ALIASES: dict[str, str] = {
    # temperature
    "t": "T", "temp": "T", "temperature": "T", "temperature_c": "T",
    "temperature_deg_c": "T", "temp_c": "T", "theta": "T",
    # moisture
    "m": "M", "mc": "M", "moisture": "M", "moisture_content": "M",
    "moisture_pct": "M", "moisture_content_pct": "M", "w": "M", "x": "M",
    "moisture_wb": "M", "moisture_db": "M",
    # bulk density
    "rho": "RHO", "density": "RHO", "bulk_density": "RHO", "d": "RHO",
    # frequency, when a model spans frequencies
    "f": "F", "freq": "F", "frequency": "F", "frequency_mhz": "F",
    # salt / ionic content
    "s": "S", "salt": "S", "salt_content": "S", "nacl": "S",
}

# Unit suffixes that may be appended to an otherwise exact alias.  This is
# intentionally a small allow-list: an earlier ``startswith`` fallback made
# unrelated names such as ``time``, ``mass`` and ``fat_content`` look like
# temperature, moisture and frequency variables merely because they began
# with T, M or F.
_VAR_UNIT_SUFFIXES: dict[str, tuple[str, ...]] = {
    "T": ("c", "deg_c", "degc", "celsius", "k", "kelvin"),
    "M": (
        "pct", "percent", "percentage", "fraction", "wb", "wet_basis",
        "db", "dry_basis",
    ),
    "RHO": ("kg_m3", "kg_m_3", "kgm3", "g_cm3", "g_cm_3", "gcm3"),
    "F": ("hz", "khz", "mhz", "ghz"),
    "S": ("pct", "percent", "percentage", "fraction", "wt_pct"),
}

# Canonical symbol -> the DielectricRecord field it populates.
VAR_TO_FIELD: dict[str, str] = {
    "T": "temperature_c",
    "M": "moisture_pct",
    "RHO": "bulk_density",
    "F": "frequency_mhz",
    "S": "salt_content",
}

# Fallback grids, used only when neither the model domain nor the paper
# metadata gives a range.
_DEFAULT_GRIDS: dict[str, list[float]] = {
    "T": [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
    "M": [10.0, 15.0, 20.0, 25.0, 30.0],
    "RHO": [],
    "F": [],
    "S": [],
}

# Target number of grid points per variable when a domain is known.
_GRID_STEPS: dict[str, int] = {"T": 8, "M": 5, "RHO": 4, "F": 0, "S": 3}


def canonical_var(name: Any) -> str | None:
    """Map a free-text variable name onto a canonical symbol (T, M, RHO...)."""
    if name is None:
        return None
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    key = re.sub(r"[^a-z0-9_]", "", key)
    key = re.sub(r"_+", "_", key).strip("_")
    if key in _VAR_ALIASES:
        return _VAR_ALIASES[key]

    # Accept only a complete known alias followed by a recognized unit or
    # basis suffix.  Sorting longest-first prevents ``kg_m3`` from being
    # partially interpreted as a shorter suffix.
    for sym, suffixes in _VAR_UNIT_SUFFIXES.items():
        for suffix in sorted(suffixes, key=len, reverse=True):
            marker = f"_{suffix}"
            if not key.endswith(marker):
                continue
            base = key[:-len(marker)]
            if _VAR_ALIASES.get(base) == sym:
                return sym
    return None


# ---------------------------------------------------------------------------
# Number parsing (PDF text is frequently mangled)
# ---------------------------------------------------------------------------

_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


def parse_number(raw: Any) -> float | None:
    """Parse a coefficient out of mangled PDF text.

    Handles: '- 31.27', '6.42 × 10 - 3', '3.17e-4', '5.83 * 1' (significance
    marker), unicode minus, superscript exponents, thin spaces.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)

    s = str(raw).strip()
    if not s:
        return None

    s = s.translate(_SUPERSCRIPT)
    s = (s.replace("−", "-").replace("–", "-").replace("—", "-")
           .replace(" ", " ").replace(" ", " "))

    # Drop significance markers: trailing '*', '*1', ' a', ' b'
    s = re.sub(r"[*†‡]+\s*\d*\s*$", "", s).strip()
    s = re.sub(r"\s+[a-z]$", "", s).strip()

    # '6.42 × 10 - 3'  ->  6.42e-3
    sci = re.match(
        r"^([+-]?\s*\d*\.?\d+)\s*[×xX*]\s*10\s*([+-]?)\s*(\d+)$", s
    )
    if sci:
        mant = float(sci.group(1).replace(" ", ""))
        sign = -1 if sci.group(2) == "-" else 1
        return mant * (10.0 ** (sign * int(sci.group(3))))

    # '3.17e-4' / '3.17E-4'
    s = re.sub(r"\s*([eE])\s*([+-]?)\s*(\d+)", r"\1\2\3", s)

    # '- 0.074' -> '-0.074'; remove spaces inside the number
    s = re.sub(r"^([+-])\s+", r"\1", s)
    s = re.sub(r"(?<=\d)\s+(?=\d)", "", s)
    s = s.replace(" ", "")

    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Model representation
# ---------------------------------------------------------------------------

@dataclass
class Term:
    """One additive term: coef * prod(var**power)."""
    coef: float
    powers: dict[str, int] = field(default_factory=dict)

    def value(self, bindings: dict[str, float]) -> float:
        out = self.coef
        for var, p in self.powers.items():
            out *= bindings[var] ** p
        return out

    def as_text(self) -> str:
        if not self.powers:
            return f"{self.coef:g}"
        parts = "".join(
            f"*{v}" if p == 1 else f"*{v}^{p}"
            for v, p in sorted(self.powers.items())
        )
        return f"{self.coef:g}{parts}"


@dataclass
class EquationModel:
    """An empirical model for one property of one material at one frequency."""
    material_name: str
    prop: str                                   # dielectric_constant | loss_factor | loss_tangent
    terms: list[Term] = field(default_factory=list)
    expression: str | None = None
    variables: list[str] = field(default_factory=list)   # canonical symbols
    domain: dict[str, tuple[float, float]] = field(default_factory=dict)
    frequency_mhz: float | None = None
    r_squared: float | None = None
    source_table: str = ""
    notes: str = ""

    # -- evaluation ---------------------------------------------------------

    def evaluate(self, bindings: dict[str, float]) -> float | None:
        """Evaluate the model. Returns None if it cannot be evaluated."""
        missing = [v for v in self.variables if v not in bindings]
        if missing:
            return None
        try:
            if self.expression:
                return _safe_eval(self.expression, bindings)
            if not self.terms:
                return None
            return sum(t.value(bindings) for t in self.terms)
        except (ArithmeticError, ValueError, KeyError, OverflowError):
            return None

    def in_domain(self, bindings: dict[str, float]) -> bool:
        for var, (lo, hi) in self.domain.items():
            if var in bindings and not (lo - 1e-9 <= bindings[var] <= hi + 1e-9):
                return False
        return True

    def as_text(self) -> str:
        if self.expression:
            return self.expression
        return " + ".join(t.as_text() for t in self.terms)


# ---------------------------------------------------------------------------
# Safe expression evaluation
# ---------------------------------------------------------------------------

_ALLOWED_FUNCS = {
    "exp": math.exp, "log": math.log, "log10": math.log10,
    "sqrt": math.sqrt, "abs": abs, "pow": pow,
}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Call,
    ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
)


def _safe_eval(expr: str, bindings: dict[str, float]) -> float:
    """Evaluate a restricted arithmetic expression without executing code."""
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            # The evaluator intentionally normalizes every rejected expression
            # to ValueError; EquationModel.evaluate treats that as invalid data.
            raise ValueError(  # noqa: TRY004
                f"disallowed syntax in expression: {type(node).__name__}"
            )
        if (
            isinstance(node, ast.Call)
            and (
                not isinstance(node.func, ast.Name)
                or node.func.id not in _ALLOWED_FUNCS
            )
        ):
            raise ValueError("disallowed function call in expression")
        if (
            isinstance(node, ast.Name)
            and node.id not in bindings
            and node.id not in _ALLOWED_FUNCS
        ):
            raise ValueError(f"unknown symbol in expression: {node.id}")
    binary_ops = {
        ast.Add: lambda left, right: left + right,
        ast.Sub: lambda left, right: left - right,
        ast.Mult: lambda left, right: left * right,
        ast.Div: lambda left, right: left / right,
        ast.Pow: lambda left, right: left ** right,
    }
    unary_ops = {ast.USub: lambda value: -value, ast.UAdd: lambda value: value}

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("only numeric constants are allowed")  # noqa: TRY004
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in bindings:
                raise ValueError(f"unknown numeric symbol: {node.id}")
            return float(bindings[node.id])
        if isinstance(node, ast.BinOp):
            operation = binary_ops.get(type(node.op))
            if operation is None:
                raise ValueError("disallowed binary operator")
            return float(operation(evaluate(node.left), evaluate(node.right)))
        if isinstance(node, ast.UnaryOp):
            operation = unary_ops.get(type(node.op))
            if operation is None:
                raise ValueError("disallowed unary operator")
            return float(operation(evaluate(node.operand)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function = _ALLOWED_FUNCS[node.func.id]
            return float(function(*(evaluate(argument) for argument in node.args)))
        raise ValueError(f"disallowed expression node: {type(node).__name__}")

    return evaluate(tree)


# ---------------------------------------------------------------------------
# Subscript (response-surface) notation
# ---------------------------------------------------------------------------

_SUBSCRIPT_KEY = re.compile(r"^[a-zA-Zα-ωΑ-Ω_\s]*?[_\s]*(\d+)$")


def parse_subscript_key(key: str, var_order: list[str]) -> dict[str, int] | None:
    """Turn a response-surface coefficient label into variable exponents.

    ``var_order`` maps subscript digit -> variable, 1-indexed:
    var_order = ["M", "T"]  =>  digit 1 is M, digit 2 is T.

        'alpha 0'   -> {}                 (constant)
        'a1'        -> {M: 1}
        'alpha 12'  -> {M: 1, T: 1}
        'alpha 112' -> {M: 2, T: 1}
        'a222'      -> {T: 3}
    """
    m = _SUBSCRIPT_KEY.match(str(key).strip())
    if not m:
        return None
    digits = m.group(1)
    if digits == "0":
        return {}
    powers: dict[str, int] = {}
    for ch in digits:
        idx = int(ch)
        if idx < 1 or idx > len(var_order):
            return None
        var = var_order[idx - 1]
        powers[var] = powers.get(var, 0) + 1
    return powers


# ---------------------------------------------------------------------------
# Build models from LLM output
# ---------------------------------------------------------------------------

def _coerce_domain(raw: Any) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        var = canonical_var(k)
        if var is None or not isinstance(v, (list, tuple)) or len(v) != 2:
            continue
        lo, hi = parse_number(v[0]), parse_number(v[1])
        if lo is None or hi is None:
            continue
        out[var] = (min(lo, hi), max(lo, hi))
    return out


def build_model(eq: dict) -> EquationModel | None:
    """Construct an EquationModel from one LLM equation entry.

    Accepts, in order of preference: terms, subscripts, expression, coefficients.
    """
    prop = str(eq.get("property", "")).strip()
    if prop not in ("dielectric_constant", "loss_factor", "loss_tangent"):
        return None

    material = str(eq.get("material_name", "unknown")).strip() or "unknown"
    freq = parse_number(eq.get("frequency_mhz"))
    r2 = parse_number(eq.get("r_squared"))
    src = str(eq.get("source_table", "") or "")
    domain = _coerce_domain(eq.get("domain"))

    # Declared variable order, e.g. {"1": "moisture", "2": "temperature"} or
    # ["moisture_pct", "temperature_c"].
    raw_vars = eq.get("variables") or eq.get("variable_order")
    var_order: list[str] = []
    if isinstance(raw_vars, dict):
        for k in sorted(raw_vars, key=lambda s: str(s)):
            cv = canonical_var(raw_vars[k])
            if cv is None:
                return None
            var_order.append(cv)
    elif isinstance(raw_vars, (list, tuple)):
        for v in raw_vars:
            cv = canonical_var(v)
            if cv is None:
                return None
            var_order.append(cv)
    elif raw_vars is not None:
        return None

    # Duplicate canonical variables make response-surface subscripts
    # ambiguous (for example, both "M" and "moisture" declared separately).
    if len(set(var_order)) != len(var_order):
        return None

    model = EquationModel(
        material_name=material, prop=prop, frequency_mhz=freq,
        r_squared=r2, source_table=src, domain=domain,
        notes=str(eq.get("notes", "") or ""),
    )

    # -- 1. explicit terms --------------------------------------------------
    terms_raw = eq.get("terms")
    if isinstance(terms_raw, list) and terms_raw:
        terms: list[Term] = []
        for t in terms_raw:
            if not isinstance(t, dict):
                return None
            coef = parse_number(t.get("coef", t.get("coefficient")))
            if coef is None:
                return None
            powers: dict[str, int] = {}
            raw_powers = t.get("vars", t.get("powers")) or {}
            if not isinstance(raw_powers, dict):
                return None
            for vname, p in raw_powers.items():
                cv = canonical_var(vname)
                pv = parse_number(p)
                if cv is None or pv is None or pv < 0 or not float(pv).is_integer():
                    return None
                powers[cv] = int(pv)
            terms.append(Term(coef, powers))
        if terms:
            model.terms = terms
            model.variables = sorted({v for t in terms for v in t.powers})
            return model

    # -- 2. response-surface subscripts -------------------------------------
    subs = eq.get("subscripts") or eq.get("coefficients_by_subscript")
    if isinstance(subs, dict) and subs and var_order:
        terms = []
        for key, raw_val in subs.items():
            powers = parse_subscript_key(key, var_order)
            if powers is None:
                continue
            coef = parse_number(raw_val)
            if coef is None:
                continue
            terms.append(Term(coef, powers))
        if terms:
            model.terms = terms
            model.variables = sorted({v for t in terms for v in t.powers})
            return model

    # -- 3. free-form expression --------------------------------------------
    expr = eq.get("expression")
    if isinstance(expr, str) and expr.strip():
        cleaned = expr.strip()
        # Strip a leading "eps' =" style left-hand side.
        if "=" in cleaned:
            cleaned = cleaned.split("=", 1)[1].strip()
        used = var_order or [
            v for v in ("T", "M", "RHO", "F", "S")
            if re.search(rf"\b{v}\b", cleaned)
        ]
        if used:
            # Normalize notation commonly emitted when an equation is copied
            # from a paper. Python treats ``^`` as XOR, and the model may
            # declare canonical F while writing the expression with ``f`` or
            # ``frequency_mhz``. Keep this conversion tightly limited to the
            # variables the response explicitly declared.
            cleaned = cleaned.replace("^", "**").replace("×", "*")
            expression_aliases = {
                "T": ("temperature_c", "temperature", "temp", "t"),
                "M": ("moisture_content_pct", "moisture_content", "moisture", "mc", "w", "m"),
                "RHO": ("bulk_density", "density", "rho"),
                "F": ("frequency_mhz", "frequency", "freq", "f"),
                "S": ("salt_content", "salt", "s"),
            }
            for canonical in used:
                aliases = expression_aliases.get(canonical, ())
                pattern = r"\b(?:" + "|".join(map(re.escape, aliases)) + r")\b"
                cleaned = re.sub(pattern, canonical, cleaned, flags=re.IGNORECASE)
            model.expression = cleaned
            model.variables = list(dict.fromkeys(used))
            return model

    # -- 4. legacy univariate coefficient list ------------------------------
    coeffs = eq.get("coefficients")
    if isinstance(coeffs, (list, tuple)) and coeffs:
        raw_var = eq.get("variable")
        if raw_var is None or not str(raw_var).strip():
            # Historical univariate payloads omitted ``variable`` and always
            # described temperature polynomials, so retain that compatibility
            # only for a genuinely missing declaration.  An unrecognized
            # declaration must fail closed rather than silently becoming T.
            var = "T"
        else:
            var = canonical_var(raw_var)
            if var is None:
                return None
        terms = []
        for i, c in enumerate(coeffs):
            cv = parse_number(c)
            if cv is None:
                # A missing coefficient changes the polynomial order and can
                # silently turn a slope into an intercept. Reject the model.
                return None
            terms.append(Term(cv, {} if i == 0 else {var: i}))
        if terms:
            model.terms = terms
            model.variables = [var]
            return model

    return None


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def _linspace(lo: float, hi: float, n: int) -> list[float]:
    if n <= 1 or hi <= lo:
        return [round(lo, 4)]
    step = (hi - lo) / (n - 1)
    return [round(lo + i * step, 4) for i in range(n)]


def _source_values_in_model_units(
    var: str,
    values: list[float] | tuple[float, float],
    model_domain: tuple[float, float] | None,
) -> list[float]:
    """Convert source metadata to the units used by a declared model domain."""
    converted = [float(value) for value in values]
    if not converted or model_domain is None:
        return converted

    # Screener metadata is stored as percent and MHz.  Printed equations may
    # instead use a moisture fraction or GHz; a sub-unit model domain is the
    # bounded evidence used elsewhere in the evaluator to recognize that case.
    if var == "M" and model_domain[1] <= 1.0 and max(converted) > 1.0:
        return [value / 100.0 for value in converted]
    if var == "F" and model_domain[1] < 100.0 and max(converted) >= 1000.0:
        return [value / 1000.0 for value in converted]
    return converted


def build_grid(
    model: EquationModel,
    paper_ranges: dict[str, tuple[float, float]] | None = None,
    paper_levels: dict[str, list[float]] | None = None,
) -> list[dict[str, float]]:
    """Cartesian grid of variable bindings at which to evaluate ``model``.

    For each variable, the usable domain is the intersection of the equation's
    declared domain and the source paper's reported range.  Discrete reported
    levels are preferred, but only when they fall inside that intersection.
    If two source constraints do not overlap, no grid is returned; evaluating
    outside a published fit domain would create unsupported data.
    """
    paper_ranges = paper_ranges or {}
    paper_levels = paper_levels or {}

    axes: dict[str, list[float]] = {}
    for var in model.variables:
        model_domain = model.domain.get(var)
        source_domain = paper_ranges.get(var)
        if source_domain:
            source_values = _source_values_in_model_units(
                var, source_domain, model_domain
            )
            source_domain = (min(source_values), max(source_values))

        if model_domain and source_domain:
            lo_hi = (
                max(model_domain[0], source_domain[0]),
                min(model_domain[1], source_domain[1]),
            )
            if lo_hi[0] > lo_hi[1] + 1e-9:
                return []
        else:
            lo_hi = model_domain or source_domain

        raw_levels = paper_levels.get(var)
        levels = (
            _source_values_in_model_units(var, raw_levels, model_domain)
            if raw_levels else None
        )
        if levels:
            if lo_hi:
                lo, hi = lo_hi
                levels = [x for x in levels if lo - 1e-9 <= x <= hi + 1e-9]
                if not levels:
                    return []
            axes[var] = sorted({round(float(x), 4) for x in levels})
            continue

        if lo_hi:
            lo, hi = lo_hi
            axes[var] = _linspace(lo, hi, _GRID_STEPS.get(var, 5))
            continue

        default = _DEFAULT_GRIDS.get(var) or []
        if not default:
            return []          # cannot responsibly guess this variable
        axes[var] = list(default)

    if not axes:
        return []

    grid: list[dict[str, float]] = [{}]
    for var, values in axes.items():
        grid = [{**b, var: v} for b in grid for v in values]
    return grid


# ---------------------------------------------------------------------------
# Physical plausibility
# ---------------------------------------------------------------------------

_PROP_BOUNDS = {
    # (microwave bound, RF bound) — RF allows large ionic-conduction losses
    "dielectric_constant": (200.0, 300.0),
    "loss_factor": (200.0, 5000.0),
    "loss_tangent": (10.0, 50.0),
}
_RF_MAX_MHZ = 100.0


def plausible(prop: str, value: float, frequency_mhz: float | None) -> bool:
    """Physical-range check, matched to configs/thresholds.yaml."""
    if value is None or not math.isfinite(value):
        return False
    if value < 0:
        return False
    mw_max, rf_max = _PROP_BOUNDS.get(prop, (200.0, 5000.0))
    is_rf = frequency_mhz is not None and frequency_mhz <= _RF_MAX_MHZ
    return value <= (rf_max if is_rf else mw_max)


@dataclass
class EvalReport:
    """Audit trail for one paper's equation evaluation."""
    models_built: int = 0
    models_unparsed: int = 0
    points_evaluated: int = 0
    points_out_of_domain: int = 0
    points_implausible: int = 0
    points_kept: int = 0
    messages: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.models_built} models built "
            f"({self.models_unparsed} unparsed), "
            f"{self.points_kept}/{self.points_evaluated} points kept, "
            f"{self.points_out_of_domain} out of domain, "
            f"{self.points_implausible} implausible"
        )
