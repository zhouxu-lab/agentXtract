"""Evaluate pipeline output against user-supplied tidy reference records.

Reference data are never bundled with the public repository. CSV, TSV, and
Excel inputs must use the public output column names; multiple Excel sheets
are concatenated when they share that tidy schema.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

NUMERIC_COLUMNS = (
    "frequency_mhz",
    "temperature_c",
    "dielectric_constant",
    "loss_factor",
    "loss_tangent",
    "moisture_content_pct",
    "electrical_conductivity_s_m",
)
REQUIRED_REFERENCE_COLUMNS = {
    "paper_id",
    "frequency_mhz",
    "temperature_c",
    "dielectric_constant",
    "loss_factor",
}


def _extract_mean(cell) -> float | None:
    """Extract a numeric mean from a scalar or ``mean ± uncertainty`` cell."""
    if cell is None or pd.isna(cell):
        return None
    if isinstance(cell, (int, float)):
        return float(cell)
    text = str(cell).strip().replace("−", "-")
    if not text:
        return None
    text = re.split(r"\s*(?:±|\+/-|\+-)\s*", text, maxsplit=1)[0].strip()
    try:
        return float(text)
    except ValueError:
        return None


def normalize_paper_id(value) -> str:
    """Normalize a source label for matching across filename conventions."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"^\s*\d+\s*[.)-]\s*", "", text)
    text = re.sub(r"[_-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _normalize_columns(
    df: pd.DataFrame,
    *,
    ensure_paper_id: bool = False,
) -> pd.DataFrame:
    frame = df.copy()
    frame.columns = [str(column).lower().strip() for column in frame.columns]
    aliases = {
        "moisture_content %": "moisture_content_pct",
        "moisture content %": "moisture_content_pct",
        "frequency (mhz)": "frequency_mhz",
        "temperature (c)": "temperature_c",
        "electrical conductivity (s/m)": "electrical_conductivity_s_m",
    }
    frame = frame.rename(columns={
        old: new
        for old, new in aliases.items()
        if old in frame.columns and new not in frame.columns
    })
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].map(_extract_mean)
    if ensure_paper_id and "paper_id" not in frame.columns:
        frame["paper_id"] = ""
    if "paper_id" in frame.columns:
        frame["paper_key"] = frame["paper_id"].map(normalize_paper_id)
    return frame


def _read_tabular(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(source, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        sheets = pd.read_excel(source, sheet_name=None)
        nonempty = {
            str(name): table
            for name, table in sheets.items()
            if table is not None and not table.dropna(how="all").empty
        }
        if not nonempty:
            return pd.DataFrame()
        normalized = {
            name: _normalize_columns(table)
            for name, table in nonempty.items()
        }
        usable = [
            table.assign(reference_sheet=name)
            for name, table in normalized.items()
            if REQUIRED_REFERENCE_COLUMNS <= set(table.columns)
        ]
        if not usable:
            missing_by_sheet = {
                name: sorted(REQUIRED_REFERENCE_COLUMNS - set(table.columns))
                for name, table in normalized.items()
            }
            raise ValueError(
                "Reference data must use the public long-form schema; "
                f"missing columns by Excel sheet: {missing_by_sheet}"
            )
        return pd.concat(usable, ignore_index=True, sort=False)
    raise ValueError(f"Unsupported tabular format: {source.suffix or '<none>'}")


def load_gold(path: str | Path) -> pd.DataFrame:
    """Load tidy reference records from CSV, TSV, or Excel."""
    frame = _normalize_columns(_read_tabular(path))
    missing = REQUIRED_REFERENCE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            "Reference data must use the public long-form schema; missing "
            f"columns: {sorted(missing)}"
        )
    if frame.empty:
        raise ValueError(f"Reference data contain no records: {path}")
    return frame


def load_pred(path: str | Path) -> pd.DataFrame:
    """Load pipeline predictions from a public-schema CSV."""
    frame = _normalize_columns(pd.read_csv(path), ensure_paper_id=True)
    if "electrical_conductivity_s_m" not in frame.columns:
        frame["electrical_conductivity_s_m"] = pd.NA
    if "salt_content" in frame.columns:
        conductivity = frame["salt_content"].astype("string").str.extract(
            r"^\s*([0-9]*\.?[0-9]+)\s*S\s*/\s*m\s*$",
            flags=re.IGNORECASE,
            expand=False,
        )
        migrate = conductivity.notna() & frame["electrical_conductivity_s_m"].isna()
        frame.loc[migrate, "electrical_conductivity_s_m"] = pd.to_numeric(
            conductivity[migrate], errors="coerce"
        )
        frame.loc[migrate, "salt_content"] = pd.NA
    return frame


def filter_pred_by_gold_paper_ids(
    gold: pd.DataFrame,
    pred: pd.DataFrame,
) -> pd.DataFrame:
    """Keep predictions belonging to a source represented in the reference."""
    if "paper_id" not in gold.columns or "paper_id" not in pred.columns:
        return pred
    reference = gold if "paper_key" in gold.columns else gold.assign(
        paper_key=gold["paper_id"].map(normalize_paper_id)
    )
    predictions = pred if "paper_key" in pred.columns else pred.assign(
        paper_key=pred["paper_id"].map(normalize_paper_id)
    )
    keys = set(reference["paper_key"].dropna().unique())
    return predictions[predictions["paper_key"].isin(keys)].reset_index(drop=True)


def filter_pred_by_gold_frequencies(
    gold: pd.DataFrame,
    pred: pd.DataFrame,
    freq_tol: float = 0.05,
) -> pd.DataFrame:
    """Keep frequencies represented in each source's reference records."""
    if "paper_id" not in gold.columns or "frequency_mhz" not in gold.columns:
        return pred
    reference = gold if "paper_key" in gold.columns else gold.assign(
        paper_key=gold["paper_id"].map(normalize_paper_id)
    )
    predictions = pred if "paper_key" in pred.columns else pred.assign(
        paper_key=pred["paper_id"].map(normalize_paper_id)
    )
    allowed = {
        key: set(group["frequency_mhz"].dropna().astype(float))
        for key, group in reference.groupby("paper_key")
    }
    keep: list[bool] = []
    for _, row in predictions.iterrows():
        frequency = row.get("frequency_mhz")
        frequencies = allowed.get(row.get("paper_key"), set())
        if pd.isna(frequency) or not frequencies:
            keep.append(True)
            continue
        keep.append(any(
            candidate > 0
            and abs(float(frequency) - candidate) / candidate <= freq_tol
            for candidate in frequencies
        ))
    return predictions[keep].reset_index(drop=True)


_MATERIAL_GENERIC_TOKENS = {
    "dried", "fresh", "kernel", "kernels", "sample", "slice", "slices",
}


def _material_tokens(value) -> set[str]:
    """Normalize a material label into identity-bearing word tokens."""
    if value is None or pd.isna(value):
        return set()
    words = re.findall(r"[a-z]+|[0-9]+(?:\.[0-9]+)?%?", str(value).lower())
    tokens: set[str] = set()
    for word in words:
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?%?", word):
            continue
        if word in {"percent", "pct", "wb", "db", "salt"}:
            continue
        if word.endswith("ies") and len(word) > 3:
            word = word[:-3] + "y"
        elif word.endswith("oes") and len(word) > 3:
            word = word[:-2]
        elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            word = word[:-1]
        if word not in _MATERIAL_GENERIC_TOKENS:
            tokens.add(word)
    return tokens


def _material_match(left, right) -> bool:
    """Accept a missing label or one identity-token set containing the other."""
    left_tokens = _material_tokens(left)
    right_tokens = _material_tokens(right)
    if not left_tokens or not right_tokens:
        return True
    return left_tokens <= right_tokens or right_tokens <= left_tokens


def match_records(
    gold: pd.DataFrame,
    pred: pd.DataFrame,
    freq_tol: float = 0.05,
    temp_tol: float = 2.0,
    moisture_tol: float = 2.0,
    require_reported_moisture: bool = False,
    require_reported_salt: bool = False,
    require_matching_moisture_basis: bool = False,
    require_reported_conductivity: bool = False,
    conductivity_tol: float = 0.01,
):
    """Match records by source and experimental conditions, never by values."""
    candidates: dict[int, list[tuple[float, int]]] = {}
    reference_by_source: dict[str, list[tuple[int, pd.Series]]] = {}
    for reference_index, reference_row in gold.iterrows():
        key = reference_row.get("paper_key") or normalize_paper_id(
            reference_row.get("paper_id")
        )
        reference_by_source.setdefault(key, []).append((reference_index, reference_row))

    for prediction_index, prediction in pred.iterrows():
        candidates[prediction_index] = []
        source_key = prediction.get("paper_key") or normalize_paper_id(
            prediction.get("paper_id")
        )
        for reference_index, reference in reference_by_source.get(source_key, []):
            predicted_frequency = prediction.get("frequency_mhz")
            reference_frequency = reference.get("frequency_mhz")
            if pd.isna(predicted_frequency) or pd.isna(reference_frequency):
                continue
            if reference_frequency > 0 and (
                abs(predicted_frequency - reference_frequency) / reference_frequency > freq_tol
            ):
                continue

            predicted_temperature = prediction.get("temperature_c")
            reference_temperature = reference.get("temperature_c")
            if pd.isna(predicted_temperature) or pd.isna(reference_temperature):
                continue
            if abs(predicted_temperature - reference_temperature) > temp_tol:
                continue
            if not _material_match(
                prediction.get("material_name"), reference.get("material_name")
            ):
                continue

            predicted_moisture = prediction.get("moisture_content_pct")
            reference_moisture = reference.get("moisture_content_pct")
            if require_reported_moisture and pd.notna(reference_moisture) and pd.isna(predicted_moisture):
                continue
            if pd.notna(predicted_moisture) and pd.notna(reference_moisture):
                if abs(predicted_moisture - reference_moisture) > moisture_tol:
                    continue
                if require_matching_moisture_basis:
                    predicted_basis = str(prediction.get("moisture_basis", "")).lower()
                    reference_basis = str(reference.get("moisture_basis", "")).lower()
                    if (
                        predicted_basis in {"wet", "dry"}
                        and reference_basis in {"wet", "dry"}
                        and predicted_basis != reference_basis
                    ):
                        continue

            predicted_salt = "" if pd.isna(prediction.get("salt_content")) else str(
                prediction.get("salt_content")
            ).strip().lower()
            reference_salt = "" if pd.isna(reference.get("salt_content")) else str(
                reference.get("salt_content")
            ).strip().lower()
            if require_reported_salt and reference_salt and not predicted_salt:
                continue
            if predicted_salt and reference_salt:
                predicted_numbers = re.findall(r"[\d.]+", predicted_salt)
                reference_numbers = re.findall(r"[\d.]+", reference_salt)
                if (
                    predicted_numbers
                    and reference_numbers
                    and abs(float(predicted_numbers[0]) - float(reference_numbers[0])) > 0.1
                ):
                    continue

            predicted_conductivity = prediction.get("electrical_conductivity_s_m")
            reference_conductivity = reference.get("electrical_conductivity_s_m")
            if (
                require_reported_conductivity
                and pd.notna(reference_conductivity)
                and pd.isna(predicted_conductivity)
            ):
                continue
            if (
                pd.notna(predicted_conductivity)
                and pd.notna(reference_conductivity)
                and abs(float(predicted_conductivity) - float(reference_conductivity))
                > conductivity_tol
            ):
                continue

            moisture_penalty = 0.0
            if pd.notna(predicted_moisture) and pd.notna(reference_moisture):
                moisture_penalty = abs(predicted_moisture - reference_moisture) * 5
            elif pd.isna(predicted_moisture) != pd.isna(reference_moisture):
                moisture_penalty = 50.0
            optional_penalty = 100.0 if bool(predicted_salt) != bool(reference_salt) else 0.0
            if pd.notna(predicted_conductivity) and pd.notna(reference_conductivity):
                optional_penalty += abs(
                    float(predicted_conductivity) - float(reference_conductivity)
                ) * 10
            elif pd.isna(predicted_conductivity) != pd.isna(reference_conductivity):
                optional_penalty += 100.0
            material_penalty = 0.0
            if _material_tokens(prediction.get("material_name")) != _material_tokens(
                reference.get("material_name")
            ):
                material_penalty = 10.0
            score = (
                abs(predicted_frequency - reference_frequency)
                + abs(predicted_temperature - reference_temperature) * 10
                + moisture_penalty
                + optional_penalty
                + material_penalty
            )
            candidates[prediction_index].append((score, reference_index))
        candidates[prediction_index].sort(key=lambda item: (item[0], item[1]))

    prediction_groups: dict[str, list[int]] = {}
    for prediction_index, row in pred.iterrows():
        key = row.get("paper_key") or normalize_paper_id(row.get("paper_id"))
        prediction_groups.setdefault(key, []).append(prediction_index)

    pairs: list[tuple[int, int]] = []
    for source_key, prediction_indices in prediction_groups.items():
        reference_indices = [
            index
            for index, row in gold.iterrows()
            if (row.get("paper_key") or normalize_paper_id(row.get("paper_id")))
            == source_key
        ]
        if not reference_indices:
            continue
        prediction_positions = {
            index: position for position, index in enumerate(prediction_indices)
        }
        reference_positions = {
            index: position for position, index in enumerate(reference_indices)
        }
        scores = [
            score
            for prediction_index in prediction_indices
            for score, reference_index in candidates[prediction_index]
            if reference_index in reference_positions
        ]
        maximum = max(scores, default=0.0)
        unmatched_cost = (maximum + 1.0) * (len(prediction_indices) + 1)
        invalid_cost = unmatched_cost * 2
        cost = np.full(
            (len(prediction_indices), len(reference_indices) + len(prediction_indices)),
            unmatched_cost,
            dtype=float,
        )
        cost[:, :len(reference_indices)] = invalid_cost
        for prediction_index in prediction_indices:
            row = prediction_positions[prediction_index]
            for score, reference_index in candidates[prediction_index]:
                if reference_index in reference_positions:
                    cost[row, reference_positions[reference_index]] = score
        row_indices, column_indices = linear_sum_assignment(cost)
        pairs.extend(
            (reference_indices[column], prediction_indices[row])
            for row, column in zip(row_indices, column_indices)
            if column < len(reference_indices) and cost[row, column] < unmatched_cost
        )

    pairs.sort()
    used_reference = {reference_index for reference_index, _ in pairs}
    used_predictions = {prediction_index for _, prediction_index in pairs}
    false_positives = [index for index in pred.index if index not in used_predictions]
    false_negatives = [index for index in gold.index if index not in used_reference]
    return pairs, false_positives, false_negatives


def compute_metrics(
    gold,
    pred,
    tp_pairs,
    fp_indices,
    fn_indices,
    value_tol: float = 0.05,
):
    """Compute record-level and value-level evaluation metrics."""
    true_positives = len(tp_pairs)
    false_positives = len(fp_indices)
    false_negatives = len(fn_indices)
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    value_errors = 0
    for reference_index, prediction_index in tp_pairs:
        reference = gold.loc[reference_index]
        prediction = pred.loc[prediction_index]
        for column in ("dielectric_constant", "loss_factor"):
            expected = reference.get(column)
            observed = prediction.get(column)
            if pd.isna(expected):
                continue
            if pd.isna(observed):
                value_errors += 1
                break
            if expected != 0 and abs(observed - expected) / abs(expected) > value_tol:
                value_errors += 1
                break
    value_accuracy = (
        (true_positives - value_errors) / true_positives if true_positives else 0
    )
    return {
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "value_accuracy": value_accuracy,
        "value_errors": value_errors,
    }


def per_paper_breakdown(gold, pred, tp_pairs, fp_indices, fn_indices):
    """Return metrics grouped by normalized source identifier."""
    raw = set(gold.get("paper_id", pd.Series(dtype=str)).dropna()) | set(
        pred.get("paper_id", pd.Series(dtype=str)).dropna()
    )
    keys = sorted({normalize_paper_id(value) for value in raw if normalize_paper_id(value)})
    breakdown = {}
    for key in keys:
        reference_count = int(gold["paper_id"].map(normalize_paper_id).eq(key).sum())
        prediction_count = int(pred["paper_id"].map(normalize_paper_id).eq(key).sum())
        true_positive_count = sum(
            normalize_paper_id(gold.loc[index].get("paper_id")) == key
            for index, _ in tp_pairs
        )
        false_positive_count = sum(
            normalize_paper_id(pred.loc[index].get("paper_id")) == key
            for index in fp_indices
        )
        false_negative_count = sum(
            normalize_paper_id(gold.loc[index].get("paper_id")) == key
            for index in fn_indices
        )
        precision = true_positive_count / (true_positive_count + false_positive_count) if true_positive_count + false_positive_count else 0
        recall = true_positive_count / (true_positive_count + false_negative_count) if true_positive_count + false_negative_count else 0
        breakdown[key] = {
            "gold": reference_count,
            "pred": prediction_count,
            "tp": true_positive_count,
            "fp": false_positive_count,
            "fn": false_negative_count,
            "precision": precision,
            "recall": recall,
        }
    return breakdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate agentXtract output")
    parser.add_argument("--gold", required=True, help="Tidy CSV, TSV, or Excel reference data")
    parser.add_argument("--pred", default="data/database/dielectric_properties.csv")
    parser.add_argument(
        "--matching-mode",
        choices=("condition-only", "moisture-required", "strict-conditions"),
        default="condition-only",
    )
    parser.add_argument("--freq-tol", type=float, default=0.05)
    parser.add_argument("--temp-tol", type=float, default=2.0)
    parser.add_argument("--moisture-tol", type=float, default=2.0)
    parser.add_argument("--value-tol", type=float, default=0.05)
    args = parser.parse_args()
    if not Path(args.gold).exists() or not Path(args.pred).exists():
        parser.error("Both --gold and --pred files must exist")

    gold = load_gold(args.gold)
    pred = filter_pred_by_gold_frequencies(
        gold,
        filter_pred_by_gold_paper_ids(gold, load_pred(args.pred)),
        freq_tol=args.freq_tol,
    )
    pairs, false_positives, false_negatives = match_records(
        gold,
        pred,
        freq_tol=args.freq_tol,
        temp_tol=args.temp_tol,
        moisture_tol=args.moisture_tol,
        require_reported_moisture=args.matching_mode != "condition-only",
        require_reported_salt=args.matching_mode == "strict-conditions",
        require_matching_moisture_basis=args.matching_mode == "strict-conditions",
        require_reported_conductivity=args.matching_mode == "strict-conditions",
    )
    metrics = compute_metrics(
        gold, pred, pairs, false_positives, false_negatives, args.value_tol
    )
    for name in ("precision", "recall", "f1", "value_accuracy"):
        print(f"{name}: {metrics[name]:.3f}")
    print(
        f"tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']} "
        f"value_errors={metrics['value_errors']}"
    )


if __name__ == "__main__":
    main()
