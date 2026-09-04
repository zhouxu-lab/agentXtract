"""Data quality audit for dielectric property database.

Phase 1: Rule-based physics checks (free, instant, deterministic)
Phase 2: LLM expert review of flagged records (optional, configured model)

Usage:
    python -m src.audit                          # Phase 1 only
    python -m src.audit --llm                    # Phase 1 + Phase 2 (LLM review)
    python -m src.audit --input path/to/csv      # Custom input
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CFG = {
    "water_upper_bound_2450": 87.0,
    "water_upper_bound_915": 88.0,
    "microwave_freq_threshold_mhz": 100.0,
    "divergence_pct": 20.0,
    "zscore_threshold": 3.0,
    "high_moisture_threshold_pct": 50.0,
    "max_loss_tangent_microwave": 10.0,
}

FLAG_COLUMNS = [
    "paper_key", "paper_uid", "paper_id", "doi", "title", "material_name",
    "frequency_mhz", "temperature_c", "moisture_content_pct",
    "moisture_basis", "salt_content", "electrical_conductivity_s_m",
    "source_table", "source_location", "data_provenance",
    "severity", "rule_id", "description", "current_value",
    "expected_range",
]


def _load_audit_config() -> dict:
    try:
        config_path = (
            Path(__file__).resolve().parents[1] / "configs" / "thresholds.yaml"
        )
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return {**DEFAULT_CFG, **cfg.get("audit", {})}
    except FileNotFoundError:
        return DEFAULT_CFG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flag(row, severity: str, rule_id: str, desc: str, value, expected: str) -> dict:
    return {
        "paper_key": row.get("_paper_key", ""),
        "paper_uid": row.get("paper_uid", ""),
        "paper_id": row.get("paper_id", ""),
        "doi": row.get("doi", ""),
        "title": row.get("title", ""),
        "material_name": row.get("material_name", ""),
        "frequency_mhz": row.get("frequency_mhz"),
        "temperature_c": row.get("temperature_c"),
        "moisture_content_pct": row.get("moisture_content_pct"),
        "moisture_basis": row.get("moisture_basis"),
        "salt_content": row.get("salt_content"),
        "electrical_conductivity_s_m": row.get("electrical_conductivity_s_m"),
        "source_table": row.get("source_table", ""),
        "source_location": row.get("source_location", ""),
        "data_provenance": row.get("data_provenance", ""),
        "severity": severity,
        "rule_id": rule_id,
        "description": desc,
        "current_value": value,
        "expected_range": expected,
    }


# ---------------------------------------------------------------------------
# Phase 1: Rule-based checks
# ---------------------------------------------------------------------------

def _check_water_upper_bound(df: pd.DataFrame, cfg: dict) -> list[dict]:
    """ε' should not exceed pure water (the highest ε' material)."""
    flags = []
    ub_2450 = cfg["water_upper_bound_2450"]
    ub_915 = cfg["water_upper_bound_915"]
    mw_thresh = cfg["microwave_freq_threshold_mhz"]

    for _, r in df.iterrows():
        dc = r["dielectric_constant"]
        if pd.isna(dc):
            continue
        freq = r["frequency_mhz"]
        if pd.isna(freq) or freq <= mw_thresh:
            continue
        limit = ub_2450 if freq >= 2000 else ub_915
        if dc > limit:
            flags.append(_flag(r, "CRITICAL", "WATER_UPPER_BOUND",
                f"ε'={dc:.1f} exceeds pure water limit ({limit}) at {freq:.0f} MHz",
                dc, f"≤ {limit}"))
    return flags


def _check_frequency_inversion(df: pd.DataFrame, cfg: dict) -> list[dict]:
    """Low-frequency ε' should be ≥ high-frequency ε' for the same conditions.

    This catches cases where the LLM extracted values from the wrong table
    or swapped frequency labels.
    """
    flags = []
    mw_thresh = cfg["microwave_freq_threshold_mhz"]

    # Group by paper + material + temp + MC
    group_cols = [
        "_paper_key", "material_name", "temperature_c",
        "moisture_content_pct", "moisture_basis", "salt_content",
        "electrical_conductivity_s_m",
    ]
    for key, group in df.groupby(group_cols, dropna=False):
        if len(group) < 2:
            continue
        rf = group[group["frequency_mhz"] <= mw_thresh]
        mw = group[group["frequency_mhz"] > mw_thresh]
        if rf.empty or mw.empty:
            continue
        rf_min_dc = rf["dielectric_constant"].min()
        mw_max_dc = mw["dielectric_constant"].max()
        if pd.isna(rf_min_dc) or pd.isna(mw_max_dc):
            continue
        if rf_min_dc < mw_max_dc * 0.8:  # allow 20% tolerance
            rf_row = rf.loc[rf["dielectric_constant"].idxmin()]
            mw_row = mw.loc[mw["dielectric_constant"].idxmax()]
            flags.append(_flag(rf_row, "CRITICAL", "FREQ_INVERSION",
                f"ε'={rf_min_dc:.1f} at {rf_row['frequency_mhz']:.0f} MHz < "
                f"ε'={mw_max_dc:.1f} at {mw_row['frequency_mhz']:.0f} MHz (same conditions)",
                rf_min_dc, f"≥ {mw_max_dc:.1f} (microwave value)"))
    return flags


def _check_negative_values(df: pd.DataFrame) -> list[dict]:
    """ε' must be ≥ 1 (vacuum) and ε'' must be ≥ 0."""
    flags = []
    for _, r in df.iterrows():
        dc = r["dielectric_constant"]
        lf = r["loss_factor"]
        if pd.notna(dc) and dc < 1.0:
            flags.append(_flag(r, "CRITICAL", "NEGATIVE_DC",
                f"ε'={dc:.2f} < 1.0 (physically impossible)", dc, "≥ 1.0"))
        if pd.notna(lf) and lf < 0:
            flags.append(_flag(r, "CRITICAL", "NEGATIVE_LF",
                f"ε''={lf:.2f} < 0 (physically impossible)", lf, "≥ 0"))
    return flags


def _check_duplicate_divergence(df: pd.DataFrame, cfg: dict) -> list[dict]:
    """Same conditions but values differ by >20%."""
    flags = []
    pct = cfg["divergence_pct"] / 100.0
    group_cols = [
        "_paper_key", "material_name", "frequency_mhz", "temperature_c",
        "moisture_content_pct", "moisture_basis", "salt_content",
        "electrical_conductivity_s_m",
    ]

    for _, group in df.groupby(group_cols, dropna=False):
        if len(group) < 2:
            continue
        dc_vals = group["dielectric_constant"].dropna()
        if len(dc_vals) >= 2:
            dc_range = dc_vals.max() - dc_vals.min()
            dc_mean = dc_vals.mean()
            if dc_mean > 0 and dc_range / dc_mean > pct:
                row = group.iloc[0]
                flags.append(_flag(row, "CRITICAL", "DUPLICATE_DIVERGE",
                    f"Same conditions, ε' ranges from {dc_vals.min():.1f} to {dc_vals.max():.1f} "
                    f"({dc_range/dc_mean*100:.0f}% spread)",
                    f"{dc_vals.min():.1f}–{dc_vals.max():.1f}", f"within {cfg['divergence_pct']}%"))
    return flags


def _check_dc_lf_swap(df: pd.DataFrame, cfg: dict) -> list[dict]:
    """At microwave frequencies, ε'' > ε' is unusual for non-salty high-moisture foods."""
    flags = []
    mw = cfg["microwave_freq_threshold_mhz"]
    mc_thresh = cfg["high_moisture_threshold_pct"]

    mask = (
        (df["frequency_mhz"] > mw)
        & df["dielectric_constant"].notna()
        & df["loss_factor"].notna()
        & (df["loss_factor"] > df["dielectric_constant"])
        & (df["moisture_content_pct"].notna())
        & (df["moisture_content_pct"] > mc_thresh)
        & (df["salt_content"].isna() | (df["salt_content"].astype(str).str.strip() == ""))
    )
    for _, r in df[mask].iterrows():
        flags.append(_flag(r, "WARNING", "DC_LF_SWAP",
            f"ε''={r['loss_factor']:.1f} > ε'={r['dielectric_constant']:.1f} at "
            f"{r['frequency_mhz']:.0f} MHz — possible column swap "
            f"(high-moisture, no salt)",
            r["loss_factor"], f"< {r['dielectric_constant']:.1f} (ε')"))
    return flags


def _check_within_paper_outliers(df: pd.DataFrame, cfg: dict) -> list[dict]:
    """Records with z-score > 3 compared to same paper's records."""
    flags = []
    z_thresh = cfg["zscore_threshold"]

    for pid, group in df.groupby("_paper_key"):
        for col, label in [("dielectric_constant", "ε'"), ("loss_factor", "ε''")]:
            vals = group[col].dropna()
            if len(vals) < 5:
                continue
            mean, std = vals.mean(), vals.std()
            if std == 0:
                continue
            for idx, val in vals.items():
                z = abs(val - mean) / std
                if z > z_thresh:
                    flags.append(_flag(group.loc[idx], "WARNING", "OUTLIER",
                        f"{label}={val:.1f} is {z:.1f}σ from paper mean "
                        f"({mean:.1f} ± {std:.1f})",
                        val, f"within {z_thresh}σ of {mean:.1f}"))
    return flags


def _check_temp_monotonicity(df: pd.DataFrame, cfg: dict) -> list[dict]:
    """At 2450 MHz for high-moisture foods, ε' should decrease with temperature."""
    flags = []
    mc_thresh = cfg["high_moisture_threshold_pct"]

    mask = (
        (df["frequency_mhz"].between(2400, 2500))
        & (df["moisture_content_pct"].notna())
        & (df["moisture_content_pct"] > mc_thresh)
    )
    subset = df[mask]
    group_cols = [
        "_paper_key", "material_name", "frequency_mhz",
        "moisture_content_pct", "moisture_basis", "salt_content",
    ]
    for _, group in subset.groupby(group_cols, dropna=False):
        if len(group) < 3:
            continue
        sorted_g = group.sort_values("temperature_c")
        dc_vals = sorted_g["dielectric_constant"].values
        temps = sorted_g["temperature_c"].values
        for i in range(1, len(dc_vals)):
            if pd.isna(dc_vals[i]) or pd.isna(dc_vals[i-1]):
                continue
            increase = dc_vals[i] - dc_vals[i-1]
            if increase > 5:  # >5 unit increase is suspicious
                row = sorted_g.iloc[i]
                flags.append(_flag(row, "WARNING", "TEMP_MONO",
                    f"ε' increased by {increase:.1f} from {temps[i-1]:.0f}°C "
                    f"({dc_vals[i-1]:.1f}) to {temps[i]:.0f}°C ({dc_vals[i]:.1f}) "
                    f"at 2450 MHz — expected decrease for high-moisture food",
                    dc_vals[i], f"≤ {dc_vals[i-1]:.1f}"))
    return flags


def _check_missing_paired_values(df: pd.DataFrame) -> list[dict]:
    """Records with ε' but no ε'' or vice versa."""
    flags = []
    dc_only = df["dielectric_constant"].notna() & df["loss_factor"].isna()
    lf_only = df["loss_factor"].notna() & df["dielectric_constant"].isna()
    for _, r in df[dc_only].iterrows():
        flags.append(_flag(r, "WARNING", "MISSING_LF",
            f"Has ε'={r['dielectric_constant']:.1f} but no ε''", "—", "both ε' and ε''"))
    for _, r in df[lf_only].iterrows():
        flags.append(_flag(r, "WARNING", "MISSING_DC",
            f"Has ε''={r['loss_factor']:.1f} but no ε'", "—", "both ε' and ε''"))
    return flags


def _check_missing_metadata(df: pd.DataFrame) -> list[dict]:
    """Records with empty title or authors."""
    flags = []
    seen_papers = set()
    for _, r in df.iterrows():
        pid = r["_paper_key"]
        if pid in seen_papers:
            continue
        missing = []
        if pd.isna(r.get("title")) or str(r.get("title", "")).strip() in ("", "nan"):
            missing.append("title")
        if pd.isna(r.get("authors")) or str(r.get("authors", "")).strip() in ("", "nan"):
            missing.append("authors")
        if missing:
            seen_papers.add(pid)
            flags.append(_flag(r, "INFO", "MISSING_META",
                f"Paper missing: {', '.join(missing)}", "—", "complete metadata"))
    return flags


def _check_unusual_loss_tangent(df: pd.DataFrame, cfg: dict) -> list[dict]:
    """tan δ > threshold at microwave frequencies."""
    flags = []
    mw = cfg["microwave_freq_threshold_mhz"]
    lt_max = cfg["max_loss_tangent_microwave"]
    mask = (
        (df["frequency_mhz"] > mw)
        & df["loss_tangent"].notna()
        & (df["loss_tangent"] > lt_max)
    )
    for _, r in df[mask].iterrows():
        flags.append(_flag(r, "INFO", "HIGH_LOSS_TAN",
            f"tan δ = {r['loss_tangent']:.2f} at {r['frequency_mhz']:.0f} MHz "
            f"(unusually high for microwave)",
            r["loss_tangent"], f"≤ {lt_max}"))
    return flags


# ---------------------------------------------------------------------------
# Phase 2: LLM expert review
# ---------------------------------------------------------------------------

async def _llm_review(flags_df: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    """Send CRITICAL and WARNING flags to the configured review model."""
    from src.utils import call_llm, load_config

    flags_df = flags_df.copy()
    source_df = source_df.copy()
    if "paper_key" not in flags_df.columns:
        flags_df["paper_key"] = "title:" + flags_df["title"].fillna("").astype(str)
    if "_paper_key" not in source_df.columns:
        source_df["_paper_key"] = (
            "title:" + source_df["title"].fillna("").astype(str)
        )

    model_config = load_config("configs/models.yaml")
    validator_config = model_config.get("validator", {})
    validator_model = validator_config.get("model")
    if not validator_model:
        raise KeyError("configs/models.yaml must define validator.model")

    review_mask = flags_df["severity"].isin(["CRITICAL", "WARNING"])
    to_review = flags_df[review_mask]
    if to_review.empty:
        logger.info("  No CRITICAL/WARNING flags to review")
        flags_df["llm_verdict"] = ""
        flags_df["llm_reasoning"] = ""
        return flags_df

    # Group flags by paper for efficient batching
    flags_df["llm_verdict"] = ""
    flags_df["llm_reasoning"] = ""

    papers = to_review["paper_key"].unique()
    logger.info(f"  LLM reviewing {len(to_review)} flags across {len(papers)} papers...")

    for pid in papers:
        paper_flags = to_review[to_review["paper_key"] == pid]
        paper_data = source_df[source_df["_paper_key"] == pid]

        # Build context: paper's data summary
        context_lines = []
        for _, r in paper_data.head(30).iterrows():
            dc = f"{r['dielectric_constant']:.1f}" if pd.notna(r["dielectric_constant"]) else "—"
            lf = f"{r['loss_factor']:.1f}" if pd.notna(r["loss_factor"]) else "—"
            mc = f"{r['moisture_content_pct']:.1f}%" if pd.notna(r["moisture_content_pct"]) else "—"
            salt = r["salt_content"] if pd.notna(r["salt_content"]) else "—"
            context_lines.append(
                f"  {r['material_name']}: f={r['frequency_mhz']}MHz T={r['temperature_c']}°C "
                f"MC={mc} salt={salt} ε'={dc} ε''={lf}"
            )

        # Build flag descriptions
        flag_lines = []
        flag_indices = []
        for idx, (_, flag) in enumerate(paper_flags.iterrows()):
            flag_lines.append(f"  Flag {idx+1} [{flag['severity']}] {flag['rule_id']}: {flag['description']}")
            flag_indices.append(_)

        prompt = f"""You are a food dielectric property expert reviewing automatically flagged data quality issues.

Paper: {paper_flags.iloc[0]['title']} ({pid})
Data sample (up to 30 rows from this paper):
{chr(10).join(context_lines)}

Flagged issues:
{chr(10).join(flag_lines)}

For each flag, provide:
1. VERDICT: "LIKELY_ERROR", "POSSIBLY_OK", or "FALSE_ALARM"
2. Brief reasoning (1-2 sentences) using dielectric property domain knowledge

Consider:
- High-moisture foods (>50% MC) have ε' = 40-80 at microwave frequencies
- At RF (27-40 MHz), ionic conduction can make ε'' much larger than ε'
- Starch gelatinization (50-90°C) causes sharp ε' increases in flour/grain
- Salty foods can have ε'' > ε' at any frequency
- ε' generally decreases with frequency for food materials

Format your response as:
Flag 1: VERDICT — reasoning
Flag 2: VERDICT — reasoning
..."""

        try:
            response, _cost = await call_llm(
                prompt=prompt,
                model=validator_model,
                max_tokens=validator_config.get("max_tokens", 1024),
                temperature=validator_config.get("temperature", 0.0),
                config=model_config,
            )

            # Parse LLM response — handles "Flag N: VERDICT — reasoning" or "Flag N: VERDICT: reasoning"
            for idx, (orig_idx, flag) in enumerate(paper_flags.iterrows()):
                for line in response.split("\n"):
                    flag_match = re.match(r"\s*Flag\s+(\d+)\b", line)
                    if flag_match is None or int(flag_match.group(1)) != idx + 1:
                        continue
                    if ("LIKELY_ERROR" in line.upper() or "POSSIBLY_OK" in line.upper() or "FALSE_ALARM" in line.upper()):
                        verdict = "UNKNOWN"
                        for v in ["LIKELY_ERROR", "POSSIBLY_OK", "FALSE_ALARM"]:
                            if v in line.upper():
                                verdict = v
                                break
                        # Extract reasoning after the verdict keyword
                        reasoning = ""
                        for sep in ["—", " - ", ":", "–"]:
                            pos = line.upper().find(verdict)
                            after = line[pos + len(verdict):]
                            after = after.lstrip(" :—–-")
                            if after.strip():
                                reasoning = after.strip()[:200]
                                break
                        flags_df.loc[orig_idx, "llm_verdict"] = verdict
                        flags_df.loc[orig_idx, "llm_reasoning"] = reasoning
                        break

        # Isolate arbitrary SDK/response failures to the current paper review.
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  LLM review failed for {pid}: {e}")

    reviewed = (flags_df["llm_verdict"] != "").sum()
    logger.info(f"  LLM reviewed {reviewed}/{len(to_review)} flags")
    return flags_df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_audit(
    csv_path: str | Path = "data/database/dielectric_properties.csv",
    output_path: str | Path = "data/database/audit_report.csv",
    use_llm: bool = False,
) -> pd.DataFrame:
    """Run data quality audit and produce report."""
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    blank = pd.Series("", index=df.index, dtype="object")
    uid = df.get("paper_uid", blank).fillna("").astype(str).str.strip()
    doi = df.get("doi", blank).fillna("").astype(str).str.strip().str.lower()
    paper_id = df.get("paper_id", blank).fillna("").astype(str).str.strip().str.lower()
    title = df.get("title", blank).fillna("").astype(str).str.strip().str.lower()
    df["_paper_key"] = uid.where(uid.ne(""), "doi:" + doi)
    df["_paper_key"] = df["_paper_key"].where(
        uid.ne("") | doi.ne(""), "paper:" + paper_id
    ).where(uid.ne("") | doi.ne("") | paper_id.ne(""), "title:" + title)
    for column in (
        "moisture_basis", "salt_content", "electrical_conductivity_s_m",
    ):
        if column not in df.columns:
            df[column] = None
    cfg = _load_audit_config()
    logger.info(f"  Auditing {len(df)} records from {csv_path}")

    # Phase 1: Rule-based checks
    all_flags = []
    rules = [
        ("WATER_UPPER_BOUND", _check_water_upper_bound, (df, cfg)),
        ("FREQ_INVERSION", _check_frequency_inversion, (df, cfg)),
        ("NEGATIVE_VALUES", _check_negative_values, (df,)),
        ("DUPLICATE_DIVERGE", _check_duplicate_divergence, (df, cfg)),
        ("DC_LF_SWAP", _check_dc_lf_swap, (df, cfg)),
        ("OUTLIER", _check_within_paper_outliers, (df, cfg)),
        ("TEMP_MONO", _check_temp_monotonicity, (df, cfg)),
        ("MISSING_PAIRED", _check_missing_paired_values, (df,)),
        ("MISSING_META", _check_missing_metadata, (df,)),
        ("HIGH_LOSS_TAN", _check_unusual_loss_tangent, (df, cfg)),
    ]
    for name, func, args in rules:
        flags = func(*args)
        if flags:
            logger.info(f"    {name}: {len(flags)} flags")
        all_flags.extend(flags)

    flags_df = pd.DataFrame(all_flags, columns=FLAG_COLUMNS)
    if flags_df.empty:
        logger.info("  No issues found!")

    # Phase 2: LLM review (optional)
    if use_llm and not flags_df.empty:
        logger.info("  Phase 2: LLM expert review...")
        flags_df = asyncio.run(_llm_review(flags_df, df))

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(".csv.tmp")
    flags_df.to_csv(temporary_output, index=False)
    temporary_output.replace(output_path)
    logger.info(f"  Audit report saved: {output_path}")

    # Print summary
    _print_summary(flags_df)
    return flags_df


def _print_summary(flags_df: pd.DataFrame):
    """Print console summary of audit results."""
    total = len(flags_df)
    print(f"\n{'='*60}")
    print("  DATA QUALITY AUDIT REPORT")
    print(f"{'='*60}")

    for severity in ["CRITICAL", "WARNING", "INFO"]:
        subset = flags_df[flags_df["severity"] == severity]
        if subset.empty:
            continue
        print(f"\n  {severity}: {len(subset)} flags")
        for rule_id, count in subset["rule_id"].value_counts().items():
            papers = subset[subset["rule_id"] == rule_id]["title"].nunique()
            print(f"    {rule_id}: {count} flags across {papers} papers")

    if "llm_verdict" in flags_df.columns:
        reviewed = flags_df[flags_df["llm_verdict"] != ""]
        if not reviewed.empty:
            print("\n  LLM REVIEW:")
            for verdict in ["LIKELY_ERROR", "POSSIBLY_OK", "FALSE_ALARM"]:
                n = (reviewed["llm_verdict"] == verdict).sum()
                if n:
                    print(f"    {verdict}: {n}")

    print(f"\n  TOTAL: {total} flags")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    parser = argparse.ArgumentParser(description="Audit dielectric property database")
    parser.add_argument("--input", default="data/database/dielectric_properties.csv",
                        help="Input CSV path")
    parser.add_argument("--output", default="data/database/audit_report.csv",
                        help="Output audit report path")
    parser.add_argument("--llm", action="store_true",
                        help="Enable Phase 2: LLM expert review of flagged records")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_audit(args.input, args.output, use_llm=args.llm)


if __name__ == "__main__":
    main()
