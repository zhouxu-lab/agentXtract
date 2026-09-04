"""Assemble extracted measurements into validated, reproducible data files.

The public assembler is deliberately corpus-agnostic. Project-specific repairs
belong in a local override file or an explicitly configured extension module;
neither is needed for ordinary pipeline runs.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import yaml

from src.paper_id import normalize_paper_id
from src.schema import (
    DATABASE_COLUMN_NAMES,
    DUCKDB_SCHEMA,
    ExtractionResult,
    ParsedPaper,
)

logger = logging.getLogger(__name__)

COLUMN_ORDER = DATABASE_COLUMN_NAMES


def _paper_identity_series(df: pd.DataFrame) -> pd.Series:
    """Return a collision-resistant per-row source key."""
    blank = pd.Series("", index=df.index, dtype="object")
    uid = df.get("paper_uid", blank).fillna("").astype(str).str.strip()
    doi = df.get("doi", blank).fillna("").astype(str).str.strip().str.lower()
    name = df.get("paper_id", blank).fillna("").astype(str).str.strip().str.lower()
    return uid.where(uid.ne(""), "doi:" + doi).where(
        uid.ne("") | doi.ne(""), "name:" + name
    )


def _stable_sort_records(df: pd.DataFrame) -> pd.DataFrame:
    """Put rows in a deterministic, provenance-aware order."""
    if df.empty:
        return df.reset_index(drop=True)
    columns = [column for column in COLUMN_ORDER if column in df.columns]
    return df.sort_values(
        columns, kind="mergesort", na_position="last"
    ).reset_index(drop=True)


def _normalize_moisture_basis(value: Any) -> str:
    """Return a stable basis key without conflating wet and dry conditions."""
    if value is None or pd.isna(value):
        return "unknown"
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "nan", "unknown", "unspecified"}:
        return "unknown"
    if normalized in {"wet", "wet basis", "wet-basis", "wb", "w.b.", "w.b"}:
        return "wet"
    if normalized in {"dry", "dry basis", "dry-basis", "db", "d.b.", "d.b"}:
        return "dry"
    return normalized


def _flatten_records(
    extraction_results: list[ExtractionResult],
    parsed_papers: dict[str, ParsedPaper],
) -> list[dict]:
    """Convert extraction results into dictionaries matching the export schema."""
    by_uid = {
        paper.paper_uid: paper
        for paper in parsed_papers.values()
        if getattr(paper, "paper_uid", "")
    }
    by_doi: dict[str, list[ParsedPaper]] = defaultdict(list)
    for paper in parsed_papers.values():
        doi = (getattr(paper.metadata, "doi", "") or paper.doi).strip().lower()
        if doi:
            by_doi[doi].append(paper)

    rows: list[dict] = []
    for result in extraction_results:
        result_uid = getattr(result, "paper_uid", "") or ""
        paper = by_uid.get(result_uid) if result_uid else None
        if paper is None and result.records:
            record_uid = getattr(result.records[0], "paper_uid", "") or ""
            paper = by_uid.get(record_uid) if record_uid else None
        if paper is None:
            matches = by_doi.get((result.doi or "").strip().lower(), [])
            if len(matches) == 1:
                paper = matches[0]
            elif len(matches) > 1:
                logger.warning(
                    "Ambiguous DOI %s maps to %d parsed sources; paper_uid is required",
                    result.doi,
                    len(matches),
                )

        metadata = paper.metadata if paper else None
        doi = metadata.doi if metadata and metadata.doi else result.doi
        title = metadata.title if metadata else ""
        authors = "; ".join(metadata.authors) if metadata and metadata.authors else ""
        year = metadata.year if metadata else None
        journal = metadata.journal if metadata else None
        method = metadata.measurement_method if metadata else None
        paper_basis = metadata.moisture_basis if metadata else None

        for record in result.records:
            source_table = (record.source_table or "").strip()
            if not source_table:
                if record.data_provenance == "measured_text" or record.extraction_source == "text":
                    source_table = "Narrative text"
                elif record.data_provenance == "equation_derived" or record.extraction_source == "equation":
                    source_table = "Regression equation"
                elif len(result.tables_processed) == 1:
                    source_table = result.tables_processed[0].replace("table_", "Table ")
                elif record.extraction_source == "table" or result.extraction_source == "table":
                    source_table = "Table (unspecified)"

            basis = record.moisture_basis
            if (
                record.moisture_content_pct is not None
                and _normalize_moisture_basis(basis) == "unknown"
                and _normalize_moisture_basis(paper_basis) != "unknown"
            ):
                basis = paper_basis

            rows.append({
                "paper_id": normalize_paper_id(
                    record.paper_id or (paper.pdf_path if paper else "")
                ),
                "paper_uid": (
                    getattr(record, "paper_uid", "")
                    or result_uid
                    or (getattr(paper, "paper_uid", "") if paper else "")
                ),
                "material_name": record.material_name,
                "frequency_mhz": record.frequency_mhz,
                "temperature_c": record.temperature_c,
                "dielectric_constant": record.dielectric_constant,
                "loss_factor": record.loss_factor,
                "loss_tangent": record.loss_tangent,
                "moisture_content_pct": record.moisture_content_pct,
                "moisture_basis": basis,
                "salt_content": record.salt_content,
                "electrical_conductivity_s_m": getattr(
                    record, "electrical_conductivity_s_m", None
                ),
                "source_table": source_table,
                "source_location": record.source_location,
                "data_provenance": record.data_provenance,
                "model_expression": record.model_expression,
                "model_r_squared": record.model_r_squared,
                "extraction_source": record.extraction_source,
                "extraction_model": record.extraction_model,
                "doi": doi,
                "title": title,
                "authors": authors,
                "year": year,
                "journal": journal,
                "measurement_method": record.measurement_method or method,
            })
    return rows


def _merge_split_records(df: pd.DataFrame) -> pd.DataFrame:
    """Merge a unique complementary property pair at identical conditions."""
    if df.empty:
        return df

    frame = df.copy()
    frame["_paper_key"] = _paper_identity_series(frame)
    frame["_freq_r"] = pd.to_numeric(
        frame["frequency_mhz"], errors="coerce"
    ).fillna(-999).round(0)
    frame["_temp_r"] = pd.to_numeric(
        frame["temperature_c"], errors="coerce"
    ).fillna(-999).round(0)
    frame["_mc_r"] = pd.to_numeric(
        frame["moisture_content_pct"], errors="coerce"
    ).fillna(-1).round(0)
    frame["_moisture_basis"] = frame.get(
        "moisture_basis", pd.Series(index=frame.index, dtype="object")
    ).apply(_normalize_moisture_basis)
    frame["_salt"] = frame["salt_content"].fillna("").astype(str).str.strip().str.lower()
    frame["_conductivity"] = frame.get(
        "electrical_conductivity_s_m", pd.Series(index=frame.index, dtype=float)
    ).fillna(-1.0).round(4)
    frame["_material"] = frame["material_name"].fillna("").astype(str).str.lower().str.strip()

    group_columns = [
        "_paper_key", "_material", "_freq_r", "_temp_r", "_mc_r",
        "_moisture_basis", "_salt", "_conductivity",
    ]
    merged_rows: list[pd.Series] = []
    for _, group in frame.groupby(group_columns, sort=False):
        if len(group) == 1:
            merged_rows.append(group.iloc[0])
            continue

        complete = group[
            group["dielectric_constant"].notna() & group["loss_factor"].notna()
        ]
        if not complete.empty:
            merged_rows.extend(row for _, row in group.iterrows())
            continue

        constant_rows = group[
            group["dielectric_constant"].notna() & group["loss_factor"].isna()
        ]
        loss_rows = group[
            group["loss_factor"].notna() & group["dielectric_constant"].isna()
        ]
        if len(constant_rows) != 1 or len(loss_rows) != 1:
            merged_rows.extend(row for _, row in group.iterrows())
            continue

        base = constant_rows.iloc[0].copy()
        complement = loss_rows.iloc[0]
        base["loss_factor"] = complement["loss_factor"]
        if pd.isna(base.get("loss_tangent")) and pd.notna(complement.get("loss_tangent")):
            base["loss_tangent"] = complement["loss_tangent"]
        for column in (
            "source_table", "source_location", "model_expression",
            "data_provenance", "extraction_source", "extraction_model",
        ):
            if column not in group.columns:
                continue
            values: list[str] = []
            for value in (base.get(column), complement.get(column)):
                text = "" if pd.isna(value) else str(value).strip()
                if text and text not in values:
                    values.append(text)
            base[column] = "; ".join(values) if values else None
        if "model_r_squared" in group.columns and pd.isna(base.get("model_r_squared")):
            base["model_r_squared"] = complement.get("model_r_squared")
        merged_rows.append(base)

    result = pd.DataFrame(merged_rows).reset_index(drop=True)
    temporary = [
        "_paper_key", "_freq_r", "_temp_r", "_mc_r", "_moisture_basis",
        "_salt", "_conductivity", "_material",
    ]
    return result.drop(columns=temporary, errors="ignore")


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact-condition duplicates with deterministic source priority."""
    if df.empty:
        return df

    frame = df.copy()
    frame["_paper_key"] = _paper_identity_series(frame)
    provenance_priority = {
        "measured_table": 0,
        "vision_table": 1,
        "measured_text": 2,
        "equation_derived": 3,
    }
    provenance = frame.get(
        "data_provenance", pd.Series(index=frame.index, dtype="object")
    ).fillna("").astype(str).str.strip().str.lower()
    source = frame.get(
        "extraction_source", pd.Series(index=frame.index, dtype="object")
    ).fillna("").astype(str).str.strip().str.lower()
    frame["_source_priority"] = provenance.map(provenance_priority).fillna(4)
    frame.loc[(frame["_source_priority"] == 4) & source.eq("table"), "_source_priority"] = 0
    frame.loc[(frame["_source_priority"] == 4) & source.eq("text"), "_source_priority"] = 2
    frame.loc[(frame["_source_priority"] == 4) & source.eq("equation"), "_source_priority"] = 3
    sort_columns = [column for column in COLUMN_ORDER if column in frame.columns]
    frame = frame.sort_values(
        ["_source_priority", *sort_columns], kind="mergesort", na_position="last"
    ).reset_index(drop=True)

    frame["_frequency"] = pd.to_numeric(
        frame["frequency_mhz"], errors="coerce"
    ).round(0)
    frame["_temperature"] = pd.to_numeric(
        frame["temperature_c"], errors="coerce"
    ).round(0)
    frame["_constant"] = pd.to_numeric(
        frame["dielectric_constant"], errors="coerce"
    ).round(1)
    frame["_loss"] = pd.to_numeric(frame["loss_factor"], errors="coerce").round(1)
    frame["_material"] = frame["material_name"].fillna("").astype(str).str.lower().str.strip()
    frame["_salt"] = frame["salt_content"].fillna("").astype(str).str.lower().str.strip()
    frame["_conductivity"] = frame.get(
        "electrical_conductivity_s_m", pd.Series(index=frame.index, dtype=float)
    ).fillna(-1.0).round(4)
    frame["_moisture"] = pd.to_numeric(
        frame.get("moisture_content_pct", pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    ).fillna(-1.0).round(1)
    frame["_moisture_basis"] = frame.get(
        "moisture_basis", pd.Series(index=frame.index, dtype="object")
    ).apply(_normalize_moisture_basis)

    condition_columns = [
        "_paper_key", "_material", "_frequency", "_temperature", "_moisture",
        "_moisture_basis", "_salt", "_conductivity",
    ]
    exact_columns = [*condition_columns, "_constant", "_loss"]
    frame = frame.drop_duplicates(subset=exact_columns, keep="first")

    complete = frame["dielectric_constant"].notna() & frame["loss_factor"].notna()
    complete_keys = {
        tuple(frame.loc[index, column] for column in condition_columns)
        for index in frame.index[complete]
    }
    keep = complete.copy()
    for index in frame.index[~complete]:
        key = tuple(frame.loc[index, column] for column in condition_columns)
        keep.loc[index] = key not in complete_keys
    frame = frame[keep].reset_index(drop=True)

    unknown_moisture = frame["_moisture"] == -1.0
    known_complete = (frame["_moisture"] != -1.0) & (
        frame["dielectric_constant"].notna() & frame["loss_factor"].notna()
    )
    non_moisture = [column for column in condition_columns if column != "_moisture"]
    known_keys = {
        tuple(frame.loc[index, column] for column in non_moisture)
        for index in frame.index[known_complete]
    }
    if known_keys:
        drop_unknown = pd.Series(False, index=frame.index)
        for index in frame.index[unknown_moisture]:
            key = tuple(frame.loc[index, column] for column in non_moisture)
            drop_unknown.loc[index] = key in known_keys
        frame = frame[~drop_unknown].reset_index(drop=True)

    temporary = [
        "_paper_key", "_source_priority", "_frequency", "_temperature",
        "_constant", "_loss", "_material", "_salt", "_conductivity",
        "_moisture", "_moisture_basis",
    ]
    return frame.drop(columns=temporary, errors="ignore").reset_index(drop=True)


def _normalize_material_names(
    df: pd.DataFrame,
    aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Normalize whitespace/case and apply optional project-owned aliases."""
    if "material_name" not in df.columns or df.empty:
        return df
    frame = df.copy()
    present = frame["material_name"].notna()
    labels = (
        frame.loc[present, "material_name"].astype(str)
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )
    counts = Counter(labels)
    variants: dict[str, set[str]] = defaultdict(set)
    for label in counts:
        variants[label.casefold()].add(label)
    canonical = {
        key: min(spellings, key=lambda value: (-counts[value], value))
        for key, spellings in variants.items()
    }
    normalized_aliases = {
        str(key).strip().casefold(): str(value).strip()
        for key, value in (aliases or {}).items()
        if str(key).strip() and str(value).strip()
    }
    canonical.update(normalized_aliases)
    frame.loc[present, "material_name"] = labels.map(
        lambda value: canonical[value.casefold()]
    )
    return frame


def _resolve_metadata(
    df: pd.DataFrame,
    parsed_papers: dict[str, ParsedPaper],
) -> pd.DataFrame:
    """Fill missing citation metadata using unambiguous source identity."""
    if df.empty or not parsed_papers:
        return df
    frame = df.copy()
    metadata_columns = ["title", "authors", "year", "journal", "measurement_method"]

    def metadata_of(paper: ParsedPaper) -> dict[str, Any] | None:
        metadata = getattr(paper, "metadata", None)
        if metadata is None:
            return None
        return {
            "title": metadata.title or "",
            "authors": "; ".join(metadata.authors) if metadata.authors else "",
            "year": metadata.year,
            "journal": metadata.journal or "",
            "measurement_method": metadata.measurement_method or "",
        }

    uid_metadata: dict[str, dict[str, Any]] = {}
    doi_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    uid_by_doi: dict[str, set[str]] = defaultdict(set)
    uid_by_name: dict[str, set[str]] = defaultdict(set)
    for paper in parsed_papers.values():
        metadata = metadata_of(paper)
        if metadata is None:
            continue
        uid = getattr(paper, "paper_uid", "")
        doi = (getattr(paper.metadata, "doi", "") or paper.doi).strip().lower()
        if uid:
            uid_metadata[uid] = metadata
            uid_by_name[normalize_paper_id(paper.pdf_path)].add(uid)
            if doi:
                uid_by_doi[doi].add(uid)
        if doi:
            doi_candidates[doi].append(metadata)
    doi_metadata = {
        doi: values[0] for doi, values in doi_candidates.items() if len(values) == 1
    }

    if "paper_uid" in frame.columns:
        blank_uid = frame["paper_uid"].isna() | frame["paper_uid"].astype(str).str.strip().eq("")
        doi_series = frame.get("doi", pd.Series("", index=frame.index)).fillna("").astype(str).str.lower()
        for doi, candidates in uid_by_doi.items():
            if len(candidates) == 1:
                mask = blank_uid & doi_series.eq(doi)
                frame.loc[mask, "paper_uid"] = next(iter(candidates))
                blank_uid &= ~mask
        if blank_uid.any() and "paper_id" in frame.columns:
            names = frame["paper_id"].fillna("").map(normalize_paper_id)
            for name, candidates in uid_by_name.items():
                if name and len(candidates) == 1:
                    mask = blank_uid & names.eq(name)
                    frame.loc[mask, "paper_uid"] = next(iter(candidates))
                    blank_uid &= ~mask

    def fill(mask: pd.Series, metadata: dict[str, Any]) -> None:
        for column in metadata_columns:
            if column not in frame.columns or not metadata.get(column):
                continue
            empty = mask & (
                frame[column].isna()
                | frame[column].astype(str).str.strip().isin(["", "None"])
            )
            frame.loc[empty, column] = metadata[column]

    if "paper_uid" in frame.columns:
        for uid, metadata in uid_metadata.items():
            fill(frame["paper_uid"].eq(uid), metadata)
    if "doi" in frame.columns:
        normalized_doi = frame["doi"].fillna("").astype(str).str.lower()
        for doi, metadata in doi_metadata.items():
            fill(normalized_doi.eq(doi), metadata)
    return frame


def _write_paper_coverage(
    df: pd.DataFrame,
    extraction_results: list[ExtractionResult],
    parsed_papers: dict[str, ParsedPaper],
    output_dir: Path,
    strict: bool = False,
) -> Path:
    """Write a per-source ledger so zero-yield inputs stay visible."""
    extracted_by_uid: dict[str, list[ExtractionResult]] = defaultdict(list)
    extracted_by_doi: dict[str, list[ExtractionResult]] = defaultdict(list)
    for result in extraction_results:
        if result.paper_uid:
            extracted_by_uid[result.paper_uid].append(result)
        if result.doi:
            extracted_by_doi[result.doi.strip().lower()].append(result)
    assembled_by_uid = Counter(
        str(value)
        for value in df.get("paper_uid", pd.Series(dtype=str)).fillna("")
        if str(value)
    )

    rows: list[dict] = []
    seen: set[str] = set()
    for paper in parsed_papers.values():
        key = paper.paper_uid or f"doi:{paper.doi.strip().lower()}"
        if key in seen:
            continue
        seen.add(key)
        doi = (getattr(paper.metadata, "doi", "") or paper.doi).strip().lower()
        results = extracted_by_uid.get(paper.paper_uid, []) if paper.paper_uid else []
        if not results and doi:
            results = extracted_by_doi.get(doi, [])
        records = [record for result in results for record in result.records]
        complete = sum(
            record.dielectric_constant is not None and record.loss_factor is not None
            for record in records
        )
        assembled = assembled_by_uid.get(paper.paper_uid, 0)
        if not getattr(paper, "parse_complete", True):
            outcome = "no_parse_artifact"
        elif not results:
            outcome = "no_extraction_artifact"
        elif any(result.timed_out for result in results):
            outcome = "extraction_timed_out"
        elif any(not result.complete for result in results):
            outcome = "extraction_incomplete"
        elif assembled:
            outcome = "contributed"
        elif not records:
            outcome = "extraction_zero_records"
        elif not complete:
            outcome = "incomplete_records_only"
        else:
            outcome = "lost_during_assembly"
        rows.append({
            "paper_uid": paper.paper_uid,
            "paper_id": normalize_paper_id(paper.pdf_path),
            "doi": doi,
            "title": paper.metadata.title,
            "extraction_artifacts": len(results),
            "extracted_records": len(records),
            "complete_extracted_records": complete,
            "assembled_records": assembled,
            "outcome": outcome,
        })

    columns = [
        "paper_uid", "paper_id", "doi", "title", "extraction_artifacts",
        "extracted_records", "complete_extracted_records", "assembled_records",
        "outcome",
    ]
    coverage = pd.DataFrame(rows, columns=columns)
    if not coverage.empty:
        coverage = coverage.sort_values(["outcome", "paper_id"], kind="mergesort")
    path = output_dir / "paper_coverage.csv"
    temporary = path.with_suffix(".csv.tmp")
    coverage.to_csv(temporary, index=False)
    temporary.replace(path)
    summary = {
        "selected_papers": len(coverage),
        "contributing_papers": int((coverage["outcome"] == "contributed").sum()),
        "zero_yield_papers": int((coverage["assembled_records"] == 0).sum()),
        "incomplete_papers": int(coverage["outcome"].isin({
            "no_parse_artifact", "extraction_timed_out", "extraction_incomplete",
        }).sum()),
        "assembled_records": len(df),
        "outcomes": coverage["outcome"].value_counts().to_dict(),
    }
    from src.utils import write_json_atomic

    write_json_atomic(output_dir / "paper_coverage_summary.json", summary)
    if summary["zero_yield_papers"] or summary["incomplete_papers"]:
        logger.warning(
            "Coverage: %d/%d inputs produced no rows; %d were incomplete; see %s",
            summary["zero_yield_papers"],
            summary["selected_papers"],
            summary["incomplete_papers"],
            path,
        )
        if strict:
            raise RuntimeError(
                f"Strict coverage failed: {summary['zero_yield_papers']} selected "
                f"papers produced no assembled records and "
                f"{summary['incomplete_papers']} had incomplete pipeline artifacts; "
                f"see {path}"
            )
    return path


def _load_validation_thresholds() -> dict:
    """Load validation bounds from ``configs/thresholds.yaml``."""
    defaults = {
        "frequency_range_mhz": [1.0, 100000.0],
        "temperature_range_c": [-80.0, 200.0],
        "dielectric_constant_range": [1.0, 200.0],
        "loss_factor_range": [0.001, 200.0],
        "moisture_range_pct": [0.0, 100.0],
        "max_loss_tangent": 10.0,
        "rf_frequency_threshold_mhz": 100.0,
        "rf_loss_factor_range": [0.001, 5000.0],
        "rf_dielectric_constant_range": [1.0, 300.0],
    }
    path = Path(__file__).resolve().parents[1] / "configs" / "thresholds.yaml"
    if not path.exists():
        return defaults
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return {**defaults, **loaded.get("validation", {})}


def _validate_records(df: pd.DataFrame) -> pd.DataFrame:
    """Remove records outside configurable physical bounds."""
    if df.empty:
        return df
    thresholds = _load_validation_thresholds()
    frequency_low, frequency_high = thresholds["frequency_range_mhz"]
    temperature_low, temperature_high = thresholds["temperature_range_c"]
    constant_low, constant_high = thresholds["dielectric_constant_range"]
    loss_low, loss_high = thresholds["loss_factor_range"]
    moisture_low, moisture_high = thresholds["moisture_range_pct"]
    rf_threshold = thresholds["rf_frequency_threshold_mhz"]
    rf_loss_low, rf_loss_high = thresholds["rf_loss_factor_range"]
    rf_constant_low, rf_constant_high = thresholds["rf_dielectric_constant_range"]

    frequency = pd.to_numeric(df["frequency_mhz"], errors="coerce")
    temperature = pd.to_numeric(df["temperature_c"], errors="coerce")
    constant = pd.to_numeric(df["dielectric_constant"], errors="coerce")
    loss = pd.to_numeric(df["loss_factor"], errors="coerce")
    moisture = pd.to_numeric(
        df.get("moisture_content_pct", pd.Series(index=df.index, dtype=float)),
        errors="coerce",
    )
    radio_frequency = frequency.notna() & frequency.le(rf_threshold)
    bad = (
        (frequency.notna() & ~frequency.between(frequency_low, frequency_high))
        | (temperature.notna() & ~temperature.between(temperature_low, temperature_high))
        | (~radio_frequency & constant.notna() & ~constant.between(constant_low, constant_high))
        | (radio_frequency & constant.notna() & ~constant.between(rf_constant_low, rf_constant_high))
        | (~radio_frequency & loss.notna() & ~loss.between(loss_low, loss_high))
        | (radio_frequency & loss.notna() & ~loss.between(rf_loss_low, rf_loss_high))
        | (moisture.notna() & ~moisture.between(moisture_low, moisture_high))
    )
    if "loss_tangent" in df.columns:
        tangent = pd.to_numeric(df["loss_tangent"], errors="coerce")
        bad |= (
            ~radio_frequency
            & tangent.notna()
            & tangent.gt(thresholds["max_loss_tangent"])
        )
    rejected = int(bad.sum())
    if rejected:
        logger.info("Validation removed %d out-of-bounds records", rejected)
    return df[~bad].reset_index(drop=True)


def _read_override_config(
    config: dict,
    provenance: dict[str, Any] | None = None,
) -> dict:
    """Load optional local assembly rules without requiring them in Git."""
    assembly = (config or {}).get("assembly", {}) or {}
    configured = assembly.get("overrides_file") or os.getenv(
        "AGENTXTRACT_ASSEMBLY_OVERRIDES"
    )
    if not configured:
        return {}
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(
            f"Configured assembly override file does not exist: {path}"
        )
    raw = path.read_bytes()
    payload = yaml.safe_load(raw.decode("utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("Assembly override file must contain a YAML mapping")
    if provenance is not None:
        provenance["override"] = {
            "selected_via": (
                "configuration" if assembly.get("overrides_file") else "environment"
            ),
            "content_sha256": hashlib.sha256(raw).hexdigest(),
        }
    return payload


def _rule_mask(df: pd.DataFrame, where: dict[str, Any]) -> pd.Series:
    """Build an exact-match mask for one declarative local override rule."""
    unknown = set(where) - set(df.columns)
    if unknown:
        raise ValueError(
            f"Assembly row rule names unknown filter columns: {sorted(unknown)}"
        )
    mask = pd.Series(True, index=df.index)
    for column, expected in where.items():
        series = df[column]
        if expected is None:
            mask &= series.isna()
        elif isinstance(expected, list):
            mask &= series.isin(expected)
        else:
            mask &= series.eq(expected)
    return mask


def _apply_configured_overrides(
    df: pd.DataFrame,
    config: dict,
    provenance: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Apply opt-in aliases and exact row rules from a local YAML file."""
    payload = _read_override_config(config, provenance=provenance)
    frame = df.copy()
    doi_overrides = payload.get("doi_overrides", {}) or {}
    if doi_overrides and "doi" in frame.columns:
        frame["doi"] = frame["doi"].replace(doi_overrides)
    aliases = payload.get("material_aliases", {}) or {}
    for rule in payload.get("row_rules", []) or []:
        if not isinstance(rule, dict) or not isinstance(rule.get("where"), dict):
            raise TypeError("Each assembly row rule requires a 'where' mapping")
        mask = _rule_mask(frame, rule["where"])
        if rule.get("drop"):
            frame = frame[~mask].copy()
            continue
        updates = rule.get("set", {}) or {}
        unknown = set(updates) - set(frame.columns)
        if unknown:
            raise ValueError(f"Assembly row rule names unknown columns: {sorted(unknown)}")
        for column, value in updates.items():
            frame.loc[mask, column] = value
    return frame.reset_index(drop=True), aliases


def _apply_extension_hooks(
    df: pd.DataFrame,
    parsed_papers: dict[str, ParsedPaper],
    config: dict,
    provenance: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run explicitly configured ``module:function`` postprocessors.

    Extension modules are ordinary Python code and therefore trusted input.
    Keep private/corpus-specific extensions outside this repository and opt in
    through ``assembly.postprocessors`` only when needed.
    """
    frame = df
    hooks = ((config or {}).get("assembly", {}) or {}).get("postprocessors", []) or []
    for dotted in hooks:
        if not isinstance(dotted, str) or ":" not in dotted:
            raise ValueError("Assembly postprocessors must use 'module:function' syntax")
        module_name, function_name = dotted.split(":", 1)
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)
        if not callable(function):
            raise TypeError(f"Assembly postprocessor {dotted!r} is not callable")
        if provenance is not None:
            source_paths: set[Path] = set()
            try:
                function_source = inspect.getsourcefile(function)
            except TypeError:
                function_source = None
            for source in (getattr(module, "__file__", None), function_source):
                if source:
                    source_paths.add(Path(source).resolve())
            readable_sources = [
                path for path in sorted(source_paths) if path.is_file()
            ]
            if not readable_sources:
                raise RuntimeError(
                    f"Cannot hash source for assembly postprocessor {dotted!r}"
                )
            provenance.setdefault("postprocessors", []).append({
                "identifier": dotted,
                "source_sha256": sorted({
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in readable_sources
                }),
            })
        result = function(frame.copy(), parsed_papers=parsed_papers, config=config)
        if not isinstance(result, pd.DataFrame):
            raise TypeError(f"Assembly postprocessor {dotted!r} did not return a DataFrame")
        frame = result
    return frame


def run(
    extraction_results: list[ExtractionResult],
    parsed_papers: dict[str, ParsedPaper],
    output_dir: Path,
    config: dict,
    append: bool = False,
) -> Path:
    """Assemble and export extracted measurements."""
    config = config or {}
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.table_extractor import _enrich_condition_metadata_from_source

    for paper in parsed_papers.values():
        _enrich_condition_metadata_from_source(paper)

    frame = pd.DataFrame(
        _flatten_records(extraction_results, parsed_papers),
        columns=COLUMN_ORDER,
    )
    assembly_extensions: dict[str, Any] = {}
    frame, aliases = _apply_configured_overrides(
        frame, config, provenance=assembly_extensions
    )
    frame = _apply_extension_hooks(
        frame, parsed_papers, config, provenance=assembly_extensions
    )
    frame = _normalize_material_names(frame, aliases=aliases)
    frame = _resolve_metadata(frame, parsed_papers)

    text_source = (
        frame["data_provenance"].eq("measured_text")
        | frame["extraction_source"].eq("text")
    )
    blank_source = frame["source_table"].isna() | frame["source_table"].astype(str).str.strip().eq("")
    frame.loc[text_source & blank_source, "source_table"] = "Narrative text"

    frame = _stable_sort_records(frame)
    frame = _merge_split_records(frame)

    tangent_mask = (
        frame["dielectric_constant"].notna()
        & frame["dielectric_constant"].gt(0)
        & frame["loss_factor"].notna()
        & frame["loss_tangent"].isna()
    )
    frame.loc[tangent_mask, "loss_tangent"] = (
        frame.loc[tangent_mask, "loss_factor"]
        / frame.loc[tangent_mask, "dielectric_constant"]
    ).round(4)

    if config.get("validation", {}).get("enabled", True):
        frame = _validate_records(frame)
    frame = _deduplicate(frame)

    for column in COLUMN_ORDER:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[COLUMN_ORDER]

    if append:
        existing_path = output_dir / "cowork.duckdb"
        if existing_path.exists():
            connection = duckdb.connect(str(existing_path), read_only=True)
            try:
                existing = connection.execute(
                    "SELECT * FROM dielectric_properties"
                ).df()
            finally:
                connection.close()
            frame = _deduplicate(
                _stable_sort_records(pd.concat([existing, frame], ignore_index=True))
            )
            frame = frame[COLUMN_ORDER]

    frame = _stable_sort_records(frame)
    csv_path = output_dir / "dielectric_properties.csv"
    csv_temporary = output_dir / "dielectric_properties.csv.tmp"
    frame.to_csv(csv_temporary, index=False)
    csv_temporary.replace(csv_path)

    parquet_path = output_dir / "dielectric_properties.parquet"
    parquet_temporary = output_dir / "dielectric_properties.parquet.tmp"
    frame.to_parquet(parquet_temporary, index=False)
    parquet_temporary.replace(parquet_path)

    database_path = output_dir / "cowork.duckdb"
    database_temporary = output_dir / "cowork.duckdb.tmp"
    database_temporary.unlink(missing_ok=True)
    connection = duckdb.connect(str(database_temporary))
    try:
        connection.execute(f"CREATE TABLE dielectric_properties ({DUCKDB_SCHEMA})")
        connection.execute("INSERT INTO dielectric_properties SELECT * FROM frame")
    finally:
        connection.close()
    database_temporary.replace(database_path)

    from src.provenance import write_run_provenance

    write_run_provenance(
        output_dir,
        frame,
        extraction_results,
        parsed_papers,
        config,
        assembly_extensions=assembly_extensions,
    )
    _write_paper_coverage(
        frame,
        extraction_results,
        parsed_papers,
        output_dir,
        strict=bool(config.get("strict_coverage", False)),
    )
    return database_path
