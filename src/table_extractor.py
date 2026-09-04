"""Table extractor — chunk → extract → merge.

Each table is processed independently. Large tables are chunked at
condition-group boundaries (≤20 rows per chunk) to avoid LLM truncation.
Chunks are extracted in parallel for speed.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import re
from pathlib import Path

from src.equation_model import (
    EquationModel,
    EvalReport,
    build_grid,
    build_model,
    plausible,
)
from src.paper_id import artifact_name, compute_uid, slugify
from src.schema import (
    CostEntry,
    DielectricRecord,
    ExtractionResult,
    PaperMetadata,
    ParsedPaper,
    ParsedTable,
    ScreenerResult,
)
from src.utils import (
    call_llm,
    load_skill,
    parse_json_safe,
    run_batch,
    strip_code_fences,
    write_json_atomic,
)

logger = logging.getLogger(__name__)


class IncompleteTableResponse(RuntimeError):
    """A model response was partial, with any salvageable rows attached."""

    def __init__(self, message: str, records: list, cost) -> None:
        super().__init__(message)
        self.records = records
        self.cost = cost


# -- Table page image rendering -----------------------------------------------

def _render_table_page(
    pdf_path: str,
    table: ParsedTable,
    image_dir: str | Path | None = None,
) -> str | None:
    """Render the PDF page containing a table as an image for vision-based extraction.

    Returns the path to the saved PNG image, or None if rendering fails.
    """
    try:
        import fitz
    except ImportError:
        return None

    pdf = Path(pdf_path)
    if not pdf.exists():
        return None

    # Determine which table number this is (from table_id like "table_2")
    table_num_match = re.search(r'(\d+)', table.table_id)
    table_num = int(table_num_match.group(1)) if table_num_match else 0
    search_term = f"Table {table_num}"

    try:
        doc = fitz.open(str(pdf))
        best_page = None
        best_score = 0

        for i in range(doc.page_count):
            page = doc[i]
            text = page.get_text()
            score = 0
            # Strong bonus for table caption (e.g., "Table 2—" at start of line)
            if re.search(rf'{search_term}\s*[-—\u2014]', text):
                score += 100
            elif search_term in text:
                score += 10
            # Check for data patterns (numbers with ±)
            score += len(re.findall(r'\d+\.\d+\s*±', text))
            if score > best_score:
                best_score = score
                best_page = i

        if best_page is None or best_score < 5:
            doc.close()
            return None

        page = doc[best_page]
        # Render at 3x zoom for crisp text
        mat = fitz.Matrix(3, 3)
        pix = page.get_pixmap(matrix=mat)

        # Save under the configured run directory. Falling back relative to
        # this package (not the caller's CWD) keeps direct helper use stable.
        img_dir = (
            Path(image_dir)
            if image_dir is not None
            else Path(__file__).resolve().parents[1] / "data" / "parsed" / "table_images"
        )
        img_dir.mkdir(parents=True, exist_ok=True)
        safe_name = f"{compute_uid(pdf)}__{slugify(pdf.stem)}"
        safe_table = slugify(table.table_id)
        img_path = img_dir / f"{safe_name}_{safe_table}.png"
        pix.save(str(img_path))
        doc.close()

        logger.info(f"    Rendered table page image: {img_path.name} ({pix.width}x{pix.height})")
        return str(img_path)
    # Page rendering is optional and PyMuPDF can raise backend-specific errors.
    except Exception as e:  # noqa: BLE001
        logger.debug(f"    Could not render table page: {e}")
        return None


# -- Table classification ----------------------------------------------------

def _is_dielectric_table(table: ParsedTable, paper_meta: PaperMetadata) -> bool:
    """Check if a table contains dielectric property data."""
    caption_lower = table.caption.lower()
    header_text = " ".join(
        str(h) for row in table.headers for h in row
    ).lower() if table.headers else ""
    # Also include first few data rows in text for detection
    rows = table.rows or [list(r.values()) for r in table.data_rows]
    data_sample = " ".join(str(c) for row in rows[:5] for c in row).lower() if rows else ""

    all_text = caption_lower + " " + header_text + " " + data_sample

    # Hard skip: penetration depth tables (OVERRIDES screener classification)
    # BUT only skip if the table does NOT also contain dielectric keywords
    # (some tables have both Dp and ε'/ε'' columns)
    penetration_keywords = [
        "penetration depth", "penetration depths",
        "dp (m", "dp(m", "dp (mm", "dp(mm", "dp (cm", "dp(cm",
        "power penetration",
    ]
    has_penetration = any(kw in all_text for kw in penetration_keywords)
    has_dielectric_kw = any(kw in all_text for kw in [
        "dielectric", "ε", "𝛜", "loss factor", "permittivity", "loss tangent",
        "tan δ", "epsilon", "ε'", "ε''",
        # Alternative notations in parsed tables
        "e 0", "e 00", "e'", "e''", "e′", "e″",
        # Abbreviations
        "dps",  # "dielectric properties"
    ])
    if has_penetration and not has_dielectric_kw:
        logger.info(f"    Hard-skip {table.table_id}: penetration depth detected in caption/headers")
        return False
    elif has_penetration and has_dielectric_kw:
        logger.info(f"    {table.table_id}: has penetration depth AND dielectric keywords — processing as mixed table")

    # Check for "(mm)" or "(cm)" in headers without any dielectric keyword
    has_mm_cm = bool(re.search(r'\(mm\)|\(cm\)|in mm|in cm', all_text))
    if has_mm_cm and not has_dielectric_kw:
        logger.info(f"    Hard-skip {table.table_id}: units (mm/cm) without dielectric keywords")
        return False

    # Heuristic: detect penetration depth tables by value pattern.
    # Penetration depth values decrease strongly with frequency and are typically 1-200 cm.
    # This catches tables without explicit "penetration depth" in caption/headers.
    if not has_dielectric_kw and rows and len(rows) >= 3:
        freq_col_count = sum(1 for h in (table.headers[0] if table.headers else [])
                             if re.search(r'\d+\s*(?:MHz|GHz)', str(h), re.IGNORECASE))
        if freq_col_count >= 3:
            decreasing_count = 0
            total_checked = 0
            for row in rows[:10]:
                # Skip first column (usually condition label like temperature)
                vals = []
                for cell in row[1:]:
                    nums = re.findall(r'[\d]+\.?\d*', str(cell).strip())
                    if nums:
                        try:
                            vals.append(float(nums[0]))
                        except ValueError:
                            pass
                if len(vals) >= 3:
                    total_checked += 1
                    is_decreasing = all(vals[i] >= vals[i+1] * 0.9 for i in range(len(vals)-1))
                    max_val = max(vals)
                    if is_decreasing and max_val < 200 and vals[0] > vals[-1] * 2:
                        decreasing_count += 1
            if total_checked >= 3 and decreasing_count / total_checked > 0.7:
                logger.info(f"    Hard-skip {table.table_id}: suspected penetration depth (monotonically decreasing values)")
                return False

    if _is_equation_table(table):
        return True

    # Use screener's classification if available
    if paper_meta.data_tables:
        # Match table_id patterns like "table_1" to "Table 1"
        for dt in paper_meta.data_tables:
            dt_norm = dt.lower().replace(" ", "_")
            tid_norm = table.table_id.lower().replace(" ", "_")
            if dt_norm == tid_norm or dt_norm.replace("table_", "table") == tid_norm:
                return True
    if paper_meta.skip_tables:
        for st in paper_meta.skip_tables:
            st_norm = st.lower().replace(" ", "_")
            tid_norm = table.table_id.lower().replace(" ", "_")
            if st_norm == tid_norm or st_norm.replace("table_", "table") == tid_norm:
                # Override screener skip if table clearly has dielectric data
                if has_dielectric_kw:
                    logger.info(f"    Overriding screener skip for {table.table_id}: has dielectric keywords")
                    return True
                return False

    # Skip indicators
    skip_keywords = [
        "penetration depth", "coefficient",
        "reference", "literature", "comparison with",
        "density", "proximate", "compositional",
        "author", "source",
    ]
    for kw in skip_keywords:
        if kw in all_text:
            has_dielectric = any(dk in all_text for dk in [
                "dielectric", "ε'", "ε''", "loss factor", "permittivity"
            ])
            if not has_dielectric:
                return False

    # Explicit dielectric indicator
    dielectric_keywords = [
        "dielectric", "ε'", "ε''", "epsilon", "loss factor",
        "permittivity", "tan δ", "loss tangent",
    ]
    for kw in dielectric_keywords:
        if kw in all_text:
            return True

    # Heuristic: check data cells for paired values
    rows = table.rows or [list(r.values()) for r in table.data_rows]
    if rows and len(rows) >= 3:
        for row in rows[:5]:
            for cell in row[1:]:
                cell_str = str(cell).strip()
                if not cell_str:
                    continue
                numbers = re.findall(r'[\d]+\.?\d*', cell_str)
                if len(numbers) >= 4:
                    return True
                if len(numbers) >= 2 and '±' in cell_str:
                    return True

    return False


def _is_equation_table(table: ParsedTable) -> bool:
    """Check if a table contains regression/polynomial equations, not raw data.

    Two-tier classifier:
    Tier 1: caption/header keywords (correlation, regression, R²) → always trust
    Tier 2: data cell equation patterns (T², ×10⁻) → only for small tables (≤15 rows)

    This prevents large measurement tables from being misclassified.
    """
    header_text = " ".join(
        str(h) for row in table.headers for h in row
    ).lower() if table.headers else ""
    caption_lower = table.caption.lower()
    text = caption_lower + " " + header_text

    # Tier 1: Strong indicators in caption/headers — always trust
    equation_keywords = [
        "correlation", "regression", "polynomial", "equation",
        "r²", "coefficient of determination",
    ]
    if any(kw in text for kw in equation_keywords):
        return True

    # Tier 2: equation syntax and coefficient-matrix structure. Explicit
    # equations may legitimately span dozens of material/frequency rows, so
    # table length is not a safe reason to suppress them.
    rows = table.rows or [list(r.values()) for r in table.data_rows]
    sample_text = " ".join(
        str(c) for row in rows[:12] for c in row
    ).lower() if rows else ""
    # Some parsers lose the caption and the printed equation but preserve a
    # coefficient matrix whose columns are simply a, b, c, ... and whose rows
    # identify eps'/eps''. Treat that structure as a model table, not as raw
    # dielectric measurements.
    header_tokens = {
        str(cell).strip().lower()
        for header_row in (table.headers or [])
        for cell in header_row
        if str(cell).strip()
    }
    coefficient_columns = len(header_tokens & {"a", "b", "c", "d", "e", "f"})
    has_property_rows = bool(re.search(
        r"(?:eps|ε|e\s*[r0'′″])", header_text + " " + sample_text
    ))
    has_fit_columns = any(
        marker in header_text
        for marker in ("r 2", "r2", "r²", "adjusted", "std. error", "standard error")
    )
    if coefficient_columns >= 2 and has_property_rows and has_fit_columns:
        return True
    explicit_equation = bool(re.search(
        r"(?:eps|ε|e\s*[r0'′″]).{0,20}(?:=|¼).{0,160}(?:[tmf]\s*(?:[²2-9]|\b)|×\s*10|\x02\s*10)",
        sample_text,
    ))
    if explicit_equation:
        return True
    has_equation_syntax = bool(re.search(
        r'[−\-+]\s*[\d.]+\s*t|t[²2]|×\s*10|[×x]\s*10[−⁻]', sample_text
    ))

    return has_equation_syntax


# -- Table chunking ----------------------------------------------------------

def _detect_groups_any_column(rows: list[list[str]], max_rows: int) -> list[list[list[str]]]:
    """Fallback: scan every column for repeating-block patterns."""
    if not rows or not rows[0]:
        return [rows]

    best_col = -1
    best_groups: list[list[list[str]]] = []

    for col_idx in range(len(rows[0])):
        vals = [row[col_idx].strip() if col_idx < len(row) else "" for row in rows]
        groups: list[list[list[str]]] = []
        current: list[list[str]] = []
        prev = None
        for i, row in enumerate(rows):
            v = vals[i]
            if v and v != prev and prev is not None and current:
                groups.append(current)
                current = [row]
            else:
                current.append(row)
            if v:
                prev = v
        if current:
            groups.append(current)

        # Good grouping = multiple groups, each ≤ max_rows, roughly equal size
        if (
            len(groups) >= 2
            and all(len(g) <= max_rows for g in groups)
            and (best_col == -1 or len(groups) > len(best_groups))
        ):
            best_col = col_idx
            best_groups = groups

    return best_groups if best_groups else [rows]


def chunk_table(table: ParsedTable, max_rows: int = 10) -> list[list[list[str]]]:
    """Split large tables into chunks for reliable extraction.

    Strategy:
    1. Detect condition-group boundaries (rows where a condition label
       changes, for example from condition A to condition B).
    2. If groups found: one chunk per group, header prepended to each.
    3. If no groups detected: split every max_rows rows.
    4. Every chunk gets the full table caption and headers.
    """
    rows = table.rows or [list(r.values()) for r in table.data_rows]
    if not rows:
        return []
    if len(rows) <= max_rows:
        return [rows]

    # Build condition groups by tracking first column value changes
    first_col_values = [row[0].strip() if row else "" for row in rows]
    groups: list[list[list[str]]] = []
    current_group: list[list[str]] = []
    prev_val = None

    for i, row in enumerate(rows):
        val = first_col_values[i]
        if val and val != prev_val and prev_val is not None and current_group:
            # First-column value changed → new condition group
            groups.append(current_group)
            current_group = [row]
        else:
            current_group.append(row)
        if val:
            prev_val = val
    if current_group:
        groups.append(current_group)

    # If detection found only 1 group (no changes detected), fall back
    # to scanning ALL columns for the one with fewest unique values in blocks
    if len(groups) <= 1 and len(rows) > max_rows:
        groups = _detect_groups_any_column(rows, max_rows)

    # Merge small groups into chunks
    chunks: list[list[list[str]]] = []
    current_chunk: list[list[str]] = []
    for group in groups:
        if len(current_chunk) + len(group) > max_rows and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
        current_chunk.extend(group)
    if current_chunk:
        chunks.append(current_chunk)

    # Force-split oversized chunks
    final_chunks: list[list[list[str]]] = []
    for chunk in chunks:
        if len(chunk) <= max_rows:
            final_chunks.append(chunk)
        else:
            for i in range(0, len(chunk), max_rows):
                final_chunks.append(chunk[i:i + max_rows])

    return final_chunks


def _describe_chunk(chunk_rows: list[list[str]], table: ParsedTable) -> str:
    if not chunk_rows:
        return ""
    first_vals = set()
    for row in chunk_rows:
        if row:
            first_vals.add(row[0].strip())
    if first_vals:
        return f"conditions: {', '.join(sorted(first_vals))}"
    return f"{len(chunk_rows)} rows"


def _get_methods_summary(paper: ParsedPaper) -> str:
    """Extract a brief methods summary (measurement method, instrument, sample prep)."""
    for section in paper.sections:
        if any(kw in section.heading.lower() for kw in ["method", "material", "experimental"]):
            text = section.text[:1500]  # 1500 chars max
            return text
    # Fallback: first 800 chars of full text
    return paper.full_text[:800]


# -- Prompt building --------------------------------------------------------

def _column_labels(table: ParsedTable) -> list[str]:
    """Return one stable, unique label for every parsed table column.

    Multi-level headers carry meaning vertically (for example, ``Loss factor``
    above ``915 MHz``).  Using only the first header row discards that context,
    while duplicate header text collapses columns when rows are represented as
    dictionaries.  Merge the non-blank header cells in each column and add a
    deterministic column suffix only when labels would otherwise repeat.
    """
    if not table.headers:
        return []

    width = max((len(row) for row in table.headers), default=0)
    labels: list[str] = []
    for column in range(width):
        parts: list[str] = []
        for row in table.headers:
            value = str(row[column]).strip() if column < len(row) else ""
            if value and value not in parts:
                parts.append(value)
        labels.append(" / ".join(parts) if parts else f"col{column + 1}")

    duplicate_labels = {label for label in labels if labels.count(label) > 1}
    return [
        f"{label} [col {index + 1}]" if label in duplicate_labels else label
        for index, label in enumerate(labels)
    ]

def _build_chunk_prompt(
    paper: ParsedPaper,
    table: ParsedTable,
    chunk_rows: list[list[str]],
    chunk_num: int,
    total_chunks: int,
    condition_desc: str,
) -> str:
    parts = [
        f"Paper DOI: {paper.doi}",
        f"Title: {paper.metadata.title}",
    ]

    if paper.metadata.primary_materials:
        parts.append(f"This paper measures: {', '.join(paper.metadata.primary_materials)}")
    if paper.metadata.measurement_frequencies_mhz:
        parts.append(f"Measurement frequencies: {paper.metadata.measurement_frequencies_mhz} MHz")
    parts.append("Extract ONLY data from this paper's own experiments, not cited values.")

    # Send only Methods section summary as context (not 3000 chars of intro)
    context = _get_methods_summary(paper)
    if context:
        parts.append(f"\nMethods context:\n{context}")

    parts.append(f"\n--- TABLE: {table.table_id} (chunk {chunk_num} of {total_chunks}) ---")
    parts.append(f"Caption: {table.caption}")

    if table.headers:
        parts.append(f"Headers: {table.headers}")

    parts.append(f"\nThis is chunk {chunk_num} of {total_chunks} from {table.table_id}.")
    parts.append(f"It contains rows for {condition_desc}.")
    parts.append(f"Data ({len(chunk_rows)} rows):")

    # Format rows with column names for clarity
    col_names = _column_labels(table)
    for i, row in enumerate(chunk_rows):
        if col_names and len(col_names) == len(row):
            named = {col_names[j]: row[j] for j in range(len(row))}
            parts.append(f"  Row {i + 1}: {named}")
        else:
            parts.append(f"  Row {i + 1}: {row}")

    footnotes = table.footnotes
    if isinstance(footnotes, list):
        footnotes = " ".join(footnotes)
    if footnotes:
        parts.append(f"Footnotes: {footnotes}")

    parts.append(f"\nExtract ALL {len(chunk_rows)} rows. After extraction, verify your record count matches the input row count.")

    return "\n".join(parts)


def _build_single_table_prompt(paper: ParsedPaper, table: ParsedTable) -> str:
    rows = table.rows or [list(r.values()) for r in table.data_rows]
    parts = [
        f"Paper DOI: {paper.doi}",
        f"Title: {paper.metadata.title}",
    ]

    if paper.metadata.primary_materials:
        parts.append(f"This paper measures: {', '.join(paper.metadata.primary_materials)}")
    if paper.metadata.measurement_frequencies_mhz:
        parts.append(f"Measurement frequencies: {paper.metadata.measurement_frequencies_mhz} MHz")
    parts.append("Extract ONLY data from this paper's own experiments, not cited values.")

    # Send only Methods section summary as context (not 3000 chars of intro)
    context = _get_methods_summary(paper)
    if context:
        parts.append(f"\nMethods context:\n{context}")

    parts.append(f"\n--- TABLE: {table.table_id} ---")
    parts.append(f"Caption: {table.caption}")

    if table.headers:
        parts.append(f"Headers: {table.headers}")

    if rows:
        parts.append(f"Data ({len(rows)} rows):")
        # Format rows with column names for clarity
        col_names = _column_labels(table)
        for i, row in enumerate(rows):
            if col_names and len(col_names) == len(row):
                named = {col_names[j]: row[j] for j in range(len(row))}
                parts.append(f"  Row {i + 1}: {named}")
            else:
                parts.append(f"  Row {i + 1}: {row}")

    footnotes = table.footnotes
    if isinstance(footnotes, list):
        footnotes = " ".join(footnotes)
    if footnotes:
        parts.append(f"Footnotes: {footnotes}")

    parts.append(f"\nExtract ALL rows. Verify your record count matches the {len(rows)} input rows.")

    return "\n".join(parts)


def _build_chunk_prompt_lite(
    paper: ParsedPaper,
    table: ParsedTable,
    chunk_rows: list[list[str]],
    chunk_num: int,
    total_chunks: int,
    condition_desc: str,
) -> str:
    """Lightweight prompt for chunks 2+. Omits methods context and paper metadata
    since the system prompt already covers extraction rules and chunk 1 established context."""
    parts = [
        f"Continue extracting from {table.table_id} (chunk {chunk_num} of {total_chunks}).",
        f"Paper: {paper.doi}",
        f"Caption: {table.caption}",
    ]

    if table.headers:
        parts.append(f"Headers: {table.headers}")

    parts.append(f"\nRows for {condition_desc} ({len(chunk_rows)} rows):")
    col_names = _column_labels(table)
    for i, row in enumerate(chunk_rows):
        if col_names and len(col_names) == len(row):
            named = {col_names[j]: row[j] for j in range(len(row))}
            parts.append(f"  Row {i + 1}: {named}")
        else:
            parts.append(f"  Row {i + 1}: {row}")

    parts.append(f"\nExtract ALL {len(chunk_rows)} rows. Verify your record count matches the input row count.")
    return "\n".join(parts)


# -- Record parsing ---------------------------------------------------------

# ISM band frequency normalization: precise values → common names
_ISM_NORMALIZE = {27.12: 27.0, 40.68: 40.0}


def _parse_records(
    data: dict,
    paper_doi: str,
    model: str,
    paper_id: str = "",
    provenance: str = "measured_table",
) -> list[DielectricRecord]:
    # Handle both {"records": [...]} and bare [...] response formats
    if isinstance(data, list):
        records_list = data
    elif isinstance(data, dict):
        records_list = data.get("records", data.get("data", []))
    else:
        logger.warning(f"  Unexpected LLM response type: {type(data)}, returning 0 records")
        return []

    records = []
    for rd in records_list:
        try:
            # Normalize moisture_basis
            mb = rd.get("moisture_basis", "unknown")
            if mb and mb.lower() in ("wet", "dry", "unknown"):
                mb = mb.lower()
            else:
                mb = None

            # Normalize ISM frequencies (27.12→27, 40.68→40)
            freq = rd.get("frequency_mhz")
            if freq is not None:
                freq = _ISM_NORMALIZE.get(freq, freq)

            records.append(DielectricRecord(
                material_name=rd.get("material_name", ""),
                dielectric_constant=rd.get("dielectric_constant"),
                loss_factor=rd.get("loss_factor"),
                loss_tangent=(
                    float(rd["loss_tangent"])
                    if rd.get("loss_tangent") not in (None, "", "n.d.", "N.A.")
                    else None
                ),
                frequency_mhz=freq,
                temperature_c=rd.get("temperature_c"),
                moisture_content_pct=rd.get("moisture_content_pct"),
                moisture_basis=mb,
                salt_content=rd.get("salt_content"),
                electrical_conductivity_s_m=rd.get(
                    "electrical_conductivity_s_m"
                ),
                measurement_method=rd.get("measurement_method"),
                source_table=rd.get("source_table", ""),
                source_location=rd.get("source_location", ""),
                doi=paper_doi,
                paper_id=paper_id,
                raw_text=rd.get("raw_text", ""),
                extraction_source=(
                    "table" if provenance in ("measured_table", "vision_table")
                    else "text"
                ),
                extraction_model=model,
                data_provenance=provenance,
            ))
        # Each model-produced row is an independent validation boundary; even
        # unusual numeric conversion failures must not discard valid siblings.
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  Skipping malformed record: {e}")
    return records


# -- Regression equation evaluation ------------------------------------------

# Standard temperature grid for evaluating regression equations (°C)
_DEFAULT_TEMPS = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]


def _eval_polynomial(coefficients: list[float], x: float) -> float:
    """Evaluate polynomial a0 + a1*x + a2*x^2 + ... at a given x.

    Retained for backward compatibility. New code goes through
    ``src.equation_model``, which handles multivariate models.
    """
    return sum(c * (x ** i) for i, c in enumerate(coefficients))


def _paper_ranges_and_levels(
    paper_meta: PaperMetadata | None,
) -> tuple[dict[str, tuple[float, float]], dict[str, list[float]]]:
    """Extract variable ranges and reported levels from paper metadata."""
    ranges: dict[str, tuple[float, float]] = {}
    levels: dict[str, list[float]] = {}
    if paper_meta is None:
        return ranges, levels

    tr = getattr(paper_meta, "temperature_range_c", None)
    if tr and tr[0] is not None and tr[1] is not None:
        lo, hi = sorted((float(tr[0]), float(tr[1])))
        ranges["T"] = (lo, hi)
        sampled = [t for t in _DEFAULT_TEMPS if lo <= t <= hi]
        levels["T"] = sorted({lo, *sampled, hi})
    else:
        # Preserve the pipeline's documented standard temperature grid for
        # legacy univariate models and incomplete screener metadata.
        levels["T"] = list(_DEFAULT_TEMPS)

    mr = getattr(paper_meta, "moisture_range_pct", None)
    if mr and mr[0] is not None and mr[1] is not None:
        ranges["M"] = (float(mr[0]), float(mr[1]))

    ml = getattr(paper_meta, "moisture_levels_pct", None) or []
    if ml:
        levels["M"] = [float(x) for x in ml]

    frequencies = getattr(paper_meta, "measurement_frequencies_mhz", None) or []
    if frequencies:
        levels["F"] = [float(x) for x in frequencies]

    return ranges, levels


def _enrich_condition_metadata_from_source(paper: ParsedPaper) -> None:
    """Fill missing moisture conditions from machine-readable source text.

    Equation tables often define ``M`` but put its measured levels in another
    table or in prose.  The general metadata pass occasionally misses that
    relationship.  This bounded fallback reads only explicit percentages; it
    never invents a range from equation behavior.
    """
    meta = paper.metadata
    if not meta.moisture_basis:
        moisture_context = []
        for table in paper.tables:
            table_text = " ".join(
                str(cell)
                for row in [*table.headers, *table.rows]
                for cell in row
            )
            if "moisture" in table_text.lower():
                moisture_context.append(table_text)
        source_text = " ".join([*moisture_context, paper.full_text or ""]).lower()
        context = " ".join(
            match.group(0)
            for match in re.finditer(
                r".{0,100}moisture.{0,100}", source_text, re.DOTALL
            )
        )
        wet = bool(re.search(r"(?:\bw\.\s*b\.?|\bwet\s+basis\b)", context))
        dry = bool(re.search(r"(?:\bd\.\s*b\.?|\bdry\s+basis\b)", context))
        if wet and not dry:
            meta.moisture_basis = "wet"
        elif dry and not wet:
            meta.moisture_basis = "dry"
    if not meta.moisture_levels_pct:
        for table in paper.tables:
            headers = " ".join(
                str(cell) for row in table.headers for cell in row
            ).lower()
            if "moisture" not in headers or not table.rows:
                continue
            levels = []
            for row in table.rows:
                if not row:
                    continue
                match = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)", str(row[0]))
                if match and 0 < float(match.group(1)) <= 100:
                    levels.append(float(match.group(1)))
            if len(set(levels)) >= 2:
                meta.moisture_levels_pct = sorted(set(levels))
                break

    if not meta.moisture_range_pct:
        if meta.moisture_levels_pct:
            meta.moisture_range_pct = (
                min(meta.moisture_levels_pct), max(meta.moisture_levels_pct)
            )
        else:
            pattern = re.compile(
                r"moisture content.{0,100}?(\d+(?:\.\d+)?)\s*%?\s*"
                r"(?:-|–|—|to)\s*(\d+(?:\.\d+)?)\s*%",
                re.IGNORECASE | re.DOTALL,
            )
            match = pattern.search(paper.full_text or "")
            if match:
                lo, hi = sorted((float(match.group(1)), float(match.group(2))))
                if 0 < lo < hi <= 100:
                    meta.moisture_range_pct = (lo, hi)


def _evaluate_equations(
    data: dict,
    paper_doi: str,
    paper_meta: PaperMetadata | None = None,
    paper_id: str = "",
    model: str | None = None,
) -> list[DielectricRecord]:
    """Convert regression equation coefficients into DielectricRecord objects
    by evaluating at standard temperatures.

    The LLM returns:
    {
      "equations": [
        {"material_name": "Sample liquid", "property": "dielectric_constant",
         "coefficients": [20.0, -0.05, -0.001], "variable": "temperature_c",
         "frequency_mhz": 900.0, "source_table": "Table A", "r_squared": 0.95},
        ...
      ]
    }

    We evaluate each equation at standard temps and pair up ε'/ε'' for the same
    material+frequency into single records.
    """
    records, _report = evaluate_equations_with_report(
        data, paper_doi, paper_meta, paper_id, model=model
    )
    return records


def evaluate_equations_with_report(
    data: dict,
    paper_doi: str,
    paper_meta: PaperMetadata | None = None,
    paper_id: str = "",
    model: str | None = None,
) -> tuple[list[DielectricRecord], EvalReport]:
    """Evaluate reported regression models into records, with an audit trail.

    Supports multivariate models — eps' and eps'' are commonly reported as
    functions of moisture content *and* temperature, not temperature alone.
    Returns the records plus an EvalReport explaining what was generated and
    what was rejected, so a paper never fails silently.
    """
    report = EvalReport()
    # ``model`` is also a natural loop-variable name throughout equation
    # evaluation.  Preserve the caller-supplied extractor model before those
    # loops so provenance cannot accidentally become an EquationModel object.
    extraction_model = model
    equations = data.get("equations", [])
    if not equations:
        return [], report

    paper_ranges, paper_levels = _paper_ranges_and_levels(paper_meta)

    paper_freqs: list[float] = []
    if paper_meta and paper_meta.measurement_frequencies_mhz:
        paper_freqs = [
            _ISM_NORMALIZE.get(f, f)
            for f in paper_meta.measurement_frequencies_mhz
        ]

    # Build models, dropping any the parser cannot make sense of.
    models: list[EquationModel] = []
    for eq in equations:
        m = build_model(eq)
        if m is None:
            report.models_unparsed += 1
            message = f"malformed equation entry: {str(eq)[:120]}"
            report.messages.append(message)
            logger.warning(f"  {message}")
            continue
        if m.frequency_mhz is not None:
            m.frequency_mhz = _ISM_NORMALIZE.get(m.frequency_mhz, m.frequency_mhz)
        models.append(m)
    report.models_built = len(models)

    if not models:
        return [], report

    # Models with no fixed frequency get expanded across the paper's
    # frequencies only when frequency is not itself an equation variable.
    # Expanding an F-dependent model first used (for example) 5 GHz as a
    # metadata label while evaluating the equation without F.
    if (
        all(m.frequency_mhz is None for m in models)
        and paper_freqs
        and not any("F" in m.variables for m in models)
    ):
        logger.info(
            f"  Models lack frequency — expanding across {paper_freqs}"
        )
        expanded: list[EquationModel] = []
        for m in models:
            for pf in paper_freqs:
                clone = copy.deepcopy(m)
                clone.frequency_mhz = pf
                expanded.append(clone)
        models = expanded

    # Pair eps' / eps'' / tan d models describing the same material and
    # frequency into one record. Papers very often put eps' in one table and
    # eps'' in the next, so source_table is NOT part of the pairing key —
    # it is only used to disambiguate when the same property is modelled
    # twice for the same material and frequency.
    from collections import defaultdict
    by_mat_freq: dict[tuple, list[EquationModel]] = defaultdict(list)
    for m in models:
        by_mat_freq[(m.material_name, m.frequency_mhz)].append(m)

    grouped: dict[tuple, dict[str, EquationModel]] = {}
    for (mat, freq), group in by_mat_freq.items():
        props_seen: dict[str, int] = defaultdict(int)
        for m in group:
            props_seen[m.prop] += 1
        if any(n > 1 for n in props_seen.values()):
            # Ambiguous — fall back to keeping source tables separate.
            for m in group:
                grouped.setdefault((mat, freq, m.source_table), {})[m.prop] = m
        else:
            src = "; ".join(sorted({m.source_table for m in group if m.source_table}))
            grouped[(mat, freq, src)] = {m.prop: m for m in group}

    records: list[DielectricRecord] = []
    for (mat, freq, src), props in grouped.items():
        dc_m = props.get("dielectric_constant")
        lf_m = props.get("loss_factor")
        lt_m = props.get("loss_tangent")

        # The grid is driven by whichever model is present; they share
        # variables in every case seen in the corpus.
        driver = dc_m or lf_m or lt_m
        grid = build_grid(driver, paper_ranges, paper_levels)
        if not grid:
            message = (
                f"no evaluable grid for {mat} @ {freq} MHz "
                f"(variables {driver.variables}, model domain {driver.domain}, "
                f"reported ranges {paper_ranges}, reported levels {paper_levels})"
            )
            report.messages.append(message)
            logger.warning(f"  {message}")
            continue

        # Some papers label moisture as percent in prose but use W as a mass
        # fraction in the printed regression.  When no domain states the unit,
        # permit a conservative, evidence-based fallback: use M/100 only if
        # every value in the declared-percent interpretation is implausible and
        # the fraction interpretation yields physically plausible values.
        moisture_eval_scale = 1.0
        if (
            "M" in driver.variables
            and "M" not in driver.domain
            and any(point.get("M", 0) > 1 for point in grid)
        ):
            def _plausible_count(
                scale: float,
                _grid=grid,
                _props=props,
                _freq=freq,
            ) -> int:
                count = 0
                for point in _grid:
                    candidate = dict(point)
                    candidate["M"] *= scale
                    for prop_name, equation_model in _props.items():
                        value = equation_model.evaluate(candidate)
                        if value is not None and plausible(prop_name, value, _freq):
                            count += 1
                return count

            percent_score = _plausible_count(1.0)
            fraction_score = _plausible_count(0.01)
            if percent_score == 0 and fraction_score >= 2:
                moisture_eval_scale = 0.01
                message = (
                    f"interpreted moisture variable as a fraction for {mat} "
                    f"@ {freq} MHz (plausible values {percent_score} -> "
                    f"{fraction_score})"
                )
                report.messages.append(message)
                logger.info(f"  {message}")

        for bindings in grid:
            report.points_evaluated += 1

            eval_bindings = dict(bindings)
            if "M" in eval_bindings:
                eval_bindings["M"] *= moisture_eval_scale

            if not driver.in_domain(eval_bindings):
                report.points_out_of_domain += 1
                continue

            # An equation variable F is commonly expressed in GHz even though
            # the database contract is MHz.  Evaluate in the printed units,
            # then normalize only the emitted record frequency.
            record_freq = freq
            if record_freq is None and "F" in bindings:
                record_freq = bindings["F"]
                if (
                    driver.domain.get("F")
                    and driver.domain["F"][1] < 100
                    and paper_freqs
                    and max(paper_freqs) >= 1000
                ):
                    record_freq *= 1000

            dc_val = dc_m.evaluate(eval_bindings) if dc_m else None
            lf_val = lf_m.evaluate(eval_bindings) if lf_m else None
            lt_val = lt_m.evaluate(eval_bindings) if lt_m else None

            # Published fits can misbehave near the corners of their own
            # domain — high-order response surfaces sometimes return negative
            # loss factors at the driest, coolest condition. Reject each
            # property on its own merits rather than discarding a whole grid
            # point, and record every rejection.
            dropped = 0
            if dc_val is not None and not plausible("dielectric_constant", dc_val, record_freq):
                report.points_implausible += 1
                report.messages.append(
                    f"implausible eps'={dc_val:.3g} for {mat} "
                    f"@ {record_freq} MHz at {bindings} — value dropped"
                )
                dc_val = None
                dropped += 1
            if lf_val is not None and not plausible("loss_factor", lf_val, record_freq):
                report.points_implausible += 1
                report.messages.append(
                    f"implausible eps''={lf_val:.3g} for {mat} "
                    f"@ {record_freq} MHz at {bindings} — value dropped"
                )
                lf_val = None
                dropped += 1
            if lt_val is not None and not plausible("loss_tangent", lt_val, record_freq):
                lt_val = None

            if dc_val is None and lf_val is None:
                continue

            model_text = "; ".join(
                f"{p}: {m.as_text()}" for p, m in props.items()
            )
            r2 = next((m.r_squared for m in props.values()
                       if m.r_squared is not None), None)

            records.append(DielectricRecord(
                material_name=mat,
                dielectric_constant=round(dc_val, 3) if dc_val is not None else None,
                loss_factor=round(lf_val, 3) if lf_val is not None else None,
                loss_tangent=round(lt_val, 4) if lt_val is not None else None,
                frequency_mhz=record_freq,
                temperature_c=bindings.get("T"),
                moisture_pct=(
                    bindings.get("M") * 100
                    if "M" in bindings
                    and driver.domain.get("M")
                    and driver.domain["M"][1] <= 1.0
                    else bindings.get("M")
                ),
                moisture_basis=(
                    paper_meta.moisture_basis
                    if paper_meta and "M" in bindings
                    else None
                ),
                salt_content=(str(bindings["S"]) if "S" in bindings else None),
                source_table=f"{src} (regression)" if src else "regression",
                doi=paper_doi,
                paper_id=paper_id,
                extraction_source="equation",
                extraction_model=extraction_model,
                data_provenance="equation_derived",
                model_expression=model_text[:500],
                model_r_squared=r2,
            ))
            report.points_kept += 1

    logger.info(f"  Equation evaluation: {report.summary()}")
    if report.points_implausible or report.models_unparsed:
        for msg in report.messages[:10]:
            logger.warning(f"    {msg}")
    return records, report


def _parse_response(
    data: dict,
    paper_doi: str,
    model: str,
    paper_meta: PaperMetadata | None = None,
    paper_id: str = "",
    provenance: str = "measured_table",
    report_sink: list | None = None,
    paper_uid: str = "",
    fallback_source_table: str = "",
) -> list[DielectricRecord]:
    """Parse LLM response — handles both direct records and equation coefficients."""
    records = _parse_records(
        data, paper_doi, model, paper_id=paper_id, provenance=provenance
    )

    # Also check for regression equations in the same response
    if isinstance(data, dict) and data.get("equations"):
        eq_records, eq_report = evaluate_equations_with_report(
            data, paper_doi, paper_meta, paper_id=paper_id, model=model
        )
        records.extend(eq_records)
        if report_sink is not None:
            report_sink.append(eq_report)

    # Stamp the content-addressed paper identity on every record. This is
    # what the assembler joins paper metadata on, replacing the mutable DOI
    # string that previously left records with no title, journal or year.
    if paper_uid:
        for r in records:
            r.paper_uid = paper_uid
    if fallback_source_table:
        canonical = fallback_source_table.replace("table_", "Table ").strip()
        for r in records:
            if not r.source_table:
                r.source_table = canonical

    return records


# -- Concurrency control -----------------------------------------------------

# Module-level semaphore — limits concurrent API calls across all tables/papers.
# Initialized lazily in extract_table() from config["default_concurrency"].
_semaphore: asyncio.Semaphore | None = None


def _get_semaphore(config: dict) -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        concurrency = config.get("default_concurrency", 5)
        _semaphore = asyncio.Semaphore(concurrency)
        logger.debug(f"  Created semaphore with concurrency={concurrency}")
    return _semaphore


def reset_semaphore() -> None:
    """Reset module-level semaphore (call between pipeline runs to pick up new config)."""
    global _semaphore
    _semaphore = None


async def _extract_one_chunk(
    paper: ParsedPaper,
    table: ParsedTable,
    chunk_rows: list[list[str]],
    chunk_num: int,
    total_chunks: int,
    system_prompt: str,
    model: str,
    max_tokens: int,
    config: dict,
) -> tuple[list[DielectricRecord], CostEntry]:
    """Extract records from a single chunk (runs under semaphore)."""
    sem = _get_semaphore(config)
    condition_desc = _describe_chunk(chunk_rows, table)

    # Chunk 1 gets full context; subsequent chunks get lightweight prompt
    if chunk_num == 1:
        prompt = _build_chunk_prompt(
            paper, table, chunk_rows, chunk_num, total_chunks, condition_desc,
        )
    else:
        prompt = _build_chunk_prompt_lite(
            paper, table, chunk_rows, chunk_num, total_chunks, condition_desc,
        )

    async with sem:
        response_text, cost_entry = await call_llm(
            prompt=prompt, system=system_prompt,
            model=model, max_tokens=max_tokens, config=config,
        )

    data = parse_json_safe(strip_code_fences(response_text))
    response_complete = bool(data.pop("_response_complete", True))
    paper_id = Path(paper.pdf_path).stem if paper.pdf_path else ""
    records = _parse_response(data, paper.doi, model, paper.metadata, paper_id=paper_id,
                             paper_uid=paper.paper_uid,
                             fallback_source_table=table.table_id)
    if not response_complete:
        raise IncompleteTableResponse(
            f"chunk {chunk_num}/{total_chunks} returned incomplete JSON",
            records,
            cost_entry,
        )
    logger.info(f"    Chunk {chunk_num}/{total_chunks}: {len(chunk_rows)} rows -> {len(records)} records")
    return records, cost_entry


# -- Extract single table ----------------------------------------------------

def _header_has_gaps(table: ParsedTable) -> bool:
    """True when the parsed header looks structurally damaged.

    Docling flattens multi-level headers. A table whose columns are labelled
    by a sub-header row -- five temperature columns under one "T (C)" span,
    say -- comes back with those labels replaced by empty strings:

        ['Moisture (%)', '', 'Frequency (MHz)', '', 'Temperature (C)', '']

    The extractor then has no way to attach a temperature to a value, and
    returns records that look complete but carry no conditions. Coverage
    cannot detect this, because the model still produces one record per row.
    """
    for row in table.headers or []:
        cells = [str(c).strip() for c in row]
        if len(cells) >= 4 and sum(1 for c in cells if not c) >= 2:
            return True
    return False


def _conditions_missing(table: ParsedTable, records: list) -> bool:
    """True when the header advertises a condition the records do not carry.

    A table captioned with a temperature range whose extracted records all
    have temperature None has been misread, however many records it yielded.
    """
    if not records:
        return False
    header_text = " ".join(
        str(c) for row in (table.headers or []) for c in row
    ).lower()
    text = f"{table.caption} {header_text}".lower()

    checks = (
        (("t (", "\u00b0c", "temperature"), "temperature_c"),
        (("m (%", "moisture", "w.b.", "wet basis"), "moisture_content_pct"),
        (("mhz", "ghz", "frequency"), "frequency_mhz"),
    )
    for keywords, field in checks:
        if not any(k in text for k in keywords):
            continue
        missing = sum(1 for r in records if getattr(r, field, None) is None)
        if missing / len(records) > 0.5:
            logger.info(
                f"    {table.table_id}: header mentions {field} but "
                f"{missing}/{len(records)} records lack it"
            )
            return True
    return False


def _usable_record_count(records: list[DielectricRecord]) -> int:
    """Count records carrying at least one dielectric measurement."""
    return sum(
        1 for record in records
        if record.dielectric_constant is not None or record.loss_factor is not None
    )


async def _apply_vision_fallback(
    table: ParsedTable,
    paper: ParsedPaper,
    config: dict,
    paper_id: str,
    records: list[DielectricRecord],
    total_cost: CostEntry,
    *,
    equation_mode: bool = False,
    table_label: str = "",
    status_fn=None,
) -> tuple[list[DielectricRecord], CostEntry]:
    """Apply the same vision-recovery policy in realtime and batch modes."""
    vision_cfg = config.get("vision_fallback", {})
    if not vision_cfg.get("enabled", True):
        return records, total_cost

    rows = table.rows or [list(row.values()) for row in table.data_rows]
    row_count = len(rows)
    if equation_mode:
        should_retry = True
    else:
        complete = sum(
            1 for record in records
            if record.dielectric_constant is not None
            and record.loss_factor is not None
        )
        coverage = complete / row_count if row_count else 1.0
        min_rows = vision_cfg.get("min_rows", 3)
        poor_coverage = coverage < vision_cfg.get("coverage_threshold", 0.5)
        damaged_header = _header_has_gaps(table)
        lost_conditions = _conditions_missing(table, records)
        should_retry = (
            row_count >= min_rows
            and (poor_coverage or not records or damaged_header or lost_conditions)
        )
        if should_retry and not poor_coverage and records:
            logger.info(
                f"    {table.table_id}: text parse suspect "
                f"(damaged_header={damaged_header}, "
                f"lost_conditions={lost_conditions}) — trying vision"
            )
        if 0 < row_count < min_rows:
            logger.debug(
                f"    Skipping vision fallback on {table.table_id}: "
                f"only {row_count} rows, not a data table"
            )

    if not should_retry:
        return records, total_cost

    vision_records, vision_cost = await _extract_table_via_vision(
        table,
        paper,
        config,
        paper_id,
        table_label,
        status_fn,
        equation_mode=equation_mode,
    )
    if vision_cost is not None:
        total_cost.input_tokens += vision_cost.input_tokens
        total_cost.output_tokens += vision_cost.output_tokens
        total_cost.cost_usd += vision_cost.cost_usd

    text_score = len(records) if equation_mode else _usable_record_count(records)
    vision_score = (
        len(vision_records)
        if equation_mode else _usable_record_count(vision_records)
    )
    if vision_score > text_score:
        kind = "Equation vision" if equation_mode else "Vision fallback"
        logger.info(
            f"    {kind} on {table.table_id}: "
            f"{vision_score} usable records vs {text_score} from text — "
            f"using vision result"
        )
        return vision_records, total_cost
    if vision_records:
        logger.info(
            f"    Vision fallback on {table.table_id} did not improve "
            f"({vision_score} vs {text_score}) — keeping text result"
        )
    return records, total_cost


async def extract_table(
    table: ParsedTable,
    paper: ParsedPaper,
    config: dict,
    status_fn=None,
    table_label: str = "",
    running_total: int = 0,
) -> tuple[list[DielectricRecord], CostEntry]:
    """Extract all records from one table.

    Chunks are processed in parallel (bounded by semaphore).
    No retry — with Haiku + proper chunking, first pass is reliable enough.

    Returns (records, aggregated_cost_entry).
    """
    from src.schema import CostEntry

    # Route: equation tables get a dedicated prompt, no chunking
    is_equation = _is_equation_table(table)
    config_key = "equation_extractor" if is_equation else "table_extractor"
    model_cfg = config.get(config_key, config.get("table_extractor", {}))
    model = model_cfg.get("model", "claude-haiku-4-5-20251001")
    max_tokens = model_cfg.get("max_tokens", 4096)
    if is_equation:
        system_prompt = load_skill("extractor_equation")
        logger.info(f"    {table.table_id}: detected as equation/correlation table")
    else:
        system_prompt = load_skill("extractor_table")

    rows = table.rows or [list(r.values()) for r in table.data_rows]
    paper_id = Path(paper.pdf_path).stem if paper.pdf_path else ""
    total_cost = CostEntry(stage="extract_table", model=model, doi=paper.doi)

    if not rows:
        return [], total_cost

    # Equation tables: always single call, no chunking
    if is_equation:
        if status_fn:
            status_fn(f"{table_label} | extracting equations")
        prompt = _build_single_table_prompt(paper, table)
        sem = _get_semaphore(config)
        async with sem:
            response_text, cost_entry = await call_llm(
                prompt=prompt, system=system_prompt,
                model=model, max_tokens=max_tokens, config=config,
            )
        total_cost.input_tokens += cost_entry.input_tokens
        total_cost.output_tokens += cost_entry.output_tokens
        total_cost.cost_usd += cost_entry.cost_usd
        data = parse_json_safe(strip_code_fences(response_text))
        response_complete = bool(data.pop("_response_complete", True))
        all_records = _parse_response(data, paper.doi, model, paper.metadata, paper_id=paper_id,
                             paper_uid=paper.paper_uid,
                             fallback_source_table=table.table_id)
        final_records, final_cost = await _apply_vision_fallback(
            table,
            paper,
            config,
            paper_id,
            all_records,
            total_cost,
            equation_mode=True,
            table_label=table_label,
            status_fn=status_fn,
        )
        if not response_complete and not any(
            record.data_provenance == "vision_table" for record in final_records
        ):
            raise IncompleteTableResponse(
                "equation response returned incomplete JSON",
                final_records,
                final_cost,
            )
        return final_records, final_cost

    # Measurement tables: chunk and extract via text
    chunks = chunk_table(table, max_rows=12)
    total_chunks = len(chunks)

    all_records = None
    if True:
        if total_chunks == 1:
            # Single chunk — no parallelism needed
            if status_fn:
                status_fn(f"{table_label} | extracting {len(rows)} rows")
            prompt = _build_single_table_prompt(paper, table)
            sem = _get_semaphore(config)
            async with sem:
                response_text, cost_entry = await call_llm(
                    prompt=prompt, system=system_prompt,
                    model=model, max_tokens=max_tokens, config=config,
                )
            total_cost.input_tokens += cost_entry.input_tokens
            total_cost.output_tokens += cost_entry.output_tokens
            total_cost.cost_usd += cost_entry.cost_usd
            data = parse_json_safe(strip_code_fences(response_text))
            response_complete = bool(data.pop("_response_complete", True))
            all_records = _parse_response(data, paper.doi, model, paper.metadata, paper_id=paper_id,
                             paper_uid=paper.paper_uid,
                             fallback_source_table=table.table_id)
            if not response_complete:
                # Let the vision fallback below replace a truncated text read
                # when possible; otherwise propagate the partial rows.
                text_response_incomplete = True
            else:
                text_response_incomplete = False
        else:
            # Multiple chunks — process in parallel
            logger.info(f"    Splitting {table.table_id} into {total_chunks} chunks ({len(rows)} rows total)")
            if status_fn:
                status_fn(f"{table_label} | {total_chunks} chunks in parallel")

            tasks = [
                _extract_one_chunk(
                    paper, table, chunk_rows, chunk_num, total_chunks,
                    system_prompt, model, max_tokens, config,
                )
                for chunk_num, chunk_rows in enumerate(chunks, 1)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_records = []
            chunk_failures = []
            for result in results:
                if isinstance(result, IncompleteTableResponse):
                    all_records.extend(result.records)
                    cost_entry = result.cost
                    chunk_failures.append(str(result))
                elif isinstance(result, Exception):
                    chunk_failures.append(
                        f"{type(result).__name__}: {str(result)[:500]}"
                    )
                    continue
                else:
                    records, cost_entry = result
                    all_records.extend(records)
                if cost_entry is None:
                    continue
                total_cost.input_tokens += cost_entry.input_tokens
                total_cost.output_tokens += cost_entry.output_tokens
                total_cost.cost_usd += cost_entry.cost_usd
            text_response_incomplete = bool(chunk_failures)

    # Coverage check
    complete = sum(1 for r in all_records
                   if r.dielectric_constant is not None and r.loss_factor is not None)
    row_count = len(rows)
    coverage = complete / row_count if row_count > 0 else 1.0

    if coverage < 0.80 and row_count > 5:
        logger.warning(
            f"    Low coverage for {table.table_id}: "
            f"{complete} complete/{row_count} rows ({coverage:.0%})."
        )

    final_records, final_cost = await _apply_vision_fallback(
        table,
        paper,
        config,
        paper_id,
        all_records,
        total_cost,
        table_label=table_label,
        status_fn=status_fn,
    )
    if text_response_incomplete and not any(
        record.data_provenance == "vision_table" for record in final_records
    ):
        reason = (
            "; ".join(chunk_failures)
            if total_chunks > 1 else "table response returned incomplete JSON"
        )
        raise IncompleteTableResponse(reason, final_records, final_cost)
    return final_records, final_cost


async def _extract_table_via_vision(
    table: ParsedTable,
    paper: ParsedPaper,
    config: dict,
    paper_id: str,
    table_label: str = "",
    status_fn=None,
    equation_mode: bool = False,
) -> tuple[list[DielectricRecord], CostEntry | None]:
    """Re-read one table from a rendered page image using a vision model.

    Returns ([], None) when the page cannot be rendered or no vision model is
    configured, so the caller simply keeps the text-based result.
    """
    vision_cfg = config.get("vision_fallback", {})
    model = vision_cfg.get("model") or config.get("vision_model")
    if not model:
        logger.debug("    Vision fallback requested but no vision model configured")
        return [], None

    if not paper.pdf_path:
        return [], None

    image_path = _render_table_page(
        paper.pdf_path,
        table,
        Path(config.get("output_dir", "data")) / "parsed" / "table_images",
    )
    if not image_path:
        logger.info(
            f"    Vision fallback: could not locate the page for {table.table_id}"
        )
        return [], None

    if status_fn:
        status_fn(f"{table_label} | vision fallback")

    system_prompt = load_skill(
        "extractor_equation" if equation_mode else "extractor_table"
    )
    prompt = (
        _build_equation_vision_prompt(paper, table)
        if equation_mode else _build_vision_prompt(paper, table)
    )
    max_tokens = vision_cfg.get("max_tokens", 8192)

    sem = _get_semaphore(config)
    try:
        async with sem:
            response_text, cost_entry = await call_llm(
                prompt=prompt, system=system_prompt, model=model,
                max_tokens=max_tokens, config=config,
                images=[{"path": image_path, "media_type": "image/png"}],
            )
    # Vision is a best-effort fallback; SDK/backend exceptions must preserve
    # the already extracted text records.
    except Exception as e:  # noqa: BLE001
        logger.warning(f"    Vision fallback failed for {table.table_id}: {e}")
        return [], None

    data = parse_json_safe(strip_code_fences(response_text))
    if not data.pop("_response_complete", True):
        logger.warning(
            "    Vision response for %s was incomplete; ignoring it",
            table.table_id,
        )
        return [], cost_entry
    records = _parse_response(
        data, paper.doi, model, paper.metadata, paper_id=paper_id,
        provenance="vision_table", paper_uid=paper.paper_uid,
    )
    # The rendered page may show more than one table, so the model sometimes
    # labels its output with a neighbouring table's number. Provenance must
    # name the table actually being re-read, or a reader cannot trace the
    # value back to its source.
    canonical = table.table_id.replace("table_", "Table ").strip()
    for r in records:
        if equation_mode:
            r.source_table = r.source_table or f"{canonical} (vision)"
        else:
            r.source_table = f"{canonical} (vision)"
    logger.info(
        f"    Vision fallback on {table.table_id} ({Path(image_path).name}): "
        f"{len(records)} records"
    )
    return records, cost_entry


def _build_equation_vision_prompt(paper: ParsedPaper, table: ParsedTable) -> str:
    """Prompt for jointly reading a printed equation and coefficient table."""
    freqs = ", ".join(str(f) for f in paper.metadata.measurement_frequencies_mhz)
    materials = ", ".join(paper.metadata.primary_materials or [])
    return (
        "The attached image is a page from a scientific paper containing a "
        "regression equation, its variable definitions or domains, and a "
        "coefficient table. Read them together.\n\n"
        f"Paper: {paper.metadata.title}\n"
        f"Materials studied: {materials or 'see the page'}\n"
        f"Measurement frequencies (MHz): {freqs or 'see the page'}\n"
        f"Target table: {table.table_id.replace('_', ' ').title()}\n\n"
        "Recover the complete model separately for dielectric constant and "
        "loss factor at every frequency represented. Map each coefficient to "
        "the exact term defined by the printed equation, preserve every sign "
        "and power of ten, and include the fitted variable domains. Do not "
        "treat coefficients a, b, c, ... as measured dielectric values. If "
        "the equation or a coefficient is unreadable, omit that model rather "
        "than guess. Return the equation JSON format required by the system "
        "instructions."
    )


def _build_vision_prompt(paper: ParsedPaper, table: ParsedTable) -> str:
    """Prompt for reading a table from a rendered page image."""
    freqs = ", ".join(str(f) for f in paper.metadata.measurement_frequencies_mhz)
    materials = ", ".join(paper.metadata.primary_materials or [])
    return (
        f"The attached image is a page from a scientific paper.\n\n"
        f"Paper: {paper.metadata.title}\n"
        f"Materials studied: {materials or 'see the page'}\n"
        f"Measurement frequencies (MHz): {freqs or 'see the page'}\n\n"
        f"Read {table.table_id.replace('_', ' ').title()} from the image and "
        f"extract every dielectric property measurement it contains.\n"
        f"Caption as parsed: {table.caption or '(not captured)'}\n\n"
        f"The text-based parse of this table was incomplete, which is why you "
        f"are being shown the page. Read the printed table directly. Pay "
        f"attention to multi-level headers, merged cells, and rows where a "
        f"material name serves as a section divider rather than repeating on "
        f"every line.\n\n"
        f"Extract ONLY values printed in this table. Do not infer, "
        f"interpolate, or carry values over from other tables or from the "
        f"surrounding text."
    )


# -- Source discrimination filter ---------------------------------------------

def filter_cited_values(
    records: list[DielectricRecord],
    paper_meta: PaperMetadata,
) -> list[DielectricRecord]:
    """Remove records that are cited from other papers, not original measurements.

    Remove if:
    - frequency_mhz not in paper_meta.measurement_frequencies_mhz
    - material_name not in paper_meta.primary_materials (fuzzy match)
    - dielectric_constant is None AND loss_factor is None
    - Either dielectric_constant or loss_factor is None (incomplete)
    """
    if not records:
        return records

    paper_freqs = set(paper_meta.measurement_frequencies_mhz)
    # Common ISM frequencies are useful aliases only when the screener
    # actually reported a frequency domain. An empty set disables filtering.
    _COMMON_ISM = {915.0, 2450.0}
    if paper_freqs:
        paper_freqs = paper_freqs | _COMMON_ISM
    primary_mats = [m.lower() for m in paper_meta.primary_materials]

    if not paper_meta.measurement_frequencies_mhz:
        logger.warning("  No measurement frequencies from screener — frequency filter DISABLED")
    if not primary_mats:
        logger.warning("  No primary materials from screener — material filter DISABLED")

    filtered = []
    for r in records:
        # Must have at least one value
        if r.dielectric_constant is None and r.loss_factor is None:
            logger.debug(f"  Filtered (no values): {r.material_name}")
            continue

        # Must have a source table
        if not r.source_table:
            logger.debug(f"  Filtered (no source_table): {r.material_name}")
            continue

        # Skip source discrimination for equation-evaluated records
        # (they are computed from the paper's own regression equations, not cited)
        if "regression" in (r.source_table or "").lower():
            filtered.append(r)
            continue

        # Check frequency
        if paper_freqs and r.frequency_mhz is not None:
            matched = any(
                abs(r.frequency_mhz - pf) / pf < 0.05
                for pf in paper_freqs
                if pf > 0
            )
            if not matched:
                logger.debug(f"  Filtered (wrong freq): {r.material_name} @ {r.frequency_mhz} MHz")
                continue

        # Check material (fuzzy) — but only if paper has many materials
        # For single-material papers, the LLM may hallucinate wrong names
        # for chunks missing context; trust the screener instead and rename.
        if primary_mats:
            r_mat = r.material_name.lower()
            material_ok = any(
                pm in r_mat or r_mat in pm or pm.split()[0] == r_mat.split()[0]
                for pm in primary_mats
            )
            if not material_ok:
                # If all primary materials are variants of the same thing,
                # rename the record instead of filtering it out
                if len(primary_mats) <= 2:
                    logger.info(f"  Renamed material '{r.material_name}' -> '{paper_meta.primary_materials[0]}'")
                    r.material_name = paper_meta.primary_materials[0]
                else:
                    logger.debug(f"  Filtered (wrong material): {r.material_name}")
                    continue

        filtered.append(r)

    removed = len(records) - len(filtered)
    if removed > 0:
        logger.info(f"  Source filter removed {removed} records (kept {len(filtered)})")

    return filtered


def merge_split_records(records: list[DielectricRecord]) -> list[DielectricRecord]:
    """Merge split records where ε' and ε'' are in separate rows.

    Some tables report ε' and ε'' in separate rows. This function combines them into
    single records by matching on (material, frequency, temperature, moisture, salt).
    """
    if not records:
        return records

    from collections import defaultdict

    # Group by matching key
    groups: dict[tuple, list[DielectricRecord]] = defaultdict(list)
    for r in records:
        key = (
            r.material_name.lower().strip(),
            round(r.frequency_mhz, 1) if r.frequency_mhz is not None else None,
            round(r.temperature_c, 1) if r.temperature_c is not None else None,
            (
                round(r.moisture_content_pct, 1)
                if r.moisture_content_pct is not None else None
            ),
            (r.moisture_basis or "unknown").strip().lower()
            .replace(" basis", ""),
            (r.salt_content or "").strip().lower(),
            (
                round(r.electrical_conductivity_s_m, 4)
                if r.electrical_conductivity_s_m is not None else None
            ),
            (r.data_provenance or "").strip().lower(),
        )
        groups[key].append(r)

    merged = []
    for key, group in groups.items():
        if len(group) == 1:
            merged.append(group[0])
            continue

        # Find records with complementary dc/lf
        complete_records = [
            r for r in group
            if r.dielectric_constant is not None and r.loss_factor is not None
        ]
        dc_records = [
            r for r in group
            if r.dielectric_constant is not None and r.loss_factor is None
        ]
        lf_records = [
            r for r in group
            if r.loss_factor is not None and r.dielectric_constant is None
        ]

        if complete_records:
            # Preserve genuine replicate or independently reported complete
            # values. Exact duplicates are handled deterministically during
            # assembly; collapsing here destroys disagreement information.
            merged.extend(group)
        elif len(dc_records) == 1 and len(lf_records) == 1:
            # Merge only when the pairing is unambiguous.
            base = dc_records[0].model_copy()
            loss_record = lf_records[0]
            base.loss_factor = loss_record.loss_factor
            if base.loss_tangent is None:
                base.loss_tangent = loss_record.loss_tangent
            for field in (
                "source_table", "source_location", "model_expression",
                "data_provenance", "extraction_source", "extraction_model",
            ):
                values = []
                for value in (getattr(base, field), getattr(loss_record, field)):
                    text = str(value).strip() if value is not None else ""
                    if text and text not in values:
                        values.append(text)
                setattr(base, field, "; ".join(values) if values else None)
            if base.model_r_squared is None:
                base.model_r_squared = loss_record.model_r_squared
            merged.append(base)
            logger.debug(f"  Merged split record: {base.material_name} @ {base.frequency_mhz} MHz, {base.temperature_c}C")
        else:
            # Multiple candidates cannot be paired without inventing a
            # relationship. Preserve them for downstream audit/deduplication.
            merged.extend(group)

    n_merged = len(records) - len(merged)
    if n_merged > 0:
        logger.info(f"  Merged {n_merged} split records ({len(records)} -> {len(merged)})")
    return merged


# -- Main run ----------------------------------------------------------------

async def extract_all_tables(
    paper: ParsedPaper,
    config: dict,
    status_fn=None,
    sink: dict | None = None,
) -> tuple[list[DielectricRecord], CostEntry]:
    """Loop over ALL tables in the paper. Extract from every data table.
    Skip non-data tables. Log everything.

    ``sink`` receives records and cost as each table completes, so a caller
    that abandons this coroutine on timeout can still keep whatever finished.
    Without it, a slow final table discards every record already extracted.

    Returns (records, aggregated_cost_entry).
    """
    from src.schema import CostEntry
    _enrich_condition_metadata_from_source(paper)
    model = config.get("table_extractor", {}).get("model", "claude-haiku-4-5-20251001")
    all_records: list[DielectricRecord] = []
    paper_cost = CostEntry(stage="extract_table", model=model, doi=paper.doi)
    if sink is not None:
        # A separate accumulator: the normal return path does its own cost
        # aggregation, so the sink must not share that object.
        sink.setdefault("records", [])
        sink["cost"] = CostEntry(
            stage="extract_table", model=model, doi=paper.doi
        )

    data_tables = [t for t in paper.tables if _is_dielectric_table(t, paper.metadata)]
    n_data = len(data_tables)

    # Keep equation tables even when another table is classified as a
    # measurement table. That other table often contains only sample ranges,
    # fit statistics, or penetration depths; dropping the equations therefore
    # drops every usable dielectric value. Provenance and downstream
    # deduplication distinguish equation-derived records.

    # Extract all data tables concurrently (semaphore limits API concurrency)
    async def _extract_one_table(table, table_idx):
        rows = table.rows or [list(r.values()) for r in table.data_rows]
        if status_fn:
            status_fn(f"Table {table_idx}/{n_data} ({table.table_id}) | {len(rows)} rows")
        logger.info(f"  Processing {table.table_id}: {table.caption[:80]}... ({len(rows)} rows)")
        try:
            recs, cost = await extract_table(
                table, paper, config, status_fn=status_fn,
                table_label=f"Table {table_idx}/{n_data} ({table.table_id})",
                running_total=0,
            )
        except IncompleteTableResponse as exc:
            recs, cost = exc.records, exc.cost
            if sink is not None:
                sink["records"].extend(recs)
                if cost is not None:
                    sc = sink["cost"]
                    sc.input_tokens += cost.input_tokens
                    sc.output_tokens += cost.output_tokens
                    sc.cost_usd += cost.cost_usd
            raise
        # Publish as soon as this table is done, so a timeout on a later
        # table cannot throw this one away.
        if sink is not None:
            sink["records"].extend(recs)
            if cost is not None:
                sc = sink["cost"]
                sc.input_tokens += cost.input_tokens
                sc.output_tokens += cost.output_tokens
                sc.cost_usd += cost.cost_usd
        return recs, cost

    tasks = []
    for table in paper.tables:
        if table not in data_tables:
            if not _is_dielectric_table(table, paper.metadata):
                logger.info(f"  Skipping {table.table_id} (not a dielectric data table): {table.caption[:80]}")
            continue
        table_idx = data_tables.index(table) + 1
        tasks.append(_extract_one_table(table, table_idx))

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        failures = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"  Table extraction failed: {result}")
                failures.append(
                    f"{type(result).__name__}: {str(result)[:500]}"
                )
                continue
            table_records, table_cost = result
            all_records.extend(table_records)
            paper_cost.input_tokens += table_cost.input_tokens
            paper_cost.output_tokens += table_cost.output_tokens
            paper_cost.cost_usd += table_cost.cost_usd
        if failures:
            if sink is not None:
                sink["failures"] = failures
            raise RuntimeError(
                f"{len(failures)} table extraction(s) failed: "
                + "; ".join(failures)
            )

    return all_records, paper_cost


def _post_process_paper_records(
    all_records: list[DielectricRecord],
    paper: ParsedPaper,
) -> list[DielectricRecord]:
    """Apply source filter, sanity filter, and merge split records for one paper."""
    # Source discrimination
    all_records = filter_cited_values(all_records, paper.metadata)

    # Physical sanity filter
    before = len(all_records)
    sane_records = []
    for r in all_records:
        dc = r.dielectric_constant
        lf = r.loss_factor
        if dc is not None and (dc < 0.1 or dc > 120):
            logger.debug(f"  Filtered (ε'={dc} out of range): {r.material_name} @ {r.frequency_mhz} MHz")
            continue
        if lf is not None and (lf < 0 or lf > 2000):
            logger.debug(f"  Filtered (ε''={lf} out of range): {r.material_name} @ {r.frequency_mhz} MHz")
            continue
        sane_records.append(r)
    if len(sane_records) < before:
        logger.info(f"  Physical sanity filter removed {before - len(sane_records)} records")

    # Merge split records
    return merge_split_records(sane_records)


def _save_paper_result(
    paper: ParsedPaper,
    all_records: list[DielectricRecord],
    results: list[ExtractionResult],
    output_dir: Path,
    checkpoint_db,
    paper_cost,
    progress,
    label: str,
    timed_out: bool = False,
    complete: bool = True,
    incomplete_reason: str = "",
) -> None:
    """Save extraction result for one paper."""
    tables_processed = [t.table_id for t in paper.tables if _is_dielectric_table(t, paper.metadata)]
    tables_skipped = [t.table_id for t in paper.tables if not _is_dielectric_table(t, paper.metadata)]

    er = ExtractionResult(
        paper_uid=paper.paper_uid,
        complete=complete and not timed_out,
        incomplete_reason=(
            incomplete_reason
            or ("Paper extraction timed out" if timed_out else "")
        ),
        doi=paper.doi,
        records=all_records,
        extraction_source="table",
        tables_processed=tables_processed,
        tables_skipped=tables_skipped,
        extraction_log=[f"Extracted from {len(paper.tables)} tables"],
        timed_out=timed_out,
        notes=(
            f"{len(all_records)} records from {len(tables_processed)} data tables"
            + (
                f" (INCOMPLETE: {incomplete_reason})"
                if incomplete_reason else
                " (INCOMPLETE: paper timed out before all tables were processed)"
                if timed_out else ""
            )
        ),
    )
    results.append(er)

    # Artifacts are named by the content-addressed paper uid so that a
    # paper's outputs cannot be split across several filenames, and so that
    # results from an unrelated corpus cannot be picked up during assembly.
    base = artifact_name(paper.paper_uid, Path(paper.pdf_path).stem
                         if paper.pdf_path else paper.doi)
    out_file = output_dir / f"{base}_table.json"
    write_json_atomic(out_file, er.model_dump(mode="json"))

    if checkpoint_db and paper_cost:
        checkpoint_db.add_cost(paper_cost)

    if progress:
        progress.advance_paper(f"{label} ({len(all_records)} records)")
    logger.info(f"  {paper.doi}: {len(all_records)} total table records")


async def _run_batch_tables(
    paired: list[tuple[ParsedPaper, ScreenerResult]],
    config: dict,
    output_dir: Path,
    checkpoint_db,
    progress,
    results: list[ExtractionResult],
) -> None:
    """Batch mode: collect all table chunk prompts, submit one batch, route results back."""
    from src.schema import CostEntry

    model_cfg = config.get("table_extractor", {})
    model = model_cfg.get("model", "claude-haiku-4-5-20251001")
    max_tokens = model_cfg.get("max_tokens", 4096)

    # Phase 1: Build all prompts (no API calls)
    # Each entry: (custom_id, prompt, system_prompt, paper, table_idx, chunk_idx)
    batch_requests = []
    # Track which papers/tables each request belongs to
    request_registry: dict[str, dict] = {}
    paper_records: dict[str, list[DielectricRecord]] = {}
    table_records: dict[tuple[str, int], list[DielectricRecord]] = {}
    paper_costs: dict[str, CostEntry] = {}
    paper_models: dict[str, set[str]] = {}
    paper_objects: dict[str, ParsedPaper] = {}
    paper_tables: dict[str, list[tuple[int, ParsedTable]]] = {}
    paper_failures: dict[str, list[str]] = {}

    for paper_idx, (paper, screener) in enumerate(paired):
        label = paper.doi.split("/")[-1]
        if not paper.tables:
            logger.info(f"  {paper.doi}: no tables, skipping table extraction")
            _save_paper_result(
                paper,
                [],
                results,
                output_dir,
                checkpoint_db,
                CostEntry(stage="extract_table", model=model, doi=paper.doi),
                progress,
                label,
            )
            continue

        if progress:
            progress.make_status_fn(label)("building prompts for batch...")

        paper_id = Path(paper.pdf_path).stem if paper.pdf_path else ""
        data_tables = [t for t in paper.tables if _is_dielectric_table(t, paper.metadata)]
        if not data_tables:
            logger.info(f"  {paper.doi}: no dielectric tables selected")
            _save_paper_result(
                paper,
                [],
                results,
                output_dir,
                checkpoint_db,
                CostEntry(stage="extract_table", model=model, doi=paper.doi),
                progress,
                label,
            )
            continue

        paper_key = paper.paper_uid or paper.doi
        paper_objects[paper_key] = paper
        paper_records[paper_key] = []
        paper_tables[paper_key] = []
        paper_costs[paper_key] = CostEntry(
            stage="extract_table", model=model, doi=paper.doi
        )
        paper_models[paper_key] = set()
        paper_failures[paper_key] = []

        for table_idx, table in enumerate(data_tables):
            rows = table.rows or [list(r.values()) for r in table.data_rows]
            if not rows:
                continue

            table_key = (paper_key, table_idx)
            table_records[table_key] = []
            paper_tables[paper_key].append((table_idx, table))

            is_equation = _is_equation_table(table)
            system_prompt = load_skill("extractor_equation") if is_equation else load_skill("extractor_table")
            request_cfg = config.get(
                "equation_extractor" if is_equation else "table_extractor",
                model_cfg,
            )
            request_model = request_cfg.get("model", model)
            request_max_tokens = request_cfg.get("max_tokens", max_tokens)
            paper_models[paper_key].add(request_model)

            if is_equation or len(rows) <= 12:
                # Single prompt for this table
                prompt = _build_single_table_prompt(paper, table)
                cid = f"tbl_{label}_p{paper_idx}_{table_idx}_0"
                content = [{"type": "text", "text": prompt}]
                batch_requests.append({
                    "custom_id": cid,
                    "model": request_model,
                    "max_tokens": request_max_tokens,
                    "temperature": 0.0,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": content}],
                })
                request_registry[cid] = {
                    "paper": paper, "table": table, "paper_id": paper_id,
                    "label": label, "chunk_num": 0, "total_chunks": 1,
                    "model": request_model, "table_key": table_key,
                }
            else:
                # Chunk the table
                chunks = chunk_table(table, max_rows=12)
                total_chunks = len(chunks)
                for chunk_idx, chunk_rows in enumerate(chunks):
                    condition_desc = _describe_chunk(chunk_rows, table)
                    if chunk_idx == 0:
                        prompt = _build_chunk_prompt(
                            paper, table, chunk_rows, chunk_idx + 1, total_chunks, condition_desc,
                        )
                    else:
                        prompt = _build_chunk_prompt_lite(
                            paper, table, chunk_rows, chunk_idx + 1, total_chunks, condition_desc,
                        )
                    cid = f"tbl_{label}_p{paper_idx}_{table_idx}_{chunk_idx}"
                    content = [{"type": "text", "text": prompt}]
                    batch_requests.append({
                        "custom_id": cid,
                        "model": request_model,
                        "max_tokens": request_max_tokens,
                        "temperature": 0.0,
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": content}],
                    })
                    request_registry[cid] = {
                        "paper": paper, "table": table, "paper_id": paper_id,
                        "label": label, "chunk_num": chunk_idx, "total_chunks": total_chunks,
                        "model": request_model, "table_key": table_key,
                    }

    if not batch_requests:
        # Papers whose selected tables contain no machine-readable rows still
        # need a durable zero-record artifact for coverage and resume logic.
        for paper_key, paper in paper_objects.items():
            label = paper.doi.split("/")[-1]
            _save_paper_result(
                paper,
                [],
                results,
                output_dir,
                checkpoint_db,
                paper_costs.get(paper_key),
                progress,
                label,
            )
        return

    # Phase 2: Submit batch and poll
    logger.info(f"Submitting table extraction batch with {len(batch_requests)} requests...")
    batch_results = await run_batch(batch_requests, config=config, poll_interval=30.0)

    # Phase 3: Route results back to papers and tables.

    for cid, info in request_registry.items():
        paper = info["paper"]
        paper_key = paper.paper_uid or paper.doi
        result_model = info["model"]
        paper_objects[paper_key] = paper

        result_pair = batch_results.get(cid)
        if not result_pair:
            logger.warning(f"  Missing batch result for {cid}")
            paper_failures[paper_key].append(f"missing result for {cid}")
            continue

        response_text, cost_entry = result_pair
        paper_costs[paper_key].input_tokens += cost_entry.input_tokens
        paper_costs[paper_key].output_tokens += cost_entry.output_tokens
        paper_costs[paper_key].cost_usd += cost_entry.cost_usd

        if not response_text:
            paper_failures[paper_key].append(f"empty result for {cid}")
            continue

        data = parse_json_safe(strip_code_fences(response_text))
        if not data.pop("_response_complete", True):
            paper_failures[paper_key].append(f"incomplete JSON for {cid}")
        records = _parse_response(data, paper.doi, result_model, paper.metadata,
                                  paper_id=info["paper_id"],
                                  paper_uid=paper.paper_uid,
                                  fallback_source_table=info["table"].table_id)
        table_records[info["table_key"]].extend(records)

    # Phase 4: Apply the same vision-recovery policy used by realtime mode,
    # then post-process and save each paper.
    for paper_key, tables in paper_tables.items():
        paper = paper_objects[paper_key]
        label = paper.doi.split("/")[-1]
        if paper_models[paper_key]:
            paper_costs[paper_key].model = ";".join(
                sorted(paper_models[paper_key])
            )
        paper_id = Path(paper.pdf_path).stem if paper.pdf_path else ""
        for table_idx, table in tables:
            records = table_records[(paper_key, table_idx)]
            records, _ = await _apply_vision_fallback(
                table,
                paper,
                config,
                paper_id,
                records,
                paper_costs[paper_key],
                equation_mode=_is_equation_table(table),
                table_label=table.table_id,
            )
            paper_records[paper_key].extend(records)
        records = paper_records[paper_key]
        records = _post_process_paper_records(records, paper)
        failures = paper_failures[paper_key]
        _save_paper_result(
            paper, records, results, output_dir,
            checkpoint_db, paper_costs.get(paper_key), progress, label,
            complete=not failures,
            incomplete_reason="; ".join(failures),
        )


async def run(
    paired: list[tuple[ParsedPaper, ScreenerResult]],
    config: dict,
    checkpoint_db=None,
    progress=None,
) -> list[ExtractionResult]:
    """Extract dielectric records from ALL tables in each paper."""
    output_dir = Path(config.get("output_dir", "data")) / "extracted"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = config.get("table_extractor", {}).get("model", "claude-haiku-4-5-20251001")
    use_batch = config.get("use_batch_api", False)

    results: list[ExtractionResult] = []

    if use_batch:
        await _run_batch_tables(paired, config, output_dir, checkpoint_db, progress, results)
        return results

    # ── Real-time mode (original logic) ──
    for paper, screener in paired:
        label = paper.doi.split("/")[-1]
        status_fn = progress.make_status_fn(label) if progress else None

        if not paper.tables:
            logger.info(f"  {paper.doi}: no tables, skipping table extraction")
            _save_paper_result(
                paper,
                [],
                results,
                output_dir,
                checkpoint_db,
                CostEntry(stage="extract_table", model=model, doi=paper.doi),
                progress,
                label,
            )
            continue

        paper_timeout = config.get("paper_timeout_seconds", 900)
        logger.info(f"Table-extracting {paper.doi} ({len(paper.tables)} tables, timeout={paper_timeout}s)...")
        sink: dict = {}
        timed_out = False
        incomplete_reason = ""
        try:
            all_records, paper_cost = await asyncio.wait_for(
                extract_all_tables(paper, config, status_fn=status_fn, sink=sink),
                timeout=paper_timeout,
            )
        except asyncio.TimeoutError:
            # Keep whatever finished. Discarding partial results turns a slow
            # paper into a paper that looks empty, which is indistinguishable
            # from a paper that genuinely has no data.
            timed_out = True
            all_records = list(sink.get("records") or [])
            paper_cost = sink.get("cost") or CostEntry(
                stage="extract_table",
                model=config.get("table_extractor", {}).get(
                    "model", "claude-haiku-4-5-20251001"),
                doi=paper.doi,
            )
            logger.warning(
                f"  {paper.doi}: extraction timed out after {paper_timeout}s. "
                f"Keeping {len(all_records)} records extracted before the "
                f"timeout; remaining tables not processed."
            )
        # This is the per-paper fault boundary: retain completed sibling tables
        # and mark the artifact incomplete for a deterministic retry.
        except Exception as exc:  # noqa: BLE001
            # Keep successful sibling-table results but mark the artifact as
            # incomplete so resume logic retries it and the pipeline exits
            # nonzero instead of silently publishing a partial paper.
            all_records = list(sink.get("records") or [])
            paper_cost = sink.get("cost") or CostEntry(
                stage="extract_table",
                model=config.get("table_extractor", {}).get(
                    "model", "claude-haiku-4-5-20251001"
                ),
                doi=paper.doi,
            )
            incomplete_reason = (
                f"{type(exc).__name__}: {str(exc)[:1000]}"
            )
            logger.error(
                "  %s: incomplete table extraction; keeping %d records: %s",
                paper.doi,
                len(all_records),
                incomplete_reason,
            )

        all_records = _post_process_paper_records(all_records, paper)
        _save_paper_result(
            paper, all_records, results, output_dir,
            checkpoint_db, paper_cost, progress, label,
            timed_out=timed_out,
            complete=not incomplete_reason,
            incomplete_reason=incomplete_reason,
        )

    return results
