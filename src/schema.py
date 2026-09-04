"""Central Pydantic models for the agentXtract dielectric-property pipeline.

Every other module imports from here. This is the single source of truth for all data shapes.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from pydantic import BaseModel, Field

# -- Enums ------------------------------------------------------------------

class PipelineStatus(str, enum.Enum):
    DISCOVERED = "discovered"
    DOWNLOADED = "downloaded"
    PARSED = "parsed"
    SCREENED = "screened"
    EXTRACTED = "extracted"
    ASSEMBLED = "assembled"
    FAILED = "failed"


class ExtractionPriority(str, enum.Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SKIP = "skip"


class Complexity(str, enum.Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class MoistureBasis(str, enum.Enum):
    WET = "wet"
    DRY = "dry"
    UNKNOWN = "unknown"


class ExtractionSource(str, enum.Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    EQUATION = "equation"


class ValidationVerdict(str, enum.Enum):
    ACCEPTED = "accepted"
    FLAGGED = "flagged"
    REJECTED = "rejected"


# -- Dielectric record (atomic unit) ----------------------------------------

class DielectricRecord(BaseModel):
    """Single measurement point. The atomic unit of the database."""
    model_config = {"populate_by_name": True}

    material_name: str
    dielectric_constant: float | None = None   # ε'
    loss_factor: float | None = None            # ε''
    loss_tangent: float | None = None           # tan δ
    frequency_mhz: float | None = None
    temperature_c: float | None = None
    moisture_content_pct: float | None = Field(None, alias="moisture_pct")
    moisture_basis: str | None = None           # "wet" or "dry"
    salt_content: str | None = None
    electrical_conductivity_s_m: float | None = None
    measurement_method: str | None = None
    doi: str = ""
    paper_uid: str = ""                            # content-addressed paper id
    paper_id: str = ""                             # normalized PDF filename stem
    source_table: str = ""                         # e.g., "Table 2"
    source_location: str = ""                      # e.g., "Table 2, row 3"
    extraction_source: str | None = None        # "table", "text", "equation"
    extraction_model: str | None = None         # model used for extraction
    raw_text: str = ""
    # How this value came to exist, for provenance-aware analysis:
    #   measured_table  - read from a table of measurements
    #   measured_text   - read from running text
    #   vision_table    - read from a page image by a vision model
    #   equation_derived- computed from a regression model reported by the paper
    data_provenance: str | None = None
    # For equation-derived records: the model as text, its R^2, and the
    # conditions it was evaluated at. Empty for measured records.
    model_expression: str = ""
    model_r_squared: float | None = None


# -- Database output schema -------------------------------------------------
# Single source of truth for the output database column order and DuckDB types.
# Assembler uses this directly — no separate COLUMN_ORDER to keep in sync.

DATABASE_COLUMNS: list[tuple[str, str, str]] = [
    # (output_column_name, DielectricRecord_field_or_meta, duckdb_type)
    # --- Measurement data ---
    ("material_name",         "material_name",         "VARCHAR"),
    ("frequency_mhz",        "frequency_mhz",         "DOUBLE"),
    ("temperature_c",        "temperature_c",          "DOUBLE"),
    ("dielectric_constant",  "dielectric_constant",    "DOUBLE"),
    ("loss_factor",          "loss_factor",            "DOUBLE"),
    ("loss_tangent",         "loss_tangent",           "DOUBLE"),
    ("moisture_content_pct",         "moisture_content_pct",           "DOUBLE"),
    ("moisture_basis",       "moisture_basis",         "VARCHAR"),
    ("salt_content",         "salt_content",           "VARCHAR"),
    ("electrical_conductivity_s_m", "electrical_conductivity_s_m", "DOUBLE"),
    # --- Provenance ---
    # paper_id is the join key back to the source paper. It is carried through
    # the whole pipeline but used to be dropped at export, which left the
    # published CSV unable to say which paper a row came from — and left
    # evaluation with nothing to join gold records on.
    ("paper_id",             "paper_id",               "VARCHAR"),
    ("paper_uid",            "paper_uid",              "VARCHAR"),
    ("doi",                  "doi",                    "VARCHAR"),
    ("source_table",         "source_table",           "VARCHAR"),
    ("source_location",      "source_location",        "VARCHAR"),
    ("data_provenance",      "data_provenance",        "VARCHAR"),
    ("model_expression",     "model_expression",       "VARCHAR"),
    ("model_r_squared",      "model_r_squared",        "DOUBLE"),
    ("extraction_source",    "extraction_source",      "VARCHAR"),
    ("extraction_model",     "extraction_model",       "VARCHAR"),
    # --- Paper metadata (joined from PaperMetadata) ---
    ("title",                "_meta_title",            "VARCHAR"),
    ("authors",              "_meta_authors",          "VARCHAR"),
    ("year",                 "_meta_year",             "INTEGER"),
    ("journal",              "_meta_journal",          "VARCHAR"),
    ("measurement_method",   "_meta_measurement_method", "VARCHAR"),
]

DATABASE_COLUMN_NAMES: list[str] = [c[0] for c in DATABASE_COLUMNS]

DUCKDB_SCHEMA: str = ", ".join(
    f'"{col}" {dtype}' for col, _, dtype in DATABASE_COLUMNS
)


# -- Parsed paper structures ------------------------------------------------

class ParsedTable(BaseModel):
    """A single table extracted from a PDF."""
    table_id: str = ""
    caption: str = ""
    headers: list[list[str]] = Field(default_factory=list)
    data_rows: list[dict[str, str]] = Field(default_factory=list)
    footnotes: list[str] = Field(default_factory=list)
    page_number: int | None = None
    # Legacy support: some parsed JSON uses 'rows' as list[list[str]]
    rows: list[list[str]] = Field(default_factory=list)


class TableChunk(BaseModel):
    """A subset of a ParsedTable for chunked extraction."""
    table_id: str
    caption: str
    headers: list[list[str]]
    data_rows: list[dict[str, str]]
    chunk_index: int
    total_chunks: int
    condition_label: str | None = None          # e.g., "Moisture content = 36% w.b."


class PaperMetadata(BaseModel):
    """Paper-level facts extracted by the Screener."""
    doi: str
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    journal: str | None = None
    abstract: str | None = None
    primary_materials: list[str] = Field(default_factory=list)
    measurement_frequencies_mhz: list[float] = Field(default_factory=list)
    temperature_range_c: tuple[float | None, float | None] | None = None
    # Moisture information, used to build honest evaluation grids for papers
    # that report empirical models instead of individual measurements.
    moisture_range_pct: tuple[float | None, float | None] | None = None
    moisture_levels_pct: list[float] = Field(default_factory=list)
    moisture_basis: str | None = None
    data_tables: list[str] = Field(default_factory=list)
    equation_tables: list[str] = Field(default_factory=list)
    skip_tables: list[str] = Field(default_factory=list)
    estimated_total_records: int = 0
    measurement_method: str | None = None
    discovery_source: str = "local_pdf"


class ParsedSection(BaseModel):
    heading: str = ""
    text: str = ""
    level: int = 1


class ParsedFigure(BaseModel):
    figure_id: str = ""
    caption: str = ""
    page_number: int | None = None
    image_path: str | None = None


class ParsedPaper(BaseModel):
    """Complete parsed representation of one PDF."""
    # Content-addressed identity. Stable across folders, renumbering and
    # re-runs; see src/paper_id.py. All artifacts are keyed on this.
    paper_uid: str = ""
    parse_complete: bool = True
    parse_error: str = ""
    doi: str
    metadata: PaperMetadata
    sections: list[ParsedSection] = Field(default_factory=list)
    tables: list[ParsedTable] = Field(default_factory=list)
    figures: list[ParsedFigure] = Field(default_factory=list)
    full_text: str = ""
    pdf_path: str = ""
    text_sections: list[dict] = Field(default_factory=list)


# -- Screener result --------------------------------------------------------

class ScreenerResult(BaseModel):
    paper_uid: str = ""
    doi: str
    complete: bool = True
    incomplete_reason: str = ""
    estimated_records: int = 0
    data_sources: list[str] = Field(default_factory=list)
    extraction_priority: ExtractionPriority = ExtractionPriority.MEDIUM
    complexity: Complexity = Complexity.MODERATE
    has_equations: bool = False
    figure_only: bool = False
    notes: str = ""
    # Whitelisted paper metadata returned by screening. Keeping it with the
    # cache lets a newly regenerated parse be enriched without paying for or
    # depending on another model call.
    metadata: dict = Field(default_factory=dict)


# -- Extraction result -------------------------------------------------------

class ExtractionResult(BaseModel):
    """Output of extraction for one paper."""
    paper_uid: str = ""
    # False when a provider failure, missing batch response, or timeout means
    # the artifact is only partial and must not satisfy resume checks.
    complete: bool = True
    incomplete_reason: str = ""
    # True when a per-paper timeout stopped extraction early. The records
    # present are real but incomplete; this must never be reported as a
    # paper that simply had no data.
    timed_out: bool = False
    doi: str
    records: list[DielectricRecord] = Field(default_factory=list)
    tables_processed: list[str] = Field(default_factory=list)
    tables_skipped: list[str] = Field(default_factory=list)
    extraction_log: list[str] = Field(default_factory=list)
    extraction_source: ExtractionSource | None = None
    notes: str = ""
    token_usage: dict = Field(default_factory=dict)
    # Audit trail for equation evaluation: how many model points were
    # generated, and how many were rejected and why. Never leave a paper
    # silently at zero records.
    equation_report: dict = Field(default_factory=dict)


# -- Validation models -------------------------------------------------------

class ValidationIssue(BaseModel):
    """A single validation issue found on a record."""
    field: str                                     # e.g., "dielectric_constant"
    message: str                                   # e.g., "Value 350.0 exceeds max 200.0"
    severity: str = "error"                        # "error" or "warning"


class ValidatedRecord(BaseModel):
    """A record annotated with its validation verdict."""
    record: DielectricRecord
    verdict: ValidationVerdict = ValidationVerdict.ACCEPTED
    issues: list[ValidationIssue] = Field(default_factory=list)


class PaperValidationSummary(BaseModel):
    """Validation summary for all records from one paper."""
    doi: str
    validated_records: list[ValidatedRecord] = Field(default_factory=list)
    accepted_records: int = 0
    flagged_records: int = 0
    rejected_records: int = 0


# -- Checkpoint / Cost -------------------------------------------------------

class PaperCheckpoint(BaseModel):
    doi: str
    status: PipelineStatus = PipelineStatus.DISCOVERED
    pdf_path: str = ""
    parse_time: datetime | None = None
    screen_time: datetime | None = None
    extract_time: datetime | None = None
    assemble_time: datetime | None = None
    error_message: str | None = None
    api_cost_usd: float = 0.0


class CostEntry(BaseModel):
    stage: str
    model: str
    doi: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
