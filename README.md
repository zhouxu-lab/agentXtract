# agentXtract

LLM-powered pipeline for extracting structured data from scientific PDF papers.

## Overview

agentXtract is a multi-agent pipeline that reads scientific PDFs, identifies relevant data tables and text, and extracts structured measurements into a validated database. This release is specialized for food dielectric-property extraction; adapting it to another domain requires changing the schemas, validators, and assembly rules as well as the prompt templates.

The default configuration uses Claude models for screening and extraction and an OpenAI model for optional audit review. Model/provider choices are configured in `configs/models.yaml`.

## Pipeline Architecture

```
PDF papers
    |
    v
[1. Parse]  ----  Docling: PDF --> structured JSON (sections, tables, figures)
    |
    v
[2. Screen] ----  Claude Haiku: classify relevance, identify data tables
    |
    v
[3. Extract] ---  Claude Haiku (tables) / Claude Sonnet (text, equations)
    |              chunked table extraction with vision fallback
    v
[4. Assemble] --  Rule-based validation, split-record merging,
                   multi-pass deduplication, DuckDB/CSV/Parquet export
```

The pipeline requests `temperature=0` when the selected provider SDK supports it. Model outputs can still vary across calls and model revisions. Prompt templates are stored as Markdown files in `skills/`.

## Output Schema

The output CSV contains 25 fields:

| Field | Description |
|-------|-------------|
| `material_name` | Material name (for example, "sample powder") |
| `frequency_mhz` | Measurement frequency in MHz |
| `temperature_c` | Temperature in Celsius |
| `dielectric_constant` | Dielectric constant (real part of permittivity) |
| `loss_factor` | Dielectric loss factor (imaginary part) |
| `loss_tangent` | Loss tangent (computed) |
| `moisture_content_pct` | Moisture content (%) |
| `moisture_basis` | Wet or dry basis |
| `salt_content` | Salt content if reported |
| `electrical_conductivity_s_m` | Electrical conductivity in S/m, if reported |
| `paper_id` | Normalized human-readable source-paper identifier |
| `paper_uid` | Stable content hash used to join all artifacts from one PDF |
| `doi` | DOI reported or recovered for the source paper |
| `source_table` | Table identifier in the source paper |
| `source_location` | Page, section, row, or other source locator when available |
| `data_provenance` | `measured_table`, `measured_text`, `vision_table`, or `equation_derived` origin |
| `model_expression` | Source equation for equation-derived records |
| `model_r_squared` | Reported fit statistic for the source equation |
| `extraction_source` | Text, table, figure, or equation extraction route |
| `extraction_model` | Model that produced the extraction |
| `title` | Paper title |
| `authors` | Author list |
| `year` | Publication year |
| `journal` | Journal name |
| `measurement_method` | Measurement technique used |

## Quick Start

### Install dependencies

Python 3.12 is the tested runtime. For a reproducible environment, install the
exact resolved dependency set:

```bash
python -m pip install -r requirements-lock.txt
```

`requirements.txt` retains minimum versions for maintainers who intentionally
want to resolve a newer environment.

The first parse downloads Docling's public OCR, layout, and table models. Allow
network access for that initial run; later parses reuse the local model cache.

### Set up API keys

```bash
cp .env.example .env
# Add keys only for the providers selected in configs/models.yaml
```

### Run the pipeline

```bash
# Place PDFs anywhere under data/corpus/ (nested folders are supported)

# Run all stages
python -m src.pipeline --stage all

# Run a single PDF
python -m src.pipeline --file path/to/paper.pdf

# Re-extract a selected batch without clearing other papers' artifacts
python -m src.pipeline --stage extract --file paper-a.pdf --file paper-b.pdf --force

# Run specific stages
python -m src.pipeline --stage parse      # Parse PDFs only
python -m src.pipeline --stage screen     # Screen only
python -m src.pipeline --stage extract    # Extract only
python -m src.pipeline --stage assemble   # Assemble from cached extractions
python -m src.pipeline --stage audit      # Rule-based audit of the assembled CSV

# Other options
python -m src.pipeline --status           # Check progress
python -m src.pipeline --dry-run          # Estimate cost before running
python -m src.pipeline --force            # Reprocess all papers
python -m src.pipeline --realtime         # Bypass batch API for an immediate run
python -m src.pipeline --strict-coverage  # Fail if any selected paper yields no rows
```

Every assembly writes `data/database/paper_coverage.csv`,
`paper_coverage_summary.json`, and `run_provenance.json`. These files account for every parsed paper and
distinguish no extraction artifact, zero extracted records, incomplete-only
records, downstream loss, and successful contribution. The default run warns
on zero-yield papers; `--strict-coverage` turns that warning into a failing
quality gate. The provenance record includes source content IDs, model names,
dependency versions, and hashes of the code, prompts, configuration, and
extraction inputs used for the assembly. Active local override content and
trusted postprocessor source files are included by content hash without
recording their absolute local paths.

### Evaluate against ground truth

```bash
python evaluate.py --gold path/to/reference_records.csv
```

Reference data must be a tidy CSV, TSV, or Excel file using the public output
column names. Source documents and annotations are not redistributed; see
[data/README.md](data/README.md).

### Optional local assembly rules

Ordinary runs use only corpus-agnostic assembly. If a project needs a documented
source correction, point `assembly.overrides_file` at a local YAML file with
`doi_overrides`, `material_aliases`, or exact-match `row_rules`. Complex trusted
logic can be supplied through `assembly.postprocessors` entries written as
`module:function`. Keep these project-owned files under `local_extensions/` or
name the YAML file `*.local.yaml`; both locations are ignored by Git.
See `configs/assembly_overrides.example.yaml` for a synthetic template.

### Compare extraction models

```bash
python benchmark.py --models haiku sonnet gpt-4.1-mini
```

The benchmark assigns each candidate to the text, table, and equation
extractors and uses the same prose/table routing as the production pipeline.
Vision fallback is disabled uniformly during cross-provider benchmarks because
the current provider wrappers do not all support image input. This keeps the
comparison modality-matched; normal pipeline runs continue to use the
`vision_fallback` setting in `configs/models.yaml`.

## Repository Structure

```
agentXtract/
├── src/                    Pipeline source code
│   ├── pipeline.py         Main orchestrator
│   ├── pdf_parser.py       Docling-based PDF parsing
│   ├── screener.py         Haiku relevance screening
│   ├── table_extractor.py  Table chunking + extraction
│   ├── text_extractor.py   Prose extraction (Sonnet)
│   ├── equation_model.py   Safe equation parsing and evaluation
│   ├── assembler.py        Validation, dedup, export
│   ├── schema.py           All Pydantic data models
│   ├── audit.py            Data quality auditing
│   ├── paper_id.py         Content-addressed paper identity and manifest
│   ├── provenance.py       Reproducibility manifest generation
│   ├── progress.py         Progress display
│   └── utils.py            API wrappers, config loading
├── skills/                 LLM prompt templates (.md)
├── configs/                YAML configuration
│   ├── models.yaml         Model selection and pricing
│   ├── thresholds.yaml     Validation bounds
│   ├── test_mode.yaml      Pipeline settings
│   └── doi_map.yaml        Optional user-supplied DOI overrides
├── data/
│   ├── README.md           Data availability and local layout
│   └── corpus/             Input PDFs (local and gitignored)
├── tests/                  Test suite
├── evaluate.py             Reference-data evaluation
├── benchmark.py            Multi-model benchmarking
├── requirements.txt        Minimum production dependencies
├── requirements-lock.txt   Locked production environment
├── requirements-ci.txt     Test-only dependencies (no Docling/Torch)
└── requirements-ci-lock.txt Locked CI environment
```

## Reproducibility

The assembly stage (`--stage assemble`) uses stable ordering and deterministic
rules. Given the same manifest, parsed metadata, extraction cache, code, and
configuration, the primary CSV/Parquet/DuckDB data are reproducible. The
provenance file includes its generation timestamp.

The repository test suite runs automatically on pushes and pull requests via
GitHub Actions. CI uses `requirements-ci-lock.txt`, which omits Docling's
GPU-capable Torch stack because the public repository has no corpus-backed PDF
parser integration test; parser adapter behavior is tested with deterministic
fixtures. Use `requirements-lock.txt` for actual PDF parsing and end-to-end
pipeline runs.

Re-running the full pipeline with fresh LLM API calls (`--stage all --force`) may produce slightly different results due to inherent LLM non-determinism, even at `temperature=0`.

## Data privacy and source rights

The pipeline sends extracted document content to the model providers configured
for a run. Use only source documents that you are authorized to process, and
review each provider's current data-retention and privacy terms before working
with confidential, unpublished, or access-restricted material.

## License

MIT License. See [LICENSE](LICENSE) for details.
