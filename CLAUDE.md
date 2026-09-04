# agentXtract — Contributor Notes

## Project overview

agentXtract extracts food dielectric-property records from scientific PDFs and
exports validated CSV, Parquet, and DuckDB datasets.

## Pipeline

```text
PDF → parse (Docling) → screen → extract → assemble → audit
```

1. `src/pdf_parser.py` converts PDFs to structured sections, tables, and
   figures. Docling is imported lazily.
2. `src/screener.py` classifies relevance and records source conditions and
   candidate tables.
3. `src/table_extractor.py` and `src/text_extractor.py` extract records. The
   default configuration uses Haiku for ordinary tables and Sonnet for
   prose, equation tables, and vision fallback; `configs/models.yaml` is the
   source of truth.
4. `src/equation_model.py` parses supported regression forms and evaluates
   them only where model domains and source-reported conditions overlap.
5. `src/assembler.py` validates, merges, deduplicates, and exports data;
   `src/audit.py` performs the optional post-assembly audit.

Content-addressed paper identity is implemented in `src/paper_id.py`, shared
data models in `src/schema.py`, API/checkpoint utilities in `src/utils.py`, and
run manifests in `src/provenance.py`. Prompts live in `skills/*.md`; do not
duplicate them in source code.

## Local data

Place authorized PDFs anywhere below `data/corpus/`. PDFs, parsed/extracted
artifacts, databases, and benchmark annotations are intentionally not tracked.
See `data/README.md` for the public data-availability statement and local
layout. Do not add corpus counts or folder splits here because local corpora can
differ between runs.

## Environments

- `requirements-lock.txt` is the locked Python 3.12 production environment and
  includes Docling for real PDF parsing.
- `requirements-ci-lock.txt` is the locked unit-test environment. It excludes
  Docling's GPU-capable Torch stack because CI has no redistributable corpus for
  an end-to-end parser test; parser adapter logic is covered with fixtures.
- `requirements.txt` and `requirements-ci.txt` contain the corresponding
  minimum-version inputs used to regenerate the locks.

## Commands

```bash
python -m src.pipeline --stage all
python -m src.pipeline --stage parse
python -m src.pipeline --file path/to/paper.pdf
python -m src.pipeline --dry-run
python -m src.pipeline --status
python -m src.pipeline --force
python evaluate.py --gold path/to/reference_records.csv
python benchmark.py --models haiku sonnet
python -m pytest tests -q
```

Use `--realtime` when selecting a provider without a supported batch wrapper.
Benchmark runs already disable batch mode and vision fallback so candidates are
compared using the same input modality.
