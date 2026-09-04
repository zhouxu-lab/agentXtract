"""Main pipeline orchestrator — agent-based coordinator.

Each paper flows through parse → screen → extract independently as a
concurrent ``PaperAgent`` coroutine.  A global assemble step runs once
all agents finish.

When the Batch API is enabled (``use_batch_api: true`` in config), the
pipeline falls back to the original sequential stage-by-stage execution
because the batch endpoints need all inputs upfront.

Usage:
    python -m src.pipeline --stage all
    python -m src.pipeline --file path/to/source.pdf
    python -m src.pipeline --stage parse
    python -m src.pipeline --stage extract
    python -m src.pipeline --dry-run
    python -m src.pipeline --status
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.paper_id import (
    Manifest,
    artifact_name,
    compute_uid,
    find_artifact,
    split_artifact_name,
)
from src.utils import write_json_atomic

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

STAGE_ORDER = [
    "parse",
    "screen",
    "extract",
    "assemble",
    "audit",
]


# ---------------------------------------------------------------------------
# Data structure — one per PDF travelling through the pipeline
# ---------------------------------------------------------------------------

@dataclass
class PaperTask:
    pdf_path: Path
    stem: str
    # Content-addressed identity, computed once from the PDF bytes. Every
    # artifact for this paper is keyed on it. See src/paper_id.py.
    uid: str = ""
    parsed_paper: object | None = None       # ParsedPaper (deferred import)
    screener_result: object | None = None    # ScreenerResult
    extraction_results: list = field(default_factory=list)
    status: str = "pending"  # pending | parsing | screening | extracting | done | skipped | failed
    error: str | None = None


# ---------------------------------------------------------------------------
# Helper functions (extracted from the old monolithic run_stage)
# ---------------------------------------------------------------------------

def _load_doi_map() -> dict[str, str]:
    """Load and normalize configs/doi_map.yaml."""
    import yaml
    doi_map_path = PROJECT_ROOT / "configs" / "doi_map.yaml"
    if not doi_map_path.exists():
        return {}
    with open(doi_map_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    normalized: dict[str, str] = {}
    for key, val in raw.items():
        normalized[key] = val
        space_key = key.replace("_", " ")
        if space_key != key:
            normalized[space_key] = val
        under_key = key.replace(" ", "_")
        if under_key != key:
            normalized[under_key] = val
    return normalized


def _apply_doi_mapping(paper, stem: str, doi_map: dict[str, str]) -> None:
    """Apply a DOI override from doi_map to *paper* (mutates in place)."""
    if paper.doi in doi_map:
        real_doi = doi_map[paper.doi]
        paper.doi = real_doi
        paper.metadata.doi = real_doi
        logger.info(f"  DOI mapped (by doi): {paper.doi}")
        return
    local_key = f"local/{stem.replace('_', ' ')}"
    local_key2 = f"local/{stem}"
    for key in (local_key, local_key2):
        if key in doi_map:
            real_doi = doi_map[key]
            paper.doi = real_doi
            paper.metadata.doi = real_doi
            logger.info(f"  DOI mapped (by stem): {paper.doi}")
            return


def _load_one_parsed_paper(parsed_dir: Path, stem: str, uid: str = ""):
    """Load a single ParsedPaper for a paper.

    Looks up by content-addressed uid first. The filename-based lookups are
    retained only to read artifacts written by older versions of the
    pipeline, which used two different and mutually incompatible naming
    conventions for the same paper.
    """
    from src.schema import ParsedPaper
    f = _find_one_parsed_file(parsed_dir, stem, uid)
    if f is None or not f.exists():
        return None
    with open(f, encoding="utf-8") as fh:
        data = json.load(fh)
    meta = data.get("metadata", {})
    tr = meta.get("temperature_range_c")
    if tr is not None and (tr[0] is None or tr[1] is None):
        meta["temperature_range_c"] = None
    return ParsedPaper.model_validate(data)


def _find_one_parsed_file(parsed_dir: Path, stem: str, uid: str = "") -> Path | None:
    """Return the parsed artifact path for one source PDF."""
    f = None
    if uid:
        f = find_artifact(parsed_dir, uid)
    if f is None:
        for legacy in (parsed_dir / f"{stem}.json",
                       parsed_dir / f"{_legacy_safe_stem(stem)}.json"):
            if legacy.exists():
                f = legacy
                break
    return f


def _clear_forced_extractions(pdf_files: list[Path], config: dict) -> list[Path]:
    """Delete stale extraction artifacts only for explicitly selected PDFs."""
    extracted_dir = Path(config["output_dir"]) / "extracted"
    parsed_dir = Path(config["output_dir"]) / "parsed"
    stale: set[Path] = set()
    for pdf_path in pdf_files:
        uid = compute_uid(pdf_path)
        stale.update(extracted_dir.glob(f"{uid}__*.json"))
        paper = _load_one_parsed_paper(parsed_dir, pdf_path.stem, uid)
        if paper is not None and paper.doi:
            legacy_label = paper.doi.replace("/", "_").replace("\\", "_")
            stale.update(extracted_dir.glob(f"{legacy_label}_*.json"))
    for artifact in stale:
        artifact.unlink()
    return sorted(stale)


def _resolve_requested_pdfs(file_args: str | list[str] | None) -> list[Path]:
    """Resolve one or more repeatable ``--file`` arguments."""
    if not file_args:
        return []
    requested = [file_args] if isinstance(file_args, str) else file_args
    resolved: list[Path] = []
    for raw_path in requested:
        selected = Path(raw_path)
        if not selected.is_absolute() and not selected.exists():
            selected = PROJECT_ROOT / selected
        selected = selected.resolve()
        if not selected.is_file():
            raise FileNotFoundError(f"PDF file not found: {selected}")
        if selected.suffix.lower() != ".pdf":
            raise ValueError(f"Input file is not a PDF: {selected}")
        resolved.append(selected)
    return resolved


def _discover_corpus_pdfs(input_dir: Path) -> list[Path]:
    """Discover PDFs recursively and case-insensitively under ``input_dir``."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        return []
    return sorted(
        (
            path.resolve()
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: path.as_posix().casefold(),
    )


def _deduplicate_pdf_paths(pdf_files: list[Path]) -> list[Path]:
    """Collapse byte-identical PDFs, retaining the first deterministic path."""
    unique: list[Path] = []
    first_by_uid: dict[str, Path] = {}
    for path in pdf_files:
        uid = compute_uid(path)
        first = first_by_uid.get(uid)
        if first is None:
            first_by_uid[uid] = path
            unique.append(path)
        else:
            logger.warning(
                "  Duplicate PDFs share content uid %s: %s, %s — "
                "processing only %s",
                uid,
                first.name,
                path.name,
                first.name,
            )
    return unique


def _sync_corpus_manifest(
    pdf_files: list[Path],
    output_dir: Path,
    *,
    preserve_existing: bool = False,
) -> Manifest:
    """Rebuild the manifest from the current corpus, including legacy parses."""
    manifest = Manifest(output_dir / "manifest.json", strict=True)
    existing = dict(manifest.entries)
    manifest.entries = dict(existing) if preserve_existing else {}
    parsed_dir = output_dir / "parsed"
    for pdf_path in pdf_files:
        uid = compute_uid(pdf_path)
        fields = dict(existing.get(uid, {}))
        fields.pop("uid", None)
        paper = _load_one_parsed_paper(parsed_dir, pdf_path.stem, uid)
        if paper is not None:
            fields.update({
                "doi": paper.doi,
                "title": paper.metadata.title,
                "year": paper.metadata.year,
            })
        fields.update({
            "pdf_path": str(pdf_path.resolve()),
            "slug": pdf_path.stem,
        })
        manifest.upsert(uid, **fields)
    manifest.save()
    return manifest


def _already_extracted(
    uid: str,
    extracted_dir: Path,
    *,
    require_text: bool = False,
) -> bool:
    """True when the expected extraction output exists for this paper.

    Keyed on the content uid. The previous version keyed on the DOI string,
    which changed between stages, so the check silently missed and stale
    results from unrelated runs were left in place. Prose-only papers require
    a text artifact; a zero-row table artifact must not mask a failed text
    extraction.
    """
    if not uid or not extracted_dir.exists():
        return False
    from src.schema import ExtractionResult

    suffix = "_text.json" if require_text else "_table.json"
    for artifact in extracted_dir.glob(f"{uid}__*{suffix}"):
        try:
            with open(artifact, encoding="utf-8") as handle:
                result = ExtractionResult.model_validate(json.load(handle))
            if result.complete:
                return True
            logger.info(
                "  Retrying incomplete extraction artifact %s: %s",
                artifact.name,
                result.incomplete_reason or "reason not recorded",
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning(
                "  Ignoring corrupt extraction artifact %s: %s",
                artifact.name,
                exc,
            )
    return False


def _legacy_safe_stem(stem: str) -> str:
    """Reproduce the old pdf_parser filename mangling, for reading only."""
    return re.sub(r"[^\w\-]", "_", stem)


def _warn_on_duplicate_pdfs(tasks: list) -> None:
    """Flag byte-identical PDFs present under more than one name.

    Compatibility helper for callers that construct ``PaperTask`` objects
    directly. Normal pipeline discovery removes duplicates before tasks are
    built.
    """
    from collections import defaultdict
    by_uid = defaultdict(list)
    for t in tasks:
        by_uid[t.uid].append(t.pdf_path)
    for uid, paths in by_uid.items():
        if len(paths) > 1:
            names = ", ".join(p.name for p in paths)
            logger.warning(
                f"  Duplicate PDFs share content uid {uid}: {names} "
                f"— processing once"
            )


def _clean_stale_text(paper, extracted_dir: Path) -> None:
    """Remove stale text-extraction JSON for papers that have data tables."""
    if paper.metadata.data_tables:
        label = paper.doi.replace("/", "_").replace("\\", "_")
        stale_paths = {extracted_dir / f"{label}_text.json"}
        if paper.paper_uid:
            stale_paths.update(
                extracted_dir.glob(f"{paper.paper_uid}__*_text.json")
            )
        for stale in sorted(stale_paths):
            if stale.exists():
                logger.info(f"  Removing stale text extraction: {stale.name}")
                stale.unlink()


def _default_screener_result(doi: str, paper_uid: str = ""):
    from src.schema import Complexity, ExtractionPriority, ScreenerResult
    return ScreenerResult(
        paper_uid=paper_uid,
        doi=doi,
        estimated_records=0,
        data_sources=[],
        extraction_priority=ExtractionPriority.MEDIUM,
        complexity=Complexity.MODERATE,
        notes="Default; screener result not found.",
    )


# ---------------------------------------------------------------------------
# Per-paper agent coroutine
# ---------------------------------------------------------------------------

async def process_paper(
    task: PaperTask,
    config: dict,
    checkpoint_db,
    stages: list[str],
    doi_map: dict[str, str],
    screen_lock: asyncio.Lock,
    parse_sem: asyncio.Semaphore,
    progress=None,
) -> None:
    """Drive one paper through parse → screen → extract."""
    from src.schema import (
        ExtractionPriority,
        PaperCheckpoint,
        PipelineStatus,
    )

    parsed_dir = Path(config["output_dir"]) / "parsed"
    extracted_dir = Path(config["output_dir"]) / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    force = config.get("force_reprocess", False)

    # ── PARSE ──────────────────────────────────────────────────────────
    if "parse" in stages:
        task.status = "parsing"
        from src.pdf_parser import run as parse_run

        async with parse_sem:
            papers = await parse_run(
                pdf_paths=[task.pdf_path],
                config=config,
                output_dir=parsed_dir,
            )
        if not papers:
            raise RuntimeError(
                f"Parse stage produced no artifact for {task.stem}; "
                "inspect the preceding parser error"
            )
        if papers:
            task.parsed_paper = papers[0]
            if not task.parsed_paper.paper_uid:
                task.parsed_paper.paper_uid = task.uid
            checkpoint_db.upsert(PaperCheckpoint(
                doi=task.uid,
                status=PipelineStatus.PARSED,
                pdf_path=str(task.pdf_path),
                parse_time=datetime.now(timezone.utc),
            ))
            logger.info(f"  Parsed: {task.stem}")

    # ── SCREEN ─────────────────────────────────────────────────────────
    if "screen" in stages:
        task.status = "screening"
        from src.screener import run as screen_run

        # Load from disk if we didn't just parse
        if task.parsed_paper is None:
            task.parsed_paper = _load_one_parsed_paper(
                parsed_dir, task.stem, task.uid
            )
        if task.parsed_paper is None:
            raise RuntimeError(f"No parsed output for {task.stem}")

        paper = task.parsed_paper
        if not paper.paper_uid:
            paper.paper_uid = task.uid
        # NOTE: the DOI is metadata, not identity. It used to be rebuilt here
        # from the filename, which produced a different key at every stage.
        # Identity is task.uid; the DOI is left exactly as parsed.

        # Serialize screener calls — screener.run() reads/writes screener_results.json
        async with screen_lock:
            results = await screen_run(
                papers=[paper],
                config=config,
                checkpoint_db=checkpoint_db,
                progress=None,
            )

        if results:
            task.screener_result = results[0]
        if task.screener_result is None or not task.screener_result.complete:
            reason = (
                task.screener_result.incomplete_reason
                if task.screener_result is not None else "no result returned"
            )
            raise RuntimeError(f"Screening failed for {task.stem}: {reason}")

        # Apply DOI mapping AFTER screening
        _apply_doi_mapping(paper, task.stem, doi_map)

        # Re-save enriched parsed paper, overwriting the artifact the parser
        # wrote rather than creating a second file under a different name.
        out_file = (find_artifact(parsed_dir, task.uid)
                    or parsed_dir / f"{artifact_name(task.uid, task.stem)}.json")
        write_json_atomic(out_file, paper.model_dump(mode="json"))

        # Keep the manifest authoritative for the record/metadata join.
        manifest = Manifest(parsed_dir.parent / "manifest.json", strict=True)
        manifest.upsert(
            task.uid,
            pdf_path=str(task.pdf_path), slug=task.stem, doi=paper.doi,
            title=paper.metadata.title, journal=paper.metadata.journal,
            year=paper.metadata.year,
            authors="; ".join(paper.metadata.authors or []),
            measurement_method=paper.metadata.measurement_method,
            data_tables=paper.metadata.data_tables,
            skip_tables=paper.metadata.skip_tables,
            estimated_total_records=paper.metadata.estimated_total_records,
        )
        manifest.save()

        logger.info(f"  Screened: {task.stem}")

    # ── EXTRACT ────────────────────────────────────────────────────────
    if "extract" in stages:
        task.status = "extracting"
        from src.table_extractor import run as table_run
        from src.text_extractor import run as text_run

        # Load from disk if needed
        if task.parsed_paper is None:
            task.parsed_paper = _load_one_parsed_paper(
                parsed_dir, task.stem, task.uid
            )
        if task.parsed_paper is None:
            raise RuntimeError(f"No parsed output for {task.stem}")

        paper = task.parsed_paper
        if not paper.paper_uid:
            paper.paper_uid = task.uid

        # Load screener result if not in memory
        if task.screener_result is None:
            all_sr = load_screener_results(parsed_dir)
            task.screener_result = _match_screener_result(paper, all_sr)
        sr = task.screener_result or _default_screener_result(
            paper.doi, paper.paper_uid
        )
        if not sr.complete:
            raise RuntimeError(
                f"Cached screening is incomplete for {task.stem}: "
                f"{sr.incomplete_reason or 'run the screen stage again'}"
            )

        # SKIP check
        if sr.extraction_priority == ExtractionPriority.SKIP:
            logger.info(f"  Skipping (screener SKIP): {task.stem}")
            task.status = "skipped"
            return

        # Incremental check
        if not force and _already_extracted(
            task.uid,
            extracted_dir,
            require_text=not bool(paper.metadata.data_tables),
        ):
            logger.info(f"  Already extracted, skipping: {task.stem}")
            task.status = "done"
            return

        # Clean stale text extraction for papers with data tables
        _clean_stale_text(paper, extracted_dir)

        # Run extraction
        paired = [(paper, sr)]
        if not paper.metadata.data_tables:
            text_results = await text_run(paired, config, checkpoint_db, progress=None)
            task.extraction_results.extend(text_results)
        table_results = await table_run(paired, config, checkpoint_db, progress=None)
        task.extraction_results.extend(table_results)

        incomplete = [
            result for result in task.extraction_results if not result.complete
        ]
        if incomplete:
            reasons = "; ".join(
                result.incomplete_reason or "unknown extraction failure"
                for result in incomplete
            )
            raise RuntimeError(f"Incomplete extraction for {task.stem}: {reasons}")

        total = sum(len(r.records) for r in task.extraction_results)
        logger.info(f"  Extracted {total} records: {task.stem}")

        # A paper that yields nothing is a result, not a non-event. Record it
        # so zero-yield papers surface in the run summary instead of quietly
        # dropping out of the database.
        if total == 0:
            reasons = []
            for r in task.extraction_results:
                if getattr(r, "equation_report", None):
                    reasons.append(str(r.equation_report.get("summary", "")))
                if r.notes:
                    reasons.append(r.notes)
            logger.warning(
                f"  ZERO RECORDS from {task.stem} "
                f"(tables={paper.metadata.data_tables}; "
                f"{'; '.join(x for x in reasons if x) or 'no reason reported'})"
            )

        checkpoint_db.upsert(PaperCheckpoint(
            doi=task.uid,
            status=PipelineStatus.EXTRACTED,
            extract_time=datetime.now(timezone.utc),
        ))

    task.status = "done"


# ---------------------------------------------------------------------------
# Coordinator — the main entry point
# ---------------------------------------------------------------------------

async def run_pipeline(args):
    """Main pipeline entry point — agent-based coordinator."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    from src.utils import CheckpointDB, load_config

    config = load_config("configs/test_mode.yaml", "configs/models.yaml", "configs/thresholds.yaml")
    for key in ("input_dir", "output_dir"):
        if key not in config:
            raise KeyError(f"Required configuration key is missing: {key}")
        configured = Path(config[key])
        if not configured.is_absolute():
            config[key] = str((PROJECT_ROOT / configured).resolve())

    input_dir = Path(config["input_dir"])
    output_dir = Path(config["output_dir"])
    all_corpus_pdfs = _deduplicate_pdf_paths(
        _discover_corpus_pdfs(input_dir)
    )

    # Determine which PDFs to process
    requested = _resolve_requested_pdfs(getattr(args, "file", None))
    if requested:
        pdf_files = _deduplicate_pdf_paths(requested)
    else:
        pdf_files = all_corpus_pdfs

    # Status and cost estimation are read-only: do not create a checkpoint DB
    # or rebuild the manifest merely because the user asked a question.
    db_path = output_dir / "pipeline.db"
    if getattr(args, "status", False):
        checkpoint_db = CheckpointDB(db_path) if db_path.exists() else None
        try:
            print_status(checkpoint_db, all_corpus_pdfs)
        finally:
            if checkpoint_db is not None:
                checkpoint_db.close()
        return 0

    if getattr(args, "dry_run", False):
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {input_dir}")
        estimate_cost(pdf_files, config)
        return 0

    stages_to_run = [s for s in STAGE_ORDER if s != "audit"] if args.stage == "all" else [args.stage]
    needs_papers = any(
        stage in {"parse", "screen", "extract"} for stage in stages_to_run
    )
    if needs_papers and not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {input_dir}")

    if pdf_files:
        # A scoped --file run must add its papers without deleting entries for
        # the rest of the corpus. A full corpus run intentionally prunes stale
        # entries so assembly cannot pick up unrelated artifacts.
        manifest_files = (
            _deduplicate_pdf_paths([*all_corpus_pdfs, *pdf_files])
            if requested else all_corpus_pdfs
        )
        _sync_corpus_manifest(
            manifest_files,
            output_dir,
            preserve_existing=bool(requested),
        )

    logger.info(f"Found {len(pdf_files)} PDFs to process")
    config["force_reprocess"] = getattr(args, "force", False)
    config["strict_coverage"] = getattr(args, "strict_coverage", False)
    if getattr(args, "realtime", False):
        config["use_batch_api"] = False

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_db = CheckpointDB(db_path)
    try:
        # Enable WAL mode for better concurrent read performance
        checkpoint_db._conn.execute("PRAGMA journal_mode=WAL")

        # Batch API requires all papers upfront — fall back to sequential execution
        if config.get("use_batch_api"):
            await _run_pipeline_sequential(
                args, pdf_files, stages_to_run, config, checkpoint_db
            )
        else:
            await _run_pipeline_concurrent(
                pdf_files, stages_to_run, config, checkpoint_db
            )
    finally:
        checkpoint_db.close()
    return 0


async def _run_pipeline_concurrent(pdf_files, stages_to_run, config, checkpoint_db):
    """Agent-based concurrent execution — each paper flows independently."""
    from rich.logging import RichHandler

    from src.progress import PipelineProgress, console

    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(RichHandler(console=console, show_path=False, markup=False))

    per_paper_stages = [
        s for s in stages_to_run if s not in {"assemble", "audit"}
    ]
    run_assemble = "assemble" in stages_to_run
    force = config.get("force_reprocess", False)

    # Clean only the selected papers before forced re-extraction. A previous
    # implementation deleted every extraction artifact even with --file.
    if force and "extract" in per_paper_stages:
        stale = _clear_forced_extractions(pdf_files, config)
        if stale:
            logger.info(f"  Cleared {len(stale)} stale extraction files for forced re-extraction")

    # Shared resources
    parse_sem = asyncio.Semaphore(2)
    screen_lock = asyncio.Lock()
    doi_map = _load_doi_map()

    # Build tasks
    tasks = [
        PaperTask(pdf_path=p, stem=p.stem, uid=compute_uid(p))
        for p in pdf_files
    ]
    _warn_on_duplicate_pdfs(tasks)

    stage_times: dict[str, float] = {}

    with PipelineProgress() as progress:
        # -- Per-paper concurrent phase ----------------------------------------
        if per_paper_stages:
            progress.start_stage("pipeline", len(tasks))
            t0 = time.time()

            async def _run_one(task: PaperTask):
                try:
                    await process_paper(
                        task, config, checkpoint_db, per_paper_stages,
                        doi_map, screen_lock, parse_sem, progress,
                    )
                # Per-paper isolation is required so one unexpected parser or
                # provider failure cannot cancel successful sibling tasks.
                except Exception as e:  # noqa: BLE001
                    task.status = "failed"
                    task.error = str(e)
                    logger.error(f"Paper {task.stem} failed: {e}")
                finally:
                    progress.advance_paper(task.stem)

            await asyncio.gather(*[_run_one(t) for t in tasks])
            stage_times["pipeline"] = time.time() - t0

        # -- Global assemble phase ---------------------------------------------
        if run_assemble:
            progress.start_stage("assemble", 1)
            t0 = time.time()
            _run_assemble(config, checkpoint_db)
            progress.advance_paper("database")
            stage_times["assemble"] = time.time() - t0

        # -- Audit phase (runs after assemble if requested) --------------------
        run_audit = "audit" in stages_to_run
        if run_audit:
            t0 = time.time()
            _run_audit(config)
            stage_times["audit"] = time.time() - t0

    # Report failures
    failed = [t for t in tasks if t.status == "failed"]
    if failed:
        logger.warning(f"{len(failed)} paper(s) failed:")
        for t in failed:
            logger.warning(f"  {t.stem}: {t.error}")

    succeeded = sum(1 for t in tasks if t.status in ("done", "skipped"))
    logger.info(f"Pipeline complete: {succeeded}/{len(tasks)} succeeded, {len(failed)} failed")

    print_pipeline_summary(checkpoint_db, stage_times)
    if failed:
        raise RuntimeError(f"{len(failed)} paper(s) failed during pipeline execution")


async def _run_pipeline_sequential(args, pdf_files, stages_to_run, config, checkpoint_db):
    """Original sequential stage-by-stage execution (for batch API mode)."""
    from rich.logging import RichHandler

    from src.progress import PipelineProgress, console

    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(RichHandler(console=console, show_path=False, markup=False))

    # Clean only the selected papers before forced re-extraction.
    force = config.get("force_reprocess", False)
    if force and "extract" in stages_to_run:
        stale = _clear_forced_extractions(pdf_files, config)
        if stale:
            logger.info(f"  Cleared {len(stale)} stale extraction files for forced re-extraction")

    stage_times: dict[str, float] = {}
    with PipelineProgress() as progress:
        for stage in stages_to_run:
            if stage not in STAGE_ORDER:
                logger.error(f"Unknown stage: {stage}")
                return
            progress.start_stage(stage, len(pdf_files))
            t0 = time.time()
            await _run_stage_sequential(stage, pdf_files, config, checkpoint_db, progress=progress)
            stage_times[stage] = time.time() - t0

    print_pipeline_summary(checkpoint_db, stage_times)


async def _run_stage_sequential(stage, pdf_files, config, checkpoint_db, progress=None):
    """Dispatch to the appropriate module (original sequential logic)."""
    from src.schema import PaperCheckpoint, PipelineStatus

    if stage == "parse":
        from src.pdf_parser import run as parse_run
        parsed = await parse_run(
            pdf_paths=pdf_files,
            config=config,
            output_dir=Path(config["output_dir"]) / "parsed",
        )
        for p in parsed:
            matched_pdf = next(
                (f for f in pdf_files if compute_uid(f) == p.paper_uid), None
            )
            checkpoint_db.upsert(PaperCheckpoint(
                doi=p.paper_uid or p.doi,
                status=PipelineStatus.PARSED,
                pdf_path=str(matched_pdf) if matched_pdf else "",
                parse_time=datetime.now(timezone.utc),
            ))
            if progress:
                progress.advance_paper(
                    matched_pdf.stem if matched_pdf else p.doi.split("/")[-1]
                )
        logger.info(f"Parsed {len(parsed)} papers")
        if len(parsed) != len(pdf_files):
            raise RuntimeError(
                f"Parse stage produced {len(parsed)}/{len(pdf_files)} paper "
                "artifacts; inspect the preceding errors"
            )

    elif stage == "screen":
        from src.schema import ParsedPaper
        from src.screener import run as screen_run
        parsed_dir = Path(config["output_dir"]) / "parsed"

        parsed_papers = []
        missing_pdfs = []
        paper_file_map: dict[int, Path] = {}
        paper_stem_map: dict[int, str] = {}
        for pdf_path in pdf_files:
            uid = compute_uid(pdf_path)
            f = _find_one_parsed_file(parsed_dir, pdf_path.stem, uid)
            if f is None:
                logger.warning(f"  No parsed artifact for {pdf_path.name}; skipping screen")
                missing_pdfs.append(pdf_path)
                continue
            parsed_text = await asyncio.to_thread(f.read_text, encoding="utf-8")
            paper = ParsedPaper.model_validate(json.loads(parsed_text))
            stem = f.stem
            uid, slug = split_artifact_name(stem)
            if uid and not paper.paper_uid:
                paper.paper_uid = uid
            # The DOI is left as parsed. It used to be rebuilt from the
            # parsed filename here, mapping '_' back to ' ', which could not
            # recover the original punctuation and minted a third identity
            # for the same paper.
            parsed_papers.append(paper)
            paper_file_map[id(paper)] = f
            paper_stem_map[id(paper)] = slug or stem

        if missing_pdfs:
            raise RuntimeError(
                f"Screen stage is missing parsed artifacts for "
                f"{len(missing_pdfs)} paper(s); run --stage parse first"
            )

        results = await screen_run(
            papers=parsed_papers,
            config=config,
            checkpoint_db=checkpoint_db,
            progress=progress,
        )
        incomplete = [result for result in results if not result.complete]
        if incomplete:
            reasons = "; ".join(
                result.incomplete_reason or result.doi
                for result in incomplete
            )
            raise RuntimeError(
                f"Screening incomplete for {len(incomplete)} paper(s): {reasons}"
            )

        doi_map = _load_doi_map()
        for paper in parsed_papers:
            _apply_doi_mapping(paper, paper_stem_map.get(id(paper), ""), doi_map)

        manifest = Manifest(parsed_dir.parent / "manifest.json", strict=True)
        for paper in parsed_papers:
            f = paper_file_map.get(id(paper))
            if f:
                write_json_atomic(f, paper.model_dump(mode="json"))
            if paper.paper_uid:
                manifest.upsert(
                    paper.paper_uid,
                    doi=paper.doi,
                    title=paper.metadata.title,
                    journal=paper.metadata.journal,
                    year=paper.metadata.year,
                    authors="; ".join(paper.metadata.authors or []),
                    measurement_method=paper.metadata.measurement_method,
                    data_tables=paper.metadata.data_tables,
                    skip_tables=paper.metadata.skip_tables,
                    estimated_total_records=paper.metadata.estimated_total_records,
                )
        manifest.save()

        logger.info(f"Screened {len(results)} papers")
        for r in results:
            logger.info(f"  {r.doi}: {r.estimated_records} records, "
                        f"priority={r.extraction_priority.value}, complexity={r.complexity.value}")

    elif stage == "extract":
        from src.schema import ExtractionPriority
        from src.table_extractor import run as table_run
        from src.text_extractor import run as text_run

        parsed_dir = Path(config["output_dir"]) / "parsed"
        parsed_papers = []
        missing_pdfs = []
        for pdf_path in pdf_files:
            paper = _load_one_parsed_paper(
                parsed_dir, pdf_path.stem, compute_uid(pdf_path)
            )
            if paper is None:
                logger.warning(f"  No parsed artifact for {pdf_path.name}; skipping extract")
                missing_pdfs.append(pdf_path)
                continue
            parsed_papers.append(paper)
        if missing_pdfs:
            raise RuntimeError(
                f"Extract stage is missing parsed artifacts for "
                f"{len(missing_pdfs)} paper(s); run --stage parse and "
                "--stage screen first"
            )
        screener_results = load_screener_results(parsed_dir)
        extracted_dir = Path(config["output_dir"]) / "extracted"
        extracted_dir.mkdir(parents=True, exist_ok=True)

        paired = pair_papers_with_screener(parsed_papers, screener_results)
        incomplete_screening = [
            sr for _paper, sr in paired if not sr.complete
        ]
        if incomplete_screening:
            raise RuntimeError(
                f"Cannot extract: {len(incomplete_screening)} cached screener "
                "result(s) are incomplete; rerun --stage screen"
            )

        skip_pairs = [(p, sr) for p, sr in paired if sr.extraction_priority == ExtractionPriority.SKIP]
        if skip_pairs:
            logger.info(f"  Skipping {len(skip_pairs)} papers flagged as SKIP by screener:")
            for p, sr in skip_pairs:
                logger.info(f"    {p.doi}: {sr.notes[:120]}")
            if progress:
                for p, _ in skip_pairs:
                    progress.advance_paper(p.doi.split("/")[-1])
        paired = [(p, sr) for p, sr in paired if sr.extraction_priority != ExtractionPriority.SKIP]

        force = config.get("force_reprocess", False)
        if not force:
            already_done = [
                (p, sr) for p, sr in paired
                if _already_extracted(
                    p.paper_uid,
                    extracted_dir,
                    require_text=not bool(p.metadata.data_tables),
                )
            ]
            if already_done:
                logger.info(f"  Skipping {len(already_done)} already-extracted papers (use --force to reprocess)")
                for p, _ in already_done:
                    if progress:
                        progress.advance_paper(p.doi.split("/")[-1])
            paired = [
                (p, sr) for p, sr in paired
                if not _already_extracted(
                    p.paper_uid,
                    extracted_dir,
                    require_text=not bool(p.metadata.data_tables),
                )
            ]

        if not paired:
            logger.info("  All papers already extracted — nothing to do.")
        else:
            logger.info(f"  Extracting {len(paired)} papers")

        text_pairs = [(p, sr) for p, sr in paired if not p.metadata.data_tables]
        if text_pairs:
            logger.info(f"  Text extraction for {len(text_pairs)} prose-only papers")

        for paper, _ in [(p, sr) for p, sr in paired if p.metadata.data_tables]:
            _clean_stale_text(paper, extracted_dir)

        text_results = await text_run(text_pairs, config, checkpoint_db, progress=progress) if text_pairs else []
        table_results = await table_run(paired, config, checkpoint_db, progress=progress)

        incomplete = [
            result for result in [*text_results, *table_results]
            if not result.complete
        ]
        if incomplete:
            reasons = "; ".join(
                result.incomplete_reason or result.doi
                for result in incomplete
            )
            raise RuntimeError(
                f"Extraction incomplete for {len(incomplete)} artifact(s): {reasons}"
            )

        total = sum(len(r.records) for r in text_results) + sum(len(r.records) for r in table_results)
        logger.info(
            f"Extracted {total} total records "
            f"(text: {sum(len(r.records) for r in text_results)}, "
            f"table: {sum(len(r.records) for r in table_results)})"
        )

        for paper, _ in paired:
            checkpoint_db.upsert(PaperCheckpoint(
                doi=paper.paper_uid or paper.doi,
                status=PipelineStatus.EXTRACTED,
                extract_time=datetime.now(timezone.utc),
            ))

    elif stage == "assemble":
        _run_assemble(config, checkpoint_db)

    elif stage == "audit":
        _run_audit(config)


# ---------------------------------------------------------------------------
# Assemble — shared between concurrent and sequential paths
# ---------------------------------------------------------------------------

def _run_assemble(config, checkpoint_db):
    """Run the global assemble stage."""
    from src.assembler import run as assemble_run
    from src.schema import PaperCheckpoint, PaperMetadata, ParsedPaper, PipelineStatus

    output_root = Path(config["output_dir"])
    parsed_papers = load_parsed_papers(output_root / "parsed")
    manifest_entries = _read_manifest_entries(output_root / "manifest.json")
    loaded_uids = {paper.paper_uid for paper in parsed_papers if paper.paper_uid}
    for uid, entry in sorted((manifest_entries or {}).items()):
        if uid in loaded_uids:
            continue
        # Keep missing parses in the coverage universe. This refreshes all
        # exports and then lets strict coverage fail with an explicit ledger
        # row, rather than silently certifying a subset of the manifest.
        parsed_papers.append(ParsedPaper(
            paper_uid=uid,
            parse_complete=False,
            parse_error=str(entry.get("parse_error", "missing parsed artifact")),
            doi=str(entry.get("doi", "") or f"local/{entry.get('slug', uid)}"),
            pdf_path=str(entry.get("pdf_path", "")),
            metadata=PaperMetadata(
                doi=str(entry.get("doi", "") or f"local/{entry.get('slug', uid)}"),
                title=str(entry.get("title", "")),
                year=entry.get("year"),
            ),
        ))
    extraction_results = load_extraction_results(
        output_root / "extracted", parsed_papers=parsed_papers
    )
    # A DOI is not a safe dictionary key: parser errors and legitimate source
    # reuse can assign one DOI to more than one PDF. Preserve every paper and
    # let the assembler join primarily by content-addressed paper_uid.
    papers_dict = {
        (p.paper_uid or f"{p.doi}#{index}"): p
        for index, p in enumerate(parsed_papers)
    }
    db_path = assemble_run(
        extraction_results=extraction_results,
        parsed_papers=papers_dict,
        output_dir=output_root / "database",
        config=config,
    )
    logger.info(f"Database assembled at {db_path}")

    for er in extraction_results:
        checkpoint_db.upsert(PaperCheckpoint(
            doi=er.paper_uid or er.doi,
            status=PipelineStatus.ASSEMBLED,
            assemble_time=datetime.now(timezone.utc),
        ))


def _run_audit(config):
    """Run the data quality audit stage."""
    from src.audit import run_audit
    csv_path = Path(config["output_dir"]) / "database" / "dielectric_properties.csv"
    output_path = Path(config["output_dir"]) / "database" / "audit_report.csv"
    run_audit(csv_path=csv_path, output_path=output_path, use_llm=False)


# ---------------------------------------------------------------------------
# Display helpers (unchanged)
# ---------------------------------------------------------------------------

def print_status(checkpoint_db, pdf_files):
    summary = checkpoint_db.summary() if checkpoint_db is not None else {}
    total_cost = checkpoint_db.total_cost() if checkpoint_db is not None else 0.0
    print(f"\n{'='*50}")
    print("  agentXtract Pipeline Status")
    print(f"{'='*50}")
    print(f"  PDFs in test corpus: {len(pdf_files)}")
    for status, count in sorted(summary.items(), key=lambda x: x[0].value):
        print(f"  {status.value:.<30} {count}")
    print(f"  {'Total API cost':.<30} ${total_cost:.4f}")
    print(f"{'='*50}\n")


def print_pipeline_summary(checkpoint_db, stage_times: dict[str, float]) -> None:
    W = 56
    print(f"\n{'='*W}")
    print("  Pipeline Run Summary")
    print(f"{'='*W}")

    rows = checkpoint_db._conn.execute(
        """SELECT doi,
                  SUM(input_tokens)  AS inp,
                  SUM(output_tokens) AS out,
                  SUM(cost_usd)      AS cost
           FROM costs GROUP BY doi ORDER BY doi"""
    ).fetchall()

    if rows:
        print(f"  {'Paper':<28} {'In tok':>8} {'Out tok':>8} {'Cost':>8}")
        print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8}")
        tot_in = tot_out = tot_cost = 0
        for r in rows:
            label = r["doi"].split("/")[-1][:27]
            print(f"  {label:<28} {r['inp']:>8,} {r['out']:>8,} ${r['cost']:>7.4f}")
            tot_in += r["inp"]; tot_out += r["out"]; tot_cost += r["cost"]
        print(f"  {'TOTAL':<28} {tot_in:>8,} {tot_out:>8,} ${tot_cost:>7.4f}")

    stage_rows = checkpoint_db._conn.execute(
        """SELECT stage,
                  SUM(input_tokens)  AS inp,
                  SUM(output_tokens) AS out,
                  SUM(cost_usd)      AS cost
           FROM costs GROUP BY stage ORDER BY stage"""
    ).fetchall()

    if stage_rows:
        print(f"\n  {'Stage':<28} {'In tok':>8} {'Out tok':>8} {'Cost':>8}")
        print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8}")
        for r in stage_rows:
            print(f"  {r['stage']:<28} {r['inp']:>8,} {r['out']:>8,} ${r['cost']:>7.4f}")

    if stage_times:
        print(f"\n  {'Stage':<28} {'Time':>8}")
        print(f"  {'-'*28} {'-'*8}")
        for stage, secs in stage_times.items():
            print(f"  {stage:<28} {secs:>7.1f}s")
        print(f"  {'TOTAL':<28} {sum(stage_times.values()):>7.1f}s")

    print(f"{'='*W}\n")


def estimate_cost(pdf_files, config):
    n = len(pdf_files)
    use_batch = config.get("use_batch_api", False)
    discount = config.get("batch_discount", 0.5) if use_batch else 1.0
    screen_cost = n * 0.005 * discount
    text_cost = n * 0.10 * discount
    table_cost = n * 0.15 * discount
    total = screen_cost + text_cost + table_cost
    mode = "BATCH (50% off)" if use_batch else "REAL-TIME"
    print(f"\n{'='*50}")
    print(f"  Cost Estimate for {n} papers [{mode}]")
    print(f"{'='*50}")
    print(f"  Screening (Haiku):              ${screen_cost:.2f}")
    print(f"  Text extraction (Sonnet):       ${text_cost:.2f}")
    print(f"  Table extraction (Haiku):       ${table_cost:.2f}")
    print(f"  {'TOTAL':.<30} ${total:.2f}")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# Loader helpers (unchanged, used by assemble and sequential fallback)
# ---------------------------------------------------------------------------

def _read_manifest_entries(manifest_path: Path) -> dict | None:
    """Read a manifest, distinguishing a missing file from a valid empty one.

    A missing manifest enables legacy bare-directory loading. A present but
    corrupt manifest cannot safely degrade to "load everything", because that
    silently reintroduces stale artifacts from unrelated corpora.
    """
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            entries = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Manifest is unreadable: {manifest_path}") from exc
    if not isinstance(entries, dict) or not all(
        isinstance(uid, str) and isinstance(entry, dict)
        for uid, entry in entries.items()
    ):
        raise RuntimeError(
            f"Manifest must map string paper uids to JSON objects: {manifest_path}"
        )
    return entries


def load_parsed_papers(parsed_dir):
    """Load this corpus's parsed papers, ignoring artifacts from earlier runs.

    ``data/parsed`` can accumulate one file per naming scheme a project has
    used — ``sample.json``, ``2. sample.json``, ``2__sample.json`` and the current
    ``<uid>__2_sample.json`` may all describe the same PDF. Loading every file made
    one paper enter the pipeline several times under different DOIs, which
    multiplied extraction cost, produced artifacts under names carrying no uid,
    and left the assembler picking between rival copies of the same paper.

    Keeping only files whose uid is in the manifest collapses each paper back
    to one identity. Without a manifest the filter is skipped, so a directory
    of hand-placed parses still works.
    """
    from src.paper_id import split_artifact_name
    from src.schema import ParsedPaper

    parsed_dir = Path(parsed_dir)
    manifest_path = parsed_dir.parent / "manifest.json"
    entries = _read_manifest_entries(manifest_path)

    papers = []
    if entries is not None:
        for uid, entry in sorted(entries.items()):
            pdf_path = Path(entry.get("pdf_path", ""))
            stem = entry.get("slug", "") or pdf_path.stem
            paper = _load_one_parsed_paper(parsed_dir, stem, uid)
            if paper is None:
                logger.warning(f"No parsed artifact for manifest paper {uid} ({stem})")
                continue
            paper.paper_uid = uid
            if str(pdf_path):
                paper.pdf_path = str(pdf_path.resolve())
            papers.append(paper)
        return papers

    for f in sorted(parsed_dir.glob("*.json")):
        if f.name.startswith("screener_"):
            continue
        uid, _slug = split_artifact_name(f.stem)
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        meta = data.get("metadata", {})
        tr = meta.get("temperature_range_c")
        if tr is not None and (tr[0] is None or tr[1] is None):
            meta["temperature_range_c"] = None
        papers.append(ParsedPaper.model_validate(data))

    return papers


def load_screener_results(parsed_dir):
    from src.schema import ScreenerResult
    results_file = parsed_dir / "screener_results.json"
    if not results_file.exists():
        return []
    with open(results_file, encoding="utf-8") as f:
        return [ScreenerResult.model_validate(r) for r in json.load(f)]


def load_extraction_results(extracted_dir, parsed_papers=None):
    """Load this corpus's extraction artifacts.

    Two rules, both learned the hard way:

    * **Key on (paper, artifact kind), not on DOI.** A paper's table pass and
      text pass share a DOI, so keying on DOI alone silently discarded one of
      them — a paper whose data came from prose lost its records whenever a
      table artifact existed too.
    * **Ignore artifacts outside the current manifest.** ``data/extracted``
      accumulates files from earlier runs and earlier corpus layouts, whose
      names encode a path rather than a content hash. Those carry a different
      DOI, so they survived deduplication and were assembled alongside the
      current run — which is how a paper with 128 real records came out with
      256, half of them stale.

    Artifacts are kept only when their uid appears in the manifest written by
    the parse stage. When no manifest exists the filter is skipped, so a bare
    extracted directory still assembles.
    """
    from src.paper_id import split_artifact_name
    from src.schema import ExtractionResult

    manifest_path = Path(extracted_dir).parent / "manifest.json"
    entries = _read_manifest_entries(manifest_path)
    known_uids = set(entries or {})

    from collections import defaultdict

    from src.paper_id import normalize_paper_id

    parsed_papers = parsed_papers or []
    by_doi = defaultdict(list)
    by_name = defaultdict(list)
    for paper in parsed_papers:
        if paper.paper_uid:
            # ParsedPaper.doi is the ingestion identifier and is often a
            # ``local/...`` fallback.  The extractor, however, writes the DOI
            # recovered into metadata.  Index both so old artifacts can be
            # migrated to their content UID without guessing from filenames.
            doi_values = {
                (paper.doi or "").strip().lower(),
                (getattr(paper.metadata, "doi", "") or "").strip().lower(),
            }
            for doi in doi_values - {""}:
                by_doi[doi].append(paper.paper_uid)
            by_name[normalize_paper_id(paper.pdf_path)].append(paper.paper_uid)

    results: dict[tuple[str, str], ExtractionResult] = {}
    result_scores: dict[tuple[str, str], int] = {}
    skipped_stale = 0
    for f in sorted(Path(extracted_dir).glob("*.json")):
        filename_uid, _slug = split_artifact_name(f.stem)
        try:
            with open(f, encoding="utf-8") as fh:
                er = ExtractionResult.model_validate(json.load(fh))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning(f"Skipping corrupt extraction file {f.name}: {e}")
            continue
        uid = filename_uid or er.paper_uid
        score = 2 if uid else 0
        if not uid:
            doi_matches = by_doi.get(er.doi.strip().lower(), [])
            if len(doi_matches) == 1:
                uid = doi_matches[0]
                score = 1
            else:
                record_names = {
                    normalize_paper_id(record.paper_id)
                    for record in er.records if record.paper_id
                }
                name_matches = {
                    candidate
                    for name in record_names
                    for candidate in by_name.get(name, [])
                }
                if len(name_matches) == 1:
                    uid = name_matches.pop()
                    score = 1
        if entries is not None and uid not in known_uids:
            skipped_stale += 1
            continue
        if uid:
            er.paper_uid = uid
            for record in er.records:
                record.paper_uid = uid
        kind = "text" if f.stem.endswith("_text") else "table"
        key = (uid or er.doi, kind)
        if score >= result_scores.get(key, -1):
            results[key] = er
            result_scores[key] = score

    if skipped_stale:
        logger.info(
            f"Ignored {skipped_stale} extraction artifacts from earlier runs "
            f"(uid not in the current manifest)"
        )
    return list(results.values())


def _match_screener_result(paper, screener_results):
    """Match screening metadata by content UID, with an unambiguous DOI fallback."""
    if paper.paper_uid:
        uid_matches = [
            result for result in screener_results
            if result.paper_uid == paper.paper_uid
        ]
        if len(uid_matches) == 1:
            return uid_matches[0]
        if len(uid_matches) > 1:
            logger.warning(
                "Multiple screener results for paper uid %s; using the last one",
                paper.paper_uid,
            )
            return uid_matches[-1]
    doi_matches = [
        result for result in screener_results
        if result.doi == paper.doi and not result.paper_uid
    ]
    return doi_matches[0] if len(doi_matches) == 1 else None


def pair_papers_with_screener(papers, screener_results):
    from src.schema import Complexity, ExtractionPriority, ScreenerResult
    paired = []
    for p in papers:
        sr = _match_screener_result(p, screener_results)
        if sr is None:
            sr = ScreenerResult(
                paper_uid=p.paper_uid,
                doi=p.doi,
                estimated_records=0,
                data_sources=[],
                extraction_priority=ExtractionPriority.MEDIUM,
                complexity=Complexity.MODERATE,
                notes="Default; screener result not found.",
            )
        paired.append((p, sr))
    return paired


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    # Rich status messages contain Unicode symbols. Legacy Windows terminals
    # otherwise crash at shutdown under a cp1252 stdout encoding.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="agentXtract pipeline")
    parser.add_argument(
        "--stage",
        choices=STAGE_ORDER + ["all"],
        default="all",
        help="Which stage to run (default: all)",
    )
    parser.add_argument(
        "--file",
        type=str,
        action="append",
        default=None,
        help="Process one PDF; repeat this option to process a selected batch",
    )
    parser.add_argument("--dry-run", action="store_true", help="Estimate cost without API calls")
    parser.add_argument("--status", action="store_true", help="Print pipeline progress summary")
    parser.add_argument("--force", action="store_true", help="Reprocess papers even if already extracted")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Use immediate model calls instead of the discounted asynchronous batch API",
    )
    parser.add_argument(
        "--strict-coverage",
        action="store_true",
        help="Fail assembly when any selected paper produces zero records",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        return asyncio.run(run_pipeline(args))
    except Exception:
        logger.exception("Pipeline failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
