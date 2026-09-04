"""Screener — uses Haiku to classify tables, extract paper metadata, and estimate data density."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from src.schema import (
    Complexity,
    ExtractionPriority,
    PaperCheckpoint,
    ParsedPaper,
    PipelineStatus,
    ScreenerResult,
)
from src.utils import call_llm, load_skill, run_batch, strip_code_fences

logger = logging.getLogger(__name__)


def _build_screener_prompt(paper: ParsedPaper) -> str:
    parts = [f"Paper DOI: {paper.doi}", f"Title: {paper.metadata.title}"]

    if paper.metadata.abstract:
        parts.append(f"\nAbstract:\n{paper.metadata.abstract}")

    text_preview = paper.full_text[:3000]
    parts.append(f"\nText preview (first 3000 chars):\n{text_preview}")

    if paper.tables:
        parts.append(f"\nTables found: {len(paper.tables)}")
        for t in paper.tables:
            parts.append(f"  {t.table_id}: caption='{t.caption}'")
            if t.headers:
                parts.append(f"    headers: {t.headers[0][:10]}")
            rows = t.rows or [list(r.values()) for r in t.data_rows]
            if rows:
                parts.append(f"    rows: {len(rows)} data rows")
                if rows[0]:
                    parts.append(f"    first row: {rows[0][:10]}")

    if paper.figures:
        parts.append(f"\nFigures found: {len(paper.figures)}")
        for fig in paper.figures:
            parts.append(f"  {fig.figure_id}: caption='{fig.caption}'")

    return "\n".join(parts)


def _detect_frequencies_from_tables(paper: ParsedPaper) -> list[float]:
    """Detect measurement frequencies from table headers and captions."""
    import re
    freqs = set()
    common_ism = {13.56, 27, 27.12, 40, 40.68, 433, 915, 1800, 2450, 5800}

    for table in paper.tables:
        text = table.caption + " " + " ".join(
            str(h) for row in table.headers for h in row
        )
        # Match patterns like "915 MHz", "2450MHz", "2.45 GHz"
        for m in re.finditer(r'(\d+(?:\.\d+)?)\s*(MHz|GHz)', text, re.IGNORECASE):
            val = float(m.group(1))
            unit = m.group(2).upper()
            if unit == "GHZ":
                val *= 1000
            if 1 <= val <= 300000:
                freqs.add(val)

    # Also check paper text (first 5000 chars — typically Methods section)
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*(MHz|GHz)', paper.full_text[:5000], re.IGNORECASE):
        val = float(m.group(1))
        unit = m.group(2).upper()
        if unit == "GHZ":
            val *= 1000
        if val in common_ism:
            freqs.add(val)

    return sorted(freqs)


def _process_screener_response(
    paper: ParsedPaper,
    response_text: str,
    cost_entry,
    results: list[ScreenerResult],
    checkpoint_db,
    progress,
    label: str,
) -> None:
    """Parse screener LLM response, enrich metadata, append to results."""
    data = json.loads(strip_code_fences(response_text))

    metadata_keys = (
        "doi", "title", "authors", "year", "journal",
        "primary_materials", "measurement_frequencies_mhz",
        "temperature_range_c", "moisture_range_pct",
        "moisture_levels_pct", "moisture_basis", "data_tables",
        "equation_tables", "skip_tables", "measurement_method",
    )
    sr = ScreenerResult(
        paper_uid=paper.paper_uid,
        doi=paper.doi,
        estimated_records=data.get("estimated_records", 0),
        data_sources=data.get("data_sources", []),
        extraction_priority=ExtractionPriority(data.get("extraction_priority", "medium")),
        complexity=Complexity(data.get("complexity", "moderate")),
        has_equations=bool(data.get("has_equations", False)),
        figure_only=bool(data.get("figure_only", False)),
        notes=data.get("notes", ""),
        metadata={key: data[key] for key in metadata_keys if key in data},
    )

    _apply_screener_metadata(paper, sr)

    results.append(sr)
    if progress:
        progress.advance_paper(label)

    if checkpoint_db:
        cost_entry.stage = "screen"
        cost_entry.doi = paper.doi
        checkpoint_db.add_cost(cost_entry)
        checkpoint_db.upsert(PaperCheckpoint(
            doi=paper.paper_uid or paper.doi,
            status=PipelineStatus.SCREENED,
            screen_time=datetime.now(timezone.utc),
        ))

    logger.info(
        f"  {paper.doi}: ~{sr.estimated_records} records, "
        f"sources={sr.data_sources}, priority={sr.extraction_priority.value}, "
        f"complexity={sr.complexity.value}, "
        f"data_tables={paper.metadata.data_tables}, "
        f"skip_tables={paper.metadata.skip_tables}"
    )


def _apply_screener_metadata(
    paper: ParsedPaper,
    result: ScreenerResult,
) -> None:
    """Replay cached screening metadata onto a parsed paper."""
    data = result.metadata
    if data.get("doi"):
        paper.doi = data["doi"]
        paper.metadata.doi = data["doi"]
        result.doi = data["doi"]
    if data.get("title"):
        paper.metadata.title = data["title"]
    if data.get("authors"):
        paper.metadata.authors = data["authors"]
    if data.get("year"):
        paper.metadata.year = data["year"]
    if data.get("journal"):
        paper.metadata.journal = data["journal"]
    if "primary_materials" in data:
        paper.metadata.primary_materials = data.get("primary_materials") or []
    if "measurement_frequencies_mhz" in data:
        paper.metadata.measurement_frequencies_mhz = (
            data.get("measurement_frequencies_mhz") or []
        )
    t_range = data.get("temperature_range_c")
    if t_range and len(t_range) == 2:
        paper.metadata.temperature_range_c = tuple(t_range)
    # Moisture information drives the evaluation grid for papers that report
    # empirical models rather than individual measurements. Without it, model
    # points can only be generated on a generic grid.
    m_range = data.get("moisture_range_pct")
    if m_range and len(m_range) == 2:
        paper.metadata.moisture_range_pct = tuple(m_range)
    m_levels = data.get("moisture_levels_pct") or []
    if isinstance(m_levels, list):
        paper.metadata.moisture_levels_pct = [
            float(x) for x in m_levels
            if isinstance(x, (int, float))
        ]
    if data.get("moisture_basis") in ("wet", "dry", "unknown"):
        paper.metadata.moisture_basis = data["moisture_basis"]
    if "data_tables" in data:
        paper.metadata.data_tables = data.get("data_tables") or []
    if "equation_tables" in data:
        paper.metadata.equation_tables = data.get("equation_tables") or []
    if "skip_tables" in data:
        paper.metadata.skip_tables = data.get("skip_tables") or []
    paper.metadata.estimated_total_records = result.estimated_records
    if "measurement_method" in data:
        paper.metadata.measurement_method = data.get("measurement_method")

    # Fallback: extract DOI via regex if screener didn't find one
    if not data.get("doi"):
        doi_match = re.search(r'10\.\d{4,}/[^\s,;"\')]+', paper.full_text)
        if doi_match:
            real_doi = doi_match.group(0).rstrip(".")
            paper.doi = real_doi
            paper.metadata.doi = real_doi
            result.doi = real_doi
            logger.info(f"  Fallback DOI extraction: {real_doi}")

    # Fallback: detect frequencies from table headers if screener missed them
    if not paper.metadata.measurement_frequencies_mhz:
        detected_freqs = _detect_frequencies_from_tables(paper)
        if detected_freqs:
            paper.metadata.measurement_frequencies_mhz = detected_freqs
            logger.info(f"  Fallback frequency detection: {detected_freqs}")


async def run(
    papers: list[ParsedPaper],
    config: dict,
    checkpoint_db=None,
    progress=None,
) -> list[ScreenerResult]:
    """Screen all parsed papers using Haiku."""
    model_cfg = config.get("screener", {})
    model = model_cfg.get("model", "claude-haiku-4-5-20251001")
    max_tokens = model_cfg.get("max_tokens", 1024)

    system_prompt = load_skill("screener")
    results: list[ScreenerResult] = []

    # Load existing screener results so we can skip already-screened papers
    parsed_dir_path = Path(config.get("output_dir", "data")) / "parsed"
    results_file_path = parsed_dir_path / "screener_results.json"
    existing_by_uid: dict[str, ScreenerResult] = {}
    existing_by_doi: dict[str, list[ScreenerResult]] = {}
    if results_file_path.exists():
        try:
            cached_results = json.loads(
                await asyncio.to_thread(
                    results_file_path.read_text, encoding="utf-8"
                )
            )
            for r in cached_results:
                sr = ScreenerResult.model_validate(r)
                if sr.paper_uid:
                    existing_by_uid[sr.paper_uid] = sr
                else:
                    existing_by_doi.setdefault(sr.doi, []).append(sr)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            raise RuntimeError(
                f"Screener cache is unreadable: {results_file_path}"
            ) from e

    # Separate cached vs. needs-screening papers
    force = bool(config.get("force_reprocess", False))
    to_screen: list[ParsedPaper] = []
    requested_doi_counts: dict[str, int] = {}
    for paper in papers:
        requested_doi_counts[paper.doi] = requested_doi_counts.get(paper.doi, 0) + 1
    for paper in papers:
        label = paper.doi.split("/")[-1]
        cached = existing_by_uid.get(paper.paper_uid) if paper.paper_uid else None
        legacy_matches = existing_by_doi.get(paper.doi, [])
        if (
            cached is None
            and requested_doi_counts.get(paper.doi) == 1
            and len(legacy_matches) == 1
        ):
            cached = legacy_matches[0]
        already_enriched = bool(
            paper.metadata.primary_materials
            or paper.metadata.data_tables
            or paper.metadata.equation_tables
            or paper.metadata.skip_tables
            or paper.metadata.measurement_frequencies_mhz
            or paper.metadata.measurement_method
            or paper.metadata.estimated_total_records
        )
        if (
            cached is not None
            and cached.complete
            and not force
            and (cached.metadata or already_enriched)
        ):
            if cached.metadata:
                _apply_screener_metadata(paper, cached)
            logger.info(f"  {paper.doi}: already screened, skipping (cached: {cached.estimated_records} records)")
            results.append(cached)
            if progress:
                progress.advance_paper(label)
        else:
            to_screen.append(paper)

    use_batch = config.get("use_batch_api", False)

    if use_batch and to_screen:
        # ── Batch mode: submit all screening requests at once for 50% discount ──
        logger.info(f"Batch screening {len(to_screen)} papers...")
        batch_requests = []
        for idx, paper in enumerate(to_screen):
            label = paper.doi.split("/")[-1]
            if progress:
                progress.make_status_fn(label)("queued for batch...")
            prompt = _build_screener_prompt(paper)
            content = [{"type": "text", "text": prompt}]
            batch_requests.append({
                "custom_id": f"screen_{label}_{idx}",
                "model": model,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "system": system_prompt,
                "messages": [{"role": "user", "content": content}],
            })

        batch_results = await run_batch(batch_requests, config=config, poll_interval=30.0)

        for idx, paper in enumerate(to_screen):
            label = paper.doi.split("/")[-1]
            custom_id = f"screen_{label}_{idx}"
            result_pair = batch_results.get(custom_id)
            if not result_pair or not result_pair[0]:
                logger.error(f"  Batch result missing for {paper.doi}")
                results.append(ScreenerResult(
                    paper_uid=paper.paper_uid,
                    complete=False,
                    incomplete_reason="Batch screening returned no result",
                    doi=paper.doi, estimated_records=0,
                    extraction_priority=ExtractionPriority.MEDIUM,
                    complexity=Complexity.MODERATE,
                    notes="Batch screening failed: no result returned.",
                ))
                continue

            response_text, cost_entry = result_pair
            try:
                _process_screener_response(
                    paper, response_text, cost_entry, results,
                    checkpoint_db, progress, label,
                )
            # Preserve per-paper failure isolation for arbitrary malformed
            # provider responses and downstream validation errors.
            except Exception as e:  # noqa: BLE001
                logger.error(f"  Failed to process batch screen result for {paper.doi}: {e}")
                results.append(ScreenerResult(
                    paper_uid=paper.paper_uid,
                    complete=False,
                    incomplete_reason=str(e),
                    doi=paper.doi, estimated_records=0,
                    extraction_priority=ExtractionPriority.MEDIUM,
                    complexity=Complexity.MODERATE,
                    notes=f"Screening failed: {e}",
                ))
    else:
        # ── Real-time mode: process papers one at a time ──
        for paper in to_screen:
            label = paper.doi.split("/")[-1]
            if progress:
                progress.make_status_fn(label)("screening...")
            logger.info(f"Screening {paper.doi}...")
            prompt = _build_screener_prompt(paper)

            try:
                response_text, cost_entry = await call_llm(
                    prompt=prompt,
                    system=system_prompt,
                    model=model,
                    max_tokens=max_tokens,
                    config=config,
                )
                _process_screener_response(
                    paper, response_text, cost_entry, results,
                    checkpoint_db, progress, label,
                )
            # Provider SDKs expose several unrelated failure hierarchies; all
            # are converted into an explicit incomplete result for this paper.
            except Exception as e:  # noqa: BLE001
                logger.error(f"  Failed to screen {paper.doi}: {e}")
                results.append(ScreenerResult(
                    paper_uid=paper.paper_uid,
                    complete=False,
                    incomplete_reason=str(e),
                    doi=paper.doi,
                    estimated_records=0,
                    extraction_priority=ExtractionPriority.MEDIUM,
                    complexity=Complexity.MODERATE,
                    notes=f"Screening failed: {e}",
                ))

    # Save results — merge with any existing results so prior papers aren't lost
    parsed_dir = Path(config.get("output_dir", "data")) / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    results_file = parsed_dir / "screener_results.json"

    existing: dict[str, dict] = {}
    if results_file.exists():
        try:
            saved_results = json.loads(
                await asyncio.to_thread(results_file.read_text, encoding="utf-8")
            )
            for r in saved_results:
                key = (
                    f"uid:{r['paper_uid']}" if r.get("paper_uid")
                    else f"doi:{r['doi']}"
                )
                existing[key] = r
        except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            raise RuntimeError(
                f"Refusing to overwrite unreadable screener cache: {results_file}"
            ) from e

    for r in results:
        key = f"uid:{r.paper_uid}" if r.paper_uid else f"doi:{r.doi}"
        existing[key] = r.model_dump(mode="json")

    temporary_results_file = results_file.with_suffix(".json.tmp")
    serialized_results = json.dumps(
        [existing[key] for key in sorted(existing)],
        indent=2,
        ensure_ascii=False,
    )

    def _write_and_replace_results():
        try:
            temporary_results_file.write_text(serialized_results, encoding="utf-8")
            temporary_results_file.replace(results_file)
        except (OSError, UnicodeError):
            try:
                temporary_results_file.unlink(missing_ok=True)
            except OSError as cleanup_error:
                logger.debug(
                    "Could not remove failed screener temporary file: %s",
                    cleanup_error,
                )
            raise

    write_operation = asyncio.create_task(
        asyncio.to_thread(_write_and_replace_results)
    )
    try:
        await asyncio.shield(write_operation)
    except asyncio.CancelledError:
        # Finish the atomic cache replacement before cancellation can trigger
        # a caller retry or leave a stale temporary file behind.
        try:
            await write_operation
        except Exception as exc:  # noqa: BLE001
            logger.debug("Screener cache write failed during cancellation: %s", exc)
        raise
    logger.info(f"Saved screener results to {results_file} ({len(existing)} total)")

    return results
