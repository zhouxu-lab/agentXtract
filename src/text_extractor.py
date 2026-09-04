"""Text extractor — extracts dielectric records from Results section text only.

Uses source discrimination to reject cited literature values.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

from src.paper_id import artifact_name, normalize_paper_id
from src.schema import (
    DielectricRecord,
    ExtractionResult,
    ParsedPaper,
    ScreenerResult,
)
from src.utils import (
    call_llm,
    load_skill,
    run_batch,
    strip_code_fences,
    write_json_atomic,
)

logger = logging.getLogger(__name__)


# Results sections longer than this are split into overlapping windows rather
# than truncated. A hard cut at 10 000 characters silently discarded the
# numerical results of long papers, because measured values usually appear
# after several pages of narrative.
TEXT_WINDOW_CHARS = 24000
TEXT_WINDOW_OVERLAP = 2000


def _window_results_text(results_text: str) -> list[str]:
    """Split a long Results section into overlapping windows.

    Overlap keeps a sentence that straddles a boundary readable in at least
    one window, so a value is not lost at the seam.
    """
    if len(results_text) <= TEXT_WINDOW_CHARS:
        return [results_text]
    windows, start = [], 0
    step = TEXT_WINDOW_CHARS - TEXT_WINDOW_OVERLAP
    while start < len(results_text):
        windows.append(results_text[start:start + TEXT_WINDOW_CHARS])
        start += step
    logger.info(
        f"  Results section is {len(results_text)} chars — "
        f"split into {len(windows)} overlapping windows "
        f"(previously truncated at 10000)"
    )
    return windows


def _build_text_prompt(paper: ParsedPaper, window: str | None = None) -> str:
    parts = [
        f"Paper DOI: {paper.doi}",
        f"Title: {paper.metadata.title}",
        f"\nThis paper's original materials: {paper.metadata.primary_materials}",
        f"This paper measured at frequencies: {paper.metadata.measurement_frequencies_mhz} MHz",
    ]

    # Extract Results section only (where original data is reported)
    # Fall back to full text if no Results section found
    results_text = window if window is not None else _extract_results_section(paper)
    parts.append(f"\n--- RESULTS SECTION ---\n{results_text}")

    return "\n".join(parts)


def _extract_results_section(paper: ParsedPaper) -> str:
    """Extract text from Results and Discussion sections only."""
    results_keywords = {"results", "result", "results and discussion",
                        "discussion", "findings"}
    skip_keywords = {"introduction", "literature", "background", "references",
                     "acknowledgment", "acknowledgement"}

    relevant_sections = []
    for section in paper.sections:
        heading_lower = section.heading.lower().strip()
        if any(kw in heading_lower for kw in skip_keywords):
            continue
        if any(kw in heading_lower for kw in results_keywords):
            relevant_sections.append(section)

    if relevant_sections:
        return "\n\n".join(f"## {s.heading}\n{s.text}" for s in relevant_sections)

    # Fallback: if no sections detected (flat text), use full text
    # but warn about it
    logger.warning("  No Results section detected — using full text for text extraction")
    return paper.full_text


def _post_filter_records(
    records: list[DielectricRecord],
    paper: ParsedPaper,
) -> list[DielectricRecord]:
    """Remove likely false positives from text extraction."""
    if not records:
        return records

    # Detect frequencies from paper metadata or table headers
    paper_frequencies: set[float] = set(paper.metadata.measurement_frequencies_mhz)
    if not paper_frequencies:
        for table in paper.tables:
            for header_row in table.headers:
                for h in header_row:
                    freq_matches = re.findall(r'(\d+(?:\.\d+)?)\s*MHz', str(h), re.IGNORECASE)
                    for fm in freq_matches:
                        paper_frequencies.add(float(fm))

    if not paper_frequencies:
        text_lower = paper.full_text[:5000].lower()
        for freq_str in re.findall(r'(\d+)\s*mhz', text_lower):
            f = float(freq_str)
            if f in {27, 40, 433, 915, 1800, 2450}:
                paper_frequencies.add(f)

    filtered = []
    for r in records:
        # Must have BOTH ε' and ε'' for text-extracted records
        if r.dielectric_constant is None or r.loss_factor is None:
            logger.debug(f"  Filtered (incomplete): {r.material_name}")
            continue

        # Check frequency
        if paper_frequencies and r.frequency_mhz is not None:
            matched = any(
                abs(r.frequency_mhz - pf) / pf < 0.05
                for pf in paper_frequencies
                if pf > 0
            )
            if not matched:
                logger.debug(f"  Filtered (wrong freq): {r.material_name} @ {r.frequency_mhz} MHz")
                continue

        filtered.append(r)

    removed = len(records) - len(filtered)
    if removed > 0:
        logger.info(f"  Post-filter removed {removed} text records (kept {len(filtered)})")

    return filtered


def _records_from_data(
    data: dict,
    paper: ParsedPaper,
    model: str,
) -> list[DielectricRecord]:
    """Validate one or more model payloads into consistently labelled rows."""
    records = []
    for rd in data.get("records", []):
        mb = rd.get("moisture_basis", "unknown")
        if mb and str(mb).lower() in ("wet", "dry", "unknown"):
            mb = str(mb).lower()
        else:
            mb = None
        records.append(DielectricRecord(
            material_name=rd.get("material_name", ""),
            dielectric_constant=rd.get("dielectric_constant"),
            loss_factor=rd.get("loss_factor"),
            loss_tangent=rd.get("loss_tangent"),
            frequency_mhz=rd.get("frequency_mhz"),
            temperature_c=rd.get("temperature_c"),
            moisture_content_pct=rd.get("moisture_content_pct"),
            moisture_basis=mb,
            salt_content=rd.get("salt_content"),
            electrical_conductivity_s_m=rd.get(
                "electrical_conductivity_s_m"
            ),
            measurement_method=rd.get("measurement_method"),
            doi=paper.doi,
            paper_uid=paper.paper_uid,
            paper_id=normalize_paper_id(paper.pdf_path),
            source_table=rd.get("source_table") or "Narrative text",
            source_location=rd.get("source_location", ""),
            extraction_source="text",
            extraction_model=model,
            data_provenance="measured_text",
            raw_text=rd.get("raw_text", ""),
        ))
    return _post_filter_records(records, paper)


async def run(
    paired: list[tuple[ParsedPaper, ScreenerResult]],
    config: dict,
    checkpoint_db=None,
    progress=None,
) -> list[ExtractionResult]:
    """Extract dielectric records from paper text (concurrent across papers)."""
    model_cfg = config.get("text_extractor", {})
    model = model_cfg.get("model", "claude-sonnet-4-6")
    max_tokens = model_cfg.get("max_tokens", 4096)
    concurrency = config.get("default_concurrency", 5)
    sem = asyncio.Semaphore(concurrency)

    system_prompt = load_skill("extractor_text")
    output_dir = Path(config.get("output_dir", "data")) / "extracted"
    output_dir.mkdir(parents=True, exist_ok=True)

    async def _extract_one(paper: ParsedPaper, screener: ScreenerResult) -> ExtractionResult:
        label = paper.doi.split("/")[-1]
        if progress:
            progress.make_status_fn(label)("text extraction...")
        logger.info(f"Text-extracting {paper.doi}...")
        windows = _window_results_text(_extract_results_section(paper))

        try:
            merged: dict = {"records": []}
            cost_entry = None
            failed_windows: list[str] = []
            for wi, window in enumerate(windows, 1):
                prompt = _build_text_prompt(paper, window=window)
                async with sem:
                    response_text, ce = await call_llm(
                        prompt=prompt, system=system_prompt,
                        model=model, max_tokens=max_tokens, config=config,
                    )
                if cost_entry is None:
                    cost_entry = ce
                else:
                    cost_entry.input_tokens += ce.input_tokens
                    cost_entry.output_tokens += ce.output_tokens
                    cost_entry.cost_usd += ce.cost_usd
                try:
                    part = json.loads(strip_code_fences(response_text))
                    if not isinstance(part, dict) or not isinstance(
                        part.get("records", []), list
                    ):
                        raise TypeError("response must contain a records array")
                except (json.JSONDecodeError, TypeError) as exc:
                    logger.warning(
                        f"  {paper.doi}: window {wi}/{len(windows)} returned "
                        f"unparseable JSON — skipping that window"
                    )
                    failed_windows.append(f"window {wi}: {type(exc).__name__}")
                    continue
                merged["records"].extend(part.get("records") or [])
                for k, v in part.items():
                    if k != "records" and k not in merged:
                        merged[k] = v
            data = merged

            records = _records_from_data(data, paper, model)

            er = ExtractionResult(
                paper_uid=paper.paper_uid,
                complete=not failed_windows,
                incomplete_reason="; ".join(failed_windows),
                doi=paper.doi,
                records=records,
                extraction_source="text",
                notes=data.get("notes", ""),
                token_usage={
                    "input": cost_entry.input_tokens,
                    "output": cost_entry.output_tokens,
                },
            )

            base = artifact_name(paper.paper_uid, Path(paper.pdf_path).stem
                                 if paper.pdf_path else paper.doi)
            out_file = output_dir / f"{base}_text.json"
            write_json_atomic(out_file, er.model_dump(mode="json"))

            if checkpoint_db:
                cost_entry.stage = "extract_text"
                cost_entry.doi = paper.doi
                checkpoint_db.add_cost(cost_entry)

            if progress:
                progress.advance_paper(f"{label} ({len(records)} records)")
            logger.info(f"  {paper.doi}: extracted {len(records)} text records (after filter)")
            return er

        except Exception as e:
            logger.exception(f"  Failed text extraction for {paper.doi}")
            return ExtractionResult(
                paper_uid=paper.paper_uid,
                complete=False,
                incomplete_reason=str(e),
                doi=paper.doi,
                extraction_source="text",
                notes=f"Extraction failed: {e}",
            )

    use_batch = config.get("use_batch_api", False)

    if use_batch and paired:
        # Batch mode uses exactly the same non-truncating windows as realtime
        # mode. Each window has an independent response and completeness flag.
        logger.info(f"Batch text-extracting {len(paired)} papers...")
        batch_requests = []
        paper_requests: list[tuple[ParsedPaper, list[str]]] = []
        for paper_index, (paper, _screener) in enumerate(paired):
            label = paper.doi.split("/")[-1]
            if progress:
                progress.make_status_fn(label)("queued for batch...")
            windows = _window_results_text(_extract_results_section(paper))
            custom_ids = []
            for window_index, window in enumerate(windows, 1):
                cid = f"text_{label}_{paper_index}_{window_index}"
                custom_ids.append(cid)
                batch_requests.append({
                    "custom_id": cid,
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "system": system_prompt,
                    "messages": [{
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": _build_text_prompt(paper, window=window),
                        }],
                    }],
                })
            paper_requests.append((paper, custom_ids))

        batch_results = await run_batch(
            batch_requests, config=config, poll_interval=30.0
        )
        results = []
        for paper, custom_ids in paper_requests:
            merged: dict = {"records": []}
            failed_windows = []
            total_input = total_output = 0
            total_cost = 0.0
            cost_entry = None
            for window_index, cid in enumerate(custom_ids, 1):
                result_pair = batch_results.get(cid)
                if not result_pair or not result_pair[0]:
                    failed_windows.append(f"window {window_index}: missing result")
                    continue
                response_text, current_cost = result_pair
                cost_entry = cost_entry or current_cost
                total_input += current_cost.input_tokens
                total_output += current_cost.output_tokens
                total_cost += current_cost.cost_usd
                try:
                    part = json.loads(strip_code_fences(response_text))
                    if not isinstance(part, dict) or not isinstance(
                        part.get("records", []), list
                    ):
                        raise TypeError("response must contain a records array")
                except (json.JSONDecodeError, TypeError) as exc:
                    failed_windows.append(
                        f"window {window_index}: {type(exc).__name__}"
                    )
                    continue
                merged["records"].extend(part.get("records") or [])
                for key, value in part.items():
                    if key != "records" and key not in merged:
                        merged[key] = value

            try:
                records = _records_from_data(merged, paper, model)
            except Exception as exc:
                logger.exception(
                    f"  Failed to process batch text result for {paper.doi}"
                )
                records = []
                failed_windows.append(f"record validation: {type(exc).__name__}")
            er = ExtractionResult(
                paper_uid=paper.paper_uid,
                complete=not failed_windows,
                incomplete_reason="; ".join(failed_windows),
                doi=paper.doi,
                records=records,
                extraction_source="text",
                notes=merged.get("notes", ""),
                token_usage={"input": total_input, "output": total_output},
            )
            base = artifact_name(
                paper.paper_uid,
                Path(paper.pdf_path).stem if paper.pdf_path else paper.doi,
            )
            write_json_atomic(
                output_dir / f"{base}_text.json", er.model_dump(mode="json")
            )
            if checkpoint_db and cost_entry:
                cost_entry.input_tokens = total_input
                cost_entry.output_tokens = total_output
                cost_entry.cost_usd = total_cost
                cost_entry.stage = "extract_text"
                cost_entry.doi = paper.doi
                checkpoint_db.add_cost(cost_entry)
            if progress:
                progress.advance_paper(f"{paper.doi.split('/')[-1]} ({len(records)} records)")
            logger.info(
                f"  {paper.doi}: extracted {len(records)} text records (batch)"
            )
            results.append(er)
        return results

    # ── Real-time mode (original) ──
    tasks = [_extract_one(paper, screener) for paper, screener in paired]
    results = await asyncio.gather(*tasks)
    return list(results)
