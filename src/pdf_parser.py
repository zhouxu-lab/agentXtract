"""PDF parser using Docling. Parses PDFs into structured JSON (sections, tables, figures)."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

from src.paper_id import Manifest, artifact_name, compute_uid
from src.schema import (
    PaperMetadata,
    ParsedFigure,
    ParsedPaper,
    ParsedSection,
    ParsedTable,
)
from src.utils import write_json_atomic

logger = logging.getLogger(__name__)


def extract_doi(text: str, pdf_path: Path) -> str:
    """Extract DOI from parsed text. Fall back to filename."""
    doi_pattern = r'10\.\d{4,}/[^\s,;"\')]+'
    match = re.search(doi_pattern, text[:5000])
    if match:
        return match.group(0).rstrip(".")
    return f"local/{pdf_path.stem}"


def _safe_stem(pdf_path: Path) -> str:
    return re.sub(r'[^\w\-]', '_', pdf_path.stem)


def _table_grid(table) -> tuple[list[list[str]], list[list[str]]]:
    """Split a Docling table into its header rows and its body rows.

    ``export_to_dataframe`` keeps a single header row, so a table labelled
    across two rows — a spanning "T (°C)" sitting over "20 30 40 50 60", say —
    loses the row that names the actual measurement conditions, and every data
    column becomes anonymous. Reading the cell grid directly keeps every header
    row, and repeats a spanned cell across the columns it covers so each column
    carries its full label.

    Returns ([], []) when the grid is unavailable, so the caller can fall back
    to the dataframe export.
    """
    data = getattr(table, "data", None)
    cells = getattr(data, "table_cells", None) if data is not None else None
    if not cells:
        return [], []

    n_rows = getattr(data, "num_rows", 0) or 0
    n_cols = getattr(data, "num_cols", 0) or 0
    for cell in cells:
        n_rows = max(n_rows, getattr(cell, "end_row_offset_idx", 0) or 0)
        n_cols = max(n_cols, getattr(cell, "end_col_offset_idx", 0) or 0)
    if n_rows <= 0 or n_cols <= 0:
        return [], []

    grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    is_header = [False] * n_rows

    for cell in cells:
        text = str(getattr(cell, "text", "") or "").strip()
        r0 = getattr(cell, "start_row_offset_idx", 0) or 0
        r1 = getattr(cell, "end_row_offset_idx", r0 + 1) or (r0 + 1)
        c0 = getattr(cell, "start_col_offset_idx", 0) or 0
        c1 = getattr(cell, "end_col_offset_idx", c0 + 1) or (c0 + 1)
        for r in range(max(r0, 0), min(r1, n_rows)):
            if getattr(cell, "column_header", False):
                is_header[r] = True
            for c in range(max(c0, 0), min(c1, n_cols)):
                grid[r][c] = text

    # Only the leading block counts as the header. A header-styled row further
    # down is a section divider, and extraction treats it as data.
    n_header = 0
    while n_header < n_rows and is_header[n_header]:
        n_header += 1

    return grid[:n_header], grid[n_header:]


def _parse_with_docling(pdf_path: Path, output_dir: Path, uid: str = "") -> ParsedPaper:
    """Parse a PDF using Docling."""
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    full_text = doc.export_to_markdown()
    doi = extract_doi(full_text, pdf_path)
    uid = uid or compute_uid(pdf_path)
    safe_name = artifact_name(uid, pdf_path.stem)

    # Extract sections
    sections: list[ParsedSection] = []
    for item in doc.iterate_items():
        item_obj = item[0] if isinstance(item, tuple) else item
        label = getattr(item_obj, 'label', None)
        label_str = str(label) if label else ''
        text_content = getattr(item_obj, 'text', '') or ''

        if 'heading' in label_str.lower() or 'title' in label_str.lower():
            level = 2 if 'section' in label_str.lower() else 1
            sections.append(ParsedSection(heading=text_content, text="", level=level))
        elif 'paragraph' in label_str.lower() or 'text' in label_str.lower():
            if sections:
                sections[-1].text += ("\n" if sections[-1].text else "") + text_content
            else:
                sections.append(ParsedSection(heading="", text=text_content, level=1))

    # Extract tables
    tables: list[ParsedTable] = []
    table_dir = output_dir / "table_images"
    table_dir.mkdir(parents=True, exist_ok=True)

    for i, table in enumerate(doc.tables):
        table_id = f"table_{i+1}"
        caption = ""
        if hasattr(table, 'captions') and table.captions:
            caption = str(table.captions[0].text) if hasattr(table.captions[0], 'text') else str(table.captions[0])

        headers, rows = _table_grid(table)
        if not headers and not rows:
            try:
                df = table.export_to_dataframe()
                headers = [[str(c) for c in df.columns]]
                rows = [[str(v) for v in row] for row in df.values]
            # Docling backends can raise provider-specific exceptions while
            # exporting a malformed table; the empty grid is the fallback.
            except Exception:  # noqa: BLE001
                headers = []
                rows = []

        tables.append(ParsedTable(
            table_id=table_id,
            caption=caption,
            headers=headers,
            rows=rows,
        ))

    # Extract figures
    figures: list[ParsedFigure] = []
    fig_dir = output_dir / "figure_images"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for i, picture in enumerate(doc.pictures):
        figure_id = f"figure_{i+1}"
        caption = ""
        if hasattr(picture, 'captions') and picture.captions:
            caption = str(picture.captions[0].text) if hasattr(picture.captions[0], 'text') else str(picture.captions[0])

        image_path = None
        try:
            img = picture.get_image(result.document)
            if img is not None:
                img_file = fig_dir / f"{safe_name}_{figure_id}.png"
                img.save(str(img_file))
                image_path = str(img_file)
        # Figure rendering is optional and backend exception types vary.
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Could not save figure image: {e}")

        figures.append(ParsedFigure(
            figure_id=figure_id,
            caption=caption,
            image_path=image_path,
        ))

    title = ""
    if sections and sections[0].heading:
        title = sections[0].heading
    elif pdf_path.stem:
        title = pdf_path.stem

    metadata = PaperMetadata(doi=doi, title=title, discovery_source="local_pdf")

    return ParsedPaper(
        paper_uid=uid,
        doi=doi,
        metadata=metadata,
        sections=sections,
        tables=tables,
        figures=figures,
        full_text=full_text,
        pdf_path=str(pdf_path),
    )


async def run(
    pdf_paths: list[Path],
    config: dict,
    output_dir: Path,
    grobid_url: str = "",
) -> list[ParsedPaper]:
    """Parse all PDFs and save results as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[ParsedPaper] = []

    # The manifest is the single join table between papers and their records.
    manifest = Manifest(output_dir.parent / "manifest.json", strict=True)

    for pdf_path in pdf_paths:
        uid = compute_uid(pdf_path)
        safe_name = artifact_name(uid, pdf_path.stem)
        out_file = output_dir / f"{safe_name}.json"

        # Idempotency: skip if already parsed (parsed JSON exists)
        if out_file.exists():
            try:
                cached_text = await asyncio.to_thread(
                    out_file.read_text, encoding="utf-8"
                )
                paper = ParsedPaper.model_validate(json.loads(cached_text))
                logger.info(
                    f"  {pdf_path.name}: already parsed, loading from cache "
                    f"({len(paper.tables)} tables, {len(paper.sections)} sections)"
                )
                paper.paper_uid = uid
                # A cache may move with the repository or the same PDF may be
                # selected under a new name. The current corpus path is
                # authoritative; retaining the old absolute path breaks page
                # rendering and assigns the wrong human-readable paper id.
                paper.pdf_path = str(pdf_path.resolve())
                manifest.upsert(
                    uid, pdf_path=str(pdf_path), slug=pdf_path.stem,
                    doi=paper.doi, title=paper.metadata.title,
                    n_tables=len(paper.tables),
                )
                results.append(paper)
                continue
            # Any invalid cache object should trigger a clean parse rather than
            # aborting the rest of the corpus.
            except Exception as e:  # noqa: BLE001
                logger.warning(f"  {pdf_path.name}: cached parse JSON invalid ({e}), re-parsing")

        # Warn about large PDFs (>50 MB) — may be slow or cause memory issues
        try:
            size_mb = pdf_path.stat().st_size / (1024 * 1024)
            if size_mb > 50:
                logger.warning(f"  {pdf_path.name}: large file ({size_mb:.1f} MB) — parsing may be slow")
        except OSError:
            pass

        logger.info(f"Parsing {pdf_path.name}...")
        try:
            paper = _parse_with_docling(pdf_path, output_dir, uid=uid)

            write_json_atomic(out_file, paper.model_dump(mode="json"))

            logger.info(
                f"  -> {paper.doi}: {len(paper.sections)} sections, "
                f"{len(paper.tables)} tables, {len(paper.figures)} figures"
            )
            manifest.upsert(
                uid, pdf_path=str(pdf_path), slug=pdf_path.stem,
                doi=paper.doi, title=paper.metadata.title,
                n_tables=len(paper.tables),
            )
            results.append(paper)

        except Exception as e:
            logger.exception(f"  Failed to parse {pdf_path.name}")
            manifest.upsert(
                uid, pdf_path=str(pdf_path), slug=pdf_path.stem,
                parse_error=str(e)[:300],
            )

    manifest.save()
    return results
