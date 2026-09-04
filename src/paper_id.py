"""Stable, content-addressed identity for source papers.

Why this exists
---------------
Identity used to be derived from the PDF filename, and the derivation was
lossy and stage-dependent:

    parse   wrote  parsed/07__Sample_Source.json  (non-word chars -> '_')
    screen  wrote  parsed/07. Sample Source.json  (raw stem)
    doi     became local/07. Sample Source
                or local/07  Sample Source        ('_' -> ' ' round trip)
                or the real DOI when doi_map had an entry

The same paper therefore appeared under several keys at once. Checkpoint
deduplication failed, extraction output written under one key was not found
under another, stale results from an earlier corpus that numbered its PDFs
differently were picked up during assembly, and paper metadata failed to join
onto extracted records.

The fix is to identify a paper by its *content*. A content hash is stable
across folders, renumbering and re-runs, and two copies of the same PDF in
different corpora collapse onto one identity automatically.

Filenames keep a human-readable slug alongside the hash so the working
directory stays browsable:

    parsed/9f2a1c4b7e30__07_Sample_Source.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

UID_LENGTH = 12
_UID_RE = re.compile(rf"^[0-9a-f]{{{UID_LENGTH}}}$")
_ARTIFACT_RE = re.compile(rf"^([0-9a-f]{{{UID_LENGTH}}})__(.+)$")


_LEADING_NUMBER_RE = re.compile(r"^\s*\d+\s*[.)\-]\s*")


def normalize_paper_id(pdf_path: str | Path) -> str:
    """Human-readable paper key, stable across platforms and corpus numbering.

    Deriving this with ``Path(pdf_path).stem`` breaks in two ways that both
    silently poison the join between extracted records and the gold standard:

    * A path recorded on Windows carries backslashes. ``pathlib`` on Linux
      treats those as ordinary characters, so the "stem" comes back as the
      whole path — ``data\\corpus\\set_a\\1. sample`` rather than
      ``sample``. Parsed JSON travels between machines, so this happens
      whenever a cached parse is reused on the other platform.
    * A corpus may number PDFs, so the same source is ``1. sample`` in one
      directory and ``7. sample`` after a renumber, and neither matches
      ``sample``.

    Splitting on both separators and dropping the leading number removes both
    failure modes. Identity for filenames and caching still comes from the
    content hash; this is only the readable key used for joins and reporting.
    """
    if not pdf_path:
        return ""
    text = str(pdf_path).replace("\\", "/")
    name = text.rsplit("/", 1)[-1]
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    name = _LEADING_NUMBER_RE.sub("", name)
    return " ".join(name.replace("_", " ").split()).strip().lower()


def slugify(name: str, max_length: int = 60) -> str:
    """Filesystem-safe, readable slug. Never used for identity."""
    slug = re.sub(r"[^\w\-]+", "_", str(name)).strip("_")
    slug = re.sub(r"_{2,}", "_", slug)
    return slug[:max_length] or "paper"


def compute_uid(pdf_path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Content hash of a PDF, truncated to UID_LENGTH hex characters.

    Falls back to hashing the absolute path if the file cannot be read, so
    the pipeline degrades rather than crashing. The fallback is logged
    because it reintroduces path dependence for that one paper.
    """
    path = Path(pdf_path)
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            while chunk := fh.read(chunk_size):
                digest.update(chunk)
    except OSError as e:
        logger.warning(
            f"Cannot read {path} for content hashing ({e}); "
            f"falling back to path-based uid"
        )
        digest = hashlib.sha256(str(path.resolve()).encode("utf-8"))
    return digest.hexdigest()[:UID_LENGTH]


def is_uid(value: Any) -> bool:
    return bool(value) and bool(_UID_RE.match(str(value)))


def artifact_name(uid: str, slug_source: str) -> str:
    """Base filename for any artifact belonging to a paper."""
    return f"{uid}__{slugify(slug_source)}"


def split_artifact_name(stem: str) -> tuple[str | None, str | None]:
    """Inverse of artifact_name. Returns (uid, slug) or (None, None)."""
    m = _ARTIFACT_RE.match(stem)
    return (m.group(1), m.group(2)) if m else (None, None)


def find_artifact(directory: Path, uid: str, suffix: str = ".json") -> Path | None:
    """Locate an artifact for ``uid`` regardless of its slug."""
    if not directory.exists():
        return None
    for f in directory.glob(f"{uid}__*{suffix}"):
        return f
    return None


# ---------------------------------------------------------------------------
# Manifest — the single join table between papers and their records
# ---------------------------------------------------------------------------

MANIFEST_NAME = "manifest.json"


class Manifest:
    """uid -> paper facts. Written at parse, read at assembly.

    This replaces joining records to metadata on a mutable DOI string, which
    is what produced records with no title, journal or year.
    """

    def __init__(self, path: Path, *, strict: bool = False):
        self.path = Path(path)
        self.entries: dict[str, dict] = {}
        if self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if not isinstance(loaded, dict) or not all(
                    isinstance(key, str) and isinstance(value, dict)
                    for key, value in loaded.items()
                ):
                    raise ValueError("manifest must map string uids to objects")
                self.entries = loaded
            except (OSError, json.JSONDecodeError, ValueError) as e:
                if strict:
                    raise RuntimeError(
                        f"Manifest is unreadable: {self.path}"
                    ) from e
                logger.warning(f"Manifest unreadable ({e}); starting a new one")
                self.entries = {}

    def upsert(self, uid: str, **fields: Any) -> None:
        entry = self.entries.setdefault(uid, {"uid": uid})
        for k, v in fields.items():
            if v is not None and v != "":
                entry[k] = v

    def get(self, uid: str) -> dict:
        return self.entries.get(uid, {})

    def uids(self) -> list[str]:
        return list(self.entries)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.entries, fh, indent=2, ensure_ascii=False, sort_keys=True)
        tmp.replace(self.path)

    # -- reporting ----------------------------------------------------------

    def contributing(self, record_uids: list[str]) -> dict[str, int]:
        """Count records per paper and expose papers that yielded nothing."""
        counts = {uid: 0 for uid in self.entries}
        for uid in record_uids:
            if uid in counts:
                counts[uid] += 1
        return counts

    def zero_yield(self, record_uids: list[str]) -> list[dict]:
        """Papers that produced no records. Never let these pass unnoticed."""
        counts = self.contributing(record_uids)
        return [self.entries[uid] for uid, n in counts.items() if n == 0]
