"""Tests for content-addressed paper identity.

These lock down the defect that let one PDF acquire different path-shaped and
DOI-based keys, which broke checkpoint deduplication, admitted stale extraction
artifacts, and left records without source metadata.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paper_id import (
    Manifest,
    artifact_name,
    compute_uid,
    find_artifact,
    is_uid,
    slugify,
    split_artifact_name,
)


def _write_pdf(tmp_path: Path, name: str, content: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(content)
    return p


# ---------------------------------------------------------------------------
# Identity is a property of the content, not the filename
# ---------------------------------------------------------------------------

def test_same_content_different_names_gets_one_identity(tmp_path):
    content = b"%PDF-1.4 synthetic fixture" * 50
    a = _write_pdf(tmp_path, "source_original.pdf", content)
    b = _write_pdf(tmp_path, "source_copy.pdf", content)
    assert compute_uid(a) == compute_uid(b)


def test_different_content_gets_different_identity(tmp_path):
    a = _write_pdf(tmp_path, "a.pdf", b"%PDF one")
    b = _write_pdf(tmp_path, "b.pdf", b"%PDF two")
    assert compute_uid(a) != compute_uid(b)


def test_uid_is_stable_across_calls(tmp_path):
    p = _write_pdf(tmp_path, "x.pdf", b"%PDF stable" * 100)
    assert compute_uid(p) == compute_uid(p)
    assert is_uid(compute_uid(p))


def test_uid_survives_a_move(tmp_path):
    sub = tmp_path / "nested"
    sub.mkdir()
    a = _write_pdf(tmp_path, "paper.pdf", b"%PDF content here")
    first = compute_uid(a)
    moved = sub / "renamed_source.pdf"
    a.rename(moved)
    assert compute_uid(moved) == first


def test_unreadable_file_does_not_crash(tmp_path):
    missing = tmp_path / "not_here.pdf"
    uid = compute_uid(missing)
    assert is_uid(uid)


# ---------------------------------------------------------------------------
# Artifact naming round-trips without losing information
# ---------------------------------------------------------------------------

def test_artifact_name_round_trip():
    uid = "9f2a1c4b7e30"
    name = artifact_name(uid, "Source Copy")
    got_uid, slug = split_artifact_name(name)
    assert got_uid == uid
    assert slug == "Source_Copy"


def test_the_old_lossy_round_trip_is_not_used_for_identity():
    """Readable slugs may be lossy; identity must not depend on them."""
    lossy = slugify("Source: Copy").replace("_", " ")
    assert lossy != "Source: Copy"
    uid = "9f2a1c4b7e30"
    assert split_artifact_name(artifact_name(uid, "Source: Copy"))[0] == uid


def test_find_artifact_locates_by_uid_regardless_of_slug(tmp_path):
    uid = "abcdef123456"
    (tmp_path / f"{uid}__Source_Copy.json").write_text("{}")
    found = find_artifact(tmp_path, uid)
    assert found is not None and found.name.startswith(uid)
    assert find_artifact(tmp_path, "000000000000") is None


def test_slugify_is_filesystem_safe():
    assert "/" not in slugify("a/b:c*d?.pdf")
    assert slugify("") == "paper"
    assert len(slugify("x" * 500)) <= 60


# ---------------------------------------------------------------------------
# Manifest — the record/metadata join table
# ---------------------------------------------------------------------------

def test_manifest_round_trip(tmp_path):
    m = Manifest(tmp_path / "manifest.json")
    m.upsert("aaaaaaaaaaaa", pdf_path="inputs/source_copy.pdf",
             title="Synthetic source title", journal="Example Journal",
             year=2020)
    m.save()

    reloaded = Manifest(tmp_path / "manifest.json")
    entry = reloaded.get("aaaaaaaaaaaa")
    assert entry["title"] == "Synthetic source title"
    assert entry["year"] == 2020


def test_manifest_upsert_does_not_erase_with_blanks(tmp_path):
    m = Manifest(tmp_path / "manifest.json")
    m.upsert("aaaaaaaaaaaa", title="Real Title")
    m.upsert("aaaaaaaaaaaa", title="", journal="Example Journal")
    assert m.get("aaaaaaaaaaaa")["title"] == "Real Title"
    assert m.get("aaaaaaaaaaaa")["journal"] == "Example Journal"


def test_manifest_reports_zero_yield_papers(tmp_path):
    """A paper that produces no records must remain visible in reporting."""
    m = Manifest(tmp_path / "manifest.json")
    for uid, title in [("a" * 12, "Yielded"), ("b" * 12, "Silent"),
                       ("c" * 12, "Also silent")]:
        m.upsert(uid, title=title)

    record_uids = ["a" * 12] * 40
    zero = m.zero_yield(record_uids)
    assert {e["title"] for e in zero} == {"Silent", "Also silent"}

    counts = m.contributing(record_uids)
    assert counts["a" * 12] == 40
    assert counts["b" * 12] == 0


def test_manifest_survives_corrupt_file(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{ not json")
    m = Manifest(path)
    assert m.entries == {}
    m.upsert("a" * 12, title="Recovered")
    m.save()
    assert Manifest(path).get("a" * 12)["title"] == "Recovered"
