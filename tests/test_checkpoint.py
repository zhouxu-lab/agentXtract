"""Tests for the checkpoint database."""

import pytest

from src.schema import CostEntry, PaperCheckpoint, PipelineStatus
from src.utils import CheckpointDB


@pytest.fixture
def db(tmp_path):
    db = CheckpointDB(tmp_path / "test.db")
    yield db
    db.close()


def test_upsert_and_get(db):
    cp = PaperCheckpoint(
        doi="10.1234/test",
        status=PipelineStatus.PARSED,
        pdf_path="/tmp/test.pdf",
    )
    db.upsert(cp)
    result = db.get("10.1234/test")
    assert result is not None
    assert result.doi == "10.1234/test"
    assert result.status == PipelineStatus.PARSED


def test_upsert_updates_status(db):
    cp1 = PaperCheckpoint(doi="10.1234/test", status=PipelineStatus.PARSED)
    db.upsert(cp1)
    cp2 = PaperCheckpoint(doi="10.1234/test", status=PipelineStatus.EXTRACTED)
    db.upsert(cp2)
    result = db.get("10.1234/test")
    assert result.status == PipelineStatus.EXTRACTED


def test_get_nonexistent(db):
    assert db.get("nonexistent") is None


def test_get_by_status(db):
    db.upsert(PaperCheckpoint(doi="a", status=PipelineStatus.PARSED))
    db.upsert(PaperCheckpoint(doi="b", status=PipelineStatus.PARSED))
    db.upsert(PaperCheckpoint(doi="c", status=PipelineStatus.EXTRACTED))
    parsed = db.get_by_status(PipelineStatus.PARSED)
    assert len(parsed) == 2


def test_summary(db):
    db.upsert(PaperCheckpoint(doi="a", status=PipelineStatus.PARSED))
    db.upsert(PaperCheckpoint(doi="b", status=PipelineStatus.PARSED))
    db.upsert(PaperCheckpoint(doi="c", status=PipelineStatus.EXTRACTED))
    s = db.summary()
    assert s[PipelineStatus.PARSED] == 2
    assert s[PipelineStatus.EXTRACTED] == 1


def test_cost_tracking(db):
    entry = CostEntry(
        stage="screen",
        model="claude-haiku-4-5-20251001",
        doi="10.1234/test",
        input_tokens=500,
        output_tokens=200,
        cost_usd=0.005,
    )
    db.add_cost(entry)
    assert db.total_cost() == pytest.approx(0.005)

    entry2 = CostEntry(
        stage="extract",
        model="claude-sonnet-4-5-20250929",
        doi="10.1234/test",
        input_tokens=4000,
        output_tokens=2000,
        cost_usd=0.10,
    )
    db.add_cost(entry2)
    assert db.total_cost() == pytest.approx(0.105)
