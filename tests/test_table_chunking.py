"""Tests for table chunking logic."""

from src.schema import ParsedTable
from src.table_extractor import chunk_table


def _make_table(rows: list[list[str]], caption: str = "Test table") -> ParsedTable:
    return ParsedTable(
        table_id="table_1",
        caption=caption,
        headers=[["Col1", "Col2", "Col3"]],
        rows=rows,
    )


def test_small_table_no_chunking():
    """Tables with ≤ max_rows should not be chunked."""
    rows = [["a", "1", "2"], ["b", "3", "4"], ["c", "5", "6"]]
    table = _make_table(rows)
    chunks = chunk_table(table, max_rows=20)
    assert len(chunks) == 1
    assert chunks[0] == rows


def test_large_table_force_split():
    """Tables without group boundaries should be force-split."""
    rows = [[f"item_{i}", str(i), str(i*2)] for i in range(25)]
    table = _make_table(rows)
    chunks = chunk_table(table, max_rows=10)
    assert len(chunks) == 3  # 10 + 10 + 5
    assert sum(len(c) for c in chunks) == 25


def test_condition_group_detection():
    """Rows with non-empty first column start new groups."""
    rows = [
        ["condition A", "10", "1.1"],
        ["", "20", "1.2"],
        ["", "30", "1.3"],
        ["condition B", "10", "2.1"],
        ["", "20", "2.2"],
        ["", "30", "2.3"],
        ["condition C", "10", "3.1"],
        ["", "20", "3.2"],
        ["", "30", "3.3"],
    ]
    table = _make_table(rows)
    # With max_rows=3, each group becomes one chunk
    chunks = chunk_table(table, max_rows=3)
    assert len(chunks) == 3
    assert chunks[0][0][0] == "condition A"
    assert chunks[1][0][0] == "condition B"
    assert chunks[2][0][0] == "condition C"


def test_group_merging():
    """Small groups should be merged up to max_rows."""
    rows = [
        ["A", "1", "2"],
        ["", "3", "4"],
        ["B", "5", "6"],
        ["", "7", "8"],
        ["C", "9", "10"],
        ["", "11", "12"],
    ]
    table = _make_table(rows)
    # With max_rows=6, all should fit in one chunk
    chunks = chunk_table(table, max_rows=6)
    assert len(chunks) == 1
    assert sum(len(c) for c in chunks) == 6


def test_empty_table():
    """Empty table should return empty chunks."""
    table = _make_table([])
    chunks = chunk_table(table, max_rows=10)
    assert chunks == []


def test_data_rows_fallback():
    """Table with data_rows instead of rows should still work."""
    table = ParsedTable(
        table_id="table_1",
        caption="Test",
        headers=[["A", "B"]],
        data_rows=[
            {"A": "item1", "B": "val1"},
            {"A": "item2", "B": "val2"},
        ],
    )
    chunks = chunk_table(table, max_rows=10)
    assert len(chunks) == 1
    assert len(chunks[0]) == 2
