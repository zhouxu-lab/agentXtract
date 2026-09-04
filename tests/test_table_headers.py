"""Multi-row table headers must survive parsing.

Docling's dataframe export keeps a single header row, so a table labelled
across two rows loses the row naming the actual measurement conditions.
_table_grid reads the cell grid instead.
"""

from types import SimpleNamespace

from src.pdf_parser import _table_grid


def _cell(text, r0, r1, c0, c1, header=False):
    return SimpleNamespace(
        text=text,
        start_row_offset_idx=r0, end_row_offset_idx=r1,
        start_col_offset_idx=c0, end_col_offset_idx=c1,
        column_header=header,
    )


def _table(cells, n_rows, n_cols):
    return SimpleNamespace(data=SimpleNamespace(
        table_cells=cells, num_rows=n_rows, num_cols=n_cols))


def test_second_header_row_is_kept():
    # f (MHz) | T (°C) spanning three columns
    #         | 20 | 40 | 60
    cells = [
        _cell("f (MHz)", 0, 2, 0, 1, header=True),
        _cell("T (°C)", 0, 1, 1, 4, header=True),
        _cell("20", 1, 2, 1, 2, header=True),
        _cell("40", 1, 2, 2, 3, header=True),
        _cell("60", 1, 2, 3, 4, header=True),
        _cell("915", 2, 3, 0, 1),
        _cell("3.4", 2, 3, 1, 2),
        _cell("3.8", 2, 3, 2, 3),
        _cell("4.3", 2, 3, 3, 4),
    ]
    headers, rows = _table_grid(_table(cells, 3, 4))

    assert len(headers) == 2, "the row carrying the temperatures was dropped"
    assert headers[1] == ["f (MHz)", "20", "40", "60"]
    assert rows == [["915", "3.4", "3.8", "4.3"]]


def test_spanning_cell_repeats_across_its_columns():
    cells = [
        _cell("f (MHz)", 0, 2, 0, 1, header=True),
        _cell("T (°C)", 0, 1, 1, 4, header=True),
        _cell("20", 1, 2, 1, 2, header=True),
        _cell("40", 1, 2, 2, 3, header=True),
        _cell("60", 1, 2, 3, 4, header=True),
    ]
    headers, _ = _table_grid(_table(cells, 2, 4))
    # Every data column reads its own label without looking sideways.
    assert headers[0] == ["f (MHz)", "T (°C)", "T (°C)", "T (°C)"]


def test_header_styled_row_below_the_body_is_data():
    cells = [
        _cell("Sample", 0, 1, 0, 1, header=True),
        _cell("e'", 0, 1, 1, 2, header=True),
        _cell("bread", 1, 2, 0, 1),
        _cell("3.4", 1, 2, 1, 2),
        _cell("Group B", 2, 3, 0, 2, header=True),   # a divider, not a header
        _cell("cake", 3, 4, 0, 1),
        _cell("4.1", 3, 4, 1, 2),
    ]
    headers, rows = _table_grid(_table(cells, 4, 2))
    assert len(headers) == 1
    assert len(rows) == 3
    assert rows[1] == ["Group B", "Group B"]


def test_missing_grid_falls_back_to_caller():
    assert _table_grid(SimpleNamespace(data=None)) == ([], [])
    assert _table_grid(SimpleNamespace()) == ([], [])


# ---------------------------------------------------------------------------
# Column labels handed to the model
# ---------------------------------------------------------------------------

from src.schema import ParsedTable
from src.table_extractor import _column_labels


def test_labels_merge_every_header_row():
    table = ParsedTable(table_id="t", headers=[
        ["MC", "T", "Dielectric constant (MHz)", "Dielectric constant (MHz)"],
        ["MC", "T", "27", "915"],
    ])
    assert _column_labels(table) == [
        "MC", "T",
        "Dielectric constant (MHz) / 27",
        "Dielectric constant (MHz) / 915",
    ]


def test_labels_are_unique_so_columns_cannot_collapse():
    # A spanning header repeated across its columns must not produce one key.
    # Building {label: value} from duplicate labels loses every column but one.
    table = ParsedTable(table_id="t", headers=[
        ["Loss factor", "Loss factor", "Loss factor"],
    ])
    labels = _column_labels(table)
    assert len(labels) == 3
    assert len(set(labels)) == 3

def test_single_header_row_is_unchanged():
    table = ParsedTable(table_id="t", headers=[["MC", "T", "e'", "e''"]])
    assert _column_labels(table) == ["MC", "T", "e'", "e''"]


def test_no_headers_gives_no_labels():
    assert _column_labels(ParsedTable(table_id="t")) == []


def test_blank_column_still_gets_a_label():
    table = ParsedTable(table_id="t", headers=[["MC", "", "e'"]])
    labels = _column_labels(table)
    assert labels[1] == "col2"
    assert len(set(labels)) == 3
