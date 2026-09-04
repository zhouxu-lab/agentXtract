"""Long Results sections must be windowed, not truncated.

A hard cut at 10 000 characters silently discarded the numerical results of
long papers: measured values usually appear several pages into the Results
section, so the truncation removed exactly the part worth extracting.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.text_extractor import (
    TEXT_WINDOW_CHARS,
    TEXT_WINDOW_OVERLAP,
    _window_results_text,
)


def test_short_text_is_one_window():
    text = "x" * 500
    assert _window_results_text(text) == [text]


def test_long_text_is_split_not_truncated():
    text = "".join(f"{i % 10}" for i in range(TEXT_WINDOW_CHARS * 3))
    windows = _window_results_text(text)
    assert len(windows) > 1
    assert sum(len(w) for w in windows) >= len(text)


def test_no_content_is_lost_at_the_seams():
    text = "".join(chr(97 + i % 26) for i in range(TEXT_WINDOW_CHARS * 2 + 137))
    windows = _window_results_text(text)
    rebuilt = windows[0]
    step = TEXT_WINDOW_CHARS - TEXT_WINDOW_OVERLAP
    for i, w in enumerate(windows[1:], 1):
        rebuilt += w[len(rebuilt) - i * step:]
    assert rebuilt == text


def test_windows_overlap_so_straddling_values_survive():
    text = "y" * (TEXT_WINDOW_CHARS * 2)
    windows = _window_results_text(text)
    step = TEXT_WINDOW_CHARS - TEXT_WINDOW_OVERLAP
    assert step < TEXT_WINDOW_CHARS, "windows must overlap"
    assert len(windows[0]) == TEXT_WINDOW_CHARS


def test_the_old_truncation_marker_is_gone():
    src = Path("src/text_extractor.py").read_text()
    assert "TEXT TRUNCATED AT 10000 CHARS" not in src
    assert "[:10000]" not in src
