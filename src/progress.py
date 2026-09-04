"""Real-time pipeline progress display using rich."""

from __future__ import annotations

from collections.abc import Callable

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

console = Console(highlight=False, force_terminal=True)


class PipelineProgress:
    """Wraps rich Progress for the pipeline's stage/paper/operation hierarchy."""

    def __init__(self):
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.fields[stage]}", justify="left"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("[dim]{task.fields[status]}"),
            console=console,
        )
        self._task_id = None

    def __enter__(self):
        self._progress.__enter__()
        return self

    def __exit__(self, *args):
        self._progress.__exit__(*args)

    def start_stage(self, stage: str, n_papers: int) -> None:
        if self._task_id is not None:
            self._progress.remove_task(self._task_id)
        self._task_id = self._progress.add_task(
            "", total=n_papers, stage=f"[{stage}]", status=""
        )

    def advance_paper(self, paper_label: str) -> None:
        if self._task_id is not None:
            self._progress.update(self._task_id, advance=1, status=f"✓ {paper_label}")

    def make_status_fn(self, paper_label: str) -> Callable[[str], None]:
        """Return a callback that updates the status field for a given paper."""
        def _update(msg: str) -> None:
            if self._task_id is not None:
                self._progress.update(
                    self._task_id,
                    status=f"{paper_label} | {msg}",
                )
        return _update
