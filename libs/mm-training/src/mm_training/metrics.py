"""TensorBoard metrics logging."""

from __future__ import annotations

import pathlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from torch import Tensor


class MetricsLogger:
    """Structured TensorBoard logger for training runs.

    Creates a timestamped run directory under ``runs/{experiment}/{timestamp}/``.
    """

    def __init__(self, experiment: str, run_dir: pathlib.Path | None = None) -> None:
        if run_dir is None:
            timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_dir = pathlib.Path("runs") / experiment / timestamp

        self._log_dir = run_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self._log_dir))

    @property
    def log_dir(self) -> pathlib.Path:
        """Return the directory where TensorBoard event files are written."""
        return self._log_dir

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a single scalar value."""
        self._writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, values: dict[str, float], step: int) -> None:
        """Log multiple scalars under a common tag."""
        self._writer.add_scalars(main_tag, values, step)

    def log_histogram(self, tag: str, values: Tensor, step: int) -> None:
        """Log a histogram of tensor values."""
        self._writer.add_histogram(tag, values, step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log a text string."""
        self._writer.add_text(tag, text, step)

    def close(self) -> None:
        """Flush and close the underlying writer."""
        self._writer.close()
