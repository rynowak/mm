"""Run manifest: capture and persist reproducibility metadata."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from mm_training.device import get_device_info

if TYPE_CHECKING:
    import pathlib


@dataclass
class RunManifest:
    """Immutable snapshot of everything needed to reproduce a training run."""

    experiment: str
    config: dict[str, object]
    seed: int
    dataset_id: str
    dataset_revision: str | None
    package_versions: dict[str, str]
    git_commit: str | None
    hardware: dict[str, str]

    # --- factory ---

    @classmethod
    def capture(
        cls,
        experiment: str,
        config: dict[str, object],
        seed: int,
        dataset_id: str,
        dataset_revision: str | None = None,
    ) -> RunManifest:
        """Auto-capture package versions, git commit, and hardware info."""
        return cls(
            experiment=experiment,
            config=config,
            seed=seed,
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            package_versions=_get_package_versions(),
            git_commit=_get_git_commit(),
            hardware=get_device_info(),
        )

    # --- persistence ---

    def save(self, path: pathlib.Path) -> None:
        """Write manifest to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: pathlib.Path) -> RunManifest:
        """Read manifest from a JSON file."""
        data = json.loads(path.read_text())
        return cls(**data)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_package_versions() -> dict[str, str]:
    """Collect versions of key packages."""
    import importlib.metadata as importlib_metadata

    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for pkg in ("torch", "tensorboard"):
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def _get_git_commit() -> str | None:
    """Return the current HEAD commit hash, or None if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
