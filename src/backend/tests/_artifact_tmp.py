"""Temp artifact dirs under the active data root.

``storage.safe_load`` (#21) confines model/pipeline ``joblib.load`` to the data
root. conftest's autouse fixture points ``DATA_DIR`` at the per-test ``tmp_path``,
so tests that dump a model to a bare ``tempfile.mkdtemp()`` (system temp, outside
the root) now get rejected on load. Route those temp dirs through here instead.
"""

from __future__ import annotations

import tempfile
from pathlib import Path


def models_base() -> Path:
    """The active ``<data root>/models`` dir, created if needed."""
    from core import storage

    base = storage.models_dir()
    base.mkdir(parents=True, exist_ok=True)
    return base


def data_tmpdir() -> str:
    """A fresh temp dir under the data root (``storage.safe_load``-loadable)."""
    return tempfile.mkdtemp(dir=models_base())
