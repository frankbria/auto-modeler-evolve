"""Filesystem-path containment for user-influenced filenames.

Every ingestion path that writes a file derives its name from user input
(``file.filename``, ``save_as_filename``, a URL path segment). Joining such a
name straight onto an upload directory lets ``../../etc/passwd`` or an absolute
path escape the project sandbox. Likewise ``extract-db`` opens a client-supplied
``db_path`` that must stay inside the server-managed upload dir.

Two primitives cover both cases:
  - ``sanitize_filename`` strips any directory component and rejects traversal /
    absolute names — use it before joining a name onto a base dir.
  - ``assert_within`` confines an already-built path to a base dir by resolving
    symlinks and comparing real paths — use it as a belt-and-braces check and
    for the ``db_path`` confinement case.

Pure, stdlib-only, no I/O beyond ``Path.resolve`` (which does not require the
path to exist).
"""

from __future__ import annotations

from pathlib import Path


class UnsafePathError(ValueError):
    """Raised when a filename / path would escape its intended directory."""


def sanitize_filename(name: str | None) -> str:
    """Return the bare filename component of ``name``, or raise.

    ``Path(name).name`` drops every directory part, so ``../../x.csv`` and
    ``/etc/x.csv`` both collapse to ``x.csv``. We then reject anything that
    still looks unsafe (empty, ``.``/``..``, or a residual separator) so the
    result is always a single safe path segment.
    """
    if not name or not name.strip():
        raise UnsafePathError("Filename is empty")
    base = Path(name).name
    if base in ("", ".", "..") or "/" in base or "\\" in base:
        raise UnsafePathError(f"Unsafe filename: {name!r}")
    return base


def resolve_within(base_dir: Path, name: str | None) -> Path:
    """Sanitize ``name`` and join it under ``base_dir``, asserting containment.

    Returns the resolved path. Raises ``UnsafePathError`` if it would land
    outside ``base_dir`` (defensive — sanitize_filename already prevents this,
    but resolving guards against symlinked base dirs).
    """
    safe_name = sanitize_filename(name)
    candidate = base_dir / safe_name
    return assert_within(base_dir, candidate)


def assert_within(base_dir: Path, candidate: Path) -> Path:
    """Raise ``UnsafePathError`` unless ``candidate`` resolves inside ``base_dir``.

    Both sides are resolved (symlinks + ``..`` collapsed) before comparison so a
    crafted ``candidate`` cannot escape via traversal or a symlink.
    """
    base_resolved = base_dir.resolve()
    candidate_resolved = candidate.resolve()
    if (
        base_resolved != candidate_resolved
        and base_resolved not in candidate_resolved.parents
    ):
        raise UnsafePathError(f"Path {candidate} escapes base directory {base_dir}")
    return candidate_resolved
