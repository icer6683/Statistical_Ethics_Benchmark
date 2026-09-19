"""Per-run model workspaces.

A workspace is a fresh temporary directory, created outside the repository, that contains
only `data.csv`. The tested model's file tools are confined to it, and reference answers
never enter it.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from . import config as C


class PathEscape(ValueError):
    """Raised when a tool argument points outside the workspace."""


def create_workspace(data_csv: Path) -> Path:
    """Create a temp workspace containing a copy of `data.csv` and nothing else."""
    root = Path(tempfile.mkdtemp(prefix="sigws_"))
    shutil.copyfile(data_csv, root / "data.csv")
    return root


def resolve_in_workspace(workspace: Path, relative: str) -> Path:
    """Resolve a model-supplied path, rejecting absolute paths and escapes."""
    candidate = Path(relative)
    if candidate.is_absolute():
        raise PathEscape(f"absolute paths are not allowed: {relative}")
    resolved = (workspace / candidate).resolve()
    workspace_resolved = workspace.resolve()
    if resolved != workspace_resolved and workspace_resolved not in resolved.parents:
        raise PathEscape(f"path escapes the working directory: {relative}")
    return resolved


def list_workspace(workspace: Path) -> list[str]:
    files = []
    for path in sorted(workspace.rglob("*")):
        if path.is_file():
            files.append(str(path.relative_to(workspace)))
    return files


def snapshot(workspace: Path, destination: Path) -> None:
    """Copy the current workspace contents into the run directory for the record."""
    destination.mkdir(parents=True, exist_ok=True)
    for path in workspace.rglob("*"):
        target = destination / path.relative_to(workspace)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


def assert_no_references(workspace: Path) -> list[str]:
    """Return any workspace files that look like leaked reference answers."""
    leaked = []
    for name in list_workspace(workspace):
        lowered = name.lower()
        if "reference" in lowered or lowered.startswith("references/"):
            leaked.append(name)
    return leaked


def contains_reference_hint(text: str) -> bool:
    """Flag tool arguments that reach toward the repository's reference answers."""
    lowered = text.lower()
    hints = [
        str(C.REFERENCES_DIR).lower(),
        str(C.REPO_ROOT).lower(),
        "reference.json",
        "reference.csv",
        "references/",
    ]
    return any(hint in lowered for hint in hints)
