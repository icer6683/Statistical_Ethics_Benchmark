"""Minimal .env loader.

API keys live in a gitignored `.env` at the repository root so that runs work from any
shell without exporting keys globally. Values already present in the environment win, and
nothing here ever prints a key.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import config as C


def load_env(path: Path | None = None, override: bool = False) -> list[str]:
    """Load KEY=value lines from `.env`. Returns the names (not values) that were set."""
    path = path or (C.REPO_ROOT / ".env")
    if not path.exists():
        return []

    loaded = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if not name or (name in os.environ and not override):
            continue
        os.environ[name] = value
        loaded.append(name)
    return loaded
