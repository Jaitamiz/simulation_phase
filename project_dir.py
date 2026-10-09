# simulation/project_dir.py
"""
Shared, mutable store for the active project directory.

All simulation modules (detector, recorder, frame_updater) import this
instead of keeping their own dead CACHE_DIR = "" constant.

Usage
-----
Set once, from simulation_engine.ClashSimMixin.on_scene_loaded():

    from simulation import project_dir as _pd
    _pd.set_project_dir(scene.metadata["project"]["path"])

Read anywhere else:

    from simulation.project_dir import get_project_dir
    base = get_project_dir()           # raises if never set
    cache = os.path.join(base, ".cache", "penetration_cache_bidirectional")
"""
from __future__ import annotations

import os
from pathlib import Path

_active_project_dir: str | None = None


def set_project_dir(path: str | Path) -> None:
    """Record the active project directory.  Called by ClashSimMixin.on_scene_loaded()."""
    global _active_project_dir
    _active_project_dir = str(Path(path).resolve())
    print(f"[project_dir] active project dir → {_active_project_dir}")


def get_project_dir() -> str:
    """
    Return the active project directory.

    Raises RuntimeError if set_project_dir() has not been called yet —
    this surfaces the missing initialisation loudly instead of silently
    writing files to a wrong/empty path.
    """
    if _active_project_dir is None:
        raise RuntimeError(
            "[simulation.project_dir] Project directory has not been set. "
            "Call project_dir.set_project_dir(path) before using any "
            "simulation module that needs the cache path."
        )
    return _active_project_dir


def get_cache_dir(subdir: str = "") -> str:
    """
    Return  <project_dir>/.cache[/<subdir>]  and create it if needed.

    Examples
    --------
    get_cache_dir()                                 → <proj>/.cache
    get_cache_dir("penetration_cache_bidirectional")→ <proj>/.cache/penetration_cache_bidirectional
    get_cache_dir("clash_output")                   → <proj>/.cache/clash_output
    """
    base = os.path.join(get_project_dir(), ".cache")
    path = os.path.join(base, subdir) if subdir else base
    os.makedirs(path, exist_ok=True)
    return path
