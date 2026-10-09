"""
clash_data_bus.py

Real-time notification channel between ClashSimMixin (GeneralViewTab / Screen 3,
the *producer* of {direction}_clash_replay.json / _clash_clips.zip) and
PostProcessMixin (PostProcessingTab / Screen 4, the *consumer*).

Why this exists
----------------
PostProcessMixin only ever scans <project>/.cache for clash JSON/ZIP files
inside on_pp_scene_loaded(), which fires exactly once per scene_bus publish
(i.e. once per "project opened"). If the simulation on Screen 3 is run (or
re-run) WHILE the project stays open, nothing tells Screen 4 that new — or
newly-overwritten — files just landed on disk. This bus is that missing
signal.

Design notes
------------
- Both tabs already live in the same Qt process (proven by scene_bus, which
  this file deliberately mirrors), so a plain pyqtSignal is simpler and more
  reliable here than a QFileSystemWatcher: no polling, no debounce-on-write
  races, and PyQt5 automatically queues cross-thread emits onto whichever
  thread the receiving QObject lives on, so this is safe even if a future
  refactor moves the simulation loop onto a worker QThread.
- The signal fires ONLY after save_recorded_frames_bidirectional() has
  actually finished writing files — never speculatively — so a receiver can
  safely re-read from disk the moment it's notified.
- `directions` is the list of sides (subset of "left"/"right"/"front"/
  "back"/"top") that were actually written on THIS call, so a receiver can
  skip re-scanning sides that have no new data yet, though the simplest and
  safest handling is to just re-scan everything (see PostProcessMixin's
  _on_clash_data_saved).
"""
from __future__ import annotations

from PyQt5.QtCore import QObject, pyqtSignal


class ClashDataBus(QObject):
    # (project_dir: str, directions: list[str])
    clash_data_saved = pyqtSignal(str, list)


# Single shared instance — imported and used exactly like scene_bus.scene_bus.
clash_data_bus = ClashDataBus()
