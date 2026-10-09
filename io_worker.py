"""
simulation/io_worker.py

Shared background I/O executor for disk-bound work that must run on
NEITHER of the two threads that matter most for responsiveness:

  1. The Qt MAIN thread          — UI, plotter/VTK rendering, event loop.
  2. The single _FrameWorker QThread (simulation_engine.py) — the ONE
     thread that owns all Open3D/GPU tensor work (raycasting, signed-
     distance queries) for a frame. This must stay a dedicated GPU-owning
     thread, per the target architecture:

        Qt main thread  -> UI / rendering only
        _FrameWorker    -> GPU/Open3D + NumPy math only   (1 thread)
        IOWorker (this) -> disk I/O only                  (2 threads)

Why this file exists (the two real blocking-I/O call sites it replaces)
------------------------------------------------------------------------
  A) recorder.py's _flush_chunk() — zipfile + JSONL writes. Previously
     called SYNCHRONOUSLY on the MAIN THREAD every _CHUNK_SIZE frames
     (record_frame_bidirectional() is invoked from
     ClashSimMixin._apply_computed_frame(), which runs on the main
     thread). This was a real, periodic UI freeze: every 10th frame the
     whole GUI stalled for however long the zip/jsonl write took.

  B) detector.py's AcceleratedBidirectionalPenetrationCache.
     save_frame_cache() — a read-modify-write of a small .npz file,
     called once per direction per frame (5x/frame) from
     _detect_single_direction(), which runs INSIDE compute_frame_data()
     on the background _FrameWorker thread. This didn't freeze the UI,
     but it serialized disk I/O with GPU raycasting on the one thread
     that should be doing GPU work only, slowing frame throughput.

  (The old ThreadPoolExecutor(max_workers=8) that lived inside
  AcceleratedBidirectionalPenetrationCache was never actually submitted
  to anywhere in the codebase — dead weight that reserved threads for no
  work while looking, misleadingly, like it explained CPU contention.
  It has been removed in favor of this shared, actually-used pool.)

Design: per-key FIFO chains on a shared ThreadPoolExecutor
------------------------------------------------------------
A plain ThreadPoolExecutor does not guarantee that two tasks submitted
for the SAME logical resource run in submission order — pool scheduling
is not FIFO-per-key. recorder.py's chunked-append design depends on
strict ordering per direction (chunk 2 must never hit disk before chunk
1, since JSONL/ZIP appends are order-sensitive). So this module chains
tasks per `key` — each new task for a key only starts after the
previous task for that SAME key has finished — while tasks under
DIFFERENT keys still run concurrently, bounded by max_workers.

detector.py's save_frame_cache() writes a unique file per (frame,
direction) call, so it needs no cross-call ordering; it simply passes a
unique key per call (the cache file path) and gets free concurrency.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Dict


class IOWorker:
    """Small, shared, ordered-per-key background executor for disk I/O."""

    def __init__(self, max_workers: int = 2, thread_name_prefix: str = "sim-io"):
        # 2 workers by default — this pool exists ONLY for disk I/O
        # (small file writes), not compute, so it deliberately stays tiny.
        # See module docstring: CPU-heavy work belongs on the single
        # _FrameWorker GPU thread or plain NumPy on the main thread's
        # cheap per-tick work, never on this pool.
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix=thread_name_prefix
        )
        self._lock = threading.Lock()
        self._chains: Dict[str, Future] = {}

    def submit(self, key: str, fn: Callable, *args, **kwargs) -> Future:
        """
        Queue fn(*args, **kwargs) to run in the background. Guaranteed to
        start only after any earlier task submitted under the same `key`
        has finished (FIFO-per-key); tasks under different keys may run
        concurrently. Returns a Future — safe to ignore (fire-and-forget)
        or wait on via .result() / wait(key) below.
        """
        with self._lock:
            prev = self._chains.get(key)

            def _run(prev_future=prev):
                if prev_future is not None:
                    try:
                        prev_future.result()
                    except Exception:
                        # An earlier write's failure must not silently wedge
                        # this key's chain forever — log-and-continue is the
                        # right behavior for a disk-write queue.
                        import traceback
                        print(f"[IOWorker] ⚠️ earlier task for key={key!r} failed:")
                        traceback.print_exc()
                return fn(*args, **kwargs)

            new_future = self._executor.submit(_run)
            self._chains[key] = new_future
            return new_future

    def wait(self, key: str, timeout: float | None = None) -> None:
        """Block the CALLING thread until every task submitted so far under
        `key` has finished. Use sparingly — e.g. once per completed pass
        (save_recorded_frames_bidirectional()), never per-frame."""
        with self._lock:
            fut = self._chains.get(key)
        if fut is not None:
            try:
                fut.result(timeout=timeout)
            except Exception:
                pass

    def wait_all(self, timeout: float | None = None) -> None:
        """Block until every key's queue has drained. Intended for app
        shutdown (closeEvent) only."""
        with self._lock:
            futures = list(self._chains.values())
        for fut in futures:
            try:
                fut.result(timeout=timeout)
            except Exception:
                pass

    def shutdown(self, wait: bool = True) -> None:
        self.wait_all()
        self._executor.shutdown(wait=wait)


# Single shared instance — imported and used exactly like clash_data_bus /
# scene_bus. Both recorder.py and detector.py import THIS instance rather
# than creating their own pools, so the whole app has exactly one small,
# bounded disk-I/O thread pool regardless of how many cache managers or
# recording passes come and go.
io_worker = IOWorker(max_workers=2)
