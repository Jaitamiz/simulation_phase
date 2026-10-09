import os, json, zipfile, io
import numpy as np

from simulation.project_dir import get_project_dir
from simulation.io_worker import io_worker

# ============================================================================
# BIDIRECTIONAL RECORDING FUNCTIONS — CHUNKED STREAMING DESIGN
# ============================================================================
#
# WHY THIS EXISTS (vs. the old "accumulate everything, write once" model):
#
#   The old design kept every recorded frame's dict + npz bytes in memory in
#   recorded_frames_X / clash_buffers_X for the ENTIRE run, and only wrote to
#   disk when save_recorded_frames_bidirectional() was explicitly called.
#   Two problems with that at scale:
#     1. Memory risk — a long trajectory (10,000+ frames) with sustained
#        clashes can grow these buffers unpredictably; there is no hard cap
#        on per-frame clash-point count, so this is not bounded in advance
#        and could reach multiple GB before anything is ever written.
#     2. All-at-once write cost — writing potentially GB-scale JSON + ZIP in
#        a single call at the very end (or on a Pause) is slow and, worse,
#        is exactly the moment most likely to be interrupted (a crash right
#        then loses EVERYTHING, not just the last few frames).
#
#   Fix: flush to disk every _CHUNK_SIZE newly-recorded frames per direction.
#   In-memory state is bounded to at most _CHUNK_SIZE unflushed frames at any
#   time. The first flush of a fresh pass truncates/creates the files (so a
#   stale file from a previous, unrelated pass doesn't linger — mirrors "if
#   file already exists, rewrite it; else create new" for a NEW pass);
#   every flush after that appends. JSON uses JSON-Lines (.jsonl) — one
#   object per line — specifically because a normal JSON array can't be
#   appended to without rewriting the whole file; JSONL can. The ZIP already
#   supports true incremental appends natively via zipfile's "a" mode.
#
# ============================================================================

_DIRECTIONS = ("left", "right", "front", "back", "top")
_CHUNK_SIZE = 10

already_saved_frames_left = set()
already_saved_frames_right = set()
already_saved_frames_front = set()
already_saved_frames_back = set()
already_saved_frames_top = set()

_already_saved_frames = {
    "left": already_saved_frames_left, "right": already_saved_frames_right,
    "front": already_saved_frames_front, "back": already_saved_frames_back,
    "top": already_saved_frames_top,
}

# Bounded, in-memory-only buffers — hold at most _CHUNK_SIZE unflushed
# frames per direction at any time. NOT the full-run history.
_pending_records = {d: [] for d in _DIRECTIONS}          # list[dict] (JSON-able)
_pending_clips = {d: [] for d in _DIRECTIONS}            # list[(frame_idx, npz_bytes)]

# Running counters / flags — cheap, O(1) memory regardless of run length.
_total_recorded = {d: 0 for d in _DIRECTIONS}            # frames ever recorded this pass
_stream_started = {d: False for d in _DIRECTIONS}        # has this pass written anything to disk yet

# Where this pass's files live. Set once via configure_recording_output()
# at the start of a pass (see ClashSimMixin.start_animation()).
_stream_output_dir = None

# Kept for any external code that still reads these flags; mirrors
# _stream_started (True once a direction has anything on disk this pass).
recording_saved_left = False
recording_saved_right = False
recording_saved_front = False
recording_saved_back = False
recording_saved_top = False

_recording_saved_flags = {
    "left": "recording_saved_left", "right": "recording_saved_right",
    "front": "recording_saved_front", "back": "recording_saved_back",
    "top": "recording_saved_top",
}


def configure_recording_output(output_dir: str) -> None:
    """
    Call once at the start of a NEW pass (same moment as reset_recording_state()
    — see ClashSimMixin.start_animation()'s `_cache_seeded` branch) so the
    streaming writer knows where to flush chunks as they fill up, without
    needing output_dir threaded through every record_frame_bidirectional() call.
    """
    global _stream_output_dir
    _stream_output_dir = output_dir
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"❌ Error creating recording output directory: {e}")


def reset_recording_state():
    """
    Clear ALL module-level recorder state so a brand-new detection pass
    starts from a clean slate instead of silently interacting with the
    previous run's data.

    Call this exactly once per NEW pass — i.e. from the same place that
    decides the penetration cache needs reseeding (frame 0 processed fresh),
    NOT on every pause/resume. See ClashSimMixin.start_animation()'s
    `_cache_seeded` branch, which is the single trigger for both.

    Note: this does NOT touch files already written to disk by a previous
    pass's chunk flushes. If the new pass never reaches a flush (e.g. the
    user restarts again immediately, or the run is very short), that old
    partial file is simply left as-is — it gets naturally overwritten the
    moment the new pass's own first chunk flushes (see _flush_chunk()'s
    truncate-on-first-flush-of-a-pass behavior), not before.
    """
    global _pending_records, _pending_clips, _total_recorded, _stream_started
    global recording_saved_left, recording_saved_right, recording_saved_front, recording_saved_back, recording_saved_top

    for d in _DIRECTIONS:
        _already_saved_frames[d].clear()
        _pending_records[d] = []
        _pending_clips[d] = []
        _total_recorded[d] = 0
        _stream_started[d] = False

    recording_saved_left = False
    recording_saved_right = False
    recording_saved_front = False
    recording_saved_back = False
    recording_saved_top = False

    print("[Recorder] 🔄 recorder state reset — starting a fresh pass")


def _write_chunk_to_disk(direction: str, records: list, clips: list,
                          is_first_flush: bool, final: bool) -> bool:
    """
    The ACTUAL disk write — zipfile + JSONL append. Runs on the shared
    io_worker background thread (see _flush_chunk() below), never on the
    Qt main thread and never on the _FrameWorker GPU thread. Pure I/O,
    no shared mutable module state is touched here — `records`/`clips`
    are an already-snapshotted, private copy handed off by the caller.
    """
    if not records or _stream_output_dir is None:
        return False

    jsonl_path = os.path.join(_stream_output_dir, f"{direction}_clash_replay.jsonl")
    zip_path = os.path.join(_stream_output_dir, f"{direction}_clash_clips.zip")

    json_mode = "w" if is_first_flush else "a"
    zip_mode = "w" if is_first_flush else "a"

    try:
        with open(jsonl_path, json_mode, encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        with zipfile.ZipFile(zip_path, zip_mode, compression=zipfile.ZIP_DEFLATED) as zf:
            for frame_idx, npz_bytes in clips:
                zf.writestr(f"frame_{frame_idx:04d}.npz", npz_bytes)

        tag = "final partial chunk" if final else "chunk"
        print(f"[Recorder] 💾 {direction.upper()} {tag}: +{len(records)} frame(s) "
              f"(total this pass: {_total_recorded[direction]}) → {jsonl_path}")
        return True

    except Exception as e:
        print(f"❌ Error flushing {direction.upper()} chunk: {e}")
        import traceback
        traceback.print_exc()
        return False


def _flush_chunk(direction: str, final: bool = False) -> bool:
    """
    Snapshot whatever is currently pending for `direction`, clear the
    pending buffers IMMEDIATELY on the CALLING thread (so recording can
    keep accumulating new frames without waiting on disk), then hand the
    actual write off to the shared io_worker background pool.

    Ordering guarantee: io_worker.submit(direction, ...) chains all tasks
    for this direction FIFO, so chunk N's write always completes before
    chunk N+1's write starts — even though both run asynchronously off
    the calling thread. This is what makes it safe to call this from the
    Qt MAIN thread (as record_frame_bidirectional() does) without ever
    blocking the UI on a zip/jsonl write.

    Returns True if a write was actually queued (not whether it has
    finished — for that, see io_worker.wait(direction), used by
    save_recorded_frames_bidirectional() at genuine pass completion).

    `final=True` is used by save_recorded_frames_bidirectional() to flush
    a trailing partial chunk (< _CHUNK_SIZE frames) at genuine completion
    — the mechanics are identical either way, this just controls logging.
    """
    global _stream_started

    records = _pending_records[direction]
    clips = _pending_clips[direction]
    if not records or _stream_output_dir is None:
        return False

    # First flush of a NEW pass truncates/creates fresh files; every
    # subsequent flush appends. Decided (and the flag flipped) HERE, on
    # the calling thread, at submission time — not after the background
    # write completes — because the io_worker chain already guarantees
    # this task runs before any later-submitted chunk for this direction,
    # so "is this the first flush" is fully determined by submission
    # order, which we already know synchronously.
    is_first_flush = not _stream_started[direction]
    _stream_started[direction] = True
    globals()[_recording_saved_flags[direction]] = True

    # Snapshot-and-clear now (cheap: just swapping list references) so the
    # module-level pending buffers are free to keep growing immediately —
    # the background thread only ever sees its own private copy.
    _pending_records[direction] = []
    _pending_clips[direction] = []

    io_worker.submit(direction, _write_chunk_to_disk,
                      direction, records, clips, is_first_flush, final)
    return True


def record_frame_bidirectional(frame_idx, pose, R, left_clash_points, left_indices,
                               right_clash_points, right_indices, front_clash_points, front_indices,
                               back_clash_points, back_indices, top_clash_points, top_indices,
                               left_grid_labels=None, right_grid_labels=None,
                               min_dist_left=None, min_dist_right=None,
                               min_dist_front=None, min_dist_back=None, min_dist_top=None):
    """Record clash data for all five directions for one frame, streaming to
    disk in chunks of _CHUNK_SIZE frames per direction (see module docstring).

    Args:
        frame_idx: frame number
        pose: (3,) position vector [x, y, z]
        R: (3,3) rotation matrix for this frame
        left_clash_points / right_/ front_/ back_/ top_clash_points: per-direction clash points
        left_indices / right_/ front_/ back_/ top_indices: per-direction clash indices
        left_grid_labels: grid labels for left side
        right_grid_labels: grid labels for right side
        min_dist_left / min_dist_right / min_dist_front / min_dist_back / min_dist_top:
            minimum distance to vehicle for each direction
    """
    left_grid_list = list(left_grid_labels) if left_grid_labels is not None else []
    right_grid_list = list(right_grid_labels) if right_grid_labels is not None else []

    _direction_args = {
        "left":  (left_clash_points, left_indices, left_grid_list, min_dist_left, "LEFT"),
        "right": (right_clash_points, right_indices, right_grid_list, min_dist_right, "RIGHT"),
        "front": (front_clash_points, front_indices, [], min_dist_front, "FRONT"),
        "back":  (back_clash_points, back_indices, [], min_dist_back, "BACK"),
        "top":   (top_clash_points, top_indices, [], min_dist_top, "TOP"),
    }

    for direction, (clash_points, indices, grid_list, min_dist, tag) in _direction_args.items():
        if frame_idx in _already_saved_frames[direction]:
            continue

        record = {
            "frame": frame_idx,
            "pose": pose.tolist(),
            "R": R.tolist(),
            "clash_file": f"frame_{frame_idx:04d}.npz",
            "grid_labels": grid_list,
            "min_distance": float(min_dist) if min_dist is not None else None,
        }

        clash_buffer = io.BytesIO()
        np.savez_compressed(
            clash_buffer,
            clash_points=clash_points,
            clash_indices=indices,
            direction=tag,
            grid_labels=grid_list,
            min_distance=min_dist if min_dist is not None else np.nan,
        )

        _pending_records[direction].append(record)
        _pending_clips[direction].append((frame_idx, clash_buffer.getvalue()))
        _total_recorded[direction] += 1
        _already_saved_frames[direction].add(frame_idx)

        print(f" [{tag}]  Frame {frame_idx}: {len(clash_points)} clashes"
              + (f", grid={grid_list}" if grid_list else ""))

        if len(_pending_records[direction]) >= _CHUNK_SIZE:
            _flush_chunk(direction)


def save_recorded_frames_bidirectional(output_dir=None):
    """
    Finalize the current pass: flush any remaining pending (< _CHUNK_SIZE)
    frames per direction to disk, so nothing sitting in the in-memory buffer
    is lost. Earlier chunks (already flushed by record_frame_bidirectional()
    as they filled up) are already on disk and are NOT rewritten here — this
    call only tops off the trailing partial chunk.

    Returns the list of directions that have ANY data on disk for this pass
    (whether flushed just now or by an earlier chunk) — used by the caller
    to decide whether to notify clash_data_bus.
    """
    global _stream_output_dir

    if output_dir is not None:
        _stream_output_dir = output_dir
    if _stream_output_dir is None:
        _stream_output_dir = get_project_dir()

    try:
        os.makedirs(_stream_output_dir, exist_ok=True)
        print(f"📁 Output directory: {_stream_output_dir}")
    except Exception as e:
        print(f"❌ Error creating output directory: {e}")
        return []

    for direction in _DIRECTIONS:
        if _pending_records[direction]:
            _flush_chunk(direction, final=True)

    saved_dirs = [d for d in _DIRECTIONS if _stream_started[d]]

    # This call only happens at genuine pass-completion / project-switch /
    # app-close (see simulation_engine.py's _save_recordings() docstring) —
    # never per-frame — so a brief blocking wait here is fine, and it's
    # what guarantees the files are FULLY on disk (not just queued) by the
    # time this function returns. Callers rely on that: clash_data_bus's
    # listener (PostProcessMixin) re-scans disk the moment it's notified,
    # and would silently read stale/partial data without this wait.
    for d in saved_dirs:
        io_worker.wait(d)
        total = _total_recorded[d]
        print(f"✅ {d.upper()} recordings on disk this pass ({total} frame(s) total)")

    return saved_dirs