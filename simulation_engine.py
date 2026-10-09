import os
import numpy as np
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMessageBox

from simulation.detector import AcceleratedBidirectionalPenetrationCache
from simulation.recorder import (
    record_frame_bidirectional,
    save_recorded_frames_bidirectional,
    reset_recording_state,
    configure_recording_output,
)
from simulation.clash_data_bus import clash_data_bus
from simulation.frame_updater import compute_frame_data, apply_frame_visualization
from simulation.project_dir import set_project_dir, get_cache_dir
from simulation.camera import estimate_global_rotation, camera_follow
from simulation.utils import (
    savgol_smooth, create_rotation_matrix, rotation_matrix,
    find_nearest_grid_points,
    sort_rectangle_points_robust, create_grid_box_lines,
    align_grid_to_wall,
)
# NOTE on the two rotation functions above:
#   create_rotation_matrix — used ONLY for the one-off calibration transform
#                             in _apply_scene_data() (deriving beam_points_centered
#                             from beam_points_raw against trajectory frame k).
#   rotation_matrix         — the PRODUCTION per-frame function actually used
#                             by frame_updater.compute_frame_data() every frame
#                             during playback. run_orientation_preflight() in
#                             detector.py explicitly requires THIS one be passed
#                             in — see its docstring's CRITICAL note — because
#                             validating against create_rotation_matrix instead
#                             would "pass" the preflight against a convention
#                             playback doesn't actually use.

import pyvista as pv
import vtk

class _FrameWorker(QThread):
    """
    Runs ONE frame's heavy compute_frame_data() call off the main thread so
    the Qt event loop (tab-switching, button clicks, etc.) stays responsive
    during playback.

    Computes exactly ONE frame per start() — this is deliberately NOT a
    free-running loop. Pacing (via animation_timer) and single-flight
    discipline (never compute frame N+1 before frame N has been applied)
    both stay controlled by the main thread in ClashSimMixin.animate_frame().
    A new _FrameWorker is created for each frame that needs computing.

    Unlike the previous (dead, unused) version of this class, this does NOT
    hold a live reference to the engine/QWidget — it's constructed with a
    plain dict of already-snapshotted, read-only inputs (numpy arrays,
    scalars) taken at dispatch time. The worker thread never touches `self`
    (a QWidget-derived object), any Qt widget, or `plotter`/VTK — only
    compute_frame_data(), which is pure NumPy + detect_penetrations_bidirectional().

    ⚠️ See frame_updater.py's module docstring: this is only actually safe
    to run off the main thread if detect_penetrations_bidirectional() /
    cache_manager never touch `plotter` or any VTK render-window-attached
    object internally. Verify detector.py before relying on this in
    production.

    Target architecture this class is part of (see simulation/io_worker.py
    for the piece that used to be missing):
        Qt main thread   -> UI, plotter/VTK rendering, event loop ONLY
        _FrameWorker (1) -> GPU/Open3D raycasting + NumPy math ONLY
        io_worker (2)    -> disk I/O ONLY (recorder chunk writes,
                             detector cache writes) — never blocks the
                             other two threads.
    This QThread must stay the ONLY thread issuing Open3D tensor queries;
    do not add a second concurrent GPU worker even under load — that
    causes GPU serialization/memory contention rather than a speedup.

    Signals
    -------
    frame_done   — emitted with compute_frame_data()'s result dict once the
                   frame has been fully computed.
    frame_failed — emitted with a string error message if computation
                   raised, so a failure doesn't just silently hang playback.
    """
    frame_done = pyqtSignal(object)
    frame_failed = pyqtSignal(str)

    def __init__(self, frame_idx, compute_kwargs, parent=None):
        super().__init__(parent)
        self._frame_idx = frame_idx
        self._compute_kwargs = compute_kwargs

    def run(self):
        try:
            result = compute_frame_data(current_frame=self._frame_idx, **self._compute_kwargs)
            self.frame_done.emit(result)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.frame_failed.emit(str(exc))

class ClashSimMixin:
    """Mixin holding the clash-detection engine, adapted to run inside
    GeneralViewTab using self.lidar_panel as the render target and
    scene_bus.SceneData as the data source."""

    # ── init (call from GeneralViewTab.__init__) ──────────────────────
    def _init_clash_engine(self):
        self.wall_points = self.wall_colors = None
        self.beam_points_raw = self.beam_colors = self.beam_polydata = None
        self.beam_ids = None
        # beam_points_centered: NOT centroid-centered anymore — see
        # _apply_scene_data()'s calibration block. It's the body-fixed
        # local array derived by undoing trajectory sample
        # self.calib_frame_idx's own pose from the already real-world-
        # aligned beam_points_raw. Name kept for minimal disruption to
        # existing consumers (_initialize_beam_geometry, compute_frame_data
        # call sites) — only its SOURCE computation changed.
        self.beam_points_centered = None
        # calib_frame_idx: which trajectory sample's (R, t) is assumed to
        # match the real-world instant the uploaded model's placement
        # represents. Defaults to 0 — override this BEFORE scene load if
        # frame 0 isn't the correct calibration instant for this project
        # (see the "why frame k" discussion — picking the wrong index here
        # bakes a fixed rotation error into every frame of playback).
        self.calib_frame_idx = 0
        self._calib_frame_used = None   # set once calibration actually runs, for diagnostics
        self.trajectory_points = self.smoothed_trajectory = None
        self.R_global = None   # chase-camera alignment; set in _apply_scene_data()
        # Chase-camera offset (metres behind / above the car). Driven by the
        # View panel's "Cam Distance" / "Cam Height" fields through
        # set_camera_offset(). Keep these literals in sync with the field
        # defaults in GeneralViewTab._build_view_panel() (that panel is built
        # before this method runs, so it can't read them from here).
        self.camera_distance = 2.0
        self.camera_height = 0.0
        # (R, t) of the last applied frame — lets set_camera_offset() re-place
        # the camera while paused. None until a frame has been applied.
        self._last_cam_pose = None
        self.trajectory_rolls = self.trajectory_pitches = self.trajectory_yaws = None
        self.grid_points = self.grid_labels = None
        self.grid_actor = None
        self.grid_visible = True
        self.trajectory_visible = True
        self.car_visible = True

        # cache_manager is created lazily in on_scene_loaded() once the
        # project dir is known.  Setting None here avoids calling
        # AcceleratedBidirectionalPenetrationCache() before set_project_dir().
        self.cache_manager = None
        self.current_frame = 0
        self.is_playing = False
        self.frame_skip = 100
        self.x_translation = 0.45
        self.thickness = 0.01
        self.sd_threshold = 0.03
        self.total_frames = None

        # Tracks whether the CURRENT self.cache_manager instance has already
        # processed frame 0 (i.e. AcceleratedBidirectionalPenetrationCache has
        # its initial_beam_points / base mesh). start_animation() only needs
        # to force-seed frame 0 when this is False — NOT on every Play press.
        # Must be flipped back to False any time self.cache_manager is
        # replaced or .cleanup()'d, since that discards the seeded state.
        self._cache_seeded = False

        # Tracks whether run_orientation_preflight() has already been shown
        # for the CURRENT project/cache_manager instance. Gates
        # _start_clash_engine() from reopening the preflight popup every
        # time on_scene_loaded() fires for the same project (it can fire
        # more than once — e.g. a partial scene followed later by the full
        # scene once cloud + model + trajectory are all present). Reset to
        # False anywhere self.cache_manager is replaced/invalidated for a
        # genuinely new project — same lifecycle as _cache_seeded.
        self._preflight_shown = False

        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate_frame)
        self.animation_speed = 100

        # Threaded playback state (see animate_frame() / _dispatch_frame_worker()
        # / _on_frame_computed()). _frame_worker holds the in-flight
        # _FrameWorker (or None).
        #
        # REMOVED (Pose B elimination): this used to also hold
        # _pending_car_data — a (current_beam, R_matrix, beam_translation)
        # tuple from apply_car_rotation(), computed synchronously at
        # dispatch time and consumed by _apply_computed_frame(). Nothing
        # detection- or camera-relevant reads Pose B anymore (see
        # _apply_computed_frame(), which reads the beam pose straight out
        # of compute_frame_data()'s result dict — Pose A), so there is
        # nothing left to stash between dispatch and completion.
        self._frame_worker = None

        self.beam_actor = self.trajectory_actor = self.text_actor = None
        self.clash_actor_left = self.clash_actor_right = None
        self.clash_actor_front = self.clash_actor_back = self.clash_actor_top = None

        # Min-distance visualization actors (LEFT/RIGHT/TOP — 9 actors total:
        # 2 spheres + 1 line per side). FRONT/BACK have min distances computed
        # and recorded, but no dedicated visualization actors.
        self.min_dist_sphere_left = self.min_dist_sphere_right = self.min_dist_sphere_top = None
        self.min_dist_line_left = self.min_dist_line_right = self.min_dist_line_top = None
        self.min_dist_vehicle_sphere_left = self.min_dist_vehicle_sphere_right = None
        self.min_dist_vehicle_sphere_top = None
        self.min_distance_visible = True

        self.min_clash_data = {
            'left':  {'distance': None, 'clash_point': None, 'grid_indices': None},
            'right': {'distance': None, 'clash_point': None, 'grid_indices': None},
        }
        self.grid_box_actors_left, self.grid_box_actors_right = [], []
        self.grid_label_actors_left, self.grid_label_actors_right = [], []

        # REMOVED (Pose B elimination): self.beam_z_offset and the
        # smoothed-camera-state pair (self.smoothed_camera_pos /
        # self.smoothed_camera_target) existed only to support
        # apply_car_rotation() and update_camera_with_rotation() — both
        # deleted below. The live camera-follow used during playback is
        # camera_follow() in frame_updater.apply_frame_visualization(),
        # which already reads Pose A's own (transformed, R, t) directly
        # and was never one of these two attributes.

        self.crop_mode = False
        self.crop_box_widget = None
        self.crop_bounds = None
        self.picking_enabled = False
        self.measurement_mode = False

        self.original_wall_points = self.original_beam_points = None
        self.original_trajectory_points = self.original_smoothed_trajectory = None

        # TODO: confirm where cache_dir actually lives (see open questions)
        self.save_dir = None  # set in _on_scene_with_config()
        self.save_dir_file= None

    # ── plotter accessor ───────────────────────────────────────────────
    # NOTE: GeneralViewTab (the concrete widget that mixes this in) already
    # defines `self.plotter` → self.lidar_panel.plotter. We deliberately do
    # NOT redefine it here — a duplicate property on the mixin can win the
    # MRO lookup depending on class ordering and silently shadow the real
    # one, which was one of the reasons rendering wasn't reaching the
    # actual viewPanel widget. If this mixin is ever reused standalone
    # (without GeneralViewTab providing `plotter`), it will raise a clear
    # AttributeError instead of failing quietly.

    # ── scene_bus hand-off ────────────────────────────────────────────
    def on_scene_loaded(self, scene) -> None:
        # ── Save any in-progress recordings from the PREVIOUS project before
        #    the scene data (and save_dir) is replaced below. ───────────────
        incoming_path = (scene.metadata.get("project") or {}).get("path", "")
        is_new_project = (
            getattr(self, "_scene_project_path", None) not in (None, incoming_path)
        )
        
        if getattr(self, "save_dir", None) is not None:
            self._save_recordings(reason="project-switch")
        if getattr(self, "is_playing", False):
            self.stop_animation()
            
        if is_new_project:
            self.wall_points = self.wall_colors = None
            self.beam_points_raw = self.beam_colors = self.beam_polydata = None
            self.beam_points_centered = None
            self.beam_ids = None
            self._calib_frame_used = None
            self.trajectory_points = self.smoothed_trajectory = None
            self.trajectory_rolls = self.trajectory_pitches = self.trajectory_yaws = None
            self.grid_points = self.grid_points_raw = self.grid_labels = None
            self.total_frames = None
            self.cache_manager = None
            self._cache_seeded = False
            self.R_global = None
            self._last_cam_pose = None
            # New project → any previously-shown preflight was for a
            # different car/trajectory and is no longer valid context.
            self._preflight_shown = False

        self._scene_project_path = incoming_path or getattr(self, "_scene_project_path", None)
        self._scene = scene

        try:
            self._apply_scene_data(scene)
        except Exception:
            import traceback
            print("[GeneralView] ❌ failed to apply scene data:")
            traceback.print_exc()
            return

        has_full_clash_data = (
            scene.cloud_points is not None
            and scene.model_points is not None
            and scene.trajectory is not None
        )

        if not has_full_clash_data:
            # Not enough data for the clash-detection / playback engine yet
            # (e.g. only the cloud or only the model has been loaded so
            # far), but render whatever IS available so the view_panel
            # isn't left blank while the rest of the project is configured.
            print("[GeneralView] partial scene — rendering preview only "
                  "(clash engine needs cloud + model + trajectory)")
            self._render_preview_only()
            return

        try:
            self._start_clash_engine(scene)
        except Exception:
            import traceback
            print("[GeneralView] ❌ clash engine failed to start, "
                  "falling back to preview render:")
            traceback.print_exc()
            try:
                self._render_preview_only()
            except Exception:
                traceback.print_exc()

    # ── scene → mixin attributes ────────────────────────────────────────
    def _apply_scene_data(self, scene) -> None:
        """Copy whatever fields are present on `scene` onto self.* — safe
        to call with a partially-populated SceneData."""
        if scene.cloud_points is not None:
            self.wall_points = scene.cloud_points
            self.wall_colors = scene.cloud_colors

        if scene.model_points is not None:
            # ── NEW APPROACH: the uploaded model is already placed in the
            # SAME real-world coordinate system as wall_points/trajectory —
            # legs already sitting on the conveyor leg track. No centroid
            # centering, no arbitrary re-origin: store it exactly as
            # uploaded. This is what previews (_render_preview_only,
            # visualize_initial_state before playback) show — zero
            # computation, what-you-uploaded-is-what-you-see.
            self.beam_points_raw = scene.model_points
            self.beam_colors     = scene.model_colors
            # beam_ids: original point IDs, row-aligned with beam_points_raw
            # (and therefore also with beam_points_centered — calibration
            # is a pure per-row rigid transform, it never reorders or drops
            # rows). Required by detector.py's _load_selective_indices() to
            # correctly translate PreviewTab's saved per-side selections
            # (which reference these same original IDs) into row indices —
            # see set_project_context() in detector.py for why a direct
            # row-index reuse is wrong once any Preview edit has happened.
            # Falls back to a plain arange if scene doesn't carry ids,
            # matching ModelRepository.set_original()'s own fallback so the
            # two stay consistent whenever this scene DID come from a
            # session with ModelRepository behind it.
            scene_model_ids = getattr(scene, "model_ids", None)
            self.beam_ids = (
                scene_model_ids if scene_model_ids is not None
                else np.arange(len(scene.model_points), dtype=np.int64)
            )
            # self.beam_points_centered is (re)computed below, once both
            # this array AND the trajectory are available — see the
            # calibration block at the end of this method.

        if scene.trajectory is not None:
            self.trajectory_points = scene.trajectory
            if scene.trajectory_rpy is not None:
                # trajectory_rpy is (N, 3) — unpack columns, not a 3-tuple
                self.trajectory_rolls, self.trajectory_pitches, self.trajectory_yaws = (
                    scene.trajectory_rpy.T   # (3, N) → three (N,) arrays
                )
                # One-off camera alignment: re-express the recorded roll/pitch/yaw
                # in the scene's world axes (X = direction of travel, Z up).
                # Wrapped so a failure here can't abort the whole scene load —
                # it only affects the chase camera's orientation.
                try:
                    self.R_global = estimate_global_rotation(
                        self.trajectory_points,
                        self.trajectory_rolls, self.trajectory_pitches, self.trajectory_yaws,
                        max_dist=2.0,
                    )
                except Exception as exc:
                    print(f"[GeneralView] ⚠️ estimate_global_rotation failed — "
                          f"camera will use the raw orientation: {exc}")
                    self.R_global = None
            self.smoothed_trajectory = savgol_smooth(self.trajectory_points)
            self.total_frames        = len(self.trajectory_points)

            # DEBUG: confirm smoothing actually changed the data. Remove
            # once trajectory smoothing is confirmed working end-to-end.
            try:
                diff = np.max(np.abs(
                    self.smoothed_trajectory - self.trajectory_points))
                print(f"[GeneralView] trajectory: {len(self.trajectory_points)} pts, "
                      f"max |smoothed - raw| = {diff:.6f}")
            except Exception as exc:
                print(f"[GeneralView] trajectory diff check failed: {exc}")

        if scene.grid is not None:
            # Keep the raw, unaligned grid separately so realignment (if
            # the wall/cloud arrives later, or is reloaded) always starts
            # from the original grid rather than compounding transforms
            # onto an already-aligned array.
            self.grid_points_raw = scene.grid
            self.grid_points     = scene.grid
            # NOTE: grid_labels carries its own (label, (x, y, z)) coords,
            # separate from grid_points. align_grid_to_wall() below only
            # transforms grid_points — it does NOT update these label
            # coordinates. Not an active bug today (nothing currently
            # renders grid_labels' positions), but if label text/markers
            # are added later, they'll need the same rotation+translation
            # applied via self.grid_translation (and the rotation angle
            # would need to be returned from align_grid_to_wall too, which
            # it currently doesn't expose).
            self.grid_labels     = scene.grid_labels

        # Align grid below the scanned cloud (wall) once both are present.
        # Re-checked on every scene update since either piece may arrive
        # before the other (grid CSV and cloud E57 are loaded independently).
        if (getattr(self, "grid_points_raw", None) is not None
                and getattr(self, "wall_points", None) is not None):
            try:
                aligned_grid, grid_translation = align_grid_to_wall(
                    self.grid_points_raw, self.wall_points)
                self.grid_points = aligned_grid
                self.grid_translation = grid_translation
            except Exception as exc:
                print(f"[GeneralView] ⚠️ grid alignment failed, using raw "
                      f"grid points: {exc}")
                self.grid_points = self.grid_points_raw

        # ── Calibration: derive the body-fixed local beam array ──────────
        # Re-checked on every call, same pattern as the grid-alignment
        # block above — model_points and trajectory can arrive in either
        # order across separate on_scene_loaded() calls (partial scenes),
        # so this can't live inside either the model or trajectory `if`
        # block above; it needs BOTH to be present.
        #
        # self.beam_points_raw is already real-world aligned (legs on the
        # conveyor leg track, in the SAME coordinates as wall_points). To
        # get a body-fixed array that per-frame rotation can safely spin
        # (R(i) @ local + t(i)), we undo trajectory sample
        # self.calib_frame_idx's own recorded pose:
        #     local = R(k)ᵀ · (beam_points_raw − t(k))
        # This is exact IF trajectory[calib_frame_idx] truly is the
        # real-world instant the uploaded model's placement represents —
        # see the "why frame k" discussion. Picking the wrong index bakes
        # a FIXED rotation/translation error into every frame of playback,
        # not just the calibration frame itself.
        if (getattr(self, "beam_points_raw", None) is not None
                and getattr(self, "trajectory_points", None) is not None
                and getattr(self, "trajectory_rolls", None) is not None):
            k = self.calib_frame_idx
            n_frames = len(self.trajectory_points)
            if not (0 <= k < n_frames):
                print(f"[GeneralView] ⚠️ calib_frame_idx={k} out of range "
                      f"[0, {n_frames}) — clamping to 0")
                k = 0
            R_k = create_rotation_matrix(
                self.trajectory_rolls[k], self.trajectory_pitches[k], self.trajectory_yaws[k])
            t_k = self.trajectory_points[k]
            self.beam_points_centered = (R_k.T @ (self.beam_points_raw - t_k).T).T
            self._calib_frame_used = k
            print(f"[GeneralView] ✅ beam calibrated against trajectory frame {k} "
                  f"(R(k)ᵀ · (beam_points_raw − t(k)))")

    # ── lightweight preview render (no clash engine required) ──────────
    def _render_preview_only(self) -> None:
        """Render whichever of cloud / model / trajectory / grid are
        currently available, without requiring the full clash-detection
        pipeline (cache_manager, beam geometry, frame playback)."""
        self.plotter.clear()
        rendered_anything = False

        if self.wall_points is not None:
            wall_cloud = pv.PolyData(self.wall_points)
            if self.wall_colors is not None:
                wall_cloud['colors'] = (self.wall_colors * 255).astype(np.uint8)
                self.wall_actor = self.plotter.add_points(
                    wall_cloud, scalars='colors', rgb=True, point_size=4,
                    name='wall', render=False)
            else:
                self.wall_actor = self.plotter.add_points(
                    wall_cloud, point_size=4, name='wall', render=False)
            rendered_anything = True

        if self.beam_points_raw is not None:
            # Preview shows the upload exactly as-is — no centering, no
            # calibration transform. See _apply_scene_data()'s note on
            # beam_points_raw.
            beam_cloud = pv.PolyData(self.beam_points_raw)
            if self.beam_colors is not None:
                beam_cloud['colors'] = (self.beam_colors * 255).astype(np.uint8)
                self.beam_actor = self.plotter.add_points(
                    beam_cloud, scalars='colors', rgb=True, point_size=5,
                    name='beam', render=False)
            else:
                self.beam_actor = self.plotter.add_points(
                    beam_cloud, color='blue', point_size=5,
                    name='beam', render=False)
            self.beam_polydata = beam_cloud
            rendered_anything = True

        if self.trajectory_points is not None and len(self.trajectory_points) > 1:
            traj_source = (self.smoothed_trajectory
                            if self.smoothed_trajectory is not None
                            else self.trajectory_points)
            traj_line = pv.PolyData(traj_source)
            traj_line.lines = np.hstack(
                [[2, i, i + 1] for i in range(len(traj_source) - 1)])
            self.trajectory_actor = self.plotter.add_mesh(
                traj_line, color='green', line_width=2,
                name='trajectory', render=False)
            rendered_anything = True

        if self.grid_points is not None:
            grid_cloud = pv.PolyData(self.grid_points)
            self.grid_actor = self.plotter.add_points(
                grid_cloud, color='red', point_size=5,
                render_points_as_spheres=True, name='grid', render=False)
            rendered_anything = True

        status = "✅ Preview" if rendered_anything else "⚠️ No data loaded yet"
        self.text_actor = self.plotter.add_text(
            f"Status: {status}", position='upper_left',
            font_size=11, color='blue', name='info_text')

        if rendered_anything:
            self.plotter.reset_camera()
        self.plotter.add_axes()
        self.plotter.render()

    # ── full clash-detection engine bring-up ────────────────────────────
    def _start_clash_engine(self, scene) -> None:
        """All four slots are present — wire up the real clash-detection
        pipeline (cache manager, beam geometry, frame playback)."""

        # ── 1. Register the active project dir so detector / recorder
        #       can resolve their cache paths immediately.
        project_entry = scene.metadata.get("project") or {}
        project_path  = project_entry.get("path", "")

        # Capture the OLD cache_manager, and its project_dir, BEFORE any
        # cleanup/replacement below — this is the fallback source of truth
        # when scene.metadata carries no usable path (see the empty-string
        # bug this replaces: self._active_project_path used to be stamped
        # with "" unconditionally, which then broke set_project_context()
        # -> _load_selective_indices()'s `if not self.project_dir` check).
        old_cache_manager = getattr(self, "cache_manager", None)
        fallback_project_path = getattr(old_cache_manager, "project_dir", None)

        if project_path:
            set_project_dir(project_path)
            self._active_project_path = project_path
        else:
            self._active_project_path = (
                fallback_project_path
                or getattr(self, "_active_project_path", None)
                or ""
            )
            print(f"[GeneralView] ⚠️ scene.metadata carried no project path — "
                  f"falling back to _active_project_path="
                  f"{self._active_project_path!r}")

        # Kept so _save_recordings() can tell clash_data_bus which project
        # these files belong to — must match PostProcessMixin's
        # self._pp_project_dir exactly (same project_entry["path"] string).

        # Build cache manager now that the project dir is registered.
        # Pass the explicit dir so it never has to call get_cache_dir() lazily.
        if old_cache_manager is not None:
            try:
                old_cache_manager.cleanup()
            except Exception as e:
                print(f"Cache cleanup warning: {e}")

        self.cache_manager = AcceleratedBidirectionalPenetrationCache(
            cache_dir=get_cache_dir("penetration_cache_bidirectional")
        )
        self._cache_seeded = False   # fresh cache instance — hasn't seen frame 0 yet
        self._preflight_shown = False  # fresh cache instance — preflight not run yet

        # ── 2. Derive save_dir from project dir (always consistent now)
        self.save_dir = get_cache_dir("clash_output")
        self.save_dir_file= get_cache_dir()

        # ── Originals for reset / crop — REFERENCE, not copy ─────────────
        # These used to be self.wall_points.copy() etc. — for wall_points
        # that's a full duplicate of the scan cloud (10GB in, 10GB out,
        # every time a project's clash engine starts). It's unnecessary:
        # self.wall_points is ALREADY the single-source-of-truth array set
        # in _apply_scene_data() (`self.wall_points = scene.cloud_points`,
        # a direct reference — no copy happened there either), and nothing
        # anywhere in this pipeline mutates these arrays in place. Crop /
        # reset always produce a NEW array via boolean/fancy indexing
        # (`self.wall_points = self.wall_points[mask]`), which rebinds the
        # `self.wall_points` name to a different object — it never writes
        # into the array self.original_wall_points still points at. So a
        # plain reference here is exactly as safe as a copy, at zero extra
        # memory: original_* and the live attribute are two names for the
        # SAME underlying array until a crop/reset reassigns the live one.
        self.original_wall_points         = self.wall_points
        self.original_wall_colors         = self.wall_colors
        self.original_beam_points         = self.beam_points_raw
        self.original_beam_colors         = self.beam_colors
        self.original_trajectory_points   = self.trajectory_points
        self.original_smoothed_trajectory = self.smoothed_trajectory

        self._initialize_beam_geometry()

        # ── Wire project dir + beam IDs so detector.py's
        # _load_selective_indices() can find and correctly interpret
        # PreviewTab's saved per-side selections. Must happen before the
        # first create_base_meshes_bidirectional_local() call — which now
        # happens EAGERLY below (forced at scene load), not lazily on
        # frame 0 of playback — so set it here, unconditionally, every
        # time (cheap, and self._active_project_path / self.beam_ids may
        # have changed on a project switch even when this SAME
        # cache_manager instance is reused).
        self.cache_manager.set_project_context(
            project_dir=self._active_project_path, beam_ids=self.beam_ids)

        # ── Orientation alignment + base-mesh creation are DEFERRED ──────
        # PREVIOUSLY this ran eagerly, right here, every time a scene
        # loaded or the General View tab was switched to — meaning the
        # (expensive, GPU/Open3D-heavy) create_base_meshes_bidirectional_
        # local() call, and the modal interactive_alignment_gui() popup,
        # fired on every scene load/tab-switch regardless of whether the
        # user was ever going to press Play. Combined with
        # restart_with_new_x_translation() also eagerly rebuilding meshes
        # on every threshold change, this meant the SAME expensive mesh
        # build could run twice for one Play press: once here at scene
        # load, and again the moment the user adjusted the Threshold
        # field before ever hitting Play.
        #
        # Moved to _ensure_meshes_ready(), now called from the START of
        # start_animation() instead — meshes are built lazily, exactly
        # once, the first time Play is actually pressed (matching this
        # method's own original pre-eager-forcing design — see
        # _ensure_meshes_ready()'s docstring). This also means scene
        # load / tab activation no longer blocks on GPU mesh construction
        # or a modal dialog, so the UI comes up immediately.
        self.current_frame = 0
        self.visualize_initial_state()

        # sync UI defaults
        self.nav_frame_edit.setText("0")
        self.skip_frame_edit.setText(str(self.frame_skip))
        # NOTE: xtrans_edit is labeled "Threshold" in the UI but is backed by
        # x_translation on this object, not sd_threshold — keep it synced to
        # self.x_translation here.
        self.xtrans_edit.setText(str(self.x_translation))
        self.slider.setRange(0, self.total_frames - 1)
        self.slider.setValue(0)
        # Initialise the frame counter label now that total_frames is known
        if hasattr(self, "_update_frame_counter"):
            self._update_frame_counter(0)

    # ── beam geometry ───────────────────────────────────────────────────
    def _initialize_beam_geometry(self):
        # FIX: bounding_box_object(beam_points_centered, x_translation) used
        # x_translation — a probe-reach THRESHOLD (default ~0.45m, see the
        # UI's "Threshold" field) — as one of the OBB's own dimensions,
        # instead of deriving the box from beam_points_centered's actual
        # extent. filter_wall_points_by_obb() requires this box to be a
        # TIGHT, axis-aligned bound of the car in its own local X/Y/Z (it
        # does R.T @ (wall_points - center) and compares directly against
        # these half-extents in that frame — see that function's
        # docstring) — a box sized from an unrelated threshold value can
        # legitimately be smaller than the car itself along whichever axis
        # x_translation replaced, which is exactly what the alignment
        # popup's OBB wireframe visually confirmed: the point cloud
        # extending past the box on one end. Wall points near the car's
        # real front/rear then get filtered against a window that doesn't
        # reach them, producing false/missing clash points independent of
        # anything about orientation or PCA.
        #
        # Correct approach: compute the box directly from
        # beam_points_centered's own min/max in its local X/Y/Z — this is
        # ALWAYS a valid, tight bound of every point, by construction,
        # regardless of x_translation or orientation.
        mins = self.beam_points_centered.min(axis=0)
        maxs = self.beam_points_centered.max(axis=0)
        self.obb_center_local = (mins + maxs) / 2.0
        # PAD by x_translation: it's legitimately the probe extrusion
        # REACH used everywhere else in this pipeline (see `length=
        # self.x_translation` in create_base_meshes_bidirectional_local()
        # below). A detection window sized to ONLY the car's tight body
        # extent, with no margin, could clip away wall points a probe can
        # legitimately reach beyond the surface — filter_wall_points_by_obb()
        # would exclude them before the raycast ever runs, causing missed
        # clashes at the edges rather than false ones. Padding by the same
        # reach keeps the window at least as large as what probes can
        # actually detect, without reintroducing the original bug (using
        # x_translation to REPLACE a body dimension instead of padding it).
        margin = max(self.x_translation, 0.0)
        self.obb_half_extents = (maxs - mins) / 2.0 + margin
        self.local_origin = self.obb_center_local  # kept for API parity; unused downstream per filter_wall_points_by_obb()'s own docstring

        cx, cy, cz = self.obb_center_local
        hx, hy, hz = self.obb_half_extents
        self.bbox_local_points = np.array([
            [cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz],
        ])

        print(f"[GeneralView] ✅ OBB rebuilt from actual point extent "
              f"(NOT x_translation): half_extents={self.obb_half_extents}, "
              f"center={self.obb_center_local}")

        # Axis-aligned fallback — see _compute_pca_oriented_obb() for the
        # tighter, PCA-oriented refinement that OVERWRITES all of the
        # above once orientation (self.cache_manager.pca_info) is
        # committed. None here signals "not yet oriented"; filter_wall_
        # points_by_obb() treats None as identity, so detection remains
        # correct (just using the axis-aligned box) even if the refinement
        # below never runs for some reason.
        self.obb_R_local = None

        self.beam_polydata = pv.PolyData(self.beam_points_centered)

    def _compute_pca_oriented_obb(self):
        """
        Refine self.obb_center_local / obb_half_extents / bbox_local_points
        / obb_R_local from the axis-aligned fallback (_initialize_beam_
        geometry()) into a TIGHT, PCA-oriented box — aligned to the car's
        true length/width/height axes (self.cache_manager.pca_info,
        committed by interactive_alignment_gui() or auto-PCA fallback
        inside create_base_meshes_bidirectional_local()), not just
        beam_points_centered's raw local X/Y/Z.

        WHY THIS MATTERS: an axis-aligned box in raw local X/Y/Z is always
        a VALID bound (every point fits inside it, by construction), but
        it is only a TIGHT one if the car's true body axes happen to
        coincide with local X/Y/Z — otherwise the box is needlessly larger
        along the diagonal. Using the car's own committed PCA axes gives
        the actual minimal bound, exactly matching what "PCA-fit OBB" is
        supposed to mean.

        MUST run AFTER create_base_meshes_bidirectional_local() (so
        self.cache_manager.pca_info is guaranteed set — either from the
        interactive GUI or its own auto-PCA fallback). No-ops (keeps the
        axis-aligned fallback) if pca_info still isn't available for some
        reason.

        Sets self.obb_R_local — the box's own orientation WITHIN the car's
        local X/Y/Z frame. Every frame, compute_frame_data() passes this
        through unchanged to detect_penetrations_bidirectional() ->
        filter_wall_points_by_obb(), which composes it with that frame's
        own (R, t) as `R @ obb_R_local` — i.e. the box gets the SAME
        two-step transform (local shape -> car's local frame -> world)
        every other piece of local geometry in this pipeline gets. This is
        what makes the box "act as a local OBB relative to the real
        world": it's defined once, in local space, and carried into world
        space fresh every frame by the same rigid transform as the beam
        mesh itself — never independently recomputed or left stale.
        """
        pca_info = getattr(self.cache_manager, "pca_info", None)
        if pca_info is None:
            print("[GeneralView] ⚠️ _compute_pca_oriented_obb(): no pca_info "
                  "committed yet — keeping axis-aligned OBB fallback")
            return

        side_info = pca_info["side_info"]
        pca_center = pca_info["pca_center"]

        # Columns = the car's true axes, expressed in its own local X/Y/Z.
        # These come straight from the COMMITTED orientation (post any
        # manual TOP/BOTTOM, LEFT/RIGHT, LENGTH/WIDTH, LENGTH/HEIGHT swap
        # in interactive_alignment_gui()) — same axes create_base_meshes_
        # bidirectional_local() actually built the probe meshes from, so
        # the detection window and the meshes can no longer disagree on
        # orientation the way the original (pre-alignment) bug allowed.
        length_vec = side_info["length_vector"] / np.linalg.norm(side_info["length_vector"])
        width_vec  = side_info["width_vector"]  / np.linalg.norm(side_info["width_vector"])
        height_vec = side_info["height_vector"] / np.linalg.norm(side_info["height_vector"])
        R_obb = np.column_stack([length_vec, width_vec, height_vec])

        # Project every beam point into the box's OWN frame (coordinates
        # along length/width/height) to get the TRUE tight extent.
        pts_in_obb_frame = (self.beam_points_centered - pca_center) @ R_obb
        mins = pts_in_obb_frame.min(axis=0)
        maxs = pts_in_obb_frame.max(axis=0)
        center_in_obb_frame = (mins + maxs) / 2.0

        # Same margin rationale as the axis-aligned fallback — pad by the
        # probe reach so the window isn't tighter than what probes can
        # actually detect.
        margin = max(self.x_translation, 0.0)
        half_extents = (maxs - mins) / 2.0 + margin

        # Convert the box's own center back into the car's local X/Y/Z:
        # center_local = pca_center + R_obb @ center_in_obb_frame
        self.obb_center_local = pca_center + R_obb @ center_in_obb_frame
        self.obb_half_extents = half_extents
        self.obb_R_local = R_obb

        # 8 corners, in the car's local X/Y/Z, for wireframe visualization
        # (_add_obb_wireframe() in detector.py) — corner_local = center +
        # R_obb @ (±hx, ±hy, ±hz), i.e. the SAME two-step composition
        # (offset in box frame -> car's local frame) used everywhere else.
        hx, hy, hz = self.obb_half_extents
        offsets = np.array([
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
            [-hx, -hy,  hz], [hx, -hy,  hz], [hx, hy,  hz], [-hx, hy,  hz],
        ])
        self.bbox_local_points = self.obb_center_local + (offsets @ R_obb.T)

        print(f"[GeneralView] ✅ OBB refined to PCA orientation — "
              f"half_extents(length,width,height)={self.obb_half_extents}, "
              f"center_local={self.obb_center_local}")

    # ── Lazy mesh/orientation build — called from start_animation() ──────
    def _ensure_meshes_ready(self) -> None:
        """
        Builds (once) whatever create_base_meshes_bidirectional_local()
        and the orientation-alignment step need, on the FIRST Play press
        rather than eagerly at scene load / tab activation — see
        _start_clash_engine()'s comment for why this was moved here.

        No-ops immediately if:
          - self.x_translation <= 0: TRAJECTORY-ONLY mode. There is no
            valid probe geometry to extrude at zero (or negative) reach —
            create_base_meshes_bidirectional_local() would otherwise hand
            Open3D's TriangleMesh.create_box() a `width=length=0`` box
            spec, which raises `RuntimeError: ... width <= 0`. This exactly
            mirrors frame_updater.compute_frame_data()'s own existing
            "x_translation <= 0 has no valid probe geometry — skip
            detection entirely" branch, which already runs happily with
            NO base meshes at all in this mode — so skipping mesh
            construction here is not just safe, it's required.
          - self.cache_manager.meshes_initialized is already True — a
            once-per-cache-manager-instance no-op, same guard
            create_base_meshes_bidirectional_local() itself uses.
          - self.beam_points_centered is None — calibration hasn't run.

        Called from the top of start_animation(), BEFORE frame 0 is
        seeded, since compute_frame_data() → detect_penetrations_
        bidirectional() needs the base meshes (and the refined,
        PCA-oriented OBB from _compute_pca_oriented_obb()) in place before
        the very first detection call.
        """
        if self.x_translation <= 0:
            print("[GeneralView] ⏭️  x_translation<=0 — trajectory-only mode, "
                  "skipping mesh/orientation build entirely (no probe "
                  "geometry to construct).")
            return
        if self.cache_manager is None or self.cache_manager.meshes_initialized:
            return
        if self.beam_points_centered is None:
            print("[GeneralView] ❌ cannot build base meshes: "
                  "self.beam_points_centered is None — calibration did not "
                  "run (see _apply_scene_data()'s calibration block).")
            return

        # ── Interactive orientation alignment (once, before first mesh
        # creation) — skip if pca_info is already committed (e.g. carried
        # over from a prior cache_manager instance by
        # restart_with_new_x_translation()) to avoid re-popping the dialog
        # needlessly.
        if self.cache_manager.pca_info is None:
            try:
                # obb_margin=self.x_translation: matches the padding
                # _compute_pca_oriented_obb() uses when it later refines
                # this into the REAL runtime detection window — so the
                # tight box shown in this popup is exactly what gets used
                # once orientation is committed, not just an approximation.
                self.cache_manager.interactive_alignment_gui(
                    self.beam_points_centered,
                    obb_center=self.obb_center_local,
                    obb_half_extents=self.obb_half_extents,
                    obb_margin=self.x_translation,
                )
            except Exception:
                import traceback
                print("[GeneralView] ❌ interactive_alignment_gui() raised — "
                      "this is why no popup appeared:")
                traceback.print_exc()

        # ── Base-mesh creation. If interactive_alignment_gui() already ran
        # above, self.pca_info / self.cache_manager._orientation_manually_
        # set are already set, so this call reuses that committed
        # orientation instead of recomputing PCA from scratch — and, per
        # create_base_meshes_bidirectional_local()'s own guard, does NOT
        # pop the separate plain-preview pv.Plotter() popup (that only
        # fires when _orientation_manually_set is False).
        try:
            self.cache_manager.create_base_meshes_bidirectional_local(
                self.beam_points_centered,
                length=self.x_translation,
                thickness=self.thickness,
                obb_center=self.obb_center_local,
                obb_half_extents=self.obb_half_extents,
            )
            # Now that pca_info is guaranteed committed (GUI or auto-PCA
            # fallback inside the call above), refine the axis-aligned
            # fallback OBB into the tight, PCA-oriented one.
            self._compute_pca_oriented_obb()
        except Exception:
            import traceback
            print("[GeneralView] ❌ create_base_meshes_bidirectional_local() "
                  "raised while lazily building base meshes on Play:")
            traceback.print_exc()

    # ── initial visualization ──────────────────────────────────────────
    def visualize_initial_state(self):
        """Rebuild the full 3-D scene for the 'ready' screen (before Play
        is ever pressed).

        NEW APPROACH: shows the uploaded model exactly as uploaded — NO
        rotation, NO translation, NO calibration transform. beam_points_raw
        is already real-world aligned (legs on the conveyor leg track, in
        the same coordinates as wall_points), so there is nothing to
        compute here; showing it as-is IS the correct picture.

        This intentionally does NOT match what compute_frame_data() will
        render once playback starts — that uses R(i) @ beam_points_centered
        + t(i) (the calibrated body-fixed array, moved by the trajectory).
        The two only coincide if self.current_frame happens to equal
        self.calib_frame_idx. That's expected and correct: this method is
        "here's what you uploaded", not "here's frame 0 of playback".
        """
        self.plotter.clear()
        self._last_cam_pose = None   # no chase pose until a frame is applied again

        # ── 1. Wall ──────────────────────────────────────────────────────
        wall_cloud = pv.PolyData(self.wall_points)
        if self.wall_colors is not None:
            wall_cloud['colors'] = (self.wall_colors * 255).astype(np.uint8)
            self.wall_actor = self.plotter.add_points(
                wall_cloud, scalars='colors', rgb=True, point_size=4,
                name='wall', render=False)
        else:
            self.wall_actor = self.plotter.add_points(
                wall_cloud, point_size=4, name='wall', render=False)

        # ── 2. Beam — raw upload, zero computation ──────────────────────
        self.beam_polydata.points = self.beam_points_raw
        self.beam_actor = self.plotter.add_points(
            self.beam_polydata, color='blue', point_size=5,
            name='beam', render=False)

        print(f"[GeneralView] ✅ Beam shown as uploaded (no transform). "
              f"Calibration will use trajectory frame {self.calib_frame_idx} "
              f"once playback starts.")

        # ── 3. Trajectory line ───────────────────────────────────────────
        print(f"[GeneralView] drawing trajectory line from "
              f"smoothed_trajectory (id={id(self.smoothed_trajectory)}, "
              f"shape={self.smoothed_trajectory.shape})")
        traj_line = pv.PolyData(self.smoothed_trajectory)
        traj_line.lines = np.hstack(
            [[2, i, i + 1] for i in range(len(self.smoothed_trajectory) - 1)])
        self.trajectory_actor = self.plotter.add_mesh(
            traj_line, color='green', line_width=2,
            name='trajectory', render=False)

        # ── 4. Grid ──────────────────────────────────────────────────────
        if self.grid_points is not None:
            grid_cloud = pv.PolyData(self.grid_points)
            self.grid_actor = self.plotter.add_points(
                grid_cloud, color='red', point_size=5,
                render_points_as_spheres=True, name='grid', render=False)

        # ── 5. Status text ───────────────────────────────────────────────
        self.text_actor = self.plotter.add_text(
            "Frame: 0\nStatus: ✅ Ready", position='upper_left',
            font_size=11, color='blue', name='info_text')

        # REMOVED (Pose B elimination): camera reset + update_camera_with_
        # rotation() call (already commented out/inert) are gone. Live
        # camera-follow during playback is camera_follow() in
        # frame_updater.apply_frame_visualization(), driven by Pose A's own
        # (transformed, R, t) each frame — nothing to reset here.

        self.plotter.render()

    # ── frame update / animation ────────────────────────────────────
    def update_frame_visualization(self):
        """Fully SYNCHRONOUS single-frame update — computes AND applies THIS
        frame before returning. Used by frame-0 seeding (start_animation()),
        manual seeks (go_to_frame()), and anywhere else that needs a
        guaranteed-complete render before the caller proceeds.

        Deliberately NOT threaded — its callers rely on the frame being
        fully rendered by the time this returns (e.g. start_animation()
        checks self._cache_seeded immediately after). Actual timer-driven
        PLAYBACK uses the threaded path instead — see animate_frame() /
        _dispatch_frame_worker() / _on_frame_computed() — to keep the Qt
        event loop responsive during a long run; this method keeps its old
        fully-blocking behavior for the seeding/seek call sites that need it.
        """
        computed = compute_frame_data(
            current_frame=self.current_frame, total_frames=self.total_frames,
            # FIX: centered array — see _apply_scene_data()'s note on
            # beam_points_centered. This is what makes the beam rotate
            # rigidly about itself instead of swinging about a
            # Z-displaced pivot, and keeps the frame-0 BVH mesh built
            # from the same frame as obb_center_local.
            beam_points=self.beam_points_centered,
            trajectory_points=self.trajectory_points,
            roll=self.trajectory_rolls, pitch=self.trajectory_pitches,
            yaw=self.trajectory_yaws,
            wall_points=self.wall_points, obb_center_local=self.obb_center_local,
            obb_half_extents=self.obb_half_extents, bbox_local_points=self.bbox_local_points,
            cache_manager=self.cache_manager, thickness=self.thickness,
            sd_threshold=self.sd_threshold, x_translation=self.x_translation,
            obb_R_local=self.obb_R_local,
        )
        self._apply_computed_frame(computed)

    def _apply_computed_frame(self, computed):
        """Shared by BOTH the synchronous path (update_frame_visualization)
        and the threaded playback path (_on_frame_computed) — applies
        plotter updates, grid analysis, recording, and label/camera sync for
        a frame that has ALREADY been computed. MUST run on the main thread
        (calls apply_frame_visualization(), which touches plotter/VTK).

        Pose B removed entirely: `computed` (Pose A — compute_frame_data()'s
        result dict) is now the ONLY beam pose anywhere in this pipeline.
        Grid/min-distance analysis and camera-follow read it directly
        (`computed["transformed"]`, `computed["beam_translation"]`) — the
        exact array that was rotated, rendered via beam_polydata.points, and
        raycasted against. There is no separate apply_car_rotation() output
        left to diverge from it.

        Matches ClashDetectionWidget.update_frame_visualization logic:
        - camera follows the beam every frame via camera_follow()
          (frame_updater.apply_frame_visualization(), driven by Pose A)
        - grid clash boxes are CLEARED when no clashes are present (not left
          over from the previous frame)
        - grid labels are extracted and forwarded to record_frame_bidirectional
        """
        # ── Apply detection results to the plotter (main thread only) ────
        (self.clash_actor_left, self.clash_actor_right,
         self.clash_actor_front, self.clash_actor_back, self.clash_actor_top,
         self.min_dist_sphere_left, self.min_dist_sphere_right, self.min_dist_sphere_top,
         self.min_dist_line_left, self.min_dist_line_right, self.min_dist_line_top,
         self.min_dist_vehicle_sphere_left, self.min_dist_vehicle_sphere_right,
         self.min_dist_vehicle_sphere_top) = apply_frame_visualization(
            plotter=self.plotter, beam_polydata=self.beam_polydata, computed=computed,
            info_label=getattr(self, "_info_label", None),
            clash_actor_left=self.clash_actor_left,
            clash_actor_right=self.clash_actor_right,
            clash_actor_front=self.clash_actor_front,
            clash_actor_back=self.clash_actor_back,
            clash_actor_top=self.clash_actor_top,
            min_dist_sphere_left=getattr(self, "min_dist_sphere_left", None),
            min_dist_sphere_right=getattr(self, "min_dist_sphere_right", None),
            min_dist_sphere_top=getattr(self, "min_dist_sphere_top", None),
            min_dist_line_left=getattr(self, "min_dist_line_left", None),
            min_dist_line_right=getattr(self, "min_dist_line_right", None),
            min_dist_line_top=getattr(self, "min_dist_line_top", None),
            min_dist_vehicle_sphere_left=getattr(self, "min_dist_vehicle_sphere_left", None),
            min_dist_vehicle_sphere_right=getattr(self, "min_dist_vehicle_sphere_right", None),
            min_dist_vehicle_sphere_top=getattr(self, "min_dist_vehicle_sphere_top", None),
            min_distance_visible=getattr(self, "min_distance_visible", True),
            R_global=getattr(self, "R_global", None),
            camera_distance=getattr(self, "camera_distance", 2.0),
            camera_height=getattr(self, "camera_height", 0.0),
        )

        left_clash_points, left_indices = computed["left_clash_points"], computed["left_indices"]
        right_clash_points, right_indices = computed["right_clash_points"], computed["right_indices"]
        front_clash_points, front_indices = computed["front_clash_points"], computed["front_indices"]
        back_clash_points, back_indices = computed["back_clash_points"], computed["back_indices"]
        top_clash_points, top_indices = computed["top_clash_points"], computed["top_indices"]
        min_dist_left, min_dist_right = computed["min_dist_left"], computed["min_dist_right"]
        min_dist_front, min_dist_back = computed["min_dist_front"], computed["min_dist_back"]
        min_dist_top = computed["min_dist_top"]
        R, t = computed["R"], computed["t"]
        self._last_cam_pose = (R, t)

        # ── FIX: use Pose A's own beam pose, not a separately-computed one.
        # computed["transformed"] is the exact array that was rotated
        # (self.beam_points_centered @ R + t), rendered via
        # beam_polydata.points, and raycasted against — the single beam
        # pose in this pipeline now.
        beam_for_analysis = computed["transformed"]

        # ── Grid clash analysis / clearing — matches reference ────────────
        if len(left_clash_points) > 0 or len(right_clash_points) > 0:
            self.analyze_minimum_clashes(
                left_clash_points, right_clash_points, beam_for_analysis)
        else:
            for actor in self.grid_box_actors_left + self.grid_box_actors_right:
                try:
                    self.plotter.remove_actor(actor, render=False)
                except Exception:
                    pass
            self.grid_box_actors_left.clear()
            self.grid_box_actors_right.clear()
            for actor in self.grid_label_actors_left + self.grid_label_actors_right:
                try:
                    self.plotter.remove_actor(actor, render=False)
                except Exception:
                    pass
            self.grid_label_actors_left.clear()
            self.grid_label_actors_right.clear()

        # ── Extract grid labels for recording ──────────────────────────────
        left_grid_labels = []
        right_grid_labels = []
        if self.min_clash_data['left']['grid_indices'] is not None:
            left_grid_labels = [
                self.grid_labels[idx][0]
                for idx in self.min_clash_data['left']['grid_indices']]
        if self.min_clash_data['right']['grid_indices'] is not None:
            right_grid_labels = [
                self.grid_labels[idx][0]
                for idx in self.min_clash_data['right']['grid_indices']]

        # ── Record clash data ──────────────────────────────────────────────
        if (len(left_clash_points) or len(right_clash_points)
                or len(front_clash_points) or len(back_clash_points)
                or len(top_clash_points)):
            record_frame_bidirectional(
                frame_idx=self.current_frame, pose=t, R=R,
                left_clash_points=left_clash_points, left_indices=left_indices,
                right_clash_points=right_clash_points, right_indices=right_indices,
                front_clash_points=front_clash_points, front_indices=front_indices,
                back_clash_points=back_clash_points, back_indices=back_indices,
                top_clash_points=top_clash_points, top_indices=top_indices,
                left_grid_labels=left_grid_labels, right_grid_labels=right_grid_labels,
                min_dist_left=min_dist_left, min_dist_right=min_dist_right,
                min_dist_front=min_dist_front, min_dist_back=min_dist_back,
                min_dist_top=min_dist_top,
            )

        # ── Sync slider + frame-number text ───────────────────────────────
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame)
        self.slider.blockSignals(False)
        if hasattr(self, "_update_frame_counter"):
            self._update_frame_counter(self.current_frame)

        # REMOVED (Pose B elimination): this block used to hold
        # cam_pos = computed["beam_translation"].copy() plus a commented-out
        # self.update_camera_with_rotation(cam_pos, R_matrix) call — both the
        # method and the R_matrix it referenced are gone now. Live
        # camera-follow during playback is camera_follow() inside
        # frame_updater.apply_frame_visualization(), already invoked above
        # (via the apply_frame_visualization() call at the top of this
        # method) using Pose A's own (transformed, R, t) — nothing left to
        # do here.

        self.plotter.add_axes()
        self.plotter.render()

        # Frame 0 running through this pipeline is exactly what gives
        # AcceleratedBidirectionalPenetrationCache its initial_beam_points /
        # base mesh — so this is the single source of truth for "is the
        # current self.cache_manager seeded", used by start_animation() to
        # decide whether it's safe to resume at a non-zero current_frame.
        if self.current_frame == 0:
            self._cache_seeded = True

    # ── Threaded playback path (keeps the Qt event loop responsive) ──────────
    def _dispatch_frame_worker(self, frame_idx: int) -> None:
        """Kick off background computation for `frame_idx` and return
        immediately. _on_frame_computed() is invoked on the main thread once
        the worker finishes (Qt auto-queues cross-thread signal delivery to
        the receiving QObject's thread — `self` lives on the main/GUI thread).

        Only ever ONE worker in flight at a time — see animate_frame()'s
        guard (self._frame_worker is not None) before this is called.

        Deliberately does NOT connect worker.finished to worker.deleteLater().
        deleteLater() schedules the underlying C++ QThread object for
        deletion on the NEXT event-loop pass — which can happen before the
        NEXT animate_frame() tick checks self._frame_worker, leaving a Python
        wrapper around an already-deleted C++ object. Calling .isRunning()
        on that raises "RuntimeError: wrapped C/C++ object of type
        FrameWorker has been deleted". Instead, self._frame_worker is
        cleared to None explicitly in _on_frame_computed()/_on_frame_failed()
        (once we know run() has actually finished), and the QThread object
        itself is left to normal Python reference counting — safe to
        garbage-collect once nothing points to it anymore, since by then
        run() has already returned.
        """
        compute_kwargs = dict(
            total_frames=self.total_frames,
            # FIX: same centered array as update_frame_visualization() —
            # see _apply_scene_data()'s beam_points_centered note. Both
            # the synchronous and threaded paths must agree on which
            # beam-point frame the detector's cache_manager was seeded
            # with, since it stays locked in for the cache's lifetime.
            beam_points=self.beam_points_centered,
            trajectory_points=self.trajectory_points,
            roll=self.trajectory_rolls, pitch=self.trajectory_pitches,
            yaw=self.trajectory_yaws,
            wall_points=self.wall_points, obb_center_local=self.obb_center_local,
            obb_half_extents=self.obb_half_extents, bbox_local_points=self.bbox_local_points,
            cache_manager=self.cache_manager, thickness=self.thickness,
            sd_threshold=self.sd_threshold, x_translation=self.x_translation,
            obb_R_local=self.obb_R_local,
        )
        worker = _FrameWorker(frame_idx, compute_kwargs)
        worker.frame_done.connect(self._on_frame_computed)
        worker.frame_failed.connect(self._on_frame_failed)
        self._frame_worker = worker   # marks "busy" until explicitly cleared below
        worker.start()

    def _on_frame_computed(self, computed: dict) -> None:
        """Runs on the main thread (queued signal from _FrameWorker).
        Applies the precomputed frame, then advances playback."""
        frame_idx = computed["frame"]

        # Clear "busy" the moment we know this worker's run() has finished —
        # but only if this signal actually belongs to the worker we're
        # CURRENTLY tracking (identified by the frame index it was
        # dispatched for). A late/stale signal from an OLDER worker must
        # not clear the slot out from under a NEWER one that may have
        # already been dispatched (e.g. after a fast seek).
        current_worker = getattr(self, "_frame_worker", None)
        if current_worker is not None and getattr(current_worker, "_frame_idx", None) == frame_idx:
            self._frame_worker = None

        if frame_idx != self.current_frame:
            # Stale result — e.g. the user seeked (go_to_frame) or restarted
            # while this frame was still computing in the background.
            # current_frame has already moved on; discard silently.
            print(f"[FrameWorker] discarding stale result for frame {frame_idx} "
                  f"(current_frame is now {self.current_frame})")
            return

        self._apply_computed_frame(computed)

        self.current_frame += self.frame_skip
        if self.current_frame >= len(self.trajectory_points):
            self.current_frame = len(self.trajectory_points) - 1
            self._finish_animation(reason="completed")
        # else: animation_timer's next tick calls animate_frame() again,
        # which dispatches the NEXT frame's worker.

    def _on_frame_failed(self, error_message: str) -> None:
        """A background frame computation raised. Stop playback rather than
        silently hanging (the timer would otherwise keep firing animate_frame(),
        which would just skip every tick forever since current_frame never
        advances while self._frame_worker stays stuck non-None)."""
        print(f"[FrameWorker] ❌ frame computation failed: {error_message}")
        self._frame_worker = None
        self.stop_animation()

    def animate_frame(self):
        """Called by animation_timer.timeout every tick. Dispatches THIS
        frame's heavy computation to a background _FrameWorker instead of
        computing it synchronously — that's what actually keeps the Qt
        event loop responsive during playback (the old QApplication.
        processEvents() call only pumped events BETWEEN frames; it did
        nothing during a single frame's own computation, which is exactly
        where the raycasting-heavy work — and therefore the freeze — lives).

        current_frame is advanced in _on_frame_computed() once the
        background computation finishes and has been applied — NOT here —
        so a slow frame simply means this tick's dispatch attempt below is
        skipped (self._frame_worker is not None guard) rather than two
        frames' worth of detection running concurrently against the same
        cache_manager.
        """
        if self.current_frame >= len(self.trajectory_points):
            self._finish_animation(reason="completed")
            return

        if getattr(self, "_frame_worker", None) is not None:
            # Previous frame's background computation hasn't finished yet —
            # skip this tick rather than starting an overlapping worker.
            # The timer will simply tick again; nothing is lost, current_frame
            # only advances once per completed+applied frame. (Deliberately
            # NOT calling .isRunning() here — see _dispatch_frame_worker()'s
            # docstring for why that can raise on an already-deleted worker.)
            return

        # REMOVED (Pose B elimination): this used to call
        # self.apply_car_rotation(self.current_frame) here for an "instant
        # glide" render before dispatching the background worker. In
        # practice it never assigned the result to beam_polydata.points, so
        # plotter.render() below was repainting the UNCHANGED beam position
        # every tick — the glide was already a no-op, just with extra
        # matrix-multiply and Pose-B-divergence cost for nothing. Removing
        # it changes no visible behavior; the beam still updates correctly
        # once _on_frame_computed() applies the real (Pose A) result.
        self.plotter.render()

        # ── BUG-1 FIX: pump the Qt event loop so tab switches and button
        # clicks are processed immediately after the beam-move render above.
        QApplication.processEvents()

        self._dispatch_frame_worker(self.current_frame)

    def toggle_play_pause(self):
        if self.is_playing:
            self.stop_animation()
        else:
            self.start_animation()

    def start_animation(self):
        if self.trajectory_points is None:
            return

        # Build (once) whatever base meshes / orientation this pass needs
        # — deferred here from scene load, see _ensure_meshes_ready()'s
        # docstring. No-ops immediately if already built, or if
        # x_translation<=0 (trajectory-only — no probe geometry needed).
        self._ensure_meshes_ready()

        # ── "initial_beam_points required for first frame" ───────────────
        # frame_updater.compute_frame_data() (and the underlying
        # AcceleratedBidirectionalPenetrationCache) MUST process frame 0 once
        # per cache-manager instance before any other frame can be queried,
        # or the cache raises "initial_beam_points required for first frame".
        #
        # NOTE: _ensure_meshes_ready() above guarantees meshes_initialized
        # is now True (skipped only in trajectory-only mode) by the time we
        # reach here. That does NOT change this method's own bookkeeping:
        # _cache_seeded still tracks whether compute_frame_data() has
        # actually processed FRAME 0 specifically (populating the
        # recorder's dedup state and the cache's frame-0-derived
        # internals), which is a distinct step from base-mesh creation and
        # still only happens here.
        #
        # PREVIOUSLY this reset self.current_frame = 0 unconditionally on
        # every single Play press — including a pause→resume where the app
        # was never closed and the cache is still perfectly valid. That threw
        # away the paused position and restarted the whole animation from
        # frame 0 every time (e.g. pause at frame 13100, hit Play again,
        # jumps back to frame 0). It only actually needs to reseed when the
        # cache genuinely hasn't processed frame 0 yet — tracked here via
        # self._cache_seeded, which is set True at the end of
        # update_frame_visualization() whenever current_frame == 0, and reset
        # to False anywhere self.cache_manager is replaced or .cleanup()'d.
        if not getattr(self, "_cache_seeded", False):
            # First Play this session (or cache was invalidated since) —
            # seed frame 0 synchronously, but remember where the user
            # actually wants to resume so we don't strand them at frame 0.
            #
            # This is also the single, unambiguous "a brand-new detection
            # pass is starting" moment — the cache needing frame 0 reseeded
            # and the recorder needing its dedup/accumulation state cleared
            # are the same event. Without this reset, a second run in the
            # same session would either get silently skipped entirely
            # (recording_saved_* already True) or have new frame data
            # silently dropped for any frame_idx already seen in the
            # previous run (already_saved_frames_* never cleared) — see
            # reset_recording_state()'s docstring in recorder.py.
            resume_at = self.current_frame
            self.current_frame = 0
            self.cache_manager.cleanup()   # discard any stale partial cache state
            reset_recording_state()
            configure_recording_output(getattr(self, "save_dir_file", None) or get_cache_dir())
            self.update_frame_visualization()   # seeds frame 0; sets _cache_seeded=True
            self.current_frame = resume_at if resume_at > 0 else self.frame_skip
        # else: cache already seeded this session — resume exactly where the
        # animation was paused (self.current_frame is untouched by
        # stop_animation(), so playback continues contiguously).

        self.is_playing = True
        self.animation_timer.start(self.animation_speed)
        self._sync_play_button()

    # ── recording auto-save ──────────────────────────────────────────
    def _save_recordings(self, reason: str = "manual") -> None:
        """Persist any recorded clash frames to disk, and notify
        PostProcessMixin (Screen 4) in real time via clash_data_bus so it can
        reload — whether this is the first time data has ever appeared for
        this project, or a re-run overwriting previously-loaded data.

        Safe to call multiple times — save_recorded_frames_bidirectional()
        only re-writes a direction whose recorded_frames_X has grown since
        its last write (see _last_saved_len in recorder.py), so repeated
        calls with nothing new are cheap no-ops.

        Called ONLY from genuine completion/restart/switch/close events —
        never from stop_animation() (Pause), which is a routine, frequent
        user action here and must have no save side effect, and never from
        restart_with_new_x_translation() (a deliberate discard, see there):
          • _finish_animation()              — trajectory reaches its last
                                               frame on its own (not a manual
                                               pause)
          • on_scene_loaded()                — project switch, saves the
                                               OUTGOING project's data first
                                               (defensive fallback — see
                                               confirm_can_switch_project())
          • confirm_can_switch_project()      — the hub-driven, dialog-gated
                                               project switch (preferred path)
          • closeEvent()                      — app window closing
        """
        try:
            print(f"[Recorder] 💾 auto-save triggered ({reason})")
            output_dir = getattr(self, "save_dir_file", None) or get_cache_dir()
            saved_dirs = save_recorded_frames_bidirectional(output_dir=output_dir) or []

            project_path = getattr(self, "_active_project_path", "") or output_dir
            if saved_dirs and project_path:
                print(f"[Recorder] 📡 notifying clash_data_bus: {saved_dirs} @ {project_path}")
                clash_data_bus.clash_data_saved.emit(project_path, saved_dirs)
        except Exception:
            import traceback
            print("[Recorder] ❌ auto-save failed:")
            traceback.print_exc()

    def stop_animation(self):
        """Pause playback only. Deliberately does NOT save anything.

        toggle_play_pause() calls this for every manual Pause, and Pause →
        Resume is a routine, frequent user action in this UI (there is no
        separate hard-Stop button) — it must not have side effects like
        writing files to disk every time. Saving now happens ONLY at
        genuine completion/restart/switch events, each of which calls
        _save_recordings() explicitly and separately from this method:
          • _finish_animation()              — trajectory reaches its last frame
          • on_scene_loaded()                — outgoing project, before switch
          • confirm_can_switch_project()      — confirmed HUB navigation
          • closeEvent() (GeneralViewTab)     — app window closing

        Also waits for any in-flight background _FrameWorker to finish
        before returning. Several callers (go_to_frame(), closeEvent(),
        confirm_can_switch_project(), restart_with_new_x_translation())
        touch cache_manager / recorder state immediately after calling this
        and assume it's now safe to do so synchronously — that's only true
        once any frame computation still running in the background has
        actually finished (stopping animation_timer only prevents NEW
        ticks; it does nothing about a worker already dispatched by a
        previous tick).
        """
        self.is_playing = False
        self.animation_timer.stop()
        self._sync_play_button()

        worker = getattr(self, "_frame_worker", None)
        if worker is not None:
            try:
                if worker.isRunning():
                    worker.wait()
            except RuntimeError:
                # Underlying C++ QThread object was already destroyed (e.g.
                # via some other teardown path) — nothing left to wait for.
                pass
            self._frame_worker = None

    def _finish_animation(self, reason: str = "completed") -> None:
        """Called ONLY when the trajectory genuinely reaches its last frame
        on its own (see animate_frame()) — not from a manual Pause. This is
        the one moment during ordinary playback where saving is actually
        warranted: the run is done, there's nothing left to resume into.
        """
        self.stop_animation()
        self._save_recordings(reason=reason)

    # NOTE: app-close handling (scenarios 4 & 5) does NOT live here.
    # GeneralViewTab.closeEvent() is defined directly on that class (in
    # general_view_tab.py) — a method defined directly on a subclass is
    # ALWAYS found before anything in its base classes, regardless of MRO
    # order, so a closeEvent() placed in this mixin would never be reached
    # while GeneralViewTab keeps its own. See general_view_tab.py's
    # closeEvent() for the actual dialog + save logic.

    # ── Project-hub switch handling (scenario 7) ─────────────────────────
    def confirm_can_switch_project(self) -> bool:
        """
        Call this from ProjectScreen's "← HUB" button handler
        (_on_back_to_hub_clicked) BEFORE it emits go_hub — that signal just
        switches which screen is visible with no way to veto it
        retroactively once it's fired, and the HUB button lives in the
        shared header, reachable from every tab (Config / General View /
        Post-Process / Report), not just this one.

        Returns True if it's safe to proceed (nothing was running, or the
        user confirmed and the current pass's data has already been
        flushed). Returns False if the user cancelled — the caller must
        abort the navigation entirely and leave the simulation untouched.

        Example wiring (project_screen.py):
            def _on_back_to_hub_clicked(self):
                if not self._general_view_tab.confirm_can_switch_project():
                    return   # user said No / closed the dialog — stay put
                self.go_hub.emit()
        """
        if not getattr(self, "is_playing", False):
            return True   # nothing running — nothing to confirm

        reply = QMessageBox.question(
            self, "Simulation Running",
            "A simulation is running on this project. Leaving to the HUB "
            "will end it. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return False   # No/cancel — caller must NOT navigate away

        self.stop_animation()   # pure pause — stop the timer before saving
        self._save_recordings(reason="project-switch-confirmed")
        return True

    def restart_animation(self):
        self.current_frame = 0
        self.is_playing = False
        self.animation_timer.stop()
        self._sync_play_button()
        self.visualize_initial_state()

        # "Restart" means the user wants a fresh pass, not a resume-in-place.
        # Without this, _cache_seeded would still be True from the previous
        # run, so the next start_animation() would skip reseeding — and the
        # recorder would keep its old already_saved_frames_* entries for the
        # low frame indices, silently dropping this new pass's data for them.
        if getattr(self, "cache_manager", None) is not None:
            try:
                self.cache_manager.cleanup()
            except Exception:
                pass
        self._cache_seeded = False

    def _sync_play_button(self) -> None:
        """Keep the ▶ / ⏸ button label in sync with self.is_playing.
        Safe to call even if `play_btn` doesn't exist on this instance."""
        btn = getattr(self, "play_btn", None)
        if btn is not None:
            btn.setText("⏸" if self.is_playing else "▶")

    def next_frame(self):
        if self.trajectory_points is None:
            return
        if self.current_frame < len(self.trajectory_points) - self.frame_skip:
            self.current_frame += self.frame_skip
        else:
            self.current_frame = len(self.trajectory_points) - 1
        self.update_frame_visualization()

    def prev_frame(self):
        if self.trajectory_points is None:
            return
        if self.current_frame >= self.frame_skip:
            self.current_frame -= self.frame_skip
        else:
            self.current_frame = 0
        self.update_frame_visualization()

    def go_to_frame(self):
        try:
            frame_num = int(self.nav_frame_edit.text().strip())
        except ValueError:
            return
        if self.trajectory_points is None or not (0 <= frame_num < len(self.trajectory_points)):
            return
        if self.is_playing:
            self.stop_animation()
        # When seeking to any non-zero frame manually, reset the cache so
        # it is not in a stale partial state. Mark it as not-yet-seeded so
        # the next start_animation() knows it must reseed frame 0 before
        # resuming playback (rather than assuming it can resume in place).
        if frame_num != 0 and getattr(self, "cache_manager", None) is not None:
            try:
                self.cache_manager.cleanup()
                self._cache_seeded = False
            except Exception:
                pass
        self.current_frame = frame_num
        self.update_frame_visualization()

    def restart_with_new_x_translation(self):
        if self.trajectory_points is None:
            return

        try:
            new_x_translation = float(self.xtrans_edit.text())
        except ValueError:
            return
    
        if new_x_translation < 0:
            QMessageBox.warning(self, "Invalid Threshold",
                                "Threshold must be 0 (trajectory only) or greater.")
            return

        if new_x_translation == 0:
            print("[GeneralView] ℹ️ Threshold set to 0 — switching to "
                  "TRAJECTORY-ONLY mode: no probe meshes will be built and "
                  "clash detection is disabled for this pass (see "
                  "frame_updater.compute_frame_data()'s x_translation<=0 "
                  "branch).")

        was_playing = self.is_playing

        if self.is_playing:
            self.stop_animation()   # pure pause now — just stops the timer

        # Per spec: restarting with a new x_translation is a deliberate
        # DISCARD, not a save point — do NOT call _save_recordings() here.
        # Whatever was pending in memory for the old x_translation (at most
        # _CHUNK_SIZE frames, since earlier chunks were already streamed to
        # disk as the old pass ran) is simply dropped by reset_recording_state()
        # the next time start_animation() begins the new pass below. Any
        # chunk(s) the OLD pass already flushed to disk remain there until
        # the NEW pass's own first chunk flush truncates/overwrites them —
        # there is no separate "undo" of already-written chunks.

        old_x_translation = self.x_translation
        self.x_translation = new_x_translation

        print(f"Old x_translation: {old_x_translation}")
        print(f"New x_translation: {new_x_translation}")

        old_cache_manager = self.cache_manager
        try:
            old_cache_manager.cleanup()
        except Exception as e:
            print(f"Cache cleanup warning: {e}")

        self.cache_manager = AcceleratedBidirectionalPenetrationCache(
            cache_dir=get_cache_dir("penetration_cache_bidirectional")
        )

        # FIX: a brand-new cache_manager instance starts with pca_info=None
        # and _orientation_manually_set=False. Left as-is, the NEXT base-
        # mesh build (lazily on frame 0, or eagerly below) would silently
        # discard whatever orientation was manually committed earlier this
        # session via interactive_alignment_gui() — recomputing raw PCA
        # from scratch and popping the UNCONTROLLED plain-preview popup
        # instead of reusing the committed one. Orientation (which way is
        # TOP/LEFT/FRONT etc.) has nothing to do with x_translation (that
        # only changes probe LENGTH), so it's safe and correct to carry it
        # straight over to the new instance.
        if getattr(old_cache_manager, "pca_info", None) is not None:
            self.cache_manager.pca_info = old_cache_manager.pca_info
            self.cache_manager._orientation_manually_set = \
                old_cache_manager._orientation_manually_set
            print("[GeneralView] ✅ carried over previously-committed "
                  "orientation to the new cache_manager for this restart "
                  "(no re-alignment popup needed)")

        # FIX: project_dir/beam_ids live on the cache_manager INSTANCE (see
        # detector.py's set_project_context()), not on this mixin — a
        # brand-new instance always starts with project_dir=None. Without
        # this call, _load_selective_indices() unconditionally fails its
        # `if not self.project_dir:` guard and every direction silently
        # falls back to the FULL, unfiltered beam instead of its saved
        # per-side (Left/Right/Top/Front/Rear) selection. Must run BEFORE
        # create_base_meshes_bidirectional_local() below — same ordering
        # requirement documented on set_project_context() itself.
        self.cache_manager.set_project_context(
            project_dir=self._active_project_path, beam_ids=self.beam_ids)

        self._cache_seeded = False     # brand-new cache instance — not seeded yet
        self._preflight_shown = False  # brand-new cache instance — preflight not run yet

        self.current_frame = 0

        # FIX: obb_center_local / obb_half_extents were frozen at the OLD
        # x_translation by _initialize_beam_geometry() (called once, back
        # in _start_clash_engine()). Without recomputing them here, the
        # wall-point candidate WINDOW that filter_wall_points_by_obb()
        # filters against every frame stays sized to the OLD threshold,
        # while the probe mesh LENGTH (rebuilt below, or lazily on next
        # frame 0, from the NEW self.x_translation) uses the new one. That
        # mismatch — OBB window and probe reach disagreeing — was flagged
        # during debugging as the leading suspect for clash points that
        # don't correspond to where the probe mesh actually sits. Recompute
        # both together, right after the threshold assignment above, so
        # they can never drift apart again.
        self._initialize_beam_geometry()

        self.visualize_initial_state()

        # Mesh rebuild for the new threshold is DEFERRED to
        # start_animation() → _ensure_meshes_ready(), same as scene load —
        # see that method's docstring. Previously this eagerly rebuilt
        # meshes right here, which caused two problems:
        #   1. If the user had also already loaded meshes once at scene
        #      load (before this file's eager-loading was itself removed),
        #      the SAME expensive build could run twice for one Play press.
        #   2. Setting the Threshold field to 0 (trajectory-only mode) and
        #      pressing Apply crashed here: create_base_meshes_
        #      bidirectional_local() → _create_direction_mesh_local() calls
        #      o3d.geometry.TriangleMesh.create_box(width=length, ...) with
        #      length=self.x_translation=0, and Open3D raises
        #      "RuntimeError: ... width <= 0" — there is no valid probe
        #      geometry to build at zero reach. _ensure_meshes_ready()
        #      guards against this (x_translation<=0 → skip mesh build
        #      entirely, matching frame_updater.compute_frame_data()'s own
        #      existing "trajectory-only, clash detection OFF" branch,
        #      which needs no base meshes at all).
        #
        # pca_info / _orientation_manually_set were already carried over
        # to the new cache_manager above, so _ensure_meshes_ready() will
        # reuse that committed orientation (no re-alignment popup) once it
        # actually runs — whether that's immediately below (was_playing)
        # or on the next Play press.

        if was_playing:
            self.start_animation()
            
    def update_frame_skip(self):
        try:
            self.frame_skip = int(self.skip_frame_edit.text() or 0)
        except ValueError:
            pass
            
    # ── Chase-camera offset (View panel: Cam Distance / Cam Height) ──────────
    def set_camera_offset(self, distance: float, height: float) -> None:
        """Set how far behind (`distance`) and above (`height`) the car the
        follow camera sits, in metres. Called by GeneralViewTab whenever a
        View-panel field changes.

        While playing, the very next applied frame already uses the new
        values (apply_frame_visualization() re-places the camera every
        frame). While paused nothing would re-place it, so do it here —
        otherwise the change stays invisible until a frame is stepped.
        """
        self.camera_distance = float(distance)
        self.camera_height = float(height)
        if not self.is_playing:
            self._refresh_chase_camera()

    def _refresh_chase_camera(self) -> None:
        """Re-place the follow camera from the LAST applied frame's pose.
        MAIN-THREAD ONLY. No-op until a frame has been applied (and again
        after visualize_initial_state() / a project switch clears the pose)."""
        pose = getattr(self, "_last_cam_pose", None)
        if pose is None:
            return
        R, t = pose
        R_global = getattr(self, "R_global", None)
        R_cam = R if R_global is None else (R_global @ R)
        camera_follow(self.plotter, R_cam, t,
                      distance=self.camera_distance, height=self.camera_height)
        self.plotter.render()

    def analyze_minimum_clashes(self, left_clash_points, right_clash_points, current_beam):
        """Find minimum clash distances and create SEPARATE square boxes for left and right sides"""
        
        if not hasattr(self, 'grid_points') or self.grid_points is None:
            return
        
        # Clear previous box actors
        for actor in self.grid_box_actors_left + self.grid_box_actors_right:
            try:
                self.plotter.remove_actor(actor, render=False)
            except:
                pass
        self.grid_box_actors_left.clear()
        self.grid_box_actors_right.clear()
        
        # Clear previous label actors
        for actor in self.grid_label_actors_left + self.grid_label_actors_right:
            try:
                self.plotter.remove_actor(actor, render=False)
            except:
                pass
        self.grid_label_actors_left.clear()
        self.grid_label_actors_right.clear()
        
        # Reset min clash data
        self.min_clash_data = {
            'left': {'distance': None, 'clash_point': None, 'grid_indices': None},
            'right': {'distance': None, 'clash_point': None, 'grid_indices': None}
        }
        
        # ========== LEFT SIDE ANALYSIS (RED BOX) ==========
        if len(left_clash_points) > 0:
            # Find closest clash point to beam
            beam_center = np.mean(current_beam, axis=0)
            distances_left = np.linalg.norm(left_clash_points - beam_center, axis=1)
            min_idx_left = np.argmin(distances_left)
            min_clash_left = left_clash_points[min_idx_left]
            min_dist_left = distances_left[min_idx_left]
            
            # Find nearest 4 grid points
            nearest_indices_left, nearest_points_left = find_nearest_grid_points(
                self.grid_points, min_clash_left, count=4
            )
            
            if len(nearest_indices_left) == 4:
                # Store data
                self.min_clash_data['left'] = {
                    'distance': min_dist_left,
                    'clash_point': min_clash_left,
                    'grid_indices': nearest_indices_left
                }
                
                # ✅ CREATE RED RECTANGULAR BOX FOR LEFT SIDE
                # Sort points to create proper rectangle
                sorted_points = sort_rectangle_points_robust(nearest_points_left)
                
                # Create wireframe box
                box_lines = create_grid_box_lines(sorted_points)
                
                # Add all lines with RED color
                for i, line in enumerate(box_lines):
                    actor = self.plotter.add_mesh(
                        line, color='red', line_width=5,
                        render=False, name=f'left_box_line_{i}'
                    )
                    self.grid_box_actors_left.append(actor)
                
                # Add labels for the 4 grid points (WHITE text on RED background)
                label_texts = []
                label_positions = []
                for idx in nearest_indices_left:
                    label, coord = self.grid_labels[idx]
                    label_texts.append(f"L-{label}")
                    # Position labels slightly above the points
                    label_positions.append(np.array(coord) + np.array([0, 0, 0.15]))
                
                label_actor = self.plotter.add_point_labels(
                    label_positions,
                    label_texts,
                    point_size=0,
                    font_size=14,
                    text_color="white",
                    shape_color="red",
                    shape_opacity=0.8,
                    bold=True,
                    show_points=False,
                    render=False,
                    name='left_grid_labels'
                )
                self.grid_label_actors_left.append(label_actor)
                
                # Add a sphere marker at the clash point
                marker = pv.Sphere(radius=0.001, center=min_clash_left)
                marker_actor = self.plotter.add_mesh(
                    marker, color='red', opacity=0.9,
                    render=False, name='left_clash_marker'
                )
                self.grid_box_actors_left.append(marker_actor)
                
                # Print grid info
                grid_labels = [self.grid_labels[idx][0] for idx in nearest_indices_left]
                print(f"   LEFT Grid Box: {', '.join(grid_labels)}")
                print(f"      Distance: {min_dist_left:.3f}m | X-Translation: +{self.x_translation:.2f}m")
        
        # ========== RIGHT SIDE ANALYSIS (BLUE BOX) ==========
        if len(right_clash_points) > 0:
            # Find closest clash point to beam
            beam_center = np.mean(current_beam, axis=0)
            distances_right = np.linalg.norm(right_clash_points - beam_center, axis=1)
            min_idx_right = np.argmin(distances_right)
            min_clash_right = right_clash_points[min_idx_right]
            min_dist_right = distances_right[min_idx_right]
            
            # Find nearest 4 grid points
            nearest_indices_right, nearest_points_right = find_nearest_grid_points(
                self.grid_points, min_clash_right, count=4
            )
            
            if len(nearest_indices_right) == 4:
                # ✅ CHECK: If right side shares same grid points as left side
                same_as_left = False
                if (self.min_clash_data['left']['grid_indices'] is not None and 
                    set(nearest_indices_right) == set(self.min_clash_data['left']['grid_indices'])):
                    same_as_left = True
                    print("   ⚠️  Right side clashes in SAME grid area as Left side")
                
                # Store data
                self.min_clash_data['right'] = {
                    'distance': min_dist_right,
                    'clash_point': min_clash_right,
                    'grid_indices': nearest_indices_right,
                    'same_as_left': same_as_left
                }
                
                # ✅ CREATE BLUE RECTANGULAR BOX FOR RIGHT SIDE
                # Sort points to create proper rectangle
                sorted_points = sort_rectangle_points_robust(nearest_points_right)
                
                # If same grid as left, offset the box slightly for visibility
                if same_as_left:
                    # Offset the box by 0.1 meters in X direction for visibility
                    sorted_points = sorted_points + np.array([0.1, 0, 0])
                    print("   ↳ Offset right box by +0.1m in X for visibility")
                
                # Create wireframe box
                box_lines = create_grid_box_lines(sorted_points)
                
                # Add all lines with BLUE color
                for i, line in enumerate(box_lines):
                    actor = self.plotter.add_mesh(
                        line, color='blue', line_width=5,
                        render=False, name=f'right_box_line_{i}'
                    )
                    self.grid_box_actors_right.append(actor)
                
                # Add labels for the 4 grid points (WHITE text on BLUE background)
                label_texts = []
                label_positions = []
                for idx in nearest_indices_right:
                    label, coord = self.grid_labels[idx]
                    label_texts.append(f"R-{label}")
                    # Position labels slightly above the points
                    label_positions.append(np.array(coord) + np.array([0, 0, 0.15]))
                
                label_actor = self.plotter.add_point_labels(
                    label_positions,
                    label_texts,
                    point_size=0,
                    font_size=14,
                    text_color="white",
                    shape_color="blue",
                    shape_opacity=0.8,
                    bold=True,
                    show_points=False,
                    render=False,
                    name='right_grid_labels'
                )
                self.grid_label_actors_right.append(label_actor)
                
                # Add a sphere marker at the clash point
                marker = pv.Sphere(radius=0.001, center=min_clash_right)
                marker_actor = self.plotter.add_mesh(
                    marker, color='blue', opacity=0.9,
                    render=False, name='right_clash_marker'
                )
                self.grid_box_actors_right.append(marker_actor)
                
                # Print grid info
                grid_labels = [self.grid_labels[idx][0] for idx in nearest_indices_right]
                if same_as_left:
                    print(f"   RIGHT Grid Box: {', '.join(grid_labels)} [SAME AS LEFT, OFFSET VISUAL]")
                else:
                    print(f"   RIGHT Grid Box: {', '.join(grid_labels)}")
                print(f"      Distance: {min_dist_right:.3f}m | X-Translation: -{self.x_translation:.2f}m")