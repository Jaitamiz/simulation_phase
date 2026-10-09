# clash_tools_mixin.py
"""
ClashToolsMixin — Clip / Measure / Erase tools for GeneralViewTab.

Ported from ClashDetectionWidget's clip_box-crop, distance-measurement,
and context-sensitive erase logic, adapted to the attribute names already
owned by ClashSimMixin (simulation_engine.py):

    wall_points / wall_colors / wall_actor
    beam_points / beam_colors / beam_actor / beam_polydata
    trajectory_points / smoothed_trajectory / trajectory_actor
    grid_points / grid_actor
    cache_manager, plotter (property on GeneralViewTab)

Mix this in ALONGSIDE ClashSimMixin:

    class GeneralViewTab(ClashSimMixin, ClashToolsMixin, QWidget):
        ...

Wire the three Tool-panel buttons (Clip / Measure / Erase) to:
    self.toggle_crop_mode
    self.toggle_distance_measurement
    self.erase_last_point

All three modes are mutually exclusive and require the animation to be
paused (same guard as the original widget). Status text, if present, is
written to `self.tool_status_label` when that attribute exists (wired
optionally from GeneralViewTab) — otherwise it's just printed.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv
import vtk

try:
    from PyQt5.QtWidgets import QMessageBox
except ImportError:  # pragma: no cover
    QMessageBox = None


class ClashToolsMixin:
    """Clip (crop) / Measure (distance) / Erase tools."""

    # =====================================================================
    # init — call once from GeneralViewTab.__init__ (after _init_clash_engine)
    # =====================================================================
    def _init_clash_tools(self):
        # Crop / clip state
        self.crop_mode = False
        self.crop_box_widget = None
        self.crop_bounds = None  # previous full state, for undo
        self.original_wall_points = self.original_wall_colors = None
        self.original_beam_points = self.original_beam_colors = None
        self.original_trajectory_points = self.original_smoothed_trajectory = None

        # Measurement state
        self._distance_enabled = False
        self.clicked_points = []
        self.current_measurement_actors = []
        self.all_measurements = []
        self.line_colors = ['magenta', 'yellow', 'lime', 'orange']
        self.color_index = 0
        self.point_picker = None

        # Single-point picker (used by erase fallback / debugging)
        self.picking_enabled = False
        self.single_point_picker = None
        self.picked_disk_actor = None
        self.picked_text_actor = None
        self.picked_box_actor = None

    # ------------------------------------------------------------------
    def _set_status(self, text: str, style: str = "") -> None:
        if hasattr(self, "tool_status_label") and self.tool_status_label is not None:
            self.tool_status_label.setText(text)
            if style:
                self.tool_status_label.setStyleSheet(style)
        print(text)

    def _warn_if_playing(self, action: str) -> bool:
        """Return True (and warn) if animation is playing and must be paused first."""
        if getattr(self, "is_playing", False):
            if QMessageBox is not None:
                QMessageBox.warning(self, "Animation Playing",
                                     f"Please pause the animation before {action}.")
            else:
                print(f"[ClashTools] Pause the animation before {action}.")
            return True
        return False

    def _disable_other_modes(self, keep: str) -> None:
        """Mutual exclusivity: entering one tool mode exits the others."""
        if keep != "crop" and self.crop_mode:
            self.exit_crop_mode()
        if keep != "distance" and self._distance_enabled:
            self._disable_distance_measurement()
        if keep != "single" and self.picking_enabled:
            self._disable_point_picker()

    # =====================================================================
    # CLIP / CROP
    # =====================================================================
    def toggle_crop_mode(self) -> None:
        if self._warn_if_playing("cropping"):
            return
        if getattr(self, "wall_points", None) is None:
            if QMessageBox is not None:
                QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        self.crop_mode = not self.crop_mode
        if self.crop_mode:
            self._disable_other_modes("crop")
            self.enter_crop_mode()
        else:
            self.exit_crop_mode()

    def enter_crop_mode(self) -> None:
        mins = self.wall_points.min(axis=0)
        maxs = self.wall_points.max(axis=0)
        center = (mins + maxs) / 2
        size = maxs - mins

        bounds = [
            center[0] - size[0] * 0.4, center[0] + size[0] * 0.4,
            center[1] - size[1] * 0.4, center[1] + size[1] * 0.4,
            center[2] - size[2] * 0.4, center[2] + size[2] * 0.4,
        ]

        self.crop_box_widget = self.plotter.add_box_widget(
            callback=lambda *_: None,  # live preview not required; apply reads planes directly
            bounds=bounds,
            rotation_enabled=True,
            color='yellow',
            use_planes=False,
        )
        try:
            self.crop_box_widget.GetHandleProperty().SetPointSize(3)
            self.crop_box_widget.GetHandleProperty().SetColor(1, 1, 0)
            self.crop_box_widget.OutlineCursorWiresOn()
        except Exception:
            pass

        self.plotter.add_key_event('a', self.apply_crop)
        self.plotter.add_key_event('A', self.apply_crop)
        self.plotter.add_key_event('Escape', self.cancel_crop)
        self.plotter.add_key_event('u', self.undo_crop)
        self.plotter.add_key_event('U', self.undo_crop)
        self.plotter.add_key_event('r', self.reset_to_original)
        self.plotter.add_key_event('R', self.reset_to_original)

        self.plotter.render()
        self._set_status("✂️ CROP MODE: drag handles | Apply / Undo / Reset via toolbar")

    def exit_crop_mode(self) -> None:
        self.crop_mode = False
        if self.crop_box_widget is not None:
            try:
                self.crop_box_widget.Off()
            except Exception:
                pass
            self.crop_box_widget = None
        self._set_status("Crop mode: disabled")

    def cancel_crop(self) -> None:
        if self.crop_mode:
            self.exit_crop_mode()

    def apply_crop(self) -> None:
        """Crop wall / beam / trajectory point clouds to the box-widget volume."""
        if not self.crop_mode or self.crop_box_widget is None:
            self._set_status("⚠️ Crop mode is not active.")
            return

        # Preserve clash actors (rebuilt by cache_manager, not part of main arrays)
        preserved_left = getattr(self, "clash_actor_left", None)
        preserved_right = getattr(self, "clash_actor_right", None)

        # Save the original (uncropped) data exactly once
        if self.original_wall_points is None:
            self.original_wall_points = self.wall_points.copy()
            self.original_wall_colors = (
                self.wall_colors.copy() if self.wall_colors is not None else None)
            self.original_beam_points = self.beam_points.copy()
            self.original_beam_colors = (
                self.beam_colors.copy() if self.beam_colors is not None else None)
            self.original_trajectory_points = self.trajectory_points.copy()
            self.original_smoothed_trajectory = self.smoothed_trajectory.copy()

        # Save the *previous* state for one-level undo
        self.crop_bounds = (
            self.wall_points.copy(), None if self.wall_colors is None else self.wall_colors.copy(),
            self.beam_points.copy(), None if self.beam_colors is None else self.beam_colors.copy(),
            self.trajectory_points.copy(), self.smoothed_trajectory.copy(),
        )

        planes = vtk.vtkPlanes()
        self.crop_box_widget.GetPlanes(planes)
        normals = planes.GetNormals()
        points_vtk = planes.GetPoints()

        def inside_mask(points: np.ndarray) -> np.ndarray:
            mask = np.ones(len(points), dtype=bool)
            for i in range(6):
                normal = np.array(normals.GetTuple(i))
                point_on_plane = np.array(points_vtk.GetPoint(i))
                d = np.dot(points - point_on_plane, normal)
                mask &= (d <= 0)
            return mask

        wall_mask = inside_mask(self.wall_points)

        # BUG-2 FIX: self.beam_points is in LOCAL object space (raw model
        # coordinates). The crop box widget lives in WORLD space (same
        # coordinate system as wall_points / the viewport). Applying the
        # world-space plane equations directly to local-space beam_points
        # means the clip boundary is in the wrong place — the beam appears
        # to clip incorrectly at both idle (frame 0) and during simulation.
        #
        # Fix: transform beam_points to world space at the CURRENT frame
        # (or frame 0 if the animation has not started) before testing
        # inside_mask, then apply the resulting boolean mask back onto the
        # local-space array so downstream code (apply_car_rotation, OBB
        # geometry) continues to work with local coordinates.
        try:
            beam_world = self.apply_car_rotation(self.current_frame)[0]
        except Exception:
            # Fallback: use raw local points if rotation data isn't ready yet
            beam_world = self.beam_points
        beam_mask = inside_mask(beam_world)

        traj_mask = inside_mask(self.trajectory_points)

        # ✅ Guard: an empty wall or beam crop would crash downstream geometry
        # (bounding_box_object, polydata rebuild, etc.) — refuse and roll back
        # rather than committing a crop that leaves either cloud with 0 points.
        if not wall_mask.any() or not beam_mask.any():
            self.crop_bounds = None  # nothing was actually changed, discard the snapshot
            empty = []
            if not wall_mask.any():
                empty.append("wall")
            if not beam_mask.any():
                empty.append("beam")
            self._set_status(
                f"⚠️ Crop box excludes all {'/'.join(empty)} points — "
                f"shrink/move the box and try again."
            )
            return

        self.wall_points = self.wall_points[wall_mask]
        if self.wall_colors is not None:
            self.wall_colors = self.wall_colors[wall_mask]

        self.beam_points = self.beam_points[beam_mask]
        if self.beam_colors is not None:
            self.beam_colors = self.beam_colors[beam_mask]

        if traj_mask.any():
            self.trajectory_points = self.trajectory_points[traj_mask]
            self.smoothed_trajectory = self.smoothed_trajectory[traj_mask]
        # else: keep full trajectory — cropping it empty would break playback

        self.exit_crop_mode()
        self._rebuild_after_geometry_change(
            preserved_left=preserved_left, preserved_right=preserved_right)

        self._set_status(
            f"✅ Crop applied — wall:{len(self.wall_points):,} "
            f"beam:{len(self.beam_points):,} traj:{len(self.trajectory_points):,}")

    def undo_crop(self) -> None:
        if self.crop_bounds is None:
            self._set_status("⚠️ No crop to undo.")
            return

        preserved_left = getattr(self, "clash_actor_left", None)
        preserved_right = getattr(self, "clash_actor_right", None)

        (self.wall_points, self.wall_colors,
         self.beam_points, self.beam_colors,
         self.trajectory_points, self.smoothed_trajectory) = self.crop_bounds
        self.crop_bounds = None

        self._rebuild_after_geometry_change(
            preserved_left=preserved_left, preserved_right=preserved_right)
        self._set_status("↶ Crop undone.")

    def reset_to_original(self) -> None:
        if self.original_wall_points is None:
            self._set_status("⚠️ No original data saved.")
            return

        self.wall_points = self.original_wall_points.copy()
        self.wall_colors = (
            self.original_wall_colors.copy() if self.original_wall_colors is not None else None)
        self.beam_points = self.original_beam_points.copy()
        self.beam_colors = (
            self.original_beam_colors.copy() if self.original_beam_colors is not None else None)
        self.trajectory_points = self.original_trajectory_points.copy()
        self.smoothed_trajectory = self.original_smoothed_trajectory.copy()
        self.crop_bounds = None

        self._rebuild_after_geometry_change()
        self._set_status("🔄 Reset to original point clouds.")

    def _rebuild_after_geometry_change(self, preserved_left=None, preserved_right=None) -> None:
        """Common rebuild path used by apply_crop / undo_crop / reset_to_original."""
        self.current_frame = 0
        if hasattr(self, "smoothed_camera_pos"):
            self.smoothed_camera_pos = None
        if getattr(self, "cache_manager", None) is not None:
            try:
                self.cache_manager.cleanup()
            except Exception:
                pass

        # After a crop self.beam_points is a subset of the already-centred
        # array, so its XY centroid has drifted.  Re-apply the same
        # "subtract XY mean, keep Z" centring that _apply_scene_data does
        # at load time, then update original_beam_center so apply_car_rotation
        # uses the correct pivot for this (potentially smaller) cloud.
        if getattr(self, "beam_points", None) is not None:
            xy_drift = self.beam_points.mean(axis=0)
            self.beam_points -= np.array([xy_drift[0], xy_drift[1], 0.0])
            self.original_beam_center = self.beam_points.mean(axis=0)

        # Beam geometry (obb, polydata) is derived from beam_points — rebuild it.
        if hasattr(self, "_initialize_beam_geometry"):
            self._initialize_beam_geometry()

        self.visualize_initial_state()  # ClashSimMixin: rebuilds wall/beam/trajectory/grid actors

        for name, actor in (("clash_actor_left", preserved_left),
                             ("clash_actor_right", preserved_right)):
            if actor is not None:
                try:
                    setattr(self, name, actor)
                    self.plotter.add_actor(actor, render=False)
                except Exception:
                    pass

        if hasattr(self, "_distance_enabled") and self._distance_enabled:
            self.refresh_picker_actors()
        if hasattr(self, "picking_enabled") and self.picking_enabled:
            self.refresh_picker_actors()

        self.plotter.render()

    # =====================================================================
    # MEASURE
    # =====================================================================
    def toggle_distance_measurement(self) -> None:
        if self._warn_if_playing("measuring distances"):
            return

        if self._distance_enabled:
            self._disable_distance_measurement()
        else:
            self._disable_other_modes("distance")
            self._enable_distance_measurement()

    def _enable_distance_measurement(self) -> None:
        self._distance_enabled = True
        self.point_picker = vtk.vtkPointPicker()
        self.point_picker.SetTolerance(0.005)
        self.refresh_picker_actors()

        self.plotter.enable_point_picking(
            callback=self._on_point_selected,
            show_message=False,
            picker=self.point_picker,
            use_picker=True,
        )
        self.plotter.add_key_event('r', self.reset_measurements)
        self.plotter.add_key_event('R', self.reset_measurements)
        self.plotter.add_key_event('e', self.erase_last_point)
        self.plotter.add_key_event('E', self.erase_last_point)

        n = len(self.clicked_points)
        if n == 1:
            self._set_status("📏 1st point (RED) ready — click 2nd point")
        else:
            self._set_status("📏 Click 1st point (Red), then 2nd (Green)")

    def _disable_distance_measurement(self) -> None:
        self._distance_enabled = False
        try:
            self.plotter.disable_picking()
        except Exception:
            pass
        self.plotter.render()
        self._set_status("Distance measurement: disabled")

    def refresh_picker_actors(self) -> None:
        """Rebuild the pick list with the CURRENT frame's actors (they get
        replaced every frame, so a stale picker silently picks nothing)."""
        if self.point_picker is None:
            return
        self.point_picker.InitializePickList()
        added = []
        for name in ("wall_actor", "beam_actor",
                     "clash_actor_left", "clash_actor_right", "clash_actor_top"):
            actor = getattr(self, name, None)
            if actor is None:
                continue
            try:
                mapper = actor.GetMapper()
                if mapper and mapper.GetInput() and mapper.GetInput().GetNumberOfPoints() > 0:
                    self.point_picker.AddPickList(actor)
                    added.append(name)
            except Exception:
                pass
        if added:
            self.point_picker.PickFromListOn()
            self.point_picker.SetTolerance(0.001)

    def _on_point_selected(self, picked_point, picker) -> None:
        self.refresh_picker_actors()
        if picked_point is None:
            return
        picked_actor = picker.GetActor()
        point_id = picker.GetPointId()
        if point_id == -1:
            return

        try:
            mapper = picked_actor.GetMapper()
            polydata = mapper.GetInput()
            local_point = np.array(polydata.GetPoint(point_id), dtype=np.float64)
            if np.any(np.isnan(local_point)) or np.any(np.isinf(local_point)):
                return
            vtk_matrix = picked_actor.GetMatrix()
            mat = np.array(
                [[vtk_matrix.GetElement(i, j) for j in range(4)] for i in range(4)],
                dtype=np.float64)
            world_point = (mat @ np.append(local_point, 1.0))[:3]
        except Exception as exc:
            print(f"[ClashTools] error getting vertex: {exc}")
            return

        marker_color = "red" if len(self.clicked_points) == 0 else "green"
        point_num = len(self.clicked_points) + 1
        self.clicked_points.append(world_point)

        marker = pv.Sphere(radius=0.02, center=world_point,
                            phi_resolution=20, theta_resolution=20)
        marker_actor = self.plotter.add_mesh(
            marker, color=marker_color, opacity=0.95, render=False,
            name=f'measure_marker_{point_num}')
        self.current_measurement_actors.append(marker_actor)

        box = pv.Cube(center=world_point, x_length=0.09, y_length=0.09, z_length=0.09)
        box_actor = self.plotter.add_mesh(
            box, color=marker_color, opacity=0.4, style='wireframe',
            line_width=3, render=False, name=f'measure_box_{point_num}')
        self.current_measurement_actors.append(box_actor)

        if len(self.clicked_points) == 2:
            p1, p2 = self.clicked_points
            dist_3d = float(np.linalg.norm(p2 - p1))
            mid = (p1 + p2) / 2.0
            line_color = self.line_colors[self.color_index % len(self.line_colors)]

            line_actor = self.plotter.add_mesh(
                pv.Line(p1, p2), color=line_color, line_width=5,
                render=False, name='measure_line')
            self.current_measurement_actors.append(line_actor)

            label_actor = self.plotter.add_point_labels(
                [mid + np.array([0, 0, 0.4])],
                [f"Distance: {dist_3d:.4f} m"],
                point_size=0, font_size=16, text_color="black",
                shape_color=line_color, shape_opacity=0.95, bold=True,
                show_points=False, render=False, name='measure_label')
            self.current_measurement_actors.append(label_actor)

            self.all_measurements.append({
                'p1': p1.copy(), 'p2': p2.copy(), 'distance_3d': dist_3d,
                'color': line_color, 'actors': self.current_measurement_actors.copy(),
            })
            self._set_status(f"📐 Distance: {dist_3d:.4f} m")

            self.color_index += 1
            self.clicked_points = []
            self.current_measurement_actors = []
        else:
            self._set_status(f"📏 1st point (RED) marked — click 2nd point")

        self.plotter.render()

    def reset_measurements(self) -> None:
        self.clicked_points = []
        for actor in self.current_measurement_actors:
            try:
                self.plotter.remove_actor(actor, render=False)
            except Exception:
                pass
        self.current_measurement_actors = []
        for m in self.all_measurements:
            for actor in m['actors']:
                try:
                    self.plotter.remove_actor(actor, render=False)
                except Exception:
                    pass
        self.all_measurements = []
        self.plotter.render()
        self._set_status("Measurements cleared.")

    # =====================================================================
    # SINGLE-POINT PICKER (used internally; exposed for completeness)
    # =====================================================================
    def toggle_point_picker(self) -> None:
        if self._warn_if_playing("picking points"):
            return
        if self.picking_enabled:
            self._disable_point_picker()
        else:
            self._disable_other_modes("single")
            self._enable_point_picker()

    def _enable_point_picker(self) -> None:
        self.picking_enabled = True
        if self.single_point_picker is None:
            self.single_point_picker = vtk.vtkPointPicker()
            self.single_point_picker.SetTolerance(0.001)
        self.refresh_picker_actors()
        self.plotter.enable_point_picking(
            callback=self.on_single_point_picked,
            show_message=False, picker=self.single_point_picker, use_picker=True)
        self._set_status("🎯 Single point picker: click to mark")

    def _disable_point_picker(self) -> None:
        self.picking_enabled = False
        try:
            self.plotter.disable_picking()
        except Exception:
            pass
        for attr in ("picked_disk_actor", "picked_text_actor", "picked_box_actor"):
            actor = getattr(self, attr, None)
            if actor is not None:
                try:
                    self.plotter.remove_actor(actor, render=False)
                except Exception:
                    pass
                setattr(self, attr, None)
        self.plotter.render()
        self._set_status("Point picker: disabled")

    def on_single_point_picked(self, picked_point, picker) -> None:
        if picked_point is None:
            return
        point_id = picker.GetPointId()
        if point_id == -1:
            return
        picked_actor = picker.GetActor()
        try:
            mapper = picked_actor.GetMapper()
            polydata = mapper.GetInput()
            exact_point = np.array(polydata.GetPoint(point_id), dtype=np.float64)
        except Exception:
            return

        for attr in ("picked_disk_actor", "picked_text_actor", "picked_box_actor"):
            actor = getattr(self, attr, None)
            if actor is not None:
                try:
                    self.plotter.remove_actor(actor, render=False)
                except Exception:
                    pass

        marker = pv.Sphere(radius=0.06, center=exact_point)
        self.picked_disk_actor = self.plotter.add_mesh(
            marker, color='yellow', opacity=0.95, render=False, name='picked_marker')
        self.picked_text_actor = self.plotter.add_point_labels(
            [exact_point + np.array([0, 0, 0.3])],
            [f"X:{exact_point[0]:.3f} Y:{exact_point[1]:.3f} Z:{exact_point[2]:.3f}"],
            font_size=14, text_color='black', shape_color='yellow',
            shape_opacity=0.9, render=False, name='picked_label')
        self.plotter.render()

    # =====================================================================
    # ERASE — context-sensitive, dispatches to whichever mode is active
    # =====================================================================
    def erase_last_point(self) -> None:
        if self.picking_enabled:
            self._disable_point_picker()
            return

        if self.crop_mode:
            # Box-widget crop has nothing to "erase" point-by-point; cancel instead.
            self.cancel_crop()
            return

        if self._distance_enabled:
            if self.clicked_points:
                self.clicked_points.pop()
                # Remove that point's marker + box (last 2 actors added)
                for _ in range(2):
                    if self.current_measurement_actors:
                        actor = self.current_measurement_actors.pop()
                        try:
                            self.plotter.remove_actor(actor, render=False)
                        except Exception:
                            pass
                self.plotter.render()
                self._set_status("Erased last measurement point.")
                return

            if self.all_measurements:
                last = self.all_measurements.pop()
                for actor in last['actors']:
                    try:
                        self.plotter.remove_actor(actor, render=False)
                    except Exception:
                        pass
                self.plotter.render()
                self._set_status(
                    f"Erased completed measurement "
                    f"({len(self.all_measurements)} remaining).")
                return

            self._set_status("⚠️ No measurement points to erase.")
            return

        self._set_status("⚠️ No active tool mode — enable Clip, Measure, or pick first.")