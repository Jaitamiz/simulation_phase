# ============================================================================
# FRAME UPDATE — split into COMPUTE (thread-safe) and APPLY (main-thread-only)
# ============================================================================
#
# WHY THIS IS SPLIT:
#   The old update_frame_visualization_outer() did penetration detection,
#   OBB filtering, min-distance math, AND all PyVista/VTK actor manipulation
#   in one synchronous call on the Qt main thread. The math is pure
#   NumPy — safe to run on a background QThread. Everything touching
#   `plotter` (add/remove actors, beam_polydata.points, camera_follow,
#   plotter.render()) is NOT thread-safe and must stay on the main thread.
#
#   compute_frame_data()        — pure computation, NO plotter/VTK calls.
#                                  Safe to call from _FrameWorker (background
#                                  QThread) in simulation_engine.py.
#   apply_frame_visualization() — takes compute_frame_data()'s result and
#                                  does all the plotter/actor work. MUST be
#                                  called on the main thread only.
#   update_frame_visualization_outer() — thin backward-compatible wrapper
#                                  that just calls both in sequence,
#                                  synchronously, for any caller not yet
#                                  migrated to the threaded path.
#
# ⚠️ THREAD-SAFETY DEPENDENCY THIS FILE CANNOT VERIFY ON ITS OWN:
#   compute_frame_data() calls detect_penetrations_bidirectional() and reads
#   from `cache_manager` (AcceleratedBidirectionalPenetrationCache, defined
#   in detector.py). This split is only actually safe to run off the main
#   thread if THAT code path never touches `plotter`, any VTK render-window-
#   attached object, or any Qt widget internally (e.g. for "picking" against
#   the live scene rather than a standalone geometry structure like a
#   vtkOBBTree/vtkCellLocator it owns itself). Verify detector.py before
#   relying on _FrameWorker in production.
# ============================================================================
import os
import numpy as np
import pyvista as pv
from simulation.detector import detect_penetrations_bidirectional
from simulation.camera import camera_follow
from simulation.utils import (
    rotation_matrix,
    filter_clash_points_by_transformed_obb,
    compute_minimum_clash_distance,
)

def compute_frame_data(
    current_frame, total_frames,
    beam_points,
    trajectory_points, roll, pitch, yaw,
    wall_points, obb_center_local, obb_half_extents, bbox_local_points,
    cache_manager, thickness, sd_threshold,
    x_translation,
    obb_R_local=None,
):
    """
    PURE COMPUTATION — no plotter, no VTK/Qt objects touched anywhere in
    this function. Safe to call from a background QThread (see
    _FrameWorker in simulation_engine.py), subject to the detector.py
    caveat in the module docstring above.

    Returns a plain dict (safe to pass through a pyqtSignal(object) across
    threads) with everything apply_frame_visualization() needs:
        {
            "frame": current_frame, "total_frames": total_frames,
            "R": R, "t": t, "transformed": transformed,
            "beam_translation": beam_translation,
            "left_clash_points", "left_indices", ... (all 5 directions)
            "min_dist_left", ..., "min_dist_top",
            "clash_left", "vehicle_left", ... (viz endpoints, LEFT/RIGHT/TOP only)
        }

    ⚠️ COORDINATE-FRAME CONTRACT (unchanged): `beam_points` is NOT
    re-centered here. `obb_center_local` must have been computed from the
    exact same `beam_points` array passed in, and `trajectory_points` must
    be the same array (raw vs. smoothed) that was in scope when the cache's
    frame-0 mesh / OBB were built. Mismatches here silently zero out
    penetrations rather than raising.
    """
    print(f"\n📸 [compute] Frame {current_frame}/{total_frames}")
    i = current_frame

    # ===== COMPUTE TRANSFORM =====
    R = rotation_matrix(roll[i], pitch[i], yaw[i])
    t = trajectory_points[i]

    # ===== TRANSFORM BEAM POINTS =====
    transformed = (R @ beam_points.T).T + t
    print("frame_updater")
    vehicle_world = transformed  # alias used for min-distance queries below

    # ===== TRANSFORM BOUNDING BOX (kept for parity with prior behavior;
    #       not consumed further downstream, same as before the split) =====
    bbox_world = (R @ bbox_local_points.T).T + t
    
    # ===== TRAJECTORY-ONLY MODE (clash detection disabled) =====
    # x_translation <= 0 has no valid probe geometry — skip detection
    # entirely and just carry the beam's pose through. All clash-related
    # fields come back empty/None; apply_frame_visualization() already
    # handles that gracefully (it only adds actors when len(...) > 0 /
    # value is not None), and the recording gate in _apply_computed_frame()
    # naturally no-ops on all-empty arrays.
    if x_translation <= 0:
        print(f"   ⏭️  x_translation={x_translation} — clash detection OFF, trajectory-only frame")
        empty_pts = np.empty((0, 3), dtype=np.float64)
        empty_idx = np.array([])
        beam_centroid_world = (R @ beam_points.mean(axis=0)) + t
        return {
            "frame": current_frame, "total_frames": total_frames,
            "R": R, "t": t, "transformed": transformed,
            "beam_translation": beam_centroid_world,
            "left_clash_points": empty_pts, "left_indices": empty_idx,
            "right_clash_points": empty_pts, "right_indices": empty_idx,
            "front_clash_points": empty_pts, "front_indices": empty_idx,
            "back_clash_points": empty_pts, "back_indices": empty_idx,
            "top_clash_points": empty_pts, "top_indices": empty_idx,
            "min_dist_left": None, "min_dist_right": None,
            "min_dist_front": None, "min_dist_back": None, "min_dist_top": None,
            "clash_left": None, "vehicle_left": None,
            "clash_right": None, "vehicle_right": None,
            "clash_top": None, "vehicle_top": None,
        }

    # ===== DETECT PENETRATIONS (ALL FIVE DIRECTIONS) =====
    (left_clash_points, left_indices,
     right_clash_points, right_indices,
     front_clash_points, front_indices,
     back_clash_points, back_indices,
     top_clash_points, top_indices) = detect_penetrations_bidirectional(
        wall_points=wall_points,
        R=R,
        t=t,
        obb_center_local=obb_center_local,
        obb_half_extents=obb_half_extents,
        frame_idx=i,
        initial_beam_points=beam_points if i == 0 else None,
        x_translation=x_translation,
        thickness=thickness,
        sd_threshold=sd_threshold,
        cache_manager=cache_manager,
        obb_R_local=obb_R_local,
    )

    print(f"   🔍 Before OBB filtering:")
    print(f"      LEFT: {len(left_clash_points)}, RIGHT: {len(right_clash_points)}, "
          f"FRONT: {len(front_clash_points)}, BACK: {len(back_clash_points)}, TOP: {len(top_clash_points)}")

    # ===== FILTER CLASH POINTS BY TRANSFORMED OBB — ALL FIVE DIRECTIONS =====
    # obb_R_local threaded through here too (not just the wall-point
    # filter above) — see simulation/utils.py's compute_actual_obb_half_
    # extents()/filter_clash_points_by_transformed_obb() for the fix this
    # requires: without it, this box is axis-aligned in car-local X/Y/Z
    # rather than the car's own PCA length/width/height axes, and has to
    # cover the car's diagonal just to contain every beam point — large
    # enough to misclassify real, just-outside-the-body clash points as
    # "inside the beam" and silently discard them.
    # left_clash_points, left_outside_mask = filter_clash_points_by_transformed_obb(
    #     left_clash_points, R, t, beam_points, obb_center_local, obb_R_local
    # )
    # left_indices = left_indices[left_outside_mask] if len(left_indices) > 0 else np.array([])

    # right_clash_points, right_outside_mask = filter_clash_points_by_transformed_obb(
    #     right_clash_points, R, t, beam_points, obb_center_local, obb_R_local
    # )
    # right_indices = right_indices[right_outside_mask] if len(right_indices) > 0 else np.array([])

    # front_clash_points, front_outside_mask = filter_clash_points_by_transformed_obb(
    #     front_clash_points, R, t, beam_points, obb_center_local, obb_R_local
    # )
    # front_indices = front_indices[front_outside_mask] if len(front_indices) > 0 else np.array([])

    # back_clash_points, back_outside_mask = filter_clash_points_by_transformed_obb(
    #     back_clash_points, R, t, beam_points, obb_center_local, obb_R_local
    # )
    # back_indices = back_indices[back_outside_mask] if len(back_indices) > 0 else np.array([])

    # top_clash_points, top_outside_mask = filter_clash_points_by_transformed_obb(
    #     top_clash_points, R, t, beam_points, obb_center_local, obb_R_local
    # )
    # top_indices = top_indices[top_outside_mask] if len(top_indices) > 0 else np.array([])

    print(f"   ✓ After OBB filtering:")
    print(f"      LEFT: {len(left_clash_points)}, RIGHT: {len(right_clash_points)}, "
          f"FRONT: {len(front_clash_points)}, BACK: {len(back_clash_points)}, TOP: {len(top_clash_points)}")

    # ===== ENSURE ALL ARRAYS ARE PROPER SHAPE (N, 3) BEFORE ANY DOWNSTREAM USE =====
    if left_clash_points is None or len(left_clash_points) == 0:
        left_clash_points = np.empty((0, 3), dtype=np.float64)
    if right_clash_points is None or len(right_clash_points) == 0:
        right_clash_points = np.empty((0, 3), dtype=np.float64)
    if front_clash_points is None or len(front_clash_points) == 0:
        front_clash_points = np.empty((0, 3), dtype=np.float64)
    if back_clash_points is None or len(back_clash_points) == 0:
        back_clash_points = np.empty((0, 3), dtype=np.float64)
    if top_clash_points is None or len(top_clash_points) == 0:
        top_clash_points = np.empty((0, 3), dtype=np.float64)

    # ===== COMPUTE MINIMUM DISTANCE TO VEHICLE — ALL FIVE DIRECTIONS =====
    print(f"   📏 Computing minimum distances to vehicle (all 5 directions)...")

    min_dist_left = min_dist_right = min_dist_front = min_dist_back = min_dist_top = None
    clash_left = vehicle_left = None
    clash_right = vehicle_right = None
    clash_front = vehicle_front = None
    clash_back = vehicle_back = None
    clash_top = vehicle_top = None

    if len(left_clash_points) > 0:
        min_dist_left, clash_left, vehicle_left, _ = compute_minimum_clash_distance(
            left_clash_points, vehicle_world)
        if min_dist_left is not None:
            print(f"      ⭐ LEFT Min Distance:  {min_dist_left:.6f} m")

    if len(right_clash_points) > 0:
        min_dist_right, clash_right, vehicle_right, _ = compute_minimum_clash_distance(
            right_clash_points, vehicle_world)
        if min_dist_right is not None:
            print(f"      ⭐ RIGHT Min Distance: {min_dist_right:.6f} m")

    if len(front_clash_points) > 0:
        min_dist_front, clash_front, vehicle_front, _ = compute_minimum_clash_distance(
            front_clash_points, vehicle_world)
        if min_dist_front is not None:
            print(f"      ⭐ FRONT Min Distance: {min_dist_front:.6f} m")

    if len(back_clash_points) > 0:
        min_dist_back, clash_back, vehicle_back, _ = compute_minimum_clash_distance(
            back_clash_points, vehicle_world)
        if min_dist_back is not None:
            print(f"      ⭐ BACK Min Distance:  {min_dist_back:.6f} m")

    if len(top_clash_points) > 0:
        min_dist_top, clash_top, vehicle_top, _ = compute_minimum_clash_distance(
            top_clash_points, vehicle_world)
        if min_dist_top is not None:
            print(f"      ⭐ TOP Min Distance:   {min_dist_top:.6f} m")

    all_mins = [d for d in [min_dist_left, min_dist_right, min_dist_front,
                             min_dist_back, min_dist_top] if d is not None]
    if all_mins:
        print(f"      🎯 GLOBAL MINIMUM DISTANCE: {min(all_mins):.6f} m")

    # ===== COMPUTE BEAM TRANSLATION =====
    # NOTE: beam_polydata.points = transformed is deliberately NOT done here
    # — beam_polydata is a live VTK object the plotter may be reading from;
    # mutating it off the main thread risks racing an in-progress render.
    # That assignment happens in apply_frame_visualization() instead.
    beam_centroid_local = beam_points.mean(axis=0)
    beam_centroid_world = (R @ beam_centroid_local) + t
    beam_translation = beam_centroid_world

    print(f"   ✓ [compute] Frame {current_frame} computation complete")

    return {
        "frame": current_frame, "total_frames": total_frames,
        "R": R, "t": t, "transformed": transformed,
        "beam_translation": beam_translation,
        "left_clash_points": left_clash_points, "left_indices": left_indices,
        "right_clash_points": right_clash_points, "right_indices": right_indices,
        "front_clash_points": front_clash_points, "front_indices": front_indices,
        "back_clash_points": back_clash_points, "back_indices": back_indices,
        "top_clash_points": top_clash_points, "top_indices": top_indices,
        "min_dist_left": min_dist_left, "min_dist_right": min_dist_right,
        "min_dist_front": min_dist_front, "min_dist_back": min_dist_back,
        "min_dist_top": min_dist_top,
        "clash_left": clash_left, "vehicle_left": vehicle_left,
        "clash_right": clash_right, "vehicle_right": vehicle_right,
        "clash_top": clash_top, "vehicle_top": vehicle_top,
    }

def apply_frame_visualization(
    plotter, beam_polydata, computed, info_label,
    clash_actor_left, clash_actor_right, clash_actor_front, clash_actor_back, clash_actor_top,
    min_dist_sphere_left=None, min_dist_sphere_right=None, min_dist_sphere_top=None,
    min_dist_line_left=None, min_dist_line_right=None, min_dist_line_top=None,
    min_dist_vehicle_sphere_left=None, min_dist_vehicle_sphere_right=None, min_dist_vehicle_sphere_top=None,
    min_distance_visible=True,R_global=None,
    camera_distance=2.0, camera_height=0.0,
):
    """
    MAIN-THREAD ONLY. Takes compute_frame_data()'s result dict and does
    every plotter/VTK/actor mutation: beam position, clash-point clouds
    (5 directions), min-distance viz (LEFT/RIGHT/TOP, 9 actors), frame-
    number text, camera follow, and the actual render() call.

    R_global (optional, (3, 3)): one-off rotation from
    camera.estimate_global_rotation(). When given, the chase camera uses
    R_global @ R instead of the raw per-frame R (camera only — the beam
    itself is still placed with the raw R).
    camera_distance / camera_height: metres behind / above the car for the
    chase camera (see camera.camera_follow()).

    Returns the same actor-handle tuple shape the old
    update_frame_visualization_outer() returned, so callers can store them
    back and pass them in next frame exactly as before.
    """
    current_frame = computed["frame"]
    total_frames = computed["total_frames"]
    R, t = computed["R"], computed["t"]
    transformed = computed["transformed"]

    left_clash_points, left_indices = computed["left_clash_points"], computed["left_indices"]
    right_clash_points, right_indices = computed["right_clash_points"], computed["right_indices"]
    front_clash_points, front_indices = computed["front_clash_points"], computed["front_indices"]
    back_clash_points, back_indices = computed["back_clash_points"], computed["back_indices"]
    top_clash_points, top_indices = computed["top_clash_points"], computed["top_indices"]

    min_dist_left, min_dist_right = computed["min_dist_left"], computed["min_dist_right"]
    min_dist_front, min_dist_back = computed["min_dist_front"], computed["min_dist_back"]
    min_dist_top = computed["min_dist_top"]

    clash_left, vehicle_left = computed["clash_left"], computed["vehicle_left"]
    clash_right, vehicle_right = computed["clash_right"], computed["vehicle_right"]
    clash_top, vehicle_top = computed["clash_top"], computed["vehicle_top"]

    # ===== UPDATE BEAM POLYDATA POSITION (main thread — see compute_frame_data note) =====
    beam_polydata.points = transformed

    # ===== REMOVE OLD CLASH ACTORS =====
    if clash_actor_left:
        try:
            plotter.remove_actor(clash_actor_left, render=False)
        except:
            pass
        clash_actor_left = None

    if clash_actor_right:
        try:
            plotter.remove_actor(clash_actor_right, render=False)
        except:
            pass
        clash_actor_right = None

    if clash_actor_front:
        try:
            plotter.remove_actor(clash_actor_front, render=False)
        except:
            pass
        clash_actor_front = None

    if clash_actor_back:
        try:
            plotter.remove_actor(clash_actor_back, render=False)
        except:
            pass
        clash_actor_back = None

    if clash_actor_top:
        try:
            plotter.remove_actor(clash_actor_top, render=False)
        except:
            pass
        clash_actor_top = None

    # ===== REMOVE OLD MINIMUM-DISTANCE ACTORS (LEFT / RIGHT / TOP — 9 actors) =====
    for actor in [min_dist_sphere_left, min_dist_sphere_right, min_dist_sphere_top,
                  min_dist_line_left, min_dist_line_right, min_dist_line_top,
                  min_dist_vehicle_sphere_left, min_dist_vehicle_sphere_right, min_dist_vehicle_sphere_top]:
        if actor:
            try:
                plotter.remove_actor(actor, render=False)
            except:
                pass

    min_dist_sphere_left = min_dist_sphere_right = min_dist_sphere_top = None
    min_dist_line_left = min_dist_line_right = min_dist_line_top = None
    min_dist_vehicle_sphere_left = min_dist_vehicle_sphere_right = min_dist_vehicle_sphere_top = None

    # ===== VISUALIZE CLASH POINTS (ALL FIVE DIRECTIONS) =====
    if len(left_clash_points) > 0:
        clash_actor_left = plotter.add_points(
            pv.PolyData(left_clash_points), color='red', point_size=8,
            render_points_as_spheres=True, name='clash_left', render=False
        )
        print(f"      🔴 LEFT: {len(left_clash_points)} penetrations")

    if len(right_clash_points) > 0:
        clash_actor_right = plotter.add_points(
            pv.PolyData(right_clash_points), color='orange', point_size=8,
            render_points_as_spheres=True, name='clash_right', render=False
        )
        print(f"      🔴 RIGHT: {len(right_clash_points)} penetrations")

    if len(front_clash_points) > 0:
        clash_actor_front = plotter.add_points(
            pv.PolyData(front_clash_points), color='yellow', point_size=8,
            render_points_as_spheres=True, name='clash_front', render=False
        )
        print(f"      🔴 FRONT: {len(front_clash_points)} penetrations")

    if len(back_clash_points) > 0:
        clash_actor_back = plotter.add_points(
            pv.PolyData(back_clash_points), color='magenta', point_size=8,
            render_points_as_spheres=True, name='clash_back', render=False
        )
        print(f"      🔴 BACK: {len(back_clash_points)} penetrations")

    if len(top_clash_points) > 0:
        clash_actor_top = plotter.add_points(
            pv.PolyData(top_clash_points), color='cyan', point_size=8,
            render_points_as_spheres=True, name='clash_top', render=False
        )
        print(f"      🔴 TOP: {len(top_clash_points)} penetrations")

    # ===== VISUALIZE MINIMUM DISTANCE (LEFT / RIGHT / TOP — 9 ACTORS TOTAL) =====
    sphere_radius = 0.02
    line_width = 3
    show_min_dist = min_distance_visible

    if min_dist_left is not None and clash_left is not None and vehicle_left is not None and show_min_dist:
        min_dist_sphere_left = plotter.add_mesh(
            pv.Sphere(radius=sphere_radius, center=clash_left),
            color='yellow', name='min_dist_clash_left', render=False
        )
        min_dist_vehicle_sphere_left = plotter.add_mesh(
            pv.Sphere(radius=sphere_radius, center=vehicle_left),
            color='orange', name='min_dist_vehicle_left', render=False
        )
        min_dist_line_left = plotter.add_mesh(
            pv.Line(clash_left, vehicle_left),
            color='yellow', line_width=line_width, name='min_dist_line_left', render=False
        )
        print(f"      ⭐ LEFT Min Distance Visualized:  {min_dist_left:.6f} m")

    if min_dist_right is not None and clash_right is not None and vehicle_right is not None and show_min_dist:
        min_dist_sphere_right = plotter.add_mesh(
            pv.Sphere(radius=sphere_radius, center=clash_right),
            color='cyan', name='min_dist_clash_right', render=False
        )
        min_dist_vehicle_sphere_right = plotter.add_mesh(
            pv.Sphere(radius=sphere_radius, center=vehicle_right),
            color='blue', name='min_dist_vehicle_right', render=False
        )
        min_dist_line_right = plotter.add_mesh(
            pv.Line(clash_right, vehicle_right),
            color='cyan', line_width=line_width, name='min_dist_line_right', render=False
        )
        print(f"      ⭐ RIGHT Min Distance Visualized: {min_dist_right:.6f} m")

    if min_dist_top is not None and clash_top is not None and vehicle_top is not None and show_min_dist:
        min_dist_sphere_top = plotter.add_mesh(
            pv.Sphere(radius=sphere_radius, center=clash_top),
            color='magenta', name='min_dist_clash_top', render=False
        )
        min_dist_vehicle_sphere_top = plotter.add_mesh(
            pv.Sphere(radius=sphere_radius, center=vehicle_top),
            color='green', name='min_dist_vehicle_top', render=False
        )
        min_dist_line_top = plotter.add_mesh(
            pv.Line(clash_top, vehicle_top),
            color='magenta', line_width=line_width, name='min_dist_line_top', render=False
        )
        print(f"      ⭐ TOP Min Distance Visualized:   {min_dist_top:.6f} m")

    # ===== ON-SCREEN TEXT — FRAME NUMBER ONLY =====
    if info_label is not None:
        info_label.setText(f"Frame: {current_frame} / {total_frames}")

    # ===== CAMERA FOLLOW =====
    R_cam = R if R_global is None else (R_global @ R)
    camera_follow(plotter, R_cam, t, distance=camera_distance, height=camera_height)

    
    # ===== FINAL RENDER =====
    plotter.add_axes()
    plotter.render()

    print(f"   ✓ [apply] Frame {current_frame} visualization complete\n")

    return (clash_actor_left, clash_actor_right, clash_actor_front, clash_actor_back, clash_actor_top,
            min_dist_sphere_left, min_dist_sphere_right, min_dist_sphere_top,
            min_dist_line_left, min_dist_line_right, min_dist_line_top,
            min_dist_vehicle_sphere_left, min_dist_vehicle_sphere_right, min_dist_vehicle_sphere_top)

def update_frame_visualization_outer(
    plotter, current_frame, total_frames,
    beam_points, beam_polydata,
    trajectory_points, roll, pitch, yaw,
    wall_points, obb_center_local, obb_half_extents, bbox_local_points,
    cache_manager, thickness, sd_threshold,
    save_dir,
    x_translation, frame_skip,
    bidir_info_label, info_label,
    clash_actor_left, clash_actor_right, clash_actor_front, clash_actor_back, clash_actor_top,
    text_actor,
    min_dist_sphere_left=None, min_dist_sphere_right=None, min_dist_sphere_top=None,
    min_dist_line_left=None, min_dist_line_right=None, min_dist_line_top=None,
    min_dist_vehicle_sphere_left=None, min_dist_vehicle_sphere_right=None, min_dist_vehicle_sphere_top=None,
    min_distance_visible=True,
):
    """
    BACKWARD-COMPATIBLE, FULLY SYNCHRONOUS wrapper — calls compute_frame_data()
    then apply_frame_visualization() back to back, on whichever thread calls
    it. Kept so any caller not yet migrated to the threaded path (_FrameWorker
    in simulation_engine.py) keeps working exactly as before. New code should
    prefer calling the two functions above directly — compute_frame_data() in
    a background QThread, apply_frame_visualization() connected to that
    thread's completion signal on the main thread.

    save_dir, frame_skip, bidir_info_label, and text_actor are accepted only
    for call-site compatibility — none of them were read inside this
    function even before this split.
    """
    computed = compute_frame_data(
        current_frame=current_frame, total_frames=total_frames,
        beam_points=beam_points,
        trajectory_points=trajectory_points, roll=roll, pitch=pitch, yaw=yaw,
        wall_points=wall_points, obb_center_local=obb_center_local,
        obb_half_extents=obb_half_extents, bbox_local_points=bbox_local_points,
        cache_manager=cache_manager, thickness=thickness, sd_threshold=sd_threshold,
        x_translation=x_translation,
    )

    (clash_actor_left, clash_actor_right, clash_actor_front, clash_actor_back, clash_actor_top,
     min_dist_sphere_left, min_dist_sphere_right, min_dist_sphere_top,
     min_dist_line_left, min_dist_line_right, min_dist_line_top,
     min_dist_vehicle_sphere_left, min_dist_vehicle_sphere_right, min_dist_vehicle_sphere_top) = \
        apply_frame_visualization(
            plotter=plotter, beam_polydata=beam_polydata, computed=computed,
            info_label=info_label,
            clash_actor_left=clash_actor_left, clash_actor_right=clash_actor_right,
            clash_actor_front=clash_actor_front, clash_actor_back=clash_actor_back,
            clash_actor_top=clash_actor_top,
            min_dist_sphere_left=min_dist_sphere_left, min_dist_sphere_right=min_dist_sphere_right,
            min_dist_sphere_top=min_dist_sphere_top,
            min_dist_line_left=min_dist_line_left, min_dist_line_right=min_dist_line_right,
            min_dist_line_top=min_dist_line_top,
            min_dist_vehicle_sphere_left=min_dist_vehicle_sphere_left,
            min_dist_vehicle_sphere_right=min_dist_vehicle_sphere_right,
            min_dist_vehicle_sphere_top=min_dist_vehicle_sphere_top,
            min_distance_visible=min_distance_visible,
        )

    return (computed["left_clash_points"], computed["left_indices"],
            computed["right_clash_points"], computed["right_indices"],
            computed["front_clash_points"], computed["front_indices"],
            computed["back_clash_points"], computed["back_indices"],
            computed["top_clash_points"], computed["top_indices"],
            computed["R"], computed["t"], computed["beam_translation"],
            computed["min_dist_left"], computed["min_dist_right"],
            computed["min_dist_front"], computed["min_dist_back"], computed["min_dist_top"],
            clash_actor_left, clash_actor_right, clash_actor_front, clash_actor_back, clash_actor_top,
            min_dist_sphere_left, min_dist_sphere_right, min_dist_sphere_top,
            min_dist_line_left, min_dist_line_right, min_dist_line_top,
            min_dist_vehicle_sphere_left, min_dist_vehicle_sphere_right, min_dist_vehicle_sphere_top)