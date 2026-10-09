import os
import json
import glob

import numpy as np
import pyvista as pv
import open3d as o3d

from simulation.project_dir import get_cache_dir
from simulation.io_worker import io_worker

# Canonical side key -> the default folder/file name preview_tab.py's
# _export_named_sides() writes it under when the Save-Sides rename
# dialog is left untouched (e.g. preview/Left/Left.npy). Used by
# AcceleratedBidirectionalPenetrationCache._load_selective_indices() as
# the fast-path lookup before falling back to scanning every
# preview/*/selection.json for a side that WAS manually renamed.
_CANONICAL_SIDE_DISPLAY = {
    "left": "Left", "right": "Right", "front": "Front",
    "rear": "Rear", "top": "Top", "bottom": "Bottom",
}


def _pca_tight_obb(points_local, pca_center, R_local, margin=0.0):
    """
    Fit a TIGHT oriented bounding box to `points_local` using the given
    orthonormal basis R_local (columns = the box's own axes — e.g.
    [length_vector, width_vector, height_vector] from a committed PCA
    orientation), rather than an axis-aligned box in the car's raw local
    X/Y/Z. An axis-aligned box is only tight if the car's true body axes
    happen to coincide with local X/Y/Z; in general it has to be sized to
    the car's DIAGONAL to still contain every point, which is exactly the
    loose, oversized box a mis-aligned axis-aligned fit produces.

    Returns (center_local, half_extents) in the SAME car-local X/Y/Z frame
    obb_center_local has always lived in; half_extents are along the box's
    OWN axes (R_local's columns) — this pairs directly with
    filter_wall_points_by_obb(obb_R_local=R_local, ...) and with
    simulation_engine.py's _compute_pca_oriented_obb(), which computes the
    identical thing for the actual runtime detection window.
    """
    pts = np.asarray(points_local, dtype=np.float64)
    proj = (pts - pca_center) @ R_local
    mins = proj.min(axis=0)
    maxs = proj.max(axis=0)
    center_in_box_frame = (mins + maxs) / 2.0
    half_extents = (maxs - mins) / 2.0 + max(margin, 0.0)
    center_local = pca_center + R_local @ center_in_box_frame
    return center_local, half_extents


def _add_obb_wireframe(plotter, center, half_extents, obb_R_local=None, color="black",
                        line_width=3, label="Detection Window (OBB)"):
    """
    Draw the detection window as a wireframe box, ORIENTED by obb_R_local
    (columns = the box's own local axes — e.g. a committed PCA length/
    width/height basis) when supplied, or axis-aligned in LOCAL/body-fixed
    space when obb_R_local is None. This is what lets the preview popups
    (interactive_alignment_gui() and visualize_points_with_direction_
    frames()) show the SAME window filter_wall_points_by_obb() actually
    filters wall candidates against — a TIGHT, correctly-oriented box
    around the car body, not a loose axis-aligned box sized to its
    diagonal. No-ops quietly if center/half_extents weren't supplied.

    Returns the actor added (or None if skipped/failed), so callers that
    redraw this live (e.g. on every orientation-swap click) can remove and
    replace it, the same way arrow actors are tracked and cleared.
    """
    if center is None or half_extents is None:
        return None
    try:
        center = np.asarray(center, dtype=float)
        half_extents = np.asarray(half_extents, dtype=float)
        R_local = np.eye(3) if obb_R_local is None else np.asarray(obb_R_local, dtype=float)

        # 8 corners: center + (±hx, ±hy, ±hz) expressed in the box's OWN
        # frame, then rotated into car-local X/Y/Z via R_local — the same
        # "offset in box frame -> car's local frame" composition used
        # everywhere else an OBB corner is built in this codebase.
        signs = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
        ], dtype=float)
        corners = center[None, :] + (signs * half_extents[None, :]) @ R_local.T

        edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        box_edges = pv.Line(corners[edges[0][0]], corners[edges[0][1]])
        for a, b in edges[1:]:
            box_edges += pv.Line(corners[a], corners[b])

        return plotter.add_mesh(box_edges, color=color, line_width=line_width,
                                 label=label)
    except Exception as exc:
        print(f"[detector] ⚠️ OBB wireframe draw skipped: {exc}")
        return None


def create_axis_frame(origin, x_axis, y_axis, z_axis, labels=("X", "Y", "Z")):
        """
        Create a visual frame with three axes and labels.
       
        Args:
            origin: (3,) origin point
            x_axis: (3,) vector for X-axis direction and length
            y_axis: (3,) vector for Y-axis direction and length
            z_axis: (3,) vector for Z-axis direction and length
            labels: tuple of 3 strings for axis labels
       
        Returns:
            pv.PolyData: combined mesh representing the frame
        """
        origin = np.asarray(origin)
       
        # Create cylinders for axes
        x_end = origin + x_axis
        y_end = origin + y_axis
        z_end = origin + z_axis
       
        # X-axis (RED by default, will be colored by plotter)
        x_cyl = pv.Cylinder(
            center=(origin + x_end) / 2,
            direction=x_axis,
            radius=np.linalg.norm(x_axis) * 0.02,
            height=np.linalg.norm(x_axis)
        )
       
        # Y-axis (GREEN)
        y_cyl = pv.Cylinder(
            center=(origin + y_end) / 2,
            direction=y_axis,
            radius=np.linalg.norm(y_axis) * 0.02,
            height=np.linalg.norm(y_axis)
        )
       
        # Z-axis (BLUE)
        z_cyl = pv.Cylinder(
            center=(origin + z_end) / 2,
            direction=z_axis,
            radius=np.linalg.norm(z_axis) * 0.02,
            height=np.linalg.norm(z_axis)
        )
       
        # Create cones at the ends
        cone_scale = np.linalg.norm(x_axis) * 0.04
       
        x_cone = pv.Cone(
            direction=x_axis,
            height=cone_scale * 2,
            radius=cone_scale,
            resolution=8
        )
        x_cone.translate(x_end)
       
        y_cone = pv.Cone(
            direction=y_axis,
            height=cone_scale * 2,
            radius=cone_scale,
            resolution=8
        )
        y_cone.translate(y_end)
       
        z_cone = pv.Cone(
            direction=z_axis,
            height=cone_scale * 2,
            radius=cone_scale,
            resolution=8
        )
        z_cone.translate(z_end)
       
        # Combine all components
        frame = x_cyl + y_cyl + z_cyl + x_cone + y_cone + z_cone
       
        return frame
   
def filter_wall_points_by_obb(
    wall_points,
    R, t,
    obb_center_initial,
    obb_half_extents,
    obb_R_local=None,
):
    """
    Filters wall points inside the car's oriented bounding box.

    Args:
        wall_points: (N,3) world-space wall points
        R: (3x3) car rotation (trajectory frame's rigid transform)
        t: (3,) car translation
        obb_center_initial: OBB center, in the car's own local X/Y/Z
            (the SAME local frame beam_points_centered lives in — NOT the
            OBB's own PCA-aligned frame; obb_R_local below handles that
            distinction)
        obb_half_extents: half-sizes of the box, along the box's OWN axes
            (length/width/height if obb_R_local is a PCA basis, or plain
            X/Y/Z if obb_R_local is None/identity)
        obb_R_local: (3x3) orientation of the box's own axes WITHIN the
            car's local X/Y/Z frame — e.g. columns = [length_vector,
            width_vector, height_vector] from a committed PCA orientation.
            None (default) = identity, i.e. axis-aligned in local X/Y/Z,
            preserving the previous behavior for any caller that doesn't
            pass this. This is what lets the box be a TIGHT, PCA-oriented
            fit even when the car's true length/width/height axes aren't
            aligned with its own local X/Y/Z — see
            simulation_engine.py's _compute_pca_oriented_obb().
    """
    if obb_R_local is None:
        obb_R_local = np.eye(3)

    # Transform OBB center to world coordinates
    obb_center_world = R @ obb_center_initial + t

    # Combined rotation: trajectory frame's rigid rotation, THEN the box's
    # own fixed orientation within local space — this is what "applying R
    # and t to the box dimensions" actually means for an ORIENTED box: the
    # box's axes need the SAME two-step composition beam mesh vertices get
    # (local shape -> car's local frame -> world), not just R alone.
    full_R = R @ obb_R_local

    # Transform wall points into the BOX'S OWN frame (not just the car's
    # local X/Y/Z) — this is the axis-aligned test, but now correctly
    # relative to the box's true orientation.
    wall_in_box_frame = (full_R.T @ (wall_points - obb_center_world).T).T

    inside_mask = (
        (np.abs(wall_in_box_frame[:, 0]) <= obb_half_extents[0]) &
        (np.abs(wall_in_box_frame[:, 1]) <= obb_half_extents[1]) &
        (np.abs(wall_in_box_frame[:, 2]) <= obb_half_extents[2])
    )

    return wall_points[inside_mask], inside_mask

# SMART PENETRATION CACHE - OPTIMIZED WITH VERTEX TRANSFORM
# ============================================================================
class AcceleratedBidirectionalPenetrationCache:
    def __init__(self, cache_dir: str = None, max_workers: int = None):
        if cache_dir is None:
            # Resolved lazily so the object can be constructed before
            # set_project_dir() is called; the dir is created on first use.
            cache_dir = get_cache_dir("penetration_cache_bidirectional")
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
       
        # ===== BASE MESHES IN LOCAL COORDINATES (CREATED ONCE) =====
        # These are created from initial_beam_points centered at origin
        # They contain the base geometry that will be transformed each frame
        self.base_mesh_left_local = None
        self.base_mesh_right_local = None
        self.base_mesh_front_local = None
        self.base_mesh_back_local = None
        self.base_mesh_top_local = None
        
        # Store LOCAL vertices explicitly for efficient transformation
        # v_world = R @ v_local + t
        self.base_vertices_left_local = None
        self.base_vertices_right_local = None
        self.base_vertices_front_local = None
        self.base_vertices_back_local = None
        self.base_vertices_top_local = None
        
        # Triangle connectivity (never changes)
        self.base_triangles_left = None
        self.base_triangles_right = None
        self.base_triangles_front = None
        self.base_triangles_back = None
        self.base_triangles_top = None
        
        # PCA and transform info
        self.base_pose = None
        self.initial_beam_points = None
        self.pca_info = None 
        # True once interactive_alignment_gui() commits a manual TOP/BOTTOM,
        # LEFT/RIGHT, LENGTH/WIDTH, or LENGTH/HEIGHT choice — checked by
        # create_base_meshes_bidirectional_local() so it does NOT let
        # _detect_top_direction()'s heuristic silently overwrite a manual
        # choice (see interactive_alignment_gui()'s docstring / the fix
        # below for why this flag exists).
        self._orientation_manually_set = False
        self.selective_indices = None
        self.use_selective_points = True
        self.selective_indices_left = None
        self.selective_indices_right = None
        self.selective_indices_top = None
        self.selective_indices_front = None
        self.selective_indices_back = None

        # Project/ID context needed to read PreviewTab's saved per-side
        # selections correctly — see set_project_context() and the
        # rewritten _load_selective_indices() below. Both must be set
        # (via set_project_context()) before create_base_meshes_
        # bidirectional_local() runs, or selective points are skipped
        # entirely (falls back to using all beam points, same as when
        # no selection files exist yet).
        self.project_dir = None
        self.beam_ids = None
       
        
        # Flag to track if base meshes have been created
        self.meshes_initialized = False

        # ===== STATIC LOCAL RAYCASTING SCENES (BUILT ONCE) =====
        # Keyed by direction ("LEFT"/"RIGHT"/"FRONT"/"BACK"/"TOP"). Built
        # once in _build_static_scenes() right after the base LOCAL meshes
        # exist, and reused for every frame thereafter — see that method's
        # docstring for why this replaced the old per-frame BVH rebuild.
        self._static_scenes: dict = {}

        # NOTE: this class used to also own a private
        # ThreadPoolExecutor(max_workers=8) here. It was never actually
        # submitted to anywhere in this file — GPU/Open3D raycasting for
        # the 5 directions runs sequentially, on purpose, on the single
        # _FrameWorker thread that owns all GPU work (see
        # simulation_engine.py's _FrameWorker / target architecture).
        # Running concurrent Open3D tensor queries from multiple threads
        # would cause GPU serialization + memory contention, not a
        # speedup — so that pool was dead weight, not a real optimization.
        # The one thing in this class that WAS worth moving off this
        # thread — save_frame_cache()'s per-frame npz read-modify-write —
        # now goes through the shared io_worker (simulation/io_worker.py)
        # instead, which is a small, dedicated disk-I/O-only pool shared
        # across the whole app. `max_workers` is kept as a constructor
        # parameter only for call-site backward compatibility; it is no
        # longer used to size anything here.
        self.max_workers = max_workers or 0

        print("✅ Bidirectional Cache initialized (GPU raycasting on the "
              "single simulation thread; disk I/O offloaded to io_worker)")
    
    # =========================================================================
    #  PCA DIRECTION IDENTIFICATION (from local beam points)
    # =========================================================================
    
    def _compute_pca(self, points):
        """
        Compute PCA of the point cloud.
        
        Args:
            points: (N, 3) point cloud (should be centered at origin)
        
        Returns:
            center: mean of points
            eigvals: eigenvalues sorted desc
            eigvecs: eigenvectors sorted to match eigvals
        """
        pts = np.asarray(points, dtype=float)
        center = pts.mean(axis=0)
        pts_centered = pts - center

        C = np.cov(pts_centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(C)

        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        return center, eigvals, eigvecs

    def _identify_perpendicular_to_height(self, points):
        """
        Identify the direction perpendicular to HEIGHT using Z-component alignment.
        Works on LOCAL beam points (centered at origin).
        
        Returns:
            direction: normalized width vector (perpendicular to height)
            side_info: dict with axis indices and dimensions
            pca_center: center of PCA
        """
        pca_center, pca_eigvals, pca_eigvecs = self._compute_pca(points)
        
        pts_centered = points - pca_center
        proj = np.dot(pts_centered, pca_eigvecs)
        
        min_proj = proj.min(axis=0)
        max_proj = proj.max(axis=0)
        extents = (max_proj - min_proj) / 2.0
        
        # Identify HEIGHT using Z-component alignment
        z_components = np.abs(pca_eigvecs[2, :])
        z_axis_idx = np.argmax(z_components)
        
        perpendicular_indices = [i for i in range(3) if i != z_axis_idx]
        perp_extents = [2 * extents[i] for i in perpendicular_indices]
        
        if perp_extents[0] > perp_extents[1]:
            length_idx = perpendicular_indices[0]
            width_idx = perpendicular_indices[1]
            length = perp_extents[0]
            width = perp_extents[1]
        else:
            length_idx = perpendicular_indices[1]
            width_idx = perpendicular_indices[0]
            length = perp_extents[1]
            width = perp_extents[0]
        
        height_extent = 2 * extents[z_axis_idx]
        width_direction = pca_eigvecs[:, width_idx]
        
        side_info = {
            "height_axis_index": z_axis_idx,
            "length_axis_index": length_idx,
            "width_axis_index": width_idx,
            "height_vector": pca_eigvecs[:, z_axis_idx],
            "length_vector": pca_eigvecs[:, length_idx],
            "width_vector": pca_eigvecs[:, width_idx],
            "dimensions": {
                "length": length,
                "width": width,
                "height": height_extent
            },
            "z_components": {
                "axis_0": z_components[0],
                "axis_1": z_components[1],
                "axis_2": z_components[2],
            }
        }
        
        width_direction_normalized = width_direction / np.linalg.norm(width_direction)
        return width_direction_normalized, side_info, pca_center

    def visualize_points_with_direction_frames(self, beam_points_local, side_info,
                                               obb_center=None, obb_half_extents=None,
                                               obb_margin=0.0):
        """
        Show popup with point cloud and directional frames BEFORE creating meshes.
        
        This visualization includes:
        - Beam points as red point cloud
        - Directional frames showing:
        - RIGHT (negative WIDTH): Green frame
        - LEFT (positive WIDTH): Red frame
        - TOP (positive HEIGHT): Blue frame
        - Axes indicator for reference
        - The detection window (OBB) — a TIGHT box fit directly to
          beam_points_local using side_info's own length/width/height
          axes (via _pca_tight_obb()), NOT a loose axis-aligned box. This
          is the same tight, PCA-oriented box simulation_engine.py's
          _compute_pca_oriented_obb() computes for the real runtime
          detection window, drawn here (with the same obb_margin padding)
          so this popup shows exactly what filter_wall_points_by_obb()
          will actually filter wall candidates against.

        Args:
            beam_points_local: (N, 3) beam points in local coordinates
            side_info: dictionary containing PCA directions and dimensions
            obb_center, obb_half_extents: unused legacy params, kept only
                for call-site compatibility — the OBB drawn here is now
                ALWAYS the tight, self-computed one described above, never
                whatever loose axis-aligned box a caller might still pass.
            obb_margin: padding added to the tight fit (e.g. the probe
                reach / x_translation), matching _compute_pca_oriented_obb().
        """
        plotter = pv.Plotter(
            title="MESH PREVIEW: Points + Direction Frames (RIGHT/LEFT/TOP)",
            window_size=(1400, 900),
            shape=(1, 1)
        )
        
        # ===== 1. ADD BEAM POINTS =====
        point_cloud = pv.PolyData(beam_points_local)
        plotter.add_points(
            point_cloud,
            color='red',
            point_size=6,
            label=f'Beam Points ({len(beam_points_local)})',
            opacity=0.8
        )
        
        # ===== 2. COMPUTE FRAME CENTER =====
        # Use PCA center or point cloud center
        frame_center = beam_points_local.mean(axis=0)
        
        # ===== 3. EXTRACT DIRECTIONS FROM side_info =====
        height_vector = side_info['height_vector'] / np.linalg.norm(side_info['height_vector'])
        length_vector = side_info['length_vector'] / np.linalg.norm(side_info['length_vector'])
        width_vector = side_info['width_vector'] / np.linalg.norm(side_info['width_vector'])
        
        # Get dimensions for scaling frames
        dims = side_info['dimensions']
        length = dims['length']
        width = dims['width']
        height = dims['height']
        
        frame_scale = max(length, width, height) * 0.15  # 15% of largest dimension
        
        print("\n" + "="*70)
        print("📊 DIRECTION FRAME PREVIEW")
        print("="*70)
        print(f"Frame Center: {frame_center}")
        print(f"Frame Scale: {frame_scale:.4f}m")
        print(f"\nDimensions:")
        print(f"  Length: {length:.3f}m")
        print(f"  Width:  {width:.3f}m")
        print(f"  Height: {height:.3f}m")
        
        # ===== 4. CREATE AND ADD DIRECTION FRAMES =====
        
        # RIGHT frame (negative WIDTH direction) - GREEN
        print(f"\n✓ RIGHT (negative WIDTH):")
        print(f"   Vector: {-width_vector}")
        right_frame = create_axis_frame(
            origin=frame_center,
            x_axis=-width_vector * frame_scale,  # Negative WIDTH
            y_axis=length_vector * frame_scale,
            z_axis=height_vector * frame_scale,
            labels=("RIGHT", "FWD", "UP")
        )
        plotter.add_mesh(
            right_frame,
            color='green',
            line_width=3,
            label='RIGHT (-WIDTH)',
            opacity=0.9
        )
        
        # LEFT frame (positive WIDTH direction) - RED
        print(f"\n✓ LEFT (positive WIDTH):")
        print(f"   Vector: {width_vector}")
        left_frame = create_axis_frame(
            origin=frame_center,
            x_axis=width_vector * frame_scale,  # Positive WIDTH
            y_axis=length_vector * frame_scale,
            z_axis=height_vector * frame_scale,
            labels=("LEFT", "FWD", "UP")
        )
        plotter.add_mesh(
            left_frame,
            color='red',
            line_width=3,
            label='LEFT (+WIDTH)',
            opacity=0.9
        )
        
        # TOP frame (positive HEIGHT direction) - BLUE
        print(f"\n✓ TOP (positive HEIGHT):")
        print(f"   Vector: {height_vector}")
        top_frame = create_axis_frame(
            origin=frame_center,
            x_axis=width_vector * frame_scale,
            y_axis=length_vector * frame_scale,
            z_axis=height_vector * frame_scale,  # Positive HEIGHT
            labels=("WIDTH", "LENGTH", "TOP")
        )
        plotter.add_mesh(
            top_frame,
            color='blue',
            line_width=3,
            label='TOP (+HEIGHT)',
            opacity=0.9
        )
        
        # ===== 5. ADD REFERENCE AXES =====
        actor = plotter.add_mesh(pv.Sphere())
        plotter.add_orientation_widget(actor)

        origin_axes = create_axis_frame(
            origin=np.array([0, 0, 0]),
            x_axis=np.array([0.3, 0, 0]),
            y_axis=np.array([0, 0.3, 0]),
            z_axis=np.array([0, 0, 0.3]),
            labels=("X", "Y", "Z")
        )
        plotter.add_mesh(
            origin_axes,
            color='gray',
            line_width=2,
            label='Origin Axes (XYZ)',
            opacity=0.5
        )

        # ===== 5b. DETECTION WINDOW (OBB) — TIGHT, PCA-oriented ==========
        # Self-computed from beam_points_local using side_info's own
        # length/width/height axes (see _pca_tight_obb()) — replaces the
        # old loose axis-aligned box entirely; obb_center/obb_half_extents
        # params are no longer read here.
        pca_center_for_obb = beam_points_local.mean(axis=0)
        tight_center, tight_half_extents = _pca_tight_obb(
            beam_points_local, pca_center_for_obb,
            np.column_stack([length_vector, width_vector, height_vector]),
            margin=obb_margin,
        )
        _add_obb_wireframe(
            plotter, tight_center, tight_half_extents,
            obb_R_local=np.column_stack([length_vector, width_vector, height_vector]),
        )

        # ===== 6. CONFIGURE PLOTTER =====
        plotter.add_legend(
            bcolor='lightgray',
            face=None,
            loc='upper right'
        )
        
        plotter.set_scale(zscale=1.0)
        plotter.camera.position = (
            frame_center[0] + 2,
            frame_center[1] + 2,
            frame_center[2] + 2
        )
        plotter.camera.focal_point = frame_center
        
        # Print confirmation
        print("\n" + "="*70)
        print("🔍 VERIFY FRAME DIRECTIONS BEFORE PROCEEDING")
        print("="*70)
        print("\n✓ RIGHT (GREEN):  Negative WIDTH direction")
        print("✓ LEFT (RED):     Positive WIDTH direction")
        print("✓ TOP (BLUE):     Positive HEIGHT direction")
        print("\nClose the window to continue with mesh creation...")
        print("="*70 + "\n")
        
        plotter.show()

    def _detect_top_direction(self, beam_points_local, side_info):
        """
        Automatically detect which direction is UP (towards top of object).
        
        Uses centroid analysis: the direction where points extend AWAY FROM center
        is the UP direction. This assumes points concentrate at the base/middle.
        
        Args:
            beam_points_local: (N, 3) beam points centered at origin
            side_info: dictionary with PCA axis information
        
        Returns:
            top_direction: normalized vector pointing UPWARD
        """
        # Get the height vector candidate from PCA
        height_vector = side_info['height_vector'] / np.linalg.norm(side_info['height_vector'])
        height_axis_idx = side_info['height_axis_index']
        
        # Method 1: Z-component dominance (if Z is clearly UP in world frame)
        z_component_abs = np.abs(height_vector[2])
        if z_component_abs > 0.9:  # Height is mostly aligned with Z-axis
            if height_vector[2] > 0:
                return height_vector  # Points UP (positive Z)
            else:
                return -height_vector  # Flip to point UP
        
        # Method 2: Centroid spread analysis
        # Project points onto the height axis
        centroid = beam_points_local.mean(axis=0)
        projections = np.dot(beam_points_local - centroid, height_vector)
        
        # Count points in positive vs negative direction
        pos_count = np.sum(projections > 0)
        neg_count = np.sum(projections < 0)
        
        # The direction with FEWER points is likely the TOP (where object tapers)
        # The direction with MORE points is likely the BOTTOM (where object is wider)
        if pos_count < neg_count:
            return height_vector  # Positive direction is TOP
        else:
            return -height_vector  # Negative direction is TOP

    def interactive_alignment_gui(self, beam_points_local, obb_center=None, obb_half_extents=None,
                                   obb_margin=0.0):
        """
        Interactive PCA orientation-alignment window (like a CAD "Orientation
        Alignment" panel). Shows the beam point cloud with the current
        TOP/BOTTOM, LEFT/RIGHT and FRONT/REAR direction arrows derived from
        PCA, plus four toggle buttons to fix the common PCA mismatches:
            - Swap TOP <-> BOTTOM      (flips the height axis)
            - Swap LEFT <-> RIGHT      (flips the width axis)
            - Swap LENGTH <-> WIDTH    (swaps which PCA axis is length vs
                                         width, i.e. rotates FRONT/REAR into
                                         LEFT/RIGHT)
            - Swap LENGTH <-> HEIGHT   (swaps which PCA axis is length vs
                                         height, i.e. rotates FRONT/REAR into
                                         TOP/BOTTOM -- use this when PCA
                                         mistook the beam's long axis for
                                         its height, or vice versa)
        Every click redraws the arrows immediately so you can see the effect
        before committing. Clicking "Continue" commits the final orientation
        into `self.pca_info` AND sets `self._orientation_manually_set = True`.
        create_base_meshes_bidirectional_local() checks that flag to skip
        _detect_top_direction()'s auto-heuristic -- see the fix in that
        method's TOP-direction block for why this matters: without it, a
        manually-chosen TOP/BOTTOM swap would be silently overwritten
        whenever the beam's height axis is well-aligned with world Z (the
        common case), since that heuristic re-derives the sign from
        scratch on every call regardless of what was already committed.
        Call this ONCE, right after you have `initial_beam_points`, and
        BEFORE the first call to `create_base_meshes_bidirectional_local`
        or `detect_penetrations_bidirectional`:
            cache_manager.interactive_alignment_gui(initial_beam_points)
            # ... continue with the existing pipeline as normal ...
        WARNING Main-thread only: this opens a modal QDialog. Only safe to
        call from wherever frame-0 seeding already runs synchronously (e.g.
        start_animation()'s update_frame_visualization() call) -- never from
        the background _FrameWorker thread.
        Args:
            beam_points_local: (N, 3) beam points in local coordinates
            obb_center, obb_half_extents: unused legacy params, kept only
                for call-site compatibility — see obb_margin below, the
                box drawn here is now ALWAYS a TIGHT fit computed live
                from beam_points_local and the CURRENT (possibly swapped)
                length/width/height axes, not a loose axis-aligned box.
            obb_margin: padding added to the tight fit (e.g. the probe
                reach / x_translation), matching
                simulation_engine.py's _compute_pca_oriented_obb() so the
                box shown here is exactly what the real runtime detection
                window will be once orientation is committed.
        Returns:
            side_info: the finalized (possibly swapped) side_info dict
        """
        _, side_info, pca_center = self._identify_perpendicular_to_height(beam_points_local)
        state = {
            "height_vector": side_info["height_vector"].copy(),
            "length_vector": side_info["length_vector"].copy(),
            "width_vector": side_info["width_vector"].copy(),
            "dimensions": dict(side_info["dimensions"]),
        }
        from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout
        from pyvistaqt import QtInteractor
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        dialog = QDialog()
        dialog.setWindowTitle("CAR POINT CLOUD - ORIENTATION & ALIGNMENT")
        dialog.resize(1500, 950)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        plotter = QtInteractor(dialog)
        layout.addWidget(plotter.interactor)
        point_cloud = pv.PolyData(beam_points_local)
        plotter.add_points(point_cloud, color="red", point_size=4, opacity=0.6,
                            render_points_as_spheres=True,
                            label=f"Point Cloud ({len(beam_points_local)})")
        frame_center = beam_points_local.mean(axis=0)
        dims = state["dimensions"]
        arrow_scale = max(dims["length"], dims["width"], dims["height"]) * 0.6
        arrow_actors = []
        def clear_arrows():
            for actor in arrow_actors:
                plotter.remove_actor(actor, render=False)
            arrow_actors.clear()
        def add_arrow(direction, color, label):
            vec = direction / np.linalg.norm(direction)
            arrow = pv.Arrow(start=frame_center, direction=vec, scale=arrow_scale)
            actor = plotter.add_mesh(arrow, color=color)
            arrow_actors.append(actor)
            tip = frame_center + vec * arrow_scale * 1.1
            label_actor = plotter.add_point_labels(
                [tip], [label], font_size=16, text_color=color,
                shape_color="white", shape_opacity=0.75, always_visible=True,
                show_points=False,
            )
            arrow_actors.append(label_actor)
        def redraw():
            clear_arrows()
            add_arrow(state["height_vector"], "blue", "TOP (+Z)")
            add_arrow(-state["height_vector"], "red", "BOTTOM (-Z)")
            add_arrow(state["width_vector"], "green", "LEFT (-X)")
            add_arrow(-state["width_vector"], "green", "RIGHT (+X)")
            add_arrow(state["length_vector"], "orange", "FRONT (+Y)")
            add_arrow(-state["length_vector"], "purple", "REAR (-Y)")
            # ── TIGHT, PCA-oriented detection window — recomputed from the
            # CURRENT (possibly just-swapped) axes every redraw, so the box
            # shown always matches what would actually be committed if
            # CONTINUE were clicked right now. Replaces the old loose,
            # axis-aligned box that never moved when orientation changed.
            R_local_now = np.column_stack(
                (state["length_vector"], state["width_vector"], state["height_vector"]))
            tight_center, tight_half_extents = _pca_tight_obb(
                beam_points_local, pca_center, R_local_now, margin=obb_margin)
            obb_actor = _add_obb_wireframe(
                plotter, tight_center, tight_half_extents, obb_R_local=R_local_now)
            if obb_actor is not None:
                arrow_actors.append(obb_actor)
            plotter.render()
        redraw()
        def swap_top_bottom(_flag):
            state["height_vector"] = -state["height_vector"]
            redraw()
        def swap_left_right(_flag):
            state["width_vector"] = -state["width_vector"]
            redraw()
        def swap_length_width(_flag):
            state["length_vector"], state["width_vector"] = (
                state["width_vector"], state["length_vector"]
            )
            state["dimensions"]["length"], state["dimensions"]["width"] = (
                state["dimensions"]["width"], state["dimensions"]["length"]
            )
            redraw()
        def swap_length_height(_flag):
            state["length_vector"], state["height_vector"] = (
                state["height_vector"], state["length_vector"]
            )
            state["dimensions"]["length"], state["dimensions"]["height"] = (
                state["dimensions"]["height"], state["dimensions"]["length"]
            )
            redraw()
        def on_continue(_flag):
            dialog.accept()
        panel_x = 1230
        y0 = 860
        plotter.add_text("ORIENTATION ALIGNMENT", position=(panel_x, y0 + 30),
                          font_size=13, color="black")
        plotter.add_checkbox_button_widget(
            swap_top_bottom, position=(panel_x, y0 - 40), size=42,
            color_on="lightblue", color_off="lightblue"
        )
        plotter.add_text("Swap TOP <-> BOTTOM", position=(panel_x + 55, y0 - 30),
                          font_size=10, color="black")
        plotter.add_checkbox_button_widget(
            swap_left_right, position=(panel_x, y0 - 100), size=42,
            color_on="lightgreen", color_off="lightgreen"
        )
        plotter.add_text("Swap LEFT <-> RIGHT", position=(panel_x + 55, y0 - 90),
                          font_size=10, color="black")
        plotter.add_checkbox_button_widget(
            swap_length_width, position=(panel_x, y0 - 160), size=42,
            color_on="moccasin", color_off="moccasin"
        )
        plotter.add_text("Swap LENGTH <-> WIDTH", position=(panel_x + 55, y0 - 150),
                          font_size=10, color="black")
        plotter.add_checkbox_button_widget(
            swap_length_height, position=(panel_x, y0 - 220), size=42,
            color_on="plum", color_off="plum"
        )
        plotter.add_text("Swap LENGTH <-> HEIGHT", position=(panel_x + 55, y0 - 210),
                          font_size=10, color="black")
        plotter.add_checkbox_button_widget(
            on_continue, position=(panel_x, y0 - 320), size=55,
            color_on="seagreen", color_off="seagreen"
        )
        plotter.add_text("CONTINUE ->", position=(panel_x + 65, y0 - 305),
                          font_size=12, color="darkgreen")
        origin_axes = create_axis_frame(
            origin=np.array([0.0, 0.0, 0.0]),
            x_axis=np.array([0.3, 0, 0]),
            y_axis=np.array([0, 0.3, 0]),
            z_axis=np.array([0, 0, 0.3]),
        )
        plotter.add_mesh(origin_axes, color="gray", opacity=0.4)
        # NOTE: the detection-window wireframe is no longer drawn here as a
        # one-off static box — it's drawn (and kept live-updated) inside
        # redraw() above instead, so it stays a TIGHT fit that tracks
        # whichever axes are currently selected, including after any of
        # the four swap buttons are clicked.
        plotter.camera.position = (
            frame_center[0] + 3, frame_center[1] - 3, frame_center[2] + 2
        )
        plotter.camera.focal_point = tuple(frame_center)
        print("\n" + "=" * 70)
        print("INTERACTIVE ORIENTATION ALIGNMENT -- verify then click CONTINUE")
        print("=" * 70)
        dialog.show()
        dialog.exec_()

        # ── COMMIT FIRST, tear down the embedded plotter SECOND ──────────
        # FIX (root cause of "manual swap didn't stick, second popup showed
        # a totally different/wrong orientation"): plotter.close() here
        # tears down an embedded QtInteractor — a known trouble spot on
        # some pyvistaqt/VTK builds, where it can itself raise. In the
        # PREVIOUS ordering, everything below that call — including
        # self.pca_info = {...} and self._orientation_manually_set = True —
        # never ran if close() raised. That exception was silently caught
        # by _start_clash_engine()'s wrapping try/except (logs a
        # traceback, doesn't crash), so the user's manual swaps in the
        # dialog above were committed to nothing: pca_info stayed None,
        # _orientation_manually_set stayed False. The very next thing that
        # ran was create_base_meshes_bidirectional_local(), which — seeing
        # pca_info is None — recomputed PCA from scratch (ignoring every
        # swap just made) AND popped the OTHER, unrelated preview popup
        # (visualize_points_with_direction_frames, gated on
        # _orientation_manually_set being False) showing whatever
        # orientation raw PCA happened to produce. That's exactly the
        # "second pop went totally wrong" symptom — it never saw the
        # committed state because there wasn't any yet.
        #
        # Committing everything BEFORE close() means a close-time
        # exception can no longer erase a successful commit; at worst it
        # leaves a leaked/still-visible interactor window, not silently
        # wrong downstream orientation.
        pca_basis = np.column_stack((
            state["length_vector"], state["width_vector"], state["height_vector"]
        ))
        print(f"[PCA] committed basis det={np.linalg.det(pca_basis):+.6f} "
              f"(diagnostic only; not used as world transform)")
        side_info["height_vector"] = state["height_vector"]
        side_info["length_vector"] = state["length_vector"]
        side_info["width_vector"] = state["width_vector"]
        side_info["dimensions"] = state["dimensions"]
        self.pca_info = {
            "direction": side_info["width_vector"] / np.linalg.norm(side_info["width_vector"]),
            "side_info": side_info,
            "pca_center": pca_center,
        }
        # FIX: mark orientation as manually committed so
        # create_base_meshes_bidirectional_local() skips the auto TOP-
        # direction heuristic and uses side_info['height_vector'] (this
        # committed value) directly instead of silently recomputing it.
        self._orientation_manually_set = True
        print("\nOrientation alignment committed to self.pca_info:")
        print(f"   Height (TOP):   {side_info['height_vector']}")
        print(f"   Length (FRONT): {side_info['length_vector']}")
        print(f"   Width  (LEFT):  {side_info['width_vector']}")
        print("   Downstream pipeline will reuse this orientation (pca_info is no longer None).\n")

        try:
            plotter.close()
        except Exception:
            import traceback
            print("[detector] ⚠️ plotter.close() raised while tearing down "
                  "the interactive alignment window — orientation was "
                  "ALREADY committed above, so this is safe to ignore:")
            traceback.print_exc()

        return side_info

    def set_project_context(self, project_dir, beam_ids):
        """
        Wire in what _load_selective_indices() needs to find and correctly
        interpret PreviewTab's saved per-side selections. Call this BEFORE
        create_base_meshes_bidirectional_local() runs (e.g. right after
        _initialize_beam_geometry() in simulation_engine.py's
        _start_clash_engine()).

        Args:
            project_dir: the active project's root directory — same value
                as simulation_engine.py's self._active_project_path.
                sides_summary.json is read from
                <project_dir>/preview/sides_summary.json (written
                by PreviewTab._export_named_sides()).
            beam_ids: (N,) array of ORIGINAL point IDs, one per row,
                aligned with whatever beam_points_local /
                beam_points_centered you pass into
                create_base_meshes_bidirectional_local(). This is
                REQUIRED to correctly translate PreviewTab's saved
                original_ids (which reference the pre-cut original point
                cloud) into row indices valid for the CURRENT (possibly
                cut/edited) beam array — see _load_selective_indices()'s
                docstring for why a direct row-index reuse is wrong.
        """
        self.project_dir = project_dir
        self.beam_ids = beam_ids

    def _load_selective_indices(self):
        """
        Load PreviewTab's per-side point selections (Left/Right/Top/
        Front/Rear) for the CURRENT beam array.

        LOOKUP ORDER, per side:
          1. FAST PATH — the well-known default location a side lands at
             when its name is never touched in the Save-Sides rename
             dialog: <project_dir>/preview/<CanonicalName>/
             <CanonicalName>.npy, e.g. preview/Left/Left.npy,
             preview/Rear/Rear.npy (see _CANONICAL_SIDE_DISPLAY below).
             Now that preview_tab.py's _VIEW_DISPLAY_NAMES and
             commit_cut() both correctly target the side the user
             actually selected, this IS what _export_named_sides()
             produces for every side left at its default name — which is
             the common case — so checking it directly, with no JSON
             parsing at all, is both simpler and the primary path.
          2. FALLBACK — only if step 1's file doesn't exist: glob every
             <project_dir>/preview/*/selection.json and match on that
             file's own "side" field. This is what still finds a side
             that WAS manually renamed in the dialog (e.g. side "left"
             saved under preview/Door/) — the on-disk folder/file name
             can't be trusted to equal the canonical side name in that
             case, but the "side" key written inside every
             selection.json always can (see side_summary["side"] = side
             in _export_named_sides()).

        WHY THIS ISN'T A DIRECT ROW-INDEX LOAD:
        preview_tab.py saves `self.data.original_ids[mask]` per side —
        IDs into the ORIGINAL, uncut point cloud (see PreviewTab.
        _export_named_sides() and _slice_point_data()'s docstring:
        "original_ids are sliced, not renumbered"). But the beam array
        this cache manager actually operates on is
        ModelRepository.active_points — working_points (the EDITED/CUT
        cloud) whenever modified=True. Those are two different index
        spaces: a saved original_id can legitimately be >= the current
        beam array's length (points were removed), or simply refer to a
        different row than its numeric value suggests. Treating a saved
        original_id as a raw row index into the current, possibly-smaller
        beam array is exactly what caused:
            IndexError: index 5049 is out of bounds for axis 0 with size 5048
        The fix: require self.beam_ids (see set_project_context()) — the
        CURRENT beam array's own original-ID column, row-aligned with it
        — and build an id -> current-row-index lookup. Saved original_ids
        that no longer exist in self.beam_ids (because that point was cut
        away in Preview) are silently dropped, not treated as an error.
        """
        self.selective_indices = None
        self.selective_indices_left = None
        self.selective_indices_right = None
        self.selective_indices_top = None
        self.selective_indices_front = None
        self.selective_indices_back = None
       

        if not self.project_dir:
            print("⚠️  No project_dir set (call set_project_context() first) — "
                  "using all beam points")
            return False
        if self.beam_ids is None:
            print("⚠️  No beam_ids set (call set_project_context() first) — "
                  "using all beam points")
            return False

        preview_dir = os.path.join(self.project_dir, "preview")

        # detector.py calls this direction "BACK"; preview_tab.py's
        # canonical side name for the same direction is "rear".
        canonical_by_direction = {
            "left": "left", "right": "right", "top": "top",
            "front": "front", "back": "rear",
        }

        # id -> current row index, built ONCE from the CURRENT beam array's
        # own IDs — this is what actually fixes the index-space mismatch.
        id_to_row = {int(pid): row for row, pid in enumerate(self.beam_ids)}

        # Fallback map (canonical side -> its own selection.json dict) is
        # only built lazily, the first time some side's fast path misses
        # — no point globbing/parsing JSON for sides that were never
        # renamed and resolve on the fast path alone.
        _fallback_cache = {"built": False, "map": {}}

        def _build_fallback_map():
            entries = {}
            for sel_path in sorted(glob.glob(os.path.join(preview_dir, "*", "selection.json"))):
                try:
                    with open(sel_path, "r", encoding="utf-8") as f:
                        side_summary = json.load(f)
                except Exception as e:
                    print(f"❌ Error reading {sel_path}: {e}")
                    continue
                side_key = side_summary.get("side")
                if not side_key:
                    continue
                side_summary["_selection_dir"] = os.path.dirname(sel_path)
                entries[side_key] = side_summary
            _fallback_cache["map"] = entries
            _fallback_cache["built"] = True

        def _load_side(direction_key):
            canonical = canonical_by_direction[direction_key]
            display = _CANONICAL_SIDE_DISPLAY[canonical]

            # 1) FAST PATH — default, un-renamed location for this side.
            ids_path = os.path.join(preview_dir, display, f"{display}.npy")
            how = "default path"

            if not os.path.exists(ids_path):
                # 2) FALLBACK — side was renamed; find it by its own
                # selection.json "side" field instead of guessing a
                # folder name.
                if not _fallback_cache["built"]:
                    _build_fallback_map()
                entry = _fallback_cache["map"].get(canonical)
                if entry is None:
                    print(f"❌ '{canonical}' not found at {ids_path}, and no "
                          f"renamed preview/*/selection.json claims "
                          f"\"side\": \"{canonical}\" either — using all "
                          f"beam points")
                    return None

                candidate = entry.get("point_ids_npy")
                sel_dir = entry.get("_selection_dir")
                name = entry.get("name")
                sibling = os.path.join(sel_dir, f"{name}.npy") if sel_dir and name else None
                if candidate and os.path.exists(candidate):
                    ids_path = candidate
                elif sibling and os.path.exists(sibling):
                    ids_path = sibling
                else:
                    print(f"❌ ids file for '{canonical}' missing on disk "
                          f"(tried default path, stored point_ids_npy, and "
                          f"sibling file)")
                    return None
                how = f"renamed as '{name}'"

            try:
                saved_ids = np.load(ids_path)
            except Exception as e:
                print(f"❌ Error loading {ids_path}: {e}")
                return None

            rows = [id_to_row[pid] for pid in saved_ids.astype(int) if pid in id_to_row]
            dropped = len(saved_ids) - len(rows)
            if dropped > 0:
                print(f"   ⚠️  '{canonical}': {dropped}/{len(saved_ids)} saved point IDs "
                      f"no longer exist in the current beam array (cut away in "
                      f"Preview) — dropped, not treated as an error.")
            if not rows:
                print(f"❌ '{canonical}': 0 of {len(saved_ids)} saved IDs matched "
                      f"the current beam array — using all beam points for this side")
                return None
            print(f"✅ Loaded '{canonical}' selection ({how}): {len(rows)} points "
                  f"(of {len(saved_ids)} saved)")
            return np.array(rows, dtype=int)

        self.selective_indices_left  = _load_side("left")
        self.selective_indices_right = _load_side("right")
        self.selective_indices_top   = _load_side("top")
        self.selective_indices_front = _load_side("front")
        self.selective_indices_back  = _load_side("back")

        # self.selective_indices (the COMBINED "all selected sides" array,
        # used only as the PCA basis in _get_filtered_beam_points()) has no
        # equivalent in preview_repository.py's actual save format — there
        # is no single combined-selection ids file, only per-side ones, and
        # PCA should run on the WHOLE beam anyway, not an arbitrary subset.
        # Deliberately left None; _get_filtered_beam_points() below treats
        # that as "use the full beam_points for the PCA-basis return value".
        self.selective_indices = None

        any_loaded = any(
            arr is not None for arr in (
                self.selective_indices_left, self.selective_indices_right,
                self.selective_indices_top, self.selective_indices_front,
                self.selective_indices_back,
            )
        )
        return any_loaded

    def _get_filtered_beam_points(self, beam_points):
        """
        Filter beam points using selective indices with validation.

        Args:
            beam_points: (N, 3) array of beam starting points

        Returns:
            filtered_points: (M, 3) array where M <= N
        """
        # No combined "all sides" index (see _load_selective_indices()'s
        # note) — PCA basis is always the FULL beam, not a filtered subset.
        filtered_points = beam_points

        def _side(indices):
            return beam_points if indices is None else beam_points[indices]

        filtered_points_left  = _side(self.selective_indices_left)
        filtered_points_right = _side(self.selective_indices_right)
        filtered_points_top   = _side(self.selective_indices_top)
        filtered_points_front = _side(self.selective_indices_front)
        filtered_points_back  = _side(self.selective_indices_back)

        return filtered_points, filtered_points_left, filtered_points_right, filtered_points_top, filtered_points_front, filtered_points_back
    
    def create_base_meshes_bidirectional_local(self, beam_points_local, length: float, thickness=0.01,
                                               force_recreate=False, obb_center=None, obb_half_extents=None):
        """
        Create BASE MESHES in LOCAL coordinates (ONCE at initialization).
        
        These meshes are created from initial_beam_points that are:
        - Already centered at origin (beam_points_local)
        - Without any rotation or translation
        
        The vertices of these meshes are stored and will be transformed
        each frame using the frame's R and T matrix.
        
        Args:
            beam_points_local: (N, 3) beam points CENTERED AT ORIGIN (local coords)
            thickness: beam thickness
            force_recreate: if True, recreate even if exists
            obb_center, obb_half_extents: optional detection-window params,
                passed straight through to visualize_points_with_direction_
                frames() (the plain-preview fallback below) so that popup
                can draw the actual OBB too, same as interactive_alignment_
                gui() does. No effect on mesh geometry itself.
        
        Returns:
            tuple: (mesh_left, mesh_right, mesh_front, mesh_back, mesh_top) with LOCAL vertices/triangles
        """
        if (self.meshes_initialized and not force_recreate):
            print("✅ Reusing existing LOCAL bidirectional meshes (LEFT, RIGHT, FRONT, BACK, TOP)")
            return (self.base_mesh_left_local, self.base_mesh_right_local, 
                    self.base_mesh_front_local, self.base_mesh_back_local, self.base_mesh_top_local)
        
        print("\n" + "="*70)
        print("🔨 CREATING BASE MESHES IN LOCAL COORDINATES (ONCE)")
        print("="*70)
        
        # Compute PCA from local beam points
        if self.pca_info is None:
            width_direction, side_info, pca_center = self._identify_perpendicular_to_height(beam_points_local)
            self.pca_info = {
                "direction": width_direction,
                "side_info": side_info,
                "pca_center": pca_center
            }
            print(f"\n📍 PCA Analysis (LOCAL COORDINATES):")
            print(f"   PCA Center: {pca_center}")
            print(f"   Width Direction: {width_direction}")
            print(f"   Height Direction: {side_info['height_vector']}")
            print(f"   Car Dimensions:")
            print(f"      Length: {side_info['dimensions']['length']:.3f}m")
            print(f"      Width:  {side_info['dimensions']['width']:.3f}m")
            print(f"      Height: {side_info['dimensions']['height']:.3f}m")
        else:
            width_direction = self.pca_info["direction"]
            side_info = self.pca_info["side_info"]
            pca_center = self.pca_info["pca_center"]
        
        print(f"\n   Extrusion Thickness: {thickness:.4f}m")
        
        # ===== TOP DIRECTION =====
        # FIX: only auto-detect if the orientation was NOT already
        # committed via interactive_alignment_gui(). Previously this
        # unconditionally called _detect_top_direction() even when
        # self.pca_info had just been set by a manual TOP<->BOTTOM swap —
        # that heuristic re-derives the sign from scratch every time, so
        # whenever the beam's height axis is well-aligned with world Z
        # (z_component_abs > 0.9, the common case), it silently forced the
        # sign back toward +Z regardless of what the user chose. The
        # manual choice is now respected: side_info['height_vector'] IS
        # the committed value in that case.
        if self._orientation_manually_set:
            height_direction = side_info['height_vector'] / np.linalg.norm(side_info['height_vector'])
            print(f"\n✅ USING MANUALLY-COMMITTED TOP DIRECTION: {height_direction}")
            print(f"   (from interactive_alignment_gui — auto-detection skipped)")
        else:
            height_direction = self._detect_top_direction(beam_points_local, side_info)
            print(f"\n✅ AUTO-DETECTED TOP DIRECTION: {height_direction}")
            print(f"   (Points spread UPWARD from centroid)")
        
        # ===== SHOW PREVIEW POPUP =====
        # FIX: skip entirely if interactive_alignment_gui() already ran —
        # that dialog already showed and confirmed the orientation (via a
        # crash-safe Qt-embedded plotter); popping this SECOND, unrelated
        # popup (a bare pv.Plotter() with its own competing native
        # interactor — see interactive_alignment_gui()'s docstring for why
        # that's a real crash risk) would be redundant at best.
        if not self._orientation_manually_set:
            print("\n Opening PREVIEW visualization...")
            print("   (Verify beam points and directional frames before mesh creation)")
            self.visualize_points_with_direction_frames(
                beam_points_local, side_info,
                obb_center=obb_center, obb_half_extents=obb_half_extents,
                obb_margin=length)
        
        # Get length direction
        length_direction = side_info['length_vector'] / np.linalg.norm(side_info['length_vector'])
        
        # CRITICAL: Load selective indices BEFORE filtering
        print("\n🔄 Loading selective indices...")
        if self.use_selective_points:
            indices_loaded = self._load_selective_indices()
            if not indices_loaded:
                print("⚠️  Proceeding with all beam points (selective indices unavailable)")
                self.use_selective_points = False
        
        # CRITICAL: Filter beam points using selective indices
        print("\n🔄 Filtering beam points...")
        beam_points_used, left_points, right_points, top_points, front_points, back_points = self._get_filtered_beam_points(beam_points_local)
          

        beam_points_local = beam_points_used  

        # ===== DEBUG: confirm each side's probe is built from a SUBSET,
        # not the whole beam. If use_selective_points is False or indices
        # failed to load, left_points/right_points/etc. are all just
        # beam_points_local in full — meaning the LEFT probe extrudes the
        # ENTIRE car leftward instead of just its left-facing surface, and
        # same for every other direction. This is cause #2 from the
        # earlier analysis — probes shaped wrong, independent of any R/t
        # or OBB issue. =====
        n_full = len(beam_points_local)
        for name, pts in (("LEFT", left_points), ("RIGHT", right_points),
                          ("TOP", top_points), ("FRONT", front_points),
                          ("BACK", back_points)):
            frac = len(pts) / max(n_full, 1)
            flag = "⚠️ USING FULL BEAM (no per-side selection)" if frac > 0.95 else "✓ subset"
            print(f"   [DEBUG] {name} probe source points: {len(pts):,} / {n_full:,} "
                  f"({frac:.0%}) {flag}")

        # LEFT mesh: extrude along positive WIDTH direction
        mesh_left = self._create_direction_mesh_local(
            left_points, thickness, length,
            "LEFT", width_direction, side_info
        )
        
        # RIGHT mesh: extrude along negative WIDTH direction
        mesh_right = self._create_direction_mesh_local(
            right_points, thickness, length,
            "RIGHT", -width_direction, side_info
        )
        
        # FRONT mesh: extrude along positive LENGTH direction
        mesh_front = self._create_direction_mesh_local(
            front_points, thickness, length,
            "FRONT", -length_direction, side_info
        )
        
        # BACK mesh: extrude along negative LENGTH direction
        mesh_back = self._create_direction_mesh_local(
            back_points, thickness, length,
            "BACK", length_direction, side_info
        )
        
        # TOP mesh: extrude along positive HEIGHT direction
        mesh_top = self._create_direction_mesh_local(
            top_points, thickness, length,
            "TOP", height_direction, side_info
        )
        
        if (mesh_left is None or mesh_right is None or mesh_front is None or 
            mesh_back is None or mesh_top is None):
            print("❌ Failed to create bidirectional meshes")
            return None, None, None, None, None
        
        # ===== STORE BASE MESHES AND VERTICES =====
        self.base_mesh_left_local = mesh_left
        self.base_mesh_right_local = mesh_right
        self.base_mesh_front_local = mesh_front
        self.base_mesh_back_local = mesh_back
        self.base_mesh_top_local = mesh_top
        
        # Extract and store vertices (for efficient transformation each frame)
        self.base_vertices_left_local = np.asarray(mesh_left.vertices, dtype=np.float64).copy()
        self.base_vertices_right_local = np.asarray(mesh_right.vertices, dtype=np.float64).copy()
        self.base_vertices_front_local = np.asarray(mesh_front.vertices, dtype=np.float64).copy()
        self.base_vertices_back_local = np.asarray(mesh_back.vertices, dtype=np.float64).copy()
        self.base_vertices_top_local = np.asarray(mesh_top.vertices, dtype=np.float64).copy()
        
        # Store triangle connectivity (never changes)
        self.base_triangles_left = np.asarray(mesh_left.triangles, dtype=np.int32).copy()
        self.base_triangles_right = np.asarray(mesh_right.triangles, dtype=np.int32).copy()
        self.base_triangles_front = np.asarray(mesh_front.triangles, dtype=np.int32).copy()
        self.base_triangles_back = np.asarray(mesh_back.triangles, dtype=np.int32).copy()
        self.base_triangles_top = np.asarray(mesh_top.triangles, dtype=np.int32).copy()
        
        self.base_pose = beam_points_local.copy()
        self.meshes_initialized = True

        # Build each direction's RaycastingScene ONCE, right here, from the
        # never-transformed LOCAL geometry — see _build_static_scenes().
        self._build_static_scenes()

        print(f"\n✅ Base LOCAL meshes created and stored:")
        print(f"   LEFT (+WIDTH):    {len(self.base_triangles_left):,} triangles, {len(self.base_vertices_left_local):,} vertices")
        print(f"   RIGHT (-WIDTH):   {len(self.base_triangles_right):,} triangles, {len(self.base_vertices_right_local):,} vertices")
        print(f"   FRONT (+LENGTH):  {len(self.base_triangles_front):,} triangles, {len(self.base_vertices_front_local):,} vertices")
        print(f"   BACK (-LENGTH):   {len(self.base_triangles_back):,} triangles, {len(self.base_vertices_back_local):,} vertices")
        print(f"   TOP (+HEIGHT):    {len(self.base_triangles_top):,} triangles, {len(self.base_vertices_top_local):,} vertices")
        print("="*70)
        
        return mesh_left, mesh_right, mesh_front, mesh_back, mesh_top

    def _create_direction_mesh_local(self, start_points, thickness, length, direction_name, 
                                     direction_vector, side_info):
        """
        Create mesh along PCA-identified direction vector in LOCAL coordinates.
        
        Args:
            start_points: beam starting points in LOCAL coords (N, 3) - centered at origin
            thickness: beam thickness
            direction_name: "LEFT", "RIGHT", "FRONT", "BACK", or "TOP"
            direction_vector: normalized direction vector
            side_info: dictionary with PCA axis information
        
        Returns:
            o3d.geometry.TriangleMesh: combined mesh in local coordinates
        """
        print(f"\n   --- Creating {direction_name} Mesh ---")
        
        dir_normalized = direction_vector / np.linalg.norm(direction_vector)
        all_segments = []
        
        # Use appropriate extrusion length
        if direction_name == "TOP":
            extrusion_length = side_info['dimensions']['height']
        elif direction_name in ["FRONT", "BACK"]:
            extrusion_length = side_info['dimensions']['length']
        else:  # LEFT, RIGHT
            extrusion_length = side_info['dimensions']['width']
        
        for idx, start in enumerate(start_points):
            box = o3d.geometry.TriangleMesh.create_box(
                width=length,
                height=thickness, 
                depth=thickness
            )
            box.compute_vertex_normals()
            
            # Calculate rotation to align box's X-axis with direction_vector
            default_dir = np.array([1, 0, 0])
            
            if np.allclose(dir_normalized, default_dir, atol=1e-6):
                R = np.eye(3)
            elif np.allclose(dir_normalized, -default_dir, atol=1e-6):
                R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
            else:
                v = np.cross(default_dir, dir_normalized)
                c = np.dot(default_dir, dir_normalized)
                vx = np.array([[0, -v[2], v[1]],
                            [v[2], 0, -v[0]],
                            [-v[1], v[0], 0]])
                
                v_norm_sq = np.linalg.norm(v) ** 2
                if v_norm_sq < 1e-10:
                    R = np.eye(3)
                else:
                    R = np.eye(3) + vx + vx @ vx * ((1 - c) / v_norm_sq)
            
            box.rotate(R, center=(0, 0, 0))
            box.translate(start)
            all_segments.append(box)
        
        if not all_segments:
            print(f"      ❌ No valid mesh segments for {direction_name}")
            return None
        
        print(f"      ✓ {direction_name}: {len(all_segments):,} segments")
        
        full_mesh = all_segments[0]
        for seg in all_segments[1:]:
            full_mesh += seg
        
        full_mesh.compute_vertex_normals()
        full_mesh.remove_duplicated_vertices()
        full_mesh.remove_degenerate_triangles()
        
        return full_mesh

    def transform_mesh_vertices_to_world(self, vertices_local, R, t):
        """
        Transform mesh vertices from LOCAL to WORLD coordinates.
        
        This is the core transformation applied each frame:
        v_world = R @ v_local + t
        
        Args:
            vertices_local: (N, 3) vertices in local coordinates
            R: (3x3) rotation matrix
            t: (3,) translation vector
        
        Returns:
            (N, 3) vertices in world coordinates
        """
        vertices_world = (R @ vertices_local.T).T + t
        return vertices_world

    def create_transformed_mesh_and_scene(self, mesh_direction, R, t, frame_idx):
        """
        Create a TRANSFORMED mesh and RAYCASTING SCENE for the current frame.
        
        Args:
            mesh_direction: "LEFT", "RIGHT", "FRONT", "BACK", or "TOP"
            R: (3x3) rotation matrix from local to world
            t: (3,) translation vector from local to world
            frame_idx: current frame number (for info)
        
        Returns:
            (mesh_world, scene) tuple or (None, None) on error
        """
        # Select the appropriate base vertices and triangles
        if mesh_direction == "LEFT":
            vertices_local = self.base_vertices_left_local
            triangles = self.base_triangles_left
        elif mesh_direction == "RIGHT":
            vertices_local = self.base_vertices_right_local
            triangles = self.base_triangles_right
        elif mesh_direction == "FRONT":
            vertices_local = self.base_vertices_front_local
            triangles = self.base_triangles_front
        elif mesh_direction == "BACK":
            vertices_local = self.base_vertices_back_local
            triangles = self.base_triangles_back
        elif mesh_direction == "TOP":
            vertices_local = self.base_vertices_top_local
            triangles = self.base_triangles_top
        else:
            print(f"❌ Unknown mesh direction: {mesh_direction}")
            return None, None
        
        # Transform vertices from LOCAL to WORLD
        vertices_world = self.transform_mesh_vertices_to_world(vertices_local, R, t)
        
        # Create mesh in world coordinates
        mesh_world = o3d.geometry.TriangleMesh()
        mesh_world.vertices = o3d.utility.Vector3dVector(vertices_world)
        mesh_world.triangles = o3d.utility.Vector3iVector(triangles)
        
        # Create raycasting scene
        try:
            tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_world)
            scene = o3d.t.geometry.RaycastingScene()
            _ = scene.add_triangles(tmesh)
            return mesh_world, scene
        except Exception as e:
            print(f"❌ Error creating raycasting scene for {mesh_direction} (frame {frame_idx}): {e}")
            return None, None

    # =========================================================================
    #  STATIC LOCAL SCENES  —  BUILT ONCE, REUSED EVERY FRAME
    # =========================================================================
    # NOTE: create_transformed_mesh_and_scene() above (transform mesh ->
    # world, rebuild TriangleMesh + RaycastingScene from scratch) is no
    # longer called from the per-frame hot path (detect_penetrations_
    # bidirectional()). It rebuilt a full BVH 5x per frame — the dominant,
    # continuous GPU cost during playback (observed as GPU pinned ~99% and
    # the app going "Not Responding"). It's left intact above only in case
    # something needs a genuine world-space mesh/scene pair later (e.g. a
    # one-off debug visualization); it is NOT part of the steady-state loop
    # anymore.
    #
    # All five direction meshes share the SAME rigid transform (R, t) each
    # frame — only the LOCAL geometry differs by direction. So instead of
    # moving 5 meshes into world space and rebuilding 5 BVHs every frame,
    # each direction's RaycastingScene is built ONCE here, directly from
    # the never-transformed LOCAL vertices/triangles. Each frame then only
    # needs the INVERSE transform applied to the (already OBB-filtered,
    # much smaller) query points — one cheap NumPy matmul, computed once
    # and shared across all 5 static scenes — instead of 5 mesh rebuilds.
    def _build_static_scenes(self) -> None:
        directions = (
            ("LEFT",  self.base_vertices_left_local,  self.base_triangles_left),
            ("RIGHT", self.base_vertices_right_local, self.base_triangles_right),
            ("FRONT", self.base_vertices_front_local, self.base_triangles_front),
            ("BACK",  self.base_vertices_back_local,  self.base_triangles_back),
            ("TOP",   self.base_vertices_top_local,   self.base_triangles_top),
        )
        self._static_scenes = {}
        for name, verts, tris in directions:
            if verts is None or tris is None:
                print(f"⚠️ Skipping static scene for {name} — base geometry missing")
                continue
            mesh_local = o3d.geometry.TriangleMesh()
            mesh_local.vertices = o3d.utility.Vector3dVector(verts)
            mesh_local.triangles = o3d.utility.Vector3iVector(tris)
            tmesh_local = o3d.t.geometry.TriangleMesh.from_legacy(mesh_local)
            scene_local = o3d.t.geometry.RaycastingScene()
            scene_local.add_triangles(tmesh_local)
            self._static_scenes[name] = scene_local

        print(f"✅ Static local raycasting scenes built ONCE for "
              f"{list(self._static_scenes.keys())} — no more per-frame BVH rebuilds")

    def get_static_scene(self, direction: str):
        """Return the pre-built, never-rebuilt RaycastingScene for
        `direction` ("LEFT"/"RIGHT"/"FRONT"/"BACK"/"TOP"), or None if
        _build_static_scenes() hasn't run yet (i.e. before the first
        frame's base-mesh initialization)."""
        return self._static_scenes.get(direction)

    def transform_points_world_to_local(self, points_world: np.ndarray,
                                        R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Inverse of transform_mesh_vertices_to_world(): p_local = Rᵀ(p_world - t).

        Signed distance is invariant under a rigid transform, so querying
        `points_world` against a world-space mesh gives the identical
        result to querying these local-space points against the matching
        LOCAL (static) scene. Since every direction shares the same (R, t)
        this frame, callers compute this ONCE per frame and reuse it for
        all 5 directions' compute_signed_distance() calls.
        """
        return (R.T @ (points_world - t).T).T

    def get_cache_filename(self, frame_idx, direction) -> str:
        """Get cache filename for a specific frame and direction"""
        return os.path.join(self.cache_dir, f"frame_{frame_idx}_{direction}_cache.npz")
   
    def load_frame_cache(self, frame_idx, direction) -> dict:
        """Load cached data for a specific frame and direction"""
        cache_file = self.get_cache_filename(frame_idx, direction)
        if not os.path.exists(cache_file):
            return {}
       
        try:
            with np.load(cache_file, allow_pickle=True) as data:
                cache_data = {}
                for key in data.files:
                    if key.startswith('x_') and key.endswith('_indices'):
                        x_val_str = key[2:-8]
                        try:
                            x_val = float(x_val_str)
                            sd_key = f"x_{x_val_str}_sd"
                            cache_data[x_val] = {
                                'indices': data[key],
                                'sd_distances': data[sd_key] if sd_key in data else None
                            }
                        except ValueError:
                            continue
                return cache_data
        except Exception as e:
            print(f"⚠️ Error loading cache for frame {frame_idx} ({direction}): {e}")
            return {}
   
    def save_frame_cache(self, frame_idx, direction, x_translation: float,
                        indices: np.ndarray, sd_distances: np.ndarray = None):
        """
        Queue this frame/direction's cache write to the shared io_worker
        background pool instead of writing inline. This method is called
        from _detect_single_direction(), which runs INSIDE
        compute_frame_data() on the single _FrameWorker thread that owns
        all GPU/Open3D raycasting for a frame — a synchronous npz
        read-modify-write here would otherwise serialize disk I/O with
        that GPU work, once per direction, 5x per frame.

        get_cache_filename(frame_idx, direction) is unique per call (one
        file per frame+direction), so no cross-call ordering is needed —
        the cache_file path itself is used as io_worker's chain key,
        which just gives free concurrency across directions/frames
        bounded by io_worker's small pool size.
        """
        cache_file = self.get_cache_filename(frame_idx, direction)
        io_worker.submit(cache_file, self._write_frame_cache_to_disk,
                          cache_file, x_translation, indices, sd_distances)

    @staticmethod
    def _write_frame_cache_to_disk(cache_file, x_translation: float,
                                    indices: np.ndarray, sd_distances: np.ndarray = None):
        """The actual npz read-modify-write. Runs on an io_worker background
        thread — never on the GPU-owning _FrameWorker thread."""
        existing_cache = {}
        if os.path.exists(cache_file):
            try:
                with np.load(cache_file, allow_pickle=True) as data:
                    for key in data.files:
                        existing_cache[key] = data[key]
            except Exception as e:
                print(f"⚠️ Error loading existing cache: {e}")

        existing_cache[f"x_{x_translation:.6f}_indices"] = indices
        if sd_distances is not None:
            existing_cache[f"x_{x_translation:.6f}_sd"] = sd_distances

        try:
            np.savez_compressed(cache_file, **existing_cache)
        except Exception as e:
            print(f"WARNING Error writing cache {cache_file}: {e}")

    def run_orientation_preflight(
        self,
        beam_points_local,
        trajectory_points,
        rolls, pitches, yaws,
        bbox_local_points,
        rotation_matrix_fn,
        start_frame=0,
        step=50,
        window_name="Orientation Preflight -- verify LEFT/RIGHT/TOP before running clash detection",
    ):
        """
        DIAGNOSTIC ONLY -- run this BEFORE trusting clash detection results.
        Animates the calibrated beam through the full trajectory using
        ONLY rotation + translation (no wall_points, no signed-distance
        query at all), so you can visually confirm LEFT/RIGHT/TOP/FRONT/
        BACK/BOTTOM stay correctly oriented at EVERY frame -- not just the
        calibration frame.

        CRITICAL: rotation_matrix_fn MUST be the exact same function
        frame_updater.compute_frame_data() uses in production (currently
        `simulation.utils.rotation_matrix`), NOT a different one such as
        `simulation.utils.create_rotation_matrix`. If those two functions
        use different Euler conventions/orders, this preflight would
        "pass" against a convention production never actually uses --
        pass in the production function explicitly, by reference, e.g.:
            from simulation.utils import rotation_matrix
            cache_manager.run_orientation_preflight(
                ..., rotation_matrix_fn=rotation_matrix)
        Do not hardcode a different rotation function here -- the whole
        point of this method is to catch exactly this kind of mismatch
        before it silently distorts every frame of playback.

        Requires create_base_meshes_bidirectional_local() (and, if used,
        interactive_alignment_gui()) to have ALREADY run on this
        cache_manager -- reads base_vertices_*_local / base_triangles_* /
        self.pca_info directly. No mesh creation happens in this method.

        Shows, all transformed by the SAME (R, t) every frame:
            - beam_points_local, as a point cloud
            - bbox_local_points (OBB corners), as a point cloud
            - the 5 probe meshes (LEFT/RIGHT/FRONT/BACK/TOP), color-coded
            - 6 direction arrows (LEFT/RIGHT/TOP/BOTTOM/FRONT/REAR),
              anchored at the beam's own local centroid, rotating rigidly
              with the body -- these are what you actually watch
            - the trajectory line (static -- already world space)

        Args:
            beam_points_local: (N,3) the CALIBRATED local beam array --
                same array you'd pass as initial_beam_points elsewhere
            trajectory_points: (F,3) world positions, one per frame
            rolls, pitches, yaws: (F,) arrays, production convention
            bbox_local_points: (8,3) OBB corners, SAME local frame as
                beam_points_local
            rotation_matrix_fn: callable(roll, pitch, yaw) -> (3,3) --
                see CRITICAL note above
            start_frame, step: which frames to play (skip for speed)
        """
        if self.pca_info is None or not self.meshes_initialized:
            raise RuntimeError(
                "run_orientation_preflight() requires create_base_meshes_bidirectional_local() "
                "(and optionally interactive_alignment_gui()) to have already run on this "
                "cache_manager -- no base_vertices_*_local / side_info to visualize yet."
            )

        side_info = self.pca_info["side_info"]

        # ---- static, once-only geometry -------------------------------
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(beam_points_local)
        pcd.paint_uniform_color([0.6, 0.6, 0.6])

        pcd_bbox = o3d.geometry.PointCloud()
        pcd_bbox.points = o3d.utility.Vector3dVector(bbox_local_points)
        pcd_bbox.paint_uniform_color([1.0, 1.0, 0.0])

        trajectory = o3d.geometry.LineSet()
        trajectory.points = o3d.utility.Vector3dVector(trajectory_points)
        trajectory.lines = o3d.utility.Vector2iVector(
            [[i, i + 1] for i in range(len(trajectory_points) - 1)])
        trajectory.paint_uniform_color([0.0, 1.0, 0.0])

        # ---- 5 probe meshes, color-coded, LOCAL vertices reused as-is --
        direction_colors = {
            "LEFT":  [1.0, 0.0, 0.0],
            "RIGHT": [0.0, 1.0, 0.0],
            "FRONT": [1.0, 0.65, 0.0],
            "BACK":  [0.6, 0.0, 0.8],
            "TOP":   [0.0, 0.5, 1.0],
        }
        probe_local_vt = {
            "LEFT":  (self.base_vertices_left_local,  self.base_triangles_left),
            "RIGHT": (self.base_vertices_right_local, self.base_triangles_right),
            "FRONT": (self.base_vertices_front_local, self.base_triangles_front),
            "BACK":  (self.base_vertices_back_local,  self.base_triangles_back),
            "TOP":   (self.base_vertices_top_local,   self.base_triangles_top),
        }
        probe_meshes = {}
        for name, (verts, tris) in probe_local_vt.items():
            if verts is None or tris is None:
                continue
            m = o3d.geometry.TriangleMesh()
            m.vertices = o3d.utility.Vector3dVector(verts)
            m.triangles = o3d.utility.Vector3iVector(tris)
            m.paint_uniform_color(direction_colors[name])
            m.compute_vertex_normals()
            probe_meshes[name] = m

        # ---- 6 direction arrows, built ONCE in local space, anchored at
        # the beam's own local centroid -- captured once, transformed
        # exactly like the probe meshes every frame after (same pattern
        # as base_vertices_*_local throughout this file). -------------
        arrow_defs = {
            "TOP":    (side_info["height_vector"],  [0.0, 0.5, 1.0]),
            "BOTTOM": (-side_info["height_vector"], [1.0, 0.0, 0.0]),
            "LEFT":   (side_info["width_vector"],   [0.0, 1.0, 0.0]),
            "RIGHT":  (-side_info["width_vector"],  [0.0, 1.0, 0.0]),
            "FRONT":  (side_info["length_vector"],  [1.0, 0.65, 0.0]),
            "REAR":   (-side_info["length_vector"], [0.6, 0.0, 0.8]),
        }
        dims = side_info["dimensions"]
        arrow_len = max(dims["length"], dims["width"], dims["height"]) * 0.6
        frame_center_local = beam_points_local.mean(axis=0)

        def _orient_arrow_local(direction_vec):
            dvec = direction_vec / np.linalg.norm(direction_vec)
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=arrow_len * 0.02, cone_radius=arrow_len * 0.04,
                cylinder_height=arrow_len * 0.8, cone_height=arrow_len * 0.2,
                resolution=8,
            )
            default_dir = np.array([0.0, 0.0, 1.0])  # create_arrow()'s default axis
            if np.allclose(dvec, default_dir, atol=1e-6):
                R_align = np.eye(3)
            elif np.allclose(dvec, -default_dir, atol=1e-6):
                R_align = np.diag([1.0, -1.0, -1.0])
            else:
                v = np.cross(default_dir, dvec)
                c = np.dot(default_dir, dvec)
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                R_align = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
            arrow.rotate(R_align, center=(0, 0, 0))
            arrow.translate(frame_center_local)
            return (np.asarray(arrow.vertices, dtype=np.float64).copy(),
                    np.asarray(arrow.triangles, dtype=np.int32).copy())

        arrow_meshes = {}
        arrow_local_vt = {}
        for name, (dvec, color) in arrow_defs.items():
            verts, tris = _orient_arrow_local(dvec)
            arrow_local_vt[name] = verts
            m = o3d.geometry.TriangleMesh()
            m.vertices = o3d.utility.Vector3dVector(verts)
            m.triangles = o3d.utility.Vector3iVector(tris)
            m.paint_uniform_color(color)
            m.compute_vertex_normals()
            arrow_meshes[name] = m

        # ---- window -----------------------------------------------------
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name)
        vis.add_geometry(pcd)
        vis.add_geometry(pcd_bbox)
        vis.add_geometry(trajectory)
        for m in probe_meshes.values():
            vis.add_geometry(m)
        for m in arrow_meshes.values():
            vis.add_geometry(m)

        print("\n" + "=" * 70)
        print(f"ORIENTATION PREFLIGHT -- rotation_matrix_fn = "
              f"{getattr(rotation_matrix_fn, '__module__', '?')}."
              f"{getattr(rotation_matrix_fn, '__qualname__', rotation_matrix_fn)}")
        print("  Watch LEFT (red) / RIGHT (green) / TOP (blue) arrows and meshes:")
        print("  they must stay pointed the same way RELATIVE TO THE BODY at")
        print("  every frame -- not just the calibration frame. Any visible")
        print("  twist that grows/changes across frames means rotation_matrix_fn")
        print("  does not match what production actually uses, or calib_frame_idx")
        print("  was wrong.")
        print("=" * 70 + "\n")

        for i in range(start_frame, len(trajectory_points), step):
            R = rotation_matrix_fn(rolls[i], pitches[i], yaws[i])
            t = trajectory_points[i]

            pcd.points = o3d.utility.Vector3dVector((R @ beam_points_local.T).T + t)
            pcd_bbox.points = o3d.utility.Vector3dVector((R @ bbox_local_points.T).T + t)

            for name, m in probe_meshes.items():
                verts_local, _ = probe_local_vt[name]
                m.vertices = o3d.utility.Vector3dVector((R @ verts_local.T).T + t)
                m.compute_vertex_normals()

            for name, m in arrow_meshes.items():
                verts_local = arrow_local_vt[name]
                m.vertices = o3d.utility.Vector3dVector((R @ verts_local.T).T + t)
                m.compute_vertex_normals()

            vis.update_geometry(pcd)
            vis.update_geometry(pcd_bbox)
            for m in probe_meshes.values():
                vis.update_geometry(m)
            for m in arrow_meshes.values():
                vis.update_geometry(m)

            vis.poll_events()
            vis.update_renderer()

        vis.destroy_window()

    def cleanup(self):
        """
        Clean up resources for THIS cache manager instance.

        Deliberately does NOT shut down io_worker here — io_worker is a
        single shared, app-wide pool (simulation/io_worker.py), and this
        cleanup() runs every time a new detection pass starts (see
        toggle_play_pause() in simulation_engine.py) or a project is
        switched. Shutting down a shared pool from a per-instance cleanup
        would break any OTHER cache manager / recording pass still using
        it. io_worker is shut down exactly once, at real app close (see
        GeneralViewTab.closeEvent()).

        Still worth waiting here, though: block briefly until any
        save_frame_cache() writes THIS instance queued are actually on
        disk, so a fresh cache manager for the next pass never races a
        stale in-flight write to the same cache_dir.
        """
        try:
            io_worker.wait_all()
            print("✅ Bidirectional cache manager cleaned up")
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")

# ============================================================================
# UPDATED PENETRATION DETECTION FUNCTION - OPTIMIZED WITH VERTEX TRANSFORM
# ============================================================================

def detect_penetrations_bidirectional(wall_points, 
                                         R, t,
                                         obb_center_local,
                                         obb_half_extents,
                                         frame_idx: int = 0, 
                                         initial_beam_points=None,
                                         x_translation:  float=0.45,
                                         thickness=0.01, sd_threshold=0.03,
                                         cache_manager=None,
                                         obb_R_local=None):
    """
    Perform penetration detection on ALL FIVE sides (LEFT, RIGHT, FRONT, BACK, and TOP).
    
    OPTIMIZED APPROACH:
    1. Create BASE MESHES ONCE in LOCAL coordinates (initial_beam_points centered at origin)
    2. Store base mesh VERTICES and TRIANGLES, and build each direction's
       RaycastingScene ONCE from that LOCAL geometry (never rebuilt again)
    3. Each frame:
       a. Inverse-transform the (OBB-filtered) query points into LOCAL
          space using R, t — one small matmul, shared across all 5
          directions since they use the same rigid transform this frame
       b. Query signed distance against the pre-built static scenes
    4. Filter wall points using OBB
    5. Detect penetrations with signed distance
    
    Args:
        wall_points: wall points in WORLD coordinates
        R: (3x3) rotation matrix from local to world
        t: (3,) translation vector from local to world
        obb_center_local: OBB center in local coordinates
        obb_half_extents: half-extents of OBB box
        frame_idx: current frame number
        initial_beam_points: initial beam points (for mesh creation on first call)
        thickness: mesh thickness
        sd_threshold: signed distance threshold
        cache_manager: cache manager instance
    
    Returns:
        tuple: (left_clash_pts, left_idx, right_clash_pts, right_idx, 
                front_clash_pts, front_idx, back_clash_pts, back_idx,
                top_clash_pts, top_idx)
    """
    if cache_manager is None:
        cache_manager = AcceleratedBidirectionalPenetrationCache()

    # ===== DEBUG: OBB / probe-length consistency check =====
    # obb_half_extents is frozen at _initialize_beam_geometry() time (scene
    # load). If x_translation has since changed (restart_with_new_x_translation)
    # without re-running _initialize_beam_geometry(), the wall-point candidate
    # WINDOW (obb_half_extents) goes stale relative to the probe mesh LENGTH
    # (rebuilt fresh below, from the CURRENT x_translation param). This print
    # makes that divergence impossible to miss.
    if obb_half_extents is not None:
        max_obb_extent = float(np.max(obb_half_extents))
        if abs(max_obb_extent - x_translation) > 1e-6 and max_obb_extent < x_translation:
            print(f"   ⚠️⚠️⚠️ OBB/PROBE MISMATCH: obb_half_extents max="
                  f"{max_obb_extent:.4f} but probe length={x_translation:.4f} — "
                  f"OBB window is SMALLER than probe reach. Wall candidates "
                  f"will be clipped before they ever reach the probe. Call "
                  f"_initialize_beam_geometry() again after changing x_translation.")

    # ===== STEP 1: CREATE BASE MESHES IN LOCAL COORDINATES (ONCE) =====
    if not cache_manager.meshes_initialized:
        if initial_beam_points is None:
            print("❌ Error: initial_beam_points required for first frame")
            return (np.array([]), np.array([]), np.array([]), np.array([]),
                    np.array([]), np.array([]), np.array([]), np.array([]),
                    np.array([]), np.array([]))
        
        # Center beam points at origin (LOCAL coordinates)
        beam_points_local = initial_beam_points
        
        print(f"\n🎬 Frame {frame_idx}: First initialization - creating base meshes...")
        mesh_left_local, mesh_right_local, mesh_front_local, mesh_back_local, mesh_top_local = (
            cache_manager.create_base_meshes_bidirectional_local(
                beam_points_local,
                length=x_translation,
                thickness=thickness
            )
        )
        
        if (mesh_left_local is None or mesh_right_local is None or mesh_front_local is None or
            mesh_back_local is None or mesh_top_local is None):
            print("❌ Failed to create base meshes")
            return (np.array([]), np.array([]), np.array([]), np.array([]),
                    np.array([]), np.array([]), np.array([]), np.array([]),
                    np.array([]), np.array([]))

    # ===== STEP 2: TRANSFORM BASE LOCAL MESHES TO WORLD, REBUILD BVH =====
    # Per-request change: base_vertices_*_local (captured once at first-time
    # mesh creation in STEP 1) are rotated+translated to world space THIS
    # frame via create_transformed_mesh_and_scene() — v_world = R @ v_local
    # + t — and a fresh RaycastingScene is built from the result, once per
    # direction, every frame. This replaces the static-local-scene +
    # inverse-query-transform path (get_static_scene() /
    # transform_points_world_to_local()) that used to run here.
    #
    # Mathematically this returns IDENTICAL clash points/indices to the
    # static-local approach: signed distance is invariant under a rigid
    # transform (sd_world(T(x)) = sd_local(x) for T(x) = Rx + t), so
    # querying wall_inside directly against this world-space scene gives
    # the exact same sd values, at every point, as querying the
    # inverse-transformed points against the local-space scene did.
    #
    # ⚠️ Cost: this rebuilds 5 BVHs (one per direction) EVERY frame instead
    # of reusing 5 that were built once — this is the per-frame cost the
    # static-scene refactor (_build_static_scenes(), see its docstring) was
    # written to eliminate. Restored here deliberately, on request.
    mesh_left,  scene_left  = cache_manager.create_transformed_mesh_and_scene("LEFT",  R, t, frame_idx)
    mesh_right, scene_right = cache_manager.create_transformed_mesh_and_scene("RIGHT", R, t, frame_idx)
    mesh_front, scene_front = cache_manager.create_transformed_mesh_and_scene("FRONT", R, t, frame_idx)
    mesh_back,  scene_back  = cache_manager.create_transformed_mesh_and_scene("BACK",  R, t, frame_idx)
    mesh_top,   scene_top   = cache_manager.create_transformed_mesh_and_scene("TOP",   R, t, frame_idx)

    if (scene_left is None or scene_right is None or scene_front is None or
        scene_back is None or scene_top is None):
        print("❌ World-space raycasting scenes unavailable (base meshes not initialized yet, "
              "or create_transformed_mesh_and_scene() failed)")
        return (np.array([]), np.array([]), np.array([]), np.array([]),
                np.array([]), np.array([]), np.array([]), np.array([]),
                np.array([]), np.array([]))

    print(f"   ✓ Rebuilt world-space raycasting scenes for LEFT, RIGHT, FRONT, BACK, TOP")

    # ===== STEP 3: FILTER WALL POINTS USING OBB =====
    wall_inside, obb_mask = filter_wall_points_by_obb(
        wall_points=wall_points,
        R=R,
        t=t,
        obb_center_initial=obb_center_local,
        obb_half_extents=obb_half_extents,
        obb_R_local=obb_R_local,
    )
    
    global_inside_idx = np.where(obb_mask)[0]
    
    if len(wall_inside) == 0:
        print(f"   ⓘ Frame {frame_idx}: No wall points inside OBB")
        return (np.array([]), np.array([]), np.array([]), np.array([]),
                np.array([]), np.array([]), np.array([]), np.array([]),
                np.array([]), np.array([]))

    print(f"   ✓ {len(wall_inside):,} wall points inside OBB")

    # ===== STEP 4: QUERY DIRECTLY IN WORLD SPACE — NO INVERSE TRANSFORM =====
    # wall_inside is already in world coordinates (it came straight out of
    # filter_wall_points_by_obb() above, which operates on world-space
    # wall_points). No transform_points_world_to_local() call — the scenes
    # built in STEP 2 are themselves in world space now.
    print(f"   🔄 Querying wall points directly in world space (no local-space transform)...")
    query = o3d.core.Tensor(wall_inside, dtype=o3d.core.Dtype.Float32)


    # ===== STEP 4: DETECT PENETRATIONS =====
    print(f"   🔍 Detecting penetrations...")
    
    left_clash_points, left_indices = _detect_single_direction(
        scene_left, wall_inside, global_inside_idx, sd_threshold, "LEFT",
        frame_idx, 0, cache_manager, query
    )

    right_clash_points, right_indices = _detect_single_direction(
        scene_right, wall_inside, global_inside_idx, sd_threshold, "RIGHT",
        frame_idx, 0, cache_manager, query
    )

    front_clash_points, front_indices = _detect_single_direction(
        scene_front, wall_inside, global_inside_idx, sd_threshold, "FRONT",
        frame_idx, 0, cache_manager, query
    )

    back_clash_points, back_indices = _detect_single_direction(
        scene_back, wall_inside, global_inside_idx, sd_threshold, "BACK",
        frame_idx, 0, cache_manager, query
    )

    top_clash_points, top_indices = _detect_single_direction(
        scene_top, wall_inside, global_inside_idx, sd_threshold, "TOP",
        frame_idx, 0, cache_manager, query
    )

    # Ensure all arrays have proper shape (N, 3) even if empty
    if len(left_clash_points) == 0:
        left_clash_points = np.empty((0, 3), dtype=np.float64)
    if len(right_clash_points) == 0:
        right_clash_points = np.empty((0, 3), dtype=np.float64)
    if len(front_clash_points) == 0:
        front_clash_points = np.empty((0, 3), dtype=np.float64)
    if len(back_clash_points) == 0:
        back_clash_points = np.empty((0, 3), dtype=np.float64)
    if len(top_clash_points) == 0:
        top_clash_points = np.empty((0, 3), dtype=np.float64)

    # Summary
    total_clashes = (len(left_clash_points) + len(right_clash_points) + 
                     len(front_clash_points) + len(back_clash_points) + len(top_clash_points))
    print(f"   ✓ Frame {frame_idx} complete: {total_clashes} total penetrations")

    return (left_clash_points, left_indices, right_clash_points, right_indices, 
            front_clash_points, front_indices, back_clash_points, back_indices,
            top_clash_points, top_indices)

def _detect_single_direction(scene, wall_points, global_indices, sd_threshold, 
                                direction, frame_idx, x_translation, cache_manager, query=None):
    """
    Helper function to detect penetrations for one direction using signed distance.
    
    Args:
        scene: WORLD-space o3d raycasting scene for THIS frame, rebuilt every
               call by create_transformed_mesh_and_scene() (per-request
               change — see detect_penetrations_bidirectional() STEP 2).
        wall_points: filtered wall points (OBB-filtered, WORLD coords) — used
               for indexing/output AND as the query itself now, since both
               `scene` and `wall_points`/`query` live in world space this
               frame (no local-space inverse transform anymore).
        global_indices: indices of wall_points in original wall_points array
        sd_threshold: signed distance threshold
        direction: "LEFT", "RIGHT", "FRONT", "BACK", or "TOP"
        frame_idx: current frame number
        x_translation: translation value (for caching)
        cache_manager: cache manager instance
    
    Returns:
        (clash_points, clash_indices)
    """
    try:
        if len(wall_points) == 0:
            cache_manager.save_frame_cache(frame_idx, direction, x_translation, np.array([]), np.array([]))
            return np.array([]), np.array([])


        if query is None:
            # scene is a WORLD-space scene, rebuilt fresh every frame from
            # this frame's (R, t) — see detect_penetrations_bidirectional()
            # STEP 2. This function has no R/t here to build a query itself,
            # so the hot path always supplies `query` = wall_inside directly
            # (already world-space, no transform needed). Hitting this
            # branch means a caller changed that contract.
            raise ValueError(
                f"_detect_single_direction({direction}) called without a "
                "`query` — scene is world-space and rebuilt per-frame, "
                "the caller must supply the (already world-space) query points."
            )
        
        sd = scene.compute_signed_distance(query).numpy().ravel()
        
        # Points inside mesh: -sd_threshold < sd < 0
        inside_idx_local = np.where((-sd_threshold < sd) & (sd < 0))[0]
        
        if inside_idx_local.size == 0:
            cache_manager.save_frame_cache(frame_idx, direction, x_translation, np.array([]), np.array([]))
            print(f"      ✓ {direction}: 0 penetrations")
            return np.array([]), np.array([])
        
        # Map back to global indices
        inside_idx_global = global_indices[inside_idx_local]
        sd_values = sd[inside_idx_local]
        clash_points = wall_points[inside_idx_local]
        
        cache_manager.save_frame_cache(frame_idx, direction, x_translation, inside_idx_global, sd_values)
        
        print(f"      🔴 {direction}: {len(clash_points)} penetrations")
        return clash_points, inside_idx_global
        
    except Exception as e:
        print(f"      ❌ Error in {direction} detection: {e}")
        cache_manager.save_frame_cache(frame_idx, direction, x_translation, np.array([]), np.array([]))
        return np.array([]), np.array([])