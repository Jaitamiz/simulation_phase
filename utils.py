import numpy as np
import string
import pyvista as pv
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree



def savgol_smooth(points, window_length=21, polyorder=3):
    if window_length >= len(points):
        window_length = len(points) - 1
        if window_length % 2 == 0:
            window_length -= 1
    return savgol_filter(points, window_length=window_length, polyorder=polyorder, axis=0)

def create_rotation_matrix(roll, pitch, yaw):
    """Create rotation matrix from roll, pitch, yaw (in degrees) using scipy"""
    rotation = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    return rotation.as_matrix()

def transform_points(points, rotation_matrix, translation):
    """Apply rotation and translation to points"""
    return (rotation_matrix @ points.T).T + translation

def create_labeled_points_grid(size=10, step=1.0, height=0.0):
    """Create grid points with labels for clash identification"""
    points = []
    labels = []
    
    char_labels = string.ascii_uppercase
    
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            x = i * step
            y = j * step
            z = height
            label = f"{char_labels[j + size]}{i + size + 1}"  # A1, B2, etc.
            
            points.append([x, y, z])
            labels.append((label, (x, y, z)))
    
    return np.array(points), labels

def find_nearest_grid_points(grid_points, target_point, count=4):
    """Find nearest grid points to target point that form a rectangle"""
    if target_point is None:
        return [], []
    
    # Get all distances
    distances = np.linalg.norm(grid_points - target_point, axis=1)
    
    # Get nearest point
    nearest_index = np.argmin(distances)
    nearest_point = grid_points[nearest_index]
    
    # Find the 3 other points that form a rectangle with the nearest point
    # Assuming grid is regular (equally spaced in X and Y)
    
    # Get unique X and Y coordinates from grid
    unique_x = np.unique(grid_points[:, 0])
    unique_y = np.unique(grid_points[:, 1])
    
    # Find the grid cell that contains the nearest point
    x_idx = np.argmin(np.abs(unique_x - nearest_point[0]))
    y_idx = np.argmin(np.abs(unique_y - nearest_point[1]))
    
    # Get the 4 points that form a rectangle around the target
    rectangle_points = []
    rectangle_indices = []
    
    # Try to get points in a 2x2 pattern
    for dx in [0, 1]:
        for dy in [0, 1]:
            if x_idx + dx < len(unique_x) and y_idx + dy < len(unique_y):
                # Find point with these coordinates
                mask = (np.abs(grid_points[:, 0] - unique_x[x_idx + dx]) < 0.001) & \
                       (np.abs(grid_points[:, 1] - unique_y[y_idx + dy]) < 0.001)
                if np.any(mask):
                    idx = np.where(mask)[0][0]
                    rectangle_points.append(grid_points[idx])
                    rectangle_indices.append(idx)
    
    # If we don't have 4 points, fall back to nearest points
    if len(rectangle_points) < 4:
        nearest_indices = np.argsort(distances)[:count]
        return nearest_indices, grid_points[nearest_indices]
    
    return rectangle_indices[:4], np.array(rectangle_points[:4])

def sort_rectangle_points_robust(points):
    """
    Sort 4 points to form rectangle: 
    A1 (bottom-left) → A2 (bottom-right) → B2 (top-right) → B1 (top-left)
    """
    if len(points) != 4:
        return points
    
    # Sort by Y (vertical), then by X (horizontal)
    sorted_by_y = points[np.argsort(points[:, 1])]  # Sort by Y ascending
    
    # Bottom row (smaller Y)
    bottom_row = sorted_by_y[:2]
    bottom_row = bottom_row[np.argsort(bottom_row[:, 0])]  # Sort by X
    
    # Top row (larger Y)
    top_row = sorted_by_y[2:]
    top_row = top_row[np.argsort(top_row[:, 0])]  # Sort by X
    
    # Return in rectangle order: bottom-left, bottom-right, top-right, top-left
    return np.vstack([bottom_row[0], bottom_row[1], top_row[1], top_row[0]])

def create_grid_box_lines(box_points):
    """Create line segments for grid box visualization"""
    if len(box_points) != 4:
        return []
    
    # Sort points to create proper rectangular box
    centroid = np.mean(box_points, axis=0)
    vectors = box_points - centroid
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    sorted_indices = np.argsort(angles)
    sorted_points = box_points[sorted_indices]
    
    # Create box lines connecting the 4 points in order
    lines = []
    for i in range(4):
        lines.append(pv.Line(sorted_points[i], sorted_points[(i + 1) % 4]))
    
    return lines

def align_grid_to_wall(grid_points, wall_points):
    """
    Align grid BELOW wall with automatic rotation based on wall orientation
    
    Strategy:
    1. Detect wall's principal direction using PCA
    2. Detect grid's principal direction using PCA
    3. Rotate grid to align perpendicular to wall
    4. Align XY centroids (horizontal alignment - centered)
    5. Position grid BELOW wall - wall BOTTOM touches grid TOP
    
    Parameters:
    -----------
    grid_points : array
        Grid coordinates
    wall_points : array
        Wall point cloud coordinates
    """
    print(f"\n{'='*70}")
    print(f"🔄 ALIGNING GRID BELOW WALL (AUTOMATIC ROTATION)")
    print(f"{'='*70}")
    
    def compute_pca_orientation(points):
        """Compute PCA and return the dominant horizontal direction"""
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        
        # Compute covariance matrix
        cov_matrix = np.cov(centered.T)
        
        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]
        
        # The first eigenvector (largest eigenvalue) is the principal direction
        # But we want the dominant horizontal direction (ignoring Z)
        # Get the eigenvector with smallest Z component (most horizontal)
        z_components = np.abs(eigvecs[2, :])
        horizontal_idx = np.argmin(z_components)
        
        # Get horizontal direction vector (project to XY plane)
        direction = eigvecs[:, horizontal_idx]
        direction[2] = 0  # Force horizontal
        direction = direction / np.linalg.norm(direction)
        
        return direction, centroid
    
    # Step 1: Compute orientations
    print("📐 Computing orientations...")
    wall_direction, wall_centroid = compute_pca_orientation(wall_points)
    grid_direction, grid_centroid = compute_pca_orientation(grid_points)
    
    print(f"   Wall principal direction: {wall_direction}")
    print(f"   Grid principal direction: {grid_direction}")
    
    # Step 2: Calculate rotation angle to make grid perpendicular to wall
    # Dot product gives cosine of angle between directions
    cos_angle = np.dot(wall_direction[:2], grid_direction[:2])
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    current_angle = np.degrees(np.arccos(cos_angle))
    
    # We want grid to be perpendicular (90°) to wall
    target_angle_diff = 90.0
    rotation_angle = target_angle_diff - current_angle
    
    print(f"   Current angle between grid and wall: {current_angle:.1f}°")
    print(f"   Required rotation: {rotation_angle:.1f}°")
    
    # Step 3: Apply rotation
    angle = np.radians(rotation_angle)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    
    # Rotate grid around its centroid
    grid_centered = grid_points - grid_centroid
    rotated_grid = grid_centered @ rotation_matrix.T
    rotated_grid = rotated_grid + grid_centroid
    
    print(f"✅ Step 3: Rotated grid by {rotation_angle:.1f}° (perpendicular to wall)")
    
    # Step 4: Align XY centroids (horizontal alignment - centered)
    grid_centroid_xy = np.mean(rotated_grid[:, :2], axis=0)
    wall_centroid_xy = wall_centroid[:2]
    xy_translation = wall_centroid_xy - grid_centroid_xy
    
    rotated_grid[:, 0] += xy_translation[0]
    rotated_grid[:, 1] += xy_translation[1]
    
    print(f"✅ Step 4: Aligned XY centroids (horizontally centered)")
    print(f"   Translation: X={xy_translation[0]:.3f}m, Y={xy_translation[1]:.3f}m")
    
    # Step 5: Position grid BELOW wall - wall bottom touches grid top
    # Get wall bounding box (min Z = bottom of wall)
    wall_min_z = np.min(wall_points[:, 2])  # Bottom of wall
    
    # Get grid bounding box after XY alignment
    grid_max_z = np.max(rotated_grid[:, 2])  # Top of grid
    
    # Calculate Z offset so grid top touches wall bottom
    z_offset = wall_min_z - grid_max_z
    
    # Apply Z translation
    rotated_grid[:, 2] += z_offset
    
    # Verify alignment
    grid_top_after = np.max(rotated_grid[:, 2])
    wall_bottom = wall_min_z
    gap = abs(grid_top_after - wall_bottom)
    
    print(f"✅ Step 5: Positioned grid BELOW wall (TOUCHING)")
    print(f"   Wall bottom Z:        {wall_bottom:.3f}m")
    print(f"   Grid top Z:           {grid_top_after:.3f}m")
    print(f"   Gap:                  {gap:.6f}m (should be ~0)")
    print(f"   Z offset applied:     {z_offset:.3f}m")
    
    # Final translation vector
    translation = np.array([xy_translation[0], xy_translation[1], z_offset])
    
    # Verify final orientation
    final_grid_direction, _ = compute_pca_orientation(rotated_grid)
    final_angle = np.degrees(np.arccos(np.clip(
        np.dot(wall_direction[:2], final_grid_direction[:2]), -1.0, 1.0
    )))
    
    print(f"{'─'*70}")
    print(f"📍 FINAL ALIGNMENT SUMMARY:")
    print(f"   ✅ Rotation: {rotation_angle:.1f}° applied")
    print(f"   ✅ Final angle between grid and wall: {final_angle:.1f}°")
    print(f"   ✅ XY Position: Centered under wall")
    print(f"   ✅ Z Position: Grid BELOW wall, touching (no gap)")
    print(f"   Translation: X={translation[0]:.3f}m, Y={translation[1]:.3f}m, Z={translation[2]:.3f}m")
    print(f"{'='*70}\n")
    
    return rotated_grid, translation

def bounding_box_object(object_points, length):
    """
    Compute OBB in car-local coordinates relative to the beam centroid,
    matching get_translation_rotation()'s local frame.
    """
    # Use same local origin as in get_translation_rotation()
   
  
    local_origin = object_points.mean(axis=0)

    car_min = object_points.min(axis=0)
    car_max = object_points.max(axis=0)

    # Extend width (along X or Y depending on your coordinate convention)
    # x_translation controls OBB extension 
    width_extension = max(0.0, float(length))
    car_min[0] -= width_extension
    car_max[0] += width_extension

    car_min[2] -= width_extension
    car_max[2] += width_extension

    obb_center_local = (car_min + car_max) * 0.5
    obb_half_extents = (car_max - car_min) * 0.5

    bbox_local = pv.Box(bounds=(
        car_min[0], car_max[0],
        car_min[1], car_max[1],
        car_min[2], car_max[2]
    ))
    bbox_local_points = np.asarray(bbox_local.points, dtype=np.float64)
    bbox_local_points = np.asarray(bbox_local_points, dtype=np.float64)
    bbox_local_points = np.ascontiguousarray(bbox_local_points)
    bbox_local_points = bbox_local_points.reshape(-1, 3)

    return obb_center_local, obb_half_extents, local_origin, bbox_local_points

#geometry utils
def rotation_matrix(roll, pitch, yaw):
    roll  = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw   = np.deg2rad(yaw)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])

    Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx

# ============================================================================
# FILTER CLASH POINTS BY TRANSFORMED OBB
# ============================================================================
def compute_actual_obb_half_extents(beam_points, obb_R_local=None):
    """
    Compute the ACTUAL, TIGHT OBB half-extents from beam geometry, with NO
    padding/extension — used to decide which clash points fall INSIDE the
    beam's own body (self-intersection artifacts against the probe mesh
    the beam itself generates), so this must be the true tight fit, not
    the probe/detection-window's PADDED box (self.obb_half_extents, which
    is deliberately extended by x_translation for the UNRELATED purpose
    of windowing candidate wall points — see simulation_engine.py's
    _compute_pca_oriented_obb()).

    FIX: this used to always take min/max per LOCAL X/Y/Z axis — i.e. an
    axis-aligned box in the CAR-LOCAL frame, not the car's own PCA
    length/width/height axes. Unless the car's true body happens to be
    aligned with local X/Y/Z (it generally isn't — calibration only fixes
    the TRAJECTORY frame's own rotation, not the car's orientation within
    it), an axis-aligned box has to be sized to cover the car's DIAGONAL
    just to contain every point. That made it silently much larger than
    the car's true tight footprint along the axes that actually matter —
    large enough that real clash points sitting just outside the true
    body (exactly the probe-mesh penetrations this pipeline exists to
    detect) still tested as "inside," and got discarded as if they were
    self-intersection artifacts. Confirmed in practice: LEFT dropped from
    15,324 -> 148 detections, RIGHT from 21,039 -> 1, almost entirely
    from this mismatch. Projecting into obb_R_local (the car's own
    committed PCA basis, columns = [length_vector, width_vector,
    height_vector]) before taking min/max gives the TRUE tight fit
    instead — the same fix, and the same reasoning, as detector.py's
    _pca_tight_obb().

    Args:
        beam_points: (N, 3) beam point cloud, in CAR-LOCAL coordinates
        obb_R_local: (3x3) orthonormal basis whose columns are the car's
            own committed length/width/height axes (same value as
            simulation_engine.py's self.obb_R_local / detector.py's
            filter_wall_points_by_obb(obb_R_local=...)), or None for
            axis-aligned (identity) — kept as the default purely so any
            caller that hasn't been updated to pass this yet still gets
            the OLD (loose) behavior rather than an error, not because
            identity is ever the CORRECT choice once a PCA orientation
            has been committed.

    Returns:
        obb_half_extents: (3,) half-extents measured ALONG obb_R_local's
            own axes (or plain local X/Y/Z if obb_R_local is None) —
            pairs with filter_clash_points_by_transformed_obb() below,
            which must use the SAME obb_R_local for its inside/outside
            test to be meaningful.
    """
    if obb_R_local is None:
        obb_R_local = np.eye(3)

    # Project into the box's own (possibly PCA-oriented) axes BEFORE
    # taking min/max — this is what makes the extent tight along the
    # car's true length/width/height instead of its local-X/Y/Z diagonal.
    proj = beam_points @ obb_R_local

    min_coords = proj.min(axis=0)
    max_coords = proj.max(axis=0)

    # Compute half-extents (distance from center to edge), ALONG the
    # box's own axes.
    obb_half_extents = (max_coords - min_coords) / 2.0

    return obb_half_extents

def filter_clash_points_by_transformed_obb(
    clash_points,
    R,
    t,
    beam_points,
    obb_center_local=None,
    obb_R_local=None,
):
    """
    Filter out clash points that are INSIDE the transformed OBB bounding box.

    This function:
    1. Computes the ACTUAL, TIGHT OBB half-extents from beam geometry
       (unpadded, PCA-oriented if obb_R_local is given — see
       compute_actual_obb_half_extents())
    2. Composes full_R = R @ obb_R_local — the trajectory frame's own
       rotation THEN the box's fixed local orientation — and transforms
       the OBB center to world coordinates using R and t
    3. Checks which clash points are inside the transformed, oriented OBB
    4. Returns only the clash points that are OUTSIDE the box

    Args:
        clash_points: (N, 3) clash points in WORLD coordinates
        R: (3x3) rotation matrix from local to world (this frame's
            trajectory rotation)
        t: (3,) translation vector from local to world
        beam_points: (M, 3) beam point cloud in CAR-LOCAL coordinates —
            same array compute_frame_data() was called with, i.e. the
            car's own body, NOT the (longer) probe/mesh geometry it
            generates.
        obb_center_local: (3,) OBB center in CAR-LOCAL coordinates
                         If None, computed as mean of beam_points
        obb_R_local: (3x3) the car's own committed PCA length/width/
            height basis, or None for axis-aligned. Composed with R
            exactly like detector.py's filter_wall_points_by_obb() does,
            so the box tested against here is the SAME tight, correctly
            oriented fit — not a loose axis-aligned box sized to the
            car's diagonal (see compute_actual_obb_half_extents()'s
            docstring for what that mismatch actually did in practice).

    Returns:
        filtered_clash_points: (K, 3) clash points OUTSIDE the OBB
        outside_mask: (N,) boolean mask where True = outside OBB
    """
    if len(clash_points) == 0:
        return np.empty((0, 3), dtype=np.float64), np.array([], dtype=bool)

    if obb_R_local is None:
        obb_R_local = np.eye(3)

    # ===== COMPUTE ACTUAL, TIGHT OBB HALF-EXTENTS (PCA-oriented, unpadded) ===
    actual_obb_half_extents = compute_actual_obb_half_extents(beam_points, obb_R_local)

    # ===== COMPUTE OBB CENTER IN LOCAL COORDINATES =====
    if obb_center_local is None:
        obb_center_local = beam_points.mean(axis=0)

    # ===== TRANSFORM OBB CENTER TO WORLD COORDINATES =====
    # obb_center_local is still expressed in the CAR-LOCAL frame (not the
    # box's own rotated frame), so this uses R alone — same convention
    # detector.py's filter_wall_points_by_obb() uses for its obb_center_world.
    obb_center_world = (R @ obb_center_local) + t

    # ===== TRANSFORM CLASH POINTS INTO THE BOX'S OWN LOCAL FRAME =====
    # full_R takes a point from the box's own (possibly PCA-oriented)
    # local frame all the way to world space — composing the trajectory
    # frame's rotation with the box's fixed local orientation, the same
    # two-step composition used everywhere else an OBB corner or filter
    # test is built in this codebase (see detector.py's
    # filter_wall_points_by_obb() and _pca_tight_obb()).
    full_R = R @ obb_R_local
    clash_local = (full_R.T @ (clash_points - obb_center_world).T).T

    # ===== CHECK WHICH CLASH POINTS ARE INSIDE THE TIGHT OBB BOX =====
    # A point is inside the box if:
    # |x_local| <= half_x AND |y_local| <= half_y AND |z_local| <= half_z
    # — now measured along the box's OWN (possibly PCA-oriented) axes.
    inside_mask = (
        (np.abs(clash_local[:, 0]) <= actual_obb_half_extents[0]) &
        (np.abs(clash_local[:, 1]) <= actual_obb_half_extents[1]) &
        (np.abs(clash_local[:, 2]) <= actual_obb_half_extents[2])
    )

    # ===== KEEP ONLY POINTS OUTSIDE THE BOX =====
    outside_mask = ~inside_mask
    filtered_clash_points = clash_points[outside_mask]

    num_filtered = np.sum(inside_mask)
    if num_filtered > 0:
        print(f"         🗑️  Removed {num_filtered} clash points inside beam OBB")

    return filtered_clash_points, outside_mask

def compute_minimum_clash_distance(clash_points, model_points_world):
    """
    Compute minimum distance from clash points to original model surface.
    
    Args:
        clash_points: (N, 3) array of clash points in world coordinates
        model_points_world: (M, 3) array of model points in world coordinates
    
    Returns:
        min_distance: float, minimum distance
        min_clash_point: (3,) closest clash point
        min_model_point: (3,) nearest model point to closest clash
        all_distances: (N,) array of distances for all clash points
    """
    if len(clash_points) == 0:
        return None, None, None, None
    
    # Build KD-Tree for fast search
    tree = cKDTree(model_points_world)
    
    # Find nearest model point for each clash
    distances, indices = tree.query(clash_points)
    
    # Find global minimum
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]
    min_clash_point = clash_points[min_idx]
    min_model_point = model_points_world[indices[min_idx]]
    
    return min_distance, min_clash_point, min_model_point, distances
