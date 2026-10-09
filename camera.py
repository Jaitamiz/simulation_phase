# ============================================================================
# CAMERA HELPERS
# ============================================================================
#
#   get_integrated_forward_vector() — direction of travel from the first ~2 m
#                                      of the trajectory (PCA vs. sliding-window
#                                      linearity, whichever best matches the
#                                      overall 2 m path).
#   estimate_global_rotation()      — ONE-OFF: 3x3 rotation that re-expresses
#                                      the recorded roll/pitch/yaw orientation
#                                      in the scene's world axes (Z up, X along
#                                      the direction of travel).
#   camera_follow()                 — PER-FRAME: places the plotter's chase
#                                      camera for a given orientation R and
#                                      position t.
#
# The first two are pure NumPy (safe from any thread). camera_follow() sets
# `plotter.camera_position`, so it is MAIN-THREAD ONLY — it is called from
# frame_updater.apply_frame_visualization(), never from the worker thread.
# ============================================================================
import numpy as np

from simulation.utils import rotation_matrix


def get_integrated_forward_vector(poses, max_dist=2.0, window_size=20):
    """
    Extracts the optimal forward vector by comparing a PCA on the early 2m segment
    against a sliding window linearity check within that same segment.
    """
    # 1. Isolate the early trajectory (up to max_dist)
    end_idx = len(poses) - 1
    for i in range(1, len(poses)):
        if np.linalg.norm(poses[i] - poses[0]) >= max_dist:
            end_idx = i
            break

    segment = poses[:end_idx + 1]

    # Handle edge cases: extremely short trajectories
    if len(segment) < 2:
        return np.array([1.0, 0.0, 0.0])

    # Dynamically adjust window size if the segment has very few points
    if len(segment) < window_size:
        window_size = max(2, len(segment) // 2)

    # 2. Candidate A: PCA on the entire early segment
    centroid_seg = np.mean(segment, axis=0)
    cov_seg = np.cov(segment - centroid_seg, rowvar=False)
    eig_vals_seg, eig_vecs_seg = np.linalg.eigh(cov_seg)
    v_pca = eig_vecs_seg[:, np.argmax(eig_vals_seg)]

    # Ensure forward direction
    macro_direction = segment[-1] - segment[0]
    macro_norm = np.linalg.norm(macro_direction)
    v_baseline = macro_direction / macro_norm if macro_norm > 1e-6 else np.array([1.0, 0.0, 0.0])

    if np.dot(v_pca, v_baseline) < 0:
        v_pca = -v_pca
    v_pca = v_pca / np.linalg.norm(v_pca)

    # 3. Candidate B: Sliding Window Linearity within the segment
    best_linearity = -1
    v_window = None

    for i in range(len(segment) - window_size + 1):
        sub_seg = segment[i: i + window_size]
        centroid_sub = np.mean(sub_seg, axis=0)
        cov_sub = np.cov(sub_seg - centroid_sub, rowvar=False)

        eig_vals_sub, eig_vecs_sub = np.linalg.eigh(cov_sub)

        # Sort eigenvalues descending
        idx = eig_vals_sub.argsort()[::-1]
        eig_vals_sub = eig_vals_sub[idx]
        eig_vecs_sub = eig_vecs_sub[:, idx]

        total_variance = np.sum(eig_vals_sub)
        if total_variance == 0:
            continue

        linearity = eig_vals_sub[0] / total_variance

        if linearity > best_linearity:
            best_linearity = linearity
            v_window = eig_vecs_sub[:, 0]
            if np.dot(v_window, sub_seg[-1] - sub_seg[0]) < 0:
                v_window = -v_window

    if v_window is not None:
        v_window = v_window / np.linalg.norm(v_window)
    else:
        v_window = v_pca

    # 4. Compare both candidates against the macro baseline direction
    score_pca = np.dot(v_pca, v_baseline)
    score_window = np.dot(v_window, v_baseline)

    # Select the vector that most closely parallels the overall 2m path
    if score_window >= score_pca:
        print(f"Selected Window Vector (Score: {score_window:.4f} vs PCA: {score_pca:.4f})")
        return v_window
    else:
        print(f"Selected PCA Segment Vector (Score: {score_pca:.4f} vs Window: {score_window:.4f})")
        return v_pca


def _orthonormal_basis(forward, up):
    """
    Right-handed orthonormal basis (columns X=forward, Y=up x X, Z=X x Y).
    Falls back to an arbitrary perpendicular if `forward` is parallel to `up`
    (otherwise the cross product is ~0 and the normalisation yields NaNs).
    """
    X = forward / np.linalg.norm(forward)
    Y = np.cross(up, X)
    n = np.linalg.norm(Y)
    if n < 1e-9:
        alt = np.array([0.0, 1.0, 0.0]) if abs(X[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        Y = np.cross(alt, X)
        n = np.linalg.norm(Y)
    Y = Y / n
    Z = np.cross(X, Y)
    return np.column_stack([X, Y, Z])


def estimate_global_rotation(poses, rolls, pitches, yaws, max_dist=2.0):
    """
    Creates a rotation matrix by explicitly aligning the vehicle's forward axis
    using the integrated PCA and Sliding Window comparison.

    Returns R_global such that, for frame i,
        R_global @ rotation_matrix(roll[i], pitch[i], yaw[i])
    is the car's orientation expressed in the scene's world axes. At frame 0
    the car's local +X maps exactly onto the trajectory's direction of travel.
    """
    print("Aligning local basis using integrated early-trajectory analysis...")

    # 1. Basis of the NEW world: X = path direction, Z ~ world up
    v_new = get_integrated_forward_vector(poses, max_dist=max_dist)
    world_up = np.array([0.0, 0.0, 1.0])  # Dataset 3 is horizontal (Z is up)
    M_new = _orthonormal_basis(v_new, world_up)

    # 2. Basis of the OLD world: the car's own local axes at frame 0
    R_local_0 = rotation_matrix(rolls[0], pitches[0], yaws[0])
    local_forward = np.array([1.0, 0.0, 0.0])  # Standard: X is Forward
    local_up = np.array([0.0, 0.0, 1.0])       # Standard: Z is Up
    M_old = _orthonormal_basis(R_local_0 @ local_forward, R_local_0 @ local_up)

    # 3. The global rotation maps the Old basis exactly onto the New basis
    #    (M_new @ M_old.T — NOT M @ M.T, which is always the identity).
    R_global = M_new @ M_old.T
    return R_global


def camera_follow(plotter, R, t, distance=2.0, height=0.0):
    """
    Calculates the exact eye, target, and up-vector for the chase camera,
    placing the camera on the negative X-axis of the car's local frame
    (behind the car), and applies it to `plotter`.

    MAIN-THREAD ONLY (mutates the plotter's camera).

    plotter  : pyvista plotter whose camera is moved.
    R        : (3, 3) car orientation. For a scene-aligned camera pass
               `R_global @ rotation_matrix(roll[i], pitch[i], yaw[i])`.
    t        : (3,) car position (also the look-at target).
    distance : metres BEHIND the car (positive = behind; applied along -X).
    height   : metres ABOVE the car (along +Z of the car's local frame).
    """
    camera_distance = -distance
    camera_height = height
    camera_up = np.array([0.0, 0.0, 1.0])

    # Local [X, Y, Z]: behind on X, centred on Y, elevated on Z.
    camera_offset_local = np.array([camera_distance, 0.0, camera_height])

    camera_offset_world = R @ camera_offset_local

    eye = t + camera_offset_world
    target = t
    plotter.camera_position = [eye, target, camera_up]