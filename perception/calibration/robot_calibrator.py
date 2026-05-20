from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..geometry.transforms import make_transform


@dataclass
class KabschResult:
    t_robot_world: np.ndarray  # (4, 4) transform mapping world -> robot
    per_point_residual_m: np.ndarray  # (N,) per-point alignment error in meters
    rmse_m: float


def kabsch_align(
    world_points_m: Sequence[Sequence[float]],
    robot_points_m: Sequence[Sequence[float]],
) -> KabschResult:
    """Rigid alignment: find T (rotation + translation) such that
    `p_robot ≈ T @ p_world` for the provided point correspondences.

    Uses the standard SVD-based Kabsch algorithm with a determinant check to
    avoid reflections.

    Both inputs are sequences of 3-vectors in meters. At least 3 non-collinear
    correspondences are required.
    """
    w = np.asarray(world_points_m, dtype=np.float64).reshape(-1, 3)
    r = np.asarray(robot_points_m, dtype=np.float64).reshape(-1, 3)
    if w.shape != r.shape:
        raise ValueError(f"point set shapes differ: world={w.shape}, robot={r.shape}")
    n = w.shape[0]
    if n < 3:
        raise ValueError(f"Kabsch needs at least 3 correspondences; got {n}")

    w_centroid = np.mean(w, axis=0)
    r_centroid = np.mean(r, axis=0)
    w_centered = w - w_centroid
    r_centered = r - r_centroid

    # Cross-covariance matrix
    h = w_centered.T @ r_centered
    u, _, vt = np.linalg.svd(h)
    # Reflection check
    d = np.sign(np.linalg.det(vt.T @ u.T))
    s = np.diag([1.0, 1.0, d])
    rotation = vt.T @ s @ u.T
    translation = r_centroid - rotation @ w_centroid

    t = make_transform(rotation, translation)

    # Residuals
    pred = (w @ rotation.T) + translation
    diffs = pred - r
    per_point = np.linalg.norm(diffs, axis=1)
    rmse = float(np.sqrt(np.mean(per_point ** 2)))

    return KabschResult(
        t_robot_world=t,
        per_point_residual_m=per_point,
        rmse_m=rmse,
    )


def gripper_tip_position_in_robot(
    coords_mm_deg: Sequence[float],
    tip_offset_z_m: float,
    assume_pointing_down: bool = True,
) -> np.ndarray:
    """Translate the robot's reported `get_coords()` (the tool0 flange pose) into
    the gripper-tip position in the robot frame.

    For v1 we assume the user orients the gripper to point straight down
    before each calibration touch — that lets us subtract the tip offset
    along robot -Z without resolving the full Euler rotation. If you want a
    full rotation-aware version later, build it from coords_mm_deg[3:].
    """
    if len(coords_mm_deg) < 6:
        raise ValueError("coords_mm_deg must be [x,y,z,rx,ry,rz] (6 values)")
    x_mm, y_mm, z_mm = coords_mm_deg[:3]
    x_m = float(x_mm) * 1e-3
    y_m = float(y_mm) * 1e-3
    z_m = float(z_mm) * 1e-3
    if assume_pointing_down:
        # gripper tip is `tip_offset_z_m` below the flange in robot Z
        z_m = z_m - float(tip_offset_z_m)
    return np.array([x_m, y_m, z_m], dtype=np.float64)
