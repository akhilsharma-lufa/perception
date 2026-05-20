from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PlaneFitResult:
    normal: np.ndarray  # unit vector, shape (3,)
    origin: np.ndarray  # a point on the plane, shape (3,) — the inlier centroid
    inlier_count: int
    inlier_ratio: float
    mean_abs_residual_m: float

    def signed_distance(self, points_xyz: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
        return (pts - self.origin) @ self.normal


def _refit_svd(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1, :]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    return normal, centroid


def ransac_plane(
    points_xyz: np.ndarray,
    distance_threshold_m: float = 0.008,
    max_iterations: int = 256,
    min_inlier_ratio: float = 0.30,
    seed: int = 0,
    orient_reference: Optional[np.ndarray] = None,
) -> Optional[PlaneFitResult]:
    """Robust plane fit via RANSAC followed by SVD refit on inliers.

    `orient_reference`, if supplied, is a point that should lie on the positive
    side of the plane (e.g. the camera position). The returned normal is flipped
    so this holds.
    """
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    if n < 32:
        return None

    rng = np.random.default_rng(seed)
    best_inlier_count = 0
    best_inliers: Optional[np.ndarray] = None

    sample_size = 3
    for _ in range(int(max_iterations)):
        idx = rng.choice(n, size=sample_size, replace=False)
        p0, p1, p2 = pts[idx[0]], pts[idx[1]], pts[idx[2]]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -float(np.dot(normal, p0))
        dists = np.abs(pts @ normal + d)
        inliers = dists < float(distance_threshold_m)
        count = int(inliers.sum())
        if count > best_inlier_count:
            best_inlier_count = count
            best_inliers = inliers

    if best_inliers is None or best_inlier_count < max(32, int(min_inlier_ratio * n)):
        return None

    inlier_pts = pts[best_inliers]
    normal, centroid = _refit_svd(inlier_pts)
    residuals = np.abs((inlier_pts - centroid) @ normal)

    if orient_reference is not None:
        ref = np.asarray(orient_reference, dtype=np.float64).reshape(3)
        if float(np.dot(ref - centroid, normal)) < 0.0:
            normal = -normal

    return PlaneFitResult(
        normal=normal,
        origin=centroid,
        inlier_count=int(best_inlier_count),
        inlier_ratio=float(best_inlier_count) / float(n),
        mean_abs_residual_m=float(np.mean(residuals)),
    )
