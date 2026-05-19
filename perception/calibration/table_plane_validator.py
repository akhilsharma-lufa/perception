from dataclasses import dataclass

import numpy as np


@dataclass
class TablePlaneValidationResult:
    valid: bool
    inlier_ratio: float
    mean_abs_residual_m: float


def validate_table_plane_consistency(
    points_xyz: np.ndarray,
    expected_normal: np.ndarray | None = None,
    max_mean_abs_residual_m: float = 0.01,
    min_inlier_ratio: float = 0.75,
) -> TablePlaneValidationResult:
    """
    Fast plane-fit sanity check for calibration sessions.
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 20:
        return TablePlaneValidationResult(False, 0.0, float("inf"))

    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1, :]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    if expected_normal is not None:
        en = np.asarray(expected_normal, dtype=np.float64).reshape(3)
        en = en / (np.linalg.norm(en) + 1e-12)
        if float(np.dot(normal, en)) < 0.0:
            normal = -normal
    residuals = np.abs(centered @ normal)
    threshold = max(max_mean_abs_residual_m * 2.0, 0.005)
    inlier_ratio = float(np.mean(residuals <= threshold))
    mean_abs_residual = float(np.mean(residuals))
    valid = inlier_ratio >= min_inlier_ratio and mean_abs_residual <= max_mean_abs_residual_m
    return TablePlaneValidationResult(valid, inlier_ratio, mean_abs_residual)
