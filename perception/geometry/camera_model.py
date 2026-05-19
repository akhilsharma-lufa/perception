from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_matrix(cls, k_3x3: np.ndarray) -> "CameraIntrinsics":
        k = np.asarray(k_3x3, dtype=np.float64)
        return cls(fx=float(k[0, 0]), fy=float(k[1, 1]), cx=float(k[0, 2]), cy=float(k[1, 2]))

    def to_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


def scale_intrinsics_for_shape(
    k_rgb: np.ndarray, rgb_shape: tuple[int, int], target_shape: tuple[int, int]
) -> np.ndarray:
    rgb_h, rgb_w = rgb_shape
    target_h, target_w = target_shape
    sx = float(target_w) / float(rgb_w)
    sy = float(target_h) / float(rgb_h)
    k = np.array(k_rgb, dtype=np.float64, copy=True)
    k[0, 0] *= sx
    k[1, 1] *= sy
    k[0, 2] *= sx
    k[1, 2] *= sy
    return k


def unproject_pixel(u: float, v: float, depth_m: float, k_3x3: np.ndarray) -> np.ndarray:
    k = np.asarray(k_3x3, dtype=np.float64)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    x = (float(u) - cx) * float(depth_m) / fx
    y = (float(v) - cy) * float(depth_m) / fy
    z = float(depth_m)
    return np.array([x, y, z], dtype=np.float64)


def project_point(p_xyz: np.ndarray, k_3x3: np.ndarray) -> np.ndarray:
    p = np.asarray(p_xyz, dtype=np.float64).reshape(3)
    if p[2] <= 1e-9:
        raise ValueError("Point has non-positive depth for projection.")
    k = np.asarray(k_3x3, dtype=np.float64)
    u = (k[0, 0] * p[0] / p[2]) + k[0, 2]
    v = (k[1, 1] * p[1] / p[2]) + k[1, 2]
    return np.array([u, v], dtype=np.float64)
