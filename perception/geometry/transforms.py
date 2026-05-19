from typing import Iterable, List

import numpy as np


def make_transform(rotation_3x3: np.ndarray, translation_3: np.ndarray) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = np.asarray(rotation_3x3, dtype=np.float64)
    t[:3, 3] = np.asarray(translation_3, dtype=np.float64).reshape(3)
    return t


def invert_transform(t_4x4: np.ndarray) -> np.ndarray:
    t = np.asarray(t_4x4, dtype=np.float64)
    r = t[:3, :3]
    p = t[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ p
    return out


def transform_point(t_4x4: np.ndarray, p_xyz: np.ndarray) -> np.ndarray:
    p = np.asarray(p_xyz, dtype=np.float64).reshape(3)
    ph = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    out = np.asarray(t_4x4, dtype=np.float64) @ ph
    return out[:3]


def average_rotations(rotations: Iterable[np.ndarray]) -> np.ndarray:
    rotations_list: List[np.ndarray] = [np.asarray(r, dtype=np.float64) for r in rotations]
    if not rotations_list:
        raise ValueError("Cannot average empty rotation set.")
    m = np.zeros((3, 3), dtype=np.float64)
    for r in rotations_list:
        m += r
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1.0
        r = u @ vt
    return r


def average_transforms(transforms: Iterable[np.ndarray]) -> np.ndarray:
    transforms_list: List[np.ndarray] = [np.asarray(t, dtype=np.float64) for t in transforms]
    if not transforms_list:
        raise ValueError("Cannot average empty transform set.")
    avg_t = np.mean([t[:3, 3] for t in transforms_list], axis=0)
    avg_r = average_rotations([t[:3, :3] for t in transforms_list])
    return make_transform(avg_r, avg_t)
