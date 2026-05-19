from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from perception.io.frame_packet import CameraPose


class RotationCode(IntEnum):
    NONE = 0
    CW90 = 1
    CCW90 = 2
    ROT180 = 3


def _quat_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def detect_orientation_from_pose(camera_pose: CameraPose) -> RotationCode:
    """Determine phone orientation from the ARKit camera pose quaternion.

    ARKit world frame has Y-up (anti-gravity).  The quaternion encodes the
    camera-to-world rotation so we can recover the gravity direction in camera
    space and decide how to rotate the image to make scene content upright.

    Camera-space conventions (ARKit rear camera):
        +X  = right in image (u direction)
        +Y  = up in 3D      (maps to *-v* in image, i.e. toward top of frame)
        -Z  = forward (into scene)

    Image-space conventions:
        u  (columns, right)  =  camera +X
        v  (rows,    down)   =  camera -Y
    """
    R = _quat_to_rotation_matrix(
        camera_pose.qx, camera_pose.qy, camera_pose.qz, camera_pose.qw
    )
    gravity_world = np.array([0.0, -1.0, 0.0])
    gravity_cam = R.T @ gravity_world

    img_grav_u = float(gravity_cam[0])
    img_grav_v = float(-gravity_cam[1])

    if abs(img_grav_v) >= abs(img_grav_u):
        if img_grav_v >= 0:
            return RotationCode.NONE
        return RotationCode.ROT180
    else:
        # Landscape branch:
        # If gravity points left in image space, content is rotated CW -> fix with CCW.
        # If gravity points right in image space, content is rotated CCW -> fix with CW.
        if img_grav_u < 0:
            return RotationCode.CCW90
        return RotationCode.CW90


def detect_orientation(
    rgb: np.ndarray,
    camera_pose: CameraPose | None = None,
) -> RotationCode:
    """Best-effort orientation detection.

    Uses the camera pose quaternion (gravity vector) when available, which
    works regardless of whether the frame dimensions change with orientation.
    Falls back to an aspect-ratio heuristic when no pose is provided.
    """
    if camera_pose is not None:
        return detect_orientation_from_pose(camera_pose)
    h, w = rgb.shape[:2]
    if h >= w:
        return RotationCode.NONE
    return RotationCode.CW90


def rotate_to_upright(image: np.ndarray, code: RotationCode) -> np.ndarray:
    if code == RotationCode.CW90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if code == RotationCode.CCW90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if code == RotationCode.ROT180:
        return cv2.rotate(image, cv2.ROTATE_180)
    return image


def unrotate_mask(mask: np.ndarray, code: RotationCode) -> np.ndarray:
    if code == RotationCode.CW90:
        return cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if code == RotationCode.CCW90:
        return cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    if code == RotationCode.ROT180:
        return cv2.rotate(mask, cv2.ROTATE_180)
    return mask


def unrotate_point(
    x: int, y: int, code: RotationCode, orig_h: int, orig_w: int
) -> tuple[int, int]:
    if code == RotationCode.CW90:
        return y, orig_w - 1 - x
    if code == RotationCode.CCW90:
        return orig_h - 1 - y, x
    if code == RotationCode.ROT180:
        return orig_w - 1 - x, orig_h - 1 - y
    return x, y
