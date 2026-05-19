from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from perception.detection import YoloDetection
from perception.geometry import (
    scale_intrinsics_for_shape,
    transform_point,
)
from perception.io.frame_packet import FramePacket
from perception.output import ObjectPoseOutput


@dataclass
class RgbdLocalizerSettings:
    confidence_floor: int = 1
    min_depth_pixels: int = 24


def _to_depth_mask(mask_rgb: np.ndarray, depth_shape: tuple[int, int]) -> np.ndarray:
    dh, dw = depth_shape
    resized = cv2.resize(mask_rgb.astype(np.float32), (dw, dh), interpolation=cv2.INTER_NEAREST)
    return resized > 0.5


def _unproject_points(depth: np.ndarray, valid: np.ndarray, k_depth: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(valid)
    if ys.size == 0:
        return None
    z = depth[ys, xs].astype(np.float64)
    fx, fy = float(k_depth[0, 0]), float(k_depth[1, 1])
    cx, cy = float(k_depth[0, 2]), float(k_depth[1, 2])
    x = ((xs.astype(np.float64) - cx) * z) / fx
    y = ((ys.astype(np.float64) - cy) * z) / fy
    return np.stack([x, y, z], axis=1)


def _yaw_from_points_xy(points_xy: np.ndarray) -> Optional[float]:
    if points_xy.shape[0] < 20:
        return None
    c = np.mean(points_xy, axis=0, keepdims=True)
    centered = points_xy - c
    cov = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = int(np.argmax(eigvals))
    axis = eigvecs[:, idx]
    return float(np.arctan2(axis[1], axis[0]))


def localize_objects_rgbd(
    packet: FramePacket,
    detections: list[YoloDetection],
    t_world_camera: np.ndarray | None,
    settings: Optional[RgbdLocalizerSettings] = None,
) -> list[ObjectPoseOutput]:
    cfg = settings or RgbdLocalizerSettings()
    rgb_h, rgb_w = packet.rgb.shape[:2]
    depth_h, depth_w = packet.depth.shape[:2]
    k_depth = scale_intrinsics_for_shape(
        packet.intrinsic_mat, rgb_shape=(rgb_h, rgb_w), target_shape=(depth_h, depth_w)
    )

    out: list[ObjectPoseOutput] = []
    for idx, det in enumerate(detections):
        mask_depth = _to_depth_mask(det.mask_rgb, (depth_h, depth_w))
        valid = mask_depth & np.isfinite(packet.depth) & (packet.depth > 0.0)

        # Confidence gating: this is the key quality improvement for depth-based object localization.
        if packet.confidence is not None and packet.confidence.shape == packet.depth.shape:
            valid &= packet.confidence >= int(cfg.confidence_floor)

        if int(np.count_nonzero(valid)) < int(cfg.min_depth_pixels):
            continue

        pts_cam = _unproject_points(packet.depth, valid, k_depth)
        if pts_cam is None or pts_cam.shape[0] < int(cfg.min_depth_pixels):
            continue

        center_cam = np.median(pts_cam, axis=0)

        # det.yaw_hint_rad is None for symmetric objects (cup, bowl) -- skip yaw entirely
        is_symmetric = det.yaw_hint_rad is None

        if t_world_camera is not None:
            pts_world = np.array([transform_point(t_world_camera, p) for p in pts_cam], dtype=np.float64)
            center_world = np.median(pts_world, axis=0)
            z_min = float(np.percentile(pts_world[:, 2], 5))
            z_max = float(np.percentile(pts_world[:, 2], 95))
            height_m = max(0.0, z_max - z_min)
            position = (float(center_world[0]), float(center_world[1]), float(center_world[2]))
            yaw_hint = None if is_symmetric else _yaw_from_points_xy(pts_world[:, :2])
        else:
            z_min = float(np.percentile(pts_cam[:, 2], 5))
            z_max = float(np.percentile(pts_cam[:, 2], 95))
            height_m = max(0.0, z_max - z_min)
            position = (float(center_cam[0]), float(center_cam[1]), float(center_cam[2]))
            yaw_hint = None if is_symmetric else _yaw_from_points_xy(pts_cam[:, :2])

        quality = float(min(1.0, np.count_nonzero(valid) / max(500.0, det.mask_rgb.sum())))
        out.append(
            ObjectPoseOutput(
                object_id=f"{det.label}_{idx}",
                label=det.label,
                position_world_xyz_m=position,
                orientation_world_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                gripper_yaw_hint_rad=yaw_hint,
                quality=quality,
                covariance_diag=(0.01, 0.01, 0.02),
                height_m=height_m,
                source_mode="world" if t_world_camera is not None else "camera",
            )
        )
    return out
