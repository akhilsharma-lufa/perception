"""Find a pointer tip in a depth frame as the local depth peak above a known plane.

This is the perception half of the auto-robot-calibration loop. The robot moves
through a sequence of joint configurations. At each waypoint we want to know
the (x,y,z) position of the pointer tip in the camera frame so it can be paired
with the robot's reported flange/tool0 position to fit `T_robot_world` via
Kabsch.

The ChArUco board defines the world frame. The pointer protrudes from the arm
end face along tool-Z; we constrain the waypoint joint configs so the pointer
points roughly upward (away from the board surface). Under that assumption,
the pointer tip is the highest point in board-frame Z within the camera view.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..geometry import scale_intrinsics_for_shape
from ..geometry.transforms import invert_transform


@dataclass
class TipDetectorSettings:
    # Pixels with board-frame Z outside [min, max] are rejected. The min keeps
    # the board surface itself from being picked as the "tip"; the max keeps a
    # vagrant pixel high above the workspace (eg. a ceiling reflection) from
    # winning.
    min_height_above_board_m: float = 0.05
    max_height_above_board_m: float = 0.50
    # Pixel window around the depth peak that gets averaged for the final tip
    # XYZ. Smaller -> sharper localization but noisier; larger -> smoother but
    # bleeds in nearby surface pixels.
    median_window_px: int = 5
    # Minimum pixels that survive the height band to call the frame a detection.
    min_peak_pixels: int = 5
    # Minimum depth change (m) vs. the reference frame to count a pixel as
    # "moved" when a reference depth is provided. Filters out static objects
    # (robot base, tripod, shelves) that are higher than the arm tip and would
    # otherwise win the depth-peak. Set to 0 to disable change-masking.
    min_change_vs_reference_m: float = 0.03


def _unproject(
    depth: np.ndarray,
    valid: np.ndarray,
    k_depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.where(valid)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float64), ys, xs
    z = depth[ys, xs].astype(np.float64)
    fx, fy = float(k_depth[0, 0]), float(k_depth[1, 1])
    cx, cy = float(k_depth[0, 2]), float(k_depth[1, 2])
    x = ((xs.astype(np.float64) - cx) * z) / fx
    y = ((ys.astype(np.float64) - cy) * z) / fy
    return np.stack([x, y, z], axis=1), ys, xs


def detect_tip_in_camera(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsic_rgb_3x3: np.ndarray,
    t_camera_board: np.ndarray,
    settings: Optional[TipDetectorSettings] = None,
    reference_depth: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Return the pointer tip XYZ in camera frame, or None if no peak found.

    If `reference_depth` is provided, pixels whose depth has not changed
    significantly (vs. cfg.min_change_vs_reference_m) compared to that
    reference frame are excluded. This eliminates static objects (robot base,
    tripod, monitors) from the depth-peak search, leaving only pixels that
    actually moved between the reference snapshot and the current frame.
    The reference frame should be captured with the arm at home BEFORE any
    waypoint motion.
    """
    cfg = settings or TipDetectorSettings()
    rgb_h, rgb_w = rgb.shape[:2]
    depth_h, depth_w = depth.shape[:2]
    k_depth = scale_intrinsics_for_shape(
        intrinsic_rgb_3x3,
        rgb_shape=(rgb_h, rgb_w),
        target_shape=(depth_h, depth_w),
    )

    valid = np.isfinite(depth) & (depth > 0.0)
    if reference_depth is not None and float(cfg.min_change_vs_reference_m) > 0.0:
        # Compare to reference; pixels with similar depth in both frames are
        # "static" and get excluded. A pixel that became valid (was invalid
        # in reference, now valid) is counted as motion.
        ref = np.asarray(reference_depth, dtype=np.float32)
        if ref.shape != depth.shape:
            # Shape mismatch -> skip change-mask rather than misalign data.
            pass
        else:
            ref_valid = np.isfinite(ref) & (ref > 0.0)
            both_valid = valid & ref_valid
            delta = np.zeros_like(depth, dtype=np.float64)
            delta[both_valid] = np.abs(
                depth[both_valid].astype(np.float64) - ref[both_valid].astype(np.float64)
            )
            change_mask = both_valid & (delta > float(cfg.min_change_vs_reference_m))
            # Pixels that are valid now but were invalid in reference also
            # count as motion (the arm moved into an unmapped region).
            new_pixels = valid & (~ref_valid)
            valid = change_mask | new_pixels

    if int(np.count_nonzero(valid)) < cfg.min_peak_pixels:
        return None

    pts_cam, ys, xs = _unproject(depth, valid, k_depth)
    if pts_cam.shape[0] == 0:
        return None

    # Transform every camera-frame XYZ into board frame.
    t_board_camera = invert_transform(t_camera_board)
    r = t_board_camera[:3, :3]
    t = t_board_camera[:3, 3]
    pts_board = (pts_cam @ r.T) + t
    z_board = pts_board[:, 2]

    # Mask to pixels strictly above the board surface and below a sane ceiling.
    in_band = (
        (z_board >= float(cfg.min_height_above_board_m))
        & (z_board <= float(cfg.max_height_above_board_m))
    )
    if int(np.count_nonzero(in_band)) < cfg.min_peak_pixels:
        return None

    z_board_band = z_board[in_band]
    ys_band = ys[in_band]
    xs_band = xs[in_band]
    pts_cam_band = pts_cam[in_band]

    # Peak: pixel with the largest board-frame Z (topmost point above board).
    peak_idx = int(np.argmax(z_board_band))
    peak_y = int(ys_band[peak_idx])
    peak_x = int(xs_band[peak_idx])

    # Median over a small window in the depth image around the peak. We
    # average in CAMERA-frame XYZ (the output we care about). To suppress
    # the wrist face when it bleeds into the window, take the median of the
    # closest-to-camera 50% of pixels — that biases toward the true tip.
    win = int(max(1, cfg.median_window_px))
    y0 = max(0, peak_y - win)
    y1 = min(depth_h, peak_y + win + 1)
    x0 = max(0, peak_x - win)
    x1 = min(depth_w, peak_x + win + 1)

    window_mask = np.zeros_like(valid)
    window_mask[y0:y1, x0:x1] = True
    window_valid = valid & window_mask
    if int(np.count_nonzero(window_valid)) < cfg.min_peak_pixels:
        return pts_cam_band[peak_idx]

    pts_win, _, _ = _unproject(depth, window_valid, k_depth)
    if pts_win.shape[0] == 0:
        return pts_cam_band[peak_idx]

    # Only keep window pixels that are ALSO above the board surface — the
    # window almost always overlaps the surrounding board plane, and including
    # those pixels biases the median back toward the plane and away from the
    # tip cluster we just localized.
    pts_win_board = (pts_win @ r.T) + t
    win_in_band = (
        (pts_win_board[:, 2] >= float(cfg.min_height_above_board_m))
        & (pts_win_board[:, 2] <= float(cfg.max_height_above_board_m))
    )
    pts_tip = pts_win[win_in_band]
    if pts_tip.shape[0] < cfg.min_peak_pixels:
        # Fall back to the single peak pixel (better than averaging plane pixels in).
        return pts_cam_band[peak_idx]
    # Among the in-band pixels, bias toward the closest 50% (smallest camera-Z
    # = tip is "more in front of" the wrist face) — this picks the actual tip
    # cluster over the wider end-face cap that may share the window.
    zs = pts_tip[:, 2]
    cutoff = float(np.percentile(zs, 50.0))
    near = pts_tip[zs <= cutoff]
    if near.shape[0] >= max(3, cfg.min_peak_pixels // 2):
        return np.median(near, axis=0)
    return np.median(pts_tip, axis=0)
