"""Automated robot-world calibration: drive the arm through a list of joint-
space waypoints and pair each pose with the camera-observed pointer tip.

Pipeline per waypoint:
    1. send_angles_deg(joint_cfg) and wait until the arm stops moving.
    2. Settle for a few hundred milliseconds.
    3. Sample N frames; for each frame:
          - detect the ChArUco board   -> T_camera_board
          - detect the pointer tip     -> tip_in_camera (3,)
    4. Median tip_in_camera and average T_camera_board across the N frames.
    5. Convert tip_in_camera -> tip_in_world via inv(T_camera_board).
    6. Read the arm's tool0 pose and convert it -> tip_in_robot via
       gripper_tip_position_in_robot(tip_offset_z = 12 mm).
    7. Append (tip_in_world, tip_in_robot) to the sample set.

Once all waypoints are sampled, run `kabsch_align` on the pairs, optionally
reject outliers by residual, and report the final T_robot_world + RMSE.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..control.mycobot_driver import MyCobotDriver
from ..geometry.transforms import average_transforms, invert_transform
from ..io.frame_packet import FramePacket
from ..io.record3d_source import Record3DSource
from .charuco_board import CharucoBoardConfig, detect_board_pose
from .robot_calibrator import (
    KabschResult,
    gripper_tip_position_in_robot,
    kabsch_align,
)
from .tip_detector import TipDetectorSettings, detect_tip_in_camera


# Hardcoded joint-space waypoint OFFSETS (J1..J6 in degrees).
#
# Column 0 is a *J1 offset from `j1_center_deg`* (see AutoCalibratorSettings),
# NOT an absolute J1 angle. The runtime adds j1_center_deg to column 0 before
# sending the angles, so the same waypoint table works for whichever side of
# the robot's base is obstacle-free. Default j1_center_deg = +90 puts the arm
# entirely on one side; pass `--j1-center -90` (in the demo) to flip to the
# other side, or `--j1-center 180` to operate directly behind home.
#
# All configurations keep the tool axis pointing roughly upward so the pointer
# tip is the highest point above the board (which the depth-peak tip detector
# relies on). J6 = 0 because the pointer is on the central axis and J6 rotation
# is irrelevant for tip position.
#
# Volumes covered:
#   - 3 broad J2/J3 "reach" classes (close, medium, far in front of base)
#   - 3 lateral classes via J1 offset (centered, +25 deg, -25 deg)
#   - 2 wrist-pitch classes via J5 to add Z variation
#
# This is a starting set; iterate after first field run if some waypoints
# place the tip out of camera view.
WAYPOINT_OFFSETS_DEG: List[List[float]] = [
    # --- near, low ---
    [   0, -50, -40,   0, -10,  0],
    [  25, -50, -40,   0, -10,  0],
    [ -25, -50, -40,   0, -10,  0],
    # --- near, high (more J5 tilt back) ---
    [   0, -55, -30,   0, -25,  0],
    [  25, -55, -30,   0, -25,  0],
    [ -25, -55, -30,   0, -25,  0],
    # --- medium, low ---
    [   0, -35, -45,   0, -10,  0],
    [  30, -35, -45,   0, -10,  0],
    [ -30, -35, -45,   0, -10,  0],
    # --- medium, high ---
    [   0, -40, -25,   0, -30,  0],
    [  25, -40, -25,   0, -30,  0],
    [ -25, -40, -25,   0, -30,  0],
    # --- far, low ---
    [   0, -20, -55,   0, -10,  0],
    [  20, -20, -55,   0, -10,  0],
    [ -20, -20, -55,   0, -10,  0],
    # --- far, high ---
    [   0, -25, -40,   0, -30,  0],
    [  20, -25, -40,   0, -30,  0],
    [ -20, -25, -40,   0, -30,  0],
    # --- two wide lateral configs to anchor the rotation ---
    [  45, -40, -30,   0, -20,  0],
    [ -45, -40, -30,   0, -20,  0],
]


def resolved_waypoints(
    offsets: List[List[float]], j1_center_deg: float
) -> List[List[float]]:
    """Apply the J1 center offset to each waypoint and return absolute joint
    targets in degrees."""
    out: List[List[float]] = []
    for row in offsets:
        cfg = [float(v) for v in row]
        cfg[0] = cfg[0] + float(j1_center_deg)
        out.append(cfg)
    return out


@dataclass
class AutoCalibratorSettings:
    tip_offset_z_m: float = 0.012
    frames_per_waypoint: int = 10
    settle_s: float = 0.4
    move_speed: int = 40
    move_timeout_s: float = 8.0
    frame_timeout_s: float = 0.5
    min_inlier_samples: int = 8
    # Multiplied by the median residual for outlier rejection. Anything above
    # `outlier_residual_factor * median` is dropped before a re-fit.
    outlier_residual_factor: float = 3.0
    # Hard floor on the rejection threshold so we don't drop everything when
    # the initial RMSE happens to be near zero by coincidence.
    outlier_residual_floor_m: float = 0.003
    # ChArUco detection quality gates.
    board_min_corners: int = 6
    board_max_reproj_error_px: float = 1.5
    tip_settings: TipDetectorSettings = field(default_factory=TipDetectorSettings)
    # Center of the J1 sweep, in degrees. The waypoint table stores J1
    # *offsets*; the absolute J1 commanded to the arm is `j1_center_deg +
    # offset`. Default +90 puts the arm 90 deg counterclockwise from home,
    # avoiding the home direction where the user's monitors live. Flip to
    # -90 if the obstacle layout is mirrored, or 180 to operate directly
    # behind the home direction.
    j1_center_deg: float = 90.0
    waypoint_offsets_deg: List[List[float]] = field(
        default_factory=lambda: [list(w) for w in WAYPOINT_OFFSETS_DEG]
    )

    def resolved_waypoints_deg(self) -> List[List[float]]:
        return resolved_waypoints(self.waypoint_offsets_deg, self.j1_center_deg)

    def j1_min_max_deg(self) -> tuple[float, float]:
        j1_vals = [row[0] + self.j1_center_deg for row in self.waypoint_offsets_deg]
        return min(j1_vals), max(j1_vals)


@dataclass
class SamplePoint:
    waypoint_idx: int
    joint_angles_deg: List[float]
    tip_world_m: np.ndarray
    tip_robot_m: np.ndarray
    n_frames: int


@dataclass
class AutoCalibResult:
    kabsch: KabschResult
    samples: List[SamplePoint]
    inlier_indices: List[int]


def _drain_stale_frames(source: Record3DSource, n: int = 2) -> None:
    """Pop any latched frames so the next wait_for_frame returns something
    captured after motion finished. The Record3DSource keeps only the most
    recent frame, but a frame caught mid-motion is still a frame; draining
    twice with a tight timeout typically clears it."""
    for _ in range(int(n)):
        source.wait_for_frame(timeout_s=0.1)


def collect_samples(
    driver: MyCobotDriver,
    source: Record3DSource,
    board_cfg: CharucoBoardConfig,
    settings: AutoCalibratorSettings,
) -> List[SamplePoint]:
    samples: List[SamplePoint] = []
    resolved = settings.resolved_waypoints_deg()
    n_wp = len(resolved)
    for i, joint_cfg in enumerate(resolved):
        print(
            f"[auto-cal] waypoint {i+1}/{n_wp}: "
            f"[{', '.join(f'{a:5.0f}' for a in joint_cfg)}]"
        )
        driver.send_angles_deg(joint_cfg, speed=int(settings.move_speed))
        try:
            driver.wait_until_done(strict=False, timeout_s=float(settings.move_timeout_s))
        except Exception as exc:
            print(f"[auto-cal]   wait_until_done warning: {exc}")
        time.sleep(float(settings.settle_s))
        _drain_stale_frames(source, n=2)

        tip_cam_samples: List[np.ndarray] = []
        t_cam_board_samples: List[np.ndarray] = []
        for _ in range(int(settings.frames_per_waypoint)):
            packet: Optional[FramePacket] = source.wait_for_frame(
                timeout_s=float(settings.frame_timeout_s)
            )
            if packet is None:
                continue
            det = detect_board_pose(
                packet.rgb,
                packet.intrinsic_mat,
                board_cfg,
                min_corners=int(settings.board_min_corners),
            )
            if det is None:
                continue
            if det.reprojection_error_px > float(settings.board_max_reproj_error_px):
                continue
            tip_cam = detect_tip_in_camera(
                packet.rgb,
                packet.depth,
                packet.intrinsic_mat,
                det.t_camera_board,
                settings.tip_settings,
            )
            if tip_cam is None:
                continue
            tip_cam_samples.append(np.asarray(tip_cam, dtype=np.float64).reshape(3))
            t_cam_board_samples.append(det.t_camera_board)

        if len(tip_cam_samples) < 3:
            print(
                f"[auto-cal]   SKIP (only {len(tip_cam_samples)} valid "
                f"tip/board detections in {settings.frames_per_waypoint} frames)"
            )
            continue

        tip_cam = np.median(np.stack(tip_cam_samples, axis=0), axis=0)
        t_cam_board = average_transforms(t_cam_board_samples)
        t_board_camera = invert_transform(t_cam_board)
        tip_world = (t_board_camera[:3, :3] @ tip_cam) + t_board_camera[:3, 3]

        try:
            coords = driver.get_coords_mm_deg(retries=4)
        except Exception as exc:
            print(f"[auto-cal]   SKIP (get_coords failed: {exc})")
            continue
        tip_robot = gripper_tip_position_in_robot(
            coords, tip_offset_z_m=float(settings.tip_offset_z_m)
        )

        samples.append(SamplePoint(
            waypoint_idx=i,
            joint_angles_deg=list(joint_cfg),
            tip_world_m=tip_world,
            tip_robot_m=tip_robot,
            n_frames=len(tip_cam_samples),
        ))
        tw = tip_world * 1000.0
        tr = tip_robot * 1000.0
        print(
            f"[auto-cal]   tip_world=({tw[0]:+6.1f}, {tw[1]:+6.1f}, {tw[2]:+6.1f}) mm  "
            f"tip_robot=({tr[0]:+6.1f}, {tr[1]:+6.1f}, {tr[2]:+6.1f}) mm  "
            f"frames={len(tip_cam_samples)}"
        )
    return samples


def fit_robot_world(
    samples: List[SamplePoint], settings: AutoCalibratorSettings
) -> AutoCalibResult:
    if len(samples) < int(settings.min_inlier_samples):
        raise RuntimeError(
            f"need at least {settings.min_inlier_samples} valid samples to fit; "
            f"got {len(samples)}"
        )
    world_pts = np.stack([s.tip_world_m for s in samples], axis=0)
    robot_pts = np.stack([s.tip_robot_m for s in samples], axis=0)
    result = kabsch_align(world_pts, robot_pts)

    residuals = np.asarray(result.per_point_residual_m, dtype=np.float64)
    med = float(np.median(residuals))
    threshold = max(
        med * float(settings.outlier_residual_factor),
        float(settings.outlier_residual_floor_m),
    )
    keep_mask = residuals <= threshold
    inlier_indices = [int(i) for i, k in enumerate(keep_mask) if bool(k)]

    if len(inlier_indices) < int(settings.min_inlier_samples):
        # Not enough inliers to safely re-fit — return the all-samples result
        # so the caller can still inspect what went wrong.
        return AutoCalibResult(
            kabsch=result,
            samples=samples,
            inlier_indices=list(range(len(samples))),
        )

    if len(inlier_indices) < len(samples):
        result = kabsch_align(world_pts[keep_mask], robot_pts[keep_mask])

    return AutoCalibResult(
        kabsch=result,
        samples=samples,
        inlier_indices=inlier_indices,
    )
