"""Replace the profile's hardcoded table_plane with an empirically-fit one.

Current `touch_calibrate.py` writes `table_plane = (normal=(0,0,-1), origin=(0,0,0))`
— i.e. the board's Z axis. If the printed board has any warp/lift or the table
isn't perfectly aligned with the board's Z, heights measured by projection onto
that hardcoded plane drift across the workspace.

This script collects depth points from the table SURFACE around the board, fits
a plane via RANSAC in the world frame, and writes it to the profile. World XY
(set by `T_robot_world`) and the board frame are unchanged.

    python -m perception.demos.fit_table_plane \\
        --profile calibration/profiles/session_multitag.json \\
        --frames 6
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from perception.calibration.charuco_board import (
    CharucoBoardConfig, detect_board_pose,
)
from perception.calibration.profiles import CalibrationProfileIO
from perception.geometry import scale_intrinsics_for_shape
from perception.geometry.plane_fit import ransac_plane
from perception.geometry.transforms import invert_transform
from perception.io.record3d_source import Record3DSource
from perception.localization.rgbd_localizer import _camera_to_world, _unproject_valid


def _board_cfg(profile) -> CharucoBoardConfig:
    s = profile.charuco_board
    if s is None:
        print("ERROR: profile has no charuco_board spec; run touch_calibrate first.", file=sys.stderr)
        sys.exit(2)
    return CharucoBoardConfig(int(s.squares_x), int(s.squares_y), float(s.square_length_m),
                              float(s.marker_length_m), str(s.dictionary_name), bool(s.legacy_pattern))


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m perception.demos.fit_table_plane")
    p.add_argument("--profile", default="calibration/profiles/session_multitag.json")
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--frames", type=int, default=6,
                   help="Number of frames to accumulate world-frame depth points from.")
    p.add_argument("--z-window-mm", type=float, default=40.0,
                   help="Keep only points within +/- this many mm of the board plane "
                        "(world Z=0). Excludes objects on the table, ceiling, etc.")
    p.add_argument("--board-margin-mm", type=float, default=10.0,
                   help="Exclude depth points whose world XY falls inside the board's "
                        "printed footprint (plus this margin) — we want to fit the "
                        "table AROUND the board, not the board paper itself.")
    p.add_argument("--write", action="store_true",
                   help="Save the fitted plane back to the profile. Otherwise just report.")
    args = p.parse_args()

    profile = CalibrationProfileIO.load(args.profile)
    if profile.get_robot_world_transform() is None:
        print("ERROR: profile has no robot_world_transform; run touch_calibrate first.", file=sys.stderr)
        sys.exit(2)
    board = _board_cfg(profile)
    board_w_m = float(board.squares_x) * float(board.square_length_m)
    board_h_m = float(board.squares_y) * float(board.square_length_m)

    src = Record3DSource(); src.connect(device_index=int(args.device_index))
    world_pts: list[np.ndarray] = []
    cam_world_sum = np.zeros(3); cam_n = 0
    try:
        captured = 0
        while captured < int(args.frames):
            pkt = src.wait_for_frame(timeout_s=0.5)
            if pkt is None:
                continue
            bd = detect_board_pose(pkt.rgb, pkt.intrinsic_mat, board)
            if bd is None or bd.reprojection_error_px > 1.5:
                continue
            t_world_camera = invert_transform(bd.t_camera_board)
            cam_world_sum = cam_world_sum + t_world_camera[:3, 3]; cam_n += 1
            dh, dw = pkt.depth.shape[:2]
            k_depth = scale_intrinsics_for_shape(pkt.intrinsic_mat,
                                                 rgb_shape=pkt.rgb.shape[:2],
                                                 target_shape=(dh, dw))
            valid = np.isfinite(pkt.depth) & (pkt.depth > 0.0)
            pts_cam, _ = _unproject_valid(pkt.depth, valid, k_depth)
            if pts_cam.shape[0] == 0:
                continue
            pts_world = _camera_to_world(pts_cam, t_world_camera)
            # Keep only points near the board's Z (table-level) and outside the
            # board's XY footprint.
            zw = float(args.z_window_mm) * 1e-3
            mz = np.abs(pts_world[:, 2]) <= zw
            mg = float(args.board_margin_mm) * 1e-3
            inside_board = (
                (pts_world[:, 0] >= -mg) & (pts_world[:, 0] <= board_w_m + mg) &
                (pts_world[:, 1] >= -mg) & (pts_world[:, 1] <= board_h_m + mg)
            )
            keep = mz & ~inside_board
            if int(keep.sum()) >= 200:
                # Subsample to keep memory bounded across frames.
                idx = np.random.default_rng(0).choice(int(keep.sum()),
                                                      size=min(int(keep.sum()), 4000),
                                                      replace=False)
                world_pts.append(pts_world[keep][idx])
            captured += 1
            print(f"  frame {captured}: kept {int(keep.sum())} table points "
                  f"(reproj {bd.reprojection_error_px:.2f}px)")
    finally:
        src.disconnect()

    if not world_pts:
        print("ERROR: no usable depth points around the board. Is the table around it visible?", file=sys.stderr)
        sys.exit(3)
    pts = np.vstack(world_pts)
    cam_world = cam_world_sum / max(1, cam_n)
    fit = ransac_plane(pts, distance_threshold_m=0.004, max_iterations=512,
                       min_inlier_ratio=0.30, seed=0, orient_reference=cam_world)
    if fit is None:
        print("ERROR: RANSAC failed (too few inliers).", file=sys.stderr); sys.exit(3)
    print(f"\nfitted table plane: normal={np.round(fit.normal, 4).tolist()}, "
          f"origin={np.round(fit.origin, 4).tolist()}, inliers={fit.inlier_count} "
          f"({fit.inlier_ratio*100:.1f}%), mean |residual|={fit.mean_abs_residual_m*1000:.2f} mm")
    tilt_deg = float(np.degrees(np.arccos(np.clip(abs(fit.normal[2]), 0.0, 1.0))))
    print(f"  tilt from board-Z: {tilt_deg:.2f}°  (0 = aligned with hardcoded (0,0,-1))")
    print(f"  origin Z offset:  {fit.origin[2]*1000:+.2f} mm  (0 = board paper is the table)")

    if args.write:
        profile.set_table_plane(
            normal_world=np.asarray(fit.normal, dtype=np.float64),
            origin_world=np.asarray(fit.origin, dtype=np.float64),
            inlier_ratio=float(fit.inlier_ratio),
            mean_abs_residual_m=float(fit.mean_abs_residual_m),
        )
        CalibrationProfileIO.save(profile, args.profile)
        print(f"\nsaved fitted table plane -> {args.profile}")
    else:
        print("\n(dry-run; pass --write to persist into the profile)")


if __name__ == "__main__":
    main()
