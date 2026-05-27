"""Diagnose why the object model (radius/height) comes out wrong.

Headless. For each frame it detects the board + YOLO target and dumps, per frame:
  - YOLO mask area (RGB res) + bbox width/height in px
  - depth points that survive the mask (the cloud the 3D fit uses) + median distance
  - height-above-table percentiles (p5/p50/p95/max)
  - the 3D circle-fit radius (what the localizer currently uses)
  - a MASK-BASED size estimate: physical Ø and height from the mask bbox scaled by
    the robust median distance (RGB resolution, robust to sparse depth)

Compare the two estimates: if the 3D fit is small/jumpy but the mask-based numbers
are stable and near the true size, the depth cloud is too sparse and we should
dimension from the mask instead.

    python -m perception.demos.cup_model_debug --classes '*' --target-label cup --frames 20
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from perception.calibration.charuco_board import CharucoBoardConfig, detect_board_pose
from perception.calibration.profiles import CalibrationProfileIO
from perception.detection.yolo_objects import YoloDetectorSettings, YoloObjectDetector
from perception.geometry import scale_intrinsics_for_shape
from perception.geometry.transforms import invert_transform
from perception.io.record3d_source import Record3DSource
from perception.localization.rgbd_localizer import (
    _erode_mask, _fit_circle_2d, _resize_mask_to_depth, _unproject_valid, _plane_basis,
)


def _board_cfg(profile) -> CharucoBoardConfig:
    s = profile.charuco_board
    return CharucoBoardConfig(int(s.squares_x), int(s.squares_y), float(s.square_length_m),
                              float(s.marker_length_m), str(s.dictionary_name), bool(s.legacy_pattern))


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m perception.demos.cup_model_debug")
    p.add_argument("--profile", default="calibration/profiles/session_multitag.json")
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--classes", default="*")
    p.add_argument("--target-label", default="cup")
    p.add_argument("--min-confidence", type=float, default=0.15)
    p.add_argument("--frames", type=int, default=20)
    args = p.parse_args()

    profile = CalibrationProfileIO.load(args.profile)
    if profile.table_plane is None:
        print("ERROR: profile has no table_plane", file=sys.stderr); sys.exit(2)
    n_world, o_world = profile.table_plane.as_arrays()
    board = _board_cfg(profile)

    wl = None if args.classes.strip() == "*" else tuple(c.strip() for c in args.classes.split(","))
    det = YoloObjectDetector(YoloDetectorSettings(min_confidence=float(args.min_confidence),
                                                  class_whitelist=wl, inference_every_n_frames=1, hold_frames=1))
    det._ensure_model()
    src = Record3DSource(); src.connect(device_index=int(args.device_index))
    print(f"{'frame':>5} {'mask_px':>8} {'bbox_wxh':>10} {'depth_pts':>9} {'dist_mm':>8} "
          f"{'h_p50':>6} {'h_p95':>6} {'h_max':>6} {'fit_Ø':>6} {'mask_Ø':>6} {'mask_h':>6}")
    seen = 0
    try:
        while seen < int(args.frames):
            pkt = src.wait_for_frame(timeout_s=0.5)
            if pkt is None:
                continue
            bd = detect_board_pose(pkt.rgb, pkt.intrinsic_mat, board)
            if bd is None:
                continue
            t_wc = invert_transform(bd.t_camera_board)
            dets = det.infer(pkt.rgb, frame_id=pkt.frame_id, camera_pose=pkt.camera_pose)
            for d in dets:
                if args.target_label not in ("*", "") and d.label != args.target_label:
                    continue
                seen += 1
                m_rgb = d.mask_rgb > 0
                ys, xs = np.where(m_rgb)
                if ys.size == 0:
                    continue
                bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
                # depth-domain cloud (what the 3D fit uses)
                dh, dw = pkt.depth.shape[:2]
                kd = scale_intrinsics_for_shape(pkt.intrinsic_mat, rgb_shape=pkt.rgb.shape[:2], target_shape=(dh, dw))
                md = _erode_mask(_resize_mask_to_depth(d.mask_rgb, (dh, dw)), 3)
                valid = md & np.isfinite(pkt.depth) & (pkt.depth > 0)
                pts_cam, _ = _unproject_valid(pkt.depth, valid, kd)
                npts = pts_cam.shape[0]
                dist = float(np.median(pts_cam[:, 2])) if npts else float("nan")
                # height percentiles above plane (world)
                if npts:
                    R = t_wc[:3, :3]; t = t_wc[:3, 3]
                    pw = pts_cam @ R.T + t
                    h = (pw - o_world) @ n_world
                    hp50, hp95, hmax = np.percentile(h, [50, 95, 100]) * 1000
                    u, v = _plane_basis(n_world)
                    rel = pw - o_world
                    fp = np.column_stack([rel @ u, rel @ v])
                    top = fp[h >= (h.max() - 0.015)]
                    fit = _fit_circle_2d(top) if top.shape[0] >= 12 else None
                    fitd = fit[2] * 2000 if fit else float("nan")
                else:
                    hp50 = hp95 = hmax = fitd = float("nan")
                # mask-based physical size: bbox px * dist / focal
                fx, fy = float(pkt.intrinsic_mat[0, 0]), float(pkt.intrinsic_mat[1, 1])
                mask_d = bw * dist / fx * 1000 if npts else float("nan")
                mask_h = bh * dist / fy * 1000 if npts else float("nan")
                print(f"{seen:5d} {int(m_rgb.sum()):8d} {f'{bw}x{bh}':>10} {npts:9d} {dist*1000:8.1f} "
                      f"{hp50:6.1f} {hp95:6.1f} {hmax:6.1f} {fitd:6.1f} {mask_d:6.1f} {mask_h:6.1f}")
    finally:
        src.disconnect()


if __name__ == "__main__":
    main()
