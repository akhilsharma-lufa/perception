"""Live ChArUco + YOLO viewer — show every detected object's world pose & size.

Builds on `aruco_world_view.py` by adding YOLO segmentation. For every
object YOLO returns:

    1.  Convert its mask to depth-resolution.
    2.  Unproject every mask pixel to a 3D point in CAMERA frame using the
        depth image and intrinsics.
    3.  Transform those points to WORLD frame using T_world_cam = inv(T_cam_board).
    4.  Take the median (x, y, z) -> position in world frame.
    5.  Approximate the object's bounding-box dimensions in world frame:
          - width_x_m = p95(x) - p5(x)   (along board X)
          - depth_y_m = p95(y) - p5(y)   (along board Y)
          - height_z_m: signed distance from base to top along board -Z
            (height ABOVE the board surface; +Z points into the table).
    6.  Overlay the mask + a panel with label, confidence, world XYZ
        in mm, and the three dimensions in mm.

This script does NOT need the robot — everything is anchored to the
printed ChArUco sheet, which IS the world frame.

Run (Mac, with iPhone Record3D streaming):

    python3 -m playground.aruco_world_objects \
        --yolo-model yolo26n-seg.pt \
        --classes "cup,bottle,wine glass,bowl,vase" \
        --min-confidence 0.30

Press `q` to quit, `s` for a snapshot.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from perception.calibration.charuco_board import (
    CharucoBoardConfig,
    detect_board_pose,
)
from perception.detection import YoloDetectorSettings, YoloObjectDetector
from perception.geometry import scale_intrinsics_for_shape
from perception.geometry.transforms import invert_transform
from perception.io import Record3DSource


_OBJECT_COLORS = [
    (30, 200, 255),
    (100, 255, 100),
    (255, 120, 60),
    (200, 100, 255),
    (60, 255, 200),
]


def _project(p_board_m, t_camera_board, k):
    p = np.asarray(p_board_m, dtype=np.float64).reshape(3)
    p_cam = t_camera_board[:3, :3] @ p + t_camera_board[:3, 3]
    if p_cam[2] <= 1e-6:
        return None
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    return (
        int(round(fx * p_cam[0] / p_cam[2] + cx)),
        int(round(fy * p_cam[1] / p_cam[2] + cy)),
    )


def _draw_axes(rgb_bgr, t_cam_board, k, axis_len_m=0.05):
    p_o = _project(np.array([0.0, 0.0, 0.0]), t_cam_board, k)
    p_x = _project(np.array([axis_len_m, 0.0, 0.0]), t_cam_board, k)
    p_y = _project(np.array([0.0, axis_len_m, 0.0]), t_cam_board, k)
    p_z = _project(np.array([0.0, 0.0, axis_len_m]), t_cam_board, k)
    if p_o is None:
        return
    if p_x is not None:
        cv2.arrowedLine(rgb_bgr, p_o, p_x, (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.25)
    if p_y is not None:
        cv2.arrowedLine(rgb_bgr, p_o, p_y, (0, 200, 0), 2, cv2.LINE_AA, tipLength=0.25)
    if p_z is not None:
        cv2.arrowedLine(rgb_bgr, p_o, p_z, (255, 80, 0), 2, cv2.LINE_AA, tipLength=0.25)


def _object_world_stats(
    mask_rgb_full: np.ndarray,
    packet,
    t_world_camera: np.ndarray,
) -> dict | None:
    """Compute world-frame stats for one YOLO detection mask.

    Returns position, bbox dims along world X/Y, height above the table,
    and the count of valid depth pixels (a quality cue).
    """
    rgb_h, rgb_w = packet.rgb.shape[:2]
    depth = packet.depth
    if depth is None:
        return None
    depth_h, depth_w = depth.shape[:2]

    # Resize mask to the depth grid (the YOLO mask is at RGB resolution).
    mask = mask_rgb_full.astype(np.uint8)
    if mask.shape[:2] != (depth_h, depth_w):
        mask = cv2.resize(mask, (depth_w, depth_h), interpolation=cv2.INTER_NEAREST)
    mask_b = mask > 0

    valid = mask_b & np.isfinite(depth) & (depth > 0.0)
    n_valid = int(np.count_nonzero(valid))
    if n_valid < 24:
        return None

    k_depth = scale_intrinsics_for_shape(
        packet.intrinsic_mat, rgb_shape=(rgb_h, rgb_w), target_shape=(depth_h, depth_w)
    )
    fx, fy = float(k_depth[0, 0]), float(k_depth[1, 1])
    cx, cy = float(k_depth[0, 2]), float(k_depth[1, 2])

    ys, xs = np.where(valid)
    z = depth[ys, xs].astype(np.float64)
    x_cam = (xs.astype(np.float64) - cx) * z / fx
    y_cam = (ys.astype(np.float64) - cy) * z / fy
    pts_cam = np.stack([x_cam, y_cam, z], axis=1)

    r = t_world_camera[:3, :3]
    t = t_world_camera[:3, 3]
    pts_world = (pts_cam @ r.T) + t

    pos = np.median(pts_world, axis=0)

    x_lo, x_hi = np.percentile(pts_world[:, 0], [5.0, 95.0])
    y_lo, y_hi = np.percentile(pts_world[:, 1], [5.0, 95.0])
    z_lo, z_hi = np.percentile(pts_world[:, 2], [5.0, 95.0])

    # World +Z points into the table (see touch_calibrate.py comment).
    # The object sits ABOVE the board, so its world-Z is NEGATIVE; height
    # above the table is the magnitude of the most-negative Z.
    # We report height as |min(z)|, robust via the 5th percentile.
    height_m = float(max(0.0, -z_lo))

    return {
        "position_world_m": (float(pos[0]), float(pos[1]), float(pos[2])),
        "width_x_m": float(x_hi - x_lo),
        "depth_y_m": float(y_hi - y_lo),
        "height_z_m": height_m,
        "n_valid": n_valid,
    }


def _draw_text_box(image, lines, x, y, text_color=(255, 255, 255), bg_color=(18, 18, 18)):
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th, padx, pady, gap = 0.45, 1, 6, 5, 5
    sizes = [cv2.getTextSize(s, font, fs, th)[0] for s in lines]
    bw = max(w for w, _ in sizes) + 2 * padx
    lh = max(h for _, h in sizes)
    bh = len(lines) * lh + (len(lines) - 1) * gap + 2 * pady
    h, w = image.shape[:2]
    x0 = int(np.clip(x, 0, max(0, w - bw - 1)))
    y0 = int(np.clip(y, 0, max(0, h - bh - 1)))
    cv2.rectangle(image, (x0, y0), (x0 + bw, y0 + bh), bg_color, -1, cv2.LINE_AA)
    cv2.rectangle(image, (x0, y0), (x0 + bw, y0 + bh), (235, 235, 235), 1, cv2.LINE_AA)
    by = y0 + pady + lh
    for s in lines:
        cv2.putText(image, s, (x0 + padx, by), font, fs, text_color, th, cv2.LINE_AA)
        by += lh + gap


def _draw_object(rgb_bgr, det, color, stats):
    h, w = rgb_bgr.shape[:2]
    mask = det.mask_rgb.astype(np.uint8)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay = np.zeros_like(rgb_bgr)
    overlay[mask > 0] = color
    rgb_bgr[:] = cv2.addWeighted(rgb_bgr, 1.0, overlay, 0.32, 0.0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb_bgr, contours, -1, color, 2, cv2.LINE_AA)
    cv2.circle(rgb_bgr, (det.anchor_x, det.anchor_y), 5, (0, 255, 255), -1, cv2.LINE_AA)

    lines = [f"{det.label} ({det.confidence:.0%})"]
    if stats is not None:
        px, py, pz = stats["position_world_m"]
        lines.append(f"world: ({px*1000:+5.0f},{py*1000:+5.0f},{pz*1000:+5.0f}) mm")
        lines.append(
            f"size: {stats['width_x_m']*1000:.0f} x {stats['depth_y_m']*1000:.0f}"
            f" x {stats['height_z_m']*1000:.0f} mm"
        )
        lines.append(f"depth pts: {stats['n_valid']}")
    else:
        lines.append("world: (no depth)")
    x, y, bw, _ = cv2.boundingRect(mask)
    _draw_text_box(rgb_bgr, lines, x=x + bw // 2 - 80, y=max(6, y - 80))


def main():
    p = argparse.ArgumentParser(description="Live ChArUco + YOLO world-frame viewer.")
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--yolo-model", default="yolo26n-seg.pt")
    p.add_argument("--min-confidence", type=float, default=0.30)
    p.add_argument(
        "--classes",
        default="cup,bottle,wine glass,bowl,vase",
        help="Comma-separated YOLO class whitelist. Use '*' to disable filtering.",
    )
    p.add_argument("--infer-every-n", type=int, default=1)
    p.add_argument("--squares-x", type=int, default=11)
    p.add_argument("--squares-y", type=int, default=8)
    p.add_argument("--square-mm", type=float, default=20.0)
    p.add_argument("--marker-mm", type=float, default=14.0)
    p.add_argument("--dict", default="DICT_4X4_50")
    p.add_argument(
        "--no-legacy-pattern", dest="legacy_pattern", action="store_false", default=True
    )
    args = p.parse_args()

    cfg = CharucoBoardConfig(
        squares_x=int(args.squares_x),
        squares_y=int(args.squares_y),
        square_length_m=float(args.square_mm) * 1e-3,
        marker_length_m=float(args.marker_mm) * 1e-3,
        dictionary_name=str(args.dict),
        legacy_pattern=bool(args.legacy_pattern),
    )

    class_whitelist = None
    raw = str(args.classes).strip()
    if raw != "*":
        class_whitelist = tuple(c.strip() for c in raw.split(",") if c.strip()) or None

    detector = YoloObjectDetector(
        YoloDetectorSettings(
            model_path=args.yolo_model,
            min_confidence=float(args.min_confidence),
            class_whitelist=class_whitelist,
            inference_every_n_frames=max(1, int(args.infer_every_n)),
            hold_frames=8,
        )
    )

    source = Record3DSource()
    print(f"[aruco-yolo] connecting to Record3D device #{args.device_index}...")
    source.connect(device_index=int(args.device_index))
    print(
        f"[aruco-yolo] yolo={args.yolo_model} classes={class_whitelist or '*'} "
        f"min_conf={args.min_confidence:.2f}"
    )
    print("[aruco-yolo] warming YOLO (first call can take 30-60s on CPU)...")
    _t = time.monotonic()
    detector._ensure_model()
    try:
        _ = detector.infer(np.zeros((480, 640, 3), dtype=np.uint8), frame_id=0)
    except Exception as exc:
        print(f"[aruco-yolo] warmup failed (will retry live): {exc}")
    print(f"[aruco-yolo] YOLO ready in {time.monotonic() - _t:.1f}s")

    win = "aruco_world_objects"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            packet = source.wait_for_frame(timeout_s=0.25)
            if packet is None:
                continue

            det = detect_board_pose(packet.rgb, packet.intrinsic_mat, cfg)
            t_world_camera = invert_transform(det.t_camera_board) if det is not None else None

            detections = detector.infer(
                packet.rgb, frame_id=packet.frame_id, camera_pose=packet.camera_pose
            )

            rgb_bgr = cv2.cvtColor(packet.rgb, cv2.COLOR_RGB2BGR)

            if det is not None:
                _draw_axes(rgb_bgr, det.t_camera_board, packet.intrinsic_mat)
                # Corner sprinkle (no labels — keeps the YOLO overlays readable).
                for (px, py) in det.corners_image:
                    cv2.circle(
                        rgb_bgr, (int(round(px)), int(round(py))),
                        2, (50, 200, 255), -1, cv2.LINE_AA,
                    )

            for i, d in enumerate(detections):
                color = _OBJECT_COLORS[i % len(_OBJECT_COLORS)]
                stats = (
                    _object_world_stats(d.mask_rgb, packet, t_world_camera)
                    if t_world_camera is not None else None
                )
                _draw_object(rgb_bgr, d, color, stats)

            anchor_status = (
                f"anchor=charuco reproj={det.reprojection_error_px:.2f}px corners={det.n_corners}"
                if det is not None else "anchor=NONE  (board not visible)"
            )
            cv2.putText(
                rgb_bgr, anchor_status, (10, rgb_bgr.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255) if det is not None else (60, 80, 255),
                2, cv2.LINE_AA,
            )

            cv2.imshow(win, rgb_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                out = Path("playground") / f"aruco_objects_{int(time.time())}.png"
                cv2.imwrite(str(out), rgb_bgr)
                print(f"[aruco-yolo] snapshot saved -> {out}")
    finally:
        source.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
