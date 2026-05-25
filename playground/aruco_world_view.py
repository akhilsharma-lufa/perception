"""Live ChArUco viewer — teaches how the board becomes the world frame.

What this script shows you, end-to-end, on every frame:

    1.  Pulls an RGB+depth frame from the iPhone via Record3D.
    2.  Runs `detect_board_pose` to find the printed ChArUco sheet and
        solves for `T_cam_board` (board pose in the camera frame).
    3.  Inverts that to get `T_world_cam` (where the camera is, in the
        board/world frame). Remember: world == board in this project.
    4.  Draws on the RGB frame:
          - every detected inner chessboard corner, numbered with its ID
          - an XYZ axis triad at the board origin (red=X, green=Y, blue=Z)
          - a labelled grid of every inner corner's world position in mm
          - a HUD with: reprojection error, n corners, camera position in
            world frame, camera optical axis direction, board pose summary
    5.  Mouse-hover-anywhere: prints the pixel's world XYZ (using the depth
        frame and `T_world_cam`) at the bottom of the window.

The intent is to make the board → world → camera relationship
something you can _see_. Move the phone around the tripod: the axes
stick to the board, the HUD coordinates change accordingly.

Run it (Mac, with the iPhone Record3D streaming):

    python3 -m playground.aruco_world_view \
        --squares-x 11 --squares-y 8 \
        --square-mm 20 --marker-mm 14 \
        --dict DICT_4X4_50

Press `q` to quit, `s` to save a snapshot of the current frame.
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
from perception.geometry import scale_intrinsics_for_shape
from perception.geometry.transforms import invert_transform
from perception.io import Record3DSource


def _project_board_point(
    p_board_m: np.ndarray,
    t_camera_board: np.ndarray,
    k: np.ndarray,
) -> tuple[int, int] | None:
    """Project a 3D point given in board-frame metres into the RGB image."""
    p = np.asarray(p_board_m, dtype=np.float64).reshape(3)
    p_cam = t_camera_board[:3, :3] @ p + t_camera_board[:3, 3]
    if p_cam[2] <= 1e-6:
        return None
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    u = int(round(fx * p_cam[0] / p_cam[2] + cx))
    v = int(round(fy * p_cam[1] / p_cam[2] + cy))
    return u, v


def _draw_axes(rgb_bgr: np.ndarray, t_camera_board: np.ndarray, k: np.ndarray, axis_len_m: float = 0.04) -> None:
    """Draw the world-frame XYZ triad at the board origin."""
    p_o = _project_board_point(np.array([0.0, 0.0, 0.0]), t_camera_board, k)
    p_x = _project_board_point(np.array([axis_len_m, 0.0, 0.0]), t_camera_board, k)
    p_y = _project_board_point(np.array([0.0, axis_len_m, 0.0]), t_camera_board, k)
    p_z = _project_board_point(np.array([0.0, 0.0, axis_len_m]), t_camera_board, k)
    if p_o is None:
        return
    if p_x is not None:
        cv2.arrowedLine(rgb_bgr, p_o, p_x, (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(rgb_bgr, "X", (p_x[0] + 5, p_x[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    if p_y is not None:
        cv2.arrowedLine(rgb_bgr, p_o, p_y, (0, 200, 0), 3, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(rgb_bgr, "Y", (p_y[0] + 5, p_y[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2, cv2.LINE_AA)
    if p_z is not None:
        cv2.arrowedLine(rgb_bgr, p_o, p_z, (255, 80, 0), 3, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(rgb_bgr, "Z", (p_z[0] + 5, p_z[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 0), 2, cv2.LINE_AA)


def _draw_inner_corners(
    rgb_bgr: np.ndarray,
    cfg: CharucoBoardConfig,
    t_camera_board: np.ndarray,
    k: np.ndarray,
    label_every: int = 2,
) -> None:
    """Project every inner chessboard corner and label its world (X,Y) mm.

    There are (squares_x - 1) * (squares_y - 1) inner corners. Labelling
    every one makes the image unreadable, so we label every Nth.
    """
    n_x = cfg.squares_x - 1
    n_y = cfg.squares_y - 1
    s_mm = cfg.square_length_m * 1000.0
    for row in range(n_y):
        for col in range(n_x):
            p_board = np.array(
                [(col + 1) * cfg.square_length_m, (row + 1) * cfg.square_length_m, 0.0]
            )
            pixel = _project_board_point(p_board, t_camera_board, k)
            if pixel is None:
                continue
            cv2.circle(rgb_bgr, pixel, 3, (50, 200, 255), -1, cv2.LINE_AA)
            if (col % label_every == 0) and (row % label_every == 0):
                lbl = f"({(col + 1) * s_mm:.0f},{(row + 1) * s_mm:.0f})"
                cv2.putText(
                    rgb_bgr,
                    lbl,
                    (pixel[0] + 4, pixel[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )


def _draw_corner_pixels(rgb_bgr: np.ndarray, corners_image: np.ndarray) -> None:
    """The actual detected corner pixels (sub-pixel) returned by OpenCV.

    These are the points that go into solvePnP. Comparing them to the
    projected-back grid above gives you an intuitive sense of how the
    reprojection error is computed.
    """
    for (px, py) in corners_image:
        cv2.circle(
            rgb_bgr,
            (int(round(px)), int(round(py))),
            5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _world_xyz_under_cursor(
    mouse_xy: tuple[int, int] | None,
    packet,
    t_world_camera: np.ndarray,
) -> tuple[float, float, float] | None:
    """Unproject a pixel under the mouse to world XYZ using the depth frame."""
    if mouse_xy is None:
        return None
    u, v = mouse_xy
    rgb_h, rgb_w = packet.rgb.shape[:2]
    depth = packet.depth
    if depth is None:
        return None
    depth_h, depth_w = depth.shape[:2]
    # Map the cursor (RGB pixel) into the lower-res depth grid.
    du = int(round(u * (depth_w / rgb_w)))
    dv = int(round(v * (depth_h / rgb_h)))
    if du < 0 or dv < 0 or du >= depth_w or dv >= depth_h:
        return None
    z = float(depth[dv, du])
    if not np.isfinite(z) or z <= 0.0:
        return None
    k_depth = scale_intrinsics_for_shape(
        packet.intrinsic_mat, rgb_shape=(rgb_h, rgb_w), target_shape=(depth_h, depth_w)
    )
    fx, fy = float(k_depth[0, 0]), float(k_depth[1, 1])
    cx, cy = float(k_depth[0, 2]), float(k_depth[1, 2])
    x_cam = (du - cx) * z / fx
    y_cam = (dv - cy) * z / fy
    p_cam = np.array([x_cam, y_cam, z], dtype=np.float64)
    p_world = t_world_camera[:3, :3] @ p_cam + t_world_camera[:3, 3]
    return float(p_world[0]), float(p_world[1]), float(p_world[2])


def _draw_hud(
    rgb_bgr: np.ndarray,
    detection,
    t_world_camera: np.ndarray | None,
    cursor_world: tuple[float, float, float] | None,
    fps: float,
) -> None:
    h, w = rgb_bgr.shape[:2]
    lines = []
    if detection is None:
        lines.append("BOARD: not detected")
    else:
        tx, ty, tz = detection.t_camera_board[:3, 3]
        lines.append(
            f"board in cam: ({tx*1000:+6.0f}, {ty*1000:+6.0f}, {tz*1000:+6.0f}) mm"
        )
        lines.append(
            f"corners={detection.n_corners}  reproj={detection.reprojection_error_px:.2f} px"
        )
        if t_world_camera is not None:
            cx, cy, cz = t_world_camera[:3, 3]
            lines.append(
                f"camera in world: ({cx*1000:+6.0f}, {cy*1000:+6.0f}, {cz*1000:+6.0f}) mm"
            )
            # The camera's optical axis in world frame: +Z_cam expressed in world.
            look_world = t_world_camera[:3, :3] @ np.array([0.0, 0.0, 1.0])
            lines.append(
                f"cam +Z in world: ({look_world[0]:+.2f}, {look_world[1]:+.2f}, {look_world[2]:+.2f})"
            )
    if cursor_world is not None:
        lines.append(
            f"under cursor: ({cursor_world[0]*1000:+6.0f}, "
            f"{cursor_world[1]*1000:+6.0f}, {cursor_world[2]*1000:+6.0f}) mm"
        )
    lines.append(f"fps: {fps:.1f}   [q] quit  [s] snapshot")

    y = 22
    for ln in lines:
        cv2.putText(rgb_bgr, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(rgb_bgr, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


def main():
    p = argparse.ArgumentParser(description="Live ChArUco world-frame viewer.")
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--squares-x", type=int, default=11)
    p.add_argument("--squares-y", type=int, default=8)
    p.add_argument("--square-mm", type=float, default=20.0)
    p.add_argument("--marker-mm", type=float, default=14.0)
    p.add_argument("--dict", default="DICT_4X4_50")
    p.add_argument(
        "--no-legacy-pattern", dest="legacy_pattern", action="store_false", default=True
    )
    p.add_argument("--label-every", type=int, default=2, help="Label every Nth inner corner.")
    args = p.parse_args()

    cfg = CharucoBoardConfig(
        squares_x=int(args.squares_x),
        squares_y=int(args.squares_y),
        square_length_m=float(args.square_mm) * 1e-3,
        marker_length_m=float(args.marker_mm) * 1e-3,
        dictionary_name=str(args.dict),
        legacy_pattern=bool(args.legacy_pattern),
    )

    source = Record3DSource()
    print(f"[aruco-view] connecting to Record3D device #{args.device_index}...")
    source.connect(device_index=int(args.device_index))
    print(
        f"[aruco-view] board: {cfg.squares_x}x{cfg.squares_y} squares, "
        f"square={cfg.square_length_m*1000:.1f}mm, marker={cfg.marker_length_m*1000:.1f}mm, "
        f"dict={cfg.dictionary_name}, legacy={cfg.legacy_pattern}"
    )

    win = "aruco_world_view"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    mouse_xy: list[tuple[int, int] | None] = [None]

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_xy[0] = (int(x), int(y))

    cv2.setMouseCallback(win, _on_mouse)

    last_t = time.monotonic()
    fps_ema = 0.0

    try:
        while True:
            packet = source.wait_for_frame(timeout_s=0.25)
            if packet is None:
                continue

            now = time.monotonic()
            dt = max(1e-3, now - last_t)
            last_t = now
            inst_fps = 1.0 / dt
            fps_ema = 0.9 * fps_ema + 0.1 * inst_fps if fps_ema > 0 else inst_fps

            det = detect_board_pose(packet.rgb, packet.intrinsic_mat, cfg)
            t_world_camera = invert_transform(det.t_camera_board) if det is not None else None

            rgb_bgr = cv2.cvtColor(packet.rgb, cv2.COLOR_RGB2BGR)
            if det is not None:
                _draw_corner_pixels(rgb_bgr, det.corners_image)
                _draw_inner_corners(rgb_bgr, cfg, det.t_camera_board, packet.intrinsic_mat, label_every=int(args.label_every))
                _draw_axes(rgb_bgr, det.t_camera_board, packet.intrinsic_mat, axis_len_m=0.05)

            cursor_world = None
            if t_world_camera is not None and mouse_xy[0] is not None:
                cursor_world = _world_xyz_under_cursor(mouse_xy[0], packet, t_world_camera)
                # Mark the cursor on the image too
                cv2.drawMarker(
                    rgb_bgr, mouse_xy[0], (200, 255, 200),
                    markerType=cv2.MARKER_CROSS, markerSize=18, thickness=1, line_type=cv2.LINE_AA,
                )

            _draw_hud(rgb_bgr, det, t_world_camera, cursor_world, fps_ema)

            cv2.imshow(win, rgb_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                out = Path("playground") / f"aruco_view_{int(time.time())}.png"
                cv2.imwrite(str(out), rgb_bgr)
                print(f"[aruco-view] snapshot saved -> {out}")
    finally:
        source.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
