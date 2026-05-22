"""Single-frame ChArUco detection debugger.

Grabs ONE RGB frame from the iPhone via Record3D, runs each step of the
detection pipeline, prints what was found at each stage, and saves the
frame to `charuco_debug.png` so you can eyeball what the camera saw.

Usage:
    python3 -m perception.demos.debug_charuco \\
        --squares-x 11 --squares-y 8 \\
        --square-mm 20 --marker-mm 14 \\
        --dict DICT_4X4_50
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from perception.calibration.charuco_board import (
    CharucoBoardConfig,
    _build_board,
    _detect_markers,
    _interpolate_charuco_corners,
    _board_chessboard_corners,
)
from perception.io import Record3DSource


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--squares-x", type=int, default=11)
    p.add_argument("--squares-y", type=int, default=8)
    p.add_argument("--square-mm", type=float, default=20.0)
    p.add_argument("--marker-mm", type=float, default=14.0)
    p.add_argument("--dict", default="DICT_4X4_50")
    p.add_argument("--save", default="charuco_debug.png")
    p.add_argument(
        "--frames",
        type=int,
        default=5,
        help="Try this many frames; print results for each.",
    )
    args = p.parse_args()

    import cv2

    cfg = CharucoBoardConfig(
        squares_x=int(args.squares_x),
        squares_y=int(args.squares_y),
        square_length_m=float(args.square_mm) * 1e-3,
        marker_length_m=float(args.marker_mm) * 1e-3,
        dictionary_name=str(args.dict),
    )
    dictionary, board = _build_board(cv2, cfg)
    print(f"[debug] cv2={cv2.__version__}")
    print(
        f"[debug] board cfg: {cfg.squares_x}x{cfg.squares_y} "
        f"square={cfg.square_length_m*1000:.1f}mm marker={cfg.marker_length_m*1000:.1f}mm "
        f"dict={cfg.dictionary_name}"
    )
    n_inner = (cfg.squares_x - 1) * (cfg.squares_y - 1)
    print(f"[debug] inner chessboard corners expected: {n_inner}")

    src = Record3DSource()
    src.connect(device_index=int(args.device_index))
    # Drain a few first frames so the iPhone settles.
    print("[debug] connecting + draining initial frames...")
    for _ in range(5):
        src.wait_for_frame(timeout_s=0.5)

    for attempt in range(int(args.frames)):
        time.sleep(0.4)
        pkt = src.wait_for_frame(timeout_s=1.0)
        if pkt is None:
            print(f"[debug] attempt {attempt+1}: no frame received")
            continue

        rgb = pkt.rgb
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        print(
            f"\n[debug] attempt {attempt+1}: "
            f"rgb={rgb.shape}  depth={pkt.depth.shape}  "
            f"gray min/mean/max={gray.min()}/{gray.mean():.1f}/{gray.max()}"
        )

        marker_corners, marker_ids = _detect_markers(cv2, gray, dictionary)
        n_markers = 0 if marker_ids is None else len(marker_ids)
        print(f"[debug]   detectMarkers: {n_markers} markers found")
        if marker_ids is not None and n_markers > 0:
            ids_flat = np.asarray(marker_ids).reshape(-1).tolist()
            print(f"[debug]   marker IDs: {sorted(ids_flat)[:20]}{'...' if n_markers > 20 else ''}")

        if n_markers == 0:
            print(
                "[debug]   -> no ArUco markers in image. "
                "Likely causes: wrong --dict, board not in frame, severe glare."
            )
            continue

        charuco_corners, charuco_ids = _interpolate_charuco_corners(
            cv2, gray, marker_corners, marker_ids, board
        )
        n_corners = 0 if charuco_corners is None else len(charuco_corners)
        print(f"[debug]   interpolateCornersCharuco: {n_corners} corners")
        if n_corners < 4:
            print(
                "[debug]   -> too few chessboard corners. "
                "Likely causes: wrong --squares-x/y, board partially occluded, "
                "or board geometry doesn't match physical print."
            )
            continue

        # Pose solve
        try:
            board_3d = _board_chessboard_corners(board)
            ids = np.asarray(charuco_ids).reshape(-1)
            img_pts = np.asarray(charuco_corners).reshape(-1, 2).astype(np.float64)
            obj_pts = board_3d[ids]
            k = pkt.intrinsic_mat.astype(np.float64)
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, k, np.zeros(5),
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok:
                proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, k, np.zeros(5))
                err = float(np.sqrt(np.mean(np.sum(
                    (proj.reshape(-1, 2) - img_pts) ** 2, axis=1
                ))))
                t = tvec.reshape(3)
                print(
                    f"[debug]   solvePnP OK: t=({t[0]*1000:+6.1f}, "
                    f"{t[1]*1000:+6.1f}, {t[2]*1000:+6.1f}) mm  reproj={err:.2f} px"
                )
            else:
                print("[debug]   solvePnP FAILED")
        except Exception as exc:
            print(f"[debug]   solvePnP exception: {exc}")

        # Save annotated frame on the LAST attempt
        if attempt == int(args.frames) - 1:
            annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
            if marker_corners is not None:
                cv2.aruco.drawDetectedMarkers(annotated, marker_corners, marker_ids)
            if charuco_corners is not None:
                # Draw small green dots on every charuco corner
                for c in charuco_corners.reshape(-1, 2):
                    cv2.circle(
                        annotated,
                        (int(round(c[0])), int(round(c[1]))),
                        3, (0, 255, 0), -1, cv2.LINE_AA,
                    )
            cv2.imwrite(args.save, annotated)
            print(f"\n[debug] saved annotated frame -> {args.save}")
            print("[debug] open it: scp it to your mac, or `eog charuco_debug.png` if you have a display")

    src.disconnect()


if __name__ == "__main__":
    main()
