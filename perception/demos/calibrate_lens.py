"""Estimate the iPhone wide-camera's lens distortion via ChArUco frames.

Record3D supplies only `(fx, fy, cx, cy)` and no distortion coefficients — the
iPhone never transmits them over the wire (see `libs/include/record3d/
Record3DStream.h`; the IntrinsicMatrixCoeffs struct has no `k1..k3` fields).
Our pipeline currently assumes zero distortion in both `solvePnP`
(`charuco_board.py`) and depth unprojection (`rgbd_localizer.py`). With a
tripod-tilted phone, unmodelled distortion at the image edges shows up as
position-dependent X/Y bias in detected world coords, with the Y bias larger
than X because the tilt sweeps the table across many image rows.

This demo collects ChArUco frames through Record3D and runs
`cv2.calibrateCamera` to estimate `K_refined` and `dist_coeffs = (k1, k2, p1,
p2, k3)`. It also bins per-corner reprojection residuals by radial distance
from the image centre, so we can see whether residuals drop after applying
`dist_coeffs` (= the lens model fits) or stay flat (= distortion is not the
culprit; look elsewhere).

Auto-capture: each frame is checked for a board detection meeting a quality
gate (min corners, max reproj px, sharp focus). If it lands in a 3x3 image
cell or tilt-bin not yet covered, it's saved. You just reposition the phone
slowly; the script picks frames as coverage grows.

    python -m perception.demos.calibrate_lens \\
        --profile calibration/profiles/session_multitag.json \\
        --target-captures 20 \\
        --output calibration/profiles/lens_calib.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from perception.calibration.charuco_board import (
    CharucoBoardConfig,
    _board_chessboard_corners,
    _build_board,
    _detect_markers,
    _interpolate_charuco_corners,
)
from perception.calibration.profiles import CalibrationProfileIO
from perception.io.record3d_source import Record3DSource


# -----------------------------------------------------------------------------
# Capture quality gates and coverage bookkeeping
# -----------------------------------------------------------------------------

@dataclass
class Capture:
    obj_pts: np.ndarray          # (N, 3) in board frame, metres
    img_pts: np.ndarray          # (N, 2) in pixels
    centroid_uv: Tuple[float, float]
    tilt_deg: float              # board-normal vs camera-Z, degrees
    reproj_px: float             # initial-pinhole reprojection, for diagnostics
    laplacian_var: float         # sharpness score


def _board_cfg_from_profile(profile) -> CharucoBoardConfig:
    s = profile.charuco_board
    if s is None:
        print("ERROR: profile has no charuco_board spec; run touch_calibrate first.",
              file=sys.stderr)
        sys.exit(2)
    return CharucoBoardConfig(
        int(s.squares_x), int(s.squares_y),
        float(s.square_length_m), float(s.marker_length_m),
        str(s.dictionary_name), bool(s.legacy_pattern),
    )


def _laplacian_var(cv2, gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _detect_corners(
    cv2, rgb: np.ndarray, dictionary, board, min_corners: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (img_pts[N,2], ids[N]) or None."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    marker_corners, marker_ids = _detect_markers(cv2, gray, dictionary)
    if marker_ids is None or len(marker_ids) == 0:
        return None
    cc, cids = _interpolate_charuco_corners(cv2, gray, marker_corners, marker_ids, board)
    if cc is None or cids is None or len(cids) < int(min_corners):
        return None
    img_pts = np.asarray(cc, dtype=np.float64).reshape(-1, 2)
    ids = np.asarray(cids, dtype=np.int64).reshape(-1)
    return img_pts, ids


def _board_pose_initial(
    cv2, obj_pts: np.ndarray, img_pts: np.ndarray, k: np.ndarray,
) -> Tuple[float, Optional[np.ndarray]]:
    """PnP with zero distortion. Returns (reproj_px, board_normal_in_camera).
    `board_normal_in_camera` is the board's +Z axis in the camera frame."""
    dist = np.zeros(5, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts.astype(np.float64), img_pts.astype(np.float64),
        k.astype(np.float64), dist, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return float("inf"), None
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, k, dist)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img_pts) ** 2, axis=1))))
    rot, _ = cv2.Rodrigues(rvec)
    normal_cam = rot[:, 2]  # board +Z in camera frame
    return err, normal_cam


def _cell_index(u: float, v: float, w: int, h: int, grid: int = 5) -> Tuple[int, int]:
    cx = min(int(grid * u / max(1, w)), grid - 1)
    cy = min(int(grid * v / max(1, h)), grid - 1)
    return cy, cx  # (row, col)


def _max_corner_radius_norm(img_pts: np.ndarray, w: int, h: int) -> float:
    """Maximum normalised radial distance of any detected corner from image
    centre. We use this to prefer-accept frames whose corners reach the
    outer rings (>0.6), which constrain k2/k3."""
    cu, cv = 0.5 * float(w), 0.5 * float(h)
    half_diag = math.hypot(cu, cv)
    r = np.sqrt((img_pts[:, 0] - cu) ** 2 + (img_pts[:, 1] - cv) ** 2)
    return float(np.max(r) / max(1e-9, half_diag))


def _tilt_bin(tilt_deg: float, edges: Tuple[float, ...]) -> int:
    """Return the index of the bin tilt_deg falls into."""
    for i, hi in enumerate(edges):
        if tilt_deg <= hi:
            return i
    return len(edges)


# -----------------------------------------------------------------------------
# Radial residual analysis
# -----------------------------------------------------------------------------

def _residuals_per_corner(
    cv2,
    captures: List[Capture],
    k: np.ndarray,
    dist: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Re-solve PnP per capture with the supplied (k, dist) and return
    (radial_distances_normalised, per-corner residuals in px)."""
    radii: List[float] = []
    res: List[float] = []
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    # normalise radius to half-diagonal of an arbitrary reference; we use the
    # captures' image span via cx,cy as a proxy (they're near the centre).
    half_diag = math.hypot(cx, cy)
    for cap in captures:
        ok, rvec, tvec = cv2.solvePnP(
            cap.obj_pts, cap.img_pts, k, dist, flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            continue
        proj, _ = cv2.projectPoints(cap.obj_pts, rvec, tvec, k, dist)
        proj = proj.reshape(-1, 2)
        d = np.sqrt(np.sum((proj - cap.img_pts) ** 2, axis=1))
        r = np.sqrt((cap.img_pts[:, 0] - cx) ** 2 + (cap.img_pts[:, 1] - cy) ** 2)
        r_norm = r / max(1e-9, half_diag)
        radii.extend(r_norm.tolist())
        res.extend(d.tolist())
    return np.asarray(radii, dtype=np.float64), np.asarray(res, dtype=np.float64)


def _print_radial_table(name: str, radii: np.ndarray, res: np.ndarray) -> None:
    if radii.size == 0:
        print(f"  [{name}] no residuals to analyse.")
        return
    bin_edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2], dtype=np.float64)
    print(f"\n  per-corner reprojection residual binned by normalised radius "
          f"({name}):")
    print(f"    {'r in':>10}  {'count':>6}  {'mean px':>8}  {'p95 px':>8}")
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        m = (radii >= lo) & (radii < hi)
        if not np.any(m):
            continue
        d = res[m]
        print(f"    [{lo:.2f},{hi:.2f})  {int(m.sum()):>6}  "
              f"{float(np.mean(d)):>8.3f}  {float(np.percentile(d, 95)):>8.3f}")


# -----------------------------------------------------------------------------
# Main capture loop
# -----------------------------------------------------------------------------

def main() -> None:
    import cv2  # deferred to keep import-on-help fast and avoid hard dep

    p = argparse.ArgumentParser(prog="python -m perception.demos.calibrate_lens")
    p.add_argument("--profile", default="calibration/profiles/session_multitag.json")
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--target-captures", type=int, default=20,
                   help="Stop once this many captures are collected.")
    p.add_argument("--min-corners", type=int, default=20)
    p.add_argument("--max-initial-reproj-px", type=float, default=6.0,
                   help="Reject frames where pinhole-only PnP already overshoots "
                        "this. We deliberately allow large values here because a "
                        "lens with real distortion produces 1-3 px PnP residuals at "
                        "the image edges — exactly the frames we need. The final "
                        "cv2.calibrateCamera LM solve is robust to noisy PnP-init.")
    p.add_argument("--min-laplacian-var", type=float, default=50.0,
                   help="Reject blurry frames (variance of Laplacian).")
    p.add_argument("--inter-capture-s", type=float, default=0.8,
                   help="Minimum seconds between consecutive auto-captures, so "
                        "you have time to reposition between grabs.")
    p.add_argument("--output", default="calibration/profiles/lens_calib.json",
                   help="Where to write the resulting K_refined + dist_coeffs.")
    args = p.parse_args()

    profile = CalibrationProfileIO.load(args.profile)
    cfg = _board_cfg_from_profile(profile)
    dictionary, board = _build_board(cv2, cfg)
    board_corners_3d = _board_chessboard_corners(board)
    print(f"board: {cfg.squares_x}x{cfg.squares_y} squares of "
          f"{cfg.square_length_m*1000:.1f} mm ({board_corners_3d.shape[0]} inner corners).")
    print(f"target captures: {int(args.target_captures)}  "
          f"min corners: {int(args.min_corners)}  "
          f"max initial reproj: {args.max_initial_reproj_px:.1f} px")

    tilt_edges = (10.0, 25.0, 40.0)  # 4 bins: <=10, (10,25], (25,40], >40 deg
    n_tilt_bins = len(tilt_edges) + 1
    grid_n = 5  # 5x5 image-cell coverage (25 cells)
    radial_edges = (0.4, 0.6, 0.8, 1.0)  # rings of normalised radius
    n_radial_bins = len(radial_edges)
    cells_seen: set[tuple[int, int]] = set()
    tilts_seen: set[int] = set()
    radial_seen: set[int] = set()
    captures: List[Capture] = []
    image_size: Optional[Tuple[int, int]] = None
    k_initial: Optional[np.ndarray] = None
    last_capture_t = 0.0

    src = Record3DSource(); src.connect(device_index=int(args.device_index))
    print("\nReady. Reposition the phone holder so the board lands in different "
          "image regions AND at different tilts. Auto-captures will appear "
          "below. Ctrl-C to stop early.\n")
    try:
        while len(captures) < int(args.target_captures):
            pkt = src.wait_for_frame(timeout_s=0.5)
            if pkt is None:
                continue
            h, w = int(pkt.rgb.shape[0]), int(pkt.rgb.shape[1])
            if image_size is None:
                image_size = (w, h)
                k_initial = np.asarray(pkt.intrinsic_mat, dtype=np.float64).copy()
                print(f"  intrinsics from Record3D (RGB {w}x{h}): "
                      f"fx={k_initial[0,0]:.2f} fy={k_initial[1,1]:.2f} "
                      f"cx={k_initial[0,2]:.2f} cy={k_initial[1,2]:.2f}")
            elif image_size != (w, h):
                # Record3D shouldn't change resolution mid-stream, but guard.
                continue

            det = _detect_corners(cv2, pkt.rgb, dictionary, board, args.min_corners)
            if det is None:
                continue
            img_pts, ids = det
            obj_pts = board_corners_3d[ids]

            gray = cv2.cvtColor(pkt.rgb, cv2.COLOR_RGB2GRAY)
            lap = _laplacian_var(cv2, gray)
            if lap < float(args.min_laplacian_var):
                continue

            reproj, normal_cam = _board_pose_initial(cv2, obj_pts, img_pts,
                                                     pkt.intrinsic_mat)
            if normal_cam is None or reproj > float(args.max_initial_reproj_px):
                continue

            # Board centroid pixel → which image cell, and tilt-bin via angle
            # of the board's +Z to the camera's +Z. Big tilt = larger angle.
            u_c = float(np.mean(img_pts[:, 0]))
            v_c = float(np.mean(img_pts[:, 1]))
            cell = _cell_index(u_c, v_c, w, h, grid=grid_n)
            tilt_deg = float(np.degrees(math.acos(min(1.0, abs(float(normal_cam[2]))))))
            tbin = _tilt_bin(tilt_deg, tilt_edges)
            # Outermost radial ring any corner of this frame reaches. Captures
            # that touch a fresh ring (especially r in [0.6,1.0)) constrain
            # k2/k3 — accept them even if the centroid cell is "used".
            r_max = _max_corner_radius_norm(img_pts, w, h)
            rbin = min(int(_tilt_bin(r_max, radial_edges)), n_radial_bins - 1)

            now = time.monotonic()
            new_cell = cell not in cells_seen
            new_tilt = tbin not in tilts_seen
            new_radial = rbin not in radial_seen
            if not (new_cell or new_tilt or new_radial):
                continue
            if now - last_capture_t < float(args.inter_capture_s):
                continue

            captures.append(Capture(
                obj_pts=obj_pts.copy(), img_pts=img_pts.copy(),
                centroid_uv=(u_c, v_c), tilt_deg=tilt_deg,
                reproj_px=reproj, laplacian_var=lap,
            ))
            cells_seen.add(cell)
            tilts_seen.add(tbin)
            radial_seen.add(rbin)
            last_capture_t = now
            print(f"  [{len(captures):2d}/{int(args.target_captures)}] "
                  f"cell={cell}  tilt={tilt_deg:5.1f}°  r_max={r_max:.2f}  "
                  f"reproj(pinhole)={reproj:.2f}px  "
                  f"corners={img_pts.shape[0]}  sharp={lap:.0f}  "
                  f"cells={len(cells_seen)}/{grid_n*grid_n}  "
                  f"tilts={len(tilts_seen)}/{n_tilt_bins}  "
                  f"radial_rings={len(radial_seen)}/{n_radial_bins}")
    except KeyboardInterrupt:
        print("\n  interrupted by user.")
    finally:
        src.disconnect()

    if len(captures) < 8 or image_size is None or k_initial is None:
        print(f"\nERROR: only {len(captures)} usable captures; need >= 8.",
              file=sys.stderr)
        sys.exit(3)
    print(f"\ncollected {len(captures)} captures; "
          f"cells={len(cells_seen)}/{grid_n*grid_n}; "
          f"tilts={len(tilts_seen)}/{n_tilt_bins}; "
          f"radial_rings={len(radial_seen)}/{n_radial_bins}.")
    if len(cells_seen) < 8:
        print("  WARNING: <8 image cells covered — calibration may be biased. "
              "Re-run and place the board in more frame regions.")
    if len(tilts_seen) < 3:
        print("  WARNING: <3 tilt bins covered — focal length and depth are "
              "weakly disambiguated. Re-run with more board tilt.")
    if len(radial_seen) < 3 or max(radial_seen) < 2:
        print("  WARNING: outer radial rings under-sampled — k2/k3 are weakly "
              "constrained. Push the board further toward the image corners.")

    obj_pts_list = [c.obj_pts.astype(np.float32) for c in captures]
    img_pts_list = [c.img_pts.astype(np.float32) for c in captures]

    # Calibrate. We rely on the iterative LM solver; rational/tangential terms
    # off keeps the model close to a textbook 5-coefficient pinhole+radial.
    flags = 0
    rms, k_refined, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts_list, img_pts_list, image_size, None, None, flags=flags,
    )
    k_refined = np.asarray(k_refined, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)

    print(f"\ncalibrateCamera RMS reprojection error: {rms:.3f} px "
          f"(across {sum(c.img_pts.shape[0] for c in captures)} corners)")
    print("\nK_initial (Record3D):")
    for row in k_initial:
        print("  " + "  ".join(f"{v:10.3f}" for v in row))
    print("K_refined (calibrateCamera):")
    for row in k_refined:
        print("  " + "  ".join(f"{v:10.3f}" for v in row))
    dfx = k_refined[0, 0] - k_initial[0, 0]
    dfy = k_refined[1, 1] - k_initial[1, 1]
    dcx = k_refined[0, 2] - k_initial[0, 2]
    dcy = k_refined[1, 2] - k_initial[1, 2]
    print(f"  delta: dfx={dfx:+.2f}  dfy={dfy:+.2f}  "
          f"dcx={dcx:+.2f}  dcy={dcy:+.2f}  (px)")

    print(f"\ndist_coeffs (k1, k2, p1, p2, k3): "
          f"[{', '.join(f'{v:+.6f}' for v in dist[:5])}]")
    k1 = float(dist[0])
    if abs(k1) < 0.02:
        print("  |k1| < 0.02 → effectively no radial distortion. ARKit is likely "
              "rectifying on-device, OR the lens-model fit is degenerate. "
              "Distortion is NOT the culprit; look at depth-to-RGB alignment.")
    else:
        print(f"  |k1| = {abs(k1):.3f} → measurable distortion in the streamed "
              "RGB. Wiring `dist_coeffs` through PnP and depth unprojection "
              "should reduce position-dependent X/Y bias.")

    # Radial residual analysis: pinhole-only vs full lens model.
    r0, d0 = _residuals_per_corner(cv2, captures, k_initial,
                                   np.zeros(5, dtype=np.float64))
    r1, d1 = _residuals_per_corner(cv2, captures, k_refined, dist)
    _print_radial_table("pinhole only (K_initial, dist=0)", r0, d0)
    _print_radial_table("with lens model (K_refined, dist)", r1, d1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "k_initial": k_initial.tolist(),
        "k_refined": k_refined.tolist(),
        "dist_coeffs": dist.tolist(),
        "image_size_wh": [int(image_size[0]), int(image_size[1])],
        "rms_reproj_px": float(rms),
        "n_captures": len(captures),
        "cells_covered": sorted(list(cells_seen)),
        "tilts_covered": sorted(list(tilts_seen)),
        "radial_rings_covered": sorted(list(radial_seen)),
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote -> {out_path}")
    print("(Profile is NOT modified by this demo. We'll wire these into "
          "PnP + depth unprojection in a separate step after reviewing the "
          "residual table above.)")


if __name__ == "__main__":
    main()
