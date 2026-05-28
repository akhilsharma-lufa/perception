"""Diagnose Record3D's intrinsic-matrix reference resolution.

The localizer assumes `packet.intrinsic_mat` is at RGB resolution and then scales
it down to the depth resolution. If Record3D actually returns it at depth (or
some intermediate) resolution, we're mis-scaling — which biases the 3D
unprojection globally and (because sx vs sy can differ) asymmetrically in X vs Y.

This tool reads one frame and decides which resolution the intrinsics actually
match, by comparing the principal point (cx, cy) against W/2, H/2 for each
candidate resolution (the principal point should be ~ the image center at the
native resolution). It also prints the horizontal field-of-view at each
candidate; iPhone wide lens FOV is ~65–73 degrees, so anything wildly off flags
a mismatch too. Read-only, no robot needed.

    python -m perception.demos.intrinsics_check --device-index 0
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from perception.io.record3d_source import Record3DSource


def _hfov_deg(fx: float, w_px: int) -> float:
    return 2.0 * math.degrees(math.atan2(w_px * 0.5, fx))


def _vfov_deg(fy: float, h_px: int) -> float:
    return 2.0 * math.degrees(math.atan2(h_px * 0.5, fy))


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m perception.demos.intrinsics_check")
    p.add_argument("--device-index", type=int, default=0)
    args = p.parse_args()

    src = Record3DSource()
    src.connect(device_index=int(args.device_index))
    pkt = None
    for _ in range(20):
        pkt = src.wait_for_frame(timeout_s=0.5)
        if pkt is not None:
            break
    src.disconnect()
    if pkt is None:
        print("No frame from Record3D — is the iPhone connected and running?", file=sys.stderr)
        sys.exit(1)

    k = np.asarray(pkt.intrinsic_mat, dtype=np.float64)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    rgb_h, rgb_w = int(pkt.rgb.shape[0]), int(pkt.rgb.shape[1])
    depth_h, depth_w = int(pkt.depth.shape[0]), int(pkt.depth.shape[1])

    print(f"intrinsic_mat (fx,fy,cx,cy) = ({fx:.2f}, {fy:.2f}, {cx:.2f}, {cy:.2f})")
    print(f"RGB shape  (H x W) = {rgb_h} x {rgb_w}")
    print(f"depth shape(H x W) = {depth_h} x {depth_w}")
    print()
    print(f"{'resolution':>14}  {'W/2':>6} {'H/2':>6} | {'|cx-W/2|':>9} {'|cy-H/2|':>9} | "
          f"{'HFOV°':>6} {'VFOV°':>6}")
    candidates = [("RGB", rgb_w, rgb_h), ("depth", depth_w, depth_h)]
    fits = []
    for name, w, h in candidates:
        dx = abs(cx - w * 0.5)
        dy = abs(cy - h * 0.5)
        hfov = _hfov_deg(fx, w)
        vfov = _vfov_deg(fy, h)
        # Distance of principal point from center, normalized to image diagonal.
        diag = math.hypot(w, h)
        score = math.hypot(dx, dy) / diag if diag else float("inf")
        fits.append((name, w, h, dx, dy, hfov, vfov, score))
        print(f"{name:>14}  {w/2:>6.1f} {h/2:>6.1f} | {dx:>9.2f} {dy:>9.2f} | "
              f"{hfov:>6.2f} {vfov:>6.2f}")

    best = min(fits, key=lambda t: t[7])
    print()
    print(f"=> principal point best matches {best[0]} resolution "
          f"({best[1]}x{best[2]}); offset {math.hypot(best[3], best[4]):.2f} px.")
    print(f"   iPhone wide lens H-FOV is typically ~65-73 degrees; sanity-check that "
          f"the chosen resolution's HFOV={best[5]:.1f}° is in that range.")

    aspect_rgb = rgb_w / rgb_h
    aspect_depth = depth_w / depth_h
    if abs(aspect_rgb - aspect_depth) > 0.01:
        print(f"   NOTE: RGB aspect {aspect_rgb:.3f} != depth aspect {aspect_depth:.3f} "
              f"-> sx and sy differ when scaling, which is a likely source of "
              f"asymmetric X vs Y bias.")

    # Tell the code path: if intrinsics are NOT at RGB res, scale_intrinsics_for_shape
    # currently mis-scales them. Report the implication.
    if best[0] == "RGB":
        print("\n   Current code (assumes RGB) is correct.")
    else:
        print(f"\n   *** Current code assumes RGB but intrinsics are at {best[0]} "
              f"resolution. scale_intrinsics_for_shape is mis-scaling. ***")


if __name__ == "__main__":
    main()
