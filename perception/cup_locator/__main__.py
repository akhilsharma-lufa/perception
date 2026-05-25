"""Smoke-test CLI for the cup locator.

Usage:
    python -m perception.cup_locator \
        --profile calibration/profiles/session_multitag.json

Exits 0 with the cup's robot-frame coordinates printed on success, 2 if no
stable detection was found within the timeout.
"""
from __future__ import annotations

import argparse
import sys

from .api import CupLocator


def main() -> None:
    p = argparse.ArgumentParser(
        prog="python -m perception.cup_locator",
        description=(
            "Detect one stable cup pose and print its coordinates in the "
            "MyCobot base frame (mm)."
        ),
    )
    p.add_argument(
        "--profile",
        default="calibration/profiles/session_multitag.json",
        help="Path to the calibration profile written by `cup_locator.calibrate`.",
    )
    p.add_argument("--target-label", default="cup")
    p.add_argument("--min-confidence", type=float, default=0.15)
    p.add_argument("--timeout-s", type=float, default=3.0)
    p.add_argument("--settle-frames", type=int, default=8)
    p.add_argument("--device-index", type=int, default=0)
    args = p.parse_args()

    with CupLocator(
        args.profile,
        target_label=args.target_label,
        min_confidence=args.min_confidence,
        settle_frames=args.settle_frames,
    ) as loc:
        pose = loc.locate(
            timeout_s=args.timeout_s,
            device_index=args.device_index,
        )
        if pose is None:
            print(
                f"[cup_locator] no stable {args.target_label!r} within "
                f"{args.timeout_s:.1f}s.",
                file=sys.stderr,
            )
            sys.exit(2)
        x_mm, y_mm, z_mm = pose.position_robot_mm
        wx, wy, wz = pose.position_world_m
        height_str = "None" if pose.height_m is None else f"{pose.height_m:.3f}"
        yaw_str = "None" if pose.yaw_hint_rad is None else f"{pose.yaw_hint_rad:+.3f}"
        print(
            f"[cup_locator] label={pose.label!r} track={pose.track_id} "
            f"quality={pose.quality:.2f}\n"
            f"  robot_mm = ({x_mm:+8.2f}, {y_mm:+8.2f}, {z_mm:+8.2f})\n"
            f"  world_m  = ({wx:+.4f}, {wy:+.4f}, {wz:+.4f})\n"
            f"  height_m = {height_str}\n"
            f"  yaw_hint = {yaw_str}"
        )


if __name__ == "__main__":
    main()
